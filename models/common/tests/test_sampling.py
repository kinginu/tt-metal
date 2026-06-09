# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import ttnn
import models.common.sampling.generator as sampling_generator_module
from models.common.sampling import LogProbsCalculator
from models.common.sampling.generator import (
    MAX_UINT32,
    SamplingGenerator,
    SamplingParams,
    SeedManager,
    format_sampling_params,
)
from models.common.sampling.tt_log_probs import MAX_TOP_LOGPROBS, LogProbsResult
from models.common.utility_functions import comp_pcc


# ---------------------------------------------------------------------------
# SeedManager host-side tests
# ---------------------------------------------------------------------------
class _ZeroRng:
    def randint(self, _low, _high):
        return 0


def _make_seed_manager_for_host_tests(max_batch_size=4):
    fake_tt_sampling = SimpleNamespace(
        _sampling_dp=1,
        mesh_device=object(),
        seeds_tt_tensor=object(),
    )
    return SeedManager(fake_tt_sampling, max_batch_size=max_batch_size)


def _capture_seed_uploads(monkeypatch):
    uploads = []

    def fake_from_torch(tensor, **_kwargs):
        uploads.append(tensor.clone())
        return tensor

    monkeypatch.setattr(sampling_generator_module.ttnn, "from_torch", fake_from_torch)
    monkeypatch.setattr(sampling_generator_module.ttnn, "copy_host_to_device_tensor", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sampling_generator_module.ttnn, "synchronize_device", lambda *_args, **_kwargs: None)
    return uploads


def test_seed_manager_never_uploads_zero_for_explicit_seed_zero(monkeypatch):
    uploads = _capture_seed_uploads(monkeypatch)
    manager = _make_seed_manager_for_host_tests()

    manager.reset_seed([0], [0])
    manager.rngs[0] = _ZeroRng()
    manager.get_new_values(empty_slots=[0])

    assert manager.seeds[0] == 0
    assert uploads[-1].tolist() == [1, MAX_UINT32, MAX_UINT32, MAX_UINT32]


def test_seed_manager_never_uploads_zero_for_unseeded_random_init(monkeypatch):
    uploads = _capture_seed_uploads(monkeypatch)
    monkeypatch.setattr(sampling_generator_module.random, "randint", lambda _low, _high: 0)
    manager = _make_seed_manager_for_host_tests()

    manager.reset_seed([None], [0])
    manager.get_new_values()

    assert uploads[-1].tolist() == [1, 1, 1, 1]

    manager.get_new_values()

    assert uploads[-1].tolist() == [MAX_UINT32] * 4


def test_seeded_decode_seed_is_stable_after_slot_reuse(monkeypatch):
    uploads = _capture_seed_uploads(monkeypatch)
    manager = _make_seed_manager_for_host_tests()

    manager.update_decode_seeds([0, None, None, None])
    manager.get_new_values(positions=torch.tensor([12, -1, -1, -1]))
    first_upload = uploads[-1].tolist()
    first_seed = first_upload[0]

    manager.reset_seed([1234], [0])
    manager.get_new_values(empty_slots=[0])

    manager.update_decode_seeds([0, None, None, None])
    manager.get_new_values(positions=torch.tensor([12, -1, -1, -1]))
    same_position_upload = uploads[-1].tolist()

    manager.get_new_values(positions=torch.tensor([13, -1, -1, -1]))
    next_position_upload = uploads[-1].tolist()

    assert first_seed not in (0, MAX_UINT32)
    assert same_position_upload == first_upload
    assert next_position_upload[0] != first_seed
    assert next_position_upload[1:] == [MAX_UINT32, MAX_UINT32, MAX_UINT32]


def test_decode_seed_update_reinitializes_when_batch_becomes_unseeded(monkeypatch):
    uploads = _capture_seed_uploads(monkeypatch)
    monkeypatch.setattr(sampling_generator_module.random, "randint", lambda _low, _high: 0)
    manager = _make_seed_manager_for_host_tests()

    manager.update_decode_seeds([0, None, None, None])
    manager.get_new_values(positions=torch.tensor([12, -1, -1, -1]))

    manager.update_decode_seeds([None, None, None, None])
    manager.get_new_values()

    assert uploads[-1].tolist() == [1, 1, 1, 1]


def test_format_sampling_params_broadcasts_scalar_seed_zero():
    params = SamplingParams(temperature=1.0, top_k=32, top_p=0.95, seed=0)

    formatted = format_sampling_params(params, max_batch_size=32)

    assert formatted.seed == [0] * 32


def test_format_sampling_params_pads_multi_seed_list_with_none():
    params = SamplingParams(temperature=[1.0, 1.0], top_k=[32, 32], top_p=[0.95, 0.95], seed=[0, 7])

    formatted = format_sampling_params(params, max_batch_size=32)

    assert formatted.seed[:4] == [0, 7, None, None]


def test_apply_prefill_state_can_use_distinct_seed_slots():
    calls = []
    generator = object.__new__(SamplingGenerator)
    generator.reset_sampling_params = lambda sampling_params, empty_slots=None: calls.append(("params", empty_slots))
    generator.reset_prompt_tokens = lambda prompt_tokens: calls.append(("prompt", prompt_tokens))
    generator.reset_output_state = lambda tokens=None: calls.append(("output", tokens))
    generator.seed_manager = SimpleNamespace(
        reset_seed=lambda seed, user_ids: calls.append(("reset_seed", seed, user_ids)),
        get_new_values=lambda empty_slots, replicate_seeds=False, positions=None: calls.append(
            ("get_new_values", empty_slots, replicate_seeds, positions)
        ),
    )
    params = SimpleNamespace(seed=[0] * 128)

    SamplingGenerator.apply_prefill_state(
        generator,
        sampling_params=params,
        prompt_tokens=None,
        empty_slots=[3],
        seed_slots=[35],
        replicate_seeds=True,
    )

    assert calls == [
        ("params", [3]),
        ("reset_seed", params.seed, [35]),
        ("get_new_values", [35], True, None),
        ("output", None),
    ]


def test_seeded_replicated_prefill_uses_positioned_rows(monkeypatch):
    uploads = _capture_seed_uploads(monkeypatch)
    manager = _make_seed_manager_for_host_tests()

    manager.reset_seed([0], [0])
    manager.get_new_values(empty_slots=[0], replicate_seeds=True, positions=[40, 41, 42, 43])
    values = uploads[-1].tolist()

    assert len(set(values)) == 4
    assert values[3] == SeedManager._stateless_device_seed_value(0, 43)


# ---------------------------------------------------------------------------
# Helper: simulate per-device top-k gather (mirrors TTSampling behaviour)
# ---------------------------------------------------------------------------
def _simulate_gathered_topk(torch_logits, num_devices, top_k=32):
    """Simulate the per-device top-k + all-gather that TTSampling performs.

    Args:
        torch_logits: Full logits tensor, shape (1, 1, B, V).
        num_devices: Number of TP devices.
        top_k: Per-device top-k count.

    Returns:
        gathered_values: (1, 1, B, num_devices * top_k) raw logit values.
        gathered_indices: (1, 1, B, num_devices * top_k) global vocab indices.
    """
    V = torch_logits.shape[-1]
    shard_size = V // num_devices
    all_values = []
    all_indices = []
    for d in range(num_devices):
        shard = torch_logits[:, :, :, d * shard_size : (d + 1) * shard_size]
        vals, local_idx = torch.topk(shard, top_k, dim=-1)
        global_idx = local_idx + d * shard_size
        all_values.append(vals)
        all_indices.append(global_idx)
    gathered_values = torch.cat(all_values, dim=-1)
    gathered_indices = torch.cat(all_indices, dim=-1)
    return gathered_values, gathered_indices


# ===========================================================================
# Top-K logprobs tests (TG Galaxy only)
# ===========================================================================

# Common TG Galaxy device parametrization for all new tests
TG_SHAPE = [1, 1, 32, 8 * 16032]  # Llama on TG with 8-chip TP sharded vocab
TG_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
}
TG_MESH_SHAPE = (8, 4)
TG_SUB_CORE_GRIDS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
    ]
)
TG_NUM_TP_DEVICES = 8  # TP dimension for Galaxy


def _skip_if_not_galaxy(mesh_device):
    """Skip test if not running on TG Galaxy (32 devices)."""
    if mesh_device.get_num_devices() != 32:
        pytest.skip(f"Test requires TG Galaxy (32 devices), got {mesh_device.get_num_devices()}")


def _push_topk_test_tensors_to_tg(torch_tensor, gathered_values, gathered_indices, mesh_device):
    """Push logits, topk values, and topk indices to a TG Galaxy mesh device."""
    logits_tt = ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, None), mesh_shape=list(mesh_device.shape)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    topk_values_tt = ttnn.from_torch(
        gathered_values,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    topk_indices_tt = ttnn.from_torch(
        gathered_indices.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return logits_tt, topk_values_tt, topk_indices_tt


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 8 * 18992],  # Qwen3 on T3K
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_log_probs_calculation(shape, mesh_device):
    seed = 1234
    torch.manual_seed(seed)

    log_probs_calculator = LogProbsCalculator(mesh_device)

    torch_tensor = torch.randn(shape)
    # shuffle the tensor in last 2 dimensions
    for i in range(shape[-2]):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)
    indices_tensor = argmax_tensor.reshape(
        argmax_tensor.shape[0], argmax_tensor.shape[1], argmax_tensor.shape[-1], argmax_tensor.shape[-2]
    )
    # Push inputs to device
    logits_tensor = ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_indices_tensor = ttnn.from_torch(
        indices_tensor,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    log_probs_calculator.set_log_probs_mode(True)
    tt_log_probs = log_probs_calculator.calculate_log_probs(logits_tensor, ttnn_indices_tensor)
    log_probs_tt_host = ttnn.to_torch(tt_log_probs, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    log_probs_tt_host = log_probs_tt_host[:, :, :1, :32]

    # Calculate log-probs for each user on each chip using torch
    log_probs_torch = F.log_softmax(torch_tensor.float(), dim=-1)
    log_probs_torch_argmax = torch.gather(log_probs_torch, dim=-1, index=argmax_tensor)
    log_probs_torch_argmax = torch.reshape(log_probs_torch_argmax, (1, 1, 1, 32))

    passing, pcc = comp_pcc(log_probs_torch_argmax, log_probs_tt_host, pcc=0.99)
    print(f"pcc={pcc}")

    assert passing, f"Assertion failed, PCC={pcc}"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 8 * 18992],  # Qwen3 on T3K
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_log_probs_returns_none_when_disabled(shape, mesh_device):
    """Test that calculate_log_probs returns None when enable_log_probs is False."""
    log_probs_calculator = LogProbsCalculator(mesh_device)

    torch_tensor = torch.randn(shape)
    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)
    indices_tensor = argmax_tensor.reshape(
        argmax_tensor.shape[0], argmax_tensor.shape[1], argmax_tensor.shape[-1], argmax_tensor.shape[-2]
    )

    logits_tensor = ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_indices_tensor = ttnn.from_torch(
        indices_tensor,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Log probs disabled (default) - should return None
    log_probs_calculator.set_log_probs_mode(False)
    result = log_probs_calculator.calculate_log_probs(logits_tensor, ttnn_indices_tensor)
    assert result is None, f"Expected None when log_probs disabled, got {type(result)}"

    # Log probs enabled - should return a tensor (not None)
    log_probs_calculator.set_log_probs_mode(True)
    num_devices = mesh_device.get_num_devices()
    result = log_probs_calculator.calculate_log_probs(logits_tensor, ttnn_indices_tensor)
    if num_devices in (8, 32) and log_probs_calculator.num_devices_for_sharding >= 2:
        assert result is not None, "Expected tensor when log_probs enabled on supported device"
    else:
        assert result is None, "Expected None on unsupported device count"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 8 * 16032],  # llama on TG with 8 chips sharded vocab
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            }
        ),
    ],
    indirect=True,
    ids=["fabric_linear"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_log_probs_with_sub_core_grids_on_galaxy(shape, mesh_device):
    seed = 1234
    torch.manual_seed(seed)

    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    log_probs_calculator = LogProbsCalculator(mesh_device, sub_core_grids)

    torch_tensor = torch.randn(shape)
    # shuffle the tensor in last 2 dimensions
    for i in range(shape[-2]):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)
    indices_tensor = argmax_tensor.reshape(
        argmax_tensor.shape[0], argmax_tensor.shape[1], argmax_tensor.shape[-1], argmax_tensor.shape[-2]
    )

    if mesh_device.get_num_devices() == 8:
        mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
    elif mesh_device.get_num_devices() == 32:
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, None), mesh_shape=list(mesh_device.shape))
    else:
        raise ValueError(f"Unsupported number of devices: {mesh_device.get_num_devices()}")

    logits_tensor = ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mesh_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_indices_tensor = ttnn.from_torch(
        indices_tensor,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    log_probs_calculator.set_log_probs_mode(True)
    tt_log_probs = log_probs_calculator.calculate_log_probs(logits_tensor, ttnn_indices_tensor)
    log_probs_tt_host = ttnn.to_torch(tt_log_probs, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    # slice from (1,1,32,256) -> (1,1,1,32)
    log_probs_tt_host = log_probs_tt_host[:, :, :1, :32]

    log_probs_torch = F.log_softmax(torch_tensor.float(), dim=-1)
    log_probs_torch_argmax = torch.gather(log_probs_torch, dim=-1, index=argmax_tensor)
    log_probs_torch_argmax = torch.reshape(log_probs_torch_argmax, (1, 1, 1, 32))

    passing, pcc = comp_pcc(log_probs_torch_argmax, log_probs_tt_host, pcc=0.99)
    print(f"pcc={pcc}")

    assert passing, f"Assertion failed, PCC={pcc}"


# ===========================================================================
# New top-K logprobs tests (TG Galaxy only)
# ===========================================================================


@pytest.mark.parametrize("shape", [TG_SHAPE])
@pytest.mark.parametrize("device_params", [TG_DEVICE_PARAMS], indirect=True, ids=["tg"])
@pytest.mark.parametrize("mesh_device", [TG_MESH_SHAPE], indirect=True)
def test_top_k_log_probs_on_galaxy(shape, mesh_device):
    """Top-K logprobs PCC check on TG Galaxy (32-device 2D mesh)."""
    _skip_if_not_galaxy(mesh_device)
    torch.manual_seed(1234)
    batch_size = shape[2]

    calc = LogProbsCalculator(mesh_device, TG_SUB_CORE_GRIDS, batch_size=batch_size, use_topk_logprobs=True)

    torch_tensor = torch.randn(shape)
    for i in range(batch_size):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

    log_probs_torch = F.log_softmax(torch_tensor.to(torch.float16), dim=-1)
    gathered_values, gathered_indices = _simulate_gathered_topk(torch_tensor, TG_NUM_TP_DEVICES)
    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)

    logits_tt, topk_values_tt, topk_indices_tt = _push_topk_test_tensors_to_tg(
        torch_tensor, gathered_values, gathered_indices, mesh_device
    )

    calc.set_log_probs_mode([True] * batch_size, num_logprobs=[5] * batch_size)
    result = calc.calculate_topk_log_probs(logits_tt, topk_values_tt, topk_indices_tt)

    assert result is not None, "Expected LogProbsResult, got None"
    assert isinstance(result, LogProbsResult)

    host_results = calc.transfer_logprobs_to_host(result, argmax_tensor.squeeze())

    composer = calc._build_mesh_composer()
    topk_logprobs_host = ttnn.to_torch(result.topk_logprobs, mesh_composer=composer)
    topk_logprobs_host = topk_logprobs_host[0, 0, ...]
    topk_indices_host = ttnn.to_torch(result.topk_indices, mesh_composer=composer)
    topk_indices_host = topk_indices_host[0, 0, ...].long()

    expected_logprobs = torch.gather(
        log_probs_torch.squeeze(0).squeeze(0),
        dim=-1,
        index=topk_indices_host,
    )

    passing, pcc = comp_pcc(expected_logprobs, topk_logprobs_host, pcc=0.99)
    print(f"Galaxy top-K logprobs PCC={pcc}")
    assert passing, f"Galaxy top-K logprobs PCC failed: {pcc}"

    for user_idx in range(batch_size):
        r = host_results[user_idx]
        assert r is not None
        sampled_id = argmax_tensor[0, 0, user_idx, 0].item()
        torch_lp = log_probs_torch[0, 0, user_idx, sampled_id].item()
        assert abs(r["returned_token"]["logprob"] - torch_lp) < 0.05


@pytest.mark.parametrize("shape", [TG_SHAPE])
@pytest.mark.parametrize("device_params", [TG_DEVICE_PARAMS], indirect=True, ids=["tg"])
@pytest.mark.parametrize("mesh_device", [TG_MESH_SHAPE], indirect=True)
def test_top_k_log_probs_returns_none_when_not_needed(shape, mesh_device):
    """calculate_topk_log_probs returns None when disabled."""
    _skip_if_not_galaxy(mesh_device)
    batch_size = shape[2]
    calc = LogProbsCalculator(mesh_device, TG_SUB_CORE_GRIDS, batch_size=batch_size, use_topk_logprobs=True)

    torch_tensor = torch.randn(shape)
    gathered_values, gathered_indices = _simulate_gathered_topk(torch_tensor, TG_NUM_TP_DEVICES)
    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)

    logits_tt, topk_values_tt, topk_indices_tt = _push_topk_test_tensors_to_tg(
        torch_tensor, gathered_values, gathered_indices, mesh_device
    )

    calc.set_log_probs_mode(False, num_logprobs=0)
    result = calc.calculate_topk_log_probs(logits_tt, topk_values_tt, topk_indices_tt)
    assert result is None, "Expected None when logprobs disabled"

    calc.set_log_probs_mode(True, num_logprobs=0)
    assert calc.topk_logprobs_needed  # needed for sampled token logprob
    result = calc.calculate_topk_log_probs(logits_tt, topk_values_tt, topk_indices_tt)
    assert result is not None, "Expected LogProbsResult when logprobs enabled"

    sampled_ids = argmax_tensor.squeeze()
    host_results = calc.transfer_logprobs_to_host(result, sampled_ids)
    assert len(host_results) == batch_size
    for i in range(batch_size):
        r = host_results[i]
        assert r is not None
        assert r["returned_token"]["token_idx"] == int(sampled_ids[i].item())
        assert len(r["top_logprobs"]["token_indices"]) == 0


@pytest.mark.parametrize("shape", [TG_SHAPE])
@pytest.mark.parametrize("device_params", [TG_DEVICE_PARAMS], indirect=True, ids=["tg"])
@pytest.mark.parametrize("mesh_device", [TG_MESH_SHAPE], indirect=True)
def test_per_user_logprobs_enabled(shape, mesh_device):
    """Mixed per-user logprobs: only even users enabled."""
    _skip_if_not_galaxy(mesh_device)
    torch.manual_seed(42)
    batch_size = shape[2]

    calc = LogProbsCalculator(mesh_device, TG_SUB_CORE_GRIDS, batch_size=batch_size, use_topk_logprobs=True)

    torch_tensor = torch.randn(shape)
    for i in range(batch_size):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

    log_probs_torch = F.log_softmax(torch_tensor.to(torch.float16), dim=-1)
    gathered_values, gathered_indices = _simulate_gathered_topk(torch_tensor, TG_NUM_TP_DEVICES)
    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)

    logits_tt, topk_values_tt, topk_indices_tt = _push_topk_test_tensors_to_tg(
        torch_tensor, gathered_values, gathered_indices, mesh_device
    )

    enable_log_probs = [i % 2 == 0 for i in range(batch_size)]
    num_logprobs_list = [5 if i % 2 == 0 else 0 for i in range(batch_size)]
    calc.set_log_probs_mode(enable_log_probs, num_logprobs=num_logprobs_list)

    result = calc.calculate_topk_log_probs(logits_tt, topk_values_tt, topk_indices_tt)
    assert result is not None

    sampled_ids = argmax_tensor.squeeze()
    host_results = calc.transfer_logprobs_to_host(result, sampled_ids)

    for i in range(batch_size):
        if enable_log_probs[i]:
            assert host_results[i] is not None
            sampled_id = int(sampled_ids[i].item())
            torch_lp = log_probs_torch[0, 0, i, sampled_id].item()
            assert abs(host_results[i]["returned_token"]["logprob"] - torch_lp) < 0.05
        else:
            assert host_results[i] is None


@pytest.mark.parametrize("shape", [TG_SHAPE])
@pytest.mark.parametrize("device_params", [TG_DEVICE_PARAMS], indirect=True, ids=["tg"])
@pytest.mark.parametrize("mesh_device", [TG_MESH_SHAPE], indirect=True)
def test_set_log_probs_mode_validation(shape, mesh_device):
    """Verify set_log_probs_mode internal state."""
    _skip_if_not_galaxy(mesh_device)
    batch_size = shape[2]
    calc = LogProbsCalculator(mesh_device, TG_SUB_CORE_GRIDS, batch_size=batch_size, use_topk_logprobs=True)

    calc.set_log_probs_mode(True)
    assert calc.enable_log_probs is True
    assert all(calc.logprobs_enabled)
    assert calc.topk_logprobs_needed  # needed for sampled token logprob

    calc.set_log_probs_mode(True, num_logprobs=5)
    assert calc.topk_logprobs_needed is True
    assert all(n == 5 for n in calc.num_logprobs)

    enable_list = [True, False, True] + [False] * (batch_size - 3)
    num_lp_list = [10, 0, 3] + [0] * (batch_size - 3)
    calc.set_log_probs_mode(enable_list, num_logprobs=num_lp_list)
    assert calc.enable_log_probs is True
    assert calc.topk_logprobs_needed is True
    assert calc.logprobs_enabled == enable_list
    assert calc.num_logprobs == num_lp_list

    calc.set_log_probs_mode(False, num_logprobs=0)
    assert calc.enable_log_probs is False

    calc.set_log_probs_mode(True, num_logprobs=0)
    assert calc.enable_log_probs is True
    assert calc.topk_logprobs_needed  # needed for sampled token logprob

    calc.set_log_probs_mode(False, num_logprobs=0)
    calc.set_log_probs_mode([True, True], num_logprobs=[10, 15], empty_slots=[2, 5])
    assert calc.logprobs_enabled[2] is True
    assert calc.logprobs_enabled[5] is True
    assert calc.logprobs_enabled[0] is False
    assert calc.num_logprobs[2] == 10
    assert calc.num_logprobs[5] == 15

    calc.set_log_probs_mode(False, num_logprobs=0)
    calc.set_log_probs_mode(True, num_logprobs=7, empty_slots=[0, 3, 4])
    assert all(calc.logprobs_enabled[i] for i in [0, 3, 4])
    assert calc.logprobs_enabled[1] is False

    calc.set_log_probs_mode([True], num_logprobs=[20], empty_slots=[1])
    assert calc.logprobs_enabled[1] is True
    assert calc.num_logprobs[0] == 7
    assert calc.num_logprobs[1] == 20


@pytest.mark.parametrize("shape", [TG_SHAPE])
@pytest.mark.parametrize("device_params", [TG_DEVICE_PARAMS], indirect=True, ids=["tg"])
@pytest.mark.parametrize("mesh_device", [TG_MESH_SHAPE], indirect=True)
def test_top_k_logprobs_pcc_torch_vs_tt(shape, mesh_device):
    """Compare host (PyTorch bfloat16) vs device (bfloat16) logprobs for full batch."""
    _skip_if_not_galaxy(mesh_device)
    torch.manual_seed(9999)
    batch_size = shape[2]
    requested_logprobs = MAX_TOP_LOGPROBS

    calc = LogProbsCalculator(mesh_device, TG_SUB_CORE_GRIDS, batch_size=batch_size, use_topk_logprobs=True)

    torch_tensor = torch.randn(shape).to(torch.bfloat16)
    for i in range(batch_size):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

    log_probs_torch = F.log_softmax(torch_tensor, dim=-1, dtype=torch.bfloat16)
    gathered_values, gathered_indices = _simulate_gathered_topk(torch_tensor, TG_NUM_TP_DEVICES)
    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)

    logits_tt, topk_values_tt, topk_indices_tt = _push_topk_test_tensors_to_tg(
        torch_tensor, gathered_values, gathered_indices, mesh_device
    )

    calc.set_log_probs_mode([True] * batch_size, num_logprobs=[requested_logprobs] * batch_size)

    result = calc.calculate_topk_log_probs(logits_tt, topk_values_tt, topk_indices_tt)
    assert result is not None

    sampled_ids = argmax_tensor.squeeze()
    host_results = calc.transfer_logprobs_to_host(result, sampled_ids)

    for user in range(batch_size):
        r = host_results[user]
        assert r is not None

        device_sampled_lp = r["returned_token"]["logprob"]
        token_idx = r["returned_token"]["token_idx"]
        torch_sampled_lp = log_probs_torch[0, 0, user, token_idx].item()
        assert abs(device_sampled_lp - torch_sampled_lp) < 0.05

        top_indices = r["top_logprobs"]["token_indices"]
        top_lps_device = torch.tensor(r["top_logprobs"]["logprobs"], dtype=torch.float32)
        assert len(top_indices) == requested_logprobs

        top_lps_torch = log_probs_torch[0, 0, user, top_indices].float()
        passing, pcc = comp_pcc(top_lps_torch.unsqueeze(0), top_lps_device.unsqueeze(0), pcc=0.98)
        assert passing, (
            f"User {user} top-{requested_logprobs} logprobs PCC failed: {pcc}\n"
            f"  device: {top_lps_device[:5].tolist()}...\n"
            f"  torch:  {top_lps_torch[:5].tolist()}..."
        )
