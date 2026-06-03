# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Small unit test for ttnn.experimental.deepseek_prefill.moe_grouped_topk().

Verifies that the new op produces results matching a PyTorch golden reference
using the recall metric (fraction of correctly selected experts per token), and
that padding awareness (via padding_config) sentinels padded rows while leaving
real rows bit-exact to the unpadded baseline.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.configuration_deepseek import DeepseekV3Config
from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate
from models.demos.deepseek_v3_d_p.tt.mla.utils import create_balanced_chunk_order, reorder_tensor_chunks
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import calculate_average_recall
from models.demos.deepseek_v3_d_p.utils.moe_test_utils import (
    GATE_KWARGS,
    N_ACT,
    TOTAL_EXPERTS,
    assert_sentinel_padding,
    balanced_local_real_tokens,
    create_padding_config,
    create_sharded_padding_config,
    generate_distinct_sigmoid_inputs,
    make_gate_inputs,
    real_token_mask,
)
from tests.ttnn.utils_for_testing import comp_pcc


def _run_indices(scores_in, bias_in, seq_len, *, padding_config=None, composer=None):
    """Run moe_grouped_topk and return trimmed int32 indices, shape [1, 1, seq_len, N_ACT]."""
    _, indices_out = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
        scores_in, bias_in, padding_config=padding_config, **GATE_KWARGS
    )
    return ttnn.to_torch(indices_out, mesh_composer=composer)[:1, :1, :seq_len, :N_ACT].to(torch.int32)


TEST_PARAMS = [(1, 1, 1), (1, 1, 33), (1, 1, 128), (1, 1, 3200)]

TEST_PARAM_IDS = ["minimal", "just_over_one_tile", "four_tiles", "realistic"]


@pytest.mark.parametrize("num_batches,batch_size,seq_len", TEST_PARAMS, ids=TEST_PARAM_IDS)
def test_moe_grouped_topk(device, num_batches, batch_size, seq_len):
    """Verify moe_grouped_topk matches the PyTorch golden reference using recall and PCC."""
    torch.manual_seed(42)

    config = DeepseekV3Config(
        hidden_size=64,
        n_routed_experts=TOTAL_EXPERTS,
        n_group=GATE_KWARGS["n_groups"],
        topk_group=GATE_KWARGS["topk_groups"],
        num_experts_per_tok=GATE_KWARGS["n_activated_experts"],
        routed_scaling_factor=GATE_KWARGS["route_scale"],
    )
    gate = MoEGate(config, use_bitonic_sort=False)

    scores = generate_distinct_sigmoid_inputs((num_batches, batch_size, seq_len, TOTAL_EXPERTS), dtype=torch.float32)
    bias = torch.randn(num_batches, batch_size, seq_len, TOTAL_EXPERTS, dtype=torch.float32)

    ref_indices, ref_weights = gate.grouped_gate_golden(
        scores,
        bias,
        GATE_KWARGS["route_scale"],
        GATE_KWARGS["epsilon"],
        GATE_KWARGS["n_groups"],
        GATE_KWARGS["summed_experts_per_group"],
        GATE_KWARGS["topk_groups"],
        GATE_KWARGS["n_activated_experts"],
    )

    ttnn_scores_in = ttnn.from_torch(scores, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_bias_in = ttnn.from_torch(bias, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_weights_out, ttnn_indices_out = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
        ttnn_scores_in, ttnn_bias_in, **GATE_KWARGS
    )

    tt_weights_torch = ttnn.to_torch(ttnn_weights_out)
    tt_indices_torch = ttnn.to_torch(ttnn_indices_out)

    # Trim padding (TILE layout pads to tile boundaries)
    tt_weights_torch = tt_weights_torch[:num_batches, :batch_size, :seq_len, :N_ACT]
    tt_indices_torch = tt_indices_torch[:num_batches, :batch_size, :seq_len, :N_ACT]

    # Flatten to 2D [num_tokens, N_ACT] for recall calculation
    tt_indices_2d = tt_indices_torch.reshape(-1, N_ACT)
    ref_indices_2d = ref_indices.reshape(-1, N_ACT)

    recall = calculate_average_recall(tt_indices_2d, ref_indices_2d)
    recall_threshold = 0.9
    recall_passed = recall >= recall_threshold
    status = "PASS" if recall_passed else "FAIL"
    logger.info(
        f"[{status}] Recall = {recall:.4f} (threshold: {recall_threshold}) "
        f"for num_batches={num_batches}, batch_size={batch_size}, seq_len={seq_len}"
    )

    pcc_threshold = 0.97
    weights_passed, weights_pcc = comp_pcc(tt_weights_torch, ref_weights, pcc_threshold)
    status = "PASS" if weights_passed else "FAIL"
    logger.info(
        f"[{status}] Weights PCC = {weights_pcc:.4f} (threshold: {pcc_threshold}) "
        f"for num_batches={num_batches}, batch_size={batch_size}, seq_len={seq_len}"
    )

    assert recall_passed, (
        f"Recall is {recall:.4f}, expected >= {recall_threshold} "
        f"for num_batches={num_batches}, batch_size={batch_size}, seq_len={seq_len}"
    )
    assert weights_passed, (
        f"Weights PCC is {weights_pcc:.4f}, expected >= {pcc_threshold} "
        f"for num_batches={num_batches}, batch_size={batch_size}, seq_len={seq_len}"
    )


SENTINEL_PARAMS = [
    # (seq_len, num_real_tokens, pad_side, id)
    (128, 100, 0, "right_pad_boundary"),
    (128, 100, 1, "left_pad_boundary"),
    (128, 64, 0, "right_pad_half"),
    (128, 1, 0, "right_pad_almost_all"),
    (128, 128, 0, "right_pad_none"),
    (33, 17, 0, "right_pad_non_tile_aligned"),
    (33, 17, 1, "left_pad_non_tile_aligned"),
]


@pytest.mark.parametrize(
    "seq_len,num_real_tokens,pad_side",
    [(s, n, p) for s, n, p, _ in SENTINEL_PARAMS],
    ids=[i for _, _, _, i in SENTINEL_PARAMS],
)
def test_moe_grouped_topk_w_padding_awareness(device, seq_len, num_real_tokens, pad_side):
    """Verify that padded token rows get sentinel indices while real rows are bit-exact to baseline."""
    scores_in, bias_in = make_gate_inputs(device, seq_len)

    baseline_indices = _run_indices(scores_in, bias_in, seq_len)
    padding_config = create_padding_config(device, num_real_tokens, pad_side)
    padded_indices = _run_indices(scores_in, bias_in, seq_len, padding_config=padding_config)

    real_mask = real_token_mask([num_real_tokens], seq_len, pad_side)
    assert_sentinel_padding(padded_indices, baseline_indices, real_mask)


SP_PADDING_AWARENESS_PARAMS = [
    # (local_real_tokens per SP shard, pad_side, id)
    ([32, 32, 17, 0], 0, "right_pad_full_partial_empty"),
    ([32, 32, 0, 0], 0, "right_pad_full_empty"),
    ([0, 17, 32, 32], 1, "left_pad_empty_partial_full"),
    ([0, 0, 32, 32], 1, "left_pad_empty_full"),
]


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(
            (4, 1),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="linear"),
            id="linear-4x1",
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "local_real_tokens,pad_side",
    [(n, p) for n, p, _ in SP_PADDING_AWARENESS_PARAMS],
    ids=[i for _, _, i in SP_PADDING_AWARENESS_PARAMS],
)
def test_moe_grouped_topk_w_padding_awareness_sp(mesh_device, local_real_tokens, pad_side):
    """Verify per-SP-shard padding config handles full, partial, and empty real-token shards."""
    seq_len_per_shard = 32
    seq_len = seq_len_per_shard * len(local_real_tokens)
    mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=mesh_device.shape)
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=mesh_device.shape)

    scores_in, bias_in = make_gate_inputs(mesh_device, seq_len, mesh_mapper=mesh_mapper)

    baseline_indices = _run_indices(scores_in, bias_in, seq_len, composer=composer)
    padding_config = create_sharded_padding_config(mesh_device, local_real_tokens, pad_side)
    padded_indices = _run_indices(scores_in, bias_in, seq_len, padding_config=padding_config, composer=composer)

    real_mask = real_token_mask(local_real_tokens, seq_len_per_shard, pad_side)
    assert_sentinel_padding(padded_indices, baseline_indices, real_mask)


BALANCED_PADDING_PARAMS = [
    # (num_real_tokens, padding_side, id)
    (100, "right", "balanced_right_pad_partial"),
    (64, "right", "balanced_right_pad_half"),
    (128, "right", "balanced_right_pad_none"),
    (1, "right", "balanced_right_pad_almost_all"),
    (100, "left", "balanced_left_pad_partial"),
    (64, "left", "balanced_left_pad_half"),
]


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(
            (4, 1),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="linear"),
            id="linear-4x1",
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "num_real_tokens,padding_side",
    [(n, p) for n, p, _ in BALANCED_PADDING_PARAMS],
    ids=[i for _, _, i in BALANCED_PADDING_PARAMS],
)
def test_moe_grouped_topk_w_balanced_padding_awareness_sp(mesh_device, num_real_tokens, padding_side):
    """Verify padding awareness works correctly with zigzag/balanced sequence placement.

    Reorders input data using the same balanced chunk order as the transformer,
    then verifies that the balanced padding config produces correct sentinel masking
    (sentinels on padded rows, bit-exact match on real rows vs. unpadded baseline).
    """
    sp_factor = mesh_device.shape[0]
    seq_len_per_shard = 32
    seq_len = seq_len_per_shard * sp_factor
    pad_side = 0 if padding_side == "right" else 1

    # Reorder with balanced chunk order (same as transformer does before sharding)
    chunk_order = create_balanced_chunk_order(sp_factor)
    transform = lambda t: reorder_tensor_chunks(t, chunk_order, seq_dim=2)  # noqa: E731

    mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=mesh_device.shape)
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=mesh_device.shape)

    scores_in, bias_in = make_gate_inputs(mesh_device, seq_len, mesh_mapper=mesh_mapper, transform=transform)

    # Baseline: no padding config (all rows treated as real)
    baseline_indices = _run_indices(scores_in, bias_in, seq_len, composer=composer)

    # Compute balanced per-device real token counts
    local_real_tokens = balanced_local_real_tokens(sp_factor, seq_len, num_real_tokens, padding_side)
    logger.info(f"Balanced local_real_tokens: {local_real_tokens} (total={sum(local_real_tokens)})")

    padding_config = create_sharded_padding_config(mesh_device, local_real_tokens, pad_side)
    padded_indices = _run_indices(scores_in, bias_in, seq_len, padding_config=padding_config, composer=composer)

    # Masks are built in the reordered/balanced local buffer order
    real_mask = real_token_mask(local_real_tokens, seq_len_per_shard, pad_side)
    assert_sentinel_padding(padded_indices, baseline_indices, real_mask)
