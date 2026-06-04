# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared test helpers for the DeepSeek-V3 MoE / grouped-topk ops.

Covers input generation, padding-config builders, real/pad masking, balanced
(zigzag) placement math, the sentinel-padding assertion, and the parametrize
scenario list used by the moe_grouped_topk / dispatch padding-awareness tests.
These build real ttnn device tensors and are intended for tests only (not
production model code).
"""

import pytest
import torch
from loguru import logger

import ttnn

# Single source of truth for the gate hyperparameters used across the MoE tests.
GATE_KWARGS = dict(
    n_groups=8,
    summed_experts_per_group=2,
    topk_groups=4,
    n_activated_experts=8,
    route_scale=0.5,
    epsilon=1e-20,
)
TOTAL_EXPERTS = 256
N_ACT = GATE_KWARGS["n_activated_experts"]
SENTINEL = TOTAL_EXPERTS  # padded rows are routed to this out-of-range expert id


def generate_distinct_sigmoid_inputs(shape, min_val=0.05, max_val=0.95, dtype=torch.float32):
    """Generate per-row-distinct logits whose sigmoids are unique within a row.

    Used to give the gate well-separated scores so top-k selection is unambiguous.
    """
    row_size = shape[-1]
    num_rows = torch.tensor(shape[:-1]).prod().item()

    num_candidates = row_size * 4
    candidates = torch.linspace(min_val, max_val, num_candidates, dtype=dtype)
    unique = candidates.unique()

    if unique.numel() < row_size:
        raise ValueError(f"Cannot generate {row_size} distinct sigmoid outputs in [{min_val}, {max_val}].")

    all_rows = []
    for _ in range(num_rows):
        perm = torch.randperm(unique.numel())[:row_size]
        sigmoid_outputs = unique[perm]
        sigmoid_outputs_f32 = sigmoid_outputs.float()
        row_inputs = torch.log(sigmoid_outputs_f32 / (1 - sigmoid_outputs_f32))
        all_rows.append(row_inputs)

    return torch.stack(all_rows).to(dtype).reshape(shape)


def create_padding_config(device, num_real_tokens, pad_side):
    """Single-device padding config: one [num_real_tokens, pad_side] row."""
    return ttnn.from_torch(
        torch.tensor([[num_real_tokens, pad_side]], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )


def create_sharded_padding_config(mesh_device, local_real_tokens, pad_side):
    """Per-SP-shard padding config: one [real, pad_side] row per shard, sharded on dim 0."""
    return ttnn.from_torch(
        torch.tensor([[real, pad_side] for real in local_real_tokens], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=mesh_device.shape),
    )


def make_gate_inputs(device, seq_len, *, num_experts=TOTAL_EXPERTS, mesh_mapper=None, transform=None, seed=42):
    """Generate (1, 1, seq_len, num_experts) score/bias inputs.

    Returns ``(scores, bias, scores_in, bias_in)``: the (possibly transformed) torch tensors —
    needed for the golden reference — and their TILE-layout ttnn uploads. ``transform`` (if
    given) is applied to the torch tensors before device upload — used by the balanced-placement
    scenarios to zigzag-reorder the sequence the same way the transformer does.
    """
    torch.manual_seed(seed)
    scores = generate_distinct_sigmoid_inputs((1, 1, seq_len, num_experts), dtype=torch.float32)
    bias = torch.randn(1, 1, seq_len, num_experts, dtype=torch.float32)
    if transform is not None:
        scores, bias = transform(scores), transform(bias)

    def to_device(t):
        return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper)

    return scores, bias, to_device(scores), to_device(bias)


def real_token_mask(real_counts, shard_len, pad_side):
    """Boolean mask of real (non-padded) rows.

    Works for the single-device case (pass ``[num_real_tokens]`` with ``shard_len=seq_len``) and
    the SP-sharded case (pass per-shard counts with ``shard_len`` = per-shard sequence length).
    """
    masks = []
    for real in real_counts:
        idx = torch.arange(shard_len)
        masks.append(idx < real if pad_side == 0 else idx >= (shard_len - real))
    return torch.cat(masks)


def balanced_local_real_tokens(sp_factor, total_tokens, num_real_tokens, padding_side):
    """Compute per-SP-device real token counts under zigzag/balanced placement."""
    num_chunks = 2 * sp_factor
    chunk_size = total_tokens // num_chunks
    result = []
    for sp_idx in range(sp_factor):
        chunk_a = sp_idx
        chunk_b = num_chunks - 1 - sp_idx
        if padding_side == "right":
            real_a = min(chunk_size, max(0, num_real_tokens - chunk_a * chunk_size))
            real_b = min(chunk_size, max(0, num_real_tokens - chunk_b * chunk_size))
        else:
            total_padded = max(0, total_tokens - num_real_tokens)
            pad_a = min(chunk_size, max(0, total_padded - chunk_a * chunk_size))
            pad_b = min(chunk_size, max(0, total_padded - chunk_b * chunk_size))
            real_a = chunk_size - pad_a
            real_b = chunk_size - pad_b
        result.append(real_a + real_b)
    return result


def assert_sentinel_padding(padded_indices, baseline_indices, real_mask, sentinel=SENTINEL):
    """Real rows must be bit-exact to the unpadded baseline; padded rows must all be sentinel."""
    if real_mask.any():
        real = padded_indices[0, 0, real_mask]
        baseline_real = baseline_indices[0, 0, real_mask]
        assert torch.equal(real, baseline_real), (
            f"Real-row indices differ from baseline!\n"
            f"  Mismatched rows: {(real != baseline_real).any(dim=-1).nonzero().flatten().tolist()}"
        )
        logger.info(f"Real-row indices: bit-exact match ({real_mask.sum().item()} rows)")

    if (~real_mask).any():
        pad_rows = padded_indices[0, 0, ~real_mask]
        expected = torch.full_like(pad_rows, sentinel)
        assert torch.equal(
            pad_rows, expected
        ), f"Padded-row indices are not all sentinel ({sentinel})!\n  Got: {pad_rows}\n  Expected: all {sentinel}"
        logger.info(f"Padded-row indices: all sentinel ({(~real_mask).sum().item()} rows)")


_SP_MARK = pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="linear")


def grouped_topk_scenarios():
    """Parametrize scenarios for the unified moe_grouped_topk test.

    Each entry binds its own ``mesh_device`` (indirect), so a single test body covers single-chip,
    SP-sharded, and balanced placements. Columns are:
        (mesh_device, seq_len_per_shard, local_real_tokens, pad_side, balanced)
    where ``seq_len = seq_len_per_shard * len(local_real_tokens)``.

    Single-chip entries use a (1, 1) mesh and omit the ``requires_mesh_topology`` mark so they run
    on any machine; SP/balanced entries use a marked (4, 1) linear mesh. Full-real entries (all
    tokens real) exercise the no-padding correctness path against the torch golden; the others
    additionally exercise sentinel masking.
    """
    scenarios = []

    # Correctness / no-padding single-chip cases (full real -> exercises the golden recall+PCC path).
    # (seq_len, id)
    for seq_len, sid in [(1, "minimal"), (33, "just_over_one_tile"), (128, "four_tiles"), (3200, "realistic")]:
        scenarios.append(pytest.param((1, 1), seq_len, [seq_len], 0, False, id=sid))

    # Single-chip padding cases (1x1 mesh): the whole sequence lives on one device.
    # (seq_len, num_real_tokens, pad_side, id)
    for seq_len, num_real_tokens, pad_side, sid in [
        (128, 100, 0, "single_right_pad_boundary"),
        (128, 100, 1, "single_left_pad_boundary"),
        (128, 64, 0, "single_right_pad_half"),
        (128, 1, 0, "single_right_pad_almost_all"),
        (33, 17, 0, "single_right_pad_non_tile_aligned"),
        (33, 17, 1, "single_left_pad_non_tile_aligned"),
    ]:
        scenarios.append(pytest.param((1, 1), seq_len, [num_real_tokens], pad_side, False, id=sid))

    # SP explicit per-shard counts (4x1 mesh): full, partial, and empty real-token shards.
    # (local_real_tokens, pad_side, id)
    for local_real_tokens, pad_side, sid in [
        ([32, 32, 17, 0], 0, "sp_right_pad_full_partial_empty"),
        ([32, 32, 0, 0], 0, "sp_right_pad_full_empty"),
        ([0, 17, 32, 32], 1, "sp_left_pad_empty_partial_full"),
        ([0, 0, 32, 32], 1, "sp_left_pad_empty_full"),
    ]:
        scenarios.append(pytest.param((4, 1), 32, local_real_tokens, pad_side, False, marks=_SP_MARK, id=sid))

    # SP with zigzag/balanced placement (4x1 mesh). Per-shard counts are precomputed (pure-python)
    # to match the transformer's balanced chunk order; the ``balanced`` flag triggers the same
    # input reorder inside the test body.
    # (num_real_tokens, padding_side, id)
    for num_real_tokens, padding_side, sid in [
        (100, "right", "balanced_right_pad_partial"),
        (64, "right", "balanced_right_pad_half"),
        (128, "right", "balanced_right_pad_none"),
        (1, "right", "balanced_right_pad_almost_all"),
        (100, "left", "balanced_left_pad_partial"),
        (64, "left", "balanced_left_pad_half"),
    ]:
        local_real_tokens = balanced_local_real_tokens(4, 128, num_real_tokens, padding_side)
        pad_side = 0 if padding_side == "right" else 1
        scenarios.append(pytest.param((4, 1), 32, local_real_tokens, pad_side, True, marks=_SP_MARK, id=sid))

    return scenarios
