# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared test helpers for the DeepSeek-V3 MoE / grouped-topk ops.

Covers input generation, padding-config builders, real/pad masking, balanced
(zigzag) placement math, and the sentinel-padding assertion used by the
moe_grouped_topk / dispatch padding-awareness tests. These build real ttnn
device tensors and are intended for tests only (not production model code).
"""

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
    """Generate (1, 1, seq_len, num_experts) score/bias inputs as TILE-layout ttnn tensors.

    ``transform`` (if given) is applied to the torch tensors before device upload — used by
    the balanced-placement test to zigzag-reorder the sequence the same way the transformer does.
    """
    torch.manual_seed(seed)
    scores = generate_distinct_sigmoid_inputs((1, 1, seq_len, num_experts), dtype=torch.float32)
    bias = torch.randn(1, 1, seq_len, num_experts, dtype=torch.float32)
    if transform is not None:
        scores, bias = transform(scores), transform(bias)

    def to_device(t):
        return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper)

    return to_device(scores), to_device(bias)


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
