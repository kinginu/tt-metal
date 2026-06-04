# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Unit test for ttnn.experimental.deepseek_prefill.moe_grouped_topk().

A single parametrized test covers single-chip, SP-sharded, and zigzag/balanced placements
(see grouped_topk_scenarios). For every scenario the unpadded baseline is validated against
a PyTorch golden reference (recall on selected experts + weights PCC). When the scenario
applies padding, the padded run is additionally checked for sentinel masking: padded rows
become the sentinel expert id while real rows stay bit-exact to the unpadded baseline.
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
    create_sharded_padding_config,
    grouped_topk_scenarios,
    make_gate_inputs,
    real_token_mask,
)
from tests.ttnn.utils_for_testing import comp_pcc

RECALL_THRESHOLD = 0.9
WEIGHTS_PCC_THRESHOLD = 0.97


def _run_topk(scores_in, bias_in, seq_len, *, padding_config=None, composer=None):
    """Run moe_grouped_topk; return (weights, indices) trimmed to [1, 1, seq_len, N_ACT]."""
    weights_out, indices_out = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
        scores_in, bias_in, padding_config=padding_config, **GATE_KWARGS
    )
    weights = ttnn.to_torch(weights_out, mesh_composer=composer)[:1, :1, :seq_len, :N_ACT]
    indices = ttnn.to_torch(indices_out, mesh_composer=composer)[:1, :1, :seq_len, :N_ACT].to(torch.int32)
    return weights, indices


def _assert_matches_golden(scores, bias, tt_weights, tt_indices):
    """Validate the unpadded baseline against the PyTorch grouped-gate golden (recall + weights PCC)."""
    config = DeepseekV3Config(
        hidden_size=64,
        n_routed_experts=TOTAL_EXPERTS,
        n_group=GATE_KWARGS["n_groups"],
        topk_group=GATE_KWARGS["topk_groups"],
        num_experts_per_tok=GATE_KWARGS["n_activated_experts"],
        routed_scaling_factor=GATE_KWARGS["route_scale"],
    )
    gate = MoEGate(config, use_bitonic_sort=False)
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

    recall = calculate_average_recall(tt_indices.reshape(-1, N_ACT), ref_indices.reshape(-1, N_ACT))
    weights_passed, weights_pcc = comp_pcc(tt_weights, ref_weights, WEIGHTS_PCC_THRESHOLD)
    logger.info(f"recall={recall:.4f} (>= {RECALL_THRESHOLD}), weights_pcc={weights_pcc} (>= {WEIGHTS_PCC_THRESHOLD})")

    assert recall >= RECALL_THRESHOLD, f"recall {recall:.4f} below {RECALL_THRESHOLD}"
    assert weights_passed, f"weights PCC {weights_pcc} below {WEIGHTS_PCC_THRESHOLD}"


@pytest.mark.parametrize(
    "mesh_device, seq_len_per_shard, local_real_tokens, pad_side, balanced",
    grouped_topk_scenarios(),
    indirect=["mesh_device"],
)
def test_moe_grouped_topk(mesh_device, seq_len_per_shard, local_real_tokens, pad_side, balanced):
    """Validate moe_grouped_topk across single-chip, SP-sharded, and balanced placements.

    For every scenario the unpadded baseline is checked against the PyTorch golden (recall +
    weights PCC). When the scenario applies padding, the padded run is additionally checked for
    sentinel masking: padded rows become the sentinel expert id and real rows stay bit-exact to
    the baseline. Recall-only cases are expressed as full-real (no-padding) scenarios.
    """
    sp_factor = len(local_real_tokens)
    seq_len = seq_len_per_shard * sp_factor

    transform = None
    if balanced:
        chunk_order = create_balanced_chunk_order(sp_factor)
        transform = lambda t: reorder_tensor_chunks(t, chunk_order, seq_dim=2)  # noqa: E731

    mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=mesh_device.shape)
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=mesh_device.shape)

    scores, bias, scores_in, bias_in = make_gate_inputs(
        mesh_device, seq_len, mesh_mapper=mesh_mapper, transform=transform
    )

    # Baseline (no padding): validate against the torch golden.
    baseline_weights, baseline_indices = _run_topk(scores_in, bias_in, seq_len, composer=composer)
    _assert_matches_golden(scores, bias, baseline_weights, baseline_indices)

    # Padding awareness: padded rows become sentinel, real rows stay bit-exact to the baseline.
    padding_config = create_sharded_padding_config(mesh_device, local_real_tokens, pad_side)
    _, padded_indices = _run_topk(scores_in, bias_in, seq_len, padding_config=padding_config, composer=composer)
    real_mask = real_token_mask(local_real_tokens, seq_len_per_shard, pad_side)
    assert_sentinel_padding(padded_indices, baseline_indices, real_mask)
