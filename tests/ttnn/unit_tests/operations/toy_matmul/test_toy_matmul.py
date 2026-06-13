# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for toy_matmul — minimal 1D in0-multicast matmul via generic_op.

Torch golden compare only: golden = a.float() @ b.float(), PCC >= 0.99 (bf16).
"""

import pytest
import torch
import ttnn

from ttnn.operations.toy_matmul import toy_matmul
from tests.ttnn.utils_for_testing import assert_with_pcc


# `blocks` is None for auto-derived in0_block_w / out_subblock_h / out_subblock_w,
# or an explicit (in0_block_w, out_subblock_h, out_subblock_w) override.
@pytest.mark.parametrize(
    "M, K, N, grid, blocks",
    [
        # Smoke: 2x2 grid, per_core_N=1, auto-derived blocks (run this one first).
        pytest.param(128, 256, 128, (2, 2), None, id="smoke__2x2_auto"),
        # Canonical: 8x8 grid, Kt=32, per_core_N=4 — fully auto-derived.
        pytest.param(128, 1024, 8192, (8, 8), None, id="canonical__8x8_auto"),
        # Third: 8x8 grid, Kt=16, per_core_N=2 — fully auto-derived.
        pytest.param(128, 512, 4096, (8, 8), None, id="third__8x8_auto"),
        # Explicit override that forces in1_num_subblocks=2 (out_subblock 2x2 on per_core_N=4,
        # per_core_M=4) to exercise the SubblockMajor writer ordering in both subblock dims.
        pytest.param(128, 1024, 8192, (8, 8), (8, 2, 2), id="canonical__8x8_override_2x2"),
    ],
)
def test_toy_matmul(device, M, K, N, grid, blocks):
    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N), dtype=torch.bfloat16)

    golden = torch_a.float() @ torch_b.float()

    a = ttnn.from_torch(
        torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    b = ttnn.from_torch(
        torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    kwargs = {}
    if blocks is not None:
        kwargs = dict(in0_block_w=blocks[0], out_subblock_h=blocks[1], out_subblock_w=blocks[2])

    out = toy_matmul(a, b, compute_grid=grid, **kwargs)
    torch_out = ttnn.to_torch(out).float()

    assert_with_pcc(golden, torch_out, 0.99)
