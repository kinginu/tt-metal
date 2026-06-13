# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
toy_matmul — a minimal, pedagogical 1D in0-multicast matmul built entirely in
Python via ttnn.generic_op + a ProgramDescriptor.

Stripped to the bone: no bias, no L1 packer accumulation, no SFPU activation, no
untilize, no sparsity, no multi-batch, no sharding, no tile padding. Assumes
tile-aligned, bf16, DRAM-interleaved inputs/outputs.

A fully-featured equivalent already exists (matmul_multicore_reuse_mcast_1d); this
is the simplest possible re-derivation for eval / teaching.
"""

import ttnn
from .toy_matmul_program_descriptor import create_program_descriptor


def toy_matmul(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    compute_grid: tuple = (8, 8),
    in0_block_w: int = None,
    out_subblock_h: int = None,
    out_subblock_w: int = None,
    fp32_dest_acc_en: bool = True,
) -> ttnn.Tensor:
    """
    Compute C = A @ B with a 1D in0-multicast strategy across `compute_grid`.

    Args:
        a: [M, K] tensor (TILE_LAYOUT, bfloat16, DRAM-interleaved).
        b: [K, N] tensor (TILE_LAYOUT, bfloat16, DRAM-interleaved).
        compute_grid: (grid_x, grid_y) worker grid. N is split across all cores.
        in0_block_w: K-block width in tiles (the contraction chunk size). Auto-derived
            from Kt when None.
        out_subblock_h, out_subblock_w: output subblock dims in tiles. Auto-derived from
            (per_core_M, per_core_N) and the DST budget when None. When supplied, must
            satisfy h*w <= the reload-safe DST limit (4 under fp32, else 8).
        fp32_dest_acc_en: fp32 DEST accumulation (recommended for bf16 inputs with Kt>1).
    """
    device = a.device()
    M = int(a.shape[-2])
    N = int(b.shape[-1])

    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([M, N]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    program_descriptor = create_program_descriptor(
        a,
        b,
        out,
        grid=compute_grid,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )
    return ttnn.generic_op([a, b, out], program_descriptor)
