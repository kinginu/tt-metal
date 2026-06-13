# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Program descriptor for toy_matmul — a minimal 1D in0-multicast matmul.

Parallelization (the contract):
  * 1D grid of num_cores; the parallel axis is N.
  * M fits in a single block: per_core_M == Mt (mcast_in0's num_blocks_y == 1).
  * N split column-wise: per_core_N = Nt / num_cores  (require Nt % num_cores == 0).
  * K split into num_k_blocks chunks of in0_block_w tiles (require Kt % in0_block_w == 0).

The A K-block is identical on every core, so one sender core (logical (0,0)) reads
it from DRAM and multicasts it to all the others via the `Pipe` helper. Each core
reads its own B columns from DRAM locally and writes its own C block.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32

# CB indices (matching the reference matmul factory).
CB_IN0 = 0
CB_IN1 = 1
CB_OUT = 4
CB_INTERMED0 = 5

# Semaphore ids.
DATA_READY = 0
CONSUMED = 1

# Auto-derivation tuning knob: cap for the K-block width when in0_block_w is not given.
# in0_block_w drives the in0/in1 CB footprint (≈ per_core_M*ibw and ibw*per_core_N tiles,
# double-buffered), so a small cap keeps L1 use modest at the cost of more K-blocks. 4 is a
# conservative chunk that divides the common tile-K values (8/16/32) and leaves ample L1.
IN0_BLOCK_W_CAP = 4


def _dst_capacity(fp32_dest_acc_en: bool, dst_full_sync_en: bool = False) -> int:
    """Usable DST tiles for a 32x32 tile. Mirrors the matmul auto-tuner's
    dst_capacity_from_flags: half-sync exposes 16 tiles, full-sync 8; fp32 accumulation
    halves it."""
    base = 8 if dst_full_sync_en else 16
    return base // 2 if fp32_dest_acc_en else base


def _choose_in0_block_w(Kt: int, cap: int = IN0_BLOCK_W_CAP) -> int:
    """Largest divisor of Kt that does not exceed `cap` (falls back to 1)."""
    for d in range(min(cap, Kt), 0, -1):
        if Kt % d == 0:
            return d
    return 1


def _choose_subblock(per_core_M: int, per_core_N: int, max_tiles: int) -> tuple:
    """Pick (out_subblock_h, out_subblock_w) dividing (per_core_M, per_core_N) whose tile
    volume is maximal but <= max_tiles. Tie-break mirrors the production matmul auto-tuner:
    largest volume, then the LLK 'fast path' (h==1 or w==1), then most square."""
    best_key, best = None, (1, 1)
    for h in range(1, per_core_M + 1):
        if per_core_M % h:
            continue
        for w in range(1, per_core_N + 1):
            if per_core_N % w or h * w > max_tiles:
                continue
            fast_path = 0 if (h == 1 or w == 1) else 1
            key = (-(h * w), fast_path, abs(h - w))
            if best_key is None or key < best_key:
                best_key, best = key, (h, w)
    return best


def create_program_descriptor(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    out: ttnn.Tensor,
    *,
    grid: tuple,
    in0_block_w: int = None,
    out_subblock_h: int = None,
    out_subblock_w: int = None,
    fp32_dest_acc_en: bool = True,
) -> ttnn.ProgramDescriptor:
    device = a.device()
    grid_x, grid_y = grid
    num_cores = grid_x * grid_y

    M, K = int(a.shape[-2]), int(a.shape[-1])
    K2, N = int(b.shape[-2]), int(b.shape[-1])
    assert K == K2, f"inner dims must match: A K={K}, B K={K2}"

    Mt, Kt, Nt = M // TILE_DIM, K // TILE_DIM, N // TILE_DIM

    assert Nt % num_cores == 0, f"Nt ({Nt}) must be divisible by num_cores ({num_cores})"
    per_core_M = Mt
    per_core_N = Nt // num_cores

    # DST sub-block tile budget. The DEST register holds (dst_full_sync_en ? 8 : 16) tiles,
    # halved again for fp32 accumulation (see matmul auto-tuner dst_capacity_from_flags).
    # With packer_l1_acc=false the K-blocking spill/reload runs through DEST
    # (copy_block_matmul_partials reloads the partial, then the matmul accumulates onto it),
    # which needs the sub-block to fit in HALF the raw capacity. Empirically (this op uses
    # half-sync DEST): the reload path is correct only for out_subblock_h*out_subblock_w <= 4
    # under fp32 (raw cap 8), or <= 8 without fp32 (raw cap 16). An 8-tile sub-block under
    # fp32 silently corrupts the accumulation (PCC ~0.6). Single-K-block matmuls would tolerate
    # the full cap, but we apply the conservative reload-safe bound uniformly.
    max_subblock_tiles = _dst_capacity(fp32_dest_acc_en) // 2

    # --- Auto-derive block params when not supplied (None) ---
    if in0_block_w is None:
        in0_block_w = _choose_in0_block_w(Kt)
    if out_subblock_h is None or out_subblock_w is None:
        derived_h, derived_w = _choose_subblock(per_core_M, per_core_N, max_subblock_tiles)
        out_subblock_h = derived_h if out_subblock_h is None else out_subblock_h
        out_subblock_w = derived_w if out_subblock_w is None else out_subblock_w

    # --- Contract assertions (also guard caller-supplied overrides) ---
    assert Kt % in0_block_w == 0, f"Kt ({Kt}) must be divisible by in0_block_w ({in0_block_w})"
    assert per_core_M % out_subblock_h == 0, f"per_core_M ({per_core_M}) % out_subblock_h ({out_subblock_h}) != 0"
    assert per_core_N % out_subblock_w == 0, f"per_core_N ({per_core_N}) % out_subblock_w ({out_subblock_w}) != 0"
    assert out_subblock_h * out_subblock_w <= max_subblock_tiles, (
        f"out_subblock_h*out_subblock_w ({out_subblock_h}*{out_subblock_w}) must be <= {max_subblock_tiles} "
        f"(reload-safe DST limit for fp32_dest_acc_en={fp32_dest_acc_en})"
    )

    num_k_blocks = Kt // in0_block_w
    in0_num_subblocks = per_core_M // out_subblock_h
    in1_num_subblocks = per_core_N // out_subblock_w

    in0_block_tiles = per_core_M * in0_block_w
    in1_block_tiles = in0_block_w * per_core_N
    out_block_tiles = per_core_M * per_core_N

    bf16_tile = ttnn.tile_size(ttnn.bfloat16)
    fp32_tile = ttnn.tile_size(ttnn.float32)

    # Intermediate (spill/reload) CB format follows the DEST accumulation precision:
    # Float32 when fp32_dest_acc_en, else bf16 (matching the output).
    interm_format = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
    interm_tile = fp32_tile if fp32_dest_acc_en else bf16_tile

    # --- Core sets ---
    sender_core = ttnn.CoreCoord(0, 0)
    sender_crs = ttnn.CoreRangeSet([ttnn.CoreRange(sender_core, sender_core)])
    all_crs = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))]
    )
    recv_ranges = []
    if grid_x > 1:  # rest of row 0
        recv_ranges.append(ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(grid_x - 1, 0)))
    if grid_y > 1:  # rows 1..grid_y-1, all columns
        recv_ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(grid_x - 1, grid_y - 1)))
    recv_crs = ttnn.CoreRangeSet(recv_ranges)

    # P3: num_active_receiver_cores is the FULL recipient count INCLUDING the sender (it sits in the
    # broadcast box and consumes its own A copy). The SenderPipe derives mcast_dests = N-1 (EXCLUDE,
    # src==dst) and ack_count = N-1 internally.
    num_active_cores = num_cores

    # --- Mcast rectangle (virtual coords of the full grid) ---
    v_lo = device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
    v_hi = device.worker_core_from_logical_core(ttnn.CoreCoord(grid_x - 1, grid_y - 1))
    vx0, vy0, vx1, vy1 = v_lo.x, v_lo.y, v_hi.x, v_hi.y
    sender_v = device.worker_core_from_logical_core(sender_core)
    sender_vx, sender_vy = sender_v.x, sender_v.y

    # --- Circular buffers (all on the full grid so the index->addr map is identical) ---
    def _cb(idx, total_size, fmt, page_size):
        return ttnn.CBDescriptor(
            total_size=total_size,
            core_ranges=all_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page_size)
            ],
        )

    cbs = [
        _cb(CB_IN0, 2 * in0_block_tiles * bf16_tile, ttnn.bfloat16, bf16_tile),
        _cb(CB_IN1, 2 * in1_block_tiles * bf16_tile, ttnn.bfloat16, bf16_tile),
        _cb(CB_OUT, out_block_tiles * bf16_tile, ttnn.bfloat16, bf16_tile),
        _cb(CB_INTERMED0, out_block_tiles * interm_tile, interm_format, interm_tile),
    ]

    # --- Semaphores (on the full grid) ---
    semaphores = [
        ttnn.SemaphoreDescriptor(id=DATA_READY, core_ranges=all_crs, initial_value=0),
        ttnn.SemaphoreDescriptor(id=CONSUMED, core_ranges=all_crs, initial_value=0),
    ]

    a_addr = a.buffer_address()
    b_addr = b.buffer_address()
    out_addr = out.buffer_address()

    named_cbs = [("cb_in0", CB_IN0), ("cb_in1", CB_IN1), ("cb_out", CB_OUT), ("cb_intermed0", CB_INTERMED0)]

    # --- Sender reader (sender core only) ---
    sender_ct = [per_core_M, in0_block_w, Kt, num_k_blocks, num_active_cores, DATA_READY, CONSUMED]
    sender_ct.extend(ttnn.TensorAccessorArgs(a).get_compile_time_args())
    sender_rt = ttnn.RuntimeArgs()
    sender_rt[sender_core.x][sender_core.y] = [a_addr, vx0, vy0, vx1, vy1]
    sender_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_in0_sender.cpp"),
        core_ranges=sender_crs,
        compile_time_args=sender_ct,
        named_compile_time_args=[("cb_in0", CB_IN0)],
        runtime_args=sender_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Receiver reader (all cores except the sender) ---
    recv_ct = [per_core_M, in0_block_w, num_k_blocks, DATA_READY, CONSUMED]
    recv_rt = ttnn.RuntimeArgs()
    for cy in range(grid_y):
        for cx in range(grid_x):
            if cx == 0 and cy == 0:
                continue
            recv_rt[cx][cy] = [sender_vx, sender_vy]
    recv_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_in0_receiver.cpp"),
        core_ranges=recv_crs,
        compile_time_args=recv_ct,
        named_compile_time_args=[("cb_in0", CB_IN0)],
        runtime_args=recv_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer (all cores): read this core's B columns, write its C block ---
    writer_ct = [
        in0_block_w,
        per_core_N,
        num_k_blocks,
        Nt,
        per_core_M,
        in0_num_subblocks,
        in1_num_subblocks,
        out_subblock_h,
        out_subblock_w,
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(b).get_compile_time_args())
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    writer_rt = ttnn.RuntimeArgs()
    for cy in range(grid_y):
        for cx in range(grid_x):
            lin = cy * grid_x + cx
            n_start = lin * per_core_N
            writer_rt[cx][cy] = [b_addr, out_addr, n_start]
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_in1_out.cpp"),
        core_ranges=all_crs,
        compile_time_args=writer_ct,
        named_compile_time_args=[("cb_in1", CB_IN1), ("cb_out", CB_OUT)],
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute (all cores) ---
    compute_ct = [in0_block_w, in0_num_subblocks, in1_num_subblocks, num_k_blocks, out_subblock_h, out_subblock_w]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"),
        core_ranges=all_crs,
        compile_time_args=compute_ct,
        named_compile_time_args=named_cbs,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=fp32_dest_acc_en,
        ),
    )

    return ttnn.ProgramDescriptor(
        kernels=[sender_kernel, recv_kernel, writer_kernel, compute_kernel],
        semaphores=semaphores,
        cbs=cbs,
    )
