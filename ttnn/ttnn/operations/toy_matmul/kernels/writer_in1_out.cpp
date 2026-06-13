// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// toy_matmul in1 reader + C writer — runs on EVERY core.
//
// Phase 1: stream this core's B columns from DRAM into cb_in1, one K-block at a
//          time (backpressured against compute via the double-buffered CB).
// Phase 2: wait for the full output block in cb_out and write it to DRAM.
//
// The output is packed by compute in SubblockMajor order, so the writer walks
// the CB in that exact order (m_subblock, n_subblock, row, col) and maps each
// tile's running byte offset to its DRAM page. Do NOT use a linear tile-row-major
// writer here — with in1_num_subblocks > 1 it would scramble the columns.
//
// Runs on RISCV_0/BRISC, NoC1 (WriterConfigDescriptor) for both B reads and C writes.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_N = get_compile_time_arg_val(1);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t Nt = get_compile_time_arg_val(3);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(4);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(5);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(6);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(8);
    constexpr auto b_args = TensorAccessorArgs<9>();
    constexpr auto out_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");

    const uint32_t b_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t n_start = get_arg_val<uint32_t>(2);

    constexpr uint32_t in1_block_tiles = in0_block_w * per_core_N;
    constexpr uint32_t out_block_tiles = per_core_M * per_core_N;

    Noc noc;
    CircularBuffer cb_in1_obj(cb_in1);
    CircularBuffer cb_out_obj(cb_out);
    const uint32_t tile_bytes = get_tile_size(cb_in1);

    const auto b = TensorAccessor(b_args, b_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    // Phase 1: read this core's B K-blocks. B is [Kt, Nt]; this core owns columns
    // [n_start, n_start + per_core_N). CB layout is k-major / n-minor.
    for (uint32_t block = 0; block < num_k_blocks; ++block) {
        cb_in1_obj.reserve_back(in1_block_tiles);
        for (uint32_t kk = 0; kk < in0_block_w; ++kk) {
            for (uint32_t n = 0; n < per_core_N; ++n) {
                const uint32_t page_id = (block * in0_block_w + kk) * Nt + (n_start + n);
                const uint32_t offset_bytes = (kk * per_core_N + n) * tile_bytes;
                noc.async_read(b, cb_in1_obj, tile_bytes, {.page_id = page_id}, {.offset_bytes = offset_bytes});
            }
        }
        noc.async_read_barrier();
        cb_in1_obj.push_back(in1_block_tiles);
    }

    // Phase 2: write the output block. Walk cb_out in the SubblockMajor order
    // compute packed it; the running read_offset tracks the physical CB layout.
    cb_out_obj.wait_front(out_block_tiles);
    uint32_t read_offset = 0;
    for (uint32_t m_sb = 0; m_sb < in0_num_subblocks; ++m_sb) {
        for (uint32_t n_sb = 0; n_sb < in1_num_subblocks; ++n_sb) {
            for (uint32_t r = 0; r < out_subblock_h; ++r) {
                for (uint32_t c = 0; c < out_subblock_w; ++c) {
                    const uint32_t m = m_sb * out_subblock_h + r;
                    const uint32_t n = n_start + n_sb * out_subblock_w + c;
                    const uint32_t page_id = m * Nt + n;
                    noc.async_write(
                        use<CircularBuffer::AddrSelector::READ_PTR>(cb_out_obj),
                        out,
                        tile_bytes,
                        {.offset_bytes = read_offset},
                        {.page_id = page_id});
                    read_offset += tile_bytes;
                }
            }
        }
    }
    noc.async_write_barrier();
    cb_out_obj.pop_front(out_block_tiles);
}
