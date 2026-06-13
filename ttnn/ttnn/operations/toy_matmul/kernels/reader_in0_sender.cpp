// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// toy_matmul in0 SENDER reader — runs on the single sender core (logical (0,0)).
//
// Each K-block: read this A K-block [per_core_M x in0_block_w] from DRAM into
// cb_in0, then multicast it to every other core via the SenderPipe helper. Because
// src == dst the Pipe infers EXCLUDE_SRC: the sender's own copy is already in
// place, so the mcast hits the num_active_cores-1 other cores (num_active_cores is
// the FULL count including this sender).
//
// Runs on RISCV_1/NCRISC, NoC0 (ReaderConfigDescriptor) — the in0 mcast rides NoC0.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t per_core_M = get_compile_time_arg_val(0);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(1);
    constexpr uint32_t Kt = get_compile_time_arg_val(2);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t num_active_cores = get_compile_time_arg_val(4);
    constexpr uint32_t data_ready_sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t consumed_sem_id = get_compile_time_arg_val(6);
    constexpr auto a_args = TensorAccessorArgs<7>();

    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");

    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t vx0 = get_arg_val<uint32_t>(1);
    const uint32_t vy0 = get_arg_val<uint32_t>(2);
    const uint32_t vx1 = get_arg_val<uint32_t>(3);
    const uint32_t vy1 = get_arg_val<uint32_t>(4);

    constexpr uint32_t in0_block_tiles = per_core_M * in0_block_w;

    Noc noc;
    CircularBuffer cb_in0_obj(cb_in0);
    const uint32_t tile_bytes = get_tile_size(cb_in0);
    const uint32_t in0_block_bytes = in0_block_tiles * tile_bytes;

    const auto a = TensorAccessor(a_args, a_addr);

    // Defaults are the matmul-in0 protocol: Staging::Flag, PRE_HANDSHAKE=true. Count + sem ids are
    // template params; the only runtime ctor input is the receiver rectangle. The local data-ready
    // VALID pre-set is owned by the ctor (INITIAL_READY defaults to VALID).
    SenderPipe<num_active_cores, data_ready_sem_id, consumed_sem_id> in0_pipe(noc, McastRect{vx0, vy0, vx1, vy1});

    for (uint32_t block = 0; block < num_k_blocks; ++block) {
        cb_in0_obj.reserve_back(in0_block_tiles);
        const uint32_t src = cb_in0_obj.get_write_ptr();

        // A is [Mt, Kt]; this K-block is A[:, block*in0_block_w : (block+1)*in0_block_w].
        // CB layout is m-major / k-minor: offset (h*in0_block_w + w) tiles.
        for (uint32_t h = 0; h < per_core_M; ++h) {
            for (uint32_t w = 0; w < in0_block_w; ++w) {
                const uint32_t page_id = h * Kt + block * in0_block_w + w;
                const uint32_t offset_bytes = (h * in0_block_w + w) * tile_bytes;
                noc.async_read(a, cb_in0_obj, tile_bytes, {.page_id = page_id}, {.offset_bytes = offset_bytes});
            }
        }
        noc.async_read_barrier();

        // src == dst -> Pipe infers EXCLUDE_SRC (own copy already staged above).
        in0_pipe.send(src, src, in0_block_bytes);

        cb_in0_obj.push_back(in0_block_tiles);
    }
}
