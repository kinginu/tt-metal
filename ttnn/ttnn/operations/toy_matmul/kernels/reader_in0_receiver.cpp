// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// toy_matmul in0 RECEIVER reader — runs on every core EXCEPT the sender.
//
// Each K-block: reserve the cb_in0 landing slot, receive the multicast A block
// from the sender, push it for compute. The reserve/push lockstep with the
// sender keeps cb_in0's write pointer identical across all cores, so the
// sender's src==dst landing address matches every receiver's slot (INV9).
//
// Runs on RISCV_1/NCRISC, NoC0 (ReaderConfigDescriptor).

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
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t data_ready_sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t consumed_sem_id = get_compile_time_arg_val(4);

    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");

    const uint32_t sender_vx = get_arg_val<uint32_t>(0);
    const uint32_t sender_vy = get_arg_val<uint32_t>(1);

    constexpr uint32_t in0_block_tiles = per_core_M * in0_block_w;

    Noc noc;
    CircularBuffer cb_in0_obj(cb_in0);

    // Receiver-side Pipe: just the noc (sem ids are template params). The sender's coords
    // (target of the consumed ack) are passed to receive().
    ReceiverPipe<data_ready_sem_id, consumed_sem_id> in0_pipe(noc);

    for (uint32_t block = 0; block < num_k_blocks; ++block) {
        cb_in0_obj.reserve_back(in0_block_tiles);
        in0_pipe.receive(sender_vx, sender_vy);
        cb_in0_obj.push_back(in0_block_tiles);
    }
}
