// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa writer: per token, drain the untilized [32,512] row-major block from compute (vDHt tile-sized
// pages, row-major bytes inside) and write each head-row to the ROW-MAJOR [1,H,S,512] output.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB_OUT_RM = 18;

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t tok_start = get_arg_val<uint32_t>(1);
    const uint32_t tok_count = get_arg_val<uint32_t>(2);

    constexpr uint32_t row_bytes = vDHt * 32 * 2;  // 512 * 2 = 1024
    const auto out = TensorAccessor(out_args, out_addr);

    for (uint32_t tok = tok_start; tok < tok_start + tok_count; ++tok) {
        cb_wait_front(CB_OUT_RM, vDHt);  // one untilized [32,512] block
        const uint32_t l1 = get_read_ptr(CB_OUT_RM);
        for (uint32_t h = 0; h < H; ++h) {
            // row h (head h) at L1 byte offset h*row_bytes; DRAM page = h*S + tok.
            noc_async_write(l1 + h * row_bytes, out.get_noc_addr(h * S + tok), row_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(CB_OUT_RM, vDHt);
    }
}
