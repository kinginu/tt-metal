// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// toy_matmul compute kernel — runs on EVERY core.
//
// Stripped-to-the-bone mirror of bmm_large_block_zm.cpp: one boot-time
// mm_block_init() followed by a single matmul_block<...> call with
// InitMode::None. SubblockMajor output layout (paired with the subblock-aware
// writer). No bias, no L1 packer accumulation, no activation, no untilize.

#include <cstdint>

#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

void kernel_main() {
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(2);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(4);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(5);

    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t cb_intermed0 = get_named_compile_time_arg_val("cb_intermed0");

    CircularBuffer in0_buf(cb_in0);
    CircularBuffer in1_buf(cb_in1);
    CircularBuffer out_buf(cb_out);
    CircularBuffer interm_buf(cb_intermed0);

    // Boot-time matmul init — the single hw_configure-bearing call. The helper
    // is then invoked with InitMode::None so it reuses LLK state.
    mm_block_init(cb_in0, cb_in1, cb_intermed0, /*transpose=*/false, out_subblock_w, out_subblock_h, in0_block_w);

    compute_kernel_lib::matmul_block<
        /*transpose=*/false,
        /*packer_l1_acc=*/false,
        compute_kernel_lib::LastBlockTarget::Out,
        compute_kernel_lib::OutputCBLayout::SubblockMajor,
        compute_kernel_lib::matmul_config::InitMode::None>(
        in0_buf,
        in1_buf,
        out_buf,
        interm_buf,
        compute_kernel_lib::MatmulBlockShape::of(
            in0_num_subblocks,
            in1_num_subblocks,
            out_subblock_h,
            out_subblock_w,
            in0_block_w,
            num_k_blocks,
            /*batch=*/1));
}
