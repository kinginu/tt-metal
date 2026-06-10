// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

inline void tilizeA_B_binary_init(uint32_t icb0, uint32_t icb1, uint32_t block) {
    UNPACK((llk_unpack_tilizeA_B_init<true, true>(icb0, icb1, block)));

    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWADD, BroadcastType::NONE, MathFidelity::LoFi>(
        icb0, icb1, 0 /*acc_to_dest*/)));
}

inline void add_tiles_math(uint32_t icb0, uint32_t icb1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<
        EltwiseBinaryType::ELWADD,
        BroadcastType::NONE,
        DST_ACCUM_MODE,
        MathFidelity::LoFi,
        EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true /*clear_fp32_dst_acc*/)));
}

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in0, dfb::out);
    tilizeA_B_binary_init(dfb::in0, dfb::in1, per_core_block_tile_cnt);

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        dfb_in0.wait_front(per_core_block_tile_cnt);
        dfb_in1.wait_front(per_core_block_tile_cnt);
        dfb_out.reserve_back(per_core_block_tile_cnt);
        unpack_tilizeA_B_block(dfb::in0, dfb::in1, per_core_block_tile_cnt, b);

        for (uint32_t i = 0; i < per_core_block_tile_cnt; ++i) {
            tile_regs_acquire();
            add_tiles_math(dfb::in0, dfb::in1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb::out);
            tile_regs_release();
        }

        dfb_out.push_back(per_core_block_tile_cnt);
        dfb_in0.pop_front(per_core_block_tile_cnt);
        dfb_in1.pop_front(per_core_block_tile_cnt);
    }
}
