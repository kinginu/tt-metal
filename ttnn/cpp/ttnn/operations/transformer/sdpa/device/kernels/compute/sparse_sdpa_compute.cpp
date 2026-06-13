// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa compute (compute_common materialized path, flash/online-softmax over k_chunks).
// Per token: tilize Q. Per chunk: tilize K, grid-reposition K->Kᵀ + compact V (drop rope cols),
// QK (transpose=true), mask add, running-max, exp, chunk-sum, PV; then flash-combine the chunk into
// the running (max, sum, output) with the exp(prev_max-cur_max) correction. After all chunks:
// normalize out *= 1/sum, untilize. n_chunks==1 degenerates to a plain single-chunk softmax.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "compute_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

// CB ids (must match SparseCB in sparse_sdpa_program_factory.cpp)
constexpr uint32_t CB_Q_RM = 0, CB_Q_IN = 1, CB_K_RM = 2, CB_K_IN = 3, CB_KT = 4, CB_V = 5;
constexpr uint32_t CB_MASK_RM = 6, CB_MASK = 7, CB_SCALE = 8, CB_QK_IM = 9;
constexpr uint32_t CB_MAX_A = 10, CB_MAX_B = 11, CB_SUM_A = 12, CB_SUM_B = 13;
constexpr uint32_t CB_OUT_A = 14, CB_OUT_B = 15, CB_CORR = 16, CB_OUT_IM = 17, CB_OUT_RM = 18;

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Skt = get_compile_time_arg_val(3);
    constexpr uint32_t n_chunks = get_compile_time_arg_val(4);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(5);
    constexpr uint32_t qk_sh = get_compile_time_arg_val(6);
    constexpr uint32_t qk_sw = get_compile_time_arg_val(7);
    constexpr uint32_t pv_sh = get_compile_time_arg_val(8);
    constexpr uint32_t pv_sw = get_compile_time_arg_val(9);

    constexpr uint32_t Sqt = 1;  // H=32 heads = one tile-row

    const uint32_t tok_count = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(CB_Q_RM, CB_Q_IN);

    for (uint32_t tok = 0; tok < tok_count; ++tok) {
        // --- tilize Q rows -> q_tiles [Sqt=1, DHt] ---
        compute_kernel_lib::tilize<DHt, CB_Q_RM, CB_Q_IN>(/*num_blocks=*/Sqt, /*total_input_pages=*/H);

        // Flash running-state CB handles (ping-pong). Reset every token; all buffers start empty.
        uint32_t max_prev = CB_MAX_A, max_cur = CB_MAX_B;
        uint32_t sum_prev = CB_SUM_A, sum_cur = CB_SUM_B;
        uint32_t out_prev = CB_OUT_A, out_cur = CB_OUT_B;

        for (uint32_t c = 0; c < n_chunks; ++c) {
            // --- tilize K rows -> k_tiles [Skt, DHt] ---
            compute_kernel_lib::tilize<DHt, CB_K_RM, CB_K_IN>(/*num_blocks=*/Skt, /*total_input_pages=*/Skt * 32);

            // --- grid-reposition k_tiles[Skt,DHt] -> Kᵀ[DHt,Skt] and compact V[Skt,vDHt] (contents unchanged) ---
            cb_wait_front(CB_K_IN, Skt * DHt);
            cb_reserve_back(CB_KT, DHt * Skt);
            cb_reserve_back(CB_V, Skt * vDHt);
            copy_tile_to_dst_init_short(CB_K_IN);
            for (uint32_t sk = 0; sk < Skt; ++sk) {
                for (uint32_t d = 0; d < DHt; ++d) {
                    tile_regs_acquire();
                    copy_tile(CB_K_IN, sk * DHt + d, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    // out-of-order index is honored only with <true>; bare pack_tile ignores it and
                    // packs sequentially (this was the Skt>=2 Kᵀ scramble bug).
                    pack_tile<true>(0, CB_KT, d * Skt + sk);  // Kᵀ position (d, sk)
                    if (d < vDHt) {
                        pack_tile<true>(0, CB_V, sk * vDHt + d);  // V = first vDHt cols
                    }
                    tile_regs_release();
                }
            }
            cb_push_back(CB_KT, DHt * Skt);
            cb_push_back(CB_V, Skt * vDHt);
            cb_pop_front(CB_K_IN, Skt * DHt);

            // --- tilize mask rows -> [Sqt=1, Skt] ---
            compute_kernel_lib::tilize<Skt, CB_MASK_RM, CB_MASK>(/*num_blocks=*/Sqt, /*total_input_pages=*/32);

            // --- QK^T = Q @ Kᵀ -> scores [Sqt, Skt] ---
            matmul_blocks(
                CB_Q_IN,
                CB_KT,
                CB_QK_IM,
                Sqt,
                Skt,
                DHt,
                /*num_blocks=*/1,
                /*in0_nsub=*/Sqt / qk_sh,
                /*in1_nsub=*/Skt / qk_sw,
                /*in0_block_w=*/DHt,
                qk_sh,
                qk_sw,
                /*transpose=*/true);

            // --- mask add (elementwise, pops mask) ---
            add_block_inplace<true>(CB_QK_IM, CB_MASK, Skt);

            // --- running max: cur_max = c>0 ? eltwise_max(prev_max, rowmax(scores)) : rowmax(scores) ---
            reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, CB_QK_IM, CB_SCALE, Sqt, VectorMode::RC>(
                max_cur, max_prev, Skt, /*do_eltwise_max=*/c > 0);

            // --- scores = exp((scores - cur_max) * scale)  (in-place; no fused reduce) ---
            sub_exp_block_bcast_cols_inplace<
                CB_QK_IM,
                Sqt,
                scale_fp32,
                /*write_inplace=*/true,
                /*do_reduce=*/false,
                VectorMode::RC>(max_cur, sum_cur /*unused*/, Skt);

            // --- chunk sum [Sqt,1] ---
            reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, CB_QK_IM, CB_SCALE, Sqt, VectorMode::RC>(
                sum_cur, sum_cur, Skt, /*do_eltwise_max=*/false);

            // --- PV = probs @ V -> cur_out [Sqt, vDHt] ---
            matmul_blocks(
                CB_QK_IM,
                CB_V,
                out_cur,
                Sqt,
                vDHt,
                Skt,
                /*num_blocks=*/1,
                /*in0_nsub=*/Sqt / pv_sh,
                /*in1_nsub=*/vDHt / pv_sw,
                /*in0_block_w=*/Skt,
                pv_sh,
                pv_sw,
                /*transpose=*/false);
            cb_pop_front(CB_QK_IM, Skt);  // matmul_blocks leaves in0 produced

            // --- flash combine the running state with this chunk ---
            if (c > 0) {
                // correction = exp((prev_max - cur_max) * scale)
                sub_exp_block<scale_fp32>(max_prev, max_cur, CB_CORR, Sqt);
                cb_pop_front(max_prev, Sqt);  // done with old running max
                // running sum: cur_sum += prev_sum * correction
                mul_tiles_bcast_cols_inplace(sum_prev, CB_CORR, Sqt);  // prev_sum *= correction (CORR kept)
                add_block_inplace<true>(sum_cur, sum_prev, Sqt);       // cur_sum += prev_sum (pops prev_sum)
                // running out: cur_out += prev_out * correction  (L1 accumulate; pops prev_out & CORR)
                mul_block_bcast_cols<Sqt, vDHt, /*immediate_pop=*/false, /*pack_accumulate=*/true>(
                    out_prev, CB_CORR, out_cur);
            }

            // swap prev <-> cur for next chunk
            uint32_t t;
            t = max_prev;
            max_prev = max_cur;
            max_cur = t;
            t = sum_prev;
            sum_prev = sum_cur;
            sum_cur = t;
            t = out_prev;
            out_prev = out_cur;
            out_cur = t;
        }

        cb_pop_front(CB_Q_IN, DHt);  // Q reused across all chunks; drop it now so >1 token/core stays clean

        // --- normalize: out = out_prev / sum_prev ---
        recip_block_inplace(sum_prev, Sqt);
        mul_block_bcast_cols_inplace<Sqt, vDHt>(out_prev, sum_prev);  // pops sum_prev

        // --- untilize needs a compile-time CB id; copy the swapped running out into the fixed CB ---
        copy_block(out_prev, CB_OUT_IM, Sqt * vDHt);  // pops out_prev
        cb_pop_front(max_prev, Sqt);                  // drop leftover running max so buffers are clean next token
        compute_kernel_lib::untilize<vDHt, CB_OUT_IM, CB_OUT_RM>(/*num_blocks=*/Sqt);
    }
}
