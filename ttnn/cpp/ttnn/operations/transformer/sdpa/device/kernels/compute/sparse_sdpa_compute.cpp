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
#include "api/compute/bcast.h"                 // add_tiles_bcast_rows
#include <tools/profiler/kernel_profiler.hpp>  // DeviceZoneScopedN

// CB ids (must match SparseCB in sparse_sdpa_program_factory.cpp)
constexpr uint32_t CB_Q_RM = 0, CB_Q_IN = 1, CB_K_RM = 2, CB_K_IN = 3, CB_KT = 4, CB_V = 5;
constexpr uint32_t CB_MASK_RM = 6, CB_MASK = 7, CB_SCALE = 8, CB_QK_IM = 9;
constexpr uint32_t CB_MAX_A = 10, CB_MAX_B = 11, CB_SUM_A = 12, CB_SUM_B = 13;
constexpr uint32_t CB_OUT_A = 14, CB_OUT_B = 15, CB_CORR = 16, CB_OUT_IM = 17, CB_OUT_RM = 18;
constexpr uint32_t CB_CTRL = 20;  // reader -> compute: active chunk count per token

// In-place scores += mask, where the mask tile holds the per-key (-inf/0) pattern in row 0 and the
// reader leaves rows 1..31 unwritten. add_tiles_bcast_rows broadcasts mask row 0 across all 32 query
// rows, so the reader builds one row instead of replicating 32 (mirrors add_block_inplace).
ALWI void add_mask_bcast_rows_inplace(uint32_t scores_cb, uint32_t mask_cb, uint32_t num_tiles) {
    add_bcast_rows_init_short(scores_cb, mask_cb);
    cb_wait_front(scores_cb, num_tiles);
    cb_wait_front(mask_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        add_tiles_bcast_rows(scores_cb, mask_cb, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, scores_cb);
        tile_regs_release();
    }
    cb_pop_front(scores_cb, num_tiles);
    cb_pop_front(mask_cb, num_tiles);
    cb_reserve_back(scores_cb, num_tiles);
    cb_push_back(scores_cb, num_tiles);
}

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

        // Active chunk count for this token (reader skips all-sentinel chunks; >=1). Compute must loop the
        // same count or it would block on cb_wait_front for K chunks the reader never produces.
        cb_wait_front(CB_CTRL, 1);
        const uint32_t n_active = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(CB_CTRL, 0));
        cb_pop_front(CB_CTRL, 1);

        // Flash running-state CB handles (ping-pong). Reset every token; all buffers start empty.
        uint32_t max_prev = CB_MAX_A, max_cur = CB_MAX_B;
        uint32_t sum_prev = CB_SUM_A, sum_cur = CB_SUM_B;
        uint32_t out_prev = CB_OUT_A, out_cur = CB_OUT_B;

        for (uint32_t c = 0; c < n_active; ++c) {
            // --- tilize K rows -> k_tiles [Skt, DHt] (waits on reader CB_K_RM -> absorbs K-read stall) ---
            {
                DeviceZoneScopedN("cmp_tilizeK");
                compute_kernel_lib::tilize<DHt, CB_K_RM, CB_K_IN>(/*num_blocks=*/Skt, /*total_input_pages=*/Skt * 32);
            }

            // --- grid-reposition k_tiles[Skt,DHt] -> Kᵀ[DHt,Skt] and compact V[Skt,vDHt] (contents unchanged) ---
            {
                DeviceZoneScopedN("cmp_repose");
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
            }

            // Mask exists only for the last active chunk (reader builds it there only); earlier active
            // chunks are fully valid -> no mask tilize/add.
            const bool has_mask = (c == n_active - 1);

            // --- tilize mask rows -> [Sqt=1, Skt] (waits on reader CB_MASK_RM) ---
            if (has_mask) {
                DeviceZoneScopedN("cmp_tilizeMask");
                compute_kernel_lib::tilize<Skt, CB_MASK_RM, CB_MASK>(/*num_blocks=*/Sqt, /*total_input_pages=*/32);
            }

            // --- QK^T = Q @ Kᵀ -> scores [Sqt, Skt] ---
            {
                DeviceZoneScopedN("cmp_QK");
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
            }

            // --- masked softmax numerator: mask add, running max, exp ((scores-max)*scale), chunk sum ---
            {
                DeviceZoneScopedN("cmp_softmax");
                if (has_mask) {
                    add_mask_bcast_rows_inplace(CB_QK_IM, CB_MASK, Skt);  // mask add (row-broadcast; pops mask)
                }
                reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, CB_QK_IM, CB_SCALE, Sqt, VectorMode::RC>(
                    max_cur, max_prev, Skt, /*do_eltwise_max=*/c > 0);
                sub_exp_block_bcast_cols_inplace<
                    CB_QK_IM,
                    Sqt,
                    scale_fp32,
                    /*write_inplace=*/true,
                    /*do_reduce=*/false,
                    VectorMode::RC>(max_cur, sum_cur /*unused*/, Skt);
                reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, CB_QK_IM, CB_SCALE, Sqt, VectorMode::RC>(
                    sum_cur, sum_cur, Skt, /*do_eltwise_max=*/false);
            }

            // --- PV = probs @ V -> cur_out [Sqt, vDHt] ---
            {
                DeviceZoneScopedN("cmp_PV");
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
            }

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
