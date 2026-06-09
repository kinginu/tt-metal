// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_softplus.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

template <bool legacy_compat = true>
inline void calculate_recip_first_column() {
    constexpr int ITERATIONS_HALF_FACE = 4;
    if constexpr (legacy_compat) {
        for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
            sfpi::vFloat in = sfpi::dst_reg[0];
            sfpi::vFloat out = ckernel::sfpu::_reciprocal_compat_<APPROX ? 2 : 3>(in);
            if constexpr (!(DST_ACCUM_MODE || APPROX)) {
                out = sfpi::convert<sfpi::vFloat16b>(out, sfpi::RoundMode::NearestEven);
            }
            sfpi::dst_reg[0] = out;
            sfpi::dst_reg += 2;
        }
    } else {
        for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
            sfpi::vFloat in = sfpi::dst_reg[0];
            sfpi::vFloat out;

            if constexpr (APPROX) {
                out = ckernel::sfpu::_sfpu_reciprocal_<0>(in);
            } else if constexpr (DST_ACCUM_MODE) {
                out = ckernel::sfpu::_sfpu_reciprocal_<2>(in);
            } else {
                out = ckernel::sfpu::_sfpu_reciprocal_<1>(in);
                out = sfpi::convert<sfpi::vFloat16b>(out, sfpi::RoundMode::NearestEven);
            }
            sfpi::dst_reg[0] = out;
            sfpi::dst_reg += 2;
        }
    }
}

template <bool SDPA_EXP_APPROX_MODE, uint16_t scale_bf16>
inline void calculate_exponential_first_column() {
    constexpr int ITERATIONS_HALF_FACE = 4;
    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result =
            ckernel::sfpu::_ckernel_sfpu_exp_accurate_<true /*SCALE_EN*/, DST_ACCUM_MODE /*is_fp32_dest_acc_en*/>(
                val, scale_bf16);
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg += 2;
    }
}

template <bool SDPA_EXP_APPROX_MODE>
inline void calculate_fused_max_sub_exp_add_tile(int scale_bf16) {
    constexpr int ITERATIONS_HALF_FACE = 4;
    constexpr uint32_t prev_max_base_idx = 0;
    constexpr uint32_t worker_max_base_idx = 32;
    constexpr uint32_t cur_max_base_idx = 64;
    constexpr uint32_t prev_sum_base_idx = 96;
    constexpr uint32_t worker_sum_base_idx = 128;

    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        sfpi::vFloat prev_max_vec = sfpi::dst_reg[prev_max_base_idx];
        sfpi::vFloat worker_max_vec = sfpi::dst_reg[worker_max_base_idx];
        sfpi::vFloat prev_sum_vec = sfpi::dst_reg[prev_sum_base_idx];
        sfpi::vFloat worker_sum_vec = sfpi::dst_reg[worker_sum_base_idx];
        v_if(prev_max_vec < worker_max_vec) { sfpi::dst_reg[cur_max_base_idx] = worker_max_vec; }
        v_else { sfpi::dst_reg[cur_max_base_idx] = prev_max_vec; }
        v_endif;
        sfpi::vFloat cur_max = sfpi::dst_reg[cur_max_base_idx];

        sfpi::vFloat diff_prev = prev_max_vec - cur_max;
        sfpi::vFloat diff_worker = worker_max_vec - cur_max;

        sfpi::vFloat exp_prev =
            ckernel::sfpu::_ckernel_sfpu_exp_accurate_<true /*SCALE_EN*/, DST_ACCUM_MODE /*is_fp32_dest_acc_en*/>(
                diff_prev, scale_bf16);
        sfpi::vFloat exp_worker =
            ckernel::sfpu::_ckernel_sfpu_exp_accurate_<true /*SCALE_EN*/, DST_ACCUM_MODE /*is_fp32_dest_acc_en*/>(
                diff_worker, scale_bf16);

        sfpi::dst_reg[prev_max_base_idx] = exp_prev;
        sfpi::dst_reg[worker_max_base_idx] = exp_worker;

        sfpi::dst_reg[worker_sum_base_idx] = exp_worker * worker_sum_vec;
        sfpi::dst_reg[prev_sum_base_idx] = exp_prev * prev_sum_vec;
        sfpi::vFloat corr_worker_sum = sfpi::dst_reg[worker_sum_base_idx];
        sfpi::vFloat corr_prev_sum = sfpi::dst_reg[prev_sum_base_idx];
        sfpi::dst_reg[prev_sum_base_idx] = corr_worker_sum + corr_prev_sum;
        sfpi::dst_reg += 2;
    }
}

template <bool SDPA_EXP_APPROX_MODE>
inline void calculate_softplus_first_column(uint param0, uint param1, uint param2) {
    constexpr int ITERATIONS_HALF_FACE = 4;
    float beta = ckernel::sfpu::Converter::as_float(param0);
    float beta_reciprocal = ckernel::sfpu::Converter::as_float(param1);
    float threshold = ckernel::sfpu::Converter::as_float(param2);
    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        ckernel::sfpu::calculate_softplus_body<APPROX, DST_ACCUM_MODE>(beta, beta_reciprocal, threshold);
        sfpi::dst_reg += 2;
    }
}

}  // namespace ckernel::sfpu
