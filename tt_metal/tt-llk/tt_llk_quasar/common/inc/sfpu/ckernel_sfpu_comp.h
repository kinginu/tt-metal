// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{
// fp16b bit patterns (upper 16 bits of the corresponding fp32), loaded as
// SFPLOADI immediates in MOD0_FLOATB mode.
constexpr std::uint32_t FP16B_ONE  = 0x3F80; // 1.0f
constexpr std::uint32_t FP16B_ZERO = 0x0000; // 0.0f

// imm12_math[11]: makes SFPSETCC read the source as FP32/SMAG32 (sign-magnitude)
// rather than INT32 two's-complement, so the zero/non-zero tests check float magnitude.
constexpr std::uint32_t SFPSETCC_IMM_FP32 = 0x800;

// Compare-to-zero: load x, default the result lane to 0.0, predicate the lanes that
// satisfy COMP_MODE, and predicated-load 1.0 into them. gez/lez OR-combine a sign
// test with the magnitude zero test so ±0 lands on the IEEE-consistent side.
template <SfpuType COMP_MODE>
inline void _calculate_zero_comp_sfp_rows_()
{
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */); // load x from dest
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ZERO);                             // result lane defaults to 0.0
    TTI_SFPENCC(1 /* imm12 */, 2 /* mod1 */);                                                        // enable CC, all lanes active

    if constexpr (COMP_MODE == SfpuType::equal_zero)
    {
        TTI_SFPSETCC(SFPSETCC_IMM_FP32, p_sfpu::LREG0, sfpi::SFPSETCC_MOD1_LREG_EQ0); // |x| == 0 (covers ±0)
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ONE);
    }
    else if constexpr (COMP_MODE == SfpuType::not_equal_zero)
    {
        TTI_SFPSETCC(SFPSETCC_IMM_FP32, p_sfpu::LREG0, sfpi::SFPSETCC_MOD1_LREG_NE0); // |x| != 0
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ONE);
    }
    else if constexpr (COMP_MODE == SfpuType::less_than_zero)
    {
        TTI_SFPSETCC(SFPSETCC_IMM_FP32, p_sfpu::LREG0, sfpi::SFPSETCC_MOD1_LREG_LT0); // sign bit set (negative)
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ONE);
    }
    else if constexpr (COMP_MODE == SfpuType::greater_than_zero)
    {
        TTI_SFPSETCC(SFPSETCC_IMM_FP32, p_sfpu::LREG0, sfpi::SFPSETCC_MOD1_LREG_GTE0); // sign bit clear (positive)
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ONE);
    }
    else if constexpr (COMP_MODE == SfpuType::greater_than_equal_zero)
    {
        // gez = positive OR zero; no single mode covers non-negative, so OR-combine.
        TTI_SFPSETCC(SFPSETCC_IMM_FP32, p_sfpu::LREG0, sfpi::SFPSETCC_MOD1_LREG_GTE0); // positive
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ONE);
        TTI_SFPENCC(1 /* imm12 */, 2 /* mod1 */);                                     // re-enable CC for all lanes
        TTI_SFPSETCC(SFPSETCC_IMM_FP32, p_sfpu::LREG0, sfpi::SFPSETCC_MOD1_LREG_EQ0); // |x| == 0 (covers -0.0)
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ONE);
    }
    else if constexpr (COMP_MODE == SfpuType::less_than_equal_zero)
    {
        // lez = negative OR zero; OR-combine sign and magnitude tests.
        TTI_SFPSETCC(SFPSETCC_IMM_FP32, p_sfpu::LREG0, sfpi::SFPSETCC_MOD1_LREG_LT0); // negative
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ONE);
        TTI_SFPENCC(1 /* imm12 */, 2 /* mod1 */);                                     // re-enable CC for all lanes
        TTI_SFPSETCC(SFPSETCC_IMM_FP32, p_sfpu::LREG0, sfpi::SFPSETCC_MOD1_LREG_EQ0); // |x| == 0 (covers +0.0)
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, FP16B_ONE);
    }

    TTI_SFPENCC(0 /* imm12 */, 2 /* mod1 */);                                                         // reset CC, all lanes active
    TTI_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */); // store boolean result
}

// APPROXIMATION_MODE is unused (no approx path); retained for dispatcher signature symmetry.
template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_zero_comp_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        _calculate_zero_comp_sfp_rows_<COMP_MODE>();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // dest_reg++ (increments by 2 rows)
    }
}

} // namespace sfpu
} // namespace ckernel
