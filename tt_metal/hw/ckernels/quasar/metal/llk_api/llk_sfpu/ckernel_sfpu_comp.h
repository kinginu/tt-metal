// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {
// fp16b bit patterns (upper 16 bits of the corresponding fp32), loaded as
// SFPLOADI immediates in MOD0_FLOATB mode.
constexpr std::uint32_t FP16B_ONE = 0x3F80;   // 1.0f
constexpr std::uint32_t FP16B_ZERO = 0x0000;  // 0.0f

// imm12_math[11]: makes SFPSETCC read the source as FP32/SMAG32 (sign-magnitude) rather than
// two's-complement INT32. SFPLOAD only ever produces FP32, SMAG32, or UINT32 in the LREG (signed
// integers load as sign-magnitude SMAG32, never 2's-complement), so this bit is set for every
// format we handle — both float and integer. The sign tests (SFPSETCC modes 0/4) read the sign
// bit regardless of this bit; it only governs the zero test, where SMAG32==0 correctly treats
// sign-magnitude ±0 as zero (2's-complement would miss -0 = 0x80000000).
constexpr std::uint32_t SFPSETCC_IMM_FP32 = 0x800;

/**
 * @brief Whether FMT is read/written as an integer (vs float) — drives the 1/0 result encoding.
 *
 * @tparam FMT: Math-side DataFormat (Int32 / Int16, both signed).
 */
template <DataFormat FMT>
inline constexpr bool _zero_comp_is_int_() {
    return FMT == DataFormat::Int32 || FMT == DataFormat::Int16;
}

/**
 * @brief SFPLOAD/SFPSTORE sfpmem mode for FMT.
 *
 * Integer formats select their explicit mode so the load/store reads/writes the right width:
 * Int32 → INT32 (sign-magnitude), Int16 → INT16 (sign-magnitude SMAG16). Float formats use
 * DEFAULT, letting the HW resolve fp16/bf16/fp32 from the format config.
 *
 * @note UInt16 is intentionally NOT supported. The SFPU comp kernel is format-agnostic at the
 *       store/load level (changing the sfpmem mode has no effect on the result), but the current
 *       Quasar emulator's UInt16 register-file-format datapath (unpack-to-dest / pack-from-dest)
 *       is broken: a UInt16 tile comes back corrupted (0/1 results read as 0x400 = 1<<10) even
 *       though the bit-identical Int16/SMAG16 path passes. This is an emulator deviation outside
 *       the SFPU kernel's control (file against the SFPU/emulator owner; pre-silicon). Int16
 *       covers the signed 16-bit case; revisit UInt16 once the emulator format path is fixed.
 *
 * @tparam FMT: Math-side DataFormat.
 */
template <DataFormat FMT>
inline constexpr std::uint32_t _zero_comp_sfpmem_mode_() {
    if constexpr (FMT == DataFormat::Int32) {
        return p_sfpu::sfpmem::INT32;
    } else if constexpr (FMT == DataFormat::Int16) {
        return p_sfpu::sfpmem::INT16;
    } else {
        return p_sfpu::sfpmem::DEFAULT;
    }
}

/**
 * @brief Number of instructions in the recorded replay body for a comparison mode.
 *
 * Format-independent (Int32 and float bodies have identical instruction counts): eqz/nez are 7,
 * the strict ltz/gtz add an AND-combine SFPSETCC (8), and gez/lez OR-combine two tests (10).
 *
 * @tparam COMP_MODE: Comparison-to-zero mode selecting the body to record.
 * @return Recorded body length in instructions.
 */
template <SfpuType COMP_MODE>
inline constexpr std::uint32_t _zero_comp_replay_len_() {
    if constexpr (COMP_MODE == SfpuType::greater_than_equal_zero || COMP_MODE == SfpuType::less_than_equal_zero) {
        return 10;
    } else if constexpr (COMP_MODE == SfpuType::less_than_zero || COMP_MODE == SfpuType::greater_than_zero) {
        return 8;
    }
    return 7;
}

/**
 * @brief Program ADDR_MOD_6 (dest.incr=2) so the replayed SFPSTORE advances the dest counter.
 *
 * Quasar's shared SFPU init only programs ADDR_MOD_7 (incr=0); this is additive and leaves
 * ADDR_MOD_7 in place for the body's SFPLOAD.
 *
 * @note Call once after @ref _llk_math_eltwise_sfpu_init_ and before @ref _calculate_zero_comp_.
 */
inline void _init_zero_comp_() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6, csr_read<CSR::TRISC_ID>());
}

/**
 * @brief Predicate (SFPSETCC) the lanes of LREG0 satisfying SETCC_MOD1.
 *
 * Reads LREG0 as FP32/SMAG32 (see @ref SFPSETCC_IMM_FP32) — correct for both float and the
 * sign-magnitude integers SFPLOAD produces.
 *
 * @tparam SETCC_MOD1: SFPSETCC mod1 predicate, e.g. sfpi::SFPSETCC_MOD1_LREG_EQ0.
 */
template <std::uint32_t SETCC_MOD1>
inline __attribute__((always_inline)) void _zero_comp_setcc_() {
    TTI_SFPSETCC(SFPSETCC_IMM_FP32, p_sfpu::LREG0, SETCC_MOD1);
}

/**
 * @brief Load the boolean result (0 or 1) into the result register LREG1 in FMT's encoding.
 *
 * Integer formats use SFPLOADI SHORT (INT16), which sign-extends the immediate into the full LREG
 * (1 -> 0x0000_0001) for a clean result at any store width. SHORT is used rather than USHORT
 * (UINT16): USHORT left-shifts the immediate by 10 inside the LREG (see ckernel_sfpu_fill.h).
 *
 * @tparam FMT: Math-side DataFormat; integer → integer 0/1, float → fp16b 0.0/1.0.
 * @tparam VALUE: false → 0, true → 1.
 */
template <DataFormat FMT, bool VALUE>
inline __attribute__((always_inline)) void _zero_comp_loadi_bool_() {
    if constexpr (_zero_comp_is_int_<FMT>()) {
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_SHORT, VALUE ? 1 : 0);
    } else {
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, VALUE ? FP16B_ONE : FP16B_ZERO);
    }
}

/**
 * @brief Emit the per-mode predication that loads 1 into the lanes satisfying COMP_MODE.
 *
 * The SFPSETCC sequence is format-independent; FMT only selects the result encoding (via
 * @ref _zero_comp_loadi_bool_). Successive SFPSETCC calls AND-combine, so the strict ltz/gtz add
 * an NE0 test (excludes ±0); gez/lez OR-combine a sign test with the equal_zero test (covers the
 * opposite-signed zero), reusing equal_zero.
 *
 * @tparam FMT: Math-side DataFormat.
 * @tparam COMP_MODE: Comparison-to-zero mode, values =
 *         <equal_zero/not_equal_zero/less_than_zero/greater_than_zero/greater_than_equal_zero/less_than_equal_zero>
 */
template <DataFormat FMT, SfpuType COMP_MODE>
struct zero_comp_fill;

template <DataFormat FMT>
struct zero_comp_fill<FMT, SfpuType::equal_zero> {
    static inline __attribute__((always_inline)) void apply() {
        _zero_comp_setcc_<sfpi::SFPSETCC_MOD1_LREG_EQ0>();  // == 0
        _zero_comp_loadi_bool_<FMT, true>();
    }
};

template <DataFormat FMT>
struct zero_comp_fill<FMT, SfpuType::not_equal_zero> {
    static inline __attribute__((always_inline)) void apply() {
        _zero_comp_setcc_<sfpi::SFPSETCC_MOD1_LREG_NE0>();  // != 0
        _zero_comp_loadi_bool_<FMT, true>();
    }
};

template <DataFormat FMT>
struct zero_comp_fill<FMT, SfpuType::less_than_zero> {
    static inline __attribute__((always_inline)) void apply() {
        _zero_comp_setcc_<sfpi::SFPSETCC_MOD1_LREG_LT0>();  // negative (sign set, incl -0)
        _zero_comp_setcc_<sfpi::SFPSETCC_MOD1_LREG_NE0>();  // AND nonzero -> strictly < 0
        _zero_comp_loadi_bool_<FMT, true>();
    }
};

template <DataFormat FMT>
struct zero_comp_fill<FMT, SfpuType::greater_than_zero> {
    static inline __attribute__((always_inline)) void apply() {
        _zero_comp_setcc_<sfpi::SFPSETCC_MOD1_LREG_GTE0>();  // positive (sign clear, incl +0)
        _zero_comp_setcc_<sfpi::SFPSETCC_MOD1_LREG_NE0>();   // AND nonzero -> strictly > 0
        _zero_comp_loadi_bool_<FMT, true>();
    }
};

template <DataFormat FMT>
struct zero_comp_fill<FMT, SfpuType::greater_than_equal_zero> {
    static inline __attribute__((always_inline)) void apply() {
        _zero_comp_setcc_<sfpi::SFPSETCC_MOD1_LREG_GTE0>();  // positive
        _zero_comp_loadi_bool_<FMT, true>();
        TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, sfpi::SFPENCC_MOD1_EI_RI);  // re-enable CC for the OR-combine
        zero_comp_fill<FMT, SfpuType::equal_zero>::apply();               // OR == 0 (covers -0.0 for float)
    }
};

template <DataFormat FMT>
struct zero_comp_fill<FMT, SfpuType::less_than_equal_zero> {
    static inline __attribute__((always_inline)) void apply() {
        _zero_comp_setcc_<sfpi::SFPSETCC_MOD1_LREG_LT0>();  // negative
        _zero_comp_loadi_bool_<FMT, true>();
        TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, sfpi::SFPENCC_MOD1_EI_RI);  // re-enable CC for the OR-combine
        zero_comp_fill<FMT, SfpuType::equal_zero>::apply();               // OR == 0 (covers +0.0 for float)
    }
};

/**
 * @brief Compute the comparison-to-zero boolean (1/0) for one SFP-row pair.
 *
 * Load x, default the result lane to 0, predicate the lanes that satisfy COMP_MODE (see
 * @ref zero_comp_fill), then store 1/0 unconditionally. FMT selects the sfpmem mode, the
 * SFPSETCC interpretation, and the 1/0 encoding. The SFPSTORE uses ADDR_MOD_6 (dest.incr=2) so
 * each replay advances the dest counter by one SFP-row pair while the load/store offsets stay
 * constant, letting the recorded instructions re-issue unchanged across iterations.
 *
 * @tparam FMT: Math-side DataFormat (Int32 or a float format).
 * @tparam COMP_MODE: Comparison-to-zero mode, values =
 *         <equal_zero/not_equal_zero/less_than_zero/greater_than_zero/greater_than_equal_zero/less_than_equal_zero>
 */
template <DataFormat FMT, SfpuType COMP_MODE>
inline __attribute__((always_inline)) void _zero_comp_body_() {
    constexpr std::uint32_t sfpmem = _zero_comp_sfpmem_mode_<FMT>();

    TTI_SFPLOAD(p_sfpu::LREG0, sfpmem, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */);  // load x from dest
    _zero_comp_loadi_bool_<FMT, false>();                                            // result lane defaults to 0
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, sfpi::SFPENCC_MOD1_EI_RI);  // enable CC + result=1: all lanes active

    zero_comp_fill<FMT, COMP_MODE>::apply();

    TTI_SFPENCC(sfpi::SFPENCC_IMM12_NEITHER, sfpi::SFPENCC_MOD1_EI_RI);  // disable CC, all lanes unconditional
    TTI_SFPSTORE(p_sfpu::LREG1, sfpmem, ADDR_MOD_6, 0 /* done */, 0 /* dest_reg */);  // store result; dest += 2 rows
}

/**
 * @brief Element-wise comparison-to-zero over a tile, written as 1/0 booleans.
 *
 * Records the per-row-pair body once into replay slots 0..len-1 (NoExec: the record pass does
 * not run the SFPU), then replays it ITERATIONS times. ADDR_MOD_6's dest.incr=2 on the SFPSTORE
 * advances the dest counter per replay, so the loop processes one SFP-row pair per iteration.
 *
 * @tparam APPROXIMATION_MODE: Unused (no approx path); retained for dispatcher signature symmetry.
 * @tparam FMT: Math-side DataFormat (Int32 or a float format).
 * @tparam COMP_MODE: Comparison-to-zero mode, values =
 *         <equal_zero/not_equal_zero/less_than_zero/greater_than_zero/greater_than_equal_zero/less_than_equal_zero>
 * @tparam ITERATIONS: Number of SFP-row pairs to process (8 for a 32×16 face).
 * @note Requires @ref _init_zero_comp_ to have programmed ADDR_MOD_6 (dest.incr=2).
 */
template <bool APPROXIMATION_MODE, DataFormat FMT, SfpuType COMP_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_zero_comp_() {
    constexpr std::uint32_t replay_len = _zero_comp_replay_len_<COMP_MODE>();

    // Record the per-row-pair body once into replay slots 0..replay_len-1 (NoExec: the
    // record pass does not run the SFPU); each replay re-issues it while ADDR_MOD_6
    // advances the dest counter, so the loop processes one SFP-row pair per iteration.
    lltt::record(0, replay_len);
    _zero_comp_body_<FMT, COMP_MODE>();

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        lltt::replay(0, replay_len);
    }
}

}  // namespace sfpu
}  // namespace ckernel
