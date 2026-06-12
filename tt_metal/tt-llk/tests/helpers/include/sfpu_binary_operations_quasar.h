// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu_binary_op_quasar.h"

// Quasar binary-SFPU op headers. Quasar exposes per-op `_calculate_*_` /
// `calculate_*` entry points rather than the metal SFPU_BINARY_CALL macros the
// Blackhole `sfpu_operations.h` dispatches through, so this is the Quasar
// equivalent of that binary dispatcher.
//
// To add a new Quasar binary SFPU op:
// 1. Include its ckernel header below.
// 2. Add the enumerator to QuasarBinaryOp (sfpu_binary_op_quasar.h).
// 3. Add the `if constexpr` branch in call_binary_sfpu_operation_quasar()
//    (and init_binary_sfpu_operation_quasar() if it needs an init step).
#include "llk_sfpu/ckernel_sfpu_binary.h"         // calculate_sfpu_binary / sfpu_binary_init (float mul/div)
#include "llk_sfpu/ckernel_sfpu_binary_max_min.h" // calculate_binary_max_min / _init_binary_max_min_
#include "sfpu/ckernel_sfpu_add.h"                // _add_int_ (int add)
#include "sfpu/ckernel_sfpu_binary_comp.h"        // calculate_binary_comp_int32 (int gt/lt/le/ge)
#include "sfpu/ckernel_sfpu_mul_int32.h"          // _mul_int32_ (int mul)

namespace test_utils
{
using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

/**
 * @brief Run the per-operation init step for a Quasar binary SFPU op.
 *
 * Mul/Div program the sfpi reciprocal constant via `sfpu_binary_init`; Max/Min
 * program ADDR_MOD_6 via `_init_binary_max_min_`; the integer ops need no init.
 *
 * @tparam OP The binary op (compile-time `QuasarBinaryOp` constant).
 * @note Pair with @ref call_binary_sfpu_operation_quasar for the calculate step.
 */
template <QuasarBinaryOp OP>
void init_binary_sfpu_operation_quasar()
{
    if constexpr (OP == QuasarBinaryOp::Mul)
    {
        sfpu_binary_init<false /*APPROX*/, BinaryOp::MUL>();
    }
    else if constexpr (OP == QuasarBinaryOp::Div)
    {
        sfpu_binary_init<false /*APPROX*/, BinaryOp::DIV>();
    }
    else if constexpr (quasar_binary_op_is_max_min(OP))
    {
        _init_binary_max_min_();
    }
    // Integer add/mul/comparison ops are stateless — no init.
}

/**
 * @brief Apply a Quasar binary SFPU op over two Dest operands into a result tile.
 *
 * Selects the matching ckernel entry point at compile time from the kernel's
 * `SFPU_BINARY_OP` constant. Operand addressing differs by family (faithful to
 * the standalone tests this consolidates):
 *   - integer ops take ROW offsets (tile index * NUM_FACES * FACE_R_DIM);
 *   - float mul/div take TILE indices;
 *   - max/min go through the unary-params wrapper with a base Dest index plus
 *     in0/in1/out tile offsets, and pick the Int32 vs Float32 path at runtime.
 *
 * @tparam OP The binary op (compile-time `QuasarBinaryOp` constant).
 * @tparam is_fp32_dest_acc_en Whether Dest is in FP32 mode.
 * @param base_dst_index Base Dest tile index (used by max/min's unary-params call).
 * @param src0_tile,src1_tile,dst_tile Operand / result tile indices.
 * @param math_format Dest math format (selects max/min's Int32 vs Float32 path).
 * @note Must be preceded by @ref init_binary_sfpu_operation_quasar for the same op.
 */
template <QuasarBinaryOp OP, bool is_fp32_dest_acc_en>
void call_binary_sfpu_operation_quasar(std::uint32_t base_dst_index, int src0_tile, int src1_tile, int dst_tile, DataFormat math_format)
{
    if constexpr (quasar_binary_op_is_int(OP))
    {
        // Integer ops address Dest by row offset (tile index * rows-per-tile).
        constexpr int stride = NUM_FACES * FACE_R_DIM;
        const int in0        = src0_tile * stride;
        const int in1        = src1_tile * stride;
        const int out        = dst_tile * stride;
        if constexpr (OP == QuasarBinaryOp::MulInt)
        {
            _llk_math_eltwise_binary_sfpu_params_(_mul_int32_<false, 8>, in0, in1, out);
        }
        else if constexpr (OP == QuasarBinaryOp::GtInt)
        {
            _llk_math_eltwise_binary_sfpu_params_(calculate_binary_comp_int32<false, 8, SfpuType::gt>, in0, in1, out);
        }
        else if constexpr (OP == QuasarBinaryOp::LtInt)
        {
            _llk_math_eltwise_binary_sfpu_params_(calculate_binary_comp_int32<false, 8, SfpuType::lt>, in0, in1, out);
        }
        else if constexpr (OP == QuasarBinaryOp::LeInt)
        {
            _llk_math_eltwise_binary_sfpu_params_(calculate_binary_comp_int32<false, 8, SfpuType::le>, in0, in1, out);
        }
        else if constexpr (OP == QuasarBinaryOp::GeInt)
        {
            _llk_math_eltwise_binary_sfpu_params_(calculate_binary_comp_int32<false, 8, SfpuType::ge>, in0, in1, out);
        }
        else // AddInt
        {
            _llk_math_eltwise_binary_sfpu_params_(_add_int_<false, 8, DataFormat::Int32, 0, false>, in0, in1, out);
        }
    }
    else if constexpr (OP == QuasarBinaryOp::Mul || OP == QuasarBinaryOp::Div)
    {
        constexpr BinaryOp BINOP = (OP == QuasarBinaryOp::Mul) ? BinaryOp::MUL : BinaryOp::DIV;
        _llk_math_eltwise_binary_sfpu_params_(
            calculate_sfpu_binary<false /*APPROX*/, BINOP, is_fp32_dest_acc_en, SFPU_ITERATIONS>, src0_tile, src1_tile, dst_tile);
    }
    else if constexpr (quasar_binary_op_is_max_min(OP))
    {
        constexpr bool IS_MAX = (OP == QuasarBinaryOp::Max);
        // All integer formats route through the Int32 path; float / MX use Float32.
        if (math_format == DataFormat::Int32)
        {
            _llk_math_eltwise_unary_sfpu_params_(
                ckernel::sfpu::calculate_binary_max_min<DataFormat::Int32, IS_MAX, 8>,
                base_dst_index,
                VectorMode::RC,
                static_cast<std::uint32_t>(src0_tile),
                static_cast<std::uint32_t>(src1_tile),
                static_cast<std::uint32_t>(dst_tile));
        }
        else
        {
            _llk_math_eltwise_unary_sfpu_params_(
                ckernel::sfpu::calculate_binary_max_min<DataFormat::Float32, IS_MAX, 8>,
                base_dst_index,
                VectorMode::RC,
                static_cast<std::uint32_t>(src0_tile),
                static_cast<std::uint32_t>(src1_tile),
                static_cast<std::uint32_t>(dst_tile));
        }
    }
}

} // namespace test_utils
