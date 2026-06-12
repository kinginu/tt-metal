// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Compile-time selector for the consolidated Quasar binary-SFPU test
// (eltwise_binary_sfpu_quasar_test.cpp). Kept dependency-free so it can be
// included at global scope in the test source, before the auto-generated
// build.h emits `constexpr QuasarBinaryOp SFPU_BINARY_OP = QuasarBinaryOp::<op>;`.
//
// The op families differ structurally on Quasar (see sfpu_binary_operations_quasar.h):
//   * *Int  — Int32 ops dispatched via _llk_math_eltwise_binary_sfpu_params_ with
//             ROW-offset operand addressing.
//   * Mul/Div — float ops via calculate_sfpu_binary, TILE-index operand addressing.
//   * Max/Min — via _llk_math_eltwise_unary_sfpu_params_ + calculate_binary_max_min
//             (in0/in1/out tile layout), needs _init_binary_max_min_ and the
//             FPU-datacopy path for non-32-bit / MX formats.
enum class QuasarBinaryOp
{
    AddInt,
    MulInt,
    GtInt,
    LtInt,
    LeInt,
    GeInt,
    Mul,
    Div,
    Max,
    Min,
};

constexpr bool quasar_binary_op_is_int(QuasarBinaryOp op)
{
    return op == QuasarBinaryOp::AddInt || op == QuasarBinaryOp::MulInt || op == QuasarBinaryOp::GtInt || op == QuasarBinaryOp::LtInt ||
           op == QuasarBinaryOp::LeInt || op == QuasarBinaryOp::GeInt;
}

constexpr bool quasar_binary_op_is_max_min(QuasarBinaryOp op)
{
    return op == QuasarBinaryOp::Max || op == QuasarBinaryOp::Min;
}
