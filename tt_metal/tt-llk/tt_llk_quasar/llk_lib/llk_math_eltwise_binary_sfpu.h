// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <utility>

#include "llk_math_eltwise_sfpu_common.h"

template <bool APPROXIMATE, class F, class... ARGS>
[[deprecated("Use SFPU_BINARY_CALL, SFPU_BINARY_CALL_MODE, SFPU_BINARY_CALL_FN, or SFPU_BINARY_CALL_CAST from llk_math_eltwise_binary_sfpu_macros.h instead.")]]
inline void _llk_math_eltwise_binary_sfpu_params_(F&& sfpu_func, std::uint32_t dst_tile_index, ARGS&&... args)
{
    _llk_math_eltwise_sfpu_params_(std::forward<F>(sfpu_func), dst_tile_index, std::forward<ARGS>(args)...);
}
