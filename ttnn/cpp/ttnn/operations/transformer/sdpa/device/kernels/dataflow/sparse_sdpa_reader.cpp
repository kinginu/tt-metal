// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa reader: gather Q rows + per-chunk indexed K rows (0xFFFFFFFF -> zero-fill) + build the
// row-major per-chunk mask; build the reduce identity scaler once. (compute_common materialized path.)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// CB ids (must match SparseCB in sparse_sdpa_program_factory.cpp)
constexpr uint32_t CB_Q_RM = 0;
constexpr uint32_t CB_K_RM = 2;
constexpr uint32_t CB_MASK_RM = 6;
constexpr uint32_t CB_SCALE = 8;
constexpr uint32_t CB_IDX = 19;

constexpr uint32_t SENTINEL = 0xFFFFFFFFu;
constexpr uint16_t NEG_INF_BF16 = 0xFF80u;
constexpr uint32_t TILE = 32;

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t T = get_compile_time_arg_val(2);
    constexpr uint32_t TOPK = get_compile_time_arg_val(3);
    constexpr uint32_t k_chunk = get_compile_time_arg_val(4);
    constexpr uint32_t n_chunks = get_compile_time_arg_val(5);
    // arg 6 = scale_packed (used by compute, not here)
    constexpr auto q_args = TensorAccessorArgs<7>();
    constexpr auto kv_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t kv_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t tok_start = get_arg_val<uint32_t>(3);
    const uint32_t tok_count = get_arg_val<uint32_t>(4);

    constexpr uint32_t row_bytes = 576 * 2;  // K/Q row
    constexpr uint32_t idx_row_bytes = TOPK * 4;
    constexpr uint32_t mask_row_bytes = k_chunk * 2;

    const auto q = TensorAccessor(q_args, q_addr);
    const auto kv = TensorAccessor(kv_args, kv_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);

    // Reduce identity scaler (value 1.0; the softmax scale is applied in compute's exp). Built once.
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        CB_SCALE,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        /*reduce_factor=*/1,
        /*compute_uses_reduce_tile=*/true>();

    // Reader-internal scratch for one token's index row.
    cb_reserve_back(CB_IDX, 1);
    const uint32_t idx_l1 = get_write_ptr(CB_IDX);
    volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);

    for (uint32_t tok = tok_start; tok < tok_start + tok_count; ++tok) {
        // Q: H head-rows -> CB_Q_RM.  Q is [1,H,S,576] row-major: row (h,tok) = page h*S+tok.
        cb_reserve_back(CB_Q_RM, H);
        const uint32_t qw = get_write_ptr(CB_Q_RM);
        for (uint32_t h = 0; h < H; ++h) {
            noc_async_read(q.get_noc_addr(h * S + tok), qw + h * row_bytes, row_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(CB_Q_RM, H);

        // indices row for this token (page = tok)
        noc_async_read(idx.get_noc_addr(tok), idx_l1, idx_row_bytes);
        noc_async_read_barrier();

        for (uint32_t c = 0; c < n_chunks; ++c) {
            // K chunk rows -> CB_K_RM (zero-fill sentinels)
            cb_reserve_back(CB_K_RM, k_chunk);
            const uint32_t kw = get_write_ptr(CB_K_RM);
            for (uint32_t j = 0; j < k_chunk; ++j) {
                const uint32_t key = idx_ptr[c * k_chunk + j];
                const uint32_t dst = kw + j * row_bytes;
                if (key == SENTINEL) {
                    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
                    for (uint32_t b = 0; b < row_bytes / 4; ++b) {
                        z[b] = 0;
                    }
                } else {
                    noc_async_read(kv.get_noc_addr(key), dst, row_bytes);
                }
            }
            noc_async_read_barrier();
            cb_push_back(CB_K_RM, k_chunk);

            // Mask: 32 row-sticks of k_chunk, col j = (sentinel ? -inf : 0). All rows identical.
            cb_reserve_back(CB_MASK_RM, TILE);
            const uint32_t mw = get_write_ptr(CB_MASK_RM);
            volatile tt_l1_ptr uint16_t* m0 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mw);
            for (uint32_t j = 0; j < k_chunk; ++j) {
                m0[j] = (idx_ptr[c * k_chunk + j] == SENTINEL) ? NEG_INF_BF16 : (uint16_t)0;
            }
            for (uint32_t r = 1; r < TILE; ++r) {
                volatile tt_l1_ptr uint16_t* mr =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mw + r * mask_row_bytes);
                for (uint32_t j = 0; j < k_chunk; ++j) {
                    mr[j] = m0[j];
                }
            }
            cb_push_back(CB_MASK_RM, TILE);
        }
    }
}
