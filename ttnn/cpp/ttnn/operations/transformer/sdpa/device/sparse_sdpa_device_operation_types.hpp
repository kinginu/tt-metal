// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <optional>

namespace ttnn::prim {

// Sparse MLA prefill (DeepSeek DSA). See PLAN_sparse_sdpa.md.
struct SparseSDPAParams {
    float scale = 1.0f;  // compile-time (folded into the program hash)
    uint32_t k_chunk_size = 128;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct SparseSDPAInputs {
    Tensor q;        // [1, H, S, 576] bf16 ROW_MAJOR
    Tensor kv;       // [1, 1, T, 576] bf16 ROW_MAJOR  (K = full 576, V = kv[..., :512])
    Tensor indices;  // [1, 1, S, TOPK] uint32 ROW_MAJOR  (0xFFFFFFFF = masked)
};

}  // namespace ttnn::prim
