// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device.hpp"
#include <tt-metalium/constants.hpp>
#include <bit>

using namespace tt::tt_metal;

namespace ttnn::prim {

static constexpr uint32_t K_DIM = 576;
static constexpr uint32_t V_DIM = 512;
static constexpr uint32_t TOPK_MAX = 2048;

void SparseSDPAOperation::validate_on_program_cache_miss(const SparseSDPAParams& attrs, const SparseSDPAInputs& t) {
    const auto& q = t.q;
    const auto& kv = t.kv;
    const auto& idx = t.indices;

    TT_FATAL(q.device()->arch() == tt::ARCH::BLACKHOLE, "sparse_sdpa is Blackhole-only");
    TT_FATAL(
        q.device() == kv.device() && q.device() == idx.device(), "sparse_sdpa: all inputs must be on the same device");

    // dtypes
    TT_FATAL(q.dtype() == DataType::BFLOAT16, "q must be bf16");
    TT_FATAL(kv.dtype() == DataType::BFLOAT16, "kv must be bf16");
    TT_FATAL(idx.dtype() == DataType::UINT32, "indices must be uint32");

    // layout / memory
    for (const Tensor* tp : {&q, &kv, &idx}) {
        TT_FATAL(tp->layout() == Layout::ROW_MAJOR, "sparse_sdpa inputs must be ROW_MAJOR");
        TT_FATAL(tp->memory_config().buffer_type() == BufferType::DRAM, "sparse_sdpa inputs must be in DRAM");
        TT_FATAL(!tp->memory_config().is_sharded(), "sparse_sdpa inputs must be interleaved");
        // Row-major page id math requires no last-dim padding.
        TT_FATAL(tp->padded_shape() == tp->logical_shape(), "sparse_sdpa inputs must not be padded");
    }

    const auto qs = q.logical_shape();
    const auto kvs = kv.logical_shape();
    const auto is = idx.logical_shape();
    TT_FATAL(qs.rank() == 4 && qs[0] == 1, "q must be [1,H,S,576]");
    const uint32_t H = qs[1];
    const uint32_t S = qs[2];
    TT_FATAL(H == 32, "sparse_sdpa Stage 1 supports H==32 (got {})", H);
    TT_FATAL(qs[3] == K_DIM, "q last dim must be {} (got {})", K_DIM, qs[3]);
    TT_FATAL(kvs.rank() == 4 && kvs[0] == 1 && kvs[1] == 1 && kvs[3] == K_DIM, "kv must be [1,1,T,576]");
    TT_FATAL(is.rank() == 4 && is[0] == 1 && is[1] == 1 && is[2] == S, "indices must be [1,1,S,TOPK]");
    const uint32_t T = kvs[2];
    const uint32_t TOPK = is[3];
    TT_FATAL(T > 0 && S > 0 && TOPK > 0, "S/T/TOPK must be > 0");
    TT_FATAL(TOPK <= TOPK_MAX, "TOPK must be <= {} (got {})", TOPK_MAX, TOPK);

    // k_chunk_size: multiple of 32, divides TOPK
    TT_FATAL(attrs.k_chunk_size >= 32 && attrs.k_chunk_size % 32 == 0, "k_chunk_size must be a multiple of 32");
    TT_FATAL(TOPK % attrs.k_chunk_size == 0, "k_chunk_size must divide TOPK");
    TT_FATAL(attrs.scale > 0.0f, "scale must be > 0");

    // row-byte alignment (32B)
    TT_FATAL((K_DIM * 2) % 32 == 0, "K row bytes must be 32B aligned");
    TT_FATAL((TOPK * 4) % 32 == 0, "indices row bytes must be 32B aligned");

    // streaming kt_inplace_v / dst_size=8 path requires fp32_dest_acc_en=false
    auto [mf, approx, fp32_dest_acc_en, dfs, pl1] =
        get_compute_kernel_config_args(q.device()->arch(), attrs.compute_kernel_config);
    TT_FATAL(!fp32_dest_acc_en, "sparse_sdpa requires fp32_dest_acc_en=false");
}

SparseSDPAOperation::spec_return_value_t SparseSDPAOperation::compute_output_specs(
    const SparseSDPAParams& attrs, const SparseSDPAInputs& t) {
    auto shape = t.q.logical_shape();  // [1, H, S, 576]
    shape[3] = V_DIM;                  // [1, H, S, 512]
    return TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), attrs.output_mem_config));
}

SparseSDPAOperation::tensor_return_value_t SparseSDPAOperation::create_output_tensors(
    const SparseSDPAParams& attrs, const SparseSDPAInputs& t) {
    return create_device_tensor(compute_output_specs(attrs, t), t.q.device());
}

ttsl::hash::hash_t SparseSDPAOperation::compute_program_hash(const SparseSDPAParams& attrs, const SparseSDPAInputs& t) {
    return operation::hash_operation<SparseSDPAOperation>(
        std::bit_cast<uint32_t>(attrs.scale),
        attrs.k_chunk_size,
        attrs.output_mem_config,
        t.q.logical_shape(),
        t.kv.logical_shape(),
        t.indices.logical_shape());
}

Tensor sparse_sdpa(
    const Tensor& q,
    const Tensor& kv,
    const Tensor& indices,
    float scale,
    uint32_t k_chunk_size,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    using OperationType = ttnn::prim::SparseSDPAOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .scale = scale,
            .k_chunk_size = k_chunk_size,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config,
        },
        OperationType::tensor_args_t{
            .q = q,
            .kv = kv,
            .indices = indices,
        });
}

}  // namespace ttnn::prim
