// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <bit>
#include <cstdlib>
#include <map>
#include <string>

using namespace tt::tt_metal;

namespace ttnn::prim {

static constexpr uint32_t K_DIM = 576;  // DHt = 18
static constexpr uint32_t V_DIM = 512;  // vDHt = 16
static constexpr uint32_t TILE = 32;

// Fixed CB id convention shared with the kernels.
// Flash/online-softmax keeps ping-pong (prev/cur) buffers for the running max, sum, and output.
enum SparseCB : uint32_t {
    CB_Q_RM = 0,  // Q rows (row-major, reader -> compute tilize)
    CB_Q_IN,      // Q tiled [Sqt=1, DHt]
    CB_K_RM,      // K chunk rows (row-major)
    CB_K_IN,      // K tiled [Skt, DHt]
    CB_KT,        // Kᵀ [DHt, Skt] (grid-reposed for QK)
    CB_V,         // compact V [Skt, vDHt]
    CB_MASK_RM,   // mask rows (row-major: 32 x k_chunk, -inf/0)
    CB_MASK,      // mask tiled [Sqt=1, Skt]
    CB_SCALE,     // reduce identity scaler (1 tile)
    CB_QK_IM,     // scores [Sqt=1, Skt]
    CB_MAX_A,     // running max ping-pong [Sqt=1, 1]
    CB_MAX_B,
    CB_SUM_A,  // running sum ping-pong [Sqt=1, 1]
    CB_SUM_B,
    CB_OUT_A,  // running out ping-pong [Sqt=1, vDHt] (single-buffered for L1 accumulation)
    CB_OUT_B,
    CB_CORR,    // exp(prev_max - cur_max) correction [Sqt=1, 1]
    CB_OUT_IM,  // fixed pre-untilize copy of the final out [Sqt=1, vDHt]
    CB_OUT_RM,  // untilized row-major out (compute -> writer)
    CB_IDX,     // reader-internal: one token's index row (uint32)
    CB_CTRL,    // reader -> compute: active chunk count (= ceil(valid_keys / k_chunk)) per token
    CB_ZERO,    // reader-internal: one prebuilt zero K-row (NoC-copied over sentinel rows)
    CB_COUNT
};

ProgramDescriptor SparseSDPAOperation::SparseSDPAProgramFactory::create_descriptor(
    const SparseSDPAParams& attrs, const SparseSDPAInputs& t, Tensor& output) {
    ProgramDescriptor desc;

    const uint32_t H = t.q.logical_shape()[1];  // 32
    const uint32_t S = t.q.logical_shape()[2];
    const uint32_t T = t.kv.logical_shape()[2];
    const uint32_t TOPK = t.indices.logical_shape()[3];

    const uint32_t DHt = K_DIM / TILE;   // 18
    const uint32_t vDHt = V_DIM / TILE;  // 16
    const uint32_t k_chunk = attrs.k_chunk_size;
    const uint32_t Skt = k_chunk / TILE;  // tiles per chunk along keys
    const uint32_t n_chunks = TOPK / k_chunk;
    const uint32_t Sqt = 1;  // H=32 heads = one tile-row
    const uint32_t scale_packed = std::bit_cast<uint32_t>(attrs.scale);

    const uint32_t bf16 = 2;
    const uint32_t q_row_bytes = K_DIM * bf16;     // 1152
    const uint32_t kchunk_bytes = k_chunk * bf16;  // mask row bytes
    const tt::DataFormat bf = tt::DataFormat::Float16_b;
    const uint32_t tile_bytes = tt::tile_size(bf);  // 2048

    auto* device = t.q.device();
    CoreCoord grid = device->compute_with_storage_grid_size();
    auto core_grid = CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    const uint32_t num_cores = grid.x * grid.y;
    const uint32_t base = S / num_cores;
    const uint32_t extra = S % num_cores;

    // ---- CBs (fixed order = SparseCB enum) ----
    const auto cb = [&](uint32_t page_size, uint32_t num_pages, tt::DataFormat df) {
        const uint32_t idx = desc.cbs.size();
        desc.cbs.push_back(CBDescriptor{
            .total_size = page_size * num_pages,
            .core_ranges = core_grid,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(idx), .data_format = df, .page_size = page_size}}},
        });
    };
    cb(q_row_bytes, H, bf);          // CB_Q_RM : H row-sticks
    cb(tile_bytes, DHt, bf);         // CB_Q_IN : [1,DHt]
    cb(q_row_bytes, k_chunk, bf);    // CB_K_RM : k_chunk row-sticks
    cb(tile_bytes, Skt * DHt, bf);   // CB_K_IN : [Skt,DHt]
    cb(tile_bytes, DHt * Skt, bf);   // CB_KT   : [DHt,Skt]
    cb(tile_bytes, Skt * vDHt, bf);  // CB_V    : [Skt,vDHt]
    cb(kchunk_bytes, TILE, bf);      // CB_MASK_RM : 32 row-sticks of k_chunk
    cb(tile_bytes, Skt, bf);         // CB_MASK : [1,Skt]
    cb(tile_bytes, 1, bf);           // CB_SCALE
    cb(tile_bytes, Skt, bf);         // CB_QK_IM : [1,Skt]
    cb(tile_bytes, Sqt, bf);         // CB_MAX_A
    cb(tile_bytes, Sqt, bf);         // CB_MAX_B
    cb(tile_bytes, Sqt, bf);         // CB_SUM_A
    cb(tile_bytes, Sqt, bf);         // CB_SUM_B
    cb(tile_bytes, vDHt, bf);        // CB_OUT_A : [1,vDHt] (single-buffered for L1 acc)
    cb(tile_bytes, vDHt, bf);        // CB_OUT_B
    cb(tile_bytes, Sqt, bf);         // CB_CORR : [1,1]
    cb(tile_bytes, vDHt, bf);        // CB_OUT_IM : [1,vDHt] fixed pre-untilize copy
    cb(tile_bytes, vDHt, bf);        // CB_OUT_RM : untilized [32,512] block (vDHt tile-sized pages)
    cb(TOPK * 4, 1, bf);             // CB_IDX : one index row (uint32 bytes)
    cb(16, 2, bf);                   // CB_CTRL : n_active per token (uint32; 16B aligned, double-buffered)
    cb(q_row_bytes, 1, bf);          // CB_ZERO : one prebuilt zero K-row (576 bf16)

    // ---- matmul subblocks ----
    auto [qk_sh, qk_sw] = detail::determine_largest_subblock_size(Sqt, Skt, 8);
    auto [pv_sh, pv_sw] = detail::determine_largest_subblock_size(Sqt, vDHt, 8);

    // ---- compile-time args ----
    // K-gather NoC trid-ring depth (bounds outstanding reads/core to fight NoC/DRAM congestion).
    // 0 = disabled (issue-all + single barrier). Sourced from env so it can be swept via JIT recompile
    // (no host rebuild). Clamp to [0,8] (HW trid range; trids 1..N_TRIDS, 0 reserved for untagged).
    // Default depth-8 NoC trid-ring: bounds outstanding K reads/core to fight DRAM congestion. Depth 8 is
    // gentle enough to be ~neutral on sparse tokens while recovering ~9% on dense (swept empirically).
    // Env override for tuning. 0 = disabled. Clamp to [0,8] (HW trid range; trids 1..N, 0 = untagged).
    uint32_t k_trids = 8;
    if (const char* e = std::getenv("SPARSE_SDPA_K_TRIDS")) {
        const int v = std::atoi(e);
        k_trids = v < 0 ? 0 : (v > 8 ? 8 : static_cast<uint32_t>(v));
    }
    // Ring only for tokens with n_active >= ring_min_active (>= half the chunks). Sparse tokens desync
    // naturally (no congestion to recover) so they skip the ring's overhead; only dense-ish tokens keep
    // all cores synced and saturate DRAM. Default n_chunks/2 was the sweet spot across the sparsity sweep.
    uint32_t ring_min_active = n_chunks / 2 < 1 ? 1 : n_chunks / 2;
    if (const char* e = std::getenv("SPARSE_SDPA_RING_MIN_ACTIVE")) {
        const int v = std::atoi(e);
        ring_min_active = v < 1 ? 1 : static_cast<uint32_t>(v);
    }
    std::vector<uint32_t> reader_ct = {H, S, T, TOPK, k_chunk, n_chunks, scale_packed, k_trids, ring_min_active};
    TensorAccessorArgs(t.q.buffer()).append_to(reader_ct);
    TensorAccessorArgs(t.kv.buffer()).append_to(reader_ct);
    TensorAccessorArgs(t.indices.buffer()).append_to(reader_ct);

    std::vector<uint32_t> writer_ct = {H, S, vDHt};
    TensorAccessorArgs(output.buffer()).append_to(writer_ct);

    std::vector<uint32_t> compute_ct = {H, DHt, vDHt, Skt, n_chunks, scale_packed, qk_sh, qk_sw, pv_sh, pv_sw};

    // ---- kernels ----
    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/";
    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kdir + "dataflow/sparse_sdpa_reader.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_grid;
    reader_desc.compile_time_args = reader_ct;
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kdir + "dataflow/sparse_sdpa_writer.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_grid;
    writer_desc.compile_time_args = writer_ct;
    writer_desc.config = WriterConfigDescriptor{};

    auto [math_fidelity, math_approx, fp32_acc, dfs, pl1] =
        get_compute_kernel_config_args(device->arch(), attrs.compute_kernel_config);
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = kdir + "compute/sparse_sdpa_compute.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_grid;
    compute_desc.compile_time_args = compute_ct;
    compute_desc.config = ComputeConfigDescriptor{.math_fidelity = math_fidelity, .fp32_dest_acc_en = false};
    // compute_common.hpp granularity/exp defines. Granularity=1 is always-correct (divides any dim).
    std::map<std::string, std::string> cdefs{
        {"STATS_GRANULARITY", "1"},
        {"SUB_EXP_GRANULARITY", "1"},
        {"MUL_BCAST_GRANULARITY", "1"},
        {"DHT_GRANULARITY", "1"},
        {"REDUCE_GRANULARITY", "1"},
        {"EXP_APPROX_MODE", std::to_string(static_cast<int>(math_approx))},
    };
    compute_desc.defines = KernelDescriptor::Defines(cdefs.begin(), cdefs.end());

    auto* q_buf = t.q.buffer();
    auto* kv_buf = t.kv.buffer();
    auto* idx_buf = t.indices.buffer();
    auto* out_buf = output.buffer();
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid.x, i / grid.x};
        uint32_t tok_start = i * base + std::min(i, extra);
        uint32_t tok_count = base + (i < extra ? 1u : 0u);
        reader_desc.emplace_runtime_args(core, {q_buf, kv_buf, idx_buf, tok_start, tok_count});
        writer_desc.emplace_runtime_args(core, {out_buf, tok_start, tok_count});
        compute_desc.emplace_runtime_args(core, {tok_start, tok_count});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

}  // namespace ttnn::prim
