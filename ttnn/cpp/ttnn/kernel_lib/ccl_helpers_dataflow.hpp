// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file ccl_helpers_dataflow.hpp
 * @brief Multi-device CCL (fabric) dataflow-kernel helpers.
 *
 * The multi-device analog of the single-device dataflow-helper library (#45698,
 * @c reduce/dfb/tilize_helpers_dataflow). It gives op authors an intent-level
 * surface for the footgun-heavy fabric plumbing — connection lifecycle + direction,
 * packet-header allocation + route programming, fabric unicast/scatter writes +
 * flow control, cross-device atomic-inc semaphores, and the recv-side wait — so they
 * express "send these pages from A to B" / "signal these ring peers" instead of
 * re-deriving routing, header framing, and the handshake.
 *
 * This is PURE DATA MOVEMENT. No compute/unpack/math/pack appears here. Reduction
 * collectives (all_reduce, reduce_scatter) are out of scope.
 *
 * It WRAPS, and does not reinvent, the existing fragmented fabric layer:
 *   - @c FabricConnectionManager (connection + per-direction @c WorkerToFabricEdmSender)
 *   - @c PacketHeaderPool (header storage)
 *   - @c ccl_routing_utils (line unicast/multicast route programming)
 *   - @c perform_payload_send and friends from @c minimal_ccl_common
 *
 * Design + per-op mapping (point_to_point unicast, all_gather ring): see
 * @c CCL_DATAFLOW_HELPER_DESIGN.md at the repo root.
 *
 * @par What the helper does NOT own (the op composes it):
 *   ring slice-walk (chip_id +/- k mod ring_size), store-and-forward relay,
 *   page<->packet coalescing/segmentation, concat-by-gather_dim output addressing,
 *   split-forwarding, address generation (TensorAccessor/ShardedAddrGen is consumed,
 *   never re-wrapped), and the all_gather fuse_op/OpSignaler matmul-fusion hooks.
 */

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"

namespace dataflow_kernel_lib::ccl {

/**
 * @brief Where a FabricStreamSender stores its packet header(s) (design decision 1).
 *
 * @c Pool (default, recommended) draws from @c PacketHeaderPool — no circular buffer
 * needed, no @c cb_reserve_back / @c cb_push_back dance in the op. @c Cb takes a
 * caller-owned packet-header CB id and reserves one page per header from it (the
 * legacy point_to_point storage; offered for ops that want caller-owned header L1).
 */
enum class HeaderPolicy : uint8_t { Pool, Cb };

/**
 * @brief A single open fabric egress endpoint in one direction.
 *
 * Owns connection build + deferred open + close, direction selection, packet-header
 * allocation (lazy, per actual use), route programming (unicast + line-multicast),
 * unicast/scatter payload writes with flow control, and cross-device atomic-inc.
 *
 * @par Lifecycle (deferred-open mirrors point_to_point):
 *   1. construct  -> builds the connection (BUILD_AND_OPEN_CONNECTION_START_ONLY)
 *   2. [optional] ccl_wait_min(...) on a pre-open semaphore
 *   3. open()     -> open_finish() + select the forward/backward direction
 *   4. set_route_*(...) once (or per hop), then write_page()/write_scatter()/inc_remote()
 *   5. close()
 *
 * Headers are allocated lazily on first use: the payload header on the first
 * set_route/write, the semaphore header on the first inc_remote. A sender that only
 * signals (e.g. the receiver's "ready" inc) therefore allocates exactly one header,
 * matching the hand-written kernel's header count (avoids program-cache-hit CB
 * overflow when using the @c Cb policy).
 *
 * @note There is intentionally no symmetric "FabricStreamReceiver" class: the receive
 *   INGRESS is a local NoC read the op owns; a receiver's only fabric activity is a
 *   brief egress (its ack/ready inc) plus a wait — both already covered here and by
 *   the ccl_wait_min/ccl_sem_reset free functions. A Receiver object would over-fit
 *   point_to_point's 2-party handshake.
 *
 * @tparam header_policy Pool (default) or Cb. See HeaderPolicy.
 *
 * @par Example (point_to_point sender, abbreviated):
 * @code
 * using namespace dataflow_kernel_lib::ccl;
 * size_t conn_arg_idx = NUM_OP_RT_ARGS;                 // start of the fabric arg block
 * bool is_forward = get_arg_val<uint32_t>(conn_arg_idx); // peek has_forward (== direction here)
 * FabricStreamSender<HeaderPolicy::Cb> tx(conn_arg_idx, is_forward, packet_header_cb_id, alignment);
 *
 * ccl_wait_min(recv_sem, 1);          // wait for the receiver's "ready"
 * ccl_sem_reset(recv_sem, 0);         // reset BEFORE our own inc (cache-reuse safe)
 * tx.open();
 * tx.set_route_unicast(num_hops);
 * for (...) tx.write_page(packet_l1_addr, payload_size, packet_idx, dst);  // op owns coalescing
 * tx.inc_remote(get_noc_addr(recv_sem), 1);   // "done"
 * tx.close();
 * @endcode
 */
template <HeaderPolicy header_policy = HeaderPolicy::Pool>
class FabricStreamSender {
public:
    /**
     * @brief Build the connection (deferred open) from runtime args (Pool policy).
     * @param conn_arg_idx  Index of the fabric arg block produced by
     *        ttnn::ccl::dataflow::append_ccl_fabric_rt_args; ADVANCED past the block.
     * @param is_forward    Send on the forward (true) or backward (false) connection.
     * @param alignment     L1 alignment used to size the on-wire payload (bytes).
     */
    FORCE_INLINE FabricStreamSender(size_t& conn_arg_idx, bool is_forward, uint32_t alignment);

    /**
     * @brief Build the connection (deferred open) from runtime args (Cb policy).
     * @param header_cb_id  Caller-owned packet-header CB; one page reserved per header.
     */
    FORCE_INLINE FabricStreamSender(size_t& conn_arg_idx, bool is_forward, uint32_t header_cb_id, uint32_t alignment);

    /// Finish opening the connection and bind the forward/backward direction.
    FORCE_INLINE void open();

    /// Close the connection.
    FORCE_INLINE void close();

    /**
     * @brief Program a 1-D unicast route (distance in hops) — point_to_point.
     * Routes through ccl_routing_utils::fabric_set_line_unicast_route (its
     * LowLatencyPacketHeader branch IS the raw fabric_set_unicast_route<false>).
     * Stored and reused by write_page and inc_remote; re-call to change hops (ring).
     */
    FORCE_INLINE void set_route_unicast(uint32_t num_hops);

    /// Program a line-unicast route from a precomputed route info (all_gather).
    FORCE_INLINE void set_route_unicast(const ccl_routing_utils::line_unicast_route_info_t& route_info);

    /// Program a line-multicast route (all_gather barrier / broadcast inc).
    FORCE_INLINE void set_route_multicast(const ccl_routing_utils::line_multicast_route_info_t& route_info);

    /**
     * @brief Unicast-write `size_bytes` from local L1 `src_l1_addr` to page `page_idx`
     *        of `dst`, then push the packet over the fabric (flow-controlled).
     * @tparam AddrGen  TensorAccessor / ShardedAddrGen (consumed, not re-wrapped).
     */
    template <class AddrGen>
    FORCE_INLINE void write_page(uint32_t src_l1_addr, uint32_t size_bytes, uint32_t page_idx, const AddrGen& dst);

    /**
     * @brief Scatter-write up to 4 (page,size) chunks in one packet (all_gather).
     * Thin wrapper over minimal_ccl_common scatter writes; not used by point_to_point.
     */
    template <class AddrGen>
    FORCE_INLINE void write_scatter(
        uint32_t src_l1_addr,
        uint32_t chunk_size_bytes,
        uint32_t num_chunks,
        uint32_t first_page_idx,
        const AddrGen& dst);

    /**
     * @brief Atomic-increment a remote semaphore over the fabric (ready/done/counting).
     * Programs the stored route on a dedicated semaphore header, then flushes.
     */
    FORCE_INLINE void inc_remote(uint64_t remote_sem_noc_addr, uint32_t val = 1);

    /// Escape hatch: the raw payload header (for ops that need stateful header tricks).
    FORCE_INLINE volatile PACKET_HEADER_TYPE* payload_header();

private:
    enum class RouteKind : uint8_t { None, Unicast, Multicast };

    FORCE_INLINE volatile PACKET_HEADER_TYPE* alloc_header();
    FORCE_INLINE void program_route(volatile PACKET_HEADER_TYPE* hdr);
    FORCE_INLINE void ensure_payload_header();

    FabricConnectionManager conn_;
    tt::tt_fabric::WorkerToFabricEdmSender* dir_ = nullptr;  // bound in open()
    volatile PACKET_HEADER_TYPE* payload_hdr_ = nullptr;     // lazy: first set_route/write
    volatile PACKET_HEADER_TYPE* sem_hdr_ = nullptr;         // lazy: first inc_remote
    uint32_t alignment_ = 0;
    uint32_t header_cb_id_ = 0;  // Cb policy only
    bool is_forward_ = true;
    RouteKind route_kind_ = RouteKind::None;
    ccl_routing_utils::line_unicast_route_info_t unicast_info_{};
    ccl_routing_utils::line_multicast_route_info_t multicast_info_{};
};

// ===========================================================================
// Recv-side coordination (no ring vocabulary; threshold is a plain count).
//
//   2-party handshake (point_to_point) : threshold = 1
//   N-party barrier   (all_gather)     : threshold = ring_size - 1
//   incremental count (all_gather)     : threshold = sem_target
// ===========================================================================

/// Block until *sem >= threshold (wraps noc_semaphore_wait_min).
FORCE_INLINE void ccl_wait_min(volatile tt_l1_ptr uint32_t* sem, uint32_t threshold);

/**
 * @brief Reset a local semaphore for program-cache reuse.
 * @warning PLACEMENT IS A FOOTGUN: a SENDER must reset BEFORE its own outgoing inc on
 *   a cache hit; a RECEIVER resets after its wait, before the next reuse.
 */
FORCE_INLINE void ccl_sem_reset(volatile tt_l1_ptr uint32_t* sem, uint32_t value = 0);

/**
 * @brief Convenience: ccl_wait_min then immediately ccl_sem_reset (the common case).
 * @warning Do NOT use where a sender must interleave its own inc between the wait and
 *   the reset — call ccl_wait_min and ccl_sem_reset separately there.
 */
FORCE_INLINE void ccl_wait_min_and_reset(
    volatile tt_l1_ptr uint32_t* sem, uint32_t threshold, uint32_t reset_value = 0);

}  // namespace dataflow_kernel_lib::ccl

#include "ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.inl"
