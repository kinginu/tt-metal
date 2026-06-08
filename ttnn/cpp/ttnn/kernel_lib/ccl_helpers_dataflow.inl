// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation file for ccl_helpers_dataflow.hpp
// Do not include directly - include ccl_helpers_dataflow.hpp instead

#pragma once

namespace dataflow_kernel_lib::ccl {

// ----------------------------------------------------------------------------
// FabricStreamSender — construction / lifecycle
// ----------------------------------------------------------------------------

template <HeaderPolicy header_policy>
FORCE_INLINE FabricStreamSender<header_policy>::FabricStreamSender(
    size_t& conn_arg_idx, bool is_forward, uint32_t alignment) :
    conn_(FabricConnectionManager::build_from_args<
          FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(conn_arg_idx)),
    alignment_(alignment),
    is_forward_(is_forward) {}

template <HeaderPolicy header_policy>
FORCE_INLINE FabricStreamSender<header_policy>::FabricStreamSender(
    size_t& conn_arg_idx, bool is_forward, uint32_t header_cb_id, uint32_t alignment) :
    conn_(FabricConnectionManager::build_from_args<
          FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(conn_arg_idx)),
    alignment_(alignment),
    header_cb_id_(header_cb_id),
    is_forward_(is_forward) {}

template <HeaderPolicy header_policy>
FORCE_INLINE void FabricStreamSender<header_policy>::open() {
    conn_.open_finish();
    dir_ = is_forward_ ? &conn_.get_forward_connection() : &conn_.get_backward_connection();
}

template <HeaderPolicy header_policy>
FORCE_INLINE void FabricStreamSender<header_policy>::close() {
    conn_.close();
}

// ----------------------------------------------------------------------------
// FabricStreamSender — headers + routes (private)
// ----------------------------------------------------------------------------

template <HeaderPolicy header_policy>
FORCE_INLINE volatile PACKET_HEADER_TYPE* FabricStreamSender<header_policy>::alloc_header() {
    if constexpr (header_policy == HeaderPolicy::Pool) {
        return PacketHeaderPool::allocate_header();
    } else {
        cb_reserve_back(header_cb_id_, 1);
        auto* hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(get_write_ptr(header_cb_id_));
        cb_push_back(header_cb_id_, 1);
        return hdr;
    }
}

template <HeaderPolicy header_policy>
FORCE_INLINE void FabricStreamSender<header_policy>::program_route(volatile PACKET_HEADER_TYPE* hdr) {
    if (route_kind_ == RouteKind::Unicast) {
        ccl_routing_utils::fabric_set_line_unicast_route(hdr, unicast_info_);
    } else if (route_kind_ == RouteKind::Multicast) {
        ccl_routing_utils::fabric_set_line_multicast_route(hdr, multicast_info_);
    }
}

template <HeaderPolicy header_policy>
FORCE_INLINE void FabricStreamSender<header_policy>::ensure_payload_header() {
    if (payload_hdr_ == nullptr) {
        payload_hdr_ = alloc_header();
        program_route(payload_hdr_);
    }
}

template <HeaderPolicy header_policy>
FORCE_INLINE void FabricStreamSender<header_policy>::set_route_unicast(uint32_t num_hops) {
    route_kind_ = RouteKind::Unicast;
    unicast_info_ = ccl_routing_utils::line_unicast_route_info_t{};
    unicast_info_.dst_mesh_id = 0;
    unicast_info_.distance_in_hops = static_cast<uint16_t>(num_hops);
    if (payload_hdr_ != nullptr) {
        program_route(payload_hdr_);
    }
}

template <HeaderPolicy header_policy>
FORCE_INLINE void FabricStreamSender<header_policy>::set_route_unicast(
    const ccl_routing_utils::line_unicast_route_info_t& route_info) {
    route_kind_ = RouteKind::Unicast;
    unicast_info_ = route_info;
    if (payload_hdr_ != nullptr) {
        program_route(payload_hdr_);
    }
}

template <HeaderPolicy header_policy>
FORCE_INLINE void FabricStreamSender<header_policy>::set_route_multicast(
    const ccl_routing_utils::line_multicast_route_info_t& route_info) {
    route_kind_ = RouteKind::Multicast;
    multicast_info_ = route_info;
    if (payload_hdr_ != nullptr) {
        program_route(payload_hdr_);
    }
}

// ----------------------------------------------------------------------------
// FabricStreamSender — writes + atomic-inc
// ----------------------------------------------------------------------------

template <HeaderPolicy header_policy>
template <class AddrGen>
FORCE_INLINE void FabricStreamSender<header_policy>::write_page(
    uint32_t src_l1_addr, uint32_t size_bytes, uint32_t page_idx, const AddrGen& dst) {
    ensure_payload_header();
    // Header carries the alignment-rounded on-wire size; the payload send moves the
    // actual bytes (mirrors point_to_point's writer_send.cpp:88-90).
    tt::tt_fabric::linear::to_noc_unicast_write(align(size_bytes, alignment_), payload_hdr_, page_idx, dst);
    perform_payload_send(*dir_, src_l1_addr, size_bytes, payload_hdr_);
}

template <HeaderPolicy header_policy>
template <class AddrGen>
FORCE_INLINE void FabricStreamSender<header_policy>::write_scatter(
    uint32_t src_l1_addr, uint32_t chunk_size_bytes, uint32_t num_chunks, uint32_t first_page_idx,
    const AddrGen& dst) {
    // Reference implementation: one unicast write per chunk. all_gather's perf path
    // packs up to 4 chunks into one scatter packet via minimal_ccl_common scatter
    // writes + stateful (set_state/with_state) headers; that optimization slots in
    // here behind the same call site.
    for (uint32_t i = 0; i < num_chunks; ++i) {
        write_page(src_l1_addr + i * chunk_size_bytes, chunk_size_bytes, first_page_idx + i, dst);
    }
}

template <HeaderPolicy header_policy>
FORCE_INLINE void FabricStreamSender<header_policy>::inc_remote(uint64_t remote_sem_noc_addr, uint32_t val) {
    if (sem_hdr_ == nullptr) {
        sem_hdr_ = alloc_header();
    }
    program_route(sem_hdr_);
    sem_hdr_->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{remote_sem_noc_addr, val});
    dir_->wait_for_empty_write_slot();
    dir_->send_payload_flush_blocking_from_address((uint32_t)sem_hdr_, sizeof(PACKET_HEADER_TYPE));
}

template <HeaderPolicy header_policy>
FORCE_INLINE volatile PACKET_HEADER_TYPE* FabricStreamSender<header_policy>::payload_header() {
    ensure_payload_header();
    return payload_hdr_;
}

// ----------------------------------------------------------------------------
// Recv-side coordination
// ----------------------------------------------------------------------------

FORCE_INLINE void ccl_wait_min(volatile tt_l1_ptr uint32_t* sem, uint32_t threshold) {
    noc_semaphore_wait_min(sem, threshold);
}

FORCE_INLINE void ccl_sem_reset(volatile tt_l1_ptr uint32_t* sem, uint32_t value) { noc_semaphore_set(sem, value); }

FORCE_INLINE void ccl_wait_min_and_reset(
    volatile tt_l1_ptr uint32_t* sem, uint32_t threshold, uint32_t reset_value) {
    noc_semaphore_wait_min(sem, threshold);
    noc_semaphore_set(sem, reset_value);
}

}  // namespace dataflow_kernel_lib::ccl
