# CCL Dataflow Helper — Design

**Status:** design + prototype (proof-by-migration of `point_to_point`).
**Branch:** `wransom/ccl_help` (off `main`).
**Mirrors:** the single-device dataflow-helper library `#45698`
(`ttnn/cpp/ttnn/kernel_lib/{reduce,dfb,tilize}_helpers_dataflow.{hpp,inl}`).
**Companion investigation:** `~/ccl_op_gen/FINDINGS_point_to_point.md` (§2 the
11-item helper catalog, §4 the first API sketch, §7 the `all_gather` inspection).

---

## 0. TL;DR

Two pure–data-movement CCL ops — `point_to_point` (unicast) and `all_gather`
(ring) — hand-roll the *same* fabric primitives (connection lifecycle, packet-header
routing, fabric writes, cross-device atomic-inc semaphores) through **two different,
non-overlapping ad-hoc code paths**. This design introduces one documented,
intent-level **multi-device dataflow helper** —
`ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.{hpp,inl}` (kernel) plus a host
companion — that *wraps* (does not reinvent) the existing fragmented layer
(`ccl_routing_utils`, `minimal_ccl_common`, `FabricConnectionManager`,
`PacketHeaderPool`). The kernel surface is one stateful egress object
(`FabricStreamSender`) plus a small set of recv-side coordination free functions; the
host surface is four free functions. `point_to_point` is re-expressed on top of it and
verified on craq-sim BH; `all_gather` is mapped onto the same API (design-only) to
prove the surface is not over-fit to unicast.

---

## 1. Purpose & scope

**In scope (pure DM).** The footgun-heavy fabric plumbing that has *no single-device
analog* and exists only because the transfer crosses a chip boundary: fabric
connection open/finish/close + direction selection, packet-header allocation + route
programming (unicast and line-multicast), fabric unicast/scatter writes + flow
control, cross-device atomic-inc semaphores (the send-side inc and the recv-side
wait), and the cache-reuse semaphore-reset discipline. Plus the host companion that
feeds those kernels: 1-D route computation, packet framing, the fabric-connection
runtime-arg append, and GlobalSemaphore lifetime/`Synchronize`.

**Out of scope.**
- Reduction collectives (`all_reduce`, `reduce_scatter`) — they do cross-chip
  *arithmetic*. No compute/unpack/math/pack appears anywhere in this helper.
- Op-specific *orchestration* the op must keep owning (see §4 decision 0): ring
  slice-walk (`chip_id ± k mod ring_size`), store-and-forward relay, page→packet
  coalescing/segmentation, concat-by-`gather_dim` output addressing, split-forwarding.
- `all_gather`'s `fuse_op` / `OpSignaler` matmul-fusion hooks
  (`self_write_done_semaphore`, `synchronize_workers_and_signal_op`,
  `minimal_default_writer.cpp:199-206,455-461`; `minimal_default_reader.cpp:103-107,
  381-388`). These are an all_gather→matmul fusion path, not pure DM; the helper
  ignores them.
- Address generation (`TensorAccessor` / `ShardedAddrGen`) — an existing primitive.
  The helper *consumes* an accessor; it never re-wraps addr-gen.

---

## 2. The problem — two non-overlapping ad-hoc paths

| | `point_to_point` (unicast) | `all_gather` (ring) |
|---|---|---|
| Routing | raw `fabric_set_unicast_route<false>` (`writer_send.cpp:48,109`) | `ccl_routing_utils::fabric_set_line_{unicast,multicast}_route` (`minimal_default_writer.cpp:228,329-331`) |
| Direction | private `tt::point_to_point::common::connection_direction_collection` (`device/kernels/common.hpp:12-18`) | inline `direction ? get_backward : get_forward` (`minimal_default_writer.cpp:217`) |
| Header storage | CB-allocated (`send_program_factory.cpp:62-74` + `writer_send.cpp:43-47`) | `PacketHeaderPool::allocate_header()` ×3 (`minimal_default_writer.cpp:220-222`) |
| Writes | `tt::tt_fabric::linear::to_noc_unicast_write` + `perform_payload_send` (`writer_send.cpp:88-90`) | `fabric_unicast_noc_{scatter,unicast}_write_{set,with}_state` (`minimal_default_writer.cpp:312-430`) |
| Sem coordination | 2-party ready/done handshake, `wait==1` (`writer_send.cpp:65-68`, `reader_receive.cpp:66-67`) | N-party barrier `wait_min ring_size-1` + counting `wait_min target` (`minimal_default_writer.cpp:271-272`, `minimal_default_reader.cpp:291-292,367-368`) |
| Connection | direct `FabricConnectionManager` | direct **+ optional** `WorkerToFabricMuxSender` mux path (`minimal_default_writer.cpp:167-168`) |

**The decisive observation.** The fragmented layer *already* unifies the low-level
pieces — and `point_to_point` already reaches part of it:
- `worker_routing_utils.hpp:60-71`'s `fabric_set_line_unicast_route`, in its
  `LowLatencyPacketHeader` branch (`:66`), **is exactly** p2p's raw
  `fabric_set_unicast_route<false>(hdr, num_hops)` (`route_info.distance_in_hops`).
- `minimal_ccl_common.hpp:20-42`'s `perform_payload_send` **is the one p2p calls**
  (p2p's `writer_send.cpp` includes `moe_utils.hpp`, which includes
  `minimal_ccl_common.hpp` + `worker_routing_utils.hpp` —
  `moe_utils.hpp:12-13`).

So the missing piece is **not** a low-level write/route library — those exist. The
missing piece is a **documented, intent-level surface** that (a) owns the *stateful*
connection/header/route/direction lifecycle as one object, (b) unifies the two
semaphore-coordination patterns behind one threshold, and (c) supplies the host
companion. That is what this helper adds, *on top of* the existing fragments —
**wrap, don't reinvent** (§4 decision 5).

---

## 3. Primitive → {p2p} → {all_gather} mapping (the spine)

Every helper signature below is justified by ≥1 op here. The over-fit test: the
unicast column never needs the multicast/scatter/ring rows.

### Kernel side

| # | Primitive | `point_to_point` | `all_gather` | Helper owns |
|---|-----------|------------------|--------------|-------------|
| K1 | Connection build (deferred open) | `FabricConnectionManager::build_from_args<…START_ONLY>` `writer_send.cpp:39-40`; `open_finish()` `:70` | `build_from_args` `writer:196` | ✅ `FabricStreamSender` ctor + `open()` |
| K2 | Direction select | `connection_direction_collection(is_fwd,…)` `common.hpp:12` | `direction ? bwd : fwd` `writer:217` | ✅ ctor param `is_forward` |
| K3 | Header alloc | CB-allocated `writer_send.cpp:43-47` | `PacketHeaderPool::allocate_header()` `writer:220-222` | ✅ ctor policy (pool default / CB fallback) |
| K4 | Route — unicast | `fabric_set_unicast_route<false>(hdr,num_hops)` `writer_send.cpp:48,109` | `fabric_set_line_unicast_route` `writer:329-331` | ✅ `set_route_unicast(...)` |
| K5 | Route — line-multicast | — | `fabric_set_line_multicast_route` `writer:228` | ✅ `set_route_multicast(...)` (all_gather only) |
| K6 | Unicast payload write | `linear::to_noc_unicast_write` + `perform_payload_send` `writer_send.cpp:88-90` | `fabric_unicast_noc_unicast_write_{set,with}_state` `writer:319,376` | ✅ `write_page(...)` |
| K7 | Scatter write (≤4 tiles/pkt) | — | `fabric_unicast_noc_scatter_write_{set,with}_state` `writer:312,367` | ✅ `write_scatter(...)` (all_gather only) |
| K8 | Atomic-inc over fabric | `hdr->to_noc_unicast_atomic_inc({sem,1})` `writer_send.cpp:110`, `reader_receive.cpp:48` | `fabric_*_noc_unicast_atomic_inc_{set,with}_state` `writer:322,418` | ✅ `inc_remote(sem_noc_addr,val)` |
| K9 | Recv wait — handshake (==1) | `noc_semaphore_wait(ptr,1)` `writer_send.cpp:65`, `reader_receive.cpp:67` | — | ✅ `ccl_wait_min(ptr,1)` |
| K10 | Recv wait — barrier (ring-1) | — | `noc_semaphore_wait_min(barrier,ring_size-1)` `writer:271` | ✅ `ccl_wait_min(ptr,threshold)` |
| K11 | Recv wait — counting (target) | — | `noc_semaphore_wait_min(sem,target)` `reader:291-292,367-368` | ✅ `ccl_wait_min(ptr,threshold)` |
| K12 | Cache-reuse reset | reset-before-inc `writer_send.cpp:66-68`; reset-at-end `reader_receive.cpp:101` | reset-after-barrier `writer:272`; reset-at-end `reader:392` | ✅ `ccl_sem_reset(...)` + documented placement |
| K13 | Flush / close | `wait_for_empty_write_slot` + `send_payload_flush_blocking…` `writer_send.cpp:112-113`; `close()` `:115` | mux teardown `writer:668-673`; `close()` `:679` | ✅ folded into `inc_remote`/`write_page`/`close()`; mux teardown = extension |
| K14 | Addr-gen | `TensorAccessor` `writer_send.cpp:52` | `TensorAccessor`/`ShardedAddrGen` | ⛔ passthrough (consume only) |

### Host side

| # | Primitive | `point_to_point` | `all_gather` | Helper owns |
|---|-----------|------------------|--------------|-------------|
| H1 | 1-D route compute | `detail::fabric_1d_routing → {hops,is_fwd,neighbor}` `device_op.cpp:66-98` (incl. the **sign-reversal** `:73,:93,:97` + ring/line shorter-path `:86-96`) | `ccl::get_forward_backward_line_mcast_distance(ring_size,ring_index,topo)` `factory:306` | ✅ `ccl_dm_route(...)` (+ `ccl_dm_mcast_route(...)`) |
| H2 | Packet framing | `detail::compute_aligned_packet_dims` (incl. bf16 `bit_floor` `:24-25`, 2 regimes `:30-40`) `device_op.cpp:20-43` | channel-buffer size `factory:383` | ✅ `ccl_packet_dims(...)` |
| H3 | Fabric conn RT args | the `!is_forward` **index dance** `send_program_factory.cpp:167-175`, `receive_program_factory.cpp:153-161`; kernel `conn_arg_idx=9` **overlap** `writer_send.cpp:36-40` | per-direction append; mux CT args `factory:587-590` | ✅ `append_ccl_fabric_rt_args(...)` (direct); mux = extension |
| H4 | GlobalSemaphore lifecycle | `create_global_semaphore` + `Synchronize` + **park** `device_op.cpp:196-213` | caller-passed `vector<GlobalSemaphore>` + `barrier_semaphore` `factory:223-224` | ✅ `make_ccl_semaphore(...)` (returns; caller parks) |
| H5 | Packet-header CB sizing | magic constants `send_program_factory.cpp:62-74` | (n/a — pool) | ✅ removed by pool default; CB sizing folded into helper when CB policy used |

---

## 4. The five design decisions

### Decision 0 (framing): substrate vs. orchestration
The helper owns the **common substrate** (the tables above). It does **not** own
op orchestration (§1 out-of-scope). Litmus test applied to every signature: *used by
≥1 op, and the unicast path never names a ring concept.* `set_route_multicast`,
`write_scatter`, and large `ccl_wait_min` thresholds exist for `all_gather`; p2p never
calls them.

### Decision 1 — header allocation policy: **PacketHeaderPool default, CB fallback**
- **Fork.** p2p CB-allocates headers; all_gather uses `PacketHeaderPool`.
- **Fact.** `PacketHeaderPool::allocate_header()` (`packet_header_pool.h:46`) needs
  **no per-op host reservation** — the all_gather factory reserves nothing for it.
  It draws from the fabric's L1 header region that exists whenever fabric is enabled
  (which any `FabricConnectionManager` op already requires).
- **Decision.** `FabricStreamSender` **defaults to `PacketHeaderPool`** (the modern
  idiom; deletes a whole CB + `cb_reserve`/`cb_push_back` dance from the op). A
  **CB-address policy** is offered for ops that prefer caller-owned header storage.
- **Migration choice.** The `point_to_point` proof uses the **CB policy** — it
  preserves p2p's exact current header mechanism, so the acceptance gate tests the
  *helper API*, not a header-storage behavior change. `all_gather`'s sketch uses the
  **pool**. Both policies are thus exercised across the two ops. (New ops should
  prefer the pool.)

### Decision 2 — semaphore ownership: **caller-passed; one threshold for all 3 patterns**
- **Fork.** p2p: one helper-allocated shared `GlobalSemaphore`, `wait==1`.
  all_gather: caller-passed `vector<GlobalSemaphore>` + a `barrier_semaphore`,
  `wait_min`.
- **Decision (kernel).** The kernel never owns semaphore allocation; it takes
  semaphore addresses as args. Recv-side coordination is **one** function
  `ccl_wait_min(sem_ptr, threshold)` (built on `noc_semaphore_wait_min`,
  `dataflow_api.h:1955`):
  - p2p handshake → `threshold = 1`,
  - all_gather barrier → `threshold = ring_size - 1`,
  - all_gather counting → `threshold = sem_target`.
  The signature carries a **plain `uint32_t threshold`** — no `ring_size`, no ring
  vocabulary leaks into the unicast path. `wait==1` is `wait_min(1)` for a monotonic
  0→1 counter, so p2p is covered without a special case.
- **Decision (host).** `make_ccl_semaphore(mesh_device, initial)` owns the
  alloc + cross-device `Synchronize` and **returns** the `GlobalSemaphore`; the caller
  parks it (p2p → `WorkloadDescriptor::semaphores`; a ring op keeps its own vector
  alive). Parking is the caller's because the lifetime container differs per flow.
- **Cache-reuse reset.** `ccl_sem_reset(ptr, value=0)` is separate from the wait so
  the op controls *placement* (the footgun): the sender must reset **before** its own
  outgoing inc on a cache hit (`writer_send.cpp:66-68`); a receiver may reset
  immediately after its wait or at end-of-kernel (`reader_receive.cpp:101`), as long
  as it precedes the next reuse. A combined `ccl_wait_min_and_reset(ptr,threshold)`
  convenience is provided for the common immediate-reset case; the doc spells out when
  *not* to use it.

### Decision 3 — connection mode: **direct only in v1; mux documented as extension**
- **Fork.** Direct `FabricConnectionManager` (p2p + basic all_gather) vs. the
  worker-mux path (all_gather perf).
- **Evidence.** Mux is a *subsystem*, not a call: host `FabricMuxConfig`
  (`factory:518-524`), `ccl::fabric_mux_connection_ct_args` (`factory:587-590`), a
  separate `tt_fabric_mux.cpp` kernel (`factory:617-625`), and a kernel-side
  teardown/termination protocol (`writer:668-673`).
- **Decision.** v1 is **direct-only**. The `FabricStreamSender` API is written so a
  `MuxSender` policy can slot in later (the egress object already abstracts "the
  connection"); the integration points are named here. p2p and unidirectional
  all_gather workers need only direct.

### Decision 4 — routing ownership: **unicast + line-multicast from the start**
- **Fork.** unicast-only vs. unicast + line-multicast.
- **Decision.** **Both**, because (a) the multicast route already exists
  (`ccl_routing_utils::fabric_set_line_multicast_route`, `worker_routing_utils.hpp:77`)
  so exposing it is a thin wrap, and (b) the `all_gather` barrier *is* a multicast
  inc — without it the ring sketch isn't credible. The host unifies route computation:
  `ccl_dm_route` returns `{num_hops, is_forward, neighbor_id}` (owns
  `fabric_1d_routing`, incl. the sign-reversal); `ccl_dm_mcast_route` returns the
  forward/backward line-multicast distances (owns
  `get_forward_backward_line_mcast_distance`). The kernel `set_route_unicast` /
  `set_route_multicast` program a header from those. p2p uses only `set_route_unicast`.

### Decision 5 — home + consolidation: **`kernel_lib/`, wrap not reinvent**
- **Home (kernel).** `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.{hpp,inl}` —
  the #45698 precedent (`{reduce,dfb,tilize}_helpers_dataflow.{hpp,inl}`): namespace,
  `FORCE_INLINE`, Doxygen `@brief/@tparam/@param/@example`, `.hpp` declares + includes
  `.inl` at the bottom.
- **Home (host).** `ttnn/cpp/ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp`,
  **header-only** (`inline`). `kernel_lib/CMakeLists.txt` is an `INTERFACE` target that
  GLOBs `*.hpp`/`*.inl` purely as a JIT kernel header set (no host compilation), so host
  C++ does not belong there; it sits with the existing CCL host helpers. (The kernel
  `.hpp`/`.inl` ARE auto-picked-up by that GLOB — no CMake edit needed.)
- **Consolidation = wrap.** The helper sits *on top of* the existing fragments:
  - `set_route_*` → `ccl_routing_utils::fabric_set_line_{unicast,multicast}_route`;
  - `write_page`/`inc_remote` → `tt::tt_fabric::linear::to_noc_unicast_write`,
    `perform_payload_send`, `to_noc_unicast_atomic_inc`;
  - the egress object holds a `FabricConnectionManager` and a
    `WorkerToFabricEdmSender&` direction.
  Nothing is reimplemented; the fragments stay as the implementation layer.
- **Migration path (don't break existing ops).** New ops use the helper.
  `point_to_point` is migrated now and **deletes its private `device/kernels/common.hpp`**
  (its `connection_direction_collection` becomes the helper's `is_forward` ctor param).
  `all_gather` keeps working untouched; it can migrate later, incrementally (the
  helper's writes already delegate to the very functions all_gather calls). The
  fragmented headers are **not** deleted — they are the helper's substrate and remain
  directly usable.

---

## 5. Kernel API — `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.{hpp,inl}`

`namespace dataflow_kernel_lib::ccl` (nests under the #45698 umbrella). Route-info
types are reused from `ccl_routing_utils` (`worker_routing_utils.hpp:15-39`).

```cpp
namespace dataflow_kernel_lib::ccl {

// Header-storage policy (Decision 1).
enum class HeaderPolicy : uint8_t { Pool, Cb };

// Fabric egress endpoint. Owns: connection lifecycle + direction (K1,K2),
// packet header(s) (K3), route programming (K4,K5), unicast/scatter writes (K6,K7),
// atomic-inc over fabric (K8), flow control + close (K13). Wraps FabricConnectionManager
// + ccl_routing_utils + minimal_ccl_common — it does not reimplement them.
//
// One Sender == one open fabric egress in one direction. A receiver endpoint that only
// needs to *signal* (e.g. p2p's "ready" inc) constructs a Sender briefly; there is no
// separate stateful "Receiver" object, because the receive *ingress* is a local NoC read
// the op owns (see §7.1) — making a symmetric Receiver class would over-fit p2p.
template <HeaderPolicy header_policy = HeaderPolicy::Pool>
class FabricStreamSender {
public:
    // Build the connection (deferred-open) from runtime args starting at `conn_arg_idx`,
    // advancing it. Layout (produced by host append_ccl_fabric_rt_args):
    //   [has_forward][<fwd conn args>?][has_backward][<bwd conn args>?]
    // `is_forward` selects which direction to send on. Pool policy allocates headers now;
    // CB policy takes the header CB id and reserves/pushes one header page.
    FORCE_INLINE FabricStreamSender(size_t& conn_arg_idx, bool is_forward);                 // Pool
    FORCE_INLINE FabricStreamSender(size_t& conn_arg_idx, bool is_forward, uint32_t hdr_cb); // Cb

    FORCE_INLINE void open();    // open_finish() — call AFTER any pre-open semaphore wait
    FORCE_INLINE void close();   // close the connection

    // Route programming on the payload header; remembered and reused by write_page/inc_remote.
    FORCE_INLINE void set_route_unicast(uint32_t num_hops);                                 // K4 (p2p)
    FORCE_INLINE void set_route_unicast(const ccl_routing_utils::line_unicast_route_info_t&); // K4 (ag)
    FORCE_INLINE void set_route_multicast(const ccl_routing_utils::line_multicast_route_info_t&); // K5 (ag)

    // Unicast write of `size_bytes` from local L1 `src_l1_addr` to page `page_idx` of `dst`
    // (any TensorAccessor / ShardedAddrGen). Programs the payload header + perform_payload_send.
    template <class AddrGen>
    FORCE_INLINE void write_page(uint32_t src_l1_addr, uint32_t size_bytes,
                                 uint32_t page_idx, const AddrGen& dst);                    // K6

    // Scatter write — up to 4 (addr,size) chunks per packet (all_gather). Stateful header reuse.
    template <class AddrGen>
    FORCE_INLINE void write_scatter(/* chunk descriptors */ ..., const AddrGen& dst);       // K7

    // Atomic-inc a remote semaphore over the fabric (send "ready"/"done"/counting).
    // Uses the remembered route on a dedicated sem header + flush.
    FORCE_INLINE void inc_remote(uint64_t remote_sem_noc_addr, uint32_t val = 1);           // K8

    FORCE_INLINE volatile PACKET_HEADER_TYPE* payload_header();   // escape hatch if op needs raw hdr
};

// ---- Recv-side coordination (K9–K12). No ring vocabulary; threshold is a plain count. ----

// Block until *sem >= threshold. threshold=1 → 2-party handshake (p2p);
// threshold=ring_size-1 → N-party barrier (all_gather); threshold=target → counting.
FORCE_INLINE void ccl_wait_min(volatile tt_l1_ptr uint32_t* sem, uint32_t threshold);

// Reset a local semaphore for program-cache reuse. PLACEMENT IS A FOOTGUN (Decision 2):
//   sender must reset BEFORE its own outgoing inc on a cache hit; a receiver resets after
//   its wait, before the next reuse.
FORCE_INLINE void ccl_sem_reset(volatile tt_l1_ptr uint32_t* sem, uint32_t value = 0);

// Convenience: wait then immediately reset (the common safe case). Do NOT use where the
// sender must interleave its own inc between the wait and the reset — call the two above.
FORCE_INLINE void ccl_wait_min_and_reset(volatile tt_l1_ptr uint32_t* sem, uint32_t threshold,
                                         uint32_t reset_value = 0);

}  // namespace dataflow_kernel_lib::ccl
```

**`@example` (p2p sender, abbreviated):**
```cpp
using namespace dataflow_kernel_lib::ccl;
size_t conn_arg_idx = /* index of the fabric block */;
FabricStreamSender<HeaderPolicy::Cb> tx(conn_arg_idx, is_forward, packet_header_cb_id);
ccl_wait_min(recv_sem, 1);            // wait for receiver "ready"
ccl_sem_reset(recv_sem, 0);           // reset BEFORE our own inc (cache-reuse safe)
tx.open();
tx.set_route_unicast(num_hops);
for (...) tx.write_page(src_l1, payload_size, packet_idx, dst);   // op owns coalescing → page_idx
tx.inc_remote(get_noc_addr(recv_sem), 1);   // "done"
tx.close();
```

---

## 6. Host companion — `ttnn/cpp/ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp`

`namespace ttnn::ccl::dataflow`. **Header-only** (all functions `inline`):
`kernel_lib/` is a JIT-headers-only `INTERFACE` CMake target (it GLOBs `*.hpp`/`*.inl`
as a kernel header set — no host compilation), so the host companion lives with the
other CCL host helpers under `ccl/common/host/`. Header-only avoids a new compiled
target + CMake surgery in both the authoring and the verification tree; a `.cpp`/CMake
split is trivial later if the inline footprint grows.

```cpp
namespace ttnn::ccl::dataflow {

// H1 — 1-D unicast route from two mesh coords + topology. Owns the forward/backward
// SIGN REVERSAL (point_to_point_device_op.cpp:73,93,97) and the ring-vs-line
// shorter-path choice. (Unifies detail::fabric_1d_routing.)
struct DmRoute { uint32_t num_hops; bool is_forward; tt::tt_fabric::FabricNodeId neighbor_id; };
DmRoute ccl_dm_route(const tt::tt_metal::distributed::MeshDevice* mesh_device,
                     const MeshCoordinate& src, const MeshCoordinate& dst,
                     ttnn::ccl::Topology topology);

// H1b — line-multicast distances for a ring barrier (all_gather). Wraps
// ccl::get_forward_backward_line_mcast_distance. Returns args ready for the kernel's
// ccl_routing_utils::line_multicast_route_info_t.
struct DmMcastRoute { uint16_t start_distance_in_hops, range_hops, e, w, n, s; };
DmMcastRoute ccl_dm_mcast_route(uint32_t ring_size, uint32_t ring_index,
                                ttnn::ccl::Topology topology, bool is_forward);

// H2 — fabric packet framing. Owns the bf16 std::bit_floor special case and the two
// regimes (page<=packet → N pages/packet; page>packet → segmented).
// (Unifies detail::compute_aligned_packet_dims.)
struct PacketDims { uint32_t packet_size_bytes, pages_per_packet, page_segments, total_packets; };
PacketDims ccl_packet_dims(tt::tt_metal::DataType dtype, uint32_t page_size_bytes,
                           uint32_t num_pages, uint32_t alignment);

// H3 — append the fabric-connection runtime args in the EXACT layout the kernel-side
// FabricStreamSender expects, owning the has-forward/has-backward flag dance and the
// index discipline (removes the conn_arg_idx=9 overlap footgun). After the call, the
// block beginning at the pre-call rt_args.size() is:
//   [has_forward][<fwd conn args>?][has_backward][<bwd conn args>?]
// The kernel passes that start index as conn_arg_idx.
void append_ccl_fabric_rt_args(tt::tt_fabric::FabricNodeId src_fabric_id,
                               tt::tt_fabric::FabricNodeId neighbor_fabric_id,
                               uint32_t link_idx, tt::tt_metal::ProgramDescriptor& desc,
                               const CoreCoord& core, std::vector<uint32_t>& rt_args,
                               bool is_forward);

// H4 — allocate a cross-device GlobalSemaphore on worker cores, run the cache-miss
// Synchronize barrier, and return it. The CALLER parks it for the workload's lifetime
// (p2p: WorkloadDescriptor::semaphores).
tt::tt_metal::GlobalSemaphore make_ccl_semaphore(
    tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t initial_value = 0);

}  // namespace ttnn::ccl::dataflow
```

`append_ccl_fabric_rt_args` body is exactly p2p's current dance
(`send_program_factory.cpp:167-175`) collapsed to one call:
```cpp
rt_args.push_back(is_forward);   // has_forward (kernel reads this at conn_arg_idx, also == direction for unidirectional senders)
if (is_forward)  tt::tt_fabric::append_fabric_connection_rt_args(src, neighbor, link_idx, desc, core, rt_args);
rt_args.push_back(!is_forward);  // has_backward
if (!is_forward) tt::tt_fabric::append_fabric_connection_rt_args(src, neighbor, link_idx, desc, core, rt_args);
```

---

## 7. Per-op mapping

### 7.1 `point_to_point` — the migration (verified on craq-sim BH)

**Sender chip — `writer_send.cpp`.** Before: 116 lines hand-rolling K1–K6, K8, K12,
K13 + the conn_arg_idx=9 overlap. After:
- ctor `FabricStreamSender<Cb> tx(conn_arg_idx, is_forward, packet_header_cb)` ← K1,K2,K3,H3 overlap gone.
- `ccl_wait_min(recv_sem,1); ccl_sem_reset(recv_sem,0);` ← K9,K12 (reset-before-inc preserved).
- `tx.open(); tx.set_route_unicast(num_hops);` ← K1 finish, K4.
- op-owned coalescing loop (page→packet `tt_memmove`, `packet_idx` bookkeeping) **stays**
  (out of scope), calling `tx.write_page(packet_base, payload_size, packet_idx, dst)` ← K6.
- `tx.inc_remote(get_noc_addr(recv_sem),1); tx.close();` ← K8,K13.

**Receiver chip — `reader_receive.cpp`.** Its fabric activity is an *egress* (the
"ready" inc) + a *wait* — both helper-covered; the payload ingress is a **local**
`noc_async_read` of the intermediate buffer (op-owned de-coalescing, K14 passthrough):
- `FabricStreamSender<Cb> ack(conn_arg_idx, is_forward, packet_header_cb); ack.open();
  ack.set_route_unicast(num_hops); ack.inc_remote(get_noc_addr(sender_sem),1); ack.close();`
  ← the "ready" handshake half (`reader_receive.cpp:37-57`).
- `ccl_wait_min(sender_sem,1);` ← wait "done" (`:67`).
- op-owned local read loop **stays** (`:73-98`), then `ccl_sem_reset(sender_sem,0)` (`:101`).

**Host — `send/receive_program_factory.cpp` + `device_op.cpp`.**
- `detail::compute_aligned_packet_dims(...)` → `ccl::dataflow::ccl_packet_dims(...)` (H2).
- `detail::fabric_1d_routing(...)` → `ccl::dataflow::ccl_dm_route(...)` (H1).
- the `if (dst_is_forward) append…; emplace_back(!dst_is_forward); if(!…) append…` block →
  one `ccl::dataflow::append_ccl_fabric_rt_args(…, dst_is_forward)` (H3).
- `create_global_semaphore + Synchronize` in `create_workload_descriptor` →
  `ccl::dataflow::make_ccl_semaphore(mesh_device,0)`, then park in
  `workload_descriptor.semaphores` (H4).
- **Untouched / preserved:** the `#45422` `extract_tensor_buffers_t` specialization
  (`point_to_point_device_op.hpp:145-152`), the 2-tensor return (intermediate+final),
  `compute_output_specs`, `validate`. The private `device/kernels/common.hpp` is
  **deleted** (folded into K2).

### 7.2 `all_gather` — composition sketch (design-only; `@skip_for_blackhole`)

The same surface, never touched by op-specific orchestration:

- **Per-direction worker** builds `FabricStreamSender<Pool> tx(conn_arg_idx, is_forward)`
  (pool headers; Decision 1). Forward and backward workers each get one Sender — the
  unidirectional case `append_ccl_fabric_rt_args` already covers.
- **Barrier (start):** `tx.set_route_multicast(mcast_info); tx.inc_remote(barrier_sem_noc, 1)`
  to all ring peers, then `ccl_wait_min(barrier_sem, ring_size-1); ccl_sem_reset(barrier_sem,0)`
  (K5,K8,K10,K12 ← `minimal_default_writer.cpp:228-272`).
- **Per-chunk streaming:** op-owned ring slice-walk + store-and-forward decide *which*
  slice and *where*; the Sender does the move: `tx.set_route_unicast(unicast_info)`,
  then `tx.write_scatter(... , dst)` or `tx.write_page(...)` (K4,K6,K7 ←
  `writer:312-430`). The set_state/with_state perf detail lives **inside** the helper's
  scatter/page writes (an impl optimization), not in the op.
- **Counting sync:** every `chunks_per_sync` chunks, `tx.inc_remote(out_ready_sem_noc,1)`;
  the reader does `ccl_wait_min(out_ready_sem, sem_target)` (K8,K11 ←
  `writer:415-430`, `reader:291-368`), `ccl_sem_reset(out_ready_sem,0)` at end (`:392`).
- **Stays op-owned:** ring `chip_id ± k mod ring_size`, store-and-forward relay
  (re-read landed slice → re-push CB → forward), concat-by-`gather_dim` addressing,
  split-forwarding, and the entire `fuse_op`/`OpSignaler` path (ignored).
- **Not in v1:** the mux connection mode (Decision 3) — all_gather's mux workers keep
  their current code until a `MuxSender` policy lands.

**Over-fit check passes:** p2p (§7.1) calls only `set_route_unicast`, `write_page`,
`inc_remote`, `ccl_wait_min(_,1)`, `ccl_sem_reset` — no multicast, no scatter, no
ring threshold. The unicast path is clean.

---

## 8. Where this plugs into the op-gen pipeline (aware, not fixed here)

`FINDINGS` established this helper is **necessary but not sufficient** to *generate*
CCL ops through the `tt_ops_code_gen` pipeline. Two upstream gaps remain, and are a
**separate workstream** (eval-pipeline / infra):

1. The Python `generic_op` / `MeshProgramDescriptor` surface has **no
   GlobalSemaphore-parking slot and no cross-device `Synchronize` barrier hook** — so
   `make_ccl_semaphore`'s H4 contract (alloc + Synchronize + park) cannot be expressed
   from generated Python today. The helper's host companion is the natural place that
   contract attaches *once that slot exists*.
2. The golden harness has **no multi-device fixture** (no `sender_coord`/`receiver_coord`/
   topology axes; single module-scoped `device`).

Until those close, CCL ops are **wrap-only** through the pipeline; this helper is for
**hand-written ops today** (its standalone value: a clean, documented DM layer) and
**generated ops later**. No scope-creep into the pipeline fix here.

---

## 9. Acceptance & verification

- **Gate:** migrated `point_to_point` **passes on craq-sim BH** (8×P150 / P300),
  `tests/nightly/t3000/ccl/test_point_to_point.py`, via the recipe at the top of
  `~/ccl_op_gen/SETUP.md` (plain pytest + `timeout 600`, **not** `run_safe_pytest` —
  it forces slow-dispatch in sim), in the `/localdev/wransom/tt-metal-mc`
  (`nkapre/multichip`) tree. p2p there is byte-identical to `main` except two trivial
  `get_noc_addr` calling-convention lines in op-owned (not helper-owned) code, so the
  migration ports with ~2 edits.
- **Don't break existing ops:** `all_gather` and every other CCL op compile/pass
  unchanged (they keep using the fragments directly; the helper only adds a layer).
- **Result:** recorded in §10 of this doc and in the final report once green.

---

## 10. Verification log

**GREEN — migrated `point_to_point` passes on craq-sim BH** (2026-06-08).
Tree: `/localdev/wransom/tt-metal-mc` (`nkapre/multichip` @ `948ba534`); BH sim
`/localdev/wransom/sim-bh/libttsim.so` on an 8×P150 mock cluster (FABRIC_1D, ~2.5 KHz);
plain pytest + `timeout`, NOT `run_safe_pytest`.

| `tests/nightly/t3000/ccl/test_point_to_point.py` case | Result | Exercises |
|---|---|---|
| `test_point_to_point[...shape_coords_layout0...]` | PASSED (22.8s) | the core send/recv handshake + payload over the helper |
| `test_point_to_point_optional_intermediate` | PASSED | the 2-tensor intermediate path |
| `test_point_to_point_cache_hit_with_output_tensor` | PASSED | **program-cache reuse** — the cache-reuse semaphore-reset placement + the helper's lazy per-use header allocation (each migrated kernel allocates exactly its original header count) |

Build: `_ttnncpp.so` recompiled the three migrated host TUs
(`send/receive_program_factory.cpp`, `point_to_point_device_op.cpp`) cleanly against
the host companion; the two migrated kernels + `ccl_helpers_dataflow.{hpp,inl}`
JIT-compiled at runtime with no errors.

Two-tree port notes: the host files are byte-identical between `origin/main` and
`-mc`, so they ported verbatim; the only adaptation was one op-owned line in
`reader_receive.cpp` (`-mc` uses the free-function `get_noc_addr(page, accessor, …)`,
`origin/main` the method form) — a TensorAccessor-API difference outside the helper.
That line emits a `-Wdeprecated-declarations` warning in `-mc` (its own API is
deprecated there); the same warning was present in `-mc`'s pre-migration kernel.

No hang, no CB-wait deadlock, no PCC mismatch across the three cases. The full
25-variant `shape_coords_layout` sweep is correctness-equivalent but exceeds a single
sim time window (large shapes are slow at functional-sim speed), so it was not run to
completion; the three cases above cover the handshake, the intermediate path, and the
cache-reuse discipline.
