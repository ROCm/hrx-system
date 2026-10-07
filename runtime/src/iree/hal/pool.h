// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_POOL_H_
#define IREE_HAL_POOL_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/hal/atomic.h"
#include "iree/hal/buffer.h"
#include "iree/hal/memory_scope.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_notification_t iree_async_notification_t;
typedef struct iree_async_frontier_tracker_t iree_async_frontier_tracker_t;

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// Result of a pool reservation acquisition operation. This is a result enum,
// NOT an iree_status_t, because acquiring reservations is a hot-path operation
// that regularly returns EXHAUSTED or OVER_BUDGET during normal operation
// (pool growth, steady-state pipelining where releases lag reservations by one
// epoch).
// Using iree_make_status() for these cases would capture backtraces on every
// transient exhaustion; expensive and misleading.
//
// The iree_status_t return from acquiring reservations is reserved for
// infrastructure failures: invalid arguments, internal corruption, slab
// provider errors. Those are exceptional and warrant backtraces.
typedef uint32_t iree_hal_pool_acquire_result_t;
enum iree_hal_pool_acquire_result_e {
  // No reservation result has been produced. Zero-initialized per-request
  // information uses this value for entries that did not participate in a
  // committed transaction.
  IREE_HAL_POOL_ACQUIRE_NONE = 0,

  // Block reserved successfully. The death frontier from the recycled block
  // was dominated by the requester's frontier or proved complete by an epoch
  // query; zero-sync reuse. No additional synchronization is required beyond
  // this requester's dependencies.
  IREE_HAL_POOL_ACQUIRE_OK = 1,

  // Block reserved from previously unused offset space (first use of this
  // region, or the block was freed with a NULL frontier). No death frontier
  // to check. Equivalent to OK for callers; no synchronization needed.
  IREE_HAL_POOL_ACQUIRE_OK_FRESH = 2,

  // Block reserved, but the death frontier was NOT dominated by the
  // requester's frontier. The block IS reserved (offset assigned), but the
  // queue scheduler must add a hidden wait on the death frontier's axes before
  // the reservation's bytes are used. The block's death frontier is returned
  // via the corresponding |out_infos| entry from
  // iree_hal_pool_acquire_reservations(), and generic
  // metadata about that dependency is returned in the same entry.
  //
  // This occurs when try-before-fence fails: prior work on this block may
  // still be executing, and the requester's frontier does not transitively
  // cover it. Queue implementations decide how to represent that hidden wait
  // in their own scheduler state; user-facing queue_alloca APIs must not
  // surface this as a transient caller-visible branch.
  IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT = 3,

  // No blocks available. All blocks are allocated, or all free blocks have
  // non-dominated death frontiers and the pool chose not to hand one out
  // as NEEDS_WAIT.
  //
  // The caller should observe the pool's notification, retry the reservation,
  // and wait on the observed token only if the pool remains exhausted. No
  // reservation was made; |out_reservations| is not modified.
  IREE_HAL_POOL_ACQUIRE_EXHAUSTED = 4,

  // The pool's budget limit would be exceeded by this reservation. The
  // request is valid and blocks may be physically available, but the budget
  // policy prevents the allocation.
  //
  // The caller should free other reservations from this pool, adjust the
  // budget, or use a different pool. No reservation was made;
  // |out_reservations| is not modified.
  IREE_HAL_POOL_ACQUIRE_OVER_BUDGET = 5,
};

// A reservation from a pool. Returned by
// iree_hal_pool_acquire_reservations() and passed to
// iree_hal_pool_release_reservations().
//
// This is a pure value type (24 bytes, no duplicated ownership). It lives on
// the stack during queue submission or is stored in the buffer that wraps it.
// The offset and byte length describe the user-visible range that may be
// materialized as a HAL buffer. Concrete pools may reserve additional backing
// bytes for alignment, block-granularity allocation, guard regions, sanitizer
// redzones, or other provider-specific metadata; those bytes are owned by the
// pool and must not be inferred from this public value.
typedef struct iree_hal_pool_reservation_t {
  // Offset of the user-visible range within the pool's managed range.
  iree_device_size_t offset;

  // User-visible byte length of the reservation. May exceed the requested size
  // due to alignment rounding or because a concrete pool exposes an entire
  // unsplit block, but it does not include hidden backing bytes such as
  // sanitizer redzones.
  iree_device_size_t byte_length;

  // Pool-internal opaque handle for returning the block on release.
  // Interpretation is strategy-specific: a TLSF release-node pointer,
  // fixed-block block index, pass-through reservation-state pointer, etc.
  // 64-bit to accommodate pointer-sized handles on all platforms.
  uint64_t block_handle;
} iree_hal_pool_reservation_t;

// Describes one allocation in a pool reservation transaction.
typedef struct iree_hal_pool_reservation_request_t {
  // Requested memory type, access, usage, placement, and alignment.
  iree_hal_buffer_params_t params;

  // Minimum number of user-visible bytes required.
  iree_device_size_t allocation_size;
} iree_hal_pool_reservation_request_t;

// Flags controlling pool reservation acquisition.
typedef uint32_t iree_hal_pool_reserve_flags_t;
enum iree_hal_pool_reserve_flag_bits_e {
  IREE_HAL_POOL_RESERVE_FLAG_NONE = 0u,

  // Allows the pool to return IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT when a
  // recycled block is available but its death frontier is not dominated by the
  // requester frontier. Callers setting this flag must either insert an
  // internal dependency on the corresponding wait frontier before the bytes
  // are used or release the reservation with that frontier to preserve the
  // block's dependency metadata.
  //
  // Callers that cannot model queue-owned hidden memory dependencies must omit
  // this flag. Such calls should receive only immediately-usable reservations
  // or transient EXHAUSTED/OVER_BUDGET results from well-behaved pools.
  IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER = 1u << 0,

  // Prevents acquiring additional backing storage or allocating host metadata
  // during this reservation attempt. This includes temporary transaction
  // staging, even for a pool whose backing capacity is fixed. Pools that need
  // such preparation return IREE_HAL_POOL_ACQUIRE_EXHAUSTED with
  // IREE_HAL_POOL_ACQUIRE_FLAG_GROWTH_REQUIRED. The caller may retry with this
  // flag cleared on its allocation path.
  //
  // Queue implementations use this inside critical sections so unbounded
  // host and platform memory allocation is routed through an explicit cold
  // path.
  IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH = 1u << 1,
};

// Generic metadata flags returned by a pool reservation acquisition.
typedef uint32_t iree_hal_pool_acquire_flags_t;
enum iree_hal_pool_acquire_flag_bits_e {
  IREE_HAL_POOL_ACQUIRE_FLAG_NONE = 0u,

  // The returned wait frontier is tainted: at least one writer freed the block
  // with no death frontier or an invalid frontier, so zero-sync reuse was
  // intentionally disabled for safety. Queue implementations should treat this
  // as a conservative dependency edge, not proof of precise happens-before.
  IREE_HAL_POOL_ACQUIRE_FLAG_WAIT_FRONTIER_TAINTED = 1u << 0,

  // The pool did not make a reservation because the caller prohibited growth
  // and the request requires additional backing storage or host metadata.
  IREE_HAL_POOL_ACQUIRE_FLAG_GROWTH_REQUIRED = 1u << 1,
};

// Generic metadata returned by a pool reservation acquisition.
//
// |reuse_frontier| preserves the selected range's prior-use prerequisite
// independently of whether this requester needs to wait. An OK reservation may
// carry a frontier already covered by the requester or known to be complete.
// Its presence alone does not require a wait; |result| determines that.
//
// A caller returning an unused reservation, including after materialization
// failure, passes |reuse_frontier| to iree_hal_pool_release_reservations(). A
// pool subdividing the reservation preserves this prerequisite for its unused
// ranges: one requester's eligibility does not establish global completion.
// Concrete pools tolerate |death_frontier| aliasing the reservation's own
// frontier storage when returning an unused reservation.
typedef struct iree_hal_pool_acquire_info_t {
  // Exact retained prior-use prerequisite, or NULL when there is none.
  // Borrowed immutable storage remains valid until this reservation is
  // released. OK_NEEDS_WAIT always has a nonempty frontier; OK may also have
  // one. Fresh and unsuccessful acquisitions have no reuse prerequisite.
  const iree_async_frontier_t* reuse_frontier;

  // Generic metadata bits describing the selected reservation.
  iree_hal_pool_acquire_flags_t flags;

  // Result for this request within the containing transaction.
  iree_hal_pool_acquire_result_t result;
} iree_hal_pool_acquire_info_t;

// Borrowed prepared storage already owned by a live pool reservation.
//
// This lets a caller-provided allocation root publish the reservation's
// target-visible memory without constructing a redundant HAL buffer object.
// The source pool and reservation retain every borrowed field. Callers retain
// |buffer| only when they need it to outlive the reservation query itself and
// release that reference before returning the reservation.
typedef struct iree_hal_pool_reservation_view_t {
  // Borrowed buffer providing the native implementation for this range.
  iree_hal_buffer_t* buffer;

  // Allocation-relative byte offset forwarded to |buffer|'s implementation.
  iree_device_size_t byte_offset;

  // User-visible byte length of the prepared range.
  iree_device_size_t byte_length;

  // Exact prepared memory facts at this range's byte zero.
  iree_hal_buffer_memory_view_t memory;
} iree_hal_pool_reservation_view_t;

// Controls how a concrete buffer object/view is materialized for a reservation.
typedef uint32_t iree_hal_pool_materialize_flags_t;
enum iree_hal_pool_materialize_flag_bits_e {
  IREE_HAL_POOL_MATERIALIZE_FLAG_NONE = 0u,

  // Transfers reservation ownership to the returned buffer. When that buffer
  // is destroyed its release callback must return |reservation| to |pool|
  // with a NULL death frontier, after applying RELEASED advice if guarded.
  // The caller applies ALLOCATED advice before use and establishes actual
  // completion of all accesses before destroying the buffer.
  //
  // Without this flag, the returned buffer is only a borrowed view of the
  // reserved bytes and the caller remains responsible for calling
  // iree_hal_pool_release_reservations() exactly once.
  IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP = 1u << 0,
};

// Controls which unused backing resources are eligible for trimming.
// The explicit retained-byte floor applies independently of these flags.
typedef uint32_t iree_hal_pool_trim_flags_t;
enum iree_hal_pool_trim_flag_bits_e {
  // Uses the implementation's normal retention policy.
  IREE_HAL_POOL_TRIM_FLAG_NONE = 0u,

  // Releases all eligible unused resources regardless of optional retention
  // targets. Takes precedence over EXCESS when both flags are present.
  IREE_HAL_POOL_TRIM_FLAG_ALL = 1u << 0,

  // Releases only resources above the implementation's retention targets.
  IREE_HAL_POOL_TRIM_FLAG_EXCESS = 1u << 1,
};

// Describes the memory capabilities of a pool. Computed at pool creation time
// from the slab provider's properties and the pool's strategy constraints.
// Used by iree_hal_pool_set_t for routing allocation requests to compatible
// pools.
typedef struct iree_hal_pool_capabilities_t {
  // Achieved owned-backing placement. AUTOMATIC promises no particular node;
  // REQUIRED reports the node enforced by native allocation. Imported storage
  // keeps its own placement and is not relocated by this guarantee.
  iree_hal_pool_placement_t placement;

  // Memory type properties provided by this pool's slab provider. Checked
  // against the required bits in iree_hal_buffer_params_t.type.
  iree_hal_memory_type_t memory_type;

  // Access permissions available to views materialized from this pool.
  iree_hal_memory_access_t allowed_access;

  // Buffer usages this pool supports. A pool backed by DEVICE_LOCAL memory
  // that isn't host-visible can't serve MAPPING usage.
  iree_hal_buffer_usage_t supported_usage;

  // Queue families that may access buffers materialized from this pool.
  // IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY means every queue family in the
  // containing logical device.
  iree_hal_queue_family_affinity_t queue_family_affinity;

  // Atomic operations supported by naturally aligned locations materialized
  // from this pool. Queue-family capabilities are required independently.
  iree_hal_atomic_operation_capabilities_t atomic_operations;

  // Minimum user-visible allocation size in bytes. Suballocating pools may
  // round internally and report 0 or 1 when they have no practical lower bound.
  iree_device_size_t min_allocation_size;

  // Strategy-specific maximum single user-visible reservation in bytes.
  // Finite pools report their managed geometry; growable pools inherit their
  // backing limit. A zero value means no strategy limit. Budgets are reported
  // separately and enforced by reservation acquisition.
  iree_device_size_t max_allocation_size;

  // Largest power-of-two alignment accepted by reservation requests.
  iree_device_size_t max_allocation_alignment;

  // Minimum independently maintained byte granule in the backing storage.
  // Suballocators align both endpoints to this granule.
  iree_device_size_t maintenance_alignment;
} iree_hal_pool_capabilities_t;

// Running statistics for a pool. All values are atomic snapshots; they may
// be momentarily inconsistent under concurrent modifications. Querying stats
// is O(1) with no internal walks or locks.
typedef struct iree_hal_pool_stats_t {
  // Total backing bytes currently charged to live reservations. This may
  // exceed the sum of reservation byte lengths due to allocation granularity,
  // hidden guard regions, sanitizer redzones, or other pool-owned overhead.
  iree_device_size_t bytes_reserved;

  // Total backing bytes in free blocks or otherwise available for reservation.
  iree_device_size_t bytes_free;

  // Returned backing bytes withheld from reuse by this pool's ASAN policy.
  // Does not include quarantine owned by a backing pool or native allocator.
  iree_device_size_t bytes_quarantined;

  // Total physical memory committed (slabs or VMM pages).
  iree_device_size_t bytes_committed;

  // Budget limit in bytes (0 = unlimited).
  iree_device_size_t budget_limit;

  // Number of currently live reservations.
  uint32_t reservation_count;

  // Number of slabs from the slab provider.
  uint32_t slab_count;

  // Total successful reservation acquisitions.
  uint64_t reserve_count;

  // Total reservation releases.
  uint64_t release_count;

  // Reserves that hit frontier-dominated reuse.
  uint64_t reuse_count;

  // Reserves where the dominance check failed.
  uint64_t reuse_miss_count;

  // Reserves from fresh, never-used offset space.
  uint64_t fresh_count;

  // Reserves that returned EXHAUSTED.
  uint64_t exhausted_count;

  // Reserves that returned OVER_BUDGET.
  uint64_t over_budget_count;

  // Reserves that returned NEEDS_WAIT.
  uint64_t wait_count;

  // Ranges removed from this pool's quarantine by pressure or explicit trim.
  // A disabled quarantine does not retain ranges or count evictions.
  uint64_t quarantine_eviction_count;
} iree_hal_pool_stats_t;

// Callback for try-before-fence epoch queries. When a death-frontier dominance
// check fails during reservation acquisition, the pool calls this to check
// whether the timeline has actually advanced past the death frontier's epoch
// on a specific axis, even though the requester's frontier hasn't imported the
// update yet.
//
// Returns true if |axis| has reached at least |epoch| (the work has
// completed). This is a host-side read of the semaphore's current value;
// no device interaction, no blocking. The pool uses this to avoid
// unnecessarily skipping reusable blocks when completion notifications are
// batched by the proactor.
//
// Set at pool creation time. Pools created with |epoch_query.fn| == NULL skip
// the try-before-fence optimization; non-dominated blocks are treated as
// genuinely unavailable for zero-sync reuse.
typedef bool(IREE_API_PTR* iree_hal_pool_epoch_query_fn_t)(
    void* user_data, iree_async_axis_t axis, uint64_t epoch);

// Bound epoch query callback and user data.
typedef struct iree_hal_pool_epoch_query_t {
  // Callback used to query whether an axis has reached an epoch.
  iree_hal_pool_epoch_query_fn_t fn;

  // User data passed to |fn|.
  void* user_data;
} iree_hal_pool_epoch_query_t;

// Returns a null epoch query callback.
static inline iree_hal_pool_epoch_query_t iree_hal_pool_epoch_query_null(void) {
  iree_hal_pool_epoch_query_t query = {NULL, NULL};
  return query;
}

//===----------------------------------------------------------------------===//
// iree_hal_pool_t
//===----------------------------------------------------------------------===//

// A memory pool that manages a region of offset space backed by one or more
// slabs of physical memory. Pools are the primary allocation interface in IREE:
// both synchronous (allocate_buffer) and asynchronous (queue_alloca/dealloca)
// allocation paths go through pools.
//
// Pools are ref-counted HAL resources, but allocations/wrapped buffers borrow
// their source pool instead of retaining it. Pool owners must ensure a pool
// outlives every reservation and buffer allocated from it. That keeps
// queue_alloca/dealloca hot paths free of per-allocation pool refcount traffic
// and matches the intended "application-scoped allocation policy" lifetime
// model.
//
// Pools can be shared across queues and threads (concurrency is handled
// internally per pool type).
//
// The pool base type is abstract; concrete pools are created by type-specific
// factory functions (iree_hal_tlsf_pool_create,
// iree_hal_fixed_block_pool_create, etc.) and used through this common
// interface.
//
// ## Allocation protocol
//
// Asynchronous (queue-ordered) allocation:
//
//   submit queue_alloca             use bytes             submit queue_dealloca
// ┌─────────────────────┐      ┌────────────────┐      ┌──────────────────────┐
// │ acquire transaction │─────▶│ borrowed views │─────▶│ release transaction  │
// │ checks death edge   │      │ no pool retain │      │ records death edge   │
// └─────────────────────┘      └────────────────┘      └──────────────────────┘
//
//   1. acquire_reservations() at submit time: finds free blocks, checks death
//      frontier dominance, and returns reservations with offsets and lengths.
//   2. query_reservation_views() publishes existing prepared storage directly
//      into caller-provided roots when supported. Other pools use
//      materialize_reservations() without ownership transfer to create backing
//      buffer views whose lifetimes are independent from the reservations.
//   3. release_reservations() at the queue implementation's dealloca retirement
//      point: returns the blocks to pool reuse metadata, tagged with a death
//      frontier.
//
// Synchronous allocation:
//   iree_hal_pool_allocate_buffer(): submits one-element acquire and
//   materialize transactions with TRANSFER_RESERVATION_OWNERSHIP in a loop,
//   waiting on the pool's notification if exhausted or its captured completion
//   tracker if a reserved range still has a pending death frontier. This is a
//   shared utility, not a vtable method.
//
// ## Death frontier integration
//
// Free blocks carry death frontiers: causal snapshots from when they were
// last freed. Reserve checks whether the requester's frontier dominates the
// death frontier for zero-sync reuse. This enables buffer recycling without
// any device synchronization in steady-state pipelined workloads.
typedef struct iree_hal_pool_t iree_hal_pool_t;

// Retains the given |pool| for the caller.
IREE_API_EXPORT void iree_hal_pool_retain(iree_hal_pool_t* pool);

// Releases the given |pool| from the caller.
IREE_API_EXPORT void iree_hal_pool_release(iree_hal_pool_t* pool);

// Acquires an all-or-none transaction of reservations for future allocations.
//
// Each returned reservation may be larger than its requested allocation size
// due to alignment rounding or block splitting constraints, but does not expose
// hidden backing bytes reserved by the pool for provider metadata, guard
// regions, or sanitizer redzones. Every request must be satisfiable by this
// pool; routing among pools is a higher-level pool implementation concern.
//
// |requester_frontier| is the queue scheduler's current causal position, used
// for death frontier dominance checking. Pass NULL to skip dominance checking
// (appropriate for synchronous allocations that don't participate in
// queue-ordered frontier tracking).
//
// |flags| controls whether the caller can accept queue-owned dependency work
// as part of the reservation. In particular, callers must set
// IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER before a pool may return
// IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT.
//
// On success (iree_ok_status()), |out_result| describes the whole transaction:
//   OK: all reservations succeeded and at least one recycled range was used.
//   OK_FRESH: all reservations succeeded from previously unused ranges.
//   OK_NEEDS_WAIT: all reservations succeeded and at least one requires a
//     hidden wait recorded in the corresponding |out_infos| entry.
//   EXHAUSTED: no reservations were made because at least one request could
//     not be satisfied.
//   OVER_BUDGET: no reservations were made because the transaction would
//     exceed the pool budget.
//
// On a successful reservation result (OK, OK_FRESH, or OK_NEEDS_WAIT), every
// output reservation and information record is assigned. Each information
// record contains that request's successful result; the transaction result
// summarizes them with OK_NEEDS_WAIT taking precedence over OK and OK taking
// precedence over OK_FRESH. Each information record whose result is
// OK_NEEDS_WAIT has a non-NULL, non-empty reuse_frontier. OK may also retain a
// reuse_frontier even though this requester needs no extra wait. Frontier
// storage is borrowed and remains immutable until its reservation is released.
//
// On EXHAUSTED or OVER_BUDGET, every information record and |out_result| are
// assigned while the reservation outputs remain untouched. One or more
// information records identify the requests that could not be satisfied with
// the transaction result and any associated flags; requests not committed or
// not examined use IREE_HAL_POOL_ACQUIRE_NONE. On an error status all outputs
// are untouched.
//
// Returns an error status (with backtrace) only for infrastructure failures:
// invalid arguments (size 0, non-power-of-two alignment), internal
// corruption, or slab provider errors. These are exceptional.
IREE_API_EXPORT iree_status_t iree_hal_pool_acquire_reservations(
    iree_hal_pool_t* pool, iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reserve_flags_t flags,
    iree_hal_pool_reservation_t* out_reservations,
    iree_hal_pool_acquire_info_t* out_infos,
    iree_hal_pool_acquire_result_t* out_result);

// Releases a transaction of reservations back to the pool's free list.
//
// |reservations| is a transaction returned by a prior successful
// iree_hal_pool_acquire_reservations() call on this pool. |death_frontier| is
// the causal snapshot attached to every freed block; typically the queue's
// dealloca completion frontier. Pass NULL for an empty frontier (the blocks are
// immediately available for zero-sync reuse by any requester).
//
// The reservation's offset is returned to the pool's reuse metadata
// immediately. The range may be considered by future reservation transactions
// from that point forward, even though device work may still be executing prior
// accesses; death frontier dominance checking gates actual reuse safety.
//
// Target-specific deallocation effects that change device-visible memory state
// are not implied by this bookkeeping call. Queue-ordered users that need
// sanitizer poisoning, guard-page changes, VMM unmapping, or similar effects
// must perform those effects after queue_dealloca waits are satisfied and
// before queue_dealloca signals are published, or at an equivalent proven
// lifetime boundary on synchronous targets.
//
// Publishes the pool's notification epoch. Releases that occur with no known
// waiter may skip platform wake work; waiters use an observe-check-wait
// protocol so this cannot lose wakeups.
IREE_API_EXPORT void iree_hal_pool_release_reservations(
    iree_hal_pool_t* pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    const iree_async_frontier_t* death_frontier);

// Returns whether this pool's reservations require explicit ASAN lifecycle
// advice. This is an immutable local property; supporting guarded child pools
// does not imply that this pool's own reservations are guarded.
IREE_API_EXPORT bool iree_hal_pool_requires_asan_advice(
    const iree_hal_pool_t* pool);

// Applies an ASAN lifecycle transition to live reservation tokens. Does nothing
// for an unguarded pool. This operation is infallible after pool construction.
//
// ALLOCATED runs after all inherited reuse prerequisites have actually
// completed and before the new user accesses the range. RELEASED runs after the
// user's accesses have actually completed, before publishing deallocation
// completion and before returning the token. Requester dominance and an
// enqueued device wait do not establish host completion. Queue implementations
// order this call at the corresponding execution boundary; synchronous
// allocation helpers do so on the caller's behalf.
//
// Acquiring, materializing and returning reservations perform no ASAN advice.
// An unused reservation returned during rollback needs neither transition.
IREE_API_EXPORT void iree_hal_pool_advise_asan_reservations(
    iree_hal_pool_t* pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_asan_range_advice_flags_t flags);

// Queries existing prepared storage for a live reservation transaction.
//
// Returns true and assigns every output when the concrete pool already owns a
// stable backing buffer for each reservation. The query performs no allocation,
// retention, materialization, native work, or synchronization. Each returned
// memory view includes the reservation's exact reuse prerequisite.
//
// Returns false without modifying outputs when this pool must materialize a
// provider-specific buffer object. Callers then use
// iree_hal_pool_materialize_reservations().
//
// |reservations| must be a successful live transaction produced by |pool|.
// The pool and transaction must outlive every use of the borrowed outputs.
IREE_API_EXPORT bool iree_hal_pool_query_reservation_views(
    iree_hal_pool_t* pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_reservation_view_t* out_views);

// Materializes concrete buffer objects or views for a reservation transaction.
//
// |requests| is the allocation request transaction used to acquire
// |reservations|. The operation validates and materializes the entire set.
//
// If |flags| includes
// IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, each returned
// buffer stores its reservation and a borrowed pointer to |pool| and releases
// that reservation with a NULL death frontier when destroyed. Ownership
// transfers only after every buffer has been materialized successfully.
//
// Otherwise the returned buffers are borrowed views and the caller keeps
// ownership of every reservation. This is the queue-allocation path's backing
// materialization primitive: transient wrappers can commit/decommit the
// borrowed views while releasing the reservations independently at a
// queue-ordered dealloca point.
//
// The operation is all-or-none. Output buffer slots are assigned only when the
// function returns OK and are otherwise untouched. |pool| must outlive every
// reservation and returned buffer.
//
// The concrete pool owns reservation bookkeeping and release callbacks. Native
// slabs are materialized through their provider; pools over retained buffer
// ranges create ordinary subspans of that backing. Generic offset allocators
// never dereference native slab payload fields. Only the reservation's visible
// range is exposed; hidden backing bytes remain owned by the source.
IREE_API_EXPORT iree_status_t iree_hal_pool_materialize_reservations(
    iree_hal_pool_t* pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffers);

// Queries the memory capabilities of the pool. O(1); reads cached fields
// computed at pool creation time.
IREE_API_EXPORT void iree_hal_pool_query_capabilities(
    const iree_hal_pool_t* pool,
    iree_hal_pool_capabilities_t* out_capabilities);

// Qualifies a child allocator's sanitizer policy against this pool's prepared
// storage. Called during construction, never during reservation selection.
IREE_API_EXPORT iree_status_t iree_hal_pool_validate_asan_options(
    const iree_hal_pool_t* pool, const iree_hal_asan_pool_options_t* options);

// Queries the pool's running statistics. O(1); atomic snapshots of
// incrementally maintained counters. Values may be momentarily inconsistent
// under concurrent modifications.
IREE_API_EXPORT void iree_hal_pool_query_stats(
    const iree_hal_pool_t* pool, iree_hal_pool_stats_t* out_stats);

// Returns unused backing while retaining at least |min_bytes_to_keep| of the
// pool's committed backing, subject to the retention policy selected by
// |flags|. The floor applies to total committed backing, including live
// allocations, not an additional reserve of free bytes. Whole-slab/page
// granularity may retain more. A floor above current backing never grows the
// pool.
//
// Only reclaimable resources are released: trimming does not wait for execution
// completion, move live allocations, or invalidate outstanding reservations.
// Pools backed by a fixed caller-supplied range retain that range. Growable
// pools remain usable and can acquire backing again after trimming to zero.
//
// Returning backing to a caching source does not necessarily release it to the
// system. Concrete implementations forward |flags| to their backing providers;
// the byte floor describes this pool's backing, not a shared provider's cache.
// Reclamation is best-effort and does not report unused capacity as an error.
IREE_API_EXPORT void iree_hal_pool_trim(iree_hal_pool_t* pool,
                                        iree_hal_pool_trim_flags_t flags,
                                        iree_device_size_t min_bytes_to_keep);

// Returns the notification for this pool's local capacity changes. Backing
// pools may publish independent capacity changes; allocation retries use the
// common pool_wait helper to observe every captured source.
//
// The notification is advisory over pool state. Callers must observe the
// notification epoch before checking the pool state, and then wait on that
// token only if the checked state still requires a wakeup. Releases may skip
// platform wake work when no wait observer exists.
IREE_API_EXPORT iree_async_notification_t* iree_hal_pool_notification(
    iree_hal_pool_t* pool);

// Allocates a buffer from the pool synchronously.
//
// This is a shared utility (NOT a vtable method) that calls
// one-element acquire and materialize transactions in a loop. If acquisition
// returns EXHAUSTED or OVER_BUDGET, the function waits on the pool's
// captured local/backing notifications and retries. Their proactor owners must
// remain polling during a capacity wait. An allocation may instead reserve a
// range with pending reuse dependencies and wait for that exact frontier
// through the completion tracker captured by the pool.
//
// Success establishes actual completion of prior accesses before returning the
// buffer. A queue dependency frontier cannot substitute for that completion.
//
// This helper is synchronous-only. Queue implementations must not call it for
// queue_alloca, because queue-owned memory-frontier waits and pool-notification
// retries are scheduler state, not host-thread blocking in this helper.
//
// |timeout| is converted to one absolute deadline shared by capacity waits,
// retries and frontier completion:
//   iree_immediate_timeout(): poll completion without registering a waiter.
//   iree_infinite_timeout(): block until a usable range becomes available.
//   iree_make_timeout_ms(N): wait up to N milliseconds.
//
// Returns IREE_STATUS_DEADLINE_EXCEEDED when a capacity wait times out or a
// timed-out frontier wait is successfully cancelled. If frontier callback
// dispatch wins the cancellation race, the function joins that callback and
// continues with its actual result, which may extend beyond the deadline.
// Callback storage is no longer in use when this function returns.
//
// A failed frontier wait returns the reservation with its original dependency
// intact. |out_buffer| is assigned only on success and is otherwise unchanged.
IREE_API_EXPORT iree_status_t iree_hal_pool_allocate_buffer(
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_timeout_t timeout,
    iree_hal_buffer_t** out_buffer);

//===----------------------------------------------------------------------===//
// iree_hal_pool_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_pool_vtable_t {
  // Destroys a concrete pool implementation.
  void(IREE_API_PTR* destroy)(iree_hal_pool_t* pool);

  // Acquires a reservation transaction from the concrete pool implementation.
  iree_status_t(IREE_API_PTR* acquire_reservations)(
      iree_hal_pool_t* pool, iree_host_size_t request_count,
      const iree_hal_pool_reservation_request_t* requests,
      const iree_async_frontier_t* requester_frontier,
      iree_hal_pool_reserve_flags_t flags,
      iree_hal_pool_reservation_t* out_reservations,
      iree_hal_pool_acquire_info_t* out_infos,
      iree_hal_pool_acquire_result_t* out_result);

  // Releases a reservation transaction to the concrete pool implementation.
  void(IREE_API_PTR* release_reservations)(
      iree_hal_pool_t* pool, iree_host_size_t reservation_count,
      const iree_hal_pool_reservation_t* reservations,
      const iree_async_frontier_t* death_frontier);

  // Queries existing prepared storage without constructing buffer objects.
  // Optional; NULL when provider-specific materialization is required.
  void(IREE_API_PTR* query_reservation_views)(
      iree_hal_pool_t* pool, iree_host_size_t reservation_count,
      const iree_hal_pool_reservation_t* reservations,
      iree_hal_pool_reservation_view_t* out_views);

  // Materializes concrete buffer objects or views for a reservation set.
  iree_status_t(IREE_API_PTR* materialize_reservations)(
      iree_hal_pool_t* pool, iree_host_size_t reservation_count,
      const iree_hal_pool_reservation_request_t* requests,
      const iree_hal_pool_reservation_t* reservations,
      iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffers);

  // Queries cached memory capabilities for routing.
  void(IREE_API_PTR* query_capabilities)(
      const iree_hal_pool_t* pool,
      iree_hal_pool_capabilities_t* out_capabilities);

  // Qualifies native range advice during child-pool construction. NULL when
  // the pool cannot supply ASAN-capable prepared storage.
  iree_status_t(IREE_API_PTR* validate_asan)(
      const iree_hal_pool_t* pool, const iree_hal_asan_pool_options_t* options);

  // Queries running pool statistics.
  void(IREE_API_PTR* query_stats)(const iree_hal_pool_t* pool,
                                  iree_hal_pool_stats_t* out_stats);

  // Trims reclaimable backing subject to policy and the retained-byte floor.
  void(IREE_API_PTR* trim)(iree_hal_pool_t* pool,
                           iree_hal_pool_trim_flags_t flags,
                           iree_device_size_t min_bytes_to_keep);

  // Applies qualified ASAN advice after caller-established actual completion.
  // Required only when the pool's asan_enabled property is true.
  void(IREE_API_PTR* advise_asan_reservations)(
      iree_hal_pool_t* pool, iree_host_size_t reservation_count,
      const iree_hal_pool_reservation_t* reservations,
      iree_hal_asan_range_advice_flags_t flags);
} iree_hal_pool_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_pool_vtable_t);

// Immutable capacity notifications captured by a pool at construction.
typedef struct iree_hal_pool_wait_source_list_t {
  // Number of distinct notifications in the immutable list.
  iree_host_size_t count;
  // Borrowed notifications, each serviced by its own proactor owner.
  iree_async_notification_t* const* values;
} iree_hal_pool_wait_source_list_t;

// Common pool state embedded at offset zero in every pool implementation.
struct iree_hal_pool_t {
  // Base HAL resource state. Must be at offset zero.
  iree_hal_resource_t resource;

  // Retained immutable access facts shared by all backing and child views.
  iree_hal_memory_contract_t* memory_contract;

  // Owned notification for changes in this pool's available capacity.
  iree_async_notification_t* notification;

  // Borrowed completion tracker for all frontiers used with this pool.
  // Its owning device group must outlive the pool and its operations.
  iree_async_frontier_tracker_t* frontier_tracker;

  // Borrowed placement-local owner captured from native storage or the
  // retained backing pool. NULL for sources without cold preparation support.
  iree_hal_memory_maintenance_t* maintenance;

  // Captured optional completion probe inherited by child allocators. Its
  // borrowed context belongs to the same sealed device group as the tracker.
  iree_hal_pool_epoch_query_t epoch_query;

  // Immutable requirement for advice on this pool's own reservation lifetimes.
  bool asan_enabled;

  // Captured local and backing capacity sources, with duplicates removed.
  iree_hal_pool_wait_source_list_t wait_sources;

  // Allocator for captured wait sources and cold synchronous wait helpers.
  iree_allocator_t wait_allocator;
};

// Initializes |out_pool| with one owning reference. Captures and retains the
// local |notification| and immutable, distinct |backing_sources|. Single-source
// pools use inline storage. Retains |memory_contract| when provided.
// Borrows the non-NULL |frontier_tracker|; all
// reservation frontiers use the tracker's registered axes. On failure no
// references are retained and the output requires no deinitialization.
IREE_API_EXPORT iree_status_t iree_hal_pool_initialize(
    const iree_hal_pool_vtable_t* vtable,
    iree_hal_memory_contract_t* memory_contract,
    iree_async_notification_t* notification,
    iree_hal_pool_wait_source_list_t backing_sources,
    iree_async_frontier_tracker_t* frontier_tracker,
    iree_allocator_t host_allocator, iree_hal_pool_t* out_pool);

// Releases common pool state during concrete destruction. The notification's
// proactor must remain alive until this call returns.
IREE_API_EXPORT void iree_hal_pool_deinitialize(iree_hal_pool_t* pool);

IREE_API_EXPORT void iree_hal_pool_destroy(iree_hal_pool_t* pool);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_POOL_H_
