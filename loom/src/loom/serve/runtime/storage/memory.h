// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_STORAGE_MEMORY_H_
#define LOOM_SERVE_RUNTIME_STORAGE_MEMORY_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loom/serve/runtime/host/wake.h"

#ifdef __cplusplus
extern "C" {
#endif

// Single-owner physical allocation domain shared by independent virtual
// buffers. Logical blocks and model geometry belong to its callers. Admission
// and result publication stay on that owner; physical and mapping operations
// use public HAL queues. These are capacity transitions, not per-token work.
// Backing supports storage, transfers, and indirect dispatch parameters.
typedef struct loom_serve_memory_pool_t loom_serve_memory_pool_t;
typedef struct loom_serve_memory_arena_t loom_serve_memory_arena_t;
typedef struct loom_serve_virtual_buffer_t loom_serve_virtual_buffer_t;
typedef struct loom_serve_memory_operation_t loom_serve_memory_operation_t;

// Physical replacement geometry qualified against one mapping source and its
// virtual targets. Logical tensor/block sizes remain independent of it.
typedef struct loom_serve_memory_geometry_t {
  // Maximum independently allocated and replaced extent in bytes.
  iree_device_size_t slab_size;
  // Common source/target alignment for physical extents and virtual addresses.
  iree_device_size_t granularity;
  // Minimum native extent, including source/target alignment.
  iree_device_size_t minimum_size;
} loom_serve_memory_geometry_t;

// The device configures bulk and sparse storage geometries. Equal physical
// classes share one cache; a short tail can match either class or bypass both.
enum { LOOM_SERVE_MEMORY_CLASS_COUNT = 2 };

// Additional physical ownership within one pool, not a reservation of credit.
// Folds distinct roots/groups without enumerating their physical slots.
typedef struct loom_serve_memory_demand_t {
  // Required bytes in each immutable physical size/alignment class.
  uint64_t cached_bytes[LOOM_SERVE_MEMORY_CLASS_COUNT];
  // Exact unmatched extents requiring native acquisition.
  uint64_t uncached_bytes;
} loom_serve_memory_demand_t;

void loom_serve_memory_demand_add(loom_serve_memory_demand_t* demand,
                                  loom_serve_memory_demand_t additional);

// Qualifies explicit power-of-two slab bytes, or the source recommendation
// when zero, against the native source and all targets in its scope.
iree_status_t loom_serve_memory_pool_query_geometry(
    const iree_hal_virtual_buffer_capabilities_t* capabilities,
    iree_hal_pool_t* storage_pool, iree_device_size_t slab_size,
    loom_serve_memory_geometry_t* out_geometry);

// Additive usage across distinct reservations or banks on one serialized owner.
// Views borrowing overlapping owners cannot be summed. Historical peaks are
// not additive and belong to the storage owner that observes their transitions.
typedef struct loom_serve_memory_usage_t {
  // Reserved virtual bytes, including final slab alignment.
  uint64_t reserved_bytes;
  // Unique backing bytes currently published by the owner.
  uint64_t committed_bytes;
  // Budget charged to accepted commitments not yet published by the owner.
  uint64_t pending_bytes;
  // Backing selected for return, until logical release is published.
  uint64_t releasing_bytes;
  // Cumulative backing returns published by the owner.
  uint64_t released_bytes;
} loom_serve_memory_usage_t;

// Statistics maintained by one storage owner, not reconstructed from views of
// independently activated or retired banks.
typedef struct loom_serve_memory_statistics_t {
  // Reserved virtual bytes, including final slab alignment.
  uint64_t reserved_bytes;
  // Unique backing bytes; pool-wide queries include cached idle slabs and
  // pending or abandoned native retirement. Per-reservation groups contain
  // only live slabs.
  uint64_t committed_bytes;
  // Pool-only committed charge whose native backing could not be returned.
  // Monotonic; neither reusable nor pending, and not attributed to live roots.
  uint64_t abandoned_bytes;
  // Physically committed idle slabs available to any reservation in the pool.
  // Only pool-wide queries include this shared cache; per-reservation is zero.
  uint64_t cached_bytes;
  // Budget charged to accepted native commitments not yet published by owner.
  uint64_t pending_bytes;
  // Backing selected for return. Per-buffer accounting ends at logical return;
  // pool-wide accounting also includes outstanding native release.
  uint64_t releasing_bytes;
  // Maximum simultaneously committed physical bytes.
  uint64_t peak_bytes;
  // Cumulative logical backing returns by reservations, including cache reuse.
  // This is not cumulative native disposal or available physical capacity.
  uint64_t released_bytes;
} loom_serve_memory_statistics_t;

// Retains the mapping-source pool and copies the exact virtual-target scope.
// The sealed group and selected transfer queue outlive this pool. Geometries
// have been qualified by query_geometry for this source and scope. limit_bytes
// bounds physical commitment, including pending native release and permanent
// backing loss. This is the exclusive source-pool owner for the budget's entire
// lifetime. Zero imposes no serving budget. Each reservation borrows this pool
// and is destroyed before the pool.
iree_status_t loom_serve_memory_pool_allocate(
    iree_hal_device_group_t* group, iree_hal_pool_scope_t scope,
    iree_hal_pool_t* storage_pool, iree_hal_queue_t* queue,
    const loom_serve_memory_geometry_t
        geometries[LOOM_SERVE_MEMORY_CLASS_COUNT],
    uint64_t limit_bytes, iree_allocator_t host_allocator,
    loom_serve_memory_pool_t** out_pool);

void loom_serve_memory_pool_free(loom_serve_memory_pool_t* pool);

loom_serve_memory_statistics_t loom_serve_memory_pool_statistics(
    loom_serve_memory_pool_t* pool);

// Returns idle physical slabs toward target_bytes of total native commitment.
// Live mappings remain intact. Accepted detachments retain this trim policy
// through completion; native disposal remains charged until it finishes.
// Ordinary model deactivation only detaches slabs and does not call this.
void loom_serve_memory_pool_trim(loom_serve_memory_pool_t* pool,
                                 uint64_t target_bytes);

// Immutable physical commitment ceiling in bytes; UINT64_MAX is unbounded.
// This is the configured budget, not currently available backing.
uint64_t loom_serve_memory_pool_limit(const loom_serve_memory_pool_t* pool);

// Owner-maintained bytes attached or promised to live virtual reservations.
// Includes pending commitments and excludes reusable idle backing. This reads
// the existing range ledger without collecting work or querying native pools.
uint64_t loom_serve_memory_pool_live_bytes(
    const loom_serve_memory_pool_t* pool);

// Clones a shared capacity-observation or reclamation failure, or returns OK.
// Those failures apply to the shared domain; the serialized owner stops new
// work and drains consumers before cleanup. Individual allocation, mapping,
// and readiness errors remain on their operation and affected buffers.
// Querying does not clear the shared failure.
iree_status_t loom_serve_memory_pool_query_status(
    loom_serve_memory_pool_t* pool);

// Installs the serialized owner's nonblocking wake on accepted and subsequent
// native work. Passing NULL detaches and joins the last callback access. The
// callback only schedules its owner and cannot reenter this pool. This owner
// observes completion/failure independently of any individual model lifetime.
// Returns the previous observer so an exclusive blocking outer owner can
// temporarily route capacity wakes to its wait and restore them afterward.
loom_serve_wake_callback_t loom_serve_memory_pool_set_observer(
    loom_serve_memory_pool_t* pool, void (*wake)(void* user_data),
    void* user_data);

// Collects completed operations and observes remaining native borrowers. True
// requires another owner pass even if its capacity wake has already arrived;
// one native capacity change does not establish that every release retired.
bool loom_serve_memory_pool_is_pending(loom_serve_memory_pool_t* pool);

// Outer-thread join of accepted native operations, not device work or logical
// reclamation that still needs its model scheduler. Never called by task
// owners. Publishes final accounting and returns any shared capacity or
// reclamation failure. Each operation retains its own result.
// Permanent backing loss remains charged; it is not unfinished native work.
iree_status_t loom_serve_memory_pool_await(loom_serve_memory_pool_t* pool);

// Cold pressure policy, independent of logical storage geometry. The callback
// may trim eligible reservations toward target_bytes of total commitment; it
// returns a status only for an actual platform/IO failure. An unreachable
// target is not an error. It must not commit memory or recurse into admission.
typedef struct loom_serve_memory_reclaimer_t {
  // Borrowed policy owner, alive until the callback is removed.
  void* user_data;
  // Serialized reclaim operation; NULL disables automatic reclamation.
  iree_status_t (*fn)(void* user_data, uint64_t target_bytes);
} loom_serve_memory_reclaimer_t;

void loom_serve_memory_pool_set_reclaimer(
    loom_serve_memory_pool_t* pool, loom_serve_memory_reclaimer_t reclaimer);

typedef enum loom_serve_memory_admission_e {
  // The requested bytes cannot currently fit and no capacity wake was armed.
  LOOM_SERVE_MEMORY_ADMISSION_CAPACITY = 0,
  // Accepted work or a native return can change capacity and wake the owner.
  // This result remains authoritative even if that completion races the return.
  LOOM_SERVE_MEMORY_ADMISSION_DEFERRED,
  // Capacity is available; the serialized owner can commit before yielding.
  LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE,
} loom_serve_memory_admission_t;

// Makes room for the complete additional demand using the installed cold
// policy. Includes pending commitments but does not itself allocate or reserve
// credit: the single owner commits before yielding admission to another caller.
// Consumers pin every residency they need before asking for capacity,
// including already backed ranges the policy could otherwise trim. Accepted
// detaches grant reusable capacity after checked return completion. Backing
// selected for physical disposal remains charged until native completion. The
// result preserves this attempt's progress observation independently of model
// retirement. If reported native loss is what prevents this demand from fitting
// and no return remains pending, returns its full cleanup cause. Such a failure
// does not poison other demands which fit the surviving capacity.
iree_status_t loom_serve_memory_pool_prepare(
    loom_serve_memory_pool_t* pool, loom_serve_memory_demand_t additional,
    loom_serve_memory_admission_t* out_admission);

// Reserves stable device addresses without committing physical storage.
// The borrowed HAL buffer remains identical across commit/trim operations.
// statistics is a caller-owned, initially zeroed accounting group shared by
// related reservations. It outlives every reservation using it. The pool also
// accounts all reservations independently, enforcing their common budget.
iree_status_t loom_serve_virtual_buffer_allocate(
    loom_serve_memory_pool_t* pool, loom_serve_memory_geometry_t geometry,
    iree_device_size_t length, iree_device_size_t alignment,
    loom_serve_memory_statistics_t* statistics,
    loom_serve_virtual_buffer_t** out_buffer);

// Reserves one address interval for independently backed parameter ranges.
// The arena owns address allocation, mapping order and the native target's
// failure domain, but no physical payload. Its owner serializes allocation and
// retirement with pool and destroys every range before freeing the arena.
// Metadata follows allocated ranges, not the unoccupied address interval.
iree_status_t loom_serve_memory_arena_allocate(
    loom_serve_memory_pool_t* pool, loom_serve_memory_geometry_t geometry,
    iree_device_size_t length, loom_serve_memory_arena_t** out_arena);

// Borrowed full address interval, suitable for a single indexed binding.
// Retaining this handle preserves addresses, not range leases or backing.
iree_hal_buffer_t* loom_serve_memory_arena_handle(
    const loom_serve_memory_arena_t* arena);

// Reserves an independent range, initially without physical payload. Physical
// slabs and short tails are local to this range; it shares no replacement
// extent with another range. The arena outlives this buffer and all its views.
// Freeing the buffer after its consumers retire makes its offset reusable.
iree_status_t loom_serve_virtual_buffer_allocate_in_arena(
    loom_serve_memory_arena_t* arena, iree_device_size_t length,
    iree_device_size_t alignment, loom_serve_memory_statistics_t* statistics,
    loom_serve_virtual_buffer_t** out_buffer);

// Terminal checked removal after every arena consumer and accepted mapping
// has retired. Required before range cleanup following a failed shared-target
// edit. Failure preserves all backing ownership for another cleanup attempt.
// Successful detach permits range destruction, never further execution.
iree_status_t loom_serve_memory_arena_detach(loom_serve_memory_arena_t* arena);

// Destroys an arena after every allocated range has been freed. NULL is valid.
void loom_serve_memory_arena_free(loom_serve_memory_arena_t* arena);

iree_hal_buffer_t* loom_serve_virtual_buffer_handle(
    const loom_serve_virtual_buffer_t* buffer);

// Independent physical replacement size in this root.
iree_device_size_t loom_serve_virtual_buffer_slab_size(
    const loom_serve_virtual_buffer_t* buffer);
// Number of independently mapped physical slots spanning this root.
iree_host_size_t loom_serve_virtual_buffer_slab_count(
    const loom_serve_virtual_buffer_t* buffer);

// Missing backing in the serialized owner's root state, excluding accepted
// commitments, in constant time. Does not collect operations or establish
// readiness. The caller resolves prior detaches or failed commitments on this
// root through operation_is_complete/await before folding replacement demand;
// unrelated source operations may remain pending.
loom_serve_memory_demand_t loom_serve_virtual_buffer_missing_demand(
    const loom_serve_virtual_buffer_t* buffer);

// Backs the slabs intersecting a caller-validated nonempty byte range. Existing
// contents survive. New contents are unspecified. No consumer may access new
// ranges until this call succeeds. Failure can leave additional slabs owned;
// they remain accounted and are reclaimed by trim/destruction.
// The pool's pressure policy may reclaim other unpinned residencies first.
iree_status_t loom_serve_virtual_buffer_commit(
    loom_serve_virtual_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length);

// One caller-validated nonempty extent in a virtual reservation. Every range
// in a commitment transaction belongs to the same physical pool.
typedef struct loom_serve_memory_range_t {
  // Borrowed reservation whose physical backing is required.
  loom_serve_virtual_buffer_t* buffer;
  // Byte offset within the reservation.
  iree_device_size_t offset;
  // Number of accessible bytes required from offset.
  iree_device_size_t length;
} loom_serve_memory_range_t;

typedef struct loom_serve_memory_footprint_t {
  // Full physical extent of the range union, including unbacked slabs.
  uint64_t total_bytes;
  // Additional commitment, excluding live or already accepted backing.
  uint64_t missing_bytes;
} loom_serve_memory_footprint_t;

// Physical footprint of the caller-validated range union. Overlapping views
// count each slot once. Runs on the pool's serialized owner without querying
// native availability, committing storage, or changing trim/admission state.
// Saturates if the sum exceeds uint64_t.
loom_serve_memory_footprint_t loom_serve_memory_ranges_footprint(
    iree_host_size_t range_count, const loom_serve_memory_range_t* ranges);

// Admits and commits the union of missing slabs across all ranges. Overlapping
// ranges and views sharing a slab are counted once. False is ordinary capacity
// backpressure: no destination slabs were added. The pressure policy may have
// reclaimed eligible unrelated reservations. Callers pin any destination that
// that policy could trim. The single owner excludes intervening admissions.
// Platform failure may leave added backing, accounted for normal teardown.
iree_status_t loom_serve_memory_pool_try_commit(
    loom_serve_memory_pool_t* pool, iree_host_size_t range_count,
    const loom_serve_memory_range_t* ranges, bool* out_admitted);

// Admits the missing slab union on the serialized owner, then allocates, maps,
// and publishes it using batched HAL queue_alloca and queue_map operations.
// Pending bytes consume budget immediately. Completion only publishes captured
// results and wakes the owner; it never invokes the pressure policy. On
// success, readiness reaches value only after mappings are usable; failure
// fails readiness and preserves partial ownership for teardown. Existing ready
// ranges need no operation and signal immediately. Capacity denial returns no
// operation or readiness signal. DEFERRED preserves this attempt's observation
// of native progress even if it completes before the caller handles the result.
//
// The caller keeps every reservation and its accounting alive until the
// returned operation retires, independently of readiness success/failure. No
// other commitment or trim may touch its pending slabs. Ranges are copied; the
// readiness semaphore is retained. Mapping edits to one reservation are ordered
// even for disjoint slabs; source allocations and separate reservations remain
// independent. Completion may precede return. Inline
// readiness callbacks may schedule the owner but cannot collect or await this
// operation; the owner joins publication through is_complete before freeing it.
iree_status_t loom_serve_memory_pool_commit_async(
    loom_serve_memory_pool_t* pool, iree_host_size_t range_count,
    const loom_serve_memory_range_t* ranges, iree_hal_semaphore_t* readiness,
    uint64_t value, loom_serve_memory_admission_t* out_admission,
    loom_serve_memory_operation_t** out_operation);

// Commits the same range union while admitting additional immediate demand
// owned by the caller (such as parameters loaded after mutable-state growth).
// The demand excludes these ranges and existing committed or pending backing.
// Denial submits neither ranges nor additional demand. Success publishes range
// ownership before returning; the caller commits additional demand in the same
// serialized owner turn, before another admission can consume that headroom.
// This checks the serving owner's ledger, not available device memory.
iree_status_t loom_serve_memory_pool_commit_with_demand_async(
    loom_serve_memory_pool_t* pool, iree_host_size_t range_count,
    const loom_serve_memory_range_t* ranges,
    loom_serve_memory_demand_t immediate, iree_hal_semaphore_t* readiness,
    uint64_t value, loom_serve_memory_admission_t* out_admission,
    loom_serve_memory_operation_t** out_operation);

// Copies the unkept physical slab set of these distinct reservations and queues
// its return to the shared cache. The caller has retired every consumer and
// keeps reservations/accounting alive through operation completion. Kept slabs
// remain usable. No other operation may edit selected slabs before completion.
// The private mapping and deallocation completions join before optional
// readiness is signaled. Inline readiness callbacks obey the same non-reentry
// contract as commit_async. Capacity becomes reusable after the last captured
// reference returns the slab to the cache; its notification wakes admission.
// Physical backing remains cached until an explicit pool trim or destruction.
// An empty set returns NULL and signals immediately.
// Queue failures fail readiness and preserve unreleased ownership. A failed
// mapping edit makes the affected reservation unusable until checked detach;
// other reservations retain their independent ownership and usability.
iree_status_t loom_serve_memory_pool_trim_async(
    loom_serve_memory_pool_t* pool, iree_host_size_t buffer_count,
    loom_serve_virtual_buffer_t* const* buffers,
    iree_hal_semaphore_t* readiness, uint64_t value,
    loom_serve_memory_operation_t** out_operation);

// Installs a nonblocking ownership wake, or detaches it with NULL. The observer
// only schedules its owner; it cannot reenter the operation. Set before
// querying completion to avoid losing the last wake. NULL operations are
// accepted.
void loom_serve_memory_operation_set_observer(
    loom_serve_memory_operation_t* operation, void (*wake)(void* user_data),
    void* user_data);

// Publishes completed native results and accounting on the serialized owner.
// True joins captured queue ownership and the publisher's final callback
// access. After successful readiness this returns true; failed readiness alone
// does not imply completion. NULL is complete. A failed operation is reported
// by operation_query_status and readiness.
bool loom_serve_memory_operation_is_complete(
    loom_serve_memory_operation_t* operation);

// Cold outer-thread join, never used by scheduler tasks or completion
// callbacks. Publishes the same results as is_complete; failure remains on
// operation/readiness.
void loom_serve_memory_operation_await(
    loom_serve_memory_operation_t* operation);

// Clones the result after is_complete or await has joined publication. NULL
// reports OK. Failed native edits invalidate only their affected reservations;
// a failed caller readiness signal does not invalidate successful mappings.
iree_status_t loom_serve_memory_operation_query_status(
    const loom_serve_memory_operation_t* operation);

// Releases a completed operation. Caller-owned reservations may then be trimmed
// or destroyed. NULL is accepted; this never waits for native work.
void loom_serve_memory_operation_free(loom_serve_memory_operation_t* operation);

// After retiring affected users, clear the keep set, mark all still-live byte
// ranges, then trim. Marking does not commit memory. A physical slab shared by
// several logical ranges stays backed if any range needs it. The caller owns
// readiness/retirement: these functions never wait for device work themselves.
void loom_serve_virtual_buffer_begin_trim(loom_serve_virtual_buffer_t* buffer);
void loom_serve_virtual_buffer_keep(loom_serve_virtual_buffer_t* buffer,
                                    iree_device_size_t offset,
                                    iree_device_size_t length);
iree_status_t loom_serve_virtual_buffer_trim(
    loom_serve_virtual_buffer_t* buffer);

// All device uses and borrowed HAL references must have retired. On a platform
// cleanup failure the object retains unreleased ownership and may be retried;
// the pool must remain alive. Null is accepted.
iree_status_t loom_serve_virtual_buffer_free(
    loom_serve_virtual_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_STORAGE_MEMORY_H_
