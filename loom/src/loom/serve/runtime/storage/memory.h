// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_STORAGE_MEMORY_H_
#define LOOM_SERVE_RUNTIME_STORAGE_MEMORY_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Single-owner physical allocation domain shared by independent virtual
// buffers. Logical blocks and model geometry belong to its callers. Admission
// and result publication stay on that owner; physical and mapping operations
// use public HAL queues. These are capacity transitions, not per-token work.
// Backing supports storage, transfers, and indirect dispatch parameters.
typedef struct loom_serve_memory_pool_t loom_serve_memory_pool_t;
typedef struct loom_serve_virtual_buffer_t loom_serve_virtual_buffer_t;
typedef struct loom_serve_memory_operation_t loom_serve_memory_operation_t;

typedef struct loom_serve_memory_statistics_t {
  // Reserved virtual bytes, including final slab alignment.
  uint64_t reserved_bytes;
  // Unique backing bytes; pool-wide queries include pending native retirement.
  uint64_t committed_bytes;
  // Budget charged to accepted native commitments not yet published by owner.
  uint64_t pending_bytes;
  // Backing selected for return. Per-buffer accounting ends at logical return;
  // pool-wide accounting also includes outstanding native release.
  uint64_t releasing_bytes;
  // Maximum simultaneously committed physical bytes.
  uint64_t peak_bytes;
  // Cumulative returned backing; pool-wide queries exclude pending native free.
  uint64_t released_bytes;
} loom_serve_memory_statistics_t;

// Retains the mapping-source pool and copies the exact virtual-target scope.
// The sealed group and selected transfer queue outlive this pool. Zero
// slab_size selects the common recommended granularity; explicit sizes satisfy
// both source and target geometry. limit_bytes bounds physical commitment,
// including native releases not yet completed. Zero imposes no serving budget.
// Each reservation borrows this pool and is destroyed before the pool.
iree_status_t loom_serve_memory_pool_allocate(
    iree_hal_device_group_t* group, iree_hal_pool_scope_t scope,
    iree_hal_pool_t* storage_pool, iree_hal_queue_t* queue,
    iree_device_size_t slab_size, uint64_t limit_bytes,
    iree_allocator_t host_allocator, loom_serve_memory_pool_t** out_pool);

void loom_serve_memory_pool_free(loom_serve_memory_pool_t* pool);

loom_serve_memory_statistics_t loom_serve_memory_pool_statistics(
    loom_serve_memory_pool_t* pool);

// Immutable physical commitment ceiling in bytes; UINT64_MAX is unbounded.
// This is the configured budget, not currently available backing.
uint64_t loom_serve_memory_pool_limit(const loom_serve_memory_pool_t* pool);

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
void loom_serve_memory_pool_set_observer(loom_serve_memory_pool_t* pool,
                                         void (*wake)(void* user_data),
                                         void* user_data);

// Collects completed operations and reports remaining native borrowers.
bool loom_serve_memory_pool_is_pending(loom_serve_memory_pool_t* pool);

// Outer-thread join of accepted native operations, not device work or logical
// reclamation that still needs its model scheduler. Never called by task
// owners. Publishes final accounting and returns any shared capacity or
// reclamation failure. Each operation retains its own result.
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

// Makes room for additional physical bytes using the installed cold policy.
// Includes pending commitments but does not itself allocate or reserve credit:
// the single owner commits before yielding admission to another caller.
// Consumers pin every residency they need before asking for capacity,
// including already backed ranges the policy could otherwise trim. Accepted
// returns grant capacity only after native completion. The result preserves
// this attempt's progress observation independently of model retirement.
iree_status_t loom_serve_memory_pool_prepare(
    loom_serve_memory_pool_t* pool, uint64_t additional_bytes,
    loom_serve_memory_admission_t* out_admission);

// Reserves stable device addresses without committing physical storage.
// The borrowed HAL buffer remains identical across commit/trim operations.
// statistics is a caller-owned, initially zeroed accounting group shared by
// related reservations. It outlives every reservation using it. The pool also
// accounts all reservations independently, enforcing their common budget.
iree_status_t loom_serve_virtual_buffer_allocate(
    loom_serve_memory_pool_t* pool, iree_device_size_t length,
    iree_device_size_t alignment, loom_serve_memory_statistics_t* statistics,
    loom_serve_virtual_buffer_t** out_buffer);
iree_hal_buffer_t* loom_serve_virtual_buffer_handle(
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
// ranges need no operation and signal immediately. Capacity denial returns
// admitted=false and no operation or readiness signal.
//
// The caller keeps every reservation and its accounting alive until the
// returned operation retires, independently of readiness success/failure. No
// other commitment or trim may touch its pending slabs. Ranges are copied; the
// readiness semaphore is retained. Completion may precede return.
iree_status_t loom_serve_memory_pool_commit_async(
    loom_serve_memory_pool_t* pool, iree_host_size_t range_count,
    const loom_serve_memory_range_t* ranges, iree_hal_semaphore_t* readiness,
    uint64_t value, bool* out_admitted,
    loom_serve_memory_operation_t** out_operation);

// Copies the unkept physical slab set of these distinct reservations and queues
// its native release. The caller has retired every consumer of that set and
// keeps reservations/accounting alive through operation completion. Kept slabs
// remain usable. No other operation may edit selected slabs before completion.
// The private mapping and deallocation completions join before optional
// readiness is signaled. Physical capacity stays charged by the source pool
// until native release completes; its capacity notification wakes admission.
// An empty set returns NULL and signals immediately.
// Native failures fail readiness, preserve unreleased ownership, and poison
// the pool for normal owner-driven shutdown.
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
// True joins captured queue ownership, independently of caller readiness. NULL
// is complete. A failed operation is reported by operation_query_status and
// readiness.
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
