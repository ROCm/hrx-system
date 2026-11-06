// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_STORAGE_VIRTUAL_POOL_H_
#define LOOM_RUNTIME_STORAGE_VIRTUAL_POOL_H_

#include "iree/base/threading/notification.h"
#include "loom/runtime/storage/memory.h"
#include "loom/runtime/wake.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prepared virtual ranges backing a logical allocator. Physical storage belongs
// exclusively to memory_pool; this pool never acquires native slabs itself.
// Range addresses survive physical trim. Queue callbacks may acquire/release
// ranges concurrently, while advance/free share memory_pool's serialized
// owner. Cold callbacks publish requests and wake that owner; ready reuse
// requires neither owner progress nor allocation.
typedef struct loom_runtime_virtual_pool_t loom_runtime_virtual_pool_t;

// Reserves one initial address range to qualify the immutable prepared-storage
// contract. No physical storage is committed. The physical owner outlives this
// pool. geometry defines physical slabs, not logical reservation sizes.
iree_status_t loom_runtime_virtual_pool_allocate(
    loom_runtime_memory_pool_t* memory_pool,
    loom_runtime_memory_geometry_t geometry, iree_allocator_t host_allocator,
    loom_runtime_virtual_pool_t** out_pool);

// Borrowed default HAL source for TLSF or another logical allocator.
iree_hal_pool_t* loom_runtime_virtual_pool_handle(
    loom_runtime_virtual_pool_t* pool);

// Creates an independently accounted HAL source on this physical owner. The
// returned interface is borrowed until pool destruction; logical allocators
// retain it. Sources share cold progress, physical admission, and
// notifications, but retain independent range reuse and trim policy. Call on
// the exclusive outer owner before publishing sources to concurrent clients.
iree_status_t loom_runtime_virtual_pool_create_source(
    loom_runtime_virtual_pool_t* pool, iree_hal_pool_t** out_source);

// Rounds logical bytes to this source's physical slab granularity. Saturates
// at UINT64_MAX when the rounded extent is not representable; no allocation
// or availability query occurs.
uint64_t loom_runtime_virtual_pool_round_size(
    const loom_runtime_virtual_pool_t* pool, uint64_t byte_length);

// Copies aggregate usage across this owner's sources without collecting native
// work or querying logical allocators. A subset of the physical owner's totals.
loom_runtime_storage_usage_t loom_runtime_virtual_pool_memory_usage(
    const loom_runtime_virtual_pool_t* pool);

// Copies one source's physical usage on the serialized owner. source is an
// interface returned by handle or create_source, not a child logical allocator.
// Disjoint source accounting keeps permanent backing separate from workspace
// whose realized bytes already fulfill protected execution credit.
loom_runtime_storage_usage_t loom_runtime_virtual_pool_source_memory_usage(
    const iree_hal_pool_t* source);

// Charged backing that cannot currently be reclaimed from this source. Includes
// unpublished acquisitions and failed ranges, excludes idle reusable ranges and
// accepted returns. A trim still awaiting consumer history remains protected.
// Called on the serialized owner; performs no allocation, native query, or
// scan.
uint64_t loom_runtime_virtual_pool_source_protected_bytes(
    const iree_hal_pool_t* source);

// Installs the owner's nonblocking wake. Detachment joins callback access.
// A blocking outer caller may instead use wait below. Neither callback may
// invoke advance, free, or another operation on the physical owner.
void loom_runtime_virtual_pool_set_observer(
    loom_runtime_virtual_pool_t* pool, loom_runtime_wake_callback_t observer);

// Advances accepted cold allocations and physical trims without waiting.
// Allocation errors complete their original HAL requests. An unowned trim
// failure propagates here and preserves the range until outer teardown.
iree_status_t loom_runtime_virtual_pool_advance(
    loom_runtime_virtual_pool_t* pool);

// Requests and advances all sources' idle backing returns on the physical
// owner, without scheduling that already-running owner. No cold acquisition or
// recursive admission is attempted. Pending history and native detaches finish
// through the same observer and subsequent owner calls.
iree_status_t loom_runtime_virtual_pool_reclaim(
    loom_runtime_virtual_pool_t* pool);
bool loom_runtime_virtual_pool_is_reclaiming(loom_runtime_virtual_pool_t* pool);

// Whether cold requests, mapping operations, or requested trims need owner
// progress. Active reservations and idle, retained backing are not pending.
bool loom_runtime_virtual_pool_is_pending(loom_runtime_virtual_pool_t* pool);

// Blocking outer-owner wait that also advances cold backing. Shared task
// schedulers use advance and their normal semaphore observation instead.
// The caller is the physical owner and excludes concurrent owner operations.
iree_status_t loom_runtime_virtual_pool_wait(loom_runtime_virtual_pool_t* pool,
                                             iree_hal_semaphore_t* semaphore,
                                             uint64_t value);

// Outer ownership join that keeps cold backing progressing after readiness
// failure. condition observes final retirement and its producer signals
// notification. Progress errors are returned only after condition is true:
// they cannot release borrowed payloads early. The exclusive physical owner
// calls this instead of waiting on notification without servicing cold work.
iree_status_t loom_runtime_virtual_pool_await(loom_runtime_virtual_pool_t* pool,
                                              iree_notification_t* notification,
                                              iree_condition_fn_t condition,
                                              void* user_data);

// Outer teardown after all clients and accepted callbacks have retired.
// Releases backing and addresses; cleanup failure retains ownership for retry.
// On success consumes this pool's initial HAL reference. NULL is accepted.
iree_status_t loom_runtime_virtual_pool_free(loom_runtime_virtual_pool_t* pool);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_STORAGE_VIRTUAL_POOL_H_
