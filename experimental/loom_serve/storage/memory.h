// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_STORAGE_MEMORY_H_
#define EXPERIMENTAL_LOOM_SERVE_STORAGE_MEMORY_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Single-owner physical allocation domain shared by independent virtual
// buffers. Logical blocks and model geometry belong to its callers. Mapping
// operations are synchronous capacity transitions, not per-token operations.
typedef struct loom_serve_memory_pool_t loom_serve_memory_pool_t;
typedef struct loom_serve_virtual_buffer_t loom_serve_virtual_buffer_t;

typedef struct loom_serve_memory_statistics_t {
  // Reserved virtual bytes, including final slab alignment.
  uint64_t reserved_bytes;
  // Unique physical bytes currently owned by the pool.
  uint64_t committed_bytes;
  // Maximum simultaneously committed physical bytes.
  uint64_t peak_bytes;
  // Cumulative physical bytes successfully returned to the allocator.
  uint64_t released_bytes;
} loom_serve_memory_statistics_t;

// Retains allocator. Zero slab_size selects the allocator's recommended
// granularity; explicit sizes must be multiples of its minimum granularity.
// limit_bytes bounds physical commitment, not virtual reservation. Zero means
// no serving-imposed limit; allocation failure still propagates. Each buffer
// borrows this pool, and must be destroyed before pool destruction.
iree_status_t loom_serve_memory_pool_create(
    iree_hal_allocator_t* device_allocator,
    iree_hal_queue_family_affinity_t queue_affinity,
    iree_device_size_t slab_size, uint64_t limit_bytes,
    loom_serve_memory_pool_t** out_pool, iree_allocator_t host_allocator);

void loom_serve_memory_pool_destroy(loom_serve_memory_pool_t* pool);

loom_serve_memory_statistics_t loom_serve_memory_pool_statistics(
    const loom_serve_memory_pool_t* pool);

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

// Makes room for additional physical bytes using the installed cold policy.
// Does not allocate or reserve credit: the single owner commits before yielding
// admission to another caller. False means ordinary capacity backpressure, not
// an allocation failure. Consumers pin every residency they need before asking
// for capacity, including already backed ranges the policy could otherwise
// trim.
iree_status_t loom_serve_memory_pool_prepare(loom_serve_memory_pool_t* pool,
                                             uint64_t additional_bytes,
                                             bool* out_admitted);

// Reserves stable device addresses without committing physical storage.
// The borrowed HAL buffer remains identical across commit/trim operations.
// statistics is a caller-owned, initially zeroed accounting group shared by
// related reservations. It outlives every reservation using it. The pool also
// accounts all reservations independently, enforcing their common budget.
iree_status_t loom_serve_virtual_buffer_create(
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

// At a retired maintenance cut, clear the keep set, mark all still-live byte
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
iree_status_t loom_serve_virtual_buffer_destroy(
    loom_serve_virtual_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_STORAGE_MEMORY_H_
