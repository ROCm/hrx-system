// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Transient buffer: a reservation handle for queue-ordered Vulkan allocations.
//
// Queue allocators can return a transient buffer to the caller before the
// physical backing is ready, then commit a real backing buffer once queue
// ordering allows it. queue_dealloca can later decommit the wrapper while the
// transient buffer object itself remains live so stale host references fail
// cleanly with IREE_STATUS_FAILED_PRECONDITION instead of accessing freed
// storage.

#ifndef IREE_HAL_DRIVER_VULKAN_TRANSIENT_BUFFER_H_
#define IREE_HAL_DRIVER_VULKAN_TRANSIENT_BUFFER_H_

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_vulkan_transient_buffer_t
    iree_hal_vulkan_transient_buffer_t;

// Creates a transient buffer with the given metadata but no backing memory.
// The buffer starts in the uncommitted state.
//
// |allocation_size| is the physical reservation size to report through
// iree_hal_buffer_allocation_size() and |byte_length| is the logical byte range
// exposed by the wrapper. |byte_length| must be <= |allocation_size|.
//
// |source_pool| is borrowed for the complete logical allocation lifetime and
// may be queried before queue ordering allows a physical reservation to be
// acquired.
iree_status_t iree_hal_vulkan_transient_buffer_create(
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_length, iree_hal_pool_t* source_pool,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Returns true if |buffer| is a Vulkan transient buffer wrapper.
bool iree_hal_vulkan_transient_buffer_isa(const iree_hal_buffer_t* buffer);

// Attaches a pool reservation to the transient buffer. |pool| must be the
// source pool captured at creation. The wrapper takes ownership until
// iree_hal_vulkan_transient_buffer_release_reservation() or wrapper destroy.
void iree_hal_vulkan_transient_buffer_attach_reservation(
    iree_hal_buffer_t* buffer, iree_hal_pool_t* pool,
    const iree_hal_pool_reservation_t* reservation);

// Stages a materialized backing buffer view for a future commit. The backing
// buffer is retained, but remains invisible to map/flush/invalidate calls until
// iree_hal_vulkan_transient_buffer_commit() publishes it.
// Native binding facts are published immediately for queue packet emission;
// dependent queue operations order their use with allocation semaphore edges.
void iree_hal_vulkan_transient_buffer_stage_backing(iree_hal_buffer_t* buffer,
                                                    iree_hal_buffer_t* backing);

// Publishes the staged backing buffer. If a causally-later deallocation already
// completed on another queue, the staged backing has been discarded and this
// is a no-op.
void iree_hal_vulkan_transient_buffer_commit(iree_hal_buffer_t* buffer);

// Returns true if queue_dealloca has been accepted for the transient buffer.
bool iree_hal_vulkan_transient_buffer_is_dealloca_queued(
    iree_hal_buffer_t* buffer);

// Transfers the attached reservation to the caller without changing the
// deallocation state. Used to roll back an allocation transaction that fails
// after materialization but before queue acceptance.
void iree_hal_vulkan_transient_buffer_take_reservation(
    iree_hal_buffer_t* buffer, iree_hal_pool_t** out_pool,
    iree_hal_pool_reservation_t* out_reservation);

// Returns the staged backing buffer used for queue packet emission, or NULL if
// no backing has been staged. This is intentionally less strict than committed
// host access: queue submissions may resolve backing storage before the alloca
// signal is visible to users, relying on semaphore dependencies for ordering.
iree_hal_buffer_t* iree_hal_vulkan_transient_buffer_backing_buffer(
    iree_hal_buffer_t* buffer);

// Returns the attached pool reservation without transferring ownership.
//
// This is a cold diagnostic/profiling helper. Returns false when the wrapper
// has no live reservation or the reservation has already been released.
bool iree_hal_vulkan_transient_buffer_query_reservation(
    iree_hal_buffer_t* buffer, iree_hal_pool_t** out_pool,
    iree_hal_pool_reservation_t* out_reservation);

// Releases the attached reservation exactly once. If the wrapper has no
// reservation or the reservation was already released, this is a no-op.
//
// |death_frontier| is forwarded to iree_hal_pool_release_reservations() when
// the reservation is still owned. Pass NULL for an immediately reusable
// reservation in synchronous/drain-ordered paths. This updates pool reuse
// metadata; target-visible release effects are performed by the queue before
// publishing dealloca completion.
void iree_hal_vulkan_transient_buffer_release_reservation(
    iree_hal_buffer_t* buffer, const iree_async_frontier_t* death_frontier);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_VULKAN_TRANSIENT_BUFFER_H_
