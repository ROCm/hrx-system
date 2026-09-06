// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Transient buffer: a reservation handle for queue-ordered task-driver
// allocations.
//
// Queue allocators can return a transient buffer to the caller before the
// physical backing is ready, then commit a real backing buffer once queue
// ordering allows it. queue_dealloca can later decommit the wrapper while the
// transient buffer object itself remains live so stale host references fail
// cleanly with IREE_STATUS_FAILED_PRECONDITION instead of accessing freed
// storage.

#ifndef IREE_HAL_DRIVERS_TASK_TRANSIENT_BUFFER_H_
#define IREE_HAL_DRIVERS_TASK_TRANSIENT_BUFFER_H_

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_task_transient_buffer_t
    iree_hal_task_transient_buffer_t;

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
iree_status_t iree_hal_task_transient_buffer_create(
    iree_hal_buffer_placement_t placement, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_device_size_t byte_length,
    iree_hal_pool_t* source_pool, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer);

// Returns true if |buffer| is a task-driver transient buffer wrapper.
bool iree_hal_task_transient_buffer_isa(const iree_hal_buffer_t* buffer);

// Returns the stable profiling id assigned to this transient buffer wrapper.
//
// The id is nonzero and unique across wrappers owned by this driver. Producers
// use it as a session-local allocation id during an active profile capture.
uint64_t iree_hal_task_transient_buffer_profile_id(iree_hal_buffer_t* buffer);

// Attaches a pool reservation to the transient buffer. |pool| must be the
// source pool captured at creation. The wrapper takes ownership until
// iree_hal_task_transient_buffer_release_reservation() or wrapper destroy.
void iree_hal_task_transient_buffer_attach_reservation(
    iree_hal_buffer_t* buffer, iree_hal_pool_t* pool,
    const iree_hal_pool_reservation_t* reservation);

// Stages a materialized backing buffer view for a future commit. The backing
// buffer is retained, but remains invisible to map/flush/invalidate calls until
// iree_hal_task_transient_buffer_commit() publishes it.
void iree_hal_task_transient_buffer_stage_backing(iree_hal_buffer_t* buffer,
                                                  iree_hal_buffer_t* backing);

// Publishes the staged backing buffer. Must be called exactly once while the
// wrapper is uncommitted and has a staged backing view.
void iree_hal_task_transient_buffer_commit(iree_hal_buffer_t* buffer);

// Decommits the backing buffer and returns the wrapper to the uncommitted
// state. Any staged-but-uncommitted backing view is also released. Queue
// implementations use this in the dealloca wait-satisfied/signal-before window
// so target-visible release effects occur before user-visible completion. Safe
// to call on an already-uncommitted wrapper.
void iree_hal_task_transient_buffer_decommit(iree_hal_buffer_t* buffer);

// Returns the attached pool reservation without transferring ownership.
//
// This is a cold diagnostic/profiling helper. Returns false when the wrapper
// has no live reservation or the reservation has already been released.
bool iree_hal_task_transient_buffer_query_reservation(
    iree_hal_buffer_t* buffer, iree_hal_pool_t** out_pool,
    iree_hal_pool_reservation_t* out_reservation);

// Marks the logical allocation epoch as captured by a queue deallocation.
// This may occur before the queue allocation has acquired a reservation.
// Returns the borrowed source pool without transferring reservation ownership.
// The mark excludes duplicate deallocations until it is either aborted or
// consumed by iree_hal_task_transient_buffer_take_dealloca_reservation().
iree_status_t iree_hal_task_transient_buffer_begin_dealloca(
    iree_hal_buffer_t* buffer, iree_hal_pool_t** out_pool);

// Restores an allocation epoch marked by begin_dealloca after queue capture
// fails. Has no effect if the epoch has already been consumed.
void iree_hal_task_transient_buffer_abort_dealloca(iree_hal_buffer_t* buffer);

// Consumes a deallocation mark and transfers its reservation to the caller.
// The source pool is borrowed and must outlive the reservation transaction.
void iree_hal_task_transient_buffer_take_dealloca_reservation(
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
void iree_hal_task_transient_buffer_release_reservation(
    iree_hal_buffer_t* buffer, const iree_async_frontier_t* death_frontier);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_TASK_TRANSIENT_BUFFER_H_
