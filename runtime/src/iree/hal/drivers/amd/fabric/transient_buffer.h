// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Transient buffer: a reservation handle for queue-ordered AMD allocations.
//
// Queue allocators can return a transient buffer to the caller before the
// physical backing is ready, then commit a real backing buffer once queue
// ordering allows it. queue_dealloca can later decommit the wrapper while the
// transient buffer object itself remains live so stale host references fail
// cleanly with IREE_STATUS_FAILED_PRECONDITION instead of accessing freed
// storage.

#ifndef IREE_HAL_DRIVERS_AMD_FABRIC_TRANSIENT_BUFFER_H_
#define IREE_HAL_DRIVERS_AMD_FABRIC_TRANSIENT_BUFFER_H_

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amd/fabric/transient_buffer_pool.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amd_fabric_transient_buffer_t
    iree_hal_amd_fabric_transient_buffer_t;

// Creates uncommitted roots for the already-validated allocation requests.
// Captures slot runs from |buffer_pool|, independently owned by each root.
// |out_buffers| starts zeroed. On failure, its initialized prefix remains owned
// by the caller and all remaining entries stay NULL.
//
// |source_pool| is borrowed for the complete logical allocation lifetime and
// may be queried before queue ordering allows a physical reservation to be
// acquired. The application's device ownership keeps |buffer_pool| alive
// through the roots' final releases, including independently held subviews.
iree_status_t iree_hal_amd_fabric_transient_buffer_create_batch(
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_hal_pool_t* source_pool,
    iree_hal_amd_fabric_transient_buffer_pool_t* buffer_pool,
    iree_host_size_t count, const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** out_buffers);

// Attaches a pool reservation to the transient buffer. |pool| must be the
// source pool captured at creation. The wrapper takes ownership until
// queue deallocation or wrapper destroy.
void iree_hal_amd_fabric_transient_buffer_attach_reservation(
    iree_hal_buffer_t* buffer, iree_hal_pool_t* pool,
    const iree_hal_pool_reservation_t* reservation);

// Stages an existing reservation view for a future commit. The backing buffer
// is retained, but remains invisible to map/flush/invalidate calls until
// iree_hal_amd_fabric_transient_buffer_commit() publishes it. Reuse history is
// copied so logical reservation return cannot invalidate the old root's view.
// Ordinary roots hold history in their slot's spare capacity; larger histories
// use another slot and can fail cold growth before staging any backing.
iree_status_t iree_hal_amd_fabric_transient_buffer_stage_backing(
    iree_hal_buffer_t* buffer, const iree_hal_pool_reservation_view_t* backing);

// Publishes the staged backing buffer. Must be called exactly once while the
// wrapper is uncommitted and has a staged backing view.
void iree_hal_amd_fabric_transient_buffer_commit(iree_hal_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_FABRIC_TRANSIENT_BUFFER_H_
