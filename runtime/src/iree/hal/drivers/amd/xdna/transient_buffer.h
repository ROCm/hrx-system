// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Transient buffer: a reservation handle for queue-ordered XDNA
// allocations.
//
// Queue allocators can return a transient buffer to the caller before the
// physical backing is ready, then commit a real backing buffer once queue
// ordering allows it. queue_dealloca can later decommit the wrapper while the
// transient buffer object itself remains live so stale host references fail
// cleanly with IREE_STATUS_FAILED_PRECONDITION instead of accessing freed
// storage.

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_TRANSIENT_BUFFER_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_TRANSIENT_BUFFER_H_

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amd_xdna_transient_buffer_t
    iree_hal_amd_xdna_transient_buffer_t;

// Creates a transient buffer with the given metadata but no backing memory.
// The buffer starts in the uncommitted state.
//
// |allocation_size| is the physical reservation size to report through
// iree_hal_buffer_allocation_size() and |byte_length| is the logical byte range
// exposed by the wrapper. |byte_length| must be <= |allocation_size|.
//
// |source_pool| is borrowed for the complete logical allocation lifetime and
// may be queried before queue ordering allows a physical reservation to be
// acquired. |block_pool| supplies one reusable fixed-size wrapper block. The
// placement device is retained until that block is returned, keeping the queue
// storage owner live without tracking individual buffers.
iree_status_t iree_hal_amd_xdna_transient_buffer_create(
    iree_hal_buffer_placement_t placement, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_device_size_t byte_length,
    iree_hal_pool_t* source_pool, iree_arena_block_pool_t* block_pool,
    iree_hal_buffer_t** out_buffer);

// Attaches a pool reservation to the transient buffer. |pool| must be the
// source pool captured at creation. The wrapper takes ownership until
// queue deallocation or wrapper destroy.
void iree_hal_amd_xdna_transient_buffer_attach_reservation(
    iree_hal_buffer_t* buffer, iree_hal_pool_t* pool,
    const iree_hal_pool_reservation_t* reservation);

// Stages an existing reservation view for a future commit. The backing buffer
// is retained, but remains invisible to map/flush/invalidate calls until
// iree_hal_amd_xdna_transient_buffer_commit() publishes it. The remaining view
// fields borrow the attached reservation and source pool.
void iree_hal_amd_xdna_transient_buffer_stage_backing(
    iree_hal_buffer_t* buffer, const iree_hal_pool_reservation_view_t* backing);

// Publishes the staged backing buffer. Must be called exactly once while the
// wrapper is uncommitted and has a staged backing view.
void iree_hal_amd_xdna_transient_buffer_commit(iree_hal_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_TRANSIENT_BUFFER_H_
