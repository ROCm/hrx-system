// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_FIXED_BLOCK_POOL_SLAB_H_
#define IREE_HAL_MEMORY_FIXED_BLOCK_POOL_SLAB_H_

#include "iree/hal/memory/buffer_range.h"
#include "iree/hal/memory/fixed_block_allocator.h"
#include "iree/hal/memory/pool_history.h"
#include "iree/hal/pool.h"

typedef struct iree_hal_fixed_block_pool_slab_t
    iree_hal_fixed_block_pool_slab_t;

// Stable reservation identity, allocated with the owning slab's metadata.
typedef struct iree_hal_fixed_block_pool_block_t {
  // Owning prepared range; the block's index is its position in slab->blocks.
  iree_hal_fixed_block_pool_slab_t* slab;
} iree_hal_fixed_block_pool_block_t;

// Immutable geometry resolved once against captured storage capabilities.
typedef struct iree_hal_fixed_block_pool_geometry_t {
  // User-visible capacity of each block.
  iree_device_size_t block_size;
  // Physical stride including hidden padding and protection.
  iree_device_size_t backing_block_size;
  // Guaranteed block alignment, also required of the parent reservation.
  iree_device_size_t alignment;
  // Number of blocks requested on each growth operation.
  uint32_t blocks_per_slab;
  // Exact history capacity for every block and whole-range return.
  uint8_t frontier_capacity;
} iree_hal_fixed_block_pool_geometry_t;

// One ordinary backing reservation and its fixed-block offset allocator.
struct iree_hal_fixed_block_pool_slab_t {
  // Previous slab in the pool's owned inventory, guarded by acquisition_mutex.
  iree_hal_fixed_block_pool_slab_t* previous;
  // Next slab in the pool's owned inventory, guarded by acquisition_mutex.
  iree_hal_fixed_block_pool_slab_t* next;
  // Changed-range work link, guarded by the pool's publication mutex.
  iree_hal_fixed_block_pool_slab_t* candidate_next;
  // Link in the creating transaction's preparation pins.
  iree_hal_fixed_block_pool_slab_t* preparation_next;
  // Whether changed-range work owns this slab, guarded by publication_mutex.
  bool candidate_queued;
  // This backing owns one presecured overflow return credit in the pool.
  bool history_owned;
  // Snapshot and preparation pins, guarded by acquisition_mutex.
  uint32_t pins;
  // Live reservations, including a releasing thread's final slab access.
  iree_atomic_int32_t live_count;
  // Parent token held until all block histories can be returned together.
  iree_hal_pool_reservation_t reservation;
  // Prepared parent view retained independently of the explicit token.
  iree_hal_pool_buffer_range_t range;
  // Offset allocator retaining each block's exact history.
  iree_hal_memory_fixed_block_allocator_t* allocator;
  // Stable identities for the allocator's blocks, in this metadata allocation.
  iree_hal_fixed_block_pool_block_t* blocks;
  // Per-block protection geometry, NULL when ASAN is disabled.
  iree_hal_asan_allocation_layout_t* asan_layouts;
  // Merged return history, with geometry.frontier_capacity inline entries.
  iree_async_frontier_t* return_frontier;
  // Number of managed blocks; any remaining bytes retain source history.
  uint32_t block_count;
};

#ifdef __cplusplus
extern "C" {
#endif

// Consumes an acquired parent token, including on failure. Preparation and
// rollback preserve its exact history and run outside acquisition locks.
// Consumes optional detached |reusable_slab| metadata, including on failure.
iree_status_t iree_hal_fixed_block_pool_slab_prepare(
    iree_hal_pool_t* backing_pool,
    const iree_hal_pool_reservation_request_t* request,
    const iree_hal_fixed_block_pool_geometry_t* geometry,
    const iree_hal_asan_pool_options_t* asan,
    iree_hal_pool_reservation_t reservation, iree_hal_pool_acquire_info_t info,
    const iree_async_frontier_t* initial_frontier,
    iree_allocator_t host_allocator,
    iree_hal_fixed_block_pool_slab_t* reusable_slab,
    iree_hal_fixed_block_pool_slab_t** out_slab);

// Retains a finite range and prepares its block metadata without a parent
// token.
iree_status_t iree_hal_fixed_block_pool_slab_allocate(
    const iree_hal_pool_buffer_range_t* range,
    const iree_hal_fixed_block_pool_geometry_t* geometry,
    const iree_hal_asan_pool_options_t* asan,
    const iree_async_frontier_t* initial_frontier,
    iree_allocator_t host_allocator,
    iree_hal_fixed_block_pool_slab_t** out_slab);

// Merges every free block and any untouched margins. The detached slab has no
// live reservations or candidate readers. False leaves all block histories
// intact so the caller can retain the slab for individual block reuse.
bool iree_hal_fixed_block_pool_slab_merge_return(
    iree_hal_fixed_block_pool_slab_t* slab,
    const iree_hal_fixed_block_pool_geometry_t* geometry,
    iree_hal_pool_history_t* history);

// Releases the prepared view and parent token while retaining free metadata.
// The caller has detached this slab from acquisition and release readers.
void iree_hal_fixed_block_pool_slab_release_backing(
    iree_hal_fixed_block_pool_slab_t* slab, iree_hal_pool_t* backing_pool);

// Releases the view, returns an optional parent token with its prepared return
// history, and destroys metadata. Runs outside pool mutation locks.
void iree_hal_fixed_block_pool_slab_free(iree_hal_fixed_block_pool_slab_t* slab,
                                         iree_hal_pool_t* backing_pool,
                                         iree_allocator_t host_allocator);

#ifdef __cplusplus
}
#endif

#endif  // IREE_HAL_MEMORY_FIXED_BLOCK_POOL_SLAB_H_
