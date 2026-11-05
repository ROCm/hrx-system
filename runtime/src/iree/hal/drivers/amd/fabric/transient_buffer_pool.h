// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_FABRIC_TRANSIENT_BUFFER_POOL_H_
#define IREE_HAL_DRIVERS_AMD_FABRIC_TRANSIENT_BUFFER_POOL_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_amd_fabric_transient_buffer_chunk_t
    iree_hal_amd_fabric_transient_buffer_chunk_t;

enum {
  // Slots keep independently modified roots on separate cache lines.
  IREE_HAL_AMD_FABRIC_TRANSIENT_BUFFER_MIN_SHIFT = 6,
  // Every representable power-of-two slot size has one directly indexed list.
  IREE_HAL_AMD_FABRIC_TRANSIENT_BUFFER_CLASS_COUNT =
      sizeof(iree_host_size_t) * 8 -
      IREE_HAL_AMD_FABRIC_TRANSIENT_BUFFER_MIN_SHIFT,
};

// Independently lived roots and copied reuse histories in bitmap-managed
// chunks. Cold growth uses the host allocator; warm slot reuse does not.
// The application-held device dominates every slot and its owning pool.
typedef struct iree_hal_amd_fabric_transient_buffer_pool_t {
  // Allocator used only to grow and retire whole chunks.
  iree_allocator_t host_allocator;
  // Serializes acquisition, available-list publication, and chunk retirement.
  iree_slim_mutex_t mutex;
  // Coalesces cache misses; cached acquisition and release never take it.
  iree_slim_mutex_t growth_mutex;
  // Chunks with available slots, directly indexed by slot size.
  iree_hal_amd_fabric_transient_buffer_chunk_t*
      available[IREE_HAL_AMD_FABRIC_TRANSIENT_BUFFER_CLASS_COUNT];
  // Allocated chunks, including full chunks absent from available lists.
  iree_host_size_t chunk_count;
} iree_hal_amd_fabric_transient_buffer_pool_t;

// A contiguous run of owned slots. Each slot is independently released, even
// when other slots from the same capture remain live.
typedef struct iree_hal_amd_fabric_transient_buffer_slots_t {
  // Borrowed owner, kept live by every occupied slot.
  iree_hal_amd_fabric_transient_buffer_chunk_t* chunk;
  // First slot's storage, aligned to a cache line.
  uint8_t* data;
  // Bytes available in each slot and distance to its next neighbor.
  iree_host_size_t stride;
  // Number of acquired slots, no greater than the requested count.
  iree_host_size_t count;
} iree_hal_amd_fabric_transient_buffer_slots_t;

void iree_hal_amd_fabric_transient_buffer_pool_initialize(
    iree_allocator_t host_allocator,
    iree_hal_amd_fabric_transient_buffer_pool_t* out_pool);

// All callers and slots have retired before deinitialization.
void iree_hal_amd_fabric_transient_buffer_pool_deinitialize(
    iree_hal_amd_fabric_transient_buffer_pool_t* pool);

// Retires currently empty chunks; concurrent acquisition and release are safe.
void iree_hal_amd_fabric_transient_buffer_pool_trim(
    iree_hal_amd_fabric_transient_buffer_pool_t* pool);

// Acquires at least one slot with |byte_length| capacity, up to |max_count|.
// Both inputs are nonzero. Failure leaves out_slots empty.
iree_status_t iree_hal_amd_fabric_transient_buffer_pool_acquire(
    iree_hal_amd_fabric_transient_buffer_pool_t* pool,
    iree_host_size_t byte_length, iree_host_size_t max_count,
    iree_hal_amd_fabric_transient_buffer_slots_t* out_slots);

// Releases one complete slot after its final user has retired.
void iree_hal_amd_fabric_transient_buffer_pool_release(
    iree_hal_amd_fabric_transient_buffer_chunk_t* chunk, void* data);

iree_host_size_t iree_hal_amd_fabric_transient_buffer_slot_capacity(
    const iree_hal_amd_fabric_transient_buffer_chunk_t* chunk);
iree_hal_amd_fabric_transient_buffer_pool_t*
iree_hal_amd_fabric_transient_buffer_slot_pool(
    const iree_hal_amd_fabric_transient_buffer_chunk_t* chunk);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_FABRIC_TRANSIENT_BUFFER_POOL_H_
