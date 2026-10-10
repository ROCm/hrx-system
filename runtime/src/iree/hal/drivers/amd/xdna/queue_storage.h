// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_STORAGE_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_STORAGE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_amd_xdna_queue_payload_chunk_t
    iree_hal_amd_xdna_queue_payload_chunk_t;

// Captured bytes split across reusable fixed-size blocks.
typedef struct iree_hal_amd_xdna_queue_payload_t {
  // First chunk in capture order, or NULL for an empty payload.
  iree_hal_amd_xdna_queue_payload_chunk_t* head;
  // Total number of captured bytes across all chunks.
  iree_host_size_t data_length;
} iree_hal_amd_xdna_queue_payload_t;

// Reusable fixed-block storage shared by every operation on one queue.
typedef struct iree_hal_amd_xdna_queue_storage_t {
  // Small blocks holding operation metadata and bounded descriptor arrays.
  iree_arena_block_pool_t metadata_block_pool;
  // Large blocks holding captured host payload bytes.
  iree_arena_block_pool_t payload_block_pool;
} iree_hal_amd_xdna_queue_storage_t;

// Per-operation ownership of blocks acquired from queue storage.
typedef struct iree_hal_amd_xdna_queue_capture_t {
  // Arena owning the operation and its metadata arrays.
  iree_arena_allocator_t metadata_arena;
  // Arena owning chunked captured host payloads.
  iree_arena_allocator_t payload_arena;
} iree_hal_amd_xdna_queue_capture_t;

// Initializes empty queue storage using |host_allocator| for cold growth.
void iree_hal_amd_xdna_queue_storage_initialize(
    iree_allocator_t host_allocator,
    iree_hal_amd_xdna_queue_storage_t* out_storage);

// Releases every unused block. All captures must already be deinitialized.
void iree_hal_amd_xdna_queue_storage_deinitialize(
    iree_hal_amd_xdna_queue_storage_t* storage);

// Releases currently unused blocks while preserving active captures.
void iree_hal_amd_xdna_queue_storage_trim(
    iree_hal_amd_xdna_queue_storage_t* storage);

// Initializes an empty per-operation capture borrowing |storage|.
void iree_hal_amd_xdna_queue_capture_initialize(
    iree_hal_amd_xdna_queue_storage_t* storage,
    iree_hal_amd_xdna_queue_capture_t* out_capture);

// Allocates one contiguous metadata record. Requests larger than a fixed block
// are rejected instead of falling through to a per-operation system allocation.
iree_status_t iree_hal_amd_xdna_queue_capture_allocate_metadata(
    iree_hal_amd_xdna_queue_capture_t* capture, iree_host_size_t byte_length,
    void** out_ptr);

// Copies |source| into fixed-size payload chunks owned by |capture|.
iree_status_t iree_hal_amd_xdna_queue_capture_payload(
    iree_hal_amd_xdna_queue_capture_t* capture, iree_const_byte_span_t source,
    iree_hal_amd_xdna_queue_payload_t* out_payload);

// Copies a complete captured payload into |target| in capture order.
// |target| must have at least payload.data_length writable bytes.
void iree_hal_amd_xdna_queue_payload_copy(
    const iree_hal_amd_xdna_queue_payload_t* payload, void* target);

// Returns all captured blocks to their queue-owned pools. The capture and
// every allocation made from it become invalid.
void iree_hal_amd_xdna_queue_capture_deinitialize(
    iree_hal_amd_xdna_queue_capture_t* capture);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_STORAGE_H_
