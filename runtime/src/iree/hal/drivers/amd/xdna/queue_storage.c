// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/queue_storage.h"

enum {
  IREE_HAL_AMD_XDNA_METADATA_BLOCK_SIZE = 4096,
  IREE_HAL_AMD_XDNA_PAYLOAD_BLOCK_SIZE = 32768,
};

struct iree_hal_amd_xdna_queue_payload_chunk_t {
  // Next chunk in capture order, or NULL for the final chunk.
  iree_hal_amd_xdna_queue_payload_chunk_t* next;
  // Number of initialized payload bytes in |data|.
  iree_host_size_t data_length;
  // Captured bytes beginning at their original logical offset.
  uint8_t data[];
};

void iree_hal_amd_xdna_queue_storage_initialize(
    iree_allocator_t host_allocator,
    iree_hal_amd_xdna_queue_storage_t* out_storage) {
  iree_arena_block_pool_initialize(IREE_HAL_AMD_XDNA_METADATA_BLOCK_SIZE,
                                   host_allocator,
                                   &out_storage->metadata_block_pool);
  iree_arena_block_pool_initialize(IREE_HAL_AMD_XDNA_PAYLOAD_BLOCK_SIZE,
                                   host_allocator,
                                   &out_storage->payload_block_pool);
}

void iree_hal_amd_xdna_queue_storage_deinitialize(
    iree_hal_amd_xdna_queue_storage_t* storage) {
  iree_arena_block_pool_deinitialize(&storage->payload_block_pool);
  iree_arena_block_pool_deinitialize(&storage->metadata_block_pool);
}

void iree_hal_amd_xdna_queue_storage_trim(
    iree_hal_amd_xdna_queue_storage_t* storage) {
  iree_arena_block_pool_trim(&storage->payload_block_pool);
  iree_arena_block_pool_trim(&storage->metadata_block_pool);
}

void iree_hal_amd_xdna_queue_capture_initialize(
    iree_hal_amd_xdna_queue_storage_t* storage,
    iree_hal_amd_xdna_queue_capture_t* out_capture) {
  iree_arena_initialize(&storage->metadata_block_pool,
                        &out_capture->metadata_arena);
  iree_arena_initialize(&storage->payload_block_pool,
                        &out_capture->payload_arena);
}

iree_status_t iree_hal_amd_xdna_queue_capture_allocate_metadata(
    iree_hal_amd_xdna_queue_capture_t* capture, iree_host_size_t byte_length,
    void** out_ptr) {
  *out_ptr = NULL;
  const iree_host_size_t maximum_length =
      iree_arena_block_pool_max_allocation_size(
          capture->metadata_arena.block_pool);
  if (byte_length > maximum_length) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "XDNA queue metadata record of %" PRIhsz
                            " bytes exceeds fixed-block capacity %" PRIhsz,
                            byte_length, maximum_length);
  }
  if (!byte_length) {
    return iree_ok_status();
  }
  return iree_arena_allocate(&capture->metadata_arena, byte_length, out_ptr);
}

iree_status_t iree_hal_amd_xdna_queue_capture_payload(
    iree_hal_amd_xdna_queue_capture_t* capture, iree_const_byte_span_t source,
    iree_hal_amd_xdna_queue_payload_t* out_payload) {
  *out_payload = (iree_hal_amd_xdna_queue_payload_t){0};
  const iree_host_size_t maximum_allocation_size =
      iree_arena_block_pool_max_allocation_size(
          capture->payload_arena.block_pool);
  const iree_host_size_t maximum_chunk_length =
      maximum_allocation_size -
      offsetof(iree_hal_amd_xdna_queue_payload_chunk_t, data);
  iree_hal_amd_xdna_queue_payload_chunk_t* tail = NULL;
  iree_host_size_t source_offset = 0;
  iree_status_t status = iree_ok_status();
  while (source_offset < source.data_length && iree_status_is_ok(status)) {
    const iree_host_size_t chunk_length =
        iree_min(source.data_length - source_offset, maximum_chunk_length);
    iree_hal_amd_xdna_queue_payload_chunk_t* chunk = NULL;
    status = iree_arena_allocate(
        &capture->payload_arena,
        offsetof(iree_hal_amd_xdna_queue_payload_chunk_t, data) + chunk_length,
        (void**)&chunk);
    if (iree_status_is_ok(status)) {
      chunk->next = NULL;
      chunk->data_length = chunk_length;
      memcpy(chunk->data, source.data + source_offset, chunk_length);
      if (tail) {
        tail->next = chunk;
      } else {
        out_payload->head = chunk;
      }
      tail = chunk;
      source_offset += chunk_length;
    }
  }
  if (iree_status_is_ok(status)) {
    out_payload->data_length = source.data_length;
  }
  return status;
}

void iree_hal_amd_xdna_queue_payload_copy(
    const iree_hal_amd_xdna_queue_payload_t* payload, void* target) {
  uint8_t* target_bytes = target;
  for (const iree_hal_amd_xdna_queue_payload_chunk_t* chunk = payload->head;
       chunk; chunk = chunk->next) {
    memcpy(target_bytes, chunk->data, chunk->data_length);
    target_bytes += chunk->data_length;
  }
}

void iree_hal_amd_xdna_queue_capture_deinitialize(
    iree_hal_amd_xdna_queue_capture_t* capture) {
  iree_arena_deinitialize(&capture->payload_arena);
  iree_arena_deinitialize(&capture->metadata_arena);
}
