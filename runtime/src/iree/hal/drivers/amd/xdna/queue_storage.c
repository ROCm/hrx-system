// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/queue_storage.h"

#include "iree/base/internal/math.h"

enum {
  IREE_HAL_AMD_XDNA_METADATA_BLOCK_SIZE = 4096,
  IREE_HAL_AMD_XDNA_PAYLOAD_BLOCK_SIZE = 32768,
};

struct iree_alignas(iree_max_align_t) iree_hal_amd_xdna_queue_metadata_block_t {
  // Next large block held by this capture.
  iree_hal_amd_xdna_queue_metadata_block_t* next;
  // Queue-owned size class to which this block returns at retirement.
  iree_arena_block_pool_t* pool;
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
  for (iree_host_size_t i = 0; i < IREE_HAL_AMD_XDNA_LARGE_METADATA_CLASS_COUNT;
       ++i) {
    iree_arena_block_pool_initialize(
        (iree_host_size_t)1 << (IREE_HAL_AMD_XDNA_LARGE_METADATA_MIN_SHIFT + i),
        host_allocator, &out_storage->large_metadata_pools[i]);
  }
  iree_arena_block_pool_initialize(IREE_HAL_AMD_XDNA_PAYLOAD_BLOCK_SIZE,
                                   host_allocator,
                                   &out_storage->payload_block_pool);
}

void iree_hal_amd_xdna_queue_storage_deinitialize(
    iree_hal_amd_xdna_queue_storage_t* storage) {
  iree_arena_block_pool_deinitialize(&storage->payload_block_pool);
  for (iree_host_size_t i = 0; i < IREE_HAL_AMD_XDNA_LARGE_METADATA_CLASS_COUNT;
       ++i) {
    iree_arena_block_pool_deinitialize(&storage->large_metadata_pools[i]);
  }
  iree_arena_block_pool_deinitialize(&storage->metadata_block_pool);
}

void iree_hal_amd_xdna_queue_storage_trim(
    iree_hal_amd_xdna_queue_storage_t* storage) {
  iree_arena_block_pool_trim(&storage->payload_block_pool);
  for (iree_host_size_t i = 0; i < IREE_HAL_AMD_XDNA_LARGE_METADATA_CLASS_COUNT;
       ++i) {
    iree_arena_block_pool_trim(&storage->large_metadata_pools[i]);
  }
  iree_arena_block_pool_trim(&storage->metadata_block_pool);
}

void iree_hal_amd_xdna_queue_capture_initialize(
    iree_hal_amd_xdna_queue_storage_t* storage,
    iree_hal_amd_xdna_queue_capture_t* out_capture) {
  out_capture->storage = storage;
  out_capture->large_metadata_head = NULL;
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
    const iree_host_size_t overhead =
        sizeof(iree_hal_amd_xdna_queue_metadata_block_t) +
        sizeof(iree_arena_block_t);
    const iree_host_size_t maximum_block_size =
        (iree_host_size_t)1 << (sizeof(iree_host_size_t) * 8 - 1);
    if (byte_length > maximum_block_size - overhead) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "XDNA queue metadata record size overflow");
    }
    const uint32_t block_shift =
        64 - iree_math_count_leading_zeros_u64(byte_length + overhead - 1);
    const uint32_t class_index =
        block_shift - IREE_HAL_AMD_XDNA_LARGE_METADATA_MIN_SHIFT;
    iree_arena_block_pool_t* pool =
        &capture->storage->large_metadata_pools[class_index];
    iree_arena_block_t* block = NULL;
    iree_hal_amd_xdna_queue_metadata_block_t* metadata = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_block_pool_acquire(pool, &block, (void**)&metadata));
    metadata->pool = pool;
    metadata->next = capture->large_metadata_head;
    capture->large_metadata_head = metadata;
    *out_ptr = metadata + 1;
    return iree_ok_status();
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
  iree_hal_amd_xdna_queue_metadata_block_t* metadata =
      capture->large_metadata_head;
  while (metadata) {
    iree_hal_amd_xdna_queue_metadata_block_t* next = metadata->next;
    iree_arena_block_pool_t* pool = metadata->pool;
    iree_arena_block_t* block = iree_arena_block_trailer(pool, metadata);
    iree_arena_block_pool_release(pool, block, block);
    metadata = next;
  }
  iree_arena_deinitialize(&capture->metadata_arena);
}
