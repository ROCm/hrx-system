// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/bulk/chunk_pool.h"

#include <string.h>

#define IREE_NET_BULK_CHUNK_INDEX_EMPTY UINT32_MAX

//===----------------------------------------------------------------------===//
// Storage
//===----------------------------------------------------------------------===//

struct iree_net_bulk_chunk_t {
  // Transfer ID this chunk belongs to.
  uint64_t transfer_id;

  // Byte offset of the chunk within the transfer.
  uint64_t chunk_offset;

  // Payload bytes retained by lease.
  iree_const_byte_span_t payload;

  // Receive lease keeping payload bytes valid.
  iree_async_buffer_lease_t lease;

  // Owner-managed scalar associated with this chunk.
  uint64_t user_value;

  // Sequence number from the DATA frame.
  uint32_t sequence;

  // Descriptor array index for returning this descriptor to the free list.
  uint32_t descriptor_index;

  // Next descriptor index while this descriptor is free.
  uint32_t next_free_index;

  // Frame flags from the DATA frame.
  iree_net_bulk_frame_flags_t flags;

  // Owner-managed storage attached to this chunk.
  iree_byte_span_t user_storage;
};

struct iree_net_bulk_chunk_pool_t {
  // Host allocator used for the pool allocation.
  iree_allocator_t host_allocator;

  // Maximum active chunks retained by the pool.
  iree_host_size_t capacity;

  // Current active chunk count.
  iree_host_size_t count;

  // Byte length of owner-managed storage per chunk.
  iree_host_size_t user_storage_size;

  // Byte stride between owner-managed storage slots.
  iree_host_size_t user_storage_stride;

  // Descriptor index at the head of the free list, or empty.
  uint32_t free_head;

  // Stable descriptor array indexed by descriptor index.
  iree_net_bulk_chunk_t* chunks;

  // Owner-managed per-chunk storage block.
  uint8_t* user_storage;
};

static void iree_net_bulk_chunk_reset(iree_net_bulk_chunk_pool_t* pool,
                                      uint32_t descriptor_index,
                                      uint32_t next_free_index) {
  iree_net_bulk_chunk_t* chunk = &pool->chunks[descriptor_index];
  iree_async_buffer_lease_release(&chunk->lease);
  chunk->transfer_id = 0;
  chunk->chunk_offset = 0;
  chunk->payload = iree_const_byte_span_empty();
  memset(&chunk->lease, 0, sizeof(chunk->lease));
  chunk->user_value = 0;
  chunk->sequence = 0;
  chunk->descriptor_index = descriptor_index;
  chunk->next_free_index = next_free_index;
  chunk->flags = IREE_NET_BULK_FRAME_FLAG_NONE;
  if (chunk->user_storage.data_length > 0) {
    memset(chunk->user_storage.data, 0, chunk->user_storage.data_length);
  }
}

static iree_status_t iree_net_bulk_chunk_pool_resolve_options(
    const iree_net_bulk_chunk_pool_options_t* options,
    iree_net_bulk_chunk_pool_options_t* out_options,
    iree_host_size_t* out_user_storage_stride) {
  iree_net_bulk_chunk_pool_options_t resolved_options =
      iree_net_bulk_chunk_pool_options_default();
  if (options) resolved_options = *options;
  if (resolved_options.capacity == 0) {
    resolved_options.capacity = IREE_NET_BULK_CHUNK_POOL_DEFAULT_CAPACITY;
  }
  if (resolved_options.capacity >= IREE_NET_BULK_CHUNK_INDEX_EMPTY) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "bulk chunk pool capacity too large: %" PRIhsz,
                            resolved_options.capacity);
  }

  if (resolved_options.user_storage_alignment == 0) {
    resolved_options.user_storage_alignment = iree_max_align_t;
  }
  if (!iree_host_size_is_valid_alignment(
          resolved_options.user_storage_alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "bulk chunk user storage alignment must be a power of two: %" PRIhsz,
        resolved_options.user_storage_alignment);
  }

  iree_host_size_t user_storage_stride = 0;
  if (resolved_options.user_storage_size > 0 &&
      !iree_host_size_checked_align(resolved_options.user_storage_size,
                                    resolved_options.user_storage_alignment,
                                    &user_storage_stride)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "bulk chunk user storage stride overflow");
  }

  *out_options = resolved_options;
  *out_user_storage_stride = user_storage_stride;
  return iree_ok_status();
}

static iree_status_t iree_net_bulk_chunk_pool_calculate_layout(
    const iree_net_bulk_chunk_pool_options_t* options,
    iree_host_size_t user_storage_stride, iree_host_size_t* out_total_size,
    iree_host_size_t* out_chunks_offset,
    iree_host_size_t* out_user_storage_offset) {
  iree_host_size_t user_storage_alignment =
      user_storage_stride > 0 ? options->user_storage_alignment : 1;
  return IREE_STRUCT_LAYOUT(
      sizeof(iree_net_bulk_chunk_pool_t), out_total_size,
      IREE_STRUCT_FIELD_ALIGNED(options->capacity, iree_net_bulk_chunk_t,
                                iree_alignof(iree_net_bulk_chunk_t),
                                out_chunks_offset),
      IREE_STRUCT_ARRAY_FIELD_ALIGNED(options->capacity, user_storage_stride,
                                      uint8_t, user_storage_alignment,
                                      out_user_storage_offset));
}

//===----------------------------------------------------------------------===//
// iree_net_bulk_chunk_pool_t
//===----------------------------------------------------------------------===//

iree_status_t iree_net_bulk_chunk_pool_allocate(
    const iree_net_bulk_chunk_pool_options_t* options,
    iree_allocator_t host_allocator, iree_net_bulk_chunk_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(out_pool);
  *out_pool = NULL;

  iree_net_bulk_chunk_pool_options_t resolved_options;
  iree_host_size_t user_storage_stride = 0;
  IREE_RETURN_IF_ERROR(iree_net_bulk_chunk_pool_resolve_options(
      options, &resolved_options, &user_storage_stride));

  iree_host_size_t total_size = 0;
  iree_host_size_t chunks_offset = 0;
  iree_host_size_t user_storage_offset = 0;
  IREE_RETURN_IF_ERROR(iree_net_bulk_chunk_pool_calculate_layout(
      &resolved_options, user_storage_stride, &total_size, &chunks_offset,
      &user_storage_offset));

  iree_net_bulk_chunk_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&pool));
  memset(pool, 0, total_size);

  uint8_t* pool_storage = (uint8_t*)pool;
  pool->host_allocator = host_allocator;
  pool->capacity = resolved_options.capacity;
  pool->user_storage_size = resolved_options.user_storage_size;
  pool->user_storage_stride = user_storage_stride;
  pool->free_head = pool->capacity > 0 ? 0 : IREE_NET_BULK_CHUNK_INDEX_EMPTY;
  pool->chunks = (iree_net_bulk_chunk_t*)(pool_storage + chunks_offset);
  pool->user_storage = (uint8_t*)(pool_storage + user_storage_offset);

  for (iree_host_size_t i = 0; i < pool->capacity; ++i) {
    iree_net_bulk_chunk_t* chunk = &pool->chunks[i];
    if (pool->user_storage_size > 0) {
      chunk->user_storage = iree_make_byte_span(
          pool->user_storage + i * pool->user_storage_stride,
          pool->user_storage_size);
    }
    uint32_t next_free_index = i + 1 < pool->capacity
                                   ? (uint32_t)(i + 1)
                                   : IREE_NET_BULK_CHUNK_INDEX_EMPTY;
    iree_net_bulk_chunk_reset(pool, (uint32_t)i, next_free_index);
  }

  *out_pool = pool;
  return iree_ok_status();
}

void iree_net_bulk_chunk_pool_free(iree_net_bulk_chunk_pool_t* pool) {
  if (!pool) return;
  iree_allocator_t host_allocator = pool->host_allocator;
  iree_net_bulk_chunk_pool_clear(pool);
  iree_allocator_free(host_allocator, pool);
}

iree_host_size_t iree_net_bulk_chunk_pool_capacity(
    const iree_net_bulk_chunk_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool);
  return pool->capacity;
}

iree_host_size_t iree_net_bulk_chunk_pool_count(
    const iree_net_bulk_chunk_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool);
  return pool->count;
}

void iree_net_bulk_chunk_pool_visit(iree_net_bulk_chunk_pool_t* pool,
                                    iree_net_bulk_chunk_visit_fn_t visitor,
                                    void* user_data) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(visitor);
  for (iree_host_size_t i = 0; i < pool->capacity; ++i) {
    iree_net_bulk_chunk_t* chunk = &pool->chunks[i];
    if (chunk->transfer_id != 0) visitor(user_data, chunk);
  }
}

void iree_net_bulk_chunk_pool_clear(iree_net_bulk_chunk_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool);
  for (iree_host_size_t i = 0; i < pool->capacity; ++i) {
    uint32_t next_free_index = i + 1 < pool->capacity
                                   ? (uint32_t)(i + 1)
                                   : IREE_NET_BULK_CHUNK_INDEX_EMPTY;
    iree_net_bulk_chunk_reset(pool, (uint32_t)i, next_free_index);
  }
  pool->free_head = pool->capacity > 0 ? 0 : IREE_NET_BULK_CHUNK_INDEX_EMPTY;
  pool->count = 0;
}

iree_status_t iree_net_bulk_chunk_pool_acquire(
    iree_net_bulk_chunk_pool_t* pool, uint64_t transfer_id,
    uint64_t chunk_offset, uint32_t sequence, iree_net_bulk_frame_flags_t flags,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease,
    uint64_t user_value, iree_net_bulk_chunk_t** out_chunk) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_chunk);
  *out_chunk = NULL;

  if (transfer_id == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bulk chunk transfer ID must be non-zero");
  }
  if (payload.data_length > 0) {
    if (!payload.data) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "bulk chunk payload has length but no data");
    }
    if (!lease || !lease->release.fn) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "bulk chunk payload requires a retainable receive lease");
    }
  }
  if (pool->free_head == IREE_NET_BULK_CHUNK_INDEX_EMPTY) {
    return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
  }

  uint32_t descriptor_index = pool->free_head;
  iree_net_bulk_chunk_t* chunk = &pool->chunks[descriptor_index];
  pool->free_head = chunk->next_free_index;

  chunk->transfer_id = transfer_id;
  chunk->chunk_offset = chunk_offset;
  chunk->payload = payload;
  if (payload.data_length > 0) {
    chunk->lease = *lease;
    memset(lease, 0, sizeof(*lease));
  }
  chunk->user_value = user_value;
  chunk->sequence = sequence;
  chunk->next_free_index = IREE_NET_BULK_CHUNK_INDEX_EMPTY;
  chunk->flags = flags;
  if (chunk->user_storage.data_length > 0) {
    memset(chunk->user_storage.data, 0, chunk->user_storage.data_length);
  }

  ++pool->count;
  *out_chunk = chunk;
  return iree_ok_status();
}

void iree_net_bulk_chunk_release(iree_net_bulk_chunk_pool_t* pool,
                                 iree_net_bulk_chunk_t* chunk) {
  if (!pool || !chunk) return;
  IREE_ASSERT_LT(chunk->descriptor_index, pool->capacity);
  IREE_ASSERT_EQ(chunk, &pool->chunks[chunk->descriptor_index]);
  IREE_ASSERT_NE(chunk->transfer_id, 0);
  IREE_ASSERT_GT(pool->count, 0);
  iree_net_bulk_chunk_reset(pool, chunk->descriptor_index, pool->free_head);
  pool->free_head = chunk->descriptor_index;
  --pool->count;
}

uint64_t iree_net_bulk_chunk_transfer_id(const iree_net_bulk_chunk_t* chunk) {
  IREE_ASSERT_ARGUMENT(chunk);
  return chunk->transfer_id;
}

uint64_t iree_net_bulk_chunk_offset(const iree_net_bulk_chunk_t* chunk) {
  IREE_ASSERT_ARGUMENT(chunk);
  return chunk->chunk_offset;
}

uint32_t iree_net_bulk_chunk_sequence(const iree_net_bulk_chunk_t* chunk) {
  IREE_ASSERT_ARGUMENT(chunk);
  return chunk->sequence;
}

iree_net_bulk_frame_flags_t iree_net_bulk_chunk_flags(
    const iree_net_bulk_chunk_t* chunk) {
  IREE_ASSERT_ARGUMENT(chunk);
  return chunk->flags;
}

iree_const_byte_span_t iree_net_bulk_chunk_payload(
    const iree_net_bulk_chunk_t* chunk) {
  IREE_ASSERT_ARGUMENT(chunk);
  return chunk->payload;
}

uint64_t iree_net_bulk_chunk_user_value(const iree_net_bulk_chunk_t* chunk) {
  IREE_ASSERT_ARGUMENT(chunk);
  return chunk->user_value;
}

void iree_net_bulk_chunk_set_user_value(iree_net_bulk_chunk_t* chunk,
                                        uint64_t user_value) {
  IREE_ASSERT_ARGUMENT(chunk);
  chunk->user_value = user_value;
}

iree_byte_span_t iree_net_bulk_chunk_user_storage(
    iree_net_bulk_chunk_t* chunk) {
  IREE_ASSERT_ARGUMENT(chunk);
  return chunk->user_storage;
}
