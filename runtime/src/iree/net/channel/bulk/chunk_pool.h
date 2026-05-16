// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fixed-capacity bulk DATA chunk descriptor pool.
//
// Chunk descriptors let a bulk transfer engine defer DATA processing while
// holding the receive buffer lease that keeps the payload bytes valid. The pool
// never allocates on acquire/release and gives higher layers a bounded unit for
// applying receiver backpressure.

#ifndef IREE_NET_CHANNEL_BULK_CHUNK_POOL_H_
#define IREE_NET_CHANNEL_BULK_CHUNK_POOL_H_

#include "iree/async/buffer_pool.h"
#include "iree/base/api.h"
#include "iree/net/channel/bulk/frame.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Default maximum number of retained chunks.
#define IREE_NET_BULK_CHUNK_POOL_DEFAULT_CAPACITY 64

typedef struct iree_net_bulk_chunk_t iree_net_bulk_chunk_t;
typedef struct iree_net_bulk_chunk_pool_t iree_net_bulk_chunk_pool_t;

// Visits an active chunk.
typedef void (*iree_net_bulk_chunk_visit_fn_t)(void* user_data,
                                               iree_net_bulk_chunk_t* chunk);

// Bulk chunk pool creation options.
typedef struct iree_net_bulk_chunk_pool_options_t {
  // Maximum active chunks retained by the pool.
  iree_host_size_t capacity;

  // Byte length of owner-managed storage attached to each chunk.
  iree_host_size_t user_storage_size;

  // Alignment of owner-managed storage. Zero selects max host alignment.
  iree_host_size_t user_storage_alignment;
} iree_net_bulk_chunk_pool_options_t;

// Returns conservative default chunk pool options.
static inline iree_net_bulk_chunk_pool_options_t
iree_net_bulk_chunk_pool_options_default(void) {
  iree_net_bulk_chunk_pool_options_t options = {0};
  options.capacity = IREE_NET_BULK_CHUNK_POOL_DEFAULT_CAPACITY;
  options.user_storage_alignment = iree_max_align_t;
  return options;
}

// Allocates a fixed-capacity chunk descriptor pool.
iree_status_t iree_net_bulk_chunk_pool_allocate(
    const iree_net_bulk_chunk_pool_options_t* options,
    iree_allocator_t host_allocator, iree_net_bulk_chunk_pool_t** out_pool);

// Frees a chunk pool, releasing any active chunk leases still retained.
//
// Owners that attach additional resources to user storage should visit active
// chunks and release those resources before calling free or clear.
void iree_net_bulk_chunk_pool_free(iree_net_bulk_chunk_pool_t* pool);

// Returns the maximum number of active chunks.
iree_host_size_t iree_net_bulk_chunk_pool_capacity(
    const iree_net_bulk_chunk_pool_t* pool);

// Returns the current number of active chunks.
iree_host_size_t iree_net_bulk_chunk_pool_count(
    const iree_net_bulk_chunk_pool_t* pool);

// Visits all active chunks in unspecified order.
//
// The visitor must not mutate |pool|. Use this for teardown/failure handling
// before releasing owner-managed resources and then clear/free the pool.
void iree_net_bulk_chunk_pool_visit(iree_net_bulk_chunk_pool_t* pool,
                                    iree_net_bulk_chunk_visit_fn_t visitor,
                                    void* user_data);

// Releases all active chunk leases and returns descriptors to the pool.
void iree_net_bulk_chunk_pool_clear(iree_net_bulk_chunk_pool_t* pool);

// Acquires a chunk descriptor and steals |lease| to keep |payload| valid.
//
// Non-empty payloads require a non-NULL retainable lease. On success with a
// non-empty payload, the lease is moved into the chunk and the caller's |lease|
// is cleared, making its post-callback release a no-op. On failure the lease is
// left untouched.
iree_status_t iree_net_bulk_chunk_pool_acquire(
    iree_net_bulk_chunk_pool_t* pool, uint64_t transfer_id,
    uint64_t chunk_offset, uint32_t sequence, iree_net_bulk_frame_flags_t flags,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease,
    uint64_t user_value, iree_net_bulk_chunk_t** out_chunk);

// Releases a chunk descriptor and its retained receive lease.
//
// |chunk| must have been acquired from |pool| and not already released.
void iree_net_bulk_chunk_release(iree_net_bulk_chunk_pool_t* pool,
                                 iree_net_bulk_chunk_t* chunk);

// Returns the transfer ID associated with a chunk.
uint64_t iree_net_bulk_chunk_transfer_id(const iree_net_bulk_chunk_t* chunk);

// Returns the chunk byte offset within the transfer.
uint64_t iree_net_bulk_chunk_offset(const iree_net_bulk_chunk_t* chunk);

// Returns the chunk sequence number.
uint32_t iree_net_bulk_chunk_sequence(const iree_net_bulk_chunk_t* chunk);

// Returns the chunk frame flags.
iree_net_bulk_frame_flags_t iree_net_bulk_chunk_flags(
    const iree_net_bulk_chunk_t* chunk);

// Returns the chunk payload bytes.
iree_const_byte_span_t iree_net_bulk_chunk_payload(
    const iree_net_bulk_chunk_t* chunk);

// Returns a chunk's owner-managed scalar value.
uint64_t iree_net_bulk_chunk_user_value(const iree_net_bulk_chunk_t* chunk);

// Updates a chunk's owner-managed scalar value.
void iree_net_bulk_chunk_set_user_value(iree_net_bulk_chunk_t* chunk,
                                        uint64_t user_value);

// Returns the owner-managed storage attached to the chunk.
iree_byte_span_t iree_net_bulk_chunk_user_storage(iree_net_bulk_chunk_t* chunk);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CHANNEL_BULK_CHUNK_POOL_H_
