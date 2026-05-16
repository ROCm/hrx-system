// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/bulk/receive_window.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Storage
//===----------------------------------------------------------------------===//

struct iree_net_bulk_receive_window_t {
  // Host allocator used for the window allocation.
  iree_allocator_t host_allocator;

  // Retained DATA chunk descriptor pool.
  iree_net_bulk_chunk_pool_t* chunk_pool;

  // Callback bundle for sending CREDIT frames.
  iree_net_bulk_receive_window_callbacks_t callbacks;

  // Preferred minimum credit batch size before flushing.
  uint32_t credit_batch_threshold;

  // Credits sent to the peer that have not yet been consumed by DATA.
  uint32_t advertised_credit_count;

  // Local free descriptor capacity not yet advertised to the peer.
  uint32_t unadvertised_credit_count;
};

static void iree_net_bulk_receive_window_assert_invariants(
    const iree_net_bulk_receive_window_t* window) {
  iree_host_size_t retained_chunk_count =
      iree_net_bulk_chunk_pool_count(window->chunk_pool);
  iree_host_size_t accounted_capacity = retained_chunk_count +
                                        window->advertised_credit_count +
                                        window->unadvertised_credit_count;
  IREE_ASSERT_EQ(accounted_capacity,
                 iree_net_bulk_chunk_pool_capacity(window->chunk_pool));
}

static iree_status_t iree_net_bulk_receive_window_resolve_options(
    const iree_net_bulk_receive_window_options_t* options,
    iree_net_bulk_receive_window_options_t* out_options) {
  iree_net_bulk_receive_window_options_t resolved_options =
      iree_net_bulk_receive_window_options_default();
  if (options) resolved_options = *options;
  if (resolved_options.chunk_pool.capacity == 0) {
    resolved_options.chunk_pool.capacity =
        IREE_NET_BULK_CHUNK_POOL_DEFAULT_CAPACITY;
  }
  if (resolved_options.chunk_pool.capacity > INT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "bulk receive window capacity too large for CREDIT frames: %" PRIhsz,
        resolved_options.chunk_pool.capacity);
  }
  if (resolved_options.credit_batch_threshold == 0) {
    resolved_options.credit_batch_threshold =
        IREE_NET_BULK_RECEIVE_WINDOW_DEFAULT_CREDIT_BATCH_THRESHOLD;
  }
  if (resolved_options.credit_batch_threshold >
      resolved_options.chunk_pool.capacity) {
    resolved_options.credit_batch_threshold =
        (uint32_t)resolved_options.chunk_pool.capacity;
  }
  *out_options = resolved_options;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_net_bulk_receive_window_t
//===----------------------------------------------------------------------===//

iree_status_t iree_net_bulk_receive_window_allocate(
    const iree_net_bulk_receive_window_options_t* options,
    iree_net_bulk_receive_window_callbacks_t callbacks,
    iree_allocator_t host_allocator,
    iree_net_bulk_receive_window_t** out_window) {
  IREE_ASSERT_ARGUMENT(out_window);
  *out_window = NULL;

  iree_status_t status = iree_ok_status();
  if (!callbacks.send_credit) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "bulk receive window requires a CREDIT send callback");
  }

  iree_net_bulk_receive_window_options_t resolved_options;
  if (iree_status_is_ok(status)) {
    status = iree_net_bulk_receive_window_resolve_options(options,
                                                          &resolved_options);
  }

  iree_net_bulk_receive_window_t* window = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, sizeof(*window), (void**)&window);
  }
  if (iree_status_is_ok(status)) {
    memset(window, 0, sizeof(*window));
    status = iree_net_bulk_chunk_pool_allocate(
        &resolved_options.chunk_pool, host_allocator, &window->chunk_pool);
  }

  if (iree_status_is_ok(status)) {
    window->host_allocator = host_allocator;
    window->callbacks = callbacks;
    window->credit_batch_threshold = resolved_options.credit_batch_threshold;
    window->advertised_credit_count = 0;
    window->unadvertised_credit_count =
        (uint32_t)iree_net_bulk_chunk_pool_capacity(window->chunk_pool);
    iree_net_bulk_receive_window_assert_invariants(window);
    *out_window = window;
  } else {
    if (window) {
      iree_net_bulk_chunk_pool_free(window->chunk_pool);
      iree_allocator_free(host_allocator, window);
    }
  }
  return status;
}

void iree_net_bulk_receive_window_free(iree_net_bulk_receive_window_t* window) {
  if (!window) return;
  iree_allocator_t host_allocator = window->host_allocator;
  iree_net_bulk_chunk_pool_free(window->chunk_pool);
  iree_allocator_free(host_allocator, window);
}

iree_host_size_t iree_net_bulk_receive_window_capacity(
    const iree_net_bulk_receive_window_t* window) {
  IREE_ASSERT_ARGUMENT(window);
  return iree_net_bulk_chunk_pool_capacity(window->chunk_pool);
}

iree_host_size_t iree_net_bulk_receive_window_count(
    const iree_net_bulk_receive_window_t* window) {
  IREE_ASSERT_ARGUMENT(window);
  return iree_net_bulk_chunk_pool_count(window->chunk_pool);
}

uint32_t iree_net_bulk_receive_window_advertised_credit_count(
    const iree_net_bulk_receive_window_t* window) {
  IREE_ASSERT_ARGUMENT(window);
  return window->advertised_credit_count;
}

uint32_t iree_net_bulk_receive_window_unadvertised_credit_count(
    const iree_net_bulk_receive_window_t* window) {
  IREE_ASSERT_ARGUMENT(window);
  return window->unadvertised_credit_count;
}

bool iree_net_bulk_receive_window_should_flush_credit(
    const iree_net_bulk_receive_window_t* window) {
  IREE_ASSERT_ARGUMENT(window);
  return window->unadvertised_credit_count >= window->credit_batch_threshold;
}

iree_status_t iree_net_bulk_receive_window_flush_credit(
    iree_net_bulk_receive_window_t* window) {
  IREE_ASSERT_ARGUMENT(window);
  iree_net_bulk_receive_window_assert_invariants(window);

  uint32_t credit_delta = window->unadvertised_credit_count;
  if (credit_delta == 0) return iree_ok_status();

  iree_status_t status =
      window->callbacks.send_credit(window->callbacks.user_data, credit_delta);
  if (iree_status_is_ok(status)) {
    window->unadvertised_credit_count = 0;
    window->advertised_credit_count += credit_delta;
    iree_net_bulk_receive_window_assert_invariants(window);
  }
  return status;
}

iree_status_t iree_net_bulk_receive_window_acquire_chunk(
    iree_net_bulk_receive_window_t* window, uint64_t transfer_id,
    uint64_t chunk_offset, uint32_t sequence, iree_net_bulk_frame_flags_t flags,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease,
    uint64_t user_value, iree_net_bulk_chunk_t** out_chunk) {
  IREE_ASSERT_ARGUMENT(window);
  IREE_ASSERT_ARGUMENT(out_chunk);
  *out_chunk = NULL;
  iree_net_bulk_receive_window_assert_invariants(window);

  if (window->advertised_credit_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "DATA frame received without advertised bulk receive credit");
  }

  iree_status_t status = iree_net_bulk_chunk_pool_acquire(
      window->chunk_pool, transfer_id, chunk_offset, sequence, flags, payload,
      lease, user_value, out_chunk);
  if (iree_status_is_ok(status)) {
    --window->advertised_credit_count;
    iree_net_bulk_receive_window_assert_invariants(window);
  }
  return status;
}

void iree_net_bulk_receive_window_release_chunk(
    iree_net_bulk_receive_window_t* window, iree_net_bulk_chunk_t* chunk) {
  if (!window || !chunk) return;
  iree_net_bulk_receive_window_assert_invariants(window);
  iree_net_bulk_chunk_release(window->chunk_pool, chunk);
  ++window->unadvertised_credit_count;
  iree_net_bulk_receive_window_assert_invariants(window);
}
