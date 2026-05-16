// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bulk receive window: bounded DATA retention and local credit accounting.
//
// The receive window owns the fixed-capacity chunk descriptor pool used by a
// bulk transfer receiver. It converts retained descriptor capacity into CREDIT
// frames and keeps the accounting explicit so temporary send backpressure never
// turns chunk cleanup into a transport failure.

#ifndef IREE_NET_CHANNEL_BULK_RECEIVE_WINDOW_H_
#define IREE_NET_CHANNEL_BULK_RECEIVE_WINDOW_H_

#include "iree/base/api.h"
#include "iree/net/channel/bulk/chunk_pool.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Default number of unadvertised credits that should trigger a flush.
#define IREE_NET_BULK_RECEIVE_WINDOW_DEFAULT_CREDIT_BATCH_THRESHOLD 1

typedef struct iree_net_bulk_receive_window_t iree_net_bulk_receive_window_t;

// Sends a CREDIT frame advertising additional local DATA chunk capacity.
typedef iree_status_t (*iree_net_bulk_receive_window_send_credit_fn_t)(
    void* user_data, uint32_t credit_delta);

// Bulk receive window callbacks.
typedef struct iree_net_bulk_receive_window_callbacks_t {
  // Sends DATA chunk credits to the peer.
  iree_net_bulk_receive_window_send_credit_fn_t send_credit;

  // User data passed to callbacks.
  void* user_data;
} iree_net_bulk_receive_window_callbacks_t;

// Bulk receive window creation options.
typedef struct iree_net_bulk_receive_window_options_t {
  // Chunk descriptor pool options for retained DATA frames.
  iree_net_bulk_chunk_pool_options_t chunk_pool;

  // Preferred minimum credit batch size before callers flush.
  uint32_t credit_batch_threshold;
} iree_net_bulk_receive_window_options_t;

// Returns conservative default receive window options.
static inline iree_net_bulk_receive_window_options_t
iree_net_bulk_receive_window_options_default(void) {
  iree_net_bulk_receive_window_options_t options = {0};
  options.chunk_pool = iree_net_bulk_chunk_pool_options_default();
  options.credit_batch_threshold =
      IREE_NET_BULK_RECEIVE_WINDOW_DEFAULT_CREDIT_BATCH_THRESHOLD;
  return options;
}

// Allocates a receive window and its fixed-capacity chunk pool.
//
// The window is not internally synchronized. Calls that mutate credit or chunk
// state must be serialized by the owning transfer engine, usually on the
// network proactor thread.
iree_status_t iree_net_bulk_receive_window_allocate(
    const iree_net_bulk_receive_window_options_t* options,
    iree_net_bulk_receive_window_callbacks_t callbacks,
    iree_allocator_t host_allocator,
    iree_net_bulk_receive_window_t** out_window);

// Frees a receive window, releasing all retained chunk leases.
void iree_net_bulk_receive_window_free(iree_net_bulk_receive_window_t* window);

// Returns retained DATA chunk capacity.
iree_host_size_t iree_net_bulk_receive_window_capacity(
    const iree_net_bulk_receive_window_t* window);

// Returns the number of DATA chunks currently retained.
iree_host_size_t iree_net_bulk_receive_window_count(
    const iree_net_bulk_receive_window_t* window);

// Returns credits advertised to the peer but not yet consumed by DATA.
uint32_t iree_net_bulk_receive_window_advertised_credit_count(
    const iree_net_bulk_receive_window_t* window);

// Returns local capacity that has not yet been advertised to the peer.
uint32_t iree_net_bulk_receive_window_unadvertised_credit_count(
    const iree_net_bulk_receive_window_t* window);

// Returns true if unadvertised credit has reached the preferred batch size.
bool iree_net_bulk_receive_window_should_flush_credit(
    const iree_net_bulk_receive_window_t* window);

// Sends all currently unadvertised local receive credits.
//
// If the send path returns RESOURCE_EXHAUSTED, credit remains unadvertised and
// callers can retry later. Any other error also leaves credit unchanged.
iree_status_t iree_net_bulk_receive_window_flush_credit(
    iree_net_bulk_receive_window_t* window);

// Acquires a DATA chunk descriptor and consumes one advertised receive credit.
//
// A DATA frame without advertised credit is a peer protocol violation. On any
// failure no credit is consumed and |lease| is left untouched.
iree_status_t iree_net_bulk_receive_window_acquire_chunk(
    iree_net_bulk_receive_window_t* window, uint64_t transfer_id,
    uint64_t chunk_offset, uint32_t sequence, iree_net_bulk_frame_flags_t flags,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease,
    uint64_t user_value, iree_net_bulk_chunk_t** out_chunk);

// Releases a retained chunk and records one unadvertised local receive credit.
//
// This does not send CREDIT itself. Callers can release many chunks and then
// call iree_net_bulk_receive_window_flush_credit once, or use
// iree_net_bulk_receive_window_should_flush_credit to follow the configured
// batch threshold.
void iree_net_bulk_receive_window_release_chunk(
    iree_net_bulk_receive_window_t* window, iree_net_bulk_chunk_t* chunk);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CHANNEL_BULK_RECEIVE_WINDOW_H_
