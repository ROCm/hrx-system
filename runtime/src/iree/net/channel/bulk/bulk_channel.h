// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bulk channel: reorderable transfer framing over a message endpoint.
//
// The bulk channel carries large payloads independently from queue/control
// traffic. Each frame is self-contained and keyed by a 64-bit transfer ID so
// DATA chunks can be delivered in any order. The channel validates wire
// framing, enforces fixed send-context capacity, and dispatches transfer
// lifecycle callbacks to the embedding transfer engine.

#ifndef IREE_NET_CHANNEL_BULK_BULK_CHANNEL_H_
#define IREE_NET_CHANNEL_BULK_BULK_CHANNEL_H_

#include "iree/async/buffer_pool.h"
#include "iree/async/span.h"
#include "iree/base/api.h"
#include "iree/net/channel/bulk/frame.h"
#include "iree/net/message_endpoint.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Channel state
//===----------------------------------------------------------------------===//

// Bulk channel lifecycle states.
typedef enum iree_net_bulk_channel_state_e {
  // Channel is created but not yet activated. No sends or receives.
  IREE_NET_BULK_CHANNEL_STATE_CREATED = 0,
  // Normal operation. All sends and receives are active.
  IREE_NET_BULK_CHANNEL_STATE_OPERATIONAL = 1,
  // Terminal error. All operations fail. Only release is valid.
  IREE_NET_BULK_CHANNEL_STATE_ERROR = 2,
} iree_net_bulk_channel_state_t;

//===----------------------------------------------------------------------===//
// Options
//===----------------------------------------------------------------------===//

// Default maximum number of send contexts that can be in flight.
#define IREE_NET_BULK_CHANNEL_DEFAULT_SEND_CONTEXT_CAPACITY 64

// Bulk channel creation options.
typedef struct iree_net_bulk_channel_options_t {
  // Maximum scatter-gather spans per send after endpoint overhead.
  iree_host_size_t max_send_spans;

  // Maximum send frames submitted but not completed at once.
  iree_host_size_t send_context_capacity;
} iree_net_bulk_channel_options_t;

// Returns conservative default bulk channel options.
static inline iree_net_bulk_channel_options_t
iree_net_bulk_channel_options_default(void) {
  iree_net_bulk_channel_options_t options = {0};
  options.send_context_capacity =
      IREE_NET_BULK_CHANNEL_DEFAULT_SEND_CONTEXT_CAPACITY;
  return options;
}

//===----------------------------------------------------------------------===//
// Callbacks
//===----------------------------------------------------------------------===//

// Called when a START frame announces a new transfer.
typedef iree_status_t (*iree_net_bulk_channel_on_start_fn_t)(
    void* user_data, uint64_t transfer_id, uint64_t total_size,
    iree_net_bulk_frame_flags_t flags);

// Called when a DATA frame is received.
//
// |chunk_data| points into the receive buffer. Retain |lease| to keep the data
// valid beyond the callback.
typedef iree_status_t (*iree_net_bulk_channel_on_data_fn_t)(
    void* user_data, uint64_t transfer_id, uint64_t chunk_offset,
    uint32_t sequence, iree_net_bulk_frame_flags_t flags,
    iree_const_byte_span_t chunk_data, iree_async_buffer_lease_t* lease);

// Called when a COMPLETE frame reports successful transfer completion.
typedef iree_status_t (*iree_net_bulk_channel_on_complete_fn_t)(
    void* user_data, uint64_t transfer_id);

// Called when an ABORT frame reports transfer cancellation or failure.
//
// |abort_data| points into the receive buffer and may be empty. Retain |lease|
// to keep the data valid beyond the callback.
typedef iree_status_t (*iree_net_bulk_channel_on_abort_fn_t)(
    void* user_data, uint64_t transfer_id, iree_const_byte_span_t abort_data,
    iree_async_buffer_lease_t* lease);

// Called when the underlying transport reports an error.
//
// After this callback, the channel is in ERROR state. |status| ownership is
// transferred to the callback.
typedef void (*iree_net_bulk_channel_on_transport_error_fn_t)(
    void* user_data, iree_status_t status);

// Called when a bulk send operation completes.
//
// |operation_user_data| echoes the value from the send call. |status| indicates
// success or failure.
typedef void (*iree_net_bulk_channel_on_send_complete_fn_t)(
    void* user_data, uint64_t operation_user_data, iree_status_t status);

// Bundled application callbacks for channel events.
typedef struct iree_net_bulk_channel_callbacks_t {
  // Required callback for START frames.
  iree_net_bulk_channel_on_start_fn_t on_start;

  // Required callback for DATA frames.
  iree_net_bulk_channel_on_data_fn_t on_data;

  // Required callback for COMPLETE frames.
  iree_net_bulk_channel_on_complete_fn_t on_complete;

  // Required callback for ABORT frames.
  iree_net_bulk_channel_on_abort_fn_t on_abort;

  // Optional callback for transport errors.
  iree_net_bulk_channel_on_transport_error_fn_t on_transport_error;

  // Optional callback for send completions.
  iree_net_bulk_channel_on_send_complete_fn_t on_send_complete;

  // User data passed as the first argument to each callback.
  void* user_data;
} iree_net_bulk_channel_callbacks_t;

//===----------------------------------------------------------------------===//
// iree_net_bulk_channel_t
//===----------------------------------------------------------------------===//

typedef struct iree_net_bulk_channel_t iree_net_bulk_channel_t;

// Creates a bulk channel that will operate over the given message endpoint.
//
// The |endpoint| is a borrowed view used for both receive and send paths. The
// caller must ensure the underlying transport object outlives the channel or
// call iree_net_bulk_channel_detach before the endpoint is destroyed.
//
// The |header_pool| provides buffers for copying bulk frame headers into stable
// storage for asynchronous scatter-gather sends. The channel takes ownership of
// the pool and frees it on destroy. Bulk DATA payloads are not copied by the
// channel; payload spans must remain valid until on_send_complete fires.
//
// |options| may be NULL to use defaults. |options->send_context_capacity|
// bounds the number of send frames that can be in flight and provides
// allocation-free steady-state send contexts.
//
// All lifecycle callbacks except transport/send completion callbacks are
// required.
iree_status_t iree_net_bulk_channel_create(
    iree_net_message_endpoint_t endpoint,
    const iree_net_bulk_channel_options_t* options,
    iree_async_buffer_pool_t* header_pool,
    iree_net_bulk_channel_callbacks_t callbacks,
    iree_allocator_t host_allocator, iree_net_bulk_channel_t** out_channel);

// Retains a reference. NULL-safe.
void iree_net_bulk_channel_retain(iree_net_bulk_channel_t* channel);

// Releases a reference. Destroys when last reference released. NULL-safe.
void iree_net_bulk_channel_release(iree_net_bulk_channel_t* channel);

// Activates the channel, enabling message receipt.
//
// Must be called from the proactor thread. Transitions CREATED -> OPERATIONAL.
iree_status_t iree_net_bulk_channel_activate(iree_net_bulk_channel_t* channel);

// Detaches the channel from its underlying endpoint.
//
// After detach, the channel cannot send or receive, but may still be retained
// by in-flight send completions and later released safely.
void iree_net_bulk_channel_detach(iree_net_bulk_channel_t* channel);

// Returns the current channel state.
iree_net_bulk_channel_state_t iree_net_bulk_channel_state(
    const iree_net_bulk_channel_t* channel);

// Returns true if any send operations are still in flight.
bool iree_net_bulk_channel_has_pending_sends(
    const iree_net_bulk_channel_t* channel);

// Returns the endpoint's current send budget.
iree_net_carrier_send_budget_t iree_net_bulk_channel_query_send_budget(
    iree_net_bulk_channel_t* channel);

// Sends a START frame for |transfer_id|.
iree_status_t iree_net_bulk_channel_send_start(
    iree_net_bulk_channel_t* channel, uint64_t transfer_id, uint64_t total_size,
    iree_net_bulk_frame_flags_t flags, uint64_t operation_user_data);

// Sends a DATA frame for |transfer_id|.
//
// |chunk_payload| is sent zero-copy and must remain valid until
// on_send_complete fires.
iree_status_t iree_net_bulk_channel_send_data(
    iree_net_bulk_channel_t* channel, uint64_t transfer_id,
    uint64_t chunk_offset, uint32_t sequence, iree_net_bulk_frame_flags_t flags,
    iree_async_span_list_t chunk_payload, uint64_t operation_user_data);

// Sends a COMPLETE frame for |transfer_id|.
iree_status_t iree_net_bulk_channel_send_complete(
    iree_net_bulk_channel_t* channel, uint64_t transfer_id,
    uint64_t operation_user_data);

// Sends an ABORT frame for |transfer_id| with optional payload data.
//
// |abort_payload| is sent zero-copy and must remain valid until
// on_send_complete fires.
iree_status_t iree_net_bulk_channel_send_abort(
    iree_net_bulk_channel_t* channel, uint64_t transfer_id,
    iree_async_span_list_t abort_payload, uint64_t operation_user_data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CHANNEL_BULK_BULK_CHANNEL_H_
