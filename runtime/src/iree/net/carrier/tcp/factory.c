// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/tcp/factory.h"

#include <string.h>

#include "iree/async/address.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/net/carrier/tcp/carrier.h"
#include "iree/net/channel/util/frame_sender.h"
#include "iree/net/channel/util/framing_adapter.h"
#include "iree/net/connection.h"
#include "iree/net/endpoint_lifecycle.h"
#include "iree/net/message_endpoint.h"

//===----------------------------------------------------------------------===//
// Internal types
//===----------------------------------------------------------------------===//

typedef struct iree_net_tcp_factory_t {
  iree_net_transport_factory_t base;
  iree_net_tcp_carrier_options_t default_options;
  iree_allocator_t host_allocator;
} iree_net_tcp_factory_t;

typedef struct iree_net_tcp_connection_t iree_net_tcp_connection_t;

// Maximum number of frames the TCP mux will queue for a stream that has not
// been activated locally yet. Valid peers may briefly outrun local deferred
// endpoint activation; sustained sends to unopened streams are a protocol
// error/backpressure failure.
#define IREE_NET_TCP_STREAM_PENDING_FRAME_LIMIT 8

typedef struct iree_net_tcp_pending_frame_t {
  // Next pending frame for the same stream slot.
  struct iree_net_tcp_pending_frame_t* next;
  // Payload byte count stored in |payload|.
  iree_host_size_t payload_length;
  // Copied payload bytes delivered when the stream activates.
  uint8_t payload[];
} iree_net_tcp_pending_frame_t;

// Per-stream state within a TCP connection.
//
// Each stream slot maps to one message endpoint visible to the caller. The
// connection dispatches incoming frames to the appropriate stream based on
// the stream_id field in the frame header.
typedef struct iree_net_tcp_stream_t {
  // Back-pointer for vtable dispatch (recover connection from stream).
  iree_net_tcp_connection_t* connection;
  // Message and error handlers set by the endpoint consumer.
  iree_net_message_endpoint_callbacks_t callbacks;
  // Coordinates stream and connection deactivation requests.
  iree_net_endpoint_lifecycle_t lifecycle;
  // Frames received before this stream was activated locally.
  iree_net_tcp_pending_frame_t* pending_head;
  // Tail pointer for O(1) pending frame append.
  iree_net_tcp_pending_frame_t* pending_tail;
  // Number of queued pending frames.
  iree_host_size_t pending_count;
  // Whether this stream is actively receiving frames.
  bool active;
} iree_net_tcp_stream_t;

// TCP connection with a flexible-length stream table.
//
// The transport stack is built eagerly at connection creation:
//   carrier → framing_adapter → mux dispatch → per-stream callbacks
//
// The framing adapter handles frame boundary detection on the TCP byte stream.
// Mux dispatch reads the stream_id from each frame header and forwards to the
// appropriate stream slot's callbacks.
typedef struct iree_net_tcp_connection_t {
  iree_net_connection_t base;
  iree_net_tcp_factory_t* factory;
  iree_async_proactor_t* proactor;
  // Referenced, not owned. Must outlive the connection.
  iree_async_buffer_pool_t* recv_pool;
  // Owns the carrier (releases it when freed).
  iree_net_framing_adapter_t* adapter;
  // Cached borrowed endpoint from the framing adapter.
  iree_net_message_endpoint_t shared_endpoint;
  // Stream table sizing.
  uint16_t max_stream_count;
  uint16_t allocated_stream_count;
  uint16_t activated_stream_count;
  // Joins stream and carrier drains during connection deactivation.
  iree_net_endpoint_deactivation_barrier_t deactivation_barrier;
  // FAM: one slot per stream, sized by max_endpoint_count.
  iree_net_tcp_stream_t streams[];
} iree_net_tcp_connection_t;

typedef enum iree_net_tcp_listener_state_e {
  IREE_NET_TCP_LISTENER_STATE_LISTENING = 0,
  IREE_NET_TCP_LISTENER_STATE_STOPPING,
  IREE_NET_TCP_LISTENER_STATE_STOPPED,
} iree_net_tcp_listener_state_t;

typedef struct iree_net_tcp_listener_t {
  iree_net_listener_t base;
  iree_net_tcp_factory_t* factory;
  iree_async_proactor_t* proactor;
  iree_async_buffer_pool_t* recv_pool;
  iree_async_socket_t* listen_socket;
  struct {
    iree_net_listener_accept_callback_t fn;
    void* user_data;
  } accept;
  iree_async_socket_accept_operation_t accept_operation;
  iree_net_tcp_listener_state_t state;
  iree_net_listener_stopped_callback_t stopped_callback;
  iree_allocator_t host_allocator;
} iree_net_tcp_listener_t;

// Heap-allocated state for an in-flight async connect operation.
typedef struct iree_net_tcp_connect_state_t {
  iree_async_socket_connect_operation_t connect_operation;
  struct {
    iree_net_transport_connect_callback_t fn;
    void* user_data;
  } callback;
  iree_net_tcp_factory_t* factory;
  iree_async_proactor_t* proactor;
  iree_async_buffer_pool_t* recv_pool;
  iree_async_socket_t* socket;
  iree_async_address_t remote_address;
  iree_allocator_t host_allocator;
} iree_net_tcp_connect_state_t;

// Heap-allocated state for deferred async endpoint delivery via NOP.
typedef struct iree_net_tcp_endpoint_deferred_t {
  iree_async_nop_operation_t nop;
  iree_net_endpoint_ready_callback_t endpoint_ready;
  iree_net_message_endpoint_t endpoint;
  iree_allocator_t host_allocator;
} iree_net_tcp_endpoint_deferred_t;

// All IREE frames share a 16-byte header layout:
//   [0..3]   magic (4 bytes)
//   [4..7]   flags/version (4 bytes)
//   [8..11]  payload_length (uint32 LE)
//   [12..13] stream_id (uint16 LE)
//   [14..15] reserved
//
// Total frame size = header + payload_length.
#define IREE_NET_TCP_FRAME_HEADER_SIZE 16

// Maximum number of spans in the original send params that we can handle
// without a separate span-list allocation. The +1 accounts for the prepended
// header span.
#define IREE_NET_TCP_STREAM_SEND_MAX_INLINE_SPANS 15

typedef struct iree_net_tcp_stream_send_context_t {
  // Allocator used for this context.
  iree_allocator_t host_allocator;
  // User data from the endpoint send request.
  uint64_t operation_user_data;
  // Stable storage for the TCP mux frame header.
  uint8_t header[IREE_NET_TCP_FRAME_HEADER_SIZE];
  // Stable storage for the send scatter-gather list.
  iree_async_span_t spans[IREE_NET_TCP_STREAM_SEND_MAX_INLINE_SPANS + 1];
} iree_net_tcp_stream_send_context_t;

typedef struct iree_net_tcp_stream_deactivate_context_t {
  // NOP operation used to deliver a non-last-stream deactivate asynchronously.
  iree_async_nop_operation_t nop;
  // Stream whose deactivate callback should be delivered.
  iree_net_tcp_stream_t* stream;
  // Allocator used for this context.
  iree_allocator_t host_allocator;
} iree_net_tcp_stream_deactivate_context_t;

//===----------------------------------------------------------------------===//
// Frame header
//===----------------------------------------------------------------------===//

// Determines frame boundaries from the header's payload_length field.
static iree_host_size_t iree_net_tcp_frame_length(
    void* user_data, iree_const_byte_span_t available) {
  (void)user_data;
  if (available.data_length < 12) return 0;
  uint32_t payload_length;
  memcpy(&payload_length, available.data + 8, sizeof(payload_length));
  return IREE_NET_TCP_FRAME_HEADER_SIZE + (iree_host_size_t)payload_length;
}

//===----------------------------------------------------------------------===//
// Mux dispatch
//===----------------------------------------------------------------------===//

static void iree_net_tcp_stream_clear_pending_frames(
    iree_net_tcp_stream_t* stream, iree_allocator_t host_allocator) {
  iree_net_tcp_pending_frame_t* pending_frame = stream->pending_head;
  while (pending_frame) {
    iree_net_tcp_pending_frame_t* next_frame = pending_frame->next;
    iree_allocator_free(host_allocator, pending_frame);
    pending_frame = next_frame;
  }
  stream->pending_head = NULL;
  stream->pending_tail = NULL;
  stream->pending_count = 0;
}

static iree_status_t iree_net_tcp_stream_enqueue_pending_frame(
    iree_net_tcp_stream_t* stream, iree_const_byte_span_t payload) {
  if (stream->pending_count >= IREE_NET_TCP_STREAM_PENDING_FRAME_LIMIT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "stream has %" PRIhsz
                            " pending frames before activation",
                            stream->pending_count);
  }

  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(
      IREE_STRUCT_LAYOUT(sizeof(iree_net_tcp_pending_frame_t), &total_size,
                         IREE_STRUCT_FIELD_FAM(payload.data_length, uint8_t)));

  iree_net_tcp_connection_t* connection = stream->connection;
  iree_allocator_t host_allocator = connection->base.host_allocator;
  iree_net_tcp_pending_frame_t* pending_frame = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, total_size,
                                             (void**)&pending_frame));
  memset(pending_frame, 0, sizeof(*pending_frame));
  pending_frame->payload_length = payload.data_length;
  memcpy(pending_frame->payload, payload.data, payload.data_length);

  if (stream->pending_tail) {
    stream->pending_tail->next = pending_frame;
  } else {
    stream->pending_head = pending_frame;
  }
  stream->pending_tail = pending_frame;
  ++stream->pending_count;
  return iree_ok_status();
}

static iree_status_t iree_net_tcp_stream_deliver_pending_frames(
    iree_net_tcp_stream_t* stream) {
  iree_net_tcp_connection_t* connection = stream->connection;
  iree_allocator_t host_allocator = connection->base.host_allocator;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && stream->pending_head) {
    iree_net_tcp_pending_frame_t* pending_frame = stream->pending_head;
    stream->pending_head = pending_frame->next;
    if (!stream->pending_head) stream->pending_tail = NULL;
    --stream->pending_count;

    iree_async_buffer_lease_t lease;
    status = iree_async_buffer_pool_acquire(connection->recv_pool, &lease);
    if (iree_status_is_ok(status)) {
      if (pending_frame->payload_length > lease.span.length) {
        status =
            iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "pending stream frame length %" PRIhsz
                             " exceeds receive buffer length %" PRIhsz,
                             pending_frame->payload_length, lease.span.length);
      } else {
        uint8_t* payload_data = iree_async_span_ptr(lease.span);
        memcpy(payload_data, pending_frame->payload,
               pending_frame->payload_length);
        iree_const_byte_span_t payload = iree_make_const_byte_span(
            payload_data, pending_frame->payload_length);
        status = stream->callbacks.on_message(stream->callbacks.user_data,
                                              payload, &lease);
      }
      iree_async_buffer_lease_release(&lease);
    }
    iree_allocator_free(host_allocator, pending_frame);
  }
  if (!iree_status_is_ok(status)) {
    iree_net_tcp_stream_clear_pending_frames(stream, host_allocator);
  }
  return status;
}

// Dispatches a complete frame to the appropriate stream based on stream_id.
//
// Reads the uint16_t stream_id from the frame header at offset 12, then strips
// the header and delivers only the payload to the per-stream callback. This
// makes the per-stream endpoint API transparent about framing: consumers send
// and receive raw payloads without knowledge of the wire format.
//
// Frames for unknown or inactive streams are dropped with a diagnostic status.
static iree_status_t iree_net_tcp_mux_dispatch(
    void* user_data, iree_const_byte_span_t message,
    iree_async_buffer_lease_t* lease) {
  iree_net_tcp_connection_t* connection = (iree_net_tcp_connection_t*)user_data;
  // NOTE: lease is borrowed — the frame_accumulator releases it on error
  // return. Do NOT release the lease on error paths here.
  if (message.data_length < IREE_NET_TCP_FRAME_HEADER_SIZE) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "frame too short for stream_id dispatch: %" PRIhsz
                            " bytes",
                            message.data_length);
  }
  uint16_t stream_id;
  memcpy(&stream_id, message.data + 12, sizeof(stream_id));
  if (stream_id >= connection->max_stream_count) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "no handler for stream_id %u", (unsigned)stream_id);
  }
  iree_net_tcp_stream_t* stream = &connection->streams[stream_id];
  // Strip the frame header — deliver only the payload to the consumer.
  iree_const_byte_span_t payload = iree_make_const_byte_span(
      message.data + IREE_NET_TCP_FRAME_HEADER_SIZE,
      message.data_length - IREE_NET_TCP_FRAME_HEADER_SIZE);
  if (!stream->active) {
    return iree_net_tcp_stream_enqueue_pending_frame(stream, payload);
  }
  if (!stream->callbacks.on_message) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream_id %u has no message handler",
                            (unsigned)stream_id);
  }
  return stream->callbacks.on_message(stream->callbacks.user_data, payload,
                                      lease);
}

// Propagates a transport error to all active streams.
static void iree_net_tcp_mux_error(void* user_data, iree_status_t status) {
  iree_net_tcp_connection_t* connection = (iree_net_tcp_connection_t*)user_data;
  for (uint16_t i = 0; i < connection->max_stream_count; ++i) {
    if (connection->streams[i].active &&
        connection->streams[i].callbacks.on_error) {
      connection->streams[i].callbacks.on_error(
          connection->streams[i].callbacks.user_data,
          iree_status_clone(status));
    }
  }
  iree_status_ignore(status);
}

static void iree_net_tcp_connection_on_send_complete(
    void* callback_user_data, iree_net_carrier_completion_kind_t kind,
    uint64_t operation_user_data, iree_status_t status,
    iree_host_size_t bytes_transferred, iree_async_buffer_lease_t* recv_lease) {
  (void)callback_user_data;
  if (kind != IREE_NET_CARRIER_COMPLETION_SEND || operation_user_data == 0) {
    iree_status_ignore(status);
    return;
  }
  iree_net_tcp_stream_send_context_t* context =
      (iree_net_tcp_stream_send_context_t*)(uintptr_t)operation_user_data;
  uint64_t nested_operation_user_data = context->operation_user_data;
  iree_allocator_t host_allocator = context->host_allocator;
  iree_allocator_free(host_allocator, context);

  if (nested_operation_user_data != 0) {
    iree_net_frame_sender_dispatch_carrier_completion(
        /*callback_user_data=*/NULL, kind, nested_operation_user_data, status,
        bytes_transferred, recv_lease);
  } else {
    iree_status_ignore(status);
  }
}

//===----------------------------------------------------------------------===//
// Per-stream endpoint vtable
//===----------------------------------------------------------------------===//

static void iree_net_tcp_stream_set_callbacks(
    void* self, iree_net_message_endpoint_callbacks_t callbacks) {
  iree_net_tcp_stream_t* stream = (iree_net_tcp_stream_t*)self;
  stream->callbacks = callbacks;
}

static iree_status_t iree_net_tcp_stream_activate(void* self) {
  iree_net_tcp_stream_t* stream = (iree_net_tcp_stream_t*)self;
  iree_net_tcp_connection_t* connection = stream->connection;
  if (!stream->callbacks.on_message) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "callbacks must be set before activation");
  }
  IREE_RETURN_IF_ERROR(
      iree_net_endpoint_lifecycle_activate(&stream->lifecycle));
  bool first_activation = (connection->activated_stream_count == 0);

  // First stream activation triggers the underlying adapter activation.
  iree_status_t status = iree_ok_status();
  if (first_activation) {
    status = iree_net_message_endpoint_activate(connection->shared_endpoint);
  }
  if (iree_status_is_ok(status)) {
    stream->active = true;
    ++connection->activated_stream_count;
    status = iree_net_tcp_stream_deliver_pending_frames(stream);
  } else {
    iree_net_endpoint_lifecycle_rollback_activation(&stream->lifecycle);
  }
  return status;
}

// NOP completion for non-last deactivation (delivers callback async).
static void iree_net_tcp_stream_deactivate_nop_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_tcp_stream_deactivate_context_t* context =
      (iree_net_tcp_stream_deactivate_context_t*)user_data;
  iree_net_tcp_stream_t* stream = context->stream;
  iree_allocator_t host_allocator = context->host_allocator;
  if (!iree_status_is_ok(status)) iree_status_abort(status);
  iree_allocator_free(host_allocator, context);
  iree_net_endpoint_lifecycle_complete_deactivation(&stream->lifecycle);
}

// Completes the last stream after the shared framing adapter has drained.
static void iree_net_tcp_stream_shared_endpoint_deactivated(void* user_data) {
  iree_net_tcp_stream_t* stream = (iree_net_tcp_stream_t*)user_data;
  iree_net_endpoint_lifecycle_complete_deactivation(&stream->lifecycle);
}

static iree_status_t iree_net_tcp_stream_deactivate(
    void* self, iree_net_message_endpoint_deactivate_fn_t callback,
    void* user_data) {
  iree_net_tcp_stream_t* stream = (iree_net_tcp_stream_t*)self;
  iree_net_tcp_connection_t* connection = stream->connection;
  if (!stream->active) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream not active");
  }
  bool requires_deferred_callback = connection->activated_stream_count > 1;
  iree_net_tcp_stream_deactivate_context_t* context = NULL;
  iree_allocator_t host_allocator = connection->base.host_allocator;
  if (requires_deferred_callback) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*context),
                                               (void**)&context));
    memset(context, 0, sizeof(*context));
    context->stream = stream;
    context->host_allocator = host_allocator;
  }

  iree_net_endpoint_lifecycle_actions_t actions =
      IREE_NET_ENDPOINT_LIFECYCLE_ACTION_NONE;
  iree_status_t status = iree_net_endpoint_lifecycle_request_deactivation(
      &stream->lifecycle, callback, user_data, &actions);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, context);
    return status;
  }
  stream->active = false;
  --connection->activated_stream_count;

  // Last stream deactivation triggers the underlying adapter deactivation.
  if (connection->activated_stream_count == 0) {
    return iree_net_message_endpoint_deactivate(
        connection->shared_endpoint,
        iree_net_tcp_stream_shared_endpoint_deactivated, stream);
  }

  // Non-last: deliver callback asynchronously via NOP.
  iree_async_operation_initialize(
      &context->nop.base, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_net_tcp_stream_deactivate_nop_complete, context);
  status =
      iree_async_proactor_submit_one(connection->proactor, &context->nop.base);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, context);
    iree_net_endpoint_lifecycle_complete_deactivation(&stream->lifecycle);
  }
  return status;
}

static iree_status_t iree_net_tcp_stream_send(
    void* self, const iree_net_message_endpoint_send_params_t* params) {
  iree_net_tcp_stream_t* stream = (iree_net_tcp_stream_t*)self;
  iree_net_tcp_connection_t* connection = stream->connection;

  // Compute total payload length across all input spans.
  uint32_t payload_length = 0;
  for (iree_host_size_t i = 0; i < params->data.count; ++i) {
    payload_length += (uint32_t)params->data.values[i].length;
  }

  if (params->data.count > IREE_NET_TCP_STREAM_SEND_MAX_INLINE_SPANS) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "send span count %" PRIhsz " exceeds inline limit %d",
        params->data.count, IREE_NET_TCP_STREAM_SEND_MAX_INLINE_SPANS);
  }

  iree_allocator_t host_allocator = connection->base.host_allocator;
  iree_net_tcp_stream_send_context_t* context = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*context),
                                             (void**)&context));
  memset(context, 0, sizeof(*context));
  context->host_allocator = host_allocator;
  context->operation_user_data = params->user_data;

  memcpy(context->header + 8, &payload_length, sizeof(payload_length));
  uint16_t stream_id =
      (uint16_t)(stream - connection->streams);  // Index from FAM base.
  memcpy(context->header + 12, &stream_id, sizeof(stream_id));

  context->spans[0] =
      iree_async_span_from_ptr(context->header, sizeof(context->header));
  for (iree_host_size_t i = 0; i < params->data.count; ++i) {
    context->spans[i + 1] = params->data.values[i];
  }

  iree_host_size_t total_span_count = params->data.count + 1;
  iree_net_message_endpoint_send_params_t framed_params = {
      .data = iree_async_span_list_make(context->spans, total_span_count),
      .user_data = (uint64_t)(uintptr_t)context,
  };
  iree_status_t status = iree_net_message_endpoint_send(
      connection->shared_endpoint, &framed_params);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, context);
  }
  return status;
}

static iree_net_carrier_send_budget_t iree_net_tcp_stream_query_send_budget(
    void* self) {
  iree_net_tcp_stream_t* stream = (iree_net_tcp_stream_t*)self;
  return iree_net_message_endpoint_query_send_budget(
      stream->connection->shared_endpoint);
}

// Reserves space for a contiguous send with the 16-byte frame header prepended.
//
// The underlying carrier allocates |IREE_NET_TCP_FRAME_HEADER_SIZE + size|
// bytes, the stream writes the frame header into the first 16 bytes, and the
// caller receives a pointer past the header where it writes the payload. On
// commit, the entire buffer (header + payload) is published as one contiguous
// frame. The carrier handle passes through unchanged.
static iree_status_t iree_net_tcp_stream_begin_send(
    void* self, iree_host_size_t size, void** out_ptr,
    iree_net_carrier_send_handle_t* out_handle) {
  iree_net_tcp_stream_t* stream = (iree_net_tcp_stream_t*)self;
  iree_net_tcp_connection_t* connection = stream->connection;

  // Reserve header + payload in the underlying carrier.
  void* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_net_message_endpoint_begin_send(
      connection->shared_endpoint, IREE_NET_TCP_FRAME_HEADER_SIZE + size,
      &buffer, out_handle));

  // Write the 16-byte frame header at the start of the buffer.
  uint32_t payload_length = (uint32_t)size;
  uint16_t stream_id =
      (uint16_t)(stream - connection->streams);  // Index from FAM base.
  uint8_t* header = (uint8_t*)buffer;
  memset(header, 0, IREE_NET_TCP_FRAME_HEADER_SIZE);
  memcpy(header + 8, &payload_length, sizeof(payload_length));
  memcpy(header + 12, &stream_id, sizeof(stream_id));

  // Caller writes payload after the header.
  *out_ptr = header + IREE_NET_TCP_FRAME_HEADER_SIZE;
  return iree_ok_status();
}

// Publishes a previously reserved send. The frame header was already written
// by begin_send; the caller has filled in the payload. The carrier handle
// passes through to the shared endpoint.
static iree_status_t iree_net_tcp_stream_commit_send(
    void* self, iree_net_carrier_send_handle_t handle) {
  iree_net_tcp_stream_t* stream = (iree_net_tcp_stream_t*)self;
  return iree_net_message_endpoint_commit_send(
      stream->connection->shared_endpoint, handle);
}

// Discards a previously reserved send without publishing any data.
static void iree_net_tcp_stream_abort_send(
    void* self, iree_net_carrier_send_handle_t handle) {
  iree_net_tcp_stream_t* stream = (iree_net_tcp_stream_t*)self;
  iree_net_message_endpoint_abort_send(stream->connection->shared_endpoint,
                                       handle);
}

static const iree_net_message_endpoint_vtable_t
    iree_net_tcp_stream_endpoint_vtable = {
        .set_callbacks = iree_net_tcp_stream_set_callbacks,
        .activate = iree_net_tcp_stream_activate,
        .deactivate = iree_net_tcp_stream_deactivate,
        .send = iree_net_tcp_stream_send,
        .query_send_budget = iree_net_tcp_stream_query_send_budget,
        .begin_send = iree_net_tcp_stream_begin_send,
        .commit_send = iree_net_tcp_stream_commit_send,
        .abort_send = iree_net_tcp_stream_abort_send,
};

//===----------------------------------------------------------------------===//
// Connection
//===----------------------------------------------------------------------===//

static const iree_net_connection_vtable_t iree_net_tcp_connection_vtable;

static void iree_net_tcp_connection_deactivate(
    iree_net_connection_t* base_connection,
    iree_net_connection_deactivate_callback_t callback) {
  iree_net_tcp_connection_t* connection =
      (iree_net_tcp_connection_t*)base_connection;

  iree_net_connection_retain(base_connection);
  iree_net_endpoint_deactivation_barrier_initialize(
      callback, &connection->deactivation_barrier);

  // Join stream-level callbacks before joining the shared carrier drain. Any
  // active stream has no independent work to drain; its lifetime remains held
  // by the shared framing adapter barrier entry.
  for (uint16_t i = 0; i < connection->max_stream_count; ++i) {
    iree_net_tcp_stream_t* stream = &connection->streams[i];
    iree_net_endpoint_lifecycle_actions_t actions =
        iree_net_endpoint_lifecycle_join_deactivation(
            &stream->lifecycle, &connection->deactivation_barrier);
    stream->active = false;
    iree_net_tcp_stream_clear_pending_frames(stream,
                                             connection->base.host_allocator);
    if (iree_any_bit_set(
            actions, IREE_NET_ENDPOINT_LIFECYCLE_ACTION_BEGIN_DEACTIVATION)) {
      iree_net_endpoint_lifecycle_complete_deactivation(&stream->lifecycle);
    }
  }
  connection->activated_stream_count = 0;

  iree_net_framing_adapter_join_deactivation(connection->adapter,
                                             &connection->deactivation_barrier);
  iree_net_endpoint_deactivation_barrier_commit(
      &connection->deactivation_barrier);
  iree_net_connection_release(base_connection);
}

static void iree_net_tcp_connection_destroy(
    iree_net_connection_t* base_connection) {
  iree_net_tcp_connection_t* connection =
      (iree_net_tcp_connection_t*)base_connection;
  iree_allocator_t host_allocator = connection->base.host_allocator;
  IREE_ASSERT(connection->activated_stream_count == 0,
              "connection destroyed with %u active streams; "
              "call iree_net_connection_deactivate before releasing",
              (unsigned)connection->activated_stream_count);
  // Adapter owns the carrier — freeing it releases both.
  for (uint16_t i = 0; i < connection->max_stream_count; ++i) {
    iree_net_tcp_stream_t* stream = &connection->streams[i];
    iree_net_tcp_stream_clear_pending_frames(stream, host_allocator);
    iree_net_endpoint_lifecycle_deinitialize(&stream->lifecycle);
  }
  if (connection->adapter) {
    iree_net_framing_adapter_free(connection->adapter);
  }
  iree_allocator_free(host_allocator, connection);
}

static iree_status_t iree_net_tcp_connection_create(
    iree_net_tcp_factory_t* factory, iree_async_proactor_t* proactor,
    iree_async_buffer_pool_t* recv_pool, iree_async_socket_t* socket,
    iree_allocator_t host_allocator, iree_net_connection_t** out_connection) {
  IREE_ASSERT_ARGUMENT(out_connection);
  *out_connection = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  uint16_t max_stream_count = factory->default_options.max_endpoint_count;

  // Overflow-checked FAM allocation.
  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(iree_net_tcp_connection_t), &total_size,
              IREE_STRUCT_FIELD_FAM(max_stream_count, iree_net_tcp_stream_t)));

  iree_net_tcp_connection_t* connection = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&connection);

  // Post-allocation: build transport stack.
  iree_net_carrier_t* carrier = NULL;
  if (iree_status_is_ok(status)) {
    memset(connection, 0, total_size);
    iree_net_connection_initialize(&iree_net_tcp_connection_vtable,
                                   host_allocator, max_stream_count,
                                   &connection->base);
    connection->factory = factory;
    connection->proactor = proactor;
    connection->recv_pool = recv_pool;
    connection->max_stream_count = max_stream_count;

    // Initialize stream back-pointers.
    for (uint16_t i = 0; i < max_stream_count; ++i) {
      connection->streams[i].connection = connection;
      iree_net_endpoint_lifecycle_initialize(&connection->streams[i].lifecycle);
    }

    // Create TCP carrier from connected socket.
    // The send completion callback dispatches to frame_sender for channels that
    // use completion-tracked sends.
    iree_net_carrier_callback_t send_callback = {
        .fn = iree_net_tcp_connection_on_send_complete,
        .user_data = connection,
    };
    status = iree_net_tcp_carrier_create(
        proactor, socket, recv_pool, factory->default_options, send_callback,
        host_allocator, &carrier);
  }

  // Create framing adapter over the carrier.
  if (iree_status_is_ok(status)) {
    iree_host_size_t max_frame_size =
        iree_async_buffer_pool_buffer_size(recv_pool);
    iree_net_frame_length_callback_t frame_length = {
        .fn = iree_net_tcp_frame_length,
        .user_data = NULL,
    };
    status = iree_net_framing_adapter_allocate(
        carrier, frame_length, max_frame_size, recv_pool, host_allocator,
        &connection->adapter);
    if (iree_status_is_ok(status)) {
      // Carrier ownership transferred to adapter.
      carrier = NULL;
    }
  }

  // Cache endpoint and set mux dispatch callbacks.
  if (iree_status_is_ok(status)) {
    connection->shared_endpoint =
        iree_net_framing_adapter_as_endpoint(connection->adapter);
    iree_net_message_endpoint_set_callbacks(
        connection->shared_endpoint,
        (iree_net_message_endpoint_callbacks_t){
            .on_message = iree_net_tcp_mux_dispatch,
            .on_error = iree_net_tcp_mux_error,
            .user_data = connection,
        });
  }

  if (iree_status_is_ok(status)) {
    *out_connection = &connection->base;
  } else {
    // Cleanup on failure. Carrier may or may not have been transferred to the
    // adapter, so release whichever still exists.
    iree_net_carrier_release(carrier);
    if (connection) iree_net_tcp_connection_destroy(&connection->base);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// NOP completion for async endpoint delivery.
static void iree_net_tcp_endpoint_deferred_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_tcp_endpoint_deferred_t* deferred =
      (iree_net_tcp_endpoint_deferred_t*)user_data;
  iree_status_ignore(status);
  deferred->endpoint_ready.fn(deferred->endpoint_ready.user_data,
                              iree_ok_status(), deferred->endpoint);
  iree_allocator_free(deferred->host_allocator, deferred);
}

static iree_status_t iree_net_tcp_connection_open_endpoint(
    iree_net_connection_t* base_connection,
    iree_net_endpoint_ready_callback_t callback) {
  iree_net_tcp_connection_t* connection =
      (iree_net_tcp_connection_t*)base_connection;

  if (connection->allocated_stream_count >= connection->max_stream_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "all %u endpoint slots allocated",
                            (unsigned)connection->max_stream_count);
  }

  uint16_t stream_id = connection->allocated_stream_count++;
  iree_net_message_endpoint_t endpoint = {
      .self = &connection->streams[stream_id],
      .vtable = &iree_net_tcp_stream_endpoint_vtable,
  };

  // Deliver callback asynchronously via NOP to ensure it fires on the proactor
  // thread (consistent with loopback carrier and safe for re-entrancy).
  iree_net_tcp_endpoint_deferred_t* deferred = NULL;
  iree_allocator_t host_allocator = connection->base.host_allocator;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*deferred),
                                             (void**)&deferred));
  memset(deferred, 0, sizeof(*deferred));
  deferred->endpoint_ready = callback;
  deferred->endpoint = endpoint;
  deferred->host_allocator = host_allocator;

  iree_async_operation_initialize(
      &deferred->nop.base, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_tcp_endpoint_deferred_complete,
      deferred);
  iree_status_t status =
      iree_async_proactor_submit_one(connection->proactor, &deferred->nop.base);
  if (!iree_status_is_ok(status)) {
    --connection->allocated_stream_count;
    iree_allocator_free(host_allocator, deferred);
  }
  return status;
}

static iree_net_carrier_t* iree_net_tcp_connection_carrier(
    iree_net_connection_t* base_connection) {
  iree_net_tcp_connection_t* connection =
      (iree_net_tcp_connection_t*)base_connection;
  return iree_net_framing_adapter_carrier(connection->adapter);
}

static iree_async_proactor_t* iree_net_tcp_connection_proactor(
    iree_net_connection_t* base_connection) {
  iree_net_tcp_connection_t* connection =
      (iree_net_tcp_connection_t*)base_connection;
  return connection->proactor;
}

static const iree_net_connection_vtable_t iree_net_tcp_connection_vtable = {
    .destroy = iree_net_tcp_connection_destroy,
    .deactivate = iree_net_tcp_connection_deactivate,
    .open_endpoint = iree_net_tcp_connection_open_endpoint,
    .proactor = iree_net_tcp_connection_proactor,
    .carrier = iree_net_tcp_connection_carrier,
};

//===----------------------------------------------------------------------===//
// Listener
//===----------------------------------------------------------------------===//

static const iree_net_listener_vtable_t iree_net_tcp_listener_vtable;

static void iree_net_tcp_listener_free(iree_net_listener_t* base_listener) {
  iree_net_tcp_listener_t* listener = (iree_net_tcp_listener_t*)base_listener;
  iree_allocator_t host_allocator = listener->host_allocator;
  iree_async_socket_release(listener->listen_socket);
  iree_allocator_free(host_allocator, listener);
}

// Accept completion callback. Fires for each accepted connection (multishot)
// or once per accept (single-shot, then resubmitted).
static void iree_net_tcp_listener_accept_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_tcp_listener_t* listener = (iree_net_tcp_listener_t*)user_data;
  iree_async_socket_accept_operation_t* accept_op =
      (iree_async_socket_accept_operation_t*)operation;

  if (listener->state == IREE_NET_TCP_LISTENER_STATE_STOPPING) {
    if (iree_status_is_ok(status) && accept_op->accepted_socket) {
      // A connection arrived between stop() and cancellation completing.
      // Deliver it — the application requested stop, not "reject pending
      // connections". The connection was already accepted by the kernel.
      iree_net_connection_t* connection = NULL;
      iree_status_t create_status = iree_net_tcp_connection_create(
          listener->factory, listener->proactor, listener->recv_pool,
          accept_op->accepted_socket, listener->host_allocator, &connection);
      if (iree_status_is_ok(create_status)) {
        accept_op->accepted_socket = NULL;  // Ownership transferred.
        listener->accept.fn(listener->accept.user_data, iree_ok_status(),
                            connection);
      } else {
        iree_async_socket_release(accept_op->accepted_socket);
        accept_op->accepted_socket = NULL;
        iree_status_ignore(create_status);
      }
    } else {
      iree_status_ignore(status);
    }

    // If this is the final completion (no MORE flag), fire stopped callback.
    if (!(flags & IREE_ASYNC_COMPLETION_FLAG_MORE)) {
      listener->state = IREE_NET_TCP_LISTENER_STATE_STOPPED;
      if (listener->stopped_callback.fn) {
        listener->stopped_callback.fn(listener->stopped_callback.user_data);
      }
    }
    return;
  }

  // Normal LISTENING state.
  if (!iree_status_is_ok(status)) {
    // Accept error — deliver to application as a failed accept.
    listener->accept.fn(listener->accept.user_data, status, NULL);
    // For non-fatal errors in single-shot mode, resubmit if still listening.
    // Multishot handles this internally.
    return;
  }

  // Wrap the accepted socket in a connection.
  iree_net_connection_t* connection = NULL;
  iree_status_t create_status = iree_net_tcp_connection_create(
      listener->factory, listener->proactor, listener->recv_pool,
      accept_op->accepted_socket, listener->host_allocator, &connection);
  if (iree_status_is_ok(create_status)) {
    accept_op->accepted_socket = NULL;  // Ownership transferred.
    listener->accept.fn(listener->accept.user_data, iree_ok_status(),
                        connection);
  } else {
    iree_async_socket_release(accept_op->accepted_socket);
    accept_op->accepted_socket = NULL;
    listener->accept.fn(listener->accept.user_data, create_status, NULL);
  }

  // For single-shot accept, resubmit to keep accepting.
  if (!(flags & IREE_ASYNC_COMPLETION_FLAG_MORE) &&
      listener->state == IREE_NET_TCP_LISTENER_STATE_LISTENING) {
    memset(&listener->accept_operation, 0, sizeof(listener->accept_operation));
    iree_async_operation_initialize(
        &listener->accept_operation.base,
        IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT, IREE_ASYNC_OPERATION_FLAG_NONE,
        iree_net_tcp_listener_accept_complete, listener);
    listener->accept_operation.listen_socket = listener->listen_socket;
    iree_status_t submit_status = iree_async_proactor_submit_one(
        listener->proactor, &listener->accept_operation.base);
    if (!iree_status_is_ok(submit_status)) {
      listener->accept.fn(listener->accept.user_data, submit_status, NULL);
    }
  }
}

static iree_status_t iree_net_tcp_listener_stop(
    iree_net_listener_t* base_listener,
    iree_net_listener_stopped_callback_t callback) {
  iree_net_tcp_listener_t* listener = (iree_net_tcp_listener_t*)base_listener;

  listener->state = IREE_NET_TCP_LISTENER_STATE_STOPPING;
  listener->stopped_callback = callback;

  // Cancel the pending accept operation. The cancellation CQE will fire
  // the accept callback with CANCELLED status, which triggers the stopped
  // notification.
  return iree_async_proactor_cancel(listener->proactor,
                                    &listener->accept_operation.base);
}

static iree_status_t iree_net_tcp_listener_query_bound_address(
    iree_net_listener_t* base_listener, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_address) {
  iree_net_tcp_listener_t* listener = (iree_net_tcp_listener_t*)base_listener;
  iree_async_address_t address;
  IREE_RETURN_IF_ERROR(
      iree_async_socket_query_local_address(listener->listen_socket, &address));
  return iree_async_address_format(&address, buffer_capacity, buffer,
                                   out_address);
}

static const iree_net_listener_vtable_t iree_net_tcp_listener_vtable = {
    .free = iree_net_tcp_listener_free,
    .stop = iree_net_tcp_listener_stop,
    .query_bound_address = iree_net_tcp_listener_query_bound_address,
};

//===----------------------------------------------------------------------===//
// Factory
//===----------------------------------------------------------------------===//

static const iree_net_transport_factory_vtable_t iree_net_tcp_factory_vtable;

IREE_API_EXPORT iree_status_t
iree_net_tcp_factory_create(iree_net_tcp_carrier_options_t default_options,
                            iree_allocator_t host_allocator,
                            iree_net_transport_factory_t** out_factory) {
  IREE_ASSERT_ARGUMENT(out_factory);
  *out_factory = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_tcp_factory_t* factory = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*factory),
                                (void**)&factory));
  memset(factory, 0, sizeof(*factory));
  iree_atomic_ref_count_init(&factory->base.ref_count);
  factory->base.vtable = &iree_net_tcp_factory_vtable;
  factory->default_options = default_options;
  factory->host_allocator = host_allocator;

  *out_factory = &factory->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_net_tcp_factory_destroy(
    iree_net_transport_factory_t* base_factory) {
  iree_net_tcp_factory_t* factory = (iree_net_tcp_factory_t*)base_factory;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = factory->host_allocator;
  iree_allocator_free(host_allocator, factory);
  IREE_TRACE_ZONE_END(z0);
}

static iree_net_transport_capabilities_t
iree_net_tcp_factory_query_capabilities(
    iree_net_transport_factory_t* base_factory) {
  (void)base_factory;
  return IREE_NET_TRANSPORT_CAPABILITY_RELIABLE |
         IREE_NET_TRANSPORT_CAPABILITY_ORDERED;
}

// Connect completion callback for factory connect.
static void iree_net_tcp_connect_complete(void* user_data,
                                          iree_async_operation_t* operation,
                                          iree_status_t status,
                                          iree_async_completion_flags_t flags) {
  iree_net_tcp_connect_state_t* state =
      (iree_net_tcp_connect_state_t*)user_data;

  if (!iree_status_is_ok(status)) {
    // Connect failed — socket was not consumed, release it.
    iree_async_socket_release(state->socket);
    state->callback.fn(state->callback.user_data, status, NULL);
    iree_allocator_free(state->host_allocator, state);
    return;
  }

  // Wrap the connected socket in a connection.
  iree_net_connection_t* connection = NULL;
  iree_status_t create_status = iree_net_tcp_connection_create(
      state->factory, state->proactor, state->recv_pool, state->socket,
      state->host_allocator, &connection);
  if (iree_status_is_ok(create_status)) {
    state->callback.fn(state->callback.user_data, iree_ok_status(), connection);
  } else {
    iree_async_socket_release(state->socket);
    state->callback.fn(state->callback.user_data, create_status, NULL);
  }
  iree_allocator_free(state->host_allocator, state);
}

static iree_status_t iree_net_tcp_factory_connect(
    iree_net_transport_factory_t* base_factory, iree_string_view_t address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_transport_connect_callback_t callback, void* user_data) {
  iree_net_tcp_factory_t* factory = (iree_net_tcp_factory_t*)base_factory;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Parse address synchronously — invalid format is a synchronous error.
  iree_async_address_t remote_address;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_address_from_string(address, &remote_address));

  // Create socket.
  iree_async_socket_t* socket = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_socket_create(proactor, IREE_ASYNC_SOCKET_TYPE_TCP,
                                   IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR |
                                       IREE_ASYNC_SOCKET_OPTION_NO_DELAY,
                                   &socket));

  // Allocate connect state.
  iree_net_tcp_connect_state_t* state = NULL;
  iree_status_t status = iree_allocator_malloc(factory->host_allocator,
                                               sizeof(*state), (void**)&state);
  if (!iree_status_is_ok(status)) {
    iree_async_socket_release(socket);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  memset(state, 0, sizeof(*state));
  state->callback.fn = callback;
  state->callback.user_data = user_data;
  state->factory = factory;
  state->proactor = proactor;
  state->recv_pool = recv_pool;
  state->socket = socket;
  state->remote_address = remote_address;
  state->host_allocator = factory->host_allocator;

  // Initialize and submit async connect.
  iree_async_operation_initialize(
      &state->connect_operation.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_tcp_connect_complete, state);
  state->connect_operation.socket = socket;
  state->connect_operation.address = remote_address;
  status =
      iree_async_proactor_submit_one(proactor, &state->connect_operation.base);
  if (!iree_status_is_ok(status)) {
    iree_async_socket_release(socket);
    iree_allocator_free(factory->host_allocator, state);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_net_tcp_factory_create_listener(
    iree_net_transport_factory_t* base_factory, iree_string_view_t bind_address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_listener_accept_callback_t accept_callback, void* user_data,
    iree_allocator_t host_allocator, iree_net_listener_t** out_listener) {
  iree_net_tcp_factory_t* factory = (iree_net_tcp_factory_t*)base_factory;
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_listener = NULL;

  // Parse bind address.
  iree_async_address_t address;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_address_from_string(bind_address, &address));

  // Create, bind, and listen on a TCP socket.
  iree_async_socket_t* listen_socket = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_socket_create(proactor, IREE_ASYNC_SOCKET_TYPE_TCP,
                                   IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR,
                                   &listen_socket));

  iree_status_t status = iree_async_socket_bind(listen_socket, &address);
  if (iree_status_is_ok(status)) {
    status = iree_async_socket_listen(listen_socket, /*backlog=*/128);
  }
  if (!iree_status_is_ok(status)) {
    iree_async_socket_release(listen_socket);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Allocate listener.
  iree_net_tcp_listener_t* listener = NULL;
  status = iree_allocator_malloc(host_allocator, sizeof(*listener),
                                 (void**)&listener);
  if (!iree_status_is_ok(status)) {
    iree_async_socket_release(listen_socket);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  memset(listener, 0, sizeof(*listener));
  listener->base.vtable = &iree_net_tcp_listener_vtable;
  listener->factory = factory;
  listener->proactor = proactor;
  listener->recv_pool = recv_pool;
  listener->listen_socket = listen_socket;
  listener->accept.fn = accept_callback;
  listener->accept.user_data = user_data;
  listener->state = IREE_NET_TCP_LISTENER_STATE_LISTENING;
  listener->host_allocator = host_allocator;

  // Submit the first accept operation. Accept completions write results back
  // into listener->accept_operation and the callback consumes those fields, so
  // use single-shot accepts and resubmit after each accepted connection.
  iree_async_operation_flags_t accept_flags = IREE_ASYNC_OPERATION_FLAG_NONE;
  iree_async_operation_initialize(
      &listener->accept_operation.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT,
      accept_flags, iree_net_tcp_listener_accept_complete, listener);
  listener->accept_operation.listen_socket = listen_socket;
  status = iree_async_proactor_submit_one(proactor,
                                          &listener->accept_operation.base);
  if (!iree_status_is_ok(status)) {
    iree_async_socket_release(listen_socket);
    listener->listen_socket = NULL;  // Prevent double-release in free.
    iree_allocator_free(host_allocator, listener);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_listener = &listener->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static const iree_net_transport_factory_vtable_t iree_net_tcp_factory_vtable = {
    .destroy = iree_net_tcp_factory_destroy,
    .query_capabilities = iree_net_tcp_factory_query_capabilities,
    .connect = iree_net_tcp_factory_connect,
    .create_listener = iree_net_tcp_factory_create_listener,
};
