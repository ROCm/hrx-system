// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/queue/queue_channel.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/net/channel/util/frame_sender.h"
#include "iree/net/channel/util/send_gate.h"

// Maximum frontier entries per send. Covers rack-scale systems with hundreds
// of queues across multiple devices and machines. Increase if needed.
#define IREE_NET_QUEUE_CHANNEL_MAX_FRONTIER_ENTRIES 32

// Maximum size of the send-path header buffer: queue frame header + two
// serialized frontiers (wait + signal) at max entry count.
#define IREE_NET_QUEUE_CHANNEL_MAX_HEADER_SIZE        \
  (IREE_NET_QUEUE_FRAME_HEADER_SIZE +                 \
   2 * (sizeof(iree_async_frontier_t) +               \
        IREE_NET_QUEUE_CHANNEL_MAX_FRONTIER_ENTRIES * \
            sizeof(iree_async_frontier_entry_t)))

//===----------------------------------------------------------------------===//
// iree_net_queue_channel_t
//===----------------------------------------------------------------------===//

struct iree_net_queue_channel_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;

  // Borrowed view into the transport. Valid until detach zeroes it.
  // After detach, endpoint.self is NULL and sends return UNAVAILABLE.
  //
  // Concurrent access during send/detach is synchronized by send_gate.
  iree_net_message_endpoint_t endpoint;

  // Gates endpoint access and signals exact send quiescence during detach.
  iree_net_channel_send_gate_t send_gate;

  // Owned fallback header pool for unusually large scatter-gather sends. Freed
  // on channel destroy. The frame_sender borrows this pointer.
  iree_async_buffer_pool_t* header_pool;

  // Embedded frame sender for the send path.
  iree_net_frame_sender_t sender;

  iree_net_queue_channel_callbacks_t callbacks;

  // Lifecycle state. Written on proactor thread, read with atomic acquire.
  iree_atomic_int32_t state;
};

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

static iree_net_queue_channel_state_t iree_net_queue_channel_load_state(
    const iree_net_queue_channel_t* channel) {
  return (iree_net_queue_channel_state_t)iree_atomic_load(
      &((iree_net_queue_channel_t*)channel)->state, iree_memory_order_acquire);
}

static void iree_net_queue_channel_set_state(
    iree_net_queue_channel_t* channel,
    iree_net_queue_channel_state_t new_state) {
  iree_atomic_store(&channel->state, (int32_t)new_state,
                    iree_memory_order_release);
}

// Submit callback for frame_sender: routes sends through the message endpoint.
// Called from any thread (worker threads via barrier completion, proactor
// thread via direct sends).
//
static iree_status_t iree_net_queue_channel_submit_send(
    void* user_data, iree_async_span_list_t data, uint64_t send_user_data) {
  iree_net_queue_channel_t* channel = (iree_net_queue_channel_t*)user_data;

  if (!iree_net_channel_send_gate_try_enter(&channel->send_gate)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "queue channel endpoint detached");
  }

  iree_net_message_endpoint_send_params_t params = {
      .data = data,
      .user_data = send_user_data,
  };
  iree_status_t status =
      iree_net_message_endpoint_send(channel->endpoint, &params);

  iree_net_channel_send_gate_leave(&channel->send_gate);
  return status;
}

// Frame sender completion callback. Routes to the channel's on_send_complete.
static void iree_net_queue_channel_on_sender_complete(
    void* callback_user_data, uint64_t operation_user_data,
    iree_status_t status) {
  iree_net_queue_channel_t* channel =
      (iree_net_queue_channel_t*)callback_user_data;
  if (channel->callbacks.on_send_complete) {
    channel->callbacks.on_send_complete(channel->callbacks.user_data,
                                        operation_user_data, status);
  } else {
    iree_status_ignore(status);
  }
}

static void iree_net_queue_channel_on_endpoint_send_ready(void* user_data) {
  iree_net_queue_channel_t* channel = (iree_net_queue_channel_t*)user_data;
  if (iree_net_queue_channel_load_state(channel) ==
          IREE_NET_QUEUE_CHANNEL_STATE_OPERATIONAL &&
      channel->callbacks.on_send_ready) {
    channel->callbacks.on_send_ready(channel->callbacks.user_data);
  }
}

//===----------------------------------------------------------------------===//
// Frontier wire format helpers
//===----------------------------------------------------------------------===//

// Returns the wire size of a frontier: 8-byte header + 16 bytes per entry.
static iree_host_size_t iree_net_queue_channel_frontier_wire_size(
    uint8_t entry_count) {
  if (entry_count == 0) return 0;
  return sizeof(iree_async_frontier_t) +
         (iree_host_size_t)entry_count * sizeof(iree_async_frontier_entry_t);
}

static iree_status_t iree_net_queue_channel_calculate_send_layout(
    uint8_t wait_frontier_entry_count, uint8_t signal_frontier_entry_count,
    iree_host_size_t command_payload_size,
    iree_net_queue_frame_flags_t* out_flags, iree_host_size_t* out_wait_size,
    iree_host_size_t* out_signal_size, uint32_t* out_payload_length,
    iree_host_size_t* out_frame_size) {
  if (wait_frontier_entry_count > IREE_NET_QUEUE_CHANNEL_MAX_FRONTIER_ENTRIES ||
      signal_frontier_entry_count >
          IREE_NET_QUEUE_CHANNEL_MAX_FRONTIER_ENTRIES) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "queue frontier entry count exceeds maximum %u (wait=%u, signal=%u)",
        IREE_NET_QUEUE_CHANNEL_MAX_FRONTIER_ENTRIES, wait_frontier_entry_count,
        signal_frontier_entry_count);
  }
  iree_host_size_t wait_size =
      iree_net_queue_channel_frontier_wire_size(wait_frontier_entry_count);
  iree_host_size_t signal_size =
      iree_net_queue_channel_frontier_wire_size(signal_frontier_entry_count);

  iree_host_size_t total_payload_size = 0;
  if (!iree_host_size_checked_add(wait_size, signal_size,
                                  &total_payload_size) ||
      !iree_host_size_checked_add(total_payload_size, command_payload_size,
                                  &total_payload_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "queue frame payload size overflow");
  }
  if (total_payload_size > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "queue frame payload too large: %" PRIhsz
                            " bytes (max %" PRIu32 ")",
                            total_payload_size, UINT32_MAX);
  }

  iree_host_size_t frame_size = 0;
  if (!iree_host_size_checked_add(IREE_NET_QUEUE_FRAME_HEADER_SIZE,
                                  total_payload_size, &frame_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "queue frame size overflow");
  }

  iree_net_queue_frame_flags_t flags = IREE_NET_QUEUE_FRAME_FLAG_NONE;
  if (wait_size > 0) flags |= IREE_NET_QUEUE_FRAME_FLAG_HAS_WAIT_FRONTIER;
  if (signal_size > 0) flags |= IREE_NET_QUEUE_FRAME_FLAG_HAS_SIGNAL_FRONTIER;

  *out_flags = flags;
  *out_wait_size = wait_size;
  *out_signal_size = signal_size;
  *out_payload_length = (uint32_t)total_payload_size;
  *out_frame_size = frame_size;
  return iree_ok_status();
}

// Parses a frontier from a byte span, advancing the cursor past the parsed
// data. Returns NULL frontier (via out_frontier) if the flag is not set.
//
// On success, |*remaining| is advanced past the frontier data.
// |*out_frontier| points into the original buffer (zero-copy).
static iree_status_t iree_net_queue_channel_parse_frontier(
    iree_const_byte_span_t* remaining,
    const iree_async_frontier_t** out_frontier) {
  *out_frontier = NULL;

  // Need at least the 8-byte frontier header.
  if (remaining->data_length < sizeof(iree_async_frontier_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "frontier header truncated: %" PRIhsz " bytes available (need %zu)",
        remaining->data_length, sizeof(iree_async_frontier_t));
  }

  // Read entry_count from the frontier header.
  const iree_async_frontier_t* frontier =
      (const iree_async_frontier_t*)remaining->data;
  iree_host_size_t total_size =
      sizeof(iree_async_frontier_t) + (iree_host_size_t)frontier->entry_count *
                                          sizeof(iree_async_frontier_entry_t);

  if (remaining->data_length < total_size) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "frontier data truncated: %" PRIhsz " bytes available (need %" PRIhsz
        " for %u entries)",
        remaining->data_length, total_size, frontier->entry_count);
  }

  *out_frontier = frontier;
  *remaining = iree_make_const_byte_span(remaining->data + total_size,
                                         remaining->data_length - total_size);
  return iree_ok_status();
}

// Serializes a frontier into |buffer|. Returns the number of bytes written.
// If |frontier| is NULL or has no entries, returns 0.
static iree_host_size_t iree_net_queue_channel_serialize_frontier(
    uint8_t* buffer, const iree_async_frontier_t* frontier) {
  if (!frontier || frontier->entry_count == 0) return 0;
  iree_host_size_t size =
      sizeof(iree_async_frontier_t) + (iree_host_size_t)frontier->entry_count *
                                          sizeof(iree_async_frontier_entry_t);
  memcpy(buffer, frontier, size);
  return size;
}

//===----------------------------------------------------------------------===//
// Receive path
//===----------------------------------------------------------------------===//

// Handles a received ADVANCE frame.
static iree_status_t iree_net_queue_channel_handle_advance(
    iree_net_queue_channel_t* channel, iree_net_queue_frame_flags_t flags,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease) {
  const iree_async_frontier_t* signal_frontier = NULL;
  iree_const_byte_span_t remaining = payload;

  // Extract signal frontier (should always be present on well-formed ADVANCE).
  if (flags & IREE_NET_QUEUE_FRAME_FLAG_HAS_SIGNAL_FRONTIER) {
    IREE_RETURN_IF_ERROR(
        iree_net_queue_channel_parse_frontier(&remaining, &signal_frontier));
  }

  if (channel->callbacks.on_advance) {
    return channel->callbacks.on_advance(channel->callbacks.user_data,
                                         signal_frontier, remaining, lease);
  }
  return iree_ok_status();
}

// Handles a received COMMAND frame.
static iree_status_t iree_net_queue_channel_handle_command(
    iree_net_queue_channel_t* channel, uint32_t stream_id,
    iree_net_queue_frame_flags_t flags, iree_const_byte_span_t payload,
    iree_async_buffer_lease_t* lease) {
  const iree_async_frontier_t* wait_frontier = NULL;
  const iree_async_frontier_t* signal_frontier = NULL;
  iree_const_byte_span_t remaining = payload;

  // Extract wait frontier if present.
  if (flags & IREE_NET_QUEUE_FRAME_FLAG_HAS_WAIT_FRONTIER) {
    IREE_RETURN_IF_ERROR(
        iree_net_queue_channel_parse_frontier(&remaining, &wait_frontier));
  }

  // Extract signal frontier if present.
  if (flags & IREE_NET_QUEUE_FRAME_FLAG_HAS_SIGNAL_FRONTIER) {
    IREE_RETURN_IF_ERROR(
        iree_net_queue_channel_parse_frontier(&remaining, &signal_frontier));
  }

  // Remaining data is the command payload.
  return channel->callbacks.on_command(channel->callbacks.user_data, stream_id,
                                       wait_frontier, signal_frontier,
                                       remaining, lease);
}

// Endpoint message callback: parses header and dispatches by frame type.
static iree_status_t iree_net_queue_channel_on_message(
    void* user_data, iree_const_byte_span_t message,
    iree_async_buffer_lease_t* lease) {
  iree_net_queue_channel_t* channel = (iree_net_queue_channel_t*)user_data;

  // In ERROR state, reject all messages.
  iree_net_queue_channel_state_t state =
      iree_net_queue_channel_load_state(channel);
  if (state == IREE_NET_QUEUE_CHANNEL_STATE_ERROR) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "queue channel is in error state");
  }

  // Validate minimum size.
  if (message.data_length < IREE_NET_QUEUE_FRAME_HEADER_SIZE) {
    return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
  }

  // Parse header (memcpy to aligned local for safety on all platforms).
  iree_net_queue_frame_header_t header;
  memcpy(&header, message.data, sizeof(header));
  IREE_RETURN_IF_ERROR(iree_net_queue_frame_header_validate(header));

  // Validate payload_length matches the message.
  iree_host_size_t expected_payload =
      message.data_length - IREE_NET_QUEUE_FRAME_HEADER_SIZE;
  if (header.payload_length != expected_payload) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "queue frame payload_length mismatch: header says %u, "
        "message has %" PRIhsz " bytes after header",
        header.payload_length, expected_payload);
  }

  // Extract payload (everything after the 16-byte header).
  iree_const_byte_span_t payload = iree_make_const_byte_span(
      message.data + IREE_NET_QUEUE_FRAME_HEADER_SIZE,
      message.data_length - IREE_NET_QUEUE_FRAME_HEADER_SIZE);

  iree_net_queue_frame_type_t type = iree_net_queue_frame_header_type(header);
  iree_net_queue_frame_flags_t flags =
      iree_net_queue_frame_header_flags(header);
  uint32_t stream_id = iree_net_queue_frame_header_stream_id(header);

  switch (type) {
    case IREE_NET_QUEUE_FRAME_TYPE_COMMAND:
      return iree_net_queue_channel_handle_command(channel, stream_id, flags,
                                                   payload, lease);

    case IREE_NET_QUEUE_FRAME_TYPE_ADVANCE:
      return iree_net_queue_channel_handle_advance(channel, flags, payload,
                                                   lease);

    case IREE_NET_QUEUE_FRAME_TYPE_DATA:
    case IREE_NET_QUEUE_FRAME_TYPE_DATA_END:
      // Fragment reassembly is not yet implemented. These frame types are
      // defined in the wire format for large payload fragmentation but no
      // current producer generates them. Fail loudly so we notice if one
      // appears rather than silently dropping data.
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "queue channel DATA/DATA_END fragment reassembly not implemented "
          "(stream_id=%u, type=0x%02X)",
          stream_id, (unsigned)type);

    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown queue frame type: 0x%02X",
                              (unsigned)type);
  }
}

// Endpoint error callback: transitions to ERROR state.
static void iree_net_queue_channel_on_endpoint_error(void* user_data,
                                                     iree_status_t status) {
  iree_net_queue_channel_t* channel = (iree_net_queue_channel_t*)user_data;

  iree_net_queue_channel_set_state(channel, IREE_NET_QUEUE_CHANNEL_STATE_ERROR);

  if (channel->callbacks.on_transport_error) {
    channel->callbacks.on_transport_error(channel->callbacks.user_data, status);
  } else {
    iree_status_ignore(status);
  }
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

static void iree_net_queue_channel_destroy(iree_net_queue_channel_t* channel);

iree_status_t iree_net_queue_channel_create(
    iree_net_message_endpoint_t endpoint, iree_host_size_t max_send_spans,
    iree_async_buffer_pool_t* header_pool,
    iree_net_queue_channel_callbacks_t callbacks,
    iree_allocator_t host_allocator, iree_net_queue_channel_t** out_channel) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(header_pool);
  IREE_ASSERT_ARGUMENT(out_channel);
  *out_channel = NULL;

  if (!callbacks.on_command) {
    // Free the pool we would have taken ownership of.
    iree_async_buffer_pool_release(header_pool);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "on_command callback is required");
  }

  iree_net_queue_channel_t* channel = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*channel), (void**)&channel);
  if (!iree_status_is_ok(status)) {
    iree_async_buffer_pool_release(header_pool);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  memset(channel, 0, sizeof(*channel));

  iree_atomic_ref_count_init(&channel->ref_count);
  channel->host_allocator = host_allocator;
  channel->endpoint = endpoint;
  iree_net_channel_send_gate_initialize(&channel->send_gate);
  channel->header_pool = header_pool;  // Takes ownership.
  channel->callbacks = callbacks;
  iree_atomic_store(&channel->state,
                    (int32_t)IREE_NET_QUEUE_CHANNEL_STATE_CREATED,
                    iree_memory_order_release);

  // Initialize the embedded frame sender for the send path.
  iree_net_frame_send_complete_callback_t send_complete = {
      .fn = iree_net_queue_channel_on_sender_complete,
      .user_data = channel,
  };
  status = iree_net_frame_sender_initialize(
      &channel->sender, iree_net_queue_channel_submit_send, channel,
      max_send_spans, header_pool, send_complete, host_allocator,
      host_allocator);
  if (!iree_status_is_ok(status)) {
    iree_net_channel_send_gate_deinitialize(&channel->send_gate);
    iree_async_buffer_pool_release(channel->header_pool);
    iree_allocator_free(host_allocator, channel);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_channel = channel;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_net_queue_channel_retain(iree_net_queue_channel_t* channel) {
  if (!channel) return;
  iree_atomic_ref_count_inc(&channel->ref_count);
}

void iree_net_queue_channel_release(iree_net_queue_channel_t* channel) {
  if (!channel) return;
  if (iree_atomic_ref_count_dec(&channel->ref_count) == 1) {
    iree_net_queue_channel_destroy(channel);
  }
}

void iree_net_queue_channel_detach(iree_net_queue_channel_t* channel) {
  if (!channel) return;

  if (iree_net_channel_send_gate_begin_close(&channel->send_gate)) {
    iree_net_channel_send_gate_await_quiescence(&channel->send_gate);
    if (channel->endpoint.self) {
      iree_net_message_endpoint_callbacks_t empty_callbacks;
      memset(&empty_callbacks, 0, sizeof(empty_callbacks));
      iree_net_message_endpoint_set_callbacks(channel->endpoint,
                                              empty_callbacks);
      memset(&channel->endpoint, 0, sizeof(channel->endpoint));
    }
    iree_net_queue_channel_set_state(channel,
                                     IREE_NET_QUEUE_CHANNEL_STATE_ERROR);
    iree_net_channel_send_gate_finish_close(&channel->send_gate);
  } else {
    iree_net_channel_send_gate_await_closed(&channel->send_gate);
  }
}

static void iree_net_queue_channel_destroy(iree_net_queue_channel_t* channel) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Clear endpoint callbacks if the channel was not already detached. The
  // session or connection owning the borrowed endpoint controls deactivation.
  if (channel->endpoint.self) {
    iree_net_message_endpoint_callbacks_t empty_callbacks;
    memset(&empty_callbacks, 0, sizeof(empty_callbacks));
    iree_net_message_endpoint_set_callbacks(channel->endpoint, empty_callbacks);
  }

  // Deinitialize the frame sender. Asserts no sends in flight.
  iree_net_frame_sender_deinitialize(&channel->sender);
  iree_net_channel_send_gate_deinitialize(&channel->send_gate);

  // Free the owned header pool. Must happen after sender deinitialize since
  // the sender borrows the pool pointer.
  iree_async_buffer_pool_release(channel->header_pool);

  iree_allocator_t host_allocator = channel->host_allocator;
  iree_allocator_free(host_allocator, channel);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_net_queue_channel_activate(
    iree_net_queue_channel_t* channel) {
  IREE_ASSERT_ARGUMENT(channel);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_queue_channel_state_t state =
      iree_net_queue_channel_load_state(channel);
  if (state != IREE_NET_QUEUE_CHANNEL_STATE_CREATED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "channel not in CREATED state (state=%d)",
                            (int)state);
  }

  iree_net_message_endpoint_callbacks_t endpoint_callbacks = {
      .on_message = iree_net_queue_channel_on_message,
      .on_error = iree_net_queue_channel_on_endpoint_error,
      .user_data = channel,
      .on_send_ready = iree_net_queue_channel_on_endpoint_send_ready,
  };
  iree_net_message_endpoint_set_callbacks(channel->endpoint,
                                          endpoint_callbacks);

  iree_status_t status = iree_net_message_endpoint_activate(channel->endpoint);
  if (iree_status_is_ok(status)) {
    iree_net_queue_channel_set_state(channel,
                                     IREE_NET_QUEUE_CHANNEL_STATE_OPERATIONAL);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Query
//===----------------------------------------------------------------------===//

iree_net_queue_channel_state_t iree_net_queue_channel_state(
    const iree_net_queue_channel_t* channel) {
  IREE_ASSERT_ARGUMENT(channel);
  return iree_net_queue_channel_load_state(channel);
}

bool iree_net_queue_channel_has_pending_sends(
    const iree_net_queue_channel_t* channel) {
  IREE_ASSERT_ARGUMENT(channel);
  return iree_net_frame_sender_has_pending(&channel->sender) ||
         iree_net_channel_send_gate_pending_count(&channel->send_gate) > 0;
}

//===----------------------------------------------------------------------===//
// Send path
//===----------------------------------------------------------------------===//

iree_status_t iree_net_queue_channel_send_command(
    iree_net_queue_channel_t* channel, uint32_t stream_id,
    const iree_async_frontier_t* wait_frontier,
    const iree_async_frontier_t* signal_frontier,
    iree_async_span_list_t command_payload, uint64_t operation_user_data) {
  IREE_ASSERT_ARGUMENT(channel);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_queue_channel_state_t state =
      iree_net_queue_channel_load_state(channel);
  if (state != IREE_NET_QUEUE_CHANNEL_STATE_OPERATIONAL) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot send COMMAND: channel state is %d",
                            (int)state);
  }

  iree_host_size_t command_payload_size = 0;
  for (iree_host_size_t i = 0; i < command_payload.count; ++i) {
    if (!iree_host_size_checked_add(command_payload_size,
                                    command_payload.values[i].length,
                                    &command_payload_size)) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "queue command payload size overflow");
    }
  }

  iree_net_queue_frame_flags_t flags = IREE_NET_QUEUE_FRAME_FLAG_NONE;
  iree_host_size_t wait_size = 0;
  iree_host_size_t signal_size = 0;
  uint32_t payload_length = 0;
  iree_host_size_t frame_size = 0;
  iree_status_t status = iree_net_queue_channel_calculate_send_layout(
      wait_frontier ? wait_frontier->entry_count : 0,
      signal_frontier ? signal_frontier->entry_count : 0, command_payload_size,
      &flags, &wait_size, &signal_size, &payload_length, &frame_size);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  iree_host_size_t header_total = frame_size - command_payload_size;

  // Build header + frontier data contiguously. The frame_sender copies this
  // into retained sender storage, so stack allocation is safe.
  //
  // The stack buffer handles up to IREE_NET_QUEUE_CHANNEL_MAX_HEADER_SIZE
  // bytes, which accommodates 32 entries per frontier — sufficient for
  // rack-scale systems with hundreds of queues. Larger frontiers will fail
  // with RESOURCE_EXHAUSTED; if needed, increase the constant and the pool
  // buffer size to match.
  uint8_t header_buffer[IREE_NET_QUEUE_CHANNEL_MAX_HEADER_SIZE];
  if (header_total > sizeof(header_buffer)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "queue frame header + frontiers too large: %" PRIhsz
                            " bytes (max %zu)",
                            header_total, sizeof(header_buffer));
  }

  // Queue frame header.
  iree_net_queue_frame_header_t frame_header;
  iree_net_queue_frame_header_initialize(IREE_NET_QUEUE_FRAME_TYPE_COMMAND,
                                         flags, payload_length, stream_id,
                                         &frame_header);
  iree_host_size_t offset = 0;
  memcpy(header_buffer + offset, &frame_header, sizeof(frame_header));
  offset += sizeof(frame_header);

  // Serialize frontiers.
  offset += iree_net_queue_channel_serialize_frontier(header_buffer + offset,
                                                      wait_frontier);
  offset += iree_net_queue_channel_serialize_frontier(header_buffer + offset,
                                                      signal_frontier);

  status = iree_net_frame_sender_send(
      &channel->sender, iree_make_const_byte_span(header_buffer, offset),
      command_payload, operation_user_data);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_net_queue_channel_begin_command(
    iree_net_queue_channel_t* channel, uint32_t stream_id,
    uint8_t wait_frontier_entry_count, uint8_t signal_frontier_entry_count,
    iree_host_size_t command_payload_length,
    iree_net_queue_channel_command_reservation_t* out_reservation) {
  IREE_ASSERT_ARGUMENT(channel);
  IREE_ASSERT_ARGUMENT(out_reservation);
  memset(out_reservation, 0, sizeof(*out_reservation));
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_queue_channel_state_t state =
      iree_net_queue_channel_load_state(channel);
  if (state != IREE_NET_QUEUE_CHANNEL_STATE_OPERATIONAL) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot send COMMAND: channel state is %d",
                            (int)state);
  }

  iree_net_queue_frame_flags_t flags = IREE_NET_QUEUE_FRAME_FLAG_NONE;
  iree_host_size_t wait_size = 0;
  iree_host_size_t signal_size = 0;
  uint32_t payload_length = 0;
  iree_host_size_t frame_size = 0;
  iree_status_t status = iree_net_queue_channel_calculate_send_layout(
      wait_frontier_entry_count, signal_frontier_entry_count,
      command_payload_length, &flags, &wait_size, &signal_size, &payload_length,
      &frame_size);

  bool send_admitted = false;
  if (iree_status_is_ok(status)) {
    send_admitted = iree_net_channel_send_gate_try_enter(&channel->send_gate);
  }
  if (iree_status_is_ok(status) && !send_admitted) {
    status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "queue channel endpoint detached");
  }

  void* frame_data = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_net_message_endpoint_begin_send(
        channel->endpoint, frame_size, &frame_data,
        &out_reservation->send_handle);
  }

  if (iree_status_is_ok(status)) {
    iree_net_queue_frame_header_t frame_header;
    iree_net_queue_frame_header_initialize(IREE_NET_QUEUE_FRAME_TYPE_COMMAND,
                                           flags, payload_length, stream_id,
                                           &frame_header);
    iree_host_size_t offset = 0;
    uint8_t* frame_bytes = (uint8_t*)frame_data;
    memcpy(frame_bytes + offset, &frame_header, sizeof(frame_header));
    offset += sizeof(frame_header);
    if (wait_frontier_entry_count > 0) {
      iree_async_frontier_t* wait_frontier =
          (iree_async_frontier_t*)(frame_bytes + offset);
      iree_async_frontier_initialize(wait_frontier, wait_frontier_entry_count);
      out_reservation->wait_entries = wait_frontier->entries;
      offset += wait_size;
    }
    if (signal_frontier_entry_count > 0) {
      iree_async_frontier_t* signal_frontier =
          (iree_async_frontier_t*)(frame_bytes + offset);
      iree_async_frontier_initialize(signal_frontier,
                                     signal_frontier_entry_count);
      out_reservation->signal_entries = signal_frontier->entries;
      offset += signal_size;
    }
    out_reservation->command_payload =
        iree_make_byte_span(frame_bytes + offset, command_payload_length);
  } else if (send_admitted) {
    iree_net_channel_send_gate_leave(&channel->send_gate);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_net_queue_channel_commit_command(
    iree_net_queue_channel_t* channel,
    iree_net_queue_channel_command_reservation_t* reservation) {
  IREE_ASSERT_ARGUMENT(channel);
  IREE_ASSERT_ARGUMENT(reservation);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_net_message_endpoint_commit_send(
      channel->endpoint, reservation->send_handle);
  iree_net_channel_send_gate_leave(&channel->send_gate);
  memset(reservation, 0, sizeof(*reservation));
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_net_queue_channel_abort_command(
    iree_net_queue_channel_t* channel,
    iree_net_queue_channel_command_reservation_t* reservation) {
  if (!channel) return;
  IREE_ASSERT_ARGUMENT(reservation);
  iree_net_message_endpoint_abort_send(channel->endpoint,
                                       reservation->send_handle);
  iree_net_channel_send_gate_leave(&channel->send_gate);
  memset(reservation, 0, sizeof(*reservation));
}

iree_status_t iree_net_queue_channel_send_advance(
    iree_net_queue_channel_t* channel,
    const iree_async_frontier_t* signal_frontier,
    iree_async_span_list_t advance_payload, uint64_t operation_user_data) {
  IREE_ASSERT_ARGUMENT(channel);
  IREE_ASSERT_ARGUMENT(signal_frontier);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_queue_channel_state_t state =
      iree_net_queue_channel_load_state(channel);
  if (state != IREE_NET_QUEUE_CHANNEL_STATE_OPERATIONAL) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot send ADVANCE: channel state is %d",
                            (int)state);
  }

  // ADVANCE always carries a signal frontier.
  iree_host_size_t signal_size =
      iree_net_queue_channel_frontier_wire_size(signal_frontier->entry_count);
  if (signal_size == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "ADVANCE frame requires a non-empty signal frontier");
  }

  iree_host_size_t header_total =
      IREE_NET_QUEUE_FRAME_HEADER_SIZE + signal_size;
  iree_net_queue_frame_flags_t flags =
      IREE_NET_QUEUE_FRAME_FLAG_HAS_SIGNAL_FRONTIER;

  // Compute total payload_length (frontier + advance data).
  iree_host_size_t advance_payload_size = 0;
  for (iree_host_size_t i = 0; i < advance_payload.count; ++i) {
    advance_payload_size += advance_payload.values[i].length;
  }
  iree_host_size_t total_payload_size = signal_size + advance_payload_size;
  if (total_payload_size > UINT32_MAX) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "ADVANCE frame payload too large: %" PRIhsz
                            " bytes (max %" PRIu32 ")",
                            total_payload_size, UINT32_MAX);
  }
  uint32_t payload_length = (uint32_t)total_payload_size;

  // Build header + signal frontier contiguously.
  uint8_t header_buffer[IREE_NET_QUEUE_CHANNEL_MAX_HEADER_SIZE];
  if (header_total > sizeof(header_buffer)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "ADVANCE frame header + frontier too large: %" PRIhsz
        " bytes (max %zu)",
        header_total, sizeof(header_buffer));
  }

  iree_net_queue_frame_header_t frame_header;
  iree_net_queue_frame_header_initialize(IREE_NET_QUEUE_FRAME_TYPE_ADVANCE,
                                         flags, payload_length,
                                         /*stream_id=*/0, &frame_header);
  iree_host_size_t offset = 0;
  memcpy(header_buffer + offset, &frame_header, sizeof(frame_header));
  offset += sizeof(frame_header);

  offset += iree_net_queue_channel_serialize_frontier(header_buffer + offset,
                                                      signal_frontier);

  iree_status_t status = iree_net_frame_sender_send_copy(
      &channel->sender, iree_make_const_byte_span(header_buffer, offset),
      advance_payload, operation_user_data);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
