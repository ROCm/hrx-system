// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/tcp/connection.h"

#include <string.h>

#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/base/threading/mutex.h"
#include "iree/net/endpoint_lifecycle.h"
#include "iree/net/framing_adapter.h"

// "IRN1" in little-endian byte order.
#define IREE_NET_TCP_FRAME_MAGIC UINT32_C(0x314E5249)

#define IREE_NET_TCP_FRAME_VERSION 1u
#define IREE_NET_TCP_FRAME_HEADER_SIZE 16u
#define IREE_NET_TCP_INDEX_NONE UINT32_MAX

typedef enum iree_net_tcp_frame_flag_bits_e {
  IREE_NET_TCP_FRAME_FLAG_NONE = 0u,
  // Announces that the endpoint can receive DATA frames.
  IREE_NET_TCP_FRAME_FLAG_ENDPOINT_ACTIVE = 1u << 0,
} iree_net_tcp_frame_flag_bits_t;
typedef uint16_t iree_net_tcp_frame_flags_t;

typedef struct iree_net_tcp_connection_t iree_net_tcp_connection_t;
typedef struct iree_net_tcp_endpoint_t iree_net_tcp_endpoint_t;

typedef enum iree_net_tcp_connection_state_e {
  // Endpoint opens may be accepted.
  IREE_NET_TCP_CONNECTION_STATE_OPEN = 0,
  // Deactivation is waiting for accepted endpoint-ready callbacks.
  IREE_NET_TCP_CONNECTION_STATE_DRAINING_READY_CALLBACKS = 1,
  // Endpoint and shared carrier drains have started.
  IREE_NET_TCP_CONNECTION_STATE_DRAINING_ENDPOINTS = 2,
  // Every endpoint and shared carrier operation has drained.
  IREE_NET_TCP_CONNECTION_STATE_DEACTIVATED = 3,
} iree_net_tcp_connection_state_t;

typedef enum iree_net_tcp_endpoint_phase_e {
  // The endpoint has not been activated locally.
  IREE_NET_TCP_ENDPOINT_PHASE_CREATED = 0,
  // A proactor operation is draining pre-activation frames.
  IREE_NET_TCP_ENDPOINT_PHASE_ACTIVATING = 1,
  // Incoming frames may be delivered directly to the consumer.
  IREE_NET_TCP_ENDPOINT_PHASE_ACTIVE = 2,
  // Endpoint deactivation has begun.
  IREE_NET_TCP_ENDPOINT_PHASE_DRAINING = 3,
  // Endpoint deactivation has completed.
  IREE_NET_TCP_ENDPOINT_PHASE_DEACTIVATED = 4,
} iree_net_tcp_endpoint_phase_t;

typedef enum iree_net_tcp_send_state_phase_e {
  // The state record is available for admission.
  IREE_NET_TCP_SEND_STATE_PHASE_FREE = 0,
  // The shared carrier owns the submitted logical send.
  IREE_NET_TCP_SEND_STATE_PHASE_RAW_IN_FLIGHT = 1,
  // Raw transmission succeeded and peer readiness owns completion.
  IREE_NET_TCP_SEND_STATE_PHASE_WAITING_FOR_PEER_ACTIVE = 2,
  // An owner-proactor NOP owns terminal completion.
  IREE_NET_TCP_SEND_STATE_PHASE_LOCAL_COMPLETION = 3,
  // Terminal-error fanout owns completion on the current proactor callback.
  IREE_NET_TCP_SEND_STATE_PHASE_TERMINAL_COMPLETION = 4,
} iree_net_tcp_send_state_phase_t;

typedef struct iree_net_tcp_pending_frame_t {
  // Storage lease moved from the framing adapter.
  iree_async_buffer_lease_t lease;

  // Payload view kept valid by |lease|.
  iree_const_byte_span_t payload;
} iree_net_tcp_pending_frame_t;

typedef enum iree_net_tcp_activation_announcement_phase_e {
  // The endpoint has not queued its one-time announcement.
  IREE_NET_TCP_ACTIVATION_ANNOUNCEMENT_PHASE_NONE = 0,
  // The endpoint is waiting in the connection control lane.
  IREE_NET_TCP_ACTIVATION_ANNOUNCEMENT_PHASE_QUEUED = 1,
  // The raw carrier owns the endpoint's ACTIVE frame.
  IREE_NET_TCP_ACTIVATION_ANNOUNCEMENT_PHASE_IN_FLIGHT = 2,
  // The announcement completed or was cancelled during drain.
  IREE_NET_TCP_ACTIVATION_ANNOUNCEMENT_PHASE_RETIRED = 3,
} iree_net_tcp_activation_announcement_phase_t;

typedef struct iree_net_tcp_send_state_t {
  // Endpoint operation retained through terminal completion.
  iree_net_tcp_endpoint_t* endpoint;

  // Index of this record in the connection send-state array.
  uint32_t index;

  // Next available record while this state is free.
  uint32_t next_free;

  // Current ownership phase.
  iree_net_tcp_send_state_phase_t phase;

  // Payload bytes represented by the framed send.
  iree_host_size_t payload_length;

  // User completion invoked after connection state is released.
  iree_net_send_completion_callback_t completion_callback;

  // Dispatches a locally terminated send on the connection proactor.
  iree_async_nop_operation_t local_completion_operation;

  // Owned status transferred through |local_completion_operation|.
  iree_status_t local_completion_status;
} iree_net_tcp_send_state_t;

typedef struct iree_net_tcp_frame_prefix_t {
  // Wire flags distinguishing DATA from control frames.
  iree_net_tcp_frame_flags_t flags;

  // Payload byte count encoded into the wire header.
  uint32_t payload_length;

  // Endpoint ordinal encoded into the wire header.
  uint16_t endpoint_ordinal;

  // Message prefix generated after the wire header.
  iree_net_send_prefix_t message_prefix;
} iree_net_tcp_frame_prefix_t;

struct iree_net_tcp_endpoint_t {
  // Connection owning this ordinal endpoint.
  iree_net_tcp_connection_t* connection;

  // Ordinal encoded in the TCP frame header.
  uint16_t ordinal;

  // Current routing and activation phase.
  iree_net_tcp_endpoint_phase_t phase;

  // Message and terminal-error callbacks installed by the consumer.
  iree_net_message_endpoint_callbacks_t callbacks;

  // True after peer ACTIVE has made ordinary send admission safe.
  bool peer_active;

  // True after any DATA arrived before local activation completed.
  bool preactivation_data_observed;

  // True while |pending_frame| owns a receive lease.
  bool pending_frame_present;

  // True while terminal-error fanout owns a lifecycle operation hold.
  bool terminal_error_callback_pending;

  // Send whose completion owns the one pre-activation DATA credit.
  iree_net_tcp_send_state_t* preactivation_send_state;

  // Coordinates endpoint operations with endpoint/connection drain.
  iree_net_endpoint_lifecycle_t lifecycle;

  // Preallocated operation delivering the endpoint-ready callback.
  iree_async_nop_operation_t ready_operation;

  // Callback owned while |ready_operation| is submitted.
  iree_net_endpoint_ready_callback_t ready_callback;

  // Preallocated operation draining frames queued before activation.
  iree_async_nop_operation_t activation_operation;

  // Current ownership phase of the one-time ACTIVE announcement.
  iree_net_tcp_activation_announcement_phase_t activation_announcement_phase;

  // Next endpoint ordinal in the connection announcement queue.
  uint32_t next_activation_ordinal;

  // Endpoint-consumer callback awaiting explicit endpoint deactivation.
  struct {
    // Function invoked after the endpoint lifecycle drains.
    iree_net_message_endpoint_deactivate_fn_t fn;
    // Opaque value passed to |fn|.
    void* user_data;
  } deactivate_callback;

  // One DATA frame retained before local activation.
  iree_net_tcp_pending_frame_t pending_frame;
};

struct iree_net_tcp_connection_t {
  // Public connection base; must be first.
  iree_net_connection_t base;

  // Serializes connection, endpoint, send-state, and control ownership.
  iree_slim_mutex_t mutex;

  // Proactor dispatching all connection callbacks. Retained.
  iree_async_proactor_t* proactor;

  // Owned framing adapter and raw TCP carrier stack.
  iree_net_framing_adapter_t* framing_adapter;

  // Borrowed wire-frame endpoint exposed by |framing_adapter|.
  iree_net_message_endpoint_t wire_endpoint;

  // Current connection lifecycle phase.
  iree_net_tcp_connection_state_t state;

  // True after ownership has transferred to a public caller.
  bool published;

  // Number of monotonically claimed endpoint ordinals.
  uint32_t opened_endpoint_count;

  // Accepted endpoint-ready operations not yet retired.
  uint32_t pending_ready_count;

  // Maximum accepted total frame extent including its header.
  uint32_t max_frame_size;

  // First terminal shared connection status, owned until destruction.
  iree_status_t terminal_status;

  // Callback awaiting complete connection deactivation.
  iree_net_connection_deactivate_callback_t deactivate_callback;

  // Barrier joining endpoint and shared adapter drains.
  iree_net_endpoint_deactivation_barrier_t deactivation_barrier;

  // Preallocated endpoint slots indexed by wire ordinal.
  iree_net_tcp_endpoint_t* endpoints;

  // Number of connection-wide framing records.
  uint32_t send_state_count;

  // Number of framing records currently available.
  uint32_t free_send_state_count;

  // Head index of the send-state free list.
  uint32_t free_send_state_head;

  // Preallocated framing completion records.
  iree_net_tcp_send_state_t* send_states;

  // First endpoint waiting to publish its ACTIVE announcement.
  uint32_t activation_queue_head;

  // Last endpoint waiting to publish its ACTIVE announcement.
  uint32_t activation_queue_tail;

  // Endpoint whose ACTIVE frame is owned by the raw carrier.
  uint32_t activation_in_flight;
};

//===----------------------------------------------------------------------===//
// Wire format
//===----------------------------------------------------------------------===//

static void iree_net_tcp_encode_frame_header(uint8_t* header,
                                             iree_net_tcp_frame_flags_t flags,
                                             uint32_t payload_length,
                                             uint16_t endpoint_ordinal) {
  iree_unaligned_store_le_u32(header + 0, IREE_NET_TCP_FRAME_MAGIC);
  iree_unaligned_store_le_u16(header + 4, IREE_NET_TCP_FRAME_VERSION);
  iree_unaligned_store_le_u16(header + 6, flags);
  iree_unaligned_store_le_u32(header + 8, payload_length);
  iree_unaligned_store_le_u16(header + 12, endpoint_ordinal);
  iree_unaligned_store_le_u16(header + 14, 0);
}

static iree_status_t iree_net_tcp_write_frame_prefix(void* user_data,
                                                     iree_byte_span_t target) {
  iree_net_tcp_frame_prefix_t* prefix = (iree_net_tcp_frame_prefix_t*)user_data;
  iree_net_tcp_encode_frame_header(target.data, prefix->flags,
                                   prefix->payload_length,
                                   prefix->endpoint_ordinal);
  if (prefix->message_prefix.length == 0) {
    return iree_ok_status();
  }
  return prefix->message_prefix.write(
      prefix->message_prefix.user_data,
      iree_make_byte_span(target.data + IREE_NET_TCP_FRAME_HEADER_SIZE,
                          prefix->message_prefix.length));
}

static iree_status_t iree_net_tcp_calculate_frame_size(
    iree_net_tcp_connection_t* connection, iree_host_size_t payload_length,
    uint32_t* out_frame_size) {
  if (payload_length > UINT32_MAX - IREE_NET_TCP_FRAME_HEADER_SIZE) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "TCP message length %" PRIhsz
                            " exceeds the 32-bit wire extent",
                            payload_length);
  }
  const uint32_t frame_size =
      (uint32_t)payload_length + IREE_NET_TCP_FRAME_HEADER_SIZE;
  if (frame_size > connection->max_frame_size) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "TCP frame size %u exceeds configured limit %u",
                            frame_size, connection->max_frame_size);
  }
  *out_frame_size = frame_size;
  return iree_ok_status();
}

static iree_status_t iree_net_tcp_resolve_frame_size(
    void* user_data, iree_const_byte_span_t available,
    iree_host_size_t* out_frame_size) {
  iree_net_tcp_connection_t* connection = (iree_net_tcp_connection_t*)user_data;
  *out_frame_size = 0;
  if (available.data_length < IREE_NET_TCP_FRAME_HEADER_SIZE) {
    return iree_ok_status();
  }

  const uint32_t magic = iree_unaligned_load_le_u32(available.data + 0);
  if (magic != IREE_NET_TCP_FRAME_MAGIC) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "invalid TCP frame magic 0x%08X", magic);
  }
  const uint16_t version = iree_unaligned_load_le_u16(available.data + 4);
  if (version != IREE_NET_TCP_FRAME_VERSION) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "unsupported TCP frame version %u", version);
  }
  const iree_net_tcp_frame_flags_t flags =
      iree_unaligned_load_le_u16(available.data + 6);
  const uint16_t endpoint_ordinal =
      iree_unaligned_load_le_u16(available.data + 12);
  if (endpoint_ordinal >= connection->base.max_endpoint_count) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS, "TCP frame endpoint ordinal %u exceeds limit %u",
        endpoint_ordinal, connection->base.max_endpoint_count);
  }
  const uint16_t reserved = iree_unaligned_load_le_u16(available.data + 14);
  if (reserved != 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "TCP frame reserved field is nonzero");
  }

  const uint32_t payload_length =
      iree_unaligned_load_le_u32(available.data + 8);
  switch (flags) {
    case IREE_NET_TCP_FRAME_FLAG_NONE:
      if (payload_length == 0) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "TCP DATA frame payload must be nonempty");
      }
      break;
    case IREE_NET_TCP_FRAME_FLAG_ENDPOINT_ACTIVE:
      if (payload_length != 0) {
        return iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "TCP endpoint ACTIVE frame must not carry a payload");
      }
      break;
    default:
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "unsupported TCP frame flags 0x%04X", flags);
  }
  iree_host_size_t frame_size = 0;
  if (!iree_host_size_checked_add(IREE_NET_TCP_FRAME_HEADER_SIZE,
                                  payload_length, &frame_size)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "TCP frame extent overflows host size");
  }
  *out_frame_size = frame_size;
  return iree_ok_status();
}

static iree_host_size_t iree_net_tcp_message_length(
    const iree_net_message_endpoint_send_params_t* params) {
  iree_host_size_t message_length = params->generated_prefix.length;
  for (iree_host_size_t i = 0; i < params->data.count; ++i) {
    message_length += params->data.values[i].length;
  }
  return message_length;
}

//===----------------------------------------------------------------------===//
// Bounded state pools
//===----------------------------------------------------------------------===//

static iree_net_tcp_send_state_t* iree_net_tcp_acquire_send_state_locked(
    iree_net_tcp_connection_t* connection, iree_net_tcp_endpoint_t* endpoint) {
  if (connection->free_send_state_head == IREE_NET_TCP_INDEX_NONE) {
    return NULL;
  }
  iree_net_tcp_send_state_t* send_state =
      &connection->send_states[connection->free_send_state_head];
  connection->free_send_state_head = send_state->next_free;
  --connection->free_send_state_count;
  send_state->next_free = IREE_NET_TCP_INDEX_NONE;
  send_state->phase = IREE_NET_TCP_SEND_STATE_PHASE_RAW_IN_FLIGHT;
  send_state->endpoint = endpoint;
  return send_state;
}

static void iree_net_tcp_release_send_state_locked(
    iree_net_tcp_connection_t* connection,
    iree_net_tcp_send_state_t* send_state) {
  IREE_ASSERT(iree_status_is_ok(send_state->local_completion_status));
  send_state->endpoint = NULL;
  send_state->phase = IREE_NET_TCP_SEND_STATE_PHASE_FREE;
  send_state->payload_length = 0;
  send_state->completion_callback = (iree_net_send_completion_callback_t){0};
  send_state->next_free = connection->free_send_state_head;
  connection->free_send_state_head = send_state->index;
  ++connection->free_send_state_count;
}

typedef struct iree_net_tcp_send_completion_t {
  // Endpoint whose lifecycle hold retires after the callback.
  iree_net_tcp_endpoint_t* endpoint;

  // Application callback receiving terminal send status.
  iree_net_send_completion_callback_t callback;

  // Logical application payload length.
  iree_host_size_t payload_length;
} iree_net_tcp_send_completion_t;

static iree_net_tcp_send_completion_t iree_net_tcp_claim_send_completion_locked(
    iree_net_tcp_connection_t* connection,
    iree_net_tcp_send_state_t* send_state) {
  IREE_ASSERT(send_state->phase != IREE_NET_TCP_SEND_STATE_PHASE_FREE);
  iree_net_tcp_endpoint_t* endpoint = send_state->endpoint;
  if (endpoint->preactivation_send_state == send_state) {
    endpoint->preactivation_send_state = NULL;
  }
  iree_net_tcp_send_completion_t completion = {
      .endpoint = endpoint,
      .callback = send_state->completion_callback,
      .payload_length = send_state->payload_length,
  };
  iree_net_tcp_release_send_state_locked(connection, send_state);
  return completion;
}

static void iree_net_tcp_invoke_send_completion(
    iree_net_tcp_send_completion_t completion, iree_status_t status,
    iree_host_size_t bytes_transferred) {
  completion.callback.fn(completion.callback.user_data, status,
                         bytes_transferred);
  iree_net_endpoint_lifecycle_end_operation(&completion.endpoint->lifecycle);
}

static iree_status_t iree_net_tcp_store_pending_frame_locked(
    iree_net_tcp_endpoint_t* endpoint, iree_const_byte_span_t payload,
    iree_async_buffer_lease_t* lease) {
  if (endpoint->preactivation_data_observed) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "TCP endpoint %u received multiple DATA frames before activation",
        endpoint->ordinal);
  }
  IREE_ASSERT(!endpoint->pending_frame_present);
  endpoint->preactivation_data_observed = true;
  endpoint->pending_frame_present = true;
  endpoint->pending_frame.lease = *lease;
  endpoint->pending_frame.payload = payload;
  *lease = (iree_async_buffer_lease_t){0};
  return iree_ok_status();
}

static bool iree_net_tcp_take_pending_frame_locked(
    iree_net_tcp_endpoint_t* endpoint,
    iree_net_tcp_pending_frame_t* out_pending_frame) {
  *out_pending_frame = (iree_net_tcp_pending_frame_t){0};
  if (!endpoint->pending_frame_present) {
    return false;
  }
  endpoint->pending_frame_present = false;
  *out_pending_frame = endpoint->pending_frame;
  endpoint->pending_frame = (iree_net_tcp_pending_frame_t){0};
  return true;
}

static void iree_net_tcp_release_pending_frame(
    iree_net_tcp_pending_frame_t* pending_frame) {
  iree_async_buffer_lease_release(&pending_frame->lease);
  *pending_frame = (iree_net_tcp_pending_frame_t){0};
}

static void iree_net_tcp_clear_endpoint_pending_frame(
    iree_net_tcp_endpoint_t* endpoint) {
  iree_net_tcp_pending_frame_t pending_frame;
  iree_slim_mutex_lock(&endpoint->connection->mutex);
  const bool has_pending_frame =
      iree_net_tcp_take_pending_frame_locked(endpoint, &pending_frame);
  iree_slim_mutex_unlock(&endpoint->connection->mutex);
  if (has_pending_frame) {
    iree_net_tcp_release_pending_frame(&pending_frame);
  }
}

static void iree_net_tcp_clear_all_pending_frames(
    iree_net_tcp_connection_t* connection) {
  for (uint32_t i = 0; i < connection->base.max_endpoint_count; ++i) {
    iree_net_tcp_clear_endpoint_pending_frame(&connection->endpoints[i]);
  }
}

//===----------------------------------------------------------------------===//
// Shared receive and terminal error dispatch
//===----------------------------------------------------------------------===//

static void iree_net_tcp_connection_record_terminal_error(
    iree_net_tcp_connection_t* connection, iree_status_t status) {
  IREE_ASSERT(!iree_status_is_ok(status),
              "terminal connection error must be non-OK");

  iree_net_connection_retain(&connection->base);
  bool is_first_error = false;
  iree_slim_mutex_lock(&connection->mutex);
  if (iree_status_is_ok(connection->terminal_status)) {
    connection->terminal_status = status;
    is_first_error = true;
    // Admit the complete fanout before any callback can begin connection
    // deactivation and change sibling endpoint phases.
    for (uint32_t i = 0; i < connection->base.max_endpoint_count; ++i) {
      iree_net_tcp_endpoint_t* endpoint = &connection->endpoints[i];
      if ((endpoint->phase == IREE_NET_TCP_ENDPOINT_PHASE_ACTIVATING ||
           endpoint->phase == IREE_NET_TCP_ENDPOINT_PHASE_ACTIVE) &&
          endpoint->callbacks.on_error &&
          iree_net_endpoint_lifecycle_try_begin_operation(
              &endpoint->lifecycle)) {
        IREE_ASSERT(!endpoint->terminal_error_callback_pending,
                    "TCP endpoint %u already has terminal error pending",
                    endpoint->ordinal);
        endpoint->terminal_error_callback_pending = true;
      }
    }
    // Claim every raw-complete rendezvous before callbacks can begin drain.
    for (uint32_t i = 0; i < connection->send_state_count; ++i) {
      iree_net_tcp_send_state_t* send_state = &connection->send_states[i];
      if (send_state->phase ==
          IREE_NET_TCP_SEND_STATE_PHASE_WAITING_FOR_PEER_ACTIVE) {
        IREE_ASSERT(send_state->endpoint->preactivation_send_state ==
                    send_state);
        send_state->endpoint->preactivation_send_state = NULL;
        send_state->phase = IREE_NET_TCP_SEND_STATE_PHASE_TERMINAL_COMPLETION;
      }
    }
  }
  iree_slim_mutex_unlock(&connection->mutex);
  if (!is_first_error) {
    iree_status_free(status);
    iree_net_connection_release(&connection->base);
    return;
  }

  iree_net_tcp_clear_all_pending_frames(connection);
  for (uint32_t i = 0; i < connection->base.max_endpoint_count; ++i) {
    iree_net_tcp_endpoint_t* endpoint = &connection->endpoints[i];
    iree_net_message_endpoint_callbacks_t callbacks = {0};
    bool callback_pending = false;
    iree_slim_mutex_lock(&connection->mutex);
    // Admission, not the current phase, owns this callback. An earlier
    // callback may already have moved the endpoint to DRAINING.
    if (endpoint->terminal_error_callback_pending) {
      endpoint->terminal_error_callback_pending = false;
      callbacks = endpoint->callbacks;
      callback_pending = true;
    }
    iree_slim_mutex_unlock(&connection->mutex);
    if (callback_pending) {
      IREE_ASSERT(callbacks.on_error,
                  "admitted TCP terminal callback has no error handler");
      callbacks.on_error(callbacks.user_data,
                         iree_status_clone(connection->terminal_status));
      iree_net_endpoint_lifecycle_end_operation(&endpoint->lifecycle);
    }
  }
  for (uint32_t i = 0; i < connection->send_state_count; ++i) {
    iree_net_tcp_send_completion_t completion = {0};
    iree_slim_mutex_lock(&connection->mutex);
    iree_net_tcp_send_state_t* send_state = &connection->send_states[i];
    if (send_state->phase ==
        IREE_NET_TCP_SEND_STATE_PHASE_TERMINAL_COMPLETION) {
      completion =
          iree_net_tcp_claim_send_completion_locked(connection, send_state);
    }
    iree_slim_mutex_unlock(&connection->mutex);
    if (completion.endpoint) {
      iree_net_tcp_invoke_send_completion(
          completion, iree_status_clone(connection->terminal_status),
          /*bytes_transferred=*/0);
    }
  }
  iree_net_connection_release(&connection->base);
}

static iree_status_t iree_net_tcp_on_wire_frame(
    void* user_data, iree_const_byte_span_t frame,
    iree_async_buffer_lease_t* lease) {
  iree_net_tcp_connection_t* connection = (iree_net_tcp_connection_t*)user_data;
  const uint16_t endpoint_ordinal = iree_unaligned_load_le_u16(frame.data + 12);
  iree_net_tcp_endpoint_t* endpoint = &connection->endpoints[endpoint_ordinal];
  const iree_net_tcp_frame_flags_t frame_flags =
      iree_unaligned_load_le_u16(frame.data + 6);
  const iree_const_byte_span_t payload = iree_make_const_byte_span(
      frame.data + IREE_NET_TCP_FRAME_HEADER_SIZE,
      frame.data_length - IREE_NET_TCP_FRAME_HEADER_SIZE);

  iree_net_message_endpoint_callbacks_t callbacks = {0};
  iree_net_tcp_send_completion_t readiness_completion = {0};
  iree_slim_mutex_lock(&connection->mutex);
  // Leaving OPEN closes receive admission before endpoint queues are cleared.
  // Frames that lose this lock race remain owned by the framing adapter and
  // are intentionally discarded during connection drain.
  if (connection->state != IREE_NET_TCP_CONNECTION_STATE_OPEN) {
    iree_slim_mutex_unlock(&connection->mutex);
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  if (!iree_status_is_ok(connection->terminal_status)) {
    status = iree_status_clone(connection->terminal_status);
  } else if (frame_flags == IREE_NET_TCP_FRAME_FLAG_ENDPOINT_ACTIVE) {
    if (endpoint->peer_active) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "TCP endpoint %u received duplicate ACTIVE announcement",
          endpoint_ordinal);
    } else {
      endpoint->peer_active = true;
      iree_net_tcp_send_state_t* send_state =
          endpoint->preactivation_send_state;
      if (send_state) {
        if (send_state->phase ==
            IREE_NET_TCP_SEND_STATE_PHASE_WAITING_FOR_PEER_ACTIVE) {
          readiness_completion =
              iree_net_tcp_claim_send_completion_locked(connection, send_state);
        } else {
          IREE_ASSERT(send_state->phase ==
                              IREE_NET_TCP_SEND_STATE_PHASE_RAW_IN_FLIGHT ||
                          send_state->phase ==
                              IREE_NET_TCP_SEND_STATE_PHASE_LOCAL_COMPLETION,
                      "TCP pre-activation send has invalid phase %d",
                      (int)send_state->phase);
          endpoint->preactivation_send_state = NULL;
        }
      }
    }
  } else if (endpoint->phase == IREE_NET_TCP_ENDPOINT_PHASE_CREATED ||
             endpoint->phase == IREE_NET_TCP_ENDPOINT_PHASE_ACTIVATING) {
    status = iree_net_tcp_store_pending_frame_locked(endpoint, payload, lease);
  } else if (endpoint->phase != IREE_NET_TCP_ENDPOINT_PHASE_ACTIVE ||
             !iree_net_endpoint_lifecycle_try_begin_operation(
                 &endpoint->lifecycle)) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "TCP endpoint %u is not active", endpoint_ordinal);
  } else {
    callbacks = endpoint->callbacks;
  }
  iree_slim_mutex_unlock(&connection->mutex);

  if (readiness_completion.endpoint) {
    iree_net_tcp_invoke_send_completion(readiness_completion, iree_ok_status(),
                                        readiness_completion.payload_length);
  } else if (callbacks.on_message) {
    status = callbacks.on_message(callbacks.user_data, payload, lease);
    iree_net_endpoint_lifecycle_end_operation(&endpoint->lifecycle);
  }
  return status;
}

static void iree_net_tcp_on_wire_error(void* user_data, iree_status_t status) {
  iree_net_tcp_connection_record_terminal_error(
      (iree_net_tcp_connection_t*)user_data, status);
}

//===----------------------------------------------------------------------===//
// Endpoint implementation
//===----------------------------------------------------------------------===//

static void iree_net_tcp_endpoint_send_complete(
    void* user_data, iree_status_t status,
    iree_host_size_t wire_bytes_transferred) {
  iree_net_tcp_send_state_t* send_state = (iree_net_tcp_send_state_t*)user_data;
  iree_net_tcp_endpoint_t* endpoint = send_state->endpoint;
  iree_net_tcp_connection_t* connection = endpoint->connection;

  const iree_host_size_t payload_length = send_state->payload_length;
  IREE_ASSERT(!iree_status_is_ok(status) ||
                  wire_bytes_transferred ==
                      payload_length + IREE_NET_TCP_FRAME_HEADER_SIZE,
              "successful TCP framed send completed %" PRIhsz " of %" PRIhsz
              " wire bytes",
              wire_bytes_transferred,
              payload_length + IREE_NET_TCP_FRAME_HEADER_SIZE);

  bool retained_for_peer_active = false;
  bool complete_with_terminal_error = false;
  bool complete_with_cancellation = false;
  iree_net_tcp_send_completion_t completion = {0};
  iree_slim_mutex_lock(&connection->mutex);
  IREE_ASSERT(send_state->phase == IREE_NET_TCP_SEND_STATE_PHASE_RAW_IN_FLIGHT,
              "TCP raw send completed from phase %d", (int)send_state->phase);
  if (iree_status_is_ok(status) && !endpoint->peer_active &&
      connection->state == IREE_NET_TCP_CONNECTION_STATE_OPEN &&
      iree_status_is_ok(connection->terminal_status) &&
      (endpoint->phase == IREE_NET_TCP_ENDPOINT_PHASE_ACTIVATING ||
       endpoint->phase == IREE_NET_TCP_ENDPOINT_PHASE_ACTIVE)) {
    IREE_ASSERT(endpoint->preactivation_send_state == send_state);
    send_state->phase = IREE_NET_TCP_SEND_STATE_PHASE_WAITING_FOR_PEER_ACTIVE;
    retained_for_peer_active = true;
  } else {
    if (iree_status_is_ok(status) && !endpoint->peer_active) {
      if (!iree_status_is_ok(connection->terminal_status)) {
        complete_with_terminal_error = true;
      } else {
        complete_with_cancellation = true;
      }
    }
    completion =
        iree_net_tcp_claim_send_completion_locked(connection, send_state);
  }
  iree_slim_mutex_unlock(&connection->mutex);

  if (retained_for_peer_active) {
    iree_status_free(status);
    return;
  }

  iree_host_size_t payload_bytes_transferred = 0;
  if (!complete_with_terminal_error && !complete_with_cancellation &&
      wire_bytes_transferred > IREE_NET_TCP_FRAME_HEADER_SIZE) {
    payload_bytes_transferred =
        iree_min(payload_length,
                 wire_bytes_transferred - IREE_NET_TCP_FRAME_HEADER_SIZE);
  }
  if (complete_with_terminal_error) {
    iree_status_free(status);
    status = iree_status_clone(connection->terminal_status);
  } else if (complete_with_cancellation) {
    iree_status_free(status);
    status =
        iree_make_status(IREE_STATUS_CANCELLED,
                         "TCP send cancelled before peer endpoint activation");
  }
  iree_net_tcp_invoke_send_completion(completion, status,
                                      payload_bytes_transferred);
}

static void iree_net_tcp_endpoint_local_send_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  IREE_ASSERT(!iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE));
  iree_net_tcp_send_state_t* send_state = (iree_net_tcp_send_state_t*)user_data;
  iree_status_t completion_status = send_state->local_completion_status;
  send_state->local_completion_status = iree_ok_status();
  completion_status = iree_status_join(completion_status, status);
  iree_net_tcp_connection_t* connection = send_state->endpoint->connection;
  iree_slim_mutex_lock(&connection->mutex);
  IREE_ASSERT(
      send_state->phase == IREE_NET_TCP_SEND_STATE_PHASE_LOCAL_COMPLETION,
      "TCP local send completed from phase %d", (int)send_state->phase);
  const iree_net_tcp_send_completion_t completion =
      iree_net_tcp_claim_send_completion_locked(connection, send_state);
  iree_slim_mutex_unlock(&connection->mutex);
  iree_net_tcp_invoke_send_completion(completion, completion_status,
                                      /*bytes_transferred=*/0);
}

// Schedules an accepted send completion on the connection's owning proactor.
// NOP submission is allocation-free and cannot exhaust backend capacity.
static void iree_net_tcp_schedule_local_send_completion(
    iree_net_tcp_send_state_t* send_state, iree_status_t status) {
  IREE_ASSERT(!iree_status_is_ok(status));
  IREE_ASSERT(send_state->phase ==
              IREE_NET_TCP_SEND_STATE_PHASE_LOCAL_COMPLETION);
  iree_net_tcp_connection_t* connection = send_state->endpoint->connection;
  send_state->local_completion_status = status;
  iree_async_operation_zero(&send_state->local_completion_operation.base,
                            sizeof(send_state->local_completion_operation));
  iree_async_operation_initialize(
      &send_state->local_completion_operation.base,
      IREE_ASYNC_OPERATION_TYPE_NOP, IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_net_tcp_endpoint_local_send_complete, send_state);
  IREE_CHECK_OK(iree_async_proactor_submit_one(
      connection->proactor, &send_state->local_completion_operation.base));
}

static iree_net_tcp_send_state_t*
iree_net_tcp_claim_waiting_send_for_cancellation_locked(
    iree_net_tcp_endpoint_t* endpoint) {
  iree_net_tcp_send_state_t* send_state = endpoint->preactivation_send_state;
  if (!send_state ||
      send_state->phase !=
          IREE_NET_TCP_SEND_STATE_PHASE_WAITING_FOR_PEER_ACTIVE) {
    return NULL;
  }
  endpoint->preactivation_send_state = NULL;
  send_state->phase = IREE_NET_TCP_SEND_STATE_PHASE_LOCAL_COMPLETION;
  return send_state;
}

static void iree_net_tcp_endpoint_set_callbacks(
    void* self, iree_net_message_endpoint_callbacks_t callbacks) {
  iree_net_tcp_endpoint_t* endpoint = (iree_net_tcp_endpoint_t*)self;
  iree_net_tcp_connection_t* connection = endpoint->connection;
  iree_slim_mutex_lock(&connection->mutex);
  endpoint->callbacks = callbacks;
  iree_slim_mutex_unlock(&connection->mutex);
}

typedef struct iree_net_tcp_activation_work_t {
  // Endpoint whose ACTIVE frame must be submitted next.
  iree_net_tcp_endpoint_t* send_endpoint;

  // First detached endpoint whose announcement hold must be retired.
  uint32_t retire_head;
} iree_net_tcp_activation_work_t;

static void iree_net_tcp_activation_announcement_complete(
    void* user_data, iree_status_t status,
    iree_host_size_t wire_bytes_transferred);

static void iree_net_tcp_collect_activation_work_locked(
    iree_net_tcp_connection_t* connection,
    iree_net_tcp_activation_work_t* out_work) {
  *out_work = (iree_net_tcp_activation_work_t){
      .retire_head = IREE_NET_TCP_INDEX_NONE,
  };
  if (connection->activation_in_flight != IREE_NET_TCP_INDEX_NONE ||
      connection->activation_queue_head == IREE_NET_TCP_INDEX_NONE) {
    return;
  }

  if (connection->state != IREE_NET_TCP_CONNECTION_STATE_OPEN ||
      !iree_status_is_ok(connection->terminal_status)) {
    out_work->retire_head = connection->activation_queue_head;
    connection->activation_queue_head = IREE_NET_TCP_INDEX_NONE;
    connection->activation_queue_tail = IREE_NET_TCP_INDEX_NONE;
    for (uint32_t ordinal = out_work->retire_head;
         ordinal != IREE_NET_TCP_INDEX_NONE;) {
      iree_net_tcp_endpoint_t* endpoint = &connection->endpoints[ordinal];
      IREE_ASSERT(endpoint->activation_announcement_phase ==
                  IREE_NET_TCP_ACTIVATION_ANNOUNCEMENT_PHASE_QUEUED);
      endpoint->activation_announcement_phase =
          IREE_NET_TCP_ACTIVATION_ANNOUNCEMENT_PHASE_RETIRED;
      ordinal = endpoint->next_activation_ordinal;
    }
    return;
  }

  const uint32_t ordinal = connection->activation_queue_head;
  iree_net_tcp_endpoint_t* endpoint = &connection->endpoints[ordinal];
  connection->activation_queue_head = endpoint->next_activation_ordinal;
  if (connection->activation_queue_head == IREE_NET_TCP_INDEX_NONE) {
    connection->activation_queue_tail = IREE_NET_TCP_INDEX_NONE;
  }
  endpoint->next_activation_ordinal = IREE_NET_TCP_INDEX_NONE;
  IREE_ASSERT(endpoint->activation_announcement_phase ==
              IREE_NET_TCP_ACTIVATION_ANNOUNCEMENT_PHASE_QUEUED);
  endpoint->activation_announcement_phase =
      IREE_NET_TCP_ACTIVATION_ANNOUNCEMENT_PHASE_IN_FLIGHT;
  connection->activation_in_flight = ordinal;
  out_work->send_endpoint = endpoint;
}

static void iree_net_tcp_enqueue_activation_announcement_locked(
    iree_net_tcp_endpoint_t* endpoint,
    iree_net_tcp_activation_work_t* out_work) {
  iree_net_tcp_connection_t* connection = endpoint->connection;
  IREE_ASSERT(endpoint->activation_announcement_phase ==
              IREE_NET_TCP_ACTIVATION_ANNOUNCEMENT_PHASE_NONE);
  endpoint->activation_announcement_phase =
      IREE_NET_TCP_ACTIVATION_ANNOUNCEMENT_PHASE_QUEUED;
  endpoint->next_activation_ordinal = IREE_NET_TCP_INDEX_NONE;
  if (connection->activation_queue_tail == IREE_NET_TCP_INDEX_NONE) {
    connection->activation_queue_head = endpoint->ordinal;
  } else {
    connection->endpoints[connection->activation_queue_tail]
        .next_activation_ordinal = endpoint->ordinal;
  }
  connection->activation_queue_tail = endpoint->ordinal;
  iree_net_tcp_collect_activation_work_locked(connection, out_work);
}

static void iree_net_tcp_retire_activation_announcement(
    iree_net_tcp_endpoint_t* endpoint) {
  iree_net_endpoint_lifecycle_end_operation(&endpoint->lifecycle);
  iree_net_connection_release(&endpoint->connection->base);
}

static void iree_net_tcp_retire_activation_queue(
    iree_net_tcp_connection_t* connection, uint32_t retire_head) {
  for (uint32_t ordinal = retire_head; ordinal != IREE_NET_TCP_INDEX_NONE;) {
    iree_net_tcp_endpoint_t* endpoint = &connection->endpoints[ordinal];
    ordinal = endpoint->next_activation_ordinal;
    endpoint->next_activation_ordinal = IREE_NET_TCP_INDEX_NONE;
    iree_net_tcp_retire_activation_announcement(endpoint);
  }
}

static iree_status_t iree_net_tcp_write_activation_announcement(
    void* user_data, iree_byte_span_t target) {
  iree_net_tcp_endpoint_t* endpoint = (iree_net_tcp_endpoint_t*)user_data;
  iree_net_tcp_encode_frame_header(target.data,
                                   IREE_NET_TCP_FRAME_FLAG_ENDPOINT_ACTIVE,
                                   /*payload_length=*/0, endpoint->ordinal);
  return iree_ok_status();
}

static void iree_net_tcp_submit_activation_announcement(
    iree_net_tcp_endpoint_t* endpoint) {
  iree_net_tcp_connection_t* connection = endpoint->connection;
  const iree_net_message_endpoint_send_params_t params = {
      .generated_prefix =
          {
              .length = IREE_NET_TCP_FRAME_HEADER_SIZE,
              .write = iree_net_tcp_write_activation_announcement,
              .user_data = endpoint,
          },
      .data = iree_async_span_list_empty(),
      .completion_callback =
          {
              .fn = iree_net_tcp_activation_announcement_complete,
              .user_data = endpoint,
          },
  };
  iree_status_t status =
      iree_net_message_endpoint_send(connection->wire_endpoint, &params);
  if (!iree_status_is_ok(status)) {
    iree_net_tcp_activation_announcement_complete(endpoint, status,
                                                  /*wire_bytes_transferred=*/0);
  }
}

static void iree_net_tcp_activation_announcement_complete(
    void* user_data, iree_status_t status,
    iree_host_size_t wire_bytes_transferred) {
  iree_net_tcp_endpoint_t* endpoint = (iree_net_tcp_endpoint_t*)user_data;
  iree_net_tcp_connection_t* connection = endpoint->connection;
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&connection->mutex);
    const bool connection_is_open =
        connection->state == IREE_NET_TCP_CONNECTION_STATE_OPEN;
    iree_slim_mutex_unlock(&connection->mutex);
    if (connection_is_open) {
      iree_net_tcp_connection_record_terminal_error(connection, status);
    } else {
      iree_status_free(status);
    }
  } else {
    IREE_ASSERT(wire_bytes_transferred == IREE_NET_TCP_FRAME_HEADER_SIZE,
                "TCP ACTIVE completed %" PRIhsz " of %u wire bytes",
                wire_bytes_transferred, IREE_NET_TCP_FRAME_HEADER_SIZE);
    iree_status_free(status);
  }

  iree_net_tcp_activation_work_t work;
  iree_slim_mutex_lock(&connection->mutex);
  IREE_ASSERT(connection->activation_in_flight == endpoint->ordinal);
  IREE_ASSERT(endpoint->activation_announcement_phase ==
              IREE_NET_TCP_ACTIVATION_ANNOUNCEMENT_PHASE_IN_FLIGHT);
  connection->activation_in_flight = IREE_NET_TCP_INDEX_NONE;
  endpoint->activation_announcement_phase =
      IREE_NET_TCP_ACTIVATION_ANNOUNCEMENT_PHASE_RETIRED;
  iree_net_tcp_collect_activation_work_locked(connection, &work);
  iree_slim_mutex_unlock(&connection->mutex);

  iree_net_tcp_retire_activation_queue(connection, work.retire_head);
  if (work.send_endpoint) {
    iree_net_tcp_submit_activation_announcement(work.send_endpoint);
  }
  iree_net_tcp_retire_activation_announcement(endpoint);
}

static void iree_net_tcp_endpoint_activation_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  IREE_ASSERT(!iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE),
              "endpoint activation NOP produced a nonterminal completion");
  iree_net_tcp_endpoint_t* endpoint = (iree_net_tcp_endpoint_t*)user_data;
  iree_net_tcp_connection_t* connection = endpoint->connection;

  bool activation_succeeded = iree_status_is_ok(status);
  if (!iree_status_is_ok(status)) {
    iree_net_tcp_connection_record_terminal_error(connection, status);
  } else {
    iree_status_free(status);
    iree_net_tcp_pending_frame_t pending_frame;
    iree_net_message_endpoint_callbacks_t callbacks = {0};
    iree_slim_mutex_lock(&connection->mutex);
    const bool has_pending_frame =
        endpoint->phase == IREE_NET_TCP_ENDPOINT_PHASE_ACTIVATING &&
        iree_net_tcp_take_pending_frame_locked(endpoint, &pending_frame);
    if (has_pending_frame) {
      callbacks = endpoint->callbacks;
    }
    iree_slim_mutex_unlock(&connection->mutex);
    if (has_pending_frame) {
      iree_status_t callback_status = callbacks.on_message(
          callbacks.user_data, pending_frame.payload, &pending_frame.lease);
      iree_net_tcp_release_pending_frame(&pending_frame);
      if (!iree_status_is_ok(callback_status)) {
        activation_succeeded = false;
        iree_net_tcp_connection_record_terminal_error(connection,
                                                      callback_status);
      }
    }
  }

  iree_net_tcp_activation_work_t work = {
      .retire_head = IREE_NET_TCP_INDEX_NONE,
  };
  bool announcement_queued = false;
  iree_slim_mutex_lock(&connection->mutex);
  if (activation_succeeded &&
      endpoint->phase == IREE_NET_TCP_ENDPOINT_PHASE_ACTIVATING &&
      connection->state == IREE_NET_TCP_CONNECTION_STATE_OPEN &&
      iree_status_is_ok(connection->terminal_status)) {
    endpoint->phase = IREE_NET_TCP_ENDPOINT_PHASE_ACTIVE;
    iree_net_tcp_enqueue_activation_announcement_locked(endpoint, &work);
    announcement_queued = true;
  }
  iree_slim_mutex_unlock(&connection->mutex);

  if (announcement_queued) {
    if (work.send_endpoint) {
      iree_net_tcp_submit_activation_announcement(work.send_endpoint);
    }
  } else {
    iree_net_endpoint_lifecycle_end_operation(&endpoint->lifecycle);
    iree_net_connection_release(&connection->base);
  }
}

static iree_status_t iree_net_tcp_endpoint_activate(void* self) {
  iree_net_tcp_endpoint_t* endpoint = (iree_net_tcp_endpoint_t*)self;
  iree_net_tcp_connection_t* connection = endpoint->connection;
  bool release_connection = false;
  iree_slim_mutex_lock(&connection->mutex);
  iree_status_t status = iree_ok_status();
  if (endpoint->phase != IREE_NET_TCP_ENDPOINT_PHASE_CREATED) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "TCP endpoint %u cannot activate from phase %d",
                              endpoint->ordinal, (int)endpoint->phase);
  } else if (!endpoint->callbacks.on_message || !endpoint->callbacks.on_error) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "message and error callbacks are required");
  } else if (connection->state != IREE_NET_TCP_CONNECTION_STATE_OPEN) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "TCP connection is deactivating");
  } else if (!iree_status_is_ok(connection->terminal_status)) {
    status = iree_status_clone(connection->terminal_status);
  } else {
    status = iree_net_endpoint_lifecycle_activate(&endpoint->lifecycle);
  }

  if (iree_status_is_ok(status)) {
    const bool operation_accepted =
        iree_net_endpoint_lifecycle_try_begin_operation(&endpoint->lifecycle);
    IREE_ASSERT(operation_accepted,
                "newly activated TCP endpoint rejected its drain operation");
    endpoint->phase = IREE_NET_TCP_ENDPOINT_PHASE_ACTIVATING;
    iree_async_operation_initialize(
        &endpoint->activation_operation.base, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE,
        iree_net_tcp_endpoint_activation_complete, endpoint);
    iree_net_connection_retain(&connection->base);
    status = iree_async_proactor_submit_one(
        connection->proactor, &endpoint->activation_operation.base);
    if (!iree_status_is_ok(status)) {
      endpoint->phase = IREE_NET_TCP_ENDPOINT_PHASE_CREATED;
      iree_net_endpoint_lifecycle_end_operation(&endpoint->lifecycle);
      iree_net_endpoint_lifecycle_rollback_activation(&endpoint->lifecycle);
      release_connection = true;
    }
  }
  iree_slim_mutex_unlock(&connection->mutex);
  if (release_connection) {
    iree_net_connection_release(&connection->base);
  }
  return status;
}

static void iree_net_tcp_endpoint_deactivation_complete(void* user_data) {
  iree_net_tcp_endpoint_t* endpoint = (iree_net_tcp_endpoint_t*)user_data;
  iree_net_tcp_connection_t* connection = endpoint->connection;
  iree_slim_mutex_lock(&connection->mutex);
  endpoint->phase = IREE_NET_TCP_ENDPOINT_PHASE_DEACTIVATED;
  iree_net_message_endpoint_deactivate_fn_t callback =
      endpoint->deactivate_callback.fn;
  void* callback_user_data = endpoint->deactivate_callback.user_data;
  endpoint->deactivate_callback.fn = NULL;
  endpoint->deactivate_callback.user_data = NULL;
  iree_slim_mutex_unlock(&connection->mutex);
  if (callback) {
    callback(callback_user_data);
  }
}

static iree_status_t iree_net_tcp_endpoint_deactivate(
    void* self, iree_net_message_endpoint_deactivate_fn_t callback,
    void* user_data) {
  iree_net_tcp_endpoint_t* endpoint = (iree_net_tcp_endpoint_t*)self;
  iree_net_tcp_connection_t* connection = endpoint->connection;
  iree_net_tcp_send_state_t* send_state_to_cancel = NULL;
  iree_slim_mutex_lock(&connection->mutex);
  iree_net_endpoint_lifecycle_actions_t actions =
      IREE_NET_ENDPOINT_LIFECYCLE_ACTION_NONE;
  iree_status_t status = iree_net_endpoint_lifecycle_request_deactivation(
      &endpoint->lifecycle, iree_net_tcp_endpoint_deactivation_complete,
      endpoint, &actions);
  if (iree_status_is_ok(status)) {
    endpoint->phase = IREE_NET_TCP_ENDPOINT_PHASE_DRAINING;
    endpoint->deactivate_callback.fn = callback;
    endpoint->deactivate_callback.user_data = user_data;
    send_state_to_cancel =
        iree_net_tcp_claim_waiting_send_for_cancellation_locked(endpoint);
  }
  iree_slim_mutex_unlock(&connection->mutex);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  if (send_state_to_cancel) {
    iree_net_tcp_schedule_local_send_completion(
        send_state_to_cancel,
        iree_make_status(IREE_STATUS_CANCELLED,
                         "TCP send cancelled during endpoint deactivation"));
  }
  iree_net_tcp_clear_endpoint_pending_frame(endpoint);
  IREE_ASSERT(
      iree_any_bit_set(actions,
                       IREE_NET_ENDPOINT_LIFECYCLE_ACTION_BEGIN_DEACTIVATION),
      "accepted endpoint deactivation did not begin owner drain");
  iree_net_endpoint_lifecycle_complete_deactivation(&endpoint->lifecycle);
  return iree_ok_status();
}

static iree_status_t iree_net_tcp_endpoint_acquire_send_state(
    iree_net_tcp_endpoint_t* endpoint,
    iree_net_tcp_send_state_t** out_send_state) {
  *out_send_state = NULL;
  iree_net_tcp_connection_t* connection = endpoint->connection;
  iree_slim_mutex_lock(&connection->mutex);
  iree_status_t status = iree_ok_status();
  if (!iree_status_is_ok(connection->terminal_status)) {
    status = iree_status_clone(connection->terminal_status);
  } else if ((endpoint->phase != IREE_NET_TCP_ENDPOINT_PHASE_ACTIVATING &&
              endpoint->phase != IREE_NET_TCP_ENDPOINT_PHASE_ACTIVE) ||
             !iree_net_endpoint_lifecycle_try_begin_operation(
                 &endpoint->lifecycle)) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "TCP endpoint %u is not active", endpoint->ordinal);
  } else if (!endpoint->peer_active && endpoint->preactivation_send_state) {
    iree_net_endpoint_lifecycle_end_operation(&endpoint->lifecycle);
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "TCP endpoint %u is waiting for peer activation",
                              endpoint->ordinal);
  } else {
    *out_send_state =
        iree_net_tcp_acquire_send_state_locked(connection, endpoint);
    if (!*out_send_state) {
      iree_net_endpoint_lifecycle_end_operation(&endpoint->lifecycle);
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "TCP connection send slots are exhausted");
    } else if (!endpoint->peer_active) {
      endpoint->preactivation_send_state = *out_send_state;
    }
  }
  iree_slim_mutex_unlock(&connection->mutex);
  return status;
}

static iree_status_t iree_net_tcp_endpoint_send(
    void* self, const iree_net_message_endpoint_send_params_t* params) {
  iree_net_tcp_endpoint_t* endpoint = (iree_net_tcp_endpoint_t*)self;
  iree_net_tcp_connection_t* connection = endpoint->connection;
  const iree_host_size_t payload_length = iree_net_tcp_message_length(params);
  uint32_t frame_size = 0;
  IREE_RETURN_IF_ERROR(iree_net_tcp_calculate_frame_size(
      connection, payload_length, &frame_size));

  iree_net_tcp_send_state_t* send_state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_net_tcp_endpoint_acquire_send_state(endpoint, &send_state));
  send_state->payload_length = payload_length;
  send_state->completion_callback = params->completion_callback;

  iree_net_tcp_frame_prefix_t frame_prefix = {
      .flags = IREE_NET_TCP_FRAME_FLAG_NONE,
      .payload_length = (uint32_t)payload_length,
      .endpoint_ordinal = endpoint->ordinal,
      .message_prefix = params->generated_prefix,
  };
  iree_net_message_endpoint_send_params_t wire_params = {
      .generated_prefix =
          {
              .length = IREE_NET_TCP_FRAME_HEADER_SIZE +
                        params->generated_prefix.length,
              .write = iree_net_tcp_write_frame_prefix,
              .user_data = &frame_prefix,
          },
      .data = params->data,
      .completion_callback =
          {
              .fn = iree_net_tcp_endpoint_send_complete,
              .user_data = send_state,
          },
  };
  iree_status_t status =
      iree_net_message_endpoint_send(connection->wire_endpoint, &wire_params);

  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&connection->mutex);
    IREE_ASSERT(send_state->phase ==
                IREE_NET_TCP_SEND_STATE_PHASE_RAW_IN_FLIGHT);
    send_state->phase = IREE_NET_TCP_SEND_STATE_PHASE_LOCAL_COMPLETION;
    iree_slim_mutex_unlock(&connection->mutex);
    iree_net_tcp_schedule_local_send_completion(send_state, status);
  }
  return iree_ok_status();
}

static iree_net_carrier_send_budget_t iree_net_tcp_endpoint_query_send_budget(
    void* self) {
  iree_net_tcp_endpoint_t* endpoint = (iree_net_tcp_endpoint_t*)self;
  iree_net_tcp_connection_t* connection = endpoint->connection;
  uint32_t free_send_state_count = 0;
  bool active = false;
  bool peer_active = false;
  bool preactivation_credit_available = false;
  iree_slim_mutex_lock(&connection->mutex);
  active = iree_status_is_ok(connection->terminal_status) &&
           (endpoint->phase == IREE_NET_TCP_ENDPOINT_PHASE_ACTIVATING ||
            endpoint->phase == IREE_NET_TCP_ENDPOINT_PHASE_ACTIVE);
  if (active) {
    free_send_state_count = connection->free_send_state_count;
    peer_active = endpoint->peer_active;
    preactivation_credit_available = !endpoint->preactivation_send_state;
  }
  iree_slim_mutex_unlock(&connection->mutex);
  if (!active) {
    return (iree_net_carrier_send_budget_t){0};
  }

  iree_net_carrier_send_budget_t budget =
      iree_net_message_endpoint_query_send_budget(connection->wire_endpoint);
  budget.slots = iree_min(budget.slots, free_send_state_count);
  if (!peer_active) {
    budget.slots =
        preactivation_credit_available ? iree_min(budget.slots, 1u) : 0u;
  }
  const iree_host_size_t max_payload_size =
      connection->max_frame_size - IREE_NET_TCP_FRAME_HEADER_SIZE;
  if (budget.bytes != IREE_HOST_SIZE_MAX) {
    budget.bytes = budget.bytes > IREE_NET_TCP_FRAME_HEADER_SIZE
                       ? budget.bytes - IREE_NET_TCP_FRAME_HEADER_SIZE
                       : 0;
  }
  budget.bytes = iree_min(budget.bytes, max_payload_size);
  return budget;
}

static const iree_net_message_endpoint_vtable_t iree_net_tcp_endpoint_vtable = {
    .set_callbacks = iree_net_tcp_endpoint_set_callbacks,
    .activate = iree_net_tcp_endpoint_activate,
    .deactivate = iree_net_tcp_endpoint_deactivate,
    .send = iree_net_tcp_endpoint_send,
    .query_send_budget = iree_net_tcp_endpoint_query_send_budget,
};

//===----------------------------------------------------------------------===//
// Connection implementation
//===----------------------------------------------------------------------===//

static void iree_net_tcp_connection_deactivation_complete(void* user_data) {
  iree_net_tcp_connection_t* connection = (iree_net_tcp_connection_t*)user_data;
  iree_slim_mutex_lock(&connection->mutex);
  IREE_ASSERT(
      connection->state == IREE_NET_TCP_CONNECTION_STATE_DRAINING_ENDPOINTS,
      "TCP connection drain completed from state %d", (int)connection->state);
  connection->state = IREE_NET_TCP_CONNECTION_STATE_DEACTIVATED;
  iree_net_connection_deactivate_callback_t callback =
      connection->deactivate_callback;
  connection->deactivate_callback =
      (iree_net_connection_deactivate_callback_t){0};
  iree_slim_mutex_unlock(&connection->mutex);

  callback.fn(callback.user_data);
  iree_net_connection_release(&connection->base);
}

static void iree_net_tcp_connection_begin_endpoint_drain(
    iree_net_tcp_connection_t* connection) {
  for (uint32_t i = 0; i < connection->base.max_endpoint_count; ++i) {
    iree_net_tcp_endpoint_t* endpoint = &connection->endpoints[i];
    iree_net_tcp_send_state_t* send_state_to_cancel = NULL;
    iree_slim_mutex_lock(&connection->mutex);
    iree_net_endpoint_lifecycle_actions_t actions =
        iree_net_endpoint_lifecycle_join_deactivation(&endpoint->lifecycle);
    if (iree_any_bit_set(
            actions, IREE_NET_ENDPOINT_LIFECYCLE_ACTION_BEGIN_DEACTIVATION)) {
      endpoint->phase = IREE_NET_TCP_ENDPOINT_PHASE_DRAINING;
    }
    send_state_to_cancel =
        iree_net_tcp_claim_waiting_send_for_cancellation_locked(endpoint);
    iree_slim_mutex_unlock(&connection->mutex);

    if (send_state_to_cancel) {
      iree_net_tcp_schedule_local_send_completion(
          send_state_to_cancel,
          iree_make_status(
              IREE_STATUS_CANCELLED,
              "TCP send cancelled during connection deactivation"));
    }
    iree_net_tcp_clear_endpoint_pending_frame(endpoint);
    if (iree_any_bit_set(
            actions, IREE_NET_ENDPOINT_LIFECYCLE_ACTION_BEGIN_DEACTIVATION)) {
      iree_net_endpoint_lifecycle_complete_deactivation(&endpoint->lifecycle);
    }
  }

  iree_net_framing_adapter_join_deactivation(connection->framing_adapter);
  iree_net_endpoint_deactivation_barrier_commit(
      &connection->deactivation_barrier,
      (iree_net_connection_deactivate_callback_t){
          .fn = iree_net_tcp_connection_deactivation_complete,
          .user_data = connection,
      });
}

static void iree_net_tcp_endpoint_ready_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  IREE_ASSERT(!iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE),
              "endpoint-ready NOP produced a nonterminal completion");
  iree_net_tcp_endpoint_t* endpoint = (iree_net_tcp_endpoint_t*)user_data;
  iree_net_tcp_connection_t* connection = endpoint->connection;

  iree_net_message_endpoint_t message_endpoint = {0};
  iree_slim_mutex_lock(&connection->mutex);
  iree_net_endpoint_ready_callback_t ready_callback = endpoint->ready_callback;
  endpoint->ready_callback = (iree_net_endpoint_ready_callback_t){0};
  if (iree_status_is_ok(status) &&
      connection->state == IREE_NET_TCP_CONNECTION_STATE_OPEN &&
      iree_status_is_ok(connection->terminal_status)) {
    message_endpoint.self = endpoint;
    message_endpoint.vtable = &iree_net_tcp_endpoint_vtable;
  } else if (iree_status_is_ok(status) &&
             !iree_status_is_ok(connection->terminal_status)) {
    iree_status_free(status);
    status = iree_status_clone(connection->terminal_status);
  } else if (iree_status_is_ok(status)) {
    status = iree_make_status(IREE_STATUS_CANCELLED,
                              "TCP connection deactivated before endpoint "
                              "ready");
  }
  iree_slim_mutex_unlock(&connection->mutex);

  ready_callback.fn(ready_callback.user_data, status, message_endpoint);

  bool begin_endpoint_drain = false;
  iree_slim_mutex_lock(&connection->mutex);
  IREE_ASSERT(connection->pending_ready_count > 0,
              "TCP connection retired an unowned ready callback");
  --connection->pending_ready_count;
  if (connection->pending_ready_count == 0 &&
      connection->state ==
          IREE_NET_TCP_CONNECTION_STATE_DRAINING_READY_CALLBACKS) {
    connection->state = IREE_NET_TCP_CONNECTION_STATE_DRAINING_ENDPOINTS;
    begin_endpoint_drain = true;
  }
  iree_slim_mutex_unlock(&connection->mutex);

  if (begin_endpoint_drain) {
    iree_net_tcp_connection_begin_endpoint_drain(connection);
  }
  iree_net_connection_release(&connection->base);
}

static void iree_net_tcp_connection_destroy(
    iree_net_connection_t* base_connection) {
  iree_net_tcp_connection_t* connection =
      (iree_net_tcp_connection_t*)base_connection;
  IREE_ASSERT(
      !connection->published ||
          connection->state == IREE_NET_TCP_CONNECTION_STATE_DEACTIVATED,
      "published TCP connection released before deactivation");
  IREE_ASSERT(connection->pending_ready_count == 0,
              "TCP connection destroyed with pending ready callbacks");
  IREE_ASSERT(connection->free_send_state_count == connection->send_state_count,
              "TCP connection destroyed with owned send states");
  IREE_ASSERT(
      connection->activation_queue_head == IREE_NET_TCP_INDEX_NONE &&
          connection->activation_queue_tail == IREE_NET_TCP_INDEX_NONE &&
          connection->activation_in_flight == IREE_NET_TCP_INDEX_NONE,
      "TCP connection destroyed with ACTIVE announcements");

  iree_allocator_t host_allocator = connection->base.host_allocator;
  for (uint32_t i = 0; i < connection->base.max_endpoint_count; ++i) {
    iree_net_tcp_endpoint_t* endpoint = &connection->endpoints[i];
    IREE_ASSERT(
        !endpoint->terminal_error_callback_pending,
        "TCP connection destroyed with terminal endpoint error pending");
    IREE_ASSERT(!endpoint->pending_frame_present,
                "TCP connection destroyed with a retained frame");
    IREE_ASSERT(!endpoint->preactivation_send_state,
                "TCP connection destroyed with a rendezvous send");
    IREE_ASSERT(endpoint->activation_announcement_phase !=
                        IREE_NET_TCP_ACTIVATION_ANNOUNCEMENT_PHASE_QUEUED &&
                    endpoint->activation_announcement_phase !=
                        IREE_NET_TCP_ACTIVATION_ANNOUNCEMENT_PHASE_IN_FLIGHT,
                "TCP connection destroyed with endpoint ACTIVE pending");
    iree_net_endpoint_lifecycle_deinitialize(&endpoint->lifecycle);
  }
  iree_status_free(connection->terminal_status);
  iree_net_framing_adapter_free(connection->framing_adapter);
  iree_async_proactor_release(connection->proactor);
  iree_slim_mutex_deinitialize(&connection->mutex);
  iree_allocator_free(host_allocator, connection);
}

static void iree_net_tcp_connection_deactivate(
    iree_net_connection_t* base_connection,
    iree_net_connection_deactivate_callback_t callback) {
  iree_net_tcp_connection_t* connection =
      (iree_net_tcp_connection_t*)base_connection;
  bool begin_endpoint_drain = false;
  bool valid_request = false;
  iree_slim_mutex_lock(&connection->mutex);
  if (connection->published &&
      connection->state == IREE_NET_TCP_CONNECTION_STATE_OPEN) {
    valid_request = true;
    connection->deactivate_callback = callback;
    iree_net_connection_retain(base_connection);
    if (connection->pending_ready_count == 0) {
      connection->state = IREE_NET_TCP_CONNECTION_STATE_DRAINING_ENDPOINTS;
      begin_endpoint_drain = true;
    } else {
      connection->state =
          IREE_NET_TCP_CONNECTION_STATE_DRAINING_READY_CALLBACKS;
    }
  }
  iree_slim_mutex_unlock(&connection->mutex);

  IREE_ASSERT(valid_request,
              "TCP connection deactivated from an invalid state");
  if (begin_endpoint_drain) {
    iree_net_tcp_connection_begin_endpoint_drain(connection);
  }
}

static iree_status_t iree_net_tcp_connection_open_endpoint(
    iree_net_connection_t* base_connection,
    iree_net_endpoint_ready_callback_t callback) {
  iree_net_tcp_connection_t* connection =
      (iree_net_tcp_connection_t*)base_connection;
  bool release_connection = false;
  iree_slim_mutex_lock(&connection->mutex);
  iree_status_t status = iree_ok_status();
  iree_net_tcp_endpoint_t* endpoint = NULL;
  if (!connection->published) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "TCP connection is not published");
  } else if (connection->state != IREE_NET_TCP_CONNECTION_STATE_OPEN) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "TCP connection is deactivating");
  } else if (!iree_status_is_ok(connection->terminal_status)) {
    status = iree_status_clone(connection->terminal_status);
  } else if (connection->opened_endpoint_count >=
             connection->base.max_endpoint_count) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "all %u TCP endpoint slots are claimed",
                              connection->base.max_endpoint_count);
  } else {
    endpoint = &connection->endpoints[connection->opened_endpoint_count];
    endpoint->ready_callback = callback;
    iree_async_operation_initialize(
        &endpoint->ready_operation.base, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_tcp_endpoint_ready_complete,
        endpoint);
    ++connection->pending_ready_count;
    iree_net_connection_retain(base_connection);
    status = iree_async_proactor_submit_one(connection->proactor,
                                            &endpoint->ready_operation.base);
    if (iree_status_is_ok(status)) {
      ++connection->opened_endpoint_count;
    } else {
      --connection->pending_ready_count;
      endpoint->ready_callback = (iree_net_endpoint_ready_callback_t){0};
      release_connection = true;
    }
  }
  iree_slim_mutex_unlock(&connection->mutex);
  if (release_connection) {
    iree_net_connection_release(base_connection);
  }
  return status;
}

static iree_async_proactor_t* iree_net_tcp_connection_proactor(
    iree_net_connection_t* base_connection) {
  return ((iree_net_tcp_connection_t*)base_connection)->proactor;
}

static const iree_net_connection_vtable_t iree_net_tcp_connection_vtable = {
    .destroy = iree_net_tcp_connection_destroy,
    .deactivate = iree_net_tcp_connection_deactivate,
    .open_endpoint = iree_net_tcp_connection_open_endpoint,
    .open_direct_endpoint = NULL,
    .proactor = iree_net_tcp_connection_proactor,
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

static iree_status_t iree_net_tcp_connection_options_validate_impl(
    const iree_net_tcp_connection_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "TCP connection options are required");
  }
  if (options->max_endpoint_count == 0 ||
      options->max_endpoint_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "TCP endpoint count must be in [1, %u]",
                            UINT16_MAX);
  }
  if (options->max_frame_size <= IREE_NET_TCP_FRAME_HEADER_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "TCP frame limit must exceed the %u-byte header",
                            IREE_NET_TCP_FRAME_HEADER_SIZE);
  }
  if (options->carrier_options.max_send_operations == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "TCP send operation limit must be nonzero");
  }
  if (options->carrier_options.max_send_operations >=
      IREE_NET_TCP_INDEX_NONE - 1u) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "TCP send operation count leaves no internal ACTIVE slot");
  }
  return iree_ok_status();
}

iree_status_t iree_net_tcp_connection_options_validate(
    const iree_net_tcp_connection_options_t* options) {
  return iree_net_tcp_connection_options_validate_impl(options);
}

iree_status_t iree_net_tcp_connection_create(
    iree_async_proactor_t* proactor, iree_async_socket_t* socket,
    iree_async_buffer_pool_t* receive_pool,
    const iree_net_tcp_connection_options_t* options,
    iree_allocator_t host_allocator, iree_net_connection_t** out_connection) {
  IREE_ASSERT_ARGUMENT(out_connection);
  *out_connection = NULL;
  if (!proactor || !socket || !receive_pool) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "proactor, connected socket, and receive pool are required");
  }
  iree_net_tcp_connection_options_t default_options =
      iree_net_tcp_connection_options_default();
  if (!options) {
    options = &default_options;
  }
  IREE_RETURN_IF_ERROR(iree_net_tcp_connection_options_validate_impl(options));

  iree_host_size_t endpoint_offset = 0;
  iree_host_size_t send_state_offset = 0;
  iree_host_size_t allocation_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_net_tcp_connection_t), &allocation_size,
      IREE_STRUCT_FIELD_ALIGNED(
          options->max_endpoint_count, iree_net_tcp_endpoint_t,
          iree_alignof(iree_net_tcp_endpoint_t), &endpoint_offset),
      IREE_STRUCT_FIELD_ALIGNED(options->carrier_options.max_send_operations,
                                iree_net_tcp_send_state_t,
                                iree_alignof(iree_net_tcp_send_state_t),
                                &send_state_offset)));

  iree_net_tcp_connection_t* connection = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, allocation_size,
                                             (void**)&connection));
  iree_net_connection_initialize(&iree_net_tcp_connection_vtable,
                                 host_allocator, options->max_endpoint_count,
                                 &connection->base);
  iree_slim_mutex_initialize(&connection->mutex);
  iree_net_endpoint_deactivation_barrier_initialize(
      &connection->deactivation_barrier);
  connection->proactor = proactor;
  iree_async_proactor_retain(proactor);
  connection->state = IREE_NET_TCP_CONNECTION_STATE_OPEN;
  connection->max_frame_size = options->max_frame_size;
  connection->terminal_status = iree_ok_status();
  connection->endpoints =
      (iree_net_tcp_endpoint_t*)((uint8_t*)connection + endpoint_offset);
  connection->send_state_count = options->carrier_options.max_send_operations;
  connection->free_send_state_count = connection->send_state_count;
  connection->free_send_state_head = 0;
  connection->send_states =
      (iree_net_tcp_send_state_t*)((uint8_t*)connection + send_state_offset);
  connection->activation_queue_head = IREE_NET_TCP_INDEX_NONE;
  connection->activation_queue_tail = IREE_NET_TCP_INDEX_NONE;
  connection->activation_in_flight = IREE_NET_TCP_INDEX_NONE;

  for (uint32_t i = 0; i < options->max_endpoint_count; ++i) {
    iree_net_tcp_endpoint_t* endpoint = &connection->endpoints[i];
    endpoint->connection = connection;
    endpoint->ordinal = (uint16_t)i;
    endpoint->phase = IREE_NET_TCP_ENDPOINT_PHASE_CREATED;
    endpoint->next_activation_ordinal = IREE_NET_TCP_INDEX_NONE;
    iree_net_endpoint_lifecycle_initialize(&connection->deactivation_barrier,
                                           &endpoint->lifecycle);
  }
  for (uint32_t i = 0; i < connection->send_state_count; ++i) {
    iree_net_tcp_send_state_t* send_state = &connection->send_states[i];
    send_state->index = i;
    send_state->local_completion_status = iree_ok_status();
    send_state->next_free =
        i + 1 < connection->send_state_count ? i + 1 : IREE_NET_TCP_INDEX_NONE;
  }

  iree_net_stream_carrier_options_t carrier_options = options->carrier_options;
  ++carrier_options.max_send_operations;
  iree_net_carrier_t* carrier = NULL;
  iree_status_t status = iree_net_stream_carrier_create(
      proactor, socket, receive_pool, &carrier_options, host_allocator,
      &carrier);
  if (iree_status_is_ok(status)) {
    iree_net_frame_length_callback_t frame_length = {
        .fn = iree_net_tcp_resolve_frame_size,
        .user_data = connection,
        .max_header_size = IREE_NET_TCP_FRAME_HEADER_SIZE,
    };
    status = iree_net_framing_adapter_allocate(
        carrier, frame_length, options->max_frame_size,
        &connection->deactivation_barrier, host_allocator,
        &connection->framing_adapter);
    if (iree_status_is_ok(status)) {
      carrier = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    connection->wire_endpoint =
        iree_net_framing_adapter_as_endpoint(connection->framing_adapter);
    iree_net_message_endpoint_set_callbacks(
        connection->wire_endpoint, (iree_net_message_endpoint_callbacks_t){
                                       .on_message = iree_net_tcp_on_wire_frame,
                                       .on_error = iree_net_tcp_on_wire_error,
                                       .user_data = connection,
                                   });
    status = iree_net_message_endpoint_activate(connection->wire_endpoint);
  }

  if (iree_status_is_ok(status)) {
    connection->published = true;
    *out_connection = &connection->base;
  } else {
    iree_net_carrier_release(carrier);
    iree_net_connection_release(&connection->base);
  }
  return status;
}
