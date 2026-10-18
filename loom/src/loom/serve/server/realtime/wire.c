// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/wire.h"

#include <inttypes.h>

struct loom_serve_realtime_wire_message_t {
  // Ordered owner-only queue link.
  loom_serve_realtime_wire_message_t* next;
  // Owned complete JSON or control payload, immutable while queued.
  iree_string_builder_t payload;
  // RFC 6455 message opcode; JSON events use TEXT.
  loom_serve_websocket_message_type_t type;
  // Response identity for cancellation, or zero for session/control events.
  uint64_t response;
  // This unsent data publication is suppressed by response cancellation.
  bool publication;
};

void loom_serve_realtime_wire_initialize(
    uint64_t session_id, loom_serve_websocket_t* socket,
    loom_serve_http_connection_t* connection, iree_allocator_t host_allocator,
    loom_serve_realtime_wire_t* wire) {
  *wire = (loom_serve_realtime_wire_t){.host_allocator = host_allocator,
                                       .socket = socket,
                                       .connection = connection,
                                       .session_id = session_id};
  iree_string_builder_initialize(host_allocator, &wire->scratch);
}

static void loom_serve_realtime_wire_message_free(
    loom_serve_realtime_wire_t* wire,
    loom_serve_realtime_wire_message_t* message) {
  wire->bytes -= message->payload.capacity;
  --wire->count;
  // Retain the larger reusable allocation without retaining an extra event.
  if (wire->scratch.capacity < message->payload.capacity) {
    iree_string_builder_t scratch = wire->scratch;
    wire->scratch = message->payload;
    message->payload = scratch;
  }
  iree_string_builder_reset(&wire->scratch);
  iree_string_builder_deinitialize(&message->payload);
  iree_allocator_free(wire->host_allocator, message);
}

void loom_serve_realtime_wire_clear(loom_serve_realtime_wire_t* wire) {
  while (wire->first) {
    loom_serve_realtime_wire_message_t* message = wire->first;
    wire->first = message->next;
    loom_serve_realtime_wire_message_free(wire, message);
  }
  wire->last = NULL;
}

void loom_serve_realtime_wire_deinitialize(loom_serve_realtime_wire_t* wire) {
  loom_serve_realtime_wire_clear(wire);
  iree_string_builder_deinitialize(&wire->scratch);
}

iree_status_t loom_serve_realtime_wire_begin(loom_serve_realtime_wire_t* wire,
                                             const char* type) {
  iree_string_builder_reset(&wire->scratch);
  return iree_string_builder_append_format(
      &wire->scratch,
      "{\"event_id\":\"event_%" PRIu64 "_%" PRIu64 "\",\"type\":\"%s\"",
      wire->session_id, ++wire->sequence, type);
}

iree_status_t loom_serve_realtime_wire_commit(loom_serve_realtime_wire_t* wire,
                                              uint64_t response,
                                              bool publication) {
  if (wire->count == 64 || wire->scratch.size > 512 * 1024 ||
      wire->scratch.capacity > 1024 * 1024 - wire->bytes) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "realtime output credit exhausted");
  }
  loom_serve_realtime_wire_message_t* message = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      wire->host_allocator, sizeof(*message), (void**)&message));
  message->payload = wire->scratch;
  iree_string_builder_initialize(wire->host_allocator, &wire->scratch);
  message->type = LOOM_SERVE_WEBSOCKET_TEXT;
  message->response = response;
  message->publication = publication;
  if (wire->last) {
    wire->last->next = message;
  } else {
    wire->first = message;
  }
  wire->last = message;
  wire->bytes += message->payload.capacity;
  ++wire->count;
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_wire_control(
    loom_serve_realtime_wire_t* wire, loom_serve_websocket_message_type_t type,
    iree_const_byte_span_t payload) {
  iree_string_builder_reset(&wire->scratch);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      &wire->scratch,
      iree_make_string_view((const char*)payload.data, payload.data_length)));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_commit(wire, 0, false));
  wire->last->type = type;
  return iree_ok_status();
}

void loom_serve_realtime_wire_cancel(loom_serve_realtime_wire_t* wire,
                                     uint64_t response) {
  loom_serve_realtime_wire_message_t* previous = NULL;
  loom_serve_realtime_wire_message_t** link = &wire->first;
  while (*link) {
    loom_serve_realtime_wire_message_t* message = *link;
    if (message->response == response && message->publication) {
      *link = message->next;
      if (wire->last == message) {
        wire->last = previous;
      }
      loom_serve_realtime_wire_message_free(wire, message);
    } else {
      previous = message;
      link = &message->next;
    }
  }
}

iree_status_t loom_serve_realtime_wire_flush(loom_serve_realtime_wire_t* wire,
                                             bool* out_progress) {
  if (!wire->first || !loom_serve_http_connection_can_send(wire->connection)) {
    return iree_ok_status();
  }
  loom_serve_realtime_wire_message_t* message = wire->first;
  IREE_RETURN_IF_ERROR(loom_serve_websocket_send(
      wire->socket, message->type,
      iree_make_const_byte_span(message->payload.buffer,
                                message->payload.size)));
  wire->first = message->next;
  if (!wire->first) {
    wire->last = NULL;
  }
  loom_serve_realtime_wire_message_free(wire, message);
  *out_progress = true;
  return iree_ok_status();
}

bool loom_serve_realtime_wire_can_publish(
    const loom_serve_realtime_wire_t* wire) {
  return !wire->first && loom_serve_http_connection_can_send(wire->connection);
}
