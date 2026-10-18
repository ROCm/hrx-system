// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/transport/websocket/connection.h"

#include <string.h>

#include "iree/base/internal/unicode.h"
#include "iree/base/threading/mutex.h"

struct loom_serve_websocket_message_t {
  // Next completed message in publication order, or NULL for the tail.
  struct loom_serve_websocket_message_t* next;
  // Initial data opcode, retained across continuation frames, or control type.
  loom_serve_websocket_message_type_t type;
  // Initialized payload bytes, excluding framing and masking.
  iree_host_size_t length;
  // Allocated payload bytes charged against the connection's retained budget.
  iree_host_size_t capacity;
  // Owned payload; published records never resize or change.
  uint8_t data[];
};

struct loom_serve_websocket_t {
  // Allocator owning all receiver storage through HTTP reclamation.
  iree_allocator_t host_allocator;
  // Borrowed connection retaining this receiver and all accepted network work.
  loom_serve_http_connection_t* connection;
  // Immutable payload and record bounds.
  loom_serve_websocket_limits_t limits;
  // Serializes receive publication with application take/release.
  iree_slim_mutex_t mutex;
  // Oldest complete immutable message, including interleaved control frames.
  loom_serve_websocket_message_t* head;
  // Last complete message, or NULL when the queue is empty.
  loom_serve_websocket_message_t* tail;
  // Incomplete data message spanning one or more frames.
  loom_serve_websocket_message_t* fragment;
  // Record receiving the current frame; may alias fragment for data frames.
  loom_serve_websocket_message_t* current;
  // Allocated records, including incomplete, queued, and taken messages.
  iree_host_size_t message_count;
  // Total allocated payload capacity across those records.
  iree_host_size_t retained_bytes;
  // Partial wire header, including its extended length and four-byte mask.
  uint8_t header[14];
  // Initialized bytes in header; zero begins the next frame.
  iree_host_size_t header_length;
  // Total header extent, established after the initial two bytes.
  iree_host_size_t header_size;
  // Validated current frame payload extent in bytes.
  iree_host_size_t frame_length;
  // Current frame bytes already unmasked, also indexing its repeating mask.
  iree_host_size_t frame_position;
  // A valid close prohibits subsequent peer messages; EOF after it is orderly.
  bool peer_closed;
};

loom_serve_websocket_limits_t loom_serve_websocket_limits_default(void) {
  return (loom_serve_websocket_limits_t){
      .message_byte_capacity = 512 * 1024,
      .retained_byte_capacity = 1024 * 1024,
      .message_capacity = 64,
  };
}

static iree_status_t loom_serve_websocket_message_allocate(
    loom_serve_websocket_t* socket, loom_serve_websocket_message_type_t type,
    loom_serve_websocket_message_t** out_message) {
  if (socket->message_count == socket->limits.message_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "WebSocket receive message capacity exhausted");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      socket->host_allocator, sizeof(**out_message), (void**)out_message));
  (*out_message)->type = type;
  ++socket->message_count;
  return iree_ok_status();
}

static iree_status_t loom_serve_websocket_message_reserve(
    loom_serve_websocket_t* socket, loom_serve_websocket_message_t** message,
    iree_host_size_t additional_bytes) {
  const iree_host_size_t length = (*message)->length;
  const iree_host_size_t capacity = (*message)->capacity;
  if (additional_bytes > socket->limits.message_byte_capacity - length) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "WebSocket message exceeds its byte capacity");
  }
  const iree_host_size_t required = length + additional_bytes;
  if (required <= capacity) {
    return iree_ok_status();
  }
  const iree_host_size_t available =
      socket->limits.retained_byte_capacity - socket->retained_bytes + capacity;
  if (required > available) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "WebSocket receive payload capacity exhausted");
  }
  // Geometric growth bounds fragmented-message copying; both allocated slack
  // and complete messages consume the same explicit aggregate byte budget.
  const iree_host_size_t growth =
      capacity
          ? capacity + iree_min(capacity,
                                socket->limits.message_byte_capacity - capacity)
          : iree_min((iree_host_size_t)256,
                     socket->limits.message_byte_capacity);
  const iree_host_size_t next_capacity =
      iree_min(available, iree_max(required, growth));
  IREE_RETURN_IF_ERROR(iree_allocator_realloc(socket->host_allocator,
                                              sizeof(**message) + next_capacity,
                                              (void**)message));
  (*message)->capacity = next_capacity;
  socket->retained_bytes += next_capacity - capacity;
  return iree_ok_status();
}

static iree_status_t loom_serve_websocket_frame_begin(
    loom_serve_websocket_t* socket) {
  const uint8_t opcode = socket->header[0] & 0x0F;
  const bool control = (opcode & 0x08) != 0;
  const uint8_t length_code = socket->header[1] & 0x7F;
  uint64_t length = length_code;
  if (length_code >= 126) {
    length = 0;
    for (iree_host_size_t i = 2; i < socket->header_size - 4; ++i) {
      length = (length << 8) | socket->header[i];
    }
  }
  if ((length_code == 126 && length < 126) ||
      (length_code == 127 && (length <= 65535 || (length >> 63)))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid masked WebSocket frame header");
  }
  if (length > socket->limits.message_byte_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "WebSocket frame exceeds message byte capacity");
  }
  if (control) {
    IREE_RETURN_IF_ERROR(loom_serve_websocket_message_allocate(
        socket, (loom_serve_websocket_message_type_t)opcode, &socket->current));
    IREE_RETURN_IF_ERROR(loom_serve_websocket_message_reserve(
        socket, &socket->current, (iree_host_size_t)length));
  } else {
    if ((opcode == 0) != (socket->fragment != NULL)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid WebSocket continuation sequence");
    }
    if (opcode != 0) {
      IREE_RETURN_IF_ERROR(loom_serve_websocket_message_allocate(
          socket, (loom_serve_websocket_message_type_t)opcode,
          &socket->fragment));
    }
    IREE_RETURN_IF_ERROR(loom_serve_websocket_message_reserve(
        socket, &socket->fragment, (iree_host_size_t)length));
    socket->current = socket->fragment;
  }
  socket->frame_length = (iree_host_size_t)length;
  socket->frame_position = 0;
  return iree_ok_status();
}

static iree_status_t loom_serve_websocket_frame_end(
    loom_serve_websocket_t* socket) {
  loom_serve_websocket_message_t* message = socket->current;
  if (socket->header[0] & 0x80) {
    if (message->type == LOOM_SERVE_WEBSOCKET_TEXT &&
        !iree_unicode_utf8_validate(iree_make_string_view(
            (const char*)message->data, message->length))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "WebSocket text is not valid UTF-8");
    }
    if (message->type == LOOM_SERVE_WEBSOCKET_CLOSE) {
      if (message->length == 1) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "WebSocket close has a truncated code");
      }
      if (message->length >= 2) {
        const uint16_t code =
            ((uint16_t)message->data[0] << 8) | message->data[1];
        const bool valid_code = (code >= 1000 && code <= 1014 && code != 1004 &&
                                 code != 1005 && code != 1006) ||
                                (code >= 3000 && code <= 4999);
        if (!valid_code ||
            !iree_unicode_utf8_validate(iree_make_string_view(
                (const char*)message->data + 2, message->length - 2))) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "WebSocket close has an invalid code or reason");
        }
      }
      socket->peer_closed = true;
    }
    if (socket->tail) {
      socket->tail->next = message;
    } else {
      socket->head = message;
    }
    socket->tail = message;
    if (message == socket->fragment) {
      socket->fragment = NULL;
    }
  }
  socket->current = NULL;
  socket->header_length = 0;
  return iree_ok_status();
}

static iree_status_t loom_serve_websocket_receive_bytes(
    loom_serve_websocket_t* socket, iree_const_byte_span_t bytes) {
  if (!bytes.data_length) {
    return socket->peer_closed
               ? iree_ok_status()
               : iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "WebSocket peer ended without a close frame");
  }
  iree_host_size_t position = 0;
  iree_status_t status = iree_ok_status();
  while (position < bytes.data_length && iree_status_is_ok(status)) {
    if (socket->peer_closed) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "WebSocket bytes follow the peer close");
      break;
    }
    if (!socket->current) {
      if (socket->header_length < 2) {
        const iree_host_size_t count =
            iree_min(2 - socket->header_length, bytes.data_length - position);
        memcpy(socket->header + socket->header_length, bytes.data + position,
               count);
        socket->header_length += count;
        position += count;
        if (socket->header_length < 2) {
          continue;
        }
        const uint8_t opcode = socket->header[0] & 0x0F;
        const uint8_t length_code = socket->header[1] & 0x7F;
        if ((socket->header[0] & 0x70) || !(socket->header[1] & 0x80) ||
            (opcode != 0 && opcode != 1 && opcode != 2 && opcode != 8 &&
             opcode != 9 && opcode != 10) ||
            ((opcode & 8) &&
             (!(socket->header[0] & 0x80) || length_code > 125))) {
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "invalid masked WebSocket frame prefix");
          break;
        }
        socket->header_size = 6 + (length_code == 126   ? 2
                                   : length_code == 127 ? 8
                                                        : 0);
      }
      const iree_host_size_t count =
          iree_min(socket->header_size - socket->header_length,
                   bytes.data_length - position);
      memcpy(socket->header + socket->header_length, bytes.data + position,
             count);
      socket->header_length += count;
      position += count;
      if (socket->header_length != socket->header_size) {
        continue;
      }
      status = loom_serve_websocket_frame_begin(socket);
    }
    if (iree_status_is_ok(status)) {
      const iree_host_size_t count =
          iree_min(socket->frame_length - socket->frame_position,
                   bytes.data_length - position);
      const uint8_t* mask = socket->header + socket->header_size - 4;
      uint8_t* target = socket->current->data + socket->current->length;
      for (iree_host_size_t i = 0; i < count; ++i) {
        target[i] =
            bytes.data[position + i] ^ mask[(socket->frame_position + i) & 3];
      }
      socket->current->length += count;
      socket->frame_position += count;
      position += count;
      if (socket->frame_position == socket->frame_length) {
        status = loom_serve_websocket_frame_end(socket);
      }
    }
  }
  return status;
}

static iree_status_t loom_serve_websocket_receive(
    void* user_data, iree_const_byte_span_t bytes) {
  loom_serve_websocket_t* socket = user_data;
  iree_slim_mutex_lock(&socket->mutex);
  iree_status_t status = loom_serve_websocket_receive_bytes(socket, bytes);
  iree_slim_mutex_unlock(&socket->mutex);
  return status;
}

static void loom_serve_websocket_release(void* user_data) {
  loom_serve_websocket_t* socket = user_data;
  loom_serve_websocket_message_t* message = socket->head;
  while (message) {
    loom_serve_websocket_message_t* next = message->next;
    iree_allocator_free(socket->host_allocator, message);
    message = next;
  }
  if (socket->current != socket->fragment) {
    iree_allocator_free(socket->host_allocator, socket->current);
  }
  iree_allocator_free(socket->host_allocator, socket->fragment);
  iree_slim_mutex_deinitialize(&socket->mutex);
  iree_allocator_free(socket->host_allocator, socket);
}

iree_status_t loom_serve_websocket_upgrade(
    loom_serve_http_connection_t* connection,
    loom_serve_websocket_limits_t limits, iree_allocator_t host_allocator,
    loom_serve_websocket_t** out_socket) {
  *out_socket = NULL;
  if (limits.message_byte_capacity < 125 ||
      limits.message_byte_capacity >
          IREE_HOST_SIZE_MAX - sizeof(loom_serve_websocket_message_t) ||
      limits.retained_byte_capacity < limits.message_byte_capacity ||
      limits.message_capacity < 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "WebSocket limits must hold a message and an "
                            "interleaved control record");
  }
  loom_serve_websocket_t* socket = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*socket), (void**)&socket));
  socket->host_allocator = host_allocator;
  socket->connection = connection;
  socket->limits = limits;
  iree_slim_mutex_initialize(&socket->mutex);
  loom_serve_http_connection_upgrade(
      connection,
      (loom_serve_http_protocol_t){loom_serve_websocket_receive,
                                   loom_serve_websocket_release, socket});
  *out_socket = socket;
  return iree_ok_status();
}

loom_serve_websocket_message_t* loom_serve_websocket_take(
    loom_serve_websocket_t* socket,
    loom_serve_websocket_message_type_t* out_type,
    iree_const_byte_span_t* out_payload) {
  iree_slim_mutex_lock(&socket->mutex);
  loom_serve_websocket_message_t* message = socket->head;
  if (message) {
    socket->head = message->next;
    if (!socket->head) {
      socket->tail = NULL;
    }
    message->next = NULL;
  }
  *out_type = message ? message->type : 0;
  *out_payload = message
                     ? iree_make_const_byte_span(message->data, message->length)
                     : iree_const_byte_span_empty();
  iree_slim_mutex_unlock(&socket->mutex);
  return message;
}

void loom_serve_websocket_message_release(
    loom_serve_websocket_t* socket, loom_serve_websocket_message_t* message) {
  iree_slim_mutex_lock(&socket->mutex);
  const iree_host_size_t capacity = message->capacity;
  iree_allocator_free(socket->host_allocator, message);
  --socket->message_count;
  socket->retained_bytes -= capacity;
  iree_slim_mutex_unlock(&socket->mutex);
}

bool loom_serve_websocket_peer_closed(loom_serve_websocket_t* socket) {
  iree_slim_mutex_lock(&socket->mutex);
  const bool closed = socket->peer_closed;
  iree_slim_mutex_unlock(&socket->mutex);
  return closed;
}

typedef struct loom_serve_websocket_send_t {
  // Final server frame header, without a masking key.
  uint8_t header[10];
  // Initialized header bytes preceding the payload.
  iree_host_size_t header_length;
  // Borrowed payload copied by the synchronous transport writer.
  iree_const_byte_span_t payload;
} loom_serve_websocket_send_t;

static iree_status_t loom_serve_websocket_write(void* user_data,
                                                iree_byte_span_t target) {
  const loom_serve_websocket_send_t* frame = user_data;
  memcpy(target.data, frame->header, frame->header_length);
  if (frame->payload.data_length) {
    memcpy(target.data + frame->header_length, frame->payload.data,
           frame->payload.data_length);
  }
  return iree_ok_status();
}

iree_status_t loom_serve_websocket_send(
    loom_serve_websocket_t* socket, loom_serve_websocket_message_type_t type,
    iree_const_byte_span_t payload) {
  if (payload.data_length > socket->limits.message_byte_capacity ||
      ((type & 8) && payload.data_length > 125)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "WebSocket output exceeds message byte capacity");
  }
  loom_serve_websocket_send_t frame = {
      .header = {0x80 | (uint8_t)type, (uint8_t)payload.data_length},
      .header_length = 2,
      .payload = payload,
  };
  if (payload.data_length >= 126) {
    const iree_host_size_t width = payload.data_length <= 65535 ? 2 : 8;
    frame.header[1] = width == 2 ? 126 : 127;
    frame.header_length += width;
    for (iree_host_size_t i = 0; i < width; ++i) {
      frame.header[2 + i] =
          (uint8_t)((uint64_t)payload.data_length >> (8 * (width - i - 1)));
    }
  }
  return loom_serve_http_connection_send_generated(
      socket->connection,
      (iree_net_send_prefix_t){frame.header_length + payload.data_length,
                               loom_serve_websocket_write, &frame});
}
