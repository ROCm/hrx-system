// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_TRANSPORT_WEBSOCKET_CONNECTION_H_
#define LOOM_SERVE_TRANSPORT_WEBSOCKET_CONNECTION_H_

#include "loom/serve/transport/http/server.h"

#ifdef __cplusplus
extern "C" {
#endif

// Complete messages, including control frames interleaved with fragments.
typedef enum loom_serve_websocket_message_type_e {
  LOOM_SERVE_WEBSOCKET_TEXT = 1,
  LOOM_SERVE_WEBSOCKET_BINARY = 2,
  LOOM_SERVE_WEBSOCKET_CLOSE = 8,
  LOOM_SERVE_WEBSOCKET_PING = 9,
  LOOM_SERVE_WEBSOCKET_PONG = 10,
} loom_serve_websocket_message_type_t;

typedef struct loom_serve_websocket_limits_t {
  // Maximum complete data message, including all of its fragments, in bytes.
  iree_host_size_t message_byte_capacity;
  // Maximum allocated payload bytes across incomplete and queued messages.
  iree_host_size_t retained_byte_capacity;
  // Maximum allocated message records, including incomplete data and control.
  iree_host_size_t message_capacity;
} loom_serve_websocket_limits_t;

// Defaults to 512 KiB per message, 1 MiB retained payload, and 64 records.
// Storage is allocated on demand; budget exhaustion fails this peer explicitly.
loom_serve_websocket_limits_t loom_serve_websocket_limits_default(void);

typedef struct loom_serve_websocket_t loom_serve_websocket_t;

// Installs a bounded RFC 6455 server receiver on a claimed HTTP connection.
// The caller has validated its handshake and sends 101 only after this call.
// On success, HTTP owns all receiver storage until application and I/O retire;
// *out_socket is borrowed until finish/abort. On failure nothing is
// transferred. HTTP notifications wake the application for input, EOF, failure,
// and output credit. No model/JSON work or application waits occur in receive
// callbacks.
iree_status_t loom_serve_websocket_upgrade(
    loom_serve_http_connection_t* connection,
    loom_serve_websocket_limits_t limits, iree_allocator_t host_allocator,
    loom_serve_websocket_t** out_socket);

// Borrows the oldest complete message until consume or connection release.
// Text and close reasons are validated UTF-8. Empty messages still return true.
// The serialized application owner may keep a view across turns or accepted CPU
// work; later receives cannot move or mutate it. Control frames are published
// immediately even during a fragmented data message. The owner sends pong/close
// replies through its ordinary output stream, preserving one send producer.
bool loom_serve_websocket_peek(loom_serve_websocket_t* socket,
                               loom_serve_websocket_message_type_t* out_type,
                               iree_const_byte_span_t* out_payload);
void loom_serve_websocket_consume(loom_serve_websocket_t* socket);

// True once a valid peer close was received, independently of queued messages.
bool loom_serve_websocket_peer_closed(loom_serve_websocket_t* socket);

// Sends one final, unmasked server message directly into admitted transport
// storage. Payload is borrowed only during this call. The application checks
// its HTTP connection's send credit; transport retains accepted output through
// finish. Messages must fit message_byte_capacity; controls fit 125 bytes.
iree_status_t loom_serve_websocket_send(
    loom_serve_websocket_t* socket, loom_serve_websocket_message_type_t type,
    iree_const_byte_span_t payload);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_TRANSPORT_WEBSOCKET_CONNECTION_H_
