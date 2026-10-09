// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_HTTP_SERVER_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_HTTP_SERVER_H_

#include "experimental/loom_serve/http/request.h"
#include "experimental/loom_serve/telemetry/stream.h"
#include "iree/async/address.h"
#include "iree/base/threading/notification.h"

#ifdef __cplusplus
extern "C" {
#endif

// Local-only HTTP byte transport. One standard proactor thread services raw TCP
// carriers independently of the application's serialized scheduler turns. Each
// connection carries one request and a close-delimited response. This layer
// knows neither model rows nor chat/SSE semantics.
typedef struct loom_serve_http_server_t loom_serve_http_server_t;
typedef struct loom_serve_http_connection_t loom_serve_http_connection_t;

// Transport budgets are independent of active model rows and queued admission.
typedef struct loom_serve_http_server_options_t {
  // Loopback TCP port; zero selects an ephemeral port.
  uint16_t port;
  // Maximum simultaneous connections, including queued and draining requests.
  iree_host_size_t connection_capacity;
  // Per-connection framing limits; completed bytes remain owned until release.
  loom_serve_http_request_limits_t request_limits;
} loom_serve_http_server_options_t;

// Defaults to port 8080, 64 connections, 16 KiB headers, 64 headers and 8 MiB
// bodies. Body storage grows on demand within these explicit bounds.
loom_serve_http_server_options_t loom_serve_http_server_options_default(void);

// Binds loopback and starts network I/O. The executable calls
// iree_async_signal_block_default before creating any threads if it wants
// SIGINT/SIGTERM delivery here. The server owns those signal subscriptions.
iree_status_t loom_serve_http_server_create(
    const loom_serve_http_server_options_t* options,
    loom_serve_http_server_t** out_server, iree_allocator_t host_allocator);

// Stops admission, cancels/drains accepted I/O and joins the polling thread.
// The application first relinquishes all connection/request views. Returns
// listener failures; individual peer failures are diagnosed and retired
// locally. Null is accepted. A fatal proactor failure aborts this experimental
// process: accepted I/O cannot be safely reclaimed after its polling owner has
// failed.
iree_status_t loom_serve_http_server_destroy(loom_serve_http_server_t* server);

// Borrowed bound address and notification, valid until server destruction.
const iree_async_address_t* loom_serve_http_server_address(
    const loom_serve_http_server_t* server);
iree_notification_t* loom_serve_http_server_notification(
    loom_serve_http_server_t* server);

// Borrowed diagnostic stream, valid until server destruction. Readers use
// independent cursors and never acquire the transport's mutex. This stream
// includes routing failures, connection rejection and retired peer failures.
const loom_serve_telemetry_stream_t* loom_serve_http_server_diagnostics(
    const loom_serve_http_server_t* server);

// Copies a borrowed error into the bounded diagnostic history, without
// formatting, allocation or I/O. The transport mutex serializes network and
// application producers; slow readers never delay them. Status ownership is
// unchanged, including when callers also propagate the error normally.
void loom_serve_http_server_record_error(loom_serve_http_server_t* server,
                                         iree_status_t status);

// Single application wakeup target. The callback only publishes a task wake;
// it never calls back into the transport or waits. Registration immediately
// wakes the observer so requests published before registration are consumed.
// Replacing with an empty callback joins all prior callback accesses. The
// observer's storage remains live until replacement or server destruction.
typedef struct loom_serve_http_observer_t {
  // Nonblocking wake callback invoked on network or application producers.
  void (*wake)(void* user_data);
  // Borrowed callback state protected by registration lifetime.
  void* user_data;
} loom_serve_http_observer_t;

void loom_serve_http_server_set_observer(loom_serve_http_server_t* server,
                                         loom_serve_http_observer_t observer);

// True after a shutdown signal or terminal listener failure.
bool loom_serve_http_server_is_stopping(loom_serve_http_server_t* server);

// Collects retired connections and claims one completed request, or returns
// null. The request and connection are borrowed until finish or abort. The
// application owns each claimed connection through serialized scheduler turns.
loom_serve_http_connection_t* loom_serve_http_server_take_request(
    loom_serve_http_server_t* server,
    const loom_serve_http_request_t** out_request);

// A failed peer cannot accept more output. Cancellation of model work belongs
// to the application and occurs at its safe stage boundary.
bool loom_serve_http_connection_failed(
    loom_serve_http_connection_t* connection);

// True when a send slot is available. Completions post the server notification;
// a slow client therefore pauses only the application row using that peer.
bool loom_serve_http_connection_can_send(
    loom_serve_http_connection_t* connection);

// Copies bytes during admission; caller storage is not borrowed afterward.
// The single application producer checks can_send before calling. Failure
// accepts no bytes and must be handled by the application.
iree_status_t loom_serve_http_connection_send(
    loom_serve_http_connection_t* connection, iree_string_view_t bytes);

// Relinquishes the connection and all request views. Finish drains every
// accepted send before closing; abort cancels outstanding network work. Neither
// waits for the network. Both invalidate the application's borrowed pointers.
void loom_serve_http_connection_finish(
    loom_serve_http_connection_t* connection);
void loom_serve_http_connection_abort(loom_serve_http_connection_t* connection);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_HTTP_SERVER_H_
