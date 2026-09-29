// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_HTTP_SERVER_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_HTTP_SERVER_H_

#include "experimental/loom_serve/http_request.h"
#include "iree/async/address.h"
#include "iree/base/threading/notification.h"

#ifdef __cplusplus
extern "C" {
#endif

// Local-only HTTP byte transport. One standard proactor thread services raw TCP
// carriers while a single application owner may block on model stages. Each
// connection carries one request and a close-delimited response. This layer
// knows neither model rows nor chat/SSE semantics.
typedef struct loom_serve_http_server_t loom_serve_http_server_t;
typedef struct loom_serve_http_connection_t loom_serve_http_connection_t;

// Binds loopback (port zero selects an ephemeral port) and starts network I/O.
// Sixteen connections, 16 KiB headers, 64 headers and 1 MiB bodies are bounded
// independently of model capacity. The executable calls
// iree_async_signal_block_default before creating any threads if it wants
// SIGINT/SIGTERM delivery here. The server owns those signal subscriptions.
iree_status_t loom_serve_http_server_create(
    uint16_t port, iree_allocator_t host_allocator,
    loom_serve_http_server_t** out_server);

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

// True after a shutdown signal or terminal listener failure.
bool loom_serve_http_server_is_stopping(loom_serve_http_server_t* server);

// Collects retired connections and claims one completed request, or returns
// null. The request and connection are borrowed until finish or abort. The
// application owns each claimed connection on its one model/scheduler thread.
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
