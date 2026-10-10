// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_HTTP_SERVER_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_HTTP_SERVER_H_

#include "experimental/loom_serve/http/request.h"
#include "experimental/loom_serve/telemetry/stream.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/notification.h"
#include "iree/net/carrier.h"

#ifdef __cplusplus
extern "C" {
#endif

// HTTP connection owner independent of socket listeners and process signals.
// Raw ordered carriers deliver bytes using the shared proactor pool while the
// application claims requests on serialized scheduler turns. Each connection
// carries one request and a close-delimited response. This layer knows neither
// model rows nor chat/SSE semantics and creates no private workers.
typedef struct loom_serve_http_server_t loom_serve_http_server_t;
typedef struct loom_serve_http_connection_t loom_serve_http_connection_t;

// Transport budgets are independent of active model rows and queued admission.
typedef struct loom_serve_http_server_options_t {
  // Maximum simultaneous connections, including queued and draining requests.
  iree_host_size_t connection_capacity;
  // Per-connection framing limits; completed bytes remain owned until release.
  loom_serve_http_request_limits_t request_limits;
} loom_serve_http_server_options_t;

// Defaults to 64 connections, 16 KiB headers, 64 headers and 8 MiB
// bodies. Body storage grows on demand within these explicit bounds.
loom_serve_http_server_options_t loom_serve_http_server_options_default(void);

// Creates a connection owner without binding listeners or registering signals.
// Retains |proactor_pool| to keep shared I/O progress alive through connection
// retirement, including after their originating listeners have stopped.
iree_status_t loom_serve_http_server_create(
    iree_async_proactor_pool_t* proactor_pool,
    const loom_serve_http_server_options_t* options,
    loom_serve_http_server_t** out_server, iree_allocator_t host_allocator);

// Cancels/drains accepted I/O and releases shared progress ownership. All
// listeners have retired and the application has relinquished all request
// views. This outer application join never runs on a task or proactor callback.
// Individual peer failures remain in diagnostic history. NULL is accepted.
void loom_serve_http_server_destroy(loom_serve_http_server_t* server);

// Borrowed progress infrastructure and notification, live until destruction.
iree_async_proactor_pool_t* loom_serve_http_server_proactor_pool(
    loom_serve_http_server_t* server);
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

// Stops request admission and wakes the application to drain its services.
// Does not cancel active connections or stop shared I/O infrastructure.
void loom_serve_http_server_request_stop(loom_serve_http_server_t* server);

// True after request_stop, independent of individual listener lifetimes.
bool loom_serve_http_server_is_stopping(loom_serve_http_server_t* server);

// Transfers a CREATED reliable ordered carrier to HTTP connection ownership.
// The carrier is either activated or released if admission is closed/full or
// setup fails. Per-peer failures are recorded without stopping other peers.
// Its progress comes from this server's pool. Callers join all accept callbacks
// before destroying the server; concurrent accepts are permitted.
void loom_serve_http_server_accept(loom_serve_http_server_t* server,
                                   iree_net_carrier_t* carrier);

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
