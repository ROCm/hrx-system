// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_TRANSPORT_HTTP_LISTENER_H_
#define LOOM_SERVE_TRANSPORT_HTTP_LISTENER_H_

#include "iree/async/address.h"
#include "iree/net/carrier/stream/carrier.h"
#include "loom/serve/transport/http/server.h"

#ifdef __cplusplus
extern "C" {
#endif

// Independent socket binding feeding the server's shared connection budget.
// Owns accept/cancellation storage, never the accepted HTTP connections. At
// shared capacity it retains one unadmitted carrier and waits asynchronously;
// further peers stay in the configured kernel backlog.
typedef struct loom_serve_http_listener_t loom_serve_http_listener_t;

typedef struct loom_serve_http_listener_options_t {
  // Numeric IPv4 host:port, [IPv6]:port, or unix:path; borrowed during create.
  iree_string_view_t address;
  // Dense entry in the server's shared proactor pool, not a private I/O thread.
  iree_host_size_t proactor_index;
  // Kernel listen backlog, independent of the shared HTTP connection budget.
  int backlog;
  // Bytes in each dedicated per-connection receive buffer.
  iree_host_size_t receive_buffer_size;
  // Receive buffers per connection; power of two for provided-buffer backends.
  iree_host_size_t receive_buffer_count;
  // Per-connection bounded send admission and generated-prefix storage.
  iree_net_stream_carrier_options_t send;
} loom_serve_http_listener_options_t;

// Loopback port 8080, backlog 64, four 16 KiB receive buffers and eight sends.
loom_serve_http_listener_options_t loom_serve_http_listener_options_default(
    void);

// Binds and begins accepting on shared I/O infrastructure. The server outlives
// this listener. Multiple listeners may target the same server. Bind failure
// leaves other listeners and accepted connections untouched.
//
// Filesystem socket paths belong exclusively to the listener from successful
// bind through destruction. Existing paths are never unlinked to make bind
// succeed. Destruction removes only a path this listener successfully bound;
// abstract Unix addresses have no filesystem ownership.
iree_status_t loom_serve_http_listener_create(
    loom_serve_http_server_t* server,
    const loom_serve_http_listener_options_t* options,
    loom_serve_http_listener_t** out_listener, iree_allocator_t allocator);

// Borrowed bound address, including an ephemeral port selected during bind.
const iree_async_address_t* loom_serve_http_listener_address(
    const loom_serve_http_listener_t* listener);

// True while a native capacity wait retains this listener's unadmitted peer.
// Notification publishes changes; the snapshot takes the listener mutex and
// borrows no peer state. A stopped listener never reports capacity pressure.
bool loom_serve_http_listener_is_waiting_for_capacity(
    loom_serve_http_listener_t* listener);

// Closes admission without waiting or touching accepted connections. Completion
// joins accept cancellation or the capacity wait's private stop edge on the
// shared poll owner, releasing any unadmitted peer. The shared I/O owner must
// remain capable of accepting the shutdown handoff.
void loom_serve_http_listener_request_stop(
    loom_serve_http_listener_t* listener);

// Readiness for nonblocking retirement. Notification is advisory; always check
// is_stopped after preparing a wait. Neither query consumes listener failures.
bool loom_serve_http_listener_is_stopped(loom_serve_http_listener_t* listener);
iree_notification_t* loom_serve_http_listener_notification(
    loom_serve_http_listener_t* listener);

// Stops and joins only this listener, then returns any listener/cleanup
// failure. Accepted HTTP streams remain live on the server's retained pool.
// Called on the outer application owner, never on a task or proactor callback.
// NULL is accepted. A prior is_stopped observation makes the join immediate.
iree_status_t loom_serve_http_listener_destroy(
    loom_serve_http_listener_t* listener);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_TRANSPORT_HTTP_LISTENER_H_
