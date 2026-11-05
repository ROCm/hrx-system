// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_HTTP_LIVE_H_
#define LOOM_SERVE_SERVER_HTTP_LIVE_H_

#include "loom/serve/server/registry.h"
#include "loom/serve/server/resources.h"
#include "loom/serve/transport/http/server.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_http_live_t loom_serve_http_live_t;

// Allocates bounded reader metadata, shared current-state copies, and one 16
// KiB formatting buffer. No timer runs without readers. Server, registry,
// resources, and wake target remain borrowed until shutdown advance
// retires the timer and every reader.
iree_status_t loom_serve_http_live_allocate(
    loom_serve_http_server_t* server, loom_serve_registry_t* registry,
    const loom_serve_resources_t* resources,
    loom_runtime_wake_callback_t observer, iree_allocator_t host_allocator,
    loom_serve_http_live_t** out_live);

// Upgrades a validated route within the router's shared claim bound, consuming
// the connection on every path. Browser admission uses the HTTP origin policy;
// loom.observe.v1 is negotiated when offered. Native clients may omit Origin.
// Allocation/handshake failures affect only this peer and are recorded through
// the HTTP diagnostic stream.
void loom_serve_http_live_accept(loom_serve_http_live_t* live,
                                 loom_serve_http_connection_t* connection,
                                 const loom_serve_http_request_t* request);

// Advances at most quota sample pages/reader messages, after ordinary service
// work.
// One application-acknowledged page may be outstanding per reader. Timer ticks
// sample shared state at four hertz. Changed state starts a sweep; otherwise a
// small heartbeat settles rates at one hertz for five seconds, then one/minute.
// No unchanged snapshot traffic or per-reader service sampling. Only timer
// failure propagates; observation failures cannot fail model execution.
// Reader visits are bounded by the current reader count, not HTTP capacity.
iree_status_t loom_serve_http_live_advance(loom_serve_http_live_t* live,
                                           bool stopping,
                                           iree_host_size_t quota,
                                           bool* out_progress);

// Claimed observer connections; shares the router's control-reserving budget.
iree_host_size_t loom_serve_http_live_count(const loom_serve_http_live_t* live);
// Reader claims or accepted timer callbacks still retain this owner.
bool loom_serve_http_live_is_pending(loom_serve_http_live_t* live);
// Frees only after stopping advance has returned all accepted ownership.
void loom_serve_http_live_free(loom_serve_http_live_t* live);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVER_HTTP_LIVE_H_
