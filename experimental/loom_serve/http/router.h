// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_HTTP_ROUTER_H_
#define EXPERIMENTAL_LOOM_SERVE_HTTP_ROUTER_H_

#include "experimental/loom_serve/http/server.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_serve_http_service_flag_bits_e {
  // Another service has accepted work or maintenance that may release shared
  // capacity. Admission can queue instead of rejecting an idle-capacity stall.
  LOOM_SERVE_HTTP_SERVICE_FLAG_SHARED_ACTIVITY = 1u << 0,
} loom_serve_http_service_flag_bits_t;
typedef uint32_t loom_serve_http_service_flags_t;

// One named service borrowing a model on the application owner thread. Network
// progress is independent; these callbacks never run concurrently. Names are
// deployment identifiers, not source model identities or checkpoint paths.
typedef struct loom_serve_http_service_t {
  // Nonempty visible ASCII name, unique and borrowed through run return. This
  // also fits the checkpoint selector's HTTP header representation.
  iree_string_view_t name;
  // Borrowed service state, outliving the router and all accepted requests.
  void* self;
  // Whether live requests or pending maintenance may release shared capacity.
  // Queued requests alone are not activity: mutually blocked queues cannot
  // make progress by waiting on one another. Called by the application owner.
  bool (*is_active)(void* self);
  // Takes the connection claim on both success and failure. Invalid requests
  // produce an HTTP rejection and success; a terminal failure ends the owner
  // loop. Request views remain valid until this service finishes or aborts.
  iree_status_t (*accept)(void* self, loom_serve_http_connection_t* connection,
                          const loom_serve_http_request_t* request,
                          loom_serve_http_service_flags_t flags);
  // Advances at most one bounded model unit and reports actual progress. A
  // turn can retry admission after another service released shared capacity.
  // False means awaiting another service or network event, never busy polling.
  iree_status_t (*advance)(void* self, loom_serve_http_service_flags_t flags,
                           bool* out_progress);
} loom_serve_http_service_t;

// Routes JSON request bodies by their model field and checkpoint operations by
// X-Loom-Model. A checkpoint request may omit its selector with exactly one
// service. GET /v1/models lists deployment names; GET /healthz is model-free.
// At most request_capacity claims are accepted before advancing each service
// once, with rotating service priority. Return leaves outstanding claims owned
// by the services; the caller destroys them before the borrowed transport.
// Callback failures terminate the loop without destroying any shared device.
iree_status_t loom_serve_http_router_run(
    loom_serve_http_server_t* server, iree_host_size_t service_count,
    const loom_serve_http_service_t* services,
    iree_host_size_t request_capacity, iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_HTTP_ROUTER_H_
