// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REALTIME_SERVER_H_
#define LOOM_SERVE_SERVER_REALTIME_SERVER_H_

#include "iree/task/executor.h"
#include "loom/serve/server/realtime/settings.h"
#include "loom/serve/server/registry.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_realtime_options_t {
  // Number of independently selectable assistant compositions.
  iree_host_size_t profile_count;
  // Borrowed immutable profiles and strings, live through router retirement.
  const loom_serve_realtime_profile_t* profiles;
  // Exact browser-origin allowlist; absence of Origin permits native clients.
  iree_string_view_list_t allowed_origins;
} loom_serve_realtime_options_t;

typedef struct loom_serve_realtime_server_t loom_serve_realtime_server_t;

// Serialized native conversation owner beside ordinary model admission. Uses
// the existing executor/registry/transport; construction performs no model I/O.
iree_status_t loom_serve_realtime_server_allocate(
    iree_task_executor_t* executor, loom_serve_http_server_t* transport,
    loom_serve_registry_t* registry, loom_serve_realtime_options_t options,
    loom_serve_wake_callback_t observer, iree_allocator_t host_allocator,
    loom_serve_realtime_server_t** out_server);
// Requires every session and accepted CPU/checkpoint effect to have retired.
void loom_serve_realtime_server_free(loom_serve_realtime_server_t* server);

bool loom_serve_realtime_server_handles(iree_string_view_t target);
// Consumes one HTTP claim. allow_wait reflects the router's shared long-lived
// claim budget, leaving control capacity across telemetry and realtime peers.
iree_status_t loom_serve_realtime_server_accept(
    loom_serve_realtime_server_t* server,
    loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request, bool allow_wait,
    iree_string_builder_t* scratch);
iree_status_t loom_serve_realtime_server_advance(
    loom_serve_realtime_server_t* server, bool stopping, bool* out_progress);
// Includes disconnected sessions whose accepted work still borrows storage.
iree_host_size_t loom_serve_realtime_server_claims(
    const loom_serve_realtime_server_t* server);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_REALTIME_SERVER_H_
