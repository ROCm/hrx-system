// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REALTIME_SESSION_H_
#define LOOM_SERVE_SERVER_REALTIME_SESSION_H_

#include "iree/task/executor.h"
#include "loom/serve/server/realtime/binding.h"
#include "loom/serve/server/realtime/settings.h"
#include "loom/serve/transport/websocket/connection.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_realtime_session_t loom_serve_realtime_session_t;

// Allocates a bounded native conversation on an already upgraded connection.
// Success transfers the HTTP claim; failure leaves it caller-owned. Profile,
// registry, executor, server, and wake target outlive final session retirement.
iree_status_t loom_serve_realtime_session_allocate(
    uint64_t id, const loom_serve_realtime_profile_t* profile,
    loom_serve_realtime_binding_t text,
    loom_serve_realtime_binding_t recognition,
    loom_serve_realtime_binding_t speech, iree_task_executor_t* executor,
    loom_serve_registry_t* registry, loom_serve_http_server_t* server,
    loom_serve_http_connection_t* connection, loom_serve_websocket_t* socket,
    loom_serve_wake_callback_t observer, iree_allocator_t host_allocator,
    loom_serve_realtime_session_t** out_session);

// Serialized owner turn: bounded wire commands, CPU collection, native stream
// admission/consumption, checkpoint effects, and transport publication. Peer
// errors retire this session. A failed native cleanup propagates to shared
// serving so original model teardown can retain/release its resources safely.
iree_status_t loom_serve_realtime_session_advance(
    loom_serve_realtime_session_t* session, bool stopping, bool* out_progress,
    bool* out_complete);
// Requires advance to report complete. No CPU borrower or logical pin remains.
void loom_serve_realtime_session_free(loom_serve_realtime_session_t* session);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_REALTIME_SESSION_H_
