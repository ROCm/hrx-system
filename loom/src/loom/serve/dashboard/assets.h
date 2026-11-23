// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_DASHBOARD_ASSETS_H_
#define LOOM_SERVE_DASHBOARD_ASSETS_H_

#include "loom/serve/transport/http/request.h"
#include "loom/serve/transport/http/server.h"

#ifdef __cplusplus
extern "C" {
#endif

// Consumes a GET claim for a redirect or an exact embedded name. Returns false
// when another route owns the path. Clients excluding gzip receive 406.
bool loom_serve_dashboard_try_send(loom_serve_http_server_t* server,
                                   loom_serve_http_connection_t* connection,
                                   const loom_serve_http_request_t* request);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_DASHBOARD_ASSETS_H_
