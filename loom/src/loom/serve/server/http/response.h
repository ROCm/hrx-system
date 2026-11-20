// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_HTTP_RESPONSE_H_
#define LOOM_SERVE_SERVER_HTTP_RESPONSE_H_

#include "loom/serve/transport/http/server.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns an owned model lookup error naming the selector and the discovery
// endpoint. The selector may borrow the scratch buffer used for the response.
iree_status_t loom_serve_http_model_not_found(iree_string_view_t name);

// Copies a complete management response into transport and relinquishes the
// claim. Peer/send failures are recorded and abort only this connection. No
// response or model storage remains borrowed by network completion.
void loom_serve_http_response_send(loom_serve_http_server_t* server,
                                   loom_serve_http_connection_t* connection,
                                   iree_string_view_t response);

// Formats a JSON routing error in scratch and relinquishes the claim. Only
// response allocation failure propagates to the serving owner.
iree_status_t loom_serve_http_response_reject(
    loom_serve_http_server_t* server, loom_serve_http_connection_t* connection,
    int code, const char* reason, iree_string_view_t message,
    iree_string_builder_t* scratch);

// Consumes an owned request failure, records it and sends the corresponding
// HTTP error. The operation failure does not terminate shared serving.
iree_status_t loom_serve_http_response_error(
    loom_serve_http_server_t* server, loom_serve_http_connection_t* connection,
    iree_status_t status, iree_string_builder_t* scratch);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_HTTP_RESPONSE_H_
