// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_TRANSPORT_WEBSOCKET_HANDSHAKE_H_
#define LOOM_SERVE_TRANSPORT_WEBSOCKET_HANDSHAKE_H_

#include "loom/serve/transport/http/request.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validates an RFC 6455 server upgrade and appends its 101 response. HTTP has
// already established HTTP/1.1 framing and unique headers. A present browser
// Origin must exactly match an allowed origin; absence permits non-browser
// clients. An empty allowlist therefore disables browser-origin upgrades.
// Selects |subprotocol| only when nonempty and present in the client's offer.
// No compression or extensions are negotiated. This does not install a
// receiver or send bytes.
iree_status_t loom_serve_websocket_handshake(
    const loom_serve_http_request_t* request,
    iree_string_view_list_t allowed_origins, iree_string_view_t subprotocol,
    iree_string_builder_t* response);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_TRANSPORT_WEBSOCKET_HANDSHAKE_H_
