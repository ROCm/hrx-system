// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_TEXT_RESPONSE_H_
#define EXPERIMENTAL_LOOM_SERVE_TEXT_RESPONSE_H_

#include "experimental/loom_serve/text/telemetry.h"
#include "loom/serve/transport/http/server.h"

#ifdef __cplusplus
extern "C" {
#endif

// Appends the text-service JSON error object and consumes failure. The caller
// supplies any HTTP/SSE framing and retains output. Serialization failures
// propagate independently of the translated request failure.
iree_status_t loom_serve_text_response_error(iree_status_t failure,
                                             iree_string_builder_t* output);

// Ends a rejected request receipt, consumes failure and relinquishes the HTTP
// claim after copying the response or aborting it. Uses independent temporary
// scratch so admission rejection can overlap CPU output serialization.
iree_status_t loom_serve_text_response_reject(
    loom_serve_text_telemetry_t* telemetry,
    loom_serve_http_connection_t* connection, loom_serve_text_receipt_t receipt,
    int code, const char* reason, iree_status_t failure,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_TEXT_RESPONSE_H_
