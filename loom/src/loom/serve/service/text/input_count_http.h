// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_INPUT_COUNT_HTTP_H_
#define LOOM_SERVE_SERVICE_TEXT_INPUT_COUNT_HTTP_H_

#include "loom/serve/service/text/input_count.h"
#include "loom/serve/transport/http/server.h"

#ifdef __cplusplus
extern "C" {
#endif

// Takes the HTTP claim, including immediate rejection of stateful headers or
// a full queue. The caller has validated the route and JSON content type.
// Accepted input borrows the connection directly; no receipt allocation or
// body copy is needed for the native counting queue.
iree_status_t loom_serve_text_input_count_http_enqueue(
    loom_serve_text_input_count_t* queue,
    loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request, loom_serve_text_receipt_t receipt,
    loom_serve_text_telemetry_t* telemetry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_INPUT_COUNT_HTTP_H_
