// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_CHECKPOINT_HTTP_H_
#define LOOM_SERVE_SERVICE_TEXT_CHECKPOINT_HTTP_H_

#include "loom/serve/service/text/checkpoint_control.h"
#include "loom/serve/transport/http/server.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parses the checkpoint route, method, body, and identifiers without accessing
// a model. The validated output borrows the connection's immutable request.
iree_status_t loom_serve_text_checkpoint_http_parse(
    const loom_serve_http_request_t* request,
    loom_serve_text_checkpoint_request_t* out_request);

// Takes a connection claim without allocating. The service owns its validated
// request views and this consumer until terminal delivery or abort.
loom_serve_text_checkpoint_consumer_t loom_serve_text_checkpoint_http_consumer(
    loom_serve_http_connection_t* connection);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_CHECKPOINT_HTTP_H_
