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

// Delivery operations borrowing an already claimed connection. The caller
// transfers the claim only after native admission accepts it; pressure and
// failure leave the connection with the frontend. No allocation or body copy.
loom_serve_text_input_count_consumer_t
loom_serve_text_input_count_http_consumer(
    loom_serve_http_connection_t* connection);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_INPUT_COUNT_HTTP_H_
