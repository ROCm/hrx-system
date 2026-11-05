// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_FINITE_HTTP_H_
#define LOOM_SERVE_SERVICE_FINITE_HTTP_H_

#include "loom/serve/service/finite/consumer.h"
#include "loom/serve/transport/http/server.h"

#ifdef __cplusplus
extern "C" {
#endif

// Moves the connection claim into a receipt. Its request body remains borrowed
// through consumer.end, including peer failure during CPU preparation.
loom_serve_finite_consumer_t loom_serve_finite_http_consumer(
    loom_serve_http_connection_t* connection);

// Consumes an unadmitted connection claim with an encoded request rejection.
// scratch is reusable caller-owned storage, reset before encoding.
iree_status_t loom_serve_finite_http_reject(
    loom_serve_http_connection_t* connection, const char* code,
    const char* type, iree_string_view_t message,
    iree_string_builder_t* scratch, bool* out_delivered);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_FINITE_HTTP_H_
