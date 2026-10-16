// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_HTTP_H_
#define LOOM_SERVE_SERVICE_TEXT_HTTP_H_

#include "loom/serve/service/text/consumer.h"
#include "loom/serve/service/text/telemetry.h"
#include "loom/serve/transport/http/server.h"

#ifdef __cplusplus
extern "C" {
#endif

// Takes the connection claim only on success. The receipt preserves input
// storage while queued, then allocates a staging packet on row admission.
// Telemetry outlives the receipt and is touched only by owner-side rejection.
iree_status_t loom_serve_text_http_consumer_allocate(
    loom_serve_http_connection_t* connection,
    loom_serve_text_telemetry_t* telemetry, loom_serve_text_receipt_t receipt,
    iree_allocator_t host_allocator, loom_serve_text_consumer_t* out_consumer);

// Ends a rejected protocol receipt, consumes failure and relinquishes its
// connection claim. Independent scratch permits rejection during CPU output.
iree_status_t loom_serve_text_http_reject(
    loom_serve_text_telemetry_t* telemetry,
    loom_serve_http_connection_t* connection, loom_serve_text_receipt_t receipt,
    int code, const char* reason, iree_status_t failure,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_HTTP_H_
