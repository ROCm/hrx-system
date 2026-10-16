// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_SPEECH_HTTP_H_
#define LOOM_SERVE_SERVICE_SPEECH_HTTP_H_

#include "loom/serve/service/speech/consumer.h"
#include "loom/serve/transport/http/server.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates a bounded PCM16 HTTP consumer. On success it owns the connection
// claim through end; failure leaves the claim with the caller. Source output
// must match the standard 24 kHz mono PCM response geometry.
iree_status_t loom_serve_speech_http_consumer_allocate(
    loom_serve_http_connection_t* connection, loom_serve_speech_format_t format,
    iree_host_size_t block_capacity, iree_allocator_t host_allocator,
    loom_serve_speech_consumer_t* out_consumer);

// Sends one complete JSON rejection and consumes the claim on every path.
// The response is copied into the transport before this function returns.
iree_status_t loom_serve_speech_http_reject(
    loom_serve_http_connection_t* connection, const char* code,
    iree_string_view_t message, iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_SPEECH_HTTP_H_
