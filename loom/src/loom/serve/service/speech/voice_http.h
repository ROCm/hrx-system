// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_SPEECH_VOICE_HTTP_H_
#define LOOM_SERVE_SERVICE_SPEECH_VOICE_HTTP_H_

#include "loom/serve/service/speech/voice.h"
#include "loom/serve/transport/http/request.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parses model/id/file multipart input into an owned native upload. Original
// bytes are copied once; no decoding, filesystem access, or model work occurs.
iree_status_t loom_serve_speech_voice_http_parse(
    const loom_serve_http_request_t* request, iree_string_view_t model,
    iree_allocator_t host_allocator,
    loom_serve_speech_voice_request_t* out_request);

// Parses the model/id JSON removal command into an initialized ID builder.
// Failure leaves the builder valid for deinitialization.
iree_status_t loom_serve_speech_voice_http_remove_parse(
    const loom_serve_http_request_t* request, iree_string_view_t model,
    iree_string_builder_t* out_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_SPEECH_VOICE_HTTP_H_
