// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_SPEECH_REQUEST_H_
#define LOOM_SERVE_SERVICE_SPEECH_REQUEST_H_

#include "iree/base/api.h"
#include "iree/base/string_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Owned synthesis input, independent of tokenization and voice conditioning.
typedef struct loom_serve_speech_request_t {
  // Decoded UTF-8 text to synthesize; never borrowed from the connection.
  iree_string_builder_t input;
  // Decoded voice selector, including custom voice objects' id field.
  iree_string_builder_t voice;
  // Optional decoded UTF-8 instructions, interpreted by the source model.
  iree_string_builder_t instructions;
  // Native sampling key; zero when omitted. Interpretation belongs to source.
  uint64_t seed;
  // Requested speech speed in [0.25, 4], with one meaning the natural speed.
  float speed;
} loom_serve_speech_request_t;

// Copies a native request without JSON encoding. Text, voice, and instructions
// must be UTF-8; text and voice are nonempty and voice contains no NUL. Speed
// is finite and in [0.25, 4]. Source preparation owns model-specific limits and
// voice/instruction support. Failure leaves a deinitializable empty value.
iree_status_t loom_serve_speech_request_initialize_text(
    iree_string_view_t input, iree_string_view_t voice,
    iree_string_view_t instructions, uint64_t seed, float speed,
    iree_allocator_t host_allocator, loom_serve_speech_request_t* out_request);

// Decodes one complete UTF-8 JSON/JSONC speech request. Model, nonempty input
// and voice are required; voice accepts a string or an object containing id.
// The binary PCM protocol requires explicit response_format="pcm" and accepts
// omitted stream_format or "audio". Speed defaults to one. Seed is an optional
// unsigned integer or decimal-string extension. Source preparation owns text
// capacity, voice availability, instruction and speed support. Unknown,
// duplicate, wrongly typed and trailing fields reject before model admission.
// Failure releases partial storage and leaves a deinitializable empty value.
iree_status_t loom_serve_speech_request_initialize(
    iree_string_view_t body, iree_string_view_t model,
    iree_allocator_t host_allocator, loom_serve_speech_request_t* out_request);

// Releases owned decoded strings. An empty value is accepted.
void loom_serve_speech_request_deinitialize(
    loom_serve_speech_request_t* request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_SPEECH_REQUEST_H_
