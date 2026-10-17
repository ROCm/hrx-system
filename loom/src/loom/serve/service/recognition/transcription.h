// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_RECOGNITION_TRANSCRIPTION_H_
#define LOOM_SERVE_SERVICE_RECOGNITION_TRANSCRIPTION_H_

#include "iree/base/api.h"
#include "loom/serve/service/recognition/document.h"
#include "loom/serve/service/recognition/input.h"
#include "loom/serve/service/recognition/request.h"
#include "loom/serve/transport/http/multipart.h"

#ifdef __cplusplus
extern "C" {
#endif

// Checked options and audio bytes borrowing one claimed multipart request.
typedef struct loom_serve_transcription_t {
  // Source-owned language, context, and decoding policy.
  loom_serve_recognition_request_t request;
  // File contents, independent of its untrusted filename.
  iree_const_byte_span_t audio;
  // Raw PCM16LE input rate; zero selects an encoded codec/container.
  uint32_t pcm_sample_rate;
  // Exact normalized frame count for raw PCM; zero for an encoded upload.
  uint64_t normalized_frame_count;
  // Plain UTF-8 text instead of the default JSON object containing text.
  bool text_response;
} loom_serve_transcription_t;

// Validates the finite transcription profile: required model/file, optional
// language/prompt/temperature, and json/text response_format. stream=false is
// accepted; streaming and alignment-specific response formats reject. Unknown
// and duplicate fields reject before decoding or source work. All views borrow
// the transport's immutable multipart storage; this function allocates nothing.
// audio/pcm file parts require a positive rate, channels matching format, and
// whole PCM16LE frames. Their normalized duration is checked without rounding
// packet boundaries. Other media types defer format selection to encoded bytes.
iree_status_t loom_serve_transcription_parse(
    loom_serve_http_part_list_t parts, iree_string_view_t model,
    loom_serve_recognition_format_t format,
    loom_serve_transcription_t* out_request);

// Appends a successful final response body. Segments are ordered by input start
// and identity, with a newline between nonempty segment texts. No timestamps,
// speaker names, or language guesses are fabricated from coverage intervals.
// The document has completed successfully and remains borrowed during encoding.
iree_status_t loom_serve_transcription_format(
    const loom_serve_recognition_document_t* document, bool text_response,
    iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_RECOGNITION_TRANSCRIPTION_H_
