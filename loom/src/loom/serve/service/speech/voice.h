// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_SPEECH_VOICE_H_
#define LOOM_SERVE_SERVICE_SPEECH_VOICE_H_

#include "loom/serve/media/audio_source.h"
#include "loom/serve/service/speech/consumer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Owned upload, movable into the service on successful admission.
// Initialization parses the same audio metadata as dictation; decoding runs on
// a CPU worker.
typedef struct loom_serve_speech_voice_request_t {
  // Owned UTF-8 identifier, used by subsequent synthesis requests.
  iree_string_builder_t id;
  // Owned original recording bytes, retained until preparation retires.
  iree_string_builder_t storage;
  // Validated format and borrowed view into storage.
  loom_serve_audio_source_t audio;
} loom_serve_speech_voice_request_t;

// Copies caller bytes; success is independent of their lifetime. Failure leaves
// a zeroed request. A voice ID is nonempty UTF-8 without embedded NUL bytes.
iree_status_t loom_serve_speech_voice_request_initialize(
    iree_string_view_t id, iree_string_view_t content_type,
    iree_const_byte_span_t audio, iree_allocator_t host_allocator,
    loom_serve_speech_voice_request_t* out_request);
void loom_serve_speech_voice_request_deinitialize(
    loom_serve_speech_voice_request_t* request);

typedef struct loom_serve_speech_voice_upload_t
    loom_serve_speech_voice_upload_t;

// Constructs caller and service ownership of a completion receipt. This is the
// service's consumer adapter; caller-facing admission is
// service_try_upload_voice. The handle contains no borrowed model/service state
// and accepts no work here.
iree_status_t loom_serve_speech_voice_upload_allocate(
    iree_allocator_t host_allocator,
    loom_serve_speech_voice_upload_t** out_upload,
    loom_serve_speech_consumer_t* out_consumer);

// Completion follows source retirement and publication of the voice ID. Pending
// status is OK; is_complete distinguishes a usable voice from ongoing work.
// The result remains readable after model removal. Calls use the serving owner.
bool loom_serve_speech_voice_upload_is_complete(
    const loom_serve_speech_voice_upload_t* upload);
iree_status_t loom_serve_speech_voice_upload_status(
    const loom_serve_speech_voice_upload_t* upload);

// Consumes the caller handle without waiting. Closing before completion cancels
// publication; accepted CPU/device work retains its ownership until retirement.
void loom_serve_speech_voice_upload_close(
    loom_serve_speech_voice_upload_t* upload);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_SPEECH_VOICE_H_
