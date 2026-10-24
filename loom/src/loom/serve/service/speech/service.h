// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_SPEECH_SERVICE_H_
#define LOOM_SERVE_SERVICE_SPEECH_SERVICE_H_

#include "loom/serve/runtime/host/host.h"
#include "loom/serve/service/service.h"
#include "loom/serve/service/speech/generator.h"
#include "loom/serve/service/speech/stream.h"
#include "loom/serve/service/speech/voice.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_speech_service_options_t {
  // Deployment selector borrowed through service retirement.
  iree_string_view_t name;
  // Positive bound on host-only requests beyond the model's resident slots.
  iree_host_size_t pending_capacity;
  // Periodic observation interval; zero disables periodic records only.
  iree_duration_t heartbeat_interval;
} loom_serve_speech_service_options_t;

typedef struct loom_serve_speech_service_t loom_serve_speech_service_t;

// Allocates a bounded streaming service without accepting asynchronous work.
// Native streams preserve the source's F32 waveform geometry. The HTTP PCM
// consumer requires 24 kHz mono output and rejects other geometry before audio.
// Source CPU work shares the host executor; device completion uses its
// proactor. A non-NULL telemetry stream remains borrowed through retirement.
iree_status_t loom_serve_speech_service_allocate(
    loom_serve_host_t* host, loom_serve_speech_generator_t generator,
    const loom_serve_speech_service_options_t* options,
    loom_serve_telemetry_stream_t* stream, iree_allocator_t host_allocator,
    loom_serve_speech_service_t** out_service);

// Opens a native output stream in the same bounded queue and compatible cohorts
// as HTTP requests. request must be initialized; successful acceptance moves it
// into the service and clears the caller's value. Success with NULL is normal
// queue pressure and leaves request unchanged. Failure leaves request
// unchanged. Calls are serialized on the serving owner. The caller closes the
// returned handle independently of model retirement, including after terminal
// failure.
iree_status_t loom_serve_speech_service_try_open(
    loom_serve_speech_service_t* service, loom_serve_speech_request_t* request,
    loom_serve_speech_stream_t** out_stream);

// Reserves a unique voice ID and prepares it through the ordinary bounded
// scheduler. Success moves request and returns a completion handle; OK with
// NULL is queue pressure and leaves request unchanged. An existing or pending
// ID returns ALREADY_EXISTS; exhausted voice slots return RESOURCE_EXHAUSTED.
// The ID becomes usable only after successful completion. Calls use the owner.
iree_status_t loom_serve_speech_service_try_upload_voice(
    loom_serve_speech_service_t* service,
    loom_serve_speech_voice_request_t* request,
    loom_serve_speech_voice_upload_t** out_upload);

// Hides a published ID immediately. Accepted synthesis requests retain the old
// voice until retirement, including requests waiting in the host queue. A new
// upload may reuse the name in a different free slot. Pending uploads are not
// published voices and cannot be removed; closing their handle cancels them.
iree_status_t loom_serve_speech_service_remove_voice(
    loom_serve_speech_service_t* service, iree_string_view_t id);

// POST /v1/audio/speech owns one bounded encoded block per resident request,
// outside copied carrier sends. Each request's credit independently determines
// eligibility for compatible source work. Original cohort membership survives
// cancellation through collection; all copies finish before feedback reuse.
// EOF, reset peers and shutdown release rows only at retired model cuts;
// failed readiness leaves accepted ownership with model teardown. Validation
// precedes the first audio response byte.
loom_serve_service_t loom_serve_speech_service_interface(
    loom_serve_speech_service_t* service);

// Frees after all accepted CPU, device and timer callbacks retire. Never waits
// for device work or destroys the borrowed model. NULL is accepted.
void loom_serve_speech_service_free(loom_serve_speech_service_t* service);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_SPEECH_SERVICE_H_
