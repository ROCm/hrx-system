// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REALTIME_GENERATION_H_
#define LOOM_SERVE_SERVER_REALTIME_GENERATION_H_

#include "iree/task/executor.h"
#include "loom/serve/media/audio_resampler.h"
#include "loom/serve/server/realtime/binding.h"
#include "loom/serve/service/text/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_realtime_generation_t
    loom_serve_realtime_generation_t;

typedef enum loom_serve_realtime_generation_filter_phase_e {
  LOOM_SERVE_REALTIME_FILTER_EMPTY,
  LOOM_SERVE_REALTIME_FILTER_PREPARING,
  LOOM_SERVE_REALTIME_FILTER_READY,
} loom_serve_realtime_generation_filter_phase_t;

// Shared by generations bound to one speech registration on one serving owner.
// Zero initialization is sufficient; construction uses the first audio worker.
typedef struct loom_serve_realtime_generation_filter_t {
  // Serialized owner state; READY publishes successful worker construction.
  loom_serve_realtime_generation_filter_phase_t phase;
  // Worker-constructed coefficients, immutable after release publishes READY.
  loom_serve_audio_resampler_filter_t value;
} loom_serve_realtime_generation_filter_t;

// Requires every borrowing generation and its accepted worker to have retired.
void loom_serve_realtime_generation_filter_deinitialize(
    loom_serve_realtime_generation_filter_t* filter);

// One native text -> continuing speech response. The caller owns its immutable
// session/checkpoint selector storage through free. Success moves text_request;
// failure leaves it untouched. The caller retains filter at a stable address
// through every borrowing generation's retirement. Each generation owns its
// mutable conversion state; shared coefficients never serialize prepared
// conversions. An empty voice selects text output without speech admission.
iree_status_t loom_serve_realtime_generation_allocate(
    iree_task_executor_t* executor, loom_serve_registry_t* registry,
    loom_serve_realtime_binding_t text, loom_serve_realtime_binding_t speech,
    loom_serve_text_request_t* text_request, iree_string_view_t voice,
    uint64_t seed, loom_serve_realtime_generation_filter_t* filter,
    loom_serve_wake_callback_t observer, iree_allocator_t host_allocator,
    loom_serve_realtime_generation_t** out_generation);

// One bounded owner turn. Network output pressure suppresses only this
// response's publication/consumption; collection and cancellation still run.
// Failure is response-local and the caller cancels before retiring the object.
iree_status_t loom_serve_realtime_generation_advance(
    loom_serve_realtime_generation_t* generation, bool can_publish,
    bool* out_progress);

// Pending source-classified content and base64 PCM. Views remain borrowed until
// the corresponding consume, cancellation, or free. Each nonempty audio view
// is one independently padded base64 packet, never a prefix of another packet.
iree_string_view_t loom_serve_realtime_generation_text(
    const loom_serve_realtime_generation_t* generation);
void loom_serve_realtime_generation_consume_text(
    loom_serve_realtime_generation_t* generation);
iree_string_view_t loom_serve_realtime_generation_audio(
    const loom_serve_realtime_generation_t* generation);
void loom_serve_realtime_generation_consume_audio(
    loom_serve_realtime_generation_t* generation);

// Published generated frame count, separate from the session's transport and
// client-playback frontiers. Never reads a still-running conversion cursor.
uint64_t loom_serve_realtime_generation_frames(
    const loom_serve_realtime_generation_t* generation);
bool loom_serve_realtime_generation_complete(
    const loom_serve_realtime_generation_t* generation);
// Final native text result may precede speech completion. Its borrowed views
// survive until cancel/free; validated tool transactions are never spoken.
const loom_serve_text_stream_result_t* loom_serve_realtime_generation_result(
    const loom_serve_realtime_generation_t* generation);

// Cancels publication immediately. A waveform borrowed by an accepted CPU job
// keeps its native handle until actual release, independently of disconnect.
void loom_serve_realtime_generation_cancel(
    loom_serve_realtime_generation_t* generation);
bool loom_serve_realtime_generation_pending(
    const loom_serve_realtime_generation_t* generation);
// Requires cancellation and no accepted worker. NULL is accepted.
void loom_serve_realtime_generation_free(
    loom_serve_realtime_generation_t* generation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_REALTIME_GENERATION_H_
