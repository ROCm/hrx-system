// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_SERVICE_H_
#define LOOM_SERVE_SERVICE_SERVICE_H_

#include "iree/tokenizer/tokenizer.h"
#include "loom/runtime/service.h"
#include "loom/runtime/telemetry/sample.h"
#include "loom/runtime/telemetry/stream.h"
#include "loom/runtime/wake.h"
#include "loom/serve/service/reasoning.h"
#include "loom/serve/transport/http/server.h"
#include "loom/service/inspection.h"
#include "loom/service/kind.h"
#include "loom/service/reconfiguration.h"
#include "loom/service/telemetry/latency.h"
#include "loom/service/telemetry/record.h"

#ifdef __cplusplus
extern "C" {
#endif

// Modality-owned input and detached output types used by native admission.
typedef struct loom_serve_speech_request_t loom_serve_speech_request_t;
typedef struct loom_serve_speech_stream_t loom_serve_speech_stream_t;
typedef struct loom_serve_speech_voice_request_t
    loom_serve_speech_voice_request_t;
typedef struct loom_serve_speech_voice_upload_t
    loom_serve_speech_voice_upload_t;
typedef struct loom_serve_recognition_request_t
    loom_serve_recognition_request_t;
typedef struct loom_serve_recognition_stream_t loom_serve_recognition_stream_t;
typedef struct loom_serve_text_request_t loom_serve_text_request_t;
typedef struct loom_serve_text_checkpoint_request_t
    loom_serve_text_checkpoint_request_t;
typedef struct loom_serve_text_checkpoint_control_t
    loom_serve_text_checkpoint_control_t;
typedef struct loom_serve_text_stream_t loom_serve_text_stream_t;

// One named service borrowing a model on serialized task-owner turns. Network
// progress is independent; these callbacks never run concurrently. Names are
// deployment identifiers, not source model identities or checkpoint paths.
typedef struct loom_serve_service_t {
  // Nonempty visible ASCII name, unique and borrowed through retirement. This
  // also fits the checkpoint selector's HTTP header representation.
  iree_string_view_t name;
  // Common progress capability; all callbacks below share runtime.self.
  loom_runtime_service_t runtime;
  // Optional HTTP inference; absent when endpoint is NULL. Takes the connection
  // claim on both success and failure. Invalid requests produce an HTTP
  // rejection and success; a terminal failure stops this service's admission
  // and requires its owner to retire accepted borrowers. Request views remain
  // valid until this service finishes or aborts.
  iree_status_t (*accept)(void* self, loom_serve_http_connection_t* connection,
                          const loom_serve_http_request_t* request,
                          loom_runtime_service_flags_t flags);
  // Optional native speech admission on this same serialized owner. Success
  // with a stream moves the initialized request; NULL success is bounded
  // pressure. Both pressure and failure leave request untouched. The returned
  // handle owns copied results independently of registration retirement.
  iree_status_t (*try_open_speech)(void* self,
                                   loom_serve_speech_request_t* request,
                                   loom_serve_speech_stream_t** out_stream);
  // Optional one-time voice preparation. Admission moves the owned recording
  // only with a non-NULL receipt; pressure/failure preserves caller ownership.
  iree_status_t (*try_upload_voice)(
      void* self, loom_serve_speech_voice_request_t* request,
      loom_serve_speech_voice_upload_t** out_upload);
  // Hides an uploaded voice without invalidating accepted synthesis captures.
  iree_status_t (*remove_voice)(void* self, iree_string_view_t id);
  // Optional native continuing-input recognition on this serialized owner.
  // Success with a stream copies request options and grants bounded PCM credit
  // in the stream's format. NULL success is pressure. The caller owns the
  // handle through close, independently of service/model lifetime.
  iree_status_t (*try_open_recognition)(
      void* self, const loom_serve_recognition_request_t* request,
      loom_serve_recognition_stream_t** out_stream);
  // Optional native chat admission, sharing the same pending queue, retained
  // histories, and packed execution. Success with a stream moves the owning
  // JSON request; pressure/failure leaves it untouched. Final semantic output
  // belongs to the caller independently of registration retirement.
  iree_status_t (*try_open_text)(void* self, loom_serve_text_request_t* request,
                                 loom_serve_text_stream_t** out_stream);
  // Optional native checkpoint control, sharing the model's existing control
  // slot and retained-history rules. Acceptance copies identifiers. NULL means
  // queue pressure; a returned handle retains its result beyond model removal.
  iree_status_t (*try_open_text_checkpoint)(
      void* self, const loom_serve_text_checkpoint_request_t* request,
      loom_serve_text_checkpoint_control_t** out_control);
  // Optional immutable input tokenizer, live through utility task release. Raw
  // utilities use independent CPU states and never call this model's methods.
  const iree_tokenizer_t* tokenizer;
  // Immutable deployment role, independent of model lifetime.
  loom_service_kind_t kind;
  // Primary inference endpoint with process lifetime; excludes utilities and
  // modality-specific control routes such as text checkpoints. NULL means
  // native-only inference and requires an absent accept callback.
  const char* endpoint;
  // Immutable client capabilities published with the prepared service. Views
  // remain valid through retirement; empty lists and zero limits mean unknown
  // or inapplicable, never unlimited. Limits describe one request, not a pool.
  struct {
    // Accepted input modalities, such as text or image.
    iree_string_view_list_t input_modalities;
    // Produced output modalities, such as text, image, embedding or decision.
    iree_string_view_list_t output_modalities;
    // Maximum combined input and output token extent of one session.
    iree_host_size_t context_length;
    // Maximum selectable output token budget, further bounded by input length.
    iree_host_size_t max_output_tokens;
    // Exact source-prepared counting at /v1/chat/completions/input_tokens.
    bool input_token_count;
    // Source-declared reasoning support and effort selection, if known.
    loom_serve_reasoning_t reasoning;
  } capabilities;
  // Copies published ownership on the serialized owner without invoking model
  // work or reading state still borrowed by a CPU/device task.
  loom_service_snapshot_t (*snapshot)(void* self);
  // Optional bounded metadata queries on the same serialized owner. No task
  // payload or model borrow escapes into an observation response.
  loom_service_inspection_t inspection;
  // Optional service-owned executable pool and preferred-choice policy. Its
  // producers and queued selections participate in is_pending retirement.
  loom_service_reconfiguration_t reconfiguration;
  // Optional observation owned by the service and live through retirement.
  struct {
    // Bounded immutable-record history; NULL when this service has no stream.
    const loom_runtime_telemetry_stream_t* stream;
    // Cumulative work owned by this named service, copied only on serialized
    // owner turns, never replayed from history. Non-NULL whenever stream is
    // present; counters may also be published without an event stream. When
    // present, stream creation is the zero-counter epoch for these totals.
    const loom_service_telemetry_counters_t* counters;
    // Optional cumulative request latency, independent of ring retention.
    // Managed services publish this alongside counters. Borrowed application
    // services may omit it; missing observations are never reported as zero.
    const loom_service_telemetry_latency_t* latency;
    // Optional source-defined numeric topics, sampled on the serialized owner.
    // The sink consumes copied values synchronously; no model borrow escapes.
    // Common request counters and latency remain in their dedicated contracts.
    iree_status_t (*sample)(void* self,
                            loom_runtime_telemetry_sample_sink_t sink);
  } telemetry;
} loom_serve_service_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_SERVICE_H_
