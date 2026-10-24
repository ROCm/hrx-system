// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_SERVICE_H_
#define LOOM_SERVE_SERVICE_SERVICE_H_

#include "iree/tokenizer/tokenizer.h"
#include "loom/serve/runtime/host/wake.h"
#include "loom/serve/service/inspection.h"
#include "loom/serve/service/kind.h"
#include "loom/serve/service/reconfiguration.h"
#include "loom/serve/telemetry/latency.h"
#include "loom/serve/telemetry/stream.h"
#include "loom/serve/transport/http/server.h"

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

typedef enum loom_serve_service_flag_bits_e {
  // Another service in the same capacity domain has accepted work or
  // maintenance that may release capacity. Admission can queue instead of
  // rejecting an idle-capacity stall.
  LOOM_SERVE_SERVICE_FLAG_SHARED_ACTIVITY = 1u << 0,
  // Admission has stopped. Consume only already accepted asynchronous work;
  // new model submissions and queued request admission are forbidden.
  LOOM_SERVE_SERVICE_FLAG_STOPPING = 1u << 1,
} loom_serve_service_flag_bits_t;
typedef uint32_t loom_serve_service_flags_t;

// Copied scheduler ownership, not hardware utilization. Finite active requests
// include preparation, capacity waiting and encoding; text active requests
// hold admitted rows. No request content or borrowed mutable buffers escape.
typedef struct loom_serve_service_snapshot_t {
  // Modality-owned phase label with process lifetime.
  const char* phase;
  // Requests waiting outside the active invocation/cohort.
  iree_host_size_t queued_requests;
  // Requests retained by the active invocation/cohort, including reset peers
  // whose accepted work still has to retire.
  iree_host_size_t active_requests;
  // Nanoseconds since the oldest queued request was accepted; zero if empty.
  iree_duration_t oldest_queued_age;
  // Nanoseconds since the oldest active request was accepted, including prior
  // queue residence; zero if empty. This is ownership age, not device time.
  iree_duration_t oldest_active_age;
} loom_serve_service_snapshot_t;

// One named service borrowing a model on serialized task-owner turns. Network
// progress is independent; these callbacks never run concurrently. Names are
// deployment identifiers, not source model identities or checkpoint paths.
typedef struct loom_serve_service_t {
  // Nonempty visible ASCII name, unique and borrowed through retirement. This
  // also fits the checkpoint selector's HTTP header representation.
  iree_string_view_t name;
  // Borrowed service state, live through observer detachment and all requests.
  void* self;
  // Installs the owner's nonblocking wake target before the first turn. The
  // owner clears it after all accepted asynchronous callbacks have retired.
  void (*set_observer)(void* self, loom_serve_wake_callback_t observer);
  // Whether an accepted asynchronous callback still borrows service storage.
  // True requires an eventual wake even after transport admission stops.
  bool (*is_pending)(void* self);
  // Whether live requests or pending maintenance may release shared capacity.
  // Queued requests alone are not activity: mutually blocked queues cannot
  // make progress by waiting on one another. Called by the serialized owner.
  bool (*is_active)(void* self);
  // Optional HTTP inference; absent when endpoint is NULL. Takes the connection
  // claim on both success and failure. Invalid requests
  // produce an HTTP rejection and success; a terminal failure ends the owner
  // loop. Request views remain valid until this service finishes or aborts.
  iree_status_t (*accept)(void* self, loom_serve_http_connection_t* connection,
                          const loom_serve_http_request_t* request,
                          loom_serve_service_flags_t flags);
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
  // Advances at most one bounded model unit and reports actual progress. A
  // turn can retry admission after another service released shared capacity.
  // False means awaiting service readiness or a network event, never polling.
  // STOPPING consumes accepted readiness without launching another model unit.
  iree_status_t (*advance)(void* self, loom_serve_service_flags_t flags,
                           bool* out_progress);
  // Optional immutable input tokenizer, live through utility task release. Raw
  // utilities use independent CPU states and never call this model's methods.
  const iree_tokenizer_t* tokenizer;
  // Immutable deployment role, independent of model lifetime.
  loom_serve_model_kind_t kind;
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
  } capabilities;
  // Copies published ownership on the serialized owner without invoking model
  // work or reading state still borrowed by a CPU/device task.
  loom_serve_service_snapshot_t (*snapshot)(void* self);
  // Optional bounded metadata queries on the same serialized owner. No task
  // payload or model borrow escapes into an observation response.
  loom_serve_http_inspection_t inspection;
  // Optional service-owned executable pool and preferred-choice policy. Its
  // producers and queued selections participate in is_pending retirement.
  loom_serve_reconfiguration_t reconfiguration;
  // Borrowed shared-capacity owner identity, never dereferenced by the router.
  // Only equal non-NULL identities can provide SHARED_ACTIVITY to each other;
  // NULL means this service has no shared-capacity relationship. The owner
  // outlives the service and every accepted asynchronous obligation.
  const void* capacity_domain;
  // Optional observation owned by the service and live through retirement.
  struct {
    // Bounded immutable-record history; NULL when this service has no stream.
    const loom_serve_telemetry_stream_t* stream;
    // Cumulative work owned by this named service, copied only on serialized
    // owner turns, never replayed from history. Non-NULL whenever stream is
    // present; counters may also be published without an event stream.
    const loom_serve_telemetry_counters_t* counters;
    // Optional cumulative request latency, independent of ring retention.
    // Managed services publish this alongside counters. Borrowed application
    // services may omit it; missing observations are never reported as zero.
    const loom_serve_telemetry_latency_t* latency;
  } telemetry;
} loom_serve_service_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_SERVICE_H_
