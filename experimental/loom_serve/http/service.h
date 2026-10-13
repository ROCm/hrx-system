// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_HTTP_SERVICE_H_
#define EXPERIMENTAL_LOOM_SERVE_HTTP_SERVICE_H_

#include "experimental/loom_serve/http/inspection.h"
#include "experimental/loom_serve/http/server.h"
#include "experimental/loom_serve/runtime/host/wake.h"
#include "experimental/loom_serve/telemetry/stream.h"
#include "iree/tokenizer/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_serve_http_service_flag_bits_e {
  // Another service in the same capacity domain has accepted work or
  // maintenance that may release capacity. Admission can queue instead of
  // rejecting an idle-capacity stall.
  LOOM_SERVE_HTTP_SERVICE_FLAG_SHARED_ACTIVITY = 1u << 0,
  // Admission has stopped. Consume only already accepted asynchronous work;
  // new model submissions and queued request admission are forbidden.
  LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING = 1u << 1,
} loom_serve_http_service_flag_bits_t;
typedef uint32_t loom_serve_http_service_flags_t;

// Copied scheduler ownership, not hardware utilization. Finite active requests
// include preparation, capacity waiting and encoding; text active requests
// hold admitted rows. No request content or borrowed mutable buffers escape.
typedef struct loom_serve_http_service_snapshot_t {
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
} loom_serve_http_service_snapshot_t;

// One named service borrowing a model on serialized task-owner turns. Network
// progress is independent; these callbacks never run concurrently. Names are
// deployment identifiers, not source model identities or checkpoint paths.
typedef struct loom_serve_http_service_t {
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
  // Takes the connection claim on both success and failure. Invalid requests
  // produce an HTTP rejection and success; a terminal failure ends the owner
  // loop. Request views remain valid until this service finishes or aborts.
  iree_status_t (*accept)(void* self, loom_serve_http_connection_t* connection,
                          const loom_serve_http_request_t* request,
                          loom_serve_http_service_flags_t flags);
  // Advances at most one bounded model unit and reports actual progress. A
  // turn can retry admission after another service released shared capacity.
  // False means awaiting service readiness or a network event, never polling.
  // STOPPING consumes accepted readiness without launching another model unit.
  iree_status_t (*advance)(void* self, loom_serve_http_service_flags_t flags,
                           bool* out_progress);
  // Optional immutable input tokenizer, live through utility task release. Raw
  // utilities use independent CPU states and never call this model's methods.
  const iree_tokenizer_t* tokenizer;
  // Immutable modality name with process lifetime, independent of the model.
  const char* kind;
  // Primary inference endpoint with process lifetime; excludes utilities and
  // modality-specific control routes such as text checkpoints.
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
  } capabilities;
  // Copies published ownership on the serialized owner without invoking model
  // work or reading state still borrowed by a CPU/device task.
  loom_serve_http_service_snapshot_t (*snapshot)(void* self);
  // Optional bounded metadata queries on the same serialized owner. No task
  // payload or model borrow escapes into an observation response.
  loom_serve_http_inspection_t inspection;
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
  } telemetry;
} loom_serve_http_service_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_HTTP_SERVICE_H_
