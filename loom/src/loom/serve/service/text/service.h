// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_SERVICE_H_
#define LOOM_SERVE_SERVICE_TEXT_SERVICE_H_

#include "loom/runtime/service.h"
#include "loom/runtime/telemetry/sample.h"
#include "loom/serve/service/text/checkpoint_control.h"
#include "loom/serve/service/text/input_count.h"
#include "loom/serve/service/text/model.h"
#include "loom/serve/service/text/schedule.h"
#include "loom/serve/service/text/stream.h"
#include "loom/service/finite/component.h"
#include "loom/service/inspection.h"
#include "loom/service/kind.h"
#include "loom/service/reconfiguration.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_text_service_options_t {
  // Borrowed deployment route name, outliving the service. Source identity and
  // chat policy remain owned by the model, independently of this route name.
  iree_string_view_t name;
  // Optional borrowed image component, outliving this service. Text creation,
  // activation and ordinary requests do not start its preparation. Accepted
  // preparation is advanced/joined by the service's serialized owner.
  loom_service_finite_component_t* image;
  // Initial resident capacity, matching the model at service creation. The
  // directory grows as requests arrive; memory credit gates active requests.
  iree_host_size_t row_count;
  // Maximum resident owners, including preparation and output borrowers.
  // Zero imposes no count limit. A nonzero limit must cover initial row_count.
  iree_host_size_t max_sessions;
  // Maximum known input tokens admitted from one row in an epoch.
  iree_host_size_t chunk_size;
  // Output bound when a request omits max_tokens.
  iree_host_size_t default_max_tokens;
  // Positive bound on claimed requests waiting without a model row.
  iree_host_size_t pending_capacity;
  // Maximum cached executable choices, including the original and pending
  // compilation attempts. Zero disables native reconfiguration.
  iree_host_size_t variant_capacity;
  // Successful overtakes permitted per blocked oldest request. Zero is FIFO;
  // deployment configuration defaults to one execution-row cohort.
  iree_host_size_t admission_overtakes;
  // Detached idle endpoints retained beyond execution rows; zero disables.
  // The model reserves this many plus one exchange endpoint in addition to
  // explicit pin capacity. Automatic entries yield to admission pressure.
  iree_host_size_t session_cache_capacity;
  // Exact intermediate token prefixes shared across sessions; zero disables.
  // Native endpoints are reserved separately from idle histories/explicit pins.
  iree_host_size_t prefix_cache_capacity;
  // Reporting interval in nanoseconds; zero disables periodic heartbeats.
  iree_duration_t heartbeat_interval;
  // Execution choice within the shared admission and retirement lifecycle.
  loom_service_text_schedule_mode_t schedule_mode;
  // Whether prompt and decode inputs may share an epoch, independent of math.
  loom_service_text_packing_mode_t packing_mode;
  // One or two device-fed epochs between transport/admission observations.
  // Two requires drafting; output credit and speculative residency cover both.
  iree_host_size_t continuation_epochs;
} loom_serve_text_service_options_t;

typedef struct loom_serve_text_service_t loom_serve_text_service_t;

// Borrowed text capabilities on the service's serialized owner. Frontends own
// transport routes and delivery; none of these observations starts model work.
typedef struct loom_serve_text_service_interface_t {
  // Deployment identifier, borrowed through service retirement.
  iree_string_view_t name;
  // Shared progress and notification, using this service as runtime.self.
  loom_runtime_service_t runtime;
  // Generative text or known-input sequence semantics.
  loom_service_kind_t kind;
  // Immutable input tokenizer, or NULL for known-input sequences.
  const iree_tokenizer_t* tokenizer;
  // Logical input plus output token bound for one retained session.
  iree_host_size_t context_length;
  // Source accepts images through the optional prepared finite component.
  bool image_input;
  // Immutable source-declared reasoning capabilities.
  loom_serve_reasoning_t reasoning;
  // Copies ownership facts without inspecting accepted task payloads.
  loom_service_snapshot_t (*snapshot)(void* self);
  // Bounded copied request and retained-session observations.
  loom_service_inspection_t inspection;
  // Executable choices and queued safe-point selection.
  loom_service_reconfiguration_t reconfiguration;
  // Read-only history and accounting, borrowed through service retirement.
  struct {
    // Bounded immutable-record history with independent reader cursors.
    const loom_runtime_telemetry_stream_t* stream;
    // Cumulative work, independent of retained history.
    const loom_service_telemetry_counters_t* counters;
    // Cumulative terminal-request latency.
    const loom_service_telemetry_latency_t* latency;
    // Source observations copied synchronously on the serialized owner.
    iree_status_t (*sample)(void* self,
                            loom_runtime_telemetry_sample_sink_t sink);
  } telemetry;
} loom_serve_text_service_interface_t;

// Creates host session/admission state around a borrowed model. CPU work shares
// the host executor; reporting shares its proactor. No per-model thread is
// created. Initial capacity matches the model; subsequent growth retains the
// same programs and parameters. Calls use its one serialized host owner.
// A non-null stream lends registration history through destruction; NULL
// creates a service-owned history for a borrowed single-model deployment.
iree_status_t loom_serve_text_service_allocate(
    loom_serve_text_model_t* model,
    const loom_serve_text_service_options_t* options,
    loom_runtime_telemetry_stream_t* stream, iree_allocator_t host_allocator,
    loom_serve_text_service_t** out_service);

// Immutable semantic role, also used by frontend route selection.
loom_service_kind_t loom_serve_text_service_kind(
    const loom_serve_text_service_t* service);

// Closes admission irreversibly without advancing or waiting for accepted
// work. Subsequent owner turns retire that work using the normal stop path.
void loom_serve_text_service_stop(loom_serve_text_service_t* service);

// Whether ingress remains open; bounded queue credit is checked by acceptance.
bool loom_serve_text_service_is_accepting(
    const loom_serve_text_service_t* service);

// Validates selectors and bounded queue credit before constructing a delivery
// receipt. create runs synchronously at most once; it retains the immutable
// body and returns the modality's consumer, or fails without taking ownership.
// It neither reenters the service nor starts source work. Successful acceptance
// transfers that receipt; pressure returns OK/false without calling create.
// arrival_time is the frontend's monotonic receipt time, before preflight.
iree_status_t loom_serve_text_service_try_accept(
    loom_serve_text_service_t* service, iree_string_view_t body,
    iree_string_view_t session, iree_string_view_t checkpoint,
    iree_time_t arrival_time,
    iree_status_t (*create)(void* user_data, iree_allocator_t host_allocator,
                            loom_serve_text_consumer_t* out_consumer),
    void* user_data, bool* out_accepted);

// Borrows immutable input through a typed delivery receipt, transferred only
// on acceptance. Pressure and failure preserve caller ownership and do not
// count a receipt. These operations queue demand without starting source work.
iree_status_t loom_serve_text_service_try_accept_input_count(
    loom_serve_text_service_t* service, iree_string_view_t body,
    iree_time_t arrival_time, loom_serve_text_input_count_consumer_t consumer,
    bool* out_accepted);
iree_status_t loom_serve_text_service_try_accept_checkpoint(
    loom_serve_text_service_t* service,
    const loom_serve_text_checkpoint_request_t* request,
    iree_time_t arrival_time, loom_serve_text_checkpoint_consumer_t consumer,
    bool* out_accepted);

// Records a frontend rejection and consumes failure through the synchronous
// reply. The reply borrows accounting only to diagnose delivery failure. It
// consumes its context's receipt on every return; an allocation failure ends
// admission and propagates, while disconnected delivery remains request-local.
iree_status_t loom_serve_text_service_reject(
    loom_serve_text_service_t* service, iree_time_t arrival_time,
    iree_status_t failure,
    iree_status_t (*reply)(void* user_data, iree_status_t failure,
                           loom_serve_text_telemetry_t* telemetry,
                           loom_serve_text_receipt_t receipt,
                           iree_allocator_t host_allocator),
    void* user_data);

// Enqueues native chat on the same serialized owner and bounded arrival queue
// as HTTP. Success with a stream moves request and copies its selectors; NULL
// success is queue pressure. Pressure and failure leave request untouched and
// do not count a receipt. Acceptance starts no source work; the caller advances
// the service to start validation on a shared CPU task and subsequent model
// execution. The caller owns the handle until close.
iree_status_t loom_serve_text_service_try_open(
    loom_serve_text_service_t* service, loom_serve_text_request_t* request,
    loom_serve_text_stream_t** out_stream);

// Enqueues one native checkpoint control beside the generation queue. A full
// control slot returns OK/NULL without counting a receipt. Acceptance copies
// validated identifiers; source lookup and native state changes occur only at
// an unborrowed model boundary. The caller owns its detached result until
// close.
iree_status_t loom_serve_text_service_try_open_checkpoint(
    loom_serve_text_service_t* service,
    const loom_serve_text_checkpoint_request_t* request,
    loom_serve_text_checkpoint_control_t** out_control);

// Accepts a stateless count using the model's ordinary input preparation. On
// acceptance moves body into an independent caller/service receipt. OK/NULL
// means queue pressure; failure or pressure leaves body untouched. Counting
// reserves no execution row, KV state, or parameters. The caller drives the
// same service owner as inference and closes the returned call when finished.
iree_status_t loom_serve_text_service_try_open_input_count(
    loom_serve_text_service_t* service, iree_string_builder_t* body,
    loom_serve_text_input_count_call_t** out_call);

// Returns borrowed callbacks for the shared owner loop. Each epoch gathers
// credited rows, submits known/verifier spans, and commits output only after
// readiness. Independent services continue while that work executes.
// Admission physically backs completion and speculative high-water, private
// recurrent writers, and partial-tail copies; shared prefixes count once.
// Bounded arrival-order queues allow only the configured successful overtakes
// of a blocked oldest request. Stalled output can retain reservations until
// consumption or cancellation; cancellation joins actual accepted borrowers.
//
// Request session selectors resume retained canonical history. Checkpoint
// selectors fork explicit immutable endpoints after full history validation.
// Idle histories and intermediate prefixes yield to shared pool pressure;
// explicit pins and accepted preparation/reservation borrowers remain
// protected. Optional image projection retains the exact text continuation and
// device dependencies without activating language parameters. Capacity retries
// reuse prepared input until history eviction invalidates its selected prefix.
//
// Acceptance queues demand without starting source work. The owner advances
// preparation, counting, checkpoint operations, and model work explicitly.
loom_serve_text_service_interface_t loom_serve_text_service_interface(
    loom_serve_text_service_t* service);

// Relinquishes the idle projector's model borrow before independent image
// component release. Admission is closed and every request/CPU/device borrower
// has retired. Text model, retained histories, and checkpoints are unchanged;
// the next image demand creates a new projector around the prepared component.
void loom_serve_text_service_release_projector(
    loom_serve_text_service_t* service);

// Relinquishes every accepted consumer receipt after readiness retirement. On
// terminal failure, unresolved device reservations remain pinned for model
// destruction; on ordinary completed-boundary shutdown, unused reservations are
// released. terminal_code is the owner loop's result, including another model's
// failure. The owner has consumed all CPU, device, compiler and heartbeat
// callbacks, including queued selection and timer cancellation receipts, before
// this call. Null is accepted.
// The borrowed model and each delivery owner outlive their final callback.
// Native caller handles retain results independently of this service.
void loom_serve_text_service_free(loom_serve_text_service_t* service,
                                  iree_status_code_t terminal_code);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_SERVICE_H_
