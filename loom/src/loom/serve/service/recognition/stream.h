// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_RECOGNITION_STREAM_H_
#define LOOM_SERVE_SERVICE_RECOGNITION_STREAM_H_

#include "loom/runtime/host/host.h"
#include "loom/runtime/service.h"
#include "loom/runtime/wake.h"
#include "loom/serve/service/recognition/document.h"
#include "loom/serve/service/recognition/generator.h"
#include "loom/service/telemetry/record.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_recognition_stream_service_t
    loom_serve_recognition_stream_service_t;
typedef struct loom_serve_recognition_stream_t loom_serve_recognition_stream_t;

// Optional owner-side notification that stream execution ownership has ended.
// The receipt remains borrowed through accepted host work, including caller
// close and shutdown. It does not retain final transcripts or run on a worker.
typedef struct loom_serve_recognition_completion_t {
  // Borrowed recipient, alive until the callback returns.
  void* self;
  // Called exactly once for an admitted stream, after host borrowers retire.
  // On terminal failure, accepted device resources remain with model teardown.
  // The status is borrowed for this call; no numerical work or blocking is
  // permitted. NULL omits notification.
  void (*end)(void* self, loom_service_telemetry_outcome_t outcome,
              const iree_status_t status);
} loom_serve_recognition_completion_t;

typedef struct loom_serve_recognition_stream_service_options_t {
  // Optional deployment identity, borrowed through service destruction.
  iree_string_view_t name;
  // Maximum copied language and prompt bytes per request.
  iree_host_size_t request_limit;
  // Maximum retained transcript bytes per request, including metadata/index.
  iree_host_size_t document_limit;
} loom_serve_recognition_stream_service_options_t;

// Allocates a typed continuing-input service shared by finite transcription
// and live audio callers. It borrows the host and generator through retirement.
// All calls are serialized on the owning task; source CPU work uses the host
// executor and device readiness uses its proactor. No private threads or waits
// are introduced. Construction accepts no asynchronous work.
// Scheduling storage has one offer/choice and two operation slots, with one
// device use each. Its owner retains it through actual model retirement.
iree_status_t loom_serve_recognition_stream_service_allocate(
    loom_runtime_host_t* host, loom_serve_recognition_generator_t generator,
    const loom_serve_recognition_stream_service_options_t* options,
    loom_runtime_service_control_binding_t scheduling,
    iree_allocator_t host_allocator,
    loom_serve_recognition_stream_service_t** out_service);

// Installs a nonblocking owner wake before first advance; detach only after
// is_pending becomes false. Final callback publication joins this registration.
void loom_serve_recognition_stream_service_set_observer(
    loom_serve_recognition_stream_service_t* service,
    loom_runtime_wake_callback_t observer);

// Opens a bounded host stream and copies its options. Success with NULL means
// all source slots are occupied; the caller retains its own queue position.
// A non-NULL handle belongs to the caller until close, independently of service
// and model lifetime. Final results need no model slot or input-ring storage.
// Output selection must be a nonempty subset of the source's capabilities.
iree_status_t loom_serve_recognition_stream_service_try_open(
    loom_serve_recognition_stream_service_t* service,
    const loom_serve_recognition_request_t* request,
    loom_serve_recognition_completion_t completion,
    loom_serve_recognition_stream_t** out_stream);

// Immutable normalized sample geometry, copied from the admitted model.
loom_serve_recognition_format_t loom_serve_recognition_stream_format(
    const loom_serve_recognition_stream_t* stream);

// Immutable source input-window capacity, including lookahead. This is not
// current write credit and remains available after execution/model retirement.
iree_host_size_t loom_serve_recognition_stream_input_capacity(
    const loom_serve_recognition_stream_t* stream);

// Appends whole finite F32LE frames in the generator's declared geometry.
// Copies only the available credit; zero accepted frames is normal pressure.
// Malformed PCM, exceeding the request frame limit, and writing after EOF fail
// without accepting any frames. Calls never wait for model or output progress.
iree_status_t loom_serve_recognition_stream_write(
    loom_serve_recognition_stream_t* stream, iree_const_byte_span_t samples,
    iree_host_size_t* out_frame_count);
iree_host_size_t loom_serve_recognition_stream_credit(
    const loom_serve_recognition_stream_t* stream);

// Marks explicit input EOF after the last accepted frame. Source alone flushes
// lookahead/remainders; network packet boundaries never pad or finalize audio.
// Repeated completion is harmless and does not resubmit final input.
void loom_serve_recognition_stream_finish(
    loom_serve_recognition_stream_t* stream);

// Borrows the next source-ordered event until consume/close. One source result
// may contain several observations or revisions. A stalled consumer makes only
// this stream ineligible for the next compatible batch. NULL means no published
// event, including while CPU collection is active.
const loom_serve_recognition_event_t* loom_serve_recognition_stream_peek(
    const loom_serve_recognition_stream_t* stream);
// Consumes the non-NULL event returned by peek on this owner turn.
void loom_serve_recognition_stream_consume(
    loom_serve_recognition_stream_t* stream);

// Completion becomes visible after execution ownership ends and the final
// pending event is consumed. A stopped service leaves a cancelled terminal
// result. Failed readiness ends this stream but does not retire device
// resources; those remain owned by model teardown.
bool loom_serve_recognition_stream_is_complete(
    const loom_serve_recognition_stream_t* stream);

// Owner-only fact: this request awaits another source unit, including while
// waiting for input or output credit. Reads only published row state, without
// calling the source. Retained final results are not waiting requests.
bool loom_serve_recognition_stream_is_waiting(
    const loom_serve_recognition_stream_t* stream);

// Returns an owned copy of the request failure, including service-wide failure.
// The caller consumes the returned status; querying does not clear the failure.
iree_status_t loom_serve_recognition_stream_status(
    const loom_serve_recognition_stream_t* stream);

// Borrows the finalized document after successful completion, or NULL before
// then or when the request did not select transcripts. Its views remain
// immutable until close, including after service/model destruction, and need
// no feedback-buffer copy.
const loom_serve_recognition_document_t* loom_serve_recognition_stream_document(
    const loom_serve_recognition_stream_t* stream);

// Consumes the caller handle immediately. Accepted batch membership survives
// cancellation, and source release/row reuse wait for a retired model cut.
void loom_serve_recognition_stream_close(
    loom_serve_recognition_stream_t* stream);

// Seals admission and withdraws offers, cancelling unfinished streams without
// consuming caller ownership. Accepted callbacks still require progress.
void loom_serve_recognition_stream_service_stop(
    loom_serve_recognition_stream_service_t* service);

// Progresses one bounded owner turn of accepted work, delivery, and mandatory
// release without starting new preparation or source selection. Input
// starvation, output pressure, and recoverable capacity denial are non-errors.
// A true progress result asks the owner to rotate and return; false awaits a
// wake, a caller write/consume/close, or a policy grant. STOPPING seals
// admission and drains callbacks. Errors retain accepted model handles for
// teardown; failed readiness is not actual retirement. Refreshes input/consumer
// facts on this serialized owner; cross-thread observations use the retained
// control.
iree_status_t loom_serve_recognition_stream_service_progress(
    loom_serve_recognition_stream_service_t* service,
    loom_runtime_service_flags_t flags, bool* out_progress);

// Accepts the current keyed opportunity, or returns a zero serial when stale.
// A source-selection unit may await more input without submitting a batch;
// published device participation is not a promise of numerical work.
iree_status_t loom_serve_recognition_stream_service_try_schedule(
    loom_serve_recognition_stream_service_t* service,
    loom_runtime_service_offer_key_t offer, uint64_t choice_ordinal,
    loom_runtime_service_flags_t flags,
    loom_runtime_service_operation_key_t* out_operation);

// Accepted CPU/device callbacks still borrow service storage. Live streams
// alone are not pending callbacks; active includes their resident source state.
bool loom_serve_recognition_stream_service_is_pending(
    const loom_serve_recognition_stream_service_t* service);
bool loom_serve_recognition_stream_service_is_active(
    const loom_serve_recognition_stream_service_t* service);

// Frees after stop/close and accepted callbacks retire. Model-owned PCM and
// accepted device resources remain with model teardown on terminal failure.
// This does not wait for a device or destroy the borrowed model. NULL is valid.
void loom_serve_recognition_stream_service_free(
    loom_serve_recognition_stream_service_t* service);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_RECOGNITION_STREAM_H_
