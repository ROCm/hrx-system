// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_TEXT_TELEMETRY_H_
#define EXPERIMENTAL_LOOM_SERVE_TEXT_TELEMETRY_H_

#include "experimental/loom_serve/telemetry/stream.h"
#include "experimental/loom_serve/text/epoch.h"
#include "experimental/loom_serve/text/heartbeat.h"
#include "experimental/loom_serve/text/session.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_text_receipt_t {
  // Monotonic identity assigned exactly once at protocol receipt.
  uint64_t serial;
  // Host arrival time before validation, queuing or device admission.
  iree_time_t arrival_time;
} loom_serve_text_receipt_t;

// Updated only on serialized service turns, never by CPU borrowers or
// callbacks.
typedef struct loom_serve_text_telemetry_t {
  // Bounded history; consumers never pin or backpressure records.
  loom_serve_telemetry_stream_t* stream;
  // False when the registration lends its history through service retirement.
  bool owns_stream;
  // Always-on model totals independent of heartbeat interval and readers.
  loom_serve_telemetry_counters_t counters;
  // Monotonic request identity independent of TCP connection identity.
  uint64_t next_serial;
  // This model's terminal failure, distinct from another model's shutdown.
  iree_status_code_t failure_code;
} loom_serve_text_telemetry_t;

// Borrows an existing registration stream or creates a private bounded stream
// when NULL. Only a private stream publishes a start here. The service destroys
// an owned stream after retiring its heartbeat and all producer/consumer uses.
iree_status_t loom_serve_text_telemetry_initialize(
    loom_serve_telemetry_stream_t* stream,
    loom_serve_text_telemetry_t* telemetry, iree_allocator_t allocator);

// Counts a protocol receipt exactly once, before validation and admission.
loom_serve_text_receipt_t loom_serve_text_telemetry_begin(
    loom_serve_text_telemetry_t* telemetry);

// Records the final protocol outcome; status is borrowed, not consumed.
void loom_serve_text_telemetry_end(loom_serve_text_telemetry_t* telemetry,
                                   loom_serve_text_receipt_t receipt,
                                   loom_serve_telemetry_outcome_t outcome,
                                   uint64_t input_tokens,
                                   iree_status_t borrowed_status);

// Diagnostic ownership boundary: copies bounded detail and consumes status.
void loom_serve_text_telemetry_diagnose(loom_serve_text_telemetry_t* telemetry,
                                        uint64_t request, iree_status_t status);

// Captures a model failure while its original status propagates normally.
void loom_serve_text_telemetry_fail(loom_serve_text_telemetry_t* telemetry,
                                    iree_status_t borrowed_status);

// Updates session/request outcomes and publishes lifetime snapshots before
// their host state is retired. Does not alter row, request or HTTP ownership.
void loom_serve_text_telemetry_request_end(
    loom_serve_text_telemetry_t* telemetry, loom_serve_text_session_t* session,
    loom_serve_telemetry_outcome_t outcome);

// Publishes final session totals and clears only its telemetry lifetime.
void loom_serve_text_telemetry_session_end(
    loom_serve_text_telemetry_t* telemetry, loom_serve_text_session_t* session);

// Accounts completed model progress at every lifetime and publishes span/epoch
// records plus the heartbeat's completed-work snapshot. Model results remain
// immutable; request phase, output translation and retirement belong to the
// service. The caller publishes each successfully completed cohort once.
void loom_serve_text_telemetry_epoch(
    loom_serve_text_telemetry_t* telemetry,
    loom_serve_text_heartbeat_t* heartbeat,
    const loom_serve_text_epoch_options_t* options,
    const loom_serve_text_epoch_t* cohort, loom_serve_text_session_t* sessions);

// Owned arguments published only by the serialized service owner. CPU jobs
// return measurements through their existing completion handoff. No record
// borrows request strings, statuses or mutable session storage.
typedef struct loom_serve_text_admission_record_t {
  // Session lifetime ID, zero until admission assigns one.
  uint64_t session;
  // Reusable device row, not a session identity.
  uint64_t row;
  // Consumed prefix preceding the appended input.
  uint64_t retained_tokens;
  // Input prepared for this request, including a retained pending token.
  uint64_t appended_tokens;
  // Page-rounded completion credit.
  uint64_t reservation_tokens;
  // Cold checkpoint bytes restored by this admission.
  uint64_t restored_bytes;
  // Requested output bound.
  uint64_t max_tokens;
  // Host time preparing the input, in nanoseconds.
  uint64_t prepare_ns;
  // Host time awaiting admission, in nanoseconds.
  uint64_t queue_ns;
  // Pool-wide outstanding completion credit at admission.
  uint64_t pool_reserved_tokens;
  // Pool-wide unique resident positions at admission.
  uint64_t pool_resident_tokens;
} loom_serve_text_admission_record_t;

typedef struct loom_serve_text_request_record_t {
  // Session lifetime receiving this request.
  uint64_t session;
  // Completed work and final outcome for this request alone.
  loom_serve_telemetry_counters_t counters;
  // Consumed prefix preceding this request.
  uint64_t retained_tokens;
  // Position after the last completed device work.
  uint64_t position;
  // Host time before admission, in nanoseconds.
  uint64_t queue_ns;
  // Admission to first decoded token; zero when no token was processed.
  uint64_t first_token_ns;
  // Receipt to owner retirement, in nanoseconds.
  uint64_t duration_ns;
  // Owned terminal wire reason; empty when interrupted before terminal output.
  char finish_reason[16];
} loom_serve_text_request_record_t;

typedef struct loom_serve_text_session_record_t {
  // Stable lifetime ID; a recycled device row starts a new lifetime.
  uint64_t session;
  // Reusable physical row index.
  uint64_t row;
  // Validated, owned client cache key; empty for anonymous requests.
  char name[65];
} loom_serve_text_session_record_t;

typedef struct loom_serve_text_session_end_record_t {
  // Retiring session lifetime ID.
  uint64_t session;
  // Cumulative requests and completed work for this lifetime.
  loom_serve_telemetry_counters_t counters;
  // Final consumed position.
  uint64_t position;
  // Row-owned positions at the last completed endpoint; may include sharing.
  uint64_t resident_tokens;
} loom_serve_text_session_end_record_t;

typedef struct loom_serve_text_epoch_record_t {
  // First issued epoch ID of this cohort.
  uint64_t epoch;
  // Completed cohort span count.
  uint64_t spans;
  // Selected bucket index.
  uint64_t shape;
  // Compiled token capacity of the bucket.
  uint64_t token_capacity;
  // Compiled independent span capacity.
  uint64_t span_capacity;
  // Prompt tokens consumed, excluding retained history.
  uint64_t prefill_tokens;
  // Generated tokens consumed by the target.
  uint64_t decode_tokens;
  // Selected tokens including EOS, before response decoding.
  uint64_t output_tokens;
  // Completed target weight traversals.
  uint64_t traversals;
  // Host-observed submission to readiness duration, not GPU timestamps.
  uint64_t duration_ns;
  // Issued packed epochs including device-fed continuations.
  uint64_t device_epochs;
  // Known inputs scheduled in the continuation.
  uint64_t continued_prefill_tokens;
  // Completed verification steps across all spans.
  uint64_t verifications;
  // Draft inputs accepted by the target, excluding pending anchors.
  uint64_t accepted_draft_inputs;
  // One loom_serve_text_schedule_mode_t.
  uint64_t schedule_mode;
  // One loom_serve_text_packing_mode_t.
  uint64_t packing_mode;
} loom_serve_text_epoch_record_t;

typedef struct loom_serve_text_span_record_t {
  // Containing cohort's first epoch ID.
  uint64_t epoch;
  // Stable session lifetime, independent of the reused row.
  uint64_t session;
  // Device row index.
  uint64_t row;
  // Consumed position before submission.
  uint64_t position;
  // Planned input count including speculative candidates.
  uint64_t tokens;
  // Actual target-consumed inputs.
  uint64_t consumed_tokens;
  // Selected outputs including EOS.
  uint64_t output_tokens;
  // Speculative output credit; zero for ordinary prefill/decode.
  uint64_t output_limit;
  // Non-draft inputs actually consumed.
  uint64_t known_tokens;
  // Completed verifications for this span.
  uint64_t verification_epochs;
  // Whether the span was consuming prompt input at submission.
  uint64_t prefill;
  // Whether the first traversal requested token selection.
  uint64_t select;
} loom_serve_text_span_record_t;

typedef struct loom_serve_text_checkpoint_record_t {
  // Owned validated checkpoint name.
  char name[65];
  // Exact consumed endpoint.
  uint64_t position;
  // Host snapshot payload retained after the operation.
  uint64_t host_bytes;
} loom_serve_text_checkpoint_record_t;

typedef struct loom_serve_text_trim_record_t {
  // Host-observed maintenance duration, in nanoseconds.
  uint64_t duration_ns;
  // Relocated logical blocks.
  uint64_t moved_blocks;
  // Device bytes copied by compaction.
  uint64_t copied_bytes;
  // Physical bytes returned at this cut.
  uint64_t released_bytes;
  // Mutable-state backing remaining after maintenance.
  uint64_t committed_bytes;
} loom_serve_text_trim_record_t;

extern const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_PREPARED;
extern const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_ADMIT;
extern const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_REQUEST_STATS;
extern const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_SESSION_BEGIN;
extern const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_SESSION_END;
extern const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_EPOCH;
extern const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_SPAN;
extern const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_CHECKPOINT_PIN;
extern const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_CHECKPOINT_SUSPEND;
extern const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_CHECKPOINT_RELEASE;
extern const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_TRIM;

// Copies a statically sized payload into the fixed record. Each schema's size
// is checked where it is defined, not revalidated on the publication hot path.
void loom_serve_text_record(loom_serve_telemetry_stream_t* stream,
                            const loom_serve_telemetry_event_t* event,
                            uint64_t request, const void* payload,
                            iree_host_size_t payload_size);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // EXPERIMENTAL_LOOM_SERVE_TEXT_TELEMETRY_H_
