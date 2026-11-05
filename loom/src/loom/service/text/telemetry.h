// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_TEXT_TELEMETRY_H_
#define LOOM_SERVICE_TEXT_TELEMETRY_H_

#include "loom/runtime/storage/statistics.h"
#include "loom/runtime/telemetry/stream.h"
#include "loom/service/telemetry/record.h"

#ifdef __cplusplus
extern "C" {
#endif

// Owned arguments published only by the serialized service owner. CPU jobs
// return measurements through their existing completion handoff. No record
// borrows request strings, statuses or mutable session storage.
typedef struct loom_service_text_admission_record_t {
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
  // Arrival-ordered queue position selected for this successful admission.
  uint64_t queue_rank;
  // Successful overtakes of the current oldest request, including this one.
  uint64_t overtakes;
  // Configured successful-overtake bound; zero is strict FIFO.
  uint64_t overtake_budget;
  // Host time preparing the input, in nanoseconds.
  uint64_t prepare_ns;
  // Host time awaiting admission, in nanoseconds.
  uint64_t queue_ns;
  // Pool-wide outstanding completion credit at admission.
  uint64_t pool_reserved_tokens;
  // Pool-wide unique resident positions at admission.
  uint64_t pool_resident_tokens;
} loom_service_text_admission_record_t;

typedef struct loom_service_text_request_record_t {
  // Session lifetime receiving this request.
  uint64_t session;
  // Completed work and final outcome for this request alone.
  loom_service_telemetry_counters_t counters;
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
} loom_service_text_request_record_t;

typedef struct loom_service_text_session_record_t {
  // Stable lifetime ID; a recycled device row starts a new lifetime.
  uint64_t session;
  // Reusable physical row index.
  uint64_t row;
  // Validated, owned client cache key; empty for anonymous requests.
  char name[65];
} loom_service_text_session_record_t;

typedef struct loom_service_text_session_end_record_t {
  // Retiring session lifetime ID.
  uint64_t session;
  // Cumulative requests and completed work for this lifetime.
  loom_service_telemetry_counters_t counters;
  // Final consumed position.
  uint64_t position;
  // Row-owned positions at the last completed endpoint; may include sharing.
  uint64_t resident_tokens;
} loom_service_text_session_end_record_t;

typedef struct loom_service_text_epoch_record_t {
  // First issued epoch ID of this cohort.
  uint64_t epoch;
  // Completed cohort span count.
  uint64_t spans;
  // Selected first-epoch bucket index.
  uint64_t shape;
  // Compiled token capacity of the first-epoch bucket, not the whole cohort.
  uint64_t token_capacity;
  // Sum of issued packed bucket capacities across the cohort. Zero means
  // unavailable for isolated/matched stages and is exposed as unavailable.
  uint64_t token_slots;
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
  // Packed execution or an isolated/matched control.
  uint8_t schedule_mode;
  // Whether prompt inputs and verifiers may share a cohort.
  uint8_t packing_mode;
  // Candidates after each anchor; multiplied by verifications for totals.
  uint8_t draft_depth;
} loom_service_text_epoch_record_t;

typedef struct loom_service_text_span_record_t {
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
} loom_service_text_span_record_t;

typedef struct loom_service_text_checkpoint_record_t {
  // Owned validated checkpoint name.
  char name[65];
  // Exact consumed endpoint.
  uint64_t position;
  // Host snapshot payload retained after the operation.
  uint64_t host_bytes;
} loom_service_text_checkpoint_record_t;

typedef struct loom_service_text_trim_record_t {
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
} loom_service_text_trim_record_t;

typedef struct loom_service_text_prefix_record_t {
  // Host readiness-registration to owner-observation nanoseconds, not GPU time.
  uint64_t readiness_ns;
  // Configured automatic frontier bound.
  uint64_t capacity;
  // Retained automatic frontiers after completed capture.
  uint64_t endpoints;
} loom_service_text_prefix_record_t;

extern const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_PREPARED;
extern const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_ADMIT;
extern const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_REQUEST_STATS;
extern const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_SESSION_BEGIN;
extern const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_SESSION_END;
extern const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_EPOCH;
extern const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_SPAN;
extern const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_CHECKPOINT_PIN;
extern const loom_runtime_telemetry_event_t
    LOOM_SERVICE_TEXT_CHECKPOINT_SUSPEND;
extern const loom_runtime_telemetry_event_t
    LOOM_SERVICE_TEXT_CHECKPOINT_RELEASE;
extern const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_TRIM;
extern const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_PREFIX_CAPTURE;

// Copies a statically sized payload into the fixed record. Each schema's size
// is checked where it is defined, not revalidated on the publication hot path.
void loom_service_text_record(loom_runtime_telemetry_stream_t* stream,
                              const loom_runtime_telemetry_event_t* event,
                              uint64_t request, const void* payload,
                              iree_host_size_t payload_size);

// Related records share a timestamp but remain independently readable after
// overwrite. Shared device memory is labeled separately from model ownership.
typedef struct loom_service_text_activity_record_t {
  // Host monotonic start of the current phase.
  uint64_t phase_start;
  // Host monotonic time of the last completed epoch.
  uint64_t last_completion;
  // Oldest queued arrival, or zero when the queue is empty.
  uint64_t oldest_queued_arrival;
  // Oldest active arrival, or zero when no row is active.
  uint64_t oldest_active_arrival;
  // Rows consuming prompt input.
  uint64_t prefill_rows;
  // Rows generating tokens.
  uint64_t decode_rows;
  // Rows waiting for peer output credit.
  uint64_t backpressured_rows;
  // Span count of the last issued epoch.
  uint64_t epoch_spans;
  // Token count of the last issued epoch.
  uint64_t epoch_tokens;
  // Allocated resident owners, independent of the latest epoch's span count.
  uint64_t session_capacity;
  // Configured resident ceiling; zero leaves concurrency memory-limited.
  uint64_t max_sessions;
} loom_service_text_activity_record_t;

typedef struct loom_service_text_work_record_t {
  // Issued epochs including device continuations.
  uint64_t issued_epochs;
  // Successfully completed epochs.
  uint64_t completed_epochs;
  // Target weight traversals including isolated spans.
  uint64_t traversals;
  // Consumed prompt tokens excluding retained history.
  uint64_t prefill_tokens;
  // Consumed pending and accepted draft tokens.
  uint64_t decode_tokens;
  // Cumulative host submission-to-readiness nanoseconds.
  uint64_t model_ns;
  // Draft candidates from completed verification.
  uint64_t proposed_tokens;
  // Accepted draft inputs excluding pending anchors.
  uint64_t accepted_inputs;
} loom_service_text_work_record_t;

typedef struct loom_service_text_memory_record_t {
  // Addressable logical token positions.
  uint64_t capacity;
  // Completion credit disjoint from resident positions.
  uint64_t reserved;
  // Unique resident token positions.
  uint64_t resident;
  // Explicitly suspended endpoint payload.
  uint64_t checkpoint_host_bytes;
  // Cold automatic sizing target, independent of active working-set admission.
  uint64_t cache_target_bytes;
  // Selected initial extra declared state and native checkpoint maps.
  uint64_t cache_additional_bytes;
  // Detached idle endpoints, independent of active execution rows.
  struct {
    // Configured retained endpoint bound.
    uint64_t capacity;
    // Owned endpoints, including one possible pending admission exchange.
    uint64_t endpoints;
  } session_cache;
  // Exact token-prefix frontiers, independent of completed session histories.
  struct {
    // Configured frontier bound.
    uint64_t capacity;
    // Owned frontiers, including a pending capture.
    uint64_t endpoints;
  } prefix_cache;
} loom_service_text_memory_record_t;

typedef struct loom_service_text_backing_record_t {
  // Model mutable-state backing excluding weights/workspace.
  loom_runtime_storage_statistics_t state;
  // Independently reclaimable model parameter backing.
  loom_runtime_storage_usage_t parameters;
} loom_service_text_backing_record_t;

extern const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_ACTIVITY;
extern const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_WORK;
extern const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_MEMORY;
extern const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_BACKING;

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVICE_TEXT_TELEMETRY_H_
