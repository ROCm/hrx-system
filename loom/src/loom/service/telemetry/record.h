// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_TELEMETRY_RECORD_H_
#define LOOM_SERVICE_TELEMETRY_RECORD_H_

#include "loom/runtime/telemetry/record.h"

#ifdef __cplusplus
extern "C" {
#endif

// Cumulative model work, maintained by the serialized owner independently of
// record retention. HTTP completion means carrier acceptance, not peer receipt.
typedef struct loom_service_telemetry_counters_t {
  // Requests reaching this model's protocol, including immediate rejections.
  uint64_t requests;
  // Successful responses published to their native or transport consumer.
  uint64_t completed;
  // Requests rejected by protocol, source validation or capacity admission.
  uint64_t rejected;
  // Requests abandoned by a peer or orderly server shutdown.
  uint64_t cancelled;
  // Requests terminated by an allocation, transport or execution failure.
  uint64_t failed;
  // Tokens in completed model work, including work for disconnected peers.
  uint64_t input_tokens;
  // Published model output tokens, independent of response encoding.
  uint64_t output_tokens;
  // Completed device invocations or cohorts, not HTTP requests.
  uint64_t batches;
} loom_service_telemetry_counters_t;

// Native field order shared by snapshots and nested service counter objects.
enum { LOOM_SERVICE_TELEMETRY_COUNTER_FIELD_COUNT = 8 };

// Expands trusted owner counters into caller-owned consumer views.
void loom_service_telemetry_counters_read_fields(
    const loom_service_telemetry_counters_t* counters,
    loom_runtime_telemetry_field_t
        fields[LOOM_SERVICE_TELEMETRY_COUNTER_FIELD_COUNT]);

extern const loom_runtime_telemetry_event_t LOOM_SERVICE_TELEMETRY_STREAM_BEGIN;
extern const loom_runtime_telemetry_event_t LOOM_SERVICE_TELEMETRY_SNAPSHOT;
extern const loom_runtime_telemetry_event_t
    LOOM_SERVICE_TELEMETRY_REQUEST_BEGIN;
extern const loom_runtime_telemetry_event_t LOOM_SERVICE_TELEMETRY_FIRST_OUTPUT;
extern const loom_runtime_telemetry_event_t LOOM_SERVICE_TELEMETRY_REQUEST_END;
extern const loom_runtime_telemetry_event_t LOOM_SERVICE_TELEMETRY_BATCH_SUBMIT;
extern const loom_runtime_telemetry_event_t LOOM_SERVICE_TELEMETRY_BATCH_END;
extern const loom_runtime_telemetry_event_t LOOM_SERVICE_TELEMETRY_ERROR;

typedef enum loom_service_telemetry_outcome_e {
  LOOM_SERVICE_TELEMETRY_COMPLETED = 0,
  LOOM_SERVICE_TELEMETRY_REJECTED,
  LOOM_SERVICE_TELEMETRY_CANCELLED,
  LOOM_SERVICE_TELEMETRY_FAILED,
} loom_service_telemetry_outcome_t;

// Stable wire label for a trusted terminal outcome.
const char* loom_service_telemetry_outcome_name(
    loom_service_telemetry_outcome_t outcome);

// Counts one owner's terminal request outcome independently of event retention.
void loom_service_telemetry_count_outcome(
    loom_service_telemetry_counters_t* counters,
    loom_service_telemetry_outcome_t outcome);

// Owned, bounded diagnostic bytes. No pointer into a request/status escapes.
typedef struct loom_service_telemetry_diagnostic_t {
  // Captured IREE status code; zero when the outcome has no failure status.
  uint64_t code;
  // Number of copied message bytes, excluding any terminator.
  uint64_t length;
  // Nonzero when the primary status message exceeded the copied storage.
  uint64_t truncated;
  // Primary status message, without formatted backtraces.
  char message[80];
} loom_service_telemetry_diagnostic_t;

// Copied service snapshot, independent of the scheduler that captures it.
typedef struct loom_service_telemetry_snapshot_record_t {
  // Cumulative work, independent of stream retention or consumers.
  loom_service_telemetry_counters_t counters;
  // Requests waiting outside the active cohort.
  uint64_t queued;
  // Requests owned by the active cohort, including cancelled peers.
  uint64_t active;
  // Owned NUL-terminated scheduler phase label, independent of model names.
  char phase[16];
} loom_service_telemetry_snapshot_record_t;

// Request outcome captured before its storage is retired.
typedef struct loom_service_telemetry_end_record_t {
  // One loom_service_telemetry_outcome_t.
  uint64_t outcome;
  // Receipt through final owner retirement, in nanoseconds.
  uint64_t duration_ns;
  // This request's completed input token work.
  uint64_t input_tokens;
  // Final status detail, bounded and independently owned.
  loom_service_telemetry_diagnostic_t diagnostic;
} loom_service_telemetry_end_record_t;

// First meaningful model output accepted by its consumer.
typedef struct loom_service_telemetry_first_output_record_t {
  // Protocol receipt to output acceptance, not peer receipt, in nanoseconds.
  uint64_t duration_ns;
} loom_service_telemetry_first_output_record_t;

// Device invocation boundary, independent of the number of clients.
typedef struct loom_service_telemetry_batch_record_t {
  // Monotonic invocation ID within the stream.
  uint64_t id;
  // Requests captured by the submitted invocation.
  uint64_t requests;
} loom_service_telemetry_batch_record_t;

// Copies the primary message without allocating or consuming the status.
void loom_service_telemetry_diagnostic_capture(
    const iree_status_t status,
    loom_service_telemetry_diagnostic_t* out_diagnostic);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVICE_TELEMETRY_RECORD_H_
