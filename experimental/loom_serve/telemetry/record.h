// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_TELEMETRY_RECORD_H_
#define EXPERIMENTAL_LOOM_SERVE_TELEMETRY_RECORD_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Cumulative model work, maintained by the serialized owner independently of
// record retention. HTTP completion means carrier acceptance, not peer receipt.
typedef struct loom_serve_telemetry_counters_t {
  // Requests reaching this model's protocol, including immediate rejections.
  uint64_t requests;
  // Successful responses copied into the HTTP carrier.
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
} loom_serve_telemetry_counters_t;

typedef struct loom_serve_telemetry_record_t loom_serve_telemetry_record_t;

// Immutable event schema with process lifetime. Renderers run on copied records
// in consumers, never during publication. They append JSON object members,
// including their leading commas, and may fail only to allocate output storage.
typedef struct loom_serve_telemetry_event_t {
  // Stable event name, independent of deployment/model identifiers.
  const char* name;
  // Consumer-only formatter; NULL for an event with no payload.
  iree_status_t (*format)(const loom_serve_telemetry_record_t* record,
                          iree_string_builder_t* output);
} loom_serve_telemetry_event_t;

extern const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_STREAM_BEGIN;
extern const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_SNAPSHOT;
extern const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_REQUEST_BEGIN;
extern const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_REQUEST_END;
extern const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_BATCH_SUBMIT;
extern const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_BATCH_END;
extern const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_ERROR;

typedef enum loom_serve_telemetry_outcome_e {
  LOOM_SERVE_TELEMETRY_COMPLETED = 0,
  LOOM_SERVE_TELEMETRY_REJECTED,
  LOOM_SERVE_TELEMETRY_CANCELLED,
  LOOM_SERVE_TELEMETRY_FAILED,
} loom_serve_telemetry_outcome_t;

// Counts one owner's terminal request outcome independently of event retention.
void loom_serve_telemetry_count_outcome(
    loom_serve_telemetry_counters_t* counters,
    loom_serve_telemetry_outcome_t outcome);

// Owned, bounded diagnostic bytes. No pointer into a request/status escapes.
typedef struct loom_serve_telemetry_diagnostic_t {
  // Captured IREE status code; zero when the outcome has no failure status.
  uint64_t code;
  // Number of copied message bytes, excluding any terminator.
  uint64_t length;
  // Nonzero when the primary status message exceeded the copied storage.
  uint64_t truncated;
  // Primary status message, without formatted backtraces.
  char message[80];
} loom_serve_telemetry_diagnostic_t;

// Fixed-size native record. Only the tagged member is meaningful; append copies
// all bytes and never formats, allocates or consults a reader. This is an
// in-process representation, not a portable binary wire format.
struct loom_serve_telemetry_record_t {
  // Monotonic host timestamp in nanoseconds, not a device-clock timestamp.
  iree_time_t time_ns;
  // Lifetime ID increasing within the stream; zero for model-wide records.
  uint64_t request;
  // Process-lifetime event schema, safe to dereference after a complete copy.
  const loom_serve_telemetry_event_t* event;
  // Owned event arguments; unused bytes are initialized by the producer.
  union {
    // Model snapshot used by periodic observation.
    struct {
      // Cumulative work, independent of ring retention or consumers.
      loom_serve_telemetry_counters_t counters;
      // Requests waiting outside the active cohort.
      uint64_t queued;
      // Requests owned by the active cohort, including cancelled peers.
      uint64_t active;
      // Owned NUL-terminated scheduler phase label, independent of model names.
      char phase[16];
    } snapshot;
    // Request end, captured before the request's storage is retired.
    struct {
      // One loom_serve_telemetry_outcome_t.
      uint64_t outcome;
      // Time from protocol admission through final owner retirement.
      uint64_t duration_ns;
      // This request's completed input token work.
      uint64_t input_tokens;
      // Final status detail, bounded and independently owned.
      loom_serve_telemetry_diagnostic_t diagnostic;
    } end;
    // Device invocation boundary, independent of the number of clients.
    struct {
      // Monotonic invocation ID within this stream.
      uint64_t id;
      // Requests captured by the submitted invocation.
      uint64_t requests;
    } batch;
    // An error consumed at a diagnostic boundary or also propagated normally.
    loom_serve_telemetry_diagnostic_t error;
    // Modality-owned payload copied by value; schema renderers decode via
    // memcpy.
    uint64_t words[16];
  } data;
};

// Copies the primary message without allocating or consuming the status.
void loom_serve_telemetry_diagnostic_capture(
    iree_status_t status, loom_serve_telemetry_diagnostic_t* out_diagnostic);

// Consumer-side JSON formatting of an owned record and its absolute position.
// Model code may define additional static event schemas without HTTP coupling.
iree_status_t loom_serve_telemetry_record_format(
    uint64_t position, const loom_serve_telemetry_record_t* record,
    iree_string_builder_t* output);
iree_status_t loom_serve_telemetry_counters_format(
    const loom_serve_telemetry_counters_t* counters,
    iree_string_builder_t* output);
iree_status_t loom_serve_telemetry_diagnostic_format(
    const loom_serve_telemetry_diagnostic_t* diagnostic,
    iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // EXPERIMENTAL_LOOM_SERVE_TELEMETRY_RECORD_H_
