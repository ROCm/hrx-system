// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_TELEMETRY_STREAM_H_
#define EXPERIMENTAL_LOOM_SERVE_TELEMETRY_STREAM_H_

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

typedef enum loom_serve_telemetry_event_e {
  LOOM_SERVE_TELEMETRY_STREAM_BEGIN = 0,
  LOOM_SERVE_TELEMETRY_SNAPSHOT,
  LOOM_SERVE_TELEMETRY_REQUEST_BEGIN,
  LOOM_SERVE_TELEMETRY_REQUEST_END,
  LOOM_SERVE_TELEMETRY_BATCH_SUBMIT,
  LOOM_SERVE_TELEMETRY_BATCH_END,
  LOOM_SERVE_TELEMETRY_ERROR,
} loom_serve_telemetry_event_t;

typedef enum loom_serve_telemetry_outcome_e {
  LOOM_SERVE_TELEMETRY_COMPLETED = 0,
  LOOM_SERVE_TELEMETRY_REJECTED,
  LOOM_SERVE_TELEMETRY_CANCELLED,
  LOOM_SERVE_TELEMETRY_FAILED,
} loom_serve_telemetry_outcome_t;

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
typedef struct loom_serve_telemetry_record_t {
  // Monotonic host timestamp in nanoseconds, not a device-clock timestamp.
  iree_time_t time_ns;
  // Lifetime ID increasing within the stream; zero for model-wide records.
  uint64_t request;
  // One loom_serve_telemetry_event_t, selecting the payload below.
  uint64_t event;
  // Owned event arguments; unused bytes are initialized by the producer.
  union {
    // Model snapshot used by stream begin and periodic observation.
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
  } data;
} loom_serve_telemetry_record_t;

// Copies the primary message without allocating or consuming the status.
void loom_serve_telemetry_diagnostic_capture(
    iree_status_t status, loom_serve_telemetry_diagnostic_t* out_diagnostic);

typedef struct loom_serve_telemetry_stream_t loom_serve_telemetry_stream_t;

// Distinguishes stream lifetimes, including recreation under the same model
// name. This is a process observation identity, not a durable storage key.
typedef struct loom_serve_telemetry_identity_t {
  // Monotonic construction time, also separating ordinary process restarts.
  uint64_t created_ns;
  // Process-wide construction ordinal, distinguishing equal clock ticks.
  uint64_t ordinal;
} loom_serve_telemetry_identity_t;

// Allocates a power-of-two record capacity. Requires lock-free 64-bit atomics.
// One externally serialized producer may migrate between task workers. Any
// number of concurrent readers own independent absolute cursors, starting at
// zero. Storage survives all producer/reader accesses; no reader retains a
// slot.
iree_status_t loom_serve_telemetry_stream_create(
    iree_host_size_t capacity, loom_serve_telemetry_stream_t** out_stream,
    iree_allocator_t allocator);
void loom_serve_telemetry_stream_destroy(loom_serve_telemetry_stream_t* stream);

loom_serve_telemetry_identity_t loom_serve_telemetry_stream_identity(
    const loom_serve_telemetry_stream_t* stream);

// Next published absolute position. Physical indices wrap; logical positions
// never do. The stream lifetime is bounded to fewer than 2^63-1 records.
uint64_t loom_serve_telemetry_stream_head(
    const loom_serve_telemetry_stream_t* stream);
iree_host_size_t loom_serve_telemetry_stream_capacity(
    const loom_serve_telemetry_stream_t* stream);

// Always advances publication; a slow or absent reader cannot delay the writer.
void loom_serve_telemetry_stream_append(
    loom_serve_telemetry_stream_t* stream,
    const loom_serve_telemetry_record_t* record);

typedef enum loom_serve_telemetry_read_e {
  // A whole record was copied and cursor advanced by one.
  LOOM_SERVE_TELEMETRY_READ_RECORD = 0,
  // No committed record exists at cursor yet; cursor is unchanged.
  LOOM_SERVE_TELEMETRY_READ_EMPTY,
  // The reader fell behind. Cursor advances to the oldest retained position;
  // the interval from its previous value to the new value was missed.
  LOOM_SERVE_TELEMETRY_READ_GAP,
  // Overwrite overlapped this one attempt. Cursor is unchanged; retry on a
  // subsequent consumer poll, without spinning or retaining producer storage.
  LOOM_SERVE_TELEMETRY_READ_BUSY,
} loom_serve_telemetry_read_t;

// Makes one bounded attempt. Output is valid only for READ_RECORD. A late
// reader starting at zero receives GAP when the stream's start record is no
// longer retained. Reading never changes the stream, acknowledges or clears
// records.
loom_serve_telemetry_read_t loom_serve_telemetry_stream_read(
    const loom_serve_telemetry_stream_t* stream, uint64_t* cursor,
    loom_serve_telemetry_record_t* out_record);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // EXPERIMENTAL_LOOM_SERVE_TELEMETRY_STREAM_H_
