// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_TELEMETRY_STREAM_H_
#define EXPERIMENTAL_LOOM_SERVE_TELEMETRY_STREAM_H_

#include "experimental/loom_serve/telemetry/record.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

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
