// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_HTTP_LOGS_H_
#define EXPERIMENTAL_LOOM_SERVE_HTTP_LOGS_H_

#include "loom/serve/telemetry/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validates an external opaque cursor against the selected stream. Empty starts
// at zero; malformed/ahead cursors fail with INVALID_ARGUMENT and a different
// stream incarnation with FAILED_PRECONDITION. Later owner turns carry the
// returned position directly and verify the selected stream's lifetime.
iree_status_t loom_serve_http_logs_resume(
    const loom_serve_telemetry_stream_t* stream, iree_string_view_t cursor,
    uint64_t* out_position);

typedef enum loom_serve_http_logs_flag_bits_e {
  LOOM_SERVE_HTTP_LOGS_FLAG_NONE = 0,
  // Includes HTTP response headers before the first batch.
  LOOM_SERVE_HTTP_LOGS_FLAG_HEADERS = 1u << 0,
  // One SSE telemetry event, with JSONL data lines and the next cursor as id.
  LOOM_SERVE_HTTP_LOGS_FLAG_SSE = 1u << 1,
} loom_serve_http_logs_flag_bits_t;
typedef uint32_t loom_serve_http_logs_flags_t;

// Copies at most 64 retained records plus metadata/current counters/cursor into
// owned output. Position belongs to this reader and advances only on success.
// It was validated against this stream's identity at the external boundary;
// the owner preserves that identity before subsequent calls. No record slot
// or counter pointer escapes. NULL counters reports unavailable, not zero.
iree_status_t loom_serve_http_logs_format(
    const loom_serve_telemetry_stream_t* stream,
    const loom_serve_telemetry_counters_t* counters, iree_string_view_t model,
    loom_serve_http_logs_flags_t flags, uint64_t* position,
    iree_string_builder_t* output);

// Renders one bounded JSONL poll into owned response storage, never retaining
// producer slots. cursor is empty for a new reader or the previous response's
// opaque cursor. Malformed/ahead cursors return INVALID_ARGUMENT; a different
// stream incarnation returns FAILED_PRECONDITION. The caller owns the model's
// serialized turn while copying counters; counters are not replayed from
// events. On success output contains a complete HTTP response with at most 64
// retained events plus stream/gap/current-counter/cursor records.
// An empty model name selects server scope. A NULL counters pointer omits the
// current snapshot and reports counters_available=false, not invented zeros.
iree_status_t loom_serve_http_logs(
    const loom_serve_telemetry_stream_t* stream,
    const loom_serve_telemetry_counters_t* counters, iree_string_view_t model,
    iree_string_view_t cursor, iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // EXPERIMENTAL_LOOM_SERVE_HTTP_LOGS_H_
