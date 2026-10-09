// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_HTTP_LOGS_H_
#define EXPERIMENTAL_LOOM_SERVE_HTTP_LOGS_H_

#include "experimental/loom_serve/telemetry/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

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
