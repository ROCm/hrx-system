// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured diagnostic capture used by parse, verify, and candidate compile.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_DIAGNOSTICS_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_DIAGNOSTICS_H_

#include "iree/base/api.h"
#include "loom/error/diagnostic.h"
#include "loom/tools/iree-benchmark-loom/model.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes |capture| to own a JSON diagnostic payload builder.
void iree_benchmark_loom_diagnostic_capture_initialize(
    iree_allocator_t allocator,
    iree_benchmark_loom_diagnostic_capture_t* capture);

// Releases storage owned by |capture|.
void iree_benchmark_loom_diagnostic_capture_deinitialize(
    iree_benchmark_loom_diagnostic_capture_t* capture);

// Returns the captured JSON array body without surrounding brackets.
iree_string_view_t iree_benchmark_loom_diagnostic_capture_json(
    const iree_benchmark_loom_diagnostic_capture_t* capture);

// Writes the captured diagnostics as a complete JSON array.
iree_status_t iree_benchmark_loom_write_diagnostic_array_json(
    const iree_benchmark_loom_diagnostic_capture_t* capture,
    loom_output_stream_t* stream);

// Diagnostic sink callback that appends one structured JSON diagnostic.
iree_status_t iree_benchmark_loom_diagnostic_capture_sink(
    void* user_data, const loom_diagnostic_t* diagnostic);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_DIAGNOSTICS_H_
