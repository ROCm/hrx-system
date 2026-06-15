// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Owns the concrete result-output sink selected for a benchmark run.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_OUTPUT_SINK_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_OUTPUT_SINK_H_

#include "iree/base/api.h"
#include "loom/tools/iree-benchmark-loom/event.h"
#include "loom/tools/iree-benchmark-loom/output.h"
#include "loom/tools/iree-benchmark-loom/snapshot.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_benchmark_loom_output_sink_t {
  // Selected concrete result format.
  iree_benchmark_loom_output_format_t format;
  // JSONL row stream used when |format| is JSONL.
  iree_benchmark_loom_jsonl_sink_t jsonl_sink;
  // True when |jsonl_sink| owns initialized state.
  bool jsonl_sink_initialized;
  // JSONL event adapter layered over |jsonl_sink|.
  iree_benchmark_loom_jsonl_event_sink_t jsonl_event_sink;
  // Snapshot aggregator used when |format| is snapshot.
  iree_benchmark_loom_snapshot_sink_t snapshot_sink;
  // True when |snapshot_sink| owns initialized state.
  bool snapshot_sink_initialized;
  // Event sink exposed to benchmark planning and execution code.
  iree_benchmark_loom_event_sink_t event_sink;
} iree_benchmark_loom_output_sink_t;

// Initializes the result-output sink and exposes its typed event sink.
iree_status_t iree_benchmark_loom_output_sink_initialize(
    iree_benchmark_loom_output_format_t format,
    iree_string_view_t results_output_path, iree_allocator_t allocator,
    iree_benchmark_loom_output_sink_t* out_sink);

// Writes or closes any deferred result output.
iree_status_t iree_benchmark_loom_output_sink_flush(
    iree_benchmark_loom_output_sink_t* sink,
    iree_string_view_t results_output_path);

// Releases all state owned by |sink|.
void iree_benchmark_loom_output_sink_deinitialize(
    iree_benchmark_loom_output_sink_t* sink);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_OUTPUT_SINK_H_
