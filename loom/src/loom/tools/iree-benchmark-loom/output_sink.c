// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/output_sink.h"

#include <string.h>

iree_status_t iree_benchmark_loom_output_sink_initialize(
    iree_benchmark_loom_output_format_t format,
    iree_string_view_t results_output_path, iree_allocator_t allocator,
    iree_benchmark_loom_output_sink_t* out_sink) {
  memset(out_sink, 0, sizeof(*out_sink));
  out_sink->format = format;
  switch (format) {
    case IREE_BENCHMARK_LOOM_OUTPUT_FORMAT_SNAPSHOT: {
      IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_sink_initialize(
          allocator, &out_sink->snapshot_sink));
      out_sink->snapshot_sink_initialized = true;
      iree_benchmark_loom_snapshot_event_sink_initialize(
          &out_sink->snapshot_sink, &out_sink->event_sink);
      return iree_ok_status();
    }
    case IREE_BENCHMARK_LOOM_OUTPUT_FORMAT_JSONL: {
      IREE_RETURN_IF_ERROR(iree_benchmark_loom_jsonl_sink_initialize(
          results_output_path, allocator, &out_sink->jsonl_sink));
      out_sink->jsonl_sink_initialized = true;
      iree_benchmark_loom_jsonl_event_sink_initialize(
          &out_sink->jsonl_sink, &out_sink->jsonl_event_sink,
          &out_sink->event_sink);
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported benchmark output format %d",
                              (int)format);
  }
}

iree_status_t iree_benchmark_loom_output_sink_flush(
    iree_benchmark_loom_output_sink_t* sink,
    iree_string_view_t results_output_path) {
  switch (sink->format) {
    case IREE_BENCHMARK_LOOM_OUTPUT_FORMAT_SNAPSHOT:
      return iree_benchmark_loom_snapshot_sink_write(&sink->snapshot_sink,
                                                     results_output_path);
    case IREE_BENCHMARK_LOOM_OUTPUT_FORMAT_JSONL:
      return iree_benchmark_loom_jsonl_sink_close(&sink->jsonl_sink);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported benchmark output format %d",
                              (int)sink->format);
  }
}

void iree_benchmark_loom_output_sink_deinitialize(
    iree_benchmark_loom_output_sink_t* sink) {
  if (sink->jsonl_sink_initialized) {
    iree_benchmark_loom_jsonl_event_sink_deinitialize(&sink->jsonl_event_sink);
  }
  if (sink->snapshot_sink_initialized) {
    iree_benchmark_loom_snapshot_sink_deinitialize(&sink->snapshot_sink);
  }
  if (sink->jsonl_sink_initialized) {
    iree_benchmark_loom_jsonl_sink_deinitialize(&sink->jsonl_sink);
  }
  *sink = (iree_benchmark_loom_output_sink_t){0};
}
