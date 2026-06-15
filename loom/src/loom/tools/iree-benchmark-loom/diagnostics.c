// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/diagnostics.h"

#include "loom/error/json_sink.h"
#include "loom/error/renderer.h"

void iree_benchmark_loom_diagnostic_capture_initialize(
    iree_allocator_t allocator,
    iree_benchmark_loom_diagnostic_capture_t* capture) {
  *capture = (iree_benchmark_loom_diagnostic_capture_t){
      .initialized = true,
      .first_diagnostic = true,
  };
  iree_string_builder_initialize(allocator, &capture->output);
  loom_output_stream_for_builder(&capture->output, &capture->stream);
}

void iree_benchmark_loom_diagnostic_capture_deinitialize(
    iree_benchmark_loom_diagnostic_capture_t* capture) {
  if (capture == NULL || !capture->initialized) {
    return;
  }
  iree_string_builder_deinitialize(&capture->output);
  *capture = (iree_benchmark_loom_diagnostic_capture_t){0};
}

iree_string_view_t iree_benchmark_loom_diagnostic_capture_json(
    const iree_benchmark_loom_diagnostic_capture_t* capture) {
  if (!capture || !capture->initialized) {
    return iree_string_view_empty();
  }
  return iree_string_builder_view(&capture->output);
}

iree_status_t iree_benchmark_loom_write_diagnostic_array_json(
    const iree_benchmark_loom_diagnostic_capture_t* capture,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(
      stream, iree_benchmark_loom_diagnostic_capture_json(capture)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_diagnostic_capture_sink(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  iree_benchmark_loom_diagnostic_capture_t* capture =
      (iree_benchmark_loom_diagnostic_capture_t*)user_data;
  if (!capture->first_diagnostic) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&capture->stream, ","));
  }
  IREE_RETURN_IF_ERROR(loom_diagnostic_json_write_object(
      &capture->stream, diagnostic,
      (loom_type_formatter_t){loom_type_format_minimal, NULL}));
  capture->first_diagnostic = false;
  switch (diagnostic->severity) {
    case LOOM_DIAGNOSTIC_ERROR:
      ++capture->error_count;
      break;
    case LOOM_DIAGNOSTIC_WARNING:
      ++capture->warning_count;
      break;
    case LOOM_DIAGNOSTIC_REMARK:
      ++capture->remark_count;
      break;
    default:
      break;
  }
  return iree_ok_status();
}
