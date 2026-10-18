// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/pass/loomc_trace.h"

#include "loom/util/stream.h"
#include "loomc/iree.h"

static loomc_status_t loom_tooling_loomc_trace_write(
    void* user_data, loomc_string_view_t fragment) {
  return loomc_status_from_iree(loom_output_stream_write(
      (loom_output_stream_t*)user_data, iree_string_view_from_loomc(fragment)));
}

static loomc_status_t loom_tooling_loomc_trace_open_artifact(
    void* user_data, const loomc_pass_trace_event_t* event,
    loomc_pass_trace_artifact_t* out_artifact) {
  const iree_string_view_t point = event->point == LOOMC_PASS_TRACE_POINT_BEFORE
                                       ? IREE_SV("before")
                                       : IREE_SV("after");
  loom_output_stream_t* stream = NULL;
  iree_string_view_t reference = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loomc_status_from_iree(loom_tooling_pass_trace_open_artifact(
          (loom_tooling_pass_trace_t*)user_data, event->event_ordinal, point,
          iree_string_view_from_loomc(event->pass_key), &stream, &reference)));
  *out_artifact = (loomc_pass_trace_artifact_t){
      .reference = loomc_string_view_from_iree(reference),
      .sink =
          {
              .write = loom_tooling_loomc_trace_write,
              .user_data = stream,
          },
  };
  return loomc_ok_status();
}

static loomc_status_t loom_tooling_loomc_trace_close_artifact(
    void* user_data, loomc_pass_trace_artifact_t* artifact) {
  (void)artifact;
  return loomc_status_from_iree(loom_tooling_pass_trace_close_artifact(
      (loom_tooling_pass_trace_t*)user_data));
}

iree_status_t loom_tooling_loomc_pass_trace_open_from_flags(
    const loom_tooling_pass_trace_open_options_t* options,
    iree_allocator_t allocator, loom_tooling_loomc_pass_trace_t* out_trace) {
  *out_trace = (loom_tooling_loomc_pass_trace_t){.allocator = allocator};
  loom_tooling_pass_trace_t* output = &out_trace->output;
  IREE_RETURN_IF_ERROR(
      loom_tooling_pass_trace_open_from_flags(options, allocator, output));
  if (!output->enabled) {
    return iree_ok_status();
  }

  const iree_host_size_t filter_count =
      output->dump_before.count + output->dump_after.count;
  if (filter_count) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        allocator, filter_count * sizeof(*out_trace->filters),
        (void**)&out_trace->filters));
  }
  for (iree_host_size_t i = 0; i < output->dump_before.count; ++i) {
    out_trace->filters[i] =
        loomc_string_view_from_iree(output->dump_before.values[i]);
  }
  for (iree_host_size_t i = 0; i < output->dump_after.count; ++i) {
    out_trace->filters[output->dump_before.count + i] =
        loomc_string_view_from_iree(output->dump_after.values[i]);
  }
  out_trace->options = (loomc_pass_trace_options_t){
      .type = LOOMC_STRUCTURE_TYPE_PASS_TRACE_OPTIONS,
      .structure_size = sizeof(out_trace->options),
      .format = output->format == LOOM_TOOLING_PASS_TRACE_FORMAT_JSONL
                    ? LOOMC_PASS_TRACE_FORMAT_JSONL
                    : LOOMC_PASS_TRACE_FORMAT_TEXT,
      .flags =
          (output->dump_before_all ? LOOMC_PASS_TRACE_FLAG_BEFORE_ALL : 0) |
          (output->dump_after_all ? LOOMC_PASS_TRACE_FLAG_AFTER_ALL : 0),
      .tool_name = loomc_string_view_from_iree(output->tool_name),
      .input_identifier = loomc_string_view_from_iree(output->input_path),
      .before_filters = output->dump_before.count ? out_trace->filters : NULL,
      .before_filter_count = output->dump_before.count,
      .after_filters = output->dump_after.count
                           ? out_trace->filters + output->dump_before.count
                           : NULL,
      .after_filter_count = output->dump_after.count,
      .sink =
          {
              .write = loom_tooling_loomc_trace_write,
              .user_data = &output->output.stream,
          },
  };
  if (loom_tooling_pass_trace_has_artifact_sink(output)) {
    out_trace->options.artifact_sink = (loomc_pass_trace_artifact_sink_t){
        .open = loom_tooling_loomc_trace_open_artifact,
        .close = loom_tooling_loomc_trace_close_artifact,
        .user_data = output,
    };
  }
  return iree_ok_status();
}

iree_status_t loom_tooling_loomc_pass_trace_close(
    loom_tooling_loomc_pass_trace_t* trace) {
  iree_status_t status = loom_tooling_pass_trace_close(&trace->output);
  iree_allocator_free(trace->allocator, trace->filters);
  *trace = (loom_tooling_loomc_pass_trace_t){0};
  return status;
}
