// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/compile_report_capture.h"

#include "loom/error/json_sink.h"

void loom_run_compile_report_capture_options_initialize(
    loom_run_compile_report_capture_options_t* out_options) {
  *out_options = (loom_run_compile_report_capture_options_t){
      .sink_format = LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_NONE,
      .detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE,
  };
}

iree_status_t loom_run_compile_report_capture_options_parse_request(
    iree_string_view_t value,
    loom_run_compile_report_capture_options_t* options) {
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("none"))) {
    options->sink_format = LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_NONE;
    options->detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("summary")) ||
      iree_string_view_equal(value, IREE_SV("json")) ||
      iree_string_view_equal(value, IREE_SV("json-summary"))) {
    options->sink_format = LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_JSON;
    options->detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("details")) ||
      iree_string_view_equal(value, IREE_SV("json-details"))) {
    options->sink_format = LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_JSON;
    options->detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("text")) ||
      iree_string_view_equal(value, IREE_SV("text-summary"))) {
    options->sink_format = LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_TEXT;
    options->detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("text-details"))) {
    options->sink_format = LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_TEXT;
    options->detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "unsupported compile report request '%.*s'; expected 'none', 'summary', "
      "'details', 'json', 'json-summary', 'json-details', 'text', "
      "'text-summary', or 'text-details'",
      (int)value.size, value.data);
}

bool loom_run_compile_report_capture_options_is_enabled(
    const loom_run_compile_report_capture_options_t* options) {
  return options != NULL &&
         options->sink_format != LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_NONE;
}

bool loom_run_compile_report_capture_is_enabled(
    const loom_run_compile_report_capture_t* capture) {
  return capture != NULL &&
         loom_run_compile_report_capture_options_is_enabled(&capture->options);
}

iree_status_t loom_run_compile_report_capture_initialize(
    const loom_run_compile_report_capture_options_t* options,
    iree_allocator_t host_allocator,
    loom_run_compile_report_capture_t* out_capture) {
  *out_capture = (loom_run_compile_report_capture_t){
      .options = *options,
      .host_allocator = host_allocator,
  };
  loom_target_compile_report_initialize(&out_capture->report, host_allocator);
  iree_string_builder_initialize(host_allocator,
                                 &out_capture->diagnostic_json_objects);
  switch (options->detail_mode) {
    case LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY:
      out_capture->report.requested_detail_flags =
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS;
      break;
    case LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS:
      out_capture->report.requested_detail_flags =
          LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_FAILURE_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_HIGH_WATER_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_MATH_LEGALIZATION_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_LEGALIZATION_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_CAPABILITY_ROWS;
      break;
    case LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE:
      break;
  }
  return iree_ok_status();
}

void loom_run_compile_report_capture_configure_compile_options(
    loom_run_compile_report_capture_t* capture,
    loom_run_candidate_compile_options_t* compile_options) {
  if (!loom_run_compile_report_capture_is_enabled(capture)) {
    return;
  }
  compile_options->report = &capture->report;
  compile_options->report_capture = capture;
  if (capture->options.detail_mode ==
      LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    compile_options->target_pipeline_options
        .source_to_low_legality_diagnostic_flags |=
        LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_ALL;
  }
}

iree_status_t loom_run_compile_report_capture_record_diagnostic(
    loom_run_compile_report_capture_t* capture,
    const loom_diagnostic_t* diagnostic, loom_type_formatter_t type_formatter) {
  if (!loom_run_compile_report_capture_is_enabled(capture) ||
      diagnostic == NULL) {
    return iree_ok_status();
  }
  if (capture->options.sink_format ==
          LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_JSON &&
      capture->options.detail_mode ==
          LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&capture->diagnostic_json_objects, &stream);
    if (capture->diagnostic_count > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\n"));
    }
    const loom_type_formatter_t formatter =
        type_formatter.fn
            ? type_formatter
            : (loom_type_formatter_t){loom_type_format_minimal, NULL};
    IREE_RETURN_IF_ERROR(
        loom_diagnostic_json_write_object(&stream, diagnostic, formatter));
  }
  ++capture->diagnostic_count;
  return iree_ok_status();
}

static iree_status_t loom_run_compile_report_capture_append_separator(
    iree_string_builder_t* builder) {
  iree_host_size_t builder_size = iree_string_builder_size(builder);
  if (builder_size == 0) {
    return iree_ok_status();
  }
  const char* buffer = iree_string_builder_buffer(builder);
  if (buffer[builder_size - 1] == '\n') {
    return iree_ok_status();
  }
  return iree_string_builder_append_string(builder, IREE_SV("\n"));
}

iree_status_t loom_run_compile_report_capture_append_text(
    const loom_run_compile_report_capture_t* capture,
    iree_string_builder_t* builder) {
  if (!loom_run_compile_report_capture_is_enabled(capture)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_run_compile_report_capture_append_separator(builder));
  const loom_target_compile_report_format_options_t format_options = {
      .mode = capture->options.detail_mode,
      .diagnostic_json_objects =
          iree_string_builder_view(&capture->diagnostic_json_objects),
      .diagnostic_count = capture->diagnostic_count,
  };
  return loom_target_compile_report_format_text(&capture->report,
                                                &format_options, builder);
}

iree_status_t loom_run_compile_report_capture_append_json(
    const loom_run_compile_report_capture_t* capture,
    loom_output_stream_t* stream) {
  if (!loom_run_compile_report_capture_is_enabled(capture)) {
    return iree_ok_status();
  }
  const loom_target_compile_report_format_options_t format_options = {
      .mode = capture->options.detail_mode,
      .diagnostic_json_objects =
          iree_string_builder_view(&capture->diagnostic_json_objects),
      .diagnostic_count = capture->diagnostic_count,
  };
  return loom_target_compile_report_format_json(&capture->report,
                                                &format_options, stream);
}

iree_status_t loom_run_compile_report_capture_append_output(
    const loom_run_compile_report_capture_t* capture,
    iree_string_builder_t* builder) {
  if (!loom_run_compile_report_capture_is_enabled(capture)) {
    return iree_ok_status();
  }
  switch (capture->options.sink_format) {
    case LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_JSON: {
      IREE_RETURN_IF_ERROR(
          loom_run_compile_report_capture_append_separator(builder));
      loom_output_stream_t stream;
      loom_output_stream_for_builder(builder, &stream);
      IREE_RETURN_IF_ERROR(
          loom_run_compile_report_capture_append_json(capture, &stream));
      return iree_string_builder_append_string(builder, IREE_SV("\n"));
    }
    case LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_TEXT:
      return loom_run_compile_report_capture_append_text(capture, builder);
    case LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_NONE:
    default:
      return iree_ok_status();
  }
}

iree_status_t loom_run_compile_report_capture_write_output(
    const loom_run_compile_report_capture_t* capture,
    loom_output_stream_t* stream, iree_allocator_t host_allocator) {
  if (!loom_run_compile_report_capture_is_enabled(capture)) {
    return iree_ok_status();
  }
  switch (capture->options.sink_format) {
    case LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_JSON: {
      IREE_RETURN_IF_ERROR(
          loom_run_compile_report_capture_append_json(capture, stream));
      return loom_output_stream_write_cstring(stream, "\n");
    }
    case LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_TEXT: {
      iree_string_builder_t builder;
      iree_string_builder_initialize(host_allocator, &builder);
      iree_status_t status =
          loom_run_compile_report_capture_append_text(capture, &builder);
      if (iree_status_is_ok(status)) {
        status = loom_output_stream_write(stream,
                                          iree_string_builder_view(&builder));
      }
      iree_string_builder_deinitialize(&builder);
      return status;
    }
    case LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_NONE:
    default:
      return iree_ok_status();
  }
}

void loom_run_compile_report_capture_deinitialize(
    loom_run_compile_report_capture_t* capture) {
  if (capture == NULL) {
    return;
  }
  iree_string_builder_deinitialize(&capture->diagnostic_json_objects);
  loom_target_compile_report_deinitialize(&capture->report);
  *capture = (loom_run_compile_report_capture_t){0};
}
