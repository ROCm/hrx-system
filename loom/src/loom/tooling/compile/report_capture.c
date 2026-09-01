// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/compile/report_capture.h"

#include "loom/error/json_sink.h"
#include "loom/ops/config/ops.h"
#include "loom/tooling/config/config.h"

void loom_compile_report_capture_options_initialize(
    loom_compile_report_capture_options_t* out_options) {
  *out_options = (loom_compile_report_capture_options_t){
      .sink_format = LOOM_COMPILE_REPORT_SINK_FORMAT_NONE,
      .detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE,
  };
}

iree_status_t loom_compile_report_capture_options_parse_request(
    iree_string_view_t value, loom_compile_report_capture_options_t* options) {
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("none"))) {
    options->sink_format = LOOM_COMPILE_REPORT_SINK_FORMAT_NONE;
    options->detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("summary")) ||
      iree_string_view_equal(value, IREE_SV("json")) ||
      iree_string_view_equal(value, IREE_SV("json-summary"))) {
    options->sink_format = LOOM_COMPILE_REPORT_SINK_FORMAT_JSON;
    options->detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("details")) ||
      iree_string_view_equal(value, IREE_SV("json-details"))) {
    options->sink_format = LOOM_COMPILE_REPORT_SINK_FORMAT_JSON;
    options->detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("text")) ||
      iree_string_view_equal(value, IREE_SV("text-summary"))) {
    options->sink_format = LOOM_COMPILE_REPORT_SINK_FORMAT_TEXT;
    options->detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("text-details"))) {
    options->sink_format = LOOM_COMPILE_REPORT_SINK_FORMAT_TEXT;
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

bool loom_compile_report_capture_options_is_enabled(
    const loom_compile_report_capture_options_t* options) {
  return options != NULL &&
         options->sink_format != LOOM_COMPILE_REPORT_SINK_FORMAT_NONE;
}

bool loom_compile_report_capture_is_enabled(
    const loom_compile_report_capture_t* capture) {
  return capture != NULL &&
         loom_compile_report_capture_options_is_enabled(&capture->options);
}

iree_status_t loom_compile_report_capture_initialize(
    const loom_compile_report_capture_options_t* options,
    iree_allocator_t host_allocator,
    loom_compile_report_capture_t* out_capture) {
  *out_capture = (loom_compile_report_capture_t){
      .options = *options,
      .host_allocator = host_allocator,
  };
  loom_target_compile_report_initialize(&out_capture->report, host_allocator);
  loom_json_value_list_initialize(host_allocator,
                                  &out_capture->diagnostics.json_values);
  switch (options->detail_mode) {
    case LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY:
      out_capture->report.requested_detail_flags =
          LOOM_TARGET_COMPILE_REPORT_DETAIL_CONFIG_BINDING_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;
      break;
    case LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS:
      out_capture->report.requested_detail_flags =
          LOOM_TARGET_COMPILE_REPORT_DETAIL_CONFIG_BINDING_ROWS |
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
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_CAPABILITY_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_INSERTION_ROWS;
      break;
    case LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE:
      break;
  }
  return iree_ok_status();
}

void loom_compile_report_capture_configure_compile_options(
    loom_compile_report_capture_t* capture,
    loom_compile_options_t* compile_options) {
  if (!loom_compile_report_capture_is_enabled(capture)) {
    return;
  }
  compile_options->report = &capture->report;
  if (capture->options.detail_mode ==
      LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    compile_options->target_pipeline_options
        .source_to_low_legality_diagnostic_flags |=
        LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_ALL;
  }
}

iree_status_t loom_compile_report_record_materialized_config(
    loom_target_compile_report_t* report, const loom_module_t* module,
    const loom_tooling_config_set_t* config_set) {
  if (report == NULL || module == NULL || config_set == NULL ||
      config_set->binding_count == 0 ||
      !loom_target_compile_report_wants_details(
          report, LOOM_TARGET_COMPILE_REPORT_DETAIL_CONFIG_BINDING_ROWS)) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < config_set->binding_count; ++i) {
    const loom_tooling_config_binding_t* binding = &config_set->bindings[i];
    const loom_string_id_t name_id =
        loom_module_lookup_string(module, binding->key);
    if (name_id == LOOM_STRING_ID_INVALID) {
      continue;
    }
    const uint16_t symbol_id = loom_module_find_symbol(module, name_id);
    if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
      continue;
    }
    const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
    if (!symbol->defining_op || !loom_config_def_isa(symbol->defining_op)) {
      continue;
    }
    const loom_target_compile_report_config_binding_row_t row = {
        .key = binding->key,
        .value = binding->value,
    };
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_record_config_binding_row(report, &row));
  }
  return iree_ok_status();
}

iree_status_t loom_compile_report_capture_record_materialized_config(
    loom_compile_report_capture_t* capture, const loom_module_t* module,
    const loom_tooling_config_set_t* config_set) {
  if (!loom_compile_report_capture_is_enabled(capture)) {
    return iree_ok_status();
  }
  return loom_compile_report_record_materialized_config(&capture->report,
                                                        module, config_set);
}

iree_status_t loom_compile_report_capture_record_diagnostic(
    loom_compile_report_capture_t* capture, const loom_diagnostic_t* diagnostic,
    loom_type_formatter_t type_formatter) {
  if (!loom_compile_report_capture_is_enabled(capture) || diagnostic == NULL) {
    return iree_ok_status();
  }
  if (capture->options.sink_format == LOOM_COMPILE_REPORT_SINK_FORMAT_JSON &&
      capture->options.detail_mode ==
          LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    loom_output_stream_t stream;
    IREE_RETURN_IF_ERROR(loom_json_value_list_begin_value(
        &capture->diagnostics.json_values, &stream));
    const loom_type_formatter_t formatter =
        type_formatter.fn
            ? type_formatter
            : (loom_type_formatter_t){loom_type_format_minimal, NULL};
    IREE_RETURN_IF_ERROR(
        loom_diagnostic_json_write_object(&stream, diagnostic, formatter));
  }
  ++capture->diagnostics.count;
  return iree_ok_status();
}

static iree_status_t loom_compile_report_capture_append_separator(
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

iree_status_t loom_compile_report_capture_append_text(
    const loom_compile_report_capture_t* capture,
    iree_string_builder_t* builder) {
  if (!loom_compile_report_capture_is_enabled(capture)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_compile_report_capture_append_separator(builder));
  const loom_target_compile_report_format_options_t format_options = {
      .mode = capture->options.detail_mode,
      .diagnostics =
          {
              .json_objects =
                  loom_json_value_list_body(&capture->diagnostics.json_values),
              .count = capture->diagnostics.count,
          },
  };
  return loom_target_compile_report_format_text(&capture->report,
                                                &format_options, builder);
}

iree_status_t loom_compile_report_capture_append_json(
    const loom_compile_report_capture_t* capture,
    loom_output_stream_t* stream) {
  if (!loom_compile_report_capture_is_enabled(capture)) {
    return iree_ok_status();
  }
  const loom_target_compile_report_format_options_t format_options = {
      .mode = capture->options.detail_mode,
      .diagnostics =
          {
              .json_objects =
                  loom_json_value_list_body(&capture->diagnostics.json_values),
              .count = capture->diagnostics.count,
          },
  };
  return loom_target_compile_report_format_json(&capture->report,
                                                &format_options, stream);
}

iree_status_t loom_compile_report_capture_append_output(
    const loom_compile_report_capture_t* capture,
    iree_string_builder_t* builder) {
  if (!loom_compile_report_capture_is_enabled(capture)) {
    return iree_ok_status();
  }
  switch (capture->options.sink_format) {
    case LOOM_COMPILE_REPORT_SINK_FORMAT_JSON: {
      IREE_RETURN_IF_ERROR(
          loom_compile_report_capture_append_separator(builder));
      loom_output_stream_t stream;
      loom_output_stream_for_builder(builder, &stream);
      IREE_RETURN_IF_ERROR(
          loom_compile_report_capture_append_json(capture, &stream));
      return iree_string_builder_append_string(builder, IREE_SV("\n"));
    }
    case LOOM_COMPILE_REPORT_SINK_FORMAT_TEXT:
      return loom_compile_report_capture_append_text(capture, builder);
    case LOOM_COMPILE_REPORT_SINK_FORMAT_NONE:
    default:
      return iree_ok_status();
  }
}

iree_status_t loom_compile_report_capture_write_output(
    const loom_compile_report_capture_t* capture, loom_output_stream_t* stream,
    iree_allocator_t host_allocator) {
  if (!loom_compile_report_capture_is_enabled(capture)) {
    return iree_ok_status();
  }
  switch (capture->options.sink_format) {
    case LOOM_COMPILE_REPORT_SINK_FORMAT_JSON: {
      IREE_RETURN_IF_ERROR(
          loom_compile_report_capture_append_json(capture, stream));
      return loom_output_stream_write_cstring(stream, "\n");
    }
    case LOOM_COMPILE_REPORT_SINK_FORMAT_TEXT: {
      iree_string_builder_t builder;
      iree_string_builder_initialize(host_allocator, &builder);
      iree_status_t status =
          loom_compile_report_capture_append_text(capture, &builder);
      if (iree_status_is_ok(status)) {
        status = loom_output_stream_write(stream,
                                          iree_string_builder_view(&builder));
      }
      iree_string_builder_deinitialize(&builder);
      return status;
    }
    case LOOM_COMPILE_REPORT_SINK_FORMAT_NONE:
    default:
      return iree_ok_status();
  }
}

void loom_compile_report_capture_deinitialize(
    loom_compile_report_capture_t* capture) {
  if (capture == NULL) {
    return;
  }
  loom_json_value_list_deinitialize(&capture->diagnostics.json_values);
  loom_target_compile_report_deinitialize(&capture->report);
  *capture = (loom_compile_report_capture_t){0};
}
