// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compile_report.h"

#include "diagnostic.h"
#include "iree/io/vec_stream.h"
#include "loom/error/json_sink.h"
#include "loomc/iree.h"
#include "loomc/status.h"
#include "module.h"
#include "result.h"

loomc_status_t loomc_compile_report_mode_parse(
    loomc_string_view_t value, loomc_compile_report_mode_t* out_mode) {
  if (out_mode == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_mode must not be NULL");
  }
  if (loomc_string_view_is_empty(value) ||
      loomc_string_view_equal(value, loomc_make_cstring_view("none"))) {
    *out_mode = LOOMC_COMPILE_REPORT_MODE_NONE;
    return loomc_ok_status();
  }
  if (loomc_string_view_equal(value, loomc_make_cstring_view("summary")) ||
      loomc_string_view_equal(value, loomc_make_cstring_view("json")) ||
      loomc_string_view_equal(value, loomc_make_cstring_view("json-summary"))) {
    *out_mode = LOOMC_COMPILE_REPORT_MODE_SUMMARY;
    return loomc_ok_status();
  }
  if (loomc_string_view_equal(value, loomc_make_cstring_view("details")) ||
      loomc_string_view_equal(value, loomc_make_cstring_view("json-details"))) {
    *out_mode = LOOMC_COMPILE_REPORT_MODE_DETAILS;
    return loomc_ok_status();
  }
  return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                           "unsupported compile report mode");
}

loomc_string_view_t loomc_compile_report_mode_name(
    loomc_compile_report_mode_t mode) {
  switch (mode) {
    case LOOMC_COMPILE_REPORT_MODE_SUMMARY:
      return loomc_make_cstring_view("summary");
    case LOOMC_COMPILE_REPORT_MODE_DETAILS:
      return loomc_make_cstring_view("details");
    case LOOMC_COMPILE_REPORT_MODE_NONE:
    default:
      return loomc_make_cstring_view("none");
  }
}

loomc_status_t loomc_compile_report_options_validate(
    const loomc_compile_report_options_t* options) {
  if (options->type != LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compile report options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compile report options structure_size is too small");
  }
  switch (options->mode) {
    case LOOMC_COMPILE_REPORT_MODE_NONE:
    case LOOMC_COMPILE_REPORT_MODE_SUMMARY:
    case LOOMC_COMPILE_REPORT_MODE_DETAILS:
      break;
    default:
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "compile report mode is invalid");
  }
  switch (options->format) {
    case LOOMC_COMPILE_REPORT_FORMAT_JSON:
    case LOOMC_COMPILE_REPORT_FORMAT_TEXT:
      break;
    default:
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "compile report format is invalid");
  }
  if (!options->identifier.data && options->identifier.size != 0) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compile report identifier has length but no data");
  }
  if (options->mode == LOOMC_COMPILE_REPORT_MODE_NONE &&
      !loomc_string_view_is_empty(options->identifier)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compile report identifier requires a non-NONE report mode");
  }
  return loomc_ok_status();
}

static iree_status_t loomc_compile_report_write_diagnostic(
    void* user_data, iree_string_view_t text) {
  return iree_io_stream_write((iree_io_stream_t*)user_data, text.size,
                              text.data);
}

static iree_status_t loomc_compile_report_capture_diagnostic(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loomc_compile_report_state_t* state = user_data;
  if (!state->diagnostics) {
    IREE_RETURN_IF_ERROR(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE, 4096, iree_arena_allocator(&state->arena),
        &state->diagnostics));
  }
  loom_output_stream_t stream = {
      .write = loomc_compile_report_write_diagnostic,
      .user_data = state->diagnostics,
  };
  if (iree_io_stream_length(state->diagnostics) != 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write(&stream, IREE_SV(",")));
  }
  const loomc_diagnostic_type_printer_t type_printer = {
      .module = diagnostic->module,
      .text_print_options = state->diagnostic_text_print_options,
  };
  const loom_type_formatter_t type_formatter =
      loomc_diagnostic_type_printer_formatter(diagnostic->module ? &type_printer
                                                                 : NULL);
  return loom_diagnostic_json_write_object(&stream, diagnostic, type_formatter);
}

void loomc_compile_report_state_initialize(
    loomc_compile_report_mode_t mode, loomc_compile_report_format_t format,
    iree_arena_block_pool_t* block_pool, loomc_result_t* result,
    loomc_compile_report_state_t* out_state) {
  *out_state = (loomc_compile_report_state_t){
      .mode = mode == LOOMC_COMPILE_REPORT_MODE_DETAILS
                  ? LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS
                  : LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY,
      .format = format,
      .result = result,
  };
  iree_arena_initialize(block_pool, &out_state->arena);
  loom_target_compile_report_initialize(iree_arena_allocator(&out_state->arena),
                                        &out_state->report);
  out_state->report.requested_detail_flags =
      loom_target_compile_report_requested_detail_flags(out_state->mode);
  if (mode == LOOMC_COMPILE_REPORT_MODE_DETAILS &&
      format == LOOMC_COMPILE_REPORT_FORMAT_JSON) {
    out_state->diagnostic_sink = (loom_diagnostic_sink_t){
        .fn = loomc_compile_report_capture_diagnostic,
        .user_data = out_state,
    };
    loomc_result_set_loom_diagnostic_sink(result, &out_state->diagnostic_sink);
  }
}

loomc_status_t loomc_compile_report_record_config_bindings(
    loomc_compile_report_state_t* state, const loomc_module_t* module) {
  loomc_status_t status = loomc_ok_status();
  for (const loomc_config_binding_record_t* binding =
           loomc_module_config_bindings(module)->head;
       binding != NULL && loomc_status_is_ok(status); binding = binding->next) {
    const loom_target_compile_report_config_binding_row_t row = {
        .key = binding->key,
        .value = binding->value,
    };
    status = loomc_status_from_iree(
        loom_target_compile_report_record_config_binding_row(&state->report,
                                                             &row));
  }
  return status;
}

loomc_status_t loomc_compile_report_state_finish(
    loomc_compile_report_state_t* state, loomc_string_view_t identifier) {
  loomc_allocator_t allocator = loomc_result_allocator(state->result);
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_from_loomc(allocator),
                                 &builder);
  iree_byte_sequence_t* diagnostics = NULL;
  loomc_status_t status = loomc_ok_status();
  if (state->diagnostics) {
    status = loomc_status_from_iree(
        iree_io_vec_stream_move_contents(state->diagnostics, &diagnostics));
  }
  const loom_target_compile_report_format_options_t format_options = {
      .mode = state->mode,
      .diagnostics =
          {
              .json_objects = diagnostics,
              .count = loomc_result_diagnostic_count(state->result),
          },
  };
  if (loomc_status_is_ok(status)) {
    if (state->format == LOOMC_COMPILE_REPORT_FORMAT_TEXT) {
      status = loomc_status_from_iree(loom_target_compile_report_format_text(
          &state->report, &format_options, &builder));
    } else {
      loom_output_stream_t stream;
      loom_output_stream_for_builder(&builder, &stream);
      status = loomc_status_from_iree(loom_target_compile_report_format_json(
          &state->report, &format_options, &stream));
    }
  }
  iree_byte_sequence_release(diagnostics);

  char* report_storage = NULL;
  if (loomc_status_is_ok(status)) {
    const iree_host_size_t report_length = iree_string_builder_size(&builder);
    report_storage = iree_string_builder_take_storage(&builder);
    const loomc_string_view_t artifact_format =
        state->format == LOOMC_COMPILE_REPORT_FORMAT_TEXT
            ? loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_TEXT)
            : loomc_make_cstring_view(
                  LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON);
    status = loomc_result_add_artifact_take_contents(
        state->result, LOOMC_ARTIFACT_KIND_REPORT, artifact_format, identifier,
        loomc_make_byte_span(report_storage, report_length));
  }
  if (loomc_status_is_ok(status)) {
    report_storage = NULL;
  }
  loomc_allocator_free(allocator, report_storage);
  iree_string_builder_deinitialize(&builder);
  return status;
}

void loomc_compile_report_state_deinitialize(
    loomc_compile_report_state_t* state) {
  if (state->mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE) {
    return;
  }
  if (state->diagnostic_sink.fn) {
    loomc_result_set_loom_diagnostic_sink(state->result, NULL);
  }
  iree_io_stream_release(state->diagnostics);
  loom_target_compile_report_deinitialize(&state->report);
  iree_arena_deinitialize(&state->arena);
  *state = (loomc_compile_report_state_t){0};
}
