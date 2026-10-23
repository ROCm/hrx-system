// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/execute.h"

#include "loom/codegen/low/text_asm.h"
#include "loom/error/diagnostic.h"
#include "loom/error/json_sink.h"
#include "loom/error/source.h"
#include "loom/format/text/printer.h"
#include "loom/pass/report.h"
#include "loom/target/provider.h"
#include "loom/target/reporting/format.h"
#include "loom/tools/loom-check/comparison.h"
#include "loom/tools/loom-check/compile.h"
#include "loom/tools/loom-check/compile_diagnostics.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/requirements.h"
#include "loom/util/diff.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"
#include "loomc/iree.h"

void loom_check_result_initialize(iree_allocator_t allocator,
                                  loom_check_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));
  iree_string_builder_initialize(allocator, &out_result->detail);
  loom_json_value_list_initialize(allocator, &out_result->diff_hunks);
  iree_string_builder_initialize(allocator, &out_result->actual_output);
  iree_string_builder_initialize(allocator, &out_result->update_edit.text);
  loom_json_value_list_initialize(allocator, &out_result->annotation_edits);
  loom_json_value_list_initialize(allocator, &out_result->diagnostics);
}

void loom_check_result_deinitialize(loom_check_result_t* result) {
  loom_json_value_list_deinitialize(&result->diagnostics);
  loom_json_value_list_deinitialize(&result->annotation_edits);
  iree_string_builder_deinitialize(&result->update_edit.text);
  iree_string_builder_deinitialize(&result->actual_output);
  loom_json_value_list_deinitialize(&result->diff_hunks);
  iree_string_builder_deinitialize(&result->detail);
}

const loom_check_emit_provider_t* loom_check_environment_lookup_emit_provider(
    const loom_check_environment_t* environment,
    iree_string_view_t target_name) {
  if (environment == NULL || environment->emit_providers.providers == NULL) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < environment->emit_providers.provider_count;
       ++i) {
    const loom_check_emit_provider_t* provider =
        environment->emit_providers.providers[i];
    if (provider == NULL || provider->match == NULL) {
      continue;
    }
    if (provider->match(provider, target_name)) {
      return provider;
    }
  }
  return NULL;
}

static iree_status_t loom_check_write_diff_hunk_json(
    const loom_diff_result_t* diff, const loom_diff_hunk_t* hunk,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("expected_start_line"), hunk->expected_start_line));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("expected_line_count"), hunk->expected_line_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("actual_start_line"), hunk->actual_start_line));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("actual_line_count"), hunk->actual_line_count));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("lines")));
  loom_json_array_writer_t lines;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &lines));
  for (iree_host_size_t i = 0; i < hunk->line_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&lines));
    const loom_diff_hunk_line_t* line = &diff->lines[hunk->line_offset + i];
    loom_json_object_writer_t line_object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &line_object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &line_object, IREE_SV("kind"),
        iree_make_cstring_view(loom_diff_hunk_line_kind_name(line->kind))));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &line_object, IREE_SV("text"), line->text));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&line_object));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&lines));
  return loom_json_object_end(&object);
}

static iree_status_t loom_check_write_diff_json_hunks(
    const loom_diff_result_t* diff, loom_check_result_t* result) {
  loom_output_stream_t stream;
  for (iree_host_size_t i = 0; i < diff->hunk_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_json_value_list_begin_value(&result->diff_hunks, &stream));
    IREE_RETURN_IF_ERROR(
        loom_check_write_diff_hunk_json(diff, &diff->hunks[i], &stream));
  }
  return iree_ok_status();
}

iree_status_t loom_check_result_record_diff(iree_string_view_t expected,
                                            iree_string_view_t actual,
                                            iree_allocator_t allocator,
                                            loom_check_result_t* result) {
  loom_diff_result_t diff = {0};
  iree_status_t status = loom_diff_compute(
      expected, actual, LOOM_DIFF_DEFAULT_CONTEXT, allocator, &diff);
  if (iree_status_is_ok(status)) {
    status = loom_diff_format_result(&diff, &result->detail);
  }
  if (iree_status_is_ok(status)) {
    status = loom_check_write_diff_json_hunks(&diff, result);
  }
  loom_diff_result_deinitialize(allocator, &diff);
  return status;
}

static loom_type_formatter_t loom_check_capture_type_formatter(
    const loom_check_diagnostic_capture_t* capture) {
  if (capture->type_formatter.fn) {
    return capture->type_formatter;
  }
  return (loom_type_formatter_t){loom_type_format_minimal, NULL};
}

static iree_status_t loom_check_append_diagnostic_json(
    loom_check_result_t* result, const loom_diagnostic_t* diagnostic,
    loom_type_formatter_t type_formatter) {
  loom_output_stream_t stream;
  IREE_RETURN_IF_ERROR(
      loom_json_value_list_begin_value(&result->diagnostics, &stream));
  IREE_RETURN_IF_ERROR(
      loom_diagnostic_json_write_object(&stream, diagnostic, type_formatter));
  return iree_ok_status();
}

iree_status_t loom_check_result_append_annotation_edit(
    loom_check_result_t* result, loom_check_update_edit_kind_t kind,
    loom_test_source_range_t range, iree_host_size_t target_line,
    iree_string_view_t text) {
  loom_output_stream_t stream;
  IREE_RETURN_IF_ERROR(
      loom_json_value_list_begin_value(&result->annotation_edits, &stream));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("kind"),
      iree_make_cstring_view(loom_check_update_edit_kind_name(kind))));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("range")));
  loom_json_object_writer_t range_object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &range_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &range_object, IREE_SV("start_byte"), range.start_byte));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &range_object, IREE_SV("end_byte"), range.end_byte));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&range_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("target_line"), target_line));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_string_field(&object, IREE_SV("text"), text));
  return loom_json_object_end(&object);
}

iree_status_t loom_check_diagnostic_capture_sink(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loom_check_diagnostic_capture_t* capture =
      (loom_check_diagnostic_capture_t*)user_data;
  const loom_type_formatter_t type_formatter =
      loom_check_capture_type_formatter(capture);
  if (capture->detail) {
    loom_output_stream_t stream;
    loom_output_stream_for_builder(capture->detail, &stream);
    const loom_diagnostic_format_options_t format_options = {
        .type_formatter = type_formatter,
    };
    IREE_RETURN_IF_ERROR(loom_diagnostic_format_with_options(
        diagnostic, &format_options, &stream));
  }
  if (capture->result) {
    IREE_RETURN_IF_ERROR(loom_check_append_diagnostic_json(
        capture->result, diagnostic, type_formatter));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Execution
//===----------------------------------------------------------------------===//

iree_status_t loom_check_execute_case(
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_input_t* input_request,
    const loom_check_environment_t* environment,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  const loom_input_format_t* input_format = NULL;
  IREE_RETURN_IF_ERROR(loom_input_format_select(
      environment->input_formats,
      test_case->input_options.format.size ? test_case->input_options.format
                                           : input_request->format,
      input_request->path, &input_format, /*out_ordinal=*/NULL));
  bool continue_execution = true;
  IREE_RETURN_IF_ERROR(loom_check_preflight_requirements(
      test_case, environment, allocator, result, &continue_execution));
  if (!continue_execution) {
    return iree_ok_status();
  }

  switch (test_case->mode) {
    case LOOM_TEST_MODE_ROUNDTRIP: {
      IREE_RETURN_IF_ERROR(loom_check_execute_roundtrip(
          test_case, filename, input_request, environment, block_pool,
          allocator, result));
      break;
    }
    case LOOM_TEST_MODE_VERIFY: {
      IREE_RETURN_IF_ERROR(loom_check_execute_verify(
          test_case, case_index, report, filename, input_request, environment,
          block_pool, allocator, result));
      break;
    }
    case LOOM_TEST_MODE_PASS: {
      IREE_RETURN_IF_ERROR(loom_check_execute_pass(
          test_case, case_index, report, filename, input_request, environment,
          block_pool, allocator, result));
      break;
    }
    case LOOM_TEST_MODE_PASS_REPORT: {
      IREE_RETURN_IF_ERROR(loom_check_execute_pass_report(
          test_case, case_index, report, filename, input_request, environment,
          block_pool, allocator, result));
      break;
    }
    case LOOM_TEST_MODE_COMPILE_REPORT: {
      IREE_RETURN_IF_ERROR(loom_check_execute_compile_report(
          test_case, case_index, report, filename, input_request, environment,
          block_pool, allocator, result));
      break;
    }
    case LOOM_TEST_MODE_FORMAT: {
      IREE_RETURN_IF_ERROR(loom_check_execute_format(
          test_case, filename, input_request, environment, block_pool,
          allocator, result));
      break;
    }
    case LOOM_TEST_MODE_EMIT: {
      IREE_RETURN_IF_ERROR(loom_check_execute_emit(
          test_case, case_index, report, filename, input_request, environment,
          block_pool, allocator, result));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown check mode: %d", (int)test_case->mode);
  }

  if (iree_all_bits_set(test_case->output_flags, LOOM_TEST_OUTPUT_CHECKS) &&
      result->raw_outcome == LOOM_CHECK_PASS &&
      (!result->has_actual_output || result->actual_output.size == 0)) {
    result->raw_outcome = LOOM_CHECK_FAIL;
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        &result->detail, "with-checks requires textual output to check\n"));
  }

  // XFAIL inversion: expected failure that fails is a pass, expected
  // failure that passes is a failure.
  if (test_case->xfail) {
    result->final_outcome = (result->raw_outcome == LOOM_CHECK_FAIL)
                                ? LOOM_CHECK_PASS
                                : LOOM_CHECK_FAIL;
  } else {
    result->final_outcome = result->raw_outcome;
  }

  return iree_ok_status();
}

static iree_status_t loom_check_verify_pass_module(
    loomc_module_t* module,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    iree_allocator_t allocator, bool* out_succeeded) {
  *out_succeeded = false;
  const loomc_module_verify_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_VERIFY_OPTIONS,
      .structure_size = sizeof(options),
      .max_errors = 100,
  };
  loomc_result_t* verification = NULL;
  iree_status_t status = iree_status_from_loomc(loomc_module_verify(
      module, &options, loomc_allocator_from_iree(allocator), &verification));
  if (iree_status_is_ok(status)) {
    *out_succeeded = loomc_result_succeeded(verification);
    status = loom_check_compile_append_result_diagnostics(diagnostic_collector,
                                                          verification);
  }
  loomc_result_release(verification);
  return status;
}

static iree_status_t loom_check_finish_diagnostics_if_needed(
    loom_check_diagnostic_collector_t* diagnostic_collector,
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_allocator_t allocator,
    loom_check_result_t* result, bool* out_failed) {
  *out_failed = false;
  if (diagnostic_collector->count == 0 && test_case->annotation_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_check_diagnostic_collector_finish(
      diagnostic_collector, test_case, case_index, report, allocator, result));
  *out_failed = result->raw_outcome == LOOM_CHECK_FAIL;
  return iree_ok_status();
}

static bool loom_check_case_has_expected_output(
    const loom_test_case_t* test_case) {
  return !iree_string_view_is_empty(iree_string_view_trim(test_case->expected));
}

static iree_status_t loom_check_execute_finish_status_failure(
    iree_status_t failure_status, iree_string_view_t operation_name,
    loom_check_result_t* result) {
  result->raw_outcome = LOOM_CHECK_FAIL;
  iree_status_t status = iree_string_builder_append_format(
      &result->detail, "%.*s returned status: ", (int)operation_name.size,
      operation_name.data);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_status(&result->detail, failure_status);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&result->detail, "\n");
  }
  iree_status_free(failure_status);
  return status;
}

typedef enum loom_check_pass_output_kind_e {
  LOOM_CHECK_PASS_OUTPUT_IR = 0,
  LOOM_CHECK_PASS_OUTPUT_COMPILE_REPORT = 1,
  LOOM_CHECK_PASS_OUTPUT_PASS_REPORT = 2,
} loom_check_pass_output_kind_t;

typedef struct loom_check_pass_target_request_t {
  // Optional source function to specialize.
  iree_string_view_t function_name;
  // Complete family:selector target spelling.
  iree_string_view_t target;
  // True when |target| was explicitly supplied.
  bool has_target;
} loom_check_pass_target_request_t;

static loom_text_print_flags_t loom_check_pass_print_flags(
    const loom_test_case_t* test_case) {
  loom_text_print_flags_t flags =
      LOOM_TEXT_PRINT_DEFAULT | LOOM_TEXT_PRINT_PREFER_LOW_ASM;
  if (iree_all_bits_set(test_case->output_flags, LOOM_TEST_OUTPUT_LOW_ASM)) {
    flags |= LOOM_TEXT_PRINT_PRESERVE_LOW_ASM;
  }
  if (iree_all_bits_set(test_case->output_flags, LOOM_TEST_OUTPUT_LOCATIONS)) {
    flags |= LOOM_TEXT_PRINT_LOCATIONS;
  }
  return flags;
}

// Compile options precede the pipeline, whose own pass options remain opaque
// to this front door. A named pipeline starts with '@', so an explicit entry
// uses entry=@function instead of overloading the pipeline spelling.
static iree_status_t loom_check_parse_pass_target(
    iree_string_view_t* pipeline, loom_check_pass_target_request_t* request) {
  *request = (loom_check_pass_target_request_t){0};
  *pipeline = iree_string_view_trim(*pipeline);
  while (iree_string_view_starts_with(*pipeline, IREE_SV("target=")) ||
         iree_string_view_starts_with(*pipeline, IREE_SV("entry="))) {
    iree_string_view_t token;
    iree_string_view_split(*pipeline, ' ', &token, pipeline);
    *pipeline = iree_string_view_trim(*pipeline);
    iree_string_view_t name;
    iree_string_view_t value;
    iree_string_view_split(token, '=', &name, &value);
    if (iree_string_view_equal(name, IREE_SV("target"))) {
      if (request->has_target) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate pass option 'target'");
      }
      request->target = value;
      request->has_target = true;
    } else {
      if (value.size <= 1 || value.data[0] != '@' ||
          !iree_string_view_is_empty(request->function_name)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "pass expects at most one entry=@function");
      }
      request->function_name = value;
    }
  }
  if (!iree_string_view_is_empty(request->function_name) &&
      !request->has_target) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass entry requires target=family:selector");
  }
  return iree_ok_status();
}

static iree_status_t loom_check_execute_pass_with_output(
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_input_t* input_request,
    const loom_check_environment_t* environment,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result, loom_check_pass_output_kind_t output_kind) {
  loom_check_compile_session_t* session = environment->compile_session;
  const loomc_allocator_t public_allocator =
      loomc_allocator_from_iree(allocator);
  loomc_module_t* module = NULL;
  iree_arena_allocator_t diagnostic_arena;
  iree_arena_initialize(block_pool, &diagnostic_arena);
  loom_check_diagnostic_collector_t diagnostic_collector = {
      .arena = &diagnostic_arena,
      .host_allocator = allocator,
      .filename = filename,
      .result = result,
  };
  iree_status_t status = loom_check_compile_admit_module(
      test_case, filename, input_request, session, environment,
      /*max_errors=*/20, &diagnostic_collector, block_pool, allocator, &module);
  if (!iree_status_is_ok(status) || module == NULL ||
      diagnostic_collector.count > 0) {
    if (iree_status_is_ok(status)) {
      status = loom_check_diagnostic_collector_finish(
          &diagnostic_collector, test_case, case_index, report, allocator,
          result);
    }
    loomc_module_release(module);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  bool succeeded = false;
  iree_string_view_t pipeline = test_case->pipeline;
  loom_check_pass_target_request_t target_request;
  if (iree_status_is_ok(status)) {
    status = loom_check_parse_pass_target(&pipeline, &target_request);
  }
  // Admit input before any transformation can erase an invalid operation.
  if (iree_status_is_ok(status)) {
    status = loom_check_verify_pass_module(module, &diagnostic_collector,
                                           allocator, &succeeded);
  }
  loomc_target_profile_t* target_profile = NULL;
  if (iree_status_is_ok(status) && succeeded && target_request.has_target) {
    status = loom_check_compile_session_select_target_profile(
        session, target_request.target, &target_profile);
    if (iree_status_is_ok(status) &&
        iree_string_view_is_empty(target_request.function_name)) {
      loomc_module_interop_view_t view = {0};
      status = loom_check_compile_project_module(module, &diagnostic_collector,
                                                 allocator, &view, &succeeded);
      if (iree_status_is_ok(status) && succeeded) {
        status = loom_check_compile_select_source_function(
            view.module, target_request.function_name,
            &target_request.function_name);
      }
    }
  }
  loomc_pass_program_t* pass_program = NULL;
  if (iree_status_is_ok(status) && succeeded) {
    loomc_result_t* preparation = NULL;
    if (iree_string_view_starts_with(pipeline, IREE_SV("@"))) {
      status =
          iree_status_from_loomc(loomc_pass_program_create_from_module_symbol(
              module, loomc_string_view_from_iree(pipeline), /*options=*/NULL,
              public_allocator, &pass_program, &preparation));
    } else {
      status =
          iree_status_from_loomc(loomc_pass_program_create_from_pipeline_text(
              session->context, loomc_string_view_from_iree(pipeline),
              /*options=*/NULL, public_allocator, &pass_program, &preparation));
    }
    if (iree_status_is_ok(status)) {
      succeeded = loomc_result_succeeded(preparation);
      status = loom_check_compile_append_result_diagnostics(
          &diagnostic_collector, preparation);
    }
    loomc_result_release(preparation);
  }
  loom_pass_report_t pass_report = {0};
  if (iree_status_is_ok(status) &&
      output_kind == LOOM_CHECK_PASS_OUTPUT_PASS_REPORT) {
    loom_pass_report_initialize(allocator, &pass_report);
  }
  loom_target_compile_report_t compile_report = {0};
  if (iree_status_is_ok(status) &&
      output_kind == LOOM_CHECK_PASS_OUTPUT_COMPILE_REPORT) {
    loom_target_compile_report_initialize(
        &compile_report, iree_arena_allocator(&diagnostic_arena));
    compile_report.requested_detail_flags =
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
  }
  if (iree_status_is_ok(status) && succeeded) {
    const loomc_native_pass_report_options_t pass_report_options = {
        .type = LOOMC_STRUCTURE_TYPE_NATIVE_PASS_REPORT_OPTIONS,
        .structure_size = sizeof(pass_report_options),
        .report = &pass_report,
    };
    const loomc_native_compile_report_options_t compile_report_options = {
        .type = LOOMC_STRUCTURE_TYPE_NATIVE_COMPILE_REPORT_OPTIONS,
        .structure_size = sizeof(compile_report_options),
        .report = &compile_report,
    };
    const void* report_options = NULL;
    if (output_kind == LOOM_CHECK_PASS_OUTPUT_PASS_REPORT) {
      report_options = &pass_report_options;
    } else if (output_kind == LOOM_CHECK_PASS_OUTPUT_COMPILE_REPORT) {
      report_options = &compile_report_options;
    }
    loomc_result_t* execution = NULL;
    if (target_profile != NULL) {
      const loomc_target_specialization_t specialization = {
          .function_symbol =
              loomc_string_view_from_iree(target_request.function_name),
          .target_profile = target_profile,
      };
      const loomc_target_specialization_options_t target_options = {
          .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
          .structure_size = sizeof(target_options),
          .next = report_options,
          .specializations = &specialization,
          .specialization_count = 1,
      };
      const loomc_compile_options_t options = {
          .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
          .structure_size = sizeof(options),
          .next = &target_options,
      };
      status = iree_status_from_loomc(loomc_compile_module(
          session->compiler, session->workspace, pass_program, module, &options,
          public_allocator, &execution));
    } else {
      const loomc_pass_run_options_t options = {
          .type = LOOMC_STRUCTURE_TYPE_PASS_RUN_OPTIONS,
          .structure_size = sizeof(options),
          .next = report_options,
      };
      status = iree_status_from_loomc(
          loomc_pass_program_run(pass_program, session->workspace, module,
                                 &options, public_allocator, &execution));
    }
    if (iree_status_is_ok(status)) {
      succeeded = loomc_result_succeeded(execution);
      status = loom_check_compile_append_result_diagnostics(
          &diagnostic_collector, execution);
    }
    loomc_result_release(execution);
  }
  bool verified = false;
  if (iree_status_is_ok(status) && succeeded) {
    status = loom_check_verify_pass_module(module, &diagnostic_collector,
                                           allocator, &verified);
  }

  bool emit_output = false;
  if (!iree_status_is_ok(status)) {
    status = loom_check_execute_finish_status_failure(
        status, IREE_SV("pass pipeline"), result);
  } else {
    bool diagnostics_failed = false;
    status = loom_check_finish_diagnostics_if_needed(
        &diagnostic_collector, test_case, case_index, report, allocator, result,
        &diagnostics_failed);
    const bool has_expected_output =
        loom_check_case_has_expected_output(test_case);
    if (iree_status_is_ok(status) && !diagnostics_failed) {
      if (succeeded) {
        emit_output = verified &&
                      (diagnostic_collector.count == 0 || has_expected_output);
      } else {
        // A diagnosed pass failure may still have a partial report to inspect.
        emit_output =
            output_kind != LOOM_CHECK_PASS_OUTPUT_IR && has_expected_output;
      }
    }
  }
  if (emit_output && output_kind == LOOM_CHECK_PASS_OUTPUT_COMPILE_REPORT) {
    loom_target_compile_report_format_options_t report_options;
    loom_target_compile_report_format_options_initialize(&report_options);
    report_options.mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS;
    status = loom_target_compile_report_format_text(
        &compile_report, &report_options, &result->actual_output);
  } else if (emit_output && output_kind == LOOM_CHECK_PASS_OUTPUT_PASS_REPORT) {
    // Wall-clock duration is observational data, not part of the stable
    // compiler-semantic contract checked by golden fixtures.
    for (iree_host_size_t i = 0; i < pass_report.invocation_count; ++i) {
      pass_report.invocations[i].duration_nanoseconds = 0;
    }
    loom_output_stream_t output_stream;
    loom_output_stream_for_builder(&result->actual_output, &output_stream);
    status = loom_pass_report_format_json(&pass_report, &output_stream);
  } else if (emit_output) {
    // Inspect the transformed IR, not the durable export projection that
    // materializes retained target contexts as additional definitions.
    loomc_module_interop_view_t view = {0};
    status = loom_check_compile_project_module(module, &diagnostic_collector,
                                               allocator, &view, &verified);
    const loom_target_low_descriptor_registry_t low_registry =
        loom_target_environment_low_descriptor_registry(
            environment->target_environment);
    loom_text_low_asm_environment_t low_asm_environment = {0};
    loom_low_descriptor_text_asm_environment_initialize(&low_registry.registry,
                                                        &low_asm_environment);
    const loom_text_print_options_t print_options = {
        .flags = loom_check_pass_print_flags(test_case),
        .low_asm_environment = low_asm_environment,
    };
    if (iree_status_is_ok(status) && verified) {
      status = loom_text_print_module_to_builder_with_options(
          view.module, &result->actual_output, &print_options);
    }
    if (iree_status_is_ok(status) && verified) {
      bool valid_output = false;
      const loomc_module_serialize_options_t reprint_options = {
          .type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
          .structure_size = sizeof(reprint_options),
          .format = LOOMC_SOURCE_FORMAT_TEXT,
          .text_presentation = iree_all_bits_set(test_case->output_flags,
                                                 LOOM_TEST_OUTPUT_LOW_ASM)
                                   ? LOOMC_MODULE_TEXT_PRESENTATION_PRESERVE
                                   : LOOMC_MODULE_TEXT_PRESENTATION_DEFAULT,
          .print_locations = iree_all_bits_set(test_case->output_flags,
                                               LOOM_TEST_OUTPUT_LOCATIONS),
      };
      status = loom_check_validate_printed_ir(
          iree_string_builder_view(&result->actual_output), session, block_pool,
          &reprint_options, result, &valid_output);
      if (iree_status_is_ok(status) && !valid_output) {
        result->raw_outcome = LOOM_CHECK_FAIL;
        emit_output = false;
      }
    }
  }
  // Reports borrow target and module identities until formatting completes.
  loom_pass_report_deinitialize(&pass_report);
  loom_target_compile_report_deinitialize(&compile_report);
  loomc_pass_program_release(pass_program);
  loomc_target_profile_release(target_profile);
  loomc_module_release(module);
  if (iree_status_is_ok(status) && emit_output) {
    result->has_actual_output = true;
  }

  if (iree_status_is_ok(status) && emit_output) {
    status = loom_check_compare_output(test_case, allocator, result);
  }

  iree_arena_deinitialize(&diagnostic_arena);
  return status;
}

iree_status_t loom_check_execute_pass(
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_input_t* input_request,
    const loom_check_environment_t* environment,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  return loom_check_execute_pass_with_output(
      test_case, case_index, report, filename, input_request, environment,
      block_pool, allocator, result, LOOM_CHECK_PASS_OUTPUT_IR);
}

iree_status_t loom_check_execute_pass_report(
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_input_t* input_request,
    const loom_check_environment_t* environment,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  return loom_check_execute_pass_with_output(
      test_case, case_index, report, filename, input_request, environment,
      block_pool, allocator, result, LOOM_CHECK_PASS_OUTPUT_PASS_REPORT);
}

iree_status_t loom_check_execute_compile_report(
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_input_t* input_request,
    const loom_check_environment_t* environment,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  return loom_check_execute_pass_with_output(
      test_case, case_index, report, filename, input_request, environment,
      block_pool, allocator, result, LOOM_CHECK_PASS_OUTPUT_COMPILE_REPORT);
}
