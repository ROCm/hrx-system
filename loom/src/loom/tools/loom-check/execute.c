// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/execute.h"

#include "loom/codegen/low/lower/lower.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/diagnostic.h"
#include "loom/error/json_sink.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/pass/builtin_registry.h"
#include "loom/pass/tooling.h"
#include "loom/target/compile_report_format.h"
#include "loom/target/predicate.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/requirements.h"
#include "loom/tools/loom-format/convert.h"
#include "loom/util/diff.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"
#include "loom/verify/verify.h"

void loom_check_result_initialize(iree_allocator_t allocator,
                                  loom_check_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));
  iree_string_builder_initialize(allocator, &out_result->detail);
  iree_string_builder_initialize(allocator, &out_result->diff_hunk_json);
  iree_string_builder_initialize(allocator, &out_result->actual_output);
  iree_string_builder_initialize(allocator, &out_result->update_edit.text);
  iree_string_builder_initialize(allocator, &out_result->annotation_edits.json);
  iree_string_builder_initialize(allocator, &out_result->diagnostic_json);
}

void loom_check_result_deinitialize(loom_check_result_t* result) {
  iree_string_builder_deinitialize(&result->diagnostic_json);
  iree_string_builder_deinitialize(&result->annotation_edits.json);
  iree_string_builder_deinitialize(&result->update_edit.text);
  iree_string_builder_deinitialize(&result->actual_output);
  iree_string_builder_deinitialize(&result->diff_hunk_json);
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
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      "\"expected_start_line\": %zu, \"expected_line_count\": %zu, "
      "\"actual_start_line\": %zu, \"actual_line_count\": %zu, \"lines\": [",
      hunk->expected_start_line, hunk->expected_line_count,
      hunk->actual_start_line, hunk->actual_line_count));
  for (iree_host_size_t i = 0; i < hunk->line_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
    }
    const loom_diff_hunk_line_t* line = &diff->lines[hunk->line_offset + i];
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "{\"kind\": "));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
        stream, loom_diff_hunk_line_kind_name(line->kind)));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ", \"text\": "));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(stream, line->text));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]}"));
  return iree_ok_status();
}

static iree_status_t loom_check_write_diff_json_hunks(
    const loom_diff_result_t* diff, loom_check_result_t* result) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&result->diff_hunk_json, &stream);
  for (iree_host_size_t i = 0; i < diff->hunk_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\n"));
    }
    IREE_RETURN_IF_ERROR(
        loom_check_write_diff_hunk_json(diff, &diff->hunks[i], &stream));
  }
  result->diff_hunk_count = diff->hunk_count;
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
  loom_output_stream_for_builder(&result->diagnostic_json, &stream);
  if (result->diagnostic_count > 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\n"));
  }
  IREE_RETURN_IF_ERROR(
      loom_diagnostic_json_write_object(&stream, diagnostic, type_formatter));
  ++result->diagnostic_count;
  return iree_ok_status();
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

iree_status_t loom_check_context_register_and_finalize(
    const loom_check_environment_t* environment, loom_context_t* context) {
  IREE_RETURN_IF_ERROR(environment->register_context.fn(
      environment->register_context.user_data, context));
  return loom_context_finalize(context);
}

iree_status_t loom_check_environment_initialize_low_descriptor_registry(
    const loom_check_environment_t* environment,
    loom_target_low_descriptor_registry_t* out_registry) {
  *out_registry = (loom_target_low_descriptor_registry_t){0};
  return environment->initialize_low_descriptor_registry.fn(
      environment->initialize_low_descriptor_registry.user_data, out_registry);
}

iree_status_t loom_check_environment_initialize_low_lower_policy_registry(
    const loom_check_environment_t* environment,
    loom_low_lower_policy_registry_t* out_registry) {
  *out_registry = (loom_low_lower_policy_registry_t){0};
  return environment->initialize_low_lower_policy_registry.fn(
      environment->initialize_low_lower_policy_registry.user_data,
      out_registry);
}

iree_status_t loom_check_environment_initialize_math_policy_registry(
    const loom_check_environment_t* environment,
    loom_target_math_policy_registry_t* out_registry) {
  *out_registry = (loom_target_math_policy_registry_t){0};
  return environment->initialize_math_policy_registry.fn(
      environment->initialize_math_policy_registry.user_data, out_registry);
}

//===----------------------------------------------------------------------===//
// Execution
//===----------------------------------------------------------------------===//

iree_status_t loom_check_execute_case(
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_environment_t* environment, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  bool continue_execution = true;
  IREE_RETURN_IF_ERROR(loom_check_preflight_requirements(
      test_case, environment, allocator, result, &continue_execution));
  if (!continue_execution) {
    return iree_ok_status();
  }

  switch (test_case->mode) {
    case LOOM_CHECK_MODE_ROUNDTRIP: {
      IREE_RETURN_IF_ERROR(
          loom_check_execute_roundtrip(test_case, filename, environment,
                                       context, block_pool, allocator, result));
      break;
    }
    case LOOM_CHECK_MODE_VERIFY: {
      IREE_RETURN_IF_ERROR(loom_check_execute_verify(
          test_case, case_index, report, filename, environment, context,
          block_pool, allocator, result));
      break;
    }
    case LOOM_CHECK_MODE_PASS: {
      IREE_RETURN_IF_ERROR(loom_check_execute_pass(
          test_case, case_index, report, filename, environment, context,
          block_pool, allocator, result));
      break;
    }
    case LOOM_CHECK_MODE_PASS_REPORT: {
      IREE_RETURN_IF_ERROR(loom_check_execute_pass_report(
          test_case, case_index, report, filename, environment, context,
          block_pool, allocator, result));
      break;
    }
    case LOOM_CHECK_MODE_FORMAT: {
      IREE_RETURN_IF_ERROR(
          loom_check_execute_format(test_case, filename, environment, context,
                                    block_pool, allocator, result));
      break;
    }
    case LOOM_CHECK_MODE_EMIT: {
      IREE_RETURN_IF_ERROR(loom_check_execute_emit(
          test_case, case_index, report, filename, environment, context,
          block_pool, allocator, result));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown check mode: %d", (int)test_case->mode);
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

static iree_status_t loom_check_verify_pass_output(
    const loom_check_case_t* test_case, iree_string_view_t filename,
    const loom_check_environment_t* environment,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    loom_module_t* module, bool* out_failed_verification) {
  *out_failed_verification = false;
  loom_source_entry_t source_entry = {0};
  loom_source_table_resolver_t resolver_data = {0};
  IREE_RETURN_IF_ERROR(loom_check_source_resolver_for_case(
      module, filename, test_case->input, &source_entry, &resolver_data));
  loom_verify_options_t verify_options = {
      .sink = {.fn = loom_check_diagnostic_collector_sink,
               .user_data = diagnostic_collector},
      .max_errors = 100,
      .source_resolver = {.fn = loom_source_table_resolve,
                          .user_data = &resolver_data},
  };
  loom_verify_result_t verify_result = {0};
  IREE_RETURN_IF_ERROR(
      loom_verify_module(module, &verify_options, &verify_result));
  *out_failed_verification = verify_result.error_count > 0;
  if (*out_failed_verification) {
    return iree_ok_status();
  }

  loom_target_low_descriptor_registry_t low_registry;
  IREE_RETURN_IF_ERROR(
      loom_check_environment_initialize_low_descriptor_registry(environment,
                                                                &low_registry));
  loom_check_diagnostic_emitter_capture_t low_diagnostic_capture = {
      .diagnostic_collector = diagnostic_collector,
      .module = module,
      .source_resolver = {.fn = loom_source_table_resolve,
                          .user_data = &resolver_data},
      .emitter = LOOM_EMITTER_VERIFIER,
  };
  loom_low_verify_options_t low_verify_options = {
      .descriptor_registry = &low_registry.registry,
      .emitter =
          {
              .fn = loom_check_diagnostic_emitter_capture_emit,
              .user_data = &low_diagnostic_capture,
          },
      .provider_list = environment->low_verify_provider_list,
      .max_errors = 100,
  };
  loom_low_verify_result_t low_verify_result = {0};
  loom_low_verify_scratch_t low_verify_scratch =
      loom_low_verify_scratch_for_module(module);
  IREE_RETURN_IF_ERROR(loom_low_verify_module(
      module, &low_verify_options, &low_verify_scratch, &low_verify_result));
  *out_failed_verification = low_verify_result.error_count > 0;
  return iree_ok_status();
}

static iree_status_t loom_check_finish_diagnostics_if_needed(
    loom_check_diagnostic_collector_t* diagnostic_collector,
    const loom_check_case_t* test_case, iree_host_size_t case_index,
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
    const loom_check_case_t* test_case) {
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
} loom_check_pass_output_kind_t;

static iree_status_t loom_check_execute_pass_with_output(
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_environment_t* environment, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result, loom_check_pass_output_kind_t output_kind) {
  loom_module_t* module = NULL;
  iree_arena_allocator_t diagnostic_arena;
  iree_arena_initialize(block_pool, &diagnostic_arena);
  loom_check_diagnostic_collector_t diagnostic_collector = {
      .arena = &diagnostic_arena,
      .host_allocator = allocator,
      .result = result,
  };
  loom_text_parse_options_t parse_options = {
      .diagnostic_sink = {.fn = loom_check_diagnostic_collector_sink,
                          .user_data = &diagnostic_collector},
      .max_errors = 20,
  };
  loom_low_descriptor_text_asm_environment_storage_t low_asm_storage = {0};
  loom_target_low_descriptor_registry_t low_registry;
  iree_status_t status =
      loom_check_environment_initialize_low_descriptor_registry(environment,
                                                                &low_registry);
  if (iree_status_is_ok(status)) {
    loom_low_descriptor_text_print_context_initialize(
        &low_registry.registry, &diagnostic_collector.type_print_context);
    loom_low_descriptor_text_asm_environment_initialize_with_diagnostics(
        &low_registry.registry, environment->low_asm_diagnostic_provider_list,
        &low_asm_storage, &parse_options.low_asm_environment);
  }
  if (iree_status_is_ok(status)) {
    status = loom_text_parse(test_case->input, filename, context, block_pool,
                             &parse_options, &module);
  }
  diagnostic_collector.module = module;
  if (!module) {
    if (iree_status_is_ok(status)) {
      status = loom_check_diagnostic_collector_finish(
          &diagnostic_collector, test_case, case_index, report, allocator,
          result);
    }
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  if (iree_status_is_ok(status) && diagnostic_collector.count > 0) {
    status = loom_check_diagnostic_collector_finish(&diagnostic_collector,
                                                    test_case, case_index,
                                                    report, allocator, result);
  }
  if (!iree_status_is_ok(status) || diagnostic_collector.count > 0) {
    loom_module_free(module);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  // Build and run the pass pipeline.
  loom_source_entry_t source_entry = {0};
  loom_source_table_resolver_t resolver_data = {0};
  status = loom_check_source_resolver_for_case(
      module, filename, test_case->input, &source_entry, &resolver_data);
  loom_check_diagnostic_emitter_capture_t pass_diagnostic_capture = {
      .diagnostic_collector = &diagnostic_collector,
      .module = module,
      .source_resolver = {.fn = loom_source_table_resolve,
                          .user_data = &resolver_data},
      .emitter = LOOM_EMITTER_PASS,
  };
  iree_diagnostic_emitter_t pass_diagnostic_emitter = {
      .fn = loom_check_diagnostic_emitter_capture_emit,
      .user_data = &pass_diagnostic_capture,
  };
  loom_pass_run_result_t run_result = {0};
  loom_target_compile_report_t compile_report = {0};
  loom_target_compile_report_t* compile_report_ref = NULL;
  if (iree_status_is_ok(status) &&
      output_kind == LOOM_CHECK_PASS_OUTPUT_COMPILE_REPORT) {
    loom_target_compile_report_initialize(
        &compile_report, iree_arena_allocator(&diagnostic_arena));
    compile_report.requested_detail_flags =
        LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS |
        LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS |
        LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS |
        LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_LEGALIZATION_ROWS;
    compile_report_ref = &compile_report;
  }
  if (iree_status_is_ok(status)) {
    loom_low_lower_policy_registry_t low_lower_policy_registry = {0};
    const loom_low_lower_policy_registry_t* low_lower_policy_registry_ref =
        NULL;
    if (environment &&
        environment->initialize_low_lower_policy_registry.fn != NULL) {
      status = loom_check_environment_initialize_low_lower_policy_registry(
          environment, &low_lower_policy_registry);
      if (iree_status_is_ok(status)) {
        low_lower_policy_registry_ref = &low_lower_policy_registry;
      }
    }
    loom_target_math_policy_registry_t math_policy_registry = {0};
    const loom_target_math_policy_registry_t* math_policy_registry_ref = NULL;
    if (iree_status_is_ok(status) && environment &&
        environment->initialize_math_policy_registry.fn != NULL) {
      status = loom_check_environment_initialize_math_policy_registry(
          environment, &math_policy_registry);
      if (iree_status_is_ok(status)) {
        math_policy_registry_ref = &math_policy_registry;
      }
    }
    loom_low_pass_environment_storage_t low_pass_environment_storage;
    loom_target_pass_predicate_provider_storage_t predicate_storage;
    loom_target_pass_predicate_provider_storage_initialize(block_pool,
                                                           &predicate_storage);
    const loom_pass_registry_t* pass_registry = loom_pass_builtin_registry();
    loom_pass_registry_storage_t pass_registry_storage = {0};
    if (iree_status_is_ok(status) && environment != NULL &&
        environment->pass_registry != NULL) {
      const loom_pass_registry_t* pass_registries[] = {
          loom_pass_builtin_registry(),
          environment->pass_registry,
      };
      status = loom_pass_registry_storage_initialize_from_registries(
          pass_registries, IREE_ARRAYSIZE(pass_registries),
          &pass_registry_storage);
      if (iree_status_is_ok(status)) {
        pass_registry =
            loom_pass_registry_storage_registry(&pass_registry_storage);
      }
    }
    loom_pass_tool_run_options_t run_options = {
        .registry = pass_registry,
        .environment = loom_low_pass_environment_storage_initialize(
            &low_registry.registry, low_lower_policy_registry_ref,
            environment ? &environment->low_legality_provider_list : NULL,
            environment ? &environment->legalizer_provider_list : NULL,
            math_policy_registry_ref, compile_report_ref,
            loom_target_selection_empty(), loom_symbol_ref_null(),
            &low_pass_environment_storage),
        .predicate_provider =
            loom_target_pass_predicate_provider(&predicate_storage),
        .block_pool = block_pool,
        .diagnostic_emitter = pass_diagnostic_emitter,
    };
    if (iree_status_is_ok(status)) {
      status = loom_pass_tool_run_flat_pipeline(module, test_case->pipeline,
                                                &run_options, &run_result);
    }
  }
  if (!iree_status_is_ok(status)) {
    status = loom_check_execute_finish_status_failure(
        status, IREE_SV("pass pipeline"), result);
    loom_module_free(module);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }
  bool has_expected_output = loom_check_case_has_expected_output(test_case);
  if (run_result.error_count > 0) {
    bool diagnostics_failed = false;
    status = loom_check_finish_diagnostics_if_needed(
        &diagnostic_collector, test_case, case_index, report, allocator, result,
        &diagnostics_failed);
    if (!iree_status_is_ok(status) || diagnostics_failed ||
        output_kind != LOOM_CHECK_PASS_OUTPUT_COMPILE_REPORT ||
        !has_expected_output) {
      loom_module_free(module);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
  } else {
    bool failed_verification = false;
    status = loom_check_verify_pass_output(test_case, filename, environment,
                                           &diagnostic_collector, module,
                                           &failed_verification);
    bool diagnostics_failed = false;
    if (iree_status_is_ok(status)) {
      status = loom_check_finish_diagnostics_if_needed(
          &diagnostic_collector, test_case, case_index, report, allocator,
          result, &diagnostics_failed);
    }
    if (!iree_status_is_ok(status) || diagnostics_failed ||
        failed_verification ||
        (diagnostic_collector.count > 0 && !has_expected_output)) {
      loom_module_free(module);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
  }

  if (output_kind == LOOM_CHECK_PASS_OUTPUT_COMPILE_REPORT) {
    loom_target_compile_report_format_options_t report_options;
    loom_target_compile_report_format_options_initialize(&report_options);
    report_options.mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS;
    status = loom_target_compile_report_format_text(
        &compile_report, &report_options, &result->actual_output);
  } else {
    loom_text_low_asm_environment_t low_asm_environment = {0};
    loom_low_descriptor_text_asm_environment_initialize(&low_registry.registry,
                                                        &low_asm_environment);
    const loom_text_print_options_t print_options = {
        .flags = LOOM_TEXT_PRINT_DEFAULT,
        .low_asm_environment = low_asm_environment,
    };
    status = loom_text_print_module_to_builder_with_options(
        module, &result->actual_output, &print_options);
  }
  loom_module_free(module);
  if (iree_status_is_ok(status)) {
    result->has_actual_output = true;
  }

  // Strip comments from the expected section.
  iree_string_builder_t stripped_expected;
  iree_string_builder_initialize(allocator, &stripped_expected);
  if (iree_status_is_ok(status)) {
    status = loom_check_strip_comments(test_case->expected, &stripped_expected);
  }
  if (iree_status_is_ok(status)) {
    iree_string_view_t actual_trimmed =
        iree_string_view_trim(iree_string_builder_view(&result->actual_output));
    iree_string_view_t expected_trimmed =
        iree_string_view_trim(iree_string_builder_view(&stripped_expected));
    if (iree_string_view_equal(actual_trimmed, expected_trimmed)) {
      result->raw_outcome = LOOM_CHECK_PASS;
    } else {
      result->raw_outcome = LOOM_CHECK_FAIL;
      status = loom_check_result_record_diff(expected_trimmed, actual_trimmed,
                                             allocator, result);
    }
  }

  iree_string_builder_deinitialize(&stripped_expected);
  iree_arena_deinitialize(&diagnostic_arena);
  return status;
}

iree_status_t loom_check_execute_pass(
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_environment_t* environment, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  return loom_check_execute_pass_with_output(
      test_case, case_index, report, filename, environment, context, block_pool,
      allocator, result, LOOM_CHECK_PASS_OUTPUT_IR);
}

iree_status_t loom_check_execute_pass_report(
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_environment_t* environment, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  return loom_check_execute_pass_with_output(
      test_case, case_index, report, filename, environment, context, block_pool,
      allocator, result, LOOM_CHECK_PASS_OUTPUT_COMPILE_REPORT);
}

iree_status_t loom_check_execute_format(
    const loom_check_case_t* test_case, iree_string_view_t filename,
    const loom_check_environment_t* environment, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  loom_module_format_t format = LOOM_MODULE_FORMAT_AUTO;
  IREE_RETURN_IF_ERROR(loom_module_format_parse(test_case->format_target,
                                                /*allow_auto=*/false, &format));

  iree_string_builder_t stripped_input;
  iree_string_builder_initialize(allocator, &stripped_input);
  iree_status_t status =
      loom_check_strip_comments(test_case->input, &stripped_input);

  loom_check_diagnostic_capture_t diagnostic_capture = {
      .detail = &result->detail,
      .result = result,
  };
  loom_target_low_descriptor_registry_t low_registry = {0};
  if (iree_status_is_ok(status)) {
    status = loom_check_environment_initialize_low_descriptor_registry(
        environment, &low_registry);
  }
  loom_text_low_asm_environment_t low_asm_environment = {0};
  loom_low_descriptor_text_asm_environment_storage_t low_asm_storage = {0};
  if (iree_status_is_ok(status)) {
    loom_low_descriptor_text_asm_environment_initialize_with_diagnostics(
        &low_registry.registry, environment->low_asm_diagnostic_provider_list,
        &low_asm_storage, &low_asm_environment);
  }
  loom_format_convert_options_t to_format_options = {
      .input_format = LOOM_MODULE_FORMAT_TEXT,
      .output_format = format,
      .diagnostic_sink = {.fn = loom_check_diagnostic_capture_sink,
                          .user_data = &diagnostic_capture},
      .low_asm_environment = low_asm_environment,
  };
  loom_format_output_t formatted_output = {0};
  iree_string_view_t stripped_view = iree_string_builder_view(&stripped_input);
  if (iree_status_is_ok(status)) {
    status = loom_format_convert(
        iree_make_const_byte_span((const uint8_t*)stripped_view.data,
                                  stripped_view.size),
        filename, context, block_pool, &to_format_options, &formatted_output,
        allocator);
  }
  iree_string_builder_deinitialize(&stripped_input);

  loom_format_output_t text_output = {0};
  loom_format_convert_options_t to_text_options = {
      .input_format = format,
      .output_format = LOOM_MODULE_FORMAT_TEXT,
      .diagnostic_sink = {.fn = loom_check_diagnostic_capture_sink,
                          .user_data = &diagnostic_capture},
      .low_asm_environment = low_asm_environment,
  };
  if (iree_status_is_ok(status)) {
    status =
        loom_format_convert(iree_make_const_byte_span(formatted_output.data,
                                                      formatted_output.length),
                            filename, context, block_pool, &to_text_options,
                            &text_output, allocator);
  }
  loom_format_output_deinitialize(&formatted_output, allocator);

  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(
        &result->actual_output,
        iree_make_string_view((const char*)text_output.data,
                              text_output.length));
  }
  loom_format_output_deinitialize(&text_output, allocator);
  IREE_RETURN_IF_ERROR(status);
  result->has_actual_output = true;

  iree_string_builder_t stripped_expected;
  iree_string_builder_initialize(allocator, &stripped_expected);
  status = loom_check_strip_comments(test_case->expected, &stripped_expected);

  if (iree_status_is_ok(status)) {
    iree_string_view_t actual_trimmed =
        iree_string_view_trim(iree_string_builder_view(&result->actual_output));
    iree_string_view_t expected_trimmed =
        iree_string_view_trim(iree_string_builder_view(&stripped_expected));
    if (iree_string_view_equal(actual_trimmed, expected_trimmed)) {
      result->raw_outcome = LOOM_CHECK_PASS;
    } else {
      result->raw_outcome = LOOM_CHECK_FAIL;
      status = loom_check_result_record_diff(expected_trimmed, actual_trimmed,
                                             allocator, result);
    }
  }

  iree_string_builder_deinitialize(&stripped_expected);
  return status;
}
