// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-test-loom: executes check.case records from ordinary Loom modules.

#include "loom/tools/iree-test-loom/main.h"

#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/path.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/stdio_stream.h"
#include "loom/sanitizer/options.h"
#include "loom/tooling/cli/help.h"
#include "loom/tooling/config/config.h"
#include "loom/tooling/context/context.h"
#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tooling/io/file.h"
#include "loom/tooling/testbench/device_event.h"
#include "loom/tooling/testbench/executor.h"
#include "loom/tooling/testbench/issue_report.h"
#include "loom/tooling/testbench/reference.h"
#include "loom/tooling/testbench/requirements.h"
#include "loom/tools/iree-test-loom/library_linker.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"
#include "loom/verify/verify.h"

IREE_FLAG(string, case, "",
          "Optional check.case symbol to execute, such as '@smoke'. Empty "
          "executes all cases in source order.");
IREE_FLAG(int32_t, sample, -1,
          "Optional concrete sample ordinal to execute for the selected case "
          "or cases. Negative executes all planned samples.");
IREE_FLAG_LIST(
    string, library,
    "Loom bytecode library linked into the authored input module. Repeat as "
    "--library=path.loombc. Libraries are linked whole in argument order.");
IREE_FLAG_NAMED(int32_t, max_samples_per_case, "max-samples-per-case",
                LOOM_TESTBENCH_DEFAULT_MAX_SAMPLES_PER_CASE,
                "Maximum number of samples planned per check.case.");
IREE_FLAG(string, pipeline, "default",
          "Pass pipeline used for HAL kernel launches. Use 'default', "
          "'none', '@symbol', or a comma-separated pass list.");
IREE_FLAG_LIST(
    string, config,
    "Compile-time config binding for HAL kernel launches. Repeat as "
    "--config=key=value. Bindings not referenced by the loaded module are "
    "ignored.");
IREE_FLAG_LIST_NAMED(
    string, config_file, "config-file",
    "JSON/JSONC config object file for HAL kernel launches. Repeat for "
    "multiple files. Nested object keys are flattened with '.' separators.");
IREE_FLAG(string, sanitizer, "none",
          "Sanitizer checks inserted by the target pipeline. Use 'none', "
          "'access', 'value', 'operation', 'race', 'asan', 'ubsan', 'tsan', "
          "'all', or a '|' separated list.");
IREE_FLAG_NAMED(string, sanitizer_reporting, "sanitizer-reporting", "default",
                "Sanitizer reporting mode used by the target pipeline. Use "
                "'default', 'trap', or 'report-only'.");

enum {
  // Target-linked requirement providers.
  IREE_TEST_LOOM_MAX_REQUIREMENT_PROVIDERS = 8,
  // Maximum device events retained per sample for structured event
  // expectations.
  IREE_TEST_LOOM_DEVICE_EVENT_CAPACITY = 256,
};

typedef struct iree_test_loom_file_provider_t {
  // Host allocator used for resolved path storage.
  iree_allocator_t host_allocator;
  // Borrowed directory containing the input module for relative fixture reads.
  iree_string_view_t input_dir;
} iree_test_loom_file_provider_t;

static bool iree_test_loom_path_is_absolute(iree_string_view_t path) {
  if (iree_string_view_is_empty(path)) {
    return false;
  }
  if (path.data[0] == '/' || path.data[0] == '\\') {
    return true;
  }
  if (path.size >= 3 && path.data[1] == ':' &&
      (path.data[2] == '/' || path.data[2] == '\\')) {
    const char drive = path.data[0];
    return (drive >= 'a' && drive <= 'z') || (drive >= 'A' && drive <= 'Z');
  }
  return false;
}

static iree_status_t iree_test_loom_dup_string_view(iree_string_view_t value,
                                                    iree_allocator_t allocator,
                                                    char** out_value) {
  *out_value = NULL;
  iree_host_size_t storage_size = 0;
  if (!iree_host_size_checked_add(value.size, 1, &storage_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "string storage size overflow");
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, storage_size, (void**)&storage));
  if (value.size != 0) {
    memcpy(storage, value.data, value.size);
  }
  storage[value.size] = '\0';
  *out_value = storage;
  return iree_ok_status();
}

static iree_status_t iree_test_loom_resolve_file_read_path(
    const iree_test_loom_file_provider_t* provider, iree_string_view_t path,
    char** out_path) {
  *out_path = NULL;
  if (loom_tooling_file_path_is_stdio(path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "check.file.read paths must name a file");
  }
  if (iree_test_loom_path_is_absolute(path) ||
      iree_string_view_is_empty(provider->input_dir)) {
    return iree_test_loom_dup_string_view(path, provider->host_allocator,
                                          out_path);
  }
  return iree_file_path_join(provider->input_dir, path,
                             provider->host_allocator, out_path);
}

static iree_status_t iree_test_loom_open_file_for_read(
    void* user_data, iree_string_view_t path, iree_io_stream_t** out_stream) {
  *out_stream = NULL;
  const iree_test_loom_file_provider_t* provider =
      (const iree_test_loom_file_provider_t*)user_data;
  char* resolved_path = NULL;
  IREE_RETURN_IF_ERROR(
      iree_test_loom_resolve_file_read_path(provider, path, &resolved_path));
  iree_status_t status = iree_io_stdio_stream_open(
      IREE_IO_STDIO_STREAM_MODE_READ, iree_make_cstring_view(resolved_path),
      provider->host_allocator, out_stream);
  iree_allocator_free(provider->host_allocator, resolved_path);
  return status;
}

static iree_status_t iree_test_loom_register_context(void* user_data,
                                                     loom_context_t* context) {
  const iree_test_loom_configuration_t* configuration =
      (const iree_test_loom_configuration_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_tooling_context_register_tool_dialects(context));
  if (configuration->register_context.fn == NULL) {
    return iree_ok_status();
  }
  return configuration->register_context.fn(
      configuration->register_context.user_data, context);
}

static iree_status_t iree_test_loom_verify_run_module(
    loom_run_module_t* run_module) {
  const loom_verify_options_t verify_options = {
      .sink = {.fn = loom_diagnostic_stderr_sink},
      .max_errors = 20,
      .source_resolver = loom_run_module_source_resolver(run_module),
  };
  loom_verify_result_t verify_result = {0};
  IREE_RETURN_IF_ERROR(
      loom_verify_module(run_module->module, &verify_options, &verify_result));
  if (verify_result.error_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "input module failed verification");
  }
  return iree_ok_status();
}

static iree_string_view_t iree_test_loom_normalize_case_name(
    iree_string_view_t case_name) {
  case_name = iree_string_view_trim(case_name);
  if (iree_string_view_starts_with(case_name, IREE_SV("@"))) {
    return iree_string_view_substr(case_name, 1, IREE_HOST_SIZE_MAX);
  }
  return case_name;
}

static bool iree_test_loom_case_matches_selection(
    const loom_testbench_case_plan_t* case_plan,
    iree_string_view_t selected_case_name) {
  return iree_string_view_is_empty(selected_case_name) ||
         iree_string_view_equal(case_plan->name, selected_case_name);
}

static iree_status_t iree_test_loom_validate_sample_flag(
    iree_host_size_t sample_count, iree_host_size_t* out_sample_ordinal,
    bool* out_has_sample) {
  *out_sample_ordinal = 0;
  *out_has_sample = false;
  if (FLAG_sample < 0) {
    return iree_ok_status();
  }
  *out_sample_ordinal = (iree_host_size_t)FLAG_sample;
  *out_has_sample = true;
  if (*out_sample_ordinal >= sample_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "--sample=%" PRIhsz
                            " exceeds selected case sample count %" PRIhsz,
                            *out_sample_ordinal, sample_count);
  }
  return iree_ok_status();
}

static bool iree_test_loom_case_has_kernel_launch(
    const loom_testbench_case_plan_t* case_plan) {
  return case_plan->kernel_launch_count != 0;
}

static bool iree_test_loom_case_has_device_event_expectation(
    const loom_testbench_case_plan_t* case_plan) {
  for (iree_host_size_t i = 0; i < case_plan->expectation_count; ++i) {
    const loom_testbench_expectation_plan_t* expectation =
        &case_plan->expectations[i];
    if (expectation->kind == LOOM_TESTBENCH_EXPECTATION_EVENT &&
        iree_string_view_equal(expectation->event.provider,
                               IREE_SV("device"))) {
      return true;
    }
  }
  return false;
}

static bool iree_test_loom_selected_cases_have_device_event_expectation(
    const loom_testbench_module_plan_t* module_plan,
    iree_string_view_t selected_case_name) {
  for (iree_host_size_t i = 0; i < module_plan->case_count; ++i) {
    const loom_testbench_case_plan_t* case_plan = &module_plan->cases[i];
    if (iree_test_loom_case_matches_selection(case_plan, selected_case_name) &&
        iree_test_loom_case_has_device_event_expectation(case_plan)) {
      return true;
    }
  }
  return false;
}

static iree_status_t iree_test_loom_append_skipped_case(
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_requirement_result_t* requirement_result,
    loom_json_array_writer_t* skipped_cases) {
  IREE_RETURN_IF_ERROR(loom_json_array_begin_element(skipped_cases));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(skipped_cases->stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("case"), case_plan->name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("provider"), requirement_result->provider));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("op"),
      loom_testbench_requirement_op_kind_name(requirement_result->op_kind)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("code"),
      loom_testbench_requirement_skip_code_name(requirement_result->code)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("provider_code"), requirement_result->provider_code));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("display_message"),
      requirement_result->display_message));
  return loom_json_object_end(&object);
}

static iree_status_t iree_test_loom_append_planning_issue(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_issue_t* issue,
    loom_json_array_writer_t* planning_issues) {
  IREE_RETURN_IF_ERROR(loom_json_array_begin_element(planning_issues));
  return loom_testbench_issue_write_json(module_plan, issue,
                                         planning_issues->stream);
}

static iree_status_t iree_test_loom_append_case_planning_issues(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    loom_json_array_writer_t* planning_issues,
    iree_host_size_t* inout_planning_issue_count) {
  for (iree_host_size_t i = 0; i < case_plan->issue_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_test_loom_append_planning_issue(
        module_plan, &case_plan->issues[i], planning_issues));
    ++*inout_planning_issue_count;
  }
  return iree_ok_status();
}

static iree_status_t iree_test_loom_configure_hal_actual_sequence(
    const iree_test_loom_configuration_t* configuration,
    loom_run_session_t* session, const loom_run_module_t* run_module,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    const loom_tooling_config_set_t* config_set,
    const loom_sanitizer_options_t* sanitizer_options,
    loom_run_hal_testbench_context_t* hal_context,
    loom_testbench_case_execution_options_t* execution_options,
    loom_run_hal_testbench_actual_sequence_t* out_sequence) {
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_context_ensure_runtime(hal_context));

  execution_options->materializer.device = hal_context->runtime.device;
  execution_options->materializer.device_allocator =
      iree_hal_device_allocator(hal_context->runtime.device);
  execution_options->materializer.buffer_params =
      loom_run_hal_testbench_host_visible_buffer_params();

  const loom_run_hal_testbench_actual_sequence_options_t sequence_options = {
      .context = hal_context,
      .session = session,
      .target_environment = configuration->target_environment,
      .filename = run_module->filename,
      .source = run_module->source,
      .pipeline = iree_make_cstring_view(FLAG_pipeline),
      .sanitizer = *sanitizer_options,
      .config_set = config_set,
      .test_module = module_plan->module,
      .case_plan = case_plan,
  };
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_actual_sequence_initialize(
      &sequence_options, out_sequence));
  execution_options->invocation.kernel_launch =
      loom_run_hal_testbench_actual_sequence_provider(out_sequence);
  return iree_ok_status();
}

static iree_status_t iree_test_loom_run_case_samples(
    const iree_test_loom_configuration_t* configuration,
    loom_run_session_t* session, const loom_run_module_t* run_module,
    const loom_testbench_module_plan_t* module_plan,
    iree_host_size_t case_index,
    const loom_testbench_case_execution_options_t* base_execution_options,
    const loom_tooling_config_set_t* config_set,
    const loom_sanitizer_options_t* sanitizer_options,
    loom_run_hal_testbench_context_t* hal_context,
    iree_arena_allocator_t* arena, loom_json_array_writer_t* samples,
    loom_json_array_writer_t* skipped_cases,
    iree_host_size_t* inout_sample_count,
    iree_host_size_t* inout_failed_sample_count,
    iree_host_size_t* inout_skipped_case_count) {
  const loom_testbench_case_plan_t* case_plan = &module_plan->cases[case_index];
  loom_testbench_requirement_provider_t
      requirement_providers[IREE_TEST_LOOM_MAX_REQUIREMENT_PROVIDERS] = {0};
  iree_host_size_t requirement_provider_count = 0;
  if (configuration->populate_requirement_providers.fn != NULL) {
    IREE_RETURN_IF_ERROR(configuration->populate_requirement_providers.fn(
        configuration->populate_requirement_providers.user_data, hal_context,
        IREE_ARRAYSIZE(requirement_providers), requirement_providers,
        &requirement_provider_count));
  }
  if (requirement_provider_count > IREE_ARRAYSIZE(requirement_providers)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "iree-test-loom requirement provider capacity exceeded");
  }
  loom_testbench_requirement_provider_registry_t requirement_registry = {0};
  loom_testbench_requirement_provider_registry_initialize(
      requirement_providers, requirement_provider_count, &requirement_registry);
  loom_testbench_requirement_result_t requirement_result = {0};
  IREE_RETURN_IF_ERROR(loom_testbench_evaluate_case_requirements(
      module_plan->module, case_plan, &requirement_registry,
      &requirement_result));
  if (requirement_result.skipped) {
    IREE_RETURN_IF_ERROR(iree_test_loom_append_skipped_case(
        case_plan, &requirement_result, skipped_cases));
    ++*inout_skipped_case_count;
    return iree_ok_status();
  }

  iree_host_size_t selected_sample_ordinal = 0;
  bool has_selected_sample = false;
  IREE_RETURN_IF_ERROR(iree_test_loom_validate_sample_flag(
      case_plan->sample_count, &selected_sample_ordinal, &has_selected_sample));

  iree_status_t status = iree_ok_status();
  loom_testbench_case_execution_options_t execution_options =
      *base_execution_options;
  loom_run_hal_testbench_actual_sequence_t hal_actual_sequence = {0};
  loom_testbench_reference_matmul_oracle_options_t matmul_oracle_options = {0};
  loom_testbench_oracle_provider_t oracle_providers[2] = {0};
  bool hal_actual_sequence_initialized = false;
  if (iree_test_loom_case_has_kernel_launch(case_plan)) {
    status = iree_test_loom_configure_hal_actual_sequence(
        configuration, session, run_module, module_plan, case_plan, config_set,
        sanitizer_options, hal_context, &execution_options,
        &hal_actual_sequence);
    hal_actual_sequence_initialized = iree_status_is_ok(status);
    if (iree_status_is_ok(status)) {
      matmul_oracle_options =
          (loom_testbench_reference_matmul_oracle_options_t){
              .device = hal_context->runtime.device,
              .device_allocator =
                  iree_hal_device_allocator(hal_context->runtime.device),
              .result_buffer_params =
                  loom_run_hal_testbench_host_visible_buffer_params(),
              .host_allocator = execution_options.materializer.host_allocator,
          };
      loom_testbench_reference_matmul_oracle_provider_initialize(
          &matmul_oracle_options, &oracle_providers[0]);
      loom_testbench_reference_tiled_matmul_oracle_provider_initialize(
          &matmul_oracle_options, &oracle_providers[1]);
      execution_options.invocation.oracle_providers =
          loom_make_testbench_oracle_provider_list(
              oracle_providers, IREE_ARRAYSIZE(oracle_providers));
    }
  }

  loom_testbench_prepared_case_t prepared_case = {0};
  if (iree_status_is_ok(status)) {
    status = loom_testbench_prepare_case_execution(
        &execution_options, module_plan, case_index, arena, &prepared_case);
  }

  bool executor_initialized = false;
  loom_testbench_case_executor_t executor = {0};
  if (iree_status_is_ok(status)) {
    status = loom_testbench_case_executor_initialize(
        &prepared_case, &execution_options, &executor);
    executor_initialized = iree_status_is_ok(status);
  }

  const iree_host_size_t begin_sample =
      has_selected_sample ? selected_sample_ordinal : 0;
  const iree_host_size_t end_sample = has_selected_sample
                                          ? selected_sample_ordinal + 1
                                          : case_plan->sample_count;
  for (iree_host_size_t sample_ordinal = begin_sample;
       iree_status_is_ok(status) && sample_ordinal < end_sample;
       ++sample_ordinal) {
    loom_testbench_case_sample_result_t sample_result = {0};
    status = loom_testbench_run_case_sample(&executor, sample_ordinal,
                                            &sample_result);
    if (iree_status_is_ok(status)) {
      status = loom_json_array_begin_element(samples);
    }
    if (iree_status_is_ok(status)) {
      status = loom_testbench_case_sample_result_write_json(&sample_result,
                                                            samples->stream);
    }
    if (iree_status_is_ok(status)) {
      ++*inout_sample_count;
      if (!sample_result.passed) {
        ++*inout_failed_sample_count;
      }
    }
  }

  if (executor_initialized) {
    loom_testbench_case_executor_deinitialize(&executor);
  }
  if (hal_actual_sequence_initialized) {
    loom_run_hal_testbench_actual_sequence_deinitialize(&hal_actual_sequence);
  }
  iree_arena_reset(arena);
  return status;
}

static iree_status_t iree_test_loom_append_config_flags(
    loom_tooling_config_set_t* config_set) {
  const iree_flag_string_list_t assignments = FLAG_config_list();
  for (iree_host_size_t i = 0; i < assignments.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_set_append_assignment(
        config_set, assignments.values[i]));
  }
  return iree_ok_status();
}

static iree_status_t iree_test_loom_append_config_files(
    loom_tooling_config_set_t* config_set, iree_allocator_t allocator) {
  const iree_flag_string_list_t paths = FLAG_config_file_list();
  for (iree_host_size_t i = 0; i < paths.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_set_append_json_file(
        config_set, paths.values[i], allocator));
  }
  return iree_ok_status();
}

static iree_status_t iree_test_loom_write_report(
    iree_host_size_t case_count, iree_host_size_t sample_count,
    iree_host_size_t failed_sample_count, iree_host_size_t skipped_case_count,
    iree_host_size_t planning_issue_count, iree_string_view_t samples,
    iree_string_view_t skipped_cases, iree_string_view_t planning_issues,
    iree_string_builder_t* output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("format"), IREE_SV("loom.test.v0")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("case_count"), case_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("sample_count"), sample_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("failed_sample_count"), failed_sample_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("skipped_case_count"), skipped_case_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("planning_issue_count"), planning_issue_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("samples")));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(&stream, samples));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("skipped_cases")));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(&stream, skipped_cases));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("planning_issues")));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(&stream, planning_issues));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  return loom_output_stream_write_char(&stream, '\n');
}

static void iree_test_loom_print_agents_markdown(FILE* stream) {
  fprintf(
      stream,
      "## iree-test-loom\n"
      "\n"
      "`iree-test-loom` executes `check.case` records from ordinary Loom\n"
      "modules and writes a structured `loom.test.v0` JSON report. Use it for\n"
      "correctness before promoting the same cases to `check.benchmark` rows.\n"
      "\n"
      "### Common flows\n"
      "\n"
      "```shell\n"
      "iree-test-loom module.loom\n"
      "iree-test-loom module.loom --case=@case_q8_block_unroll_wg64\n"
      "iree-test-loom module.loom --case=@sampled_choice --sample=1\n"
      "iree-test-loom test.loom --library=motifs.loombc\n"
      "iree-test-loom module.loom --max-samples-per-case=16\n"
      "iree-test-loom module.loom --pipeline=@hal_actual_pipeline\n"
      "iree-test-loom module.loom --config=model.hidden_size=4096\n"
      "iree-test-loom module.loom --config-file=model_config.jsonc\n"
      "iree-test-loom module.loom --sanitizer=tsan\n"
      "iree-test-loom module.loom --sanitizer=asan "
      "--sanitizer-reporting=report-only\n"
      "```\n"
      "\n"
      "`--case=@name` selects one checked case; empty selection runs cases in\n"
      "source order. `--sample=N` selects one planned sample for sampled "
      "cases.\n"
      "`--max-samples-per-case=N` bounds planning for generator-heavy cases.\n"
      "\n"
      "### Kernel launches\n"
      "\n"
      "A case can mix reference/oracle checks with HAL kernel launches. "
      "Kernel\n"
      "launches use the selected HAL artifact provider, target provider, "
      "and HAL device linked into this binary. "
      "`--pipeline=default|none|@symbol|pass,list` controls the HAL kernel "
      "compile pipeline. `--config=key=value` and `--config-file=path` "
      "materialize config declarations in the private compile copy before "
      "lowering runs. Case sample values remain runtime invocation inputs. "
      "`--sanitizer=...` and "
      "`--sanitizer-reporting=...` enable the same target-pipeline "
      "instrumentation accepted by `iree-benchmark-loom`.\n"
      "\n"
      "### Report shape\n"
      "\n"
      "```shell\n"
      "iree-test-loom module.loom --case=@smoke | jq '.failed_sample_count'\n"
      "iree-test-loom module.loom | jq '.samples[] | {case, sample_ordinal, "
      "passed}'\n"
      "iree-test-loom module.loom | jq '.samples[] | select(.issues) | "
      "{case, issue: .issues[]}'\n"
      "iree-test-loom module.loom | jq '.skipped_cases[]? | {case, provider, "
      "op, code, provider_code}'\n"
      "iree-test-loom module.loom | jq '.planning_issues[]? | {case, kind, "
      "op, source_location, fix_hint}'\n"
      "```\n"
      "\n"
      "The report carries `case_count`, `sample_count`, "
      "`failed_sample_count`,\n"
      "`skipped_case_count`, `planning_issue_count`, `samples`, "
      "`skipped_cases`,\n"
      "and `planning_issues`. Skipped cases use\n"
      "stable `op` and `code` fields; `provider_code` is provider-defined and\n"
      "`display_message` is human-facing only. Failed samples may include "
      "`issues`\n"
      "when execution could not reach expectations, such as a compile "
      "rejection\n"
      "with stable `category`, `provider`, `stage`, and `kind` fields. "
      "Planning\n"
      "issues carry stable\n"
      "`kind`, `case`, `op`, `source_location`, and optional `fix_hint` "
      "fields.\n"
      "A nonzero failed sample or planning issue count makes the process fail\n"
      "after the JSON report is written.\n");
}

int iree_test_loom_main(int argc, char** argv,
                        const iree_test_loom_configuration_t* configuration) {
  iree_flags_set_usage(
      configuration->tool_name,
      "Executes check.case records from a normal Loom module.\n"
      "\n"
      "Usage:\n"
      "  iree-test-loom file.loom --case=@smoke\n"
      "  cat module.loom | iree-test-loom -\n"
      "  iree-test-loom --agents_md\n");
  for (int i = 1; i < argc; ++i) {
    if (loom_tooling_cli_is_agents_markdown_arg(argv[i])) {
      iree_test_loom_print_agents_markdown(stdout);
      return 0;
    }
  }
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  loom_tooling_cli_set_default_help_filter();
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t allocator = iree_allocator_system();
  loom_tooling_config_set_t config_set;
  loom_tooling_config_set_initialize(allocator, &config_set);
  iree_io_file_contents_t* contents = NULL;
  loom_run_session_t session = {0};
  loom_run_module_t run_module = {0};
  loom_sanitizer_options_t sanitizer_options = {0};
  loom_run_hal_testbench_context_t hal_context = {0};
  loom_testbench_device_event_capture_t device_event_capture = {0};
  bool device_event_capture_initialized = false;
  iree_arena_allocator_t plan_arena;
  memset(&plan_arena, 0, sizeof(plan_arena));
  iree_arena_allocator_t execution_arena;
  memset(&execution_arena, 0, sizeof(execution_arena));
  iree_string_builder_t sample_output;
  iree_string_builder_initialize(allocator, &sample_output);
  iree_string_builder_t skipped_output;
  iree_string_builder_initialize(allocator, &skipped_output);
  iree_string_builder_t planning_issue_output;
  iree_string_builder_initialize(allocator, &planning_issue_output);
  iree_string_builder_t report_output;
  iree_string_builder_initialize(allocator, &report_output);
  int exit_code = 0;

  loom_output_stream_t sample_stream;
  loom_output_stream_for_builder(&sample_output, &sample_stream);
  loom_json_array_writer_t samples;
  iree_status_t status = loom_json_array_begin(&sample_stream, &samples);
  loom_output_stream_t skipped_stream;
  loom_output_stream_for_builder(&skipped_output, &skipped_stream);
  loom_json_array_writer_t skipped_cases;
  if (iree_status_is_ok(status)) {
    status = loom_json_array_begin(&skipped_stream, &skipped_cases);
  }
  loom_output_stream_t planning_issue_stream;
  loom_output_stream_for_builder(&planning_issue_output,
                                 &planning_issue_stream);
  loom_json_array_writer_t planning_issues;
  if (iree_status_is_ok(status)) {
    status = loom_json_array_begin(&planning_issue_stream, &planning_issues);
  }
  if (iree_status_is_ok(status) && argc > 2) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "iree-test-loom accepts at most one input file or '-' for stdin; got "
        "%d inputs",
        argc - 1);
  }
  if (iree_status_is_ok(status) && FLAG_max_samples_per_case <= 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--max-samples-per-case must be positive; got "
                              "%d",
                              (int)FLAG_max_samples_per_case);
  }
  if (iree_status_is_ok(status)) {
    status = iree_test_loom_append_config_flags(&config_set);
  }
  if (iree_status_is_ok(status)) {
    status = iree_test_loom_append_config_files(&config_set, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_sanitizer_options_parse_checks(
        iree_make_cstring_view(FLAG_sanitizer), IREE_SV("--sanitizer"),
        &sanitizer_options);
  }
  if (iree_status_is_ok(status)) {
    status = loom_sanitizer_reporting_mode_parse(
        iree_make_cstring_view(FLAG_sanitizer_reporting),
        IREE_SV("--sanitizer-reporting"), &sanitizer_options.reporting_mode);
  }

  if (iree_status_is_ok(status)) {
    loom_run_session_options_t session_options = {0};
    loom_run_session_options_initialize(&session_options);
    session_options.host_allocator = allocator;
    session_options.register_context = (loom_run_register_context_callback_t){
        .fn = iree_test_loom_register_context,
        .user_data = (void*)configuration,
    };
    session_options.initialize_low_descriptor_registry =
        configuration->initialize_low_descriptor_registry;
    status = loom_run_session_initialize(&session_options, &session);
  }
  if (iree_status_is_ok(status)) {
    loom_run_hal_testbench_context_initialize(
        configuration->hal_artifact_provider_registry, allocator, &hal_context);
    loom_run_hal_testbench_context_set_runtime_sanitizer_options(
        &hal_context, &sanitizer_options);
  }

  const iree_string_view_t input_path =
      argc < 2 ? iree_string_view_empty() : iree_make_cstring_view(argv[1]);
  const iree_string_view_t filename =
      (argc < 2 || iree_string_view_equal(input_path, IREE_SV("-")))
          ? IREE_SV("<stdin>")
          : input_path;
  iree_test_loom_file_provider_t file_provider = {
      // Host allocator used for resolved path storage.
      .host_allocator = allocator,
      // Directory containing the input module for relative fixture reads.
      .input_dir = iree_string_view_equal(filename, IREE_SV("<stdin>"))
                       ? iree_string_view_empty()
                       : iree_file_path_dirname(filename),
  };
  iree_string_view_t source = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    status = loom_tooling_read_input_file(input_path, allocator, &contents);
    if (iree_status_is_ok(status)) {
      source = loom_tooling_file_contents_string_view(contents);
    }
  }
  if (iree_status_is_ok(status)) {
    loom_run_module_parse_options_t parse_options = {0};
    loom_run_module_parse_options_initialize(&parse_options);
    parse_options.filename = filename;
    parse_options.source = source;
    status = loom_run_module_parse(&session, &parse_options, &run_module);
  }
  if (iree_status_is_ok(status)) {
    const iree_flag_string_list_t libraries = FLAG_library_list();
    status = iree_test_loom_link_libraries(&session, &run_module,
                                           (iree_string_view_list_t){
                                               .count = libraries.count,
                                               .values = libraries.values,
                                           });
  }
  if (iree_status_is_ok(status)) {
    status = iree_test_loom_verify_run_module(&run_module);
  }

  if (iree_status_is_ok(status)) {
    iree_arena_initialize(loom_run_session_block_pool(&session), &plan_arena);
    iree_arena_initialize(loom_run_session_block_pool(&session),
                          &execution_arena);
    loom_testbench_plan_options_t plan_options = {0};
    loom_testbench_plan_options_initialize(&plan_options);
    plan_options.max_samples_per_case =
        (iree_host_size_t)FLAG_max_samples_per_case;
    loom_testbench_module_plan_t module_plan = {0};
    status = loom_testbench_plan_module(run_module.module, &plan_options,
                                        &plan_arena, &module_plan);
    const iree_string_view_t selected_case_name =
        iree_test_loom_normalize_case_name(iree_make_cstring_view(FLAG_case));
    loom_testbench_case_execution_options_t execution_options = {0};
    loom_testbench_case_execution_options_initialize(&execution_options);
    execution_options.materializer.host_allocator = allocator;
    execution_options.materializer.open_read_file =
        (loom_testbench_file_open_callback_t){
            .fn = iree_test_loom_open_file_for_read,
            .user_data = &file_provider,
        };
    if (iree_status_is_ok(status) &&
        iree_test_loom_selected_cases_have_device_event_expectation(
            &module_plan, selected_case_name)) {
      status = loom_testbench_device_event_capture_initialize(
          IREE_TEST_LOOM_DEVICE_EVENT_CAPACITY, allocator,
          &device_event_capture);
      if (iree_status_is_ok(status)) {
        device_event_capture_initialized = true;
        execution_options.device_event_capture = &device_event_capture;
        loom_run_hal_testbench_context_set_device_event_sink(
            &hal_context,
            loom_testbench_device_event_capture_sink(&device_event_capture));
      }
    }

    iree_host_size_t selected_case_count = 0;
    iree_host_size_t sample_count = 0;
    iree_host_size_t failed_sample_count = 0;
    iree_host_size_t skipped_case_count = 0;
    iree_host_size_t planning_issue_count = 0;
    for (iree_host_size_t case_index = 0;
         iree_status_is_ok(status) && case_index < module_plan.case_count;
         ++case_index) {
      const loom_testbench_case_plan_t* case_plan =
          &module_plan.cases[case_index];
      if (!iree_test_loom_case_matches_selection(case_plan,
                                                 selected_case_name)) {
        continue;
      }
      ++selected_case_count;
      if (case_plan->issue_count != 0) {
        status = iree_test_loom_append_case_planning_issues(
            &module_plan, case_plan, &planning_issues, &planning_issue_count);
      } else {
        status = iree_test_loom_run_case_samples(
            configuration, &session, &run_module, &module_plan, case_index,
            &execution_options, &config_set, &sanitizer_options, &hal_context,
            &execution_arena, &samples, &skipped_cases, &sample_count,
            &failed_sample_count, &skipped_case_count);
      }
    }
    if (iree_status_is_ok(status) && selected_case_count == 0) {
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND, "no check.case matched '%.*s'",
          (int)selected_case_name.size, selected_case_name.data);
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_array_end(&samples);
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_array_end(&skipped_cases);
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_array_end(&planning_issues);
    }
    if (iree_status_is_ok(status)) {
      status = iree_test_loom_write_report(
          selected_case_count, sample_count, failed_sample_count,
          skipped_case_count, planning_issue_count,
          iree_string_builder_view(&sample_output),
          iree_string_builder_view(&skipped_output),
          iree_string_builder_view(&planning_issue_output), &report_output);
    }
    if (iree_status_is_ok(status)) {
      status =
          loom_tooling_write_stdout(iree_string_builder_view(&report_output));
    }
    if (iree_status_is_ok(status) &&
        (failed_sample_count != 0 || planning_issue_count != 0)) {
      exit_code = 1;
    }
  }

  const bool had_error = !iree_status_is_ok(status);
  if (had_error) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = 1;
  }

  iree_string_builder_deinitialize(&report_output);
  iree_string_builder_deinitialize(&planning_issue_output);
  iree_string_builder_deinitialize(&skipped_output);
  iree_string_builder_deinitialize(&sample_output);
  iree_arena_deinitialize(&execution_arena);
  iree_arena_deinitialize(&plan_arena);
  loom_run_hal_testbench_context_deinitialize(&hal_context);
  loom_tooling_config_set_deinitialize(&config_set);
  if (device_event_capture_initialized) {
    loom_testbench_device_event_capture_deinitialize(&device_event_capture);
  }
  loom_run_module_deinitialize(&run_module);
  iree_io_file_contents_free(contents);
  loom_run_session_deinitialize(&session);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
