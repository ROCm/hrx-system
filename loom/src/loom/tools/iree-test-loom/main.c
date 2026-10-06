// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-test-loom: executes checked records from ordinary Loom modules.

#include "loom/tools/iree-test-loom/main.h"

#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/path.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/stdio_stream.h"
#include "loom/config/text_binding.h"
#include "loom/sanitizer/options.h"
#include "loom/tooling/cli/help.h"
#include "loom/tooling/cli/loomc_options.h"
#include "loom/tooling/cli/loomc_result.h"
#include "loom/tooling/config/config.h"
#include "loom/tooling/execution/hal/scenario_profile.h"
#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tooling/input/flags.h"
#include "loom/tooling/input/loomc.h"
#include "loom/tooling/io/file.h"
#include "loom/tooling/testbench/device_event.h"
#include "loom/tooling/testbench/executor.h"
#include "loom/tooling/testbench/issue_report.h"
#include "loom/tooling/testbench/reference.h"
#include "loom/tooling/testbench/requirements.h"
#include "loom/tools/iree-test-loom/xfail.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"
#include "loomc/interop.h"
#include "loomc/iree.h"
#include "loomc/link.h"

IREE_FLAG(string, case, "",
          "Optional check.case or check.scenario symbol to execute, such as "
          "'@smoke'. Empty executes all records in source order.");
IREE_FLAG(int32_t, sample, -1,
          "Optional concrete sample ordinal to execute for the selected case "
          "or cases. Negative executes all planned samples.");
IREE_FLAG_LIST(
    string, library,
    "Source or bytecode library linked into the authored input. Repeat as "
    "--library=path.loombc. Libraries are linked whole in argument order.");
IREE_FLAG_NAMED(int32_t, max_samples_per_case, "max-samples-per-case",
                LOOM_TESTBENCH_DEFAULT_MAX_SAMPLES_PER_CASE,
                "Maximum number of samples planned per check.case.");
IREE_FLAG(string, pipeline, "default",
          "Pass pipeline used for HAL kernel launches. 'default' runs the "
          "normal compiler pipeline. 'none' disables all compiler "
          "transformations and requires emission-ready input. Use '@symbol' "
          "or a comma-separated pass list for an explicit pipeline.");
IREE_FLAG(string, target, "",
          "Optional compiler target as `family:selector` for every HAL kernel "
          "launch. The selected device must be able to load the target. Empty "
          "selects a compatible target from the device and authored kernel.");
IREE_FLAG_LIST(
    string, config,
    "Compile-time config binding for kernel and function compilation. "
    "Repeat as --config=key=value. "
    "Bindings not referenced by the loaded module are ignored.");
IREE_FLAG_LIST_NAMED(
    string, config_file, "config-file",
    "JSON/JSONC config object file for kernel and function compilation. "
    "Repeat for multiple files. "
    "Nested object keys are flattened with '.' separators.");
IREE_FLAG(string, sanitizer, "none",
          "Sanitizer checks inserted by the target pipeline. Use 'none', "
          "'access', 'value', 'operation', 'race', 'asan', 'ubsan', 'tsan', "
          "'all', or a '|' separated list.");
IREE_FLAG_NAMED(string, sanitizer_reporting, "sanitizer-reporting", "default",
                "Sanitizer reporting mode used by the target pipeline. Use "
                "'default', 'trap', or 'report-only'.");
IREE_FLAG_LIST_NAMED(
    string, xfail, "xfail",
    "Required record failure as '@record=DOMAIN/NNN[,DOMAIN/NNN...]'. "
    "Repeat once per record. XPASS and diagnostic drift fail the run.");
IREE_FLAG_LIST_NAMED(
    string, allow_failure, "allow-failure",
    "Permitted record failure as '@record=DOMAIN/NNN[,DOMAIN/NNN...]'. "
    "A pass is valid, while any other diagnostic fails the run.");

enum {
  // Total bytes retained per runner scratch arena block.
  IREE_TEST_LOOM_BLOCK_POOL_BLOCK_SIZE = 128 * 1024,
  // Target-linked requirement providers.
  IREE_TEST_LOOM_MAX_REQUIREMENT_PROVIDERS = 8,
  // Trials submitted through one prepared product callback.
  IREE_TEST_LOOM_SCENARIO_BATCH_CAPACITY = 64,
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

static iree_status_t iree_test_loom_require_loomc_result(
    const loomc_result_t* result, iree_string_view_t operation) {
  bool succeeded = false;
  IREE_RETURN_IF_ERROR(
      loom_tooling_cli_print_loomc_result(stderr, result, &succeeded));
  return succeeded
             ? iree_ok_status()
             : iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%.*s failed",
                                (int)operation.size, operation.data);
}

static iree_status_t iree_test_loom_admit_module(
    const iree_test_loom_configuration_t* configuration,
    const loom_input_options_t* input_options, iree_string_view_t input_path,
    loomc_context_t* context, loomc_workspace_t* workspace,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    loomc_module_t** out_module) {
  *out_module = NULL;
  iree_io_file_contents_t* contents = NULL;
  loomc_result_t* result = NULL;
  iree_status_t status =
      loom_tooling_read_input_file(input_path, host_allocator, &contents);
  if (iree_status_is_ok(status)) {
    const loom_tooling_loomc_input_options_t admission_options = {
        .providers = configuration->input_providers,
        .input = *input_options,
        .path = input_path,
        .source = loom_tooling_file_contents_string_view(contents),
        .import = configuration->import,
        .import_user_data = configuration->import_user_data,
    };
    status = loom_tooling_input_admit_loomc_module(
        &admission_options, context, workspace, block_pool, out_module, &result,
        host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_test_loom_require_loomc_result(result,
                                                 IREE_SV("source admission"));
  }
  loomc_result_release(result);
  iree_io_file_contents_free(contents);
  if (!iree_status_is_ok(status)) {
    loomc_module_release(*out_module);
    *out_module = NULL;
  }
  return status;
}

static iree_status_t iree_test_loom_link_libraries(
    const iree_test_loom_configuration_t* configuration,
    const loom_input_options_t* input_options, iree_string_view_t input_path,
    iree_string_view_list_t library_paths, loomc_context_t* context,
    loomc_workspace_t* workspace, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator, loomc_module_t** inout_module) {
  if (library_paths.count == 0) {
    return iree_ok_status();
  }
  if (library_paths.values == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "library path count is non-zero but values is NULL");
  }

  iree_host_size_t provider_count = 0;
  if (!iree_host_size_checked_add(library_paths.count, 1, &provider_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "library provider count overflow");
  }
  loomc_link_module_provider_t* providers = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, provider_count, sizeof(*providers), (void**)&providers));
  memset(providers, 0, provider_count * sizeof(*providers));
  providers[0] = (loomc_link_module_provider_t){
      .module = *inout_module,
      .provider_name = loomc_string_view_from_iree(input_path),
      .role = LOOMC_LINK_PROVIDER_ROLE_INPUT,
  };

  iree_status_t status = iree_ok_status();
  iree_host_size_t admitted_library_count = 0;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < library_paths.count; ++i) {
    const iree_string_view_t library_path = library_paths.values[i];
    if (loom_tooling_file_path_is_stdio(library_path)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "--library requires a filesystem path");
      break;
    }
    loomc_module_t* library_module = NULL;
    status = iree_test_loom_admit_module(
        configuration, input_options, library_path, context, workspace,
        block_pool, host_allocator, &library_module);
    if (iree_status_is_ok(status)) {
      providers[i + 1] = (loomc_link_module_provider_t){
          .module = library_module,
          .provider_name = loomc_string_view_from_iree(library_path),
          .role = LOOMC_LINK_PROVIDER_ROLE_INPUT,
      };
      ++admitted_library_count;
    }
  }

  loomc_linker_t* linker = NULL;
  if (iree_status_is_ok(status)) {
    const loomc_linker_options_t linker_options = {
        .type = LOOMC_STRUCTURE_TYPE_LINKER_OPTIONS,
        .structure_size = sizeof(linker_options),
        .module_name = loomc_make_cstring_view("linked"),
    };
    status = iree_status_from_loomc(loomc_linker_create(
        context, &linker_options, loomc_allocator_from_iree(host_allocator),
        &linker));
  }
  loomc_module_t* linked_module = NULL;
  loomc_result_t* result = NULL;
  if (iree_status_is_ok(status)) {
    const loomc_link_options_t link_options = {
        .type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS,
        .structure_size = sizeof(link_options),
        .module_name = loomc_make_cstring_view("linked"),
        .mode = LOOMC_LINK_MODE_MERGE,
        .module_providers = providers,
        .module_provider_count = provider_count,
    };
    status = iree_status_from_loomc(loomc_link_module(
        linker, workspace, &link_options, &linked_module, &result));
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_test_loom_require_loomc_result(result, IREE_SV("module linking"));
  }
  if (iree_status_is_ok(status)) {
    loomc_module_release(*inout_module);
    *inout_module = linked_module;
    linked_module = NULL;
  }

  loomc_result_release(result);
  loomc_module_release(linked_module);
  loomc_linker_release(linker);
  for (iree_host_size_t i = 0; i < admitted_library_count; ++i) {
    loomc_module_release((loomc_module_t*)providers[i + 1].module);
  }
  iree_allocator_free(host_allocator, providers);
  return status;
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

static bool iree_test_loom_scenario_matches_selection(
    const loom_testbench_scenario_plan_t* scenario_plan,
    iree_string_view_t selected_case_name) {
  return iree_string_view_is_empty(selected_case_name) ||
         iree_string_view_equal(scenario_plan->name, selected_case_name);
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

static iree_status_t iree_test_loom_append_scenario_planning_issues(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_scenario_plan_t* scenario_plan,
    loom_json_array_writer_t* planning_issues,
    iree_host_size_t* inout_planning_issue_count) {
  for (iree_host_size_t i = 0; i < scenario_plan->issue_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_test_loom_append_planning_issue(
        module_plan, &scenario_plan->issues[i], planning_issues));
    ++*inout_planning_issue_count;
  }
  return iree_ok_status();
}

static iree_status_t iree_test_loom_configure_hal_actual_sequence(
    const loom_testbench_compilation_t* compilation,
    const loom_module_t* native_module,
    const loom_testbench_case_plan_t* case_plan,
    const loomc_pass_program_t* pass_program,
    loomc_target_profile_t* requested_target_profile,
    const loomc_sanitizer_options_t* sanitizer,
    loom_testbench_compile_result_callback_t result_callback,
    loom_run_hal_testbench_context_t* hal_context,
    loom_testbench_case_execution_options_t* execution_options,
    loom_run_hal_testbench_actual_sequence_t* out_sequence) {
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_context_ensure_runtime(hal_context));

  execution_options->materializer.device_allocator =
      iree_hal_device_allocator(hal_context->runtime.device);
  execution_options->materializer.buffer_params =
      loom_run_hal_testbench_host_visible_buffer_params();

  const loom_run_hal_testbench_actual_sequence_options_t sequence_options = {
      .context = hal_context,
      .compilation = compilation,
      .native_module = native_module,
      .pass_program = pass_program,
      .requested_target_profile = requested_target_profile,
      .sanitizer = sanitizer,
      .case_plan = case_plan,
      .result_callback = result_callback,
  };
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_actual_sequence_initialize(
      &sequence_options, out_sequence));
  execution_options->invocation.kernel_launch =
      loom_run_hal_testbench_actual_sequence_provider(out_sequence);
  return iree_ok_status();
}

static iree_status_t iree_test_loom_run_case_samples(
    const iree_test_loom_configuration_t* configuration,
    const loom_testbench_compilation_t* compilation,
    const loom_module_t* native_module,
    const loom_testbench_module_plan_t* module_plan,
    iree_host_size_t case_index,
    const loom_testbench_case_execution_options_t* base_execution_options,
    const loomc_pass_program_t* pass_program,
    loomc_target_profile_t* requested_target_profile,
    const loomc_sanitizer_options_t* sanitizer,
    loom_testbench_compile_result_callback_t result_callback,
    iree_test_loom_diagnostic_capture_t* diagnostic_capture,
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
        compilation, native_module, case_plan, pass_program,
        requested_target_profile, sanitizer, result_callback, hal_context,
        &execution_options, &hal_actual_sequence);
    hal_actual_sequence_initialized = iree_status_is_ok(status);
    if (iree_status_is_ok(status)) {
      matmul_oracle_options.device_allocator =
          iree_hal_device_allocator(hal_context->runtime.device);
      matmul_oracle_options.result_buffer_params =
          loom_run_hal_testbench_host_visible_buffer_params();
      matmul_oracle_options.host_allocator =
          execution_options.materializer.host_allocator;
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
      iree_test_loom_capture_expectation_report(
          diagnostic_capture, sample_result.expectation_report);
    }
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

static iree_status_t iree_test_loom_run_scenario(
    const loom_testbench_module_plan_t* module_plan,
    iree_host_size_t scenario_index,
    const loom_testbench_scenario_execution_options_t* execution_options,
    const loom_testbench_value_materializer_options_t* materializer_options,
    loom_testbench_entropy_t entropy_root,
    iree_test_loom_diagnostic_capture_t* diagnostic_capture,
    loom_json_array_writer_t* trial_results,
    iree_host_size_t* inout_trial_count,
    iree_host_size_t* inout_failed_trial_count) {
  const loom_testbench_scenario_plan_t* scenario =
      &module_plan->scenarios[scenario_index];
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t configuration_ordinal = 0;
       iree_status_is_ok(status) &&
       configuration_ordinal < scenario->configuration_count;
       ++configuration_ordinal) {
    loom_testbench_scenario_configuration_values_t configuration = {0};
    status = loom_testbench_scenario_configuration_values_initialize(
        module_plan->module, scenario, execution_options->host_allocator,
        &configuration);
    if (iree_status_is_ok(status)) {
      status = loom_testbench_scenario_configuration_values_materialize(
          materializer_options, entropy_root, configuration_ordinal,
          &configuration);
    }

    loom_testbench_prepared_scenario_configuration_t prepared = {0};
    if (iree_status_is_ok(status)) {
      status = loom_testbench_prepare_scenario_configuration(
          execution_options, &configuration,
          LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS, &prepared);
    }
    for (iree_host_size_t trial_index = 0;
         iree_status_is_ok(status) && trial_index < scenario->trial_count;
         ++trial_index) {
      const loom_testbench_trial_plan_t* trial = &scenario->trials[trial_index];
      const iree_host_size_t batch_capacity =
          iree_min(trial->trial_count,
                   (iree_host_size_t)IREE_TEST_LOOM_SCENARIO_BATCH_CAPACITY);
      if (batch_capacity == 0) {
        continue;
      }
      loom_testbench_scenario_trial_executor_t executor = {0};
      status = loom_testbench_scenario_trial_executor_initialize(
          &prepared, trial_index, materializer_options, batch_capacity,
          &executor);
      for (iree_host_size_t first_trial_ordinal = 0;
           iree_status_is_ok(status) &&
           first_trial_ordinal < trial->trial_count;
           first_trial_ordinal += batch_capacity) {
        const iree_host_size_t batch_count =
            iree_min(batch_capacity, trial->trial_count - first_trial_ordinal);
        loom_testbench_scenario_trial_result_list_t results = {0};
        status = loom_testbench_run_scenario_trial_batch(
            &executor, first_trial_ordinal, batch_count, &results);
        for (iree_host_size_t result_index = 0;
             iree_status_is_ok(status) && result_index < results.count;
             ++result_index) {
          iree_test_loom_capture_expectation_report(
              diagnostic_capture,
              results.values[result_index].expectation_report);
          status = loom_json_array_begin_element(trial_results);
          if (iree_status_is_ok(status)) {
            status = loom_testbench_scenario_trial_result_write_json(
                &results.values[result_index], trial_results->stream);
          }
          if (iree_status_is_ok(status)) {
            ++*inout_trial_count;
            if (!results.values[result_index].passed) {
              ++*inout_failed_trial_count;
            }
          }
        }
      }
      loom_testbench_scenario_trial_executor_deinitialize(&executor);
    }
    loom_testbench_prepared_scenario_configuration_deinitialize(&prepared);
    loom_testbench_scenario_configuration_values_deinitialize(&configuration);
  }
  return status;
}

static iree_status_t iree_test_loom_append_config_flags(
    loom_config_text_binding_set_t* config_set) {
  const iree_flag_string_list_t assignments = FLAG_config_list();
  for (iree_host_size_t i = 0; i < assignments.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_text_binding_set_append_assignment(
        config_set, assignments.values[i]));
  }
  return iree_ok_status();
}

static iree_status_t iree_test_loom_append_config_files(
    loom_config_text_binding_set_t* config_set, iree_allocator_t allocator) {
  const iree_flag_string_list_t paths = FLAG_config_file_list();
  for (iree_host_size_t i = 0; i < paths.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_text_binding_set_append_json_file(
        config_set, paths.values[i], allocator));
  }
  return iree_ok_status();
}

static iree_status_t iree_test_loom_write_report(
    iree_host_size_t case_count, iree_host_size_t scenario_count,
    iree_host_size_t sample_count, iree_host_size_t failed_sample_count,
    iree_host_size_t trial_count, iree_host_size_t failed_trial_count,
    iree_host_size_t skipped_case_count, iree_host_size_t planning_issue_count,
    const iree_test_loom_xfail_list_t* xfails, iree_string_view_t samples,
    iree_string_view_t trials, iree_string_view_t skipped_cases,
    iree_string_view_t planning_issues, iree_string_builder_t* output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("format"), IREE_SV("loom.test.v0")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("case_count"), case_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("scenario_count"), scenario_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("sample_count"), sample_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("failed_sample_count"), failed_sample_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("trial_count"), trial_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("failed_trial_count"), failed_trial_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("skipped_case_count"), skipped_case_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("planning_issue_count"), planning_issue_count));
  const iree_test_loom_xfail_counts_t xfail_counts =
      iree_test_loom_count_xfails(xfails);
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("xfail_count"), xfail_counts.xfail_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("xpass_count"), xfail_counts.xpass_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("xfail_mismatch_count"), xfail_counts.mismatch_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("samples")));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(&stream, samples));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("trials")));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(&stream, trials));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("skipped_cases")));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(&stream, skipped_cases));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("planning_issues")));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(&stream, planning_issues));
  IREE_RETURN_IF_ERROR(iree_test_loom_write_xfails_json(xfails, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  return loom_output_stream_write_char(&stream, '\n');
}

static void iree_test_loom_print_agents_markdown(FILE* stream) {
  fprintf(
      stream,
      "## iree-test-loom\n"
      "\n"
      "`iree-test-loom` executes `check.case` and `check.scenario` records "
      "from\n"
      "ordinary Loom modules and writes a structured `loom.test.v0` JSON "
      "report.\n"
      "Use it for correctness before promoting the same records to\n"
      "`check.benchmark` rows.\n"
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
      "iree-test-loom module.loom --device=amdgpu "
      "--target=amdgpu:gfx11-generic\n"
      "iree-test-loom module.loom --config=model.hidden_size=4096\n"
      "iree-test-loom module.loom --config-file=model_config.jsonc\n"
      "iree-test-loom module.loom --sanitizer=tsan\n"
      "iree-test-loom module.loom --sanitizer=asan "
      "--sanitizer-reporting=report-only\n"
      "iree-test-loom module.loom --xfail=@unsupported=TARGET/003\n"
      "iree-test-loom module.loom "
      "--allow-failure=@device_dependent=TARGET/003\n"
      "```\n"
      "\n"
      "`--case=@name` selects one checked case or scenario; empty selection "
      "runs\n"
      "all records. `--sample=N` selects one planned sample for `check.case`\n"
      "records. Scenario trial domains execute in bounded batches.\n"
      "`--max-samples-per-case=N` bounds planning for generator-heavy cases.\n"
      "`--xfail=@record=DOMAIN/NNN` requires that record to fail with the "
      "named\n"
      "compile or expectation diagnostic. `--allow-failure` also accepts a "
      "pass.\n"
      "Comma-separated diagnostics express target-dependent failure modes; "
      "any\n"
      "other diagnostic fails the run.\n"
      "\n"
      "### Kernel launches\n"
      "\n"
      "A case can mix reference/oracle checks with HAL kernel launches. "
      "Kernel\n"
      "launches use the selected HAL artifact provider, target provider, "
      "and HAL device linked into this binary. "
      "An explicit `--device=DRIVER` is validated even when the selected cases "
      "contain no kernel launches; unavailable selections identify the "
      "missing driver. "
      "`--target=family:selector` forces the compiler target for every HAL "
      "kernel without changing the device selected by `--device`. "
      "`--pipeline=default|none|@symbol|pass,list` controls the HAL kernel "
      "compile pipeline. `none` disables all compiler transformations and "
      "requires emission-ready input. `--config=key=value` and "
      "`--config-file=path` "
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
      "iree-test-loom module.loom | jq '.trials[] | {scenario, "
      "configuration_ordinal, trial_index, trial_ordinal, passed}'\n"
      "iree-test-loom module.loom | jq '.skipped_cases[]? | {case, provider, "
      "op, code, provider_code}'\n"
      "iree-test-loom module.loom | jq '.planning_issues[]? | {case, kind, "
      "op, source_location, fix_hint}'\n"
      "```\n"
      "\n"
      "The report carries case/sample and scenario/trial counts, including\n"
      "`failed_sample_count` and `failed_trial_count`, plus\n"
      "`skipped_case_count`, `planning_issue_count`, `samples`, "
      "`trials`, `skipped_cases`,\n"
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
      "A nonzero failed sample, failed trial, or planning issue count makes "
      "the\n"
      "process fail after the JSON report is written.\n");
}

int iree_test_loom_main(int argc, char** argv,
                        const iree_test_loom_configuration_t* configuration) {
  iree_flags_set_usage(
      configuration->tool_name,
      "Executes check.case and check.scenario records from a normal Loom "
      "module.\n"
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
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(IREE_TEST_LOOM_BLOCK_POOL_BLOCK_SIZE,
                                   allocator, &block_pool);
  loom_config_text_binding_set_t config_set;
  loom_config_text_binding_set_initialize(allocator, &config_set);
  loomc_config_binding_t* config_bindings = NULL;
  loomc_config_options_t config_options = {0};
  loomc_context_t* compiler_context = NULL;
  loomc_workspace_t* compiler_workspace = NULL;
  loomc_compiler_t* compiler = NULL;
  loomc_module_t* module = NULL;
  loomc_pass_program_t* pass_program = NULL;
  loomc_target_profile_t* requested_target_profile = NULL;
  loomc_module_interop_view_t module_view = {0};
  const loom_module_t* native_module = NULL;
  const loom_source_table_resolver_t* source_table = NULL;
  loom_sanitizer_options_t sanitizer_options = {0};
  loomc_sanitizer_options_t loomc_sanitizer_options = {0};
  loom_run_hal_testbench_context_t hal_context = {0};
  loom_run_hal_testbench_scenario_profile_t hal_scenario_profile = {0};
  iree_test_loom_xfail_list_t xfails = {0};
  iree_test_loom_diagnostic_capture_t diagnostic_capture = {0};
  iree_hal_allocator_t* host_device_allocator = NULL;
  loom_testbench_device_event_capture_t device_event_capture = {0};
  bool device_event_capture_initialized = false;
  iree_arena_allocator_t plan_arena;
  memset(&plan_arena, 0, sizeof(plan_arena));
  iree_arena_allocator_t execution_arena;
  memset(&execution_arena, 0, sizeof(execution_arena));
  iree_string_builder_t sample_output;
  iree_string_builder_initialize(allocator, &sample_output);
  iree_string_builder_t trial_output;
  iree_string_builder_initialize(allocator, &trial_output);
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
  const iree_flag_string_list_t xfail_flags = FLAG_xfail_list();
  const iree_flag_string_list_t allow_failure_flags = FLAG_allow_failure_list();
  iree_status_t status = iree_test_loom_xfail_list_initialize(
      (iree_string_view_list_t){
          .count = xfail_flags.count,
          .values = xfail_flags.values,
      },
      (iree_string_view_list_t){
          .count = allow_failure_flags.count,
          .values = allow_failure_flags.values,
      },
      allocator, &xfails);
  if (iree_status_is_ok(status)) {
    status = loom_json_array_begin(&sample_stream, &samples);
  }
  loom_output_stream_t trial_stream;
  loom_output_stream_for_builder(&trial_output, &trial_stream);
  loom_json_array_writer_t trials;
  if (iree_status_is_ok(status)) {
    status = loom_json_array_begin(&trial_stream, &trials);
  }
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
    status = loom_tooling_cli_make_loomc_config_options(
        &config_set, allocator, &config_bindings, &config_options);
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
  const bool sanitizer_enabled =
      loom_sanitizer_options_is_enabled(&sanitizer_options);
  const iree_string_view_t pipeline =
      iree_string_view_trim(iree_make_cstring_view(FLAG_pipeline));
  if (iree_status_is_ok(status) && sanitizer_enabled &&
      !loom_tooling_cli_pipeline_uses_default(pipeline)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--sanitizer and --sanitizer-reporting require --pipeline=default");
  }
  if (iree_status_is_ok(status) && sanitizer_enabled) {
    loom_tooling_cli_make_loomc_sanitizer_options(&sanitizer_options,
                                                  &loomc_sanitizer_options);
  }

  if (iree_status_is_ok(status)) {
    const loomc_context_target_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        .structure_size = sizeof(target_options),
        .target_environment = configuration->target_environment,
    };
    const loomc_context_options_t context_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        .structure_size = sizeof(context_options),
        .next =
            configuration->target_environment != NULL ? &target_options : NULL,
    };
    status = iree_status_from_loomc(loomc_context_create(
        &context_options, loomc_allocator_from_iree(allocator),
        &compiler_context));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_workspace_create(
        /*options=*/NULL, loomc_allocator_from_iree(allocator),
        &compiler_workspace));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(
        loomc_compiler_create(compiler_context, /*options=*/NULL,
                              loomc_allocator_from_iree(allocator), &compiler));
  }
  if (iree_status_is_ok(status)) {
    loom_run_hal_testbench_context_initialize(
        configuration->target_environment, configuration->hal_target_routes,
        configuration->hal_target_route_count, allocator, &hal_context);
    status =
        loom_run_hal_testbench_context_validate_explicit_device(&hal_context);
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
  const loom_input_options_t input_options = loom_input_options_from_flags();
  if (iree_status_is_ok(status)) {
    status = iree_test_loom_admit_module(
        configuration, &input_options, input_path, compiler_context,
        compiler_workspace, &block_pool, allocator, &module);
  }
  if (iree_status_is_ok(status)) {
    const iree_flag_string_list_t libraries = FLAG_library_list();
    status = iree_test_loom_link_libraries(
        configuration, &input_options, filename,
        (iree_string_view_list_t){
            .count = libraries.count,
            .values = libraries.values,
        },
        compiler_context, compiler_workspace, &block_pool, allocator, &module);
  }
  if (iree_status_is_ok(status)) {
    loomc_result_t* result = NULL;
    status = iree_status_from_loomc(loomc_module_get_interop_view(
        module, loomc_allocator_from_iree(allocator), &module_view, &result));
    if (iree_status_is_ok(status)) {
      status = iree_test_loom_require_loomc_result(
          result, IREE_SV("module verification"));
    }
    loomc_result_release(result);
  }
  if (iree_status_is_ok(status)) {
    // Execution providers only inspect or independently clone this
    // exact-version native view. The public module retains all borrowed state.
    native_module = module_view.module;
    source_table = module_view.source_table;
  }
  const iree_string_view_t requested_target =
      iree_string_view_trim(iree_make_cstring_view(FLAG_target));
  if (iree_status_is_ok(status) &&
      !iree_string_view_is_empty(requested_target)) {
    status = iree_status_from_loomc(loomc_target_profile_select(
        configuration->target_environment,
        loomc_string_view_from_iree(requested_target),
        loomc_allocator_from_iree(allocator), &requested_target_profile));
  }
  if (iree_status_is_ok(status)) {
    loomc_result_t* result = NULL;
    status = loom_tooling_cli_prepare_loomc_pass_program(
        compiler_context, module, pipeline, IREE_SV("iree-test-loom pipeline"),
        &pass_program, &result, allocator);
    if (iree_status_is_ok(status) && result != NULL) {
      status = iree_test_loom_require_loomc_result(
          result, IREE_SV("pass program preparation"));
    }
    loomc_result_release(result);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_context_add_module_runtime_requirements(
        &hal_context, module,
        sanitizer_enabled ? &loomc_sanitizer_options : NULL);
  }

  if (iree_status_is_ok(status)) {
    iree_arena_initialize(&block_pool, &plan_arena);
    iree_arena_initialize(&block_pool, &execution_arena);
    loom_testbench_plan_options_t plan_options = {0};
    loom_testbench_plan_options_initialize(&plan_options);
    plan_options.max_samples_per_case =
        (iree_host_size_t)FLAG_max_samples_per_case;
    loom_testbench_module_plan_t module_plan = {0};
    status = loom_testbench_plan_module(native_module, &plan_options,
                                        &plan_arena, &module_plan);
    const iree_string_view_t selected_case_name =
        iree_test_loom_normalize_case_name(iree_make_cstring_view(FLAG_case));
    if (iree_status_is_ok(status)) {
      status = iree_test_loom_validate_xfails(&xfails, &module_plan,
                                              selected_case_name);
    }
    loom_testbench_case_execution_options_t execution_options = {0};
    loom_testbench_case_execution_options_initialize(&execution_options);
    loom_testbench_scenario_execution_options_t scenario_execution_options = {
        0};
    loom_testbench_scenario_execution_options_initialize(
        &scenario_execution_options);
    const loom_testbench_case_plan_t** selected_cases = NULL;
    iree_host_size_t selected_case_count = 0;
    const loom_testbench_compilation_t compilation = {
        .compiler = compiler,
        .workspace = compiler_workspace,
        .module = module,
        .config = &config_options,
    };
    const loom_testbench_compile_result_callback_t compile_result_callback = {
        .fn = iree_test_loom_diagnostic_capture_loomc_result,
        .user_data = &diagnostic_capture,
    };
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate_array(&plan_arena, module_plan.case_count,
                                         sizeof(*selected_cases),
                                         (void**)&selected_cases);
    }
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < module_plan.case_count; ++i) {
        const loom_testbench_case_plan_t* case_plan = &module_plan.cases[i];
        if (iree_test_loom_case_matches_selection(case_plan,
                                                  selected_case_name)) {
          selected_cases[selected_case_count++] = case_plan;
        }
      }
      iree_host_size_t function_call_capacity = 0;
      for (iree_host_size_t i = 0; i < selected_case_count; ++i) {
        if (selected_cases[i]->issue_count == 0) {
          function_call_capacity += selected_cases[i]->invocation_count;
        }
      }
      for (iree_host_size_t scenario_index = 0;
           scenario_index < module_plan.scenario_count; ++scenario_index) {
        const loom_testbench_scenario_plan_t* scenario =
            &module_plan.scenarios[scenario_index];
        if (scenario->issue_count != 0 ||
            !iree_test_loom_scenario_matches_selection(scenario,
                                                       selected_case_name)) {
          continue;
        }
        for (iree_host_size_t trial_index = 0;
             trial_index < scenario->trial_count; ++trial_index) {
          const loom_testbench_trial_plan_t* trial =
              &scenario->trials[trial_index];
          function_call_capacity += trial->recipe_step_count;
        }
      }
      const loom_testbench_invocation_plan_t** function_calls = NULL;
      if (function_call_capacity != 0) {
        status = iree_arena_allocate_array(&plan_arena, function_call_capacity,
                                           sizeof(*function_calls),
                                           (void**)&function_calls);
      }
      iree_host_size_t function_call_count = 0;
      for (iree_host_size_t i = 0;
           iree_status_is_ok(status) && i < selected_case_count; ++i) {
        const loom_testbench_case_plan_t* case_plan = selected_cases[i];
        if (case_plan->issue_count != 0) {
          continue;
        }
        for (iree_host_size_t j = 0; j < case_plan->invocation_count; ++j) {
          const loom_testbench_invocation_plan_t* invocation =
              &case_plan->invocations[j];
          if (invocation->kind == LOOM_TESTBENCH_INVOCATION_FUNCTION_CALL) {
            function_calls[function_call_count++] = invocation;
          }
        }
      }
      for (iree_host_size_t scenario_index = 0;
           iree_status_is_ok(status) &&
           scenario_index < module_plan.scenario_count;
           ++scenario_index) {
        const loom_testbench_scenario_plan_t* scenario =
            &module_plan.scenarios[scenario_index];
        if (scenario->issue_count != 0 ||
            !iree_test_loom_scenario_matches_selection(scenario,
                                                       selected_case_name)) {
          continue;
        }
        for (iree_host_size_t trial_index = 0;
             trial_index < scenario->trial_count; ++trial_index) {
          const loom_testbench_trial_plan_t* trial =
              &scenario->trials[trial_index];
          for (iree_host_size_t step_index = 0;
               step_index < trial->recipe_step_count; ++step_index) {
            const loom_testbench_trial_recipe_step_t* step =
                &trial->recipe_steps[step_index];
            if (step->kind == LOOM_TESTBENCH_TRIAL_RECIPE_STEP_GENERATOR) {
              function_calls[function_call_count++] = &step->generator;
            }
          }
        }
      }
      if (iree_status_is_ok(status) && function_call_count != 0 &&
          configuration->function_call_provider.fn) {
        execution_options.invocation.function_call =
            configuration->function_call_provider.fn(
                configuration->function_call_provider.user_data, &compilation,
                (loom_testbench_invocation_plan_list_t){
                    .values = function_calls,
                    .count = function_call_count,
                },
                compile_result_callback);
      }
      scenario_execution_options.function_call =
          execution_options.invocation.function_call;
      if (configuration->scenario_target_profile.fn != NULL) {
        scenario_execution_options.target =
            configuration->scenario_target_profile.fn(
                configuration->scenario_target_profile.user_data, &compilation,
                source_table,
                iree_test_loom_diagnostic_capture_sink(&diagnostic_capture),
                compile_result_callback);
      }
      if (configuration->scenario_oracle_profile.fn != NULL) {
        scenario_execution_options.oracle =
            configuration->scenario_oracle_profile.fn(
                configuration->scenario_oracle_profile.user_data, &compilation,
                source_table,
                iree_test_loom_diagnostic_capture_sink(&diagnostic_capture),
                compile_result_callback);
      }
    }
    execution_options.materializer.host_allocator = allocator;
    execution_options.materializer.open_read_file.fn =
        iree_test_loom_open_file_for_read;
    execution_options.materializer.open_read_file.user_data = &file_provider;
    if (iree_status_is_ok(status)) {
      status = loom_testbench_device_event_capture_initialize(
          LOOM_TESTBENCH_DEVICE_EVENT_DEFAULT_CAPACITY, allocator,
          &device_event_capture);
      if (iree_status_is_ok(status)) {
        device_event_capture_initialized = true;
        execution_options.device_event_capture = &device_event_capture;
        scenario_execution_options.device_event_capture = &device_event_capture;
        loom_run_hal_testbench_context_set_device_event_sink(
            &hal_context,
            loom_testbench_device_event_capture_sink(&device_event_capture));
      }
    }

    iree_host_size_t selected_scenario_count = 0;
    for (iree_host_size_t i = 0; i < module_plan.scenario_count; ++i) {
      if (iree_test_loom_scenario_matches_selection(&module_plan.scenarios[i],
                                                    selected_case_name)) {
        ++selected_scenario_count;
      }
    }
    if (iree_status_is_ok(status) && selected_scenario_count != 0 &&
        !iree_string_view_is_empty(hal_context.driver_name)) {
      status = loom_run_hal_testbench_context_ensure_runtime(&hal_context);
      if (iree_status_is_ok(status)) {
        const iree_string_view_t target = iree_make_cstring_view(FLAG_target);
        const loom_run_hal_testbench_actual_provider_options_t
            provider_options = {
                .context = &hal_context,
                .compilation = &compilation,
                .module = module,
                .native_module = native_module,
                .pass_program = pass_program,
                .requested_target_profile = requested_target_profile,
                .sanitizer =
                    sanitizer_enabled ? &loomc_sanitizer_options : NULL,
                .result_callback = compile_result_callback,
            };
        loom_run_hal_testbench_scenario_profile_initialize(
            iree_string_view_is_empty(target) ? hal_context.driver_name
                                              : target,
            &provider_options, source_table,
            iree_test_loom_diagnostic_capture_sink(&diagnostic_capture),
            &hal_scenario_profile);
        scenario_execution_options.target =
            loom_run_hal_testbench_scenario_execution_profile(
                &hal_scenario_profile);
      }
    }
    if (iree_status_is_ok(status) && selected_scenario_count != 0 &&
        execution_options.materializer.device_allocator == NULL) {
      // Scenario values are shared with host oracles. Each execution profile
      // stages those values into its own device storage as needed.
      status =
          iree_hal_allocator_create_heap(IREE_SV("iree-test-loom"), allocator,
                                         allocator, &host_device_allocator);
      if (iree_status_is_ok(status)) {
        execution_options.materializer.device_allocator = host_device_allocator;
      }
    }
    iree_host_size_t sample_count = 0;
    iree_host_size_t failed_sample_count = 0;
    iree_host_size_t trial_count = 0;
    iree_host_size_t failed_trial_count = 0;
    iree_host_size_t skipped_case_count = 0;
    iree_host_size_t planning_issue_count = 0;
    if (iree_status_is_ok(status) && selected_scenario_count != 0 &&
        FLAG_sample >= 0) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--sample selects check.case samples and cannot select scenario "
          "trials");
    }
    for (iree_host_size_t selection_index = 0;
         iree_status_is_ok(status) && selection_index < selected_case_count;
         ++selection_index) {
      const loom_testbench_case_plan_t* case_plan =
          selected_cases[selection_index];
      const iree_host_size_t case_index = case_plan - module_plan.cases;
      iree_test_loom_xfail_t* xfail =
          iree_test_loom_xfail_list_find(&xfails, case_plan->name);
      iree_test_loom_diagnostic_capture_begin(&diagnostic_capture, xfail);
      const iree_host_size_t failed_sample_begin = failed_sample_count;
      const iree_host_size_t skipped_case_begin = skipped_case_count;
      const bool had_planning_issues = case_plan->issue_count != 0;
      if (had_planning_issues) {
        status = iree_test_loom_append_case_planning_issues(
            &module_plan, case_plan, &planning_issues, &planning_issue_count);
      } else {
        status = iree_test_loom_run_case_samples(
            configuration, &compilation, native_module, &module_plan,
            case_index, &execution_options, pass_program,
            requested_target_profile,
            sanitizer_enabled ? &loomc_sanitizer_options : NULL,
            compile_result_callback, &diagnostic_capture, &hal_context,
            &execution_arena, &samples, &skipped_cases, &sample_count,
            &failed_sample_count, &skipped_case_count);
      }
      if (!iree_test_loom_xfail_try_accept_compile_failure(
              xfail, &diagnostic_capture, &status) &&
          iree_status_is_ok(status) && xfail != NULL) {
        const iree_host_size_t failed_sample_delta =
            failed_sample_count - failed_sample_begin;
        const bool skipped = skipped_case_count != skipped_case_begin;
        if (had_planning_issues || skipped) {
          iree_test_loom_finish_xfail(
              xfail, &diagnostic_capture,
              IREE_TEST_LOOM_XFAIL_OUTCOME_DIAGNOSTIC_MISMATCH, 0, 0);
        } else if (failed_sample_delta == 0) {
          iree_test_loom_finish_xfail(
              xfail, &diagnostic_capture,
              xfail->policy == IREE_TEST_LOOM_XFAIL_POLICY_ALLOW_PASS
                  ? IREE_TEST_LOOM_XFAIL_OUTCOME_ALLOWED_PASS
                  : IREE_TEST_LOOM_XFAIL_OUTCOME_UNEXPECTED_PASS,
              0, 0);
        } else if (diagnostic_capture.matched_expected_diagnostic) {
          iree_test_loom_finish_xfail(
              xfail, &diagnostic_capture,
              IREE_TEST_LOOM_XFAIL_OUTCOME_EXPECTED_FAILURE,
              failed_sample_delta, 0);
        } else {
          iree_test_loom_finish_xfail(
              xfail, &diagnostic_capture,
              IREE_TEST_LOOM_XFAIL_OUTCOME_DIAGNOSTIC_MISMATCH, 0, 0);
        }
      }
    }
    const loom_testbench_entropy_t entropy_root =
        loom_testbench_entropy_root(0);
    for (iree_host_size_t scenario_index = 0;
         iree_status_is_ok(status) &&
         scenario_index < module_plan.scenario_count;
         ++scenario_index) {
      const loom_testbench_scenario_plan_t* scenario =
          &module_plan.scenarios[scenario_index];
      if (!iree_test_loom_scenario_matches_selection(scenario,
                                                     selected_case_name)) {
        continue;
      }
      iree_test_loom_xfail_t* xfail =
          iree_test_loom_xfail_list_find(&xfails, scenario->name);
      iree_test_loom_diagnostic_capture_begin(&diagnostic_capture, xfail);
      const iree_host_size_t failed_trial_begin = failed_trial_count;
      const bool had_planning_issues = scenario->issue_count != 0;
      if (had_planning_issues) {
        status = iree_test_loom_append_scenario_planning_issues(
            &module_plan, scenario, &planning_issues, &planning_issue_count);
      } else {
        status = iree_test_loom_run_scenario(
            &module_plan, scenario_index, &scenario_execution_options,
            &execution_options.materializer, entropy_root, &diagnostic_capture,
            &trials, &trial_count, &failed_trial_count);
      }
      if (!iree_test_loom_xfail_try_accept_compile_failure(
              xfail, &diagnostic_capture, &status) &&
          iree_status_is_ok(status) && xfail != NULL) {
        const iree_host_size_t failed_trial_delta =
            failed_trial_count - failed_trial_begin;
        if (had_planning_issues) {
          iree_test_loom_finish_xfail(
              xfail, &diagnostic_capture,
              IREE_TEST_LOOM_XFAIL_OUTCOME_DIAGNOSTIC_MISMATCH, 0, 0);
        } else if (failed_trial_delta == 0) {
          iree_test_loom_finish_xfail(
              xfail, &diagnostic_capture,
              xfail->policy == IREE_TEST_LOOM_XFAIL_POLICY_ALLOW_PASS
                  ? IREE_TEST_LOOM_XFAIL_OUTCOME_ALLOWED_PASS
                  : IREE_TEST_LOOM_XFAIL_OUTCOME_UNEXPECTED_PASS,
              0, 0);
        } else if (diagnostic_capture.matched_expected_diagnostic) {
          iree_test_loom_finish_xfail(
              xfail, &diagnostic_capture,
              IREE_TEST_LOOM_XFAIL_OUTCOME_EXPECTED_FAILURE, 0,
              failed_trial_delta);
        } else {
          iree_test_loom_finish_xfail(
              xfail, &diagnostic_capture,
              IREE_TEST_LOOM_XFAIL_OUTCOME_DIAGNOSTIC_MISMATCH, 0, 0);
        }
      }
    }
    if (iree_status_is_ok(status) && selected_case_count == 0 &&
        selected_scenario_count == 0) {
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "no check.case or check.scenario matched '%.*s'",
          (int)selected_case_name.size, selected_case_name.data);
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_array_end(&samples);
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_array_end(&trials);
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_array_end(&skipped_cases);
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_array_end(&planning_issues);
    }
    if (iree_status_is_ok(status)) {
      status = iree_test_loom_write_report(
          selected_case_count, selected_scenario_count, sample_count,
          failed_sample_count, trial_count, failed_trial_count,
          skipped_case_count, planning_issue_count, &xfails,
          iree_string_builder_view(&sample_output),
          iree_string_builder_view(&trial_output),
          iree_string_builder_view(&skipped_output),
          iree_string_builder_view(&planning_issue_output), &report_output);
    }
    if (iree_status_is_ok(status)) {
      status =
          loom_tooling_write_stdout(iree_string_builder_view(&report_output));
    }
    if (iree_status_is_ok(status)) {
      const iree_test_loom_xfail_counts_t xfail_counts =
          iree_test_loom_count_xfails(&xfails);
      if (xfail_counts.accepted_failed_sample_count > failed_sample_count) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "accepted xfail sample count exceeds raw failed sample count");
      } else if (xfail_counts.accepted_failed_trial_count >
                 failed_trial_count) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "accepted xfail trial count exceeds raw failed trial count");
      } else if (failed_sample_count -
                         xfail_counts.accepted_failed_sample_count !=
                     0 ||
                 failed_trial_count -
                         xfail_counts.accepted_failed_trial_count !=
                     0 ||
                 planning_issue_count != 0 || xfail_counts.xpass_count != 0 ||
                 xfail_counts.mismatch_count != 0) {
        exit_code = 1;
      }
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
  iree_string_builder_deinitialize(&trial_output);
  iree_string_builder_deinitialize(&sample_output);
  iree_arena_deinitialize(&execution_arena);
  iree_arena_deinitialize(&plan_arena);
  iree_hal_allocator_release(host_device_allocator);
  loom_run_hal_testbench_context_deinitialize(&hal_context);
  loom_config_text_binding_set_deinitialize(&config_set);
  if (device_event_capture_initialized) {
    loom_testbench_device_event_capture_deinitialize(&device_event_capture);
  }
  iree_arena_block_pool_deinitialize(&block_pool);
  loomc_target_profile_release(requested_target_profile);
  loomc_pass_program_release(pass_program);
  loomc_module_release(module);
  loomc_compiler_release(compiler);
  loomc_workspace_release(compiler_workspace);
  loomc_context_release(compiler_context);
  iree_allocator_free(allocator, config_bindings);
  iree_test_loom_xfail_list_deinitialize(&xfails, allocator);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
