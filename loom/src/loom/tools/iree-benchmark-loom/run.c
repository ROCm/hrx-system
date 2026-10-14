// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/run.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/config/text_binding.h"
#include "loom/ir/module.h"
#include "loom/tooling/cli/loomc_options.h"
#include "loom/tooling/cli/loomc_result.h"
#include "loom/tooling/config/input.h"
#include "loom/tooling/input/loomc.h"
#include "loom/tooling/io/file.h"
#include "loom/tooling/testbench/device_event.h"
#include "loom/tooling/testbench/executor.h"
#include "loom/tools/iree-benchmark-loom/comparison_execution.h"
#include "loom/tools/iree-benchmark-loom/context.h"
#include "loom/tools/iree-benchmark-loom/diagnostics.h"
#include "loom/tools/iree-benchmark-loom/hal_actual.h"
#include "loom/tools/iree-benchmark-loom/manifest.h"
#include "loom/tools/iree-benchmark-loom/model.h"
#include "loom/tools/iree-benchmark-loom/module_query.h"
#include "loom/tools/iree-benchmark-loom/output.h"
#include "loom/tools/iree-benchmark-loom/output_sink.h"
#include "loom/tools/iree-benchmark-loom/work_execution.h"
#include "loom/tools/iree-benchmark-loom/work_plan.h"
#include "loomc/interop.h"
#include "loomc/iree.h"

enum {
  // Total bytes retained per runner scratch arena block.
  IREE_BENCHMARK_LOOM_BLOCK_POOL_BLOCK_SIZE = 128 * 1024,
};

static iree_status_t iree_benchmark_loom_compile_report_options_initialize(
    const iree_benchmark_loom_options_t* options,
    loomc_compile_report_options_t* out_options) {
  *out_options = (loomc_compile_report_options_t){
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
      .structure_size = sizeof(*out_options),
      .format = LOOMC_COMPILE_REPORT_FORMAT_JSON,
      .identifier = loomc_make_cstring_view("compile_report"),
  };
  if (iree_string_view_starts_with(options->compile_report, IREE_SV("text"))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "iree-benchmark-loom emits structured JSON reports; use "
        "--compile-report=summary, details, json-summary, or json-details");
  }
  return iree_status_from_loomc(loomc_compile_report_mode_parse(
      loomc_string_view_from_iree(options->compile_report),
      &out_options->mode));
}

static iree_status_t iree_benchmark_loom_artifact_manifest_options_initialize(
    const iree_benchmark_loom_options_t* options,
    loomc_artifact_manifest_options_t* out_options) {
  *out_options = (loomc_artifact_manifest_options_t){
      .type = LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS,
      .structure_size = sizeof(*out_options),
      .identifier = loomc_make_cstring_view("artifact_manifest"),
  };
  IREE_RETURN_IF_ERROR(
      iree_status_from_loomc(loomc_artifact_manifest_mode_parse(
          loomc_string_view_from_iree(options->artifact_manifest),
          &out_options->mode)));
  if (out_options->mode != LOOMC_ARTIFACT_MANIFEST_MODE_NONE &&
      iree_string_view_is_empty(options->artifact_bundle_dir)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--artifact-manifest requires --artifact-bundle-dir so the manifest "
        "sidecars have a stable output location");
  }
  if (out_options->mode != LOOMC_ARTIFACT_MANIFEST_MODE_NONE &&
      options->artifact_bundle_policy <
          IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_DEBUG) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--artifact-manifest requires --artifact-bundle-policy=debug or full "
        "so requested manifest sidecars are retained");
  }
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_append_config_assignments(
    loom_config_text_binding_set_t* config_set,
    iree_string_view_list_t assignments) {
  for (iree_host_size_t i = 0; i < assignments.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_text_binding_set_append_assignment(
        config_set, assignments.values[i]));
  }
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_append_config_files(
    loom_config_text_binding_set_t* config_set, iree_string_view_list_t paths,
    iree_allocator_t allocator) {
  for (iree_host_size_t i = 0; i < paths.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_text_binding_set_append_json_file(
        config_set, paths.values[i], allocator));
  }
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_print_compile_result(
    void* user_data, const loomc_result_t* result) {
  (void)user_data;
  bool succeeded = false;
  return loom_tooling_cli_print_loomc_result(stderr, result, &succeeded);
}

static bool iree_benchmark_loom_compare_selects_benchmark(
    iree_string_view_t compare, iree_string_view_t benchmark_name) {
  iree_string_view_t remaining = iree_string_view_trim(compare);
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &token, &remaining);
    token = iree_benchmark_loom_normalize_selection_name(token);
    if (iree_string_view_equal(token, benchmark_name)) {
      return true;
    }
    remaining = iree_string_view_trim(remaining);
  }
  return false;
}

static bool iree_benchmark_loom_selects_benchmark_for_planning(
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_options_t* options,
    const loom_testbench_benchmark_plan_t* benchmark_plan) {
  if (!iree_string_view_is_empty(options->compare)) {
    return iree_benchmark_loom_compare_selects_benchmark(options->compare,
                                                         benchmark_plan->name);
  }
  if (!iree_benchmark_loom_benchmark_matches_selection(
          benchmark_plan, options->selected_benchmark)) {
    return false;
  }
  if (benchmark_plan->case_index >= module_plan->case_count) {
    return true;
  }
  return iree_benchmark_loom_case_matches_selection(
      &module_plan->cases[benchmark_plan->case_index], options->selected_case);
}

static iree_status_t iree_benchmark_loom_emit_planning_failure(
    const iree_benchmark_loom_event_sink_t* event_sink,
    const iree_benchmark_loom_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_issue_t* issue,
    iree_host_size_t* inout_failure_count) {
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_event_sink_emit_planning_failure(
      event_sink, run, IREE_SV("plan"), IREE_SV("testbench_planning"),
      IREE_SV("selected benchmark cannot be planned for execution"),
      module_plan, issue, 1));
  ++*inout_failure_count;
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_emit_selected_planning_issues(
    const iree_benchmark_loom_event_sink_t* event_sink,
    const iree_benchmark_loom_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_options_t* options,
    iree_host_size_t* inout_failure_count) {
  for (iree_host_size_t benchmark_index = 0;
       benchmark_index < module_plan->benchmark_count; ++benchmark_index) {
    const loom_testbench_benchmark_plan_t* benchmark_plan =
        &module_plan->benchmarks[benchmark_index];
    if (!iree_benchmark_loom_selects_benchmark_for_planning(
            module_plan, options, benchmark_plan)) {
      continue;
    }

    for (iree_host_size_t i = 0; i < module_plan->issue_count; ++i) {
      const loom_testbench_issue_t* issue = &module_plan->issues[i];
      if (issue->benchmark_index == benchmark_index) {
        IREE_RETURN_IF_ERROR(iree_benchmark_loom_emit_planning_failure(
            event_sink, run, module_plan, issue, inout_failure_count));
      }
    }

    if (benchmark_plan->case_index >= module_plan->case_count) {
      continue;
    }
    const loom_testbench_case_plan_t* case_plan =
        &module_plan->cases[benchmark_plan->case_index];
    for (iree_host_size_t i = 0; i < case_plan->issue_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_benchmark_loom_emit_planning_failure(
          event_sink, run, module_plan, &case_plan->issues[i],
          inout_failure_count));
    }
  }
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_run_file(
    const iree_benchmark_loom_file_run_options_t* options,
    iree_benchmark_loom_run_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_result);
  *out_result = (iree_benchmark_loom_run_result_t){0};

  iree_benchmark_loom_options_t normalized_benchmark_options =
      *options->benchmark_options;
  normalized_benchmark_options.selected_case =
      iree_benchmark_loom_normalize_selection_name(
          normalized_benchmark_options.selected_case);
  normalized_benchmark_options.selected_benchmark =
      iree_benchmark_loom_normalize_selection_name(
          normalized_benchmark_options.selected_benchmark);
  const iree_benchmark_loom_options_t* benchmark_options =
      &normalized_benchmark_options;
  const iree_allocator_t allocator = options->host_allocator;
  loom_config_text_binding_set_t config_set;
  loom_config_text_binding_set_initialize(allocator, &config_set);
  loomc_config_binding_t* config_bindings = NULL;
  loomc_config_options_t config_options = {0};
  const iree_string_view_t input_path = options->input_path;
  const bool compare_requested =
      !iree_string_view_is_empty(benchmark_options->compare);

  iree_io_file_contents_t* contents = NULL;
  loomc_context_t* compiler_context = NULL;
  loomc_workspace_t* compiler_workspace = NULL;
  loomc_compiler_t* compiler = NULL;
  loomc_module_t* module = NULL;
  loomc_pass_program_t* pass_program = NULL;
  loomc_target_profile_t* requested_target_profile = NULL;
  loomc_sanitizer_options_t loomc_sanitizer_options = {0};
  loomc_module_interop_view_t module_view = {0};
  const loom_module_t* native_module = NULL;
  const loom_source_table_resolver_t* source_table = NULL;
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(IREE_BENCHMARK_LOOM_BLOCK_POOL_BLOCK_SIZE,
                                   allocator, &block_pool);
  iree_benchmark_loom_artifact_bundle_t artifact_bundle = {0};
  iree_benchmark_loom_file_provider_t file_provider = {0};
  iree_benchmark_loom_hal_context_t hal_context = {0};
  iree_benchmark_loom_hal_context_initialize(options->configuration, allocator,
                                             &hal_context);
  loom_testbench_device_event_capture_t device_event_capture = {0};
  bool device_event_capture_initialized = false;
  iree_arena_allocator_t plan_arena;
  memset(&plan_arena, 0, sizeof(plan_arena));
  iree_arena_allocator_t execution_arena;
  memset(&execution_arena, 0, sizeof(execution_arena));
  iree_benchmark_loom_output_sink_t output_sink = {0};
  bool output_sink_initialized = false;
  const iree_benchmark_loom_event_sink_t* event_sink = options->event_sink;
  iree_benchmark_loom_diagnostic_capture_t source_diagnostics = {0};
  iree_benchmark_loom_diagnostic_capture_initialize(allocator,
                                                    &source_diagnostics);
  iree_host_size_t planned_case_count = 0;
  iree_host_size_t planned_benchmark_count = 0;
  iree_host_size_t selected_benchmark_count = 0;
  iree_host_size_t logical_sample_count = 0;
  iree_host_size_t work_item_count = 0;
  iree_host_size_t failure_count = 0;
  iree_host_size_t failed_benchmark_count = 0;
  iree_host_size_t correctness_sample_count = 0;
  iree_host_size_t correctness_failed_sample_count = 0;
  int exit_code = 0;

  iree_status_t status =
      loom_run_hal_testbench_context_validate_explicit_device(
          &hal_context.execution);
  const bool sanitizer_enabled =
      loom_sanitizer_options_is_enabled(&benchmark_options->sanitizer);
  if (iree_status_is_ok(status) && sanitizer_enabled &&
      !loom_tooling_cli_pipeline_uses_default(benchmark_options->pipeline)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--sanitizer and --sanitizer-reporting require --pipeline=default");
  }
  if (iree_status_is_ok(status) && sanitizer_enabled) {
    loom_tooling_cli_make_loomc_sanitizer_options(&benchmark_options->sanitizer,
                                                  &loomc_sanitizer_options);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_append_config_assignments(
        &config_set, benchmark_options->config_assignments);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_append_config_files(
        &config_set, benchmark_options->config_files, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_cli_make_loomc_config_options(
        &config_set, allocator, &config_bindings, &config_options);
  }
  loomc_compile_report_options_t compile_report_options = {0};
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_compile_report_options_initialize(
        benchmark_options, &compile_report_options);
  }
  loomc_artifact_manifest_options_t artifact_manifest_options = {0};
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_artifact_manifest_options_initialize(
        benchmark_options, &artifact_manifest_options);
  }
  if (iree_status_is_ok(status)) {
    iree_benchmark_loom_artifact_bundle_options_t artifact_bundle_options = {
        .dir = benchmark_options->artifact_bundle_dir,
        .policy = benchmark_options->artifact_bundle_policy,
        .output_format = benchmark_options->output_format,
    };
    status = iree_benchmark_loom_artifact_bundle_initialize(
        &artifact_bundle_options, allocator, &artifact_bundle);
    if (iree_status_is_ok(status)) {
      hal_context.artifact_bundle = &artifact_bundle;
    }
  }

  if (iree_status_is_ok(status)) {
    const loomc_context_target_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        .structure_size = sizeof(target_options),
        .target_environment = options->configuration->target_environment,
    };
    const loomc_context_options_t context_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        .structure_size = sizeof(context_options),
        .next = options->configuration->target_environment != NULL
                    ? &target_options
                    : NULL,
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
  const iree_string_view_t filename =
      (iree_string_view_is_empty(input_path) ||
       iree_string_view_equal(input_path, IREE_SV("-")))
          ? IREE_SV("<stdin>")
          : input_path;
  char run_id_storage[32];
  snprintf(run_id_storage, sizeof(run_id_storage), "r%016" PRIx64,
           (uint64_t)iree_time_now());
  iree_string_view_t run_id = iree_make_cstring_view(run_id_storage);
  const iree_string_view_t results_output_path =
      iree_benchmark_loom_effective_results_output_path(
          benchmark_options->output, &artifact_bundle);
  const iree_string_view_t profile_artifacts_dir =
      iree_benchmark_loom_effective_profile_artifacts_dir(
          benchmark_options->profile_artifacts_dir, &artifact_bundle);
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_file_provider_initialize(
        filename, run_id, benchmark_options->file_output_root,
        artifact_bundle.file_output_dir, &artifact_bundle, allocator,
        &file_provider);
  }
  const iree_benchmark_loom_run_identity_t run_identity = {
      .run_id = run_id,
      .source = filename,
      .results_path = iree_string_view_is_empty(results_output_path)
                          ? IREE_SV("-")
                          : results_output_path,
      .file_output_dir = file_provider.output_root,
      .profile_artifacts_dir = profile_artifacts_dir,
      .artifact_bundle_dir = artifact_bundle.dir,
      .artifact_bundle_policy = iree_benchmark_loom_artifact_bundle_policy_name(
          artifact_bundle.policy),
  };
  if (iree_status_is_ok(status)) {
    if (event_sink == NULL) {
      status = iree_benchmark_loom_output_sink_initialize(
          benchmark_options->output_format, results_output_path, allocator,
          &output_sink);
      if (iree_status_is_ok(status)) {
        output_sink_initialized = true;
        event_sink = &output_sink.event_sink;
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_benchmark_loom_event_sink_emit_run(
          event_sink, &run_identity, benchmark_options->dry_run,
          &benchmark_options->sanitizer);
    }
  }
  iree_string_view_t source = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    status = loom_tooling_read_input_file(input_path, allocator, &contents);
    if (iree_status_is_ok(status)) {
      source = loom_tooling_file_contents_string_view(contents);
    }
  }
  if (iree_status_is_ok(status)) {
    const loom_tooling_loomc_input_options_t admission_options = {
        .providers = options->configuration->input_providers,
        .input = benchmark_options->input,
        .path = input_path,
        .source = source,
        .import = options->configuration->import,
        .import_user_data = options->configuration->import_user_data,
    };
    loomc_result_t* result = NULL;
    status = loom_tooling_input_admit_loomc_module(
        &admission_options, compiler_context, compiler_workspace, &block_pool,
        &module, &result, allocator);
    if (iree_status_is_ok(status)) {
      status = iree_benchmark_loom_diagnostic_capture_loomc_result(
          &source_diagnostics, result);
    }
    if (iree_status_is_ok(status) && !loomc_result_succeeded(result)) {
      status = iree_benchmark_loom_event_sink_emit_failure(
          event_sink, &run_identity, IREE_SV("parse"), IREE_SV("diagnostics"),
          IREE_SV("input module has parse errors"), &source_diagnostics);
      ++failure_count;
      exit_code = 1;
    }
    loomc_result_release(result);
  }

  if (iree_status_is_ok(status) && failure_count == 0) {
    iree_benchmark_loom_diagnostic_capture_deinitialize(&source_diagnostics);
    iree_benchmark_loom_diagnostic_capture_initialize(allocator,
                                                      &source_diagnostics);
    loomc_result_t* result = NULL;
    status = iree_status_from_loomc(loomc_module_get_interop_view(
        module, loomc_allocator_from_iree(allocator), &module_view, &result));
    if (iree_status_is_ok(status)) {
      status = iree_benchmark_loom_diagnostic_capture_loomc_result(
          &source_diagnostics, result);
    }
    if (iree_status_is_ok(status) && !loomc_result_succeeded(result)) {
      status = iree_benchmark_loom_event_sink_emit_failure(
          event_sink, &run_identity, IREE_SV("verify"), IREE_SV("diagnostics"),
          IREE_SV("input module failed verification"), &source_diagnostics);
      ++failure_count;
      exit_code = 1;
    }
    loomc_result_release(result);
  }

  if (iree_status_is_ok(status) && failure_count == 0) {
    // Native planning and HAL/Wasm execution inspect this exact-version view.
    // The public module owns its IR and source snapshots for the full run.
    native_module = module_view.module;
    source_table = module_view.source_table;
  }

  const iree_string_view_t requested_target =
      iree_string_view_trim(benchmark_options->target);
  if (iree_status_is_ok(status) && failure_count == 0 &&
      !iree_string_view_is_empty(requested_target)) {
    status = iree_status_from_loomc(loomc_target_profile_select(
        options->configuration->target_environment,
        loomc_string_view_from_iree(requested_target),
        loomc_allocator_from_iree(allocator), &requested_target_profile));
  }
  if (iree_status_is_ok(status) && failure_count == 0) {
    iree_benchmark_loom_diagnostic_capture_deinitialize(&source_diagnostics);
    iree_benchmark_loom_diagnostic_capture_initialize(allocator,
                                                      &source_diagnostics);
    loomc_result_t* result = NULL;
    status = loom_tooling_cli_prepare_loomc_pass_program(
        compiler_context, module, benchmark_options->pipeline,
        IREE_SV("iree-benchmark-loom pipeline"), &pass_program, &result,
        allocator);
    if (iree_status_is_ok(status) && result != NULL) {
      status = iree_benchmark_loom_diagnostic_capture_loomc_result(
          &source_diagnostics, result);
    }
    if (iree_status_is_ok(status) && result != NULL &&
        !loomc_result_succeeded(result)) {
      status = iree_benchmark_loom_event_sink_emit_failure(
          event_sink, &run_identity, IREE_SV("compile"),
          IREE_SV("pipeline_diagnostics"),
          IREE_SV("pass program preparation failed"), &source_diagnostics);
      ++failure_count;
      exit_code = 1;
    }
    loomc_result_release(result);
  }

  if (iree_status_is_ok(status) && failure_count == 0) {
    status = loom_run_hal_testbench_context_add_module_runtime_requirements(
        &hal_context.execution, module,
        sanitizer_enabled ? &loomc_sanitizer_options : NULL);
  }

  if (iree_status_is_ok(status) && failure_count == 0) {
    iree_arena_initialize(&block_pool, &plan_arena);
    iree_arena_initialize(&block_pool, &execution_arena);
    loom_testbench_plan_options_t plan_options = {0};
    loom_testbench_plan_options_initialize(&plan_options);
    plan_options.max_samples_per_case = benchmark_options->max_samples_per_case;
    loom_testbench_module_plan_t module_plan = {0};
    status = loom_testbench_plan_module(native_module, &plan_options,
                                        &plan_arena, &module_plan);
    if (iree_status_is_ok(status)) {
      planned_case_count = module_plan.case_count;
      planned_benchmark_count = module_plan.benchmark_count;
      status = iree_benchmark_loom_emit_selected_planning_issues(
          event_sink, &run_identity, &module_plan, benchmark_options,
          &failure_count);
      if (iree_status_is_ok(status) && failure_count != 0) {
        exit_code = 1;
      }
    }
    loom_testbench_case_execution_options_t execution_options = {0};
    loom_testbench_case_execution_options_initialize(&execution_options);
    execution_options.materializer.host_allocator = allocator;
    execution_options.materializer.open_read_file =
        (loom_testbench_file_open_callback_t){
            .fn = iree_benchmark_loom_open_file_for_read,
            .user_data = &file_provider,
        };
    execution_options.materializer.open_write_file =
        (loom_testbench_file_open_callback_t){
            .fn = iree_benchmark_loom_open_file_for_write,
            .user_data = &file_provider,
        };

    iree_benchmark_loom_work_plan_t work_plan = {0};
    bool work_plan_initialized = false;
    if (iree_status_is_ok(status) && failure_count == 0) {
      status = iree_benchmark_loom_work_plan_initialize(
          &module_plan, benchmark_options, allocator, &work_plan);
      if (iree_status_is_ok(status)) {
        work_plan_initialized = true;
        selected_benchmark_count = work_plan.selected_benchmark_count;
        logical_sample_count = work_plan.logical_sample_count;
        work_item_count = work_plan.work_item_count;
      }
    }

    if (iree_status_is_ok(status) && !benchmark_options->dry_run) {
      status = loom_testbench_device_event_capture_initialize(
          LOOM_TESTBENCH_DEVICE_EVENT_DEFAULT_CAPACITY, allocator,
          &device_event_capture);
      if (iree_status_is_ok(status)) {
        device_event_capture_initialized = true;
        execution_options.device_event_capture = &device_event_capture;
        loom_run_hal_testbench_context_set_device_event_sink(
            &hal_context.execution,
            loom_testbench_device_event_capture_sink(&device_event_capture));
      }
    }

    const loom_testbench_compilation_t compilation = {
        .compiler = compiler,
        .workspace = compiler_workspace,
        .module = module,
        .config = &config_options,
    };
    const loom_testbench_compile_result_callback_t compile_result_callback = {
        .fn = iree_benchmark_loom_print_compile_result,
    };
    const iree_benchmark_loom_hal_compilation_options_t hal_compilation = {
        .compilation = &compilation,
        .native_module = native_module,
        .source_table = source_table,
        .pass_program = pass_program,
        .requested_target_profile = requested_target_profile,
        .sanitizer = sanitizer_enabled ? &loomc_sanitizer_options : NULL,
        .result_callback = compile_result_callback,
        .compile_report = compile_report_options,
        .artifact_manifest = artifact_manifest_options,
    };
    const loom_testbench_function_call_provider_callback_t function_calls =
        options->configuration->function_call_provider;
    if (iree_status_is_ok(status) && failure_count == 0 && function_calls.fn) {
      iree_host_size_t function_call_capacity = 0;
      for (iree_host_size_t i = 0; i < work_plan.selected_benchmark_count;
           ++i) {
        const loom_testbench_case_plan_t* case_plan =
            work_plan.selected_benchmarks[i].case_plan;
        if (case_plan != NULL && case_plan->issue_count == 0) {
          function_call_capacity += case_plan->invocation_count;
        }
        const loom_testbench_scenario_plan_t* scenario_plan =
            work_plan.selected_benchmarks[i].scenario_plan;
        if (scenario_plan != NULL && scenario_plan->issue_count == 0) {
          for (iree_host_size_t trial_index = 0;
               trial_index < scenario_plan->trial_count; ++trial_index) {
            function_call_capacity +=
                scenario_plan->trials[trial_index].recipe_step_count;
          }
        }
      }
      const loom_testbench_invocation_plan_t** selected_calls = NULL;
      if (function_call_capacity != 0) {
        status = iree_arena_allocate_array(&plan_arena, function_call_capacity,
                                           sizeof(*selected_calls),
                                           (void**)&selected_calls);
      }
      iree_host_size_t function_call_count = 0;
      for (iree_host_size_t i = 0;
           iree_status_is_ok(status) && i < work_plan.selected_benchmark_count;
           ++i) {
        const loom_testbench_case_plan_t* case_plan =
            work_plan.selected_benchmarks[i].case_plan;
        if (case_plan != NULL && case_plan->issue_count == 0) {
          for (iree_host_size_t j = 0; j < case_plan->invocation_count; ++j) {
            const loom_testbench_invocation_plan_t* invocation =
                &case_plan->invocations[j];
            if (invocation->kind == LOOM_TESTBENCH_INVOCATION_FUNCTION_CALL) {
              selected_calls[function_call_count++] = invocation;
            }
          }
        }
        const loom_testbench_scenario_plan_t* scenario_plan =
            work_plan.selected_benchmarks[i].scenario_plan;
        if (scenario_plan == NULL || scenario_plan->issue_count != 0) {
          continue;
        }
        for (iree_host_size_t trial_index = 0;
             trial_index < scenario_plan->trial_count; ++trial_index) {
          const loom_testbench_trial_plan_t* trial =
              &scenario_plan->trials[trial_index];
          for (iree_host_size_t step_index = 0;
               step_index < trial->recipe_step_count; ++step_index) {
            const loom_testbench_trial_recipe_step_t* step =
                &trial->recipe_steps[step_index];
            if (step->kind == LOOM_TESTBENCH_TRIAL_RECIPE_STEP_GENERATOR) {
              selected_calls[function_call_count++] = &step->generator;
            }
          }
        }
      }
      if (iree_status_is_ok(status) && function_call_count != 0) {
        execution_options.invocation.function_call =
            function_calls.fn(function_calls.user_data, &compilation,
                              (loom_testbench_invocation_plan_list_t){
                                  .values = selected_calls,
                                  .count = function_call_count,
                              },
                              compile_result_callback);
      }
    }

    if (iree_status_is_ok(status) && failure_count == 0 && compare_requested) {
      const iree_benchmark_loom_selected_benchmark_t* selections =
          work_plan.selected_benchmarks;
      const iree_host_size_t selection_count =
          work_plan.selected_benchmark_count;
      for (iree_host_size_t i = 0;
           iree_status_is_ok(status) && i < selection_count; ++i) {
        if (selections[i].policy.measure_kind !=
            IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE) {
          status =
              iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                               "--compare benchmark `%.*s` must use measure = "
                               "\"dispatch_complete\"",
                               (int)selections[i].benchmark_plan->name.size,
                               selections[i].benchmark_plan->name.data);
          break;
        }
        status = iree_benchmark_loom_event_sink_emit_plan(
            event_sink, &run_identity, module_plan.module, &selections[i],
            benchmark_options);
        if (iree_status_is_ok(status) && benchmark_options->dry_run) {
          status = iree_benchmark_loom_event_sink_emit_device(
              event_sink, &run_identity, &hal_context);
        }
      }
      if (iree_status_is_ok(status) && !benchmark_options->dry_run) {
        const iree_benchmark_loom_comparison_execution_options_t
            comparison_execution_options = {
                .run = &run_identity,
                .module_plan = &module_plan,
                .work_plan = &work_plan,
                .benchmark_options = benchmark_options,
                .hal_context = &hal_context,
                .compilation = &hal_compilation,
                .case_execution_options = &execution_options,
                .execution_arena = &execution_arena,
                .host_allocator = allocator,
                .event_sink = event_sink,
            };
        status = iree_benchmark_loom_run_dispatch_comparison(
            &comparison_execution_options, &correctness_sample_count,
            &correctness_failed_sample_count, &failed_benchmark_count);
      }
    }

    for (iree_host_size_t selection_index = 0;
         iree_status_is_ok(status) && failure_count == 0 &&
         !compare_requested &&
         selection_index < work_plan.selected_benchmark_count;
         ++selection_index) {
      const iree_benchmark_loom_selected_benchmark_t* selection =
          &work_plan.selected_benchmarks[selection_index];
      status = iree_benchmark_loom_event_sink_emit_plan(
          event_sink, &run_identity, module_plan.module, selection,
          benchmark_options);
      if (iree_status_is_ok(status) && benchmark_options->dry_run &&
          selection->policy.measure_kind ==
              IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE) {
        status = iree_benchmark_loom_event_sink_emit_device(
            event_sink, &run_identity, &hal_context);
      }
    }
    if (iree_status_is_ok(status) && failure_count == 0 &&
        benchmark_options->dry_run) {
      status = iree_benchmark_loom_event_sink_emit_work_plan(
          event_sink, &run_identity, module_plan.module, &work_plan);
    }
    if (iree_status_is_ok(status) && failure_count == 0 && !compare_requested &&
        !benchmark_options->dry_run) {
      const iree_benchmark_loom_work_plan_execution_options_t
          work_execution_options = {
              .run = &run_identity,
              .module_plan = &module_plan,
              .work_plan = &work_plan,
              .benchmark_options = benchmark_options,
              .hal_context = &hal_context,
              .compilation = &hal_compilation,
              .case_execution_options = &execution_options,
              .execution_arena = &execution_arena,
              .host_allocator = allocator,
              .event_sink = event_sink,
          };
      status = iree_benchmark_loom_run_work_plan(
          &work_execution_options, &correctness_sample_count,
          &correctness_failed_sample_count, &failed_benchmark_count);
    }
    if (iree_status_is_ok(status) &&
        (failed_benchmark_count != 0 || correctness_failed_sample_count != 0)) {
      exit_code = 1;
    }
    if (work_plan_initialized) {
      iree_benchmark_loom_work_plan_deinitialize(&work_plan);
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_event_sink_emit_summary(
        event_sink, &run_identity, &artifact_bundle, planned_case_count,
        planned_benchmark_count, selected_benchmark_count, logical_sample_count,
        work_item_count, failure_count, failed_benchmark_count,
        correctness_sample_count, correctness_failed_sample_count,
        benchmark_options->dry_run);
  }
  if (iree_status_is_ok(status) && output_sink_initialized) {
    status = iree_benchmark_loom_output_sink_flush(&output_sink,
                                                   results_output_path);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_write_artifact_bundle_manifest(
        &artifact_bundle, &run_identity, &hal_context, source,
        options->command_line_json, benchmark_options->dry_run, allocator);
  }
  if (iree_status_is_ok(status) && failure_count != 0) {
    exit_code = 1;
  }
  if (!iree_status_is_ok(status)) {
    exit_code = 1;
  }

  iree_benchmark_loom_diagnostic_capture_deinitialize(&source_diagnostics);
  if (output_sink_initialized) {
    iree_benchmark_loom_output_sink_deinitialize(&output_sink);
  }
  iree_arena_deinitialize(&execution_arena);
  iree_arena_deinitialize(&plan_arena);
  iree_benchmark_loom_hal_context_deinitialize(&hal_context);
  if (device_event_capture_initialized) {
    loom_testbench_device_event_capture_deinitialize(&device_event_capture);
  }
  loom_config_text_binding_set_deinitialize(&config_set);
  iree_benchmark_loom_file_provider_deinitialize(&file_provider);
  iree_benchmark_loom_artifact_bundle_deinitialize(&artifact_bundle);
  iree_io_file_contents_free(contents);
  iree_arena_block_pool_deinitialize(&block_pool);
  loomc_target_profile_release(requested_target_profile);
  loomc_pass_program_release(pass_program);
  loomc_module_release(module);
  loomc_compiler_release(compiler);
  loomc_workspace_release(compiler_workspace);
  loomc_context_release(compiler_context);
  iree_allocator_free(allocator, config_bindings);

  out_result->exit_code = exit_code;
  return status;
}
