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
#include "loom/ir/module.h"
#include "loom/target/artifact_manifest.h"
#include "loom/tooling/execution/compile_options.h"
#include "loom/tooling/execution/compile_report_capture.h"
#include "loom/tooling/io/file.h"
#include "loom/tooling/testbench/executor.h"
#include "loom/tools/iree-benchmark-loom/comparison_execution.h"
#include "loom/tools/iree-benchmark-loom/context.h"
#include "loom/tools/iree-benchmark-loom/diagnostics.h"
#include "loom/tools/iree-benchmark-loom/manifest.h"
#include "loom/tools/iree-benchmark-loom/model.h"
#include "loom/tools/iree-benchmark-loom/module_query.h"
#include "loom/tools/iree-benchmark-loom/output.h"
#include "loom/tools/iree-benchmark-loom/output_sink.h"
#include "loom/tools/iree-benchmark-loom/session.h"
#include "loom/tools/iree-benchmark-loom/work_execution.h"
#include "loom/tools/iree-benchmark-loom/work_plan.h"
#include "loom/verify/verify.h"

static iree_status_t iree_benchmark_loom_compile_report_options_initialize(
    const iree_benchmark_loom_options_t* options,
    loom_run_compile_report_capture_options_t* out_options) {
  loom_run_compile_report_capture_options_initialize(out_options);
  IREE_RETURN_IF_ERROR(loom_run_compile_report_capture_options_parse_request(
      options->compile_report, out_options));
  if (out_options->sink_format == LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_TEXT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "iree-benchmark-loom emits structured JSON reports; use "
        "--compile-report=summary, details, json-summary, or json-details");
  }
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_artifact_manifest_options_initialize(
    const iree_benchmark_loom_options_t* options,
    loom_run_candidate_artifact_manifest_options_t* out_options) {
  *out_options = (loom_run_candidate_artifact_manifest_options_t){0};
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_mode_parse(
      options->artifact_manifest, &out_options->mode));
  if (out_options->mode != LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE &&
      iree_string_view_is_empty(options->artifact_bundle_dir)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--artifact-manifest requires --artifact-bundle-dir so the manifest "
        "sidecars have a stable output location");
  }
  if (out_options->mode != LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE &&
      options->artifact_bundle_policy <
          IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_DEBUG) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--artifact-manifest requires --artifact-bundle-policy=debug or full "
        "so requested manifest sidecars are retained");
  }
  out_options->identifier = IREE_SV("artifact_manifest");
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
  const iree_string_view_t input_path = options->input_path;
  const bool compare_requested =
      !iree_string_view_is_empty(benchmark_options->compare);

  iree_io_file_contents_t* contents = NULL;
  loom_run_session_t session = {0};
  loom_run_module_t run_module = {0};
  iree_benchmark_loom_artifact_bundle_t artifact_bundle = {0};
  iree_benchmark_loom_file_provider_t file_provider = {0};
  iree_benchmark_loom_hal_context_t hal_context = {0};
  iree_benchmark_loom_hal_context_initialize(options->configuration, allocator,
                                             &hal_context);
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

  iree_status_t status = iree_ok_status();
  loom_run_compile_report_capture_options_t compile_report_options = {0};
  status = iree_benchmark_loom_compile_report_options_initialize(
      benchmark_options, &compile_report_options);
  loom_run_candidate_artifact_manifest_options_t artifact_manifest_options = {
      0};
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
    status = iree_benchmark_loom_session_initialize(options->configuration,
                                                    allocator, &session);
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
        filename, run_id, benchmark_options->file_output_dir,
        artifact_bundle.file_output_dir, &artifact_bundle, allocator,
        &file_provider);
  }
  const iree_benchmark_loom_run_identity_t run_identity = {
      .run_id = run_id,
      .source = filename,
      .results_path = iree_string_view_is_empty(results_output_path)
                          ? IREE_SV("-")
                          : results_output_path,
      .file_output_dir = file_provider.output_dir,
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
          benchmark_options->sample_compilation_mode);
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
    loom_run_module_parse_options_t parse_options = {0};
    loom_run_module_parse_options_initialize(&parse_options);
    parse_options.filename = filename;
    parse_options.source = source;
    parse_options.diagnostic_sink = (loom_diagnostic_sink_t){
        .fn = iree_benchmark_loom_diagnostic_capture_sink,
        .user_data = &source_diagnostics,
    };
    status = loom_run_module_parse(&session, &parse_options, &run_module);
    if (!iree_status_is_ok(status) && source_diagnostics.error_count != 0) {
      // The diagnostic sink owns the input rejection evidence; the status only
      // carries the same non-infrastructure failure.
      iree_status_free(status);
      status = iree_benchmark_loom_event_sink_emit_failure(
          event_sink, &run_identity, IREE_SV("parse"), IREE_SV("diagnostics"),
          IREE_SV("input module has parse errors"), &source_diagnostics);
      ++failure_count;
      exit_code = 1;
    }
  }

  if (iree_status_is_ok(status) && failure_count == 0) {
    iree_benchmark_loom_diagnostic_capture_deinitialize(&source_diagnostics);
    iree_benchmark_loom_diagnostic_capture_initialize(allocator,
                                                      &source_diagnostics);
    loom_verify_options_t verify_options = {0};
    verify_options.sink = (loom_diagnostic_sink_t){
        .fn = iree_benchmark_loom_diagnostic_capture_sink,
        .user_data = &source_diagnostics,
    };
    verify_options.max_errors = 20;
    verify_options.source_resolver =
        loom_run_module_source_resolver(&run_module);
    loom_verify_result_t verify_result = {0};
    status =
        loom_verify_module(run_module.module, &verify_options, &verify_result);
    if (iree_status_is_ok(status) && verify_result.error_count != 0) {
      status = iree_benchmark_loom_event_sink_emit_failure(
          event_sink, &run_identity, IREE_SV("verify"), IREE_SV("diagnostics"),
          IREE_SV("input module failed verification"), &source_diagnostics);
      ++failure_count;
      exit_code = 1;
    }
  }

  if (iree_status_is_ok(status) && failure_count == 0) {
    iree_arena_initialize(loom_run_session_block_pool(&session), &plan_arena);
    iree_arena_initialize(loom_run_session_block_pool(&session),
                          &execution_arena);
    loom_testbench_plan_options_t plan_options = {0};
    loom_testbench_plan_options_initialize(&plan_options);
    plan_options.max_samples_per_case = benchmark_options->max_samples_per_case;
    loom_testbench_module_plan_t module_plan = {0};
    status = loom_testbench_plan_module(run_module.module, &plan_options,
                                        &plan_arena, &module_plan);
    if (iree_status_is_ok(status)) {
      planned_case_count = module_plan.case_count;
      planned_benchmark_count = module_plan.benchmark_count;
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
    if (iree_status_is_ok(status)) {
      status = iree_benchmark_loom_work_plan_initialize(
          &module_plan, benchmark_options, allocator, &work_plan);
      if (iree_status_is_ok(status)) {
        work_plan_initialized = true;
        selected_benchmark_count = work_plan.selected_benchmark_count;
        logical_sample_count = work_plan.logical_sample_count;
        work_item_count = work_plan.work_item_count;
      }
    }

    if (iree_status_is_ok(status) && compare_requested) {
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
            benchmark_options, benchmark_options->sample_compilation_mode);
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
                .session = &session,
                .filename = filename,
                .source = source,
                .compile_report_options = &compile_report_options,
                .artifact_manifest_options = &artifact_manifest_options,
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
         iree_status_is_ok(status) && !compare_requested &&
         selection_index < work_plan.selected_benchmark_count;
         ++selection_index) {
      const iree_benchmark_loom_selected_benchmark_t* selection =
          &work_plan.selected_benchmarks[selection_index];
      status = iree_benchmark_loom_event_sink_emit_plan(
          event_sink, &run_identity, module_plan.module, selection,
          benchmark_options, benchmark_options->sample_compilation_mode);
      if (iree_status_is_ok(status) && benchmark_options->dry_run &&
          selection->policy.measure_kind ==
              IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE) {
        status = iree_benchmark_loom_event_sink_emit_device(
            event_sink, &run_identity, &hal_context);
      }
    }
    if (iree_status_is_ok(status) && benchmark_options->dry_run) {
      status = iree_benchmark_loom_event_sink_emit_work_plan(
          event_sink, &run_identity, module_plan.module, &work_plan);
    }
    if (iree_status_is_ok(status) && !compare_requested &&
        !benchmark_options->dry_run) {
      const iree_benchmark_loom_work_plan_execution_options_t
          work_execution_options = {
              .run = &run_identity,
              .module_plan = &module_plan,
              .work_plan = &work_plan,
              .benchmark_options = benchmark_options,
              .hal_context = &hal_context,
              .session = &session,
              .filename = filename,
              .source = source,
              .compile_report_options = &compile_report_options,
              .artifact_manifest_options = &artifact_manifest_options,
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
        benchmark_options->dry_run, benchmark_options->sample_compilation_mode);
  }
  if (iree_status_is_ok(status) && output_sink_initialized) {
    status = iree_benchmark_loom_output_sink_flush(&output_sink,
                                                   results_output_path);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_write_artifact_bundle_manifest(
        &artifact_bundle, &run_identity, &hal_context, source,
        options->command_line_json, benchmark_options->dry_run,
        benchmark_options->sample_compilation_mode, allocator);
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
  loom_run_module_deinitialize(&run_module);
  iree_benchmark_loom_hal_context_deinitialize(&hal_context);
  iree_benchmark_loom_file_provider_deinitialize(&file_provider);
  iree_benchmark_loom_artifact_bundle_deinitialize(&artifact_bundle);
  iree_io_file_contents_free(contents);
  loom_run_session_deinitialize(&session);

  out_result->exit_code = exit_code;
  return status;
}
