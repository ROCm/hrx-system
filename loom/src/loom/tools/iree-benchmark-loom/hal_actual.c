// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/hal_actual.h"

#include <string.h>

#include "loom/tooling/execution/compile_options.h"
#include "loom/tools/iree-benchmark-loom/diagnostics.h"
#include "loom/tools/iree-benchmark-loom/module_query.h"

iree_status_t iree_benchmark_loom_hal_actual_provider_initialize(
    iree_benchmark_loom_hal_context_t* context, loom_run_session_t* session,
    const loom_run_module_t* run_module, iree_string_view_t pipeline,
    loom_sanitizer_options_t sanitizer,
    const loom_testbench_invocation_plan_t* kernel_launch,
    iree_string_view_t artifact_path_suffix,
    const loom_run_compile_report_capture_options_t* compile_report_options,
    const loom_run_candidate_artifact_manifest_options_t*
        artifact_manifest_options,
    iree_benchmark_loom_hal_actual_provider_t* out_provider) {
  *out_provider = (iree_benchmark_loom_hal_actual_provider_t){
      .context = context,
      .artifact_path_suffix = artifact_path_suffix,
  };
  iree_allocator_t host_allocator = context->execution.host_allocator;
  iree_benchmark_loom_diagnostic_capture_initialize(host_allocator,
                                                    &out_provider->diagnostics);
  loom_run_candidate_artifact_manifest_options_t artifact_manifest = {0};
  if (artifact_manifest_options != NULL) {
    artifact_manifest = *artifact_manifest_options;
  }
  iree_status_t status = loom_run_compile_report_capture_initialize(
      compile_report_options, host_allocator,
      &out_provider->compile_report_capture);
  if (iree_status_is_ok(status)) {
    out_provider->compile_report_capture_initialized = true;
    loom_run_candidate_compile_options_t report_options = {0};
    loom_run_candidate_compile_options_initialize(&report_options);
    loom_run_compile_report_capture_configure_compile_options(
        &out_provider->compile_report_capture, &report_options);
    loom_run_candidate_artifact_flags_t artifact_flags = 0;
    if (context->artifact_bundle != NULL && context->artifact_bundle->enabled &&
        context->artifact_bundle->policy >=
            IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_DEBUG) {
      artifact_flags |= LOOM_RUN_CANDIDATE_ARTIFACT_FLAG_TARGET_LISTING;
    }
    loom_run_hal_testbench_actual_provider_options_t provider_options = {
        .context = &context->execution,
        .session = session,
        .target_environment = context->configuration->target_environment,
        .run_module = run_module,
        .pipeline = pipeline,
        .sanitizer = sanitizer,
        .config_set = context->config_set,
        .kernel_launch = kernel_launch,
        .diagnostic_sink =
            (loom_diagnostic_sink_t){
                .fn = iree_benchmark_loom_diagnostic_capture_sink,
                .user_data = &out_provider->diagnostics,
            },
        .report = report_options.report,
        .artifact_flags = artifact_flags,
        .artifact_manifest = artifact_manifest,
    };
    loom_run_hal_testbench_actual_provider_initialize(&provider_options,
                                                      &out_provider->execution);
  } else {
    iree_benchmark_loom_diagnostic_capture_deinitialize(
        &out_provider->diagnostics);
    *out_provider = (iree_benchmark_loom_hal_actual_provider_t){0};
  }
  return status;
}

void iree_benchmark_loom_hal_actual_provider_deinitialize(
    iree_benchmark_loom_hal_actual_provider_t* provider) {
  if (provider == NULL) {
    return;
  }
  if (provider->context == NULL) {
    *provider = (iree_benchmark_loom_hal_actual_provider_t){0};
    return;
  }
  loom_run_hal_testbench_actual_provider_deinitialize(&provider->execution);
  if (provider->compile_report_capture_initialized) {
    loom_run_compile_report_capture_deinitialize(
        &provider->compile_report_capture);
  }
  iree_allocator_t host_allocator = provider->context->execution.host_allocator;
  iree_allocator_free(host_allocator, provider->hal_executable_path_storage);
  iree_allocator_free(host_allocator, provider->target_artifact_path_storage);
  iree_allocator_free(host_allocator, provider->target_listing_path_storage);
  iree_allocator_free(host_allocator,
                      provider->compile_report_artifact_path_storage);
  iree_allocator_free(host_allocator, provider->artifact_manifest_path_storage);
  iree_benchmark_loom_diagnostic_capture_deinitialize(&provider->diagnostics);
  *provider = (iree_benchmark_loom_hal_actual_provider_t){0};
}

iree_status_t iree_benchmark_loom_hal_actual_sequence_initialize(
    iree_benchmark_loom_hal_context_t* context, loom_run_session_t* session,
    const loom_run_module_t* run_module, iree_string_view_t pipeline,
    loom_sanitizer_options_t sanitizer,
    const loom_testbench_case_plan_t* case_plan,
    const loom_run_compile_report_capture_options_t* compile_report_options,
    const loom_run_candidate_artifact_manifest_options_t*
        artifact_manifest_options,
    iree_benchmark_loom_hal_actual_sequence_t* out_sequence) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(case_plan);
  IREE_ASSERT_ARGUMENT(out_sequence);
  iree_allocator_t host_allocator = context->execution.host_allocator;
  *out_sequence = (iree_benchmark_loom_hal_actual_sequence_t){
      .host_allocator = host_allocator,
  };

  iree_host_size_t kernel_launch_count = 0;
  iree_status_t status = loom_run_hal_testbench_count_kernel_launches(
      case_plan, &kernel_launch_count);
  if (iree_status_is_ok(status) && kernel_launch_count == 0) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL actual sequence requires at least one kernel launch in check.case "
        "`%.*s`",
        (int)case_plan->name.size, case_plan->name.data);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, kernel_launch_count,
                                         sizeof(*out_sequence->providers),
                                         (void**)&out_sequence->providers);
  }
  if (iree_status_is_ok(status)) {
    memset(out_sequence->providers, 0,
           kernel_launch_count * sizeof(*out_sequence->providers));
  }

  iree_host_size_t provider_index = 0;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < case_plan->invocation_count; ++i) {
    const loom_testbench_invocation_plan_t* invocation =
        &case_plan->invocations[i];
    if (invocation->kind != LOOM_TESTBENCH_INVOCATION_KERNEL_LAUNCH) {
      continue;
    }
    iree_string_view_t artifact_path_suffix = iree_string_view_empty();
    if (kernel_launch_count > 1) {
      status = iree_benchmark_loom_module_symbol_name_from_ref(
          run_module->module, invocation->callee_ref, &artifact_path_suffix);
    }
    if (iree_status_is_ok(status)) {
      status = iree_benchmark_loom_hal_actual_provider_initialize(
          context, session, run_module, pipeline, sanitizer, invocation,
          artifact_path_suffix, compile_report_options,
          artifact_manifest_options, &out_sequence->providers[provider_index]);
    }
    if (iree_status_is_ok(status)) {
      out_sequence->provider_count = ++provider_index;
    }
  }
  loom_run_hal_testbench_actual_provider_t** execution_providers = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, out_sequence->provider_count,
        sizeof(*execution_providers), (void**)&execution_providers);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < out_sequence->provider_count; ++i) {
    execution_providers[i] = &out_sequence->providers[i].execution;
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_actual_sequence_execution_create(
        case_plan, out_sequence->provider_count, execution_providers,
        host_allocator, &out_sequence->execution);
  }
  iree_allocator_free(host_allocator, execution_providers);
  if (!iree_status_is_ok(status)) {
    iree_benchmark_loom_hal_actual_sequence_deinitialize(out_sequence);
  }
  return status;
}

void iree_benchmark_loom_hal_actual_sequence_deinitialize(
    iree_benchmark_loom_hal_actual_sequence_t* sequence) {
  if (sequence == NULL) {
    return;
  }
  loom_run_hal_testbench_actual_sequence_execution_destroy(sequence->execution);
  for (iree_host_size_t i = 0; i < sequence->provider_count; ++i) {
    iree_benchmark_loom_hal_actual_provider_deinitialize(
        &sequence->providers[i]);
  }
  iree_allocator_free(sequence->host_allocator, sequence->providers);
  *sequence = (iree_benchmark_loom_hal_actual_sequence_t){0};
}

iree_status_t iree_benchmark_loom_hal_actual_provider_compile(
    iree_benchmark_loom_hal_actual_provider_t* provider) {
  return loom_run_hal_testbench_actual_provider_compile(&provider->execution);
}

void iree_benchmark_loom_benchmark_result_set_compile_rejection(
    const iree_benchmark_loom_hal_actual_provider_t* provider,
    iree_benchmark_loom_benchmark_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));
  out_result->state = IREE_SV("compile_failed");
  out_result->has_failure = true;
  out_result->failure_entry =
      provider->execution.invocation_options.function_name;
  out_result->failure_stage = provider->execution.compile_failure_stage;
  out_result->failure_kind = provider->execution.compile_failure_kind;
  out_result->failure_message = provider->execution.compile_failure_message;
  out_result->diagnostic_error_count = provider->diagnostics.error_count;
  out_result->diagnostic_warning_count = provider->diagnostics.warning_count;
  out_result->diagnostic_remark_count = provider->diagnostics.remark_count;
  out_result->diagnostic_json =
      iree_benchmark_loom_diagnostic_capture_json(&provider->diagnostics);
  if (provider->execution.compile_report_available) {
    out_result->compile_report_capture = &provider->compile_report_capture;
  }
  out_result->compile_report_artifact_path =
      provider->compile_report_artifact_path;
  out_result->artifact_manifest_path = provider->artifact_manifest_path;
  out_result->target_artifact_path = provider->target_artifact_path;
  out_result->target_listing_path = provider->target_listing_path;
  out_result->hal_executable_path = provider->hal_executable_path;
}

iree_status_t iree_benchmark_loom_hal_actual_sequence_compile(
    iree_benchmark_loom_hal_actual_sequence_t* sequence) {
  for (iree_host_size_t i = 0; i < sequence->provider_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_hal_actual_provider_compile(
        &sequence->providers[i]));
  }
  return iree_ok_status();
}

const iree_benchmark_loom_hal_actual_provider_t*
iree_benchmark_loom_hal_actual_sequence_first_rejection(
    const iree_benchmark_loom_hal_actual_sequence_t* sequence) {
  for (iree_host_size_t i = 0; i < sequence->provider_count; ++i) {
    const iree_benchmark_loom_hal_actual_provider_t* provider =
        &sequence->providers[i];
    if (provider->execution.compile_rejected) {
      return provider;
    }
  }
  return NULL;
}
