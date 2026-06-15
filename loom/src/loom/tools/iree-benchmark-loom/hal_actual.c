// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/hal_actual.h"

#include <string.h>

#include "loom/tooling/execution/compile_options.h"
#include "loom/tools/iree-benchmark-loom/diagnostics.h"

iree_status_t iree_benchmark_loom_hal_actual_provider_initialize(
    iree_benchmark_loom_hal_context_t* context, loom_run_session_t* session,
    iree_string_view_t filename, iree_string_view_t source,
    iree_string_view_t pipeline, const loom_module_t* test_module,
    const loom_testbench_invocation_plan_t* actual_invocation,
    iree_string_view_t sample_compilation,
    const loom_testbench_case_plan_t* sample_constant_case_plan,
    iree_host_size_t sample_constant_ordinal, bool has_sample_constant_ordinal,
    const loom_run_compile_report_capture_options_t* compile_report_options,
    const loom_run_candidate_artifact_manifest_options_t*
        artifact_manifest_options,
    iree_benchmark_loom_hal_actual_provider_t* out_provider) {
  *out_provider = (iree_benchmark_loom_hal_actual_provider_t){
      .context = context,
      .sample_compilation = sample_compilation,
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
        .filename = filename,
        .source = source,
        .pipeline = pipeline,
        .test_module = test_module,
        .actual_invocation = actual_invocation,
        .sample_constant_case_plan = sample_constant_case_plan,
        .sample_constant_ordinal = sample_constant_ordinal,
        .has_sample_constant_ordinal = has_sample_constant_ordinal,
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

iree_status_t iree_benchmark_loom_hal_actual_provider_compile(
    iree_benchmark_loom_hal_actual_provider_t* provider) {
  return loom_run_hal_testbench_actual_provider_compile(&provider->execution);
}

void iree_benchmark_loom_benchmark_result_set_compile_rejection(
    const iree_benchmark_loom_hal_actual_provider_t* provider,
    iree_benchmark_loom_benchmark_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));
  out_result->status = IREE_SV("compile_failed");
  out_result->has_failure = true;
  out_result->failure_stage = provider->execution.compile_failure_stage;
  out_result->failure_kind = provider->execution.compile_failure_kind;
  out_result->diagnostic_error_count = provider->diagnostics.error_count;
  out_result->diagnostic_warning_count = provider->diagnostics.warning_count;
  out_result->diagnostic_remark_count = provider->diagnostics.remark_count;
  out_result->diagnostic_json =
      iree_string_builder_view(&provider->diagnostics.output);
  out_result->sample_compilation = provider->sample_compilation;
  if (provider->execution.has_sample_constant_ordinal) {
    out_result->has_sample_ordinal = true;
    out_result->sample_ordinal = provider->execution.sample_constant_ordinal;
    out_result->samples_per_iteration = 1;
  }
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
    loom_run_hal_testbench_actual_sequence_t* sequence) {
  for (iree_host_size_t i = 0; i < sequence->provider_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_run_hal_testbench_actual_provider_compile(
        &sequence->providers[i]));
  }
  return iree_ok_status();
}

const loom_run_hal_testbench_actual_provider_t*
iree_benchmark_loom_hal_actual_sequence_first_rejection(
    const loom_run_hal_testbench_actual_sequence_t* sequence) {
  for (iree_host_size_t i = 0; i < sequence->provider_count; ++i) {
    const loom_run_hal_testbench_actual_provider_t* provider =
        &sequence->providers[i];
    if (provider->compile_rejected) {
      return provider;
    }
  }
  return NULL;
}

void iree_benchmark_loom_benchmark_result_set_sequence_compile_rejection(
    const loom_run_hal_testbench_actual_provider_t* provider,
    iree_string_view_t sample_compilation,
    iree_benchmark_loom_benchmark_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));
  out_result->status = IREE_SV("compile_failed");
  out_result->has_failure = true;
  out_result->failure_stage = provider->compile_failure_stage;
  out_result->failure_kind = provider->compile_failure_kind;
  out_result->diagnostic_error_count = provider->diagnostic_error_count;
  out_result->diagnostic_warning_count = provider->diagnostic_warning_count;
  out_result->diagnostic_remark_count = provider->diagnostic_remark_count;
  out_result->sample_compilation = sample_compilation;
}
