// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/hal_actual.h"

#include <string.h>

#include "loom/tools/iree-benchmark-loom/diagnostics.h"
#include "loom/tools/iree-benchmark-loom/module_query.h"

static iree_status_t iree_benchmark_loom_hal_actual_observe_result(
    void* user_data, const loomc_result_t* result) {
  return iree_benchmark_loom_diagnostic_capture_loomc_result(
      (iree_benchmark_loom_diagnostic_capture_t*)user_data, result);
}

void iree_benchmark_loom_hal_actual_provider_initialize(
    iree_benchmark_loom_hal_context_t* context,
    const iree_benchmark_loom_hal_compilation_options_t* compilation_options,
    const loom_testbench_invocation_plan_t* kernel_launch,
    iree_string_view_t artifact_path_suffix,
    iree_benchmark_loom_hal_actual_provider_t* out_provider) {
  *out_provider = (iree_benchmark_loom_hal_actual_provider_t){
      .context = context,
      .artifact_path_suffix = artifact_path_suffix,
  };
  iree_allocator_t host_allocator = context->execution.host_allocator;
  iree_benchmark_loom_diagnostic_capture_initialize(host_allocator,
                                                    &out_provider->diagnostics);
  loomc_emit_artifact_flags_t artifact_flags = 0;
  if (context->artifact_bundle != NULL && context->artifact_bundle->enabled &&
      context->artifact_bundle->policy >=
          IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_DEBUG) {
    artifact_flags |= LOOMC_EMIT_ARTIFACT_FLAG_TARGET_LISTING;
  }
  loom_run_hal_testbench_actual_provider_options_t provider_options = {
      .context = &context->execution,
      .compilation = compilation_options->compilation,
      .module = compilation_options->compilation->module,
      .native_module = compilation_options->native_module,
      .pass_program = compilation_options->pass_program,
      .requested_target_profile = compilation_options->requested_target_profile,
      .sanitizer = compilation_options->sanitizer,
      .invocation = kernel_launch,
      .result_callback =
          {
              .fn = iree_benchmark_loom_hal_actual_observe_result,
              .user_data = &out_provider->diagnostics,
          },
      .compile_report = compilation_options->compile_report,
      .artifact_manifest = compilation_options->artifact_manifest,
      .emit_artifact_flags = artifact_flags,
  };
  loom_run_hal_testbench_actual_provider_initialize(&provider_options,
                                                    &out_provider->execution);
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
    iree_benchmark_loom_hal_context_t* context,
    const iree_benchmark_loom_hal_compilation_options_t* compilation_options,
    const loom_testbench_case_plan_t* case_plan,
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
          compilation_options->native_module, invocation->callee_ref,
          &artifact_path_suffix);
    }
    if (iree_status_is_ok(status)) {
      iree_benchmark_loom_hal_actual_provider_initialize(
          context, compilation_options, invocation, artifact_path_suffix,
          &out_sequence->providers[provider_index]);
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
  out_result->compile_report = provider->execution.artifacts.compile_report;
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
