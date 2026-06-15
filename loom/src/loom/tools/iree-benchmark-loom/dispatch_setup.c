// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/dispatch_setup.h"

#include "iree/hal/api.h"
#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tools/iree-benchmark-loom/case_execution.h"
#include "loom/tools/iree-benchmark-loom/report.h"
#include "loom/tools/iree-benchmark-loom/testbench.h"

static iree_status_t iree_benchmark_loom_initialize_sequence_compile_context(
    const iree_benchmark_loom_dispatch_setup_options_t* options,
    const iree_benchmark_loom_dispatch_compile_item_t* compile_item,
    const iree_benchmark_loom_selected_benchmark_t* selection,
    iree_benchmark_loom_dispatch_compile_context_t* context) {
  const loom_testbench_case_plan_t* case_plan = selection->case_plan;
  context->uses_sequence = true;
  context->execution_options = *options->case_execution_options;
  context->benchmark_materializer =
      options->case_execution_options->materializer;

  iree_status_t status = loom_run_hal_testbench_context_ensure_runtime(
      &options->hal_context->execution);
  loom_testbench_requirement_result_t requirement_result = {0};
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_evaluate_case_requirements(
        options->hal_context->configuration, &options->hal_context->execution,
        options->module_plan, case_plan, &requirement_result);
  }
  if (iree_status_is_ok(status) && requirement_result.skipped) {
    context->skipped = true;
    return iree_ok_status();
  }

  if (iree_status_is_ok(status)) {
    context->execution_options.materializer.device =
        options->hal_context->execution.runtime.device;
    context->execution_options.materializer.device_allocator =
        iree_hal_device_allocator(
            options->hal_context->execution.runtime.device);
    context->execution_options.materializer.buffer_params =
        loom_run_hal_testbench_host_visible_buffer_params();
    const loom_run_hal_testbench_actual_sequence_options_t sequence_options = {
        .context = &options->hal_context->execution,
        .session = options->session,
        .target_environment =
            options->hal_context->configuration->target_environment,
        .filename = options->filename,
        .source = options->source,
        .pipeline = options->benchmark_options->pipeline,
        .test_module = options->module_plan->module,
        .case_plan = case_plan,
        .sample_constant_case_plan = case_plan,
        .sample_constant_ordinal = compile_item->case_sample_ordinal,
        .has_sample_constant_ordinal = compile_item->has_case_sample_ordinal,
    };
    status = loom_run_hal_testbench_actual_sequence_initialize(
        &sequence_options, &context->hal_sequence);
  }
  if (iree_status_is_ok(status)) {
    context->hal_sequence_initialized = true;
    context->execution_options.invocation.invoke_actual =
        (loom_testbench_invocation_callback_t){
            .fn = loom_run_hal_testbench_actual_sequence_invoke,
            .user_data = &context->hal_sequence,
        };
    iree_benchmark_loom_configure_reference_oracles(
        &options->hal_context->execution,
        context->execution_options.materializer.host_allocator,
        &context->reference_oracles, &context->execution_options);
    context->benchmark_materializer = context->execution_options.materializer;
    context->benchmark_materializer.buffer_params =
        (iree_hal_buffer_params_t){0};
    status =
        iree_benchmark_loom_hal_actual_sequence_compile(&context->hal_sequence);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_event_sink_emit_device(
        options->event_sink, options->run, options->hal_context);
  }
  if (iree_status_is_ok(status)) {
    context->rejected_sequence_provider =
        iree_benchmark_loom_hal_actual_sequence_first_rejection(
            &context->hal_sequence);
  }
  return status;
}

static iree_status_t iree_benchmark_loom_initialize_single_compile_context(
    const iree_benchmark_loom_dispatch_setup_options_t* options,
    const iree_benchmark_loom_dispatch_compile_item_t* compile_item,
    const iree_benchmark_loom_selected_benchmark_t* selection,
    iree_benchmark_loom_dispatch_compile_context_t* context) {
  const loom_testbench_invocation_plan_t* actual_invocation = NULL;
  const loom_testbench_benchmark_plan_t* benchmark_plan =
      selection->benchmark_plan;
  const loom_testbench_case_plan_t* case_plan = selection->case_plan;
  context->execution_options = *options->case_execution_options;
  context->benchmark_materializer =
      options->case_execution_options->materializer;

  iree_status_t status = loom_run_hal_testbench_select_actual_invocation(
      case_plan, &actual_invocation);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_context_ensure_runtime(
        &options->hal_context->execution);
  }
  loom_testbench_requirement_result_t requirement_result = {0};
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_evaluate_case_requirements(
        options->hal_context->configuration, &options->hal_context->execution,
        options->module_plan, case_plan, &requirement_result);
  }
  if (iree_status_is_ok(status) && requirement_result.skipped) {
    context->skipped = true;
    return iree_ok_status();
  }

  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_hal_actual_provider_initialize(
        options->hal_context, options->session, options->filename,
        options->source, options->benchmark_options->pipeline,
        options->module_plan->module, actual_invocation,
        compile_item->sample_compilation, case_plan,
        compile_item->case_sample_ordinal,
        compile_item->has_case_sample_ordinal, options->compile_report_options,
        options->artifact_manifest_options, &context->hal_provider);
  }
  if (iree_status_is_ok(status)) {
    context->hal_provider_initialized = true;
    context->execution_options.materializer.device =
        options->hal_context->execution.runtime.device;
    context->execution_options.materializer.device_allocator =
        iree_hal_device_allocator(
            options->hal_context->execution.runtime.device);
    context->execution_options.materializer.buffer_params =
        loom_run_hal_testbench_host_visible_buffer_params();
    context->execution_options.invocation.invoke_actual =
        (loom_testbench_invocation_callback_t){
            .fn = loom_run_hal_testbench_actual_invoke,
            .user_data = &context->hal_provider.execution,
        };
    iree_benchmark_loom_configure_reference_oracles(
        &options->hal_context->execution,
        context->execution_options.materializer.host_allocator,
        &context->reference_oracles, &context->execution_options);
    context->benchmark_materializer = context->execution_options.materializer;
    context->benchmark_materializer.buffer_params =
        (iree_hal_buffer_params_t){0};
    status =
        iree_benchmark_loom_hal_actual_provider_compile(&context->hal_provider);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_write_compiled_artifacts(
        options->run, &selection->identity, &context->hal_provider,
        options->host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_write_compile_report_artifact(
        options->run, &selection->identity, benchmark_plan, case_plan,
        &context->hal_provider, options->host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_event_sink_emit_device(
        options->event_sink, options->run, options->hal_context);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_event_sink_emit_compile(
        options->event_sink, options->run, &selection->identity, benchmark_plan,
        case_plan, &context->hal_provider);
  }
  return status;
}

iree_status_t iree_benchmark_loom_dispatch_compile_context_initialize(
    const iree_benchmark_loom_dispatch_setup_options_t* options,
    const iree_benchmark_loom_dispatch_compile_item_t* compile_item,
    iree_benchmark_loom_dispatch_compile_context_t* context) {
  if (context->initialized) {
    return iree_ok_status();
  }
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &options->work_plan
           ->selected_benchmarks[compile_item->representative_selection_index];
  iree_host_size_t actual_invocation_count = 0;
  iree_status_t status = loom_run_hal_testbench_count_actual_invocations(
      selection->case_plan, &actual_invocation_count);
  if (iree_status_is_ok(status) && actual_invocation_count > 1) {
    status = iree_benchmark_loom_initialize_sequence_compile_context(
        options, compile_item, selection, context);
  } else if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_initialize_single_compile_context(
        options, compile_item, selection, context);
  }
  if (iree_status_is_ok(status)) {
    context->initialized = true;
  } else {
    iree_benchmark_loom_dispatch_compile_context_deinitialize(context);
  }
  return status;
}

void iree_benchmark_loom_dispatch_compile_context_deinitialize(
    iree_benchmark_loom_dispatch_compile_context_t* context) {
  if (context->hal_sequence_initialized) {
    loom_run_hal_testbench_actual_sequence_deinitialize(&context->hal_sequence);
  }
  if (context->hal_provider_initialized) {
    iree_benchmark_loom_hal_actual_provider_deinitialize(
        &context->hal_provider);
  }
  *context = (iree_benchmark_loom_dispatch_compile_context_t){0};
}

iree_status_t iree_benchmark_loom_prepare_dispatch_work_item(
    const iree_benchmark_loom_dispatch_setup_options_t* options,
    const iree_benchmark_loom_work_item_t* work_item,
    iree_benchmark_loom_dispatch_compile_context_t* compile_context,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count,
    iree_benchmark_loom_dispatch_work_item_state_t* out_state) {
  *out_state = (iree_benchmark_loom_dispatch_work_item_state_t){0};
  const iree_benchmark_loom_work_plan_t* work_plan = options->work_plan;
  const iree_benchmark_loom_dispatch_compile_item_t* compile_item =
      &work_plan
           ->dispatch_compile_items[work_item->dispatch_compile_item_index];

  IREE_RETURN_IF_ERROR(iree_benchmark_loom_dispatch_compile_context_initialize(
      options, compile_item, compile_context));
  if (compile_context->skipped) {
    iree_benchmark_loom_benchmark_result_t benchmark_result = {
        .status = IREE_SV("skipped"),
        .sample_compilation = work_item->sample_compilation,
    };
    return iree_benchmark_loom_emit_work_item_result_aliases(
        options->run, options->module_plan, work_plan, work_item,
        &benchmark_result,
        /*correctness_sample_count=*/0,
        /*correctness_failed_sample_count=*/0, options->event_sink,
        inout_failed_benchmark_count);
  }

  if (compile_context->uses_sequence) {
    const loom_run_hal_testbench_actual_provider_t* rejected_provider =
        compile_context->rejected_sequence_provider;
    if (rejected_provider != NULL) {
      iree_benchmark_loom_benchmark_result_t benchmark_result = {0};
      iree_benchmark_loom_benchmark_result_set_sequence_compile_rejection(
          rejected_provider, work_item->sample_compilation, &benchmark_result);
      benchmark_result.has_sample_ordinal = true;
      benchmark_result.sample_ordinal = work_item->case_sample_ordinal;
      benchmark_result.samples_per_iteration = 1;
      return iree_benchmark_loom_emit_work_item_result_aliases(
          options->run, options->module_plan, work_plan, work_item,
          &benchmark_result,
          /*correctness_sample_count=*/0,
          /*correctness_failed_sample_count=*/0, options->event_sink,
          inout_failed_benchmark_count);
    }
  } else if (compile_context->hal_provider.execution.compile_rejected) {
    iree_benchmark_loom_benchmark_result_t benchmark_result = {0};
    iree_benchmark_loom_benchmark_result_set_compile_rejection(
        &compile_context->hal_provider, &benchmark_result);
    benchmark_result.has_sample_ordinal = true;
    benchmark_result.sample_ordinal = work_item->case_sample_ordinal;
    benchmark_result.samples_per_iteration = 1;
    return iree_benchmark_loom_emit_work_item_result_aliases(
        options->run, options->module_plan, work_plan, work_item,
        &benchmark_result,
        /*correctness_sample_count=*/0,
        /*correctness_failed_sample_count=*/0, options->event_sink,
        inout_failed_benchmark_count);
  }

  iree_host_size_t correctness_sample_count = 0;
  iree_host_size_t correctness_failed_sample_count = 0;
  iree_status_t status = iree_benchmark_loom_run_work_item_correctness_range(
      options->run, options->module_plan, work_plan, work_item,
      &compile_context->execution_options, options->execution_arena,
      options->event_sink, &correctness_sample_count,
      &correctness_failed_sample_count);
  if (iree_status_is_ok(status)) {
    *inout_correctness_sample_count += correctness_sample_count;
    *inout_correctness_failed_sample_count += correctness_failed_sample_count;
  }
  if (iree_status_is_ok(status) && correctness_failed_sample_count != 0) {
    iree_benchmark_loom_benchmark_result_t benchmark_result = {
        .executed = false,
        .passed = false,
        .sample_compilation = work_item->sample_compilation,
        .has_sample_ordinal = true,
        .sample_ordinal = work_item->case_sample_ordinal,
        .samples_per_iteration = correctness_sample_count,
        .failed_sample_count = correctness_failed_sample_count,
    };
    if (!compile_context->uses_sequence) {
      benchmark_result.compile_report_artifact_path =
          compile_context->hal_provider.compile_report_artifact_path;
      benchmark_result.artifact_manifest_path =
          compile_context->hal_provider.artifact_manifest_path;
      benchmark_result.target_artifact_path =
          compile_context->hal_provider.target_artifact_path;
      benchmark_result.target_listing_path =
          compile_context->hal_provider.target_listing_path;
      benchmark_result.hal_executable_path =
          compile_context->hal_provider.hal_executable_path;
    }
    status = iree_benchmark_loom_emit_work_item_result_aliases(
        options->run, options->module_plan, work_plan, work_item,
        &benchmark_result, correctness_sample_count,
        correctness_failed_sample_count, options->event_sink,
        inout_failed_benchmark_count);
  }
  if (iree_status_is_ok(status) && correctness_failed_sample_count == 0) {
    *out_state = (iree_benchmark_loom_dispatch_work_item_state_t){
        .runnable = true,
        .correctness_sample_count = correctness_sample_count,
        .correctness_failed_sample_count = correctness_failed_sample_count,
    };
  }
  return status;
}
