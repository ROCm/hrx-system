// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-benchmark/hal_setup.h"

#include "iree/hal/api.h"
#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tools/loom-benchmark/case_execution.h"
#include "loom/tools/loom-benchmark/report.h"
#include "loom/tools/loom-benchmark/testbench.h"

static iree_status_t loom_benchmark_initialize_sequence_compile_context(
    const loom_benchmark_hal_setup_options_t* options,
    const loom_benchmark_selected_benchmark_t* selection,
    iree_host_size_t execution_sample_count,
    loom_benchmark_hal_compile_context_t* context) {
  const loom_testbench_case_plan_t* case_plan = selection->case_plan;
  context->uses_sequence = true;
  context->execution_options = *options->case_execution_options;

  iree_status_t status = loom_run_hal_testbench_context_ensure_runtime(
      &options->hal_context->execution);

  if (iree_status_is_ok(status)) {
    // Keep correctness values in the caller's CPU heap. Sequence execution
    // owns the upload/readback lifecycle; host-mappable GPU memory may be
    // uncached for the reference and comparison paths.
    context->execution_options.materializer.staging_pool =
        options->hal_context->execution.runtime.staging_pool;
    context->execution_options.materializer.transfer_queue =
        options->hal_context->execution.runtime.transfer_queue;
    status = loom_benchmark_hal_actual_sequence_initialize(
        options->hal_context, options->compilation, case_plan,
        execution_sample_count, &context->hal_sequence);
  }
  if (iree_status_is_ok(status)) {
    context->hal_sequence_initialized = true;
    context->execution_options.invocation.kernel_launch =
        loom_run_hal_testbench_actual_sequence_execution_provider(
            context->hal_sequence.execution);
    loom_benchmark_configure_reference_oracles(
        &options->hal_context->execution,
        context->execution_options.materializer.host_allocator,
        &context->reference_oracles, &context->execution_options);
    context->benchmark_execution_options = context->execution_options;
    context->benchmark_execution_options.materializer.target_pool =
        options->hal_context->execution.runtime.device_pool;
    status = loom_benchmark_hal_actual_sequence_compile(&context->hal_sequence);
  }
  if (iree_status_is_ok(status)) {
    context->rejected_sequence_provider =
        loom_benchmark_hal_actual_sequence_first_rejection(
            &context->hal_sequence);
  }
  return status;
}

static iree_status_t loom_benchmark_initialize_single_compile_context(
    const loom_benchmark_hal_setup_options_t* options,
    const loom_benchmark_selected_benchmark_t* selection,
    loom_benchmark_hal_compile_context_t* context) {
  const loom_testbench_invocation_plan_t* kernel_launch = NULL;
  const loom_testbench_case_plan_t* case_plan = selection->case_plan;
  context->execution_options = *options->case_execution_options;

  iree_status_t status =
      loom_run_hal_testbench_select_kernel_launch(case_plan, &kernel_launch);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_context_ensure_runtime(
        &options->hal_context->execution);
  }

  if (iree_status_is_ok(status)) {
    loom_benchmark_hal_actual_provider_initialize(
        options->hal_context, options->compilation, kernel_launch,
        iree_string_view_empty(), &context->hal_provider);
    context->hal_provider_initialized = true;
    // Preserve CPU-owned correctness values; timed dispatch inputs use the
    // device pool selected separately below.
    context->execution_options.materializer.staging_pool =
        options->hal_context->execution.runtime.staging_pool;
    context->execution_options.materializer.transfer_queue =
        options->hal_context->execution.runtime.transfer_queue;
    context->execution_options.invocation.kernel_launch =
        (loom_testbench_invocation_provider_t){
            .invoke = loom_run_hal_testbench_actual_invoke,
            .user_data = &context->hal_provider.execution,
        };
    loom_benchmark_configure_reference_oracles(
        &options->hal_context->execution,
        context->execution_options.materializer.host_allocator,
        &context->reference_oracles, &context->execution_options);
    context->benchmark_execution_options = context->execution_options;
    context->benchmark_execution_options.materializer.target_pool =
        options->hal_context->execution.runtime.device_pool;
    status = loom_benchmark_hal_actual_provider_compile(&context->hal_provider);
  }
  return status;
}

iree_status_t loom_benchmark_hal_compile_context_initialize(
    const loom_benchmark_hal_setup_options_t* options,
    const loom_benchmark_hal_compile_item_t* compile_item,
    loom_benchmark_hal_compile_context_t* context) {
  if (context->initialized) {
    return iree_ok_status();
  }
  context->compile_item = compile_item;
  const loom_benchmark_selected_benchmark_t* selection =
      &options->work_plan
           ->selected_benchmarks[compile_item->representative_selection_index];
  iree_status_t status = iree_ok_status();
  if (selection->case_plan->kernel_launch_count > 1) {
    status = loom_benchmark_initialize_sequence_compile_context(
        options, selection, compile_item->execution_sample_count, context);
  } else {
    status = loom_benchmark_initialize_single_compile_context(
        options, selection, context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_event_sink_emit_device(
        options->event_sink, options->run, options->hal_context);
  }
  loom_benchmark_hal_actual_provider_t* providers =
      context->uses_sequence ? context->hal_sequence.providers
                             : &context->hal_provider;
  const iree_host_size_t provider_count =
      context->uses_sequence ? context->hal_sequence.provider_count : 1;
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < provider_count;
       ++i) {
    if (providers[i].compile_entry_index != i) {
      continue;
    }
    status = loom_benchmark_write_compiled_artifacts(
        options->run, &selection->identity, &providers[i],
        options->host_allocator);
    if (iree_status_is_ok(status)) {
      status = loom_benchmark_write_compile_report_artifact(
          options->run, &selection->identity, selection->benchmark_plan,
          selection->case_plan, &providers[i], options->host_allocator);
    }
    if (iree_status_is_ok(status)) {
      status = loom_benchmark_event_sink_emit_compile(
          options->event_sink, options->run, &selection->identity, compile_item,
          selection->benchmark_plan, selection->case_plan, &providers[i]);
    }
  }
  if (iree_status_is_ok(status)) {
    context->initialized = true;
  } else {
    loom_benchmark_hal_compile_context_deinitialize(context);
  }
  return status;
}

void loom_benchmark_hal_compile_context_deinitialize(
    loom_benchmark_hal_compile_context_t* context) {
  if (context->hal_sequence_initialized) {
    loom_benchmark_hal_actual_sequence_deinitialize(&context->hal_sequence);
  }
  if (context->hal_provider_initialized) {
    loom_benchmark_hal_actual_provider_deinitialize(&context->hal_provider);
  }
  *context = (loom_benchmark_hal_compile_context_t){0};
}

void loom_benchmark_hal_compile_context_set_result_artifacts(
    const loom_benchmark_hal_compile_context_t* context,
    loom_benchmark_benchmark_result_t* result) {
  result->compile_item = context->compile_item;
  if (context->uses_sequence) {
    return;
  }
  const loom_benchmark_hal_actual_provider_t* provider = &context->hal_provider;
  result->compile_report = provider->execution.artifacts.compile_report;
  result->compile_report_artifact_path = provider->compile_report_artifact_path;
  result->artifact_manifest_path = provider->artifact_manifest_path;
  result->target_artifact_path = provider->target_artifact_path;
  result->target_listing_path = provider->target_listing_path;
  result->hal_executable_path = provider->hal_executable_path;
}

void loom_benchmark_hal_work_item_state_deinitialize(
    loom_benchmark_hal_work_item_state_t* state) {
  if (state == NULL) {
    return;
  }
  loom_benchmark_launch_evidence_deinitialize(&state->launch_evidence);
  *state = (loom_benchmark_hal_work_item_state_t){0};
}

typedef struct loom_benchmark_launch_capture_context_t {
  // Compiled providers whose resolved launch configurations are being copied.
  loom_benchmark_hal_compile_context_t* compile_context;
  // Work-item-owned destination for ordered launch records.
  loom_benchmark_launch_evidence_t* evidence;
  // Next preallocated launch record to populate.
  iree_host_size_t next_record_ordinal;
  // Next preallocated workload value to populate.
  iree_host_size_t next_workload_value_ordinal;
} loom_benchmark_launch_capture_context_t;

static void loom_benchmark_capture_case_sample_launches(
    void* user_data, loom_testbench_case_sample_t sample) {
  loom_benchmark_launch_capture_context_t* capture =
      (loom_benchmark_launch_capture_context_t*)user_data;
  loom_benchmark_hal_compile_context_t* compile_context =
      capture->compile_context;
  if (compile_context->uses_sequence) {
    for (iree_host_size_t i = 0;
         i < compile_context->hal_sequence.provider_count; ++i) {
      const loom_run_hal_testbench_actual_provider_t* provider =
          &compile_context->hal_sequence.providers[i].execution;
      loom_benchmark_launch_evidence_capture(
          provider, sample, i,
          compile_context->hal_sequence.providers[i].compile_entry_index,
          capture->next_record_ordinal++, capture->next_workload_value_ordinal,
          capture->evidence);
      capture->next_workload_value_ordinal +=
          provider->invocation->workload_count;
    }
  } else {
    const loom_run_hal_testbench_actual_provider_t* provider =
        &compile_context->hal_provider.execution;
    loom_benchmark_launch_evidence_capture(
        provider, sample,
        /*sequence_step_ordinal=*/0, /*compile_entry_index=*/0,
        capture->next_record_ordinal++, capture->next_workload_value_ordinal,
        capture->evidence);
    capture->next_workload_value_ordinal +=
        provider->invocation->workload_count;
  }
}

iree_status_t loom_benchmark_prepare_hal_work_item(
    const loom_benchmark_hal_setup_options_t* options,
    const loom_benchmark_work_item_t* work_item,
    loom_benchmark_hal_compile_context_t* compile_context,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count,
    loom_benchmark_hal_work_item_state_t* out_state) {
  *out_state = (loom_benchmark_hal_work_item_state_t){0};
  const loom_benchmark_work_plan_t* work_plan = options->work_plan;
  const loom_benchmark_hal_compile_item_t* compile_item =
      &work_plan->hal_compile_items[work_item->hal_compile_item_index];

  const loom_benchmark_selected_benchmark_t* selection =
      &work_plan
           ->selected_benchmarks[compile_item->representative_selection_index];
  if (selection->skip_reason != NULL) {
    loom_benchmark_benchmark_result_t benchmark_result = {
        .state = IREE_SV("skipped"),
        .skip_reason = selection->skip_reason,
    };
    return loom_benchmark_emit_work_item_result_aliases(
        options->run, options->module_plan, work_plan, work_item,
        &benchmark_result,
        /*correctness_sample_count=*/0,
        /*correctness_failed_sample_count=*/0, options->event_sink,
        inout_failed_benchmark_count);
  }

  IREE_RETURN_IF_ERROR(loom_benchmark_hal_compile_context_initialize(
      options, compile_item, compile_context));
  if (compile_context->uses_sequence) {
    const loom_benchmark_hal_actual_provider_t* rejected_provider =
        compile_context->rejected_sequence_provider;
    if (rejected_provider != NULL) {
      loom_benchmark_benchmark_result_t benchmark_result = {0};
      loom_benchmark_benchmark_result_set_compile_rejection(rejected_provider,
                                                            &benchmark_result);
      benchmark_result.compile_item = compile_item;
      return loom_benchmark_emit_work_item_result_aliases(
          options->run, options->module_plan, work_plan, work_item,
          &benchmark_result,
          /*correctness_sample_count=*/0,
          /*correctness_failed_sample_count=*/0, options->event_sink,
          inout_failed_benchmark_count);
    }
  } else if (compile_context->hal_provider.execution.compile_rejected) {
    loom_benchmark_benchmark_result_t benchmark_result = {0};
    loom_benchmark_benchmark_result_set_compile_rejection(
        &compile_context->hal_provider, &benchmark_result);
    benchmark_result.compile_item = compile_item;
    return loom_benchmark_emit_work_item_result_aliases(
        options->run, options->module_plan, work_plan, work_item,
        &benchmark_result,
        /*correctness_sample_count=*/0,
        /*correctness_failed_sample_count=*/0, options->event_sink,
        inout_failed_benchmark_count);
  }

  iree_host_size_t correctness_sample_count = 0;
  iree_host_size_t correctness_failed_sample_count = 0;
  const iree_host_size_t sample_count =
      work_item->end_benchmark_sample - work_item->begin_benchmark_sample;
  const iree_host_size_t launches_per_sample =
      compile_context->uses_sequence
          ? compile_context->hal_sequence.provider_count
          : 1;
  bool evidence_size_valid = true;
  iree_host_size_t workload_values_per_sample = 0;
  if (compile_context->uses_sequence) {
    for (iree_host_size_t i = 0;
         i < compile_context->hal_sequence.provider_count; ++i) {
      const iree_host_size_t workload_count =
          compile_context->hal_sequence.providers[i]
              .execution.invocation->workload_count;
      if (!iree_host_size_checked_add(workload_values_per_sample,
                                      workload_count,
                                      &workload_values_per_sample)) {
        evidence_size_valid = false;
        break;
      }
    }
  } else {
    workload_values_per_sample =
        compile_context->hal_provider.execution.invocation->workload_count;
  }
  iree_host_size_t launch_record_count = 0;
  iree_host_size_t workload_value_count = 0;
  iree_status_t status = iree_ok_status();
  if (!evidence_size_valid ||
      !iree_host_size_checked_mul(sample_count, launches_per_sample,
                                  &launch_record_count) ||
      !iree_host_size_checked_mul(sample_count, workload_values_per_sample,
                                  &workload_value_count)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "benchmark launch evidence size overflowed host "
                              "size limits");
  }
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_launch_evidence_initialize(
        selection->case_plan, launch_record_count, workload_value_count,
        options->host_allocator, &out_state->launch_evidence);
  }
  loom_benchmark_launch_capture_context_t launch_capture = {
      .compile_context = compile_context,
      .evidence = &out_state->launch_evidence,
  };
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_run_work_item_correctness_range(
        options->run, options->module_plan, work_plan, work_item,
        &compile_context->execution_options, options->execution_arena,
        (loom_benchmark_case_sample_observer_t){
            .fn = loom_benchmark_capture_case_sample_launches,
            .user_data = &launch_capture,
        },
        options->event_sink, &correctness_sample_count,
        &correctness_failed_sample_count);
  }
  if (iree_status_is_ok(status)) {
    IREE_ASSERT(launch_capture.next_record_ordinal == launch_record_count);
    IREE_ASSERT(launch_capture.next_workload_value_ordinal ==
                workload_value_count);
  }
  if (iree_status_is_ok(status)) {
    *inout_correctness_sample_count += correctness_sample_count;
    *inout_correctness_failed_sample_count += correctness_failed_sample_count;
  }
  if (iree_status_is_ok(status) && correctness_failed_sample_count != 0) {
    loom_benchmark_benchmark_result_t benchmark_result = {
        .executed = false,
        .passed = false,
        .samples_per_iteration = correctness_sample_count,
        .failed_sample_count = correctness_failed_sample_count,
    };
    benchmark_result.launch_evidence = &out_state->launch_evidence;
    loom_benchmark_hal_compile_context_set_result_artifacts(compile_context,
                                                            &benchmark_result);
    status = loom_benchmark_emit_work_item_result_aliases(
        options->run, options->module_plan, work_plan, work_item,
        &benchmark_result, correctness_sample_count,
        correctness_failed_sample_count, options->event_sink,
        inout_failed_benchmark_count);
    loom_benchmark_hal_work_item_state_deinitialize(out_state);
  }
  if (iree_status_is_ok(status) && correctness_failed_sample_count == 0) {
    out_state->runnable = true;
    out_state->correctness_sample_count = correctness_sample_count;
    out_state->correctness_failed_sample_count =
        correctness_failed_sample_count;
  }
  if (!iree_status_is_ok(status)) {
    loom_benchmark_hal_work_item_state_deinitialize(out_state);
  }
  return status;
}
