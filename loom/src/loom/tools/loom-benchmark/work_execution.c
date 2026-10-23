// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-benchmark/work_execution.h"

#include <string.h>

#include "loom/tools/loom-benchmark/case_execution.h"
#include "loom/tools/loom-benchmark/dispatch_benchmark.h"
#include "loom/tools/loom-benchmark/hal_setup.h"
#include "loom/tools/loom-benchmark/scenario_execution.h"

static iree_status_t loom_benchmark_run_dispatch_sample_work_item(
    const loom_benchmark_work_plan_execution_options_t* options,
    const loom_benchmark_hal_setup_options_t* hal_options,
    const loom_benchmark_work_item_t* work_item,
    loom_benchmark_hal_compile_context_t* compile_context,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  const loom_benchmark_work_plan_t* work_plan = options->work_plan;
  const loom_benchmark_selected_benchmark_t* selection =
      &work_plan
           ->selected_benchmarks[work_item->representative_selection_index];
  const loom_testbench_benchmark_plan_t* benchmark_plan =
      selection->benchmark_plan;
  const loom_benchmark_benchmark_policy_t* policy = &selection->policy;

  loom_benchmark_hal_work_item_state_t work_item_state = {0};
  iree_status_t status = loom_benchmark_prepare_hal_work_item(
      hal_options, work_item, compile_context, inout_correctness_sample_count,
      inout_correctness_failed_sample_count, inout_failed_benchmark_count,
      &work_item_state);
  if (iree_status_is_ok(status) && work_item_state.runnable) {
    if (compile_context->uses_sequence) {
      loom_benchmark_benchmark_result_t benchmark_result = {0};
      status = loom_benchmark_run_hal_sequence_benchmark_sample(
          options->run, &selection->identity, options->module_plan,
          benchmark_plan, policy, options->benchmark_options,
          options->hal_context, &compile_context->hal_sequence,
          &compile_context->benchmark_execution_options,
          options->execution_arena, work_item->begin_benchmark_sample,
          options->host_allocator, &benchmark_result);
      if (iree_status_is_ok(status)) {
        benchmark_result.compile_item = compile_context->compile_item;
        benchmark_result.launch_evidence = &work_item_state.launch_evidence;
        status = loom_benchmark_emit_work_item_result_aliases(
            options->run, options->module_plan, work_plan, work_item,
            &benchmark_result, work_item_state.correctness_sample_count,
            work_item_state.correctness_failed_sample_count,
            options->event_sink, inout_failed_benchmark_count);
      }
    } else {
      loom_benchmark_benchmark_result_t benchmark_result = {0};
      status = loom_benchmark_run_hal_benchmark_sample(
          options->run, &selection->identity, options->module_plan,
          benchmark_plan, policy, options->benchmark_options,
          &compile_context->hal_provider,
          &compile_context->benchmark_execution_options,
          options->execution_arena, work_item->begin_benchmark_sample,
          options->host_allocator, &benchmark_result);
      if (iree_status_is_ok(status)) {
        benchmark_result.compile_item = compile_context->compile_item;
        benchmark_result.launch_evidence = &work_item_state.launch_evidence;
        status = loom_benchmark_emit_work_item_result_aliases(
            options->run, options->module_plan, work_plan, work_item,
            &benchmark_result, work_item_state.correctness_sample_count,
            work_item_state.correctness_failed_sample_count,
            options->event_sink, inout_failed_benchmark_count);
      }
    }
  }
  loom_benchmark_hal_work_item_state_deinitialize(&work_item_state);
  return status;
}

static iree_status_t loom_benchmark_run_hal_case_end_to_end_work_item(
    const loom_benchmark_work_plan_execution_options_t* options,
    const loom_benchmark_hal_setup_options_t* hal_options,
    const loom_benchmark_work_item_t* work_item,
    loom_benchmark_hal_compile_context_t* compile_context,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  loom_benchmark_hal_work_item_state_t work_item_state = {0};
  iree_status_t status = loom_benchmark_prepare_hal_work_item(
      hal_options, work_item, compile_context, inout_correctness_sample_count,
      inout_correctness_failed_sample_count, inout_failed_benchmark_count,
      &work_item_state);
  if (iree_status_is_ok(status) && work_item_state.runnable) {
    loom_benchmark_benchmark_result_t benchmark_result = {0};
    status = loom_benchmark_measure_case_end_to_end_work_item(
        options->module_plan, options->work_plan, work_item,
        &compile_context->execution_options, options->execution_arena,
        options->host_allocator, &benchmark_result);
    if (iree_status_is_ok(status)) {
      benchmark_result.launch_evidence = &work_item_state.launch_evidence;
      loom_benchmark_hal_compile_context_set_result_artifacts(
          compile_context, &benchmark_result);
      status = loom_benchmark_emit_work_item_result_aliases(
          options->run, options->module_plan, options->work_plan, work_item,
          &benchmark_result, work_item_state.correctness_sample_count,
          work_item_state.correctness_failed_sample_count, options->event_sink,
          inout_failed_benchmark_count);
    }
  }
  loom_benchmark_hal_work_item_state_deinitialize(&work_item_state);
  return status;
}

iree_status_t loom_benchmark_run_work_plan(
    const loom_benchmark_work_plan_execution_options_t* options,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  const loom_benchmark_work_plan_t* work_plan = options->work_plan;
  loom_benchmark_hal_compile_context_t* hal_compile_contexts = NULL;
  loom_benchmark_scenario_execution_t* scenario_execution = NULL;
  iree_status_t status = iree_ok_status();
  if (work_plan->hal_compile_item_count != 0) {
    status = iree_allocator_malloc_array(
        options->host_allocator, work_plan->hal_compile_item_count,
        sizeof(*hal_compile_contexts), (void**)&hal_compile_contexts);
    if (iree_status_is_ok(status)) {
      memset(hal_compile_contexts, 0,
             work_plan->hal_compile_item_count * sizeof(*hal_compile_contexts));
    }
  }
  const loom_benchmark_hal_setup_options_t hal_options = {
      .run = options->run,
      .module_plan = options->module_plan,
      .work_plan = options->work_plan,
      .benchmark_options = options->benchmark_options,
      .hal_context = options->hal_context,
      .compilation = options->compilation,
      .case_execution_options = options->case_execution_options,
      .execution_arena = options->execution_arena,
      .host_allocator = options->host_allocator,
      .event_sink = options->event_sink,
  };

  for (iree_host_size_t work_item_index = 0;
       iree_status_is_ok(status) &&
       work_item_index < work_plan->work_item_count;
       ++work_item_index) {
    const loom_benchmark_work_item_t* work_item =
        &work_plan->work_items[work_item_index];
    const loom_benchmark_selected_benchmark_t* selection =
        &work_plan
             ->selected_benchmarks[work_item->representative_selection_index];
    if (selection->skip_reason != NULL) {
      const loom_benchmark_benchmark_result_t benchmark_result = {
          .state = IREE_SV("skipped"),
          .skip_reason = selection->skip_reason,
      };
      status = loom_benchmark_emit_work_item_result_aliases(
          options->run, options->module_plan, work_plan, work_item,
          &benchmark_result, /*correctness_sample_count=*/0,
          /*correctness_failed_sample_count=*/0, options->event_sink,
          inout_failed_benchmark_count);
      continue;
    }
    switch ((loom_benchmark_work_item_kind_t)work_item->kind) {
      case LOOM_BENCHMARK_WORK_ITEM_CASE_END_TO_END: {
        if (work_item->hal_compile_item_index ==
            LOOM_BENCHMARK_WORK_PLAN_INDEX_INVALID) {
          status = loom_benchmark_run_case_end_to_end_work_item(
              options->run, options->module_plan, work_plan, work_item,
              options->case_execution_options, options->execution_arena,
              options->host_allocator, options->event_sink,
              inout_correctness_sample_count,
              inout_correctness_failed_sample_count,
              inout_failed_benchmark_count);
        } else {
          status = loom_benchmark_run_hal_case_end_to_end_work_item(
              options, &hal_options, work_item,
              &hal_compile_contexts[work_item->hal_compile_item_index],
              inout_correctness_sample_count,
              inout_correctness_failed_sample_count,
              inout_failed_benchmark_count);
        }
        break;
      }
      case LOOM_BENCHMARK_WORK_ITEM_DISPATCH_SAMPLE: {
        status = loom_benchmark_run_dispatch_sample_work_item(
            options, &hal_options, work_item,
            &hal_compile_contexts[work_item->hal_compile_item_index],
            inout_correctness_sample_count,
            inout_correctness_failed_sample_count,
            inout_failed_benchmark_count);
        break;
      }
      case LOOM_BENCHMARK_WORK_ITEM_SCENARIO_TRIAL: {
        if (scenario_execution == NULL) {
          status = loom_benchmark_scenario_execution_allocate(
              options, iree_string_view_empty(), &scenario_execution);
        }
        if (iree_status_is_ok(status)) {
          status = loom_benchmark_run_scenario_work_item(
              scenario_execution, work_item, inout_correctness_sample_count,
              inout_correctness_failed_sample_count,
              inout_failed_benchmark_count);
        }
        break;
      }
      case LOOM_BENCHMARK_WORK_ITEM_NONE:
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unsupported benchmark work item kind %d",
                                  (int)work_item->kind);
        break;
    }
  }

  if (hal_compile_contexts != NULL) {
    for (iree_host_size_t i = 0; i < work_plan->hal_compile_item_count; ++i) {
      loom_benchmark_hal_compile_context_deinitialize(&hal_compile_contexts[i]);
    }
  }
  loom_benchmark_scenario_execution_free(scenario_execution);
  iree_allocator_free(options->host_allocator, hal_compile_contexts);
  return status;
}
