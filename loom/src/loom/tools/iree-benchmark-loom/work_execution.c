// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/work_execution.h"

#include <inttypes.h>
#include <string.h>

#include "loom/tools/iree-benchmark-loom/case_execution.h"
#include "loom/tools/iree-benchmark-loom/dispatch_benchmark.h"
#include "loom/tools/iree-benchmark-loom/dispatch_setup.h"

static iree_status_t iree_benchmark_loom_run_dispatch_sample_work_item(
    const iree_benchmark_loom_work_plan_execution_options_t* options,
    const iree_benchmark_loom_dispatch_setup_options_t* dispatch_options,
    const iree_benchmark_loom_work_item_t* work_item,
    iree_benchmark_loom_dispatch_compile_context_t* compile_context,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  const iree_benchmark_loom_work_plan_t* work_plan = options->work_plan;
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &work_plan
           ->selected_benchmarks[work_item->representative_selection_index];
  const loom_testbench_benchmark_plan_t* benchmark_plan =
      selection->benchmark_plan;
  const loom_testbench_case_plan_t* case_plan = selection->case_plan;
  const iree_benchmark_loom_benchmark_policy_t* policy = &selection->policy;

  iree_benchmark_loom_dispatch_work_item_state_t work_item_state = {0};
  iree_status_t status = iree_benchmark_loom_prepare_dispatch_work_item(
      dispatch_options, work_item, compile_context,
      inout_correctness_sample_count, inout_correctness_failed_sample_count,
      inout_failed_benchmark_count, &work_item_state);
  if (iree_status_is_ok(status) && work_item_state.runnable) {
    if (compile_context->uses_sequence) {
      iree_benchmark_loom_benchmark_result_t benchmark_result = {0};
      status = iree_benchmark_loom_run_hal_sequence_benchmark_sample(
          options->run, &selection->identity, options->module_plan,
          benchmark_plan, case_plan, policy, options->benchmark_options,
          options->hal_context, &compile_context->hal_sequence,
          &compile_context->benchmark_materializer,
          work_item->sample_compilation, work_item->begin_benchmark_sample,
          options->host_allocator, &benchmark_result);
      if (iree_status_is_ok(status)) {
        status = iree_benchmark_loom_emit_work_item_result_aliases(
            options->run, options->module_plan, work_plan, work_item,
            &benchmark_result, work_item_state.correctness_sample_count,
            work_item_state.correctness_failed_sample_count,
            options->event_sink, inout_failed_benchmark_count);
      }
    } else {
      iree_benchmark_loom_benchmark_result_t benchmark_result = {0};
      status = iree_benchmark_loom_run_hal_benchmark_sample(
          options->run, &selection->identity, options->module_plan,
          benchmark_plan, case_plan, policy, options->benchmark_options,
          &compile_context->hal_provider,
          &compile_context->benchmark_materializer,
          work_item->begin_benchmark_sample, options->host_allocator,
          &benchmark_result);
      if (iree_status_is_ok(status)) {
        status = iree_benchmark_loom_emit_work_item_result_aliases(
            options->run, options->module_plan, work_plan, work_item,
            &benchmark_result, work_item_state.correctness_sample_count,
            work_item_state.correctness_failed_sample_count,
            options->event_sink, inout_failed_benchmark_count);
      }
    }
  }
  return status;
}

iree_status_t iree_benchmark_loom_run_work_plan(
    const iree_benchmark_loom_work_plan_execution_options_t* options,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  const iree_benchmark_loom_work_plan_t* work_plan = options->work_plan;
  iree_benchmark_loom_dispatch_compile_context_t* dispatch_compile_contexts =
      NULL;
  iree_status_t status = iree_ok_status();
  if (work_plan->dispatch_compile_item_count != 0) {
    status = iree_allocator_malloc_array(
        options->host_allocator, work_plan->dispatch_compile_item_count,
        sizeof(*dispatch_compile_contexts), (void**)&dispatch_compile_contexts);
    if (iree_status_is_ok(status)) {
      memset(dispatch_compile_contexts, 0,
             work_plan->dispatch_compile_item_count *
                 sizeof(*dispatch_compile_contexts));
    }
  }
  const iree_benchmark_loom_dispatch_setup_options_t dispatch_options = {
      .run = options->run,
      .module_plan = options->module_plan,
      .work_plan = options->work_plan,
      .benchmark_options = options->benchmark_options,
      .hal_context = options->hal_context,
      .session = options->session,
      .filename = options->filename,
      .source = options->source,
      .compile_report_options = options->compile_report_options,
      .artifact_manifest_options = options->artifact_manifest_options,
      .case_execution_options = options->case_execution_options,
      .execution_arena = options->execution_arena,
      .host_allocator = options->host_allocator,
      .event_sink = options->event_sink,
  };

  for (iree_host_size_t work_item_index = 0;
       iree_status_is_ok(status) &&
       work_item_index < work_plan->work_item_count;
       ++work_item_index) {
    const iree_benchmark_loom_work_item_t* work_item =
        &work_plan->work_items[work_item_index];
    switch ((iree_benchmark_loom_work_item_kind_t)work_item->kind) {
      case IREE_BENCHMARK_LOOM_WORK_ITEM_CASE_END_TO_END: {
        status = iree_benchmark_loom_run_case_end_to_end_work_item(
            options->run, options->module_plan, work_plan, work_item,
            options->case_execution_options, options->execution_arena,
            options->host_allocator, options->event_sink,
            inout_correctness_sample_count,
            inout_correctness_failed_sample_count,
            inout_failed_benchmark_count);
        break;
      }
      case IREE_BENCHMARK_LOOM_WORK_ITEM_DISPATCH_SAMPLE: {
        if (work_item->dispatch_compile_item_index >=
            work_plan->dispatch_compile_item_count) {
          status = iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "dispatch benchmark work item references compile item %" PRIhsz
              " but the work plan only has %" PRIhsz " compile items",
              work_item->dispatch_compile_item_index,
              work_plan->dispatch_compile_item_count);
          break;
        }
        status = iree_benchmark_loom_run_dispatch_sample_work_item(
            options, &dispatch_options, work_item,
            &dispatch_compile_contexts[work_item->dispatch_compile_item_index],
            inout_correctness_sample_count,
            inout_correctness_failed_sample_count,
            inout_failed_benchmark_count);
        break;
      }
      case IREE_BENCHMARK_LOOM_WORK_ITEM_NONE:
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unsupported benchmark work item kind %d",
                                  (int)work_item->kind);
        break;
    }
  }

  if (dispatch_compile_contexts != NULL) {
    for (iree_host_size_t i = 0; i < work_plan->dispatch_compile_item_count;
         ++i) {
      iree_benchmark_loom_dispatch_compile_context_deinitialize(
          &dispatch_compile_contexts[i]);
    }
  }
  iree_allocator_free(options->host_allocator, dispatch_compile_contexts);
  return status;
}
