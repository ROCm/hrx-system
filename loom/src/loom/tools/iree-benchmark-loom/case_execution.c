// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/case_execution.h"

#include <inttypes.h>
#include <string.h>

#include "loom/tools/iree-benchmark-loom/timing.h"

iree_host_size_t iree_benchmark_loom_case_sample_from_benchmark_sample(
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t benchmark_sample_ordinal) {
  return loom_testbench_benchmark_sample_case_ordinal(case_plan, benchmark_plan,
                                                      benchmark_sample_ordinal);
}

static iree_status_t iree_benchmark_loom_prepare_case_executor(
    const loom_testbench_module_plan_t* module_plan,
    iree_host_size_t case_index,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* arena,
    loom_testbench_prepared_case_t* out_prepared_case,
    loom_testbench_case_executor_t* out_executor, bool* out_initialized) {
  *out_initialized = false;
  IREE_RETURN_IF_ERROR(loom_testbench_prepare_case_execution(
      execution_options, module_plan, case_index, arena, out_prepared_case));
  IREE_RETURN_IF_ERROR(loom_testbench_case_executor_initialize(
      out_prepared_case, execution_options, out_executor));
  *out_initialized = true;
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_run_case_iteration(
    loom_testbench_case_executor_t* executor,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan, iree_host_size_t begin_sample,
    iree_host_size_t end_sample, iree_host_size_t* inout_failed_sample_count) {
  for (iree_host_size_t benchmark_sample_ordinal = begin_sample;
       benchmark_sample_ordinal < end_sample; ++benchmark_sample_ordinal) {
    const iree_host_size_t case_sample_ordinal =
        iree_benchmark_loom_case_sample_from_benchmark_sample(
            benchmark_plan, case_plan, benchmark_sample_ordinal);
    loom_testbench_case_sample_result_t sample_result = {0};
    IREE_RETURN_IF_ERROR(loom_testbench_run_case_sample(
        executor, case_sample_ordinal, &sample_result));
    if (!sample_result.passed) {
      ++*inout_failed_sample_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_run_benchmark_iterations(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_execution_options_t* execution_options,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    iree_host_size_t begin_sample, iree_host_size_t end_sample,
    iree_arena_allocator_t* arena, iree_allocator_t host_allocator,
    iree_benchmark_loom_benchmark_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));

  const loom_testbench_case_plan_t* case_plan =
      &module_plan->cases[benchmark_plan->case_index];
  out_result->samples_per_iteration = end_sample - begin_sample;
  if (end_sample == begin_sample + 1) {
    out_result->has_sample_ordinal = true;
    out_result->sample_ordinal =
        iree_benchmark_loom_case_sample_from_benchmark_sample(
            benchmark_plan, case_plan, begin_sample);
  }

  iree_duration_t* durations = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, policy->iterations * sizeof(*durations),
      (void**)&durations));

  iree_status_t status = iree_ok_status();
  loom_testbench_prepared_case_t prepared_case = {0};
  loom_testbench_case_executor_t executor = {0};
  bool executor_initialized = false;
  status = iree_benchmark_loom_prepare_case_executor(
      module_plan, benchmark_plan->case_index, execution_options, arena,
      &prepared_case, &executor, &executor_initialized);

  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < policy->warmup_iterations; ++i) {
    status = iree_benchmark_loom_run_case_iteration(
        &executor, benchmark_plan, case_plan, begin_sample, end_sample,
        &out_result->failed_sample_count);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < policy->iterations; ++i) {
    const iree_time_t start_time_ns = iree_time_now();
    status = iree_benchmark_loom_run_case_iteration(
        &executor, benchmark_plan, case_plan, begin_sample, end_sample,
        &out_result->failed_sample_count);
    const iree_time_t end_time_ns = iree_time_now();
    durations[i] =
        end_time_ns >= start_time_ns ? end_time_ns - start_time_ns : 0;
  }

  if (iree_status_is_ok(status)) {
    out_result->executed = true;
    out_result->passed = out_result->failed_sample_count == 0;
    iree_benchmark_loom_compute_timing_stats(durations, policy->iterations,
                                             &out_result->timing);
  }

  if (executor_initialized) {
    loom_testbench_case_executor_deinitialize(&executor);
  }
  iree_arena_reset(arena);
  iree_allocator_free(host_allocator, durations);
  return status;
}

iree_status_t iree_benchmark_loom_run_case_correctness_range(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    iree_host_size_t case_index,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_string_view_t sample_compilation, iree_host_size_t begin_sample,
    iree_host_size_t end_sample, iree_arena_allocator_t* arena,
    const iree_benchmark_loom_event_sink_t* event_sink,
    iree_host_size_t* out_sample_count,
    iree_host_size_t* out_failed_sample_count) {
  *out_sample_count = 0;
  *out_failed_sample_count = 0;

  const loom_testbench_case_plan_t* case_plan = &module_plan->cases[case_index];
  iree_status_t status = iree_ok_status();
  loom_testbench_prepared_case_t prepared_case = {0};
  loom_testbench_case_executor_t executor = {0};
  bool executor_initialized = false;
  status = iree_benchmark_loom_prepare_case_executor(
      module_plan, case_index, execution_options, arena, &prepared_case,
      &executor, &executor_initialized);

  for (iree_host_size_t benchmark_sample_ordinal = begin_sample;
       iree_status_is_ok(status) && benchmark_sample_ordinal < end_sample;
       ++benchmark_sample_ordinal) {
    const iree_host_size_t case_sample_ordinal =
        iree_benchmark_loom_case_sample_from_benchmark_sample(
            benchmark_plan, case_plan, benchmark_sample_ordinal);
    loom_testbench_case_sample_result_t sample_result = {0};
    status = loom_testbench_run_case_sample(&executor, case_sample_ordinal,
                                            &sample_result);
    if (iree_status_is_ok(status)) {
      status = iree_benchmark_loom_event_sink_emit_sample(
          event_sink, run, candidate, IREE_BENCHMARK_LOOM_INDEX_INVALID,
          module_plan->module, benchmark_plan, case_plan, sample_compilation,
          benchmark_sample_ordinal, case_sample_ordinal, &sample_result);
    }
    if (iree_status_is_ok(status)) {
      ++*out_sample_count;
      if (!sample_result.passed) {
        ++*out_failed_sample_count;
      }
    }
  }

  if (executor_initialized) {
    loom_testbench_case_executor_deinitialize(&executor);
  }
  iree_arena_reset(arena);
  return status;
}

static iree_status_t iree_benchmark_loom_emit_work_item_sample_aliases(
    const iree_benchmark_loom_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_work_plan_t* work_plan,
    const iree_benchmark_loom_work_item_t* work_item,
    iree_host_size_t sample_offset,
    const loom_testbench_case_sample_result_t* sample_result,
    const iree_benchmark_loom_event_sink_t* event_sink) {
  for (iree_host_size_t logical_sample_index = 0;
       logical_sample_index < work_plan->logical_sample_count;
       ++logical_sample_index) {
    const iree_benchmark_loom_logical_sample_t* logical_sample =
        &work_plan->logical_samples[logical_sample_index];
    if (logical_sample->work_item_index != work_item->work_item_index) {
      continue;
    }
    const iree_benchmark_loom_selected_benchmark_t* selection =
        &work_plan->selected_benchmarks[logical_sample->selection_index];
    const iree_host_size_t benchmark_sample_ordinal =
        logical_sample->begin_benchmark_sample + sample_offset;
    if (benchmark_sample_ordinal >= logical_sample->end_benchmark_sample) {
      continue;
    }
    const iree_host_size_t case_sample_ordinal =
        iree_benchmark_loom_case_sample_from_benchmark_sample(
            selection->benchmark_plan, selection->case_plan,
            benchmark_sample_ordinal);
    if (case_sample_ordinal != sample_result->sample_ordinal) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "benchmark work item alias for `%.*s` mapped benchmark sample "
          "%" PRIhsz " to case sample %" PRIhsz
          " but physical work item produced case sample %" PRIhsz,
          (int)selection->benchmark_plan->name.size,
          selection->benchmark_plan->name.data, benchmark_sample_ordinal,
          case_sample_ordinal, sample_result->sample_ordinal);
    }
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_event_sink_emit_sample(
        event_sink, run, &selection->identity, work_item->work_item_index,
        module_plan->module, selection->benchmark_plan, selection->case_plan,
        logical_sample->sample_compilation, benchmark_sample_ordinal,
        case_sample_ordinal, sample_result));
  }
  return iree_ok_status();
}

static bool iree_benchmark_loom_benchmark_result_counts_as_failed(
    const iree_benchmark_loom_benchmark_result_t* benchmark_result) {
  if (iree_string_view_equal(benchmark_result->status, IREE_SV("skipped"))) {
    return false;
  }
  return !benchmark_result->executed || !benchmark_result->passed;
}

iree_status_t iree_benchmark_loom_emit_work_item_result_aliases(
    const iree_benchmark_loom_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_work_plan_t* work_plan,
    const iree_benchmark_loom_work_item_t* work_item,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count,
    const iree_benchmark_loom_event_sink_t* event_sink,
    iree_host_size_t* inout_failed_benchmark_count) {
  for (iree_host_size_t logical_sample_index = 0;
       logical_sample_index < work_plan->logical_sample_count;
       ++logical_sample_index) {
    const iree_benchmark_loom_logical_sample_t* logical_sample =
        &work_plan->logical_samples[logical_sample_index];
    if (logical_sample->work_item_index != work_item->work_item_index) {
      continue;
    }
    const iree_benchmark_loom_selected_benchmark_t* selection =
        &work_plan->selected_benchmarks[logical_sample->selection_index];
    iree_benchmark_loom_benchmark_result_t alias_result = *benchmark_result;
    if (logical_sample->has_case_sample_ordinal) {
      alias_result.has_sample_ordinal = true;
      alias_result.sample_ordinal = logical_sample->case_sample_ordinal;
    }
    if (alias_result.samples_per_iteration == 0 &&
        logical_sample->has_case_sample_ordinal) {
      alias_result.samples_per_iteration = 1;
    }
    if (iree_benchmark_loom_benchmark_result_counts_as_failed(&alias_result)) {
      ++*inout_failed_benchmark_count;
    }
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_event_sink_emit_benchmark_result(
        event_sink, run, &selection->identity, work_item->work_item_index,
        module_plan->module, selection->benchmark_plan, selection->case_plan,
        &selection->policy, &alias_result, correctness_sample_count,
        correctness_failed_sample_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_event_sink_emit_profile(
        event_sink, run, &selection->identity, work_item->work_item_index,
        module_plan->module, selection->benchmark_plan, selection->case_plan,
        &selection->policy, &alias_result));
  }
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_run_work_item_correctness_range(
    const iree_benchmark_loom_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_work_plan_t* work_plan,
    const iree_benchmark_loom_work_item_t* work_item,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* arena,
    const iree_benchmark_loom_event_sink_t* event_sink,
    iree_host_size_t* out_sample_count,
    iree_host_size_t* out_failed_sample_count) {
  *out_sample_count = 0;
  *out_failed_sample_count = 0;
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &work_plan
           ->selected_benchmarks[work_item->representative_selection_index];
  const loom_testbench_benchmark_plan_t* benchmark_plan =
      selection->benchmark_plan;
  const loom_testbench_case_plan_t* case_plan = selection->case_plan;

  loom_testbench_prepared_case_t prepared_case = {0};
  loom_testbench_case_executor_t executor = {0};
  bool executor_initialized = false;
  iree_status_t status = iree_benchmark_loom_prepare_case_executor(
      module_plan, benchmark_plan->case_index, execution_options, arena,
      &prepared_case, &executor, &executor_initialized);

  for (iree_host_size_t benchmark_sample_ordinal =
           work_item->begin_benchmark_sample;
       iree_status_is_ok(status) &&
       benchmark_sample_ordinal < work_item->end_benchmark_sample;
       ++benchmark_sample_ordinal) {
    const iree_host_size_t sample_offset =
        benchmark_sample_ordinal - work_item->begin_benchmark_sample;
    const iree_host_size_t case_sample_ordinal =
        iree_benchmark_loom_case_sample_from_benchmark_sample(
            benchmark_plan, case_plan, benchmark_sample_ordinal);
    loom_testbench_case_sample_result_t sample_result = {0};
    status = loom_testbench_run_case_sample(&executor, case_sample_ordinal,
                                            &sample_result);
    if (iree_status_is_ok(status)) {
      status = iree_benchmark_loom_emit_work_item_sample_aliases(
          run, module_plan, work_plan, work_item, sample_offset, &sample_result,
          event_sink);
    }
    if (iree_status_is_ok(status)) {
      ++*out_sample_count;
      if (!sample_result.passed) {
        ++*out_failed_sample_count;
      }
    }
  }

  if (executor_initialized) {
    loom_testbench_case_executor_deinitialize(&executor);
  }
  iree_arena_reset(arena);
  return status;
}

iree_status_t iree_benchmark_loom_run_case_end_to_end_work_item(
    const iree_benchmark_loom_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_work_plan_t* work_plan,
    const iree_benchmark_loom_work_item_t* work_item,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* execution_arena, iree_allocator_t allocator,
    const iree_benchmark_loom_event_sink_t* event_sink,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &work_plan
           ->selected_benchmarks[work_item->representative_selection_index];
  const loom_testbench_benchmark_plan_t* benchmark_plan =
      selection->benchmark_plan;
  const iree_benchmark_loom_benchmark_policy_t* policy = &selection->policy;
  loom_testbench_case_execution_options_t benchmark_execution_options =
      *execution_options;

  iree_host_size_t correctness_sample_count = 0;
  iree_host_size_t correctness_failed_sample_count = 0;
  iree_status_t status = iree_benchmark_loom_run_work_item_correctness_range(
      run, module_plan, work_plan, work_item, &benchmark_execution_options,
      execution_arena, event_sink, &correctness_sample_count,
      &correctness_failed_sample_count);
  if (iree_status_is_ok(status)) {
    *inout_correctness_sample_count += correctness_sample_count;
    *inout_correctness_failed_sample_count += correctness_failed_sample_count;
  }

  iree_benchmark_loom_benchmark_result_t benchmark_result = {0};
  if (iree_status_is_ok(status) && correctness_failed_sample_count == 0) {
    status = iree_benchmark_loom_run_benchmark_iterations(
        module_plan, benchmark_plan, &benchmark_execution_options, policy,
        work_item->begin_benchmark_sample, work_item->end_benchmark_sample,
        execution_arena, allocator, &benchmark_result);
  }
  if (iree_status_is_ok(status) && correctness_failed_sample_count != 0) {
    benchmark_result = (iree_benchmark_loom_benchmark_result_t){
        .executed = false,
        .passed = false,
        .samples_per_iteration = correctness_sample_count,
        .failed_sample_count = correctness_failed_sample_count,
    };
    if (work_item->has_case_sample_ordinal) {
      benchmark_result.has_sample_ordinal = true;
      benchmark_result.sample_ordinal = work_item->case_sample_ordinal;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_emit_work_item_result_aliases(
        run, module_plan, work_plan, work_item, &benchmark_result,
        correctness_sample_count, correctness_failed_sample_count, event_sink,
        inout_failed_benchmark_count);
  }
  return status;
}
