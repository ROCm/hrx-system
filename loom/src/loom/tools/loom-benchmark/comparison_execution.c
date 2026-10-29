// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-benchmark/comparison_execution.h"

#include <inttypes.h>
#include <string.h>

#include "loom/ir/module.h"
#include "loom/tooling/execution/hal/benchmark.h"
#include "loom/tools/loom-benchmark/case_execution.h"
#include "loom/tools/loom-benchmark/dispatch_benchmark.h"
#include "loom/tools/loom-benchmark/hal_setup.h"
#include "loom/tools/loom-benchmark/module_query.h"
#include "loom/tools/loom-benchmark/scenario_execution.h"

static iree_status_t loom_benchmark_comparison_candidate_record_timing(
    loom_benchmark_dispatch_comparison_candidate_t* candidate,
    const loom_benchmark_benchmark_result_t* benchmark_result) {
  if (!benchmark_result->executed || !benchmark_result->passed ||
      !benchmark_result->has_hal_benchmark) {
    return iree_ok_status();
  }
  if (candidate->sample_count >= candidate->sample_capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "interleaved comparison collected more timing rows than planned");
  }
  const loom_run_benchmark_timing_stats_t* operation_timing =
      &benchmark_result->hal_benchmark.timing.operation_timing;
  candidate->p50_samples[candidate->sample_count] = operation_timing->p50_ns;
  candidate->p90_samples[candidate->sample_count] = operation_timing->p90_ns;
  ++candidate->sample_count;
  return iree_ok_status();
}

static bool loom_benchmark_sample_attr_equal(loom_attribute_t lhs,
                                             loom_attribute_t rhs) {
  if (lhs.kind != rhs.kind) {
    return false;
  }
  switch ((loom_attr_kind_t)lhs.kind) {
    case LOOM_ATTR_I64:
      return loom_attr_as_i64(lhs) == loom_attr_as_i64(rhs);
    case LOOM_ATTR_F64:
      return loom_attr_as_f64(lhs) == loom_attr_as_f64(rhs);
    case LOOM_ATTR_BOOL:
      return loom_attr_as_bool(lhs) == loom_attr_as_bool(rhs);
    case LOOM_ATTR_STRING:
      return loom_attr_as_string_id(lhs) == loom_attr_as_string_id(rhs);
    case LOOM_ATTR_STRING_ARRAY:
    case LOOM_ATTR_F64_ARRAY:
      return loom_attribute_equal(&lhs, &rhs);
    default:
      return lhs.raw == rhs.raw;
  }
}

static iree_status_t loom_benchmark_validate_comparison_sample_parameters(
    const loom_module_t* module,
    const loom_benchmark_dispatch_comparison_candidate_t* baseline,
    const loom_benchmark_dispatch_comparison_candidate_t* candidate) {
  const loom_testbench_case_plan_t* baseline_case =
      baseline->selection->case_plan;
  const loom_testbench_case_plan_t* candidate_case =
      candidate->selection->case_plan;
  const iree_string_view_t baseline_name =
      baseline->selection->benchmark_plan->name;
  const iree_string_view_t candidate_name =
      candidate->selection->benchmark_plan->name;
  loom_testbench_case_sample_t baseline_case_sample =
      loom_testbench_benchmark_case_sample(baseline->selection->benchmark_plan,
                                           baseline->begin_sample);
  loom_testbench_case_sample_t candidate_case_sample =
      loom_testbench_benchmark_case_sample(candidate->selection->benchmark_plan,
                                           candidate->begin_sample);
  if (baseline_case->parameter_count != candidate_case->parameter_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "comparison candidate `%.*s` has %" PRIhsz
        " case parameters, but baseline `%.*s` has %" PRIhsz,
        (int)candidate_name.size, candidate_name.data,
        candidate_case->parameter_count, (int)baseline_name.size,
        baseline_name.data, baseline_case->parameter_count);
  }

  for (iree_host_size_t parameter_index = 0;
       parameter_index < baseline_case->parameter_count; ++parameter_index) {
    const loom_testbench_parameter_plan_t* baseline_parameter =
        &baseline_case->parameters[parameter_index];
    const loom_testbench_parameter_plan_t* candidate_parameter =
        &candidate_case->parameters[parameter_index];
    iree_string_view_t baseline_parameter_name = baseline_parameter->name;
    if (iree_string_view_is_empty(baseline_parameter_name)) {
      baseline_parameter_name =
          loom_benchmark_value_name(module, baseline_parameter->value_id);
    }
    iree_string_view_t candidate_parameter_name = candidate_parameter->name;
    if (iree_string_view_is_empty(candidate_parameter_name)) {
      candidate_parameter_name =
          loom_benchmark_value_name(module, candidate_parameter->value_id);
    }
    if (!iree_string_view_equal(baseline_parameter_name,
                                candidate_parameter_name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "comparison candidate `%.*s` case parameter %" PRIhsz
          " is `%.*s`, but baseline `%.*s` uses `%.*s`",
          (int)candidate_name.size, candidate_name.data, parameter_index,
          (int)candidate_parameter_name.size, candidate_parameter_name.data,
          (int)baseline_name.size, baseline_name.data,
          (int)baseline_parameter_name.size, baseline_parameter_name.data);
    }
    if (baseline_parameter->kind != candidate_parameter->kind ||
        !loom_type_equal(baseline_parameter->type, candidate_parameter->type)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "comparison candidate `%.*s` case parameter `%.*s` does not match "
          "baseline `%.*s` parameter kind/type",
          (int)candidate_name.size, candidate_name.data,
          (int)baseline_parameter_name.size, baseline_parameter_name.data,
          (int)baseline_name.size, baseline_name.data);
    }

    const iree_host_size_t baseline_parameter_sample =
        loom_testbench_case_sample_next(&baseline_case_sample,
                                        baseline_parameter->sample_count);
    const iree_host_size_t candidate_parameter_sample =
        loom_testbench_case_sample_next(&candidate_case_sample,
                                        candidate_parameter->sample_count);
    loom_attribute_t baseline_value = loom_attr_absent();
    IREE_RETURN_IF_ERROR(loom_testbench_parameter_sample_value(
        baseline_parameter, baseline_parameter_sample, &baseline_value));
    loom_attribute_t candidate_value = loom_attr_absent();
    IREE_RETURN_IF_ERROR(loom_testbench_parameter_sample_value(
        candidate_parameter, candidate_parameter_sample, &candidate_value));
    if (!loom_benchmark_sample_attr_equal(baseline_value, candidate_value)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "comparison candidate `%.*s` case parameter `%.*s` sample differs "
          "from baseline `%.*s`; use --sample= to select matching samples",
          (int)candidate_name.size, candidate_name.data,
          (int)baseline_parameter_name.size, baseline_parameter_name.data,
          (int)baseline_name.size, baseline_name.data);
    }
  }

  return iree_ok_status();
}

static iree_status_t loom_benchmark_validate_comparison_samples(
    const loom_module_t* module,
    const loom_benchmark_dispatch_comparison_candidate_t* candidates,
    iree_host_size_t candidate_count) {
  if (candidate_count < 2 || candidates[0].selection->scenario_plan != NULL) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 1; i < candidate_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_benchmark_validate_comparison_sample_parameters(
        module, &candidates[0], &candidates[i]));
  }
  return iree_ok_status();
}

iree_host_size_t loom_benchmark_dispatch_comparison_sample_capacity(
    loom_benchmark_interleave_mode_t interleave_mode,
    iree_host_size_t candidate_count, iree_host_size_t candidate_index,
    iree_host_size_t repetitions) {
  if (interleave_mode == LOOM_BENCHMARK_INTERLEAVE_ABABA) {
    return candidate_index == 0 ? (candidate_count - 1) * (repetitions + 1)
                                : repetitions;
  }
  return repetitions;
}

static iree_status_t loom_benchmark_find_comparison_logical_sample(
    const loom_benchmark_work_plan_t* work_plan,
    iree_host_size_t selection_index,
    const loom_benchmark_logical_sample_t** out_logical_sample) {
  *out_logical_sample = NULL;
  iree_host_size_t match_count = 0;
  for (iree_host_size_t i = 0; i < work_plan->logical_sample_count; ++i) {
    const loom_benchmark_logical_sample_t* logical_sample =
        &work_plan->logical_samples[i];
    if (logical_sample->selection_index == selection_index) {
      *out_logical_sample = logical_sample;
      ++match_count;
    }
  }
  if (match_count != 1) {
    const loom_benchmark_selected_benchmark_t* selection =
        &work_plan->selected_benchmarks[selection_index];
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "interleaved comparison benchmark `%.*s` selected %" PRIhsz
        " samples; use --sample= or benchmark parameters to select exactly "
        "one concrete sample",
        (int)selection->benchmark_plan->name.size,
        selection->benchmark_plan->name.data, match_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_benchmark_initialize_dispatch_comparison_candidates(
    const loom_benchmark_comparison_execution_options_t* options,
    loom_benchmark_dispatch_comparison_candidate_t** out_candidates) {
  *out_candidates = NULL;
  const loom_benchmark_work_plan_t* work_plan = options->work_plan;
  const loom_benchmark_options_t* benchmark_options =
      options->benchmark_options;
  const iree_host_size_t selection_count = work_plan->selected_benchmark_count;
  if (benchmark_options->interleave_mode == LOOM_BENCHMARK_INTERLEAVE_ABABA &&
      selection_count != 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ABABA comparison requires exactly two selected "
                            "benchmarks; use --interleave=round_robin for "
                            "ABCD-style comparisons");
  }
  if (benchmark_options->repetitions == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "interleaved comparison repetitions must be "
                            "positive");
  }
  if (selection_count > 26) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "interleaved comparison supports at most 26 "
                            "selected benchmarks; got %" PRIhsz,
                            selection_count);
  }

  loom_benchmark_dispatch_comparison_candidate_t* candidates = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(options->host_allocator, selection_count,
                                  sizeof(*candidates), (void**)&candidates));
  memset(candidates, 0, selection_count * sizeof(*candidates));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < selection_count;
       ++i) {
    const loom_benchmark_selected_benchmark_t* selection =
        &work_plan->selected_benchmarks[i];
    if (selection->policy.measure_kind !=
        LOOM_BENCHMARK_MEASURE_DISPATCH_COMPLETE) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "interleaved comparison benchmark `%.*s` must use "
                           "measure = \"dispatch_complete\"",
                           (int)selection->benchmark_plan->name.size,
                           selection->benchmark_plan->name.data);
      break;
    }
    const loom_benchmark_logical_sample_t* logical_sample = NULL;
    status = loom_benchmark_find_comparison_logical_sample(work_plan, i,
                                                           &logical_sample);
    if (!iree_status_is_ok(status)) {
      break;
    }
    candidates[i].selection = selection;
    candidates[i].module = options->module_plan->module;
    candidates[i].work_item_index = logical_sample->work_item_index;
    candidates[i].begin_sample = logical_sample->begin_benchmark_sample;
    candidates[i].end_sample = logical_sample->end_benchmark_sample;
    const iree_host_size_t sample_capacity =
        loom_benchmark_dispatch_comparison_sample_capacity(
            benchmark_options->interleave_mode, selection_count, i,
            benchmark_options->repetitions);
    candidates[i].sample_capacity = sample_capacity;
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc_array(options->host_allocator, sample_capacity,
                                      sizeof(*candidates[i].p50_samples),
                                      (void**)&candidates[i].p50_samples);
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc_array(options->host_allocator, sample_capacity,
                                      sizeof(*candidates[i].p90_samples),
                                      (void**)&candidates[i].p90_samples);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_candidates = candidates;
  } else {
    for (iree_host_size_t i = 0; i < selection_count; ++i) {
      iree_allocator_free(options->host_allocator, candidates[i].p50_samples);
      iree_allocator_free(options->host_allocator, candidates[i].p90_samples);
    }
    iree_allocator_free(options->host_allocator, candidates);
  }
  return status;
}

static void loom_benchmark_dispatch_comparison_candidate_deinitialize(
    loom_benchmark_dispatch_comparison_candidate_t* candidate,
    iree_allocator_t allocator) {
  iree_allocator_free(allocator, candidate->p50_samples);
  iree_allocator_free(allocator, candidate->p90_samples);
  *candidate = (loom_benchmark_dispatch_comparison_candidate_t){0};
}

static void loom_benchmark_dispatch_comparison_candidates_deinitialize(
    loom_benchmark_dispatch_comparison_candidate_t* candidates,
    iree_host_size_t candidate_count, iree_allocator_t allocator) {
  if (candidates == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    loom_benchmark_dispatch_comparison_candidate_deinitialize(&candidates[i],
                                                              allocator);
  }
  iree_allocator_free(allocator, candidates);
}

static iree_status_t loom_benchmark_run_comparison_window(
    const loom_benchmark_comparison_execution_options_t* options,
    loom_benchmark_dispatch_comparison_candidate_t* candidate,
    loom_benchmark_scenario_execution_t* scenario_execution,
    loom_benchmark_hal_compile_context_t* compile_contexts,
    const loom_benchmark_hal_work_item_state_t* work_item_states,
    const loom_benchmark_candidate_identity_t* baseline,
    iree_string_view_t comparison_group, iree_string_view_t method,
    iree_host_size_t order_index, iree_host_size_t repetition_index,
    char schedule_token, iree_host_size_t* inout_failed_benchmark_count) {
  if (!candidate->runnable) {
    return iree_ok_status();
  }
  loom_benchmark_benchmark_policy_t measurement_policy =
      candidate->selection->policy;
  const bool profile_suppressed =
      iree_any_bit_set(measurement_policy.hal_options.flags,
                       LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH);
  measurement_policy.hal_options.flags &=
      ~LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH;

  loom_benchmark_benchmark_result_t benchmark_result = {0};
  iree_status_t status = iree_ok_status();
  if (scenario_execution != NULL) {
    status = loom_benchmark_scenario_execution_measure(
        scenario_execution, &measurement_policy, &benchmark_result);
  } else {
    const loom_benchmark_work_plan_t* work_plan = options->work_plan;
    const loom_benchmark_work_item_t* work_item =
        &work_plan->work_items[candidate->work_item_index];
    loom_benchmark_hal_compile_context_t* compile_context =
        &compile_contexts[work_item->hal_compile_item_index];
    if (compile_context->uses_sequence) {
      status = loom_benchmark_run_hal_sequence_benchmark_sample(
          options->run, &candidate->selection->identity, options->module_plan,
          candidate->selection->benchmark_plan, &measurement_policy,
          options->benchmark_options, options->hal_context,
          &compile_context->hal_sequence,
          &compile_context->benchmark_execution_options,
          options->execution_arena, candidate->begin_sample,
          options->host_allocator, &benchmark_result);
    } else {
      status = loom_benchmark_run_hal_benchmark_sample(
          options->run, &candidate->selection->identity, options->module_plan,
          candidate->selection->benchmark_plan, &measurement_policy,
          options->benchmark_options, &compile_context->hal_provider,
          &compile_context->benchmark_execution_options,
          options->execution_arena, candidate->begin_sample,
          options->host_allocator, &benchmark_result);
    }
    benchmark_result.compile_item = compile_context->compile_item;
    benchmark_result.launch_evidence =
        &work_item_states[candidate->work_item_index].launch_evidence;
  }
  if (iree_status_is_ok(status) &&
      (!benchmark_result.executed || !benchmark_result.passed)) {
    ++*inout_failed_benchmark_count;
  }
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_comparison_candidate_record_timing(
        candidate, &benchmark_result);
  }
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_event_sink_emit_benchmark_repetition(
        options->event_sink, options->run, candidate, baseline,
        comparison_group, method, order_index, repetition_index, schedule_token,
        profile_suppressed, &benchmark_result);
  }
  return status;
}

static iree_status_t loom_benchmark_prepare_dispatch_comparison_work(
    const loom_benchmark_comparison_execution_options_t* options,
    const loom_benchmark_hal_setup_options_t* hal_options,
    loom_benchmark_hal_compile_context_t* compile_contexts,
    loom_benchmark_hal_work_item_state_t* work_item_states,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  const loom_benchmark_work_plan_t* work_plan = options->work_plan;
  for (iree_host_size_t work_item_index = 0;
       work_item_index < work_plan->work_item_count; ++work_item_index) {
    const loom_benchmark_work_item_t* work_item =
        &work_plan->work_items[work_item_index];
    IREE_RETURN_IF_ERROR(loom_benchmark_prepare_hal_work_item(
        hal_options, work_item,
        &compile_contexts[work_item->hal_compile_item_index],
        inout_correctness_sample_count, inout_correctness_failed_sample_count,
        inout_failed_benchmark_count, &work_item_states[work_item_index]));
  }
  return iree_ok_status();
}

static void loom_benchmark_apply_dispatch_comparison_work_state(
    const loom_benchmark_hal_work_item_state_t* work_item_states,
    loom_benchmark_dispatch_comparison_candidate_t* candidates,
    iree_host_size_t candidate_count) {
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    loom_benchmark_dispatch_comparison_candidate_t* candidate = &candidates[i];
    const loom_benchmark_hal_work_item_state_t* state =
        &work_item_states[candidate->work_item_index];
    candidate->correctness_sample_count = state->correctness_sample_count;
    candidate->correctness_failed_sample_count =
        state->correctness_failed_sample_count;
    candidate->runnable = state->runnable;
  }
}

static iree_status_t loom_benchmark_prepare_scenario_comparison_work(
    const loom_benchmark_comparison_execution_options_t* options,
    const loom_benchmark_work_plan_execution_options_t* execution_options,
    loom_benchmark_dispatch_comparison_candidate_t* candidates,
    iree_host_size_t candidate_count,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count,
    loom_benchmark_scenario_execution_t*** out_executions) {
  *out_executions = NULL;
  loom_benchmark_scenario_execution_t** executions = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(options->host_allocator, candidate_count,
                                  sizeof(*executions), (void**)&executions));
  memset(executions, 0, candidate_count * sizeof(*executions));
  *out_executions = executions;

  const iree_string_view_t entropy_domain =
      candidates[0].selection->scenario_plan->name;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < candidate_count;
       ++i) {
    status = loom_benchmark_scenario_execution_allocate(
        execution_options, entropy_domain, &executions[i]);
    if (iree_status_is_ok(status)) {
      const loom_benchmark_work_item_t* work_item =
          &options->work_plan->work_items[candidates[i].work_item_index];
      status = loom_benchmark_scenario_execution_prepare(
          executions[i], candidates[i].selection, work_item->work_item_index,
          candidates[i].begin_sample, work_item->scenario_coordinate,
          &candidates[i].runnable);
    }
    if (iree_status_is_ok(status)) {
      candidates[i].correctness_sample_count =
          options->benchmark_options->scenario_correctness ? 1 : 0;
      candidates[i].correctness_failed_sample_count = !candidates[i].runnable;
      *inout_correctness_sample_count += candidates[i].correctness_sample_count;
      *inout_correctness_failed_sample_count += !candidates[i].runnable;
      *inout_failed_benchmark_count += !candidates[i].runnable;
    }
  }
  for (iree_host_size_t i = 1; iree_status_is_ok(status) && i < candidate_count;
       ++i) {
    if (candidates[0].runnable && candidates[i].runnable) {
      status = loom_benchmark_scenario_execution_validate_equivalent(
          executions[0], executions[i]);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_event_sink_emit_device(
        options->event_sink, options->run, options->hal_context);
  }
  return status;
}

static void loom_benchmark_scenario_comparison_executions_free(
    loom_benchmark_scenario_execution_t** executions,
    iree_host_size_t execution_count, iree_allocator_t allocator) {
  if (executions == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < execution_count; ++i) {
    loom_benchmark_scenario_execution_free(executions[i]);
  }
  iree_allocator_free(allocator, executions);
}

iree_status_t loom_benchmark_run_dispatch_comparison(
    const loom_benchmark_comparison_execution_options_t* options,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  const loom_benchmark_work_plan_t* work_plan = options->work_plan;
  const loom_benchmark_options_t* benchmark_options =
      options->benchmark_options;
  const iree_host_size_t selection_count = work_plan->selected_benchmark_count;
  loom_benchmark_dispatch_comparison_candidate_t* candidates = NULL;
  loom_benchmark_hal_compile_context_t* compile_contexts = NULL;
  loom_benchmark_hal_work_item_state_t* work_item_states = NULL;
  loom_benchmark_scenario_execution_t** scenario_executions = NULL;
  iree_status_t status =
      loom_benchmark_initialize_dispatch_comparison_candidates(options,
                                                               &candidates);
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_validate_comparison_samples(
        options->module_plan->module, candidates, selection_count);
  }
  const bool compares_scenarios =
      iree_status_is_ok(status) &&
      candidates[0].selection->scenario_plan != NULL;
  if (iree_status_is_ok(status) && !compares_scenarios &&
      work_plan->hal_compile_item_count != 0) {
    status = iree_allocator_malloc_array(
        options->host_allocator, work_plan->hal_compile_item_count,
        sizeof(*compile_contexts), (void**)&compile_contexts);
    if (iree_status_is_ok(status)) {
      memset(compile_contexts, 0,
             work_plan->hal_compile_item_count * sizeof(*compile_contexts));
    }
  }
  if (iree_status_is_ok(status) && !compares_scenarios &&
      work_plan->work_item_count != 0) {
    status = iree_allocator_malloc_array(
        options->host_allocator, work_plan->work_item_count,
        sizeof(*work_item_states), (void**)&work_item_states);
    if (iree_status_is_ok(status)) {
      memset(work_item_states, 0,
             work_plan->work_item_count * sizeof(*work_item_states));
    }
  }
  const loom_benchmark_hal_setup_options_t hal_options = {
      .run = options->run,
      .module_plan = options->module_plan,
      .work_plan = work_plan,
      .benchmark_options = benchmark_options,
      .hal_context = options->hal_context,
      .compilation = options->compilation,
      .case_execution_options = options->case_execution_options,
      .execution_arena = options->execution_arena,
      .host_allocator = options->host_allocator,
      .event_sink = options->event_sink,
  };
  const loom_benchmark_work_plan_execution_options_t execution_options = {
      .run = options->run,
      .module_plan = options->module_plan,
      .work_plan = work_plan,
      .benchmark_options = benchmark_options,
      .hal_context = options->hal_context,
      .compilation = options->compilation,
      .case_execution_options = options->case_execution_options,
      .scenario_oracle_profile = options->scenario_oracle_profile,
      .execution_arena = options->execution_arena,
      .host_allocator = options->host_allocator,
      .event_sink = options->event_sink,
  };
  if (iree_status_is_ok(status) && compares_scenarios) {
    status = loom_benchmark_prepare_scenario_comparison_work(
        options, &execution_options, candidates, selection_count,
        inout_correctness_sample_count, inout_correctness_failed_sample_count,
        inout_failed_benchmark_count, &scenario_executions);
  } else if (iree_status_is_ok(status)) {
    status = loom_benchmark_prepare_dispatch_comparison_work(
        options, &hal_options, compile_contexts, work_item_states,
        inout_correctness_sample_count, inout_correctness_failed_sample_count,
        inout_failed_benchmark_count);
  }
  if (iree_status_is_ok(status) && !compares_scenarios) {
    loom_benchmark_apply_dispatch_comparison_work_state(
        work_item_states, candidates, selection_count);
  }

  const loom_benchmark_candidate_identity_t* baseline = NULL;
  iree_string_view_t comparison_group = iree_string_view_empty();
  iree_string_view_t method = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    baseline = &work_plan->selected_benchmarks[0].identity;
    comparison_group = work_plan->selected_benchmarks[0].benchmark_plan->name;
    method =
        loom_benchmark_interleave_mode_name(benchmark_options->interleave_mode);
  }
  iree_host_size_t order_index = 0;
  if (iree_status_is_ok(status) &&
      benchmark_options->interleave_mode == LOOM_BENCHMARK_INTERLEAVE_ABABA) {
    for (iree_host_size_t candidate_index = 1;
         iree_status_is_ok(status) && candidate_index < selection_count;
         ++candidate_index) {
      status = loom_benchmark_run_comparison_window(
          options, &candidates[0],
          compares_scenarios ? scenario_executions[0] : NULL, compile_contexts,
          work_item_states, baseline, comparison_group, method, order_index++,
          /*repetition_index=*/0, 'A', inout_failed_benchmark_count);
      for (iree_host_size_t repetition_index = 0;
           iree_status_is_ok(status) &&
           repetition_index < benchmark_options->repetitions;
           ++repetition_index) {
        status = loom_benchmark_run_comparison_window(
            options, &candidates[candidate_index],
            compares_scenarios ? scenario_executions[candidate_index] : NULL,
            compile_contexts, work_item_states, baseline, comparison_group,
            method, order_index++, repetition_index,
            (char)('A' + candidate_index), inout_failed_benchmark_count);
        if (iree_status_is_ok(status)) {
          status = loom_benchmark_run_comparison_window(
              options, &candidates[0],
              compares_scenarios ? scenario_executions[0] : NULL,
              compile_contexts, work_item_states, baseline, comparison_group,
              method, order_index++, repetition_index + 1, 'A',
              inout_failed_benchmark_count);
        }
      }
    }
  } else if (iree_status_is_ok(status) &&
             benchmark_options->interleave_mode ==
                 LOOM_BENCHMARK_INTERLEAVE_ROUND_ROBIN) {
    for (iree_host_size_t repetition_index = 0;
         iree_status_is_ok(status) &&
         repetition_index < benchmark_options->repetitions;
         ++repetition_index) {
      for (iree_host_size_t candidate_index = 0;
           iree_status_is_ok(status) && candidate_index < selection_count;
           ++candidate_index) {
        status = loom_benchmark_run_comparison_window(
            options, &candidates[candidate_index],
            compares_scenarios ? scenario_executions[candidate_index] : NULL,
            compile_contexts, work_item_states, baseline, comparison_group,
            method, order_index++, repetition_index,
            (char)('A' + candidate_index), inout_failed_benchmark_count);
      }
    }
  }

  for (iree_host_size_t candidate_index = 1;
       iree_status_is_ok(status) && candidate_index < selection_count;
       ++candidate_index) {
    status = loom_benchmark_event_sink_emit_comparison(
        options->event_sink, options->run, &candidates[0],
        &candidates[candidate_index], comparison_group, method);
  }

  if (compile_contexts != NULL) {
    for (iree_host_size_t i = 0; i < work_plan->hal_compile_item_count; ++i) {
      loom_benchmark_hal_compile_context_deinitialize(&compile_contexts[i]);
    }
  }
  if (work_item_states != NULL) {
    for (iree_host_size_t i = 0; i < work_plan->work_item_count; ++i) {
      loom_benchmark_hal_work_item_state_deinitialize(&work_item_states[i]);
    }
  }
  iree_allocator_free(options->host_allocator, work_item_states);
  iree_allocator_free(options->host_allocator, compile_contexts);
  loom_benchmark_scenario_comparison_executions_free(
      scenario_executions, selection_count, options->host_allocator);
  loom_benchmark_dispatch_comparison_candidates_deinitialize(
      candidates, selection_count, options->host_allocator);
  return status;
}
