// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/comparison_execution.h"

#include <inttypes.h>
#include <string.h>

#include "loom/ir/module.h"
#include "loom/tooling/execution/hal/benchmark.h"
#include "loom/tools/iree-benchmark-loom/case_execution.h"
#include "loom/tools/iree-benchmark-loom/dispatch_benchmark.h"
#include "loom/tools/iree-benchmark-loom/dispatch_setup.h"
#include "loom/tools/iree-benchmark-loom/module_query.h"

static iree_status_t iree_benchmark_loom_comparison_candidate_record_timing(
    iree_benchmark_loom_dispatch_comparison_candidate_t* candidate,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result) {
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

static bool iree_benchmark_loom_sample_attr_equal(loom_attribute_t lhs,
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
    default:
      return lhs.raw == rhs.raw;
  }
}

static iree_status_t iree_benchmark_loom_validate_comparison_sample_parameters(
    const loom_module_t* module,
    const iree_benchmark_loom_dispatch_comparison_candidate_t* baseline,
    const iree_benchmark_loom_dispatch_comparison_candidate_t* candidate) {
  const loom_testbench_case_plan_t* baseline_case =
      baseline->selection->case_plan;
  const loom_testbench_case_plan_t* candidate_case =
      candidate->selection->case_plan;
  const iree_string_view_t baseline_name =
      baseline->selection->benchmark_plan->name;
  const iree_string_view_t candidate_name =
      candidate->selection->benchmark_plan->name;
  const iree_host_size_t baseline_case_sample =
      iree_benchmark_loom_case_sample_from_benchmark_sample(
          baseline->selection->benchmark_plan, baseline_case,
          baseline->begin_sample);
  const iree_host_size_t candidate_case_sample =
      iree_benchmark_loom_case_sample_from_benchmark_sample(
          candidate->selection->benchmark_plan, candidate_case,
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
          iree_benchmark_loom_value_name(module, baseline_parameter->value_id);
    }
    iree_string_view_t candidate_parameter_name = candidate_parameter->name;
    if (iree_string_view_is_empty(candidate_parameter_name)) {
      candidate_parameter_name =
          iree_benchmark_loom_value_name(module, candidate_parameter->value_id);
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
        loom_testbench_case_sample_parameter_ordinal(
            baseline_case, baseline_case_sample, parameter_index);
    const iree_host_size_t candidate_parameter_sample =
        loom_testbench_case_sample_parameter_ordinal(
            candidate_case, candidate_case_sample, parameter_index);
    loom_attribute_t baseline_value = loom_attr_absent();
    IREE_RETURN_IF_ERROR(loom_testbench_parameter_sample_value(
        baseline_parameter, baseline_parameter_sample, &baseline_value));
    loom_attribute_t candidate_value = loom_attr_absent();
    IREE_RETURN_IF_ERROR(loom_testbench_parameter_sample_value(
        candidate_parameter, candidate_parameter_sample, &candidate_value));
    if (!iree_benchmark_loom_sample_attr_equal(baseline_value,
                                               candidate_value)) {
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

static iree_status_t iree_benchmark_loom_validate_comparison_samples(
    const loom_module_t* module,
    const iree_benchmark_loom_dispatch_comparison_candidate_t* candidates,
    iree_host_size_t candidate_count) {
  if (candidate_count < 2) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 1; i < candidate_count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_validate_comparison_sample_parameters(
            module, &candidates[0], &candidates[i]));
  }
  return iree_ok_status();
}

iree_host_size_t iree_benchmark_loom_dispatch_comparison_sample_capacity(
    iree_benchmark_loom_interleave_mode_t interleave_mode,
    iree_host_size_t candidate_count, iree_host_size_t candidate_index,
    iree_host_size_t repetitions) {
  if (interleave_mode == IREE_BENCHMARK_LOOM_INTERLEAVE_ABABA) {
    return candidate_index == 0 ? (candidate_count - 1) * (repetitions + 1)
                                : repetitions;
  }
  return repetitions;
}

static iree_status_t iree_benchmark_loom_find_comparison_logical_sample(
    const iree_benchmark_loom_work_plan_t* work_plan,
    iree_host_size_t selection_index,
    const iree_benchmark_loom_logical_sample_t** out_logical_sample) {
  *out_logical_sample = NULL;
  iree_host_size_t match_count = 0;
  for (iree_host_size_t i = 0; i < work_plan->logical_sample_count; ++i) {
    const iree_benchmark_loom_logical_sample_t* logical_sample =
        &work_plan->logical_samples[i];
    if (logical_sample->selection_index == selection_index) {
      *out_logical_sample = logical_sample;
      ++match_count;
    }
  }
  if (match_count != 1) {
    const iree_benchmark_loom_selected_benchmark_t* selection =
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

static iree_status_t
iree_benchmark_loom_initialize_dispatch_comparison_candidates(
    const iree_benchmark_loom_comparison_execution_options_t* options,
    iree_benchmark_loom_dispatch_comparison_candidate_t** out_candidates) {
  *out_candidates = NULL;
  const iree_benchmark_loom_work_plan_t* work_plan = options->work_plan;
  const iree_benchmark_loom_options_t* benchmark_options =
      options->benchmark_options;
  const iree_host_size_t selection_count = work_plan->selected_benchmark_count;
  if (benchmark_options->sample_compilation_mode ==
      IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_BOTH) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "interleaved comparisons require one sample-compilation mode; use "
        "--sample-compilation=once or --sample-compilation=per_sample");
  }
  if (benchmark_options->interleave_mode ==
          IREE_BENCHMARK_LOOM_INTERLEAVE_ABABA &&
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

  iree_benchmark_loom_dispatch_comparison_candidate_t* candidates = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(options->host_allocator, selection_count,
                                  sizeof(*candidates), (void**)&candidates));
  memset(candidates, 0, selection_count * sizeof(*candidates));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < selection_count;
       ++i) {
    const iree_benchmark_loom_selected_benchmark_t* selection =
        &work_plan->selected_benchmarks[i];
    if (selection->policy.measure_kind !=
        IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "interleaved comparison benchmark `%.*s` must use "
                           "measure = \"dispatch_complete\"",
                           (int)selection->benchmark_plan->name.size,
                           selection->benchmark_plan->name.data);
      break;
    }
    const iree_benchmark_loom_logical_sample_t* logical_sample = NULL;
    status = iree_benchmark_loom_find_comparison_logical_sample(
        work_plan, i, &logical_sample);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (logical_sample->end_benchmark_sample !=
        logical_sample->begin_benchmark_sample + 1) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "interleaved comparison benchmark `%.*s` selected a non-scalar "
          "sample window",
          (int)selection->benchmark_plan->name.size,
          selection->benchmark_plan->name.data);
      break;
    }
    if (logical_sample->work_item_index >= work_plan->work_item_count) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "comparison logical sample references work item %" PRIhsz
          " but the work plan only has %" PRIhsz " work items",
          logical_sample->work_item_index, work_plan->work_item_count);
      break;
    }
    const iree_benchmark_loom_work_item_t* work_item =
        &work_plan->work_items[logical_sample->work_item_index];
    if (work_item->kind != IREE_BENCHMARK_LOOM_WORK_ITEM_DISPATCH_SAMPLE) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "interleaved comparison benchmark `%.*s` must "
                                "use dispatch_complete work items",
                                (int)selection->benchmark_plan->name.size,
                                selection->benchmark_plan->name.data);
      break;
    }
    candidates[i].selection = selection;
    candidates[i].module = options->module_plan->module;
    candidates[i].work_item_index = logical_sample->work_item_index;
    candidates[i].sample_compilation = logical_sample->sample_compilation;
    candidates[i].begin_sample = logical_sample->begin_benchmark_sample;
    candidates[i].end_sample = logical_sample->end_benchmark_sample;
    const iree_host_size_t sample_capacity =
        iree_benchmark_loom_dispatch_comparison_sample_capacity(
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

static void iree_benchmark_loom_dispatch_comparison_candidate_deinitialize(
    iree_benchmark_loom_dispatch_comparison_candidate_t* candidate,
    iree_allocator_t allocator) {
  iree_allocator_free(allocator, candidate->p50_samples);
  iree_allocator_free(allocator, candidate->p90_samples);
  *candidate = (iree_benchmark_loom_dispatch_comparison_candidate_t){0};
}

static void iree_benchmark_loom_dispatch_comparison_candidates_deinitialize(
    iree_benchmark_loom_dispatch_comparison_candidate_t* candidates,
    iree_host_size_t candidate_count, iree_allocator_t allocator) {
  if (candidates == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    iree_benchmark_loom_dispatch_comparison_candidate_deinitialize(
        &candidates[i], allocator);
  }
  iree_allocator_free(allocator, candidates);
}

static iree_status_t iree_benchmark_loom_run_comparison_window(
    const iree_benchmark_loom_comparison_execution_options_t* options,
    iree_benchmark_loom_dispatch_comparison_candidate_t* candidate,
    iree_benchmark_loom_dispatch_compile_context_t* compile_contexts,
    const iree_benchmark_loom_candidate_identity_t* baseline,
    iree_string_view_t comparison_group, iree_string_view_t method,
    iree_host_size_t order_index, iree_host_size_t repetition_index,
    char schedule_token, iree_host_size_t* inout_failed_benchmark_count) {
  if (!candidate->runnable) {
    return iree_ok_status();
  }
  iree_benchmark_loom_benchmark_policy_t measurement_policy =
      candidate->selection->policy;
  const bool profile_suppressed =
      iree_any_bit_set(measurement_policy.hal_options.flags,
                       LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH);
  measurement_policy.hal_options.flags &=
      ~LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH;

  const iree_benchmark_loom_work_plan_t* work_plan = options->work_plan;
  const iree_benchmark_loom_work_item_t* work_item =
      &work_plan->work_items[candidate->work_item_index];
  iree_benchmark_loom_dispatch_compile_context_t* compile_context =
      &compile_contexts[work_item->dispatch_compile_item_index];

  iree_benchmark_loom_benchmark_result_t benchmark_result = {0};
  iree_status_t status = iree_ok_status();
  if (compile_context->uses_sequence) {
    status = iree_benchmark_loom_run_hal_sequence_benchmark_sample(
        options->run, &candidate->selection->identity, options->module_plan,
        candidate->selection->benchmark_plan, candidate->selection->case_plan,
        &measurement_policy, options->benchmark_options, options->hal_context,
        &compile_context->hal_sequence,
        &compile_context->benchmark_materializer, candidate->sample_compilation,
        candidate->begin_sample, options->host_allocator, &benchmark_result);
  } else {
    status = iree_benchmark_loom_run_hal_benchmark_sample(
        options->run, &candidate->selection->identity, options->module_plan,
        candidate->selection->benchmark_plan, candidate->selection->case_plan,
        &measurement_policy, options->benchmark_options,
        &compile_context->hal_provider,
        &compile_context->benchmark_materializer, candidate->begin_sample,
        options->host_allocator, &benchmark_result);
  }
  if (iree_status_is_ok(status) &&
      (!benchmark_result.executed || !benchmark_result.passed)) {
    ++*inout_failed_benchmark_count;
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_comparison_candidate_record_timing(
        candidate, &benchmark_result);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_event_sink_emit_benchmark_repetition(
        options->event_sink, options->run, candidate, baseline,
        comparison_group, method, order_index, repetition_index, schedule_token,
        profile_suppressed, &benchmark_result);
  }
  return status;
}

static iree_status_t iree_benchmark_loom_prepare_dispatch_comparison_work(
    const iree_benchmark_loom_comparison_execution_options_t* options,
    const iree_benchmark_loom_dispatch_setup_options_t* dispatch_options,
    iree_benchmark_loom_dispatch_compile_context_t* compile_contexts,
    iree_benchmark_loom_dispatch_work_item_state_t* work_item_states,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  const iree_benchmark_loom_work_plan_t* work_plan = options->work_plan;
  for (iree_host_size_t work_item_index = 0;
       work_item_index < work_plan->work_item_count; ++work_item_index) {
    const iree_benchmark_loom_work_item_t* work_item =
        &work_plan->work_items[work_item_index];
    if (work_item->kind != IREE_BENCHMARK_LOOM_WORK_ITEM_DISPATCH_SAMPLE) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "interleaved comparison work item %" PRIhsz
                              " is not a dispatch_complete sample",
                              work_item_index);
    }
    if (work_item->dispatch_compile_item_index >=
        work_plan->dispatch_compile_item_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "dispatch comparison work item %" PRIhsz
          " references compile item %" PRIhsz
          " but the work plan only has %" PRIhsz " compile items",
          work_item_index, work_item->dispatch_compile_item_index,
          work_plan->dispatch_compile_item_count);
    }
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_prepare_dispatch_work_item(
        dispatch_options, work_item,
        &compile_contexts[work_item->dispatch_compile_item_index],
        inout_correctness_sample_count, inout_correctness_failed_sample_count,
        inout_failed_benchmark_count, &work_item_states[work_item_index]));
  }
  return iree_ok_status();
}

static void iree_benchmark_loom_apply_dispatch_comparison_work_state(
    const iree_benchmark_loom_dispatch_work_item_state_t* work_item_states,
    iree_benchmark_loom_dispatch_comparison_candidate_t* candidates,
    iree_host_size_t candidate_count) {
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    iree_benchmark_loom_dispatch_comparison_candidate_t* candidate =
        &candidates[i];
    const iree_benchmark_loom_dispatch_work_item_state_t* state =
        &work_item_states[candidate->work_item_index];
    candidate->correctness_sample_count = state->correctness_sample_count;
    candidate->correctness_failed_sample_count =
        state->correctness_failed_sample_count;
    candidate->runnable = state->runnable;
  }
}

iree_status_t iree_benchmark_loom_run_dispatch_comparison(
    const iree_benchmark_loom_comparison_execution_options_t* options,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  const iree_benchmark_loom_work_plan_t* work_plan = options->work_plan;
  const iree_benchmark_loom_options_t* benchmark_options =
      options->benchmark_options;
  const iree_host_size_t selection_count = work_plan->selected_benchmark_count;
  iree_benchmark_loom_dispatch_comparison_candidate_t* candidates = NULL;
  iree_benchmark_loom_dispatch_compile_context_t* compile_contexts = NULL;
  iree_benchmark_loom_dispatch_work_item_state_t* work_item_states = NULL;
  iree_status_t status =
      iree_benchmark_loom_initialize_dispatch_comparison_candidates(
          options, &candidates);
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_validate_comparison_samples(
        options->module_plan->module, candidates, selection_count);
  }
  if (iree_status_is_ok(status) &&
      work_plan->dispatch_compile_item_count != 0) {
    status = iree_allocator_malloc_array(
        options->host_allocator, work_plan->dispatch_compile_item_count,
        sizeof(*compile_contexts), (void**)&compile_contexts);
    if (iree_status_is_ok(status)) {
      memset(
          compile_contexts, 0,
          work_plan->dispatch_compile_item_count * sizeof(*compile_contexts));
    }
  }
  if (iree_status_is_ok(status) && work_plan->work_item_count != 0) {
    status = iree_allocator_malloc_array(
        options->host_allocator, work_plan->work_item_count,
        sizeof(*work_item_states), (void**)&work_item_states);
    if (iree_status_is_ok(status)) {
      memset(work_item_states, 0,
             work_plan->work_item_count * sizeof(*work_item_states));
    }
  }
  const iree_benchmark_loom_dispatch_setup_options_t dispatch_options = {
      .run = options->run,
      .module_plan = options->module_plan,
      .work_plan = work_plan,
      .benchmark_options = benchmark_options,
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
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_prepare_dispatch_comparison_work(
        options, &dispatch_options, compile_contexts, work_item_states,
        inout_correctness_sample_count, inout_correctness_failed_sample_count,
        inout_failed_benchmark_count);
  }
  if (iree_status_is_ok(status)) {
    iree_benchmark_loom_apply_dispatch_comparison_work_state(
        work_item_states, candidates, selection_count);
  }

  const iree_benchmark_loom_candidate_identity_t* baseline = NULL;
  iree_string_view_t comparison_group = iree_string_view_empty();
  iree_string_view_t method = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    baseline = &work_plan->selected_benchmarks[0].identity;
    comparison_group = work_plan->selected_benchmarks[0].benchmark_plan->name;
    method = iree_benchmark_loom_interleave_mode_name(
        benchmark_options->interleave_mode);
  }
  iree_host_size_t order_index = 0;
  if (iree_status_is_ok(status) && benchmark_options->interleave_mode ==
                                       IREE_BENCHMARK_LOOM_INTERLEAVE_ABABA) {
    for (iree_host_size_t candidate_index = 1;
         iree_status_is_ok(status) && candidate_index < selection_count;
         ++candidate_index) {
      status = iree_benchmark_loom_run_comparison_window(
          options, &candidates[0], compile_contexts, baseline, comparison_group,
          method, order_index++, /*repetition_index=*/0, 'A',
          inout_failed_benchmark_count);
      for (iree_host_size_t repetition_index = 0;
           iree_status_is_ok(status) &&
           repetition_index < benchmark_options->repetitions;
           ++repetition_index) {
        status = iree_benchmark_loom_run_comparison_window(
            options, &candidates[candidate_index], compile_contexts, baseline,
            comparison_group, method, order_index++, repetition_index,
            (char)('A' + candidate_index), inout_failed_benchmark_count);
        if (iree_status_is_ok(status)) {
          status = iree_benchmark_loom_run_comparison_window(
              options, &candidates[0], compile_contexts, baseline,
              comparison_group, method, order_index++, repetition_index + 1,
              'A', inout_failed_benchmark_count);
        }
      }
    }
  } else if (iree_status_is_ok(status) &&
             benchmark_options->interleave_mode ==
                 IREE_BENCHMARK_LOOM_INTERLEAVE_ROUND_ROBIN) {
    for (iree_host_size_t repetition_index = 0;
         iree_status_is_ok(status) &&
         repetition_index < benchmark_options->repetitions;
         ++repetition_index) {
      for (iree_host_size_t candidate_index = 0;
           iree_status_is_ok(status) && candidate_index < selection_count;
           ++candidate_index) {
        status = iree_benchmark_loom_run_comparison_window(
            options, &candidates[candidate_index], compile_contexts, baseline,
            comparison_group, method, order_index++, repetition_index,
            (char)('A' + candidate_index), inout_failed_benchmark_count);
      }
    }
  }

  for (iree_host_size_t candidate_index = 1;
       iree_status_is_ok(status) && candidate_index < selection_count;
       ++candidate_index) {
    status = iree_benchmark_loom_event_sink_emit_comparison(
        options->event_sink, options->run, &candidates[0],
        &candidates[candidate_index], comparison_group, method);
  }

  if (compile_contexts != NULL) {
    for (iree_host_size_t i = 0; i < work_plan->dispatch_compile_item_count;
         ++i) {
      iree_benchmark_loom_dispatch_compile_context_deinitialize(
          &compile_contexts[i]);
    }
  }
  iree_allocator_free(options->host_allocator, work_item_states);
  iree_allocator_free(options->host_allocator, compile_contexts);
  iree_benchmark_loom_dispatch_comparison_candidates_deinitialize(
      candidates, selection_count, options->host_allocator);
  return status;
}
