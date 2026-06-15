// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/event.h"

#include "loom/tools/iree-benchmark-loom/profile_report.h"
#include "loom/tools/iree-benchmark-loom/report.h"

iree_status_t iree_benchmark_loom_event_sink_emit(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_event_t* event) {
  IREE_ASSERT_ARGUMENT(sink);
  IREE_ASSERT_ARGUMENT(event);
  if (sink->emit == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "benchmark event sink has no emit callback");
  }
  return sink->emit(sink->user_data, event);
}

iree_status_t iree_benchmark_loom_event_sink_emit_run(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run, bool dry_run,
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode) {
  IREE_ASSERT_ARGUMENT(run);
  return iree_benchmark_loom_event_sink_emit(
      sink, &(iree_benchmark_loom_event_t){
                .kind = IREE_BENCHMARK_LOOM_EVENT_RUN,
                .run =
                    {
                        .run = run,
                        .dry_run = dry_run,
                        .sample_compilation_mode = sample_compilation_mode,
                    },
            });
}

iree_status_t iree_benchmark_loom_event_sink_emit_plan(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run, const loom_module_t* module,
    const iree_benchmark_loom_selected_benchmark_t* selection,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode) {
  IREE_ASSERT_ARGUMENT(run);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(selection);
  IREE_ASSERT_ARGUMENT(options);
  return iree_benchmark_loom_event_sink_emit(
      sink, &(iree_benchmark_loom_event_t){
                .kind = IREE_BENCHMARK_LOOM_EVENT_PLAN,
                .plan =
                    {
                        .run = run,
                        .module = module,
                        .selection = selection,
                        .options = options,
                        .sample_compilation_mode = sample_compilation_mode,
                    },
            });
}

iree_status_t iree_benchmark_loom_event_sink_emit_work_plan(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run, const loom_module_t* module,
    const iree_benchmark_loom_work_plan_t* work_plan) {
  IREE_ASSERT_ARGUMENT(run);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(work_plan);
  return iree_benchmark_loom_event_sink_emit(
      sink, &(iree_benchmark_loom_event_t){
                .kind = IREE_BENCHMARK_LOOM_EVENT_WORK_PLAN,
                .work_plan =
                    {
                        .run = run,
                        .module = module,
                        .work_plan = work_plan,
                    },
            });
}

iree_status_t iree_benchmark_loom_event_sink_emit_summary(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_artifact_bundle_t* artifact_bundle,
    iree_host_size_t planned_case_count,
    iree_host_size_t planned_benchmark_count,
    iree_host_size_t selected_benchmark_count,
    iree_host_size_t logical_sample_count, iree_host_size_t work_item_count,
    iree_host_size_t failure_count, iree_host_size_t failed_benchmark_count,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count, bool dry_run,
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode) {
  IREE_ASSERT_ARGUMENT(run);
  return iree_benchmark_loom_event_sink_emit(
      sink, &(iree_benchmark_loom_event_t){
                .kind = IREE_BENCHMARK_LOOM_EVENT_SUMMARY,
                .summary =
                    {
                        .run = run,
                        .artifact_bundle = artifact_bundle,
                        .planned_case_count = planned_case_count,
                        .planned_benchmark_count = planned_benchmark_count,
                        .selected_benchmark_count = selected_benchmark_count,
                        .logical_sample_count = logical_sample_count,
                        .work_item_count = work_item_count,
                        .failure_count = failure_count,
                        .failed_benchmark_count = failed_benchmark_count,
                        .correctness_sample_count = correctness_sample_count,
                        .correctness_failed_sample_count =
                            correctness_failed_sample_count,
                        .dry_run = dry_run,
                        .sample_compilation_mode = sample_compilation_mode,
                    },
            });
}

iree_status_t iree_benchmark_loom_event_sink_emit_device(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run,
    iree_benchmark_loom_hal_context_t* context) {
  IREE_ASSERT_ARGUMENT(run);
  IREE_ASSERT_ARGUMENT(context);
  return iree_benchmark_loom_event_sink_emit(
      sink, &(iree_benchmark_loom_event_t){
                .kind = IREE_BENCHMARK_LOOM_EVENT_DEVICE,
                .device =
                    {
                        .run = run,
                        .context = context,
                    },
            });
}

iree_status_t iree_benchmark_loom_event_sink_emit_compile(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_hal_actual_provider_t* provider) {
  IREE_ASSERT_ARGUMENT(run);
  IREE_ASSERT_ARGUMENT(candidate);
  IREE_ASSERT_ARGUMENT(benchmark_plan);
  IREE_ASSERT_ARGUMENT(case_plan);
  IREE_ASSERT_ARGUMENT(provider);
  return iree_benchmark_loom_event_sink_emit(
      sink, &(iree_benchmark_loom_event_t){
                .kind = IREE_BENCHMARK_LOOM_EVENT_COMPILE,
                .compile =
                    {
                        .run = run,
                        .candidate = candidate,
                        .benchmark_plan = benchmark_plan,
                        .case_plan = case_plan,
                        .provider = provider,
                    },
            });
}

iree_status_t iree_benchmark_loom_event_sink_emit_sample(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    iree_string_view_t sample_compilation,
    iree_host_size_t benchmark_sample_ordinal,
    iree_host_size_t case_sample_ordinal,
    const loom_testbench_case_sample_result_t* sample_result) {
  IREE_ASSERT_ARGUMENT(run);
  IREE_ASSERT_ARGUMENT(candidate);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(benchmark_plan);
  IREE_ASSERT_ARGUMENT(case_plan);
  IREE_ASSERT_ARGUMENT(sample_result);
  return iree_benchmark_loom_event_sink_emit(
      sink, &(iree_benchmark_loom_event_t){
                .kind = IREE_BENCHMARK_LOOM_EVENT_SAMPLE,
                .sample =
                    {
                        .run = run,
                        .candidate = candidate,
                        .work_item_index = work_item_index,
                        .module = module,
                        .benchmark_plan = benchmark_plan,
                        .case_plan = case_plan,
                        .sample_compilation = sample_compilation,
                        .benchmark_sample_ordinal = benchmark_sample_ordinal,
                        .case_sample_ordinal = case_sample_ordinal,
                        .sample_result = sample_result,
                    },
            });
}

iree_status_t iree_benchmark_loom_event_sink_emit_benchmark_result(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count) {
  IREE_ASSERT_ARGUMENT(run);
  IREE_ASSERT_ARGUMENT(candidate);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(benchmark_plan);
  IREE_ASSERT_ARGUMENT(case_plan);
  IREE_ASSERT_ARGUMENT(policy);
  IREE_ASSERT_ARGUMENT(benchmark_result);
  return iree_benchmark_loom_event_sink_emit(
      sink, &(iree_benchmark_loom_event_t){
                .kind = IREE_BENCHMARK_LOOM_EVENT_BENCHMARK_RESULT,
                .benchmark_result =
                    {
                        .run = run,
                        .candidate = candidate,
                        .work_item_index = work_item_index,
                        .module = module,
                        .benchmark_plan = benchmark_plan,
                        .case_plan = case_plan,
                        .policy = policy,
                        .benchmark_result = benchmark_result,
                        .correctness_sample_count = correctness_sample_count,
                        .correctness_failed_sample_count =
                            correctness_failed_sample_count,
                    },
            });
}

iree_status_t iree_benchmark_loom_event_sink_emit_profile(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result) {
  IREE_ASSERT_ARGUMENT(run);
  IREE_ASSERT_ARGUMENT(candidate);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(benchmark_plan);
  IREE_ASSERT_ARGUMENT(case_plan);
  IREE_ASSERT_ARGUMENT(policy);
  IREE_ASSERT_ARGUMENT(benchmark_result);
  return iree_benchmark_loom_event_sink_emit(
      sink, &(iree_benchmark_loom_event_t){
                .kind = IREE_BENCHMARK_LOOM_EVENT_PROFILE,
                .profile =
                    {
                        .run = run,
                        .candidate = candidate,
                        .work_item_index = work_item_index,
                        .module = module,
                        .benchmark_plan = benchmark_plan,
                        .case_plan = case_plan,
                        .policy = policy,
                        .benchmark_result = benchmark_result,
                    },
            });
}

iree_status_t iree_benchmark_loom_event_sink_emit_failure(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run, iree_string_view_t stage,
    iree_string_view_t kind, iree_string_view_t message,
    const iree_benchmark_loom_diagnostic_capture_t* diagnostics) {
  IREE_ASSERT_ARGUMENT(run);
  return iree_benchmark_loom_event_sink_emit(
      sink, &(iree_benchmark_loom_event_t){
                .kind = IREE_BENCHMARK_LOOM_EVENT_FAILURE,
                .failure =
                    {
                        .run = run,
                        .stage = stage,
                        .kind = kind,
                        .message = message,
                        .diagnostics = diagnostics,
                    },
            });
}

iree_status_t iree_benchmark_loom_event_sink_emit_benchmark_repetition(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_dispatch_comparison_candidate_t* candidate,
    const iree_benchmark_loom_candidate_identity_t* baseline,
    iree_string_view_t comparison_group, iree_string_view_t method,
    iree_host_size_t order_index, iree_host_size_t repetition_index,
    char schedule_token, bool profile_suppressed,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result) {
  IREE_ASSERT_ARGUMENT(run);
  IREE_ASSERT_ARGUMENT(candidate);
  IREE_ASSERT_ARGUMENT(baseline);
  IREE_ASSERT_ARGUMENT(benchmark_result);
  return iree_benchmark_loom_event_sink_emit(
      sink, &(iree_benchmark_loom_event_t){
                .kind = IREE_BENCHMARK_LOOM_EVENT_BENCHMARK_REPETITION,
                .benchmark_repetition =
                    {
                        .run = run,
                        .candidate = candidate,
                        .baseline = baseline,
                        .comparison_group = comparison_group,
                        .method = method,
                        .order_index = order_index,
                        .repetition_index = repetition_index,
                        .schedule_token = schedule_token,
                        .profile_suppressed = profile_suppressed,
                        .benchmark_result = benchmark_result,
                    },
            });
}

iree_status_t iree_benchmark_loom_event_sink_emit_comparison(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_dispatch_comparison_candidate_t* baseline,
    const iree_benchmark_loom_dispatch_comparison_candidate_t* candidate,
    iree_string_view_t comparison_group, iree_string_view_t method) {
  IREE_ASSERT_ARGUMENT(run);
  IREE_ASSERT_ARGUMENT(baseline);
  IREE_ASSERT_ARGUMENT(candidate);
  return iree_benchmark_loom_event_sink_emit(
      sink, &(iree_benchmark_loom_event_t){
                .kind = IREE_BENCHMARK_LOOM_EVENT_COMPARISON,
                .comparison =
                    {
                        .run = run,
                        .baseline = baseline,
                        .candidate = candidate,
                        .comparison_group = comparison_group,
                        .method = method,
                    },
            });
}

static iree_status_t iree_benchmark_loom_jsonl_event_sink_emit(
    void* user_data, const iree_benchmark_loom_event_t* event) {
  iree_benchmark_loom_jsonl_event_sink_t* adapter =
      (iree_benchmark_loom_jsonl_event_sink_t*)user_data;
  iree_benchmark_loom_jsonl_sink_t* jsonl_sink = adapter->jsonl_sink;
  switch (event->kind) {
    case IREE_BENCHMARK_LOOM_EVENT_RUN:
      return iree_benchmark_loom_jsonl_sink_end(
          jsonl_sink, iree_benchmark_loom_append_run_row(
                          event->run.run, event->run.dry_run,
                          event->run.sample_compilation_mode,
                          iree_benchmark_loom_jsonl_sink_begin(jsonl_sink)));
    case IREE_BENCHMARK_LOOM_EVENT_PLAN:
      return iree_benchmark_loom_jsonl_sink_end(
          jsonl_sink,
          iree_benchmark_loom_append_plan_row(
              event->plan.run, &event->plan.selection->identity,
              event->plan.module, event->plan.selection->benchmark_plan,
              event->plan.selection->case_plan, &event->plan.selection->policy,
              event->plan.options, event->plan.sample_compilation_mode,
              jsonl_sink->host_allocator,
              iree_benchmark_loom_jsonl_sink_begin(jsonl_sink)));
    case IREE_BENCHMARK_LOOM_EVENT_SUMMARY:
      return iree_benchmark_loom_jsonl_sink_end(
          jsonl_sink,
          iree_benchmark_loom_append_summary_row(
              event->summary.run, event->summary.artifact_bundle,
              event->summary.planned_case_count,
              event->summary.planned_benchmark_count,
              event->summary.selected_benchmark_count,
              event->summary.logical_sample_count,
              event->summary.work_item_count, event->summary.failure_count,
              event->summary.failed_benchmark_count,
              event->summary.correctness_sample_count,
              event->summary.correctness_failed_sample_count,
              event->summary.dry_run, event->summary.sample_compilation_mode,
              iree_benchmark_loom_jsonl_sink_begin(jsonl_sink)));
    case IREE_BENCHMARK_LOOM_EVENT_DEVICE:
      return iree_benchmark_loom_jsonl_sink_end(
          jsonl_sink, iree_benchmark_loom_append_device_row(
                          event->device.run, event->device.context,
                          &adapter->device_row_state,
                          iree_benchmark_loom_jsonl_sink_begin(jsonl_sink)));
    case IREE_BENCHMARK_LOOM_EVENT_COMPILE:
      return iree_benchmark_loom_jsonl_sink_end(
          jsonl_sink, iree_benchmark_loom_append_compile_row(
                          event->compile.run, event->compile.candidate,
                          event->compile.benchmark_plan,
                          event->compile.case_plan, event->compile.provider,
                          iree_benchmark_loom_jsonl_sink_begin(jsonl_sink)));
    case IREE_BENCHMARK_LOOM_EVENT_SAMPLE:
      return iree_benchmark_loom_jsonl_sink_end(
          jsonl_sink,
          iree_benchmark_loom_append_sample_row(
              event->sample.run, event->sample.candidate,
              event->sample.work_item_index, event->sample.module,
              event->sample.benchmark_plan, event->sample.case_plan,
              event->sample.sample_compilation,
              event->sample.benchmark_sample_ordinal,
              event->sample.case_sample_ordinal, event->sample.sample_result,
              iree_benchmark_loom_jsonl_sink_begin(jsonl_sink)));
    case IREE_BENCHMARK_LOOM_EVENT_BENCHMARK_RESULT:
      return iree_benchmark_loom_jsonl_sink_end(
          jsonl_sink,
          iree_benchmark_loom_append_benchmark_result(
              event->benchmark_result.run, event->benchmark_result.candidate,
              event->benchmark_result.work_item_index,
              event->benchmark_result.module,
              event->benchmark_result.benchmark_plan,
              event->benchmark_result.case_plan, event->benchmark_result.policy,
              event->benchmark_result.benchmark_result,
              event->benchmark_result.correctness_sample_count,
              event->benchmark_result.correctness_failed_sample_count,
              iree_benchmark_loom_jsonl_sink_begin(jsonl_sink)));
    case IREE_BENCHMARK_LOOM_EVENT_PROFILE:
      return iree_benchmark_loom_jsonl_sink_end(
          jsonl_sink,
          iree_benchmark_loom_append_profile_row(
              event->profile.run, event->profile.candidate,
              event->profile.work_item_index, event->profile.module,
              event->profile.benchmark_plan, event->profile.case_plan,
              event->profile.policy, event->profile.benchmark_result,
              jsonl_sink->host_allocator,
              iree_benchmark_loom_jsonl_sink_begin(jsonl_sink)));
    case IREE_BENCHMARK_LOOM_EVENT_FAILURE:
      return iree_benchmark_loom_jsonl_sink_end(
          jsonl_sink,
          iree_benchmark_loom_append_failure_row(
              event->failure.run, event->failure.stage, event->failure.kind,
              event->failure.message, event->failure.diagnostics,
              iree_benchmark_loom_jsonl_sink_begin(jsonl_sink)));
    case IREE_BENCHMARK_LOOM_EVENT_BENCHMARK_REPETITION:
      return iree_benchmark_loom_jsonl_sink_end(
          jsonl_sink, iree_benchmark_loom_append_benchmark_repetition_row(
                          event->benchmark_repetition.run,
                          event->benchmark_repetition.candidate,
                          event->benchmark_repetition.baseline,
                          event->benchmark_repetition.comparison_group,
                          event->benchmark_repetition.method,
                          event->benchmark_repetition.order_index,
                          event->benchmark_repetition.repetition_index,
                          event->benchmark_repetition.schedule_token,
                          event->benchmark_repetition.profile_suppressed,
                          event->benchmark_repetition.benchmark_result,
                          iree_benchmark_loom_jsonl_sink_begin(jsonl_sink)));
    case IREE_BENCHMARK_LOOM_EVENT_COMPARISON:
      return iree_benchmark_loom_jsonl_sink_end(
          jsonl_sink,
          iree_benchmark_loom_append_comparison_row(
              event->comparison.run, event->comparison.baseline,
              event->comparison.candidate, event->comparison.comparison_group,
              event->comparison.method,
              iree_benchmark_loom_jsonl_sink_begin(jsonl_sink)));
    case IREE_BENCHMARK_LOOM_EVENT_WORK_PLAN:
      return iree_ok_status();
    case IREE_BENCHMARK_LOOM_EVENT_NONE:
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported benchmark event kind %d",
                              (int)event->kind);
  }
}

void iree_benchmark_loom_jsonl_event_sink_initialize(
    iree_benchmark_loom_jsonl_sink_t* jsonl_sink,
    iree_benchmark_loom_jsonl_event_sink_t* out_adapter,
    iree_benchmark_loom_event_sink_t* out_sink) {
  IREE_ASSERT_ARGUMENT(jsonl_sink);
  IREE_ASSERT_ARGUMENT(out_adapter);
  IREE_ASSERT_ARGUMENT(out_sink);
  *out_adapter = (iree_benchmark_loom_jsonl_event_sink_t){
      .jsonl_sink = jsonl_sink,
  };
  *out_sink = (iree_benchmark_loom_event_sink_t){
      .emit = iree_benchmark_loom_jsonl_event_sink_emit,
      .user_data = out_adapter,
  };
}

void iree_benchmark_loom_jsonl_event_sink_deinitialize(
    iree_benchmark_loom_jsonl_event_sink_t* adapter) {
  IREE_ASSERT_ARGUMENT(adapter);
  *adapter = (iree_benchmark_loom_jsonl_event_sink_t){0};
}
