// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/executor.h"

#include <string.h>

#include "loom/util/json.h"

void loom_testbench_case_execution_options_initialize(
    loom_testbench_case_execution_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  loom_testbench_value_materializer_options_initialize(
      &out_options->materializer);
  loom_testbench_invocation_options_initialize(&out_options->invocation);
  loom_testbench_expectation_options_initialize(&out_options->expectation);
}

iree_status_t loom_testbench_prepare_case_execution(
    const loom_testbench_case_execution_options_t* options,
    const loom_testbench_module_plan_t* module_plan,
    iree_host_size_t case_index, iree_arena_allocator_t* arena,
    loom_testbench_prepared_case_t* out_prepared_case) {
  memset(out_prepared_case, 0, sizeof(*out_prepared_case));

  if (case_index >= module_plan->case_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "case index %" PRIhsz
                            " exceeds module case count %" PRIhsz,
                            case_index, module_plan->case_count);
  }
  const loom_testbench_case_plan_t* case_plan = &module_plan->cases[case_index];
  if (case_plan->issue_count != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "case `%.*s` has %" PRIhsz
                            " unresolved planning issue(s)",
                            (int)case_plan->name.size, case_plan->name.data,
                            case_plan->issue_count);
  }

  out_prepared_case->module = module_plan->module;
  out_prepared_case->case_plan = case_plan;
  IREE_RETURN_IF_ERROR(loom_testbench_prepare_case_invocations(
      &options->invocation, case_plan, arena,
      &out_prepared_case->invocation_schedule));
  return loom_testbench_prepare_case_expectations(
      &options->expectation, case_plan, arena,
      &out_prepared_case->expectation_schedule);
}

iree_status_t loom_testbench_case_executor_initialize(
    const loom_testbench_prepared_case_t* prepared_case,
    const loom_testbench_case_execution_options_t* options,
    loom_testbench_case_executor_t* out_executor) {
  memset(out_executor, 0, sizeof(*out_executor));

  iree_allocator_t host_allocator = options->materializer.host_allocator;
  if (iree_allocator_is_null(host_allocator)) {
    host_allocator = iree_allocator_system();
  }
  out_executor->prepared_case = prepared_case;
  out_executor->materializer_options = options->materializer;
  out_executor->materializer_options.host_allocator = host_allocator;

  iree_status_t status = loom_testbench_value_table_initialize(
      prepared_case->module, prepared_case->case_plan, host_allocator,
      &out_executor->value_table);
  if (iree_status_is_ok(status)) {
    status = loom_testbench_invocation_executor_initialize(
        &prepared_case->invocation_schedule, host_allocator,
        &out_executor->invocation_executor);
  }
  if (iree_status_is_ok(status)) {
    status = loom_testbench_expectation_report_initialize(
        prepared_case->expectation_schedule.expectation_count, host_allocator,
        &out_executor->expectation_report);
  }
  if (!iree_status_is_ok(status)) {
    loom_testbench_case_executor_deinitialize(out_executor);
  }
  return status;
}

void loom_testbench_case_executor_deinitialize(
    loom_testbench_case_executor_t* executor) {
  if (!executor) {
    return;
  }
  loom_testbench_expectation_report_deinitialize(&executor->expectation_report);
  loom_testbench_invocation_executor_deinitialize(
      &executor->invocation_executor);
  loom_testbench_value_table_deinitialize(&executor->value_table);
  memset(executor, 0, sizeof(*executor));
}

iree_status_t loom_testbench_run_case_sample(
    loom_testbench_case_executor_t* executor, iree_host_size_t sample_ordinal,
    loom_testbench_case_sample_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));

  const loom_testbench_case_plan_t* case_plan =
      executor->prepared_case->case_plan;
  loom_testbench_value_table_reset(&executor->value_table);
  loom_testbench_expectation_report_reset(&executor->expectation_report);

  IREE_RETURN_IF_ERROR(loom_testbench_materialize_case_sample(
      &executor->materializer_options, case_plan, sample_ordinal,
      &executor->value_table));
  IREE_RETURN_IF_ERROR(loom_testbench_run_case_invocations(
      &executor->invocation_executor, &executor->value_table));
  IREE_RETURN_IF_ERROR(loom_testbench_evaluate_case_expectations(
      &executor->prepared_case->expectation_schedule, &executor->value_table,
      &executor->expectation_report));

  bool case_failed = executor->expectation_report.failure_count != 0;
  IREE_RETURN_IF_ERROR(loom_testbench_write_case_files(
      &executor->materializer_options, case_plan, &executor->value_table,
      case_failed));

  out_result->case_plan = case_plan;
  out_result->sample_ordinal = sample_ordinal;
  out_result->passed = !case_failed;
  out_result->expectation_report = &executor->expectation_report;
  return iree_ok_status();
}

iree_status_t loom_testbench_case_sample_result_write_json(
    const loom_testbench_case_sample_result_t* result,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\"case\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(stream, result->case_plan->name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"sample_ordinal\":%" PRIhsz, result->sample_ordinal));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"passed\":"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      stream, result->passed ? "true" : "false"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"expectations\":"));
  IREE_RETURN_IF_ERROR(loom_testbench_expectation_report_write_json(
      result->expectation_report, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  return iree_ok_status();
}
