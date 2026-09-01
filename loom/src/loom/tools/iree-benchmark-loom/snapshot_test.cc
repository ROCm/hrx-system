// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/snapshot.h"

#include "iree/base/internal/json.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/tools/iree-benchmark-loom/hal_actual.h"
#include "loom/tools/iree-benchmark-loom/launch_evidence.h"

namespace loom {
namespace {

static const loom_sanitizer_options_t kNoSanitizer = {};

static iree_string_view_t SnapshotJson(
    iree_benchmark_loom_snapshot_sink_t* snapshot,
    iree_string_builder_t* storage) {
  iree_string_builder_reset(storage);
  IREE_EXPECT_OK(
      iree_benchmark_loom_snapshot_sink_append_json(snapshot, storage));
  return iree_string_builder_view(storage);
}

static iree_string_view_t ParseJsonDocument(iree_string_view_t json) {
  iree_string_view_t cursor = json;
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_consume_value(&cursor, &value));
  IREE_EXPECT_OK(iree_json_consume_insignificant(&cursor));
  EXPECT_TRUE(iree_string_view_is_empty(cursor));
  return value;
}

static iree_string_view_t LookupObject(iree_string_view_t object,
                                       iree_string_view_t key) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_lookup_object_value(object, key, &value));
  return value;
}

static iree_string_view_t TryLookupObject(iree_string_view_t object,
                                          iree_string_view_t key) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_try_lookup_object_value(object, key, &value));
  return value;
}

static iree_string_view_t FirstArrayElement(iree_string_view_t array) {
  iree_string_view_t element = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_array_get(array, 0, &element));
  return element;
}

static bool JsonArrayContainsString(iree_string_view_t array,
                                    iree_string_view_t expected) {
  iree_host_size_t count = 0;
  IREE_EXPECT_OK(iree_json_array_length(array, &count));
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_string_view_t element = iree_string_view_empty();
    IREE_EXPECT_OK(iree_json_array_get(array, i, &element));
    if (iree_string_view_equal(element, expected)) {
      return true;
    }
  }
  return false;
}

TEST(BenchmarkSnapshotSinkTest, AggregatesDeduplicatedWorkItems) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_benchmark_loom_snapshot_sink_t snapshot = {};
  IREE_ASSERT_OK(
      iree_benchmark_loom_snapshot_sink_initialize(allocator, &snapshot));
  iree_benchmark_loom_event_sink_t event_sink = {};
  iree_benchmark_loom_snapshot_event_sink_initialize(&snapshot, &event_sink);

  iree_benchmark_loom_run_identity_t run = {};
  run.run_id = IREE_SV("run");
  run.source = IREE_SV("input.loom");
  run.results_path = IREE_SV("-");
  run.file_output_dir = IREE_SV("/tmp/loom");
  iree_benchmark_loom_candidate_identity_t candidate0 = {};
  candidate0.candidate_id = IREE_SV("c0");
  candidate0.candidate_index = 0;
  iree_benchmark_loom_candidate_identity_t candidate1 = {};
  candidate1.candidate_id = IREE_SV("c1");
  candidate1.candidate_index = 1;
  loom_module_t module = {};
  loom_testbench_benchmark_plan_t benchmark_plan = {};
  benchmark_plan.name = IREE_SV("kernel_latency");
  loom_testbench_case_plan_t case_plan = {};
  case_plan.name = IREE_SV("kernel_case");
  iree_benchmark_loom_benchmark_policy_t policy = {};
  policy.measure_kind = IREE_BENCHMARK_LOOM_MEASURE_CASE_END_TO_END;
  policy.measure = IREE_SV("case_end_to_end");
  iree_benchmark_loom_benchmark_result_t result = {};
  result.executed = true;
  result.passed = true;
  result.has_sample_ordinal = true;
  result.sample_ordinal = 0;
  result.samples_per_iteration = 1;
  result.timing.count = 3;
  result.timing.total_ns = 90;
  result.timing.minimum_ns = 20;
  result.timing.maximum_ns = 40;
  result.timing.mean_ns = 30.0;
  result.timing.p50_ns = 30;
  result.timing.p90_ns = 40;
  iree_benchmark_loom_workload_value_t workload_values[] = {
      {/*.type=*/LOOM_SCALAR_TYPE_INDEX, /*.value=*/4096},
      {/*.type=*/LOOM_SCALAR_TYPE_I32, /*.value=*/513},
  };
  iree_benchmark_loom_launch_record_t launch_record = {
      /*.case_sample_ordinal=*/0,
      /*.sequence_step_ordinal=*/0,
      /*.entry=*/IREE_SV("kernel_entry"),
      /*.workload_values=*/workload_values,
      /*.workload_value_count=*/IREE_ARRAYSIZE(workload_values),
      /*.launch_config=*/
      {
          /*.fields=*/
          LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
              LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE |
              LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE,
          /*.workgroup_count=*/{64, 2, 1},
          /*.workgroup_size=*/{64, 1, 1},
          /*.subgroup_size=*/32,
      },
  };
  iree_benchmark_loom_launch_evidence_t launch_evidence = {
      /*.host_allocator=*/{},
      /*.records=*/&launch_record,
      /*.record_count=*/1,
      /*.workload_values=*/workload_values,
      /*.workload_value_count=*/IREE_ARRAYSIZE(workload_values),
  };
  result.launch_evidence = &launch_evidence;
  loom_sanitizer_options_t sanitizer = {};
  sanitizer.checks = LOOM_SANITIZER_CHECK_RACE;
  sanitizer.reporting_mode = LOOM_SANITIZER_REPORTING_MODE_REPORT_ONLY;

  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_run(
      &event_sink, &run, /*dry_run=*/false, &sanitizer));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_benchmark_result(
      &event_sink, &run, &candidate0, /*work_item_index=*/7, &module,
      &benchmark_plan, &case_plan, &policy, &result,
      /*correctness_sample_count=*/1,
      /*correctness_failed_sample_count=*/0));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_benchmark_result(
      &event_sink, &run, &candidate1, /*work_item_index=*/7, &module,
      &benchmark_plan, &case_plan, &policy, &result,
      /*correctness_sample_count=*/1,
      /*correctness_failed_sample_count=*/0));
  workload_values[0].value = 1;
  launch_record.launch_config.workgroup_count.x = 1;
  iree_benchmark_loom_artifact_bundle_t bundle = {};
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_summary(
      &event_sink, &run, &bundle, /*planned_case_count=*/1,
      /*planned_benchmark_count=*/2, /*selected_benchmark_count=*/2,
      /*logical_sample_count=*/2, /*work_item_count=*/1,
      /*failure_count=*/0, /*failed_benchmark_count=*/0,
      /*correctness_sample_count=*/1, /*correctness_failed_sample_count=*/0,
      /*dry_run=*/false));

  iree_string_builder_t output;
  iree_string_builder_initialize(allocator, &output);
  iree_string_view_t root = ParseJsonDocument(SnapshotJson(&snapshot, &output));

  iree_string_view_t output_format =
      LookupObject(root, IREE_SV("output_format"));
  EXPECT_TRUE(iree_string_view_equal(output_format, IREE_SV("snapshot")));
  iree_string_view_t sanitizer_json = LookupObject(root, IREE_SV("sanitizer"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(sanitizer_json, IREE_SV("checks")), IREE_SV("race")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(sanitizer_json, IREE_SV("reporting_mode")),
      IREE_SV("report-only")));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(root, IREE_SV("work_item_count"))));
  iree_string_view_t summary = LookupObject(root, IREE_SV("summary"));
  iree_string_view_t planned = LookupObject(summary, IREE_SV("planned"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(planned, IREE_SV("work_item_count")), IREE_SV("1")));

  iree_string_view_t work_items = LookupObject(root, IREE_SV("work_items"));
  iree_host_size_t work_item_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(work_items, &work_item_count));
  EXPECT_EQ(work_item_count, 1u);

  iree_string_view_t benchmarks = LookupObject(root, IREE_SV("benchmarks"));
  iree_host_size_t benchmark_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(benchmarks, &benchmark_count));
  EXPECT_EQ(benchmark_count, 2u);

  iree_string_view_t first_work_item = FirstArrayElement(work_items);
  iree_string_view_t policy_json =
      LookupObject(first_work_item, IREE_SV("policy"));
  EXPECT_TRUE(
      iree_string_view_equal(LookupObject(policy_json, IREE_SV("measure")),
                             IREE_SV("case_end_to_end")));
  iree_string_view_t correctness =
      LookupObject(first_work_item, IREE_SV("correctness"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(correctness, IREE_SV("sample_count")), IREE_SV("1")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(correctness, IREE_SV("failed_sample_count")), IREE_SV("0")));
  iree_string_view_t measurement =
      LookupObject(first_work_item, IREE_SV("measurement"));
  EXPECT_FALSE(iree_string_view_is_empty(
      LookupObject(measurement, IREE_SV("timing_ns"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(first_work_item, IREE_SV("timing_ns"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(measurement, IREE_SV("batch_timing_ns"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(measurement, IREE_SV("dispatch_timing_ns"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(measurement, IREE_SV("operation_timing_ns"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(first_work_item, IREE_SV("profile_replay"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(first_work_item, IREE_SV("compile_report"))));
  iree_string_view_t launches =
      LookupObject(first_work_item, IREE_SV("launches"));
  iree_string_view_t launch = FirstArrayElement(launches);
  iree_string_view_t workload = LookupObject(launch, IREE_SV("workload"));
  iree_string_view_t first_workload_value = FirstArrayElement(workload);
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(first_workload_value, IREE_SV("type")), IREE_SV("index")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(first_workload_value, IREE_SV("value")), IREE_SV("4096")));
  iree_string_view_t launch_config =
      LookupObject(launch, IREE_SV("launch_config"));
  iree_string_view_t workgroup_count =
      LookupObject(launch_config, IREE_SV("workgroup_count"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(workgroup_count, IREE_SV("x")), IREE_SV("64")));
  EXPECT_TRUE(
      iree_string_view_is_empty(TryLookupObject(root, IREE_SV("failures"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(root, IREE_SV("failed_samples"))));

  iree_string_builder_deinitialize(&output);
  iree_benchmark_loom_snapshot_sink_deinitialize(&snapshot);
}

TEST(BenchmarkSnapshotSinkTest, IncludesRequestedProfileSummary) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_benchmark_loom_snapshot_sink_t snapshot = {};
  IREE_ASSERT_OK(
      iree_benchmark_loom_snapshot_sink_initialize(allocator, &snapshot));
  iree_benchmark_loom_event_sink_t event_sink = {};
  iree_benchmark_loom_snapshot_event_sink_initialize(&snapshot, &event_sink);

  iree_benchmark_loom_run_identity_t run = {};
  run.run_id = IREE_SV("run");
  run.source = IREE_SV("input.loom");
  run.results_path = IREE_SV("-");
  iree_benchmark_loom_candidate_identity_t candidate = {};
  candidate.candidate_id = IREE_SV("c0");
  loom_module_t module = {};
  loom_testbench_benchmark_plan_t benchmark_plan = {};
  benchmark_plan.name = IREE_SV("kernel_latency");
  loom_testbench_case_plan_t case_plan = {};
  case_plan.name = IREE_SV("kernel_case");
  iree_benchmark_loom_benchmark_policy_t policy = {};
  policy.measure = IREE_SV("dispatch_complete");
  iree_benchmark_loom_benchmark_result_t result = {};
  result.executed = true;
  result.passed = true;
  result.samples_per_iteration = 1;
  result.has_hal_benchmark = true;
  result.hal_benchmark.timing.batch_size = 16;
  result.hal_benchmark.timing.measured_batch_count = 3;
  result.hal_benchmark.timing.measured_operation_count = 48;
  result.hal_benchmark.timing.stop_reason =
      LOOM_RUN_BENCHMARK_STOP_REASON_STABLE;
  result.hal_benchmark.timing.operation_timing.count = 3;
  result.hal_benchmark.timing.operation_timing.total_ns = 90;
  result.hal_benchmark.timing.operation_timing.minimum_ns = 20;
  result.hal_benchmark.timing.operation_timing.maximum_ns = 40;
  result.hal_benchmark.timing.operation_timing.mean_ns = 30.0;
  result.hal_benchmark.timing.operation_timing.p50_ns = 30;
  result.hal_benchmark.timing.operation_timing.p90_ns = 40;
  result.hal_benchmark.profile_replay.requested = true;

  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_run(
      &event_sink, &run, /*dry_run=*/false, &kNoSanitizer));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_benchmark_result(
      &event_sink, &run, &candidate, /*work_item_index=*/0, &module,
      &benchmark_plan, &case_plan, &policy, &result,
      /*correctness_sample_count=*/1,
      /*correctness_failed_sample_count=*/0));
  iree_benchmark_loom_artifact_bundle_t bundle = {};
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_summary(
      &event_sink, &run, &bundle, /*planned_case_count=*/1,
      /*planned_benchmark_count=*/1, /*selected_benchmark_count=*/1,
      /*logical_sample_count=*/1, /*work_item_count=*/1,
      /*failure_count=*/0, /*failed_benchmark_count=*/0,
      /*correctness_sample_count=*/1, /*correctness_failed_sample_count=*/0,
      /*dry_run=*/false));

  iree_string_builder_t output;
  iree_string_builder_initialize(allocator, &output);
  iree_string_view_t root = ParseJsonDocument(SnapshotJson(&snapshot, &output));
  iree_string_view_t work_items = LookupObject(root, IREE_SV("work_items"));
  iree_string_view_t first_work_item = FirstArrayElement(work_items);
  iree_string_view_t profile_replay =
      LookupObject(first_work_item, IREE_SV("profile_replay"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(profile_replay, IREE_SV("measurement_relationship")),
      IREE_SV("distinct_execution")));
  iree_string_view_t profile = LookupObject(profile_replay, IREE_SV("summary"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(profile, IREE_SV("requested")), IREE_SV("true")));
  EXPECT_TRUE(iree_string_view_equal(LookupObject(profile, IREE_SV("executed")),
                                     IREE_SV("false")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(profile, IREE_SV("row_count")), IREE_SV("0")));
  iree_host_size_t profile_row_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(LookupObject(profile, IREE_SV("rows")),
                                        &profile_row_count));
  EXPECT_EQ(profile_row_count, 0u);

  iree_string_builder_deinitialize(&output);
  iree_benchmark_loom_snapshot_sink_deinitialize(&snapshot);
}

TEST(BenchmarkSnapshotSinkTest, IncludesHalTimingCountsAndWarnings) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_benchmark_loom_snapshot_sink_t snapshot = {};
  IREE_ASSERT_OK(
      iree_benchmark_loom_snapshot_sink_initialize(allocator, &snapshot));
  iree_benchmark_loom_event_sink_t event_sink = {};
  iree_benchmark_loom_snapshot_event_sink_initialize(&snapshot, &event_sink);

  iree_benchmark_loom_run_identity_t run = {};
  run.run_id = IREE_SV("run");
  run.source = IREE_SV("input.loom");
  run.results_path = IREE_SV("-");
  iree_benchmark_loom_candidate_identity_t candidate = {};
  candidate.candidate_id = IREE_SV("c0");
  loom_module_t module = {};
  loom_testbench_benchmark_plan_t benchmark_plan = {};
  benchmark_plan.name = IREE_SV("kernel_latency");
  loom_testbench_case_plan_t case_plan = {};
  case_plan.name = IREE_SV("kernel_case");
  iree_benchmark_loom_benchmark_policy_t policy = {};
  policy.measure = IREE_SV("dispatch_complete");
  policy.hal_options.timing.stable_p90_to_p50_delta_ppm = 100000;
  iree_benchmark_loom_benchmark_result_t result = {};
  result.executed = true;
  result.passed = true;
  result.samples_per_iteration = 1;
  result.has_hal_benchmark = true;
  result.hal_benchmark.timing.batch_size = 1;
  result.hal_benchmark.timing.measured_batch_count = 3;
  result.hal_benchmark.timing.measured_operation_count = 3;
  result.hal_benchmark.timing.measured_duration_ns = 900;
  result.hal_benchmark.timing.stop_reason =
      LOOM_RUN_BENCHMARK_STOP_REASON_MAX_BATCH_COUNT;
  result.hal_benchmark.timing.batch_timing.count = 3;
  result.hal_benchmark.timing.batch_timing.total_ns = 900;
  result.hal_benchmark.timing.batch_timing.minimum_ns = 200;
  result.hal_benchmark.timing.batch_timing.maximum_ns = 400;
  result.hal_benchmark.timing.batch_timing.mean_ns = 300.0;
  result.hal_benchmark.timing.batch_timing.p50_ns = 300;
  result.hal_benchmark.timing.batch_timing.p90_ns = 400;
  result.hal_benchmark.timing.batch_timing.p90_to_p50_delta_ppm = 333333;
  result.hal_benchmark.timing.operation_timing =
      result.hal_benchmark.timing.batch_timing;
  result.data_cache.populated = true;
  result.data_cache.dispatches_per_batch = 6;

  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_run(
      &event_sink, &run, /*dry_run=*/false, &kNoSanitizer));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_benchmark_result(
      &event_sink, &run, &candidate, /*work_item_index=*/0, &module,
      &benchmark_plan, &case_plan, &policy, &result,
      /*correctness_sample_count=*/1,
      /*correctness_failed_sample_count=*/0));
  iree_benchmark_loom_artifact_bundle_t bundle = {};
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_summary(
      &event_sink, &run, &bundle, /*planned_case_count=*/1,
      /*planned_benchmark_count=*/1, /*selected_benchmark_count=*/1,
      /*logical_sample_count=*/1, /*work_item_count=*/1,
      /*failure_count=*/0, /*failed_benchmark_count=*/0,
      /*correctness_sample_count=*/1, /*correctness_failed_sample_count=*/0,
      /*dry_run=*/false));

  iree_string_builder_t output;
  iree_string_builder_initialize(allocator, &output);
  iree_string_view_t root = ParseJsonDocument(SnapshotJson(&snapshot, &output));
  iree_string_view_t work_items = LookupObject(root, IREE_SV("work_items"));
  iree_string_view_t first_work_item = FirstArrayElement(work_items);
  iree_string_view_t measurement =
      LookupObject(first_work_item, IREE_SV("measurement"));
  EXPECT_TRUE(iree_string_view_is_empty(TryLookupObject(
      first_work_item, IREE_SV("logical_operations_per_batch"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(first_work_item, IREE_SV("operation_timing_ns"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(measurement, IREE_SV("timing_ns"))));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(measurement, IREE_SV("logical_operations_per_batch")),
      IREE_SV("1")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(measurement, IREE_SV("physical_dispatches_per_batch")),
      IREE_SV("6")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(measurement,
                   IREE_SV("physical_dispatches_per_logical_operation")),
      IREE_SV("6")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(measurement, IREE_SV("measured_logical_operation_count")),
      IREE_SV("3")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(measurement, IREE_SV("measured_physical_dispatch_count")),
      IREE_SV("18")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(measurement, IREE_SV("mean_physical_dispatch_duration_ns")),
      IREE_SV("50.000")));
  EXPECT_FALSE(iree_string_view_is_empty(
      LookupObject(measurement, IREE_SV("batch_timing_ns"))));
  EXPECT_FALSE(iree_string_view_is_empty(
      LookupObject(measurement, IREE_SV("operation_timing_ns"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(measurement, IREE_SV("dispatch_timing_ns"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(measurement, IREE_SV("measured_dispatch_count"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(measurement, IREE_SV("measured_operation_count"))));

  iree_string_view_t timing_interpretation =
      LookupObject(measurement, IREE_SV("timing_interpretation"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(timing_interpretation, IREE_SV("score")),
      IREE_SV("operation_timing_ns")));
  iree_string_view_t warnings =
      LookupObject(timing_interpretation, IREE_SV("warnings"));
  EXPECT_TRUE(
      JsonArrayContainsString(warnings, IREE_SV("short_measured_duration")));
  EXPECT_TRUE(JsonArrayContainsString(
      warnings, IREE_SV("single_logical_operation_batch")));
  EXPECT_TRUE(JsonArrayContainsString(
      warnings, IREE_SV("low_physical_dispatch_sample_count")));
  EXPECT_TRUE(JsonArrayContainsString(
      warnings, IREE_SV("sub_microsecond_logical_operation")));
  EXPECT_TRUE(
      JsonArrayContainsString(warnings, IREE_SV("unstable_p90_to_p50")));

  iree_string_builder_deinitialize(&output);
  iree_benchmark_loom_snapshot_sink_deinitialize(&snapshot);
}

TEST(BenchmarkSnapshotSinkTest, IncludesRequestedCompileReport) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_benchmark_loom_snapshot_sink_t snapshot = {};
  IREE_ASSERT_OK(
      iree_benchmark_loom_snapshot_sink_initialize(allocator, &snapshot));
  iree_benchmark_loom_event_sink_t event_sink = {};
  iree_benchmark_loom_snapshot_event_sink_initialize(&snapshot, &event_sink);

  loom_compile_report_capture_options_t capture_options = {};
  loom_compile_report_capture_options_initialize(&capture_options);
  capture_options.sink_format = LOOM_COMPILE_REPORT_SINK_FORMAT_JSON;
  capture_options.detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY;
  loom_compile_report_capture_t capture = {};
  IREE_ASSERT_OK(loom_compile_report_capture_initialize(&capture_options,
                                                        allocator, &capture));
  capture.report.artifact_kind =
      LOOM_TARGET_COMPILE_ARTIFACT_KIND_TARGET_ARTIFACT;
  capture.report.detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_WORKLOAD |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_DYNAMIC_INSTRUCTION_MIX;
  capture.report.dynamic_instruction_mix.memory_read_byte_count = 12;
  capture.report.dynamic_instruction_mix.memory_write_byte_count = 4;
  capture.report.dynamic_instruction_mix.global_load_count = 3;
  capture.report.dynamic_instruction_mix.global_store_count = 1;
  capture.report.workload.flags =
      LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_SIZE |
      LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_COUNT |
      LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_SIZE |
      LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKGROUP_COUNT |
      LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT;
  capture.report.workload.workgroup_size = {4, 1, 1};
  capture.report.workload.workgroup_count = {2, 1, 1};
  capture.report.workload.flat_workgroup_size = 4;
  capture.report.workload.dispatch_workgroup_count = 2;
  capture.report.workload.dispatch_workitem_count = 8;

  iree_benchmark_loom_run_identity_t run = {};
  run.run_id = IREE_SV("run");
  run.source = IREE_SV("input.loom");
  run.results_path = IREE_SV("-");
  iree_benchmark_loom_candidate_identity_t candidate = {};
  candidate.candidate_id = IREE_SV("c0");
  loom_module_t module = {};
  loom_testbench_benchmark_plan_t benchmark_plan = {};
  benchmark_plan.name = IREE_SV("kernel_latency");
  loom_testbench_case_plan_t case_plan = {};
  case_plan.name = IREE_SV("kernel_case");
  iree_benchmark_loom_benchmark_policy_t policy = {};
  policy.measure = IREE_SV("case_end_to_end");
  iree_benchmark_loom_benchmark_result_t result = {};
  result.executed = true;
  result.passed = true;
  result.samples_per_iteration = 1;
  result.timing.count = 1;
  result.timing.total_ns = 10;
  result.timing.minimum_ns = 10;
  result.timing.maximum_ns = 10;
  result.timing.mean_ns = 10.0;
  result.timing.p50_ns = 10;
  result.timing.p90_ns = 10;
  result.compile_report_capture = &capture;

  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_run(
      &event_sink, &run, /*dry_run=*/false, &kNoSanitizer));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_benchmark_result(
      &event_sink, &run, &candidate, /*work_item_index=*/0, &module,
      &benchmark_plan, &case_plan, &policy, &result,
      /*correctness_sample_count=*/1,
      /*correctness_failed_sample_count=*/0));
  iree_benchmark_loom_artifact_bundle_t bundle = {};
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_summary(
      &event_sink, &run, &bundle, /*planned_case_count=*/1,
      /*planned_benchmark_count=*/1, /*selected_benchmark_count=*/1,
      /*logical_sample_count=*/1, /*work_item_count=*/1,
      /*failure_count=*/0, /*failed_benchmark_count=*/0,
      /*correctness_sample_count=*/1, /*correctness_failed_sample_count=*/0,
      /*dry_run=*/false));

  iree_string_builder_t output;
  iree_string_builder_initialize(allocator, &output);
  iree_string_view_t root = ParseJsonDocument(SnapshotJson(&snapshot, &output));
  iree_string_view_t work_items = LookupObject(root, IREE_SV("work_items"));
  iree_string_view_t first_work_item = FirstArrayElement(work_items);
  iree_string_view_t compile_report =
      LookupObject(first_work_item, IREE_SV("compile_report"));
  iree_string_view_t artifact_kind =
      LookupObject(compile_report, IREE_SV("artifact_kind"));
  EXPECT_TRUE(
      iree_string_view_equal(artifact_kind, IREE_SV("target-artifact")));
  iree_string_view_t economics =
      LookupObject(compile_report, IREE_SV("economics"));
  iree_string_view_t memory = LookupObject(economics, IREE_SV("memory"));
  iree_string_view_t per_workitem_issued =
      LookupObject(memory, IREE_SV("per_workitem_issued"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(per_workitem_issued, IREE_SV("read_bytes")), IREE_SV("12")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(per_workitem_issued, IREE_SV("write_bytes")), IREE_SV("4")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(per_workitem_issued, IREE_SV("global_load_count")),
      IREE_SV("3")));
  iree_string_view_t dispatch_issued =
      LookupObject(memory, IREE_SV("dispatch_issued"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(dispatch_issued, IREE_SV("read_bytes")), IREE_SV("96")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(dispatch_issued, IREE_SV("write_bytes")), IREE_SV("32")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(dispatch_issued, IREE_SV("global_store_count")),
      IREE_SV("8")));

  iree_string_builder_deinitialize(&output);
  loom_compile_report_capture_deinitialize(&capture);
  iree_benchmark_loom_snapshot_sink_deinitialize(&snapshot);
}

TEST(BenchmarkSnapshotSinkTest, IncludesFailurePayloadsOnFailure) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_benchmark_loom_snapshot_sink_t snapshot = {};
  IREE_ASSERT_OK(
      iree_benchmark_loom_snapshot_sink_initialize(allocator, &snapshot));
  iree_benchmark_loom_event_sink_t event_sink = {};
  iree_benchmark_loom_snapshot_event_sink_initialize(&snapshot, &event_sink);

  iree_benchmark_loom_run_identity_t run = {};
  run.run_id = IREE_SV("run");
  run.source = IREE_SV("input.loom");
  run.results_path = IREE_SV("-");
  iree_benchmark_loom_candidate_identity_t candidate = {};
  candidate.candidate_id = IREE_SV("c0");
  loom_module_t module = {};
  loom_testbench_benchmark_plan_t benchmark_plan = {};
  benchmark_plan.name = IREE_SV("kernel_latency");
  loom_testbench_case_plan_t case_plan = {};
  case_plan.name = IREE_SV("kernel_case");
  iree_benchmark_loom_benchmark_policy_t policy = {};
  policy.measure = IREE_SV("dispatch_complete");
  iree_benchmark_loom_hal_actual_provider_t provider = {};
  provider.execution.invocation_options.function_name =
      IREE_SV("rejected_kernel");
  provider.execution.compile_failure_stage = IREE_SV("compile");
  provider.execution.compile_failure_kind = IREE_SV("diagnostics");
  provider.execution.compile_failure_message =
      IREE_SV("candidate did not lower");
  provider.execution.compile_rejected = true;
  provider.diagnostics.error_count = 1;
  iree_benchmark_loom_benchmark_result_t result = {};
  iree_benchmark_loom_benchmark_result_set_compile_rejection(&provider,
                                                             &result);
  result.diagnostic_json = IREE_SV("{\"message\":\"bad op\"}");

  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_run(
      &event_sink, &run, /*dry_run=*/false, &kNoSanitizer));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_failure(
      &event_sink, &run, IREE_SV("parse"), IREE_SV("diagnostics"),
      IREE_SV("input module has parse errors"), /*diagnostics=*/NULL));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_benchmark_result(
      &event_sink, &run, &candidate, /*work_item_index=*/0, &module,
      &benchmark_plan, &case_plan, &policy, &result,
      /*correctness_sample_count=*/0,
      /*correctness_failed_sample_count=*/0));
  iree_benchmark_loom_artifact_bundle_t bundle = {};
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_summary(
      &event_sink, &run, &bundle, /*planned_case_count=*/1,
      /*planned_benchmark_count=*/1, /*selected_benchmark_count=*/1,
      /*logical_sample_count=*/1, /*work_item_count=*/1,
      /*failure_count=*/1, /*failed_benchmark_count=*/1,
      /*correctness_sample_count=*/0, /*correctness_failed_sample_count=*/0,
      /*dry_run=*/false));

  iree_string_builder_t output;
  iree_string_builder_initialize(allocator, &output);
  iree_string_view_t root = ParseJsonDocument(SnapshotJson(&snapshot, &output));
  iree_string_view_t failures = LookupObject(root, IREE_SV("failures"));
  iree_host_size_t failure_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(failures, &failure_count));
  EXPECT_EQ(failure_count, 1u);
  iree_string_view_t work_items = LookupObject(root, IREE_SV("work_items"));
  iree_string_view_t first_work_item = FirstArrayElement(work_items);
  iree_string_view_t failure =
      LookupObject(first_work_item, IREE_SV("failure"));
  iree_string_view_t failure_entry = LookupObject(failure, IREE_SV("entry"));
  EXPECT_TRUE(
      iree_string_view_equal(failure_entry, IREE_SV("rejected_kernel")));
  iree_string_view_t failure_stage = LookupObject(failure, IREE_SV("stage"));
  EXPECT_TRUE(iree_string_view_equal(failure_stage, IREE_SV("compile")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(failure, IREE_SV("diagnostic_error_count")), IREE_SV("1")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(failure, IREE_SV("diagnostic_warning_count")),
      IREE_SV("0")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(failure, IREE_SV("diagnostic_remark_count")), IREE_SV("0")));
  iree_string_view_t diagnostics =
      LookupObject(failure, IREE_SV("diagnostics"));
  iree_host_size_t diagnostic_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(diagnostics, &diagnostic_count));
  EXPECT_EQ(diagnostic_count, 1u);

  iree_string_builder_deinitialize(&output);
  iree_benchmark_loom_snapshot_sink_deinitialize(&snapshot);
}

TEST(BenchmarkSnapshotSinkTest, DryRunReportsPlannedWorkAliases) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_benchmark_loom_snapshot_sink_t snapshot = {};
  IREE_ASSERT_OK(
      iree_benchmark_loom_snapshot_sink_initialize(allocator, &snapshot));
  iree_benchmark_loom_event_sink_t event_sink = {};
  iree_benchmark_loom_snapshot_event_sink_initialize(&snapshot, &event_sink);

  iree_benchmark_loom_run_identity_t run = {};
  run.run_id = IREE_SV("run");
  run.source = IREE_SV("input.loom");
  run.results_path = IREE_SV("-");
  run.file_output_dir = IREE_SV("/tmp/loom");
  iree_benchmark_loom_selected_benchmark_t selections[2] = {};
  selections[0].identity.candidate_id = IREE_SV("c0");
  selections[0].identity.candidate_index = 0;
  selections[1].identity.candidate_id = IREE_SV("c1");
  selections[1].identity.candidate_index = 1;
  loom_testbench_benchmark_plan_t benchmark_plans[2] = {};
  benchmark_plans[0].name = IREE_SV("kernel_latency_default");
  benchmark_plans[0].sample_count = 1;
  benchmark_plans[1].name = IREE_SV("kernel_latency_alias");
  benchmark_plans[1].sample_count = 1;
  loom_testbench_case_plan_t case_plan = {};
  case_plan.name = IREE_SV("kernel_case");
  selections[0].benchmark_plan = &benchmark_plans[0];
  selections[0].case_plan = &case_plan;
  selections[0].policy.measure = IREE_SV("dispatch_complete");
  selections[1].benchmark_plan = &benchmark_plans[1];
  selections[1].case_plan = &case_plan;
  selections[1].policy.measure = IREE_SV("dispatch_complete");
  iree_benchmark_loom_logical_sample_t logical_samples[2] = {};
  logical_samples[0].selection_index = 0;
  logical_samples[0].begin_benchmark_sample = 0;
  logical_samples[0].end_benchmark_sample = 1;
  logical_samples[0].has_case_sample_ordinal = true;
  logical_samples[0].case_sample_ordinal = 0;
  logical_samples[0].work_item_index = 7;
  logical_samples[1].selection_index = 1;
  logical_samples[1].begin_benchmark_sample = 0;
  logical_samples[1].end_benchmark_sample = 1;
  logical_samples[1].has_case_sample_ordinal = true;
  logical_samples[1].case_sample_ordinal = 0;
  logical_samples[1].work_item_index = 7;
  iree_benchmark_loom_work_item_t work_item = {};
  work_item.kind = IREE_BENCHMARK_LOOM_WORK_ITEM_DISPATCH_SAMPLE;
  work_item.work_item_index = 7;
  work_item.representative_selection_index = 0;
  work_item.hal_compile_item_index = 0;
  work_item.begin_benchmark_sample = 0;
  work_item.end_benchmark_sample = 1;
  work_item.has_case_sample_ordinal = true;
  work_item.case_sample_ordinal = 0;
  iree_benchmark_loom_work_plan_t work_plan = {};
  work_plan.selected_benchmarks = selections;
  work_plan.selected_benchmark_count = IREE_ARRAYSIZE(selections);
  work_plan.logical_samples = logical_samples;
  work_plan.logical_sample_count = IREE_ARRAYSIZE(logical_samples);
  work_plan.work_items = &work_item;
  work_plan.work_item_count = 1;
  loom_module_t module = {};
  iree_benchmark_loom_artifact_bundle_t bundle = {};

  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_run(
      &event_sink, &run, /*dry_run=*/true, &kNoSanitizer));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_work_plan(
      &event_sink, &run, &module, &work_plan));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_summary(
      &event_sink, &run, &bundle, /*planned_case_count=*/1,
      /*planned_benchmark_count=*/2, /*selected_benchmark_count=*/2,
      /*logical_sample_count=*/2, /*work_item_count=*/1,
      /*failure_count=*/0, /*failed_benchmark_count=*/0,
      /*correctness_sample_count=*/0, /*correctness_failed_sample_count=*/0,
      /*dry_run=*/true));

  iree_string_builder_t output;
  iree_string_builder_initialize(allocator, &output);
  iree_string_view_t root = ParseJsonDocument(SnapshotJson(&snapshot, &output));

  bool dry_run = false;
  IREE_ASSERT_OK(iree_json_lookup_bool(root, IREE_SV("dry_run"), &dry_run));
  EXPECT_TRUE(dry_run);
  iree_string_view_t benchmarks = iree_string_view_empty();
  IREE_ASSERT_OK(
      iree_json_lookup_object_value(root, IREE_SV("benchmarks"), &benchmarks));
  iree_host_size_t benchmark_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(benchmarks, &benchmark_count));
  EXPECT_EQ(benchmark_count, 2u);
  iree_string_view_t work_items = iree_string_view_empty();
  IREE_ASSERT_OK(
      iree_json_lookup_object_value(root, IREE_SV("work_items"), &work_items));
  iree_host_size_t work_item_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(work_items, &work_item_count));
  EXPECT_EQ(work_item_count, 1u);

  iree_string_view_t first_work_item = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_array_get(work_items, 0, &first_work_item));
  iree_string_view_t work_item_index = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_lookup_object_value(
      first_work_item, IREE_SV("work_item_index"), &work_item_index));
  EXPECT_TRUE(iree_string_view_equal(work_item_index, IREE_SV("7")));
  iree_string_view_t work_item_kind = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_lookup_object_value(first_work_item, IREE_SV("kind"),
                                               &work_item_kind));
  EXPECT_TRUE(
      iree_string_view_equal(work_item_kind, IREE_SV("dispatch_sample")));
  iree_string_view_t representative_candidate_id = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_lookup_object_value(
      first_work_item, IREE_SV("representative_candidate_id"),
      &representative_candidate_id));
  EXPECT_TRUE(
      iree_string_view_equal(representative_candidate_id, IREE_SV("c0")));

  iree_string_view_t first_benchmark = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_array_get(benchmarks, 0, &first_benchmark));
  iree_string_view_t second_benchmark = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_array_get(benchmarks, 1, &second_benchmark));
  iree_string_view_t first_candidate_id = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_lookup_object_value(
      first_benchmark, IREE_SV("candidate_id"), &first_candidate_id));
  EXPECT_TRUE(iree_string_view_equal(first_candidate_id, IREE_SV("c0")));
  iree_string_view_t second_candidate_id = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_lookup_object_value(
      second_benchmark, IREE_SV("candidate_id"), &second_candidate_id));
  EXPECT_TRUE(iree_string_view_equal(second_candidate_id, IREE_SV("c1")));
  iree_string_view_t first_alias_work_item = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_lookup_object_value(
      first_benchmark, IREE_SV("work_item_index"), &first_alias_work_item));
  EXPECT_TRUE(iree_string_view_equal(first_alias_work_item, IREE_SV("7")));
  iree_string_view_t second_alias_work_item = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_lookup_object_value(
      second_benchmark, IREE_SV("work_item_index"), &second_alias_work_item));
  EXPECT_TRUE(iree_string_view_equal(second_alias_work_item, IREE_SV("7")));

  iree_string_builder_deinitialize(&output);
  iree_benchmark_loom_snapshot_sink_deinitialize(&snapshot);
}

}  // namespace
}  // namespace loom
