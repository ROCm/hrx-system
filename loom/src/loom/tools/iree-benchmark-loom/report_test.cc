// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/report.h"

#include <cstring>
#include <fstream>
#include <memory>
#include <string>

#include "iree/base/internal/json.h"
#include "iree/io/byte_sequence.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/tools/iree-benchmark-loom/launch_evidence.h"
#include "loom/tools/iree-benchmark-loom/manifest.h"

namespace loom {
namespace {

using ByteSequencePtr =
    std::unique_ptr<iree_io_byte_sequence_t,
                    decltype(&iree_io_byte_sequence_release)>;

static iree_status_t CloneByteSpanToSequence(
    iree_const_byte_span_t source, iree_allocator_t allocator,
    iree_io_byte_sequence_t** out_sequence) {
  *out_sequence = nullptr;
  void* data = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_clone(allocator, source, &data));
  iree_byte_span_t contents = iree_make_byte_span(data, source.data_length);
  iree_status_t status = iree_io_byte_sequence_create_from_span_move(
      &contents, allocator, out_sequence);
  iree_allocator_free(allocator, contents.data);
  return status;
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

static void ExpectObjectValueEquals(iree_string_view_t object,
                                    iree_string_view_t key,
                                    iree_string_view_t expected) {
  EXPECT_TRUE(iree_string_view_equal(LookupObject(object, key), expected));
}

static void ExpectStatusObject(iree_string_view_t status,
                               iree_status_code_t expected_code,
                               iree_string_view_t expected_message) {
  std::string code = std::to_string(static_cast<uint32_t>(expected_code));
  ExpectObjectValueEquals(status, IREE_SV("code"),
                          iree_make_cstring_view(code.c_str()));
  ExpectObjectValueEquals(
      status, IREE_SV("name"),
      iree_make_cstring_view(iree_status_code_string(expected_code)));
  if (iree_string_view_is_empty(expected_message)) {
    EXPECT_TRUE(
        iree_string_view_is_empty(TryLookupObject(status, IREE_SV("message"))));
  } else {
    ExpectObjectValueEquals(status, IREE_SV("message"), expected_message);
  }
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

TEST(BenchmarkReportTest, WritesStatusFieldJson) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);

  loom_json_object_writer_t object;
  IREE_ASSERT_OK(loom_json_object_begin(&stream, &object));
  IREE_ASSERT_OK(iree_benchmark_loom_write_status_field_json(
      IREE_STATUS_UNAVAILABLE, IREE_SV("profile decode failed"), &object));
  IREE_ASSERT_OK(loom_json_object_end(&object));

  iree_string_view_t root =
      ParseJsonDocument(iree_string_builder_view(&builder));
  iree_string_view_t status = LookupObject(root, IREE_SV("status"));
  ExpectStatusObject(status, IREE_STATUS_UNAVAILABLE,
                     IREE_SV("profile decode failed"));

  iree_string_builder_deinitialize(&builder);
}

TEST(BenchmarkReportTest, OmitsEmptyStatusCodeMessage) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);

  loom_json_object_writer_t object;
  IREE_ASSERT_OK(loom_json_object_begin(&stream, &object));
  IREE_ASSERT_OK(iree_benchmark_loom_write_status_field_json(
      IREE_STATUS_UNAVAILABLE, iree_string_view_empty(), &object));
  IREE_ASSERT_OK(loom_json_object_end(&object));

  iree_string_view_t root =
      ParseJsonDocument(iree_string_builder_view(&builder));
  iree_string_view_t status = LookupObject(root, IREE_SV("status"));
  ExpectStatusObject(status, IREE_STATUS_UNAVAILABLE, iree_string_view_empty());

  iree_string_builder_deinitialize(&builder);
}

TEST(BenchmarkReportTest, WritesHalProfileErrorWithStatusCodeFields) {
  loom_run_hal_profile_summary_t profile = {};
  profile.requested = true;
  profile.executed = true;
  profile.has_error = true;
  profile.error_code = IREE_STATUS_RESOURCE_EXHAUSTED;
  const iree_string_view_t message = IREE_SV("profile collection failed");
  profile.error_message_length = message.size;
  memcpy(profile.error_message, message.data, message.size);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(
      iree_benchmark_loom_write_hal_profile_summary_json(&profile, &stream));

  iree_string_view_t root =
      ParseJsonDocument(iree_string_builder_view(&builder));
  iree_string_view_t status = LookupObject(root, IREE_SV("status"));
  ExpectStatusObject(status, IREE_STATUS_RESOURCE_EXHAUSTED,
                     IREE_SV("profile collection failed"));

  iree_string_builder_deinitialize(&builder);
}

TEST(BenchmarkReportTest, WritesCanonicalCompileReportTree) {
  iree_allocator_t allocator = iree_allocator_system();
  loom_run_compile_report_capture_options_t capture_options = {};
  loom_run_compile_report_capture_options_initialize(&capture_options);
  capture_options.sink_format = LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_JSON;
  capture_options.detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY;
  loom_run_compile_report_capture_t capture = {};
  IREE_ASSERT_OK(loom_run_compile_report_capture_initialize(
      &capture_options, allocator, &capture));

  loom_target_compile_report_t* report = &capture.report;
  report->artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE;
  report->backend_name = IREE_SV("test-hal");
  report->target_family_name = IREE_SV("test");
  report->target_key = IREE_SV("test-target");
  report->function_name = IREE_SV("candidate_kernel");
  const loom_target_compile_report_target_capability_row_t capability_row = {
      /*.function_name=*/report->function_name,
      /*.target_family_name=*/report->target_family_name,
      /*.namespace_name=*/IREE_SV("test"),
      /*.key=*/IREE_SV("matrix_feature_profile"),
      /*.value_kind=*/LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_STRING,
      /*.value_u64=*/0,
      /*.value_string=*/IREE_SV("test-profile"),
  };
  IREE_ASSERT_OK(loom_target_compile_report_record_target_capability_row(
      report, &capability_row));
  loom_target_compile_report_record_status(report, IREE_STATUS_OK);
  loom_target_compile_report_record_schedule(
      report, /*node_count=*/31, /*scheduled_node_count=*/29,
      /*dependency_count=*/17, /*resource_use_count=*/13,
      /*hazard_gap_count=*/7, /*model_summary_count=*/5,
      /*pressure_summary_count=*/3, /*peak_live_units=*/128);
  loom_target_compile_report_static_instruction_mix_t mix = {};
  mix.descriptor_count = 11;
  mix.vector_alu_count = 9;
  mix.local_memory_count = 4;
  loom_target_compile_report_record_static_instruction_mix(report, &mix);
  loom_target_compile_report_record_allocation(
      report, /*assignment_count=*/23, /*spill_count=*/2,
      /*spill_plan_count=*/1, /*coalesced_copy_count=*/8,
      /*materialized_copy_count=*/3, /*storage_lease_count=*/11,
      /*storage_lease_instance_count=*/9,
      /*storage_release_action_count=*/4);
  loom_target_compile_report_record_allocation_materialization(
      report, /*spill_storage_count=*/4, /*spill_storage_bytes=*/40,
      /*spill_store_count=*/5, /*spill_store_bytes=*/50, /*reload_count=*/6,
      /*reload_bytes=*/60);
  loom_target_compile_report_record_emission(report, /*instruction_count=*/37,
                                             /*code_byte_count=*/148,
                                             /*code_storage_byte_count=*/160);
  loom_target_compile_report_record_memory(report, /*private_memory_bytes=*/64,
                                           /*local_memory_bytes=*/256);

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

  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(iree_benchmark_loom_write_benchmark_result_json(
      &benchmark_plan, &case_plan, &policy, &result,
      /*correctness_sample_count=*/1,
      /*correctness_failed_sample_count=*/0, &stream));

  iree_string_view_t root =
      ParseJsonDocument(iree_string_builder_view(&builder));
  EXPECT_TRUE(
      iree_string_view_is_empty(TryLookupObject(root, IREE_SV("measure"))));
  iree_string_view_t policy_json = LookupObject(root, IREE_SV("policy"));
  ExpectObjectValueEquals(policy_json, IREE_SV("measure"),
                          IREE_SV("case_end_to_end"));
  iree_string_view_t correctness = LookupObject(root, IREE_SV("correctness"));
  ExpectObjectValueEquals(correctness, IREE_SV("sample_count"), IREE_SV("1"));
  ExpectObjectValueEquals(correctness, IREE_SV("failed_sample_count"),
                          IREE_SV("0"));
  iree_string_view_t measurement = LookupObject(root, IREE_SV("measurement"));
  ExpectObjectValueEquals(measurement, IREE_SV("samples_per_iteration"),
                          IREE_SV("1"));
  ExpectObjectValueEquals(measurement, IREE_SV("failed_sample_count"),
                          IREE_SV("0"));
  EXPECT_FALSE(iree_string_view_is_empty(
      LookupObject(measurement, IREE_SV("timing_ns"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(root, IREE_SV("static_summary"))));
  iree_string_view_t compile_report =
      LookupObject(root, IREE_SV("compile_report"));
  ExpectObjectValueEquals(compile_report, IREE_SV("artifact_kind"),
                          IREE_SV("vm-archive"));
  ExpectStatusObject(LookupObject(compile_report, IREE_SV("status")),
                     IREE_STATUS_OK, iree_string_view_empty());
  ExpectObjectValueEquals(compile_report, IREE_SV("backend"),
                          IREE_SV("test-hal"));
  ExpectObjectValueEquals(compile_report, IREE_SV("target_family"),
                          IREE_SV("test"));
  ExpectObjectValueEquals(compile_report, IREE_SV("target_key"),
                          IREE_SV("test-target"));
  ExpectObjectValueEquals(compile_report, IREE_SV("function"),
                          IREE_SV("candidate_kernel"));

  iree_string_view_t schedule =
      LookupObject(compile_report, IREE_SV("schedule"));
  ExpectObjectValueEquals(schedule, IREE_SV("node_count"), IREE_SV("31"));
  ExpectObjectValueEquals(
      schedule, IREE_SV("register_pressure_peak_live_units"), IREE_SV("128"));
  iree_string_view_t mix_json =
      LookupObject(compile_report, IREE_SV("static_instruction_mix"));
  ExpectObjectValueEquals(mix_json, IREE_SV("descriptor_count"), IREE_SV("11"));
  ExpectObjectValueEquals(mix_json, IREE_SV("vector_alu_count"), IREE_SV("9"));
  ExpectObjectValueEquals(mix_json, IREE_SV("local_memory_count"),
                          IREE_SV("4"));
  iree_string_view_t allocation =
      LookupObject(compile_report, IREE_SV("allocation"));
  ExpectObjectValueEquals(allocation, IREE_SV("spill_count"), IREE_SV("2"));
  ExpectObjectValueEquals(
      allocation, IREE_SV("materialized_spill_storage_count"), IREE_SV("4"));
  ExpectObjectValueEquals(
      allocation, IREE_SV("materialized_spill_storage_bytes"), IREE_SV("40"));
  ExpectObjectValueEquals(allocation, IREE_SV("materialized_spill_store_count"),
                          IREE_SV("5"));
  ExpectObjectValueEquals(allocation, IREE_SV("materialized_spill_store_bytes"),
                          IREE_SV("50"));
  ExpectObjectValueEquals(allocation, IREE_SV("materialized_reload_count"),
                          IREE_SV("6"));
  ExpectObjectValueEquals(allocation, IREE_SV("materialized_reload_bytes"),
                          IREE_SV("60"));
  ExpectObjectValueEquals(allocation, IREE_SV("storage_lease_count"),
                          IREE_SV("11"));
  ExpectObjectValueEquals(allocation, IREE_SV("storage_release_action_count"),
                          IREE_SV("4"));
  iree_string_view_t emission =
      LookupObject(compile_report, IREE_SV("emission"));
  ExpectObjectValueEquals(emission, IREE_SV("code_byte_count"), IREE_SV("148"));
  iree_string_view_t memory = LookupObject(compile_report, IREE_SV("memory"));
  ExpectObjectValueEquals(memory, IREE_SV("private_bytes"), IREE_SV("64"));
  ExpectObjectValueEquals(memory, IREE_SV("local_bytes"), IREE_SV("256"));
  iree_string_view_t capability_rows =
      LookupObject(compile_report, IREE_SV("target_capability_rows"));
  ExpectObjectValueEquals(capability_rows, IREE_SV("count"), IREE_SV("1"));
  iree_string_view_t rows = LookupObject(capability_rows, IREE_SV("rows"));
  iree_string_view_t first_capability_row = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_array_get(rows, 0, &first_capability_row));
  ExpectObjectValueEquals(first_capability_row, IREE_SV("namespace"),
                          IREE_SV("test"));
  ExpectObjectValueEquals(first_capability_row, IREE_SV("key"),
                          IREE_SV("matrix_feature_profile"));
  ExpectObjectValueEquals(first_capability_row, IREE_SV("value_string"),
                          IREE_SV("test-profile"));

  iree_string_builder_deinitialize(&builder);
  loom_run_compile_report_capture_deinitialize(&capture);
}

TEST(BenchmarkReportTest, WritesHalTimingCountsAndWarnings) {
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
  result.data_cache.correctness_materialization =
      IREE_BENCHMARK_LOOM_BUFFER_MATERIALIZATION_HOST_VISIBLE;
  result.data_cache.measurement_materialization =
      IREE_BENCHMARK_LOOM_BUFFER_MATERIALIZATION_DEVICE_LOCAL;
  result.data_cache.binding_count = 2;
  result.data_cache.binding_ring_count = 1;
  result.data_cache.command_buffer_ring_count = 1;
  result.data_cache.dispatches_per_batch = 6;
  result.data_cache.requested_min_ring_bytes = 4096;
  result.data_cache.binding_set_bytes = 8192;
  result.data_cache.binding_ring_bytes = 8192;
  result.artifact_manifest_path =
      IREE_SV("bundle/artifact_manifests/run_candidate_artifact_manifest.json");

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(iree_benchmark_loom_write_benchmark_result_json(
      &benchmark_plan, &case_plan, &policy, &result,
      /*correctness_sample_count=*/1,
      /*correctness_failed_sample_count=*/0, &stream));

  iree_string_view_t root =
      ParseJsonDocument(iree_string_builder_view(&builder));
  iree_string_view_t policy_json = LookupObject(root, IREE_SV("policy"));
  ExpectObjectValueEquals(policy_json, IREE_SV("measure"),
                          IREE_SV("dispatch_complete"));
  iree_string_view_t correctness = LookupObject(root, IREE_SV("correctness"));
  ExpectObjectValueEquals(correctness, IREE_SV("sample_count"), IREE_SV("1"));
  ExpectObjectValueEquals(correctness, IREE_SV("failed_sample_count"),
                          IREE_SV("0"));
  iree_string_view_t measurement = LookupObject(root, IREE_SV("measurement"));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(root, IREE_SV("logical_operations_per_batch"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(root, IREE_SV("operation_timing_ns"))));
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
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(measurement, IREE_SV("dispatch_timing_ns"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(measurement, IREE_SV("measured_dispatch_count"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(measurement, IREE_SV("measured_operation_count"))));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(root, IREE_SV("artifact_manifest_path")),
      result.artifact_manifest_path));
  iree_string_view_t data_cache = LookupObject(root, IREE_SV("data_cache"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(data_cache, IREE_SV("correctness_materialization")),
      IREE_SV("host_visible")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(data_cache, IREE_SV("measurement_materialization")),
      IREE_SV("device_local")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(data_cache, IREE_SV("binding_ring_count")), IREE_SV("1")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(data_cache, IREE_SV("ring_byte_target_met")),
      IREE_SV("true")));

  iree_string_view_t timing_interpretation =
      LookupObject(measurement, IREE_SV("timing_interpretation"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(timing_interpretation, IREE_SV("score")),
      IREE_SV("operation_timing_ns")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(timing_interpretation, IREE_SV("score_time_domain")),
      IREE_SV("host_wall")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(timing_interpretation, IREE_SV("score_meaning")),
      IREE_SV("host_queue_completion_normalized_logical_operation_time")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(timing_interpretation, IREE_SV("score_unit")),
      IREE_SV("logical_operation")));
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

  iree_string_builder_deinitialize(&builder);
}

TEST(BenchmarkReportTest, WritesExactWorkloadAndResolvedLaunchConfig) {
  loom_testbench_benchmark_plan_t benchmark_plan = {};
  benchmark_plan.name = IREE_SV("dynamic_latency");
  loom_testbench_case_plan_t case_plan = {};
  case_plan.name = IREE_SV("dynamic_case");
  iree_benchmark_loom_benchmark_policy_t policy = {};
  policy.measure = IREE_SV("dispatch_complete");

  iree_benchmark_loom_workload_value_t workload_values[] = {
      {
          /*.type=*/LOOM_SCALAR_TYPE_INDEX,
          /*.value=*/4096,
      },
      {
          /*.type=*/LOOM_SCALAR_TYPE_I32,
          /*.value=*/513,
      },
  };
  iree_benchmark_loom_launch_record_t launch_record = {
      /*.case_sample_ordinal=*/3,
      /*.sequence_step_ordinal=*/0,
      /*.entry=*/IREE_SV("dynamic_kernel"),
      /*.workload_values=*/workload_values,
      /*.workload_value_count=*/IREE_ARRAYSIZE(workload_values),
      /*.launch_config=*/
      {
          /*.fields=*/
          LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
              LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE |
              LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE,
          /*.workgroup_count=*/{65, 2, 1},
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
  iree_benchmark_loom_benchmark_result_t result = {};
  result.executed = true;
  result.passed = true;
  result.samples_per_iteration = 1;
  result.launch_evidence = &launch_evidence;

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(iree_benchmark_loom_write_benchmark_result_json(
      &benchmark_plan, &case_plan, &policy, &result,
      /*correctness_sample_count=*/1,
      /*correctness_failed_sample_count=*/0, &stream));

  const iree_string_view_t root =
      ParseJsonDocument(iree_string_builder_view(&builder));
  const iree_string_view_t launches = LookupObject(root, IREE_SV("launches"));
  iree_host_size_t launch_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(launches, &launch_count));
  ASSERT_EQ(launch_count, 1u);
  iree_string_view_t launch = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_array_get(launches, 0, &launch));
  ExpectObjectValueEquals(launch, IREE_SV("case_sample_index"), IREE_SV("3"));
  ExpectObjectValueEquals(launch, IREE_SV("entry"), IREE_SV("dynamic_kernel"));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(launch, IREE_SV("workload_count"))));

  const iree_string_view_t workload = LookupObject(launch, IREE_SV("workload"));
  iree_host_size_t workload_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(workload, &workload_count));
  ASSERT_EQ(workload_count, 2u);
  iree_string_view_t second_workload = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_array_get(workload, 1, &second_workload));
  ExpectObjectValueEquals(second_workload, IREE_SV("type"), IREE_SV("i32"));
  ExpectObjectValueEquals(second_workload, IREE_SV("value"), IREE_SV("513"));

  const iree_string_view_t launch_config =
      LookupObject(launch, IREE_SV("launch_config"));
  const iree_string_view_t workgroup_count =
      LookupObject(launch_config, IREE_SV("workgroup_count"));
  ExpectObjectValueEquals(workgroup_count, IREE_SV("x"), IREE_SV("65"));
  ExpectObjectValueEquals(workgroup_count, IREE_SV("y"), IREE_SV("2"));
  const iree_string_view_t workgroup_size =
      LookupObject(launch_config, IREE_SV("workgroup_size"));
  ExpectObjectValueEquals(workgroup_size, IREE_SV("x"), IREE_SV("64"));
  ExpectObjectValueEquals(launch_config, IREE_SV("subgroup_size"),
                          IREE_SV("32"));

  iree_string_builder_deinitialize(&builder);
}

TEST(BenchmarkReportTest, ScopesComparableDispatchTimingToProfileReplay) {
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
  result.hal_benchmark.timing.measured_batch_count = 4;
  result.hal_benchmark.timing.measured_operation_count = 64;
  result.hal_benchmark.timing.measured_duration_ns = 64000;
  result.hal_benchmark.timing.stop_reason =
      LOOM_RUN_BENCHMARK_STOP_REASON_STABLE;
  result.hal_benchmark.timing.batch_timing.count = 4;
  result.hal_benchmark.timing.batch_timing.total_ns = 64000;
  result.hal_benchmark.timing.batch_timing.minimum_ns = 15000;
  result.hal_benchmark.timing.batch_timing.maximum_ns = 17000;
  result.hal_benchmark.timing.batch_timing.mean_ns = 16000.0;
  result.hal_benchmark.timing.batch_timing.p50_ns = 16000;
  result.hal_benchmark.timing.batch_timing.p90_ns = 17000;
  result.hal_benchmark.timing.operation_timing.count = 4;
  result.hal_benchmark.timing.operation_timing.total_ns = 4000;
  result.hal_benchmark.timing.operation_timing.minimum_ns = 900;
  result.hal_benchmark.timing.operation_timing.maximum_ns = 1100;
  result.hal_benchmark.timing.operation_timing.mean_ns = 1000.0;
  result.hal_benchmark.timing.operation_timing.p50_ns = 1000;
  result.hal_benchmark.timing.operation_timing.p90_ns = 1100;
  result.data_cache.populated = true;
  result.data_cache.dispatches_per_batch = 16;

  loom_run_hal_profile_summary_t* profile =
      &result.hal_benchmark.profile_replay;
  profile->requested = true;
  profile->executed = true;
  profile->row_count = 1;
  profile->captured_row_count = 1;
  loom_run_hal_profile_row_summary_t* row = &profile->rows[0];
  row->row_type = IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_FUNCTION;
  row->time_domain = IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_DEVICE_TICK;
  row->flags = IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_TIMING;
  row->physical_device_ordinal = 0;
  row->function_ordinal = 0;
  row->sample_count = 16;
  row->first_start_time = 100;
  row->last_end_time = 900;
  row->total_duration = 1600;
  row->minimum_duration = 80;
  row->maximum_duration = 120;
  row->has_scaled_duration_ns = true;
  row->total_duration_ns = 1600;
  row->minimum_duration_ns = 80;
  row->maximum_duration_ns = 120;
  profile->dispatch_distribution.available = true;
  profile->dispatch_distribution.complete = true;
  profile->dispatch_distribution.comparable = true;
  profile->dispatch_distribution.homogeneous_function = true;
  profile->dispatch_distribution.source_row_type =
      IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_COMMAND_OPERATION;
  profile->dispatch_distribution.physical_device_ordinal = 0;
  profile->dispatch_distribution.time_domain =
      IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_DEVICE_TICK;
  profile->dispatch_distribution.executable_id = 1;
  profile->dispatch_distribution.function_ordinal = 0;
  profile->dispatch_distribution.source_sample_count = 16;
  profile->dispatch_distribution.duration_ns.count = 16;
  profile->dispatch_distribution.duration_ns.total_ns = 1600;
  profile->dispatch_distribution.duration_ns.minimum_ns = 80;
  profile->dispatch_distribution.duration_ns.maximum_ns = 120;
  profile->dispatch_distribution.duration_ns.mean_ns = 100.0;
  profile->dispatch_distribution.duration_ns.p50_ns = 96;
  profile->dispatch_distribution.duration_ns.p90_ns = 112;
  profile->dispatch_distribution.duration_ns.p90_to_p50_delta_ppm = 166666;

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(iree_benchmark_loom_write_benchmark_result_json(
      &benchmark_plan, &case_plan, &policy, &result,
      /*correctness_sample_count=*/1,
      /*correctness_failed_sample_count=*/0, &stream));

  iree_string_view_t root =
      ParseJsonDocument(iree_string_builder_view(&builder));
  iree_string_view_t measurement = LookupObject(root, IREE_SV("measurement"));
  iree_string_view_t timing_interpretation =
      LookupObject(measurement, IREE_SV("timing_interpretation"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(timing_interpretation, IREE_SV("score_meaning")),
      IREE_SV("host_queue_completion_throughput_normalized_batch_time")));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(timing_interpretation, IREE_SV("device_score"))));
  iree_string_view_t warnings =
      LookupObject(timing_interpretation, IREE_SV("warnings"));
  EXPECT_FALSE(JsonArrayContainsString(warnings, IREE_SV("dispatch_overlap")));

  iree_string_view_t profile_replay =
      LookupObject(root, IREE_SV("profile_replay"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(profile_replay, IREE_SV("measurement_relationship")),
      IREE_SV("distinct_execution")));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(profile_replay, IREE_SV("comparison"))));
  iree_string_view_t profiled_dispatch =
      LookupObject(profile_replay, IREE_SV("dispatch_timing"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(profiled_dispatch, IREE_SV("available")), IREE_SV("true")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(profiled_dispatch, IREE_SV("overlapped")), IREE_SV("true")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(profiled_dispatch, IREE_SV("valid_sample_count")),
      IREE_SV("16")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(profiled_dispatch, IREE_SV("span")), IREE_SV("800")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(profiled_dispatch, IREE_SV("total")), IREE_SV("1600")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(profiled_dispatch, IREE_SV("overlap_ratio_ppm")),
      IREE_SV("2000000")));
  iree_string_view_t duration_ns =
      LookupObject(profiled_dispatch, IREE_SV("duration_ns"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(duration_ns, IREE_SV("count")), IREE_SV("16")));
  EXPECT_TRUE(iree_string_view_equal(LookupObject(duration_ns, IREE_SV("mean")),
                                     IREE_SV("100.000")));
  iree_string_view_t dispatch_distribution =
      LookupObject(profiled_dispatch, IREE_SV("dispatch_distribution"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(dispatch_distribution, IREE_SV("complete")),
      IREE_SV("true")));
  iree_string_view_t distribution_duration_ns =
      LookupObject(dispatch_distribution, IREE_SV("duration_ns"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(distribution_duration_ns, IREE_SV("p50")), IREE_SV("96")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(distribution_duration_ns, IREE_SV("p90")), IREE_SV("112")));
  iree_string_view_t profile_warnings =
      LookupObject(profile_replay, IREE_SV("warnings"));
  EXPECT_TRUE(
      JsonArrayContainsString(profile_warnings, IREE_SV("dispatch_overlap")));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(root, IREE_SV("profiled_dispatch_timing"))));
  EXPECT_TRUE(
      iree_string_view_is_empty(TryLookupObject(root, IREE_SV("profile"))));

  row->last_end_time = 2100;
  iree_string_builder_reset(&builder);
  IREE_ASSERT_OK(iree_benchmark_loom_write_benchmark_result_json(
      &benchmark_plan, &case_plan, &policy, &result,
      /*correctness_sample_count=*/1,
      /*correctness_failed_sample_count=*/0, &stream));
  iree_string_view_t serialized_root =
      ParseJsonDocument(iree_string_builder_view(&builder));
  iree_string_view_t serialized_measurement =
      LookupObject(serialized_root, IREE_SV("measurement"));
  iree_string_view_t serialized_interpretation =
      LookupObject(serialized_measurement, IREE_SV("timing_interpretation"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(serialized_interpretation, IREE_SV("score_meaning")),
      IREE_SV("host_queue_completion_throughput_normalized_batch_time")));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(serialized_interpretation, IREE_SV("device_score"))));
  iree_string_view_t serialized_profile_replay =
      LookupObject(serialized_root, IREE_SV("profile_replay"));
  iree_string_view_t comparison =
      LookupObject(serialized_profile_replay, IREE_SV("comparison"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(comparison, IREE_SV("metric")),
      IREE_SV("dispatch_timing.dispatch_distribution.duration_ns.p50")));
  EXPECT_TRUE(
      iree_string_view_equal(LookupObject(comparison, IREE_SV("time_domain")),
                             IREE_SV("device_tick_scaled_ns")));
  EXPECT_TRUE(
      iree_string_view_equal(LookupObject(comparison, IREE_SV("meaning")),
                             IREE_SV("profile_replay_physical_dispatch_p50")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(comparison, IREE_SV("sample_count")), IREE_SV("16")));
  iree_string_view_t serialized_profile_warnings =
      LookupObject(serialized_profile_replay, IREE_SV("warnings"));
  EXPECT_FALSE(JsonArrayContainsString(serialized_profile_warnings,
                                       IREE_SV("dispatch_overlap")));

  iree_string_builder_deinitialize(&builder);
}

TEST(BenchmarkReportTest, WritesArtifactManifestSidecarPath) {
  iree::testing::TempFilePath bundle_dir("loom_benchmark_bundle");
  iree_benchmark_loom_artifact_bundle_options_t bundle_options = {};
  bundle_options.dir = bundle_dir.path_view();
  bundle_options.policy = IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_DEBUG;
  bundle_options.output_format = IREE_BENCHMARK_LOOM_OUTPUT_FORMAT_SNAPSHOT;
  iree_benchmark_loom_artifact_bundle_t bundle = {};
  IREE_ASSERT_OK(iree_benchmark_loom_artifact_bundle_initialize(
      &bundle_options, iree_allocator_system(), &bundle));

  iree_benchmark_loom_hal_context_t context = {};
  context.artifact_bundle = &bundle;
  const char kManifestJson[] = "{\"kind\":\"loom.artifact_manifest\"}";
  iree_io_byte_sequence_t* manifest_sequence = nullptr;
  IREE_ASSERT_OK(CloneByteSpanToSequence(
      iree_make_const_byte_span(kManifestJson, sizeof(kManifestJson) - 1),
      iree_allocator_system(), &manifest_sequence));
  ByteSequencePtr manifest_sequence_owner(manifest_sequence,
                                          iree_io_byte_sequence_release);
  loom_target_emit_sidecar_artifact_t sidecar = {};
  sidecar.kind = LOOM_TARGET_EMIT_SIDECAR_ARTIFACT_KIND_ARTIFACT_MANIFEST;
  sidecar.identifier = IREE_SV("artifact_manifest");
  sidecar.contents = manifest_sequence;

  iree_benchmark_loom_hal_actual_provider_t provider = {};
  provider.context = &context;
  provider.execution.candidate_initialized = true;
  provider.execution.candidate.compiled = true;
  provider.execution.candidate.artifact.sidecars = &sidecar;
  provider.execution.candidate.artifact.sidecar_count = 1;

  iree_benchmark_loom_run_identity_t run = {};
  run.run_id = IREE_SV("run");
  iree_benchmark_loom_candidate_identity_t candidate = {};
  candidate.candidate_id = IREE_SV("candidate");
  IREE_ASSERT_OK(iree_benchmark_loom_write_compiled_artifacts(
      &run, &candidate, &provider, iree_allocator_system()));

  EXPECT_FALSE(iree_string_view_is_empty(provider.artifact_manifest_path));
  EXPECT_EQ(iree_benchmark_loom_artifact_bundle_file_count(
                &bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_ARTIFACT_MANIFEST),
            1u);

  std::string manifest_path(provider.artifact_manifest_path.data,
                            provider.artifact_manifest_path.size);
  EXPECT_NE(manifest_path.find("artifact_manifests"), std::string::npos);
  EXPECT_NE(manifest_path.find("_artifact_manifest.json"), std::string::npos);
  std::ifstream manifest_file(manifest_path);
  ASSERT_TRUE(manifest_file.is_open());
  std::string manifest_contents((std::istreambuf_iterator<char>(manifest_file)),
                                std::istreambuf_iterator<char>());
  EXPECT_EQ(manifest_contents, kManifestJson);

  iree_allocator_free(iree_allocator_system(),
                      provider.artifact_manifest_path_storage);
  iree_benchmark_loom_artifact_bundle_deinitialize(&bundle);
}

TEST(BenchmarkReportTest, WritesManifestFileIdentityErrors) {
  iree::testing::TempFilePath bundle_dir("loom_benchmark_manifest_bundle");
  iree_benchmark_loom_artifact_bundle_options_t bundle_options = {};
  bundle_options.dir = bundle_dir.path_view();
  bundle_options.policy = IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_DEBUG;
  bundle_options.output_format = IREE_BENCHMARK_LOOM_OUTPUT_FORMAT_SNAPSHOT;
  iree_benchmark_loom_artifact_bundle_t bundle = {};
  IREE_ASSERT_OK(iree_benchmark_loom_artifact_bundle_initialize(
      &bundle_options, iree_allocator_system(), &bundle));

  std::string source_path = bundle_dir.path() + "/missing-source.loom";
  iree_benchmark_loom_run_identity_t run = {};
  run.run_id = IREE_SV("run");
  run.source = iree_make_string_view(source_path.data(), source_path.size());
  run.results_path = bundle.results_path;
  run.file_output_dir = bundle.file_output_dir;
  run.profile_artifacts_dir = bundle.profile_artifacts_dir;
  run.artifact_bundle_dir = bundle.dir;
  run.artifact_bundle_policy = IREE_SV("debug");
  iree_benchmark_loom_hal_context_t hal_context = {};
  IREE_ASSERT_OK(iree_benchmark_loom_write_artifact_bundle_manifest(
      &bundle, &run, &hal_context, IREE_SV("source text"),
      IREE_SV("[\"iree-benchmark-loom\"]"), /*dry_run=*/false,
      iree_allocator_system()));

  std::string manifest_path(bundle.manifest_path.data,
                            bundle.manifest_path.size);
  std::ifstream manifest_file(manifest_path);
  ASSERT_TRUE(manifest_file.is_open());
  std::string manifest_contents((std::istreambuf_iterator<char>(manifest_file)),
                                std::istreambuf_iterator<char>());

  iree_string_view_t root = ParseJsonDocument(iree_make_string_view(
      manifest_contents.data(), manifest_contents.size()));
  iree_string_view_t source_identity =
      LookupObject(root, IREE_SV("source_identity"));
  iree_string_view_t source_file =
      LookupObject(source_identity, IREE_SV("file"));
  iree_string_view_t identity = LookupObject(source_file, IREE_SV("identity"));
  ExpectObjectValueEquals(identity, IREE_SV("state"), IREE_SV("stat_failed"));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(identity, IREE_SV("status_string"))));
  EXPECT_TRUE(
      iree_string_view_is_empty(TryLookupObject(identity, IREE_SV("error"))));
  iree_string_view_t status = LookupObject(identity, IREE_SV("status"));
  EXPECT_FALSE(
      iree_string_view_is_empty(LookupObject(status, IREE_SV("code"))));
  EXPECT_FALSE(
      iree_string_view_is_empty(LookupObject(status, IREE_SV("name"))));
  EXPECT_FALSE(
      iree_string_view_is_empty(LookupObject(status, IREE_SV("message"))));

  iree_benchmark_loom_artifact_bundle_deinitialize(&bundle);
}

}  // namespace
}  // namespace loom
