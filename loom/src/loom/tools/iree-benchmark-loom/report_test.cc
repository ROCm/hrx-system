// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/report.h"

#include <fstream>
#include <string>

#include "iree/base/internal/json.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace loom {
namespace {

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
  report->backend_name = IREE_SV("amdgpu-hal");
  report->target_family_name = IREE_SV("amdgpu");
  report->target_key = IREE_SV("gfx1100");
  report->function_name = IREE_SV("candidate_kernel");
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
      /*materialized_copy_count=*/3);
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
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(root, IREE_SV("static_summary"))));
  iree_string_view_t compile_report =
      LookupObject(root, IREE_SV("compile_report"));
  ExpectObjectValueEquals(compile_report, IREE_SV("artifact_kind"),
                          IREE_SV("vm-archive"));
  ExpectObjectValueEquals(compile_report, IREE_SV("backend"),
                          IREE_SV("amdgpu-hal"));
  ExpectObjectValueEquals(compile_report, IREE_SV("target_family"),
                          IREE_SV("amdgpu"));
  ExpectObjectValueEquals(compile_report, IREE_SV("target_key"),
                          IREE_SV("gfx1100"));
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
  iree_string_view_t emission =
      LookupObject(compile_report, IREE_SV("emission"));
  ExpectObjectValueEquals(emission, IREE_SV("code_byte_count"), IREE_SV("148"));
  iree_string_view_t memory = LookupObject(compile_report, IREE_SV("memory"));
  ExpectObjectValueEquals(memory, IREE_SV("local_bytes"), IREE_SV("256"));

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
  result.data_cache.binding_count = 2;
  result.data_cache.binding_ring_count = 6;
  result.data_cache.command_buffer_ring_count = 1;
  result.data_cache.dispatches_per_batch = 6;
  result.data_cache.requested_min_ring_bytes = 4096;
  result.data_cache.binding_set_bytes = 64;
  result.data_cache.binding_ring_bytes = 384;
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
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(root, IREE_SV("logical_operations_per_batch")),
      IREE_SV("1")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(root, IREE_SV("physical_dispatches_per_batch")),
      IREE_SV("6")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(root, IREE_SV("physical_dispatches_per_logical_operation")),
      IREE_SV("6")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(root, IREE_SV("measured_logical_operation_count")),
      IREE_SV("3")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(root, IREE_SV("measured_physical_dispatch_count")),
      IREE_SV("18")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(root, IREE_SV("mean_physical_dispatch_duration_ns")),
      IREE_SV("50.000")));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(root, IREE_SV("dispatch_timing_ns"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(root, IREE_SV("measured_dispatch_count"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(root, IREE_SV("measured_operation_count"))));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(root, IREE_SV("artifact_manifest_path")),
      result.artifact_manifest_path));

  iree_string_view_t timing_interpretation =
      LookupObject(root, IREE_SV("timing_interpretation"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(timing_interpretation, IREE_SV("score")),
      IREE_SV("operation_timing_ns")));
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
  loom_target_emit_sidecar_artifact_t sidecar = {};
  sidecar.kind = LOOM_TARGET_EMIT_SIDECAR_ARTIFACT_KIND_ARTIFACT_MANIFEST;
  sidecar.identifier = IREE_SV("artifact_manifest");
  sidecar.contents =
      iree_make_const_byte_span(kManifestJson, sizeof(kManifestJson) - 1);

  iree_benchmark_loom_hal_actual_provider_t provider = {};
  provider.context = &context;
  provider.sample_compilation = IREE_SV("once");
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

}  // namespace
}  // namespace loom
