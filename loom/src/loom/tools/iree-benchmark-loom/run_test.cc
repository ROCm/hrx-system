// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/run.h"

#include <fstream>
#include <string>

#include "iree/base/internal/json.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/target/low_descriptor_registry_core_test.h"

namespace loom {
namespace {

typedef struct event_collector_t {
  iree_host_size_t event_count;
  iree_benchmark_loom_event_t events[16];
} event_collector_t;

static iree_status_t collect_event(void* user_data,
                                   const iree_benchmark_loom_event_t* event) {
  event_collector_t* collector = (event_collector_t*)user_data;
  if (collector->event_count == IREE_ARRAYSIZE(collector->events)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "benchmark event collector is full");
  }
  collector->events[collector->event_count++] = *event;
  return iree_ok_status();
}

static iree_status_t InitializeLowDescriptorRegistry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  (void)user_data;
  loom_target_core_test_low_descriptor_registry_initialize(out_registry);
  return iree_ok_status();
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

static iree_string_view_t ParseJsonRow(const std::string& row) {
  return ParseJsonDocument(iree_make_string_view(
      row.data(), static_cast<iree_host_size_t>(row.size())));
}

static void ExpectJsonRowKind(const std::string& row,
                              iree_string_view_t expected_kind) {
  iree_string_view_t root = ParseJsonRow(row);
  iree_string_view_t actual_kind = LookupObject(root, IREE_SV("row"));
  EXPECT_TRUE(iree_string_view_equal(actual_kind, expected_kind));
}

TEST(BenchmarkRunTest, RunsFileThroughCallerOwnedSink) {
  iree::testing::TempFilePath input_path("loom-benchmark-run", ".loom");
  std::ofstream input_file(input_path.path());
  input_file << R"(
check.case @sampled_choice {
  %value = check.param.choice values([5, 7]) name("value") : i32
  check.expect.equal actual(%value) expected(%value) : i32
  check.return
}

check.benchmark<@sampled_choice> @sampled_choice_value7 {value = 7}
)";
  input_file.close();
  ASSERT_TRUE(input_path.Exists());

  iree::testing::TempFilePath output_path("loom-benchmark-run-output", ".json");
  iree_benchmark_loom_options_t benchmark_options = {};
  iree_benchmark_loom_options_initialize(&benchmark_options);
  benchmark_options.selected_benchmark = IREE_SV("@sampled_choice_value7");
  benchmark_options.measure = IREE_SV("case_end_to_end");
  benchmark_options.output = output_path.path_view();
  benchmark_options.iterations = 1;
  benchmark_options.warmup_iterations = 0;

  event_collector_t collector = {};
  iree_benchmark_loom_event_sink_t event_sink = {};
  event_sink.emit = collect_event;
  event_sink.user_data = &collector;
  iree_benchmark_loom_configuration_t configuration = {};
  configuration.tool_name = "iree-benchmark-loom-test";
  configuration.initialize_low_descriptor_registry =
      (loom_run_initialize_low_descriptor_registry_callback_t){
          /*.fn=*/InitializeLowDescriptorRegistry,
      };
  iree_benchmark_loom_file_run_options_t run_options = {};
  run_options.configuration = &configuration;
  run_options.benchmark_options = &benchmark_options;
  run_options.input_path = input_path.path_view();
  run_options.command_line_json = IREE_SV("[\"iree-benchmark-loom-test\"]");
  run_options.event_sink = &event_sink;
  run_options.host_allocator = iree_allocator_system();

  iree_benchmark_loom_run_result_t run_result = {};
  IREE_ASSERT_OK(iree_benchmark_loom_run_file(&run_options, &run_result));

  EXPECT_EQ(run_result.exit_code, 0);
  EXPECT_FALSE(output_path.Exists());
  ASSERT_EQ(collector.event_count, 6u);
  EXPECT_EQ(collector.events[0].kind, IREE_BENCHMARK_LOOM_EVENT_RUN);
  EXPECT_EQ(collector.events[1].kind, IREE_BENCHMARK_LOOM_EVENT_PLAN);
  EXPECT_EQ(collector.events[2].kind, IREE_BENCHMARK_LOOM_EVENT_SAMPLE);
  EXPECT_EQ(collector.events[3].kind,
            IREE_BENCHMARK_LOOM_EVENT_BENCHMARK_RESULT);
  EXPECT_EQ(collector.events[4].kind, IREE_BENCHMARK_LOOM_EVENT_PROFILE);
  EXPECT_EQ(collector.events[5].kind, IREE_BENCHMARK_LOOM_EVENT_SUMMARY);
  EXPECT_EQ(collector.events[5].summary.planned_case_count, 1u);
  EXPECT_EQ(collector.events[5].summary.planned_benchmark_count, 1u);
  EXPECT_EQ(collector.events[5].summary.selected_benchmark_count, 1u);
  EXPECT_EQ(collector.events[5].summary.logical_sample_count, 1u);
  EXPECT_EQ(collector.events[5].summary.work_item_count, 1u);
  EXPECT_EQ(collector.events[5].summary.correctness_sample_count, 1u);
  EXPECT_EQ(collector.events[5].summary.correctness_failed_sample_count, 0u);
}

TEST(BenchmarkRunTest, WritesJsonlOutputThroughConfiguredSink) {
  iree::testing::TempFilePath input_path("loom-benchmark-run-jsonl", ".loom");
  std::ofstream input_file(input_path.path());
  input_file << R"(
check.case @sampled_choice {
  %value = check.param.choice values([5, 7]) name("value") : i32
  check.expect.equal actual(%value) expected(%value) : i32
  check.return
}

check.benchmark<@sampled_choice> @sampled_choice_value7 {value = 7}
)";
  input_file.close();
  ASSERT_TRUE(input_path.Exists());

  iree::testing::TempFilePath output_path("loom-benchmark-run", ".jsonl");
  iree_benchmark_loom_options_t benchmark_options = {};
  iree_benchmark_loom_options_initialize(&benchmark_options);
  benchmark_options.selected_benchmark = IREE_SV("@sampled_choice_value7");
  benchmark_options.measure = IREE_SV("case_end_to_end");
  benchmark_options.output = output_path.path_view();
  benchmark_options.output_format = IREE_BENCHMARK_LOOM_OUTPUT_FORMAT_JSONL;
  benchmark_options.iterations = 1;
  benchmark_options.warmup_iterations = 0;

  iree_benchmark_loom_configuration_t configuration = {};
  configuration.tool_name = "iree-benchmark-loom-test";
  configuration.initialize_low_descriptor_registry =
      (loom_run_initialize_low_descriptor_registry_callback_t){
          /*.fn=*/InitializeLowDescriptorRegistry,
      };
  iree_benchmark_loom_file_run_options_t run_options = {};
  run_options.configuration = &configuration;
  run_options.benchmark_options = &benchmark_options;
  run_options.input_path = input_path.path_view();
  run_options.command_line_json = IREE_SV("[\"iree-benchmark-loom-test\"]");
  run_options.host_allocator = iree_allocator_system();

  iree_benchmark_loom_run_result_t run_result = {};
  IREE_ASSERT_OK(iree_benchmark_loom_run_file(&run_options, &run_result));

  EXPECT_EQ(run_result.exit_code, 0);
  ASSERT_TRUE(output_path.Exists());
  std::ifstream output_file(output_path.path());
  std::string run_row;
  std::string plan_row;
  std::string sample_row;
  std::string benchmark_row;
  std::string summary_row;
  std::string trailing_row;
  std::getline(output_file, run_row);
  std::getline(output_file, plan_row);
  std::getline(output_file, sample_row);
  std::getline(output_file, benchmark_row);
  std::getline(output_file, summary_row);
  std::getline(output_file, trailing_row);
  ExpectJsonRowKind(run_row, IREE_SV("run"));
  ExpectJsonRowKind(plan_row, IREE_SV("plan"));
  ExpectJsonRowKind(sample_row, IREE_SV("sample"));
  ExpectJsonRowKind(benchmark_row, IREE_SV("benchmark"));
  ExpectJsonRowKind(summary_row, IREE_SV("summary"));
  EXPECT_TRUE(trailing_row.empty());

  iree_string_view_t benchmark_root = ParseJsonRow(benchmark_row);
  iree_string_view_t work_item_index =
      LookupObject(benchmark_root, IREE_SV("work_item_index"));
  EXPECT_TRUE(iree_string_view_equal(work_item_index, IREE_SV("0")));
  iree_string_view_t benchmark_result =
      LookupObject(benchmark_root, IREE_SV("benchmark_result"));
  iree_string_view_t benchmark_status =
      LookupObject(benchmark_result, IREE_SV("status"));
  EXPECT_TRUE(iree_string_view_equal(benchmark_status, IREE_SV("ok")));
}

}  // namespace
}  // namespace loom
