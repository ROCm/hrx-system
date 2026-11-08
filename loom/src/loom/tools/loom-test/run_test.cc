// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-test/run.h"

#include <fstream>
#include <string>

#include "iree/base/internal/json.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/tools/loom-test/options.h"
#include "loom/util/stream.h"

namespace {

struct RunOutcome {
  iree::Status status;
  loom_test_run_result_t result = {};
  std::string report;
};

iree_string_view_t ParseJsonDocument(const std::string& json) {
  iree_string_view_t cursor = iree_make_string_view(
      json.data(), static_cast<iree_host_size_t>(json.size()));
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_consume_value(&cursor, &value));
  IREE_EXPECT_OK(iree_json_consume_insignificant(&cursor));
  EXPECT_TRUE(iree_string_view_is_empty(cursor));
  return value;
}

iree_string_view_t LookupObject(iree_string_view_t object,
                                iree_string_view_t key) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_lookup_object_value(object, key, &value));
  return value;
}

RunOutcome RunFile(iree_string_view_t input_path,
                   const loom_test_options_t& test_options) {
  const loom_test_configuration_t configuration = {
      .tool_name = "loom-test-run-test",
  };
  iree_string_builder_t report_builder;
  iree_string_builder_initialize(iree_allocator_system(), &report_builder);
  loom_output_stream_t report_output;
  loom_output_stream_for_builder(&report_builder, &report_output);
  const loom_test_file_run_options_t run_options = {
      .configuration = &configuration,
      .test_options = &test_options,
      .input_path = input_path,
      .report_output = &report_output,
      .host_allocator = iree_allocator_system(),
  };
  RunOutcome outcome;
  outcome.status = loom_test_run_file(&run_options, &outcome.result);
  outcome.report.assign(iree_string_builder_buffer(&report_builder),
                        iree_string_builder_size(&report_builder));
  iree_string_builder_deinitialize(&report_builder);
  return outcome;
}

TEST(TestRunTest, RunsSelectedSampleThroughStructuredBoundary) {
  iree::testing::TempFilePath input_path("loom-test-run", ".loom");
  std::ofstream input_file(input_path.path());
  input_file << R"(
check.case @sampled_choice {
  %value = check.param.choice values([5, 7]) : i32
  check.expect.equal actual(%value) expected(%value) : i32
  check.return
}
)";
  input_file.close();
  ASSERT_TRUE(input_path.Exists());

  loom_test_options_t test_options = {};
  loom_test_options_initialize(&test_options);
  test_options.selected_case = IREE_SV("@sampled_choice");
  test_options.sample_ordinal = 1;
  RunOutcome outcome = RunFile(input_path.path_view(), test_options);

  EXPECT_TRUE(outcome.status.ok()) << outcome.status.ToString();
  EXPECT_EQ(outcome.result.exit_code, 0);
  EXPECT_EQ(outcome.result.case_count, 1u);
  EXPECT_EQ(outcome.result.scenario_count, 0u);
  EXPECT_EQ(outcome.result.sample_count, 1u);
  EXPECT_EQ(outcome.result.failed_sample_count, 0u);
  EXPECT_EQ(outcome.result.planning_issue_count, 0u);

  const iree_string_view_t report = ParseJsonDocument(outcome.report);
  EXPECT_TRUE(iree_string_view_equal(LookupObject(report, IREE_SV("format")),
                                     IREE_SV("loom.test.v0")));
}

TEST(TestRunTest, QualifiesExpectedFailureWithoutProcessOrchestration) {
  iree::testing::TempFilePath input_path("loom-test-xfail", ".loom");
  std::ofstream input_file(input_path.path());
  input_file << R"(
check.case @equal_mismatch {
  %actual = check.literal value(43) : i32
  %expected = check.literal value(42) : i32
  check.expect.equal actual(%actual) expected(%expected) : i32
  check.return
}
)";
  input_file.close();
  ASSERT_TRUE(input_path.Exists());

  iree_string_view_t expected_failure = IREE_SV("@equal_mismatch=EXPECT/001");
  loom_test_options_t test_options = {};
  loom_test_options_initialize(&test_options);
  test_options.expected_failures = {
      .count = 1,
      .values = &expected_failure,
  };
  RunOutcome outcome = RunFile(input_path.path_view(), test_options);

  EXPECT_TRUE(outcome.status.ok()) << outcome.status.ToString();
  EXPECT_EQ(outcome.result.exit_code, 0);
  EXPECT_EQ(outcome.result.sample_count, 1u);
  EXPECT_EQ(outcome.result.failed_sample_count, 1u);
  EXPECT_EQ(outcome.result.xfail_count, 1u);
  EXPECT_EQ(outcome.result.xpass_count, 0u);
  EXPECT_EQ(outcome.result.xfail_mismatch_count, 0u);
}

}  // namespace
