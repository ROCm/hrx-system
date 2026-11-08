// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-test/run.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "iree/base/internal/json.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/tools/loom-test/options.h"
#include "loom/tools/loom-test/scenario_test_provider.h"
#include "loom/util/stream.h"

namespace {

using ::iree::testing::status::StatusIs;
using ::testing::HasSubstr;

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

iree_string_view_t EnvironmentPath(const char* variable) {
  const char* path = std::getenv(variable);
  EXPECT_NE(path, nullptr) << variable;
  return path != nullptr ? iree_make_cstring_view(path)
                         : IREE_SV("missing-test-data");
}

RunOutcome RunFile(const loom_test_configuration_t& configuration,
                   iree_string_view_t input_path,
                   const loom_test_options_t& test_options,
                   bool capture_report = false) {
  iree_string_builder_t report_builder;
  loom_output_stream_t report_output;
  if (capture_report) {
    iree_string_builder_initialize(iree_allocator_system(), &report_builder);
    loom_output_stream_for_builder(&report_builder, &report_output);
  } else {
    loom_output_stream_null(&report_output);
  }
  const loom_test_file_run_options_t run_options = {
      .configuration = &configuration,
      .test_options = &test_options,
      .input_path = input_path,
      .report_output = &report_output,
      .host_allocator = iree_allocator_system(),
  };
  RunOutcome outcome;
  outcome.status = loom_test_run_file(&run_options, &outcome.result);
  if (capture_report) {
    outcome.report.assign(iree_string_builder_buffer(&report_builder),
                          iree_string_builder_size(&report_builder));
    iree_string_builder_deinitialize(&report_builder);
  }
  return outcome;
}

RunOutcome RunFile(iree_string_view_t input_path,
                   const loom_test_options_t& test_options,
                   bool capture_report = false) {
  const loom_test_configuration_t configuration = {
      .tool_name = "loom-test-run-test",
  };
  return RunFile(configuration, input_path, test_options, capture_report);
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
  RunOutcome outcome =
      RunFile(input_path.path_view(), test_options, /*capture_report=*/true);

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

TEST(TestRunTest, LinksSourceAndBytecodeLibraries) {
  const iree_string_view_t input_path =
      EnvironmentPath("LOOM_TEST_LINKED_INPUT_SOURCE");
  const char* library_variables[] = {
      "LOOM_TEST_LIBRARY_SOURCE",
      "LOOM_TEST_LIBRARY_BYTECODE",
  };
  for (const char* library_variable : library_variables) {
    SCOPED_TRACE(library_variable);
    iree_string_view_t library_path = EnvironmentPath(library_variable);
    loom_test_options_t test_options = {};
    loom_test_options_initialize(&test_options);
    test_options.libraries = {
        .count = 1,
        .values = &library_path,
    };

    RunOutcome outcome = RunFile(input_path, test_options);
    EXPECT_TRUE(outcome.status.ok()) << outcome.status.ToString();
    EXPECT_EQ(outcome.result.exit_code, 0);
    EXPECT_EQ(outcome.result.case_count, 2u);
    EXPECT_EQ(outcome.result.sample_count, 2u);
    EXPECT_EQ(outcome.result.failed_sample_count, 0u);
  }
}

TEST(TestRunTest, RejectsDuplicateDefinitionsForSourceAndBytecodeInputs) {
  const struct {
    const char* input_variable;
    const char* library_variable;
  } cases[] = {
      {"LOOM_TEST_DUPLICATE_SOURCE", "LOOM_TEST_LIBRARY_SOURCE"},
      {"LOOM_TEST_DUPLICATE_SOURCE", "LOOM_TEST_LIBRARY_BYTECODE"},
      {"LOOM_TEST_LINKED_INPUT_BYTECODE", "LOOM_TEST_DUPLICATE_SOURCE"},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.input_variable);
    SCOPED_TRACE(test_case.library_variable);
    iree_string_view_t library_path =
        EnvironmentPath(test_case.library_variable);
    loom_test_options_t test_options = {};
    loom_test_options_initialize(&test_options);
    test_options.libraries = {
        .count = 1,
        .values = &library_path,
    };

    ::testing::internal::CaptureStderr();
    RunOutcome outcome =
        RunFile(EnvironmentPath(test_case.input_variable), test_options);
    const std::string diagnostics = ::testing::internal::GetCapturedStderr();
    EXPECT_THAT(outcome.status, StatusIs(iree::StatusCode::kInvalidArgument));
    EXPECT_THAT(diagnostics, HasSubstr("error [SYMBOL/005]"));
    EXPECT_THAT(diagnostics,
                HasSubstr("symbol '@collision' is already defined"));
    EXPECT_THAT(diagnostics, HasSubstr("first definition here"));
    EXPECT_EQ(outcome.result.exit_code, 1);
  }
}

TEST(TestRunTest, ResolvesAuthoredFileReadsAndExplicitWriteRoot) {
  const iree_string_view_t source_path =
      EnvironmentPath("LOOM_TEST_FILE_SOURCE");
  std::ifstream checked_in_source(
      std::string(source_path.data, source_path.size), std::ios::binary);
  ASSERT_TRUE(checked_in_source.good());
  iree::testing::TempFilePath input_path("loom-test-files", ".loom");
  std::ofstream copied_source(input_path.path(), std::ios::binary);
  copied_source << checked_in_source.rdbuf();
  copied_source.close();
  ASSERT_TRUE(input_path.Exists());

  const iree_string_view_t bytecode_fixture_path =
      EnvironmentPath("LOOM_TEST_FILE_BYTECODE");
  std::ifstream bytecode_fixture(
      std::string(bytecode_fixture_path.data, bytecode_fixture_path.size),
      std::ios::binary);
  ASSERT_TRUE(bytecode_fixture.good());
  iree::testing::TempFilePath bytecode_path("loom-test-files", ".loombc");
  std::ofstream copied_bytecode(bytecode_path.path(), std::ios::binary);
  copied_bytecode << bytecode_fixture.rdbuf();
  copied_bytecode.close();
  ASSERT_TRUE(bytecode_path.Exists());

  loom_test_options_t write_options = {};
  loom_test_options_initialize(&write_options);
  write_options.selected_case = IREE_SV("@write_actual");
  RunOutcome missing_root = RunFile(input_path.path_view(), write_options);
  EXPECT_THAT(missing_root.status,
              StatusIs(iree::StatusCode::kFailedPrecondition));

  write_options.file_output_root = IREE_SV("-");
  RunOutcome stdout_root = RunFile(input_path.path_view(), write_options);
  EXPECT_THAT(stdout_root.status, StatusIs(iree::StatusCode::kInvalidArgument));

  const std::filesystem::path output_root =
      std::filesystem::path(input_path.path()).parent_path() / "outputs";
  const std::string output_root_string = output_root.string();
  write_options.file_output_root = iree_make_string_view(
      output_root_string.data(), output_root_string.size());
  RunOutcome write = RunFile(input_path.path_view(), write_options);
  EXPECT_TRUE(write.status.ok()) << write.status.ToString();
  EXPECT_EQ(write.result.exit_code, 0);
  EXPECT_EQ(write.result.sample_count, 1u);
  EXPECT_EQ(write.result.failed_sample_count, 0u);
  const std::filesystem::path written_file = output_root / "nested/actual.npy";
  ASSERT_TRUE(std::filesystem::is_regular_file(written_file));
  EXPECT_GT(std::filesystem::file_size(written_file), 0u);

  loom_test_options_t read_options = {};
  loom_test_options_initialize(&read_options);
  read_options.selected_case = IREE_SV("@read_actual");
  RunOutcome read = RunFile(bytecode_path.path_view(), read_options);
  EXPECT_TRUE(read.status.ok()) << read.status.ToString();
  EXPECT_EQ(read.result.exit_code, 0);
  EXPECT_EQ(read.result.sample_count, 1u);
  EXPECT_EQ(read.result.failed_sample_count, 0u);
}

struct FailurePolicyCase {
  const char* name;
  const char* source_variable;
  const char* selected_case;
  int32_t sample_ordinal;
  const char* qualification;
  bool allow_failure;
  int exit_code;
  iree_host_size_t failed_sample_count;
  iree_host_size_t xfail_count;
  iree_host_size_t xpass_count;
  iree_host_size_t mismatch_count;
};

class FailurePolicyTest : public ::testing::TestWithParam<FailurePolicyCase> {};

TEST_P(FailurePolicyTest, AppliesPolicyToRunnerOutcome) {
  const FailurePolicyCase& test_case = GetParam();
  iree_string_view_t qualification =
      iree_make_cstring_view(test_case.qualification);
  loom_test_options_t test_options = {};
  loom_test_options_initialize(&test_options);
  test_options.selected_case = iree_make_cstring_view(test_case.selected_case);
  test_options.sample_ordinal = test_case.sample_ordinal;
  iree_string_view_list_t* policy = test_case.allow_failure
                                        ? &test_options.allowed_failures
                                        : &test_options.expected_failures;
  *policy = {
      .count = 1,
      .values = &qualification,
  };

  RunOutcome outcome =
      RunFile(EnvironmentPath(test_case.source_variable), test_options);
  EXPECT_TRUE(outcome.status.ok()) << outcome.status.ToString();
  EXPECT_EQ(outcome.result.exit_code, test_case.exit_code);
  EXPECT_EQ(outcome.result.sample_count, 1u);
  EXPECT_EQ(outcome.result.failed_sample_count, test_case.failed_sample_count);
  EXPECT_EQ(outcome.result.xfail_count, test_case.xfail_count);
  EXPECT_EQ(outcome.result.xpass_count, test_case.xpass_count);
  EXPECT_EQ(outcome.result.xfail_mismatch_count, test_case.mismatch_count);
}

INSTANTIATE_TEST_SUITE_P(
    TestRunTest, FailurePolicyTest,
    ::testing::Values(
        FailurePolicyCase{"StrictUnexpectedPass", "LOOM_TEST_SCALAR_SOURCE",
                          "@sampled_choice", 1, "@sampled_choice=TARGET/003",
                          false, 1, 0, 0, 1, 0},
        FailurePolicyCase{"StrictDiagnosticMismatch", "LOOM_TEST_XFAIL_SOURCE",
                          "@equal_mismatch", -1, "@equal_mismatch=EXPECT/999",
                          false, 1, 1, 0, 0, 1},
        FailurePolicyCase{"StrictExpectedFailure", "LOOM_TEST_XFAIL_SOURCE",
                          "@equal_mismatch", -1, "@equal_mismatch=EXPECT/001",
                          false, 0, 1, 1, 0, 0},
        FailurePolicyCase{"StrictAnyExpectedDiagnostic",
                          "LOOM_TEST_XFAIL_SOURCE", "@equal_mismatch", -1,
                          "@equal_mismatch=EXPECT/002,EXPECT/001", false, 0, 1,
                          1, 0, 0},
        FailurePolicyCase{"AllowedPass", "LOOM_TEST_SCALAR_SOURCE",
                          "@sampled_choice", 1, "@sampled_choice=TARGET/003",
                          true, 0, 0, 0, 0, 0},
        FailurePolicyCase{"AllowedExpectedFailure", "LOOM_TEST_XFAIL_SOURCE",
                          "@equal_mismatch", -1, "@equal_mismatch=EXPECT/001",
                          true, 0, 1, 1, 0, 0},
        FailurePolicyCase{"AllowedDiagnosticMismatch", "LOOM_TEST_XFAIL_SOURCE",
                          "@equal_mismatch", -1, "@equal_mismatch=EXPECT/002",
                          true, 1, 1, 0, 0, 1}),
    [](const ::testing::TestParamInfo<FailurePolicyCase>& info) {
      return info.param.name;
    });

TEST(TestRunTest, SkippedScenarioCannotSatisfyFailurePolicy) {
  const bool allow_failure_values[] = {false, true};
  for (bool allow_failure : allow_failure_values) {
    SCOPED_TRACE(allow_failure);
    iree_string_view_t qualification =
        IREE_SV("@missing_scenario_requirement=TARGET/001");
    loom_test_options_t test_options = {};
    loom_test_options_initialize(&test_options);
    test_options.selected_case = IREE_SV("@missing_scenario_requirement");
    iree_string_view_list_t* policy = allow_failure
                                          ? &test_options.allowed_failures
                                          : &test_options.expected_failures;
    *policy = {
        .count = 1,
        .values = &qualification,
    };

    RunOutcome outcome =
        RunFile(EnvironmentPath("LOOM_TEST_REQUIREMENTS_SOURCE"), test_options);
    EXPECT_TRUE(outcome.status.ok()) << outcome.status.ToString();
    EXPECT_EQ(outcome.result.exit_code, 1);
    EXPECT_EQ(outcome.result.scenario_count, 1u);
    EXPECT_EQ(outcome.result.trial_count, 0u);
    EXPECT_EQ(outcome.result.skipped_scenario_count, 1u);
    EXPECT_EQ(outcome.result.xfail_count, 0u);
    EXPECT_EQ(outcome.result.xpass_count, 0u);
    EXPECT_EQ(outcome.result.xfail_mismatch_count, 1u);
  }
}

loom_test_configuration_t ControlledScenarioConfiguration() {
  loom_test_configuration_t configuration = {
      .tool_name = "loom-test-controlled-scenario-test",
  };
  loom_test_scenario_test_provider_configure(&configuration);
  return configuration;
}

TEST(TestRunTest, ReportsControlledScenarioFailuresThroughTypedCounts) {
  const struct {
    const char* name;
    iree_host_size_t trial_count;
  } cases[] = {
      {"@wrong_result", 3},    {"@wrong_post_state", 3}, {"@wrong_guard", 3},
      {"@wrong_reference", 3}, {"@missing_outcome", 1},
  };
  const loom_test_configuration_t configuration =
      ControlledScenarioConfiguration();
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    loom_test_options_t test_options = {};
    loom_test_options_initialize(&test_options);
    test_options.selected_case = iree_make_cstring_view(test_case.name);
    RunOutcome outcome = RunFile(
        configuration, EnvironmentPath("LOOM_TEST_SCENARIO_FAILURES_SOURCE"),
        test_options);
    EXPECT_TRUE(outcome.status.ok()) << outcome.status.ToString();
    EXPECT_EQ(outcome.result.exit_code, 1);
    EXPECT_EQ(outcome.result.case_count, 0u);
    EXPECT_EQ(outcome.result.scenario_count, 1u);
    EXPECT_EQ(outcome.result.trial_count, test_case.trial_count);
    EXPECT_EQ(outcome.result.failed_trial_count, 1u);
  }
}

TEST(TestRunTest, QualifiesControlledScenarioFailure) {
  const struct {
    const char* diagnostic;
    int exit_code;
    iree_host_size_t xfail_count;
    iree_host_size_t mismatch_count;
  } cases[] = {
      {"@wrong_post_state=EXPECT/002", 0, 1, 0},
      {"@wrong_post_state=EXPECT/001", 1, 0, 1},
  };
  const loom_test_configuration_t configuration =
      ControlledScenarioConfiguration();
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.diagnostic);
    iree_string_view_t expected_failure =
        iree_make_cstring_view(test_case.diagnostic);
    loom_test_options_t test_options = {};
    loom_test_options_initialize(&test_options);
    test_options.selected_case = IREE_SV("@wrong_post_state");
    test_options.expected_failures = {
        .count = 1,
        .values = &expected_failure,
    };
    RunOutcome outcome = RunFile(
        configuration, EnvironmentPath("LOOM_TEST_SCENARIO_FAILURES_SOURCE"),
        test_options);
    EXPECT_TRUE(outcome.status.ok()) << outcome.status.ToString();
    EXPECT_EQ(outcome.result.exit_code, test_case.exit_code);
    EXPECT_EQ(outcome.result.failed_trial_count, 1u);
    EXPECT_EQ(outcome.result.xfail_count, test_case.xfail_count);
    EXPECT_EQ(outcome.result.xpass_count, 0u);
    EXPECT_EQ(outcome.result.xfail_mismatch_count, test_case.mismatch_count);
  }
}

TEST(TestRunTest, AppliesControlledScenarioRequirements) {
  const struct {
    const char* name;
    iree_host_size_t trial_count;
    iree_host_size_t skipped_count;
  } cases[] = {
      {"@admitted", 1, 0},    {"@unsatisfied", 0, 1},      {"@matched", 0, 1},
      {"@unavailable", 0, 1}, {"@unavailable_skip", 0, 1},
  };
  const loom_test_configuration_t configuration =
      ControlledScenarioConfiguration();
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    loom_test_options_t test_options = {};
    loom_test_options_initialize(&test_options);
    test_options.selected_case = iree_make_cstring_view(test_case.name);
    RunOutcome outcome =
        RunFile(configuration,
                EnvironmentPath("LOOM_TEST_SCENARIO_REQUIREMENTS_SOURCE"),
                test_options);
    EXPECT_TRUE(outcome.status.ok()) << outcome.status.ToString();
    EXPECT_EQ(outcome.result.exit_code, 0);
    EXPECT_EQ(outcome.result.scenario_count, 1u);
    EXPECT_EQ(outcome.result.trial_count, test_case.trial_count);
    EXPECT_EQ(outcome.result.failed_trial_count, 0u);
    EXPECT_EQ(outcome.result.skipped_scenario_count, test_case.skipped_count);
  }
}

TEST(TestRunTest, PropagatesControlledRequirementProviderFailure) {
  const loom_test_configuration_t configuration =
      ControlledScenarioConfiguration();
  loom_test_options_t test_options = {};
  loom_test_options_initialize(&test_options);
  test_options.selected_case = IREE_SV("@provider_error");
  RunOutcome outcome = RunFile(
      configuration, EnvironmentPath("LOOM_TEST_SCENARIO_REQUIREMENTS_SOURCE"),
      test_options);
  EXPECT_THAT(outcome.status, StatusIs(iree::StatusCode::kUnavailable));
  EXPECT_THAT(outcome.status.ToString(),
              HasSubstr("controlled requirement provider failed"));
  EXPECT_EQ(outcome.result.exit_code, 1);
}

TEST(TestRunTest, PrintsAgentDocumentationDirectly) {
  ::testing::internal::CaptureStdout();
  loom_test_print_agents_markdown(stdout);
  const std::string documentation = ::testing::internal::GetCapturedStdout();
  EXPECT_THAT(documentation, HasSubstr("## loom-test"));
  EXPECT_THAT(documentation, HasSubstr("### Common flows"));
}

}  // namespace
