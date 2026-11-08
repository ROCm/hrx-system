// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>
#include <string>

#include "iree/base/internal/json.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/tooling/target/vm/testbench.h"
#include "loom/tools/loom-test/options.h"
#include "loom/tools/loom-test/run.h"
#include "loom/util/stream.h"
#include "loomc/iree.h"
#include "loomc/target/configured.h"

namespace {

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

iree_string_view_t TryLookupObject(iree_string_view_t object,
                                   iree_string_view_t key) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_try_lookup_object_value(object, key, &value));
  return value;
}

iree_string_view_t LookupArrayElement(iree_string_view_t array,
                                      iree_host_size_t index) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_array_get(array, index, &value));
  return value;
}

void ExpectObjectValueEquals(iree_string_view_t object, iree_string_view_t key,
                             iree_string_view_t expected) {
  EXPECT_TRUE(iree_string_view_equal(LookupObject(object, key), expected));
}

iree_string_view_t EnvironmentPath(const char* variable) {
  const char* path = std::getenv(variable);
  EXPECT_NE(path, nullptr) << variable;
  return path != nullptr ? iree_make_cstring_view(path)
                         : IREE_SV("missing-test-data");
}

struct RunOutcome {
  iree::Status status;
  loom_test_run_result_t result = {};
  std::string report;
};

class VmRunTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_status_from_loomc(loomc_target_environment_create_configured(
            loomc_allocator_system(), &target_environment_)));
    ASSERT_NE(target_environment_, nullptr);
    IREE_ASSERT_OK(loom_vm_testbench_initialize(
        target_environment_, iree_allocator_system(), &testbench_));
    configuration_ = {
        .tool_name = "loom-test-vm-run-test",
        .target_environment = target_environment_,
        .function_call_provider =
            {
                .fn = loom_vm_testbench_invocation_provider,
                .user_data = &testbench_,
            },
        .scenario_target_profile =
            {
                .fn = loom_vm_testbench_execution_profile,
                .user_data = &testbench_,
            },
        .scenario_oracle_profile =
            {
                .fn = loom_vm_testbench_execution_profile,
                .user_data = &testbench_,
            },
    };
  }

  void TearDown() override {
    loom_vm_testbench_deinitialize(&testbench_);
    loomc_target_environment_release(target_environment_);
  }

  RunOutcome Run(iree_string_view_t input_path,
                 iree_string_view_t selected_case = iree_string_view_empty(),
                 bool capture_report = false) {
    loom_test_options_t test_options = {};
    loom_test_options_initialize(&test_options);
    test_options.selected_case = selected_case;
    iree_string_builder_t report_builder;
    loom_output_stream_t report_output;
    if (capture_report) {
      iree_string_builder_initialize(iree_allocator_system(), &report_builder);
      loom_output_stream_for_builder(&report_builder, &report_output);
    } else {
      loom_output_stream_null(&report_output);
    }
    const loom_test_file_run_options_t run_options = {
        .configuration = &configuration_,
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

  loomc_target_environment_t* target_environment_ = nullptr;
  loom_vm_testbench_t testbench_ = {};
  loom_test_configuration_t configuration_ = {};
};

TEST_F(VmRunTest, RunsBatchedScenarioProducts) {
  RunOutcome outcome =
      Run(EnvironmentPath("LOOM_TEST_SCENARIO_SOURCE"), IREE_SV("@batched_vm"));
  EXPECT_TRUE(outcome.status.ok()) << outcome.status.ToString();
  EXPECT_EQ(outcome.result.exit_code, 0);
  EXPECT_EQ(outcome.result.case_count, 0u);
  EXPECT_EQ(outcome.result.scenario_count, 1u);
  EXPECT_EQ(outcome.result.trial_count, 5u);
  EXPECT_EQ(outcome.result.failed_trial_count, 0u);
}

TEST_F(VmRunTest, ComparesReturnedReferencesByRecipeIdentity) {
  RunOutcome outcome = Run(EnvironmentPath("LOOM_TEST_SCENARIO_SOURCE"),
                           IREE_SV("@returned_reference_vm"));
  EXPECT_TRUE(outcome.status.ok()) << outcome.status.ToString();
  EXPECT_EQ(outcome.result.exit_code, 0);
  EXPECT_EQ(outcome.result.scenario_count, 1u);
  EXPECT_EQ(outcome.result.trial_count, 2u);
  EXPECT_EQ(outcome.result.failed_trial_count, 0u);
}

TEST_F(VmRunTest, RunsAdmittedRecordsAndSkipsMissingRequirements) {
  RunOutcome outcome = Run(EnvironmentPath("LOOM_TEST_REQUIREMENTS_SOURCE"),
                           iree_string_view_empty(), /*capture_report=*/true);
  EXPECT_TRUE(outcome.status.ok()) << outcome.status.ToString();
  EXPECT_EQ(outcome.result.exit_code, 0);
  EXPECT_EQ(outcome.result.case_count, 3u);
  EXPECT_EQ(outcome.result.scenario_count, 3u);
  EXPECT_EQ(outcome.result.sample_count, 1u);
  EXPECT_EQ(outcome.result.failed_sample_count, 0u);
  EXPECT_EQ(outcome.result.trial_count, 2u);
  EXPECT_EQ(outcome.result.failed_trial_count, 0u);
  EXPECT_EQ(outcome.result.skipped_case_count, 2u);
  EXPECT_EQ(outcome.result.skipped_scenario_count, 2u);

  const iree_string_view_t report = ParseJsonDocument(outcome.report);
  const iree_string_view_t skipped_cases =
      LookupObject(report, IREE_SV("skipped_cases"));
  iree_host_size_t skipped_case_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(skipped_cases, &skipped_case_count));
  ASSERT_EQ(skipped_case_count, 2u);
  const iree_string_view_t missing_case_requirement =
      LookupArrayElement(skipped_cases, 0);
  ExpectObjectValueEquals(missing_case_requirement, IREE_SV("case"),
                          IREE_SV("missing_provider_requirement"));
  ExpectObjectValueEquals(missing_case_requirement, IREE_SV("provider"),
                          IREE_SV("missing.provider"));
  ExpectObjectValueEquals(missing_case_requirement, IREE_SV("op"),
                          IREE_SV("requires"));
  ExpectObjectValueEquals(missing_case_requirement, IREE_SV("code"),
                          IREE_SV("provider_not_registered"));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(missing_case_requirement, IREE_SV("reason"))));

  const iree_string_view_t missing_case_skip =
      LookupArrayElement(skipped_cases, 1);
  ExpectObjectValueEquals(missing_case_skip, IREE_SV("case"),
                          IREE_SV("missing_provider_skip_if"));
  ExpectObjectValueEquals(missing_case_skip, IREE_SV("op"), IREE_SV("skip_if"));
  ExpectObjectValueEquals(missing_case_skip, IREE_SV("display_message"),
                          IREE_SV("human skip display text"));

  const iree_string_view_t skipped_scenarios =
      LookupObject(report, IREE_SV("skipped_scenarios"));
  iree_host_size_t skipped_scenario_count = 0;
  IREE_ASSERT_OK(
      iree_json_array_length(skipped_scenarios, &skipped_scenario_count));
  ASSERT_EQ(skipped_scenario_count, 2u);
  const iree_string_view_t missing_scenario_requirement =
      LookupArrayElement(skipped_scenarios, 0);
  ExpectObjectValueEquals(missing_scenario_requirement, IREE_SV("scenario"),
                          IREE_SV("missing_scenario_requirement"));
  ExpectObjectValueEquals(missing_scenario_requirement, IREE_SV("op"),
                          IREE_SV("requires"));
  ExpectObjectValueEquals(missing_scenario_requirement, IREE_SV("code"),
                          IREE_SV("provider_not_registered"));

  const iree_string_view_t missing_scenario_skip =
      LookupArrayElement(skipped_scenarios, 1);
  ExpectObjectValueEquals(missing_scenario_skip, IREE_SV("scenario"),
                          IREE_SV("missing_scenario_skip_if"));
  ExpectObjectValueEquals(missing_scenario_skip, IREE_SV("op"),
                          IREE_SV("skip_if"));
  ExpectObjectValueEquals(missing_scenario_skip, IREE_SV("display_message"),
                          IREE_SV("scenario requires an available provider"));
}

}  // namespace
