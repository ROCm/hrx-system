// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-test/xfail.h"

#include "iree/base/internal/json.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"
#include "loom/tooling/testbench/expectation.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

namespace {

using ::iree::testing::status::StatusIs;
using ::testing::HasSubstr;

iree_string_view_list_t MakeList(iree_string_view_t* values,
                                 iree_host_size_t count) {
  return iree_string_view_list_t{
      .count = count,
      .values = values,
  };
}

iree_string_view_t LookupObject(iree_string_view_t object,
                                iree_string_view_t key) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_lookup_object_value(object, key, &value));
  return value;
}

iree_string_view_t LookupArrayElement(iree_string_view_t array,
                                      iree_host_size_t index) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_array_get(array, index, &value));
  return value;
}

class XfailTest : public ::testing::Test {
 protected:
  void TearDown() override {
    loom_test_xfail_list_deinitialize(&list_, iree_allocator_system());
  }

  void Initialize(iree_string_view_list_t strict_values,
                  iree_string_view_list_t allow_failure_values) {
    IREE_ASSERT_OK(loom_test_xfail_list_initialize(
        strict_values, allow_failure_values, iree_allocator_system(), &list_));
  }

  loom_test_xfail_list_t list_ = {};
};

TEST_F(XfailTest, ParsesPoliciesAndDiagnosticSets) {
  iree_string_view_t strict[] = {
      IREE_SV("@strict=EXPECT/001,EXPECT/007"),
  };
  iree_string_view_t allowed[] = {
      IREE_SV("@allowed=TARGET/003"),
  };
  Initialize(MakeList(strict, IREE_ARRAYSIZE(strict)),
             MakeList(allowed, IREE_ARRAYSIZE(allowed)));

  ASSERT_EQ(list_.count, 2u);
  const loom_test_xfail_t* strict_xfail =
      loom_test_xfail_list_find(&list_, IREE_SV("strict"));
  ASSERT_NE(strict_xfail, nullptr);
  EXPECT_EQ(strict_xfail->policy, LOOM_TEST_XFAIL_POLICY_STRICT);
  ASSERT_EQ(strict_xfail->diagnostic_count, 2u);
  EXPECT_EQ(strict_xfail->diagnostic_refs[0],
            LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_EXPECT, 1));
  EXPECT_EQ(strict_xfail->diagnostic_refs[1],
            LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_EXPECT, 7));

  const loom_test_xfail_t* allowed_xfail =
      loom_test_xfail_list_find(&list_, IREE_SV("allowed"));
  ASSERT_NE(allowed_xfail, nullptr);
  EXPECT_EQ(allowed_xfail->policy, LOOM_TEST_XFAIL_POLICY_ALLOW_PASS);
  EXPECT_EQ(allowed_xfail->diagnostic_refs[0],
            LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_TARGET, 3));
  EXPECT_EQ(loom_test_xfail_list_find(&list_, IREE_SV("missing")), nullptr);
}

TEST(XfailParsingTest, RejectsMalformedAndAmbiguousPolicies) {
  const struct {
    const char* strict;
    const char* allowed;
    const char* message;
  } cases[] = {
      {"strict=EXPECT/001", "", "must use"},
      {"@strict=EXPECT/1", "", "canonical diagnostic identity"},
      {"@strict=EXPECT/001,", "", "empty diagnostic identity"},
      {"@strict=EXPECT/001,EXPECT/001", "", "repeats diagnostic"},
      {"@same=EXPECT/001", "@same=EXPECT/002", "repeats record"},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.strict);
    iree_string_view_t strict_value = iree_make_cstring_view(test_case.strict);
    iree_string_view_t allowed_value =
        iree_make_cstring_view(test_case.allowed);
    const iree_string_view_list_t strict = strict_value.size == 0
                                               ? MakeList(nullptr, 0)
                                               : MakeList(&strict_value, 1);
    const iree_string_view_list_t allowed = allowed_value.size == 0
                                                ? MakeList(nullptr, 0)
                                                : MakeList(&allowed_value, 1);
    loom_test_xfail_list_t list = {};
    iree::Status status = loom_test_xfail_list_initialize(
        strict, allowed, iree_allocator_system(), &list);
    EXPECT_THAT(status, StatusIs(iree::StatusCode::kInvalidArgument));
    EXPECT_THAT(status.ToString(), HasSubstr(test_case.message));
    EXPECT_EQ(list.count, 0u);
    EXPECT_EQ(list.values, nullptr);
  }
}

TEST_F(XfailTest, ValidatesSelectedRecordExactlyOnce) {
  iree_string_view_t strict[] = {
      IREE_SV("@case_a=EXPECT/001"),
  };
  Initialize(MakeList(strict, IREE_ARRAYSIZE(strict)), MakeList(nullptr, 0));

  loom_testbench_case_plan_t cases[2] = {};
  cases[0].name = IREE_SV("case_a");
  cases[1].name = IREE_SV("case_b");
  loom_testbench_module_plan_t plan = {
      .cases = cases,
      .case_count = IREE_ARRAYSIZE(cases),
  };
  IREE_EXPECT_OK(loom_test_validate_xfails(&list_, &plan, IREE_SV("case_a")));

  iree::Status unselected =
      loom_test_validate_xfails(&list_, &plan, IREE_SV("case_b"));
  EXPECT_THAT(unselected, StatusIs(iree::StatusCode::kInvalidArgument));
  EXPECT_THAT(unselected.ToString(), HasSubstr("is not selected"));

  plan.case_count = 0;
  iree::Status missing =
      loom_test_validate_xfails(&list_, &plan, iree_string_view_empty());
  EXPECT_THAT(missing, StatusIs(iree::StatusCode::kNotFound));
  EXPECT_THAT(missing.ToString(), HasSubstr("matched 0 check records"));

  plan.case_count = IREE_ARRAYSIZE(cases);
  loom_testbench_scenario_plan_t scenario = {
      .name = IREE_SV("case_a"),
  };
  plan.scenarios = &scenario;
  plan.scenario_count = 1;
  iree::Status ambiguous =
      loom_test_validate_xfails(&list_, &plan, iree_string_view_empty());
  EXPECT_THAT(ambiguous, StatusIs(iree::StatusCode::kNotFound));
  EXPECT_THAT(ambiguous.ToString(), HasSubstr("matched 2 check records"));
}

TEST_F(XfailTest, CapturesExpectationDiagnosticsAndAcceptsCompileFailure) {
  iree_string_view_t strict[] = {
      IREE_SV("@case_a=EXPECT/001,EXPECT/007"),
  };
  Initialize(MakeList(strict, IREE_ARRAYSIZE(strict)), MakeList(nullptr, 0));
  loom_test_xfail_t* xfail = &list_.values[0];

  loom_test_diagnostic_capture_t capture = {};
  loom_test_diagnostic_capture_begin(&capture, xfail);
  loom_testbench_expectation_failure_t failures[] = {
      {.diagnostic_ref = LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_EXPECT, 6)},
      {.diagnostic_ref = LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_EXPECT, 7)},
  };
  loom_testbench_expectation_report_t report = {
      .failures = failures,
      .failure_count = IREE_ARRAYSIZE(failures),
  };
  loom_test_capture_expectation_report(&capture, &report);
  EXPECT_EQ(capture.first_error_ref,
            LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_EXPECT, 6));
  EXPECT_TRUE(capture.matched_expected_diagnostic);

  iree_status_t status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                          "expected compilation failure");
  EXPECT_TRUE(
      loom_test_xfail_try_accept_compile_failure(xfail, &capture, &status));
  IREE_EXPECT_OK(status);
  EXPECT_EQ(xfail->outcome, LOOM_TEST_XFAIL_OUTCOME_EXPECTED_FAILURE);
  EXPECT_EQ(xfail->observed_diagnostic_ref,
            LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_EXPECT, 6));

  status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "wrong status");
  EXPECT_FALSE(
      loom_test_xfail_try_accept_compile_failure(xfail, &capture, &status));
  iree_status_free(status);
}

TEST_F(XfailTest, CountsOutcomesAndWritesStableReport) {
  iree_string_view_t strict[] = {
      IREE_SV("@expected=EXPECT/001"),
      IREE_SV("@unexpected=EXPECT/002"),
      IREE_SV("@mismatch=EXPECT/003"),
  };
  iree_string_view_t allowed[] = {
      IREE_SV("@allowed=EXPECT/004"),
  };
  Initialize(MakeList(strict, IREE_ARRAYSIZE(strict)),
             MakeList(allowed, IREE_ARRAYSIZE(allowed)));

  loom_test_diagnostic_capture_t capture = {
      .first_error_ref = LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_EXPECT, 1),
  };
  loom_test_finish_xfail(&list_.values[0], &capture,
                         LOOM_TEST_XFAIL_OUTCOME_EXPECTED_FAILURE, 3, 5);
  loom_test_finish_xfail(&list_.values[1], &capture,
                         LOOM_TEST_XFAIL_OUTCOME_UNEXPECTED_PASS, 0, 0);
  capture.first_error_ref = LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_EXPECT, 6);
  loom_test_finish_xfail(&list_.values[2], &capture,
                         LOOM_TEST_XFAIL_OUTCOME_DIAGNOSTIC_MISMATCH, 0, 0);
  capture.first_error_ref = LOOM_ERROR_REF_NONE;
  loom_test_finish_xfail(&list_.values[3], &capture,
                         LOOM_TEST_XFAIL_OUTCOME_ALLOWED_PASS, 0, 0);

  const loom_test_xfail_counts_t counts = loom_test_count_xfails(&list_);
  EXPECT_EQ(counts.xfail_count, 1u);
  EXPECT_EQ(counts.xpass_count, 1u);
  EXPECT_EQ(counts.mismatch_count, 1u);
  EXPECT_EQ(counts.accepted_failed_sample_count, 3u);
  EXPECT_EQ(counts.accepted_failed_trial_count, 5u);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  loom_json_object_writer_t object;
  IREE_ASSERT_OK(loom_json_object_begin(&stream, &object));
  IREE_ASSERT_OK(loom_test_write_xfails_json(&list_, &object));
  IREE_ASSERT_OK(loom_json_object_end(&object));

  iree_string_view_t root = iree_string_builder_view(&builder);
  iree_string_view_t xfails = LookupObject(root, IREE_SV("xfails"));
  iree_host_size_t count = 0;
  IREE_ASSERT_OK(iree_json_array_length(xfails, &count));
  ASSERT_EQ(count, 4u);
  iree_string_view_t mismatch = LookupArrayElement(xfails, 2);
  EXPECT_TRUE(iree_string_view_equal(LookupObject(mismatch, IREE_SV("record")),
                                     IREE_SV("@mismatch")));
  EXPECT_TRUE(iree_string_view_equal(LookupObject(mismatch, IREE_SV("outcome")),
                                     IREE_SV("diagnostic_mismatch")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(mismatch, IREE_SV("observed_diagnostic")),
      IREE_SV("EXPECT/006")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(LookupArrayElement(xfails, 3), IREE_SV("policy")),
      IREE_SV("allow_failure")));
  iree_string_builder_deinitialize(&builder);
}

}  // namespace
