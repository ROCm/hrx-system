// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/verify.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/pass/ops.h"
#include "loom/pass/test/harness.h"

namespace loom {
namespace {

class PassVerifyTest : public PassTestHarness {};

TEST_F(PassVerifyTest, VerifiesModuleAndFuncPipelines) {
  IREE_ASSERT_OK(Verify(IREE_SV(
      "pass.pipeline<module> @cleanup pipeline {\n"
      "  test.module-noop\n"
      "  for func {\n"
      "    test.noop\n"
      "    repeat fixed(count = 2) {\n"
      "      test.options(count = 3, mode = beta, string = \"payload\")\n"
      "    }\n"
      "  }\n"
      "}\n"
      "\n"
      "pass.pipeline<func> @function_cleanup pipeline {\n"
      "  where name(value = \"matmul\") {\n"
      "    test.noop\n"
      "  }\n"
      "}\n"
      "\n"
      "pass.pipeline<module> @debug pipeline {\n"
      "  call @cleanup\n"
      "  fail \"expected cleanup to run\"\n"
      "  halt \"inspect IR\"\n"
      "}\n")));
}

TEST_F(PassVerifyTest, AllowsModulesWithoutPassPipelines) {
  IREE_ASSERT_OK(Verify(IREE_SV("")));
}

TEST_F(PassVerifyTest, RejectsNullPipelineOp) {
  loom_module_t* module = Parse(IREE_SV(""));
  ASSERT_NE(module, nullptr);
  loom_pass_verify_options_t options = {
      /*.registry=*/loom_test_pass_registry(),
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_verify_pipeline_op(module, nullptr, &options, scratch_arena()));
}

TEST_F(PassVerifyTest, VerifiesSatisfiedDescriptorRequirement) {
  IREE_ASSERT_OK(Verify(IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                                "  test.requires-target\n"
                                "}\n"),
                        PassTestTargetRecordEnvironment()));
}

TEST_F(PassVerifyTest, RejectsUnknownPassKey) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                             "  definitely-not-a-pass\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsUnavailablePassKey) {
  ExpectVerifyStatus(IREE_STATUS_FAILED_PRECONDITION,
                     IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                             "  test.unavailable\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsWrongAnchor) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                             "  test.noop\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsMalformedOptions) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                             "  test.options(count = 99)\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsUnknownOptions) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                             "  test.options(unknown = 1)\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsMissingRequiredOptions) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                             "  test.required\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsUnsatisfiedDescriptorRequirement) {
  ExpectVerifyStatus(IREE_STATUS_FAILED_PRECONDITION,
                     IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                             "  test.requires-target\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsNestedSameAnchorFor) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                             "  for func {\n"
                             "    for func {\n"
                             "      test.noop\n"
                             "    }\n"
                             "  }\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsEmptyWherePredicate) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                    "  where name() {\n"
                    "    test.noop\n"
                    "  }\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);
  loom_op_t* pipeline_op = loom_block_op(loom_region_entry_block(module->body),
                                         /*op_index=*/0);
  loom_block_t* pipeline_body =
      loom_region_entry_block(loom_pass_pipeline_body(pipeline_op));
  loom_op_t* where_op = loom_block_op(pipeline_body, /*op_index=*/0);
  ASSERT_TRUE(loom_pass_where_isa(where_op));
  loom_string_id_t empty_string_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, iree_string_view_empty(),
                                           &empty_string_id));
  loom_op_attrs(where_op)[loom_pass_where_predicate_ATTR_INDEX] =
      loom_attr_string(empty_string_id);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, VerifyModule(module));
}

TEST_F(PassVerifyTest, VerifiesBuiltInWherePredicates) {
  IREE_ASSERT_OK(Verify(
      IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
              "  for func {\n"
              "    where name(value = \"matmul\") {\n"
              "      test.noop\n"
              "    }\n"
              "    where attr(name = \"visibility\", value = \"public\") {\n"
              "      test.noop\n"
              "    }\n"
              "    where trait(name = \"isolated-from-above\") {\n"
              "      test.noop\n"
              "    }\n"
              "  }\n"
              "}\n")));
}

TEST_F(PassVerifyTest, RejectsMalformedWherePredicateAttrs) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                             "  where name(value = 1) {\n"
                             "    test.noop\n"
                             "  }\n"
                             "}\n"));
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                             "  where attr(name = 1) {\n"
                             "    test.noop\n"
                             "  }\n"
                             "}\n"));
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                             "  where trait(name = \"definitely-a-trait\") {\n"
                             "    test.noop\n"
                             "  }\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsUnknownWherePredicateWithoutProvider) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                             "  where target(value = \"gfx-test\") {\n"
                             "    test.noop\n"
                             "  }\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, VerifiesProviderWherePredicate) {
  PassTestPredicateCapture predicate_capture;
  loom_pass_predicate_provider_t provider =
      PassTestTargetPredicateProvider(&predicate_capture);
  IREE_ASSERT_OK(Verify(IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                                "  where target(value = \"gfx-test\") {\n"
                                "    test.noop\n"
                                "  }\n"
                                "}\n"),
                        loom_pass_environment_empty(), provider));
  EXPECT_EQ(predicate_capture.verify_count, 1);
}

TEST_F(PassVerifyTest, RejectsMissingRegionTerminator) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                    "  test.noop\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);
  loom_op_t* pipeline_op = loom_block_op(loom_region_entry_block(module->body),
                                         /*op_index=*/0);
  loom_block_t* pipeline_body =
      loom_region_entry_block(loom_pass_pipeline_body(pipeline_op));
  ASSERT_TRUE(loom_pass_yield_isa(pipeline_body->last_op));
  pipeline_body->last_op = pipeline_body->first_op;
  pipeline_body->first_op->next_op = nullptr;
  pipeline_body->op_count = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, VerifyModule(module));
}

TEST_F(PassVerifyTest, RejectsRepeatMissingCount) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                             "  repeat fixed {\n"
                             "    test.noop\n"
                             "  }\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsRepeatUntilConvergedWithCount) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                             "  repeat until_converged(count = 2) {\n"
                             "    test.noop\n"
                             "  }\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsUnresolvedCall) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                             "  call @missing\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsCallAnchorMismatch) {
  ExpectVerifyStatus(
      IREE_STATUS_INVALID_ARGUMENT,
      IREE_SV("pass.pipeline<func> @function_cleanup pipeline {\n"
              "  test.noop\n"
              "}\n"
              "\n"
              "pass.pipeline<module> @pipeline pipeline {\n"
              "  call @function_cleanup\n"
              "}\n"));
}

TEST_F(PassVerifyTest, RejectsCallCycle) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<module> @a pipeline {\n"
                             "  call @b\n"
                             "}\n"
                             "\n"
                             "pass.pipeline<module> @b pipeline {\n"
                             "  call @a\n"
                             "}\n"));
}

}  // namespace
}  // namespace loom
