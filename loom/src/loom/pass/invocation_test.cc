// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/invocation.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ops/pass/ops.h"
#include "loom/pass/test/harness.h"
#include "loom/pass/test/registry.h"

namespace loom {
namespace {

class PassInvocationTest : public PassTestHarness {
 protected:
  const loom_op_t* FirstRunOp(loom_module_t* module) {
    const loom_op_t* run = PipelineBodyOp(module, 0, 0);
    EXPECT_TRUE(loom_pass_run_isa(run));
    return run;
  }

  loom_pass_invocation_t Resolve(loom_module_t* module, const loom_op_t* op,
                                 loom_pass_kind_t required_kind) {
    loom_pass_invocation_t invocation = {};
    IREE_EXPECT_OK(loom_pass_invocation_resolve_run_op(
        module, loom_test_pass_registry(), op, required_kind, scratch_arena(),
        &invocation));
    return invocation;
  }
};

TEST_F(PassInvocationTest, ResolvesPassRunAndDecodesOptions) {
  loom_module_t* module = Parse(
      IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
              "  test.options(count = 7, mode = beta, string = \"payload\")\n"
              "}\n"));
  const loom_op_t* op = FirstRunOp(module);

  loom_pass_invocation_t invocation = Resolve(module, op, LOOM_PASS_FUNCTION);

  EXPECT_EQ(invocation.source_op, op);
  EXPECT_TRUE(iree_string_view_equal(invocation.key, IREE_SV("test.options")));
  ASSERT_NE(invocation.descriptor, nullptr);
  ASSERT_NE(invocation.info, nullptr);
  EXPECT_EQ(invocation.info->kind, LOOM_PASS_FUNCTION);

  const loom_pass_decoded_options_t* options = &invocation.decoded_options;
  EXPECT_EQ(options->descriptor, invocation.descriptor);
  ASSERT_EQ(options->option_count, 3u);
  ASSERT_NE(options->options, nullptr);
  EXPECT_TRUE(options->options[0].present);
  EXPECT_EQ(options->options[0].uint32_value, 7u);
  EXPECT_TRUE(options->options[1].present);
  EXPECT_EQ(options->options[1].enum_value_index, 1u);
  EXPECT_TRUE(options->options[2].present);
  EXPECT_TRUE(iree_string_view_equal(options->options[2].string_value,
                                     IREE_SV("payload")));
}

TEST_F(PassInvocationTest, ResolvesAbsentOptionalOptions) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                    "  test.options\n"
                    "}\n"));
  const loom_op_t* op = FirstRunOp(module);

  loom_pass_invocation_t invocation = Resolve(module, op, LOOM_PASS_FUNCTION);

  ASSERT_EQ(invocation.decoded_options.option_count, 3u);
  ASSERT_NE(invocation.decoded_options.options, nullptr);
  EXPECT_FALSE(invocation.decoded_options.options[0].present);
  EXPECT_FALSE(invocation.decoded_options.options[1].present);
  EXPECT_FALSE(invocation.decoded_options.options[2].present);
}

TEST_F(PassInvocationTest, RejectsUnknownPassKey) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                    "  definitely-not-a-pass\n"
                    "}\n"));
  const loom_op_t* op = FirstRunOp(module);

  loom_pass_invocation_t invocation = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_invocation_resolve_run_op(
                            module, loom_test_pass_registry(), op,
                            LOOM_PASS_FUNCTION, scratch_arena(), &invocation));
}

TEST_F(PassInvocationTest, RejectsUnavailablePassKey) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                    "  test.unavailable\n"
                    "}\n"));
  const loom_op_t* op = FirstRunOp(module);

  loom_pass_invocation_t invocation = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_pass_invocation_resolve_run_op(
                            module, loom_test_pass_registry(), op,
                            LOOM_PASS_FUNCTION, scratch_arena(), &invocation));
}

TEST_F(PassInvocationTest, RejectsWrongAnchor) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                    "  test.module-noop\n"
                    "}\n"));
  const loom_op_t* op = FirstRunOp(module);

  loom_pass_invocation_t invocation = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_invocation_resolve_run_op(
                            module, loom_test_pass_registry(), op,
                            LOOM_PASS_FUNCTION, scratch_arena(), &invocation));
}

TEST_F(PassInvocationTest, RejectsMalformedOptions) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                    "  test.options(count = 99)\n"
                    "}\n"));
  const loom_op_t* op = FirstRunOp(module);

  loom_pass_invocation_t invocation = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_invocation_resolve_run_op(
                            module, loom_test_pass_registry(), op,
                            LOOM_PASS_FUNCTION, scratch_arena(), &invocation));
}

}  // namespace
}  // namespace loom
