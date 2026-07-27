// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/tooling.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/pass/test/harness.h"

namespace loom {
namespace {

class PassToolingTest : public PassTestHarness {};

TEST_F(PassToolingTest, RunsNamedModulePipeline) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @cleanup pipeline {\n"
                    "  test.module-noop\n"
                    "  for func {\n"
                    "    test.noop\n"
                    "  }\n"
                    "}\n"
                    "test.func @first() {\n"
                    "  test.yield\n"
                    "}\n"
                    "test.func @second() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  loom_test_pass_trace_t trace = {};
  IREE_ASSERT_OK(RunPipelineSymbol(module, IREE_SV("@cleanup"), &trace));

  EXPECT_EQ(trace.module_noop_invocation_count, 1);
  EXPECT_EQ(trace.noop_invocation_count, 2);
}

TEST_F(PassToolingTest, RunsNamedFunctionPipelineAcrossFunctions) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<func> @function_cleanup pipeline {\n"
                    "  test.noop\n"
                    "}\n"
                    "test.func @first() {\n"
                    "  test.yield\n"
                    "}\n"
                    "test.func @second() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  loom_test_pass_trace_t trace = {};
  IREE_ASSERT_OK(
      RunPipelineSymbol(module, IREE_SV("function_cleanup"), &trace));

  EXPECT_EQ(trace.noop_invocation_count, 2);
}

TEST_F(PassToolingTest, RunsFlatPipelineWithDescriptorOptions) {
  loom_module_t* module =
      Parse(IREE_SV("test.func @first() {\n"
                    "  test.yield\n"
                    "}\n"
                    "test.func @second() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  loom_test_pass_trace_t trace = {};
  IREE_ASSERT_OK(RunFlatPipeline(
      module,
      IREE_SV("test.module-noop,test.options{count=5,mode=beta,string=flat}"),
      &trace));

  EXPECT_EQ(trace.module_noop_invocation_count, 1);
  EXPECT_EQ(trace.options_invocation_count, 2);
  EXPECT_EQ(trace.options_decoded_create_count, 2);
  EXPECT_EQ(trace.decoded_options_count_value, 5u);
  EXPECT_EQ(trace.decoded_options_mode_index, 1u);
  EXPECT_TRUE(iree_string_view_equal(trace.decoded_options_string_value,
                                     IREE_SV("flat")));
}

TEST_F(PassToolingTest, GroupsAdjacentFlatFunctionPassesPerFunction) {
  loom_module_t* module =
      Parse(IREE_SV("test.func @first() {\n"
                    "  test.yield\n"
                    "}\n"
                    "test.func @second() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  loom_test_pass_trace_t trace = {};
  IREE_ASSERT_OK(
      RunFlatPipeline(module, IREE_SV("test.noop,test.mark-changed"), &trace));

  ASSERT_EQ(trace.event_count, 4u);
  ExpectTraceEvent(trace, 0, IREE_SV("test.noop"), IREE_SV("first"));
  ExpectTraceEvent(trace, 1, IREE_SV("test.mark-changed"), IREE_SV("first"));
  ExpectTraceEvent(trace, 2, IREE_SV("test.noop"), IREE_SV("second"));
  ExpectTraceEvent(trace, 3, IREE_SV("test.mark-changed"), IREE_SV("second"));
}

TEST_F(PassToolingTest, FlatModulePassesSeparateFunctionPassGroups) {
  loom_module_t* module =
      Parse(IREE_SV("test.func @first() {\n"
                    "  test.yield\n"
                    "}\n"
                    "test.func @second() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  loom_test_pass_trace_t trace = {};
  IREE_ASSERT_OK(RunFlatPipeline(
      module, IREE_SV("test.noop,test.module-noop,test.mark-changed"), &trace));

  ASSERT_EQ(trace.event_count, 5u);
  ExpectTraceEvent(trace, 0, IREE_SV("test.noop"), IREE_SV("first"));
  ExpectTraceEvent(trace, 1, IREE_SV("test.noop"), IREE_SV("second"));
  ExpectTraceEvent(trace, 2, IREE_SV("test.module-noop"), IREE_SV("<module>"));
  ExpectTraceEvent(trace, 3, IREE_SV("test.mark-changed"), IREE_SV("first"));
  ExpectTraceEvent(trace, 4, IREE_SV("test.mark-changed"), IREE_SV("second"));
}

TEST_F(PassToolingTest, RejectsUnknownFlatPipelinePass) {
  loom_module_t* module = Parse(IREE_SV(""));
  ASSERT_NE(module, nullptr);

  loom_test_pass_trace_t trace = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      RunFlatPipeline(module, IREE_SV("missing.pass"), &trace));
}

}  // namespace
}  // namespace loom
