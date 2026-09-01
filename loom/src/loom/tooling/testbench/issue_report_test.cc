// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/issue_report.h"

#include "iree/base/internal/arena.h"
#include "iree/base/internal/json.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"
#include "loom/ops/index/ops.h"

namespace loom {
namespace {

static iree_string_view_t LookupObject(iree_string_view_t object,
                                       iree_string_view_t key) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_lookup_object_value(object, key, &value));
  return value;
}

class TestbenchIssueReportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &plan_arena_);

    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CHECK, loom_check_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    iree_arena_deinitialize(&plan_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t *
                                                              out_count);

  void RegisterDialect(loom_dialect_id_t dialect_id, DialectVtablesFn fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  loom_module_t* ParseModule(const char* source) {
    loom_text_parse_options_t options = {};
    options.max_errors = 20;
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source),
                                   IREE_SV("issue_report_test.loom"), &context_,
                                   &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t plan_arena_;
  loom_context_t context_;
};

TEST_F(TestbenchIssueReportTest, WritesStructuredUnsupportedOperationEvidence) {
  loom_module_t* module = ParseModule(R"(
check.case @unsupported_constant_case {
  %value = index.constant 1 : index
  check.return
}
)");
  ASSERT_NE(module, nullptr);

  loom_testbench_module_plan_t plan = {};
  IREE_ASSERT_OK(
      loom_testbench_plan_module(module, nullptr, &plan_arena_, &plan));
  ASSERT_EQ(plan.issue_count, 1u);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(loom_testbench_issue_array_write_json(
      &plan, plan.issues, plan.issue_count, &stream));

  iree_string_view_t cursor = iree_string_builder_view(&builder);
  iree_string_view_t issues = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_consume_value(&cursor, &issues));
  IREE_ASSERT_OK(iree_json_consume_insignificant(&cursor));
  EXPECT_TRUE(iree_string_view_is_empty(cursor));

  iree_host_size_t issue_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(issues, &issue_count));
  ASSERT_EQ(issue_count, 1u);
  iree_string_view_t issue = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_array_get(issues, 0, &issue));
  EXPECT_TRUE(iree_string_view_equal(LookupObject(issue, IREE_SV("kind")),
                                     IREE_SV("unsupported_case_body_op")));
  EXPECT_TRUE(iree_string_view_equal(LookupObject(issue, IREE_SV("case")),
                                     IREE_SV("unsupported_constant_case")));
  EXPECT_TRUE(iree_string_view_equal(LookupObject(issue, IREE_SV("op")),
                                     IREE_SV("index.constant")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(issue, IREE_SV("fix_hint")),
      IREE_SV("use check.literal for scalar literals inside check.case")));

  const iree_string_view_t source_location =
      LookupObject(issue, IREE_SV("source_location"));
  EXPECT_TRUE(
      iree_string_view_equal(LookupObject(source_location, IREE_SV("filename")),
                             IREE_SV("issue_report_test.loom")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(source_location, IREE_SV("start_line")), IREE_SV("3")));

  iree_string_builder_deinitialize(&builder);
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
