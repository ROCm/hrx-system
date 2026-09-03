// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/testing/test_diagnostic.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TestDiagnosticTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(TestDiagnosticTest, MatchesStructuredConstraints) {
  const loom_error_def_t* error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE,
                                                        /*code=*/3);
  ASSERT_NE(error, nullptr);
  iree_string_view_t param_values[] = {IREE_SV("operand"), IREE_SV("i32"),
                                       IREE_SV("floating-point scalar")};
  loom_test_diagnostic_t diagnostic = {
      .severity = LOOM_DIAGNOSTIC_ERROR,
      .domain = LOOM_ERROR_DOMAIN_TYPE,
      .code = 3,
      .error = error,
      .origin_line = 7,
      .message = IREE_SV("operand requires a floating-point scalar"),
      .param_values = param_values,
      .param_value_count = IREE_ARRAYSIZE(param_values),
  };
  loom_test_annotation_t annotation = {
      .message_substring_count = 1,
      .param_match_count = 1,
      .severity = LOOM_DIAGNOSTIC_ERROR,
      .domain = LOOM_ERROR_DOMAIN_TYPE,
      .code = 3,
      .target_line = 7,
      .message_substrings = {IREE_SV("floating-point")},
      .param_matches = {{IREE_SV("actual_type"), IREE_SV("i32")}},
  };

  EXPECT_TRUE(
      loom_test_diagnostic_matches_annotation(&diagnostic, &annotation));
  annotation.target_line = 8;
  EXPECT_FALSE(
      loom_test_diagnostic_matches_annotation(&diagnostic, &annotation));
}

TEST_F(TestDiagnosticTest, FindsMaximumOneToOneMatching) {
  loom_test_diagnostic_t diagnostics[] = {
      {
          .severity = LOOM_DIAGNOSTIC_ERROR,
          .domain = LOOM_ERROR_DOMAIN_PARSE,
          .code = 1,
          .origin_line = 1,
          .message = IREE_SV("specific failure"),
      },
      {
          .severity = LOOM_DIAGNOSTIC_ERROR,
          .domain = LOOM_ERROR_DOMAIN_PARSE,
          .code = 1,
          .origin_line = 1,
          .message = IREE_SV("other failure"),
      },
  };
  loom_test_annotation_t annotations[] = {
      {
          .severity = LOOM_DIAGNOSTIC_ERROR,
          .domain = LOOM_ERROR_DOMAIN_PARSE,
          .code = 1,
          .target_line = 1,
      },
      {
          .message_substring_count = 1,
          .severity = LOOM_DIAGNOSTIC_ERROR,
          .domain = LOOM_ERROR_DOMAIN_PARSE,
          .code = 1,
          .target_line = 1,
          .message_substrings = {IREE_SV("specific")},
      },
  };

  iree_host_size_t* annotation_to_diagnostic = nullptr;
  IREE_ASSERT_OK(loom_test_diagnostics_match_annotations(
      diagnostics, IREE_ARRAYSIZE(diagnostics), annotations,
      IREE_ARRAYSIZE(annotations), &arena_, &annotation_to_diagnostic));
  EXPECT_TRUE(diagnostics[0].matched);
  EXPECT_TRUE(diagnostics[1].matched);
  EXPECT_EQ(annotation_to_diagnostic[0], 1u);
  EXPECT_EQ(annotation_to_diagnostic[1], 0u);
}

}  // namespace
