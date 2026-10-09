// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/testing/test_diagnostic.h"

#include <string>

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
  loom_test_diagnostic_param_t params[] = {
      {/*.name=*/IREE_SV("operand_name"), /*.value=*/IREE_SV("operand")},
      {/*.name=*/IREE_SV("actual_type"), /*.value=*/IREE_SV("i32")},
      {/*.name=*/IREE_SV("expected_type"),
       /*.value=*/IREE_SV("floating-point scalar")},
  };
  loom_test_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.domain = LOOM_ERROR_DOMAIN_TYPE;
  diagnostic.code = 3;
  diagnostic.origin.line = 7;
  diagnostic.origin.filename = IREE_SV("case.loom-test");
  diagnostic.message = IREE_SV("operand requires a floating-point scalar");
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);

  loom_test_annotation_t annotation = {};
  annotation.message_substring_count = 1;
  annotation.param_match_count = 1;
  annotation.severity = LOOM_DIAGNOSTIC_ERROR;
  annotation.domain = LOOM_ERROR_DOMAIN_TYPE;
  annotation.code = 3;
  annotation.target_line = 7;
  annotation.message_substrings[0] = IREE_SV("floating-point");
  annotation.param_matches[0].name = IREE_SV("actual_type");
  annotation.param_matches[0].value = IREE_SV("i32");

  EXPECT_TRUE(loom_test_diagnostic_matches_annotation(
      &diagnostic, &annotation, IREE_SV("case.loom-test")));
  annotation.target_line = 8;
  EXPECT_FALSE(loom_test_diagnostic_matches_annotation(
      &diagnostic, &annotation, IREE_SV("case.loom-test")));
}

TEST_F(TestDiagnosticTest, FindsMaximumOneToOneMatching) {
  loom_test_diagnostic_t diagnostics[2] = {};
  diagnostics[0].severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostics[0].domain = LOOM_ERROR_DOMAIN_PARSE;
  diagnostics[0].code = 1;
  diagnostics[0].origin.line = 1;
  diagnostics[0].origin.filename = IREE_SV("case.loom-test");
  diagnostics[0].message = IREE_SV("specific failure");
  diagnostics[1].severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostics[1].domain = LOOM_ERROR_DOMAIN_PARSE;
  diagnostics[1].code = 1;
  diagnostics[1].origin.line = 1;
  diagnostics[1].origin.filename = IREE_SV("case.loom-test");
  diagnostics[1].message = IREE_SV("other failure");

  loom_test_annotation_t annotations[2] = {};
  annotations[0].severity = LOOM_DIAGNOSTIC_ERROR;
  annotations[0].domain = LOOM_ERROR_DOMAIN_PARSE;
  annotations[0].code = 1;
  annotations[0].target_line = 1;
  annotations[1].message_substring_count = 1;
  annotations[1].severity = LOOM_DIAGNOSTIC_ERROR;
  annotations[1].domain = LOOM_ERROR_DOMAIN_PARSE;
  annotations[1].code = 1;
  annotations[1].target_line = 1;
  annotations[1].message_substrings[0] = IREE_SV("specific");

  iree_host_size_t* annotation_to_diagnostic = nullptr;
  IREE_ASSERT_OK(loom_test_diagnostics_match_annotations(
      diagnostics, IREE_ARRAYSIZE(diagnostics), annotations,
      IREE_ARRAYSIZE(annotations), IREE_SV("case.loom-test"), &arena_,
      &annotation_to_diagnostic));
  EXPECT_TRUE(diagnostics[0].matched);
  EXPECT_TRUE(diagnostics[1].matched);
  EXPECT_EQ(annotation_to_diagnostic[0], 1u);
  EXPECT_EQ(annotation_to_diagnostic[1], 0u);
}

TEST_F(TestDiagnosticTest, MaterializationPreservesSourceIdentity) {
  std::string filename = "included.h";
  loom_diagnostic_param_t parameter =
      loom_param_string(IREE_SV("invalid input"));
  loom_diagnostic_t emitted = {};
  emitted.severity = LOOM_DIAGNOSTIC_ERROR;
  emitted.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 36);
  ASSERT_NE(emitted.error, nullptr);
  emitted.params = &parameter;
  emitted.param_count = 1;
  emitted.origin.filename =
      iree_make_string_view(filename.data(), filename.size());
  emitted.origin.start_line = 1;
  loom_test_diagnostic_format_options_t options = {};
  loom_test_diagnostic_t diagnostic = {};
  IREE_ASSERT_OK(loom_test_diagnostic_materialize(
      &emitted, &options, &arena_, iree_allocator_system(), &diagnostic));
  filename.assign(filename.size(), '?');

  loom_test_annotation_t annotation = {.severity = LOOM_DIAGNOSTIC_ERROR,
                                       .domain = LOOM_ERROR_DOMAIN_PARSE,
                                       .code = 36,
                                       .target_line = 1};
  EXPECT_FALSE(loom_test_diagnostic_matches_annotation(
      &diagnostic, &annotation, IREE_SV("case.loom-test")));
  EXPECT_TRUE(loom_test_diagnostic_matches_annotation(&diagnostic, &annotation,
                                                      IREE_SV("included.h")));
}

}  // namespace
