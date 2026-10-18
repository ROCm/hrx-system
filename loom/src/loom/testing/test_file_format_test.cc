// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/testing/test_file_format.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/testing/target_environment.h"
#include "loomc/iree.h"

namespace loom {
namespace {

using ::iree::testing::status::StatusIs;
using ::testing::HasSubstr;

class LoomTestFileFormatTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loomc_target_environment_t* environment = nullptr;
    IREE_ASSERT_OK(
        iree_status_from_loomc(loomc_target_environment_create_test_ir(
            loomc_allocator_system(), &environment)));
    const loomc_context_target_options_t targets = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        .structure_size = sizeof(targets),
        .target_environment = environment,
    };
    const loomc_context_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        .structure_size = sizeof(options),
        .next = &targets,
    };
    IREE_ASSERT_OK(iree_status_from_loomc(
        loomc_context_create(&options, loomc_allocator_system(), &context_)));
    loomc_target_environment_release(environment);
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_workspace_create(
        nullptr, loomc_allocator_system(), &workspace_)));
  }

  void TearDown() override {
    loomc_workspace_release(workspace_);
    loomc_context_release(context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  std::string Format(iree_string_view_t source) {
    iree_string_builder_t output;
    iree_string_builder_initialize(iree_allocator_system(), &output);
    IREE_EXPECT_OK(loom_test_file_format(source, IREE_SV("test.loom-test"),
                                         context_, workspace_, &block_pool_,
                                         iree_allocator_system(), &output));
    std::string result(iree_string_builder_buffer(&output),
                       iree_string_builder_size(&output));
    iree_string_builder_deinitialize(&output);
    return result;
  }

  iree_status_t TryFormat(iree_string_view_t source) {
    iree_string_builder_t output;
    iree_string_builder_initialize(iree_allocator_system(), &output);
    iree_status_t status = loom_test_file_format(
        source, IREE_SV("test.loom-test"), context_, workspace_, &block_pool_,
        iree_allocator_system(), &output);
    iree_string_builder_deinitialize(&output);
    return status;
  }

  loomc_context_t* context_ = nullptr;
  loomc_workspace_t* workspace_ = nullptr;
  iree_arena_block_pool_t block_pool_;
};

TEST_F(LoomTestFileFormatTest, FormatsInputAndPreservesExpectedOutput) {
  const iree_string_view_t source = IREE_SV(
      "// RUN: pass cse\n"
      "\n"
      "func.def @identity(%value:index)->(index){\n"
      "func.return %value:index\n"
      "}\n"
      "\n"
      "// This prose belongs to the test case, not the module.\n"
      "\n"
      "// ----\n"
      "purposefully noncanonical expected output\n");
  EXPECT_EQ(Format(source),
            "// RUN: pass cse\n"
            "\n"
            "func.def @identity(%value: index) -> (index) {\n"
            "  func.return %value : index\n"
            "}\n"
            "\n"
            "// This prose belongs to the test case, not the module.\n"
            "\n"
            "// ----\n"
            "purposefully noncanonical expected output\n");
}

TEST_F(LoomTestFileFormatTest, FormatsInputAndPreservesChecks) {
  const iree_string_view_t source = IREE_SV(
      "// RUN: with-checks pass cse\n\n"
      "func.def @identity(%value:index)->(index){\n"
      "func.return %value:index\n}\n\n// ----\n"
      "// The result matters; operation numbering does not.\n"
      "CHECK: func.return * : index\nCHECK-NOT: *index.add*\n");
  const std::string formatted = Format(source);
  EXPECT_THAT(formatted,
              HasSubstr("func.def @identity(%value: index) -> (index) {\n"));
  EXPECT_THAT(
      formatted,
      HasSubstr("// ----\n"
                "// The result matters; operation numbering does not.\n"
                "CHECK: func.return * : index\nCHECK-NOT: *index.add*\n"));
  EXPECT_EQ(Format(iree_make_string_view(formatted.data(), formatted.size())),
            formatted);
}

TEST_F(LoomTestFileFormatTest, PreservesSourceConversionInputSpelling) {
  const iree_string_view_t source = IREE_SV(
      "// RUN: roundtrip\n"
      "\n"
      "func.def @identity(%value:index)->(index){\n"
      "func.return %value:index\n"
      "}\n"
      "// ----\n"
      "func.def @identity(%value: index) -> (index) {\n"
      "  func.return %value : index\n"
      "}\n");
  EXPECT_EQ(Format(source), std::string(source.data, source.size));
}

TEST_F(LoomTestFileFormatTest, PreservesUnparseableSourceConversionInput) {
  const iree_string_view_t source = IREE_SV(
      "// RUN: roundtrip\n"
      "\n"
      "purposefully invalid source spelling\n"
      "// ----\n"
      "expected conversion output\n");
  EXPECT_EQ(Format(source), std::string(source.data, source.size));
}

TEST_F(LoomTestFileFormatTest, CanonicalizesImplicitRoundtripToPreferredAsm) {
  const iree_string_view_t source = IREE_SV(
      "// RUN: roundtrip\n"
      "\n"
      "test.target<low_core> @test_target\n"
      "\n"
      "low.func.def target<test.low.core>(@test_target) @identity("
      "%value: reg<test.i32>) -> (reg<test.i32>) {\n"
      "  low.return %value : reg<test.i32>\n"
      "}\n");
  EXPECT_EQ(Format(source),
            "// RUN: roundtrip\n"
            "\n"
            "test.target<low_core> @test_target\n"
            "\n"
            "low.func.def target<test.low.core>(@test_target) @identity("
            "%value: reg<test.i32>) -> (reg<test.i32>) asm {\n"
            "  return %value\n"
            "}\n");
}

TEST_F(LoomTestFileFormatTest, PreservesLocationsRequestedByCase) {
  const iree_string_view_t source = IREE_SV(
      "// RUN: with-locations pass cse\n"
      "\n"
      "func.def @identity(%value:index)->(index) {\n"
      "  func.return %value:index loc(\"authored.loom\":7:3)\n"
      "}\n");
  EXPECT_THAT(Format(source),
              ::testing::HasSubstr("loc(\"authored.loom\":7:3)"));
}

TEST_F(LoomTestFileFormatTest, RoundtripInputDoesNotRequireVerification) {
  const iree_string_view_t source = IREE_SV(
      "// RUN: roundtrip\n"
      "\n"
      "%value = test.constant 1 : i32\n");
  EXPECT_EQ(Format(source), std::string(source.data, source.size));
}

TEST_F(LoomTestFileFormatTest, PassInputDoesNotRequireVerification) {
  const iree_string_view_t source = IREE_SV(
      "// RUN: pass cse\n"
      "\n"
      "func.def @invalid(%value:f32) where [eq(%value,4)]{\n"
      "func.return\n"
      "}\n");
  EXPECT_EQ(Format(source),
            "// RUN: pass cse\n"
            "\n"
            "func.def @invalid(%value: f32) where [eq(%value, 4)] {\n"
            "  func.return\n"
            "}\n");
}

TEST_F(LoomTestFileFormatTest, PreservesExactlyAnnotatedParserFailure) {
  const iree_string_view_t source = IREE_SV(
      "// RUN: verify\n"
      "\n"
      "func.def @unknown_op() {\n"
      "  // ERROR@+1: PARSE/006 \"bogus.nonexistent\"\n"
      "  bogus.nonexistent\n"
      "}\n");
  EXPECT_EQ(Format(source), std::string(source.data, source.size));
}

TEST_F(LoomTestFileFormatTest, PreservesEofDiagnosticAfterTrailingTrivia) {
  const iree_string_view_t source = IREE_SV(
      "// RUN: verify\n"
      "\n"
      "// ERROR@+5: PARSE/003 \"symbol\"\n"
      "func.decl\n"
      "\n"
      "// This comment and blank line precede the diagnostic's EOF.\n"
      "\n");
  EXPECT_EQ(Format(source), std::string(source.data, source.size));
}

TEST_F(LoomTestFileFormatTest, PreservesExactlyAnnotatedVerifierFailure) {
  const iree_string_view_t source = IREE_SV(
      "// RUN: verify\n"
      "\n"
      "// ERROR@+1: TYPE/003 {operand_name=\"predicates[0].arg[0]\", "
      "actual_type=\"f32\", expected_constraint=\"integer, index, or offset "
      "scalar\"}\n"
      "func.def @invalid(%value: f32) where [eq(%value, 4)] {\n"
      "  func.return\n"
      "}\n");
  EXPECT_EQ(Format(source), std::string(source.data, source.size));
}

TEST_F(LoomTestFileFormatTest, PreservesTargetTypeDiagnosticParameters) {
  const iree_string_view_t source = IREE_SV(
      "// RUN: verify\n\n"
      "test.target<low_core> @target\n\n"
      "low.func.def target<test.low.core>(@target) @copy(%pair: reg<test.i32 "
      "x2>) {\n"
      "  // ERROR@+1: TYPE/001 {field_a=\"source\", type_a=\"reg<test.i32 "
      "x2>\", "
      "field_b=\"result\", type_b=\"reg<test.i32>\"}\n"
      "  %copy = low.copy %pair : reg<test.i32 x2> -> reg<test.i32>\n"
      "  low.return\n}\n");
  EXPECT_EQ(Format(source), std::string(source.data, source.size));
}

TEST_F(LoomTestFileFormatTest, RejectsUnmatchedParserFailure) {
  const iree_string_view_t source = IREE_SV(
      "// RUN: verify\n"
      "\n"
      "func.def @unknown_op() {\n"
      "  // ERROR@+1: PARSE/001\n"
      "  bogus.nonexistent\n"
      "}\n");
  iree::Status status(TryFormat(source));
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.ToString(), HasSubstr("while formatting case 1"));
}

TEST_F(LoomTestFileFormatTest, XfailAloneDoesNotExemptInvalidInput) {
  const iree_string_view_t source = IREE_SV(
      "// RUN: verify\n"
      "// XFAIL: parser failure without an annotation\n"
      "\n"
      "func.def @unknown_op() {\n"
      "  bogus.nonexistent\n"
      "}\n");
  iree::Status status(TryFormat(source));
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace loom
