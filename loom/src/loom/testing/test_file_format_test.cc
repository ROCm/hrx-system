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
#include "loom/codegen/low/text_asm.h"
#include "loom/target/test/descriptors.h"
#include "loom/testing/context.h"

namespace loom {
namespace {

using ::iree::testing::status::StatusIs;
using ::testing::HasSubstr;

static const loom_low_descriptor_set_provider_t kLowDescriptorSetProviders[] = {
    loom_test_low_core_descriptor_set,
};

class LoomTestFileFormatTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_testing_context_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    low_descriptor_registry_.descriptor_set_providers =
        kLowDescriptorSetProviders;
    low_descriptor_registry_.descriptor_set_provider_count =
        IREE_ARRAYSIZE(kLowDescriptorSetProviders);
    loom_low_descriptor_text_asm_environment_initialize(
        &low_descriptor_registry_, &low_asm_environment_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  std::string Format(iree_string_view_t source) {
    iree_string_builder_t output;
    iree_string_builder_initialize(iree_allocator_system(), &output);
    IREE_EXPECT_OK(loom_test_file_format(
        source, IREE_SV("test.loom-test"), &context_, &block_pool_,
        low_asm_environment_, iree_allocator_system(), &output));
    std::string result(iree_string_builder_buffer(&output),
                       iree_string_builder_size(&output));
    iree_string_builder_deinitialize(&output);
    return result;
  }

  iree_status_t TryFormat(iree_string_view_t source) {
    iree_string_builder_t output;
    iree_string_builder_initialize(iree_allocator_system(), &output);
    iree_status_t status = loom_test_file_format(
        source, IREE_SV("test.loom-test"), &context_, &block_pool_,
        low_asm_environment_, iree_allocator_system(), &output);
    iree_string_builder_deinitialize(&output);
    return status;
  }

  loom_context_t context_;
  iree_arena_block_pool_t block_pool_;
  loom_low_descriptor_registry_t low_descriptor_registry_ = {};
  loom_text_low_asm_environment_t low_asm_environment_ = {};
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
