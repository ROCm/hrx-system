// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/check.h"

#include <cstring>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class CheckParseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    allocator_ = iree_allocator_system();
    iree_arena_block_pool_initialize(32 * 1024, allocator_, &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t Parse(const char* source) {
    // Reset the arena between parses so each test starts fresh.
    iree_arena_reset(&arena_);
    file_ = {};
    source_ = iree_make_cstring_view(source);
    return loom_check_parse(source_, &arena_, &file_);
  }

  iree_host_size_t OffsetOf(const char* fragment) {
    const char* found = strstr(source_.data, fragment);
    EXPECT_NE(found, nullptr);
    return found ? (iree_host_size_t)(found - source_.data) : 0;
  }

  void ExpectRange(loom_check_source_range_t source_range,
                   iree_host_size_t start_byte, iree_host_size_t end_byte) {
    EXPECT_EQ(source_range.start_byte, start_byte);
    EXPECT_EQ(source_range.end_byte, end_byte);
  }

  void ExpectRangeForFragment(loom_check_source_range_t source_range,
                              const char* fragment) {
    iree_host_size_t start_byte = OffsetOf(fragment);
    ExpectRange(source_range, start_byte, start_byte + strlen(fragment));
  }

  void ExpectRequirementName(const loom_check_case_t& test_case,
                             iree_host_size_t requirement_index,
                             const char* expected_name) {
    ASSERT_LT(requirement_index, test_case.requirement_count);
    EXPECT_TRUE(
        iree_string_view_equal(test_case.requirements[requirement_index],
                               iree_make_cstring_view(expected_name)));
  }

  iree_allocator_t allocator_;
  iree_arena_block_pool_t block_pool_ = {};
  iree_arena_allocator_t arena_ = {};
  iree_string_view_t source_;
  loom_check_file_t file_ = {};
};

// ===----------------------------------------------------------------------===
// Directive parsing
// ===----------------------------------------------------------------------===

TEST_F(CheckParseTest, NoDirectiveDefaultsToRoundtrip) {
  IREE_ASSERT_OK(Parse("func.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_ROUNDTRIP);
}

TEST_F(CheckParseTest, ExplicitRoundtrip) {
  IREE_ASSERT_OK(Parse("// RUN: roundtrip\nfunc.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_ROUNDTRIP);
}

TEST_F(CheckParseTest, VerifyMode) {
  IREE_ASSERT_OK(Parse("// RUN: verify\nfunc.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_VERIFY);
}

TEST_F(CheckParseTest, PassMode) {
  IREE_ASSERT_OK(Parse("// RUN: pass dce,cse\nfunc.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_PASS);
  EXPECT_TRUE(iree_string_view_equal(file_.cases[0].pipeline,
                                     iree_make_cstring_view("dce,cse")));
}

TEST_F(CheckParseTest, FormatMode) {
  IREE_ASSERT_OK(Parse("// RUN: format bytecode\nfunc.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_FORMAT);
  EXPECT_TRUE(iree_string_view_equal(file_.cases[0].format_target,
                                     iree_make_cstring_view("bytecode")));
}

TEST_F(CheckParseTest, EmitMode) {
  IREE_ASSERT_OK(Parse("// RUN: emit target-form @entry\nfunc.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_EMIT);
  EXPECT_TRUE(
      iree_string_view_equal(file_.cases[0].emit_target,
                             iree_make_cstring_view("target-form @entry")));
}

TEST_F(CheckParseTest, EmitBitcodeMode) {
  IREE_ASSERT_OK(
      Parse("// RUN: emit target-bitcode target-key-a\nfunc.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_EMIT);
  EXPECT_TRUE(iree_string_view_equal(
      file_.cases[0].emit_target,
      iree_make_cstring_view("target-bitcode target-key-a")));
}

TEST_F(CheckParseTest, EmitObjectMode) {
  IREE_ASSERT_OK(
      Parse("// RUN: emit target-object target-key-a\nfunc.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_EMIT);
  EXPECT_TRUE(iree_string_view_equal(
      file_.cases[0].emit_target,
      iree_make_cstring_view("target-object target-key-a")));
}

TEST_F(CheckParseTest, XfailDirective) {
  IREE_ASSERT_OK(
      Parse("// XFAIL: parser does not support this yet\n"
            "func.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_TRUE(file_.cases[0].xfail);
  EXPECT_TRUE(iree_string_view_equal(
      file_.cases[0].xfail_reason,
      iree_make_cstring_view("parser does not support this yet")));
}

TEST_F(CheckParseTest, RunAndXfail) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// XFAIL: known verifier gap\n"
            "func.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_VERIFY);
  EXPECT_TRUE(file_.cases[0].xfail);
}

TEST_F(CheckParseTest, RequiresDirective) {
  IREE_ASSERT_OK(
      Parse("// RUN: emit target-bitcode\n"
            "// REQUIRES: tool-dis, tool-backend\n"
            "func.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_TRUE(file_.cases[0].has_requires_directive);
  ASSERT_EQ(file_.cases[0].requirement_count, 2);
  ExpectRequirementName(file_.cases[0], 0, "tool-dis");
  ExpectRequirementName(file_.cases[0], 1, "tool-backend");
}

TEST_F(CheckParseTest, RequiresDirectiveAllowsWhitespaceSeparatedNames) {
  IREE_ASSERT_OK(
      Parse("// REQUIRES: tool-dis tool tool-backend\n"
            "func.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  ASSERT_EQ(file_.cases[0].requirement_count, 3);
  ExpectRequirementName(file_.cases[0], 0, "tool-dis");
  ExpectRequirementName(file_.cases[0], 1, "tool");
  ExpectRequirementName(file_.cases[0], 2, "tool-backend");
}

TEST_F(CheckParseTest, TemplateDirectiveInPreamble) {
  IREE_ASSERT_OK(Parse(
      "// TEMPLATE: loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
      "// RUN: verify\n"
      "\n"
      "func.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_TRUE(file_.has_template_directive);
  EXPECT_TRUE(iree_string_view_equal(
      file_.template_path,
      iree_make_cstring_view(
          "loom/src/loom/test/corpus/vector/arithmetic.loom-test")));
  ExpectRangeForFragment(
      file_.template_directive_range,
      "// TEMPLATE: loom/src/loom/test/corpus/vector/arithmetic.loom-test");
  EXPECT_EQ(file_.default_mode, LOOM_CHECK_MODE_VERIFY);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_VERIFY);
  EXPECT_TRUE(loom_check_source_range_is_empty(file_.cases[0].separator_range));
}

TEST_F(CheckParseTest, TemplateDirectiveOnlyPreambleHasNoCases) {
  IREE_ASSERT_OK(Parse(
      "// TEMPLATE: loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
      "// RUN: verify\n"
      "\n"));
  EXPECT_TRUE(file_.has_template_directive);
  EXPECT_EQ(file_.default_mode, LOOM_CHECK_MODE_VERIFY);
  EXPECT_EQ(file_.case_count, 0);
}

TEST_F(CheckParseTest, TemplateDirectiveRequiresPurePreamble) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Parse("func.def @f() {}\n"
            "// TEMPLATE: "
            "loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
            "func.def @g() {}\n"));
}

TEST_F(CheckParseTest, TemplateDirectiveDoesNotRequireCaseSeparator) {
  IREE_ASSERT_OK(
      Parse("// TEMPLATE: "
            "loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
            "func.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_TRUE(file_.has_template_directive);
}

TEST_F(CheckParseTest, TemplateDirectiveRejectsSeparatorBeforeFirstCase) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Parse("// TEMPLATE: "
            "loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
            "\n"
            "// ====\n"
            "func.def @f() {}\n"));
}

TEST_F(CheckParseTest, TemplateDirectiveInCaseErrors) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Parse("func.def @first() {}\n"
            "\n"
            "// ====\n"
            "\n"
            "// TEMPLATE: "
            "loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
            "func.def @f() {}\n"));
}

TEST_F(CheckParseTest, TemplateDirectiveEmptyPathErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// TEMPLATE: \n"
                              "func.def @f() {}\n"));
}

TEST_F(CheckParseTest, TemplateDirectiveAbsolutePathErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// TEMPLATE: /tmp/corpus.loom-test\n"
                              "func.def @f() {}\n"));
}

TEST_F(CheckParseTest, TemplateDirectiveParentSegmentErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// TEMPLATE: runtime/src/../corpus.loom-test\n"
                              "func.def @f() {}\n"));
}

TEST_F(CheckParseTest, MultipleTemplateDirectivesError) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Parse("// TEMPLATE: "
            "loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
            "// TEMPLATE: loom/src/loom/test/corpus/vector/memory.loom-test\n"
            "\n"
            "func.def @f() {}\n"));
}

TEST_F(CheckParseTest, CaseDirectiveErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// CASE: f32_add\n"
                              "func.def @f32_add() {}\n"));
}

TEST_F(CheckParseTest, UnknownModeErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: foobar\nfunc.def @f() {}\n"));
}

TEST_F(CheckParseTest, MultipleRunDirectivesError) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: roundtrip\n"
                              "// RUN: verify\n"
                              "func.def @f() {}\n"));
}

TEST_F(CheckParseTest, DirectiveAfterIRIsStrayError) {
  // A "// RUN:" line after IR is a stray directive, not a comment.
  // This catches the silent misconfiguration where a comment or blank
  // line pushes a directive into the body.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("func.def @f() {}\n"
                              "// RUN: verify\n"));
}

TEST_F(CheckParseTest, PassWithoutPipelineErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse("// RUN: pass \n"));
}

TEST_F(CheckParseTest, FormatWithoutTargetErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: format \n"));
}

TEST_F(CheckParseTest, EmitWithoutTargetErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse("// RUN: emit \n"));
}

TEST_F(CheckParseTest, RunModeIsUnsupported) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: run --function=main\n"));
}

TEST_F(CheckParseTest, RunDirectiveWithoutSpaceAfterColonErrors) {
  // "// RUN:roundtrip" is malformed — the required space after the
  // colon is missing. This must error, not silently fall through as
  // a comment. Previously this was treated as a comment, which meant
  // a typo could silently change a test's mode to roundtrip.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN:roundtrip\nfunc.def @f() {}\n"));
}

TEST_F(CheckParseTest, XfailDirectiveWithoutSpaceAfterColonErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// XFAIL:reason\nfunc.def @f() {}\n"));
}

TEST_F(CheckParseTest, RequiresDirectiveWithoutSpaceAfterColonErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// REQUIRES:tool-dis\nfunc.def @f() {}\n"));
}

TEST_F(CheckParseTest, RequiresDirectiveTrailingCommaErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// REQUIRES: tool-dis,\nfunc.def @f() {}\n"));
}

TEST_F(CheckParseTest, RequiresDirectiveEmptyNameErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// REQUIRES: tool-dis,,tool\n"
                              "func.def @f() {}\n"));
}

TEST_F(CheckParseTest, RequiresDirectiveInvalidNameErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// REQUIRES: tool/dis\nfunc.def @f() {}\n"));
}

TEST_F(CheckParseTest, RequiresDirectiveDuplicateNameErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// REQUIRES: tool-dis, tool-dis\n"
                              "func.def @f() {}\n"));
}

TEST_F(CheckParseTest, RunDirectiveWithoutSpaceAfterSlashesErrors) {
  // "//RUN: verify" is a natural typo. Without detection, this would
  // be silently consumed as a header comment and the test would default
  // to roundtrip mode — a false pass waiting to happen.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("//RUN: verify\nfunc.def @f() {}\n"));
}

TEST_F(CheckParseTest, XfailDirectiveWithoutSpaceAfterSlashesErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("//XFAIL: reason\nfunc.def @f() {}\n"));
}

TEST_F(CheckParseTest, RequiresDirectiveWithoutSpaceAfterSlashesErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("//REQUIRES: tool-dis\nfunc.def @f() {}\n"));
}

// ===----------------------------------------------------------------------===
// Stray directive detection
// ===----------------------------------------------------------------------===

TEST_F(CheckParseTest, StrayRunAfterAnnotationErrors) {
  // A RUN directive after an annotation line has started the body.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// ERROR@+1: TYPE/001\n"
                              "// RUN: verify\n"
                              "  %x = test.bad : i32\n"));
}

TEST_F(CheckParseTest, StrayXfailAfterIRErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("func.def @f() {}\n"
                              "// XFAIL: late reason\n"));
}

TEST_F(CheckParseTest, StrayRequiresAfterIRErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("func.def @f() {}\n"
                              "// REQUIRES: tool-dis\n"));
}

TEST_F(CheckParseTest, StrayRunInSecondCaseErrors) {
  // Stray detection applies per-case.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: roundtrip\n"
                              "func.def @a() {}\n"
                              "// ====\n"
                              "func.def @b() {}\n"
                              "// RUN: verify\n"));
}

// ===----------------------------------------------------------------------===
// Header comment consumption
// ===----------------------------------------------------------------------===

TEST_F(CheckParseTest, CommentsBeforeDirectivesConsumed) {
  // Non-annotation comments before and between directives are
  // consumed as part of the header, not included in the input.
  IREE_ASSERT_OK(
      Parse("// Tests edge case for DCE.\n"
            "// RUN: pass dce\n"
            "func.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_PASS);
  EXPECT_TRUE(iree_string_view_find(file_.cases[0].input,
                                    iree_make_cstring_view("edge case"),
                                    0) == IREE_STRING_VIEW_NPOS);
}

TEST_F(CheckParseTest, CommentsBetweenDirectivesConsumed) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// This is a descriptive comment.\n"
            "// XFAIL: reason\n"
            "func.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_VERIFY);
  EXPECT_TRUE(file_.cases[0].xfail);
  EXPECT_TRUE(iree_string_view_find(file_.cases[0].input,
                                    iree_make_cstring_view("descriptive"),
                                    0) == IREE_STRING_VIEW_NPOS);
}

TEST_F(CheckParseTest, AnnotationLineStartsBody) {
  // A standalone annotation line starts the body even though it looks
  // like a comment. It must not be consumed by the header.
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR@+1: TYPE/001\n"
            "  %x = test.bad : i32\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  EXPECT_EQ(file_.cases[0].annotations[0].target_line, 2);
}

// ===----------------------------------------------------------------------===
// Section splitting
// ===----------------------------------------------------------------------===

TEST_F(CheckParseTest, NoSeparatorInputEqualsExpected) {
  IREE_ASSERT_OK(Parse("func.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_FALSE(file_.cases[0].has_expected_section);
  EXPECT_TRUE(
      iree_string_view_equal(file_.cases[0].input, file_.cases[0].expected));
}

TEST_F(CheckParseTest, ExpectedSeparatorSplitsContent) {
  IREE_ASSERT_OK(
      Parse("func.def @f() {\n"
            "  %x = test.constant 42 : i32\n"
            "}\n"
            "// ----\n"
            "func.def @f() {\n"
            "}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_TRUE(file_.cases[0].has_expected_section);
  EXPECT_TRUE(iree_string_view_find(file_.cases[0].input,
                                    iree_make_cstring_view("test.constant"),
                                    0) != IREE_STRING_VIEW_NPOS);
  EXPECT_TRUE(iree_string_view_find(file_.cases[0].expected,
                                    iree_make_cstring_view("test.constant"),
                                    0) == IREE_STRING_VIEW_NPOS);
}

TEST_F(CheckParseTest, MultipleCases) {
  IREE_ASSERT_OK(
      Parse("func.def @first() {}\n"
            "// ====\n"
            "func.def @second() {}\n"
            "// ====\n"
            "func.def @third() {}\n"));
  ASSERT_EQ(file_.case_count, 3);
}

TEST_F(CheckParseTest, CaseSeparator) {
  IREE_ASSERT_OK(
      Parse("func.def @a() {}\n"
            "// ====\n"
            "func.def @b() {}\n"));
  ASSERT_EQ(file_.case_count, 2);
}

TEST_F(CheckParseTest, DirectivesNotInInput) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "func.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_TRUE(iree_string_view_find(file_.cases[0].input,
                                    iree_make_cstring_view("// RUN:"),
                                    0) == IREE_STRING_VIEW_NPOS);
  EXPECT_TRUE(iree_string_view_find(file_.cases[0].input,
                                    iree_make_cstring_view("func.def"),
                                    0) != IREE_STRING_VIEW_NPOS);
}

TEST_F(CheckParseTest, EmptyExpectedSection) {
  IREE_ASSERT_OK(
      Parse("func.def @f() {\n"
            "  %x = test.constant 42 : i32\n"
            "}\n"
            "// ----\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_TRUE(file_.cases[0].has_expected_section);
  EXPECT_TRUE(iree_string_view_is_empty(file_.cases[0].expected));
}

TEST_F(CheckParseTest, BothSeparatorsInOneFile) {
  IREE_ASSERT_OK(
      Parse("// RUN: pass dce\n"
            "func.def @a() {\n"
            "  %dead = test.constant 1 : i32\n"
            "}\n"
            "// ----\n"
            "func.def @a() {\n"
            "}\n"
            "// ====\n"
            "\n"
            "// RUN: roundtrip\n"
            "func.def @b() {}\n"));
  ASSERT_EQ(file_.case_count, 2);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_PASS);
  EXPECT_TRUE(file_.cases[0].has_expected_section);
  EXPECT_EQ(file_.cases[1].mode, LOOM_CHECK_MODE_ROUNDTRIP);
  EXPECT_FALSE(file_.cases[1].has_expected_section);
}

// ===----------------------------------------------------------------------===
// Annotation parsing
// ===----------------------------------------------------------------------===

TEST_F(CheckParseTest, AnnotationWithDomainAndCode) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "func.def @f() {\n"
            "  // ERROR@+1: TYPE/001\n"
            "  %x = test.bad : i32\n"
            "}\n"));
  ASSERT_EQ(file_.case_count, 1);
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  const auto& ann = file_.cases[0].annotations[0];
  EXPECT_EQ(ann.severity, LOOM_DIAGNOSTIC_ERROR);
  EXPECT_EQ(ann.domain, LOOM_ERROR_DOMAIN_TYPE);
  EXPECT_EQ(ann.code, 1);
  EXPECT_EQ(ann.target_line, 3);  // Line 2 targets line 3 (+1).
}

TEST_F(CheckParseTest, AnnotationWithSubstring) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR@+1: TYPE/001 \"does not match\"\n"
            "  %x = test.bad : i32\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  const auto& ann = file_.cases[0].annotations[0];
  ASSERT_EQ(ann.message_substring_count, 1);
  EXPECT_TRUE(iree_string_view_equal(ann.message_substrings[0],
                                     iree_make_cstring_view("does not match")));
}

TEST_F(CheckParseTest, AnnotationWithMultipleSubstrings) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR@+1: STRUCTURE/013 \"operand 3\" \"result 0\"\n"
            "  %x = test.bad : i32\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  const auto& ann = file_.cases[0].annotations[0];
  EXPECT_EQ(ann.domain, LOOM_ERROR_DOMAIN_STRUCTURE);
  EXPECT_EQ(ann.code, 13);
  ASSERT_EQ(ann.message_substring_count, 2);
  EXPECT_TRUE(iree_string_view_equal(ann.message_substrings[0],
                                     iree_make_cstring_view("operand 3")));
  EXPECT_TRUE(iree_string_view_equal(ann.message_substrings[1],
                                     iree_make_cstring_view("result 0")));
}

TEST_F(CheckParseTest, AnnotationWithParamMatchers) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR@+1: DOMINANCE/011 {op_name=\"test.add.i32\", "
            "field_name=\"lhs\"}\n"
            "  %x = test.bad : i32\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  const auto& ann = file_.cases[0].annotations[0];
  EXPECT_EQ(ann.domain, LOOM_ERROR_DOMAIN_DOMINANCE);
  EXPECT_EQ(ann.code, 11);
  ASSERT_EQ(ann.param_match_count, 2);
  EXPECT_TRUE(iree_string_view_equal(ann.param_matches[0].name,
                                     iree_make_cstring_view("op_name")));
  EXPECT_TRUE(iree_string_view_equal(ann.param_matches[0].value,
                                     iree_make_cstring_view("test.add.i32")));
  EXPECT_TRUE(iree_string_view_equal(ann.param_matches[1].name,
                                     iree_make_cstring_view("field_name")));
  EXPECT_TRUE(iree_string_view_equal(ann.param_matches[1].value,
                                     iree_make_cstring_view("lhs")));
  EXPECT_EQ(ann.message_substring_count, 0);
}

TEST_F(CheckParseTest, AnnotationMultipleSubstringsNoDomain) {
  // Substring-only annotations also accept multiple quoted matchers.
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR@+1: \"first\" \"second\" \"third\"\n"
            "  %x = test.bad : i32\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  const auto& ann = file_.cases[0].annotations[0];
  EXPECT_EQ(ann.domain, LOOM_ERROR_DOMAIN_COUNT_);
  ASSERT_EQ(ann.message_substring_count, 3);
  EXPECT_TRUE(iree_string_view_equal(ann.message_substrings[0],
                                     iree_make_cstring_view("first")));
  EXPECT_TRUE(iree_string_view_equal(ann.message_substrings[1],
                                     iree_make_cstring_view("second")));
  EXPECT_TRUE(iree_string_view_equal(ann.message_substrings[2],
                                     iree_make_cstring_view("third")));
}

TEST_F(CheckParseTest, AnnotationTooManySubstringsRejected) {
  // Five substrings exceeds LOOM_CHECK_MAX_ANNOTATION_SUBSTRINGS (4).
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Parse("// RUN: verify\n"
            "// ERROR@+1: TYPE/001 \"a\" \"b\" \"c\" \"d\" \"e\"\n"
            "  %x = test.bad : i32\n"));
}

TEST_F(CheckParseTest, AnnotationSubstringOnly) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR@+1: \"something went wrong\"\n"
            "  %x = test.bad : i32\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  const auto& ann = file_.cases[0].annotations[0];
  EXPECT_EQ(ann.domain, LOOM_ERROR_DOMAIN_COUNT_);
  EXPECT_EQ(ann.code, 0);
  ASSERT_EQ(ann.message_substring_count, 1);
  EXPECT_TRUE(
      iree_string_view_equal(ann.message_substrings[0],
                             iree_make_cstring_view("something went wrong")));
}

TEST_F(CheckParseTest, AnnotationWarning) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// WARNING@+1: FOLD/003\n"
            "  %x = test.something : i32\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  EXPECT_EQ(file_.cases[0].annotations[0].severity, LOOM_DIAGNOSTIC_WARNING);
}

TEST_F(CheckParseTest, AnnotationRemark) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// REMARK@+1: \"optimized\"\n"
            "  %x = test.something : i32\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  EXPECT_EQ(file_.cases[0].annotations[0].severity, LOOM_DIAGNOSTIC_REMARK);
}

TEST_F(CheckParseTest, AnnotationPositiveOffset) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR@+1: TYPE/001\n"
            "  %x = test.bad : i32\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  // Annotation is on line 1, targets line 2 (+1).
  EXPECT_EQ(file_.cases[0].annotations[0].target_line, 2);
}

TEST_F(CheckParseTest, AnnotationNegativeOffset) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "  %x = test.bad : i32\n"
            "// ERROR@-1: TYPE/001\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  // Annotation is on line 2, targets line 1 (-1).
  EXPECT_EQ(file_.cases[0].annotations[0].target_line, 1);
}

TEST_F(CheckParseTest, MultipleAnnotations) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR@+1: TYPE/001\n"
            "  %x = test.bad : i32\n"
            "// WARNING@+1: TYPE/002\n"
            "  %y = test.bad : f32\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 2);
  EXPECT_EQ(file_.cases[0].annotations[0].severity, LOOM_DIAGNOSTIC_ERROR);
  EXPECT_EQ(file_.cases[0].annotations[1].severity, LOOM_DIAGNOSTIC_WARNING);
}

TEST_F(CheckParseTest, MultipleAnnotationsTargetingSameLine) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR@+2: TYPE/001\n"
            "// WARNING@+1: TYPE/002\n"
            "  %x = test.bad : i32\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 2);
  // Both should target line 3 (the IR line).
  EXPECT_EQ(file_.cases[0].annotations[0].target_line, 3);
  EXPECT_EQ(file_.cases[0].annotations[1].target_line, 3);
}

TEST_F(CheckParseTest, StandaloneAnnotation) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR@+1: PARSE/006\n"
            "  %x = test.unknown_op : i32\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  EXPECT_EQ(file_.cases[0].annotations[0].target_line, 2);
  EXPECT_EQ(file_.cases[0].annotations[0].domain, LOOM_ERROR_DOMAIN_PARSE);
  EXPECT_EQ(file_.cases[0].annotations[0].code, 6);
}

TEST_F(CheckParseTest, AnnotationDomainWithoutCode) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR@+1: TYPE\n"
            "  %x = test.bad : i32\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  EXPECT_EQ(file_.cases[0].annotations[0].domain, LOOM_ERROR_DOMAIN_TYPE);
  EXPECT_EQ(file_.cases[0].annotations[0].code, 0);
}

TEST_F(CheckParseTest, LowercaseAnnotationIsNotRecognized) {
  // Annotations use uppercase severity keywords. Lowercase "error:"
  // is an ordinary comment, not an annotation. This is the key
  // distinction that makes preamble comment consumption safe.
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// error: TYPE/001\n"
            "  %x = test.bad : i32\n"));
  EXPECT_EQ(file_.cases[0].annotation_count, 0);
}

TEST_F(CheckParseTest, LowercaseWarningIsNotRecognized) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// warning: FOLD/003\n"
            "  %x = test.bad : i32\n"));
  EXPECT_EQ(file_.cases[0].annotation_count, 0);
}

// ===----------------------------------------------------------------------===
// Comment stripping
// ===----------------------------------------------------------------------===

TEST_F(CheckParseTest, StripCommentsRemovesStandaloneComments) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator_, &builder);
  IREE_ASSERT_OK(loom_check_strip_comments(
      iree_make_cstring_view("// This is a comment\n"
                             "  %x = test.addi %a, %b : i32\n"),
      &builder));
  iree_string_view_t result = iree_string_builder_view(&builder);
  EXPECT_TRUE(iree_string_view_find(result, iree_make_cstring_view("This is"),
                                    0) == IREE_STRING_VIEW_NPOS);
  EXPECT_TRUE(iree_string_view_find(result, iree_make_cstring_view("test.addi"),
                                    0) != IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);
}

TEST_F(CheckParseTest, StripCommentsPreservesBlankLines) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator_, &builder);
  IREE_ASSERT_OK(loom_check_strip_comments(
      iree_make_cstring_view("  %x = test.addi %a, %b : i32\n"
                             "\n"
                             "  %y = test.muli %x, %x : i32\n"),
      &builder));
  iree_string_view_t result = iree_string_builder_view(&builder);
  EXPECT_TRUE(iree_string_view_find(result, iree_make_cstring_view("\n\n"),
                                    0) != IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);
}

TEST_F(CheckParseTest, StripCommentsPreservesTrailingComment) {
  // Trailing comments on IR lines are preserved as-is. Only standalone
  // comment lines (where the trimmed line starts with "//") are stripped.
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator_, &builder);
  IREE_ASSERT_OK(loom_check_strip_comments(
      iree_make_cstring_view("  %r = test.addi %x, %y : i32  // add them\n"),
      &builder));
  iree_string_view_t result = iree_string_builder_view(&builder);
  EXPECT_TRUE(iree_string_view_find(result, iree_make_cstring_view("test.addi"),
                                    0) != IREE_STRING_VIEW_NPOS);
  EXPECT_TRUE(iree_string_view_find(result, iree_make_cstring_view("add them"),
                                    0) != IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);
}

TEST_F(CheckParseTest, StripCommentsPreservesIndentation) {
  // IR lines must preserve their original indentation. Standalone
  // comment lines become blank lines without affecting adjacent lines.
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator_, &builder);
  IREE_ASSERT_OK(loom_check_strip_comments(
      iree_make_cstring_view("  // standalone comment\n"
                             "    %x = test.addi %a, %b : i32\n"),
      &builder));
  iree_string_view_t result = iree_string_builder_view(&builder);
  EXPECT_TRUE(iree_string_view_find(
                  result, iree_make_cstring_view("    %x = test.addi"), 0) !=
              IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);
}

// ===----------------------------------------------------------------------===
// Edge cases
// ===----------------------------------------------------------------------===

TEST_F(CheckParseTest, EmptySource) {
  IREE_ASSERT_OK(Parse(""));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_TRUE(iree_string_view_is_empty(file_.cases[0].input));
}

TEST_F(CheckParseTest, OnlyDirectives) {
  IREE_ASSERT_OK(Parse("// RUN: verify\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_VERIFY);
  EXPECT_TRUE(iree_string_view_is_empty(file_.cases[0].input));
}

TEST_F(CheckParseTest, PerCaseDirectives) {
  IREE_ASSERT_OK(
      Parse("// RUN: roundtrip\n"
            "func.def @a() {}\n"
            "// ====\n"
            "\n"
            "// RUN: verify\n"
            "func.def @b() {}\n"));
  ASSERT_EQ(file_.case_count, 2);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_ROUNDTRIP);
  EXPECT_EQ(file_.cases[1].mode, LOOM_CHECK_MODE_VERIFY);
}

TEST_F(CheckParseTest, CaseSeparatorAsFirstLineErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// ====\n"
                              "func.def @f() {}\n"));
}

TEST_F(CheckParseTest, CaseSeparatorAfterEmptyDirectivePreambleErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: verify\n"
                              "// ====\n"
                              "func.def @f() {}\n"));
}

TEST_F(CheckParseTest, CaseSeparatorAsLastLine) {
  IREE_ASSERT_OK(
      Parse("func.def @f() {}\n"
            "// ====\n"));
  ASSERT_EQ(file_.case_count, 2);
  EXPECT_TRUE(iree_string_view_find(file_.cases[0].input,
                                    iree_make_cstring_view("func.def"),
                                    0) != IREE_STRING_VIEW_NPOS);
  EXPECT_TRUE(iree_string_view_is_empty(file_.cases[1].input));
}

TEST_F(CheckParseTest, ExpectedSeparatorWithLeadingWhitespace) {
  IREE_ASSERT_OK(
      Parse("func.def @f() {\n"
            "}\n"
            "  // ----\n"
            "func.def @g() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_TRUE(file_.cases[0].has_expected_section);
}

TEST_F(CheckParseTest, ExpectedSeparatorWithTrailingTextIsNot) {
  IREE_ASSERT_OK(
      Parse("func.def @f() {}\n"
            "// ----extra\n"
            "func.def @g() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_FALSE(file_.cases[0].has_expected_section);
  EXPECT_TRUE(iree_string_view_find(file_.cases[0].input,
                                    iree_make_cstring_view("----extra"),
                                    0) != IREE_STRING_VIEW_NPOS);
}

TEST_F(CheckParseTest, OnlyFirstExpectedSeparatorSplits) {
  IREE_ASSERT_OK(
      Parse("input line\n"
            "// ----\n"
            "expected line\n"
            "// ----\n"
            "still expected\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_TRUE(file_.cases[0].has_expected_section);
  EXPECT_TRUE(iree_string_view_find(file_.cases[0].expected,
                                    iree_make_cstring_view("still expected"),
                                    0) != IREE_STRING_VIEW_NPOS);
}

TEST_F(CheckParseTest, CaseSeparatorWithExtraEquals) {
  // Extra equals signs beyond the minimum four are allowed.
  IREE_ASSERT_OK(
      Parse("func.def @a() {}\n"
            "// ========\n"
            "func.def @b() {}\n"));
  ASSERT_EQ(file_.case_count, 2);
}

TEST_F(CheckParseTest, NoTrailingNewline) {
  IREE_ASSERT_OK(Parse("func.def @f() {}"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_TRUE(iree_string_view_find(file_.cases[0].input,
                                    iree_make_cstring_view("func.def"),
                                    0) != IREE_STRING_VIEW_NPOS);
}

TEST_F(CheckParseTest, OnlyWhitespace) {
  IREE_ASSERT_OK(Parse("  \n\n  \n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_ROUNDTRIP);
}

TEST_F(CheckParseTest, BlankLinesBetweenDirectives) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "\n"
            "// XFAIL: reason\n"
            "func.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_VERIFY);
  EXPECT_TRUE(file_.cases[0].xfail);
  EXPECT_TRUE(iree_string_view_find(file_.cases[0].input,
                                    iree_make_cstring_view("// RUN:"),
                                    0) == IREE_STRING_VIEW_NPOS);
  EXPECT_TRUE(iree_string_view_find(file_.cases[0].input,
                                    iree_make_cstring_view("// XFAIL:"),
                                    0) == IREE_STRING_VIEW_NPOS);
}

// ===----------------------------------------------------------------------===
// Source ranges
// ===----------------------------------------------------------------------===

TEST_F(CheckParseTest, SourceRangesForSingleCaseSections) {
  const char* source =
      "// RUN: pass dce\n"
      "// XFAIL: nope\n"
      "func.def @f() {}\n"
      "// ----\n"
      "func.def @g() {}\n";
  IREE_ASSERT_OK(Parse(source));
  ASSERT_EQ(file_.case_count, 1);

  const loom_check_case_t& test_case = file_.cases[0];
  ExpectRange(test_case.source_range, 0, strlen(source));
  EXPECT_TRUE(loom_check_source_range_is_empty(test_case.separator_range));
  ExpectRangeForFragment(test_case.run_directive_range, "// RUN: pass dce");
  ExpectRangeForFragment(test_case.xfail_directive_range, "// XFAIL: nope");
  ExpectRange(test_case.input_range, OffsetOf("func.def @f()"),
              OffsetOf("// ----"));
  ExpectRangeForFragment(test_case.expected_separator_range, "// ----");
  ExpectRange(test_case.expected_range, OffsetOf("func.def @g()"),
              strlen(source));
}

TEST_F(CheckParseTest, SourceRangesForSeparatorsAndAnnotations) {
  const char* source =
      "// RUN: verify\n"
      "func.def @first() {}\n"
      "// ====\n"
      "  // ERROR@+1: PARSE/006 \"bogus\"\n"
      "  bogus.nonexistent\n";
  IREE_ASSERT_OK(Parse(source));
  ASSERT_EQ(file_.case_count, 2);
  ASSERT_EQ(file_.cases[1].annotation_count, 1);

  const loom_check_case_t& test_case = file_.cases[1];
  ExpectRangeForFragment(test_case.separator_range, "// ====");
  EXPECT_TRUE(loom_check_source_range_is_empty(test_case.run_directive_range));
  ExpectRange(test_case.source_range, OffsetOf("  // ERROR"), strlen(source));
  ExpectRange(test_case.input_range, OffsetOf("  // ERROR"), strlen(source));
  ExpectRangeForFragment(test_case.annotations[0].source_range,
                         "  // ERROR@+1: PARSE/006 \"bogus\"");
}

// ===----------------------------------------------------------------------===
// CRLF handling
// ===----------------------------------------------------------------------===

TEST_F(CheckParseTest, CRLFLineEndingsParseIdentically) {
  // Windows-style CRLF should produce the same results as LF.
  IREE_ASSERT_OK(
      Parse("// RUN: verify\r\n"
            "// ERROR@+1: TYPE/001\r\n"
            "  %x = test.bad : i32\r\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_VERIFY);
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  EXPECT_EQ(file_.cases[0].annotations[0].severity, LOOM_DIAGNOSTIC_ERROR);
  EXPECT_EQ(file_.cases[0].annotations[0].code, 1);
}

TEST_F(CheckParseTest, CRLFSeparators) {
  IREE_ASSERT_OK(
      Parse("func.def @a() {}\r\n"
            "// ====\r\n"
            "func.def @b() {}\r\n"
            "// ----\r\n"
            "func.def @c() {}\r\n"));
  ASSERT_EQ(file_.case_count, 2);
  EXPECT_TRUE(file_.cases[1].has_expected_section);
}

TEST_F(CheckParseTest, CRLFDirective) {
  // "// RUN: roundtrip\r\n" — the \r must be stripped before mode
  // parsing, otherwise "roundtrip\r" != "roundtrip".
  IREE_ASSERT_OK(Parse("// RUN: roundtrip\r\nfunc.def @f() {}\r\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_ROUNDTRIP);
}

// ===----------------------------------------------------------------------===
// Directive parsing — error paths and boundary cases
// ===----------------------------------------------------------------------===

TEST_F(CheckParseTest, PassWithoutSpaceIsUnknownMode) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse("// RUN: pass\n"));
}

TEST_F(CheckParseTest, VerifyWithTrailingTextIsUnknownMode) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: verify extra\n"));
}

TEST_F(CheckParseTest, ModeIsCaseSensitive) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: Roundtrip\n"));
}

// ===----------------------------------------------------------------------===
// Section splitting — boundary cases
// ===----------------------------------------------------------------------===

// (These are covered above in the Edge cases and Section splitting sections.)

// ===----------------------------------------------------------------------===
// Annotation parsing — error paths
// ===----------------------------------------------------------------------===

TEST_F(CheckParseTest, AnnotationOffsetMissingColon) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: verify\n"
                              "// ERROR@+1 TYPE/001\n"
                              "  %x = test.bad : i32\n"));
}

TEST_F(CheckParseTest, AnnotationOffsetNonNumeric) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: verify\n"
                              "// ERROR@+abc: TYPE/001\n"
                              "  %x = test.bad : i32\n"));
}

TEST_F(CheckParseTest, AnnotationNegativeOffsetUnderflow) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: verify\n"
                              "  %x = test.bad : i32\n"
                              "// ERROR@-5: TYPE/001\n"));
}

TEST_F(CheckParseTest, AnnotationNegativeOffsetValue) {
  // A negative number after @ (e.g. @+-1) is rejected because the
  // atoi will return a negative value which we explicitly disallow.
  // The @-N syntax uses a sign character, not a negative number.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: verify\n"
                              "// ERROR@+-1: TYPE/001\n"
                              "  %x = test.bad : i32\n"));
}

TEST_F(CheckParseTest, AnnotationUnterminatedLeadingQuote) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: verify\n"
                              "// ERROR: \"unterminated\n"
                              "  %x = test.bad : i32\n"));
}

TEST_F(CheckParseTest, AnnotationUnterminatedTrailingQuote) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: verify\n"
                              "// ERROR: TYPE/001 \"unterminated\n"
                              "  %x = test.bad : i32\n"));
}

TEST_F(CheckParseTest, AnnotationUnquotedTrailingText) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: verify\n"
                              "// ERROR: TYPE/001 extra\n"
                              "  %x = test.bad : i32\n"));
}

TEST_F(CheckParseTest, AnnotationParamMatcherRequiresQuotedValue) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: verify\n"
                              "// ERROR: DOMINANCE/011 {op_name=test.add.i32}\n"
                              "  %x = test.bad : i32\n"));
}

TEST_F(CheckParseTest, AnnotationNonNumericCodeErrors) {
  // "TYPE/abc" — non-numeric code after '/' must be rejected.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: verify\n"
                              "// ERROR: TYPE/abc\n"
                              "  %x = test.bad : i32\n"));
}

TEST_F(CheckParseTest, AnnotationEmptyCodeAfterSlashErrors) {
  // "TYPE/" — empty code after '/' must be rejected.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: verify\n"
                              "// ERROR: TYPE/\n"
                              "  %x = test.bad : i32\n"));
}

TEST_F(CheckParseTest, AnnotationUnknownDomainErrors) {
  // "BOGUS/001" — domain name not in the known set must be rejected.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: verify\n"
                              "// ERROR: BOGUS/001\n"
                              "  %x = test.bad : i32\n"));
}

TEST_F(CheckParseTest, AnnotationUnknownDomainWithoutCodeErrors) {
  // Bare "BOGUS" (no code) is also rejected.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: verify\n"
                              "// ERROR: BOGUS\n"
                              "  %x = test.bad : i32\n"));
}

TEST_F(CheckParseTest, AnnotationWithEmptyMatcher) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR:\n"
            "  %x = test.bad : i32\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  const auto& ann = file_.cases[0].annotations[0];
  EXPECT_EQ(ann.severity, LOOM_DIAGNOSTIC_ERROR);
  EXPECT_EQ(ann.domain, LOOM_ERROR_DOMAIN_COUNT_);
  EXPECT_EQ(ann.code, 0);
  EXPECT_EQ(ann.message_substring_count, 0);
}

TEST_F(CheckParseTest, SeverityWordInNormalTextNotAnnotation) {
  // "this ERROR is expected" has "ERROR" but no colon immediately
  // after the keyword + optional offset, so it's not an annotation.
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// this ERROR is expected\n"
            "  %x = test.ok : i32\n"));
  EXPECT_EQ(file_.cases[0].annotation_count, 0);
}

TEST_F(CheckParseTest, AnnotationNotMatchedInStringLiteral) {
  // "// ERROR:" inside a string literal is on a non-comment line and
  // should not be recognized as an annotation. extract_comment_text
  // only examines standalone comment lines.
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "  %x = test.string \"// ERROR: TYPE/001\"\n"));
  EXPECT_EQ(file_.cases[0].annotation_count, 0);
}

TEST_F(CheckParseTest, AnnotationOffsetWithoutSign) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR@1: TYPE/001\n"
            "  %x = test.bad : i32\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  EXPECT_EQ(file_.cases[0].annotations[0].target_line, 2);
}

TEST_F(CheckParseTest, AnnotationOffsetZero) {
  // @+0 targets the annotation's own line.
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR@+0: TYPE/001\n"
            "  %x = test.bad : i32\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  EXPECT_EQ(file_.cases[0].annotations[0].target_line, 1);
}

TEST_F(CheckParseTest, AnnotationCodeExplicitZero) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR@+1: TYPE/000\n"
            "  %x = test.bad : i32\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  EXPECT_EQ(file_.cases[0].annotations[0].code, 0);
}

TEST_F(CheckParseTest, AnnotationWithExtraWhitespace) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR:   TYPE/001   \"msg\"  \n"
            "  %x = test.bad : i32\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  const auto& ann = file_.cases[0].annotations[0];
  EXPECT_EQ(ann.domain, LOOM_ERROR_DOMAIN_TYPE);
  EXPECT_EQ(ann.code, 1);
  ASSERT_EQ(ann.message_substring_count, 1);
  EXPECT_TRUE(iree_string_view_equal(ann.message_substrings[0],
                                     iree_make_cstring_view("msg")));
}

TEST_F(CheckParseTest, AnnotationOnLastLineNoNewline) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR@+1: TYPE/001\n"
            "  %x = test.bad : i32"));
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  EXPECT_EQ(file_.cases[0].annotations[0].code, 1);
}

TEST_F(CheckParseTest, SingleAnnotationPerComment) {
  // Only one annotation per comment line. A comment like
  // "// ERROR: A // WARNING: B" finds "ERROR: A // WARNING: B" as the
  // comment text — the second "//" is just part of the text, not a
  // second annotation.
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR@+1: TYPE/001\n"
            "  %x = test.bad : i32\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
}

// ===----------------------------------------------------------------------===
// Comment stripping — boundary cases
// ===----------------------------------------------------------------------===

TEST_F(CheckParseTest, StripCommentsEmptyInput) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator_, &builder);
  IREE_ASSERT_OK(loom_check_strip_comments(iree_string_view_empty(), &builder));
  EXPECT_EQ(iree_string_builder_size(&builder), 0);
  iree_string_builder_deinitialize(&builder);
}

TEST_F(CheckParseTest, StripCommentsBareDoubleSlash) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator_, &builder);
  IREE_ASSERT_OK(loom_check_strip_comments(
      iree_make_cstring_view("//\n"
                             "  %x = test.addi %a, %b : i32\n"),
      &builder));
  iree_string_view_t result = iree_string_builder_view(&builder);
  EXPECT_TRUE(
      iree_string_view_starts_with(result, iree_make_cstring_view("\n")));
  EXPECT_TRUE(iree_string_view_find(result, iree_make_cstring_view("test.addi"),
                                    0) != IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);
}

TEST_F(CheckParseTest, StripCommentsNoTrailingNewline) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator_, &builder);
  IREE_ASSERT_OK(loom_check_strip_comments(
      iree_make_cstring_view("  %x = test.addi %a, %b : i32"), &builder));
  iree_string_view_t result = iree_string_builder_view(&builder);
  EXPECT_TRUE(iree_string_view_find(result, iree_make_cstring_view("test.addi"),
                                    0) != IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);
}

// ===----------------------------------------------------------------------===
// Structural integrity
// ===----------------------------------------------------------------------===

TEST_F(CheckParseTest, InputAndExpectedPointIntoSource) {
  const char* source =
      "func.def @f() {\n"
      "}\n"
      "// ----\n"
      "func.def @g() {}\n";
  IREE_ASSERT_OK(Parse(source));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_GE(file_.cases[0].input.data, source);
  EXPECT_LE(file_.cases[0].input.data + file_.cases[0].input.size,
            source + strlen(source));
  EXPECT_GE(file_.cases[0].expected.data, source);
  EXPECT_LE(file_.cases[0].expected.data + file_.cases[0].expected.size,
            source + strlen(source));
}

TEST_F(CheckParseTest, AnnotationLineNumbersAccountForDirectives) {
  // Line numbers are relative to the input section, not the file.
  // The "// RUN: verify" directive is consumed and not counted.
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR@+1: TYPE/001\n"
            "  %x = test.bad : i32\n"
            "  %y = test.ok : f32\n"
            "// ERROR@+1: TYPE/002\n"
            "  %z = test.bad : i64\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 2);
  EXPECT_EQ(file_.cases[0].annotations[0].target_line, 2);
  EXPECT_EQ(file_.cases[0].annotations[1].target_line, 5);
}

TEST_F(CheckParseTest, DirectivesPreservedAcrossParsing) {
  IREE_ASSERT_OK(
      Parse("// RUN: roundtrip\n"
            "func.def @a() {}\n"
            "// ====\n"
            "\n"
            "// RUN: verify\n"
            "func.def @b() {}\n"
            "// ====\n"
            "\n"
            "// RUN: pass dce\n"
            "func.def @c() {}\n"));
  ASSERT_EQ(file_.case_count, 3);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_ROUNDTRIP);
  EXPECT_EQ(file_.cases[1].mode, LOOM_CHECK_MODE_VERIFY);
  EXPECT_EQ(file_.cases[2].mode, LOOM_CHECK_MODE_PASS);
}

TEST_F(CheckParseTest, ManyCasesExercisesArenaBulkAllocation) {
  // 65 separators = 66 cases. Exercises the arena's ability to handle
  // many small allocations (one cases array + 66 annotation arrays).
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator_, &builder);
  IREE_ASSERT_OK(
      iree_string_builder_append_cstring(&builder, "func.def @case_0() {}\n"));
  for (int i = 1; i <= 65; ++i) {
    IREE_ASSERT_OK(iree_string_builder_append_cstring(&builder, "// ====\n"));
    char body[64];
    iree_snprintf(body, sizeof(body), "func.def @case_%d() {}\n", i);
    IREE_ASSERT_OK(iree_string_builder_append_cstring(&builder, body));
  }
  iree_string_view_t source = iree_string_builder_view(&builder);
  IREE_ASSERT_OK(loom_check_parse(source, &arena_, &file_));
  ASSERT_EQ(file_.case_count, 66);
  iree_string_builder_deinitialize(&builder);
}

// ===----------------------------------------------------------------------===
// Annotation parsing — trailing garbage after quotes
// ===----------------------------------------------------------------------===

TEST_F(CheckParseTest, AnnotationTrailingGarbageAfterSubstringErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: verify\n"
                              "// ERROR: \"msg\" garbage\n"
                              "  %x = test.bad : i32\n"));
}

TEST_F(CheckParseTest, AnnotationTrailingGarbageAfterDomainSubstringErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: verify\n"
                              "// ERROR: TYPE/001 \"msg\" extra\n"
                              "  %x = test.bad : i32\n"));
}

// ===----------------------------------------------------------------------===
// Annotation code boundary values
// ===----------------------------------------------------------------------===

TEST_F(CheckParseTest, AnnotationCodeAtUint16Max) {
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR@+1: TYPE/65535\n"
            "  %x = test.bad : i32\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  EXPECT_EQ(file_.cases[0].annotations[0].code, 65535);
}

TEST_F(CheckParseTest, AnnotationCodeAboveUint16MaxErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: verify\n"
                              "// ERROR: TYPE/65536\n"
                              "  %x = test.bad : i32\n"));
}

TEST_F(CheckParseTest, AnnotationCodeNegativeErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: verify\n"
                              "// ERROR: TYPE/-1\n"
                              "  %x = test.bad : i32\n"));
}

// ===----------------------------------------------------------------------===
// Annotations in expected section
// ===----------------------------------------------------------------------===

TEST_F(CheckParseTest, AnnotationInExpectedSectionErrors) {
  // Annotations below the // ---- separator are always a mistake.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: verify\n"
                              "  %x = test.bad : i32\n"
                              "// ----\n"
                              "// ERROR: TYPE/001\n"
                              "  %x = test.bad : i32\n"));
}

TEST_F(CheckParseTest, AnnotationInExpectedSectionCRLFErrors) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse("// RUN: verify\r\n"
                              "  %x = test.bad : i32\r\n"
                              "// ----\r\n"
                              "// ERROR: TYPE/001\r\n"
                              "  %x = test.bad : i32\r\n"));
}

TEST_F(CheckParseTest, NonAnnotationCommentInExpectedSectionOk) {
  // Regular comments in the expected section are fine.
  IREE_ASSERT_OK(
      Parse("func.def @f() {}\n"
            "// ----\n"
            "// This is just a comment, not an annotation.\n"
            "func.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_TRUE(file_.cases[0].has_expected_section);
}

// ===----------------------------------------------------------------------===
// Comment stripping — CRLF
// ===----------------------------------------------------------------------===

TEST_F(CheckParseTest, StripCommentsCRLF) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator_, &builder);
  IREE_ASSERT_OK(loom_check_strip_comments(
      iree_make_cstring_view("// standalone\r\n"
                             "  %x = test.addi %a, %b : i32\r\n"),
      &builder));
  iree_string_view_t result = iree_string_builder_view(&builder);
  EXPECT_TRUE(
      iree_string_view_starts_with(result, iree_make_cstring_view("\n")));
  EXPECT_TRUE(iree_string_view_find(result, iree_make_cstring_view("test.addi"),
                                    0) != IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);
}

// ===----------------------------------------------------------------------===
// NULL-backed empty views
// ===----------------------------------------------------------------------===

TEST_F(CheckParseTest, NullBackedEmptyViewParses) {
  // iree_string_view_empty() has data == NULL, size == 0. The parser
  // must handle this without undefined behavior (NULL + 0 is UB).
  iree_string_view_t empty = iree_string_view_empty();
  file_ = {};
  IREE_ASSERT_OK(loom_check_parse(empty, &arena_, &file_));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_TRUE(iree_string_view_is_empty(file_.cases[0].input));
}

// ===----------------------------------------------------------------------===
// Annotation without space after // (consistency)
// ===----------------------------------------------------------------------===

TEST_F(CheckParseTest, AnnotationWithoutSpaceAfterSlashesRecognized) {
  // "//ERROR@+1: TYPE/001" (no space after //) must be recognized as
  // an annotation, not consumed as a header comment. This ensures
  // is_annotation_line and extract_comment_text agree.
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "//ERROR@+1: TYPE/001\n"
            "  %x = test.bad : i32\n"));
  ASSERT_EQ(file_.cases[0].annotation_count, 1);
  EXPECT_EQ(file_.cases[0].annotations[0].target_line, 2);
  EXPECT_EQ(file_.cases[0].annotations[0].domain, LOOM_ERROR_DOMAIN_TYPE);
  EXPECT_EQ(file_.cases[0].annotations[0].code, 1);
}

// ===----------------------------------------------------------------------===
// Expected separator not consumed as header comment
// ===----------------------------------------------------------------------===

TEST_F(CheckParseTest, ExpectedSeparatorNotConsumedByHeader) {
  // A case with directives followed immediately by // ---- must not
  // have the separator consumed as a header comment. The expected
  // section should be recognized even when there is no IR before it.
  IREE_ASSERT_OK(
      Parse("// RUN: pass dce\n"
            "// ----\n"
            "func.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_TRUE(file_.cases[0].has_expected_section);
  EXPECT_TRUE(iree_string_view_find(file_.cases[0].expected,
                                    iree_make_cstring_view("func.def"),
                                    0) != IREE_STRING_VIEW_NPOS);
}

// ===----------------------------------------------------------------------===
// RUN inheritance
// ===----------------------------------------------------------------------===

TEST_F(CheckParseTest, FirstCaseRunInherited) {
  // The first case's // RUN: directive sets the file default. Cases without
  // their own // RUN: inherit from it.
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR@+1: PARSE/006\n"
            "bogus.nonexistent\n"
            "// ====\n"
            "func.def @clean() {}\n"));
  ASSERT_EQ(file_.case_count, 2);
  EXPECT_EQ(file_.default_mode, LOOM_CHECK_MODE_VERIFY);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_VERIFY);
  EXPECT_TRUE(file_.cases[0].has_run_directive);
  EXPECT_EQ(file_.cases[1].mode, LOOM_CHECK_MODE_VERIFY);
  EXPECT_FALSE(file_.cases[1].has_run_directive);
}

TEST_F(CheckParseTest, FirstCaseRunOverriddenByCase) {
  // A case with its own // RUN: overrides the file default.
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "// ERROR@+1: PARSE/006\n"
            "bogus.nonexistent\n"
            "// ====\n"
            "\n"
            "// RUN: roundtrip\n"
            "func.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 2);
  EXPECT_EQ(file_.default_mode, LOOM_CHECK_MODE_VERIFY);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_VERIFY);
  EXPECT_TRUE(file_.cases[0].has_run_directive);
  EXPECT_EQ(file_.cases[1].mode, LOOM_CHECK_MODE_ROUNDTRIP);
  EXPECT_TRUE(file_.cases[1].has_run_directive);
}

TEST_F(CheckParseTest, FirstCasePassModeInherited) {
  // File default can be pass mode with a pipeline.
  IREE_ASSERT_OK(
      Parse("// RUN: pass dce\n"
            "func.def @a() {}\n"
            "// ====\n"
            "func.def @b() {}\n"));
  ASSERT_EQ(file_.case_count, 2);
  EXPECT_EQ(file_.default_mode, LOOM_CHECK_MODE_PASS);
  EXPECT_TRUE(iree_string_view_equal(file_.default_pipeline,
                                     iree_make_cstring_view("dce")));
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_PASS);
  EXPECT_TRUE(iree_string_view_equal(file_.cases[0].pipeline,
                                     iree_make_cstring_view("dce")));
  EXPECT_EQ(file_.cases[1].mode, LOOM_CHECK_MODE_PASS);
  EXPECT_TRUE(iree_string_view_equal(file_.cases[1].pipeline,
                                     iree_make_cstring_view("dce")));
}

TEST_F(CheckParseTest, FirstCaseEmitModeInherited) {
  IREE_ASSERT_OK(
      Parse("// RUN: emit target-form @entry\n"
            "func.def @a() {}\n"
            "// ====\n"
            "func.def @b() {}\n"));
  ASSERT_EQ(file_.case_count, 2);
  EXPECT_EQ(file_.default_mode, LOOM_CHECK_MODE_EMIT);
  EXPECT_TRUE(iree_string_view_equal(
      file_.default_emit_target, iree_make_cstring_view("target-form @entry")));
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_EMIT);
  EXPECT_TRUE(
      iree_string_view_equal(file_.cases[0].emit_target,
                             iree_make_cstring_view("target-form @entry")));
  EXPECT_EQ(file_.cases[1].mode, LOOM_CHECK_MODE_EMIT);
  EXPECT_TRUE(
      iree_string_view_equal(file_.cases[1].emit_target,
                             iree_make_cstring_view("target-form @entry")));
}

TEST_F(CheckParseTest, FirstCaseRequiresInheritedAndCombined) {
  IREE_ASSERT_OK(
      Parse("// REQUIRES: tool-dis\n"
            "func.def @a() {}\n"
            "// ====\n"
            "\n"
            "// REQUIRES: tool-backend, tool-dis\n"
            "func.def @b() {}\n"));
  ASSERT_EQ(file_.case_count, 2);
  ASSERT_EQ(file_.default_requirement_count, 1);
  EXPECT_TRUE(iree_string_view_equal(file_.default_requirements[0],
                                     iree_make_cstring_view("tool-dis")));
  EXPECT_TRUE(file_.cases[0].has_requires_directive);
  ASSERT_EQ(file_.cases[0].requirement_count, 1);
  ExpectRequirementName(file_.cases[0], 0, "tool-dis");
  EXPECT_TRUE(file_.cases[1].has_requires_directive);
  ASSERT_EQ(file_.cases[1].requirement_count, 2);
  ExpectRequirementName(file_.cases[1], 0, "tool-dis");
  ExpectRequirementName(file_.cases[1], 1, "tool-backend");
}

TEST_F(CheckParseTest, NoPreambleDefaultIsRoundtrip) {
  // No preamble (file does not start with // ====), no RUN in any
  // case. All cases default to roundtrip.
  IREE_ASSERT_OK(
      Parse("func.def @a() {}\n"
            "// ====\n"
            "func.def @b() {}\n"));
  ASSERT_EQ(file_.case_count, 2);
  EXPECT_EQ(file_.default_mode, LOOM_CHECK_MODE_ROUNDTRIP);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_ROUNDTRIP);
  EXPECT_EQ(file_.cases[1].mode, LOOM_CHECK_MODE_ROUNDTRIP);
}

TEST_F(CheckParseTest, FirstCaseRunBecomesDefault) {
  // When the file does not start with // ==== the first case's RUN
  // directive serves as both its own mode and the file default.
  IREE_ASSERT_OK(
      Parse("// RUN: verify\n"
            "func.def @a() {}\n"
            "// ====\n"
            "\n"
            "// ERROR@+1: PARSE/006\n"
            "bogus.nonexistent\n"));
  ASSERT_EQ(file_.case_count, 2);
  EXPECT_EQ(file_.default_mode, LOOM_CHECK_MODE_VERIFY);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_VERIFY);
  EXPECT_TRUE(file_.cases[0].has_run_directive);
  EXPECT_EQ(file_.cases[1].mode, LOOM_CHECK_MODE_VERIFY);
  EXPECT_FALSE(file_.cases[1].has_run_directive);
}

TEST_F(CheckParseTest, FirstCaseRequiresBecomesDefault) {
  IREE_ASSERT_OK(
      Parse("// REQUIRES: tool-dis\n"
            "func.def @a() {}\n"
            "// ====\n"
            "func.def @b() {}\n"));
  ASSERT_EQ(file_.case_count, 2);
  ASSERT_EQ(file_.default_requirement_count, 1);
  ASSERT_EQ(file_.cases[0].requirement_count, 1);
  ASSERT_EQ(file_.cases[1].requirement_count, 1);
  EXPECT_TRUE(file_.cases[0].has_requires_directive);
  EXPECT_FALSE(file_.cases[1].has_requires_directive);
  ExpectRequirementName(file_.cases[1], 0, "tool-dis");
}

TEST_F(CheckParseTest, HasRunDirectiveFlag) {
  IREE_ASSERT_OK(
      Parse("// RUN: roundtrip\n"
            "func.def @a() {}\n"
            "// ====\n"
            "func.def @b() {}\n"));
  ASSERT_EQ(file_.case_count, 2);
  EXPECT_TRUE(file_.cases[0].has_run_directive);
  EXPECT_FALSE(file_.cases[1].has_run_directive);
}

TEST_F(CheckParseTest, SingleCaseNoRunDefaultsToRoundtrip) {
  IREE_ASSERT_OK(Parse("func.def @f() {}\n"));
  ASSERT_EQ(file_.case_count, 1);
  EXPECT_EQ(file_.cases[0].mode, LOOM_CHECK_MODE_ROUNDTRIP);
  EXPECT_FALSE(file_.cases[0].has_run_directive);
  EXPECT_EQ(file_.default_mode, LOOM_CHECK_MODE_ROUNDTRIP);
}

}  // namespace
