// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/diff.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

// Helper: compute diff with default context and return the result string.
// The builder is owned by the caller for lifetime management.
class DiffTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_string_builder_initialize(iree_allocator_system(), &builder_);
  }
  void TearDown() override { iree_string_builder_deinitialize(&builder_); }

  // Runs loom_diff and returns the diff output as a string view.
  iree_string_view_t Diff(
      iree_string_view_t expected, iree_string_view_t actual,
      iree_host_size_t context = LOOM_DIFF_DEFAULT_CONTEXT) {
    iree_status_t status = loom_diff(expected, actual, context,
                                     iree_allocator_system(), &builder_);
    IREE_EXPECT_OK(status);
    return iree_string_builder_view(&builder_);
  }

  // Returns true if the diff output contains |substring|.
  bool DiffContains(iree_string_view_t expected, iree_string_view_t actual,
                    const char* substring,
                    iree_host_size_t context = LOOM_DIFF_DEFAULT_CONTEXT) {
    iree_string_view_t diff = Diff(expected, actual, context);
    return iree_string_view_find(diff, iree_make_cstring_view(substring), 0) !=
           IREE_STRING_VIEW_NPOS;
  }

  iree_string_builder_t builder_;
};

//===----------------------------------------------------------------------===//
// Identity
//===----------------------------------------------------------------------===//

TEST_F(DiffTest, IdenticalStringsProduceEmptyDiff) {
  iree_string_view_t diff = Diff(IREE_SV("abc\ndef\n"), IREE_SV("abc\ndef\n"));
  EXPECT_EQ(diff.size, 0u);
}

TEST_F(DiffTest, IdenticalEmptyStringsProduceEmptyDiff) {
  iree_string_view_t diff = Diff(IREE_SV(""), IREE_SV(""));
  EXPECT_EQ(diff.size, 0u);
}

TEST_F(DiffTest, IdenticalSingleLineProducesEmptyDiff) {
  iree_string_view_t diff = Diff(IREE_SV("hello\n"), IREE_SV("hello\n"));
  EXPECT_EQ(diff.size, 0u);
}

//===----------------------------------------------------------------------===//
// Single-line changes
//===----------------------------------------------------------------------===//

TEST_F(DiffTest, SingleLineAddition) {
  EXPECT_TRUE(
      DiffContains(IREE_SV("aaa\nccc\n"), IREE_SV("aaa\nbbb\nccc\n"), "+bbb"));
}

TEST_F(DiffTest, SingleLineRemoval) {
  EXPECT_TRUE(
      DiffContains(IREE_SV("aaa\nbbb\nccc\n"), IREE_SV("aaa\nccc\n"), "-bbb"));
}

TEST_F(DiffTest, SingleLineChange) {
  iree_string_view_t diff =
      Diff(IREE_SV("aaa\nold\nccc\n"), IREE_SV("aaa\nnew\nccc\n"));
  EXPECT_NE(iree_string_view_find(diff, IREE_SV("-old"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(diff, IREE_SV("+new"), 0),
            IREE_STRING_VIEW_NPOS);
}

//===----------------------------------------------------------------------===//
// Empty inputs
//===----------------------------------------------------------------------===//

TEST_F(DiffTest, EmptyExpectedAllAdded) {
  iree_string_view_t diff = Diff(IREE_SV(""), IREE_SV("aaa\nbbb\n"));
  EXPECT_NE(iree_string_view_find(diff, IREE_SV("+aaa"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(diff, IREE_SV("+bbb"), 0),
            IREE_STRING_VIEW_NPOS);
}

TEST_F(DiffTest, EmptyActualAllRemoved) {
  iree_string_view_t diff = Diff(IREE_SV("aaa\nbbb\n"), IREE_SV(""));
  EXPECT_NE(iree_string_view_find(diff, IREE_SV("-aaa"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(diff, IREE_SV("-bbb"), 0),
            IREE_STRING_VIEW_NPOS);
}

//===----------------------------------------------------------------------===//
// Header
//===----------------------------------------------------------------------===//

TEST_F(DiffTest, HeaderPresent) {
  iree_string_view_t diff = Diff(IREE_SV("old\n"), IREE_SV("new\n"));
  EXPECT_NE(iree_string_view_find(diff, IREE_SV("--- expected"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(diff, IREE_SV("+++ actual"), 0),
            IREE_STRING_VIEW_NPOS);
}

TEST_F(DiffTest, NoHeaderWhenIdentical) {
  iree_string_view_t diff = Diff(IREE_SV("same\n"), IREE_SV("same\n"));
  EXPECT_EQ(diff.size, 0u);
}

//===----------------------------------------------------------------------===//
// Hunk headers
//===----------------------------------------------------------------------===//

TEST_F(DiffTest, HunkHeaderLineNumbers) {
  // Single line change at line 1.
  iree_string_view_t diff = Diff(IREE_SV("old\n"), IREE_SV("new\n"));
  EXPECT_NE(iree_string_view_find(diff, IREE_SV("@@ -1,1 +1,1 @@"), 0),
            IREE_STRING_VIEW_NPOS);
}

TEST_F(DiffTest, StructuredHunkLineNumbersAndKinds) {
  loom_diff_result_t result = {};
  IREE_ASSERT_OK(loom_diff_compute(IREE_SV("aaa\nold\nccc\n"),
                                   IREE_SV("aaa\nnew\nccc\n"), 1,
                                   iree_allocator_system(), &result));
  ASSERT_EQ(result.hunk_count, 1u);
  EXPECT_EQ(result.hunks[0].expected_start_line, 1u);
  EXPECT_EQ(result.hunks[0].expected_line_count, 3u);
  EXPECT_EQ(result.hunks[0].actual_start_line, 1u);
  EXPECT_EQ(result.hunks[0].actual_line_count, 3u);
  ASSERT_EQ(result.hunks[0].line_count, 4u);
  const iree_host_size_t line_offset = result.hunks[0].line_offset;
  EXPECT_EQ(result.lines[line_offset + 0].kind, LOOM_DIFF_HUNK_LINE_CONTEXT);
  EXPECT_EQ(result.lines[line_offset + 1].kind, LOOM_DIFF_HUNK_LINE_DELETE);
  EXPECT_EQ(result.lines[line_offset + 2].kind, LOOM_DIFF_HUNK_LINE_INSERT);
  EXPECT_EQ(result.lines[line_offset + 3].kind, LOOM_DIFF_HUNK_LINE_CONTEXT);
  EXPECT_TRUE(iree_string_view_equal(result.lines[line_offset + 1].text,
                                     IREE_SV("old")));
  EXPECT_TRUE(iree_string_view_equal(result.lines[line_offset + 2].text,
                                     IREE_SV("new")));
  EXPECT_STREQ(
      loom_diff_hunk_line_kind_name(result.lines[line_offset + 2].kind),
      "insert");
  loom_diff_result_deinitialize(iree_allocator_system(), &result);
}

//===----------------------------------------------------------------------===//
// Context lines
//===----------------------------------------------------------------------===//

TEST_F(DiffTest, ContextLinesIncluded) {
  // Change at line 3. With context=1, lines 2 and 4 should appear.
  iree_string_view_t diff =
      Diff(IREE_SV("1\n2\n3\n4\n5\n"), IREE_SV("1\n2\nX\n4\n5\n"), 1);
  EXPECT_NE(iree_string_view_find(diff, IREE_SV(" 2"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(diff, IREE_SV("-3"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(diff, IREE_SV("+X"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(diff, IREE_SV(" 4"), 0),
            IREE_STRING_VIEW_NPOS);
}

TEST_F(DiffTest, ZeroContextShowsOnlyChanges) {
  iree_string_view_t diff =
      Diff(IREE_SV("1\n2\n3\n4\n5\n"), IREE_SV("1\n2\nX\n4\n5\n"), 0);
  EXPECT_NE(iree_string_view_find(diff, IREE_SV("-3"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(diff, IREE_SV("+X"), 0),
            IREE_STRING_VIEW_NPOS);
  // Lines 1, 2, 4, 5 should not appear as context.
  EXPECT_EQ(iree_string_view_find(diff, IREE_SV(" 1"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(iree_string_view_find(diff, IREE_SV(" 5"), 0),
            IREE_STRING_VIEW_NPOS);
}

//===----------------------------------------------------------------------===//
// Multiple hunks
//===----------------------------------------------------------------------===//

TEST_F(DiffTest, TwoSeparateChanges) {
  // Changes at lines 2 and 8, with context=1. The gap (lines 4-6) is
  // larger than 2*context, so we get two separate hunks.
  iree_string_view_t expected = IREE_SV("1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n");
  iree_string_view_t actual = IREE_SV("1\nA\n3\n4\n5\n6\n7\nB\n9\n10\n");
  iree_string_view_t diff = Diff(expected, actual, 1);
  // Both changes should appear.
  EXPECT_NE(iree_string_view_find(diff, IREE_SV("-2\n"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(diff, IREE_SV("+A\n"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(diff, IREE_SV("-8\n"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(diff, IREE_SV("+B\n"), 0),
            IREE_STRING_VIEW_NPOS);
  // Should have two hunk headers. Find the first @@ line, skip past
  // it, and find a second one.
  iree_host_size_t first_header =
      iree_string_view_find(diff, IREE_SV("@@ "), 0);
  ASSERT_NE(first_header, IREE_STRING_VIEW_NPOS);
  iree_host_size_t after_first =
      iree_string_view_find(diff, IREE_SV("\n"), first_header);
  ASSERT_NE(after_first, IREE_STRING_VIEW_NPOS);
  iree_host_size_t second_header =
      iree_string_view_find(diff, IREE_SV("@@ "), after_first);
  EXPECT_NE(second_header, IREE_STRING_VIEW_NPOS);
}

//===----------------------------------------------------------------------===//
// Trailing newline handling
//===----------------------------------------------------------------------===//

TEST_F(DiffTest, NoTrailingNewline) {
  iree_string_view_t diff = Diff(IREE_SV("old"), IREE_SV("new"));
  EXPECT_NE(iree_string_view_find(diff, IREE_SV("-old"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(diff, IREE_SV("+new"), 0),
            IREE_STRING_VIEW_NPOS);
}

TEST_F(DiffTest, MixedTrailingNewline) {
  // One has trailing newline, other doesn't — they differ.
  iree_string_view_t diff = Diff(IREE_SV("line\n"), IREE_SV("line"));
  // These are identical content but differ in trailing newline
  // representation. Since we strip newlines from line content, the
  // single line "line" is the same in both. The diff should be empty.
  EXPECT_EQ(diff.size, 0u);
}

//===----------------------------------------------------------------------===//
// IR-like content
//===----------------------------------------------------------------------===//

TEST_F(DiffTest, LoomIRDiff) {
  iree_string_view_t expected = IREE_SV(
      "func.def @test(%x: i32) -> i32 {\n"
      "  %result = test.addi %x, %x : i32\n"
      "  func.return %result : i32\n"
      "}\n");
  iree_string_view_t actual = IREE_SV(
      "func.def @test(%x: i32) -> i32 {\n"
      "  %dead = test.constant 42 : i32\n"
      "  %result = test.addi %x, %x : i32\n"
      "  func.return %result : i32\n"
      "}\n");
  iree_string_view_t diff = Diff(expected, actual);
  EXPECT_NE(iree_string_view_find(diff, IREE_SV("+  %dead"), 0),
            IREE_STRING_VIEW_NPOS);
  // The other lines should appear as context, not as changes.
  EXPECT_EQ(iree_string_view_find(diff, IREE_SV("-  %result"), 0),
            IREE_STRING_VIEW_NPOS);
}

}  // namespace
}  // namespace loom
