// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/update.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/tools/loom-check/check.h"

namespace {

class UpdateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    iree_string_builder_initialize(iree_allocator_system(), &new_source_);
  }

  void TearDown() override {
    iree_string_builder_deinitialize(&new_source_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Parses |source| and returns the file structure. Arena-allocated.
  loom_check_file_t Parse(const char* source) {
    source_ = iree_make_cstring_view(source);
    loom_check_file_t file = {0};
    IREE_EXPECT_OK(loom_check_parse(source_, &arena_, &file));
    return file;
  }

  // Applies updates and returns the reconstructed source as a string.
  std::string Apply(const loom_check_file_t& file,
                    const loom_check_case_update_t* updates,
                    iree_host_size_t* out_update_count) {
    IREE_EXPECT_OK(loom_check_apply_updates(source_, &file, updates,
                                            &new_source_, out_update_count));
    return std::string(iree_string_builder_buffer(&new_source_),
                       iree_string_builder_size(&new_source_));
  }

  // Builds one machine-readable update edit and returns its replacement text.
  std::string BuildUpdateEdit(const loom_check_case_t& test_case,
                              iree_string_view_t actual_output,
                              loom_check_update_edit_t* out_edit) {
    IREE_EXPECT_OK(loom_check_build_update_edit(
        source_, &test_case, actual_output, &new_source_, out_edit));
    return std::string(iree_string_builder_buffer(&new_source_),
                       iree_string_builder_size(&new_source_));
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  iree_string_builder_t new_source_;
  iree_string_view_t source_;
};

TEST_F(UpdateTest, ReplacesExpectedSection) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @f() {\n"
      "}\n"
      "\n"
      "// ----\n"
      "old expected\n");

  ASSERT_EQ(file.case_count, 1u);

  loom_check_case_update_t updates[1] = {};
  updates[0].needs_update = true;
  updates[0].actual_output = iree_make_cstring_view("new expected\n");
  updates[0].expected_start = file.cases[0].expected.data;
  updates[0].expected_end =
      file.cases[0].expected.data + file.cases[0].expected.size;

  iree_host_size_t update_count = 0;
  std::string result = Apply(file, updates, &update_count);

  EXPECT_EQ(update_count, 1u);
  EXPECT_NE(result.find("// ----\n"), std::string::npos);
  EXPECT_NE(result.find("new expected\n"), std::string::npos);
  EXPECT_EQ(result.find("old expected"), std::string::npos);
}

TEST_F(UpdateTest, BuildEditReplacesExpectedSection) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @f() {\n"
      "}\n"
      "\n"
      "// ----\n"
      "old expected\n");

  ASSERT_EQ(file.case_count, 1u);

  loom_check_update_edit_t edit = {};
  std::string text = BuildUpdateEdit(
      file.cases[0], iree_make_cstring_view("new expected\n"), &edit);

  EXPECT_EQ(edit.kind, LOOM_CHECK_UPDATE_EDIT_REPLACE_EXPECTED_OUTPUT);
  EXPECT_EQ(edit.range.start_byte, file.cases[0].expected_range.start_byte);
  EXPECT_EQ(edit.range.end_byte, file.cases[0].expected_range.end_byte);
  EXPECT_EQ(text, "new expected\n");
  EXPECT_STREQ(loom_check_update_edit_kind_name(edit.kind),
               "replace_expected_output");
}

TEST_F(UpdateTest, InsertsExpectedSeparator) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @f() {\n"
      "}\n");

  ASSERT_EQ(file.case_count, 1u);

  loom_check_case_update_t updates[1] = {};
  updates[0].needs_update = true;
  updates[0].actual_output = iree_make_cstring_view("formatted output\n");
  updates[0].input_end = file.cases[0].input.data + file.cases[0].input.size;

  iree_host_size_t update_count = 0;
  std::string result = Apply(file, updates, &update_count);

  EXPECT_EQ(update_count, 1u);
  // Blank line before // ----, then the separator and output.
  EXPECT_NE(result.find("}\n\n// ----\n"), std::string::npos);
  EXPECT_NE(result.find("formatted output\n"), std::string::npos);
  // The original input should still be present.
  EXPECT_NE(result.find("func.def @f()"), std::string::npos);
}

TEST_F(UpdateTest, BuildEditInsertsExpectedSeparator) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @f() {\n"
      "}\n");

  ASSERT_EQ(file.case_count, 1u);

  loom_check_update_edit_t edit = {};
  std::string text = BuildUpdateEdit(
      file.cases[0], iree_make_cstring_view("formatted output\n"), &edit);

  EXPECT_EQ(edit.kind, LOOM_CHECK_UPDATE_EDIT_INSERT_EXPECTED_OUTPUT);
  EXPECT_EQ(edit.range.start_byte, file.cases[0].input_range.end_byte);
  EXPECT_EQ(edit.range.end_byte, file.cases[0].input_range.end_byte);
  EXPECT_EQ(text, "\n// ----\nformatted output\n");
  EXPECT_STREQ(loom_check_update_edit_kind_name(edit.kind),
               "insert_expected_output");
}

TEST_F(UpdateTest, BuildEditInsertsExpectedSeparatorForEmptyInsertedOutput) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @f() {\n"
      "}\n");

  ASSERT_EQ(file.case_count, 1u);

  loom_check_update_edit_t edit = {};
  std::string text =
      BuildUpdateEdit(file.cases[0], iree_string_view_empty(), &edit);

  EXPECT_EQ(edit.kind, LOOM_CHECK_UPDATE_EDIT_INSERT_EXPECTED_OUTPUT);
  EXPECT_EQ(edit.range.start_byte, file.cases[0].input_range.end_byte);
  EXPECT_EQ(edit.range.end_byte, file.cases[0].input_range.end_byte);
  EXPECT_EQ(text, "\n// ----\n");
}

TEST_F(UpdateTest, InsertsExpectedSeparatorForEmptyInsertedOutput) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @f() {\n"
      "}\n");

  ASSERT_EQ(file.case_count, 1u);

  loom_check_case_update_t updates[1] = {};
  updates[0].needs_update = true;
  updates[0].actual_output = iree_string_view_empty();
  updates[0].input_end = file.cases[0].input.data + file.cases[0].input.size;

  iree_host_size_t update_count = 0;
  std::string result = Apply(file, updates, &update_count);

  EXPECT_EQ(update_count, 1u);
  EXPECT_NE(result.find("}\n\n// ----\n"), std::string::npos);
}

TEST_F(UpdateTest, InsertsExpectedWithFollowingCase) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @a() {\n"
      "}\n"
      "\n"
      "// ====\n"
      "\n"
      "func.def @b() {\n"
      "}\n");

  ASSERT_EQ(file.case_count, 2u);

  loom_check_case_update_t updates[2] = {};
  updates[0].needs_update = true;
  updates[0].actual_output = iree_make_cstring_view("output a\n");
  updates[0].input_end = file.cases[0].input.data + file.cases[0].input.size;

  iree_host_size_t update_count = 0;
  std::string result = Apply(file, updates, &update_count);

  EXPECT_EQ(update_count, 1u);
  // Blank line before // ----.
  EXPECT_NE(result.find("}\n\n// ----\n"), std::string::npos);
  // Blank line between output and // ====.
  EXPECT_NE(result.find("output a\n\n"), std::string::npos);
  // The // ==== and second case should be preserved.
  EXPECT_NE(result.find("// ====\n"), std::string::npos);
  EXPECT_NE(result.find("func.def @b()"), std::string::npos);
}

TEST_F(UpdateTest, PreservesUnchangedCase) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @a() {\n"
      "}\n"
      "\n"
      "// ====\n"
      "\n"
      "func.def @b() {\n"
      "}\n"
      "\n"
      "// ----\n"
      "old b output\n");

  ASSERT_EQ(file.case_count, 2u);

  loom_check_case_update_t updates[2] = {};
  // Only update the second case.
  updates[1].needs_update = true;
  updates[1].actual_output = iree_make_cstring_view("new b output\n");
  updates[1].expected_start = file.cases[1].expected.data;
  updates[1].expected_end =
      file.cases[1].expected.data + file.cases[1].expected.size;

  iree_host_size_t update_count = 0;
  std::string result = Apply(file, updates, &update_count);

  EXPECT_EQ(update_count, 1u);
  // First case should be verbatim.
  EXPECT_NE(result.find("func.def @a()"), std::string::npos);
  // Second case should have the new output.
  EXPECT_NE(result.find("new b output"), std::string::npos);
  EXPECT_EQ(result.find("old b output"), std::string::npos);
}

TEST_F(UpdateTest, UpdateMultipleCases) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @a() {\n"
      "}\n"
      "\n"
      "// ----\n"
      "old a\n"
      "\n"
      "// ====\n"
      "\n"
      "func.def @b() {\n"
      "}\n"
      "\n"
      "// ====\n"
      "\n"
      "func.def @c() {\n"
      "}\n"
      "\n"
      "// ----\n"
      "old c\n");

  ASSERT_EQ(file.case_count, 3u);

  loom_check_case_update_t updates[3] = {};
  // Update cases 0 and 2, leave case 1 alone.
  updates[0].needs_update = true;
  updates[0].actual_output = iree_make_cstring_view("new a\n");
  updates[0].expected_start = file.cases[0].expected.data;
  updates[0].expected_end =
      file.cases[0].expected.data + file.cases[0].expected.size;

  updates[2].needs_update = true;
  updates[2].actual_output = iree_make_cstring_view("new c\n");
  updates[2].expected_start = file.cases[2].expected.data;
  updates[2].expected_end =
      file.cases[2].expected.data + file.cases[2].expected.size;

  iree_host_size_t update_count = 0;
  std::string result = Apply(file, updates, &update_count);

  EXPECT_EQ(update_count, 2u);
  EXPECT_NE(result.find("new a\n"), std::string::npos);
  EXPECT_EQ(result.find("old a"), std::string::npos);
  // Case b preserved.
  EXPECT_NE(result.find("func.def @b()"), std::string::npos);
  EXPECT_NE(result.find("new c\n"), std::string::npos);
  EXPECT_EQ(result.find("old c"), std::string::npos);
  // Blank lines around separators preserved after replacement.
  EXPECT_NE(result.find("new a\n\n// ====\n"), std::string::npos);
  EXPECT_NE(result.find("new c\n"), std::string::npos);
}

TEST_F(UpdateTest, EnsuresTrailingNewline) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @f() {\n"
      "}\n"
      "\n"
      "// ----\n"
      "old\n");

  ASSERT_EQ(file.case_count, 1u);

  // actual_output lacks trailing newline.
  loom_check_case_update_t updates[1] = {};
  updates[0].needs_update = true;
  updates[0].actual_output = iree_make_cstring_view("no trailing newline");
  updates[0].expected_start = file.cases[0].expected.data;
  updates[0].expected_end =
      file.cases[0].expected.data + file.cases[0].expected.size;

  iree_host_size_t update_count = 0;
  std::string result = Apply(file, updates, &update_count);

  EXPECT_EQ(update_count, 1u);
  // Should have added a trailing newline.
  EXPECT_NE(result.find("no trailing newline\n"), std::string::npos);
}

TEST_F(UpdateTest, PreservesDirectivesAndSeparators) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @unchanged() {}\n"
      "// ====\n"
      "\n"
      "func.def @f() {\n"
      "}\n"
      "\n"
      "// ----\n"
      "old\n");

  ASSERT_EQ(file.case_count, 2u);

  loom_check_case_update_t updates[2] = {};
  updates[1].needs_update = true;
  updates[1].actual_output = iree_make_cstring_view("new\n");
  updates[1].expected_start = file.cases[1].expected.data;
  updates[1].expected_end =
      file.cases[1].expected.data + file.cases[1].expected.size;

  iree_host_size_t update_count = 0;
  std::string result = Apply(file, updates, &update_count);

  EXPECT_EQ(update_count, 1u);
  // RUN directive and case separator preserved.
  EXPECT_NE(result.find("// RUN: roundtrip\n"), std::string::npos);
  EXPECT_NE(result.find("// ====\n"), std::string::npos);
  EXPECT_NE(result.find("// ----\n"), std::string::npos);
}

TEST_F(UpdateTest, NoChangesReturnsZeroCount) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @f() {\n"
      "}\n");

  ASSERT_EQ(file.case_count, 1u);

  loom_check_case_update_t updates[1] = {};
  // No updates needed.

  iree_host_size_t update_count = 0;
  std::string result = Apply(file, updates, &update_count);

  EXPECT_EQ(update_count, 0u);
  // The full source should be reproduced verbatim.
  EXPECT_EQ(result, std::string(source_.data, source_.size));
}

TEST_F(UpdateTest, EmptyActualOutput) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @f() {\n"
      "}\n"
      "\n"
      "// ----\n"
      "old\n");

  ASSERT_EQ(file.case_count, 1u);

  loom_check_case_update_t updates[1] = {};
  updates[0].needs_update = true;
  updates[0].actual_output = iree_string_view_empty();
  updates[0].expected_start = file.cases[0].expected.data;
  updates[0].expected_end =
      file.cases[0].expected.data + file.cases[0].expected.size;

  iree_host_size_t update_count = 0;
  std::string result = Apply(file, updates, &update_count);

  EXPECT_EQ(update_count, 1u);
  // The old expected section should be gone, replaced by empty.
  EXPECT_EQ(result.find("old"), std::string::npos);
  // The separator should still be there from the input part.
  EXPECT_NE(result.find("// ----\n"), std::string::npos);
}

TEST_F(UpdateTest, ReplacementPreservesBlankLinesAroundSeparators) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @a() {\n"
      "}\n"
      "\n"
      "// ----\n"
      "old a\n"
      "\n"
      "// ====\n"
      "\n"
      "func.def @b() {\n"
      "}\n");

  ASSERT_EQ(file.case_count, 2u);

  loom_check_case_update_t updates[2] = {};
  updates[0].needs_update = true;
  updates[0].actual_output = iree_make_cstring_view("new a\n");
  updates[0].expected_start = file.cases[0].expected.data;
  updates[0].expected_end =
      file.cases[0].expected.data + file.cases[0].expected.size;

  iree_host_size_t update_count = 0;
  std::string result = Apply(file, updates, &update_count);

  EXPECT_EQ(update_count, 1u);
  // The blank line between output and // ==== should be preserved.
  EXPECT_NE(result.find("new a\n\n// ====\n"), std::string::npos);
  // Blank line after // ==== should be preserved.
  EXPECT_NE(result.find("// ====\n\nfunc.def @b()"), std::string::npos);
}

}  // namespace
