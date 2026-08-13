// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/source_trivia.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

static iree_status_t AcceptDiagnostic(void* user_data,
                                      const loom_diagnostic_t* diagnostic) {
  (void)user_data;
  (void)diagnostic;
  return iree_ok_status();
}

class BytecodeSourceTriviaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    loom_bytecode_reader_decoder_initialize(
        loom_diagnostic_sink_t{AcceptDiagnostic, NULL},
        IREE_SV("source_trivia_test.loombc"), &error_count_, &decoder_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_bytecode_reader_decoder_t decoder_ = {};
  uint32_t error_count_ = 0;
  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(BytecodeSourceTriviaTest, ValidationAdvancesWithoutAllocation) {
  const uint8_t data[] = {0x05, 0x03, 'o', 'n', 'e', 0x03, 't', 'w', 'o', 0xA5};
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(data, sizeof(data), 4096,
                                         IREE_SV("TEST"), &cursor);

  IREE_ASSERT_OK(loom_bytecode_source_trivia_validate(&decoder_, &cursor));

  EXPECT_EQ(cursor.cursor.position, sizeof(data) - 1);
  EXPECT_EQ(data[cursor.cursor.position], 0xA5u);
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeSourceTriviaTest, MaterializationPublishesConcreteTrivia) {
  const uint8_t data[] = {0x05, 0x03, 'o', 'n', 'e', 0x03, 't', 'w', 'o'};
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(data, sizeof(data), 8192,
                                         IREE_SV("TEST"), &cursor);

  loom_bytecode_source_trivia_t source_trivia = {};
  IREE_ASSERT_OK(loom_bytecode_source_trivia_materialize(
      &decoder_, &cursor, &arena_, &source_trivia));

  EXPECT_TRUE(source_trivia.leading_blank_line);
  ASSERT_EQ(source_trivia.comment_count, 2u);
  EXPECT_TRUE(
      iree_string_view_equal(source_trivia.comments[0], IREE_SV("one")));
  EXPECT_TRUE(
      iree_string_view_equal(source_trivia.comments[1], IREE_SV("two")));
  EXPECT_EQ(cursor.cursor.position, sizeof(data));
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeSourceTriviaTest, MaterializationNormalizesCommentPrefix) {
  const uint8_t data[] = {0x02, 0x04, ' ', 'o', 'n', 'e'};
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(data, sizeof(data), 10240,
                                         IREE_SV("TEST"), &cursor);

  loom_bytecode_source_trivia_t source_trivia = {};
  IREE_ASSERT_OK(loom_bytecode_source_trivia_materialize(
      &decoder_, &cursor, &arena_, &source_trivia));

  ASSERT_EQ(source_trivia.comment_count, 1u);
  EXPECT_TRUE(
      iree_string_view_equal(source_trivia.comments[0], IREE_SV("one")));
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeSourceTriviaTest, InvalidUtf8FailsValidation) {
  const uint8_t data[] = {0x02, 0x01, 0xFF};
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(data, sizeof(data), 11264,
                                         IREE_SV("TEST"), &cursor);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEFERRED,
      loom_bytecode_source_trivia_validate(&decoder_, &cursor));

  EXPECT_EQ(error_count_, 1u);
}

TEST_F(BytecodeSourceTriviaTest, TruncatedCommentFailsThroughDecoder) {
  const uint8_t data[] = {0x02, 0x04, 'o', 'n', 'e'};
  loom_bytecode_reader_cursor_t cursor;
  loom_bytecode_reader_cursor_initialize(data, sizeof(data), 12288,
                                         IREE_SV("TEST"), &cursor);

  loom_bytecode_source_trivia_t source_trivia = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED,
                        loom_bytecode_source_trivia_materialize(
                            &decoder_, &cursor, &arena_, &source_trivia));

  EXPECT_EQ(error_count_, 1u);
  EXPECT_EQ(source_trivia.comment_count, 0u);
  EXPECT_EQ(source_trivia.comments, nullptr);
}

}  // namespace
}  // namespace loom
