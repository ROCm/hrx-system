// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/qwen_schedule.h"

#include "iree/testing/gtest.h"

namespace {

TEST(QwenScheduleTest, PromptChunksFillAroundDecodeAndShortTails) {
  const iree_host_size_t ready[] = {80, 1, 3, 1, 50, 0};
  loom_serve_qwen_scheduled_span_t spans[6];
  iree_host_size_t cursor = 0;
  ASSERT_EQ(loom_serve_qwen_schedule(6, ready, 64, 6, 32, &cursor, spans), 5);
  const iree_host_size_t expected[] = {32, 1, 3, 1, 27};
  for (iree_host_size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(spans[i].row_index, i);
    EXPECT_EQ(spans[i].token_count, expected[i]);
  }
  EXPECT_EQ(cursor, 1);
}

TEST(QwenScheduleTest, CapacityPressureRotatesWithoutStarvation) {
  const iree_host_size_t ready[] = {100, 1, 100, 1, 100, 1, 100, 1};
  for (iree_host_size_t span_capacity : {2u, 8u}) {
    loom_serve_qwen_scheduled_span_t spans[8];
    iree_host_size_t cursor = 0;
    iree_host_size_t visits[8] = {};
    for (int epoch = 0; epoch < 4; ++epoch) {
      ASSERT_EQ(loom_serve_qwen_schedule(8, ready, 2, span_capacity, 32,
                                         &cursor, spans),
                2);
      for (const auto& span : {spans[0], spans[1]}) {
        EXPECT_EQ(span.token_count, 1);
        ++visits[span.row_index];
      }
    }
    for (auto count : visits) {
      EXPECT_EQ(count, 1);
    }
  }
}

TEST(QwenScheduleTest, PromptExpansionRotatesWhenAllRowsFit) {
  const iree_host_size_t ready[] = {100, 100, 100};
  loom_serve_qwen_scheduled_span_t spans[3];
  iree_host_size_t cursor = 0;
  for (iree_host_size_t epoch = 0; epoch < 3; ++epoch) {
    ASSERT_EQ(loom_serve_qwen_schedule(3, ready, 8, 3, 8, &cursor, spans), 3);
    EXPECT_EQ(spans[0].row_index, epoch);
    EXPECT_EQ(spans[0].token_count, 6);
    EXPECT_EQ(spans[1].token_count, 1);
    EXPECT_EQ(spans[2].token_count, 1);
  }
}

TEST(QwenScheduleTest, PausedRowsConsumeNeitherTokensNorSpanSlots) {
  const iree_host_size_t ready[] = {0, 1, 0, 7, 0, 2};
  loom_serve_qwen_scheduled_span_t spans[6];
  iree_host_size_t cursor = 4;
  ASSERT_EQ(loom_serve_qwen_schedule(6, ready, 16, 3, 16, &cursor, spans), 3);
  EXPECT_EQ(spans[0].row_index, 5);
  EXPECT_EQ(spans[0].token_count, 2);
  EXPECT_EQ(spans[1].row_index, 1);
  EXPECT_EQ(spans[1].token_count, 1);
  EXPECT_EQ(spans[2].row_index, 3);
  EXPECT_EQ(spans[2].token_count, 7);
}

TEST(QwenScheduleTest, EmptyReadinessDoesNotAdvanceCursor) {
  const iree_host_size_t ready[] = {0, 0, 0};
  loom_serve_qwen_scheduled_span_t spans[3];
  iree_host_size_t cursor = 2;
  EXPECT_EQ(loom_serve_qwen_schedule(3, ready, 8, 3, 8, &cursor, spans), 0);
  EXPECT_EQ(cursor, 2);
}

TEST(QwenScheduleTest, ExhaustsFiniteMixedWorkExactlyOnce) {
  iree_host_size_t ready[] = {71, 1, 3, 256, 1, 23, 1, 91};
  const iree_host_size_t expected[] = {71, 1, 3, 256, 1, 23, 1, 91};
  iree_host_size_t consumed[8] = {};
  iree_host_size_t cursor = 0;
  loom_serve_qwen_scheduled_span_t spans[8];
  iree_host_size_t count = 0;
  while (
      (count = loom_serve_qwen_schedule(8, ready, 32, 4, 24, &cursor, spans))) {
    iree_host_size_t total = 0;
    uint32_t seen = 0;
    for (iree_host_size_t i = 0; i < count; ++i) {
      const auto span = spans[i];
      ASSERT_LT(span.row_index, 8);
      EXPECT_EQ(seen & (1u << span.row_index), 0);
      seen |= 1u << span.row_index;
      ASSERT_GT(span.token_count, 0);
      ASSERT_LE(span.token_count, ready[span.row_index]);
      EXPECT_LE(span.token_count, 24);
      ready[span.row_index] -= span.token_count;
      consumed[span.row_index] += span.token_count;
      total += span.token_count;
    }
    EXPECT_LE(total, 32);
  }
  for (iree_host_size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(consumed[i], expected[i]);
  }
}

}  // namespace
