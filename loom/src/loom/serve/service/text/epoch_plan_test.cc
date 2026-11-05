// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/epoch_plan.h"

#include "iree/testing/gtest.h"
#include "loom/serve/service/text/state.h"

namespace {

TEST(TextEpochPlanTest, KnownAndSpeculativeContinuationsPreserveSourceFacts) {
  loom_serve_text_state_row_t records[2] = {};
  // Directory identity, not physical record order, selects each retained state.
  loom_serve_text_state_row_t* rows[] = {&records[1], &records[0]};
  rows[0]->position = 7;
  rows[0]->source_position = 31;
  rows[1]->position = 9;
  rows[1]->source_position = 45;
  uint8_t plan[4 * TEXT_PLAN_FIELD_COUNT * 4] = {};
  uint8_t input[16 * 4] = {};
  const auto plan_bytes = iree_make_byte_span(plan, sizeof(plan));
  const auto input_bytes = iree_make_byte_span(input, sizeof(input));
  const int32_t tokens[] = {10, 11, 12, 13};
  const loom_serve_text_span_t first[] = {
      {1, 2, tokens, 0},
      {0, 4, tokens + 2,
       LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT |
           LOOM_SERVICE_TEXT_SPAN_FLAG_PROPOSE},
  };
  const uint32_t limits[] = {0, 8};
  EXPECT_EQ(loom_serve_text_epoch_plan_write(rows, plan_bytes, input_bytes, 2,
                                             first, limits, nullptr, 0, 0),
            3u);
  auto field = [&](size_t record, size_t index) {
    return iree_unaligned_load_le_u32(
        plan + (record * TEXT_PLAN_FIELD_COUNT + index) * 4);
  };
  EXPECT_EQ(field(0, TEXT_PLAN_FIELD_ROW), 1u);
  EXPECT_EQ(field(0, TEXT_PLAN_FIELD_POSITION), 9u);
  EXPECT_EQ(field(0, TEXT_PLAN_FIELD_SOURCE_POSITION), 45u);
  EXPECT_EQ(field(0, TEXT_PLAN_FIELD_INPUT_COUNT), 2u);
  EXPECT_EQ(field(1, TEXT_PLAN_FIELD_INPUT_BEGIN), 2u);
  EXPECT_EQ(field(1, TEXT_PLAN_FIELD_INPUT_COUNT), 1u);
  EXPECT_EQ(field(1, TEXT_PLAN_FIELD_SOURCE_POSITION), 31u);
  EXPECT_EQ(field(1, TEXT_PLAN_FIELD_OUTPUT_LIMIT), 8u);
  const loom_serve_text_span_t next[] = {
      {0, 4, nullptr,
       LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT |
           LOOM_SERVICE_TEXT_SPAN_FLAG_PROPOSE},
      {1, 1, tokens + 3, LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT},
  };
  const uint32_t next_limits[] = {8, 0};
  const uint32_t first_indices[] = {1, 0};
  EXPECT_EQ(
      loom_serve_text_epoch_plan_write(rows, plan_bytes, input_bytes, 2, next,
                                       next_limits, first_indices, 2, 3),
      4u);
  EXPECT_EQ(field(2, TEXT_PLAN_FIELD_FIRST_SPAN), 1u);
  EXPECT_EQ(field(2, TEXT_PLAN_FIELD_INPUT_COUNT), 0u);
  EXPECT_EQ(field(3, TEXT_PLAN_FIELD_FIRST_SPAN), 0u);
  EXPECT_EQ(field(3, TEXT_PLAN_FIELD_INPUT_BEGIN), 3u);
  EXPECT_EQ(field(3, TEXT_PLAN_FIELD_SOURCE_POSITION), 45u);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(iree_unaligned_load_le_u32(input + i * 4), tokens[i]);
  }
}

TEST(TextEpochPlanTest, PublicationKeepsOpaquePositionSeparateFromCounts) {
  uint8_t record[TEXT_PROGRESS_FIELD_COUNT * 4] = {};
  const uint32_t counts[] = {6, 2, 4, 1, 47};
  for (size_t i = 0; i < IREE_ARRAYSIZE(counts); ++i) {
    iree_unaligned_store_le_u32(record + i * 4, counts[i]);
  }
  for (size_t i = 0; i < 4; ++i) {
    iree_unaligned_store_le_u32(record + (TEXT_PROGRESS_FIELD_TOKENS + i) * 4,
                                10 + i);
  }
  loom_serve_text_result_t result;
  uint32_t source_position;
  loom_serve_text_epoch_plan_read_progress(record, &result, &source_position);
  EXPECT_EQ(result.consumed_count, 6u);
  EXPECT_EQ(result.known_count, 2u);
  EXPECT_EQ(result.output_count, 4u);
  EXPECT_EQ(result.verification_count, 1u);
  EXPECT_EQ(source_position, 47u);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(result.tokens[i], 10 + i);
  }
}

}  // namespace
