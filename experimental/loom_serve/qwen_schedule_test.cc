// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/qwen_schedule.h"

#include "iree/testing/gtest.h"

namespace {

TEST(QwenScheduleTest, DefaultCatalogCoversTheCapacityFamily) {
  for (iree_host_size_t rows = 1; rows <= 16; ++rows) {
    for (iree_host_size_t tokens = 1; tokens <= 512; ++tokens) {
      loom_serve_qwen_shape_t
          shapes[LOOM_SERVE_QWEN_DEFAULT_SHAPE_CAPACITY + 1] = {};
      const iree_host_size_t count =
          loom_serve_qwen_default_shapes(rows, tokens, shapes);
      ASSERT_GT(count, 0);
      ASSERT_LE(count, LOOM_SERVE_QWEN_DEFAULT_SHAPE_CAPACITY);
      EXPECT_EQ(shapes[count].token_capacity, 0);
      EXPECT_EQ(shapes[count].span_capacity, 0);
      EXPECT_EQ(shapes[0].token_capacity, iree_min(32, tokens));
      EXPECT_EQ(shapes[0].span_capacity, 1);
      EXPECT_EQ(shapes[count - 1].token_capacity, tokens);
      if (rows <= tokens) {
        EXPECT_EQ(shapes[count - 1].span_capacity, rows);
      }
      for (iree_host_size_t i = 0; i < count; ++i) {
        const auto shape = shapes[i];
        EXPECT_LE(shape.token_capacity, tokens);
        EXPECT_LE(shape.span_capacity, rows);
        EXPECT_LE(shape.span_capacity, shape.token_capacity);
        if (i) {
          const auto previous = shapes[i - 1];
          if (previous.token_capacity == shape.token_capacity) {
            EXPECT_EQ(shape.span_capacity,
                      iree_min(previous.span_capacity * 2, rows));
          } else {
            EXPECT_EQ(shape.token_capacity,
                      iree_min(previous.token_capacity * 2, tokens));
            EXPECT_EQ(shape.span_capacity, 1);
          }
        }
      }
    }
  }
}

TEST(QwenScheduleTest, DefaultCatalogShrinksBothAxesWithLiveReadiness) {
  loom_serve_qwen_shape_t shapes[LOOM_SERVE_QWEN_DEFAULT_SHAPE_CAPACITY];
  const iree_host_size_t shape_count =
      loom_serve_qwen_default_shapes(16, 512, shapes);
  ASSERT_EQ(shape_count, 25);
  loom_serve_qwen_scheduled_span_t spans[16], scratch[16];
  for (iree_host_size_t active = 1; active <= 16; ++active) {
    for (iree_host_size_t minimum : {1u, 4u}) {
      loom_serve_qwen_ready_span_t ready[16] = {};
      for (iree_host_size_t row = 16 - active; row < 16; ++row) {
        ready[row] = {minimum, minimum};
      }
      iree_host_size_t cursor = 0, shape = 0, expected_spans = 1;
      while (expected_spans < active) {
        expected_spans *= 2;
      }
      ASSERT_EQ(
          loom_serve_qwen_schedule_shapes(16, ready, shape_count, shapes, 512,
                                          &cursor, spans, scratch, &shape),
          active);
      EXPECT_EQ(shapes[shape].token_capacity, active * minimum > 32 ? 64 : 32);
      EXPECT_EQ(shapes[shape].span_capacity, expected_spans);
      for (iree_host_size_t i = 0; i < active; ++i) {
        EXPECT_EQ(spans[i].row_index, 16 - active + i);
        EXPECT_EQ(spans[i].token_count, minimum);
      }
    }
  }
  loom_serve_qwen_ready_span_t ready[16];
  for (auto& row : ready) {
    row = {4, 4};
  }
  ready[15] = {1000, 1};
  iree_host_size_t cursor = 0, shape = 0;
  ASSERT_EQ(loom_serve_qwen_schedule_shapes(16, ready, shape_count, shapes, 512,
                                            &cursor, spans, scratch, &shape),
            16);
  EXPECT_EQ(shapes[shape].token_capacity, 512);
  EXPECT_EQ(shapes[shape].span_capacity, 16);
  EXPECT_EQ(spans[15].token_count, 452);
}

TEST(QwenScheduleTest, PromptChunksFillAroundDecodeAndShortTails) {
  const loom_serve_qwen_ready_span_t ready[] = {{80, 1}, {1, 1},  {3, 1},
                                                {1, 1},  {50, 1}, {0, 0}};
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
  const loom_serve_qwen_ready_span_t ready[] = {
      {100, 1}, {1, 1}, {100, 1}, {1, 1}, {100, 1}, {1, 1}, {100, 1}, {1, 1}};
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
  const loom_serve_qwen_ready_span_t ready[] = {{100, 1}, {100, 1}, {100, 1}};
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
  const loom_serve_qwen_ready_span_t ready[] = {{0, 0}, {1, 1}, {0, 0},
                                                {7, 1}, {0, 0}, {2, 1}};
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
  const loom_serve_qwen_ready_span_t ready[3] = {};
  loom_serve_qwen_scheduled_span_t spans[3];
  iree_host_size_t cursor = 2;
  EXPECT_EQ(loom_serve_qwen_schedule(3, ready, 8, 3, 8, &cursor, spans), 0);
  EXPECT_EQ(cursor, 2);
}

TEST(QwenScheduleTest, ExhaustsFiniteMixedWorkExactlyOnce) {
  loom_serve_qwen_ready_span_t ready[] = {{71, 1}, {1, 1},  {3, 1}, {256, 1},
                                          {1, 1},  {23, 1}, {1, 1}, {91, 1}};
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
      ASSERT_LE(span.token_count, ready[span.row_index].token_count);
      EXPECT_LE(span.token_count, 24);
      ready[span.row_index].token_count -= span.token_count;
      consumed[span.row_index] += span.token_count;
      total += span.token_count;
    }
    EXPECT_LE(total, 32);
  }
  for (iree_host_size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(consumed[i], expected[i]);
  }
}

TEST(QwenScheduleTest, CachedShapesShrinkWithoutLosingReadyWork) {
  const loom_serve_qwen_shape_t shapes[] = {{512, 8}, {32, 8}, {128, 8}};
  loom_serve_qwen_ready_span_t ready[] = {{500, 1}, {1, 1}, {1, 1}, {1, 1},
                                          {1, 1},   {1, 1}, {1, 1}, {1, 1}};
  loom_serve_qwen_scheduled_span_t spans[8], scratch[8];
  iree_host_size_t cursor = 0, shape = 0;
  ASSERT_EQ(loom_serve_qwen_schedule_shapes(8, ready, 3, shapes, 512, &cursor,
                                            spans, scratch, &shape),
            8);
  EXPECT_EQ(shape, 0);
  EXPECT_EQ(spans[0].token_count, 500);
  EXPECT_EQ(cursor, 1);
  ready[0].token_count = 70;
  ASSERT_EQ(loom_serve_qwen_schedule_shapes(8, ready, 3, shapes, 512, &cursor,
                                            spans, scratch, &shape),
            8);
  EXPECT_EQ(shape, 2);
  EXPECT_EQ(cursor, 2);
  ready[0].token_count = 1;
  ASSERT_EQ(loom_serve_qwen_schedule_shapes(8, ready, 3, shapes, 512, &cursor,
                                            spans, scratch, &shape),
            8);
  EXPECT_EQ(shape, 1);
  EXPECT_EQ(cursor, 3);
  for (const auto& span : spans) {
    EXPECT_EQ(span.token_count, 1);
  }
}

TEST(QwenScheduleTest, ShapePlanningKeepsTokenAndSpanConstraintsIndependent) {
  const loom_serve_qwen_shape_t shapes[] = {{512, 1}, {128, 8}, {128, 4}};
  const loom_serve_qwen_ready_span_t ready[] = {{1, 1}, {1, 1}, {1, 1}, {1, 1},
                                                {0, 0}, {0, 0}, {0, 0}, {0, 0}};
  loom_serve_qwen_scheduled_span_t spans[8], scratch[8];
  iree_host_size_t cursor = 0, shape = 0;
  ASSERT_EQ(loom_serve_qwen_schedule_shapes(8, ready, 3, shapes, 512, &cursor,
                                            spans, scratch, &shape),
            4);
  EXPECT_EQ(shape, 2);
  EXPECT_EQ(cursor, 4);
  const loom_serve_qwen_ready_span_t empty[8] = {};
  EXPECT_EQ(loom_serve_qwen_schedule_shapes(8, empty, 3, shapes, 512, &cursor,
                                            spans, scratch, &shape),
            0);
  EXPECT_EQ(cursor, 4);
}

TEST(QwenScheduleTest, ShapePlanningHonorsChunkLimitBeforeSizing) {
  const loom_serve_qwen_shape_t shapes[] = {{512, 8}, {128, 8}, {32, 8}};
  const loom_serve_qwen_ready_span_t ready[] = {
      {1000, 1}, {1, 1}, {1, 1}, {1, 1}};
  loom_serve_qwen_scheduled_span_t spans[4], scratch[4];
  iree_host_size_t cursor = 0, shape = 0;
  EXPECT_EQ(loom_serve_qwen_schedule_shapes(4, ready, 3, shapes, 64, &cursor,
                                            spans, scratch, &shape),
            4);
  EXPECT_EQ(shape, 1);
  EXPECT_EQ(spans[0].token_count, 64);
}

TEST(QwenScheduleTest, EquallyOccupiedShapesKeepReadyPeers) {
  const loom_serve_qwen_shape_t catalogs[][2] = {{{512, 1}, {512, 4}},
                                                 {{512, 4}, {512, 1}}};
  for (iree_host_size_t minimum : {1u, 4u}) {
    SCOPED_TRACE(minimum);
    const loom_serve_qwen_ready_span_t ready[] = {
        {512, 1}, {minimum, minimum}, {1, 1}, {minimum, minimum}};
    for (const auto& shapes : catalogs) {
      SCOPED_TRACE(shapes[0].span_capacity);
      loom_serve_qwen_scheduled_span_t spans[4], scratch[4];
      iree_host_size_t cursor = 0, shape = 0;
      const iree_host_size_t count = loom_serve_qwen_schedule_shapes(
          4, ready, 2, shapes, 512, &cursor, spans, scratch, &shape);
      EXPECT_EQ(count, 4);
      EXPECT_EQ(shapes[shape].span_capacity, 4);
      EXPECT_EQ(cursor, 1);
      const iree_host_size_t expected[] = {512 - 2 * minimum - 1, minimum, 1,
                                           minimum};
      for (iree_host_size_t i = 0; i < count; ++i) {
        EXPECT_EQ(spans[i].row_index, i);
        EXPECT_EQ(spans[i].token_count, expected[i]);
      }
    }
  }
}

TEST(QwenScheduleTest, ReservesWholeVerifiersBeforePromptExpansion) {
  const loom_serve_qwen_ready_span_t ready[] = {
      {80, 1}, {4, 4}, {3, 1}, {4, 4}, {50, 1}, {4, 4}, {1, 1}, {4, 4}};
  const loom_serve_qwen_shape_t shapes[] = {{128, 8}, {32, 8}};
  loom_serve_qwen_scheduled_span_t spans[8], scratch[8];
  iree_host_size_t cursor = 0, shape = 0;
  ASSERT_EQ(loom_serve_qwen_schedule(8, ready, 32, 8, 7, &cursor, spans), 8);
  const iree_host_size_t expected[] = {7, 4, 3, 4, 5, 4, 1, 4};
  for (iree_host_size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(spans[i].row_index, i);
    EXPECT_EQ(spans[i].token_count, expected[i]);
  }
  cursor = 0;
  ASSERT_EQ(loom_serve_qwen_schedule_shapes(8, ready, 2, shapes, 6, &cursor,
                                            spans, scratch, &shape),
            8);
  EXPECT_EQ(shape, 1);
  EXPECT_EQ(spans[0].token_count, 6);
  EXPECT_EQ(spans[4].token_count, 6);
  cursor = 0;
  ASSERT_EQ(loom_serve_qwen_schedule(8, ready, 32, 8, 1, &cursor, spans), 8);
  for (iree_host_size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(spans[i].token_count, ready[i].minimum_count);
  }
}

TEST(QwenScheduleTest, FillsGapsWithoutSplittingOrStarvingVerifiers) {
  const loom_serve_qwen_ready_span_t ready[] = {{4, 4}, {4, 4}, {30, 1}};
  loom_serve_qwen_scheduled_span_t spans[3];
  iree_host_size_t cursor = 0;
  iree_host_size_t visits[3] = {};
  for (int epoch = 0; epoch < 3; ++epoch) {
    ASSERT_EQ(loom_serve_qwen_schedule(3, ready, 6, 3, 6, &cursor, spans), 2);
    iree_host_size_t tokens = 0;
    for (iree_host_size_t i = 0; i < 2; ++i) {
      const auto span = spans[i];
      EXPECT_EQ(span.token_count, span.row_index == 2 ? 2 : 4);
      tokens += span.token_count;
      ++visits[span.row_index];
    }
    EXPECT_EQ(tokens, 6);
  }
  for (auto count : visits) {
    EXPECT_GT(count, 0);
  }
}

TEST(QwenScheduleTest, CompletionReservationCoversLegalWriteHighWater) {
  for (iree_host_size_t context = 1; context <= 260; ++context) {
    for (iree_host_size_t input = 1; input <= context; ++input) {
      for (iree_host_size_t outputs = 1;
           outputs <= iree_min(17, context - input + 1); ++outputs) {
        for (iree_host_size_t depth : {0u, 3u}) {
          // Every selected count is reachable by accepting only the anchor.
          // Enumerate the service's actual launch rule instead of reproducing
          // the closed-form reservation calculation.
          iree_host_size_t high_water = input;
          for (iree_host_size_t selected = 1; selected < outputs; ++selected) {
            const iree_host_size_t position = input + selected - 1;
            const bool verify =
                depth && outputs - selected > 1 && context - position >= 4;
            high_water = iree_max(high_water, position + (verify ? 4 : 1));
          }
          const iree_host_size_t expected = ((high_water + 63) / 64) * 64;
          ASSERT_EQ(loom_serve_qwen_request_reservation(context, input, outputs,
                                                        depth, 64),
                    expected)
              << "context=" << context << " input=" << input
              << " outputs=" << outputs << " depth=" << depth;
        }
      }
    }
  }
}

}  // namespace
