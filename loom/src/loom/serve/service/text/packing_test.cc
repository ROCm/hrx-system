// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/packing.h"

#include "iree/testing/gtest.h"

namespace {

TEST(PackingTest, CallerOwnsRowCapacityAndIndivisibleSpanLengths) {
  loom_serve_ready_span_t ready[40];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(ready); ++i) {
    const iree_host_size_t count = i % 2 ? 8 : 1;
    ready[i] = {count, count};
  }
  const loom_serve_packing_shape_t shapes[] = {{64, 16}, {128, 32}, {256, 40}};
  loom_serve_packed_span_t spans[40], scratch[40];
  iree_host_size_t cursor = 0, shape = 0;
  ASSERT_EQ(loom_serve_pack_shapes(IREE_ARRAYSIZE(ready), ready,
                                   IREE_ARRAYSIZE(shapes), shapes, 256, &cursor,
                                   spans, scratch, &shape),
            IREE_ARRAYSIZE(ready));
  EXPECT_EQ(shape, 2);
  EXPECT_EQ(cursor, 1);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(ready); ++i) {
    EXPECT_EQ(spans[i].row_index, i);
    EXPECT_EQ(spans[i].token_count, ready[i].token_count);
  }
}

TEST(PackingTest, PromptChunksFillAroundDecodeAndShortTails) {
  const loom_serve_ready_span_t ready[] = {{80, 1}, {1, 1},  {3, 1},
                                           {1, 1},  {50, 1}, {0, 0}};
  loom_serve_packed_span_t spans[6];
  iree_host_size_t cursor = 0;
  ASSERT_EQ(loom_serve_pack_spans(6, ready, 64, 6, 32, &cursor, spans), 5);
  const iree_host_size_t expected[] = {32, 1, 3, 1, 27};
  for (iree_host_size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(spans[i].row_index, i);
    EXPECT_EQ(spans[i].token_count, expected[i]);
  }
  EXPECT_EQ(cursor, 1);
}

TEST(PackingTest, CapacityPressureRotatesWithoutStarvation) {
  const loom_serve_packing_shape_t shapes[] = {{2, 8}, {8, 2}};
  for (uint32_t mask = 1; mask < 256; ++mask) {
    SCOPED_TRACE(mask);
    loom_serve_ready_span_t ready[8] = {};
    iree_host_size_t ready_count = 0;
    for (iree_host_size_t row = 0; row < 8; ++row) {
      if (mask & (1u << row)) {
        ready[row] = {100, 1};
        ++ready_count;
      }
    }
    for (const auto& shape : shapes) {
      SCOPED_TRACE(shape.token_capacity);
      for (iree_host_size_t start = 0; start < 8; ++start) {
        SCOPED_TRACE(start);
        loom_serve_packed_span_t spans[8];
        iree_host_size_t cursor = start;
        iree_host_size_t visits[8] = {};
        const iree_host_size_t expected_count = iree_min(ready_count, 2);
        for (iree_host_size_t epoch = 0; epoch < ready_count; ++epoch) {
          ASSERT_EQ(
              loom_serve_pack_spans(8, ready, shape.token_capacity,
                                    shape.span_capacity, 8, &cursor, spans),
              expected_count);
          for (iree_host_size_t i = 0; i < expected_count; ++i) {
            ++visits[spans[i].row_index];
          }
        }
        for (iree_host_size_t row = 0; row < 8; ++row) {
          EXPECT_EQ(visits[row], (mask & (1u << row)) ? expected_count : 0);
        }
      }
    }
  }
}

TEST(PackingTest, PromptExpansionRotatesWhenAllRowsFit) {
  const loom_serve_ready_span_t ready[] = {{100, 1}, {100, 1}, {100, 1}};
  loom_serve_packed_span_t spans[3];
  iree_host_size_t cursor = 0;
  for (iree_host_size_t epoch = 0; epoch < 3; ++epoch) {
    ASSERT_EQ(loom_serve_pack_spans(3, ready, 8, 3, 8, &cursor, spans), 3);
    EXPECT_EQ(spans[0].row_index, epoch);
    EXPECT_EQ(spans[0].token_count, 6);
    EXPECT_EQ(spans[1].token_count, 1);
    EXPECT_EQ(spans[2].token_count, 1);
  }
}

TEST(PackingTest, PromptExpansionRotatesAcrossEveryReadyRowLayout) {
  for (uint32_t mask = 1; mask < 256; ++mask) {
    SCOPED_TRACE(mask);
    loom_serve_ready_span_t ready[8] = {};
    iree_host_size_t ready_count = 0;
    for (iree_host_size_t row = 0; row < 8; ++row) {
      if (mask & (1u << row)) {
        ready[row] = {64, 1};
        ++ready_count;
      }
    }
    for (iree_host_size_t start = 0; start < 8; ++start) {
      SCOPED_TRACE(start);
      iree_host_size_t cursor = start;
      iree_host_size_t expanded[8] = {};
      loom_serve_packed_span_t spans[8];
      for (iree_host_size_t epoch = 0; epoch < ready_count; ++epoch) {
        ASSERT_EQ(loom_serve_pack_spans(8, ready, 16, ready_count, 16, &cursor,
                                        spans),
                  ready_count);
        for (iree_host_size_t i = 0; i < ready_count; ++i) {
          expanded[spans[i].row_index] += spans[i].token_count > 1;
        }
      }
      for (iree_host_size_t row = 0; row < 8; ++row) {
        EXPECT_EQ(expanded[row], (mask & (1u << row)) ? 1 : 0);
      }
    }
  }
}

TEST(PackingTest, DeferredVerifierGetsNextPriorityAcrossInactiveRows) {
  const loom_serve_ready_span_t ready[] = {{4, 4}, {4, 4}, {30, 1}, {0, 0}};
  loom_serve_packed_span_t spans[4];
  iree_host_size_t cursor = 0;
  iree_host_size_t visits[4] = {};
  for (iree_host_size_t epoch = 0; epoch < 2; ++epoch) {
    ASSERT_EQ(loom_serve_pack_spans(4, ready, 5, 4, 5, &cursor, spans), 2);
    for (iree_host_size_t i = 0; i < 2; ++i) {
      EXPECT_EQ(spans[i].token_count, ready[spans[i].row_index].minimum_count);
      ++visits[spans[i].row_index];
    }
  }
  EXPECT_EQ(visits[0], 1);
  EXPECT_EQ(visits[1], 1);
  EXPECT_EQ(visits[2], 2);
  EXPECT_EQ(visits[3], 0);
}

TEST(PackingTest, IneligibleMinimumDoesNotPinExpansionPriority) {
  const loom_serve_ready_span_t ready[] = {{16, 16}, {32, 1}, {32, 1}, {0, 0}};
  loom_serve_packed_span_t spans[4];
  iree_host_size_t cursor = 0;
  for (iree_host_size_t row : {1u, 2u}) {
    ASSERT_EQ(loom_serve_pack_spans(4, ready, 8, 2, 8, &cursor, spans), 2);
    EXPECT_EQ(spans[0].row_index, row);
    EXPECT_EQ(spans[0].token_count, 7);
    EXPECT_EQ(spans[1].token_count, 1);
  }
}

TEST(PackingTest, PausedRowsConsumeNeitherTokensNorSpanSlots) {
  const loom_serve_ready_span_t ready[] = {{0, 0}, {1, 1}, {0, 0},
                                           {7, 1}, {0, 0}, {2, 1}};
  loom_serve_packed_span_t spans[6];
  iree_host_size_t cursor = 4;
  ASSERT_EQ(loom_serve_pack_spans(6, ready, 16, 3, 16, &cursor, spans), 3);
  EXPECT_EQ(spans[0].row_index, 5);
  EXPECT_EQ(spans[0].token_count, 2);
  EXPECT_EQ(spans[1].row_index, 1);
  EXPECT_EQ(spans[1].token_count, 1);
  EXPECT_EQ(spans[2].row_index, 3);
  EXPECT_EQ(spans[2].token_count, 7);
}

TEST(PackingTest, EmptyReadinessDoesNotAdvanceCursor) {
  const loom_serve_ready_span_t ready[3] = {};
  loom_serve_packed_span_t spans[3];
  iree_host_size_t cursor = 2;
  EXPECT_EQ(loom_serve_pack_spans(3, ready, 8, 3, 8, &cursor, spans), 0);
  EXPECT_EQ(cursor, 2);
}

TEST(PackingTest, ExhaustsFiniteMixedWorkExactlyOnce) {
  loom_serve_ready_span_t ready[] = {{71, 1}, {1, 1},  {3, 1}, {256, 1},
                                     {1, 1},  {23, 1}, {1, 1}, {91, 1}};
  const iree_host_size_t expected[] = {71, 1, 3, 256, 1, 23, 1, 91};
  iree_host_size_t consumed[8] = {};
  iree_host_size_t cursor = 0;
  loom_serve_packed_span_t spans[8];
  iree_host_size_t count = 0;
  while ((count = loom_serve_pack_spans(8, ready, 32, 4, 24, &cursor, spans))) {
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

TEST(PackingTest, CachedShapesShrinkWithoutLosingReadyWork) {
  const loom_serve_packing_shape_t shapes[] = {{512, 8}, {32, 8}, {128, 8}};
  loom_serve_ready_span_t ready[] = {{500, 1}, {1, 1}, {1, 1}, {1, 1},
                                     {1, 1},   {1, 1}, {1, 1}, {1, 1}};
  loom_serve_packed_span_t spans[8], scratch[8];
  iree_host_size_t cursor = 0, shape = 0;
  ASSERT_EQ(loom_serve_pack_shapes(8, ready, 3, shapes, 512, &cursor, spans,
                                   scratch, &shape),
            8);
  EXPECT_EQ(shape, 0);
  EXPECT_EQ(spans[0].token_count, 500);
  EXPECT_EQ(cursor, 1);
  ready[0].token_count = 70;
  ASSERT_EQ(loom_serve_pack_shapes(8, ready, 3, shapes, 512, &cursor, spans,
                                   scratch, &shape),
            8);
  EXPECT_EQ(shape, 2);
  EXPECT_EQ(cursor, 2);
  ready[0].token_count = 1;
  ASSERT_EQ(loom_serve_pack_shapes(8, ready, 3, shapes, 512, &cursor, spans,
                                   scratch, &shape),
            8);
  EXPECT_EQ(shape, 1);
  EXPECT_EQ(cursor, 3);
  for (const auto& span : spans) {
    EXPECT_EQ(span.token_count, 1);
  }
}

TEST(PackingTest, ShapePlanningKeepsTokenAndSpanConstraintsIndependent) {
  const loom_serve_packing_shape_t shapes[] = {{512, 1}, {128, 8}, {128, 4}};
  const loom_serve_ready_span_t ready[] = {{1, 1}, {1, 1}, {1, 1}, {1, 1},
                                           {0, 0}, {0, 0}, {0, 0}, {0, 0}};
  loom_serve_packed_span_t spans[8], scratch[8];
  iree_host_size_t cursor = 0, shape = 0;
  ASSERT_EQ(loom_serve_pack_shapes(8, ready, 3, shapes, 512, &cursor, spans,
                                   scratch, &shape),
            4);
  EXPECT_EQ(shape, 2);
  const iree_host_size_t saved_cursor = cursor;
  const loom_serve_ready_span_t empty[8] = {};
  EXPECT_EQ(loom_serve_pack_shapes(8, empty, 3, shapes, 512, &cursor, spans,
                                   scratch, &shape),
            0);
  EXPECT_EQ(cursor, saved_cursor);
}

TEST(PackingTest, ShapePlanningHonorsChunkLimitBeforeSizing) {
  const loom_serve_packing_shape_t shapes[] = {{512, 8}, {128, 8}, {32, 8}};
  const loom_serve_ready_span_t ready[] = {{1000, 1}, {1, 1}, {1, 1}, {1, 1}};
  loom_serve_packed_span_t spans[4], scratch[4];
  iree_host_size_t cursor = 0, shape = 0;
  EXPECT_EQ(loom_serve_pack_shapes(4, ready, 3, shapes, 64, &cursor, spans,
                                   scratch, &shape),
            4);
  EXPECT_EQ(shape, 1);
  EXPECT_EQ(spans[0].token_count, 64);
}

TEST(PackingTest, EquallyOccupiedShapesKeepReadyPeers) {
  const loom_serve_packing_shape_t catalogs[][2] = {{{512, 1}, {512, 4}},
                                                    {{512, 4}, {512, 1}}};
  for (iree_host_size_t minimum : {1u, 4u}) {
    SCOPED_TRACE(minimum);
    const loom_serve_ready_span_t ready[] = {
        {512, 1}, {minimum, minimum}, {1, 1}, {minimum, minimum}};
    for (const auto& shapes : catalogs) {
      SCOPED_TRACE(shapes[0].span_capacity);
      loom_serve_packed_span_t spans[4], scratch[4];
      iree_host_size_t cursor = 0, shape = 0;
      const iree_host_size_t count = loom_serve_pack_shapes(
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

TEST(PackingTest, ReservesWholeVerifiersBeforePromptExpansion) {
  const loom_serve_ready_span_t ready[] = {{80, 1}, {4, 4}, {3, 1}, {4, 4},
                                           {50, 1}, {4, 4}, {1, 1}, {4, 4}};
  const loom_serve_packing_shape_t shapes[] = {{128, 8}, {32, 8}};
  loom_serve_packed_span_t spans[8], scratch[8];
  iree_host_size_t cursor = 0, shape = 0;
  ASSERT_EQ(loom_serve_pack_spans(8, ready, 32, 8, 7, &cursor, spans), 8);
  const iree_host_size_t expected[] = {7, 4, 3, 4, 5, 4, 1, 4};
  for (iree_host_size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(spans[i].row_index, i);
    EXPECT_EQ(spans[i].token_count, expected[i]);
  }
  cursor = 0;
  ASSERT_EQ(loom_serve_pack_shapes(8, ready, 2, shapes, 6, &cursor, spans,
                                   scratch, &shape),
            8);
  EXPECT_EQ(shape, 1);
  EXPECT_EQ(spans[0].token_count, 6);
  EXPECT_EQ(spans[4].token_count, 6);
  cursor = 0;
  ASSERT_EQ(loom_serve_pack_spans(8, ready, 32, 8, 1, &cursor, spans), 8);
  for (iree_host_size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(spans[i].token_count, ready[i].minimum_count);
  }
}

TEST(PackingTest, FillsGapsWithoutSplittingOrStarvingVerifiers) {
  const loom_serve_ready_span_t ready[] = {{4, 4}, {4, 4}, {30, 1}};
  loom_serve_packed_span_t spans[3];
  iree_host_size_t cursor = 0;
  iree_host_size_t visits[3] = {};
  for (int epoch = 0; epoch < 3; ++epoch) {
    ASSERT_EQ(loom_serve_pack_spans(3, ready, 6, 3, 6, &cursor, spans), 2);
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

}  // namespace
