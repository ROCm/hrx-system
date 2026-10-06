// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/scheduling/packing.h"

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
  const loom_serve_ready_span_t ready[] = {{100, 1}, {1, 1}, {100, 1}, {1, 1},
                                           {100, 1}, {1, 1}, {100, 1}, {1, 1}};
  for (iree_host_size_t span_capacity : {2u, 8u}) {
    loom_serve_packed_span_t spans[8];
    iree_host_size_t cursor = 0;
    iree_host_size_t visits[8] = {};
    for (int epoch = 0; epoch < 4; ++epoch) {
      ASSERT_EQ(
          loom_serve_pack_spans(8, ready, 2, span_capacity, 32, &cursor, spans),
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
  EXPECT_EQ(cursor, 4);
  const loom_serve_ready_span_t empty[8] = {};
  EXPECT_EQ(loom_serve_pack_shapes(8, empty, 3, shapes, 512, &cursor, spans,
                                   scratch, &shape),
            0);
  EXPECT_EQ(cursor, 4);
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
