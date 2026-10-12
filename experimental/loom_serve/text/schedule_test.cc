// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/schedule.h"

#include <vector>

#include "experimental/loom_serve/scheduling/packing.h"
#include "iree/testing/gtest.h"

namespace {

TEST(TextScheduleTest, DefaultCatalogCoversTheCapacityFamily) {
  std::vector<iree_host_size_t> capacities;
  for (iree_host_size_t tokens = 1; tokens <= 512; ++tokens) {
    capacities.push_back(tokens);
  }
  // Exercise both sides of every larger power transition and the terminal
  // context capacity without turning catalog construction into a large run.
  for (iree_host_size_t tokens = 512; tokens <= 262144; tokens *= 2) {
    capacities.push_back(tokens - 1);
    capacities.push_back(tokens);
    if (tokens < 262144) {
      capacities.push_back(tokens + 1);
    }
  }
  for (iree_host_size_t rows = 1; rows <= 16; ++rows) {
    for (iree_host_size_t tokens : capacities) {
      SCOPED_TRACE(rows);
      SCOPED_TRACE(tokens);
      loom_serve_packing_shape_t
          shapes[LOOM_SERVE_TEXT_DEFAULT_SHAPE_CAPACITY + 1] = {};
      const iree_host_size_t count =
          loom_serve_text_default_shapes(rows, tokens, shapes);
      ASSERT_GT(count, 0);
      ASSERT_LE(count, LOOM_SERVE_TEXT_DEFAULT_SHAPE_CAPACITY);
      EXPECT_EQ(shapes[count].token_capacity, 0);
      EXPECT_EQ(shapes[count].span_capacity, 0);
      EXPECT_EQ(shapes[0].token_capacity, iree_min(4, tokens));
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

TEST(TextScheduleTest, DefaultCatalogShrinksBothAxesWithLiveReadiness) {
  loom_serve_packing_shape_t shapes[LOOM_SERVE_TEXT_DEFAULT_SHAPE_CAPACITY];
  const iree_host_size_t shape_count =
      loom_serve_text_default_shapes(16, 512, shapes);
  ASSERT_EQ(shape_count, 37);
  loom_serve_packed_span_t spans[16], scratch[16];
  for (iree_host_size_t active = 1; active <= 16; ++active) {
    for (iree_host_size_t minimum = 1; minimum <= 8; ++minimum) {
      loom_serve_ready_span_t ready[16] = {};
      for (iree_host_size_t row = 16 - active; row < 16; ++row) {
        ready[row] = {minimum, minimum};
      }
      iree_host_size_t cursor = 0, shape = 0, expected_spans = 1;
      while (expected_spans < active) {
        expected_spans *= 2;
      }
      ASSERT_EQ(loom_serve_pack_shapes(16, ready, shape_count, shapes, 512,
                                       &cursor, spans, scratch, &shape),
                active);
      iree_host_size_t expected_tokens = 4;
      while (expected_tokens < active * minimum) {
        expected_tokens *= 2;
      }
      EXPECT_EQ(shapes[shape].token_capacity, expected_tokens);
      EXPECT_EQ(shapes[shape].span_capacity, expected_spans);
      for (iree_host_size_t i = 0; i < active; ++i) {
        EXPECT_EQ(spans[i].row_index, 16 - active + i);
        EXPECT_EQ(spans[i].token_count, minimum);
      }
    }
  }
  loom_serve_ready_span_t ready[16];
  for (auto& row : ready) {
    row = {4, 4};
  }
  ready[15] = {1000, 1};
  iree_host_size_t cursor = 0, shape = 0;
  ASSERT_EQ(loom_serve_pack_shapes(16, ready, shape_count, shapes, 512, &cursor,
                                   spans, scratch, &shape),
            16);
  EXPECT_EQ(shapes[shape].token_capacity, 512);
  EXPECT_EQ(shapes[shape].span_capacity, 16);
  EXPECT_EQ(spans[15].token_count, 452);
}

TEST(TextScheduleTest, CompletionReservationCoversLegalWriteHighWater) {
  for (iree_host_size_t context = 1; context <= 260; ++context) {
    for (iree_host_size_t input = 1; input <= context; ++input) {
      for (iree_host_size_t outputs = 1;
           outputs <= iree_min(17, context - input + 1); ++outputs) {
        for (iree_host_size_t depth = 0; depth <= 7; ++depth) {
          // Every selected count is reachable by accepting only the anchor.
          // Enumerate the service's actual launch rule instead of reproducing
          // the closed-form reservation calculation.
          iree_host_size_t high_water = input;
          for (iree_host_size_t selected = 1; selected < outputs; ++selected) {
            const iree_host_size_t position = input + selected - 1;
            const bool verify = depth && outputs - selected > 1 &&
                                context - position >= depth + 1;
            high_water =
                iree_max(high_water, position + (verify ? depth + 1 : 1));
          }
          const iree_host_size_t expected = ((high_water + 63) / 64) * 64;
          ASSERT_EQ(loom_serve_text_request_reservation(context, input, outputs,
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
