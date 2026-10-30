// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/schedule.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "loom/serve/service/text/packing.h"

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
  capacities.push_back(INT32_MAX);
  for (iree_host_size_t tokens : capacities) {
    SCOPED_TRACE(tokens);
    const iree_host_size_t count =
        loom_serve_text_default_shapes(tokens, nullptr);
    std::vector<loom_serve_packing_shape_t> shapes(count + 1);
    EXPECT_EQ(loom_serve_text_default_shapes(tokens, shapes.data()), count);
    ASSERT_GT(count, 0);
    EXPECT_EQ(shapes[count].token_capacity, 0);
    EXPECT_EQ(shapes[count].span_capacity, 0);
    EXPECT_EQ(shapes[0].token_capacity, 1);
    EXPECT_EQ(shapes[0].span_capacity, 1);
    EXPECT_EQ(shapes[count - 1].token_capacity, tokens);
    EXPECT_EQ(shapes[count - 1].span_capacity, tokens);
    for (iree_host_size_t i = 0; i < count; ++i) {
      const auto shape = shapes[i];
      EXPECT_LE(shape.token_capacity, tokens);
      EXPECT_LE(shape.span_capacity, shape.token_capacity);
      if (i) {
        const auto previous = shapes[i - 1];
        if (previous.token_capacity == shape.token_capacity) {
          EXPECT_EQ(shape.span_capacity,
                    iree_min(previous.span_capacity * 2, shape.token_capacity));
        } else {
          EXPECT_EQ(previous.span_capacity, previous.token_capacity);
          EXPECT_EQ(shape.token_capacity,
                    iree_min(previous.token_capacity * 2, tokens));
          EXPECT_EQ(shape.span_capacity, 1);
        }
      }
    }
  }
}

TEST(TextScheduleTest, DefaultCatalogShrinksBothAxesWithLiveReadiness) {
  const iree_host_size_t shape_count =
      loom_serve_text_default_shapes(512, nullptr);
  std::vector<loom_serve_packing_shape_t> shapes(shape_count);
  loom_serve_text_default_shapes(512, shapes.data());
  ASSERT_EQ(shape_count, 55);
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
      ASSERT_EQ(loom_serve_pack_shapes(16, ready, shape_count, shapes.data(),
                                       512, &cursor, spans, scratch, &shape),
                active);
      iree_host_size_t expected_tokens = 1;
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
  ASSERT_EQ(loom_serve_pack_shapes(16, ready, shape_count, shapes.data(), 512,
                                   &cursor, spans, scratch, &shape),
            16);
  EXPECT_EQ(shapes[shape].token_capacity, 512);
  EXPECT_EQ(shapes[shape].span_capacity, 16);
  EXPECT_EQ(spans[15].token_count, 452);
}

TEST(TextScheduleTest, SameCatalogFillsWiderCohortsAsResidentsGrow) {
  const iree_host_size_t token_capacity = 128;
  const iree_host_size_t shape_count =
      loom_serve_text_default_shapes(token_capacity, nullptr);
  std::vector<loom_serve_packing_shape_t> shapes(shape_count);
  loom_serve_text_default_shapes(token_capacity, shapes.data());
  for (iree_host_size_t residents : {1u, 4u, 17u, 65u, 100u, 129u, 257u}) {
    SCOPED_TRACE(residents);
    std::vector<loom_serve_ready_span_t> ready(residents, {1, 1});
    std::vector<loom_serve_packed_span_t> spans(token_capacity);
    std::vector<loom_serve_packed_span_t> scratch(token_capacity);
    iree_host_size_t cursor = 0, shape = 0;
    const iree_host_size_t expected = iree_min(residents, token_capacity);
    EXPECT_EQ(loom_serve_pack_shapes(residents, ready.data(), shape_count,
                                     shapes.data(), token_capacity, &cursor,
                                     spans.data(), scratch.data(), &shape),
              expected);
    iree_host_size_t bucket = 1;
    while (bucket < expected) {
      bucket *= 2;
    }
    EXPECT_EQ(shapes[shape].token_capacity, bucket);
    EXPECT_EQ(shapes[shape].span_capacity, bucket);
    for (iree_host_size_t i = 0; i < expected; ++i) {
      EXPECT_EQ(spans[i].row_index, i);
      EXPECT_EQ(spans[i].token_count, 1);
    }
    if (residents > token_capacity) {
      EXPECT_EQ(cursor, token_capacity);
    }
  }
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

TEST(TextScheduleTest, CompletionReservationHandlesWideSourcePages) {
  for (iree_host_size_t page :
       {iree_host_size_t{1}, iree_host_size_t{3}, iree_host_size_t{262143},
        iree_host_size_t{UINT32_MAX}}) {
    for (iree_host_size_t extent : {1u, 3u, 262143u, 262144u}) {
      SCOPED_TRACE(page);
      SCOPED_TRACE(extent);
      const uint64_t expected = ((uint64_t{extent} + page - 1) / page) * page;
      EXPECT_EQ(loom_serve_text_request_reservation(262144, extent, 1, 0, page),
                expected);
    }
  }
}

}  // namespace
