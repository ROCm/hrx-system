// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/liveness_events.h"

#include <algorithm>
#include <random>
#include <tuple>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class LivenessEventsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void CheckOrder(std::vector<loom_liveness_event_t> events,
                  uint32_t maximum_point) {
    auto expected = events;
    std::sort(expected.begin(), expected.end(),
              [](const auto& a, const auto& b) {
                return std::tie(a.point, a.value_delta, a.value_id) <
                       std::tie(b.point, b.value_delta, b.value_id);
              });
    IREE_ASSERT_OK(loom_liveness_events_sort(events.data(), events.size(),
                                             maximum_point, &arena_));
    for (size_t i = 0; i < events.size(); ++i) {
      SCOPED_TRACE(i);
      EXPECT_EQ(events[i].point, expected[i].point);
      EXPECT_EQ(events[i].value_delta, expected[i].value_delta);
      EXPECT_EQ(events[i].value_id, expected[i].value_id);
      EXPECT_EQ(events[i].value_ordinal, expected[i].value_ordinal);
    }
    EXPECT_LE(arena_.used_allocation_size,
              events.size() * sizeof(loom_liveness_event_t));
    if (maximum_point >= events.size()) {
      EXPECT_EQ(arena_.used_allocation_size, 0u);
    }
    iree_arena_reset(&arena_);
  }

  // Backing blocks shared by successive endpoint-ordering cases.
  iree_arena_block_pool_t pool_;
  // Sort scratch reset after each oracle comparison.
  iree_arena_allocator_t arena_;
};

TEST_F(LivenessEventsTest, EmptyStreamNeedsNoStorage) {
  IREE_ASSERT_OK(loom_liveness_events_sort(nullptr, 0, 0, &arena_));
  EXPECT_EQ(arena_.used_allocation_size, 0u);
}

TEST_F(LivenessEventsTest, MatchesEndpointOrderAcrossPointDomains) {
  // Nonempty half-open segments model the producer contract. Domains include
  // coincident endpoints, empty point buckets, the dense/sparse boundary, and
  // the largest program point without allocating scratch proportional to it.
  std::mt19937 random(0x11FE);
  for (uint32_t segment_count : {1u, 32u, 33u, 128u, 1255u}) {
    SCOPED_TRACE(segment_count);
    for (uint32_t maximum_point :
         {1u, 17u, segment_count * 2u - 1u, segment_count * 2u, UINT32_MAX}) {
      SCOPED_TRACE(maximum_point);
      std::vector<loom_liveness_event_t> events;
      for (uint32_t ordinal = 0; ordinal < segment_count; ++ordinal) {
        const uint32_t start = random() % maximum_point;
        const uint32_t end = start + 1u + random() % (maximum_point - start);
        // SSA IDs and retained interval ordinals are distinct index domains.
        const loom_value_id_t value_id = (segment_count - ordinal) * 7u;
        events.push_back({end, value_id, ordinal, -1});
        events.push_back({start, value_id, ordinal, 1});
      }
      CheckOrder(events, maximum_point);
      std::reverse(events.begin(), events.end());
      CheckOrder(events, maximum_point);
      std::shuffle(events.begin(), events.end(), random);
      CheckOrder(events, maximum_point);
    }
  }
}

}  // namespace
