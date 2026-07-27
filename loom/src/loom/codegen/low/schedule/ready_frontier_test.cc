// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/ready_frontier.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class ScheduleReadyFrontierTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(128 * 1024, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
};

static loom_low_schedule_ready_keys_t MakeKeys(uint64_t source,
                                               uint64_t pressure,
                                               uint64_t schedule,
                                               uint64_t storage) {
  return {{source, pressure, schedule, storage}};
}

TEST_F(ScheduleReadyFrontierTest, MaintainsViewsAcrossSegmentBoundaries) {
  loom_low_schedule_ready_frontier_t frontier;
  IREE_ASSERT_OK(loom_low_schedule_ready_frontier_initialize(
      /*node_capacity=*/5000, /*descriptor_count=*/3,
      LOOM_LOW_SCHEDULE_READY_VIEW_COUNT, &arena_, &frontier));

  const auto keys4999 = MakeKeys(50, 100, 50, 100);
  loom_low_schedule_ready_frontier_insert(&frontier, 4999, &keys4999, 1);
  const auto keys513 = MakeKeys(10, 20, 60, 80);
  loom_low_schedule_ready_frontier_insert(&frontier, 513, &keys513, 1);
  const auto keys4097 = MakeKeys(30, 5, 40, 90);
  loom_low_schedule_ready_frontier_insert(&frontier, 4097, &keys4097, 2);

  EXPECT_EQ(loom_low_schedule_ready_frontier_count(&frontier), 3u);
  uint32_t source_nodes[3];
  EXPECT_EQ(
      loom_low_schedule_ready_frontier_copy_best(
          &frontier, LOOM_LOW_SCHEDULE_READY_VIEW_SOURCE, 3, source_nodes),
      3u);
  EXPECT_EQ(source_nodes[0], 513u);
  EXPECT_EQ(source_nodes[1], 4097u);
  EXPECT_EQ(source_nodes[2], 4999u);
  uint32_t best_node = LOOM_LOW_SCHEDULE_READY_NODE_NONE;
  EXPECT_EQ(
      loom_low_schedule_ready_frontier_copy_best(
          &frontier, LOOM_LOW_SCHEDULE_READY_VIEW_PRESSURE, 1, &best_node),
      1u);
  EXPECT_EQ(best_node, 4097u);
  EXPECT_EQ(
      loom_low_schedule_ready_frontier_copy_best(
          &frontier, LOOM_LOW_SCHEDULE_READY_VIEW_SCHEDULE, 1, &best_node),
      1u);
  EXPECT_EQ(best_node, 4097u);
  EXPECT_EQ(loom_low_schedule_ready_frontier_copy_best(
                &frontier, LOOM_LOW_SCHEDULE_READY_VIEW_STORAGE, 1, &best_node),
            1u);
  EXPECT_EQ(best_node, 513u);
  EXPECT_EQ(loom_low_schedule_ready_frontier_descriptor_head(&frontier, 1),
            513u);
  EXPECT_EQ(loom_low_schedule_ready_frontier_descriptor_count(&frontier, 1),
            2u);

  loom_low_schedule_ready_frontier_update_key(
      &frontier, LOOM_LOW_SCHEDULE_READY_VIEW_PRESSURE, 4999, 1);
  uint32_t pressure_nodes[2];
  EXPECT_EQ(
      loom_low_schedule_ready_frontier_copy_best(
          &frontier, LOOM_LOW_SCHEDULE_READY_VIEW_PRESSURE, 2, pressure_nodes),
      2u);
  EXPECT_EQ(pressure_nodes[0], 4999u);
  EXPECT_EQ(pressure_nodes[1], 4097u);

  loom_low_schedule_ready_frontier_remove(&frontier, 513);
  EXPECT_FALSE(loom_low_schedule_ready_frontier_contains(&frontier, 513));
  EXPECT_EQ(
      loom_low_schedule_ready_frontier_copy_best(
          &frontier, LOOM_LOW_SCHEDULE_READY_VIEW_SOURCE, 2, source_nodes),
      2u);
  EXPECT_EQ(source_nodes[0], 4097u);
  EXPECT_EQ(source_nodes[1], 4999u);
  EXPECT_EQ(loom_low_schedule_ready_frontier_descriptor_head(&frontier, 1),
            4999u);
  EXPECT_EQ(loom_low_schedule_ready_frontier_descriptor_count(&frontier, 1),
            1u);

  loom_low_schedule_ready_frontier_remove(&frontier, 4999);
  loom_low_schedule_ready_frontier_remove(&frontier, 4097);
  EXPECT_EQ(loom_low_schedule_ready_frontier_count(&frontier), 0u);
  EXPECT_EQ(loom_low_schedule_ready_frontier_descriptor_count(&frontier, 1),
            0u);
  EXPECT_EQ(loom_low_schedule_ready_frontier_descriptor_count(&frontier, 2),
            0u);
}

TEST_F(ScheduleReadyFrontierTest, UsesNodeIndexAsStableKeyTieBreaker) {
  loom_low_schedule_ready_frontier_t frontier;
  IREE_ASSERT_OK(loom_low_schedule_ready_frontier_initialize(
      /*node_capacity=*/8, /*descriptor_count=*/0,
      /*view_count=*/1, &arena_, &frontier));
  const auto keys = MakeKeys(7, 0, 0, 0);
  loom_low_schedule_ready_frontier_insert(&frontier, 6, &keys,
                                          LOOM_LOW_SCHEDULE_READY_NODE_NONE);
  loom_low_schedule_ready_frontier_insert(&frontier, 2, &keys,
                                          LOOM_LOW_SCHEDULE_READY_NODE_NONE);
  uint32_t best_node = LOOM_LOW_SCHEDULE_READY_NODE_NONE;
  EXPECT_EQ(loom_low_schedule_ready_frontier_copy_best(
                &frontier, LOOM_LOW_SCHEDULE_READY_VIEW_SOURCE, 1, &best_node),
            1u);
  EXPECT_EQ(best_node, 2u);
}

TEST_F(ScheduleReadyFrontierTest, MatchesSortedOracleUnderMutations) {
  static constexpr uint32_t kNodeCapacity = 5000;
  static constexpr uint32_t kDescriptorCount = 7;
  loom_low_schedule_ready_frontier_t frontier;
  IREE_ASSERT_OK(loom_low_schedule_ready_frontier_initialize(
      kNodeCapacity, kDescriptorCount, LOOM_LOW_SCHEDULE_READY_VIEW_COUNT,
      &arena_, &frontier));

  std::vector<loom_low_schedule_ready_keys_t> keys(kNodeCapacity);
  std::vector<uint32_t> descriptors(kNodeCapacity,
                                    LOOM_LOW_SCHEDULE_READY_NODE_NONE);
  std::vector<uint8_t> active(kNodeCapacity, 0);
  uint32_t random_state = 0xC001C0DEu;
  const auto next_random = [&random_state]() {
    random_state = random_state * 1664525u + 1013904223u;
    return random_state;
  };

  const auto verify = [&]() {
    std::vector<uint32_t> expected_nodes;
    for (uint32_t node = 0; node < kNodeCapacity; ++node) {
      if (active[node]) expected_nodes.push_back(node);
    }
    ASSERT_EQ(loom_low_schedule_ready_frontier_count(&frontier),
              expected_nodes.size());
    for (uint8_t view = 0; view < LOOM_LOW_SCHEDULE_READY_VIEW_COUNT; ++view) {
      std::sort(expected_nodes.begin(), expected_nodes.end(),
                [&](uint32_t left, uint32_t right) {
                  const uint64_t left_key = keys[left].values[view];
                  const uint64_t right_key = keys[right].values[view];
                  return left_key != right_key ? left_key < right_key
                                               : left < right;
                });
      uint32_t actual_nodes[LOOM_LOW_SCHEDULE_READY_COPY_CAPACITY];
      const uint8_t expected_count = static_cast<uint8_t>(std::min<size_t>(
          expected_nodes.size(), LOOM_LOW_SCHEDULE_READY_COPY_CAPACITY));
      ASSERT_EQ(
          loom_low_schedule_ready_frontier_copy_best(
              &frontier, static_cast<loom_low_schedule_ready_view_t>(view),
              LOOM_LOW_SCHEDULE_READY_COPY_CAPACITY, actual_nodes),
          expected_count);
      for (uint8_t i = 0; i < expected_count; ++i) {
        EXPECT_EQ(actual_nodes[i], expected_nodes[i]);
      }
    }
    for (uint32_t descriptor = 0; descriptor < kDescriptorCount; ++descriptor) {
      uint32_t expected_count = 0;
      for (uint32_t node = 0; node < kNodeCapacity; ++node) {
        expected_count += active[node] && descriptors[node] == descriptor;
      }
      EXPECT_EQ(loom_low_schedule_ready_frontier_descriptor_count(&frontier,
                                                                  descriptor),
                expected_count);
    }
  };

  for (uint32_t node = 0; node < 4200; ++node) {
    keys[node] = MakeKeys(next_random() % 257u, next_random() % 257u,
                          next_random() % 257u, next_random() % 257u);
    descriptors[node] = next_random() % (kDescriptorCount + 1u);
    if (descriptors[node] == kDescriptorCount) {
      descriptors[node] = LOOM_LOW_SCHEDULE_READY_NODE_NONE;
    }
    loom_low_schedule_ready_frontier_insert(&frontier, node, &keys[node],
                                            descriptors[node]);
    active[node] = 1;
  }
  verify();

  for (uint32_t step = 0; step < 4096; ++step) {
    const uint32_t node = next_random() % kNodeCapacity;
    if (!active[node]) {
      keys[node] = MakeKeys(next_random() % 257u, next_random() % 257u,
                            next_random() % 257u, next_random() % 257u);
      descriptors[node] = next_random() % (kDescriptorCount + 1u);
      if (descriptors[node] == kDescriptorCount) {
        descriptors[node] = LOOM_LOW_SCHEDULE_READY_NODE_NONE;
      }
      loom_low_schedule_ready_frontier_insert(&frontier, node, &keys[node],
                                              descriptors[node]);
      active[node] = 1;
    } else if ((next_random() & 3u) == 0) {
      loom_low_schedule_ready_frontier_remove(&frontier, node);
      active[node] = 0;
    } else {
      const auto view = static_cast<loom_low_schedule_ready_view_t>(
          next_random() % LOOM_LOW_SCHEDULE_READY_VIEW_COUNT);
      const uint64_t key = next_random() % 257u;
      keys[node].values[view] = key;
      loom_low_schedule_ready_frontier_update_key(&frontier, view, node, key);
    }
    if ((step & 31u) == 0) verify();
  }
  verify();
}

}  // namespace
}  // namespace loom
