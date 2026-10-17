// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/dependencies.h"

#include <cstdint>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class ScheduleDependenciesTest : public ::testing::Test {
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

TEST_F(ScheduleDependenciesTest, StableAcrossSegmentBoundaries) {
  loom_low_schedule_dependency_graph_t source;
  loom_low_schedule_dependency_graph_initialize(&source);
  constexpr uint32_t kDependencyCount =
      LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_CAPACITY + 1;
  for (uint32_t i = 0; i < kDependencyCount; ++i) {
    loom_low_schedule_dependency_t dependency = {
        .producer_node = i,
        .consumer_node = i + 1,
        .minimum_issue_separation_cycles = (i & 1u) == 0 ? -3 : 5,
        .producer_attachment_index = static_cast<uint16_t>(i & 7u),
        .consumer_attachment_index = static_cast<uint16_t>(i & 15u),
        .producer_event_id = static_cast<uint16_t>(i & 31u),
        .consumer_event_id = static_cast<uint16_t>(i & 63u),
        .value_operand_index = static_cast<uint16_t>(i & 3u),
        .producer_attachment_kind =
            LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_OPERAND,
        .consumer_attachment_kind =
            LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_EFFECT,
        .kind = LOOM_LOW_SCHEDULE_DEPENDENCY_SSA,
        .separation_source = LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_EVENT_PAIR,
        .model_quality = 4,
    };
    IREE_ASSERT_OK(loom_low_schedule_dependency_graph_append(
        &source, dependency, &arena_));
  }

  loom_low_schedule_dependency_graph_t graph;
  loom_low_schedule_dependency_graph_move(&source, &graph);

  EXPECT_EQ(source.count, 0u);
  EXPECT_EQ(graph.count, kDependencyCount);
  for (uint32_t i : {0u, kDependencyCount - 2, kDependencyCount - 1}) {
    const loom_low_schedule_dependency_t* dependency =
        loom_low_schedule_dependency_graph_at(&graph, i);
    EXPECT_EQ(dependency->producer_node, i);
    EXPECT_EQ(dependency->consumer_node, i + 1);
    EXPECT_EQ(dependency->minimum_issue_separation_cycles,
              (i & 1u) == 0 ? -3 : 5);
    EXPECT_EQ(dependency->producer_attachment_index, i & 7u);
    EXPECT_EQ(dependency->consumer_attachment_index, i & 15u);
    EXPECT_EQ(dependency->producer_event_id, i & 31u);
    EXPECT_EQ(dependency->consumer_event_id, i & 63u);
    EXPECT_EQ(dependency->value_operand_index, i & 3u);
    EXPECT_EQ(dependency->producer_attachment_kind,
              LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_OPERAND);
    EXPECT_EQ(dependency->consumer_attachment_kind,
              LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_EFFECT);
    EXPECT_EQ(dependency->separation_source,
              LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_EVENT_PAIR);
    EXPECT_EQ(dependency->model_quality, 4);
  }
}

TEST_F(ScheduleDependenciesTest, EmptyRetainedRangeAllocatesNothing) {
  loom_low_schedule_dependency_graph_t graph;
  loom_low_schedule_dependency_graph_initialize(&graph);
  loom_low_schedule_dependency_slice_t slice;
  IREE_ASSERT_OK(loom_low_schedule_dependency_graph_retain_range(
      &graph, {0, 0}, &arena_, &slice));
  EXPECT_EQ(slice.count, 0u);
  EXPECT_EQ(slice.segments, nullptr);
  EXPECT_EQ(arena_.used_allocation_size, 0u);
}

TEST_F(ScheduleDependenciesTest, RetainedRangeOutlivesGraphStorage) {
  constexpr uint32_t kCapacity = LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_CAPACITY;
  const loom_low_schedule_dependency_range_t ranges[] = {
      {0, 1},
      {kCapacity - 1, 1},
      {0, kCapacity + 2},
      {kCapacity - 1, kCapacity + 2},
      {17 * kCapacity + 1, kCapacity + 2},
  };
  for (const auto range : ranges) {
    SCOPED_TRACE(range.start);
    SCOPED_TRACE(range.count);
    iree_arena_allocator_t retained_arena;
    iree_arena_initialize(&block_pool_, &retained_arena);
    loom_low_schedule_dependency_graph_t graph;
    loom_low_schedule_dependency_graph_initialize(&graph);
    // Later phases may append to the last retained segment, then allocate
    // more scratch-owned segments. The published slice retains only its range.
    const uint32_t dependency_count = range.start + range.count + kCapacity;
    for (uint32_t i = 0; i < dependency_count; ++i) {
      const loom_low_schedule_dependency_t dependency = {
          .producer_node = i,
          .consumer_node = i + 1,
          .minimum_issue_separation_cycles = (i & 1u) == 0 ? -3 : 5,
          .producer_attachment_index = static_cast<uint16_t>(i & 7u),
          .consumer_attachment_index = static_cast<uint16_t>(i & 15u),
          .producer_event_id = static_cast<uint16_t>(i & 31u),
          .consumer_event_id = static_cast<uint16_t>(i & 63u),
          .value_operand_index = UINT16_MAX,
          .producer_attachment_kind =
              LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_EFFECT,
          .consumer_attachment_kind =
              LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_EFFECT,
          .kind = LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT,
          .separation_source = LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_EVENT_PAIR,
          .model_quality = 4,
      };
      auto* arena = i >= range.start && i < range.start + range.count
                        ? &retained_arena
                        : &arena_;
      IREE_ASSERT_OK(
          loom_low_schedule_dependency_graph_append(&graph, dependency, arena));
    }

    loom_low_schedule_dependency_slice_t slice;
    IREE_ASSERT_OK(loom_low_schedule_dependency_graph_retain_range(
        &graph, range, &retained_arena, &slice));
    EXPECT_EQ(slice.count, range.count);
    for (uint32_t ordinal = 0; ordinal < range.count; ++ordinal) {
      const uint32_t i = range.start + ordinal;
      const auto* original = loom_low_schedule_dependency_graph_at(&graph, i);
      const auto* retained =
          loom_low_schedule_dependency_slice_at(&slice, ordinal);
      if (range.start % kCapacity != 0 &&
          i / kCapacity == range.start / kCapacity) {
        EXPECT_NE(retained, original);
      } else {
        EXPECT_EQ(retained, original);
      }
    }
    iree_arena_reset(&arena_);
    iree_arena_block_pool_trim(&block_pool_);
    for (uint32_t ordinal = 0; ordinal < range.count; ++ordinal) {
      const uint32_t i = range.start + ordinal;
      const auto* dependency =
          loom_low_schedule_dependency_slice_at(&slice, ordinal);
      EXPECT_EQ(dependency->producer_node, i);
      EXPECT_EQ(dependency->consumer_node, i + 1);
      EXPECT_EQ(dependency->minimum_issue_separation_cycles,
                (i & 1u) == 0 ? -3 : 5);
      EXPECT_EQ(dependency->producer_attachment_index, i & 7u);
      EXPECT_EQ(dependency->consumer_attachment_index, i & 15u);
      EXPECT_EQ(dependency->producer_event_id, i & 31u);
      EXPECT_EQ(dependency->consumer_event_id, i & 63u);
      EXPECT_EQ(dependency->value_operand_index, UINT16_MAX);
      EXPECT_EQ(dependency->producer_attachment_kind,
                LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_EFFECT);
      EXPECT_EQ(dependency->consumer_attachment_kind,
                LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_EFFECT);
      EXPECT_EQ(dependency->kind, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT);
      EXPECT_EQ(dependency->separation_source,
                LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_EVENT_PAIR);
      EXPECT_EQ(dependency->model_quality, 4);
    }
    iree_arena_deinitialize(&retained_arena);
  }
}

}  // namespace
}  // namespace loom
