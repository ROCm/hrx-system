// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "graph_dependencies.h"

#include <array>
#include <vector>

#include "iree/testing/gtest.h"

namespace {

TEST(GraphDependenciesTest, EmptyScheduleHasNoCompletionProducer) {
  const hrx_graph_schedule_t schedule = {};
  hrx_graph_dependency_cursor_t cursor;
  hrx_graph_dependency_cursor_initialize(&schedule, &cursor);
  EXPECT_EQ(cursor.block_count, 0u);
  EXPECT_EQ(cursor.semaphore_count, 0u);
  hrx_graph_dependency_block_t block;
  EXPECT_FALSE(hrx_graph_dependency_cursor_next(&cursor, &block));
}

TEST(GraphDependenciesTest, AdjacentPartitionsJoinEveryPredecessor) {
  for (uint32_t first_width = 1; first_width <= HRX_GRAPH_MAX_WORKSTREAMS;
       ++first_width) {
    for (uint32_t second_width = 1; second_width <= HRX_GRAPH_MAX_WORKSTREAMS;
         ++second_width) {
      SCOPED_TRACE(first_width);
      SCOPED_TRACE(second_width);
      hrx_graph_partition_t partitions[] = {
          {0, first_width, first_width},
          {first_width, second_width, second_width},
      };
      const hrx_graph_schedule_t schedule = {
          .partition_count = 2,
          .block_count = first_width + second_width,
          .partitions = partitions,
      };
      hrx_graph_dependency_cursor_t cursor;
      hrx_graph_dependency_cursor_initialize(&schedule, &cursor);
      std::vector<iree_hal_semaphore_t*> semaphores(cursor.semaphore_count);
      // Only list construction is under test: entries are borrowed opaque
      // resources, never dereferenced by the dependency resolver.
      std::array<iree_hal_semaphore_t*, 2> external_semaphores = {};
      uint64_t external_values[] = {11, 23};
      const iree_hal_semaphore_list_t external = {
          .count = 2,
          .semaphores = external_semaphores.data(),
          .payload_values = external_values,
      };
      uint64_t values[HRX_GRAPH_MAX_WORKSTREAMS];
      for (auto& value : values) {
        value = 7;
      }
      uint32_t ordinal = 0;
      hrx_graph_dependency_block_t block;
      while (hrx_graph_dependency_cursor_next(&cursor, &block)) {
        iree_hal_semaphore_list_t waits, signals;
        hrx_graph_dependencies_resolve(
            block.dependencies, ordinal, cursor.semaphore_count,
            semaphores.data(), values, external, external, &waits, &signals);
        if (ordinal < first_width) {
          EXPECT_EQ(block.partition, &partitions[0]);
          EXPECT_EQ(block.stream_index, ordinal);
          EXPECT_EQ(waits.count, external.count);
          EXPECT_EQ(waits.semaphores, external.semaphores);
          EXPECT_EQ(waits.payload_values, external.payload_values);
        } else {
          const bool terminal_join = ordinal == schedule.block_count;
          EXPECT_EQ(block.partition, terminal_join ? nullptr : &partitions[1]);
          EXPECT_EQ(block.stream_index,
                    terminal_join ? 0 : ordinal - first_width);
          EXPECT_EQ(waits.count, terminal_join ? second_width : first_width);
          EXPECT_EQ(waits.semaphores,
                    semaphores.data() + (terminal_join ? first_width : 0));
          for (uint32_t i = 0; i < waits.count; ++i) {
            EXPECT_EQ(waits.payload_values[i], 7u);
          }
        }
        if (ordinal < cursor.semaphore_count) {
          EXPECT_EQ(signals.count, 1u);
          EXPECT_EQ(signals.semaphores, semaphores.data() + ordinal);
          EXPECT_EQ(signals.payload_values[0], 7u);
        } else {
          EXPECT_EQ(signals.count, external.count);
          EXPECT_EQ(signals.semaphores, external.semaphores);
          EXPECT_EQ(signals.payload_values, external.payload_values);
        }
        ++ordinal;
      }
      EXPECT_EQ(ordinal, first_width + second_width + (second_width > 1));
      EXPECT_EQ(ordinal, cursor.block_count);
      EXPECT_EQ(cursor.semaphore_count, ordinal - 1);
    }
  }
}

TEST(GraphDependenciesTest, LargeSchedulesKeepFullSemaphoreOrdinals) {
  constexpr uint32_t kCount = 65540;
  std::vector<hrx_graph_partition_t> partitions(kCount);
  for (uint32_t i = 0; i < kCount; ++i) {
    partitions[i] = {i, 1, 1};
  }
  const hrx_graph_schedule_t schedule = {
      .partition_count = kCount,
      .block_count = kCount,
      .partitions = partitions.data(),
  };
  hrx_graph_dependency_cursor_t cursor;
  hrx_graph_dependency_cursor_initialize(&schedule, &cursor);
  std::vector<iree_hal_semaphore_t*> semaphores(kCount - 1);
  uint64_t values[HRX_GRAPH_MAX_WORKSTREAMS] = {19};
  uint32_t ordinal = 0;
  hrx_graph_dependency_block_t block;
  while (hrx_graph_dependency_cursor_next(&cursor, &block)) {
    iree_hal_semaphore_list_t waits, signals;
    hrx_graph_dependencies_resolve(block.dependencies, ordinal,
                                   cursor.semaphore_count, semaphores.data(),
                                   values, {}, {}, &waits, &signals);
    EXPECT_EQ(block.partition, &partitions[ordinal]);
    if (ordinal) {
      EXPECT_EQ(waits.count, 1u);
      EXPECT_EQ(waits.semaphores, semaphores.data() + ordinal - 1);
      EXPECT_EQ(waits.payload_values[0], 19u);
    }
    if (ordinal + 1 < kCount) {
      EXPECT_EQ(signals.count, 1u);
      EXPECT_EQ(signals.semaphores, semaphores.data() + ordinal);
    }
    ++ordinal;
  }
  EXPECT_EQ(ordinal, kCount);
}

}  // namespace
