// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_vector_partition.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

static loom_value_fact_vector_partition_t MakeExactPartition(
    const loom_value_fact_vector_partition_component_t* components,
    uint32_t count, uint32_t lane_count, uint32_t regular_lane_count) {
  return {components, nullptr, count, lane_count, regular_lane_count, 0, 0};
}

static loom_value_fact_vector_partition_t MakeRegularPartition(
    const loom_value_fact_vector_partition_component_t* components,
    uint32_t count, uint32_t lane_count, uint32_t regular_lane_count) {
  return {components, components,        count, lane_count, regular_lane_count,
          count,      regular_lane_count};
}

class FactVectorPartitionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&table_, &arena_, 0));
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Backing storage shared by independently scoped fact tables.
  iree_arena_block_pool_t block_pool_;
  // Storage owning the primary fact table and its partitions.
  iree_arena_allocator_t arena_;
  // Primary scope under test.
  loom_value_fact_table_t table_ = {};
};

TEST_F(FactVectorPartitionTest, PartitionsOwnComponentsAndTrackScope) {
  loom_value_fact_vector_partition_component_t components[] = {
      {1, 16}, {2, 16}, {3, 8}};
  auto regular_partition = MakeRegularPartition(components, 3, 40, 16);
  regular_partition.flags =
      LOOM_VALUE_FACT_VECTOR_PARTITION_FLAG_COMPUTED_PACKETS;
  bool changed = false;
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &table_, 7, regular_partition, &changed));
  EXPECT_TRUE(changed);
  const auto first = loom_value_fact_table_query_vector_partition(&table_, 7);
  ASSERT_EQ(first.count, 3u);
  EXPECT_NE(first.components, components);
  EXPECT_EQ(first.lane_count, 40u);
  EXPECT_EQ(first.regular_lane_count, 16u);
  EXPECT_EQ(first.snapshot_components, first.components);
  EXPECT_EQ(first.snapshot_count, 3u);
  EXPECT_EQ(first.snapshot_lane_count, 16u);
  EXPECT_EQ(first.flags,
            LOOM_VALUE_FACT_VECTOR_PARTITION_FLAG_COMPUTED_PACKETS);
  EXPECT_EQ(first.components[0].value, 1u);
  EXPECT_EQ(first.components[2].value, 3u);
  EXPECT_EQ(first.components[2].lane_count, 8u);

  const auto bytes = arena_.used_allocation_size;
  changed = false;
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &table_, 7, regular_partition, &changed));
  EXPECT_FALSE(changed);
  EXPECT_EQ(loom_value_fact_table_query_vector_partition(&table_, 7).components,
            first.components);
  EXPECT_EQ(arena_.used_allocation_size, bytes);

  components[0] = {4, 1};
  components[1] = {5, 1};
  changed = false;
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &table_, 7, MakeExactPartition(components, 2, 2, 0), &changed));
  EXPECT_TRUE(changed);
  const auto second = loom_value_fact_table_query_vector_partition(&table_, 7);
  ASSERT_EQ(second.count, 2u);
  EXPECT_EQ(second.lane_count, 2u);
  EXPECT_EQ(second.regular_lane_count, 0u);
  EXPECT_EQ(second.snapshot_count, 0u);
  EXPECT_EQ(second.components[0].value, 4u);
  EXPECT_EQ(first.components[0].value, 1u);

  loom_value_fact_table_clear_scope(&table_);
  EXPECT_EQ(table_.vector_partitions, nullptr);
  EXPECT_EQ(loom_value_fact_table_query_vector_partition(&table_, 7).count, 0u);
}

TEST_F(FactVectorPartitionTest, AliasesShareAndInvalidationWithdrawsFacts) {
  const loom_value_fact_vector_partition_component_t components[] = {{1, 16},
                                                                     {2, 8}};
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &table_, 7, MakeRegularPartition(components, 2, 24, 16),
      /*inout_changed=*/nullptr));
  bool changed = false;
  IREE_ASSERT_OK(
      loom_value_fact_table_forward_vector_partition(&table_, 7, 8, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(
      loom_value_fact_table_query_vector_partition(&table_, 8).components,
      loom_value_fact_table_query_vector_partition(&table_, 7).components);

  loom_value_fact_table_undefine(&table_, 7);
  EXPECT_EQ(loom_value_fact_table_query_vector_partition(&table_, 7).count, 0u);
  EXPECT_EQ(loom_value_fact_table_query_vector_partition(&table_, 8).count, 2u);
  changed = false;
  IREE_ASSERT_OK(
      loom_value_fact_table_forward_vector_partition(&table_, 7, 8, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(loom_value_fact_table_query_vector_partition(&table_, 8).count, 0u);
}

TEST_F(FactVectorPartitionTest, SameIdSnapshotsOwnPartitionValues) {
  iree_arena_allocator_t source_arena;
  iree_arena_initialize(&block_pool_, &source_arena);
  loom_value_fact_table_t source = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &source_arena, 0));
  const loom_value_fact_vector_partition_component_t components[] = {
      {1, 16}, {2, 16}, {3, 8}};
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &source, 7, MakeRegularPartition(components, 3, 40, 16),
      /*inout_changed=*/nullptr));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&source, 7, loom_value_facts_unknown()));
  const loom_value_id_t selected_values[] = {7};
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &table_, {&source, selected_values, 1}, nullptr));
  EXPECT_NE(
      loom_value_fact_table_query_vector_partition(&table_, 7).components,
      loom_value_fact_table_query_vector_partition(&source, 7).components);
  iree_arena_deinitialize(&source_arena);

  const auto copied = loom_value_fact_table_query_vector_partition(&table_, 7);
  ASSERT_EQ(copied.count, 3u);
  EXPECT_EQ(copied.lane_count, 40u);
  EXPECT_EQ(copied.regular_lane_count, 16u);
  EXPECT_EQ(copied.snapshot_components, copied.components);
  EXPECT_EQ(copied.snapshot_count, 3u);
  EXPECT_EQ(copied.snapshot_lane_count, 16u);
  EXPECT_EQ(copied.components[0].value, 1u);
  EXPECT_EQ(copied.components[2].value, 3u);
}

TEST_F(FactVectorPartitionTest, ComponentScratchGrowsAndReusesStorage) {
  loom_value_fact_vector_partition_component_t* first = nullptr;
  IREE_ASSERT_OK(loom_value_fact_table_vector_partition_component_scratch(
      &table_, 2, &first));
  ASSERT_NE(first, nullptr);
  first[0] = {1, 1};

  loom_value_fact_vector_partition_component_t* reused = nullptr;
  IREE_ASSERT_OK(loom_value_fact_table_vector_partition_component_scratch(
      &table_, 1, &reused));
  EXPECT_EQ(reused, first);
  EXPECT_EQ(reused[0].value, 1u);

  loom_value_fact_vector_partition_component_t* grown = nullptr;
  IREE_ASSERT_OK(loom_value_fact_table_vector_partition_component_scratch(
      &table_, 4, &grown));
  EXPECT_NE(grown, first);
}

TEST_F(FactVectorPartitionTest, RetainsRegularPartitionBeyondExpansionLimit) {
  constexpr uint32_t kComponentCount =
      LOOM_VALUE_FACT_VECTOR_PARTITION_EXPANSION_LIMIT + 1u;
  loom_value_fact_vector_partition_component_t components[kComponentCount];
  for (uint32_t i = 0; i < kComponentCount; ++i) {
    components[i] = {i + 1, 16};
  }
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &table_, 66,
      MakeRegularPartition(components, kComponentCount, kComponentCount * 16u,
                           16),
      /*inout_changed=*/nullptr));

  const auto partition =
      loom_value_fact_table_query_vector_partition(&table_, 66);
  ASSERT_EQ(partition.count, kComponentCount);
  EXPECT_EQ(partition.lane_count, kComponentCount * 16u);
  EXPECT_EQ(partition.regular_lane_count, 16u);
  EXPECT_EQ(partition.snapshot_count, kComponentCount);
  EXPECT_EQ(partition.snapshot_lane_count, 16u);
  EXPECT_EQ(partition.components[0].value, 1u);
  EXPECT_EQ(partition.components[partition.count - 1].value, kComponentCount);
}

TEST_F(FactVectorPartitionTest, ExactAndSnapshotViewsRetainDifferentWidths) {
  const loom_value_fact_vector_partition_component_t exact_components[] = {
      {1, 16}, {2, 16}, {3, 8}};
  const loom_value_fact_vector_partition_component_t snapshot_components[] = {
      {4, 32}, {5, 8}};
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &table_, 7, {exact_components, snapshot_components, 3, 40, 16, 2, 32},
      /*inout_changed=*/nullptr));

  const auto partition =
      loom_value_fact_table_query_vector_partition(&table_, 7);
  ASSERT_EQ(partition.count, 3u);
  ASSERT_EQ(partition.snapshot_count, 2u);
  EXPECT_EQ(partition.regular_lane_count, 16u);
  EXPECT_EQ(partition.snapshot_lane_count, 32u);
  EXPECT_NE(partition.components, exact_components);
  EXPECT_NE(partition.snapshot_components, snapshot_components);
  EXPECT_NE(partition.components, partition.snapshot_components);
  EXPECT_EQ(partition.components[2].value, 3u);
  EXPECT_EQ(partition.snapshot_components[1].value, 5u);
}

TEST_F(FactVectorPartitionTest, ReplacementRetargetsBothViewsAndAliases) {
  const loom_value_fact_vector_partition_component_t components[] = {
      {1, 16}, {2, 16}, {1, 16}, {2, 16}};
  const loom_value_fact_vector_partition_component_t snapshot[] = {{3, 32},
                                                                   {4, 32}};
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &table_, 7, {components, snapshot, 4, 64, 16, 2, 32}, nullptr));
  IREE_ASSERT_OK(
      loom_value_fact_table_forward_vector_partition(&table_, 7, 8, nullptr));
  const auto borrowed =
      loom_value_fact_table_query_vector_partition(&table_, 8);

  IREE_ASSERT_OK(
      loom_value_fact_table_replace_vector_partition_value(&table_, 1, 1024));
  IREE_ASSERT_OK(
      loom_value_fact_table_replace_vector_partition_value(&table_, 1024, 2));
  IREE_ASSERT_OK(
      loom_value_fact_table_replace_vector_partition_value(&table_, 3, 4));
  for (uint32_t i = 0; i < borrowed.count; ++i) {
    EXPECT_EQ(borrowed.components[i].value, 2u);
  }
  EXPECT_EQ(borrowed.snapshot_components[0].value, 4u);
  EXPECT_EQ(borrowed.snapshot_components[1].value, 4u);
  EXPECT_EQ(loom_value_fact_table_query_vector_partition(&table_, 7).components,
            borrowed.components);

  loom_value_fact_table_invalidate_vector_partition_value(&table_, 1);
  loom_value_fact_table_invalidate_vector_partition_value(&table_, 1024);
  EXPECT_EQ(loom_value_fact_table_query_vector_partition(&table_, 8).count, 4u);
  loom_value_fact_table_invalidate_vector_partition_value(&table_, 4);
  const auto exact = loom_value_fact_table_query_vector_partition(&table_, 8);
  EXPECT_EQ(exact.count, 4u);
  EXPECT_EQ(exact.snapshot_count, 0u);
  loom_value_fact_table_invalidate_vector_partition_value(&table_, 2);
  EXPECT_EQ(loom_value_fact_table_query_vector_partition(&table_, 7).count, 0u);
  EXPECT_EQ(loom_value_fact_table_query_vector_partition(&table_, 8).count, 0u);
}

TEST_F(FactVectorPartitionTest, ErasurePreservesIndependentSnapshot) {
  const loom_value_fact_vector_partition_component_t components[] = {{1, 1},
                                                                     {2, 1}};
  const loom_value_fact_vector_partition_component_t snapshot[] = {{3, 2}};
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &table_, 7, {components, snapshot, 2, 2, 0, 1, 2}, nullptr));
  loom_value_fact_table_invalidate_vector_partition_value(&table_, 1);
  const auto remaining =
      loom_value_fact_table_query_vector_partition(&table_, 7);
  EXPECT_EQ(remaining.count, 0u);
  ASSERT_EQ(remaining.snapshot_count, 1u);
  EXPECT_EQ(remaining.snapshot_components[0].value, 3u);
  // A copied fact retains a surviving snapshot even when its exact leaves
  // are no longer materializable in the source table.
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &table_, 8, remaining, nullptr));
  IREE_ASSERT_OK(
      loom_value_fact_table_replace_vector_partition_value(&table_, 3, 4));
  EXPECT_EQ(loom_value_fact_table_query_vector_partition(&table_, 8)
                .snapshot_components[0]
                .value,
            4u);
  loom_value_fact_table_invalidate_vector_partition_value(&table_, 4);
  EXPECT_EQ(
      loom_value_fact_table_query_vector_partition(&table_, 8).snapshot_count,
      0u);
}

TEST_F(FactVectorPartitionTest,
       SparseValuesPreserveBorrowedViewsAcrossGrowthAndReplacement) {
  constexpr loom_value_id_t kBase = 65536;
  constexpr loom_value_id_t kPartition = kBase + 7;
  constexpr uint32_t kAliasCount = 32;
  const loom_value_fact_vector_partition_component_t components[] = {
      {kBase + 1, 16}, {kBase + 2, 16}};
  const loom_value_fact_vector_partition_component_t snapshot[] = {
      {kBase + 3, 32}};
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &table_, kPartition, {components, snapshot, 2, 32, 16, 1, 32}, nullptr));
  const auto borrowed =
      loom_value_fact_table_query_vector_partition(&table_, kPartition);

  // A scope may mix original values and aliases created much later in the
  // module. Both alias insertion and replacement grow the index while callers
  // retain the component views, whose addresses must stay stable.
  loom_value_id_t replacement = kBase + 1;
  for (uint32_t i = 0; i < kAliasCount; ++i) {
    const loom_value_id_t alias = kBase + 4096 + i * 257;
    IREE_ASSERT_OK(loom_value_fact_table_forward_vector_partition(
        &table_, kPartition, alias, nullptr));
    const loom_value_id_t next_replacement = kBase + 32768 + i * 1021;
    IREE_ASSERT_OK(loom_value_fact_table_replace_vector_partition_value(
        &table_, replacement, next_replacement));
    loom_value_fact_table_invalidate_vector_partition_value(&table_,
                                                            replacement);
    replacement = next_replacement;
    EXPECT_EQ(borrowed.components[0].value, replacement);
    EXPECT_EQ(borrowed.snapshot_components[0].value, kBase + 3);
  }
  for (uint32_t i = 0; i < kAliasCount; ++i) {
    const loom_value_id_t alias = kBase + 4096 + i * 257;
    const auto partition =
        loom_value_fact_table_query_vector_partition(&table_, alias);
    ASSERT_EQ(partition.count, 2u);
    ASSERT_EQ(partition.snapshot_count, 1u);
    EXPECT_EQ(partition.components, borrowed.components);
    EXPECT_EQ(partition.snapshot_components, borrowed.snapshot_components);
  }
  EXPECT_EQ(loom_value_fact_table_query_vector_partition(&table_, 1).count, 0u);
  EXPECT_EQ(loom_value_fact_table_query_vector_partition(&table_,
                                                         LOOM_VALUE_ID_INVALID)
                .count,
            0u);

  loom_value_fact_table_invalidate_vector_partition_value(&table_, replacement);
  const auto remaining =
      loom_value_fact_table_query_vector_partition(&table_, kPartition);
  EXPECT_EQ(remaining.count, 0u);
  ASSERT_EQ(remaining.snapshot_count, 1u);
  EXPECT_EQ(remaining.snapshot_components, borrowed.snapshot_components);
  loom_value_fact_table_invalidate_vector_partition_value(&table_, kBase + 3);
  EXPECT_EQ(loom_value_fact_table_query_vector_partition(&table_, kPartition)
                .snapshot_count,
            0u);

  // Cleared keys are reusable, and clearing the entire scope discards every
  // alias and use chain before another scope publishes the same value IDs.
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &table_, kPartition, MakeRegularPartition(components, 2, 32, 16),
      nullptr));
  EXPECT_EQ(loom_value_fact_table_query_vector_partition(&table_, kPartition)
                .components[0]
                .value,
            kBase + 1);
  loom_value_fact_table_clear_scope(&table_);
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &table_, kPartition, MakeRegularPartition(components, 2, 32, 16),
      nullptr));
  EXPECT_EQ(
      loom_value_fact_table_query_vector_partition(&table_, kBase + 4096).count,
      0u);
  EXPECT_EQ(loom_value_fact_table_query_vector_partition(&table_, kPartition)
                .snapshot_count,
            2u);
}

TEST_F(FactVectorPartitionTest, SharedComponentsAreUpdatedOnce) {
  const loom_value_fact_vector_partition_component_t components[] = {{1, 16},
                                                                     {2, 16}};
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &table_, 7, MakeRegularPartition(components, 2, 32, 16), nullptr));
  IREE_ASSERT_OK(
      loom_value_fact_table_forward_vector_partition(&table_, 7, 8, nullptr));
  loom_value_fact_table_invalidate_vector_partition_value(&table_, 7);
  IREE_ASSERT_OK(
      loom_value_fact_table_replace_vector_partition_value(&table_, 1, 2));
  const auto remaining =
      loom_value_fact_table_query_vector_partition(&table_, 8);
  ASSERT_EQ(remaining.count, 2u);
  EXPECT_EQ(remaining.components, remaining.snapshot_components);
  EXPECT_EQ(remaining.components[0].value, 2u);
  EXPECT_EQ(remaining.components[1].value, 2u);
  loom_value_fact_table_invalidate_vector_partition_value(&table_, 2);
  const auto erased = loom_value_fact_table_query_vector_partition(&table_, 8);
  EXPECT_EQ(erased.count, 0u);
  EXPECT_EQ(erased.snapshot_count, 0u);
}

}  // namespace
}  // namespace loom
