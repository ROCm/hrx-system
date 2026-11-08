// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/ordinal_component.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class LowAllocationOrdinalComponentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    iree_arena_initialize(&pool_, &scratch_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  iree_arena_block_pool_t pool_;
  iree_arena_allocator_t arena_;
  iree_arena_allocator_t scratch_arena_;
};

TEST_F(LowAllocationOrdinalComponentTest,
       BuildsTransitiveHardComponentsAndRetainsAliases) {
  const loom_value_ordinal_t storage_origins[] = {0, 0, 2, 3, 4, 5, 6};
  const loom_low_placement_relation_t relations[] = {
      {
          .result_ordinal = 1,
          .source_ordinal = 2,
          .kind = LOOM_LOW_PLACEMENT_RELATION_SAME_REGISTER_ORDINAL,
          .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD,
      },
      {
          .result_ordinal = 2,
          .source_ordinal = 3,
          .kind = LOOM_LOW_PLACEMENT_RELATION_SAME_REGISTER_ORDINAL,
          .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD,
      },
      {
          .result_ordinal = 3,
          .source_ordinal = 4,
          .kind = LOOM_LOW_PLACEMENT_RELATION_SAME_REGISTER_ORDINAL,
          .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED,
      },
      {
          .result_ordinal = 3,
          .source_ordinal = 6,
          .kind = LOOM_LOW_PLACEMENT_RELATION_DISJOINT_STORAGE,
          .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD,
      },
  };

  loom_low_allocation_ordinal_components_t components = {};
  IREE_ASSERT_OK(loom_low_allocation_ordinal_components_build(
      IREE_ARRAYSIZE(storage_origins), storage_origins, relations,
      IREE_ARRAYSIZE(relations), &arena_, &scratch_arena_, &components));

  ASSERT_EQ(components.count, 1u);
  ASSERT_NE(components.indices_by_value_ordinal, nullptr);
  EXPECT_EQ(components.indices_by_value_ordinal[0], 0u);
  EXPECT_EQ(components.indices_by_value_ordinal[1], 0u);
  EXPECT_EQ(components.indices_by_value_ordinal[2], 0u);
  EXPECT_EQ(components.indices_by_value_ordinal[3], 0u);
  EXPECT_EQ(components.indices_by_value_ordinal[4], UINT32_MAX);
  EXPECT_EQ(components.indices_by_value_ordinal[5], UINT32_MAX);
  EXPECT_EQ(components.indices_by_value_ordinal[6], UINT32_MAX);
  ASSERT_EQ(components.ranges[0].start, 0u);
  ASSERT_EQ(components.ranges[0].count, 4u);
  ASSERT_EQ(components.ranges[0].origin_count, 3u);
  EXPECT_THAT(std::vector<loom_value_ordinal_t>(components.members,
                                                components.members + 4),
              ::testing::ElementsAre(0, 2, 3, 1));

  const loom_low_placement_table_t placement = {
      .value_count = IREE_ARRAYSIZE(storage_origins),
      .hard_register_ordinal_components_by_value_ordinal =
          components.indices_by_value_ordinal,
      .hard_register_ordinal_ranges = components.ranges,
      .hard_register_ordinal_members = components.members,
      .hard_register_ordinal_component_count = components.count,
  };
  const auto component =
      loom_low_placement_hard_register_ordinal_component(&placement, 1);
  EXPECT_EQ(component.index, 0u);
  EXPECT_EQ(component.count, 4u);
  EXPECT_EQ(component.origin_count, 3u);
  EXPECT_EQ(component.members, components.members);
  const auto unconstrained =
      loom_low_placement_hard_register_ordinal_component(&placement, 4);
  EXPECT_EQ(unconstrained.index, UINT32_MAX);
  EXPECT_EQ(unconstrained.count, 0u);
  EXPECT_EQ(unconstrained.members, nullptr);
}

TEST_F(LowAllocationOrdinalComponentTest,
       RetainsDistinctOriginsBeforeLongTiedChains) {
  constexpr uint32_t kAliasCount = 512;
  std::vector<loom_value_ordinal_t> origins(5 + kAliasCount);
  for (uint32_t i = 0; i < 5; ++i) {
    origins[i] = i;
  }
  for (uint32_t i = 5; i < origins.size(); ++i) {
    origins[i] = i % 2;
  }
  const loom_low_placement_relation_t relations[] = {
      {
          .result_ordinal = 0,
          .source_ordinal = 1,
          .kind = LOOM_LOW_PLACEMENT_RELATION_SAME_REGISTER_ORDINAL,
          .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD,
      },
      {
          .result_ordinal = 2,
          .source_ordinal = 3,
          .kind = LOOM_LOW_PLACEMENT_RELATION_SAME_REGISTER_ORDINAL,
          .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD,
      },
  };
  loom_low_allocation_ordinal_components_t components = {};
  IREE_ASSERT_OK(loom_low_allocation_ordinal_components_build(
      origins.size(), origins.data(), relations, IREE_ARRAYSIZE(relations),
      &arena_, &scratch_arena_, &components));

  ASSERT_EQ(components.count, 2u);
  const auto first = components.ranges[components.indices_by_value_ordinal[0]];
  ASSERT_EQ(first.origin_count, 2u);
  ASSERT_EQ(first.count, 2u + kAliasCount);
  EXPECT_EQ(components.members[first.start], 0u);
  EXPECT_EQ(components.members[first.start + 1], 1u);
  for (uint32_t i = 0; i < kAliasCount; ++i) {
    EXPECT_EQ(components.members[first.start + 2 + i], 5u + i);
    EXPECT_EQ(components.indices_by_value_ordinal[5 + i],
              components.indices_by_value_ordinal[0]);
  }
  const auto second = components.ranges[components.indices_by_value_ordinal[2]];
  ASSERT_EQ(second.origin_count, 2u);
  ASSERT_EQ(second.count, 2u);
  EXPECT_EQ(components.members[second.start], 2u);
  EXPECT_EQ(components.members[second.start + 1], 3u);
  EXPECT_EQ(components.indices_by_value_ordinal[4], UINT32_MAX);
}

}  // namespace
}  // namespace loom
