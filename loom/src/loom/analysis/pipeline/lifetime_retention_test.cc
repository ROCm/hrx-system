// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/lifetime_retention.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class PipelineLifetimeRetentionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    IREE_ASSERT_OK(loom_pipeline_lifetimes_index(&arena_, &lifetimes_));
    IREE_ASSERT_OK(loom_pipeline_lifetime_projection_prepare(
        &lifetimes_, requests_, 3, &arena_, &projection_));
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  // A parent resource coexists with either of two serial child resources.
  static constexpr auto none = IREE_HOST_SIZE_MAX;
  // Indexed producer tree with one OWNER leaf per construction.
  const loom_pipeline_lifetime_t nodes_[7] = {
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 0, none, 1, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 0, 0, none, 2},
      {LOOM_PIPELINE_LIFETIME_SERIAL, 0, 0, 3, none},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 1, 2, 5, 4},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 2, 2, 6, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 1, 3, none, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 2, 4, none, none},
  };
  // Complete lifetimes retained throughout numeric proposals.
  loom_pipeline_lifetimes_t lifetimes_ = {.nodes = nodes_, .count = 7};
  // Stable parent/child resource identities; selection supplies their sizes.
  loom_pipeline_lifetime_request_t requests_[3] = {
      {.lifetime = 1}, {.lifetime = 5}, {.lifetime = 6}};
  // Shared topology reused by selection and exact final placement.
  loom_pipeline_lifetime_projection_t* projection_ = nullptr;
  // Allocation backing for the fixture's retained facts.
  iree_arena_block_pool_t blocks_;
  // Lifetime owner for projections and selection scratch.
  iree_arena_allocator_t arena_;
};

TEST_F(PipelineLifetimeRetentionTest, SharesCopiesAndPreservesActiveGroups) {
  const iree_host_size_t root[] = {0}, first[] = {1}, second[] = {2};
  const iree_host_size_t shared[] = {1, 2}, already_resident[] = {0, 1};
  const loom_pipeline_lifetime_retention_item_t items[] = {
      {2, root, 1},   {4, shared, 2},           {3, first, 1},
      {5, second, 1}, {1, already_resident, 2}, {},
  };
  const struct {
    // Capacity permits none, some, or all of the candidate promotions.
    uint64_t capacity;
    // Active groups can require nonempty storage even when it all fits.
    uint64_t minimum_length;
    // Selected item ordinals; unused input slots remain unselected.
    bool retained[6];
    // Final group demands, including the existing resident copy.
    uint64_t lengths[3];
  } cases[] = {
      {0, 1, {false, false, false, false, false, false}, {3, 8, 9}},
      {12, 1, {false, true, false, false, true, false}, {7, 3, 5}},
      {15, 1, {false, true, true, false, false, false}, {10, 1, 5}},
      {16, 0, {false, true, true, true, true, false}, {15, 0, 0}},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.capacity);
    SCOPED_TRACE(test_case.minimum_length);
    bool retained[6];
    const loom_pipeline_lifetime_retention_policy_t policy = {
        .resident_group = 0,
        .minimum_group_length = test_case.minimum_length,
        .capacity = test_case.capacity,
    };
    IREE_ASSERT_OK(loom_pipeline_lifetime_retain(
        projection_, requests_, 3, items, 6, &policy, &arena_, retained));
    for (size_t i = 0; i < 6; ++i) {
      EXPECT_EQ(retained[i], test_case.retained[i]);
    }
    for (size_t i = 0; i < 3; ++i) {
      EXPECT_EQ(requests_[i].length, test_case.lengths[i]);
    }
  }
}

TEST_F(PipelineLifetimeRetentionTest, EqualSavingsUseStableInputPriority) {
  const iree_host_size_t root[] = {0}, first[] = {1}, second[] = {2};
  const loom_pipeline_lifetime_retention_item_t items[] = {
      {2, root, 1}, {4, first, 1}, {4, first, 1}, {2, first, 1}, {8, second, 1},
  };
  const loom_pipeline_lifetime_retention_policy_t policy = {
      .resident_group = 0,
      .minimum_group_length = 1,
      .capacity = 14,
  };
  bool retained[5];
  IREE_ASSERT_OK(loom_pipeline_lifetime_retain(projection_, requests_, 3, items,
                                               5, &policy, &arena_, retained));
  const bool expected[] = {false, true, false, false, false};
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(retained[i], expected[i]);
  }
  EXPECT_EQ(requests_[0].length, 6u);
  EXPECT_EQ(requests_[1].length, 6u);
  EXPECT_EQ(requests_[2].length, 8u);
  // Final exact placement uses the same topology after rejected proposals.
  // A target may replace conservative allocation units with exact byte sizes.
  requests_[0].length = 21;
  requests_[0].alignment = 8;
  requests_[1].length = 23;
  requests_[1].alignment = 4;
  requests_[2].length = 31;
  requests_[2].alignment = 16;
  loom_pipeline_lifetime_extent_t extent;
  IREE_ASSERT_OK(loom_pipeline_lifetime_projection_evaluate(
      projection_, requests_, &extent));
  EXPECT_EQ(extent.length, 63u);
  EXPECT_EQ(extent.alignment, 16u);
  EXPECT_EQ(requests_[0].offset, 0u);
  EXPECT_EQ(requests_[1].offset, 32u);
  EXPECT_EQ(requests_[2].offset, 32u);
}

}  // namespace
