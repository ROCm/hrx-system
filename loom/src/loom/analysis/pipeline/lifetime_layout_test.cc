// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/lifetime_layout.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class PipelineLifetimeLayoutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  // Parent-owned state coexists with two children of one serial scope.
  static constexpr auto none = IREE_HOST_SIZE_MAX;
  // Retained producer shape: every construction has a separate OWNER leaf.
  const loom_pipeline_lifetime_t nodes_[7] = {
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 0, none, 1, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 0, 0, none, 2},
      {LOOM_PIPELINE_LIFETIME_SERIAL, 0, 0, 3, none},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 1, 2, 5, 4},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 2, 2, 6, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 1, 3, none, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 2, 4, none, none},
  };
  // Direct owner identities retained by lifetime construction.
  const iree_host_size_t owners_[3] = {1, 5, 6};
  // Complete graph consumed by both storage and semaphore placement.
  const loom_pipeline_lifetimes_t lifetimes_ = {
      .nodes = nodes_,
      .count = 7,
      .owners = owners_,
      .has_serial = true,
  };
  // Backing for the retained layout result.
  iree_arena_block_pool_t blocks_;
  // Allocation owner for requests and placements.
  iree_arena_allocator_t arena_;
};

TEST_F(PipelineLifetimeLayoutTest, AlignmentAndOwnerLocalRangesAcrossPools) {
  loom_pipeline_lifetime_request_t requests[] = {
      {.lifetime = 1, .pool = 0, .length = 3, .alignment = 1},
      {.lifetime = 5, .pool = 0, .length = 5, .alignment = 4},
      {.lifetime = 6, .pool = 0, .length = 11, .alignment = 8},
      {.lifetime = 5, .pool = 0, .length = 2, .alignment = 2},
      {.lifetime = 1, .pool = 1, .length = 1, .alignment = 1},
      {.lifetime = 5, .pool = 1, .length = 3, .alignment = 8},
      {.lifetime = 6, .pool = 1, .length = 17, .alignment = 16},
  };
  const loom_pipeline_lifetime_extent_t* extents = nullptr;
  IREE_ASSERT_OK(loom_pipeline_lifetime_layout(
      &lifetimes_, requests, IREE_ARRAYSIZE(requests), 2, &arena_, &extents));
  const uint64_t offsets[] = {0, 8, 8, 14, 0, 16, 16};
  for (size_t i = 0; i < IREE_ARRAYSIZE(requests); ++i) {
    EXPECT_EQ(requests[i].offset, offsets[i]);
    EXPECT_EQ(requests[i].offset % requests[i].alignment, 0u);
  }
  EXPECT_EQ(extents[0].length, 19u);
  EXPECT_EQ(extents[0].alignment, 8u);
  EXPECT_EQ(extents[1].length, 33u);
  EXPECT_EQ(extents[1].alignment, 16u);
}

TEST_F(PipelineLifetimeLayoutTest, ASubsetKeepsTheCompleteOwnerLifetime) {
  // An unused serial child contributes no space. The selected child still
  // cannot overlap parent state, even when its request is enumerated first.
  loom_pipeline_lifetime_request_t requests[] = {
      {.lifetime = 6, .pool = 0, .length = 2, .alignment = 1},
      {.lifetime = 1, .pool = 0, .length = 1, .alignment = 1},
  };
  const loom_pipeline_lifetime_extent_t* extents = nullptr;
  IREE_ASSERT_OK(loom_pipeline_lifetime_layout(
      &lifetimes_, requests, IREE_ARRAYSIZE(requests), 1, &arena_, &extents));
  EXPECT_EQ(requests[0].offset, 1u);
  EXPECT_EQ(requests[1].offset, 0u);
  EXPECT_EQ(extents[0].length, 3u);
}

}  // namespace
