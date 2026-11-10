// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/semaphores.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/pipeline/invocations.h"
#include "loom/pass/types.h"
#include "loom/target/arch/amd/xdna/array/facts.h"

namespace {

class Aie2pSemaphoresTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    pass_.arena = &arena_;
    context_.pass = &pass_;
    context_.channels = selected_;
    context_.pool_tiles = pools_;
    for (size_t i = 0; i < 3; ++i) {
      occurrences_[i].ordinal = i;
      sources_[i].capacity = 3;
      channels_[i].invocation = &occurrences_[i];
      channels_[i].source = &sources_[i];
      selected_[i].source = &sources_[i];
    }
    realization_.invocations.channels = channels_;
    realization_.invocations.channel_count = IREE_ARRAYSIZE(channels_);
    realization_.invocations.lifetimes = {
        .nodes = nodes_,
        .count = 7,
        .owners = owners_,
        .has_serial = true,
    };
    for (uint16_t i = 0; i < 2; ++i) {
      tiles_[i].coordinate = {0, static_cast<uint16_t>(2 + i)};
      tiles_[i].facts = loom_xdna_array_tile_facts(
          loom_xdna_npu2_array_family(), tiles_[i].coordinate);
      tiles_[i].next_lock = 3;
    }
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }
  void Assign() {
    bool valid = false;
    IREE_ASSERT_OK(loom_aie2p_native_assign_channel_locks(
        &context_, &realization_, &valid));
    ASSERT_TRUE(valid);
  }

  // Root state coexists with both children, which share one serial scope.
  static constexpr auto none = IREE_HOST_SIZE_MAX;
  // Parent-before-child lifetime nodes from the common producer contract.
  loom_pipeline_lifetime_t nodes_[7] = {
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 0, none, 1, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 0, 0, none, 2},
      {LOOM_PIPELINE_LIFETIME_SERIAL, 0, 0, 3, none},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 1, 2, 5, 4},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 2, 2, 6, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 1, 3, none, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 2, 4, none, none},
  };
  // Owner nodes indexed by original invocation order.
  const iree_host_size_t owners_[3] = {1, 5, 6};
  // Construction identities referenced by the channel inventory.
  loom_pipeline_invocation_t occurrences_[3] = {};
  // Logical record capacities, separate from compressed physical credits.
  loom_pipeline_resource_channel_t sources_[3] = {};
  // Bound channel identities and their construction owners.
  loom_pipeline_invocation_channel_t channels_[3] = {};
  // Target-owned physical protocol selections.
  loom_aie2p_native_channel_t selected_[3] = {};
  // Physical semaphore banks with independently allocated IDs.
  loom_aie2p_native_tile_t tiles_[2] = {};
  // Canonical tile selection for each backing pool.
  loom_aie2p_native_tile_t* pools_[2] = {&tiles_[0], &tiles_[1]};
  // Complete lifetime and channel input consumed by lock assignment.
  loom_pipeline_realization_t realization_ = {};
  // Pass allocation context; successful placement emits no diagnostics.
  loom_pass_t pass_ = {};
  // Selected native protocol state.
  loom_aie2p_native_context_t context_ = {};
  // Backing for arena-owned layout results.
  iree_arena_block_pool_t blocks_;
  // Allocation owner for one test's selections.
  iree_arena_allocator_t arena_;
};

TEST_F(Aie2pSemaphoresTest, CompatibilityUsesPhysicalCreditsOnEachTileKind) {
  sources_[2].capacity = 96;
  for (const uint16_t row : {1, 2}) {
    SCOPED_TRACE(row);
    tiles_[0].coordinate = {0, row};
    tiles_[0].facts = loom_xdna_array_tile_facts(loom_xdna_npu2_array_family(),
                                                 tiles_[0].coordinate);
    for (const uint32_t credits : {1, 3}) {
      SCOPED_TRACE(credits);
      tiles_[0].next_lock = 3;
      selected_[2].replay = {.record_count = 96, .credits = credits};
      Assign();
      EXPECT_EQ(selected_[0].free_lock, 3u);
      EXPECT_EQ(selected_[1].free_lock, 5u);
      EXPECT_EQ(selected_[2].free_lock, credits == 3 ? 5u : 7u);
      EXPECT_EQ(selected_[2].ready_lock, credits == 3 ? 6u : 8u);
      EXPECT_TRUE(selected_[0].initializes_locks);
      EXPECT_TRUE(selected_[1].initializes_locks);
      EXPECT_EQ(selected_[2].initializes_locks, credits != 3);
      EXPECT_EQ(tiles_[0].next_lock, credits == 3 ? 7u : 9u);
    }
  }
}

TEST_F(Aie2pSemaphoresTest, ConcurrentChildrenHaveDistinctPairs) {
  nodes_[2].kind = LOOM_PIPELINE_LIFETIME_CONCURRENT;
  realization_.invocations.lifetimes.has_serial = false;
  Assign();
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(selected_[i].free_lock, 3u + 2 * i);
    EXPECT_TRUE(selected_[i].initializes_locks);
  }
  EXPECT_EQ(tiles_[0].next_lock, 9u);
}

TEST_F(Aie2pSemaphoresTest, DescendantsCoexistWithTheirOwnersInSerialScopes) {
  // The second child now belongs to the first child's concurrent region.
  // An enclosing serial scope cannot make these overlapping lifetimes alias.
  nodes_[3].next_sibling = none;
  nodes_[4].parent = 3;
  nodes_[5].next_sibling = 4;
  occurrences_[2].parent = &occurrences_[1];
  Assign();
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(selected_[i].free_lock, 3u + 2 * i);
    EXPECT_TRUE(selected_[i].initializes_locks);
  }
  EXPECT_EQ(tiles_[0].next_lock, 9u);
}

TEST_F(Aie2pSemaphoresTest, CompatibleCreditsRemainInTheirPhysicalBank) {
  selected_[1].pool_index = 1;
  tiles_[1].next_lock = 9;
  Assign();
  EXPECT_EQ(selected_[0].free_lock, 3u);
  EXPECT_EQ(selected_[1].free_lock, 9u);
  EXPECT_EQ(selected_[2].free_lock, 5u);
  EXPECT_EQ(tiles_[0].next_lock, 7u);
  EXPECT_EQ(tiles_[1].next_lock, 11u);
  for (const auto& channel : selected_) {
    EXPECT_TRUE(channel.initializes_locks);
  }
}

}  // namespace
