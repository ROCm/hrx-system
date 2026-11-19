// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/entries.h"

#include <array>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/pipeline/lifetime_layout.h"

namespace {

class PipelineEntriesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    IREE_ASSERT_OK(loom_pipeline_lifetimes_index(&arena_, &lifetimes_));
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  static constexpr auto none = IREE_HOST_SIZE_MAX;
  // Root owns a wrapper. Its serial A/B phases coexist with concurrent D.
  const loom_pipeline_lifetime_t nodes_[11] = {
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 0, none, 1, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 0, 0, none, 2},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 1, 0, 3, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 1, 2, none, 4},
      {LOOM_PIPELINE_LIFETIME_SERIAL, 1, 2, 5, 6},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 2, 4, 7, 8},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 4, 2, 10, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 2, 5, none, none},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 3, 4, 9, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 3, 8, none, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 4, 6, none, none},
  };
  // Construction ordinals map directly to their separate state lifetimes.
  const iree_host_size_t owners_[5] = {1, 3, 7, 9, 10};
  // Immutable construction structure plus its retained ancestry index.
  loom_pipeline_lifetimes_t lifetimes_ = {
      .nodes = nodes_,
      .count = IREE_ARRAYSIZE(nodes_),
      .owners = owners_,
      .has_serial = true,
  };
  // Direct dependencies, already ordered by the source call-graph producer.
  const iree_host_size_t edges_[9] = {0, 1, 2, 1, 3, 4, 5, 6, 1};
  // H calls G; A calls H and its local helper; B calls H; wrapper calls A/B/D.
  std::array<loom_pipeline_entry_calls_t, 8> entries_ = {{
      {.owner = none},
      {.owner = none, .callees = &edges_[0], .callee_count = 1},
      {.owner = none},
      {.owner = 2, .callees = &edges_[1], .callee_count = 2},
      {.owner = 3, .callees = &edges_[3], .callee_count = 1},
      {.owner = 4},
      {.owner = 1, .callees = &edges_[4], .callee_count = 3},
      {.owner = 0, .callees = &edges_[7], .callee_count = 1},
  }};
  // Arena blocks outlive every retained lifetime index and layout.
  iree_arena_block_pool_t blocks_;
  // Scratch owner for the analysis and physical placement under test.
  iree_arena_allocator_t arena_;
};

TEST_F(PipelineEntriesTest, SharedHelpersRetainTheirEnclosingConstruction) {
  std::array<iree_host_size_t, 8> owners;
  loom_pipeline_entry_owners_project(&lifetimes_, entries_.data(),
                                     entries_.size(), owners.data());
  EXPECT_EQ(owners, (std::array<iree_host_size_t, 8>{1, 1, 2, 2, 3, 4, 1, 0}));

  // G/H/wrapper remain resident together. A and its local helper overlay B;
  // concurrent D has separate space. The outer root stays outside both.
  const uint64_t lengths[] = {16, 32, 16, 64, 128, 32, 16, 16};
  loom_pipeline_lifetime_request_t requests[8];
  for (size_t i = 0; i < owners.size(); ++i) {
    requests[i] = {
        .lifetime = lifetimes_.owners[owners[i]],
        .pool = 0,
        .length = lengths[i],
        .alignment = 16,
    };
  }
  const loom_pipeline_lifetime_extent_t* extents = nullptr;
  IREE_ASSERT_OK(loom_pipeline_lifetime_layout(
      &lifetimes_, requests, IREE_ARRAYSIZE(requests), 1, &arena_, &extents));
  const uint64_t offsets[] = {0, 16, 64, 80, 64, 192, 48, 224};
  for (size_t i = 0; i < IREE_ARRAYSIZE(requests); ++i) {
    EXPECT_EQ(requests[i].offset, offsets[i]);
  }
  EXPECT_EQ(extents[0].length, 240u);
}

TEST_F(PipelineEntriesTest, LoopPrioritySurvivesIndependentCompilation) {
  uint32_t depths[] = {0, 0, 2, 1, 0};
  lifetimes_.loop_depths = depths;
  loom_pipeline_entry_scope_t* scope = nullptr;
  IREE_ASSERT_OK(loom_pipeline_entry_scope_allocate(
      &loom_pipeline_entry_scope_type, &lifetimes_, IREE_ARRAYSIZE(depths),
      &arena_, &scope));
  const loom_pipeline_entry_scope_t* copy = nullptr;
  IREE_ASSERT_OK(loom_pipeline_entry_scope_clone(scope, &arena_, &copy));
  EXPECT_NE(copy->lifetimes.loop_depths, scope->lifetimes.loop_depths);
  depths[2] = 0;
  EXPECT_EQ(copy->lifetimes.loop_depths[2], 2u);
  EXPECT_EQ(copy->lifetimes.loop_depths[3], 1u);
}

TEST_F(PipelineEntriesTest, RootUsePromotesTheEntireSharedHelperChain) {
  // A direct root use of H must also keep H's callee G resident at the root.
  entries_.back().callee_count = 2;
  std::array<iree_host_size_t, 8> owners;
  loom_pipeline_entry_owners_project(&lifetimes_, entries_.data(),
                                     entries_.size(), owners.data());
  EXPECT_EQ(owners, (std::array<iree_host_size_t, 8>{0, 0, 2, 2, 3, 4, 1, 0}));
}

TEST_F(PipelineEntriesTest, OrdinaryCallClosureKeepsTheRootLifetime) {
  // Removing phase boundaries makes the complete ordinary helper closure live
  // with its root, regardless of the source pipeline's unused serial regions.
  for (size_t i = 0; i + 1 < entries_.size(); ++i) {
    entries_[i].owner = none;
  }
  std::array<iree_host_size_t, 8> owners;
  loom_pipeline_entry_owners_project(&lifetimes_, entries_.data(),
                                     entries_.size(), owners.data());
  EXPECT_EQ(owners, (std::array<iree_host_size_t, 8>{0, 0, 0, 0, 0, 0, 0, 0}));
}

}  // namespace
