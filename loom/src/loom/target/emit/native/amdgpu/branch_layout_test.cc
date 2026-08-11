// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/branch_layout.h"

#include <array>
#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

static loom_amdgpu_branch_layout_block_t BlockAt(uint64_t byte_offset) {
  return {/*byte_offset=*/byte_offset};
}

static loom_amdgpu_branch_layout_input_edge_t EdgeTo(
    uint64_t source_byte_offset, uint32_t target_block_index) {
  return {/*source_byte_offset=*/source_byte_offset,
          /*target_block_index=*/target_block_index};
}

static loom_amdgpu_branch_layout_anchor_t AnchorAt(uint64_t byte_offset,
                                                   uint32_t packet_index) {
  return {/*byte_offset=*/byte_offset, /*packet_index=*/packet_index};
}

class AmdgpuBranchLayoutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  template <size_t BlockCount, size_t EdgeCount, size_t AnchorCount>
  loom_amdgpu_branch_layout_t Build(
      uint64_t byte_length,
      const std::array<loom_amdgpu_branch_layout_block_t, BlockCount>& blocks,
      const std::array<loom_amdgpu_branch_layout_input_edge_t, EdgeCount>&
          edges,
      const std::array<loom_amdgpu_branch_layout_anchor_t, AnchorCount>&
          anchors) {
    const loom_amdgpu_branch_layout_input_t input = {
        /*byte_length=*/byte_length,
        /*blocks=*/blocks.data(),
        /*block_count=*/blocks.size(),
        /*edges=*/edges.data(),
        /*edge_count=*/edges.size(),
        /*anchors=*/anchors.data(),
        /*anchor_count=*/anchors.size(),
    };
    loom_amdgpu_branch_layout_t layout = {};
    IREE_EXPECT_OK(loom_amdgpu_branch_layout_build(&input, &arena_, &layout));
    return layout;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(AmdgpuBranchLayoutTest, LeavesSignedRangeBoundariesUnchanged) {
  constexpr uint64_t kForwardTarget = 4u + uint64_t{INT16_MAX} * 4u;
  constexpr uint64_t kBackwardSource = uint64_t{INT16_MAX} * 4u;
  const std::array blocks = {
      BlockAt(0),
      BlockAt(kForwardTarget),
  };
  const std::array edges = {
      EdgeTo(/*source_byte_offset=*/0, /*target_block_index=*/1),
      EdgeTo(/*source_byte_offset=*/kBackwardSource,
             /*target_block_index=*/0),
  };
  const std::array<loom_amdgpu_branch_layout_anchor_t, 0> anchors = {};

  const loom_amdgpu_branch_layout_t layout =
      Build(kForwardTarget + 4u, blocks, edges, anchors);
  EXPECT_EQ(layout.island_count, 0u);
  EXPECT_EQ(layout.group_count, 0u);
  EXPECT_EQ(layout.edge_count, 0u);
}

TEST_F(AmdgpuBranchLayoutTest, RelaxesJustOutsideForwardRange) {
  constexpr uint64_t kTarget = 4u + (uint64_t{INT16_MAX} + 1u) * 4u;
  const std::array blocks = {
      BlockAt(0),
      BlockAt(kTarget),
  };
  const std::array edges = {
      EdgeTo(/*source_byte_offset=*/0, /*target_block_index=*/1),
  };
  const std::array anchors = {
      AnchorAt(/*byte_offset=*/(kTarget / 8u) * 4u, /*packet_index=*/7),
  };

  const loom_amdgpu_branch_layout_t layout =
      Build(kTarget + 4u, blocks, edges, anchors);
  ASSERT_EQ(layout.edge_count, 1u);
  ASSERT_EQ(layout.island_count, 1u);
  ASSERT_EQ(layout.group_count, 1u);
  EXPECT_EQ(layout.byte_length, kTarget + 12u);
  EXPECT_EQ(layout.groups[0].packet_index, 7u);
  EXPECT_EQ(layout.groups[0].island_count, 1u);
  EXPECT_EQ(layout.edges[0].target.kind, LOOM_AMDGPU_BRANCH_TARGET_ISLAND);
  EXPECT_EQ(layout.edges[0].target.index, 0u);
  EXPECT_EQ(layout.islands[0].target.kind, LOOM_AMDGPU_BRANCH_TARGET_BLOCK);
  EXPECT_EQ(layout.islands[0].target.index, 1u);
  EXPECT_EQ(layout.edges[0].relative_dword_offset, 16384);
  EXPECT_EQ(layout.islands[0].relative_dword_offset, 16385);
}

TEST_F(AmdgpuBranchLayoutTest, RelaxesJustOutsideBackwardRange) {
  constexpr uint64_t kSource = (uint64_t{-INT16_MIN} + 1u) * 4u - 4u;
  const std::array blocks = {
      BlockAt(0),
  };
  const std::array edges = {
      EdgeTo(/*source_byte_offset=*/kSource, /*target_block_index=*/0),
  };
  const std::array anchors = {
      AnchorAt(/*byte_offset=*/kSource / 2u, /*packet_index=*/11),
  };

  const loom_amdgpu_branch_layout_t layout =
      Build(kSource + 4u, blocks, edges, anchors);
  ASSERT_EQ(layout.edge_count, 1u);
  ASSERT_EQ(layout.island_count, 1u);
  EXPECT_EQ(layout.edges[0].relative_dword_offset, -16386);
  EXPECT_EQ(layout.islands[0].relative_dword_offset, -16386);
}

TEST_F(AmdgpuBranchLayoutTest, RecomputesDirectSiblingEdgeAfterInsertion) {
  constexpr uint64_t kFarTarget = 4u + (uint64_t{INT16_MAX} + 1u) * 4u;
  constexpr uint64_t kNearTarget = 70u * 1024u;
  const std::array blocks = {
      BlockAt(0),
      BlockAt(kNearTarget),
      BlockAt(kFarTarget),
  };
  const std::array edges = {
      EdgeTo(/*source_byte_offset=*/0, /*target_block_index=*/2),
      EdgeTo(/*source_byte_offset=*/4, /*target_block_index=*/1),
  };
  const std::array anchors = {
      AnchorAt(/*byte_offset=*/(kFarTarget / 8u) * 4u,
               /*packet_index=*/5),
  };

  const loom_amdgpu_branch_layout_t layout =
      Build(kFarTarget + 4u, blocks, edges, anchors);
  ASSERT_EQ(layout.edge_count, 2u);
  ASSERT_EQ(layout.island_count, 1u);
  EXPECT_EQ(layout.edges[1].target.kind, LOOM_AMDGPU_BRANCH_TARGET_BLOCK);
  EXPECT_EQ(layout.edges[1].target.index, 1u);
  EXPECT_EQ(layout.edges[1].relative_dword_offset, 17920);
}

TEST_F(AmdgpuBranchLayoutTest, BuildsConvergedMultiHopPath) {
  constexpr uint64_t kTarget = 1024u * 1024u;
  const std::array blocks = {
      BlockAt(0),
      BlockAt(kTarget),
  };
  const std::array edges = {
      EdgeTo(/*source_byte_offset=*/0, /*target_block_index=*/1),
  };
  std::array<loom_amdgpu_branch_layout_anchor_t, 15> anchors = {};
  for (uint32_t i = 0; i < anchors.size(); ++i) {
    anchors[i] =
        AnchorAt(uint64_t{i + 1u} * 64u * 1024u, /*packet_index=*/i + 1u);
  }

  const loom_amdgpu_branch_layout_t layout =
      Build(kTarget + 4u, blocks, edges, anchors);
  ASSERT_GT(layout.island_count, 1u);
  for (iree_host_size_t i = 0; i < layout.edge_count; ++i) {
    EXPECT_GE(layout.edges[i].relative_dword_offset, INT16_MIN);
    EXPECT_LE(layout.edges[i].relative_dword_offset, INT16_MAX);
  }
  for (iree_host_size_t i = 0; i < layout.island_count; ++i) {
    EXPECT_GE(layout.islands[i].relative_dword_offset, INT16_MIN);
    EXPECT_LE(layout.islands[i].relative_dword_offset, INT16_MAX);
  }
}

}  // namespace
}  // namespace loom
