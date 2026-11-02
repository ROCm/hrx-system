// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/short_wait.h"

#include <memory>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/util/cfg_graph_test_util.h"

namespace loom {
namespace {

class AmdgpuShortWaitTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(16 * 1024, iree_allocator_system(),
                                     &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_amdgpu_low_descriptor_registry_initialize(&registry_);
    descriptor_set_ = loom_low_descriptor_registry_lookup(
        &registry_.registry, IREE_SV("amdgpu.rdna4.gfx125x.core"));
    ASSERT_NE(descriptor_set_, nullptr);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void Initialize(const std::vector<std::vector<uint16_t>>& edges) {
    graph_ = std::make_unique<testing::CfgGraph>(edges);
    nodes_.resize(edges.size());
    blocks_.resize(edges.size());
    order_.resize(edges.size());
    for (uint32_t i = 0; i < edges.size(); ++i) {
      nodes_[i].block_index = i;
      order_[i] = i;
      blocks_[i] = {
          .node_start = i,
          .node_count = 1,
          .scheduled_node_start = i,
          .scheduled_node_count = 1,
      };
    }
    schedule_ = {
        .target = {.descriptor_set = descriptor_set_},
        .blocks = blocks_.data(),
        .block_count = blocks_.size(),
        .cfg_graph = *graph_->get(),
        .nodes = nodes_.data(),
        .node_count = nodes_.size(),
        .scheduled_node_indices = order_.data(),
        .scheduled_node_count = order_.size(),
    };
    physical_extents_.assign(descriptor_set_->reg_class_count, 0);
    physical_extents_[LOOM_AMDGPU_REG_CLASS_ID_VGPR] = 4;
    allocation_.physical_extents = {
        .ends_by_reg_class = physical_extents_.data(),
        .count = physical_extents_.size(),
    };
    IREE_ASSERT_OK(loom_amdgpu_short_wait_allocate(
        &schedule_, &allocation_,
        LOOM_AMDGPU_PROCESSOR_SCHEDULING_FIXED_DATA_WAIT_STATES,
        LOOM_AMDGPU_DESCRIPTOR_TRAIT_DPP, /*dot_identity_mask=*/0, &arena_,
        &state_));
    ASSERT_NE(state_, nullptr);
  }

  static iree_status_t Emit(void* user_data,
                            const loom_amdgpu_wait_state_t* wait_state) {
    static_cast<AmdgpuShortWaitTest*>(user_data)->waits_.push_back(*wait_state);
    return iree_ok_status();
  }

  static constexpr loom_amdgpu_short_wait_reasons_t kLanePermute =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_LANE_PERMUTE_VGPR_READ;

  // Recycled storage for each independent analysis fixture.
  iree_arena_block_pool_t pool_;
  // Analysis lifetime reset by the fixture teardown.
  iree_arena_allocator_t arena_;
  // Canonical AMDGPU descriptor tables supplying physical register classes.
  loom_target_low_descriptor_registry_t registry_ = {};
  // Selected descriptor contract shared by the schedule and allocation.
  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
  // Concrete control-flow graph backing the borrowed schedule view.
  std::unique_ptr<testing::CfgGraph> graph_;
  // One synthetic packet node per block.
  std::vector<loom_low_schedule_node_t> nodes_;
  // Schedule block ranges matching the synthetic packet nodes.
  std::vector<loom_low_schedule_block_t> blocks_;
  // Packet order grouped by source block.
  std::vector<uint32_t> order_;
  // Physical register high-water marks indexed by descriptor register class.
  std::vector<uint32_t> physical_extents_;
  // Minimal production schedule consumed by the short-wait owner.
  loom_low_schedule_table_t schedule_ = {};
  // Minimal physical allocation consumed by the short-wait owner.
  loom_low_allocation_table_t allocation_ = {};
  // Arena-owned short-window analysis under test.
  loom_amdgpu_short_wait_t* state_ = nullptr;
  // Resolved cross-block waits emitted in final packet order.
  std::vector<loom_amdgpu_wait_state_t> waits_;
};

TEST_F(AmdgpuShortWaitTest, DrainReturnsTheStrongestLocalWindowAndResetsIt) {
  Initialize({{}});
  loom_amdgpu_short_wait_begin_block(state_, 0);
  loom_amdgpu_short_wait_record(
      state_, /*producer_node=*/0, LOOM_AMDGPU_REG_CLASS_ID_VGPR, /*base=*/0,
      /*count=*/1, /*instruction_end_offset=*/1, kLanePermute, /*cycles=*/2);
  loom_amdgpu_short_wait_advance(state_, 1);
  const loom_low_packet_view_t packet = loom_low_packet_at(&schedule_, 0);
  const loom_amdgpu_short_wait_match_t match =
      loom_amdgpu_short_wait_drain(state_, &packet, /*instruction_offset=*/1);
  EXPECT_EQ(match.reason, LOOM_AMDGPU_WAIT_STATE_REASON_LANE_PERMUTE_VGPR_READ);
  EXPECT_EQ(match.producer_node, 0u);
  EXPECT_EQ(match.required_cycles, 2u);
  EXPECT_EQ(match.observed_cycles, 0u);
  EXPECT_EQ(match.cycles, 2u);
  EXPECT_EQ(loom_amdgpu_short_wait_query(
                state_, LOOM_AMDGPU_REG_CLASS_ID_VGPR, /*base=*/0,
                /*count=*/1, /*instruction_offset=*/0, kLanePermute)
                .cycles,
            0u);
}

TEST_F(AmdgpuShortWaitTest, DrainPlacesIncomingWaitAtTheInternalBoundary) {
  Initialize({{1}, {}});
  loom_amdgpu_short_wait_begin_block(state_, 0);
  loom_amdgpu_short_wait_record(
      state_, /*producer_node=*/0, LOOM_AMDGPU_REG_CLASS_ID_VGPR, /*base=*/0,
      /*count=*/1, /*instruction_end_offset=*/1, kLanePermute, /*cycles=*/2);
  loom_amdgpu_short_wait_advance(state_, 1);
  IREE_ASSERT_OK(loom_amdgpu_short_wait_end_block(state_));

  loom_amdgpu_short_wait_begin_block(state_, 1);
  const loom_low_packet_view_t packet = loom_low_packet_at(&schedule_, 1);
  EXPECT_EQ(
      loom_amdgpu_short_wait_drain(state_, &packet, /*instruction_offset=*/3)
          .cycles,
      0u);
  IREE_ASSERT_OK(loom_amdgpu_short_wait_end_block(state_));
  IREE_ASSERT_OK(loom_amdgpu_short_wait_resolve(state_, Emit, this));

  ASSERT_EQ(waits_.size(), 1u);
  EXPECT_EQ(waits_[0].reason,
            LOOM_AMDGPU_WAIT_STATE_REASON_LANE_PERMUTE_VGPR_READ);
  EXPECT_EQ(waits_[0].action, LOOM_AMDGPU_WAIT_STATE_ACTION_S_NOP);
  EXPECT_EQ(waits_[0].block_index, 1u);
  EXPECT_EQ(waits_[0].node_index, 1u);
  EXPECT_EQ(waits_[0].instruction_offset, 3u);
  EXPECT_EQ(waits_[0].required_cycle_count, 2u);
  EXPECT_EQ(waits_[0].observed_cycle_count, 0u);
  EXPECT_EQ(waits_[0].cycle_count, 2u);
}

TEST_F(AmdgpuShortWaitTest, DrainRetainsOnlyThePostBoundarySuffix) {
  Initialize({{1}, {}});
  loom_amdgpu_short_wait_begin_block(state_, 0);
  loom_amdgpu_short_wait_record(
      state_, /*producer_node=*/0, LOOM_AMDGPU_REG_CLASS_ID_VGPR, /*base=*/0,
      /*count=*/1, /*instruction_end_offset=*/1, kLanePermute, /*cycles=*/2);
  loom_amdgpu_short_wait_advance(state_, 1);
  const loom_low_packet_view_t first_packet = loom_low_packet_at(&schedule_, 0);
  EXPECT_EQ(loom_amdgpu_short_wait_drain(state_, &first_packet,
                                         /*instruction_offset=*/1)
                .cycles,
            2u);
  loom_amdgpu_short_wait_record(
      state_, /*producer_node=*/0, LOOM_AMDGPU_REG_CLASS_ID_VGPR, /*base=*/1,
      /*count=*/1, /*instruction_end_offset=*/1, kLanePermute, /*cycles=*/2);
  loom_amdgpu_short_wait_advance(state_, 1);
  IREE_ASSERT_OK(loom_amdgpu_short_wait_end_block(state_));

  loom_amdgpu_short_wait_begin_block(state_, 1);
  const loom_low_packet_view_t second_packet =
      loom_low_packet_at(&schedule_, 1);
  EXPECT_EQ(loom_amdgpu_short_wait_drain(state_, &second_packet,
                                         /*instruction_offset=*/0)
                .cycles,
            0u);
  IREE_ASSERT_OK(loom_amdgpu_short_wait_end_block(state_));
  IREE_ASSERT_OK(loom_amdgpu_short_wait_resolve(state_, Emit, this));

  ASSERT_EQ(waits_.size(), 1u);
  EXPECT_EQ(waits_[0].node_index, 1u);
  EXPECT_EQ(waits_[0].instruction_offset, 0u);
  EXPECT_EQ(waits_[0].cycle_count, 2u);
}

}  // namespace
}  // namespace loom
