// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packet_progress.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

enum SyntheticProgressClass {
  kSyntheticProgressPipe = 7,
  kSyntheticProgressScoreboard = 8,
};

struct PacketProgressTestState {
  loom_low_descriptor_t descriptor = {};
  loom_low_descriptor_set_t descriptor_set = {};
  loom_module_t module = {};
  loom_op_t function_op = {};
  loom_region_t region = {};
  loom_block_t block = {};
  loom_block_t* region_blocks[1] = {};
  loom_low_schedule_block_t blocks[1] = {};
  loom_low_schedule_node_t nodes[2] = {};
  uint32_t scheduled_node_indices[2] = {};
  loom_low_schedule_table_t schedule = {};
  loom_low_allocation_table_t allocation = {};
};

class LowPacketProgressTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    InitializePacketProgressTestState(&state_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  static void InitializePacketProgressTestState(
      PacketProgressTestState* state) {
    state->descriptor_set.descriptors = &state->descriptor;
    state->descriptor_set.descriptor_count = 1;

    state->region_blocks[0] = &state->block;
    state->region.block_count = IREE_ARRAYSIZE(state->region_blocks);
    state->region.block_capacity = IREE_ARRAYSIZE(state->region_blocks);
    state->region.blocks = state->region_blocks;
    state->block.parent_region = &state->region;
    state->block.region_index = 0;

    state->blocks[0].block = &state->block;
    state->blocks[0].node_start = 0;
    state->blocks[0].node_count = IREE_ARRAYSIZE(state->nodes);
    state->blocks[0].scheduled_node_start = 0;
    state->blocks[0].scheduled_node_count =
        IREE_ARRAYSIZE(state->scheduled_node_indices);

    state->nodes[0].block = &state->block;
    state->nodes[0].block_index = 0;
    state->nodes[0].source_ordinal = 0;
    state->nodes[0].scheduled_ordinal = 0;
    state->nodes[0].kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR;
    state->nodes[0].descriptor = &state->descriptor;
    state->nodes[1].block = &state->block;
    state->nodes[1].block_index = 0;
    state->nodes[1].source_ordinal = 1;
    state->nodes[1].scheduled_ordinal = 1;
    state->nodes[1].kind = LOOM_LOW_SCHEDULE_NODE_STRUCTURAL;

    state->scheduled_node_indices[0] = 0;
    state->scheduled_node_indices[1] = 1;

    state->schedule.module = &state->module;
    state->schedule.function_op = &state->function_op;
    state->schedule.target.descriptor_set = &state->descriptor_set;
    state->schedule.blocks = state->blocks;
    state->schedule.block_count = IREE_ARRAYSIZE(state->blocks);
    state->schedule.nodes = state->nodes;
    state->schedule.node_count = IREE_ARRAYSIZE(state->nodes);
    state->schedule.scheduled_node_indices = state->scheduled_node_indices;
    state->schedule.scheduled_node_count =
        IREE_ARRAYSIZE(state->scheduled_node_indices);

    state->allocation.module = &state->module;
    state->allocation.function_op = &state->function_op;
    state->allocation.target.descriptor_set = &state->descriptor_set;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  PacketProgressTestState state_;
};

iree_status_t EmitEvent(loom_low_packet_progress_emit_fn_t emit,
                        void* emit_user_data, uint16_t progress_class_id,
                        iree_string_view_t progress_class_name,
                        loom_low_packet_progress_action_t action,
                        uint32_t units) {
  const loom_low_packet_progress_event_t event = {
      /*.progress_class_id=*/progress_class_id,
      /*.progress_class_name=*/progress_class_name,
      /*.action=*/action,
      /*.units=*/units,
  };
  return emit(emit_user_data, &event);
}

iree_status_t SyntheticProgressQuery(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_view_t* packet,
    loom_low_packet_progress_emit_fn_t emit, void* emit_user_data) {
  (void)user_data;
  (void)schedule;
  (void)allocation;
  if (packet->node_index == 0) {
    return EmitEvent(emit, emit_user_data, kSyntheticProgressPipe,
                     IREE_SV("synthetic.pipe"),
                     LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE, 2);
  }
  IREE_RETURN_IF_ERROR(EmitEvent(emit, emit_user_data,
                                 kSyntheticProgressScoreboard,
                                 IREE_SV("synthetic.scoreboard"),
                                 LOOM_LOW_PACKET_PROGRESS_ACTION_RESET, 0));
  return EmitEvent(emit, emit_user_data, kSyntheticProgressPipe,
                   IREE_SV("synthetic.pipe"),
                   LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE, 1);
}

iree_status_t EmptyProgressQuery(void* user_data,
                                 const loom_low_schedule_table_t* schedule,
                                 const loom_low_allocation_table_t* allocation,
                                 const loom_low_packet_view_t* packet,
                                 loom_low_packet_progress_emit_fn_t emit,
                                 void* emit_user_data) {
  (void)user_data;
  (void)schedule;
  (void)allocation;
  (void)packet;
  (void)emit;
  (void)emit_user_data;
  return iree_ok_status();
}

TEST_F(LowPacketProgressTest, BuildsSyntheticTargetProgressRecords) {
  const loom_low_packet_progress_provider_t provider = {
      /*.user_data=*/{},
      /*.query=*/SyntheticProgressQuery,
  };
  loom_low_packet_progress_table_t table = {};
  IREE_ASSERT_OK(loom_low_packet_progress_build(
      &state_.schedule, &state_.allocation, &provider, &arena_, &table));

  ASSERT_EQ(table.schedule, &state_.schedule);
  ASSERT_EQ(table.allocation, &state_.allocation);
  ASSERT_EQ(table.record_count, 3u);
  ASSERT_NE(table.records, nullptr);

  EXPECT_EQ(table.records[0].packet_index, 0u);
  EXPECT_EQ(table.records[0].node_index, 0u);
  EXPECT_EQ(table.records[0].block_index, 0u);
  EXPECT_EQ(table.records[0].scheduled_ordinal, 0u);
  EXPECT_EQ(table.records[0].progress_class_id, kSyntheticProgressPipe);
  EXPECT_TRUE(iree_string_view_equal(table.records[0].progress_class_name,
                                     IREE_SV("synthetic.pipe")));
  EXPECT_EQ(table.records[0].action, LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE);
  EXPECT_EQ(table.records[0].units, 2u);

  EXPECT_EQ(table.records[1].packet_index, 1u);
  EXPECT_EQ(table.records[1].node_index, 1u);
  EXPECT_EQ(table.records[1].progress_class_id, kSyntheticProgressScoreboard);
  EXPECT_TRUE(iree_string_view_equal(table.records[1].progress_class_name,
                                     IREE_SV("synthetic.scoreboard")));
  EXPECT_EQ(table.records[1].action, LOOM_LOW_PACKET_PROGRESS_ACTION_RESET);
  EXPECT_EQ(table.records[1].units, 0u);

  EXPECT_EQ(table.records[2].packet_index, 1u);
  EXPECT_EQ(table.records[2].node_index, 1u);
  EXPECT_EQ(table.records[2].progress_class_id, kSyntheticProgressPipe);
  EXPECT_TRUE(iree_string_view_equal(table.records[2].progress_class_name,
                                     IREE_SV("synthetic.pipe")));
  EXPECT_EQ(table.records[2].action, LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE);
  EXPECT_EQ(table.records[2].units, 1u);
}

TEST_F(LowPacketProgressTest, BuildsEmptyProgressTable) {
  const loom_low_packet_progress_provider_t provider = {
      /*.user_data=*/{},
      /*.query=*/EmptyProgressQuery,
  };
  loom_low_packet_progress_table_t table = {};
  IREE_ASSERT_OK(loom_low_packet_progress_build(
      &state_.schedule, &state_.allocation, &provider, &arena_, &table));
  EXPECT_EQ(table.record_count, 0u);
  EXPECT_EQ(table.records, nullptr);
}

iree_status_t InvalidProgressQuery(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_view_t* packet,
    loom_low_packet_progress_emit_fn_t emit, void* emit_user_data) {
  (void)user_data;
  (void)schedule;
  (void)allocation;
  (void)packet;
  return EmitEvent(emit, emit_user_data, kSyntheticProgressPipe,
                   IREE_SV("synthetic.pipe"),
                   LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE, 0);
}

TEST_F(LowPacketProgressTest, RejectsInvalidProgressEvents) {
  const loom_low_packet_progress_provider_t provider = {
      /*.user_data=*/{},
      /*.query=*/InvalidProgressQuery,
  };
  loom_low_packet_progress_table_t table = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_low_packet_progress_build(&state_.schedule, &state_.allocation,
                                     &provider, &arena_, &table));
}

}  // namespace
}  // namespace loom
