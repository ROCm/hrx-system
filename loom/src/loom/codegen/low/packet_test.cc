// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packet.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

struct PacketTestState {
  loom_low_descriptor_t descriptors[2] = {};
  loom_low_asm_form_t asm_forms[2] = {};
  loom_low_descriptor_set_t descriptor_set = {};
  loom_module_t module = {};
  loom_op_t function_op = {};
  loom_region_t region = {};
  loom_block_t block = {};
  loom_block_t* region_blocks[1] = {};
  loom_low_schedule_block_t blocks[1] = {};
  loom_low_schedule_node_t nodes[2] = {};
  uint32_t scheduled_node_indices[2] = {};
  uint32_t selected_asm_form_ordinals[2] = {};
  loom_low_schedule_table_t schedule = {};
  loom_low_packet_asm_form_table_t asm_form_table = {};
  loom_low_allocation_table_t allocation = {};
};

void InitializePacketTestState(PacketTestState* state) {
  state->descriptors[0].canonical_asm_form_ordinal =
      LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  state->descriptors[1].canonical_asm_form_ordinal = 0;

  state->asm_forms[0].descriptor_ordinal = 1;
  state->asm_forms[1].descriptor_ordinal = 0;

  state->descriptor_set.descriptors = state->descriptors;
  state->descriptor_set.descriptor_count = IREE_ARRAYSIZE(state->descriptors);
  state->descriptor_set.asm_forms = state->asm_forms;
  state->descriptor_set.asm_form_count = IREE_ARRAYSIZE(state->asm_forms);

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
  state->nodes[0].scheduled_ordinal = 1;
  state->nodes[0].kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR;
  state->nodes[0].descriptor = &state->descriptors[1];
  state->nodes[1].block = &state->block;
  state->nodes[1].block_index = 0;
  state->nodes[1].source_ordinal = 1;
  state->nodes[1].scheduled_ordinal = 0;
  state->nodes[1].kind = LOOM_LOW_SCHEDULE_NODE_STRUCTURAL;

  state->scheduled_node_indices[0] = 1;
  state->scheduled_node_indices[1] = 0;

  state->selected_asm_form_ordinals[0] = LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  state->selected_asm_form_ordinals[1] = 0;

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

  state->asm_form_table.module = &state->module;
  state->asm_form_table.function_op = &state->function_op;
  state->asm_form_table.target.descriptor_set = &state->descriptor_set;
  state->asm_form_table.asm_form_ordinals = state->selected_asm_form_ordinals;
  state->asm_form_table.asm_form_ordinal_count =
      IREE_ARRAYSIZE(state->selected_asm_form_ordinals);

  state->allocation.module = &state->module;
  state->allocation.function_op = &state->function_op;
  state->allocation.target.descriptor_set = &state->descriptor_set;
}

TEST(LowPacketTest, ViewsScheduledPackets) {
  PacketTestState state;
  InitializePacketTestState(&state);

  EXPECT_EQ(loom_low_packet_count(&state.schedule), 2u);

  uint32_t node_index = LOOM_LOW_SCHEDULE_NODE_NONE;
  IREE_ASSERT_OK(
      loom_low_packet_node_index_at(&state.schedule, 0, &node_index));
  EXPECT_EQ(node_index, 1u);

  loom_low_packet_view_t packet;
  IREE_ASSERT_OK(
      loom_low_packet_view_at(&state.schedule, &state.allocation, 0, &packet));
  EXPECT_EQ(packet.packet_index, 0u);
  EXPECT_EQ(packet.node_index, 1u);
  EXPECT_EQ(packet.node, &state.nodes[1]);
  EXPECT_EQ(packet.descriptor, nullptr);

  IREE_ASSERT_OK(
      loom_low_packet_view_at(&state.schedule, &state.allocation, 1, &packet));
  EXPECT_EQ(packet.packet_index, 1u);
  EXPECT_EQ(packet.node_index, 0u);
  EXPECT_EQ(packet.node, &state.nodes[0]);
  EXPECT_EQ(packet.descriptor, &state.descriptors[1]);
}

TEST(LowPacketTest, ViewsBlockScheduledOrdinals) {
  PacketTestState state;
  InitializePacketTestState(&state);

  iree_host_size_t packet_index = LOOM_LOW_PACKET_INDEX_NONE;
  IREE_ASSERT_OK(loom_low_packet_index_at_block_ordinal(&state.schedule, 0, 0,
                                                        &packet_index));
  EXPECT_EQ(packet_index, 0u);

  uint32_t node_index = LOOM_LOW_SCHEDULE_NODE_NONE;
  IREE_ASSERT_OK(loom_low_packet_node_index_at_block_ordinal(&state.schedule, 0,
                                                             0, &node_index));
  EXPECT_EQ(node_index, 1u);

  loom_low_packet_view_t packet;
  IREE_ASSERT_OK(loom_low_packet_view_at_block_ordinal(
      &state.schedule, &state.allocation, 0, 1, &packet));
  EXPECT_EQ(packet.packet_index, 1u);
  EXPECT_EQ(packet.node_index, 0u);
  EXPECT_EQ(packet.node, &state.nodes[0]);
  EXPECT_EQ(packet.descriptor, &state.descriptors[1]);
}

TEST(LowPacketTest, RejectsInvalidBlockScheduledOrdinal) {
  PacketTestState state;
  InitializePacketTestState(&state);

  iree_host_size_t packet_index = LOOM_LOW_PACKET_INDEX_NONE;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_low_packet_index_at_block_ordinal(
                            &state.schedule, 1, 0, &packet_index));
  EXPECT_EQ(packet_index, LOOM_LOW_PACKET_INDEX_NONE);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_low_packet_index_at_block_ordinal(
                            &state.schedule, 0, 2, &packet_index));
  EXPECT_EQ(packet_index, LOOM_LOW_PACKET_INDEX_NONE);
}

TEST(LowPacketTest, RejectsBlockOrdinalOutsidePacketStream) {
  PacketTestState state;
  InitializePacketTestState(&state);
  state.blocks[0].scheduled_node_start = 1;
  state.blocks[0].scheduled_node_count = 2;

  iree_host_size_t packet_index = LOOM_LOW_PACKET_INDEX_NONE;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_low_packet_index_at_block_ordinal(
                            &state.schedule, 0, 1, &packet_index));
  EXPECT_EQ(packet_index, LOOM_LOW_PACKET_INDEX_NONE);
}

TEST(LowPacketTest, RejectsBlockOrdinalReferencingInvalidNode) {
  PacketTestState state;
  InitializePacketTestState(&state);
  state.scheduled_node_indices[0] = 2;

  uint32_t node_index = LOOM_LOW_SCHEDULE_NODE_NONE;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_low_packet_node_index_at_block_ordinal(
                            &state.schedule, 0, 0, &node_index));
  EXPECT_EQ(node_index, LOOM_LOW_SCHEDULE_NODE_NONE);
}

TEST(LowPacketTest, ValidatesSelectedAsmForms) {
  PacketTestState state;
  InitializePacketTestState(&state);

  IREE_EXPECT_OK(loom_low_packet_validate_asm_form_table(
      &state.schedule, &state.asm_form_table));

  loom_low_packet_view_t packet;
  IREE_ASSERT_OK(
      loom_low_packet_view_at(&state.schedule, &state.allocation, 1, &packet));

  uint32_t asm_form_ordinal = LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_packet_lookup_asm_form(
      &state.schedule, &state.asm_form_table, &packet, &asm_form_ordinal));
  EXPECT_EQ(asm_form_ordinal, 0u);
}

TEST(LowPacketTest, FallsBackToCanonicalAsmForm) {
  PacketTestState state;
  InitializePacketTestState(&state);
  state.selected_asm_form_ordinals[1] = LOOM_LOW_ASM_FORM_ORDINAL_NONE;

  loom_low_packet_view_t packet;
  IREE_ASSERT_OK(
      loom_low_packet_view_at(&state.schedule, &state.allocation, 1, &packet));

  uint32_t asm_form_ordinal = LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_packet_lookup_asm_form(
      &state.schedule, &state.asm_form_table, &packet, &asm_form_ordinal));
  EXPECT_EQ(asm_form_ordinal, 0u);
  asm_form_ordinal = LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_packet_lookup_asm_form(
      &state.schedule, /*asm_forms=*/nullptr, &packet, &asm_form_ordinal));
  EXPECT_EQ(asm_form_ordinal, 0u);
}

TEST(LowPacketTest, RejectsSelectedAsmFormCountMismatch) {
  PacketTestState state;
  InitializePacketTestState(&state);
  state.asm_form_table.asm_form_ordinal_count = 1;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_low_packet_validate_asm_form_table(
                            &state.schedule, &state.asm_form_table));
}

TEST(LowPacketTest, RejectsSelectedAsmFormForStructuralPacket) {
  PacketTestState state;
  InitializePacketTestState(&state);
  state.selected_asm_form_ordinals[0] = 0;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_low_packet_validate_asm_form_table(
                            &state.schedule, &state.asm_form_table));
}

TEST(LowPacketTest, RejectsSelectedAsmFormDescriptorMismatch) {
  PacketTestState state;
  InitializePacketTestState(&state);
  state.selected_asm_form_ordinals[1] = 1;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_low_packet_validate_asm_form_table(
                            &state.schedule, &state.asm_form_table));
}

TEST(LowPacketTest, RejectsMismatchedTables) {
  PacketTestState state;
  InitializePacketTestState(&state);
  loom_op_t other_function_op = {};
  state.allocation.function_op = &other_function_op;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_low_packet_validate_tables(&state.schedule, &state.allocation));
}

TEST(LowPacketTest, RejectsUnnamedTableFunction) {
  PacketTestState state;
  InitializePacketTestState(&state);
  state.schedule.function_op = nullptr;
  state.allocation.function_op = nullptr;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_low_packet_validate_tables(&state.schedule, &state.allocation));
}

TEST(LowPacketTest, RejectsOutOfRangePacketIndex) {
  PacketTestState state;
  InitializePacketTestState(&state);

  uint32_t node_index = LOOM_LOW_SCHEDULE_NODE_NONE;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_low_packet_node_index_at(&state.schedule, 2, &node_index));
  EXPECT_EQ(node_index, LOOM_LOW_SCHEDULE_NODE_NONE);
}

TEST(LowPacketTest, MapsBlocksAndHazardGapsToPacketIndices) {
  PacketTestState state;
  InitializePacketTestState(&state);
  state.blocks[0].scheduled_node_start = 10;

  EXPECT_EQ(loom_low_packet_block_index(&state.schedule, &state.block), 0u);
  loom_block_t other_block = {};
  EXPECT_EQ(loom_low_packet_block_index(&state.schedule, &other_block),
            LOOM_LOW_PACKET_INDEX_NONE);

  const loom_low_schedule_hazard_gap_t hazard_gap = {
      /*.producer_node=*/{},
      /*.consumer_node=*/{},
      /*.block_index=*/0,
  };
  EXPECT_EQ(
      loom_low_packet_hazard_gap_packet_index(&state.schedule, &hazard_gap, 2),
      12u);
}

}  // namespace
}  // namespace loom
