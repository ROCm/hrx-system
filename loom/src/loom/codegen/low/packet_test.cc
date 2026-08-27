// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packet.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ops/low/ops.h"

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

struct PacketAttrTestOp {
  loom_op_t op = {};
  loom_attribute_t attrs[3] = {};
};

void InitializePacketTestState(PacketTestState* state) {
  state->descriptors[0].canonical_asm_form_ordinal =
      LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  state->descriptors[1].canonical_asm_form_ordinal = 0;

  state->asm_forms[0].descriptor_ordinal = 1;
  state->asm_forms[0].result_value_type_start =
      LOOM_LOW_ASM_RESULT_VALUE_TYPE_START_NONE;
  state->asm_forms[1].descriptor_ordinal = 0;
  state->asm_forms[1].result_value_type_start =
      LOOM_LOW_ASM_RESULT_VALUE_TYPE_START_NONE;

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

TEST(LowPacketTest, GetsDescriptorPacketOpAttrs) {
  loom_named_attr_t named_attrs[1] = {};
  named_attrs[0].name_id = 7;
  named_attrs[0].value = loom_attr_i64(42);

  PacketAttrTestOp low_op_storage;
  low_op_storage.op.kind = LOOM_OP_LOW_OP;
  low_op_storage.op.attribute_count = IREE_ARRAYSIZE(low_op_storage.attrs);
  low_op_storage.attrs[loom_low_op_attrs_ATTR_INDEX] =
      loom_make_canonical_attr_dict(named_attrs, IREE_ARRAYSIZE(named_attrs));

  loom_named_attr_slice_t attrs = loom_named_attr_slice_empty();
  uint16_t attrs_attr_index = UINT16_MAX;
  EXPECT_TRUE(loom_low_packet_try_op_attrs(&low_op_storage.op, &attrs,
                                           &attrs_attr_index));
  EXPECT_EQ(attrs.entries, named_attrs);
  EXPECT_EQ(attrs.count, 1u);
  EXPECT_EQ(attrs_attr_index, loom_low_op_attrs_ATTR_INDEX);

  PacketAttrTestOp low_const_storage;
  low_const_storage.op.kind = LOOM_OP_LOW_CONST;
  low_const_storage.op.attribute_count =
      IREE_ARRAYSIZE(low_const_storage.attrs);
  low_const_storage.attrs[loom_low_const_attrs_ATTR_INDEX] =
      loom_make_canonical_attr_dict(named_attrs, IREE_ARRAYSIZE(named_attrs));

  attrs = loom_named_attr_slice_empty();
  attrs_attr_index = UINT16_MAX;
  EXPECT_TRUE(loom_low_packet_try_op_attrs(&low_const_storage.op, &attrs,
                                           &attrs_attr_index));
  EXPECT_EQ(attrs.entries, named_attrs);
  EXPECT_EQ(attrs.count, 1u);
  EXPECT_EQ(attrs_attr_index, loom_low_const_attrs_ATTR_INDEX);
}

TEST(LowPacketTest, GetsPacketViewAttrs) {
  loom_named_attr_t named_attrs[1] = {};
  named_attrs[0].name_id = 7;
  named_attrs[0].value = loom_attr_i64(42);

  PacketAttrTestOp low_op_storage;
  low_op_storage.op.kind = LOOM_OP_LOW_OP;
  low_op_storage.op.attribute_count = IREE_ARRAYSIZE(low_op_storage.attrs);
  low_op_storage.attrs[loom_low_op_attrs_ATTR_INDEX] =
      loom_make_canonical_attr_dict(named_attrs, IREE_ARRAYSIZE(named_attrs));

  loom_low_schedule_node_t node = {};
  node.op = &low_op_storage.op;
  loom_low_packet_view_t packet = {};
  packet.node = &node;

  loom_named_attr_slice_t attrs = loom_low_packet_attrs(&packet);
  EXPECT_EQ(attrs.entries, named_attrs);
  EXPECT_EQ(attrs.count, 1u);

  node.op = nullptr;
  attrs = loom_low_packet_attrs(&packet);
  EXPECT_EQ(attrs.entries, nullptr);
  EXPECT_EQ(attrs.count, 0u);
}

TEST(LowPacketTest, ValidatesSelectedAsmForms) {
  PacketTestState state;
  InitializePacketTestState(&state);

  IREE_EXPECT_OK(loom_low_packet_validate_asm_form_table(
      &state.schedule, &state.asm_form_table));

  const loom_low_packet_view_t packet = loom_low_packet_at(&state.schedule, 1);

  uint32_t asm_form_ordinal = LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_packet_lookup_asm_form(
      &state.schedule, &state.asm_form_table, &packet, &asm_form_ordinal));
  EXPECT_EQ(asm_form_ordinal, 0u);
}

TEST(LowPacketTest, FallsBackToCanonicalAsmForm) {
  PacketTestState state;
  InitializePacketTestState(&state);
  state.selected_asm_form_ordinals[1] = LOOM_LOW_ASM_FORM_ORDINAL_NONE;

  const loom_low_packet_view_t packet = loom_low_packet_at(&state.schedule, 1);

  uint32_t asm_form_ordinal = LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_packet_lookup_asm_form(
      &state.schedule, &state.asm_form_table, &packet, &asm_form_ordinal));
  EXPECT_EQ(asm_form_ordinal, 0u);
  asm_form_ordinal = LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_packet_lookup_asm_form(
      &state.schedule, /*asm_forms=*/nullptr, &packet, &asm_form_ordinal));
  EXPECT_EQ(asm_form_ordinal, 0u);
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

TEST(LowDescriptorTest, IndexesPacketOperandRoles) {
  loom_low_operand_t operands[6] = {};
  operands[0].role = LOOM_LOW_OPERAND_ROLE_RESULT;
  operands[0].source_value_index = 0;
  operands[1].role = LOOM_LOW_OPERAND_ROLE_OPERAND;
  operands[1].source_value_index = 0;
  operands[2].role = LOOM_LOW_OPERAND_ROLE_RESOURCE;
  operands[2].source_value_index = 1;
  operands[2].flags = LOOM_LOW_OPERAND_FLAG_IMPLICIT;
  operands[3].role = LOOM_LOW_OPERAND_ROLE_IMPLICIT;
  operands[3].source_value_index = LOOM_LOW_ID_NONE;
  operands[3].flags = LOOM_LOW_OPERAND_FLAG_IMPLICIT;
  operands[4].role = LOOM_LOW_OPERAND_ROLE_PREDICATE;
  operands[4].source_value_index = 2;
  operands[5].role = LOOM_LOW_OPERAND_ROLE_RESOURCE;
  operands[5].source_value_index = 3;

  loom_low_constraint_t constraints[2] = {};
  constraints[0].kind = LOOM_LOW_CONSTRAINT_KIND_TIED;
  constraints[0].lhs_operand_index = 0;
  constraints[0].rhs_operand_index = 2;
  constraints[1].kind = LOOM_LOW_CONSTRAINT_KIND_TIED;
  constraints[1].lhs_operand_index = 0;
  constraints[1].rhs_operand_index = 4;

  loom_low_descriptor_t descriptor = {};
  descriptor.operand_start = 0;
  descriptor.result_count = 1;
  descriptor.operand_count = IREE_ARRAYSIZE(operands);
  descriptor.minimum_packet_operand_count = 4;
  descriptor.constraint_start = 0;
  descriptor.constraint_count = IREE_ARRAYSIZE(constraints);

  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.operands = operands;
  descriptor_set.operand_count = IREE_ARRAYSIZE(operands);
  descriptor_set.constraints = constraints;
  descriptor_set.constraint_count = IREE_ARRAYSIZE(constraints);

  EXPECT_FALSE(loom_low_descriptor_operand_maps_to_packet_operand(
      &descriptor_set, &descriptor, 0));
  EXPECT_TRUE(loom_low_descriptor_operand_maps_to_packet_operand(
      &descriptor_set, &descriptor, 1));
  EXPECT_TRUE(loom_low_descriptor_operand_maps_to_packet_operand(
      &descriptor_set, &descriptor, 2));
  EXPECT_FALSE(loom_low_descriptor_operand_maps_to_packet_operand(
      &descriptor_set, &descriptor, 3));
  EXPECT_TRUE(loom_low_descriptor_operand_maps_to_packet_operand(
      &descriptor_set, &descriptor, 4));
  EXPECT_TRUE(loom_low_descriptor_operand_maps_to_packet_operand(
      &descriptor_set, &descriptor, 5));

  EXPECT_EQ(
      loom_low_descriptor_operand_packet_index(&descriptor_set, &descriptor, 1),
      0u);
  EXPECT_EQ(
      loom_low_descriptor_operand_packet_index(&descriptor_set, &descriptor, 2),
      1u);
  EXPECT_EQ(
      loom_low_descriptor_operand_packet_index(&descriptor_set, &descriptor, 4),
      2u);
  EXPECT_EQ(
      loom_low_descriptor_operand_packet_index(&descriptor_set, &descriptor, 5),
      3u);

  EXPECT_TRUE(loom_low_descriptor_operands_are_tied(&descriptor_set,
                                                    &descriptor, 0, 2));
  EXPECT_TRUE(loom_low_descriptor_operands_are_tied(&descriptor_set,
                                                    &descriptor, 0, 4));
  EXPECT_TRUE(loom_low_descriptor_operands_are_tied(&descriptor_set,
                                                    &descriptor, 4, 0));
}

}  // namespace
}  // namespace loom
