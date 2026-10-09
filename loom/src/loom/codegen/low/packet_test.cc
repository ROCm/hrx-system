// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packet.h"

#include "iree/testing/gtest.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ops/low/ops.h"

namespace loom {
namespace {

struct PacketAttrTestOp {
  loom_op_t op = {};
  loom_attribute_t attrs[3] = {};
};

TEST(LowPacketTest, ReadsSparseImmediateValuesAndPreservesOmission) {
  loom_named_attr_t entries[] = {
      {1, 0, loom_attr_i64(17)},
      {2, 0, loom_attr_symbol(loom_symbol_ref_t{0, 3})},
  };
  PacketAttrTestOp storage;
  storage.op.kind = LOOM_OP_LOW_OP;
  storage.op.attribute_count = IREE_ARRAYSIZE(storage.attrs);
  loom_low_op_initialize_attrs(
      &storage.op,
      loom_make_canonical_attr_dict(entries, IREE_ARRAYSIZE(entries)));
  loom_low_schedule_node_t node = {
      .op = &storage.op, .immediate_presence = (1u << 1) | (1u << 31)};
  loom_low_packet_view_t packet = {.node = &node};
  loom_low_immediate_t field =
      {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment sequencing
           // spans intervening work.
  field.attribute_mask = 1u << 1;
  EXPECT_EQ(loom_low_packet_immediate_attr(&packet, &field).i64, 17);
  field.attribute_mask = 1u << 31;
  const loom_attribute_t symbol =
      loom_low_packet_immediate_attr(&packet, &field);
  EXPECT_EQ(symbol.kind, LOOM_ATTR_SYMBOL);
  EXPECT_EQ(symbol.symbol.symbol_id, 3u);
  field.attribute_mask = 1u << 3;
  field.default_value = 42;
  EXPECT_EQ(loom_low_packet_immediate_attr(&packet, &field).kind,
            LOOM_ATTR_ABSENT);
}

TEST(LowPacketTest, GetsDescriptorPacketOpAttrs) {
  loom_named_attr_t named_attrs[1] = {};
  named_attrs[0].name_id = 7;
  named_attrs[0].value = loom_attr_i64(42);

  PacketAttrTestOp low_op_storage;
  low_op_storage.op.kind = LOOM_OP_LOW_OP;
  low_op_storage.op.attribute_count = IREE_ARRAYSIZE(low_op_storage.attrs);
  loom_low_op_initialize_attrs(
      &low_op_storage.op,
      loom_make_canonical_attr_dict(named_attrs, IREE_ARRAYSIZE(named_attrs)));

  loom_named_attr_slice_t attrs = loom_named_attr_slice_empty();
  uint16_t attrs_attr_index = UINT16_MAX;
  EXPECT_TRUE(loom_low_packet_try_op_attrs(&low_op_storage.op, &attrs,
                                           &attrs_attr_index));
  EXPECT_EQ(attrs.entries, named_attrs);
  EXPECT_EQ(attrs.count, 1u);
  EXPECT_EQ(attrs_attr_index, loom_low_op_attrs_diagnostic_ref().index);

  PacketAttrTestOp low_const_storage;
  low_const_storage.op.kind = LOOM_OP_LOW_CONST;
  low_const_storage.op.attribute_count =
      IREE_ARRAYSIZE(low_const_storage.attrs);
  loom_low_const_initialize_attrs(
      &low_const_storage.op,
      loom_make_canonical_attr_dict(named_attrs, IREE_ARRAYSIZE(named_attrs)));

  attrs = loom_named_attr_slice_empty();
  attrs_attr_index = UINT16_MAX;
  EXPECT_TRUE(loom_low_packet_try_op_attrs(&low_const_storage.op, &attrs,
                                           &attrs_attr_index));
  EXPECT_EQ(attrs.entries, named_attrs);
  EXPECT_EQ(attrs.count, 1u);
  EXPECT_EQ(attrs_attr_index, loom_low_const_attrs_diagnostic_ref().index);
}

TEST(LowPacketTest, GetsPacketViewAttrs) {
  loom_named_attr_t named_attrs[1] = {};
  named_attrs[0].name_id = 7;
  named_attrs[0].value = loom_attr_i64(42);

  PacketAttrTestOp low_op_storage;
  low_op_storage.op.kind = LOOM_OP_LOW_OP;
  low_op_storage.op.attribute_count = IREE_ARRAYSIZE(low_op_storage.attrs);
  loom_low_op_initialize_attrs(
      &low_op_storage.op,
      loom_make_canonical_attr_dict(named_attrs, IREE_ARRAYSIZE(named_attrs)));

  loom_low_schedule_node_t node =
      {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment sequencing
           // spans intervening work.
  node.op = &low_op_storage.op;
  loom_low_packet_view_t packet = {.node = &node};

  loom_named_attr_slice_t attrs = loom_low_packet_attrs(&packet);
  EXPECT_EQ(attrs.entries, named_attrs);
  EXPECT_EQ(attrs.count, 1u);

  node.op = nullptr;
  attrs = loom_low_packet_attrs(&packet);
  EXPECT_EQ(attrs.entries, nullptr);
  EXPECT_EQ(attrs.count, 0u);
}

TEST(LowPacketTest, MapsBlocksAndHazardGapsToPacketIndices) {
  loom_region_t region = {};
  loom_block_t block = {};
  loom_block_t* region_blocks[] = {&block};
  region.block_count = IREE_ARRAYSIZE(region_blocks);
  region.block_capacity = IREE_ARRAYSIZE(region_blocks);
  region.blocks = region_blocks;
  block.parent_region = &region;
  block.region_index = 0;

  loom_low_schedule_block_t blocks[1] = {};
  blocks[0].block = &block;
  blocks[0].scheduled_node_start = 10;
  loom_low_schedule_table_t schedule = {.blocks = blocks,
                                        .block_count = IREE_ARRAYSIZE(blocks)};

  EXPECT_EQ(loom_low_packet_block_index(&schedule, &block), 0u);
  loom_block_t other_block = {};
  EXPECT_EQ(loom_low_packet_block_index(&schedule, &other_block),
            LOOM_LOW_PACKET_INDEX_NONE);

  const loom_low_schedule_hazard_gap_t hazard_gap = {
      .producer_node = {},
      .consumer_node = {},
      .block_index = 0,
  };
  EXPECT_EQ(loom_low_packet_hazard_gap_packet_index(&schedule, &hazard_gap, 2),
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

  loom_low_descriptor_t descriptor =
      {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment order
           // differs from declaration order.
  descriptor.operand_start = 0;
  descriptor.result_count = 1;
  descriptor.operand_count = IREE_ARRAYSIZE(operands);
  descriptor.minimum_packet_operand_count = 4;
  descriptor.constraint_start = 0;
  descriptor.constraint_count = IREE_ARRAYSIZE(constraints);

  loom_low_descriptor_set_t descriptor_set = {
      .operands = operands,
      .operand_count = IREE_ARRAYSIZE(operands),
      .constraints = constraints,
      .constraint_count = IREE_ARRAYSIZE(constraints)};

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
