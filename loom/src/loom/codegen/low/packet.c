// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packet.h"

#include "loom/codegen/low/immediates.h"
#include "loom/ops/low/ops.h"

bool loom_low_packet_try_op_attrs(const loom_op_t* op,
                                  loom_named_attr_slice_t* out_attrs,
                                  uint16_t* out_attrs_attr_index) {
  if (out_attrs != NULL) {
    *out_attrs = loom_named_attr_slice_empty();
  }
  if (out_attrs_attr_index != NULL) {
    *out_attrs_attr_index = UINT16_MAX;
  }
  if (loom_low_op_isa(op)) {
    if (out_attrs != NULL) {
      *out_attrs = loom_low_op_attrs(op);
    }
    if (out_attrs_attr_index != NULL) {
      *out_attrs_attr_index = loom_low_op_attrs_diagnostic_ref().index;
    }
    return true;
  }
  if (loom_low_const_isa(op)) {
    if (out_attrs != NULL) {
      *out_attrs = loom_low_const_attrs(op);
    }
    if (out_attrs_attr_index != NULL) {
      *out_attrs_attr_index = loom_low_const_attrs_diagnostic_ref().index;
    }
    return true;
  }
  if (loom_low_op_cond_br_isa(op)) {
    if (out_attrs != NULL) {
      *out_attrs = loom_low_op_cond_br_attrs(op);
    }
    if (out_attrs_attr_index != NULL) {
      *out_attrs_attr_index = loom_low_op_cond_br_attrs_diagnostic_ref().index;
    }
    return true;
  }
  return false;
}

loom_named_attr_slice_t loom_low_packet_attrs(
    const loom_low_packet_view_t* packet) {
  if (packet == NULL || packet->node == NULL) {
    return loom_named_attr_slice_empty();
  }
  loom_named_attr_slice_t attrs = loom_named_attr_slice_empty();
  (void)loom_low_packet_try_op_attrs(packet->node->op, &attrs, NULL);
  return attrs;
}

loom_attribute_t loom_low_packet_immediate_attr(
    const loom_low_packet_view_t* packet,
    const loom_low_immediate_t* immediate) {
  return loom_low_bound_immediate_attr(loom_low_packet_attrs(packet),
                                       packet->node->immediate_presence,
                                       immediate);
}

int64_t loom_low_packet_immediate_i64(const loom_low_packet_view_t* packet,
                                      const loom_low_immediate_t* immediate) {
  const loom_attribute_t attr =
      loom_low_packet_immediate_attr(packet, immediate);
  return attr.kind == LOOM_ATTR_ABSENT ? immediate->default_value : attr.i64;
}

uint32_t loom_low_packet_block_index(const loom_low_schedule_table_t* schedule,
                                     const loom_block_t* block) {
  if (!schedule || !block) {
    return LOOM_LOW_PACKET_INDEX_NONE;
  }
  uint16_t block_index = 0;
  if (!loom_region_try_block_index(block->parent_region, block, &block_index) ||
      block_index >= schedule->block_count) {
    return LOOM_LOW_PACKET_INDEX_NONE;
  }
  return schedule->blocks[block_index].block == block
             ? block_index
             : LOOM_LOW_PACKET_INDEX_NONE;
}

uint32_t loom_low_packet_hazard_gap_packet_index(
    const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_hazard_gap_t* hazard_gap,
    uint32_t scheduled_ordinal) {
  if (!schedule || !hazard_gap ||
      hazard_gap->block_index >= schedule->block_count) {
    return LOOM_LOW_PACKET_INDEX_NONE;
  }
  const loom_low_schedule_block_t* block =
      &schedule->blocks[hazard_gap->block_index];
  const uint64_t packet_index =
      (uint64_t)block->scheduled_node_start + scheduled_ordinal;
  return packet_index < LOOM_LOW_PACKET_INDEX_NONE ? (uint32_t)packet_index
                                                   : LOOM_LOW_PACKET_INDEX_NONE;
}
