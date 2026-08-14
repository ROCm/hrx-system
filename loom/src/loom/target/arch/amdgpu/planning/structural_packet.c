// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/structural_packet.h"

#include "iree/base/bitfield.h"
#include "loom/codegen/low/allocation/move_topology.h"
#include "loom/codegen/low/packet.h"
#include "loom/ops/low/ops.h"

static bool loom_amdgpu_structural_packet_branch_falls_through(
    const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_node_t* node, const loom_block_t* destination) {
  const uint32_t destination_block_index =
      loom_low_packet_block_index(schedule, destination);
  return destination_block_index != LOOM_LOW_PACKET_INDEX_NONE &&
         destination_block_index == node->block_index + 1;
}

static uint64_t loom_amdgpu_structural_packet_control_transfer_count(
    const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_node_t* node) {
  const loom_op_t* op = node->op;
  if (loom_low_return_isa(op)) {
    return 1;
  }
  if (loom_low_br_isa(op)) {
    return loom_amdgpu_structural_packet_branch_falls_through(
               schedule, node, loom_low_br_dest(op))
               ? 0
               : 1;
  }
  if (!loom_low_cond_br_isa(op)) {
    return 0;
  }

  const loom_block_t* true_destination = loom_low_cond_br_true_dest(op);
  const loom_block_t* false_destination = loom_low_cond_br_false_dest(op);
  const bool true_fallthrough =
      loom_amdgpu_structural_packet_branch_falls_through(schedule, node,
                                                         true_destination);
  if (true_destination == false_destination) {
    return true_fallthrough ? 0 : 1;
  }
  const bool false_fallthrough =
      loom_amdgpu_structural_packet_branch_falls_through(schedule, node,
                                                         false_destination);
  return true_fallthrough || false_fallthrough ? 1 : 2;
}

static void loom_amdgpu_structural_packet_analyze_move_range(
    loom_low_move_range_t move_range,
    loom_amdgpu_structural_packet_info_t* out_info) {
  out_info->flags |= LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MOVEMENT;
  out_info->instruction_count += move_range.count;
  out_info->moves = move_range;
  if (move_range.count == 0) {
    return;
  }
  out_info->flags |= LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES;
}

static void loom_amdgpu_structural_packet_analyze_packet_moves(
    const loom_low_allocation_table_t* allocation, uint32_t source_ordinal,
    loom_amdgpu_structural_packet_analysis_flags_t analysis_flags,
    loom_amdgpu_structural_packet_info_t* out_info) {
  out_info->flags = LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MOVEMENT;
  if (allocation == NULL) {
    if (iree_all_bits_set(
            analysis_flags,
            LOOM_AMDGPU_STRUCTURAL_PACKET_ANALYSIS_FLAG_REQUIRE_ALLOCATION)) {
      out_info->flags |= LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES;
      out_info->instruction_count = 1;
    } else {
      out_info->flags |=
          LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_FORWARDS_DEPENDENCIES;
    }
    return;
  }
  const loom_low_allocation_packet_move_group_t* group =
      loom_low_allocation_find_packet_move_group_by_source_ordinal(
          allocation, source_ordinal);
  if (group == NULL || group->move_group.moves.count == 0) {
    out_info->flags |= LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_FORWARDS_DEPENDENCIES;
    return;
  }
  loom_amdgpu_structural_packet_analyze_move_range(group->move_group.moves,
                                                   out_info);
}

loom_amdgpu_structural_packet_info_t loom_amdgpu_structural_packet_analyze(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_schedule_node_t* node,
    loom_amdgpu_structural_packet_analysis_flags_t analysis_flags) {
  loom_amdgpu_structural_packet_info_t info = {0};
  const loom_op_t* op = node->op;
  if (op == NULL) {
    info.flags = LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES;
    info.instruction_count = 1;
    return info;
  }
  if (loom_traits_are_compile_time_only(node->traits)) {
    return info;
  }
  const loom_low_allocation_packet_move_op_kind_t packet_move_kind =
      loom_low_allocation_move_topology_packet_move_op_kind(op);
  if (packet_move_kind != LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_NONE) {
    loom_amdgpu_structural_packet_analyze_packet_moves(
        allocation, node->source_ordinal, analysis_flags, &info);
    return info;
  }
  if (loom_low_live_in_isa(op) || loom_low_storage_reserve_isa(op)) {
    return info;
  }
  if (loom_low_storage_view_isa(op)) {
    info.flags = LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_FORWARDS_DEPENDENCIES;
    return info;
  }
  if (loom_low_storage_address_isa(op)) {
    info.flags = LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES;
    info.instruction_count = 1;
    info.vector_alu_instruction_count = 1;
    return info;
  }
  if (loom_low_br_isa(op)) {
    if (allocation != NULL && loom_low_br_args(op).count != 0) {
      const loom_low_allocation_edge_copy_group_t* group =
          loom_low_allocation_find_edge_copy_group_by_source_ordinal(
              allocation, node->source_ordinal);
      IREE_ASSERT(group != NULL);
      loom_amdgpu_structural_packet_analyze_move_range(group->move_group.moves,
                                                       &info);
    }
    const uint64_t control_transfer_count =
        loom_amdgpu_structural_packet_control_transfer_count(schedule, node);
    info.instruction_count += control_transfer_count;
    info.scalar_alu_instruction_count += control_transfer_count;
    if (control_transfer_count != 0) {
      info.flags |= LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES;
    }
    if (info.instruction_count == 0) {
      info.flags |= LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_FORWARDS_DEPENDENCIES;
    }
    return info;
  }
  if (loom_low_cond_br_isa(op)) {
    const uint64_t control_transfer_count =
        loom_amdgpu_structural_packet_control_transfer_count(schedule, node);
    if (control_transfer_count != 0) {
      info.flags = LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES |
                   LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_READS_SCC;
      info.instruction_count = control_transfer_count;
      info.scalar_alu_instruction_count = control_transfer_count;
    }
    return info;
  }
  if (loom_low_return_isa(op)) {
    info.flags = LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES;
    info.instruction_count = 1;
    info.scalar_alu_instruction_count = 1;
    return info;
  }
  info.flags = LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES;
  info.instruction_count = 1;
  return info;
}
