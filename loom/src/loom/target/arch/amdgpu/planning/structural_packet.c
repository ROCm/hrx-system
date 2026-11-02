// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/structural_packet.h"

#include "iree/base/bitfield.h"
#include "loom/codegen/low/allocation/move_topology.h"
#include "loom/target/arch/amdgpu/abi/call_abi.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static void loom_amdgpu_structural_packet_append_move_group(
    const loom_low_allocation_table_t* allocation,
    const loom_low_move_group_t* move_group,
    loom_amdgpu_structural_packet_info_t* out_info) {
  const loom_low_move_range_t move_range = move_group->moves;
  out_info->flags |= LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MOVEMENT;
  if (move_range.count == 0) {
    return;
  }
  IREE_ASSERT_LT(out_info->move_segment_count,
                 LOOM_AMDGPU_STRUCTURAL_PACKET_MOVE_SEGMENT_CAPACITY);
  out_info->move_segments[out_info->move_segment_count++] =
      (loom_amdgpu_structural_move_segment_t){
          .group = move_group,
          .instruction_offset = out_info->instruction_count,
      };
  out_info->instruction_count += move_range.count;
  out_info->flags |= LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES;
  for (iree_host_size_t i = 0; i < move_range.count; ++i) {
    const loom_low_move_t* move = &allocation->moves[move_range.start + i];
    if (move->destination.descriptor_reg_class_id ==
        LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
      out_info->flags |= LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_USES_VECTOR_ALU;
    }
  }
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
  if (group == NULL) {
    out_info->flags |= LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_FORWARDS_DEPENDENCIES;
    return;
  }
  if (iree_any_bit_set(
          group->transfer_flags,
          LOOM_LOW_ALLOCATION_PACKET_TRANSFER_GROUP_FLAG_FORWARDED)) {
    out_info->flags |= LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_FORWARDS_DEPENDENCIES;
  }
  if (group->move_group.moves.count != 0) {
    loom_amdgpu_structural_packet_append_move_group(
        allocation, &group->move_group, out_info);
  }
}

loom_amdgpu_structural_packet_info_t loom_amdgpu_structural_packet_analyze(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation, uint32_t node_index,
    loom_amdgpu_structural_packet_analysis_flags_t analysis_flags) {
  loom_amdgpu_structural_packet_info_t info = {0};
  const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
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
    loom_amdgpu_structural_packet_analyze_packet_moves(allocation, node_index,
                                                       analysis_flags, &info);
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
  if (loom_low_func_call_isa(op)) {
    info.flags = LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES |
                 LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_CALL;
    const loom_low_allocation_call_moves_t* moves = NULL;
    if (allocation != NULL) {
      moves = loom_low_allocation_find_call_moves_by_source_ordinal(allocation,
                                                                    node_index);
      IREE_ASSERT(moves != NULL);
      loom_amdgpu_structural_packet_append_move_group(allocation,
                                                      &moves->arguments, &info);
    }
    info.instruction_count +=
        LOOM_AMDGPU_CALL_ABI_CALL_CONTROL_INSTRUCTION_COUNT;
    info.scalar_alu_instruction_count +=
        LOOM_AMDGPU_CALL_ABI_CALL_CONTROL_INSTRUCTION_COUNT;
    if (moves != NULL) {
      loom_amdgpu_structural_packet_append_move_group(allocation,
                                                      &moves->results, &info);
    }
    return info;
  }
  if (loom_direct_branch_isa(op)) {
    if (allocation != NULL && loom_direct_branch_args(op).count != 0) {
      const loom_low_allocation_edge_copy_group_t* group =
          loom_low_allocation_find_edge_copy_group_by_source_ordinal(
              allocation, node_index);
      IREE_ASSERT(group != NULL);
      loom_amdgpu_structural_packet_append_move_group(
          allocation, &group->move_group, &info);
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
      if (control_transfer_count == 2) {
        info.flags |= LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_CONDITIONAL_TAIL;
      }
    }
    return info;
  }
  if (loom_low_return_isa(op) || loom_low_unreachable_isa(op)) {
    info.flags = LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES;
    info.instruction_count = 1;
    info.scalar_alu_instruction_count = 1;
    return info;
  }
  info.flags = LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES;
  info.instruction_count = 1;
  return info;
}
