// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/structural_packet.h"

#include "iree/base/bitfield.h"
#include "loom/codegen/low/allocation/move_topology.h"
#include "loom/ops/low/ops.h"

static void loom_amdgpu_structural_packet_analyze_moves(
    const loom_low_allocation_table_t* allocation, uint32_t source_ordinal,
    loom_amdgpu_structural_packet_info_t* out_info) {
  out_info->flags = LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MOVEMENT;
  if (allocation == NULL) {
    out_info->flags |= LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_FORWARDS_DEPENDENCIES;
    return;
  }
  const loom_low_allocation_packet_move_group_t* group =
      loom_low_allocation_find_packet_move_group_by_source_ordinal(
          allocation, source_ordinal);
  if (group == NULL) {
    out_info->flags |= LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_FORWARDS_DEPENDENCIES;
  } else {
    out_info->flags |= LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES |
                       LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_WRITES_VALU;
    out_info->instruction_count = group->move_group.moves.count;
  }
}

iree_status_t loom_amdgpu_structural_packet_analyze(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op,
    uint32_t source_ordinal,
    loom_amdgpu_structural_packet_analysis_flags_t analysis_flags,
    loom_amdgpu_structural_packet_info_t* out_info) {
  IREE_ASSERT_ARGUMENT(out_info);
  *out_info = (loom_amdgpu_structural_packet_info_t){0};
  if (op == NULL) {
    out_info->flags = LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES;
    out_info->instruction_count = 1;
    return iree_ok_status();
  }
  if (allocation == NULL &&
      iree_all_bits_set(
          analysis_flags,
          LOOM_AMDGPU_STRUCTURAL_PACKET_ANALYSIS_FLAG_REQUIRE_ALLOCATION)) {
    if (loom_low_allocation_move_topology_op_has_packet_moves(op)) {
      out_info->flags = LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MOVEMENT |
                        LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES;
      out_info->instruction_count = 1;
      return iree_ok_status();
    }
  }
  const loom_low_allocation_packet_move_op_kind_t packet_move_kind =
      loom_low_allocation_move_topology_packet_move_op_kind(op);
  if (packet_move_kind != LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_NONE) {
    loom_amdgpu_structural_packet_analyze_moves(allocation, source_ordinal,
                                                out_info);
    return iree_ok_status();
  }
  if (loom_low_live_in_isa(op) || loom_low_storage_reserve_isa(op)) {
    return iree_ok_status();
  }
  if (loom_low_cond_br_isa(op)) {
    out_info->flags = LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES |
                      LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_WRITES_SALU;
    out_info->instruction_count = 1;
    return iree_ok_status();
  }
  out_info->flags = LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES;
  out_info->instruction_count = 1;
  return iree_ok_status();
}
