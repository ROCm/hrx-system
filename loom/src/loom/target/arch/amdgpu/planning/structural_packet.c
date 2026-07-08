// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/structural_packet.h"

#include "iree/base/bitfield.h"
#include "loom/codegen/low/allocation/move_topology.h"
#include "loom/codegen/low/move_sequence.h"
#include "loom/ops/low/ops.h"

static const loom_low_allocation_assignment_t*
loom_amdgpu_structural_packet_assignment(
    const loom_low_allocation_table_t* allocation, loom_value_id_t value_id) {
  if (allocation == NULL || value_id == LOOM_VALUE_ID_INVALID) {
    return NULL;
  }
  return loom_low_allocation_try_map_active_value_assignment(allocation,
                                                             value_id, NULL);
}

static iree_status_t loom_amdgpu_structural_packet_analyze_copy(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op,
    loom_amdgpu_structural_packet_info_t* out_info) {
  out_info->flags = LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MOVEMENT;
  out_info->instruction_count = 0;
  const loom_low_allocation_assignment_t* source_assignment =
      loom_amdgpu_structural_packet_assignment(allocation,
                                               loom_low_copy_source(op));
  const loom_low_allocation_assignment_t* result_assignment =
      loom_amdgpu_structural_packet_assignment(allocation,
                                               loom_low_copy_result(op));
  if (source_assignment == NULL || result_assignment == NULL) {
    out_info->flags |= LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_FORWARDS_DEPENDENCIES;
    return iree_ok_status();
  }
  if (source_assignment->location_count != result_assignment->location_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU low.copy allocation is malformed");
  }
  if (loom_low_allocation_assignment_location_range_equal(source_assignment,
                                                          result_assignment)) {
    out_info->flags |= LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_FORWARDS_DEPENDENCIES;
    return iree_ok_status();
  }
  out_info->flags |= LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES |
                     LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_WRITES_VALU;
  out_info->instruction_count = result_assignment->location_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_structural_packet_analyze_units(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op,
    loom_low_allocation_packet_move_op_kind_t packet_move_kind,
    loom_amdgpu_structural_packet_info_t* out_info) {
  out_info->flags = LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MOVEMENT;
  iree_host_size_t move_count = 0;
  switch (packet_move_kind) {
    case LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_SLICE: {
      IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_slice_units(
          allocation, op, &move_count));
      break;
    }
    case LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_CONCAT: {
      IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_concat_units(
          allocation, op, &move_count));
      break;
    }
    case LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_NONE:
    case LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_COPY:
      IREE_ASSERT_UNREACHABLE(
          "AMDGPU structural packet unit analysis requires slice or concat");
      return iree_status_from_code(IREE_STATUS_INTERNAL);
  }
  out_info->instruction_count = move_count;
  if (move_count == 0) {
    out_info->flags |= LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_FORWARDS_DEPENDENCIES;
  } else {
    out_info->flags |= LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES |
                       LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_WRITES_VALU;
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_structural_packet_analyze(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op,
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
  switch (packet_move_kind) {
    case LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_COPY:
      return loom_amdgpu_structural_packet_analyze_copy(allocation, op,
                                                        out_info);
    case LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_SLICE:
    case LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_CONCAT:
      return loom_amdgpu_structural_packet_analyze_units(
          allocation, op, packet_move_kind, out_info);
    case LOOM_LOW_ALLOCATION_PACKET_MOVE_OP_NONE:
      break;
  }
  if (loom_low_live_in_isa(op) || loom_low_storage_reserve_isa(op)) {
    return iree_ok_status();
  }
  out_info->flags = LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES;
  out_info->instruction_count = 1;
  return iree_ok_status();
}
