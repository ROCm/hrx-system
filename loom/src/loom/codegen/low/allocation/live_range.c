// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/live_range.h"

#include "loom/codegen/low/allocation/storage.h"

bool loom_low_allocation_live_range_assignment_overlaps_interval(
    const loom_low_allocation_assignment_t* assignment,
    const loom_liveness_interval_t* interval) {
  IREE_ASSERT_ARGUMENT(assignment);
  IREE_ASSERT_ARGUMENT(interval);
  return assignment->start_point < interval->end_point &&
         interval->start_point < assignment->end_point;
}

bool loom_low_allocation_live_range_interval_is_allocatable(
    const loom_liveness_interval_t* interval) {
  IREE_ASSERT_ARGUMENT(interval);
  return interval->value_class.type_kind == LOOM_TYPE_REGISTER &&
         interval->unit_count > 0;
}

uint32_t loom_low_allocation_live_range_interval_storage_end_point(
    const loom_liveness_interval_t* interval) {
  IREE_ASSERT_ARGUMENT(interval);
  if (interval->end_point > interval->start_point) {
    return interval->end_point;
  }
  return interval->start_point == UINT32_MAX ? UINT32_MAX
                                             : interval->start_point + 1u;
}

uint32_t loom_low_allocation_live_range_interval_initial_unit_end_point(
    const loom_liveness_interval_t* interval) {
  IREE_ASSERT_ARGUMENT(interval);
  if (interval->end_point == interval->start_point) {
    return loom_low_allocation_live_range_interval_storage_end_point(interval);
  }
  return interval->start_point;
}

static bool loom_low_allocation_live_range_is_power_of_two_u32(uint32_t value) {
  return value != 0 && (value & (value - 1u)) == 0;
}

uint32_t loom_low_allocation_live_range_interval_alignment(
    const loom_liveness_interval_t* interval) {
  IREE_ASSERT_ARGUMENT(interval);
  if (interval->unit_count <= 1 ||
      !loom_low_allocation_live_range_is_power_of_two_u32(
          interval->unit_count)) {
    return 1;
  }
  return interval->unit_count;
}

uint32_t loom_low_allocation_live_range_assignment_unit_end_point(
    const uint32_t* unit_end_points, iree_host_size_t unit_end_point_count,
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_offset) {
  IREE_ASSERT_ARGUMENT(assignment);
  if (unit_offset >= assignment->unit_count) {
    return assignment->end_point;
  }
  IREE_ASSERT_ARGUMENT(unit_end_points);
  const uint64_t unit_index =
      (uint64_t)assignment->unit_end_point_start + unit_offset;
  IREE_ASSERT(unit_index < unit_end_point_count,
              "allocation unit end point must be in range");
  return unit_end_points[(iree_host_size_t)unit_index];
}

uint32_t loom_low_allocation_live_range_assignment_max_unit_end_point(
    const uint32_t* unit_end_points, iree_host_size_t unit_end_point_count,
    const loom_low_allocation_assignment_t* assignment) {
  IREE_ASSERT_ARGUMENT(assignment);
  uint32_t end_point = assignment->end_point;
  for (uint32_t unit_offset = 0; unit_offset < assignment->unit_count;
       ++unit_offset) {
    const uint32_t unit_end_point =
        loom_low_allocation_live_range_assignment_unit_end_point(
            unit_end_points, unit_end_point_count, assignment, unit_offset);
    if (end_point < unit_end_point) {
      end_point = unit_end_point;
    }
  }
  return end_point;
}

static bool loom_low_allocation_value_list_contains(
    const loom_value_id_t* values, iree_host_size_t count,
    loom_value_id_t value_id) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (values[i] == value_id) {
      return true;
    }
  }
  return false;
}

static bool loom_low_allocation_value_range_may_be_live_in_block(
    loom_value_id_t value_id, uint32_t start_point, uint32_t end_point,
    const loom_liveness_block_info_t* block_info) {
  if (loom_low_allocation_value_list_contains(
          block_info->live_in_values, block_info->live_in_count, value_id) ||
      loom_low_allocation_value_list_contains(
          block_info->live_out_values, block_info->live_out_count, value_id)) {
    return true;
  }
  if (block_info->start_point <= start_point &&
      start_point < block_info->end_point) {
    return true;
  }
  return block_info->start_point < end_point &&
         end_point <= block_info->end_point;
}

bool loom_low_allocation_live_range_values_overlap(
    const loom_liveness_analysis_t* liveness, loom_value_id_t lhs_value_id,
    uint32_t lhs_start_point, uint32_t lhs_end_point,
    loom_value_id_t rhs_value_id, uint32_t rhs_start_point,
    uint32_t rhs_end_point) {
  IREE_ASSERT_ARGUMENT(liveness);
  const uint32_t overlap_start =
      lhs_start_point > rhs_start_point ? lhs_start_point : rhs_start_point;
  const uint32_t overlap_end =
      lhs_end_point < rhs_end_point ? lhs_end_point : rhs_end_point;
  if (overlap_start >= overlap_end) {
    return false;
  }
  for (iree_host_size_t i = 0; i < liveness->block_count; ++i) {
    const loom_liveness_block_info_t* block_info = &liveness->blocks[i];
    const uint32_t block_overlap_start = overlap_start > block_info->start_point
                                             ? overlap_start
                                             : block_info->start_point;
    const uint32_t block_overlap_end = overlap_end < block_info->end_point
                                           ? overlap_end
                                           : block_info->end_point;
    if (block_overlap_start >= block_overlap_end) {
      continue;
    }
    if (loom_low_allocation_value_range_may_be_live_in_block(
            lhs_value_id, lhs_start_point, lhs_end_point, block_info) &&
        loom_low_allocation_value_range_may_be_live_in_block(
            rhs_value_id, rhs_start_point, rhs_end_point, block_info)) {
      return true;
    }
  }
  return false;
}

iree_status_t loom_low_allocation_live_range_op_program_point(
    const loom_liveness_analysis_t* liveness, const loom_op_t* op,
    uint32_t* out_program_point) {
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(op);
  IREE_ASSERT_ARGUMENT(out_program_point);
  if (loom_liveness_analysis_includes_region_tree(liveness)) {
    return loom_liveness_op_program_point(liveness, loom_liveness_order_empty(),
                                          op, out_program_point);
  }
  *out_program_point = UINT32_MAX;
  const loom_liveness_block_info_t* block_info =
      loom_liveness_block_info_for_block(liveness, op->parent_block);
  if (!block_info) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation cannot find liveness block for low operation");
  }
  uint32_t program_point = block_info->start_point;
  const loom_op_t* block_op = op->parent_block->first_op;
  while (block_op && block_op != op) {
    ++program_point;
    block_op = block_op->next_op;
  }
  if (block_op != op) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low operation is not linked in its parent block");
  }
  *out_program_point = program_point;
  return iree_ok_status();
}

iree_status_t loom_low_allocation_live_range_ordered_op_program_point(
    const loom_liveness_analysis_t* liveness, const loom_region_t* body,
    loom_liveness_order_t liveness_order, const loom_op_t* op,
    uint32_t* out_program_point) {
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(body);
  IREE_ASSERT_ARGUMENT(op);
  IREE_ASSERT_ARGUMENT(out_program_point);
  if (loom_liveness_analysis_includes_region_tree(liveness)) {
    return loom_liveness_op_program_point(liveness, liveness_order, op,
                                          out_program_point);
  }
  if (loom_liveness_order_is_empty(liveness_order)) {
    return loom_low_allocation_live_range_op_program_point(liveness, op,
                                                           out_program_point);
  }
  uint16_t block_index = 0;
  if (!loom_region_try_block_index(body, op->parent_block, &block_index) ||
      block_index >= liveness_order.block_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation cannot find ordered block for low operation");
  }
  const loom_liveness_block_info_t* block_info =
      loom_liveness_block_info_for_block(liveness, op->parent_block);
  if (!block_info) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation cannot find ordered liveness block for low operation");
  }
  const loom_liveness_block_order_t* block_order =
      &liveness_order.blocks[block_index];
  for (iree_host_size_t i = 0; i < block_order->op_count; ++i) {
    if (block_order->ops[i] != op) {
      continue;
    }
    if (i > UINT32_MAX - block_info->start_point) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "ordered low operation program point exceeds uint32_t");
    }
    *out_program_point = block_info->start_point + (uint32_t)i;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "ordered low operation has no operation order entry");
}

bool loom_low_allocation_live_range_assignments_conflict(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness, const uint32_t* unit_end_points,
    iree_host_size_t unit_end_point_count,
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(lhs);
  IREE_ASSERT_ARGUMENT(rhs);
  if (!loom_low_allocation_assignment_is_register_like(lhs) ||
      !loom_low_allocation_assignment_is_register_like(rhs)) {
    return false;
  }
  if (!loom_low_allocation_storage_assignment_classes_share(descriptor_set, lhs,
                                                            rhs)) {
    return false;
  }
  const uint64_t lhs_end = (uint64_t)lhs->location_base + lhs->location_count;
  const uint64_t rhs_end = (uint64_t)rhs->location_base + rhs->location_count;
  const uint64_t overlap_begin = lhs->location_base > rhs->location_base
                                     ? lhs->location_base
                                     : rhs->location_base;
  const uint64_t overlap_end = lhs_end < rhs_end ? lhs_end : rhs_end;
  if (overlap_begin >= overlap_end) {
    return false;
  }
  for (uint64_t location = overlap_begin; location < overlap_end; ++location) {
    const uint32_t lhs_unit_offset = (uint32_t)(location - lhs->location_base);
    const uint32_t rhs_unit_offset = (uint32_t)(location - rhs->location_base);
    const uint32_t lhs_unit_end_point =
        loom_low_allocation_live_range_assignment_unit_end_point(
            unit_end_points, unit_end_point_count, lhs, lhs_unit_offset);
    const uint32_t rhs_unit_end_point =
        loom_low_allocation_live_range_assignment_unit_end_point(
            unit_end_points, unit_end_point_count, rhs, rhs_unit_offset);
    if (lhs->start_point >= rhs_unit_end_point ||
        rhs->start_point >= lhs_unit_end_point) {
      continue;
    }
    if (loom_low_allocation_live_range_values_overlap(
            liveness, lhs->value_id, lhs->start_point, lhs_unit_end_point,
            rhs->value_id, rhs->start_point, rhs_unit_end_point)) {
      return true;
    }
  }
  return false;
}
