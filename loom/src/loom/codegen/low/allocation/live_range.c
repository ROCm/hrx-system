// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/live_range.h"

#include <stdint.h>

#include "loom/codegen/low/allocation/storage.h"
#include "loom/target/registers.h"

struct loom_low_allocation_op_point_entry_t {
  // Index of the operation row represented by this entry.
  uint32_t operation_index;
  // Next entry in the hashed operation bucket.
  uint32_t next_entry;
};

static uint32_t loom_low_allocation_round_up_to_power_of_two_u32(
    uint32_t value) {
  if (value <= 1) {
    return 1;
  }
  --value;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  return value == UINT32_MAX ? 0 : value + 1u;
}

static uint32_t loom_low_allocation_op_point_hash(const loom_op_t* op) {
  uintptr_t bits = (uintptr_t)op;
  uint64_t hash = (uint64_t)(bits >> 4);
  hash ^= hash >> 33;
  hash *= 0xFF51AFD7ED558CCDull;
  hash ^= hash >> 33;
  hash *= 0xC4CEB9FE1A85EC53ull;
  hash ^= hash >> 33;
  return (uint32_t)hash;
}

static bool loom_low_allocation_op_point_index_is_empty(
    const loom_low_allocation_op_point_index_t* index) {
  return index->liveness == NULL || index->bucket_heads == NULL ||
         index->bucket_count == 0 || index->entries == NULL;
}

static void loom_low_allocation_op_point_index_insert(
    loom_low_allocation_op_point_index_t* index, uint32_t operation_index) {
  const loom_liveness_operation_point_t* operation_point =
      &index->liveness->operation_points[operation_index];
  const uint32_t bucket_index =
      loom_low_allocation_op_point_hash(operation_point->op) &
      (index->bucket_count - 1u);
  const uint32_t entry_index = index->entry_count++;
  loom_low_allocation_op_point_entry_t* entry = &index->entries[entry_index];
  *entry = (loom_low_allocation_op_point_entry_t){
      .operation_index = operation_index,
      .next_entry = index->bucket_heads[bucket_index],
  };
  index->bucket_heads[bucket_index] = entry_index;
}

iree_status_t loom_low_allocation_op_point_index_initialize(
    const loom_liveness_analysis_t* liveness, iree_arena_allocator_t* arena,
    loom_low_allocation_op_point_index_t* out_index) {
  *out_index = (loom_low_allocation_op_point_index_t){0};

  const iree_host_size_t entry_capacity = liveness->operation_count;
  if (entry_capacity == 0) {
    return iree_ok_status();
  }
  if (entry_capacity > UINT32_MAX / 2u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low operation point index exceeds u32 range");
  }
  const uint32_t bucket_count =
      loom_low_allocation_round_up_to_power_of_two_u32(
          (uint32_t)entry_capacity * 2u);
  if (bucket_count == 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low operation point bucket count exceeds u32");
  }

  out_index->liveness = liveness;
  out_index->bucket_count = bucket_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, bucket_count, sizeof(*out_index->bucket_heads),
      (void**)&out_index->bucket_heads));
  for (uint32_t i = 0; i < bucket_count; ++i) {
    out_index->bucket_heads[i] = UINT32_MAX;
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, entry_capacity,
                                                 sizeof(*out_index->entries),
                                                 (void**)&out_index->entries));
  for (uint32_t operation_index = 0;
       operation_index < liveness->operation_count; ++operation_index) {
    loom_low_allocation_op_point_index_insert(out_index, operation_index);
  }
  return iree_ok_status();
}

bool loom_low_allocation_op_point_index_try_lookup(
    const loom_low_allocation_op_point_index_t* index, const loom_op_t* op,
    uint32_t* out_program_point) {
  *out_program_point = UINT32_MAX;
  if (loom_low_allocation_op_point_index_is_empty(index)) {
    return false;
  }
  const uint32_t bucket_index =
      loom_low_allocation_op_point_hash(op) & (index->bucket_count - 1u);
  uint32_t entry_index = index->bucket_heads[bucket_index];
  while (entry_index != UINT32_MAX) {
    const loom_low_allocation_op_point_entry_t* entry =
        &index->entries[entry_index];
    const loom_liveness_operation_point_t* operation_point =
        &index->liveness->operation_points[entry->operation_index];
    if (operation_point->op == op) {
      *out_program_point = operation_point->start_point;
      return true;
    }
    entry_index = entry->next_entry;
  }
  return false;
}

iree_status_t loom_low_allocation_op_point_index_lookup(
    const loom_low_allocation_op_point_index_t* index, const loom_op_t* op,
    uint32_t* out_program_point) {
  if (loom_low_allocation_op_point_index_try_lookup(index, op,
                                                    out_program_point)) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "low operation point index cannot find operation");
}

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

uint32_t loom_low_allocation_live_range_interval_alignment(
    const loom_liveness_interval_t* interval) {
  IREE_ASSERT_ARGUMENT(interval);
  return loom_low_register_unit_alignment(interval->unit_count);
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
  if (lhs->start_point >= rhs->end_point ||
      rhs->start_point >= lhs->end_point) {
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
  if (lhs->liveness_segments.count != 0 && rhs->liveness_segments.count != 0 &&
      !loom_liveness_segment_ranges_overlap(liveness, lhs->liveness_segments,
                                            rhs->liveness_segments)) {
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
    return true;
  }
  return false;
}
