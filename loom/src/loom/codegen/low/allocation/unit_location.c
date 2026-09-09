// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/unit_location.h"

#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/storage.h"

loom_low_move_location_t loom_low_allocation_assignment_unit_location(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_index) {
  IREE_ASSERT_LT(unit_index, assignment->location_count);
  uint32_t location = assignment->location_base + unit_index;
  if (loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, assignment)) {
    const bool resolved =
        loom_low_allocation_storage_assignment_unit_physical_register(
            descriptor_set, assignment, unit_index, &location);
    IREE_ASSERT_TRUE(resolved);
  }
  return (loom_low_move_location_t){
      .location_kind = assignment->location_kind,
      .descriptor_reg_class_id = assignment->descriptor_reg_class_id,
      .location = location,
  };
}

bool loom_low_allocation_unit_locations_equal(
    const loom_low_move_location_t* lhs, const loom_low_move_location_t* rhs) {
  IREE_ASSERT_ARGUMENT(lhs);
  IREE_ASSERT_ARGUMENT(rhs);
  return lhs->location_kind == rhs->location_kind &&
         lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id &&
         lhs->location == rhs->location;
}

bool loom_low_allocation_unit_storage_classes_equal(
    const loom_low_move_location_t* lhs, const loom_low_move_location_t* rhs) {
  IREE_ASSERT_ARGUMENT(lhs);
  IREE_ASSERT_ARGUMENT(rhs);
  return lhs->location_kind == rhs->location_kind &&
         lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id;
}

bool loom_low_allocation_unit_locations_form_register_move(
    const loom_low_move_location_t* source,
    const loom_low_move_location_t* destination) {
  IREE_ASSERT_ARGUMENT(source);
  IREE_ASSERT_ARGUMENT(destination);
  return loom_low_allocation_location_kind_is_register_like(
             source->location_kind) &&
         loom_low_allocation_location_kind_is_register_like(
             destination->location_kind) &&
         !loom_low_allocation_unit_locations_equal(source, destination);
}

bool loom_low_allocation_unit_location_is_live_at_point(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_move_location_t* location, uint32_t point) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(unit_liveness);
  IREE_ASSERT_ARGUMENT(location);
  const loom_low_allocation_assignment_t location_assignment = {
      .descriptor_reg_class_id = location->descriptor_reg_class_id,
      .location_kind = location->location_kind,
      .location_base = location->location,
      .location_count = 1,
  };
  for (iree_host_size_t i = 0; i < assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment = &assignments[i];
    if (assignment->location_kind != location->location_kind ||
        point < assignment->start_point) {
      continue;
    }
    if (!loom_low_allocation_storage_assignment_ranges_overlap(
            descriptor_set, assignment, &location_assignment)) {
      continue;
    }
    // A single explicit wide register can alias several independently live
    // units of the assignment. Every overlapping unit must be dead before
    // that register is available as scratch. Linear classes map one-to-one.
    const bool is_explicit =
        loom_low_allocation_storage_assignment_uses_explicit_physical_register(
            descriptor_set, assignment);
    const uint32_t unit_begin =
        is_explicit ? 0 : location->location - assignment->location_base;
    const uint32_t unit_end =
        is_explicit ? assignment->location_count : unit_begin + 1u;
    for (uint32_t unit_offset = unit_begin; unit_offset < unit_end;
         ++unit_offset) {
      const uint32_t unit_start_point =
          loom_low_allocation_live_range_assignment_unit_start_point(
              unit_liveness->start_points, unit_liveness->point_count,
              assignment, unit_offset);
      const uint32_t unit_end_point =
          loom_low_allocation_live_range_assignment_unit_end_point(
              unit_liveness->end_points, unit_liveness->point_count, assignment,
              unit_offset);
      if (point < unit_start_point || point >= unit_end_point) continue;
      if (!is_explicit ||
          loom_low_allocation_storage_assignment_subranges_overlap(
              descriptor_set, assignment, unit_offset, &location_assignment, 0,
              /*unit_count=*/1)) {
        return true;
      }
    }
  }
  return false;
}
