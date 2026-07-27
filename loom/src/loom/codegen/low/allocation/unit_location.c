// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/unit_location.h"

#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/storage.h"

iree_status_t loom_low_allocation_assignment_unit_location(
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_index,
    loom_low_allocation_unit_location_t* out_location) {
  IREE_ASSERT_ARGUMENT(assignment);
  IREE_ASSERT_ARGUMENT(out_location);
  *out_location = (loom_low_allocation_unit_location_t){0};
  if (unit_index >= assignment->location_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation unit index exceeds assignment range");
  }
  if (assignment->location_base > UINT32_MAX - unit_index) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation unit location exceeds uint32_t");
  }
  *out_location = (loom_low_allocation_unit_location_t){
      .location_kind = assignment->location_kind,
      .value_class = assignment->value_class,
      .descriptor_reg_class_id = assignment->descriptor_reg_class_id,
      .location = assignment->location_base + unit_index,
  };
  return iree_ok_status();
}

bool loom_low_allocation_unit_locations_equal(
    const loom_low_allocation_unit_location_t* lhs,
    const loom_low_allocation_unit_location_t* rhs) {
  IREE_ASSERT_ARGUMENT(lhs);
  IREE_ASSERT_ARGUMENT(rhs);
  return lhs->location_kind == rhs->location_kind &&
         lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id &&
         lhs->location == rhs->location;
}

bool loom_low_allocation_unit_storage_classes_equal(
    const loom_low_allocation_unit_location_t* lhs,
    const loom_low_allocation_unit_location_t* rhs) {
  IREE_ASSERT_ARGUMENT(lhs);
  IREE_ASSERT_ARGUMENT(rhs);
  return lhs->location_kind == rhs->location_kind &&
         loom_liveness_value_class_equal(lhs->value_class, rhs->value_class) &&
         lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id;
}

bool loom_low_allocation_unit_locations_form_register_move(
    const loom_low_allocation_unit_location_t* source,
    const loom_low_allocation_unit_location_t* destination) {
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
    iree_host_size_t assignment_count, const uint32_t* unit_end_points,
    iree_host_size_t unit_end_point_count,
    const loom_low_allocation_unit_location_t* location, uint32_t point) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(location);
  for (iree_host_size_t i = 0; i < assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment = &assignments[i];
    if (assignment->location_kind != location->location_kind ||
        point < assignment->start_point) {
      continue;
    }
    if (!loom_low_allocation_storage_reg_classes_share(
            descriptor_set, assignment->descriptor_reg_class_id,
            location->descriptor_reg_class_id)) {
      continue;
    }
    const uint64_t assignment_end =
        (uint64_t)assignment->location_base + assignment->location_count;
    if (location->location < assignment->location_base ||
        location->location >= assignment_end) {
      continue;
    }
    const uint32_t unit_offset =
        (uint32_t)(location->location - assignment->location_base);
    if (point <
        loom_low_allocation_live_range_assignment_unit_end_point(
            unit_end_points, unit_end_point_count, assignment, unit_offset)) {
      return true;
    }
  }
  return false;
}
