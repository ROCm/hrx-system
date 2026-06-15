// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/assignment.h"

bool loom_low_allocation_location_kind_is_known(
    loom_low_allocation_location_kind_t location_kind) {
  switch (location_kind) {
    case LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED:
    case LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER:
    case LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID:
    case LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT:
      return true;
    default:
      return false;
  }
}

iree_string_view_t loom_low_allocation_location_kind_name(
    loom_low_allocation_location_kind_t location_kind) {
  switch (location_kind) {
    case LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED:
      return IREE_SV("unassigned");
    case LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER:
      return IREE_SV("physical_register");
    case LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID:
      return IREE_SV("target_id");
    case LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT:
      return IREE_SV("spill_slot");
    default:
      return IREE_SV("unknown");
  }
}

bool loom_low_allocation_location_kind_is_register_like(
    loom_low_allocation_location_kind_t location_kind) {
  return location_kind == LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
         location_kind == LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID;
}

bool loom_low_allocation_assignment_storage_class_equal(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  IREE_ASSERT_ARGUMENT(lhs);
  IREE_ASSERT_ARGUMENT(rhs);
  return lhs->location_kind == rhs->location_kind &&
         lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id;
}

bool loom_low_allocation_assignment_is_physical_register_class(
    const loom_low_allocation_assignment_t* assignment,
    uint16_t descriptor_reg_class_id) {
  IREE_ASSERT_ARGUMENT(assignment);
  return assignment->location_kind ==
             LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER &&
         assignment->descriptor_reg_class_id == descriptor_reg_class_id &&
         assignment->location_count != 0;
}

bool loom_low_allocation_assignment_is_register_like(
    const loom_low_allocation_assignment_t* assignment) {
  IREE_ASSERT_ARGUMENT(assignment);
  return loom_low_allocation_location_kind_is_register_like(
      assignment->location_kind);
}

bool loom_low_allocation_assignment_is_spill_slot(
    const loom_low_allocation_assignment_t* assignment) {
  IREE_ASSERT_ARGUMENT(assignment);
  return assignment->location_kind == LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT;
}

bool loom_low_allocation_assignment_location_exclusive_end(
    const loom_low_allocation_assignment_t* assignment, uint64_t* out_end) {
  IREE_ASSERT_ARGUMENT(assignment);
  IREE_ASSERT_ARGUMENT(out_end);
  *out_end = 0;
  if (assignment->location_count == 0) {
    return false;
  }
  *out_end = (uint64_t)assignment->location_base + assignment->location_count;
  return true;
}

bool loom_low_allocation_assignment_location_range_equal(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  IREE_ASSERT_ARGUMENT(lhs);
  IREE_ASSERT_ARGUMENT(rhs);
  return lhs->location_count != 0 && rhs->location_count != 0 &&
         loom_low_allocation_assignment_storage_class_equal(lhs, rhs) &&
         lhs->location_base == rhs->location_base &&
         lhs->location_count == rhs->location_count;
}

bool loom_low_allocation_assignment_location_ranges_overlap(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  IREE_ASSERT_ARGUMENT(lhs);
  IREE_ASSERT_ARGUMENT(rhs);
  if (lhs->location_count == 0 || rhs->location_count == 0 ||
      !loom_low_allocation_assignment_storage_class_equal(lhs, rhs)) {
    return false;
  }
  const uint64_t lhs_begin = lhs->location_base;
  const uint64_t rhs_begin = rhs->location_base;
  const uint64_t lhs_end = lhs_begin + lhs->location_count;
  const uint64_t rhs_end = rhs_begin + rhs->location_count;
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

bool loom_low_allocation_assignment_subranges_equal(
    const loom_low_allocation_assignment_t* lhs, uint32_t lhs_start,
    const loom_low_allocation_assignment_t* rhs, uint32_t rhs_start,
    uint32_t unit_count) {
  IREE_ASSERT_ARGUMENT(lhs);
  IREE_ASSERT_ARGUMENT(rhs);
  return unit_count != 0 &&
         loom_low_allocation_assignment_storage_class_equal(lhs, rhs) &&
         (uint64_t)lhs->location_base + lhs_start ==
             (uint64_t)rhs->location_base + rhs_start;
}

bool loom_low_allocation_assignment_subranges_overlap(
    const loom_low_allocation_assignment_t* lhs, uint32_t lhs_start,
    const loom_low_allocation_assignment_t* rhs, uint32_t rhs_start,
    uint32_t unit_count) {
  IREE_ASSERT_ARGUMENT(lhs);
  IREE_ASSERT_ARGUMENT(rhs);
  if (!loom_low_allocation_assignment_storage_class_equal(lhs, rhs)) {
    return false;
  }
  const uint64_t lhs_begin = (uint64_t)lhs->location_base + lhs_start;
  const uint64_t rhs_begin = (uint64_t)rhs->location_base + rhs_start;
  const uint64_t lhs_end = lhs_begin + unit_count;
  const uint64_t rhs_end = rhs_begin + unit_count;
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}
