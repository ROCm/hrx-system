// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/storage.h"

loom_low_allocation_location_kind_t
loom_low_allocation_storage_reg_class_location_kind(
    const loom_low_reg_class_t* reg_class) {
  IREE_ASSERT_ARGUMENT(reg_class);
  if (reg_class->allocatable_count > 0 ||
      iree_any_bit_set(reg_class->flags, LOOM_LOW_REG_CLASS_FLAG_PHYSICAL)) {
    return LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  }
  return LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID;
}

bool loom_low_allocation_storage_reg_classes_share(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t lhs_reg_class_id,
    uint16_t rhs_reg_class_id) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  if (lhs_reg_class_id == rhs_reg_class_id) {
    return true;
  }
  if (lhs_reg_class_id >= descriptor_set->reg_class_count ||
      rhs_reg_class_id >= descriptor_set->reg_class_count) {
    return false;
  }
  return loom_low_reg_class_storage_key(descriptor_set, lhs_reg_class_id) ==
         loom_low_reg_class_storage_key(descriptor_set, rhs_reg_class_id);
}

bool loom_low_allocation_storage_assignment_classes_share(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(lhs);
  IREE_ASSERT_ARGUMENT(rhs);
  return lhs->location_kind == rhs->location_kind &&
         loom_low_allocation_storage_reg_classes_share(
             descriptor_set, lhs->descriptor_reg_class_id,
             rhs->descriptor_reg_class_id);
}

bool loom_low_allocation_storage_assignment_ranges_equal(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(lhs);
  IREE_ASSERT_ARGUMENT(rhs);
  return lhs->location_count != 0 && rhs->location_count != 0 &&
         loom_low_allocation_storage_assignment_classes_share(descriptor_set,
                                                              lhs, rhs) &&
         lhs->location_base == rhs->location_base &&
         lhs->location_count == rhs->location_count;
}

bool loom_low_allocation_storage_assignment_locations_share(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(lhs);
  IREE_ASSERT_ARGUMENT(rhs);
  return lhs->location_base == rhs->location_base &&
         lhs->location_count == rhs->location_count &&
         loom_low_allocation_storage_assignment_classes_share(descriptor_set,
                                                              lhs, rhs);
}

bool loom_low_allocation_storage_assignment_ranges_overlap(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(lhs);
  IREE_ASSERT_ARGUMENT(rhs);
  if (lhs->location_count == 0 || rhs->location_count == 0 ||
      !loom_low_allocation_storage_assignment_classes_share(descriptor_set, lhs,
                                                            rhs)) {
    return false;
  }
  const uint64_t lhs_begin = lhs->location_base;
  const uint64_t rhs_begin = rhs->location_base;
  const uint64_t lhs_end = lhs_begin + lhs->location_count;
  const uint64_t rhs_end = rhs_begin + rhs->location_count;
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

bool loom_low_allocation_storage_assignment_subranges_equal(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* lhs, uint32_t lhs_start,
    const loom_low_allocation_assignment_t* rhs, uint32_t rhs_start,
    uint32_t unit_count) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(lhs);
  IREE_ASSERT_ARGUMENT(rhs);
  return unit_count != 0 &&
         loom_low_allocation_storage_assignment_classes_share(descriptor_set,
                                                              lhs, rhs) &&
         (uint64_t)lhs->location_base + lhs_start ==
             (uint64_t)rhs->location_base + rhs_start;
}

bool loom_low_allocation_storage_assignment_subranges_overlap(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* lhs, uint32_t lhs_start,
    const loom_low_allocation_assignment_t* rhs, uint32_t rhs_start,
    uint32_t unit_count) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(lhs);
  IREE_ASSERT_ARGUMENT(rhs);
  if (!loom_low_allocation_storage_assignment_classes_share(descriptor_set, lhs,
                                                            rhs)) {
    return false;
  }
  const uint64_t lhs_begin = (uint64_t)lhs->location_base + lhs_start;
  const uint64_t rhs_begin = (uint64_t)rhs->location_base + rhs_start;
  const uint64_t lhs_end = lhs_begin + unit_count;
  const uint64_t rhs_end = rhs_begin + unit_count;
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}
