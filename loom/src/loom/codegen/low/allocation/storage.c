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

bool loom_low_allocation_storage_assignment_uses_explicit_physical_register(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignment) {
  if (assignment->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
      assignment->descriptor_reg_class_id >= descriptor_set->reg_class_count) {
    return false;
  }
  return loom_low_reg_class_uses_explicit_physical_registers(
      &descriptor_set->reg_classes[assignment->descriptor_reg_class_id]);
}

uint32_t loom_low_allocation_storage_assignment_atomic_unit_count(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignment) {
  if (!loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, assignment)) {
    return assignment->location_count;
  }
  const loom_low_physical_register_t* physical_register =
      loom_low_descriptor_set_physical_register_at(descriptor_set,
                                                   assignment->location_base);
  IREE_ASSERT(physical_register != NULL);
  return physical_register->atomic_unit_count;
}

void loom_low_allocation_storage_assignment_atomic_unit(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignment,
    uint32_t atomic_unit_ordinal, uint32_t* out_storage_key,
    uint32_t* out_location) {
  if (!loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, assignment)) {
    *out_storage_key = loom_low_reg_class_storage_key(
        descriptor_set, assignment->descriptor_reg_class_id);
    *out_location = assignment->location_base + atomic_unit_ordinal;
    return;
  }
  const loom_low_physical_register_t* physical_register =
      loom_low_descriptor_set_physical_register_at(descriptor_set,
                                                   assignment->location_base);
  IREE_ASSERT(physical_register != NULL);
  IREE_ASSERT_LT(atomic_unit_ordinal, physical_register->atomic_unit_count);
  *out_storage_key = 0;
  *out_location =
      descriptor_set->physical_register_atomic_units
          [physical_register->atomic_unit_start + atomic_unit_ordinal];
}

uint32_t loom_low_allocation_storage_assignment_pressure_extent(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignment) {
  if (!loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, assignment)) {
    const uint64_t end =
        (uint64_t)assignment->location_base + assignment->location_count;
    return end > UINT32_MAX ? UINT32_MAX : (uint32_t)end;
  }
  uint16_t candidate_ordinal = 0;
  const bool found = loom_low_descriptor_set_find_physical_register_candidate(
      descriptor_set, assignment->descriptor_reg_class_id,
      assignment->location_base, &candidate_ordinal);
  IREE_ASSERT_TRUE(found);
  return (uint32_t)candidate_ordinal + 1;
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
  const bool lhs_is_explicit =
      loom_low_reg_class_uses_explicit_physical_registers(
          &descriptor_set->reg_classes[lhs_reg_class_id]);
  const bool rhs_is_explicit =
      loom_low_reg_class_uses_explicit_physical_registers(
          &descriptor_set->reg_classes[rhs_reg_class_id]);
  if (lhs_is_explicit || rhs_is_explicit) {
    return lhs_is_explicit && rhs_is_explicit;
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
  if (lhs->location_count == 0 || rhs->location_count == 0 ||
      !loom_low_allocation_storage_assignment_classes_share(descriptor_set, lhs,
                                                            rhs)) {
    return false;
  }
  return lhs->location_base == rhs->location_base &&
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
  const bool lhs_is_explicit =
      loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, lhs);
  const bool rhs_is_explicit =
      loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, rhs);
  if (lhs_is_explicit || rhs_is_explicit) {
    if (!lhs_is_explicit || !rhs_is_explicit) return false;
    uint16_t lhs_atomic_unit_count = 0;
    const uint16_t* lhs_atomic_units =
        loom_low_descriptor_set_physical_register_atomic_units(
            descriptor_set, lhs->location_base, &lhs_atomic_unit_count);
    uint16_t rhs_atomic_unit_count = 0;
    const uint16_t* rhs_atomic_units =
        loom_low_descriptor_set_physical_register_atomic_units(
            descriptor_set, rhs->location_base, &rhs_atomic_unit_count);
    uint16_t lhs_index = 0;
    uint16_t rhs_index = 0;
    while (lhs_index < lhs_atomic_unit_count &&
           rhs_index < rhs_atomic_unit_count) {
      if (lhs_atomic_units[lhs_index] == rhs_atomic_units[rhs_index]) {
        return true;
      }
      if (lhs_atomic_units[lhs_index] < rhs_atomic_units[rhs_index]) {
        ++lhs_index;
      } else {
        ++rhs_index;
      }
    }
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
  if (loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, lhs) ||
      loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, rhs)) {
    return lhs_start == 0 && rhs_start == 0 && unit_count == 1 &&
           loom_low_allocation_storage_assignment_ranges_equal(descriptor_set,
                                                               lhs, rhs);
  }
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
  if (loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, lhs) ||
      loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, rhs)) {
    return lhs_start == 0 && rhs_start == 0 && unit_count == 1 &&
           loom_low_allocation_storage_assignment_ranges_overlap(descriptor_set,
                                                                 lhs, rhs);
  }
  const uint64_t lhs_begin = (uint64_t)lhs->location_base + lhs_start;
  const uint64_t rhs_begin = (uint64_t)rhs->location_base + rhs_start;
  const uint64_t lhs_end = lhs_begin + unit_count;
  const uint64_t rhs_end = rhs_begin + unit_count;
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

bool loom_low_allocation_storage_placement_relation_satisfied(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_placement_relation_t* relation,
    const loom_low_allocation_assignment_t* result_assignment,
    const loom_low_allocation_assignment_t* source_assignment) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(relation);
  IREE_ASSERT_ARGUMENT(result_assignment);
  IREE_ASSERT_ARGUMENT(source_assignment);
  IREE_ASSERT_LE(relation->result_unit_offset, result_assignment->unit_count);
  IREE_ASSERT_LE(relation->unit_count,
                 result_assignment->unit_count - relation->result_unit_offset);
  IREE_ASSERT_LE(relation->source_unit_offset, source_assignment->unit_count);
  IREE_ASSERT_LE(relation->unit_count,
                 source_assignment->unit_count - relation->source_unit_offset);

  switch (relation->kind) {
    case LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE:
    case LOOM_LOW_PLACEMENT_RELATION_SUBRANGE:
    case LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART:
      return loom_low_allocation_storage_assignment_subranges_equal(
          descriptor_set, result_assignment, relation->result_unit_offset,
          source_assignment, relation->source_unit_offset,
          relation->unit_count);
    case LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION:
      IREE_ASSERT_NE(relation->location_mask, 0u);
      if (!loom_low_allocation_storage_assignment_classes_share(
              descriptor_set, result_assignment, source_assignment)) {
        return false;
      }
      if (loom_low_allocation_storage_assignment_uses_explicit_physical_register(
              descriptor_set, result_assignment)) {
        return false;
      }
      return ((((uint64_t)result_assignment->location_base +
                relation->result_unit_offset) ^
               ((uint64_t)source_assignment->location_base +
                relation->source_unit_offset)) &
              relation->location_mask) != 0;
    case LOOM_LOW_PLACEMENT_RELATION_DISJOINT_STORAGE:
      return !loom_low_allocation_storage_assignment_subranges_overlap(
          descriptor_set, result_assignment, relation->result_unit_offset,
          source_assignment, relation->source_unit_offset,
          relation->unit_count);
    default:
      return false;
  }
}
