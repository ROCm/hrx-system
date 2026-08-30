// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/storage.h"

#include <string.h>

typedef struct loom_low_allocation_explicit_register_view_t {
  const uint16_t* atomic_units;
  const uint16_t* unit_candidate_ordinals;
  uint16_t atomic_unit_count;
  uint16_t unit_count;
  uint16_t first_candidate_ordinal;
  uint16_t pressure_extent;
  uint16_t requested_unit_physical_register_id;
} loom_low_allocation_explicit_register_view_t;

typedef struct loom_low_allocation_explicit_register_subrange_t {
  loom_low_allocation_explicit_register_view_t view;
  uint32_t unit_start;
  uint32_t unit_count;
} loom_low_allocation_explicit_register_subrange_t;

static uint16_t loom_low_allocation_storage_view_unit_candidate_ordinal(
    const loom_low_allocation_explicit_register_view_t* view,
    uint32_t unit_index) {
  IREE_ASSERT_LT(unit_index, view->unit_count);
  return view->unit_candidate_ordinals
             ? view->unit_candidate_ordinals[unit_index]
             : view->first_candidate_ordinal;
}

static bool loom_low_allocation_storage_resolve_explicit_register_view(
    const loom_low_descriptor_set_t* descriptor_set,
    uint16_t descriptor_reg_class_id, uint32_t physical_register_id,
    uint32_t unit_count, uint32_t requested_unit_index,
    loom_low_allocation_explicit_register_view_t* out_view) {
  *out_view = (loom_low_allocation_explicit_register_view_t){0};
  if (descriptor_set == NULL ||
      descriptor_reg_class_id >= descriptor_set->reg_class_count ||
      unit_count == 0 || unit_count > UINT16_MAX) {
    return false;
  }
  const loom_low_reg_class_t* reg_class =
      &descriptor_set->reg_classes[descriptor_reg_class_id];
  if (!loom_low_reg_class_uses_explicit_physical_registers(reg_class) ||
      unit_count > reg_class->allocatable_count) {
    return false;
  }
  const loom_low_physical_register_t* physical_register =
      loom_low_descriptor_set_physical_register_at(descriptor_set,
                                                   physical_register_id);
  if (physical_register == NULL) {
    return false;
  }
  const uint16_t* physical_atomic_units =
      &descriptor_set->physical_register_atomic_units[physical_register
                                                          ->atomic_unit_start];

  const uint16_t* unit_candidate_ordinals = NULL;
  uint16_t first_candidate_ordinal = 0;
  if (unit_count == 1) {
    if (!loom_low_descriptor_set_find_physical_register_candidate(
            descriptor_set, descriptor_reg_class_id, physical_register_id,
            &first_candidate_ordinal)) {
      return false;
    }
  } else {
    const loom_low_physical_register_view_t* physical_register_view =
        loom_low_descriptor_set_find_physical_register_view(
            descriptor_set, descriptor_reg_class_id, physical_register_id,
            unit_count);
    if (physical_register_view == NULL) {
      return false;
    }
    unit_candidate_ordinals =
        loom_low_descriptor_set_physical_register_view_unit_candidate_ordinals(
            descriptor_set, physical_register_view);
    first_candidate_ordinal = unit_candidate_ordinals[0];
  }

  uint16_t pressure_extent = 0;
  uint16_t requested_unit_physical_register_id = 0;
  for (uint32_t unit_index = 0; unit_index < unit_count; ++unit_index) {
    const uint16_t candidate_ordinal = unit_candidate_ordinals
                                           ? unit_candidate_ordinals[unit_index]
                                           : first_candidate_ordinal;
    pressure_extent =
        iree_max(pressure_extent, (uint16_t)(candidate_ordinal + 1));
    if (unit_index == requested_unit_index) {
      requested_unit_physical_register_id =
          loom_low_descriptor_set_physical_register_candidate(
              descriptor_set, descriptor_reg_class_id, candidate_ordinal);
    }
  }

  *out_view = (loom_low_allocation_explicit_register_view_t){
      .atomic_units = physical_atomic_units,
      .unit_candidate_ordinals = unit_candidate_ordinals,
      .atomic_unit_count = physical_register->atomic_unit_count,
      .unit_count = (uint16_t)unit_count,
      .first_candidate_ordinal = first_candidate_ordinal,
      .pressure_extent = pressure_extent,
      .requested_unit_physical_register_id =
          requested_unit_physical_register_id,
  };
  return true;
}

bool loom_low_allocation_storage_explicit_physical_register_view(
    const loom_low_descriptor_set_t* descriptor_set,
    uint16_t descriptor_reg_class_id, uint32_t physical_register_id,
    uint32_t unit_count, uint32_t* out_first_candidate_ordinal,
    uint32_t* out_pressure_extent) {
  loom_low_allocation_explicit_register_view_t view;
  if (!loom_low_allocation_storage_resolve_explicit_register_view(
          descriptor_set, descriptor_reg_class_id, physical_register_id,
          unit_count, UINT32_MAX, &view)) {
    return false;
  }
  if (out_pressure_extent) {
    *out_pressure_extent = view.pressure_extent;
  }
  if (out_first_candidate_ordinal) {
    *out_first_candidate_ordinal = view.first_candidate_ordinal;
  }
  return true;
}

bool loom_low_allocation_storage_assignment_unit_physical_register(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_index,
    uint32_t* out_physical_register_id) {
  IREE_ASSERT_ARGUMENT(out_physical_register_id);
  *out_physical_register_id = UINT32_MAX;
  if (!loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, assignment) ||
      unit_index >= assignment->location_count) {
    return false;
  }
  loom_low_allocation_explicit_register_view_t view;
  if (!loom_low_allocation_storage_resolve_explicit_register_view(
          descriptor_set, assignment->descriptor_reg_class_id,
          assignment->location_base, assignment->location_count, unit_index,
          &view)) {
    return false;
  }
  *out_physical_register_id = view.requested_unit_physical_register_id;
  return true;
}

static bool loom_low_allocation_storage_resolve_explicit_subrange(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_start,
    uint32_t unit_count,
    loom_low_allocation_explicit_register_subrange_t* out_subrange) {
  *out_subrange = (loom_low_allocation_explicit_register_subrange_t){0};
  if (unit_count == 0 || unit_start > assignment->location_count ||
      unit_count > assignment->location_count - unit_start) {
    return false;
  }
  if (!loom_low_allocation_storage_resolve_explicit_register_view(
          descriptor_set, assignment->descriptor_reg_class_id,
          assignment->location_base, assignment->location_count, UINT32_MAX,
          &out_subrange->view)) {
    return false;
  }
  out_subrange->unit_start = unit_start;
  out_subrange->unit_count = unit_count;
  return true;
}

static const loom_low_physical_register_t*
loom_low_allocation_storage_subrange_unit_physical_register(
    const loom_low_descriptor_set_t* descriptor_set,
    uint16_t descriptor_reg_class_id,
    const loom_low_allocation_explicit_register_subrange_t* subrange,
    uint32_t unit_index) {
  IREE_ASSERT_LT(unit_index, subrange->unit_count);
  const uint16_t candidate_ordinal =
      loom_low_allocation_storage_view_unit_candidate_ordinal(
          &subrange->view, subrange->unit_start + unit_index);
  const uint16_t physical_register_id =
      loom_low_descriptor_set_physical_register_candidate(
          descriptor_set, descriptor_reg_class_id, candidate_ordinal);
  return loom_low_descriptor_set_physical_register_at(descriptor_set,
                                                      physical_register_id);
}

static bool loom_low_allocation_storage_sorted_atomic_units_overlap(
    const uint16_t* lhs, uint32_t lhs_count, const uint16_t* rhs,
    uint32_t rhs_count) {
  uint32_t lhs_index = 0;
  uint32_t rhs_index = 0;
  while (lhs_index < lhs_count && rhs_index < rhs_count) {
    if (lhs[lhs_index] == rhs[rhs_index]) {
      return true;
    }
    if (lhs[lhs_index] < rhs[rhs_index]) {
      ++lhs_index;
    } else {
      ++rhs_index;
    }
  }
  return false;
}

static bool loom_low_allocation_storage_sorted_atomic_units_contain(
    const uint16_t* atomic_units, uint16_t atomic_unit_count,
    uint16_t requested_atomic_unit) {
  uint16_t begin = 0;
  uint16_t end = atomic_unit_count;
  while (begin < end) {
    const uint16_t mid = begin + (uint16_t)((end - begin) / 2);
    if (atomic_units[mid] < requested_atomic_unit) {
      begin = mid + 1;
    } else {
      end = mid;
    }
  }
  return begin < atomic_unit_count &&
         atomic_units[begin] == requested_atomic_unit;
}

static uint32_t loom_low_allocation_storage_explicit_subrange_atomic_unit_count(
    const loom_low_descriptor_set_t* descriptor_set,
    uint16_t descriptor_reg_class_id,
    const loom_low_allocation_explicit_register_subrange_t* subrange) {
  uint32_t atomic_unit_count = 0;
  for (uint32_t i = 0; i < subrange->unit_count; ++i) {
    const loom_low_physical_register_t* physical_register =
        loom_low_allocation_storage_subrange_unit_physical_register(
            descriptor_set, descriptor_reg_class_id, subrange, i);
    atomic_unit_count += physical_register->atomic_unit_count;
  }
  return atomic_unit_count;
}

static bool loom_low_allocation_storage_explicit_subrange_contains_atomic_unit(
    const loom_low_descriptor_set_t* descriptor_set,
    uint16_t descriptor_reg_class_id,
    const loom_low_allocation_explicit_register_subrange_t* subrange,
    uint16_t requested_atomic_unit) {
  for (uint32_t i = 0; i < subrange->unit_count; ++i) {
    const loom_low_physical_register_t* physical_register =
        loom_low_allocation_storage_subrange_unit_physical_register(
            descriptor_set, descriptor_reg_class_id, subrange, i);
    const uint16_t* atomic_units =
        descriptor_set->physical_register_atomic_units +
        physical_register->atomic_unit_start;
    if (loom_low_allocation_storage_sorted_atomic_units_contain(
            atomic_units, physical_register->atomic_unit_count,
            requested_atomic_unit)) {
      return true;
    }
  }
  return false;
}

static bool loom_low_allocation_storage_explicit_subranges_equal(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* lhs, uint32_t lhs_start,
    uint32_t lhs_count, const loom_low_allocation_assignment_t* rhs,
    uint32_t rhs_start, uint32_t rhs_count) {
  loom_low_allocation_explicit_register_subrange_t lhs_subrange;
  loom_low_allocation_explicit_register_subrange_t rhs_subrange;
  if (!loom_low_allocation_storage_resolve_explicit_subrange(
          descriptor_set, lhs, lhs_start, lhs_count, &lhs_subrange) ||
      !loom_low_allocation_storage_resolve_explicit_subrange(
          descriptor_set, rhs, rhs_start, rhs_count, &rhs_subrange)) {
    return false;
  }
  if (lhs_start == 0 && lhs_count == lhs->location_count && rhs_start == 0 &&
      rhs_count == rhs->location_count) {
    return lhs_subrange.view.atomic_unit_count ==
               rhs_subrange.view.atomic_unit_count &&
           memcmp(lhs_subrange.view.atomic_units,
                  rhs_subrange.view.atomic_units,
                  (iree_host_size_t)lhs_subrange.view.atomic_unit_count *
                      sizeof(*lhs_subrange.view.atomic_units)) == 0;
  }
  const uint32_t lhs_atomic_unit_count =
      loom_low_allocation_storage_explicit_subrange_atomic_unit_count(
          descriptor_set, lhs->descriptor_reg_class_id, &lhs_subrange);
  const uint32_t rhs_atomic_unit_count =
      loom_low_allocation_storage_explicit_subrange_atomic_unit_count(
          descriptor_set, rhs->descriptor_reg_class_id, &rhs_subrange);
  if (lhs_atomic_unit_count != rhs_atomic_unit_count) {
    return false;
  }
  for (uint32_t i = 0; i < lhs_subrange.unit_count; ++i) {
    const loom_low_physical_register_t* physical_register =
        loom_low_allocation_storage_subrange_unit_physical_register(
            descriptor_set, lhs->descriptor_reg_class_id, &lhs_subrange, i);
    const uint16_t* atomic_units =
        descriptor_set->physical_register_atomic_units +
        physical_register->atomic_unit_start;
    for (uint16_t j = 0; j < physical_register->atomic_unit_count; ++j) {
      if (!loom_low_allocation_storage_explicit_subrange_contains_atomic_unit(
              descriptor_set, rhs->descriptor_reg_class_id, &rhs_subrange,
              atomic_units[j])) {
        return false;
      }
    }
  }
  return true;
}

static bool loom_low_allocation_storage_explicit_subranges_overlap(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* lhs, uint32_t lhs_start,
    uint32_t lhs_count, const loom_low_allocation_assignment_t* rhs,
    uint32_t rhs_start, uint32_t rhs_count) {
  loom_low_allocation_explicit_register_subrange_t lhs_subrange;
  loom_low_allocation_explicit_register_subrange_t rhs_subrange;
  if (!loom_low_allocation_storage_resolve_explicit_subrange(
          descriptor_set, lhs, lhs_start, lhs_count, &lhs_subrange) ||
      !loom_low_allocation_storage_resolve_explicit_subrange(
          descriptor_set, rhs, rhs_start, rhs_count, &rhs_subrange)) {
    return false;
  }
  if (lhs_start == 0 && lhs_count == lhs->location_count && rhs_start == 0 &&
      rhs_count == rhs->location_count) {
    return loom_low_allocation_storage_sorted_atomic_units_overlap(
        lhs_subrange.view.atomic_units, lhs_subrange.view.atomic_unit_count,
        rhs_subrange.view.atomic_units, rhs_subrange.view.atomic_unit_count);
  }
  for (uint32_t lhs_index = 0; lhs_index < lhs_subrange.unit_count;
       ++lhs_index) {
    const loom_low_physical_register_t* lhs_physical_register =
        loom_low_allocation_storage_subrange_unit_physical_register(
            descriptor_set, lhs->descriptor_reg_class_id, &lhs_subrange,
            lhs_index);
    const uint16_t* lhs_atomic_units =
        descriptor_set->physical_register_atomic_units +
        lhs_physical_register->atomic_unit_start;
    for (uint32_t rhs_index = 0; rhs_index < rhs_subrange.unit_count;
         ++rhs_index) {
      const loom_low_physical_register_t* rhs_physical_register =
          loom_low_allocation_storage_subrange_unit_physical_register(
              descriptor_set, rhs->descriptor_reg_class_id, &rhs_subrange,
              rhs_index);
      const uint16_t* rhs_atomic_units =
          descriptor_set->physical_register_atomic_units +
          rhs_physical_register->atomic_unit_start;
      if (loom_low_allocation_storage_sorted_atomic_units_overlap(
              lhs_atomic_units, lhs_physical_register->atomic_unit_count,
              rhs_atomic_units, rhs_physical_register->atomic_unit_count)) {
        return true;
      }
    }
  }
  return false;
}

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
  loom_low_allocation_explicit_register_view_t view;
  const bool resolved =
      loom_low_allocation_storage_resolve_explicit_register_view(
          descriptor_set, assignment->descriptor_reg_class_id,
          assignment->location_base, assignment->location_count, UINT32_MAX,
          &view);
  IREE_ASSERT_TRUE(resolved);
  return view.atomic_unit_count;
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
  loom_low_allocation_explicit_register_view_t view;
  const bool resolved =
      loom_low_allocation_storage_resolve_explicit_register_view(
          descriptor_set, assignment->descriptor_reg_class_id,
          assignment->location_base, assignment->location_count, UINT32_MAX,
          &view);
  IREE_ASSERT_TRUE(resolved);
  IREE_ASSERT_LT(atomic_unit_ordinal, view.atomic_unit_count);
  *out_storage_key = 0;
  *out_location = view.atomic_units[atomic_unit_ordinal];
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
  uint32_t pressure_extent = 0;
  const bool resolved =
      loom_low_allocation_storage_explicit_physical_register_view(
          descriptor_set, assignment->descriptor_reg_class_id,
          assignment->location_base, assignment->location_count,
          /*out_first_candidate_ordinal=*/NULL, &pressure_extent);
  IREE_ASSERT_TRUE(resolved);
  return pressure_extent;
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
  const bool lhs_is_explicit =
      loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, lhs);
  const bool rhs_is_explicit =
      loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, rhs);
  if (lhs_is_explicit || rhs_is_explicit) {
    if (!lhs_is_explicit || !rhs_is_explicit) {
      return false;
    }
    return loom_low_allocation_storage_explicit_subranges_equal(
        descriptor_set, lhs, 0, lhs->location_count, rhs, 0,
        rhs->location_count);
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
  if (lhs->location_count != 0 && rhs->location_count != 0) {
    return loom_low_allocation_storage_assignment_ranges_equal(descriptor_set,
                                                               lhs, rhs);
  }
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
    return loom_low_allocation_storage_explicit_subranges_overlap(
        descriptor_set, lhs, 0, lhs->location_count, rhs, 0,
        rhs->location_count);
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
    return loom_low_allocation_storage_explicit_subranges_equal(
        descriptor_set, lhs, lhs_start, unit_count, rhs, rhs_start, unit_count);
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
    return loom_low_allocation_storage_explicit_subranges_overlap(
        descriptor_set, lhs, lhs_start, unit_count, rhs, rhs_start, unit_count);
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
