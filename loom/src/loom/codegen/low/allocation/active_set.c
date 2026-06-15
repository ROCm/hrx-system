// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/active_set.h"

#include "loom/codegen/low/allocation/live_range.h"

static bool loom_low_allocation_value_id_is_ignored(
    loom_value_id_t value_id, const loom_value_id_t* ignored_value_ids,
    uint16_t ignored_value_count) {
  for (uint16_t i = 0; i < ignored_value_count; ++i) {
    if (ignored_value_ids[i] == value_id) {
      return true;
    }
  }
  return false;
}

static bool loom_low_allocation_active_assignment_less(
    const loom_low_allocation_assignment_t* assignments, uint32_t lhs_index,
    uint32_t rhs_index) {
  const loom_low_allocation_assignment_t* lhs = &assignments[lhs_index];
  const loom_low_allocation_assignment_t* rhs = &assignments[rhs_index];
  if (lhs->end_point != rhs->end_point) {
    return lhs->end_point < rhs->end_point;
  }
  if (lhs->start_point != rhs->start_point) {
    return lhs->start_point < rhs->start_point;
  }
  if (lhs->value_id != rhs->value_id) {
    return lhs->value_id < rhs->value_id;
  }
  return lhs_index < rhs_index;
}

static bool loom_low_allocation_active_set_scan_conflicts(
    const loom_low_allocation_active_set_t* active_set,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness, const uint32_t* unit_end_points,
    iree_host_size_t unit_end_point_count,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count,
    bool unindexed_only) {
  for (iree_host_size_t i = 0; i < active_set->count; ++i) {
    const uint32_t assignment_index =
        active_set->assignment_indices[active_set->start + i];
    IREE_ASSERT_LT(assignment_index, assignment_count);
    const loom_low_allocation_assignment_t* existing =
        &assignments[assignment_index];
    const bool assignment_is_indexed =
        loom_low_allocation_active_unit_index_contains_assignment(
            &active_set->units, assignment_index);
    if (unindexed_only && assignment_is_indexed) {
      continue;
    }
    if (loom_low_allocation_active_assignment_conflicts(
            descriptor_set, liveness, unit_end_points, unit_end_point_count,
            existing, candidate, ignored_value_ids, ignored_value_count)) {
      return true;
    }
  }
  return false;
}

iree_status_t loom_low_allocation_active_set_initialize(
    iree_host_size_t assignment_capacity, iree_host_size_t unit_capacity,
    iree_arena_allocator_t* arena,
    loom_low_allocation_active_set_t* out_active_set) {
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_active_set);
  *out_active_set = (loom_low_allocation_active_set_t){0};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, assignment_capacity, sizeof(*out_active_set->assignment_indices),
      (void**)&out_active_set->assignment_indices));
  return loom_low_allocation_active_unit_index_initialize(
      assignment_capacity, unit_capacity, arena, &out_active_set->units);
}

bool loom_low_allocation_active_assignment_conflicts(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness, const uint32_t* unit_end_points,
    iree_host_size_t unit_end_point_count,
    const loom_low_allocation_assignment_t* existing,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(existing);
  IREE_ASSERT_ARGUMENT(candidate);
  if (loom_low_allocation_value_id_is_ignored(
          existing->value_id, ignored_value_ids, ignored_value_count)) {
    return false;
  }
  if (existing->location_kind != candidate->location_kind) {
    return false;
  }
  return loom_low_allocation_live_range_assignments_conflict(
      descriptor_set, liveness, unit_end_points, unit_end_point_count, existing,
      candidate);
}

bool loom_low_allocation_active_set_conflicts(
    loom_low_allocation_active_set_t* active_set,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness, const uint32_t* unit_end_points,
    iree_host_size_t unit_end_point_count,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count) {
  IREE_ASSERT_ARGUMENT(active_set);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(assignments);
  IREE_ASSERT_ARGUMENT(candidate);
  if (loom_low_allocation_active_unit_index_is_enabled(&active_set->units)) {
    if (loom_low_allocation_active_unit_index_conflicts(
            &active_set->units, descriptor_set, liveness, unit_end_points,
            unit_end_point_count, assignments, assignment_count, candidate,
            ignored_value_ids, ignored_value_count)) {
      return true;
    }
    return loom_low_allocation_active_unit_index_unindexed_count(
               &active_set->units) != 0 &&
           loom_low_allocation_active_set_scan_conflicts(
               active_set, descriptor_set, liveness, unit_end_points,
               unit_end_point_count, assignments, assignment_count, candidate,
               ignored_value_ids, ignored_value_count,
               /*unindexed_only=*/true);
  }
  return loom_low_allocation_active_set_scan_conflicts(
      active_set, descriptor_set, liveness, unit_end_points,
      unit_end_point_count, assignments, assignment_count, candidate,
      ignored_value_ids, ignored_value_count, /*unindexed_only=*/false);
}

void loom_low_allocation_active_set_expire(
    loom_low_allocation_active_set_t* active_set,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, uint32_t start_point) {
  IREE_ASSERT_ARGUMENT(active_set);
  IREE_ASSERT_ARGUMENT(assignments);
  while (active_set->count > 0) {
    const uint32_t assignment_index =
        active_set->assignment_indices[active_set->start];
    IREE_ASSERT_LT(assignment_index, assignment_count);
    const loom_low_allocation_assignment_t* assignment =
        &assignments[assignment_index];
    if (assignment->end_point > start_point) {
      return;
    }
    loom_low_allocation_active_unit_index_remove_assignment(
        &active_set->units, assignments, assignment_count, assignment_index);
    ++active_set->start;
    --active_set->count;
  }
}

void loom_low_allocation_active_set_insert(
    loom_low_allocation_active_set_t* active_set,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, uint32_t assignment_index) {
  IREE_ASSERT_ARGUMENT(active_set);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(assignments);
  IREE_ASSERT(active_set->assignment_indices != NULL);
  IREE_ASSERT_LT(assignment_index, assignment_count);
  iree_host_size_t insert_index = active_set->start + active_set->count;
  while (insert_index > active_set->start) {
    const uint32_t previous_assignment_index =
        active_set->assignment_indices[insert_index - 1];
    if (loom_low_allocation_active_assignment_less(
            assignments, previous_assignment_index, assignment_index)) {
      break;
    }
    active_set->assignment_indices[insert_index] = previous_assignment_index;
    --insert_index;
  }
  active_set->assignment_indices[insert_index] = assignment_index;
  ++active_set->count;
  loom_low_allocation_active_unit_index_insert_assignment(
      &active_set->units, descriptor_set, assignments, assignment_count,
      assignment_index);
}

void loom_low_allocation_active_set_remove_assignment_units(
    loom_low_allocation_active_set_t* active_set,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, uint32_t assignment_index) {
  IREE_ASSERT_ARGUMENT(active_set);
  IREE_ASSERT_ARGUMENT(assignments);
  loom_low_allocation_active_unit_index_remove_assignment(
      &active_set->units, assignments, assignment_count, assignment_index);
}
