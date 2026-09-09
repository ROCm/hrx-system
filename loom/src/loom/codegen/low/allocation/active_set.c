// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/active_set.h"

#include <string.h>

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

static bool loom_low_allocation_active_set_scan_conflicts(
    const loom_low_allocation_active_set_t* active_set,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count) {
  for (iree_host_size_t i = 0; i < active_set->count; ++i) {
    const uint32_t assignment_index = active_set->assignment_indices[i];
    IREE_ASSERT_LT(assignment_index, assignment_count);
    const loom_low_allocation_assignment_t* existing =
        &assignments[assignment_index];
    if (loom_low_allocation_active_assignment_conflicts(
            descriptor_set, active_set->liveness, unit_liveness, existing,
            candidate, ignored_value_ids, ignored_value_count)) {
      return true;
    }
  }
  return false;
}

iree_status_t loom_low_allocation_active_set_initialize(
    const loom_liveness_analysis_t* liveness,
    iree_host_size_t assignment_capacity, iree_host_size_t program_point_count,
    iree_host_size_t unit_capacity, iree_arena_allocator_t* arena,
    loom_low_allocation_active_set_t* out_active_set) {
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_active_set);
  *out_active_set = (loom_low_allocation_active_set_t){0};
  out_active_set->liveness = liveness;
  out_active_set->program_point_count = program_point_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, assignment_capacity, sizeof(*out_active_set->assignment_indices),
      (void**)&out_active_set->assignment_indices));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, assignment_capacity, sizeof(*out_active_set->entries),
      (void**)&out_active_set->entries));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, program_point_count, sizeof(*out_active_set->expiration_heads),
      (void**)&out_active_set->expiration_heads));
  if (program_point_count != 0) {
    memset(out_active_set->expiration_heads, 0xFF,
           program_point_count * sizeof(*out_active_set->expiration_heads));
  }
  return loom_low_allocation_active_unit_index_initialize(
      assignment_capacity, unit_capacity, arena, &out_active_set->units);
}

bool loom_low_allocation_active_assignment_conflicts(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_allocation_assignment_t* existing,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(unit_liveness);
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
      descriptor_set, liveness, unit_liveness->start_points,
      unit_liveness->end_points, unit_liveness->point_count, existing,
      candidate);
}

bool loom_low_allocation_active_set_conflicts(
    loom_low_allocation_active_set_t* active_set,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count) {
  IREE_ASSERT_ARGUMENT(active_set);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(assignments);
  IREE_ASSERT_ARGUMENT(candidate);
  if (loom_low_allocation_active_unit_index_is_enabled(&active_set->units)) {
    return loom_low_allocation_active_unit_index_conflicts(
        &active_set->units, descriptor_set, active_set->liveness, unit_liveness,
        assignments, assignment_count, candidate, ignored_value_ids,
        ignored_value_count);
  }
  return loom_low_allocation_active_set_scan_conflicts(
      active_set, descriptor_set, unit_liveness, assignments, assignment_count,
      candidate, ignored_value_ids, ignored_value_count);
}

void loom_low_allocation_active_set_remove(
    loom_low_allocation_active_set_t* active_set,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, uint32_t assignment_index) {
  loom_low_allocation_active_entry_t* entry =
      &active_set->entries[assignment_index];
  IREE_ASSERT_LT(entry->position, active_set->count);
  loom_low_allocation_active_unit_index_remove_assignment(
      &active_set->units, assignments, assignment_count, assignment_index);
  const uint32_t last_assignment_index =
      active_set->assignment_indices[--active_set->count];
  active_set->assignment_indices[entry->position] = last_assignment_index;
  active_set->entries[last_assignment_index].position = entry->position;
  entry->position = UINT32_MAX;
}

void loom_low_allocation_active_set_expire(
    loom_low_allocation_active_set_t* active_set,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, uint32_t start_point) {
  const iree_host_size_t end_point =
      iree_min((uint64_t)start_point + 1, active_set->program_point_count);
  while (active_set->next_expiration_point < end_point) {
    uint32_t assignment_index =
        active_set->expiration_heads[active_set->next_expiration_point++];
    while (assignment_index != UINT32_MAX) {
      const loom_low_allocation_active_entry_t* entry =
          &active_set->entries[assignment_index];
      if (entry->position != UINT32_MAX) {
        loom_low_allocation_active_set_remove(
            active_set, assignments, assignment_count, assignment_index);
      }
      assignment_index = entry->next_expiration;
    }
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
  const uint32_t end_point = assignments[assignment_index].end_point;
  IREE_ASSERT_LT(end_point, active_set->program_point_count);
  IREE_ASSERT_GE(end_point, active_set->next_expiration_point);
  active_set->entries[assignment_index] = (loom_low_allocation_active_entry_t){
      .position = (uint32_t)active_set->count,
      .next_expiration = active_set->expiration_heads[end_point],
  };
  active_set->expiration_heads[end_point] = assignment_index;
  active_set->assignment_indices[active_set->count++] = assignment_index;
  loom_low_allocation_active_unit_index_insert_assignment(
      &active_set->units, descriptor_set, assignments, assignment_count,
      assignment_index);
}
