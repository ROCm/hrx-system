// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/active_capacity.h"

#include <string.h>

#include "loom/codegen/low/allocation/live_range.h"

// Every placement-connected component is visited once. A queued value is
// marked immediately; the queue then doubles as the component member list.
static iree_status_t loom_low_allocation_active_capacity_component_starts(
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement, iree_arena_allocator_t* arena,
    uint32_t** out_starts) {
  *out_starts = NULL;
  if (!placement || placement->relation_count == 0) {
    return iree_ok_status();
  }
  uint32_t* starts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, liveness->value_count, sizeof(*starts), (void**)&starts));
  memset(starts, 0xFF, liveness->value_count * sizeof(*starts));
  loom_value_ordinal_t* members = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, liveness->value_count, sizeof(*members), (void**)&members));
  for (loom_value_ordinal_t root = 0; root < liveness->value_count; ++root) {
    if (starts[root] != UINT32_MAX) continue;
    starts[root] = 0;
    members[0] = root;
    iree_host_size_t member_count = 1;
    uint32_t first_start = UINT32_MAX;
    for (iree_host_size_t i = 0; i < member_count; ++i) {
      const loom_value_ordinal_t ordinal = members[i];
      const loom_liveness_interval_t* interval =
          loom_liveness_interval_for_value_ordinal(liveness, ordinal);
      if (interval &&
          loom_low_allocation_live_range_interval_is_allocatable(interval)) {
        first_start = iree_min(first_start, interval->start_point);
      }
      const loom_low_placement_relation_range_t result_range =
          loom_low_placement_relation_range_for_value_ordinal(placement,
                                                              ordinal);
      for (uint32_t j = 0; j < result_range.count; ++j) {
        const loom_value_ordinal_t source =
            placement->relations[result_range.start + j].source_ordinal;
        if (starts[source] == UINT32_MAX) {
          starts[source] = 0;
          members[member_count++] = source;
        }
      }
      const loom_low_placement_relation_range_t source_range =
          loom_low_placement_relation_range_for_source_value_ordinal(placement,
                                                                     ordinal);
      for (uint32_t j = 0; j < source_range.count; ++j) {
        const loom_value_ordinal_t result =
            placement
                ->relations[placement->relation_indices_by_source_ordinal
                                [source_range.start + j]]
                .result_ordinal;
        if (starts[result] == UINT32_MAX) {
          starts[result] = 0;
          members[member_count++] = result;
        }
      }
    }
    // Components without allocatable values still need a completed marker.
    if (first_start == UINT32_MAX) first_start = 0;
    for (iree_host_size_t i = 0; i < member_count; ++i) {
      starts[members[i]] = first_start;
    }
  }
  *out_starts = starts;
  return iree_ok_status();
}

static uint32_t loom_low_allocation_active_capacity_storage_end(
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    loom_value_ordinal_t ordinal, const loom_liveness_interval_t* interval) {
  const loom_low_allocation_assignment_t assignment = {
      .end_point =
          loom_low_allocation_live_range_interval_storage_end_point(interval),
      .unit_count = interval->unit_count,
      .unit_point_start =
          loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
              unit_liveness, liveness, ordinal),
  };
  return loom_low_allocation_live_range_assignment_max_unit_end_point(
      unit_liveness->end_points, unit_liveness->point_count, &assignment);
}

static iree_status_t loom_low_allocation_active_capacity_build(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_placement_table_t* placement, iree_arena_allocator_t* arena,
    loom_low_allocation_active_capacity_t* out_capacity) {
  uint32_t last_point = 0;
  for (iree_host_size_t i = 0; i < liveness->interval_count; ++i) {
    const loom_liveness_interval_t* interval = &liveness->intervals[i];
    if (loom_low_allocation_live_range_interval_is_allocatable(interval)) {
      last_point = iree_max(
          last_point,
          loom_low_allocation_live_range_interval_storage_end_point(interval));
    }
  }
  for (iree_host_size_t i = 0; i < unit_liveness->point_count; ++i) {
    last_point = iree_max(last_point, unit_liveness->end_points[i]);
  }
  if ((uint64_t)last_point + 1 > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "active assignment calendar exceeds host size");
  }
  const iree_host_size_t point_count = (iree_host_size_t)last_point + 1;
  uint32_t* component_starts = NULL;
  IREE_RETURN_IF_ERROR(loom_low_allocation_active_capacity_component_starts(
      liveness, placement, arena, &component_starts));
  int64_t* deltas = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, point_count, sizeof(*deltas), (void**)&deltas));
  memset(deltas, 0, point_count * sizeof(*deltas));
  uint64_t total_units = 0;
  for (loom_value_ordinal_t ordinal = 0; ordinal < liveness->value_count;
       ++ordinal) {
    const loom_liveness_interval_t* interval =
        loom_liveness_interval_for_value_ordinal(liveness, ordinal);
    if (!interval ||
        !loom_low_allocation_live_range_interval_is_allocatable(interval)) {
      continue;
    }
    const loom_low_reg_class_t* reg_class =
        &descriptor_set->reg_classes[interval->value_class.register_class_id];
    const uint32_t width =
        loom_low_reg_class_uses_explicit_physical_registers(reg_class)
            ? reg_class->physical_atomic_unit_count
            : 1;
    const uint64_t units = (uint64_t)interval->unit_count * width;
    if (units > INT64_MAX - total_units) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "active assignment unit bound exceeds 64 bits");
    }
    total_units += units;
    const uint32_t start =
        component_starts ? component_starts[ordinal] : interval->start_point;
    const uint32_t end = loom_low_allocation_active_capacity_storage_end(
        liveness, unit_liveness, ordinal, interval);
    deltas[start] += (int64_t)units;
    deltas[end] -= (int64_t)units;
  }
  int64_t active_units = 0;
  int64_t peak_units = 0;
  for (iree_host_size_t i = 0; i < point_count; ++i) {
    active_units += deltas[i];
    peak_units = iree_max(peak_units, active_units);
  }
  if ((uint64_t)peak_units > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "active assignment unit bound exceeds host size");
  }
  *out_capacity = (loom_low_allocation_active_capacity_t){
      .program_point_count = point_count,
      .unit_count = (iree_host_size_t)peak_units,
  };
  return iree_ok_status();
}

iree_status_t loom_low_allocation_active_capacity_calculate(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_placement_table_t* placement,
    iree_arena_block_pool_t* block_pool,
    loom_low_allocation_active_capacity_t* out_capacity) {
  *out_capacity = (loom_low_allocation_active_capacity_t){0};
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(block_pool, &scratch_arena);
  iree_status_t status = loom_low_allocation_active_capacity_build(
      descriptor_set, liveness, unit_liveness, placement, &scratch_arena,
      out_capacity);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
