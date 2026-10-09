// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/unit_liveness.h"

#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/ir/module.h"
#include "loom/target/registers.h"

bool loom_low_allocation_unit_liveness_storage_is_ignored(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    loom_value_id_t value_id, const loom_value_id_t* ignored_value_ids,
    uint16_t ignored_value_count) {
  const loom_low_placement_table_t* placement =
      unit_liveness->tied_storage_placement;
  for (uint16_t i = 0; i < ignored_value_count; ++i) {
    if (ignored_value_ids[i] == value_id) {
      return true;
    }
    if (placement != NULL) {
      const loom_value_ordinal_t* roots =
          placement->tied_storage_origins_by_value_ordinal;
      const loom_value_ordinal_t value_ordinal =
          loom_module_value_ordinal_scratch_lookup(placement->module, value_id);
      const loom_value_ordinal_t ignored_ordinal =
          loom_module_value_ordinal_scratch_lookup(placement->module,
                                                   ignored_value_ids[i]);
      if (roots[value_ordinal] == roots[ignored_ordinal]) {
        return true;
      }
    }
  }
  return false;
}

enum {
  // Cover the common adjacent sparse gap within one compact point span before
  // using a branchier binary search for longer skips.
  LOOM_LOW_ALLOCATION_CLOBBER_LINEAR_SEEK_COUNT = 4u,
};

static uint32_t loom_low_allocation_unit_liveness_clobber_point_bound(
    const uint32_t* points, uint32_t begin, uint32_t end, uint32_t point) {
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2u;
    if (points[middle] < point) {
      begin = middle + 1u;
    } else {
      end = middle;
    }
  }
  return begin;
}

static uint32_t loom_low_allocation_unit_liveness_clobber_point_seek(
    const uint32_t* points, uint32_t begin, uint32_t end, uint32_t point) {
  if (begin == end || points[begin] >= point) {
    return begin;
  }
  if (points[end - 1u] < point) {
    return end;
  }
  const uint32_t scan_count = iree_min(
      end - begin, (uint32_t)LOOM_LOW_ALLOCATION_CLOBBER_LINEAR_SEEK_COUNT);
  const uint32_t scan_end = begin + scan_count;
  while (begin < scan_end && points[begin] < point) {
    ++begin;
  }
  return begin < end && points[begin] < point
             ? loom_low_allocation_unit_liveness_clobber_point_bound(
                   points, begin, end, point)
             : begin;
}

static bool loom_low_allocation_unit_liveness_clobber_window_conflicts(
    const loom_low_allocation_clobber_index_t* clobbers, uint32_t range_end,
    uint32_t start_point, uint32_t end_point, uint32_t definition_point,
    uint32_t* cursor) {
  if (start_point >= end_point) {
    return false;
  }
  const uint32_t* points = clobbers->points;
  *cursor = loom_low_allocation_unit_liveness_clobber_point_seek(
      points, *cursor, range_end, start_point);
  while (*cursor < range_end && points[*cursor] < end_point) {
    if (points[*cursor] != definition_point ||
        clobbers->permitted_definitions.words == NULL ||
        !iree_bitmap_test(clobbers->permitted_definitions, *cursor)) {
      return true;
    }
    ++*cursor;
  }
  return false;
}

static bool loom_low_allocation_unit_liveness_location_is_clobbered(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_allocation_assignment_t* candidate, uint32_t location,
    uint32_t start_point, uint32_t end_point) {
  const loom_low_allocation_clobber_index_t* clobbers =
      &unit_liveness->clobbers;
  const loom_low_allocation_clobber_range_t range =
      loom_low_allocation_clobber_index_range(
          clobbers, candidate->descriptor_reg_class_id, location);
  if (range.start == range.end) {
    return false;
  }
  uint32_t cursor = range.start;
  if (candidate->liveness_segments.count == 0) {
    return loom_low_allocation_unit_liveness_clobber_window_conflicts(
        clobbers, range.end, start_point, end_point, start_point, &cursor);
  }
  for (uint32_t segment_index = 0;
       segment_index < candidate->liveness_segments.count; ++segment_index) {
    const loom_liveness_segment_t* segment =
        &unit_liveness->storage_segments
             .entries[candidate->liveness_segments.start + segment_index];
    const uint32_t window_start = iree_max(start_point, segment->start_point);
    const uint32_t window_end = iree_min(end_point, segment->end_point);
    if (window_start >= window_end) {
      continue;
    }
    if (loom_low_allocation_unit_liveness_clobber_window_conflicts(
            clobbers, range.end, window_start, window_end, start_point,
            &cursor)) {
      return true;
    }
  }
  return false;
}

bool loom_low_allocation_unit_liveness_clobber_conflicts(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* candidate) {
  if (unit_liveness->clobbers.points == NULL ||
      candidate->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    return false;
  }
  const bool is_explicit =
      loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, candidate);
  for (uint32_t unit = 0; unit < candidate->location_count; ++unit) {
    const uint32_t start_point =
        loom_low_allocation_live_range_assignment_unit_start_point(
            unit_liveness->start_points, unit_liveness->point_count, candidate,
            unit);
    const uint32_t end_point =
        loom_low_allocation_live_range_assignment_unit_end_point(
            unit_liveness->end_points, unit_liveness->point_count, candidate,
            unit);
    const uint16_t* atomic_units = NULL;
    uint16_t atomic_unit_count = 1;
    if (is_explicit) {
      uint32_t physical_register_id = 0;
      const bool resolved =
          loom_low_allocation_storage_assignment_unit_physical_register(
              descriptor_set, candidate, unit, &physical_register_id);
      IREE_ASSERT(resolved, "accepted assignment must name its physical units");
      atomic_units = loom_low_descriptor_set_physical_register_atomic_units(
          descriptor_set, physical_register_id, &atomic_unit_count);
    }
    for (uint16_t atomic_unit = 0; atomic_unit < atomic_unit_count;
         ++atomic_unit) {
      const uint32_t location = atomic_units != NULL
                                    ? atomic_units[atomic_unit]
                                    : candidate->location_base + unit;
      if (loom_low_allocation_unit_liveness_location_is_clobbered(
              unit_liveness, candidate, location, start_point, end_point)) {
        return true;
      }
    }
  }
  return false;
}

uint32_t loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    loom_value_ordinal_t value_ordinal) {
  IREE_ASSERT_ARGUMENT(unit_liveness);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_LT(value_ordinal, liveness->value_count);
  return unit_liveness->values[value_ordinal].unit_point_start;
}

const uint32_t*
loom_low_allocation_unit_liveness_start_points_for_value_ordinal(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    loom_value_ordinal_t value_ordinal) {
  const uint32_t start =
      loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
          unit_liveness, liveness, value_ordinal);
  return start == UINT32_MAX ? NULL : &unit_liveness->start_points[start];
}

loom_liveness_segment_range_t
loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    loom_value_ordinal_t value_ordinal) {
  IREE_ASSERT_ARGUMENT(unit_liveness);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_LT(value_ordinal, liveness->value_count);
  if (iree_bitmap_test(unit_liveness->values_with_incomplete_storage_segments,
                       value_ordinal)) {
    return unit_liveness->storage_segments.tied_sources != NULL
               ? unit_liveness->storage_segments.tied_sources[value_ordinal]
               : (loom_liveness_segment_range_t){0};
  }
  return loom_liveness_segment_range_for_value_ordinal(liveness, value_ordinal);
}

bool loom_low_allocation_unit_liveness_storage_component_live_at_point(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement,
    loom_value_ordinal_t value_ordinal, uint32_t unit_offset,
    uint32_t unit_count, uint32_t program_point) {
  const loom_liveness_interval_t* value_interval =
      loom_liveness_interval_for_value_ordinal(liveness, value_ordinal);
  if (value_interval == NULL || unit_count == 0) {
    return false;
  }
  IREE_ASSERT_LE(unit_offset, value_interval->unit_count);
  IREE_ASSERT_LE(unit_count, value_interval->unit_count - unit_offset);
  const loom_value_ordinal_t storage_ordinal =
      placement->tied_storage_origins_by_value_ordinal == NULL
          ? value_ordinal
          : placement->tied_storage_origins_by_value_ordinal[value_ordinal];
  const loom_liveness_segment_range_t segments =
      loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
          unit_liveness, liveness, storage_ordinal);
  const bool storage_live_at_point =
      segments.count == 0
          ? unit_liveness->values[storage_ordinal].acquisition_start_point <=
                program_point
          : loom_liveness_segment_range_contains(
                unit_liveness->storage_segments.entries, segments,
                program_point);
  if (!storage_live_at_point) {
    return false;
  }

  const uint32_t point_start =
      loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
          unit_liveness, liveness, storage_ordinal);
  IREE_ASSERT_NE(point_start, UINT32_MAX);
  for (uint32_t i = 0; i < unit_count; ++i) {
    if (unit_liveness->end_points[point_start + unit_offset + i] >
        program_point) {
      return true;
    }
  }
  return false;
}
