// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/scalar_packing.h"

#include <string.h>

#include "loom/codegen/low/allocation/live_range.h"

iree_status_t loom_low_allocation_scalar_packing_build(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_interval_order_t* order,
    iree_arena_allocator_t* arena,
    loom_low_allocation_scalar_packing_t* out_packing) {
  *out_packing = (loom_low_allocation_scalar_packing_t){0};
  bool has_aggregate = false;
  for (iree_host_size_t i = 0; i < order->interval_count; ++i) {
    const loom_liveness_interval_t* interval = order->intervals[i];
    const loom_low_reg_class_t* reg_class =
        &descriptor_set->reg_classes[interval->value_class.register_class_id];
    if (interval->unit_count > 1 &&
        !loom_low_reg_class_uses_explicit_physical_registers(reg_class)) {
      has_aggregate = true;
      break;
    }
  }
  if (!has_aggregate) return iree_ok_status();

  const iree_host_size_t class_count = descriptor_set->reg_class_count;
  uint32_t* frontiers = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, class_count, sizeof(*frontiers), (void**)&frontiers));
  memset(frontiers, 0, class_count * sizeof(*frontiers));
  for (iree_host_size_t i = 0; i < liveness->pressure_summary_count; ++i) {
    const loom_liveness_pressure_summary_t* summary =
        &liveness->pressure_summaries[i];
    if (summary->value_class.type_kind != LOOM_TYPE_REGISTER) continue;
    const uint16_t class_id = summary->value_class.register_class_id;
    if (!loom_low_reg_class_uses_explicit_physical_registers(
            &descriptor_set->reg_classes[class_id])) {
      frontiers[class_id] = summary->peak_live_units;
    }
  }
  uint32_t* points = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, class_count, sizeof(*points), (void**)&points));
  memset(points, 0, class_count * sizeof(*points));
  const iree_host_size_t word_count =
      iree_bitmap_calculate_words(liveness->interval_count);
  iree_bitmap_t overlaps = {.bit_count = liveness->interval_count};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, word_count, sizeof(*overlaps.words), (void**)&overlaps.words));
  memset(overlaps.words, 0, word_count * sizeof(*overlaps.words));

  // Earlier-starting aggregates overlap iff their latest end crosses the
  // scalar's start. Dead definitions still reserve their definition point.
  for (iree_host_size_t i = 0; i < order->interval_count; ++i) {
    const loom_liveness_interval_t* interval = order->intervals[i];
    const uint16_t class_id = interval->value_class.register_class_id;
    if (!frontiers[class_id]) continue;
    if (interval->unit_count > 1) {
      points[class_id] = iree_max(
          points[class_id],
          loom_low_allocation_live_range_interval_storage_end_point(interval));
    } else if (points[class_id] > interval->start_point) {
      iree_bitmap_set(overlaps, interval - liveness->intervals);
    }
  }
  // Later-starting aggregates overlap iff the nearest start precedes the
  // scalar's end. Together the sweeps cover both containment directions.
  memset(points, 0xFF, class_count * sizeof(*points));
  for (iree_host_size_t i = order->interval_count; i > 0; --i) {
    const loom_liveness_interval_t* interval = order->intervals[i - 1];
    const uint16_t class_id = interval->value_class.register_class_id;
    if (!frontiers[class_id]) continue;
    if (interval->unit_count > 1) {
      points[class_id] = interval->start_point;
    } else if (points[class_id] <
               loom_low_allocation_live_range_interval_storage_end_point(
                   interval)) {
      iree_bitmap_set(overlaps, interval - liveness->intervals);
    }
  }
  *out_packing = (loom_low_allocation_scalar_packing_t){
      .overlapping_intervals = overlaps,
      .frontiers_by_reg_class = frontiers,
  };
  return iree_ok_status();
}
