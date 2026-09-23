// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/liveness_pressure.h"

#include <string.h>

#include "loom/analysis/liveness_events.h"
#include "loom/util/adaptive_sort.h"

// Differences use unsigned modular arithmetic. A complete point prefix is the
// sum of at most UINT32_MAX live values with uint32_t widths, so its exact
// total fits uint64_t even when individual endpoint differences wrap.
typedef struct loom_liveness_pressure_count_t {
  // Register units contributed by the live values or endpoint differences.
  uint64_t units;
  // Number of live values or their endpoint differences.
  uint64_t values;
} loom_liveness_pressure_count_t;

typedef struct loom_liveness_pressure_class_t {
  // Class identity and first maximal pressure observed for it.
  loom_liveness_pressure_summary_t summary;
  // Earliest nonempty segment that introduces this class.
  struct {
    // Segment start point.
    uint32_t point;
    // Lowest SSA identity starting at that point.
    loom_value_id_t value_id;
  } first;
  // Current live totals during a point-ordered sweep.
  loom_liveness_pressure_count_t live;
} loom_liveness_pressure_class_t;

typedef struct loom_liveness_pressure_classes_t {
  // Class records in discovery order until final summary ordering.
  loom_liveness_pressure_class_t* entries;
  // Number of initialized class records.
  iree_host_size_t count;
  // Allocated class record capacity.
  iree_host_size_t capacity;
  // Class index for each value with nonempty segments; other entries are
  // unused.
  uint32_t* value_indices;
  // Greatest segment endpoint, retained while classifying live values.
  uint32_t maximum_point;
} loom_liveness_pressure_classes_t;

static iree_status_t loom_liveness_pressure_classes_initialize(
    const loom_liveness_analysis_t* analysis,
    iree_arena_allocator_t* scratch_arena,
    loom_liveness_pressure_classes_t* out_classes) {
  *out_classes = (loom_liveness_pressure_classes_t){0};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, analysis->value_count, sizeof(*out_classes->value_indices),
      (void**)&out_classes->value_indices));
  for (loom_value_ordinal_t ordinal = 0; ordinal < analysis->value_count;
       ++ordinal) {
    const loom_liveness_segment_range_t range =
        analysis->value_segment_ranges[ordinal];
    if (range.count == 0) {
      continue;
    }
    const loom_liveness_interval_t* interval =
        &analysis->intervals[analysis->value_interval_indices[ordinal]];
    const uint32_t first_point = analysis->segments[range.start].start_point;
    // Live-out ranges include the block exit point, so their one-past end can
    // extend beyond the final block's own point extent.
    out_classes->maximum_point =
        iree_max(out_classes->maximum_point,
                 analysis->segments[range.start + range.count - 1].end_point);
    iree_host_size_t class_index = 0;
    while (class_index < out_classes->count &&
           !loom_liveness_value_class_equal(
               out_classes->entries[class_index].summary.value_class,
               interval->value_class)) {
      ++class_index;
    }
    if (class_index == out_classes->count) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          scratch_arena, out_classes->count, out_classes->count + 1,
          sizeof(*out_classes->entries), &out_classes->capacity,
          (void**)&out_classes->entries));
      out_classes->entries[out_classes->count++] =
          (loom_liveness_pressure_class_t){
              .summary = {.value_class = interval->value_class},
              .first = {.point = first_point, .value_id = interval->value_id},
          };
    } else {
      loom_liveness_pressure_class_t* entry =
          &out_classes->entries[class_index];
      if (first_point < entry->first.point ||
          (first_point == entry->first.point &&
           interval->value_id < entry->first.value_id)) {
        entry->first.point = first_point;
        entry->first.value_id = interval->value_id;
      }
    }
    out_classes->value_indices[ordinal] = (uint32_t)class_index;
  }
  return iree_ok_status();
}

static const loom_block_t* loom_liveness_pressure_block_at_point(
    const loom_liveness_analysis_t* analysis, uint32_t point,
    iree_host_size_t* block_index) {
  while (*block_index + 1 < analysis->block_count &&
         point > analysis->blocks[*block_index].end_point) {
    ++*block_index;
  }
  const loom_liveness_block_info_t* block = &analysis->blocks[*block_index];
  return block->start_point <= point && point <= block->end_point ? block->block
                                                                  : NULL;
}

static iree_status_t loom_liveness_pressure_record_peak(
    loom_liveness_pressure_class_t* entry, uint32_t point,
    const loom_block_t* block) {
  if (entry->live.units > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "liveness pressure sweep exceeds uint32_t");
  }
  loom_liveness_pressure_summary_t* summary = &entry->summary;
  if (entry->live.values != 0 &&
      (entry->live.units > summary->peak_live_units ||
       (entry->live.units == summary->peak_live_units &&
        entry->live.values > summary->peak_live_values))) {
    summary->peak_live_units = (uint32_t)entry->live.units;
    summary->peak_live_values = (uint32_t)entry->live.values;
    summary->peak_block = block;
    summary->peak_point = point;
  }
  return iree_ok_status();
}

static iree_status_t loom_liveness_pressure_sweep_dense(
    const loom_liveness_analysis_t* analysis, iree_host_size_t point_count,
    iree_arena_allocator_t* scratch_arena,
    loom_liveness_pressure_classes_t* classes) {
  const iree_host_size_t delta_count = point_count * classes->count;
  loom_liveness_pressure_count_t* deltas = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, delta_count, sizeof(*deltas), (void**)&deltas));
  memset(deltas, 0, delta_count * sizeof(*deltas));
  for (loom_value_ordinal_t ordinal = 0; ordinal < analysis->value_count;
       ++ordinal) {
    const loom_liveness_segment_range_t range =
        analysis->value_segment_ranges[ordinal];
    if (range.count == 0) {
      continue;
    }
    const uint32_t class_index = classes->value_indices[ordinal];
    const uint32_t units =
        analysis->intervals[analysis->value_interval_indices[ordinal]]
            .unit_count;
    for (uint32_t i = 0; i < range.count; ++i) {
      const loom_liveness_segment_t* segment =
          &analysis->segments[range.start + i];
      loom_liveness_pressure_count_t* start =
          &deltas[(iree_host_size_t)segment->start_point * classes->count +
                  class_index];
      loom_liveness_pressure_count_t* end =
          &deltas[(iree_host_size_t)segment->end_point * classes->count +
                  class_index];
      start->units += units;
      ++start->values;
      end->units -= units;
      --end->values;
    }
  }
  iree_host_size_t block_index = 0;
  for (iree_host_size_t point = 0; point < point_count; ++point) {
    const loom_block_t* block = loom_liveness_pressure_block_at_point(
        analysis, (uint32_t)point, &block_index);
    for (iree_host_size_t i = 0; i < classes->count; ++i) {
      loom_liveness_pressure_class_t* entry = &classes->entries[i];
      entry->live.units += deltas[point * classes->count + i].units;
      entry->live.values += deltas[point * classes->count + i].values;
      IREE_RETURN_IF_ERROR(
          loom_liveness_pressure_record_peak(entry, (uint32_t)point, block));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_liveness_pressure_sweep_sparse(
    const loom_liveness_analysis_t* analysis, iree_host_size_t event_count,
    uint32_t maximum_point, iree_arena_allocator_t* scratch_arena,
    loom_liveness_pressure_classes_t* classes) {
  loom_liveness_event_t* events = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, event_count, sizeof(*events), (void**)&events));
  iree_host_size_t event_index = 0;
  for (loom_value_ordinal_t ordinal = 0; ordinal < analysis->value_count;
       ++ordinal) {
    const loom_liveness_segment_range_t range =
        analysis->value_segment_ranges[ordinal];
    for (uint32_t i = 0; i < range.count; ++i) {
      const loom_liveness_segment_t* segment =
          &analysis->segments[range.start + i];
      events[event_index++] = (loom_liveness_event_t){
          .point = segment->start_point,
          .value_id = analysis->value_ids[ordinal],
          .value_ordinal = ordinal,
          .value_delta = 1,
      };
      events[event_index++] = (loom_liveness_event_t){
          .point = segment->end_point,
          .value_id = analysis->value_ids[ordinal],
          .value_ordinal = ordinal,
          .value_delta = -1,
      };
    }
  }
  IREE_RETURN_IF_ERROR(loom_liveness_events_sort(events, event_count,
                                                 maximum_point, scratch_arena));
  iree_host_size_t block_index = 0;
  for (iree_host_size_t i = 0; i < event_count;) {
    const uint32_t point = events[i].point;
    do {
      const loom_value_ordinal_t ordinal = events[i].value_ordinal;
      loom_liveness_pressure_count_t* live =
          &classes->entries[classes->value_indices[ordinal]].live;
      const uint32_t units =
          analysis->intervals[analysis->value_interval_indices[ordinal]]
              .unit_count;
      if (events[i].value_delta > 0) {
        live->units += units;
        ++live->values;
      } else {
        live->units -= units;
        --live->values;
      }
      ++i;
    } while (i < event_count && events[i].point == point);
    const loom_block_t* block =
        loom_liveness_pressure_block_at_point(analysis, point, &block_index);
    for (iree_host_size_t j = 0; j < classes->count; ++j) {
      IREE_RETURN_IF_ERROR(loom_liveness_pressure_record_peak(
          &classes->entries[j], point, block));
    }
  }
  return iree_ok_status();
}

static bool loom_liveness_pressure_class_less(
    const loom_liveness_pressure_class_t* lhs,
    const loom_liveness_pressure_class_t* rhs) {
  return lhs->first.point < rhs->first.point ||
         (lhs->first.point == rhs->first.point &&
          lhs->first.value_id < rhs->first.value_id);
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_liveness_pressure_classes_sort,
                          loom_liveness_pressure_class_t,
                          loom_liveness_pressure_class_less)

iree_status_t loom_liveness_compute_segment_pressure(
    const loom_liveness_analysis_t* analysis,
    iree_arena_allocator_t* scratch_arena, iree_arena_allocator_t* result_arena,
    const loom_liveness_pressure_summary_t** out_summaries,
    iree_host_size_t* out_summary_count) {
  *out_summaries = NULL;
  *out_summary_count = 0;
  if (analysis->segment_count == 0) {
    return iree_ok_status();
  }
  if (analysis->segment_count > IREE_HOST_SIZE_MAX / 2) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "liveness pressure event count exceeds host size");
  }
  loom_liveness_pressure_classes_t classes;
  IREE_RETURN_IF_ERROR(loom_liveness_pressure_classes_initialize(
      analysis, scratch_arena, &classes));
  const uint32_t maximum_point = classes.maximum_point;
  const uint64_t point_count = (uint64_t)maximum_point + 1;
  const iree_host_size_t event_count = analysis->segment_count * 2;
  // Both entries are 16 bytes. Admit a dense table only when its storage and
  // point/class sweep are bounded by the endpoint representation.
  static_assert(
      sizeof(loom_liveness_pressure_count_t) == sizeof(loom_liveness_event_t),
      "pressure admission compares equal-sized entries");
  if (point_count * classes.count <= event_count) {
    IREE_RETURN_IF_ERROR(loom_liveness_pressure_sweep_dense(
        analysis, (iree_host_size_t)point_count, scratch_arena, &classes));
  } else {
    IREE_RETURN_IF_ERROR(loom_liveness_pressure_sweep_sparse(
        analysis, event_count, maximum_point, scratch_arena, &classes));
  }
  loom_liveness_pressure_classes_sort(classes.entries, classes.count);
  loom_liveness_pressure_summary_t* summaries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      result_arena, classes.count, sizeof(*summaries), (void**)&summaries));
  for (iree_host_size_t i = 0; i < classes.count; ++i) {
    summaries[i] = classes.entries[i].summary;
  }
  *out_summaries = summaries;
  *out_summary_count = classes.count;
  return iree_ok_status();
}
