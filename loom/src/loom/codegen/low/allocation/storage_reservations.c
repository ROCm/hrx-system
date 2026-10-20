// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/storage_reservations.h"

#include <string.h>

// A contribution starts on one program-point bucket and is relinked into its
// source's merged reservation during the ascending point sweep.
typedef struct loom_low_allocation_storage_segment_contribution_t {
  // Source whose physical location must remain available for this lifetime.
  loom_value_ordinal_t source_ordinal;
  // Next contribution in the point bucket, then next merged source segment.
  uint32_t next;
  // Half-open physical reservation.
  loom_liveness_segment_t segment;
} loom_low_allocation_storage_segment_contribution_t;

typedef struct loom_low_allocation_storage_segment_chain_t {
  // First merged contribution, or UINT32_MAX for an empty chain.
  uint32_t head;
  // Last merged contribution, or UINT32_MAX for an empty chain.
  uint32_t tail;
  // Number of merged segments.
  uint32_t count;
  // True when this storage origin has tied or decomposed-read obligations.
  bool needs_reservation;
} loom_low_allocation_storage_segment_chain_t;

static bool loom_low_allocation_unit_liveness_can_refine_tie(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_placement_relation_t* relation) {
  return relation->cause == LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT &&
         unit_liveness->values[relation->source_ordinal].unit_point_start !=
             UINT32_MAX &&
         unit_liveness->values[relation->result_ordinal].unit_point_start !=
             UINT32_MAX;
}

// Backward decomposed reads have explicit physical contributions. Their
// semantic segments remain useful even when those reads made SSA-only storage
// incomplete. Other extensions retain their conservative per-unit hull.
static loom_liveness_segment_range_t
loom_low_allocation_storage_reservations_semantic_range(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    loom_value_ordinal_t value_ordinal) {
  const loom_liveness_interval_t* interval =
      loom_liveness_interval_for_value_ordinal(liveness, value_ordinal);
  if (unit_liveness->storage_segments.demand_count != 0 &&
      unit_liveness->values[value_ordinal].acquisition_start_point <
          interval->start_point &&
      !iree_bitmap_test(
          unit_liveness->values_with_independent_storage_extensions,
          value_ordinal)) {
    return loom_liveness_segment_range_for_value_ordinal(liveness,
                                                         value_ordinal);
  }
  return loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
      unit_liveness, liveness, value_ordinal);
}

static void loom_low_allocation_unit_liveness_contribute_segments(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    loom_value_ordinal_t source_ordinal, loom_value_ordinal_t value_ordinal,
    uint32_t* point_heads,
    loom_low_allocation_storage_segment_contribution_t* contributions,
    uint32_t* contribution_count) {
  const loom_liveness_segment_range_t range =
      loom_low_allocation_storage_reservations_semantic_range(
          unit_liveness, liveness, value_ordinal);
  loom_liveness_segment_t contiguous = {0};
  const loom_liveness_segment_t* segments = NULL;
  uint32_t count = range.count;
  if (count != 0) {
    segments = &unit_liveness->storage_segments.entries[range.start];
  } else {
    // A result with early-clobber/edge storage or no semantic uses still
    // reserves its concrete writes. Preserve its initial per-unit hull.
    const loom_liveness_interval_t* interval =
        loom_liveness_interval_for_value_ordinal(liveness, value_ordinal);
    const uint32_t unit_start =
        unit_liveness->values[value_ordinal].unit_point_start;
    contiguous.start_point =
        unit_liveness->values[value_ordinal].acquisition_start_point;
    for (uint32_t i = 0; i < interval->unit_count; ++i) {
      contiguous.end_point = iree_max(
          contiguous.end_point, unit_liveness->end_points[unit_start + i]);
    }
    segments = &contiguous;
    count = 1;
  }
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t index = (*contribution_count)++;
    const uint32_t point = segments[i].start_point;
    contributions[index] = (loom_low_allocation_storage_segment_contribution_t){
        .source_ordinal = source_ordinal,
        .next = point_heads[point],
        .segment = segments[i],
    };
    point_heads[point] = index;
  }
}

static loom_value_ordinal_t loom_low_allocation_unit_liveness_storage_origin(
    const loom_low_placement_table_t* placement,
    loom_value_ordinal_t value_ordinal) {
  return placement->tied_storage_origins_by_value_ordinal != NULL
             ? placement->tied_storage_origins_by_value_ordinal[value_ordinal]
             : value_ordinal;
}

// Semantic lifetimes and retained physical reads contribute to their storage
// component. Program-point buckets merge every component in one ordered sweep
// without sorting, repeated transitive unions, or chain-length growth.
static iree_status_t loom_low_allocation_unit_liveness_build_storage_segments(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement,
    iree_arena_allocator_t* scratch_arena, iree_arena_allocator_t* result_arena,
    iree_arena_allocator_t* decision_arena) {
  loom_low_allocation_storage_segment_chain_t* chains = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, liveness->value_count, sizeof(*chains), (void**)&chains));
  for (iree_host_size_t i = 0; i < liveness->value_count; ++i) {
    chains[i] = (loom_low_allocation_storage_segment_chain_t){
        .head = UINT32_MAX, .tail = UINT32_MAX};
  }
  for (iree_host_size_t i = 0; i < placement->relation_count; ++i) {
    const loom_low_placement_relation_t* relation = &placement->relations[i];
    if (!loom_low_allocation_unit_liveness_can_refine_tie(unit_liveness,
                                                          relation)) {
      continue;
    }
    const loom_value_ordinal_t origin =
        placement
            ->tied_storage_origins_by_value_ordinal[relation->source_ordinal];
    chains[origin].needs_reservation = true;
  }

  for (iree_host_size_t i = 0; i < unit_liveness->storage_segments.demand_count;
       ++i) {
    const loom_value_ordinal_t origin =
        loom_low_allocation_unit_liveness_storage_origin(
            placement,
            unit_liveness->storage_segments.demands[i].value_ordinal);
    chains[origin].needs_reservation = true;
  }

  uint64_t capacity = unit_liveness->storage_segments.demand_count;
  for (loom_value_ordinal_t i = 0; i < liveness->value_count; ++i) {
    const loom_value_ordinal_t origin =
        loom_low_allocation_unit_liveness_storage_origin(placement, i);
    if (!chains[origin].needs_reservation) {
      continue;
    }
    const loom_liveness_segment_range_t range =
        loom_low_allocation_storage_reservations_semantic_range(unit_liveness,
                                                                liveness, i);
    capacity += iree_max(range.count, 1u);
  }
  if (capacity > UINT32_MAX - liveness->segment_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low storage segment count exceeds u32 range");
  }
  loom_low_allocation_storage_segment_contribution_t* contributions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, (iree_host_size_t)capacity, sizeof(*contributions),
      (void**)&contributions));
  const uint64_t point_capacity =
      (uint64_t)liveness->blocks[liveness->block_count - 1].end_point + 1;
  if (point_capacity > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low storage point count exceeds host size");
  }
  const iree_host_size_t point_count = (iree_host_size_t)point_capacity;
  uint32_t* point_heads = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, point_count, sizeof(*point_heads), (void**)&point_heads));
  memset(point_heads, 0xFF, point_count * sizeof(*point_heads));
  uint32_t contribution_count = 0;
  for (loom_value_ordinal_t i = 0; i < liveness->value_count; ++i) {
    const loom_value_ordinal_t origin =
        loom_low_allocation_unit_liveness_storage_origin(placement, i);
    if (!chains[origin].needs_reservation) {
      continue;
    }
    loom_low_allocation_unit_liveness_contribute_segments(
        unit_liveness, liveness, origin, i, point_heads, contributions,
        &contribution_count);
  }
  for (iree_host_size_t i = 0; i < unit_liveness->storage_segments.demand_count;
       ++i) {
    const loom_low_allocation_storage_demand_t* demand =
        &unit_liveness->storage_segments.demands[i];
    const uint32_t point = demand->segment.start_point;
    const uint32_t index = contribution_count++;
    contributions[index] = (loom_low_allocation_storage_segment_contribution_t){
        .source_ordinal = loom_low_allocation_unit_liveness_storage_origin(
            placement, demand->value_ordinal),
        .next = point_heads[point],
        .segment = demand->segment,
    };
    point_heads[point] = index;
  }
  IREE_ASSERT_EQ(contribution_count, (uint32_t)capacity);
  uint32_t segment_count = (uint32_t)liveness->segment_count;
  for (iree_host_size_t point = 0; point < point_count; ++point) {
    uint32_t index = point_heads[point];
    while (index != UINT32_MAX) {
      loom_low_allocation_storage_segment_contribution_t* contribution =
          &contributions[index];
      const uint32_t next = contribution->next;
      loom_low_allocation_storage_segment_chain_t* chain =
          &chains[contribution->source_ordinal];
      if (chain->tail != UINT32_MAX &&
          contribution->segment.start_point <=
              contributions[chain->tail].segment.end_point) {
        contributions[chain->tail].segment.end_point =
            iree_max(contributions[chain->tail].segment.end_point,
                     contribution->segment.end_point);
      } else {
        if (chain->tail == UINT32_MAX) {
          chain->head = index;
        } else {
          contributions[chain->tail].next = index;
        }
        contribution->next = UINT32_MAX;
        chain->tail = index;
        ++chain->count;
        ++segment_count;
      }
      index = next;
    }
  }
  loom_liveness_segment_t* segments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      result_arena, segment_count, sizeof(*segments), (void**)&segments));
  memcpy(segments, unit_liveness->storage_segments.entries,
         liveness->segment_count * sizeof(*segments));
  loom_liveness_segment_range_t* ranges = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      decision_arena, liveness->value_count, sizeof(*ranges), (void**)&ranges));
  uint32_t segment_index = (uint32_t)liveness->segment_count;
  for (loom_value_ordinal_t i = 0; i < liveness->value_count; ++i) {
    ranges[i] = (loom_liveness_segment_range_t){0};
    if (!chains[i].needs_reservation) {
      continue;
    }
    ranges[i] = (loom_liveness_segment_range_t){.start = segment_index,
                                                .count = chains[i].count};
    for (uint32_t index = chains[i].head; index != UINT32_MAX;
         index = contributions[index].next) {
      segments[segment_index++] = contributions[index].segment;
    }
  }
  for (iree_host_size_t i = 0; i < placement->relation_count; ++i) {
    const loom_low_placement_relation_t* relation = &placement->relations[i];
    if (!loom_low_allocation_unit_liveness_can_refine_tie(unit_liveness,
                                                          relation)) {
      continue;
    }
    const loom_value_ordinal_t origin =
        placement
            ->tied_storage_origins_by_value_ordinal[relation->source_ordinal];
    ranges[relation->source_ordinal] = ranges[origin];
  }
  IREE_ASSERT_EQ(segment_index, segment_count);
  unit_liveness->storage_segments.entries = segments;
  unit_liveness->storage_segments.refined_ranges = ranges;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_unit_liveness_refine_storage_segments(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement,
    iree_arena_allocator_t* result_arena,
    iree_arena_allocator_t* decision_arena) {
  if (liveness->block_count < 2) {
    return iree_ok_status();
  }
  bool has_cross_block_tie = false;
  for (iree_host_size_t i = 0; i < placement->relation_count; ++i) {
    const loom_low_placement_relation_t* relation = &placement->relations[i];
    if (!loom_low_allocation_unit_liveness_can_refine_tie(unit_liveness,
                                                          relation)) {
      continue;
    }
    const loom_liveness_interval_t* source =
        &liveness->intervals
             [liveness->value_interval_indices[relation->source_ordinal]];
    const loom_liveness_interval_t* result =
        &liveness->intervals
             [liveness->value_interval_indices[relation->result_ordinal]];
    if (source->block_count > 1 || result->block_count > 1) {
      has_cross_block_tie = true;
      break;
    }
  }
  if (!has_cross_block_tie &&
      unit_liveness->storage_segments.demand_count == 0) {
    return iree_ok_status();
  }
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(result_arena->block_pool, &scratch_arena);
  iree_status_t status =
      loom_low_allocation_unit_liveness_build_storage_segments(
          unit_liveness, liveness, placement, &scratch_arena, result_arena,
          decision_arena);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

// Acquisition starts flow from users to sources in the retained storage order.
// Only ancestors needed earlier move; ordinary forward aliases keep their own
// starts. The origin alone retains the component's complete physical lifetime.
static void loom_low_allocation_unit_liveness_retain_tied_component_lifetimes(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_placement_table_t* placement) {
  for (loom_value_ordinal_t cursor = 0;
       cursor < placement->storage_value_order_count; ++cursor) {
    const loom_value_ordinal_t result_ordinal =
        placement->storage_value_order[cursor];
    const loom_low_placement_relation_range_t range =
        placement->ranges_by_result_ordinal[result_ordinal];
    for (uint32_t i = 0; i < range.count; ++i) {
      const loom_low_placement_relation_t* relation =
          &placement->relations[range.start + i];
      if (!loom_low_allocation_unit_liveness_can_refine_tie(unit_liveness,
                                                            relation)) {
        continue;
      }
      loom_low_allocation_unit_liveness_value_t* source =
          &unit_liveness->values[relation->source_ordinal];
      const loom_low_allocation_unit_liveness_value_t* result =
          &unit_liveness->values[result_ordinal];
      source->acquisition_start_point = iree_min(
          source->acquisition_start_point, result->acquisition_start_point);
      const loom_value_ordinal_t origin_ordinal =
          placement
              ->tied_storage_origins_by_value_ordinal[relation->source_ordinal];
      const uint32_t origin_unit_point_start =
          unit_liveness->values[origin_ordinal].unit_point_start;
      for (uint32_t unit_index = 0; unit_index < relation->unit_count;
           ++unit_index) {
        const uint32_t origin_point_index =
            origin_unit_point_start + relation->source_unit_offset + unit_index;
        const uint32_t result_point_index = result->unit_point_start +
                                            relation->result_unit_offset +
                                            unit_index;
        unit_liveness->start_points[origin_point_index] =
            iree_min(unit_liveness->start_points[origin_point_index],
                     unit_liveness->start_points[result_point_index]);
        unit_liveness->end_points[origin_point_index] =
            iree_max(unit_liveness->end_points[origin_point_index],
                     unit_liveness->end_points[result_point_index]);
      }
    }
  }
}

iree_status_t loom_low_allocation_storage_reservations_build(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement,
    iree_arena_allocator_t* result_arena,
    iree_arena_allocator_t* decision_arena) {
  loom_low_allocation_unit_liveness_retain_tied_component_lifetimes(
      unit_liveness, placement);
  return loom_low_allocation_unit_liveness_refine_storage_segments(
      unit_liveness, liveness, placement, result_arena, decision_arena);
}
