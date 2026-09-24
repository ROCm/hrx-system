// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/destructive_reuse.h"

#include <string.h>

// Structural SSA relations form a DAG even when CFG block order differs from
// definition order. Edge transfers are not identity relations in this graph.
static bool loom_low_allocation_reuse_relation(
    const loom_low_placement_relation_t* relation) {
  return loom_low_placement_relation_can_alias(relation) &&
         relation->cause >= LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT &&
         relation->cause <= LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT;
}

static iree_status_t loom_low_allocation_refine_destructive_reuse_build(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    loom_low_placement_table_t* placement, iree_arena_allocator_t* scratch) {
  uint32_t* pending_users = NULL;
  loom_value_ordinal_t* order = NULL;
  uint32_t* preservation_ends = NULL;
  uint32_t* first_writes = NULL;
  uint64_t* tied_preservation_words = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, placement->value_count, sizeof(*pending_users),
      (void**)&pending_users));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, placement->value_count, sizeof(*order), (void**)&order));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, unit_liveness->point_count, sizeof(*preservation_ends),
      (void**)&preservation_ends));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(scratch, unit_liveness->point_count,
                                sizeof(*first_writes), (void**)&first_writes));
  const iree_host_size_t tied_preservation_word_count =
      iree_bitmap_calculate_words(unit_liveness->point_count);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, tied_preservation_word_count, sizeof(*tied_preservation_words),
      (void**)&tied_preservation_words));
  iree_bitmap_t tied_preservation_units = {
      .bit_count = unit_liveness->point_count,
      .words = tied_preservation_words,
  };
  memset(pending_users, 0, placement->value_count * sizeof(*pending_users));
  memcpy(preservation_ends, unit_liveness->end_points,
         unit_liveness->point_count * sizeof(*preservation_ends));
  memset(first_writes, 0xFF,
         unit_liveness->point_count * sizeof(*first_writes));
  iree_bitmap_reset_all(tied_preservation_units);
  const uint32_t* unit_starts = unit_liveness->point_starts_by_value_ordinal;
  for (iree_host_size_t i = 0; i < placement->relation_count; ++i) {
    const loom_low_placement_relation_t* relation = &placement->relations[i];
    if (!loom_low_allocation_reuse_relation(relation)) {
      continue;
    }
    ++pending_users[relation->source_ordinal];
    const uint32_t source_start =
        unit_starts[relation->source_ordinal] + relation->source_unit_offset;
    const uint32_t result_start =
        unit_starts[relation->result_ordinal] + relation->result_unit_offset;
    if (relation->cause == LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT) {
      // A required identity makes the preservation bound describe the whole
      // storage family rather than either value's semantic SSA segments.
      iree_bitmap_set_span(tied_preservation_units, source_start,
                           relation->unit_count);
      iree_bitmap_set_span(tied_preservation_units, result_start,
                           relation->unit_count);
    }
    if (iree_any_bit_set(relation->flags,
                         LOOM_LOW_PLACEMENT_RELATION_FLAG_WRITES_STORAGE)) {
      for (uint32_t unit = 0; unit < relation->unit_count; ++unit) {
        first_writes[source_start + unit] =
            iree_min(first_writes[source_start + unit], relation->write_point);
      }
    }
  }

  // Visit users before sources. Required equalities retain the latest storage
  // observation through each tied-result chain, independently of SSA lifetime.
  loom_value_ordinal_t order_count = 0;
  for (loom_value_ordinal_t i = 0; i < placement->value_count; ++i) {
    if (pending_users[i] == 0) {
      order[order_count++] = i;
    }
  }
  for (loom_value_ordinal_t cursor = 0; cursor < order_count; ++cursor) {
    const loom_low_placement_relation_range_t range =
        placement->ranges_by_result_ordinal[order[cursor]];
    for (uint32_t i = 0; i < range.count; ++i) {
      const loom_low_placement_relation_t* relation =
          &placement->relations[range.start + i];
      if (!loom_low_allocation_reuse_relation(relation)) {
        continue;
      }
      if (relation->cause == LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT) {
        const uint32_t source_start = unit_starts[relation->source_ordinal] +
                                      relation->source_unit_offset;
        const uint32_t result_start = unit_starts[relation->result_ordinal] +
                                      relation->result_unit_offset;
        for (uint32_t unit = 0; unit < relation->unit_count; ++unit) {
          preservation_ends[source_start + unit] =
              iree_max(preservation_ends[source_start + unit],
                       preservation_ends[result_start + unit]);
        }
      }
      if (--pending_users[relation->source_ordinal] == 0) {
        order[order_count++] = relation->source_ordinal;
      }
    }
  }
  IREE_ASSERT_EQ(order_count, placement->value_count,
                 "structural SSA storage relations must be acyclic");

  // Identity siblings also share the family's storage observation bound.
  for (loom_value_ordinal_t cursor = order_count; cursor > 0; --cursor) {
    const loom_low_placement_relation_range_t range =
        placement->ranges_by_result_ordinal[order[cursor - 1]];
    for (uint32_t i = 0; i < range.count; ++i) {
      const loom_low_placement_relation_t* relation =
          &placement->relations[range.start + i];
      if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT) {
        continue;
      }
      const uint32_t source_start =
          unit_starts[relation->source_ordinal] + relation->source_unit_offset;
      const uint32_t result_start =
          unit_starts[relation->result_ordinal] + relation->result_unit_offset;
      for (uint32_t unit = 0; unit < relation->unit_count; ++unit) {
        preservation_ends[result_start + unit] =
            preservation_ends[source_start + unit];
      }
    }
  }

  // Retain the first possible write through each optional identity path. A
  // materialized relation cuts that path, so its sources do not inherit the
  // result's writes. Each relation and its unit mapping are visited once.
  for (loom_value_ordinal_t cursor = 0; cursor < order_count; ++cursor) {
    const loom_low_placement_relation_range_t range =
        placement->ranges_by_result_ordinal[order[cursor]];
    for (uint32_t i = 0; i < range.count; ++i) {
      loom_low_placement_relation_t* relation =
          &placement->relations[range.start + i];
      if (!loom_low_allocation_reuse_relation(relation)) {
        continue;
      }
      const uint32_t source_start =
          unit_starts[relation->source_ordinal] + relation->source_unit_offset;
      const uint32_t result_start =
          unit_starts[relation->result_ordinal] + relation->result_unit_offset;
      bool requires_copy = false;
      if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT) {
        const bool source_segments_complete = !iree_bitmap_test(
            unit_liveness->values_with_incomplete_storage_segments,
            relation->source_ordinal);
        const loom_liveness_segment_range_t source_segments =
            source_segments_complete
                ? loom_liveness_segment_range_for_value_ordinal(
                      liveness, relation->source_ordinal)
                : (loom_liveness_segment_range_t){0};
        uint32_t queried_write_point = UINT32_MAX;
        bool source_live_at_queried_write = false;
        for (uint32_t unit = 0; unit < relation->unit_count; ++unit) {
          const uint32_t preservation_end =
              preservation_ends[source_start + unit];
          const uint32_t first_write = first_writes[result_start + unit];
          if (preservation_end <= first_write) {
            continue;
          }
          if (!source_segments_complete ||
              iree_bitmap_test(tied_preservation_units, source_start + unit)) {
            requires_copy = true;
            break;
          }
          if (queried_write_point != first_write) {
            queried_write_point = first_write;
            source_live_at_queried_write = loom_liveness_segment_range_contains(
                liveness->segments, source_segments, first_write);
          }
          if (source_live_at_queried_write) {
            requires_copy = true;
            break;
          }
        }
      }
      if (requires_copy) {
        relation->flags &= ~LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
        continue;
      }
      for (uint32_t unit = 0; unit < relation->unit_count; ++unit) {
        first_writes[source_start + unit] =
            iree_min(first_writes[source_start + unit],
                     first_writes[result_start + unit]);
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_refine_destructive_reuse(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    loom_low_placement_table_t* placement, iree_arena_allocator_t* arena) {
  bool has_write = false;
  bool has_optional_alias = false;
  for (iree_host_size_t i = 0; i < placement->relation_count; ++i) {
    const loom_low_placement_relation_t* relation = &placement->relations[i];
    has_write |= iree_any_bit_set(
        relation->flags, LOOM_LOW_PLACEMENT_RELATION_FLAG_WRITES_STORAGE);
    has_optional_alias |=
        relation->cause != LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT &&
        loom_low_allocation_reuse_relation(relation);
  }
  if (!has_write || !has_optional_alias) {
    return iree_ok_status();
  }
  iree_arena_allocator_t scratch;
  iree_arena_initialize(arena->block_pool, &scratch);
  iree_status_t status = loom_low_allocation_refine_destructive_reuse_build(
      unit_liveness, liveness, placement, &scratch);
  iree_arena_deinitialize(&scratch);
  return status;
}
