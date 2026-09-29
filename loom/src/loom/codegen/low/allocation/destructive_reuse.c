// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/destructive_reuse.h"

#include <string.h>

static bool loom_low_allocation_reuse_relation(
    const loom_low_placement_relation_t* relation) {
  return loom_low_placement_relation_can_alias(relation) &&
         relation->cause >= LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT &&
         relation->cause <= LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT;
}

// Returns true when any mapped storage component remains observable at its
// counterpart's first write. Equal adjacent write points share one indexed
// component query, keeping wide relations proportional to mapped units.
static bool loom_low_allocation_storage_observed_at_first_writes(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement,
    loom_value_ordinal_t observed_ordinal, uint32_t observed_unit_offset,
    const uint32_t* first_writes, uint32_t write_unit_start,
    uint32_t unit_count) {
  uint32_t unit = 0;
  while (unit < unit_count) {
    const uint32_t write_point = first_writes[write_unit_start + unit];
    uint32_t run_count = 1;
    while (run_count < unit_count - unit &&
           first_writes[write_unit_start + unit + run_count] == write_point) {
      ++run_count;
    }
    if (write_point != UINT32_MAX &&
        loom_low_allocation_unit_liveness_storage_component_live_at_point(
            unit_liveness, liveness, placement, observed_ordinal,
            observed_unit_offset + unit, run_count, write_point)) {
      return true;
    }
    unit += run_count;
  }
  return false;
}

static iree_status_t loom_low_allocation_refine_destructive_reuse_build(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    loom_low_placement_table_t* placement, iree_arena_allocator_t* scratch) {
  uint32_t* first_writes = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(scratch, unit_liveness->point_count,
                                sizeof(*first_writes), (void**)&first_writes));
  memset(first_writes, 0xFF,
         unit_liveness->point_count * sizeof(*first_writes));
  const uint32_t* unit_starts = unit_liveness->point_starts_by_value_ordinal;
  for (iree_host_size_t i = 0; i < placement->relation_count; ++i) {
    const loom_low_placement_relation_t* relation = &placement->relations[i];
    if (!loom_low_allocation_reuse_relation(relation)) {
      continue;
    }
    const uint32_t source_start =
        unit_starts[relation->source_ordinal] + relation->source_unit_offset;
    if (iree_any_bit_set(relation->flags,
                         LOOM_LOW_PLACEMENT_RELATION_FLAG_WRITES_STORAGE)) {
      for (uint32_t unit = 0; unit < relation->unit_count; ++unit) {
        first_writes[source_start + unit] =
            iree_min(first_writes[source_start + unit], relation->write_point);
      }
    }
  }

  const loom_value_ordinal_t* order = placement->storage_value_order;
  const loom_value_ordinal_t order_count = placement->storage_value_order_count;
  IREE_ASSERT_EQ(order_count, placement->value_count);

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
        requires_copy = loom_low_allocation_storage_observed_at_first_writes(
            unit_liveness, liveness, placement, relation->source_ordinal,
            relation->source_unit_offset, first_writes, result_start,
            relation->unit_count);
        if (!requires_copy) {
          requires_copy = loom_low_allocation_storage_observed_at_first_writes(
              unit_liveness, liveness, placement, relation->result_ordinal,
              relation->result_unit_offset, first_writes, source_start,
              relation->unit_count);
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
