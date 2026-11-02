// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/content_identity.h"

#include <string.h>

#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/placement.h"

static bool loom_low_allocation_content_relation_preserves_identity(
    const loom_low_placement_relation_t* relation) {
  return relation->cause >= LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT &&
         relation->cause <= LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT &&
         loom_low_placement_relation_can_alias(relation) &&
         !iree_any_bit_set(relation->flags,
                           LOOM_LOW_PLACEMENT_RELATION_FLAG_WRITES_STORAGE |
                               LOOM_LOW_PLACEMENT_RELATION_FLAG_CAPTURED_PART);
}

iree_status_t loom_low_allocation_content_identity_initialize(
    const loom_low_placement_table_t* placement,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    iree_arena_allocator_t* arena,
    loom_low_allocation_content_identity_t* out_identity) {
  *out_identity = (loom_low_allocation_content_identity_t){0};
  if (!iree_any_bit_set(placement->storage.flags,
                        LOOM_LOW_PLACEMENT_STORAGE_FLAG_OPTIONAL_ALIASES |
                            LOOM_LOW_PLACEMENT_STORAGE_FLAG_IDENTITY_ALIASES) ||
      placement->storage_value_order_count == 0 ||
      unit_liveness->point_count == 0) {
    return iree_ok_status();
  }
  uint32_t* origins = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, unit_liveness->point_count, sizeof(*origins), (void**)&origins));
  for (iree_host_size_t i = 0; i < unit_liveness->point_count; ++i) {
    origins[i] = i;
  }
  // The retained order visits users before sources, so reverse traversal
  // assigns each user from an already canonical source root.
  for (loom_value_ordinal_t cursor = placement->storage_value_order_count;
       cursor > 0; --cursor) {
    const loom_value_ordinal_t result_ordinal =
        placement->storage_value_order[cursor - 1];
    const uint32_t result_start =
        unit_liveness->values[result_ordinal].unit_point_start;
    if (result_start == UINT32_MAX) {
      continue;
    }
    const loom_low_placement_relation_range_t range =
        placement->ranges_by_result_ordinal[result_ordinal];
    for (uint32_t i = 0; i < range.count; ++i) {
      const loom_low_placement_relation_t* relation =
          &placement->relations[range.start + i];
      if (!loom_low_allocation_content_relation_preserves_identity(relation)) {
        continue;
      }
      const uint32_t source_start =
          unit_liveness->values[relation->source_ordinal].unit_point_start;
      if (source_start == UINT32_MAX) {
        continue;
      }
      for (uint32_t unit = 0; unit < relation->unit_count; ++unit) {
        origins[result_start + relation->result_unit_offset + unit] =
            origins[source_start + relation->source_unit_offset + unit];
      }
    }
  }
  out_identity->origins = origins;
  if (placement->storage.write_relation_count != 0) {
    const iree_host_size_t word_count =
        iree_bitmap_calculate_words(unit_liveness->point_count);
    uint64_t* written_words = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, word_count, sizeof(*written_words), (void**)&written_words));
    memset(written_words, 0, word_count * sizeof(*written_words));
    out_identity->written_origins = (iree_bitmap_t){
        .bit_count = unit_liveness->point_count,
        .words = written_words,
    };
    for (uint32_t i = 0; i < placement->storage.write_relation_count; ++i) {
      const loom_low_placement_relation_t* relation =
          &placement->relations[placement->storage.write_relation_indices[i]];
      const uint32_t start =
          unit_liveness->values[relation->source_ordinal].unit_point_start;
      for (uint32_t unit = 0; unit < relation->unit_count; ++unit) {
        iree_bitmap_set(out_identity->written_origins,
                        origins[start + relation->source_unit_offset + unit]);
      }
    }
  }
  return iree_ok_status();
}
