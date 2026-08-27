// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Per-allocation-unit storage lifetime refinements.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_UNIT_LIVENESS_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_UNIT_LIVENESS_H_

#include "iree/base/api.h"
#include "iree/base/bitmap.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/placement.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/ir.h"
#include "loom/ir/local_value_domain.h"

#ifdef __cplusplus
extern "C" {
#endif

// Mutable unit-liveness state indexed by liveness value ordinal.
typedef struct loom_low_allocation_unit_liveness_t {
  // First per-unit lifetime record indexed by liveness local value ordinal.
  // Values without allocatable unit liveness contain UINT32_MAX.
  uint32_t* point_starts_by_value_ordinal;
  // Per-assignment-unit storage start points.
  uint32_t* start_points;
  // Mutable per-assignment-unit live end points.
  uint32_t* end_points;
  // Number of initialized records in |start_points| and |end_points|.
  iree_host_size_t point_count;
  // Values whose concrete storage lifetime is not fully represented by their
  // semantic sparse segments.
  iree_bitmap_t values_with_incomplete_storage_segments;
} loom_low_allocation_unit_liveness_t;

// Initializes |out_unit_liveness| from value-granular liveness and IR use
// structure. The resulting points refine register intervals down to their
// target allocation units for low.slice, descriptor early-clobber hazards, and
// structured loop backedges.
iree_status_t loom_low_allocation_unit_liveness_initialize(
    const loom_module_t* module, const loom_low_resolved_target_t* target,
    const loom_low_placement_table_t* placement,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness, iree_arena_allocator_t* arena,
    loom_low_allocation_unit_liveness_t* out_unit_liveness);

// Returns the first unit-lifetime record for |value_ordinal|, or UINT32_MAX
// when the value has no allocatable unit-liveness records.
uint32_t loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    loom_value_ordinal_t value_ordinal);

// Returns the per-unit storage start points for |value_ordinal|, or NULL when
// the value has no allocatable unit-liveness records.
const uint32_t*
loom_low_allocation_unit_liveness_start_points_for_value_ordinal(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    loom_value_ordinal_t value_ordinal);

// Returns the sparse segment range that is complete for physical storage
// conflicts. Values with decomposed edge-handoff units return an empty range so
// conflict checks conservatively use their refined linear unit lifetimes.
loom_liveness_segment_range_t
loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    loom_value_ordinal_t value_ordinal);

// Propagates storage lifetimes across structural placement relations. Exact
// tied results extend source ends and carry source starts into results.
// Contiguous aggregate parts carry source starts into potential result
// reservations.
iree_status_t loom_low_allocation_unit_liveness_propagate_storage_relations(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_UNIT_LIVENESS_H_
