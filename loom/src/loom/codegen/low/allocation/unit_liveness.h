// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Per-allocation-unit live end points.

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
  // Unit end-point starts indexed by liveness local value ordinal. Values
  // without allocatable unit liveness contain UINT32_MAX.
  uint32_t* end_point_starts_by_value_ordinal;
  // Mutable per-assignment-unit live end points.
  uint32_t* end_points;
  // Number of initialized records in |end_points|.
  iree_host_size_t end_point_count;
  // Values whose concrete units remain live through a decomposed edge payload.
  // Their semantic sparse segments are incomplete for storage conflicts.
  iree_bitmap_t values_with_edge_handoff_units;
} loom_low_allocation_unit_liveness_t;

// Initializes |out_unit_liveness| from value-granular liveness and IR use
// structure. The resulting end points refine register intervals down to their
// target allocation units for low.slice, descriptor early-clobber hazards, and
// structured loop backedges.
iree_status_t loom_low_allocation_unit_liveness_initialize(
    const loom_module_t* module, loom_region_t* body,
    const loom_low_resolved_target_t* target,
    loom_liveness_order_t liveness_order,
    const loom_low_placement_table_t* placement,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness, iree_arena_allocator_t* arena,
    loom_low_allocation_unit_liveness_t* out_unit_liveness);

// Returns the unit end-point start for |value_ordinal|, or UINT32_MAX when the
// value has no allocatable unit-liveness records.
uint32_t loom_low_allocation_unit_liveness_end_point_start_for_value_ordinal(
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

// Extends tied-result source unit end points to cover the tied result unit end
// points they must share storage with.
iree_status_t loom_low_allocation_unit_liveness_extend_for_tied_results(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_UNIT_LIVENESS_H_
