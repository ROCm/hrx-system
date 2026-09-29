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
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/placement.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/ir.h"
#include "loom/ir/local_value_domain.h"

#ifdef __cplusplus
extern "C" {
#endif

// Indexed physical write point retained by the unit-liveness producer.
typedef struct loom_low_allocation_clobber_t loom_low_allocation_clobber_t;

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
  // Sparse physical reservations, separate from semantic SSA liveness.
  struct {
    // Borrowed semantic segments, or arena-owned semantic prefix followed by
    // tied-source reservations. Assignment ranges index this table.
    const loom_liveness_segment_t* entries;
    // Optional arena-owned tied-source ranges indexed by value ordinal.
    // Empty entries retain conservative per-unit bounds for incomplete values.
    const loom_liveness_segment_range_t* tied_sources;
  } storage_segments;
  // Implicit physical writes, sorted by storage identity and program point
  // after construction. These occupy storage without defining SSA values.
  struct {
    // Arena-owned atomic-unit write points.
    loom_low_allocation_clobber_t* entries;
    // Number of initialized write points.
    iree_host_size_t count;
    // Allocated entry capacity used during construction.
    iree_host_size_t capacity;
    // Smallest explicit atomic unit written when atomic_unit_end is nonzero.
    uint32_t atomic_unit_begin;
    // One past the largest explicit atomic unit written; zero for none.
    uint32_t atomic_unit_end;
  } clobbers;
} loom_low_allocation_unit_liveness_t;

// Initializes |out_unit_liveness| from value-granular liveness and IR use
// structure. The resulting points refine register intervals down to their
// target allocation units for low.slice, descriptor early-clobber hazards, and
// structured loop backedges.
iree_status_t loom_low_allocation_unit_liveness_initialize(
    const loom_low_resolved_target_t* target,
    const loom_low_placement_table_t* placement,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness, iree_arena_allocator_t* arena,
    loom_low_allocation_unit_liveness_t* out_unit_liveness);

// Returns true when an implicit physical write overlaps |candidate|'s refined
// per-unit storage lifetime and sparse live segments. Write points remain
// reusable before and after the instruction; no register is globally reserved.
bool loom_low_allocation_unit_liveness_clobber_conflicts(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* candidate);

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
// conflicts, indexing |unit_liveness->storage_segments.entries|. Values with
// decomposed edge-handoff units return an empty range so conflict checks
// conservatively use their refined linear unit lifetimes.
loom_liveness_segment_range_t
loom_low_allocation_unit_liveness_storage_segment_range_for_value_ordinal(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    loom_value_ordinal_t value_ordinal);

// Completes physical lifetime facts for mandatory tied-storage components.
// Component origins retain every member's per-unit end and sparse segments so
// destructive-reuse refinement can query exact old-content observations before
// deciding which optional storage relations remain aliasable.
iree_status_t loom_low_allocation_unit_liveness_retain_tied_storage(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement, iree_arena_allocator_t* arena);

// Returns true when any unit in |unit_offset, unit_count| of |value_ordinal|'s
// required tied component retains concrete storage across |program_point|.
// The retained component origin owns the complete per-unit ends and sparse
// physical segments after retain_tied_storage, making the query independent of
// tied-chain depth.
bool loom_low_allocation_unit_liveness_storage_component_live_at_point(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement,
    loom_value_ordinal_t value_ordinal, uint32_t unit_offset,
    uint32_t unit_count, uint32_t program_point);

// Propagates storage starts across the final structural placement relations.
// Sources flow into tied results, and contiguous aggregate parts carry source
// starts into accepted result reservations. Call after optional alias
// permissions have been refined.
void loom_low_allocation_unit_liveness_propagate_storage_relations(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_placement_table_t* placement);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_UNIT_LIVENESS_H_
