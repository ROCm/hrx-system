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
#include "loom/util/cfg_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

// Indexed physical write point retained by the unit-liveness producer.
typedef struct loom_low_allocation_clobber_t loom_low_allocation_clobber_t;
typedef struct loom_low_allocation_write_interference_t
    loom_low_allocation_write_interference_t;

// Per-value storage lifetime facts retained beside the per-unit lifetime
// cursor.
typedef struct loom_low_allocation_unit_liveness_value_t {
  // First per-unit lifetime record, or UINT32_MAX for a nonallocatable value.
  uint32_t unit_point_start;
  // Earliest storage acquisition required by this value and its mandatory tied
  // descendants, or UINT32_MAX when the value has no allocatable lifetime.
  uint32_t acquisition_start_point;
} loom_low_allocation_unit_liveness_value_t;

// A whole component read at an edge that bypasses the aggregate's SSA storage.
// Concat relations consume complete source values; a partial source is a
// separate slice value, not a subrange duplicated in this observation record.
typedef struct loom_low_allocation_decomposed_use_t {
  // Next node in the required storage component's observation list, or zero.
  uint32_t next_node;
  // Index of the observing operation in canonical liveness operation points.
  uint32_t operation_index;
} loom_low_allocation_decomposed_use_t;

// Mutable unit-liveness state indexed by liveness value ordinal. Published
// per-unit points and storage segments have result-arena lifetime; all other
// owned state has allocation-decision lifetime.
typedef struct loom_low_allocation_unit_liveness_t {
  // Borrowed required storage identities, or NULL when no tied components
  // contribute reservations. The placement value domain remains acquired
  // throughout allocation.
  const loom_low_placement_table_t* tied_storage_placement;
  // Per-value storage facts indexed by liveness local value ordinal.
  loom_low_allocation_unit_liveness_value_t* values;
  // Result-arena-owned per-assignment-unit storage start points.
  uint32_t* start_points;
  // Result-arena-owned mutable per-assignment-unit live end points.
  uint32_t* end_points;
  // Number of initialized records in |start_points| and |end_points|, at most
  // UINT32_MAX. Every nonempty value range ends at or before this count, so no
  // record index aliases the UINT32_MAX sentinel.
  iree_host_size_t point_count;
  // Values whose concrete storage lifetime is not fully represented by their
  // semantic sparse segments.
  iree_bitmap_t values_with_incomplete_storage_segments;
  // Required-component observations retained by the existing edge-use producer.
  // One-based nodes 1..V identify SSA values; V+1..V+D identify decomposed
  // reads. Each component starts at its origin value and ends at node zero.
  struct {
    // Decision-arena-owned successors indexed by value ordinal. An origin's
    // successor starts its indirect observations and other required members.
    // NULL without tied components or decomposed reads at control-flow edges.
    uint32_t* value_links;
    // Decision-arena-owned decomposed reads, indexed by node minus V+1.
    loom_low_allocation_decomposed_use_t* entries;
    // Number of initialized decomposed reads; V+count fits in uint32_t.
    uint32_t count;
    // Capacity of the decomposed-read construction array.
    iree_host_size_t capacity;
  } observations;
  // Linear physical units implicitly read or written at location zero,
  // indexed by descriptor register class. NULL when no such operands occur.
  // Retaining reads as well as writes anchors every implicit location during
  // final numbering, independently of the write-point conflict index.
  uint16_t* implicit_location_counts_by_reg_class;
  // Sparse physical reservations, separate from semantic SSA liveness.
  struct {
    // Borrowed semantic segments, or result-arena-owned semantic prefix
    // followed by tied-source reservations. Assignment ranges index this table.
    const loom_liveness_segment_t* entries;
    // Optional decision-arena-owned tied-source ranges indexed by value
    // ordinal. Empty entries retain conservative per-unit bounds for incomplete
    // values.
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
  // Instruction reads retained beyond semantic value death. Operand events
  // are collected here and finalized after fixed bindings, before assignment.
  loom_low_allocation_write_interference_t* write_interference;
} loom_low_allocation_unit_liveness_t;

// Returns true when |value_id|'s required storage component is excluded by
// |ignored_value_ids|. Callers establish alias or relocation legality before
// excluding a component; its separate SSA names are one physical reservation.
bool loom_low_allocation_unit_liveness_storage_is_ignored(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    loom_value_id_t value_id, const loom_value_id_t* ignored_value_ids,
    uint16_t ignored_value_count);

// Initializes |out_unit_liveness| from value-granular liveness and IR use
// structure over the canonical |cfg_graph|. The resulting points refine
// register intervals down to target allocation units across CFG boundaries,
// low.slice uses, descriptor early-clobber hazards, and structured backedges.
// The complete register-unit extent is bounded before point allocation;
// retained-read construction consumes subsets of this same bounded domain.
// Published point arrays are owned by |result_arena|; query metadata and
// physical access indexes are owned by |decision_arena| through final physical
// numbering. The arenas must be distinct. Construction scratch borrows the
// result arena's tail and is released before returning.
iree_status_t loom_low_allocation_unit_liveness_initialize(
    const loom_low_resolved_target_t* target,
    const loom_low_placement_table_t* placement,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness, const loom_cfg_graph_t* cfg_graph,
    iree_arena_allocator_t* result_arena,
    iree_arena_allocator_t* decision_arena,
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
// Each ancestor acquires storage by its earliest mandatory descendant start.
// Component origins retain every member's physical unit lifetime and sparse
// segments so destructive-reuse refinement can query exact old-content
// observations before deciding which optional relations remain aliasable.
// Published segments use |result_arena|; query ranges use |decision_arena|.
iree_status_t loom_low_allocation_unit_liveness_retain_tied_storage(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement,
    iree_arena_allocator_t* result_arena,
    iree_arena_allocator_t* decision_arena);

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
