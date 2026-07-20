// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation live-range and per-unit end-point queries.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_LIVE_RANGE_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_LIVE_RANGE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_allocation_op_point_entry_t
    loom_low_allocation_op_point_entry_t;

// Hash index mapping low operations to allocation liveness program points.
typedef struct loom_low_allocation_op_point_index_t {
  // Bucket heads into |entries|. Empty buckets contain UINT32_MAX.
  uint32_t* bucket_heads;
  // Power-of-two number of entries in |bucket_heads|.
  uint32_t bucket_count;
  // Operation point entries stored in insertion order.
  loom_low_allocation_op_point_entry_t* entries;
  // Number of initialized entries in |entries|.
  iree_host_size_t entry_count;
} loom_low_allocation_op_point_index_t;

// Returns true when |assignment|'s storage lifetime overlaps |interval|'s
// semantic lifetime.
bool loom_low_allocation_live_range_assignment_overlaps_interval(
    const loom_low_allocation_assignment_t* assignment,
    const loom_liveness_interval_t* interval);

// Returns true when |interval| needs target-visible allocation storage.
bool loom_low_allocation_live_range_interval_is_allocatable(
    const loom_liveness_interval_t* interval);

// Returns the one-past-last storage point for |interval|. A semantic dead
// result still writes a physical destination at its definition boundary, so it
// occupies storage until the next program boundary.
uint32_t loom_low_allocation_live_range_interval_storage_end_point(
    const loom_liveness_interval_t* interval);

// Returns the initial per-unit end point for |interval| before use refinement.
uint32_t loom_low_allocation_live_range_interval_initial_unit_end_point(
    const loom_liveness_interval_t* interval);

// Returns the preferred base-location alignment for |interval|.
uint32_t loom_low_allocation_live_range_interval_alignment(
    const loom_liveness_interval_t* interval);

// Returns the one-past-last live program point for one assigned unit. Unit
// offsets outside |assignment|'s unit-count domain fall back to the assignment
// end point, matching whole-assignment lifetime semantics. |unit_end_points|
// may be NULL only when no in-domain unit offset will be read.
uint32_t loom_low_allocation_live_range_assignment_unit_end_point(
    const uint32_t* unit_end_points, iree_host_size_t unit_end_point_count,
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_offset);

// Returns the maximum one-past-last live program point across all assigned
// units and the whole-assignment end point. |unit_end_points| may be NULL only
// when |assignment| has no units.
uint32_t loom_low_allocation_live_range_assignment_max_unit_end_point(
    const uint32_t* unit_end_points, iree_host_size_t unit_end_point_count,
    const loom_low_allocation_assignment_t* assignment);

// Returns true when two value live ranges can overlap in at least one CFG
// block. Linear interval overlap alone is not enough for branch placement,
// where phi-style values can interleave in the program-point numbering without
// ever being live together in an observable block.
bool loom_low_allocation_live_range_values_overlap(
    const loom_liveness_analysis_t* liveness, loom_value_id_t lhs_value_id,
    uint32_t lhs_start_point, uint32_t lhs_end_point,
    loom_value_id_t rhs_value_id, uint32_t rhs_start_point,
    uint32_t rhs_end_point);

// Returns |op|'s liveness program point by walking its parent block from the
// block start point. This is only valid for source-order liveness.
iree_status_t loom_low_allocation_live_range_op_program_point(
    const loom_liveness_analysis_t* liveness, const loom_op_t* op,
    uint32_t* out_program_point);

// Returns |op|'s liveness program point under an optional explicit operation
// order. Empty orders fall back to source order.
iree_status_t loom_low_allocation_live_range_ordered_op_program_point(
    const loom_liveness_analysis_t* liveness, const loom_region_t* body,
    loom_liveness_order_t liveness_order, const loom_op_t* op,
    uint32_t* out_program_point);

// Builds an operation-to-program-point index using the same point model as
// |liveness|. Empty operation orders use source order; non-empty orders use the
// caller-provided root block order while nested regions remain in source order.
iree_status_t loom_low_allocation_op_point_index_initialize(
    const loom_liveness_analysis_t* liveness,
    loom_liveness_order_t liveness_order, iree_arena_allocator_t* arena,
    loom_low_allocation_op_point_index_t* out_index);

// Returns true and populates |out_program_point| when |op| is indexed.
bool loom_low_allocation_op_point_index_try_lookup(
    const loom_low_allocation_op_point_index_t* index, const loom_op_t* op,
    uint32_t* out_program_point);

// Looks up |op| in |index| or fails if the index does not cover it.
iree_status_t loom_low_allocation_op_point_index_lookup(
    const loom_low_allocation_op_point_index_t* index, const loom_op_t* op,
    uint32_t* out_program_point);

// Returns true when two assignments have overlapping live target-visible
// storage units under descriptor aliasing and per-unit end points.
// Assignment sparse segment ranges, when present, must belong to |liveness|.
// |unit_end_points| may be NULL only when no in-domain unit offset will be
// read.
bool loom_low_allocation_live_range_assignments_conflict(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness, const uint32_t* unit_end_points,
    iree_host_size_t unit_end_point_count,
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_LIVE_RANGE_H_
