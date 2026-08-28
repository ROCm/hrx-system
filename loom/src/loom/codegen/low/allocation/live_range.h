// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation live-range and per-unit end-point queries.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_LIVE_RANGE_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_LIVE_RANGE_H_

#include "iree/base/api.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

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

// Returns the required base-location alignment for |interval| in a register
// class with |reg_class_flags|.
uint32_t loom_low_allocation_live_range_interval_alignment(
    const loom_liveness_interval_t* interval,
    loom_low_reg_class_flags_t reg_class_flags);

// Returns the one-past-last live program point for one assigned unit. Unit
// offsets outside |assignment|'s unit-count domain fall back to the assignment
// end point, matching whole-assignment lifetime semantics. |unit_end_points|
// may be NULL only when no in-domain unit offset will be read.
uint32_t loom_low_allocation_live_range_assignment_unit_end_point(
    const uint32_t* unit_end_points, iree_host_size_t unit_point_count,
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_offset);

// Returns the live start point for |unit_offset| in |assignment|. Unit offsets
// outside the assignment domain and assignments without refined starts use the
// whole-assignment start point. |unit_start_points| may be NULL only when no
// in-domain refined unit offset will be read.
uint32_t loom_low_allocation_live_range_assignment_unit_start_point(
    const uint32_t* unit_start_points, iree_host_size_t unit_start_point_count,
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_offset);

// Returns the maximum one-past-last live program point across all assigned
// units and the whole-assignment end point. |unit_end_points| may be NULL only
// when |assignment| has no units.
uint32_t loom_low_allocation_live_range_assignment_max_unit_end_point(
    const uint32_t* unit_end_points, iree_host_size_t unit_point_count,
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

// Returns true when two assignments have overlapping live target-visible
// storage units under descriptor aliasing and per-unit storage lifetimes.
// Assignment sparse segment ranges, when present, must belong to |liveness|.
bool loom_low_allocation_live_range_assignments_conflict(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness, const uint32_t* unit_start_points,
    const uint32_t* unit_end_points, iree_host_size_t unit_point_count,
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_LIVE_RANGE_H_
