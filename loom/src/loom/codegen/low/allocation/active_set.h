// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Ordered active assignment window used during interval allocation.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_ACTIVE_SET_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_ACTIVE_SET_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/active_unit.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

// Active assignments sorted by increasing end point, plus a unit index for
// fast conflicts against register-like assignments.
typedef struct loom_low_allocation_active_set_t {
  // Assignment indices sorted by increasing assignment end point.
  uint32_t* assignment_indices;
  // First active entry in |assignment_indices|.
  iree_host_size_t start;
  // Number of active entries in |assignment_indices|.
  iree_host_size_t count;
  // Hash index for active register-like assignment units.
  loom_low_allocation_active_unit_index_t units;
} loom_low_allocation_active_set_t;

// Initializes |out_active_set| for |assignment_capacity| assignments and
// |unit_capacity| active unit-index entries.
iree_status_t loom_low_allocation_active_set_initialize(
    iree_host_size_t assignment_capacity, iree_host_size_t unit_capacity,
    iree_arena_allocator_t* arena,
    loom_low_allocation_active_set_t* out_active_set);

// Returns true when |existing| conflicts with |candidate|.
bool loom_low_allocation_active_assignment_conflicts(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness, const uint32_t* unit_end_points,
    iree_host_size_t unit_end_point_count,
    const loom_low_allocation_assignment_t* existing,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count);

// Returns true when |candidate| conflicts with an active assignment.
bool loom_low_allocation_active_set_conflicts(
    loom_low_allocation_active_set_t* active_set,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness, const uint32_t* unit_end_points,
    iree_host_size_t unit_end_point_count,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count);

// Removes expired assignments from the active window and active unit index.
void loom_low_allocation_active_set_expire(
    loom_low_allocation_active_set_t* active_set,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, uint32_t start_point);

// Inserts |assignment_index| into the active window and active unit index.
void loom_low_allocation_active_set_insert(
    loom_low_allocation_active_set_t* active_set,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, uint32_t assignment_index);

// Removes |assignment_index|'s units from the active unit index while leaving
// the assignment in the ordered active window.
void loom_low_allocation_active_set_remove_assignment_units(
    loom_low_allocation_active_set_t* active_set,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, uint32_t assignment_index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_ACTIVE_SET_H_
