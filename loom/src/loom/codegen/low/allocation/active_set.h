// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Active assignments and monotone lifetime expiration during allocation.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_ACTIVE_SET_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_ACTIVE_SET_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/active_unit.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

// An assignment's membership and scheduled expiration in the active set.
typedef struct loom_low_allocation_active_entry_t {
  // Position in the dense active assignment array, or UINT32_MAX after removal.
  uint32_t position;
  // Next assignment expiring at the same program point, or UINT32_MAX.
  uint32_t next_expiration;
} loom_low_allocation_active_entry_t;

// Dense active membership and an expiration calendar over program points.
//
// Insertion and removal take constant time. A monotone sweep visits each
// calendar point and expiration entry once, independent of lifetime nesting.
// Register conflicts use the unit index; unindexed storage scans only active
// assignments, not expired or spilled history.
typedef struct loom_low_allocation_active_set_t {
  // Sparse liveness segments used to reject false linear-interval conflicts.
  const loom_liveness_analysis_t* liveness;
  // Densely packed active assignment indices, in unspecified order.
  uint32_t* assignment_indices;
  // Membership and expiration links indexed by assignment index.
  loom_low_allocation_active_entry_t* entries;
  // First assignment expiring at each program point, or UINT32_MAX.
  uint32_t* expiration_heads;
  // Number of program points represented by the expiration calendar.
  iree_host_size_t program_point_count;
  // First program point whose expiration list has not been consumed.
  iree_host_size_t next_expiration_point;
  // Number of active entries in |assignment_indices|.
  iree_host_size_t count;
  // Hash index for active register-like assignment units.
  loom_low_allocation_active_unit_index_t units;
} loom_low_allocation_active_set_t;

// Initializes |out_active_set| for |assignment_capacity| assignments and
// |unit_capacity| active unit-index entries. All inserted assignments must end
// before |program_point_count|. |liveness| is borrowed and must outlive the
// set.
iree_status_t loom_low_allocation_active_set_initialize(
    const loom_liveness_analysis_t* liveness,
    iree_host_size_t assignment_capacity, iree_host_size_t program_point_count,
    iree_host_size_t unit_capacity, iree_arena_allocator_t* arena,
    loom_low_allocation_active_set_t* out_active_set);

// Returns true when |existing| conflicts with |candidate|. Both assignments'
// sparse segment ranges, when present, must belong to |liveness|.
bool loom_low_allocation_active_assignment_conflicts(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_allocation_assignment_t* existing,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count);

// Returns true when |candidate| conflicts with an active assignment.
bool loom_low_allocation_active_set_conflicts(
    loom_low_allocation_active_set_t* active_set,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count);

// Removes assignments ending at or before |start_point|. Calls advance
// monotonically; each subsequent insertion must end after the swept point.
void loom_low_allocation_active_set_expire(
    loom_low_allocation_active_set_t* active_set,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, uint32_t start_point);

// Inserts |assignment_index| into the active set and expiration calendar.
// Each assignment is inserted once; its end point remains immutable until
// the calendar has consumed its expiration, even if removed before then.
void loom_low_allocation_active_set_insert(
    loom_low_allocation_active_set_t* active_set,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, uint32_t assignment_index);

// Removes an active assignment and its unit-index entries. Its calendar entry
// remains pending and is skipped when its original end point is reached.
void loom_low_allocation_active_set_remove(
    loom_low_allocation_active_set_t* active_set,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, uint32_t assignment_index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_ACTIVE_SET_H_
