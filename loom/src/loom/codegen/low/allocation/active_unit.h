// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Hash index for active target-visible allocation units.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_ACTIVE_UNIT_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_ACTIVE_UNIT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_allocation_active_unit_entry_t
    loom_low_allocation_active_unit_entry_t;

// Hash index over active register-like assignment units.
typedef struct loom_low_allocation_active_unit_index_t {
  // Bucket heads into |entries|. Missing buckets contain UINT32_MAX.
  uint32_t* bucket_heads;
  // Power-of-two number of entries in |bucket_heads|.
  uint32_t bucket_count;
  // Reusable unit entries for active assignments.
  loom_low_allocation_active_unit_entry_t* entries;
  // Fixed entry capacity, shared by active units and the free list.
  uint32_t entry_capacity;
  // High-water count of initialized entries in |entries|.
  uint32_t entry_count;
  // Number of entries currently indexed in unit buckets.
  uint32_t active_entry_count;
  // First reusable entry, or UINT32_MAX when the free list is empty.
  uint32_t free_entry_head;
  // First entry for each assignment index. Inactive assignments contain
  // UINT32_MAX.
  uint32_t* entry_starts_by_assignment_index;
  // Number of assignment-index entries tracked by this index.
  iree_host_size_t assignment_capacity;
  // Per-assignment query generations used to skip duplicate range hits.
  uint32_t* seen_generations_by_assignment_index;
  // Current non-zero query generation.
  uint32_t seen_generation;
} loom_low_allocation_active_unit_index_t;

// Initializes |out_index| for up to |assignment_capacity| assignments and
// |unit_capacity| simultaneously indexed units. Tiny indexes are left disabled
// for bounded linear scans. Unrepresentable capacities fail initialization.
iree_status_t loom_low_allocation_active_unit_index_initialize(
    iree_host_size_t assignment_capacity, iree_host_size_t unit_capacity,
    iree_arena_allocator_t* arena,
    loom_low_allocation_active_unit_index_t* out_index);

// Returns true when |index| has allocated storage and can answer hash queries.
bool loom_low_allocation_active_unit_index_is_enabled(
    const loom_low_allocation_active_unit_index_t* index);

// Returns true when |candidate| conflicts with an indexed active assignment.
// Assignment sparse segment ranges, when present, must belong to |liveness|.
bool loom_low_allocation_active_unit_index_conflicts(
    loom_low_allocation_active_unit_index_t* index,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count);

// Appends indexed active assignments that conflict with |candidate| to
// |assignment_indices|. Duplicate range hits for the same assignment are
// suppressed. Assignment sparse segment ranges, when present, must belong to
// |liveness|.
iree_status_t loom_low_allocation_active_unit_index_collect_conflicts(
    loom_low_allocation_active_unit_index_t* index,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count,
    uint32_t* assignment_indices, uint16_t assignment_capacity,
    uint16_t* inout_assignment_count);

// Inserts a register-like |assignment_index| into an enabled |index|. The
// caller's active-unit bound must cover all simultaneously inserted units.
void loom_low_allocation_active_unit_index_insert_assignment(
    loom_low_allocation_active_unit_index_t* index,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, uint32_t assignment_index);

// Removes a previously inserted |assignment_index| from an enabled |index|.
void loom_low_allocation_active_unit_index_remove_assignment(
    loom_low_allocation_active_unit_index_t* index,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, uint32_t assignment_index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_ACTIVE_UNIT_H_
