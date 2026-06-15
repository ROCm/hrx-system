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
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/assignment.h"
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
  // Unit entries stored for active and previously-active assignments.
  loom_low_allocation_active_unit_entry_t* entries;
  // Maximum number of entries that can be appended to |entries|.
  iree_host_size_t entry_capacity;
  // Number of initialized entries in |entries|.
  iree_host_size_t entry_count;
  // First entry for each assignment index. Unindexed assignments contain
  // UINT32_MAX.
  uint32_t* entry_starts_by_assignment_index;
  // Number of assignment-index entries tracked by this index.
  iree_host_size_t assignment_capacity;
  // Currently active register-like assignments not represented in |entries|.
  iree_host_size_t unindexed_count;
  // Per-assignment query generations used to skip duplicate range hits.
  uint32_t* seen_generations_by_assignment_index;
  // Current non-zero query generation.
  uint32_t seen_generation;
} loom_low_allocation_active_unit_index_t;

// Initializes |out_index| for up to |assignment_capacity| assignments and
// |unit_capacity| indexed units. Tiny or oversized indexes are left disabled.
iree_status_t loom_low_allocation_active_unit_index_initialize(
    iree_host_size_t assignment_capacity, iree_host_size_t unit_capacity,
    iree_arena_allocator_t* arena,
    loom_low_allocation_active_unit_index_t* out_index);

// Returns true when |index| has allocated storage and can answer hash queries.
bool loom_low_allocation_active_unit_index_is_enabled(
    const loom_low_allocation_active_unit_index_t* index);

// Returns true when |assignment_index| is currently represented in |index|.
bool loom_low_allocation_active_unit_index_contains_assignment(
    const loom_low_allocation_active_unit_index_t* index,
    uint32_t assignment_index);

// Returns the number of currently active register-like assignments that could
// not be represented in |index|.
iree_host_size_t loom_low_allocation_active_unit_index_unindexed_count(
    const loom_low_allocation_active_unit_index_t* index);

// Returns true when |candidate| conflicts with an indexed active assignment.
bool loom_low_allocation_active_unit_index_conflicts(
    loom_low_allocation_active_unit_index_t* index,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness, const uint32_t* unit_end_points,
    iree_host_size_t unit_end_point_count,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count);

// Inserts |assignment_index| into |index| when its units can be represented.
// Register-like assignments that cannot fit are counted as unindexed.
void loom_low_allocation_active_unit_index_insert_assignment(
    loom_low_allocation_active_unit_index_t* index,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, uint32_t assignment_index);

// Removes |assignment_index| from |index| or from the unindexed active count.
void loom_low_allocation_active_unit_index_remove_assignment(
    loom_low_allocation_active_unit_index_t* index,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, uint32_t assignment_index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_ACTIVE_UNIT_H_
