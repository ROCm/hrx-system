// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Spill eligibility, coupled victim selection, and reconstruction cost ranking.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_SPILL_SEARCH_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_SPILL_SEARCH_H_

#include "loom/codegen/low/allocation/search.h"

#ifdef __cplusplus
extern "C" {
#endif

// Reusable spill-search arrays owned by one interval-assignment attempt. Start
// zero-initialized; storage grows in the assignment scratch arena and is
// released with that arena. Contents have no meaning across searches.
typedef struct loom_low_allocation_spill_search_workspace_t {
  // Entry capacity of each of the three coallocated arrays below.
  iree_host_size_t capacity;
  // Conflicting assignments for the location currently being considered.
  uint32_t* candidate_assignment_indices;
  // Best victim set found during the current search.
  uint32_t* best_assignment_indices;
  // Candidate victims excluded while checking remaining location conflicts.
  loom_value_id_t* ignored_value_ids;
  // Candidate victims already selected, indexed by active-set position.
  // This keeps covering-source deduplication linear without another u32 array.
  iree_bitmap_t candidate_selected_assignments;
  // Entry capacity of |ignored_storage_lease_value_ids|.
  iree_host_size_t ignored_storage_lease_capacity;
  // Selected victim values whose storage leases retire with their spills.
  // Allocated lazily because ordinary victim sets use |ignored_value_ids|.
  loom_value_id_t* ignored_storage_lease_value_ids;
  // Entry capacity of |best_released_value_ids|.
  iree_host_size_t released_value_capacity;
  // Best source reservations released by materializing covering victims.
  // Allocated lazily because ordinary victim searches never need this array.
  loom_value_id_t* best_released_value_ids;
} loom_low_allocation_spill_search_workspace_t;

// Active assignment set selected for spilling before an interval is assigned.
typedef struct loom_low_allocation_spill_search_victim_set_t {
  // Base location where the incoming interval can be placed after spilling.
  uint32_t location_base;
  // Assignment indices to spill before placing the incoming interval. Borrowed
  // from the workspace until its next search or scratch-arena release.
  const uint32_t* assignment_indices;
  // Number of entries in |assignment_indices|.
  uint16_t assignment_count;
  // Active source values whose extended physical reservation ends when the
  // selected covering victims are materialized. Borrowed from the workspace
  // until its next search or scratch-arena release.
  const loom_value_id_t* released_value_ids;
  // Number of entries in |released_value_ids|.
  uint16_t released_value_count;
  // True when a legal victim set was found.
  bool found;
} loom_low_allocation_spill_search_victim_set_t;

// Returns whether an active assignment may be spilled, and the capacity needed
// to materialize its spill slot when requested.
iree_status_t loom_low_allocation_spill_search_assignment_capacity(
    const loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* assignment, bool* out_can_spill,
    loom_low_allocation_class_capacity_t* out_capacity);

// Finds the best active-assignment victim set that would make |interval|
// assignable under |capacity|. Grows |workspace| in |arena| as needed; consumes
// no additional storage while the active set fits its current capacity. The
// returned indices must be consumed before the next search using |workspace|.
iree_status_t loom_low_allocation_spill_search_find_victims(
    loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_allocation_class_capacity_t* capacity,
    bool interval_requires_register,
    loom_low_allocation_spill_search_workspace_t* workspace,
    iree_arena_allocator_t* arena,
    loom_low_allocation_spill_search_victim_set_t* out_victim_set);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_SPILL_SEARCH_H_
