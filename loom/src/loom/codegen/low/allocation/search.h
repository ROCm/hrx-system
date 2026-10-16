// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Concrete location search and active spill-victim selection.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_SEARCH_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_SEARCH_H_

#include "iree/base/api.h"
#include "iree/base/bitmap.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/active_set.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/allocation/assignment_map.h"
#include "loom/codegen/low/allocation/physical_domains.h"
#include "loom/codegen/low/allocation/preference.h"
#include "loom/codegen/low/allocation/scalar_packing.h"
#include "loom/codegen/low/allocation/spill_plan.h"
#include "loom/codegen/low/allocation/storage_lease.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/placement.h"
#include "loom/ir/ir.h"
#include "loom/target/residency.h"

#ifdef __cplusplus
extern "C" {
#endif

// Borrowed allocator facts used when probing physical storage.
typedef struct loom_low_allocation_search_context_t {
  // Module with the allocated function's local value domain acquired.
  const loom_module_t* module;
  // Function control-flow graph used to construct |liveness|.
  const loom_cfg_graph_t* cfg_graph;
  // Descriptor set selected for the low function.
  const loom_low_descriptor_set_t* descriptor_set;
  // Liveness facts for the allocated low function body.
  const loom_liveness_analysis_t* liveness;
  // Per-allocation-unit liveness facts for |liveness|.
  const loom_low_allocation_unit_liveness_t* unit_liveness;
  // Target storage budgets, fixed values, and reserved ranges.
  loom_low_allocation_target_constraints_t* target_constraints;
  // Current assignment lookup table.
  const loom_low_allocation_assignment_map_t* assignment_map;
  // Function-local structural and concrete-location placement relations.
  const loom_low_placement_table_t* placement;
  // Allocation-owned bound instruction preferences, or NULL when absent.
  const loom_low_placement_preference_index_t* preferences;
  // Reusable attempt-local preference query storage.
  loom_low_allocation_preference_workspace_t* preference_workspace;
  // Active assignment window at the interval currently being assigned.
  loom_low_allocation_active_set_t* active_set;
  // Attempt-local ordered availability of immutable fixed claims.
  loom_low_allocation_fixed_availability_t* fixed_availability;
  // Materialized storage leases and release eligibility.
  loom_low_allocation_storage_lease_state_t* storage_leases;
  // Cached predicted spill traffic, dense by liveness value ordinal. A
  // store_count of UINT32_MAX means the entry is not computed yet.
  loom_low_allocation_spill_plan_traffic_t* spill_traffic_by_value_ordinal;
  // Function-local residency view used for physical extent decisions.
  loom_target_residency_view_t residency;
  // Borrowed bitmap indexed by module value ID. Set values have definitions
  // that write physical storage outside their semantic execution domain.
  iree_bitmap_t expanded_execution_values;
  // Scalar/aggregate lifetime preferences computed before coloring.
  loom_low_allocation_scalar_packing_t scalar_packing;
  // Optional retained scalar physical-domain preferences for this attempt.
  const loom_low_allocation_physical_domains_t* physical_domains;
  // First fixed retained origin rejecting a location in the current search,
  // encoded as its fixed-value index plus one. Zero means none encountered.
  uint32_t retained_fixed_value_index_plus_one;
} loom_low_allocation_search_context_t;

// Reusable spill-search arrays owned by one interval-assignment attempt. Start
// zero-initialized; storage grows in the assignment scratch arena and is
// released with that arena. Contents have no meaning across searches.
typedef struct loom_low_allocation_search_workspace_t {
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
} loom_low_allocation_search_workspace_t;

// Active assignment set selected for spilling before an interval is assigned.
typedef struct loom_low_allocation_search_spill_victim_set_t {
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
} loom_low_allocation_search_spill_victim_set_t;

// Returns true when |candidate| conflicts with active assignments, fixed
// values, reserved ranges, or storage leases.
bool loom_low_allocation_search_assignment_conflicts(
    loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count,
    const loom_value_id_t* ignored_storage_lease_value_ids,
    uint16_t ignored_storage_lease_value_count,
    loom_low_allocation_storage_release_policy_t release_policy);

// Returns true when the interval assignment candidate conflicts with active
// assignments, fixed values, reserved ranges, or storage leases.
bool loom_low_allocation_search_location_conflicts(
    loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* interval, uint16_t reg_class_id,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count, const loom_value_id_t* ignored_value_ids,
    uint16_t ignored_value_count,
    const loom_value_id_t* ignored_storage_lease_value_ids,
    uint16_t ignored_storage_lease_value_count,
    loom_low_allocation_storage_release_policy_t release_policy);

// Finds a concrete location for |interval| under |capacity| using the context's
// packing plan. Placement preferences and hard conflicts remain primary.
bool loom_low_allocation_search_find_free_location(
    loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* interval,
    loom_low_allocation_class_capacity_t capacity, uint32_t* out_base);

// Enumerates each aligned linear base in [0, last_base] exactly once, with
// preferred-aligned bases first. Both alignments are positive powers of two,
// preferred_alignment >= required_alignment, and candidate_index is less than
// last_base / required_alignment + 1. This order expresses a packing preference
// without excluding legal gaps or changing explicit physical-view semantics.
uint32_t loom_low_allocation_search_linear_candidate_base(
    uint64_t candidate_index, uint32_t last_base, uint32_t required_alignment,
    uint32_t preferred_alignment);

// Attainable residency after extending the current physical frontier to cover
// this candidate. An absent resource model returns zero for every candidate.
uint32_t loom_low_allocation_search_assignment_residency_tier(
    const loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* candidate);

// Returns whether an active assignment may be spilled, and the capacity needed
// to materialize its spill slot when requested.
iree_status_t loom_low_allocation_search_assignment_spill_capacity(
    const loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* assignment, bool* out_can_spill,
    loom_low_allocation_class_capacity_t* out_capacity);

// Finds the best active-assignment victim set that would make |interval|
// assignable under |capacity|. Grows |workspace| in |arena| as needed; consumes
// no additional storage while the active set fits its current capacity. The
// returned indices must be consumed before the next search using |workspace|.
iree_status_t loom_low_allocation_search_find_active_spill_victim_set(
    loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_allocation_class_capacity_t* capacity,
    bool interval_requires_register,
    loom_low_allocation_search_workspace_t* workspace,
    iree_arena_allocator_t* arena,
    loom_low_allocation_search_spill_victim_set_t* out_victim_set);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_SEARCH_H_
