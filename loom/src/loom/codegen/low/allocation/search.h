// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Concrete location search and active spill-victim selection.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_SEARCH_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_SEARCH_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/active_set.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/allocation/assignment_map.h"
#include "loom/codegen/low/allocation/storage_lease.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Borrowed allocator facts used when probing physical storage.
typedef struct loom_low_allocation_search_context_t {
  // Module containing the allocated low function.
  const loom_module_t* module;
  // Descriptor set selected for the low function.
  const loom_low_descriptor_set_t* descriptor_set;
  // Liveness facts for the allocated low function body.
  const loom_liveness_analysis_t* liveness;
  // Per-allocation-unit liveness facts for |liveness|.
  const loom_low_allocation_unit_liveness_t* unit_liveness;
  // Target storage budgets, fixed values, and reserved ranges.
  const loom_low_allocation_target_constraints_t* target_constraints;
  // Current assignment lookup table.
  const loom_low_allocation_assignment_map_t* assignment_map;
  // Active assignment window at the interval currently being assigned.
  loom_low_allocation_active_set_t* active_set;
  // Materialized storage leases and release eligibility.
  const loom_low_allocation_storage_lease_state_t* storage_leases;
} loom_low_allocation_search_context_t;

// Active assignment set selected for spilling before an interval is assigned.
typedef struct loom_low_allocation_search_spill_victim_set_t {
  // Base location where the incoming interval can be placed after spilling.
  uint32_t location_base;
  // Assignment indices to spill before placing the incoming interval.
  const uint32_t* assignment_indices;
  // Number of entries in |assignment_indices|.
  uint16_t assignment_count;
  // True when a legal victim set was found.
  bool found;
} loom_low_allocation_search_spill_victim_set_t;

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

// Finds the first concrete location for |interval| under |capacity|.
bool loom_low_allocation_search_find_free_location(
    loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* interval,
    loom_low_allocation_class_capacity_t capacity, uint32_t* out_base);

// Returns whether an active assignment may be spilled, and the capacity needed
// to materialize its spill slot when requested.
iree_status_t loom_low_allocation_search_assignment_spill_capacity(
    const loom_low_allocation_search_context_t* context,
    const loom_low_allocation_assignment_t* assignment, bool* out_can_spill,
    loom_low_allocation_class_capacity_t* out_capacity);

// Finds the best active-assignment victim set that would make |interval|
// assignable under |capacity|. The returned assignment indices are arena-owned.
iree_status_t loom_low_allocation_search_find_active_spill_victim_set(
    loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_allocation_class_capacity_t* capacity,
    bool interval_requires_register, iree_arena_allocator_t* arena,
    loom_low_allocation_search_spill_victim_set_t* out_victim_set);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_SEARCH_H_
