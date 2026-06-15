// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Placement-driven interval coalescing during low allocation.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_COALESCING_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_COALESCING_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/consumption.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/allocation/assignment_map.h"
#include "loom/codegen/low/allocation/search.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/placement.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef iree_status_t (*loom_low_allocation_coalescing_append_assignment_fn_t)(
    void* user_data, const loom_low_allocation_assignment_t* assignment,
    const loom_value_id_t* ignored_storage_lease_value_ids,
    uint16_t ignored_storage_lease_value_count, uint32_t* out_assignment_index);

typedef iree_status_t (*loom_low_allocation_coalescing_consumption_query_fn_t)(
    void* user_data, loom_consumption_region_query_t** out_query);

typedef struct loom_low_allocation_coalescing_context_t {
  // Arena used for temporary coalescing scratch.
  iree_arena_allocator_t* arena;
  // Liveness facts for the allocated low function body.
  const loom_liveness_analysis_t* liveness;
  // Function-local placement relations.
  const loom_low_placement_table_t* placement;
  // Target storage budgets, fixed values, and reserved ranges.
  const loom_low_allocation_target_constraints_t* target_constraints;
  // Current assignment lookup table.
  const loom_low_allocation_assignment_map_t* assignment_map;
  // Concrete location search context at the current interval.
  loom_low_allocation_search_context_t* search_context;
  // Callback that commits one coalesced assignment.
  loom_low_allocation_coalescing_append_assignment_fn_t append_assignment;
  // Callback that lazily returns the function consumption query.
  loom_low_allocation_coalescing_consumption_query_fn_t consumption_query;
  // Opaque caller state passed to callbacks.
  void* user_data;
} loom_low_allocation_coalescing_context_t;

// Attempts to assign a descriptor tied-result interval to its tied source.
iree_status_t loom_low_allocation_coalescing_assign_tied_interval(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval, bool* out_assigned);

// Attempts to assign a low.concat source interval using already-known concat,
// sibling-source, or branch-destination storage.
iree_status_t loom_low_allocation_coalescing_assign_concat_source_interval(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval, bool* out_assigned);

// Attempts to assign an interval using result-keyed structural placement
// relations such as low.copy, low.slice, low.concat, and low.br destinations.
iree_status_t loom_low_allocation_coalescing_assign_structural_interval(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval, bool* out_assigned);

// Attempts to assign a low.br source interval using already-assigned branch
// destination storage.
iree_status_t loom_low_allocation_coalescing_assign_branch_source_interval(
    loom_low_allocation_coalescing_context_t* context,
    const loom_liveness_interval_t* interval, bool* out_assigned);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_COALESCING_H_
