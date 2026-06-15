// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Linear interval assignment construction for target-low allocation.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_INTERVAL_ASSIGNMENT_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_INTERVAL_ASSIGNMENT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/allocation/assignment_map.h"
#include "loom/codegen/low/allocation/storage_lease.h"
#include "loom/codegen/low/allocation/table.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/placement.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_allocation_interval_assignment_context_t {
  // Module containing the allocated low function.
  loom_module_t* module;
  // Body region of the allocated low function.
  loom_region_t* body;
  // Low function definition operation being allocated.
  const loom_op_t* function_op;
  // Resolved target selected by the low function.
  const loom_low_resolved_target_t* target;
  // Liveness facts for the allocated low function body.
  const loom_liveness_analysis_t* liveness;
  // Function-local placement relations over |liveness|.
  const loom_low_placement_table_t* placement;
  // Mutable target storage budgets, fixed values, and reserved ranges.
  loom_low_allocation_target_constraints_t* target_constraints;
  // Per-allocation-unit liveness facts for |liveness|.
  const loom_low_allocation_unit_liveness_t* unit_liveness;
  // Mutable assignment-backed storage leases and release actions.
  loom_low_allocation_storage_lease_state_t* storage_leases;
  // Arena owning returned assignment, spill, and remark arrays.
  iree_arena_allocator_t* arena;
} loom_low_allocation_interval_assignment_context_t;

typedef struct loom_low_allocation_interval_assignment_result_t {
  // Assignment records in allocation order.
  loom_low_allocation_assignment_t* assignments;
  // Number of initialized assignment records.
  iree_host_size_t assignment_count;
  // Assignment indices by liveness local value ordinal.
  uint32_t* assignment_indices_by_value_ordinal;
  // Lookup table over assignments and liveness-local value ordinals.
  loom_low_allocation_assignment_map_t assignment_map;
  // Spill materialization plan records in assignment order.
  loom_low_allocation_spill_plan_t* spill_plans;
  // Number of initialized spill materialization plan records.
  iree_host_size_t spill_plan_count;
  // Allocation remark records in assignment order.
  loom_low_allocation_remark_t* remarks;
  // Number of initialized allocation remark records.
  iree_host_size_t remark_count;
  // Number of assignments whose location kind is SPILL_SLOT.
  iree_host_size_t spill_count;
} loom_low_allocation_interval_assignment_result_t;

// Assigns concrete locations for allocatable intervals in |context| and writes
// arena-owned assignment, spill-plan, remark, and lookup table state.
iree_status_t loom_low_allocation_interval_assignment_build(
    const loom_low_allocation_interval_assignment_context_t* context,
    loom_low_allocation_interval_assignment_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_INTERVAL_ASSIGNMENT_H_
