// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Linear interval assignment construction for target-low allocation.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_INTERVAL_ASSIGNMENT_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_INTERVAL_ASSIGNMENT_H_

#include "iree/base/api.h"
#include "iree/base/bitmap.h"
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
#include "loom/target/residency.h"
#include "loom/util/cfg_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

struct loom_low_schedule_table_t;
struct loom_low_allocation_concat_affinity_t;

// Concrete-location ordering used for one whole-function assignment attempt.
typedef enum loom_low_allocation_search_strategy_e {
  // Searches legal locations from low to high.
  LOOM_LOW_ALLOCATION_SEARCH_STRATEGY_FIRST_FIT = 0,
  // Ranks spill victim sets by predicted traffic per unit of remaining
  // register-pressure area removed.
  LOOM_LOW_ALLOCATION_SEARCH_STRATEGY_PRESSURE_EFFICIENCY = 1,
  // Separates overlapping scalar and wide intervals across the feasible
  // liveness-pressure frontier to repair first-fit fragmentation.
  LOOM_LOW_ALLOCATION_SEARCH_STRATEGY_FRAGMENTATION_REPAIR = 2,
  // Reuses an exact retained location for every current value. The attempt
  // stops without a planning error when any location is unavailable under the
  // current liveness and target constraints.
  LOOM_LOW_ALLOCATION_SEARCH_STRATEGY_RETAINED_LOCATIONS = 3,
} loom_low_allocation_search_strategy_t;

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
  // Acquired value domain shared by liveness and consumption queries.
  const loom_local_value_domain_t* value_domain;
  // Optional final schedule defining |liveness|'s top-level operation order.
  const struct loom_low_schedule_table_t* schedule;
  // Function-local placement relations over |liveness|.
  const loom_low_placement_table_t* placement;
  // Bound instruction preferences owned by the enclosing allocation pass.
  const loom_low_placement_preference_index_t* preferences;
  // Unique concat-chain spans retained after optional alias refinement.
  const struct loom_low_allocation_concat_affinity_t* concat_affinity;
  // Mutable target storage budgets, fixed values, and reserved ranges.
  loom_low_allocation_target_constraints_t* target_constraints;
  // Borrowed incoming locations indexed by formal-argument ordinal.
  const loom_low_allocation_abi_location_t* entry_locations;
  // Number of entries in |entry_locations|.
  iree_host_size_t entry_location_count;
  // Proven boundary representations, independent of physical register search.
  const loom_low_storage_transport_t* storage_transport;
  // Per-allocation-unit liveness facts for |liveness|.
  const loom_low_allocation_unit_liveness_t* unit_liveness;
  // Mutable assignment-backed storage leases and release actions.
  loom_low_allocation_storage_lease_state_t* storage_leases;
  // Arena owning returned assignment, spill, and remark arrays.
  iree_arena_allocator_t* arena;
  // Shared read-only control-flow graph for |body|.
  const loom_cfg_graph_t* function_cfg_graph;
  // Function-local residency view used for physical extent decisions.
  loom_target_residency_view_t residency;
  // Borrowed bitmap indexed by module value ID. Set values require register
  // storage throughout allocation.
  iree_bitmap_t required_register_values;
  // Borrowed bitmap indexed by module value ID. Set values have definitions
  // that write physical storage outside their semantic execution domain.
  iree_bitmap_t expanded_execution_values;
  // Optional exact locations indexed by the current liveness value ordinal.
  // Required by RETAINED_LOCATIONS and ignored by other strategies.
  const loom_low_allocation_assignment_t* const*
      retained_assignments_by_value_ordinal;
  // Optional borrowed exclusive spill-traffic limit for this assignment
  // attempt. Reaching the limit ends the attempt with a partial result that
  // the comparison caller must discard. Null leaves the attempt unbounded.
  const uint64_t* spill_traffic_exclusive_limit;
  // Concrete-location ordering for this whole-function assignment attempt.
  loom_low_allocation_search_strategy_t search_strategy;
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
  // Per-unit end points selected by this assignment attempt. This normally
  // borrows the input analysis and owns a private copy only when spill
  // materialization retires a structural source observation.
  const uint32_t* unit_end_points;
  // Spill materialization plan records in spill-decision order. Null when no
  // assignment spills; otherwise sized to |spill_count|.
  loom_low_allocation_spill_plan_t* spill_plans;
  // Number of initialized spill materialization plan records.
  iree_host_size_t spill_plan_count;
  // Distinct retained-read blockers owned by this assignment attempt.
  loom_low_allocation_retained_fixed_values_t retained_fixed_values;
  // Predicted store and reload bytes across the spill materialization plans.
  uint64_t spill_traffic_bytes;
  // Allocation remark records in spill-decision order. Null when no assignment
  // spills; otherwise sized to |spill_count|.
  loom_low_allocation_remark_t* remarks;
  // Number of initialized allocation remark records.
  iree_host_size_t remark_count;
  // Number of assignments whose location kind is SPILL_SLOT.
  iree_host_size_t spill_count;
  // True when assignment stopped after reaching the comparison-only spill
  // traffic limit. All assignment state is partial and must be discarded.
  bool reached_spill_traffic_limit;
  // True when an exact retained location could not be replayed. This is a
  // legitimate comparison rejection; all partial assignment state is
  // discarded by the caller.
  bool retained_location_replay_failed;
  // Interval ordering found a multi-unit interval in a contiguous register
  // class. Without one, fragmentation repair repeats the same placement.
  bool has_packable_aggregates;
} loom_low_allocation_interval_assignment_result_t;

// Assigns concrete locations for allocatable intervals in |context| and writes
// arena-owned assignment, spill-plan, remark, and lookup table state. Working
// indexes borrow |scratch_arena|'s tail and are released before returning.
// |scratch_arena| must be distinct from |context->arena|; its earlier contents
// remain valid throughout assignment and after returning.
iree_status_t loom_low_allocation_interval_assignment_build(
    const loom_low_allocation_interval_assignment_context_t* context,
    iree_arena_allocator_t* scratch_arena,
    loom_low_allocation_interval_assignment_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_INTERVAL_ASSIGNMENT_H_
