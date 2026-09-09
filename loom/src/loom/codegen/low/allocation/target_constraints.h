// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target storage constraints for low allocation.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_TARGET_CONSTRAINTS_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_TARGET_CONSTRAINTS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/placement.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/ir/local_value_domain.h"
#include "loom/target/registers.h"

#ifdef __cplusplus
extern "C" {
#endif

// Optional fixed register budget used by tests, tuning loops, and target
// overlays. A missing budget uses the descriptor set's allocatable unit count;
// a descriptor class with allocatable_count == 0 is treated as unbounded.
typedef struct loom_low_allocation_budget_t {
  // Stable register-class name.
  iree_string_view_t register_class;
  // Maximum allocation units available for |register_class|.
  uint32_t max_units;
} loom_low_allocation_budget_t;

// Fixed physical location for one SSA value.
//
// Fixed values model ABI or target live-ins/outs that enter allocation as
// ordinary SSA values but must occupy a specific target-visible location while
// live. The location is reusable outside the value's live interval.
typedef struct loom_low_allocation_fixed_value_t {
  // SSA value forced to the fixed location.
  loom_value_id_t value_id;
  // Target-visible fixed location kind.
  loom_low_allocation_location_kind_t location_kind;
  // Base physical register or target ID.
  uint32_t location_base;
  // Number of contiguous units fixed at |location_base|.
  uint32_t location_count;
} loom_low_allocation_fixed_value_t;

// Whole-function location range owned by target machinery.
//
// Reserved ranges model architectural state that is never allocatable for
// ordinary values in the current low function, such as special registers or
// permanently reserved target IDs. Use fixed values instead for ABI live-ins
// whose registers can be reused after their last use.
typedef struct loom_low_allocation_reserved_range_t {
  // Stable register-class name.
  iree_string_view_t register_class;
  // Target-visible reserved location kind.
  loom_low_allocation_location_kind_t location_kind;
  // First physical register or target ID in the reserved range.
  uint32_t location_base;
  // Number of contiguous units reserved at |location_base|.
  uint32_t location_count;
} loom_low_allocation_reserved_range_t;

typedef struct loom_low_allocation_resolved_budget_t {
  // Descriptor-set-local register class ID.
  uint16_t descriptor_reg_class_id;
  // Maximum allocation units available for the class.
  uint32_t max_units;
} loom_low_allocation_resolved_budget_t;

typedef struct loom_low_allocation_resolved_reserved_range_t {
  // Descriptor-set-local register class ID.
  uint16_t descriptor_reg_class_id;
  // Target-visible reserved location kind.
  loom_low_allocation_location_kind_t location_kind;
  // First physical register or target ID in the reserved range.
  uint32_t location_base;
  // Number of contiguous units reserved at |location_base|.
  uint32_t location_count;
} loom_low_allocation_resolved_reserved_range_t;

// Target-validated fixed value prepared for allocation.
//
// |assignment| contains the propagated sparse and per-unit storage lifetime
// used by conflict searches. Explicit preassignments are copied directly into
// the final allocation table; target-implied locations reserve their storage
// until the ordinary interval assignment path selects them.
typedef struct loom_low_allocation_resolved_fixed_value_t {
  // Complete assignment at the required target-visible location.
  loom_low_allocation_assignment_t assignment;
  // Liveness-local ordinal for |assignment.value_id|.
  loom_value_ordinal_t value_ordinal;
  // Exclusive semantic live-range end before storage lifetime refinement.
  uint32_t semantic_end_point;
} loom_low_allocation_resolved_fixed_value_t;

typedef struct loom_low_allocation_class_capacity_t {
  // Descriptor-set-local register class ID.
  uint16_t descriptor_reg_class_id;
  // Location kind used for this class.
  loom_low_allocation_location_kind_t location_kind;
  // Maximum allocation units when |is_bounded| is true.
  uint32_t max_units;
  // Number of bits in one allocation unit.
  uint16_t alloc_unit_bits;
  // Storage space used when this class spills.
  loom_low_spill_slot_space_t spill_slot_space;
  // True when values in this class can be assigned to spill slots.
  bool is_spillable;
  // True when |max_units| is a hard allocation budget.
  bool is_bounded;
} loom_low_allocation_class_capacity_t;

typedef enum loom_low_allocation_failure_blocking_kind_e {
  // No specific blocking constraint was recorded.
  LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_UNKNOWN = 0,
  // The failing interval itself is wider than the register-class budget.
  LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_INTERVAL_EXCEEDS_BUDGET = 1,
  // A live assignment occupies a candidate location and cannot be evicted.
  LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_ACTIVE_ASSIGNMENT = 2,
  // A fixed value, reserved range, or storage lease blocks a candidate
  // location.
  LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_LOCATION_CONSTRAINT = 3,
  // The allocator scanned the candidate locations without finding a legal
  // placement or a more specific blocking constraint.
  LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_NO_ASSIGNABLE_LOCATION = 4,
} loom_low_allocation_failure_blocking_kind_t;

// Terminal storage-planning failure. Interval failures retain the conflicting
// value and storage witness; move failures retain the operation needing
// scratch.
typedef struct loom_low_allocation_failure_t {
  // Stable failure code emitted by the structured diagnostic.
  iree_string_view_t failure_code;
  // Operation whose value or required move could not be assigned storage.
  const loom_op_t* op;
  // SSA value whose interval could not be assigned, or INVALID for a move.
  loom_value_id_t value_id;
  // Pressure/allocation class requiring storage.
  loom_liveness_value_class_t value_class;
  // Descriptor-set-local register class ID for |value_class|.
  uint16_t descriptor_reg_class_id;
  // Program point where the failed interval starts, or UINT32_MAX for a move.
  uint32_t start_point;
  // Exclusive storage end of the failed interval, or UINT32_MAX for a move.
  uint32_t end_point;
  // Allocation units required by the failed interval.
  uint32_t required_unit_count;
  // Maximum allocation units available, or UINT32_MAX when unbounded.
  uint32_t budget_units;
  // Maximum boundary-live units observed for this pressure class.
  uint32_t peak_live_units;
  // Candidate location kind used while diagnosing the failure.
  loom_low_allocation_location_kind_t location_kind;
  // Candidate base physical register or target ID used for conflict reporting,
  // or UINT32_MAX when no concrete candidate was inspected.
  uint32_t location_base;
  // Candidate location width used for conflict reporting, or zero when no
  // concrete candidate was inspected.
  uint32_t location_count;
  // Structured category describing the first blocking constraint found.
  loom_low_allocation_failure_blocking_kind_t blocking_kind;
  // Assignment index for an active-assignment conflict, or UINT32_MAX.
  uint32_t conflict_assignment_index;
  // SSA value occupying the conflicting assignment, or LOOM_VALUE_ID_INVALID.
  loom_value_id_t conflict_value_id;
  // Program point where the conflicting assignment starts, or UINT32_MAX.
  uint32_t conflict_start_point;
  // One-past-last storage program point for the conflicting assignment, or
  // UINT32_MAX.
  uint32_t conflict_end_point;
  // Conflicting assignment location kind.
  loom_low_allocation_location_kind_t conflict_location_kind;
  // Conflicting assignment base physical register or target ID, or UINT32_MAX.
  uint32_t conflict_location_base;
  // Conflicting assignment location width, or zero when unavailable.
  uint32_t conflict_location_count;
} loom_low_allocation_failure_t;

// Returns true when |failure| describes a terminal hard-allocation failure.
static inline bool loom_low_allocation_failure_is_present(
    const loom_low_allocation_failure_t* failure) {
  return failure != NULL && !iree_string_view_is_empty(failure->failure_code);
}
// Immutable interval-tree entry owned by the resolved target constraints.
typedef struct loom_low_allocation_fixed_interval_t
    loom_low_allocation_fixed_interval_t;

// Resolved target-owned constraints used while assigning concrete storage.
typedef struct loom_low_allocation_target_constraints_t {
  // Module containing the allocated low function.
  const loom_module_t* module;
  // Low function definition operation being allocated.
  const loom_op_t* function_op;
  // Resolved target selected by the low function.
  const loom_low_resolved_target_t* target;
  // Structured diagnostic emitter for allocation failures and feedback.
  iree_diagnostic_emitter_t emitter;
  // Number of error diagnostics emitted while resolving target constraints.
  uint32_t error_count;
  // Terminal allocation or move failure retained for table consumers.
  loom_low_allocation_failure_t failure;
  // Resolved explicit per-class register budgets.
  loom_low_allocation_resolved_budget_t* budgets;
  // Number of entries in |budgets|.
  iree_host_size_t budget_count;
  // Resolved fixed SSA value locations. Explicit preassignments occupy the
  // leading |preassigned_fixed_value_count| entries and are followed by
  // target-implied mandatory locations.
  loom_low_allocation_resolved_fixed_value_t* fixed_values;
  // Number of explicit fixed values that must be preassigned.
  iree_host_size_t preassigned_fixed_value_count;
  // Total number of entries in |fixed_values|.
  iree_host_size_t fixed_value_count;
  // Invocation-local indexes over the immutable resolved fixed assignments.
  struct {
    // Acquired domain retained by the owning allocation frame.
    const loom_local_value_domain_t* value_domain;
    // One-based fixed-value indices by local ordinal; zero denotes no entry.
    uint32_t* indices_by_ordinal;
    // Balanced interval tree in start-point order, with implicit child ranges.
    loom_low_allocation_fixed_interval_t* intervals;
  } fixed_index;
  // Resolved whole-function target-owned location ranges.
  loom_low_allocation_resolved_reserved_range_t* reserved_ranges;
  // Number of entries in |reserved_ranges|.
  iree_host_size_t reserved_range_count;
  // Maximum allocated value or move-scratch location end indexed by
  // descriptor register class ID.
  uint32_t* max_assigned_location_end_by_reg_class;
} loom_low_allocation_target_constraints_t;

// Resolves budgets and reserved ranges and initializes assignment search
// bounds for |target|.
iree_status_t loom_low_allocation_target_constraints_initialize(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_low_resolved_target_t* target,
    const loom_low_allocation_budget_t* budgets, iree_host_size_t budget_count,
    const loom_low_allocation_reserved_range_t* reserved_ranges,
    iree_host_size_t reserved_range_count, iree_diagnostic_emitter_t emitter,
    iree_arena_allocator_t* arena,
    loom_low_allocation_target_constraints_t* out_constraints);

// Resolves fixed values into complete assignments after |unit_liveness| has
// been initialized and propagated for |liveness|.
iree_status_t loom_low_allocation_target_constraints_resolve_fixed_values(
    loom_low_allocation_target_constraints_t* constraints,
    const loom_liveness_analysis_t* liveness,
    const loom_local_value_domain_t* value_domain,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_allocation_fixed_value_t* fixed_values,
    iree_host_size_t fixed_value_count, iree_arena_allocator_t* arena);

// Resolves |value_class| to a descriptor-set-local register class.
iree_status_t loom_low_allocation_target_constraints_resolve_reg_class(
    const loom_low_allocation_target_constraints_t* constraints,
    loom_liveness_value_class_t value_class, uint16_t* out_reg_class_id,
    const loom_low_reg_class_t** out_reg_class);

// Returns the effective capacity for |reg_class_id|.
iree_status_t loom_low_allocation_target_constraints_reg_class_capacity(
    const loom_low_allocation_target_constraints_t* constraints,
    uint16_t reg_class_id, loom_low_allocation_class_capacity_t* out_capacity);

// Resolves |value_class| and returns its effective capacity.
iree_status_t loom_low_allocation_target_constraints_class_capacity(
    const loom_low_allocation_target_constraints_t* constraints,
    loom_liveness_value_class_t value_class,
    loom_low_allocation_class_capacity_t* out_capacity);

// Resolves |interval|'s value class and applies descriptor-result placement
// windows imposed by the interval's defining packet.
iree_status_t loom_low_allocation_target_constraints_interval_capacity(
    const loom_low_allocation_target_constraints_t* constraints,
    const loom_liveness_interval_t* interval,
    loom_low_allocation_class_capacity_t* out_capacity);

// Returns true when |location_*| fits inside |capacity|.
bool loom_low_allocation_target_constraints_location_range_fits_capacity(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_class_capacity_t* capacity,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count);

// Validates that a register-like location range is legal for |reg_class_id|.
iree_status_t
loom_low_allocation_target_constraints_validate_register_location_capacity(
    loom_low_allocation_target_constraints_t* constraints,
    uint16_t reg_class_id, loom_low_allocation_location_kind_t location_kind,
    uint32_t location_base, uint32_t location_count,
    iree_string_view_t request_kind, const loom_op_t* diagnostic_op,
    bool* out_valid);

// Records a terminal move-capacity failure without emitting diagnostics. The
// completed allocation table owns the diagnostic publication boundary.
void loom_low_allocation_target_constraints_record_move_failure(
    loom_low_allocation_target_constraints_t* constraints, const loom_op_t* op,
    loom_liveness_value_class_t value_class, uint32_t budget_units,
    uint32_t peak_units, iree_string_view_t failure_code);

// Returns the fixed value record for |value_id|, or NULL when the value is not
// fixed.
const loom_low_allocation_resolved_fixed_value_t*
loom_low_allocation_target_constraints_fixed_value_for_value(
    const loom_low_allocation_target_constraints_t* constraints,
    loom_value_id_t value_id);

// Returns the explicit fixed value that must be preassigned for |value_id|, or
// NULL when the value has no explicit fixed-location request.
const loom_low_allocation_resolved_fixed_value_t*
loom_low_allocation_target_constraints_preassigned_fixed_value_for_value(
    const loom_low_allocation_target_constraints_t* constraints,
    loom_value_id_t value_id);

// Includes a concrete target-visible location in the per-class physical
// extent. Fixed architectural register windows do not contribute.
void loom_low_allocation_target_constraints_record_location_extent(
    loom_low_allocation_target_constraints_t* constraints,
    uint16_t descriptor_reg_class_id,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count);

// Rebuilds the per-class assignment bounds from |assignments| after their
// concrete locations change.
void loom_low_allocation_target_constraints_rebuild_assignment_location_ends(
    loom_low_allocation_target_constraints_t* constraints,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count);

// Returns the exclusive upper search bound implied by assigned, fixed, and
// reserved storage for |reg_class_id| and |location_kind|.
uint32_t loom_low_allocation_target_constraints_assigned_location_search_limit(
    const loom_low_allocation_target_constraints_t* constraints,
    uint16_t reg_class_id, loom_low_allocation_location_kind_t location_kind);

// Returns true when |candidate| conflicts with a fixed value. Whole-value hard
// ties in |placement| share storage even before either interval is assigned.
bool loom_low_allocation_target_constraints_fixed_value_conflicts(
    const loom_low_allocation_target_constraints_t* constraints,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_placement_table_t* placement,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count);

// Returns true when the location range conflicts with a reserved range.
bool loom_low_allocation_target_constraints_reserved_range_conflicts(
    const loom_low_allocation_target_constraints_t* constraints,
    uint16_t reg_class_id, loom_low_allocation_location_kind_t location_kind,
    uint32_t location_base, uint32_t location_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_TARGET_CONSTRAINTS_H_
