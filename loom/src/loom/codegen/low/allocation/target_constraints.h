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
  // Stable register-class name such as "vm.i32" or "amdgpu.vgpr".
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
  // Stable register-class name such as "amdgpu.sgpr" or "x86.gpr".
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

typedef struct loom_low_allocation_resolved_fixed_value_t {
  // SSA value forced to the fixed location.
  loom_value_id_t value_id;
  // Liveness-local ordinal for |value_id|.
  loom_value_ordinal_t value_ordinal;
  // Descriptor-set-local register class ID for |value_id|.
  uint16_t descriptor_reg_class_id;
  // Live interval for |value_id|.
  const loom_liveness_interval_t* interval;
  // Target-visible fixed location kind.
  loom_low_allocation_location_kind_t location_kind;
  // Base physical register or target ID.
  uint32_t location_base;
  // Number of contiguous units fixed at |location_base|.
  uint32_t location_count;
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
  // Resolved explicit per-class register budgets.
  loom_low_allocation_resolved_budget_t* budgets;
  // Number of entries in |budgets|.
  iree_host_size_t budget_count;
  // Resolved fixed SSA value locations.
  loom_low_allocation_resolved_fixed_value_t* fixed_values;
  // Number of entries in |fixed_values|.
  iree_host_size_t fixed_value_count;
  // Resolved whole-function target-owned location ranges.
  loom_low_allocation_resolved_reserved_range_t* reserved_ranges;
  // Number of entries in |reserved_ranges|.
  iree_host_size_t reserved_range_count;
  // Maximum assigned location end indexed by descriptor register class ID.
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

// Resolves fixed values against |liveness| and |value_domain|.
iree_status_t loom_low_allocation_target_constraints_resolve_fixed_values(
    loom_low_allocation_target_constraints_t* constraints,
    const loom_liveness_analysis_t* liveness,
    const loom_local_value_domain_t* value_domain,
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

// Returns true when |location_*| fits inside |capacity|.
bool loom_low_allocation_target_constraints_location_range_fits_capacity(
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

// Emits a structured allocation-capacity failure for |value_class|.
iree_status_t loom_low_allocation_target_constraints_emit_failure(
    loom_low_allocation_target_constraints_t* constraints, const loom_op_t* op,
    loom_liveness_value_class_t value_class, uint32_t budget_units,
    uint32_t peak_units, iree_string_view_t failure_code);

// Returns the fixed value record for |value_id|, or NULL when the value is not
// fixed.
const loom_low_allocation_resolved_fixed_value_t*
loom_low_allocation_target_constraints_fixed_value_for_value(
    const loom_low_allocation_target_constraints_t* constraints,
    loom_value_id_t value_id);

// Records |assignment|'s concrete location in the per-class search bounds.
void loom_low_allocation_target_constraints_record_assignment_location_end(
    loom_low_allocation_target_constraints_t* constraints,
    const loom_low_allocation_assignment_t* assignment);

// Returns the exclusive upper search bound implied by assigned, fixed, and
// reserved storage for |reg_class_id| and |location_kind|.
uint32_t loom_low_allocation_target_constraints_assigned_location_search_limit(
    const loom_low_allocation_target_constraints_t* constraints,
    uint16_t reg_class_id, loom_low_allocation_location_kind_t location_kind);

// Returns true when |candidate| conflicts with a fixed value.
bool loom_low_allocation_target_constraints_fixed_value_conflicts(
    const loom_low_allocation_target_constraints_t* constraints,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
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
