// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Assignment records and concrete location-range predicates.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_ASSIGNMENT_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_ASSIGNMENT_H_

#include "iree/base/api.h"
#include "loom/analysis/liveness.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_low_allocation_location_kind_e {
  // Location has not been assigned.
  LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED = 0,
  // Interval is assigned to target-visible physical registers.
  LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER = 1,
  // Interval is assigned to target-local IDs such as VM locals or SPIR-V IDs.
  LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID = 2,
  // Interval must be spilled into a stack, scratch, or private slot.
  LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT = 3,
} loom_low_allocation_location_kind_t;

// Assignment for one liveness interval.
typedef struct loom_low_allocation_assignment_t {
  // SSA value represented by this assignment.
  loom_value_id_t value_id;
  // Pressure/allocation class for |value_id|.
  loom_liveness_value_class_t value_class;
  // Descriptor-set-local register class ID for |value_class|.
  uint16_t descriptor_reg_class_id;
  // First storage program point covered by this assignment.
  uint32_t start_point;
  // One-past-last storage program point covered by this assignment.
  //
  // This may extend past semantic liveness for target-visible dead definitions:
  // even an unused result writes its destination register.
  uint32_t end_point;
  // Allocation units required by this interval.
  uint32_t unit_count;
  // Assigned location kind.
  loom_low_allocation_location_kind_t location_kind;
  // Base physical register, target ID, or spill slot ordinal.
  uint32_t location_base;
  // Number of contiguous units assigned at |location_base|.
  uint32_t location_count;
  // First per-unit end-point entry in the allocation table.
  uint32_t unit_end_point_start;
} loom_low_allocation_assignment_t;

// Returns true when |location_kind| is one of the defined allocation location
// kinds.
bool loom_low_allocation_location_kind_is_known(
    loom_low_allocation_location_kind_t location_kind);

// Returns the stable diagnostic/JSON spelling for |location_kind|.
iree_string_view_t loom_low_allocation_location_kind_name(
    loom_low_allocation_location_kind_t location_kind);

// Returns true when |location_kind| names target-visible register-like storage.
bool loom_low_allocation_location_kind_is_register_like(
    loom_low_allocation_location_kind_t location_kind);

// Returns true when two assignments have the same concrete location kind and
// descriptor register class. Alias-set-aware storage checks require a
// descriptor set and are performed by allocation storage helpers.
bool loom_low_allocation_assignment_storage_class_equal(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs);

// Returns true when |assignment| names a non-empty physical register range in
// |descriptor_reg_class_id|.
bool loom_low_allocation_assignment_is_physical_register_class(
    const loom_low_allocation_assignment_t* assignment,
    uint16_t descriptor_reg_class_id);

// Returns true when |assignment| names target-visible register-like storage.
bool loom_low_allocation_assignment_is_register_like(
    const loom_low_allocation_assignment_t* assignment);

// Returns true when |assignment| names a spill slot.
bool loom_low_allocation_assignment_is_spill_slot(
    const loom_low_allocation_assignment_t* assignment);

// Returns the exclusive end of |assignment|'s concrete location range.
bool loom_low_allocation_assignment_location_exclusive_end(
    const loom_low_allocation_assignment_t* assignment, uint64_t* out_end);

// Returns true when two assignments name the same non-empty concrete location
// range.
bool loom_low_allocation_assignment_location_range_equal(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs);

// Returns true when two non-empty assignments overlap in concrete location
// storage.
bool loom_low_allocation_assignment_location_ranges_overlap(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs);

// Returns true when two same-length assignment subranges name the same units.
bool loom_low_allocation_assignment_subranges_equal(
    const loom_low_allocation_assignment_t* lhs, uint32_t lhs_start,
    const loom_low_allocation_assignment_t* rhs, uint32_t rhs_start,
    uint32_t unit_count);

// Returns true when two same-length assignment subranges overlap in storage.
bool loom_low_allocation_assignment_subranges_overlap(
    const loom_low_allocation_assignment_t* lhs, uint32_t lhs_start,
    const loom_low_allocation_assignment_t* rhs, uint32_t rhs_start,
    uint32_t unit_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_ASSIGNMENT_H_
