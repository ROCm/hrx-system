// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Concrete per-unit target storage locations.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_UNIT_LOCATION_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_UNIT_LOCATION_H_

#include "iree/base/api.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_allocation_unit_location_t {
  // Target-visible storage kind.
  loom_low_allocation_location_kind_t location_kind;
  // Storage class for the unit.
  loom_liveness_value_class_t value_class;
  // Descriptor-set-local register class ID for |value_class|.
  uint16_t descriptor_reg_class_id;
  // Physical register, target ID, or spill slot ordinal.
  uint32_t location;
} loom_low_allocation_unit_location_t;

// Returns the concrete storage location for one unit of |assignment|.
iree_status_t loom_low_allocation_assignment_unit_location(
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_index,
    loom_low_allocation_unit_location_t* out_location);

// Returns true when two unit locations name the same concrete unit.
bool loom_low_allocation_unit_locations_equal(
    const loom_low_allocation_unit_location_t* lhs,
    const loom_low_allocation_unit_location_t* rhs);

// Returns true when two unit locations are in the same allocation storage
// class. This ignores the concrete unit ordinal.
bool loom_low_allocation_unit_storage_classes_equal(
    const loom_low_allocation_unit_location_t* lhs,
    const loom_low_allocation_unit_location_t* rhs);

// Returns true when |source| to |destination| requires a target-register move.
bool loom_low_allocation_unit_locations_form_register_move(
    const loom_low_allocation_unit_location_t* source,
    const loom_low_allocation_unit_location_t* destination);

// Returns true when |location| is occupied by any assignment at |point|.
bool loom_low_allocation_unit_location_is_live_at_point(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, const uint32_t* unit_end_points,
    iree_host_size_t unit_end_point_count,
    const loom_low_allocation_unit_location_t* location, uint32_t point);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_UNIT_LOCATION_H_
