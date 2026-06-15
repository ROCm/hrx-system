// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor-alias-aware allocation storage predicates.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_STORAGE_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_STORAGE_H_

#include "iree/base/api.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the allocation location kind used for values in |reg_class|.
loom_low_allocation_location_kind_t
loom_low_allocation_storage_reg_class_location_kind(
    const loom_low_reg_class_t* reg_class);

// Returns true when the two descriptor register classes address the same
// backing storage space. Classes share storage when they are the same class or
// when both opt into the same non-zero alias set.
bool loom_low_allocation_storage_reg_classes_share(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t lhs_reg_class_id,
    uint16_t rhs_reg_class_id);

// Returns true when two assignments name the same target-visible storage space
// under |descriptor_set|'s alias contracts.
bool loom_low_allocation_storage_assignment_classes_share(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs);

// Returns true when two assignments name the same non-empty target storage
// range under |descriptor_set|'s alias contracts.
bool loom_low_allocation_storage_assignment_ranges_equal(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs);

// Returns true when two assignments cover the same target-visible storage
// location under |descriptor_set|'s alias contracts. Unlike range equality,
// this mirrors allocation coalescing semantics and does not require a non-empty
// location range.
bool loom_low_allocation_storage_assignment_locations_share(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs);

// Returns true when two non-empty assignments overlap in target storage under
// |descriptor_set|'s alias contracts.
bool loom_low_allocation_storage_assignment_ranges_overlap(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs);

// Returns true when two same-length assignment subranges name the same target
// storage units under |descriptor_set|'s alias contracts.
bool loom_low_allocation_storage_assignment_subranges_equal(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* lhs, uint32_t lhs_start,
    const loom_low_allocation_assignment_t* rhs, uint32_t rhs_start,
    uint32_t unit_count);

// Returns true when two same-length assignment subranges overlap in target
// storage under |descriptor_set|'s alias contracts.
bool loom_low_allocation_storage_assignment_subranges_overlap(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* lhs, uint32_t lhs_start,
    const loom_low_allocation_assignment_t* rhs, uint32_t rhs_start,
    uint32_t unit_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_STORAGE_H_
