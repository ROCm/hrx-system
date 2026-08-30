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
#include "loom/codegen/low/placement.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the allocation location kind used for values in |reg_class|.
loom_low_allocation_location_kind_t
loom_low_allocation_storage_reg_class_location_kind(
    const loom_low_reg_class_t* reg_class);

// Returns true when |assignment| uses an explicit physical-register class.
bool loom_low_allocation_storage_assignment_uses_explicit_physical_register(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignment);

// Returns true when |physical_register_id| is an ordered |unit_count|-unit
// view of |descriptor_reg_class_id|. The first candidate ordinal identifies
// the direct candidate covering logical unit zero. The pressure extent is one
// past the highest direct candidate ordinal occupied by the view. Candidate
// ordinals define preference and pressure order, not linear-location
// alignment; the declared view itself defines legal unit grouping.
bool loom_low_allocation_storage_explicit_physical_register_view(
    const loom_low_descriptor_set_t* descriptor_set,
    uint16_t descriptor_reg_class_id, uint32_t physical_register_id,
    uint32_t unit_count, uint32_t* out_first_candidate_ordinal,
    uint32_t* out_pressure_extent);

// Resolves one logical unit of an explicit physical-register assignment to
// the direct class-candidate register that names it.
bool loom_low_allocation_storage_assignment_unit_physical_register(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_index,
    uint32_t* out_physical_register_id);

// Returns the number of atomic storage units used to index |assignment|.
// Linear assignments return |location_count|; explicit physical-register
// views return their arbitrary atomic-unit set size.
uint32_t loom_low_allocation_storage_assignment_atomic_unit_count(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignment);

// Resolves one assignment atomic unit to its storage namespace and location.
// |atomic_unit_ordinal| must be less than the count returned above.
void loom_low_allocation_storage_assignment_atomic_unit(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignment,
    uint32_t atomic_unit_ordinal, uint32_t* out_storage_key,
    uint32_t* out_location);

// Returns the class-local pressure extent of |assignment|. Linear assignments
// use their exclusive numeric end; explicit physical-register views use one
// past their highest occupied direct-candidate preference ordinal.
uint32_t loom_low_allocation_storage_assignment_pressure_extent(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignment);

// Returns true when the two descriptor register classes address the same
// backing storage space. Linear classes share when they are the same class or
// opt into the same non-zero alias set. All explicit physical-register classes
// share the descriptor set's global atomic-unit namespace.
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

// Returns true when assignments satisfy a concrete placement relation.
bool loom_low_allocation_storage_placement_relation_satisfied(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_placement_relation_t* relation,
    const loom_low_allocation_assignment_t* result_assignment,
    const loom_low_allocation_assignment_t* source_assignment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_STORAGE_H_
