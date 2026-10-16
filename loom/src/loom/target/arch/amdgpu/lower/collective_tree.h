// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU scan and reduction trees shared by subgroup and workgroup stages.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_COLLECTIVE_TREE_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_COLLECTIVE_TREE_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/collective_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolves optional DPP move and fused-combine capabilities once for a
// collective plan. Absent descriptors leave the corresponding field empty.
iree_status_t loom_amdgpu_resolve_collective_dpp_descriptors(
    loom_low_lower_context_t* context, loom_combining_kind_t kind,
    loom_amdgpu_subgroup_payload_kind_t payload_kind,
    loom_low_lower_resolved_descriptor_t* dpp_move,
    loom_low_lower_resolved_descriptor_t* dpp_combine);

// Selects a tree exchange strategy for a validated collective shape. The
// enclosing collective plan owns target lane reach and publication semantics.
// Descriptor capabilities are borrowed from the enclosing collective plan;
// this shape query performs no descriptor lookup or IR analysis.
loom_amdgpu_subgroup_reduce_crosslane_kind_t
loom_amdgpu_subgroup_reduce_choose_crosslane_kind(
    uint32_t active_lane_count,
    const loom_low_lower_resolved_descriptor_t* dpp_move,
    const loom_low_lower_resolved_descriptor_t* dpp_combine,
    const loom_low_lower_resolved_descriptor_t* permlanex16);

// Resolves the target's DPP and row-exchange capabilities and selects the
// exchange strategy for a subgroup reduction. The resolved descriptors may be
// reused to select another tree shape within the same collective.
iree_status_t loom_amdgpu_select_subgroup_reduce_crosslane_kind(
    loom_low_lower_context_t* context, uint32_t active_lane_count,
    loom_combining_kind_t kind,
    loom_amdgpu_subgroup_payload_kind_t payload_kind,
    loom_low_lower_resolved_descriptor_t* dpp_move,
    loom_low_lower_resolved_descriptor_t* dpp_combine,
    loom_low_lower_resolved_descriptor_t* permlanex16,
    loom_amdgpu_subgroup_reduce_crosslane_kind_t* out_crosslane_kind);

// Returns whether the target has a fused DPP combine or a DPP move for
// source collective legality checking.
bool loom_amdgpu_subgroup_reduce_dpp_row_descriptor_is_present(
    const loom_low_descriptor_set_t* descriptor_set, loom_combining_kind_t kind,
    loom_amdgpu_subgroup_payload_kind_t payload_kind);

// Combines two payload registers with the selected reduction descriptor.
iree_status_t loom_amdgpu_emit_subgroup_combine(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t lane_type, loom_value_id_t* out_result);

// Combines a DPP-permuted first operand with the unmodified second operand.
iree_status_t loom_amdgpu_emit_subgroup_dpp_combine_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, uint32_t dpp_ctrl, loom_type_t lane_type,
    loom_value_id_t* out_low_result);

// Selects the participating result or unchanged accumulator using a lane guard.
iree_status_t loom_amdgpu_emit_subgroup_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* select_descriptor,
    loom_value_id_t false_value, loom_value_id_t true_value,
    loom_value_id_t guard, loom_type_t lane_type, loom_value_id_t* out_result);

// Compares lane coordinates with the selected predicate descriptor.
iree_status_t loom_amdgpu_emit_subgroup_lane_compare(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t mask_type, loom_value_id_t* out_guard);

// Reduces each payload register across the selected participating lanes.
// A dynamic lane count selects the guarded down tree for a partial tail;
// otherwise the retained cross-lane strategy owns the exchange mechanics.
iree_status_t loom_amdgpu_emit_subgroup_reduce_tree(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan, loom_value_id_t lane_id,
    loom_type_t lane_type, loom_value_id_t dynamic_active_lane_count,
    loom_value_id_t* inout_registers);

// Combines 32-lane halves through SGPRs and broadcasts the completed payload
// for a selected wave64 scalar-broadcast reduction.
iree_status_t loom_amdgpu_emit_subgroup_reduce_scalar_broadcast(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan, loom_value_id_t lane_id,
    loom_type_t lane_type, loom_value_id_t* inout_registers);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_COLLECTIVE_TREE_H_
