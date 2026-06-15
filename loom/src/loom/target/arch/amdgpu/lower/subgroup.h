// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for subgroup collective source operations.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SUBGROUP_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SUBGROUP_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects a native AMDGPU cross-lane packet for a source subgroup shuffle.
iree_status_t loom_amdgpu_select_kernel_subgroup_shuffle_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_shuffle_plan_t* out_plan, bool* out_selected);

// Lowers a source subgroup shuffle using one DS bpermute per 32-bit payload
// register.
iree_status_t loom_amdgpu_lower_kernel_subgroup_shuffle(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_shuffle_plan_t* plan);

// Verifies source subgroup shuffle legality for native AMDGPU lowering.
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_shuffle(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects native AMDGPU cross-lane packets for a source subgroup reduce.
iree_status_t loom_amdgpu_select_kernel_subgroup_reduce_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_reduce_plan_t* out_plan, bool* out_selected);

// Lowers a source subgroup reduce using selected cross-lane tree steps and
// native VGPR combining packets.
iree_status_t loom_amdgpu_lower_kernel_subgroup_reduce(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan);

// Verifies source subgroup reduce legality for native AMDGPU lowering.
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_reduce(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects native AMDGPU cross-lane packets for a source subgroup scan.
iree_status_t loom_amdgpu_select_kernel_subgroup_scan_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_scan_plan_t* out_plan, bool* out_selected);

// Lowers a source subgroup scan using DS bpermute prefix steps, native VGPR
// combining packets, and per-step lane-bound masks.
iree_status_t loom_amdgpu_lower_kernel_subgroup_scan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_scan_plan_t* plan);

// Verifies source subgroup scan legality for native AMDGPU lowering.
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_scan(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects native AMDGPU EXEC-mask packets for source subgroup active.mask.
iree_status_t loom_amdgpu_select_kernel_subgroup_active_mask_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_active_mask_plan_t* out_plan, bool* out_selected);

// Lowers a source subgroup active.mask by reading EXEC.
iree_status_t loom_amdgpu_lower_kernel_subgroup_active_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_active_mask_plan_t* plan);

// Verifies source subgroup active.mask legality for native AMDGPU lowering.
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_active_mask(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects native AMDGPU EXEC-mask packets for source subgroup vote.ballot.
iree_status_t loom_amdgpu_select_kernel_subgroup_ballot_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_ballot_plan_t* out_plan, bool* out_selected);

// Lowers a source subgroup vote.ballot by exposing the native predicate mask.
iree_status_t loom_amdgpu_lower_kernel_subgroup_ballot(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_ballot_plan_t* plan);

// Verifies source subgroup vote.ballot legality for native AMDGPU lowering.
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_ballot(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects native AMDGPU SALU packets for source subgroup vote.any.
iree_status_t loom_amdgpu_select_kernel_subgroup_vote_any_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_vote_any_plan_t* out_plan, bool* out_selected);

// Lowers a source subgroup vote.any by comparing a predicate mask with zero.
iree_status_t loom_amdgpu_lower_kernel_subgroup_vote_any(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_vote_any_plan_t* plan);

// Verifies source subgroup vote.any legality for native AMDGPU lowering.
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_vote_any(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects native AMDGPU SALU packets for source subgroup vote.all.
iree_status_t loom_amdgpu_select_kernel_subgroup_vote_all_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_vote_all_plan_t* out_plan, bool* out_selected);

// Lowers a source subgroup vote.all by comparing predicate and EXEC masks.
iree_status_t loom_amdgpu_lower_kernel_subgroup_vote_all(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_vote_all_plan_t* plan);

// Verifies source subgroup vote.all legality for native AMDGPU lowering.
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_vote_all(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects a native AMDGPU cross-lane packet for a source subgroup broadcast.
iree_status_t loom_amdgpu_select_kernel_subgroup_broadcast_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_broadcast_plan_t* out_plan, bool* out_selected);

// Lowers a source subgroup broadcast using one DS bpermute per 32-bit payload
// register.
iree_status_t loom_amdgpu_lower_kernel_subgroup_broadcast(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_broadcast_plan_t* plan);

// Verifies source subgroup broadcast legality for native AMDGPU lowering.
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_broadcast(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects a native AMDGPU first-active lane read for a source subgroup
// broadcast.first.
iree_status_t loom_amdgpu_select_kernel_subgroup_broadcast_first_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_broadcast_first_plan_t* out_plan, bool* out_selected);

// Lowers a source subgroup broadcast.first using one V_READFIRSTLANE per
// 32-bit payload register.
iree_status_t loom_amdgpu_lower_kernel_subgroup_broadcast_first(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_broadcast_first_plan_t* plan);

// Verifies source subgroup broadcast.first legality for native AMDGPU lowering.
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_broadcast_first(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Rejects source subgroup match ops that require explicit target legalization
// before AMDGPU source-to-low packet selection.
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_match(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SUBGROUP_H_
