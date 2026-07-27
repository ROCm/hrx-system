// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Private AMDGPU mask, compare, select, and clamp lowering contracts.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MASK_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MASK_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/amdgpu/lower/plan.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects the AMDGPU mask compare plan for a source vector.cmpi op.
iree_status_t loom_amdgpu_select_vector_cmpi_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_compare_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.cmpi op from its selected AMDGPU mask compare plan.
iree_status_t loom_amdgpu_lower_vector_cmpi(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_compare_plan_t* plan);

// Selects the AMDGPU mask compare plan for a source vector.cmpf op.
iree_status_t loom_amdgpu_select_vector_cmpf_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_compare_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.cmpf op from its selected AMDGPU mask compare plan.
iree_status_t loom_amdgpu_lower_vector_cmpf(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_compare_plan_t* plan);

// Selects the AMDGPU mask compare plan for a source scalar.cmpf op.
iree_status_t loom_amdgpu_select_scalar_cmpf_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_compare_plan_t* out_plan, bool* out_selected);

// Lowers a source scalar.cmpf op from its selected AMDGPU mask compare plan.
iree_status_t loom_amdgpu_lower_scalar_cmpf(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_compare_plan_t* plan);

// Selects the AMDGPU clamp plan for a source scalar.clampf op.
iree_status_t loom_amdgpu_select_scalar_clampf_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_clampf_plan_t* out_plan, bool* out_selected);

// Selects the AMDGPU clamp plan for a source vector.clampf op.
iree_status_t loom_amdgpu_select_vector_clampf_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_clampf_plan_t* out_plan, bool* out_selected);

// Lowers a source scalar.clampf or vector.clampf op from its selected AMDGPU
// plan.
iree_status_t loom_amdgpu_lower_clampf(loom_low_lower_context_t* context,
                                       const loom_op_t* source_op,
                                       const loom_amdgpu_clampf_plan_t* plan);

// Selects the AMDGPU vector.select plan using explicit SGPR-pair masks and b32
// cndmask packets.
iree_status_t loom_amdgpu_select_vector_select_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_select_plan_t* out_plan, bool* out_selected);

// Verifies AMDGPU low legality for vector.select packed payload forms.
iree_status_t loom_amdgpu_low_legality_verify_vector_select(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects an AMDGPU scf.select plan using either SCC-controlled scalar selects
// or explicit SGPR-pair masks and b32 cndmask packets.
iree_status_t loom_amdgpu_select_scf_select_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_select_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.select or scf.select op from its selected AMDGPU plan.
iree_status_t loom_amdgpu_lower_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_select_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MASK_H_
