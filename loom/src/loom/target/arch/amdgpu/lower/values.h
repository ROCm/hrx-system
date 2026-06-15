// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for scalar and vector value-construction source operations.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUES_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUES_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects a plan for value-construction source ops.
iree_status_t loom_amdgpu_select_value_plan(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_low_lower_plan_t* out_plan);

// Preselects target-owned value-construction combines that must claim their
// producer before generated source-lowering contracts see it.
iree_status_t loom_amdgpu_preselect_value_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

// Returns true when value lowering may preselect |source_op| before generated
// source-lowering contracts see it.
bool loom_amdgpu_value_plan_needs_preselection(const loom_op_t* source_op);

// Lowers a value-construction source op using its selected plan.
iree_status_t loom_amdgpu_lower_value_op(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t plan);

// Marks the exact source values consumed by a selected value plan.
void loom_amdgpu_mark_value_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan);

// Verifies AMDGPU low legality for vector coordinate construction source ops.
iree_status_t loom_amdgpu_low_legality_verify_vector_iota(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies AMDGPU low legality for full-width offset arithmetic source ops.
iree_status_t loom_amdgpu_low_legality_verify_offset_add(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies AMDGPU low legality for full-width offset compare source ops.
iree_status_t loom_amdgpu_low_legality_verify_offset_compare(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies AMDGPU low legality for i64 scalar compares owned by value lowering.
iree_status_t loom_amdgpu_low_legality_verify_scalar_cmpi_i64(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies AMDGPU low legality for scalar conversions owned by value lowering.
iree_status_t loom_amdgpu_low_legality_verify_scalar_conversion(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUES_H_
