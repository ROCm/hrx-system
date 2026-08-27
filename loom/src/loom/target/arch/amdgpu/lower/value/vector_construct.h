// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for vector value-construction source operations.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_VECTOR_CONSTRUCT_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_VECTOR_CONSTRUCT_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects an AMDGPU plan for vector.iota, vector.from_elements, vector.splat,
// or vector.insert.
iree_status_t loom_amdgpu_select_vector_construct_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

// Preselects FMA-mix vector construction that must claim producers before
// generated source-lowering contracts see them.
iree_status_t loom_amdgpu_preselect_vector_construct_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

// Lowers a selected AMDGPU vector-construction plan.
iree_status_t loom_amdgpu_lower_vector_construct_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan);

// Marks source values consumed by a selected vector-construction plan.
void loom_amdgpu_mark_vector_construct_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan);

// Verifies AMDGPU low legality for vector.iota.
iree_status_t loom_amdgpu_low_legality_verify_vector_iota(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies AMDGPU low legality for vector.from_elements.
iree_status_t loom_amdgpu_low_legality_verify_vector_from_elements(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_VECTOR_CONSTRUCT_H_
