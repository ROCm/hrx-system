// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for invocation-local vector transforms.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_VECTOR_TRANSFORM_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_VECTOR_TRANSFORM_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when the target can lower |source_op| without scalarizing its
// source vector in target legalization.
bool loom_amdgpu_vector_transform_can_lower(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_op_t* source_op);

// Selects an AMDGPU vector-transform plan.
iree_status_t loom_amdgpu_select_vector_transform_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_transform_plan_t* out_plan, bool* out_selected);

// Lowers an AMDGPU vector-transform plan.
iree_status_t loom_amdgpu_lower_vector_transform(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_transform_plan_t* plan);

// Returns the compile-report strategy key for |plan|.
iree_string_view_t loom_amdgpu_vector_transform_plan_key(
    const loom_amdgpu_vector_transform_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_VECTOR_TRANSFORM_H_
