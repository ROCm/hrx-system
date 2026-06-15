// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for source vector dot operations.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_DOT_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_DOT_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects the AMDGPU dotf lowering topology from source fast-math flags and
// descriptor availability.
iree_status_t loom_amdgpu_select_vector_dotf_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_dotf_plan_t* out_plan, bool* out_selected);

// Emits an AMDGPU scalarized f32 dot product.
iree_status_t loom_amdgpu_lower_vector_dotf(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_dotf_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_DOT_H_
