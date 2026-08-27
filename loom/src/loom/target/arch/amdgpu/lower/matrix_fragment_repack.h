// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for vector fragment role and layout repacks.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_REPACK_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_REPACK_H_

#include "loom/target/arch/amdgpu/lower/matrix_fragment_state.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects an AMDGPU matrix-fragment repack plan.
iree_status_t loom_amdgpu_select_vector_fragment_repack_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_repack_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.fragment.repack op.
iree_status_t loom_amdgpu_lower_vector_fragment_repack(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan);

// Returns the compile-report plan key for a selected fragment repack plan.
iree_string_view_t loom_amdgpu_fragment_repack_plan_key(
    const loom_amdgpu_fragment_repack_plan_t* plan);

// Returns true when lowering the selected plan derives fragment coordinates
// from the current subgroup lane ID.
bool loom_amdgpu_fragment_repack_plan_requires_lane_id(
    const loom_amdgpu_fragment_repack_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_REPACK_H_
