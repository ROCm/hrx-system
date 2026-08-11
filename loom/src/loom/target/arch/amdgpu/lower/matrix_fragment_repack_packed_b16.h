// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU packed-B16 matrix fragment role repacking.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_REPACK_PACKED_B16_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_REPACK_PACKED_B16_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects an in-register packed-B16 result-to-RHS fragment transition.
bool loom_amdgpu_select_result_to_rhs_packed_b16_fragment_repack_plan(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_matrix_fragment_role_layout_t* source_role_layout,
    const loom_matrix_fragment_role_layout_t* result_role_layout,
    loom_amdgpu_fragment_repack_plan_t* plan);

// Lowers a packed-B16 result-to-RHS fragment transition.
iree_status_t loom_amdgpu_lower_result_to_rhs_packed_b16_fragment_repack(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_REPACK_PACKED_B16_H_
