// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for scalar integer bit-count operations.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_BIT_COUNT_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_BIT_COUNT_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects an AMDGPU scalar trailing-zero-count plan.
iree_status_t loom_amdgpu_select_scalar_cttz_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_scalar_cttz_plan_t* out_plan, bool* out_selected);

// Lowers an AMDGPU scalar trailing-zero-count plan.
iree_status_t loom_amdgpu_lower_scalar_cttz(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_cttz_plan_t* plan);

// Verifies AMDGPU low legality for scalar trailing-zero count.
iree_status_t loom_amdgpu_low_legality_verify_scalar_cttz(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_BIT_COUNT_H_
