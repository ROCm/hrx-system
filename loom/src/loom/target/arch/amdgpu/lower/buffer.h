// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for buffer construction source operations.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_BUFFER_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_BUFFER_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects a plan for buffer construction source ops.
iree_status_t loom_amdgpu_select_buffer_plan(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t* out_plan);

// Records buffer construction analysis state needed by later target-low
// legality providers. The callback never claims the op as handled; generic
// target-low legality still owns whether the op is accepted or rejected.
iree_status_t loom_amdgpu_low_legality_record_buffer_op(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Lowers a buffer construction source op using its selected plan.
iree_status_t loom_amdgpu_lower_buffer_op(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_low_lower_plan_t plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_BUFFER_H_
