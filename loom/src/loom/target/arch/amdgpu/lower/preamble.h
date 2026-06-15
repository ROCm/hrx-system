// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for kernel preamble query operations.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_PREAMBLE_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_PREAMBLE_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects a plan for kernel preamble source ops.
iree_status_t loom_amdgpu_select_preamble_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

// Emits structural preamble live-ins for the current low function.
iree_status_t loom_amdgpu_emit_preamble(void* user_data,
                                        loom_low_lower_context_t* context);

// Emits entry-block setup packets that depend on structural ABI imports.
iree_status_t loom_amdgpu_emit_entry_setup(void* user_data,
                                           loom_low_lower_context_t* context);

// Lowers a kernel preamble source op using its pre-bound live-in value.
iree_status_t loom_amdgpu_lower_preamble_op(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op);

// Verifies AMDGPU low legality for launch preamble query source ops.
iree_status_t loom_amdgpu_low_legality_verify_kernel_preamble(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_PREAMBLE_H_
