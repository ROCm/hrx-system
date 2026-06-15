// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for kernel async source operations.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_ASYNC_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_ASYNC_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Verifies source kernel.async group/wait legality for AMDGPU target-low
// selection.
iree_status_t loom_amdgpu_low_legality_verify_kernel_async(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects an AMDGPU async gather packet for a source kernel.async.gather op.
iree_status_t loom_amdgpu_select_kernel_async_gather_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_async_gather_plan_t* out_plan, bool* out_selected);

// Selects an AMDGPU wait packet for a source kernel.async.wait op.
iree_status_t loom_amdgpu_select_kernel_async_wait_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_async_wait_plan_t* out_plan, bool* out_selected);

// Lowers a source kernel.async.gather to a global-to-LDS packet and elides its
// async token.
iree_status_t loom_amdgpu_lower_kernel_async_gather(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_async_gather_plan_t* plan);

// Lowers a source kernel.async.wait to an explicit wait packet when the source
// group contains target-visible async transfers.
iree_status_t loom_amdgpu_lower_kernel_async_wait(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_async_wait_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_ASYNC_H_
