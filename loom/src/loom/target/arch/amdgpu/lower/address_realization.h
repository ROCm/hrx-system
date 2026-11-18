// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_ADDRESS_REALIZATION_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_ADDRESS_REALIZATION_H_

#include "loom/codegen/low/lower/realization.h"
#include "loom/target/arch/amdgpu/lower/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Adds target-shaped address candidates to the shared placement owner. The
// complete VADDR can be shared by all packets of one memory operation.
iree_status_t loom_amdgpu_prepare_memory_address_realizations(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_plan_t* plan);
iree_status_t loom_amdgpu_prepare_fragment_address_realization(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_ADDRESS_REALIZATION_H_
