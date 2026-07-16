// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU source-storage demand policy for selected value-lowering plans.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_STORAGE_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_STORAGE_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Marks the exact source values consumed by a selected value-lowering plan.
void loom_amdgpu_mark_value_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_STORAGE_H_
