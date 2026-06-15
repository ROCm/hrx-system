// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V source-to-low lowering for Workgroup storage.

#ifndef LOOM_TARGET_ARCH_SPIRV_LOWER_WORKGROUP_H_
#define LOOM_TARGET_ARCH_SPIRV_LOWER_WORKGROUP_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects Workgroup storage plans for buffer allocation and typed views.
iree_status_t loom_spirv_select_workgroup_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

// Emits a previously selected Workgroup storage plan.
iree_status_t loom_spirv_lower_workgroup_op(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_low_lower_plan_t plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_LOWER_WORKGROUP_H_
