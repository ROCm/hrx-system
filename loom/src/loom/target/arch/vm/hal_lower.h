// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_VM_HAL_LOWER_H_
#define LOOM_TARGET_ARCH_VM_HAL_LOWER_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Projects one verified HAL source operation into a static VM call template.
// The common lowering driver owns traversal and physical value placement.
iree_status_t loom_vm_hal_select(loom_low_lower_context_t* context,
                                 const loom_op_t* source_op,
                                 loom_low_lower_plan_t* out_plan);

// Emits a native declaration and call from the retained projection. No source
// shape analysis or target legality decisions occur during emission.
iree_status_t loom_vm_hal_emit(loom_low_lower_context_t* context,
                               const loom_op_t* source_op,
                               loom_low_lower_plan_t plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_HAL_LOWER_H_
