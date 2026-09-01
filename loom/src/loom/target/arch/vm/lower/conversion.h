// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core VM scalar and static-vector conversion lowering.

#ifndef LOOM_TARGET_ARCH_VM_LOWER_CONVERSION_H_
#define LOOM_TARGET_ARCH_VM_LOWER_CONVERSION_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects an exact Core VM lowering for a scalar or static-vector conversion.
bool loom_vm_conversion_try_select_op(const loom_module_t* module,
                                      const loom_op_t* source_op,
                                      loom_low_lower_plan_t* out_plan);

// Emits one selected scalar conversion or its lane-wise vector form.
//
// |out_handled| is true when |plan| belongs to this lowering.
iree_status_t loom_vm_conversion_emit_op(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t plan,
                                         bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_LOWER_CONVERSION_H_
