// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core VM static-vector carrier lowering.

#ifndef LOOM_TARGET_ARCH_VM_LOWER_VECTOR_H_
#define LOOM_TARGET_ARCH_VM_LOWER_VECTOR_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Source legality for structural operations on static vector carriers.
extern const loom_target_low_legality_provider_t
    loom_vm_vector_low_legality_provider;

// Selects a structural static-vector carrier lowering when available.
bool loom_vm_vector_try_select_op(const loom_module_t* module,
                                  const loom_op_t* source_op,
                                  loom_low_lower_plan_t* out_plan);

// Emits one selected static-vector carrier operation.
//
// |out_handled| is true when |plan| belongs to this lowering.
iree_status_t loom_vm_vector_emit_op(loom_low_lower_context_t* context,
                                     const loom_op_t* source_op,
                                     loom_low_lower_plan_t plan,
                                     bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_LOWER_VECTOR_H_
