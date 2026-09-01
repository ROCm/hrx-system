// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core VM buffer-memory lowering.

#ifndef LOOM_TARGET_ARCH_VM_LOWER_MEMORY_H_
#define LOOM_TARGET_ARCH_VM_LOWER_MEMORY_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Source legality for Core VM buffer-backed memory accesses.
extern const loom_target_low_legality_provider_t
    loom_vm_memory_low_legality_provider;

// Selects a Core VM buffer-memory lowering when available.
iree_status_t loom_vm_memory_try_select_op(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_low_lower_plan_t* out_plan);

// Marks the exact source values retained by a selected memory plan.
//
// Returns true when |plan| belongs to this lowering.
bool loom_vm_memory_mark_plan_storage_demands(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_low_lower_plan_t plan);

// Emits one selected Core VM buffer-memory access.
//
// |out_handled| is true when |plan| belongs to this lowering.
iree_status_t loom_vm_memory_emit_op(loom_low_lower_context_t* context,
                                     const loom_op_t* source_op,
                                     loom_low_lower_plan_t plan,
                                     bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_LOWER_MEMORY_H_
