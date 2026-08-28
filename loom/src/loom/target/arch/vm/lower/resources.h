// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core VM module-resource planning and source lowering.

#ifndef LOOM_TARGET_ARCH_VM_LOWER_RESOURCES_H_
#define LOOM_TARGET_ARCH_VM_LOWER_RESOURCES_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Tries to select a Core VM global or read-only data access lowering.
//
// Leaves |out_plan| empty when the op or referenced resource is not supported.
iree_status_t loom_vm_module_resource_try_select_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

// Emits a Core VM global or read-only data access selected above.
iree_status_t loom_vm_module_resource_emit_op(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_low_lower_plan_t plan,
                                              bool* out_handled);

// Materializes physical VM resource records from the module resource plan.
iree_status_t loom_vm_module_resources_finalize(
    loom_module_t* module, loom_low_lower_module_state_t* module_state,
    iree_arena_allocator_t* scratch_arena);

// Source legality provider for VM module-resource accesses.
extern const loom_target_low_legality_provider_t
    loom_vm_module_resource_low_legality_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_LOWER_RESOURCES_H_
