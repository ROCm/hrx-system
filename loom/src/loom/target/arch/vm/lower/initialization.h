// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical VM module initialization boundary materialization.

#ifndef LOOM_TARGET_ARCH_VM_LOWER_INITIALIZATION_H_
#define LOOM_TARGET_ARCH_VM_LOWER_INITIALIZATION_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Canonicalizes or synthesizes the executable VM initializer function.
iree_status_t loom_vm_materialize_initializer(
    loom_module_t* module, iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena,
    loom_low_lower_prepare_module_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_LOWER_INITIALIZATION_H_
