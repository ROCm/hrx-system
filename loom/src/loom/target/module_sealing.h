// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lossless projection of compiler target facts into ordinary module IR.

#ifndef LOOM_TARGET_MODULE_SEALING_H_
#define LOOM_TARGET_MODULE_SEALING_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/function_version.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates a standalone IR-only clone that preserves every target-refined
// function version in |function_versions| as an ordinary target definition and
// function target reference.
//
// Equal producer-owned target context ordinals share one definition. An exact
// authored target definition is projected directly into the clone; a context
// without an exact authored witness is materialized from its resolved target
// facts through the owning provider. Semantically equivalent but independently
// produced contexts remain distinct.
//
// The source module and function versions remain unchanged. A non-target
// function-version representation is rejected because this layer cannot prove
// that its compiler-owned semantics survived the clone. The source module and
// function versions must otherwise be verified compiler-owned state.
//
// The caller owns |*out_sealed_module| and must release it with
// loom_module_free().
iree_status_t loom_target_module_seal(
    const loom_module_t* source_module,
    const loom_function_version_list_t* function_versions,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_module_t** out_sealed_module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_MODULE_SEALING_H_
