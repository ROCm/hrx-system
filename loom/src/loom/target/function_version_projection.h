// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lossless projection of compiler-owned target function versions into module
// IR.

#ifndef LOOM_TARGET_FUNCTION_VERSION_PROJECTION_H_
#define LOOM_TARGET_FUNCTION_VERSION_PROJECTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/function_version.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Projects |source_module| into a standalone IR-only module that preserves
// every target-refined function version in |function_versions| as an ordinary
// target definition and function target reference.
//
// Equal producer-owned target context ordinals share one definition. An exact
// authored target definition is projected directly into the clone; when
// distinct exact authored definitions witness the same context, one is used by
// all its versions so Low call target identity survives serialization. A
// context without an exact authored witness is materialized from its resolved
// target facts through the owning provider. Semantically equivalent but
// independently produced contexts remain distinct.
//
// Source IDs are preserved by the complete module clone.
// The source module and function versions remain unchanged. A non-target
// function-version representation is rejected because this layer cannot prove
// that its compiler-owned semantics survived the clone. The source module and
// function versions must otherwise be verified compiler-owned state.
//
// The caller owns |*out_projected_module| and must release it with
// loom_module_free().
//
// When |out_versions| is non-NULL, it must be initialized and empty. The clone
// retains independent target facts, applied schedules, packet memory proofs,
// and loop execution proofs in that owner. It can outlive the source module and
// its compilation storage. NULL requests only the standalone IR projection
// used by serializers.
iree_status_t loom_target_function_versions_project_module(
    const loom_module_t* source_module,
    const loom_function_version_list_t* function_versions,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_function_version_owner_t* out_versions,
    loom_module_t** out_projected_module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_FUNCTION_VERSION_PROJECTION_H_
