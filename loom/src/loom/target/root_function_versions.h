// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Stable compiler versions for explicit source roots.

#ifndef LOOM_TARGET_ROOT_FUNCTION_VERSIONS_H_
#define LOOM_TARGET_ROOT_FUNCTION_VERSIONS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/function_version.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_environment_t loom_target_environment_t;

// Resolves explicit source roots to stable compiler function versions.
//
// Existing specialized versions in |version_owner| are reused. A root with an
// exact authored target receives an authored version before mutable compiler
// passes begin. This lets later passes update the version's live function while
// callers retain direct identity for the requested root. A targetless root or a
// root authored against a target declaration must already have a specialized
// version.
//
// |root_symbol_ids| and |out_root_version_ordinals| contain |root_count|
// entries in the same caller-defined order. Duplicate roots resolve to the
// same version ordinal.
// Temporary indexing uses |scratch_arena|; created versions and their target
// facts use |version_owner|'s arena. Invalid function contracts emit structured
// diagnostics, increment |out_error_count|, and return OK.
iree_status_t loom_target_root_function_versions_prepare(
    const loom_target_environment_t* environment, loom_module_t* module,
    const loom_symbol_id_t* root_symbol_ids, iree_host_size_t root_count,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena,
    loom_function_version_owner_t* version_owner,
    loom_function_version_ordinal_t* out_root_version_ordinals,
    uint32_t* out_error_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ROOT_FUNCTION_VERSIONS_H_
