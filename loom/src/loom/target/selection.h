// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Invocation-selected target pass capability.
//
// Target selections are reusable target-profile facts. Some compiler phases
// also need a module-local target record symbol so target-like source
// predicates, provider selection, and produced target-low funcs can agree on
// the selected target without rewriting source root attrs. This capability
// carries both.

#ifndef LOOM_TARGET_SELECTION_H_
#define LOOM_TARGET_SELECTION_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/ir/module.h"
#include "loom/pass/environment.h"
#include "loom/pass/types.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Capability type for loom_target_pass_capability_t.
extern const loom_pass_environment_capability_type_t
    loom_target_pass_capability_type;

typedef struct loom_target_pass_capability_t {
  // Base capability header. Must remain the first field.
  loom_pass_environment_capability_t base;

  // Invocation-selected target bundle and target-owned payload.
  loom_target_selection_t target_selection;

  // Module-local target record materialized for |target_selection|, or null.
  loom_symbol_ref_t target_ref;
} loom_target_pass_capability_t;

// Creates a borrowed target pass capability.
loom_target_pass_capability_t loom_target_pass_capability_make(
    loom_target_selection_t target_selection, loom_symbol_ref_t target_ref);

// Looks up the target capability from |environment|. Returns NULL when absent.
const loom_target_pass_capability_t*
loom_target_pass_capability_from_environment(
    const loom_pass_environment_t* environment);

// Looks up the target capability from |pass->environment|. Returns NULL when
// absent.
const loom_target_pass_capability_t* loom_target_pass_capability_from_pass(
    const loom_pass_t* pass);

// Returns the invocation-selected target bundle/payload, or an empty selection.
loom_target_selection_t loom_target_pass_capability_target_selection(
    const loom_target_pass_capability_t* capability);

// Returns the module-local target record materialized for the invocation.
loom_symbol_ref_t loom_target_pass_capability_target_ref(
    const loom_target_pass_capability_t* capability);

// Returns |authored_target_ref| when present, otherwise the invocation target
// ref carried by |capability|.
loom_symbol_ref_t loom_target_effective_target_ref(
    loom_symbol_ref_t authored_target_ref,
    const loom_target_pass_capability_t* capability);

// Compacts module symbols while preserving the invocation target ref carried by
// |pass|, if any.
//
// Pass environments may carry a module-local target symbol ref across multiple
// passes. Ordinary module compaction rewrites refs inside the module but cannot
// discover that external pass-environment ref, so symbol transforms that
// compact after erasing ops should use this helper.
iree_status_t loom_target_pass_compact_symbols_preserving_target_ref(
    const loom_pass_t* pass, loom_module_t* module,
    iree_arena_allocator_t* scratch_arena, iree_host_size_t* out_removed_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_SELECTION_H_
