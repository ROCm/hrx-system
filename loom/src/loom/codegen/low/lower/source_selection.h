// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source module selection for source-to-low lowering.
//
// This is cold compilation setup: it resolves each source symbol's effective
// target, checks that the low lowering policy supports the resulting target
// contract, and returns the concrete inputs needed by the core lowerer.
// Authored func target records win over invocation target selection; compatible
// invocation selections refine the target bundle and provide target-owned
// payloads without rewriting source attrs.

#ifndef LOOM_CODEGEN_LOW_LOWER_SOURCE_SELECTION_H_
#define LOOM_CODEGEN_LOW_LOWER_SOURCE_SELECTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_source_selection_options_t {
  // Target lowering policies linked into the caller.
  const loom_low_lower_policy_registry_t* policy_registry;

  // Structured diagnostic emitter for invalid target contracts encountered
  // while selecting source functions.
  iree_diagnostic_emitter_t diagnostic_emitter;

  // User-facing lowering kind used in diagnostics.
  iree_string_view_t lowering_kind;

  // Optional runtime-selected target overlay. When compatible with a source
  // function's module target record, this bundle refines the target contract
  // used by legality and lowering while preserving the module target symbol.
  loom_target_selection_t target_selection;

  // Module-local target record materialized for |target_selection|, or null.
  loom_symbol_ref_t target_ref;
} loom_low_source_selection_options_t;

typedef enum loom_low_source_selection_kind_e {
  // Target-bound function body selected for source-to-low lowering.
  LOOM_LOW_SOURCE_SELECTION_FUNCTION = 1,
  // Target-bound external declaration selected for low import declaration.
  LOOM_LOW_SOURCE_SELECTION_IMPORT_DECL = 2,
} loom_low_source_selection_kind_t;

typedef struct loom_low_source_selection_t {
  // Selected source symbol category.
  loom_low_source_selection_kind_t kind;

  // Source func-like op selected for lowering.
  loom_func_like_t func;

  // Module-local target record symbol referenced by |func|.
  loom_symbol_ref_t target_ref;

  // Storage for the effective target bundle selected by |func|.
  loom_target_bundle_storage_t target_bundle_storage;

  // Effective target bundle selected by |func|.
  const loom_target_bundle_t* target_bundle;

  // Target-owned payload associated with |target_bundle|, or NULL when the
  // bundle came only from module target records.
  const void* target_data;

  // Lowering policy selected by |target_bundle|.
  const loom_low_lower_policy_t* policy;
} loom_low_source_selection_t;

typedef struct loom_low_source_selection_list_t {
  // Source func-like symbols selected for lowering.
  loom_low_source_selection_t* values;

  // Number of source selections in |values|.
  iree_host_size_t count;
} loom_low_source_selection_list_t;

// Selects all source funcs and import declarations compatible with the injected
// target-low registries.
//
// The returned selection array is allocated from |arena| and remains valid for
// the arena lifetime. A module with no compatible symbols succeeds with an
// empty list so module passes can be no-ops.
iree_status_t loom_low_select_source_symbols(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_source_selection_list_t* out_selection_list);

// Selects all source funcs compatible with the injected target-low registries.
//
// The returned selection array is allocated from |arena| and remains valid for
// the arena lifetime. A module with no compatible funcs succeeds with an empty
// list so module passes can be no-ops.
iree_status_t loom_low_select_source_funcs(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_source_selection_list_t* out_selection_list);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_SOURCE_SELECTION_H_
