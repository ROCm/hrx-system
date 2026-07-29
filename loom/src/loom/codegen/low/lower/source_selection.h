// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source module selection for source-to-low lowering.
//
// This is cold compilation setup: it resolves each source symbol's effective
// target, checks that the low lowering policy supports the resulting target
// contract, and returns the concrete inputs needed by the core lowerer. Target
// identity always comes from the function's durable target record. Specialized
// functions may additionally carry provider-owned profile facts that are not
// representable in IR.

#ifndef LOOM_CODEGEN_LOW_LOWER_SOURCE_SELECTION_H_
#define LOOM_CODEGEN_LOW_LOWER_SOURCE_SELECTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/target/specialization.h"
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

  // Optional per-function supplemental profiles for specialized functions.
  const loom_target_specialization_context_t* specialization_context;

  // True to collect compatible different-topology target candidates.
  bool collect_target_candidates;
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

  // Borrowed source function symbol name.
  iree_string_view_t function_name;

  // Whether |target_ref| was authored or bound by specialization.
  loom_target_binding_source_t target_source;

  // Module-local target record symbol referenced by |func|.
  loom_symbol_ref_t target_ref;

  // Borrowed durable target record referenced by |target_ref|.
  const loom_op_t* target_op;

  // Borrowed module symbol name for |target_ref|.
  iree_string_view_t target_symbol_name;

  // Storage for the effective target bundle selected by |func|.
  loom_target_bundle_storage_t target_bundle_storage;

  // Effective target bundle selected by |func|.
  const loom_target_bundle_t* target_bundle;

  // Supplemental specialization profile for |func|, or NULL when the function
  // was not specialized in this invocation. Target identity remains entirely
  // defined by |target_ref| and |target_bundle|.
  const loom_target_profile_t* target_profile;

  // Number of compatible module target records with different topology.
  uint32_t candidate_target_count;

  // First compatible different-topology target symbol name, if any.
  iree_string_view_t candidate_target_symbol_name;

  // First compatible different-topology target bundle name, if any.
  iree_string_view_t candidate_target_bundle_name;

  // First compatible different-topology target snapshot name, if any.
  iree_string_view_t candidate_target_snapshot_name;

  // First compatible different-topology target config name, if any.
  iree_string_view_t candidate_target_config_name;

  // First compatible different-topology target fixed subgroup size, if any.
  uint32_t candidate_target_subgroup_size;

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

// Invokes each distinct selected policy's module finalizer once in first-use
// order. Policies without module finalizers are skipped.
iree_status_t loom_low_source_selection_finalize_policies(
    loom_module_t* module,
    const loom_low_source_selection_list_t* selection_list,
    loom_low_lower_module_state_t* module_state,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_SOURCE_SELECTION_H_
