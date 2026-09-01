// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source module selection for source-to-low lowering.
//
// This is once-per-module JIT compilation setup. It resolves each source
// symbol's target binding, checks that the low lowering policy supports the
// resulting target contract, and returns the concrete inputs needed by the
// core lowerer. Specialized functions use their compiler-owned
// function-version facts; unrefined functions use facts projected from their
// authored target witness.

#ifndef LOOM_CODEGEN_LOW_LOWER_SOURCE_SELECTION_H_
#define LOOM_CODEGEN_LOW_LOWER_SOURCE_SELECTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/target/function_version.h"
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

  // Concrete compiler function versions participating in this lowering.
  const loom_function_version_list_t* function_versions;

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

  // Mutable identity for the target-refined compiler version, or NULL when
  // unrefined. The target facts reachable through this selection are immutable.
  loom_function_version_t* version_handle;

  // Position of |version_handle| in the lowering function-version list.
  loom_function_version_ordinal_t version_ordinal;

  // Whether target facts came from authorship alone or specialization.
  loom_target_binding_source_t target_source;

  // Authored module-local target record symbol referenced by |func|, or an
  // invalid ref when a targetless function was refined by the invocation.
  loom_symbol_ref_t target_ref;

  // Borrowed immutable function target facts for |func|.
  const loom_target_facts_t* target_facts;

  // Borrowed module symbol name for |target_ref|, or empty when targetless.
  iree_string_view_t target_symbol_name;

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

  // Lowering policy selected by |target_facts|.
  const loom_low_lower_policy_t* policy;
} loom_low_source_selection_t;

// Returns the common target bundle projected into |selection|'s facts.
static inline const loom_target_bundle_t*
loom_low_source_selection_target_bundle(
    const loom_low_source_selection_t* selection) {
  return loom_target_facts_bundle(selection->target_facts);
}

typedef struct loom_low_source_selection_list_t {
  // Source func-like symbols selected for lowering.
  loom_low_source_selection_t* values;

  // Number of source selections in |values|.
  iree_host_size_t count;
} loom_low_source_selection_list_t;

// Invokes each distinct target-selected policy's source module preparation
// callback once in target-record order.
//
// Preparation occurs before source symbol selection because callbacks may add
// ordinary source functions that must participate in lowering. The returned
// result is valid and unchanged when no target record selects a policy with a
// preparation callback.
iree_status_t loom_low_prepare_source_module(
    loom_module_t* module, const loom_low_source_selection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_lower_prepare_module_result_t* out_result);

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

// Selects all target-bound func bodies compatible with the injected target-low
// registries, including bodies already authored or lowered in Low IR.
//
// The returned selection array is allocated from |arena| and remains valid for
// the arena lifetime. A module with no compatible funcs succeeds with an empty
// list so module passes can be no-ops.
iree_status_t loom_low_select_target_bound_funcs(
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
