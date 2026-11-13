// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-bound function selection for Low lowering and legalization.
//
// This is once-per-module JIT compilation setup. Source selectors distinguish
// source function and kernel operations from existing Low IR before resolving
// target bindings. The broader target-bound selector lets legalization revisit
// both forms. Specialized functions use their compiler-owned function-version
// facts; unrefined functions use facts projected from their authored target
// witness.

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
  // while selecting functions.
  iree_diagnostic_emitter_t diagnostic_emitter;

  // Concrete compiler function versions participating in this lowering.
  const loom_function_version_list_t* function_versions;

  // True to collect compatible different-topology target candidates.
  bool collect_target_candidates;
} loom_low_source_selection_options_t;

typedef enum loom_low_source_selection_kind_e {
  // Non-Low function body selected for lowering or legalization.
  LOOM_LOW_SOURCE_SELECTION_FUNCTION = 1,
  // Source declaration selected for Low callable signature lowering.
  LOOM_LOW_SOURCE_SELECTION_DECLARATION = 2,
  // Existing Low definition or declaration for projection or legalization.
  LOOM_LOW_SOURCE_SELECTION_REPRESENTATION = 3,
} loom_low_source_selection_kind_t;

// Specialization evidence captured only for requested target reports.
typedef struct loom_low_source_selection_report_t {
  // Borrowed module symbol name for the authored target, or empty when
  // targetless.
  iree_string_view_t target_symbol_name;

  // Compatible different-topology targets, summarized in module order.
  struct {
    // Number of compatible target records.
    uint32_t count;
    // Borrowed symbol name of the first candidate, or empty when absent.
    iree_string_view_t symbol_name;
    // Immutable bundle of the first candidate, or NULL when absent.
    const loom_target_bundle_t* bundle;
  } candidates;
} loom_low_source_selection_report_t;

typedef struct loom_low_source_selection_t {
  // Selected symbol category.
  loom_low_source_selection_kind_t kind;

  // Selected func-like op.
  loom_func_like_t func;

  // Borrowed function symbol name.
  iree_string_view_t function_name;

  // Mutable identity for the target-refined compiler version, or NULL when
  // unrefined. The target facts reachable through this selection are immutable.
  loom_function_version_t* version_handle;

  // Whether target facts came from authorship alone or specialization.
  loom_target_binding_source_t target_source;

  // Authored module-local target record symbol referenced by |func|, or an
  // invalid ref when a targetless function was refined by the invocation.
  loom_symbol_ref_t target_ref;

  // Borrowed immutable function target facts for |func|.
  const loom_target_facts_t* target_facts;

  // Optional report-only specialization evidence, absent unless requested.
  const loom_low_source_selection_report_t* report;

  // Lowering policy selected by |target_facts|.
  const loom_low_lower_policy_t* policy;
} loom_low_source_selection_t;

// Returns the common target bundle projected into |selection|'s facts.
static inline const loom_target_bundle_t*
loom_low_source_selection_target_bundle(
    const loom_low_source_selection_t* selection) {
  return loom_target_facts_bundle(selection->target_facts);
}

// Initializes |scope| with the exact executable export selected for
// |selection| and returns its forwarding emitter. |scope| must outlive every
// use of the returned emitter.
iree_diagnostic_emitter_t loom_low_source_selection_scope_diagnostics(
    const loom_low_source_selection_t* selection,
    iree_diagnostic_emitter_t target, loom_diagnostic_export_scope_t* scope);

typedef struct loom_low_source_selection_list_t {
  // Selected func-like symbols.
  loom_low_source_selection_t* values;

  // Number of selections in |values|.
  iree_host_size_t count;
} loom_low_source_selection_list_t;

// Selects compatible source definitions/declarations and existing Low
// representation projections in one symbol-table traversal. All categories
// share one symbol-fact table and immutable function-version snapshot.
// Standalone source lowering binds authored targets for execution; existing
// Low functions and function versions retain their selected modes. The
// function-only selectors below preserve partial facts for legalization.
//
// The returned selection array is allocated from |arena| and remains valid for
// the arena lifetime. A module with no compatible symbols succeeds with an
// empty list so module passes can be no-ops.
iree_status_t loom_low_select_lowering_symbols(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_source_selection_list_t* out_selection_list);

// Selects all source function and kernel definitions compatible with the
// injected target-low registries.
//
// The returned selection array is allocated from |arena| and remains valid for
// the arena lifetime. A module with no compatible funcs succeeds with an empty
// list so module passes can be no-ops.
iree_status_t loom_low_select_source_funcs(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_source_selection_list_t* out_selection_list);

// Selects all target-bound function bodies compatible with the injected
// target-low registries, including bodies already authored or lowered in Low
// IR.
//
// The returned selection array is allocated from |arena| and remains valid for
// the arena lifetime. A module with no compatible funcs succeeds with an empty
// list so module passes can be no-ops.
iree_status_t loom_low_select_target_bound_funcs(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_source_selection_list_t* out_selection_list);

// Invokes each distinct source-lowering policy's module finalizer once in
// first-use order. Existing Low projections and policies without finalizers
// contribute no module resources and are skipped. Consumes the list's policy
// bindings after all source plans have executed, reusing those fields for the
// distinct-policy prefix. Other fields and the selection count are unchanged.
iree_status_t loom_low_source_selection_finalize_policies(
    loom_module_t* module, loom_low_source_selection_list_t* selection_list,
    loom_low_lower_module_state_t* module_state,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_SOURCE_SELECTION_H_
