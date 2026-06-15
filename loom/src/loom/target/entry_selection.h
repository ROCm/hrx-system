// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target entry selection and diagnostics.
//
// These utilities own the target-neutral mechanics shared by artifact-emission
// front doors: generic and low verification, func-entry selection, func-owned
// target record resolution, and diagnostic emission with source ranges. They do
// not run pass pipelines or emit artifacts.

#ifndef LOOM_TARGET_ENTRY_SELECTION_H_
#define LOOM_TARGET_ENTRY_SELECTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/ir.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/types.h"
#include "loom/verify/verify.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_entry_options_t {
  // Optional func symbol to select. Empty requires exactly one compatible func
  // with a target record. A leading '@' is accepted for command-line
  // ergonomics.
  iree_string_view_t entry_symbol;
  // Diagnostic sink used for verification, lowering, scheduling, and
  // allocation diagnostics. A NULL callback still counts diagnostics.
  loom_diagnostic_sink_t diagnostic_sink;
  // Source resolver used to render carets for op-backed diagnostics.
  loom_source_resolver_t source_resolver;
  // Maximum diagnostics to emit before the active subsystem stops walking.
  // Zero lets callers use a backend-defined default.
  uint32_t max_errors;
  // Optional device- or environment-selected bundle used after the entry's
  // module target record has passed compatibility checks. Function-local ABI
  // contracts are re-applied to this bundle before the entry is returned.
  const loom_target_bundle_t* effective_target_bundle;
} loom_target_entry_options_t;

typedef struct loom_target_entry_t {
  // Source or low entry func selected for artifact emission.
  loom_func_like_t func;
  // Borrowed selected func symbol name.
  iree_string_view_t func_name;
  // Module-local symbol reference for |func|.
  loom_symbol_ref_t func_ref;
  // Module-local target record symbol referenced by |func|.
  loom_symbol_ref_t target_ref;
  // Borrowed target record symbol entry referenced by |target_ref|.
  const loom_symbol_t* target_symbol;
  // Borrowed target record op referenced by |target_ref|.
  loom_op_t* target_op;
  // Materialized target bundle selected by |func|. The export plan is the
  // func-owned effective export plan, not a shared target-record backreference.
  loom_target_bundle_storage_t bundle_storage;
} loom_target_entry_t;

typedef struct loom_target_entry_list_t {
  // Arena-owned entry descriptors in module order.
  loom_target_entry_t* values;
  // Number of entries in |values|.
  uint16_t count;
} loom_target_entry_list_t;

typedef struct loom_target_entry_diagnostic_emitter_t {
  // Module containing op locations referenced by emitted diagnostics.
  const loom_module_t* module;
  // Source resolver for original source-backed locations.
  loom_source_resolver_t source_resolver;
  // Final diagnostic sink that owns rendering or capture policy.
  loom_diagnostic_sink_t diagnostic_sink;
  // Subsystem identity stored in materialized diagnostics.
  loom_emitter_t emitter;
  // Error diagnostics emitted through this materializer.
  uint32_t error_count;
  // Warning diagnostics emitted through this materializer.
  uint32_t warning_count;
  // Remark diagnostics emitted through this materializer.
  uint32_t remark_count;
} loom_target_entry_diagnostic_emitter_t;

typedef bool(IREE_API_PTR* loom_target_entry_predicate_fn_t)(
    void* user_data, const loom_target_entry_t* entry);

typedef struct loom_target_entry_predicate_t {
  // Callback invoked to test whether one target-resolved entry is compatible
  // with the concrete backend emission path.
  loom_target_entry_predicate_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_target_entry_predicate_t;

// Returns |options->max_errors| when present, otherwise |default_max_errors|.
uint32_t loom_target_entry_max_errors(
    const loom_target_entry_options_t* options, uint32_t default_max_errors);

// Returns the normalized symbol name without surrounding whitespace or a
// leading '@'.
iree_string_view_t loom_target_entry_normalize_symbol_name(
    iree_string_view_t symbol_name);

// Returns the normalized entry symbol name without a leading '@'.
iree_string_view_t loom_target_entry_symbol_name(
    const loom_target_entry_options_t* options);

// Initializes a diagnostic emitter for target subsystems.
void loom_target_entry_diagnostic_emitter_initialize(
    const loom_module_t* module, const loom_target_entry_options_t* options,
    loom_emitter_t emitter,
    loom_target_entry_diagnostic_emitter_t* out_emitter);

// Returns a callback wrapper for |emitter|.
iree_diagnostic_emitter_t loom_target_entry_emitter(
    loom_target_entry_diagnostic_emitter_t* emitter);

// Runs generic module verification. Status is reserved for infrastructure
// failures; verification errors are reported through |out_result|.
iree_status_t loom_target_entry_verify_module(
    const loom_module_t* module, const loom_target_entry_options_t* options,
    uint32_t default_max_errors, loom_verify_result_t* out_result);

// Runs target-low semantic verification. Status is reserved for infrastructure
// failures; verification errors are reported through |out_result|.
iree_status_t loom_target_entry_verify_low_module(
    const loom_module_t* module,
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    loom_target_selection_t target_selection, uint32_t max_errors,
    loom_low_verify_provider_list_t low_verify_provider_list,
    loom_low_verify_scratch_t* scratch, loom_low_verify_result_t* out_result);

// Selects either the named func entry or the only compatible func entry
// according to |predicate|.
iree_status_t loom_target_entry_select_entry(
    const loom_module_t* module, const loom_target_entry_options_t* options,
    loom_target_entry_predicate_t predicate,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    iree_string_view_t entry_kind, iree_arena_allocator_t* arena,
    bool* out_selected, loom_target_entry_t* out_entry);

// Selects every exported compatible func entry according to |predicate| in
// top-level module operation order.
iree_status_t loom_target_entry_select_all_entries(
    const loom_module_t* module, const loom_target_entry_options_t* options,
    loom_target_entry_predicate_t predicate,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    iree_string_view_t entry_kind, iree_arena_allocator_t* arena,
    bool* out_selected, loom_target_entry_list_t* out_entries);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ENTRY_SELECTION_H_
