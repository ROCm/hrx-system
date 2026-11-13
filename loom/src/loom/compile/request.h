// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compiler entry, root, target, and format request resolution.

#ifndef LOOM_COMPILE_REQUEST_H_
#define LOOM_COMPILE_REQUEST_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/error/source.h"
#include "loom/ir/module.h"
#include "loom/target/entry_selection.h"
#include "loom/target/provider.h"
#include "loom/target/specialization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Entry category inferred from compile roots.
typedef enum loom_compile_entry_kind_e {
  LOOM_COMPILE_ENTRY_KIND_INVALID = 0,
  LOOM_COMPILE_ENTRY_KIND_KERNEL = 1,
  LOOM_COMPILE_ENTRY_KIND_COMMAND = 2,
  LOOM_COMPILE_ENTRY_KIND_MODULE = 3,
} loom_compile_entry_kind_t;

// Resolved entry category and roots for one compilation.
typedef struct loom_compile_entry_selection_t {
  // Entry category inferred from the selected roots.
  loom_compile_entry_kind_t kind;
  // Selected roots, preserving explicit order/duplicates. Derived roots own
  // their names in the caller arena. Empty selects the entire module.
  iree_string_view_list_t roots;
  // Common target family authored on selected kernel roots, or NULL.
  const loom_target_fact_type_t* target_fact_type;
  // Number of selected kernel roots without an authored target.
  iree_host_size_t untargeted_kernel_count;
} loom_compile_entry_selection_t;

// User constraints applied while resolving one compilation request.
typedef struct loom_compile_request_options_t {
  // Explicit root names, or an empty list to derive selection from the module.
  iree_string_view_list_t roots;
  // Optional exact artifact format.
  iree_string_view_t format;
  // Optional immutable target profile selected from the target environment.
  const loom_target_profile_t* target_profile;
  // Canonical root names to exclude after entry-category inference and before
  // specialization and materialization. Cannot be combined with |roots|.
  iree_string_view_list_t excluded_roots;
  // Structured diagnostic emitter for source-backed request errors.
  iree_diagnostic_emitter_t diagnostic_emitter;
} loom_compile_request_options_t;

// Fully resolved compile request borrowing immutable configured state.
typedef struct loom_compile_request_t {
  // Entry category and root selection.
  loom_compile_entry_selection_t selection;
  // Target-owned artifact emitter for the resolved kernel or module entries.
  const loom_target_emitter_t* target_emitter;
  // Explicit target selected by the caller, or NULL for authored targets.
  const loom_target_profile_t* target_profile;
} loom_compile_request_t;

// Ownership of the source module passed to compile request materialization.
typedef enum loom_compile_request_source_ownership_e {
  // The source is borrowed and materialization produces an independent module.
  LOOM_COMPILE_REQUEST_SOURCE_BORROWED = 0,
  // Ownership is transferred and materialization may reuse or replace it.
  LOOM_COMPILE_REQUEST_SOURCE_TRANSFERRED = 1,
} loom_compile_request_source_ownership_t;

// Resolves one homogeneous entry category and its compile roots, an optional
// explicit target profile, and a target emitter. Explicit roots are borrowed.
// Otherwise the module must expose at most one category of default entry;
// mixed categories require explicit roots. Command-program roots are rejected
// because they require the LoomC command-program transaction. Exclusions apply
// after inference and derived names are copied into |arena|. Emitter resolution
// never probes an emitter by compiling. An omitted format selects the target
// family's unique canonical kernel or module emitter. Source-backed request
// errors are emitted through |options| and counted in |out_error_count|;
// status is reserved for request misuse and infrastructure failures. A nonzero
// error count leaves |out_request| empty.
iree_status_t loom_compile_request_resolve(
    const loom_module_t* module, const loom_compile_request_options_t* options,
    const loom_target_environment_t* target_environment,
    iree_arena_allocator_t* arena, loom_compile_request_t* out_request,
    uint32_t* out_error_count);

// Materializes the roots selected by a resolved compile request and applies an
// explicit target at the compile boundary.
//
// Root materialization establishes the deployment ABI independently of check
// launches in the input module and excludes every unselected root. Kernel
// entries are specialized into standalone target-specific IR immediately.
// Module entries return per-function specialization requests for the caller's
// compile pipeline; the requests and their borrowed module names remain valid
// until |arena| or |*out_module| is released. Requests without an explicit
// target return an empty specialization list.
//
// A borrowed |source_module| is unchanged and materialization always links an
// independently owned output, including when the entire module is selected. A
// transferred source may be reused when no linking is required or replaced and
// freed. The caller owns |*out_module| whenever it is non-NULL, including on
// failure. Source storage referenced by |sources| must outlive the call and
// output module. Link and specialization rejection is counted in
// |out_error_count|; status represents infrastructure failures only.
iree_status_t loom_compile_request_materialize(
    const loom_compile_request_t* request,
    const loom_target_environment_t* target_environment,
    const loom_target_entry_options_t* entry_options,
    const loom_module_t* source_module,
    loom_compile_request_source_ownership_t source_ownership,
    loom_source_table_projection_t* sources, iree_arena_allocator_t* arena,
    iree_arena_block_pool_t* block_pool, loom_module_t** out_module,
    loom_target_specialization_request_list_t* out_target_specializations,
    uint32_t* out_error_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_COMPILE_REQUEST_H_
