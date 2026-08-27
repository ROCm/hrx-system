// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Incremental materialized module linker for in-memory Loom IR.
//
// This is the inner link step for modules that have already been parsed,
// bytecode-decoded, or otherwise materialized into loom_module_t. Larger
// library workflows should discover reachable symbols through a module index,
// materialize only those modules/functions, and then stream those inputs
// through this linker. Each added module is cloned into the output immediately;
// callers may release a source module as soon as loom_linker_add_module()
// returns.

#ifndef LOOM_LINK_LINKER_H_
#define LOOM_LINK_LINKER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Options controlling one link operation.
typedef struct loom_link_options_t {
  // Name assigned to the linked output module.
  iree_string_view_t module_name;
  // Root symbol names to materialize. Function-like roots are retained in the
  // linked output. An empty list links every materialized source symbol.
  iree_string_view_list_t root_symbols;
} loom_link_options_t;

// Stateful incremental linker. Owns the output module until finish transfers it
// to the caller.
typedef struct loom_linker_t loom_linker_t;

// Options controlling linker construction.
typedef struct loom_linker_options_t {
  // Name assigned to the linked output module. Defaults to "linked".
  iree_string_view_t module_name;
  // Maximum number of target symbols carrying sparse-plan state.
  // Zero keeps merge/dense links allocation-free for this state.
  iree_host_size_t planned_symbol_capacity;
} loom_linker_options_t;

// Options controlling one input module add.
typedef struct loom_linker_add_options_t {
  // Reachability anchor names to materialize from this module. An empty list
  // links every materialized top-level symbol in the module. Anchors are not
  // output roots until passed to loom_linker_finalize_roots.
  iree_string_view_list_t root_symbols;
} loom_linker_add_options_t;

// Requested outward disposition of one planned source symbol.
typedef uint8_t loom_linker_symbol_output_t;
enum loom_linker_symbol_output_e {
  // Preserve the authored visibility, export, import, and retention contract.
  LOOM_LINKER_SYMBOL_OUTPUT_AUTHORED = 0,
  // Internalize the symbol as an implementation dependency.
  LOOM_LINKER_SYMBOL_OUTPUT_DEPENDENCY = 1,
  // Preserve the requester-facing contract and retain the resulting symbol.
  LOOM_LINKER_SYMBOL_OUTPUT_ROOT = 2,
};

// Output dispositions parallel to an exact source symbol selection.
typedef struct loom_linker_source_symbol_output_list_t {
  // Number of output dispositions.
  iree_host_size_t count;
  // Dispositions in source-selection order.
  const loom_linker_symbol_output_t* values;
} loom_linker_source_symbol_output_list_t;

static inline loom_linker_source_symbol_output_list_t
loom_linker_source_symbol_output_list_empty(void) {
  return (loom_linker_source_symbol_output_list_t){0};
}

// Exact module-local source symbol ordinals selected by a link plan.
typedef struct loom_linker_source_symbol_list_t {
  // Number of ordinals in the selection.
  iree_host_size_t count;
  // Strictly increasing module-local source symbol ordinals.
  const iree_host_size_t* ordinals;
} loom_linker_source_symbol_list_t;

// One omitted source symbol already projected into this linker's target.
typedef struct loom_linker_source_symbol_binding_t {
  // Module-local source symbol ordinal referenced by selected source IR.
  uint32_t source_ordinal;
  // Existing linker-target symbol replacing source_ordinal.
  loom_symbol_ref_t target;
} loom_linker_source_symbol_binding_t;

// Exact source-to-target bindings accompanying one sparse selection.
typedef struct loom_linker_source_symbol_binding_list_t {
  // Number of entries in values.
  iree_host_size_t count;
  // Strictly increasing bindings owned by the caller for the add duration.
  // Bound ordinals must be in range and absent from source_symbols; target
  // refs must come from an earlier add to this linker.
  const loom_linker_source_symbol_binding_t* values;
} loom_linker_source_symbol_binding_list_t;

// Returns an empty exact source-symbol binding list.
static inline loom_linker_source_symbol_binding_list_t
loom_linker_source_symbol_binding_list_empty(void) {
  return (loom_linker_source_symbol_binding_list_t){0};
}

// Caller-provided output storage for projected target symbol references.
typedef struct loom_linker_target_symbol_list_t {
  // Number of writable entries in values.
  iree_host_size_t count;
  // Target references written in source-selection order.
  loom_symbol_ref_t* values;
} loom_linker_target_symbol_list_t;

static inline loom_linker_target_symbol_list_t
loom_linker_target_symbol_list_empty(void) {
  return (loom_linker_target_symbol_list_t){0};
}

// Allocates an incremental linker over |context|.
//
// The linker owns a fresh target module allocated from |block_pool|. The
// returned linker must be released with loom_linker_free(), even after a
// successful finish, so its scratch state can be returned to the block pool.
iree_status_t loom_linker_allocate(loom_context_t* context,
                                   const loom_linker_options_t* options,
                                   iree_arena_block_pool_t* block_pool,
                                   iree_allocator_t allocator,
                                   loom_linker_t** out_linker);

// Frees |linker| and any unfinished target module it still owns.
void loom_linker_free(loom_linker_t* linker);

// Adds one materialized source module to |linker|.
//
// The source module must share the linker's context. Symbol references are
// remapped as IR is cloned:
// - Public/imported/exported, config, declaration, and unresolved-anchor names
//   use global link identity.
// - Concrete private definitions keep module-local identity. If their authored
//   name is already occupied by another concrete definition, the new private
//   definition is assigned a deterministic fresh target name.
// - A concrete definition may fill an existing declaration or unresolved anchor
//   with the same authored name; compatible declaration contracts are merged
//   into the selected definition.
//
// The linker retains no pointers into |source_module| after this call returns.
iree_status_t loom_linker_add_module(loom_linker_t* linker,
                                     const loom_module_t* source_module,
                                     const loom_linker_add_options_t* options);

// Adds an exact precomputed source symbol selection to |linker|.
//
// |source_symbols| must be strictly increasing. The caller owns dependency
// closure: references from selected IR to omitted source symbols fail unless
// |source_bindings| explicitly maps them to symbols projected by a prior add.
// Bindings do not clone, merge, or otherwise inspect the omitted source
// definition; the caller owns compatibility with the projected target.
// When |source_outputs| is non-empty it must have one entry per selected source
// symbol. Authored disposition preserves the source linkage surface;
// dependency disposition internalizes it; root disposition preserves and
// retains the requester-facing surface. When
// |out_target_symbols| is non-empty it must have one entry per selected source
// symbol and receives the corresponding stable linked-module references. The
// linker retains no pointers into any source or output storage after this call
// returns.
iree_status_t loom_linker_add_module_symbols(
    loom_linker_t* linker, const loom_module_t* source_module,
    loom_linker_source_symbol_list_t source_symbols,
    loom_linker_source_symbol_binding_list_t source_bindings,
    loom_linker_source_symbol_output_list_t source_outputs,
    loom_linker_target_symbol_list_t out_target_symbols);

// Adds every symbol from an exact dependency-closed source module.
//
// This is the dense counterpart to loom_linker_add_module_symbols for compact
// modules produced by sparse materialization. Every source symbol is cloned
// without reachability discovery, and source symbol IDs map directly into the
// exact target-symbol projection. Non-symbol module metadata is also cloned.
// |source_outputs| follows the same contract as
// loom_linker_add_module_symbols and uses dense symbol order. When
// |out_target_symbols| is non-empty it must have one entry per source symbol
// and receives the corresponding stable linked-module references. The linker
// retains no pointers into any source or output storage after this call
// returns.
iree_status_t loom_linker_add_exact_module(
    loom_linker_t* linker, const loom_module_t* source_module,
    loom_linker_source_symbol_output_list_t source_outputs,
    loom_linker_target_symbol_list_t out_target_symbols);

// Finalizes explicit output |root_symbols| after all modules have been added.
//
// Every named root must have a materialized definition or declaration.
// Function-like roots are marked retained so their module-boundary identity
// survives later transforms. Per-add root symbols are reachability anchors and
// may include transitive dependencies selected by a higher-level link plan;
// they are not finalized automatically.
iree_status_t loom_linker_finalize_roots(loom_linker_t* linker,
                                         iree_string_view_list_t root_symbols);

// Finalizes the linked output module and transfers ownership to the caller.
//
// The caller owns *out_module on success and must release it with
// loom_module_free(). The linker remains valid only for loom_linker_free().
iree_status_t loom_linker_finish(loom_linker_t* linker,
                                 loom_module_t** out_module);

// Links already-materialized source modules into a freshly allocated output
// module. This is a compatibility wrapper around the incremental linker.
//
// All source modules must share the same finalized context. The returned module
// is owned by the caller and must be released with loom_module_free().
//
// Symbol policy:
// - Concrete private definitions keep per-input identity and are renamed on
//   conflict.
// - Public/imported/exported, config, declaration, and unresolved-anchor names
//   resolve by global link identity.
// - A func.decl may be superseded by a concrete definition with the same name;
//   compatible declaration-owned signature, target, ABI, export, import, and
//   modifier contracts merge into the selected symbol-defining op.
// - Multiple public concrete definitions for the same name are rejected.
// - Unresolved references stay unresolved for the verifier to diagnose.
//
// When options.root_symbols is non-empty, the linker treats source modules as
// libraries: it materializes only those roots and the module-local symbols
// reachable from their attributes/regions. A declaration that is superseded by
// a reachable definition provides the structural insertion point for that
// definition, so small harness modules can replace func.decl placeholders with
// library definitions without concatenating the entire library.
iree_status_t loom_link_materialized_modules(
    const loom_module_t* const* source_modules,
    iree_host_size_t source_module_count, const loom_link_options_t* options,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_module_t** out_module);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_LINK_LINKER_H_
