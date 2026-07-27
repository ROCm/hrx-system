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
  // Root symbol names to materialize. An empty list links every materialized
  // source symbol.
  iree_string_view_list_t root_symbols;
} loom_link_options_t;

// Stateful incremental linker. Owns the output module until finish transfers it
// to the caller.
typedef struct loom_linker_t loom_linker_t;

// Options controlling linker construction.
typedef struct loom_linker_options_t {
  // Name assigned to the linked output module. Defaults to "linked".
  iree_string_view_t module_name;
} loom_linker_options_t;

// Options controlling one input module add.
typedef struct loom_linker_add_options_t {
  // Root symbol names to materialize from this module. An empty list links
  // every materialized top-level symbol in the module.
  iree_string_view_list_t root_symbols;
} loom_linker_add_options_t;

// Creates an incremental linker over |context|.
//
// The linker owns a fresh target module allocated from |block_pool|. The
// returned linker must be released with loom_linker_free(), even after a
// successful finish, so its scratch state can be returned to the block pool.
iree_status_t loom_linker_create(loom_context_t* context,
                                 const loom_linker_options_t* options,
                                 iree_arena_block_pool_t* block_pool,
                                 iree_allocator_t allocator,
                                 loom_linker_t** out_linker);

// Releases |linker| and any unfinished target module it still owns.
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
