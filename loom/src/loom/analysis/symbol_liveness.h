// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Rebuildable symbol liveness analysis.
//
// The engine computes live module symbols from an explicit root policy,
// concrete symbol dependency edges, and optional dialect/provider contributed
// edges discovered while scanning reachable symbol bodies.

#ifndef LOOM_ANALYSIS_SYMBOL_LIVENESS_H_
#define LOOM_ANALYSIS_SYMBOL_LIVENESS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_dependencies.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_symbol_liveness_t loom_symbol_liveness_t;
typedef struct loom_symbol_liveness_contributor_context_t
    loom_symbol_liveness_contributor_context_t;
typedef struct loom_symbol_liveness_contributor_t
    loom_symbol_liveness_contributor_t;
typedef struct loom_symbol_liveness_options_t loom_symbol_liveness_options_t;

typedef uint32_t loom_symbol_liveness_flags_t;

typedef enum loom_symbol_liveness_flag_bits_e {
  // Treat module-root dependency records, such as static module encodings, as
  // roots.
  LOOM_SYMBOL_LIVENESS_INCLUDE_MODULE_EDGES = 1u << 0,
} loom_symbol_liveness_flag_bits_e;

// Returns true when |symbol| should seed the live set.
typedef bool (*loom_symbol_liveness_root_query_fn_t)(
    void* user_data, const loom_module_t* module, loom_symbol_id_t symbol_id,
    const loom_symbol_t* symbol);

// Visits one operation inside a reachable symbol body and may add synthetic
// liveness edges through loom_symbol_liveness_mark_* helpers.
typedef iree_status_t (*loom_symbol_liveness_visit_op_fn_t)(
    void* user_data, loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* op);

// Context passed to liveness contributors.
typedef struct loom_symbol_liveness_contributor_context_t {
  // Module being analyzed.
  const loom_module_t* module;

  // Concrete symbol dependencies for the same module snapshot.
  const loom_symbol_dependency_table_t* dependencies;

  // Arena owned by this liveness computation.
  iree_arena_allocator_t* arena;

  // Live source symbol whose body is being scanned.
  loom_symbol_id_t source_symbol_id;

  // Live source symbol record.
  const loom_symbol_t* source_symbol;

  // Private engine state used by loom_symbol_liveness_mark_* helpers.
  void* engine_state;
} loom_symbol_liveness_contributor_context_t;

// Dialect/provider liveness contributor.
typedef struct loom_symbol_liveness_contributor_t {
  // Optional callback invoked for each op in reachable symbol bodies.
  loom_symbol_liveness_visit_op_fn_t visit_op;

  // Opaque payload passed to visit_op.
  void* user_data;
} loom_symbol_liveness_contributor_t;

// Options for one liveness computation.
typedef struct loom_symbol_liveness_options_t {
  // Root and module-edge behavior flags.
  loom_symbol_liveness_flags_t flags;

  // Optional root classifier. If NULL, only module edges and contributed edges
  // from already-marked symbols can seed liveness.
  loom_symbol_liveness_root_query_fn_t root_query;

  // Opaque payload passed to root_query.
  void* root_query_user_data;

  // Borrowed contributor array.
  const loom_symbol_liveness_contributor_t* contributors;

  // Number of contributor entries.
  iree_host_size_t contributor_count;
} loom_symbol_liveness_options_t;

// Immutable liveness result for one module snapshot.
typedef struct loom_symbol_liveness_t {
  // Module this result describes.
  const loom_module_t* module;

  // Concrete dependency table used for this result.
  const loom_symbol_dependency_table_t* dependencies;

  // One byte per symbol: non-zero means live.
  const uint8_t* live_symbols;

  // Number of entries in live_symbols.
  iree_host_size_t symbol_count;

  // Number of concrete dependency edges traversed from live symbols.
  uint32_t concrete_edge_count;

  // Number of contributor-added symbol edges.
  uint32_t contributed_edge_count;
} loom_symbol_liveness_t;

// Marks a module-local symbol live from a contributor.
iree_status_t loom_symbol_liveness_mark_symbol_id(
    loom_symbol_liveness_contributor_context_t* context,
    loom_symbol_id_t symbol_id);

// Marks a module-local symbol reference live from a contributor. Invalid and
// cross-module refs are ignored.
iree_status_t loom_symbol_liveness_mark_symbol_ref(
    loom_symbol_liveness_contributor_context_t* context, loom_symbol_ref_t ref);

// Computes symbol liveness into |arena|.
iree_status_t loom_symbol_liveness_compute(
    const loom_module_t* module,
    const loom_symbol_dependency_table_t* dependencies,
    const loom_symbol_liveness_options_t* options,
    iree_arena_allocator_t* arena, loom_symbol_liveness_t* out_liveness);

// Returns true when |symbol_id| is live in |liveness|.
bool loom_symbol_liveness_is_live(const loom_symbol_liveness_t* liveness,
                                  loom_symbol_id_t symbol_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_SYMBOL_LIVENESS_H_
