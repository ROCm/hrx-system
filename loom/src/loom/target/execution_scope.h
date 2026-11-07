// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_EXECUTION_SCOPE_H_
#define LOOM_TARGET_EXECUTION_SCOPE_H_

#include "loom/analysis/symbol_facts.h"
#include "loom/analysis/symbol_references.h"
#include "loom/target/resolved_target.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_environment_t loom_target_environment_t;

// Target environment of one independently executing region. The reference
// analysis owns its structural scope; this table projects that retained scope
// without inspecting the region or any callee body.
typedef struct loom_target_execution_scope_t {
  // Authored witness, absent when projection changes its execution contract.
  loom_symbol_ref_t witness;
  // Immutable worker facts, or NULL for an unresolved target declaration.
  const loom_target_facts_t* facts;
  // Linked family provider, or NULL when the environment has none.
  const loom_target_provider_t* provider;
} loom_target_execution_scope_t;

// One immutable symbol-reference snapshot's worker environments. The containing
// function's target must stay fixed for the lifetime of this table. A callgraph
// solver exploring several versions of one definition owns distinct contexts.
// Zero initialization creates an empty table.
typedef struct loom_target_execution_scope_table_t {
  // Arena-owned entries indexed by the reference scope ordinal minus one.
  // NULL until the first worker is requested; each populated slot owns a row.
  loom_target_execution_scope_t** values;
} loom_target_execution_scope_table_t;

// Resolves a nonzero execution scope once. An explicit region target replaces
// |inherited| even when unresolved; an omitted target inherits it. Projection
// uses the family's worker contract and preserves a witness only if exact.
// |symbol_facts| owns authored target projections; all cache storage belongs to
// |arena| and is discarded with the reference snapshot.
iree_status_t loom_target_execution_scope_resolve(
    loom_target_execution_scope_table_t* table,
    const loom_symbol_reference_table_t* references,
    loom_symbol_reference_execution_scope_id_t scope,
    const loom_module_t* module, loom_symbol_fact_table_t* symbol_facts,
    const loom_target_environment_t* environment,
    loom_target_execution_scope_t inherited, iree_arena_allocator_t* arena,
    const loom_target_execution_scope_t** out_scope);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_EXECUTION_SCOPE_H_
