// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Private storage for symbolic queries under hypothetical conditions.

#ifndef LOOM_ANALYSIS_SYMBOLIC_SCOPE_H_
#define LOOM_ANALYSIS_SYMBOLIC_SCOPE_H_

#include "loom/analysis/symbolic_expr.h"
#include "loom/ir/intern_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// A temporary scope indexes only the values its proof visits. The owning
// condition arena retains the index and all expression/fact payloads together.
struct loom_symbolic_scope_t {
  // Parent traversal workspace, borrowed only between completed queries.
  loom_condition_query_t* condition_query;
  // Maps source value identities to stable, compact scope-local ordinals.
  loom_intern_table_t index;
  // Source identity at each assigned ordinal; count is index.count.
  loom_value_id_t* values;
  // Allocated entry count in values.
  iree_host_size_t capacity;
};

// Finds a value already visited by the temporary scope, or returns INVALID.
loom_value_ordinal_t loom_symbolic_scope_find(
    const loom_symbolic_scope_t* scope, loom_value_id_t value);

// Assigns a compact ordinal on first visit. Only arena growth can fail.
iree_status_t loom_symbolic_scope_register(loom_symbolic_scope_t* scope,
                                           iree_arena_allocator_t* arena,
                                           loom_value_id_t value,
                                           loom_value_ordinal_t* out_ordinal);

// Returns the owning predicate traversal workspace. Symbolic expansion and
// conditional proof enter it only after the preceding query has completed.
static inline loom_condition_query_t* loom_symbolic_expr_condition_query(
    loom_symbolic_expr_context_t* context) {
  return context->scoped_storage ? context->scoped_storage->condition_query
                                 : &context->condition_query;
}

static inline loom_value_ordinal_t loom_symbolic_expr_try_value_ordinal(
    const loom_symbolic_expr_context_t* context, loom_value_id_t value_id) {
  if (context->scoped_storage) {
    return loom_symbolic_scope_find(context->scoped_storage, value_id);
  }
  return context->value_domain ? loom_local_value_domain_try_ordinal(
                                     context->value_domain, value_id)
                               : (loom_value_ordinal_t)value_id;
}

static inline iree_status_t loom_symbolic_expr_resolve_value_ordinal(
    loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    loom_value_ordinal_t* out_ordinal) {
  if (context->scoped_storage) {
    return loom_symbolic_scope_register(context->scoped_storage, context->arena,
                                        value_id, out_ordinal);
  }
  if (context->value_domain) {
    return loom_local_value_domain_register_value(context->value_domain,
                                                  value_id, out_ordinal);
  }
  *out_ordinal = (loom_value_ordinal_t)value_id;
  return iree_ok_status();
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_SYMBOLIC_SCOPE_H_
