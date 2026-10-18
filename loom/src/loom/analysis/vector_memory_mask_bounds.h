// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Symbolic active-lane bounds for vector memory masks.

#ifndef LOOM_ANALYSIS_VECTOR_MEMORY_MASK_BOUNDS_H_
#define LOOM_ANALYSIS_VECTOR_MEMORY_MASK_BOUNDS_H_

#include "loom/analysis/symbolic_expr.h"
#include "loom/ir/module.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Active-lane facts for one vector memory mask.
typedef struct loom_vector_memory_mask_bounds_t {
  // True when every mask lane is proven inactive.
  bool definitely_empty;

  // True when lower_bound and upper_bound describe a unit-step prefix mask.
  bool has_unit_range;

  // First coordinate tested by lane zero when has_unit_range is true.
  loom_symbolic_expr_t lower_bound;

  // Exclusive coordinate bound when has_unit_range is true.
  loom_symbolic_expr_t upper_bound;
} loom_vector_memory_mask_bounds_t;

// Reusable condition derivation for the latest uniform predicate and lexical
// scope. Allocated lazily in the symbolic context's arena and valid while its
// module and value facts are unchanged.
typedef struct loom_vector_memory_mask_scope_t loom_vector_memory_mask_scope_t;

// Analyzes one vector memory mask and activates the access's condition scope in
// |expression_context|, invalidating symbolic memoization when its facts
// change. A uniform mask contributes its scalar predicate only for this access.
// An invalid mask selects |lexical_scope| without adding facts. The caller
// restores that lexical scope before analyzing nested control flow.
//
// |inout_mask_scope| starts NULL and retains only the latest predicate's
// complete derivation, reusing its high-water storage across subsequent
// queries. Returned bounds and the active scope remain valid until the next
// query or scope change.
iree_status_t loom_vector_memory_mask_bounds_analyze(
    loom_symbolic_expr_context_t* expression_context,
    const loom_condition_fact_scope_t* lexical_scope,
    loom_value_id_t mask_value_id,
    loom_vector_memory_mask_scope_t** inout_mask_scope,
    loom_vector_memory_mask_bounds_t* out_bounds);

// Returns the exclusive tail end when |bounds| describes a unit-step prefix
// whose lower bound is proven equal to |origin|.
iree_status_t loom_vector_memory_mask_bounds_tail_end(
    loom_symbolic_expr_context_t* expression_context,
    const loom_vector_memory_mask_bounds_t* bounds,
    const loom_symbolic_expr_t* origin, loom_symbolic_expr_t* out_end,
    bool* out_known);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_VECTOR_MEMORY_MASK_BOUNDS_H_
