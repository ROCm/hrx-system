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

// Analyzes one vector memory mask using the active path facts in
// |expression_context|.
iree_status_t loom_vector_memory_mask_bounds_analyze(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_symbolic_expr_context_t* expression_context,
    loom_value_id_t mask_value_id,
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
