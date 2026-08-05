// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation-free bounded symbolic expression summaries.
//
// Bounded summaries trade completeness for a fixed amount of caller-owned
// storage. They preserve small affine producer graphs and conservatively
// represent unsupported or oversized graphs as their original SSA value.

#ifndef LOOM_ANALYSIS_SYMBOLIC_EXPR_BOUNDED_H_
#define LOOM_ANALYSIS_SYMBOLIC_EXPR_BOUNDED_H_

#include "loom/analysis/symbolic_expr.h"

#ifdef __cplusplus
extern "C" {
#endif

// Builds a bounded linear expression for |value_id| using caller-owned term
// storage. Unsupported or too-large producer graphs conservatively summarize
// as the original SSA value. Passing no term storage produces a facts-only
// expression.
void loom_symbolic_expr_from_value_bounded(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_symbolic_term_t* terms,
    iree_host_size_t term_capacity, loom_symbolic_expr_t* out_expression);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_SYMBOLIC_EXPR_BOUNDED_H_
