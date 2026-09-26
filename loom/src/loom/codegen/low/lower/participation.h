// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Optional source execution evidence for memory reports.

#ifndef LOOM_CODEGEN_LOW_LOWER_PARTICIPATION_H_
#define LOOM_CODEGEN_LOW_LOWER_PARTICIPATION_H_

#include "loom/analysis/condition_facts.h"
#include "loom/analysis/symbolic_expr.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_context_t loom_low_lower_context_t;

typedef enum loom_low_lower_participation_kind_e {
  LOOM_LOW_LOWER_PARTICIPATION_UNKNOWN = 0,
  LOOM_LOW_LOWER_PARTICIPATION_FULL = 1,
  LOOM_LOW_LOWER_PARTICIPATION_COMPARISON = 2,
} loom_low_lower_participation_kind_t;

// Completed numeric proof for one comparison operand.
typedef struct loom_low_lower_participation_operand_t {
  // Canonical exact expression and its conservative value facts.
  loom_symbolic_expr_t expression;
  // Optional retained quotient/remainder function for the expression.
  const loom_symbolic_projection_t* projection;
} loom_low_lower_participation_operand_t;

// A complete entry selector and its completed numeric operand proofs. All
// referenced terms share the function arena and immutable source snapshot.
typedef struct loom_low_lower_participation_condition_t {
  // Exact semantics of the complete Boolean selector.
  loom_condition_integer_comparison_t comparison;
  // Retained numeric summary for the left comparison operand.
  loom_low_lower_participation_operand_t lhs;
  // Retained numeric summary for the right comparison operand.
  loom_low_lower_participation_operand_t rhs;
  // Comparison outcome selecting this entry.
  bool assumed_truth;
} loom_low_lower_participation_condition_t;

// Execution evidence relative to a complete subgroup. This describes entry
// participation, not execution frequency.
typedef struct loom_low_lower_participation_t {
  // Whether entry participation is unknown, complete, or comparison-selected.
  loom_low_lower_participation_kind_t kind;
  // Arena-owned complete selector when kind is COMPARISON; otherwise NULL.
  const loom_low_lower_participation_condition_t* condition;
} loom_low_lower_participation_t;

// Returns retained subgroup participation for one source operation. The caller
// must have enabled detail reports. The first query summarizes the root CFG
// once in the function arena; nested structured operations use shared
// uniformity proofs. Comparison operands are expanded by the shared expression
// owner, so target observers only read indexed summaries and never walk source
// producers. The source IR and fact snapshot must remain immutable for this
// report lifetime. Status represents allocation failure; unproven participation
// is UNKNOWN.
iree_status_t loom_low_lower_source_subgroup_participation(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_participation_t* out_participation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_PARTICIPATION_H_
