// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Symbolic linear expressions over Loom SSA values.
//
// The representation is intentionally narrow and exact:
//
//   constant + sum(coefficient_i * %value_i)
//
// Terms are normalized by value ID, duplicate terms are combined, and zero
// coefficients are removed. Unsupported or nonlinear producer ops do not make
// analysis fail; the defining SSA result remains a symbolic variable and range
// facts stay attached. This lets consumers such as view alias analysis prove
// the common affine cases while preserving a conservative escape hatch.
//
// Storage is caller-owned. The context memoizes value-to-expression queries and
// owns a reusable scratch term buffer so fixed-point analyses can query without
// allocating on every comparison. Reset the context memo when the module or
// fact table changes.

#ifndef LOOM_ANALYSIS_SYMBOLIC_EXPR_H_
#define LOOM_ANALYSIS_SYMBOLIC_EXPR_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/integer_relation.h"
#include "loom/ir/facts.h"
#include "loom/ir/ir.h"
#include "loom/ir/module.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Default cap on terms retained in one symbolic expression.
#define LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT 64

typedef struct loom_symbolic_expr_memo_entry_t loom_symbolic_expr_memo_entry_t;
typedef struct loom_condition_fact_set_t loom_condition_fact_set_t;

// A single coefficient times an SSA value.
typedef struct loom_symbolic_term_t {
  // Signed coefficient multiplying value_id.
  int64_t coefficient;

  // SSA value used for algebraic identity and cancellation.
  loom_value_id_t value_id;

  // SSA value whose identity-chain predicates may prove term relations.
  loom_value_id_t relation_value_id;
} loom_symbolic_term_t;

// Flags describing which parts of a symbolic expression are usable.
enum loom_symbolic_expr_flag_bits_e {
  // The expression's terms exactly model the represented value.
  LOOM_SYMBOLIC_EXPR_FLAG_LINEAR = 1u << 0,
};
typedef uint32_t loom_symbolic_expr_flags_t;

// A normalized symbolic linear expression with conservative value facts.
typedef struct loom_symbolic_expr_t {
  // Constant term in the expression.
  int64_t constant;

  // Normalized term array owned by the symbolic expression context arena.
  const loom_symbolic_term_t* terms;

  // Number of entries in terms.
  iree_host_size_t term_count;

  // Conservative facts for the expression value.
  loom_value_facts_t facts;

  // Bitfield of loom_symbolic_expr_flag_bits_e.
  loom_symbolic_expr_flags_t flags;
} loom_symbolic_expr_t;

// Per-analysis state for symbolic expression queries.
typedef struct loom_symbolic_expr_context_t {
  // Module containing SSA value definitions queried by the context.
  const loom_module_t* module;

  // Dense facts used to seed ranges, exact constants, and divisibility.
  const loom_value_fact_table_t* fact_table;

  // Optional edge-local facts applied during branch-sensitive proofs.
  const loom_condition_fact_set_t* condition_facts;

  // Arena used for memo entries, retained term arrays, and scratch growth.
  iree_arena_allocator_t* arena;

  // Maximum number of terms retained before degrading to facts-only.
  iree_host_size_t maximum_term_count;

  // Memo entries indexed by value ID.
  loom_symbolic_expr_memo_entry_t* memo_entries;

  // Allocated memo entry count.
  iree_host_size_t memo_capacity;

  // Reusable term buffer for normalization and comparison.
  loom_symbolic_term_t* scratch_terms;

  // Allocated scratch term count.
  iree_host_size_t scratch_term_capacity;

  // Recursive select-case proof depth, capped to keep proof work bounded.
  uint8_t condition_proof_depth;
} loom_symbolic_expr_context_t;

// Tri-state proof result for symbolic comparisons.
typedef enum loom_symbolic_proof_result_e {
  // The relation could not be proven either way.
  LOOM_SYMBOLIC_PROOF_UNKNOWN = 0,

  // The relation is proven true.
  LOOM_SYMBOLIC_PROOF_TRUE = 1,

  // The relation is proven false.
  LOOM_SYMBOLIC_PROOF_FALSE = 2,
} loom_symbolic_proof_result_t;

// Compact replacement form for a symbolic value difference.
typedef enum loom_symbolic_value_difference_kind_e {
  // The difference is not representable as a single existing value or constant.
  LOOM_SYMBOLIC_VALUE_DIFFERENCE_UNKNOWN = 0,

  // The difference is the exact integer in |constant|.
  LOOM_SYMBOLIC_VALUE_DIFFERENCE_CONSTANT = 1,

  // The difference is the existing SSA value |value_id|.
  LOOM_SYMBOLIC_VALUE_DIFFERENCE_VALUE = 2,
} loom_symbolic_value_difference_kind_t;

// Difference summary for left_value - right_value.
typedef struct loom_symbolic_value_difference_t {
  // Kind of replacement represented by this difference.
  loom_symbolic_value_difference_kind_t kind;

  // Exact integer when kind is LOOM_SYMBOLIC_VALUE_DIFFERENCE_CONSTANT.
  int64_t constant;

  // Existing SSA value when kind is LOOM_SYMBOLIC_VALUE_DIFFERENCE_VALUE.
  loom_value_id_t value_id;
} loom_symbolic_value_difference_t;

static inline bool loom_symbolic_expr_is_linear(
    const loom_symbolic_expr_t* expression) {
  return iree_any_bit_set(expression->flags, LOOM_SYMBOLIC_EXPR_FLAG_LINEAR);
}

static inline bool loom_symbolic_expr_is_constant(
    const loom_symbolic_expr_t* expression) {
  return loom_symbolic_expr_is_linear(expression) &&
         expression->term_count == 0;
}

// Initializes a symbolic expression context. The caller owns |arena| and may
// reset it only after all expressions produced by this context are dead.
void loom_symbolic_expr_context_initialize(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    iree_arena_allocator_t* arena, loom_symbolic_expr_context_t* out_context);

// Clears memoized value expressions while retaining scratch and memo capacity.
void loom_symbolic_expr_context_reset(loom_symbolic_expr_context_t* context);

// Constructs a facts-only expression. This is the conservative result for
// unsupported nonlinear arithmetic when no precise SSA variable is available.
void loom_symbolic_expr_unknown(loom_value_facts_t facts,
                                loom_symbolic_expr_t* out_expression);

// Constructs an exact constant expression.
void loom_symbolic_expr_constant(int64_t value,
                                 loom_symbolic_expr_t* out_expression);

// Constructs an exact symbolic variable expression for |value_id| using any
// facts available in the context.
iree_status_t loom_symbolic_expr_value(loom_symbolic_expr_context_t* context,
                                       loom_value_id_t value_id,
                                       loom_symbolic_expr_t* out_expression);

// Returns the memoized symbolic expression for an SSA value.
iree_status_t loom_symbolic_expr_from_value(
    loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    loom_symbolic_expr_t* out_expression);

// Computes left + right.
iree_status_t loom_symbolic_expr_add(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_expr_t* out_expression);

// Computes left - right.
iree_status_t loom_symbolic_expr_sub(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_expr_t* out_expression);

// Computes expression * multiplier.
iree_status_t loom_symbolic_expr_mul_i64(loom_symbolic_expr_context_t* context,
                                         const loom_symbolic_expr_t* expression,
                                         int64_t multiplier,
                                         loom_symbolic_expr_t* out_expression);

// Attempts to prove left <= right. The implementation uses exact term
// cancellation first and falls back to interval facts without allocating new
// retained expression storage.
iree_status_t loom_symbolic_expr_prove_le(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_proof_result_t* out_result);

// Simplifies left_value - right_value when the normalized difference is a
// single existing value or an exact integer constant.
iree_status_t loom_symbolic_expr_simplify_value_difference(
    loom_symbolic_expr_context_t* context, loom_value_id_t left_value,
    loom_value_id_t right_value,
    loom_symbolic_value_difference_t* out_difference);

// Attempts to prove an integer relation between two SSA values.
iree_status_t loom_symbolic_expr_prove_value_relation(
    loom_symbolic_expr_context_t* context,
    loom_symbolic_integer_relation_t relation, loom_value_id_t left_value,
    loom_value_id_t right_value, loom_symbolic_proof_result_t* out_result);

// Builds a bounded linear expression for |value_id| using caller-owned term
// storage. This is a hot-path, allocation-free summary for consumers that only
// need small affine shapes and cannot plumb arena-backed expression status.
// Unsupported or too-large producer graphs conservatively summarize as the
// original SSA value instead of failing.
void loom_symbolic_expr_from_value_bounded(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_symbolic_term_t* terms,
    iree_host_size_t term_capacity, loom_symbolic_expr_t* out_expression);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_SYMBOLIC_EXPR_H_
