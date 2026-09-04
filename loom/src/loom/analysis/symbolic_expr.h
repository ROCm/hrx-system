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
// allocating on every comparison. Reset the context memo when the module,
// fact table, or active condition fact set changes.

#ifndef LOOM_ANALYSIS_SYMBOLIC_EXPR_H_
#define LOOM_ANALYSIS_SYMBOLIC_EXPR_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/condition_facts.h"
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

// Stable summary for one analyzed SSA value.
typedef struct loom_symbolic_expr_summary_t {
  // Normalized symbolic expression for the value.
  loom_symbolic_expr_t expression;

  // SSA value exactly materializing expression's nonconstant terms, or
  // LOOM_VALUE_ID_INVALID when no such value was retained during expansion.
  loom_value_id_t materialized_dynamic_value_id;
} loom_symbolic_expr_summary_t;

// Memoized condition-refined facts for one SSA value. This state is owned by
// loom_symbolic_expr_context_t and retained across context resets.
typedef struct loom_symbolic_expr_condition_fact_memo_entry_t {
  // Memo state: zero is empty, one is visiting, and two is ready.
  uint8_t state;

  // Maximum producer depth represented by facts.
  uint8_t depth;

  // Cached facts when state is ready.
  loom_value_facts_t facts;
} loom_symbolic_expr_condition_fact_memo_entry_t;

// Per-analysis state for symbolic expression queries.
typedef struct loom_symbolic_expr_context_t {
  // Module containing SSA value definitions queried by the context.
  const loom_module_t* module;

  // Optional active domain mapping module value IDs to compact local ordinals.
  loom_local_value_domain_t* value_domain;

  // Dense facts used to seed ranges, exact constants, and divisibility.
  const loom_value_fact_table_t* fact_table;

  // Optional edge-local facts applied during branch-sensitive proofs. Reset
  // the context after changing this pointer or the facts it references.
  const loom_condition_fact_set_t* condition_facts;

  // Reusable traversal state for branch-sensitive fact queries.
  loom_condition_query_t condition_query;

  // Arena used for memo entries, retained term arrays, and scratch growth.
  iree_arena_allocator_t* arena;

  // Maximum number of terms retained before degrading to facts-only.
  iree_host_size_t maximum_term_count;

  // Memo entries indexed by storage ordinal.
  loom_symbolic_expr_memo_entry_t* memo_entries;

  // Allocated memo entry count.
  iree_host_size_t memo_capacity;

  // Storage ordinals whose memo entries are live in the current epoch.
  loom_value_ordinal_t* touched_memo_ordinals;

  // Number of populated entries in touched_memo_ordinals.
  iree_host_size_t touched_memo_ordinal_count;

  // Allocated entry count in touched_memo_ordinals.
  iree_host_size_t touched_memo_ordinal_capacity;

  // Condition-refined fact memo entries indexed by storage ordinal.
  loom_symbolic_expr_condition_fact_memo_entry_t* condition_fact_memo_entries;

  // Allocated condition-refined fact memo entry count.
  iree_host_size_t condition_fact_memo_capacity;

  // Storage ordinals whose condition-fact memo entries are live.
  loom_value_ordinal_t* touched_condition_fact_memo_ordinals;

  // Number of populated entries in touched_condition_fact_memo_ordinals.
  iree_host_size_t touched_condition_fact_memo_ordinal_count;

  // Allocated entry count in touched_condition_fact_memo_ordinals.
  iree_host_size_t touched_condition_fact_memo_ordinal_capacity;

  // Reusable term buffer for normalization and comparison.
  loom_symbolic_term_t* scratch_terms;

  // Allocated scratch term count.
  iree_host_size_t scratch_term_capacity;

  // Recursive select-case proof depth, capped to keep proof work bounded.
  uint8_t condition_proof_depth;
} loom_symbolic_expr_context_t;

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
// reset it only after all expressions produced by this context are dead. When
// provided, |value_domain| must remain acquired for the context lifetime and
// may be extended by queries that encounter rewrite-created values.
void loom_symbolic_expr_context_initialize(
    const loom_module_t* module, loom_local_value_domain_t* value_domain,
    const loom_value_fact_table_t* fact_table, iree_arena_allocator_t* arena,
    loom_symbolic_expr_context_t* out_context);

// Clears memoized expressions and condition-refined facts while retaining
// scratch and memo capacity.
void loom_symbolic_expr_context_reset(loom_symbolic_expr_context_t* context);

// Copies a previously computed value summary without expanding |value_id| or
// mutating the context. Returns false when the value has not been computed
// since the last reset.
bool loom_symbolic_expr_context_try_lookup_summary(
    const loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    loom_symbolic_expr_summary_t* out_summary);

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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_SYMBOLIC_EXPR_H_
