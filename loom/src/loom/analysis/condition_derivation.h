// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical edge-local facts and their arena-backed derivation storage.

#ifndef LOOM_ANALYSIS_CONDITION_DERIVATION_H_
#define LOOM_ANALYSIS_CONDITION_DERIVATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/integer_relation.h"
#include "loom/ir/facts.h"
#include "loom/ir/types.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_condition_integer_operand_kind_e {
  // Operand is an SSA value.
  LOOM_CONDITION_INTEGER_OPERAND_VALUE = 0,
  // Operand is an exact signed integer constant.
  LOOM_CONDITION_INTEGER_OPERAND_CONSTANT = 1,
} loom_condition_integer_operand_kind_t;

typedef struct loom_condition_integer_operand_t {
  // Operand category.
  loom_condition_integer_operand_kind_t kind;
  // SSA value ID when kind is LOOM_CONDITION_INTEGER_OPERAND_VALUE.
  loom_value_id_t value_id;
  // Exact integer when kind is LOOM_CONDITION_INTEGER_OPERAND_CONSTANT.
  int64_t constant;
} loom_condition_integer_operand_t;

typedef struct loom_condition_integer_relation_t {
  // Relation between left and right operands.
  loom_symbolic_integer_relation_t relation;
  // Left-hand side of the relation.
  loom_condition_integer_operand_t left;
  // Right-hand side of the relation.
  loom_condition_integer_operand_t right;
} loom_condition_integer_relation_t;

// Exact predicate truth derived while traversing a condition expression.
// Register predicates retain the branch outcome, not an integer value of one.
typedef struct loom_condition_boolean_fact_t {
  // SSA predicate whose truth is known.
  loom_value_id_t value_id;

  // Exact truth of the predicate.
  bool value;
} loom_condition_boolean_fact_t;

typedef struct loom_condition_fact_set_t {
  // Caller-owned storage for integer relations. Query APIs append each exact
  // relation at most once.
  loom_condition_integer_relation_t* integer_relations;
  // Number of populated entries in integer_relations.
  iree_host_size_t integer_relation_count;
  // Allocated entry count for integer_relations.
  iree_host_size_t integer_relation_capacity;
} loom_condition_fact_set_t;

// Complete arena-backed result of deriving one condition expression.
typedef struct loom_condition_derivation_t {
  // All integer relations implied by the selected condition outcome.
  loom_condition_fact_set_t integer_facts;

  // Exact truth for every visited Boolean SSA value.
  loom_condition_boolean_fact_t* boolean_facts;

  // Number of populated entries in boolean_facts.
  iree_host_size_t boolean_fact_count;

  // Allocated entry count for boolean_facts.
  iree_host_size_t boolean_fact_capacity;

  // Arena owning relation and Boolean storage.
  iree_arena_allocator_t* arena;
} loom_condition_derivation_t;

// Initializes a caller-owned fact set over fixed storage.
void loom_condition_fact_set_initialize(
    loom_condition_integer_relation_t* integer_relation_storage,
    iree_host_size_t integer_relation_capacity,
    loom_condition_fact_set_t* out_facts);

// Resets a fact set while retaining caller-owned storage.
void loom_condition_fact_set_reset(loom_condition_fact_set_t* facts);

// Initializes a complete derivation whose storage grows in |arena|. Repeated
// resets and queries reuse the allocated high-water capacity.
void loom_condition_derivation_initialize(
    iree_arena_allocator_t* arena, loom_condition_derivation_t* out_derivation);

// Resets a complete derivation while retaining its arena-backed storage.
void loom_condition_derivation_reset(loom_condition_derivation_t* derivation);

// Appends one integer consequence, canonicalizing its operand order. The
// producer completes the derivation before exposing it to consumers.
iree_status_t loom_condition_derivation_append_integer_relation(
    loom_condition_derivation_t* derivation,
    loom_condition_integer_relation_t relation);

// Appends exact Boolean truth. A NULL derivation omits Boolean storage for a
// bounded integer-only query.
iree_status_t loom_condition_derivation_append_boolean_fact(
    loom_condition_derivation_t* derivation, loom_value_id_t value_id,
    bool value);

// Reusable construction storage. Its arena may outlive individual results;
// no relation or result pointer survives a completed or failed query. Releasing
// that arena requires resetting the owner pointer to NULL before reuse.
typedef struct loom_condition_composition_scratch_t
    loom_condition_composition_scratch_t;

// Composes condition-owned anchors through original local and ambient alias
// relations, then sorts and deduplicates both result classes. Every published
// consequence retains an original condition operand; two ambient-only values
// are never connected. Scratch is allocated lazily in |scratch_arena| and
// retains high-water capacities through |inout_scratch|. Results live only in
// the derivation arena; separate result storage survives releasing scratch.
// Allocation failure leaves an incomplete derivation; resetting the derivation
// permits a fresh query with the same scratch. Scratch and result arenas may be
// the same.
iree_status_t loom_condition_derivation_complete(
    const loom_value_fact_table_t* fact_table,
    iree_arena_allocator_t* scratch_arena,
    loom_condition_composition_scratch_t** inout_scratch,
    loom_condition_derivation_t* derivation);

// Applies a single integer relation to scalar range facts for |value_id| when
// the relation can be reduced to value-vs-constant form. Value-to-value
// relations remain useful to symbolic consumers even when this returns false.
bool loom_condition_integer_relation_apply_to_value_facts(
    const loom_condition_integer_relation_t* relation,
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_value_facts_t* inout_facts);

// Converts a relation into an assume predicate for |value_id| when that value
// is one side of the relation. The produced predicate is normalized so
// args[0] is |value_id|; the opposite side is either a literal constant when
// known exact or an SSA value reference.
bool loom_condition_integer_relation_make_predicate_for_value(
    const loom_condition_integer_relation_t* relation,
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_predicate_t* out_predicate);

// Applies all applicable relations in |facts| to scalar range facts for
// |value_id|. Returns true if at least one relation was applicable.
bool loom_condition_fact_set_apply_to_value_facts(
    const loom_condition_fact_set_t* facts,
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_value_facts_t* inout_facts);

// Returns true when two integer relation operands identify the same value or
// constant.
bool loom_condition_integer_operands_equal(
    loom_condition_integer_operand_t left,
    loom_condition_integer_operand_t right);

// Returns true when |known| can answer whether |queried| is true. If the
// relations share operands but imply the opposite result, |out_result| is set
// to false.
bool loom_condition_integer_relation_implies(
    const loom_condition_integer_relation_t* known,
    const loom_condition_integer_relation_t* queried, bool* out_result);

// Returns the possible integer order outcomes from the conjunction of
// edge-local relations over |left| and |right|. Retained SSA identities and
// exact scalar values participate in operand equivalence. No applicable
// relation leaves every outcome possible; contradictory relations return zero.
loom_integer_order_outcome_bits_t loom_condition_fact_set_query_integer_order(
    const loom_condition_fact_set_t* facts,
    const loom_value_fact_table_t* fact_table,
    loom_condition_integer_operand_t left,
    loom_condition_integer_operand_t right);

// Attempts to evaluate |queried| from the conjunction of edge-local relations
// in |facts| over the same operands. Retained SSA identities and exact scalar
// values in |fact_table| participate in operand equivalence, so aliases retain
// path proofs and SSA constants match their literals. Returns true when proven
// either true or false and writes that result to |out_result|. Contradictory
// relations leave the result unknown.
bool loom_condition_fact_set_proves_integer_relation(
    const loom_condition_fact_set_t* facts,
    const loom_value_fact_table_t* fact_table,
    const loom_condition_integer_relation_t* queried, bool* out_result);

// Returns true when each relation implies the other.
bool loom_condition_integer_relations_equivalent(
    const loom_condition_integer_relation_t* left,
    const loom_condition_integer_relation_t* right);

// Computes the strongest common relation preserved by |left| and the
// conjunction of |right_count| relations in |right|. Combines all relations
// over the same operands, including swapped operands. Returns false when no
// common constraint remains; unrelated operand pairs impose no constraint.
bool loom_condition_integer_relation_meet(
    const loom_condition_integer_relation_t* left,
    const loom_condition_integer_relation_t* right,
    iree_host_size_t right_count,
    loom_condition_integer_relation_t* out_relation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_CONDITION_DERIVATION_H_
