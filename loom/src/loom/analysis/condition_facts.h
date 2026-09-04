// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Edge-local facts implied by boolean condition values.
//
// This analysis is the reverse direction of ordinary fact inference: given an
// i1 value and an assumed result on one control-flow edge, it derives relations
// that are valid only on that edge. For example, the true edge of
// `index.cmp slt, %i, %n` implies `%i < %n`; the false edge implies
// `%i >= %n`.
//
// The representation is intentionally not just loom_value_facts_t. Absolute
// facts can tighten `%i < 16`, but a useful compiler also needs to preserve
// value-to-value relations such as `%i < %n` for symbolic range and alias
// analysis.

#ifndef LOOM_ANALYSIS_CONDITION_FACTS_H_
#define LOOM_ANALYSIS_CONDITION_FACTS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/integer_relation.h"
#include "loom/ir/facts.h"
#include "loom/ir/ir.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
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

typedef struct loom_condition_fact_set_t {
  // Caller-owned storage for integer relations. Query APIs append each exact
  // relation at most once.
  loom_condition_integer_relation_t* integer_relations;
  // Number of populated entries in integer_relations.
  iree_host_size_t integer_relation_count;
  // Allocated entry count for integer_relations.
  iree_host_size_t integer_relation_capacity;
} loom_condition_fact_set_t;

// One dialect-owned operand refinement guaranteed on a selected condition
// edge. The descriptor and condition op are borrowed from immutable compiler
// state; source is the descriptor-selected condition operand.
typedef struct loom_condition_edge_refinement_t {
  // Condition operation whose result controls the selected edge.
  const loom_op_t* condition_op;
  // Dialect-owned materialization contract for condition_op.
  const loom_condition_refinement_descriptor_t* descriptor;
  // Source operand refined by the selected edge.
  loom_value_id_t source;
  // Truth value assumed for condition_op on the selected edge.
  bool assumed_truth;
} loom_condition_edge_refinement_t;

// Caller-owned bounded storage for semantic condition refinements.
typedef struct loom_condition_edge_refinement_set_t {
  // Refinement storage populated in condition-expression traversal order.
  loom_condition_edge_refinement_t* refinements;
  // Number of populated refinements.
  iree_host_size_t refinement_count;
  // Allocated entry count in refinements.
  iree_host_size_t refinement_capacity;
} loom_condition_edge_refinement_set_t;

typedef struct loom_condition_query_frame_t loom_condition_query_frame_t;

// Reusable state for condition-fact derivation and proof.
//
// The query is tied to one module and is not reentrant. All dynamic storage is
// retained in |arena| so repeated queries reuse their high-water capacity.
// Module mutation may append values between queries; indexed state grows on
// demand to cover the current value table.
typedef struct loom_condition_query_t {
  // Module containing every queried condition value.
  const loom_module_t* module;
  // Optional active domain mapping module value IDs to compact local ordinals.
  loom_local_value_domain_t* value_domain;
  // Arena retaining indexed state and worklist capacity.
  iree_arena_allocator_t* arena;
  // Per-value visitation or memo state indexed by storage ordinal.
  uint8_t* value_states;
  // Allocated entry count in value_states.
  iree_host_size_t value_state_capacity;
  // Storage ordinals whose state must be cleared after the active query.
  loom_value_ordinal_t* touched_ordinals;
  // Number of populated entries in touched_ordinals.
  iree_host_size_t touched_ordinal_count;
  // Allocated entry count in touched_ordinals.
  iree_host_size_t touched_ordinal_capacity;
  // Explicit traversal frames for the active query.
  loom_condition_query_frame_t* frames;
  // Number of active entries in frames.
  iree_host_size_t frame_count;
  // Allocated entry count in frames.
  iree_host_size_t frame_capacity;
} loom_condition_query_t;

// Initializes reusable query state without allocating. |module| and |arena|
// must remain valid for the complete query lifetime. When provided,
// |value_domain| must remain acquired for that lifetime and may be extended by
// queries that encounter rewrite-created values.
void loom_condition_query_initialize(const loom_module_t* module,
                                     loom_local_value_domain_t* value_domain,
                                     iree_arena_allocator_t* arena,
                                     loom_condition_query_t* out_query);

// Initializes a caller-owned fact set over fixed storage.
void loom_condition_fact_set_initialize(
    loom_condition_integer_relation_t* integer_relation_storage,
    iree_host_size_t integer_relation_capacity,
    loom_condition_fact_set_t* out_facts);

// Resets a fact set while retaining caller-owned storage.
void loom_condition_fact_set_reset(loom_condition_fact_set_t* facts);

// Initializes caller-owned semantic refinement storage.
void loom_condition_edge_refinement_set_initialize(
    loom_condition_edge_refinement_t* refinement_storage,
    iree_host_size_t refinement_capacity,
    loom_condition_edge_refinement_set_t* out_refinements);

// Resets semantic refinement storage while retaining caller-owned memory.
void loom_condition_edge_refinement_set_reset(
    loom_condition_edge_refinement_set_t* refinements);

// Derives facts implied by assuming |condition_value| evaluates to
// |assumed_truth|. An otherwise opaque i1 producer contributes the fundamental
// relation that its result equals one or zero on the selected edge; recognized
// producers additionally expose relations over their operands. |fact_table|
// may be NULL to query without ambient value facts. |out_complete| is false
// when caller-owned output storage was too small; returned relations remain a
// conservative subset in that case.
iree_status_t loom_condition_facts_query(
    loom_condition_query_t* query, const loom_value_fact_table_t* fact_table,
    loom_value_id_t condition_value, bool assumed_truth,
    loom_condition_fact_set_t* out_facts, bool* out_complete);

// Derives integer relations and dialect-owned semantic refinements in one
// traversal of a boolean condition expression. Either output may use empty
// caller-owned storage when that fact class is not needed.
iree_status_t loom_condition_facts_query_edge(
    loom_condition_query_t* query, const loom_value_fact_table_t* fact_table,
    loom_value_id_t condition_value, bool assumed_truth,
    loom_condition_fact_set_t* out_facts,
    loom_condition_edge_refinement_set_t* out_refinements, bool* out_complete);

// Appends facts implied by assuming |condition_value| evaluates to
// |assumed_truth| into |inout_facts|. This has the same derivation semantics as
// loom_condition_facts_query but preserves existing relations so callers can
// compose multiple edge conditions.
iree_status_t loom_condition_facts_query_into(
    loom_condition_query_t* query, const loom_value_fact_table_t* fact_table,
    loom_value_id_t condition_value, bool assumed_truth,
    loom_condition_fact_set_t* inout_facts, bool* out_complete);

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

// Attempts to evaluate |queried| from one of the edge-local relations in
// |facts|. Exact scalar values in |fact_table| participate in operand
// identity, so a relation against an SSA constant can prove the equivalent
// relation against a literal. Returns true when the relation is proven either
// true or false and writes that result to |out_result|.
bool loom_condition_fact_set_proves_integer_relation(
    const loom_condition_fact_set_t* facts,
    const loom_value_fact_table_t* fact_table,
    const loom_condition_integer_relation_t* queried, bool* out_result);

// Returns true when each relation implies the other.
bool loom_condition_integer_relations_equivalent(
    const loom_condition_integer_relation_t* left,
    const loom_condition_integer_relation_t* right);

// Computes the strongest common relation preserved by both |left| and |right|.
bool loom_condition_integer_relation_meet(
    const loom_condition_integer_relation_t* left,
    const loom_condition_integer_relation_t* right,
    loom_condition_integer_relation_t* out_relation);

// Attempts to prove that |condition_value| is exact after applying edge-local
// |facts| to the values it depends on. Unsupported condition forms are valid
// and set |out_proven| to false. |fact_table| may be NULL to prove only from
// edge relations.
iree_status_t loom_condition_fact_set_proves_condition(
    loom_condition_query_t* query, const loom_value_fact_table_t* fact_table,
    const loom_condition_fact_set_t* facts, loom_value_id_t condition_value,
    bool* out_condition, bool* out_proven);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_CONDITION_FACTS_H_
