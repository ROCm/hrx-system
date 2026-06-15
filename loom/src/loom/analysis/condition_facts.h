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
#include "loom/analysis/integer_relation.h"
#include "loom/ir/facts.h"
#include "loom/ir/ir.h"
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
  // Caller-owned storage for integer relations.
  loom_condition_integer_relation_t* integer_relations;
  // Number of populated entries in integer_relations.
  iree_host_size_t integer_relation_count;
  // Allocated entry count for integer_relations.
  iree_host_size_t integer_relation_capacity;
} loom_condition_fact_set_t;

// Initializes a caller-owned fact set over fixed storage.
void loom_condition_fact_set_initialize(
    loom_condition_integer_relation_t* integer_relation_storage,
    iree_host_size_t integer_relation_capacity,
    loom_condition_fact_set_t* out_facts);

// Resets a fact set while retaining caller-owned storage.
void loom_condition_fact_set_reset(loom_condition_fact_set_t* facts);

// Derives facts implied by assuming |condition_value| evaluates to
// |assumed_truth|. Unknown condition producers are valid and simply produce an
// empty fact set. |fact_table| may be NULL to query without ambient value
// facts. Returns false if the caller-owned storage was too small or recursion
// was capped; returned relations remain a conservative subset in that case.
bool loom_condition_facts_query(const loom_module_t* module,
                                const loom_value_fact_table_t* fact_table,
                                loom_value_id_t condition_value,
                                bool assumed_truth,
                                loom_condition_fact_set_t* out_facts);

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

// Attempts to prove that |condition_value| is exact after applying edge-local
// |facts| to the values it depends on. Unsupported condition forms are valid
// and return false. |fact_table| may be NULL to prove only from edge relations.
bool loom_condition_fact_set_proves_condition(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_condition_fact_set_t* facts, loom_value_id_t condition_value,
    bool* out_condition);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_CONDITION_FACTS_H_
