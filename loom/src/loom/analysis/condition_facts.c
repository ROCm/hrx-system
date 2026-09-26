// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/condition_facts.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ops/index/carrier.h"
#include "loom/ops/index/compare.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/compare.h"
#include "loom/ops/scalar/ops.h"
#include "loom/util/adaptive_sort.h"

#define LOOM_CONDITION_QUERY_INITIAL_CAPACITY 16

typedef enum loom_condition_query_frame_phase_e {
  LOOM_CONDITION_QUERY_FRAME_DERIVE = 0,
  LOOM_CONDITION_QUERY_FRAME_PROVE_ENTER = 1,
  LOOM_CONDITION_QUERY_FRAME_PROVE_AFTER_LEFT = 2,
  LOOM_CONDITION_QUERY_FRAME_PROVE_AFTER_RIGHT = 3,
} loom_condition_query_frame_phase_t;

struct loom_condition_query_frame_t {
  // Condition SSA value being evaluated.
  loom_value_id_t value_id;
  // Current iterative traversal phase.
  loom_condition_query_frame_phase_t phase;
  // Truth value assumed by fact derivation.
  bool assumed_truth;
};

void loom_condition_query_initialize(const loom_module_t* module,
                                     loom_local_value_domain_t* value_domain,
                                     iree_arena_allocator_t* arena,
                                     loom_condition_query_t* out_query) {
  IREE_ASSERT(value_domain == NULL ||
              (loom_local_value_domain_is_acquired(value_domain) &&
               value_domain->module == module));
  *out_query = (loom_condition_query_t){
      .module = module,
      .value_domain = value_domain,
      .arena = arena,
  };
}

static iree_status_t loom_condition_query_ensure_value_state_capacity(
    loom_condition_query_t* query, iree_host_size_t required_capacity) {
  if (required_capacity <= query->value_state_capacity) {
    return iree_ok_status();
  }
  const iree_host_size_t old_capacity = query->value_state_capacity;
  void* value_states = query->value_states;
  IREE_RETURN_IF_ERROR(
      iree_arena_grow_array(query->arena, old_capacity, required_capacity,
                            sizeof(*query->value_states),
                            &query->value_state_capacity, &value_states));
  query->value_states = (uint8_t*)value_states;
  memset(query->value_states + old_capacity, 0,
         query->value_state_capacity - old_capacity);
  return iree_ok_status();
}

static iree_status_t loom_condition_query_resolve_value_ordinal(
    loom_condition_query_t* query, loom_value_id_t value_id,
    loom_value_ordinal_t* out_value_ordinal) {
  if (query->value_domain != NULL) {
    IREE_RETURN_IF_ERROR(loom_local_value_domain_register_value(
        query->value_domain, query->arena, value_id, out_value_ordinal));
  } else {
    *out_value_ordinal = (loom_value_ordinal_t)value_id;
  }
  return loom_condition_query_ensure_value_state_capacity(
      query, (iree_host_size_t)*out_value_ordinal + 1);
}

static loom_value_ordinal_t loom_condition_query_value_ordinal(
    const loom_condition_query_t* query, loom_value_id_t value_id) {
  return query->value_domain != NULL
             ? loom_local_value_domain_ordinal(query->value_domain, value_id)
             : (loom_value_ordinal_t)value_id;
}

static iree_status_t loom_condition_query_touch_ordinal(
    loom_condition_query_t* query, loom_value_ordinal_t value_ordinal) {
  if (query->value_states[value_ordinal] != 0) {
    return iree_ok_status();
  }
  if (query->touched_ordinal_count >= query->touched_ordinal_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        query->arena, query->touched_ordinal_count,
        query->touched_ordinal_count + 1, sizeof(*query->touched_ordinals),
        &query->touched_ordinal_capacity, (void**)&query->touched_ordinals));
  }
  query->touched_ordinals[query->touched_ordinal_count++] = value_ordinal;
  return iree_ok_status();
}

static iree_status_t loom_condition_query_push_frame(
    loom_condition_query_t* query, loom_value_id_t value_id,
    loom_condition_query_frame_phase_t phase, bool assumed_truth) {
  if (query->frame_count >= query->frame_capacity) {
    const iree_host_size_t minimum_capacity =
        query->frame_capacity == 0 ? LOOM_CONDITION_QUERY_INITIAL_CAPACITY
                                   : query->frame_count + 1;
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(query->arena, query->frame_count,
                              minimum_capacity, sizeof(*query->frames),
                              &query->frame_capacity, (void**)&query->frames));
  }
  query->frames[query->frame_count++] = (loom_condition_query_frame_t){
      .value_id = value_id,
      .phase = phase,
      .assumed_truth = assumed_truth,
  };
  return iree_ok_status();
}

static void loom_condition_query_begin(loom_condition_query_t* query) {
  IREE_ASSERT(query->module != NULL);
  IREE_ASSERT(query->arena != NULL);
  IREE_ASSERT_EQ(query->touched_ordinal_count, 0);
  IREE_ASSERT_EQ(query->frame_count, 0);
}

static void loom_condition_query_end(loom_condition_query_t* query) {
  for (iree_host_size_t i = 0; i < query->touched_ordinal_count; ++i) {
    query->value_states[query->touched_ordinals[i]] = 0;
  }
  query->touched_ordinal_count = 0;
  query->frame_count = 0;
}

void loom_condition_fact_set_initialize(
    loom_condition_integer_relation_t* integer_relation_storage,
    iree_host_size_t integer_relation_capacity,
    loom_condition_fact_set_t* out_facts) {
  *out_facts = (loom_condition_fact_set_t){
      .integer_relations = integer_relation_storage,
      .integer_relation_count = 0,
      .integer_relation_capacity = integer_relation_capacity,
  };
}

void loom_condition_fact_set_reset(loom_condition_fact_set_t* facts) {
  facts->integer_relation_count = 0;
}

void loom_condition_derivation_initialize(
    iree_arena_allocator_t* arena,
    loom_condition_derivation_t* out_derivation) {
  *out_derivation = (loom_condition_derivation_t){
      .arena = arena,
  };
}

void loom_condition_derivation_reset(loom_condition_derivation_t* derivation) {
  loom_condition_fact_set_reset(&derivation->integer_facts);
  derivation->boolean_fact_count = 0;
}

void loom_condition_edge_refinement_set_initialize(
    loom_condition_edge_refinement_t* refinement_storage,
    iree_host_size_t refinement_capacity,
    loom_condition_edge_refinement_set_t* out_refinements) {
  *out_refinements = (loom_condition_edge_refinement_set_t){
      .refinements = refinement_storage,
      .refinement_count = 0,
      .refinement_capacity = refinement_capacity,
  };
}

void loom_condition_edge_refinement_set_reset(
    loom_condition_edge_refinement_set_t* refinements) {
  refinements->refinement_count = 0;
}

static bool loom_condition_edge_refinement_set_append(
    const loom_module_t* module, const loom_op_t* condition_op,
    bool assumed_truth, loom_condition_edge_refinement_set_t* out_refinements) {
  if (out_refinements == NULL) {
    return true;
  }
  const loom_condition_refinement_descriptor_t* descriptor =
      loom_context_resolve_condition_refinement(module->context,
                                                condition_op->kind);
  if (descriptor == NULL) {
    return true;
  }
  loom_condition_refinement_truth_flags_t required_truth_flag =
      assumed_truth ? LOOM_CONDITION_REFINEMENT_TRUTH_TRUE
                    : LOOM_CONDITION_REFINEMENT_TRUTH_FALSE;
  if ((descriptor->truth_flags & required_truth_flag) == 0) {
    return true;
  }
  for (iree_host_size_t i = 0; i < out_refinements->refinement_count; ++i) {
    const loom_condition_edge_refinement_t* existing =
        &out_refinements->refinements[i];
    if (existing->condition_op == condition_op &&
        existing->assumed_truth == assumed_truth) {
      return true;
    }
  }
  if (out_refinements->refinement_count >=
      out_refinements->refinement_capacity) {
    return false;
  }
  out_refinements->refinements[out_refinements->refinement_count++] =
      (loom_condition_edge_refinement_t){
          .condition_op = condition_op,
          .descriptor = descriptor,
          .source = loom_op_const_operands(
              condition_op)[descriptor->source_operand_index],
          .assumed_truth = assumed_truth,
      };
  return true;
}

bool loom_condition_integer_operands_equal(
    loom_condition_integer_operand_t left,
    loom_condition_integer_operand_t right) {
  if (left.kind != right.kind) {
    return false;
  }
  switch (left.kind) {
    case LOOM_CONDITION_INTEGER_OPERAND_VALUE:
      return left.value_id == right.value_id;
    case LOOM_CONDITION_INTEGER_OPERAND_CONSTANT:
      return left.constant == right.constant;
    default:
      return false;
  }
}

static loom_condition_integer_operand_t loom_condition_value_operand(
    loom_value_id_t value_id) {
  return (loom_condition_integer_operand_t){
      .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
      .value_id = value_id,
      .constant = 0,
  };
}

static int loom_condition_integer_operand_compare(
    loom_condition_integer_operand_t left,
    loom_condition_integer_operand_t right) {
  if (left.kind != right.kind) {
    return left.kind < right.kind ? -1 : 1;
  }
  if (left.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
    return (left.value_id > right.value_id) - (left.value_id < right.value_id);
  }
  return (left.constant > right.constant) - (left.constant < right.constant);
}

static loom_condition_integer_relation_t
loom_condition_integer_relation_canonicalize(
    loom_condition_integer_relation_t relation) {
  if (loom_condition_integer_operand_compare(relation.right, relation.left) <
      0) {
    const loom_condition_integer_operand_t left = relation.left;
    relation.left = relation.right;
    relation.right = left;
    relation.relation = loom_symbolic_integer_relation_swap(relation.relation);
  }
  return relation;
}

static bool loom_condition_integer_relation_less(
    const loom_condition_integer_relation_t* left,
    const loom_condition_integer_relation_t* right) {
  int comparison =
      loom_condition_integer_operand_compare(left->left, right->left);
  if (comparison != 0) {
    return comparison < 0;
  }
  comparison =
      loom_condition_integer_operand_compare(left->right, right->right);
  return comparison != 0 ? comparison < 0 : left->relation < right->relation;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_condition_sort_integer_relations,
                          loom_condition_integer_relation_t,
                          loom_condition_integer_relation_less)

static bool loom_condition_boolean_fact_less(
    const loom_condition_boolean_fact_t* left,
    const loom_condition_boolean_fact_t* right) {
  return left->value_id < right->value_id ||
         (left->value_id == right->value_id && left->value < right->value);
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_condition_sort_boolean_facts,
                          loom_condition_boolean_fact_t,
                          loom_condition_boolean_fact_less)

static iree_status_t loom_condition_fact_set_reserve_integer_relations(
    loom_condition_derivation_t* derivation) {
  loom_condition_fact_set_t* facts = &derivation->integer_facts;
  if (facts->integer_relation_count < facts->integer_relation_capacity) {
    return iree_ok_status();
  }
  const iree_host_size_t minimum_capacity =
      facts->integer_relation_capacity == 0
          ? LOOM_CONDITION_QUERY_INITIAL_CAPACITY
          : facts->integer_relation_count + 1;
  return iree_arena_grow_array(
      derivation->arena, facts->integer_relation_count, minimum_capacity,
      sizeof(*facts->integer_relations), &facts->integer_relation_capacity,
      (void**)&facts->integer_relations);
}

static iree_status_t loom_condition_fact_set_append_integer_relation(
    loom_condition_fact_set_t* facts, loom_condition_derivation_t* derivation,
    loom_condition_integer_relation_t relation, bool* out_complete) {
  if (derivation != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_condition_fact_set_reserve_integer_relations(derivation));
    facts->integer_relations[facts->integer_relation_count++] =
        loom_condition_integer_relation_canonicalize(relation);
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < facts->integer_relation_count; ++i) {
    const loom_condition_integer_relation_t* existing =
        &facts->integer_relations[i];
    if (existing->relation == relation.relation &&
        loom_condition_integer_operands_equal(existing->left, relation.left) &&
        loom_condition_integer_operands_equal(existing->right,
                                              relation.right)) {
      return iree_ok_status();
    }
  }
  if (!facts->integer_relations ||
      facts->integer_relation_count >= facts->integer_relation_capacity) {
    *out_complete = false;
    return iree_ok_status();
  }
  facts->integer_relations[facts->integer_relation_count++] = relation;
  return iree_ok_status();
}

static iree_status_t loom_condition_fact_set_append_boolean_fact(
    loom_condition_derivation_t* derivation, loom_value_id_t value_id,
    bool value) {
  if (derivation == NULL) {
    return iree_ok_status();
  }
  if (derivation->boolean_fact_count >= derivation->boolean_fact_capacity) {
    const iree_host_size_t minimum_capacity =
        derivation->boolean_fact_capacity == 0
            ? LOOM_CONDITION_QUERY_INITIAL_CAPACITY
            : derivation->boolean_fact_count + 1;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        derivation->arena, derivation->boolean_fact_count, minimum_capacity,
        sizeof(*derivation->boolean_facts), &derivation->boolean_fact_capacity,
        (void**)&derivation->boolean_facts));
  }
  derivation->boolean_facts[derivation->boolean_fact_count++] =
      (loom_condition_boolean_fact_t){
          .value_id = value_id,
          .value = value,
      };
  return iree_ok_status();
}

static void loom_condition_derivation_finalize(
    loom_condition_derivation_t* derivation) {
  loom_condition_fact_set_t* facts = &derivation->integer_facts;
  loom_condition_sort_integer_relations(facts->integer_relations,
                                        facts->integer_relation_count);
  iree_host_size_t relation_count = 0;
  for (iree_host_size_t i = 0; i < facts->integer_relation_count; ++i) {
    if (relation_count == 0 ||
        loom_condition_integer_relation_less(
            &facts->integer_relations[relation_count - 1],
            &facts->integer_relations[i])) {
      facts->integer_relations[relation_count++] = facts->integer_relations[i];
    }
  }
  facts->integer_relation_count = relation_count;

  loom_condition_sort_boolean_facts(derivation->boolean_facts,
                                    derivation->boolean_fact_count);
  iree_host_size_t boolean_fact_count = 0;
  for (iree_host_size_t i = 0; i < derivation->boolean_fact_count; ++i) {
    const loom_condition_boolean_fact_t fact = derivation->boolean_facts[i];
    if (boolean_fact_count == 0 ||
        loom_condition_boolean_fact_less(
            &derivation->boolean_facts[boolean_fact_count - 1], &fact)) {
      derivation->boolean_facts[boolean_fact_count++] = fact;
    }
  }
  derivation->boolean_fact_count = boolean_fact_count;
}

static loom_value_facts_t loom_condition_lookup_facts(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id) {
  if (!fact_table) {
    return loom_value_facts_unknown();
  }
  return loom_value_fact_table_lookup(fact_table, value_id);
}

static bool loom_condition_value_exact_integer(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    int64_t* out_value) {
  loom_value_facts_t facts = loom_condition_lookup_facts(fact_table, value_id);
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

static bool loom_condition_values_are_non_negative(
    const loom_value_fact_table_t* fact_table, loom_value_id_t left_value,
    loom_value_id_t right_value) {
  return loom_value_facts_is_non_negative(
             loom_condition_lookup_facts(fact_table, left_value)) &&
         loom_value_facts_is_non_negative(
             loom_condition_lookup_facts(fact_table, right_value));
}

static bool loom_condition_value_is_i1(const loom_module_t* module,
                                       loom_value_id_t value_id) {
  if (!module || value_id >= module->values.count) {
    return false;
  }
  loom_type_t type = loom_module_value_type(module, value_id);
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1;
}

static iree_status_t loom_condition_facts_query_opaque_boolean(
    const loom_module_t* module, loom_value_id_t condition_value,
    bool assumed_truth, loom_condition_fact_set_t* out_facts,
    loom_condition_derivation_t* out_derivation, bool* out_complete) {
  if (!loom_condition_value_is_i1(module, condition_value)) {
    return iree_ok_status();
  }
  const loom_condition_integer_relation_t assertion = {
      .relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ,
      .left = loom_condition_value_operand(condition_value),
      .right =
          {
              .kind = LOOM_CONDITION_INTEGER_OPERAND_CONSTANT,
              .value_id = LOOM_VALUE_ID_INVALID,
              .constant = assumed_truth ? 1 : 0,
          },
  };
  return loom_condition_fact_set_append_integer_relation(
      out_facts, out_derivation, assertion, out_complete);
}

static bool loom_condition_facts_exact_bool(loom_value_facts_t facts,
                                            bool* out_value) {
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  const int64_t value = facts.range_lo;
  if (value != 0 && value != 1) {
    return false;
  }
  *out_value = value != 0;
  return true;
}

static bool loom_condition_index_predicate_relation(
    uint8_t predicate, loom_symbolic_integer_relation_t* out_relation,
    bool* out_unsigned_order) {
  *out_unsigned_order = false;
  switch ((loom_index_cmp_predicate_t)predicate) {
    case LOOM_INDEX_CMP_PREDICATE_EQ:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_NE:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_NE;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_SLT:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_SLE:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LE;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_SGT:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GT;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_SGE:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GE;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_ULT:
      *out_unsigned_order = true;
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_ULE:
      *out_unsigned_order = true;
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LE;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_UGT:
      *out_unsigned_order = true;
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GT;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_UGE:
      *out_unsigned_order = true;
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GE;
      return true;
    default:
      return false;
  }
}

static bool loom_condition_scalar_cmpi_predicate_relation(
    uint8_t predicate, loom_symbolic_integer_relation_t* out_relation,
    bool* out_unsigned_order) {
  *out_unsigned_order = false;
  switch ((loom_scalar_cmpi_predicate_t)predicate) {
    case LOOM_SCALAR_CMPI_PREDICATE_EQ:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_NE:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_NE;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_SLT:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_SLE:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LE;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_SGT:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GT;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_SGE:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GE;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_ULT:
      *out_unsigned_order = true;
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_ULE:
      *out_unsigned_order = true;
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LE;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_UGT:
      *out_unsigned_order = true;
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GT;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_UGE:
      *out_unsigned_order = true;
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GE;
      return true;
    default:
      return false;
  }
}

static loom_condition_integer_comparison_t
loom_condition_integer_comparison_from_op(const loom_module_t* module,
                                          const loom_op_t* op) {
  const loom_value_id_t lhs = loom_op_const_operands(op)[0];
  return (loom_condition_integer_comparison_t){
      .kind = op->kind,
      .operand_type =
          loom_type_element_type(loom_module_value_type(module, lhs)),
      .predicate = loom_index_cmp_isa(op) ? loom_index_cmp_predicate(op)
                                          : loom_scalar_cmpi_predicate(op),
      .lhs = lhs,
      .rhs = loom_op_const_operands(op)[1],
  };
}

bool loom_condition_integer_comparison_describe(
    const loom_module_t* module, const loom_op_t* op,
    loom_condition_integer_comparison_t* out_comparison) {
  if (!loom_index_cmp_isa(op) && !loom_scalar_cmpi_isa(op)) {
    return false;
  }
  *out_comparison = loom_condition_integer_comparison_from_op(module, op);
  return true;
}

static bool loom_condition_integer_comparison_same_value_result(
    const loom_condition_integer_comparison_t* comparison, bool* out_result) {
  return comparison->lhs == comparison->rhs &&
         (comparison->kind == LOOM_OP_INDEX_CMP
              ? loom_index_cmp_same_value_result(comparison->predicate,
                                                 out_result)
              : loom_scalar_cmpi_same_value_result(comparison->predicate,
                                                   out_result));
}

bool loom_condition_integer_comparison_evaluate(
    const loom_condition_integer_comparison_t* comparison,
    const loom_fact_context_t* context, const loom_value_facts_t* lhs_facts,
    const loom_value_facts_t* rhs_facts, bool* out_result) {
  if (loom_condition_integer_comparison_same_value_result(comparison,
                                                          out_result)) {
    return true;
  }
  return comparison->kind == LOOM_OP_INDEX_CMP
             ? loom_index_cmp_result_from_facts(
                   context, comparison->operand_type, comparison->predicate,
                   lhs_facts, rhs_facts, out_result)
             : loom_scalar_cmpi_result_from_facts(
                   comparison->operand_type, comparison->predicate, lhs_facts,
                   rhs_facts, out_result);
}

static bool loom_condition_integer_comparison_relation(
    const loom_condition_integer_comparison_t* comparison,
    loom_symbolic_integer_relation_t* out_relation, bool* out_unsigned_order) {
  return comparison->kind == LOOM_OP_INDEX_CMP
             ? loom_condition_index_predicate_relation(
                   comparison->predicate, out_relation, out_unsigned_order)
             : loom_condition_scalar_cmpi_predicate_relation(
                   loom_scalar_cmpi_range_predicate(comparison->operand_type,
                                                    comparison->predicate),
                   out_relation, out_unsigned_order);
}

static iree_status_t loom_condition_facts_query_integer_compare(
    const loom_module_t* module, const loom_op_t* op,
    loom_condition_fact_set_t* facts,
    loom_condition_derivation_t* out_derivation,
    const loom_value_fact_table_t* fact_table, bool assumed_truth,
    bool* out_complete) {
  const loom_condition_integer_comparison_t comparison =
      loom_condition_integer_comparison_from_op(module, op);
  const loom_value_id_t left_value = comparison.lhs;
  const loom_value_id_t right_value = comparison.rhs;
  loom_symbolic_integer_relation_t relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
  bool unsigned_order = false;
  const bool has_relation = loom_condition_integer_comparison_relation(
      &comparison, &relation, &unsigned_order);
  if (!has_relation) {
    return iree_ok_status();
  }
  if (!assumed_truth) {
    relation = loom_symbolic_integer_relation_invert(relation);
  }

  if (unsigned_order && !loom_condition_values_are_non_negative(
                            fact_table, left_value, right_value)) {
    const bool ascending = relation == LOOM_SYMBOLIC_INTEGER_RELATION_LT ||
                           relation == LOOM_SYMBOLIC_INTEGER_RELATION_LE;
    const loom_value_id_t lower = ascending ? left_value : right_value;
    const loom_value_id_t upper = ascending ? right_value : left_value;
    const loom_value_facts_t upper_facts =
        loom_condition_lookup_facts(fact_table, upper);
    if (!loom_value_facts_is_non_negative(upper_facts)) {
      return iree_ok_status();
    }
    // A bound with the carrier's sign bit set admits negative signed values.
    // Targetless address facts retain their mathematical i64 source domain.
    if (comparison.kind == LOOM_OP_INDEX_CMP &&
        !loom_index_value_facts_fit_signed_target_carrier(
            fact_table ? &fact_table->context : NULL, comparison.operand_type,
            upper_facts)) {
      return iree_ok_status();
    }
    // On this edge lower <=u upper, where upper has no sign bit. Therefore
    // lower is nonnegative and the retained signed order is equivalent.
    const loom_condition_integer_relation_t nonnegative = {
        .relation = LOOM_SYMBOLIC_INTEGER_RELATION_GE,
        .left = loom_condition_value_operand(lower),
        .right = {.kind = LOOM_CONDITION_INTEGER_OPERAND_CONSTANT,
                  .value_id = LOOM_VALUE_ID_INVALID,
                  .constant = 0},
    };
    IREE_RETURN_IF_ERROR(loom_condition_fact_set_append_integer_relation(
        facts, out_derivation, nonnegative, out_complete));
  }

  const loom_condition_integer_relation_t assertion = {
      .relation = relation,
      .left = loom_condition_value_operand(left_value),
      .right = loom_condition_value_operand(right_value),
  };
  return loom_condition_fact_set_append_integer_relation(
      facts, out_derivation, assertion, out_complete);
}

enum {
  LOOM_CONDITION_DERIVATION_VISITED_FALSE = 1u << 0,
  LOOM_CONDITION_DERIVATION_VISITED_TRUE = 1u << 1,
};

static iree_status_t loom_condition_query_push_derivation(
    loom_condition_query_t* query, loom_value_id_t value_id,
    bool assumed_truth) {
  if (value_id >= query->module->values.count) {
    return iree_ok_status();
  }
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(loom_condition_query_resolve_value_ordinal(
      query, value_id, &value_ordinal));
  const uint8_t visited_bit = assumed_truth
                                  ? LOOM_CONDITION_DERIVATION_VISITED_TRUE
                                  : LOOM_CONDITION_DERIVATION_VISITED_FALSE;
  if ((query->value_states[value_ordinal] & visited_bit) != 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_condition_query_touch_ordinal(query, value_ordinal));
  query->value_states[value_ordinal] |= visited_bit;
  return loom_condition_query_push_frame(
      query, value_id, LOOM_CONDITION_QUERY_FRAME_DERIVE, assumed_truth);
}

static bool loom_condition_boolean_op_has_i1_signature(
    const loom_module_t* module, const loom_op_t* op, loom_value_id_t lhs,
    loom_value_id_t rhs) {
  return loom_condition_value_is_i1(module, loom_op_const_results(op)[0]) &&
         loom_condition_value_is_i1(module, lhs) &&
         loom_condition_value_is_i1(module, rhs);
}

static iree_status_t loom_condition_facts_process_derivation(
    loom_condition_query_t* query, const loom_value_fact_table_t* fact_table,
    const loom_condition_query_frame_t* frame,
    loom_condition_fact_set_t* out_facts,
    loom_condition_derivation_t* out_derivation,
    loom_condition_edge_refinement_set_t* out_refinements, bool* out_complete) {
  const loom_module_t* module = query->module;
  if (loom_condition_value_is_i1(module, frame->value_id)) {
    IREE_RETURN_IF_ERROR(loom_condition_fact_set_append_boolean_fact(
        out_derivation, frame->value_id, frame->assumed_truth));
  }
  const loom_value_t* value = loom_module_value(module, frame->value_id);
  if (loom_value_is_block_arg(value)) {
    return loom_condition_facts_query_opaque_boolean(
        module, frame->value_id, frame->assumed_truth, out_facts,
        out_derivation, out_complete);
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op) {
    return loom_condition_facts_query_opaque_boolean(
        module, frame->value_id, frame->assumed_truth, out_facts,
        out_derivation, out_complete);
  }

  *out_complete &= loom_condition_edge_refinement_set_append(
      module, defining_op, frame->assumed_truth, out_refinements);

  switch (defining_op->kind) {
    case LOOM_OP_INDEX_CMP:
    case LOOM_OP_SCALAR_CMPI:
      return loom_condition_facts_query_integer_compare(
          module, defining_op, out_facts, out_derivation, fact_table,
          frame->assumed_truth, out_complete);
    case LOOM_OP_SCALAR_ANDI: {
      const loom_value_id_t lhs = loom_scalar_andi_lhs(defining_op);
      const loom_value_id_t rhs = loom_scalar_andi_rhs(defining_op);
      if (!loom_condition_boolean_op_has_i1_signature(module, defining_op, lhs,
                                                      rhs)) {
        return iree_ok_status();
      }
      if (frame->assumed_truth) {
        IREE_RETURN_IF_ERROR(
            loom_condition_query_push_derivation(query, rhs, true));
        return loom_condition_query_push_derivation(query, lhs, true);
      }
      bool exact_truth = false;
      if (loom_condition_facts_exact_bool(
              loom_condition_lookup_facts(fact_table, lhs), &exact_truth)) {
        return exact_truth
                   ? loom_condition_query_push_derivation(query, rhs, false)
                   : iree_ok_status();
      }
      if (loom_condition_facts_exact_bool(
              loom_condition_lookup_facts(fact_table, rhs), &exact_truth)) {
        return exact_truth
                   ? loom_condition_query_push_derivation(query, lhs, false)
                   : iree_ok_status();
      }
      return iree_ok_status();
    }
    case LOOM_OP_SCALAR_ORI: {
      const loom_value_id_t lhs = loom_scalar_ori_lhs(defining_op);
      const loom_value_id_t rhs = loom_scalar_ori_rhs(defining_op);
      if (!loom_condition_boolean_op_has_i1_signature(module, defining_op, lhs,
                                                      rhs)) {
        return iree_ok_status();
      }
      if (!frame->assumed_truth) {
        IREE_RETURN_IF_ERROR(
            loom_condition_query_push_derivation(query, rhs, false));
        return loom_condition_query_push_derivation(query, lhs, false);
      }
      bool exact_truth = false;
      if (loom_condition_facts_exact_bool(
              loom_condition_lookup_facts(fact_table, lhs), &exact_truth)) {
        return !exact_truth
                   ? loom_condition_query_push_derivation(query, rhs, true)
                   : iree_ok_status();
      }
      if (loom_condition_facts_exact_bool(
              loom_condition_lookup_facts(fact_table, rhs), &exact_truth)) {
        return !exact_truth
                   ? loom_condition_query_push_derivation(query, lhs, true)
                   : iree_ok_status();
      }
      return iree_ok_status();
    }
    case LOOM_OP_SCALAR_XORI: {
      const loom_value_id_t lhs = loom_scalar_xori_lhs(defining_op);
      const loom_value_id_t rhs = loom_scalar_xori_rhs(defining_op);
      if (!loom_condition_boolean_op_has_i1_signature(module, defining_op, lhs,
                                                      rhs)) {
        return iree_ok_status();
      }
      bool exact_truth = false;
      if (loom_condition_facts_exact_bool(
              loom_condition_lookup_facts(fact_table, lhs), &exact_truth)) {
        return loom_condition_query_push_derivation(
            query, rhs, frame->assumed_truth != exact_truth);
      }
      if (loom_condition_facts_exact_bool(
              loom_condition_lookup_facts(fact_table, rhs), &exact_truth)) {
        return loom_condition_query_push_derivation(
            query, lhs, frame->assumed_truth != exact_truth);
      }
      return iree_ok_status();
    }
    default:
      return loom_condition_facts_query_opaque_boolean(
          module, frame->value_id, frame->assumed_truth, out_facts,
          out_derivation, out_complete);
  }
}

static iree_status_t loom_condition_facts_query_impl(
    loom_condition_query_t* query, const loom_value_fact_table_t* fact_table,
    loom_value_id_t condition_value, bool assumed_truth,
    loom_condition_fact_set_t* out_facts,
    loom_condition_derivation_t* out_derivation,
    loom_condition_edge_refinement_set_t* out_refinements, bool* out_complete) {
  *out_complete = true;
  loom_condition_query_begin(query);
  iree_status_t status = iree_ok_status();
  if (condition_value < query->module->values.count) {
    const loom_condition_query_frame_t root_frame = {
        .value_id = condition_value,
        .phase = LOOM_CONDITION_QUERY_FRAME_DERIVE,
        .assumed_truth = assumed_truth,
    };
    status = loom_condition_facts_process_derivation(
        query, fact_table, &root_frame, out_facts, out_derivation,
        out_refinements, out_complete);
  }
  while (iree_status_is_ok(status) && query->frame_count > 0) {
    const loom_condition_query_frame_t frame =
        query->frames[--query->frame_count];
    status = loom_condition_facts_process_derivation(
        query, fact_table, &frame, out_facts, out_derivation, out_refinements,
        out_complete);
  }
  loom_condition_query_end(query);
  return status;
}

iree_status_t loom_condition_facts_query(
    loom_condition_query_t* query, const loom_value_fact_table_t* fact_table,
    loom_value_id_t condition_value, bool assumed_truth,
    loom_condition_fact_set_t* out_facts, bool* out_complete) {
  loom_condition_fact_set_reset(out_facts);
  return loom_condition_facts_query_impl(
      query, fact_table, condition_value, assumed_truth, out_facts,
      /*out_derivation=*/NULL, /*out_refinements=*/NULL, out_complete);
}

iree_status_t loom_condition_facts_query_complete(
    loom_condition_query_t* query, const loom_value_fact_table_t* fact_table,
    loom_value_id_t condition_value, bool assumed_truth,
    loom_condition_derivation_t* out_derivation) {
  loom_condition_derivation_reset(out_derivation);
  bool complete = false;
  iree_status_t status = loom_condition_facts_query_impl(
      query, fact_table, condition_value, assumed_truth,
      &out_derivation->integer_facts, out_derivation,
      /*out_refinements=*/NULL, &complete);
  if (iree_status_is_ok(status)) {
    loom_condition_derivation_finalize(out_derivation);
  }
  return status;
}

iree_status_t loom_condition_facts_query_conjunction_complete(
    loom_condition_query_t* query, const loom_value_fact_table_t* fact_table,
    const loom_condition_assumption_t* assumptions,
    iree_host_size_t assumption_count,
    loom_condition_derivation_t* out_derivation) {
  loom_condition_derivation_reset(out_derivation);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < assumption_count && iree_status_is_ok(status); ++i) {
    bool complete = false;
    status = loom_condition_facts_query_impl(
        query, fact_table, assumptions[i].condition,
        assumptions[i].assumed_truth, &out_derivation->integer_facts,
        out_derivation,
        /*out_refinements=*/NULL, &complete);
  }
  if (iree_status_is_ok(status)) {
    loom_condition_derivation_finalize(out_derivation);
  }
  return status;
}

iree_status_t loom_condition_facts_query_edge(
    loom_condition_query_t* query, const loom_value_fact_table_t* fact_table,
    loom_value_id_t condition_value, bool assumed_truth,
    loom_condition_fact_set_t* out_facts,
    loom_condition_edge_refinement_set_t* out_refinements, bool* out_complete) {
  loom_condition_fact_set_reset(out_facts);
  loom_condition_edge_refinement_set_reset(out_refinements);
  return loom_condition_facts_query_impl(
      query, fact_table, condition_value, assumed_truth, out_facts,
      /*out_derivation=*/NULL, out_refinements, out_complete);
}

iree_status_t loom_condition_facts_query_into(
    loom_condition_query_t* query, const loom_value_fact_table_t* fact_table,
    loom_value_id_t condition_value, bool assumed_truth,
    loom_condition_fact_set_t* inout_facts, bool* out_complete) {
  return loom_condition_facts_query_impl(
      query, fact_table, condition_value, assumed_truth, inout_facts,
      /*out_derivation=*/NULL, /*out_refinements=*/NULL, out_complete);
}

static loom_value_facts_t loom_condition_edge_value_facts(
    const loom_value_fact_table_t* fact_table,
    const loom_condition_fact_resolver_t* resolver, loom_value_id_t value_id) {
  loom_value_facts_t value_facts =
      loom_condition_lookup_facts(fact_table, value_id);
  if (resolver != NULL && resolver->apply_to_value_facts != NULL) {
    (void)resolver->apply_to_value_facts(resolver->user_data, fact_table,
                                         value_id, &value_facts);
  }
  return value_facts;
}

static bool loom_condition_fact_resolver_proves_integer_comparison(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_condition_fact_resolver_t* resolver,
    const loom_op_t* defining_op, bool* out_condition) {
  const loom_condition_integer_comparison_t comparison =
      loom_condition_integer_comparison_from_op(module, defining_op);
  if (loom_condition_integer_comparison_same_value_result(&comparison,
                                                          out_condition)) {
    return true;
  }

  loom_condition_integer_relation_t relation = {
      .left = loom_condition_value_operand(comparison.lhs),
      .right = loom_condition_value_operand(comparison.rhs),
  };
  bool unsigned_order = false;
  if (loom_condition_integer_comparison_relation(
          &comparison, &relation.relation, &unsigned_order) &&
      (!unsigned_order || loom_condition_values_are_non_negative(
                              fact_table, comparison.lhs, comparison.rhs)) &&
      resolver != NULL && resolver->proves_integer_relation != NULL &&
      resolver->proves_integer_relation(resolver->user_data, fact_table,
                                        &relation, out_condition)) {
    return true;
  }

  const loom_value_facts_t lhs_facts =
      loom_condition_edge_value_facts(fact_table, resolver, comparison.lhs);
  const loom_value_facts_t rhs_facts =
      loom_condition_edge_value_facts(fact_table, resolver, comparison.rhs);
  return loom_condition_integer_comparison_evaluate(
      &comparison, fact_table ? &fact_table->context : NULL, &lhs_facts,
      &rhs_facts, out_condition);
}

typedef enum loom_condition_proof_state_e {
  LOOM_CONDITION_PROOF_UNVISITED = 0,
  LOOM_CONDITION_PROOF_PENDING = 1,
  LOOM_CONDITION_PROOF_UNKNOWN = 2,
  LOOM_CONDITION_PROOF_FALSE = 3,
  LOOM_CONDITION_PROOF_TRUE = 4,
} loom_condition_proof_state_t;

static bool loom_condition_proof_state_is_known(
    loom_condition_proof_state_t state, bool* out_condition) {
  if (state == LOOM_CONDITION_PROOF_FALSE) {
    *out_condition = false;
    return true;
  }
  if (state == LOOM_CONDITION_PROOF_TRUE) {
    *out_condition = true;
    return true;
  }
  *out_condition = false;
  return false;
}

static void loom_condition_query_finish_proof_frame(
    loom_condition_query_t* query, loom_condition_proof_state_t state) {
  IREE_ASSERT_GT(query->frame_count, 0);
  const loom_value_id_t value_id =
      query->frames[query->frame_count - 1].value_id;
  const loom_value_ordinal_t value_ordinal =
      loom_condition_query_value_ordinal(query, value_id);
  query->value_states[value_ordinal] = (uint8_t)state;
  --query->frame_count;
}

static iree_status_t loom_condition_query_push_proof(
    loom_condition_query_t* query, loom_value_id_t value_id) {
  if (value_id >= query->module->values.count) {
    return iree_ok_status();
  }
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(loom_condition_query_resolve_value_ordinal(
      query, value_id, &value_ordinal));
  if (query->value_states[value_ordinal] != LOOM_CONDITION_PROOF_UNVISITED) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_condition_query_touch_ordinal(query, value_ordinal));
  query->value_states[value_ordinal] = LOOM_CONDITION_PROOF_PENDING;
  return loom_condition_query_push_frame(query, value_id,
                                         LOOM_CONDITION_QUERY_FRAME_PROVE_ENTER,
                                         /*assumed_truth=*/false);
}

static const loom_op_t* loom_condition_query_proof_defining_op(
    const loom_condition_query_t* query, loom_value_id_t value_id) {
  if (value_id >= query->module->values.count) {
    return NULL;
  }
  const loom_value_t* value = loom_module_value(query->module, value_id);
  if (loom_value_is_block_arg(value) || loom_value_def_index(value) != 0) {
    return NULL;
  }
  return loom_value_def_op(value);
}

static bool loom_condition_query_boolean_operands(
    const loom_condition_query_t* query, const loom_op_t* defining_op,
    loom_value_id_t* out_lhs, loom_value_id_t* out_rhs) {
  switch (defining_op->kind) {
    case LOOM_OP_SCALAR_ANDI:
      *out_lhs = loom_scalar_andi_lhs(defining_op);
      *out_rhs = loom_scalar_andi_rhs(defining_op);
      break;
    case LOOM_OP_SCALAR_ORI:
      *out_lhs = loom_scalar_ori_lhs(defining_op);
      *out_rhs = loom_scalar_ori_rhs(defining_op);
      break;
    case LOOM_OP_SCALAR_XORI:
      *out_lhs = loom_scalar_xori_lhs(defining_op);
      *out_rhs = loom_scalar_xori_rhs(defining_op);
      break;
    default:
      return false;
  }
  return loom_condition_boolean_op_has_i1_signature(query->module, defining_op,
                                                    *out_lhs, *out_rhs);
}

static loom_condition_proof_state_t loom_condition_query_evaluate_direct_proof(
    const loom_condition_query_t* query,
    const loom_value_fact_table_t* fact_table,
    const loom_condition_fact_resolver_t* resolver, loom_value_id_t value_id,
    bool* out_requires_composition) {
  *out_requires_composition = false;
  bool condition = false;
  if (resolver != NULL && resolver->query_boolean != NULL &&
      resolver->query_boolean(resolver->user_data, value_id, &condition)) {
    return condition ? LOOM_CONDITION_PROOF_TRUE : LOOM_CONDITION_PROOF_FALSE;
  }
  if (loom_condition_facts_exact_bool(
          loom_condition_edge_value_facts(fact_table, resolver, value_id),
          &condition)) {
    return condition ? LOOM_CONDITION_PROOF_TRUE : LOOM_CONDITION_PROOF_FALSE;
  }

  const loom_op_t* defining_op =
      loom_condition_query_proof_defining_op(query, value_id);
  if (!defining_op) {
    return LOOM_CONDITION_PROOF_UNKNOWN;
  }

  bool proven = false;
  switch (defining_op->kind) {
    case LOOM_OP_INDEX_CMP:
    case LOOM_OP_SCALAR_CMPI:
      proven = loom_condition_fact_resolver_proves_integer_comparison(
          query->module, fact_table, resolver, defining_op, &condition);
      break;
    default: {
      loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
      loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
      *out_requires_composition =
          loom_condition_query_boolean_operands(query, defining_op, &lhs, &rhs);
      break;
    }
  }
  return proven ? (condition ? LOOM_CONDITION_PROOF_TRUE
                             : LOOM_CONDITION_PROOF_FALSE)
                : LOOM_CONDITION_PROOF_UNKNOWN;
}

static iree_status_t loom_condition_query_enter_proof_frame(
    loom_condition_query_t* query, const loom_value_fact_table_t* fact_table,
    const loom_condition_fact_resolver_t* resolver) {
  loom_condition_query_frame_t* frame = &query->frames[query->frame_count - 1];
  const loom_op_t* defining_op =
      loom_condition_query_proof_defining_op(query, frame->value_id);
  bool requires_composition = false;
  const loom_condition_proof_state_t direct_state =
      loom_condition_query_evaluate_direct_proof(
          query, fact_table, resolver, frame->value_id, &requires_composition);
  if (!requires_composition) {
    loom_condition_query_finish_proof_frame(query, direct_state);
    return iree_ok_status();
  }

  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  const bool has_boolean_operands =
      loom_condition_query_boolean_operands(query, defining_op, &lhs, &rhs);
  IREE_ASSERT(has_boolean_operands);
  (void)has_boolean_operands;
  frame->phase = LOOM_CONDITION_QUERY_FRAME_PROVE_AFTER_LEFT;
  return loom_condition_query_push_proof(query, lhs);
}

static iree_status_t loom_condition_query_continue_proof_frame(
    loom_condition_query_t* query) {
  loom_condition_query_frame_t* frame = &query->frames[query->frame_count - 1];
  const loom_op_t* defining_op =
      loom_condition_query_proof_defining_op(query, frame->value_id);
  IREE_ASSERT(defining_op != NULL);

  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  const bool has_boolean_operands =
      loom_condition_query_boolean_operands(query, defining_op, &lhs, &rhs);
  IREE_ASSERT(has_boolean_operands);
  (void)has_boolean_operands;

  bool lhs_value = false;
  const loom_value_ordinal_t lhs_ordinal =
      loom_condition_query_value_ordinal(query, lhs);
  const bool lhs_known = loom_condition_proof_state_is_known(
      (loom_condition_proof_state_t)query->value_states[lhs_ordinal],
      &lhs_value);
  if (frame->phase == LOOM_CONDITION_QUERY_FRAME_PROVE_AFTER_LEFT) {
    if ((defining_op->kind == LOOM_OP_SCALAR_ANDI && lhs_known && !lhs_value) ||
        (defining_op->kind == LOOM_OP_SCALAR_ORI && lhs_known && lhs_value)) {
      loom_condition_query_finish_proof_frame(
          query,
          lhs_value ? LOOM_CONDITION_PROOF_TRUE : LOOM_CONDITION_PROOF_FALSE);
      return iree_ok_status();
    }
    frame->phase = LOOM_CONDITION_QUERY_FRAME_PROVE_AFTER_RIGHT;
    return loom_condition_query_push_proof(query, rhs);
  }

  bool rhs_value = false;
  const loom_value_ordinal_t rhs_ordinal =
      loom_condition_query_value_ordinal(query, rhs);
  const bool rhs_known = loom_condition_proof_state_is_known(
      (loom_condition_proof_state_t)query->value_states[rhs_ordinal],
      &rhs_value);
  loom_condition_proof_state_t result = LOOM_CONDITION_PROOF_UNKNOWN;
  switch (defining_op->kind) {
    case LOOM_OP_SCALAR_ANDI:
      if ((lhs_known && !lhs_value) || (rhs_known && !rhs_value)) {
        result = LOOM_CONDITION_PROOF_FALSE;
      } else if (lhs_known && rhs_known) {
        result = LOOM_CONDITION_PROOF_TRUE;
      }
      break;
    case LOOM_OP_SCALAR_ORI:
      if ((lhs_known && lhs_value) || (rhs_known && rhs_value)) {
        result = LOOM_CONDITION_PROOF_TRUE;
      } else if (lhs_known && rhs_known) {
        result = LOOM_CONDITION_PROOF_FALSE;
      }
      break;
    case LOOM_OP_SCALAR_XORI:
      if (lhs_known && rhs_known) {
        result = lhs_value != rhs_value ? LOOM_CONDITION_PROOF_TRUE
                                        : LOOM_CONDITION_PROOF_FALSE;
      }
      break;
    default:
      break;
  }
  loom_condition_query_finish_proof_frame(query, result);
  return iree_ok_status();
}

iree_status_t loom_condition_fact_resolver_proves_condition(
    loom_condition_query_t* query, const loom_value_fact_table_t* fact_table,
    const loom_condition_fact_resolver_t* resolver,
    loom_value_id_t condition_value, bool* out_condition, bool* out_proven) {
  *out_condition = false;
  *out_proven = false;
  loom_condition_query_begin(query);
  bool requires_composition = false;
  const loom_condition_proof_state_t direct_state =
      loom_condition_query_evaluate_direct_proof(
          query, fact_table, resolver, condition_value, &requires_composition);
  if (!requires_composition) {
    *out_proven =
        loom_condition_proof_state_is_known(direct_state, out_condition);
    loom_condition_query_end(query);
    return iree_ok_status();
  }
  iree_status_t status =
      loom_condition_query_push_proof(query, condition_value);
  while (iree_status_is_ok(status) && query->frame_count > 0) {
    const loom_condition_query_frame_phase_t phase =
        query->frames[query->frame_count - 1].phase;
    if (phase == LOOM_CONDITION_QUERY_FRAME_PROVE_ENTER) {
      status =
          loom_condition_query_enter_proof_frame(query, fact_table, resolver);
    } else {
      status = loom_condition_query_continue_proof_frame(query);
    }
  }
  if (iree_status_is_ok(status) &&
      condition_value < query->module->values.count) {
    const loom_value_ordinal_t condition_ordinal =
        loom_condition_query_value_ordinal(query, condition_value);
    *out_proven = loom_condition_proof_state_is_known(
        (loom_condition_proof_state_t)query->value_states[condition_ordinal],
        out_condition);
  }
  loom_condition_query_end(query);
  return status;
}

static bool loom_condition_flat_facts_apply_to_value_facts(
    const void* user_data, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_value_facts_t* inout_facts) {
  return loom_condition_fact_set_apply_to_value_facts(
      (const loom_condition_fact_set_t*)user_data, fact_table, value_id,
      inout_facts);
}

static bool loom_condition_flat_facts_prove_integer_relation(
    const void* user_data, const loom_value_fact_table_t* fact_table,
    const loom_condition_integer_relation_t* queried, bool* out_result) {
  return loom_condition_fact_set_proves_integer_relation(
      (const loom_condition_fact_set_t*)user_data, fact_table, queried,
      out_result);
}

iree_status_t loom_condition_fact_set_proves_condition(
    loom_condition_query_t* query, const loom_value_fact_table_t* fact_table,
    const loom_condition_fact_set_t* facts, loom_value_id_t condition_value,
    bool* out_condition, bool* out_proven) {
  const loom_condition_fact_resolver_t resolver = {
      .user_data = facts,
      .apply_to_value_facts = loom_condition_flat_facts_apply_to_value_facts,
      .proves_integer_relation =
          loom_condition_flat_facts_prove_integer_relation,
  };
  return loom_condition_fact_resolver_proves_condition(
      query, fact_table, facts != NULL ? &resolver : NULL, condition_value,
      out_condition, out_proven);
}

static bool loom_condition_relation_to_predicate_kind(
    loom_symbolic_integer_relation_t relation, uint8_t* out_kind) {
  switch (relation) {
    case LOOM_SYMBOLIC_INTEGER_RELATION_EQ:
      *out_kind = LOOM_PREDICATE_EQ;
      return true;
    case LOOM_SYMBOLIC_INTEGER_RELATION_NE:
      *out_kind = LOOM_PREDICATE_NE;
      return true;
    case LOOM_SYMBOLIC_INTEGER_RELATION_LT:
      *out_kind = LOOM_PREDICATE_LT;
      return true;
    case LOOM_SYMBOLIC_INTEGER_RELATION_LE:
      *out_kind = LOOM_PREDICATE_LE;
      return true;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GT:
      *out_kind = LOOM_PREDICATE_GT;
      return true;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GE:
      *out_kind = LOOM_PREDICATE_GE;
      return true;
    default:
      return false;
  }
}

static bool loom_condition_operand_matches_value(
    const loom_value_fact_table_t* fact_table,
    loom_condition_integer_operand_t operand, loom_value_id_t value_id) {
  return operand.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE &&
         loom_value_fact_table_query_identity(fact_table, operand.value_id) ==
             loom_value_fact_table_query_identity(fact_table, value_id);
}

bool loom_condition_integer_relation_make_predicate_for_value(
    const loom_condition_integer_relation_t* relation,
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_predicate_t* out_predicate) {
  loom_symbolic_integer_relation_t normalized_relation = relation->relation;
  loom_condition_integer_operand_t other = {0};
  if (loom_condition_operand_matches_value(fact_table, relation->left,
                                           value_id)) {
    other = relation->right;
  } else if (loom_condition_operand_matches_value(fact_table, relation->right,
                                                  value_id)) {
    other = relation->left;
    normalized_relation =
        loom_symbolic_integer_relation_swap(normalized_relation);
  } else {
    return false;
  }

  uint8_t predicate_kind = 0;
  if (!loom_condition_relation_to_predicate_kind(normalized_relation,
                                                 &predicate_kind)) {
    return false;
  }

  *out_predicate = (loom_predicate_t){
      .kind = predicate_kind,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_NONE, LOOM_PRED_ARG_NONE},
      .args = {value_id, 0, 0},
  };
  switch (other.kind) {
    case LOOM_CONDITION_INTEGER_OPERAND_CONSTANT:
      out_predicate->arg_tags[1] = LOOM_PRED_ARG_CONST;
      out_predicate->args[1] = other.constant;
      return true;
    case LOOM_CONDITION_INTEGER_OPERAND_VALUE: {
      int64_t constant = 0;
      if (loom_condition_value_exact_integer(fact_table, other.value_id,
                                             &constant)) {
        out_predicate->arg_tags[1] = LOOM_PRED_ARG_CONST;
        out_predicate->args[1] = constant;
      } else {
        out_predicate->arg_tags[1] = LOOM_PRED_ARG_VALUE;
        out_predicate->args[1] = other.value_id;
      }
      return true;
    }
    default:
      return false;
  }
}

bool loom_condition_integer_relation_apply_to_value_facts(
    const loom_condition_integer_relation_t* relation,
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_value_facts_t* inout_facts) {
  loom_predicate_t predicate = {0};
  if (!loom_condition_integer_relation_make_predicate_for_value(
          relation, fact_table, value_id, &predicate)) {
    return false;
  }
  if (predicate.arg_tags[1] == LOOM_PRED_ARG_CONST) {
    loom_value_facts_apply_predicate(inout_facts, &predicate);
    return true;
  }
  return loom_value_facts_refine_relation(
      predicate.kind, *inout_facts,
      loom_value_fact_table_lookup(fact_table,
                                   (loom_value_id_t)predicate.args[1]),
      inout_facts, NULL);
}

bool loom_condition_fact_set_apply_to_value_facts(
    const loom_condition_fact_set_t* facts,
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_value_facts_t* inout_facts) {
  bool applied = false;
  for (iree_host_size_t i = 0; i < facts->integer_relation_count; ++i) {
    applied |= loom_condition_integer_relation_apply_to_value_facts(
        &facts->integer_relations[i], fact_table, value_id, inout_facts);
  }
  return applied;
}

static bool loom_condition_integer_operand_exact_i64(
    const loom_value_fact_table_t* fact_table,
    loom_condition_integer_operand_t operand, int64_t* out_value) {
  switch (operand.kind) {
    case LOOM_CONDITION_INTEGER_OPERAND_VALUE:
      return loom_condition_value_exact_integer(fact_table, operand.value_id,
                                                out_value);
    case LOOM_CONDITION_INTEGER_OPERAND_CONSTANT:
      *out_value = operand.constant;
      return true;
    default:
      return false;
  }
}

static bool loom_condition_integer_operands_equivalent_with_facts(
    loom_condition_integer_operand_t left,
    loom_condition_integer_operand_t right,
    const loom_value_fact_table_t* fact_table) {
  if (loom_condition_integer_operands_equal(left, right)) {
    return true;
  }
  if (left.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE &&
      right.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE &&
      loom_value_fact_table_query_identity(fact_table, left.value_id) ==
          loom_value_fact_table_query_identity(fact_table, right.value_id)) {
    return true;
  }
  int64_t left_value = 0;
  int64_t right_value = 0;
  return fact_table != NULL &&
         loom_condition_integer_operand_exact_i64(fact_table, left,
                                                  &left_value) &&
         loom_condition_integer_operand_exact_i64(fact_table, right,
                                                  &right_value) &&
         left_value == right_value;
}

bool loom_condition_integer_relation_implies(
    const loom_condition_integer_relation_t* known,
    const loom_condition_integer_relation_t* queried, bool* out_result) {
  if (loom_condition_integer_operands_equal(known->left, queried->left) &&
      loom_condition_integer_operands_equal(known->right, queried->right)) {
    return loom_symbolic_integer_relation_implies(
        known->relation, queried->relation, out_result);
  }

  if (loom_condition_integer_operands_equal(known->left, queried->right) &&
      loom_condition_integer_operands_equal(known->right, queried->left)) {
    return loom_symbolic_integer_relation_implies(
        loom_symbolic_integer_relation_swap(known->relation), queried->relation,
        out_result);
  }

  return false;
}

// Bits represent the possible less/equal/greater comparison outcomes.
static const uint8_t loom_condition_integer_relation_outcomes[] = {
    [LOOM_SYMBOLIC_INTEGER_RELATION_EQ] = 2,
    [LOOM_SYMBOLIC_INTEGER_RELATION_NE] = 5,
    [LOOM_SYMBOLIC_INTEGER_RELATION_LT] = 1,
    [LOOM_SYMBOLIC_INTEGER_RELATION_LE] = 3,
    [LOOM_SYMBOLIC_INTEGER_RELATION_GT] = 4,
    [LOOM_SYMBOLIC_INTEGER_RELATION_GE] = 6,
};

bool loom_condition_fact_set_proves_integer_relation(
    const loom_condition_fact_set_t* facts,
    const loom_value_fact_table_t* fact_table,
    const loom_condition_integer_relation_t* queried, bool* out_result) {
  if (!facts) {
    return false;
  }
  uint8_t outcomes = 7;
  for (iree_host_size_t i = 0; i < facts->integer_relation_count; ++i) {
    const loom_condition_integer_relation_t* known =
        &facts->integer_relations[i];
    if (loom_condition_integer_operands_equivalent_with_facts(
            known->left, queried->left, fact_table) &&
        loom_condition_integer_operands_equivalent_with_facts(
            known->right, queried->right, fact_table)) {
      outcomes &= loom_condition_integer_relation_outcomes[known->relation];
    } else if (loom_condition_integer_operands_equivalent_with_facts(
                   known->left, queried->right, fact_table) &&
               loom_condition_integer_operands_equivalent_with_facts(
                   known->right, queried->left, fact_table)) {
      outcomes &= loom_condition_integer_relation_outcomes
          [loom_symbolic_integer_relation_swap(known->relation)];
    }
  }
  if (outcomes == 0) {
    return false;
  }
  const uint8_t matching_outcomes =
      outcomes & loom_condition_integer_relation_outcomes[queried->relation];
  if (matching_outcomes != 0 && matching_outcomes != outcomes) {
    return false;
  }
  *out_result = matching_outcomes != 0;
  return true;
}

bool loom_condition_integer_relations_equivalent(
    const loom_condition_integer_relation_t* left,
    const loom_condition_integer_relation_t* right) {
  bool left_implies_right = false;
  if (!loom_condition_integer_relation_implies(left, right,
                                               &left_implies_right) ||
      !left_implies_right) {
    return false;
  }

  bool right_implies_left = false;
  return loom_condition_integer_relation_implies(right, left,
                                                 &right_implies_left) &&
         right_implies_left;
}

bool loom_condition_integer_relation_meet(
    const loom_condition_integer_relation_t* left,
    const loom_condition_integer_relation_t* right,
    iree_host_size_t right_count,
    loom_condition_integer_relation_t* out_relation) {
  // Conjunction within one edge intersects outcomes; a CFG join unions them.
  static const loom_symbolic_integer_relation_t outcomes_relation[] = {
      [1] = LOOM_SYMBOLIC_INTEGER_RELATION_LT,
      [2] = LOOM_SYMBOLIC_INTEGER_RELATION_EQ,
      [3] = LOOM_SYMBOLIC_INTEGER_RELATION_LE,
      [4] = LOOM_SYMBOLIC_INTEGER_RELATION_GT,
      [5] = LOOM_SYMBOLIC_INTEGER_RELATION_NE,
      [6] = LOOM_SYMBOLIC_INTEGER_RELATION_GE,
  };
  uint8_t right_outcomes = 7;
  for (iree_host_size_t i = 0; i < right_count; ++i) {
    loom_symbolic_integer_relation_t relation = right[i].relation;
    if (loom_condition_integer_operands_equal(left->left, right[i].left) &&
        loom_condition_integer_operands_equal(left->right, right[i].right)) {
      right_outcomes &= loom_condition_integer_relation_outcomes[relation];
    } else if (loom_condition_integer_operands_equal(left->left,
                                                     right[i].right) &&
               loom_condition_integer_operands_equal(left->right,
                                                     right[i].left)) {
      relation = loom_symbolic_integer_relation_swap(relation);
      right_outcomes &= loom_condition_integer_relation_outcomes[relation];
    }
  }
  uint8_t common_outcomes =
      loom_condition_integer_relation_outcomes[left->relation] | right_outcomes;
  if (common_outcomes == 7) {
    return false;
  }
  *out_relation = *left;
  out_relation->relation = outcomes_relation[common_outcomes];
  return true;
}
