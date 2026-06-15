// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/condition_facts.h"

#include "loom/ops/index/compare.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/compare.h"
#include "loom/ops/scalar/ops.h"

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

static loom_condition_integer_operand_t loom_condition_value_operand(
    loom_value_id_t value_id) {
  return (loom_condition_integer_operand_t){
      .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
      .value_id = value_id,
      .constant = 0,
  };
}

static bool loom_condition_fact_set_append_integer_relation(
    loom_condition_fact_set_t* facts,
    loom_condition_integer_relation_t relation) {
  if (!facts->integer_relations ||
      facts->integer_relation_count >= facts->integer_relation_capacity) {
    return false;
  }
  facts->integer_relations[facts->integer_relation_count++] = relation;
  return true;
}

static bool loom_condition_fact_set_append_all(
    loom_condition_fact_set_t* facts,
    const loom_condition_fact_set_t* new_facts) {
  bool complete = true;
  for (iree_host_size_t i = 0; i < new_facts->integer_relation_count; ++i) {
    if (!loom_condition_fact_set_append_integer_relation(
            facts, new_facts->integer_relations[i])) {
      complete = false;
    }
  }
  return complete;
}

static loom_value_facts_t loom_condition_lookup_facts(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id) {
  if (!fact_table) return loom_value_facts_unknown();
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
  if (!module || value_id >= module->values.count) return false;
  loom_type_t type = loom_module_value_type(module, value_id);
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1;
}

static bool loom_condition_value_exact_bool(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    bool* out_value) {
  int64_t value = 0;
  if (!loom_condition_value_exact_integer(fact_table, value_id, &value)) {
    return false;
  }
  if (value != 0 && value != 1) return false;
  *out_value = value != 0;
  return true;
}

static bool loom_condition_index_predicate_relation(
    uint8_t predicate, const loom_value_fact_table_t* fact_table,
    loom_value_id_t left_value, loom_value_id_t right_value,
    loom_symbolic_integer_relation_t* out_relation) {
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
      if (!loom_condition_values_are_non_negative(fact_table, left_value,
                                                  right_value)) {
        return false;
      }
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_ULE:
      if (!loom_condition_values_are_non_negative(fact_table, left_value,
                                                  right_value)) {
        return false;
      }
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LE;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_UGT:
      if (!loom_condition_values_are_non_negative(fact_table, left_value,
                                                  right_value)) {
        return false;
      }
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GT;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_UGE:
      if (!loom_condition_values_are_non_negative(fact_table, left_value,
                                                  right_value)) {
        return false;
      }
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GE;
      return true;
    default:
      return false;
  }
}

static bool loom_condition_scalar_cmpi_predicate_relation(
    uint8_t predicate, const loom_value_fact_table_t* fact_table,
    loom_value_id_t left_value, loom_value_id_t right_value,
    loom_symbolic_integer_relation_t* out_relation) {
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
      if (!loom_condition_values_are_non_negative(fact_table, left_value,
                                                  right_value)) {
        return false;
      }
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_ULE:
      if (!loom_condition_values_are_non_negative(fact_table, left_value,
                                                  right_value)) {
        return false;
      }
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LE;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_UGT:
      if (!loom_condition_values_are_non_negative(fact_table, left_value,
                                                  right_value)) {
        return false;
      }
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GT;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_UGE:
      if (!loom_condition_values_are_non_negative(fact_table, left_value,
                                                  right_value)) {
        return false;
      }
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GE;
      return true;
    default:
      return false;
  }
}

static bool loom_condition_facts_query_integer_compare(
    loom_condition_fact_set_t* facts, const loom_value_fact_table_t* fact_table,
    loom_value_id_t left_value, loom_value_id_t right_value, uint8_t predicate,
    bool assumed_truth,
    bool (*predicate_relation)(uint8_t, const loom_value_fact_table_t*,
                               loom_value_id_t, loom_value_id_t,
                               loom_symbolic_integer_relation_t*)) {
  loom_symbolic_integer_relation_t relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
  if (!predicate_relation(predicate, fact_table, left_value, right_value,
                          &relation)) {
    return true;
  }
  if (!assumed_truth) {
    relation = loom_symbolic_integer_relation_invert(relation);
  }

  loom_condition_integer_relation_t assertion = {
      .relation = relation,
      .left = loom_condition_value_operand(left_value),
      .right = loom_condition_value_operand(right_value),
  };
  return loom_condition_fact_set_append_integer_relation(facts, assertion);
}

static bool loom_condition_facts_query_impl(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t condition_value, bool assumed_truth,
    loom_condition_fact_set_t* out_facts, uint8_t recursion_depth);

static bool loom_condition_facts_query_child(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t condition_value, bool assumed_truth,
    loom_condition_fact_set_t* out_facts, uint8_t recursion_depth) {
  loom_condition_integer_relation_t relation_storage[16];
  loom_condition_fact_set_t child_facts;
  loom_condition_fact_set_initialize(
      relation_storage, IREE_ARRAYSIZE(relation_storage), &child_facts);
  bool child_complete = loom_condition_facts_query_impl(
      module, fact_table, condition_value, assumed_truth, &child_facts,
      (uint8_t)(recursion_depth + 1));
  bool append_complete =
      loom_condition_fact_set_append_all(out_facts, &child_facts);
  return child_complete && append_complete;
}

static bool loom_condition_facts_query_boolean_and(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t lhs, loom_value_id_t rhs, bool assumed_truth,
    loom_condition_fact_set_t* out_facts, uint8_t recursion_depth) {
  if (assumed_truth) {
    bool lhs_complete = loom_condition_facts_query_child(
        module, fact_table, lhs, true, out_facts, recursion_depth);
    bool rhs_complete = loom_condition_facts_query_child(
        module, fact_table, rhs, true, out_facts, recursion_depth);
    return lhs_complete && rhs_complete;
  }

  bool lhs_truth = false;
  bool rhs_truth = false;
  if (loom_condition_value_exact_bool(fact_table, lhs, &lhs_truth)) {
    if (lhs_truth) {
      return loom_condition_facts_query_child(module, fact_table, rhs, false,
                                              out_facts, recursion_depth);
    }
    return true;
  }
  if (loom_condition_value_exact_bool(fact_table, rhs, &rhs_truth)) {
    if (rhs_truth) {
      return loom_condition_facts_query_child(module, fact_table, lhs, false,
                                              out_facts, recursion_depth);
    }
    return true;
  }
  return true;
}

static bool loom_condition_facts_query_boolean_or(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t lhs, loom_value_id_t rhs, bool assumed_truth,
    loom_condition_fact_set_t* out_facts, uint8_t recursion_depth) {
  if (!assumed_truth) {
    bool lhs_complete = loom_condition_facts_query_child(
        module, fact_table, lhs, false, out_facts, recursion_depth);
    bool rhs_complete = loom_condition_facts_query_child(
        module, fact_table, rhs, false, out_facts, recursion_depth);
    return lhs_complete && rhs_complete;
  }

  bool lhs_truth = false;
  bool rhs_truth = false;
  if (loom_condition_value_exact_bool(fact_table, lhs, &lhs_truth)) {
    if (!lhs_truth) {
      return loom_condition_facts_query_child(module, fact_table, rhs, true,
                                              out_facts, recursion_depth);
    }
    return true;
  }
  if (loom_condition_value_exact_bool(fact_table, rhs, &rhs_truth)) {
    if (!rhs_truth) {
      return loom_condition_facts_query_child(module, fact_table, lhs, true,
                                              out_facts, recursion_depth);
    }
    return true;
  }
  return true;
}

static bool loom_condition_facts_query_boolean_xor(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t lhs, loom_value_id_t rhs, bool assumed_truth,
    loom_condition_fact_set_t* out_facts, uint8_t recursion_depth) {
  bool lhs_truth = false;
  bool rhs_truth = false;
  if (loom_condition_value_exact_bool(fact_table, lhs, &lhs_truth)) {
    return loom_condition_facts_query_child(module, fact_table, rhs,
                                            assumed_truth != lhs_truth,
                                            out_facts, recursion_depth);
  }
  if (loom_condition_value_exact_bool(fact_table, rhs, &rhs_truth)) {
    return loom_condition_facts_query_child(module, fact_table, lhs,
                                            assumed_truth != rhs_truth,
                                            out_facts, recursion_depth);
  }
  return true;
}

static bool loom_condition_facts_query_impl(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t condition_value, bool assumed_truth,
    loom_condition_fact_set_t* out_facts, uint8_t recursion_depth) {
  if (recursion_depth > 16) return false;
  loom_condition_fact_set_reset(out_facts);
  if (!module || condition_value >= module->values.count) {
    return true;
  }
  const loom_value_t* value = loom_module_value(module, condition_value);
  if (loom_value_is_block_arg(value)) return true;
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op) return true;

  switch (defining_op->kind) {
    case LOOM_OP_INDEX_CMP:
      return loom_condition_facts_query_integer_compare(
          out_facts, fact_table, loom_index_cmp_lhs(defining_op),
          loom_index_cmp_rhs(defining_op),
          loom_index_cmp_predicate(defining_op), assumed_truth,
          loom_condition_index_predicate_relation);
    case LOOM_OP_SCALAR_CMPI:
      return loom_condition_facts_query_integer_compare(
          out_facts, fact_table, loom_scalar_cmpi_lhs(defining_op),
          loom_scalar_cmpi_rhs(defining_op),
          loom_scalar_cmpi_predicate(defining_op), assumed_truth,
          loom_condition_scalar_cmpi_predicate_relation);
    case LOOM_OP_SCALAR_ANDI:
      if (!loom_condition_value_is_i1(module,
                                      loom_op_const_results(defining_op)[0]) ||
          !loom_condition_value_is_i1(module,
                                      loom_scalar_andi_lhs(defining_op)) ||
          !loom_condition_value_is_i1(module,
                                      loom_scalar_andi_rhs(defining_op))) {
        return true;
      }
      return loom_condition_facts_query_boolean_and(
          module, fact_table, loom_scalar_andi_lhs(defining_op),
          loom_scalar_andi_rhs(defining_op), assumed_truth, out_facts,
          recursion_depth);
    case LOOM_OP_SCALAR_ORI:
      if (!loom_condition_value_is_i1(module,
                                      loom_op_const_results(defining_op)[0]) ||
          !loom_condition_value_is_i1(module,
                                      loom_scalar_ori_lhs(defining_op)) ||
          !loom_condition_value_is_i1(module,
                                      loom_scalar_ori_rhs(defining_op))) {
        return true;
      }
      return loom_condition_facts_query_boolean_or(
          module, fact_table, loom_scalar_ori_lhs(defining_op),
          loom_scalar_ori_rhs(defining_op), assumed_truth, out_facts,
          recursion_depth);
    case LOOM_OP_SCALAR_XORI:
      if (!loom_condition_value_is_i1(module,
                                      loom_op_const_results(defining_op)[0]) ||
          !loom_condition_value_is_i1(module,
                                      loom_scalar_xori_lhs(defining_op)) ||
          !loom_condition_value_is_i1(module,
                                      loom_scalar_xori_rhs(defining_op))) {
        return true;
      }
      return loom_condition_facts_query_boolean_xor(
          module, fact_table, loom_scalar_xori_lhs(defining_op),
          loom_scalar_xori_rhs(defining_op), assumed_truth, out_facts,
          recursion_depth);
    default:
      return true;
  }
}

bool loom_condition_facts_query(const loom_module_t* module,
                                const loom_value_fact_table_t* fact_table,
                                loom_value_id_t condition_value,
                                bool assumed_truth,
                                loom_condition_fact_set_t* out_facts) {
  return loom_condition_facts_query_impl(module, fact_table, condition_value,
                                         assumed_truth, out_facts,
                                         /*recursion_depth=*/0);
}

static loom_value_facts_t loom_condition_edge_value_facts(
    const loom_value_fact_table_t* fact_table,
    const loom_condition_fact_set_t* edge_facts, loom_value_id_t value_id) {
  loom_value_facts_t value_facts =
      loom_condition_lookup_facts(fact_table, value_id);
  if (edge_facts != NULL) {
    (void)loom_condition_fact_set_apply_to_value_facts(edge_facts, fact_table,
                                                       value_id, &value_facts);
  }
  return value_facts;
}

static bool loom_condition_fact_set_proves_index_cmp(
    const loom_value_fact_table_t* fact_table,
    const loom_condition_fact_set_t* edge_facts, const loom_op_t* defining_op,
    bool* out_condition) {
  const loom_value_id_t lhs = loom_index_cmp_lhs(defining_op);
  const loom_value_id_t rhs = loom_index_cmp_rhs(defining_op);
  if (lhs == rhs && loom_index_cmp_same_value_result(
                        loom_index_cmp_predicate(defining_op), out_condition)) {
    return true;
  }

  const loom_value_facts_t lhs_facts =
      loom_condition_edge_value_facts(fact_table, edge_facts, lhs);
  const loom_value_facts_t rhs_facts =
      loom_condition_edge_value_facts(fact_table, edge_facts, rhs);
  return loom_index_cmp_result_from_facts(loom_index_cmp_predicate(defining_op),
                                          &lhs_facts, &rhs_facts,
                                          out_condition);
}

static bool loom_condition_fact_set_proves_scalar_cmpi(
    const loom_value_fact_table_t* fact_table,
    const loom_condition_fact_set_t* edge_facts, const loom_op_t* defining_op,
    bool* out_condition) {
  const loom_value_id_t lhs = loom_scalar_cmpi_lhs(defining_op);
  const loom_value_id_t rhs = loom_scalar_cmpi_rhs(defining_op);
  if (lhs == rhs &&
      loom_scalar_cmpi_same_value_result(
          loom_scalar_cmpi_predicate(defining_op), out_condition)) {
    return true;
  }

  const loom_value_facts_t lhs_facts =
      loom_condition_edge_value_facts(fact_table, edge_facts, lhs);
  const loom_value_facts_t rhs_facts =
      loom_condition_edge_value_facts(fact_table, edge_facts, rhs);
  return loom_scalar_cmpi_result_from_facts(
      loom_scalar_cmpi_predicate(defining_op), &lhs_facts, &rhs_facts,
      out_condition);
}

static bool loom_condition_fact_set_proves_condition_impl(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_condition_fact_set_t* edge_facts,
    loom_value_id_t condition_value, bool* out_condition,
    uint8_t recursion_depth);

static bool loom_condition_fact_set_proves_boolean_and(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_condition_fact_set_t* edge_facts, const loom_op_t* defining_op,
    bool* out_condition, uint8_t recursion_depth) {
  bool lhs = false;
  const bool lhs_known = loom_condition_fact_set_proves_condition_impl(
      module, fact_table, edge_facts, loom_scalar_andi_lhs(defining_op), &lhs,
      recursion_depth);
  if (lhs_known && !lhs) {
    *out_condition = false;
    return true;
  }

  bool rhs = false;
  const bool rhs_known = loom_condition_fact_set_proves_condition_impl(
      module, fact_table, edge_facts, loom_scalar_andi_rhs(defining_op), &rhs,
      recursion_depth);
  if (rhs_known && !rhs) {
    *out_condition = false;
    return true;
  }
  if (!lhs_known || !rhs_known) {
    return false;
  }

  *out_condition = true;
  return true;
}

static bool loom_condition_fact_set_proves_boolean_or(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_condition_fact_set_t* edge_facts, const loom_op_t* defining_op,
    bool* out_condition, uint8_t recursion_depth) {
  bool lhs = false;
  const bool lhs_known = loom_condition_fact_set_proves_condition_impl(
      module, fact_table, edge_facts, loom_scalar_ori_lhs(defining_op), &lhs,
      recursion_depth);
  if (lhs_known && lhs) {
    *out_condition = true;
    return true;
  }

  bool rhs = false;
  const bool rhs_known = loom_condition_fact_set_proves_condition_impl(
      module, fact_table, edge_facts, loom_scalar_ori_rhs(defining_op), &rhs,
      recursion_depth);
  if (rhs_known && rhs) {
    *out_condition = true;
    return true;
  }
  if (!lhs_known || !rhs_known) {
    return false;
  }

  *out_condition = false;
  return true;
}

static bool loom_condition_fact_set_proves_boolean_xor(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_condition_fact_set_t* edge_facts, const loom_op_t* defining_op,
    bool* out_condition, uint8_t recursion_depth) {
  bool lhs = false;
  if (!loom_condition_fact_set_proves_condition_impl(
          module, fact_table, edge_facts, loom_scalar_xori_lhs(defining_op),
          &lhs, recursion_depth)) {
    return false;
  }
  bool rhs = false;
  if (!loom_condition_fact_set_proves_condition_impl(
          module, fact_table, edge_facts, loom_scalar_xori_rhs(defining_op),
          &rhs, recursion_depth)) {
    return false;
  }
  *out_condition = lhs != rhs;
  return true;
}

static bool loom_condition_fact_set_proves_condition_impl(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_condition_fact_set_t* edge_facts,
    loom_value_id_t condition_value, bool* out_condition,
    uint8_t recursion_depth) {
  *out_condition = false;
  if (recursion_depth > 16) return false;
  if (loom_condition_value_exact_bool(fact_table, condition_value,
                                      out_condition)) {
    return true;
  }
  if (!module || condition_value >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, condition_value);
  if (loom_value_is_block_arg(value) || loom_value_def_index(value) != 0) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op) return false;

  switch (defining_op->kind) {
    case LOOM_OP_INDEX_CMP:
      return loom_condition_fact_set_proves_index_cmp(
          fact_table, edge_facts, defining_op, out_condition);
    case LOOM_OP_SCALAR_CMPI:
      return loom_condition_fact_set_proves_scalar_cmpi(
          fact_table, edge_facts, defining_op, out_condition);
    case LOOM_OP_SCALAR_ANDI:
      if (!loom_condition_value_is_i1(module,
                                      loom_op_const_results(defining_op)[0]) ||
          !loom_condition_value_is_i1(module,
                                      loom_scalar_andi_lhs(defining_op)) ||
          !loom_condition_value_is_i1(module,
                                      loom_scalar_andi_rhs(defining_op))) {
        return false;
      }
      return loom_condition_fact_set_proves_boolean_and(
          module, fact_table, edge_facts, defining_op, out_condition,
          (uint8_t)(recursion_depth + 1));
    case LOOM_OP_SCALAR_ORI:
      if (!loom_condition_value_is_i1(module,
                                      loom_op_const_results(defining_op)[0]) ||
          !loom_condition_value_is_i1(module,
                                      loom_scalar_ori_lhs(defining_op)) ||
          !loom_condition_value_is_i1(module,
                                      loom_scalar_ori_rhs(defining_op))) {
        return false;
      }
      return loom_condition_fact_set_proves_boolean_or(
          module, fact_table, edge_facts, defining_op, out_condition,
          (uint8_t)(recursion_depth + 1));
    case LOOM_OP_SCALAR_XORI:
      if (!loom_condition_value_is_i1(module,
                                      loom_op_const_results(defining_op)[0]) ||
          !loom_condition_value_is_i1(module,
                                      loom_scalar_xori_lhs(defining_op)) ||
          !loom_condition_value_is_i1(module,
                                      loom_scalar_xori_rhs(defining_op))) {
        return false;
      }
      return loom_condition_fact_set_proves_boolean_xor(
          module, fact_table, edge_facts, defining_op, out_condition,
          (uint8_t)(recursion_depth + 1));
    default:
      return false;
  }
}

bool loom_condition_fact_set_proves_condition(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_condition_fact_set_t* facts, loom_value_id_t condition_value,
    bool* out_condition) {
  return loom_condition_fact_set_proves_condition_impl(
      module, fact_table, facts, condition_value, out_condition,
      /*recursion_depth=*/0);
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
    loom_condition_integer_operand_t operand, loom_value_id_t value_id) {
  return operand.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE &&
         operand.value_id == value_id;
}

bool loom_condition_integer_relation_make_predicate_for_value(
    const loom_condition_integer_relation_t* relation,
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_predicate_t* out_predicate) {
  loom_symbolic_integer_relation_t normalized_relation = relation->relation;
  loom_condition_integer_operand_t other = {0};
  if (loom_condition_operand_matches_value(relation->left, value_id)) {
    other = relation->right;
  } else if (loom_condition_operand_matches_value(relation->right, value_id)) {
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
  if (predicate.arg_tags[1] != LOOM_PRED_ARG_CONST) return false;
  loom_value_facts_apply_predicate(inout_facts, &predicate);
  return true;
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
