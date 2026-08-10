// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/condition_facts.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ops/index/compare.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/compare.h"
#include "loom/ops/scalar/ops.h"

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
                                     iree_arena_allocator_t* arena,
                                     loom_condition_query_t* out_query) {
  *out_query = (loom_condition_query_t){
      .module = module,
      .arena = arena,
  };
}

static iree_status_t loom_condition_query_ensure_value_state_capacity(
    loom_condition_query_t* query) {
  const iree_host_size_t required_capacity = query->module->values.count;
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

static iree_status_t loom_condition_query_touch_value(
    loom_condition_query_t* query, loom_value_id_t value_id) {
  if (query->value_states[value_id] != 0) return iree_ok_status();
  if (query->touched_value_count >= query->touched_value_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        query->arena, query->touched_value_count,
        query->touched_value_count + 1, sizeof(*query->touched_values),
        &query->touched_value_capacity, (void**)&query->touched_values));
  }
  query->touched_values[query->touched_value_count++] = value_id;
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
  IREE_ASSERT_EQ(query->touched_value_count, 0);
  IREE_ASSERT_EQ(query->frame_count, 0);
}

static void loom_condition_query_end(loom_condition_query_t* query) {
  for (iree_host_size_t i = 0; i < query->touched_value_count; ++i) {
    query->value_states[query->touched_values[i]] = 0;
  }
  query->touched_value_count = 0;
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
  if (out_refinements == NULL) return true;
  const loom_condition_refinement_descriptor_t* descriptor =
      loom_context_resolve_condition_refinement(module->context,
                                                condition_op->kind);
  if (descriptor == NULL) return true;
  loom_condition_refinement_truth_flags_t required_truth_flag =
      assumed_truth ? LOOM_CONDITION_REFINEMENT_TRUTH_TRUE
                    : LOOM_CONDITION_REFINEMENT_TRUTH_FALSE;
  if ((descriptor->truth_flags & required_truth_flag) == 0) return true;
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
  if (left.kind != right.kind) return false;
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

static bool loom_condition_fact_set_append_integer_relation(
    loom_condition_fact_set_t* facts,
    loom_condition_integer_relation_t relation) {
  for (iree_host_size_t i = 0; i < facts->integer_relation_count; ++i) {
    const loom_condition_integer_relation_t* existing =
        &facts->integer_relations[i];
    if (existing->relation == relation.relation &&
        loom_condition_integer_operands_equal(existing->left, relation.left) &&
        loom_condition_integer_operands_equal(existing->right,
                                              relation.right)) {
      return true;
    }
  }
  if (!facts->integer_relations ||
      facts->integer_relation_count >= facts->integer_relation_capacity) {
    return false;
  }
  facts->integer_relations[facts->integer_relation_count++] = relation;
  return true;
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

static bool loom_condition_facts_query_opaque_boolean(
    const loom_module_t* module, loom_value_id_t condition_value,
    bool assumed_truth, loom_condition_fact_set_t* out_facts) {
  if (!loom_condition_value_is_i1(module, condition_value)) return true;
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
  return loom_condition_fact_set_append_integer_relation(out_facts, assertion);
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

enum {
  LOOM_CONDITION_DERIVATION_VISITED_FALSE = 1u << 0,
  LOOM_CONDITION_DERIVATION_VISITED_TRUE = 1u << 1,
};

static iree_status_t loom_condition_query_push_derivation(
    loom_condition_query_t* query, loom_value_id_t value_id,
    bool assumed_truth) {
  if (value_id >= query->module->values.count) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_condition_query_ensure_value_state_capacity(query));
  const uint8_t visited_bit = assumed_truth
                                  ? LOOM_CONDITION_DERIVATION_VISITED_TRUE
                                  : LOOM_CONDITION_DERIVATION_VISITED_FALSE;
  if ((query->value_states[value_id] & visited_bit) != 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_condition_query_touch_value(query, value_id));
  query->value_states[value_id] |= visited_bit;
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
    loom_condition_edge_refinement_set_t* out_refinements, bool* out_complete) {
  const loom_module_t* module = query->module;
  const loom_value_t* value = loom_module_value(module, frame->value_id);
  if (loom_value_is_block_arg(value)) {
    *out_complete &= loom_condition_facts_query_opaque_boolean(
        module, frame->value_id, frame->assumed_truth, out_facts);
    return iree_ok_status();
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op) {
    *out_complete &= loom_condition_facts_query_opaque_boolean(
        module, frame->value_id, frame->assumed_truth, out_facts);
    return iree_ok_status();
  }

  *out_complete &= loom_condition_edge_refinement_set_append(
      module, defining_op, frame->assumed_truth, out_refinements);

  switch (defining_op->kind) {
    case LOOM_OP_INDEX_CMP:
      *out_complete &= loom_condition_facts_query_integer_compare(
          out_facts, fact_table, loom_index_cmp_lhs(defining_op),
          loom_index_cmp_rhs(defining_op),
          loom_index_cmp_predicate(defining_op), frame->assumed_truth,
          loom_condition_index_predicate_relation);
      return iree_ok_status();
    case LOOM_OP_SCALAR_CMPI:
      *out_complete &= loom_condition_facts_query_integer_compare(
          out_facts, fact_table, loom_scalar_cmpi_lhs(defining_op),
          loom_scalar_cmpi_rhs(defining_op),
          loom_scalar_cmpi_predicate(defining_op), frame->assumed_truth,
          loom_condition_scalar_cmpi_predicate_relation);
      return iree_ok_status();
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
      if (loom_condition_value_exact_bool(fact_table, lhs, &exact_truth)) {
        return exact_truth
                   ? loom_condition_query_push_derivation(query, rhs, false)
                   : iree_ok_status();
      }
      if (loom_condition_value_exact_bool(fact_table, rhs, &exact_truth)) {
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
      if (loom_condition_value_exact_bool(fact_table, lhs, &exact_truth)) {
        return !exact_truth
                   ? loom_condition_query_push_derivation(query, rhs, true)
                   : iree_ok_status();
      }
      if (loom_condition_value_exact_bool(fact_table, rhs, &exact_truth)) {
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
      if (loom_condition_value_exact_bool(fact_table, lhs, &exact_truth)) {
        return loom_condition_query_push_derivation(
            query, rhs, frame->assumed_truth != exact_truth);
      }
      if (loom_condition_value_exact_bool(fact_table, rhs, &exact_truth)) {
        return loom_condition_query_push_derivation(
            query, lhs, frame->assumed_truth != exact_truth);
      }
      return iree_ok_status();
    }
    default:
      *out_complete &= loom_condition_facts_query_opaque_boolean(
          module, frame->value_id, frame->assumed_truth, out_facts);
      return iree_ok_status();
  }
}

static iree_status_t loom_condition_facts_query_impl(
    loom_condition_query_t* query, const loom_value_fact_table_t* fact_table,
    loom_value_id_t condition_value, bool assumed_truth,
    loom_condition_fact_set_t* out_facts,
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
        query, fact_table, &root_frame, out_facts, out_refinements,
        out_complete);
  }
  while (iree_status_is_ok(status) && query->frame_count > 0) {
    const loom_condition_query_frame_t frame =
        query->frames[--query->frame_count];
    status = loom_condition_facts_process_derivation(
        query, fact_table, &frame, out_facts, out_refinements, out_complete);
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
      /*out_refinements=*/NULL, out_complete);
}

iree_status_t loom_condition_facts_query_edge(
    loom_condition_query_t* query, const loom_value_fact_table_t* fact_table,
    loom_value_id_t condition_value, bool assumed_truth,
    loom_condition_fact_set_t* out_facts,
    loom_condition_edge_refinement_set_t* out_refinements, bool* out_complete) {
  loom_condition_fact_set_reset(out_facts);
  loom_condition_edge_refinement_set_reset(out_refinements);
  return loom_condition_facts_query_impl(query, fact_table, condition_value,
                                         assumed_truth, out_facts,
                                         out_refinements, out_complete);
}

iree_status_t loom_condition_facts_query_into(
    loom_condition_query_t* query, const loom_value_fact_table_t* fact_table,
    loom_value_id_t condition_value, bool assumed_truth,
    loom_condition_fact_set_t* inout_facts, bool* out_complete) {
  return loom_condition_facts_query_impl(
      query, fact_table, condition_value, assumed_truth, inout_facts,
      /*out_refinements=*/NULL, out_complete);
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
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
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
  loom_type_t operand_type = loom_module_value_type(module, lhs);
  if (!loom_type_is_scalar(operand_type)) return false;
  return loom_index_cmp_result_from_facts(
      fact_table ? &fact_table->context : NULL,
      loom_type_element_type(operand_type),
      loom_index_cmp_predicate(defining_op), &lhs_facts, &rhs_facts,
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
  query->value_states[value_id] = (uint8_t)state;
  --query->frame_count;
}

static iree_status_t loom_condition_query_push_proof(
    loom_condition_query_t* query, loom_value_id_t value_id) {
  if (value_id >= query->module->values.count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_condition_query_ensure_value_state_capacity(query));
  if (query->value_states[value_id] != LOOM_CONDITION_PROOF_UNVISITED) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_condition_query_touch_value(query, value_id));
  query->value_states[value_id] = LOOM_CONDITION_PROOF_PENDING;
  return loom_condition_query_push_frame(query, value_id,
                                         LOOM_CONDITION_QUERY_FRAME_PROVE_ENTER,
                                         /*assumed_truth=*/false);
}

static const loom_op_t* loom_condition_query_proof_defining_op(
    const loom_condition_query_t* query, loom_value_id_t value_id) {
  if (value_id >= query->module->values.count) return NULL;
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
    const loom_condition_fact_set_t* edge_facts, loom_value_id_t value_id,
    bool* out_requires_composition) {
  *out_requires_composition = false;
  bool condition = false;
  if (loom_condition_value_exact_bool(fact_table, value_id, &condition)) {
    return condition ? LOOM_CONDITION_PROOF_TRUE : LOOM_CONDITION_PROOF_FALSE;
  }

  const loom_op_t* defining_op =
      loom_condition_query_proof_defining_op(query, value_id);
  if (!defining_op) return LOOM_CONDITION_PROOF_UNKNOWN;

  bool proven = false;
  switch (defining_op->kind) {
    case LOOM_OP_INDEX_CMP:
      proven = loom_condition_fact_set_proves_index_cmp(
          query->module, fact_table, edge_facts, defining_op, &condition);
      break;
    case LOOM_OP_SCALAR_CMPI:
      proven = loom_condition_fact_set_proves_scalar_cmpi(
          fact_table, edge_facts, defining_op, &condition);
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
    const loom_condition_fact_set_t* edge_facts) {
  loom_condition_query_frame_t* frame = &query->frames[query->frame_count - 1];
  const loom_op_t* defining_op =
      loom_condition_query_proof_defining_op(query, frame->value_id);
  bool requires_composition = false;
  const loom_condition_proof_state_t direct_state =
      loom_condition_query_evaluate_direct_proof(query, fact_table, edge_facts,
                                                 frame->value_id,
                                                 &requires_composition);
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
  const bool lhs_known = loom_condition_proof_state_is_known(
      (loom_condition_proof_state_t)query->value_states[lhs], &lhs_value);
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
  const bool rhs_known = loom_condition_proof_state_is_known(
      (loom_condition_proof_state_t)query->value_states[rhs], &rhs_value);
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

iree_status_t loom_condition_fact_set_proves_condition(
    loom_condition_query_t* query, const loom_value_fact_table_t* fact_table,
    const loom_condition_fact_set_t* facts, loom_value_id_t condition_value,
    bool* out_condition, bool* out_proven) {
  *out_condition = false;
  *out_proven = false;
  loom_condition_query_begin(query);
  bool requires_composition = false;
  const loom_condition_proof_state_t direct_state =
      loom_condition_query_evaluate_direct_proof(
          query, fact_table, facts, condition_value, &requires_composition);
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
      status = loom_condition_query_enter_proof_frame(query, fact_table, facts);
    } else {
      status = loom_condition_query_continue_proof_frame(query);
    }
  }
  if (iree_status_is_ok(status) &&
      condition_value < query->module->values.count) {
    *out_proven = loom_condition_proof_state_is_known(
        (loom_condition_proof_state_t)query->value_states[condition_value],
        out_condition);
  }
  loom_condition_query_end(query);
  return status;
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
  if (loom_condition_integer_operands_equal(left, right)) return true;
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

bool loom_condition_fact_set_proves_integer_relation(
    const loom_condition_fact_set_t* facts,
    const loom_value_fact_table_t* fact_table,
    const loom_condition_integer_relation_t* queried, bool* out_result) {
  if (!facts) return false;
  for (iree_host_size_t i = 0; i < facts->integer_relation_count; ++i) {
    if (loom_condition_integer_relation_implies(&facts->integer_relations[i],
                                                queried, out_result)) {
      return true;
    }
    const loom_condition_integer_relation_t* known =
        &facts->integer_relations[i];
    const bool left_operands_equal =
        loom_condition_integer_operands_equivalent_with_facts(
            known->left, queried->left, fact_table);
    const bool right_operands_equal =
        loom_condition_integer_operands_equivalent_with_facts(
            known->right, queried->right, fact_table);
    if (left_operands_equal && right_operands_equal) {
      return loom_symbolic_integer_relation_implies(
          known->relation, queried->relation, out_result);
    }

    const bool swapped_left_operands_equal =
        loom_condition_integer_operands_equivalent_with_facts(
            known->left, queried->right, fact_table);
    const bool swapped_right_operands_equal =
        loom_condition_integer_operands_equivalent_with_facts(
            known->right, queried->left, fact_table);
    if (swapped_left_operands_equal && swapped_right_operands_equal) {
      return loom_symbolic_integer_relation_implies(
          loom_symbolic_integer_relation_swap(known->relation),
          queried->relation, out_result);
    }
  }
  return false;
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
    loom_condition_integer_relation_t* out_relation) {
  bool implication_result = false;
  if (loom_condition_integer_relation_implies(right, left,
                                              &implication_result) &&
      implication_result) {
    *out_relation = *left;
    return true;
  }

  if (loom_condition_integer_relation_implies(left, right,
                                              &implication_result) &&
      implication_result) {
    *out_relation = *right;
    return true;
  }

  return false;
}
