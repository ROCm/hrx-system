// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/call_predicates.h"

#include "loom/analysis/condition_facts.h"
#include "loom/analysis/symbolic_expr_proof.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/decision/predicate.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

static iree_status_t loom_low_call_emit_contract_error(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, iree_string_view_t reason) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(callee_name),
      loom_param_string(reason),
  };
  return loom_low_lower_emit_target_context_error(
      context, source_op, LOOM_ERR_TARGET_072, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_call_emit_precondition_error(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, uint32_t predicate_index,
    uint8_t predicate_kind, loom_decision_truth_t proof_result) {
  const char* predicate_kind_name = loom_predicate_kind_name(predicate_kind);
  const iree_string_view_t proof_result_name =
      proof_result == LOOM_DECISION_TRUTH_FALSE ? IREE_SV("proven false")
                                                : IREE_SV("not proven");
  const loom_diagnostic_param_t params[] = {
      loom_param_string(callee_name),
      loom_param_u32(predicate_index),
      loom_param_string(predicate_kind_name != NULL
                            ? iree_make_cstring_view(predicate_kind_name)
                            : IREE_SV("<invalid>")),
      loom_param_string(proof_result_name),
  };
  return loom_low_lower_emit_target_context_error(
      context, source_op, LOOM_ERR_TARGET_078, params, IREE_ARRAYSIZE(params));
}

static bool loom_low_call_find_argument(const loom_value_id_t* callee_arguments,
                                        uint16_t callee_argument_count,
                                        loom_value_id_t value_id,
                                        uint16_t* out_argument_index) {
  if (out_argument_index) *out_argument_index = 0;
  for (uint16_t i = 0; i < callee_argument_count; ++i) {
    if (callee_arguments[i] != value_id) continue;
    if (out_argument_index) *out_argument_index = i;
    return true;
  }
  return false;
}

static bool loom_low_call_predicate_integer_relation(
    uint8_t predicate_kind,
    loom_symbolic_integer_relation_t* out_integer_relation) {
  switch ((loom_predicate_kind_t)predicate_kind) {
    case LOOM_PREDICATE_EQ:
      *out_integer_relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
      return true;
    case LOOM_PREDICATE_NE:
      *out_integer_relation = LOOM_SYMBOLIC_INTEGER_RELATION_NE;
      return true;
    case LOOM_PREDICATE_LT:
      *out_integer_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT;
      return true;
    case LOOM_PREDICATE_LE:
      *out_integer_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LE;
      return true;
    case LOOM_PREDICATE_GT:
      *out_integer_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GT;
      return true;
    case LOOM_PREDICATE_GE:
    case LOOM_PREDICATE_MIN:
      *out_integer_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GE;
      return true;
    case LOOM_PREDICATE_MAX:
      *out_integer_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LE;
      return true;
    case LOOM_PREDICATE_MUL:
    case LOOM_PREDICATE_POW2:
    case LOOM_PREDICATE_RANGE:
    case LOOM_PREDICATE_NOT_NAN:
    case LOOM_PREDICATE_NOT_INF:
    case LOOM_PREDICATE_FINITE:
    case LOOM_PREDICATE_COUNT_:
      return false;
  }
  return false;
}

static bool loom_low_call_decision_operand_as_integer(
    const loom_decision_predicate_operand_t* operand,
    loom_condition_integer_operand_t* out_integer_operand) {
  *out_integer_operand = (loom_condition_integer_operand_t){0};
  if (operand->identity != LOOM_DECISION_OPERAND_IDENTITY_NONE) {
    out_integer_operand->kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE;
    out_integer_operand->value_id = operand->identity;
    return true;
  }
  int64_t constant = 0;
  if (!loom_value_facts_as_exact_i64(operand->facts, &constant)) return false;
  out_integer_operand->kind = LOOM_CONDITION_INTEGER_OPERAND_CONSTANT;
  out_integer_operand->constant = constant;
  return true;
}

static bool loom_low_call_caller_predicate_operand_as_integer(
    const loom_predicate_t* predicate, uint8_t argument_index,
    const loom_value_id_t* caller_arguments, uint16_t caller_argument_count,
    loom_condition_integer_operand_t* out_integer_operand) {
  *out_integer_operand = (loom_condition_integer_operand_t){0};
  switch ((loom_predicate_arg_tag_t)predicate->arg_tags[argument_index]) {
    case LOOM_PRED_ARG_CONST:
      out_integer_operand->kind = LOOM_CONDITION_INTEGER_OPERAND_CONSTANT;
      out_integer_operand->constant = predicate->args[argument_index];
      return true;
    case LOOM_PRED_ARG_VALUE: {
      const int64_t raw_value_id = predicate->args[argument_index];
      if (raw_value_id < 0 || raw_value_id > UINT32_MAX) return false;
      if (!loom_low_call_find_argument(caller_arguments, caller_argument_count,
                                       (loom_value_id_t)raw_value_id,
                                       /*out_argument_index=*/NULL)) {
        return false;
      }
      out_integer_operand->kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE;
      out_integer_operand->value_id = (loom_value_id_t)raw_value_id;
      return true;
    }
    case LOOM_PRED_ARG_NONE:
    default:
      return false;
  }
}

static iree_status_t loom_low_call_integer_operands_match(
    loom_low_lower_context_t* context,
    loom_condition_integer_operand_t known_operand,
    loom_condition_integer_operand_t queried_operand, bool* out_match) {
  *out_match = false;
  if (known_operand.kind == LOOM_CONDITION_INTEGER_OPERAND_CONSTANT &&
      queried_operand.kind == LOOM_CONDITION_INTEGER_OPERAND_CONSTANT) {
    *out_match = known_operand.constant == queried_operand.constant;
    return iree_ok_status();
  }
  loom_symbolic_expr_context_t* expression_context =
      loom_low_lower_context_symbolic_expr_context(context);
  if (known_operand.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE &&
      queried_operand.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
    loom_symbolic_proof_result_t equality = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    IREE_RETURN_IF_ERROR(
        loom_symbolic_expr_prove_value_relation_with_active_facts(
            expression_context, LOOM_SYMBOLIC_INTEGER_RELATION_EQ,
            known_operand.value_id, queried_operand.value_id, &equality));
    *out_match = equality == LOOM_SYMBOLIC_PROOF_TRUE;
    return iree_ok_status();
  }

  const loom_condition_integer_operand_t value_operand =
      known_operand.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE
          ? known_operand
          : queried_operand;
  const loom_condition_integer_operand_t constant_operand =
      known_operand.kind == LOOM_CONDITION_INTEGER_OPERAND_CONSTANT
          ? known_operand
          : queried_operand;
  if (value_operand.kind != LOOM_CONDITION_INTEGER_OPERAND_VALUE ||
      constant_operand.kind != LOOM_CONDITION_INTEGER_OPERAND_CONSTANT) {
    return iree_ok_status();
  }
  loom_symbolic_expr_t value_expression = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
      expression_context, value_operand.value_id, &value_expression));
  int64_t exact_value = 0;
  *out_match =
      loom_value_facts_as_exact_i64(value_expression.facts, &exact_value) &&
      exact_value == constant_operand.constant;
  return iree_ok_status();
}

static iree_status_t loom_low_call_caller_contract_proves_relation(
    loom_low_lower_context_t* context,
    const loom_condition_integer_relation_t* queried_relation,
    loom_decision_truth_t* out_proof_result) {
  *out_proof_result = LOOM_DECISION_TRUTH_UNKNOWN;
  uint16_t caller_argument_count = 0;
  const loom_value_id_t* caller_arguments =
      loom_func_like_arg_ids(context->source_function, &caller_argument_count);
  uint16_t caller_predicate_count = 0;
  const loom_predicate_t* caller_predicates = loom_func_like_predicates(
      context->source_function, &caller_predicate_count);
  for (uint16_t i = 0; i < caller_predicate_count; ++i) {
    // Only caller argument predicates are available at the invocation. A
    // predicate involving a function result is a postcondition, not evidence
    // for work still inside the function body.
    const loom_predicate_t* caller_predicate = &caller_predicates[i];
    loom_condition_integer_relation_t known_relation = {0};
    if (caller_predicate->arg_count != 2 ||
        !loom_low_call_predicate_integer_relation(caller_predicate->kind,
                                                  &known_relation.relation) ||
        !loom_low_call_caller_predicate_operand_as_integer(
            caller_predicate, 0, caller_arguments, caller_argument_count,
            &known_relation.left) ||
        !loom_low_call_caller_predicate_operand_as_integer(
            caller_predicate, 1, caller_arguments, caller_argument_count,
            &known_relation.right)) {
      continue;
    }

    bool left_matches = false;
    IREE_RETURN_IF_ERROR(loom_low_call_integer_operands_match(
        context, known_relation.left, queried_relation->left, &left_matches));
    bool right_matches = false;
    IREE_RETURN_IF_ERROR(loom_low_call_integer_operands_match(
        context, known_relation.right, queried_relation->right,
        &right_matches));
    bool implication_result = false;
    if (left_matches && right_matches &&
        loom_symbolic_integer_relation_implies(known_relation.relation,
                                               queried_relation->relation,
                                               &implication_result)) {
      *out_proof_result = implication_result ? LOOM_DECISION_TRUTH_TRUE
                                             : LOOM_DECISION_TRUTH_FALSE;
      return iree_ok_status();
    }

    bool swapped_left_matches = false;
    IREE_RETURN_IF_ERROR(loom_low_call_integer_operands_match(
        context, known_relation.left, queried_relation->right,
        &swapped_left_matches));
    bool swapped_right_matches = false;
    IREE_RETURN_IF_ERROR(loom_low_call_integer_operands_match(
        context, known_relation.right, queried_relation->left,
        &swapped_right_matches));
    if (swapped_left_matches && swapped_right_matches &&
        loom_symbolic_integer_relation_implies(
            loom_symbolic_integer_relation_swap(known_relation.relation),
            queried_relation->relation, &implication_result)) {
      *out_proof_result = implication_result ? LOOM_DECISION_TRUTH_TRUE
                                             : LOOM_DECISION_TRUTH_FALSE;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_call_prove_predicate(
    loom_low_lower_context_t* context, uint8_t predicate_kind,
    const loom_decision_predicate_operand_t predicate_operands[3],
    loom_decision_truth_t* out_proof_result) {
  // Scalar facts handle exact values, ranges, divisibility, and floating-point
  // classes. Relational predicates then use symbolic/path proofs and finally
  // implication from the caller's entry contract. None of these paths mutate
  // facts: low.assume is built only after this proof returns true.
  *out_proof_result =
      loom_decision_predicate_evaluate(predicate_kind, predicate_operands);
  if (*out_proof_result != LOOM_DECISION_TRUTH_UNKNOWN) {
    return iree_ok_status();
  }

  loom_condition_integer_relation_t queried_relation = {0};
  if (!loom_low_call_predicate_integer_relation(predicate_kind,
                                                &queried_relation.relation) ||
      !loom_low_call_decision_operand_as_integer(&predicate_operands[0],
                                                 &queried_relation.left) ||
      !loom_low_call_decision_operand_as_integer(&predicate_operands[1],
                                                 &queried_relation.right)) {
    return iree_ok_status();
  }

  if (queried_relation.left.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE &&
      queried_relation.right.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
    loom_symbolic_proof_result_t symbolic_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    IREE_RETURN_IF_ERROR(
        loom_symbolic_expr_prove_value_relation_with_active_facts(
            loom_low_lower_context_symbolic_expr_context(context),
            queried_relation.relation, queried_relation.left.value_id,
            queried_relation.right.value_id, &symbolic_result));
    if (symbolic_result != LOOM_SYMBOLIC_PROOF_UNKNOWN) {
      *out_proof_result = symbolic_result == LOOM_SYMBOLIC_PROOF_TRUE
                              ? LOOM_DECISION_TRUTH_TRUE
                              : LOOM_DECISION_TRUTH_FALSE;
      return iree_ok_status();
    }
  }

  return loom_low_call_caller_contract_proves_relation(
      context, &queried_relation, out_proof_result);
}

static iree_status_t loom_low_call_resolve_precondition_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, const loom_predicate_t* predicate,
    uint8_t predicate_argument_index, const loom_value_id_t* callee_arguments,
    uint16_t callee_argument_count, loom_value_slice_t source_operands,
    loom_decision_predicate_operand_t* out_operand) {
  *out_operand = (loom_decision_predicate_operand_t){
      .facts = loom_value_facts_unknown(),
      .identity = LOOM_DECISION_OPERAND_IDENTITY_NONE,
  };
  switch (
      (loom_predicate_arg_tag_t)predicate->arg_tags[predicate_argument_index]) {
    case LOOM_PRED_ARG_CONST:
      out_operand->facts =
          loom_value_facts_exact_i64(predicate->args[predicate_argument_index]);
      return iree_ok_status();
    case LOOM_PRED_ARG_VALUE: {
      const int64_t raw_value_id = predicate->args[predicate_argument_index];
      if (raw_value_id < 0 || raw_value_id > UINT32_MAX) {
        return loom_low_call_emit_contract_error(
            context, source_op, callee_name,
            IREE_SV("a helper precondition has an invalid SSA value"));
      }
      uint16_t callee_argument_index = 0;
      if (!loom_low_call_find_argument(callee_arguments, callee_argument_count,
                                       (loom_value_id_t)raw_value_id,
                                       &callee_argument_index)) {
        return loom_low_call_emit_contract_error(
            context, source_op, callee_name,
            IREE_SV("helper preconditions may reference only helper "
                    "arguments"));
      }
      if (callee_argument_index >= source_operands.count) {
        return loom_low_call_emit_contract_error(
            context, source_op, callee_name,
            IREE_SV("a helper precondition argument has no invocation "
                    "operand"));
      }
      const loom_value_id_t source_value =
          source_operands.values[callee_argument_index];
      out_operand->identity = source_value;
      loom_symbolic_expr_t source_expression = {0};
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
          loom_low_lower_context_symbolic_expr_context(context), source_value,
          &source_expression));
      out_operand->facts = source_expression.facts;
      return iree_ok_status();
    }
    case LOOM_PRED_ARG_NONE:
    default:
      return loom_low_call_emit_contract_error(
          context, source_op, callee_name,
          IREE_SV("a helper precondition has an invalid argument kind"));
  }
}

static iree_status_t loom_low_call_prove_argument_contract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, const loom_predicate_t* predicates,
    uint16_t predicate_count, const loom_value_id_t* callee_arguments,
    uint16_t callee_argument_count, loom_value_slice_t source_operands) {
  for (uint16_t predicate_index = 0; predicate_index < predicate_count;
       ++predicate_index) {
    const loom_predicate_t* predicate = &predicates[predicate_index];
    const uint8_t expected_argument_count =
        loom_predicate_kind_argument_count(predicate->kind);
    if (expected_argument_count == UINT8_MAX ||
        predicate->arg_count != expected_argument_count ||
        predicate->arg_count > IREE_ARRAYSIZE(predicate->args)) {
      return loom_low_call_emit_contract_error(
          context, source_op, callee_name,
          IREE_SV("a helper precondition has an invalid predicate shape"));
    }

    loom_decision_predicate_operand_t predicate_operands[3] = {0};
    for (uint8_t argument_index = 0; argument_index < predicate->arg_count;
         ++argument_index) {
      IREE_RETURN_IF_ERROR(loom_low_call_resolve_precondition_operand(
          context, source_op, callee_name, predicate, argument_index,
          callee_arguments, callee_argument_count, source_operands,
          &predicate_operands[argument_index]));
    }
    loom_decision_truth_t proof_result = LOOM_DECISION_TRUTH_UNKNOWN;
    IREE_RETURN_IF_ERROR(loom_low_call_prove_predicate(
        context, predicate->kind, predicate_operands, &proof_result));
    if (proof_result != LOOM_DECISION_TRUTH_TRUE) {
      return loom_low_call_emit_precondition_error(
          context, source_op, callee_name, predicate_index, predicate->kind,
          proof_result);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_materialize_call_argument_contract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, loom_func_like_t callee,
    const loom_value_id_t* callee_arguments, uint16_t callee_argument_count,
    loom_value_slice_t source_operands, loom_ir_remap_t* remap) {
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(callee, &predicate_count);
  if (predicate_count == 0) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_low_call_prove_argument_contract(
      context, source_op, callee_name, predicates, predicate_count,
      callee_arguments, callee_argument_count, source_operands));

  loom_predicate_t* remapped_predicates = NULL;
  IREE_RETURN_IF_ERROR(loom_ir_remap_predicate_list(
      remap, predicates, predicate_count, &remapped_predicates));

  loom_value_id_t* low_arguments = NULL;
  loom_type_t* low_argument_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, callee_argument_count, sizeof(*low_arguments),
      (void**)&low_arguments));
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, callee_argument_count, sizeof(*low_argument_types),
      (void**)&low_argument_types));
  for (uint16_t i = 0; i < callee_argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(remap, callee_arguments[i],
                                                     &low_arguments[i]));
    low_argument_types[i] = loom_module_value_type(
        loom_low_lower_context_module(context), low_arguments[i]);
  }

  loom_op_t* assume_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_assume_build(
      loom_low_lower_context_builder(context), low_arguments,
      callee_argument_count, remapped_predicates, predicate_count,
      low_argument_types, callee_argument_count, source_op->location,
      &assume_op));
  const loom_value_slice_t assumed_arguments =
      loom_low_assume_results(assume_op);
  return loom_ir_remap_map_values(
      remap, callee_arguments, assumed_arguments.values, callee_argument_count);
}
