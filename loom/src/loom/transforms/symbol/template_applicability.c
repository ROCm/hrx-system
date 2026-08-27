// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/template_applicability.h"

#include <stdint.h>

#include "loom/decision/predicate.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/template/ops.h"

static loom_value_slice_t loom_template_applicability_application_operands(
    const loom_op_t* application_op) {
  if (loom_template_apply_isa(application_op)) {
    return loom_template_apply_operands(application_op);
  }
  IREE_ASSERT(loom_template_call_isa(application_op));
  return loom_template_call_operands(application_op);
}

static loom_value_slice_t loom_template_applicability_application_results(
    const loom_op_t* application_op) {
  if (loom_template_apply_isa(application_op)) {
    return loom_template_apply_results(application_op);
  }
  IREE_ASSERT(loom_template_call_isa(application_op));
  return loom_template_call_results(application_op);
}

loom_template_applicability_contract_t
loom_template_applicability_provider_contract(
    const loom_template_provider_summary_t* provider) {
  return (loom_template_applicability_contract_t){
      .module = provider->module,
      .target_symbol = provider->target_symbol,
      .target_facts = provider->target_facts,
      .argument_ids = provider->argument_ids,
      .result_ids = provider->result_ids,
      .predicates = provider->predicates,
      .target_conditions = provider->target_conditions,
      .argument_count = provider->argument_count,
      .result_count = provider->result_count,
      .predicate_count = provider->predicate_count,
      .target_condition_count = provider->target_condition_count,
  };
}

static loom_template_provider_feasibility_t
loom_template_applicability_target_feasibility(
    const loom_module_t* application_module,
    const loom_template_applicability_target_t* application_target,
    const loom_template_applicability_contract_t* contract) {
  if (!loom_symbol_ref_is_valid(contract->target_symbol) &&
      contract->target_facts == NULL) {
    return LOOM_TEMPLATE_PROVIDER_MATCH;
  }
  if (contract->module == application_module &&
      loom_symbol_ref_is_valid(application_target->witness) &&
      contract->target_symbol.module_id ==
          application_target->witness.module_id &&
      contract->target_symbol.symbol_id ==
          application_target->witness.symbol_id) {
    return LOOM_TEMPLATE_PROVIDER_MATCH;
  }
  if (application_target->facts == NULL) {
    return LOOM_TEMPLATE_PROVIDER_MAYBE;
  }
  const loom_target_facts_t* target_requirement = contract->target_facts;
  if (target_requirement == NULL) {
    return LOOM_TEMPLATE_PROVIDER_MAYBE;
  }
  return loom_target_facts_satisfy_identity_requirement(
             application_target->facts, target_requirement)
             ? LOOM_TEMPLATE_PROVIDER_MATCH
             : LOOM_TEMPLATE_PROVIDER_REJECT;
}

static bool loom_template_applicability_remap_contract_value(
    const loom_op_t* application_op,
    const loom_template_applicability_contract_t* contract,
    loom_value_id_t contract_value_id,
    loom_value_id_t* out_application_value_id) {
  loom_value_slice_t operands =
      loom_template_applicability_application_operands(application_op);
  for (uint16_t i = 0; i < contract->argument_count && i < operands.count;
       ++i) {
    if (contract->argument_ids[i] == contract_value_id) {
      *out_application_value_id = operands.values[i];
      return true;
    }
  }

  loom_value_slice_t results =
      loom_template_applicability_application_results(application_op);
  for (uint16_t i = 0; i < contract->result_count && i < results.count; ++i) {
    if (contract->result_ids[i] == contract_value_id) {
      *out_application_value_id = results.values[i];
      return true;
    }
  }

  return false;
}

static void loom_template_predicate_arg_initialize(
    loom_decision_predicate_operand_t* out_arg) {
  *out_arg = (loom_decision_predicate_operand_t){
      .facts = loom_value_facts_unknown(),
      .identity = LOOM_DECISION_OPERAND_IDENTITY_NONE,
  };
}

static bool loom_template_applicability_resolve_application_value_arg(
    const loom_module_t* application_module,
    const loom_template_applicability_facts_t* application_facts,
    loom_value_id_t value_id, loom_decision_predicate_operand_t* out_arg) {
  if (value_id >= application_module->values.count) {
    return false;
  }
  out_arg->identity = value_id;
  if (application_facts->values) {
    out_arg->facts =
        loom_value_fact_table_lookup(application_facts->values, value_id);
    (void)loom_condition_fact_set_apply_to_value_facts(
        &application_facts->path, application_facts->values, value_id,
        &out_arg->facts);
  }
  return true;
}

static bool loom_template_applicability_resolve_contract_predicate_arg(
    const loom_module_t* application_module, const loom_op_t* application_op,
    const loom_template_applicability_contract_t* contract,
    const loom_template_applicability_facts_t* application_facts,
    const loom_predicate_t* predicate, uint8_t argument_index,
    loom_decision_predicate_operand_t* out_arg) {
  loom_template_predicate_arg_initialize(out_arg);
  if (argument_index >= predicate->arg_count) {
    return false;
  }

  switch ((loom_predicate_arg_tag_t)predicate->arg_tags[argument_index]) {
    case LOOM_PRED_ARG_CONST:
      out_arg->facts =
          loom_value_facts_exact_i64(predicate->args[argument_index]);
      return true;
    case LOOM_PRED_ARG_VALUE: {
      int64_t raw_value_id = predicate->args[argument_index];
      if (raw_value_id < 0 || raw_value_id > UINT32_MAX) {
        return false;
      }
      loom_value_id_t contract_value_id = (loom_value_id_t)raw_value_id;
      loom_value_id_t application_value_id = LOOM_VALUE_ID_INVALID;
      if (!loom_template_applicability_remap_contract_value(
              application_op, contract, contract_value_id,
              &application_value_id)) {
        return false;
      }
      return loom_template_applicability_resolve_application_value_arg(
          application_module, application_facts, application_value_id, out_arg);
    }
    case LOOM_PRED_ARG_NONE:
    default:
      return false;
  }
}

static bool loom_template_applicability_resolve_application_predicate_arg(
    const loom_module_t* application_module,
    const loom_template_applicability_facts_t* application_facts,
    const loom_predicate_t* predicate, uint8_t argument_index,
    loom_decision_predicate_operand_t* out_arg) {
  loom_template_predicate_arg_initialize(out_arg);
  if (argument_index >= predicate->arg_count) {
    return false;
  }

  switch ((loom_predicate_arg_tag_t)predicate->arg_tags[argument_index]) {
    case LOOM_PRED_ARG_CONST:
      out_arg->facts =
          loom_value_facts_exact_i64(predicate->args[argument_index]);
      return true;
    case LOOM_PRED_ARG_VALUE: {
      const int64_t raw_value_id = predicate->args[argument_index];
      if (raw_value_id < 0 || raw_value_id > UINT32_MAX) {
        return false;
      }
      return loom_template_applicability_resolve_application_value_arg(
          application_module, application_facts, (loom_value_id_t)raw_value_id,
          out_arg);
    }
    case LOOM_PRED_ARG_NONE:
    default:
      return false;
  }
}

static loom_template_provider_feasibility_t
loom_template_applicability_feasibility_from_truth(
    loom_decision_truth_t truth) {
  switch (truth) {
    case LOOM_DECISION_TRUTH_FALSE:
      return LOOM_TEMPLATE_PROVIDER_REJECT;
    case LOOM_DECISION_TRUTH_TRUE:
      return LOOM_TEMPLATE_PROVIDER_MATCH;
    case LOOM_DECISION_TRUTH_UNKNOWN:
      return LOOM_TEMPLATE_PROVIDER_MAYBE;
  }
  IREE_ASSERT_UNREACHABLE("invalid decision truth value");
  IREE_BUILTIN_UNREACHABLE();
}

static bool loom_template_predicate_relation_kind(
    uint8_t predicate_kind, loom_symbolic_integer_relation_t* out_relation) {
  switch ((loom_predicate_kind_t)predicate_kind) {
    case LOOM_PREDICATE_EQ:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
      return true;
    case LOOM_PREDICATE_NE:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_NE;
      return true;
    case LOOM_PREDICATE_LT:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT;
      return true;
    case LOOM_PREDICATE_LE:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LE;
      return true;
    case LOOM_PREDICATE_GT:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GT;
      return true;
    case LOOM_PREDICATE_GE:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GE;
      return true;
    default:
      return false;
  }
}

static bool loom_template_predicate_arg_as_condition_operand(
    const loom_decision_predicate_operand_t* arg,
    loom_condition_integer_operand_t* out_operand) {
  *out_operand = (loom_condition_integer_operand_t){0};
  if (arg->identity != LOOM_DECISION_OPERAND_IDENTITY_NONE) {
    out_operand->kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE;
    out_operand->value_id = arg->identity;
    return true;
  }
  int64_t constant = 0;
  if (!loom_value_facts_as_exact_i64(arg->facts, &constant)) {
    return false;
  }
  out_operand->kind = LOOM_CONDITION_INTEGER_OPERAND_CONSTANT;
  out_operand->constant = constant;
  return true;
}

static loom_template_provider_feasibility_t
loom_template_applicability_evaluate_relation(
    const loom_decision_predicate_operand_t* lhs,
    const loom_decision_predicate_operand_t* rhs, uint8_t predicate_kind,
    const loom_template_applicability_facts_t* application_facts) {
  loom_condition_integer_relation_t queried_relation = {0};
  if (loom_template_predicate_relation_kind(predicate_kind,
                                            &queried_relation.relation) &&
      loom_template_predicate_arg_as_condition_operand(
          lhs, &queried_relation.left) &&
      loom_template_predicate_arg_as_condition_operand(
          rhs, &queried_relation.right)) {
    bool relation_result = false;
    if (loom_condition_fact_set_proves_integer_relation(
            &application_facts->path, application_facts->values,
            &queried_relation, &relation_result)) {
      return relation_result ? LOOM_TEMPLATE_PROVIDER_MATCH
                             : LOOM_TEMPLATE_PROVIDER_REJECT;
    }
  }

  const loom_decision_predicate_operand_t operands[3] = {*lhs, *rhs};
  return loom_template_applicability_feasibility_from_truth(
      loom_decision_predicate_evaluate(predicate_kind, operands));
}

static loom_template_provider_feasibility_t
loom_template_applicability_evaluate_resolved_predicate(
    const loom_template_applicability_facts_t* application_facts,
    const loom_predicate_t* predicate,
    loom_decision_predicate_operand_t args[3]) {
  switch ((loom_predicate_kind_t)predicate->kind) {
    case LOOM_PREDICATE_EQ:
    case LOOM_PREDICATE_NE:
    case LOOM_PREDICATE_LT:
    case LOOM_PREDICATE_LE:
    case LOOM_PREDICATE_GT:
    case LOOM_PREDICATE_GE:
      return loom_template_applicability_evaluate_relation(
          &args[0], &args[1], predicate->kind, application_facts);
    case LOOM_PREDICATE_MIN:
      return loom_template_applicability_evaluate_relation(
          &args[0], &args[1], LOOM_PREDICATE_GE, application_facts);
    case LOOM_PREDICATE_MAX:
      return loom_template_applicability_evaluate_relation(
          &args[0], &args[1], LOOM_PREDICATE_LE, application_facts);
    case LOOM_PREDICATE_MUL:
    case LOOM_PREDICATE_POW2:
    case LOOM_PREDICATE_RANGE:
    case LOOM_PREDICATE_NOT_NAN:
    case LOOM_PREDICATE_NOT_INF:
    case LOOM_PREDICATE_FINITE:
      return loom_template_applicability_feasibility_from_truth(
          loom_decision_predicate_evaluate(predicate->kind, args));
    case LOOM_PREDICATE_COUNT_:
      IREE_ASSERT_UNREACHABLE("invalid template predicate kind");
      IREE_BUILTIN_UNREACHABLE();
  }
  IREE_ASSERT_UNREACHABLE("invalid template predicate kind");
  IREE_BUILTIN_UNREACHABLE();
}

static bool loom_template_applicability_predicate_arity_is_valid(
    const loom_predicate_t* predicate) {
  const uint8_t expected_argument_count =
      loom_predicate_kind_argument_count(predicate->kind);
  return expected_argument_count != UINT8_MAX &&
         predicate->arg_count == expected_argument_count &&
         predicate->arg_count <= IREE_ARRAYSIZE(predicate->args);
}

static loom_template_provider_feasibility_t
loom_template_applicability_evaluate_contract_predicate(
    const loom_module_t* application_module, const loom_op_t* application_op,
    const loom_template_applicability_contract_t* contract,
    const loom_template_applicability_facts_t* application_facts,
    const loom_predicate_t* predicate) {
  if (!loom_template_applicability_predicate_arity_is_valid(predicate)) {
    return LOOM_TEMPLATE_PROVIDER_MAYBE;
  }

  loom_decision_predicate_operand_t args[3];
  for (uint8_t i = 0; i < predicate->arg_count; ++i) {
    if (!loom_template_applicability_resolve_contract_predicate_arg(
            application_module, application_op, contract, application_facts,
            predicate, i, &args[i])) {
      return LOOM_TEMPLATE_PROVIDER_MAYBE;
    }
  }
  return loom_template_applicability_evaluate_resolved_predicate(
      application_facts, predicate, args);
}

static loom_template_provider_feasibility_t
loom_template_applicability_evaluate_application_predicate(
    const loom_module_t* application_module,
    const loom_template_applicability_facts_t* application_facts,
    const loom_predicate_t* predicate) {
  if (!loom_template_applicability_predicate_arity_is_valid(predicate)) {
    return LOOM_TEMPLATE_PROVIDER_MAYBE;
  }

  loom_decision_predicate_operand_t args[3];
  for (uint8_t i = 0; i < predicate->arg_count; ++i) {
    if (!loom_template_applicability_resolve_application_predicate_arg(
            application_module, application_facts, predicate, i, &args[i])) {
      return LOOM_TEMPLATE_PROVIDER_MAYBE;
    }
  }
  return loom_template_applicability_evaluate_resolved_predicate(
      application_facts, predicate, args);
}

static loom_template_provider_feasibility_t
loom_template_applicability_evaluate_predicates(
    const loom_module_t* application_module, const loom_op_t* application_op,
    const loom_template_applicability_contract_t* contract,
    const loom_template_applicability_facts_t* application_facts) {
  loom_template_provider_feasibility_t feasibility =
      LOOM_TEMPLATE_PROVIDER_MATCH;
  for (uint16_t i = 0; i < contract->predicate_count; ++i) {
    const loom_template_provider_feasibility_t predicate_feasibility =
        loom_template_applicability_evaluate_contract_predicate(
            application_module, application_op, contract, application_facts,
            &contract->predicates[i]);
    if (predicate_feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) {
      return LOOM_TEMPLATE_PROVIDER_REJECT;
    }
    if (predicate_feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE) {
      feasibility = LOOM_TEMPLATE_PROVIDER_MAYBE;
    }
  }
  return feasibility;
}

static bool loom_template_applicability_application_has_ancestor(
    const loom_op_t* application_op, loom_op_kind_t ancestor_kind) {
  for (const loom_op_t* ancestor = application_op->parent_op; ancestor;
       ancestor = ancestor->parent_op) {
    if (ancestor->kind == ancestor_kind) {
      return true;
    }
  }
  return false;
}

static loom_template_provider_feasibility_t
loom_template_applicability_evaluate_target_condition_query_value(
    const loom_module_t* application_module,
    const loom_target_condition_t* condition,
    const loom_template_applicability_facts_t* application_facts,
    loom_value_id_t value_id) {
  if (application_facts->values == NULL ||
      condition->descriptor->project_query_predicate == NULL) {
    return LOOM_TEMPLATE_PROVIDER_MAYBE;
  }
  loom_value_fact_contextual_query_origin_t origin = {0};
  if (!loom_value_fact_table_query_contextual_query_origin(
          application_facts->values, application_module, value_id, &origin) ||
      origin.family_kind != loom_attr_as_parameterized_kind(condition->value)) {
    return LOOM_TEMPLATE_PROVIDER_MAYBE;
  }

  loom_predicate_t predicate = {0};
  if (!condition->descriptor->project_query_predicate(
          condition->value, origin.key, value_id, &predicate)) {
    return LOOM_TEMPLATE_PROVIDER_MAYBE;
  }
  return loom_template_applicability_evaluate_application_predicate(
      application_module, application_facts, &predicate);
}

static loom_template_provider_feasibility_t
loom_template_applicability_evaluate_target_condition_queries(
    const loom_module_t* application_module,
    const loom_target_condition_t* condition,
    const loom_template_applicability_facts_t* application_facts) {
  bool has_match = false;
  bool has_reject = false;
  for (iree_host_size_t i = 0;
       i < application_facts->path.integer_relation_count; ++i) {
    const loom_condition_integer_relation_t* relation =
        &application_facts->path.integer_relations[i];
    for (uint8_t operand_index = 0; operand_index < 2; ++operand_index) {
      const loom_condition_integer_operand_t operand =
          operand_index == 0 ? relation->left : relation->right;
      if (operand.kind != LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
        continue;
      }
      const loom_template_provider_feasibility_t feasibility =
          loom_template_applicability_evaluate_target_condition_query_value(
              application_module, condition, application_facts,
              operand.value_id);
      has_match |= feasibility == LOOM_TEMPLATE_PROVIDER_MATCH;
      has_reject |= feasibility == LOOM_TEMPLATE_PROVIDER_REJECT;
    }
  }
  if (has_match && has_reject) {
    return LOOM_TEMPLATE_PROVIDER_MAYBE;
  }
  if (has_match) {
    return LOOM_TEMPLATE_PROVIDER_MATCH;
  }
  if (has_reject) {
    return LOOM_TEMPLATE_PROVIDER_REJECT;
  }
  return LOOM_TEMPLATE_PROVIDER_MAYBE;
}

static loom_template_provider_feasibility_t
loom_template_applicability_evaluate_target_conditions(
    const loom_module_t* application_module,
    const loom_template_applicability_contract_t* contract,
    const loom_target_facts_t* target_facts,
    const loom_template_applicability_facts_t* application_facts,
    const loom_target_condition_t** out_unresolved_condition) {
  *out_unresolved_condition = NULL;
  loom_template_provider_feasibility_t feasibility =
      LOOM_TEMPLATE_PROVIDER_MATCH;
  for (uint16_t i = 0; i < contract->target_condition_count; ++i) {
    const loom_target_condition_t* condition = &contract->target_conditions[i];
    const loom_target_condition_outcome_t outcome =
        loom_target_condition_evaluate(condition->descriptor, condition->value,
                                       target_facts);
    switch (outcome) {
      case LOOM_TARGET_CONDITION_MATCH:
        break;
      case LOOM_TARGET_CONDITION_UNKNOWN:
      case LOOM_TARGET_CONDITION_UNBOUND: {
        const loom_template_provider_feasibility_t query_feasibility =
            loom_template_applicability_evaluate_target_condition_queries(
                application_module, condition, application_facts);
        if (query_feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) {
          return LOOM_TEMPLATE_PROVIDER_REJECT;
        }
        if (query_feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE) {
          feasibility = LOOM_TEMPLATE_PROVIDER_MAYBE;
          if (*out_unresolved_condition == NULL) {
            *out_unresolved_condition = condition;
          }
        }
        break;
      }
      case LOOM_TARGET_CONDITION_REJECT:
        return LOOM_TEMPLATE_PROVIDER_REJECT;
      default:
        IREE_ASSERT_UNREACHABLE("target condition returned an invalid outcome");
        IREE_BUILTIN_UNREACHABLE();
    }
  }
  return feasibility;
}

static void loom_template_applicability_classify_constraints(
    const loom_module_t* application_module, const loom_op_t* application_op,
    const loom_template_applicability_contract_t* contract,
    const loom_template_applicability_target_t* application_target,
    const loom_template_applicability_facts_t* application_facts,
    loom_template_provider_classification_t* classification) {
  const loom_target_condition_t* unresolved_target_condition = NULL;
  const loom_template_provider_feasibility_t target_condition_feasibility =
      loom_template_applicability_evaluate_target_conditions(
          application_module, contract, application_target->facts,
          application_facts, &unresolved_target_condition);
  if (target_condition_feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) {
    return;
  }

  loom_template_provider_feasibility_t predicate_feasibility =
      LOOM_TEMPLATE_PROVIDER_MATCH;
  if (contract->predicate_count > 0) {
    predicate_feasibility = loom_template_applicability_evaluate_predicates(
        application_module, application_op, contract, application_facts);
    if (predicate_feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) {
      return;
    }
  }

  classification->feasibility =
      classification->target_feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE ||
              target_condition_feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE ||
              predicate_feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE
          ? LOOM_TEMPLATE_PROVIDER_MAYBE
          : LOOM_TEMPLATE_PROVIDER_MATCH;
  if (classification->target_feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE) {
    classification->unresolved_reason =
        LOOM_TEMPLATE_PROVIDER_UNRESOLVED_TARGET_IDENTITY;
  } else if (target_condition_feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE) {
    classification->unresolved_reason =
        LOOM_TEMPLATE_PROVIDER_UNRESOLVED_TARGET_CONDITION;
    classification->unresolved_target_condition = unresolved_target_condition;
  } else if (predicate_feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE) {
    classification->unresolved_reason =
        LOOM_TEMPLATE_PROVIDER_UNRESOLVED_VALUE_PREDICATE;
  }
}

void loom_template_applicability_classify_provider(
    const loom_module_t* application_module, const loom_op_t* application_op,
    const loom_template_provider_summary_t* provider,
    const loom_template_applicability_target_t* application_target,
    const loom_template_applicability_facts_t* application_facts,
    loom_template_provider_classification_t* out_classification) {
  *out_classification = (loom_template_provider_classification_t){
      .feasibility = LOOM_TEMPLATE_PROVIDER_REJECT,
      .target_feasibility = LOOM_TEMPLATE_PROVIDER_REJECT,
  };
  const loom_template_applicability_contract_t contract =
      loom_template_applicability_provider_contract(provider);
  out_classification->target_feasibility =
      loom_template_applicability_target_feasibility(
          application_module, application_target, &contract);
  if (out_classification->target_feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) {
    return;
  }

  loom_template_applicability_classify_constraints(
      application_module, application_op, &contract, application_target,
      application_facts, out_classification);
}

void loom_template_applicability_classify_contract(
    const loom_module_t* application_module, const loom_op_t* application_op,
    const loom_template_applicability_contract_t* contract,
    const loom_template_applicability_target_t* application_target,
    const loom_template_applicability_facts_t* application_facts,
    loom_template_provider_classification_t* out_classification) {
  *out_classification = (loom_template_provider_classification_t){
      .feasibility = LOOM_TEMPLATE_PROVIDER_REJECT,
      .target_feasibility = LOOM_TEMPLATE_PROVIDER_REJECT,
  };
  out_classification->target_feasibility =
      loom_template_applicability_target_feasibility(
          application_module, application_target, contract);
  if (out_classification->target_feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) {
    return;
  }
  loom_template_applicability_classify_constraints(
      application_module, application_op, contract, application_target,
      application_facts, out_classification);
}

bool loom_template_applicability_requires_application_facts(
    const loom_op_t* application_op,
    const loom_template_applicability_contract_t* contract,
    const loom_target_facts_t* target_facts) {
  if (contract->predicate_count > 0) {
    return true;
  }
  if (!loom_template_applicability_application_has_ancestor(application_op,
                                                            LOOM_OP_SCF_IF)) {
    return false;
  }
  for (uint16_t i = 0; i < contract->target_condition_count; ++i) {
    const loom_target_condition_t* condition = &contract->target_conditions[i];
    if (condition->descriptor->project_query_predicate == NULL) {
      continue;
    }
    const loom_target_condition_outcome_t outcome =
        loom_target_condition_evaluate(condition->descriptor, condition->value,
                                       target_facts);
    if (outcome == LOOM_TARGET_CONDITION_UNKNOWN ||
        outcome == LOOM_TARGET_CONDITION_UNBOUND) {
      return true;
    }
  }
  return false;
}
