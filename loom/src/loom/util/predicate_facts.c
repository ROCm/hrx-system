// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/predicate_facts.h"

#include <stdint.h>

#include "iree/base/internal/math.h"
#include "loom/analysis/integer_relation.h"
#include "loom/ir/context.h"
#include "loom/util/fact_ordinal_map.h"
#include "loom/util/fact_relation.h"

// At most this many aliases bound each predicate's scan to constant work.
#define LOOM_VALUE_FACT_ALIAS_LINEAR_SCAN_LIMIT 16

// At most this many predicates bound the number of full alias-list scans.
#define LOOM_VALUE_FACT_PREDICATE_LINEAR_SCAN_LIMIT 4

uint16_t loom_value_facts_export_integer_refinements(
    loom_value_id_t value, loom_value_facts_t inferred_facts,
    loom_value_facts_t proven_facts, loom_predicate_t* predicates,
    uint16_t capacity) {
  if (capacity == 0 || value == LOOM_VALUE_ID_INVALID ||
      loom_value_facts_is_float(proven_facts) ||
      loom_value_facts_is_unknown(proven_facts)) {
    return 0;
  }

  uint16_t count = 0;
  const bool exports_range = loom_value_facts_is_float(inferred_facts) ||
                             proven_facts.range_lo > inferred_facts.range_lo ||
                             proven_facts.range_hi < inferred_facts.range_hi;
  if (exports_range) {
    if (loom_value_facts_is_exact(proven_facts)) {
      predicates[count++] = (loom_predicate_t){
          .kind = LOOM_PREDICATE_EQ,
          .arg_count = 2,
          .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                       LOOM_PRED_ARG_NONE},
          .args = {value, proven_facts.range_lo, 0},
      };
    } else {
      predicates[count++] = (loom_predicate_t){
          .kind = LOOM_PREDICATE_RANGE,
          .arg_count = 3,
          .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                       LOOM_PRED_ARG_CONST},
          .args = {value, proven_facts.range_lo, proven_facts.range_hi},
      };
    }
  }

  // Equality carries the complete integer value and therefore implies every
  // other representable refinement.
  if (loom_value_facts_is_exact(proven_facts)) {
    return count;
  }

  if (count < capacity && proven_facts.known_divisor > 1 &&
      (loom_value_facts_is_float(inferred_facts) ||
       inferred_facts.known_divisor % proven_facts.known_divisor != 0)) {
    predicates[count++] = (loom_predicate_t){
        .kind = LOOM_PREDICATE_MULTIPLE_OF,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                     LOOM_PRED_ARG_NONE},
        .args = {value, proven_facts.known_divisor, 0},
    };
  }

  if (count < capacity && loom_value_facts_is_power_of_two(proven_facts) &&
      !loom_value_facts_is_power_of_two(inferred_facts)) {
    predicates[count++] = (loom_predicate_t){
        .kind = LOOM_PREDICATE_POWER_OF_TWO,
        .arg_count = 1,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_NONE,
                     LOOM_PRED_ARG_NONE},
        .args = {value, 0, 0},
    };
  } else if (count < capacity && loom_value_facts_is_non_zero(proven_facts) &&
             !loom_value_facts_is_non_zero(inferred_facts) &&
             !loom_value_facts_is_power_of_two(inferred_facts) &&
             !(exports_range &&
               (proven_facts.range_hi < 0 || proven_facts.range_lo > 0))) {
    predicates[count++] = (loom_predicate_t){
        .kind = LOOM_PREDICATE_NE,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                     LOOM_PRED_ARG_NONE},
        .args = {value, 0, 0},
    };
  }
  return count;
}

static bool loom_value_fact_table_resolve_predicate_operand(
    const loom_value_fact_table_t* table, const loom_predicate_t* predicate,
    uint8_t argument_index, loom_decision_predicate_operand_t* out_operand) {
  *out_operand = (loom_decision_predicate_operand_t){
      .facts = loom_value_facts_unknown(),
      .identity = LOOM_DECISION_OPERAND_IDENTITY_NONE,
  };
  switch ((loom_predicate_arg_tag_t)predicate->arg_tags[argument_index]) {
    case LOOM_PRED_ARG_CONST:
      out_operand->facts =
          loom_value_facts_exact_i64(predicate->args[argument_index]);
      return true;
    case LOOM_PRED_ARG_VALUE: {
      if (predicate->args[argument_index] < 0) {
        return false;
      }
      const loom_value_id_t value_id =
          (loom_value_id_t)predicate->args[argument_index];
      if (table) {
        out_operand->facts = loom_value_fact_table_lookup(table, value_id);
      }
      out_operand->identity =
          loom_value_fact_table_query_identity(table, value_id);
      return true;
    }
    case LOOM_PRED_ARG_NONE:
    case LOOM_PRED_ARG_COUNT_:
      return false;
  }
  return false;
}

static loom_decision_truth_t
loom_value_fact_table_evaluate_retained_signed_relation(
    const loom_value_fact_table_t* table, loom_predicate_kind_t predicate_kind,
    loom_predicate_arg_tag_t lhs_tag, int64_t lhs_argument,
    const loom_decision_predicate_operand_t* lhs_operand,
    loom_predicate_arg_tag_t rhs_tag, int64_t rhs_argument,
    const loom_decision_predicate_operand_t* rhs_operand) {
  if (loom_value_facts_is_float(lhs_operand->facts) ||
      loom_value_facts_is_float(rhs_operand->facts)) {
    return LOOM_DECISION_TRUTH_UNKNOWN;
  }

  loom_symbolic_integer_relation_t relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
  if (!loom_symbolic_integer_relation_from_predicate_kind(predicate_kind,
                                                          &relation)) {
    return LOOM_DECISION_TRUTH_UNKNOWN;
  }

  loom_integer_order_outcome_bits_t outcomes =
      LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL;
  if (lhs_tag == LOOM_PRED_ARG_VALUE && rhs_tag == LOOM_PRED_ARG_VALUE) {
    outcomes = loom_value_fact_table_query_signed_integer_order(
        table, (loom_value_id_t)lhs_argument, (loom_value_id_t)rhs_argument);
  } else if (lhs_tag == LOOM_PRED_ARG_VALUE && rhs_tag == LOOM_PRED_ARG_CONST) {
    outcomes = loom_value_fact_table_query_signed_integer_literal_order(
        table, (loom_value_id_t)lhs_argument, rhs_argument);
  } else if (lhs_tag == LOOM_PRED_ARG_CONST && rhs_tag == LOOM_PRED_ARG_VALUE) {
    outcomes = loom_integer_order_outcomes_swap(
        loom_value_fact_table_query_signed_integer_literal_order(
            table, (loom_value_id_t)rhs_argument, lhs_argument));
  }

  bool result = false;
  if (!loom_symbolic_integer_relation_resolve(outcomes, relation, &result)) {
    return LOOM_DECISION_TRUTH_UNKNOWN;
  }
  return result ? LOOM_DECISION_TRUTH_TRUE : LOOM_DECISION_TRUTH_FALSE;
}

static loom_decision_truth_t loom_value_fact_table_evaluate_dynamic_range(
    const loom_value_fact_table_t* table, const loom_predicate_t* predicate,
    const loom_decision_predicate_operand_t operands[3]) {
  const loom_decision_truth_t lower =
      loom_value_fact_table_evaluate_retained_signed_relation(
          table, LOOM_PREDICATE_LE,
          (loom_predicate_arg_tag_t)predicate->arg_tags[1], predicate->args[1],
          &operands[1], (loom_predicate_arg_tag_t)predicate->arg_tags[0],
          predicate->args[0], &operands[0]);
  const loom_decision_truth_t upper =
      loom_value_fact_table_evaluate_retained_signed_relation(
          table, LOOM_PREDICATE_LE,
          (loom_predicate_arg_tag_t)predicate->arg_tags[0], predicate->args[0],
          &operands[0], (loom_predicate_arg_tag_t)predicate->arg_tags[2],
          predicate->args[2], &operands[2]);
  if (lower == LOOM_DECISION_TRUTH_FALSE ||
      upper == LOOM_DECISION_TRUTH_FALSE) {
    return LOOM_DECISION_TRUTH_FALSE;
  }
  return lower == LOOM_DECISION_TRUTH_TRUE && upper == LOOM_DECISION_TRUTH_TRUE
             ? LOOM_DECISION_TRUTH_TRUE
             : LOOM_DECISION_TRUTH_UNKNOWN;
}

loom_decision_truth_t loom_value_fact_table_evaluate_predicate(
    const loom_value_fact_table_t* table, const loom_predicate_t* predicate) {
  loom_decision_predicate_operand_t operands[3] = {0};
  for (uint8_t i = 0; i < predicate->arg_count; ++i) {
    if (!loom_value_fact_table_resolve_predicate_operand(table, predicate, i,
                                                         &operands[i])) {
      return LOOM_DECISION_TRUTH_UNKNOWN;
    }
  }

  const loom_decision_truth_t scalar_result =
      loom_decision_predicate_evaluate(predicate->kind, operands);
  if (scalar_result != LOOM_DECISION_TRUTH_UNKNOWN) {
    return scalar_result;
  }
  if (predicate->kind == LOOM_PREDICATE_RANGE) {
    return loom_value_fact_table_evaluate_dynamic_range(table, predicate,
                                                        operands);
  }
  return loom_value_fact_table_evaluate_retained_signed_relation(
      table, predicate->kind, (loom_predicate_arg_tag_t)predicate->arg_tags[0],
      predicate->args[0], &operands[0],
      (loom_predicate_arg_tag_t)predicate->arg_tags[1], predicate->args[1],
      &operands[1]);
}

bool loom_value_fact_table_proves_predicate_list(
    const loom_value_fact_table_t* table, const loom_predicate_t* predicates,
    uint16_t predicate_count) {
  for (uint16_t i = 0; i < predicate_count; ++i) {
    if (loom_value_fact_table_evaluate_predicate(table, &predicates[i]) !=
        LOOM_DECISION_TRUTH_TRUE) {
      return false;
    }
  }
  return true;
}

static bool loom_value_fact_table_predicate_arguments_equal(
    const loom_value_fact_table_t* table, loom_predicate_arg_tag_t lhs_tag,
    int64_t lhs_argument, loom_predicate_arg_tag_t rhs_tag,
    int64_t rhs_argument) {
  if (lhs_tag != rhs_tag) {
    return false;
  }
  if (lhs_tag == LOOM_PRED_ARG_VALUE) {
    if (lhs_argument < 0 || rhs_argument < 0) {
      return false;
    }
    return loom_value_fact_table_query_identity(
               table, (loom_value_id_t)lhs_argument) ==
           loom_value_fact_table_query_identity(table,
                                                (loom_value_id_t)rhs_argument);
  }
  return lhs_argument == rhs_argument;
}

static bool loom_value_fact_table_predicate_operands_equal(
    const loom_value_fact_table_t* table, const loom_predicate_t* lhs,
    const loom_predicate_t* rhs, bool swapped) {
  if (lhs->arg_count != rhs->arg_count) {
    return false;
  }
  for (uint8_t i = 0; i < lhs->arg_count; ++i) {
    const uint8_t rhs_index = swapped ? (uint8_t)(lhs->arg_count - i - 1) : i;
    if (!loom_value_fact_table_predicate_arguments_equal(
            table, (loom_predicate_arg_tag_t)lhs->arg_tags[i], lhs->args[i],
            (loom_predicate_arg_tag_t)rhs->arg_tags[rhs_index],
            rhs->args[rhs_index])) {
      return false;
    }
  }
  return true;
}

static bool loom_value_fact_table_unsigned_relation_outcomes(
    loom_predicate_kind_t kind,
    loom_integer_order_outcome_bits_t* out_outcomes) {
  switch (kind) {
    case LOOM_PREDICATE_EQ:
      *out_outcomes = LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL;
      return true;
    case LOOM_PREDICATE_NE:
      *out_outcomes = LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS |
                      LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER;
      return true;
    case LOOM_PREDICATE_ULT:
      *out_outcomes = LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS;
      return true;
    case LOOM_PREDICATE_ULE:
      *out_outcomes = LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS |
                      LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL;
      return true;
    case LOOM_PREDICATE_UGT:
      *out_outcomes = LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER;
      return true;
    case LOOM_PREDICATE_UGE:
      *out_outcomes = LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL |
                      LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER;
      return true;
    default:
      return false;
  }
}

static bool loom_value_fact_table_relation_predicate_implies(
    const loom_value_fact_table_t* table, const loom_predicate_t* premise,
    const loom_predicate_t* conclusion) {
  loom_symbolic_integer_relation_t premise_relation =
      LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
  loom_symbolic_integer_relation_t conclusion_relation =
      LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
  if (loom_symbolic_integer_relation_from_predicate_kind(premise->kind,
                                                         &premise_relation) &&
      loom_symbolic_integer_relation_from_predicate_kind(
          conclusion->kind, &conclusion_relation)) {
    if (loom_value_fact_table_predicate_operands_equal(
            table, premise, conclusion, /*swapped=*/true)) {
      premise_relation = loom_symbolic_integer_relation_swap(premise_relation);
    } else if (!loom_value_fact_table_predicate_operands_equal(
                   table, premise, conclusion, /*swapped=*/false)) {
      return false;
    }
    bool result = false;
    return loom_symbolic_integer_relation_implies(
               premise_relation, conclusion_relation, &result) &&
           result;
  }

  loom_integer_order_outcome_bits_t premise_outcomes = 0;
  loom_integer_order_outcome_bits_t conclusion_outcomes = 0;
  if (!loom_value_fact_table_unsigned_relation_outcomes(premise->kind,
                                                        &premise_outcomes) ||
      !loom_value_fact_table_unsigned_relation_outcomes(conclusion->kind,
                                                        &conclusion_outcomes)) {
    return false;
  }
  if (loom_value_fact_table_predicate_operands_equal(table, premise, conclusion,
                                                     /*swapped=*/true)) {
    premise_outcomes = loom_integer_order_outcomes_swap(premise_outcomes);
  } else if (!loom_value_fact_table_predicate_operands_equal(
                 table, premise, conclusion, /*swapped=*/false)) {
    return false;
  }
  return (premise_outcomes & ~conclusion_outcomes) == 0;
}

static bool loom_value_fact_table_range_bound_implies(
    const loom_value_fact_table_t* table, loom_predicate_arg_tag_t premise_tag,
    int64_t premise_argument, loom_predicate_arg_tag_t conclusion_tag,
    int64_t conclusion_argument, bool is_lower_bound) {
  if (premise_tag == LOOM_PRED_ARG_CONST &&
      conclusion_tag == LOOM_PRED_ARG_CONST) {
    return is_lower_bound ? premise_argument >= conclusion_argument
                          : premise_argument <= conclusion_argument;
  }
  return loom_value_fact_table_predicate_arguments_equal(
      table, premise_tag, premise_argument, conclusion_tag,
      conclusion_argument);
}

bool loom_value_fact_table_predicate_implies(
    const loom_value_fact_table_t* table, const loom_predicate_t* premise,
    const loom_predicate_t* conclusion) {
  if (premise->arg_count == conclusion->arg_count &&
      premise->kind == conclusion->kind &&
      loom_value_fact_table_predicate_operands_equal(table, premise, conclusion,
                                                     /*swapped=*/false)) {
    return true;
  }

  if (loom_value_fact_table_relation_predicate_implies(table, premise,
                                                       conclusion)) {
    return true;
  }

  if (premise->kind == LOOM_PREDICATE_MULTIPLE_OF &&
      conclusion->kind == LOOM_PREDICATE_MULTIPLE_OF &&
      loom_value_fact_table_predicate_arguments_equal(
          table, (loom_predicate_arg_tag_t)premise->arg_tags[0],
          premise->args[0], (loom_predicate_arg_tag_t)conclusion->arg_tags[0],
          conclusion->args[0]) &&
      premise->arg_tags[1] == LOOM_PRED_ARG_CONST &&
      conclusion->arg_tags[1] == LOOM_PRED_ARG_CONST && premise->args[1] > 0 &&
      conclusion->args[1] > 0) {
    return premise->args[1] % conclusion->args[1] == 0;
  }

  if (premise->kind == LOOM_PREDICATE_RANGE &&
      conclusion->kind == LOOM_PREDICATE_RANGE &&
      loom_value_fact_table_predicate_arguments_equal(
          table, (loom_predicate_arg_tag_t)premise->arg_tags[0],
          premise->args[0], (loom_predicate_arg_tag_t)conclusion->arg_tags[0],
          conclusion->args[0])) {
    return loom_value_fact_table_range_bound_implies(
               table, (loom_predicate_arg_tag_t)premise->arg_tags[1],
               premise->args[1],
               (loom_predicate_arg_tag_t)conclusion->arg_tags[1],
               conclusion->args[1], /*is_lower_bound=*/true) &&
           loom_value_fact_table_range_bound_implies(
               table, (loom_predicate_arg_tag_t)premise->arg_tags[2],
               premise->args[2],
               (loom_predicate_arg_tag_t)conclusion->arg_tags[2],
               conclusion->args[2], /*is_lower_bound=*/false);
  }

  if (premise->kind == LOOM_PREDICATE_FINITE &&
      (conclusion->kind == LOOM_PREDICATE_NOT_NAN ||
       conclusion->kind == LOOM_PREDICATE_NOT_INF) &&
      premise->arg_count == 1 && conclusion->arg_count == 1) {
    return loom_value_fact_table_predicate_operands_equal(
        table, premise, conclusion, /*swapped=*/false);
  }

  return false;
}

bool loom_value_fact_table_carrier_implies_predicate(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* carrier, const loom_predicate_t* conclusion) {
  if (!iree_all_bits_set(carrier->traits,
                         LOOM_TRAIT_PURE | LOOM_TRAIT_FACT_IDENTITY)) {
    return false;
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, carrier);
  const loom_attribute_t* attributes = loom_op_const_attrs(carrier);
  for (uint8_t i = 0; i < vtable->attribute_count; ++i) {
    if (vtable->attr_descriptors[i].attr_kind != LOOM_ATTR_PREDICATE_LIST) {
      continue;
    }
    const loom_attribute_t* premises = &attributes[i];
    for (uint16_t j = 0; j < premises->count; ++j) {
      if (loom_value_fact_table_predicate_implies(
              table, &premises->predicate_list[j], conclusion)) {
        return true;
      }
    }
  }
  return false;
}

static bool loom_value_fact_alias_map_resolve(
    const loom_value_fact_table_t* table,
    const loom_value_fact_ordinal_map_t* alias_map,
    const loom_value_facts_t* alias_facts, loom_value_id_t value_id,
    bool* out_is_alias, uint16_t* out_alias_ordinal,
    loom_value_facts_t* out_facts) {
  if (out_is_alias) {
    *out_is_alias = false;
  }
  if (out_alias_ordinal) {
    *out_alias_ordinal = 0;
  }
  uint16_t alias_ordinal = 0;
  if (loom_value_fact_ordinal_map_find(alias_map, value_id, &alias_ordinal)) {
    if (out_is_alias) {
      *out_is_alias = true;
    }
    if (out_alias_ordinal) {
      *out_alias_ordinal = alias_ordinal;
    }
    *out_facts = alias_facts[alias_ordinal];
    return true;
  }
  return loom_value_fact_table_try_lookup(table, value_id, out_facts);
}

static void loom_value_fact_alias_map_refine_relation(
    const loom_value_fact_table_t* table,
    const loom_value_fact_ordinal_map_t* alias_map, uint8_t predicate_kind,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_value_facts_t* inout_facts) {
  bool lhs_is_alias = false;
  bool rhs_is_alias = false;
  uint16_t lhs_ordinal = 0;
  uint16_t rhs_ordinal = 0;
  loom_value_facts_t lhs_facts = loom_value_facts_unknown();
  loom_value_facts_t rhs_facts = loom_value_facts_unknown();
  if (!loom_value_fact_alias_map_resolve(table, alias_map, inout_facts, lhs,
                                         &lhs_is_alias, &lhs_ordinal,
                                         &lhs_facts) ||
      !loom_value_fact_alias_map_resolve(table, alias_map, inout_facts, rhs,
                                         &rhs_is_alias, &rhs_ordinal,
                                         &rhs_facts) ||
      (!lhs_is_alias && !rhs_is_alias) ||
      loom_value_facts_is_float(lhs_facts) ||
      loom_value_facts_is_float(rhs_facts)) {
    return;
  }

  loom_value_facts_t* lhs_result =
      lhs_is_alias ? &inout_facts[lhs_ordinal] : NULL;
  loom_value_facts_t* rhs_result =
      rhs_is_alias ? &inout_facts[rhs_ordinal] : NULL;
  (void)loom_value_facts_refine_relation(predicate_kind, lhs_facts, rhs_facts,
                                         lhs_result, rhs_result);
}

static void loom_value_fact_table_apply_alias_predicates_with_map(
    const loom_value_fact_table_t* table,
    const loom_value_fact_ordinal_map_t* alias_map,
    const loom_predicate_t* predicates, uint16_t predicate_count,
    loom_value_facts_t* inout_facts) {
  // Apply predicates whose bounds are already literals or exact aliases.
  // Predicates retaining non-exact value operands are consumed by the
  // relational interval refinement below.
  for (uint16_t i = 0; i < predicate_count; ++i) {
    const loom_predicate_t* predicate = &predicates[i];
    if (predicate->arg_count == 0 ||
        predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE ||
        predicate->args[0] < 0) {
      continue;
    }
    uint16_t target_ordinal = 0;
    if (!loom_value_fact_ordinal_map_find(
            alias_map, (loom_value_id_t)predicate->args[0], &target_ordinal)) {
      continue;
    }

    loom_predicate_t resolved_predicate = *predicate;
    for (uint8_t argument_index = 1;
         argument_index < resolved_predicate.arg_count; ++argument_index) {
      if (resolved_predicate.arg_tags[argument_index] != LOOM_PRED_ARG_VALUE ||
          resolved_predicate.args[argument_index] < 0) {
        continue;
      }
      loom_value_facts_t argument_facts = loom_value_facts_unknown();
      if (!loom_value_fact_alias_map_resolve(
              table, alias_map, inout_facts,
              (loom_value_id_t)resolved_predicate.args[argument_index],
              /*out_is_alias=*/NULL, /*out_alias_ordinal=*/NULL,
              &argument_facts)) {
        continue;
      }
      int64_t exact_value = 0;
      if (!loom_value_facts_as_exact_i64(argument_facts, &exact_value)) {
        continue;
      }
      resolved_predicate.arg_tags[argument_index] = LOOM_PRED_ARG_CONST;
      resolved_predicate.args[argument_index] = exact_value;
    }
    loom_value_facts_apply_predicate(&inout_facts[target_ordinal],
                                     &resolved_predicate);
  }

  // Relational predicates constrain aliases from the known interval of their
  // counterpart. RANGE decomposes into lower <= value <= upper so each dynamic
  // endpoint contributes independently. External facts are read-only; only
  // aliases have result slots.
  for (uint16_t i = 0; i < predicate_count; ++i) {
    const loom_predicate_t* predicate = &predicates[i];
    if (predicate->kind == LOOM_PREDICATE_RANGE) {
      if (predicate->arg_count == 3 &&
          predicate->arg_tags[0] == LOOM_PRED_ARG_VALUE &&
          predicate->args[0] >= 0) {
        // Literal endpoints were already applied above. Only dynamic
        // endpoints require relational refinement.
        if (predicate->arg_tags[1] == LOOM_PRED_ARG_VALUE &&
            predicate->args[1] >= 0) {
          loom_value_fact_alias_map_refine_relation(
              table, alias_map, LOOM_PREDICATE_LE,
              (loom_value_id_t)predicate->args[1],
              (loom_value_id_t)predicate->args[0], inout_facts);
        }
        if (predicate->arg_tags[2] == LOOM_PRED_ARG_VALUE &&
            predicate->args[2] >= 0) {
          loom_value_fact_alias_map_refine_relation(
              table, alias_map, LOOM_PREDICATE_LE,
              (loom_value_id_t)predicate->args[0],
              (loom_value_id_t)predicate->args[2], inout_facts);
        }
      }
      continue;
    }
    if (predicate->arg_count != 2 ||
        predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE ||
        predicate->arg_tags[1] != LOOM_PRED_ARG_VALUE ||
        predicate->args[0] < 0 || predicate->args[1] < 0) {
      continue;
    }
    loom_value_fact_alias_map_refine_relation(
        table, alias_map, predicate->kind, (loom_value_id_t)predicate->args[0],
        (loom_value_id_t)predicate->args[1], inout_facts);
  }
}

iree_status_t loom_value_fact_table_apply_alias_predicates(
    loom_value_fact_table_t* table, const loom_value_id_t* values,
    uint16_t value_count, const loom_predicate_t* predicates,
    uint16_t predicate_count, loom_value_facts_t* inout_facts) {
  const bool use_index =
      value_count > LOOM_VALUE_FACT_ALIAS_LINEAR_SCAN_LIMIT &&
      predicate_count > LOOM_VALUE_FACT_PREDICATE_LINEAR_SCAN_LIMIT;
  loom_value_fact_ordinal_map_t alias_map = {0};
  IREE_RETURN_IF_ERROR(loom_value_fact_ordinal_map_initialize(
      table, values, value_count, use_index, &alias_map));

  loom_value_fact_table_apply_alias_predicates_with_map(
      table, &alias_map, predicates, predicate_count, inout_facts);
  loom_value_fact_ordinal_map_reset(&alias_map);
  return iree_ok_status();
}
