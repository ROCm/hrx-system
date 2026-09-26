// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_expr_proof.h"

#include <string.h>

#include "loom/analysis/condition_facts.h"
#include "loom/analysis/symbolic_projection.h"
#include "loom/analysis/symbolic_value.h"
#include "loom/ir/attribute.h"
#include "loom/util/adaptive_sort.h"

#define LOOM_SYMBOLIC_EXPR_SELECT_CURSOR_INLINE_CAPACITY 4
#define LOOM_SYMBOLIC_EXPR_SELECT_ASSUMPTION_INLINE_CAPACITY 8

typedef enum loom_symbolic_expr_proof_scope_e {
  LOOM_SYMBOLIC_EXPR_PROOF_SCOPE_ACTIVE_FACTS = 0,
  LOOM_SYMBOLIC_EXPR_PROOF_SCOPE_CONDITION_ASSUMPTION = 1,
  LOOM_SYMBOLIC_EXPR_PROOF_SCOPE_SELECT_CASES = 2,
} loom_symbolic_expr_proof_scope_t;

static iree_status_t loom_symbolic_expr_proof_ensure_scratch_terms(
    loom_symbolic_expr_context_t* context, iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= context->scratch_term_capacity) {
    return iree_ok_status();
  }
  void* terms = context->scratch_terms;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      context->arena, 0, minimum_capacity, sizeof(*context->scratch_terms),
      &context->scratch_term_capacity, &terms));
  context->scratch_terms = (loom_symbolic_term_t*)terms;
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_proof_stabilize_scratch_terms(
    loom_symbolic_expr_context_t* context, iree_host_size_t term_count,
    loom_symbolic_term_t* stack_terms, iree_host_size_t stack_term_capacity,
    const loom_symbolic_term_t** out_terms) {
  if (term_count == 0) {
    *out_terms = NULL;
    return iree_ok_status();
  }
  loom_symbolic_term_t* stable_terms = stack_terms;
  if (term_count > stack_term_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, term_count,
                                                   sizeof(*stable_terms),
                                                   (void**)&stable_terms));
  }
  memcpy(stable_terms, context->scratch_terms,
         term_count * sizeof(*stable_terms));
  *out_terms = stable_terms;
  return iree_ok_status();
}

static bool loom_symbolic_expr_proof_term_less(
    const loom_symbolic_term_t* lhs, const loom_symbolic_term_t* rhs) {
  return lhs->value_id < rhs->value_id;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_symbolic_expr_proof_sort_terms,
                          loom_symbolic_term_t,
                          loom_symbolic_expr_proof_term_less)

static bool loom_symbolic_expr_proof_checked_term_count(
    iree_host_size_t left_count, iree_host_size_t right_count,
    iree_host_size_t* out_count) {
  if (left_count > IREE_HOST_SIZE_MAX - right_count) {
    return false;
  }
  *out_count = left_count + right_count;
  return true;
}

static bool loom_symbolic_expr_proof_constant_value(
    const loom_symbolic_expr_t* expression, int64_t* out_value) {
  if (!loom_symbolic_expr_is_constant(expression)) {
    return false;
  }
  *out_value = expression->constant;
  return true;
}

//===----------------------------------------------------------------------===//
// Linear and interval proofs
//===----------------------------------------------------------------------===//

static loom_symbolic_proof_result_t loom_symbolic_expr_prove_le_by_facts(
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression) {
  if (loom_value_facts_is_float(left_expression->facts) ||
      loom_value_facts_is_float(right_expression->facts)) {
    return LOOM_SYMBOLIC_PROOF_UNKNOWN;
  }
  if (left_expression->facts.range_hi <= right_expression->facts.range_lo) {
    return LOOM_SYMBOLIC_PROOF_TRUE;
  }
  if (left_expression->facts.range_lo > right_expression->facts.range_hi) {
    return LOOM_SYMBOLIC_PROOF_FALSE;
  }
  return LOOM_SYMBOLIC_PROOF_UNKNOWN;
}

static bool loom_symbolic_expr_accumulate_checked(int64_t term_min,
                                                  int64_t term_max,
                                                  int64_t* inout_min,
                                                  int64_t* inout_max) {
  int64_t new_min = 0;
  int64_t new_max = 0;
  if (!iree_checked_add_i64(*inout_min, term_min, &new_min)) {
    return false;
  }
  if (!iree_checked_add_i64(*inout_max, term_max, &new_max)) {
    return false;
  }
  *inout_min = new_min;
  *inout_max = new_max;
  return true;
}

static loom_value_facts_t loom_symbolic_expr_intersect_integer_facts(
    loom_value_facts_t lhs, loom_value_facts_t rhs) {
  if (loom_value_facts_is_unknown(lhs) || loom_value_facts_is_float(lhs)) {
    return rhs;
  }
  if (loom_value_facts_is_unknown(rhs) || loom_value_facts_is_float(rhs)) {
    return lhs;
  }
  int64_t lower_bound = iree_max(lhs.range_lo, rhs.range_lo);
  int64_t upper_bound = iree_min(lhs.range_hi, rhs.range_hi);
  if (lower_bound > upper_bound) {
    return loom_value_facts_unknown();
  }
  return loom_value_facts_make(lower_bound, upper_bound, 1);
}

static iree_status_t loom_symbolic_expr_term_facts(
    loom_symbolic_expr_context_t* context, const loom_symbolic_term_t term,
    loom_value_facts_t* out_facts) {
  loom_value_facts_t facts = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_value_lookup_condition_refined_facts(
      context, term.value_id, &facts));
  if (term.relation_value_id == LOOM_VALUE_ID_INVALID ||
      term.relation_value_id == term.value_id) {
    *out_facts = facts;
    return iree_ok_status();
  }
  loom_value_facts_t relation_facts = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_value_lookup_condition_refined_facts(
      context, term.relation_value_id, &relation_facts));
  facts = loom_symbolic_expr_intersect_integer_facts(facts, relation_facts);
  IREE_RETURN_IF_ERROR(
      loom_symbolic_value_apply_identity_chain_predicates_to_facts(
          context, term.relation_value_id, &facts));
  *out_facts = facts;
  return iree_ok_status();
}

static void loom_symbolic_expr_mul_interval_bound(int64_t lhs, int64_t rhs,
                                                  int64_t* out_product) {
  if (iree_checked_mul_i64(lhs, rhs, out_product)) {
    return;
  }
  *out_product = (lhs < 0) != (rhs < 0) ? INT64_MIN : INT64_MAX;
}

static iree_status_t loom_symbolic_expr_term_interval(
    loom_symbolic_expr_context_t* context, const loom_symbolic_term_t term,
    int64_t* out_min, int64_t* out_max, bool* out_known) {
  *out_known = false;
  loom_value_facts_t facts = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_term_facts(context, term, &facts));
  if (loom_value_facts_is_unknown(facts) || loom_value_facts_is_float(facts)) {
    return iree_ok_status();
  }
  int64_t lower_product = 0;
  int64_t upper_product = 0;
  if (term.coefficient >= 0) {
    loom_symbolic_expr_mul_interval_bound(term.coefficient, facts.range_lo,
                                          &lower_product);
    loom_symbolic_expr_mul_interval_bound(term.coefficient, facts.range_hi,
                                          &upper_product);
  } else {
    loom_symbolic_expr_mul_interval_bound(term.coefficient, facts.range_hi,
                                          &lower_product);
    loom_symbolic_expr_mul_interval_bound(term.coefficient, facts.range_lo,
                                          &upper_product);
  }
  *out_min = lower_product;
  *out_max = upper_product;
  *out_known = true;
  return iree_ok_status();
}

typedef struct loom_symbolic_expr_term_interval_t {
  // Minimum value the scaled term can contribute.
  int64_t minimum;
  // Maximum value the scaled term can contribute.
  int64_t maximum;
  // True when minimum/maximum are known for this term.
  bool known;
} loom_symbolic_expr_term_interval_t;

static iree_status_t loom_symbolic_expr_build_term_intervals(
    loom_symbolic_expr_context_t* context, const loom_symbolic_term_t* terms,
    iree_host_size_t term_count,
    loom_symbolic_expr_term_interval_t* out_intervals) {
  for (iree_host_size_t i = 0; i < term_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_term_interval(
        context, terms[i], &out_intervals[i].minimum, &out_intervals[i].maximum,
        &out_intervals[i].known));
  }
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_normalize_difference_into_scratch(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression, int64_t* out_constant,
    iree_host_size_t* out_term_count, bool* out_linear) {
  *out_constant = 0;
  *out_term_count = 0;
  *out_linear = false;
  if (!loom_symbolic_expr_is_linear(left_expression) ||
      !loom_symbolic_expr_is_linear(right_expression)) {
    return iree_ok_status();
  }

  if (!iree_checked_sub_i64(left_expression->constant,
                            right_expression->constant, out_constant)) {
    return iree_ok_status();
  }

  if (context->projections.count != 0) {
    const loom_symbolic_projection_t* left_projection =
        loom_symbolic_expr_lookup_projection(context, left_expression);
    const loom_symbolic_projection_t* right_projection =
        loom_symbolic_expr_lookup_projection(context, right_expression);
    if (left_projection && right_projection &&
        loom_symbolic_projection_equal(left_projection, right_projection)) {
      *out_linear = true;
      return iree_ok_status();
    }
  }

  iree_host_size_t term_count = 0;
  if (!loom_symbolic_expr_proof_checked_term_count(left_expression->term_count,
                                                   right_expression->term_count,
                                                   &term_count)) {
    return iree_ok_status();
  }
  if (term_count > context->maximum_term_count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_proof_ensure_scratch_terms(context, term_count));
  iree_host_size_t term_ordinal = 0;
  for (iree_host_size_t i = 0; i < left_expression->term_count; ++i) {
    context->scratch_terms[term_ordinal++] = left_expression->terms[i];
  }
  for (iree_host_size_t i = 0; i < right_expression->term_count; ++i) {
    int64_t coefficient = right_expression->terms[i].coefficient;
    if (coefficient == INT64_MIN) {
      return iree_ok_status();
    }
    context->scratch_terms[term_ordinal++] = (loom_symbolic_term_t){
        .coefficient = -coefficient,
        .value_id = right_expression->terms[i].value_id,
        .relation_value_id = right_expression->terms[i].relation_value_id,
    };
  }
  if (term_ordinal > 1) {
    loom_symbolic_expr_proof_sort_terms(context->scratch_terms, term_ordinal);
  }

  iree_host_size_t write_index = 0;
  for (iree_host_size_t read_index = 0; read_index < term_ordinal;) {
    loom_value_id_t value_id = context->scratch_terms[read_index].value_id;
    loom_value_id_t relation_value_id =
        context->scratch_terms[read_index].relation_value_id;
    if (relation_value_id == LOOM_VALUE_ID_INVALID) {
      relation_value_id = value_id;
    }
    int64_t coefficient = 0;
    while (read_index < term_ordinal &&
           context->scratch_terms[read_index].value_id == value_id) {
      loom_value_id_t term_relation_value_id =
          context->scratch_terms[read_index].relation_value_id;
      if (term_relation_value_id == LOOM_VALUE_ID_INVALID) {
        term_relation_value_id = value_id;
      }
      if (term_relation_value_id != relation_value_id) {
        relation_value_id = value_id;
      }
      int64_t new_coefficient = 0;
      if (!iree_checked_add_i64(coefficient,
                                context->scratch_terms[read_index].coefficient,
                                &new_coefficient)) {
        return iree_ok_status();
      }
      coefficient = new_coefficient;
      ++read_index;
    }
    if (coefficient == 0) {
      continue;
    }
    context->scratch_terms[write_index++] = (loom_symbolic_term_t){
        .coefficient = coefficient,
        .value_id = value_id,
        .relation_value_id = relation_value_id,
    };
  }
  *out_term_count = write_index;
  *out_linear = true;
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_prove_le_linear(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_proof_result_t* out_result) {
  int64_t constant = 0;
  iree_host_size_t term_count = 0;
  bool linear = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_normalize_difference_into_scratch(
      context, left_expression, right_expression, &constant, &term_count,
      &linear));
  if (!linear) {
    *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    return iree_ok_status();
  }

  if (term_count == 0) {
    *out_result =
        constant <= 0 ? LOOM_SYMBOLIC_PROOF_TRUE : LOOM_SYMBOLIC_PROOF_FALSE;
    return iree_ok_status();
  }

  loom_symbolic_term_t stack_terms[LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT];
  const loom_symbolic_term_t* terms = NULL;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_proof_stabilize_scratch_terms(
      context, term_count, stack_terms, IREE_ARRAYSIZE(stack_terms), &terms));

  int64_t minimum = constant;
  int64_t maximum = constant;
  for (iree_host_size_t i = 0; i < term_count; ++i) {
    int64_t term_minimum = 0;
    int64_t term_maximum = 0;
    bool term_interval_known = false;
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_term_interval(
        context, terms[i], &term_minimum, &term_maximum, &term_interval_known));
    if (!term_interval_known ||
        !loom_symbolic_expr_accumulate_checked(term_minimum, term_maximum,
                                               &minimum, &maximum)) {
      *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
      return iree_ok_status();
    }
  }
  if (maximum <= 0) {
    *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
  } else if (minimum > 0) {
    *out_result = LOOM_SYMBOLIC_PROOF_FALSE;
  } else {
    *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Difference simplification
//===----------------------------------------------------------------------===//

iree_status_t loom_symbolic_expr_simplify_value_difference(
    loom_symbolic_expr_context_t* context, loom_value_id_t left_value,
    loom_value_id_t right_value,
    loom_symbolic_value_difference_t* out_difference) {
  *out_difference = (loom_symbolic_value_difference_t){
      .kind = LOOM_SYMBOLIC_VALUE_DIFFERENCE_UNKNOWN,
      .constant = 0,
      .value_id = LOOM_VALUE_ID_INVALID,
  };
  loom_symbolic_expr_t left_expression = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(context, left_value, &left_expression));
  loom_symbolic_expr_t right_expression = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(context, right_value, &right_expression));

  int64_t constant = 0;
  iree_host_size_t term_count = 0;
  bool linear = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_normalize_difference_into_scratch(
      context, &left_expression, &right_expression, &constant, &term_count,
      &linear));
  if (!linear) {
    return iree_ok_status();
  }

  if (term_count == 0) {
    *out_difference = (loom_symbolic_value_difference_t){
        .kind = LOOM_SYMBOLIC_VALUE_DIFFERENCE_CONSTANT,
        .constant = constant,
        .value_id = LOOM_VALUE_ID_INVALID,
    };
    return iree_ok_status();
  }
  if (constant == 0 && term_count == 1 &&
      context->scratch_terms[0].coefficient == 1) {
    *out_difference = (loom_symbolic_value_difference_t){
        .kind = LOOM_SYMBOLIC_VALUE_DIFFERENCE_VALUE,
        .constant = 0,
        .value_id = context->scratch_terms[0].relation_value_id,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_condition_operand_expression(
    loom_symbolic_expr_context_t* context,
    loom_condition_integer_operand_t operand,
    loom_symbolic_expr_t* out_expression) {
  if (operand.kind == LOOM_CONDITION_INTEGER_OPERAND_CONSTANT) {
    loom_symbolic_expr_constant(operand.constant, out_expression);
    return iree_ok_status();
  }
  return loom_symbolic_expr_from_value(context, operand.value_id,
                                       out_expression);
}

static bool loom_symbolic_expr_condition_relation_upper_bound(
    const loom_condition_integer_relation_t* relation,
    loom_condition_integer_operand_t* out_lower,
    loom_condition_integer_operand_t* out_upper, bool* out_strict) {
  switch (relation->relation) {
    case LOOM_SYMBOLIC_INTEGER_RELATION_EQ:
    case LOOM_SYMBOLIC_INTEGER_RELATION_LE:
      *out_lower = relation->left;
      *out_upper = relation->right;
      *out_strict = false;
      return true;
    case LOOM_SYMBOLIC_INTEGER_RELATION_LT:
      *out_lower = relation->left;
      *out_upper = relation->right;
      *out_strict = true;
      return true;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GE:
      *out_lower = relation->right;
      *out_upper = relation->left;
      *out_strict = false;
      return true;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GT:
      *out_lower = relation->right;
      *out_upper = relation->left;
      *out_strict = true;
      return true;
    default:
      return false;
  }
}

static loom_value_id_t loom_symbolic_expr_term_relation_value(
    const loom_symbolic_term_t* term) {
  return term->relation_value_id == LOOM_VALUE_ID_INVALID
             ? term->value_id
             : term->relation_value_id;
}

static bool loom_symbolic_expr_terms_are_exact_multiple(
    const loom_symbolic_term_t* expression_terms,
    iree_host_size_t expression_term_count,
    const loom_symbolic_term_t* relation_terms,
    iree_host_size_t relation_term_count, bool positive_multiplier,
    int64_t* out_multiplier) {
  *out_multiplier = 0;
  if (expression_term_count == 0 ||
      expression_term_count != relation_term_count ||
      relation_terms[0].coefficient == 0 ||
      (expression_terms[0].coefficient == INT64_MIN &&
       relation_terms[0].coefficient == -1)) {
    return false;
  }
  const int64_t multiplier =
      expression_terms[0].coefficient / relation_terms[0].coefficient;
  if (multiplier == 0 || (multiplier > 0) != positive_multiplier ||
      expression_terms[0].coefficient % relation_terms[0].coefficient != 0) {
    return false;
  }
  for (iree_host_size_t i = 0; i < expression_term_count; ++i) {
    int64_t scaled_coefficient = 0;
    if (expression_terms[i].value_id != relation_terms[i].value_id ||
        !iree_checked_mul_i64(relation_terms[i].coefficient, multiplier,
                              &scaled_coefficient) ||
        expression_terms[i].coefficient != scaled_coefficient) {
      return false;
    }
  }
  *out_multiplier = multiplier;
  return true;
}

static iree_status_t loom_symbolic_expr_terms_are_multiple(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_term_t* expression_terms,
    iree_host_size_t expression_term_count,
    const loom_symbolic_term_t* relation_terms,
    iree_host_size_t relation_term_count, bool positive_multiplier,
    bool* out_match, int64_t* out_multiplier) {
  *out_match = false;
  *out_multiplier = 0;
  if (loom_symbolic_expr_terms_are_exact_multiple(
          expression_terms, expression_term_count, relation_terms,
          relation_term_count, positive_multiplier, out_multiplier)) {
    *out_match = true;
    return iree_ok_status();
  }
  if (expression_term_count == 0 ||
      expression_term_count != relation_term_count ||
      expression_term_count > LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT) {
    return iree_ok_status();
  }

  bool matched_relation_terms[LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT] = {false};
  int64_t multiplier = 0;
  for (iree_host_size_t expression_index = 0;
       expression_index < expression_term_count; ++expression_index) {
    bool term_matched = false;
    for (iree_host_size_t relation_index = 0;
         relation_index < relation_term_count; ++relation_index) {
      if (matched_relation_terms[relation_index] ||
          relation_terms[relation_index].coefficient == 0) {
        continue;
      }

      int64_t candidate_multiplier = multiplier;
      if (candidate_multiplier == 0) {
        if (expression_terms[expression_index].coefficient == INT64_MIN &&
            relation_terms[relation_index].coefficient == -1) {
          continue;
        }
        candidate_multiplier = expression_terms[expression_index].coefficient /
                               relation_terms[relation_index].coefficient;
        if (candidate_multiplier == 0 ||
            (candidate_multiplier > 0) != positive_multiplier ||
            expression_terms[expression_index].coefficient %
                    relation_terms[relation_index].coefficient !=
                0) {
          continue;
        }
      }
      int64_t scaled_coefficient = 0;
      if (!iree_checked_mul_i64(relation_terms[relation_index].coefficient,
                                candidate_multiplier, &scaled_coefficient) ||
          expression_terms[expression_index].coefficient !=
              scaled_coefficient) {
        continue;
      }

      bool values_match = false;
      IREE_RETURN_IF_ERROR(loom_symbolic_values_semantically_match(
          context,
          loom_symbolic_expr_term_relation_value(
              &expression_terms[expression_index]),
          loom_symbolic_expr_term_relation_value(
              &relation_terms[relation_index]),
          &values_match));
      if (!values_match) {
        continue;
      }

      multiplier = candidate_multiplier;
      matched_relation_terms[relation_index] = true;
      term_matched = true;
      break;
    }
    if (!term_matched) {
      return iree_ok_status();
    }
  }

  *out_match = true;
  *out_multiplier = multiplier;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Active edge relations
//===----------------------------------------------------------------------===//

// Matches an expanded query against an active edge relation. For integral
// values, scale * lower + residual <= scale * upper follows from lower < upper
// when scale > 0 and residual <= scale, or from lower <= upper when scale > 0
// and residual <= 0. Equality permits either sign when residual <= 0.
typedef struct loom_symbolic_expr_condition_relation_proof_t {
  // Symbolic context used to expand relation operands.
  loom_symbolic_expr_context_t* context;

  // Normalized terms in the queried expression difference.
  const loom_symbolic_term_t* expression_terms;

  // Number of entries in expression_terms.
  iree_host_size_t expression_term_count;

  // Constant in the queried expression difference.
  int64_t expression_constant;

  // Terminal proof result.
  loom_symbolic_proof_result_t result;

  // Terminal expansion failure.
  iree_status_t status;
} loom_symbolic_expr_condition_relation_proof_t;

static bool loom_symbolic_expr_visit_condition_relation(
    void* user_data, const loom_condition_integer_relation_t* relation) {
  loom_symbolic_expr_condition_relation_proof_t* proof =
      (loom_symbolic_expr_condition_relation_proof_t*)user_data;
  loom_condition_integer_operand_t lower_operand = {0};
  loom_condition_integer_operand_t upper_operand = {0};
  bool strict = false;
  if (!loom_symbolic_expr_condition_relation_upper_bound(
          relation, &lower_operand, &upper_operand, &strict)) {
    return true;
  }

  loom_symbolic_expr_t lower_expression = {0};
  loom_symbolic_expr_t upper_expression = {0};
  proof->status = loom_symbolic_expr_condition_operand_expression(
      proof->context, lower_operand, &lower_expression);
  if (!iree_status_is_ok(proof->status)) {
    return false;
  }
  proof->status = loom_symbolic_expr_condition_operand_expression(
      proof->context, upper_operand, &upper_expression);
  if (!iree_status_is_ok(proof->status)) {
    return false;
  }
  int64_t relation_constant = 0;
  iree_host_size_t relation_term_count = 0;
  bool relation_linear = false;
  proof->status = loom_symbolic_expr_normalize_difference_into_scratch(
      proof->context, &lower_expression, &upper_expression, &relation_constant,
      &relation_term_count, &relation_linear);
  if (!iree_status_is_ok(proof->status)) {
    return false;
  }
  if (!relation_linear ||
      relation_term_count > LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT) {
    return true;
  }
  loom_symbolic_term_t relation_terms[LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT];
  memcpy(relation_terms, proof->context->scratch_terms,
         relation_term_count * sizeof(*relation_terms));

  bool terms_match = false;
  int64_t multiplier = 0;
  proof->status = loom_symbolic_expr_terms_are_multiple(
      proof->context, proof->expression_terms, proof->expression_term_count,
      relation_terms, relation_term_count, /*positive_multiplier=*/true,
      &terms_match, &multiplier);
  if (!iree_status_is_ok(proof->status)) {
    return false;
  }
  if (!terms_match && relation->relation == LOOM_SYMBOLIC_INTEGER_RELATION_EQ) {
    proof->status = loom_symbolic_expr_terms_are_multiple(
        proof->context, proof->expression_terms, proof->expression_term_count,
        relation_terms, relation_term_count,
        /*positive_multiplier=*/false, &terms_match, &multiplier);
    if (!iree_status_is_ok(proof->status)) {
      return false;
    }
  }
  if (!terms_match) {
    return true;
  }
  int64_t scaled_relation_constant = 0;
  int64_t residual_constant = 0;
  if (!iree_checked_mul_i64(relation_constant, multiplier,
                            &scaled_relation_constant) ||
      !iree_checked_sub_i64(proof->expression_constant,
                            scaled_relation_constant, &residual_constant)) {
    return true;
  }
  if (residual_constant <= (strict ? multiplier : 0)) {
    proof->result = LOOM_SYMBOLIC_PROOF_TRUE;
    return false;
  }
  return true;
}

static iree_status_t loom_symbolic_expr_prove_le_by_condition_relations(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_proof_result_t* out_result) {
  *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  if (!loom_condition_fact_scope_has_integer_relations(
          context->condition_scope)) {
    return iree_ok_status();
  }

  int64_t expression_constant = 0;
  iree_host_size_t expression_term_count = 0;
  bool expression_linear = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_normalize_difference_into_scratch(
      context, left_expression, right_expression, &expression_constant,
      &expression_term_count, &expression_linear));
  if (!expression_linear || expression_term_count == 0 ||
      expression_term_count > LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT) {
    return iree_ok_status();
  }
  loom_symbolic_term_t expression_terms[LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT];
  memcpy(expression_terms, context->scratch_terms,
         expression_term_count * sizeof(*expression_terms));

  loom_condition_integer_operand_t
      anchors[LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT * 2 + 2];
  iree_host_size_t anchor_count = 0;
  for (iree_host_size_t i = 0; i < expression_term_count; ++i) {
    const loom_value_id_t candidates[] = {
        expression_terms[i].value_id,
        expression_terms[i].relation_value_id,
    };
    for (iree_host_size_t candidate_index = 0;
         candidate_index < IREE_ARRAYSIZE(candidates); ++candidate_index) {
      bool duplicate = false;
      for (iree_host_size_t anchor_index = 0; anchor_index < anchor_count;
           ++anchor_index) {
        duplicate |=
            anchors[anchor_index].value_id == candidates[candidate_index];
      }
      if (!duplicate) {
        anchors[anchor_count++] = (loom_condition_integer_operand_t){
            .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
            .value_id = candidates[candidate_index],
        };
      }
    }
  }
  // Retained CFG relations are indexed by either operand, so exact proof
  // endpoints participate as anchors alongside symbolic term values.
  const bool left_is_constant = loom_symbolic_expr_is_constant(left_expression);
  if (left_is_constant) {
    anchors[anchor_count++] = (loom_condition_integer_operand_t){
        .kind = LOOM_CONDITION_INTEGER_OPERAND_CONSTANT,
        .value_id = LOOM_VALUE_ID_INVALID,
        .constant = left_expression->constant,
    };
  }
  if (loom_symbolic_expr_is_constant(right_expression) &&
      (!left_is_constant ||
       left_expression->constant != right_expression->constant)) {
    anchors[anchor_count++] = (loom_condition_integer_operand_t){
        .kind = LOOM_CONDITION_INTEGER_OPERAND_CONSTANT,
        .value_id = LOOM_VALUE_ID_INVALID,
        .constant = right_expression->constant,
    };
  }

  loom_symbolic_expr_condition_relation_proof_t proof = {
      .context = context,
      .expression_terms = expression_terms,
      .expression_term_count = expression_term_count,
      .expression_constant = expression_constant,
      .result = LOOM_SYMBOLIC_PROOF_UNKNOWN,
      .status = iree_ok_status(),
  };
  (void)loom_condition_fact_scope_for_each_anchored_while(
      context->condition_scope, context->fact_table, anchors, anchor_count,
      loom_symbolic_expr_visit_condition_relation, &proof);
  if (iree_status_is_ok(proof.status)) {
    *out_result = proof.result;
  }
  return proof.status;
}

static bool loom_symbolic_expr_scale_linear_view(
    const loom_symbolic_expr_t* expression, int64_t multiplier,
    loom_symbolic_term_t* term_storage, iree_host_size_t term_storage_capacity,
    loom_symbolic_expr_t* out_expression) {
  if (!loom_symbolic_expr_is_linear(expression) || multiplier <= 0 ||
      expression->term_count > term_storage_capacity) {
    return false;
  }

  int64_t constant = 0;
  if (!iree_checked_mul_i64(expression->constant, multiplier, &constant)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < expression->term_count; ++i) {
    int64_t coefficient = 0;
    if (!iree_checked_mul_i64(expression->terms[i].coefficient, multiplier,
                              &coefficient)) {
      return false;
    }
    term_storage[i] = expression->terms[i];
    term_storage[i].coefficient = coefficient;
  }

  loom_value_facts_t multiplier_facts = loom_value_facts_exact_i64(multiplier);
  loom_value_facts_t facts = loom_value_facts_unknown();
  loom_value_facts_muli(&expression->facts, &multiplier_facts, &facts);
  *out_expression = (loom_symbolic_expr_t){
      .constant = constant,
      .terms = expression->term_count == 0 ? NULL : term_storage,
      .term_count = expression->term_count,
      .facts = facts,
      .flags = LOOM_SYMBOLIC_EXPR_FLAG_LINEAR,
  };
  return true;
}

static void loom_symbolic_expr_residual_excluding_pair(
    int64_t constant, const loom_symbolic_term_t* terms,
    iree_host_size_t term_count, iree_host_size_t first_index,
    iree_host_size_t second_index, loom_symbolic_term_t* term_storage,
    loom_symbolic_expr_t* out_expression) {
  iree_host_size_t residual_count = 0;
  for (iree_host_size_t i = 0; i < term_count; ++i) {
    if (i == first_index || i == second_index) {
      continue;
    }
    term_storage[residual_count++] = terms[i];
  }
  *out_expression = (loom_symbolic_expr_t){
      .constant = constant,
      .terms = residual_count == 0 ? NULL : term_storage,
      .term_count = residual_count,
      .facts = loom_value_facts_unknown(),
      .flags = LOOM_SYMBOLIC_EXPR_FLAG_LINEAR,
  };
}

static iree_status_t loom_symbolic_expr_try_common_product_factor(
    loom_symbolic_expr_context_t* context, loom_value_id_t common_factor,
    loom_value_id_t positive_relation_value,
    loom_value_id_t negative_relation_value, int64_t coefficient,
    const loom_symbolic_expr_t* residual,
    loom_symbolic_proof_result_t* out_result) {
  bool common_factor_non_negative = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_value_is_non_negative(
      context, common_factor, &common_factor_non_negative));
  if (!common_factor_non_negative) {
    return iree_ok_status();
  }

  loom_symbolic_proof_result_t relation = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_symbolic_value_prove_relation(
      context, LOOM_SYMBOLIC_INTEGER_RELATION_LT, positive_relation_value,
      negative_relation_value, &relation));
  if (relation == LOOM_SYMBOLIC_PROOF_TRUE) {
    loom_symbolic_expr_t factor = {0};
    IREE_RETURN_IF_ERROR(
        loom_symbolic_expr_from_value(context, common_factor, &factor));
    loom_symbolic_term_t
        scaled_factor_terms[LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT];
    loom_symbolic_expr_t scaled_factor = {0};
    if (loom_symbolic_expr_scale_linear_view(
            &factor, coefficient, scaled_factor_terms,
            IREE_ARRAYSIZE(scaled_factor_terms), &scaled_factor)) {
      loom_symbolic_proof_result_t residual_proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le(
          context, residual, &scaled_factor, &residual_proof));
      if (residual_proof == LOOM_SYMBOLIC_PROOF_TRUE) {
        *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
        return iree_ok_status();
      }
    }
  }

  relation = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_symbolic_value_prove_relation(
      context, LOOM_SYMBOLIC_INTEGER_RELATION_LE, positive_relation_value,
      negative_relation_value, &relation));
  if (relation == LOOM_SYMBOLIC_PROOF_TRUE) {
    loom_symbolic_expr_t zero = {0};
    loom_symbolic_expr_constant(0, &zero);
    loom_symbolic_proof_result_t residual_proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    IREE_RETURN_IF_ERROR(
        loom_symbolic_expr_prove_le(context, residual, &zero, &residual_proof));
    if (residual_proof == LOOM_SYMBOLIC_PROOF_TRUE) {
      *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Product and scaled relations
//===----------------------------------------------------------------------===//

// Proves row-major forms such as row * stride + residual <= rows * stride.
// Dynamic products remain opaque symbolic terms, so identify a shared
// non-negative factor and discharge the row and residual bounds separately.
static iree_status_t loom_symbolic_expr_prove_le_by_common_product_factor(
    loom_symbolic_expr_context_t* context, int64_t constant,
    const loom_symbolic_term_t* terms, iree_host_size_t term_count,
    loom_symbolic_proof_result_t* out_result) {
  *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  if (term_count < 2 || term_count > LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < term_count; ++i) {
    for (iree_host_size_t j = i + 1; j < term_count; ++j) {
      const loom_symbolic_term_t* positive_term = &terms[i];
      const loom_symbolic_term_t* negative_term = &terms[j];
      if (positive_term->coefficient < 0) {
        positive_term = &terms[j];
        negative_term = &terms[i];
      }
      if (positive_term->coefficient <= 0 || negative_term->coefficient >= 0 ||
          negative_term->coefficient == INT64_MIN ||
          positive_term->coefficient != -negative_term->coefficient) {
        continue;
      }

      loom_value_id_t positive_factors[2] = {LOOM_VALUE_ID_INVALID,
                                             LOOM_VALUE_ID_INVALID};
      loom_value_id_t negative_factors[2] = {LOOM_VALUE_ID_INVALID,
                                             LOOM_VALUE_ID_INVALID};
      if (!loom_symbolic_value_product_factors(context, positive_term->value_id,
                                               &positive_factors[0],
                                               &positive_factors[1]) ||
          !loom_symbolic_value_product_factors(context, negative_term->value_id,
                                               &negative_factors[0],
                                               &negative_factors[1])) {
        continue;
      }

      loom_symbolic_term_t
          residual_terms[LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT];
      loom_symbolic_expr_t residual = {0};
      loom_symbolic_expr_residual_excluding_pair(constant, terms, term_count, i,
                                                 j, residual_terms, &residual);
      for (uint8_t positive_factor_index = 0; positive_factor_index < 2;
           ++positive_factor_index) {
        for (uint8_t negative_factor_index = 0; negative_factor_index < 2;
             ++negative_factor_index) {
          bool factors_match = false;
          IREE_RETURN_IF_ERROR(loom_symbolic_values_match(
              context, positive_factors[positive_factor_index],
              negative_factors[negative_factor_index], &factors_match));
          if (!factors_match) {
            continue;
          }

          IREE_RETURN_IF_ERROR(loom_symbolic_expr_try_common_product_factor(
              context, positive_factors[positive_factor_index],
              positive_factors[1 - positive_factor_index],
              negative_factors[1 - negative_factor_index],
              positive_term->coefficient, &residual, out_result));
          if (*out_result != LOOM_SYMBOLIC_PROOF_UNKNOWN) {
            return iree_ok_status();
          }
        }
      }
    }
  }
  return iree_ok_status();
}

static bool loom_symbolic_expr_scaled_pair_terms(
    const loom_symbolic_term_t* terms, iree_host_size_t term_count,
    int64_t* out_scale, loom_value_id_t* out_positive_relation_value,
    loom_value_id_t* out_negative_relation_value) {
  if (term_count != 2) {
    return false;
  }
  const loom_symbolic_term_t* positive_term = NULL;
  const loom_symbolic_term_t* negative_term = NULL;
  if (terms[0].coefficient > 0 && terms[1].coefficient < 0) {
    positive_term = &terms[0];
    negative_term = &terms[1];
  } else if (terms[1].coefficient > 0 && terms[0].coefficient < 0) {
    positive_term = &terms[1];
    negative_term = &terms[0];
  } else {
    return false;
  }
  if (negative_term->coefficient == INT64_MIN ||
      positive_term->coefficient != -negative_term->coefficient) {
    return false;
  }
  *out_scale = positive_term->coefficient;
  *out_positive_relation_value = positive_term->relation_value_id;
  if (*out_positive_relation_value == LOOM_VALUE_ID_INVALID) {
    *out_positive_relation_value = positive_term->value_id;
  }
  *out_negative_relation_value = negative_term->relation_value_id;
  if (*out_negative_relation_value == LOOM_VALUE_ID_INVALID) {
    *out_negative_relation_value = negative_term->value_id;
  }
  return true;
}

static bool loom_symbolic_expr_scaled_pair_term_indices(
    const loom_symbolic_term_t* terms, iree_host_size_t positive_index,
    iree_host_size_t negative_index, int64_t* out_scale,
    loom_value_id_t* out_positive_relation_value,
    loom_value_id_t* out_negative_relation_value) {
  const loom_symbolic_term_t* positive_term = &terms[positive_index];
  const loom_symbolic_term_t* negative_term = &terms[negative_index];
  if (positive_term->coefficient <= 0 || negative_term->coefficient >= 0 ||
      negative_term->coefficient == INT64_MIN ||
      positive_term->coefficient != -negative_term->coefficient) {
    return false;
  }
  *out_scale = positive_term->coefficient;
  *out_positive_relation_value = positive_term->relation_value_id;
  if (*out_positive_relation_value == LOOM_VALUE_ID_INVALID) {
    *out_positive_relation_value = positive_term->value_id;
  }
  *out_negative_relation_value = negative_term->relation_value_id;
  if (*out_negative_relation_value == LOOM_VALUE_ID_INVALID) {
    *out_negative_relation_value = negative_term->value_id;
  }
  return true;
}

static bool loom_symbolic_expr_scaled_pair_from_indices(
    const loom_symbolic_term_t* terms, iree_host_size_t first_index,
    iree_host_size_t second_index, int64_t* out_scale,
    loom_value_id_t* out_positive_relation_value,
    loom_value_id_t* out_negative_relation_value) {
  if (terms[first_index].coefficient > 0) {
    return loom_symbolic_expr_scaled_pair_term_indices(
        terms, first_index, second_index, out_scale,
        out_positive_relation_value, out_negative_relation_value);
  }
  return loom_symbolic_expr_scaled_pair_term_indices(
      terms, second_index, first_index, out_scale, out_positive_relation_value,
      out_negative_relation_value);
}

static void loom_symbolic_expr_residual_interval_excluding_pair(
    const loom_symbolic_expr_term_interval_t* intervals,
    iree_host_size_t term_count, iree_host_size_t first_index,
    iree_host_size_t second_index, int64_t constant, int64_t* out_maximum,
    bool* out_known) {
  *out_known = false;
  int64_t minimum = constant;
  int64_t maximum = constant;
  for (iree_host_size_t i = 0; i < term_count; ++i) {
    if (i == first_index || i == second_index) {
      continue;
    }
    const loom_symbolic_expr_term_interval_t* interval = &intervals[i];
    if (!interval->known ||
        !loom_symbolic_expr_accumulate_checked(
            interval->minimum, interval->maximum, &minimum, &maximum)) {
      return;
    }
  }
  *out_maximum = maximum;
  *out_known = true;
}

static iree_status_t
loom_symbolic_expr_prove_le_by_scaled_relation_with_residual(
    loom_symbolic_expr_context_t* context, int64_t constant,
    const loom_symbolic_term_t* normalized_terms, iree_host_size_t term_count,
    loom_symbolic_proof_result_t* out_result) {
  *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  if (term_count < 2 || term_count > LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT) {
    return iree_ok_status();
  }

  loom_symbolic_expr_term_interval_t
      intervals[LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT];
  bool intervals_built = false;

  for (iree_host_size_t i = 0; i < term_count; ++i) {
    for (iree_host_size_t j = i + 1; j < term_count; ++j) {
      int64_t scale = 0;
      loom_value_id_t positive_relation_value = LOOM_VALUE_ID_INVALID;
      loom_value_id_t negative_relation_value = LOOM_VALUE_ID_INVALID;
      if (!loom_symbolic_expr_scaled_pair_from_indices(
              normalized_terms, i, j, &scale, &positive_relation_value,
              &negative_relation_value)) {
        continue;
      }
      if (!intervals_built) {
        IREE_RETURN_IF_ERROR(loom_symbolic_expr_build_term_intervals(
            context, normalized_terms, term_count, intervals));
        intervals_built = true;
      }

      int64_t residual_maximum = 0;
      bool residual_interval_known = false;
      loom_symbolic_expr_residual_interval_excluding_pair(
          intervals, term_count, i, j, constant, &residual_maximum,
          &residual_interval_known);
      if (!residual_interval_known) {
        continue;
      }

      loom_symbolic_proof_result_t strict_relation =
          LOOM_SYMBOLIC_PROOF_UNKNOWN;
      IREE_RETURN_IF_ERROR(loom_symbolic_value_prove_relation(
          context, LOOM_SYMBOLIC_INTEGER_RELATION_LT, positive_relation_value,
          negative_relation_value, &strict_relation));
      if (strict_relation == LOOM_SYMBOLIC_PROOF_TRUE &&
          residual_maximum <= scale) {
        *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
        return iree_ok_status();
      }

      loom_symbolic_proof_result_t nonstrict_relation =
          LOOM_SYMBOLIC_PROOF_UNKNOWN;
      IREE_RETURN_IF_ERROR(loom_symbolic_value_prove_relation(
          context, LOOM_SYMBOLIC_INTEGER_RELATION_LE, positive_relation_value,
          negative_relation_value, &nonstrict_relation));
      if (nonstrict_relation == LOOM_SYMBOLIC_PROOF_TRUE &&
          residual_maximum <= 0) {
        *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_prove_le_by_scaled_relation(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_expr_proof_scope_t proof_scope,
    loom_symbolic_proof_result_t* out_result) {
  *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  int64_t constant = 0;
  iree_host_size_t term_count = 0;
  bool linear = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_normalize_difference_into_scratch(
      context, left_expression, right_expression, &constant, &term_count,
      &linear));
  if (!linear) {
    return iree_ok_status();
  }

  loom_symbolic_term_t stable_terms[LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT];
  const loom_symbolic_term_t* terms = NULL;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_proof_stabilize_scratch_terms(
      context, term_count, stable_terms, IREE_ARRAYSIZE(stable_terms), &terms));

  if (term_count == 1 && terms[0].coefficient > 0) {
    loom_value_id_t positive_relation_value = terms[0].relation_value_id;
    if (positive_relation_value == LOOM_VALUE_ID_INVALID) {
      positive_relation_value = terms[0].value_id;
    }
    bool scaled_static_assumption_matched = false;
    IREE_RETURN_IF_ERROR(loom_symbolic_value_prove_scaled_static_assumption(
        context, positive_relation_value, positive_relation_value,
        terms[0].coefficient, constant, &scaled_static_assumption_matched,
        out_result));
    if (scaled_static_assumption_matched) {
      return iree_ok_status();
    }
  }

  int64_t scale = 0;
  loom_value_id_t positive_relation_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t negative_relation_value = LOOM_VALUE_ID_INVALID;
  if (!loom_symbolic_expr_scaled_pair_terms(terms, term_count, &scale,
                                            &positive_relation_value,
                                            &negative_relation_value)) {
    IREE_RETURN_IF_ERROR(
        loom_symbolic_expr_prove_le_by_scaled_relation_with_residual(
            context, constant, terms, term_count, out_result));
    if (*out_result == LOOM_SYMBOLIC_PROOF_UNKNOWN &&
        proof_scope != LOOM_SYMBOLIC_EXPR_PROOF_SCOPE_ACTIVE_FACTS) {
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le_by_common_product_factor(
          context, constant, terms, term_count, out_result));
    }
    return iree_ok_status();
  }

  loom_symbolic_proof_result_t strict_relation = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_symbolic_value_prove_relation(
      context, LOOM_SYMBOLIC_INTEGER_RELATION_LT, positive_relation_value,
      negative_relation_value, &strict_relation));
  if (strict_relation == LOOM_SYMBOLIC_PROOF_TRUE && constant <= scale) {
    *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
    return iree_ok_status();
  }
  if (strict_relation == LOOM_SYMBOLIC_PROOF_FALSE && constant > 0) {
    *out_result = LOOM_SYMBOLIC_PROOF_FALSE;
    return iree_ok_status();
  }

  bool scaled_assumption_matched = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_value_prove_scaled_assumption(
      context, positive_relation_value, positive_relation_value,
      negative_relation_value, scale, constant, &scaled_assumption_matched,
      out_result));
  if (scaled_assumption_matched) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_symbolic_value_prove_scaled_assumption(
      context, negative_relation_value, positive_relation_value,
      negative_relation_value, scale, constant, &scaled_assumption_matched,
      out_result));
  if (scaled_assumption_matched) {
    return iree_ok_status();
  }

  loom_symbolic_proof_result_t nonstrict_relation = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_symbolic_value_prove_relation(
      context, LOOM_SYMBOLIC_INTEGER_RELATION_LE, positive_relation_value,
      negative_relation_value, &nonstrict_relation));
  if (nonstrict_relation == LOOM_SYMBOLIC_PROOF_TRUE && constant <= 0) {
    *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
    return iree_ok_status();
  }
  int64_t false_lower_bound = 0;
  if (nonstrict_relation == LOOM_SYMBOLIC_PROOF_FALSE &&
      iree_checked_add_i64(scale, constant, &false_lower_bound) &&
      false_lower_bound > 0) {
    *out_result = LOOM_SYMBOLIC_PROOF_FALSE;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_prove_le_by_scaled_relation_with_residual(
          context, constant, terms, term_count, out_result));
  if (*out_result == LOOM_SYMBOLIC_PROOF_UNKNOWN &&
      proof_scope != LOOM_SYMBOLIC_EXPR_PROOF_SCOPE_ACTIVE_FACTS) {
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le_by_common_product_factor(
        context, constant, terms, term_count, out_result));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Select case proofs
//===----------------------------------------------------------------------===//

static iree_status_t loom_symbolic_expr_prove_le_by_select_cases(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_proof_result_t* out_result);

static iree_status_t loom_symbolic_expr_prove_le_with_scope(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_expr_proof_scope_t proof_scope,
    loom_symbolic_proof_result_t* out_result) {
  *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  if (loom_symbolic_expr_is_linear(left_expression) &&
      loom_symbolic_expr_is_linear(right_expression)) {
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le_linear(
        context, left_expression, right_expression, out_result));
    if (*out_result != LOOM_SYMBOLIC_PROOF_UNKNOWN) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le_by_condition_relations(
        context, left_expression, right_expression, out_result));
    if (*out_result != LOOM_SYMBOLIC_PROOF_UNKNOWN) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le_by_scaled_relation(
        context, left_expression, right_expression, proof_scope, out_result));
    if (*out_result != LOOM_SYMBOLIC_PROOF_UNKNOWN) {
      return iree_ok_status();
    }
    if (proof_scope == LOOM_SYMBOLIC_EXPR_PROOF_SCOPE_SELECT_CASES) {
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le_by_select_cases(
          context, left_expression, right_expression, out_result));
      if (*out_result != LOOM_SYMBOLIC_PROOF_UNKNOWN) {
        return iree_ok_status();
      }
    }
  }
  // Expanded expressions may still carry stronger facts from their defining
  // SSA value, such as index.assume range facts on a value that algebraically
  // expands back to its unconstrained source.
  *out_result =
      loom_symbolic_expr_prove_le_by_facts(left_expression, right_expression);
  return iree_ok_status();
}

typedef struct loom_symbolic_expr_select_cursor_entry_t {
  // Canonical root used to deduplicate identical dependency sets.
  loom_value_set_id_t root;
  // Next condition available from cursor.
  loom_value_id_t next_condition;
  // Increasing-order cursor over one canonical dependency set.
  loom_value_set_cursor_t cursor;
} loom_symbolic_expr_select_cursor_entry_t;

typedef struct loom_symbolic_expr_select_iterator_t {
  // Active cursor entries, initially backed by inline_entries.
  loom_symbolic_expr_select_cursor_entry_t* entries;
  // Number of distinct nonempty dependency roots.
  iree_host_size_t count;
  // Allocated entry capacity.
  iree_host_size_t capacity;
  // Inline storage for common one-to-four-root expressions.
  loom_symbolic_expr_select_cursor_entry_t
      inline_entries[LOOM_SYMBOLIC_EXPR_SELECT_CURSOR_INLINE_CAPACITY];
} loom_symbolic_expr_select_iterator_t;

typedef struct loom_symbolic_expr_select_assumptions_t {
  // Collected outcomes, initially backed by inline_values.
  loom_condition_assumption_t* values;
  // Number of collected outcomes.
  iree_host_size_t count;
  // Allocated outcome capacity.
  iree_host_size_t capacity;
  // Inline storage for the common small conjunction.
  loom_condition_assumption_t
      inline_values[LOOM_SYMBOLIC_EXPR_SELECT_ASSUMPTION_INLINE_CAPACITY];
} loom_symbolic_expr_select_assumptions_t;

static void loom_symbolic_expr_select_iterator_initialize(
    loom_symbolic_expr_select_iterator_t* out_iterator) {
  *out_iterator = (loom_symbolic_expr_select_iterator_t){0};
  out_iterator->entries = out_iterator->inline_entries;
  out_iterator->capacity = IREE_ARRAYSIZE(out_iterator->inline_entries);
}

static void loom_symbolic_expr_select_assumptions_initialize(
    loom_symbolic_expr_select_assumptions_t* out_assumptions) {
  *out_assumptions = (loom_symbolic_expr_select_assumptions_t){0};
  out_assumptions->values = out_assumptions->inline_values;
  out_assumptions->capacity = IREE_ARRAYSIZE(out_assumptions->inline_values);
}

#if IREE_HAVE_ATTRIBUTE(minsize)
__attribute__((minsize))
#endif  // IREE_HAVE_ATTRIBUTE(minsize)
IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_symbolic_expr_select_assumptions_append(
    loom_value_id_t condition, bool truth, iree_arena_allocator_t* arena,
    loom_symbolic_expr_select_assumptions_t* assumptions) {
  if (assumptions->count == assumptions->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, assumptions->count, assumptions->count + 1,
        sizeof(*assumptions->values), &assumptions->capacity,
        (void**)&assumptions->values));
  }
  assumptions->values[assumptions->count++] = (loom_condition_assumption_t){
      .condition = condition,
      .assumed_truth = truth,
  };
  return iree_ok_status();
}

#if IREE_HAVE_ATTRIBUTE(minsize)
__attribute__((minsize))
#endif  // IREE_HAVE_ATTRIBUTE(minsize)
IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_symbolic_expr_select_iterator_add(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    iree_host_size_t maximum_root_count, iree_arena_allocator_t* arena,
    loom_symbolic_expr_select_iterator_t* iterator) {
  loom_value_set_cursor_t cursor;
  const loom_value_set_id_t root =
      loom_value_fact_table_select_dependencies_begin(fact_table, value_id,
                                                      &cursor);
  if (!root) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < iterator->count; ++i) {
    if (iterator->entries[i].root == root) {
      return iree_ok_status();
    }
  }
  if (iterator->count == iterator->capacity) {
    loom_symbolic_expr_select_cursor_entry_t* entries = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, maximum_root_count, sizeof(*entries), (void**)&entries));
    memcpy(entries, iterator->entries, iterator->count * sizeof(*entries));
    iterator->entries = entries;
    iterator->capacity = maximum_root_count;
  }
  loom_symbolic_expr_select_cursor_entry_t* entry =
      &iterator->entries[iterator->count++];
  *entry = (loom_symbolic_expr_select_cursor_entry_t){
      .root = root,
      .next_condition = loom_value_set_cursor_next(&cursor),
      .cursor = cursor,
  };
  return iree_ok_status();
}

#if IREE_HAVE_ATTRIBUTE(minsize)
__attribute__((minsize))
#endif  // IREE_HAVE_ATTRIBUTE(minsize)
IREE_ATTRIBUTE_NOINLINE static loom_value_id_t
loom_symbolic_expr_select_iterator_next(
    loom_symbolic_expr_select_iterator_t* iterator) {
  loom_value_id_t next_condition = LOOM_VALUE_ID_INVALID;
  for (iree_host_size_t i = 0; i < iterator->count; ++i) {
    next_condition =
        iree_min(next_condition, iterator->entries[i].next_condition);
  }
  if (next_condition == LOOM_VALUE_ID_INVALID) {
    return LOOM_VALUE_ID_INVALID;
  }
  for (iree_host_size_t i = 0; i < iterator->count; ++i) {
    loom_symbolic_expr_select_cursor_entry_t* entry = &iterator->entries[i];
    if (entry->next_condition == next_condition) {
      entry->next_condition = loom_value_set_cursor_next(&entry->cursor);
    }
  }
  return next_condition;
}

static iree_status_t loom_symbolic_expr_select_iterator_for_le(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression, iree_arena_allocator_t* arena,
    loom_symbolic_expr_select_iterator_t* out_iterator) {
  loom_symbolic_expr_select_iterator_initialize(out_iterator);

  int64_t constant = 0;
  iree_host_size_t term_count = 0;
  bool linear = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_normalize_difference_into_scratch(
      context, left_expression, right_expression, &constant, &term_count,
      &linear));
  (void)constant;
  if (!linear) {
    return iree_ok_status();
  }

  const iree_host_size_t maximum_root_count = term_count * 2;
  for (iree_host_size_t i = 0; i < term_count; ++i) {
    const loom_symbolic_term_t term = context->scratch_terms[i];
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_select_iterator_add(
        context->fact_table, term.value_id, maximum_root_count, arena,
        out_iterator));
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_select_iterator_add(
        context->fact_table, term.relation_value_id, maximum_root_count, arena,
        out_iterator));
  }
  return iree_ok_status();
}

#if IREE_HAVE_ATTRIBUTE(minsize)
__attribute__((minsize))
#endif  // IREE_HAVE_ATTRIBUTE(minsize)
IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_symbolic_expr_prove_le_with_condition_assumption(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression, loom_value_id_t condition,
    bool assumed_truth, loom_symbolic_proof_result_t* out_result) {
  *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  iree_arena_allocator_t derivation_arena;
  iree_arena_initialize(context->arena->block_pool, &derivation_arena);
  loom_condition_derivation_t derivation;
  loom_condition_derivation_initialize(&derivation_arena, &derivation);
  iree_status_t status = loom_condition_facts_query_complete(
      &context->condition_query, context->fact_table, condition, assumed_truth,
      &derivation);
  if (iree_status_is_ok(status) &&
      (derivation.integer_facts.integer_relation_count != 0 ||
       derivation.boolean_fact_count != 0)) {
    const loom_condition_fact_scope_t* previous_scope =
        context->condition_scope;
    loom_condition_fact_scope_t condition_scope;
    loom_condition_fact_scope_initialize_local(previous_scope, &derivation,
                                               &condition_scope);
    context->condition_scope = &condition_scope;
    loom_symbolic_expr_context_reset(context);
    status = loom_symbolic_expr_prove_le_with_scope(
        context, left_expression, right_expression,
        LOOM_SYMBOLIC_EXPR_PROOF_SCOPE_CONDITION_ASSUMPTION, out_result);
    context->condition_scope = previous_scope;
    loom_symbolic_expr_context_reset(context);
  }
  iree_arena_deinitialize(&derivation_arena);
  return status;
}

static iree_status_t loom_symbolic_expr_prove_le_by_select_cases(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_proof_result_t* out_result) {
  *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  iree_arena_allocator_t condition_arena;
  iree_arena_initialize(context->arena->block_pool, &condition_arena);
  loom_symbolic_expr_select_iterator_t conditions;
  iree_status_t status = loom_symbolic_expr_select_iterator_for_le(
      context, left_expression, right_expression, &condition_arena,
      &conditions);
  loom_symbolic_expr_select_assumptions_t residual_assumptions;
  loom_symbolic_expr_select_assumptions_initialize(&residual_assumptions);
  // A one-sided proof covers that condition outcome. The only assignments not
  // covered by all such proofs satisfy the conjunction of their opposite
  // outcomes, which is checked once below. Every retained condition is
  // considered, so additional dependencies cannot displace an existing case
  // and case exploration remains one pass over the dependency set.
  while (iree_status_is_ok(status)) {
    const loom_value_id_t condition =
        loom_symbolic_expr_select_iterator_next(&conditions);
    if (condition == LOOM_VALUE_ID_INVALID) {
      break;
    }

    bool proven_condition = false;
    bool condition_is_proven = false;
    if (context->condition_scope) {
      status = loom_condition_fact_scope_proves_condition(
          &context->condition_query, context->fact_table,
          context->condition_scope, condition, &proven_condition,
          &condition_is_proven);
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
    if (condition_is_proven) {
      status = loom_symbolic_expr_prove_le_with_condition_assumption(
          context, left_expression, right_expression, condition,
          proven_condition, out_result);
      if (!iree_status_is_ok(status) ||
          *out_result == LOOM_SYMBOLIC_PROOF_TRUE) {
        break;
      }
      *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
      status = loom_symbolic_expr_select_assumptions_append(
          condition, proven_condition, &condition_arena, &residual_assumptions);
      continue;
    }

    loom_symbolic_proof_result_t true_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    status = loom_symbolic_expr_prove_le_with_condition_assumption(
        context, left_expression, right_expression, condition,
        /*assumed_truth=*/true, &true_result);
    if (!iree_status_is_ok(status)) {
      break;
    }
    loom_symbolic_proof_result_t false_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    status = loom_symbolic_expr_prove_le_with_condition_assumption(
        context, left_expression, right_expression, condition,
        /*assumed_truth=*/false, &false_result);
    if (!iree_status_is_ok(status)) {
      break;
    }
    const bool true_proven = true_result == LOOM_SYMBOLIC_PROOF_TRUE;
    const bool false_proven = false_result == LOOM_SYMBOLIC_PROOF_TRUE;
    if (true_proven && false_proven) {
      *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
      break;
    }
    if (true_proven != false_proven) {
      status = loom_symbolic_expr_select_assumptions_append(
          condition, /*truth=*/!true_proven, &condition_arena,
          &residual_assumptions);
    }
  }

  const loom_condition_fact_scope_t* previous_scope = context->condition_scope;
  loom_condition_derivation_t residual_derivation;
  loom_condition_fact_scope_t residual_scope;
  if (iree_status_is_ok(status) && *out_result == LOOM_SYMBOLIC_PROOF_UNKNOWN &&
      residual_assumptions.count != 0) {
    loom_condition_derivation_initialize(&condition_arena,
                                         &residual_derivation);
    status = loom_condition_facts_query_conjunction_complete(
        &context->condition_query, context->fact_table,
        residual_assumptions.values, residual_assumptions.count,
        &residual_derivation);
    if (iree_status_is_ok(status)) {
      loom_condition_fact_scope_initialize_local(
          previous_scope, &residual_derivation, &residual_scope);
      context->condition_scope = &residual_scope;
      loom_symbolic_expr_context_reset(context);
      status = loom_symbolic_expr_prove_le_with_scope(
          context, left_expression, right_expression,
          LOOM_SYMBOLIC_EXPR_PROOF_SCOPE_CONDITION_ASSUMPTION, out_result);
      if (*out_result == LOOM_SYMBOLIC_PROOF_FALSE) {
        *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
      }
      context->condition_scope = previous_scope;
      loom_symbolic_expr_context_reset(context);
    }
  }
  iree_arena_deinitialize(&condition_arena);
  return status;
}

//===----------------------------------------------------------------------===//
// Public proof APIs
//===----------------------------------------------------------------------===//

iree_status_t loom_symbolic_expr_prove_le(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_proof_result_t* out_result) {
  return loom_symbolic_expr_prove_le_with_scope(
      context, left_expression, right_expression,
      LOOM_SYMBOLIC_EXPR_PROOF_SCOPE_SELECT_CASES, out_result);
}

static iree_status_t loom_symbolic_expr_prove_equal(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_expr_proof_scope_t proof_scope,
    loom_symbolic_proof_result_t* out_result) {
  loom_symbolic_proof_result_t left_le_right = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le_with_scope(
      context, left_expression, right_expression, proof_scope, &left_le_right));
  loom_symbolic_proof_result_t right_le_left = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le_with_scope(
      context, right_expression, left_expression, proof_scope, &right_le_left));
  if (left_le_right == LOOM_SYMBOLIC_PROOF_TRUE &&
      right_le_left == LOOM_SYMBOLIC_PROOF_TRUE) {
    *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
  } else if (left_le_right == LOOM_SYMBOLIC_PROOF_FALSE ||
             right_le_left == LOOM_SYMBOLIC_PROOF_FALSE) {
    *out_result = LOOM_SYMBOLIC_PROOF_FALSE;
  } else {
    *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  }
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_prove_less_than(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_expr_proof_scope_t proof_scope,
    loom_symbolic_proof_result_t* out_result) {
  loom_symbolic_proof_result_t right_le_left = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le_with_scope(
      context, right_expression, left_expression, proof_scope, &right_le_left));
  if (right_le_left == LOOM_SYMBOLIC_PROOF_TRUE) {
    *out_result = LOOM_SYMBOLIC_PROOF_FALSE;
  } else if (right_le_left == LOOM_SYMBOLIC_PROOF_FALSE) {
    *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
  } else {
    *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  }
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_prove_value_relation_with_scope(
    loom_symbolic_expr_context_t* context,
    loom_symbolic_integer_relation_t relation, loom_value_id_t left_value,
    loom_value_id_t right_value, loom_symbolic_expr_proof_scope_t proof_scope,
    loom_symbolic_proof_result_t* out_result) {
  IREE_RETURN_IF_ERROR(loom_symbolic_value_prove_relation(
      context, relation, left_value, right_value, out_result));
  if (*out_result != LOOM_SYMBOLIC_PROOF_UNKNOWN) {
    return iree_ok_status();
  }

  loom_symbolic_expr_t left_expression = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(context, left_value, &left_expression));
  loom_symbolic_expr_t right_expression = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(context, right_value, &right_expression));

  switch (relation) {
    case LOOM_SYMBOLIC_INTEGER_RELATION_EQ:
      return loom_symbolic_expr_prove_equal(context, &left_expression,
                                            &right_expression, proof_scope,
                                            out_result);
    case LOOM_SYMBOLIC_INTEGER_RELATION_NE: {
      loom_symbolic_proof_result_t equal = LOOM_SYMBOLIC_PROOF_UNKNOWN;
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_equal(
          context, &left_expression, &right_expression, proof_scope, &equal));
      if (equal == LOOM_SYMBOLIC_PROOF_TRUE) {
        *out_result = LOOM_SYMBOLIC_PROOF_FALSE;
      } else if (equal == LOOM_SYMBOLIC_PROOF_FALSE) {
        *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
      }
      return iree_ok_status();
    }
    case LOOM_SYMBOLIC_INTEGER_RELATION_LT:
      return loom_symbolic_expr_prove_less_than(context, &left_expression,
                                                &right_expression, proof_scope,
                                                out_result);
    case LOOM_SYMBOLIC_INTEGER_RELATION_LE:
      return loom_symbolic_expr_prove_le_with_scope(context, &left_expression,
                                                    &right_expression,
                                                    proof_scope, out_result);
    case LOOM_SYMBOLIC_INTEGER_RELATION_GT:
      return loom_symbolic_expr_prove_less_than(context, &right_expression,
                                                &left_expression, proof_scope,
                                                out_result);
    case LOOM_SYMBOLIC_INTEGER_RELATION_GE:
      return loom_symbolic_expr_prove_le_with_scope(context, &right_expression,
                                                    &left_expression,
                                                    proof_scope, out_result);
    default:
      return iree_ok_status();
  }
}

iree_status_t loom_symbolic_expr_prove_value_relation(
    loom_symbolic_expr_context_t* context,
    loom_symbolic_integer_relation_t relation, loom_value_id_t left_value,
    loom_value_id_t right_value, loom_symbolic_proof_result_t* out_result) {
  return loom_symbolic_expr_prove_value_relation_with_scope(
      context, relation, left_value, right_value,
      LOOM_SYMBOLIC_EXPR_PROOF_SCOPE_SELECT_CASES, out_result);
}

iree_status_t loom_symbolic_expr_prove_value_relation_with_active_facts(
    loom_symbolic_expr_context_t* context,
    loom_symbolic_integer_relation_t relation, loom_value_id_t left_value,
    loom_value_id_t right_value, loom_symbolic_proof_result_t* out_result) {
  return loom_symbolic_expr_prove_value_relation_with_scope(
      context, relation, left_value, right_value,
      LOOM_SYMBOLIC_EXPR_PROOF_SCOPE_ACTIVE_FACTS, out_result);
}
