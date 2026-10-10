// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_congruence.h"

#include "iree/base/internal/math.h"
#include "loom/util/adaptive_sort.h"
#include "loom/util/fact_table.h"

static uint64_t loom_symbolic_congruence_magnitude(int64_t value) {
  return value < 0 ? UINT64_C(0) - (uint64_t)value : (uint64_t)value;
}

static const loom_symbolic_expr_t* loom_symbolic_congruence_form(
    const loom_symbolic_expr_t* expression) {
  return expression->congruence ? &expression->congruence->expression
                                : expression;
}

static iree_status_t loom_symbolic_congruence_retain(
    loom_symbolic_expr_context_t* context, const loom_symbolic_expr_t* form,
    uint64_t modulus, loom_symbolic_expr_t* output) {
  if (modulus <= 1 || !loom_symbolic_expr_is_linear(form)) {
    return iree_ok_status();
  }
  loom_symbolic_congruence_t* retained = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(context->arena, sizeof(*retained),
                                           (void**)&retained));
  *retained = (loom_symbolic_congruence_t){
      .modulus = modulus,
      .expression = *form,
  };
  retained->expression.congruence = NULL;
  output->congruence = retained;
  return iree_ok_status();
}

iree_status_t loom_symbolic_congruence_restrict(
    loom_symbolic_expr_context_t* context, const loom_symbolic_expr_t* input,
    uint64_t modulus, loom_symbolic_expr_t* output) {
  if (input->congruence) {
    modulus = iree_math_gcd_u64(modulus, input->congruence->modulus);
  }
  return loom_symbolic_congruence_retain(
      context, loom_symbolic_congruence_form(input), modulus, output);
}

iree_status_t loom_symbolic_congruence_combine(
    loom_symbolic_expr_context_t* context, const loom_symbolic_expr_t* left,
    const loom_symbolic_expr_t* right, int64_t sign,
    loom_symbolic_expr_t* output) {
  if (!left->congruence && !right->congruence) {
    return iree_ok_status();
  }
  const uint64_t modulus =
      iree_math_gcd_u64(left->congruence ? left->congruence->modulus : 0,
                        right->congruence ? right->congruence->modulus : 0);
  if (modulus <= 1) {
    return iree_ok_status();
  }
  loom_symbolic_expr_t left_form = *loom_symbolic_congruence_form(left);
  loom_symbolic_expr_t right_form = *loom_symbolic_congruence_form(right);
  left_form.congruence = NULL;
  right_form.congruence = NULL;
  loom_symbolic_expr_t combined;
  IREE_RETURN_IF_ERROR(
      sign == 1
          ? loom_symbolic_expr_add(context, &left_form, &right_form, &combined)
          : loom_symbolic_expr_sub(context, &left_form, &right_form,
                                   &combined));
  return loom_symbolic_congruence_retain(context, &combined, modulus, output);
}

iree_status_t loom_symbolic_congruence_scale(
    loom_symbolic_expr_context_t* context, const loom_symbolic_expr_t* input,
    int64_t multiplier, loom_symbolic_expr_t* output) {
  if (!input->congruence || multiplier == 0) {
    return iree_ok_status();
  }
  const uint64_t magnitude = loom_symbolic_congruence_magnitude(multiplier);
  if (input->congruence->modulus > UINT64_MAX / magnitude) {
    return iree_ok_status();
  }
  loom_symbolic_expr_t scaled;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_mul_i64(
      context, &input->congruence->expression, multiplier, &scaled));
  return loom_symbolic_congruence_retain(
      context, &scaled, input->congruence->modulus * magnitude, output);
}

static uint64_t loom_symbolic_congruence_residue(int64_t value,
                                                 uint64_t modulus) {
  const uint64_t residue = loom_symbolic_congruence_magnitude(value) % modulus;
  return value < 0 && residue != 0 ? modulus - residue : residue;
}

static bool loom_symbolic_congruence_residue_excludes_interval(int64_t constant,
                                                               uint64_t modulus,
                                                               int64_t lower,
                                                               int64_t upper) {
  if (lower > upper) {
    return true;
  }
  if (modulus <= 1) {
    return false;
  }
  const uint64_t residue = loom_symbolic_congruence_residue(constant, modulus);
  const uint64_t lower_residue =
      loom_symbolic_congruence_residue(lower, modulus);
  const uint64_t distance = residue >= lower_residue
                                ? residue - lower_residue
                                : modulus - (lower_residue - residue);
  return distance > (uint64_t)upper - (uint64_t)lower;
}

bool loom_symbolic_congruence_excludes_difference(
    const loom_symbolic_expr_t* left, const loom_symbolic_expr_t* right,
    int64_t lower, int64_t upper) {
  uint64_t modulus =
      iree_math_gcd_u64(left->congruence ? left->congruence->modulus : 0,
                        right->congruence ? right->congruence->modulus : 0);
  left = loom_symbolic_congruence_form(left);
  right = loom_symbolic_congruence_form(right);
  if (!loom_symbolic_expr_is_linear(left) ||
      !loom_symbolic_expr_is_linear(right)) {
    return false;
  }
  int64_t constant = 0;
  if (!iree_checked_sub_i64(left->constant, right->constant, &constant)) {
    return false;
  }
  iree_host_size_t left_index = 0, right_index = 0;
  while (left_index < left->term_count || right_index < right->term_count) {
    int64_t coefficient;
    if (right_index == right->term_count ||
        (left_index < left->term_count &&
         left->terms[left_index].value_id <
             right->terms[right_index].value_id)) {
      coefficient = left->terms[left_index++].coefficient;
    } else if (left_index == left->term_count ||
               right->terms[right_index].value_id <
                   left->terms[left_index].value_id) {
      // Only the magnitude matters for this unpaired term.
      coefficient = right->terms[right_index++].coefficient;
    } else {
      if (!iree_checked_sub_i64(left->terms[left_index++].coefficient,
                                right->terms[right_index++].coefficient,
                                &coefficient)) {
        return false;
      }
    }
    modulus = iree_math_gcd_u64(
        modulus, loom_symbolic_congruence_magnitude(coefficient));
  }
  if (modulus == 0) {
    return constant < lower || constant > upper;
  }
  return loom_symbolic_congruence_residue_excludes_interval(constant, modulus,
                                                            lower, upper);
}

// One exact affine term whose retained periodic selector may cancel with
// another term in the same evaluation. Only single-source selectors are
// candidates; all other exact terms remain bounded residuals.
typedef struct loom_symbolic_congruence_candidate_t {
  // Shared SSA value in the selector's retained congruence.
  loom_value_id_t source_value_id;
  // Difference-term ordinal retained across the source-identity sort.
  uint16_t difference_term_ordinal;
  // Scaled source coefficient in the outer expression difference.
  int64_t source_coefficient;
  // Scaled constant in the selector's retained congruence.
  int64_t constant;
  // Scaled selector modulus.
  uint64_t modulus;
} loom_symbolic_congruence_candidate_t;

static bool loom_symbolic_congruence_candidate_source_less(
    const loom_symbolic_congruence_candidate_t* lhs,
    const loom_symbolic_congruence_candidate_t* rhs) {
  return lhs->source_value_id < rhs->source_value_id;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_symbolic_congruence_sort_candidates_by_source,
                          loom_symbolic_congruence_candidate_t,
                          loom_symbolic_congruence_candidate_source_less)

enum {
  LOOM_SYMBOLIC_CONGRUENCE_MAX_DIFFERENCE_TERM_COUNT =
      2 * LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT,
};
static_assert(LOOM_SYMBOLIC_CONGRUENCE_MAX_DIFFERENCE_TERM_COUNT <= UINT16_MAX,
              "difference term ordinals must fit in uint16_t");

typedef struct loom_symbolic_congruence_difference_cursor_t {
  // Exact normalized expression contributing positively.
  const loom_symbolic_expr_t* left;
  // Exact normalized expression contributing negatively.
  const loom_symbolic_expr_t* right;
  // Next unread left term.
  iree_host_size_t left_index;
  // Next unread right term.
  iree_host_size_t right_index;
  // Whether coefficient negation or subtraction overflowed.
  bool failed;
} loom_symbolic_congruence_difference_cursor_t;

static bool loom_symbolic_congruence_difference_cursor_next(
    loom_symbolic_congruence_difference_cursor_t* cursor,
    loom_symbolic_term_t* out_term) {
  while (cursor->left_index < cursor->left->term_count ||
         cursor->right_index < cursor->right->term_count) {
    if (cursor->right_index == cursor->right->term_count ||
        (cursor->left_index < cursor->left->term_count &&
         cursor->left->terms[cursor->left_index].value_id <
             cursor->right->terms[cursor->right_index].value_id)) {
      *out_term = cursor->left->terms[cursor->left_index++];
    } else if (cursor->left_index == cursor->left->term_count ||
               cursor->right->terms[cursor->right_index].value_id <
                   cursor->left->terms[cursor->left_index].value_id) {
      *out_term = cursor->right->terms[cursor->right_index++];
      if (out_term->coefficient == INT64_MIN) {
        cursor->failed = true;
        return false;
      }
      out_term->coefficient = -out_term->coefficient;
    } else {
      const loom_symbolic_term_t left_term =
          cursor->left->terms[cursor->left_index++];
      const loom_symbolic_term_t right_term =
          cursor->right->terms[cursor->right_index++];
      int64_t coefficient = 0;
      if (!iree_checked_sub_i64(left_term.coefficient, right_term.coefficient,
                                &coefficient)) {
        cursor->failed = true;
        return false;
      }
      loom_value_id_t relation_value_id = left_term.relation_value_id;
      if (relation_value_id == LOOM_VALUE_ID_INVALID) {
        relation_value_id = left_term.value_id;
      }
      loom_value_id_t right_relation_value_id = right_term.relation_value_id;
      if (right_relation_value_id == LOOM_VALUE_ID_INVALID) {
        right_relation_value_id = right_term.value_id;
      }
      if (right_relation_value_id != relation_value_id) {
        relation_value_id = left_term.value_id;
      }
      *out_term = (loom_symbolic_term_t){
          .coefficient = coefficient,
          .value_id = left_term.value_id,
          .relation_value_id = relation_value_id,
      };
    }
    if (out_term->coefficient != 0) {
      return true;
    }
  }
  return false;
}

static loom_value_facts_t loom_symbolic_congruence_intersect_integer_facts(
    loom_value_facts_t lhs, loom_value_facts_t rhs) {
  if (loom_value_facts_is_unknown(lhs) || loom_value_facts_is_float(lhs)) {
    return rhs;
  }
  if (loom_value_facts_is_unknown(rhs) || loom_value_facts_is_float(rhs)) {
    return lhs;
  }
  const int64_t lower_bound = iree_max(lhs.range_lo, rhs.range_lo);
  const int64_t upper_bound = iree_min(lhs.range_hi, rhs.range_hi);
  return lower_bound <= upper_bound
             ? loom_value_facts_make(lower_bound, upper_bound, 1)
             : loom_value_facts_unknown();
}

static bool loom_symbolic_congruence_term_interval(
    const loom_symbolic_expr_context_t* context,
    const loom_symbolic_term_t term, int64_t* out_minimum,
    int64_t* out_maximum) {
  if (!context->fact_table) {
    return false;
  }
  loom_value_facts_t facts =
      loom_value_fact_table_lookup(context->fact_table, term.value_id);
  if (term.relation_value_id != LOOM_VALUE_ID_INVALID &&
      term.relation_value_id != term.value_id) {
    facts = loom_symbolic_congruence_intersect_integer_facts(
        facts, loom_value_fact_table_lookup(context->fact_table,
                                            term.relation_value_id));
  }
  if (loom_value_facts_is_unknown(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  const int64_t lower_bound =
      term.coefficient < 0 ? facts.range_hi : facts.range_lo;
  const int64_t upper_bound =
      term.coefficient < 0 ? facts.range_lo : facts.range_hi;
  return iree_checked_mul_i64(term.coefficient, lower_bound, out_minimum) &&
         iree_checked_mul_i64(term.coefficient, upper_bound, out_maximum);
}

#if IREE_HAVE_ATTRIBUTE(minsize)
__attribute__((minsize))
#endif  // IREE_HAVE_ATTRIBUTE(minsize)
IREE_ATTRIBUTE_NOINLINE static bool
loom_symbolic_congruence_prove_with_bounded_residual(
    const loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression, int64_t lower,
    int64_t upper) {
  if (!loom_symbolic_expr_is_linear(left_expression) ||
      !loom_symbolic_expr_is_linear(right_expression)) {
    return false;
  }

  int64_t proof_constant = 0;
  if (!iree_checked_sub_i64(left_expression->constant,
                            right_expression->constant, &proof_constant)) {
    return false;
  }

  loom_symbolic_congruence_candidate_t
      candidates[LOOM_SYMBOLIC_CONGRUENCE_MAX_DIFFERENCE_TERM_COUNT];
  iree_host_size_t candidate_count = 0;
  iree_host_size_t difference_term_count = 0;
  loom_symbolic_congruence_difference_cursor_t cursor = {
      .left = left_expression,
      .right = right_expression,
  };
  bool difference_term_overflow = false;
  loom_symbolic_term_t term = {0};
  while (loom_symbolic_congruence_difference_cursor_next(&cursor, &term)) {
    if (difference_term_count ==
        LOOM_SYMBOLIC_CONGRUENCE_MAX_DIFFERENCE_TERM_COUNT) {
      difference_term_overflow = true;
      break;
    }
    const uint16_t difference_term_ordinal = (uint16_t)difference_term_count++;
    loom_symbolic_expr_summary_t summary = {0};
    if (!loom_symbolic_expr_context_try_lookup_summary(context, term.value_id,
                                                       &summary) ||
        !summary.expression.congruence) {
      continue;
    }
    const loom_symbolic_congruence_t* congruence =
        summary.expression.congruence;
    if (!loom_symbolic_expr_is_linear(&congruence->expression) ||
        congruence->expression.term_count != 1) {
      continue;
    }
    int64_t source_coefficient = 0;
    int64_t constant = 0;
    const uint64_t coefficient_magnitude =
        loom_symbolic_congruence_magnitude(term.coefficient);
    if (!iree_checked_mul_i64(term.coefficient,
                              congruence->expression.terms[0].coefficient,
                              &source_coefficient) ||
        !iree_checked_mul_i64(term.coefficient, congruence->expression.constant,
                              &constant) ||
        coefficient_magnitude == 0 ||
        congruence->modulus > UINT64_MAX / coefficient_magnitude) {
      continue;
    }
    candidates[candidate_count++] = (loom_symbolic_congruence_candidate_t){
        .source_value_id = congruence->expression.terms[0].value_id,
        .difference_term_ordinal = difference_term_ordinal,
        .source_coefficient = source_coefficient,
        .constant = constant,
        .modulus = congruence->modulus * coefficient_magnitude,
    };
  }
  if (cursor.failed || difference_term_overflow || candidate_count < 2) {
    return false;
  }

  loom_symbolic_congruence_sort_candidates_by_source(candidates,
                                                     candidate_count);
  uint64_t proof_modulus = 0;
  uint64_t selected_term_bits
      [(LOOM_SYMBOLIC_CONGRUENCE_MAX_DIFFERENCE_TERM_COUNT + 63) / 64] = {0};
  for (iree_host_size_t group_begin = 0; group_begin < candidate_count;) {
    iree_host_size_t group_end = group_begin + 1;
    while (group_end < candidate_count &&
           candidates[group_end].source_value_id ==
               candidates[group_begin].source_value_id) {
      ++group_end;
    }
    int64_t source_coefficient = 0;
    int64_t group_constant = 0;
    uint64_t group_modulus = 0;
    bool group_valid = true;
    for (iree_host_size_t i = group_begin; i < group_end; ++i) {
      group_valid = group_valid &&
                    iree_checked_add_i64(source_coefficient,
                                         candidates[i].source_coefficient,
                                         &source_coefficient) &&
                    iree_checked_add_i64(group_constant, candidates[i].constant,
                                         &group_constant);
      group_modulus = iree_math_gcd_u64(group_modulus, candidates[i].modulus);
    }
    int64_t combined_constant = 0;
    const bool selected = group_valid && source_coefficient == 0 &&
                          group_modulus > 1 &&
                          iree_checked_add_i64(proof_constant, group_constant,
                                               &combined_constant);
    if (selected) {
      proof_constant = combined_constant;
      proof_modulus = iree_math_gcd_u64(proof_modulus, group_modulus);
      for (iree_host_size_t i = group_begin; i < group_end; ++i) {
        const uint16_t ordinal = candidates[i].difference_term_ordinal;
        selected_term_bits[ordinal / 64] |= UINT64_C(1) << (ordinal % 64);
      }
    }
    group_begin = group_end;
  }
  if (proof_modulus <= 1) {
    return false;
  }

  int64_t residual_minimum = 0;
  int64_t residual_maximum = 0;
  iree_host_size_t difference_term_index = 0;
  cursor = (loom_symbolic_congruence_difference_cursor_t){
      .left = left_expression,
      .right = right_expression,
  };
  while (loom_symbolic_congruence_difference_cursor_next(&cursor, &term)) {
    const bool selected = (selected_term_bits[difference_term_index / 64] &
                           (UINT64_C(1) << (difference_term_index % 64))) != 0;
    ++difference_term_index;
    if (selected) {
      continue;
    }
    int64_t term_minimum = 0;
    int64_t term_maximum = 0;
    if (!loom_symbolic_congruence_term_interval(context, term, &term_minimum,
                                                &term_maximum) ||
        !iree_checked_add_i64(residual_minimum, term_minimum,
                              &residual_minimum) ||
        !iree_checked_add_i64(residual_maximum, term_maximum,
                              &residual_maximum)) {
      return false;
    }
  }
  if (cursor.failed) {
    return false;
  }

  int64_t adjusted_lower = 0;
  int64_t adjusted_upper = 0;
  if (!iree_checked_sub_i64(lower, residual_maximum, &adjusted_lower) ||
      !iree_checked_sub_i64(upper, residual_minimum, &adjusted_upper)) {
    return false;
  }
  return loom_symbolic_congruence_residue_excludes_interval(
      proof_constant, proof_modulus, adjusted_lower, adjusted_upper);
}

bool loom_symbolic_congruence_prove_difference_outside_interval(
    const loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression, int64_t lower,
    int64_t upper) {
  if (lower > upper) {
    return true;
  }
  if (loom_symbolic_congruence_excludes_difference(
          left_expression, right_expression, lower, upper)) {
    return true;
  }
  if (!left_expression->congruence && !right_expression->congruence) {
    return false;
  }
  return loom_symbolic_congruence_prove_with_bounded_residual(
      context, left_expression, right_expression, lower, upper);
}
