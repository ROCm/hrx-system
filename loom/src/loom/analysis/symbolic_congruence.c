// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_congruence.h"

#include "iree/base/internal/math.h"

static uint64_t loom_symbolic_congruence_magnitude(int64_t value) {
  return value < 0 ? UINT64_C(0) - (uint64_t)value : (uint64_t)value;
}

const loom_symbolic_expr_t* loom_symbolic_congruence_form(
    const loom_symbolic_expr_t* expression) {
  IREE_ASSERT_ARGUMENT(expression);
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
  const uint64_t residue = loom_symbolic_congruence_residue(constant, modulus);
  const uint64_t lower_residue =
      loom_symbolic_congruence_residue(lower, modulus);
  const uint64_t distance = residue >= lower_residue
                                ? residue - lower_residue
                                : modulus - (lower_residue - residue);
  return distance > (uint64_t)upper - (uint64_t)lower;
}
