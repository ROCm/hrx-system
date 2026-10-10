// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_product.h"

#include <string.h>

#include "iree/base/internal/math.h"

bool loom_symbolic_product_range(loom_value_facts_t left,
                                 loom_value_facts_t right,
                                 loom_value_facts_t* out_facts) {
  if (loom_value_facts_is_float(left) || loom_value_facts_is_float(right)) {
    return false;
  }
  int64_t endpoints[4];
  if (!iree_checked_mul_i64(left.range_lo, right.range_lo, &endpoints[0]) ||
      !iree_checked_mul_i64(left.range_lo, right.range_hi, &endpoints[1]) ||
      !iree_checked_mul_i64(left.range_hi, right.range_lo, &endpoints[2]) ||
      !iree_checked_mul_i64(left.range_hi, right.range_hi, &endpoints[3])) {
    return false;
  }
  loom_value_facts_muli(&left, &right, out_facts);
  return true;
}

static loom_symbolic_product_t loom_symbolic_product_operand(
    const loom_symbolic_expr_context_t* context,
    const loom_value_id_t* value_id, const loom_symbolic_expr_t* expression) {
  loom_symbolic_product_t product = {
      .scale = 1,
      .factors = value_id,
      .factor_count = 1,
  };
  if (loom_symbolic_expr_is_linear(expression) && expression->constant == 0 &&
      expression->term_count == 1) {
    const loom_symbolic_term_t* term = &expression->terms[0];
    const loom_symbolic_product_t* retained =
        loom_symbolic_expr_lookup_product(context, term->value_id);
    if (retained) {
      product = *retained;
      if (!iree_checked_mul_i64(product.scale, term->coefficient,
                                &product.scale)) {
        product.factor_count = 0;
      }
    } else {
      product.scale = term->coefficient;
      product.factors = &term->value_id;
    }
  }
  return product;
}

iree_status_t loom_symbolic_product_multiply(
    loom_symbolic_expr_context_t* context, loom_value_id_t left_value,
    const loom_symbolic_expr_t* left_expression, loom_value_id_t right_value,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_product_t* out_product) {
  *out_product = (loom_symbolic_product_t){0};
  const loom_symbolic_product_t left =
      loom_symbolic_product_operand(context, &left_value, left_expression);
  const loom_symbolic_product_t right =
      loom_symbolic_product_operand(context, &right_value, right_expression);
  int64_t scale = 0;
  if (left.factor_count == 0 || right.factor_count == 0 ||
      left.factor_count > context->maximum_term_count ||
      right.factor_count > context->maximum_term_count - left.factor_count ||
      !iree_checked_mul_i64(left.scale, right.scale, &scale)) {
    return iree_ok_status();
  }
  const iree_host_size_t factor_count = left.factor_count + right.factor_count;
  loom_value_id_t* factors = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, factor_count, sizeof(*factors), (void**)&factors));
  iree_host_size_t left_index = 0;
  iree_host_size_t right_index = 0;
  for (iree_host_size_t i = 0; i < factor_count; ++i) {
    if (right_index == right.factor_count ||
        (left_index < left.factor_count &&
         left.factors[left_index] <= right.factors[right_index])) {
      factors[i] = left.factors[left_index++];
    } else {
      factors[i] = right.factors[right_index++];
    }
  }
  *out_product = (loom_symbolic_product_t){
      .scale = scale,
      .factors = factors,
      .factor_count = factor_count,
  };
  return iree_ok_status();
}

bool loom_symbolic_product_match_terms(
    const loom_symbolic_expr_context_t* context,
    const loom_symbolic_term_t* left, const loom_symbolic_term_t* right,
    int64_t* out_left_coefficient, int64_t* out_right_coefficient) {
  const loom_symbolic_product_t* left_product =
      loom_symbolic_expr_lookup_product(context, left->value_id);
  const loom_symbolic_product_t* right_product =
      loom_symbolic_expr_lookup_product(context, right->value_id);
  return left_product && right_product &&
         left_product->factor_count == right_product->factor_count &&
         memcmp(left_product->factors, right_product->factors,
                left_product->factor_count * sizeof(*left_product->factors)) ==
             0 &&
         iree_checked_mul_i64(left->coefficient, left_product->scale,
                              out_left_coefficient) &&
         iree_checked_mul_i64(right->coefficient, right_product->scale,
                              out_right_coefficient);
}
