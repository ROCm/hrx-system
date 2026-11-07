// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/math/inverse_hyperbolic.h"

#include <math.h>

typedef loom_math_evaluation_value_t loom_math_value_t;

// Odd polynomials retain tiny inputs without subtracting nearly equal values.
// Coefficients are in increasing order of the even power multiplying x.
static iree_status_t loom_math_build_odd_polynomial(
    loom_math_evaluation_t* evaluation, loom_math_value_t* input,
    const double* coefficients, iree_host_size_t coefficient_count,
    loom_math_value_t* out_value) {
  loom_math_value_t square;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_MULF, input, input, &square));
  loom_math_value_t accumulator;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_constant(
      evaluation, coefficients[coefficient_count - 1], &accumulator));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = coefficient_count - 1;
       i > 0 && iree_status_is_ok(status); --i) {
    loom_math_value_t coefficient;
    status = loom_math_evaluation_build_constant(
        evaluation, coefficients[i - 1], &coefficient);
    if (iree_status_is_ok(status)) {
      status = loom_math_evaluation_build_multiply_add(
          evaluation, &accumulator, &square, &coefficient, &accumulator);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  return loom_math_evaluation_build_binary(evaluation,
                                           LOOM_MATH_EVALUATION_BINARY_MULF,
                                           input, &accumulator, out_value);
}

static iree_status_t loom_math_build_atanh_polynomial(
    loom_math_evaluation_t* evaluation, loom_math_value_t* input,
    loom_math_value_t* out_value) {
  // The omitted tail is below 1.6e-9 for |x| <= 0.5.
  static const double kCoefficients[] = {
      1.0,        1.0 / 3.0,  1.0 / 5.0,  1.0 / 7.0,  1.0 / 9.0,  1.0 / 11.0,
      1.0 / 13.0, 1.0 / 15.0, 1.0 / 17.0, 1.0 / 19.0, 1.0 / 21.0, 1.0 / 23.0,
  };
  return loom_math_build_odd_polynomial(evaluation, input, kCoefficients,
                                        IREE_ARRAYSIZE(kCoefficients),
                                        out_value);
}

static iree_status_t loom_math_build_natural_log(
    loom_math_evaluation_t* evaluation, loom_math_value_t* input,
    loom_math_value_t* out_value) {
  loom_math_value_t log2;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_unary(
      evaluation, LOOM_MATH_EVALUATION_UNARY_LOG2F, input, &log2));
  loom_math_value_t ln2;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_constant(
      evaluation, 0.69314718055994530942, &ln2));
  return loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_MULF, &log2, &ln2, out_value);
}

// log(x + sqrt(r)) becomes log(x) + ln(2) for x >= 4096. Its omitted
// correction is below 1.5e-8 there, and avoiding 2*x preserves finite results
// through FLT_MAX. Select the argument before the logarithm to emit it once.
static iree_status_t loom_math_build_log_root(
    loom_math_evaluation_t* evaluation, loom_math_value_t* input,
    loom_math_value_t* radicand, loom_math_value_t* out_value) {
  loom_math_value_t root;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_unary(
      evaluation, LOOM_MATH_EVALUATION_UNARY_SQRTF, radicand, &root));
  loom_math_value_t sum;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_ADDF, input, &root, &sum));
  loom_math_value_t threshold;
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(evaluation, 4096.0, &threshold));
  loom_value_id_t large;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_ordered_greater_equal(
      evaluation, input, &threshold, &large));
  loom_math_value_t argument;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_select(
      evaluation, large, input, &sum, &argument));
  loom_math_value_t logarithm;
  IREE_RETURN_IF_ERROR(
      loom_math_build_natural_log(evaluation, &argument, &logarithm));
  loom_math_value_t ln2;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_constant(
      evaluation, 0.69314718055994530942, &ln2));
  loom_math_value_t large_result;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_ADDF, &logarithm, &ln2,
      &large_result));
  return loom_math_evaluation_build_select(evaluation, large, &large_result,
                                           &logarithm, out_value);
}

static iree_status_t loom_math_build_asinh(loom_math_evaluation_t* evaluation,
                                           loom_math_value_t* out_value) {
  loom_math_value_t magnitude;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_unary(
      evaluation, LOOM_MATH_EVALUATION_UNARY_ABSF, &evaluation->input,
      &magnitude));
  // Through x^17, the alternating-series remainder is below 2e-8 at 0.5.
  static const double kCoefficients[] = {
      1.0,
      -1.0 / 6.0,
      3.0 / 40.0,
      -5.0 / 112.0,
      35.0 / 1152.0,
      -63.0 / 2816.0,
      231.0 / 13312.0,
      -143.0 / 10240.0,
      6435.0 / 557056.0,
  };
  loom_math_value_t small_result;
  IREE_RETURN_IF_ERROR(loom_math_build_odd_polynomial(
      evaluation, &magnitude, kCoefficients, IREE_ARRAYSIZE(kCoefficients),
      &small_result));
  loom_math_value_t one;
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(evaluation, 1.0, &one));
  loom_math_value_t radicand;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_multiply_add(
      evaluation, &magnitude, &magnitude, &one, &radicand));
  loom_math_value_t regular_result;
  IREE_RETURN_IF_ERROR(loom_math_build_log_root(evaluation, &magnitude,
                                                &radicand, &regular_result));
  loom_math_value_t threshold;
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(evaluation, 0.5, &threshold));
  loom_value_id_t regular;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_ordered_greater_equal(
      evaluation, &magnitude, &threshold, &regular));
  loom_math_value_t result;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_select(
      evaluation, regular, &regular_result, &small_result, &result));
  return loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_COPYSIGNF, &result,
      &evaluation->input, out_value);
}

static iree_status_t loom_math_build_acosh(loom_math_evaluation_t* evaluation,
                                           loom_math_value_t* out_value) {
  loom_math_value_t one;
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(evaluation, 1.0, &one));
  loom_math_value_t below;
  loom_math_value_t above;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_SUBF, &evaluation->input, &one,
      &below));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_ADDF, &evaluation->input, &one,
      &above));
  // acosh(x) = 2*atanh(sqrt((x-1)/(x+1))). For 1 <= x < 1.5 the
  // polynomial argument is below sqrt(0.2); x-1 is exact by Sterbenz's lemma.
  loom_math_value_t ratio;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_DIVF, &below, &above, &ratio));
  loom_math_value_t near_input;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_unary(
      evaluation, LOOM_MATH_EVALUATION_UNARY_SQRTF, &ratio, &near_input));
  loom_math_value_t near_result;
  IREE_RETURN_IF_ERROR(
      loom_math_build_atanh_polynomial(evaluation, &near_input, &near_result));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_ADDF, &near_result, &near_result,
      &near_result));
  loom_math_value_t radicand;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_MULF, &below, &above, &radicand));
  loom_math_value_t regular_result;
  IREE_RETURN_IF_ERROR(loom_math_build_log_root(evaluation, &evaluation->input,
                                                &radicand, &regular_result));
  loom_math_value_t threshold;
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(evaluation, 1.5, &threshold));
  loom_value_id_t regular;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_ordered_greater_equal(
      evaluation, &evaluation->input, &threshold, &regular));
  loom_math_value_t result;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_select(
      evaluation, regular, &regular_result, &near_result, &result));
  // Values below -1 also have a positive ratio; the domain check cannot be
  // delegated to the square root. NaN inputs follow the same false arm.
  loom_value_id_t in_domain;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_ordered_greater_equal(
      evaluation, &evaluation->input, &one, &in_domain));
  loom_math_value_t nan_value;
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(evaluation, NAN, &nan_value));
  return loom_math_evaluation_build_select(evaluation, in_domain, &result,
                                           &nan_value, out_value);
}

static iree_status_t loom_math_build_atanh(loom_math_evaluation_t* evaluation,
                                           loom_math_value_t* out_value) {
  loom_math_value_t magnitude;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_unary(
      evaluation, LOOM_MATH_EVALUATION_UNARY_ABSF, &evaluation->input,
      &magnitude));
  loom_math_value_t small_result;
  IREE_RETURN_IF_ERROR(
      loom_math_build_atanh_polynomial(evaluation, &magnitude, &small_result));
  loom_math_value_t one;
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(evaluation, 1.0, &one));
  loom_math_value_t numerator;
  loom_math_value_t denominator;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_ADDF, &one, &magnitude,
      &numerator));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_SUBF, &one, &magnitude,
      &denominator));
  loom_math_value_t ratio;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_DIVF, &numerator, &denominator,
      &ratio));
  loom_math_value_t logarithm;
  IREE_RETURN_IF_ERROR(
      loom_math_build_natural_log(evaluation, &ratio, &logarithm));
  loom_math_value_t half;
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(evaluation, 0.5, &half));
  loom_math_value_t regular_result;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_MULF, &half, &logarithm,
      &regular_result));
  loom_value_id_t regular;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_ordered_greater_equal(
      evaluation, &magnitude, &half, &regular));
  loom_math_value_t result;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_select(
      evaluation, regular, &regular_result, &small_result, &result));
  // Logarithm primitives may only define nonnegative arguments. Preserve
  // atanh's domain independently of their result for a negative ratio.
  loom_value_id_t in_domain;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_ordered_greater_equal(
      evaluation, &one, &magnitude, &in_domain));
  loom_math_value_t nan_value;
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(evaluation, NAN, &nan_value));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_select(
      evaluation, in_domain, &result, &nan_value, &result));
  return loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_COPYSIGNF, &result,
      &evaluation->input, out_value);
}

iree_status_t loom_math_build_inverse_hyperbolic(
    loom_target_math_op_t math_op, loom_math_evaluation_t* evaluation,
    loom_math_evaluation_value_t* out_value) {
  switch (math_op) {
    case LOOM_TARGET_MATH_OP_ASINHF:
      return loom_math_build_asinh(evaluation, out_value);
    case LOOM_TARGET_MATH_OP_ACOSHF:
      return loom_math_build_acosh(evaluation, out_value);
    case LOOM_TARGET_MATH_OP_ATANHF:
      return loom_math_build_atanh(evaluation, out_value);
    default:
      IREE_ASSERT_UNREACHABLE("recipe selected non-inverse-hyperbolic op");
      IREE_BUILTIN_UNREACHABLE();
  }
}
