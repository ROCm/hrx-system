// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/math/logarithm.h"

#include <math.h>

#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"

typedef loom_math_evaluation_value_t loom_math_logarithm_value_t;

// frexp owns exact normalization, including subnormals on targets whose
// ordinary floating arithmetic flushes them. Keep that operation visible to
// target legalization instead of introducing another IEEE decoder here.
static iree_status_t loom_math_logarithm_decompose(
    loom_math_evaluation_t* evaluation, loom_math_logarithm_value_t* input,
    loom_math_logarithm_value_t* out_fraction,
    loom_math_logarithm_value_t* out_exponent) {
  loom_builder_t* builder = &evaluation->rewriter->builder;
  const loom_type_t float_type = evaluation->value_type;
  loom_type_t exponent_type = float_type;
  exponent_type.header = loom_type_make_header(
      loom_type_kind(float_type), LOOM_SCALAR_TYPE_I32,
      loom_type_rank(float_type), loom_type_flags(float_type));
  const bool is_vector = loom_type_is_vector(float_type);
  loom_op_t* decomposed = NULL;
  if (is_vector) {
    const loom_type_t result_types[] = {float_type, exponent_type};
    IREE_RETURN_IF_ERROR(
        loom_vector_frexpf_build(builder, input->accumulator, result_types,
                                 evaluation->location, &decomposed));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_scalar_frexpf_build(builder, input->accumulator, float_type,
                                 evaluation->location, &decomposed));
  }
  *out_fraction =
      loom_math_evaluation_value_from_id(loom_op_results(decomposed)[0]);
  loom_op_t* converted = NULL;
  IREE_RETURN_IF_ERROR(
      (is_vector ? loom_vector_sitofp_build : loom_scalar_sitofp_build)(
          builder, loom_op_results(decomposed)[1], exponent_type, float_type,
          evaluation->location, &converted));
  *out_exponent =
      loom_math_evaluation_value_from_id(loom_op_results(converted)[0]);
  return iree_ok_status();
}

// log(m) = 2*s*(1 + s^2/3 + s^4/5 + ...), s = (m-1)/(m+1).
// Reducing m to [sqrt(1/2), sqrt(2)) bounds |s| by 0.172. Five terms
// leave less than 8e-10 absolute truncation error; ten leave less than 9e-18.
// The subtraction m-1 is exact, retaining accuracy on both sides of one.
static iree_status_t loom_math_logarithm_reduced(
    loom_math_evaluation_t* evaluation, loom_math_logarithm_value_t* fraction,
    loom_math_logarithm_value_t* exponent,
    loom_math_logarithm_value_t* out_logarithm) {
  loom_math_logarithm_value_t one, threshold, doubled, decremented;
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(evaluation, 1.0, &one));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_constant(
      evaluation, 0.7071067811865475244, &threshold));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_ADDF, fraction, fraction,
      &doubled));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_SUBF, exponent, &one,
      &decremented));
  loom_value_id_t upper_half;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_ordered_greater_equal(
      evaluation, fraction, &threshold, &upper_half));
  loom_math_logarithm_value_t mantissa;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_select(
      evaluation, upper_half, fraction, &doubled, &mantissa));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_select(
      evaluation, upper_half, exponent, &decremented, exponent));
  loom_math_logarithm_value_t numerator, denominator, reduced, square;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_SUBF, &mantissa, &one,
      &numerator));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_ADDF, &mantissa, &one,
      &denominator));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_DIVF, &numerator, &denominator,
      &reduced));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_MULF, &reduced, &reduced,
      &square));
  const int term_count =
      loom_type_element_type(evaluation->value_type) == LOOM_SCALAR_TYPE_F64
          ? 10
          : 5;
  static const double kCoefficients[] = {
      1.0,        1.0 / 3.0,  1.0 / 5.0,  1.0 / 7.0,  1.0 / 9.0,
      1.0 / 11.0, 1.0 / 13.0, 1.0 / 15.0, 1.0 / 17.0, 1.0 / 19.0,
  };
  loom_math_logarithm_value_t polynomial;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_constant(
      evaluation, kCoefficients[term_count - 1], &polynomial));
  iree_status_t status = iree_ok_status();
  for (int term = term_count - 2; term >= 0 && iree_status_is_ok(status);
       --term) {
    loom_math_logarithm_value_t coefficient;
    status = loom_math_evaluation_build_constant(
        evaluation, kCoefficients[term], &coefficient);
    if (iree_status_is_ok(status)) {
      status = loom_math_evaluation_build_fma(evaluation, &polynomial, &square,
                                              &coefficient, &polynomial);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  loom_math_logarithm_value_t twice_reduced;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_ADDF, &reduced, &reduced,
      &twice_reduced));
  return loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_MULF, &twice_reduced, &polynomial,
      out_logarithm);
}

static iree_status_t loom_math_logarithm_base(
    loom_target_math_op_t math_op, loom_math_evaluation_t* evaluation,
    loom_math_logarithm_value_t* input,
    loom_math_logarithm_value_t* out_value) {
  loom_math_logarithm_value_t fraction, exponent, logarithm;
  IREE_RETURN_IF_ERROR(
      loom_math_logarithm_decompose(evaluation, input, &fraction, &exponent));
  IREE_RETURN_IF_ERROR(loom_math_logarithm_reduced(evaluation, &fraction,
                                                   &exponent, &logarithm));
  loom_math_logarithm_value_t scale, result;
  if (math_op == LOOM_TARGET_MATH_OP_LOG2F) {
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_constant(
        evaluation, 1.4426950408889634074, &scale));
    // Reconstructing in base two preserves exact powers of two.
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_fma(
        evaluation, &logarithm, &scale, &exponent, &result));
  } else {
    const bool base_ten = math_op == LOOM_TARGET_MATH_OP_LOG10F;
    if (base_ten) {
      IREE_RETURN_IF_ERROR(loom_math_evaluation_build_constant(
          evaluation, 0.4342944819032518277, &scale));
      IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
          evaluation, LOOM_MATH_EVALUATION_BINARY_MULF, &logarithm, &scale,
          &logarithm));
    }
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_constant(
        evaluation, base_ten ? 0.3010299956639811952 : 0.6931471805599453094,
        &scale));
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_fma(
        evaluation, &exponent, &scale, &logarithm, &result));
  }

  // Classify the normalized fraction, not the original input: a floating
  // comparison against a subnormal input may otherwise mistake it for zero.
  loom_math_logarithm_value_t zero, infinity, negative_infinity, nan;
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(evaluation, 0.0, &zero));
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(evaluation, INFINITY, &infinity));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_constant(
      evaluation, -INFINITY, &negative_infinity));
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(evaluation, NAN, &nan));
  loom_value_id_t nonnegative, is_zero, is_infinite;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_ordered_greater_equal(
      evaluation, &fraction, &zero, &nonnegative));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_ordered_equal(
      evaluation, &fraction, &zero, &is_zero));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_ordered_equal(
      evaluation, &fraction, &infinity, &is_infinite));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_select(
      evaluation, nonnegative, &result, &nan, &result));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_select(
      evaluation, is_zero, &negative_infinity, &result, &result));
  return loom_math_evaluation_build_select(evaluation, is_infinite, &infinity,
                                           &result, out_value);
}

iree_status_t loom_math_build_logarithm(
    loom_target_math_op_t math_op, loom_math_evaluation_t* evaluation,
    loom_math_logarithm_value_t* out_value) {
  // Source permissions do not prove anything about intermediate infinities or
  // NaNs. Reassociation also destroys log1p's rounded-increment correction.
  loom_math_evaluation_t arithmetic = *evaluation;
  arithmetic.fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_CONTRACT;
  arithmetic.recipe_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_NONE;
  if (math_op != LOOM_TARGET_MATH_OP_LOG1PF) {
    return loom_math_logarithm_base(math_op, &arithmetic, &arithmetic.input,
                                    out_value);
  }

  // For u = round(1+x), log1p(x) ~= log(u) * x/(u-1). When u rounds to
  // one the result is x itself, including signed zeros and subnormals.
  loom_math_logarithm_value_t one, sum, increment, denominator, ratio;
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(&arithmetic, 1.0, &one));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      &arithmetic, LOOM_MATH_EVALUATION_BINARY_ADDF, &one, &arithmetic.input,
      &sum));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      &arithmetic, LOOM_MATH_EVALUATION_BINARY_SUBF, &sum, &one, &increment));
  loom_value_id_t rounded_to_one;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_ordered_equal(
      &arithmetic, &sum, &one, &rounded_to_one));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_select(
      &arithmetic, rounded_to_one, &one, &increment, &denominator));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      &arithmetic, LOOM_MATH_EVALUATION_BINARY_DIVF, &arithmetic.input,
      &denominator, &ratio));
  loom_math_logarithm_value_t logarithm, corrected, result;
  IREE_RETURN_IF_ERROR(loom_math_logarithm_base(LOOM_TARGET_MATH_OP_LOGF,
                                                &arithmetic, &sum, &logarithm));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      &arithmetic, LOOM_MATH_EVALUATION_BINARY_MULF, &logarithm, &ratio,
      &corrected));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_select(
      &arithmetic, rounded_to_one, &arithmetic.input, &corrected, &result));
  loom_math_logarithm_value_t infinity;
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(&arithmetic, INFINITY, &infinity));
  loom_value_id_t is_infinite;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_ordered_equal(
      &arithmetic, &arithmetic.input, &infinity, &is_infinite));
  return loom_math_evaluation_build_select(&arithmetic, is_infinite, &infinity,
                                           &result, out_value);
}
