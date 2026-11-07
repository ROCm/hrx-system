// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/math/exponential.h"

#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"

typedef loom_math_evaluation_value_t loom_math_exponential_value_t;

// The reduction supplies an integral exponent inside the normal range. Adding
// 1.5*2^fraction_bits puts that integer in the low significand bits; biasing
// and shifting those bits constructs 2^exponent without a floating conversion
// or the normalization and rounding required by general ldexp.
static iree_status_t loom_math_exponential_scale(
    loom_math_evaluation_t* evaluation, loom_math_exponential_value_t* exponent,
    loom_math_exponential_value_t* out_value) {
  const loom_type_t float_type = evaluation->value_type;
  const bool wide = loom_type_element_type(float_type) == LOOM_SCALAR_TYPE_F64;
  const bool is_vector = loom_type_is_vector(float_type);
  loom_type_t bits_type = float_type;
  bits_type.header = loom_type_make_header(
      loom_type_kind(float_type),
      wide ? LOOM_SCALAR_TYPE_I64 : LOOM_SCALAR_TYPE_I32,
      loom_type_rank(float_type), loom_type_flags(float_type));
  loom_math_exponential_value_t shift, shifted;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_constant(
      evaluation, wide ? 0x1.8p52 : 0x1.8p23, &shift));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_ADDF, exponent, &shift,
      &shifted));
  loom_builder_t* builder = &evaluation->rewriter->builder;
  loom_op_t* bits = NULL;
  IREE_RETURN_IF_ERROR(
      (is_vector ? loom_vector_bitcast_build : loom_scalar_bitcast_build)(
          builder, shifted.accumulator, float_type, bits_type,
          evaluation->location, &bits));
  loom_op_t* bias = NULL;
  loom_op_t* fraction_bits = NULL;
  if (is_vector) {
    IREE_RETURN_IF_ERROR(loom_vector_splati_build(
        builder, wide ? 1023 : 127, bits_type, evaluation->location, &bias));
    IREE_RETURN_IF_ERROR(
        loom_vector_splati_build(builder, wide ? 52 : 23, bits_type,
                                 evaluation->location, &fraction_bits));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_scalar_constant_build(builder, loom_attr_i64(wide ? 1023 : 127),
                                   bits_type, evaluation->location, &bias));
    IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
        builder, loom_attr_i64(wide ? 52 : 23), bits_type, evaluation->location,
        &fraction_bits));
  }
  loom_op_t* biased = NULL;
  IREE_RETURN_IF_ERROR(
      (is_vector ? loom_vector_addi_build : loom_scalar_addi_build)(
          builder, 0, loom_op_results(bits)[0], loom_op_results(bias)[0],
          bits_type, evaluation->location, &biased));
  loom_op_t* encoded = NULL;
  IREE_RETURN_IF_ERROR((is_vector ? loom_vector_shli_build
                                  : loom_scalar_shli_build)(
      builder, 0, loom_op_results(biased)[0], loom_op_results(fraction_bits)[0],
      bits_type, evaluation->location, &encoded));
  loom_op_t* result = NULL;
  IREE_RETURN_IF_ERROR(
      (is_vector ? loom_vector_bitcast_build : loom_scalar_bitcast_build)(
          builder, loom_op_results(encoded)[0], bits_type, float_type,
          evaluation->location, &result));
  *out_value = loom_math_evaluation_value_from_id(loom_op_results(result)[0]);
  return iree_ok_status();
}

// expm1(r) = r + r^2*(1/2! + r/3! + ...), |r| <= ln(2)/2.
// Degrees seven and thirteen bound absolute truncation error by 6e-9 and
// 6e-18 respectively. Keeping r separate retains small differences that
// would disappear in a rounded exp(r)-1.
static iree_status_t loom_math_exponential_reduced(
    loom_math_evaluation_t* evaluation, loom_math_exponential_value_t* residual,
    loom_math_exponential_value_t* out_value) {
  static const double kCoefficients[] = {
      1.0 / 2,       1.0 / 6,        1.0 / 24,        1.0 / 120,
      1.0 / 720,     1.0 / 5040,     1.0 / 40320,     1.0 / 362880,
      1.0 / 3628800, 1.0 / 39916800, 1.0 / 479001600, 1.0 / 6227020800,
  };
  const int term_count =
      loom_type_element_type(evaluation->value_type) == LOOM_SCALAR_TYPE_F64
          ? 12
          : 6;
  loom_math_exponential_value_t polynomial;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_constant(
      evaluation, kCoefficients[term_count - 1], &polynomial));
  iree_status_t status = iree_ok_status();
  for (int term = term_count - 2; term >= 0 && iree_status_is_ok(status);
       --term) {
    loom_math_exponential_value_t coefficient;
    status = loom_math_evaluation_build_constant(
        evaluation, kCoefficients[term], &coefficient);
    if (iree_status_is_ok(status)) {
      status = loom_math_evaluation_build_fma(evaluation, &polynomial, residual,
                                              &coefficient, &polynomial);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  loom_math_exponential_value_t square;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_MULF, residual, residual,
      &square));
  return loom_math_evaluation_build_fma(evaluation, &square, &polynomial,
                                        residual, out_value);
}

iree_status_t loom_math_build_exponential(
    loom_target_math_op_t math_op, loom_math_evaluation_t* evaluation,
    loom_math_exponential_value_t* out_value) {
  // Source permissions do not apply to the range-reduction intermediates.
  // Reassociation would erase the split correction and small differences.
  loom_math_evaluation_t arithmetic = *evaluation;
  arithmetic.fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_NONE;
  arithmetic.recipe_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_NONE;
  const bool wide =
      loom_type_element_type(arithmetic.value_type) == LOOM_SCALAR_TYPE_F64;
  const bool base_two = math_op == LOOM_TARGET_MATH_OP_EXP2F;
  const bool difference = math_op == LOOM_TARGET_MATH_OP_EXPM1F;
  const double lower = difference ? (wide ? -40.0 : -20.0)
                       : base_two ? (wide ? -1076.0 : -151.0)
                                  : (wide ? -746.0 : -105.0);
  const double upper =
      base_two ? (wide ? 1024.0 : 128.0) : (wide ? 710.0 : 89.0);
  loom_math_exponential_value_t input;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_clamp(
      &arithmetic, &arithmetic.input, lower, upper, &input));
  loom_math_exponential_value_t scaled = input;
  if (!base_two) {
    loom_math_exponential_value_t log2_e;
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_constant(
        &arithmetic, 0x1.71547652b82fep0, &log2_e));
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
        &arithmetic, LOOM_MATH_EVALUATION_BINARY_MULF, &input, &log2_e,
        &scaled));
  }
  loom_math_exponential_value_t exponent;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_unary(
      &arithmetic, LOOM_MATH_EVALUATION_UNARY_ROUNDEVENF, &scaled, &exponent));
  loom_math_exponential_value_t residual, high, low;
  if (base_two) {
    loom_math_exponential_value_t fraction, product;
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
        &arithmetic, LOOM_MATH_EVALUATION_BINARY_SUBF, &input, &exponent,
        &fraction));
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_constant(
        &arithmetic, 0x1.62e42fefa39efp-1, &high));
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_constant(
        &arithmetic, 0x1.abc9e3b39803fp-56, &low));
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
        &arithmetic, LOOM_MATH_EVALUATION_BINARY_MULF, &fraction, &high,
        &product));
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_fma(
        &arithmetic, &fraction, &low, &product, &residual));
  } else {
    // The leading product is exact throughout the finite reconstruction range.
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_constant(
        &arithmetic, wide ? -0x1.62e42fep-1 : -0x1.62e4p-1, &high));
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_constant(
        &arithmetic, wide ? -0x1.f473de6af278fp-30 : -0x1.7f7d1cp-20, &low));
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_fma(
        &arithmetic, &exponent, &high, &input, &residual));
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_fma(
        &arithmetic, &exponent, &low, &residual, &residual));
  }
  loom_math_exponential_value_t reduced;
  IREE_RETURN_IF_ERROR(
      loom_math_exponential_reduced(&arithmetic, &residual, &reduced));

  // Both factors are normal, even when the result is subnormal or the full
  // scale would overflow. Only the final multiplication rounds a subnormal.
  loom_math_exponential_value_t main_exponent, tail_exponent, scale, tail;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_clamp(
      &arithmetic, &exponent, wide ? -1022.0 : -126.0, wide ? 1023.0 : 127.0,
      &main_exponent));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      &arithmetic, LOOM_MATH_EVALUATION_BINARY_SUBF, &exponent, &main_exponent,
      &tail_exponent));
  IREE_RETURN_IF_ERROR(
      loom_math_exponential_scale(&arithmetic, &main_exponent, &scale));
  IREE_RETURN_IF_ERROR(
      loom_math_exponential_scale(&arithmetic, &tail_exponent, &tail));
  loom_math_exponential_value_t addend = scale;
  if (difference) {
    // expm1's lower clamp excludes subnormal scales. The tail is one except
    // at the largest exponent, where it is two. Subtract its reciprocal before
    // scaling: (scale*(1+expm1(r))-1/tail)*tail.
    loom_math_exponential_value_t one, half, offset;
    IREE_RETURN_IF_ERROR(
        loom_math_evaluation_build_constant(&arithmetic, 1.0, &one));
    IREE_RETURN_IF_ERROR(
        loom_math_evaluation_build_constant(&arithmetic, 0.5, &half));
    loom_value_id_t whole_scale;
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_ordered_equal(
        &arithmetic, &exponent, &main_exponent, &whole_scale));
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_select(
        &arithmetic, whole_scale, &one, &half, &offset));
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
        &arithmetic, LOOM_MATH_EVALUATION_BINARY_SUBF, &scale, &offset,
        &addend));
  }
  loom_math_exponential_value_t result;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_fma(
      &arithmetic, &scale, &reduced, &addend, &result));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      &arithmetic, LOOM_MATH_EVALUATION_BINARY_MULF, &result, &tail, &result));
  if (difference) {
    // In this interval expm1 rounds to the input. Selecting the source value
    // also preserves subnormal bits and negative zero under flush-to-zero.
    loom_math_exponential_value_t magnitude, threshold;
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_unary(
        &arithmetic, LOOM_MATH_EVALUATION_UNARY_ABSF, &arithmetic.input,
        &magnitude));
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_constant(
        &arithmetic, wide ? 0x1p-54 : 0x1p-25, &threshold));
    loom_value_id_t tiny;
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_ordered_greater_equal(
        &arithmetic, &threshold, &magnitude, &tiny));
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_select(
        &arithmetic, tiny, &arithmetic.input, &result, &result));
  }
  *out_value = result;
  return iree_ok_status();
}
