// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/math/trigonometry.h"

#include <math.h>

typedef loom_math_evaluation_value_t loom_math_value_t;

iree_status_t loom_math_build_preserved_trigonometric(
    loom_target_math_op_t math_op, loom_math_evaluation_t* evaluation,
    loom_math_value_t* out_value) {
  const bool is_cosine = math_op == LOOM_TARGET_MATH_OP_COSF ||
                         math_op == LOOM_TARGET_MATH_OP_COSTURNSF;
  const bool is_turns = math_op == LOOM_TARGET_MATH_OP_SINTURNSF ||
                        math_op == LOOM_TARGET_MATH_OP_COSTURNSF;
  const loom_math_evaluation_unary_op_t native_op =
      is_turns ? (is_cosine ? LOOM_MATH_EVALUATION_UNARY_COSTURNSF
                            : LOOM_MATH_EVALUATION_UNARY_SINTURNSF)
               : (is_cosine ? LOOM_MATH_EVALUATION_UNARY_COSF
                            : LOOM_MATH_EVALUATION_UNARY_SINF);
  loom_math_value_t result;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_unary(
      evaluation, native_op, &evaluation->input, &result));
  loom_math_value_t zero;
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(evaluation, 0.0, &zero));
  loom_value_id_t is_zero;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_ordered_equal(
      evaluation, &result, &zero, &is_zero));
  loom_math_value_t signed_zero = zero;
  if (!is_cosine) {
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
        evaluation, LOOM_MATH_EVALUATION_BINARY_COPYSIGNF, &zero,
        &evaluation->input, &signed_zero));
  }
  return loom_math_evaluation_build_select(evaluation, is_zero, &signed_zero,
                                           &result, out_value);
}

static iree_status_t loom_math_trigonometric_select_cardinal(
    loom_math_evaluation_t* evaluation, const loom_math_value_t* magnitude,
    double cardinal, const loom_math_value_t* cardinal_value,
    const loom_math_value_t* ordinary_value, loom_math_value_t* out_value) {
  loom_math_value_t bound;
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(evaluation, cardinal, &bound));
  loom_value_id_t condition;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_ordered_equal(
      evaluation, magnitude, &bound, &condition));
  return loom_math_evaluation_build_select(
      evaluation, condition, cardinal_value, ordinary_value, out_value);
}

iree_status_t loom_math_build_trigonometric(loom_target_math_op_t math_op,
                                            loom_math_evaluation_t* evaluation,
                                            loom_math_value_t* out_value) {
  const bool is_cosine = math_op == LOOM_TARGET_MATH_OP_COSF ||
                         math_op == LOOM_TARGET_MATH_OP_COSTURNSF;
  const loom_math_evaluation_unary_op_t native_op =
      is_cosine ? LOOM_MATH_EVALUATION_UNARY_COSF
                : LOOM_MATH_EVALUATION_UNARY_SINF;
  if (math_op == LOOM_TARGET_MATH_OP_SINF ||
      math_op == LOOM_TARGET_MATH_OP_COSF) {
    return loom_math_evaluation_build_unary(evaluation, native_op,
                                            &evaluation->input, out_value);
  }

  // Subtract the nearest integer before scaling. The residual lies in
  // [-0.5, 0.5], and large finite inputs with no fraction reduce exactly to
  // zero. Scaling the original input would lose phase or overflow.
  loom_math_value_t integral;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_unary(
      evaluation, LOOM_MATH_EVALUATION_UNARY_ROUNDEVENF, &evaluation->input,
      &integral));
  loom_math_value_t residual;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_SUBF, &evaluation->input,
      &integral, &residual));
  loom_math_value_t two_pi;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_constant(
      evaluation, 0x1.921fb54442d18p+2, &two_pi));
  loom_math_value_t angle;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
      evaluation, LOOM_MATH_EVALUATION_BINARY_MULF, &residual, &two_pi,
      &angle));
  loom_math_value_t result;
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_unary(evaluation, native_op, &angle, &result));

  loom_math_value_t magnitude;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_unary(
      evaluation, LOOM_MATH_EVALUATION_UNARY_ABSF, &residual, &magnitude));
  loom_math_value_t zero;
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(evaluation, 0.0, &zero));
  loom_math_value_t unit;
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(evaluation, 1.0, &unit));
  if (is_cosine) {
    loom_math_value_t negative_unit;
    IREE_RETURN_IF_ERROR(
        loom_math_evaluation_build_constant(evaluation, -1.0, &negative_unit));
    IREE_RETURN_IF_ERROR(loom_math_trigonometric_select_cardinal(
        evaluation, &magnitude, 0.0, &unit, &result, &result));
    IREE_RETURN_IF_ERROR(loom_math_trigonometric_select_cardinal(
        evaluation, &magnitude, 0.5, &negative_unit, &result, &result));
    IREE_RETURN_IF_ERROR(loom_math_trigonometric_select_cardinal(
        evaluation, &magnitude, 0.25, &zero, &result, &result));
  } else {
    loom_math_value_t signed_zero;
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
        evaluation, LOOM_MATH_EVALUATION_BINARY_COPYSIGNF, &zero,
        &evaluation->input, &signed_zero));
    loom_math_value_t signed_unit;
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_binary(
        evaluation, LOOM_MATH_EVALUATION_BINARY_COPYSIGNF, &unit, &residual,
        &signed_unit));
    IREE_RETURN_IF_ERROR(loom_math_trigonometric_select_cardinal(
        evaluation, &magnitude, 0.0, &signed_zero, &result, &result));
    IREE_RETURN_IF_ERROR(loom_math_trigonometric_select_cardinal(
        evaluation, &magnitude, 0.5, &signed_zero, &result, &result));
    IREE_RETURN_IF_ERROR(loom_math_trigonometric_select_cardinal(
        evaluation, &magnitude, 0.25, &signed_unit, &result, &result));
  }

  // Nonfinite inputs produce an unordered residual. Preserve the source NaN
  // result independently of the native approximation's exceptional behavior.
  loom_value_id_t ordered;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_ordered_equal(
      evaluation, &residual, &residual, &ordered));
  loom_math_value_t nan_value;
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(evaluation, NAN, &nan_value));
  return loom_math_evaluation_build_select(evaluation, ordered, &result,
                                           &nan_value, out_value);
}
