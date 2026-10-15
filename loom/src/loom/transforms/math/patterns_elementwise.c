// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/float_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/math/evaluation.h"
#include "loom/transforms/math/patterns.h"
#include "loom/transforms/math/recipe_program.h"

typedef iree_status_t (*loom_math_legalize_binary_build_fn_t)(
    loom_builder_t* builder, uint8_t instance_flags, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);

typedef iree_status_t (*loom_math_legalize_cast_build_fn_t)(
    loom_builder_t* builder, uint8_t instance_flags, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

static double loom_math_legalize_gelu_logistic_scale(const loom_op_t* op) {
  return loom_scalar_geluf_isa(op) ? loom_scalar_geluf_scale(op)
                                   : loom_vector_geluf_scale(op);
}

typedef loom_math_evaluation_t loom_math_legalize_source_t;
typedef loom_math_evaluation_value_t loom_math_legalize_value_t;

static iree_status_t loom_math_legalize_build_constant(
    loom_builder_t* builder, loom_math_legalize_source_t* source, double value,
    loom_math_legalize_value_t* out_value) {
  (void)builder;
  return loom_math_evaluation_build_constant(source, value, out_value);
}

static iree_status_t loom_math_legalize_build_unary(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_evaluation_unary_op_t unary_op,
    const loom_math_legalize_value_t* input,
    loom_math_legalize_value_t* out_value) {
  (void)builder;
  return loom_math_evaluation_build_unary(source, unary_op, input, out_value);
}

static iree_status_t loom_math_legalize_build_binary(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_evaluation_binary_op_t binary_op, loom_math_legalize_value_t* lhs,
    loom_math_legalize_value_t* rhs, loom_math_legalize_value_t* out_value) {
  (void)builder;
  return loom_math_evaluation_build_binary(source, binary_op, lhs, rhs,
                                           out_value);
}

static iree_status_t loom_math_legalize_build_cast(
    loom_builder_t* builder, loom_math_legalize_cast_build_fn_t build,
    loom_value_id_t input, loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(build(builder, /*instance_flags=*/0, input, input_type,
                             result_type, location, &op));
  *out_value = loom_op_results(op)[0];
  return iree_ok_status();
}

static iree_status_t loom_math_legalize_build_division(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* lhs, loom_math_legalize_value_t* rhs,
    loom_math_legalize_value_t* out_value) {
  return loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_DIVF, lhs, rhs, out_value);
}

static iree_status_t loom_math_legalize_build_rsqrt_sqrt_div(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* out_value) {
  loom_math_legalize_value_t one = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t root = loom_math_evaluation_value_invalid();
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 1.0, &one));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, LOOM_MATH_EVALUATION_UNARY_SQRTF, &source->input,
      &root));
  return loom_math_legalize_build_division(builder, source, &one, &root,
                                           out_value);
}

static iree_status_t loom_math_legalize_build_ternary(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* a, loom_math_legalize_value_t* b,
    const loom_math_legalize_value_t* c,
    loom_math_legalize_value_t* out_value) {
  (void)builder;
  return loom_math_evaluation_build_fma(source, a, b, c, out_value);
}

static iree_status_t loom_math_legalize_build_polynomial(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* input, const double* coefficients,
    iree_host_size_t coefficient_count, loom_math_legalize_value_t* out_value) {
  loom_math_legalize_value_t accumulator = loom_math_evaluation_value_invalid();
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, coefficients[0], &accumulator));
  for (iree_host_size_t i = 1; i < coefficient_count; ++i) {
    loom_math_legalize_value_t coefficient =
        loom_math_evaluation_value_invalid();
    IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
        builder, source, coefficients[i], &coefficient));
    IREE_RETURN_IF_ERROR(loom_math_legalize_build_ternary(
        builder, source, &accumulator, input, &coefficient, &accumulator));
  }
  *out_value = accumulator;
  return iree_ok_status();
}

static iree_status_t loom_math_legalize_build_clampf(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* value, double lower, double upper,
    loom_math_legalize_value_t* out_value) {
  (void)builder;
  return loom_math_evaluation_build_clamp(source, value, lower, upper,
                                          out_value);
}

static iree_status_t loom_math_legalize_build_erf_rational(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* input, loom_math_legalize_value_t* out_value) {
  static const double kAlphaCoefficients[] = {
      -2.72614225801306e-10, 2.77068142495902e-08,  -2.10102402082508e-06,
      -5.69250639462346e-05, -7.34990630326855e-04, -2.95459980854025e-03,
      -1.60960333262415e-02,
  };
  static const double kBetaCoefficients[] = {
      -1.45660718464996e-05, -2.13374055278905e-04, -1.68282697438203e-03,
      -7.37332916720468e-03, -1.42647390514189e-02,
  };

  loom_math_legalize_value_t clamped_input =
      loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t input_squared =
      loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t alpha = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t beta = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t numerator = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t quotient = loom_math_evaluation_value_invalid();
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_clampf(
      builder, source, input, -4.0, 4.0, &clamped_input));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_MULF, &clamped_input,
      &clamped_input, &input_squared));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_polynomial(
      builder, source, &input_squared, kAlphaCoefficients,
      IREE_ARRAYSIZE(kAlphaCoefficients), &alpha));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_polynomial(
      builder, source, &input_squared, kBetaCoefficients,
      IREE_ARRAYSIZE(kBetaCoefficients), &beta));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_MULF, &clamped_input, &alpha,
      &numerator));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_division(
      builder, source, &numerator, &beta, &quotient));
  return loom_math_legalize_build_clampf(builder, source, &quotient, -1.0, 1.0,
                                         out_value);
}

static iree_status_t loom_math_legalize_build_exp_exp2(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* out_value) {
  loom_math_legalize_value_t log2_e = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t scaled = loom_math_evaluation_value_invalid();
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, 1.44269504088896340736, &log2_e));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_MULF, &source->input,
      &log2_e, &scaled));
  return loom_math_legalize_build_unary(
      builder, source, LOOM_MATH_EVALUATION_UNARY_EXP2F, &scaled, out_value);
}

static iree_status_t loom_math_legalize_build_log_log2(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* out_value) {
  loom_math_legalize_value_t ln2 = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t log2_input = loom_math_evaluation_value_invalid();
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, 0.69314718055994530942, &ln2));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, LOOM_MATH_EVALUATION_UNARY_LOG2F, &source->input,
      &log2_input));
  return loom_math_legalize_build_binary(builder, source,
                                         LOOM_MATH_EVALUATION_BINARY_MULF,
                                         &log2_input, &ln2, out_value);
}

static bool loom_math_legalize_value_matches_float_constant(
    const loom_math_legalize_source_t* source,
    const loom_math_legalize_value_t* value, double expected) {
  const loom_value_id_t value_id = loom_math_evaluation_value_id(value);
  loom_value_facts_t facts =
      loom_rewriter_value_facts(source->rewriter, value_id);
  if (source->rewriter->fact_table != NULL) {
    loom_value_fact_uniform_element_t uniform_element = {0};
    if (loom_value_facts_query_uniform_element(
            &source->rewriter->fact_table->context, facts, &uniform_element)) {
      facts = uniform_element.element;
    }
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(
      loom_module_value_type(source->rewriter->module, value_id));
  double actual = 0.0;
  if (!loom_value_facts_as_exact_float(scalar_type, facts, &actual)) {
    return false;
  }
  loom_value_facts_t expected_facts =
      loom_value_facts_exact_float(scalar_type, expected);
  double rounded_expected = 0.0;
  return loom_value_facts_as_exact_float(scalar_type, expected_facts,
                                         &rounded_expected) &&
         actual == rounded_expected;
}

static bool loom_math_legalize_try_project_turns_input(
    const loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* out_input) {
  const loom_value_t* input_value = loom_module_value(
      source->rewriter->module, loom_math_evaluation_value_id(&source->input));
  if (loom_value_is_block_arg(input_value)) {
    return false;
  }
  const loom_op_t* input_op = loom_value_def_op(input_value);
  if (input_op == NULL) {
    return false;
  }

  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  switch (input_op->kind) {
    case LOOM_OP_SCALAR_MULF:
      lhs = loom_scalar_mulf_lhs(input_op);
      rhs = loom_scalar_mulf_rhs(input_op);
      break;
    case LOOM_OP_VECTOR_MULF:
      lhs = loom_vector_mulf_lhs(input_op);
      rhs = loom_vector_mulf_rhs(input_op);
      break;
    default:
      return false;
  }

  const double two_pi = 6.28318530717958647692;
  loom_math_legalize_value_t lhs_value =
      loom_math_evaluation_value_from_id(lhs);
  loom_math_legalize_value_t rhs_value =
      loom_math_evaluation_value_from_id(rhs);
  if (loom_math_legalize_value_matches_float_constant(source, &rhs_value,
                                                      two_pi)) {
    *out_input = lhs_value;
    return true;
  }
  if (loom_math_legalize_value_matches_float_constant(source, &lhs_value,
                                                      two_pi)) {
    *out_input = rhs_value;
    return true;
  }
  return false;
}

static iree_status_t loom_math_legalize_build_turns_input(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* out_value) {
  if (loom_math_legalize_try_project_turns_input(source, out_value)) {
    return iree_ok_status();
  }

  loom_math_legalize_value_t inverse_two_pi =
      loom_math_evaluation_value_invalid();
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, 0.15915494309189533577, &inverse_two_pi));
  return loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_MULF, &source->input,
      &inverse_two_pi, out_value);
}

static iree_status_t loom_math_legalize_build_sin_turns(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* out_value) {
  loom_math_legalize_value_t turns_input = loom_math_evaluation_value_invalid();
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_turns_input(builder, source, &turns_input));
  return loom_math_legalize_build_unary(builder, source,
                                        LOOM_MATH_EVALUATION_UNARY_SINTURNSF,
                                        &turns_input, out_value);
}

static iree_status_t loom_math_legalize_build_cos_turns(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* out_value) {
  loom_math_legalize_value_t turns_input = loom_math_evaluation_value_invalid();
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_turns_input(builder, source, &turns_input));
  return loom_math_legalize_build_unary(builder, source,
                                        LOOM_MATH_EVALUATION_UNARY_COSTURNSF,
                                        &turns_input, out_value);
}

static iree_status_t loom_math_legalize_build_logistic_exp2(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* out_value) {
  loom_math_legalize_value_t negative_log2_e =
      loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t one = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t scaled = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t exponent = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t denominator = loom_math_evaluation_value_invalid();
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, -1.44269504088896340736, &negative_log2_e));
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 1.0, &one));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_MULF, &source->input,
      &negative_log2_e, &scaled));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, LOOM_MATH_EVALUATION_UNARY_EXP2F, &scaled, &exponent));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_ADDF, &one, &exponent,
      &denominator));
  return loom_math_legalize_build_division(builder, source, &one, &denominator,
                                           out_value);
}

static iree_status_t loom_math_legalize_build_tanh_logistic(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* input, loom_math_legalize_value_t* out_value) {
  loom_math_legalize_value_t negative_one =
      loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t two = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t scaled = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t logistic = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t doubled_logistic =
      loom_math_evaluation_value_invalid();
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, -1.0, &negative_one));
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 2.0, &two));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_MULF, input, &two, &scaled));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, LOOM_MATH_EVALUATION_UNARY_LOGISTICF, &scaled,
      &logistic));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_MULF, &logistic, &two,
      &doubled_logistic));
  return loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_ADDF, &doubled_logistic,
      &negative_one, out_value);
}

static iree_status_t loom_math_legalize_build_logistic_tanh(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* out_value) {
  loom_math_legalize_value_t half = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t one = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t scaled = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t tangent = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t shifted = loom_math_evaluation_value_invalid();
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 0.5, &half));
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 1.0, &one));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_MULF, &source->input, &half,
      &scaled));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, LOOM_MATH_EVALUATION_UNARY_TANHF, &scaled, &tangent));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_ADDF, &one, &tangent,
      &shifted));
  return loom_math_legalize_build_binary(builder, source,
                                         LOOM_MATH_EVALUATION_BINARY_MULF,
                                         &half, &shifted, out_value);
}

static iree_status_t loom_math_legalize_build_pow_log2_exp2(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* out_value) {
  loom_math_legalize_value_t log2_base = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t exponent = loom_math_evaluation_value_invalid();
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, LOOM_MATH_EVALUATION_UNARY_LOG2F, &source->input,
      &log2_base));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_MULF, &log2_base,
      &source->secondary_input, &exponent));
  return loom_math_legalize_build_unary(
      builder, source, LOOM_MATH_EVALUATION_UNARY_EXP2F, &exponent, out_value);
}

static iree_status_t loom_math_legalize_build_silu_logistic(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* out_value) {
  loom_math_legalize_value_t logistic = loom_math_evaluation_value_invalid();
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, LOOM_MATH_EVALUATION_UNARY_LOGISTICF, &source->input,
      &logistic));
  return loom_math_legalize_build_binary(builder, source,
                                         LOOM_MATH_EVALUATION_BINARY_MULF,
                                         &source->input, &logistic, out_value);
}

static iree_status_t loom_math_legalize_build_softplus_exp2(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* out_value) {
  // softplus(x) = max(x, 0) + log1p(exp(-abs(x))). The exponential cannot
  // overflow, and its argument may become -infinity for a finite input.
  // Reassociation would erase the rounding correction in (1 + t) - 1.
  source->fastmath_flags &= ~(LOOM_TARGET_MATH_FASTMATH_FLAG_REASSOC |
                              LOOM_TARGET_MATH_FASTMATH_FLAG_NINF);
  loom_math_legalize_value_t negative_log2_e =
      loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t ln2 = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t zero = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t one = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t next_one = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t magnitude = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t scaled = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t exponent = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t sum = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t increment = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t safe_increment =
      loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t correction = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t log2_sum = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t log_sum = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t corrected = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t tail = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t positive = loom_math_evaluation_value_invalid();
  loom_value_id_t has_increment = LOOM_VALUE_ID_INVALID;
  loom_value_id_t is_nonnegative = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, -1.44269504088896340736, &negative_log2_e));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, 0.69314718055994530942, &ln2));
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 0.0, &zero));
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 1.0, &one));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, 1.00000011920928955078125, &next_one));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, LOOM_MATH_EVALUATION_UNARY_ABSF, &source->input,
      &magnitude));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_MULF, &magnitude,
      &negative_log2_e, &scaled));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, LOOM_MATH_EVALUATION_UNARY_EXP2F, &scaled, &exponent));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_ADDF, &one, &exponent,
      &sum));
  // For t in [0, 1], u = round(1 + t) is in [1, 2], so u - 1 is exact.
  // log(u) * t / (u - 1) corrects the rounded increment. If u rounds to 1,
  // log1p(t) rounds to t. Select a nonzero denominator before division so
  // this path never evaluates 0/0, including at either input infinity.
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_SUBF, &sum, &one,
      &increment));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_ordered_greater_equal(
      source, &sum, &next_one, &has_increment));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_select(
      source, has_increment, &increment, &one, &safe_increment));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_division(
      builder, source, &exponent, &safe_increment, &correction));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, LOOM_MATH_EVALUATION_UNARY_LOG2F, &sum, &log2_sum));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_MULF, &log2_sum, &ln2,
      &log_sum));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_MULF, &log_sum, &correction,
      &corrected));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_select(
      source, has_increment, &corrected, &exponent, &tail));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_ordered_greater_equal(
      source, &source->input, &zero, &is_nonnegative));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_select(
      source, is_nonnegative, &source->input, &zero, &positive));
  return loom_math_legalize_build_binary(builder, source,
                                         LOOM_MATH_EVALUATION_BINARY_ADDF,
                                         &positive, &tail, out_value);
}

static iree_status_t loom_math_legalize_build_gelu_tanh(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* out_value) {
  loom_math_legalize_value_t half = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t one = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t cubic_coefficient =
      loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t sqrt_2_over_pi =
      loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t input_squared =
      loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t input_cubed = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t cubic_term = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t inner_sum = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t scaled = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t tangent = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t one_plus_tanh =
      loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t half_input = loom_math_evaluation_value_invalid();
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 0.5, &half));
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 1.0, &one));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, 0.044715, &cubic_coefficient));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, 0.79788456080286535588, &sqrt_2_over_pi));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_MULF, &source->input,
      &source->input, &input_squared));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_MULF, &input_squared,
      &source->input, &input_cubed));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_MULF, &cubic_coefficient,
      &input_cubed, &cubic_term));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_ADDF, &source->input,
      &cubic_term, &inner_sum));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_MULF, &sqrt_2_over_pi,
      &inner_sum, &scaled));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, LOOM_MATH_EVALUATION_UNARY_TANHF, &scaled, &tangent));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_ADDF, &one, &tangent,
      &one_plus_tanh));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_MULF, &half, &source->input,
      &half_input));
  return loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_MULF, &half_input,
      &one_plus_tanh, out_value);
}

static iree_status_t loom_math_legalize_build_gelu_erf(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* out_value) {
  loom_math_legalize_value_t half = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t one = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t inverse_sqrt2 =
      loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t scaled = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t erf = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t one_plus_erf =
      loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t half_input = loom_math_evaluation_value_invalid();
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 0.5, &half));
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 1.0, &one));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, 0.70710678118654752440, &inverse_sqrt2));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_MULF, &source->input,
      &inverse_sqrt2, &scaled));
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_erf_rational(builder, source, &scaled, &erf));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_ADDF, &one, &erf,
      &one_plus_erf));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_MULF, &half, &source->input,
      &half_input));
  return loom_math_legalize_build_binary(builder, source,
                                         LOOM_MATH_EVALUATION_BINARY_MULF,
                                         &half_input, &one_plus_erf, out_value);
}

typedef struct loom_math_legalize_binary_source_t {
  // Source left-hand operand.
  loom_value_id_t lhs;
  // Source right-hand operand.
  loom_value_id_t rhs;
  // Source result type preserved by the replacement expression.
  loom_type_t result_type;
  // Intermediate f32 scalar or vector type used for arithmetic.
  loom_type_t widened_type;
  // Source fast-math flags forwarded to replacement arithmetic.
  uint8_t fastmath_flags;
  // Source location copied to replacement ops.
  loom_location_id_t location;
  // Builds the widened arithmetic op.
  loom_math_legalize_binary_build_fn_t binary_build;
  // Builds the source-domain extension to f32.
  loom_math_legalize_cast_build_fn_t extf_build;
  // Builds the source-domain truncation from f32.
  loom_math_legalize_cast_build_fn_t fptrunc_build;
} loom_math_legalize_binary_source_t;

static loom_type_t loom_math_legalize_type_with_element(
    loom_type_t source_type, loom_scalar_type_t element_type) {
  loom_type_t result_type = source_type;
  result_type.header = loom_type_make_header(
      loom_type_kind(source_type), element_type, loom_type_rank(source_type),
      loom_type_flags(source_type));
  return result_type;
}

static iree_status_t loom_math_legalize_build_cmpf(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    const loom_math_legalize_value_t* lhs,
    const loom_math_legalize_value_t* rhs, loom_value_id_t* out_value) {
  (void)builder;
  return loom_math_evaluation_build_ordered_greater_equal(source, lhs, rhs,
                                                          out_value);
}

static iree_status_t loom_math_legalize_build_select(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_value_id_t condition, const loom_math_legalize_value_t* true_value,
    const loom_math_legalize_value_t* false_value,
    loom_math_legalize_value_t* out_value) {
  (void)builder;
  return loom_math_evaluation_build_select(source, condition, true_value,
                                           false_value, out_value);
}

static iree_status_t loom_math_legalize_build_round_away(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* out_value) {
  loom_math_legalize_value_t half = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t one = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t truncated = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t remainder = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t magnitude = loom_math_evaluation_value_invalid();
  loom_value_id_t needs_rounding = LOOM_VALUE_ID_INVALID;
  loom_math_legalize_value_t increment = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t rounded = loom_math_evaluation_value_invalid();
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 0.5, &half));
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 1.0, &one));
  // For finite inputs, subtracting the integral part is exact. Adding 0.5 is
  // not: the representable value immediately below 0.5 would round up to 1.
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, LOOM_MATH_EVALUATION_UNARY_TRUNCF, &source->input,
      &truncated));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_SUBF, &source->input,
      &truncated, &remainder));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, LOOM_MATH_EVALUATION_UNARY_ABSF, &remainder,
      &magnitude));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_cmpf(
      builder, source, &magnitude, &half, &needs_rounding));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_COPYSIGNF, &one,
      &source->input, &increment));
  // An integral value, infinity or NaN selects the truncation unchanged.
  // This also preserves the sign of zero without a width-dependent cutoff.
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_ADDF, &truncated, &increment,
      &rounded));
  return loom_math_legalize_build_select(builder, source, needs_rounding,
                                         &rounded, &truncated, out_value);
}

static void loom_math_legalize_binary_source_initialize(
    const loom_math_legalize_recipe_context_t* context, const loom_op_t* op,
    loom_math_legalize_binary_source_t* out_source) {
  *out_source = (loom_math_legalize_binary_source_t){
      .result_type =
          loom_module_value_type(context->module, loom_op_results(op)[0]),
      .location = op->location,
  };
  out_source->widened_type = loom_math_legalize_type_with_element(
      out_source->result_type, LOOM_SCALAR_TYPE_F32);
  if (loom_type_is_scalar(out_source->result_type)) {
    out_source->extf_build = loom_scalar_extf_build;
    out_source->fptrunc_build = loom_scalar_fptrunc_build;
  } else {
    out_source->extf_build = loom_vector_extf_build;
    out_source->fptrunc_build = loom_vector_fptrunc_build;
  }
  switch (op->kind) {
    case LOOM_OP_SCALAR_ADDF:
      out_source->lhs = loom_scalar_addf_lhs(op);
      out_source->rhs = loom_scalar_addf_rhs(op);
      out_source->fastmath_flags = loom_scalar_addf_fastmath(op);
      out_source->binary_build = loom_scalar_addf_build;
      return;
    case LOOM_OP_SCALAR_SUBF:
      out_source->lhs = loom_scalar_subf_lhs(op);
      out_source->rhs = loom_scalar_subf_rhs(op);
      out_source->fastmath_flags = loom_scalar_subf_fastmath(op);
      out_source->binary_build = loom_scalar_subf_build;
      return;
    case LOOM_OP_SCALAR_MULF:
      out_source->lhs = loom_scalar_mulf_lhs(op);
      out_source->rhs = loom_scalar_mulf_rhs(op);
      out_source->fastmath_flags = loom_scalar_mulf_fastmath(op);
      out_source->binary_build = loom_scalar_mulf_build;
      return;
    case LOOM_OP_VECTOR_ADDF:
      out_source->lhs = loom_vector_addf_lhs(op);
      out_source->rhs = loom_vector_addf_rhs(op);
      out_source->fastmath_flags = loom_vector_addf_fastmath(op);
      out_source->binary_build = loom_vector_addf_build;
      return;
    case LOOM_OP_VECTOR_SUBF:
      out_source->lhs = loom_vector_subf_lhs(op);
      out_source->rhs = loom_vector_subf_rhs(op);
      out_source->fastmath_flags = loom_vector_subf_fastmath(op);
      out_source->binary_build = loom_vector_subf_build;
      return;
    case LOOM_OP_VECTOR_MULF:
      out_source->lhs = loom_vector_mulf_lhs(op);
      out_source->rhs = loom_vector_mulf_rhs(op);
      out_source->fastmath_flags = loom_vector_mulf_fastmath(op);
      out_source->binary_build = loom_vector_mulf_build;
      return;
    default:
      break;
  }
  IREE_ASSERT_UNREACHABLE("math recipe selected unsupported binary op");
  IREE_BUILTIN_UNREACHABLE();
}

// Projects through an exact extension already present on a recipe operand.
// The widening recipe owns the replacement expression and should present the
// original source type to target legalization instead of retaining an
// incidental intermediate representation selected by source IR.
static void loom_math_legalize_project_exact_float_extension(
    const loom_module_t* module, loom_value_id_t* value,
    loom_type_t* value_type) {
  const loom_value_t* input_value = loom_module_value(module, *value);
  if (loom_value_is_block_arg(input_value)) {
    return;
  }
  const loom_op_t* input_op = loom_value_def_op(input_value);
  if (input_op == NULL) {
    return;
  }
  if (input_op->instance_flags != 0) {
    return;
  }

  if (loom_type_is_scalar(*value_type) && loom_scalar_extf_isa(input_op)) {
    *value = loom_scalar_extf_input(input_op);
    *value_type = loom_module_value_type(module, *value);
  } else if (loom_type_is_vector(*value_type) &&
             loom_vector_extf_isa(input_op)) {
    *value = loom_vector_extf_input(input_op);
    *value_type = loom_module_value_type(module, *value);
  }
}

// Widens one recipe operand directly from its exact source representation.
// Uniform vectors retain their scalar representation so the target sees one
// scalar conversion followed by a splat instead of a lane-wise conversion.
static iree_status_t loom_math_legalize_build_widen_f32_operand(
    loom_builder_t* builder, const loom_math_legalize_recipe_context_t* context,
    const loom_math_legalize_binary_source_t* source, loom_value_id_t input,
    loom_value_id_t* out_value) {
  loom_type_t input_type = source->result_type;
  const loom_value_t* input_value = loom_module_value(context->module, input);
  const loom_op_t* input_op = loom_value_is_block_arg(input_value)
                                  ? NULL
                                  : loom_value_def_op(input_value);
  if (loom_type_is_vector(input_type) && input_op != NULL &&
      loom_vector_splat_isa(input_op)) {
    loom_value_id_t scalar = loom_vector_splat_scalar(input_op);
    loom_type_t scalar_type = loom_module_value_type(context->module, scalar);
    loom_math_legalize_project_exact_float_extension(context->module, &scalar,
                                                     &scalar_type);

    loom_op_t* widened_scalar_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_extf_build(
        builder, /*instance_flags=*/0, scalar, scalar_type,
        loom_type_scalar(LOOM_SCALAR_TYPE_F32), source->location,
        &widened_scalar_op));
    loom_op_t* splat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_splat_build(
        builder, loom_scalar_extf_result(widened_scalar_op),
        source->widened_type, source->location, &splat_op));
    *out_value = loom_vector_splat_result(splat_op);
    return iree_ok_status();
  }

  loom_math_legalize_project_exact_float_extension(context->module, &input,
                                                   &input_type);
  return loom_math_legalize_build_cast(builder, source->extf_build, input,
                                       input_type, source->widened_type,
                                       source->location, out_value);
}

// Addition, subtraction and multiplication of FP8, f16 and bf16 operands round
// correctly through IEEE f32 arithmetic with gradual underflow.
// Other arithmetic requires its own proof against intermediate rounding.
static iree_status_t loom_math_legalize_build_widen_f32_round(
    loom_builder_t* builder, const loom_math_legalize_recipe_context_t* context,
    const loom_op_t* op, loom_value_id_t* out_value) {
  loom_math_legalize_binary_source_t source;
  loom_math_legalize_binary_source_initialize(context, op, &source);

  loom_value_id_t wide_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_widen_f32_operand(
      builder, context, &source, source.lhs, &wide_lhs));
  loom_value_id_t wide_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_widen_f32_operand(
      builder, context, &source, source.rhs, &wide_rhs));

  loom_op_t* wide_op = NULL;
  IREE_RETURN_IF_ERROR(
      source.binary_build(builder, source.fastmath_flags, wide_lhs, wide_rhs,
                          source.widened_type, source.location, &wide_op));
  const loom_value_id_t wide_result = loom_op_results(wide_op)[0];

  return loom_math_legalize_build_cast(
      builder, source.fptrunc_build, wide_result, source.widened_type,
      source.result_type, source.location, out_value);
}

static iree_status_t loom_math_legalize_build_gelu_logistic(
    loom_builder_t* builder, loom_math_legalize_source_t* source,
    const loom_op_t* op, loom_math_legalize_value_t* out_value) {
  loom_math_legalize_value_t scale = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t scaled = loom_math_evaluation_value_invalid();
  loom_math_legalize_value_t logistic = loom_math_evaluation_value_invalid();
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, loom_math_legalize_gelu_logistic_scale(op), &scale));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, LOOM_MATH_EVALUATION_BINARY_MULF, &scale, &source->input,
      &scaled));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, LOOM_MATH_EVALUATION_UNARY_LOGISTICF, &scaled,
      &logistic));
  return loom_math_legalize_build_binary(builder, source,
                                         LOOM_MATH_EVALUATION_BINARY_MULF,
                                         &source->input, &logistic, out_value);
}

static iree_status_t loom_math_legalize_build_evaluated_recipe(
    const loom_math_legalize_recipe_context_t* context, loom_op_t* op,
    loom_rewriter_t* rewriter, loom_math_legalize_source_t* source,
    loom_math_legalize_value_t* out_value) {
  switch (context->decision.recipe) {
    case LOOM_TARGET_MATH_RECIPE_EXP_EXP2_F32:
      return loom_math_legalize_build_exp_exp2(&rewriter->builder, source,
                                               out_value);
    case LOOM_TARGET_MATH_RECIPE_LOG_LOG2_F32:
      return loom_math_legalize_build_log_log2(&rewriter->builder, source,
                                               out_value);
    case LOOM_TARGET_MATH_RECIPE_TANH_LOGISTIC_F32:
      return loom_math_legalize_build_tanh_logistic(&rewriter->builder, source,
                                                    &source->input, out_value);
    case LOOM_TARGET_MATH_RECIPE_LOGISTIC_TANH_F32:
      return loom_math_legalize_build_logistic_tanh(&rewriter->builder, source,
                                                    out_value);
    case LOOM_TARGET_MATH_RECIPE_POW_LOG2_EXP2_F32:
      return loom_math_legalize_build_pow_log2_exp2(&rewriter->builder, source,
                                                    out_value);
    case LOOM_TARGET_MATH_RECIPE_ROUND_AWAY:
      return loom_math_legalize_build_round_away(&rewriter->builder, source,
                                                 out_value);
    case LOOM_TARGET_MATH_RECIPE_SIN_TURNS_F32:
      return loom_math_legalize_build_sin_turns(&rewriter->builder, source,
                                                out_value);
    case LOOM_TARGET_MATH_RECIPE_COS_TURNS_F32:
      return loom_math_legalize_build_cos_turns(&rewriter->builder, source,
                                                out_value);
    case LOOM_TARGET_MATH_RECIPE_ERF_RATIONAL_F32:
      return loom_math_legalize_build_erf_rational(&rewriter->builder, source,
                                                   &source->input, out_value);
    case LOOM_TARGET_MATH_RECIPE_LOGISTIC_EXP2_F32:
      return loom_math_legalize_build_logistic_exp2(&rewriter->builder, source,
                                                    out_value);
    case LOOM_TARGET_MATH_RECIPE_SILU_LOGISTIC_F32:
      return loom_math_legalize_build_silu_logistic(&rewriter->builder, source,
                                                    out_value);
    case LOOM_TARGET_MATH_RECIPE_SOFTPLUS_EXP2_F32:
      return loom_math_legalize_build_softplus_exp2(&rewriter->builder, source,
                                                    out_value);
    case LOOM_TARGET_MATH_RECIPE_GELU_TANH_F32:
      return loom_math_legalize_build_gelu_tanh(&rewriter->builder, source,
                                                out_value);
    case LOOM_TARGET_MATH_RECIPE_GELU_ERF_F32:
      return loom_math_legalize_build_gelu_erf(&rewriter->builder, source,
                                               out_value);
    case LOOM_TARGET_MATH_RECIPE_GELU_LOGISTIC_F32:
      return loom_math_legalize_build_gelu_logistic(&rewriter->builder, source,
                                                    op, out_value);
    case LOOM_TARGET_MATH_RECIPE_RSQRT_SQRT_DIV:
      return loom_math_legalize_build_rsqrt_sqrt_div(&rewriter->builder, source,
                                                     out_value);
    case LOOM_TARGET_MATH_RECIPE_APPROXIMATE_EXP2_F32:
      return loom_math_recipe_program_evaluate(
          source, LOOM_MATH_RECIPE_PROGRAM_EXP2_F32, &source->input, out_value);
    case LOOM_TARGET_MATH_RECIPE_APPROXIMATE_SIN_TURNS_F32:
      return loom_math_recipe_program_evaluate(
          source, LOOM_MATH_RECIPE_PROGRAM_SIN_TURNS_F32, &source->input,
          out_value);
    case LOOM_TARGET_MATH_RECIPE_APPROXIMATE_COS_TURNS_F32:
      return loom_math_recipe_program_evaluate(
          source, LOOM_MATH_RECIPE_PROGRAM_COS_TURNS_F32, &source->input,
          out_value);
    case LOOM_TARGET_MATH_RECIPE_WIDEN_F32_ROUND:
      IREE_ASSERT_UNREACHABLE("recipe is not elementwise math legalization");
      IREE_BUILTIN_UNREACHABLE();
    case LOOM_TARGET_MATH_RECIPE_UNKNOWN:
      break;
  }
  IREE_ASSERT_UNREACHABLE("unknown math recipe");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_math_legalize_build_recipe(
    const loom_math_legalize_recipe_context_t* context, loom_op_t* op,
    loom_rewriter_t* rewriter, loom_value_id_t* out_value) {
  if (context->decision.recipe == LOOM_TARGET_MATH_RECIPE_WIDEN_F32_ROUND) {
    return loom_math_legalize_build_widen_f32_round(&rewriter->builder, context,
                                                    op, out_value);
  }

  loom_math_legalize_source_t source;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_initialize(
      context->module, &context->query, &context->decision, op, rewriter,
      &source));
  loom_math_legalize_value_t value = loom_math_evaluation_value_invalid();
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_evaluated_recipe(
      context, op, rewriter, &source, &value));
  return loom_math_evaluation_export(&source, &value, out_value);
}

static iree_status_t loom_math_legalize_rewrite_math_op(
    const loom_math_legalize_recipe_context_t* context, loom_op_t* op,
    loom_rewriter_t* rewriter) {
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_recipe(context, op, rewriter, &replacement));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static bool loom_math_legalize_elementwise_recipe_is_supported(
    loom_target_math_recipe_t recipe) {
  switch (recipe) {
    case LOOM_TARGET_MATH_RECIPE_EXP_EXP2_F32:
    case LOOM_TARGET_MATH_RECIPE_LOG_LOG2_F32:
    case LOOM_TARGET_MATH_RECIPE_SIN_TURNS_F32:
    case LOOM_TARGET_MATH_RECIPE_COS_TURNS_F32:
    case LOOM_TARGET_MATH_RECIPE_TANH_LOGISTIC_F32:
    case LOOM_TARGET_MATH_RECIPE_LOGISTIC_TANH_F32:
    case LOOM_TARGET_MATH_RECIPE_POW_LOG2_EXP2_F32:
    case LOOM_TARGET_MATH_RECIPE_ROUND_AWAY:
    case LOOM_TARGET_MATH_RECIPE_ERF_RATIONAL_F32:
    case LOOM_TARGET_MATH_RECIPE_LOGISTIC_EXP2_F32:
    case LOOM_TARGET_MATH_RECIPE_SILU_LOGISTIC_F32:
    case LOOM_TARGET_MATH_RECIPE_SOFTPLUS_EXP2_F32:
    case LOOM_TARGET_MATH_RECIPE_GELU_TANH_F32:
    case LOOM_TARGET_MATH_RECIPE_GELU_ERF_F32:
    case LOOM_TARGET_MATH_RECIPE_GELU_LOGISTIC_F32:
    case LOOM_TARGET_MATH_RECIPE_RSQRT_SQRT_DIV:
    case LOOM_TARGET_MATH_RECIPE_APPROXIMATE_EXP2_F32:
    case LOOM_TARGET_MATH_RECIPE_APPROXIMATE_SIN_TURNS_F32:
    case LOOM_TARGET_MATH_RECIPE_APPROXIMATE_COS_TURNS_F32:
    case LOOM_TARGET_MATH_RECIPE_WIDEN_F32_ROUND:
      return true;
    case LOOM_TARGET_MATH_RECIPE_UNKNOWN:
      return false;
  }
  return false;
}

iree_status_t loom_math_legalize_rewrite_elementwise_recipe(
    const loom_math_legalize_recipe_context_t* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_rewritten) {
  if (!loom_math_legalize_elementwise_recipe_is_supported(
          context->decision.recipe)) {
    return iree_ok_status();
  }
  *out_rewritten = true;
  return loom_math_legalize_rewrite_math_op(context, op, rewriter);
}
