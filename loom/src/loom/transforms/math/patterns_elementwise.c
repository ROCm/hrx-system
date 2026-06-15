// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <math.h>

#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/math/patterns.h"

typedef iree_status_t (*loom_math_legalize_constant_build_fn_t)(
    loom_builder_t* builder, loom_attribute_t value, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

typedef iree_status_t (*loom_math_legalize_unary_build_fn_t)(
    loom_builder_t* builder, uint8_t instance_flags, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op);

typedef iree_status_t (*loom_math_legalize_binary_build_fn_t)(
    loom_builder_t* builder, uint8_t instance_flags, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);

typedef iree_status_t (*loom_math_legalize_ternary_build_fn_t)(
    loom_builder_t* builder, uint8_t instance_flags, loom_value_id_t a,
    loom_value_id_t b, loom_value_id_t c, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

typedef iree_status_t (*loom_math_legalize_clampf_build_fn_t)(
    loom_builder_t* builder, uint8_t mode, uint8_t instance_flags,
    loom_value_id_t value, loom_value_id_t lower, loom_value_id_t upper,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op);

typedef iree_status_t (*loom_math_legalize_cast_build_fn_t)(
    loom_builder_t* builder, loom_value_id_t input, loom_type_t input_type,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op);

typedef struct loom_math_legalize_lane_builders_t {
  // Builds a uniform scalar or vector constant in the source lane domain.
  loom_math_legalize_constant_build_fn_t constant;
  // Builds a lane-wise addition in the source lane domain.
  loom_math_legalize_binary_build_fn_t addf;
  // Builds a lane-wise multiplication in the source lane domain.
  loom_math_legalize_binary_build_fn_t mulf;
  // Builds a lane-wise division in the source lane domain.
  loom_math_legalize_binary_build_fn_t divf;
  // Builds a lane-wise fused multiply-add in the source lane domain.
  loom_math_legalize_ternary_build_fn_t fmaf;
  // Builds a lane-wise clamp in the source lane domain.
  loom_math_legalize_clampf_build_fn_t clampf;
  // Builds a lane-wise base-2 exponential in the source lane domain.
  loom_math_legalize_unary_build_fn_t exp2f;
  // Builds a lane-wise base-2 logarithm in the source lane domain.
  loom_math_legalize_unary_build_fn_t log2f;
  // Builds a lane-wise sine over turns in the source lane domain.
  loom_math_legalize_unary_build_fn_t sinturnsf;
  // Builds a lane-wise cosine over turns in the source lane domain.
  loom_math_legalize_unary_build_fn_t costurnsf;
  // Builds a lane-wise logistic op in the source lane domain.
  loom_math_legalize_unary_build_fn_t logisticf;
  // Builds a lane-wise floating-point precision extension.
  loom_math_legalize_cast_build_fn_t extf;
  // Builds a lane-wise floating-point precision truncation.
  loom_math_legalize_cast_build_fn_t fptrunc;
} loom_math_legalize_lane_builders_t;

typedef struct loom_math_legalize_source_t {
  // Primary source op operand that feeds the semantic math op.
  loom_value_id_t input;
  // Secondary source op operand for binary semantic math ops.
  loom_value_id_t secondary_input;
  // Source result type preserved by the replacement expression.
  loom_type_t result_type;
  // Source fast-math flags forwarded to replacement floating-point ops.
  uint8_t fastmath_flags;
  // Target-authorized fast-math flags for recipe-internal ops.
  uint8_t recipe_fastmath_flags;
  // Source location copied to replacement ops.
  loom_location_id_t location;
  // Builder table for the source scalar/vector lane domain.
  const loom_math_legalize_lane_builders_t* lane_builders;
  // Rewriter driving the recipe, including the current value facts.
  loom_rewriter_t* rewriter;
} loom_math_legalize_source_t;

static uint8_t loom_math_legalize_clampf_mode(uint8_t fastmath_flags) {
  const uint8_t number_flags =
      LOOM_SCALAR_FASTMATHFLAGS_NNAN | LOOM_SCALAR_FASTMATHFLAGS_NSZ;
  return (fastmath_flags & number_flags) == number_flags
             ? LOOM_SCALAR_CLAMPF_MODE_NUMBER
             : LOOM_SCALAR_CLAMPF_MODE_ORDERED;
}

static uint8_t loom_math_legalize_recipe_fastmath_flags(
    const loom_math_legalize_recipe_context_t* context) {
  return context->decision.recipe_fastmath_flags &
         LOOM_TARGET_MATH_FASTMATH_FLAG_FAST;
}

static iree_status_t loom_math_legalize_scalar_clampf_build(
    loom_builder_t* builder, uint8_t mode, uint8_t instance_flags,
    loom_value_id_t value, loom_value_id_t lower, loom_value_id_t upper,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op) {
  return loom_scalar_clampf_build(builder, (loom_scalar_clampf_mode_t)mode,
                                  instance_flags, value, lower, upper,
                                  result_type, location, out_op);
}

static iree_status_t loom_math_legalize_vector_clampf_build(
    loom_builder_t* builder, uint8_t mode, uint8_t instance_flags,
    loom_value_id_t value, loom_value_id_t lower, loom_value_id_t upper,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op) {
  return loom_vector_clampf_build(builder, (loom_vector_clampf_mode_t)mode,
                                  instance_flags, value, lower, upper,
                                  result_type, location, out_op);
}

static const loom_math_legalize_lane_builders_t kScalarLaneBuilders = {
    .constant = loom_scalar_constant_build,
    .addf = loom_scalar_addf_build,
    .mulf = loom_scalar_mulf_build,
    .divf = loom_scalar_divf_build,
    .fmaf = loom_scalar_fmaf_build,
    .clampf = loom_math_legalize_scalar_clampf_build,
    .exp2f = loom_scalar_exp2f_build,
    .log2f = loom_scalar_log2f_build,
    .sinturnsf = loom_scalar_sinturnsf_build,
    .costurnsf = loom_scalar_costurnsf_build,
    .logisticf = loom_scalar_logisticf_build,
    .extf = loom_scalar_extf_build,
    .fptrunc = loom_scalar_fptrunc_build,
};

static const loom_math_legalize_lane_builders_t kVectorLaneBuilders = {
    .constant = loom_vector_constant_build,
    .addf = loom_vector_addf_build,
    .mulf = loom_vector_mulf_build,
    .divf = loom_vector_divf_build,
    .fmaf = loom_vector_fmaf_build,
    .clampf = loom_math_legalize_vector_clampf_build,
    .exp2f = loom_vector_exp2f_build,
    .log2f = loom_vector_log2f_build,
    .sinturnsf = loom_vector_sinturnsf_build,
    .costurnsf = loom_vector_costurnsf_build,
    .logisticf = loom_vector_logisticf_build,
    .extf = loom_vector_extf_build,
    .fptrunc = loom_vector_fptrunc_build,
};

static iree_status_t loom_math_legalize_source_initialize(
    const loom_math_legalize_recipe_context_t* context, const loom_op_t* op,
    loom_rewriter_t* rewriter, loom_math_legalize_source_t* out_source) {
  const loom_math_legalize_lane_builders_t* lane_builders = NULL;
  switch (context->query.lane_domain) {
    case LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR:
      lane_builders = &kScalarLaneBuilders;
      break;
    case LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR:
      lane_builders = &kVectorLaneBuilders;
      break;
    case LOOM_TARGET_MATH_LANE_DOMAIN_UNKNOWN:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "math recipe selected for unknown lane domain");
  }

  if (op->operand_count < 1 || op->result_count != 1) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "math recipe selected for op with unsupported operand/result shape");
  }

  const loom_value_id_t* operands = loom_op_const_operands(op);
  *out_source = (loom_math_legalize_source_t){
      .input = operands[0],
      .secondary_input =
          op->operand_count >= 2 ? operands[1] : LOOM_VALUE_ID_INVALID,
      .result_type =
          loom_module_value_type(context->module, loom_op_results(op)[0]),
      .fastmath_flags = op->instance_flags,
      .recipe_fastmath_flags =
          loom_math_legalize_recipe_fastmath_flags(context),
      .location = op->location,
      .lane_builders = lane_builders,
      .rewriter = rewriter,
  };
  return iree_ok_status();
}

static double loom_math_legalize_gelu_logistic_scale(const loom_op_t* op) {
  return loom_attr_as_f64(loom_op_attrs(op)[1]);
}

static iree_status_t loom_math_legalize_build_constant(
    loom_builder_t* builder, const loom_math_legalize_source_t* source,
    double value, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(source->lane_builders->constant(
      builder, loom_attr_f64(value), source->result_type, source->location,
      &op));
  *out_value = loom_op_results(op)[0];
  return iree_ok_status();
}

static iree_status_t loom_math_legalize_build_unary(
    loom_builder_t* builder, const loom_math_legalize_source_t* source,
    loom_math_legalize_unary_build_fn_t build, loom_value_id_t input,
    loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(build(builder, source->fastmath_flags, input,
                             source->result_type, source->location, &op));
  *out_value = loom_op_results(op)[0];
  return iree_ok_status();
}

static iree_status_t loom_math_legalize_build_binary(
    loom_builder_t* builder, const loom_math_legalize_source_t* source,
    loom_math_legalize_binary_build_fn_t build, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(build(builder, source->fastmath_flags, lhs, rhs,
                             source->result_type, source->location, &op));
  *out_value = loom_op_results(op)[0];
  return iree_ok_status();
}

static iree_status_t loom_math_legalize_build_cast(
    loom_builder_t* builder, loom_math_legalize_cast_build_fn_t build,
    loom_value_id_t input, loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(
      build(builder, input, input_type, result_type, location, &op));
  *out_value = loom_op_results(op)[0];
  return iree_ok_status();
}

static iree_status_t loom_math_legalize_build_division(
    loom_builder_t* builder, const loom_math_legalize_source_t* source,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  const uint8_t fastmath_flags =
      source->fastmath_flags | source->recipe_fastmath_flags;
  IREE_RETURN_IF_ERROR(source->lane_builders->divf(builder, fastmath_flags, lhs,
                                                   rhs, source->result_type,
                                                   source->location, &op));
  *out_value = loom_op_results(op)[0];
  return iree_ok_status();
}

static iree_status_t loom_math_legalize_build_ternary(
    loom_builder_t* builder, const loom_math_legalize_source_t* source,
    loom_math_legalize_ternary_build_fn_t build, loom_value_id_t a,
    loom_value_id_t b, loom_value_id_t c, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(build(builder, source->fastmath_flags, a, b, c,
                             source->result_type, source->location, &op));
  *out_value = loom_op_results(op)[0];
  return iree_ok_status();
}

static iree_status_t loom_math_legalize_build_polynomial(
    loom_builder_t* builder, const loom_math_legalize_source_t* source,
    loom_value_id_t input, const double* coefficients,
    iree_host_size_t coefficient_count, loom_value_id_t* out_value) {
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, coefficients[0], &accumulator));
  for (iree_host_size_t i = 1; i < coefficient_count; ++i) {
    loom_value_id_t coefficient = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
        builder, source, coefficients[i], &coefficient));
    IREE_RETURN_IF_ERROR(loom_math_legalize_build_ternary(
        builder, source, source->lane_builders->fmaf, accumulator, input,
        coefficient, &accumulator));
  }
  *out_value = accumulator;
  return iree_ok_status();
}

static iree_status_t loom_math_legalize_build_clampf(
    loom_builder_t* builder, const loom_math_legalize_source_t* source,
    loom_value_id_t value, double lower, double upper,
    loom_value_id_t* out_value) {
  loom_value_id_t lower_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, lower, &lower_value));
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, upper, &upper_value));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(source->lane_builders->clampf(
      builder, loom_math_legalize_clampf_mode(source->fastmath_flags),
      source->fastmath_flags, value, lower_value, upper_value,
      source->result_type, source->location, &op));
  *out_value = loom_op_results(op)[0];
  return iree_ok_status();
}

static iree_status_t loom_math_legalize_build_erf_rational(
    loom_builder_t* builder, const loom_math_legalize_source_t* source,
    loom_value_id_t input, loom_value_id_t* out_value) {
  static const double kAlphaCoefficients[] = {
      -2.72614225801306e-10, 2.77068142495902e-08,  -2.10102402082508e-06,
      -5.69250639462346e-05, -7.34990630326855e-04, -2.95459980854025e-03,
      -1.60960333262415e-02,
  };
  static const double kBetaCoefficients[] = {
      -1.45660718464996e-05, -2.13374055278905e-04, -1.68282697438203e-03,
      -7.37332916720468e-03, -1.42647390514189e-02,
  };

  loom_value_id_t clamped_input = LOOM_VALUE_ID_INVALID;
  loom_value_id_t input_squared = LOOM_VALUE_ID_INVALID;
  loom_value_id_t alpha = LOOM_VALUE_ID_INVALID;
  loom_value_id_t beta = LOOM_VALUE_ID_INVALID;
  loom_value_id_t numerator = LOOM_VALUE_ID_INVALID;
  loom_value_id_t quotient = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_clampf(
      builder, source, input, -4.0, 4.0, &clamped_input));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, clamped_input,
      clamped_input, &input_squared));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_polynomial(
      builder, source, input_squared, kAlphaCoefficients,
      IREE_ARRAYSIZE(kAlphaCoefficients), &alpha));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_polynomial(
      builder, source, input_squared, kBetaCoefficients,
      IREE_ARRAYSIZE(kBetaCoefficients), &beta));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, clamped_input, alpha,
      &numerator));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_division(
      builder, source, numerator, beta, &quotient));
  return loom_math_legalize_build_clampf(builder, source, quotient, -1.0, 1.0,
                                         out_value);
}

static iree_status_t loom_math_legalize_build_exp_exp2(
    loom_builder_t* builder, const loom_math_legalize_source_t* source,
    loom_value_id_t* out_value) {
  loom_value_id_t log2_e = LOOM_VALUE_ID_INVALID;
  loom_value_id_t scaled = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, 1.44269504088896340736, &log2_e));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, source->input, log2_e,
      &scaled));
  return loom_math_legalize_build_unary(
      builder, source, source->lane_builders->exp2f, scaled, out_value);
}

static iree_status_t loom_math_legalize_build_log_log2(
    loom_builder_t* builder, const loom_math_legalize_source_t* source,
    loom_value_id_t* out_value) {
  loom_value_id_t ln2 = LOOM_VALUE_ID_INVALID;
  loom_value_id_t log2_input = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, 0.69314718055994530942, &ln2));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, source->lane_builders->log2f, source->input,
      &log2_input));
  return loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, log2_input, ln2, out_value);
}

static bool loom_math_legalize_value_is_f64_close(
    const loom_math_legalize_source_t* source, loom_value_id_t value,
    double expected) {
  loom_value_facts_t facts = loom_rewriter_value_facts(source->rewriter, value);
  if (source->rewriter->fact_table != NULL) {
    loom_value_fact_uniform_element_t uniform_element = {0};
    if (loom_value_facts_query_uniform_element(
            &source->rewriter->fact_table->context, facts, &uniform_element)) {
      facts = uniform_element.element;
    }
  }
  if (!loom_value_facts_is_exact(facts) || !loom_value_facts_is_float(facts)) {
    return false;
  }
  const double actual = loom_value_facts_as_f64(facts);
  return fabs(actual - expected) <= 1.0e-9;
}

static bool loom_math_legalize_try_project_turns_input(
    const loom_math_legalize_source_t* source, loom_value_id_t* out_input) {
  const loom_value_t* input_value =
      loom_module_value(source->rewriter->module, source->input);
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
  if (loom_math_legalize_value_is_f64_close(source, rhs, two_pi)) {
    *out_input = lhs;
    return true;
  }
  if (loom_math_legalize_value_is_f64_close(source, lhs, two_pi)) {
    *out_input = rhs;
    return true;
  }
  return false;
}

static iree_status_t loom_math_legalize_build_turns_input(
    loom_builder_t* builder, const loom_math_legalize_source_t* source,
    loom_value_id_t* out_value) {
  if (loom_math_legalize_try_project_turns_input(source, out_value)) {
    return iree_ok_status();
  }

  loom_value_id_t inverse_two_pi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, 0.15915494309189533577, &inverse_two_pi));
  return loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, source->input,
      inverse_two_pi, out_value);
}

static iree_status_t loom_math_legalize_build_sin_turns(
    loom_builder_t* builder, const loom_math_legalize_source_t* source,
    loom_value_id_t* out_value) {
  loom_value_id_t turns_input = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_turns_input(builder, source, &turns_input));
  return loom_math_legalize_build_unary(builder, source,
                                        source->lane_builders->sinturnsf,
                                        turns_input, out_value);
}

static iree_status_t loom_math_legalize_build_cos_turns(
    loom_builder_t* builder, const loom_math_legalize_source_t* source,
    loom_value_id_t* out_value) {
  loom_value_id_t turns_input = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_turns_input(builder, source, &turns_input));
  return loom_math_legalize_build_unary(builder, source,
                                        source->lane_builders->costurnsf,
                                        turns_input, out_value);
}

static iree_status_t loom_math_legalize_build_logistic_exp2(
    loom_builder_t* builder, const loom_math_legalize_source_t* source,
    loom_value_id_t* out_value) {
  loom_value_id_t negative_log2_e = LOOM_VALUE_ID_INVALID;
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  loom_value_id_t scaled = LOOM_VALUE_ID_INVALID;
  loom_value_id_t exponent = LOOM_VALUE_ID_INVALID;
  loom_value_id_t denominator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, -1.44269504088896340736, &negative_log2_e));
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 1.0, &one));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, source->input,
      negative_log2_e, &scaled));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, source->lane_builders->exp2f, scaled, &exponent));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->addf, one, exponent,
      &denominator));
  return loom_math_legalize_build_division(builder, source, one, denominator,
                                           out_value);
}

static iree_status_t loom_math_legalize_build_tanh_logistic(
    loom_builder_t* builder, const loom_math_legalize_source_t* source,
    loom_value_id_t input, loom_value_id_t* out_value) {
  loom_value_id_t negative_one = LOOM_VALUE_ID_INVALID;
  loom_value_id_t two = LOOM_VALUE_ID_INVALID;
  loom_value_id_t scaled = LOOM_VALUE_ID_INVALID;
  loom_value_id_t logistic = LOOM_VALUE_ID_INVALID;
  loom_value_id_t doubled_logistic = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, -1.0, &negative_one));
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 2.0, &two));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, input, two, &scaled));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, source->lane_builders->logisticf, scaled, &logistic));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, logistic, two,
      &doubled_logistic));
  return loom_math_legalize_build_binary(
      builder, source, source->lane_builders->addf, doubled_logistic,
      negative_one, out_value);
}

static iree_status_t loom_math_legalize_build_pow_log2_exp2(
    loom_builder_t* builder, const loom_math_legalize_source_t* source,
    loom_value_id_t* out_value) {
  loom_value_id_t log2_base = LOOM_VALUE_ID_INVALID;
  loom_value_id_t exponent = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, source->lane_builders->log2f, source->input,
      &log2_base));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, log2_base,
      source->secondary_input, &exponent));
  return loom_math_legalize_build_unary(
      builder, source, source->lane_builders->exp2f, exponent, out_value);
}

static iree_status_t loom_math_legalize_build_silu_logistic(
    loom_builder_t* builder, const loom_math_legalize_source_t* source,
    loom_value_id_t* out_value) {
  loom_value_id_t logistic = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, source->lane_builders->logisticf, source->input,
      &logistic));
  return loom_math_legalize_build_binary(builder, source,
                                         source->lane_builders->mulf,
                                         source->input, logistic, out_value);
}

static iree_status_t loom_math_legalize_build_softplus_exp2(
    loom_builder_t* builder, const loom_math_legalize_source_t* source,
    loom_value_id_t* out_value) {
  loom_value_id_t log2_e = LOOM_VALUE_ID_INVALID;
  loom_value_id_t ln2 = LOOM_VALUE_ID_INVALID;
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  loom_value_id_t scaled = LOOM_VALUE_ID_INVALID;
  loom_value_id_t exponent = LOOM_VALUE_ID_INVALID;
  loom_value_id_t sum = LOOM_VALUE_ID_INVALID;
  loom_value_id_t log2_sum = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, 1.44269504088896340736, &log2_e));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, 0.69314718055994530942, &ln2));
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 1.0, &one));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, source->input, log2_e,
      &scaled));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, source->lane_builders->exp2f, scaled, &exponent));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->addf, one, exponent, &sum));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, source->lane_builders->log2f, sum, &log2_sum));
  return loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, log2_sum, ln2, out_value);
}

static iree_status_t loom_math_legalize_build_gelu_tanh(
    loom_builder_t* builder, const loom_math_legalize_source_t* source,
    loom_value_id_t* out_value) {
  loom_value_id_t half = LOOM_VALUE_ID_INVALID;
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  loom_value_id_t cubic_coefficient = LOOM_VALUE_ID_INVALID;
  loom_value_id_t sqrt_2_over_pi = LOOM_VALUE_ID_INVALID;
  loom_value_id_t input_squared = LOOM_VALUE_ID_INVALID;
  loom_value_id_t input_cubed = LOOM_VALUE_ID_INVALID;
  loom_value_id_t cubic_term = LOOM_VALUE_ID_INVALID;
  loom_value_id_t inner_sum = LOOM_VALUE_ID_INVALID;
  loom_value_id_t scaled = LOOM_VALUE_ID_INVALID;
  loom_value_id_t tanh_approximation = LOOM_VALUE_ID_INVALID;
  loom_value_id_t one_plus_tanh = LOOM_VALUE_ID_INVALID;
  loom_value_id_t half_input = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 0.5, &half));
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 1.0, &one));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, 0.044715, &cubic_coefficient));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, 0.79788456080286535588, &sqrt_2_over_pi));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, source->input,
      source->input, &input_squared));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, input_squared,
      source->input, &input_cubed));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, cubic_coefficient,
      input_cubed, &cubic_term));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->addf, source->input, cubic_term,
      &inner_sum));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, sqrt_2_over_pi, inner_sum,
      &scaled));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_tanh_logistic(
      builder, source, scaled, &tanh_approximation));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->addf, one, tanh_approximation,
      &one_plus_tanh));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, half, source->input,
      &half_input));
  return loom_math_legalize_build_binary(builder, source,
                                         source->lane_builders->mulf,
                                         half_input, one_plus_tanh, out_value);
}

static iree_status_t loom_math_legalize_build_gelu_erf(
    loom_builder_t* builder, const loom_math_legalize_source_t* source,
    loom_value_id_t* out_value) {
  loom_value_id_t half = LOOM_VALUE_ID_INVALID;
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  loom_value_id_t inverse_sqrt2 = LOOM_VALUE_ID_INVALID;
  loom_value_id_t scaled = LOOM_VALUE_ID_INVALID;
  loom_value_id_t erf = LOOM_VALUE_ID_INVALID;
  loom_value_id_t one_plus_erf = LOOM_VALUE_ID_INVALID;
  loom_value_id_t half_input = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 0.5, &half));
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_constant(builder, source, 1.0, &one));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, 0.70710678118654752440, &inverse_sqrt2));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, source->input,
      inverse_sqrt2, &scaled));
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_build_erf_rational(builder, source, scaled, &erf));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->addf, one, erf, &one_plus_erf));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, half, source->input,
      &half_input));
  return loom_math_legalize_build_binary(builder, source,
                                         source->lane_builders->mulf,
                                         half_input, one_plus_erf, out_value);
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
  // Builder table for the source scalar/vector lane domain.
  const loom_math_legalize_lane_builders_t* lane_builders;
  // Builds the widened arithmetic op.
  loom_math_legalize_binary_build_fn_t binary_build;
} loom_math_legalize_binary_source_t;

static loom_type_t loom_math_legalize_type_with_element(
    loom_type_t source_type, loom_scalar_type_t element_type) {
  loom_type_t result_type = source_type;
  result_type.header = loom_type_make_header(
      loom_type_kind(source_type), element_type, loom_type_rank(source_type),
      loom_type_flags(source_type));
  return result_type;
}

static iree_status_t loom_math_legalize_binary_source_initialize(
    const loom_math_legalize_recipe_context_t* context, const loom_op_t* op,
    loom_math_legalize_binary_source_t* out_source) {
  *out_source = (loom_math_legalize_binary_source_t){
      .result_type =
          loom_module_value_type(context->module, loom_op_results(op)[0]),
      .location = op->location,
  };
  out_source->widened_type = loom_math_legalize_type_with_element(
      out_source->result_type, LOOM_SCALAR_TYPE_F32);
  switch (op->kind) {
    case LOOM_OP_SCALAR_ADDF:
      out_source->lhs = loom_scalar_addf_lhs(op);
      out_source->rhs = loom_scalar_addf_rhs(op);
      out_source->fastmath_flags = loom_scalar_addf_fastmath(op);
      out_source->lane_builders = &kScalarLaneBuilders;
      out_source->binary_build = loom_scalar_addf_build;
      return iree_ok_status();
    case LOOM_OP_SCALAR_MULF:
      out_source->lhs = loom_scalar_mulf_lhs(op);
      out_source->rhs = loom_scalar_mulf_rhs(op);
      out_source->fastmath_flags = loom_scalar_mulf_fastmath(op);
      out_source->lane_builders = &kScalarLaneBuilders;
      out_source->binary_build = loom_scalar_mulf_build;
      return iree_ok_status();
    case LOOM_OP_VECTOR_ADDF:
      out_source->lhs = loom_vector_addf_lhs(op);
      out_source->rhs = loom_vector_addf_rhs(op);
      out_source->fastmath_flags = loom_vector_addf_fastmath(op);
      out_source->lane_builders = &kVectorLaneBuilders;
      out_source->binary_build = loom_vector_addf_build;
      return iree_ok_status();
    case LOOM_OP_VECTOR_MULF:
      out_source->lhs = loom_vector_mulf_lhs(op);
      out_source->rhs = loom_vector_mulf_rhs(op);
      out_source->fastmath_flags = loom_vector_mulf_fastmath(op);
      out_source->lane_builders = &kVectorLaneBuilders;
      out_source->binary_build = loom_vector_mulf_build;
      return iree_ok_status();
    default:
      break;
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "math recipe row referenced unsupported op kind %u",
                          op->kind);
}

static iree_status_t loom_math_legalize_build_widen_f32_round_bf16(
    loom_builder_t* builder, const loom_math_legalize_recipe_context_t* context,
    const loom_op_t* op, loom_value_id_t* out_value) {
  loom_math_legalize_binary_source_t source;
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_binary_source_initialize(context, op, &source));

  loom_value_id_t wide_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_cast(
      builder, source.lane_builders->extf, source.lhs, source.result_type,
      source.widened_type, source.location, &wide_lhs));
  loom_value_id_t wide_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_cast(
      builder, source.lane_builders->extf, source.rhs, source.result_type,
      source.widened_type, source.location, &wide_rhs));

  loom_op_t* wide_op = NULL;
  IREE_RETURN_IF_ERROR(
      source.binary_build(builder, source.fastmath_flags, wide_lhs, wide_rhs,
                          source.widened_type, source.location, &wide_op));
  const loom_value_id_t wide_result = loom_op_results(wide_op)[0];

  return loom_math_legalize_build_cast(
      builder, source.lane_builders->fptrunc, wide_result, source.widened_type,
      source.result_type, source.location, out_value);
}

static iree_status_t loom_math_legalize_build_gelu_logistic(
    loom_builder_t* builder, const loom_math_legalize_source_t* source,
    const loom_op_t* op, loom_value_id_t* out_value) {
  loom_value_id_t scale = LOOM_VALUE_ID_INVALID;
  loom_value_id_t scaled = LOOM_VALUE_ID_INVALID;
  loom_value_id_t logistic = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_constant(
      builder, source, loom_math_legalize_gelu_logistic_scale(op), &scale));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_binary(
      builder, source, source->lane_builders->mulf, scale, source->input,
      &scaled));
  IREE_RETURN_IF_ERROR(loom_math_legalize_build_unary(
      builder, source, source->lane_builders->logisticf, scaled, &logistic));
  return loom_math_legalize_build_binary(builder, source,
                                         source->lane_builders->mulf,
                                         source->input, logistic, out_value);
}

static iree_status_t loom_math_legalize_build_recipe(
    const loom_math_legalize_recipe_context_t* context, loom_op_t* op,
    loom_rewriter_t* rewriter, loom_value_id_t* out_value) {
  if (context->decision.recipe ==
      LOOM_TARGET_MATH_RECIPE_WIDEN_F32_ROUND_BF16) {
    return loom_math_legalize_build_widen_f32_round_bf16(
        &rewriter->builder, context, op, out_value);
  }

  loom_math_legalize_source_t source;
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_source_initialize(context, op, rewriter, &source));
  switch (context->decision.recipe) {
    case LOOM_TARGET_MATH_RECIPE_EXP_EXP2_F32:
      return loom_math_legalize_build_exp_exp2(&rewriter->builder, &source,
                                               out_value);
    case LOOM_TARGET_MATH_RECIPE_LOG_LOG2_F32:
      return loom_math_legalize_build_log_log2(&rewriter->builder, &source,
                                               out_value);
    case LOOM_TARGET_MATH_RECIPE_TANH_LOGISTIC_F32:
      return loom_math_legalize_build_tanh_logistic(&rewriter->builder, &source,
                                                    source.input, out_value);
    case LOOM_TARGET_MATH_RECIPE_POW_LOG2_EXP2_F32:
      return loom_math_legalize_build_pow_log2_exp2(&rewriter->builder, &source,
                                                    out_value);
    case LOOM_TARGET_MATH_RECIPE_SIN_TURNS_F32:
      return loom_math_legalize_build_sin_turns(&rewriter->builder, &source,
                                                out_value);
    case LOOM_TARGET_MATH_RECIPE_COS_TURNS_F32:
      return loom_math_legalize_build_cos_turns(&rewriter->builder, &source,
                                                out_value);
    case LOOM_TARGET_MATH_RECIPE_ERF_RATIONAL_F32:
      return loom_math_legalize_build_erf_rational(&rewriter->builder, &source,
                                                   source.input, out_value);
    case LOOM_TARGET_MATH_RECIPE_LOGISTIC_EXP2_F32:
      return loom_math_legalize_build_logistic_exp2(&rewriter->builder, &source,
                                                    out_value);
    case LOOM_TARGET_MATH_RECIPE_SILU_LOGISTIC_F32:
      return loom_math_legalize_build_silu_logistic(&rewriter->builder, &source,
                                                    out_value);
    case LOOM_TARGET_MATH_RECIPE_SOFTPLUS_EXP2_F32:
      return loom_math_legalize_build_softplus_exp2(&rewriter->builder, &source,
                                                    out_value);
    case LOOM_TARGET_MATH_RECIPE_GELU_TANH_F32:
      return loom_math_legalize_build_gelu_tanh(&rewriter->builder, &source,
                                                out_value);
    case LOOM_TARGET_MATH_RECIPE_GELU_ERF_F32:
      return loom_math_legalize_build_gelu_erf(&rewriter->builder, &source,
                                               out_value);
    case LOOM_TARGET_MATH_RECIPE_GELU_LOGISTIC_F32:
      return loom_math_legalize_build_gelu_logistic(&rewriter->builder, &source,
                                                    op, out_value);
    case LOOM_TARGET_MATH_RECIPE_WIDEN_F32_ROUND_BF16:
      IREE_ASSERT_UNREACHABLE("recipe is not elementwise math legalization");
      IREE_BUILTIN_UNREACHABLE();
    case LOOM_TARGET_MATH_RECIPE_UNKNOWN:
      break;
  }
  return iree_make_status(IREE_STATUS_INTERNAL, "unknown math recipe %u",
                          context->decision.recipe);
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
    case LOOM_TARGET_MATH_RECIPE_POW_LOG2_EXP2_F32:
    case LOOM_TARGET_MATH_RECIPE_ERF_RATIONAL_F32:
    case LOOM_TARGET_MATH_RECIPE_LOGISTIC_EXP2_F32:
    case LOOM_TARGET_MATH_RECIPE_SILU_LOGISTIC_F32:
    case LOOM_TARGET_MATH_RECIPE_SOFTPLUS_EXP2_F32:
    case LOOM_TARGET_MATH_RECIPE_GELU_TANH_F32:
    case LOOM_TARGET_MATH_RECIPE_GELU_ERF_F32:
    case LOOM_TARGET_MATH_RECIPE_GELU_LOGISTIC_F32:
    case LOOM_TARGET_MATH_RECIPE_WIDEN_F32_ROUND_BF16:
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
