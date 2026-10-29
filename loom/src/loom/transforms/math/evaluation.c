// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/math/evaluation.h"

#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"

static_assert(LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_DAZ ==
                  LOOM_SCALAR_FLOATCONVERSIONFLAGS_DAZ,
              "math and scalar conversion input flags must match");
static_assert(LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_FTZ ==
                  LOOM_SCALAR_FLOATCONVERSIONFLAGS_FTZ,
              "math and scalar conversion result flags must match");
static_assert(LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_DAZ ==
                  LOOM_VECTOR_FLOATCONVERSIONFLAGS_DAZ,
              "math and vector conversion input flags must match");
static_assert(LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_FTZ ==
                  LOOM_VECTOR_FLOATCONVERSIONFLAGS_FTZ,
              "math and vector conversion result flags must match");

typedef iree_status_t (*loom_math_evaluation_constantf_build_fn_t)(
    loom_builder_t* builder, double value, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

typedef iree_status_t (*loom_math_evaluation_constanti_build_fn_t)(
    loom_builder_t* builder, int64_t value, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

typedef iree_status_t (*loom_math_evaluation_unary_build_fn_t)(
    loom_builder_t* builder, uint8_t instance_flags, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op);

typedef iree_status_t (*loom_math_evaluation_binary_build_fn_t)(
    loom_builder_t* builder, uint8_t instance_flags, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);

typedef iree_status_t (*loom_math_evaluation_cmpf_build_fn_t)(
    loom_builder_t* builder, uint8_t instance_flags, uint8_t predicate,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_type_t operand_type,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op);

typedef iree_status_t (*loom_math_evaluation_select_build_fn_t)(
    loom_builder_t* builder, loom_value_id_t condition,
    loom_value_id_t true_value, loom_value_id_t false_value,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op);

typedef iree_status_t (*loom_math_evaluation_ternary_build_fn_t)(
    loom_builder_t* builder, uint8_t instance_flags, loom_value_id_t a,
    loom_value_id_t b, loom_value_id_t c, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

typedef iree_status_t (*loom_math_evaluation_clampf_build_fn_t)(
    loom_builder_t* builder, uint8_t mode, uint8_t instance_flags,
    loom_value_id_t value, loom_value_id_t lower, loom_value_id_t upper,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op);

typedef iree_status_t (*loom_math_evaluation_cast_build_fn_t)(
    loom_builder_t* builder, uint8_t instance_flags, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

struct loom_math_evaluation_lane_ops_t {
  // Builds a uniform scalar or vector floating-point constant.
  loom_math_evaluation_constantf_build_fn_t constantf;
  // Builds a uniform scalar or vector integer constant.
  loom_math_evaluation_constanti_build_fn_t constanti;
  // Builds lane-wise floating-point addition.
  loom_math_evaluation_binary_build_fn_t addf;
  // Builds lane-wise floating-point subtraction.
  loom_math_evaluation_binary_build_fn_t subf;
  // Builds lane-wise floating-point multiplication.
  loom_math_evaluation_binary_build_fn_t mulf;
  // Builds lane-wise floating-point division.
  loom_math_evaluation_binary_build_fn_t divf;
  // Builds lane-wise fused multiply-add.
  loom_math_evaluation_ternary_build_fn_t fmaf;
  // Builds lane-wise floating-point clamp.
  loom_math_evaluation_clampf_build_fn_t clampf;
  // Builds a lane-wise floating-point comparison.
  loom_math_evaluation_cmpf_build_fn_t cmpf;
  // Ordered-greater-or-equal predicate for the selected dialect.
  uint8_t cmpf_ordered_greater_equal_predicate;
  // Ordered-equal predicate for the selected dialect.
  uint8_t cmpf_ordered_equal_predicate;
  // Selects values in the scalar or vector lane domain.
  loom_math_evaluation_select_build_fn_t select;
  // Builds lane-wise floating-point absolute value.
  loom_math_evaluation_unary_build_fn_t absf;
  // Builds lane-wise floating-point copysign.
  loom_math_evaluation_binary_build_fn_t copysignf;
  // Builds lane-wise base-2 exponential.
  loom_math_evaluation_unary_build_fn_t exp2f;
  // Builds lane-wise base-2 logarithm.
  loom_math_evaluation_unary_build_fn_t log2f;
  // Builds lane-wise sine over turns.
  loom_math_evaluation_unary_build_fn_t sinturnsf;
  // Builds lane-wise cosine over turns.
  loom_math_evaluation_unary_build_fn_t costurnsf;
  // Builds lane-wise logistic.
  loom_math_evaluation_unary_build_fn_t logisticf;
  // Builds lane-wise hyperbolic tangent.
  loom_math_evaluation_unary_build_fn_t tanhf;
  // Builds lane-wise truncation toward zero.
  loom_math_evaluation_unary_build_fn_t truncf;
  // Builds a lane-wise floating-point square root.
  loom_math_evaluation_unary_build_fn_t sqrtf;
  // Builds a lane-wise floating-point reciprocal square root.
  loom_math_evaluation_unary_build_fn_t rsqrtf;
  // Builds lane-wise sine over radians.
  loom_math_evaluation_unary_build_fn_t sinf;
  // Builds lane-wise cosine over radians.
  loom_math_evaluation_unary_build_fn_t cosf;
  // Builds lane-wise rounding to the nearest even integer.
  loom_math_evaluation_unary_build_fn_t roundevenf;
};

static iree_status_t loom_math_evaluation_scalar_clampf_build(
    loom_builder_t* builder, uint8_t mode, uint8_t instance_flags,
    loom_value_id_t value, loom_value_id_t lower, loom_value_id_t upper,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op) {
  return loom_scalar_clampf_build(builder, (loom_scalar_clampf_mode_t)mode,
                                  instance_flags, value, lower, upper,
                                  result_type, location, out_op);
}

static iree_status_t loom_math_evaluation_vector_clampf_build(
    loom_builder_t* builder, uint8_t mode, uint8_t instance_flags,
    loom_value_id_t value, loom_value_id_t lower, loom_value_id_t upper,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op) {
  return loom_vector_clampf_build(builder, (loom_vector_clampf_mode_t)mode,
                                  instance_flags, value, lower, upper,
                                  result_type, location, out_op);
}

static iree_status_t loom_math_evaluation_scalar_copysignf_build(
    loom_builder_t* builder, uint8_t instance_flags, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op) {
  return loom_scalar_copysignf_build(builder, instance_flags, lhs, rhs,
                                     result_type, location, out_op);
}

static iree_status_t loom_math_evaluation_scalar_cmpf_build(
    loom_builder_t* builder, uint8_t instance_flags, uint8_t predicate,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_type_t operand_type,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op) {
  (void)operand_type;
  (void)result_type;
  return loom_scalar_cmpf_build(builder, instance_flags,
                                (loom_scalar_cmpf_predicate_t)predicate, lhs,
                                rhs, location, out_op);
}

static iree_status_t loom_math_evaluation_vector_cmpf_build(
    loom_builder_t* builder, uint8_t instance_flags, uint8_t predicate,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_type_t operand_type,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op) {
  return loom_vector_cmpf_build(builder, instance_flags, predicate, lhs, rhs,
                                operand_type, result_type, location, out_op);
}

static iree_status_t loom_math_evaluation_scalar_constantf_build(
    loom_builder_t* builder, double value, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op) {
  return loom_scalar_constant_build(builder, loom_attr_f64(value), result_type,
                                    location, out_op);
}

static iree_status_t loom_math_evaluation_scalar_constanti_build(
    loom_builder_t* builder, int64_t value, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op) {
  return loom_scalar_constant_build(builder, loom_attr_i64(value), result_type,
                                    location, out_op);
}

static const loom_math_evaluation_lane_ops_t kScalarLaneOps = {
    .constantf = loom_math_evaluation_scalar_constantf_build,
    .constanti = loom_math_evaluation_scalar_constanti_build,
    .addf = loom_scalar_addf_build,
    .subf = loom_scalar_subf_build,
    .mulf = loom_scalar_mulf_build,
    .divf = loom_scalar_divf_build,
    .fmaf = loom_scalar_fmaf_build,
    .clampf = loom_math_evaluation_scalar_clampf_build,
    .cmpf = loom_math_evaluation_scalar_cmpf_build,
    .cmpf_ordered_greater_equal_predicate = LOOM_SCALAR_CMPF_PREDICATE_OGE,
    .cmpf_ordered_equal_predicate = LOOM_SCALAR_CMPF_PREDICATE_OEQ,
    .select = loom_scf_select_build,
    .absf = loom_scalar_absf_build,
    .copysignf = loom_math_evaluation_scalar_copysignf_build,
    .exp2f = loom_scalar_exp2f_build,
    .log2f = loom_scalar_log2f_build,
    .sinturnsf = loom_scalar_sinturnsf_build,
    .costurnsf = loom_scalar_costurnsf_build,
    .logisticf = loom_scalar_logisticf_build,
    .tanhf = loom_scalar_tanhf_build,
    .truncf = loom_scalar_truncf_build,
    .sqrtf = loom_scalar_sqrtf_build,
    .rsqrtf = loom_scalar_rsqrtf_build,
    .sinf = loom_scalar_sinf_build,
    .cosf = loom_scalar_cosf_build,
    .roundevenf = loom_scalar_roundevenf_build,
};

static const loom_math_evaluation_lane_ops_t kVectorLaneOps = {
    .constantf = loom_vector_splatf_build,
    .constanti = loom_vector_splati_build,
    .addf = loom_vector_addf_build,
    .subf = loom_vector_subf_build,
    .mulf = loom_vector_mulf_build,
    .divf = loom_vector_divf_build,
    .fmaf = loom_vector_fmaf_build,
    .clampf = loom_math_evaluation_vector_clampf_build,
    .cmpf = loom_math_evaluation_vector_cmpf_build,
    .cmpf_ordered_greater_equal_predicate = LOOM_VECTOR_CMPF_PREDICATE_OGE,
    .cmpf_ordered_equal_predicate = LOOM_VECTOR_CMPF_PREDICATE_OEQ,
    .select = loom_vector_select_build,
    .absf = loom_vector_absf_build,
    .copysignf = loom_vector_copysignf_build,
    .exp2f = loom_vector_exp2f_build,
    .log2f = loom_vector_log2f_build,
    .sinturnsf = loom_vector_sinturnsf_build,
    .costurnsf = loom_vector_costurnsf_build,
    .logisticf = loom_vector_logisticf_build,
    .tanhf = loom_vector_tanhf_build,
    .truncf = loom_vector_truncf_build,
    .sqrtf = loom_vector_sqrtf_build,
    .rsqrtf = loom_vector_rsqrtf_build,
    .sinf = loom_vector_sinf_build,
    .cosf = loom_vector_cosf_build,
    .roundevenf = loom_vector_roundevenf_build,
};

static loom_type_t loom_math_evaluation_type_with_element(
    loom_type_t source_type, loom_scalar_type_t element_type) {
  loom_type_t result_type = source_type;
  result_type.header = loom_type_make_header(
      loom_type_kind(source_type), element_type, loom_type_rank(source_type),
      loom_type_flags(source_type));
  return result_type;
}

static loom_type_t loom_math_evaluation_packet_type(
    loom_scalar_type_t element_type, uint64_t element_count) {
  return loom_type_shaped_1d(LOOM_TYPE_VECTOR, element_type, element_count, 0);
}

static uint8_t loom_math_evaluation_clampf_mode(uint8_t fastmath_flags) {
  const uint8_t number_flags =
      LOOM_SCALAR_FASTMATHFLAGS_NNAN | LOOM_SCALAR_FASTMATHFLAGS_NSZ;
  return (fastmath_flags & number_flags) == number_flags
             ? LOOM_SCALAR_CLAMPF_MODE_NUMBER
             : LOOM_SCALAR_CLAMPF_MODE_ORDERED;
}

static iree_status_t loom_math_evaluation_build_cast(
    loom_builder_t* builder, uint8_t instance_flags, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  if (loom_type_equal(input_type, result_type)) {
    *out_value = input;
    return iree_ok_status();
  }

  const bool is_vector = loom_type_is_vector(input_type);
  const loom_scalar_type_t input_element = loom_type_element_type(input_type);
  const loom_scalar_type_t result_element = loom_type_element_type(result_type);
  const int32_t input_width = loom_scalar_type_bitwidth(input_element);
  const int32_t result_width = loom_scalar_type_bitwidth(result_element);
  if (input_width == result_width) {
    const loom_type_t intermediate_type =
        loom_math_evaluation_type_with_element(input_type,
                                               LOOM_SCALAR_TYPE_F32);
    const uint8_t input_flags =
        instance_flags & LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_DAZ;
    const uint8_t result_flags =
        instance_flags & LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_FTZ;
    loom_value_id_t intermediate = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_cast(
        builder, input_flags, input, input_type, intermediate_type, location,
        &intermediate));
    return loom_math_evaluation_build_cast(builder, result_flags, intermediate,
                                           intermediate_type, result_type,
                                           location, out_value);
  }

  const loom_math_evaluation_cast_build_fn_t build =
      input_width < result_width
          ? (is_vector ? loom_vector_extf_build : loom_scalar_extf_build)
          : (is_vector ? loom_vector_fptrunc_build : loom_scalar_fptrunc_build);
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(build(builder, instance_flags, input, input_type,
                             result_type, location, &op));
  *out_value = loom_op_results(op)[0];
  return iree_ok_status();
}

static iree_status_t loom_math_evaluation_import_grouped_product(
    loom_math_evaluation_t* evaluation, loom_value_id_t source,
    loom_math_evaluation_value_t* out_value) {
  loom_builder_t* builder = &evaluation->rewriter->builder;
  const loom_type_t source_type = evaluation->source_result_type;
  const loom_scalar_type_t source_element = loom_type_element_type(source_type);
  if (loom_type_is_scalar(source_type)) {
    const loom_type_t source_packet_type = loom_math_evaluation_packet_type(
        source_element, evaluation->descriptor.packet_lane_count);
    loom_op_t* splat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_splat_build(
        builder, source, source_packet_type, evaluation->location, &splat_op));
    loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_cast(
        builder, evaluation->descriptor.conversion.input_flags,
        loom_vector_splat_result(splat_op), source_packet_type,
        evaluation->value_type, evaluation->location, &accumulator));
    *out_value = loom_math_evaluation_value_from_id(accumulator);
    return iree_ok_status();
  }

  const loom_type_t flat_source_type = loom_math_evaluation_packet_type(
      source_element, evaluation->source_element_count);
  loom_value_id_t flat_source = source;
  if (!loom_type_equal(source_type, flat_source_type)) {
    loom_op_t* bitcast_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
        builder, source, source_type, flat_source_type, evaluation->location,
        &bitcast_op));
    flat_source = loom_vector_bitcast_result(bitcast_op);
  }

  const loom_type_t flat_accumulator_type = loom_math_evaluation_packet_type(
      evaluation->descriptor.accumulator_element_type,
      evaluation->source_element_count);
  loom_value_id_t flat_accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_cast(
      builder, evaluation->descriptor.conversion.input_flags, flat_source,
      flat_source_type, flat_accumulator_type, evaluation->location,
      &flat_accumulator));
  if (evaluation->source_element_count ==
      evaluation->descriptor.packet_lane_count) {
    *out_value = loom_math_evaluation_value_from_id(flat_accumulator);
    return iree_ok_status();
  }

  const loom_type_t padding_type = loom_math_evaluation_packet_type(
      evaluation->descriptor.accumulator_element_type,
      evaluation->descriptor.packet_lane_count -
          evaluation->source_element_count);
  loom_op_t* padding_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_splatf_build(
      builder, 0.0, padding_type, evaluation->location, &padding_op));
  const loom_value_id_t inputs[] = {
      flat_accumulator,
      loom_vector_splatf_result(padding_op),
  };
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_concat_build(
      builder, 0, inputs, IREE_ARRAYSIZE(inputs), evaluation->value_type,
      evaluation->location, &concat_op));
  *out_value =
      loom_math_evaluation_value_from_id(loom_vector_concat_result(concat_op));
  return iree_ok_status();
}

iree_status_t loom_math_evaluation_initialize(
    const loom_module_t* module, const loom_target_math_query_t* query,
    const loom_target_math_policy_decision_t* decision, const loom_op_t* op,
    loom_rewriter_t* rewriter, loom_math_evaluation_t* out_evaluation) {
  IREE_ASSERT(op->operand_count >= 1 && op->result_count == 1);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_type_t source_result_type =
      loom_module_value_type(module, loom_op_results(op)[0]);
  *out_evaluation = (loom_math_evaluation_t){
      .source_input = operands[0],
      .input = loom_math_evaluation_value_invalid(),
      .secondary_input = loom_math_evaluation_value_invalid(),
      .source_result_type = source_result_type,
      .value_type = source_result_type,
      .fastmath_flags = query->fastmath_flags,
      .recipe_fastmath_flags =
          decision->recipe_fastmath_flags & LOOM_TARGET_MATH_FASTMATH_FLAG_FAST,
      .location = op->location,
      .descriptor = decision->evaluation,
      .rewriter = rewriter,
      .zero_accumulator = LOOM_VALUE_ID_INVALID,
      .zero_product = LOOM_VALUE_ID_INVALID,
  };

  switch (decision->evaluation.kind) {
    case LOOM_TARGET_MATH_EVALUATION_SOURCE:
      switch (query->lane_domain) {
        case LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR:
          out_evaluation->lane_ops = &kScalarLaneOps;
          break;
        case LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR:
          out_evaluation->lane_ops = &kVectorLaneOps;
          break;
        case LOOM_TARGET_MATH_LANE_DOMAIN_UNKNOWN:
          IREE_ASSERT_UNREACHABLE("math recipe selected unknown lane domain");
          IREE_BUILTIN_UNREACHABLE();
      }
      out_evaluation->input = loom_math_evaluation_value_from_id(operands[0]);
      if (op->operand_count >= 2) {
        out_evaluation->secondary_input =
            loom_math_evaluation_value_from_id(operands[1]);
      }
      return iree_ok_status();

    case LOOM_TARGET_MATH_EVALUATION_WIDENED: {
      out_evaluation->lane_ops =
          query->lane_domain == LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR
              ? &kScalarLaneOps
              : &kVectorLaneOps;
      out_evaluation->value_type = loom_math_evaluation_type_with_element(
          source_result_type, decision->evaluation.accumulator_element_type);
      loom_value_id_t input = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_math_evaluation_build_cast(
          &rewriter->builder, decision->evaluation.conversion.input_flags,
          operands[0], source_result_type, out_evaluation->value_type,
          op->location, &input));
      out_evaluation->input = loom_math_evaluation_value_from_id(input);
      if (op->operand_count >= 2) {
        loom_value_id_t secondary = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_math_evaluation_build_cast(
            &rewriter->builder, decision->evaluation.conversion.input_flags,
            operands[1], source_result_type, out_evaluation->value_type,
            op->location, &secondary));
        out_evaluation->secondary_input =
            loom_math_evaluation_value_from_id(secondary);
      }
      return iree_ok_status();
    }

    case LOOM_TARGET_MATH_EVALUATION_GROUPED_PRODUCT: {
      const loom_target_math_evaluation_t* descriptor = &decision->evaluation;
      IREE_ASSERT(descriptor->product_element_type == LOOM_SCALAR_TYPE_F16 ||
                  descriptor->product_element_type == LOOM_SCALAR_TYPE_BF16);
      IREE_ASSERT(descriptor->accumulator_element_type == LOOM_SCALAR_TYPE_F32);
      IREE_ASSERT(descriptor->packet_lane_count > 0);
      out_evaluation->lane_ops = &kVectorLaneOps;
      out_evaluation->value_type = loom_math_evaluation_packet_type(
          descriptor->accumulator_element_type, descriptor->packet_lane_count);
      out_evaluation->product_type = loom_math_evaluation_packet_type(
          descriptor->product_element_type, descriptor->packet_lane_count);
      out_evaluation->paired_product_type = loom_math_evaluation_packet_type(
          descriptor->product_element_type,
          (uint64_t)descriptor->packet_lane_count * 2);
      if (loom_type_is_scalar(source_result_type)) {
        out_evaluation->source_element_count = 1;
      } else {
        const bool has_static_element_count = loom_type_static_element_count(
            source_result_type, &out_evaluation->source_element_count);
        IREE_ASSERT(has_static_element_count &&
                    out_evaluation->source_element_count >= 1 &&
                    out_evaluation->source_element_count <=
                        descriptor->packet_lane_count);
      }
      IREE_RETURN_IF_ERROR(loom_math_evaluation_import_grouped_product(
          out_evaluation, operands[0], &out_evaluation->input));
      if (op->operand_count >= 2) {
        IREE_RETURN_IF_ERROR(loom_math_evaluation_import_grouped_product(
            out_evaluation, operands[1], &out_evaluation->secondary_input));
      }
      return iree_ok_status();
    }
  }
  IREE_ASSERT_UNREACHABLE("math recipe selected unknown evaluation kind");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_math_evaluation_build_zero(
    loom_math_evaluation_t* evaluation, loom_type_t type,
    loom_value_id_t* inout_value) {
  if (*inout_value != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_splatf_build(
      &evaluation->rewriter->builder, 0.0, type, evaluation->location, &op));
  *inout_value = loom_vector_splatf_result(op);
  return iree_ok_status();
}

static iree_status_t loom_math_evaluation_materialize_product(
    loom_math_evaluation_t* evaluation, loom_math_evaluation_value_t* value) {
  if (value->product != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  loom_builder_t* builder = &evaluation->rewriter->builder;
  loom_value_id_t narrow = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_cast(
      builder, evaluation->descriptor.conversion.product_flags,
      value->accumulator, evaluation->value_type, evaluation->product_type,
      evaluation->location, &narrow));
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_zero(
      evaluation, evaluation->product_type, &evaluation->zero_product));
  loom_op_t* interleave_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_interleave_build(
      builder, 0, narrow, evaluation->zero_product,
      evaluation->paired_product_type, evaluation->location, &interleave_op));
  value->product = loom_vector_interleave_result(interleave_op);
  return iree_ok_status();
}

iree_status_t loom_math_evaluation_export(
    loom_math_evaluation_t* evaluation,
    const loom_math_evaluation_value_t* value, loom_value_id_t* out_value) {
  if (evaluation->descriptor.kind == LOOM_TARGET_MATH_EVALUATION_SOURCE) {
    *out_value = value->accumulator;
    return iree_ok_status();
  }
  if (evaluation->descriptor.kind == LOOM_TARGET_MATH_EVALUATION_WIDENED) {
    return loom_math_evaluation_build_cast(
        &evaluation->rewriter->builder,
        evaluation->descriptor.conversion.result_flags, value->accumulator,
        evaluation->value_type, evaluation->source_result_type,
        evaluation->location, out_value);
  }

  loom_builder_t* builder = &evaluation->rewriter->builder;
  const loom_scalar_type_t source_element =
      loom_type_element_type(evaluation->source_result_type);
  if (loom_type_is_scalar(evaluation->source_result_type)) {
    loom_value_id_t packet = value->accumulator;
    loom_type_t packet_type = evaluation->value_type;
    if (source_element != evaluation->descriptor.accumulator_element_type) {
      packet_type = loom_math_evaluation_packet_type(
          source_element, evaluation->descriptor.packet_lane_count);
      IREE_RETURN_IF_ERROR(loom_math_evaluation_build_cast(
          builder, evaluation->descriptor.conversion.result_flags,
          value->accumulator, evaluation->value_type, packet_type,
          evaluation->location, &packet));
    }
    const int64_t static_index = 0;
    loom_op_t* extract_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_extract_build(
        builder, packet, NULL, 0, &static_index, 1,
        evaluation->source_result_type, evaluation->location, &extract_op));
    *out_value = loom_vector_extract_result(extract_op);
    return iree_ok_status();
  }

  loom_value_id_t flat_value = value->accumulator;
  loom_type_t flat_value_type = evaluation->value_type;
  if (evaluation->source_element_count !=
      evaluation->descriptor.packet_lane_count) {
    flat_value_type = loom_math_evaluation_packet_type(
        evaluation->descriptor.accumulator_element_type,
        evaluation->source_element_count);
    const int64_t static_offset = 0;
    loom_op_t* slice_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_slice_build(
        builder, flat_value, NULL, 0, &static_offset, 1, flat_value_type,
        evaluation->location, &slice_op));
    flat_value = loom_vector_slice_result(slice_op);
  }

  const loom_type_t flat_source_type = loom_math_evaluation_packet_type(
      source_element, evaluation->source_element_count);
  loom_value_id_t flat_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_math_evaluation_build_cast(
      builder, evaluation->descriptor.conversion.result_flags, flat_value,
      flat_value_type, flat_source_type, evaluation->location, &flat_source));
  if (loom_type_equal(flat_source_type, evaluation->source_result_type)) {
    *out_value = flat_source;
    return iree_ok_status();
  }

  loom_op_t* bitcast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
      builder, flat_source, flat_source_type, evaluation->source_result_type,
      evaluation->location, &bitcast_op));
  *out_value = loom_vector_bitcast_result(bitcast_op);
  return iree_ok_status();
}

iree_status_t loom_math_evaluation_build_constant(
    loom_math_evaluation_t* evaluation, double value,
    loom_math_evaluation_value_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(evaluation->lane_ops->constantf(
      &evaluation->rewriter->builder, value, evaluation->value_type,
      evaluation->location, &op));
  *out_value = loom_math_evaluation_value_from_id(loom_op_results(op)[0]);
  return iree_ok_status();
}

iree_status_t loom_math_evaluation_build_unary(
    loom_math_evaluation_t* evaluation,
    loom_math_evaluation_unary_op_t unary_op,
    const loom_math_evaluation_value_t* input,
    loom_math_evaluation_value_t* out_value) {
  loom_math_evaluation_unary_build_fn_t build = NULL;
  switch (unary_op) {
    case LOOM_MATH_EVALUATION_UNARY_ABSF:
      build = evaluation->lane_ops->absf;
      break;
    case LOOM_MATH_EVALUATION_UNARY_EXP2F:
      build = evaluation->lane_ops->exp2f;
      break;
    case LOOM_MATH_EVALUATION_UNARY_LOG2F:
      build = evaluation->lane_ops->log2f;
      break;
    case LOOM_MATH_EVALUATION_UNARY_SINTURNSF:
      build = evaluation->lane_ops->sinturnsf;
      break;
    case LOOM_MATH_EVALUATION_UNARY_COSTURNSF:
      build = evaluation->lane_ops->costurnsf;
      break;
    case LOOM_MATH_EVALUATION_UNARY_LOGISTICF:
      build = evaluation->lane_ops->logisticf;
      break;
    case LOOM_MATH_EVALUATION_UNARY_TANHF:
      build = evaluation->lane_ops->tanhf;
      break;
    case LOOM_MATH_EVALUATION_UNARY_TRUNCF:
      build = evaluation->lane_ops->truncf;
      break;
    case LOOM_MATH_EVALUATION_UNARY_SQRTF:
      build = evaluation->lane_ops->sqrtf;
      break;
    case LOOM_MATH_EVALUATION_UNARY_RSQRTF:
      build = evaluation->lane_ops->rsqrtf;
      break;
    case LOOM_MATH_EVALUATION_UNARY_SINF:
      build = evaluation->lane_ops->sinf;
      break;
    case LOOM_MATH_EVALUATION_UNARY_COSF:
      build = evaluation->lane_ops->cosf;
      break;
    case LOOM_MATH_EVALUATION_UNARY_ROUNDEVENF:
      build = evaluation->lane_ops->roundevenf;
      break;
  }
  IREE_ASSERT(build != NULL);
  uint8_t recipe_fastmath_flags =
      evaluation->recipe_fastmath_flags & LOOM_TARGET_MATH_FASTMATH_FLAG_AFN;
  if (unary_op == LOOM_MATH_EVALUATION_UNARY_RSQRTF) {
    recipe_fastmath_flags |=
        evaluation->recipe_fastmath_flags & LOOM_TARGET_MATH_FASTMATH_FLAG_ARCP;
  }
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(build(&evaluation->rewriter->builder,
                             evaluation->fastmath_flags | recipe_fastmath_flags,
                             input->accumulator, evaluation->value_type,
                             evaluation->location, &op));
  *out_value = loom_math_evaluation_value_from_id(loom_op_results(op)[0]);
  return iree_ok_status();
}

static iree_status_t loom_math_evaluation_build_grouped_product(
    loom_math_evaluation_t* evaluation, loom_math_evaluation_value_t* lhs,
    loom_math_evaluation_value_t* rhs, loom_value_id_t accumulator,
    loom_math_evaluation_value_t* out_value) {
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_materialize_product(evaluation, lhs));
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_materialize_product(evaluation, rhs));
  loom_op_t* dot_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_dot2f_build(
      &evaluation->rewriter->builder, lhs->product, rhs->product, accumulator,
      evaluation->value_type, evaluation->location, &dot_op));
  *out_value =
      loom_math_evaluation_value_from_id(loom_vector_dot2f_result(dot_op));
  return iree_ok_status();
}

iree_status_t loom_math_evaluation_build_binary(
    loom_math_evaluation_t* evaluation,
    loom_math_evaluation_binary_op_t binary_op,
    loom_math_evaluation_value_t* lhs, loom_math_evaluation_value_t* rhs,
    loom_math_evaluation_value_t* out_value) {
  if (binary_op == LOOM_MATH_EVALUATION_BINARY_MULF &&
      evaluation->descriptor.kind ==
          LOOM_TARGET_MATH_EVALUATION_GROUPED_PRODUCT) {
    IREE_RETURN_IF_ERROR(loom_math_evaluation_build_zero(
        evaluation, evaluation->value_type, &evaluation->zero_accumulator));
    return loom_math_evaluation_build_grouped_product(
        evaluation, lhs, rhs, evaluation->zero_accumulator, out_value);
  }

  loom_math_evaluation_binary_build_fn_t build = NULL;
  uint8_t fastmath_flags = evaluation->fastmath_flags;
  switch (binary_op) {
    case LOOM_MATH_EVALUATION_BINARY_ADDF:
      build = evaluation->lane_ops->addf;
      break;
    case LOOM_MATH_EVALUATION_BINARY_SUBF:
      build = evaluation->lane_ops->subf;
      break;
    case LOOM_MATH_EVALUATION_BINARY_MULF:
      build = evaluation->lane_ops->mulf;
      break;
    case LOOM_MATH_EVALUATION_BINARY_DIVF:
      build = evaluation->lane_ops->divf;
      fastmath_flags |= evaluation->recipe_fastmath_flags;
      break;
    case LOOM_MATH_EVALUATION_BINARY_COPYSIGNF:
      build = evaluation->lane_ops->copysignf;
      break;
  }
  IREE_ASSERT(build != NULL);
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(build(
      &evaluation->rewriter->builder, fastmath_flags, lhs->accumulator,
      rhs->accumulator, evaluation->value_type, evaluation->location, &op));
  *out_value = loom_math_evaluation_value_from_id(loom_op_results(op)[0]);
  return iree_ok_status();
}

iree_status_t loom_math_evaluation_build_fma(
    loom_math_evaluation_t* evaluation, loom_math_evaluation_value_t* lhs,
    loom_math_evaluation_value_t* rhs,
    const loom_math_evaluation_value_t* accumulator,
    loom_math_evaluation_value_t* out_value) {
  if (evaluation->descriptor.kind ==
      LOOM_TARGET_MATH_EVALUATION_GROUPED_PRODUCT) {
    return loom_math_evaluation_build_grouped_product(
        evaluation, lhs, rhs, accumulator->accumulator, out_value);
  }
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(evaluation->lane_ops->fmaf(
      &evaluation->rewriter->builder, evaluation->fastmath_flags,
      lhs->accumulator, rhs->accumulator, accumulator->accumulator,
      evaluation->value_type, evaluation->location, &op));
  *out_value = loom_math_evaluation_value_from_id(loom_op_results(op)[0]);
  return iree_ok_status();
}

iree_status_t loom_math_evaluation_build_clamp(
    loom_math_evaluation_t* evaluation, loom_math_evaluation_value_t* value,
    double lower, double upper, loom_math_evaluation_value_t* out_value) {
  loom_math_evaluation_value_t lower_value;
  loom_math_evaluation_value_t upper_value;
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(evaluation, lower, &lower_value));
  IREE_RETURN_IF_ERROR(
      loom_math_evaluation_build_constant(evaluation, upper, &upper_value));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(evaluation->lane_ops->clampf(
      &evaluation->rewriter->builder,
      loom_math_evaluation_clampf_mode(evaluation->fastmath_flags),
      evaluation->fastmath_flags, value->accumulator, lower_value.accumulator,
      upper_value.accumulator, evaluation->value_type, evaluation->location,
      &op));
  *out_value = loom_math_evaluation_value_from_id(loom_op_results(op)[0]);
  return iree_ok_status();
}

static iree_status_t loom_math_evaluation_build_comparison(
    loom_math_evaluation_t* evaluation, uint8_t predicate,
    const loom_math_evaluation_value_t* lhs,
    const loom_math_evaluation_value_t* rhs, loom_value_id_t* out_value) {
  const loom_type_t mask_type = loom_math_evaluation_type_with_element(
      evaluation->value_type, LOOM_SCALAR_TYPE_I1);
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(evaluation->lane_ops->cmpf(
      &evaluation->rewriter->builder, evaluation->fastmath_flags, predicate,
      lhs->accumulator, rhs->accumulator, evaluation->value_type, mask_type,
      evaluation->location, &op));
  *out_value = loom_op_results(op)[0];
  return iree_ok_status();
}

iree_status_t loom_math_evaluation_build_ordered_greater_equal(
    loom_math_evaluation_t* evaluation, const loom_math_evaluation_value_t* lhs,
    const loom_math_evaluation_value_t* rhs, loom_value_id_t* out_value) {
  return loom_math_evaluation_build_comparison(
      evaluation, evaluation->lane_ops->cmpf_ordered_greater_equal_predicate,
      lhs, rhs, out_value);
}

iree_status_t loom_math_evaluation_build_ordered_equal(
    loom_math_evaluation_t* evaluation, const loom_math_evaluation_value_t* lhs,
    const loom_math_evaluation_value_t* rhs, loom_value_id_t* out_value) {
  return loom_math_evaluation_build_comparison(
      evaluation, evaluation->lane_ops->cmpf_ordered_equal_predicate, lhs, rhs,
      out_value);
}

iree_status_t loom_math_evaluation_build_positive_normal_finite_condition(
    loom_math_evaluation_t* evaluation,
    const loom_math_evaluation_value_t* input, loom_value_id_t* out_condition) {
  loom_scalar_type_t integer_element_type = LOOM_SCALAR_TYPE_NONE;
  uint64_t minimum_normal_bits = 0;
  uint64_t finite_normal_span = 0;
  switch (loom_type_element_type(evaluation->value_type)) {
    case LOOM_SCALAR_TYPE_F32:
      integer_element_type = LOOM_SCALAR_TYPE_I32;
      minimum_normal_bits = UINT32_C(0x00800000);
      finite_normal_span = UINT32_C(0x7EFFFFFF);
      break;
    case LOOM_SCALAR_TYPE_F64:
      integer_element_type = LOOM_SCALAR_TYPE_I64;
      minimum_normal_bits = UINT64_C(0x0010000000000000);
      finite_normal_span = UINT64_C(0x7FDFFFFFFFFFFFFF);
      break;
    default:
      IREE_ASSERT_UNREACHABLE(
          "reciprocal-root refinement requires F32 or F64 evaluation");
      IREE_BUILTIN_UNREACHABLE();
  }

  loom_builder_t* builder = &evaluation->rewriter->builder;
  const bool is_scalar_condition = loom_type_is_scalar(evaluation->value_type);
  const loom_type_t integer_type = loom_math_evaluation_type_with_element(
      evaluation->value_type, integer_element_type);
  const loom_type_t mask_type = loom_math_evaluation_type_with_element(
      evaluation->value_type, LOOM_SCALAR_TYPE_I1);
  loom_op_t* minimum_op = NULL;
  IREE_RETURN_IF_ERROR(evaluation->lane_ops->constanti(
      builder, (int64_t)minimum_normal_bits, integer_type, evaluation->location,
      &minimum_op));
  loom_op_t* span_op = NULL;
  IREE_RETURN_IF_ERROR(evaluation->lane_ops->constanti(
      builder, (int64_t)finite_normal_span, integer_type, evaluation->location,
      &span_op));

  loom_op_t* bits_op = NULL;
  loom_op_t* offset_op = NULL;
  loom_op_t* condition_op = NULL;
  if (is_scalar_condition) {
    IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(
        builder, input->accumulator, evaluation->value_type, integer_type,
        evaluation->location, &bits_op));
    IREE_RETURN_IF_ERROR(loom_scalar_subi_build(
        builder, /*instance_flags=*/0, loom_scalar_bitcast_result(bits_op),
        loom_op_results(minimum_op)[0], integer_type, evaluation->location,
        &offset_op));
    IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
        builder, LOOM_SCALAR_CMPI_PREDICATE_ULE,
        loom_scalar_subi_result(offset_op), loom_op_results(span_op)[0],
        evaluation->location, &condition_op));
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
        builder, input->accumulator, evaluation->value_type, integer_type,
        evaluation->location, &bits_op));
    IREE_RETURN_IF_ERROR(loom_vector_subi_build(
        builder, /*instance_flags=*/0, loom_vector_bitcast_result(bits_op),
        loom_op_results(minimum_op)[0], integer_type, evaluation->location,
        &offset_op));
    IREE_RETURN_IF_ERROR(loom_vector_cmpi_build(
        builder, LOOM_VECTOR_CMPI_PREDICATE_ULE,
        loom_vector_subi_result(offset_op), loom_op_results(span_op)[0],
        integer_type, mask_type, evaluation->location, &condition_op));
  }
  *out_condition = loom_op_results(condition_op)[0];
  return iree_ok_status();
}

iree_status_t loom_math_evaluation_build_select(
    loom_math_evaluation_t* evaluation, loom_value_id_t condition,
    const loom_math_evaluation_value_t* true_value,
    const loom_math_evaluation_value_t* false_value,
    loom_math_evaluation_value_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(evaluation->lane_ops->select(
      &evaluation->rewriter->builder, condition, true_value->accumulator,
      false_value->accumulator, evaluation->value_type, evaluation->location,
      &op));
  *out_value = loom_math_evaluation_value_from_id(loom_op_results(op)[0]);
  return iree_ok_status();
}
