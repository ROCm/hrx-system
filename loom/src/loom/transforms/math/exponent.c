// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/math/exponent.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/rewrite/rewriter.h"

typedef struct loom_math_exponent_builder_t {
  // Insertion point and allocation owner for the replacement expression.
  loom_builder_t* builder;
  // Authored location shared by the replacement operations.
  loom_location_id_t location;
  // Scalar or vector floating representation of the working value.
  loom_type_t float_type;
  // Equal-width integer representation used to manipulate IEEE fields.
  loom_type_t bits_type;
  // Signed exponent representation, preserving the source lane shape.
  loom_type_t exponent_type;
  // Scalar or vector predicate representation.
  loom_type_t predicate_type;
  // Whether each operation acts on a vector of independent lanes.
  bool is_vector;
  // Number of explicit significand bits in the working IEEE representation.
  int32_t fraction_bits;
  // Bias of the working IEEE exponent field.
  int32_t bias;
} loom_math_exponent_builder_t;

static loom_type_t loom_math_exponent_element_type(loom_type_t type,
                                                   loom_scalar_type_t element) {
  type.header =
      loom_type_make_header(loom_type_kind(type), element, loom_type_rank(type),
                            loom_type_flags(type));
  return type;
}

// Both recipes use the same small set of elementwise operations. The table
// carries scalar/vector selection and arity; construction consumes these
// trusted contracts without per-operation dispatch or builder wrappers.
#define LOOM_MATH_EXPONENT_OPS(_)     \
  _(CONSTANT, CONSTANT, SPLATI, 0, 1) \
  _(BITCAST, BITCAST, BITCAST, 1, 0)  \
  _(EXTSI, EXTSI, EXTSI, 1, 0)        \
  _(TRUNCI, TRUNCI, TRUNCI, 1, 0)     \
  _(CTLZI, CTLZI, CTLZI, 1, 0)        \
  _(ADDI, ADDI, ADDI, 2, 0)           \
  _(SUBI, SUBI, SUBI, 2, 0)           \
  _(ANDI, ANDI, ANDI, 2, 0)           \
  _(ORI, ORI, ORI, 2, 0)              \
  _(SHLI, SHLI, SHLI, 2, 0)           \
  _(SHRUI, SHRUI, SHRUI, 2, 0)        \
  _(MINSI, MINSI, MINSI, 2, 0)        \
  _(MAXSI, MAXSI, MAXSI, 2, 0)        \
  _(CMPI, CMPI, CMPI, 2, 1)

typedef enum loom_math_exponent_opcode_e {
#define LOOM_MATH_EXPONENT_ENUM(name, scalar, vector, operands, attributes) \
  LOOM_MATH_EXPONENT_##name,
  LOOM_MATH_EXPONENT_OPS(LOOM_MATH_EXPONENT_ENUM)
#undef LOOM_MATH_EXPONENT_ENUM
      LOOM_MATH_EXPONENT_SELECT,
} loom_math_exponent_opcode_t;

typedef struct loom_math_exponent_descriptor_t {
  // Operation kind in the scalar and vector lane domains, respectively.
  loom_op_kind_t kinds[2];
  // Fixed number of input values.
  uint8_t operand_count;
  // Fixed number of attributes.
  uint8_t attribute_count;
} loom_math_exponent_descriptor_t;

static const loom_math_exponent_descriptor_t kExponentDescriptors[] = {
#define LOOM_MATH_EXPONENT_ROW(name, scalar, vector, operands, attributes) \
  [LOOM_MATH_EXPONENT_##                                                   \
      name] = {{LOOM_OP_SCALAR_##scalar, LOOM_OP_VECTOR_##vector},         \
               operands,                                                   \
               attributes},
    LOOM_MATH_EXPONENT_OPS(LOOM_MATH_EXPONENT_ROW)
#undef LOOM_MATH_EXPONENT_ROW
        [LOOM_MATH_EXPONENT_SELECT] =
            {{LOOM_OP_SCF_SELECT, LOOM_OP_VECTOR_SELECT}, 3, 0},
};
#undef LOOM_MATH_EXPONENT_OPS

static iree_status_t loom_math_exponent_emit(
    loom_math_exponent_builder_t* builder, loom_math_exponent_opcode_t opcode,
    loom_type_t type, loom_value_id_t first, loom_value_id_t second,
    loom_value_id_t third, loom_attribute_t attribute,
    loom_value_id_t* out_value) {
  const loom_math_exponent_descriptor_t* descriptor =
      &kExponentDescriptors[opcode];
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      builder->builder, descriptor->kinds[builder->is_vector],
      descriptor->operand_count, 1, 0, 0, descriptor->attribute_count,
      builder->location, &op));
  const loom_value_id_t operands[] = {first, second, third};
  memcpy(loom_op_operands(op), operands,
         descriptor->operand_count * sizeof(*operands));
  if (descriptor->attribute_count) {
    loom_op_attrs(op)[0] = attribute;
  }
  IREE_RETURN_IF_ERROR(loom_builder_define_value(builder->builder, type,
                                                 &loom_op_results(op)[0]));
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder->builder, op));
  *out_value = loom_op_results(op)[0];
  return iree_ok_status();
}

static iree_status_t loom_math_exponent_constant(
    loom_math_exponent_builder_t* builder, loom_type_t type, int64_t value,
    loom_value_id_t* out_value) {
  if (loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32) {
    value = (int32_t)value;
  }
  return loom_math_exponent_emit(builder, LOOM_MATH_EXPONENT_CONSTANT, type, 0,
                                 0, 0, loom_attr_i64(value), out_value);
}

static iree_status_t loom_math_exponent_unary(
    loom_math_exponent_builder_t* builder, loom_math_exponent_opcode_t opcode,
    loom_type_t type, loom_value_id_t input, loom_value_id_t* out_value) {
  return loom_math_exponent_emit(builder, opcode, type, input, 0, 0,
                                 (loom_attribute_t){0}, out_value);
}

static iree_status_t loom_math_exponent_binary(
    loom_math_exponent_builder_t* builder, loom_math_exponent_opcode_t opcode,
    loom_type_t type, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_value_id_t* out_value) {
  return loom_math_exponent_emit(builder, opcode, type, lhs, rhs, 0,
                                 (loom_attribute_t){0}, out_value);
}

static iree_status_t loom_math_exponent_compare(
    loom_math_exponent_builder_t* builder, uint8_t predicate,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_value_id_t* out_value) {
  return loom_math_exponent_emit(builder, LOOM_MATH_EXPONENT_CMPI,
                                 builder->predicate_type, lhs, rhs, 0,
                                 loom_attr_enum(predicate), out_value);
}

static iree_status_t loom_math_exponent_select(
    loom_math_exponent_builder_t* builder, loom_type_t type,
    loom_value_id_t condition, loom_value_id_t true_value,
    loom_value_id_t false_value, loom_value_id_t* out_value) {
  return loom_math_exponent_emit(builder, LOOM_MATH_EXPONENT_SELECT, type,
                                 condition, true_value, false_value,
                                 (loom_attribute_t){0}, out_value);
}

// Integer normalization preserves subnormal inputs even when the target's
// ordinary floating arithmetic flushes them. Both exponent operations consume
// the same normalized fraction and biased exponent.
typedef struct loom_math_exponent_parts_t {
  // Original floating bit pattern, including exceptional payloads.
  loom_value_id_t bits;
  // Original sign bit.
  loom_value_id_t sign;
  // Normalized fraction with its implicit leading bit removed.
  loom_value_id_t fraction;
  // Biased exponent after subnormal normalization, in the integer bit domain.
  loom_value_id_t exponent;
  // Whether the original input is zero, infinity, or NaN.
  loom_value_id_t exceptional;
} loom_math_exponent_parts_t;

static iree_status_t loom_math_exponent_normalize(
    loom_math_exponent_builder_t* builder, loom_value_id_t input,
    loom_math_exponent_parts_t* out_parts) {
  const int32_t fraction_bits = builder->fraction_bits;
  const uint64_t sign_mask =
      fraction_bits == 52 ? UINT64_C(1) << 63 : UINT64_C(1) << 31;
  const loom_type_t type = builder->bits_type;
  loom_value_id_t sign_mask_value, magnitude_mask, magnitude, fraction_mask;
  IREE_RETURN_IF_ERROR(loom_math_exponent_unary(
      builder, LOOM_MATH_EXPONENT_BITCAST, type, input, &out_parts->bits));
  IREE_RETURN_IF_ERROR(
      loom_math_exponent_constant(builder, type, sign_mask, &sign_mask_value));
  IREE_RETURN_IF_ERROR(loom_math_exponent_constant(builder, type, sign_mask - 1,
                                                   &magnitude_mask));
  IREE_RETURN_IF_ERROR(loom_math_exponent_binary(
      builder, LOOM_MATH_EXPONENT_ANDI, type, out_parts->bits, sign_mask_value,
      &out_parts->sign));
  IREE_RETURN_IF_ERROR(
      loom_math_exponent_binary(builder, LOOM_MATH_EXPONENT_ANDI, type,
                                out_parts->bits, magnitude_mask, &magnitude));
  IREE_RETURN_IF_ERROR(loom_math_exponent_constant(
      builder, type, (UINT64_C(1) << fraction_bits) - 1, &fraction_mask));

  loom_value_id_t zero, one, shift, normal_leading, leading, normalization;
  IREE_RETURN_IF_ERROR(loom_math_exponent_constant(builder, type, 0, &zero));
  IREE_RETURN_IF_ERROR(loom_math_exponent_constant(builder, type, 1, &one));
  IREE_RETURN_IF_ERROR(
      loom_math_exponent_constant(builder, type, fraction_bits, &shift));
  IREE_RETURN_IF_ERROR(loom_math_exponent_constant(
      builder, type, (fraction_bits == 52 ? 63 : 31) - fraction_bits,
      &normal_leading));
  IREE_RETURN_IF_ERROR(loom_math_exponent_unary(
      builder, LOOM_MATH_EXPONENT_CTLZI, type, magnitude, &leading));
  IREE_RETURN_IF_ERROR(
      loom_math_exponent_binary(builder, LOOM_MATH_EXPONENT_SUBI, type, leading,
                                normal_leading, &normalization));
  IREE_RETURN_IF_ERROR(
      loom_math_exponent_binary(builder, LOOM_MATH_EXPONENT_MAXSI, type,
                                normalization, zero, &normalization));
  loom_value_id_t normalized;
  IREE_RETURN_IF_ERROR(
      loom_math_exponent_binary(builder, LOOM_MATH_EXPONENT_SHLI, type,
                                magnitude, normalization, &normalized));
  IREE_RETURN_IF_ERROR(loom_math_exponent_binary(
      builder, LOOM_MATH_EXPONENT_ANDI, type, normalized, fraction_mask,
      &out_parts->fraction));
  loom_value_id_t encoded;
  IREE_RETURN_IF_ERROR(loom_math_exponent_binary(
      builder, LOOM_MATH_EXPONENT_SHRUI, type, magnitude, shift, &encoded));
  IREE_RETURN_IF_ERROR(loom_math_exponent_binary(
      builder, LOOM_MATH_EXPONENT_MAXSI, type, encoded, one, &encoded));
  IREE_RETURN_IF_ERROR(
      loom_math_exponent_binary(builder, LOOM_MATH_EXPONENT_SUBI, type, encoded,
                                normalization, &out_parts->exponent));

  loom_value_id_t infinity, is_zero, is_special;
  IREE_RETURN_IF_ERROR(loom_math_exponent_constant(
      builder, type, (uint64_t)(2 * builder->bias + 1) << fraction_bits,
      &infinity));
  IREE_RETURN_IF_ERROR(loom_math_exponent_compare(
      builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, magnitude, zero, &is_zero));
  IREE_RETURN_IF_ERROR(
      loom_math_exponent_compare(builder, LOOM_SCALAR_CMPI_PREDICATE_UGE,
                                 magnitude, infinity, &is_special));
  return loom_math_exponent_binary(builder, LOOM_MATH_EXPONENT_ORI,
                                   builder->predicate_type, is_zero, is_special,
                                   &out_parts->exceptional);
}

static iree_status_t loom_math_exponent_decompose(
    loom_math_exponent_builder_t* builder,
    const loom_math_exponent_parts_t* parts, loom_value_id_t* out_fraction,
    loom_value_id_t* out_exponent) {
  const loom_type_t type = builder->bits_type;
  loom_value_id_t half, fraction, bias, exponent, zero;
  IREE_RETURN_IF_ERROR(loom_math_exponent_constant(
      builder, type, (uint64_t)(builder->bias - 1) << builder->fraction_bits,
      &half));
  IREE_RETURN_IF_ERROR(loom_math_exponent_binary(
      builder, LOOM_MATH_EXPONENT_ORI, type, parts->fraction, half, &fraction));
  IREE_RETURN_IF_ERROR(loom_math_exponent_binary(
      builder, LOOM_MATH_EXPONENT_ORI, type, fraction, parts->sign, &fraction));
  IREE_RETURN_IF_ERROR(loom_math_exponent_select(
      builder, type, parts->exceptional, parts->bits, fraction, &fraction));
  IREE_RETURN_IF_ERROR(
      loom_math_exponent_unary(builder, LOOM_MATH_EXPONENT_BITCAST,
                               builder->float_type, fraction, out_fraction));
  IREE_RETURN_IF_ERROR(
      loom_math_exponent_constant(builder, type, builder->bias - 1, &bias));
  IREE_RETURN_IF_ERROR(
      loom_math_exponent_binary(builder, LOOM_MATH_EXPONENT_SUBI, type,
                                parts->exponent, bias, &exponent));
  IREE_RETURN_IF_ERROR(loom_math_exponent_constant(builder, type, 0, &zero));
  IREE_RETURN_IF_ERROR(loom_math_exponent_select(
      builder, type, parts->exceptional, zero, exponent, &exponent));
  if (builder->fraction_bits == 52) {
    IREE_RETURN_IF_ERROR(
        loom_math_exponent_unary(builder, LOOM_MATH_EXPONENT_TRUNCI,
                                 builder->exponent_type, exponent, &exponent));
  }
  *out_exponent = exponent;
  return iree_ok_status();
}

static iree_status_t loom_math_exponent_scale(
    loom_math_exponent_builder_t* builder,
    const loom_math_exponent_parts_t* parts, loom_value_id_t exponent,
    loom_value_id_t* out_value) {
  const loom_type_t type = builder->bits_type;
  if (builder->fraction_bits == 52) {
    IREE_RETURN_IF_ERROR(loom_math_exponent_unary(
        builder, LOOM_MATH_EXPONENT_EXTSI, type, exponent, &exponent));
  }
  // Beyond these bounds every finite nonzero value has overflowed or rounded
  // to zero. Bound before adding so INT32_MIN/MAX cannot wrap the exponent.
  loom_value_id_t lower, upper;
  IREE_RETURN_IF_ERROR(
      loom_math_exponent_constant(builder, type, -4096, &lower));
  IREE_RETURN_IF_ERROR(
      loom_math_exponent_constant(builder, type, 4096, &upper));
  IREE_RETURN_IF_ERROR(loom_math_exponent_binary(
      builder, LOOM_MATH_EXPONENT_MAXSI, type, exponent, lower, &exponent));
  IREE_RETURN_IF_ERROR(loom_math_exponent_binary(
      builder, LOOM_MATH_EXPONENT_MINSI, type, exponent, upper, &exponent));
  IREE_RETURN_IF_ERROR(
      loom_math_exponent_binary(builder, LOOM_MATH_EXPONENT_ADDI, type,
                                parts->exponent, exponent, &exponent));
  loom_value_id_t zero, one, shift, hidden, significand;
  IREE_RETURN_IF_ERROR(loom_math_exponent_constant(builder, type, 0, &zero));
  IREE_RETURN_IF_ERROR(loom_math_exponent_constant(builder, type, 1, &one));
  IREE_RETURN_IF_ERROR(loom_math_exponent_constant(
      builder, type, builder->fraction_bits, &shift));
  IREE_RETURN_IF_ERROR(loom_math_exponent_constant(
      builder, type, UINT64_C(1) << builder->fraction_bits, &hidden));
  IREE_RETURN_IF_ERROR(
      loom_math_exponent_binary(builder, LOOM_MATH_EXPONENT_ORI, type,
                                parts->fraction, hidden, &significand));
  loom_value_id_t normal_bits;
  IREE_RETURN_IF_ERROR(loom_math_exponent_binary(
      builder, LOOM_MATH_EXPONENT_SHLI, type, exponent, shift, &normal_bits));
  IREE_RETURN_IF_ERROR(
      loom_math_exponent_binary(builder, LOOM_MATH_EXPONENT_ORI, type,
                                normal_bits, parts->fraction, &normal_bits));

  // Add half a discarded unit minus one, plus the retained low bit. This
  // rounds once to nearest-even, including ties to zero and to minimum normal.
  loom_value_id_t maximum_shift, right_shift, half_shift, half, rounding_bias;
  IREE_RETURN_IF_ERROR(loom_math_exponent_constant(
      builder, type, builder->fraction_bits + 2, &maximum_shift));
  IREE_RETURN_IF_ERROR(loom_math_exponent_binary(
      builder, LOOM_MATH_EXPONENT_SUBI, type, one, exponent, &right_shift));
  IREE_RETURN_IF_ERROR(loom_math_exponent_binary(
      builder, LOOM_MATH_EXPONENT_MAXSI, type, right_shift, one, &right_shift));
  IREE_RETURN_IF_ERROR(
      loom_math_exponent_binary(builder, LOOM_MATH_EXPONENT_MINSI, type,
                                right_shift, maximum_shift, &right_shift));
  IREE_RETURN_IF_ERROR(loom_math_exponent_binary(
      builder, LOOM_MATH_EXPONENT_SUBI, type, right_shift, one, &half_shift));
  IREE_RETURN_IF_ERROR(loom_math_exponent_binary(
      builder, LOOM_MATH_EXPONENT_SHLI, type, one, half_shift, &half));
  IREE_RETURN_IF_ERROR(loom_math_exponent_binary(
      builder, LOOM_MATH_EXPONENT_SUBI, type, half, one, &rounding_bias));
  loom_value_id_t retained, low_bit, rounded;
  IREE_RETURN_IF_ERROR(
      loom_math_exponent_binary(builder, LOOM_MATH_EXPONENT_SHRUI, type,
                                significand, right_shift, &retained));
  IREE_RETURN_IF_ERROR(loom_math_exponent_binary(
      builder, LOOM_MATH_EXPONENT_ANDI, type, retained, one, &low_bit));
  IREE_RETURN_IF_ERROR(
      loom_math_exponent_binary(builder, LOOM_MATH_EXPONENT_ADDI, type,
                                significand, rounding_bias, &rounded));
  IREE_RETURN_IF_ERROR(loom_math_exponent_binary(
      builder, LOOM_MATH_EXPONENT_ADDI, type, rounded, low_bit, &rounded));
  IREE_RETURN_IF_ERROR(loom_math_exponent_binary(
      builder, LOOM_MATH_EXPONENT_SHRUI, type, rounded, right_shift, &rounded));
  loom_value_id_t is_normal, result;
  IREE_RETURN_IF_ERROR(loom_math_exponent_compare(
      builder, LOOM_SCALAR_CMPI_PREDICATE_SGT, exponent, zero, &is_normal));
  IREE_RETURN_IF_ERROR(loom_math_exponent_select(
      builder, type, is_normal, normal_bits, rounded, &result));

  loom_value_id_t maximum_exponent, infinity, overflow;
  IREE_RETURN_IF_ERROR(loom_math_exponent_constant(
      builder, type, 2 * builder->bias + 1, &maximum_exponent));
  IREE_RETURN_IF_ERROR(loom_math_exponent_constant(
      builder, type,
      (uint64_t)(2 * builder->bias + 1) << builder->fraction_bits, &infinity));
  IREE_RETURN_IF_ERROR(
      loom_math_exponent_compare(builder, LOOM_SCALAR_CMPI_PREDICATE_SGE,
                                 exponent, maximum_exponent, &overflow));
  IREE_RETURN_IF_ERROR(loom_math_exponent_select(builder, type, overflow,
                                                 infinity, result, &result));
  IREE_RETURN_IF_ERROR(loom_math_exponent_binary(
      builder, LOOM_MATH_EXPONENT_ORI, type, result, parts->sign, &result));
  IREE_RETURN_IF_ERROR(loom_math_exponent_select(
      builder, type, parts->exceptional, parts->bits, result, &result));
  return loom_math_exponent_unary(builder, LOOM_MATH_EXPONENT_BITCAST,
                                  builder->float_type, result, out_value);
}

iree_status_t loom_math_legalize_exponent(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_type_t source_type =
      loom_module_value_type(context->module, loom_op_results(op)[0]);
  const loom_scalar_type_t element = loom_type_element_type(source_type);
  const bool is_vector = loom_type_is_vector(source_type);
  // A scalar-only packet policy consumes the semantic lane program directly.
  // Expanding it into vector integer arithmetic here would prevent later lane
  // lowering from selecting the target's scalar exponent instructions.
  const loom_target_vector_packet_policy_t* packet_policy =
      context->vector_packet_policy;
  if (is_vector && packet_policy &&
      packet_policy->native_bit_count_count == 0 &&
      packet_policy->native_lane_count_count == 0) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT;
    return iree_ok_status();
  }
  const bool is_scale =
      loom_scalar_ldexpf_isa(op) || loom_vector_ldexpf_isa(op);
  const loom_type_t exponent_type =
      loom_math_exponent_element_type(source_type, LOOM_SCALAR_TYPE_I32);
  loom_value_id_t results[2] = {0};
  if (loom_scalar_type_bitwidth(element) < 32) {
    const loom_type_t working_type =
        loom_math_exponent_element_type(source_type, LOOM_SCALAR_TYPE_F32);
    loom_op_t* extended = NULL;
    IREE_RETURN_IF_ERROR(
        (is_vector ? loom_vector_extf_build : loom_scalar_extf_build)(
            &rewriter->builder, 0, loom_op_operands(op)[0], source_type,
            working_type, op->location, &extended));
    loom_value_id_t input = loom_op_results(extended)[0];
    loom_op_t* widened = NULL;
    if (is_scale) {
      IREE_RETURN_IF_ERROR(
          (is_vector ? loom_vector_ldexpf_build : loom_scalar_ldexpf_build)(
              &rewriter->builder, input, loom_op_operands(op)[1], working_type,
              op->location, &widened));
    } else {
      if (is_vector) {
        const loom_type_t result_types[] = {working_type, exponent_type};
        IREE_RETURN_IF_ERROR(loom_vector_frexpf_build(
            &rewriter->builder, input, result_types, op->location, &widened));
      } else {
        IREE_RETURN_IF_ERROR(loom_scalar_frexpf_build(
            &rewriter->builder, input, working_type, op->location, &widened));
      }
      results[1] = loom_op_results(widened)[1];
    }
    loom_op_t* narrowed = NULL;
    IREE_RETURN_IF_ERROR(
        (is_vector ? loom_vector_fptrunc_build : loom_scalar_fptrunc_build)(
            &rewriter->builder, 0, loom_op_results(widened)[0], working_type,
            source_type, op->location, &narrowed));
    results[0] = loom_op_results(narrowed)[0];
  } else {
    const bool is_f64 = element == LOOM_SCALAR_TYPE_F64;
    loom_math_exponent_builder_t builder = {
        .builder = &rewriter->builder,
        .location = op->location,
        .float_type = source_type,
        .bits_type = loom_math_exponent_element_type(
            source_type, is_f64 ? LOOM_SCALAR_TYPE_I64 : LOOM_SCALAR_TYPE_I32),
        .exponent_type = exponent_type,
        .predicate_type =
            loom_math_exponent_element_type(source_type, LOOM_SCALAR_TYPE_I1),
        .is_vector = is_vector,
        .fraction_bits = is_f64 ? 52 : 23,
        .bias = is_f64 ? 1023 : 127,
    };
    loom_math_exponent_parts_t parts;
    IREE_RETURN_IF_ERROR(loom_math_exponent_normalize(
        &builder, loom_op_operands(op)[0], &parts));
    if (is_scale) {
      IREE_RETURN_IF_ERROR(loom_math_exponent_scale(
          &builder, &parts, loom_op_operands(op)[1], &results[0]));
    } else {
      IREE_RETURN_IF_ERROR(loom_math_exponent_decompose(
          &builder, &parts, &results[0], &results[1]));
    }
  }
  const uint16_t result_count = is_scale ? 1 : 2;
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, results, result_count, checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, results, result_count));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}
