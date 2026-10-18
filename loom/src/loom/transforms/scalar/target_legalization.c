// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scalar/target_legalization.h"

#include <math.h>

#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/numeric_format.h"
#include "loom/util/predicate_facts.h"

bool loom_scalar_match_multiply_add(
    const loom_module_t* module, const loom_op_t* multiply_op,
    loom_scalar_multiply_add_match_t* out_match) {
  *out_match = (loom_scalar_multiply_add_match_t){0};
  if (!loom_scalar_muli_isa(multiply_op) ||
      loom_scalar_muli_overflow(multiply_op) != 0) {
    return false;
  }

  const loom_value_id_t product = loom_scalar_muli_result(multiply_op);
  const loom_value_t* product_value = loom_module_value(module, product);
  const loom_use_t* product_use = loom_value_single_use(product_value);
  if (product_use == NULL || loom_value_has_attribute_uses(product_value) ||
      loom_module_value_has_type_uses(module, product)) {
    return false;
  }
  loom_op_t* add_op = loom_use_user_op(*product_use);
  if (!loom_scalar_addi_isa(add_op) || loom_scalar_addi_overflow(add_op) != 0) {
    return false;
  }
  const loom_value_id_t addend = loom_scalar_addi_lhs(add_op) == product
                                     ? loom_scalar_addi_rhs(add_op)
                                     : loom_scalar_addi_lhs(add_op);
  *out_match = (loom_scalar_multiply_add_match_t){
      .multiply_op = (loom_op_t*)multiply_op,
      .add_op = add_op,
      .addend = addend,
  };
  return true;
}

iree_status_t loom_scalar_fuse_multiply_add_match(
    loom_rewriter_t* rewriter, const loom_scalar_multiply_add_match_t* match) {
  loom_builder_set_before(&rewriter->builder, match->add_op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t result = loom_scalar_addi_result(match->add_op);
  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_fmai_build(
      &rewriter->builder, /*instance_flags=*/0,
      loom_scalar_muli_lhs(match->multiply_op),
      loom_scalar_muli_rhs(match->multiply_op), match->addend,
      loom_module_value_type(rewriter->module, result), match->add_op->location,
      &replacement_op));
  loom_value_id_t replacement = loom_scalar_fmai_result(replacement_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, match->add_op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, match->add_op, &replacement, 1));
  return loom_rewriter_erase(rewriter, match->multiply_op);
}

static iree_status_t loom_scalar_legalize_build_scalar_constant(
    loom_builder_t* builder, loom_location_id_t location, loom_type_t type,
    int64_t value, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(builder, loom_attr_i64(value),
                                                  type, location, &op));
  *out_value = loom_scalar_constant_result(op);
  return iree_ok_status();
}

static iree_status_t loom_scalar_legalize_build_i32_constant(
    loom_builder_t* builder, loom_location_id_t location, int64_t value,
    loom_value_id_t* out_value) {
  return loom_scalar_legalize_build_scalar_constant(
      builder, location, loom_type_scalar(LOOM_SCALAR_TYPE_I32), value,
      out_value);
}

static iree_status_t loom_scalar_legalize_build_binary_i32(
    loom_builder_t* builder, loom_location_id_t location, loom_op_kind_t kind,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_value_id_t* out_value) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* op = NULL;
  switch (kind) {
    case LOOM_OP_SCALAR_ADDI:
      (void)0;
      IREE_RETURN_IF_ERROR(loom_scalar_addi_build(builder, 0, lhs, rhs,
                                                  i32_type, location, &op));
      *out_value = loom_scalar_addi_result(op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_ANDI:
      (void)0;
      IREE_RETURN_IF_ERROR(
          loom_scalar_andi_build(builder, lhs, rhs, i32_type, location, &op));
      *out_value = loom_scalar_andi_result(op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_ORI:
      (void)0;
      IREE_RETURN_IF_ERROR(
          loom_scalar_ori_build(builder, lhs, rhs, i32_type, location, &op));
      *out_value = loom_scalar_ori_result(op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_SHLI:
      (void)0;
      IREE_RETURN_IF_ERROR(loom_scalar_shli_build(builder, 0, lhs, rhs,
                                                  i32_type, location, &op));
      *out_value = loom_scalar_shli_result(op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_SHRUI:
      (void)0;
      IREE_RETURN_IF_ERROR(
          loom_scalar_shrui_build(builder, lhs, rhs, i32_type, location, &op));
      *out_value = loom_scalar_shrui_result(op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_SUBI:
      (void)0;
      IREE_RETURN_IF_ERROR(loom_scalar_subi_build(builder, 0, lhs, rhs,
                                                  i32_type, location, &op));
      *out_value = loom_scalar_subi_result(op);
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported scalar i32 legalizer op");
  }
}

static iree_status_t loom_scalar_legalize_build_binary_i32_const_rhs(
    loom_builder_t* builder, loom_location_id_t location, loom_op_kind_t kind,
    loom_value_id_t lhs, int64_t rhs_value, loom_value_id_t* out_value) {
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_i32_constant(
      builder, location, rhs_value, &rhs));
  return loom_scalar_legalize_build_binary_i32(builder, location, kind, lhs,
                                               rhs, out_value);
}

static iree_status_t loom_scalar_legalize_build_cmpi_i32(
    loom_builder_t* builder, loom_location_id_t location,
    loom_scalar_cmpi_predicate_t predicate, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_cmpi_build(builder, predicate, lhs, rhs, location, &op));
  *out_value = loom_scalar_cmpi_result(op);
  return iree_ok_status();
}

static iree_status_t loom_scalar_legalize_build_cmpi_i32_const_rhs(
    loom_builder_t* builder, loom_location_id_t location,
    loom_scalar_cmpi_predicate_t predicate, loom_value_id_t lhs,
    int64_t rhs_value, loom_value_id_t* out_value) {
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_i32_constant(
      builder, location, rhs_value, &rhs));
  return loom_scalar_legalize_build_cmpi_i32(builder, location, predicate, lhs,
                                             rhs, out_value);
}

static iree_status_t loom_scalar_legalize_build_select(
    loom_builder_t* builder, loom_location_id_t location, loom_type_t type,
    loom_value_id_t condition, loom_value_id_t true_value,
    loom_value_id_t false_value, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_select_build(builder, condition, true_value,
                                             false_value, type, location, &op));
  *out_value = loom_scf_select_result(op);
  return iree_ok_status();
}

static iree_status_t loom_scalar_legalize_build_select_i32(
    loom_builder_t* builder, loom_location_id_t location,
    loom_value_id_t condition, loom_value_id_t true_value,
    loom_value_id_t false_value, loom_value_id_t* out_value) {
  return loom_scalar_legalize_build_select(
      builder, location, loom_type_scalar(LOOM_SCALAR_TYPE_I32), condition,
      true_value, false_value, out_value);
}

static iree_status_t loom_scalar_legalize_build_or_i32(
    loom_builder_t* builder, loom_location_id_t location, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_value) {
  return loom_scalar_legalize_build_binary_i32(
      builder, location, LOOM_OP_SCALAR_ORI, lhs, rhs, out_value);
}

typedef struct loom_scalar_fp8_result_format_t {
  // Scalar type represented by the format.
  loom_scalar_type_t type;
  // Total number of encoded bits.
  uint8_t bit_width;
  // Number of encoded exponent bits.
  uint8_t exponent_bits;
  // Number of encoded mantissa bits.
  uint8_t mantissa_bits;
  // Exponent bias used by the encoded type.
  uint8_t exponent_bias;
} loom_scalar_fp8_result_format_t;

static loom_scalar_fp8_result_format_t loom_scalar_fp8_result_format(
    loom_scalar_type_t type) {
  switch (type) {
    case LOOM_SCALAR_TYPE_F16:
      return (loom_scalar_fp8_result_format_t){
          .type = type,
          .bit_width = 16,
          .exponent_bits = 5,
          .mantissa_bits = 10,
          .exponent_bias = 15,
      };
    case LOOM_SCALAR_TYPE_BF16:
      return (loom_scalar_fp8_result_format_t){
          .type = type,
          .bit_width = 16,
          .exponent_bits = 8,
          .mantissa_bits = 7,
          .exponent_bias = 127,
      };
    case LOOM_SCALAR_TYPE_F32:
      return (loom_scalar_fp8_result_format_t){
          .type = type,
          .bit_width = 32,
          .exponent_bits = 8,
          .mantissa_bits = 23,
          .exponent_bias = 127,
      };
    default:
      IREE_ASSERT_UNREACHABLE("unsupported FP8 extension result type");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_scalar_legalize_build_fp8_leading_index(
    loom_builder_t* builder, loom_location_id_t location,
    const loom_scalar_type_fp8_format_t* format, loom_value_id_t mantissa,
    loom_value_id_t* out_value) {
  loom_value_id_t leading_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_i32_constant(
      builder, location, 0, &leading_index));
  for (uint8_t i = 1; i < format->mantissa_bits; ++i) {
    loom_value_id_t mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32_const_rhs(
        builder, location, LOOM_OP_SCALAR_ANDI, mantissa, INT64_C(1) << i,
        &mask));
    loom_value_id_t has_bit = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_cmpi_i32_const_rhs(
        builder, location, LOOM_SCALAR_CMPI_PREDICATE_NE, mask, 0, &has_bit));
    loom_value_id_t candidate = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_i32_constant(
        builder, location, i, &candidate));
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_select_i32(
        builder, location, has_bit, candidate, leading_index, &leading_index));
  }
  *out_value = leading_index;
  return iree_ok_status();
}

static iree_status_t loom_scalar_legalize_build_fp8_subnormal_bits(
    loom_builder_t* builder, loom_location_id_t location,
    const loom_scalar_type_fp8_format_t* format, loom_value_id_t sign_bits,
    const loom_scalar_fp8_result_format_t* result_format,
    loom_value_id_t mantissa, loom_value_id_t* out_value) {
  const int32_t source_min_exponent =
      1 - format->exponent_bias - format->mantissa_bits;
  const int32_t source_max_subnormal_exponent = -format->exponent_bias;
  const int32_t result_min_normal_exponent = 1 - result_format->exponent_bias;
  if (source_max_subnormal_exponent < result_min_normal_exponent) {
    // Every source subnormal remains subnormal in the result. All supported
    // result formats represent the source exactly, so this shift is integral.
    const int32_t shift = result_format->exponent_bias +
                          result_format->mantissa_bits - format->exponent_bias -
                          format->mantissa_bits;
    loom_value_id_t fraction_bits = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32_const_rhs(
        builder, location, LOOM_OP_SCALAR_SHLI, mantissa, shift,
        &fraction_bits));
    return loom_scalar_legalize_build_or_i32(builder, location, sign_bits,
                                             fraction_bits, out_value);
  }
  IREE_ASSERT(source_min_exponent >= result_min_normal_exponent,
              "FP8 subnormal range must not straddle the result normal range");

  loom_value_id_t leading_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_fp8_leading_index(
      builder, location, format, mantissa, &leading_index));
  loom_value_id_t exponent_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32_const_rhs(
      builder, location, LOOM_OP_SCALAR_ADDI, leading_index,
      result_format->exponent_bias + 1 - format->exponent_bias -
          format->mantissa_bits,
      &exponent_bits));
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32_const_rhs(
      builder, location, LOOM_OP_SCALAR_SHLI, exponent_bits,
      result_format->mantissa_bits, &exponent_bits));

  loom_value_id_t mantissa_shift = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_i32_constant(
      builder, location, result_format->mantissa_bits, &mantissa_shift));
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32(
      builder, location, LOOM_OP_SCALAR_SUBI, mantissa_shift, leading_index,
      &mantissa_shift));
  loom_value_id_t fraction_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32(
      builder, location, LOOM_OP_SCALAR_SHLI, mantissa, mantissa_shift,
      &fraction_bits));
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32_const_rhs(
      builder, location, LOOM_OP_SCALAR_ANDI, fraction_bits,
      (INT64_C(1) << result_format->mantissa_bits) - 1, &fraction_bits));

  loom_value_id_t nonzero_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_or_i32(
      builder, location, sign_bits, exponent_bits, &nonzero_bits));
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_or_i32(
      builder, location, nonzero_bits, fraction_bits, &nonzero_bits));
  loom_value_id_t is_zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_cmpi_i32_const_rhs(
      builder, location, LOOM_SCALAR_CMPI_PREDICATE_EQ, mantissa, 0, &is_zero));
  return loom_scalar_legalize_build_select_i32(
      builder, location, is_zero, sign_bits, nonzero_bits, out_value);
}

static iree_status_t loom_scalar_legalize_build_fp8_to_float(
    loom_builder_t* builder, loom_location_id_t location,
    loom_value_id_t byte_value, const loom_scalar_type_fp8_format_t* format,
    const loom_scalar_fp8_result_format_t* result_format,
    loom_value_id_t* out_value) {
  const int32_t sign_shift = format->exponent_bits + format->mantissa_bits;
  const int32_t sign_mask = 1 << sign_shift;
  const int32_t mantissa_mask = (1 << format->mantissa_bits) - 1;
  const int32_t exponent_mask = ((1 << format->exponent_bits) - 1)
                                << format->mantissa_bits;
  const int32_t exponent_all_ones = (1 << format->exponent_bits) - 1;

  loom_op_t* extend_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_extui_build(
      builder, byte_value, loom_type_scalar(LOOM_SCALAR_TYPE_I8),
      loom_type_scalar(LOOM_SCALAR_TYPE_I32), location, &extend_op));
  loom_value_id_t source_i32 = loom_scalar_extui_result(extend_op);

  loom_value_id_t sign_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32_const_rhs(
      builder, location, LOOM_OP_SCALAR_ANDI, source_i32, sign_mask,
      &sign_bits));
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32_const_rhs(
      builder, location, LOOM_OP_SCALAR_SHLI, sign_bits,
      result_format->bit_width - 1 - sign_shift, &sign_bits));

  loom_value_id_t exponent = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32_const_rhs(
      builder, location, LOOM_OP_SCALAR_ANDI, source_i32, exponent_mask,
      &exponent));
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32_const_rhs(
      builder, location, LOOM_OP_SCALAR_SHRUI, exponent, format->mantissa_bits,
      &exponent));

  loom_value_id_t mantissa = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32_const_rhs(
      builder, location, LOOM_OP_SCALAR_ANDI, source_i32, mantissa_mask,
      &mantissa));

  loom_value_id_t normal_exponent = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32_const_rhs(
      builder, location, LOOM_OP_SCALAR_ADDI, exponent,
      result_format->exponent_bias - format->exponent_bias, &normal_exponent));
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32_const_rhs(
      builder, location, LOOM_OP_SCALAR_SHLI, normal_exponent,
      result_format->mantissa_bits, &normal_exponent));
  loom_value_id_t normal_mantissa = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32_const_rhs(
      builder, location, LOOM_OP_SCALAR_SHLI, mantissa,
      result_format->mantissa_bits - format->mantissa_bits, &normal_mantissa));
  loom_value_id_t normal_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_or_i32(
      builder, location, sign_bits, normal_exponent, &normal_bits));
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_or_i32(
      builder, location, normal_bits, normal_mantissa, &normal_bits));

  loom_value_id_t subnormal_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_fp8_subnormal_bits(
      builder, location, format, sign_bits, result_format, mantissa,
      &subnormal_bits));
  loom_value_id_t exponent_is_zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_cmpi_i32_const_rhs(
      builder, location, LOOM_SCALAR_CMPI_PREDICATE_EQ, exponent, 0,
      &exponent_is_zero));
  loom_value_id_t finite_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_select_i32(
      builder, location, exponent_is_zero, subnormal_bits, normal_bits,
      &finite_bits));

  loom_value_id_t result_bits = finite_bits;
  loom_value_id_t exponent_is_top = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_cmpi_i32_const_rhs(
      builder, location, LOOM_SCALAR_CMPI_PREDICATE_EQ, exponent,
      exponent_all_ones, &exponent_is_top));
  loom_value_id_t quiet_nan_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_i32_constant(
      builder, location,
      (((INT64_C(1) << result_format->exponent_bits) - 1)
       << result_format->mantissa_bits) |
          (INT64_C(1) << (result_format->mantissa_bits - 1)),
      &quiet_nan_bits));
  if (format->special_policy == LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_IEEE) {
    loom_value_id_t infinity_bits = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_i32_constant(
        builder, location,
        ((INT64_C(1) << result_format->exponent_bits) - 1)
            << result_format->mantissa_bits,
        &infinity_bits));
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_or_i32(
        builder, location, sign_bits, infinity_bits, &infinity_bits));
    loom_value_id_t mantissa_is_zero = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_cmpi_i32_const_rhs(
        builder, location, LOOM_SCALAR_CMPI_PREDICATE_EQ, mantissa, 0,
        &mantissa_is_zero));
    loom_value_id_t top_bits = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_select_i32(
        builder, location, mantissa_is_zero, infinity_bits, quiet_nan_bits,
        &top_bits));
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_select_i32(
        builder, location, exponent_is_top, top_bits, finite_bits,
        &result_bits));
  } else {
    loom_value_id_t mantissa_is_nan = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_cmpi_i32_const_rhs(
        builder, location, LOOM_SCALAR_CMPI_PREDICATE_EQ, mantissa,
        mantissa_mask, &mantissa_is_nan));
    loom_value_id_t false_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_scalar_constant(
        builder, location, loom_type_scalar(LOOM_SCALAR_TYPE_I1), 0,
        &false_value));
    loom_value_id_t is_nan = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_select(
        builder, location, loom_type_scalar(LOOM_SCALAR_TYPE_I1),
        exponent_is_top, mantissa_is_nan, false_value, &is_nan));
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_select_i32(
        builder, location, is_nan, quiet_nan_bits, finite_bits, &result_bits));
  }

  loom_value_id_t carrier = result_bits;
  loom_type_t carrier_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  if (result_format->bit_width == 16) {
    carrier_type = loom_type_scalar(LOOM_SCALAR_TYPE_I16);
    loom_op_t* truncate_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_trunci_build(
        builder, result_bits, loom_type_scalar(LOOM_SCALAR_TYPE_I32),
        carrier_type, location, &truncate_op));
    carrier = loom_scalar_trunci_result(truncate_op);
  }
  loom_op_t* bitcast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(
      builder, carrier, carrier_type, loom_type_scalar(result_format->type),
      location, &bitcast_op));
  *out_value = loom_scalar_bitcast_result(bitcast_op);
  return iree_ok_status();
}

static iree_status_t loom_scalar_legalize_build_fp8_storage_byte(
    loom_builder_t* builder, loom_location_id_t location,
    loom_value_id_t fp8_value, loom_type_t fp8_type,
    loom_value_id_t* out_byte_value) {
  loom_op_t* bitcast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(
      builder, fp8_value, fp8_type, loom_type_scalar(LOOM_SCALAR_TYPE_I8),
      location, &bitcast_op));
  *out_byte_value = loom_scalar_bitcast_result(bitcast_op);
  return iree_ok_status();
}

bool loom_scalar_match_float8_to_bfloat_extension(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op) {
  (void)entry;
  const loom_value_id_t input = loom_scalar_extf_input(op);
  const loom_type_t input_type = loom_module_value_type(context->module, input);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_scalar_extf_result(op));
  return loom_type_is_scalar(input_type) &&
         loom_scalar_type_set_contains(
             LOOM_SCALAR_TYPE_SET_F8E4M3 | LOOM_SCALAR_TYPE_SET_F8E5M2,
             loom_type_element_type(input_type)) &&
         loom_type_is_scalar(result_type) &&
         loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_BF16;
}

iree_status_t loom_scalar_rewrite_float8_extension(
    loom_target_legalization_context_t* context, loom_op_t* op) {
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t input = loom_scalar_extf_input(op);
  const loom_type_t input_type = loom_module_value_type(context->module, input);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_scalar_extf_result(op));
  loom_scalar_type_fp8_format_t input_format = {0};
  const bool is_fp8 = loom_scalar_type_fp8_format(
      loom_type_element_type(input_type), &input_format);
  IREE_ASSERT(is_fp8, "matched FP8 extension must have an FP8 input");
  (void)is_fp8;
  const loom_scalar_fp8_result_format_t result_format =
      loom_scalar_fp8_result_format(loom_type_element_type(result_type));
  loom_value_id_t byte_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_fp8_storage_byte(
      builder, op->location, input, input_type, &byte_value));
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_fp8_to_float(
      builder, op->location, byte_value, &input_format, &result_format,
      &replacement));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  return iree_ok_status();
}

static bool loom_scalar_match_extf(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op) {
  (void)entry;
  const loom_type_t input_type =
      loom_module_value_type(context->module, loom_scalar_extf_input(op));
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_scalar_extf_result(op));
  if (!loom_type_is_scalar(input_type)) {
    return false;
  }
  const loom_scalar_type_t input_element = loom_type_element_type(input_type);
  const loom_scalar_type_t result_element = loom_type_element_type(result_type);
  if (result_element == LOOM_SCALAR_TYPE_F64) {
    return loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_FLOAT_LE16,
                                         input_element);
  }
  return loom_scalar_type_set_contains(
             LOOM_SCALAR_TYPE_SET_F8E4M3 | LOOM_SCALAR_TYPE_SET_F8E5M2,
             input_element) &&
         (result_element == LOOM_SCALAR_TYPE_F16 ||
          result_element == LOOM_SCALAR_TYPE_F32);
}

static iree_status_t loom_scalar_legalize_extf(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  const loom_type_t input_type =
      loom_module_value_type(context->module, loom_scalar_extf_input(op));
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_scalar_extf_result(op));

  if (loom_type_element_type(result_type) != LOOM_SCALAR_TYPE_F64) {
    IREE_RETURN_IF_ERROR(loom_scalar_rewrite_float8_extension(context, op));
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  // F32 represents every narrow source value exactly. Keep these edges as
  // ordinary extensions so the target can select its native F32/F64 paths.
  loom_op_t* widened = NULL;
  const uint8_t input_flags =
      op->instance_flags & LOOM_SCALAR_FLOATCONVERSIONFLAGS_DAZ;
  IREE_RETURN_IF_ERROR(loom_scalar_extf_build(
      &rewriter->builder, input_flags, loom_scalar_extf_input(op), input_type,
      f32_type, op->location, &widened));
  loom_op_t* converted = NULL;
  const uint8_t result_flags =
      op->instance_flags & LOOM_SCALAR_FLOATCONVERSIONFLAGS_FTZ;
  IREE_RETURN_IF_ERROR(loom_scalar_extf_build(
      &rewriter->builder, result_flags, loom_scalar_extf_result(widened),
      f32_type, result_type, op->location, &converted));
  const loom_value_id_t replacement = loom_scalar_extf_result(converted);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static iree_status_t loom_scalar_legalize_boolean_extension(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };

  const loom_value_id_t input = loom_op_operands(op)[0];
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_op_results(op)[0]);

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t true_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_scalar_constant(
      &rewriter->builder, op->location, result_type,
      loom_scalar_extsi_isa(op) ? -1 : 1, &true_value));
  loom_value_id_t false_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_scalar_constant(
      &rewriter->builder, op->location, result_type, 0, &false_value));
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_select(
      &rewriter->builder, op->location, result_type, input, true_value,
      false_value, &replacement));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

// Routes narrow integer extensions through the i32 carrier shared by targets
// without every logical width pair. Unsigned extension to i32 is the primitive
// carrier operation: signed values normalize with an i32 shift pair, smaller
// results truncate that carrier, and wider results extend it once more.
static iree_status_t loom_scalar_legalize_narrow_integer_extension(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };

  const bool signed_extension = loom_scalar_extsi_isa(op);
  const loom_value_id_t input = loom_op_operands(op)[0];
  const loom_type_t input_type = loom_module_value_type(context->module, input);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_op_results(op)[0]);
  const int32_t input_width =
      loom_scalar_type_bitwidth(loom_type_element_type(input_type));
  const int32_t result_width =
      loom_scalar_type_bitwidth(loom_type_element_type(result_type));
  const loom_type_t working_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // This is already the carrier primitive used by the decomposition. Leaving
  // it intact prevents a self-rewrite when a target reports it as unhandled.
  if (!signed_extension && result_width == 32) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* conversion = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_extui_build(
      builder, input, input_type, working_type, op->location, &conversion));
  loom_value_id_t value = loom_scalar_extui_result(conversion);
  if (signed_extension) {
    loom_value_id_t shift = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_i32_constant(
        builder, op->location, 32 - input_width, &shift));
    loom_op_t* shifted = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_shli_build(
        builder, 0, value, shift, working_type, op->location, &shifted));
    IREE_RETURN_IF_ERROR(
        loom_scalar_shrsi_build(builder, loom_scalar_shli_result(shifted),
                                shift, working_type, op->location, &shifted));
    value = loom_scalar_shrsi_result(shifted);
  }

  if (result_width < 32) {
    IREE_RETURN_IF_ERROR(loom_scalar_trunci_build(
        builder, value, working_type, result_type, op->location, &conversion));
    value = loom_scalar_trunci_result(conversion);
  } else if (result_width > 32) {
    if (signed_extension) {
      IREE_RETURN_IF_ERROR(loom_scalar_extsi_build(builder, value, working_type,
                                                   result_type, op->location,
                                                   &conversion));
      value = loom_scalar_extsi_result(conversion);
    } else {
      IREE_RETURN_IF_ERROR(loom_scalar_extui_build(builder, value, working_type,
                                                   result_type, op->location,
                                                   &conversion));
      value = loom_scalar_extui_result(conversion);
    }
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &value, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &value, 1));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

static iree_status_t loom_scalar_legalize_bitfield_extract(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  const bool signed_extract = op->kind == LOOM_OP_SCALAR_BITFIELD_EXTRACTS;
  loom_value_id_t value = signed_extract
                              ? loom_scalar_bitfield_extracts_source(op)
                              : loom_scalar_bitfield_extractu_source(op);
  const int64_t offset = signed_extract
                             ? loom_scalar_bitfield_extracts_offset(op)
                             : loom_scalar_bitfield_extractu_offset(op);
  const int64_t width = signed_extract
                            ? loom_scalar_bitfield_extracts_width(op)
                            : loom_scalar_bitfield_extractu_width(op);
  const loom_type_t source_type =
      loom_module_value_type(context->module, value);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_op_results(op)[0]);
  const int32_t source_width =
      loom_scalar_type_bitwidth(loom_type_element_type(source_type));
  const int32_t result_width =
      loom_scalar_type_bitwidth(loom_type_element_type(result_type));
  const int32_t working_width =
      width == source_width ? source_width : (source_width <= 32 ? 32 : 64);
  const loom_type_t working_type = working_width == source_width
                                       ? source_type
                                       : loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);

  // A full-width field needs only the result cast. Partial fields use a 32/64
  // bit carrier even when the source is narrow. Signed extraction moves the
  // field's sign bit to the carrier's sign bit before arithmetic shifting;
  // unsigned extraction uses a mask to expose the result's known zero bits.
  loom_op_t* scalar_op = NULL;
  if (source_width < working_width) {
    IREE_RETURN_IF_ERROR(loom_scalar_extui_build(
        builder, value, source_type, working_type, op->location, &scalar_op));
    value = loom_scalar_extui_result(scalar_op);
  }
  if (signed_extract && working_width - offset - width != 0) {
    loom_value_id_t shift = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_scalar_constant(
        builder, op->location, working_type, working_width - offset - width,
        &shift));
    IREE_RETURN_IF_ERROR(loom_scalar_shli_build(
        builder, 0, value, shift, working_type, op->location, &scalar_op));
    value = loom_scalar_shli_result(scalar_op);
  }
  const int64_t right_shift = signed_extract ? working_width - width : offset;
  if (right_shift != 0) {
    loom_value_id_t shift = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_scalar_constant(
        builder, op->location, working_type, right_shift, &shift));
    if (signed_extract) {
      IREE_RETURN_IF_ERROR(loom_scalar_shrsi_build(
          builder, value, shift, working_type, op->location, &scalar_op));
    } else {
      IREE_RETURN_IF_ERROR(loom_scalar_shrui_build(
          builder, value, shift, working_type, op->location, &scalar_op));
    }
    value = loom_op_results(scalar_op)[0];
  }
  if (!signed_extract && width < source_width - offset) {
    loom_value_id_t mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_scalar_constant(
        builder, op->location, working_type, UINT64_MAX >> (64 - width),
        &mask));
    IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
        builder, value, mask, working_type, op->location, &scalar_op));
    value = loom_scalar_andi_result(scalar_op);
  }
  if (result_width < working_width) {
    IREE_RETURN_IF_ERROR(loom_scalar_trunci_build(
        builder, value, working_type, result_type, op->location, &scalar_op));
    value = loom_scalar_trunci_result(scalar_op);
  } else if (result_width > working_width) {
    if (signed_extract) {
      IREE_RETURN_IF_ERROR(loom_scalar_extsi_build(
          builder, value, working_type, result_type, op->location, &scalar_op));
    } else {
      IREE_RETURN_IF_ERROR(loom_scalar_extui_build(
          builder, value, working_type, result_type, op->location, &scalar_op));
    }
    value = loom_op_results(scalar_op)[0];
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &value, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &value, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static iree_status_t loom_scalar_legalize_fmai(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_scalar_fmai_result(op));

  // Fused no-wrap facts do not prove that the decomposed multiply and add can
  // each carry independent no-wrap flags, so the reference split is
  // conservative.
  loom_op_t* product_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_muli_build(
      &rewriter->builder, 0, loom_scalar_fmai_a(op), loom_scalar_fmai_b(op),
      result_type, op->location, &product_op));
  loom_op_t* add_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_addi_build(
      &rewriter->builder, 0, loom_scalar_fmai_c(op),
      loom_scalar_muli_result(product_op), result_type, op->location, &add_op));

  loom_value_id_t replacement = loom_scalar_addi_result(add_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static iree_status_t loom_scalar_legalize_widen_integer_operands(
    loom_builder_t* builder, loom_op_t* op, bool signed_operands,
    loom_value_id_t operands[2]) {
  const loom_type_t source_type =
      loom_module_value_type(builder->module, loom_op_operands(op)[0]);
  const loom_type_t working_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  for (int i = 0; i < 2; ++i) {
    loom_op_t* extension = NULL;
    if (signed_operands) {
      IREE_RETURN_IF_ERROR(
          loom_scalar_extsi_build(builder, loom_op_operands(op)[i], source_type,
                                  working_type, op->location, &extension));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_scalar_extui_build(builder, loom_op_operands(op)[i], source_type,
                                  working_type, op->location, &extension));
    }
    operands[i] = loom_op_results(extension)[0];
  }
  return iree_ok_status();
}

// Decomposes a rotation without changing its logical integer domain. Both
// shift counts are masked independently so a zero rotation never materializes
// an undefined full-width shift. Target-native rotation contracts retain
// precedence over this portable recipe.
static iree_status_t loom_scalar_legalize_rotation(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t value = loom_op_operands(op)[0];
  const loom_value_id_t count = loom_op_operands(op)[1];
  const loom_type_t type = loom_module_value_type(context->module, value);
  const int32_t bit_width =
      loom_scalar_type_bitwidth(loom_type_element_type(type));

  loom_value_id_t replacement = value;
  if (bit_width != 1) {
    loom_op_t* mask_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
        builder, loom_attr_i64(bit_width - 1), type, op->location, &mask_op));
    loom_op_t* amount_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
        builder, count, loom_scalar_constant_result(mask_op), type,
        op->location, &amount_op));

    loom_op_t* zero_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
        builder, loom_attr_i64(0), type, op->location, &zero_op));
    loom_op_t* negative_amount_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_subi_build(
        builder, /*instance_flags=*/0, loom_scalar_constant_result(zero_op),
        loom_scalar_andi_result(amount_op), type, op->location,
        &negative_amount_op));
    loom_op_t* reverse_amount_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
        builder, loom_scalar_subi_result(negative_amount_op),
        loom_scalar_constant_result(mask_op), type, op->location,
        &reverse_amount_op));

    const loom_value_id_t amount = loom_scalar_andi_result(amount_op);
    const loom_value_id_t reverse_amount =
        loom_scalar_andi_result(reverse_amount_op);
    const bool rotate_left = loom_scalar_rotli_isa(op);
    loom_op_t* left_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_shli_build(
        builder, /*instance_flags=*/0, value,
        rotate_left ? amount : reverse_amount, type, op->location, &left_op));
    loom_op_t* right_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_shrui_build(
        builder, value, rotate_left ? reverse_amount : amount, type,
        op->location, &right_op));
    loom_op_t* combine_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_ori_build(
        builder, loom_scalar_shli_result(left_op),
        loom_scalar_shrui_result(right_op), type, op->location, &combine_op));
    replacement = loom_scalar_ori_result(combine_op);
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

// Targets with 32-bit arithmetic can implement narrower integer operations by
// widening their inputs and truncating each result. Keeping that truncation at
// the original operation boundary preserves wrapping before a later shift,
// division, or comparison consumes the value.
static iree_status_t loom_scalar_legalize_narrow_binary(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_op_results(op)[0]);

  // A left shift may sign-extend its narrow operands: every defined shift
  // amount is nonnegative and truncation preserves the low result bits. Unlike
  // zero extension, this form remains stable when generic canonicalization
  // moves a proven shift before an extension.
  const bool sign_extend_operands =
      op->kind == LOOM_OP_SCALAR_DIVSI || op->kind == LOOM_OP_SCALAR_REMSI ||
      op->kind == LOOM_OP_SCALAR_SHLI || op->kind == LOOM_OP_SCALAR_SHRSI;
  const loom_type_t working_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t operands[2];
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_widen_integer_operands(
      builder, op, sign_extend_operands, operands));

  // The registered binary families have no attributes. Narrow no-wrap flags
  // do not describe the widened intermediate, so its flags remain empty.
  loom_op_t* widened = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(builder, op->kind, 2, 1, 0, 0,
                                                0, op->location, &widened));
  loom_op_operands(widened)[0] = operands[0];
  loom_op_operands(widened)[1] = operands[1];
  IREE_RETURN_IF_ERROR(loom_builder_define_result(
      builder, working_type, &loom_op_results(widened)[0]));
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, widened));
  loom_op_t* truncation = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_trunci_build(
      builder, loom_op_results(widened)[0], working_type, result_type,
      op->location, &truncation));
  loom_value_id_t replacement = loom_scalar_trunci_result(truncation);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

// Preserves the shared quotient/remainder computation while widening narrow
// integer carriers. Targets that accept the i32 combined operation can select
// one native divide. The portable fallback computes the quotient once and
// reconstructs the remainder so every target can reuse its quotient recipe.
static iree_status_t loom_scalar_preserve_reconstructed_facts(
    loom_target_legalization_context_t* context,
    loom_value_id_t original_result, loom_value_id_t reconstructed_result,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_result) {
  *out_result = reconstructed_result;
  if (!context->fact_table) {
    return iree_ok_status();
  }
  loom_predicate_t predicates[2];
  const uint16_t predicate_count = loom_value_facts_export_integer_refinements(
      reconstructed_result,
      loom_value_fact_table_lookup(context->fact_table, reconstructed_result),
      loom_value_fact_table_lookup(context->fact_table, original_result),
      predicates, IREE_ARRAYSIZE(predicates));
  if (predicate_count == 0) {
    return iree_ok_status();
  }
  loom_op_t* assumption = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_assume_build(
      &context->rewriter->builder, &reconstructed_result, 1, predicates,
      predicate_count, &result_type, 1, location, &assumption));
  *out_result = loom_scalar_assume_results(assumption).values[0];
  return iree_ok_status();
}

static iree_status_t loom_scalar_legalize_divrem(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };

  const loom_value_id_t* old_results = loom_op_const_results(op);
  const loom_type_t result_type =
      loom_module_value_type(context->module, old_results[0]);
  const int32_t bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(result_type));
  const bool is_signed = op->kind == LOOM_OP_SCALAR_DIVREMSI;
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t replacements[2] = {LOOM_VALUE_ID_INVALID,
                                     LOOM_VALUE_ID_INVALID};

  if (bit_count < 32) {
    loom_value_id_t operands[2];
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_widen_integer_operands(
        builder, op, is_signed, operands));
    const loom_type_t working_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    const loom_type_t working_result_types[2] = {working_type, working_type};
    loom_op_t* widened = NULL;
    if (is_signed) {
      IREE_RETURN_IF_ERROR(loom_scalar_divremsi_build(
          builder, operands[0], operands[1], working_result_types, op->location,
          &widened));
    } else {
      IREE_RETURN_IF_ERROR(loom_scalar_divremui_build(
          builder, operands[0], operands[1], working_result_types, op->location,
          &widened));
    }
    for (uint16_t i = 0; i < IREE_ARRAYSIZE(replacements); ++i) {
      loom_op_t* truncation = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_trunci_build(
          builder, loom_op_results(widened)[i], working_type, result_type,
          op->location, &truncation));
      replacements[i] = loom_scalar_trunci_result(truncation);
    }
  } else {
    const loom_value_id_t lhs = loom_op_const_operands(op)[0];
    const loom_value_id_t rhs = loom_op_const_operands(op)[1];
    loom_op_t* quotient = NULL;
    if (is_signed) {
      IREE_RETURN_IF_ERROR(loom_scalar_divsi_build(
          builder, lhs, rhs, result_type, op->location, &quotient));
    } else {
      IREE_RETURN_IF_ERROR(loom_scalar_divui_build(
          builder, lhs, rhs, result_type, op->location, &quotient));
    }
    loom_op_t* product = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_muli_build(
        builder, /*instance_flags=*/0, loom_op_results(quotient)[0], rhs,
        result_type, op->location, &product));
    loom_op_t* remainder = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_subi_build(
        builder, /*instance_flags=*/0, lhs, loom_scalar_muli_result(product),
        result_type, op->location, &remainder));
    replacements[0] = loom_op_results(quotient)[0];
    IREE_RETURN_IF_ERROR(loom_scalar_preserve_reconstructed_facts(
        context, old_results[1], loom_scalar_subi_result(remainder),
        result_type, op->location, &replacements[1]));
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, IREE_ARRAYSIZE(replacements), checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacements, IREE_ARRAYSIZE(replacements)));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

// Count the source bits, not the carrier bits. The CTZ sentinel both supplies
// the narrow zero result and proves that the widened count input is nonzero.
static iree_status_t loom_scalar_legalize_narrow_bit_count(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t input = loom_op_operands(op)[0];
  const loom_type_t type = loom_module_value_type(context->module, input);
  const int32_t width = loom_scalar_type_bitwidth(loom_type_element_type(type));
  loom_value_id_t replacement = input;
  if (width == 1) {
    // A predicate has population count equal to itself; both zero counts are
    // its complement. No wider arithmetic or target count instruction is
    // needed.
    if (op->kind != LOOM_OP_SCALAR_CTPOPI) {
      loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_scalar_constant(
          builder, op->location, type, 0, &zero));
      loom_op_t* comparison = NULL;
      IREE_RETURN_IF_ERROR(
          loom_scalar_cmpi_build(builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, input,
                                 zero, op->location, &comparison));
      replacement = loom_scalar_cmpi_result(comparison);
    }
  } else {
    const loom_type_t working_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    loom_op_t* extension = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_extui_build(
        builder, input, type, working_type, op->location, &extension));
    loom_value_id_t operand = loom_scalar_extui_result(extension);
    if (op->kind == LOOM_OP_SCALAR_CTTZI) {
      IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32_const_rhs(
          builder, op->location, LOOM_OP_SCALAR_ORI, operand,
          INT64_C(1) << width, &operand));
    }
    loom_op_t* count = NULL;
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op(builder, op->kind, 1, 1, 0, 0,
                                                  0, op->location, &count));
    loom_op_operands(count)[0] = operand;
    IREE_RETURN_IF_ERROR(loom_builder_define_result(
        builder, working_type, &loom_op_results(count)[0]));
    IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, count));
    replacement = loom_op_results(count)[0];
    if (op->kind == LOOM_OP_SCALAR_CTLZI) {
      IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32_const_rhs(
          builder, op->location, LOOM_OP_SCALAR_SUBI, replacement, 32 - width,
          &replacement));
    }
    loom_op_t* truncation = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_trunci_build(
        builder, replacement, working_type, type, op->location, &truncation));
    replacement = loom_scalar_trunci_result(truncation);
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

// Composes signed-unary integer operations from target arithmetic primitives.
// One-bit negation and absolute value are identities under wrapping i1
// semantics. Narrow values use one signed i32 carrier so every intermediate
// observes the authored sign while truncation preserves the minimum-value
// wraparound case.
static iree_status_t loom_scalar_legalize_signed_unary(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t input = loom_op_const_operands(op)[0];
  const loom_type_t input_type = loom_module_value_type(context->module, input);
  const int32_t input_width =
      loom_scalar_type_bitwidth(loom_type_element_type(input_type));

  loom_value_id_t replacement = input;
  if (input_width > 1) {
    const int32_t working_width = input_width < 32 ? 32 : input_width;
    const loom_type_t working_type = loom_type_scalar(
        working_width == 32 ? LOOM_SCALAR_TYPE_I32 : LOOM_SCALAR_TYPE_I64);
    loom_value_id_t value = input;
    if (input_width < working_width) {
      loom_op_t* extension = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_extsi_build(
          builder, value, input_type, working_type, op->location, &extension));
      value = loom_scalar_extsi_result(extension);
    }

    if (loom_scalar_negi_isa(op)) {
      loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_scalar_constant(
          builder, op->location, working_type, 0, &zero));
      loom_op_t* subtraction = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_subi_build(
          builder, 0, zero, value, working_type, op->location, &subtraction));
      replacement = loom_scalar_subi_result(subtraction);
    } else {
      loom_value_id_t shift_amount = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_scalar_constant(
          builder, op->location, working_type, working_width - 1,
          &shift_amount));
      loom_op_t* shift = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_shrsi_build(
          builder, value, shift_amount, working_type, op->location, &shift));
      const loom_value_id_t sign = loom_scalar_shrsi_result(shift);
      loom_op_t* magnitude = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_xori_build(
          builder, value, sign, working_type, op->location, &magnitude));
      loom_op_t* subtraction = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_subi_build(
          builder, 0, loom_scalar_xori_result(magnitude), sign, working_type,
          op->location, &subtraction));
      replacement = loom_scalar_subi_result(subtraction);
    }

    if (input_width < working_width) {
      loom_op_t* truncation = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_trunci_build(builder, replacement,
                                                    working_type, input_type,
                                                    op->location, &truncation));
      replacement = loom_scalar_trunci_result(truncation);
    }
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static iree_status_t loom_scalar_legalize_narrow_cmpi(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  const loom_scalar_cmpi_predicate_t predicate = loom_scalar_cmpi_predicate(op);
  const bool signed_operands = predicate == LOOM_SCALAR_CMPI_PREDICATE_SLT ||
                               predicate == LOOM_SCALAR_CMPI_PREDICATE_SLE ||
                               predicate == LOOM_SCALAR_CMPI_PREDICATE_SGT ||
                               predicate == LOOM_SCALAR_CMPI_PREDICATE_SGE;
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t operands[2];
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_widen_integer_operands(
      builder, op, signed_operands, operands));
  loom_op_t* comparison = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
      builder, predicate, operands[0], operands[1], op->location, &comparison));
  loom_value_id_t replacement = loom_scalar_cmpi_result(comparison);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

// Native extrema contracts take precedence over this comparison/selection
// decomposition. Ordinary cleanup preserves it because extrema formation is
// restricted to the pre-legalization combine phase.
static iree_status_t loom_scalar_legalize_integer_extrema(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  const bool is_signed = loom_scalar_minsi_isa(op) || loom_scalar_maxsi_isa(op);
  const bool is_minimum =
      loom_scalar_minsi_isa(op) || loom_scalar_minui_isa(op);
  const loom_scalar_cmpi_predicate_t predicate =
      is_signed ? (is_minimum ? LOOM_SCALAR_CMPI_PREDICATE_SLT
                              : LOOM_SCALAR_CMPI_PREDICATE_SGT)
                : (is_minimum ? LOOM_SCALAR_CMPI_PREDICATE_ULT
                              : LOOM_SCALAR_CMPI_PREDICATE_UGT);
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t lhs = loom_op_operands(op)[0];
  const loom_value_id_t rhs = loom_op_operands(op)[1];
  const loom_type_t type = loom_module_value_type(context->module, lhs);
  loom_op_t* comparison = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(builder, predicate, lhs, rhs,
                                              op->location, &comparison));
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_select(
      builder, op->location, type, loom_scalar_cmpi_result(comparison), lhs,
      rhs, &replacement));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

typedef uint8_t loom_scalar_float_extremum_kind_t;
enum loom_scalar_float_extremum_kind_e {
  LOOM_SCALAR_FLOAT_EXTREMUM_MINIMUM = 0,
  LOOM_SCALAR_FLOAT_EXTREMUM_MAXIMUM = 1,
};

// Builds an exact F32 number-preferring extremum from terminal scalar
// operations. Ordered comparisons select unequal numeric operands. Equal
// finite operands have identical bits, except for opposing signed zeros: OR
// selects -0 for minimum and AND selects +0 for maximum. The final selects
// choose the numeric peer when exactly one operand is NaN.
static iree_status_t loom_scalar_legalize_build_f32_number_extremum(
    loom_builder_t* builder, loom_location_id_t location,
    loom_scalar_float_extremum_kind_t kind, uint8_t fastmath,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_value_id_t* out_result) {
  const loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* lhs_bits_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(
      builder, lhs, f32_type, i32_type, location, &lhs_bits_op));
  loom_op_t* rhs_bits_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(
      builder, rhs, f32_type, i32_type, location, &rhs_bits_op));
  loom_value_id_t tie_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32(
      builder, location,
      kind == LOOM_SCALAR_FLOAT_EXTREMUM_MINIMUM ? LOOM_OP_SCALAR_ORI
                                                 : LOOM_OP_SCALAR_ANDI,
      loom_scalar_bitcast_result(lhs_bits_op),
      loom_scalar_bitcast_result(rhs_bits_op), &tie_bits));
  loom_op_t* tie_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(builder, tie_bits, i32_type,
                                                 f32_type, location, &tie_op));
  const loom_value_id_t tie = loom_scalar_bitcast_result(tie_op);

  const loom_scalar_cmpf_predicate_t predicate =
      kind == LOOM_SCALAR_FLOAT_EXTREMUM_MINIMUM
          ? LOOM_SCALAR_CMPF_PREDICATE_OLT
          : LOOM_SCALAR_CMPF_PREDICATE_OGT;
  loom_op_t* lhs_ordered_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpf_build(builder, fastmath, predicate, lhs,
                                              rhs, location, &lhs_ordered_op));
  loom_op_t* rhs_ordered_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpf_build(builder, fastmath, predicate, rhs,
                                              lhs, location, &rhs_ordered_op));
  loom_value_id_t rhs_or_tie = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_select(
      builder, location, f32_type, loom_scalar_cmpf_result(rhs_ordered_op), rhs,
      tie, &rhs_or_tie));
  loom_value_id_t numeric = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_select(
      builder, location, f32_type, loom_scalar_cmpf_result(lhs_ordered_op), lhs,
      rhs_or_tie, &numeric));

  loom_op_t* lhs_nan_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpf_build(builder, fastmath,
                                              LOOM_SCALAR_CMPF_PREDICATE_UNE,
                                              lhs, lhs, location, &lhs_nan_op));
  loom_op_t* rhs_nan_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpf_build(builder, fastmath,
                                              LOOM_SCALAR_CMPF_PREDICATE_UNE,
                                              rhs, rhs, location, &rhs_nan_op));
  loom_value_id_t rhs_checked = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_select(
      builder, location, f32_type, loom_scalar_cmpf_result(rhs_nan_op), lhs,
      numeric, &rhs_checked));
  return loom_scalar_legalize_build_select(builder, location, f32_type,
                                           loom_scalar_cmpf_result(lhs_nan_op),
                                           rhs, rhs_checked, out_result);
}

iree_status_t loom_scalar_legalize_f32_number_extremum(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_f32_number_extremum(
      builder, op->location,
      loom_scalar_minnumf_isa(op) ? LOOM_SCALAR_FLOAT_EXTREMUM_MINIMUM
                                  : LOOM_SCALAR_FLOAT_EXTREMUM_MAXIMUM,
      op->instance_flags, loom_op_operands(op)[0], loom_op_operands(op)[1],
      &replacement));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

iree_status_t loom_scalar_legalize_f32_clamp(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  const loom_value_id_t value = loom_scalar_clampf_value(op);
  const loom_value_id_t lower = loom_scalar_clampf_lower(op);
  const loom_value_id_t upper = loom_scalar_clampf_upper(op);
  const uint8_t fastmath = op->instance_flags;
  loom_value_id_t lower_bounded = LOOM_VALUE_ID_INVALID;
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  switch (loom_scalar_clampf_mode(op)) {
    case LOOM_SCALAR_CLAMPF_MODE_ORDERED: {
      loom_op_t* below_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_cmpf_build(
          builder, fastmath, LOOM_SCALAR_CMPF_PREDICATE_OLT, value, lower,
          op->location, &below_op));
      IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_select(
          builder, op->location, type, loom_scalar_cmpf_result(below_op), lower,
          value, &lower_bounded));
      loom_op_t* above_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_cmpf_build(
          builder, fastmath, LOOM_SCALAR_CMPF_PREDICATE_OGT, lower_bounded,
          upper, op->location, &above_op));
      IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_select(
          builder, op->location, type, loom_scalar_cmpf_result(above_op), upper,
          lower_bounded, &replacement));
      break;
    }
    case LOOM_SCALAR_CLAMPF_MODE_NUMBER: {
      IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_f32_number_extremum(
          builder, op->location, LOOM_SCALAR_FLOAT_EXTREMUM_MAXIMUM, fastmath,
          value, lower, &lower_bounded));
      IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_f32_number_extremum(
          builder, op->location, LOOM_SCALAR_FLOAT_EXTREMUM_MINIMUM, fastmath,
          lower_bounded, upper, &replacement));
      break;
    }
    case LOOM_SCALAR_CLAMPF_MODE_IEEE: {
      loom_op_t* maximum_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_maximumf_build(
          builder, fastmath, value, lower, type, op->location, &maximum_op));
      loom_op_t* minimum_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_minimumf_build(
          builder, fastmath, loom_scalar_maximumf_result(maximum_op), upper,
          type, op->location, &minimum_op));
      replacement = loom_scalar_minimumf_result(minimum_op);
      break;
    }
    case LOOM_SCALAR_CLAMPF_MODE_COUNT_:
      IREE_ASSERT_UNREACHABLE("verified clamp has an invalid mode");
      IREE_BUILTIN_UNREACHABLE();
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

// F32 represents every numeric FP8, F16, and BF16 value exactly. Comparisons
// therefore preserve their predicate result, while extrema and clamps return
// an input value or a NaN that can be narrowed without intermediate rounding
// or a NaN payload guarantee.
static iree_status_t loom_scalar_legalize_exact_narrow_float(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_type_t type =
      loom_module_value_type(context->module, loom_op_operands(op)[0]);
  const loom_type_t working_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  const iree_host_size_t operand_count = loom_scalar_clampf_isa(op) ? 3 : 2;
  loom_value_id_t operands[3];
  for (iree_host_size_t i = 0; i < operand_count; ++i) {
    loom_op_t* extension = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_extf_build(
        builder, /*instance_flags=*/0, loom_op_operands(op)[i], type,
        working_type, op->location, &extension));
    operands[i] = loom_scalar_extf_result(extension);
  }

  loom_op_t* widened = NULL;
  if (loom_scalar_cmpf_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_scalar_cmpf_build(
        builder, op->instance_flags, loom_scalar_cmpf_predicate(op),
        operands[0], operands[1], op->location, &widened));
  } else if (loom_scalar_clampf_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_scalar_clampf_build(
        builder, loom_scalar_clampf_mode(op), op->instance_flags, operands[0],
        operands[1], operands[2], working_type, op->location, &widened));
  } else {
    // All four extrema have the same attribute-free binary shape and flags.
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op(builder, op->kind, 2, 1, 0, 0,
                                                  0, op->location, &widened));
    widened->instance_flags = op->instance_flags;
    loom_op_operands(widened)[0] = operands[0];
    loom_op_operands(widened)[1] = operands[1];
    IREE_RETURN_IF_ERROR(loom_builder_define_result(
        builder, working_type, &loom_op_results(widened)[0]));
    IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, widened));
  }

  loom_value_id_t replacement = loom_op_results(widened)[0];
  if (!loom_scalar_cmpf_isa(op)) {
    loom_op_t* truncation = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_fptrunc_build(
        builder, /*instance_flags=*/0, replacement, working_type, type,
        op->location, &truncation));
    replacement = loom_scalar_fptrunc_result(truncation);
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

// Number-preferring extrema already implement signed-zero ordering. An
// unordered comparison adds the IEEE NaN policy without changing numeric
// operands or exposing a NaN payload guarantee.
iree_status_t loom_scalar_legalize_ieee_extrema(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t lhs = loom_op_operands(op)[0];
  const loom_value_id_t rhs = loom_op_operands(op)[1];
  const loom_type_t type = loom_module_value_type(context->module, lhs);

  loom_op_t* number = NULL;
  if (loom_scalar_minimumf_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_scalar_minnumf_build(
        builder, op->instance_flags, lhs, rhs, type, op->location, &number));
  } else {
    IREE_RETURN_IF_ERROR(loom_scalar_maxnumf_build(
        builder, op->instance_flags, lhs, rhs, type, op->location, &number));
  }
  loom_op_t* unordered = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpf_build(
      builder, op->instance_flags, LOOM_SCALAR_CMPF_PREDICATE_UNO, lhs, rhs,
      op->location, &unordered));
  loom_op_t* nan = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(builder, loom_attr_f64(NAN),
                                                  type, op->location, &nan));
  loom_op_t* select = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_select_build(
      builder, loom_scalar_cmpf_result(unordered),
      loom_scalar_constant_result(nan), loom_op_results(number)[0], type,
      op->location, &select));
  const loom_value_id_t replacement = loom_scf_select_result(select);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

static iree_status_t loom_scalar_legalize_float_classification(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  const loom_value_id_t input = loom_op_operands(op)[0];
  const loom_type_t float_type = loom_module_value_type(context->module, input);
  loom_numeric_float_encoding_t encoding = {0};
  if (!loom_numeric_float_encoding(loom_type_element_type(float_type),
                                   &encoding)) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  if (op->kind == LOOM_OP_SCALAR_ISINFF &&
      encoding.special_layout == LOOM_NUMERIC_FLOAT_SPECIAL_LAYOUT_FINITE_NAN) {
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_scalar_constant(
        builder, op->location, loom_type_scalar(LOOM_SCALAR_TYPE_I1), 0,
        &replacement));
  } else {
    const loom_type_t integer_type = loom_type_scalar(encoding.integer_type);
    loom_op_t* bitcast_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(
        builder, input, float_type, integer_type, op->location, &bitcast_op));
    loom_value_id_t magnitude_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_scalar_constant(
        builder, op->location, integer_type, (int64_t)encoding.magnitude_mask,
        &magnitude_mask));
    loom_op_t* magnitude_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
        builder, loom_scalar_bitcast_result(bitcast_op), magnitude_mask,
        integer_type, op->location, &magnitude_op));
    loom_value_id_t special_magnitude = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_scalar_constant(
        builder, op->location, integer_type,
        (int64_t)encoding.special_magnitude, &special_magnitude));

    loom_scalar_cmpi_predicate_t predicate = LOOM_SCALAR_CMPI_PREDICATE_EQ;
    if (op->kind == LOOM_OP_SCALAR_ISNANF) {
      predicate =
          encoding.special_layout == LOOM_NUMERIC_FLOAT_SPECIAL_LAYOUT_IEEE
              ? LOOM_SCALAR_CMPI_PREDICATE_UGT
              : LOOM_SCALAR_CMPI_PREDICATE_EQ;
    } else if (op->kind == LOOM_OP_SCALAR_ISFINITEF) {
      predicate =
          encoding.special_layout == LOOM_NUMERIC_FLOAT_SPECIAL_LAYOUT_IEEE
              ? LOOM_SCALAR_CMPI_PREDICATE_ULT
              : LOOM_SCALAR_CMPI_PREDICATE_NE;
    }
    loom_op_t* comparison_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
        builder, predicate, loom_scalar_andi_result(magnitude_op),
        special_magnitude, op->location, &comparison_op));
    replacement = loom_scalar_cmpi_result(comparison_op);
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

static const loom_target_legalizer_rule_t kScalarLegalizerRules[] = {
    {
        .root_kind = LOOM_OP_SCALAR_NEGI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_INTEGER,
        .legalize = loom_scalar_legalize_signed_unary,
    },
    {
        .root_kind = LOOM_OP_SCALAR_ABSI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_INTEGER,
        .legalize = loom_scalar_legalize_signed_unary,
    },
    {
        .root_kind = LOOM_OP_SCALAR_CTLZI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I1 |
                                       LOOM_SCALAR_TYPE_SET_I8 |
                                       LOOM_SCALAR_TYPE_SET_I16,
        .legalize = loom_scalar_legalize_narrow_bit_count,
    },
    {
        .root_kind = LOOM_OP_SCALAR_CTTZI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I1 |
                                       LOOM_SCALAR_TYPE_SET_I8 |
                                       LOOM_SCALAR_TYPE_SET_I16,
        .legalize = loom_scalar_legalize_narrow_bit_count,
    },
    {
        .root_kind = LOOM_OP_SCALAR_CTPOPI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I1 |
                                       LOOM_SCALAR_TYPE_SET_I8 |
                                       LOOM_SCALAR_TYPE_SET_I16,
        .legalize = loom_scalar_legalize_narrow_bit_count,
    },
    {
        .root_kind = LOOM_OP_SCALAR_MINSI,
        .legalize = loom_scalar_legalize_integer_extrema,
    },
    {
        .root_kind = LOOM_OP_SCALAR_MAXSI,
        .legalize = loom_scalar_legalize_integer_extrema,
    },
    {
        .root_kind = LOOM_OP_SCALAR_MINUI,
        .legalize = loom_scalar_legalize_integer_extrema,
    },
    {
        .root_kind = LOOM_OP_SCALAR_MAXUI,
        .legalize = loom_scalar_legalize_integer_extrema,
    },
    {
        .root_kind = LOOM_OP_SCALAR_CLAMPF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_F32,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_UNHANDLED,
        .legalize = loom_scalar_legalize_f32_clamp,
    },
    {
        .root_kind = LOOM_OP_SCALAR_MINNUMF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_F32,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_UNHANDLED,
        .legalize = loom_scalar_legalize_f32_number_extremum,
    },
    {
        .root_kind = LOOM_OP_SCALAR_MAXNUMF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_F32,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_UNHANDLED,
        .legalize = loom_scalar_legalize_f32_number_extremum,
    },
    {
        .root_kind = LOOM_OP_SCALAR_CMPF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT_LE16,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_scalar_legalize_exact_narrow_float,
    },
    {
        .root_kind = LOOM_OP_SCALAR_CLAMPF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT_LE16,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_scalar_legalize_exact_narrow_float,
    },
    {
        .root_kind = LOOM_OP_SCALAR_MINNUMF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT_LE16,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_scalar_legalize_exact_narrow_float,
    },
    {
        .root_kind = LOOM_OP_SCALAR_MAXNUMF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT_LE16,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_scalar_legalize_exact_narrow_float,
    },
    {
        .root_kind = LOOM_OP_SCALAR_MINIMUMF,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_FLOAT_LE16 & ~LOOM_SCALAR_TYPE_SET_F16,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_scalar_legalize_exact_narrow_float,
    },
    {
        .root_kind = LOOM_OP_SCALAR_MAXIMUMF,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_FLOAT_LE16 & ~LOOM_SCALAR_TYPE_SET_F16,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_scalar_legalize_exact_narrow_float,
    },
    {
        .root_kind = LOOM_OP_SCALAR_MINIMUMF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_F16 |
                                       LOOM_SCALAR_TYPE_SET_F32 |
                                       LOOM_SCALAR_TYPE_SET_F64,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_scalar_legalize_ieee_extrema,
    },
    {
        .root_kind = LOOM_OP_SCALAR_ISNANF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_scalar_legalize_float_classification,
    },
    {
        .root_kind = LOOM_OP_SCALAR_ISINFF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_scalar_legalize_float_classification,
    },
    {
        .root_kind = LOOM_OP_SCALAR_ISFINITEF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_scalar_legalize_float_classification,
    },
    {
        .root_kind = LOOM_OP_SCALAR_MAXIMUMF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_F16 |
                                       LOOM_SCALAR_TYPE_SET_F32 |
                                       LOOM_SCALAR_TYPE_SET_F64,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_scalar_legalize_ieee_extrema,
    },
    {
        .root_kind = LOOM_OP_SCALAR_CMPI,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_I8 | LOOM_SCALAR_TYPE_SET_I16,
        .legalize = loom_scalar_legalize_narrow_cmpi,
    },
    {
        .root_kind = LOOM_OP_SCALAR_ADDI,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_I8 | LOOM_SCALAR_TYPE_SET_I16,
        .legalize = loom_scalar_legalize_narrow_binary,
    },
    {
        .root_kind = LOOM_OP_SCALAR_SUBI,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_I8 | LOOM_SCALAR_TYPE_SET_I16,
        .legalize = loom_scalar_legalize_narrow_binary,
    },
    {
        .root_kind = LOOM_OP_SCALAR_MULI,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_I8 | LOOM_SCALAR_TYPE_SET_I16,
        .legalize = loom_scalar_legalize_narrow_binary,
    },
    {
        .root_kind = LOOM_OP_SCALAR_DIVSI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I1 |
                                       LOOM_SCALAR_TYPE_SET_I8 |
                                       LOOM_SCALAR_TYPE_SET_I16,
        .legalize = loom_scalar_legalize_narrow_binary,
    },
    {
        .root_kind = LOOM_OP_SCALAR_DIVUI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I1 |
                                       LOOM_SCALAR_TYPE_SET_I8 |
                                       LOOM_SCALAR_TYPE_SET_I16,
        .legalize = loom_scalar_legalize_narrow_binary,
    },
    {
        .root_kind = LOOM_OP_SCALAR_DIVREMSI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_INTEGER,
        .legalize = loom_scalar_legalize_divrem,
    },
    {
        .root_kind = LOOM_OP_SCALAR_DIVREMUI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_INTEGER,
        .legalize = loom_scalar_legalize_divrem,
    },
    {
        .root_kind = LOOM_OP_SCALAR_REMSI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I1 |
                                       LOOM_SCALAR_TYPE_SET_I8 |
                                       LOOM_SCALAR_TYPE_SET_I16,
        .legalize = loom_scalar_legalize_narrow_binary,
    },
    {
        .root_kind = LOOM_OP_SCALAR_REMUI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I1 |
                                       LOOM_SCALAR_TYPE_SET_I8 |
                                       LOOM_SCALAR_TYPE_SET_I16,
        .legalize = loom_scalar_legalize_narrow_binary,
    },
    {
        .root_kind = LOOM_OP_SCALAR_ANDI,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_I8 | LOOM_SCALAR_TYPE_SET_I16,
        .legalize = loom_scalar_legalize_narrow_binary,
    },
    {
        .root_kind = LOOM_OP_SCALAR_ORI,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_I8 | LOOM_SCALAR_TYPE_SET_I16,
        .legalize = loom_scalar_legalize_narrow_binary,
    },
    {
        .root_kind = LOOM_OP_SCALAR_XORI,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_I8 | LOOM_SCALAR_TYPE_SET_I16,
        .legalize = loom_scalar_legalize_narrow_binary,
    },
    {
        .root_kind = LOOM_OP_SCALAR_SHLI,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_I8 | LOOM_SCALAR_TYPE_SET_I16,
        .legalize = loom_scalar_legalize_narrow_binary,
    },
    {
        .root_kind = LOOM_OP_SCALAR_SHRSI,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_I8 | LOOM_SCALAR_TYPE_SET_I16,
        .legalize = loom_scalar_legalize_narrow_binary,
    },
    {
        .root_kind = LOOM_OP_SCALAR_SHRUI,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_I8 | LOOM_SCALAR_TYPE_SET_I16,
        .legalize = loom_scalar_legalize_narrow_binary,
    },
    {
        .root_kind = LOOM_OP_SCALAR_ROTLI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_INTEGER,
        .legalize = loom_scalar_legalize_rotation,
    },
    {
        .root_kind = LOOM_OP_SCALAR_ROTRI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_INTEGER,
        .legalize = loom_scalar_legalize_rotation,
    },
    {
        .root_kind = LOOM_OP_SCALAR_BITFIELD_EXTRACTU,
        .legalize = loom_scalar_legalize_bitfield_extract,
    },
    {
        .root_kind = LOOM_OP_SCALAR_BITFIELD_EXTRACTS,
        .legalize = loom_scalar_legalize_bitfield_extract,
    },
    {
        .root_kind = LOOM_OP_SCALAR_EXTF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT_LE16,
        .match = loom_scalar_match_extf,
        .legalize = loom_scalar_legalize_extf,
    },
    {
        .root_kind = LOOM_OP_SCALAR_EXTSI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I1,
        .legalize = loom_scalar_legalize_boolean_extension,
    },
    {
        .root_kind = LOOM_OP_SCALAR_EXTSI,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_I8 | LOOM_SCALAR_TYPE_SET_I16,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_scalar_legalize_narrow_integer_extension,
    },
    {
        .root_kind = LOOM_OP_SCALAR_EXTUI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I1,
        .legalize = loom_scalar_legalize_boolean_extension,
    },
    {
        .root_kind = LOOM_OP_SCALAR_EXTUI,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_I8 | LOOM_SCALAR_TYPE_SET_I16,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_scalar_legalize_narrow_integer_extension,
    },
    {
        .root_kind = LOOM_OP_SCALAR_FMAI,
        .legalize = loom_scalar_legalize_fmai,
    },
};

static const loom_target_legalizer_provider_t kScalarLegalizerProvider = {
    .name = IREE_SVL("scalar"),
    .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE,
    .rules = kScalarLegalizerRules,
    .rule_count = IREE_ARRAYSIZE(kScalarLegalizerRules),
};

const loom_target_legalizer_provider_t* loom_scalar_target_legalizer_provider(
    void) {
  return &kScalarLegalizerProvider;
}
