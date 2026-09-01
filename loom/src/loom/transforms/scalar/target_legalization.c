// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scalar/target_legalization.h"

#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/rewrite/rewriter.h"

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

static iree_status_t loom_scalar_legalize_build_binary_integer(
    loom_builder_t* builder, loom_location_id_t location, loom_op_kind_t kind,
    loom_type_t type, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  switch (kind) {
    case LOOM_OP_SCALAR_ADDI:
      (void)0;
      IREE_RETURN_IF_ERROR(
          loom_scalar_addi_build(builder, 0, lhs, rhs, type, location, &op));
      *out_value = loom_scalar_addi_result(op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_SUBI:
      (void)0;
      IREE_RETURN_IF_ERROR(
          loom_scalar_subi_build(builder, 0, lhs, rhs, type, location, &op));
      *out_value = loom_scalar_subi_result(op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_MULI:
      (void)0;
      IREE_RETURN_IF_ERROR(
          loom_scalar_muli_build(builder, 0, lhs, rhs, type, location, &op));
      *out_value = loom_scalar_muli_result(op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_MINSI:
      (void)0;
      IREE_RETURN_IF_ERROR(
          loom_scalar_minsi_build(builder, lhs, rhs, type, location, &op));
      *out_value = loom_scalar_minsi_result(op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_MAXSI:
      (void)0;
      IREE_RETURN_IF_ERROR(
          loom_scalar_maxsi_build(builder, lhs, rhs, type, location, &op));
      *out_value = loom_scalar_maxsi_result(op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_MINUI:
      (void)0;
      IREE_RETURN_IF_ERROR(
          loom_scalar_minui_build(builder, lhs, rhs, type, location, &op));
      *out_value = loom_scalar_minui_result(op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_MAXUI:
      (void)0;
      IREE_RETURN_IF_ERROR(
          loom_scalar_maxui_build(builder, lhs, rhs, type, location, &op));
      *out_value = loom_scalar_maxui_result(op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_ANDI:
      (void)0;
      IREE_RETURN_IF_ERROR(
          loom_scalar_andi_build(builder, lhs, rhs, type, location, &op));
      *out_value = loom_scalar_andi_result(op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_ORI:
      (void)0;
      IREE_RETURN_IF_ERROR(
          loom_scalar_ori_build(builder, lhs, rhs, type, location, &op));
      *out_value = loom_scalar_ori_result(op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_XORI:
      (void)0;
      IREE_RETURN_IF_ERROR(
          loom_scalar_xori_build(builder, lhs, rhs, type, location, &op));
      *out_value = loom_scalar_xori_result(op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_SHLI:
      (void)0;
      IREE_RETURN_IF_ERROR(
          loom_scalar_shli_build(builder, 0, lhs, rhs, type, location, &op));
      *out_value = loom_scalar_shli_result(op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_SHRSI:
      (void)0;
      IREE_RETURN_IF_ERROR(
          loom_scalar_shrsi_build(builder, lhs, rhs, type, location, &op));
      *out_value = loom_scalar_shrsi_result(op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_SHRUI:
      (void)0;
      IREE_RETURN_IF_ERROR(
          loom_scalar_shrui_build(builder, lhs, rhs, type, location, &op));
      *out_value = loom_scalar_shrui_result(op);
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported scalar integer legalizer op");
  }
}

static iree_status_t loom_scalar_legalize_build_binary_i32(
    loom_builder_t* builder, loom_location_id_t location, loom_op_kind_t kind,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_value_id_t* out_value) {
  return loom_scalar_legalize_build_binary_integer(
      builder, location, kind, loom_type_scalar(LOOM_SCALAR_TYPE_I32), lhs, rhs,
      out_value);
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

static iree_status_t loom_scalar_legalize_build_cmpi_integer(
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
  return loom_scalar_legalize_build_cmpi_integer(builder, location, predicate,
                                                 lhs, rhs, out_value);
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

static bool loom_scalar_legalize_is_narrow_integer(loom_type_t type) {
  if (!loom_type_is_scalar(type)) return false;
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  const int32_t bit_width = loom_scalar_type_bitwidth(scalar_type);
  return loom_scalar_type_is_integer(scalar_type) && bit_width > 1 &&
         bit_width < 32;
}

static iree_status_t loom_scalar_legalize_extend_integer(
    loom_builder_t* builder, loom_location_id_t location,
    loom_value_id_t source, loom_type_t source_type, bool signed_extension,
    loom_value_id_t* out_result) {
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* op = NULL;
  if (signed_extension) {
    IREE_RETURN_IF_ERROR(loom_scalar_extsi_build(builder, source, source_type,
                                                 i32_type, location, &op));
    *out_result = loom_scalar_extsi_result(op);
  } else {
    IREE_RETURN_IF_ERROR(loom_scalar_extui_build(builder, source, source_type,
                                                 i32_type, location, &op));
    *out_result = loom_scalar_extui_result(op);
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_target_legalize_narrow_integer_binary_reference(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };

  const loom_value_id_t source_result = loom_op_const_results(op)[0];
  const loom_type_t source_type =
      loom_module_value_type(context->module, source_result);
  if (!loom_scalar_legalize_is_narrow_integer(source_type)) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t* source_operands = loom_op_const_operands(op);
  // Modular arithmetic and raw bit operations use zero-extended carriers.
  // Signed min/max and the source of an arithmetic right shift require signed
  // carriers. Truncation restores the authored width in every case.
  const bool signed_lhs = op->kind == LOOM_OP_SCALAR_MINSI ||
                          op->kind == LOOM_OP_SCALAR_MAXSI ||
                          op->kind == LOOM_OP_SCALAR_SHRSI;
  const bool signed_rhs =
      op->kind == LOOM_OP_SCALAR_MINSI || op->kind == LOOM_OP_SCALAR_MAXSI;
  loom_value_id_t lhs_i32 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_extend_integer(
      &rewriter->builder, op->location, source_operands[0], source_type,
      signed_lhs, &lhs_i32));
  loom_value_id_t rhs_i32 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_extend_integer(
      &rewriter->builder, op->location, source_operands[1], source_type,
      signed_rhs, &rhs_i32));

  loom_value_id_t wide_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32(
      &rewriter->builder, op->location, op->kind, lhs_i32, rhs_i32,
      &wide_result));
  loom_op_t* truncate_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_trunci_build(
      &rewriter->builder, wide_result, loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      source_type, op->location, &truncate_op));
  loom_value_id_t replacement = loom_scalar_trunci_result(truncate_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
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
    loom_value_id_t mantissa, loom_value_id_t* out_value) {
  loom_value_id_t leading_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_fp8_leading_index(
      builder, location, format, mantissa, &leading_index));
  loom_value_id_t exponent_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32_const_rhs(
      builder, location, LOOM_OP_SCALAR_ADDI, leading_index,
      128 - format->exponent_bias - format->mantissa_bits, &exponent_bits));
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32_const_rhs(
      builder, location, LOOM_OP_SCALAR_SHLI, exponent_bits, 23,
      &exponent_bits));

  loom_value_id_t mantissa_shift = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_i32_constant(
      builder, location, 23, &mantissa_shift));
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32(
      builder, location, LOOM_OP_SCALAR_SUBI, mantissa_shift, leading_index,
      &mantissa_shift));
  loom_value_id_t fraction_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32(
      builder, location, LOOM_OP_SCALAR_SHLI, mantissa, mantissa_shift,
      &fraction_bits));
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32_const_rhs(
      builder, location, LOOM_OP_SCALAR_ANDI, fraction_bits, 0x007FFFFF,
      &fraction_bits));

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

static iree_status_t loom_scalar_legalize_build_fp8_to_f32(
    loom_builder_t* builder, loom_location_id_t location,
    loom_value_id_t byte_value, const loom_scalar_type_fp8_format_t* format,
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
      builder, location, LOOM_OP_SCALAR_SHLI, sign_bits, 31 - sign_shift,
      &sign_bits));

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
      127 - format->exponent_bias, &normal_exponent));
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32_const_rhs(
      builder, location, LOOM_OP_SCALAR_SHLI, normal_exponent, 23,
      &normal_exponent));
  loom_value_id_t normal_mantissa = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_i32_const_rhs(
      builder, location, LOOM_OP_SCALAR_SHLI, mantissa,
      23 - format->mantissa_bits, &normal_mantissa));
  loom_value_id_t normal_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_or_i32(
      builder, location, sign_bits, normal_exponent, &normal_bits));
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_or_i32(
      builder, location, normal_bits, normal_mantissa, &normal_bits));

  loom_value_id_t subnormal_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_fp8_subnormal_bits(
      builder, location, format, sign_bits, mantissa, &subnormal_bits));
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
      builder, location, 0x7FC00000, &quiet_nan_bits));
  if (format->special_policy == LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_IEEE) {
    loom_value_id_t infinity_bits = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_i32_constant(
        builder, location, 0x7F800000, &infinity_bits));
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

  loom_op_t* bitcast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(
      builder, result_bits, loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      loom_type_scalar(LOOM_SCALAR_TYPE_F32), location, &bitcast_op));
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

static iree_status_t loom_scalar_legalize_build_bitfield_carrier(
    loom_builder_t* builder, loom_location_id_t location,
    loom_value_id_t source, loom_type_t source_type,
    loom_type_t* out_carrier_type, loom_value_id_t* out_carrier) {
  const int32_t source_width =
      loom_scalar_type_bitwidth(loom_type_element_type(source_type));
  const loom_type_t carrier_type = loom_type_scalar(
      source_width > 32 ? LOOM_SCALAR_TYPE_I64 : LOOM_SCALAR_TYPE_I32);
  *out_carrier_type = carrier_type;
  if (loom_type_equal(source_type, carrier_type)) {
    *out_carrier = source;
    return iree_ok_status();
  }

  // Bitfield extraction treats the source as raw bits. Zero-extension into a
  // canonical carrier preserves those bits while keeping the reference
  // sequence on the i32/i64 integer baseline shared by targets.
  loom_op_t* extend_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_extui_build(
      builder, source, source_type, carrier_type, location, &extend_op));
  *out_carrier = loom_scalar_extui_result(extend_op);
  return iree_ok_status();
}

static iree_status_t loom_scalar_legalize_build_bitfield_result(
    loom_builder_t* builder, loom_location_id_t location,
    loom_value_id_t carrier, loom_type_t carrier_type, loom_type_t result_type,
    bool signed_extract, loom_value_id_t* out_result) {
  if (loom_type_equal(carrier_type, result_type)) {
    *out_result = carrier;
    return iree_ok_status();
  }

  const loom_scalar_type_t result_scalar_type =
      loom_type_element_type(result_type);
  if (result_scalar_type == LOOM_SCALAR_TYPE_I1) {
    loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_scalar_constant(
        builder, location, carrier_type, 0, &zero));
    return loom_scalar_legalize_build_cmpi_integer(
        builder, location, LOOM_SCALAR_CMPI_PREDICATE_NE, carrier, zero,
        out_result);
  }

  const int32_t result_width = loom_scalar_type_bitwidth(result_scalar_type);
  const int32_t carrier_width =
      loom_scalar_type_bitwidth(loom_type_element_type(carrier_type));
  loom_op_t* cast_op = NULL;
  if (carrier_width > result_width) {
    IREE_RETURN_IF_ERROR(loom_scalar_trunci_build(
        builder, carrier, carrier_type, result_type, location, &cast_op));
    *out_result = loom_scalar_trunci_result(cast_op);
  } else if (signed_extract) {
    IREE_RETURN_IF_ERROR(loom_scalar_extsi_build(
        builder, carrier, carrier_type, result_type, location, &cast_op));
    *out_result = loom_scalar_extsi_result(cast_op);
  } else {
    IREE_RETURN_IF_ERROR(loom_scalar_extui_build(
        builder, carrier, carrier_type, result_type, location, &cast_op));
    *out_result = loom_scalar_extui_result(cast_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_scalar_legalize_bitfield_extract(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_value_id_t source, loom_value_id_t result, int64_t offset,
    int64_t width, bool signed_extract,
    loom_target_legalizer_result_t* out_result) {
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };

  const loom_type_t source_type =
      loom_module_value_type(context->module, source);
  const loom_type_t result_type =
      loom_module_value_type(context->module, result);
  if (!loom_type_is_scalar(source_type) || !loom_type_is_scalar(result_type) ||
      !loom_scalar_type_is_integer(loom_type_element_type(source_type)) ||
      !loom_scalar_type_is_integer(loom_type_element_type(result_type))) {
    return iree_ok_status();
  }
  const int32_t source_width =
      loom_scalar_type_bitwidth(loom_type_element_type(source_type));
  const int32_t result_width =
      loom_scalar_type_bitwidth(loom_type_element_type(result_type));
  if (offset < 0 || offset >= source_width || width <= 0 ||
      width > source_width - offset || width > result_width) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);

  loom_type_t carrier_type = loom_type_none();
  loom_value_id_t carrier = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_bitfield_carrier(
      &rewriter->builder, op->location, source, source_type, &carrier_type,
      &carrier));
  const int32_t carrier_width =
      loom_scalar_type_bitwidth(loom_type_element_type(carrier_type));

  loom_value_id_t extracted = LOOM_VALUE_ID_INVALID;
  if (signed_extract) {
    loom_value_id_t left_shift = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_scalar_constant(
        &rewriter->builder, op->location, carrier_type,
        carrier_width - offset - width, &left_shift));
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_integer(
        &rewriter->builder, op->location, LOOM_OP_SCALAR_SHLI, carrier_type,
        carrier, left_shift, &extracted));

    loom_value_id_t right_shift = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_scalar_constant(
        &rewriter->builder, op->location, carrier_type, carrier_width - width,
        &right_shift));
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_integer(
        &rewriter->builder, op->location, LOOM_OP_SCALAR_SHRSI, carrier_type,
        extracted, right_shift, &extracted));
  } else {
    loom_value_id_t right_shift = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_scalar_constant(
        &rewriter->builder, op->location, carrier_type, offset, &right_shift));
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_integer(
        &rewriter->builder, op->location, LOOM_OP_SCALAR_SHRUI, carrier_type,
        carrier, right_shift, &extracted));

    const int64_t mask = width == 64
                             ? INT64_C(-1)
                             : (int64_t)((UINT64_C(1) << width) - UINT64_C(1));
    loom_value_id_t mask_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_scalar_constant(
        &rewriter->builder, op->location, carrier_type, mask, &mask_value));
    IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_binary_integer(
        &rewriter->builder, op->location, LOOM_OP_SCALAR_ANDI, carrier_type,
        extracted, mask_value, &extracted));
  }

  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_bitfield_result(
      &rewriter->builder, op->location, extracted, carrier_type, result_type,
      signed_extract, &replacement));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static iree_status_t loom_scalar_legalize_bitfield_extractu(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  return loom_scalar_legalize_bitfield_extract(
      context, op, loom_scalar_bitfield_extractu_source(op),
      loom_scalar_bitfield_extractu_result(op),
      loom_scalar_bitfield_extractu_offset(op),
      loom_scalar_bitfield_extractu_width(op), /*signed_extract=*/false,
      out_result);
}

static iree_status_t loom_scalar_legalize_bitfield_extracts(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  return loom_scalar_legalize_bitfield_extract(
      context, op, loom_scalar_bitfield_extracts_source(op),
      loom_scalar_bitfield_extracts_result(op),
      loom_scalar_bitfield_extracts_offset(op),
      loom_scalar_bitfield_extracts_width(op), /*signed_extract=*/true,
      out_result);
}

static iree_status_t loom_scalar_legalize_extf(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };

  const loom_type_t input_type =
      loom_module_value_type(context->module, loom_scalar_extf_input(op));
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_scalar_extf_result(op));
  loom_scalar_type_fp8_format_t format = {0};
  if (!loom_type_is_scalar(input_type) ||
      !loom_scalar_type_fp8_format(loom_type_element_type(input_type),
                                   &format) ||
      !loom_type_equal(result_type, loom_type_scalar(LOOM_SCALAR_TYPE_F32))) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t byte_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_fp8_storage_byte(
      &rewriter->builder, op->location, loom_scalar_extf_input(op), input_type,
      &byte_value));
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_fp8_to_f32(
      &rewriter->builder, op->location, byte_value, &format, &replacement));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static iree_status_t loom_scalar_legalize_extui(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };

  const loom_type_t input_type =
      loom_module_value_type(context->module, loom_scalar_extui_input(op));
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_scalar_extui_result(op));
  if (!loom_type_equal(input_type, loom_type_scalar(LOOM_SCALAR_TYPE_I1)) ||
      !loom_type_is_scalar(result_type) ||
      !loom_scalar_type_is_integer(loom_type_element_type(result_type)) ||
      loom_scalar_type_bitwidth(loom_type_element_type(result_type)) <= 1) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t true_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_scalar_constant(
      &rewriter->builder, op->location, result_type, 1, &true_value));
  loom_value_id_t false_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_scalar_constant(
      &rewriter->builder, op->location, result_type, 0, &false_value));
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_legalize_build_select(
      &rewriter->builder, op->location, result_type,
      loom_scalar_extui_input(op), true_value, false_value, &replacement));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
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

static const loom_target_legalizer_rule_t kScalarLegalizerRules[] = {
    {
        .root_kind = LOOM_OP_SCALAR_BITFIELD_EXTRACTU,
        .legalize = loom_scalar_legalize_bitfield_extractu,
    },
    {
        .root_kind = LOOM_OP_SCALAR_BITFIELD_EXTRACTS,
        .legalize = loom_scalar_legalize_bitfield_extracts,
    },
    {
        .root_kind = LOOM_OP_SCALAR_EXTF,
        .legalize = loom_scalar_legalize_extf,
    },
    {
        .root_kind = LOOM_OP_SCALAR_EXTUI,
        .legalize = loom_scalar_legalize_extui,
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
