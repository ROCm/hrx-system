// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/compare.h"
#include "loom/ops/scalar/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/math.h"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static loom_type_t loom_scalar_single_result_type(loom_rewriter_t* rewriter,
                                                  const loom_op_t* op) {
  return loom_module_value_type(rewriter->module, loom_op_const_results(op)[0]);
}

static loom_op_t* loom_scalar_defining_op(loom_rewriter_t* rewriter,
                                          loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      (iree_host_size_t)value_id >= rewriter->module->values.count) {
    return NULL;
  }
  loom_value_t* value = loom_module_value(rewriter->module, value_id);
  if (loom_value_is_block_arg(value)) return NULL;
  return loom_value_def_op(value);
}

static bool loom_scalar_op_has_no_instance_flags(const loom_op_t* op) {
  return op->instance_flags == 0;
}

static bool loom_scalar_query_exact_i64(loom_rewriter_t* rewriter,
                                        loom_value_id_t value_id,
                                        int64_t* out_value) {
  loom_value_facts_t facts = loom_rewriter_value_facts(rewriter, value_id);
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

static bool loom_scalar_value_facts_are_exact_i64(loom_rewriter_t* rewriter,
                                                  loom_value_id_t value_id,
                                                  int64_t expected_value) {
  int64_t actual_value = 0;
  return loom_scalar_query_exact_i64(rewriter, value_id, &actual_value) &&
         actual_value == expected_value;
}

static bool loom_scalar_values_are_same_or_same_exact_i64(
    loom_rewriter_t* rewriter, loom_value_id_t lhs, loom_value_id_t rhs) {
  if (lhs == rhs) return true;
  int64_t lhs_value = 0;
  int64_t rhs_value = 0;
  return loom_scalar_query_exact_i64(rewriter, lhs, &lhs_value) &&
         loom_scalar_query_exact_i64(rewriter, rhs, &rhs_value) &&
         lhs_value == rhs_value;
}

static bool loom_scalar_value_facts_are_non_negative(loom_rewriter_t* rewriter,
                                                     loom_value_id_t value_id) {
  return loom_value_facts_is_non_negative(
      loom_rewriter_value_facts(rewriter, value_id));
}

static bool loom_scalar_value_facts_are_non_zero(loom_rewriter_t* rewriter,
                                                 loom_value_id_t value_id) {
  return loom_value_facts_is_non_zero(
      loom_rewriter_value_facts(rewriter, value_id));
}

static bool loom_scalar_query_exact_f64(loom_rewriter_t* rewriter,
                                        loom_value_id_t value_id,
                                        double* out_value) {
  loom_value_facts_t facts = loom_rewriter_value_facts(rewriter, value_id);
  if (!loom_value_facts_is_exact(facts) || !loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = loom_value_facts_as_f64(facts);
  return true;
}

static bool loom_scalar_value_facts_are_exact_f64(loom_rewriter_t* rewriter,
                                                  loom_value_id_t value_id,
                                                  double expected_value) {
  double actual_value = 0.0;
  return loom_scalar_query_exact_f64(rewriter, value_id, &actual_value) &&
         actual_value == expected_value;
}

static bool loom_scalar_fastmath_has_all(const loom_op_t* op, uint8_t flags) {
  return (op->instance_flags & flags) == flags;
}

static bool loom_scalar_type_is_i1(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1;
}

static bool loom_scalar_type_is_integer_scalar(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_scalar_type_is_integer(loom_type_element_type(type));
}

static bool loom_scalar_type_is_float_scalar(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_scalar_type_is_float(loom_type_element_type(type));
}

static bool loom_scalar_type_query_bitwidth(loom_type_t type,
                                            int32_t* out_bitwidth) {
  if (!loom_type_is_scalar(type)) return false;
  int32_t bitwidth = loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (bitwidth <= 0) return false;
  *out_bitwidth = bitwidth;
  return true;
}

static iree_status_t loom_scalar_replace_single_result_with_value(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t replacement) {
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static iree_status_t loom_scalar_replace_single_result_with_i64_constant(
    loom_op_t* op, loom_rewriter_t* rewriter, int64_t value) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_build_constant(rewriter, loom_value_facts_exact_i64(value),
                                   loom_scalar_single_result_type(rewriter, op),
                                   op->location, &replacement));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_scalar_replace_single_result_with_value(op, rewriter,
                                                      replacement);
}

static iree_status_t loom_scalar_replace_single_result_with_f64_constant(
    loom_op_t* op, loom_rewriter_t* rewriter, double value) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_build_constant(rewriter, loom_value_facts_exact_f64(value),
                                   loom_scalar_single_result_type(rewriter, op),
                                   op->location, &replacement));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_scalar_replace_single_result_with_value(op, rewriter,
                                                      replacement);
}

static iree_status_t loom_scalar_replace_mulf_with_negated_constant(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t input,
    double constant_value) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_type_t result_type = loom_scalar_single_result_type(rewriter, op);

  loom_value_id_t negated_constant = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_rewriter_build_constant(
      rewriter, loom_value_facts_exact_f64(-constant_value), result_type,
      op->location, &negated_constant));

  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_mulf_build(
      &rewriter->builder, loom_scalar_mulf_fastmath(op), input,
      negated_constant, result_type, op->location, &replacement_op));
  loom_value_id_t replacement = loom_scalar_mulf_result(replacement_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_scalar_replace_single_result_with_value(op, rewriter,
                                                      replacement);
}

static iree_status_t loom_scalar_materialize_or_reuse_i64_constant(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t candidate,
    int64_t value, loom_type_t type, loom_value_id_t* out_value_id) {
  int64_t candidate_value = 0;
  if (candidate != LOOM_VALUE_ID_INVALID &&
      loom_scalar_query_exact_i64(rewriter, candidate, &candidate_value) &&
      candidate_value == value) {
    *out_value_id = candidate;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_rewriter_build_constant(rewriter, loom_value_facts_exact_i64(value),
                                   type, op->location, out_value_id));
  return iree_ok_status();
}

static iree_status_t loom_scalar_replace_single_result_with_binary_op(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_kind_t kind,
    uint8_t instance_flags, loom_value_id_t lhs, loom_value_id_t rhs) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_type_t result_type = loom_scalar_single_result_type(rewriter, op);

  loom_op_t* replacement_op = NULL;
  switch (kind) {
    case LOOM_OP_SCALAR_ADDI: {
      IREE_RETURN_IF_ERROR(
          loom_scalar_addi_build(&rewriter->builder, instance_flags, lhs, rhs,
                                 result_type, op->location, &replacement_op));
      break;
    }
    case LOOM_OP_SCALAR_MULI: {
      IREE_RETURN_IF_ERROR(
          loom_scalar_muli_build(&rewriter->builder, instance_flags, lhs, rhs,
                                 result_type, op->location, &replacement_op));
      break;
    }
    case LOOM_OP_SCALAR_DIVUI: {
      IREE_RETURN_IF_ERROR(loom_scalar_divui_build(&rewriter->builder, lhs, rhs,
                                                   result_type, op->location,
                                                   &replacement_op));
      break;
    }
    case LOOM_OP_SCALAR_REMUI: {
      IREE_RETURN_IF_ERROR(loom_scalar_remui_build(&rewriter->builder, lhs, rhs,
                                                   result_type, op->location,
                                                   &replacement_op));
      break;
    }
    case LOOM_OP_SCALAR_ADDF: {
      IREE_RETURN_IF_ERROR(
          loom_scalar_addf_build(&rewriter->builder, instance_flags, lhs, rhs,
                                 result_type, op->location, &replacement_op));
      break;
    }
    case LOOM_OP_SCALAR_MULF: {
      IREE_RETURN_IF_ERROR(
          loom_scalar_mulf_build(&rewriter->builder, instance_flags, lhs, rhs,
                                 result_type, op->location, &replacement_op));
      break;
    }
    case LOOM_OP_SCALAR_SHLI: {
      IREE_RETURN_IF_ERROR(
          loom_scalar_shli_build(&rewriter->builder, instance_flags, lhs, rhs,
                                 result_type, op->location, &replacement_op));
      break;
    }
    case LOOM_OP_SCALAR_SHRUI: {
      IREE_RETURN_IF_ERROR(loom_scalar_shrui_build(&rewriter->builder, lhs, rhs,
                                                   result_type, op->location,
                                                   &replacement_op));
      break;
    }
    case LOOM_OP_SCALAR_SHRSI: {
      IREE_RETURN_IF_ERROR(loom_scalar_shrsi_build(&rewriter->builder, lhs, rhs,
                                                   result_type, op->location,
                                                   &replacement_op));
      break;
    }
    case LOOM_OP_SCALAR_ANDI: {
      IREE_RETURN_IF_ERROR(loom_scalar_andi_build(&rewriter->builder, lhs, rhs,
                                                  result_type, op->location,
                                                  &replacement_op));
      break;
    }
    case LOOM_OP_SCALAR_ORI: {
      IREE_RETURN_IF_ERROR(loom_scalar_ori_build(&rewriter->builder, lhs, rhs,
                                                 result_type, op->location,
                                                 &replacement_op));
      break;
    }
    case LOOM_OP_SCALAR_XORI: {
      IREE_RETURN_IF_ERROR(loom_scalar_xori_build(&rewriter->builder, lhs, rhs,
                                                  result_type, op->location,
                                                  &replacement_op));
      break;
    }
    case LOOM_OP_SCALAR_ROTLI: {
      IREE_RETURN_IF_ERROR(loom_scalar_rotli_build(&rewriter->builder, lhs, rhs,
                                                   result_type, op->location,
                                                   &replacement_op));
      break;
    }
    case LOOM_OP_SCALAR_ROTRI: {
      IREE_RETURN_IF_ERROR(loom_scalar_rotri_build(&rewriter->builder, lhs, rhs,
                                                   result_type, op->location,
                                                   &replacement_op));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported replacement scalar op kind %u",
                              (unsigned)kind);
  }

  loom_value_id_t replacement = loom_op_const_results(replacement_op)[0];
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_scalar_replace_single_result_with_value(op, rewriter,
                                                      replacement);
}

static iree_status_t loom_scalar_replace_single_result_with_unary_op(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_kind_t kind,
    uint8_t instance_flags, loom_value_id_t input) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_type_t result_type = loom_scalar_single_result_type(rewriter, op);

  loom_op_t* replacement_op = NULL;
  switch (kind) {
    case LOOM_OP_SCALAR_NEGI: {
      IREE_RETURN_IF_ERROR(loom_scalar_negi_build(&rewriter->builder, input,
                                                  result_type, op->location,
                                                  &replacement_op));
      break;
    }
    case LOOM_OP_SCALAR_ABSI: {
      IREE_RETURN_IF_ERROR(loom_scalar_absi_build(&rewriter->builder, input,
                                                  result_type, op->location,
                                                  &replacement_op));
      break;
    }
    case LOOM_OP_SCALAR_NEGF: {
      IREE_RETURN_IF_ERROR(
          loom_scalar_negf_build(&rewriter->builder, instance_flags, input,
                                 result_type, op->location, &replacement_op));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported replacement scalar op kind %u",
                              (unsigned)kind);
  }

  loom_value_id_t replacement = loom_op_const_results(replacement_op)[0];
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_scalar_replace_single_result_with_value(op, rewriter,
                                                      replacement);
}

static iree_status_t loom_scalar_replace_single_result_with_addi(
    loom_op_t* op, loom_rewriter_t* rewriter, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs) {
  return loom_scalar_replace_single_result_with_binary_op(
      op, rewriter, LOOM_OP_SCALAR_ADDI, instance_flags, lhs, rhs);
}

static iree_status_t loom_scalar_replace_single_result_with_muli(
    loom_op_t* op, loom_rewriter_t* rewriter, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs) {
  return loom_scalar_replace_single_result_with_binary_op(
      op, rewriter, LOOM_OP_SCALAR_MULI, instance_flags, lhs, rhs);
}

static iree_status_t loom_scalar_replace_single_result_with_scaled_shift(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_kind_t kind,
    uint8_t instance_flags, loom_value_id_t value, int64_t shift,
    loom_value_id_t reusable_constant) {
  loom_type_t result_type = loom_scalar_single_result_type(rewriter, op);
  int32_t bitwidth =
      loom_scalar_type_bitwidth(loom_type_element_type(result_type));
  if (shift < 0 || bitwidth <= 0 || shift >= bitwidth) return iree_ok_status();

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t shift_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_materialize_or_reuse_i64_constant(
      op, rewriter, reusable_constant, shift, result_type, &shift_value));
  return loom_scalar_replace_single_result_with_binary_op(
      op, rewriter, kind, instance_flags, value, shift_value);
}

static bool loom_scalar_match_addi_with_exact_constant(
    loom_rewriter_t* rewriter, loom_op_t* add_op, loom_value_id_t* out_value,
    loom_value_id_t* out_constant, int64_t* out_constant_value) {
  if (!add_op || !loom_scalar_addi_isa(add_op)) return false;

  loom_value_id_t lhs = loom_scalar_addi_lhs(add_op);
  loom_value_id_t rhs = loom_scalar_addi_rhs(add_op);
  int64_t lhs_constant = 0;
  if (loom_scalar_query_exact_i64(rewriter, lhs, &lhs_constant)) {
    *out_value = rhs;
    *out_constant = lhs;
    *out_constant_value = lhs_constant;
    return true;
  }
  int64_t rhs_constant = 0;
  if (loom_scalar_query_exact_i64(rewriter, rhs, &rhs_constant)) {
    *out_value = lhs;
    *out_constant = rhs;
    *out_constant_value = rhs_constant;
    return true;
  }
  return false;
}

static bool loom_scalar_signed_integer_extremes(loom_type_t type,
                                                int64_t* out_minimum,
                                                int64_t* out_maximum) {
  int32_t bitwidth = loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (bitwidth <= 1 || bitwidth > 64) return false;
  if (bitwidth == 64) {
    *out_minimum = INT64_MIN;
    *out_maximum = INT64_MAX;
    return true;
  }
  *out_minimum = -(((int64_t)1) << (bitwidth - 1));
  *out_maximum = (((int64_t)1) << (bitwidth - 1)) - 1;
  return true;
}

static bool loom_scalar_unsigned_integer_maximum(loom_type_t type,
                                                 int64_t* out_maximum) {
  int32_t bitwidth = loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (bitwidth <= 0 || bitwidth >= 63) return false;
  *out_maximum = (((int64_t)1) << bitwidth) - 1;
  return true;
}

static bool loom_scalar_integer_value_is_all_ones(loom_type_t type,
                                                  int64_t value) {
  if (value == -1) return true;
  int64_t maximum = 0;
  return loom_scalar_unsigned_integer_maximum(type, &maximum) &&
         value == maximum;
}

static bool loom_scalar_shift_amount_is_valid(loom_type_t type,
                                              int64_t amount) {
  int32_t bitwidth = loom_scalar_type_bitwidth(loom_type_element_type(type));
  return amount >= 0 && bitwidth > 0 && amount < bitwidth;
}

static iree_status_t loom_scalar_replace_single_result_with_cmpi(
    loom_op_t* op, loom_rewriter_t* rewriter, uint8_t predicate,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_type_t operand_type) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
      &rewriter->builder, predicate, lhs, rhs, operand_type,
      loom_scalar_single_result_type(rewriter, op), op->location,
      &replacement_op));
  loom_value_id_t replacement = loom_scalar_cmpi_result(replacement_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_scalar_replace_single_result_with_value(op, rewriter,
                                                      replacement);
}

static iree_status_t loom_scalar_replace_single_result_with_cast_op(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_kind_t kind,
    loom_value_id_t input, loom_type_t input_type) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_type_t result_type = loom_scalar_single_result_type(rewriter, op);

  loom_op_t* replacement_op = NULL;
  switch (kind) {
    case LOOM_OP_SCALAR_EXTF: {
      IREE_RETURN_IF_ERROR(
          loom_scalar_extf_build(&rewriter->builder, input, input_type,
                                 result_type, op->location, &replacement_op));
      break;
    }
    case LOOM_OP_SCALAR_FPTRUNC: {
      IREE_RETURN_IF_ERROR(loom_scalar_fptrunc_build(
          &rewriter->builder, input, input_type, result_type, op->location,
          &replacement_op));
      break;
    }
    case LOOM_OP_SCALAR_EXTSI: {
      IREE_RETURN_IF_ERROR(
          loom_scalar_extsi_build(&rewriter->builder, input, input_type,
                                  result_type, op->location, &replacement_op));
      break;
    }
    case LOOM_OP_SCALAR_EXTUI: {
      IREE_RETURN_IF_ERROR(
          loom_scalar_extui_build(&rewriter->builder, input, input_type,
                                  result_type, op->location, &replacement_op));
      break;
    }
    case LOOM_OP_SCALAR_TRUNCI: {
      IREE_RETURN_IF_ERROR(
          loom_scalar_trunci_build(&rewriter->builder, input, input_type,
                                   result_type, op->location, &replacement_op));
      break;
    }
    case LOOM_OP_SCALAR_BITCAST: {
      IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(
          &rewriter->builder, input, input_type, result_type, op->location,
          &replacement_op));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported replacement scalar cast kind %u",
                              (unsigned)kind);
  }

  loom_value_id_t replacement = loom_op_const_results(replacement_op)[0];
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_scalar_replace_single_result_with_value(op, rewriter,
                                                      replacement);
}

static iree_status_t loom_scalar_replace_single_result_with_integer_resize(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t input,
    loom_type_t input_type, loom_op_kind_t extension_kind) {
  loom_type_t result_type = loom_scalar_single_result_type(rewriter, op);
  if (loom_type_equal(input_type, result_type)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, input);
  }
  if (!loom_scalar_type_is_integer_scalar(input_type) ||
      !loom_scalar_type_is_integer_scalar(result_type)) {
    return iree_ok_status();
  }

  int32_t input_bitwidth = 0;
  int32_t result_bitwidth = 0;
  if (!loom_scalar_type_query_bitwidth(input_type, &input_bitwidth) ||
      !loom_scalar_type_query_bitwidth(result_type, &result_bitwidth) ||
      input_bitwidth == result_bitwidth) {
    return iree_ok_status();
  }

  loom_op_kind_t replacement_kind =
      result_bitwidth > input_bitwidth ? extension_kind : LOOM_OP_SCALAR_TRUNCI;
  return loom_scalar_replace_single_result_with_cast_op(
      op, rewriter, replacement_kind, input, input_type);
}

static iree_status_t loom_scalar_replace_single_result_with_float_resize(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t input,
    loom_type_t input_type) {
  loom_type_t result_type = loom_scalar_single_result_type(rewriter, op);
  if (loom_type_equal(input_type, result_type)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, input);
  }
  if (!loom_scalar_type_is_float_scalar(input_type) ||
      !loom_scalar_type_is_float_scalar(result_type)) {
    return iree_ok_status();
  }

  int32_t input_bitwidth = 0;
  int32_t result_bitwidth = 0;
  if (!loom_scalar_type_query_bitwidth(input_type, &input_bitwidth) ||
      !loom_scalar_type_query_bitwidth(result_type, &result_bitwidth) ||
      input_bitwidth == result_bitwidth) {
    return iree_ok_status();
  }

  loom_op_kind_t replacement_kind = result_bitwidth > input_bitwidth
                                        ? LOOM_OP_SCALAR_EXTF
                                        : LOOM_OP_SCALAR_FPTRUNC;
  return loom_scalar_replace_single_result_with_cast_op(
      op, rewriter, replacement_kind, input, input_type);
}

static bool loom_scalar_op_is_float_rounding(const loom_op_t* op) {
  switch (op->kind) {
    case LOOM_OP_SCALAR_CEILF:
    case LOOM_OP_SCALAR_FLOORF:
    case LOOM_OP_SCALAR_ROUNDF:
    case LOOM_OP_SCALAR_ROUNDEVENF:
    case LOOM_OP_SCALAR_TRUNCF:
      return true;
    default:
      return false;
  }
}

//===----------------------------------------------------------------------===//
// Integer arithmetic
//===----------------------------------------------------------------------===//

iree_status_t loom_scalar_addi_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_scalar_addi_lhs(op);
  loom_value_id_t rhs = loom_scalar_addi_rhs(op);
  if (loom_scalar_value_facts_are_exact_i64(rewriter, lhs, 0)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, rhs);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, rhs, 0)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }

  loom_op_t* lhs_def = loom_scalar_defining_op(rewriter, lhs);
  if (loom_scalar_op_has_no_instance_flags(op) && lhs_def &&
      loom_scalar_subi_isa(lhs_def) &&
      loom_scalar_op_has_no_instance_flags(lhs_def) &&
      loom_scalar_subi_rhs(lhs_def) == rhs) {
    return loom_scalar_replace_single_result_with_value(
        op, rewriter, loom_scalar_subi_lhs(lhs_def));
  }
  loom_op_t* rhs_def = loom_scalar_defining_op(rewriter, rhs);
  if (loom_scalar_op_has_no_instance_flags(op) && rhs_def &&
      loom_scalar_subi_isa(rhs_def) &&
      loom_scalar_op_has_no_instance_flags(rhs_def) &&
      loom_scalar_subi_rhs(rhs_def) == lhs) {
    return loom_scalar_replace_single_result_with_value(
        op, rewriter, loom_scalar_subi_lhs(rhs_def));
  }

  int64_t rhs_constant = 0;
  if (!loom_scalar_query_exact_i64(rewriter, rhs, &rhs_constant)) {
    if (loom_scalar_query_exact_i64(rewriter, lhs, &rhs_constant)) {
      return loom_scalar_replace_single_result_with_addi(
          op, rewriter, loom_scalar_addi_overflow(op), rhs, lhs);
    }
    return iree_ok_status();
  }
  if (!loom_scalar_op_has_no_instance_flags(op)) return iree_ok_status();

  loom_value_id_t inner_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t reusable_constant = LOOM_VALUE_ID_INVALID;
  int64_t inner_constant = 0;
  if (!loom_scalar_match_addi_with_exact_constant(
          rewriter, lhs_def, &inner_value, &reusable_constant,
          &inner_constant)) {
    return iree_ok_status();
  }
  if (!loom_scalar_op_has_no_instance_flags(lhs_def)) return iree_ok_status();

  int64_t combined_constant = 0;
  if (!loom_checked_add_i64(inner_constant, rhs_constant, &combined_constant)) {
    return iree_ok_status();
  }
  if (combined_constant == 0) {
    return loom_scalar_replace_single_result_with_value(op, rewriter,
                                                        inner_value);
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t combined_constant_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_materialize_or_reuse_i64_constant(
      op, rewriter, reusable_constant, combined_constant,
      loom_scalar_single_result_type(rewriter, op), &combined_constant_value));
  return loom_scalar_replace_single_result_with_addi(
      op, rewriter, /*instance_flags=*/0, inner_value, combined_constant_value);
}

iree_status_t loom_scalar_subi_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_scalar_subi_lhs(op);
  loom_value_id_t rhs = loom_scalar_subi_rhs(op);
  if (loom_scalar_value_facts_are_exact_i64(rewriter, rhs, 0)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }
  if (lhs == rhs) {
    return loom_scalar_replace_single_result_with_i64_constant(op, rewriter, 0);
  }

  loom_op_t* lhs_def = loom_scalar_defining_op(rewriter, lhs);
  if (loom_scalar_op_has_no_instance_flags(op) && lhs_def &&
      loom_scalar_addi_isa(lhs_def) &&
      loom_scalar_op_has_no_instance_flags(lhs_def)) {
    if (loom_scalar_addi_lhs(lhs_def) == rhs) {
      return loom_scalar_replace_single_result_with_value(
          op, rewriter, loom_scalar_addi_rhs(lhs_def));
    }
    if (loom_scalar_addi_rhs(lhs_def) == rhs) {
      return loom_scalar_replace_single_result_with_value(
          op, rewriter, loom_scalar_addi_lhs(lhs_def));
    }
  }

  int64_t rhs_constant = 0;
  if (!loom_scalar_op_has_no_instance_flags(op) ||
      !loom_scalar_query_exact_i64(rewriter, rhs, &rhs_constant) ||
      rhs_constant == INT64_MIN) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t negated_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_materialize_or_reuse_i64_constant(
      op, rewriter, LOOM_VALUE_ID_INVALID, -rhs_constant,
      loom_scalar_single_result_type(rewriter, op), &negated_rhs));
  return loom_scalar_replace_single_result_with_addi(
      op, rewriter, /*instance_flags=*/0, lhs, negated_rhs);
}

iree_status_t loom_scalar_muli_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_scalar_muli_lhs(op);
  loom_value_id_t rhs = loom_scalar_muli_rhs(op);
  if (loom_scalar_value_facts_are_exact_i64(rewriter, lhs, 0) ||
      loom_scalar_value_facts_are_exact_i64(rewriter, rhs, 0)) {
    return loom_scalar_replace_single_result_with_i64_constant(op, rewriter, 0);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, lhs, 1)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, rhs);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, rhs, 1)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }
  if (loom_scalar_op_has_no_instance_flags(op) &&
      loom_scalar_value_facts_are_exact_i64(rewriter, lhs, -1)) {
    return loom_scalar_replace_single_result_with_unary_op(
        op, rewriter, LOOM_OP_SCALAR_NEGI, /*instance_flags=*/0, rhs);
  }
  if (loom_scalar_op_has_no_instance_flags(op) &&
      loom_scalar_value_facts_are_exact_i64(rewriter, rhs, -1)) {
    return loom_scalar_replace_single_result_with_unary_op(
        op, rewriter, LOOM_OP_SCALAR_NEGI, /*instance_flags=*/0, lhs);
  }
  if (!loom_scalar_op_has_no_instance_flags(op)) return iree_ok_status();

  int64_t factor = 0;
  if (loom_scalar_query_exact_i64(rewriter, lhs, &factor)) {
    if (!loom_is_power_of_two_i64(factor)) return iree_ok_status();
    return loom_scalar_replace_single_result_with_scaled_shift(
        op, rewriter, LOOM_OP_SCALAR_SHLI, /*instance_flags=*/0, rhs,
        loom_ilog2_i64(factor), lhs);
  }
  if (loom_scalar_query_exact_i64(rewriter, rhs, &factor)) {
    if (!loom_is_power_of_two_i64(factor)) return iree_ok_status();
    return loom_scalar_replace_single_result_with_scaled_shift(
        op, rewriter, LOOM_OP_SCALAR_SHLI, /*instance_flags=*/0, lhs,
        loom_ilog2_i64(factor), rhs);
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_divsi_canonicalize(loom_op_t* op,
                                             loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_scalar_divsi_lhs(op);
  loom_value_id_t rhs = loom_scalar_divsi_rhs(op);
  if (lhs == rhs && loom_scalar_value_facts_are_non_zero(rewriter, rhs)) {
    return loom_scalar_replace_single_result_with_i64_constant(op, rewriter, 1);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, lhs, 0) &&
      loom_scalar_value_facts_are_non_zero(rewriter, rhs)) {
    return loom_scalar_replace_single_result_with_i64_constant(op, rewriter, 0);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, rhs, 1)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }
  if (loom_scalar_value_facts_are_non_negative(rewriter, lhs) &&
      loom_scalar_value_facts_are_non_negative(rewriter, rhs)) {
    return loom_scalar_replace_single_result_with_binary_op(
        op, rewriter, LOOM_OP_SCALAR_DIVUI, /*instance_flags=*/0, lhs, rhs);
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_divui_canonicalize(loom_op_t* op,
                                             loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_scalar_divui_lhs(op);
  loom_value_id_t rhs = loom_scalar_divui_rhs(op);
  if (lhs == rhs && loom_scalar_value_facts_are_non_zero(rewriter, rhs)) {
    return loom_scalar_replace_single_result_with_i64_constant(op, rewriter, 1);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, lhs, 0) &&
      loom_scalar_value_facts_are_non_zero(rewriter, rhs)) {
    return loom_scalar_replace_single_result_with_i64_constant(op, rewriter, 0);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, rhs, 1)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }

  int64_t divisor = 0;
  if (loom_scalar_query_exact_i64(rewriter, rhs, &divisor) &&
      loom_is_power_of_two_i64(divisor)) {
    return loom_scalar_replace_single_result_with_scaled_shift(
        op, rewriter, LOOM_OP_SCALAR_SHRUI, /*instance_flags=*/0, lhs,
        loom_ilog2_i64(divisor), rhs);
  }

  return iree_ok_status();
}

iree_status_t loom_scalar_remsi_canonicalize(loom_op_t* op,
                                             loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_scalar_remsi_lhs(op);
  loom_value_id_t rhs = loom_scalar_remsi_rhs(op);
  if (lhs == rhs && loom_scalar_value_facts_are_non_zero(rewriter, rhs)) {
    return loom_scalar_replace_single_result_with_i64_constant(op, rewriter, 0);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, lhs, 0) &&
      loom_scalar_value_facts_are_non_zero(rewriter, rhs)) {
    return loom_scalar_replace_single_result_with_i64_constant(op, rewriter, 0);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, rhs, 1)) {
    return loom_scalar_replace_single_result_with_i64_constant(op, rewriter, 0);
  }
  if (loom_scalar_value_facts_are_non_negative(rewriter, lhs) &&
      loom_scalar_value_facts_are_non_negative(rewriter, rhs)) {
    return loom_scalar_replace_single_result_with_binary_op(
        op, rewriter, LOOM_OP_SCALAR_REMUI, /*instance_flags=*/0, lhs, rhs);
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_remui_canonicalize(loom_op_t* op,
                                             loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_scalar_remui_lhs(op);
  loom_value_id_t rhs = loom_scalar_remui_rhs(op);
  if (lhs == rhs && loom_scalar_value_facts_are_non_zero(rewriter, rhs)) {
    return loom_scalar_replace_single_result_with_i64_constant(op, rewriter, 0);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, lhs, 0) &&
      loom_scalar_value_facts_are_non_zero(rewriter, rhs)) {
    return loom_scalar_replace_single_result_with_i64_constant(op, rewriter, 0);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, rhs, 1)) {
    return loom_scalar_replace_single_result_with_i64_constant(op, rewriter, 0);
  }

  int64_t divisor = 0;
  if (!loom_scalar_query_exact_i64(rewriter, rhs, &divisor) ||
      !loom_is_power_of_two_i64(divisor)) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t mask_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_materialize_or_reuse_i64_constant(
      op, rewriter, LOOM_VALUE_ID_INVALID, divisor - 1,
      loom_scalar_single_result_type(rewriter, op), &mask_value));
  return loom_scalar_replace_single_result_with_binary_op(
      op, rewriter, LOOM_OP_SCALAR_ANDI, /*instance_flags=*/0, lhs, mask_value);
}

iree_status_t loom_scalar_ceildivsi_canonicalize(loom_op_t* op,
                                                 loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_scalar_ceildivsi_lhs(op);
  loom_value_id_t rhs = loom_scalar_ceildivsi_rhs(op);
  if (loom_scalar_value_facts_are_exact_i64(rewriter, lhs, 0) &&
      loom_scalar_value_facts_are_non_zero(rewriter, rhs)) {
    return loom_scalar_replace_single_result_with_i64_constant(op, rewriter, 0);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, rhs, 1)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_ceildivui_canonicalize(loom_op_t* op,
                                                 loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_scalar_ceildivui_lhs(op);
  loom_value_id_t rhs = loom_scalar_ceildivui_rhs(op);
  if (loom_scalar_value_facts_are_exact_i64(rewriter, lhs, 0) &&
      loom_scalar_value_facts_are_non_zero(rewriter, rhs)) {
    return loom_scalar_replace_single_result_with_i64_constant(op, rewriter, 0);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, rhs, 1)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_floordivsi_canonicalize(loom_op_t* op,
                                                  loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_scalar_floordivsi_lhs(op);
  loom_value_id_t rhs = loom_scalar_floordivsi_rhs(op);
  if (loom_scalar_value_facts_are_exact_i64(rewriter, lhs, 0) &&
      loom_scalar_value_facts_are_non_zero(rewriter, rhs)) {
    return loom_scalar_replace_single_result_with_i64_constant(op, rewriter, 0);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, rhs, 1)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_negi_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  loom_op_t* input_def =
      loom_scalar_defining_op(rewriter, loom_scalar_negi_input(op));
  if (!input_def || !loom_scalar_negi_isa(input_def)) return iree_ok_status();
  return loom_scalar_replace_single_result_with_value(
      op, rewriter, loom_scalar_negi_input(input_def));
}

iree_status_t loom_scalar_absi_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  loom_value_id_t input = loom_scalar_absi_input(op);
  if (loom_scalar_value_facts_are_non_negative(rewriter, input)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, input);
  }

  loom_op_t* input_def = loom_scalar_defining_op(rewriter, input);
  if (!input_def) return iree_ok_status();
  if (loom_scalar_absi_isa(input_def)) {
    return loom_scalar_replace_single_result_with_value(
        op, rewriter, loom_scalar_absi_result(input_def));
  }
  if (loom_scalar_negi_isa(input_def)) {
    IREE_RETURN_IF_ERROR(loom_rewriter_set_operand(
        rewriter, op, 0, loom_scalar_negi_input(input_def)));
  }
  return iree_ok_status();
}

static iree_status_t loom_scalar_signed_minmax_canonicalize(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t lhs,
    loom_value_id_t rhs, bool is_minimum) {
  if (lhs == rhs) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }

  int64_t minimum = 0;
  int64_t maximum = 0;
  if (!loom_scalar_signed_integer_extremes(
          loom_scalar_single_result_type(rewriter, op), &minimum, &maximum)) {
    return iree_ok_status();
  }

  int64_t lhs_value = 0;
  if (loom_scalar_query_exact_i64(rewriter, lhs, &lhs_value)) {
    if (lhs_value == (is_minimum ? minimum : maximum)) {
      return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
    }
    if (lhs_value == (is_minimum ? maximum : minimum)) {
      return loom_scalar_replace_single_result_with_value(op, rewriter, rhs);
    }
  }
  int64_t rhs_value = 0;
  if (loom_scalar_query_exact_i64(rewriter, rhs, &rhs_value)) {
    if (rhs_value == (is_minimum ? minimum : maximum)) {
      return loom_scalar_replace_single_result_with_value(op, rewriter, rhs);
    }
    if (rhs_value == (is_minimum ? maximum : minimum)) {
      return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_minsi_canonicalize(loom_op_t* op,
                                             loom_rewriter_t* rewriter) {
  return loom_scalar_signed_minmax_canonicalize(
      op, rewriter, loom_scalar_minsi_lhs(op), loom_scalar_minsi_rhs(op),
      /*is_minimum=*/true);
}

iree_status_t loom_scalar_maxsi_canonicalize(loom_op_t* op,
                                             loom_rewriter_t* rewriter) {
  return loom_scalar_signed_minmax_canonicalize(
      op, rewriter, loom_scalar_maxsi_lhs(op), loom_scalar_maxsi_rhs(op),
      /*is_minimum=*/false);
}

static iree_status_t loom_scalar_unsigned_minmax_canonicalize(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t lhs,
    loom_value_id_t rhs, bool is_minimum) {
  if (lhs == rhs) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }

  int64_t maximum = 0;
  if (!loom_scalar_unsigned_integer_maximum(
          loom_scalar_single_result_type(rewriter, op), &maximum)) {
    return iree_ok_status();
  }

  int64_t lhs_value = 0;
  if (loom_scalar_query_exact_i64(rewriter, lhs, &lhs_value)) {
    if (lhs_value == (is_minimum ? 0 : maximum)) {
      return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
    }
    if (lhs_value == (is_minimum ? maximum : 0)) {
      return loom_scalar_replace_single_result_with_value(op, rewriter, rhs);
    }
  }
  int64_t rhs_value = 0;
  if (loom_scalar_query_exact_i64(rewriter, rhs, &rhs_value)) {
    if (rhs_value == (is_minimum ? 0 : maximum)) {
      return loom_scalar_replace_single_result_with_value(op, rewriter, rhs);
    }
    if (rhs_value == (is_minimum ? maximum : 0)) {
      return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_minui_canonicalize(loom_op_t* op,
                                             loom_rewriter_t* rewriter) {
  return loom_scalar_unsigned_minmax_canonicalize(
      op, rewriter, loom_scalar_minui_lhs(op), loom_scalar_minui_rhs(op),
      /*is_minimum=*/true);
}

iree_status_t loom_scalar_maxui_canonicalize(loom_op_t* op,
                                             loom_rewriter_t* rewriter) {
  return loom_scalar_unsigned_minmax_canonicalize(
      op, rewriter, loom_scalar_maxui_lhs(op), loom_scalar_maxui_rhs(op),
      /*is_minimum=*/false);
}

iree_status_t loom_scalar_fmai_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  loom_value_id_t a = loom_scalar_fmai_a(op);
  loom_value_id_t b = loom_scalar_fmai_b(op);
  loom_value_id_t c = loom_scalar_fmai_c(op);

  if (loom_scalar_value_facts_are_exact_i64(rewriter, a, 0) ||
      loom_scalar_value_facts_are_exact_i64(rewriter, b, 0)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, c);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, c, 0)) {
    return loom_scalar_replace_single_result_with_muli(
        op, rewriter, loom_scalar_fmai_overflow(op), a, b);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, a, 1)) {
    return loom_scalar_replace_single_result_with_addi(
        op, rewriter, loom_scalar_fmai_overflow(op), b, c);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, b, 1)) {
    return loom_scalar_replace_single_result_with_addi(
        op, rewriter, loom_scalar_fmai_overflow(op), a, c);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Floating-point arithmetic
//===----------------------------------------------------------------------===//

iree_status_t loom_scalar_addf_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  if (!loom_scalar_fastmath_has_all(op, LOOM_SCALAR_FASTMATHFLAGS_NSZ)) {
    return iree_ok_status();
  }
  loom_value_id_t lhs = loom_scalar_addf_lhs(op);
  loom_value_id_t rhs = loom_scalar_addf_rhs(op);
  if (loom_scalar_value_facts_are_exact_f64(rewriter, lhs, 0.0)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, rhs);
  }
  if (loom_scalar_value_facts_are_exact_f64(rewriter, rhs, 0.0)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_subf_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  if (!loom_scalar_fastmath_has_all(op, LOOM_SCALAR_FASTMATHFLAGS_NSZ)) {
    return iree_ok_status();
  }
  loom_value_id_t rhs = loom_scalar_subf_rhs(op);
  if (loom_scalar_value_facts_are_exact_f64(rewriter, rhs, 0.0)) {
    return loom_scalar_replace_single_result_with_value(
        op, rewriter, loom_scalar_subf_lhs(op));
  }
  const uint8_t neg_flags =
      LOOM_SCALAR_FASTMATHFLAGS_NNAN | LOOM_SCALAR_FASTMATHFLAGS_NSZ;
  loom_value_id_t lhs = loom_scalar_subf_lhs(op);
  if (loom_scalar_fastmath_has_all(op, neg_flags) &&
      loom_scalar_value_facts_are_exact_f64(rewriter, lhs, 0.0)) {
    return loom_scalar_replace_single_result_with_unary_op(
        op, rewriter, LOOM_OP_SCALAR_NEGF, loom_scalar_subf_fastmath(op), rhs);
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_mulf_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_scalar_mulf_lhs(op);
  loom_value_id_t rhs = loom_scalar_mulf_rhs(op);
  if (loom_scalar_value_facts_are_exact_f64(rewriter, lhs, 1.0)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, rhs);
  }
  if (loom_scalar_value_facts_are_exact_f64(rewriter, rhs, 1.0)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }

  loom_op_t* lhs_def = loom_scalar_defining_op(rewriter, lhs);
  double rhs_value = 0.0;
  if (lhs_def && loom_scalar_negf_isa(lhs_def) &&
      loom_scalar_query_exact_f64(rewriter, rhs, &rhs_value)) {
    return loom_scalar_replace_mulf_with_negated_constant(
        op, rewriter, loom_scalar_negf_input(lhs_def), rhs_value);
  }
  loom_op_t* rhs_def = loom_scalar_defining_op(rewriter, rhs);
  double lhs_value = 0.0;
  if (rhs_def && loom_scalar_negf_isa(rhs_def) &&
      loom_scalar_query_exact_f64(rewriter, lhs, &lhs_value)) {
    return loom_scalar_replace_mulf_with_negated_constant(
        op, rewriter, loom_scalar_negf_input(rhs_def), lhs_value);
  }

  const uint8_t zero_flags = LOOM_SCALAR_FASTMATHFLAGS_NNAN |
                             LOOM_SCALAR_FASTMATHFLAGS_NINF |
                             LOOM_SCALAR_FASTMATHFLAGS_NSZ;
  if (!loom_scalar_fastmath_has_all(op, zero_flags)) return iree_ok_status();
  if (loom_scalar_value_facts_are_exact_f64(rewriter, lhs, 0.0) ||
      loom_scalar_value_facts_are_exact_f64(rewriter, rhs, 0.0)) {
    return loom_scalar_replace_single_result_with_f64_constant(op, rewriter,
                                                               0.0);
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_divf_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  loom_value_id_t rhs = loom_scalar_divf_rhs(op);
  if (loom_scalar_value_facts_are_exact_f64(rewriter, rhs, 1.0)) {
    return loom_scalar_replace_single_result_with_value(
        op, rewriter, loom_scalar_divf_lhs(op));
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_negf_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  loom_op_t* input_def =
      loom_scalar_defining_op(rewriter, loom_scalar_negf_input(op));
  if (!input_def || !loom_scalar_negf_isa(input_def)) return iree_ok_status();
  return loom_scalar_replace_single_result_with_value(
      op, rewriter, loom_scalar_negf_input(input_def));
}

iree_status_t loom_scalar_absf_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  loom_value_id_t input = loom_scalar_absf_input(op);
  loom_op_t* input_def = loom_scalar_defining_op(rewriter, input);
  if (!input_def) return iree_ok_status();
  if (loom_scalar_absf_isa(input_def)) {
    return loom_scalar_replace_single_result_with_value(
        op, rewriter, loom_scalar_absf_result(input_def));
  }
  if (loom_scalar_negf_isa(input_def)) {
    IREE_RETURN_IF_ERROR(loom_rewriter_set_operand(
        rewriter, op, 0, loom_scalar_negf_input(input_def)));
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_copysignf_canonicalize(loom_op_t* op,
                                                 loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_scalar_copysignf_lhs(op);
  loom_value_id_t rhs = loom_scalar_copysignf_rhs(op);
  if (lhs == rhs) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_clampf_canonicalize(loom_op_t* op,
                                              loom_rewriter_t* rewriter) {
  loom_value_facts_t facts =
      loom_rewriter_value_facts(rewriter, loom_scalar_clampf_result(op));
  if (loom_value_facts_is_exact(facts) && loom_value_facts_is_float(facts)) {
    return loom_scalar_replace_single_result_with_f64_constant(
        op, rewriter, loom_value_facts_as_f64(facts));
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_fmaf_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  loom_value_id_t a = loom_scalar_fmaf_a(op);
  loom_value_id_t b = loom_scalar_fmaf_b(op);
  loom_value_id_t c = loom_scalar_fmaf_c(op);

  const uint8_t zero_flags = LOOM_SCALAR_FASTMATHFLAGS_NNAN |
                             LOOM_SCALAR_FASTMATHFLAGS_NINF |
                             LOOM_SCALAR_FASTMATHFLAGS_NSZ;
  if (loom_scalar_fastmath_has_all(op, zero_flags) &&
      (loom_scalar_value_facts_are_exact_f64(rewriter, a, 0.0) ||
       loom_scalar_value_facts_are_exact_f64(rewriter, b, 0.0))) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, c);
  }
  if (loom_scalar_fastmath_has_all(op, LOOM_SCALAR_FASTMATHFLAGS_NSZ) &&
      loom_scalar_value_facts_are_exact_f64(rewriter, c, 0.0)) {
    return loom_scalar_replace_single_result_with_binary_op(
        op, rewriter, LOOM_OP_SCALAR_MULF, loom_scalar_fmaf_fastmath(op), a, b);
  }
  if (loom_scalar_value_facts_are_exact_f64(rewriter, a, 1.0)) {
    return loom_scalar_replace_single_result_with_binary_op(
        op, rewriter, LOOM_OP_SCALAR_ADDF, loom_scalar_fmaf_fastmath(op), b, c);
  }
  if (loom_scalar_value_facts_are_exact_f64(rewriter, b, 1.0)) {
    return loom_scalar_replace_single_result_with_binary_op(
        op, rewriter, LOOM_OP_SCALAR_ADDF, loom_scalar_fmaf_fastmath(op), a, c);
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_powf_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_scalar_powf_lhs(op);
  loom_value_id_t rhs = loom_scalar_powf_rhs(op);
  if (loom_scalar_value_facts_are_exact_f64(rewriter, rhs, 0.0) ||
      loom_scalar_value_facts_are_exact_f64(rewriter, lhs, 1.0)) {
    return loom_scalar_replace_single_result_with_f64_constant(op, rewriter,
                                                               1.0);
  }
  if (loom_scalar_value_facts_are_exact_f64(rewriter, rhs, 1.0)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }
  if (loom_scalar_value_facts_are_exact_f64(rewriter, rhs, 2.0)) {
    return loom_scalar_replace_single_result_with_binary_op(
        op, rewriter, LOOM_OP_SCALAR_MULF, loom_scalar_powf_fastmath(op), lhs,
        lhs);
  }
  return iree_ok_status();
}

static iree_status_t loom_scalar_rounding_canonicalize(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t input) {
  loom_op_t* input_def = loom_scalar_defining_op(rewriter, input);
  if (!input_def || !loom_scalar_op_is_float_rounding(input_def)) {
    return iree_ok_status();
  }
  return loom_scalar_replace_single_result_with_value(op, rewriter, input);
}

iree_status_t loom_scalar_ceilf_canonicalize(loom_op_t* op,
                                             loom_rewriter_t* rewriter) {
  return loom_scalar_rounding_canonicalize(op, rewriter,
                                           loom_scalar_ceilf_input(op));
}

iree_status_t loom_scalar_floorf_canonicalize(loom_op_t* op,
                                              loom_rewriter_t* rewriter) {
  return loom_scalar_rounding_canonicalize(op, rewriter,
                                           loom_scalar_floorf_input(op));
}

iree_status_t loom_scalar_roundf_canonicalize(loom_op_t* op,
                                              loom_rewriter_t* rewriter) {
  return loom_scalar_rounding_canonicalize(op, rewriter,
                                           loom_scalar_roundf_input(op));
}

iree_status_t loom_scalar_roundevenf_canonicalize(loom_op_t* op,
                                                  loom_rewriter_t* rewriter) {
  return loom_scalar_rounding_canonicalize(op, rewriter,
                                           loom_scalar_roundevenf_input(op));
}

iree_status_t loom_scalar_truncf_canonicalize(loom_op_t* op,
                                              loom_rewriter_t* rewriter) {
  return loom_scalar_rounding_canonicalize(op, rewriter,
                                           loom_scalar_truncf_input(op));
}

//===----------------------------------------------------------------------===//
// Bitwise
//===----------------------------------------------------------------------===//

iree_status_t loom_scalar_andi_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_scalar_andi_lhs(op);
  loom_value_id_t rhs = loom_scalar_andi_rhs(op);
  if (lhs == rhs) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, lhs, 0) ||
      loom_scalar_value_facts_are_exact_i64(rewriter, rhs, 0)) {
    return loom_scalar_replace_single_result_with_i64_constant(op, rewriter, 0);
  }
  loom_type_t type = loom_scalar_single_result_type(rewriter, op);
  int64_t lhs_value = 0;
  if (loom_scalar_query_exact_i64(rewriter, lhs, &lhs_value) &&
      loom_scalar_integer_value_is_all_ones(type, lhs_value)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, rhs);
  }
  int64_t rhs_value = 0;
  if (loom_scalar_query_exact_i64(rewriter, rhs, &rhs_value) &&
      loom_scalar_integer_value_is_all_ones(type, rhs_value)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_ori_canonicalize(loom_op_t* op,
                                           loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_scalar_ori_lhs(op);
  loom_value_id_t rhs = loom_scalar_ori_rhs(op);
  if (lhs == rhs) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, lhs, 0)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, rhs);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, rhs, 0)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }
  loom_type_t type = loom_scalar_single_result_type(rewriter, op);
  int64_t lhs_value = 0;
  if (loom_scalar_query_exact_i64(rewriter, lhs, &lhs_value) &&
      loom_scalar_integer_value_is_all_ones(type, lhs_value)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }
  int64_t rhs_value = 0;
  if (loom_scalar_query_exact_i64(rewriter, rhs, &rhs_value) &&
      loom_scalar_integer_value_is_all_ones(type, rhs_value)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, rhs);
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_xori_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_scalar_xori_lhs(op);
  loom_value_id_t rhs = loom_scalar_xori_rhs(op);
  if (lhs == rhs) {
    return loom_scalar_replace_single_result_with_i64_constant(op, rewriter, 0);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, lhs, 0)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, rhs);
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, rhs, 0)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }

  loom_op_t* lhs_def = loom_scalar_defining_op(rewriter, lhs);
  if (lhs_def && loom_scalar_xori_isa(lhs_def)) {
    if (loom_scalar_values_are_same_or_same_exact_i64(
            rewriter, loom_scalar_xori_lhs(lhs_def), rhs)) {
      return loom_scalar_replace_single_result_with_value(
          op, rewriter, loom_scalar_xori_rhs(lhs_def));
    }
    if (loom_scalar_values_are_same_or_same_exact_i64(
            rewriter, loom_scalar_xori_rhs(lhs_def), rhs)) {
      return loom_scalar_replace_single_result_with_value(
          op, rewriter, loom_scalar_xori_lhs(lhs_def));
    }
  }
  loom_op_t* rhs_def = loom_scalar_defining_op(rewriter, rhs);
  if (rhs_def && loom_scalar_xori_isa(rhs_def)) {
    if (loom_scalar_values_are_same_or_same_exact_i64(
            rewriter, loom_scalar_xori_lhs(rhs_def), lhs)) {
      return loom_scalar_replace_single_result_with_value(
          op, rewriter, loom_scalar_xori_rhs(rhs_def));
    }
    if (loom_scalar_values_are_same_or_same_exact_i64(
            rewriter, loom_scalar_xori_rhs(rhs_def), lhs)) {
      return loom_scalar_replace_single_result_with_value(
          op, rewriter, loom_scalar_xori_lhs(rhs_def));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_scalar_shift_canonicalize(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_kind_t kind,
    uint8_t instance_flags, loom_value_id_t lhs, loom_value_id_t rhs) {
  loom_type_t type = loom_scalar_single_result_type(rewriter, op);
  int64_t amount = 0;
  bool has_exact_amount = loom_scalar_query_exact_i64(rewriter, rhs, &amount);
  if (has_exact_amount) {
    if (amount == 0) {
      return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
    }
    if (loom_scalar_shift_amount_is_valid(type, amount) &&
        loom_scalar_value_facts_are_exact_i64(rewriter, lhs, 0)) {
      return loom_scalar_replace_single_result_with_i64_constant(op, rewriter,
                                                                 0);
    }
  }

  if (!has_exact_amount) return iree_ok_status();
  if (instance_flags != 0) return iree_ok_status();
  loom_op_t* lhs_def = loom_scalar_defining_op(rewriter, lhs);
  if (!lhs_def || lhs_def->kind != kind || lhs_def->instance_flags != 0) {
    return iree_ok_status();
  }

  int64_t inner_amount = 0;
  if (!loom_scalar_query_exact_i64(rewriter, loom_op_const_operands(lhs_def)[1],
                                   &inner_amount)) {
    return iree_ok_status();
  }
  int64_t combined_amount = 0;
  if (!loom_checked_add_i64(inner_amount, amount, &combined_amount) ||
      !loom_scalar_shift_amount_is_valid(type, combined_amount)) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t combined_amount_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_materialize_or_reuse_i64_constant(
      op, rewriter, rhs, combined_amount, type, &combined_amount_value));
  return loom_scalar_replace_single_result_with_binary_op(
      op, rewriter, kind, /*instance_flags=*/0,
      loom_op_const_operands(lhs_def)[0], combined_amount_value);
}

iree_status_t loom_scalar_shli_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  return loom_scalar_shift_canonicalize(
      op, rewriter, LOOM_OP_SCALAR_SHLI, loom_scalar_shli_overflow(op),
      loom_scalar_shli_lhs(op), loom_scalar_shli_rhs(op));
}

iree_status_t loom_scalar_shrsi_canonicalize(loom_op_t* op,
                                             loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_scalar_shrsi_lhs(op);
  loom_value_id_t rhs = loom_scalar_shrsi_rhs(op);
  if (loom_scalar_value_facts_are_exact_i64(rewriter, rhs, 0)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }
  if (loom_scalar_value_facts_are_non_negative(rewriter, lhs)) {
    return loom_scalar_replace_single_result_with_binary_op(
        op, rewriter, LOOM_OP_SCALAR_SHRUI, /*instance_flags=*/0, lhs, rhs);
  }
  return loom_scalar_shift_canonicalize(
      op, rewriter, LOOM_OP_SCALAR_SHRSI, /*instance_flags=*/0,
      loom_scalar_shrsi_lhs(op), loom_scalar_shrsi_rhs(op));
}

iree_status_t loom_scalar_shrui_canonicalize(loom_op_t* op,
                                             loom_rewriter_t* rewriter) {
  return loom_scalar_shift_canonicalize(
      op, rewriter, LOOM_OP_SCALAR_SHRUI, /*instance_flags=*/0,
      loom_scalar_shrui_lhs(op), loom_scalar_shrui_rhs(op));
}

iree_status_t loom_scalar_rotli_canonicalize(loom_op_t* op,
                                             loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_scalar_rotli_lhs(op);
  loom_value_id_t rhs = loom_scalar_rotli_rhs(op);
  if (loom_scalar_value_facts_are_exact_i64(rewriter, rhs, 0)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_rotri_canonicalize(loom_op_t* op,
                                             loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_scalar_rotri_lhs(op);
  loom_value_id_t rhs = loom_scalar_rotri_rhs(op);
  int64_t amount = 0;
  if (!loom_scalar_query_exact_i64(rewriter, rhs, &amount)) {
    return iree_ok_status();
  }
  if (amount == 0) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, lhs);
  }

  loom_type_t type = loom_scalar_single_result_type(rewriter, op);
  int32_t bitwidth = loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (!loom_scalar_shift_amount_is_valid(type, amount)) return iree_ok_status();
  int64_t left_amount = bitwidth - amount;

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t left_amount_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_materialize_or_reuse_i64_constant(
      op, rewriter, LOOM_VALUE_ID_INVALID, left_amount, type,
      &left_amount_value));
  return loom_scalar_replace_single_result_with_binary_op(
      op, rewriter, LOOM_OP_SCALAR_ROTLI, /*instance_flags=*/0, lhs,
      left_amount_value);
}

//===----------------------------------------------------------------------===//
// Comparison and selection
//===----------------------------------------------------------------------===//

static iree_status_t loom_scalar_cmpi_unsigned_zero_canonicalize(
    loom_op_t* op, loom_rewriter_t* rewriter, uint8_t predicate,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_type_t operand_type,
    bool* out_changed) {
  *out_changed = true;
  if (loom_scalar_value_facts_are_exact_i64(rewriter, rhs, 0)) {
    switch ((loom_scalar_cmpi_predicate_t)predicate) {
      case LOOM_SCALAR_CMPI_PREDICATE_ULT:
        return loom_scalar_replace_single_result_with_i64_constant(op, rewriter,
                                                                   0);
      case LOOM_SCALAR_CMPI_PREDICATE_UGE:
        return loom_scalar_replace_single_result_with_i64_constant(op, rewriter,
                                                                   1);
      case LOOM_SCALAR_CMPI_PREDICATE_ULE:
        return loom_scalar_replace_single_result_with_cmpi(
            op, rewriter, LOOM_SCALAR_CMPI_PREDICATE_EQ, lhs, rhs,
            operand_type);
      case LOOM_SCALAR_CMPI_PREDICATE_UGT:
        return loom_scalar_replace_single_result_with_cmpi(
            op, rewriter, LOOM_SCALAR_CMPI_PREDICATE_NE, lhs, rhs,
            operand_type);
      default:
        break;
    }
  }
  if (loom_scalar_value_facts_are_exact_i64(rewriter, lhs, 0)) {
    switch ((loom_scalar_cmpi_predicate_t)predicate) {
      case LOOM_SCALAR_CMPI_PREDICATE_ULE:
        return loom_scalar_replace_single_result_with_i64_constant(op, rewriter,
                                                                   1);
      case LOOM_SCALAR_CMPI_PREDICATE_UGT:
        return loom_scalar_replace_single_result_with_i64_constant(op, rewriter,
                                                                   0);
      case LOOM_SCALAR_CMPI_PREDICATE_ULT:
        return loom_scalar_replace_single_result_with_cmpi(
            op, rewriter, LOOM_SCALAR_CMPI_PREDICATE_NE, rhs, lhs,
            operand_type);
      case LOOM_SCALAR_CMPI_PREDICATE_UGE:
        return loom_scalar_replace_single_result_with_cmpi(
            op, rewriter, LOOM_SCALAR_CMPI_PREDICATE_EQ, rhs, lhs,
            operand_type);
      default:
        break;
    }
  }
  *out_changed = false;
  return iree_ok_status();
}

iree_status_t loom_scalar_cmpi_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_scalar_cmpi_lhs(op);
  loom_value_id_t rhs = loom_scalar_cmpi_rhs(op);
  uint8_t predicate = loom_scalar_cmpi_predicate(op);

  bool result = false;
  loom_value_facts_t lhs_facts = loom_rewriter_value_facts(rewriter, lhs);
  loom_value_facts_t rhs_facts = loom_rewriter_value_facts(rewriter, rhs);
  if ((lhs == rhs && loom_scalar_cmpi_same_value_result(predicate, &result)) ||
      loom_scalar_cmpi_result_from_facts(predicate, &lhs_facts, &rhs_facts,
                                         &result)) {
    return loom_scalar_replace_single_result_with_i64_constant(op, rewriter,
                                                               result ? 1 : 0);
  }

  loom_type_t operand_type = loom_module_value_type(rewriter->module, lhs);
  bool changed = false;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_unsigned_zero_canonicalize(
      op, rewriter, predicate, lhs, rhs, operand_type, &changed));
  if (changed) return iree_ok_status();

  int64_t lhs_value = 0;
  int64_t rhs_value = 0;
  if (loom_scalar_query_exact_i64(rewriter, lhs, &lhs_value) &&
      !loom_scalar_query_exact_i64(rewriter, rhs, &rhs_value)) {
    return loom_scalar_replace_single_result_with_cmpi(
        op, rewriter, loom_scalar_cmpi_swapped_predicate(predicate), rhs, lhs,
        operand_type);
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_cmpf_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_scalar_cmpf_lhs(op);
  loom_value_id_t rhs = loom_scalar_cmpf_rhs(op);
  uint8_t predicate = loom_scalar_cmpf_predicate(op);

  bool result = false;
  if (lhs == rhs &&
      loom_scalar_fastmath_has_all(op, LOOM_SCALAR_FASTMATHFLAGS_NNAN) &&
      loom_scalar_cmpf_same_value_result(predicate, &result)) {
    return loom_scalar_replace_single_result_with_i64_constant(op, rewriter,
                                                               result ? 1 : 0);
  }

  double lhs_value = 0.0;
  double rhs_value = 0.0;
  if (loom_scalar_query_exact_f64(rewriter, lhs, &lhs_value) &&
      loom_scalar_query_exact_f64(rewriter, rhs, &rhs_value) &&
      loom_scalar_cmpf_exact_result(predicate, lhs_value, rhs_value, &result)) {
    return loom_scalar_replace_single_result_with_i64_constant(op, rewriter,
                                                               result ? 1 : 0);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Conversions
//===----------------------------------------------------------------------===//

iree_status_t loom_scalar_extf_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  loom_value_id_t input = loom_scalar_extf_input(op);
  loom_type_t input_type = loom_module_value_type(rewriter->module, input);
  if (loom_type_equal(input_type,
                      loom_scalar_single_result_type(rewriter, op))) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, input);
  }

  loom_op_t* input_def = loom_scalar_defining_op(rewriter, input);
  if (input_def && loom_scalar_extf_isa(input_def)) {
    loom_value_id_t inner_input = loom_scalar_extf_input(input_def);
    return loom_scalar_replace_single_result_with_float_resize(
        op, rewriter, inner_input,
        loom_module_value_type(rewriter->module, inner_input));
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_fptrunc_canonicalize(loom_op_t* op,
                                               loom_rewriter_t* rewriter) {
  loom_value_id_t input = loom_scalar_fptrunc_input(op);
  loom_type_t input_type = loom_module_value_type(rewriter->module, input);
  if (loom_type_equal(input_type,
                      loom_scalar_single_result_type(rewriter, op))) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, input);
  }

  loom_op_t* input_def = loom_scalar_defining_op(rewriter, input);
  if (input_def && loom_scalar_extf_isa(input_def)) {
    loom_value_id_t inner_input = loom_scalar_extf_input(input_def);
    return loom_scalar_replace_single_result_with_float_resize(
        op, rewriter, inner_input,
        loom_module_value_type(rewriter->module, inner_input));
  }
  if (input_def && loom_scalar_fptrunc_isa(input_def)) {
    loom_value_id_t inner_input = loom_scalar_fptrunc_input(input_def);
    return loom_scalar_replace_single_result_with_float_resize(
        op, rewriter, inner_input,
        loom_module_value_type(rewriter->module, inner_input));
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_extsi_canonicalize(loom_op_t* op,
                                             loom_rewriter_t* rewriter) {
  loom_value_id_t input = loom_scalar_extsi_input(op);
  loom_type_t input_type = loom_module_value_type(rewriter->module, input);
  if (loom_type_equal(input_type,
                      loom_scalar_single_result_type(rewriter, op))) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, input);
  }

  loom_op_t* input_def = loom_scalar_defining_op(rewriter, input);
  if (input_def && loom_scalar_extsi_isa(input_def)) {
    loom_value_id_t inner_input = loom_scalar_extsi_input(input_def);
    return loom_scalar_replace_single_result_with_integer_resize(
        op, rewriter, inner_input,
        loom_module_value_type(rewriter->module, inner_input),
        LOOM_OP_SCALAR_EXTSI);
  }
  if (input_def && loom_scalar_extui_isa(input_def)) {
    loom_value_id_t inner_input = loom_scalar_extui_input(input_def);
    return loom_scalar_replace_single_result_with_integer_resize(
        op, rewriter, inner_input,
        loom_module_value_type(rewriter->module, inner_input),
        LOOM_OP_SCALAR_EXTUI);
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_extui_canonicalize(loom_op_t* op,
                                             loom_rewriter_t* rewriter) {
  loom_value_id_t input = loom_scalar_extui_input(op);
  loom_type_t input_type = loom_module_value_type(rewriter->module, input);
  if (loom_type_equal(input_type,
                      loom_scalar_single_result_type(rewriter, op))) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, input);
  }

  loom_op_t* input_def = loom_scalar_defining_op(rewriter, input);
  if (input_def && loom_scalar_extui_isa(input_def)) {
    loom_value_id_t inner_input = loom_scalar_extui_input(input_def);
    return loom_scalar_replace_single_result_with_integer_resize(
        op, rewriter, inner_input,
        loom_module_value_type(rewriter->module, inner_input),
        LOOM_OP_SCALAR_EXTUI);
  }
  if (input_def && loom_scalar_extsi_isa(input_def)) {
    loom_value_id_t inner_input = loom_scalar_extsi_input(input_def);
    if (loom_scalar_value_facts_are_non_negative(rewriter, inner_input)) {
      return loom_scalar_replace_single_result_with_integer_resize(
          op, rewriter, inner_input,
          loom_module_value_type(rewriter->module, inner_input),
          LOOM_OP_SCALAR_EXTUI);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_trunci_canonicalize(loom_op_t* op,
                                              loom_rewriter_t* rewriter) {
  loom_value_id_t input = loom_scalar_trunci_input(op);
  loom_type_t input_type = loom_module_value_type(rewriter->module, input);
  loom_type_t result_type = loom_scalar_single_result_type(rewriter, op);
  if (loom_type_equal(input_type, result_type)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, input);
  }

  loom_op_t* input_def = loom_scalar_defining_op(rewriter, input);
  if (input_def && loom_scalar_extsi_isa(input_def)) {
    loom_value_id_t inner_input = loom_scalar_extsi_input(input_def);
    return loom_scalar_replace_single_result_with_integer_resize(
        op, rewriter, inner_input,
        loom_module_value_type(rewriter->module, inner_input),
        LOOM_OP_SCALAR_EXTSI);
  }
  if (input_def && loom_scalar_extui_isa(input_def)) {
    loom_value_id_t inner_input = loom_scalar_extui_input(input_def);
    return loom_scalar_replace_single_result_with_integer_resize(
        op, rewriter, inner_input,
        loom_module_value_type(rewriter->module, inner_input),
        LOOM_OP_SCALAR_EXTUI);
  }
  if (input_def && loom_scalar_trunci_isa(input_def)) {
    int32_t input_bitwidth = 0;
    int32_t result_bitwidth = 0;
    if (!loom_scalar_type_query_bitwidth(input_type, &input_bitwidth) ||
        !loom_scalar_type_query_bitwidth(result_type, &result_bitwidth) ||
        result_bitwidth >= input_bitwidth) {
      return iree_ok_status();
    }
    loom_value_id_t inner_input = loom_scalar_trunci_input(input_def);
    return loom_scalar_replace_single_result_with_integer_resize(
        op, rewriter, inner_input,
        loom_module_value_type(rewriter->module, inner_input),
        LOOM_OP_SCALAR_EXTSI);
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_bitcast_canonicalize(loom_op_t* op,
                                               loom_rewriter_t* rewriter) {
  loom_value_id_t input = loom_scalar_bitcast_input(op);
  loom_type_t input_type = loom_module_value_type(rewriter->module, input);
  loom_type_t result_type = loom_scalar_single_result_type(rewriter, op);
  if (loom_type_equal(input_type, result_type)) {
    return loom_scalar_replace_single_result_with_value(op, rewriter, input);
  }

  loom_op_t* input_def = loom_scalar_defining_op(rewriter, input);
  if (!input_def || !loom_scalar_bitcast_isa(input_def))
    return iree_ok_status();
  loom_value_id_t inner_input = loom_scalar_bitcast_input(input_def);
  if (!loom_type_equal(loom_module_value_type(rewriter->module, inner_input),
                       result_type)) {
    return iree_ok_status();
  }
  return loom_scalar_replace_single_result_with_value(op, rewriter,
                                                      inner_input);
}
