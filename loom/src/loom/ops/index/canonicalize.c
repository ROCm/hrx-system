// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"
#include "loom/ops/index/compare.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/math.h"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static bool loom_index_type_is_index(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX;
}

static bool loom_index_query_exact_i64(loom_rewriter_t* rewriter,
                                       loom_value_id_t value,
                                       int64_t* out_value) {
  loom_value_facts_t facts = loom_rewriter_value_facts(rewriter, value);
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

static bool loom_index_value_facts_are_exact_i64(loom_rewriter_t* rewriter,
                                                 loom_value_id_t value,
                                                 int64_t expected_value) {
  int64_t actual_value = 0;
  return loom_index_query_exact_i64(rewriter, value, &actual_value) &&
         actual_value == expected_value;
}

static bool loom_index_values_are_same_or_same_exact_i64(
    loom_rewriter_t* rewriter, loom_value_id_t lhs, loom_value_id_t rhs) {
  if (lhs == rhs) return true;
  int64_t lhs_value = 0;
  int64_t rhs_value = 0;
  return loom_index_query_exact_i64(rewriter, lhs, &lhs_value) &&
         loom_index_query_exact_i64(rewriter, rhs, &rhs_value) &&
         lhs_value == rhs_value;
}

static bool loom_index_value_facts_are_non_negative(loom_rewriter_t* rewriter,
                                                    loom_value_id_t value) {
  return loom_value_facts_is_non_negative(
      loom_rewriter_value_facts(rewriter, value));
}

static bool loom_index_value_facts_are_positive(loom_rewriter_t* rewriter,
                                                loom_value_id_t value) {
  return loom_value_facts_is_positive(
      loom_rewriter_value_facts(rewriter, value));
}

static loom_op_t* loom_index_defining_op(loom_rewriter_t* rewriter,
                                         loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      (iree_host_size_t)value_id >= rewriter->module->values.count) {
    return NULL;
  }
  loom_value_t* value = loom_module_value(rewriter->module, value_id);
  if (loom_value_is_block_arg(value)) return NULL;
  return loom_value_def_op(value);
}

static loom_value_id_t loom_index_strip_identity_assume(
    loom_rewriter_t* rewriter, loom_value_id_t value_id) {
  for (uint8_t depth = 0; depth < 4; ++depth) {
    loom_op_t* assume_op = loom_index_defining_op(rewriter, value_id);
    if (!assume_op || !loom_index_assume_isa(assume_op)) return value_id;
    loom_value_slice_t results = loom_index_assume_results(assume_op);
    loom_value_slice_t values = loom_index_assume_values(assume_op);
    if (results.count != values.count) return value_id;
    bool stripped = false;
    for (uint16_t i = 0; i < results.count; ++i) {
      if (results.values[i] != value_id) continue;
      value_id = values.values[i];
      stripped = true;
      break;
    }
    if (!stripped) return value_id;
  }
  return value_id;
}

static iree_status_t loom_index_replace_single_result_with_value(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t replacement) {
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static iree_status_t loom_index_replace_single_result_with_index_constant(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_type_t result_type,
    int64_t value) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_constant_build(&rewriter->builder, loom_attr_i64(value),
                                result_type, op->location, &constant_op));
  loom_value_id_t replacement = loom_index_constant_result(constant_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_index_replace_single_result_with_value(op, rewriter, replacement);
}

static iree_status_t loom_index_replace_single_result_with_i1_constant(
    loom_op_t* op, loom_rewriter_t* rewriter, bool value) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      &rewriter->builder, loom_attr_bool(value),
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), op->location, &constant_op));
  loom_value_id_t replacement = loom_scalar_constant_result(constant_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_index_replace_single_result_with_value(op, rewriter, replacement);
}

static iree_status_t loom_index_replace_single_result_with_binary_op(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_kind_t kind,
    loom_value_id_t lhs, loom_value_id_t rhs) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_op_const_results(op)[0]);

  loom_op_t* replacement_op = NULL;
  switch (kind) {
    case LOOM_OP_INDEX_ADD: {
      IREE_RETURN_IF_ERROR(loom_index_add_build(&rewriter->builder, lhs, rhs,
                                                result_type, op->location,
                                                &replacement_op));
      break;
    }
    case LOOM_OP_INDEX_MUL: {
      IREE_RETURN_IF_ERROR(loom_index_mul_build(&rewriter->builder, lhs, rhs,
                                                result_type, op->location,
                                                &replacement_op));
      break;
    }
    case LOOM_OP_INDEX_ANDI: {
      IREE_RETURN_IF_ERROR(loom_index_andi_build(&rewriter->builder, lhs, rhs,
                                                 result_type, op->location,
                                                 &replacement_op));
      break;
    }
    case LOOM_OP_INDEX_ORI: {
      IREE_RETURN_IF_ERROR(loom_index_ori_build(&rewriter->builder, lhs, rhs,
                                                result_type, op->location,
                                                &replacement_op));
      break;
    }
    case LOOM_OP_INDEX_XORI: {
      IREE_RETURN_IF_ERROR(loom_index_xori_build(&rewriter->builder, lhs, rhs,
                                                 result_type, op->location,
                                                 &replacement_op));
      break;
    }
    case LOOM_OP_INDEX_SHLI: {
      IREE_RETURN_IF_ERROR(loom_index_shli_build(&rewriter->builder, lhs, rhs,
                                                 result_type, op->location,
                                                 &replacement_op));
      break;
    }
    case LOOM_OP_INDEX_SHRSI: {
      IREE_RETURN_IF_ERROR(loom_index_shrsi_build(&rewriter->builder, lhs, rhs,
                                                  result_type, op->location,
                                                  &replacement_op));
      break;
    }
    case LOOM_OP_INDEX_SHRUI: {
      IREE_RETURN_IF_ERROR(loom_index_shrui_build(&rewriter->builder, lhs, rhs,
                                                  result_type, op->location,
                                                  &replacement_op));
      break;
    }
    case LOOM_OP_INDEX_ROTLI: {
      IREE_RETURN_IF_ERROR(loom_index_rotli_build(&rewriter->builder, lhs, rhs,
                                                  result_type, op->location,
                                                  &replacement_op));
      break;
    }
    case LOOM_OP_INDEX_ROTRI: {
      IREE_RETURN_IF_ERROR(loom_index_rotri_build(&rewriter->builder, lhs, rhs,
                                                  result_type, op->location,
                                                  &replacement_op));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported replacement index op kind %u",
                              (unsigned)kind);
  }

  loom_value_id_t replacement = loom_op_const_results(replacement_op)[0];
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_index_replace_single_result_with_value(op, rewriter, replacement);
}

static iree_status_t loom_index_materialize_or_reuse_index_constant(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t candidate,
    int64_t value, loom_value_id_t* out_value) {
  int64_t candidate_value = 0;
  if (candidate != LOOM_VALUE_ID_INVALID &&
      loom_index_query_exact_i64(rewriter, candidate, &candidate_value) &&
      candidate_value == value) {
    *out_value = candidate;
    return iree_ok_status();
  }

  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      &rewriter->builder, loom_attr_i64(value),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), op->location, &constant_op));
  *out_value = loom_index_constant_result(constant_op);
  return iree_ok_status();
}

static iree_status_t loom_index_replace_single_result_with_binary_constant_op(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_kind_t kind,
    loom_value_id_t lhs, int64_t rhs, loom_value_id_t reusable_constant) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t rhs_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_index_materialize_or_reuse_index_constant(
      op, rewriter, reusable_constant, rhs, &rhs_value));
  return loom_index_replace_single_result_with_binary_op(op, rewriter, kind,
                                                         lhs, rhs_value);
}

static iree_status_t loom_index_replace_single_result_with_madd_op(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t a,
    loom_value_id_t b, loom_value_id_t c) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_op_const_results(op)[0]);

  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_madd_build(
      &rewriter->builder, a, b, c, result_type, op->location, &replacement_op));
  loom_value_id_t replacement = loom_index_madd_result(replacement_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_index_replace_single_result_with_value(op, rewriter, replacement);
}

static iree_status_t loom_index_replace_single_result_with_scaled_madd_op(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t value,
    int64_t factor, loom_value_id_t factor_value, loom_value_id_t addend) {
  if (factor == 1) {
    return loom_index_replace_single_result_with_binary_op(
        op, rewriter, LOOM_OP_INDEX_ADD, addend, value);
  }
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t scale_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_index_materialize_or_reuse_index_constant(
      op, rewriter, factor_value, factor, &scale_value));
  return loom_index_replace_single_result_with_madd_op(op, rewriter, value,
                                                       scale_value, addend);
}

static bool loom_index_shift_amount_is_valid(loom_type_t type, int64_t amount) {
  int32_t bitwidth = loom_scalar_type_bitwidth(loom_type_element_type(type));
  return amount >= 0 && bitwidth > 0 && amount < bitwidth;
}

static bool loom_index_match_mul_with_exact_positive_factor(
    loom_rewriter_t* rewriter, loom_op_t* mul_op, loom_value_id_t* out_value,
    int64_t* out_factor) {
  if (!mul_op || !loom_index_mul_isa(mul_op)) return false;

  loom_value_id_t lhs = loom_index_mul_lhs(mul_op);
  loom_value_id_t rhs = loom_index_mul_rhs(mul_op);
  int64_t lhs_factor = 0;
  if (loom_index_query_exact_i64(rewriter, lhs, &lhs_factor) &&
      lhs_factor > 0) {
    *out_value = rhs;
    *out_factor = lhs_factor;
    return true;
  }
  int64_t rhs_factor = 0;
  if (loom_index_query_exact_i64(rewriter, rhs, &rhs_factor) &&
      rhs_factor > 0) {
    *out_value = lhs;
    *out_factor = rhs_factor;
    return true;
  }
  return false;
}

typedef struct loom_index_scaled_value_t {
  loom_value_id_t value;
  int64_t factor;
  loom_value_id_t factor_value;
} loom_index_scaled_value_t;

static bool loom_index_match_scaled_value_with_exact_positive_factor(
    loom_rewriter_t* rewriter, loom_op_t* op,
    loom_index_scaled_value_t* out_scaled_value) {
  if (!op) return false;
  if (loom_index_mul_isa(op)) {
    loom_value_id_t scaled_value = LOOM_VALUE_ID_INVALID;
    int64_t factor = 0;
    if (!loom_index_match_mul_with_exact_positive_factor(
            rewriter, op, &scaled_value, &factor)) {
      return false;
    }
    loom_value_id_t lhs = loom_index_mul_lhs(op);
    loom_value_id_t rhs = loom_index_mul_rhs(op);
    out_scaled_value->value = scaled_value;
    out_scaled_value->factor = factor;
    out_scaled_value->factor_value = scaled_value == lhs ? rhs : lhs;
    return true;
  }
  if (loom_index_shli_isa(op)) {
    loom_value_id_t amount_value = loom_index_shli_rhs(op);
    int64_t amount = 0;
    if (!loom_index_query_exact_i64(rewriter, amount_value, &amount)) {
      return false;
    }
    loom_type_t result_type =
        loom_module_value_type(rewriter->module, loom_index_shli_result(op));
    if (!loom_index_shift_amount_is_valid(result_type, amount) ||
        amount >= 63) {
      return false;
    }
    out_scaled_value->value = loom_index_shli_lhs(op);
    out_scaled_value->factor = ((int64_t)1) << amount;
    out_scaled_value->factor_value = LOOM_VALUE_ID_INVALID;
    return true;
  }
  return false;
}

static bool loom_index_match_madd_with_exact_positive_factor(
    loom_rewriter_t* rewriter, loom_op_t* madd_op, loom_value_id_t* out_value,
    int64_t* out_factor, loom_value_id_t* out_addend,
    int64_t* out_addend_value) {
  if (!madd_op || !loom_index_madd_isa(madd_op)) return false;

  loom_value_id_t a = loom_index_madd_a(madd_op);
  loom_value_id_t b = loom_index_madd_b(madd_op);
  int64_t a_factor = 0;
  if (loom_index_query_exact_i64(rewriter, a, &a_factor) && a_factor > 0) {
    *out_value = b;
    *out_factor = a_factor;
  } else {
    int64_t b_factor = 0;
    if (!loom_index_query_exact_i64(rewriter, b, &b_factor) || b_factor <= 0) {
      return false;
    }
    *out_value = a;
    *out_factor = b_factor;
  }

  *out_addend = loom_index_madd_c(madd_op);
  return loom_index_query_exact_i64(rewriter, *out_addend, out_addend_value);
}

typedef enum loom_index_radix_term_kind_e {
  LOOM_INDEX_RADIX_TERM_QUOTIENT,
  LOOM_INDEX_RADIX_TERM_REMAINDER,
} loom_index_radix_term_kind_t;

typedef struct loom_index_scaled_radix_term_t {
  // Whether this term is the quotient or remainder half of the reconstruction.
  loom_index_radix_term_kind_t kind;
  // Original value split by the radix operation.
  loom_value_id_t source;
  // Exact positive divisor or equivalent power-of-two radix.
  int64_t radix;
  // Exact positive scale applied to this term.
  int64_t scale;
  // Existing SSA value for the scale, when one can be reused.
  loom_value_id_t scale_value;
} loom_index_scaled_radix_term_t;

static bool loom_index_exact_positive_power_of_two_plus_one(
    int64_t mask, int64_t* out_radix) {
  if (mask <= 0 || mask == INT64_MAX) return false;
  int64_t radix = mask + 1;
  if (!loom_is_power_of_two_i64(radix)) return false;
  *out_radix = radix;
  return true;
}

static bool loom_index_exact_shift_radix(loom_rewriter_t* rewriter,
                                         loom_value_id_t shift_value,
                                         int64_t* out_radix) {
  int64_t shift = 0;
  if (!loom_index_query_exact_i64(rewriter, shift_value, &shift) || shift < 0 ||
      shift >= 62) {
    return false;
  }
  *out_radix = (int64_t)1 << shift;
  return *out_radix > 1;
}

static bool loom_index_match_scaled_term(loom_rewriter_t* rewriter,
                                         loom_value_id_t term,
                                         loom_value_id_t* out_value,
                                         int64_t* out_scale,
                                         loom_value_id_t* out_scale_value) {
  term = loom_index_strip_identity_assume(rewriter, term);
  *out_value = term;
  *out_scale = 1;
  *out_scale_value = LOOM_VALUE_ID_INVALID;

  loom_op_t* term_def = loom_index_defining_op(rewriter, term);
  if (!term_def || !loom_index_mul_isa(term_def)) return true;

  loom_value_id_t lhs = loom_index_mul_lhs(term_def);
  loom_value_id_t rhs = loom_index_mul_rhs(term_def);
  int64_t lhs_scale = 0;
  if (loom_index_query_exact_i64(rewriter, lhs, &lhs_scale) && lhs_scale > 0) {
    *out_value = rhs;
    *out_scale = lhs_scale;
    *out_scale_value = lhs;
    return true;
  }
  int64_t rhs_scale = 0;
  if (loom_index_query_exact_i64(rewriter, rhs, &rhs_scale) && rhs_scale > 0) {
    *out_value = lhs;
    *out_scale = rhs_scale;
    *out_scale_value = rhs;
    return true;
  }
  return false;
}

static bool loom_index_match_scaled_radix_term(
    loom_rewriter_t* rewriter, loom_value_id_t term,
    loom_index_scaled_radix_term_t* out_term) {
  *out_term = (loom_index_scaled_radix_term_t){
      .kind = LOOM_INDEX_RADIX_TERM_QUOTIENT,
      .source = LOOM_VALUE_ID_INVALID,
      .radix = 0,
      .scale = 1,
      .scale_value = LOOM_VALUE_ID_INVALID,
  };

  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  if (!loom_index_match_scaled_term(rewriter, term, &value, &out_term->scale,
                                    &out_term->scale_value)) {
    return false;
  }
  value = loom_index_strip_identity_assume(rewriter, value);

  loom_op_t* value_def = loom_index_defining_op(rewriter, value);
  if (value_def && loom_index_div_isa(value_def)) {
    int64_t divisor = 0;
    if (!loom_index_query_exact_i64(rewriter, loom_index_div_rhs(value_def),
                                    &divisor) ||
        divisor <= 1) {
      return false;
    }
    out_term->kind = LOOM_INDEX_RADIX_TERM_QUOTIENT;
    out_term->source = loom_index_div_lhs(value_def);
    out_term->radix = divisor;
    return true;
  }
  if (value_def && loom_index_rem_isa(value_def)) {
    int64_t divisor = 0;
    if (!loom_index_query_exact_i64(rewriter, loom_index_rem_rhs(value_def),
                                    &divisor) ||
        divisor <= 1) {
      return false;
    }
    out_term->kind = LOOM_INDEX_RADIX_TERM_REMAINDER;
    out_term->source = loom_index_rem_lhs(value_def);
    out_term->radix = divisor;
    return true;
  }
  if (value_def && loom_index_shrui_isa(value_def)) {
    int64_t radix = 0;
    if (!loom_index_exact_shift_radix(rewriter, loom_index_shrui_rhs(value_def),
                                      &radix)) {
      return false;
    }
    out_term->kind = LOOM_INDEX_RADIX_TERM_QUOTIENT;
    out_term->source = loom_index_shrui_lhs(value_def);
    out_term->radix = radix;
    return true;
  }
  if (value_def && loom_index_andi_isa(value_def)) {
    int64_t mask = 0;
    int64_t radix = 0;
    if (!loom_index_query_exact_i64(rewriter, loom_index_andi_rhs(value_def),
                                    &mask) ||
        !loom_index_exact_positive_power_of_two_plus_one(mask, &radix)) {
      return false;
    }
    out_term->kind = LOOM_INDEX_RADIX_TERM_REMAINDER;
    out_term->source = loom_index_andi_lhs(value_def);
    out_term->radix = radix;
    return true;
  }
  return false;
}

static bool loom_index_scaled_radix_terms_are_complementary(
    loom_rewriter_t* rewriter, const loom_index_scaled_radix_term_t* lhs,
    const loom_index_scaled_radix_term_t* rhs,
    const loom_index_scaled_radix_term_t** out_quotient,
    const loom_index_scaled_radix_term_t** out_remainder) {
  if (lhs->kind == rhs->kind) return false;
  const loom_index_scaled_radix_term_t* quotient =
      lhs->kind == LOOM_INDEX_RADIX_TERM_QUOTIENT ? lhs : rhs;
  const loom_index_scaled_radix_term_t* remainder =
      lhs->kind == LOOM_INDEX_RADIX_TERM_REMAINDER ? lhs : rhs;
  if (quotient->source != remainder->source) return false;
  if (quotient->radix != remainder->radix) return false;
  if (!loom_index_value_facts_are_non_negative(rewriter, quotient->source)) {
    return false;
  }
  int64_t expected_quotient_scale = 0;
  if (!loom_checked_mul_i64(remainder->scale, quotient->radix,
                            &expected_quotient_scale) ||
      quotient->scale != expected_quotient_scale) {
    return false;
  }
  *out_quotient = quotient;
  *out_remainder = remainder;
  return true;
}

static bool loom_index_integer_value_is_all_ones(loom_type_t type,
                                                 int64_t value) {
  if (value == -1) return true;
  int32_t bitwidth = loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (bitwidth <= 0 || bitwidth >= 63) return false;
  return value == (((int64_t)1 << bitwidth) - 1);
}

static iree_status_t loom_index_replace_single_result_with_scaled_value(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t value,
    int64_t scale, loom_value_id_t reusable_constant) {
  if (scale == 1) {
    return loom_index_replace_single_result_with_value(op, rewriter, value);
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t scale_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_index_materialize_or_reuse_index_constant(
      op, rewriter, reusable_constant, scale, &scale_value));

  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_op_const_results(op)[0]);
  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_mul_build(&rewriter->builder, value,
                                            scale_value, result_type,
                                            op->location, &replacement_op));
  loom_value_id_t replacement = loom_index_mul_result(replacement_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_index_replace_single_result_with_value(op, rewriter, replacement);
}

static iree_status_t loom_index_try_fold_scaled_radix_reconstruction(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t lhs,
    loom_value_id_t rhs, bool* out_folded) {
  *out_folded = false;
  loom_index_scaled_radix_term_t lhs_term;
  loom_index_scaled_radix_term_t rhs_term;
  if (!loom_index_match_scaled_radix_term(rewriter, lhs, &lhs_term) ||
      !loom_index_match_scaled_radix_term(rewriter, rhs, &rhs_term)) {
    return iree_ok_status();
  }
  const loom_index_scaled_radix_term_t* quotient = NULL;
  const loom_index_scaled_radix_term_t* remainder = NULL;
  if (!loom_index_scaled_radix_terms_are_complementary(
          rewriter, &lhs_term, &rhs_term, &quotient, &remainder)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_index_replace_single_result_with_scaled_value(
      op, rewriter, quotient->source, remainder->scale,
      remainder->scale_value));
  *out_folded = true;
  return iree_ok_status();
}

static bool loom_index_match_scaled_radix_product(
    loom_rewriter_t* rewriter, loom_value_id_t value, loom_value_id_t scale,
    loom_index_scaled_radix_term_t* out_term) {
  int64_t scale_factor = 0;
  if (!loom_index_query_exact_i64(rewriter, scale, &scale_factor) ||
      scale_factor <= 0) {
    return false;
  }
  if (!loom_index_match_scaled_radix_term(rewriter, value, out_term)) {
    return false;
  }

  int64_t combined_scale = 0;
  if (!loom_checked_mul_i64(out_term->scale, scale_factor, &combined_scale)) {
    return false;
  }
  out_term->scale = combined_scale;
  out_term->scale_value = out_term->scale_value == LOOM_VALUE_ID_INVALID
                              ? scale
                              : LOOM_VALUE_ID_INVALID;
  return true;
}

static iree_status_t loom_index_try_fold_madd_scaled_radix_reconstruction(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t a,
    loom_value_id_t b, loom_value_id_t c, bool* out_folded) {
  *out_folded = false;
  loom_index_scaled_radix_term_t product_term;
  if (!loom_index_match_scaled_radix_product(rewriter, a, b, &product_term) &&
      !loom_index_match_scaled_radix_product(rewriter, b, a, &product_term)) {
    return iree_ok_status();
  }

  loom_index_scaled_radix_term_t addend_term;
  if (!loom_index_match_scaled_radix_term(rewriter, c, &addend_term)) {
    return iree_ok_status();
  }
  const loom_index_scaled_radix_term_t* quotient = NULL;
  const loom_index_scaled_radix_term_t* remainder = NULL;
  if (!loom_index_scaled_radix_terms_are_complementary(
          rewriter, &product_term, &addend_term, &quotient, &remainder)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_index_replace_single_result_with_scaled_value(
      op, rewriter, quotient->source, remainder->scale,
      remainder->scale_value));
  *out_folded = true;
  return iree_ok_status();
}

static iree_status_t loom_index_replace_single_result_with_scaled_add(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t value,
    int64_t scale, loom_value_id_t reusable_scale, int64_t addend,
    loom_value_id_t reusable_addend) {
  if (addend == 0) {
    return loom_index_replace_single_result_with_scaled_value(
        op, rewriter, value, scale, reusable_scale);
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t addend_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_index_materialize_or_reuse_index_constant(
      op, rewriter, reusable_addend, addend, &addend_value));

  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  if (scale == 1) {
    loom_op_t* replacement_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_index_add_build(&rewriter->builder, value, addend_value,
                             loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                             op->location, &replacement_op));
    replacement = loom_index_add_result(replacement_op);
  } else {
    loom_value_id_t scale_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_index_materialize_or_reuse_index_constant(
        op, rewriter, reusable_scale, scale, &scale_value));
    loom_op_t* replacement_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_madd_build(
        &rewriter->builder, value, scale_value, addend_value,
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), op->location,
        &replacement_op));
    replacement = loom_index_madd_result(replacement_op);
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_index_replace_single_result_with_value(op, rewriter, replacement);
}

//===----------------------------------------------------------------------===//
// Arithmetic
//===----------------------------------------------------------------------===//

typedef enum loom_index_minmax_kind_e {
  LOOM_INDEX_MINMAX_MIN,
  LOOM_INDEX_MINMAX_MAX,
} loom_index_minmax_kind_t;

iree_status_t loom_index_add_canonicalize(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_index_add_lhs(op);
  loom_value_id_t rhs = loom_index_add_rhs(op);
  if (loom_index_value_facts_are_exact_i64(rewriter, lhs, 0)) {
    return loom_index_replace_single_result_with_value(op, rewriter, rhs);
  }
  if (loom_index_value_facts_are_exact_i64(rewriter, rhs, 0)) {
    return loom_index_replace_single_result_with_value(op, rewriter, lhs);
  }

  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_index_add_result(op));
  if (!loom_index_type_is_index(result_type)) return iree_ok_status();

  bool folded_scaled_radix = false;
  IREE_RETURN_IF_ERROR(loom_index_try_fold_scaled_radix_reconstruction(
      op, rewriter, lhs, rhs, &folded_scaled_radix));
  if (folded_scaled_radix) return iree_ok_status();

  loom_op_t* lhs_def = loom_index_defining_op(rewriter, lhs);
  loom_index_scaled_value_t lhs_scaled;
  if (loom_index_match_scaled_value_with_exact_positive_factor(
          rewriter, lhs_def, &lhs_scaled)) {
    if (lhs_scaled.factor == 1) {
      return iree_ok_status();
    }
    return loom_index_replace_single_result_with_scaled_madd_op(
        op, rewriter, lhs_scaled.value, lhs_scaled.factor,
        lhs_scaled.factor_value, rhs);
  }
  loom_op_t* rhs_def = loom_index_defining_op(rewriter, rhs);
  loom_index_scaled_value_t rhs_scaled;
  if (loom_index_match_scaled_value_with_exact_positive_factor(
          rewriter, rhs_def, &rhs_scaled)) {
    if (rhs_scaled.factor == 1) {
      return iree_ok_status();
    }
    return loom_index_replace_single_result_with_scaled_madd_op(
        op, rewriter, rhs_scaled.value, rhs_scaled.factor,
        rhs_scaled.factor_value, lhs);
  }
  return iree_ok_status();
}

iree_status_t loom_index_sub_canonicalize(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_index_sub_lhs(op);
  loom_value_id_t rhs = loom_index_sub_rhs(op);
  if (loom_index_value_facts_are_exact_i64(rewriter, rhs, 0)) {
    return loom_index_replace_single_result_with_value(op, rewriter, lhs);
  }
  if (lhs == rhs) {
    loom_type_t result_type =
        loom_module_value_type(rewriter->module, loom_index_sub_result(op));
    return loom_index_replace_single_result_with_index_constant(op, rewriter,
                                                                result_type, 0);
  }
  return iree_ok_status();
}

iree_status_t loom_index_mul_canonicalize(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_index_mul_lhs(op);
  loom_value_id_t rhs = loom_index_mul_rhs(op);
  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_index_mul_result(op));
  if (loom_index_value_facts_are_exact_i64(rewriter, lhs, 0) ||
      loom_index_value_facts_are_exact_i64(rewriter, rhs, 0)) {
    return loom_index_replace_single_result_with_index_constant(op, rewriter,
                                                                result_type, 0);
  }
  if (loom_index_value_facts_are_exact_i64(rewriter, lhs, 1)) {
    return loom_index_replace_single_result_with_value(op, rewriter, rhs);
  }
  if (loom_index_value_facts_are_exact_i64(rewriter, rhs, 1)) {
    return loom_index_replace_single_result_with_value(op, rewriter, lhs);
  }
  loom_index_scaled_value_t scaled;
  if (loom_index_match_scaled_value_with_exact_positive_factor(rewriter, op,
                                                               &scaled) &&
      scaled.factor > 1 && loom_is_power_of_two_i64(scaled.factor)) {
    int64_t amount = loom_ilog2_i64(scaled.factor);
    if (loom_index_shift_amount_is_valid(result_type, amount)) {
      return loom_index_replace_single_result_with_binary_constant_op(
          op, rewriter, LOOM_OP_INDEX_SHLI, scaled.value, amount,
          LOOM_VALUE_ID_INVALID);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_index_scale_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  loom_value_id_t index = loom_index_scale_index(op);
  loom_value_id_t stride = loom_index_scale_stride(op);
  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_index_scale_result(op));
  if (loom_index_value_facts_are_exact_i64(rewriter, index, 0) ||
      loom_index_value_facts_are_exact_i64(rewriter, stride, 0)) {
    return loom_index_replace_single_result_with_index_constant(op, rewriter,
                                                                result_type, 0);
  }
  if (loom_index_value_facts_are_exact_i64(rewriter, index, 1)) {
    return loom_index_replace_single_result_with_value(op, rewriter, stride);
  }
  return iree_ok_status();
}

iree_status_t loom_index_div_canonicalize(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_index_div_lhs(op);
  loom_value_id_t rhs = loom_index_div_rhs(op);
  if (lhs == rhs && loom_index_value_facts_are_positive(rewriter, lhs)) {
    loom_type_t result_type =
        loom_module_value_type(rewriter->module, loom_index_div_result(op));
    return loom_index_replace_single_result_with_index_constant(op, rewriter,
                                                                result_type, 1);
  }
  if (loom_index_value_facts_are_exact_i64(rewriter, lhs, 0) &&
      loom_index_value_facts_are_positive(rewriter, rhs)) {
    loom_type_t result_type =
        loom_module_value_type(rewriter->module, loom_index_div_result(op));
    return loom_index_replace_single_result_with_index_constant(op, rewriter,
                                                                result_type, 0);
  }
  if (loom_index_value_facts_are_exact_i64(rewriter, loom_index_div_rhs(op),
                                           1)) {
    return loom_index_replace_single_result_with_value(op, rewriter,
                                                       loom_index_div_lhs(op));
  }

  int64_t divisor = 0;
  if (!loom_index_query_exact_i64(rewriter, rhs, &divisor) || divisor <= 0) {
    return iree_ok_status();
  }

  loom_op_t* lhs_def = loom_index_defining_op(rewriter, lhs);
  loom_value_id_t scaled_value = LOOM_VALUE_ID_INVALID;
  int64_t scale = 0;
  if (loom_index_match_mul_with_exact_positive_factor(rewriter, lhs_def,
                                                      &scaled_value, &scale) &&
      loom_index_value_facts_are_non_negative(rewriter, lhs) &&
      loom_index_value_facts_are_non_negative(rewriter, scaled_value) &&
      scale % divisor == 0) {
    return loom_index_replace_single_result_with_scaled_value(
        op, rewriter, scaled_value, scale / divisor, rhs);
  }

  if (lhs_def && loom_index_madd_isa(lhs_def)) {
    loom_value_id_t addend_value = LOOM_VALUE_ID_INVALID;
    int64_t addend = 0;
    if (loom_index_match_madd_with_exact_positive_factor(
            rewriter, lhs_def, &scaled_value, &scale, &addend_value, &addend) &&
        addend >= 0 && addend % divisor == 0 &&
        loom_index_value_facts_are_non_negative(rewriter, lhs) &&
        loom_index_value_facts_are_non_negative(rewriter, scaled_value) &&
        scale % divisor == 0) {
      return loom_index_replace_single_result_with_scaled_add(
          op, rewriter, scaled_value, scale / divisor, rhs, addend / divisor,
          addend_value);
    }
  }
  if (loom_is_power_of_two_i64(divisor) &&
      loom_index_value_facts_are_non_negative(rewriter, lhs)) {
    return loom_index_replace_single_result_with_binary_constant_op(
        op, rewriter, LOOM_OP_INDEX_SHRUI, lhs, loom_ilog2_i64(divisor), rhs);
  }
  return iree_ok_status();
}

iree_status_t loom_index_rem_canonicalize(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_index_rem_lhs(op);
  loom_value_id_t rhs = loom_index_rem_rhs(op);
  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_index_rem_result(op));
  if (lhs == rhs && loom_index_value_facts_are_positive(rewriter, rhs)) {
    return loom_index_replace_single_result_with_index_constant(op, rewriter,
                                                                result_type, 0);
  }
  if (loom_index_value_facts_are_exact_i64(rewriter, lhs, 0) &&
      loom_index_value_facts_are_positive(rewriter, rhs)) {
    return loom_index_replace_single_result_with_index_constant(op, rewriter,
                                                                result_type, 0);
  }
  int64_t divisor = 0;
  if (loom_index_query_exact_i64(rewriter, rhs, &divisor) && divisor > 0) {
    loom_value_facts_t lhs_facts = loom_rewriter_value_facts(rewriter, lhs);
    if (!loom_value_facts_is_float(lhs_facts) &&
        loom_value_facts_is_non_negative(lhs_facts) &&
        loom_value_facts_divisible_by(lhs_facts, divisor)) {
      return loom_index_replace_single_result_with_index_constant(
          op, rewriter, result_type, 0);
    }
  }
  if (loom_index_value_facts_are_exact_i64(rewriter, loom_index_rem_rhs(op),
                                           1)) {
    return loom_index_replace_single_result_with_index_constant(op, rewriter,
                                                                result_type, 0);
  }
  if (loom_index_query_exact_i64(rewriter, rhs, &divisor) &&
      loom_is_power_of_two_i64(divisor) &&
      loom_index_value_facts_are_non_negative(rewriter, lhs)) {
    return loom_index_replace_single_result_with_binary_constant_op(
        op, rewriter, LOOM_OP_INDEX_ANDI, lhs, divisor - 1, rhs);
  }
  return iree_ok_status();
}

static iree_status_t loom_index_minmax_canonicalize(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_index_minmax_kind_t kind) {
  if (lhs == rhs) {
    return loom_index_replace_single_result_with_value(op, rewriter, lhs);
  }

  loom_value_facts_t lhs_facts = loom_rewriter_value_facts(rewriter, lhs);
  loom_value_facts_t rhs_facts = loom_rewriter_value_facts(rewriter, rhs);
  bool lhs_le_rhs = false;
  if (!loom_index_cmp_result_from_facts(LOOM_INDEX_CMP_PREDICATE_SLE,
                                        &lhs_facts, &rhs_facts, &lhs_le_rhs)) {
    return iree_ok_status();
  }

  loom_value_id_t replacement = kind == LOOM_INDEX_MINMAX_MIN
                                    ? (lhs_le_rhs ? lhs : rhs)
                                    : (lhs_le_rhs ? rhs : lhs);
  return loom_index_replace_single_result_with_value(op, rewriter, replacement);
}

iree_status_t loom_index_min_canonicalize(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  return loom_index_minmax_canonicalize(op, rewriter, loom_index_min_lhs(op),
                                        loom_index_min_rhs(op),
                                        LOOM_INDEX_MINMAX_MIN);
}

iree_status_t loom_index_max_canonicalize(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  return loom_index_minmax_canonicalize(op, rewriter, loom_index_max_lhs(op),
                                        loom_index_max_rhs(op),
                                        LOOM_INDEX_MINMAX_MAX);
}

iree_status_t loom_index_madd_canonicalize(loom_op_t* op,
                                           loom_rewriter_t* rewriter) {
  loom_value_id_t a = loom_index_madd_a(op);
  loom_value_id_t b = loom_index_madd_b(op);
  loom_value_id_t c = loom_index_madd_c(op);
  if (loom_index_value_facts_are_exact_i64(rewriter, a, 0) ||
      loom_index_value_facts_are_exact_i64(rewriter, b, 0)) {
    return loom_index_replace_single_result_with_value(op, rewriter, c);
  }
  if (loom_index_value_facts_are_exact_i64(rewriter, c, 0)) {
    return loom_index_replace_single_result_with_binary_op(
        op, rewriter, LOOM_OP_INDEX_MUL, a, b);
  }
  if (loom_index_value_facts_are_exact_i64(rewriter, a, 1)) {
    return loom_index_replace_single_result_with_binary_op(
        op, rewriter, LOOM_OP_INDEX_ADD, c, b);
  }
  if (loom_index_value_facts_are_exact_i64(rewriter, b, 1)) {
    return loom_index_replace_single_result_with_binary_op(
        op, rewriter, LOOM_OP_INDEX_ADD, c, a);
  }

  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_index_madd_result(op));
  if (!loom_index_type_is_index(result_type)) return iree_ok_status();
  bool folded_scaled_radix = false;
  IREE_RETURN_IF_ERROR(loom_index_try_fold_madd_scaled_radix_reconstruction(
      op, rewriter, a, b, c, &folded_scaled_radix));
  if (folded_scaled_radix) return iree_ok_status();

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Bitwise
//===----------------------------------------------------------------------===//

iree_status_t loom_index_andi_canonicalize(loom_op_t* op,
                                           loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_index_andi_lhs(op);
  loom_value_id_t rhs = loom_index_andi_rhs(op);
  if (lhs == rhs) {
    return loom_index_replace_single_result_with_value(op, rewriter, lhs);
  }
  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_index_andi_result(op));
  if (loom_index_value_facts_are_exact_i64(rewriter, lhs, 0) ||
      loom_index_value_facts_are_exact_i64(rewriter, rhs, 0)) {
    return loom_index_replace_single_result_with_index_constant(op, rewriter,
                                                                result_type, 0);
  }
  int64_t lhs_value = 0;
  if (loom_index_query_exact_i64(rewriter, lhs, &lhs_value) &&
      loom_index_integer_value_is_all_ones(result_type, lhs_value)) {
    return loom_index_replace_single_result_with_value(op, rewriter, rhs);
  }
  int64_t rhs_value = 0;
  if (loom_index_query_exact_i64(rewriter, rhs, &rhs_value) &&
      loom_index_integer_value_is_all_ones(result_type, rhs_value)) {
    return loom_index_replace_single_result_with_value(op, rewriter, lhs);
  }
  return iree_ok_status();
}

iree_status_t loom_index_ori_canonicalize(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_index_ori_lhs(op);
  loom_value_id_t rhs = loom_index_ori_rhs(op);
  if (lhs == rhs) {
    return loom_index_replace_single_result_with_value(op, rewriter, lhs);
  }
  if (loom_index_value_facts_are_exact_i64(rewriter, lhs, 0)) {
    return loom_index_replace_single_result_with_value(op, rewriter, rhs);
  }
  if (loom_index_value_facts_are_exact_i64(rewriter, rhs, 0)) {
    return loom_index_replace_single_result_with_value(op, rewriter, lhs);
  }
  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_index_ori_result(op));
  int64_t lhs_value = 0;
  if (loom_index_query_exact_i64(rewriter, lhs, &lhs_value) &&
      loom_index_integer_value_is_all_ones(result_type, lhs_value)) {
    return loom_index_replace_single_result_with_value(op, rewriter, lhs);
  }
  int64_t rhs_value = 0;
  if (loom_index_query_exact_i64(rewriter, rhs, &rhs_value) &&
      loom_index_integer_value_is_all_ones(result_type, rhs_value)) {
    return loom_index_replace_single_result_with_value(op, rewriter, rhs);
  }
  return iree_ok_status();
}

iree_status_t loom_index_xori_canonicalize(loom_op_t* op,
                                           loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_index_xori_lhs(op);
  loom_value_id_t rhs = loom_index_xori_rhs(op);
  if (lhs == rhs) {
    loom_type_t result_type =
        loom_module_value_type(rewriter->module, loom_index_xori_result(op));
    return loom_index_replace_single_result_with_index_constant(op, rewriter,
                                                                result_type, 0);
  }
  if (loom_index_value_facts_are_exact_i64(rewriter, lhs, 0)) {
    return loom_index_replace_single_result_with_value(op, rewriter, rhs);
  }
  if (loom_index_value_facts_are_exact_i64(rewriter, rhs, 0)) {
    return loom_index_replace_single_result_with_value(op, rewriter, lhs);
  }

  loom_op_t* lhs_def = loom_index_defining_op(rewriter, lhs);
  if (lhs_def && loom_index_xori_isa(lhs_def)) {
    if (loom_index_values_are_same_or_same_exact_i64(
            rewriter, loom_index_xori_lhs(lhs_def), rhs)) {
      return loom_index_replace_single_result_with_value(
          op, rewriter, loom_index_xori_rhs(lhs_def));
    }
    if (loom_index_values_are_same_or_same_exact_i64(
            rewriter, loom_index_xori_rhs(lhs_def), rhs)) {
      return loom_index_replace_single_result_with_value(
          op, rewriter, loom_index_xori_lhs(lhs_def));
    }
  }
  loom_op_t* rhs_def = loom_index_defining_op(rewriter, rhs);
  if (rhs_def && loom_index_xori_isa(rhs_def)) {
    if (loom_index_values_are_same_or_same_exact_i64(
            rewriter, loom_index_xori_lhs(rhs_def), lhs)) {
      return loom_index_replace_single_result_with_value(
          op, rewriter, loom_index_xori_rhs(rhs_def));
    }
    if (loom_index_values_are_same_or_same_exact_i64(
            rewriter, loom_index_xori_rhs(rhs_def), lhs)) {
      return loom_index_replace_single_result_with_value(
          op, rewriter, loom_index_xori_lhs(rhs_def));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_index_shift_canonicalize(loom_op_t* op,
                                                   loom_rewriter_t* rewriter,
                                                   loom_op_kind_t kind,
                                                   loom_value_id_t lhs,
                                                   loom_value_id_t rhs) {
  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_op_const_results(op)[0]);
  int64_t amount = 0;
  bool has_exact_amount = loom_index_query_exact_i64(rewriter, rhs, &amount);
  if (has_exact_amount) {
    if (amount == 0) {
      return loom_index_replace_single_result_with_value(op, rewriter, lhs);
    }
    if (loom_index_shift_amount_is_valid(result_type, amount) &&
        loom_index_value_facts_are_exact_i64(rewriter, lhs, 0)) {
      return loom_index_replace_single_result_with_index_constant(
          op, rewriter, result_type, 0);
    }
  }

  if (!has_exact_amount) return iree_ok_status();
  loom_op_t* lhs_def = loom_index_defining_op(rewriter, lhs);
  if (!lhs_def || lhs_def->kind != kind) return iree_ok_status();

  int64_t inner_amount = 0;
  if (!loom_index_query_exact_i64(rewriter, loom_op_const_operands(lhs_def)[1],
                                  &inner_amount)) {
    return iree_ok_status();
  }
  int64_t combined_amount = 0;
  if (!loom_checked_add_i64(inner_amount, amount, &combined_amount) ||
      !loom_index_shift_amount_is_valid(result_type, combined_amount)) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t combined_amount_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_index_materialize_or_reuse_index_constant(
      op, rewriter, rhs, combined_amount, &combined_amount_value));
  return loom_index_replace_single_result_with_binary_op(
      op, rewriter, kind, loom_op_const_operands(lhs_def)[0],
      combined_amount_value);
}

iree_status_t loom_index_shli_canonicalize(loom_op_t* op,
                                           loom_rewriter_t* rewriter) {
  return loom_index_shift_canonicalize(op, rewriter, LOOM_OP_INDEX_SHLI,
                                       loom_index_shli_lhs(op),
                                       loom_index_shli_rhs(op));
}

iree_status_t loom_index_shrsi_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_index_shrsi_lhs(op);
  loom_value_id_t rhs = loom_index_shrsi_rhs(op);
  if (loom_index_value_facts_are_exact_i64(rewriter, rhs, 0)) {
    return loom_index_replace_single_result_with_value(op, rewriter, lhs);
  }
  if (loom_index_value_facts_are_non_negative(rewriter, lhs)) {
    return loom_index_replace_single_result_with_binary_op(
        op, rewriter, LOOM_OP_INDEX_SHRUI, lhs, rhs);
  }
  return loom_index_shift_canonicalize(op, rewriter, LOOM_OP_INDEX_SHRSI, lhs,
                                       rhs);
}

iree_status_t loom_index_shrui_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  return loom_index_shift_canonicalize(op, rewriter, LOOM_OP_INDEX_SHRUI,
                                       loom_index_shrui_lhs(op),
                                       loom_index_shrui_rhs(op));
}

iree_status_t loom_index_rotli_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_index_rotli_lhs(op);
  if (loom_index_value_facts_are_exact_i64(rewriter, loom_index_rotli_rhs(op),
                                           0)) {
    return loom_index_replace_single_result_with_value(op, rewriter, lhs);
  }
  return iree_ok_status();
}

iree_status_t loom_index_rotri_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_index_rotri_lhs(op);
  loom_value_id_t rhs = loom_index_rotri_rhs(op);
  int64_t amount = 0;
  if (!loom_index_query_exact_i64(rewriter, rhs, &amount)) {
    return iree_ok_status();
  }
  if (amount == 0) {
    return loom_index_replace_single_result_with_value(op, rewriter, lhs);
  }

  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_index_rotri_result(op));
  if (!loom_index_shift_amount_is_valid(result_type, amount)) {
    return iree_ok_status();
  }
  int32_t bitwidth =
      loom_scalar_type_bitwidth(loom_type_element_type(result_type));
  int64_t left_amount = bitwidth - amount;

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t left_amount_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_index_materialize_or_reuse_index_constant(
      op, rewriter, LOOM_VALUE_ID_INVALID, left_amount, &left_amount_value));
  return loom_index_replace_single_result_with_binary_op(
      op, rewriter, LOOM_OP_INDEX_ROTLI, lhs, left_amount_value);
}

//===----------------------------------------------------------------------===//
// Casts, comparisons, and selection
//===----------------------------------------------------------------------===//

iree_status_t loom_index_cast_canonicalize(loom_op_t* op,
                                           loom_rewriter_t* rewriter) {
  loom_type_t input_type =
      loom_module_value_type(rewriter->module, loom_index_cast_input(op));
  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_index_cast_result(op));
  if (!loom_type_equal(input_type, result_type)) return iree_ok_status();
  return loom_index_replace_single_result_with_value(op, rewriter,
                                                     loom_index_cast_input(op));
}

iree_status_t loom_index_cmp_canonicalize(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_index_cmp_lhs(op);
  loom_value_id_t rhs = loom_index_cmp_rhs(op);
  uint8_t predicate = loom_index_cmp_predicate(op);
  bool result = false;
  loom_value_facts_t lhs_facts = loom_rewriter_value_facts(rewriter, lhs);
  loom_value_facts_t rhs_facts = loom_rewriter_value_facts(rewriter, rhs);
  if ((lhs == rhs && loom_index_cmp_same_value_result(predicate, &result)) ||
      loom_index_cmp_result_from_facts(predicate, &lhs_facts, &rhs_facts,
                                       &result)) {
    return loom_index_replace_single_result_with_i1_constant(op, rewriter,
                                                             result);
  }
  return iree_ok_status();
}
