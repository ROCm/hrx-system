// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scalar/wide_division_legalization.h"

#include <stdint.h>

#include "iree/base/internal/math.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/util/fact_table.h"
#include "loom/util/predicate_facts.h"

typedef enum loom_scalar_wide_division_result_e {
  LOOM_SCALAR_WIDE_DIVISION_QUOTIENT,
  LOOM_SCALAR_WIDE_DIVISION_REMAINDER,
  LOOM_SCALAR_WIDE_DIVISION_PAIR,
} loom_scalar_wide_division_result_t;

typedef struct loom_scalar_wide_division_match_t {
  // Original full-width dividend.
  loom_value_id_t dividend;
  // Original nonzero divisor.
  loom_value_id_t divisor;
  // Original i64 or index type shared by the operands and results.
  loom_type_t source_type;
  // Source operation's quotient, remainder, or fused result contract.
  loom_scalar_wide_division_result_t result;
  // Whether source operands use signed, truncation-toward-zero semantics.
  bool is_signed;
  // Positive word divisor bound, or zero for an unrestricted divisor.
  uint32_t divisor_maximum;
  // Largest safe remainder recurrence chunk derived from divisor_maximum.
  uint8_t chunk_width;
} loom_scalar_wide_division_match_t;

static bool loom_scalar_match_wide_division(
    const loom_target_legalization_context_t* context, const loom_op_t* op,
    loom_scalar_wide_division_match_t* out_match) {
  loom_scalar_wide_division_result_t result;
  switch (op->kind) {
    case LOOM_OP_SCALAR_DIVSI:
    case LOOM_OP_SCALAR_DIVUI:
    case LOOM_OP_INDEX_DIV:
      result = LOOM_SCALAR_WIDE_DIVISION_QUOTIENT;
      break;
    case LOOM_OP_SCALAR_REMSI:
    case LOOM_OP_SCALAR_REMUI:
    case LOOM_OP_INDEX_REM:
      result = LOOM_SCALAR_WIDE_DIVISION_REMAINDER;
      break;
    case LOOM_OP_SCALAR_DIVREMSI:
    case LOOM_OP_SCALAR_DIVREMUI:
    case LOOM_OP_INDEX_DIVREM:
      result = LOOM_SCALAR_WIDE_DIVISION_PAIR;
      break;
    default:
      return false;
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_type_t source_type =
      loom_module_value_type(context->module, operands[0]);
  const loom_scalar_type_t element_type = loom_type_element_type(source_type);
  if (!loom_type_is_scalar(source_type) ||
      (element_type != LOOM_SCALAR_TYPE_I64 &&
       element_type != LOOM_SCALAR_TYPE_INDEX)) {
    return false;
  }
  const loom_value_facts_t divisor_facts =
      context->fact_table
          ? loom_value_fact_table_lookup(context->fact_table, operands[1])
          : loom_value_facts_unknown();
  if (element_type == LOOM_SCALAR_TYPE_INDEX) {
    const loom_value_facts_t dividend_facts =
        context->fact_table
            ? loom_value_fact_table_lookup(context->fact_table, operands[0])
            : loom_value_facts_unknown();
    if (!loom_value_facts_is_non_negative(dividend_facts) ||
        !loom_value_facts_is_positive(divisor_facts)) {
      return false;
    }
  }
  int64_t exact_divisor = 0;
  if (loom_value_facts_as_exact_i64(divisor_facts, &exact_divisor)) {
    return false;
  }
  const bool is_signed = loom_scalar_divsi_isa(op) ||
                         loom_scalar_remsi_isa(op) ||
                         loom_scalar_divremsi_isa(op);
  const uint32_t divisor_maximum =
      !is_signed && loom_value_facts_is_positive(divisor_facts) &&
              loom_value_facts_fit_unsigned_bit_count(divisor_facts, 32)
          ? (uint32_t)divisor_facts.range_hi
          : 0;
  *out_match = (loom_scalar_wide_division_match_t){
      .dividend = operands[0],
      .divisor = operands[1],
      .source_type = source_type,
      .result = result,
      .is_signed = is_signed,
      .divisor_maximum = divisor_maximum,
      .chunk_width = divisor_maximum <= 1
                         ? 0
                         : (uint8_t)iree_math_count_leading_zeros_u32(
                               divisor_maximum - 1u),
  };
  return true;
}

static bool loom_scalar_match_wide_division_rule(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op) {
  (void)entry;
  loom_scalar_wide_division_match_t match;
  return loom_scalar_match_wide_division(context, op, &match);
}

static iree_status_t loom_scalar_wide_division_build_constant(
    loom_builder_t* builder, int64_t value, loom_scalar_type_t element_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      builder, loom_attr_i64(value), loom_type_scalar(element_type), location,
      &constant_op));
  *out_value = loom_scalar_constant_result(constant_op);
  return iree_ok_status();
}

static iree_status_t loom_scalar_wide_division_build_i32_shift(
    loom_builder_t* builder, bool left, loom_value_id_t value, uint32_t amount,
    loom_location_id_t location, loom_value_id_t* out_value) {
  loom_value_id_t amount_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_constant(
      builder, amount, LOOM_SCALAR_TYPE_I32, location, &amount_value));
  loom_op_t* shift_op = NULL;
  if (left) {
    IREE_RETURN_IF_ERROR(loom_scalar_shli_build(
        builder, /*instance_flags=*/0, value, amount_value,
        loom_type_scalar(LOOM_SCALAR_TYPE_I32), location, &shift_op));
    *out_value = loom_scalar_shli_result(shift_op);
  } else {
    IREE_RETURN_IF_ERROR(loom_scalar_shrui_build(
        builder, value, amount_value, loom_type_scalar(LOOM_SCALAR_TYPE_I32),
        location, &shift_op));
    *out_value = loom_scalar_shrui_result(shift_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_scalar_wide_division_build_i32_mask(
    loom_builder_t* builder, loom_value_id_t value, uint32_t mask,
    loom_location_id_t location, loom_value_id_t* out_value) {
  loom_value_id_t mask_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_constant(
      builder, mask, LOOM_SCALAR_TYPE_I32, location, &mask_value));
  loom_op_t* mask_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
      builder, value, mask_value, loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      location, &mask_op));
  *out_value = loom_scalar_andi_result(mask_op);
  return iree_ok_status();
}

static iree_status_t loom_scalar_wide_division_build_i32_remainder(
    loom_builder_t* builder, loom_value_id_t dividend, loom_value_id_t divisor,
    loom_location_id_t location, loom_value_id_t* out_value) {
  loom_op_t* remainder_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_remui_build(
      builder, dividend, divisor, loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      location, &remainder_op));
  *out_value = loom_scalar_remui_result(remainder_op);
  return iree_ok_status();
}

typedef uint8_t loom_scalar_wide_division_binary_kind_t;
enum loom_scalar_wide_division_binary_kind_e {
  LOOM_SCALAR_WIDE_DIVISION_BINARY_ADD = 0,
  LOOM_SCALAR_WIDE_DIVISION_BINARY_SUB = 1,
  LOOM_SCALAR_WIDE_DIVISION_BINARY_MUL = 2,
  LOOM_SCALAR_WIDE_DIVISION_BINARY_AND = 3,
  LOOM_SCALAR_WIDE_DIVISION_BINARY_OR = 4,
};

static iree_status_t loom_scalar_wide_division_build_binary(
    loom_builder_t* builder, loom_scalar_wide_division_binary_kind_t kind,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  loom_op_t* binary_op = NULL;
  switch (kind) {
    case LOOM_SCALAR_WIDE_DIVISION_BINARY_ADD: {
      IREE_RETURN_IF_ERROR(loom_scalar_addi_build(builder, /*instance_flags=*/0,
                                                  lhs, rhs, result_type,
                                                  location, &binary_op));
      *out_value = loom_scalar_addi_result(binary_op);
      return iree_ok_status();
    }
    case LOOM_SCALAR_WIDE_DIVISION_BINARY_SUB: {
      IREE_RETURN_IF_ERROR(loom_scalar_subi_build(builder, /*instance_flags=*/0,
                                                  lhs, rhs, result_type,
                                                  location, &binary_op));
      *out_value = loom_scalar_subi_result(binary_op);
      return iree_ok_status();
    }
    case LOOM_SCALAR_WIDE_DIVISION_BINARY_MUL: {
      IREE_RETURN_IF_ERROR(loom_scalar_muli_build(builder, /*instance_flags=*/0,
                                                  lhs, rhs, result_type,
                                                  location, &binary_op));
      *out_value = loom_scalar_muli_result(binary_op);
      return iree_ok_status();
    }
    case LOOM_SCALAR_WIDE_DIVISION_BINARY_AND: {
      IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
          builder, lhs, rhs, result_type, location, &binary_op));
      *out_value = loom_scalar_andi_result(binary_op);
      return iree_ok_status();
    }
    case LOOM_SCALAR_WIDE_DIVISION_BINARY_OR: {
      IREE_RETURN_IF_ERROR(loom_scalar_ori_build(builder, lhs, rhs, result_type,
                                                 location, &binary_op));
      *out_value = loom_scalar_ori_result(binary_op);
      return iree_ok_status();
    }
    default:
      IREE_ASSERT_UNREACHABLE("unknown wide division binary kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_scalar_wide_division_build_shift(
    loom_builder_t* builder, bool left, loom_value_id_t value,
    loom_value_id_t amount, loom_type_t type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  loom_op_t* shift_op = NULL;
  if (left) {
    IREE_RETURN_IF_ERROR(loom_scalar_shli_build(builder, /*instance_flags=*/0,
                                                value, amount, type, location,
                                                &shift_op));
    *out_value = loom_scalar_shli_result(shift_op);
  } else {
    IREE_RETURN_IF_ERROR(loom_scalar_shrui_build(builder, value, amount, type,
                                                 location, &shift_op));
    *out_value = loom_scalar_shrui_result(shift_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_scalar_wide_division_build_compare(
    loom_builder_t* builder, loom_scalar_cmpi_predicate_t predicate,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_condition) {
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(builder, predicate, lhs, rhs,
                                              location, &compare_op));
  *out_condition = loom_scalar_cmpi_result(compare_op);
  return iree_ok_status();
}

static iree_status_t loom_scalar_wide_division_build_select(
    loom_builder_t* builder, loom_value_id_t condition,
    loom_value_id_t true_value, loom_value_id_t false_value, loom_type_t type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_select_build(
      builder, condition, true_value, false_value, type, location, &select_op));
  *out_value = loom_scf_select_result(select_op);
  return iree_ok_status();
}

static iree_status_t loom_scalar_wide_division_build_divrem(
    loom_builder_t* builder, loom_value_id_t dividend, loom_value_id_t divisor,
    loom_location_id_t location, loom_value_id_t* out_quotient,
    loom_value_id_t* out_remainder) {
  const loom_type_t result_types[2] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      loom_type_scalar(LOOM_SCALAR_TYPE_I32),
  };
  loom_op_t* division_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_divremui_build(
      builder, dividend, divisor, result_types, location, &division_op));
  *out_quotient = loom_scalar_divremui_quotient(division_op);
  *out_remainder = loom_scalar_divremui_remainder(division_op);
  return iree_ok_status();
}

static iree_status_t loom_scalar_wide_division_cast_input_to_i64(
    loom_builder_t* builder, loom_value_id_t input, loom_type_t input_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  if (loom_type_element_type(input_type) == LOOM_SCALAR_TYPE_I64) {
    *out_value = input;
    return iree_ok_status();
  }
  loom_op_t* cast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(
      builder, input, input_type, loom_type_scalar(LOOM_SCALAR_TYPE_I64),
      location, &cast_op));
  *out_value = loom_index_cast_result(cast_op);
  return iree_ok_status();
}

static iree_status_t loom_scalar_wide_division_build_assume_range(
    loom_builder_t* builder, loom_value_id_t value, int64_t minimum,
    int64_t maximum, loom_type_t type, loom_location_id_t location,
    loom_value_id_t* out_value);

static iree_status_t loom_scalar_wide_division_extract_word(
    loom_builder_t* builder, loom_value_id_t value, uint32_t offset,
    loom_location_id_t location, loom_value_id_t* out_value) {
  loom_op_t* extract_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_bitfield_extractu_build(
      builder, value, offset, /*width=*/32,
      loom_type_scalar(LOOM_SCALAR_TYPE_I32), location, &extract_op));
  *out_value = loom_scalar_bitfield_extractu_result(extract_op);
  return iree_ok_status();
}

static iree_status_t loom_scalar_wide_division_extend_word(
    loom_builder_t* builder, loom_value_id_t value, loom_location_id_t location,
    loom_value_id_t* out_value) {
  loom_op_t* extend_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_extui_build(
      builder, value, loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      loom_type_scalar(LOOM_SCALAR_TYPE_I64), location, &extend_op));
  *out_value = loom_scalar_extui_result(extend_op);
  return iree_ok_status();
}

static iree_status_t loom_scalar_wide_division_join_words(
    loom_builder_t* builder, loom_value_id_t low, loom_value_id_t high,
    loom_location_id_t location, loom_value_id_t* out_value) {
  const loom_type_t i64_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  IREE_RETURN_IF_ERROR(
      loom_scalar_wide_division_extend_word(builder, low, location, &low));
  IREE_RETURN_IF_ERROR(
      loom_scalar_wide_division_extend_word(builder, high, location, &high));
  loom_value_id_t shift = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_constant(
      builder, 32, LOOM_SCALAR_TYPE_I64, location, &shift));
  loom_value_id_t upper = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_shift(
      builder, /*left=*/true, high, shift, i64_type, location, &upper));
  return loom_scalar_wide_division_build_binary(
      builder, LOOM_SCALAR_WIDE_DIVISION_BINARY_OR, upper, low, i64_type,
      location, out_value);
}

static iree_status_t loom_scalar_wide_division_split_inputs(
    loom_builder_t* builder, const loom_scalar_wide_division_match_t* match,
    loom_location_id_t location, loom_value_id_t* out_low,
    loom_value_id_t* out_high, loom_value_id_t* out_divisor) {
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_type_t i64_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  loom_value_id_t dividend = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_cast_input_to_i64(
      builder, match->dividend, match->source_type, location, &dividend));
  loom_value_id_t divisor = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_cast_input_to_i64(
      builder, match->divisor, match->source_type, location, &divisor));

  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_extract_word(
      builder, dividend, 0, location, out_low));
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_extract_word(
      builder, dividend, 32, location, out_high));

  loom_op_t* divisor_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_trunci_build(
      builder, divisor, i64_type, i32_type, location, &divisor_op));
  *out_divisor = loom_scalar_trunci_result(divisor_op);
  if (match->divisor_maximum <= INT32_MAX) {
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_assume_range(
        builder, *out_divisor, 1, match->divisor_maximum, i32_type, location,
        out_divisor));
  } else {
    // Truncation preserves nonzero for a word divisor, but the complete
    // unsigned word domain cannot be expressed as a positive i32 range.
    const loom_predicate_t nonzero = {
        .kind = LOOM_PREDICATE_NE,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                     LOOM_PRED_ARG_NONE},
        .args = {*out_divisor, 0, 0},
    };
    loom_op_t* assume_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_assume_build(builder, out_divisor, 1,
                                                  &nonzero, 1, &i32_type, 1,
                                                  location, &assume_op));
    *out_divisor = loom_scalar_assume_results(assume_op).values[0];
  }
  return iree_ok_status();
}

static iree_status_t loom_scalar_wide_division_build_bounded(
    loom_builder_t* builder, const loom_scalar_wide_division_match_t* match,
    loom_location_id_t location, loom_value_id_t* out_remainder) {
  // Reduce the high word first, then consume the low word from most to least
  // significant chunk. The fact-derived width guarantees
  //   remainder * 2^width + chunk <= UINT32_MAX
  // for every step, so each recurrence is an exact i32 remainder on targets
  // without a native wide operation.
  loom_value_id_t low = LOOM_VALUE_ID_INVALID;
  loom_value_id_t high = LOOM_VALUE_ID_INVALID;
  loom_value_id_t divisor = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_split_inputs(
      builder, match, location, &low, &high, &divisor));

  loom_value_id_t remainder = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_i32_remainder(
      builder, high, divisor, location, &remainder));
  uint32_t remaining_width = 32;
  while (remaining_width != 0) {
    const bool is_first_chunk = remaining_width == 32;
    const uint32_t width = iree_min(remaining_width, match->chunk_width);
    remaining_width -= width;

    loom_value_id_t chunk = low;
    if (remaining_width != 0) {
      IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_i32_shift(
          builder, /*left=*/false, chunk, remaining_width, location, &chunk));
    }
    if (!is_first_chunk) {
      const uint32_t mask = (UINT32_C(1) << width) - 1u;
      IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_i32_mask(
          builder, chunk, mask, location, &chunk));
    }

    loom_value_id_t shifted_remainder = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_i32_shift(
        builder, /*left=*/true, remainder, width, location,
        &shifted_remainder));
    loom_op_t* combined_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_addi_build(
        builder, /*instance_flags=*/0, shifted_remainder, chunk,
        loom_type_scalar(LOOM_SCALAR_TYPE_I32), location, &combined_op));
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_i32_remainder(
        builder, loom_scalar_addi_result(combined_op), divisor, location,
        &remainder));
  }
  *out_remainder = remainder;
  return iree_ok_status();
}

static iree_status_t loom_scalar_wide_division_build_halfwords(
    loom_builder_t* builder, loom_value_id_t value,
    loom_value_id_t halfword_mask, loom_value_id_t halfword_shift,
    loom_location_id_t location, loom_value_id_t* out_low,
    loom_value_id_t* out_high) {
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_binary(
      builder, LOOM_SCALAR_WIDE_DIVISION_BINARY_AND, value, halfword_mask,
      i32_type, location, out_low));
  return loom_scalar_wide_division_build_shift(builder, /*left=*/false, value,
                                               halfword_shift, i32_type,
                                               location, out_high);
}

static iree_status_t loom_scalar_wide_division_build_assume_range(
    loom_builder_t* builder, loom_value_id_t value, int64_t minimum,
    int64_t maximum, loom_type_t type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  const loom_predicate_t range = {
      .kind = LOOM_PREDICATE_RANGE,
      .arg_count = 3,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                   LOOM_PRED_ARG_CONST},
      .args = {value, minimum, maximum},
  };
  loom_op_t* assume_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_assume_build(
      builder, &value, 1, &range, 1, &type, 1, location, &assume_op));
  *out_value = loom_scalar_assume_results(assume_op).values[0];
  return iree_ok_status();
}

static iree_status_t loom_scalar_wide_division_correct_digit(
    loom_builder_t* builder, loom_value_id_t divisor_high,
    loom_value_id_t divisor_low, loom_value_id_t next_digit,
    loom_value_id_t halfword_base, loom_value_id_t one,
    loom_value_id_t halfword_shift, loom_location_id_t location,
    loom_value_id_t* inout_quotient, loom_value_id_t* inout_remainder) {
  // Correct a base-2^16 quotient estimate. Normalization permits at most two
  // decrements in total. The explicit quotient >= base case handles estimate
  // overflow; two guarded passes implement the low-divisor-half test. Once the
  // partial remainder reaches the base, the estimate is known safe and the
  // remaining correction selects are inert.
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t decremented = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_binary(
      builder, LOOM_SCALAR_WIDE_DIVISION_BINARY_SUB, *inout_quotient, one,
      i32_type, location, &decremented));
  loom_value_id_t increased_remainder = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_binary(
      builder, LOOM_SCALAR_WIDE_DIVISION_BINARY_ADD, *inout_remainder,
      divisor_high, i32_type, location, &increased_remainder));
  loom_value_id_t estimate_overflow = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_compare(
      builder, LOOM_SCALAR_CMPI_PREDICATE_UGE, *inout_quotient, halfword_base,
      location, &estimate_overflow));
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_select(
      builder, estimate_overflow, decremented, *inout_quotient, i32_type,
      location, inout_quotient));
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_select(
      builder, estimate_overflow, increased_remainder, *inout_remainder,
      i32_type, location, inout_remainder));

  for (uint32_t i = 0; i < 2; ++i) {
    loom_value_id_t product = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_binary(
        builder, LOOM_SCALAR_WIDE_DIVISION_BINARY_MUL, *inout_quotient,
        divisor_low, i32_type, location, &product));
    loom_value_id_t shifted_remainder = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_shift(
        builder, /*left=*/true, *inout_remainder, halfword_shift, i32_type,
        location, &shifted_remainder));
    loom_value_id_t comparison_limit = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_binary(
        builder, LOOM_SCALAR_WIDE_DIVISION_BINARY_ADD, shifted_remainder,
        next_digit, i32_type, location, &comparison_limit));
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_binary(
        builder, LOOM_SCALAR_WIDE_DIVISION_BINARY_SUB, *inout_quotient, one,
        i32_type, location, &decremented));
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_binary(
        builder, LOOM_SCALAR_WIDE_DIVISION_BINARY_ADD, *inout_remainder,
        divisor_high, i32_type, location, &increased_remainder));
    loom_value_id_t product_too_large = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_compare(
        builder, LOOM_SCALAR_CMPI_PREDICATE_UGT, product, comparison_limit,
        location, &product_too_large));
    loom_value_id_t corrected_quotient = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_select(
        builder, product_too_large, decremented, *inout_quotient, i32_type,
        location, &corrected_quotient));
    loom_value_id_t corrected_remainder = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_select(
        builder, product_too_large, increased_remainder, *inout_remainder,
        i32_type, location, &corrected_remainder));
    loom_value_id_t correction_active = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_compare(
        builder, LOOM_SCALAR_CMPI_PREDICATE_ULT, *inout_remainder,
        halfword_base, location, &correction_active));
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_select(
        builder, correction_active, corrected_quotient, *inout_quotient,
        i32_type, location, inout_quotient));
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_select(
        builder, correction_active, corrected_remainder, *inout_remainder,
        i32_type, location, inout_remainder));
  }
  return iree_ok_status();
}

static iree_status_t loom_scalar_wide_division_estimate_digit(
    loom_builder_t* builder, loom_value_id_t dividend,
    loom_value_id_t divisor_high, loom_value_id_t divisor_low,
    loom_value_id_t next_digit, loom_value_id_t halfword_base,
    loom_value_id_t one, loom_value_id_t halfword_shift,
    loom_location_id_t location, loom_value_id_t* out_quotient) {
  loom_value_id_t quotient = LOOM_VALUE_ID_INVALID;
  loom_value_id_t remainder = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_divrem(
      builder, dividend, divisor_high, location, &quotient, &remainder));
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_correct_digit(
      builder, divisor_high, divisor_low, next_digit, halfword_base, one,
      halfword_shift, location, &quotient, &remainder));
  *out_quotient = quotient;
  return iree_ok_status();
}

static iree_status_t loom_scalar_wide_division_build_normalized_pair(
    loom_builder_t* builder, loom_value_id_t low, loom_value_id_t reduced_high,
    loom_value_id_t divisor, loom_location_id_t location,
    loom_value_id_t* out_normalization_shift,
    loom_value_id_t* out_normalized_divisor,
    loom_value_id_t* out_normalized_low, loom_value_id_t* out_normalized_high) {
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_type_t i64_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  loom_op_t* count_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_ctlzi_build(builder, divisor, i32_type, location, &count_op));
  *out_normalization_shift = loom_scalar_ctlzi_result(count_op);
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_shift(
      builder, /*left=*/true, divisor, *out_normalization_shift, i32_type,
      location, out_normalized_divisor));

  loom_value_id_t reduced_dividend = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_join_words(
      builder, low, reduced_high, location, &reduced_dividend));
  loom_value_id_t wide_shift = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_extend_word(
      builder, *out_normalization_shift, location, &wide_shift));
  loom_value_id_t normalized_dividend = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_shift(
      builder, /*left=*/true, reduced_dividend, wide_shift, i64_type, location,
      &normalized_dividend));
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_extract_word(
      builder, normalized_dividend, 0, location, out_normalized_low));
  return loom_scalar_wide_division_extract_word(
      builder, normalized_dividend, 32, location, out_normalized_high);
}

static iree_status_t loom_scalar_wide_division_build_word_divrem(
    loom_builder_t* builder, loom_value_id_t low, loom_value_id_t high,
    loom_value_id_t divisor, loom_location_id_t location,
    loom_value_id_t* out_quotient, loom_value_id_t* out_remainder) {
  // Reduce the high word first, making the remaining quotient fit one word.
  // Its two base-2^16 digits use exact u32 division and fixed corrections.
  // A remainder-only caller does not materialize the quotient's word assembly.
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t quotient_upper = LOOM_VALUE_ID_INVALID;
  loom_value_id_t reduced_high = LOOM_VALUE_ID_INVALID;
  if (out_quotient) {
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_divrem(
        builder, high, divisor, location, &quotient_upper, &reduced_high));
  } else {
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_i32_remainder(
        builder, high, divisor, location, &reduced_high));
  }

  loom_value_id_t normalization_shift = LOOM_VALUE_ID_INVALID;
  loom_value_id_t normalized_divisor = LOOM_VALUE_ID_INVALID;
  loom_value_id_t normalized_low = LOOM_VALUE_ID_INVALID;
  loom_value_id_t normalized_high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_normalized_pair(
      builder, low, reduced_high, divisor, location, &normalization_shift,
      &normalized_divisor, &normalized_low, &normalized_high));

  loom_value_id_t halfword_mask = LOOM_VALUE_ID_INVALID;
  loom_value_id_t halfword_base = LOOM_VALUE_ID_INVALID;
  loom_value_id_t halfword_shift = LOOM_VALUE_ID_INVALID;
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_constant(
      builder, UINT16_MAX, LOOM_SCALAR_TYPE_I32, location, &halfword_mask));
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_constant(
      builder, UINT32_C(1) << 16, LOOM_SCALAR_TYPE_I32, location,
      &halfword_base));
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_constant(
      builder, 16, LOOM_SCALAR_TYPE_I32, location, &halfword_shift));
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_constant(
      builder, 1, LOOM_SCALAR_TYPE_I32, location, &one));

  loom_value_id_t divisor_low = LOOM_VALUE_ID_INVALID;
  loom_value_id_t divisor_high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_halfwords(
      builder, normalized_divisor, halfword_mask, halfword_shift, location,
      &divisor_low, &divisor_high));
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_assume_range(
      builder, divisor_high, UINT16_C(1) << 15, UINT16_MAX, i32_type, location,
      &divisor_high));
  loom_value_id_t low_low = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_halfwords(
      builder, normalized_low, halfword_mask, halfword_shift, location,
      &low_low, &low_high));

  loom_value_id_t quotient_high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_estimate_digit(
      builder, normalized_high, divisor_high, divisor_low, low_high,
      halfword_base, one, halfword_shift, location, &quotient_high));
  loom_value_id_t shifted_high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_shift(
      builder, /*left=*/true, normalized_high, halfword_shift, i32_type,
      location, &shifted_high));
  loom_value_id_t middle_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_binary(
      builder, LOOM_SCALAR_WIDE_DIVISION_BINARY_ADD, shifted_high, low_high,
      i32_type, location, &middle_source));
  loom_value_id_t high_product = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_binary(
      builder, LOOM_SCALAR_WIDE_DIVISION_BINARY_MUL, quotient_high,
      normalized_divisor, i32_type, location, &high_product));
  loom_value_id_t middle = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_binary(
      builder, LOOM_SCALAR_WIDE_DIVISION_BINARY_SUB, middle_source,
      high_product, i32_type, location, &middle));

  loom_value_id_t quotient_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_estimate_digit(
      builder, middle, divisor_high, divisor_low, low_low, halfword_base, one,
      halfword_shift, location, &quotient_low));
  loom_value_id_t shifted_middle = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_shift(
      builder, /*left=*/true, middle, halfword_shift, i32_type, location,
      &shifted_middle));
  loom_value_id_t remainder_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_binary(
      builder, LOOM_SCALAR_WIDE_DIVISION_BINARY_ADD, shifted_middle, low_low,
      i32_type, location, &remainder_source));
  loom_value_id_t low_product = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_binary(
      builder, LOOM_SCALAR_WIDE_DIVISION_BINARY_MUL, quotient_low,
      normalized_divisor, i32_type, location, &low_product));
  loom_value_id_t normalized_remainder = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_binary(
      builder, LOOM_SCALAR_WIDE_DIVISION_BINARY_SUB, remainder_source,
      low_product, i32_type, location, &normalized_remainder));
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_shift(
      builder, /*left=*/false, normalized_remainder, normalization_shift,
      i32_type, location, out_remainder));
  if (out_quotient) {
    loom_value_id_t shifted_quotient = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_i32_shift(
        builder, /*left=*/true, quotient_high, 16, location,
        &shifted_quotient));
    loom_value_id_t quotient_lower = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_binary(
        builder, LOOM_SCALAR_WIDE_DIVISION_BINARY_OR, shifted_quotient,
        quotient_low, i32_type, location, &quotient_lower));
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_join_words(
        builder, quotient_lower, quotient_upper, location, out_quotient));
  }
  return iree_ok_status();
}

static iree_status_t loom_scalar_wide_division_build_general(
    loom_builder_t* builder, loom_value_id_t dividend, loom_value_id_t divisor,
    loom_location_id_t location, loom_value_id_t* out_quotient,
    loom_value_id_t* out_remainder) {
  // Let s = clz(high(divisor)), including s=32 for a one-word divisor.
  // Dividing dividend >> (32-s) by high(divisor << s) gives an estimate
  // that is exact for word divisors and at most one high otherwise. Both
  // domains share the same normalized two-by-one word division, with no
  // runtime branch or bit-at-a-time loop.
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_type_t i64_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  loom_value_id_t high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_extract_word(
      builder, divisor, 32, location, &high));
  loom_op_t* count_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_ctlzi_build(builder, high, i32_type, location, &count_op));
  loom_value_id_t shift = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_extend_word(
      builder, loom_scalar_ctlzi_result(count_op), location, &shift));
  loom_value_id_t normalized_divisor = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_shift(
      builder, /*left=*/true, divisor, shift, i64_type, location,
      &normalized_divisor));
  loom_value_id_t divisor_word = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_extract_word(
      builder, normalized_divisor, 32, location, &divisor_word));
  loom_value_id_t divisor_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_extract_word(
      builder, normalized_divisor, 0, location, &divisor_low));

  loom_value_id_t word_width = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_constant(
      builder, 32, LOOM_SCALAR_TYPE_I64, location, &word_width));
  loom_value_id_t right_shift = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_binary(
      builder, LOOM_SCALAR_WIDE_DIVISION_BINARY_SUB, word_width, shift,
      i64_type, location, &right_shift));
  loom_value_id_t upper = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_shift(
      builder, /*left=*/false, dividend, right_shift, i64_type, location,
      &upper));
  // Preserve the bounded division as source IR so targets with a native
  // two-word divider can select it. The portable fallback handles this
  // strictly narrower divisor domain directly, without renormalizing it.
  loom_value_id_t extended_divisor = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_extend_word(
      builder, divisor_word, location, &extended_divisor));
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_assume_range(
      builder, extended_divisor, 1, UINT32_MAX, i64_type, location,
      &extended_divisor));
  const loom_type_t division_types[2] = {i64_type, i64_type};
  loom_op_t* division_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_divremui_build(builder, upper, extended_divisor,
                                 division_types, location, &division_op));
  loom_value_id_t quotient = loom_scalar_divremui_quotient(division_op);
  loom_value_id_t remainder = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_extract_word(
      builder, loom_scalar_divremui_remainder(division_op), 0, location,
      &remainder));

  // For a full-width divisor, write divisor = K*v+r with K=2^(32-s),
  // v>=2^31 and r<K. The real quotient estimate exceeds the exact ratio by
  // less than 4*(K-1)/K^2 <= 1. The low-word product test therefore needs
  // only one decrement. For a word divisor its low normalized word is zero,
  // making the same test false even when the quotient occupies two words.
  loom_value_id_t quotient_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_extract_word(
      builder, quotient, 0, location, &quotient_low));
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_extend_word(
      builder, quotient_low, location, &quotient_low));
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_extend_word(
      builder, divisor_low, location, &divisor_low));
  loom_value_id_t low_product = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_binary(
      builder, LOOM_SCALAR_WIDE_DIVISION_BINARY_MUL, quotient_low, divisor_low,
      i64_type, location, &low_product));
  loom_value_id_t shifted_dividend = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_shift(
      builder, /*left=*/true, dividend, shift, i64_type, location,
      &shifted_dividend));
  loom_value_id_t next_word = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_extract_word(
      builder, shifted_dividend, 0, location, &next_word));
  loom_value_id_t limit = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_join_words(
      builder, next_word, remainder, location, &limit));
  loom_value_id_t overestimate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_compare(
      builder, LOOM_SCALAR_CMPI_PREDICATE_UGT, low_product, limit, location,
      &overestimate));
  loom_op_t* correction_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_extui_build(
      builder, overestimate, loom_type_scalar(LOOM_SCALAR_TYPE_I1), i64_type,
      location, &correction_op));
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_binary(
      builder, LOOM_SCALAR_WIDE_DIVISION_BINARY_SUB, quotient,
      loom_scalar_extui_result(correction_op), i64_type, location,
      out_quotient));
  if (out_remainder) {
    loom_value_id_t product = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_binary(
        builder, LOOM_SCALAR_WIDE_DIVISION_BINARY_MUL, *out_quotient, divisor,
        i64_type, location, &product));
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_binary(
        builder, LOOM_SCALAR_WIDE_DIVISION_BINARY_SUB, dividend, product,
        i64_type, location, out_remainder));
  }
  return iree_ok_status();
}

static iree_status_t loom_scalar_wide_division_apply_sign(
    loom_builder_t* builder, loom_value_id_t value, loom_value_id_t sign,
    loom_location_id_t location, loom_value_id_t* out_value) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  loom_op_t* xor_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_xori_build(builder, value, sign, type, location, &xor_op));
  return loom_scalar_wide_division_build_binary(
      builder, LOOM_SCALAR_WIDE_DIVISION_BINARY_SUB,
      loom_scalar_xori_result(xor_op), sign, type, location, out_value);
}

static iree_status_t loom_scalar_wide_division_magnitude(
    loom_builder_t* builder, loom_value_id_t value, loom_location_id_t location,
    loom_value_id_t* out_sign, loom_value_id_t* out_magnitude) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  loom_value_id_t shift = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_constant(
      builder, 63, LOOM_SCALAR_TYPE_I64, location, &shift));
  loom_op_t* sign_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_shrsi_build(builder, value, shift, type, location, &sign_op));
  *out_sign = loom_scalar_shrsi_result(sign_op);
  return loom_scalar_wide_division_apply_sign(builder, value, *out_sign,
                                              location, out_magnitude);
}

static iree_status_t loom_scalar_wide_division_build_results(
    loom_builder_t* builder, const loom_scalar_wide_division_match_t* match,
    loom_location_id_t location, loom_value_id_t* out_results) {
  const bool has_quotient =
      match->result != LOOM_SCALAR_WIDE_DIVISION_REMAINDER;
  const bool has_remainder =
      match->result != LOOM_SCALAR_WIDE_DIVISION_QUOTIENT;
  loom_value_id_t dividend = LOOM_VALUE_ID_INVALID;
  loom_value_id_t divisor = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_cast_input_to_i64(
      builder, match->dividend, match->source_type, location, &dividend));
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_cast_input_to_i64(
      builder, match->divisor, match->source_type, location, &divisor));
  loom_value_id_t dividend_sign = LOOM_VALUE_ID_INVALID;
  loom_value_id_t divisor_sign = LOOM_VALUE_ID_INVALID;
  if (match->is_signed) {
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_magnitude(
        builder, dividend, location, &dividend_sign, &dividend));
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_magnitude(
        builder, divisor, location, &divisor_sign, &divisor));
  }
  loom_value_id_t quotient = LOOM_VALUE_ID_INVALID;
  loom_value_id_t remainder = LOOM_VALUE_ID_INVALID;
  if (match->divisor_maximum) {
    loom_value_id_t low = LOOM_VALUE_ID_INVALID;
    loom_value_id_t high = LOOM_VALUE_ID_INVALID;
    loom_value_id_t divisor_word = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_split_inputs(
        builder, match, location, &low, &high, &divisor_word));
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_word_divrem(
        builder, low, high, divisor_word, location,
        has_quotient ? &quotient : NULL, &remainder));
    if (has_remainder) {
      IREE_RETURN_IF_ERROR(loom_scalar_wide_division_extend_word(
          builder, remainder, location, &remainder));
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_general(
        builder, dividend, divisor, location, &quotient,
        has_remainder ? &remainder : NULL));
  }
  if (match->is_signed) {
    if (has_quotient) {
      loom_op_t* quotient_sign_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_xori_build(
          builder, dividend_sign, divisor_sign,
          loom_type_scalar(LOOM_SCALAR_TYPE_I64), location, &quotient_sign_op));
      IREE_RETURN_IF_ERROR(loom_scalar_wide_division_apply_sign(
          builder, quotient, loom_scalar_xori_result(quotient_sign_op),
          location, &quotient));
    }
    if (has_remainder) {
      IREE_RETURN_IF_ERROR(loom_scalar_wide_division_apply_sign(
          builder, remainder, dividend_sign, location, &remainder));
    }
  }
  uint32_t result_count = 0;
  if (has_quotient) {
    out_results[result_count++] = quotient;
  }
  if (has_remainder) {
    out_results[result_count++] = remainder;
  }
  return iree_ok_status();
}

static iree_status_t loom_scalar_wide_division_publish_result(
    loom_builder_t* builder, const loom_scalar_wide_division_match_t* match,
    loom_value_id_t remainder, loom_location_id_t location,
    loom_value_id_t* out_result) {
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_type_t i64_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  loom_op_t* extension_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_extui_build(
      builder, remainder, i32_type, i64_type, location, &extension_op));
  loom_value_id_t extended = loom_scalar_extui_result(extension_op);
  IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_assume_range(
      builder, extended, 0, (int64_t)match->divisor_maximum - 1, i64_type,
      location, &extended));
  if (loom_type_element_type(match->source_type) == LOOM_SCALAR_TYPE_I64) {
    *out_result = extended;
    return iree_ok_status();
  }
  loom_op_t* cast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(
      builder, extended, i64_type, match->source_type, location, &cast_op));
  *out_result = loom_index_cast_result(cast_op);
  return iree_ok_status();
}

static iree_status_t loom_scalar_legalize_wide_division(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  loom_scalar_wide_division_match_t match;
  if (!loom_scalar_match_wide_division(context, op, &match)) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t replacements[2] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  const uint32_t result_count =
      match.result == LOOM_SCALAR_WIDE_DIVISION_PAIR ? 2 : 1;
  if (match.result == LOOM_SCALAR_WIDE_DIVISION_REMAINDER &&
      match.divisor_maximum) {
    loom_value_id_t remainder = LOOM_VALUE_ID_INVALID;
    if (match.chunk_width >= 4) {
      IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_bounded(
          &rewriter->builder, &match, op->location, &remainder));
    } else {
      loom_value_id_t low = LOOM_VALUE_ID_INVALID;
      loom_value_id_t high = LOOM_VALUE_ID_INVALID;
      loom_value_id_t divisor = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_scalar_wide_division_split_inputs(
          &rewriter->builder, &match, op->location, &low, &high, &divisor));
      IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_word_divrem(
          &rewriter->builder, low, high, divisor, op->location,
          /*out_quotient=*/NULL, &remainder));
    }
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_publish_result(
        &rewriter->builder, &match, remainder, op->location, &replacements[0]));
  } else {
    IREE_RETURN_IF_ERROR(loom_scalar_wide_division_build_results(
        &rewriter->builder, &match, op->location, replacements));
    const loom_type_t i64_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
    for (uint32_t i = 0; i < result_count; ++i) {
      // Retain the original operation's facts instead of requiring later
      // stages to rediscover bounds through the arithmetic decomposition.
      if (context->fact_table) {
        loom_predicate_t
            predicates[LOOM_VALUE_FACT_INTEGER_REFINEMENT_CAPACITY];
        const uint16_t count = loom_value_facts_export_integer_refinements(
            replacements[i], loom_value_facts_unknown(),
            loom_value_fact_table_lookup(context->fact_table,
                                         loom_op_const_results(op)[i]),
            predicates, IREE_ARRAYSIZE(predicates));
        if (count) {
          loom_op_t* assume_op = NULL;
          IREE_RETURN_IF_ERROR(loom_scalar_assume_build(
              &rewriter->builder, &replacements[i], 1, predicates, count,
              &i64_type, 1, op->location, &assume_op));
          replacements[i] = loom_scalar_assume_results(assume_op).values[0];
        }
      }
      if (loom_type_element_type(match.source_type) == LOOM_SCALAR_TYPE_INDEX) {
        loom_op_t* cast_op = NULL;
        IREE_RETURN_IF_ERROR(
            loom_index_cast_build(&rewriter->builder, replacements[i], i64_type,
                                  match.source_type, op->location, &cast_op));
        replacements[i] = loom_index_cast_result(cast_op);
      }
    }
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, result_count, checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacements, result_count));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

// An unhandled fused operation may decompose to a native quotient. Its
// generic decomposition retains precedence unless the target explicitly owns
// and rejects the fused contract. Missing scalar contracts use this fallback.
static const loom_target_legalizer_rule_t kWideDivisionLegalizerRules[] = {
    {
        .root_kind = LOOM_OP_SCALAR_DIVSI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I64,
        .match = loom_scalar_match_wide_division_rule,
        .legalize = loom_scalar_legalize_wide_division,
    },
    {
        .root_kind = LOOM_OP_SCALAR_DIVUI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I64,
        .match = loom_scalar_match_wide_division_rule,
        .legalize = loom_scalar_legalize_wide_division,
    },
    {
        .root_kind = LOOM_OP_SCALAR_REMSI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I64,
        .match = loom_scalar_match_wide_division_rule,
        .legalize = loom_scalar_legalize_wide_division,
    },
    {
        .root_kind = LOOM_OP_SCALAR_REMUI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I64,
        .match = loom_scalar_match_wide_division_rule,
        .legalize = loom_scalar_legalize_wide_division,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .root_kind = LOOM_OP_SCALAR_DIVREMSI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I64,
        .match = loom_scalar_match_wide_division_rule,
        .legalize = loom_scalar_legalize_wide_division,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .root_kind = LOOM_OP_SCALAR_DIVREMUI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I64,
        .match = loom_scalar_match_wide_division_rule,
        .legalize = loom_scalar_legalize_wide_division,
    },
    {
        .root_kind = LOOM_OP_INDEX_DIV,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_INDEX,
        .match = loom_scalar_match_wide_division_rule,
        .legalize = loom_scalar_legalize_wide_division,
    },
    {
        .root_kind = LOOM_OP_INDEX_REM,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_INDEX,
        .match = loom_scalar_match_wide_division_rule,
        .legalize = loom_scalar_legalize_wide_division,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .root_kind = LOOM_OP_INDEX_DIVREM,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_INDEX,
        .match = loom_scalar_match_wide_division_rule,
        .legalize = loom_scalar_legalize_wide_division,
    },
};

const loom_target_legalizer_provider_t
    loom_scalar_wide_division_legalizer_provider_storage = {
        .name = IREE_SVL("wide-integer-division"),
        .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE,
        .rules = kWideDivisionLegalizerRules,
        .rule_count = IREE_ARRAYSIZE(kWideDivisionLegalizerRules),
};
