// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scalar/division_refinement.h"

#include "loom/ops/scalar/ops.h"
#include "loom/util/predicate_facts.h"

iree_status_t loom_scalar_division_apply_sign(
    loom_builder_t* builder, loom_value_id_t value, loom_value_id_t sign,
    loom_type_t type, loom_location_id_t location, loom_value_id_t* out_value) {
  loom_op_t* xor_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_xori_build(builder, value, sign, type, location, &xor_op));
  loom_op_t* subtraction = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_subi_build(
      builder, /*instance_flags=*/0, loom_scalar_xori_result(xor_op), sign,
      type, location, &subtraction));
  *out_value = loom_scalar_subi_result(subtraction);
  return iree_ok_status();
}

iree_status_t loom_scalar_division_build_magnitude(
    loom_builder_t* builder, loom_value_id_t value, loom_type_t type,
    loom_location_id_t location, loom_value_id_t* out_sign,
    loom_value_id_t* out_magnitude) {
  const int32_t bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(type));
  loom_op_t* shift = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      builder, loom_attr_i64(bit_count - 1), type, location, &shift));
  loom_op_t* sign = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_shrsi_build(
      builder, value, loom_scalar_constant_result(shift), type, location,
      &sign));
  *out_sign = loom_scalar_shrsi_result(sign);
  return loom_scalar_division_apply_sign(builder, value, *out_sign, type,
                                         location, out_magnitude);
}

iree_status_t loom_scalar_division_refine_unsigned_results(
    loom_builder_t* builder, loom_value_id_t dividend, loom_value_id_t divisor,
    loom_type_t type, bool signed_magnitudes, loom_location_id_t location,
    loom_value_id_t* quotient, loom_value_id_t* remainder) {
  const loom_type_t types[2] = {type, type};
  loom_value_id_t values[2];
  loom_predicate_t predicates[3];
  uint16_t count = 0;
  if (quotient) {
    values[count] = *quotient;
    predicates[count++] = (loom_predicate_t){
        .kind = LOOM_PREDICATE_ULE,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
        .args = {*quotient, dividend},
    };
  }
  if (remainder) {
    values[count] = *remainder;
    predicates[count++] = (loom_predicate_t){
        .kind = LOOM_PREDICATE_ULT,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
        .args = {*remainder, divisor},
    };
  }
  uint16_t predicate_count = count;
  if (remainder && signed_magnitudes) {
    // The remainder is strictly smaller than the divisor's magnitude, which
    // is at most the carrier's high bit, including the minimum signed value.
    predicates[predicate_count++] = (loom_predicate_t){
        .kind = LOOM_PREDICATE_GE,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
        .args = {*remainder, 0},
    };
  }
  loom_op_t* assumption = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_assume_build(
      builder, values, count, predicates, predicate_count, types, count,
      location, &assumption));
  const loom_value_id_t* results = loom_op_const_results(assumption);
  uint16_t ordinal = 0;
  if (quotient) {
    *quotient = results[ordinal++];
  }
  if (remainder) {
    *remainder = results[ordinal];
  }
  return iree_ok_status();
}

// Defined division has a nonzero divisor. Its magnitude minus one therefore
// fits the signed carrier, including when the divisor is the signed minimum.
// These predicates keep that bound useful after a later scope proves that the
// divisor's magnitude is its original value.
static iree_status_t loom_scalar_division_build_remainder_bounds(
    loom_builder_t* builder, loom_value_id_t divisor, loom_type_t type,
    loom_location_id_t location, loom_value_id_t* out_lower,
    loom_value_id_t* out_upper) {
  loom_value_id_t sign = LOOM_VALUE_ID_INVALID;
  loom_value_id_t magnitude = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scalar_division_build_magnitude(
      builder, divisor, type, location, &sign, &magnitude));
  loom_op_t* one = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(builder, loom_attr_i64(1),
                                                  type, location, &one));
  loom_op_t* upper_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_subi_build(
      builder, /*instance_flags=*/0, magnitude,
      loom_scalar_constant_result(one), type, location, &upper_op));
  const loom_value_id_t upper = loom_scalar_subi_result(upper_op);
  const loom_predicate_t predicates[2] = {
      {
          .kind = LOOM_PREDICATE_GE,
          .arg_count = 2,
          .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
          .args = {upper, 0},
      },
      {
          .kind = LOOM_PREDICATE_ULT,
          .arg_count = 2,
          .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
          .args = {upper, magnitude},
      },
  };
  loom_op_t* assumption = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_assume_build(
      builder, &upper, 1, predicates, IREE_ARRAYSIZE(predicates), &type, 1,
      location, &assumption));
  *out_upper = loom_op_const_results(assumption)[0];
  loom_op_t* lower_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_negi_build(builder, *out_upper, type, location, &lower_op));
  *out_lower = loom_scalar_negi_result(lower_op);
  return iree_ok_status();
}

// Arithmetic reconstruction need not infer the original remainder's numeric
// facts or its semantic bounds against the runtime divisor.
iree_status_t loom_scalar_division_refine_reconstructed_remainder(
    loom_builder_t* builder, const loom_value_fact_table_t* fact_table,
    loom_value_id_t original_result, loom_value_id_t reconstructed_result,
    loom_value_id_t divisor, loom_type_t result_type, bool is_signed,
    loom_location_id_t location, loom_value_id_t* out_result) {
  *out_result = reconstructed_result;
  loom_predicate_t predicates[LOOM_VALUE_FACT_INTEGER_REFINEMENT_CAPACITY + 1];
  uint16_t predicate_count = 0;
  loom_value_facts_t divisor_facts = loom_value_facts_unknown();
  if (fact_table) {
    divisor_facts = loom_value_fact_table_lookup(fact_table, divisor);
    predicate_count = loom_value_facts_export_integer_refinements(
        reconstructed_result,
        loom_value_fact_table_lookup(fact_table, reconstructed_result),
        loom_value_fact_table_lookup(fact_table, original_result), predicates,
        IREE_ARRAYSIZE(predicates));
  }
  // Preserve semantic bounds independently of the facts available during
  // this mutable rewrite. A later guard may prove the divisor positive.
  if (!is_signed || loom_value_facts_is_positive(divisor_facts)) {
    predicates[predicate_count++] = (loom_predicate_t){
        .kind = is_signed ? LOOM_PREDICATE_LT : LOOM_PREDICATE_ULT,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE,
                     LOOM_PRED_ARG_NONE},
        .args = {reconstructed_result, divisor, 0},
    };
  } else {
    loom_value_id_t lower = LOOM_VALUE_ID_INVALID;
    loom_value_id_t upper = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scalar_division_build_remainder_bounds(
        builder, divisor, result_type, location, &lower, &upper));
    predicates[predicate_count++] = (loom_predicate_t){
        .kind = LOOM_PREDICATE_RANGE,
        .arg_count = 3,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE,
                     LOOM_PRED_ARG_VALUE},
        .args = {reconstructed_result, lower, upper},
    };
  }
  loom_op_t* assumption = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_assume_build(
      builder, &reconstructed_result, 1, predicates, predicate_count,
      &result_type, 1, location, &assumption));
  *out_result = loom_scalar_assume_results(assumption).values[0];
  return iree_ok_status();
}
