// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/legalization_compare.h"

#include "loom/ir/module.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"

typedef enum loom_aie2p_float_compare_relation_e {
  LOOM_AIE2P_FLOAT_COMPARE_RELATION_EQ = 0,
  LOOM_AIE2P_FLOAT_COMPARE_RELATION_GT,
  LOOM_AIE2P_FLOAT_COMPARE_RELATION_GE,
  LOOM_AIE2P_FLOAT_COMPARE_RELATION_LT,
  LOOM_AIE2P_FLOAT_COMPARE_RELATION_LE,
  LOOM_AIE2P_FLOAT_COMPARE_RELATION_NE,
  LOOM_AIE2P_FLOAT_COMPARE_RELATION_ORDERED,
  LOOM_AIE2P_FLOAT_COMPARE_RELATION_UNORDERED,
} loom_aie2p_float_compare_relation_t;

typedef enum loom_aie2p_float_compare_nan_policy_e {
  LOOM_AIE2P_FLOAT_COMPARE_NAN_POLICY_CLEAR = 0,
  LOOM_AIE2P_FLOAT_COMPARE_NAN_POLICY_SET,
} loom_aie2p_float_compare_nan_policy_t;

typedef struct loom_aie2p_float_compare_plan_t {
  // Numeric relation evaluated after mapping non-NaN values to integer keys.
  uint8_t relation;
  // Result behavior when either operand is a NaN.
  uint8_t nan_policy;
} loom_aie2p_float_compare_plan_t;

static const loom_aie2p_float_compare_plan_t kAie2pFloatComparePlans[] = {
    [LOOM_VECTOR_CMPF_PREDICATE_OEQ] =
        {LOOM_AIE2P_FLOAT_COMPARE_RELATION_EQ,
         LOOM_AIE2P_FLOAT_COMPARE_NAN_POLICY_CLEAR},
    [LOOM_VECTOR_CMPF_PREDICATE_OGT] =
        {LOOM_AIE2P_FLOAT_COMPARE_RELATION_GT,
         LOOM_AIE2P_FLOAT_COMPARE_NAN_POLICY_CLEAR},
    [LOOM_VECTOR_CMPF_PREDICATE_OGE] =
        {LOOM_AIE2P_FLOAT_COMPARE_RELATION_GE,
         LOOM_AIE2P_FLOAT_COMPARE_NAN_POLICY_CLEAR},
    [LOOM_VECTOR_CMPF_PREDICATE_OLT] =
        {LOOM_AIE2P_FLOAT_COMPARE_RELATION_LT,
         LOOM_AIE2P_FLOAT_COMPARE_NAN_POLICY_CLEAR},
    [LOOM_VECTOR_CMPF_PREDICATE_OLE] =
        {LOOM_AIE2P_FLOAT_COMPARE_RELATION_LE,
         LOOM_AIE2P_FLOAT_COMPARE_NAN_POLICY_CLEAR},
    [LOOM_VECTOR_CMPF_PREDICATE_ONE] =
        {LOOM_AIE2P_FLOAT_COMPARE_RELATION_NE,
         LOOM_AIE2P_FLOAT_COMPARE_NAN_POLICY_CLEAR},
    [LOOM_VECTOR_CMPF_PREDICATE_ORD] =
        {LOOM_AIE2P_FLOAT_COMPARE_RELATION_ORDERED,
         LOOM_AIE2P_FLOAT_COMPARE_NAN_POLICY_CLEAR},
    [LOOM_VECTOR_CMPF_PREDICATE_UEQ] =
        {LOOM_AIE2P_FLOAT_COMPARE_RELATION_EQ,
         LOOM_AIE2P_FLOAT_COMPARE_NAN_POLICY_SET},
    [LOOM_VECTOR_CMPF_PREDICATE_UGT] =
        {LOOM_AIE2P_FLOAT_COMPARE_RELATION_GT,
         LOOM_AIE2P_FLOAT_COMPARE_NAN_POLICY_SET},
    [LOOM_VECTOR_CMPF_PREDICATE_UGE] =
        {LOOM_AIE2P_FLOAT_COMPARE_RELATION_GE,
         LOOM_AIE2P_FLOAT_COMPARE_NAN_POLICY_SET},
    [LOOM_VECTOR_CMPF_PREDICATE_ULT] =
        {LOOM_AIE2P_FLOAT_COMPARE_RELATION_LT,
         LOOM_AIE2P_FLOAT_COMPARE_NAN_POLICY_SET},
    [LOOM_VECTOR_CMPF_PREDICATE_ULE] =
        {LOOM_AIE2P_FLOAT_COMPARE_RELATION_LE,
         LOOM_AIE2P_FLOAT_COMPARE_NAN_POLICY_SET},
    [LOOM_VECTOR_CMPF_PREDICATE_UNE] =
        {LOOM_AIE2P_FLOAT_COMPARE_RELATION_NE,
         LOOM_AIE2P_FLOAT_COMPARE_NAN_POLICY_SET},
    [LOOM_VECTOR_CMPF_PREDICATE_UNO] =
        {LOOM_AIE2P_FLOAT_COMPARE_RELATION_UNORDERED,
         LOOM_AIE2P_FLOAT_COMPARE_NAN_POLICY_SET},
};

static const uint8_t kAie2pFloatCompareIntegerPredicates[] = {
    [LOOM_AIE2P_FLOAT_COMPARE_RELATION_EQ] = LOOM_VECTOR_CMPI_PREDICATE_EQ,
    [LOOM_AIE2P_FLOAT_COMPARE_RELATION_GT] = LOOM_VECTOR_CMPI_PREDICATE_UGT,
    [LOOM_AIE2P_FLOAT_COMPARE_RELATION_GE] = LOOM_VECTOR_CMPI_PREDICATE_UGE,
    [LOOM_AIE2P_FLOAT_COMPARE_RELATION_LT] = LOOM_VECTOR_CMPI_PREDICATE_ULT,
    [LOOM_AIE2P_FLOAT_COMPARE_RELATION_LE] = LOOM_VECTOR_CMPI_PREDICATE_ULE,
    [LOOM_AIE2P_FLOAT_COMPARE_RELATION_NE] = LOOM_VECTOR_CMPI_PREDICATE_NE,
};

typedef struct loom_aie2p_float_compare_values_t {
  // Left operand reinterpreted as integer bits.
  loom_value_id_t lhs_bits;
  // Right operand reinterpreted as integer bits.
  loom_value_id_t rhs_bits;
  // Left operand with its sign bit cleared.
  loom_value_id_t lhs_magnitude;
  // Right operand with its sign bit cleared.
  loom_value_id_t rhs_magnitude;
  // Per-lane mask set when either operand is a NaN.
  loom_value_id_t unordered;
} loom_aie2p_float_compare_values_t;

static loom_type_t loom_aie2p_vector_type_with_element(
    loom_type_t source_type, loom_scalar_type_t element_type) {
  source_type.header = loom_type_make_header(
      loom_type_kind(source_type), element_type, loom_type_rank(source_type),
      loom_type_flags(source_type));
  return source_type;
}

static iree_status_t loom_aie2p_build_vector_integer_constant(
    loom_builder_t* builder, loom_location_id_t location,
    loom_type_t vector_type, int64_t value, loom_value_id_t* out_value) {
  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_constant_build(
      builder, loom_attr_i64(value), vector_type, location, &constant_op));
  *out_value = loom_vector_constant_result(constant_op);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_build_vector_compare(
    loom_builder_t* builder, loom_location_id_t location, uint8_t predicate,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_type_t operand_type,
    loom_type_t result_type, loom_value_id_t* out_value) {
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_cmpi_build(builder, predicate, lhs, rhs,
                                              operand_type, result_type,
                                              location, &compare_op));
  *out_value = loom_vector_cmpi_result(compare_op);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_build_float_compare_values(
    loom_builder_t* builder, loom_location_id_t location, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t float_type, loom_type_t integer_type,
    loom_type_t predicate_type, uint32_t bit_count, uint32_t infinity_bits,
    loom_aie2p_float_compare_values_t* out_values) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(builder, lhs, float_type,
                                                 integer_type, location, &op));
  out_values->lhs_bits = loom_vector_bitcast_result(op);
  IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(builder, rhs, float_type,
                                                 integer_type, location, &op));
  out_values->rhs_bits = loom_vector_bitcast_result(op);

  loom_value_id_t magnitude_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_build_vector_integer_constant(
      builder, location, integer_type, (UINT64_C(1) << (bit_count - 1)) - 1,
      &magnitude_mask));
  IREE_RETURN_IF_ERROR(loom_vector_andi_build(builder, out_values->lhs_bits,
                                              magnitude_mask, integer_type,
                                              location, &op));
  out_values->lhs_magnitude = loom_vector_andi_result(op);
  IREE_RETURN_IF_ERROR(loom_vector_andi_build(builder, out_values->rhs_bits,
                                              magnitude_mask, integer_type,
                                              location, &op));
  out_values->rhs_magnitude = loom_vector_andi_result(op);

  loom_value_id_t infinity = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_build_vector_integer_constant(
      builder, location, integer_type, infinity_bits, &infinity));
  loom_value_id_t lhs_nan = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs_nan = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_build_vector_compare(
      builder, location, LOOM_VECTOR_CMPI_PREDICATE_ULT, infinity,
      out_values->lhs_magnitude, integer_type, predicate_type, &lhs_nan));
  IREE_RETURN_IF_ERROR(loom_aie2p_build_vector_compare(
      builder, location, LOOM_VECTOR_CMPI_PREDICATE_ULT, infinity,
      out_values->rhs_magnitude, integer_type, predicate_type, &rhs_nan));
  IREE_RETURN_IF_ERROR(loom_vector_ori_build(builder, lhs_nan, rhs_nan,
                                             predicate_type, location, &op));
  out_values->unordered = loom_vector_ori_result(op);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_build_float_sort_key(
    loom_builder_t* builder, loom_location_id_t location, loom_value_id_t bits,
    loom_value_id_t magnitude, loom_value_id_t zero, loom_value_id_t sign_mask,
    loom_value_id_t all_ones, loom_type_t integer_type,
    loom_type_t predicate_type, loom_value_id_t* out_key) {
  // Map non-NaN IEEE bit patterns to monotonically ordered unsigned integers.
  // Canonicalize both zeros, move positive values above the sign boundary, and
  // reverse negative bit patterns so ordinary integer compares preserve order.
  loom_value_id_t is_zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_build_vector_compare(
      builder, location, LOOM_VECTOR_CMPI_PREDICATE_EQ, magnitude, zero,
      integer_type, predicate_type, &is_zero));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_select_build(builder, is_zero, zero, bits,
                                                integer_type, location, &op));
  const loom_value_id_t canonical_bits = loom_vector_select_result(op);

  loom_value_id_t is_negative = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_build_vector_compare(
      builder, location, LOOM_VECTOR_CMPI_PREDICATE_SLT, canonical_bits, zero,
      integer_type, predicate_type, &is_negative));
  IREE_RETURN_IF_ERROR(loom_vector_addi_build(
      builder, 0, canonical_bits, sign_mask, integer_type, location, &op));
  const loom_value_id_t positive_key = loom_vector_addi_result(op);
  IREE_RETURN_IF_ERROR(loom_vector_subi_build(
      builder, 0, all_ones, canonical_bits, integer_type, location, &op));
  const loom_value_id_t negative_key = loom_vector_subi_result(op);
  IREE_RETURN_IF_ERROR(loom_vector_select_build(builder, is_negative,
                                                negative_key, positive_key,
                                                integer_type, location, &op));
  *out_key = loom_vector_select_result(op);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_build_float_base_relation(
    loom_builder_t* builder, loom_location_id_t location,
    loom_aie2p_float_compare_relation_t relation, loom_value_id_t lhs_key,
    loom_value_id_t rhs_key, loom_type_t integer_type,
    loom_type_t predicate_type, loom_value_id_t* out_value) {
  return loom_aie2p_build_vector_compare(
      builder, location, kAie2pFloatCompareIntegerPredicates[relation], lhs_key,
      rhs_key, integer_type, predicate_type, out_value);
}

iree_status_t loom_aie2p_legalize_vector_cmpf(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->descriptor_set != loom_aie2p_core_descriptor_set()) {
    return iree_ok_status();
  }

  const loom_value_id_t lhs = loom_vector_cmpf_lhs(op);
  const loom_type_t float_type = loom_module_value_type(context->module, lhs);
  const loom_scalar_type_t element_type = loom_type_element_type(float_type);
  uint32_t infinity_bits = 0;
  switch (element_type) {
    case LOOM_SCALAR_TYPE_F16:
      infinity_bits = UINT32_C(0x7C00);
      break;
    case LOOM_SCALAR_TYPE_BF16:
      infinity_bits = UINT32_C(0x7F80);
      break;
    case LOOM_SCALAR_TYPE_F32:
      infinity_bits = UINT32_C(0x7F800000);
      break;
    default:
      return iree_ok_status();
  }
  const uint32_t bit_count = loom_scalar_type_bitwidth(element_type);
  uint64_t element_count = 0;
  if (!loom_type_static_element_count(float_type, &element_count) ||
      element_count == 0 || element_count > 512 / bit_count) {
    return iree_ok_status();
  }

  const loom_type_t integer_type = loom_aie2p_vector_type_with_element(
      float_type,
      bit_count == 16 ? LOOM_SCALAR_TYPE_I16 : LOOM_SCALAR_TYPE_I32);
  const loom_type_t predicate_type =
      loom_aie2p_vector_type_with_element(float_type, LOOM_SCALAR_TYPE_I1);
  const loom_aie2p_float_compare_plan_t plan =
      kAie2pFloatComparePlans[loom_vector_cmpf_predicate(op)];

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_aie2p_float_compare_values_t values = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_build_float_compare_values(
      &rewriter->builder, op->location, lhs, loom_vector_cmpf_rhs(op),
      float_type, integer_type, predicate_type, bit_count, infinity_bits,
      &values));

  loom_value_id_t replacement = values.unordered;
  if (plan.relation != LOOM_AIE2P_FLOAT_COMPARE_RELATION_UNORDERED) {
    loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_build_vector_integer_constant(
        &rewriter->builder, op->location, integer_type, 0, &zero));
    if (plan.relation == LOOM_AIE2P_FLOAT_COMPARE_RELATION_ORDERED) {
      loom_value_id_t all_true = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_aie2p_build_vector_compare(
          &rewriter->builder, op->location, LOOM_VECTOR_CMPI_PREDICATE_EQ, zero,
          zero, integer_type, predicate_type, &all_true));
      loom_op_t* invert_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_vector_xori_build(&rewriter->builder, all_true, values.unordered,
                                 predicate_type, op->location, &invert_op));
      replacement = loom_vector_xori_result(invert_op);
    } else {
      loom_value_id_t sign_mask = LOOM_VALUE_ID_INVALID;
      loom_value_id_t all_ones = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_aie2p_build_vector_integer_constant(
          &rewriter->builder, op->location, integer_type,
          -(INT64_C(1) << (bit_count - 1)), &sign_mask));
      IREE_RETURN_IF_ERROR(loom_aie2p_build_vector_integer_constant(
          &rewriter->builder, op->location, integer_type, -1, &all_ones));
      loom_value_id_t lhs_key = LOOM_VALUE_ID_INVALID;
      loom_value_id_t rhs_key = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_aie2p_build_float_sort_key(
          &rewriter->builder, op->location, values.lhs_bits,
          values.lhs_magnitude, zero, sign_mask, all_ones, integer_type,
          predicate_type, &lhs_key));
      IREE_RETURN_IF_ERROR(loom_aie2p_build_float_sort_key(
          &rewriter->builder, op->location, values.rhs_bits,
          values.rhs_magnitude, zero, sign_mask, all_ones, integer_type,
          predicate_type, &rhs_key));
      loom_value_id_t base = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_aie2p_build_float_base_relation(
          &rewriter->builder, op->location, plan.relation, lhs_key, rhs_key,
          integer_type, predicate_type, &base));

      loom_op_t* combine_op = NULL;
      if (plan.nan_policy == LOOM_AIE2P_FLOAT_COMPARE_NAN_POLICY_SET) {
        IREE_RETURN_IF_ERROR(
            loom_vector_ori_build(&rewriter->builder, base, values.unordered,
                                  predicate_type, op->location, &combine_op));
        replacement = loom_vector_ori_result(combine_op);
      } else {
        // Clear unordered lanes without materializing an all-true predicate:
        // base ^ (base & unordered) is equivalent to base & ~unordered.
        IREE_RETURN_IF_ERROR(
            loom_vector_andi_build(&rewriter->builder, base, values.unordered,
                                   predicate_type, op->location, &combine_op));
        const loom_value_id_t unordered_base =
            loom_vector_andi_result(combine_op);
        IREE_RETURN_IF_ERROR(
            loom_vector_xori_build(&rewriter->builder, base, unordered_base,
                                   predicate_type, op->location, &combine_op));
        replacement = loom_vector_xori_result(combine_op);
      }
    }
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}
