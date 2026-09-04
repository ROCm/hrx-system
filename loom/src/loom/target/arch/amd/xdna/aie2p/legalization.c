// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/legalization.h"

#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/transforms/vector/packet_legalization.h"
#include "loom/transforms/vector/to_scalar.h"

static const uint16_t kAie2pVectorPacketBitCounts[] = {128u, 256u, 512u};

static const loom_vector_packet_policy_t kAie2pVectorPacketPolicy = {
    .native_bit_counts = kAie2pVectorPacketBitCounts,
    .native_bit_count_count = IREE_ARRAYSIZE(kAie2pVectorPacketBitCounts),
    .maximum_unpacketized_bit_count = 0,
};

static bool loom_aie2p_legalizer_descriptor_set_is_core(
    const loom_low_descriptor_set_t* descriptor_set) {
  return descriptor_set == loom_aie2p_core_descriptor_set();
}

typedef enum loom_aie2p_f32_compare_relation_e {
  LOOM_AIE2P_F32_COMPARE_RELATION_EQ = 0,
  LOOM_AIE2P_F32_COMPARE_RELATION_GT,
  LOOM_AIE2P_F32_COMPARE_RELATION_GE,
  LOOM_AIE2P_F32_COMPARE_RELATION_LT,
  LOOM_AIE2P_F32_COMPARE_RELATION_LE,
  LOOM_AIE2P_F32_COMPARE_RELATION_NE,
  LOOM_AIE2P_F32_COMPARE_RELATION_ORDERED,
  LOOM_AIE2P_F32_COMPARE_RELATION_UNORDERED,
} loom_aie2p_f32_compare_relation_t;

typedef enum loom_aie2p_f32_compare_nan_policy_e {
  LOOM_AIE2P_F32_COMPARE_NAN_POLICY_CLEAR = 0,
  LOOM_AIE2P_F32_COMPARE_NAN_POLICY_SET,
} loom_aie2p_f32_compare_nan_policy_t;

typedef struct loom_aie2p_f32_compare_plan_t {
  // Numeric relation evaluated after mapping non-NaN values to integer keys.
  uint8_t relation;
  // Result behavior when either operand is a NaN.
  uint8_t nan_policy;
} loom_aie2p_f32_compare_plan_t;

static const loom_aie2p_f32_compare_plan_t kAie2pF32ComparePlans[] = {
    [LOOM_VECTOR_CMPF_PREDICATE_OEQ] =
        {LOOM_AIE2P_F32_COMPARE_RELATION_EQ,
         LOOM_AIE2P_F32_COMPARE_NAN_POLICY_CLEAR},
    [LOOM_VECTOR_CMPF_PREDICATE_OGT] =
        {LOOM_AIE2P_F32_COMPARE_RELATION_GT,
         LOOM_AIE2P_F32_COMPARE_NAN_POLICY_CLEAR},
    [LOOM_VECTOR_CMPF_PREDICATE_OGE] =
        {LOOM_AIE2P_F32_COMPARE_RELATION_GE,
         LOOM_AIE2P_F32_COMPARE_NAN_POLICY_CLEAR},
    [LOOM_VECTOR_CMPF_PREDICATE_OLT] =
        {LOOM_AIE2P_F32_COMPARE_RELATION_LT,
         LOOM_AIE2P_F32_COMPARE_NAN_POLICY_CLEAR},
    [LOOM_VECTOR_CMPF_PREDICATE_OLE] =
        {LOOM_AIE2P_F32_COMPARE_RELATION_LE,
         LOOM_AIE2P_F32_COMPARE_NAN_POLICY_CLEAR},
    [LOOM_VECTOR_CMPF_PREDICATE_ONE] =
        {LOOM_AIE2P_F32_COMPARE_RELATION_NE,
         LOOM_AIE2P_F32_COMPARE_NAN_POLICY_CLEAR},
    [LOOM_VECTOR_CMPF_PREDICATE_ORD] =
        {LOOM_AIE2P_F32_COMPARE_RELATION_ORDERED,
         LOOM_AIE2P_F32_COMPARE_NAN_POLICY_CLEAR},
    [LOOM_VECTOR_CMPF_PREDICATE_UEQ] = {LOOM_AIE2P_F32_COMPARE_RELATION_EQ,
                                        LOOM_AIE2P_F32_COMPARE_NAN_POLICY_SET},
    [LOOM_VECTOR_CMPF_PREDICATE_UGT] = {LOOM_AIE2P_F32_COMPARE_RELATION_GT,
                                        LOOM_AIE2P_F32_COMPARE_NAN_POLICY_SET},
    [LOOM_VECTOR_CMPF_PREDICATE_UGE] = {LOOM_AIE2P_F32_COMPARE_RELATION_GE,
                                        LOOM_AIE2P_F32_COMPARE_NAN_POLICY_SET},
    [LOOM_VECTOR_CMPF_PREDICATE_ULT] = {LOOM_AIE2P_F32_COMPARE_RELATION_LT,
                                        LOOM_AIE2P_F32_COMPARE_NAN_POLICY_SET},
    [LOOM_VECTOR_CMPF_PREDICATE_ULE] = {LOOM_AIE2P_F32_COMPARE_RELATION_LE,
                                        LOOM_AIE2P_F32_COMPARE_NAN_POLICY_SET},
    [LOOM_VECTOR_CMPF_PREDICATE_UNE] = {LOOM_AIE2P_F32_COMPARE_RELATION_NE,
                                        LOOM_AIE2P_F32_COMPARE_NAN_POLICY_SET},
    [LOOM_VECTOR_CMPF_PREDICATE_UNO] =
        {LOOM_AIE2P_F32_COMPARE_RELATION_UNORDERED,
         LOOM_AIE2P_F32_COMPARE_NAN_POLICY_SET},
};

static const uint8_t kAie2pF32CompareIntegerPredicates[] = {
    [LOOM_AIE2P_F32_COMPARE_RELATION_EQ] = LOOM_VECTOR_CMPI_PREDICATE_EQ,
    [LOOM_AIE2P_F32_COMPARE_RELATION_GT] = LOOM_VECTOR_CMPI_PREDICATE_UGT,
    [LOOM_AIE2P_F32_COMPARE_RELATION_GE] = LOOM_VECTOR_CMPI_PREDICATE_UGE,
    [LOOM_AIE2P_F32_COMPARE_RELATION_LT] = LOOM_VECTOR_CMPI_PREDICATE_ULT,
    [LOOM_AIE2P_F32_COMPARE_RELATION_LE] = LOOM_VECTOR_CMPI_PREDICATE_ULE,
    [LOOM_AIE2P_F32_COMPARE_RELATION_NE] = LOOM_VECTOR_CMPI_PREDICATE_NE,
};

typedef struct loom_aie2p_f32_compare_values_t {
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
} loom_aie2p_f32_compare_values_t;

static loom_type_t loom_aie2p_vector_type_with_element(
    loom_type_t source_type, loom_scalar_type_t element_type) {
  source_type.header = loom_type_make_header(
      loom_type_kind(source_type), element_type, loom_type_rank(source_type),
      loom_type_flags(source_type));
  return source_type;
}

static iree_status_t loom_aie2p_build_vector_i32_constant(
    loom_builder_t* builder, loom_location_id_t location,
    loom_type_t vector_type, int32_t value, loom_value_id_t* out_value) {
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

static iree_status_t loom_aie2p_build_f32_compare_values(
    loom_builder_t* builder, loom_location_id_t location, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t f32_type, loom_type_t i32_type,
    loom_type_t predicate_type, loom_aie2p_f32_compare_values_t* out_values) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(builder, lhs, f32_type,
                                                 i32_type, location, &op));
  out_values->lhs_bits = loom_vector_bitcast_result(op);
  IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(builder, rhs, f32_type,
                                                 i32_type, location, &op));
  out_values->rhs_bits = loom_vector_bitcast_result(op);

  loom_value_id_t magnitude_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_build_vector_i32_constant(
      builder, location, i32_type, INT32_MAX, &magnitude_mask));
  IREE_RETURN_IF_ERROR(loom_vector_andi_build(
      builder, out_values->lhs_bits, magnitude_mask, i32_type, location, &op));
  out_values->lhs_magnitude = loom_vector_andi_result(op);
  IREE_RETURN_IF_ERROR(loom_vector_andi_build(
      builder, out_values->rhs_bits, magnitude_mask, i32_type, location, &op));
  out_values->rhs_magnitude = loom_vector_andi_result(op);

  loom_value_id_t infinity = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_build_vector_i32_constant(
      builder, location, i32_type, INT32_C(0x7F800000), &infinity));
  loom_value_id_t lhs_nan = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs_nan = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_build_vector_compare(
      builder, location, LOOM_VECTOR_CMPI_PREDICATE_ULT, infinity,
      out_values->lhs_magnitude, i32_type, predicate_type, &lhs_nan));
  IREE_RETURN_IF_ERROR(loom_aie2p_build_vector_compare(
      builder, location, LOOM_VECTOR_CMPI_PREDICATE_ULT, infinity,
      out_values->rhs_magnitude, i32_type, predicate_type, &rhs_nan));
  IREE_RETURN_IF_ERROR(loom_vector_ori_build(builder, lhs_nan, rhs_nan,
                                             predicate_type, location, &op));
  out_values->unordered = loom_vector_ori_result(op);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_build_f32_sort_key(
    loom_builder_t* builder, loom_location_id_t location, loom_value_id_t bits,
    loom_value_id_t magnitude, loom_value_id_t zero, loom_value_id_t sign_mask,
    loom_value_id_t all_ones, loom_type_t i32_type, loom_type_t predicate_type,
    loom_value_id_t* out_key) {
  // Map non-NaN IEEE bit patterns to monotonically ordered unsigned integers.
  // Canonicalize both zeros, move positive values above the sign boundary, and
  // reverse negative bit patterns so ordinary integer compares preserve order.
  loom_value_id_t is_zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_build_vector_compare(
      builder, location, LOOM_VECTOR_CMPI_PREDICATE_EQ, magnitude, zero,
      i32_type, predicate_type, &is_zero));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_select_build(builder, is_zero, zero, bits,
                                                i32_type, location, &op));
  const loom_value_id_t canonical_bits = loom_vector_select_result(op);

  loom_value_id_t is_negative = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_build_vector_compare(
      builder, location, LOOM_VECTOR_CMPI_PREDICATE_SLT, canonical_bits, zero,
      i32_type, predicate_type, &is_negative));
  IREE_RETURN_IF_ERROR(loom_vector_addi_build(
      builder, 0, canonical_bits, sign_mask, i32_type, location, &op));
  const loom_value_id_t positive_key = loom_vector_addi_result(op);
  IREE_RETURN_IF_ERROR(loom_vector_subi_build(
      builder, 0, all_ones, canonical_bits, i32_type, location, &op));
  const loom_value_id_t negative_key = loom_vector_subi_result(op);
  IREE_RETURN_IF_ERROR(loom_vector_select_build(builder, is_negative,
                                                negative_key, positive_key,
                                                i32_type, location, &op));
  *out_key = loom_vector_select_result(op);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_build_f32_base_relation(
    loom_builder_t* builder, loom_location_id_t location,
    loom_aie2p_f32_compare_relation_t relation, loom_value_id_t lhs_key,
    loom_value_id_t rhs_key, loom_type_t i32_type, loom_type_t predicate_type,
    loom_value_id_t* out_value) {
  return loom_aie2p_build_vector_compare(
      builder, location, kAie2pF32CompareIntegerPredicates[relation], lhs_key,
      rhs_key, i32_type, predicate_type, out_value);
}

static iree_status_t loom_aie2p_legalize_vector_cmpf(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_aie2p_legalizer_descriptor_set_is_core(context->descriptor_set)) {
    return iree_ok_status();
  }

  const loom_value_id_t lhs = loom_vector_cmpf_lhs(op);
  const loom_type_t f32_type = loom_module_value_type(context->module, lhs);
  uint64_t element_count = 0;
  if (!loom_type_is_vector(f32_type) ||
      loom_type_element_type(f32_type) != LOOM_SCALAR_TYPE_F32 ||
      !loom_type_static_element_count(f32_type, &element_count) ||
      element_count == 0 || element_count > 16) {
    return iree_ok_status();
  }

  const loom_type_t i32_type =
      loom_aie2p_vector_type_with_element(f32_type, LOOM_SCALAR_TYPE_I32);
  const loom_type_t predicate_type =
      loom_aie2p_vector_type_with_element(f32_type, LOOM_SCALAR_TYPE_I1);
  const loom_aie2p_f32_compare_plan_t plan =
      kAie2pF32ComparePlans[loom_vector_cmpf_predicate(op)];

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_aie2p_f32_compare_values_t values = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_build_f32_compare_values(
      &rewriter->builder, op->location, lhs, loom_vector_cmpf_rhs(op), f32_type,
      i32_type, predicate_type, &values));

  loom_value_id_t replacement = values.unordered;
  if (plan.relation != LOOM_AIE2P_F32_COMPARE_RELATION_UNORDERED) {
    loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_build_vector_i32_constant(
        &rewriter->builder, op->location, i32_type, 0, &zero));
    if (plan.relation == LOOM_AIE2P_F32_COMPARE_RELATION_ORDERED) {
      loom_value_id_t all_true = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_aie2p_build_vector_compare(
          &rewriter->builder, op->location, LOOM_VECTOR_CMPI_PREDICATE_EQ, zero,
          zero, i32_type, predicate_type, &all_true));
      loom_op_t* invert_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_vector_xori_build(&rewriter->builder, all_true, values.unordered,
                                 predicate_type, op->location, &invert_op));
      replacement = loom_vector_xori_result(invert_op);
    } else {
      loom_value_id_t sign_mask = LOOM_VALUE_ID_INVALID;
      loom_value_id_t all_ones = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_aie2p_build_vector_i32_constant(
          &rewriter->builder, op->location, i32_type, INT32_MIN, &sign_mask));
      IREE_RETURN_IF_ERROR(loom_aie2p_build_vector_i32_constant(
          &rewriter->builder, op->location, i32_type, -1, &all_ones));
      loom_value_id_t lhs_key = LOOM_VALUE_ID_INVALID;
      loom_value_id_t rhs_key = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_aie2p_build_f32_sort_key(
          &rewriter->builder, op->location, values.lhs_bits,
          values.lhs_magnitude, zero, sign_mask, all_ones, i32_type,
          predicate_type, &lhs_key));
      IREE_RETURN_IF_ERROR(loom_aie2p_build_f32_sort_key(
          &rewriter->builder, op->location, values.rhs_bits,
          values.rhs_magnitude, zero, sign_mask, all_ones, i32_type,
          predicate_type, &rhs_key));
      loom_value_id_t base = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_aie2p_build_f32_base_relation(
          &rewriter->builder, op->location, plan.relation, lhs_key, rhs_key,
          i32_type, predicate_type, &base));

      loom_op_t* combine_op = NULL;
      if (plan.nan_policy == LOOM_AIE2P_F32_COMPARE_NAN_POLICY_SET) {
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

static iree_status_t loom_aie2p_legalize_vector_load(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_aie2p_legalizer_descriptor_set_is_core(context->descriptor_set)) {
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_descriptor_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_legalize_vector_store(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_aie2p_legalizer_descriptor_set_is_core(context->descriptor_set)) {
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_store(
      context, op, &kAie2pVectorPacketPolicy, &rewritten));
  if (!rewritten) {
    IREE_RETURN_IF_ERROR(loom_vector_store_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_legalize_vector_reduce(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_aie2p_legalizer_descriptor_set_is_core(context->descriptor_set)) {
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_reduce(
      context, op, &kAie2pVectorPacketPolicy, &rewritten));
  if (!rewritten) {
    IREE_RETURN_IF_ERROR(loom_vector_reduce_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static const loom_target_legalizer_rule_t kAie2pLegalizerRules[] = {
    {
        .root_kind = LOOM_OP_VECTOR_CMPF,
        .legalize = loom_aie2p_legalize_vector_cmpf,
    },
    {
        .root_kind = LOOM_OP_VECTOR_LOAD,
        .legalize = loom_aie2p_legalize_vector_load,
    },
    {
        .root_kind = LOOM_OP_VECTOR_STORE,
        .legalize = loom_aie2p_legalize_vector_store,
    },
    {
        .root_kind = LOOM_OP_VECTOR_REDUCE,
        .legalize = loom_aie2p_legalize_vector_reduce,
    },
};

const loom_target_legalizer_provider_t
    loom_aie2p_target_legalizer_provider_storage = {
        .name = IREE_SVL("aie2p"),
        .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_TARGET,
        .rules = kAie2pLegalizerRules,
        .rule_count = IREE_ARRAYSIZE(kAie2pLegalizerRules),
};

const loom_target_legalizer_provider_t* loom_aie2p_target_legalizer_provider(
    void) {
  return &loom_aie2p_target_legalizer_provider_storage;
}
