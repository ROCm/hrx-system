// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/lower/encode.h"

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/fragment.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"

static bool loom_aie2p_schema_is_bfp(
    loom_value_fact_encoded_operand_schema_t schema) {
  const loom_value_fact_encoded_operand_schema_t native = {
      .element_format = LOOM_VALUE_FACT_NUMERIC_FORMAT_BFP16EBS8,
      .payload_packing = LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT,
      .rounding_policy = LOOM_VALUE_FACT_ROUNDING_POLICY_FLUSH_SUBNORMAL,
      .payload_register_count = 18,
      .payload_element_count = 64,
  };
  return loom_value_fact_encoded_operand_schema_equal(schema, native);
}

bool loom_aie2p_value_has_bfp_storage(const loom_value_fact_table_t* fact_table,
                                      loom_value_id_t value) {
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, value);
  loom_value_fact_encoding_summary_t summary = {0};
  if (loom_value_facts_query_encoding_summary(&fact_table->context, facts,
                                              &summary)) {
    return loom_aie2p_schema_is_bfp(summary.storage_schema.encoded_operand);
  }
  loom_vector_fragment_fact_t fragment;
  return loom_vector_fragment_fact_query_value_facts(&fact_table->context,
                                                     facts, &fragment) &&
         loom_aie2p_schema_is_bfp(fragment.encoded_operand);
}

bool loom_aie2p_encode_matches(loom_low_lower_context_t* context,
                               const loom_op_t* source_op) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type =
      loom_module_value_type(module, loom_vector_encode_source(source_op));
  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_encode_result(source_op));
  return loom_type_is_vector(source_type) && loom_type_rank(source_type) == 1 &&
         loom_type_element_type(source_type) == LOOM_SCALAR_TYPE_BF16 &&
         loom_type_dim_static_size_at(source_type, 0) == 64 &&
         loom_type_is_vector(result_type) && loom_type_rank(result_type) == 1 &&
         loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_I8 &&
         loom_type_dim_static_size_at(result_type, 0) == 72 &&
         loom_vector_encode_auxiliary(source_op).count == 0 &&
         loom_aie2p_value_has_bfp_storage(
             loom_low_lower_context_fact_table(context),
             loom_vector_encode_result(source_op));
}

static iree_status_t loom_aie2p_emit_conversion(
    loom_low_lower_context_t* context, uint32_t ordinal, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_result) {
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor =
          &loom_low_lower_context_descriptor_set(context)->descriptors[ordinal],
  };
  loom_op_t* converted = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, &input, 1, loom_named_attr_slice_empty(),
      &result_type, 1, NULL, 0, location, &converted));
  *out_result = loom_low_op_results(converted).values[0];
  return iree_ok_status();
}

iree_status_t loom_aie2p_emit_encode(loom_low_lower_context_t* context,
                                     const loom_op_t* source_op) {
  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_encode_source(source_op), &source));
  loom_type_t input_half_type = loom_type_none();
  loom_type_t float_half_type = loom_type_none();
  loom_type_t float_type = loom_type_none();
  loom_type_t encoded_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, AIE2P_CORE_REG_CLASS_ID_AIE2P_VEC256, 2, &input_half_type));
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, AIE2P_CORE_REG_CLASS_ID_AIE2P_MBMS, 2, &float_half_type));
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, AIE2P_CORE_REG_CLASS_ID_AIE2P_MBMS, 4, &float_type));
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, AIE2P_CORE_REG_CLASS_ID_AIE2P_MEXA, 1, &encoded_type));
  loom_value_id_t halves[2];
  for (uint32_t half = 0; half < 2; ++half) {
    loom_op_t* slice = NULL;
    IREE_RETURN_IF_ERROR(loom_low_slice_build(
        loom_low_lower_context_builder(context), source, half * 2,
        input_half_type, source_op->location, &slice));
    IREE_RETURN_IF_ERROR(loom_aie2p_emit_conversion(
        context, AIE2P_CORE_DESCRIPTOR_REF_CONVERT_BF16X32_TO_F32X32,
        loom_low_slice_result(slice), float_half_type, source_op->location,
        &halves[half]));
  }
  loom_op_t* joined = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(loom_low_lower_context_builder(context), halves, 2,
                            float_type, source_op->location, &joined));

  const loom_low_lower_resolved_descriptor_t rounding = {
      .descriptor = &loom_low_lower_context_descriptor_set(context)->descriptors
                         [AIE2P_CORE_DESCRIPTOR_REF_STATE_ROUNDING_IMMEDIATE],
  };
  loom_string_id_t immediate = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      loom_low_lower_context_module(context), IREE_SV("i"), &immediate));
  const loom_named_attr_t mode = {.name_id = immediate,
                                  .value = loom_attr_i64(12)};
  loom_op_t* rounding_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &rounding, NULL, 0, loom_make_named_attr_slice(&mode, 1), NULL,
      0, NULL, 0, source_op->location, &rounding_op));

  loom_value_id_t encoded = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_emit_conversion(
      context, AIE2P_CORE_DESCRIPTOR_REF_CONVERT_F32X64_BFP16EBS8,
      loom_low_concat_result(joined), encoded_type, source_op->location,
      &encoded));
  return loom_low_lower_bind_value(
      context, loom_vector_encode_result(source_op), encoded);
}
