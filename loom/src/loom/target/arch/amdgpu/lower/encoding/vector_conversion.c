// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/encoding/vector_conversion.h"

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/encoding/auxiliary.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/vector/storage.h"
#include "loom/target/arch/amdgpu/lower/arithmetic.h"
#include "loom/target/arch/amdgpu/lower/bitpack.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/encoding/float16.h"
#include "loom/target/arch/amdgpu/lower/encoding/fp4.h"
#include "loom/target/arch/amdgpu/lower/encoding/fp8.h"
#include "loom/target/arch/amdgpu/lower/encoding/fp8_encode.h"
#include "loom/target/arch/amdgpu/lower/encoding/fp8_vector_conversion.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static bool loom_amdgpu_type_is_16bit_float_packed_vector(
    loom_type_t type, loom_scalar_type_t element_type, uint32_t* out_lane_count,
    uint32_t* out_register_count) {
  *out_lane_count = 0;
  *out_register_count = 0;
  if (loom_type_element_type(type) != element_type) {
    return false;
  }
  uint32_t payload_bit_count = 0;
  if (!loom_amdgpu_type_packed_16bit_float_storage(type, &payload_bit_count,
                                                   out_register_count)) {
    return false;
  }
  *out_lane_count = payload_bit_count / 16u;
  return true;
}

static bool loom_amdgpu_type_is_extf_packed_float_vector(
    loom_type_t type, loom_scalar_type_t element_type, uint32_t* out_lane_count,
    uint32_t* out_register_count) {
  *out_lane_count = 0;
  *out_register_count = 0;
  if (loom_type_element_type(type) != element_type) {
    return false;
  }
  uint32_t payload_bit_count = 0;
  if (element_type == LOOM_SCALAR_TYPE_F8E4M3 ||
      element_type == LOOM_SCALAR_TYPE_F8E5M2) {
    if (!loom_amdgpu_type_packed_8bit_float_storage(type, &payload_bit_count,
                                                    out_register_count)) {
      return false;
    }
    *out_lane_count = payload_bit_count / 8u;
    return true;
  }
  if (!loom_amdgpu_type_packed_16bit_float_storage(type, &payload_bit_count,
                                                   out_register_count)) {
    return false;
  }
  *out_lane_count = payload_bit_count / 16u;
  return true;
}

static loom_value_fact_numeric_format_flags_t
loom_amdgpu_vector_storage_numeric_format(
    const loom_value_fact_table_t* fact_table, loom_value_id_t source,
    loom_value_fact_numeric_format_flags_t fallback_format) {
  loom_value_fact_encoding_summary_t summary = {0};
  if (loom_value_facts_query_encoding_summary(
          &fact_table->context,
          loom_value_fact_table_lookup(fact_table, source), &summary) &&
      summary.storage_schema.encoded_operand.element_format !=
          LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE) {
    return summary.storage_schema.encoded_operand.element_format;
  }
  return fallback_format;
}

static bool loom_amdgpu_direct_fp8_e8m0_pk8_descriptor_available(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_fact_numeric_format_flags_t source_format,
    loom_scalar_type_t result_element_type) {
  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  return loom_amdgpu_fp8_e8m0_pk8_descriptor_ref(
             source_format, result_element_type, &descriptor_ref) &&
         loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref);
}

static bool loom_amdgpu_direct_fp8_e8m0_pk8_storage_matches(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source, loom_scalar_type_t source_element_type,
    uint32_t source_lane_count) {
  if (source_lane_count == 0 || (source_lane_count & 7u) != 0) {
    return false;
  }

  uint32_t storage_lane_offset = 0;
  uint32_t storage_lane_stride = 1;
  uint32_t storage_lane_count = source_lane_count;
  uint32_t storage_register_count = 0;
  if (!loom_amdgpu_type_is_extf_packed_float_vector(
          loom_module_value_type(module, source), source_element_type,
          &storage_lane_count, &storage_register_count)) {
    return false;
  }

  loom_value_fact_static_lane_origin_t lane_origin = {0};
  if (fact_table && loom_value_fact_table_query_static_lane_origin(
                        fact_table, module, source, &lane_origin)) {
    uint32_t origin_lane_count = 0;
    uint32_t origin_register_count = 0;
    const loom_type_t origin_source_type =
        loom_module_value_type(module, lane_origin.source_value_id);
    if (loom_amdgpu_type_is_extf_packed_float_vector(
            origin_source_type, source_element_type, &origin_lane_count,
            &origin_register_count)) {
      storage_lane_offset = lane_origin.source_lane_offset;
      storage_lane_stride = lane_origin.source_lane_stride;
      storage_lane_count = origin_lane_count;
      storage_register_count = origin_register_count;
    }
  }

  if (storage_lane_stride != 1u) {
    return false;
  }
  for (uint32_t lane_index = 0; lane_index < source_lane_count;
       lane_index += 8u) {
    const uint64_t storage_lane =
        (uint64_t)storage_lane_offset + (uint64_t)lane_index;
    if ((storage_lane & 3u) != 0 || storage_lane + 7u >= storage_lane_count) {
      return false;
    }
    const uint64_t source_register_index = storage_lane / 4u;
    if (source_register_index + 1u >= storage_register_count ||
        source_register_index > UINT32_MAX) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_vector_decode_scale_element_type(
    loom_value_fact_numeric_format_flags_t scale_format,
    loom_scalar_type_t* out_element_type) {
  switch (scale_format) {
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F32:
      *out_element_type = LOOM_SCALAR_TYPE_F32;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0:
      *out_element_type = LOOM_SCALAR_TYPE_I32;
      return true;
    default:
      return false;
  }
}

bool loom_amdgpu_vector_decode_scale_source(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_value_fact_numeric_format_flags_t scale_format,
    loom_value_id_t* out_scale_source) {
  *out_scale_source = LOOM_VALUE_ID_INVALID;
  loom_scalar_type_t scale_element_type = LOOM_SCALAR_TYPE_NONE;
  if (!loom_amdgpu_vector_decode_scale_element_type(scale_format,
                                                    &scale_element_type)) {
    return false;
  }
  loom_encoding_auxiliary_view_t auxiliary_view = {0};
  iree_string_view_t unknown_key = iree_string_view_empty();
  if (!loom_encoding_auxiliary_view_resolve(
          module, loom_vector_decode_auxiliary(source_op),
          loom_vector_decode_auxiliary_names(source_op), &auxiliary_view,
          &unknown_key)) {
    return false;
  }
  if (auxiliary_view.present_keys !=
      loom_encoding_auxiliary_key_flag(LOOM_ENCODING_AUXILIARY_KEY_SCALE)) {
    return false;
  }
  const loom_value_id_t scale_source =
      auxiliary_view.values[LOOM_ENCODING_AUXILIARY_KEY_SCALE];
  if (scale_source == LOOM_VALUE_ID_INVALID ||
      scale_source >= module->values.count ||
      loom_vector_static_rank1_lane_count(
          loom_module_value_type(module, scale_source), scale_element_type,
          1) != 1) {
    return false;
  }
  *out_scale_source = scale_source;
  return true;
}

static loom_value_id_t loom_amdgpu_vector_decode_materialized_scale_source(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t scale_source) {
  loom_value_id_t scalar_source = LOOM_VALUE_ID_INVALID;
  if (loom_value_fact_table_query_uniform_element_origin(
          fact_table, module, scale_source, &scalar_source)) {
    return scalar_source;
  }
  return scale_source;
}

iree_status_t loom_amdgpu_lookup_vector_scale_source(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t* out_low_scale) {
  IREE_ASSERT_NE(plan->scale_source, LOOM_VALUE_ID_INVALID);
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->scale_source, out_low_scale));
  return loom_amdgpu_materialize_full_low_vgpr_b32(
      context, source_op, *out_low_scale, out_low_scale);
}

bool loom_amdgpu_vector_decode_can_lower_as_fp8_conversion(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_op_t* source_op) {
  if (!loom_vector_decode_isa(source_op) || fact_table == NULL) {
    return false;
  }

  const loom_value_id_t source = loom_vector_decode_payload(source_op);
  const loom_value_id_t result = loom_vector_decode_result(source_op);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type) ||
      !loom_type_shape_equals(source_type, result_type)) {
    return false;
  }

  uint32_t source_lane_count = 0;
  uint32_t source_register_count = 0;
  const loom_scalar_type_t source_element_type =
      loom_type_element_type(source_type);
  if (!loom_amdgpu_type_is_extf_packed_float_vector(
          source_type, source_element_type, &source_lane_count,
          &source_register_count)) {
    return false;
  }
  (void)source_register_count;

  const loom_scalar_type_t result_element_type =
      loom_type_element_type(result_type);
  const bool has_auxiliary =
      loom_vector_decode_auxiliary(source_op).count != 0 ||
      loom_vector_decode_auxiliary_names(source_op).count != 0;
  if (result_element_type == LOOM_SCALAR_TYPE_F32) {
    if (loom_amdgpu_vector_f32_register_count(result_type) !=
        source_lane_count) {
      return false;
    }
  } else if (loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_16BIT_FLOAT,
                                           result_element_type)) {
    uint32_t result_lane_count = 0;
    uint32_t result_register_count = 0;
    if (!loom_amdgpu_type_is_16bit_float_packed_vector(
            result_type, result_element_type, &result_lane_count,
            &result_register_count) ||
        result_lane_count != source_lane_count) {
      return false;
    }
    (void)result_register_count;
  } else {
    return false;
  }

  loom_value_fact_encoding_summary_t summary = {0};
  if (!loom_value_facts_query_encoding_summary(
          &fact_table->context,
          loom_value_fact_table_lookup(fact_table,
                                       loom_vector_decode_schema(source_op)),
          &summary)) {
    return false;
  }
  if (!has_auxiliary) {
    return loom_amdgpu_fp8_encoded_operand_schema_matches(
        summary.storage_schema.encoded_operand, source_element_type,
        source_lane_count,
        LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_UNSCALED);
  }

  loom_value_id_t scale_source = LOOM_VALUE_ID_INVALID;
  switch (summary.storage_schema.encoded_operand.scale_format) {
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F32:
      return loom_amdgpu_vector_decode_scale_source(
                 module, source_op, LOOM_VALUE_FACT_NUMERIC_FORMAT_F32,
                 &scale_source) &&
             loom_amdgpu_fp8_encoded_operand_schema_matches(
                 summary.storage_schema.encoded_operand, source_element_type,
                 source_lane_count,
                 LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_SCALE_F32);
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0: {
      const loom_value_fact_numeric_format_flags_t descriptor_source_format =
          loom_amdgpu_fp8_descriptor_source_format(
              summary.storage_schema.encoded_operand.element_format,
              loom_value_fact_table_lookup(
                  fact_table, loom_op_const_results(source_op)[0]));
      return loom_amdgpu_vector_decode_scale_source(
                 module, source_op, LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0,
                 &scale_source) &&
             loom_amdgpu_fp8_encoded_operand_schema_matches(
                 summary.storage_schema.encoded_operand, source_element_type,
                 source_lane_count,
                 LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_SCALE_E8M0) &&
             loom_amdgpu_direct_fp8_e8m0_pk8_descriptor_available(
                 descriptor_set, descriptor_source_format,
                 result_element_type) &&
             loom_amdgpu_direct_fp8_e8m0_pk8_storage_matches(
                 module, fact_table, source, source_element_type,
                 source_lane_count);
    }
    default:
      return false;
  }
}

static bool loom_amdgpu_select_vector_encode_fp8_plan(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    loom_amdgpu_fp8_encode_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_fp8_encode_plan_t){0};
  if (!loom_vector_encode_isa(source_op) || fact_table == NULL ||
      loom_vector_encode_auxiliary(source_op).count != 0 ||
      loom_vector_encode_auxiliary_names(source_op).count != 0) {
    return false;
  }

  const loom_value_id_t source = loom_vector_encode_source(source_op);
  const loom_value_id_t result = loom_vector_encode_result(source_op);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type) ||
      !loom_type_shape_equals(source_type, result_type)) {
    return false;
  }

  const loom_scalar_type_t source_element_type =
      loom_type_element_type(source_type);
  uint32_t source_lane_count = 0;
  uint32_t source_register_count = 0;
  if (source_element_type == LOOM_SCALAR_TYPE_F32) {
    source_lane_count = loom_amdgpu_vector_f32_register_count(source_type);
    source_register_count = source_lane_count;
  } else if (!loom_amdgpu_type_is_16bit_float_packed_vector(
                 source_type, source_element_type, &source_lane_count,
                 &source_register_count)) {
    return false;
  }
  if (source_lane_count == 0 || source_register_count == 0) {
    return false;
  }

  const loom_scalar_type_t result_element_type =
      loom_type_element_type(result_type);
  uint32_t result_lane_count = 0;
  uint32_t result_register_count = 0;
  if (!loom_amdgpu_type_is_extf_packed_float_vector(
          result_type, result_element_type, &result_lane_count,
          &result_register_count) ||
      result_lane_count != source_lane_count || result_register_count == 0) {
    return false;
  }

  loom_value_fact_encoding_summary_t summary = {0};
  if (!loom_value_facts_query_encoding_summary(
          &fact_table->context,
          loom_value_fact_table_lookup(fact_table,
                                       loom_vector_encode_schema(source_op)),
          &summary)) {
    return false;
  }
  const loom_value_fact_encoded_operand_schema_t schema =
      summary.storage_schema.encoded_operand;
  const loom_value_fact_rounding_policy_flags_t unsupported_policies =
      schema.rounding_policy & ~LOOM_VALUE_FACT_ROUNDING_POLICY_FINITE_ONLY;
  if (unsupported_policies != LOOM_VALUE_FACT_ROUNDING_POLICY_NONE ||
      !loom_amdgpu_fp8_encoded_operand_schema_matches(
          schema, result_element_type, result_lane_count,
          LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_UNSCALED)) {
    return false;
  }

  return loom_amdgpu_select_fp8_encode_plan(descriptor_set, source_element_type,
                                            result_element_type,
                                            schema.element_format, out_plan);
}

iree_status_t loom_amdgpu_low_legality_verify_vector_decode(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  *out_handled = false;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  if (loom_amdgpu_vector_decode_can_lower_as_fp4_conversion(
          loom_target_low_legality_module(context),
          loom_target_low_legality_fact_table(context),
          loom_target_low_legality_descriptor_set(context), op) ||
      loom_amdgpu_vector_decode_can_lower_as_fp8_conversion(
          loom_target_low_legality_module(context),
          loom_target_low_legality_fact_table(context),
          loom_target_low_legality_descriptor_set(context), op)) {
    *out_handled = true;
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_vector_encode(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  *out_handled = false;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  loom_amdgpu_fp8_encode_plan_t plan = {0};
  *out_handled = loom_amdgpu_select_vector_encode_fp8_plan(
      loom_target_low_legality_module(context),
      loom_target_low_legality_fact_table(context),
      loom_target_low_legality_descriptor_set(context), op, &plan);
  return iree_ok_status();
}

#define LOOM_AMDGPU_VECTOR_OP_INDEX(op_kind) ((uint8_t)((op_kind) & 0xFFu))
#define LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_ROW(op, kind_) \
  [LOOM_AMDGPU_VECTOR_OP_INDEX(LOOM_OP_VECTOR_##op)] =                \
      LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_##kind_

static const loom_amdgpu_vector_16bit_float_conversion_kind_t
    kAmdgpuVector16BitFloatConversionKindByVectorOp[LOOM_OP_VECTOR_COUNT_] = {
        LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_ROW(EXTF, EXTF),
        LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_ROW(FPTRUNC, FPTRUNC),
        LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_ROW(DECODE, DECODE),
        LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_ROW(ENCODE, ENCODE),
};

#undef LOOM_AMDGPU_VECTOR_OP_INDEX
#undef LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_ROW

static loom_amdgpu_vector_16bit_float_conversion_kind_t
loom_amdgpu_vector_16bit_float_conversion_kind(loom_op_kind_t op_kind) {
  if (loom_op_dialect_id(op_kind) != LOOM_DIALECT_VECTOR) {
    return LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_NONE;
  }
  const uint8_t op_index = loom_op_dialect_index(op_kind);
  if (op_index >=
      IREE_ARRAYSIZE(kAmdgpuVector16BitFloatConversionKindByVectorOp)) {
    return LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_NONE;
  }
  return kAmdgpuVector16BitFloatConversionKindByVectorOp[op_index];
}

static bool loom_amdgpu_scalar_type_is_fp8(loom_scalar_type_t type) {
  return type == LOOM_SCALAR_TYPE_F8E4M3 || type == LOOM_SCALAR_TYPE_F8E5M2;
}

static iree_status_t
loom_amdgpu_vector_16bit_float_conversion_plan_from_accepted_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* fp8_encode,
    loom_amdgpu_vector_16bit_float_conversion_plan_t* out_plan) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  *out_plan = (loom_amdgpu_vector_16bit_float_conversion_plan_t){0};

  const loom_value_id_t source = loom_op_const_operands(source_op)[0];
  const loom_value_id_t result = loom_op_const_results(source_op)[0];
  const loom_amdgpu_vector_16bit_float_conversion_kind_t kind =
      loom_amdgpu_vector_16bit_float_conversion_kind(source_op->kind);
  IREE_ASSERT_NE(kind, LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_NONE);

  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);
  loom_scalar_type_t source_element_type = loom_type_element_type(source_type);
  loom_value_fact_numeric_format_flags_t source_format =
      loom_amdgpu_vector_storage_numeric_format(
          fact_table, source,
          loom_numeric_format_from_scalar_type(source_element_type));
  const loom_scalar_type_t result_element_type =
      loom_type_element_type(result_type);
  uint32_t source_lane_count = 0;
  uint32_t source_register_count = 0;
  loom_value_id_t storage_source = source;
  loom_value_id_t content_fact_source = source;
  loom_value_id_t scale_source = LOOM_VALUE_ID_INVALID;
  loom_value_fact_numeric_format_flags_t scale_format =
      LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE;
  uint32_t scale_group_element_count = 0;
  uint32_t storage_lane_offset = 0;
  uint32_t storage_lane_stride = 1;
  uint32_t storage_lane_count = 0;
  uint32_t storage_register_count = 0;
  uint32_t result_lane_count = 0;
  uint32_t result_register_count = 0;
  if (kind == LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_EXTF ||
      kind == LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_DECODE) {
    if (kind == LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_DECODE) {
      loom_value_fact_encoding_summary_t summary = {0};
      const bool has_summary = loom_value_facts_query_encoding_summary(
          &fact_table->context,
          loom_value_fact_table_lookup(fact_table,
                                       loom_vector_decode_schema(source_op)),
          &summary);
      IREE_ASSERT(has_summary);
      source_format = summary.storage_schema.encoded_operand.element_format;
      scale_format = summary.storage_schema.encoded_operand.scale_format;
      scale_group_element_count =
          summary.storage_schema.encoded_operand.scale_group.element_count;
      if (loom_amdgpu_vector_decode_scale_source(module, source_op,
                                                 scale_format, &scale_source)) {
        scale_source = loom_amdgpu_vector_decode_materialized_scale_source(
            module, fact_table, scale_source);
      } else {
        scale_format = LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE;
      }
    }
    IREE_ASSERT_TRUE(
        result_element_type == LOOM_SCALAR_TYPE_F32 ||
        loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_16BIT_FLOAT,
                                      result_element_type));
    const bool source_matches = loom_amdgpu_type_is_extf_packed_float_vector(
        source_type, source_element_type, &source_lane_count,
        &source_register_count);
    IREE_ASSERT(source_matches);
    storage_lane_count = source_lane_count;
    storage_register_count = source_register_count;
    loom_value_fact_static_lane_origin_t lane_origin = {0};
    if (loom_value_fact_table_query_static_lane_origin(fact_table, module,
                                                       source, &lane_origin)) {
      uint32_t origin_lane_count = 0;
      uint32_t origin_register_count = 0;
      const loom_type_t origin_source_type =
          loom_module_value_type(module, lane_origin.source_value_id);
      if (loom_amdgpu_type_is_extf_packed_float_vector(
              origin_source_type, source_element_type, &origin_lane_count,
              &origin_register_count)) {
        storage_source = lane_origin.source_value_id;
        storage_lane_offset = lane_origin.source_lane_offset;
        storage_lane_stride = lane_origin.source_lane_stride;
        storage_lane_count = origin_lane_count;
        storage_register_count = origin_register_count;
      }
    }
    content_fact_source =
        kind == LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_DECODE
            ? result
            : storage_source;
    if (result_element_type == LOOM_SCALAR_TYPE_F32) {
      result_lane_count = loom_amdgpu_vector_f32_register_count(result_type);
      result_register_count = result_lane_count;
    } else {
      const bool result_matches = loom_amdgpu_type_is_16bit_float_packed_vector(
          result_type, result_element_type, &result_lane_count,
          &result_register_count);
      IREE_ASSERT(result_matches);
    }
  } else {
    const bool result_is_fp8 = result_element_type == LOOM_SCALAR_TYPE_F8E4M3 ||
                               result_element_type == LOOM_SCALAR_TYPE_F8E5M2;
    if (result_is_fp8) {
      IREE_ASSERT(fp8_encode != NULL);
      if (source_element_type == LOOM_SCALAR_TYPE_F32) {
        source_lane_count = loom_amdgpu_vector_f32_register_count(source_type);
        source_register_count = source_lane_count;
      } else {
        IREE_ASSERT(source_element_type == LOOM_SCALAR_TYPE_F16 ||
                    source_element_type == LOOM_SCALAR_TYPE_BF16);
        const bool source_matches =
            loom_amdgpu_type_is_16bit_float_packed_vector(
                source_type, source_element_type, &source_lane_count,
                &source_register_count);
        IREE_ASSERT(source_matches);
      }
      const bool result_matches = loom_amdgpu_type_is_extf_packed_float_vector(
          result_type, result_element_type, &result_lane_count,
          &result_register_count);
      IREE_ASSERT(result_matches);
      storage_lane_count = source_lane_count;
      storage_register_count = source_register_count;
    } else {
      IREE_ASSERT_EQ(source_element_type, LOOM_SCALAR_TYPE_F32);
      source_lane_count = loom_amdgpu_vector_f32_register_count(source_type);
      source_register_count = source_lane_count;
      storage_lane_count = source_lane_count;
      const bool result_matches = loom_amdgpu_type_is_16bit_float_packed_vector(
          result_type, result_element_type, &result_lane_count,
          &result_register_count);
      IREE_ASSERT(result_matches);
      storage_register_count = source_register_count;
    }

    loom_value_fact_static_lane_origin_t lane_origin = {0};
    if (!result_is_fp8 && loom_value_fact_table_query_static_lane_origin(
                              fact_table, module, source, &lane_origin)) {
      uint32_t origin_lane_count = 0;
      uint32_t origin_register_count = 0;
      const loom_type_t origin_source_type =
          loom_module_value_type(module, lane_origin.source_value_id);
      const loom_scalar_type_t origin_element_type =
          loom_type_element_type(origin_source_type);
      if (lane_origin.source_lane_offset == 0 &&
          lane_origin.source_lane_stride == 1 &&
          origin_element_type == result_element_type &&
          loom_amdgpu_type_is_16bit_float_packed_vector(
              origin_source_type, origin_element_type, &origin_lane_count,
              &origin_register_count) &&
          origin_lane_count == source_lane_count &&
          origin_register_count == result_register_count) {
        source_element_type = origin_element_type;
        source_format =
            loom_numeric_format_from_scalar_type(origin_element_type);
        storage_source = lane_origin.source_value_id;
        content_fact_source = storage_source;
        storage_lane_offset = lane_origin.source_lane_offset;
        storage_lane_stride = lane_origin.source_lane_stride;
        storage_lane_count = origin_lane_count;
        storage_register_count = origin_register_count;
      }
    }

    loom_value_fact_uniform_scale_origin_t scale_origin = {0};
    if (!result_is_fp8 && source_element_type == LOOM_SCALAR_TYPE_F32 &&
        loom_value_fact_table_query_uniform_scale_origin(
            fact_table, module, source, &scale_origin)) {
      lane_origin = (loom_value_fact_static_lane_origin_t){0};
      if (loom_value_fact_table_query_static_lane_origin(
              fact_table, module, scale_origin.source_value_id, &lane_origin)) {
        uint32_t origin_lane_count = 0;
        uint32_t origin_register_count = 0;
        const loom_type_t origin_source_type =
            loom_module_value_type(module, lane_origin.source_value_id);
        const loom_scalar_type_t origin_element_type =
            loom_type_element_type(origin_source_type);
        const uint64_t last_storage_lane =
            source_lane_count == 0
                ? UINT64_MAX
                : (uint64_t)lane_origin.source_lane_offset +
                      (uint64_t)(source_lane_count - 1u) *
                          (uint64_t)lane_origin.source_lane_stride;
        const bool is_origin_fp8 =
            origin_element_type == LOOM_SCALAR_TYPE_F8E4M3 ||
            origin_element_type == LOOM_SCALAR_TYPE_F8E5M2;
        const loom_low_lower_resolved_descriptor_t* scale_descriptor = NULL;
        if (is_origin_fp8) {
          const loom_value_fact_numeric_format_flags_t origin_format =
              loom_amdgpu_vector_storage_numeric_format(
                  fact_table, lane_origin.source_value_id,
                  loom_numeric_format_from_scalar_type(origin_element_type));
          IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_scalef32_descriptor(
              context, origin_format, result_element_type, &scale_descriptor));
        }
        if (scale_descriptor != NULL &&
            loom_amdgpu_type_is_extf_packed_float_vector(
                origin_source_type, origin_element_type, &origin_lane_count,
                &origin_register_count) &&
            last_storage_lane < origin_lane_count) {
          source_element_type = origin_element_type;
          source_format =
              loom_numeric_format_from_scalar_type(origin_element_type);
          storage_source = lane_origin.source_value_id;
          content_fact_source = storage_source;
          scale_source = scale_origin.scale_value_id;
          scale_format = LOOM_VALUE_FACT_NUMERIC_FORMAT_F32;
          scale_group_element_count = source_lane_count;
          storage_lane_offset = lane_origin.source_lane_offset;
          storage_lane_stride = lane_origin.source_lane_stride;
          storage_lane_count = origin_lane_count;
          storage_register_count = origin_register_count;
        }
      }
    }
  }
  IREE_ASSERT_NE(source_lane_count, 0u);
  IREE_ASSERT_EQ(source_lane_count, result_lane_count);
  IREE_ASSERT_NE(source_register_count, 0u);
  IREE_ASSERT_NE(storage_lane_count, 0u);
  IREE_ASSERT_NE(storage_register_count, 0u);
  IREE_ASSERT_NE(storage_lane_stride, 0u);
  IREE_ASSERT_NE(result_register_count, 0u);
  IREE_ASSERT_LE(scale_group_element_count, UINT8_MAX);
  IREE_ASSERT_LE(source_lane_count, UINT8_MAX);
  IREE_ASSERT_LE(source_register_count, UINT8_MAX);
  IREE_ASSERT_LE(storage_lane_offset, UINT8_MAX);
  IREE_ASSERT_LE(storage_lane_stride, UINT8_MAX);
  IREE_ASSERT_LE(storage_lane_count, UINT8_MAX);
  IREE_ASSERT_LE(storage_register_count, UINT8_MAX);
  IREE_ASSERT_LE(result_register_count, UINT8_MAX);

  if (source_element_type == LOOM_SCALAR_TYPE_F8E4M3 ||
      source_element_type == LOOM_SCALAR_TYPE_F8E5M2) {
    source_format = loom_amdgpu_vector_storage_numeric_format(
        fact_table, storage_source, source_format);
  }
  const loom_value_facts_t content_facts =
      loom_value_fact_table_lookup(fact_table, content_fact_source);
  const loom_value_fact_numeric_format_flags_t descriptor_source_format =
      loom_amdgpu_fp8_descriptor_source_format(source_format, content_facts);

  *out_plan = (loom_amdgpu_vector_16bit_float_conversion_plan_t){
      .source = source,
      .result = result,
      .storage_source = storage_source,
      .content_fact_source = content_fact_source,
      .scale_source = scale_source,
      .scale_format = scale_format,
      .source_format = source_format,
      .descriptor_source_format = descriptor_source_format,
      .kind = kind,
      .source_element_type = source_element_type,
      .result_element_type = result_element_type,
      .strategy_kind =
          fp8_encode != NULL
              ? LOOM_AMDGPU_VECTOR_FLOAT_CONVERSION_STRATEGY_FP8_ENCODE
              : LOOM_AMDGPU_VECTOR_FLOAT_CONVERSION_STRATEGY_STANDARD,
      .scale_group_element_count = (uint8_t)scale_group_element_count,
      .lane_count = (uint8_t)source_lane_count,
      .source_register_count = (uint8_t)source_register_count,
      .storage_lane_offset = (uint8_t)storage_lane_offset,
      .storage_lane_stride = (uint8_t)storage_lane_stride,
      .storage_lane_count = (uint8_t)storage_lane_count,
      .storage_register_count = (uint8_t)storage_register_count,
      .result_register_count = (uint8_t)result_register_count,
      .strategy =
          {
              .fp8_encode = fp8_encode != NULL
                                ? *fp8_encode
                                : (loom_amdgpu_fp8_encode_plan_t){0},
          },
  };
  if (fp8_encode == NULL &&
      loom_amdgpu_scalar_type_is_fp8(source_element_type)) {
    loom_amdgpu_select_vector_fp8_decode_plan(context, out_plan);
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_vector_16bit_float_conversion_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_16bit_float_conversion_plan_t* out_plan,
    bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_16bit_float_conversion_plan_t){0};
  const loom_amdgpu_vector_16bit_float_conversion_kind_t kind =
      loom_amdgpu_vector_16bit_float_conversion_kind(source_op->kind);
  loom_amdgpu_fp8_encode_plan_t fp8_encode = {0};
  if (kind == LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_DECODE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_select_fp4_decode_plan(
        context, source_op, out_plan, out_selected));
    if (!*out_selected) {
      *out_selected = loom_amdgpu_vector_decode_can_lower_as_fp8_conversion(
          loom_low_lower_context_module(context),
          loom_low_lower_context_fact_table(context),
          loom_low_lower_context_descriptor_set(context), source_op);
    }
  } else if (kind == LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_ENCODE) {
    *out_selected = loom_amdgpu_select_vector_encode_fp8_plan(
        loom_low_lower_context_module(context),
        loom_low_lower_context_fact_table(context),
        loom_low_lower_context_descriptor_set(context), source_op, &fp8_encode);
  } else if (kind == LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_FPTRUNC) {
    const loom_module_t* module = loom_low_lower_context_module(context);
    const loom_value_id_t source = loom_op_const_operands(source_op)[0];
    const loom_value_id_t result = loom_op_const_results(source_op)[0];
    const loom_scalar_type_t source_element_type =
        loom_type_element_type(loom_module_value_type(module, source));
    const loom_scalar_type_t result_element_type =
        loom_type_element_type(loom_module_value_type(module, result));
    const bool result_is_fp8 = result_element_type == LOOM_SCALAR_TYPE_F8E4M3 ||
                               result_element_type == LOOM_SCALAR_TYPE_F8E5M2;
    if (result_is_fp8) {
      *out_selected = loom_amdgpu_select_fp8_encode_plan(
          loom_low_lower_context_descriptor_set(context), source_element_type,
          result_element_type,
          loom_numeric_format_from_scalar_type(result_element_type),
          &fp8_encode);
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_select_arithmetic_contract(
          context, source_op, out_selected));
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_select_arithmetic_contract(
        context, source_op, out_selected));
  }
  if (*out_selected &&
      out_plan->strategy_kind !=
          LOOM_AMDGPU_VECTOR_FLOAT_CONVERSION_STRATEGY_FP4_DECODE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_vector_16bit_float_conversion_plan_from_accepted_op(
            context, source_op,
            fp8_encode.kind == LOOM_AMDGPU_FP8_ENCODE_KIND_NONE ? NULL
                                                                : &fp8_encode,
            out_plan));
  }
  return iree_ok_status();
}

iree_string_view_t loom_amdgpu_vector_16bit_float_conversion_plan_key(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  if (plan->strategy_kind ==
      LOOM_AMDGPU_VECTOR_FLOAT_CONVERSION_STRATEGY_FP4_DECODE) {
    return loom_amdgpu_fp4_decode_plan_key(plan);
  }
  if (plan->strategy_kind ==
      LOOM_AMDGPU_VECTOR_FLOAT_CONVERSION_STRATEGY_FP8_ENCODE) {
    return loom_amdgpu_fp8_encode_plan_key(&plan->strategy.fp8_encode,
                                           plan->source_element_type);
  }
  if (loom_amdgpu_scalar_type_is_fp8(plan->source_element_type)) {
    return loom_amdgpu_vector_fp8_conversion_plan_key(plan);
  }
  if (plan->source_element_type == LOOM_SCALAR_TYPE_F32) {
    switch (plan->result_element_type) {
      case LOOM_SCALAR_TYPE_BF16:
        return loom_amdgpu_descriptor_set_has_ref(
                   loom_low_lower_context_descriptor_set(context),
                   LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF16_F32)
                   ? IREE_SV(
                         "amdgpu.vector_16bit_float_conversion.strategy."
                         "f32_to_packed_bf16_native")
                   : IREE_SV(
                         "amdgpu.vector_16bit_float_conversion.strategy."
                         "f32_to_packed_bf16_integer_pack");
      case LOOM_SCALAR_TYPE_F16:
        return IREE_SV(
            "amdgpu.vector_16bit_float_conversion.strategy.f32_to_packed_f16");
      default:
        return iree_string_view_empty();
    }
  }
  if (plan->source_element_type == LOOM_SCALAR_TYPE_F16 &&
      plan->result_element_type == LOOM_SCALAR_TYPE_F32) {
    return IREE_SV("amdgpu.vector_16bit_float_conversion.strategy.f16_to_f32");
  }
  if (plan->source_element_type == LOOM_SCALAR_TYPE_BF16 &&
      plan->result_element_type == LOOM_SCALAR_TYPE_F32) {
    return IREE_SV("amdgpu.vector_16bit_float_conversion.strategy.bf16_to_f32");
  }
  if (plan->kind == LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_FPTRUNC &&
      plan->source_element_type == plan->result_element_type) {
    switch (plan->result_element_type) {
      case LOOM_SCALAR_TYPE_BF16:
        return IREE_SV(
            "amdgpu.vector_16bit_float_conversion.strategy."
            "packed_bf16_origin_copy");
      case LOOM_SCALAR_TYPE_F16:
        return IREE_SV(
            "amdgpu.vector_16bit_float_conversion.strategy."
            "packed_f16_origin_copy");
      default:
        return iree_string_view_empty();
    }
  }
  return iree_string_view_empty();
}

static iree_status_t loom_amdgpu_lower_vector_16bit_float_extf(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  if (plan->strategy_kind ==
      LOOM_AMDGPU_VECTOR_FLOAT_CONVERSION_STRATEGY_FP4_DECODE) {
    return loom_amdgpu_lower_vector_fp4_decode(context, source_op, plan);
  }
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->storage_source, &low_source));

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t source_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_source);
  if (loom_type_kind(source_lane_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_type(context, &source_lane_type));
  }
  loom_type_t result_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &result_lane_type));

  if (loom_amdgpu_scalar_type_is_fp8(plan->source_element_type)) {
    return loom_amdgpu_lower_vector_fp8_conversion(context, source_op, plan,
                                                   low_source, source_lane_type,
                                                   result_lane_type);
  }

  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->lane_count; ++i) {
    const uint64_t storage_lane =
        (uint64_t)plan->storage_lane_offset +
        (uint64_t)i * (uint64_t)plan->storage_lane_stride;
    IREE_ASSERT_LE(storage_lane, UINT32_MAX);
    if (plan->source_element_type == LOOM_SCALAR_TYPE_BF16) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_bf16_range_lane_as_f32_bits(
          context, source_op, low_source, plan->storage_register_count,
          (uint32_t)storage_lane, source_lane_type, result_lane_type,
          &lanes[i]));
    } else {
      loom_value_id_t half_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_f16_lane_as_low_bits(
          context, source_op, low_source, plan->storage_register_count,
          (uint32_t)storage_lane, source_lane_type, &half_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_unary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_F16,
          half_lane, result_lane_type, &lanes[i]));
    }
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             lanes, plan->lane_count);
}

static iree_status_t loom_amdgpu_lower_vector_f32_to_packed_f16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t lane_type) {
  loom_value_id_t packed_registers[LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES];
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
    const uint32_t lane_base = register_index * 2u;
    for (uint32_t register_lane = 0; register_lane < 2u; ++register_lane) {
      const uint32_t lane_index = lane_base + register_lane;
      if (lane_index >= plan->lane_count) {
        break;
      }
      loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
          context, source_op, low_source, plan->source_register_count,
          lane_index, source_lane_type, &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_pack_f32_lane_to_f16_register(
          context, source_op, source_lane, register_lane, lane_type, &packed));
    }
    packed_registers[register_index] = packed;
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             packed_registers,
                                             plan->result_register_count);
}

static iree_status_t loom_amdgpu_lower_vector_f32_to_packed_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t lane_type) {
  const loom_amdgpu_float16_pack_descriptors_t* descriptors = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_get_float16_pack_descriptors(context, &descriptors));

  loom_value_id_t packed_registers[LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES];
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    const uint32_t lane_base = register_index * 2u;
    loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, low_source, plan->source_register_count, lane_base,
        source_lane_type, &source_lane));
    if (lane_base + 1u < plan->lane_count) {
      loom_value_id_t high_source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
          context, source_op, low_source, plan->source_register_count,
          lane_base + 1u, source_lane_type, &high_source_lane));
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
              context, source_op, descriptors, source_lane, high_source_lane,
              lane_type, &packed_registers[register_index]));
    } else if (iree_any_bit_set(
                   descriptors->flags,
                   LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_NATIVE_BF16)) {
      loom_value_id_t zero_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
          lane_type, &zero_lane));
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
              context, source_op, descriptors, source_lane, zero_lane,
              lane_type, &packed_registers[register_index]));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_f32_to_bf16_lane_with_descriptors(
          context, source_op, descriptors, source_lane, lane_type,
          &packed_registers[register_index]));
    }
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             packed_registers,
                                             plan->result_register_count);
}

static bool loom_amdgpu_vector_16bit_float_fptrunc_has_storage_origin(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  return plan->kind == LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_FPTRUNC &&
         plan->source_element_type == plan->result_element_type &&
         loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_16BIT_FLOAT,
                                       plan->result_element_type);
}

static iree_status_t loom_amdgpu_materialize_fp8_encode_f32_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t lane_type, uint32_t lane_index, loom_value_id_t* out_lane) {
  if (plan->source_element_type == LOOM_SCALAR_TYPE_F32) {
    return loom_amdgpu_extract_low_register_unit(
        context, source_op, low_source, plan->source_register_count, lane_index,
        source_lane_type, out_lane);
  }

  IREE_ASSERT(plan->source_element_type == LOOM_SCALAR_TYPE_F16 ||
              plan->source_element_type == LOOM_SCALAR_TYPE_BF16);
  if (plan->source_element_type == LOOM_SCALAR_TYPE_BF16) {
    return loom_amdgpu_extract_bf16_range_lane_as_f32_bits(
        context, source_op, low_source, plan->source_register_count, lane_index,
        source_lane_type, lane_type, out_lane);
  }
  loom_value_id_t f16_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_f16_lane_as_low_bits(
      context, source_op, low_source, plan->source_register_count, lane_index,
      source_lane_type, &f16_lane));
  return loom_amdgpu_emit_vgpr_unary(context, source_op,
                                     LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_F16,
                                     f16_lane, lane_type, out_lane);
}

static iree_status_t loom_amdgpu_materialize_fp8_encode_f16_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t lane_type, uint32_t lane_base, loom_value_id_t* out_pair) {
  IREE_ASSERT_EQ(plan->source_element_type, LOOM_SCALAR_TYPE_F16);
  if (lane_base + 1u < plan->lane_count) {
    return loom_amdgpu_extract_low_register_unit(
        context, source_op, low_source, plan->source_register_count,
        lane_base / 2u, source_lane_type, out_pair);
  }

  loom_value_id_t low_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_f16_lane_as_low_bits(
      context, source_op, low_source, plan->source_register_count, lane_base,
      source_lane_type, &low_lane));
  return loom_amdgpu_emit_fp8_encode_duplicate_f16_lane(
      context, source_op, low_lane, lane_type, out_pair);
}

static iree_status_t loom_amdgpu_materialize_fp8_encode_software_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t lane_type, uint32_t lane_index, loom_value_id_t* out_lane) {
  if (plan->strategy.fp8_encode.kind ==
      LOOM_AMDGPU_FP8_ENCODE_KIND_F16_SOFTWARE_E5M2) {
    return loom_amdgpu_extract_f16_lane_as_low_bits(
        context, source_op, low_source, plan->source_register_count, lane_index,
        source_lane_type, out_lane);
  }
  return loom_amdgpu_materialize_fp8_encode_f32_lane(
      context, source_op, plan, low_source, source_lane_type, lane_type,
      lane_index, out_lane);
}

static iree_status_t loom_amdgpu_emit_fp8_encode_software_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* emission_state,
    loom_value_id_t source, loom_value_id_t* out_encoded) {
  if (plan->kind == LOOM_AMDGPU_FP8_ENCODE_KIND_F16_SOFTWARE_E5M2) {
    return loom_amdgpu_emit_fp8_encode_software_f16_e5m2_lane(
        context, source_op, plan, emission_state, source, out_encoded);
  }
  return loom_amdgpu_emit_fp8_encode_software_f32_lane(
      context, source_op, plan, emission_state, source, out_encoded);
}

static iree_status_t
loom_amdgpu_lower_vector_fp8_encode_software_literal_permute(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* emission_state,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t lane_type) {
  IREE_ASSERT_EQ(plan->strategy.fp8_encode.packed_i8_permute.kind,
                 LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_LITERAL_SELECTOR);
  IREE_ASSERT_LE(plan->result_register_count,
                 LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS);

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {
      0};
  const bool has_packed_f16_e4m3 =
      loom_amdgpu_fp8_encode_plan_has_packed_f16_e4m3(
          &plan->strategy.fp8_encode);
  const bool has_packed_f16_e5m2 =
      loom_amdgpu_fp8_encode_plan_has_packed_f16_e5m2(
          &plan->strategy.fp8_encode);
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    const uint32_t lane_base = register_index * 4u;
    const uint32_t register_lane_count =
        iree_min(4u, plan->lane_count - lane_base);
    if (register_lane_count >= 3u &&
        (has_packed_f16_e4m3 || has_packed_f16_e5m2)) {
      loom_value_id_t low_source_pair = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_fp8_encode_f16_pair(
          context, source_op, plan, low_source, source_lane_type, lane_type,
          lane_base, &low_source_pair));
      loom_value_id_t high_source_pair = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_fp8_encode_f16_pair(
          context, source_op, plan, low_source, source_lane_type, lane_type,
          lane_base + 2u, &high_source_pair));
      if (has_packed_f16_e4m3) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fp8_encode_software_f16_e4m3_pairs(
                context, source_op, &plan->strategy.fp8_encode, emission_state,
                low_source_pair, high_source_pair,
                &result_registers[register_index]));
      } else {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fp8_encode_software_f16_e5m2_pairs(
                context, source_op, &plan->strategy.fp8_encode, emission_state,
                low_source_pair, high_source_pair,
                &result_registers[register_index]));
      }
      continue;
    }

    loom_value_id_t source_lanes[4] = {
        LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID,
    };
    for (uint32_t register_lane = 0; register_lane < register_lane_count;
         ++register_lane) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_fp8_encode_software_lane(
          context, source_op, plan, low_source, source_lane_type, lane_type,
          lane_base + register_lane, &source_lanes[register_lane]));
    }

    if (register_lane_count >= 3u) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_software_packed_lanes(
          context, source_op, &plan->strategy.fp8_encode, emission_state,
          source_lanes, register_lane_count,
          &result_registers[register_index]));
      continue;
    }

    loom_value_id_t encoded_lanes[4] = {
        LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID,
    };
    for (uint32_t register_lane = 0; register_lane < register_lane_count;
         ++register_lane) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_software_lane(
          context, source_op, &plan->strategy.fp8_encode, emission_state,
          source_lanes[register_lane], &encoded_lanes[register_lane]));
    }
    for (uint32_t register_lane = register_lane_count; register_lane < 4u;
         ++register_lane) {
      encoded_lanes[register_lane] = encoded_lanes[register_lane_count - 1u];
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_pack_i8_lanes_with_permute(
        context, source_op, &plan->strategy.fp8_encode.packed_i8_permute,
        encoded_lanes, IREE_ARRAYSIZE(encoded_lanes), lane_type,
        &result_registers[register_index]));
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             result_registers,
                                             plan->result_register_count);
}

static iree_status_t loom_amdgpu_lower_vector_fp8_encode_software(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* emission_state,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t lane_type) {
  IREE_ASSERT(
      loom_amdgpu_fp8_encode_plan_is_software(&plan->strategy.fp8_encode));
  if (plan->strategy.fp8_encode.packed_i8_permute.kind ==
      LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_LITERAL_SELECTOR) {
    return loom_amdgpu_lower_vector_fp8_encode_software_literal_permute(
        context, source_op, plan, emission_state, low_source, source_lane_type,
        lane_type);
  }

  const uint32_t packed_lane_count = plan->result_register_count * 4u;
  IREE_ASSERT_LE(packed_lane_count, LOOM_AMDGPU_MAX_PACKED_I8_LANES);

  loom_value_id_t encoded_lanes[LOOM_AMDGPU_MAX_PACKED_I8_LANES] = {0};
  for (uint32_t lane = 0; lane < plan->lane_count; ++lane) {
    loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_fp8_encode_software_lane(
        context, source_op, plan, low_source, source_lane_type, lane_type, lane,
        &source_lane));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_software_lane(
        context, source_op, &plan->strategy.fp8_encode, emission_state,
        source_lane, &encoded_lanes[lane]));
  }
  for (uint32_t lane = plan->lane_count; lane < packed_lane_count; ++lane) {
    encoded_lanes[lane] = encoded_lanes[plan->lane_count - 1u];
  }

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {
      0};
  if (plan->strategy.fp8_encode.packed_i8_permute.kind !=
      LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_NONE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_pack_i8_lanes_with_permute(
        context, source_op, &plan->strategy.fp8_encode.packed_i8_permute,
        encoded_lanes, packed_lane_count, lane_type, result_registers));
  } else {
    for (uint32_t register_index = 0;
         register_index < plan->result_register_count; ++register_index) {
      loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
      for (uint32_t register_lane = 0; register_lane < 4u; ++register_lane) {
        const uint32_t lane = register_index * 4u + register_lane;
        IREE_RETURN_IF_ERROR(loom_amdgpu_pack_bits_into_register(
            context, source_op, encoded_lanes[lane], register_lane * 8u,
            lane_type, &packed));
      }
      result_registers[register_index] = packed;
    }
  }
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             result_registers,
                                             plan->result_register_count);
}

static iree_status_t loom_amdgpu_lower_vector_fp8_encode(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t source_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_source);
  if (loom_type_kind(source_lane_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_type(context, &source_lane_type));
  }
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  loom_amdgpu_fp8_encode_emission_state_t emission_state = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_initialize_fp8_encode_emission(
      context, source_op, &plan->strategy.fp8_encode, plan->lane_count,
      lane_type, &emission_state));
  if (loom_amdgpu_fp8_encode_plan_is_software(&plan->strategy.fp8_encode)) {
    return loom_amdgpu_lower_vector_fp8_encode_software(
        context, source_op, plan, &emission_state, low_source, source_lane_type,
        lane_type);
  }

  IREE_ASSERT_LE(plan->result_register_count,
                 LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS);
  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {
      0};
  for (uint32_t result_register_index = 0;
       result_register_index < plan->result_register_count;
       ++result_register_index) {
    const uint32_t lane_base = result_register_index * 4u;
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
    if (loom_amdgpu_fp8_encode_plan_is_fnuz_bridge(
            &plan->strategy.fp8_encode)) {
      loom_value_id_t f32_lanes[4] = {
          LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
          LOOM_VALUE_ID_INVALID};
      const uint32_t register_lane_count =
          iree_min(4u, plan->lane_count - lane_base);
      for (uint32_t register_lane = 0; register_lane < register_lane_count;
           ++register_lane) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_fp8_encode_f32_lane(
            context, source_op, plan, low_source, source_lane_type, lane_type,
            lane_base + register_lane, &f32_lanes[register_lane]));
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_fnuz_f32_lanes(
          context, source_op, &plan->strategy.fp8_encode, &emission_state,
          f32_lanes, register_lane_count, &packed));
    } else if (plan->strategy.fp8_encode.kind ==
               LOOM_AMDGPU_FP8_ENCODE_KIND_F16_PAIR) {
      loom_value_id_t low_pair = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_fp8_encode_f16_pair(
          context, source_op, plan, low_source, source_lane_type, lane_type,
          lane_base, &low_pair));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_low_pair(
          context, source_op, &plan->strategy.fp8_encode, &emission_state,
          low_pair, LOOM_VALUE_ID_INVALID, &packed));
      loom_value_id_t high_pair = low_pair;
      if (lane_base + 2u < plan->lane_count) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_fp8_encode_f16_pair(
            context, source_op, plan, low_source, source_lane_type, lane_type,
            lane_base + 2u, &high_pair));
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_high_pair(
          context, source_op, &plan->strategy.fp8_encode, &emission_state,
          packed, high_pair, LOOM_VALUE_ID_INVALID, &packed));
    } else {
      loom_value_id_t f32_lanes[4] = {
          LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
          LOOM_VALUE_ID_INVALID};
      const uint32_t register_lane_count =
          iree_min(4u, plan->lane_count - lane_base);
      for (uint32_t register_lane = 0; register_lane < register_lane_count;
           ++register_lane) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_fp8_encode_f32_lane(
            context, source_op, plan, low_source, source_lane_type, lane_type,
            lane_base + register_lane, &f32_lanes[register_lane]));
      }
      if (register_lane_count == 1u) {
        f32_lanes[1] = f32_lanes[0];
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_low_pair(
          context, source_op, &plan->strategy.fp8_encode, &emission_state,
          f32_lanes[0], f32_lanes[1], &packed));
      if (register_lane_count < 3u) {
        f32_lanes[2] = f32_lanes[register_lane_count - 1u];
      }
      if (register_lane_count < 4u) {
        f32_lanes[3] = f32_lanes[register_lane_count - 1u];
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_high_pair(
          context, source_op, &plan->strategy.fp8_encode, &emission_state,
          packed, f32_lanes[2], f32_lanes[3], &packed));
      if (loom_amdgpu_fp8_encode_plan_canonicalizes_native_nan(
              &plan->strategy.fp8_encode)) {
        const uint32_t canonical_lane_count =
            register_lane_count <= 2u ? 2u : 4u;
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fp8_encode_native_nan_canonicalization(
                context, source_op, &plan->strategy.fp8_encode, &emission_state,
                f32_lanes, canonical_lane_count, packed, &packed));
      }
    }
    result_registers[result_register_index] = packed;
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             result_registers,
                                             plan->result_register_count);
}

static iree_status_t
loom_amdgpu_lower_vector_16bit_float_fptrunc_from_storage_origin(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  (void)source_op;
  return loom_low_lower_bind_value_alias(context, plan->storage_source,
                                         plan->result);
}

static iree_status_t loom_amdgpu_lower_vector_16bit_float_fptrunc(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  if (plan->strategy_kind ==
      LOOM_AMDGPU_VECTOR_FLOAT_CONVERSION_STRATEGY_FP8_ENCODE) {
    return loom_amdgpu_lower_vector_fp8_encode(context, source_op, plan);
  }
  if (plan->source_element_type == LOOM_SCALAR_TYPE_F8E4M3 ||
      plan->source_element_type == LOOM_SCALAR_TYPE_F8E5M2) {
    return loom_amdgpu_lower_vector_16bit_float_extf(context, source_op, plan);
  }
  if (loom_amdgpu_vector_16bit_float_fptrunc_has_storage_origin(plan)) {
    return loom_amdgpu_lower_vector_16bit_float_fptrunc_from_storage_origin(
        context, source_op, plan);
  }

  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t source_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_source);
  if (loom_type_kind(source_lane_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_type(context, &source_lane_type));
  }
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  if (plan->result_element_type == LOOM_SCALAR_TYPE_BF16) {
    return loom_amdgpu_lower_vector_f32_to_packed_bf16(
        context, source_op, plan, low_source, source_lane_type, lane_type);
  }
  return loom_amdgpu_lower_vector_f32_to_packed_f16(
      context, source_op, plan, low_source, source_lane_type, lane_type);
}

iree_status_t loom_amdgpu_lower_vector_16bit_float_conversion(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_EXTF:
    case LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_DECODE:
      return loom_amdgpu_lower_vector_16bit_float_extf(context, source_op,
                                                       plan);
    case LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_FPTRUNC:
    case LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_ENCODE:
      return loom_amdgpu_lower_vector_16bit_float_fptrunc(context, source_op,
                                                          plan);
    default:
      IREE_ASSERT_UNREACHABLE("unknown 16-bit float conversion plan");
      IREE_BUILTIN_UNREACHABLE();
  }
}
