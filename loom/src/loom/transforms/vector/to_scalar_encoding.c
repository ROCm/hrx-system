// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/to_scalar_encoding.h"

#include "loom/analysis/contract.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"

//===----------------------------------------------------------------------===//
// Schema support
//===----------------------------------------------------------------------===//

static bool loom_vector_to_scalar_flags_are_single_or_none(uint64_t flags) {
  return flags == 0 || (flags & (flags - 1)) == 0;
}

static bool loom_vector_to_scalar_flags_are_supported(uint64_t flags,
                                                      uint64_t supported) {
  return (flags & ~supported) == 0 &&
         loom_vector_to_scalar_flags_are_single_or_none(flags);
}

static bool loom_vector_to_scalar_numeric_format_to_scalar_type(
    loom_value_fact_numeric_format_flags_t format,
    loom_scalar_type_t* out_scalar_type) {
  switch (format) {
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F64:
      *out_scalar_type = LOOM_SCALAR_TYPE_F64;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F32:
      *out_scalar_type = LOOM_SCALAR_TYPE_F32;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F16:
      *out_scalar_type = LOOM_SCALAR_TYPE_F16;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_BF16:
      *out_scalar_type = LOOM_SCALAR_TYPE_BF16;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FNUZ:
      *out_scalar_type = LOOM_SCALAR_TYPE_F8E4M3;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2FNUZ:
      *out_scalar_type = LOOM_SCALAR_TYPE_F8E5M2;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_I32:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_U32:
      *out_scalar_type = LOOM_SCALAR_TYPE_I32;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_I16:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_U16:
      *out_scalar_type = LOOM_SCALAR_TYPE_I16;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_I8:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_U8:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I8:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_CODEBOOK_INDEX:
      *out_scalar_type = LOOM_SCALAR_TYPE_I8;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_I1:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_U1:
      *out_scalar_type = LOOM_SCALAR_TYPE_I1;
      return true;
    default:
      return false;
  }
}

static bool loom_vector_to_scalar_numeric_format_is_unsigned(
    loom_value_fact_numeric_format_flags_t format) {
  switch (format) {
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_U32:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_U16:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_U8:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_U1:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_CODEBOOK_INDEX:
      return true;
    default:
      return false;
  }
}

static bool loom_vector_to_scalar_encoded_schema_uses_codebook(
    loom_value_fact_encoded_operand_schema_t schema) {
  return schema.codebook_policy ==
         LOOM_VALUE_FACT_CODEBOOK_POLICY_DYNAMIC_TABLE_OPERAND;
}

static bool loom_vector_to_scalar_encoded_schema_has_scale(
    loom_value_fact_encoded_operand_schema_t schema) {
  return schema.scale_operand_count != 0;
}

static bool loom_vector_to_scalar_encoded_schema_has_scale_affine(
    loom_value_fact_encoded_operand_schema_t schema) {
  return iree_any_bit_set(
      schema.affine_policy,
      LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_ONLY |
          LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_MIN |
          LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_BIAS |
          LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_ZERO_POINT);
}

static bool loom_vector_to_scalar_encoded_schema_auxiliary_is_supported(
    loom_value_fact_encoded_operand_schema_t schema,
    loom_vector_encoding_auxiliary_view_t auxiliary) {
  loom_vector_encoding_auxiliary_key_flags_t required_keys = 0;
  if (!loom_vector_encoding_auxiliary_required_keys_from_schema(
          schema, &required_keys, NULL)) {
    return false;
  }
  if ((auxiliary.present_keys & required_keys) != required_keys) {
    return false;
  }
  return (auxiliary.present_keys & ~required_keys) == 0;
}

static bool loom_vector_to_scalar_encoded_schema_is_supported(
    loom_value_fact_encoded_operand_schema_t schema) {
  if (loom_value_fact_encoded_operand_schema_is_unknown(schema)) {
    return false;
  }
  if (schema.payload_packing != LOOM_VALUE_FACT_PAYLOAD_PACKING_DENSE_LANES) {
    return false;
  }
  if (!loom_vector_to_scalar_flags_are_supported(
          schema.element_format,
          LOOM_VALUE_FACT_NUMERIC_FORMAT_F64 |
              LOOM_VALUE_FACT_NUMERIC_FORMAT_F32 |
              LOOM_VALUE_FACT_NUMERIC_FORMAT_F16 |
              LOOM_VALUE_FACT_NUMERIC_FORMAT_BF16 |
              LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3 |
              LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FNUZ |
              LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2 |
              LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2FNUZ |
              LOOM_VALUE_FACT_NUMERIC_FORMAT_I32 |
              LOOM_VALUE_FACT_NUMERIC_FORMAT_U32 |
              LOOM_VALUE_FACT_NUMERIC_FORMAT_I16 |
              LOOM_VALUE_FACT_NUMERIC_FORMAT_U16 |
              LOOM_VALUE_FACT_NUMERIC_FORMAT_I8 |
              LOOM_VALUE_FACT_NUMERIC_FORMAT_U8 |
              LOOM_VALUE_FACT_NUMERIC_FORMAT_I1 |
              LOOM_VALUE_FACT_NUMERIC_FORMAT_U1 |
              LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I8 |
              LOOM_VALUE_FACT_NUMERIC_FORMAT_CODEBOOK_INDEX)) {
    return false;
  }
  if (!loom_vector_to_scalar_flags_are_supported(
          schema.scale_format, LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE |
                                   LOOM_VALUE_FACT_NUMERIC_FORMAT_F32 |
                                   LOOM_VALUE_FACT_NUMERIC_FORMAT_F16 |
                                   LOOM_VALUE_FACT_NUMERIC_FORMAT_BF16 |
                                   LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3 |
                                   LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FNUZ |
                                   LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2)) {
    return false;
  }
  if (schema.secondary_scale_format != LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE) {
    return false;
  }
  if (!loom_vector_to_scalar_flags_are_supported(
          schema.scale_topology,
          LOOM_VALUE_FACT_SCALE_TOPOLOGY_NONE |
              LOOM_VALUE_FACT_SCALE_TOPOLOGY_TENSOR_GLOBAL |
              LOOM_VALUE_FACT_SCALE_TOPOLOGY_ROW |
              LOOM_VALUE_FACT_SCALE_TOPOLOGY_COLUMN |
              LOOM_VALUE_FACT_SCALE_TOPOLOGY_GROUP_1D |
              LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D)) {
    return false;
  }
  if (!loom_vector_to_scalar_flags_are_supported(
          schema.affine_policy,
          LOOM_VALUE_FACT_AFFINE_POLICY_NONE |
              LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_ONLY |
              LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_MIN |
              LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_BIAS |
              LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_ZERO_POINT)) {
    return false;
  }
  if (schema.rounding_policy != LOOM_VALUE_FACT_ROUNDING_POLICY_NONE ||
      schema.sparsity_policy != LOOM_VALUE_FACT_SPARSITY_POLICY_NONE ||
      schema.scale_operand_count > 1) {
    return false;
  }
  if (!loom_vector_to_scalar_flags_are_supported(
          schema.codebook_policy,
          LOOM_VALUE_FACT_CODEBOOK_POLICY_NONE |
              LOOM_VALUE_FACT_CODEBOOK_POLICY_DYNAMIC_TABLE_OPERAND)) {
    return false;
  }
  if (loom_vector_to_scalar_encoded_schema_has_scale(schema)) {
    if (!loom_vector_to_scalar_encoded_schema_has_scale_affine(schema) ||
        schema.scale_topology == LOOM_VALUE_FACT_SCALE_TOPOLOGY_NONE ||
        schema.scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE) {
      return false;
    }
    if (iree_any_bit_set(schema.scale_topology,
                         LOOM_VALUE_FACT_SCALE_TOPOLOGY_GROUP_1D |
                             LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D) &&
        schema.scale_group_element_count == 0) {
      return false;
    }
  } else if (schema.scale_topology != LOOM_VALUE_FACT_SCALE_TOPOLOGY_NONE ||
             loom_vector_to_scalar_encoded_schema_has_scale_affine(schema)) {
    return false;
  }
  if (loom_vector_to_scalar_encoded_schema_uses_codebook(schema) &&
      schema.element_format != LOOM_VALUE_FACT_NUMERIC_FORMAT_CODEBOOK_INDEX) {
    return false;
  }
  return true;
}

static bool loom_vector_to_scalar_numeric_lane_cast_is_supported(
    loom_type_t input_type, loom_type_t result_type) {
  if (!loom_type_is_scalar(input_type) || !loom_type_is_scalar(result_type)) {
    return false;
  }
  if (loom_type_equal(input_type, result_type)) {
    return true;
  }

  loom_scalar_type_t input_scalar_type = loom_type_element_type(input_type);
  loom_scalar_type_t result_scalar_type = loom_type_element_type(result_type);
  int32_t input_width = loom_scalar_type_bitwidth(input_scalar_type);
  int32_t result_width = loom_scalar_type_bitwidth(result_scalar_type);
  if (input_width <= 0 || result_width <= 0) {
    return false;
  }

  if (loom_scalar_type_is_float(input_scalar_type) &&
      loom_scalar_type_is_float(result_scalar_type)) {
    return input_width != result_width;
  }
  if (loom_scalar_type_is_integer(input_scalar_type) &&
      loom_scalar_type_is_integer(result_scalar_type)) {
    return true;
  }
  return loom_scalar_type_is_integer(input_scalar_type) &&
         loom_scalar_type_is_float(result_scalar_type);
}

static bool loom_vector_to_scalar_encoded_auxiliary_lane_type(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_encoded_matrix_operand_t* operand,
    loom_vector_encoding_auxiliary_key_t key, loom_type_t* out_lane_type) {
  loom_value_id_t value = operand->auxiliary.values[key];
  if (value == LOOM_VALUE_ID_INVALID ||
      value >= state->rewriter->module->values.count) {
    return false;
  }
  loom_type_t value_type =
      loom_module_value_type(state->rewriter->module, value);
  if (!loom_type_is_vector(value_type) || loom_type_rank(value_type) != 1) {
    return false;
  }
  *out_lane_type = loom_vector_to_scalar_lane_type(value_type);
  return true;
}

static bool loom_vector_to_scalar_encoded_auxiliary_lane_cast_is_supported(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_encoded_matrix_operand_t* operand,
    loom_vector_encoding_auxiliary_key_t key, loom_type_t result_type) {
  loom_type_t lane_type = {0};
  return loom_vector_to_scalar_encoded_auxiliary_lane_type(state, operand, key,
                                                           &lane_type) &&
         loom_vector_to_scalar_numeric_lane_cast_is_supported(lane_type,
                                                              result_type);
}

static bool loom_vector_to_scalar_encoded_auxiliary_matches_format(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_encoded_matrix_operand_t* operand,
    loom_vector_encoding_auxiliary_key_t key,
    loom_value_fact_numeric_format_flags_t format, loom_type_t result_type) {
  loom_scalar_type_t expected_scalar_type = 0;
  if (!loom_vector_to_scalar_numeric_format_to_scalar_type(
          format, &expected_scalar_type)) {
    return false;
  }
  loom_type_t lane_type = {0};
  return loom_vector_to_scalar_encoded_auxiliary_lane_type(state, operand, key,
                                                           &lane_type) &&
         loom_type_element_type(lane_type) == expected_scalar_type &&
         loom_vector_to_scalar_numeric_lane_cast_is_supported(lane_type,
                                                              result_type);
}

static bool loom_vector_to_scalar_encoded_logical_element_count_matches(
    const loom_vector_to_scalar_encoded_matrix_operand_t* operand) {
  if (operand->rows.is_dynamic || operand->columns.is_dynamic) {
    return true;
  }
  if (operand->rows.static_value < 0 || operand->columns.static_value < 0) {
    return false;
  }
  const uint64_t row_count = (uint64_t)operand->rows.static_value;
  const uint64_t column_count = (uint64_t)operand->columns.static_value;
  if (column_count != 0 && row_count > UINT64_MAX / column_count) {
    return false;
  }
  uint64_t element_count = row_count * column_count;
  return element_count <= UINT16_MAX &&
         operand->schema.payload_element_count == (uint16_t)element_count;
}

static bool loom_vector_to_scalar_encoded_raw_lane_type_matches(
    loom_value_fact_encoded_operand_schema_t schema,
    loom_type_t raw_lane_type) {
  loom_scalar_type_t expected_scalar_type = 0;
  if (!loom_vector_to_scalar_numeric_format_to_scalar_type(
          schema.element_format, &expected_scalar_type)) {
    return false;
  }
  return loom_type_element_type(raw_lane_type) == expected_scalar_type;
}

static bool loom_vector_to_scalar_encoded_affine_is_supported(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_encoded_matrix_operand_t* operand,
    loom_type_t result_type) {
  if (!loom_vector_to_scalar_encoded_schema_has_scale_affine(operand->schema)) {
    return true;
  }
  if (!loom_scalar_type_is_float(loom_type_element_type(result_type))) {
    return false;
  }

  if (loom_vector_to_scalar_encoded_schema_has_scale(operand->schema) &&
      !loom_vector_to_scalar_encoded_auxiliary_matches_format(
          state, operand, LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE,
          operand->schema.scale_format, result_type)) {
    return false;
  }

  switch (operand->schema.affine_policy) {
    case LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_ZERO_POINT:
      return loom_vector_to_scalar_encoded_auxiliary_matches_format(
          state, operand, LOOM_VECTOR_ENCODING_AUXILIARY_KEY_ZERO_POINT,
          operand->schema.element_format, result_type);
    case LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_MIN:
      return loom_vector_to_scalar_encoded_auxiliary_lane_cast_is_supported(
          state, operand, LOOM_VECTOR_ENCODING_AUXILIARY_KEY_MINIMUM,
          result_type);
    case LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_BIAS:
      return loom_vector_to_scalar_encoded_auxiliary_lane_cast_is_supported(
          state, operand, LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIAS, result_type);
    case LOOM_VALUE_FACT_AFFINE_POLICY_NONE:
    case LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_ONLY:
    default:
      return true;
  }
}

static bool loom_vector_to_scalar_encoded_codebook_is_supported(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_encoded_matrix_operand_t* operand,
    loom_type_t result_type) {
  if (!loom_vector_to_scalar_encoded_schema_uses_codebook(operand->schema)) {
    return true;
  }
  loom_type_t table_lane_type = {0};
  if (!loom_vector_to_scalar_encoded_auxiliary_lane_type(
          state, operand, LOOM_VECTOR_ENCODING_AUXILIARY_KEY_CODEBOOK,
          &table_lane_type)) {
    return false;
  }
  return loom_scalar_type_is_float(loom_type_element_type(table_lane_type)) &&
         loom_scalar_type_is_float(loom_type_element_type(result_type)) &&
         loom_vector_to_scalar_numeric_lane_cast_is_supported(table_lane_type,
                                                              result_type);
}

bool loom_vector_to_scalar_encoded_matrix_operand_is_supported(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_encoded_matrix_operand_t* operand,
    loom_type_t raw_lane_type, loom_type_t result_type) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(operand);
  return loom_vector_to_scalar_encoded_matrix_operand_rejection_bits(
             state, operand, raw_lane_type, result_type) ==
         LOOM_CONTRACT_REJECTION_NONE;
}

uint32_t loom_vector_to_scalar_encoded_matrix_operand_rejection_bits(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_encoded_matrix_operand_t* operand,
    loom_type_t raw_lane_type, loom_type_t result_type) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(operand);
  loom_contract_rejection_bits_t rejection_bits = LOOM_CONTRACT_REJECTION_NONE;
  if (!loom_vector_to_scalar_encoded_schema_is_supported(operand->schema)) {
    return LOOM_CONTRACT_REJECTION_SCHEMA;
  }
  if (!loom_vector_to_scalar_encoded_schema_auxiliary_is_supported(
          operand->schema, operand->auxiliary)) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_AUXILIARY_OPERAND;
  }
  if (!loom_vector_to_scalar_encoded_logical_element_count_matches(operand)) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_SHAPE;
  }
  if (!loom_vector_to_scalar_encoded_raw_lane_type_matches(operand->schema,
                                                           raw_lane_type) ||
      !loom_vector_to_scalar_numeric_lane_cast_is_supported(raw_lane_type,
                                                            result_type)) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_NUMERIC;
  }
  if (!loom_vector_to_scalar_encoded_affine_is_supported(state, operand,
                                                         result_type)) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_NUMERIC |
                      LOOM_CONTRACT_REJECTION_AUXILIARY_OPERAND;
  }
  if (!loom_vector_to_scalar_encoded_codebook_is_supported(state, operand,
                                                           result_type)) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_NUMERIC |
                      LOOM_CONTRACT_REJECTION_AUXILIARY_OPERAND;
  }
  return rejection_bits;
}

//===----------------------------------------------------------------------===//
// Lane construction
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_to_scalar_cast_lane_to_index(
    loom_vector_to_scalar_state_t* state, loom_value_id_t input,
    loom_type_t input_type, loom_value_id_t* out_index) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  if (loom_type_equal(input_type, index_type)) {
    *out_index = input;
    return iree_ok_status();
  }
  loom_op_t* cast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(&state->rewriter->builder, input,
                                             input_type, index_type,
                                             state->location, &cast_op));
  *out_index = loom_index_cast_result(cast_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_encoded_auxiliary_lane(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_encoded_matrix_operand_t* operand,
    loom_vector_encoding_auxiliary_key_t key,
    loom_vector_to_scalar_index_term_t index, loom_value_id_t* out_lane) {
  loom_value_id_t vector_value = operand->auxiliary.values[key];
  loom_vector_to_scalar_index_list_t indices = {0};
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_terms_to_index_list(state, &index, 1, &indices));
  return loom_vector_to_scalar_materialize_lane(state, vector_value, indices,
                                                out_lane);
}

static iree_status_t loom_vector_to_scalar_encoded_scale_index(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_encoded_matrix_operand_t* operand,
    loom_vector_to_scalar_index_term_t row,
    loom_vector_to_scalar_index_term_t column,
    loom_vector_to_scalar_index_term_t ordinal,
    loom_vector_to_scalar_index_term_t* out_index) {
  switch (operand->schema.scale_topology) {
    case LOOM_VALUE_FACT_SCALE_TOPOLOGY_TENSOR_GLOBAL:
      *out_index = loom_vector_to_scalar_static_term(0);
      return iree_ok_status();
    case LOOM_VALUE_FACT_SCALE_TOPOLOGY_ROW:
      *out_index = row;
      return iree_ok_status();
    case LOOM_VALUE_FACT_SCALE_TOPOLOGY_COLUMN:
      *out_index = column;
      return iree_ok_status();
    case LOOM_VALUE_FACT_SCALE_TOPOLOGY_GROUP_1D:
    case LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D:
      return loom_vector_to_scalar_build_term_binary(
          state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_DIV, ordinal,
          loom_vector_to_scalar_static_term(
              operand->schema.scale_group_element_count),
          out_index);
    case LOOM_VALUE_FACT_SCALE_TOPOLOGY_NONE:
    default:
      *out_index = loom_vector_to_scalar_static_term(0);
      return iree_ok_status();
  }
}

static iree_status_t loom_vector_to_scalar_encoded_codebook_lane(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_encoded_matrix_operand_t* operand,
    loom_value_id_t raw_lane, loom_type_t raw_lane_type,
    loom_type_t result_type, loom_value_id_t* out_lane) {
  loom_value_id_t table_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_lane_to_index(
      state, raw_lane, raw_lane_type, &table_index));

  loom_vector_to_scalar_index_list_t table_indices = {
      .dynamic_indices = &table_index,
      .rank = 1,
  };
  loom_value_id_t codebook =
      operand->auxiliary.values[LOOM_VECTOR_ENCODING_AUXILIARY_KEY_CODEBOOK];
  loom_value_id_t table_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, codebook, table_indices, &table_lane));

  loom_type_t codebook_type =
      loom_module_value_type(state->rewriter->module, codebook);
  loom_type_t table_lane_type = loom_vector_to_scalar_lane_type(codebook_type);
  return loom_vector_to_scalar_cast_numeric_lane(
      state, table_lane, table_lane_type, result_type,
      /*unsigned_input=*/false, out_lane);
}

static iree_status_t loom_vector_to_scalar_encoded_raw_value_lane(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_encoded_matrix_operand_t* operand,
    loom_value_id_t raw_lane, loom_type_t raw_lane_type,
    loom_type_t result_type, loom_value_id_t* out_lane) {
  if (loom_vector_to_scalar_encoded_schema_uses_codebook(operand->schema)) {
    return loom_vector_to_scalar_encoded_codebook_lane(
        state, operand, raw_lane, raw_lane_type, result_type, out_lane);
  }
  return loom_vector_to_scalar_cast_numeric_lane(
      state, raw_lane, raw_lane_type, result_type,
      loom_vector_to_scalar_numeric_format_is_unsigned(
          operand->schema.element_format),
      out_lane);
}

static iree_status_t loom_vector_to_scalar_encoded_affine_operand_lane(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_encoded_matrix_operand_t* operand,
    loom_vector_encoding_auxiliary_key_t key,
    loom_vector_to_scalar_index_term_t index, loom_type_t result_type,
    bool unsigned_input, loom_value_id_t* out_lane) {
  loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_encoded_auxiliary_lane(
      state, operand, key, index, &lane));
  loom_value_id_t vector_value = operand->auxiliary.values[key];
  loom_type_t vector_type =
      loom_module_value_type(state->rewriter->module, vector_value);
  loom_type_t lane_type = loom_vector_to_scalar_lane_type(vector_type);
  return loom_vector_to_scalar_cast_numeric_lane(
      state, lane, lane_type, result_type, unsigned_input, out_lane);
}

static iree_status_t loom_vector_to_scalar_encoded_apply_scale(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_encoded_matrix_operand_t* operand,
    loom_vector_to_scalar_index_term_t scale_index, loom_type_t result_type,
    loom_value_id_t input, loom_value_id_t* out_lane) {
  if (!loom_vector_to_scalar_encoded_schema_has_scale(operand->schema)) {
    *out_lane = input;
    return iree_ok_status();
  }
  loom_value_id_t scale = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_encoded_affine_operand_lane(
      state, operand, LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE, scale_index,
      result_type, /*unsigned_input=*/false, &scale));
  return loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_MULF, input, scale, result_type, out_lane);
}

static iree_status_t loom_vector_to_scalar_encoded_apply_affine(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_encoded_matrix_operand_t* operand,
    loom_vector_to_scalar_index_term_t scale_index, loom_type_t result_type,
    loom_value_id_t input, loom_value_id_t* out_lane) {
  loom_value_id_t value = input;
  if (operand->schema.affine_policy ==
      LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_ZERO_POINT) {
    loom_value_id_t zero_point = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_encoded_affine_operand_lane(
        state, operand, LOOM_VECTOR_ENCODING_AUXILIARY_KEY_ZERO_POINT,
        scale_index, result_type,
        loom_vector_to_scalar_numeric_format_is_unsigned(
            operand->schema.element_format),
        &zero_point));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_binary(
        state, LOOM_OP_SCALAR_SUBF, value, zero_point, result_type, &value));
  }

  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_encoded_apply_scale(
      state, operand, scale_index, result_type, value, &value));

  loom_vector_encoding_auxiliary_key_t offset_key =
      LOOM_VECTOR_ENCODING_AUXILIARY_KEY_COUNT_;
  switch (operand->schema.affine_policy) {
    case LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_MIN:
      offset_key = LOOM_VECTOR_ENCODING_AUXILIARY_KEY_MINIMUM;
      break;
    case LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_BIAS:
      offset_key = LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIAS;
      break;
    case LOOM_VALUE_FACT_AFFINE_POLICY_NONE:
    case LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_ONLY:
    case LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_ZERO_POINT:
    default:
      break;
  }
  if (offset_key == LOOM_VECTOR_ENCODING_AUXILIARY_KEY_COUNT_) {
    *out_lane = value;
    return iree_ok_status();
  }

  loom_value_id_t offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_encoded_affine_operand_lane(
      state, operand, offset_key, scale_index, result_type,
      /*unsigned_input=*/false, &offset));
  return loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ADDF, value, offset, result_type, out_lane);
}

iree_status_t loom_vector_to_scalar_build_encoded_matrix_lane(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_encoded_matrix_operand_t* operand,
    loom_value_id_t raw_lane, loom_type_t raw_lane_type,
    loom_type_t result_type, loom_vector_to_scalar_index_term_t row,
    loom_vector_to_scalar_index_term_t column,
    loom_vector_to_scalar_index_term_t ordinal, loom_value_id_t* out_lane) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(operand);
  IREE_ASSERT_ARGUMENT(out_lane);
  if (!loom_vector_to_scalar_encoded_matrix_operand_is_supported(
          state, operand, raw_lane_type, result_type)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "unsupported encoded matrix operand reached scalar reference builder");
  }

  loom_value_id_t decoded = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_encoded_raw_value_lane(
      state, operand, raw_lane, raw_lane_type, result_type, &decoded));

  loom_vector_to_scalar_index_term_t scale_index = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_encoded_scale_index(
      state, operand, row, column, ordinal, &scale_index));
  return loom_vector_to_scalar_encoded_apply_affine(
      state, operand, scale_index, result_type, decoded, out_lane);
}
