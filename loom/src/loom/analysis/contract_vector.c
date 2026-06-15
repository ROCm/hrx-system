// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/contract_vector.h"

#include "loom/analysis/contract_storage.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/vector/encoding_auxiliary.h"
#include "loom/ops/vector/fragment.h"
#include "loom/ops/vector/ops.h"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static bool loom_contract_vector_fail(
    loom_contract_rejection_bits_t rejection_bits,
    loom_contract_diagnostic_t* out_diagnostic) {
  if (out_diagnostic) {
    out_diagnostic->rejection_bits = rejection_bits;
  }
  return false;
}

static bool loom_contract_vector_value_type(const loom_module_t* module,
                                            loom_value_id_t value_id,
                                            loom_type_t* out_type) {
  *out_type = loom_type_none();
  if (value_id >= module->values.count) {
    return false;
  }
  *out_type = loom_module_value_type(module, value_id);
  return true;
}

static bool loom_contract_vector_static_element_count(
    loom_type_t type, uint64_t* out_element_count) {
  *out_element_count = 0;
  if (!loom_type_is_vector(type)) {
    return false;
  }
  return loom_type_static_element_count(type, out_element_count);
}

static bool loom_contract_vector_assign_uint16(uint64_t value,
                                               uint16_t* out_value) {
  if (value > UINT16_MAX) {
    return false;
  }
  *out_value = (uint16_t)value;
  return true;
}

static bool loom_contract_vector_assign_scaled_uint16(uint64_t value,
                                                      uint64_t multiplier,
                                                      uint16_t* out_value) {
  if (multiplier == 0 || value > UINT16_MAX / multiplier) {
    return false;
  }
  *out_value = (uint16_t)(value * multiplier);
  return true;
}

static bool loom_contract_vector_assign_dense_payload(
    loom_type_t type, loom_contract_operand_t* operand,
    loom_contract_rejection_bits_t* out_rejection_bits) {
  uint64_t element_count = 0;
  if (!loom_contract_vector_static_element_count(type, &element_count) ||
      element_count == 0 || element_count > UINT16_MAX) {
    *out_rejection_bits = LOOM_CONTRACT_REJECTION_SHAPE;
    return false;
  }
  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (element_bit_count <= 0) {
    *out_rejection_bits = LOOM_CONTRACT_REJECTION_NUMERIC;
    return false;
  }
  if (element_count > UINT64_MAX / (uint32_t)element_bit_count) {
    *out_rejection_bits = LOOM_CONTRACT_REJECTION_SHAPE;
    return false;
  }
  const uint64_t payload_bit_count =
      element_count * (uint32_t)element_bit_count;
  const uint64_t register_count =
      payload_bit_count / 32u + (payload_bit_count % 32u != 0 ? 1u : 0u);
  if (register_count == 0 || register_count > UINT16_MAX) {
    *out_rejection_bits = LOOM_CONTRACT_REJECTION_SHAPE;
    return false;
  }
  operand->payload_register_count = (uint16_t)register_count;
  operand->payload_element_count = (uint16_t)element_count;
  return true;
}

static bool loom_contract_vector_static_grouped_shapes_match(
    loom_type_t source_type, loom_type_t result_type, int64_t group_size) {
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type) ||
      !loom_type_is_all_static(source_type) ||
      !loom_type_is_all_static(result_type)) {
    return false;
  }
  const uint8_t rank = loom_type_rank(source_type);
  if (rank == 0 || loom_type_rank(result_type) != rank) {
    return false;
  }
  for (uint8_t i = 0; i + 1 < rank; ++i) {
    if (loom_type_dim_static_size_at(source_type, i) !=
        loom_type_dim_static_size_at(result_type, i)) {
      return false;
    }
  }
  const int64_t source_last_extent =
      loom_type_dim_static_size_at(source_type, rank - 1);
  const int64_t result_last_extent =
      loom_type_dim_static_size_at(result_type, rank - 1);
  return source_last_extent >= 0 && result_last_extent >= 0 &&
         source_last_extent % group_size == 0 &&
         source_last_extent / group_size == result_last_extent;
}

static loom_contract_operand_t loom_contract_vector_operand(
    loom_contract_operand_role_t role,
    loom_contract_numeric_type_t numeric_type) {
  return (loom_contract_operand_t){
      .role = role,
      .numeric_type = numeric_type,
  };
}

static bool loom_contract_vector_dot4i_numeric_types(
    loom_vector_dot4i_kind_t kind, loom_contract_numeric_type_t* out_lhs_type,
    loom_contract_numeric_type_t* out_rhs_type) {
  *out_lhs_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  *out_rhs_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  switch (kind) {
    case LOOM_VECTOR_DOT4I_KIND_S8S8:
      *out_lhs_type = LOOM_CONTRACT_NUMERIC_I8;
      *out_rhs_type = LOOM_CONTRACT_NUMERIC_I8;
      return true;
    case LOOM_VECTOR_DOT4I_KIND_U8S8:
      *out_lhs_type = LOOM_CONTRACT_NUMERIC_U8;
      *out_rhs_type = LOOM_CONTRACT_NUMERIC_I8;
      return true;
    case LOOM_VECTOR_DOT4I_KIND_S8U8:
      *out_lhs_type = LOOM_CONTRACT_NUMERIC_I8;
      *out_rhs_type = LOOM_CONTRACT_NUMERIC_U8;
      return true;
    case LOOM_VECTOR_DOT4I_KIND_U8U8:
      *out_lhs_type = LOOM_CONTRACT_NUMERIC_U8;
      *out_rhs_type = LOOM_CONTRACT_NUMERIC_U8;
      return true;
    case LOOM_VECTOR_DOT4I_KIND_COUNT_:
    default:
      return false;
  }
}

static bool loom_contract_vector_populate_fragment(
    uint64_t input_lane_count, uint64_t result_lane_count,
    uint64_t source_lane_bit_width, uint16_t k_group_size,
    loom_contract_request_t* request) {
  if (!loom_contract_vector_assign_uint16(
          input_lane_count, &request->fragment.source_lane_count) ||
      !loom_contract_vector_assign_uint16(
          result_lane_count, &request->fragment.result_lane_count) ||
      !loom_contract_vector_assign_scaled_uint16(
          input_lane_count, source_lane_bit_width,
          &request->fragment.vector_bit_width)) {
    return false;
  }
  request->fragment.atom_bits = LOOM_CONTRACT_FRAGMENT_VECTOR_LANE;
  request->shape.m = (int64_t)request->fragment.result_lane_count;
  request->shape.n = 1;
  request->shape.k = (int64_t)request->fragment.source_lane_count;
  request->k_group_size = k_group_size;
  return true;
}

static bool loom_contract_vector_finish(
    loom_lowering_policy_t policy, loom_contract_request_t* request,
    loom_contract_diagnostic_t* out_diagnostic) {
  request->capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_CPU_PACKED_DOT;
  request->policy = policy;
  if (!loom_contract_request_validate(request, out_diagnostic)) {
    return false;
  }
  return true;
}

static bool loom_contract_vector_exact_positive_i64(loom_value_facts_t facts,
                                                    int64_t* out_value) {
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts) ||
      facts.range_lo <= 0) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

typedef struct loom_contract_vector_shape_dimension_t {
  // Exact positive value when value facts prove it, or 0 when dynamic.
  int64_t exact_value;

  // SSA value carrying the dimension.
  loom_contract_value_ref_t value_ref;
} loom_contract_vector_shape_dimension_t;

static bool loom_contract_vector_query_shape_dimension(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_contract_vector_shape_dimension_t* out_dimension) {
  *out_dimension = (loom_contract_vector_shape_dimension_t){0};
  if (value_id == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  out_dimension->value_ref = loom_contract_value_ref_from_value_id(value_id);
  int64_t exact_value = 0;
  if (loom_contract_vector_exact_positive_i64(
          loom_value_fact_table_lookup(fact_table, value_id), &exact_value)) {
    out_dimension->exact_value = exact_value;
  }
  return true;
}

static bool loom_contract_vector_shape_dimensions_match(
    loom_contract_vector_shape_dimension_t lhs,
    loom_contract_vector_shape_dimension_t rhs,
    loom_contract_vector_shape_dimension_t* out_dimension) {
  *out_dimension = lhs;
  if (lhs.exact_value > 0 && rhs.exact_value > 0) {
    if (lhs.exact_value != rhs.exact_value) {
      return false;
    }
    return true;
  }
  if (!loom_contract_value_ref_is_present(lhs.value_ref) ||
      lhs.value_ref != rhs.value_ref) {
    return false;
  }
  out_dimension->exact_value =
      lhs.exact_value > 0 ? lhs.exact_value : rhs.exact_value;
  return true;
}

static bool loom_contract_vector_query_fragment_fact(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_vector_fragment_role_flags_t role_flags,
    loom_vector_fragment_fact_t* out_fact) {
  loom_vector_fragment_fact_t fact;
  if (value_id == LOOM_VALUE_ID_INVALID ||
      !loom_vector_fragment_fact_query_value_facts(
          &fact_table->context,
          loom_value_fact_table_lookup(fact_table, value_id), &fact) ||
      !iree_any_bit_set(fact.role_flags, role_flags) || fact.shape_rank != 2) {
    return false;
  }
  *out_fact = fact;
  return true;
}

static const loom_vector_encoding_auxiliary_key_flags_t
    loom_contract_vector_auxiliary_key_flags[] = {
        [LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SCALE] =
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE,
        [LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SECONDARY_SCALE] =
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SECONDARY_SCALE |
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE2 |
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE3 |
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE4 |
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE5 |
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE6 |
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE7,
        [LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_ZERO_POINT] =
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_ZERO_POINT,
        [LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_MIN] =
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_MINIMUM,
        [LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SPARSE_METADATA] =
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SPARSITY |
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_METADATA |
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_INDICES |
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_OFFSETS |
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_MASK,
        [LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_CODEBOOK_TABLE] =
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_CODEBOOK |
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_THRESHOLDS |
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_CENTROIDS,
        [LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_RESIDUAL] =
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_RESIDUAL |
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_OUTLIERS,
        [LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SIGN] =
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SIGNS,
        [LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_RUNTIME_AMAX] =
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_AMAX,
};
static_assert(IREE_ARRAYSIZE(loom_contract_vector_auxiliary_key_flags) ==
                  LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_COUNT_,
              "contract auxiliary map must cover every key");

static bool loom_contract_vector_auxiliary_group_value(
    const loom_vector_encoding_auxiliary_view_t* auxiliary,
    loom_vector_encoding_auxiliary_key_flags_t vector_key_flags,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_vector_encoding_auxiliary_key_flags_t present_group_keys =
      auxiliary->present_keys & vector_key_flags;
  if (present_group_keys == 0) {
    return false;
  }
  for (uint8_t i = 0; i < LOOM_VECTOR_ENCODING_AUXILIARY_KEY_COUNT_; ++i) {
    loom_vector_encoding_auxiliary_key_t vector_key =
        (loom_vector_encoding_auxiliary_key_t)i;
    if (!iree_any_bit_set(
            present_group_keys,
            loom_vector_encoding_auxiliary_key_flag(vector_key))) {
      continue;
    }
    *out_value = auxiliary->values[vector_key];
    return *out_value != LOOM_VALUE_ID_INVALID;
  }
  return false;
}

static void loom_contract_vector_populate_auxiliary_operands(
    const loom_vector_encoding_auxiliary_view_t* auxiliary,
    loom_contract_encoded_operand_t* encoded) {
  for (uint8_t i = 0; i < LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_COUNT_; ++i) {
    loom_contract_auxiliary_operand_key_t contract_key =
        (loom_contract_auxiliary_operand_key_t)i;
    loom_value_id_t value = LOOM_VALUE_ID_INVALID;
    if (!loom_contract_vector_auxiliary_group_value(
            auxiliary, loom_contract_vector_auxiliary_key_flags[contract_key],
            &value)) {
      continue;
    }
    encoded->available_auxiliary_operands |=
        loom_contract_auxiliary_operand_key_flag(contract_key);
    encoded->auxiliary_value_refs[contract_key] =
        loom_contract_value_ref_from_value_id(value);
  }
}

static bool loom_contract_vector_fragment_storage_schema(
    loom_vector_fragment_fact_t fact,
    loom_value_fact_storage_schema_t* out_schema) {
  *out_schema = (loom_value_fact_storage_schema_t){
      .static_spec_encoding_id = fact.static_schema_encoding_id,
      .encoded_operand = fact.encoded_operand,
  };
  return !loom_value_fact_encoded_operand_schema_is_unknown(
      out_schema->encoded_operand);
}

static bool loom_contract_vector_operand_from_fragment(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_vector_fragment_fact_t fact, loom_contract_operand_role_t role,
    loom_contract_operand_t* out_operand,
    loom_contract_rejection_bits_t* out_rejection_bits) {
  *out_operand = (loom_contract_operand_t){
      .role = role,
  };
  *out_rejection_bits = LOOM_CONTRACT_REJECTION_NONE;

  if (iree_any_bit_set(fact.flags, LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_SCHEMA)) {
    loom_value_fact_storage_schema_t schema = {0};
    if (!loom_contract_vector_fragment_storage_schema(fact, &schema) ||
        !loom_contract_operand_from_storage_schema(role, schema, out_operand)) {
      *out_rejection_bits = LOOM_CONTRACT_REJECTION_SCHEMA;
      return false;
    }
    loom_contract_vector_populate_auxiliary_operands(&fact.auxiliary,
                                                     &out_operand->encoded);
    return true;
  }

  loom_type_t type = loom_type_none();
  if (!loom_contract_vector_value_type(module, value_id, &type) ||
      !loom_type_is_vector(type) ||
      !loom_contract_numeric_type_from_scalar(
          loom_type_element_type(type), false, &out_operand->numeric_type)) {
    *out_rejection_bits = LOOM_CONTRACT_REJECTION_NUMERIC;
    return false;
  }
  if (!loom_contract_vector_assign_dense_payload(type, out_operand,
                                                 out_rejection_bits)) {
    return false;
  }
  out_operand->role = role;
  return true;
}

static bool loom_contract_vector_numeric_is_float(
    loom_contract_numeric_type_t numeric_type) {
  switch (numeric_type) {
    case LOOM_CONTRACT_NUMERIC_F16:
    case LOOM_CONTRACT_NUMERIC_BF16:
    case LOOM_CONTRACT_NUMERIC_F32:
    case LOOM_CONTRACT_NUMERIC_F64:
    case LOOM_CONTRACT_NUMERIC_FP8:
    case LOOM_CONTRACT_NUMERIC_BF8:
    case LOOM_CONTRACT_NUMERIC_FP6:
    case LOOM_CONTRACT_NUMERIC_BF6:
    case LOOM_CONTRACT_NUMERIC_FP4:
      return true;
    default:
      return false;
  }
}

static bool loom_contract_vector_numeric_is_integer(
    loom_contract_numeric_type_t numeric_type) {
  switch (numeric_type) {
    case LOOM_CONTRACT_NUMERIC_I4:
    case LOOM_CONTRACT_NUMERIC_U4:
    case LOOM_CONTRACT_NUMERIC_I8:
    case LOOM_CONTRACT_NUMERIC_U8:
    case LOOM_CONTRACT_NUMERIC_I16:
    case LOOM_CONTRACT_NUMERIC_U16:
    case LOOM_CONTRACT_NUMERIC_I32:
    case LOOM_CONTRACT_NUMERIC_U32:
      return true;
    default:
      return false;
  }
}

static loom_contract_arithmetic_t loom_contract_vector_mma_arithmetic(
    const loom_contract_request_t* request) {
  const loom_contract_numeric_type_t lhs = request->lhs.numeric_type;
  const loom_contract_numeric_type_t rhs = request->rhs.numeric_type;
  const loom_contract_numeric_type_t accumulator =
      request->accumulator.numeric_type;
  const loom_contract_numeric_type_t result = request->result.numeric_type;
  if (loom_contract_vector_numeric_is_float(lhs) &&
      loom_contract_vector_numeric_is_float(rhs) &&
      loom_contract_vector_numeric_is_float(accumulator) &&
      loom_contract_vector_numeric_is_float(result)) {
    return lhs == rhs && lhs == accumulator && lhs == result
               ? LOOM_CONTRACT_ARITHMETIC_FLOAT_DOT
               : LOOM_CONTRACT_ARITHMETIC_MIXED_DOT;
  }
  if (loom_contract_vector_numeric_is_integer(lhs) &&
      loom_contract_vector_numeric_is_integer(rhs) &&
      loom_contract_vector_numeric_is_integer(accumulator) &&
      loom_contract_vector_numeric_is_integer(result)) {
    return LOOM_CONTRACT_ARITHMETIC_INTEGER_DOT;
  }
  return LOOM_CONTRACT_ARITHMETIC_MIXED_DOT;
}

static bool loom_contract_vector_mma_fail(
    loom_contract_rejection_bits_t rejection_bits,
    loom_contract_diagnostic_t* out_diagnostic) {
  if (out_diagnostic) {
    out_diagnostic->rejection_bits = rejection_bits;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Dot op adapters
//===----------------------------------------------------------------------===//

static bool loom_contract_request_from_dot2f_op(
    const loom_module_t* module, const loom_op_t* op,
    loom_lowering_policy_t policy, loom_contract_request_t* request,
    loom_contract_diagnostic_t* out_diagnostic) {
  if (!loom_vector_dot2f_isa(op)) {
    return false;
  }

  loom_type_t lhs_type = loom_type_none();
  loom_type_t rhs_type = loom_type_none();
  loom_type_t accumulator_type = loom_type_none();
  loom_type_t result_type = loom_type_none();
  if (!loom_contract_vector_value_type(module, loom_vector_dot2f_lhs(op),
                                       &lhs_type) ||
      !loom_contract_vector_value_type(module, loom_vector_dot2f_rhs(op),
                                       &rhs_type) ||
      !loom_contract_vector_value_type(module, loom_vector_dot2f_acc(op),
                                       &accumulator_type) ||
      !loom_contract_vector_value_type(module, loom_vector_dot2f_result(op),
                                       &result_type)) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_INVALID_REQUEST,
                                     out_diagnostic);
  }

  loom_contract_numeric_type_t source_numeric_type =
      LOOM_CONTRACT_NUMERIC_UNKNOWN;
  loom_contract_numeric_type_t rhs_numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  loom_contract_numeric_type_t accumulator_numeric_type =
      LOOM_CONTRACT_NUMERIC_UNKNOWN;
  loom_contract_numeric_type_t result_numeric_type =
      LOOM_CONTRACT_NUMERIC_UNKNOWN;
  if (!loom_contract_numeric_type_from_scalar(loom_type_element_type(lhs_type),
                                              false, &source_numeric_type) ||
      !loom_contract_numeric_type_from_scalar(loom_type_element_type(rhs_type),
                                              false, &rhs_numeric_type) ||
      source_numeric_type != rhs_numeric_type ||
      !loom_contract_numeric_type_from_scalar(
          loom_type_element_type(accumulator_type), false,
          &accumulator_numeric_type) ||
      !loom_contract_numeric_type_from_scalar(
          loom_type_element_type(result_type), false, &result_numeric_type) ||
      accumulator_numeric_type != LOOM_CONTRACT_NUMERIC_F32 ||
      result_numeric_type != LOOM_CONTRACT_NUMERIC_F32) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_NUMERIC,
                                     out_diagnostic);
  }
  if (!loom_type_shape_equals(lhs_type, rhs_type) ||
      !loom_type_shape_equals(accumulator_type, result_type) ||
      !loom_contract_vector_static_grouped_shapes_match(lhs_type, result_type,
                                                        2)) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_SHAPE,
                                     out_diagnostic);
  }

  uint64_t input_lane_count = 0;
  uint64_t rhs_lane_count = 0;
  uint64_t result_lane_count = 0;
  uint64_t accumulator_lane_count = 0;
  if (!loom_contract_vector_static_element_count(lhs_type, &input_lane_count) ||
      !loom_contract_vector_static_element_count(rhs_type, &rhs_lane_count) ||
      !loom_contract_vector_static_element_count(result_type,
                                                 &result_lane_count) ||
      !loom_contract_vector_static_element_count(accumulator_type,
                                                 &accumulator_lane_count) ||
      rhs_lane_count != input_lane_count ||
      accumulator_lane_count != result_lane_count) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_SHAPE,
                                     out_diagnostic);
  }

  request->kind = LOOM_CONTRACT_KIND_VECTOR_DOT;
  request->arithmetic = LOOM_CONTRACT_ARITHMETIC_FLOAT_DOT;
  request->lhs = loom_contract_vector_operand(LOOM_CONTRACT_OPERAND_ROLE_LHS,
                                              source_numeric_type);
  request->rhs = loom_contract_vector_operand(LOOM_CONTRACT_OPERAND_ROLE_RHS,
                                              source_numeric_type);
  request->accumulator = loom_contract_vector_operand(
      LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, LOOM_CONTRACT_NUMERIC_F32);
  request->result = loom_contract_vector_operand(
      LOOM_CONTRACT_OPERAND_ROLE_RESULT, LOOM_CONTRACT_NUMERIC_F32);
  if (!loom_contract_vector_populate_fragment(
          input_lane_count, result_lane_count, 16, 2, request)) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_SHAPE,
                                     out_diagnostic);
  }
  return loom_contract_vector_finish(policy, request, out_diagnostic);
}

static bool loom_contract_request_from_dot4i_op(
    const loom_module_t* module, const loom_op_t* op,
    loom_lowering_policy_t policy, loom_contract_request_t* request,
    loom_contract_diagnostic_t* out_diagnostic) {
  if (!loom_vector_dot4i_isa(op)) {
    return false;
  }

  loom_contract_numeric_type_t lhs_numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  loom_contract_numeric_type_t rhs_numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  if (!loom_contract_vector_dot4i_numeric_types(
          (loom_vector_dot4i_kind_t)loom_vector_dot4i_kind(op),
          &lhs_numeric_type, &rhs_numeric_type)) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_NUMERIC,
                                     out_diagnostic);
  }

  loom_type_t lhs_type = loom_type_none();
  loom_type_t rhs_type = loom_type_none();
  loom_type_t accumulator_type = loom_type_none();
  loom_type_t result_type = loom_type_none();
  if (!loom_contract_vector_value_type(module, loom_vector_dot4i_lhs(op),
                                       &lhs_type) ||
      !loom_contract_vector_value_type(module, loom_vector_dot4i_rhs(op),
                                       &rhs_type) ||
      !loom_contract_vector_value_type(module, loom_vector_dot4i_acc(op),
                                       &accumulator_type) ||
      !loom_contract_vector_value_type(module, loom_vector_dot4i_result(op),
                                       &result_type)) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_INVALID_REQUEST,
                                     out_diagnostic);
  }
  if (loom_type_element_type(lhs_type) != LOOM_SCALAR_TYPE_I8 ||
      loom_type_element_type(rhs_type) != LOOM_SCALAR_TYPE_I8 ||
      loom_type_element_type(accumulator_type) != LOOM_SCALAR_TYPE_I32 ||
      loom_type_element_type(result_type) != LOOM_SCALAR_TYPE_I32) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_NUMERIC,
                                     out_diagnostic);
  }
  if (!loom_type_shape_equals(lhs_type, rhs_type) ||
      !loom_type_shape_equals(accumulator_type, result_type) ||
      !loom_contract_vector_static_grouped_shapes_match(lhs_type, result_type,
                                                        4)) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_SHAPE,
                                     out_diagnostic);
  }

  uint64_t input_lane_count = 0;
  uint64_t rhs_lane_count = 0;
  uint64_t result_lane_count = 0;
  uint64_t accumulator_lane_count = 0;
  if (!loom_contract_vector_static_element_count(lhs_type, &input_lane_count) ||
      !loom_contract_vector_static_element_count(rhs_type, &rhs_lane_count) ||
      !loom_contract_vector_static_element_count(result_type,
                                                 &result_lane_count) ||
      !loom_contract_vector_static_element_count(accumulator_type,
                                                 &accumulator_lane_count) ||
      rhs_lane_count != input_lane_count ||
      accumulator_lane_count != result_lane_count) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_SHAPE,
                                     out_diagnostic);
  }

  request->kind = LOOM_CONTRACT_KIND_VECTOR_DOT;
  request->arithmetic = LOOM_CONTRACT_ARITHMETIC_INTEGER_DOT;
  request->lhs = loom_contract_vector_operand(LOOM_CONTRACT_OPERAND_ROLE_LHS,
                                              lhs_numeric_type);
  request->rhs = loom_contract_vector_operand(LOOM_CONTRACT_OPERAND_ROLE_RHS,
                                              rhs_numeric_type);
  request->accumulator = loom_contract_vector_operand(
      LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, LOOM_CONTRACT_NUMERIC_I32);
  request->result = loom_contract_vector_operand(
      LOOM_CONTRACT_OPERAND_ROLE_RESULT, LOOM_CONTRACT_NUMERIC_I32);
  if (!loom_contract_vector_populate_fragment(
          input_lane_count, result_lane_count, 8, 4, request)) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_SHAPE,
                                     out_diagnostic);
  }
  return loom_contract_vector_finish(policy, request, out_diagnostic);
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

bool loom_contract_request_from_vector_dot_op(
    const loom_module_t* module, const loom_op_t* op,
    loom_lowering_policy_t policy, loom_contract_request_t* out_request,
    loom_contract_diagnostic_t* out_diagnostic) {
  loom_contract_request_initialize(out_request);
  if (out_diagnostic) {
    *out_diagnostic = (loom_contract_diagnostic_t){0};
  }

  if (loom_vector_dot2f_isa(op)) {
    return loom_contract_request_from_dot2f_op(module, op, policy, out_request,
                                               out_diagnostic);
  }
  if (loom_vector_dot4i_isa(op)) {
    return loom_contract_request_from_dot4i_op(module, op, policy, out_request,
                                               out_diagnostic);
  }
  return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_INVALID_REQUEST,
                                   out_diagnostic);
}

bool loom_contract_request_from_vector_mma_op(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_op_t* op, const loom_contract_vector_mma_options_t* options,
    loom_contract_request_t* out_request,
    loom_contract_diagnostic_t* out_diagnostic) {
  loom_contract_request_initialize(out_request);
  if (out_diagnostic) {
    *out_diagnostic = (loom_contract_diagnostic_t){0};
  }

  if (!loom_vector_mma_isa(op) || fact_table == NULL || options == NULL) {
    return loom_contract_vector_mma_fail(
        LOOM_CONTRACT_REJECTION_INVALID_REQUEST, out_diagnostic);
  }

  loom_vector_fragment_fact_t lhs_fragment = {0};
  loom_vector_fragment_fact_t rhs_fragment = {0};
  loom_vector_fragment_fact_t init_fragment = {0};
  if (!loom_contract_vector_query_fragment_fact(
          fact_table, loom_vector_mma_lhs(op),
          LOOM_VECTOR_FRAGMENT_ROLE_FLAG_LHS, &lhs_fragment) ||
      !loom_contract_vector_query_fragment_fact(
          fact_table, loom_vector_mma_rhs(op),
          LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RHS, &rhs_fragment) ||
      !loom_contract_vector_query_fragment_fact(
          fact_table, loom_vector_mma_init(op),
          LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
              LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT,
          &init_fragment)) {
    return loom_contract_vector_mma_fail(LOOM_CONTRACT_REJECTION_ROLE,
                                         out_diagnostic);
  }

  loom_contract_vector_shape_dimension_t lhs_m = {0};
  loom_contract_vector_shape_dimension_t lhs_k = {0};
  loom_contract_vector_shape_dimension_t rhs_k = {0};
  loom_contract_vector_shape_dimension_t rhs_n = {0};
  loom_contract_vector_shape_dimension_t init_m = {0};
  loom_contract_vector_shape_dimension_t init_n = {0};
  loom_contract_vector_shape_dimension_t m = {0};
  loom_contract_vector_shape_dimension_t n = {0};
  loom_contract_vector_shape_dimension_t k = {0};
  if (!loom_contract_vector_query_shape_dimension(
          fact_table, lhs_fragment.shape_value_ids[0], &lhs_m) ||
      !loom_contract_vector_query_shape_dimension(
          fact_table, lhs_fragment.shape_value_ids[1], &lhs_k) ||
      !loom_contract_vector_query_shape_dimension(
          fact_table, rhs_fragment.shape_value_ids[0], &rhs_k) ||
      !loom_contract_vector_query_shape_dimension(
          fact_table, rhs_fragment.shape_value_ids[1], &rhs_n) ||
      !loom_contract_vector_query_shape_dimension(
          fact_table, init_fragment.shape_value_ids[0], &init_m) ||
      !loom_contract_vector_query_shape_dimension(
          fact_table, init_fragment.shape_value_ids[1], &init_n) ||
      !loom_contract_vector_shape_dimensions_match(lhs_m, init_m, &m) ||
      !loom_contract_vector_shape_dimensions_match(rhs_n, init_n, &n) ||
      !loom_contract_vector_shape_dimensions_match(lhs_k, rhs_k, &k)) {
    return loom_contract_vector_mma_fail(LOOM_CONTRACT_REJECTION_SHAPE,
                                         out_diagnostic);
  }

  loom_contract_rejection_bits_t rejection_bits = LOOM_CONTRACT_REJECTION_NONE;
  if (!loom_contract_vector_operand_from_fragment(
          module, loom_vector_mma_lhs(op), lhs_fragment,
          LOOM_CONTRACT_OPERAND_ROLE_LHS, &out_request->lhs, &rejection_bits)) {
    return loom_contract_vector_mma_fail(rejection_bits, out_diagnostic);
  }
  if (!loom_contract_vector_operand_from_fragment(
          module, loom_vector_mma_rhs(op), rhs_fragment,
          LOOM_CONTRACT_OPERAND_ROLE_RHS, &out_request->rhs, &rejection_bits)) {
    return loom_contract_vector_mma_fail(rejection_bits, out_diagnostic);
  }
  if (!loom_contract_vector_operand_from_fragment(
          module, loom_vector_mma_init(op), init_fragment,
          LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, &out_request->accumulator,
          &rejection_bits)) {
    return loom_contract_vector_mma_fail(rejection_bits, out_diagnostic);
  }
  if (!loom_contract_vector_operand_from_fragment(
          module, loom_vector_mma_result(op), init_fragment,
          LOOM_CONTRACT_OPERAND_ROLE_RESULT, &out_request->result,
          &rejection_bits)) {
    return loom_contract_vector_mma_fail(rejection_bits, out_diagnostic);
  }

  out_request->kind = LOOM_CONTRACT_KIND_MATRIX_MULTIPLY;
  out_request->shape = (loom_contract_shape_t){
      .m = m.exact_value,
      .n = n.exact_value,
      .k = k.exact_value,
  };
  out_request->shape_value_refs = (loom_contract_shape_value_refs_t){
      .m = m.value_ref,
      .n = n.value_ref,
      .k = k.value_ref,
  };
  out_request->k_group_size = options->k_group_size;
  out_request->fragment = options->fragment;
  out_request->capability_class = options->capability_class;
  out_request->policy = options->policy;
  out_request->arithmetic = loom_contract_vector_mma_arithmetic(out_request);
  if (!loom_contract_request_validate(out_request, out_diagnostic)) {
    return false;
  }
  return true;
}
