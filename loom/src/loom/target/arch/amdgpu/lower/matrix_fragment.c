// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/matrix_fragment.h"

#include <stdint.h>

#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/fact_table.h"

static bool loom_amdgpu_fragment_memory_exact_nonnegative_i64(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    int64_t* out_value) {
  return loom_amdgpu_value_facts_as_exact_non_negative_i64(
      loom_value_fact_table_lookup(fact_table, value_id), out_value);
}

bool loom_amdgpu_matrix_fragment_tile_shape_matches(
    const loom_value_fact_table_t* fact_table,
    loom_amdgpu_matrix_tile_shape_t shape, loom_contract_operand_role_t role,
    loom_value_id_t blocks, loom_value_id_t rows, loom_value_id_t columns) {
  int64_t block_count = 1;
  if (blocks != LOOM_VALUE_ID_INVALID &&
      !loom_amdgpu_fragment_memory_exact_nonnegative_i64(fact_table, blocks,
                                                         &block_count)) {
    return false;
  }
  int64_t row_count = 0;
  int64_t column_count = 0;
  if (!loom_amdgpu_fragment_memory_exact_nonnegative_i64(fact_table, rows,
                                                         &row_count) ||
      !loom_amdgpu_fragment_memory_exact_nonnegative_i64(fact_table, columns,
                                                         &column_count)) {
    return false;
  }

  if (block_count != shape.block_count) {
    return false;
  }
  switch (role) {
    case LOOM_CONTRACT_OPERAND_ROLE_LHS:
      return row_count == shape.result_row_count &&
             column_count == shape.reduction_count;
    case LOOM_CONTRACT_OPERAND_ROLE_RHS:
      return row_count == shape.reduction_count &&
             column_count == shape.result_column_count;
    case LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR:
    case LOOM_CONTRACT_OPERAND_ROLE_RESULT:
      return row_count == shape.result_row_count &&
             column_count == shape.result_column_count;
    case LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN:
    default:
      return false;
  }
}

bool loom_amdgpu_matrix_fragment_shape_matches(
    const loom_value_fact_table_t* fact_table,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role, loom_value_id_t blocks,
    loom_value_id_t rows, loom_value_id_t columns) {
  return loom_amdgpu_matrix_fragment_tile_shape_matches(
      fact_table, layout->tile_shape, role, blocks, rows, columns);
}

bool loom_amdgpu_matrix_fragment_role_is_result_like(
    loom_contract_operand_role_t role) {
  return role == LOOM_CONTRACT_OPERAND_ROLE_RESULT ||
         role == LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR;
}

bool loom_amdgpu_matrix_fragment_role_layout_uses_low_subword(
    const loom_matrix_fragment_role_layout_t* role_layout) {
  return role_layout != NULL && role_layout->coordinate_element_stride > 1;
}

uint16_t loom_amdgpu_matrix_fragment_payload_elements_per_register(
    const loom_matrix_fragment_role_layout_t* role_layout) {
  if (role_layout == NULL || role_layout->element_bit_count == 0 ||
      role_layout->element_bit_count > 32 ||
      (32 % role_layout->element_bit_count) != 0) {
    return 0;
  }
  return (uint16_t)(32 / role_layout->element_bit_count);
}

bool loom_amdgpu_matrix_fragment_role_layout_uses_packed_b16_elements(
    loom_contract_operand_role_t role,
    const loom_matrix_fragment_role_layout_t* role_layout) {
  return role_layout != NULL &&
         loom_amdgpu_matrix_fragment_role_is_result_like(role) &&
         role_layout->element_bit_count ==
             LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_BIT_COUNT &&
         loom_amdgpu_matrix_fragment_payload_elements_per_register(
             role_layout) == LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT &&
         role_layout->packed_element_axis == LOOM_MATRIX_FRAGMENT_AXIS_ROW;
}

bool loom_amdgpu_matrix_fragment_payload_matches_role_storage(
    loom_type_t payload_type, loom_scalar_type_t expected_element_type,
    const loom_matrix_fragment_role_layout_t* role_layout) {
  loom_amdgpu_vector_storage_t storage = {0};
  return role_layout != NULL &&
         loom_amdgpu_type_vector_storage(payload_type, &storage) &&
         storage.element_type == expected_element_type &&
         storage.element_bit_count == role_layout->element_bit_count &&
         storage.element_count == role_layout->payload_element_count &&
         storage.register_count == role_layout->register_count;
}

static bool loom_amdgpu_fragment_memory_scalar_type_from_numeric(
    loom_amdgpu_matrix_numeric_type_t numeric_type,
    loom_scalar_type_t* out_element_type) {
  *out_element_type = LOOM_SCALAR_TYPE_COUNT_;
  switch (numeric_type) {
    case LOOM_AMDGPU_MATRIX_NUMERIC_F64:
      *out_element_type = LOOM_SCALAR_TYPE_F64;
      return true;
    case LOOM_AMDGPU_MATRIX_NUMERIC_F16:
      *out_element_type = LOOM_SCALAR_TYPE_F16;
      return true;
    case LOOM_AMDGPU_MATRIX_NUMERIC_BF16:
      *out_element_type = LOOM_SCALAR_TYPE_BF16;
      return true;
    case LOOM_AMDGPU_MATRIX_NUMERIC_FP8:
      *out_element_type = LOOM_SCALAR_TYPE_F8E4M3;
      return true;
    case LOOM_AMDGPU_MATRIX_NUMERIC_BF8:
      *out_element_type = LOOM_SCALAR_TYPE_F8E5M2;
      return true;
    case LOOM_AMDGPU_MATRIX_NUMERIC_F32:
    case LOOM_AMDGPU_MATRIX_NUMERIC_XF32:
      *out_element_type = LOOM_SCALAR_TYPE_F32;
      return true;
    case LOOM_AMDGPU_MATRIX_NUMERIC_I8:
    case LOOM_AMDGPU_MATRIX_NUMERIC_IU8:
      *out_element_type = LOOM_SCALAR_TYPE_I8;
      return true;
    case LOOM_AMDGPU_MATRIX_NUMERIC_I32:
      *out_element_type = LOOM_SCALAR_TYPE_I32;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_descriptor_payload(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    loom_contract_operand_role_t role,
    loom_amdgpu_matrix_payload_shape_t* out_payload) {
  *out_payload = (loom_amdgpu_matrix_payload_shape_t){0};
  switch (role) {
    case LOOM_CONTRACT_OPERAND_ROLE_LHS:
      *out_payload = descriptor->lhs_payload;
      return true;
    case LOOM_CONTRACT_OPERAND_ROLE_RHS:
      *out_payload = descriptor->rhs_payload;
      return true;
    case LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR:
      *out_payload = descriptor->accumulator_payload;
      return true;
    case LOOM_CONTRACT_OPERAND_ROLE_RESULT:
      *out_payload = descriptor->result_payload;
      return true;
    case LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN:
    default:
      return false;
  }
}

bool loom_amdgpu_matrix_fragment_descriptor_role_storage(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    loom_contract_operand_role_t role,
    loom_amdgpu_matrix_fragment_role_storage_t* out_storage) {
  *out_storage = (loom_amdgpu_matrix_fragment_role_storage_t){0};
  out_storage->element_type = LOOM_SCALAR_TYPE_COUNT_;
  return loom_amdgpu_fragment_memory_descriptor_payload(
             descriptor, role, &out_storage->payload) &&
         loom_amdgpu_fragment_memory_scalar_type_from_numeric(
             out_storage->payload.numeric_type, &out_storage->element_type);
}

bool loom_amdgpu_matrix_fragment_contract_is_available(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_matrix_feature_bits_t feature_bits, uint32_t wave_size) {
  return loom_amdgpu_matrix_contract_is_available(descriptor, feature_bits,
                                                  wave_size) &&
         loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                            descriptor->low_descriptor_ref);
}

iree_host_size_t loom_amdgpu_matrix_fragment_contract_candidate_count(
    const loom_amdgpu_matrix_fragment_contract_candidates_t* candidates) {
  return candidates != NULL ? candidates->descriptor_count
                            : loom_amdgpu_matrix_contract_descriptor_count();
}

const loom_amdgpu_matrix_contract_descriptor_t*
loom_amdgpu_matrix_fragment_contract_candidate_at(
    const loom_amdgpu_matrix_fragment_contract_candidates_t* candidates,
    iree_host_size_t index) {
  return candidates != NULL ? candidates->descriptors[index]
                            : loom_amdgpu_matrix_contract_descriptor_at(index);
}

loom_amdgpu_matrix_feature_bits_t loom_amdgpu_matrix_fragment_feature_bits(
    const loom_amdgpu_target_facts_t* target_facts) {
  IREE_ASSERT(target_facts != NULL,
              "AMDGPU matrix fragments require AMDGPU target facts");
  loom_amdgpu_matrix_feature_bits_t feature_bits = 0;
  (void)loom_amdgpu_matrix_feature_bits_from_profile(
      target_facts->properties.processor->features.matrix, &feature_bits);
  return feature_bits;
}
