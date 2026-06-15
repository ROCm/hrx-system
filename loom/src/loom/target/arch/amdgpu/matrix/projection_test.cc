// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/matrix/projection.h"

#include "iree/testing/gtest.h"
#include "loom/analysis/contract_storage.h"

namespace {

loom_contract_operand_t Operand(loom_contract_operand_role_t role,
                                loom_contract_numeric_type_t numeric_type) {
  return (loom_contract_operand_t){
      /*.role=*/role,
      /*.numeric_type=*/numeric_type,
  };
}

loom_value_fact_storage_schema_t EncodedSchema(
    loom_value_fact_numeric_format_flags_t element_format,
    uint16_t scale_group_element_count) {
  loom_value_fact_storage_schema_t schema = {};
  schema.encoded_operand.element_format = element_format;
  schema.encoded_operand.payload_packing =
      LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT;
  schema.encoded_operand.scale_topology =
      scale_group_element_count == 0 ? LOOM_VALUE_FACT_SCALE_TOPOLOGY_NONE
                                     : LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D;
  schema.encoded_operand.payload_register_count = 4;
  schema.encoded_operand.payload_element_count = 32;
  schema.encoded_operand.scale_group_element_count = scale_group_element_count;
  schema.encoded_operand.scale_operand_count =
      scale_group_element_count == 0 ? 0 : 1;
  return schema;
}

void AttachEncodedSchema(loom_value_fact_storage_schema_t schema,
                         loom_contract_operand_t* operand) {
  operand->payload_register_count =
      schema.encoded_operand.payload_register_count;
  operand->payload_element_count = schema.encoded_operand.payload_element_count;
  operand->encoded.source_schema = schema;
  operand->encoded.target_schema = schema;
  operand->encoded.required_auxiliary_operands =
      loom_contract_required_auxiliary_operands_from_storage_schema(schema);
  operand->encoded.available_auxiliary_operands =
      operand->encoded.required_auxiliary_operands;
  for (uint8_t i = 0; i < LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_COUNT_; ++i) {
    loom_contract_auxiliary_operand_key_t key =
        (loom_contract_auxiliary_operand_key_t)i;
    if (!iree_any_bit_set(operand->encoded.required_auxiliary_operands,
                          loom_contract_auxiliary_operand_key_flag(key))) {
      continue;
    }
    operand->encoded.auxiliary_value_refs[key] =
        loom_contract_value_ref_from_value_id(200 + i);
  }
  operand->encoded.available_capability_flags =
      loom_contract_available_capability_flags_from_storage_schema(schema);
  operand->encoded.required_capability_flags =
      loom_contract_required_capability_flags_from_storage_schema(schema);
}

loom_contract_request_t MatrixRequest(
    int64_t m, int64_t n, int64_t k,
    loom_contract_numeric_type_t lhs_numeric_type,
    loom_contract_numeric_type_t rhs_numeric_type,
    loom_contract_numeric_type_t accumulator_numeric_type,
    loom_contract_numeric_type_t result_numeric_type) {
  loom_contract_request_t request = {};
  loom_contract_request_initialize(&request);
  request.kind = LOOM_CONTRACT_KIND_MATRIX_MULTIPLY;
  request.arithmetic = LOOM_CONTRACT_ARITHMETIC_MIXED_DOT;
  request.shape = {
      /*.m=*/m,
      /*.n=*/n,
      /*.k=*/k,
  };
  request.k_group_size = 1;
  request.lhs = Operand(LOOM_CONTRACT_OPERAND_ROLE_LHS, lhs_numeric_type);
  request.rhs = Operand(LOOM_CONTRACT_OPERAND_ROLE_RHS, rhs_numeric_type);
  request.accumulator =
      Operand(LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, accumulator_numeric_type);
  request.result =
      Operand(LOOM_CONTRACT_OPERAND_ROLE_RESULT, result_numeric_type);
  request.fragment = {
      /*.atom_bits=*/LOOM_CONTRACT_FRAGMENT_SUBGROUP_LANE,
      /*.vector_bit_width=*/{},
      /*.source_lane_count=*/{},
      /*.result_lane_count=*/{},
      /*.subgroup_size=*/64,
  };
  request.capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX;
  request.policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED;
  return request;
}

TEST(MatrixContractProjectionTest, RejectsMissingSubgroupFragmentFacts) {
  loom_contract_request_t contract = MatrixRequest(
      16, 16, 16, LOOM_CONTRACT_NUMERIC_I8, LOOM_CONTRACT_NUMERIC_I8,
      LOOM_CONTRACT_NUMERIC_I32, LOOM_CONTRACT_NUMERIC_I32);
  contract.fragment.atom_bits = 0;

  loom_amdgpu_matrix_contract_match_request_t amdgpu_request = {};
  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_amdgpu_matrix_contract_match_request_from_contract(
      &contract, 0, 64, &amdgpu_request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_FRAGMENT);
}

TEST(MatrixContractProjectionTest, RejectsDynamicShapeValueRefs) {
  loom_contract_request_t contract = MatrixRequest(
      16, 16, 0, LOOM_CONTRACT_NUMERIC_I8, LOOM_CONTRACT_NUMERIC_I8,
      LOOM_CONTRACT_NUMERIC_I32, LOOM_CONTRACT_NUMERIC_I32);
  contract.shape_value_refs.k = loom_contract_value_ref_from_value_id(42);

  loom_contract_diagnostic_t generic_diagnostic = {};
  ASSERT_TRUE(loom_contract_request_validate(&contract, &generic_diagnostic));
  EXPECT_EQ(generic_diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_NONE);

  loom_amdgpu_matrix_contract_match_request_t amdgpu_request = {};
  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_amdgpu_matrix_contract_match_request_from_contract(
      &contract, 0, 64, &amdgpu_request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_SHAPE);
}

TEST(MatrixContractProjectionTest, ProjectsRoleLocalScaleFacts) {
  loom_contract_request_t contract = MatrixRequest(
      16, 16, 16, LOOM_CONTRACT_NUMERIC_FP4, LOOM_CONTRACT_NUMERIC_FP4,
      LOOM_CONTRACT_NUMERIC_F32, LOOM_CONTRACT_NUMERIC_F32);
  AttachEncodedSchema(EncodedSchema(LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1, 32),
                      &contract.lhs);
  AttachEncodedSchema(EncodedSchema(LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1, 32),
                      &contract.rhs);

  loom_amdgpu_matrix_contract_match_request_t amdgpu_request = {};
  loom_contract_diagnostic_t diagnostic = {};
  ASSERT_TRUE(loom_amdgpu_matrix_contract_match_request_from_contract(
      &contract, 0, 64, &amdgpu_request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_NONE);
  EXPECT_EQ(amdgpu_request.scale_kind, LOOM_AMDGPU_MATRIX_SCALE_32);
  EXPECT_TRUE(iree_any_bit_set(amdgpu_request.available_flags,
                               LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED));
  EXPECT_TRUE(
      iree_any_bit_set(amdgpu_request.required_flags,
                       LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS));
}

TEST(MatrixContractProjectionTest, ProjectsZeroScaleFallbackCapabilityFacts) {
  loom_contract_request_t contract = MatrixRequest(
      16, 16, 16, LOOM_CONTRACT_NUMERIC_FP4, LOOM_CONTRACT_NUMERIC_FP4,
      LOOM_CONTRACT_NUMERIC_F32, LOOM_CONTRACT_NUMERIC_F32);
  loom_value_fact_storage_schema_t lhs_schema =
      EncodedSchema(LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1, 32);
  lhs_schema.encoded_operand.flags |=
      LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ZERO_SCALE_FALLBACK;
  loom_value_fact_storage_schema_t rhs_schema =
      EncodedSchema(LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1, 32);
  rhs_schema.encoded_operand.flags |=
      LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ZERO_SCALE_FALLBACK;
  AttachEncodedSchema(lhs_schema, &contract.lhs);
  AttachEncodedSchema(rhs_schema, &contract.rhs);

  loom_amdgpu_matrix_contract_match_request_t amdgpu_request = {};
  loom_contract_diagnostic_t diagnostic = {};
  ASSERT_TRUE(loom_amdgpu_matrix_contract_match_request_from_contract(
      &contract, 0, 64, &amdgpu_request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_NONE);
  EXPECT_TRUE(
      iree_any_bit_set(amdgpu_request.available_flags,
                       LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK));
  EXPECT_FALSE(
      iree_any_bit_set(amdgpu_request.required_flags,
                       LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK));
}

TEST(MatrixContractProjectionTest, RejectsMismatchedRoleLocalScaleFacts) {
  loom_contract_request_t contract = MatrixRequest(
      16, 16, 16, LOOM_CONTRACT_NUMERIC_FP4, LOOM_CONTRACT_NUMERIC_FP4,
      LOOM_CONTRACT_NUMERIC_F32, LOOM_CONTRACT_NUMERIC_F32);
  AttachEncodedSchema(EncodedSchema(LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1, 32),
                      &contract.lhs);
  AttachEncodedSchema(EncodedSchema(LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1, 16),
                      &contract.rhs);

  loom_amdgpu_matrix_contract_match_request_t amdgpu_request = {};
  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_amdgpu_matrix_contract_match_request_from_contract(
      &contract, 0, 64, &amdgpu_request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_CAPABILITY);
}

}  // namespace
