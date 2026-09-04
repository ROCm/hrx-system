// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/contract.h"

#include "iree/testing/gtest.h"

namespace {

loom_contract_operand_t Operand(loom_contract_operand_role_t role,
                                loom_contract_numeric_type_t numeric_type) {
  return (loom_contract_operand_t){
      /*.role=*/role,
      /*.numeric_type=*/numeric_type,
  };
}

loom_value_fact_storage_schema_t EncodedTargetSchema() {
  loom_value_fact_storage_schema_t schema = {};
  schema.encoded_operand.element_format = LOOM_VALUE_FACT_NUMERIC_FORMAT_I8;
  schema.encoded_operand.payload_packing =
      LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT;
  schema.encoded_operand.payload_register_count = 1;
  schema.encoded_operand.payload_element_count = 4;
  return schema;
}

loom_value_fact_storage_schema_t ScaledEncodedTargetSchema() {
  loom_value_fact_storage_schema_t schema = EncodedTargetSchema();
  schema.encoded_operand.element_format =
      LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1;
  schema.encoded_operand.scale_topology =
      LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D;
  schema.encoded_operand.scale_group.element_count = 32;
  schema.encoded_operand.scale_group.shape[0] = 32;
  schema.encoded_operand.scale_operand_count = 1;
  return schema;
}

loom_contract_request_t CompletePackedDotRequest() {
  loom_contract_request_t request = {};
  loom_contract_request_initialize(&request);
  request.kind = LOOM_CONTRACT_KIND_MATRIX_MULTIPLY;
  request.arithmetic = LOOM_CONTRACT_ARITHMETIC_FLOAT_DOT;
  request.shape = {
      /*.m=*/8,
      /*.n=*/1,
      /*.k=*/16,
      /*.block_count=*/1,
  };
  request.k_group_size = 2;
  request.lhs =
      Operand(LOOM_CONTRACT_OPERAND_ROLE_LHS, LOOM_CONTRACT_NUMERIC_BF16);
  request.rhs =
      Operand(LOOM_CONTRACT_OPERAND_ROLE_RHS, LOOM_CONTRACT_NUMERIC_BF16);
  request.accumulator = Operand(LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR,
                                LOOM_CONTRACT_NUMERIC_F32);
  request.result =
      Operand(LOOM_CONTRACT_OPERAND_ROLE_RESULT, LOOM_CONTRACT_NUMERIC_F32);
  request.fragment.atom_bits = LOOM_CONTRACT_FRAGMENT_VECTOR_LANE;
  request.capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_CPU_PACKED_DOT;
  request.policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED;
  return request;
}

TEST(ContractTest, ValidatesCompletePackedDotRequest) {
  loom_contract_request_t request = CompletePackedDotRequest();

  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_TRUE(loom_contract_request_validate(&request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_NONE);
}

TEST(ContractTest, ValidatesDynamicShapeWithValueRefs) {
  loom_contract_request_t request = CompletePackedDotRequest();
  request.shape = {};
  request.shape.block_count = 1;
  request.shape_value_refs = {
      /*.m=*/loom_contract_value_ref_from_value_id(10),
      /*.n=*/loom_contract_value_ref_from_value_id(11),
      /*.k=*/loom_contract_value_ref_from_value_id(12),
      /*.block_count=*/loom_contract_value_ref_absent(),
      /*.k_group_size=*/loom_contract_value_ref_from_value_id(13),
  };
  request.k_group_size = 0;

  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_TRUE(loom_contract_request_validate(&request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_NONE);
}

TEST(ContractTest, TransposesMatrixProductRequest) {
  loom_contract_request_t request = CompletePackedDotRequest();
  request.shape.m = 8;
  request.shape.n = 4;
  request.shape_value_refs.m = loom_contract_value_ref_from_value_id(10);
  request.shape_value_refs.n = loom_contract_value_ref_from_value_id(11);
  request.lhs.numeric_type = LOOM_CONTRACT_NUMERIC_F16;
  request.lhs.payload_register_count = 2;
  request.lhs.encoded.available_auxiliary_operands =
      LOOM_CONTRACT_AUXILIARY_OPERAND_SCALE;
  request.lhs.encoded
      .auxiliary_value_refs[LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SCALE] =
      loom_contract_value_ref_from_value_id(20);
  request.rhs.numeric_type = LOOM_CONTRACT_NUMERIC_BF16;
  request.rhs.payload_register_count = 3;
  request.rhs.encoded.available_auxiliary_operands =
      LOOM_CONTRACT_AUXILIARY_OPERAND_SPARSE_METADATA;
  request.rhs.encoded.auxiliary_value_refs
      [LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SPARSE_METADATA] =
      loom_contract_value_ref_from_value_id(21);

  loom_contract_request_transpose_mn(&request);

  EXPECT_EQ(request.shape.m, 4);
  EXPECT_EQ(request.shape.n, 8);
  EXPECT_EQ(loom_contract_value_ref_value_id(request.shape_value_refs.m), 11u);
  EXPECT_EQ(loom_contract_value_ref_value_id(request.shape_value_refs.n), 10u);
  EXPECT_EQ(request.lhs.role, LOOM_CONTRACT_OPERAND_ROLE_LHS);
  EXPECT_EQ(request.lhs.numeric_type, LOOM_CONTRACT_NUMERIC_BF16);
  EXPECT_EQ(request.lhs.payload_register_count, 3);
  EXPECT_EQ(loom_contract_value_ref_value_id(
                request.lhs.encoded.auxiliary_value_refs
                    [LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SPARSE_METADATA]),
            21u);
  EXPECT_EQ(request.rhs.role, LOOM_CONTRACT_OPERAND_ROLE_RHS);
  EXPECT_EQ(request.rhs.numeric_type, LOOM_CONTRACT_NUMERIC_F16);
  EXPECT_EQ(request.rhs.payload_register_count, 2);
  EXPECT_EQ(
      loom_contract_value_ref_value_id(
          request.rhs.encoded
              .auxiliary_value_refs[LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SCALE]),
      20u);
}

TEST(ContractTest, RejectsDynamicShapeWithoutValueRef) {
  loom_contract_request_t request = CompletePackedDotRequest();
  request.shape.k = 0;

  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_validate(&request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_SHAPE);
}

TEST(ContractTest, RejectsNegativeShapeWithValueRef) {
  loom_contract_request_t request = CompletePackedDotRequest();
  request.shape.m = -1;
  request.shape_value_refs.m = loom_contract_value_ref_from_value_id(10);

  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_validate(&request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_SHAPE);
}

TEST(ContractTest, RejectsUnavailableRequiredCapability) {
  loom_contract_request_t request = CompletePackedDotRequest();
  request.lhs.encoded.target_schema = EncodedTargetSchema();
  request.lhs.encoded.required_capability_flags =
      LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS;

  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_validate(&request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_CAPABILITY);
}

TEST(ContractTest, RejectsMissingAuxiliaryOperands) {
  loom_contract_request_t request = CompletePackedDotRequest();
  request.lhs.encoded.target_schema = ScaledEncodedTargetSchema();
  request.lhs.encoded.required_auxiliary_operands =
      LOOM_CONTRACT_AUXILIARY_OPERAND_SCALE;
  request.lhs.encoded.available_capability_flags =
      LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS;
  request.lhs.encoded.required_capability_flags =
      LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS;

  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_validate(&request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_CONTRACT_REJECTION_AUXILIARY_OPERAND);
}

TEST(ContractTest, RejectsUnknownSchemaFactsSeparately) {
  loom_contract_request_t request = CompletePackedDotRequest();
  request.lhs.encoded.available_auxiliary_operands =
      LOOM_CONTRACT_AUXILIARY_OPERAND_SCALE;
  request.lhs.encoded
      .auxiliary_value_refs[LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SCALE] =
      loom_contract_value_ref_from_value_id(42);
  request.lhs.encoded.required_auxiliary_operands =
      LOOM_CONTRACT_AUXILIARY_OPERAND_SCALE;
  request.lhs.encoded.available_capability_flags =
      LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS;
  request.lhs.encoded.required_capability_flags =
      LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS;

  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_validate(&request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_SCHEMA);
}

TEST(ContractTest, RejectsAvailableAuxiliaryOperandWithoutValueRef) {
  loom_contract_request_t request = CompletePackedDotRequest();
  request.lhs.encoded.target_schema = ScaledEncodedTargetSchema();
  request.lhs.encoded.available_auxiliary_operands =
      LOOM_CONTRACT_AUXILIARY_OPERAND_SCALE;
  request.lhs.encoded.required_auxiliary_operands =
      LOOM_CONTRACT_AUXILIARY_OPERAND_SCALE;
  request.lhs.encoded.available_capability_flags =
      LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS;
  request.lhs.encoded.required_capability_flags =
      LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS;

  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_validate(&request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_CONTRACT_REJECTION_AUXILIARY_OPERAND);
}

TEST(ContractTest, RejectsAuxiliaryValueRefWithoutAvailableOperand) {
  loom_contract_request_t request = CompletePackedDotRequest();
  request.lhs.encoded.target_schema = ScaledEncodedTargetSchema();
  request.lhs.encoded
      .auxiliary_value_refs[LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SCALE] =
      loom_contract_value_ref_from_value_id(42);

  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_validate(&request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_CONTRACT_REJECTION_AUXILIARY_OPERAND);
}

TEST(ContractTest, RejectsMissingShapeRoleAndCapability) {
  loom_contract_request_t request = {};
  loom_contract_request_initialize(&request);
  request.kind = LOOM_CONTRACT_KIND_MATRIX_MULTIPLY;
  request.arithmetic = LOOM_CONTRACT_ARITHMETIC_INTEGER_DOT;
  request.shape = {
      /*.m=*/16,
      /*.n=*/16,
      /*.k=*/0,
      /*.block_count=*/1,
  };
  request.k_group_size = 4;
  request.lhs =
      Operand(LOOM_CONTRACT_OPERAND_ROLE_LHS, LOOM_CONTRACT_NUMERIC_U8);
  request.rhs =
      Operand(LOOM_CONTRACT_OPERAND_ROLE_RHS, LOOM_CONTRACT_NUMERIC_I8);
  request.accumulator =
      Operand(LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN, LOOM_CONTRACT_NUMERIC_I32);
  request.result =
      Operand(LOOM_CONTRACT_OPERAND_ROLE_RESULT, LOOM_CONTRACT_NUMERIC_I32);
  request.lhs.encoded.target_schema = EncodedTargetSchema();
  request.lhs.encoded.required_capability_flags =
      LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS;

  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_validate(&request, &diagnostic));
  EXPECT_TRUE(iree_any_bit_set(diagnostic.rejection_bits,
                               LOOM_CONTRACT_REJECTION_SHAPE));
  EXPECT_TRUE(iree_any_bit_set(diagnostic.rejection_bits,
                               LOOM_CONTRACT_REJECTION_ROLE));
  EXPECT_TRUE(iree_any_bit_set(diagnostic.rejection_bits,
                               LOOM_CONTRACT_REJECTION_CAPABILITY));
}

TEST(ContractTest, MapsScalarNumericTypes) {
  loom_contract_numeric_type_t numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  ASSERT_TRUE(loom_contract_numeric_type_from_scalar(LOOM_SCALAR_TYPE_I8, false,
                                                     &numeric_type));
  EXPECT_EQ(numeric_type, LOOM_CONTRACT_NUMERIC_I8);
  ASSERT_TRUE(loom_contract_numeric_type_from_scalar(LOOM_SCALAR_TYPE_I8, true,
                                                     &numeric_type));
  EXPECT_EQ(numeric_type, LOOM_CONTRACT_NUMERIC_U8);
  ASSERT_TRUE(loom_contract_numeric_type_from_scalar(LOOM_SCALAR_TYPE_BF16,
                                                     false, &numeric_type));
  EXPECT_EQ(numeric_type, LOOM_CONTRACT_NUMERIC_BF16);
  ASSERT_TRUE(loom_contract_numeric_type_from_scalar(LOOM_SCALAR_TYPE_F8E4M3,
                                                     false, &numeric_type));
  EXPECT_EQ(numeric_type, LOOM_CONTRACT_NUMERIC_FP8);
  ASSERT_TRUE(loom_contract_numeric_type_from_scalar(LOOM_SCALAR_TYPE_F8E5M2,
                                                     false, &numeric_type));
  EXPECT_EQ(numeric_type, LOOM_CONTRACT_NUMERIC_BF8);
}

TEST(ContractTest, ReportsNumericTypeBitWidths) {
  EXPECT_EQ(loom_contract_numeric_bit_width(LOOM_CONTRACT_NUMERIC_UNKNOWN), 0);
  EXPECT_EQ(loom_contract_numeric_bit_width(LOOM_CONTRACT_NUMERIC_I4), 4);
  EXPECT_EQ(loom_contract_numeric_bit_width(LOOM_CONTRACT_NUMERIC_FP6), 6);
  EXPECT_EQ(loom_contract_numeric_bit_width(LOOM_CONTRACT_NUMERIC_BF8), 8);
  EXPECT_EQ(loom_contract_numeric_bit_width(LOOM_CONTRACT_NUMERIC_BF16), 16);
  EXPECT_EQ(loom_contract_numeric_bit_width(LOOM_CONTRACT_NUMERIC_TF32), 32);
  EXPECT_EQ(loom_contract_numeric_bit_width(LOOM_CONTRACT_NUMERIC_F64), 64);
}

}  // namespace
