// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/contract_storage.h"

#include "iree/testing/gtest.h"

namespace {

loom_value_fact_storage_schema_t EncodedSchema(
    loom_value_fact_numeric_format_flags_t element_format,
    uint16_t scale_group_element_count,
    loom_value_fact_numeric_format_flags_t scale_format,
    uint16_t payload_register_count, uint16_t payload_element_count) {
  loom_value_fact_storage_schema_t schema = {};
  schema.encoded_operand.element_format = element_format;
  schema.encoded_operand.payload_packing =
      LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT;
  schema.encoded_operand.scale_format = scale_format;
  schema.encoded_operand.scale_topology =
      scale_group_element_count == 0 ? LOOM_VALUE_FACT_SCALE_TOPOLOGY_NONE
                                     : LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D;
  schema.encoded_operand.payload_register_count = payload_register_count;
  schema.encoded_operand.payload_element_count = payload_element_count;
  schema.encoded_operand.scale_group_element_count = scale_group_element_count;
  schema.encoded_operand.scale_operand_count =
      scale_group_element_count == 0 ? 0 : 1;
  if (scale_group_element_count != 0) {
    schema.encoded_operand.flags |=
        LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ZERO_SCALE_FALLBACK;
  }
  return schema;
}

loom_contract_view_payload_t PlainPayload(
    loom_contract_operand_role_t role,
    loom_contract_numeric_type_t numeric_type) {
  return (loom_contract_view_payload_t){
      /*.kind=*/LOOM_CONTRACT_VIEW_PAYLOAD_PLAIN_ELEMENT,
      /*.operand=*/
      (loom_contract_operand_t){
          /*.role=*/role,
          /*.numeric_type=*/numeric_type,
      },
  };
}

void AttachAvailableAuxiliaryData(loom_contract_encoded_operand_t* encoded) {
  encoded->available_auxiliary_operands = encoded->required_auxiliary_operands;
  for (uint8_t i = 0; i < LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_COUNT_; ++i) {
    loom_contract_auxiliary_operand_key_t key =
        (loom_contract_auxiliary_operand_key_t)i;
    if (!iree_any_bit_set(encoded->required_auxiliary_operands,
                          loom_contract_auxiliary_operand_key_flag(key))) {
      continue;
    }
    encoded->auxiliary_value_refs[key] =
        loom_contract_value_ref_from_value_id(100 + i);
  }
}

loom_contract_view_payload_t MatrixPayload(
    loom_contract_operand_role_t role,
    loom_value_fact_storage_schema_t schema) {
  loom_contract_view_payload_t payload = {
      /*.kind=*/LOOM_CONTRACT_VIEW_PAYLOAD_UNSUPPORTED_STORAGE_SCHEMA,
  };
  if (!loom_contract_operand_from_storage_schema(role, schema,
                                                 &payload.operand)) {
    return payload;
  }
  payload.kind = LOOM_CONTRACT_VIEW_PAYLOAD_ENCODED_OPERAND_SCHEMA;
  return payload;
}

loom_contract_view_payload_t MatrixPayloadWithAuxiliaryData(
    loom_contract_operand_role_t role,
    loom_value_fact_storage_schema_t schema) {
  loom_contract_view_payload_t payload = MatrixPayload(role, schema);
  AttachAvailableAuxiliaryData(&payload.operand.encoded);
  return payload;
}

TEST(ContractStorageTest, MapsMatrixStorageSchemaToGenericOperand) {
  loom_value_fact_storage_schema_t schema =
      EncodedSchema(LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E3M2, 32,
                    LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE, 6, 32);

  loom_contract_operand_t operand = {};
  ASSERT_TRUE(loom_contract_operand_from_storage_schema(
      LOOM_CONTRACT_OPERAND_ROLE_LHS, schema, &operand));
  EXPECT_EQ(operand.role, LOOM_CONTRACT_OPERAND_ROLE_LHS);
  EXPECT_EQ(operand.numeric_type, LOOM_CONTRACT_NUMERIC_FP6);
  EXPECT_EQ(operand.payload_register_count, 6);
  EXPECT_EQ(operand.payload_element_count, 32);
  EXPECT_EQ(operand.encoded.source_schema.encoded_operand.element_format,
            LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E3M2);
  EXPECT_TRUE(iree_any_bit_set(operand.encoded.required_auxiliary_operands,
                               LOOM_CONTRACT_AUXILIARY_OPERAND_SCALE));
  EXPECT_EQ(operand.encoded.available_auxiliary_operands, 0u);

  loom_contract_scale_kind_t scale_kind = LOOM_CONTRACT_SCALE_UNKNOWN;
  ASSERT_TRUE(
      loom_contract_scale_kind_from_storage_schema(schema, &scale_kind));
  EXPECT_EQ(scale_kind, LOOM_CONTRACT_SCALE_32);

  const loom_contract_capability_flags_t available_flags =
      loom_contract_available_capability_flags_from_storage_schema(schema);
  EXPECT_TRUE(iree_any_bit_set(available_flags,
                               LOOM_CONTRACT_CAPABILITY_FORMAT_SELECTORS));
  EXPECT_TRUE(iree_any_bit_set(available_flags,
                               LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS));
  EXPECT_TRUE(iree_any_bit_set(available_flags,
                               LOOM_CONTRACT_CAPABILITY_ZERO_SCALE_FALLBACK));
  EXPECT_FALSE(iree_any_bit_set(
      available_flags, LOOM_CONTRACT_CAPABILITY_SCALE_FORMAT_SELECTORS));

  const loom_contract_capability_flags_t required_flags =
      loom_contract_required_capability_flags_from_storage_schema(schema);
  EXPECT_TRUE(iree_any_bit_set(required_flags,
                               LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS));
  EXPECT_TRUE(iree_any_bit_set(required_flags,
                               LOOM_CONTRACT_CAPABILITY_FORMAT_SELECTORS));
  EXPECT_FALSE(iree_any_bit_set(required_flags,
                                LOOM_CONTRACT_CAPABILITY_ZERO_SCALE_FALLBACK));
}

TEST(ContractStorageTest, RejectsUnknownMatrixFormat) {
  loom_value_fact_storage_schema_t schema = {};

  loom_contract_operand_t operand = {};
  EXPECT_FALSE(loom_contract_operand_from_storage_schema(
      LOOM_CONTRACT_OPERAND_ROLE_LHS, schema, &operand));
  EXPECT_EQ(operand.numeric_type, LOOM_CONTRACT_NUMERIC_UNKNOWN);
}

TEST(ContractStorageTest, QueriesPlainViewPayloadFromElementType) {
  loom_type_t view_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I8, loom_dim_pack_static(32), 0);

  loom_contract_view_payload_t payload = {};
  ASSERT_TRUE(loom_contract_view_payload_from_type(
      nullptr, nullptr, view_type, LOOM_CONTRACT_OPERAND_ROLE_LHS,
      /*plain_integer_is_unsigned=*/true, &payload));
  EXPECT_EQ(payload.kind, LOOM_CONTRACT_VIEW_PAYLOAD_PLAIN_ELEMENT);
  EXPECT_EQ(payload.operand.role, LOOM_CONTRACT_OPERAND_ROLE_LHS);
  EXPECT_EQ(payload.operand.numeric_type, LOOM_CONTRACT_NUMERIC_U8);
  EXPECT_EQ(payload.operand.encoded.available_capability_flags, 0u);
}

TEST(ContractStorageTest, BuildsMatrixRequestFromPayloadFacts) {
  loom_value_fact_storage_schema_t lhs_schema =
      EncodedSchema(LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E3M2, 32,
                    LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE, 6, 32);
  loom_value_fact_storage_schema_t rhs_schema =
      EncodedSchema(LOOM_VALUE_FACT_NUMERIC_FORMAT_BF6, 32,
                    LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE, 6, 32);

  loom_contract_matrix_request_options_t options = {};
  options.shape = (loom_contract_shape_t){/*.m=*/16, /*.n=*/16, /*.k=*/128};
  options.k_group_size = 1;
  options.lhs = MatrixPayloadWithAuxiliaryData(LOOM_CONTRACT_OPERAND_ROLE_LHS,
                                               lhs_schema);
  options.rhs = MatrixPayloadWithAuxiliaryData(LOOM_CONTRACT_OPERAND_ROLE_RHS,
                                               rhs_schema);
  options.accumulator_numeric_type = LOOM_CONTRACT_NUMERIC_F32;
  options.result_numeric_type = LOOM_CONTRACT_NUMERIC_F32;
  options.arithmetic = LOOM_CONTRACT_ARITHMETIC_MIXED_DOT;
  options.fragment = (loom_contract_fragment_t){
      /*.atom_bits=*/LOOM_CONTRACT_FRAGMENT_SUBGROUP_LANE,
      /*.vector_bit_width=*/{},
      /*.source_lane_count=*/{},
      /*.result_lane_count=*/{},
      /*.subgroup_size=*/64,
  };
  options.capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX;
  options.policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED;

  loom_contract_request_t request = {};
  loom_contract_diagnostic_t diagnostic = {};
  ASSERT_TRUE(loom_contract_request_from_matrix_payloads(&options, &request,
                                                         &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_NONE);
  EXPECT_EQ(request.kind, LOOM_CONTRACT_KIND_MATRIX_MULTIPLY);
  EXPECT_EQ(request.shape.m, 16);
  EXPECT_EQ(request.shape.n, 16);
  EXPECT_EQ(request.shape.k, 128);
  EXPECT_EQ(request.lhs.numeric_type, LOOM_CONTRACT_NUMERIC_FP6);
  EXPECT_EQ(request.lhs.payload_register_count, 6);
  EXPECT_EQ(request.rhs.numeric_type, LOOM_CONTRACT_NUMERIC_BF6);
  EXPECT_EQ(request.accumulator.numeric_type, LOOM_CONTRACT_NUMERIC_F32);
  EXPECT_EQ(request.result.numeric_type, LOOM_CONTRACT_NUMERIC_F32);
  EXPECT_TRUE(iree_all_bits_set(
      loom_contract_request_available_capability_flags(&request),
      LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS |
          LOOM_CONTRACT_CAPABILITY_FORMAT_SELECTORS));
  EXPECT_TRUE(iree_all_bits_set(
      loom_contract_request_required_capability_flags(&request),
      LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS |
          LOOM_CONTRACT_CAPABILITY_FORMAT_SELECTORS));
}

TEST(ContractStorageTest, BuildsMatrixRequestWithDynamicShapeRefs) {
  loom_contract_matrix_request_options_t options = {};
  options.shape = (loom_contract_shape_t){/*.m=*/0, /*.n=*/16, /*.k=*/128};
  options.shape_value_refs.m = loom_contract_value_ref_from_value_id(42);
  options.k_group_size = 1;
  options.lhs =
      PlainPayload(LOOM_CONTRACT_OPERAND_ROLE_LHS, LOOM_CONTRACT_NUMERIC_F16);
  options.rhs =
      PlainPayload(LOOM_CONTRACT_OPERAND_ROLE_RHS, LOOM_CONTRACT_NUMERIC_F16);
  options.accumulator_numeric_type = LOOM_CONTRACT_NUMERIC_F32;
  options.result_numeric_type = LOOM_CONTRACT_NUMERIC_F32;
  options.arithmetic = LOOM_CONTRACT_ARITHMETIC_FLOAT_DOT;
  options.fragment = (loom_contract_fragment_t){
      /*.atom_bits=*/LOOM_CONTRACT_FRAGMENT_SUBGROUP_LANE,
      /*.vector_bit_width=*/{},
      /*.source_lane_count=*/{},
      /*.result_lane_count=*/{},
      /*.subgroup_size=*/64,
  };
  options.capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX;
  options.policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED;

  loom_contract_request_t request = {};
  loom_contract_diagnostic_t diagnostic = {};
  ASSERT_TRUE(loom_contract_request_from_matrix_payloads(&options, &request,
                                                         &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_NONE);
  EXPECT_EQ(request.shape.m, 0);
  EXPECT_EQ(loom_contract_value_ref_value_id(request.shape_value_refs.m), 42u);
  EXPECT_EQ(request.shape.n, 16);
  EXPECT_EQ(request.shape.k, 128);
}

TEST(ContractStorageTest, BuildsPackedDotRequestFromPlainPayloadFacts) {
  loom_contract_matrix_request_options_t options = {};
  options.shape = (loom_contract_shape_t){/*.m=*/8, /*.n=*/1, /*.k=*/32};
  options.k_group_size = 4;
  options.lhs =
      PlainPayload(LOOM_CONTRACT_OPERAND_ROLE_LHS, LOOM_CONTRACT_NUMERIC_U8);
  options.rhs =
      PlainPayload(LOOM_CONTRACT_OPERAND_ROLE_RHS, LOOM_CONTRACT_NUMERIC_I8);
  options.accumulator_numeric_type = LOOM_CONTRACT_NUMERIC_I32;
  options.result_numeric_type = LOOM_CONTRACT_NUMERIC_I32;
  options.arithmetic = LOOM_CONTRACT_ARITHMETIC_INTEGER_DOT;
  options.fragment = (loom_contract_fragment_t){
      /*.atom_bits=*/LOOM_CONTRACT_FRAGMENT_VECTOR_LANE,
      /*.vector_bit_width=*/256,
      /*.source_lane_count=*/32,
      /*.result_lane_count=*/8,
  };
  options.capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_CPU_PACKED_DOT;
  options.policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED;

  loom_contract_request_t request = {};
  ASSERT_TRUE(
      loom_contract_request_from_matrix_payloads(&options, &request, nullptr));
  EXPECT_EQ(request.kind, LOOM_CONTRACT_KIND_MATRIX_MULTIPLY);
  EXPECT_EQ(request.arithmetic, LOOM_CONTRACT_ARITHMETIC_INTEGER_DOT);
  EXPECT_EQ(request.k_group_size, 4);
  EXPECT_EQ(request.lhs.numeric_type, LOOM_CONTRACT_NUMERIC_U8);
  EXPECT_EQ(request.rhs.numeric_type, LOOM_CONTRACT_NUMERIC_I8);
  EXPECT_EQ(request.fragment.atom_bits, LOOM_CONTRACT_FRAGMENT_VECTOR_LANE);
  EXPECT_EQ(request.fragment.vector_bit_width, 256);
}

TEST(ContractStorageTest, RejectsUnsupportedPayloadForOptimizedContract) {
  loom_contract_matrix_request_options_t options = {};
  options.shape = (loom_contract_shape_t){/*.m=*/16, /*.n=*/16, /*.k=*/128};
  options.k_group_size = 1;
  options.lhs.kind = LOOM_CONTRACT_VIEW_PAYLOAD_UNSUPPORTED_STORAGE_SCHEMA;
  options.rhs =
      PlainPayload(LOOM_CONTRACT_OPERAND_ROLE_RHS, LOOM_CONTRACT_NUMERIC_I8);
  options.accumulator_numeric_type = LOOM_CONTRACT_NUMERIC_I32;
  options.result_numeric_type = LOOM_CONTRACT_NUMERIC_I32;
  options.arithmetic = LOOM_CONTRACT_ARITHMETIC_MIXED_DOT;
  options.fragment.atom_bits = LOOM_CONTRACT_FRAGMENT_SUBGROUP_LANE;
  options.capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX;
  options.policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED;

  loom_contract_request_t request = {};
  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_from_matrix_payloads(&options, &request,
                                                          &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_NUMERIC);
}

TEST(ContractStorageTest, RejectsMissingAuxiliaryDataOperands) {
  loom_value_fact_storage_schema_t lhs_schema =
      EncodedSchema(LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E3M2, 32,
                    LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE, 6, 32);
  loom_value_fact_storage_schema_t rhs_schema =
      EncodedSchema(LOOM_VALUE_FACT_NUMERIC_FORMAT_BF6, 32,
                    LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE, 6, 32);

  loom_contract_matrix_request_options_t options = {};
  options.shape = (loom_contract_shape_t){/*.m=*/16, /*.n=*/16, /*.k=*/128};
  options.k_group_size = 1;
  options.lhs = MatrixPayload(LOOM_CONTRACT_OPERAND_ROLE_LHS, lhs_schema);
  options.rhs = MatrixPayload(LOOM_CONTRACT_OPERAND_ROLE_RHS, rhs_schema);
  options.accumulator_numeric_type = LOOM_CONTRACT_NUMERIC_F32;
  options.result_numeric_type = LOOM_CONTRACT_NUMERIC_F32;
  options.arithmetic = LOOM_CONTRACT_ARITHMETIC_MIXED_DOT;
  options.fragment.atom_bits = LOOM_CONTRACT_FRAGMENT_SUBGROUP_LANE;
  options.capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX;
  options.policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED;

  loom_contract_request_t request = {};
  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_from_matrix_payloads(&options, &request,
                                                          &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_CONTRACT_REJECTION_AUXILIARY_OPERAND);
}

}  // namespace
