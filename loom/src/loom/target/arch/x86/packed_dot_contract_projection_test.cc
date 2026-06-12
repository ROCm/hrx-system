// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/packed_dot_contract_projection.h"

#include "iree/testing/gtest.h"
#include "loom/analysis/contract_storage.h"

namespace {

loom_contract_arithmetic_t ArithmeticForAccumulator(
    loom_contract_numeric_type_t accumulator_numeric_type) {
  switch (accumulator_numeric_type) {
    case LOOM_CONTRACT_NUMERIC_F16:
    case LOOM_CONTRACT_NUMERIC_BF16:
    case LOOM_CONTRACT_NUMERIC_F32:
    case LOOM_CONTRACT_NUMERIC_F64:
      return LOOM_CONTRACT_ARITHMETIC_FLOAT_DOT;
    default:
      return LOOM_CONTRACT_ARITHMETIC_INTEGER_DOT;
  }
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

bool BuildPackedDotRequest(
    loom_contract_numeric_type_t lhs_numeric_type,
    loom_contract_numeric_type_t rhs_numeric_type,
    loom_contract_numeric_type_t accumulator_numeric_type,
    loom_contract_numeric_type_t result_numeric_type, uint16_t vector_bit_width,
    uint16_t source_lane_count, uint16_t result_lane_count,
    uint16_t k_group_size, loom_contract_request_t* out_request) {
  loom_contract_matrix_request_options_t options = {};
  options.shape = {
      /*.m=*/result_lane_count,
      /*.n=*/1,
      /*.k=*/source_lane_count,
  };
  options.k_group_size = k_group_size;
  options.lhs = PlainPayload(LOOM_CONTRACT_OPERAND_ROLE_LHS, lhs_numeric_type);
  options.rhs = PlainPayload(LOOM_CONTRACT_OPERAND_ROLE_RHS, rhs_numeric_type);
  options.accumulator_numeric_type = accumulator_numeric_type;
  options.result_numeric_type = result_numeric_type;
  options.arithmetic = ArithmeticForAccumulator(accumulator_numeric_type);
  options.fragment = {
      /*.atom_bits=*/LOOM_CONTRACT_FRAGMENT_VECTOR_LANE,
      /*.vector_bit_width=*/vector_bit_width,
      /*.source_lane_count=*/source_lane_count,
      /*.result_lane_count=*/result_lane_count,
  };
  options.capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_CPU_PACKED_DOT;
  options.policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED;
  return loom_contract_request_from_matrix_payloads(&options, out_request,
                                                    NULL);
}

TEST(PackedDotContractProjectionTest, RejectsMissingVectorFragmentFacts) {
  loom_contract_request_t contract = {};
  ASSERT_TRUE(BuildPackedDotRequest(
      LOOM_CONTRACT_NUMERIC_BF16, LOOM_CONTRACT_NUMERIC_BF16,
      LOOM_CONTRACT_NUMERIC_F32, LOOM_CONTRACT_NUMERIC_F32, 256, 16, 8, 2,
      &contract));
  contract.fragment.atom_bits = 0;

  loom_x86_packed_dot_match_request_t x86_request = {};
  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_x86_packed_dot_match_request_from_contract(
      &contract, 0, &x86_request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_FRAGMENT);
}

TEST(PackedDotContractProjectionTest, RejectsDynamicKGroupValueRef) {
  loom_contract_request_t contract = {};
  ASSERT_TRUE(BuildPackedDotRequest(
      LOOM_CONTRACT_NUMERIC_BF16, LOOM_CONTRACT_NUMERIC_BF16,
      LOOM_CONTRACT_NUMERIC_F32, LOOM_CONTRACT_NUMERIC_F32, 256, 16, 8, 2,
      &contract));
  contract.k_group_size = 0;
  contract.shape_value_refs.k_group_size =
      loom_contract_value_ref_from_value_id(42);

  loom_contract_diagnostic_t generic_diagnostic = {};
  ASSERT_TRUE(loom_contract_request_validate(&contract, &generic_diagnostic));
  EXPECT_EQ(generic_diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_NONE);

  loom_x86_packed_dot_match_request_t x86_request = {};
  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_x86_packed_dot_match_request_from_contract(
      &contract, 0, &x86_request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_SHAPE);
}

TEST(PackedDotContractProjectionTest, RejectsUnsupportedNumericType) {
  loom_contract_request_t contract = {};
  ASSERT_TRUE(BuildPackedDotRequest(
      LOOM_CONTRACT_NUMERIC_FP8, LOOM_CONTRACT_NUMERIC_BF8,
      LOOM_CONTRACT_NUMERIC_F32, LOOM_CONTRACT_NUMERIC_F32, 256, 16, 8, 2,
      &contract));

  loom_x86_packed_dot_match_request_t x86_request = {};
  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_x86_packed_dot_match_request_from_contract(
      &contract, 0, &x86_request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_NUMERIC);
}

}  // namespace
