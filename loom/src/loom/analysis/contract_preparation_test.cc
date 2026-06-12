// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/contract_preparation.h"

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

loom_value_fact_storage_schema_t BlockQuantSchema(uint16_t static_schema_id) {
  loom_value_fact_storage_schema_t schema = {};
  schema.static_spec_encoding_id = static_schema_id;
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

loom_contract_view_payload_t BlockQuantPayload(
    loom_contract_operand_role_t role,
    loom_value_fact_storage_schema_t schema) {
  return (loom_contract_view_payload_t){
      /*.kind=*/LOOM_CONTRACT_VIEW_PAYLOAD_UNSUPPORTED_STORAGE_SCHEMA,
      /*.operand=*/
      (loom_contract_operand_t){
          /*.role=*/role,
          /*.numeric_type=*/{},
          /*.payload_register_count=*/{},
          /*.payload_element_count=*/{},
          /*.encoded=*/
          (loom_contract_encoded_operand_t){
              /*.source_schema=*/schema,
          },
      },
  };
}

loom_value_fact_address_layout_t StridedLayout(
    const loom_value_facts_t* strides, uint8_t rank) {
  return (loom_value_fact_address_layout_t){
      /*.kind=*/LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
      /*.rank=*/rank,
      /*.strides=*/strides,
  };
}

loom_contract_operand_preparation_options_t BaseRhsOptions(
    loom_contract_view_payload_t payload,
    loom_value_fact_address_layout_t address_layout) {
  return (loom_contract_operand_preparation_options_t){
      /*.role=*/LOOM_CONTRACT_OPERAND_ROLE_RHS,
      /*.family=*/LOOM_CONTRACT_PREPARATION_FAMILY_MMT4D_RHS_N_MAJOR_BLOCKED,
      /*.availability=*/LOOM_CONTRACT_PREPARATION_AVAILABILITY_AVAILABLE,
      /*.policy=*/LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED,
      /*.source_payload=*/payload,
      /*.logical_flags=*/{},
      /*.address_layout=*/address_layout,
      /*.numeric_transform=*/LOOM_CONTRACT_NUMERIC_TRANSFORM_NONE,
  };
}

loom_contract_matrix_request_options_t BaseMatrixOptions(
    loom_contract_view_payload_t lhs, loom_contract_view_payload_t rhs) {
  return (loom_contract_matrix_request_options_t){
      /*.shape=*/(loom_contract_shape_t){
          /*.m=*/16,
          /*.n=*/16,
          /*.k=*/64,
      },
      /*.shape_value_refs=*/{},
      /*.k_group_size=*/4,
      /*.lhs=*/lhs,
      /*.rhs=*/rhs,
      /*.accumulator_numeric_type=*/LOOM_CONTRACT_NUMERIC_I32,
      /*.result_numeric_type=*/LOOM_CONTRACT_NUMERIC_I32,
      /*.arithmetic=*/LOOM_CONTRACT_ARITHMETIC_INTEGER_DOT,
      /*.fragment=*/{},
      /*.capability_class=*/{},
      /*.policy=*/LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED,
  };
}

TEST(ContractPreparationTest,
     DistinguishesLogicalTransposeFromPhysicalPacking) {
  const loom_value_facts_t rhs_strides[] = {
      loom_value_facts_exact_i64(1024),
      loom_value_facts_exact_i64(64),
      loom_value_facts_exact_i64(16),
      loom_value_facts_exact_i64(1),
  };
  loom_contract_operand_preparation_options_t options = BaseRhsOptions(
      PlainPayload(LOOM_CONTRACT_OPERAND_ROLE_RHS, LOOM_CONTRACT_NUMERIC_I8),
      StridedLayout(rhs_strides, IREE_ARRAYSIZE(rhs_strides)));
  options.logical_flags = LOOM_CONTRACT_PREPARATION_LOGICAL_RHS_TRANSPOSE;

  loom_contract_operand_preparation_t preparation = {};
  ASSERT_TRUE(loom_contract_operand_preparation_select(&options, &preparation,
                                                       nullptr));
  EXPECT_TRUE(iree_any_bit_set(
      preparation.flags, LOOM_CONTRACT_PREPARATION_LOGICAL_RHS_TRANSPOSE));
  EXPECT_TRUE(iree_any_bit_set(preparation.flags,
                               LOOM_CONTRACT_PREPARATION_PHYSICAL_PACKING));
  EXPECT_TRUE(iree_any_bit_set(
      preparation.flags, LOOM_CONTRACT_PREPARATION_PHYSICAL_N_MAJOR_BLOCKED));
  EXPECT_EQ(preparation.family,
            LOOM_CONTRACT_PREPARATION_FAMILY_MMT4D_RHS_N_MAJOR_BLOCKED);
  EXPECT_EQ(preparation.address_layout.rank, 4u);
  EXPECT_EQ(preparation.address_layout.strides[0].range_lo, 1024);
  EXPECT_EQ(preparation.numeric_transform,
            LOOM_CONTRACT_NUMERIC_TRANSFORM_NONE);
}

TEST(ContractPreparationTest, PreservesStorageSchemasThroughPacking) {
  const loom_value_facts_t rhs_strides[] = {
      loom_value_facts_exact_i64(256),
      loom_value_facts_exact_i64(32),
      loom_value_facts_exact_i64(8),
      loom_value_facts_exact_i64(1),
  };
  loom_value_fact_storage_schema_t matrix_schema =
      EncodedSchema(LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E3M2, 32,
                    LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE, 6, 32);
  loom_contract_operand_preparation_options_t options = BaseRhsOptions(
      MatrixPayload(LOOM_CONTRACT_OPERAND_ROLE_RHS, matrix_schema),
      StridedLayout(rhs_strides, IREE_ARRAYSIZE(rhs_strides)));

  loom_contract_operand_preparation_t preparation = {};
  ASSERT_TRUE(loom_contract_operand_preparation_select(&options, &preparation,
                                                       nullptr));
  EXPECT_TRUE(iree_any_bit_set(preparation.flags,
                               LOOM_CONTRACT_PREPARATION_HAS_ADDRESS_LAYOUT));
  EXPECT_TRUE(iree_any_bit_set(
      preparation.flags, LOOM_CONTRACT_PREPARATION_PRESERVES_STORAGE_SCHEMA));
  EXPECT_EQ(preparation.storage_schema.encoded_operand.element_format,
            LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E3M2);
  EXPECT_EQ(preparation.storage_schema.encoded_operand.payload_register_count,
            6u);

  loom_value_fact_storage_schema_t block_quant_schema = BlockQuantSchema(7);
  options = BaseRhsOptions(
      BlockQuantPayload(LOOM_CONTRACT_OPERAND_ROLE_RHS, block_quant_schema),
      StridedLayout(rhs_strides, IREE_ARRAYSIZE(rhs_strides)));
  ASSERT_TRUE(loom_contract_operand_preparation_select(&options, &preparation,
                                                       nullptr));
  EXPECT_EQ(preparation.storage_schema.static_spec_encoding_id, 7u);
  EXPECT_EQ(preparation.storage_schema.encoded_operand.element_format,
            LOOM_VALUE_FACT_NUMERIC_FORMAT_UNKNOWN);
  EXPECT_EQ(preparation.source_payload.kind,
            LOOM_CONTRACT_VIEW_PAYLOAD_UNSUPPORTED_STORAGE_SCHEMA);
}

TEST(ContractPreparationTest,
     KeepsNumericTransformSeparateFromPhysicalPacking) {
  loom_contract_operand_preparation_options_t transform_options = {};
  transform_options.role = LOOM_CONTRACT_OPERAND_ROLE_RHS;
  transform_options.family = LOOM_CONTRACT_PREPARATION_FAMILY_NUMERIC_TRANSFORM;
  transform_options.availability =
      LOOM_CONTRACT_PREPARATION_AVAILABILITY_AVAILABLE;
  transform_options.policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED;
  transform_options.source_payload =
      BlockQuantPayload(LOOM_CONTRACT_OPERAND_ROLE_RHS, BlockQuantSchema(9));
  transform_options.numeric_transform =
      LOOM_CONTRACT_NUMERIC_TRANSFORM_DECODE_REPACK;

  loom_contract_operand_preparation_t transform = {};
  ASSERT_TRUE(loom_contract_operand_preparation_select(&transform_options,
                                                       &transform, nullptr));
  EXPECT_EQ(transform.family,
            LOOM_CONTRACT_PREPARATION_FAMILY_NUMERIC_TRANSFORM);
  EXPECT_EQ(transform.numeric_transform,
            LOOM_CONTRACT_NUMERIC_TRANSFORM_DECODE_REPACK);
  EXPECT_FALSE(iree_any_bit_set(transform.flags,
                                LOOM_CONTRACT_PREPARATION_PHYSICAL_PACKING));

  const loom_value_facts_t rhs_strides[] = {
      loom_value_facts_exact_i64(512),
      loom_value_facts_exact_i64(32),
      loom_value_facts_exact_i64(8),
      loom_value_facts_exact_i64(1),
  };
  loom_contract_operand_preparation_options_t packing_options = BaseRhsOptions(
      PlainPayload(LOOM_CONTRACT_OPERAND_ROLE_RHS, LOOM_CONTRACT_NUMERIC_I8),
      StridedLayout(rhs_strides, IREE_ARRAYSIZE(rhs_strides)));
  packing_options.numeric_transform =
      LOOM_CONTRACT_NUMERIC_TRANSFORM_DECODE_REPACK;

  loom_contract_preparation_diagnostic_t diagnostic = {};
  loom_contract_operand_preparation_t packing = {};
  EXPECT_FALSE(loom_contract_operand_preparation_select(&packing_options,
                                                        &packing, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_CONTRACT_PREPARATION_REJECTION_NUMERIC_TRANSFORM);
}

TEST(ContractPreparationTest,
     OptimizedRequiredFailsWhenPreparationUnavailable) {
  const loom_value_facts_t rhs_strides[] = {
      loom_value_facts_exact_i64(512),
      loom_value_facts_exact_i64(32),
      loom_value_facts_exact_i64(8),
      loom_value_facts_exact_i64(1),
  };
  loom_contract_operand_preparation_options_t options = BaseRhsOptions(
      PlainPayload(LOOM_CONTRACT_OPERAND_ROLE_RHS, LOOM_CONTRACT_NUMERIC_I8),
      StridedLayout(rhs_strides, IREE_ARRAYSIZE(rhs_strides)));
  options.availability = LOOM_CONTRACT_PREPARATION_AVAILABILITY_UNAVAILABLE;
  options.policy = LOOM_LOWERING_POLICY_OPTIMIZED_REQUIRED;

  loom_contract_preparation_diagnostic_t diagnostic = {};
  loom_contract_operand_preparation_t preparation = {};
  EXPECT_FALSE(loom_contract_operand_preparation_select(&options, &preparation,
                                                        &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_CONTRACT_PREPARATION_REJECTION_UNAVAILABLE |
                LOOM_CONTRACT_PREPARATION_REJECTION_POLICY);

  options.availability = LOOM_CONTRACT_PREPARATION_AVAILABILITY_TOO_LATE;
  diagnostic = {};
  EXPECT_FALSE(loom_contract_operand_preparation_select(&options, &preparation,
                                                        &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_CONTRACT_PREPARATION_REJECTION_TOO_LATE |
                LOOM_CONTRACT_PREPARATION_REJECTION_POLICY);
}

TEST(ContractPreparationTest,
     ReferencePolicyFallsBackWhenPreparationUnavailable) {
  const loom_value_facts_t rhs_strides[] = {
      loom_value_facts_exact_i64(512),
      loom_value_facts_exact_i64(32),
      loom_value_facts_exact_i64(8),
      loom_value_facts_exact_i64(1),
  };
  loom_contract_operand_preparation_options_t options = BaseRhsOptions(
      PlainPayload(LOOM_CONTRACT_OPERAND_ROLE_RHS, LOOM_CONTRACT_NUMERIC_I8),
      StridedLayout(rhs_strides, IREE_ARRAYSIZE(rhs_strides)));
  options.availability = LOOM_CONTRACT_PREPARATION_AVAILABILITY_UNAVAILABLE;
  options.policy = LOOM_LOWERING_POLICY_REFERENCE_ALLOWED;

  loom_contract_operand_preparation_t preparation = {};
  ASSERT_TRUE(loom_contract_operand_preparation_select(&options, &preparation,
                                                       nullptr));
  EXPECT_EQ(preparation.family, LOOM_CONTRACT_PREPARATION_FAMILY_NONE);
  EXPECT_FALSE(iree_any_bit_set(preparation.flags,
                                LOOM_CONTRACT_PREPARATION_PHYSICAL_PACKING));
}

TEST(ContractPreparationTest, SamePayloadsFeedCpuAndGpuPreparationFamilies) {
  loom_contract_view_payload_t lhs =
      PlainPayload(LOOM_CONTRACT_OPERAND_ROLE_LHS, LOOM_CONTRACT_NUMERIC_U8);
  loom_contract_view_payload_t rhs =
      PlainPayload(LOOM_CONTRACT_OPERAND_ROLE_RHS, LOOM_CONTRACT_NUMERIC_I8);
  loom_contract_matrix_request_options_t cpu_options =
      BaseMatrixOptions(lhs, rhs);
  cpu_options.fragment = (loom_contract_fragment_t){
      /*.atom_bits=*/LOOM_CONTRACT_FRAGMENT_VECTOR_LANE,
      /*.vector_bit_width=*/256,
      /*.source_lane_count=*/32,
      /*.result_lane_count=*/8,
  };
  cpu_options.capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_CPU_PACKED_DOT;

  loom_contract_matrix_request_options_t gpu_options =
      BaseMatrixOptions(lhs, rhs);
  gpu_options.fragment = (loom_contract_fragment_t){
      /*.atom_bits=*/LOOM_CONTRACT_FRAGMENT_SUBGROUP_LANE,
      /*.vector_bit_width=*/{},
      /*.source_lane_count=*/{},
      /*.result_lane_count=*/{},
      /*.subgroup_size=*/64,
  };
  gpu_options.capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX;

  loom_contract_request_t cpu_request = {};
  loom_contract_request_t gpu_request = {};
  ASSERT_TRUE(loom_contract_request_from_matrix_payloads(
      &cpu_options, &cpu_request, nullptr));
  ASSERT_TRUE(loom_contract_request_from_matrix_payloads(
      &gpu_options, &gpu_request, nullptr));
  EXPECT_EQ(cpu_request.shape.m, gpu_request.shape.m);
  EXPECT_EQ(cpu_request.shape.n, gpu_request.shape.n);
  EXPECT_EQ(cpu_request.shape.k, gpu_request.shape.k);
  EXPECT_EQ(cpu_request.lhs.numeric_type, gpu_request.lhs.numeric_type);
  EXPECT_EQ(cpu_request.rhs.numeric_type, gpu_request.rhs.numeric_type);
  EXPECT_EQ(cpu_request.capability_class,
            LOOM_CONTRACT_CAPABILITY_CLASS_CPU_PACKED_DOT);
  EXPECT_EQ(gpu_request.capability_class,
            LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX);

  const loom_value_facts_t rhs_strides[] = {
      loom_value_facts_exact_i64(512),
      loom_value_facts_exact_i64(32),
      loom_value_facts_exact_i64(8),
      loom_value_facts_exact_i64(1),
  };
  loom_contract_operand_preparation_options_t cpu_preparation_options =
      BaseRhsOptions(rhs,
                     StridedLayout(rhs_strides, IREE_ARRAYSIZE(rhs_strides)));
  loom_contract_operand_preparation_options_t gpu_preparation_options = {};
  gpu_preparation_options.role = LOOM_CONTRACT_OPERAND_ROLE_RHS;
  gpu_preparation_options.family =
      LOOM_CONTRACT_PREPARATION_FAMILY_SUBGROUP_MATRIX_FRAGMENT;
  gpu_preparation_options.availability =
      LOOM_CONTRACT_PREPARATION_AVAILABILITY_AVAILABLE;
  gpu_preparation_options.policy =
      LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED;
  gpu_preparation_options.source_payload = rhs;
  gpu_preparation_options.numeric_transform =
      LOOM_CONTRACT_NUMERIC_TRANSFORM_NONE;

  loom_contract_operand_preparation_t cpu_preparation = {};
  loom_contract_operand_preparation_t gpu_preparation = {};
  ASSERT_TRUE(loom_contract_operand_preparation_select(
      &cpu_preparation_options, &cpu_preparation, nullptr));
  ASSERT_TRUE(loom_contract_operand_preparation_select(
      &gpu_preparation_options, &gpu_preparation, nullptr));
  EXPECT_EQ(cpu_preparation.family,
            LOOM_CONTRACT_PREPARATION_FAMILY_MMT4D_RHS_N_MAJOR_BLOCKED);
  EXPECT_TRUE(
      iree_any_bit_set(cpu_preparation.flags,
                       LOOM_CONTRACT_PREPARATION_PHYSICAL_N_MAJOR_BLOCKED));
  EXPECT_EQ(gpu_preparation.family,
            LOOM_CONTRACT_PREPARATION_FAMILY_SUBGROUP_MATRIX_FRAGMENT);
  EXPECT_TRUE(iree_any_bit_set(gpu_preparation.flags,
                               LOOM_CONTRACT_PREPARATION_FRAGMENT_OWNERSHIP));
  EXPECT_FALSE(
      iree_any_bit_set(gpu_preparation.flags,
                       LOOM_CONTRACT_PREPARATION_PHYSICAL_N_MAJOR_BLOCKED));
}

TEST(ContractPreparationTest, RejectsMmt4dPreparationForNonRhsRole) {
  const loom_value_facts_t lhs_strides[] = {
      loom_value_facts_exact_i64(512),
      loom_value_facts_exact_i64(32),
      loom_value_facts_exact_i64(8),
      loom_value_facts_exact_i64(1),
  };
  loom_contract_operand_preparation_options_t options = {};
  options.role = LOOM_CONTRACT_OPERAND_ROLE_LHS;
  options.family = LOOM_CONTRACT_PREPARATION_FAMILY_MMT4D_RHS_N_MAJOR_BLOCKED;
  options.availability = LOOM_CONTRACT_PREPARATION_AVAILABILITY_AVAILABLE;
  options.policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED;
  options.source_payload =
      PlainPayload(LOOM_CONTRACT_OPERAND_ROLE_LHS, LOOM_CONTRACT_NUMERIC_I8);
  options.address_layout =
      StridedLayout(lhs_strides, IREE_ARRAYSIZE(lhs_strides));
  options.numeric_transform = LOOM_CONTRACT_NUMERIC_TRANSFORM_NONE;

  loom_contract_preparation_diagnostic_t diagnostic = {};
  loom_contract_operand_preparation_t preparation = {};
  EXPECT_FALSE(loom_contract_operand_preparation_select(&options, &preparation,
                                                        &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_CONTRACT_PREPARATION_REJECTION_ROLE);
}

TEST(ContractPreparationTest, RejectsMalformedPreparationFacts) {
  loom_contract_operand_preparation_options_t options = BaseRhsOptions(
      PlainPayload(LOOM_CONTRACT_OPERAND_ROLE_LHS, LOOM_CONTRACT_NUMERIC_I8),
      (loom_value_fact_address_layout_t){
          /*.kind=*/LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE,
      });
  options.logical_flags = LOOM_CONTRACT_PREPARATION_PHYSICAL_PACKING;

  loom_contract_preparation_diagnostic_t diagnostic = {};
  loom_contract_operand_preparation_t preparation = {};
  EXPECT_FALSE(loom_contract_operand_preparation_select(&options, &preparation,
                                                        &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_CONTRACT_PREPARATION_REJECTION_INVALID_REQUEST);

  options.logical_flags = 0;
  diagnostic = {};
  EXPECT_FALSE(loom_contract_operand_preparation_select(&options, &preparation,
                                                        &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_CONTRACT_PREPARATION_REJECTION_ROLE);

  options.source_payload =
      PlainPayload(LOOM_CONTRACT_OPERAND_ROLE_RHS, LOOM_CONTRACT_NUMERIC_I8);
  diagnostic = {};
  EXPECT_FALSE(loom_contract_operand_preparation_select(&options, &preparation,
                                                        &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_CONTRACT_PREPARATION_REJECTION_ADDRESS_LAYOUT);
}

}  // namespace
