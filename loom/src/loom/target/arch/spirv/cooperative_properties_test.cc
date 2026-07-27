// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/cooperative_properties.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"

namespace {

constexpr loom_spirv_feature_bits_t kF16CooperativeMatrixFeatures =
    LOOM_SPIRV_FEATURE_VULKAN_SHADER |
    LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR | LOOM_SPIRV_FEATURE_FLOAT16;

constexpr loom_spirv_feature_bits_t kBf16CooperativeMatrixFeatures =
    LOOM_SPIRV_FEATURE_VULKAN_SHADER |
    LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR |
    LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR |
    LOOM_SPIRV_FEATURE_BFLOAT16_COOPERATIVE_MATRIX_KHR;

constexpr loom_spirv_feature_bits_t kS8CooperativeMatrixFeatures =
    LOOM_SPIRV_FEATURE_VULKAN_SHADER |
    LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR | LOOM_SPIRV_FEATURE_INT8 |
    LOOM_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS;

constexpr loom_spirv_feature_bits_t kU8CooperativeMatrixFeatures =
    LOOM_SPIRV_FEATURE_VULKAN_SHADER |
    LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR | LOOM_SPIRV_FEATURE_INT8 |
    LOOM_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS;

void PreparePropertySet(loom_spirv_feature_bits_t feature_bits,
                        loom_spirv_cooperative_property_set_t* property_set) {
  loom_spirv_feature_set_t feature_set = {};
  IREE_ASSERT_OK(loom_spirv_feature_set_prepare(IREE_SV("test.spirv"),
                                                feature_bits, &feature_set));
  loom_spirv_cooperative_property_set_prepare(&feature_set, property_set);
}

loom_spirv_cooperative_matrix_query_t F16MatrixQuery(
    loom_lowering_policy_t policy) {
  return {
      /*.m_size=*/16,
      /*.n_size=*/16,
      /*.k_size=*/16,
      /*.lhs_type=*/LOOM_SPIRV_SCALAR_TYPE_F16,
      /*.rhs_type=*/LOOM_SPIRV_SCALAR_TYPE_F16,
      /*.accumulator_type=*/LOOM_SPIRV_SCALAR_TYPE_F32,
      /*.result_type=*/LOOM_SPIRV_SCALAR_TYPE_F32,
      /*.scope=*/LOOM_SPIRV_SCOPE_SUBGROUP,
      /*.layout=*/LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_KHR,
      /*.storage_class=*/LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER,
      /*.operand_flags=*/{},
      /*.policy=*/policy,
  };
}

loom_spirv_cooperative_matrix_query_t Bf16MatrixQuery(
    loom_lowering_policy_t policy) {
  return {
      /*.m_size=*/16,
      /*.n_size=*/16,
      /*.k_size=*/16,
      /*.lhs_type=*/LOOM_SPIRV_SCALAR_TYPE_BF16,
      /*.rhs_type=*/LOOM_SPIRV_SCALAR_TYPE_BF16,
      /*.accumulator_type=*/LOOM_SPIRV_SCALAR_TYPE_F32,
      /*.result_type=*/LOOM_SPIRV_SCALAR_TYPE_F32,
      /*.scope=*/LOOM_SPIRV_SCOPE_SUBGROUP,
      /*.layout=*/LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_KHR,
      /*.storage_class=*/LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER,
      /*.operand_flags=*/{},
      /*.policy=*/policy,
  };
}

loom_spirv_cooperative_matrix_query_t S8MatrixQuery(
    loom_lowering_policy_t policy) {
  return {
      /*.m_size=*/16,
      /*.n_size=*/16,
      /*.k_size=*/32,
      /*.lhs_type=*/LOOM_SPIRV_SCALAR_TYPE_S8,
      /*.rhs_type=*/LOOM_SPIRV_SCALAR_TYPE_S8,
      /*.accumulator_type=*/LOOM_SPIRV_SCALAR_TYPE_S32,
      /*.result_type=*/LOOM_SPIRV_SCALAR_TYPE_S32,
      /*.scope=*/LOOM_SPIRV_SCOPE_SUBGROUP,
      /*.layout=*/LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_KHR,
      /*.storage_class=*/LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER,
      /*.operand_flags=*/
      LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_A_SIGNED_COMPONENTS |
          LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_B_SIGNED_COMPONENTS |
          LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_C_SIGNED_COMPONENTS |
          LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_RESULT_SIGNED_COMPONENTS |
          LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_SATURATING_ACCUMULATION,
      /*.policy=*/policy,
  };
}

loom_spirv_cooperative_matrix_query_t U8MatrixQuery(
    loom_lowering_policy_t policy) {
  return {
      /*.m_size=*/16,
      /*.n_size=*/16,
      /*.k_size=*/32,
      /*.lhs_type=*/LOOM_SPIRV_SCALAR_TYPE_U8,
      /*.rhs_type=*/LOOM_SPIRV_SCALAR_TYPE_U8,
      /*.accumulator_type=*/LOOM_SPIRV_SCALAR_TYPE_U32,
      /*.result_type=*/LOOM_SPIRV_SCALAR_TYPE_U32,
      /*.scope=*/LOOM_SPIRV_SCOPE_SUBGROUP,
      /*.layout=*/LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_KHR,
      /*.storage_class=*/LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER,
      /*.operand_flags=*/{},
      /*.policy=*/policy,
  };
}

loom_spirv_cooperative_vector_query_t PackedS8VectorQuery(
    loom_lowering_policy_t policy) {
  return {
      /*.m_size=*/32,
      /*.k_size=*/32,
      /*.input_type=*/LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT32_NV,
      /*.input_interpretation=*/LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT8_PACKED_NV,
      /*.matrix_interpretation=*/LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT8_NV,
      /*.bias_interpretation=*/LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV,
      /*.result_type=*/LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV,
      /*.matrix_layout=*/
      LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV,
      /*.storage_class=*/LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER,
      /*.flags=*/{},
      /*.policy=*/policy,
  };
}

loom_spirv_cooperative_vector_query_t F8E4M3VectorQuery(
    loom_lowering_policy_t policy) {
  loom_spirv_component_type_t f8e4m3_component_type =
      LOOM_SPIRV_COMPONENT_TYPE_MAX;
  EXPECT_TRUE(loom_spirv_component_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3, &f8e4m3_component_type));
  return {
      /*.m_size=*/16,
      /*.k_size=*/32,
      /*.input_type=*/LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV,
      /*.input_interpretation=*/f8e4m3_component_type,
      /*.matrix_interpretation=*/f8e4m3_component_type,
      /*.bias_interpretation=*/LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV,
      /*.result_type=*/LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV,
      /*.matrix_layout=*/
      LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV,
      /*.storage_class=*/LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER,
      /*.flags=*/{},
      /*.policy=*/policy,
  };
}

TEST(SpirvCooperativePropertiesTest, MapsNumericFormatsToScalarTypes) {
  loom_spirv_scalar_type_t scalar_type = LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;
  EXPECT_TRUE(loom_spirv_scalar_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_F16, &scalar_type));
  EXPECT_EQ(scalar_type, LOOM_SPIRV_SCALAR_TYPE_F16);
  EXPECT_TRUE(loom_spirv_scalar_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_BF16, &scalar_type));
  EXPECT_EQ(scalar_type, LOOM_SPIRV_SCALAR_TYPE_BF16);
  EXPECT_TRUE(loom_spirv_scalar_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I8, &scalar_type));
  EXPECT_EQ(scalar_type, LOOM_SPIRV_SCALAR_TYPE_S8);
  EXPECT_TRUE(loom_spirv_scalar_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_U8, &scalar_type));
  EXPECT_EQ(scalar_type, LOOM_SPIRV_SCALAR_TYPE_U8);

  const loom_spirv_scalar_type_descriptor_t* descriptor =
      loom_spirv_scalar_type_descriptor(LOOM_SPIRV_SCALAR_TYPE_BF16);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->kind, LOOM_SPIRV_SCALAR_TYPE_KIND_FLOAT);
  EXPECT_EQ(descriptor->bit_width, 16);
  EXPECT_EQ(descriptor->fp_encoding, LOOM_SPIRV_FP_ENCODING_B_FLOAT16_KHR);
  EXPECT_EQ(descriptor->required_feature_bits,
            LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR);
}

TEST(SpirvCooperativePropertiesTest,
     RejectsUnknownAndMultiValuedScalarNumericFormats) {
  loom_spirv_scalar_type_t scalar_type = LOOM_SPIRV_SCALAR_TYPE_F16;
  EXPECT_FALSE(loom_spirv_scalar_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_UNKNOWN, &scalar_type));
  EXPECT_EQ(scalar_type, LOOM_SPIRV_SCALAR_TYPE_UNKNOWN);
  EXPECT_FALSE(loom_spirv_scalar_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_F16 | LOOM_VALUE_FACT_NUMERIC_FORMAT_BF16,
      &scalar_type));
  EXPECT_EQ(scalar_type, LOOM_SPIRV_SCALAR_TYPE_UNKNOWN);
  EXPECT_FALSE(loom_spirv_scalar_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3, &scalar_type));
  EXPECT_EQ(scalar_type, LOOM_SPIRV_SCALAR_TYPE_UNKNOWN);
  EXPECT_FALSE(loom_spirv_scalar_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_I4, &scalar_type));
  EXPECT_EQ(scalar_type, LOOM_SPIRV_SCALAR_TYPE_UNKNOWN);
}

TEST(SpirvCooperativePropertiesTest,
     MapsNumericFormatsToComponentEnumOperands) {
  loom_spirv_component_type_t component_type = LOOM_SPIRV_COMPONENT_TYPE_MAX;
  EXPECT_TRUE(loom_spirv_component_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_F16, &component_type));
  EXPECT_EQ(component_type, LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV);
  EXPECT_TRUE(loom_spirv_component_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I8, &component_type));
  EXPECT_EQ(component_type, LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT8_NV);
  EXPECT_TRUE(loom_spirv_component_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_U8, &component_type));
  EXPECT_EQ(component_type, LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT8_NV);
  EXPECT_TRUE(loom_spirv_component_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3, &component_type));
  EXPECT_EQ(component_type, LOOM_SPIRV_COMPONENT_TYPE_FLOAT_E4_M3_NV);
  EXPECT_TRUE(loom_spirv_component_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_BF8, &component_type));
  EXPECT_EQ(component_type, LOOM_SPIRV_COMPONENT_TYPE_FLOAT_E5_M2_NV);
}

TEST(SpirvCooperativePropertiesTest, RejectsBfloat16AsComponentEnumOperand) {
  loom_spirv_component_type_t component_type =
      LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV;
  EXPECT_FALSE(loom_spirv_component_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_UNKNOWN, &component_type));
  EXPECT_EQ(component_type, LOOM_SPIRV_COMPONENT_TYPE_MAX);
  EXPECT_FALSE(loom_spirv_component_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_BF16, &component_type));
  EXPECT_EQ(component_type, LOOM_SPIRV_COMPONENT_TYPE_MAX);
  EXPECT_FALSE(loom_spirv_component_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_F16 | LOOM_VALUE_FACT_NUMERIC_FORMAT_BF16,
      &component_type));
  EXPECT_EQ(component_type, LOOM_SPIRV_COMPONENT_TYPE_MAX);
  EXPECT_FALSE(loom_spirv_component_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_I4, &component_type));
  EXPECT_EQ(component_type, LOOM_SPIRV_COMPONENT_TYPE_MAX);
}

TEST(SpirvCooperativePropertiesTest, PreparesSelectedPropertyTables) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(LOOM_SPIRV_FEATURE_VULKAN_SHADER |
                         LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR |
                         LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV,
                     &property_set);

  EXPECT_TRUE(iree_all_bits_set(property_set.feature_bits,
                                LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR));
  EXPECT_TRUE(iree_all_bits_set(property_set.feature_bits,
                                LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV));
  EXPECT_GT(property_set.matrix_property_count, 0u);
  EXPECT_GT(property_set.matrix_shape_span_count, 0u);
  EXPECT_GT(property_set.vector_property_count, 0u);
  EXPECT_GT(property_set.vector_shape_span_count, 0u);
}

TEST(SpirvCooperativePropertiesTest, SelectsCooperativeMatrixByShapeAndFacts) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(kF16CooperativeMatrixFeatures, &property_set);
  const loom_spirv_cooperative_matrix_query_t query =
      F16MatrixQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  const loom_spirv_cooperative_matrix_property_t* property =
      loom_spirv_cooperative_matrix_property_select(&property_set, &query,
                                                    &diagnostic);

  ASSERT_NE(property, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      property->name,
      IREE_SV("khr.cooperative_matrix.f16.16x16x16.f32.subgroup")));
  EXPECT_EQ(
      property->mul_add_descriptor_ref,
      SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_COOPERATIVE_MATRIX_MUL_ADD_KHR_F16_M16N16K16_F32_SUBGROUP);
  EXPECT_EQ(diagnostic.status, LOOM_SPIRV_COOPERATIVE_SELECTION_MATCHED);
}

TEST(SpirvCooperativePropertiesTest, SelectsFromOwnedMatrixRowSubset) {
  iree_host_size_t model_count = 0;
  const loom_spirv_cooperative_matrix_property_t* model_rows =
      loom_spirv_cooperative_matrix_model_properties(&model_count);
  ASSERT_GE(model_count, 2u);

  loom_spirv_cooperative_property_storage_t storage = {};
  IREE_ASSERT_OK(loom_spirv_cooperative_property_storage_initialize(
      kF16CooperativeMatrixFeatures, model_rows, 1,
      /*vector_properties=*/NULL, /*vector_property_count=*/0,
      iree_allocator_system(), &storage));

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  const loom_spirv_cooperative_matrix_query_t f16_query =
      F16MatrixQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);
  ASSERT_NE(loom_spirv_cooperative_matrix_property_select(
                &storage.set, &f16_query, &diagnostic),
            nullptr);
  const loom_spirv_cooperative_matrix_query_t bf16_query =
      Bf16MatrixQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);
  EXPECT_EQ(loom_spirv_cooperative_matrix_property_select(
                &storage.set, &bf16_query, &diagnostic),
            nullptr);
  EXPECT_EQ(diagnostic.status,
            LOOM_SPIRV_COOPERATIVE_SELECTION_REQUIRED_PROPERTY_MISSING);
  EXPECT_TRUE(
      iree_any_bit_set(diagnostic.rejection_flags,
                       LOOM_SPIRV_COOPERATIVE_REJECTION_COMPONENT_TYPE));

  loom_spirv_cooperative_property_storage_deinitialize(&storage,
                                                       iree_allocator_system());
}

TEST(SpirvCooperativePropertiesTest, SelectsFromOwnedVectorRowSubset) {
  iree_host_size_t model_count = 0;
  const loom_spirv_cooperative_vector_property_t* model_rows =
      loom_spirv_cooperative_vector_model_properties(&model_count);
  ASSERT_GE(model_count, 1u);
  const loom_spirv_cooperative_vector_property_t* packed_s8_row = nullptr;
  for (iree_host_size_t i = 0; i < model_count; ++i) {
    if (iree_string_view_equal(
            model_rows[i].name,
            IREE_SV("nv.cooperative_vector.u32.32x32.s8_packed"))) {
      packed_s8_row = &model_rows[i];
      break;
    }
  }
  ASSERT_NE(packed_s8_row, nullptr);

  loom_spirv_cooperative_property_storage_t storage = {};
  IREE_ASSERT_OK(loom_spirv_cooperative_property_storage_initialize(
      LOOM_SPIRV_FEATURE_VULKAN_SHADER |
          LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV,
      /*matrix_properties=*/NULL, /*matrix_property_count=*/0, packed_s8_row, 1,
      iree_allocator_system(), &storage));

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  const loom_spirv_cooperative_vector_query_t s8_query =
      PackedS8VectorQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);
  EXPECT_NE(loom_spirv_cooperative_vector_property_select(
                &storage.set, &s8_query, &diagnostic),
            nullptr);
  loom_spirv_cooperative_vector_query_t component_miss_query = s8_query;
  component_miss_query.input_interpretation =
      LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV;
  EXPECT_EQ(loom_spirv_cooperative_vector_property_select(
                &storage.set, &component_miss_query, &diagnostic),
            nullptr);
  EXPECT_EQ(diagnostic.status,
            LOOM_SPIRV_COOPERATIVE_SELECTION_REQUIRED_PROPERTY_MISSING);
  EXPECT_TRUE(
      iree_any_bit_set(diagnostic.rejection_flags,
                       LOOM_SPIRV_COOPERATIVE_REJECTION_COMPONENT_TYPE));

  loom_spirv_cooperative_property_storage_deinitialize(&storage,
                                                       iree_allocator_system());
}

TEST(SpirvCooperativePropertiesTest, SelectsBfloat16CooperativeMatrixRow) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(kBf16CooperativeMatrixFeatures, &property_set);
  const loom_spirv_cooperative_matrix_query_t query =
      Bf16MatrixQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  const loom_spirv_cooperative_matrix_property_t* property =
      loom_spirv_cooperative_matrix_property_select(&property_set, &query,
                                                    &diagnostic);

  ASSERT_NE(property, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      property->name,
      IREE_SV("khr.cooperative_matrix.bf16.16x16x16.f32.subgroup")));
  EXPECT_EQ(property->lhs_type, LOOM_SPIRV_SCALAR_TYPE_BF16);
  EXPECT_EQ(property->rhs_type, LOOM_SPIRV_SCALAR_TYPE_BF16);
  EXPECT_EQ(property->accumulator_type, LOOM_SPIRV_SCALAR_TYPE_F32);
  EXPECT_EQ(
      property->mul_add_descriptor_ref,
      SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_COOPERATIVE_MATRIX_MUL_ADD_KHR_BF16_M16N16K16_F32_SUBGROUP);
  EXPECT_EQ(diagnostic.status, LOOM_SPIRV_COOPERATIVE_SELECTION_MATCHED);
}

TEST(SpirvCooperativePropertiesTest,
     SelectsSignedSaturatingS8CooperativeMatrixRow) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(kS8CooperativeMatrixFeatures, &property_set);
  const loom_spirv_cooperative_matrix_query_t query =
      S8MatrixQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  const loom_spirv_cooperative_matrix_property_t* property =
      loom_spirv_cooperative_matrix_property_select(&property_set, &query,
                                                    &diagnostic);

  ASSERT_NE(property, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      property->name, IREE_SV("khr.cooperative_matrix.s8.16x16x32.s32.subgroup."
                              "signed_saturating")));
  EXPECT_EQ(property->operand_flags, query.operand_flags);
  EXPECT_EQ(
      property->mul_add_descriptor_ref,
      SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_COOPERATIVE_MATRIX_MUL_ADD_KHR_S8_M16N16K32_S32_SUBGROUP_SIGNED_SATURATING);
  EXPECT_EQ(diagnostic.status, LOOM_SPIRV_COOPERATIVE_SELECTION_MATCHED);
}

TEST(SpirvCooperativePropertiesTest, SelectsUnsignedU8CooperativeMatrixRow) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(kU8CooperativeMatrixFeatures, &property_set);
  const loom_spirv_cooperative_matrix_query_t query =
      U8MatrixQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  const loom_spirv_cooperative_matrix_property_t* property =
      loom_spirv_cooperative_matrix_property_select(&property_set, &query,
                                                    &diagnostic);

  ASSERT_NE(property, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      property->name,
      IREE_SV("khr.cooperative_matrix.u8.16x16x32.u32.subgroup")));
  EXPECT_EQ(property->lhs_type, LOOM_SPIRV_SCALAR_TYPE_U8);
  EXPECT_EQ(property->rhs_type, LOOM_SPIRV_SCALAR_TYPE_U8);
  EXPECT_EQ(property->accumulator_type, LOOM_SPIRV_SCALAR_TYPE_U32);
  EXPECT_EQ(property->result_type, LOOM_SPIRV_SCALAR_TYPE_U32);
  EXPECT_EQ(property->operand_flags, 0u);
  EXPECT_EQ(
      property->mul_add_descriptor_ref,
      SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_COOPERATIVE_MATRIX_MUL_ADD_KHR_U8_M16N16K32_U32_SUBGROUP);
  EXPECT_EQ(diagnostic.status, LOOM_SPIRV_COOPERATIVE_SELECTION_MATCHED);
}

TEST(SpirvCooperativePropertiesTest,
     DistinguishesUnsignedMatrixRowsFromSignedOperandFlags) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(kU8CooperativeMatrixFeatures, &property_set);
  loom_spirv_cooperative_matrix_query_t query =
      U8MatrixQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);
  query.operand_flags =
      LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_A_SIGNED_COMPONENTS;

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_spirv_cooperative_matrix_property_select(&property_set, &query,
                                                          &diagnostic),
            nullptr);

  EXPECT_EQ(diagnostic.status,
            LOOM_SPIRV_COOPERATIVE_SELECTION_REQUIRED_PROPERTY_MISSING);
  EXPECT_TRUE(iree_any_bit_set(diagnostic.rejection_flags,
                               LOOM_SPIRV_COOPERATIVE_REJECTION_OPERANDS));
}

TEST(SpirvCooperativePropertiesTest,
     RejectsBfloat16MatrixWhenCooperativeFeatureIsMissing) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(LOOM_SPIRV_FEATURE_VULKAN_SHADER |
                         LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR |
                         LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR,
                     &property_set);
  const loom_spirv_cooperative_matrix_query_t query =
      Bf16MatrixQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_spirv_cooperative_matrix_property_select(&property_set, &query,
                                                          &diagnostic),
            nullptr);

  EXPECT_EQ(diagnostic.status,
            LOOM_SPIRV_COOPERATIVE_SELECTION_REQUIRED_FEATURE_MISSING);
  EXPECT_EQ(diagnostic.feature_atom,
            LOOM_SPIRV_FEATURE_ATOM_BFLOAT16_COOPERATIVE_MATRIX_KHR);
  EXPECT_TRUE(iree_any_bit_set(diagnostic.rejection_flags,
                               LOOM_SPIRV_COOPERATIVE_REJECTION_FEATURE));
  EXPECT_EQ(diagnostic.type_candidate_count, 1);
}

TEST(SpirvCooperativePropertiesTest, RejectsMissingMatrixFeatureSeparately) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(LOOM_SPIRV_FEATURE_VULKAN_SHADER, &property_set);
  const loom_spirv_cooperative_matrix_query_t query =
      F16MatrixQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_spirv_cooperative_matrix_property_select(&property_set, &query,
                                                          &diagnostic),
            nullptr);

  EXPECT_EQ(diagnostic.status,
            LOOM_SPIRV_COOPERATIVE_SELECTION_REQUIRED_FEATURE_MISSING);
  EXPECT_EQ(diagnostic.feature_atom,
            LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_MATRIX_KHR);
  EXPECT_TRUE(iree_any_bit_set(diagnostic.rejection_flags,
                               LOOM_SPIRV_COOPERATIVE_REJECTION_FEATURE));
}

TEST(SpirvCooperativePropertiesTest, RecordsMatrixPropertyMissAndFallback) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(kF16CooperativeMatrixFeatures, &property_set);
  loom_spirv_cooperative_matrix_query_t query =
      F16MatrixQuery(LOOM_LOWERING_POLICY_REFERENCE_ALLOWED);
  query.k_size = 8;

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_spirv_cooperative_matrix_property_select(&property_set, &query,
                                                          &diagnostic),
            nullptr);

  EXPECT_EQ(diagnostic.status,
            LOOM_SPIRV_COOPERATIVE_SELECTION_FALLBACK_PERMITTED);
  EXPECT_TRUE(iree_any_bit_set(diagnostic.rejection_flags,
                               LOOM_SPIRV_COOPERATIVE_REJECTION_SHAPE));
  EXPECT_TRUE(
      iree_any_bit_set(diagnostic.rejection_flags,
                       LOOM_SPIRV_COOPERATIVE_REJECTION_POLICY_FALLBACK));
}

TEST(SpirvCooperativePropertiesTest,
     RequiresExactCooperativeMatrixOperandFlags) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(kF16CooperativeMatrixFeatures, &property_set);
  loom_spirv_cooperative_matrix_query_t query =
      F16MatrixQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);
  query.operand_flags =
      LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_SATURATING_ACCUMULATION;

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_spirv_cooperative_matrix_property_select(&property_set, &query,
                                                          &diagnostic),
            nullptr);
  EXPECT_EQ(diagnostic.status,
            LOOM_SPIRV_COOPERATIVE_SELECTION_REQUIRED_PROPERTY_MISSING);
  EXPECT_TRUE(iree_any_bit_set(diagnostic.rejection_flags,
                               LOOM_SPIRV_COOPERATIVE_REJECTION_OPERANDS));
}

TEST(SpirvCooperativePropertiesTest, SelectsCooperativeVectorInterpretation) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(LOOM_SPIRV_FEATURE_VULKAN_SHADER |
                         LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV,
                     &property_set);
  const loom_spirv_cooperative_vector_query_t query =
      PackedS8VectorQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  const loom_spirv_cooperative_vector_property_t* property =
      loom_spirv_cooperative_vector_property_select(&property_set, &query,
                                                    &diagnostic);

  ASSERT_NE(property, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      property->name, IREE_SV("nv.cooperative_vector.u32.32x32.s8_packed")));
  EXPECT_EQ(diagnostic.status, LOOM_SPIRV_COOPERATIVE_SELECTION_MATCHED);
}

TEST(SpirvCooperativePropertiesTest, SelectsNarrowFloatVectorInterpretation) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(LOOM_SPIRV_FEATURE_VULKAN_SHADER |
                         LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV,
                     &property_set);
  const loom_spirv_cooperative_vector_query_t query =
      F8E4M3VectorQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  const loom_spirv_cooperative_vector_property_t* property =
      loom_spirv_cooperative_vector_property_select(&property_set, &query,
                                                    &diagnostic);

  ASSERT_NE(property, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      property->name, IREE_SV("nv.cooperative_vector.f16.16x32.e4m3")));
  EXPECT_EQ(property->input_interpretation,
            LOOM_SPIRV_COMPONENT_TYPE_FLOAT_E4_M3_NV);
  EXPECT_EQ(property->matrix_interpretation,
            LOOM_SPIRV_COMPONENT_TYPE_FLOAT_E4_M3_NV);
  EXPECT_EQ(diagnostic.status, LOOM_SPIRV_COOPERATIVE_SELECTION_MATCHED);
}

TEST(SpirvCooperativePropertiesTest,
     DistinguishesVectorPhysicalAndInterpretedMisses) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(LOOM_SPIRV_FEATURE_VULKAN_SHADER |
                         LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV,
                     &property_set);
  loom_spirv_cooperative_vector_query_t query =
      PackedS8VectorQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);
  query.input_type = LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT8_PACKED_NV;
  query.matrix_interpretation = LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT8_NV;

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_spirv_cooperative_vector_property_select(&property_set, &query,
                                                          &diagnostic),
            nullptr);

  EXPECT_EQ(diagnostic.status,
            LOOM_SPIRV_COOPERATIVE_SELECTION_REQUIRED_PROPERTY_MISSING);
  EXPECT_TRUE(
      iree_any_bit_set(diagnostic.rejection_flags,
                       LOOM_SPIRV_COOPERATIVE_REJECTION_COMPONENT_TYPE));
  EXPECT_TRUE(iree_any_bit_set(diagnostic.rejection_flags,
                               LOOM_SPIRV_COOPERATIVE_REJECTION_INPUT_TYPE));
  EXPECT_TRUE(
      iree_any_bit_set(diagnostic.rejection_flags,
                       LOOM_SPIRV_COOPERATIVE_REJECTION_MATRIX_INTERPRETATION));
}

TEST(SpirvCooperativePropertiesTest, RequiresTrainingFeatureForTrainingQuery) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(LOOM_SPIRV_FEATURE_VULKAN_SHADER |
                         LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV,
                     &property_set);
  loom_spirv_cooperative_vector_query_t query =
      PackedS8VectorQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);
  query.matrix_layout =
      LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL_NV;
  query.flags = LOOM_SPIRV_COOPERATIVE_VECTOR_FLAG_TRAINING;

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_spirv_cooperative_vector_property_select(&property_set, &query,
                                                          &diagnostic),
            nullptr);

  EXPECT_EQ(diagnostic.status,
            LOOM_SPIRV_COOPERATIVE_SELECTION_REQUIRED_FEATURE_MISSING);
  EXPECT_EQ(diagnostic.feature_atom,
            LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_VECTOR_TRAINING_NV);
}

TEST(SpirvCooperativePropertiesTest, SelectsCooperativeVectorTrainingRow) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(LOOM_SPIRV_FEATURE_VULKAN_SHADER |
                         LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV |
                         LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_TRAINING_NV,
                     &property_set);
  loom_spirv_cooperative_vector_query_t query =
      PackedS8VectorQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);
  query.matrix_layout =
      LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL_NV;
  query.flags = LOOM_SPIRV_COOPERATIVE_VECTOR_FLAG_TRAINING;

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  const loom_spirv_cooperative_vector_property_t* property =
      loom_spirv_cooperative_vector_property_select(&property_set, &query,
                                                    &diagnostic);

  ASSERT_NE(property, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      property->name,
      IREE_SV("nv.cooperative_vector.training.u32.32x32.s8_packed")));
  EXPECT_EQ(diagnostic.status, LOOM_SPIRV_COOPERATIVE_SELECTION_MATCHED);
}

}  // namespace
