// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/cooperative_properties.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/scalar_types.h"

namespace {

constexpr loom_spirv_feature_bits_t kF16CooperativeMatrixFeatures =
    LOOM_SPIRV_FEATURE_VULKAN_SHADER |
    LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR | LOOM_SPIRV_FEATURE_FLOAT16;

constexpr loom_spirv_feature_bits_t kBf16CooperativeMatrixFeatures =
    LOOM_SPIRV_FEATURE_VULKAN_SHADER |
    LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR |
    LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR |
    LOOM_SPIRV_FEATURE_BFLOAT16_COOPERATIVE_MATRIX_KHR;

constexpr loom_spirv_feature_bits_t kIntegerCooperativeMatrixFeatures =
    LOOM_SPIRV_FEATURE_VULKAN_SHADER |
    LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR | LOOM_SPIRV_FEATURE_INT8;

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
      .m_size = 16,
      .n_size = 16,
      .k_size = 16,
      .lhs_type = LOOM_SPIRV_SCALAR_TYPE_F16,
      .rhs_type = LOOM_SPIRV_SCALAR_TYPE_F16,
      .accumulator_type = LOOM_SPIRV_SCALAR_TYPE_F32,
      .result_type = LOOM_SPIRV_SCALAR_TYPE_F32,
      .scope = LOOM_SPIRV_SCOPE_SUBGROUP,
      .layout = LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_KHR,
      .storage_class = LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER,
      .operand_flags = {},
      .policy = policy,
  };
}

loom_spirv_cooperative_matrix_query_t Bf16MatrixQuery(
    loom_lowering_policy_t policy) {
  return {
      .m_size = 16,
      .n_size = 16,
      .k_size = 16,
      .lhs_type = LOOM_SPIRV_SCALAR_TYPE_BF16,
      .rhs_type = LOOM_SPIRV_SCALAR_TYPE_BF16,
      .accumulator_type = LOOM_SPIRV_SCALAR_TYPE_F32,
      .result_type = LOOM_SPIRV_SCALAR_TYPE_F32,
      .scope = LOOM_SPIRV_SCOPE_SUBGROUP,
      .layout = LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_KHR,
      .storage_class = LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER,
      .operand_flags = {},
      .policy = policy,
  };
}

loom_spirv_cooperative_matrix_query_t S8MatrixQuery(
    loom_lowering_policy_t policy) {
  return {
      .m_size = 16,
      .n_size = 16,
      .k_size = 32,
      .lhs_type = LOOM_SPIRV_SCALAR_TYPE_S8,
      .rhs_type = LOOM_SPIRV_SCALAR_TYPE_S8,
      .accumulator_type = LOOM_SPIRV_SCALAR_TYPE_S32,
      .result_type = LOOM_SPIRV_SCALAR_TYPE_S32,
      .scope = LOOM_SPIRV_SCOPE_SUBGROUP,
      .layout = LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_KHR,
      .storage_class = LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER,
      .operand_flags =
          LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_A_SIGNED_COMPONENTS |
          LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_B_SIGNED_COMPONENTS |
          LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_C_SIGNED_COMPONENTS |
          LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_RESULT_SIGNED_COMPONENTS |
          LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_SATURATING_ACCUMULATION,
      .policy = policy,
  };
}

loom_spirv_cooperative_matrix_query_t U8MatrixQuery(
    loom_lowering_policy_t policy) {
  return {
      .m_size = 16,
      .n_size = 16,
      .k_size = 32,
      .lhs_type = LOOM_SPIRV_SCALAR_TYPE_U8,
      .rhs_type = LOOM_SPIRV_SCALAR_TYPE_U8,
      .accumulator_type = LOOM_SPIRV_SCALAR_TYPE_U32,
      .result_type = LOOM_SPIRV_SCALAR_TYPE_U32,
      .scope = LOOM_SPIRV_SCOPE_SUBGROUP,
      .layout = LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_KHR,
      .storage_class = LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER,
      .operand_flags = {},
      .policy = policy,
  };
}

loom_spirv_cooperative_vector_query_t PackedS8VectorQuery(
    loom_lowering_policy_t policy) {
  return {
      .m_size = 32,
      .k_size = 32,
      .input_type = LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT32_NV,
      .input_interpretation = LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT8_PACKED_NV,
      .matrix_interpretation = LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT8_NV,
      .bias_interpretation = LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV,
      .result_type = LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV,
      .matrix_layout =
          LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV,
      .storage_class = LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER,
      .flags = {},
      .policy = policy,
  };
}

loom_spirv_cooperative_vector_query_t F8E4M3VectorQuery(
    loom_lowering_policy_t policy) {
  loom_spirv_component_type_t f8e4m3_component_type =
      LOOM_SPIRV_COMPONENT_TYPE_MAX;
  EXPECT_TRUE(loom_spirv_component_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3, &f8e4m3_component_type));
  return {
      .m_size = 16,
      .k_size = 32,
      .input_type = LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV,
      .input_interpretation = f8e4m3_component_type,
      .matrix_interpretation = f8e4m3_component_type,
      .bias_interpretation = LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV,
      .result_type = LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV,
      .matrix_layout =
          LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV,
      .storage_class = LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER,
      .flags = {},
      .policy = policy,
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

bool ComponentTypesMatch(loom_spirv_scalar_type_t lhs,
                         loom_spirv_scalar_type_t rhs) {
  const auto* lhs_descriptor = loom_spirv_scalar_type_descriptor(lhs);
  const auto* rhs_descriptor = loom_spirv_scalar_type_descriptor(rhs);
  if (lhs_descriptor->kind == LOOM_SPIRV_SCALAR_TYPE_KIND_FLOAT ||
      rhs_descriptor->kind == LOOM_SPIRV_SCALAR_TYPE_KIND_FLOAT) {
    return lhs == rhs;
  }
  return lhs_descriptor->bit_width == rhs_descriptor->bit_width;
}

// Independent Vulkan property matching oracle: each matrix type matches one
// operand of an advertised property, with C and D both admitting accumulators.
bool PropertyRowsAdmitType(
    const loom_spirv_cooperative_property_set_t& properties, uint16_t rows,
    uint16_t columns, loom_spirv_scalar_type_t component_type,
    loom_spirv_scope_t scope, loom_spirv_cooperative_matrix_use_t use) {
  for (uint32_t i = 0; i < properties.matrix_property_count; ++i) {
    const auto& row = properties.matrix_properties[i];
    if (row.scope != scope) {
      continue;
    }
    switch (use) {
      case LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_AKHR:
        if (row.m_size == rows && row.k_size == columns &&
            ComponentTypesMatch(row.lhs_type, component_type)) {
          return true;
        }
        break;
      case LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_BKHR:
        if (row.k_size == rows && row.n_size == columns &&
            ComponentTypesMatch(row.rhs_type, component_type)) {
          return true;
        }
        break;
      case LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_ACCUMULATOR_KHR:
        if (row.m_size == rows && row.n_size == columns &&
            (ComponentTypesMatch(row.accumulator_type, component_type) ||
             ComponentTypesMatch(row.result_type, component_type))) {
          return true;
        }
        break;
      default:
        break;
    }
  }
  return false;
}

constexpr loom_spirv_feature_bits_t kMatrixTypeFeatures =
    kF16CooperativeMatrixFeatures | kBf16CooperativeMatrixFeatures |
    kIntegerCooperativeMatrixFeatures | LOOM_SPIRV_FEATURE_FLOAT64 |
    LOOM_SPIRV_FEATURE_INT16 | LOOM_SPIRV_FEATURE_INT64;

TEST(SpirvCooperativePropertiesTest, NativeTypesMatchAdvertisedOperands) {
  loom_spirv_cooperative_property_set_t catalog = {};
  PreparePropertySet(kMatrixTypeFeatures, &catalog);
  loom_spirv_cooperative_property_storage_t storage = {};
  IREE_ASSERT_OK(loom_spirv_cooperative_property_storage_initialize(
      catalog.feature_bits, catalog.matrix_properties,
      catalog.matrix_property_count, nullptr, 0, iree_allocator_system(),
      &storage));

  for (uint16_t rows : {8, 16, 32}) {
    for (uint16_t columns : {8, 16, 32}) {
      for (auto scope :
           {LOOM_SPIRV_SCOPE_WORKGROUP, LOOM_SPIRV_SCOPE_SUBGROUP}) {
        for (uint32_t scalar = LOOM_SPIRV_SCALAR_TYPE_F16;
             scalar <= LOOM_SPIRV_SCALAR_TYPE_U64; ++scalar) {
          const auto component_type =
              static_cast<loom_spirv_scalar_type_t>(scalar);
          for (uint32_t role = 0; role < 3; ++role) {
            const auto use =
                static_cast<loom_spirv_cooperative_matrix_use_t>(role);
            const bool expected = PropertyRowsAdmitType(
                catalog, rows, columns, component_type, scope, use);
            EXPECT_EQ(loom_spirv_cooperative_matrix_type_is_supported(
                          &catalog, rows, columns, component_type, scope, use),
                      expected);
            EXPECT_EQ(
                loom_spirv_cooperative_matrix_type_is_supported(
                    &storage.set, rows, columns, component_type, scope, use),
                expected);
          }
        }
      }
    }
  }
  loom_spirv_cooperative_property_storage_deinitialize(&storage,
                                                       iree_allocator_system());
}

TEST(SpirvCooperativePropertiesTest, RestrictedProfilesPreserveTypeAndScope) {
  loom_spirv_cooperative_property_set_t catalog = {};
  PreparePropertySet(kMatrixTypeFeatures, &catalog);
  for (uint32_t scalar = LOOM_SPIRV_SCALAR_TYPE_F16;
       scalar <= LOOM_SPIRV_SCALAR_TYPE_U64; ++scalar) {
    const auto component_type = static_cast<loom_spirv_scalar_type_t>(scalar);
    auto property = catalog.matrix_properties[0];
    property.m_size = 8;
    property.n_size = 16;
    property.k_size = 32;
    property.lhs_type = component_type;
    property.rhs_type = LOOM_SPIRV_SCALAR_TYPE_U8;
    property.accumulator_type = LOOM_SPIRV_SCALAR_TYPE_F16;
    property.result_type = LOOM_SPIRV_SCALAR_TYPE_F32;
    property.scope = LOOM_SPIRV_SCOPE_WORKGROUP;
    property.required_feature_bits = kMatrixTypeFeatures;
    property.mul_add_descriptor_ref = 0;
    loom_spirv_cooperative_property_storage_t storage = {};
    IREE_ASSERT_OK(loom_spirv_cooperative_property_storage_initialize(
        catalog.feature_bits, &property, 1, nullptr, 0, iree_allocator_system(),
        &storage));
    for (uint16_t rows : {8, 16, 32}) {
      for (uint16_t columns : {8, 16, 32}) {
        for (auto scope :
             {LOOM_SPIRV_SCOPE_WORKGROUP, LOOM_SPIRV_SCOPE_SUBGROUP}) {
          for (uint32_t role = 0; role < 3; ++role) {
            const auto use =
                static_cast<loom_spirv_cooperative_matrix_use_t>(role);
            for (uint32_t query_scalar = LOOM_SPIRV_SCALAR_TYPE_F16;
                 query_scalar <= LOOM_SPIRV_SCALAR_TYPE_U64; ++query_scalar) {
              const auto query_type =
                  static_cast<loom_spirv_scalar_type_t>(query_scalar);
              EXPECT_EQ(
                  loom_spirv_cooperative_matrix_type_is_supported(
                      &storage.set, rows, columns, query_type, scope, use),
                  PropertyRowsAdmitType(storage.set, rows, columns, query_type,
                                        scope, use));
            }
          }
        }
      }
    }
    const auto required_features =
        loom_spirv_scalar_type_descriptor(component_type)
            ->required_feature_bits;
    if (required_features) {
      storage.set.feature_bits &= ~required_features;
      EXPECT_FALSE(loom_spirv_cooperative_matrix_type_is_supported(
          &storage.set, 8, 32, component_type, LOOM_SPIRV_SCOPE_WORKGROUP,
          LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_AKHR));
    }
    loom_spirv_cooperative_property_storage_deinitialize(
        &storage, iree_allocator_system());
  }
}

TEST(SpirvCooperativePropertiesTest, TypeCapabilitiesAreIndependentOfMulAdd) {
  loom_spirv_cooperative_property_set_t catalog = {};
  PreparePropertySet(kMatrixTypeFeatures, &catalog);
  const auto accumulator =
      LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_ACCUMULATOR_KHR;
  const auto lhs = LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_AKHR;
  // An F32 accumulator is valid without enabling the F16 operands of its MMA.
  catalog.feature_bits = LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR;
  EXPECT_TRUE(loom_spirv_cooperative_matrix_type_is_supported(
      &catalog, 16, 16, LOOM_SPIRV_SCALAR_TYPE_F32, LOOM_SPIRV_SCOPE_SUBGROUP,
      accumulator));
  EXPECT_FALSE(loom_spirv_cooperative_matrix_type_is_supported(
      &catalog, 16, 16, LOOM_SPIRV_SCALAR_TYPE_F16, LOOM_SPIRV_SCOPE_SUBGROUP,
      lhs));
  EXPECT_FALSE(loom_spirv_cooperative_matrix_type_is_supported(
      &catalog, 16, 32, LOOM_SPIRV_SCALAR_TYPE_S8, LOOM_SPIRV_SCOPE_SUBGROUP,
      lhs));
  for (auto missing : {LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR,
                       LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR,
                       LOOM_SPIRV_FEATURE_BFLOAT16_COOPERATIVE_MATRIX_KHR}) {
    catalog.feature_bits = kMatrixTypeFeatures & ~missing;
    EXPECT_FALSE(loom_spirv_cooperative_matrix_type_is_supported(
        &catalog, 16, 16, LOOM_SPIRV_SCALAR_TYPE_BF16,
        LOOM_SPIRV_SCOPE_SUBGROUP, lhs));
  }
  catalog.feature_bits =
      kMatrixTypeFeatures & ~LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR;
  EXPECT_FALSE(loom_spirv_cooperative_matrix_type_is_supported(
      &catalog, 16, 16, LOOM_SPIRV_SCALAR_TYPE_F32, LOOM_SPIRV_SCOPE_SUBGROUP,
      accumulator));
  loom_spirv_cooperative_property_storage_t empty = {};
  IREE_ASSERT_OK(loom_spirv_cooperative_property_storage_initialize(
      kMatrixTypeFeatures, nullptr, 0, nullptr, 0, iree_allocator_system(),
      &empty));
  EXPECT_FALSE(loom_spirv_cooperative_matrix_type_is_supported(
      &empty.set, 16, 16, LOOM_SPIRV_SCALAR_TYPE_F32, LOOM_SPIRV_SCOPE_SUBGROUP,
      accumulator));
  loom_spirv_cooperative_property_storage_deinitialize(&empty,
                                                       iree_allocator_system());
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

TEST(SpirvCooperativePropertiesTest, MatchesFloatingAccumulatorAndResultTypes) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(kF16CooperativeMatrixFeatures, &property_set);
  for (auto accumulator_type :
       {LOOM_SPIRV_SCALAR_TYPE_F16, LOOM_SPIRV_SCALAR_TYPE_F32}) {
    for (auto result_type :
         {LOOM_SPIRV_SCALAR_TYPE_F16, LOOM_SPIRV_SCALAR_TYPE_F32}) {
      auto query =
          F16MatrixQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);
      query.accumulator_type = accumulator_type;
      query.result_type = result_type;
      loom_spirv_cooperative_diagnostic_t diagnostic = {};
      const auto* property = loom_spirv_cooperative_matrix_property_select(
          &property_set, &query, &diagnostic);
      if (accumulator_type == result_type) {
        ASSERT_NE(property, nullptr);
        EXPECT_EQ(property->accumulator_type, accumulator_type);
        EXPECT_EQ(property->result_type, result_type);
        EXPECT_EQ(diagnostic.status, LOOM_SPIRV_COOPERATIVE_SELECTION_MATCHED);
      } else {
        EXPECT_EQ(property, nullptr);
        EXPECT_EQ(diagnostic.status,
                  LOOM_SPIRV_COOPERATIVE_SELECTION_REQUIRED_PROPERTY_MISSING);
        EXPECT_TRUE(
            iree_any_bit_set(diagnostic.rejection_flags,
                             LOOM_SPIRV_COOPERATIVE_REJECTION_COMPONENT_TYPE));
      }
    }
  }
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
  PreparePropertySet(kIntegerCooperativeMatrixFeatures, &property_set);
  const loom_spirv_cooperative_matrix_query_t query =
      S8MatrixQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  const loom_spirv_cooperative_matrix_property_t* property =
      loom_spirv_cooperative_matrix_property_select(&property_set, &query,
                                                    &diagnostic);

  ASSERT_NE(property, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      property->name, IREE_SV("khr.cooperative_matrix.s8.16x16x32.s32.subgroup."
                              "saturating")));
  EXPECT_EQ(property->operand_flags, query.operand_flags);
  EXPECT_EQ(
      property->mul_add_descriptor_ref,
      SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_COOPERATIVE_MATRIX_MUL_ADD_KHR_S8_M16N16K32_S32_SUBGROUP_SATURATING);
  EXPECT_EQ(diagnostic.status, LOOM_SPIRV_COOPERATIVE_SELECTION_MATCHED);
}

TEST(SpirvCooperativePropertiesTest, SelectsUnsignedU8CooperativeMatrixRow) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(kIntegerCooperativeMatrixFeatures, &property_set);
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
     SelectsCompleteEightBitIntegerMatrixFamily) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(kIntegerCooperativeMatrixFeatures, &property_set);
  for (uint16_t reduction : {16, 32}) {
    // Bits 0..3 select signed A, B, C/Result, and saturation independently.
    for (uint32_t variant = 0; variant < 16; ++variant) {
      auto query =
          U8MatrixQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);
      query.k_size = reduction;
      query.lhs_type =
          (variant & 1) ? LOOM_SPIRV_SCALAR_TYPE_S8 : LOOM_SPIRV_SCALAR_TYPE_U8;
      query.rhs_type =
          (variant & 2) ? LOOM_SPIRV_SCALAR_TYPE_S8 : LOOM_SPIRV_SCALAR_TYPE_U8;
      query.accumulator_type = query.result_type =
          (variant & 4) ? LOOM_SPIRV_SCALAR_TYPE_S32
                        : LOOM_SPIRV_SCALAR_TYPE_U32;
      query.operand_flags =
          (variant & 3) | ((variant & 4) ? 12 : 0) | ((variant & 8) ? 16 : 0);
      SCOPED_TRACE(::testing::Message()
                   << "K=" << reduction << ", flags=" << query.operand_flags);
      for (auto storage_class :
           {LOOM_SPIRV_STORAGE_CLASS_WORKGROUP,
            LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER,
            LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER}) {
        query.storage_class = storage_class;
        loom_spirv_cooperative_diagnostic_t diagnostic = {};
        const auto* property = loom_spirv_cooperative_matrix_property_select(
            &property_set, &query, &diagnostic);
        ASSERT_NE(property, nullptr);
        EXPECT_EQ(property->k_size, query.k_size);
        EXPECT_EQ(property->lhs_type, query.lhs_type);
        EXPECT_EQ(property->rhs_type, query.rhs_type);
        EXPECT_EQ(property->accumulator_type, query.accumulator_type);
        EXPECT_EQ(property->result_type, query.result_type);
        EXPECT_EQ(property->operand_flags, query.operand_flags);
        EXPECT_EQ(diagnostic.status, LOOM_SPIRV_COOPERATIVE_SELECTION_MATCHED);
      }
    }
  }
}

TEST(SpirvCooperativePropertiesTest,
     IntegerMatrixSubsetKeepsShapeAndSaturationConstraints) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(kIntegerCooperativeMatrixFeatures, &property_set);
  auto query = S8MatrixQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);
  query.k_size = 16;
  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  const auto* row = loom_spirv_cooperative_matrix_property_select(
      &property_set, &query, &diagnostic);
  ASSERT_NE(row, nullptr);
  loom_spirv_cooperative_property_storage_t storage = {};
  IREE_ASSERT_OK(loom_spirv_cooperative_property_storage_initialize(
      kIntegerCooperativeMatrixFeatures, row, 1,
      /*vector_properties=*/nullptr, /*vector_property_count=*/0,
      iree_allocator_system(), &storage));
  EXPECT_NE(loom_spirv_cooperative_matrix_property_select(&storage.set, &query,
                                                          &diagnostic),
            nullptr);
  query.operand_flags &=
      ~LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_SATURATING_ACCUMULATION;
  EXPECT_EQ(loom_spirv_cooperative_matrix_property_select(&storage.set, &query,
                                                          &diagnostic),
            nullptr);
  EXPECT_TRUE(iree_any_bit_set(diagnostic.rejection_flags,
                               LOOM_SPIRV_COOPERATIVE_REJECTION_OPERANDS));
  query.k_size = 32;
  EXPECT_EQ(loom_spirv_cooperative_matrix_property_select(&storage.set, &query,
                                                          &diagnostic),
            nullptr);
  EXPECT_TRUE(iree_any_bit_set(diagnostic.rejection_flags,
                               LOOM_SPIRV_COOPERATIVE_REJECTION_SHAPE));
  loom_spirv_cooperative_property_storage_deinitialize(&storage,
                                                       iree_allocator_system());
}

TEST(SpirvCooperativePropertiesTest,
     DistinguishesUnsignedMatrixRowsFromSignedOperandFlags) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(kIntegerCooperativeMatrixFeatures, &property_set);
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
