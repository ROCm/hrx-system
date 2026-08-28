// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/matrix/contract.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"

namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

const loom_amdgpu_matrix_contract_descriptor_t* FindDescriptor(
    const char* name) {
  iree_string_view_t expected_name = iree_make_cstring_view(name);
  const iree_host_size_t count = loom_amdgpu_matrix_contract_descriptor_count();
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
        loom_amdgpu_matrix_contract_descriptor_at(i);
    if (iree_string_view_equal(descriptor->name, expected_name)) {
      return descriptor;
    }
  }
  return nullptr;
}

loom_amdgpu_matrix_payload_shape_t PayloadShape(
    loom_amdgpu_matrix_numeric_type_t numeric_type) {
  loom_amdgpu_matrix_payload_shape_t payload_shape = {};
  payload_shape.numeric_type = numeric_type;
  return payload_shape;
}

loom_amdgpu_matrix_contract_match_request_t MatchRequest(
    loom_amdgpu_matrix_family_t family, uint16_t result_row_count,
    uint16_t result_column_count, uint16_t reduction_count,
    loom_amdgpu_matrix_numeric_type_t lhs_numeric_type,
    loom_amdgpu_matrix_numeric_type_t rhs_numeric_type,
    loom_amdgpu_matrix_numeric_type_t accumulator_numeric_type,
    loom_amdgpu_matrix_numeric_type_t result_numeric_type,
    loom_amdgpu_matrix_scale_kind_t scale_kind,
    loom_amdgpu_matrix_feature_bits_t feature_bits, uint32_t wave_size,
    loom_amdgpu_matrix_contract_flags_t available_flags,
    loom_amdgpu_matrix_contract_flags_t required_flags) {
  loom_amdgpu_matrix_contract_match_request_t request = {};
  request.family = family;
  request.tile_shape.block_count = 1;
  request.tile_shape.result_row_count = result_row_count;
  request.tile_shape.result_column_count = result_column_count;
  request.tile_shape.reduction_count = reduction_count;
  request.lhs_payload = PayloadShape(lhs_numeric_type);
  request.rhs_payload = PayloadShape(rhs_numeric_type);
  request.accumulator_payload = PayloadShape(accumulator_numeric_type);
  request.result_payload = PayloadShape(result_numeric_type);
  request.scale_kind = scale_kind;
  request.feature_bits = feature_bits;
  request.wave_size = wave_size;
  request.available_flags = available_flags;
  request.required_flags = required_flags;
  return request;
}

TEST(MatrixContractTest, DescriptorNamesAreUnique) {
  const iree_host_size_t count = loom_amdgpu_matrix_contract_descriptor_count();
  ASSERT_GT(count, 0u);
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* lhs =
        loom_amdgpu_matrix_contract_descriptor_at(i);
    ASSERT_NE(lhs, nullptr);
    EXPECT_FALSE(iree_string_view_is_empty(lhs->name));
    EXPECT_FALSE(iree_string_view_is_empty(lhs->llvm_intrinsic_name));
    for (iree_host_size_t j = i + 1; j < count; ++j) {
      const loom_amdgpu_matrix_contract_descriptor_t* rhs =
          loom_amdgpu_matrix_contract_descriptor_at(j);
      ASSERT_NE(rhs, nullptr);
      EXPECT_FALSE(iree_string_view_equal(lhs->name, rhs->name))
          << ToString(lhs->name);
    }
  }
  EXPECT_EQ(loom_amdgpu_matrix_contract_descriptor_at(count), nullptr);
}

TEST(MatrixContractTest, Gfx942Fp8MfmaDescriptor) {
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      FindDescriptor("mfma.f32.16x16x32.fp8.fp8");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_MFMA);
  EXPECT_EQ(ToString(descriptor->llvm_intrinsic_name),
            "llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8");
  EXPECT_EQ(descriptor->tile_shape.result_row_count, 16);
  EXPECT_EQ(descriptor->tile_shape.result_column_count, 16);
  EXPECT_EQ(descriptor->tile_shape.reduction_count, 32);
  EXPECT_EQ(descriptor->lhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_FP8);
  EXPECT_EQ(descriptor->rhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_FP8);
  EXPECT_EQ(descriptor->accumulator_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32);
  EXPECT_EQ(descriptor->result_payload.register_count, 4);
  EXPECT_EQ(descriptor->scale_kind, LOOM_AMDGPU_MATRIX_SCALE_NONE);
}

TEST(MatrixContractTest, Gfx90aBf16OneKMfmaDescriptor) {
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      FindDescriptor("mfma.f32.16x16x16.bf16.1k");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_MFMA);
  EXPECT_EQ(ToString(descriptor->llvm_intrinsic_name),
            "llvm.amdgcn.mfma.f32.16x16x16.bf16.1k");
  EXPECT_EQ(descriptor->tile_shape.result_row_count, 16);
  EXPECT_EQ(descriptor->tile_shape.result_column_count, 16);
  EXPECT_EQ(descriptor->tile_shape.reduction_count, 16);
  EXPECT_EQ(descriptor->lhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16);
  EXPECT_EQ(descriptor->lhs_payload.register_count, 2);
  EXPECT_EQ(descriptor->accumulator_payload.register_count, 4);
}

TEST(MatrixContractTest, Gfx908LegacyMfmaDescriptor) {
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      FindDescriptor("mfma.f32.32x32x8.f16");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_MFMA);
  EXPECT_EQ(descriptor->required_feature_bits,
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908);
  EXPECT_EQ(descriptor->tile_shape.result_row_count, 32);
  EXPECT_EQ(descriptor->tile_shape.result_column_count, 32);
  EXPECT_EQ(descriptor->tile_shape.reduction_count, 8);
  EXPECT_EQ(descriptor->lhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16);
  EXPECT_EQ(descriptor->result_payload.register_count, 16);
}

TEST(MatrixContractTest, Gfx908AndGfx90aOnlyMfmaFeatureProfiles) {
  loom_amdgpu_matrix_feature_bits_t feature_bits = 0;
  ASSERT_TRUE(loom_amdgpu_matrix_feature_bits_from_profile(
      LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908, &feature_bits));
  EXPECT_TRUE(iree_all_bits_set(feature_bits,
                                LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908_GFX90A));

  ASSERT_TRUE(loom_amdgpu_matrix_feature_bits_from_profile(
      LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A, &feature_bits));
  EXPECT_TRUE(iree_all_bits_set(feature_bits,
                                LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908_GFX90A));

  ASSERT_TRUE(loom_amdgpu_matrix_feature_bits_from_profile(
      LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940, &feature_bits));
  EXPECT_FALSE(iree_any_bit_set(feature_bits,
                                LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908_GFX90A));

  ASSERT_TRUE(loom_amdgpu_matrix_feature_bits_from_profile(
      LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950, &feature_bits));
  EXPECT_FALSE(iree_any_bit_set(feature_bits,
                                LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908_GFX90A));
}

TEST(MatrixContractTest, FeatureInfoCoversKnownFeatureBits) {
  loom_amdgpu_matrix_feature_bits_t seen_bits = 0;
  const iree_host_size_t count = loom_amdgpu_matrix_feature_info_count();
  ASSERT_GT(count, 0u);
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_amdgpu_matrix_feature_info_t* feature_info =
        loom_amdgpu_matrix_feature_info_at(i);
    ASSERT_NE(feature_info, nullptr);
    EXPECT_NE(feature_info->feature_bit, 0u);
    EXPECT_EQ(feature_info->feature_bit & (feature_info->feature_bit - 1), 0u);
    EXPECT_FALSE(iree_string_view_is_empty(feature_info->name));
    EXPECT_FALSE(iree_any_bit_set(seen_bits, feature_info->feature_bit))
        << ToString(feature_info->name);
    seen_bits |= feature_info->feature_bit;
  }
  EXPECT_EQ(loom_amdgpu_matrix_feature_info_at(count), nullptr);

  const loom_amdgpu_matrix_feature_bits_t expected_bits =
      LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908 |
      LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908_GFX90A |
      LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_BF16_1K |
      LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_F64 |
      LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_FP8 |
      LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_I8 |
      LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_XF32 |
      LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950 |
      LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950_SCALE_F8F6F4 |
      LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX940 |
      LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX940_FP8 |
      LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX950 |
      LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11 |
      LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX12 |
      LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX12 |
      LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250 |
      LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250_SCALE_F8F6F4 |
      LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX1250;
  EXPECT_EQ(seen_bits, expected_bits);
}

TEST(MatrixContractTest, Gfx908SmallLegacyMfmaDescriptor) {
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      FindDescriptor("mfma.f32.4x4x2.bf16");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_MFMA);
  EXPECT_EQ(descriptor->required_feature_bits,
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908_GFX90A);
  EXPECT_EQ(ToString(descriptor->llvm_intrinsic_name),
            "llvm.amdgcn.mfma.f32.4x4x2.bf16");
  EXPECT_EQ(descriptor->tile_shape.result_row_count, 4);
  EXPECT_EQ(descriptor->tile_shape.result_column_count, 4);
  EXPECT_EQ(descriptor->tile_shape.reduction_count, 2);
  EXPECT_EQ(descriptor->tile_shape.block_count, 16);
  EXPECT_EQ(descriptor->lhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16);
  EXPECT_EQ(descriptor->lhs_payload.register_count, 1);
  EXPECT_EQ(descriptor->lhs_payload.element_count, 2);
  EXPECT_EQ(descriptor->result_payload.register_count, 4);
}

TEST(MatrixContractTest, MatcherDistinguishesBlockedMfmaShape) {
  loom_amdgpu_matrix_contract_match_request_t request = MatchRequest(
      LOOM_AMDGPU_MATRIX_FAMILY_MFMA, 4, 4, 2, LOOM_AMDGPU_MATRIX_NUMERIC_BF16,
      LOOM_AMDGPU_MATRIX_NUMERIC_BF16, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
      LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_SCALE_NONE,
      LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908_GFX90A, 64, 0, 0);
  loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(&request, &diagnostic), nullptr);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_TILE_SHAPE);

  request.tile_shape.block_count = 16;
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      loom_amdgpu_matrix_contract_select(&request, &diagnostic);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor->name), "mfma.f32.4x4x2.bf16");
  EXPECT_EQ(descriptor->tile_shape.block_count, 16);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE);
}

TEST(MatrixContractTest, Gfx908LegacyIntegerMfmaDescriptor) {
  const loom_amdgpu_matrix_contract_descriptor_t* small_descriptor =
      FindDescriptor("mfma.i32.4x4x4.i8");
  ASSERT_NE(small_descriptor, nullptr);
  EXPECT_EQ(small_descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_MFMA);
  EXPECT_EQ(small_descriptor->required_feature_bits,
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908);
  EXPECT_EQ(small_descriptor->tile_shape.result_row_count, 4);
  EXPECT_EQ(small_descriptor->tile_shape.result_column_count, 4);
  EXPECT_EQ(small_descriptor->tile_shape.reduction_count, 4);
  EXPECT_EQ(small_descriptor->lhs_payload.register_count, 1);
  EXPECT_EQ(small_descriptor->result_payload.register_count, 4);

  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      FindDescriptor("mfma.i32.16x16x16.i8");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_MFMA);
  EXPECT_EQ(descriptor->required_feature_bits,
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908_GFX90A);
  EXPECT_EQ(descriptor->tile_shape.result_row_count, 16);
  EXPECT_EQ(descriptor->tile_shape.result_column_count, 16);
  EXPECT_EQ(descriptor->tile_shape.reduction_count, 16);
  EXPECT_EQ(descriptor->lhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_I8);
  EXPECT_EQ(descriptor->lhs_payload.register_count, 1);
  EXPECT_EQ(descriptor->lhs_payload.element_count, 4);
  EXPECT_EQ(descriptor->accumulator_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32);
  EXPECT_EQ(descriptor->result_payload.register_count, 4);
}

TEST(MatrixContractTest, Gfx90aDoubleMfmaDescriptor) {
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      FindDescriptor("mfma.f64.16x16x4.f64");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_MFMA);
  EXPECT_EQ(descriptor->required_feature_bits,
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_F64);
  EXPECT_EQ(descriptor->tile_shape.result_row_count, 16);
  EXPECT_EQ(descriptor->tile_shape.result_column_count, 16);
  EXPECT_EQ(descriptor->tile_shape.reduction_count, 4);
  EXPECT_EQ(descriptor->lhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_F64);
  EXPECT_EQ(descriptor->lhs_payload.register_count, 2);
  EXPECT_EQ(descriptor->lhs_payload.element_count, 1);
  EXPECT_EQ(descriptor->result_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_F64);
  EXPECT_EQ(descriptor->result_payload.register_count, 8);
  EXPECT_EQ(descriptor->result_payload.element_count, 4);
  EXPECT_EQ(descriptor->fragment_layout_kind,
            LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F64_16X16X4_F64);
}

TEST(MatrixContractTest, Gfx950DenseMfmaDescriptorsExposeTargetLowIds) {
  struct Case {
    const char* descriptor_name;
    loom_amdgpu_descriptor_ref_t low_descriptor_ref;
    loom_amdgpu_matrix_numeric_type_t input_numeric_type;
    loom_amdgpu_matrix_numeric_type_t accumulator_numeric_type;
    uint16_t result_row_count;
    uint16_t result_column_count;
    uint16_t reduction_count;
    uint16_t input_register_count;
    uint16_t input_element_count;
    uint16_t accumulator_register_count;
    uint16_t accumulator_element_count;
  };
  const Case cases[] = {
      {
          "mfma.f32.16x16x32.f16",
          LOOM_AMDGPU_DESCRIPTOR_REF_V_MFMA_F32_16X16X32_F16,
          LOOM_AMDGPU_MATRIX_NUMERIC_F16,
          LOOM_AMDGPU_MATRIX_NUMERIC_F32,
          16,
          16,
          32,
          4,
          8,
          4,
          4,
      },
      {
          "mfma.f32.16x16x32.bf16",
          LOOM_AMDGPU_DESCRIPTOR_REF_V_MFMA_F32_16X16X32_BF16,
          LOOM_AMDGPU_MATRIX_NUMERIC_BF16,
          LOOM_AMDGPU_MATRIX_NUMERIC_F32,
          16,
          16,
          32,
          4,
          8,
          4,
          4,
      },
      {
          "mfma.f32.32x32x16.f16",
          LOOM_AMDGPU_DESCRIPTOR_REF_V_MFMA_F32_32X32X16_F16,
          LOOM_AMDGPU_MATRIX_NUMERIC_F16,
          LOOM_AMDGPU_MATRIX_NUMERIC_F32,
          32,
          32,
          16,
          4,
          8,
          16,
          16,
      },
      {
          "mfma.f32.32x32x16.bf16",
          LOOM_AMDGPU_DESCRIPTOR_REF_V_MFMA_F32_32X32X16_BF16,
          LOOM_AMDGPU_MATRIX_NUMERIC_BF16,
          LOOM_AMDGPU_MATRIX_NUMERIC_F32,
          32,
          32,
          16,
          4,
          8,
          16,
          16,
      },
      {
          "mfma.i32.16x16x64.i8",
          LOOM_AMDGPU_DESCRIPTOR_REF_V_MFMA_I32_16X16X64_I8,
          LOOM_AMDGPU_MATRIX_NUMERIC_I8,
          LOOM_AMDGPU_MATRIX_NUMERIC_I32,
          16,
          16,
          64,
          4,
          16,
          4,
          4,
      },
      {
          "mfma.i32.32x32x32.i8",
          LOOM_AMDGPU_DESCRIPTOR_REF_V_MFMA_I32_32X32X32_I8,
          LOOM_AMDGPU_MATRIX_NUMERIC_I8,
          LOOM_AMDGPU_MATRIX_NUMERIC_I32,
          32,
          32,
          32,
          4,
          16,
          16,
          16,
      },
  };
  for (const Case& test_case : cases) {
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
        FindDescriptor(test_case.descriptor_name);
    ASSERT_NE(descriptor, nullptr) << test_case.descriptor_name;
    EXPECT_EQ(descriptor->low_descriptor_ref, test_case.low_descriptor_ref)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->required_feature_bits,
              LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_MFMA)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->tile_shape.result_row_count,
              test_case.result_row_count)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->tile_shape.result_column_count,
              test_case.result_column_count)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->tile_shape.reduction_count, test_case.reduction_count)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->lhs_payload.numeric_type,
              test_case.input_numeric_type)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->lhs_payload.register_count,
              test_case.input_register_count)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->lhs_payload.element_count,
              test_case.input_element_count)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->rhs_payload.numeric_type,
              test_case.input_numeric_type)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->rhs_payload.register_count,
              test_case.input_register_count)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->rhs_payload.element_count,
              test_case.input_element_count)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->accumulator_payload.numeric_type,
              test_case.accumulator_numeric_type)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->accumulator_payload.register_count,
              test_case.accumulator_register_count)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->accumulator_payload.element_count,
              test_case.accumulator_element_count)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->result_payload.numeric_type,
              test_case.accumulator_numeric_type)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->result_payload.register_count,
              test_case.accumulator_register_count)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->result_payload.element_count,
              test_case.accumulator_element_count)
        << test_case.descriptor_name;
  }
}

TEST(MatrixContractTest, MatcherSelectedGfx950MfmaDescriptorCarriesLowId) {
  loom_amdgpu_matrix_contract_match_request_t request = MatchRequest(
      LOOM_AMDGPU_MATRIX_FAMILY_MFMA, 16, 16, 32,
      LOOM_AMDGPU_MATRIX_NUMERIC_F16, LOOM_AMDGPU_MATRIX_NUMERIC_F16,
      LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
      LOOM_AMDGPU_MATRIX_SCALE_NONE, LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950, 64,
      0, 0);
  loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      loom_amdgpu_matrix_contract_select(&request, &diagnostic);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_MFMA_F32_16X16X32_F16);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE);
}

TEST(MatrixContractTest, Gfx950ScaledFp4MfmaDescriptor) {
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      FindDescriptor("mfma.scale.f32.32x32x64.f8f6f4.f4.f4");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_MFMA);
  EXPECT_EQ(descriptor->tile_shape.result_row_count, 32);
  EXPECT_EQ(descriptor->tile_shape.result_column_count, 32);
  EXPECT_EQ(descriptor->tile_shape.reduction_count, 64);
  EXPECT_EQ(descriptor->lhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_FP4);
  EXPECT_EQ(descriptor->lhs_payload.register_count, 4);
  EXPECT_EQ(descriptor->lhs_payload.element_count, 32);
  EXPECT_EQ(descriptor->rhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_FP4);
  EXPECT_EQ(descriptor->rhs_payload.register_count, 4);
  EXPECT_EQ(descriptor->rhs_payload.element_count, 32);
  EXPECT_EQ(descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED);
  EXPECT_EQ(descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS);
  EXPECT_EQ(
      descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK,
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK);
  EXPECT_EQ(descriptor->scale_kind, LOOM_AMDGPU_MATRIX_SCALE_32);
  EXPECT_EQ(descriptor->implicit_scale_format_selector_bits,
            1u << LOOM_AMDGPU_MATRIX_SCALE_FORMAT_SELECTOR_E8M0);
  EXPECT_EQ(descriptor->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_MFMA_SCALE_F32_32X32X64_F8F6F4_F4_F4);

  const loom_amdgpu_matrix_contract_descriptor_t* compact_descriptor =
      FindDescriptor("mfma.scale.f32.16x16x128.f8f6f4.f4.f4");
  ASSERT_NE(compact_descriptor, nullptr);
  EXPECT_EQ(compact_descriptor->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_MFMA_SCALE_F32_16X16X128_F8F6F4_F4_F4);
}

TEST(MatrixContractTest, Gfx950SelectorFormatsChooseEveryPhysicalMfmaAbi) {
  struct NumericCase {
    loom_amdgpu_matrix_numeric_type_t numeric_type;
    const char* physical_token;
    loom_amdgpu_matrix_numeric_type_t contract_numeric_type;
    uint16_t register_count;
  };
  static const NumericCase numeric_cases[] = {
      {LOOM_AMDGPU_MATRIX_NUMERIC_FP8, "f8", LOOM_AMDGPU_MATRIX_NUMERIC_F8, 8},
      {LOOM_AMDGPU_MATRIX_NUMERIC_BF8, "f8", LOOM_AMDGPU_MATRIX_NUMERIC_F8, 8},
      {LOOM_AMDGPU_MATRIX_NUMERIC_FP6, "f6", LOOM_AMDGPU_MATRIX_NUMERIC_F6, 6},
      {LOOM_AMDGPU_MATRIX_NUMERIC_BF6, "f6", LOOM_AMDGPU_MATRIX_NUMERIC_F6, 6},
      {LOOM_AMDGPU_MATRIX_NUMERIC_FP4, "f4", LOOM_AMDGPU_MATRIX_NUMERIC_FP4, 4},
  };
  struct TileCase {
    uint16_t row_count;
    uint16_t column_count;
    uint16_t reduction_count;
    uint16_t accumulator_register_count;
    const char* shape_token;
  };
  static const TileCase tile_cases[] = {
      {16, 16, 128, 4, "16x16x128"},
      {32, 32, 64, 16, "32x32x64"},
  };
  struct ScaleCase {
    loom_amdgpu_matrix_scale_kind_t scale_kind;
    const char* name_prefix;
  };
  static const ScaleCase scale_cases[] = {
      {LOOM_AMDGPU_MATRIX_SCALE_NONE, ""},
      {LOOM_AMDGPU_MATRIX_SCALE_32, "scale."},
  };

  for (const ScaleCase& scale_case : scale_cases) {
    for (const TileCase& tile_case : tile_cases) {
      for (const NumericCase& lhs_case : numeric_cases) {
        for (const NumericCase& rhs_case : numeric_cases) {
          loom_amdgpu_matrix_contract_flags_t available_flags =
              LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS;
          loom_amdgpu_matrix_contract_flags_t required_flags =
              LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS;
          if (scale_case.scale_kind == LOOM_AMDGPU_MATRIX_SCALE_32) {
            available_flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED |
                               LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS;
            required_flags = LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED;
          }
          loom_amdgpu_matrix_contract_match_request_t request = MatchRequest(
              LOOM_AMDGPU_MATRIX_FAMILY_MFMA, tile_case.row_count,
              tile_case.column_count, tile_case.reduction_count,
              lhs_case.numeric_type, rhs_case.numeric_type,
              LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
              scale_case.scale_kind,
              LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950 |
                  LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950_SCALE_F8F6F4,
              64, available_flags, required_flags);
          request.lhs_payload.register_count = lhs_case.register_count;
          request.lhs_payload.element_count = 32;
          request.rhs_payload.register_count = rhs_case.register_count;
          request.rhs_payload.element_count = 32;
          request.accumulator_payload.register_count =
              tile_case.accumulator_register_count;
          request.accumulator_payload.element_count =
              tile_case.accumulator_register_count;
          request.result_payload.register_count =
              tile_case.accumulator_register_count;
          request.result_payload.element_count =
              tile_case.accumulator_register_count;
          if (scale_case.scale_kind == LOOM_AMDGPU_MATRIX_SCALE_32) {
            request.lhs_scale_format_selector_bits =
                1u << LOOM_AMDGPU_MATRIX_SCALE_FORMAT_SELECTOR_E8M0;
            request.rhs_scale_format_selector_bits =
                1u << LOOM_AMDGPU_MATRIX_SCALE_FORMAT_SELECTOR_E8M0;
          }

          loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
          const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
              loom_amdgpu_matrix_contract_select(&request, &diagnostic);
          ASSERT_NE(descriptor, nullptr);
          const std::string expected_name =
              std::string("mfma.") + scale_case.name_prefix + "f32." +
              tile_case.shape_token + ".f8f6f4." + lhs_case.physical_token +
              "." + rhs_case.physical_token;
          EXPECT_EQ(ToString(descriptor->name), expected_name);
          EXPECT_EQ(descriptor->lhs_payload.numeric_type,
                    lhs_case.contract_numeric_type);
          EXPECT_EQ(descriptor->lhs_payload.register_count,
                    lhs_case.register_count);
          EXPECT_EQ(descriptor->rhs_payload.numeric_type,
                    rhs_case.contract_numeric_type);
          EXPECT_EQ(descriptor->rhs_payload.register_count,
                    rhs_case.register_count);
          EXPECT_NE(descriptor->low_descriptor_ref,
                    LOOM_AMDGPU_MATRIX_LOW_DESCRIPTOR_REF_NONE);
          EXPECT_EQ(diagnostic.rejection_bits,
                    LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE);
        }
      }
    }
  }
}

TEST(MatrixContractTest, Gfx1250WmmaScale16Descriptor) {
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      FindDescriptor("wmma.scale16.f32.16x16x128.f8f6f4.f8.f8");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_WMMA);
  EXPECT_EQ(descriptor->tile_shape.reduction_count, 128);
  EXPECT_EQ(descriptor->result_payload.register_count, 8);
  EXPECT_EQ(descriptor->scale_kind, LOOM_AMDGPU_MATRIX_SCALE_16);
  EXPECT_EQ(descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS);
  EXPECT_EQ(descriptor->implicit_scale_format_selector_bits, 0u);
  EXPECT_EQ(descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS);
  EXPECT_EQ(descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE, 0u);
  EXPECT_EQ(
      descriptor->low_descriptor_ref,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_SCALE16_F32_16X16X128_F8F6F4_F8_F8);
}

TEST(MatrixContractTest, Gfx125xSelectorFormatsChooseEveryPhysicalWmmaAbi) {
  struct NumericCase {
    loom_amdgpu_matrix_numeric_type_t numeric_type;
    const char* physical_token;
    loom_amdgpu_matrix_numeric_type_t contract_numeric_type;
    uint16_t register_count;
    uint16_t element_bit_count;
  };
  static const NumericCase numeric_cases[] = {
      {LOOM_AMDGPU_MATRIX_NUMERIC_FP8, "f8", LOOM_AMDGPU_MATRIX_NUMERIC_F8, 16,
       8},
      {LOOM_AMDGPU_MATRIX_NUMERIC_BF8, "f8", LOOM_AMDGPU_MATRIX_NUMERIC_F8, 16,
       8},
      {LOOM_AMDGPU_MATRIX_NUMERIC_FP6, "f6", LOOM_AMDGPU_MATRIX_NUMERIC_F6, 12,
       6},
      {LOOM_AMDGPU_MATRIX_NUMERIC_BF6, "f6", LOOM_AMDGPU_MATRIX_NUMERIC_F6, 12,
       6},
      {LOOM_AMDGPU_MATRIX_NUMERIC_FP4, "f4", LOOM_AMDGPU_MATRIX_NUMERIC_FP4, 8,
       4},
  };
  struct ScaleCase {
    loom_amdgpu_matrix_scale_kind_t scale_kind;
    const char* name_prefix;
  };
  static const ScaleCase scale_cases[] = {
      {LOOM_AMDGPU_MATRIX_SCALE_NONE, ""},
      {LOOM_AMDGPU_MATRIX_SCALE_32, "scale."},
      {LOOM_AMDGPU_MATRIX_SCALE_16, "scale16."},
  };

  for (const ScaleCase& scale_case : scale_cases) {
    for (const NumericCase& lhs_case : numeric_cases) {
      for (const NumericCase& rhs_case : numeric_cases) {
        loom_amdgpu_matrix_contract_flags_t flags =
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS;
        if (scale_case.scale_kind == LOOM_AMDGPU_MATRIX_SCALE_NONE) {
          flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER;
        } else {
          flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED |
                   LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS;
        }
        loom_amdgpu_matrix_contract_match_request_t request = MatchRequest(
            LOOM_AMDGPU_MATRIX_FAMILY_WMMA, 16, 16, 128, lhs_case.numeric_type,
            rhs_case.numeric_type, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, scale_case.scale_kind,
            LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250 |
                LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250_SCALE_F8F6F4,
            32, flags, flags);
        request.lhs_payload.register_count = lhs_case.register_count;
        request.lhs_payload.element_count = 64;
        request.rhs_payload.register_count = rhs_case.register_count;
        request.rhs_payload.element_count = 64;
        request.accumulator_payload.register_count = 8;
        request.accumulator_payload.element_count = 8;
        request.result_payload.register_count = 8;
        request.result_payload.element_count = 8;

        loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
        const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
            loom_amdgpu_matrix_contract_select(&request, &diagnostic);
        ASSERT_NE(descriptor, nullptr);
        const std::string expected_name =
            std::string("wmma.") + scale_case.name_prefix +
            "f32.16x16x128.f8f6f4." + lhs_case.physical_token + "." +
            rhs_case.physical_token;
        EXPECT_EQ(ToString(descriptor->name), expected_name);
        EXPECT_EQ(descriptor->lhs_payload.numeric_type,
                  lhs_case.contract_numeric_type);
        EXPECT_EQ(descriptor->rhs_payload.numeric_type,
                  rhs_case.contract_numeric_type);
        EXPECT_NE(descriptor->low_descriptor_ref,
                  LOOM_AMDGPU_MATRIX_LOW_DESCRIPTOR_REF_NONE);

        const loom_amdgpu_matrix_fragment_layout_t* layout =
            loom_amdgpu_matrix_contract_descriptor_fragment_layout(descriptor);
        ASSERT_NE(layout, nullptr);
        EXPECT_EQ(layout->lhs.register_count, lhs_case.register_count);
        EXPECT_EQ(layout->lhs.element_bit_count, lhs_case.element_bit_count);
        EXPECT_EQ(layout->rhs.register_count, rhs_case.register_count);
        EXPECT_EQ(layout->rhs.element_bit_count, rhs_case.element_bit_count);
        EXPECT_EQ(diagnostic.rejection_bits,
                  LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE);
      }
    }
  }
}

TEST(MatrixContractTest, Gfx1250WmmaModifierDescriptors) {
  const loom_amdgpu_matrix_contract_descriptor_t* f32_descriptor =
      FindDescriptor("wmma.f32.16x16x4.f32");
  ASSERT_NE(f32_descriptor, nullptr);
  EXPECT_EQ(f32_descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_WMMA);
  EXPECT_EQ(f32_descriptor->tile_shape.reduction_count, 4);
  EXPECT_EQ(f32_descriptor->lhs_payload.register_count, 2);
  EXPECT_EQ(f32_descriptor->result_payload.register_count, 8);
  EXPECT_EQ(
      f32_descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_AB_MODIFIERS,
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_AB_MODIFIERS);
  EXPECT_EQ(f32_descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE);

  const loom_amdgpu_matrix_contract_descriptor_t* mixed_descriptor =
      FindDescriptor("wmma.bf16f32.16x16x32.bf16");
  ASSERT_NE(mixed_descriptor, nullptr);
  EXPECT_EQ(mixed_descriptor->lhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16);
  EXPECT_EQ(mixed_descriptor->accumulator_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32);
  EXPECT_EQ(mixed_descriptor->result_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16);
  EXPECT_EQ(mixed_descriptor->result_payload.register_count, 4);
}

TEST(MatrixContractTest, Gfx1250WmmaF4Descriptors) {
  const loom_amdgpu_matrix_contract_descriptor_t* unscaled =
      FindDescriptor("wmma.f32.32x16x128.f4");
  ASSERT_NE(unscaled, nullptr);
  EXPECT_EQ(unscaled->tile_shape.result_row_count, 32);
  EXPECT_EQ(unscaled->tile_shape.result_column_count, 16);
  EXPECT_EQ(unscaled->lhs_payload.numeric_type, LOOM_AMDGPU_MATRIX_NUMERIC_FP4);
  EXPECT_EQ(unscaled->lhs_payload.element_count, 128);
  EXPECT_EQ(unscaled->rhs_payload.element_count, 64);
  EXPECT_EQ(unscaled->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER);

  const loom_amdgpu_matrix_contract_descriptor_t* scaled =
      FindDescriptor("wmma.scale.f32.32x16x128.f4");
  ASSERT_NE(scaled, nullptr);
  EXPECT_EQ(scaled->scale_kind, LOOM_AMDGPU_MATRIX_SCALE_32);
  EXPECT_EQ(scaled->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED);
  EXPECT_EQ(scaled->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS);
  EXPECT_EQ(scaled->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS,
            0u);
}

TEST(MatrixContractTest, Gfx12SwmmacBaseDescriptors) {
  const loom_amdgpu_matrix_contract_descriptor_t* f16_descriptor =
      FindDescriptor("swmmac.f32.16x16x32.f16");
  ASSERT_NE(f16_descriptor, nullptr);
  EXPECT_EQ(f16_descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC);
  EXPECT_EQ(f16_descriptor->required_feature_bits,
            LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX12);
  EXPECT_EQ(f16_descriptor->wave_size_bits, LOOM_AMDGPU_MATRIX_WAVE_SIZE_32);
  EXPECT_EQ(f16_descriptor->tile_shape.reduction_count, 32);
  EXPECT_EQ(f16_descriptor->lhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16);
  EXPECT_EQ(f16_descriptor->lhs_payload.register_count, 4);
  EXPECT_EQ(f16_descriptor->lhs_payload.element_count, 8);
  EXPECT_EQ(f16_descriptor->rhs_payload.register_count, 8);
  EXPECT_EQ(f16_descriptor->rhs_payload.element_count, 16);
  EXPECT_EQ(f16_descriptor->result_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32);
  EXPECT_EQ(f16_descriptor->result_payload.register_count, 8);
  EXPECT_EQ(f16_descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE);

  const loom_amdgpu_matrix_contract_descriptor_t* bf16_descriptor =
      FindDescriptor("swmmac.bf16.16x16x32.bf16");
  ASSERT_NE(bf16_descriptor, nullptr);
  EXPECT_EQ(bf16_descriptor->lhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16);
  EXPECT_EQ(bf16_descriptor->result_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16);

  const loom_amdgpu_matrix_contract_descriptor_t* iu8_descriptor =
      FindDescriptor("swmmac.i32.16x16x32.iu8");
  ASSERT_NE(iu8_descriptor, nullptr);
  EXPECT_EQ(iu8_descriptor->lhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_IU8);
  EXPECT_EQ(iu8_descriptor->rhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_IU8);
  EXPECT_EQ(iu8_descriptor->result_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32);
  EXPECT_EQ(iu8_descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE);
  EXPECT_EQ(
      iu8_descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT,
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT);
  EXPECT_EQ(iu8_descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP);

  const loom_amdgpu_matrix_contract_descriptor_t* iu4_descriptor =
      FindDescriptor("swmmac.i32.16x16x32.iu4");
  ASSERT_NE(iu4_descriptor, nullptr);
  EXPECT_EQ(iu4_descriptor->lhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_IU4);
}

TEST(MatrixContractTest, SparseDescriptorsCarrySparseFlag) {
  const loom_amdgpu_matrix_contract_descriptor_t* smfmac =
      FindDescriptor("smfmac.f32.16x16x128.fp8.fp8");
  ASSERT_NE(smfmac, nullptr);
  EXPECT_EQ(smfmac->family, LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC);
  EXPECT_EQ(smfmac->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE);

  const loom_amdgpu_matrix_contract_descriptor_t* swmmac =
      FindDescriptor("swmmac.i32.16x16x64.iu4");
  ASSERT_NE(swmmac, nullptr);
  EXPECT_EQ(swmmac->family, LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC);
  EXPECT_EQ(swmmac->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE);
  EXPECT_EQ(swmmac->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT);
}

TEST(MatrixContractTest, SparseFp8CrossProductDescriptors) {
  const loom_amdgpu_matrix_contract_descriptor_t* smfmac =
      FindDescriptor("smfmac.f32.16x16x64.bf8.fp8");
  ASSERT_NE(smfmac, nullptr);
  EXPECT_EQ(smfmac->family, LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC);
  EXPECT_EQ(smfmac->tile_shape.reduction_count, 64);
  EXPECT_EQ(smfmac->lhs_payload.numeric_type, LOOM_AMDGPU_MATRIX_NUMERIC_BF8);
  EXPECT_EQ(smfmac->rhs_payload.numeric_type, LOOM_AMDGPU_MATRIX_NUMERIC_FP8);
  EXPECT_EQ(smfmac->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE);

  const loom_amdgpu_matrix_contract_descriptor_t* swmmac =
      FindDescriptor("swmmac.f32.16x16x32.fp8.bf8");
  ASSERT_NE(swmmac, nullptr);
  EXPECT_EQ(swmmac->family, LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC);
  EXPECT_EQ(swmmac->tile_shape.reduction_count, 32);
  EXPECT_EQ(swmmac->lhs_payload.numeric_type, LOOM_AMDGPU_MATRIX_NUMERIC_FP8);
  EXPECT_EQ(swmmac->rhs_payload.numeric_type, LOOM_AMDGPU_MATRIX_NUMERIC_BF8);
  EXPECT_EQ(swmmac->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE);

  const loom_amdgpu_matrix_contract_descriptor_t* gfx1250_swmmac =
      FindDescriptor("swmmac.f16.16x16x128.bf8.fp8");
  ASSERT_NE(gfx1250_swmmac, nullptr);
  EXPECT_EQ(gfx1250_swmmac->family, LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC);
  EXPECT_EQ(gfx1250_swmmac->required_feature_bits,
            LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX1250);
  EXPECT_EQ(gfx1250_swmmac->tile_shape.reduction_count, 128);
  EXPECT_EQ(gfx1250_swmmac->lhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF8);
  EXPECT_EQ(gfx1250_swmmac->rhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_FP8);
  EXPECT_EQ(gfx1250_swmmac->lhs_payload.register_count, 8);
  EXPECT_EQ(gfx1250_swmmac->rhs_payload.register_count, 16);
  EXPECT_EQ(gfx1250_swmmac->result_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16);
}

TEST(MatrixContractTest, SparseDescriptorsCarryCompressedFragmentLayouts) {
  for (iree_host_size_t i = 0;
       i < loom_amdgpu_matrix_contract_descriptor_count(); ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
        loom_amdgpu_matrix_contract_descriptor_at(i);
    ASSERT_NE(descriptor, nullptr);
    if (!iree_any_bit_set(descriptor->flags,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE)) {
      continue;
    }
    const loom_amdgpu_matrix_fragment_layout_t* layout =
        loom_amdgpu_matrix_contract_descriptor_fragment_layout(descriptor);
    ASSERT_NE(layout, nullptr) << ToString(descriptor->name);
    ASSERT_TRUE(descriptor->family == LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC ||
                descriptor->family == LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC)
        << ToString(descriptor->name);
    ASSERT_TRUE(descriptor->wave_size_bits == LOOM_AMDGPU_MATRIX_WAVE_SIZE_32 ||
                descriptor->wave_size_bits == LOOM_AMDGPU_MATRIX_WAVE_SIZE_64)
        << ToString(descriptor->name);
    const uint32_t descriptor_wave_size =
        descriptor->wave_size_bits == LOOM_AMDGPU_MATRIX_WAVE_SIZE_64 ? 64 : 32;
    EXPECT_EQ(layout->wave_size, descriptor_wave_size)
        << ToString(descriptor->name);
    EXPECT_EQ(layout->tile_shape.result_row_count,
              descriptor->tile_shape.result_row_count)
        << ToString(descriptor->name);
    EXPECT_EQ(layout->tile_shape.result_column_count,
              descriptor->tile_shape.result_column_count)
        << ToString(descriptor->name);
    EXPECT_EQ(layout->tile_shape.reduction_count,
              descriptor->tile_shape.reduction_count)
        << ToString(descriptor->name);
    EXPECT_EQ(layout->lhs.reduction_group.storage_element_count, 2)
        << ToString(descriptor->name);
    EXPECT_EQ(layout->lhs.reduction_group.logical_element_count, 4)
        << ToString(descriptor->name);
    EXPECT_EQ(layout->rhs.reduction_group.storage_element_count, 0)
        << ToString(descriptor->name);
    EXPECT_EQ(layout->rhs.reduction_group.logical_element_count, 0)
        << ToString(descriptor->name);
    EXPECT_EQ(layout->lhs.register_count,
              descriptor->lhs_payload.register_count)
        << ToString(descriptor->name);
    EXPECT_EQ(layout->lhs.payload_element_count,
              descriptor->lhs_payload.element_count)
        << ToString(descriptor->name);
    EXPECT_EQ(layout->rhs.register_count,
              descriptor->rhs_payload.register_count)
        << ToString(descriptor->name);
    EXPECT_EQ(layout->rhs.payload_element_count,
              descriptor->rhs_payload.element_count)
        << ToString(descriptor->name);
    EXPECT_EQ(layout->accumulator.register_count,
              descriptor->accumulator_payload.register_count)
        << ToString(descriptor->name);
    EXPECT_EQ(layout->result.register_count,
              descriptor->result_payload.register_count)
        << ToString(descriptor->name);
    EXPECT_EQ(
        descriptor->source_requirement_flags &
            LOOM_AMDGPU_MATRIX_CONTRACT_SOURCE_REQUIREMENT_FRAGMENT_LAYOUT,
        0u)
        << ToString(descriptor->name);
  }
}

TEST(MatrixContractTest, WmmaFp8CrossProductDescriptors) {
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      FindDescriptor("wmma.f32.16x16x16.bf8.fp8");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_WMMA);
  EXPECT_EQ(descriptor->tile_shape.reduction_count, 16);
  EXPECT_EQ(descriptor->lhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF8);
  EXPECT_EQ(descriptor->rhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_FP8);
}

TEST(MatrixContractTest, MatrixSemanticTagsDeriveTargetLowIds) {
  struct Case {
    const char* descriptor_name;
    loom_amdgpu_descriptor_ref_t low_descriptor_ref;
  };
  const Case cases[] = {
      {
          "mfma.f32.16x16x32.bf8.bf8",
          LOOM_AMDGPU_DESCRIPTOR_REF_V_MFMA_F32_16X16X32_BF8_BF8,
      },
      {
          "smfmac.f32.16x16x64.bf8.bf8",
          LOOM_AMDGPU_DESCRIPTOR_REF_V_SMFMAC_F32_16X16X64_BF8_BF8,
      },
      {
          "swmmac.f32.16x16x32.f16",
          LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F32_16X16X32_F16,
      },
  };
  for (const Case& test_case : cases) {
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
        FindDescriptor(test_case.descriptor_name);
    ASSERT_NE(descriptor, nullptr) << test_case.descriptor_name;
    EXPECT_EQ(descriptor->low_descriptor_ref, test_case.low_descriptor_ref)
        << test_case.descriptor_name;
  }
}

TEST(MatrixContractTest, WaitStateDescriptorsAreFixedWaitFamilies) {
  const loom_amdgpu_matrix_contract_descriptor_t* mfma =
      loom_amdgpu_matrix_contract_wait_state_descriptor_for_low_descriptor_ref(
          LOOM_AMDGPU_DESCRIPTOR_REF_V_MFMA_F32_16X16X32_BF8_BF8);
  ASSERT_NE(mfma, nullptr);
  EXPECT_EQ(mfma->family, LOOM_AMDGPU_MATRIX_FAMILY_MFMA);
  EXPECT_EQ(mfma->result_payload.register_count, 4);

  const loom_amdgpu_matrix_contract_descriptor_t* smfmac =
      loom_amdgpu_matrix_contract_wait_state_descriptor_for_low_descriptor_ref(
          LOOM_AMDGPU_DESCRIPTOR_REF_V_SMFMAC_F32_16X16X64_BF8_BF8);
  ASSERT_NE(smfmac, nullptr);
  EXPECT_EQ(smfmac->family, LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC);
  EXPECT_EQ(smfmac->result_payload.register_count, 4);

  EXPECT_EQ(
      loom_amdgpu_matrix_contract_wait_state_descriptor_for_low_descriptor_ref(
          LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X16_F16),
      nullptr);
  EXPECT_EQ(
      loom_amdgpu_matrix_contract_wait_state_descriptor_for_low_descriptor_ref(
          LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F32_16X16X32_F16),
      nullptr);
}

TEST(MatrixContractTest, WmmaDescriptorsExposeTargetLowIds) {
  const loom_amdgpu_matrix_contract_descriptor_t* f32_f16 =
      FindDescriptor("wmma.f32.16x16x16.f16");
  ASSERT_NE(f32_f16, nullptr);
  EXPECT_EQ(f32_f16->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X16_F16);

  const loom_amdgpu_matrix_contract_descriptor_t* i32_iu8 =
      FindDescriptor("wmma.i32.16x16x16.iu8");
  ASSERT_NE(i32_iu8, nullptr);
  EXPECT_EQ(i32_iu8->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU8);
  EXPECT_EQ(i32_iu8->lhs_payload.register_count, 4);
  EXPECT_EQ(i32_iu8->accumulator_payload.register_count, 8);
  EXPECT_EQ(i32_iu8->wave_size_bits, LOOM_AMDGPU_MATRIX_WAVE_SIZE_32);
  EXPECT_EQ(i32_iu8->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT);
  EXPECT_EQ(i32_iu8->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP);

  const loom_amdgpu_matrix_contract_descriptor_t* i32_iu8_w64 =
      FindDescriptor("wmma.i32.16x16x16.iu8.w64");
  ASSERT_NE(i32_iu8_w64, nullptr);
  EXPECT_EQ(i32_iu8_w64->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU8_W64);
  EXPECT_EQ(i32_iu8_w64->lhs_payload.register_count, 4);
  EXPECT_EQ(i32_iu8_w64->accumulator_payload.register_count, 4);
  EXPECT_EQ(i32_iu8_w64->wave_size_bits, LOOM_AMDGPU_MATRIX_WAVE_SIZE_64);
  EXPECT_EQ(i32_iu8_w64->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT);
  EXPECT_EQ(i32_iu8_w64->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP);

  const loom_amdgpu_matrix_contract_descriptor_t* i32_iu4 =
      FindDescriptor("wmma.i32.16x16x16.iu4");
  ASSERT_NE(i32_iu4, nullptr);
  EXPECT_EQ(i32_iu4->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU4);
  EXPECT_EQ(i32_iu4->lhs_payload.register_count, 2);
  EXPECT_EQ(i32_iu4->accumulator_payload.register_count, 8);
  EXPECT_EQ(i32_iu4->wave_size_bits, LOOM_AMDGPU_MATRIX_WAVE_SIZE_32);

  const loom_amdgpu_matrix_contract_descriptor_t* i32_iu4_w64 =
      FindDescriptor("wmma.i32.16x16x16.iu4.w64");
  ASSERT_NE(i32_iu4_w64, nullptr);
  EXPECT_EQ(i32_iu4_w64->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU4_W64);
  EXPECT_EQ(i32_iu4_w64->lhs_payload.register_count, 2);
  EXPECT_EQ(i32_iu4_w64->accumulator_payload.register_count, 4);
  EXPECT_EQ(i32_iu4_w64->wave_size_bits, LOOM_AMDGPU_MATRIX_WAVE_SIZE_64);

  const loom_amdgpu_matrix_contract_descriptor_t* f32_f16_gfx1250 =
      FindDescriptor("wmma.f32.16x16x32.f16");
  ASSERT_NE(f32_f16_gfx1250, nullptr);
  EXPECT_EQ(f32_f16_gfx1250->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X32_F16);

  const loom_amdgpu_matrix_contract_descriptor_t* scaled =
      FindDescriptor("wmma.scale.f32.16x16x128.f8f6f4.f8.f8");
  ASSERT_NE(scaled, nullptr);
  EXPECT_EQ(scaled->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_SCALE_F32_16X16X128_F8F6F4_F8_F8);
}

TEST(MatrixContractTest, Cdna3DescriptorsExposeTargetLowIds) {
  const loom_amdgpu_matrix_contract_descriptor_t* mfma_f32_f16 =
      FindDescriptor("mfma.f32.16x16x16.f16");
  ASSERT_NE(mfma_f32_f16, nullptr);
  EXPECT_EQ(mfma_f32_f16->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_MFMA_F32_16X16X16_F16);

  const loom_amdgpu_matrix_contract_descriptor_t* mfma_f32_bf16 =
      FindDescriptor("mfma.f32.16x16x16.bf16.1k");
  ASSERT_NE(mfma_f32_bf16, nullptr);
  EXPECT_EQ(mfma_f32_bf16->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_MFMA_F32_16X16X16_BF16);

  const loom_amdgpu_matrix_contract_descriptor_t* smfmac_f32_bf16 =
      FindDescriptor("smfmac.f32.16x16x32.bf16");
  ASSERT_NE(smfmac_f32_bf16, nullptr);
  EXPECT_EQ(smfmac_f32_bf16->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_SMFMAC_F32_16X16X32_BF16);
  EXPECT_EQ(smfmac_f32_bf16->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE);

  const loom_amdgpu_matrix_contract_descriptor_t* smfmac_f32_f16 =
      FindDescriptor("smfmac.f32.32x32x16.f16");
  ASSERT_NE(smfmac_f32_f16, nullptr);
  EXPECT_EQ(smfmac_f32_f16->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_SMFMAC_F32_32X32X16_F16);
  EXPECT_EQ(smfmac_f32_f16->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE);
}

TEST(MatrixContractTest, GeneratedFragmentLayoutsMatchDescriptors) {
  iree_host_size_t fragment_layout_count = 0;
  const iree_host_size_t descriptor_count =
      loom_amdgpu_matrix_contract_descriptor_count();
  for (iree_host_size_t i = 0; i < descriptor_count; ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
        loom_amdgpu_matrix_contract_descriptor_at(i);
    ASSERT_NE(descriptor, nullptr);
    SCOPED_TRACE(ToString(descriptor->name));

    const loom_amdgpu_matrix_fragment_layout_t* layout =
        loom_amdgpu_matrix_contract_descriptor_fragment_layout(descriptor);
    if (descriptor->fragment_layout_kind ==
        LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_UNKNOWN) {
      EXPECT_EQ(layout, nullptr);
      continue;
    }

    ++fragment_layout_count;
    ASSERT_NE(layout, nullptr);
    EXPECT_EQ(layout->kind, descriptor->fragment_layout_kind);
    EXPECT_FALSE(iree_string_view_is_empty(layout->name));
    EXPECT_EQ(layout->tile_shape.block_count,
              descriptor->tile_shape.block_count);
    EXPECT_EQ(layout->tile_shape.result_row_count,
              descriptor->tile_shape.result_row_count);
    EXPECT_EQ(layout->tile_shape.result_column_count,
              descriptor->tile_shape.result_column_count);
    EXPECT_EQ(layout->tile_shape.reduction_count,
              descriptor->tile_shape.reduction_count);

    loom_amdgpu_matrix_wave_size_bits_t layout_wave_size_bit = 0;
    switch (layout->wave_size) {
      case 32:
        layout_wave_size_bit = LOOM_AMDGPU_MATRIX_WAVE_SIZE_32;
        break;
      case 64:
        layout_wave_size_bit = LOOM_AMDGPU_MATRIX_WAVE_SIZE_64;
        break;
      default:
        ADD_FAILURE() << "unsupported fragment layout wave size "
                      << layout->wave_size;
        continue;
    }
    EXPECT_TRUE(
        iree_any_bit_set(descriptor->wave_size_bits, layout_wave_size_bit));

    const loom_amdgpu_matrix_payload_shape_t* payloads[] = {
        &descriptor->lhs_payload,
        &descriptor->rhs_payload,
        &descriptor->accumulator_payload,
        &descriptor->result_payload,
    };
    const loom_matrix_fragment_role_layout_t* role_layouts[] = {
        &layout->lhs,
        &layout->rhs,
        &layout->accumulator,
        &layout->result,
    };
    const loom_contract_operand_role_t roles[] = {
        LOOM_CONTRACT_OPERAND_ROLE_LHS,
        LOOM_CONTRACT_OPERAND_ROLE_RHS,
        LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR,
        LOOM_CONTRACT_OPERAND_ROLE_RESULT,
    };
    for (iree_host_size_t role_index = 0; role_index < IREE_ARRAYSIZE(roles);
         ++role_index) {
      EXPECT_EQ(role_layouts[role_index]->role, roles[role_index]);
      EXPECT_EQ(role_layouts[role_index]->register_count,
                payloads[role_index]->register_count);
      EXPECT_EQ(role_layouts[role_index]->payload_element_count,
                payloads[role_index]->element_count);
      EXPECT_NE(role_layouts[role_index]->coordinate_projection_plan, nullptr);
    }
  }
  EXPECT_GT(fragment_layout_count, 0u);
}

TEST(MatrixContractTest, Gfx11F32ResultPublishesAdjacentF16Pairs) {
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_fragment_layout_for_kind(
          LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16_F16);
  ASSERT_NE(layout, nullptr);

  const loom_matrix_fragment_packed_b16_publication_t& row =
      layout->result.packed_b16_publications.row;
  EXPECT_EQ(row.publishing_participant_and_mask, 16);
  EXPECT_EQ(row.publishing_participant_equal_value, 0);
  EXPECT_EQ(row.paired_participant_xor_mask, 16);

  const loom_matrix_fragment_packed_b16_publication_t& column =
      layout->result.packed_b16_publications.column;
  EXPECT_EQ(column.publishing_participant_and_mask, 1);
  EXPECT_EQ(column.publishing_participant_equal_value, 0);
  EXPECT_EQ(column.paired_participant_xor_mask, 1);
}

TEST(MatrixContractTest, MatcherRejectsGfx12WmmaPayloadWithoutGfx12Feature) {
  loom_amdgpu_matrix_contract_match_request_t request = MatchRequest(
      LOOM_AMDGPU_MATRIX_FAMILY_WMMA, 16, 16, 16,
      LOOM_AMDGPU_MATRIX_NUMERIC_F16, LOOM_AMDGPU_MATRIX_NUMERIC_F16,
      LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
      LOOM_AMDGPU_MATRIX_SCALE_NONE, LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11, 32,
      0, 0);
  request.lhs_payload.register_count = 4;
  request.lhs_payload.element_count = 8;
  request.rhs_payload.register_count = 4;
  request.rhs_payload.element_count = 8;
  request.accumulator_payload.register_count = 8;
  request.accumulator_payload.element_count = 8;
  request.result_payload.register_count = 8;
  request.result_payload.element_count = 8;

  loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      loom_amdgpu_matrix_contract_select(&request, &diagnostic);
  EXPECT_EQ(descriptor, nullptr);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_FEATURES);
}

TEST(MatrixContractTest, MatcherSelectsMatchingDescriptor) {
  loom_amdgpu_matrix_contract_match_request_t request = MatchRequest(
      LOOM_AMDGPU_MATRIX_FAMILY_MFMA, 16, 16, 32,
      LOOM_AMDGPU_MATRIX_NUMERIC_FP8, LOOM_AMDGPU_MATRIX_NUMERIC_FP8,
      LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
      LOOM_AMDGPU_MATRIX_SCALE_NONE, LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_FP8,
      64, 0, 0);
  loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      loom_amdgpu_matrix_contract_select(&request, &diagnostic);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_MFMA);
  EXPECT_EQ(descriptor->tile_shape.result_row_count, 16);
  EXPECT_EQ(descriptor->tile_shape.result_column_count, 16);
  EXPECT_EQ(descriptor->tile_shape.reduction_count, 32);
  EXPECT_EQ(descriptor->lhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_FP8);
  EXPECT_EQ(descriptor->rhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_FP8);
  EXPECT_EQ(descriptor->accumulator_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32);
  EXPECT_EQ(descriptor->result_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32);
  EXPECT_EQ(descriptor->scale_kind, LOOM_AMDGPU_MATRIX_SCALE_NONE);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE);
  EXPECT_EQ(diagnostic.wave_candidate_count, 1u);
}

TEST(MatrixContractTest, MatcherSelectsDefinedGfx1250SparseBf16Semantics) {
  EXPECT_EQ(FindDescriptor("swmmac.bf16f32.16x16x64.bf16"), nullptr);

  const loom_amdgpu_matrix_contract_flags_t source_flags =
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_AB_MODIFIERS |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE;
  loom_amdgpu_matrix_contract_match_request_t request = MatchRequest(
      LOOM_AMDGPU_MATRIX_FAMILY_UNKNOWN, 16, 16, 64,
      LOOM_AMDGPU_MATRIX_NUMERIC_BF16, LOOM_AMDGPU_MATRIX_NUMERIC_BF16,
      LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
      LOOM_AMDGPU_MATRIX_SCALE_NONE, LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX1250,
      32, source_flags, source_flags);
  request.lhs_payload.register_count = 8;
  request.lhs_payload.element_count = 16;
  request.rhs_payload.register_count = 16;
  request.rhs_payload.element_count = 32;
  request.accumulator_payload.register_count = 8;
  request.accumulator_payload.element_count = 8;
  request.result_payload.register_count = 8;
  request.result_payload.element_count = 8;

  loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      loom_amdgpu_matrix_contract_select(&request, &diagnostic);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor->name), "swmmac.f32.16x16x64.bf16");
  EXPECT_EQ(descriptor->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F32_16X16X64_BF16);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE);
}

TEST(MatrixContractTest, MatcherSelectsRdnaIntegerWmmaLowDescriptors) {
  const loom_amdgpu_matrix_contract_flags_t integer_wmma_flags =
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP;
  const loom_amdgpu_matrix_feature_bits_t gfx11_features =
      LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11;
  const loom_amdgpu_matrix_feature_bits_t gfx12_features =
      LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11 |
      LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX12 |
      LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX12;
  struct Case {
    // Matrix feature bits available on the selected target.
    loom_amdgpu_matrix_feature_bits_t feature_bits;
    // Matrix input numeric type requested by the source contract.
    loom_amdgpu_matrix_numeric_type_t numeric_type;
    // Number of VGPRs carrying each source operand.
    uint16_t input_register_count;
    // Number of unpacked source elements carried by each source operand.
    uint16_t input_element_count;
    // Number of VGPRs carrying the accumulator and result operands.
    uint16_t accumulator_register_count;
    // Number of accumulator/result elements carried by each operand.
    uint16_t accumulator_element_count;
    // Requested subgroup size.
    uint32_t wave_size;
    // Expected target-low descriptor ref for native lowering.
    loom_amdgpu_descriptor_ref_t expected_low_descriptor_ref;
  };
  const Case cases[] = {
      {gfx11_features, LOOM_AMDGPU_MATRIX_NUMERIC_IU8, 4, 16, 8, 8, 32,
       LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU8},
      {gfx11_features, LOOM_AMDGPU_MATRIX_NUMERIC_I8, 4, 16, 8, 8, 32,
       LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU8},
      {gfx11_features, LOOM_AMDGPU_MATRIX_NUMERIC_IU4, 2, 16, 8, 8, 32,
       LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU4},
      {gfx11_features, LOOM_AMDGPU_MATRIX_NUMERIC_I4, 2, 16, 8, 8, 32,
       LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU4},
      {gfx12_features, LOOM_AMDGPU_MATRIX_NUMERIC_IU8, 4, 16, 8, 8, 32,
       LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU8},
      {gfx12_features, LOOM_AMDGPU_MATRIX_NUMERIC_I8, 4, 16, 8, 8, 32,
       LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU8},
      {gfx12_features, LOOM_AMDGPU_MATRIX_NUMERIC_IU4, 2, 16, 8, 8, 32,
       LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU4},
      {gfx12_features, LOOM_AMDGPU_MATRIX_NUMERIC_I4, 2, 16, 8, 8, 32,
       LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU4},
      {gfx11_features, LOOM_AMDGPU_MATRIX_NUMERIC_IU8, 4, 16, 4, 4, 64,
       LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU8_W64},
      {gfx11_features, LOOM_AMDGPU_MATRIX_NUMERIC_I8, 4, 16, 4, 4, 64,
       LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU8_W64},
      {gfx11_features, LOOM_AMDGPU_MATRIX_NUMERIC_IU4, 2, 16, 4, 4, 64,
       LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU4_W64},
      {gfx11_features, LOOM_AMDGPU_MATRIX_NUMERIC_I4, 2, 16, 4, 4, 64,
       LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU4_W64},
  };
  for (const Case& test_case : cases) {
    loom_amdgpu_matrix_contract_match_request_t request = MatchRequest(
        LOOM_AMDGPU_MATRIX_FAMILY_WMMA, 16, 16, 16, test_case.numeric_type,
        test_case.numeric_type, LOOM_AMDGPU_MATRIX_NUMERIC_I32,
        LOOM_AMDGPU_MATRIX_NUMERIC_I32, LOOM_AMDGPU_MATRIX_SCALE_NONE,
        test_case.feature_bits, test_case.wave_size, integer_wmma_flags, 0);
    request.lhs_payload.register_count = test_case.input_register_count;
    request.lhs_payload.element_count = test_case.input_element_count;
    request.rhs_payload.register_count = test_case.input_register_count;
    request.rhs_payload.element_count = test_case.input_element_count;
    request.accumulator_payload.register_count =
        test_case.accumulator_register_count;
    request.accumulator_payload.element_count =
        test_case.accumulator_element_count;
    request.result_payload.register_count =
        test_case.accumulator_register_count;
    request.result_payload.element_count = test_case.accumulator_element_count;
    loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
        loom_amdgpu_matrix_contract_select(&request, &diagnostic);
    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_WMMA);
    const loom_amdgpu_matrix_numeric_type_t descriptor_numeric_type =
        test_case.numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_I8
            ? LOOM_AMDGPU_MATRIX_NUMERIC_IU8
        : test_case.numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_I4
            ? LOOM_AMDGPU_MATRIX_NUMERIC_IU4
            : test_case.numeric_type;
    EXPECT_EQ(descriptor->lhs_payload.numeric_type, descriptor_numeric_type);
    EXPECT_EQ(descriptor->rhs_payload.numeric_type, descriptor_numeric_type);
    EXPECT_EQ(descriptor->low_descriptor_ref,
              test_case.expected_low_descriptor_ref);
    EXPECT_EQ(diagnostic.rejection_bits,
              LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE);
  }
}

TEST(MatrixContractTest, MatcherRejectsMissingFeatureAfterSemanticMatch) {
  const loom_amdgpu_matrix_contract_flags_t scale_flags =
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS;
  loom_amdgpu_matrix_contract_match_request_t request = MatchRequest(
      LOOM_AMDGPU_MATRIX_FAMILY_MFMA, 32, 32, 64,
      LOOM_AMDGPU_MATRIX_NUMERIC_FP4, LOOM_AMDGPU_MATRIX_NUMERIC_FP4,
      LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
      LOOM_AMDGPU_MATRIX_SCALE_32, LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950, 64,
      scale_flags, scale_flags);
  loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      loom_amdgpu_matrix_contract_select(&request, &diagnostic);
  EXPECT_EQ(descriptor, nullptr);
  EXPECT_EQ(diagnostic.flag_candidate_count, 1u);
  EXPECT_EQ(diagnostic.feature_candidate_count, 0u);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_FEATURES);
}

TEST(MatrixContractTest, MatcherRejectsTileShapeAndPayloadMismatches) {
  loom_amdgpu_matrix_contract_match_request_t shape_request = MatchRequest(
      LOOM_AMDGPU_MATRIX_FAMILY_MFMA, 17, 16, 32,
      LOOM_AMDGPU_MATRIX_NUMERIC_FP8, LOOM_AMDGPU_MATRIX_NUMERIC_FP8,
      LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
      LOOM_AMDGPU_MATRIX_SCALE_NONE, LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_FP8,
      64, 0, 0);
  loom_amdgpu_matrix_contract_match_diagnostic_t shape_diagnostic = {};
  EXPECT_EQ(
      loom_amdgpu_matrix_contract_select(&shape_request, &shape_diagnostic),
      nullptr);
  EXPECT_EQ(shape_diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_TILE_SHAPE);

  loom_amdgpu_matrix_contract_match_request_t payload_request = MatchRequest(
      LOOM_AMDGPU_MATRIX_FAMILY_MFMA, 16, 16, 32,
      LOOM_AMDGPU_MATRIX_NUMERIC_F16, LOOM_AMDGPU_MATRIX_NUMERIC_FP8,
      LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
      LOOM_AMDGPU_MATRIX_SCALE_NONE, LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_FP8,
      64, 0, 0);
  loom_amdgpu_matrix_contract_match_diagnostic_t payload_diagnostic = {};
  EXPECT_EQ(
      loom_amdgpu_matrix_contract_select(&payload_request, &payload_diagnostic),
      nullptr);
  EXPECT_EQ(payload_diagnostic.rejection_bits &
                LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_LHS_PAYLOAD,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_LHS_PAYLOAD);
}

TEST(MatrixContractTest, MatcherRejectsMissingMatrixFormatFacts) {
  const loom_amdgpu_matrix_contract_flags_t available_flags =
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED;
  const loom_amdgpu_matrix_contract_flags_t required_flags =
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS;
  loom_amdgpu_matrix_contract_match_request_t request = MatchRequest(
      LOOM_AMDGPU_MATRIX_FAMILY_MFMA, 32, 32, 64,
      LOOM_AMDGPU_MATRIX_NUMERIC_FP4, LOOM_AMDGPU_MATRIX_NUMERIC_FP4,
      LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
      LOOM_AMDGPU_MATRIX_SCALE_32,
      LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950 |
          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950_SCALE_F8F6F4,
      64, available_flags, required_flags);
  loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(&request, &diagnostic), nullptr);
  EXPECT_EQ(diagnostic.rejection_bits &
                LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_MATRIX_FORMATS,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_MATRIX_FORMATS);
}

TEST(MatrixContractTest, MatcherAcceptsImplicitScaleFormatFacts) {
  const loom_amdgpu_matrix_contract_flags_t available_flags =
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS;
  const loom_amdgpu_matrix_contract_flags_t required_flags =
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED;
  loom_amdgpu_matrix_contract_match_request_t request = MatchRequest(
      LOOM_AMDGPU_MATRIX_FAMILY_MFMA, 16, 16, 128,
      LOOM_AMDGPU_MATRIX_NUMERIC_FP4, LOOM_AMDGPU_MATRIX_NUMERIC_FP4,
      LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
      LOOM_AMDGPU_MATRIX_SCALE_32,
      LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950 |
          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950_SCALE_F8F6F4,
      64, available_flags, required_flags);
  request.lhs_scale_format_selector_bits =
      1u << LOOM_AMDGPU_MATRIX_SCALE_FORMAT_SELECTOR_E8M0;
  request.rhs_scale_format_selector_bits =
      1u << LOOM_AMDGPU_MATRIX_SCALE_FORMAT_SELECTOR_E8M0;

  loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      loom_amdgpu_matrix_contract_select(&request, &diagnostic);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_MFMA_SCALE_F32_16X16X128_F8F6F4_F4_F4);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE);
}

TEST(MatrixContractTest, MatcherRejectsImplicitScaleFormatMismatch) {
  const loom_amdgpu_matrix_contract_flags_t available_flags =
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS;
  const loom_amdgpu_matrix_contract_flags_t required_flags =
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED;
  loom_amdgpu_matrix_contract_match_request_t request = MatchRequest(
      LOOM_AMDGPU_MATRIX_FAMILY_MFMA, 16, 16, 128,
      LOOM_AMDGPU_MATRIX_NUMERIC_FP4, LOOM_AMDGPU_MATRIX_NUMERIC_FP4,
      LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
      LOOM_AMDGPU_MATRIX_SCALE_32,
      LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950 |
          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950_SCALE_F8F6F4,
      64, available_flags, required_flags);
  request.lhs_scale_format_selector_bits =
      1u << LOOM_AMDGPU_MATRIX_SCALE_FORMAT_SELECTOR_FP8_E4M3;
  request.rhs_scale_format_selector_bits =
      1u << LOOM_AMDGPU_MATRIX_SCALE_FORMAT_SELECTOR_FP8_E4M3;

  loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(&request, &diagnostic), nullptr);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_SCALE_FORMAT);
}

TEST(MatrixContractTest, MatcherRejectsScaleKindMismatch) {
  loom_amdgpu_matrix_contract_match_request_t request = MatchRequest(
      LOOM_AMDGPU_MATRIX_FAMILY_MFMA, 32, 32, 64,
      LOOM_AMDGPU_MATRIX_NUMERIC_FP4, LOOM_AMDGPU_MATRIX_NUMERIC_FP4,
      LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
      LOOM_AMDGPU_MATRIX_SCALE_16,
      LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950 |
          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950_SCALE_F8F6F4,
      64, 0, 0);
  loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(&request, &diagnostic), nullptr);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_SCALE_KIND);
}

TEST(MatrixContractTest, MatcherRejectsSparseAndReuseRequirements) {
  loom_amdgpu_matrix_contract_match_request_t sparse_request = MatchRequest(
      LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC, 16, 16, 32,
      LOOM_AMDGPU_MATRIX_NUMERIC_IU8, LOOM_AMDGPU_MATRIX_NUMERIC_IU8,
      LOOM_AMDGPU_MATRIX_NUMERIC_I32, LOOM_AMDGPU_MATRIX_NUMERIC_I32,
      LOOM_AMDGPU_MATRIX_SCALE_NONE, LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX12,
      32,
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT |
          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP,
      0);
  loom_amdgpu_matrix_contract_match_diagnostic_t sparse_diagnostic = {};
  EXPECT_EQ(
      loom_amdgpu_matrix_contract_select(&sparse_request, &sparse_diagnostic),
      nullptr);
  EXPECT_EQ(sparse_diagnostic.rejection_bits &
                LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SPARSE,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SPARSE);

  const loom_amdgpu_matrix_contract_flags_t modifier_flags =
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_AB_MODIFIERS |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER;
  loom_amdgpu_matrix_contract_match_request_t reuse_request = MatchRequest(
      LOOM_AMDGPU_MATRIX_FAMILY_WMMA, 16, 16, 4, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
      LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
      LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_SCALE_NONE,
      LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250, 32, modifier_flags, 0);
  loom_amdgpu_matrix_contract_match_diagnostic_t reuse_diagnostic = {};
  EXPECT_EQ(
      loom_amdgpu_matrix_contract_select(&reuse_request, &reuse_diagnostic),
      nullptr);
  EXPECT_EQ(reuse_diagnostic.rejection_bits &
                LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_REUSE,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_REUSE);
}

TEST(MatrixContractTest, MatcherRejectsUnsupportedRequiredFlags) {
  loom_amdgpu_matrix_contract_match_request_t request = MatchRequest(
      LOOM_AMDGPU_MATRIX_FAMILY_MFMA, 16, 16, 32,
      LOOM_AMDGPU_MATRIX_NUMERIC_FP8, LOOM_AMDGPU_MATRIX_NUMERIC_FP8,
      LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
      LOOM_AMDGPU_MATRIX_SCALE_NONE, LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_FP8,
      64, LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE,
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE);
  loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(&request, &diagnostic), nullptr);
  EXPECT_EQ(diagnostic.rejection_bits &
                LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_REQUIRED_FLAGS,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_REQUIRED_FLAGS);
  EXPECT_EQ(diagnostic.rejection_bits &
                LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_REUSE,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_REUSE);
}

TEST(MatrixContractTest, MatcherRejectsUnsupportedWaveSize) {
  loom_amdgpu_matrix_contract_match_request_t request = MatchRequest(
      LOOM_AMDGPU_MATRIX_FAMILY_MFMA, 16, 16, 32,
      LOOM_AMDGPU_MATRIX_NUMERIC_FP8, LOOM_AMDGPU_MATRIX_NUMERIC_FP8,
      LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
      LOOM_AMDGPU_MATRIX_SCALE_NONE, LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_FP8,
      48, 0, 0);
  loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(&request, &diagnostic), nullptr);
  EXPECT_EQ(diagnostic.feature_candidate_count, 1u);
  EXPECT_EQ(diagnostic.wave_candidate_count, 0u);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_WAVE_SIZE);
}

TEST(MatrixContractTest, MatcherRejectsInvalidRequest) {
  loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(nullptr, &diagnostic), nullptr);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_INVALID_REQUEST);
}

TEST(MatrixContractTest, ProcessorFeatureBitsGateAvailability) {
  loom_amdgpu_matrix_feature_bits_t gfx908_features = 0;
  IREE_ASSERT_OK(loom_amdgpu_matrix_feature_bits_from_processor(
      IREE_SV("gfx908"), &gfx908_features));
  loom_amdgpu_matrix_feature_bits_t gfx942_features = 0;
  IREE_ASSERT_OK(loom_amdgpu_matrix_feature_bits_from_processor(
      IREE_SV("gfx942"), &gfx942_features));
  loom_amdgpu_matrix_feature_bits_t gfx90a_features = 0;
  IREE_ASSERT_OK(loom_amdgpu_matrix_feature_bits_from_processor(
      IREE_SV("gfx90a"), &gfx90a_features));
  loom_amdgpu_matrix_feature_bits_t gfx950_features = 0;
  IREE_ASSERT_OK(loom_amdgpu_matrix_feature_bits_from_processor(
      IREE_SV("gfx950"), &gfx950_features));
  loom_amdgpu_matrix_feature_bits_t gfx9_4_generic_features = 0;
  IREE_ASSERT_OK(loom_amdgpu_matrix_feature_bits_from_processor(
      IREE_SV("gfx9-4-generic"), &gfx9_4_generic_features));
  loom_amdgpu_matrix_feature_bits_t gfx1250_features = 0;
  IREE_ASSERT_OK(loom_amdgpu_matrix_feature_bits_from_processor(
      IREE_SV("gfx1250"), &gfx1250_features));

  const loom_amdgpu_matrix_contract_descriptor_t* fp8_mfma =
      FindDescriptor("mfma.f32.16x16x32.fp8.fp8");
  ASSERT_NE(fp8_mfma, nullptr);
  EXPECT_TRUE(
      loom_amdgpu_matrix_contract_is_available(fp8_mfma, gfx942_features, 64));
  EXPECT_FALSE(loom_amdgpu_matrix_contract_is_available(
      fp8_mfma, gfx9_4_generic_features, 64));

  const loom_amdgpu_matrix_contract_descriptor_t* fp8_smfmac =
      FindDescriptor("smfmac.f32.16x16x64.fp8.fp8");
  ASSERT_NE(fp8_smfmac, nullptr);
  EXPECT_TRUE(loom_amdgpu_matrix_contract_is_available(fp8_smfmac,
                                                       gfx942_features, 64));
  EXPECT_TRUE(loom_amdgpu_matrix_contract_is_available(fp8_smfmac,
                                                       gfx950_features, 64));
  EXPECT_FALSE(loom_amdgpu_matrix_contract_is_available(
      fp8_smfmac, gfx9_4_generic_features, 64));

  const loom_amdgpu_matrix_contract_descriptor_t* bf16_smfmac =
      FindDescriptor("smfmac.f32.16x16x32.bf16");
  ASSERT_NE(bf16_smfmac, nullptr);
  EXPECT_TRUE(loom_amdgpu_matrix_contract_is_available(
      bf16_smfmac, gfx9_4_generic_features, 64));

  const loom_amdgpu_matrix_contract_descriptor_t* i8_mfma =
      FindDescriptor("mfma.i32.16x16x32.i8");
  ASSERT_NE(i8_mfma, nullptr);
  EXPECT_TRUE(loom_amdgpu_matrix_contract_is_available(
      i8_mfma, gfx9_4_generic_features, 64));

  const loom_amdgpu_matrix_contract_descriptor_t* double_mfma =
      FindDescriptor("mfma.f64.16x16x4.f64");
  ASSERT_NE(double_mfma, nullptr);
  EXPECT_FALSE(loom_amdgpu_matrix_contract_is_available(double_mfma,
                                                        gfx908_features, 64));
  EXPECT_TRUE(loom_amdgpu_matrix_contract_is_available(double_mfma,
                                                       gfx90a_features, 64));

  const loom_amdgpu_matrix_contract_descriptor_t* scaled_mfma =
      FindDescriptor("mfma.scale.f32.16x16x128.f8f6f4.f4.f4");
  ASSERT_NE(scaled_mfma, nullptr);
  EXPECT_FALSE(loom_amdgpu_matrix_contract_is_available(scaled_mfma,
                                                        gfx942_features, 64));
  EXPECT_TRUE(loom_amdgpu_matrix_contract_is_available(scaled_mfma,
                                                       gfx950_features, 64));

  const loom_amdgpu_matrix_contract_descriptor_t* scaled_wmma =
      FindDescriptor("wmma.scale.f32.16x16x128.f8f6f4.f8.f8");
  ASSERT_NE(scaled_wmma, nullptr);
  EXPECT_FALSE(loom_amdgpu_matrix_contract_is_available(scaled_wmma,
                                                        gfx950_features, 64));
  EXPECT_TRUE(loom_amdgpu_matrix_contract_is_available(scaled_wmma,
                                                       gfx1250_features, 32));
}

TEST(MatrixContractTest, ScaleFeatureDoesNotGateUnscaledDescriptors) {
  const loom_amdgpu_matrix_contract_descriptor_t* unscaled_mfma =
      FindDescriptor("mfma.f32.16x16x32.f16");
  ASSERT_NE(unscaled_mfma, nullptr);
  EXPECT_TRUE(loom_amdgpu_matrix_contract_is_available(
      unscaled_mfma, LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950, 64));

  const loom_amdgpu_matrix_contract_descriptor_t* scaled_mfma =
      FindDescriptor("mfma.scale.f32.16x16x128.f8f6f4.f4.f4");
  ASSERT_NE(scaled_mfma, nullptr);
  EXPECT_FALSE(loom_amdgpu_matrix_contract_is_available(
      scaled_mfma, LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950, 64));

  const loom_amdgpu_matrix_contract_descriptor_t* unscaled_wmma =
      FindDescriptor("wmma.f32.16x16x32.f16");
  ASSERT_NE(unscaled_wmma, nullptr);
  EXPECT_TRUE(loom_amdgpu_matrix_contract_is_available(
      unscaled_wmma, LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250, 32));

  const loom_amdgpu_matrix_contract_descriptor_t* unscaled_f8_wmma =
      FindDescriptor("wmma.f32.16x16x128.f8f6f4.f8.f8");
  ASSERT_NE(unscaled_f8_wmma, nullptr);
  EXPECT_TRUE(loom_amdgpu_matrix_contract_is_available(
      unscaled_f8_wmma, LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250, 32));

  const loom_amdgpu_matrix_contract_descriptor_t* scaled_wmma =
      FindDescriptor("wmma.scale.f32.16x16x128.f8f6f4.f8.f8");
  ASSERT_NE(scaled_wmma, nullptr);
  EXPECT_FALSE(loom_amdgpu_matrix_contract_is_available(
      scaled_wmma, LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250, 32));
}

TEST(MatrixContractTest, ProcessorFeatureBitsRejectUnknownProcessor) {
  loom_amdgpu_matrix_feature_bits_t feature_bits = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_amdgpu_matrix_feature_bits_from_processor(
                            IREE_SV("gfx9999"), &feature_bits));
}

TEST(MatrixContractTest, ProcessorFeatureBitsMatchEveryTargetInfoProfile) {
  const iree_host_size_t processor_count =
      loom_amdgpu_target_info_processor_count();
  ASSERT_GT(processor_count, 0u);
  for (iree_host_size_t i = 0; i < processor_count; ++i) {
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_processor_at(i);
    ASSERT_NE(processor, nullptr);

    loom_amdgpu_matrix_feature_bits_t expected_feature_bits = 0;
    const bool has_matrix_profile =
        loom_amdgpu_matrix_feature_bits_from_profile(
            processor->properties.features.matrix, &expected_feature_bits);

    loom_amdgpu_matrix_feature_bits_t actual_feature_bits = 0;
    iree_status_t status = loom_amdgpu_matrix_feature_bits_from_processor(
        processor->name, &actual_feature_bits);
    if (has_matrix_profile) {
      IREE_EXPECT_OK(status) << ToString(processor->name);
      EXPECT_EQ(actual_feature_bits, expected_feature_bits)
          << ToString(processor->name);
    } else {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, status)
          << ToString(processor->name);
      EXPECT_EQ(actual_feature_bits, 0u) << ToString(processor->name);
    }
  }
}

TEST(MatrixContractTest, Gfx1250RejectsLegacyWmmaDescriptors) {
  loom_amdgpu_matrix_feature_bits_t gfx1250_features = 0;
  IREE_ASSERT_OK(loom_amdgpu_matrix_feature_bits_from_processor(
      IREE_SV("gfx1250"), &gfx1250_features));

  const loom_amdgpu_matrix_contract_descriptor_t* gfx11 =
      FindDescriptor("wmma.f32.16x16x16.bf16");
  const loom_amdgpu_matrix_contract_descriptor_t* gfx12 =
      FindDescriptor("wmma.f32.16x16x16.bf16.gfx12");
  const loom_amdgpu_matrix_contract_descriptor_t* gfx1250 =
      FindDescriptor("wmma.f32.16x16x32.bf16");
  ASSERT_NE(gfx11, nullptr);
  ASSERT_NE(gfx12, nullptr);
  ASSERT_NE(gfx1250, nullptr);
  EXPECT_FALSE(
      loom_amdgpu_matrix_contract_is_available(gfx11, gfx1250_features, 32));
  EXPECT_FALSE(
      loom_amdgpu_matrix_contract_is_available(gfx12, gfx1250_features, 32));
  EXPECT_TRUE(
      loom_amdgpu_matrix_contract_is_available(gfx1250, gfx1250_features, 32));
}

TEST(MatrixContractTest, NamesAreStable) {
  EXPECT_EQ(ToString(loom_amdgpu_matrix_family_name(
                LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC)),
            "swmmac");
  EXPECT_EQ(ToString(loom_amdgpu_matrix_numeric_type_name(
                LOOM_AMDGPU_MATRIX_NUMERIC_F8)),
            "f8");
  EXPECT_EQ(ToString(loom_amdgpu_matrix_numeric_type_name(
                LOOM_AMDGPU_MATRIX_NUMERIC_F6)),
            "f6");
  EXPECT_EQ(ToString(loom_amdgpu_matrix_numeric_type_name(
                LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4)),
            "f8f6f4");
  EXPECT_EQ(
      ToString(loom_amdgpu_matrix_scale_kind_name(LOOM_AMDGPU_MATRIX_SCALE_16)),
      "scale16");
}

}  // namespace
