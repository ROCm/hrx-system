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

void ExpectFragmentRoleLayout(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role,
    loom_amdgpu_matrix_fragment_map_kind_t map_kind, uint16_t register_count,
    uint16_t elements_per_register, uint16_t element_bit_count,
    loom_amdgpu_matrix_fragment_coordinate_flags_t coordinate_flags) {
  const loom_amdgpu_matrix_fragment_role_layout_t* role_layout =
      loom_amdgpu_matrix_fragment_role_layout(layout, role);
  ASSERT_NE(role_layout, nullptr);
  EXPECT_EQ(role_layout->role, role);
  EXPECT_EQ(role_layout->map_kind,
            static_cast<loom_matrix_fragment_map_kind_t>(map_kind));
  EXPECT_EQ(role_layout->register_count, register_count);
  EXPECT_EQ(role_layout->elements_per_register, elements_per_register);
  EXPECT_EQ(role_layout->element_bit_count, element_bit_count);
  EXPECT_EQ(role_layout->coordinate_flags, coordinate_flags);
}

void ExpectFragmentCoordinate(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role, uint16_t lane, uint16_t register_index,
    uint16_t element_index,
    loom_amdgpu_matrix_fragment_coordinate_flags_t coordinate_flags,
    uint16_t row, uint16_t column, uint16_t reduction) {
  loom_amdgpu_matrix_fragment_coordinate_t coordinate = {};
  ASSERT_TRUE(loom_amdgpu_matrix_fragment_coordinate(
      layout, role, lane, register_index, element_index, &coordinate));
  EXPECT_EQ(coordinate.coordinate_flags, coordinate_flags);
  EXPECT_EQ(coordinate.row, row);
  EXPECT_EQ(coordinate.column, column);
  EXPECT_EQ(coordinate.reduction, reduction);
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
  EXPECT_EQ(descriptor->lhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16);
  EXPECT_EQ(descriptor->lhs_payload.register_count, 1);
  EXPECT_EQ(descriptor->lhs_payload.element_count, 2);
  EXPECT_EQ(descriptor->result_payload.register_count, 4);
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

TEST(MatrixContractTest, Gfx950ScaledMfmaDescriptor) {
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      FindDescriptor("mfma.scale.f32.32x32x64.f8f6f4");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_MFMA);
  EXPECT_EQ(descriptor->tile_shape.result_row_count, 32);
  EXPECT_EQ(descriptor->tile_shape.result_column_count, 32);
  EXPECT_EQ(descriptor->tile_shape.reduction_count, 64);
  EXPECT_EQ(descriptor->lhs_payload.numeric_type,
            LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4);
  EXPECT_EQ(descriptor->lhs_payload.register_count, 8);
  EXPECT_EQ(descriptor->lhs_payload.element_count, 32);
  EXPECT_EQ(descriptor->rhs_payload.register_count, 8);
  EXPECT_EQ(descriptor->rhs_payload.element_count, 32);
  EXPECT_EQ(descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED);
  EXPECT_EQ(descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS);
  EXPECT_EQ(
      descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK,
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK);
  EXPECT_EQ(descriptor->scale_kind, LOOM_AMDGPU_MATRIX_SCALE_32);
  EXPECT_EQ(descriptor->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_MFMA_SCALE_F32_32X32X64_F8F6F4);

  const loom_amdgpu_matrix_contract_descriptor_t* compact_descriptor =
      FindDescriptor("mfma.scale.f32.16x16x128.f8f6f4");
  ASSERT_NE(compact_descriptor, nullptr);
  EXPECT_EQ(compact_descriptor->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_MFMA_SCALE_F32_16X16X128_F8F6F4);
}

TEST(MatrixContractTest, Gfx1250WmmaScale16Descriptor) {
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      FindDescriptor("wmma.scale16.f32.16x16x128.f8f6f4");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_WMMA);
  EXPECT_EQ(descriptor->tile_shape.reduction_count, 128);
  EXPECT_EQ(descriptor->result_payload.register_count, 8);
  EXPECT_EQ(descriptor->scale_kind, LOOM_AMDGPU_MATRIX_SCALE_16);
  EXPECT_EQ(descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS);
  EXPECT_EQ(descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS);
  EXPECT_EQ(descriptor->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE, 0u);
  EXPECT_EQ(
      descriptor->low_descriptor_ref,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_SCALE16_F32_16X16X128_F8F6F4_F8_F8);
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
  EXPECT_EQ(f16_descriptor->wave_size_bits, LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY);
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
  EXPECT_EQ(i32_iu8->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT);
  EXPECT_EQ(i32_iu8->flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP);

  const loom_amdgpu_matrix_contract_descriptor_t* i32_iu4 =
      FindDescriptor("wmma.i32.16x16x16.iu4");
  ASSERT_NE(i32_iu4, nullptr);
  EXPECT_EQ(i32_iu4->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU4);

  const loom_amdgpu_matrix_contract_descriptor_t* f32_f16_gfx1250 =
      FindDescriptor("wmma.f32.16x16x32.f16");
  ASSERT_NE(f32_f16_gfx1250, nullptr);
  EXPECT_EQ(f32_f16_gfx1250->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X32_F16);

  const loom_amdgpu_matrix_contract_descriptor_t* scaled =
      FindDescriptor("wmma.scale.f32.16x16x128.f8f6f4");
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

TEST(MatrixContractTest, DenseF32FragmentDescriptorsCarryLayouts) {
  struct Case {
    const char* descriptor_name;
    loom_amdgpu_descriptor_ref_t low_descriptor_ref;
    loom_amdgpu_matrix_fragment_layout_kind_t layout_kind;
    const char* layout_name;
    loom_amdgpu_matrix_numeric_type_t input_numeric_type;
    uint16_t input_register_count;
    uint16_t input_element_count;
    uint16_t accumulator_register_count;
    uint16_t accumulator_element_count;
    loom_amdgpu_matrix_wave_size_bits_t wave_size_bits;
    uint16_t layout_wave_size;
  };
  const Case cases[] = {
      {
          "wmma.f32.16x16x16.f16",
          LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X16_F16,
          LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16_F16,
          "rdna3.wmmar3.f32.16x16x16.f16",
          LOOM_AMDGPU_MATRIX_NUMERIC_F16,
          8,
          16,
          8,
          8,
          LOOM_AMDGPU_MATRIX_WAVE_SIZE_32,
          32,
      },
      {
          "wmma.f32.16x16x16.bf16",
          LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X16_BF16,
          LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16_BF16,
          "rdna3.wmmar3.f32.16x16x16.bf16",
          LOOM_AMDGPU_MATRIX_NUMERIC_BF16,
          8,
          16,
          8,
          8,
          LOOM_AMDGPU_MATRIX_WAVE_SIZE_32,
          32,
      },
      {
          "mfma.f32.16x16x16.f16",
          LOOM_AMDGPU_DESCRIPTOR_REF_V_MFMA_F32_16X16X16_F16,
          LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X16_F16,
          "cdna.mfma.f32.16x16x16.f16",
          LOOM_AMDGPU_MATRIX_NUMERIC_F16,
          2,
          4,
          4,
          4,
          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
          64,
      },
      {
          "mfma.f32.16x16x16.bf16.1k",
          LOOM_AMDGPU_DESCRIPTOR_REF_V_MFMA_F32_16X16X16_BF16,
          LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X16_BF16,
          "cdna.mfma.f32.16x16x16.bf16",
          LOOM_AMDGPU_MATRIX_NUMERIC_BF16,
          2,
          4,
          4,
          4,
          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
          64,
      },
  };
  for (const Case& test_case : cases) {
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
        FindDescriptor(test_case.descriptor_name);
    ASSERT_NE(descriptor, nullptr) << test_case.descriptor_name;
    EXPECT_EQ(descriptor->low_descriptor_ref, test_case.low_descriptor_ref)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->wave_size_bits, test_case.wave_size_bits)
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
              LOOM_AMDGPU_MATRIX_NUMERIC_F32)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->accumulator_payload.register_count,
              test_case.accumulator_register_count)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->accumulator_payload.element_count,
              test_case.accumulator_element_count)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->result_payload.numeric_type,
              LOOM_AMDGPU_MATRIX_NUMERIC_F32)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->result_payload.register_count,
              test_case.accumulator_register_count)
        << test_case.descriptor_name;
    EXPECT_EQ(descriptor->result_payload.element_count,
              test_case.accumulator_element_count)
        << test_case.descriptor_name;

    const loom_amdgpu_matrix_fragment_layout_t* layout =
        loom_amdgpu_matrix_contract_descriptor_fragment_layout(descriptor);
    ASSERT_NE(layout, nullptr) << test_case.descriptor_name;
    EXPECT_EQ(layout->kind, test_case.layout_kind) << test_case.descriptor_name;
    EXPECT_EQ(ToString(layout->name), test_case.layout_name)
        << test_case.descriptor_name;
    EXPECT_EQ(layout->wave_size, test_case.layout_wave_size)
        << test_case.descriptor_name;
  }
}

TEST(MatrixContractTest, Rdna3Wmmar3F32F16LayoutMapsFragments) {
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      FindDescriptor("wmma.f32.16x16x16.f16");
  ASSERT_NE(descriptor, nullptr);
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_contract_descriptor_fragment_layout(descriptor);
  ASSERT_NE(layout, nullptr);
  EXPECT_EQ(layout->kind,
            LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16_F16);
  EXPECT_EQ(ToString(layout->name), "rdna3.wmmar3.f32.16x16x16.f16");
  EXPECT_EQ(descriptor->wave_size_bits, LOOM_AMDGPU_MATRIX_WAVE_SIZE_32);
  EXPECT_EQ(layout->wave_size, 32);
  EXPECT_EQ(layout->tile_shape.result_row_count, 16);
  EXPECT_EQ(layout->tile_shape.result_column_count, 16);
  EXPECT_EQ(layout->tile_shape.reduction_count, 16);
  EXPECT_EQ(descriptor->lhs_payload.register_count, 8);
  EXPECT_EQ(descriptor->lhs_payload.element_count, 16);
  EXPECT_EQ(descriptor->rhs_payload.register_count, 8);
  EXPECT_EQ(descriptor->rhs_payload.element_count, 16);
  EXPECT_EQ(descriptor->accumulator_payload.register_count, 8);
  EXPECT_EQ(descriptor->accumulator_payload.element_count, 8);
  EXPECT_EQ(descriptor->result_payload.register_count, 8);
  EXPECT_EQ(descriptor->result_payload.element_count, 8);

  constexpr loom_amdgpu_matrix_fragment_coordinate_flags_t kLhsCoordinates =
      LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_ROW |
      LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_REDUCTION;
  constexpr loom_amdgpu_matrix_fragment_coordinate_flags_t kRhsCoordinates =
      LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_COLUMN |
      LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_REDUCTION;
  constexpr loom_amdgpu_matrix_fragment_coordinate_flags_t
      kAccumulatorCoordinates = LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_ROW |
                                LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_COLUMN;
  ExpectFragmentRoleLayout(
      layout, LOOM_CONTRACT_OPERAND_ROLE_LHS,
      LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION, 8, 2, 16,
      kLhsCoordinates);
  ExpectFragmentRoleLayout(
      layout, LOOM_CONTRACT_OPERAND_ROLE_RHS,
      LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_PACKED_REDUCTION, 8, 2,
      16, kRhsCoordinates);
  ExpectFragmentRoleLayout(
      layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR,
      LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN, 8, 1, 32,
      kAccumulatorCoordinates);
  ExpectFragmentRoleLayout(
      layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT,
      LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN, 8, 1, 32,
      kAccumulatorCoordinates);

  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 0, 0, 0,
                           kLhsCoordinates, 0, 0, 0);
  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 15, 0, 1,
                           kLhsCoordinates, 15, 0, 1);
  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 16, 7, 1,
                           kLhsCoordinates, 0, 0, 15);

  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_RHS, 0, 0, 0,
                           kRhsCoordinates, 0, 0, 0);
  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_RHS, 15, 0, 1,
                           kRhsCoordinates, 0, 15, 1);
  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_RHS, 31, 7, 1,
                           kRhsCoordinates, 0, 15, 15);

  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 0, 0,
                           0, kAccumulatorCoordinates, 0, 0, 0);
  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 15,
                           0, 0, kAccumulatorCoordinates, 0, 15, 0);
  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 16,
                           0, 0, kAccumulatorCoordinates, 1, 0, 0);
  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 31, 7, 0,
                           kAccumulatorCoordinates, 15, 15, 0);

  loom_amdgpu_matrix_fragment_coordinate_t coordinate = {};
  EXPECT_FALSE(loom_amdgpu_matrix_fragment_coordinate(
      layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 32, 0, 0, &coordinate));
  EXPECT_FALSE(loom_amdgpu_matrix_fragment_coordinate(
      layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 0, 8, 0, &coordinate));
}

TEST(MatrixContractTest, MatcherSelectedWmmar3DescriptorCarriesLayoutFacts) {
  loom_amdgpu_matrix_contract_match_request_t request = MatchRequest(
      LOOM_AMDGPU_MATRIX_FAMILY_WMMA, 16, 16, 16,
      LOOM_AMDGPU_MATRIX_NUMERIC_F16, LOOM_AMDGPU_MATRIX_NUMERIC_F16,
      LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
      LOOM_AMDGPU_MATRIX_SCALE_NONE, LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11, 32,
      0, 0);
  loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      loom_amdgpu_matrix_contract_select(&request, &diagnostic);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X16_F16);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE);
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_contract_descriptor_fragment_layout(descriptor);
  ASSERT_NE(layout, nullptr);
  EXPECT_EQ(layout->kind,
            LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16_F16);
}

TEST(MatrixContractTest, CdnaMfmaF32Bf16LayoutMapsFragments) {
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      FindDescriptor("mfma.f32.16x16x16.bf16.1k");
  ASSERT_NE(descriptor, nullptr);
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_contract_descriptor_fragment_layout(descriptor);
  ASSERT_NE(layout, nullptr);
  EXPECT_EQ(layout->kind,
            LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X16_BF16);
  EXPECT_EQ(ToString(layout->name), "cdna.mfma.f32.16x16x16.bf16");
  EXPECT_EQ(descriptor->wave_size_bits, LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY);
  EXPECT_EQ(layout->wave_size, 64);
  EXPECT_EQ(layout->tile_shape.result_row_count, 16);
  EXPECT_EQ(layout->tile_shape.result_column_count, 16);
  EXPECT_EQ(layout->tile_shape.reduction_count, 16);
  EXPECT_EQ(descriptor->lhs_payload.register_count, 2);
  EXPECT_EQ(descriptor->lhs_payload.element_count, 4);
  EXPECT_EQ(descriptor->rhs_payload.register_count, 2);
  EXPECT_EQ(descriptor->rhs_payload.element_count, 4);
  EXPECT_EQ(descriptor->accumulator_payload.register_count, 4);
  EXPECT_EQ(descriptor->accumulator_payload.element_count, 4);
  EXPECT_EQ(descriptor->result_payload.register_count, 4);
  EXPECT_EQ(descriptor->result_payload.element_count, 4);

  constexpr loom_amdgpu_matrix_fragment_coordinate_flags_t kLhsCoordinates =
      LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_ROW |
      LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_REDUCTION;
  constexpr loom_amdgpu_matrix_fragment_coordinate_flags_t kRhsCoordinates =
      LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_COLUMN |
      LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_REDUCTION;
  constexpr loom_amdgpu_matrix_fragment_coordinate_flags_t
      kAccumulatorCoordinates = LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_ROW |
                                LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_COLUMN;
  ExpectFragmentRoleLayout(
      layout, LOOM_CONTRACT_OPERAND_ROLE_LHS,
      LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_LANE_GROUP_PACKED_REDUCTION,
      2, 2, 16, kLhsCoordinates);
  ExpectFragmentRoleLayout(
      layout, LOOM_CONTRACT_OPERAND_ROLE_RHS,
      LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_LANE_GROUP_PACKED_REDUCTION,
      2, 2, 16, kRhsCoordinates);
  ExpectFragmentRoleLayout(
      layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR,
      LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN, 4, 1, 32,
      kAccumulatorCoordinates);
  ExpectFragmentRoleLayout(
      layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT,
      LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN, 4, 1, 32,
      kAccumulatorCoordinates);

  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 0, 0, 0,
                           kLhsCoordinates, 0, 0, 0);
  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 15, 1, 1,
                           kLhsCoordinates, 15, 0, 3);
  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 16, 0, 0,
                           kLhsCoordinates, 0, 0, 4);
  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 63, 1, 1,
                           kLhsCoordinates, 15, 0, 15);

  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_RHS, 0, 0, 0,
                           kRhsCoordinates, 0, 0, 0);
  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_RHS, 15, 1, 1,
                           kRhsCoordinates, 0, 15, 3);
  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_RHS, 16, 0, 0,
                           kRhsCoordinates, 0, 0, 4);
  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_RHS, 63, 1, 1,
                           kRhsCoordinates, 0, 15, 15);

  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 0, 0,
                           0, kAccumulatorCoordinates, 0, 0, 0);
  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 15,
                           3, 0, kAccumulatorCoordinates, 3, 15, 0);
  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, 16,
                           0, 0, kAccumulatorCoordinates, 4, 0, 0);
  ExpectFragmentCoordinate(layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 63, 3, 0,
                           kAccumulatorCoordinates, 15, 15, 0);

  loom_amdgpu_matrix_fragment_coordinate_t coordinate = {};
  EXPECT_FALSE(loom_amdgpu_matrix_fragment_coordinate(
      layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, 64, 0, 0, &coordinate));
  EXPECT_FALSE(loom_amdgpu_matrix_fragment_coordinate(
      layout, LOOM_CONTRACT_OPERAND_ROLE_LHS, 0, 2, 0, &coordinate));
}

TEST(MatrixContractTest, MatcherSelectedCdnaMfmaDescriptorCarriesLayoutFacts) {
  loom_amdgpu_matrix_contract_match_request_t request = MatchRequest(
      LOOM_AMDGPU_MATRIX_FAMILY_MFMA, 16, 16, 16,
      LOOM_AMDGPU_MATRIX_NUMERIC_BF16, LOOM_AMDGPU_MATRIX_NUMERIC_BF16,
      LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
      LOOM_AMDGPU_MATRIX_SCALE_NONE,
      LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_BF16_1K, 64, 0, 0);
  loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      loom_amdgpu_matrix_contract_select(&request, &diagnostic);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->low_descriptor_ref,
            LOOM_AMDGPU_DESCRIPTOR_REF_V_MFMA_F32_16X16X16_BF16);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE);
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_contract_descriptor_fragment_layout(descriptor);
  ASSERT_NE(layout, nullptr);
  EXPECT_EQ(layout->kind,
            LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X16_BF16);
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

TEST(MatrixContractTest, MatcherSelectsRdnaIntegerWmmaLowDescriptors) {
  const loom_amdgpu_matrix_contract_flags_t integer_wmma_flags =
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP;
  const loom_amdgpu_matrix_feature_bits_t gfx12_features =
      LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11 |
      LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX12 |
      LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX12;
  struct Case {
    // Matrix input numeric type requested by the source contract.
    loom_amdgpu_matrix_numeric_type_t numeric_type;
    // Expected target-low descriptor ref for native lowering.
    loom_amdgpu_descriptor_ref_t expected_low_descriptor_ref;
  };
  const Case cases[] = {
      {LOOM_AMDGPU_MATRIX_NUMERIC_IU8,
       LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU8},
      {LOOM_AMDGPU_MATRIX_NUMERIC_IU4,
       LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU4},
  };
  const loom_amdgpu_matrix_feature_bits_t feature_cases[] = {
      LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11,
      gfx12_features,
  };
  for (loom_amdgpu_matrix_feature_bits_t feature_bits : feature_cases) {
    for (const Case& test_case : cases) {
      loom_amdgpu_matrix_contract_match_request_t request = MatchRequest(
          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, 16, 16, 16, test_case.numeric_type,
          test_case.numeric_type, LOOM_AMDGPU_MATRIX_NUMERIC_I32,
          LOOM_AMDGPU_MATRIX_NUMERIC_I32, LOOM_AMDGPU_MATRIX_SCALE_NONE,
          feature_bits, 32, integer_wmma_flags, 0);
      loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
      const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
          loom_amdgpu_matrix_contract_select(&request, &diagnostic);
      ASSERT_NE(descriptor, nullptr);
      EXPECT_EQ(descriptor->family, LOOM_AMDGPU_MATRIX_FAMILY_WMMA);
      EXPECT_EQ(descriptor->lhs_payload.numeric_type, test_case.numeric_type);
      EXPECT_EQ(descriptor->rhs_payload.numeric_type, test_case.numeric_type);
      EXPECT_EQ(descriptor->low_descriptor_ref,
                test_case.expected_low_descriptor_ref);
      EXPECT_EQ(diagnostic.rejection_bits,
                LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE);
    }
  }
}

TEST(MatrixContractTest, MatcherRejectsMissingFeatureAfterSemanticMatch) {
  const loom_amdgpu_matrix_contract_flags_t scale_flags =
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS;
  loom_amdgpu_matrix_contract_match_request_t request = MatchRequest(
      LOOM_AMDGPU_MATRIX_FAMILY_MFMA, 32, 32, 64,
      LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4, LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4,
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
      LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4, LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4,
      LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
      LOOM_AMDGPU_MATRIX_SCALE_32,
      LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950_SCALE_F8F6F4, 64, available_flags,
      required_flags);
  loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(&request, &diagnostic), nullptr);
  EXPECT_EQ(diagnostic.rejection_bits &
                LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_MATRIX_FORMATS,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_MATRIX_FORMATS);
}

TEST(MatrixContractTest, MatcherRejectsScaleKindMismatch) {
  loom_amdgpu_matrix_contract_match_request_t request = MatchRequest(
      LOOM_AMDGPU_MATRIX_FAMILY_MFMA, 32, 32, 64,
      LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4, LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4,
      LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
      LOOM_AMDGPU_MATRIX_SCALE_NONE,
      LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950_SCALE_F8F6F4, 64, 0, 0);
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
  loom_amdgpu_matrix_feature_bits_t gfx1250_features = 0;
  IREE_ASSERT_OK(loom_amdgpu_matrix_feature_bits_from_processor(
      IREE_SV("gfx1250"), &gfx1250_features));

  const loom_amdgpu_matrix_contract_descriptor_t* fp8_mfma =
      FindDescriptor("mfma.f32.16x16x32.fp8.fp8");
  ASSERT_NE(fp8_mfma, nullptr);
  EXPECT_TRUE(
      loom_amdgpu_matrix_contract_is_available(fp8_mfma, gfx942_features, 64));

  const loom_amdgpu_matrix_contract_descriptor_t* double_mfma =
      FindDescriptor("mfma.f64.16x16x4.f64");
  ASSERT_NE(double_mfma, nullptr);
  EXPECT_FALSE(loom_amdgpu_matrix_contract_is_available(double_mfma,
                                                        gfx908_features, 64));
  EXPECT_TRUE(loom_amdgpu_matrix_contract_is_available(double_mfma,
                                                       gfx90a_features, 64));

  const loom_amdgpu_matrix_contract_descriptor_t* scaled_mfma =
      FindDescriptor("mfma.scale.f32.16x16x128.f8f6f4");
  ASSERT_NE(scaled_mfma, nullptr);
  EXPECT_FALSE(loom_amdgpu_matrix_contract_is_available(scaled_mfma,
                                                        gfx942_features, 64));
  EXPECT_TRUE(loom_amdgpu_matrix_contract_is_available(scaled_mfma,
                                                       gfx950_features, 64));

  const loom_amdgpu_matrix_contract_descriptor_t* scaled_wmma =
      FindDescriptor("wmma.scale.f32.16x16x128.f8f6f4");
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
      FindDescriptor("mfma.scale.f32.16x16x128.f8f6f4");
  ASSERT_NE(scaled_mfma, nullptr);
  EXPECT_FALSE(loom_amdgpu_matrix_contract_is_available(
      scaled_mfma, LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950, 64));

  const loom_amdgpu_matrix_contract_descriptor_t* unscaled_wmma =
      FindDescriptor("wmma.f32.16x16x32.f16");
  ASSERT_NE(unscaled_wmma, nullptr);
  EXPECT_TRUE(loom_amdgpu_matrix_contract_is_available(
      unscaled_wmma, LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250, 32));

  const loom_amdgpu_matrix_contract_descriptor_t* unscaled_f8_wmma =
      FindDescriptor("wmma.f32.16x16x128.f8f6f4");
  ASSERT_NE(unscaled_f8_wmma, nullptr);
  EXPECT_TRUE(loom_amdgpu_matrix_contract_is_available(
      unscaled_f8_wmma, LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250, 32));

  const loom_amdgpu_matrix_contract_descriptor_t* scaled_wmma =
      FindDescriptor("wmma.scale.f32.16x16x128.f8f6f4");
  ASSERT_NE(scaled_wmma, nullptr);
  EXPECT_FALSE(loom_amdgpu_matrix_contract_is_available(
      scaled_wmma, LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250, 32));
}

TEST(MatrixContractTest, ProcessorFeatureBitsRejectUnknownProcessor) {
  loom_amdgpu_matrix_feature_bits_t feature_bits = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        loom_amdgpu_matrix_feature_bits_from_processor(
                            IREE_SV("gfx9999"), &feature_bits));
}

TEST(MatrixContractTest, ProcessorFeatureBitsUseTargetInfoAliases) {
  loom_amdgpu_matrix_feature_bits_t gfx1170_features = 0;
  IREE_ASSERT_OK(loom_amdgpu_matrix_feature_bits_from_processor(
      IREE_SV("gfx1170"), &gfx1170_features));
  EXPECT_EQ(gfx1170_features, LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11 |
                                  LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX12 |
                                  LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX12);

  loom_amdgpu_matrix_feature_bits_t gfx1251_features = 0;
  IREE_ASSERT_OK(loom_amdgpu_matrix_feature_bits_from_processor(
      IREE_SV("gfx1251"), &gfx1251_features));
  EXPECT_NE(gfx1251_features & LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250,
            UINT64_C(0));
  EXPECT_NE(
      gfx1251_features & LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250_SCALE_F8F6F4,
      UINT64_C(0));
}

TEST(MatrixContractTest, NamesAreStable) {
  EXPECT_EQ(ToString(loom_amdgpu_matrix_family_name(
                LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC)),
            "swmmac");
  EXPECT_EQ(ToString(loom_amdgpu_matrix_numeric_type_name(
                LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4)),
            "f8f6f4");
  EXPECT_EQ(
      ToString(loom_amdgpu_matrix_scale_kind_name(LOOM_AMDGPU_MATRIX_SCALE_16)),
      "scale16");
}

}  // namespace
