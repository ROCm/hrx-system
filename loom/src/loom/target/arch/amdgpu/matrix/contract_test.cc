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

iree_host_size_t FindDescriptorOrdinal(const char* name) {
  iree_string_view_t expected_name = iree_make_cstring_view(name);
  const iree_host_size_t count = loom_amdgpu_matrix_contract_descriptor_count();
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
        loom_amdgpu_matrix_contract_descriptor_at(i);
    if (iree_string_view_equal(descriptor->name, expected_name)) {
      return i;
    }
  }
  return count;
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

TEST(MatrixContractTest, RealizationCatalogCoversEveryDescriptor) {
  const iree_host_size_t count = loom_amdgpu_matrix_contract_descriptor_count();
  iree_host_size_t exact_count = 0;
  iree_host_size_t operand_exchanged_count = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
        loom_amdgpu_matrix_contract_descriptor_at(i);
    ASSERT_NE(descriptor, nullptr);
    const loom_amdgpu_matrix_contract_realization_choices_t* choices =
        &descriptor->realization;
    if (descriptor->fragment_layout_kind ==
        LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_UNKNOWN) {
      EXPECT_EQ(choices->canonical_result_representation_id,
                LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_NONE);
      EXPECT_EQ(choices->operand_exchanged_contract_ordinal,
                LOOM_AMDGPU_MATRIX_CONTRACT_ORDINAL_NONE);
      continue;
    }
    ++exact_count;
    EXPECT_NE(loom_amdgpu_matrix_result_representation_at(
                  choices->canonical_result_representation_id),
              nullptr);
    if (choices->operand_exchanged_contract_ordinal ==
        LOOM_AMDGPU_MATRIX_CONTRACT_ORDINAL_NONE) {
      EXPECT_EQ(choices->operand_exchanged_result_representation_id,
                LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_NONE);
      continue;
    }
    ++operand_exchanged_count;
    EXPECT_LT(choices->operand_exchanged_contract_ordinal, count);
    EXPECT_NE(loom_amdgpu_matrix_result_representation_at(
                  choices->operand_exchanged_result_representation_id),
              nullptr);
  }
  EXPECT_EQ(exact_count, 217u);
  EXPECT_EQ(operand_exchanged_count, 150u);
  EXPECT_EQ(loom_amdgpu_matrix_result_representation_at(
                LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_NONE),
            nullptr);
}

TEST(MatrixContractTest, Gfx11F16F32HasTransposedResultRealization) {
  const iree_host_size_t contract_ordinal =
      FindDescriptorOrdinal("wmma.f32.16x16x16.f16");
  ASSERT_LT(contract_ordinal, loom_amdgpu_matrix_contract_descriptor_count());
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      loom_amdgpu_matrix_contract_descriptor_at(contract_ordinal);
  ASSERT_NE(descriptor, nullptr);
  const loom_amdgpu_matrix_contract_realization_choices_t* choices =
      &descriptor->realization;
  EXPECT_EQ(choices->operand_exchanged_contract_ordinal, contract_ordinal);

  const loom_amdgpu_matrix_result_representation_t* representation =
      loom_amdgpu_matrix_result_representation_at(
          choices->operand_exchanged_result_representation_id);
  ASSERT_NE(representation, nullptr);
  EXPECT_GT(representation->fragment_layout_kind,
            LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_UNKNOWN);
  EXPECT_LT(representation->fragment_layout_kind,
            LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_COUNT);
  EXPECT_EQ(representation->flags,
            LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_FLAG_TRANSPOSE_MN);
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
}

TEST(MatrixContractTest, MatcherDistinguishesBlockedMfmaShape) {
  loom_amdgpu_matrix_contract_match_request_t request = MatchRequest(
      LOOM_AMDGPU_MATRIX_FAMILY_MFMA, 4, 4, 2, LOOM_AMDGPU_MATRIX_NUMERIC_BF16,
      LOOM_AMDGPU_MATRIX_NUMERIC_BF16, LOOM_AMDGPU_MATRIX_NUMERIC_F32,
      LOOM_AMDGPU_MATRIX_NUMERIC_F32, LOOM_AMDGPU_MATRIX_SCALE_NONE,
      LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908_GFX90A, 64, 0, 0);
  loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(&request, nullptr, &diagnostic),
            nullptr);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_TILE_SHAPE);

  request.tile_shape.block_count = 16;
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      loom_amdgpu_matrix_contract_select(&request, nullptr, &diagnostic);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor->name), "mfma.f32.4x4x2.bf16");
  EXPECT_EQ(descriptor->tile_shape.block_count, 16);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE);
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
      loom_amdgpu_matrix_contract_select(&request, nullptr, &diagnostic);
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
  uint16_t descriptor_ordinal = LOOM_AMDGPU_MATRIX_CONTRACT_ORDINAL_NONE;
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      loom_amdgpu_matrix_contract_select(&request, &descriptor_ordinal,
                                         &diagnostic);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_NE(descriptor_ordinal, LOOM_AMDGPU_MATRIX_CONTRACT_ORDINAL_NONE);
  EXPECT_EQ(loom_amdgpu_matrix_contract_descriptor_at(descriptor_ordinal),
            descriptor);
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
      loom_amdgpu_matrix_contract_select(&request, nullptr, &diagnostic);
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
        loom_amdgpu_matrix_contract_select(&request, nullptr, &diagnostic);
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
      loom_amdgpu_matrix_contract_select(&request, nullptr, &diagnostic);
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
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(&shape_request, nullptr,
                                               &shape_diagnostic),
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
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(&payload_request, nullptr,
                                               &payload_diagnostic),
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
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(&request, nullptr, &diagnostic),
            nullptr);
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
      loom_amdgpu_matrix_contract_select(&request, nullptr, &diagnostic);
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
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(&request, nullptr, &diagnostic),
            nullptr);
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
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(&request, nullptr, &diagnostic),
            nullptr);
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
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(&sparse_request, nullptr,
                                               &sparse_diagnostic),
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
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(&reuse_request, nullptr,
                                               &reuse_diagnostic),
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
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(&request, nullptr, &diagnostic),
            nullptr);
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
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(&request, nullptr, &diagnostic),
            nullptr);
  EXPECT_EQ(diagnostic.feature_candidate_count, 1u);
  EXPECT_EQ(diagnostic.wave_candidate_count, 0u);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_WAVE_SIZE);
}

TEST(MatrixContractTest, MatcherRejectsInvalidRequest) {
  loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {};
  uint16_t descriptor_ordinal = 0;
  EXPECT_EQ(loom_amdgpu_matrix_contract_select(nullptr, &descriptor_ordinal,
                                               &diagnostic),
            nullptr);
  EXPECT_EQ(descriptor_ordinal, LOOM_AMDGPU_MATRIX_CONTRACT_ORDINAL_NONE);
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
