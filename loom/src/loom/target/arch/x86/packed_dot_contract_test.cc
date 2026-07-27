// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/packed_dot_contract.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

const loom_x86_packed_dot_descriptor_t* FindDescriptor(const char* name) {
  iree_string_view_t expected_name = iree_make_cstring_view(name);
  const iree_host_size_t count = loom_x86_packed_dot_descriptor_count();
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_x86_packed_dot_descriptor_t* descriptor =
        loom_x86_packed_dot_descriptor_at(i);
    if (iree_string_view_equal(descriptor->name, expected_name)) {
      return descriptor;
    }
  }
  return nullptr;
}

loom_x86_packed_dot_feature_bits_t FeatureBits(const char* name) {
  loom_x86_packed_dot_feature_bits_t feature_bits = 0;
  IREE_EXPECT_OK(loom_x86_packed_dot_feature_bits_for_name(
      iree_make_cstring_view(name), &feature_bits));
  return feature_bits;
}

loom_x86_packed_dot_match_request_t MatchRequest(
    loom_x86_packed_dot_family_t family, uint16_t vector_bit_width,
    uint16_t input_lane_count, uint16_t result_lane_count,
    uint16_t reduction_group_size,
    loom_x86_packed_dot_numeric_type_t lhs_numeric_type,
    loom_x86_packed_dot_numeric_type_t rhs_numeric_type,
    loom_x86_packed_dot_numeric_type_t accumulator_numeric_type,
    loom_x86_packed_dot_numeric_type_t result_numeric_type,
    loom_x86_packed_dot_feature_bits_t feature_bits,
    loom_x86_packed_dot_contract_flags_t required_flags) {
  loom_x86_packed_dot_match_request_t request = {};
  request.family = family;
  request.shape.vector_bit_width = vector_bit_width;
  request.shape.input_lane_count = input_lane_count;
  request.shape.result_lane_count = result_lane_count;
  request.shape.reduction_group_size = reduction_group_size;
  request.lhs_numeric_type = lhs_numeric_type;
  request.rhs_numeric_type = rhs_numeric_type;
  request.accumulator_numeric_type = accumulator_numeric_type;
  request.result_numeric_type = result_numeric_type;
  request.feature_bits = feature_bits;
  request.required_flags = required_flags;
  return request;
}

TEST(PackedDotContractTest, DescriptorNamesAreUnique) {
  const iree_host_size_t count = loom_x86_packed_dot_descriptor_count();
  ASSERT_GT(count, 0u);
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_x86_packed_dot_descriptor_t* lhs =
        loom_x86_packed_dot_descriptor_at(i);
    ASSERT_NE(lhs, nullptr);
    EXPECT_FALSE(iree_string_view_is_empty(lhs->name));
    EXPECT_FALSE(iree_string_view_is_empty(lhs->llvm_intrinsic_name));
    EXPECT_FALSE(iree_string_view_is_empty(lhs->instruction_mnemonic));
    for (iree_host_size_t j = i + 1; j < count; ++j) {
      const loom_x86_packed_dot_descriptor_t* rhs =
          loom_x86_packed_dot_descriptor_at(j);
      ASSERT_NE(rhs, nullptr);
      EXPECT_FALSE(iree_string_view_equal(lhs->name, rhs->name))
          << ToString(lhs->name);
    }
  }
  EXPECT_EQ(loom_x86_packed_dot_descriptor_at(count), nullptr);
}

TEST(PackedDotContractTest, NamesExposeStableDisplayStrings) {
  EXPECT_EQ(ToString(loom_x86_packed_dot_family_name(
                LOOM_X86_PACKED_DOT_FAMILY_AVX_VNNI_INT8)),
            "x86-avx-vnni-int8");
  EXPECT_EQ(ToString(loom_x86_packed_dot_numeric_type_name(
                LOOM_X86_PACKED_DOT_NUMERIC_U16)),
            "u16");
  EXPECT_EQ(ToString(loom_x86_packed_dot_family_name(
                LOOM_X86_PACKED_DOT_FAMILY_UNKNOWN)),
            "unknown");
}

TEST(PackedDotContractTest, Avx512VnniByteDescriptor) {
  const loom_x86_packed_dot_descriptor_t* descriptor =
      FindDescriptor("x86.avx512_vnni.vpdpbusd.zmm");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_X86_PACKED_DOT_FAMILY_AVX512_VNNI);
  EXPECT_EQ(ToString(descriptor->llvm_intrinsic_name),
            "llvm.x86.avx512.vpdpbusd.512");
  EXPECT_EQ(descriptor->llvm_source_abi,
            LOOM_X86_PACKED_DOT_LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR);
  EXPECT_EQ(ToString(descriptor->instruction_mnemonic), "vpdpbusd");
  EXPECT_EQ(descriptor->required_feature_bits, LOOM_X86_FEATURE_AVX512_VNNI);
  EXPECT_EQ(descriptor->shape.vector_bit_width, 512);
  EXPECT_EQ(descriptor->shape.input_lane_count, 64);
  EXPECT_EQ(descriptor->shape.result_lane_count, 16);
  EXPECT_EQ(descriptor->shape.reduction_group_size, 4);
  EXPECT_EQ(descriptor->lhs_numeric_type, LOOM_X86_PACKED_DOT_NUMERIC_U8);
  EXPECT_EQ(descriptor->rhs_numeric_type, LOOM_X86_PACKED_DOT_NUMERIC_I8);
  EXPECT_EQ(descriptor->result_numeric_type, LOOM_X86_PACKED_DOT_NUMERIC_I32);
}

TEST(PackedDotContractTest, AvxVnniInt8SignedSignedDescriptor) {
  const loom_x86_packed_dot_descriptor_t* descriptor =
      FindDescriptor("x86.avx_vnni_int8.vpdpbssd.ymm");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_X86_PACKED_DOT_FAMILY_AVX_VNNI_INT8);
  EXPECT_EQ(ToString(descriptor->llvm_intrinsic_name),
            "llvm.x86.avx2.vpdpbssd.256");
  EXPECT_EQ(descriptor->llvm_source_abi,
            LOOM_X86_PACKED_DOT_LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR);
  EXPECT_EQ(descriptor->shape.vector_bit_width, 256);
  EXPECT_EQ(descriptor->lhs_numeric_type, LOOM_X86_PACKED_DOT_NUMERIC_I8);
  EXPECT_EQ(descriptor->rhs_numeric_type, LOOM_X86_PACKED_DOT_NUMERIC_I8);
}

TEST(PackedDotContractTest, Avx10Fp16Descriptor) {
  const loom_x86_packed_dot_descriptor_t* descriptor =
      FindDescriptor("x86.avx10_2.vdpphps.zmm");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_X86_PACKED_DOT_FAMILY_AVX10_2);
  EXPECT_EQ(ToString(descriptor->llvm_intrinsic_name),
            "llvm.x86.avx10.vdpphps.512");
  EXPECT_EQ(descriptor->llvm_source_abi,
            LOOM_X86_PACKED_DOT_LLVM_SOURCE_ABI_PAYLOAD);
  EXPECT_EQ(descriptor->shape.reduction_group_size, 2);
  EXPECT_EQ(descriptor->lhs_numeric_type, LOOM_X86_PACKED_DOT_NUMERIC_F16);
  EXPECT_EQ(descriptor->accumulator_numeric_type,
            LOOM_X86_PACKED_DOT_NUMERIC_F32);
}

TEST(PackedDotContractTest, Avx512Bf16Descriptor) {
  const loom_x86_packed_dot_descriptor_t* descriptor =
      FindDescriptor("x86.avx512_bf16.vdpbf16ps.ymm");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_X86_PACKED_DOT_FAMILY_AVX512_BF16);
  EXPECT_EQ(descriptor->required_feature_bits,
            LOOM_X86_FEATURE_AVX512_BF16 | LOOM_X86_FEATURE_AVX512_VL);
  EXPECT_EQ(descriptor->llvm_source_abi,
            LOOM_X86_PACKED_DOT_LLVM_SOURCE_ABI_PAYLOAD);
  EXPECT_EQ(ToString(descriptor->instruction_mnemonic), "vdpbf16ps");
  EXPECT_EQ(descriptor->lhs_numeric_type, LOOM_X86_PACKED_DOT_NUMERIC_BF16);
  EXPECT_EQ(descriptor->result_numeric_type, LOOM_X86_PACKED_DOT_NUMERIC_F32);
}

TEST(PackedDotContractTest, FeatureNamesMapToFeatureBits) {
  EXPECT_EQ(FeatureBits("x86-avx-vnni"), LOOM_X86_FEATURE_AVX_VNNI);
  EXPECT_EQ(FeatureBits("x86-avx-vnni-int8"), LOOM_X86_FEATURE_AVX_VNNI_INT8);
  EXPECT_EQ(FeatureBits("x86-avx512-vnni-vl"),
            LOOM_X86_FEATURE_AVX512_VNNI | LOOM_X86_FEATURE_AVX512_VL);
  EXPECT_EQ(FeatureBits("x86-avx512-bf16-vl"),
            LOOM_X86_FEATURE_AVX512_BF16 | LOOM_X86_FEATURE_AVX512_VL);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_x86_packed_dot_feature_bits_for_name(
                            IREE_SV("x86-imaginary"), nullptr));
  loom_x86_packed_dot_feature_bits_t feature_bits = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_x86_packed_dot_feature_bits_for_name(
                            IREE_SV("x86-imaginary"), &feature_bits));
}

TEST(PackedDotContractTest, SelectsAvxVnniByteDot) {
  const loom_x86_packed_dot_match_request_t request = MatchRequest(
      LOOM_X86_PACKED_DOT_FAMILY_UNKNOWN, 256, 32, 8, 4,
      LOOM_X86_PACKED_DOT_NUMERIC_U8, LOOM_X86_PACKED_DOT_NUMERIC_I8,
      LOOM_X86_PACKED_DOT_NUMERIC_I32, LOOM_X86_PACKED_DOT_NUMERIC_I32,
      FeatureBits("x86-avx-vnni"), 0);

  loom_x86_packed_dot_match_diagnostic_t diagnostic = {};
  const loom_x86_packed_dot_descriptor_t* descriptor =
      loom_x86_packed_dot_select(&request, &diagnostic);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_X86_PACKED_DOT_FAMILY_AVX_VNNI);
  EXPECT_EQ(descriptor->shape.vector_bit_width, 256);
  EXPECT_EQ(descriptor->shape.input_lane_count, 32);
  EXPECT_EQ(descriptor->shape.result_lane_count, 8);
  EXPECT_EQ(descriptor->shape.reduction_group_size, 4);
  EXPECT_EQ(descriptor->lhs_numeric_type, LOOM_X86_PACKED_DOT_NUMERIC_U8);
  EXPECT_EQ(descriptor->rhs_numeric_type, LOOM_X86_PACKED_DOT_NUMERIC_I8);
  EXPECT_EQ(descriptor->accumulator_numeric_type,
            LOOM_X86_PACKED_DOT_NUMERIC_I32);
  EXPECT_EQ(descriptor->result_numeric_type, LOOM_X86_PACKED_DOT_NUMERIC_I32);
  EXPECT_EQ(descriptor->flags, 0u);
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_NONE);
  EXPECT_GT(diagnostic.payload_candidate_count, 0u);
  EXPECT_GT(diagnostic.feature_candidate_count, 0u);
}

TEST(PackedDotContractTest, SelectsSaturatingVariantWhenRequired) {
  const loom_x86_packed_dot_match_request_t request = MatchRequest(
      LOOM_X86_PACKED_DOT_FAMILY_AVX_VNNI, 256, 32, 8, 4,
      LOOM_X86_PACKED_DOT_NUMERIC_U8, LOOM_X86_PACKED_DOT_NUMERIC_I8,
      LOOM_X86_PACKED_DOT_NUMERIC_I32, LOOM_X86_PACKED_DOT_NUMERIC_I32,
      FeatureBits("x86-avx-vnni"),
      LOOM_X86_PACKED_DOT_CONTRACT_FLAG_SATURATING);

  const loom_x86_packed_dot_descriptor_t* descriptor =
      loom_x86_packed_dot_select(&request, nullptr);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_X86_PACKED_DOT_FAMILY_AVX_VNNI);
  EXPECT_EQ(descriptor->shape.vector_bit_width, 256);
  EXPECT_EQ(descriptor->shape.reduction_group_size, 4);
  EXPECT_EQ(descriptor->flags & LOOM_X86_PACKED_DOT_CONTRACT_FLAG_SATURATING,
            LOOM_X86_PACKED_DOT_CONTRACT_FLAG_SATURATING);
}

TEST(PackedDotContractTest, SignedSignedByteDotNeedsInt8Feature) {
  const loom_x86_packed_dot_match_request_t avx_vnni_request = MatchRequest(
      LOOM_X86_PACKED_DOT_FAMILY_UNKNOWN, 256, 32, 8, 4,
      LOOM_X86_PACKED_DOT_NUMERIC_I8, LOOM_X86_PACKED_DOT_NUMERIC_I8,
      LOOM_X86_PACKED_DOT_NUMERIC_I32, LOOM_X86_PACKED_DOT_NUMERIC_I32,
      FeatureBits("x86-avx-vnni"), 0);

  loom_x86_packed_dot_match_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_x86_packed_dot_select(&avx_vnni_request, &diagnostic),
            nullptr);
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_FEATURES);
  EXPECT_GT(diagnostic.payload_candidate_count, 0u);

  const loom_x86_packed_dot_match_request_t int8_request = MatchRequest(
      LOOM_X86_PACKED_DOT_FAMILY_UNKNOWN, 256, 32, 8, 4,
      LOOM_X86_PACKED_DOT_NUMERIC_I8, LOOM_X86_PACKED_DOT_NUMERIC_I8,
      LOOM_X86_PACKED_DOT_NUMERIC_I32, LOOM_X86_PACKED_DOT_NUMERIC_I32,
      FeatureBits("x86-avx-vnni-int8"), 0);
  const loom_x86_packed_dot_descriptor_t* descriptor =
      loom_x86_packed_dot_select(&int8_request, nullptr);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->family, LOOM_X86_PACKED_DOT_FAMILY_AVX_VNNI_INT8);
  EXPECT_EQ(descriptor->lhs_numeric_type, LOOM_X86_PACKED_DOT_NUMERIC_I8);
  EXPECT_EQ(descriptor->rhs_numeric_type, LOOM_X86_PACKED_DOT_NUMERIC_I8);
}

TEST(PackedDotContractTest, RejectsPackedI4FallbackShape) {
  const loom_x86_packed_dot_match_request_t request = MatchRequest(
      LOOM_X86_PACKED_DOT_FAMILY_UNKNOWN, 128, 4, 4, 8,
      LOOM_X86_PACKED_DOT_NUMERIC_I8, LOOM_X86_PACKED_DOT_NUMERIC_I8,
      LOOM_X86_PACKED_DOT_NUMERIC_I32, LOOM_X86_PACKED_DOT_NUMERIC_I32,
      FeatureBits("x86-avx-vnni-int8"), 0);

  loom_x86_packed_dot_match_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_x86_packed_dot_select(&request, &diagnostic), nullptr);
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_SHAPE);
  EXPECT_GT(diagnostic.family_candidate_count, 0u);
  EXPECT_EQ(diagnostic.shape_candidate_count, 0u);
}

TEST(PackedDotContractTest, RejectsNullRequest) {
  loom_x86_packed_dot_match_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_x86_packed_dot_select(nullptr, &diagnostic), nullptr);
  EXPECT_EQ(diagnostic.rejection_bits,
            LOOM_X86_PACKED_DOT_REJECTION_INVALID_REQUEST);
}

}  // namespace
