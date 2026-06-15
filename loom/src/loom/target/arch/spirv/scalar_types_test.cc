// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/scalar_types.h"

#include "iree/testing/gtest.h"

namespace {

void ExpectScalarType(loom_spirv_scalar_type_t type,
                      iree_string_view_t expected_name,
                      uint8_t expected_bit_width,
                      loom_spirv_scalar_type_kind_t expected_kind,
                      loom_spirv_fp_encoding_t expected_fp_encoding,
                      loom_spirv_feature_bits_t expected_feature_bits) {
  const loom_spirv_scalar_type_descriptor_t* descriptor =
      loom_spirv_scalar_type_descriptor(type);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->type, type);
  EXPECT_TRUE(iree_string_view_equal(descriptor->name, expected_name));
  EXPECT_EQ(descriptor->bit_width, expected_bit_width);
  EXPECT_EQ(descriptor->kind, expected_kind);
  EXPECT_EQ(descriptor->fp_encoding, expected_fp_encoding);
  EXPECT_EQ(descriptor->required_feature_bits, expected_feature_bits);
}

TEST(SpirvScalarTypesTest, DescriptorsCarryCoreKhrFeatureFacts) {
  ExpectScalarType(LOOM_SPIRV_SCALAR_TYPE_F16, IREE_SV("f16"), 16,
                   LOOM_SPIRV_SCALAR_TYPE_KIND_FLOAT,
                   LOOM_SPIRV_FP_ENCODING_MAX, LOOM_SPIRV_FEATURE_FLOAT16);
  ExpectScalarType(LOOM_SPIRV_SCALAR_TYPE_F32, IREE_SV("f32"), 32,
                   LOOM_SPIRV_SCALAR_TYPE_KIND_FLOAT,
                   LOOM_SPIRV_FP_ENCODING_MAX, 0);
  ExpectScalarType(LOOM_SPIRV_SCALAR_TYPE_F64, IREE_SV("f64"), 64,
                   LOOM_SPIRV_SCALAR_TYPE_KIND_FLOAT,
                   LOOM_SPIRV_FP_ENCODING_MAX, LOOM_SPIRV_FEATURE_FLOAT64);
  ExpectScalarType(LOOM_SPIRV_SCALAR_TYPE_BF16, IREE_SV("bf16"), 16,
                   LOOM_SPIRV_SCALAR_TYPE_KIND_FLOAT,
                   LOOM_SPIRV_FP_ENCODING_B_FLOAT16_KHR,
                   LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR);

  ExpectScalarType(LOOM_SPIRV_SCALAR_TYPE_S8, IREE_SV("s8"), 8,
                   LOOM_SPIRV_SCALAR_TYPE_KIND_SIGNED_INT,
                   LOOM_SPIRV_FP_ENCODING_MAX, LOOM_SPIRV_FEATURE_INT8);
  ExpectScalarType(LOOM_SPIRV_SCALAR_TYPE_S16, IREE_SV("s16"), 16,
                   LOOM_SPIRV_SCALAR_TYPE_KIND_SIGNED_INT,
                   LOOM_SPIRV_FP_ENCODING_MAX, LOOM_SPIRV_FEATURE_INT16);
  ExpectScalarType(LOOM_SPIRV_SCALAR_TYPE_S32, IREE_SV("s32"), 32,
                   LOOM_SPIRV_SCALAR_TYPE_KIND_SIGNED_INT,
                   LOOM_SPIRV_FP_ENCODING_MAX, 0);
  ExpectScalarType(LOOM_SPIRV_SCALAR_TYPE_S64, IREE_SV("s64"), 64,
                   LOOM_SPIRV_SCALAR_TYPE_KIND_SIGNED_INT,
                   LOOM_SPIRV_FP_ENCODING_MAX, LOOM_SPIRV_FEATURE_INT64);

  ExpectScalarType(LOOM_SPIRV_SCALAR_TYPE_U8, IREE_SV("u8"), 8,
                   LOOM_SPIRV_SCALAR_TYPE_KIND_UNSIGNED_INT,
                   LOOM_SPIRV_FP_ENCODING_MAX, LOOM_SPIRV_FEATURE_INT8);
  ExpectScalarType(LOOM_SPIRV_SCALAR_TYPE_U16, IREE_SV("u16"), 16,
                   LOOM_SPIRV_SCALAR_TYPE_KIND_UNSIGNED_INT,
                   LOOM_SPIRV_FP_ENCODING_MAX, LOOM_SPIRV_FEATURE_INT16);
  ExpectScalarType(LOOM_SPIRV_SCALAR_TYPE_U32, IREE_SV("u32"), 32,
                   LOOM_SPIRV_SCALAR_TYPE_KIND_UNSIGNED_INT,
                   LOOM_SPIRV_FP_ENCODING_MAX, 0);
  ExpectScalarType(LOOM_SPIRV_SCALAR_TYPE_U64, IREE_SV("u64"), 64,
                   LOOM_SPIRV_SCALAR_TYPE_KIND_UNSIGNED_INT,
                   LOOM_SPIRV_FP_ENCODING_MAX, LOOM_SPIRV_FEATURE_INT64);
}

TEST(SpirvScalarTypesTest, NumericFormatsProjectToScalarTypes) {
  loom_spirv_scalar_type_t scalar_type = LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;

  EXPECT_TRUE(loom_spirv_scalar_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_F32, &scalar_type));
  EXPECT_EQ(scalar_type, LOOM_SPIRV_SCALAR_TYPE_F32);

  EXPECT_TRUE(loom_spirv_scalar_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_BF16, &scalar_type));
  EXPECT_EQ(scalar_type, LOOM_SPIRV_SCALAR_TYPE_BF16);

  EXPECT_TRUE(loom_spirv_scalar_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_I8, &scalar_type));
  EXPECT_EQ(scalar_type, LOOM_SPIRV_SCALAR_TYPE_S8);

  EXPECT_TRUE(loom_spirv_scalar_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_U16, &scalar_type));
  EXPECT_EQ(scalar_type, LOOM_SPIRV_SCALAR_TYPE_U16);

  EXPECT_FALSE(loom_spirv_scalar_type_from_numeric_format(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_UNKNOWN, &scalar_type));
  EXPECT_EQ(scalar_type, LOOM_SPIRV_SCALAR_TYPE_UNKNOWN);
}

}  // namespace
