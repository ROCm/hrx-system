// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/numeric_format.h"

#include "iree/testing/gtest.h"

namespace {

TEST(NumericFormatTest, RejectsUnknownAndMultiBitFormats) {
  const loom_numeric_format_info_t* info = nullptr;
  EXPECT_FALSE(
      loom_numeric_format_info(LOOM_VALUE_FACT_NUMERIC_FORMAT_UNKNOWN, &info));
  EXPECT_EQ(info, nullptr);

  EXPECT_FALSE(loom_numeric_format_info(LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3 |
                                            LOOM_VALUE_FACT_NUMERIC_FORMAT_BF8,
                                        &info));
  EXPECT_EQ(info, nullptr);
}

TEST(NumericFormatTest, DescribesFp8AndBf8Semantics) {
  const loom_numeric_format_info_t* fp8 = nullptr;
  ASSERT_TRUE(
      loom_numeric_format_info(LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3, &fp8));
  EXPECT_EQ(fp8->kind, LOOM_NUMERIC_FORMAT_KIND_FLOAT);
  EXPECT_EQ(fp8->float_family, LOOM_NUMERIC_FLOAT_FAMILY_FP8);
  EXPECT_EQ(fp8->storage_bit_count, 8);
  EXPECT_EQ(fp8->exponent_bit_count, 4);
  EXPECT_EQ(fp8->mantissa_bit_count, 3);
  EXPECT_TRUE(
      iree_any_bit_set(fp8->flags, LOOM_NUMERIC_FORMAT_FLAG_HAS_INFINITY));
  EXPECT_TRUE(loom_numeric_format_needs_encoded_payload_selector(fp8->format));

  const loom_numeric_format_info_t* fp8_fn = nullptr;
  ASSERT_TRUE(loom_numeric_format_info(LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN,
                                       &fp8_fn));
  EXPECT_TRUE(
      iree_any_bit_set(fp8_fn->flags, LOOM_NUMERIC_FORMAT_FLAG_FINITE_ONLY));
  EXPECT_FALSE(
      iree_any_bit_set(fp8_fn->flags, LOOM_NUMERIC_FORMAT_FLAG_HAS_INFINITY));

  const loom_numeric_format_info_t* bf8 = nullptr;
  ASSERT_TRUE(
      loom_numeric_format_info(LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2, &bf8));
  EXPECT_EQ(bf8->kind, LOOM_NUMERIC_FORMAT_KIND_FLOAT);
  EXPECT_EQ(bf8->float_family, LOOM_NUMERIC_FLOAT_FAMILY_BF8);
  EXPECT_EQ(bf8->storage_bit_count, 8);
  EXPECT_EQ(bf8->exponent_bit_count, 5);
  EXPECT_EQ(bf8->mantissa_bit_count, 2);
  EXPECT_TRUE(
      iree_any_bit_set(bf8->flags, LOOM_NUMERIC_FORMAT_FLAG_HAS_INFINITY));
  EXPECT_TRUE(loom_numeric_format_needs_encoded_payload_selector(bf8->format));
}

TEST(NumericFormatTest, DescribesFnuZSpecialValuePolicy) {
  const loom_numeric_format_info_t* info = nullptr;
  ASSERT_TRUE(loom_numeric_format_info(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FNUZ, &info));
  EXPECT_TRUE(
      iree_any_bit_set(info->flags, LOOM_NUMERIC_FORMAT_FLAG_FINITE_ONLY));
  EXPECT_TRUE(
      iree_any_bit_set(info->flags, LOOM_NUMERIC_FORMAT_FLAG_UNSIGNED_ZERO));
  EXPECT_TRUE(loom_numeric_format_is_finite_only(info->format));
}

TEST(NumericFormatTest, DescribesFutureNarrowFloatPayloads) {
  const loom_numeric_format_info_t* fp6 = nullptr;
  ASSERT_TRUE(
      loom_numeric_format_info(LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E2M3, &fp6));
  EXPECT_EQ(fp6->float_family, LOOM_NUMERIC_FLOAT_FAMILY_FP6);
  EXPECT_EQ(fp6->storage_bit_count, 6);
  EXPECT_EQ(fp6->exponent_bit_count, 2);
  EXPECT_EQ(fp6->mantissa_bit_count, 3);
  EXPECT_TRUE(loom_numeric_format_is_finite_only(fp6->format));
  EXPECT_TRUE(loom_numeric_format_needs_encoded_payload_selector(fp6->format));

  const loom_numeric_format_info_t* bf6 = nullptr;
  ASSERT_TRUE(
      loom_numeric_format_info(LOOM_VALUE_FACT_NUMERIC_FORMAT_BF6, &bf6));
  EXPECT_EQ(bf6->float_family, LOOM_NUMERIC_FLOAT_FAMILY_BF6);
  EXPECT_EQ(bf6->storage_bit_count, 6);
  EXPECT_EQ(bf6->exponent_bit_count, 5);
  EXPECT_EQ(bf6->mantissa_bit_count, 0);
  EXPECT_FALSE(loom_numeric_format_is_finite_only(bf6->format));
  EXPECT_TRUE(loom_numeric_format_needs_encoded_payload_selector(bf6->format));

  const loom_numeric_format_info_t* fp4 = nullptr;
  ASSERT_TRUE(
      loom_numeric_format_info(LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1, &fp4));
  EXPECT_EQ(fp4->float_family, LOOM_NUMERIC_FLOAT_FAMILY_FP4);
  EXPECT_EQ(fp4->storage_bit_count, 4);
  EXPECT_EQ(fp4->exponent_bit_count, 2);
  EXPECT_EQ(fp4->mantissa_bit_count, 1);
  EXPECT_TRUE(loom_numeric_format_is_finite_only(fp4->format));
  EXPECT_TRUE(loom_numeric_format_needs_encoded_payload_selector(fp4->format));
}

TEST(NumericFormatTest, LeavesScaleExponentFormatOutOfPayloadSelectors) {
  const loom_numeric_format_info_t* info = nullptr;
  ASSERT_TRUE(
      loom_numeric_format_info(LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0, &info));
  EXPECT_EQ(info->float_family, LOOM_NUMERIC_FLOAT_FAMILY_F8_E8M0);
  EXPECT_EQ(info->storage_bit_count, 8);
  EXPECT_EQ(info->exponent_bit_count, 8);
  EXPECT_EQ(info->mantissa_bit_count, 0);
  EXPECT_FALSE(
      loom_numeric_format_needs_encoded_payload_selector(info->format));
}

TEST(NumericFormatTest, MapsDirectScalarTypesToNumericFormats) {
  EXPECT_EQ(loom_numeric_format_from_scalar_type(LOOM_SCALAR_TYPE_I1),
            LOOM_VALUE_FACT_NUMERIC_FORMAT_I1);
  EXPECT_EQ(loom_numeric_format_from_scalar_type(LOOM_SCALAR_TYPE_I8),
            LOOM_VALUE_FACT_NUMERIC_FORMAT_I8);
  EXPECT_EQ(loom_numeric_format_from_scalar_type(LOOM_SCALAR_TYPE_I16),
            LOOM_VALUE_FACT_NUMERIC_FORMAT_I16);
  EXPECT_EQ(loom_numeric_format_from_scalar_type(LOOM_SCALAR_TYPE_I32),
            LOOM_VALUE_FACT_NUMERIC_FORMAT_I32);
  EXPECT_EQ(loom_numeric_format_from_scalar_type(LOOM_SCALAR_TYPE_F8E4M3),
            LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN);
  EXPECT_EQ(loom_numeric_format_from_scalar_type(LOOM_SCALAR_TYPE_F8E5M2),
            LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2);
  EXPECT_EQ(loom_numeric_format_from_scalar_type(LOOM_SCALAR_TYPE_F16),
            LOOM_VALUE_FACT_NUMERIC_FORMAT_F16);
  EXPECT_EQ(loom_numeric_format_from_scalar_type(LOOM_SCALAR_TYPE_BF16),
            LOOM_VALUE_FACT_NUMERIC_FORMAT_BF16);
  EXPECT_EQ(loom_numeric_format_from_scalar_type(LOOM_SCALAR_TYPE_F32),
            LOOM_VALUE_FACT_NUMERIC_FORMAT_F32);
  EXPECT_EQ(loom_numeric_format_from_scalar_type(LOOM_SCALAR_TYPE_F64),
            LOOM_VALUE_FACT_NUMERIC_FORMAT_F64);
  EXPECT_EQ(loom_numeric_format_from_scalar_type(LOOM_SCALAR_TYPE_INDEX),
            LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE);
  EXPECT_EQ(loom_numeric_format_from_scalar_type(LOOM_SCALAR_TYPE_OFFSET),
            LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE);
  EXPECT_EQ(loom_numeric_format_from_scalar_type(LOOM_SCALAR_TYPE_I64),
            LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE);
  EXPECT_EQ(loom_numeric_format_from_scalar_type(LOOM_SCALAR_TYPE_NONE),
            LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE);
  EXPECT_EQ(loom_numeric_format_from_scalar_type(LOOM_SCALAR_TYPE_COUNT_),
            LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE);
}

TEST(NumericFormatTest, MapsNumericFormatsToDirectScalarTypes) {
  loom_scalar_type_t type = LOOM_SCALAR_TYPE_NONE;
  EXPECT_TRUE(loom_numeric_format_direct_scalar_type(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN, &type));
  EXPECT_EQ(type, LOOM_SCALAR_TYPE_F8E4M3);
  EXPECT_TRUE(loom_numeric_format_direct_scalar_type(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2FNUZ, &type));
  EXPECT_EQ(type, LOOM_SCALAR_TYPE_F8E5M2);
  EXPECT_TRUE(loom_numeric_format_direct_scalar_type(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_U32, &type));
  EXPECT_EQ(type, LOOM_SCALAR_TYPE_I32);
  EXPECT_TRUE(loom_numeric_format_direct_scalar_type(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I8, &type));
  EXPECT_EQ(type, LOOM_SCALAR_TYPE_I8);

  EXPECT_FALSE(loom_numeric_format_direct_scalar_type(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1, &type));
  EXPECT_FALSE(loom_numeric_format_direct_scalar_type(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_UNKNOWN, &type));
}

TEST(NumericFormatTest, DescribesUnsignedIntegerSemantics) {
  EXPECT_TRUE(loom_numeric_format_uses_unsigned_integer_semantics(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_U8));
  EXPECT_TRUE(loom_numeric_format_uses_unsigned_integer_semantics(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_CODEBOOK_INDEX));
  EXPECT_FALSE(loom_numeric_format_uses_unsigned_integer_semantics(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_I8));
  EXPECT_FALSE(loom_numeric_format_uses_unsigned_integer_semantics(
      LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN));
}

}  // namespace
