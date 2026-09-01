// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/math.h"

#include <cfloat>
#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

#include "iree/base/internal/fpu_state.h"
#include "iree/testing/gtest.h"

namespace {

//==============================================================================
// Bitwise rotation (aka circular shifts)
//==============================================================================

TEST(BitwiseRotationTest, ROTL32) {
  EXPECT_EQ(0u, iree_math_rotl_u32(0u, 0u));
  EXPECT_EQ(1u, iree_math_rotl_u32(1u, 0u));
  EXPECT_EQ(2u, iree_math_rotl_u32(1u, 1u));
  EXPECT_EQ(UINT32_C(0x80000000), iree_math_rotl_u32(1u, 31u));
  EXPECT_EQ(1u, iree_math_rotl_u32(1u, 32u));
  EXPECT_EQ(2u, iree_math_rotl_u32(1u, 33u));
  EXPECT_EQ(UINT32_C(0x34567812), iree_math_rotl_u32(UINT32_C(0x12345678), 8u));
  EXPECT_EQ(UINT32_MAX, iree_math_rotl_u32(UINT32_MAX, 17u));
}

TEST(BitwiseRotationTest, ROTR32) {
  EXPECT_EQ(0u, iree_math_rotr_u32(0u, 0u));
  EXPECT_EQ(1u, iree_math_rotr_u32(1u, 0u));
  EXPECT_EQ(UINT32_C(0x80000000), iree_math_rotr_u32(1u, 1u));
  EXPECT_EQ(2u, iree_math_rotr_u32(1u, 31u));
  EXPECT_EQ(1u, iree_math_rotr_u32(1u, 32u));
  EXPECT_EQ(UINT32_C(0x80000000), iree_math_rotr_u32(1u, 33u));
  EXPECT_EQ(UINT32_C(0x78123456), iree_math_rotr_u32(UINT32_C(0x12345678), 8u));
  EXPECT_EQ(UINT32_MAX, iree_math_rotr_u32(UINT32_MAX, 17u));
}

TEST(BitwiseRotationTest, ROTL64) {
  EXPECT_EQ(0ull, iree_math_rotl_u64(0ull, 0u));
  EXPECT_EQ(1ull, iree_math_rotl_u64(1ull, 0u));
  EXPECT_EQ(2ull, iree_math_rotl_u64(1ull, 1u));
  EXPECT_EQ(UINT64_C(0x8000000000000000), iree_math_rotl_u64(1ull, 63u));
  EXPECT_EQ(1ull, iree_math_rotl_u64(1ull, 64u));
  EXPECT_EQ(2ull, iree_math_rotl_u64(1ull, 65u));
  EXPECT_EQ(UINT64_C(0x23456789ABCDEF01),
            iree_math_rotl_u64(UINT64_C(0x0123456789ABCDEF), 8u));
  EXPECT_EQ(UINT64_MAX, iree_math_rotl_u64(UINT64_MAX, 17u));
}

TEST(BitwiseRotationTest, ROTR64) {
  EXPECT_EQ(0ull, iree_math_rotr_u64(0ull, 0u));
  EXPECT_EQ(1ull, iree_math_rotr_u64(1ull, 0u));
  EXPECT_EQ(UINT64_C(0x8000000000000000), iree_math_rotr_u64(1ull, 1u));
  EXPECT_EQ(2ull, iree_math_rotr_u64(1ull, 63u));
  EXPECT_EQ(1ull, iree_math_rotr_u64(1ull, 64u));
  EXPECT_EQ(UINT64_C(0x8000000000000000), iree_math_rotr_u64(1ull, 65u));
  EXPECT_EQ(UINT64_C(0xEF0123456789ABCD),
            iree_math_rotr_u64(UINT64_C(0x0123456789ABCDEF), 8u));
  EXPECT_EQ(UINT64_MAX, iree_math_rotr_u64(UINT64_MAX, 17u));
}

//==============================================================================
// Bit scanning/counting
//==============================================================================

TEST(BitwiseScansTest, CLZ32) {
  EXPECT_EQ(32, iree_math_count_leading_zeros_u32(uint32_t{}));
  EXPECT_EQ(0, iree_math_count_leading_zeros_u32(~uint32_t{}));
  for (int index = 0; index < 32; index++) {
    uint32_t x = 1u << index;
    const int cnt = 31 - index;
    ASSERT_EQ(cnt, iree_math_count_leading_zeros_u32(x)) << index;
    ASSERT_EQ(cnt, iree_math_count_leading_zeros_u32(x + x - 1)) << index;
  }
}

TEST(BitwiseScansTest, CLZ64) {
  EXPECT_EQ(64, iree_math_count_leading_zeros_u64(uint64_t{}));
  EXPECT_EQ(0, iree_math_count_leading_zeros_u64(~uint64_t{}));
  for (int index = 0; index < 64; index++) {
    uint64_t x = 1ull << index;
    const int cnt = 63 - index;
    ASSERT_EQ(cnt, iree_math_count_leading_zeros_u64(x)) << index;
    ASSERT_EQ(cnt, iree_math_count_leading_zeros_u64(x + x - 1)) << index;
  }
}

TEST(BitwiseScansTest, CTZ32) {
  EXPECT_EQ(0, iree_math_count_trailing_zeros_u32(~uint32_t{}));
  for (int index = 0; index < 32; index++) {
    uint32_t x = static_cast<uint32_t>(1) << index;
    const int cnt = index;
    ASSERT_EQ(cnt, iree_math_count_trailing_zeros_u32(x)) << index;
    ASSERT_EQ(cnt, iree_math_count_trailing_zeros_u32(~(x - 1))) << index;
  }
}

TEST(BitwiseScansTest, CTZ64) {
  // iree_math_count_trailing_zeros_u32
  EXPECT_EQ(0, iree_math_count_trailing_zeros_u64(~uint64_t{}));
  for (int index = 0; index < 64; index++) {
    uint64_t x = static_cast<uint64_t>(1) << index;
    const int cnt = index;
    ASSERT_EQ(cnt, iree_math_count_trailing_zeros_u64(x)) << index;
    ASSERT_EQ(cnt, iree_math_count_trailing_zeros_u64(~(x - 1))) << index;
  }
}

//==============================================================================
// Population count
//==============================================================================

TEST(PopulationCountTest, Ones32) {
  EXPECT_EQ(0, iree_math_count_ones_u32(0u));
  EXPECT_EQ(1, iree_math_count_ones_u32(1u));
  EXPECT_EQ(29, iree_math_count_ones_u32(-15u));
  EXPECT_EQ(5, iree_math_count_ones_u32(341u));
  EXPECT_EQ(32, iree_math_count_ones_u32(UINT32_MAX));
  EXPECT_EQ(31, iree_math_count_ones_u32(UINT32_MAX - 1));
}

TEST(PopulationCountTest, Ones64) {
  EXPECT_EQ(0, iree_math_count_ones_u64(0ull));
  EXPECT_EQ(1, iree_math_count_ones_u64(1ull));
  EXPECT_EQ(61, iree_math_count_ones_u64(-15ull));
  EXPECT_EQ(5, iree_math_count_ones_u64(341ull));
  EXPECT_EQ(64, iree_math_count_ones_u64(UINT64_MAX));
  EXPECT_EQ(63, iree_math_count_ones_u64(UINT64_MAX - 1ull));
}

TEST(BitWidthTest, MaskLowBits) {
  EXPECT_EQ(0u, iree_math_mask_low_bits_u32(UINT32_MAX, 0));
  EXPECT_EQ(UINT32_C(0xFF), iree_math_mask_low_bits_u32(UINT32_MAX, 8));
  EXPECT_EQ(UINT32_C(0x5678),
            iree_math_mask_low_bits_u32(UINT32_C(0x12345678), 16));
  EXPECT_EQ(UINT32_C(0x12345678),
            iree_math_mask_low_bits_u32(UINT32_C(0x12345678), 32));
  EXPECT_EQ(0ull, iree_math_mask_low_bits_u64(UINT64_MAX, -1));
  EXPECT_EQ(UINT64_C(0xFFFF), iree_math_mask_low_bits_u64(UINT64_MAX, 16));
  EXPECT_EQ(UINT64_MAX, iree_math_mask_low_bits_u64(UINT64_MAX, 65));
}

TEST(BitWidthTest, CountBits) {
  EXPECT_EQ(31, iree_math_count_leading_zeros_u64_width(1, 32));
  EXPECT_EQ(7, iree_math_count_leading_zeros_u64_width(1, 8));
  EXPECT_EQ(32, iree_math_count_leading_zeros_u64_width(0, 32));
  EXPECT_EQ(0, iree_math_count_leading_zeros_u64_width(1, 0));
  EXPECT_EQ(3, iree_math_count_trailing_zeros_u64_width(8, 32));
  EXPECT_EQ(8, iree_math_count_trailing_zeros_u64_width(0, 8));
  EXPECT_EQ(32, iree_math_count_ones_u64_width(UINT64_MAX, 32));
  EXPECT_EQ(8, iree_math_count_ones_u64_width(UINT64_MAX, 8));
}

TEST(IntegerNumberTheoryTest, GreatestCommonDivisor) {
  EXPECT_EQ(0ull, iree_math_gcd_u64(0, 0));
  EXPECT_EQ(5ull, iree_math_gcd_u64(0, 5));
  EXPECT_EQ(4ull, iree_math_gcd_u64(12, 8));
  EXPECT_EQ(1ull, iree_math_gcd_u64(17, 13));
  EXPECT_EQ(256ull, iree_math_gcd_u64(256, 1024));
  EXPECT_EQ(4ull, iree_math_gcd_i64(-12, 8));
  EXPECT_EQ(UINT64_C(1) << 63, iree_math_gcd_i64(INT64_MIN, 0));
}

TEST(IntegerNumberTheoryTest, LeastCommonMultiple) {
  uint64_t result = 0;
  EXPECT_TRUE(iree_math_checked_lcm_u64(0, 5, &result));
  EXPECT_EQ(0ull, result);
  EXPECT_TRUE(iree_math_checked_lcm_u64(16, 24, &result));
  EXPECT_EQ(48ull, result);
  EXPECT_FALSE(iree_math_checked_lcm_u64(UINT64_MAX, 2, &result));

  int64_t signed_result = 0;
  EXPECT_TRUE(iree_math_checked_lcm_i64(-16, 24, &signed_result));
  EXPECT_EQ(48, signed_result);
  EXPECT_FALSE(iree_math_checked_lcm_i64(INT64_MIN, 1, &signed_result));
}

TEST(IntegerNumberTheoryTest, FloorLog2) {
  EXPECT_EQ(0, iree_math_floor_log2_u64(1));
  EXPECT_EQ(1, iree_math_floor_log2_u64(3));
  EXPECT_EQ(10, iree_math_floor_log2_u64(1024));
  EXPECT_EQ(63, iree_math_floor_log2_u64(UINT64_MAX));
}

TEST(IntegerNumberTheoryTest, IsPowerOfTwoSigned) {
  EXPECT_TRUE(iree_math_is_power_of_two_i64(1));
  EXPECT_TRUE(iree_math_is_power_of_two_i64(INT64_C(1) << 62));
  EXPECT_FALSE(iree_math_is_power_of_two_i64(0));
  EXPECT_FALSE(iree_math_is_power_of_two_i64(-1));
  EXPECT_FALSE(iree_math_is_power_of_two_i64(INT64_MIN));
  EXPECT_FALSE(iree_math_is_power_of_two_i64(6));
}

//==============================================================================
// Rounding and alignment
//==============================================================================

static void ExpectMulU64ToU128(uint64_t lhs, uint64_t rhs,
                               uint64_t expected_high, uint64_t expected_low) {
  SCOPED_TRACE(::testing::Message() << "lhs=" << lhs << ", rhs=" << rhs
                                    << ", expected_high=" << expected_high
                                    << ", expected_low=" << expected_low);

  uint64_t high = UINT64_MAX;
  uint64_t low = UINT64_MAX;
  iree_math_mul_u64_to_u128(lhs, rhs, &high, &low);
  EXPECT_EQ(expected_high, high);
  EXPECT_EQ(expected_low, low);

  high = UINT64_MAX;
  low = UINT64_MAX;
  iree_math_mul_u64_to_u128(rhs, lhs, &high, &low);
  EXPECT_EQ(expected_high, high);
  EXPECT_EQ(expected_low, low);
}

TEST(RoundingTest, MulU64ToU128) {
  ExpectMulU64ToU128(0, UINT64_MAX, 0, 0);
  ExpectMulU64ToU128(1, UINT64_MAX, 0, UINT64_MAX);
  ExpectMulU64ToU128(UINT32_MAX, UINT32_MAX, 0, UINT64_C(0xFFFFFFFE00000001));
  ExpectMulU64ToU128(UINT64_C(1) << 32, UINT64_C(1) << 32, 1, 0);
  ExpectMulU64ToU128(UINT64_MAX, 2, 1, UINT64_MAX - 1ull);
  ExpectMulU64ToU128(UINT64_MAX, UINT64_MAX, UINT64_MAX - 1ull, 1);
}

static void ExpectDivU128ByU64ToU64(uint64_t high, uint64_t low,
                                    uint64_t denominator, bool expected_ok,
                                    uint64_t expected_quotient = 0) {
  SCOPED_TRACE(::testing::Message() << "high=" << high << ", low=" << low
                                    << ", denominator=" << denominator);

  uint64_t quotient = 1;
  EXPECT_EQ(expected_ok, iree_math_div_u128_by_u64_to_u64(
                             high, low, denominator, &quotient));
  EXPECT_EQ(expected_ok ? expected_quotient : 0ull, quotient);
}

TEST(RoundingTest, DivU128ByU64ToU64) {
  ExpectDivU128ByU64ToU64(0, 100, 4, true, 25);
  ExpectDivU128ByU64ToU64(1, 0, 2, true, UINT64_C(1) << 63);
  ExpectDivU128ByU64ToU64(16, UINT64_MAX, 17, true, UINT64_MAX);
  ExpectDivU128ByU64ToU64(UINT64_MAX - 1ull, UINT64_MAX, UINT64_MAX, true,
                          UINT64_MAX);

  // Exercises the 65-bit shifted remainder case in the restoring division loop.
  ExpectDivU128ByU64ToU64(UINT64_C(1) << 63, 0, UINT64_MAX, true,
                          UINT64_C(1) << 63);

  ExpectDivU128ByU64ToU64(0, 1, 0, false);
  ExpectDivU128ByU64ToU64(5, 0, 5, false);
  ExpectDivU128ByU64ToU64(6, 0, 5, false);
}

static void ExpectRoundMulDivU64(uint64_t value, uint64_t numerator,
                                 uint64_t denominator, bool expected_ok,
                                 uint64_t expected_result = 0) {
  SCOPED_TRACE(::testing::Message()
               << "value=" << value << ", numerator=" << numerator
               << ", denominator=" << denominator);

  uint64_t result = 1;
  EXPECT_EQ(expected_ok, iree_math_round_mul_div_u64(value, numerator,
                                                     denominator, &result));
  EXPECT_EQ(expected_ok ? expected_result : 0ull, result);

  result = 1;
  EXPECT_EQ(expected_ok, iree_math_round_mul_div_u64_portable(
                             value, numerator, denominator, &result));
  EXPECT_EQ(expected_ok ? expected_result : 0ull, result);
}

TEST(RoundingTest, RoundMulDivU64) {
  ExpectRoundMulDivU64(1, 1, 0, false);
  ExpectRoundMulDivU64(0, 123, 456, true, 0);
  ExpectRoundMulDivU64(123, 0, 456, true, 0);

  ExpectRoundMulDivU64(10, 10, 4, true, 25);
  ExpectRoundMulDivU64(1, 1, 2, true, 1);
  ExpectRoundMulDivU64(1, 1, 3, true, 0);
  ExpectRoundMulDivU64(2, 1, 3, true, 1);

  ExpectRoundMulDivU64(UINT64_MAX, 1, UINT64_MAX, true, 1);
  ExpectRoundMulDivU64(UINT64_MAX, 2, UINT64_MAX, true, 2);
  ExpectRoundMulDivU64(UINT64_MAX, 2, 2, true, UINT64_MAX);
  ExpectRoundMulDivU64(UINT64_MAX, 3, UINT64_MAX, true, 3);
  ExpectRoundMulDivU64(UINT64_MAX, UINT64_MAX, UINT64_MAX, true, UINT64_MAX);
  ExpectRoundMulDivU64(UINT64_MAX, UINT64_C(1) << 63, UINT64_C(1) << 63, true,
                       UINT64_MAX);

  ExpectRoundMulDivU64(UINT64_MAX, UINT64_MAX, 1, false);
  ExpectRoundMulDivU64(UINT64_MAX, UINT64_MAX, UINT64_MAX - 1ull, false);
}

#if defined(__SIZEOF_INT128__) && !defined(IREE_COMPILER_MSVC_COMPAT)
TEST(RoundingTest, RoundMulDivU64PortableMatchesUInt128) {
  const uint64_t values[] = {
      0,
      1,
      2,
      3,
      17,
      UINT32_MAX,
      UINT64_C(1) << 32,
      UINT64_C(1) << 63,
      UINT64_MAX - 1,
      UINT64_MAX,
  };
  for (uint64_t value : values) {
    for (uint64_t numerator : values) {
      for (uint64_t denominator : values) {
        if (denominator == 0) continue;
        SCOPED_TRACE(::testing::Message()
                     << "value=" << value << ", numerator=" << numerator
                     << ", denominator=" << denominator);

        const __uint128_t rounded_product =
            (__uint128_t)value * (__uint128_t)numerator + denominator / 2;
        const __uint128_t quotient = rounded_product / denominator;
        const bool expected_ok = quotient <= UINT64_MAX;
        const uint64_t expected_result = expected_ok ? (uint64_t)quotient : 0;

        uint64_t result = 1;
        EXPECT_EQ(expected_ok, iree_math_round_mul_div_u64_portable(
                                   value, numerator, denominator, &result));
        EXPECT_EQ(expected_result, result);
      }
    }
  }
}
#endif  // defined(__SIZEOF_INT128__) && !defined(IREE_COMPILER_MSVC_COMPAT)

TEST(RoundingTest, UpToNextPow232) {
  constexpr uint32_t kUint16Max = UINT16_MAX;
  constexpr uint32_t kUint32Max = UINT32_MAX;
  EXPECT_EQ(0u, iree_math_round_up_to_pow2_u32(0u));
  EXPECT_EQ(1u, iree_math_round_up_to_pow2_u32(1u));
  EXPECT_EQ(2u, iree_math_round_up_to_pow2_u32(2u));
  EXPECT_EQ(4u, iree_math_round_up_to_pow2_u32(3u));
  EXPECT_EQ(8u, iree_math_round_up_to_pow2_u32(8u));
  EXPECT_EQ(16u, iree_math_round_up_to_pow2_u32(9u));
  EXPECT_EQ(kUint16Max + 1u, iree_math_round_up_to_pow2_u32(kUint16Max - 1u));
  EXPECT_EQ(kUint16Max + 1u, iree_math_round_up_to_pow2_u32(kUint16Max));
  EXPECT_EQ(kUint16Max + 1u, iree_math_round_up_to_pow2_u32(kUint16Max + 1u));
  EXPECT_EQ(131072u, iree_math_round_up_to_pow2_u32(kUint16Max + 2u));
  EXPECT_EQ(262144u, iree_math_round_up_to_pow2_u32(262144u - 1u));
  EXPECT_EQ(0x80000000u, iree_math_round_up_to_pow2_u32(0x7FFFFFFFu));
  EXPECT_EQ(0x80000000u, iree_math_round_up_to_pow2_u32(0x80000000u));

  // NOTE: wrap to 0.
  EXPECT_EQ(0u, iree_math_round_up_to_pow2_u32(0x80000001u));
  EXPECT_EQ(0u, iree_math_round_up_to_pow2_u32(kUint32Max - 1u));
  EXPECT_EQ(0u, iree_math_round_up_to_pow2_u32(kUint32Max));
}

TEST(RoundingTest, UpToNextPow264) {
  constexpr uint64_t kUint16Max = UINT16_MAX;
  constexpr uint64_t kUint64Max = UINT64_MAX;
  EXPECT_EQ(0ull, iree_math_round_up_to_pow2_u64(0ull));
  EXPECT_EQ(1ull, iree_math_round_up_to_pow2_u64(1ull));
  EXPECT_EQ(2ull, iree_math_round_up_to_pow2_u64(2ull));
  EXPECT_EQ(4ull, iree_math_round_up_to_pow2_u64(3ull));
  EXPECT_EQ(8ull, iree_math_round_up_to_pow2_u64(8ull));
  EXPECT_EQ(16ull, iree_math_round_up_to_pow2_u64(9ull));
  EXPECT_EQ(kUint16Max + 1ull,
            iree_math_round_up_to_pow2_u64(kUint16Max - 1ull));
  EXPECT_EQ(kUint16Max + 1ull, iree_math_round_up_to_pow2_u64(kUint16Max));
  EXPECT_EQ(kUint16Max + 1ull,
            iree_math_round_up_to_pow2_u64(kUint16Max + 1ull));
  EXPECT_EQ(131072ull, iree_math_round_up_to_pow2_u64(kUint16Max + 2ull));
  EXPECT_EQ(0x100000000ull, iree_math_round_up_to_pow2_u64(0xFFFFFFFEull));
  EXPECT_EQ(0x100000000ull, iree_math_round_up_to_pow2_u64(0xFFFFFFFFull));
  EXPECT_EQ(0x80000000ull, iree_math_round_up_to_pow2_u64(0x7FFFFFFFull));
  EXPECT_EQ(0x80000000ull, iree_math_round_up_to_pow2_u64(0x80000000ull));
  EXPECT_EQ(0x100000000ull, iree_math_round_up_to_pow2_u64(0x80000001ull));

  // NOTE: wrap to 0.
  EXPECT_EQ(0ull, iree_math_round_up_to_pow2_u64(0x8000000000000001ull));
  EXPECT_EQ(0ull, iree_math_round_up_to_pow2_u64(kUint64Max - 1ull));
  EXPECT_EQ(0ull, iree_math_round_up_to_pow2_u64(kUint64Max));
}

//==============================================================================
// FP16 support
//==============================================================================

TEST(F16ConversionTest, F32ToF16) {
  constexpr float kF16Max = 65504.f;
  constexpr float kF16Min = 1.f / 16384.f;
  // Within range, normal truncation.
  EXPECT_EQ(0x0000, iree_math_f32_to_f16(0.f));
  EXPECT_EQ(0x8000, iree_math_f32_to_f16(-0.f));
  EXPECT_EQ(0x3400, iree_math_f32_to_f16(0.25f));
  EXPECT_EQ(0xd646, iree_math_f32_to_f16(-100.375f));
  EXPECT_EQ(0x7BFF, iree_math_f32_to_f16(kF16Max));
  EXPECT_EQ(0xFBFF, iree_math_f32_to_f16(-kF16Max));
  EXPECT_EQ(0x0400, iree_math_f32_to_f16(kF16Min));
  EXPECT_EQ(0x8400, iree_math_f32_to_f16(-kF16Min));
  // Infinity
  EXPECT_EQ(0x7c00, iree_math_f32_to_f16(INFINITY));
  EXPECT_EQ(0xfc00, iree_math_f32_to_f16(-INFINITY));
  // Overflow
  EXPECT_EQ(0x7C00, iree_math_f32_to_f16(FLT_MAX));
  EXPECT_EQ(0xFC00, iree_math_f32_to_f16(-FLT_MAX));
  // Important case to test: overflow due to rounding to nearest-even of 65520
  // to 65536.
  EXPECT_EQ(0x7C00, iree_math_f32_to_f16(65520.f));
  EXPECT_EQ(0xFC00, iree_math_f32_to_f16(-65520.f));
  EXPECT_EQ(0x7C00, iree_math_f32_to_f16(65536.f));
  EXPECT_EQ(0xFC00, iree_math_f32_to_f16(-65536.f));
  // Underflow
  EXPECT_EQ(0, iree_math_f32_to_f16(FLT_MIN));
  EXPECT_EQ(0x8000, iree_math_f32_to_f16(-FLT_MIN));
  EXPECT_EQ(0x00A8, iree_math_f32_to_f16(1.0e-05f));
  EXPECT_EQ(0x80A8, iree_math_f32_to_f16(-1.0e-05f));
  EXPECT_EQ(0x03FF, iree_math_f32_to_f16(6.1e-05f));  // Near largest denormal
  EXPECT_EQ(0x83FF, iree_math_f32_to_f16(-6.1e-05f));

  // Denormals.
  EXPECT_EQ(0x0200, iree_math_f32_to_f16(kF16Min / 2));
  EXPECT_EQ(0x8200, iree_math_f32_to_f16(-kF16Min / 2));
}

template <typename UintType>
static void CheckDenormals(
    int exp_bits, int mantissa_bits, int bias_tweak, bool have_neg_zero,
    std::function<UintType(float)> convert_f32_to_small,
    std::function<float(UintType)> convert_small_to_f32) {
  IREE_MATH_FP_FORMAT_CONSTANTS(small_, exp_bits, mantissa_bits, bias_tweak)
  for (UintType m = 0; m <= small_mantissa_mask; ++m) {
    float value = std::ldexp(m, 1 - small_exp_bias - small_mantissa_bits);
    EXPECT_EQ(value, convert_small_to_f32(m));
    EXPECT_EQ(m, convert_f32_to_small(value));
    const float half = std::ldexp(1.0f, -small_exp_bias - mantissa_bits);
    // m + 1 is the next representable value after the denormal, even if m was
    // the largest denormal, as in that case the mantissa overflows into the
    // exponent, resulting in the smallest normal value.
    const UintType rounded_midpoint = (m & 1) ? m + 1 : m;
    EXPECT_EQ(rounded_midpoint, convert_f32_to_small(value + half));
    if (m != 0 || have_neg_zero) {
      // Test negative values, similar to the above code for positive values.
      EXPECT_EQ(-value, convert_small_to_f32(m | small_sign_mask));
      EXPECT_EQ(m | small_sign_mask, convert_f32_to_small(-value));
      EXPECT_EQ(rounded_midpoint | small_sign_mask,
                convert_f32_to_small(-value - half));
    }
  }
}

static float DecodePositiveFinite(uint32_t bits, int exp_bits,
                                  int mantissa_bits, int bias_tweak) {
  const uint32_t mantissa_mask = (1u << mantissa_bits) - 1;
  const uint32_t encoded_exp = (bits >> mantissa_bits) & ((1u << exp_bits) - 1);
  const uint32_t mantissa = bits & mantissa_mask;
  const int exp_bias = bias_tweak + (1 << (exp_bits - 1)) - 1;
  const uint32_t significand =
      encoded_exp == 0 ? mantissa : (1u << mantissa_bits) | mantissa;
  const int arithmetic_exp =
      (encoded_exp == 0 ? 1 : static_cast<int>(encoded_exp)) - exp_bias -
      mantissa_bits;
  return std::ldexp(static_cast<float>(significand), arithmetic_exp);
}

template <typename UintType>
static void CheckFiniteRoundingBoundaries(
    int exp_bits, int mantissa_bits, int bias_tweak, UintType max_finite,
    std::function<UintType(float)> convert_f32_to_small) {
  const UintType sign_mask = UintType{1} << (exp_bits + mantissa_bits);
  for (UintType lower = 0; lower < max_finite; ++lower) {
    SCOPED_TRACE(::testing::Message() << "lower payload " << +lower);
    const UintType upper = lower + 1;
    const float lower_value =
        DecodePositiveFinite(lower, exp_bits, mantissa_bits, bias_tweak);
    const float upper_value =
        DecodePositiveFinite(upper, exp_bits, mantissa_bits, bias_tweak);
    const float midpoint = lower_value + (upper_value - lower_value) * 0.5f;
    const float below_midpoint = std::nextafter(midpoint, lower_value);
    const float above_midpoint = std::nextafter(midpoint, upper_value);
    const UintType rounded_midpoint = (lower & 1) ? upper : lower;

    EXPECT_EQ(lower, convert_f32_to_small(lower_value));
    EXPECT_EQ(lower, convert_f32_to_small(below_midpoint));
    EXPECT_EQ(rounded_midpoint, convert_f32_to_small(midpoint));
    EXPECT_EQ(upper, convert_f32_to_small(above_midpoint));

    EXPECT_EQ(sign_mask | lower, convert_f32_to_small(-lower_value));
    EXPECT_EQ(sign_mask | lower, convert_f32_to_small(-below_midpoint));
    EXPECT_EQ(sign_mask | rounded_midpoint, convert_f32_to_small(-midpoint));
    EXPECT_EQ(sign_mask | upper, convert_f32_to_small(-above_midpoint));
  }
}

enum class WideningExpectationKind {
  kExactBits,
  kQuietNaN,
};

struct WideningExpectation {
  // Expected result classification.
  WideningExpectationKind kind;
  // Exact f32 payload when |kind| is |kExactBits|.
  uint32_t f32_bits;
  // Exact f64 payload when |kind| is |kExactBits|.
  uint64_t f64_bits;
};

enum NarrowFloatFormatFlagBits : uint32_t {
  kNarrowFloatHasInfinity = 1u << 0,
  kNarrowFloatHasNaN = 1u << 1,
  kNarrowFloatNaNAsNegativeZero = 1u << 2,
};
using NarrowFloatFormatFlags = uint32_t;

struct NarrowFloatFormat {
  // Name included in per-payload failure traces.
  const char* name;
  // Number of encoded exponent bits.
  int exponent_bits;
  // Number of encoded mantissa bits.
  int mantissa_bits;
  // Adjustment to the conventional exponent bias.
  int bias_tweak;
  // Special encoding policies for the format.
  NarrowFloatFormatFlags flags;
};

static WideningExpectation ExactWideningExpectation(uint32_t f32_bits) {
  float f32_value = 0.0f;
  std::memcpy(&f32_value, &f32_bits, sizeof(f32_value));
  const double f64_value = static_cast<double>(f32_value);
  uint64_t f64_bits = 0;
  std::memcpy(&f64_bits, &f64_value, sizeof(f64_bits));
  return {WideningExpectationKind::kExactBits, f32_bits, f64_bits};
}

static std::vector<WideningExpectation> BuildWideningExpectations(
    NarrowFloatFormat format) {
  const uint32_t sign_mask = 1u
                             << (format.exponent_bits + format.mantissa_bits);
  const uint32_t mantissa_mask = (1u << format.mantissa_bits) - 1;
  const uint32_t exponent_mask = ((1u << format.exponent_bits) - 1)
                                 << format.mantissa_bits;
  std::vector<WideningExpectation> expectations(sign_mask << 1);
  for (uint32_t source = 0; source < expectations.size(); ++source) {
    const uint32_t exponent = source & exponent_mask;
    const uint32_t mantissa = source & mantissa_mask;
    bool is_nan = false;
    bool is_infinity = false;
    if (exponent == exponent_mask) {
      if (format.flags & kNarrowFloatHasInfinity) {
        is_nan = (format.flags & kNarrowFloatHasNaN) && mantissa != 0;
        is_infinity = !is_nan;
      } else if ((format.flags & kNarrowFloatHasNaN) &&
                 !(format.flags & kNarrowFloatNaNAsNegativeZero)) {
        is_nan = mantissa == mantissa_mask;
      }
    }
    if ((format.flags & kNarrowFloatNaNAsNegativeZero) && source == sign_mask) {
      is_nan = true;
    }

    WideningExpectation& expectation = expectations[source];
    if (is_nan) {
      expectation.kind = WideningExpectationKind::kQuietNaN;
    } else {
      uint32_t f32_bits = 0;
      if (is_infinity) {
        f32_bits = UINT32_C(0x7F800000);
      } else {
        const float value =
            DecodePositiveFinite(source & (sign_mask - 1), format.exponent_bits,
                                 format.mantissa_bits, format.bias_tweak);
        std::memcpy(&f32_bits, &value, sizeof(f32_bits));
      }
      if (source & sign_mask) f32_bits |= UINT32_C(0x80000000);
      expectation = ExactWideningExpectation(f32_bits);
    }
  }
  return expectations;
}

template <typename UintType>
static void CheckWideningExpectations(
    const char* format_name,
    const std::vector<WideningExpectation>& expectations,
    std::function<float(UintType)> convert_small_to_f32,
    std::function<double(UintType)> convert_small_to_f64) {
  const auto check_expectations = [&]() {
    for (uint32_t source = 0; source < expectations.size(); ++source) {
      SCOPED_TRACE(::testing::Message()
                   << format_name << " source payload " << source);
      const float f32_value =
          convert_small_to_f32(static_cast<UintType>(source));
      uint32_t actual_f32_bits = 0;
      std::memcpy(&actual_f32_bits, &f32_value, sizeof(actual_f32_bits));
      const double f64_value =
          convert_small_to_f64(static_cast<UintType>(source));
      uint64_t actual_f64_bits = 0;
      std::memcpy(&actual_f64_bits, &f64_value, sizeof(actual_f64_bits));
      const WideningExpectation& expectation = expectations[source];
      if (expectation.kind == WideningExpectationKind::kQuietNaN) {
        ASSERT_EQ(UINT32_C(0x7F800000), actual_f32_bits & UINT32_C(0x7F800000));
        ASSERT_NE(0u, actual_f32_bits & UINT32_C(0x007FFFFF));
        ASSERT_NE(0u, actual_f32_bits & UINT32_C(0x00400000));
        ASSERT_EQ(UINT64_C(0x7FF0000000000000),
                  actual_f64_bits & UINT64_C(0x7FF0000000000000));
        ASSERT_NE(UINT64_C(0), actual_f64_bits & UINT64_C(0x000FFFFFFFFFFFFF));
        ASSERT_NE(UINT64_C(0), actual_f64_bits & UINT64_C(0x0008000000000000));
      } else {
        ASSERT_EQ(expectation.f32_bits, actual_f32_bits);
        ASSERT_EQ(expectation.f64_bits, actual_f64_bits);
      }
    }
  };

  check_expectations();
  const iree_fpu_state_t fpu_state =
      iree_fpu_state_push(IREE_FPU_STATE_FLAG_FLUSH_DENORMALS_TO_ZERO);
  check_expectations();
  iree_fpu_state_pop(fpu_state);
}

template <typename UintType>
static void CheckWideningPayloads(
    NarrowFloatFormat format,
    std::function<float(UintType)> convert_small_to_f32,
    std::function<double(UintType)> convert_small_to_f64) {
  CheckWideningExpectations(format.name, BuildWideningExpectations(format),
                            convert_small_to_f32, convert_small_to_f64);
}

TEST(FloatWideningTest, EveryPayloadIsExactUnderFlushToZero) {
  CheckWideningPayloads<uint16_t>(
      {"f16", 5, 10, 0, kNarrowFloatHasInfinity | kNarrowFloatHasNaN},
      iree_math_f16_to_f32, iree_math_f16_to_f64);
  CheckWideningPayloads<uint16_t>(
      {"bf16", 8, 7, 0, kNarrowFloatHasInfinity | kNarrowFloatHasNaN},
      iree_math_bf16_to_f32, iree_math_bf16_to_f64);
  CheckWideningPayloads<uint8_t>(
      {"f8e5m2", 5, 2, 0, kNarrowFloatHasInfinity | kNarrowFloatHasNaN},
      iree_math_f8e5m2_to_f32, iree_math_f8e5m2_to_f64);
  CheckWideningPayloads<uint8_t>({"f8e4m3fn", 4, 3, 0, kNarrowFloatHasNaN},
                                 iree_math_f8e4m3fn_to_f32,
                                 iree_math_f8e4m3fn_to_f64);
  CheckWideningPayloads<uint8_t>(
      {"f8e5m2fnuz", 5, 2, 1,
       kNarrowFloatHasNaN | kNarrowFloatNaNAsNegativeZero},
      iree_math_f8e5m2fnuz_to_f32, iree_math_f8e5m2fnuz_to_f64);
  CheckWideningPayloads<uint8_t>(
      {"f8e4m3fnuz", 4, 3, 1,
       kNarrowFloatHasNaN | kNarrowFloatNaNAsNegativeZero},
      iree_math_f8e4m3fnuz_to_f32, iree_math_f8e4m3fnuz_to_f64);
  CheckWideningPayloads<uint8_t>({"f6e3m2fn", 3, 2, 0, 0},
                                 iree_math_f6e3m2fn_to_f32,
                                 iree_math_f6e3m2fn_to_f64);
  CheckWideningPayloads<uint8_t>({"f6e2m3fn", 2, 3, 0, 0},
                                 iree_math_f6e2m3fn_to_f32,
                                 iree_math_f6e2m3fn_to_f64);
  CheckWideningPayloads<uint8_t>({"f4e2m1fn", 2, 1, 0, 0},
                                 iree_math_f4e2m1fn_to_f32,
                                 iree_math_f4e2m1fn_to_f64);
}

TEST(FloatWideningTest, EveryE8M0PayloadIsExactUnderFlushToZero) {
  std::vector<WideningExpectation> expectations(256);
  for (uint32_t source = 0; source < expectations.size(); ++source) {
    if (source == 0xFF) {
      expectations[source].kind = WideningExpectationKind::kQuietNaN;
    } else {
      const uint32_t f32_bits =
          source == 0 ? UINT32_C(0x00400000) : source << 23;
      expectations[source] = ExactWideningExpectation(f32_bits);
    }
  }
  CheckWideningExpectations<uint8_t>("f8e8m0fnu", expectations,
                                     iree_math_f8e8m0fnu_to_f32,
                                     iree_math_f8e8m0fnu_to_f64);
}

TEST(F16ConversionTest, Denormals) {
  CheckDenormals<uint16_t>(5, 10, /*bias_tweak=*/0, /*have_neg_zero=*/true,
                           iree_math_f32_to_f16, iree_math_f16_to_f32);
}

TEST(F16ConversionTest, FiniteRoundingBoundaries) {
  CheckFiniteRoundingBoundaries<uint16_t>(
      5, 10, /*bias_tweak=*/0, /*max_finite=*/0x7BFF, iree_math_f32_to_f16);
}

TEST(F16ConversionTest, F32ToF16ToF32) {
  constexpr float kF16Max = 65504.f;
  constexpr float kF16Min = 1.f / 16384.f;
  // Within range, should just round.
  EXPECT_EQ(0.f, iree_math_f16_to_f32(iree_math_f32_to_f16(0.f)));
  EXPECT_EQ(-0.f, iree_math_f16_to_f32(iree_math_f32_to_f16(-0.f)));
  EXPECT_EQ(0.25f, iree_math_f16_to_f32(iree_math_f32_to_f16(0.25f)));
  EXPECT_EQ(-0.25f, iree_math_f16_to_f32(iree_math_f32_to_f16(-0.25f)));
  EXPECT_EQ(100.375f, iree_math_f16_to_f32(iree_math_f32_to_f16(100.375f)));
  EXPECT_EQ(-100.375f, iree_math_f16_to_f32(iree_math_f32_to_f16(-100.375f)));
  EXPECT_EQ(100.375f, iree_math_f16_to_f32(iree_math_f32_to_f16(100.4f)));
  EXPECT_EQ(-100.375f, iree_math_f16_to_f32(iree_math_f32_to_f16(-100.4f)));
  EXPECT_EQ(kF16Max, iree_math_f16_to_f32(iree_math_f32_to_f16(kF16Max)));
  EXPECT_EQ(-kF16Max, iree_math_f16_to_f32(iree_math_f32_to_f16(-kF16Max)));
  EXPECT_EQ(kF16Min, iree_math_f16_to_f32(iree_math_f32_to_f16(kF16Min)));
  EXPECT_EQ(-kF16Min, iree_math_f16_to_f32(iree_math_f32_to_f16(-kF16Min)));
  // Powers of two should always be exactly representable across the
  // exponent range.
  EXPECT_EQ(32768.f, iree_math_f16_to_f32(iree_math_f32_to_f16(32768.f)));
  EXPECT_EQ(-32768.f, iree_math_f16_to_f32(iree_math_f32_to_f16(-32768.f)));
  // Other integers should be exactly representable up to 2048 thanks to the
  // 10-bit mantissa. The rounding mode should be nearest-even. With the 10-bit
  // mantissa, rounding half-integers just below 2048 should be literally to the
  // nearest even integer.
  //
  // Note: the case of 2047.5 is particularly important to test, because as it
  // gets rounded to 2048, that rounding involves an increment of the exponent,
  // so there is some code in the software implementation that is only exercised
  // by this case.
  EXPECT_EQ(2046.0f, iree_math_f16_to_f32(iree_math_f32_to_f16(2046.0f)));
  EXPECT_EQ(2046.0f, iree_math_f16_to_f32(iree_math_f32_to_f16(2046.5f)));
  EXPECT_EQ(2047.0f, iree_math_f16_to_f32(iree_math_f32_to_f16(2047.0f)));
  EXPECT_EQ(2048.0f, iree_math_f16_to_f32(iree_math_f32_to_f16(2047.5f)));
  EXPECT_EQ(2048.0f, iree_math_f16_to_f32(iree_math_f32_to_f16(2048.0f)));
  EXPECT_EQ(-2046.0f, iree_math_f16_to_f32(iree_math_f32_to_f16(-2046.0f)));
  EXPECT_EQ(-2046.0f, iree_math_f16_to_f32(iree_math_f32_to_f16(-2046.5f)));
  EXPECT_EQ(-2047.0f, iree_math_f16_to_f32(iree_math_f32_to_f16(-2047.0f)));
  EXPECT_EQ(-2048.0f, iree_math_f16_to_f32(iree_math_f32_to_f16(-2047.5f)));
  EXPECT_EQ(-2048.0f, iree_math_f16_to_f32(iree_math_f32_to_f16(-2048.0f)));
  // Overflow
  EXPECT_EQ(INFINITY, iree_math_f16_to_f32(iree_math_f32_to_f16(FLT_MAX)));
  EXPECT_EQ(-INFINITY, iree_math_f16_to_f32(iree_math_f32_to_f16(-FLT_MAX)));
  EXPECT_GT(kF16Max + 1.f,
            iree_math_f16_to_f32(iree_math_f32_to_f16(kF16Max + 1.f)));
  // Underflow
  EXPECT_EQ(0.0f, iree_math_f16_to_f32(iree_math_f32_to_f16(FLT_MIN)));
  EXPECT_EQ(0.0f, iree_math_f16_to_f32(iree_math_f32_to_f16(-FLT_MIN)));
  // Inf and Nan
  EXPECT_EQ(INFINITY, iree_math_f16_to_f32(iree_math_f32_to_f16(INFINITY)));
  EXPECT_EQ(-INFINITY, iree_math_f16_to_f32(iree_math_f32_to_f16(-INFINITY)));
  // Check that the result is a Nan with nan != nan.
  float nan = iree_math_f16_to_f32(iree_math_f32_to_f16(NAN));
  EXPECT_NE(nan, nan);
}

//==============================================================================
// Bfloat16 support
//==============================================================================

TEST(BF16ConversionTest, F32ToBF16) {
  // Within range, normal truncation.
  EXPECT_EQ(0x0000, iree_math_f32_to_bf16(0.f));
  EXPECT_EQ(0x8000, iree_math_f32_to_bf16(-0.f));
  EXPECT_EQ(0x3e80, iree_math_f32_to_bf16(0.25f));
  EXPECT_EQ(0xc2c9, iree_math_f32_to_bf16(-100.375f));
  // Infinity
  EXPECT_EQ(0x7f80, iree_math_f32_to_bf16(INFINITY));
  EXPECT_EQ(0xff80, iree_math_f32_to_bf16(-INFINITY));
  // No overflow or underflow, just rounding, as bfloat16 has nearly the same
  // range as float32.
  EXPECT_EQ(0x7f80, iree_math_f32_to_bf16(FLT_MAX));
  EXPECT_EQ(0xff80, iree_math_f32_to_bf16(-FLT_MAX));
  EXPECT_EQ(0x0080, iree_math_f32_to_bf16(FLT_MIN));
  EXPECT_EQ(0x8080, iree_math_f32_to_bf16(-FLT_MIN));
  // Test some round-to-nearest-even. F32->BF16 is interesting because F32
  // denormals can round to nonzero BF16 denormals.
  EXPECT_EQ(0x0000, iree_math_f32_to_bf16(FLT_MIN * 1.0f / 256.f));
  EXPECT_EQ(0x0001, iree_math_f32_to_bf16(FLT_MIN * 2.0f / 256.f));
  EXPECT_EQ(0x0002, iree_math_f32_to_bf16(FLT_MIN * 3.0f / 256.f));
  EXPECT_EQ(0x0002, iree_math_f32_to_bf16(FLT_MIN * 4.0f / 256.f));
  EXPECT_EQ(0x0002, iree_math_f32_to_bf16(FLT_MIN * 5.0f / 256.f));
  EXPECT_EQ(0x0003, iree_math_f32_to_bf16(FLT_MIN * 6.0f / 256.f));
  EXPECT_EQ(0x0004, iree_math_f32_to_bf16(FLT_MIN * 7.0f / 256.f));
  EXPECT_EQ(0x8000, iree_math_f32_to_bf16(FLT_MIN * -1.0f / 256.f));
  EXPECT_EQ(0x8001, iree_math_f32_to_bf16(FLT_MIN * -2.0f / 256.f));
  EXPECT_EQ(0x8002, iree_math_f32_to_bf16(FLT_MIN * -3.0f / 256.f));
  EXPECT_EQ(0x8002, iree_math_f32_to_bf16(FLT_MIN * -4.0f / 256.f));
  EXPECT_EQ(0x8002, iree_math_f32_to_bf16(FLT_MIN * -5.0f / 256.f));
  EXPECT_EQ(0x8003, iree_math_f32_to_bf16(FLT_MIN * -6.0f / 256.f));
  EXPECT_EQ(0x8004, iree_math_f32_to_bf16(FLT_MIN * -7.0f / 256.f));
}

TEST(BF16ConversionTest, Denormals) {
  CheckDenormals<uint16_t>(8, 7, /*bias_tweak=*/0, /*have_neg_zero=*/true,
                           iree_math_f32_to_bf16, iree_math_bf16_to_f32);
}

TEST(BF16ConversionTest, F32ToBF16ToF32) {
  // Within range, should just round.
  EXPECT_EQ(0.f, iree_math_bf16_to_f32(iree_math_f32_to_bf16(0.f)));
  EXPECT_EQ(-0.f, iree_math_bf16_to_f32(iree_math_f32_to_bf16(-0.f)));
  EXPECT_EQ(0.25f, iree_math_bf16_to_f32(iree_math_f32_to_bf16(0.25f)));
  EXPECT_EQ(-0.25f, iree_math_bf16_to_f32(iree_math_f32_to_bf16(-0.25f)));
  EXPECT_EQ(100.5f, iree_math_bf16_to_f32(iree_math_f32_to_bf16(100.375f)));
  EXPECT_EQ(-100.5f, iree_math_bf16_to_f32(iree_math_f32_to_bf16(-100.375f)));
  // Powers of two should always be exactly representable across the
  // exponent range.
  EXPECT_EQ(16777216.f,
            iree_math_bf16_to_f32(iree_math_f32_to_bf16(16777216.f)));
  EXPECT_EQ(-16777216.f,
            iree_math_bf16_to_f32(iree_math_f32_to_bf16(-16777216.f)));
  // Other integers should be exactly representable up to 256 thanks to the
  // 7-bit mantissa. The rounding mode should be nearest-even. With the 7-bit
  // mantissa, rounding half-integers just below 256 should be literally to the
  // nearest even integer.
  //
  // Note: the case of 255.5 is particularly important to test, because as it
  // gets rounded to 256, that rounding involves an increment of the exponent,
  // so there is some code in the software implementation that is only exercised
  // by this case.
  EXPECT_EQ(254.0f, iree_math_bf16_to_f32(iree_math_f32_to_bf16(254.0f)));
  EXPECT_EQ(254.0f, iree_math_bf16_to_f32(iree_math_f32_to_bf16(254.5f)));
  EXPECT_EQ(255.0f, iree_math_bf16_to_f32(iree_math_f32_to_bf16(255.0f)));
  EXPECT_EQ(256.0f, iree_math_bf16_to_f32(iree_math_f32_to_bf16(255.5f)));
  EXPECT_EQ(256.0f, iree_math_bf16_to_f32(iree_math_f32_to_bf16(256.0f)));
  EXPECT_EQ(-254.0f, iree_math_bf16_to_f32(iree_math_f32_to_bf16(-254.0f)));
  EXPECT_EQ(-254.0f, iree_math_bf16_to_f32(iree_math_f32_to_bf16(-254.5f)));
  EXPECT_EQ(-255.0f, iree_math_bf16_to_f32(iree_math_f32_to_bf16(-255.0f)));
  EXPECT_EQ(-256.0f, iree_math_bf16_to_f32(iree_math_f32_to_bf16(-255.5f)));
  EXPECT_EQ(-256.0f, iree_math_bf16_to_f32(iree_math_f32_to_bf16(-256.0f)));
  // Large finite values may round to infinity.
  EXPECT_EQ(INFINITY, iree_math_bf16_to_f32(iree_math_f32_to_bf16(FLT_MAX)));
  // Smallest normal values.
  EXPECT_EQ(FLT_MIN, iree_math_bf16_to_f32(iree_math_f32_to_bf16(FLT_MIN)));
  EXPECT_EQ(-FLT_MIN, iree_math_bf16_to_f32(iree_math_f32_to_bf16(-FLT_MIN)));

  // Inf and Nan
  EXPECT_EQ(INFINITY, iree_math_bf16_to_f32(iree_math_f32_to_bf16(INFINITY)));
  EXPECT_EQ(-INFINITY, iree_math_bf16_to_f32(iree_math_f32_to_bf16(-INFINITY)));
  // Check that the result is a Nan with nan != nan.
  float nan = iree_math_bf16_to_f32(iree_math_f32_to_bf16(NAN));
  EXPECT_NE(nan, nan);
}

//==============================================================================
// F8E5M2 support
//==============================================================================

TEST(F8E5M2ConversionTest, F32ToF8E5M2) {
  // See https://arxiv.org/pdf/2209.05433.pdf, Table 1.
  constexpr float kF8E5M2Max = 57344.f;
  constexpr float kF8E5M2Min = 1.f / 16384.f;
  // Within range, normal truncation.
  EXPECT_EQ(0x00, iree_math_f32_to_f8e5m2(0.f));
  EXPECT_EQ(0x80, iree_math_f32_to_f8e5m2(-0.f));
  EXPECT_EQ(0x34, iree_math_f32_to_f8e5m2(0.25f));
  EXPECT_EQ(0xd6, iree_math_f32_to_f8e5m2(-100.375f));
  EXPECT_EQ(0x7A, iree_math_f32_to_f8e5m2(49152.f));
  EXPECT_EQ(0xFA, iree_math_f32_to_f8e5m2(-49152.f));
  EXPECT_EQ(0x7B, iree_math_f32_to_f8e5m2(kF8E5M2Max));
  EXPECT_EQ(0xFB, iree_math_f32_to_f8e5m2(-kF8E5M2Max));
  EXPECT_EQ(0x04, iree_math_f32_to_f8e5m2(kF8E5M2Min));
  EXPECT_EQ(0x84, iree_math_f32_to_f8e5m2(-kF8E5M2Min));
  // Infinity
  EXPECT_EQ(0x7c, iree_math_f32_to_f8e5m2(INFINITY));
  EXPECT_EQ(0xfc, iree_math_f32_to_f8e5m2(-INFINITY));
  // Overflow
  EXPECT_EQ(0x7C, iree_math_f32_to_f8e5m2(FLT_MAX));
  EXPECT_EQ(0xFC, iree_math_f32_to_f8e5m2(-FLT_MAX));
  // Important case to test: overflow due to rounding to nearest-even of 61440
  // to 65536.
  EXPECT_EQ(0x7B, iree_math_f32_to_f8e5m2(61439.f));
  EXPECT_EQ(0xFB, iree_math_f32_to_f8e5m2(-61439.f));
  EXPECT_EQ(0x7C, iree_math_f32_to_f8e5m2(61440.f));
  EXPECT_EQ(0xFC, iree_math_f32_to_f8e5m2(-61440.f));
  EXPECT_EQ(0x7C, iree_math_f32_to_f8e5m2(65536.f));
  EXPECT_EQ(0xFC, iree_math_f32_to_f8e5m2(-65536.f));
  // Underflow
  EXPECT_EQ(0, iree_math_f32_to_f8e5m2(FLT_MIN));
  EXPECT_EQ(0x80, iree_math_f32_to_f8e5m2(-FLT_MIN));
  EXPECT_EQ(0x02, iree_math_f32_to_f8e5m2(kF8E5M2Min * 0.5f));
  EXPECT_EQ(0x82, iree_math_f32_to_f8e5m2(-kF8E5M2Min * 0.5f));
  EXPECT_EQ(0x03, iree_math_f32_to_f8e5m2(kF8E5M2Min * 0.75f));
  EXPECT_EQ(0x83, iree_math_f32_to_f8e5m2(-kF8E5M2Min * 0.75f));
}

TEST(F8E5M2ConversionTest, Denormals) {
  CheckDenormals<uint8_t>(5, 2, /*bias_tweak=*/0, /*have_neg_zero=*/true,
                          iree_math_f32_to_f8e5m2, iree_math_f8e5m2_to_f32);
}

TEST(F8E5M2ConversionTest, FiniteRoundingBoundaries) {
  CheckFiniteRoundingBoundaries<uint8_t>(
      5, 2, /*bias_tweak=*/0, /*max_finite=*/0x7B, iree_math_f32_to_f8e5m2);
}

TEST(F8E5M2ConversionTest, F32ToF8E5M2ToF32) {
  // See https://arxiv.org/pdf/2209.05433.pdf, Table 1.
  constexpr float kF8E5M2Max = 57344.f;
  constexpr float kF8E5M2Min = 1.f / 16384.f;
  // Within range, should just round.
  EXPECT_EQ(0.f, iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(0.)));
  EXPECT_EQ(-0.f, iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(-0.)));
  EXPECT_EQ(0.25f, iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(0.25f)));
  EXPECT_EQ(-0.25f, iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(-0.25f)));
  EXPECT_EQ(96.f, iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(100.375f)));
  EXPECT_EQ(-96.f, iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(-100.375f)));
  EXPECT_EQ(96.f, iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(96.f)));
  EXPECT_EQ(-96.f, iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(-96.f)));
  EXPECT_EQ(kF8E5M2Max,
            iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(kF8E5M2Max)));
  EXPECT_EQ(-kF8E5M2Max,
            iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(-kF8E5M2Max)));
  EXPECT_EQ(kF8E5M2Min,
            iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(kF8E5M2Min)));
  EXPECT_EQ(-kF8E5M2Min,
            iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(-kF8E5M2Min)));
  // Powers of two should always be exactly representable across the
  // exponent range.
  EXPECT_EQ(32768.f, iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(32768.f)));
  EXPECT_EQ(-32768.f,
            iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(-32768.f)));
  // Overflow
  EXPECT_EQ(INFINITY,
            iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(FLT_MAX)));
  EXPECT_EQ(-INFINITY,
            iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(-FLT_MAX)));
  EXPECT_GT(kF8E5M2Max + 1.f,
            iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(kF8E5M2Max + 1.f)));
  // Underflow
  EXPECT_EQ(0.0f, iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(FLT_MIN)));
  EXPECT_EQ(0.0f, iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(-FLT_MIN)));
  // Inf and Nan
  EXPECT_EQ(INFINITY,
            iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(INFINITY)));
  EXPECT_EQ(-INFINITY,
            iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(-INFINITY)));
  // Check that the result is a Nan with nan != nan.
  float nan = iree_math_f8e5m2_to_f32(iree_math_f32_to_f8e5m2(NAN));
  EXPECT_NE(nan, nan);
}

//==============================================================================
// F8E4M3FN support
//==============================================================================

TEST(F8E4M3FNConversionTest, F32ToF8E4M3FN) {
  // See https://arxiv.org/pdf/2209.05433.pdf, Table 1.
  // The F8E4M3FN format is special: it has no infinities, and has some larger
  // finite values instead. The paper uses F8E4M3 as the label, unlike APFloat
  // and MLIR.
  constexpr float kF8E4M3FNMax = 448.f;
  constexpr float kF8E4M3FNMin = 1.f / 64.f;
  // Within range, normal truncation.
  EXPECT_EQ(0x00, iree_math_f32_to_f8e4m3fn(0.f));
  EXPECT_EQ(0x80, iree_math_f32_to_f8e4m3fn(-0.f));
  EXPECT_EQ(0x28, iree_math_f32_to_f8e4m3fn(0.25f));
  EXPECT_EQ(0xED, iree_math_f32_to_f8e4m3fn(-100.375f));
  // Extra large finite values thanks to not having infinities.
  EXPECT_EQ(0x78, iree_math_f32_to_f8e4m3fn(256.0f));
  EXPECT_EQ(0x79, iree_math_f32_to_f8e4m3fn(288.0f));
  EXPECT_EQ(0x7A, iree_math_f32_to_f8e4m3fn(320.0f));
  EXPECT_EQ(0x7B, iree_math_f32_to_f8e4m3fn(352.0f));
  EXPECT_EQ(0x7C, iree_math_f32_to_f8e4m3fn(384.0f));
  EXPECT_EQ(0x7D, iree_math_f32_to_f8e4m3fn(416.0f));
  EXPECT_EQ(0x7E, iree_math_f32_to_f8e4m3fn(kF8E4M3FNMax));
  EXPECT_EQ(0xFE, iree_math_f32_to_f8e4m3fn(-kF8E4M3FNMax));
  // Min normal values.
  EXPECT_EQ(0x08, iree_math_f32_to_f8e4m3fn(kF8E4M3FNMin));
  EXPECT_EQ(0x88, iree_math_f32_to_f8e4m3fn(-kF8E4M3FNMin));
  // Test some round-to-nearest-even behavior.
  EXPECT_EQ(0x70, iree_math_f32_to_f8e4m3fn(136.0f));
  EXPECT_EQ(0x72, iree_math_f32_to_f8e4m3fn(152.0f));
  EXPECT_EQ(0x72, iree_math_f32_to_f8e4m3fn(168.0f));
  EXPECT_EQ(0x74, iree_math_f32_to_f8e4m3fn(184.0f));
  EXPECT_EQ(0x78, iree_math_f32_to_f8e4m3fn(272.0f));
  EXPECT_EQ(0x7A, iree_math_f32_to_f8e4m3fn(304.0f));
  EXPECT_EQ(0x7A, iree_math_f32_to_f8e4m3fn(336.0f));
  EXPECT_EQ(0x7C, iree_math_f32_to_f8e4m3fn(368.0f));
  // Test round-to-nearest-even for denormals.
  EXPECT_EQ(0x00, iree_math_f32_to_f8e4m3fn(0.5f / 512.f));
  EXPECT_EQ(0x01, iree_math_f32_to_f8e4m3fn(1.f / 512.f));
  EXPECT_EQ(0x02, iree_math_f32_to_f8e4m3fn(1.5f / 512.f));
  EXPECT_EQ(0x02, iree_math_f32_to_f8e4m3fn(2.f / 512.f));
  EXPECT_EQ(0x02, iree_math_f32_to_f8e4m3fn(2.5f / 512.f));
  EXPECT_EQ(0x03, iree_math_f32_to_f8e4m3fn(3.f / 512.f));
  EXPECT_EQ(0x04, iree_math_f32_to_f8e4m3fn(3.5f / 512.f));
  EXPECT_EQ(0x80, iree_math_f32_to_f8e4m3fn(-0.5f / 512.f));
  EXPECT_EQ(0x81, iree_math_f32_to_f8e4m3fn(-1.f / 512.f));
  EXPECT_EQ(0x82, iree_math_f32_to_f8e4m3fn(-1.5f / 512.f));
  EXPECT_EQ(0x82, iree_math_f32_to_f8e4m3fn(-2.f / 512.f));
  EXPECT_EQ(0x82, iree_math_f32_to_f8e4m3fn(-2.5f / 512.f));
  EXPECT_EQ(0x83, iree_math_f32_to_f8e4m3fn(-3.f / 512.f));
  EXPECT_EQ(0x84, iree_math_f32_to_f8e4m3fn(-3.5f / 512.f));
  // Underflow
  EXPECT_EQ(0, iree_math_f32_to_f8e4m3fn(FLT_MIN));
  EXPECT_EQ(0x80, iree_math_f32_to_f8e4m3fn(-FLT_MIN));
}

TEST(F8E4M3FNConversionTest, SaturatesOverflowAndPreservesNan) {
  constexpr float kMaxFinite = 448.0f;
  constexpr float kOverflowMidpoint = 464.0f;
  const float just_above_max = std::nextafter(kMaxFinite, INFINITY);
  const float below_midpoint = std::nextafter(kOverflowMidpoint, kMaxFinite);
  const float above_midpoint = std::nextafter(kOverflowMidpoint, INFINITY);
  auto expect_saturated = [](float value) {
    EXPECT_EQ(0x7E, iree_math_f32_to_f8e4m3fn(value));
    EXPECT_EQ(0xFE, iree_math_f32_to_f8e4m3fn(-value));
  };

  expect_saturated(just_above_max);
  expect_saturated(below_midpoint);
  expect_saturated(kOverflowMidpoint);
  expect_saturated(above_midpoint);
  expect_saturated(465.0f);
  expect_saturated(511.0f);
  expect_saturated(FLT_MAX);
  expect_saturated(INFINITY);

  auto f32_from_bits = [](uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  };
  EXPECT_EQ(0x7F,
            iree_math_f32_to_f8e4m3fn(f32_from_bits(UINT32_C(0x7FC00000))));
  EXPECT_EQ(0x7F,
            iree_math_f32_to_f8e4m3fn(f32_from_bits(UINT32_C(0x7F800001))));
  EXPECT_EQ(0xFF,
            iree_math_f32_to_f8e4m3fn(f32_from_bits(UINT32_C(0xFFC00000))));
  EXPECT_EQ(0xFF,
            iree_math_f32_to_f8e4m3fn(f32_from_bits(UINT32_C(0xFF800001))));
}

TEST(F8E4M3FNConversionTest, Denormals) {
  CheckDenormals<uint8_t>(4, 3, /*bias_tweak=*/0, /*have_neg_zero=*/true,
                          iree_math_f32_to_f8e4m3fn, iree_math_f8e4m3fn_to_f32);
}

TEST(F8E4M3FNConversionTest, FiniteRoundingBoundaries) {
  CheckFiniteRoundingBoundaries<uint8_t>(
      4, 3, /*bias_tweak=*/0, /*max_finite=*/0x7E, iree_math_f32_to_f8e4m3fn);
}

TEST(F8E4M3FNConversionTest, F32ToF8E4M3FNToF32) {
  // See https://arxiv.org/pdf/2209.05433.pdf, Table 1.
  // The F8E4M3FN format is special: it has no infinities, and has some larger
  // finite values instead.
  constexpr float kF8E4M3FNMax = 448.f;
  constexpr float kF8E4M3FNMin = 1.f / 64.f;
  // Within range, should just round.
  EXPECT_EQ(0.f, iree_math_f8e4m3fn_to_f32(iree_math_f32_to_f8e4m3fn(0.f)));
  EXPECT_EQ(-0.f, iree_math_f8e4m3fn_to_f32(iree_math_f32_to_f8e4m3fn(-0.f)));
  EXPECT_EQ(0.25f, iree_math_f8e4m3fn_to_f32(iree_math_f32_to_f8e4m3fn(0.25f)));
  EXPECT_EQ(-0.25f,
            iree_math_f8e4m3fn_to_f32(iree_math_f32_to_f8e4m3fn(-0.25f)));
  EXPECT_EQ(104.f,
            iree_math_f8e4m3fn_to_f32(iree_math_f32_to_f8e4m3fn(100.375f)));
  EXPECT_EQ(-104.f,
            iree_math_f8e4m3fn_to_f32(iree_math_f32_to_f8e4m3fn(-100.375f)));
  EXPECT_EQ(104.f,
            iree_math_f8e4m3fn_to_f32(iree_math_f32_to_f8e4m3fn(100.4f)));
  EXPECT_EQ(-104.f,
            iree_math_f8e4m3fn_to_f32(iree_math_f32_to_f8e4m3fn(-100.4f)));
  EXPECT_EQ(kF8E4M3FNMax,
            iree_math_f8e4m3fn_to_f32(iree_math_f32_to_f8e4m3fn(kF8E4M3FNMax)));
  EXPECT_EQ(-kF8E4M3FNMax, iree_math_f8e4m3fn_to_f32(
                               iree_math_f32_to_f8e4m3fn(-kF8E4M3FNMax)));
  EXPECT_EQ(kF8E4M3FNMin,
            iree_math_f8e4m3fn_to_f32(iree_math_f32_to_f8e4m3fn(kF8E4M3FNMin)));
  EXPECT_EQ(-kF8E4M3FNMin, iree_math_f8e4m3fn_to_f32(
                               iree_math_f32_to_f8e4m3fn(-kF8E4M3FNMin)));
  // Powers of two should always be exactly representable across the
  // exponent range.
  EXPECT_EQ(256.f, iree_math_f8e4m3fn_to_f32(iree_math_f32_to_f8e4m3fn(256.f)));
  EXPECT_EQ(-256.f,
            iree_math_f8e4m3fn_to_f32(iree_math_f32_to_f8e4m3fn(-256.f)));
  // Overflow saturates to signed maximum finite.
  EXPECT_EQ(kF8E4M3FNMax,
            iree_math_f8e4m3fn_to_f32(iree_math_f32_to_f8e4m3fn(FLT_MAX)));
  EXPECT_EQ(-kF8E4M3FNMax,
            iree_math_f8e4m3fn_to_f32(iree_math_f32_to_f8e4m3fn(-FLT_MAX)));
  // Underflow
  EXPECT_EQ(0.0f,
            iree_math_f8e4m3fn_to_f32(iree_math_f32_to_f8e4m3fn(FLT_MIN)));
  EXPECT_EQ(0.0f,
            iree_math_f8e4m3fn_to_f32(iree_math_f32_to_f8e4m3fn(-FLT_MIN)));
  // Infinities saturate while NaNs remain NaN.
  EXPECT_EQ(kF8E4M3FNMax,
            iree_math_f8e4m3fn_to_f32(iree_math_f32_to_f8e4m3fn(INFINITY)));
  EXPECT_EQ(-kF8E4M3FNMax,
            iree_math_f8e4m3fn_to_f32(iree_math_f32_to_f8e4m3fn(-INFINITY)));
  float nan = iree_math_f8e4m3fn_to_f32(iree_math_f32_to_f8e4m3fn(NAN));
  EXPECT_NE(nan, nan);
}

//==============================================================================
// F8E5M2FNUZ support
//==============================================================================

TEST(F8E5M2FNUZConversionTest, F32ToF8E5M2FNUZ) {
  constexpr float kF8E5M2FNUZMax = 57344.f;
  constexpr float kF8E5M2FNUZMin = 1.f / 32768.f;
  // Within range, normal truncation.
  EXPECT_EQ(0x00, iree_math_f32_to_f8e5m2fnuz(0.f));
  EXPECT_EQ(0x00, iree_math_f32_to_f8e5m2fnuz(-0.f));  // No negative zero.
  EXPECT_EQ(0x38, iree_math_f32_to_f8e5m2fnuz(0.25f));
  EXPECT_EQ(0xDA, iree_math_f32_to_f8e5m2fnuz(-100.375f));
  EXPECT_EQ(0x7E, iree_math_f32_to_f8e5m2fnuz(49152.f));
  EXPECT_EQ(0xFE, iree_math_f32_to_f8e5m2fnuz(-49152.f));
  EXPECT_EQ(0x7F, iree_math_f32_to_f8e5m2fnuz(kF8E5M2FNUZMax));
  EXPECT_EQ(0xFF, iree_math_f32_to_f8e5m2fnuz(-kF8E5M2FNUZMax));
  EXPECT_EQ(0x04, iree_math_f32_to_f8e5m2fnuz(kF8E5M2FNUZMin));
  EXPECT_EQ(0x84, iree_math_f32_to_f8e5m2fnuz(-kF8E5M2FNUZMin));
  // No infinities, so they convert to NaN, encoded as negative zero.
  EXPECT_EQ(0x80, iree_math_f32_to_f8e5m2fnuz(INFINITY));
  EXPECT_EQ(0x80, iree_math_f32_to_f8e5m2fnuz(-INFINITY));
  // Overflow.
  EXPECT_EQ(0x80, iree_math_f32_to_f8e5m2fnuz(FLT_MAX));
  EXPECT_EQ(0x80, iree_math_f32_to_f8e5m2fnuz(-FLT_MAX));
  // Underflow
  EXPECT_EQ(0, iree_math_f32_to_f8e5m2fnuz(FLT_MIN));
  EXPECT_EQ(0, iree_math_f32_to_f8e5m2fnuz(-FLT_MIN));  // No negative zero.
}

TEST(F8E5M2FNUZConversionTest, Denormals) {
  CheckDenormals<uint8_t>(5, 2, /*bias_tweak=*/1, /*have_neg_zero=*/false,
                          iree_math_f32_to_f8e5m2fnuz,
                          iree_math_f8e5m2fnuz_to_f32);
}

TEST(F8E5M2FNUZConversionTest, F32ToF8E5M2ToF32FNUZ) {
  constexpr float kF8E5M2FNUZMax = 57344.f;
  constexpr float kF8E5M2FNUZMin = 1.f / 32768.f;
  // Within range, should just round.
  EXPECT_EQ(0.f, iree_math_f8e5m2fnuz_to_f32(iree_math_f32_to_f8e5m2fnuz(0.f)));
  EXPECT_EQ(-0.f,
            iree_math_f8e5m2fnuz_to_f32(iree_math_f32_to_f8e5m2fnuz(-0.f)));
  EXPECT_EQ(0.25f,
            iree_math_f8e5m2fnuz_to_f32(iree_math_f32_to_f8e5m2fnuz(0.25f)));
  EXPECT_EQ(-0.25f,
            iree_math_f8e5m2fnuz_to_f32(iree_math_f32_to_f8e5m2fnuz(-0.25f)));
  EXPECT_EQ(96.f,
            iree_math_f8e5m2fnuz_to_f32(iree_math_f32_to_f8e5m2fnuz(100.375f)));
  EXPECT_EQ(-96.f, iree_math_f8e5m2fnuz_to_f32(
                       iree_math_f32_to_f8e5m2fnuz(-100.375f)));
  EXPECT_EQ(96.f,
            iree_math_f8e5m2fnuz_to_f32(iree_math_f32_to_f8e5m2fnuz(96.f)));
  EXPECT_EQ(-96.f,
            iree_math_f8e5m2fnuz_to_f32(iree_math_f32_to_f8e5m2fnuz(-96.f)));
  EXPECT_EQ(kF8E5M2FNUZMax, iree_math_f8e5m2fnuz_to_f32(
                                iree_math_f32_to_f8e5m2fnuz(kF8E5M2FNUZMax)));
  EXPECT_EQ(-kF8E5M2FNUZMax, iree_math_f8e5m2fnuz_to_f32(
                                 iree_math_f32_to_f8e5m2fnuz(-kF8E5M2FNUZMax)));
  EXPECT_EQ(kF8E5M2FNUZMin, iree_math_f8e5m2fnuz_to_f32(
                                iree_math_f32_to_f8e5m2fnuz(kF8E5M2FNUZMin)));
  EXPECT_EQ(-kF8E5M2FNUZMin, iree_math_f8e5m2fnuz_to_f32(
                                 iree_math_f32_to_f8e5m2fnuz(-kF8E5M2FNUZMin)));
  // Powers of two should always be exactly representable across the
  // exponent range.
  EXPECT_EQ(32768.f,
            iree_math_f8e5m2fnuz_to_f32(iree_math_f32_to_f8e5m2fnuz(32768.f)));
  EXPECT_EQ(-32768.f,
            iree_math_f8e5m2fnuz_to_f32(iree_math_f32_to_f8e5m2fnuz(-32768.f)));
  // Overflow
  EXPECT_TRUE(std::isnan(
      iree_math_f8e5m2fnuz_to_f32(iree_math_f32_to_f8e5m2fnuz(FLT_MAX))));
  EXPECT_TRUE(std::isnan(
      iree_math_f8e5m2fnuz_to_f32(iree_math_f32_to_f8e5m2fnuz(-FLT_MAX))));
  EXPECT_GT(kF8E5M2FNUZMax + 1.f,
            iree_math_f8e5m2fnuz_to_f32(
                iree_math_f32_to_f8e5m2fnuz(kF8E5M2FNUZMax + 1.f)));
  // Underflow
  EXPECT_EQ(0.0f,
            iree_math_f8e5m2fnuz_to_f32(iree_math_f32_to_f8e5m2fnuz(FLT_MIN)));
  EXPECT_EQ(0.0f,
            iree_math_f8e5m2fnuz_to_f32(iree_math_f32_to_f8e5m2fnuz(-FLT_MIN)));
  // Inf and NaN. No infinities, so we get NaN.
  EXPECT_TRUE(std::isnan(
      iree_math_f8e5m2fnuz_to_f32(iree_math_f32_to_f8e5m2fnuz(INFINITY))));
  EXPECT_TRUE(std::isnan(
      iree_math_f8e5m2fnuz_to_f32(iree_math_f32_to_f8e5m2fnuz(-INFINITY))));
  EXPECT_TRUE(std::isnan(
      iree_math_f8e5m2fnuz_to_f32(iree_math_f32_to_f8e5m2fnuz(NAN))));
}

//==============================================================================
// F8E4M3FNUZ support
//==============================================================================

TEST(F8E4M3FNUZConversionTest, F32ToF8E4M3FNUZ) {
  // Found on MI-300 AMD GPUs, where it's called "FP8". See their manual or
  // APFloat. Has no infinities and used the bit pattern normally used for -0
  // (0x80) for NaN.
  constexpr float kF8E4M3FNUZMax = 240.f;
  constexpr float kF8E4M3FNUZMin = 1.f / 128.f;
  // Within range, normal truncation.
  EXPECT_EQ(0x00, iree_math_f32_to_f8e4m3fnuz(0.f));
  EXPECT_EQ(0x00, iree_math_f32_to_f8e4m3fnuz(-0.f));  // No negative zero.
  EXPECT_EQ(0x30, iree_math_f32_to_f8e4m3fnuz(0.25f));
  EXPECT_EQ(0xF5, iree_math_f32_to_f8e4m3fnuz(-100.375f));
  // Extra large finite values thanks to not having infinities.
  EXPECT_EQ(0x7F, iree_math_f32_to_f8e4m3fnuz(kF8E4M3FNUZMax));
  EXPECT_EQ(0x7F, iree_math_f32_to_f8e4m3fnuz(247.0f));
  EXPECT_EQ(0xFF, iree_math_f32_to_f8e4m3fnuz(-kF8E4M3FNUZMax));
  EXPECT_EQ(0xFF, iree_math_f32_to_f8e4m3fnuz(-247.0f));
  // First value that overflows.
  EXPECT_EQ(0x80, iree_math_f32_to_f8e4m3fnuz(248.0f));
  EXPECT_EQ(0x80, iree_math_f32_to_f8e4m3fnuz(-248.0f));
  // Min normal values.
  EXPECT_EQ(0x08, iree_math_f32_to_f8e4m3fnuz(kF8E4M3FNUZMin));
  EXPECT_EQ(0x88, iree_math_f32_to_f8e4m3fnuz(-kF8E4M3FNUZMin));
  // Infinity
  EXPECT_EQ(0x80, iree_math_f32_to_f8e4m3fnuz(INFINITY));
  EXPECT_EQ(0x80, iree_math_f32_to_f8e4m3fnuz(-INFINITY));
  // Overflow
  EXPECT_EQ(0x80, iree_math_f32_to_f8e4m3fnuz(FLT_MAX));
  EXPECT_EQ(0x80, iree_math_f32_to_f8e4m3fnuz(-FLT_MAX));
  // Test some round-to-nearest-even behavior.
  EXPECT_EQ(0x78, iree_math_f32_to_f8e4m3fnuz(136.0f));
  EXPECT_EQ(0x7A, iree_math_f32_to_f8e4m3fnuz(152.0f));
  EXPECT_EQ(0x7A, iree_math_f32_to_f8e4m3fnuz(168.0f));
  EXPECT_EQ(0x7C, iree_math_f32_to_f8e4m3fnuz(184.0f));
  // Underflow
  EXPECT_EQ(0, iree_math_f32_to_f8e4m3fnuz(FLT_MIN));
  EXPECT_EQ(0, iree_math_f32_to_f8e4m3fnuz(-FLT_MIN));
}

TEST(F8E4M3FNUZConversionTest, Denormals) {
  CheckDenormals<uint8_t>(4, 3, /*bias_tweak=*/1, /*have_neg_zero=*/false,
                          iree_math_f32_to_f8e4m3fnuz,
                          iree_math_f8e4m3fnuz_to_f32);
}

TEST(F8E4M3FNUZConversionTest, F32ToF8E4M3FNUZToF32) {
  // Found on MI-300 AMD GPUs, where it's called "FP8". See their manual or
  // APFloat. Has no infinities and used the bit pattern normally used for -0
  // (0x80) for NaN.
  constexpr float kF8E4M3FNUZMax = 240.f;
  constexpr float kF8E4M3FNUZMin = 1.f / 128.f;
  // Within range, should just round.
  EXPECT_EQ(0.f, iree_math_f8e4m3fnuz_to_f32(iree_math_f32_to_f8e4m3fnuz(0.f)));
  EXPECT_EQ(-0.f,
            iree_math_f8e4m3fnuz_to_f32(iree_math_f32_to_f8e4m3fnuz(-0.f)));
  EXPECT_EQ(0.25f,
            iree_math_f8e4m3fnuz_to_f32(iree_math_f32_to_f8e4m3fnuz(0.25f)));
  EXPECT_EQ(-0.25f,
            iree_math_f8e4m3fnuz_to_f32(iree_math_f32_to_f8e4m3fnuz(-0.25f)));
  EXPECT_EQ(104.f,
            iree_math_f8e4m3fnuz_to_f32(iree_math_f32_to_f8e4m3fnuz(100.375f)));
  EXPECT_EQ(-104.f, iree_math_f8e4m3fnuz_to_f32(
                        iree_math_f32_to_f8e4m3fnuz(-100.375f)));
  EXPECT_EQ(104.f,
            iree_math_f8e4m3fnuz_to_f32(iree_math_f32_to_f8e4m3fnuz(100.4f)));
  EXPECT_EQ(-104.f,
            iree_math_f8e4m3fnuz_to_f32(iree_math_f32_to_f8e4m3fnuz(-100.4f)));
  EXPECT_EQ(kF8E4M3FNUZMax, iree_math_f8e4m3fnuz_to_f32(
                                iree_math_f32_to_f8e4m3fnuz(kF8E4M3FNUZMax)));
  EXPECT_EQ(-kF8E4M3FNUZMax, iree_math_f8e4m3fnuz_to_f32(
                                 iree_math_f32_to_f8e4m3fnuz(-kF8E4M3FNUZMax)));
  EXPECT_EQ(kF8E4M3FNUZMin, iree_math_f8e4m3fnuz_to_f32(
                                iree_math_f32_to_f8e4m3fnuz(kF8E4M3FNUZMin)));
  EXPECT_EQ(-kF8E4M3FNUZMin, iree_math_f8e4m3fnuz_to_f32(
                                 iree_math_f32_to_f8e4m3fnuz(-kF8E4M3FNUZMin)));
  // Powers of two should always be exactly representable across the
  // exponent range.
  EXPECT_EQ(128.f,
            iree_math_f8e4m3fnuz_to_f32(iree_math_f32_to_f8e4m3fnuz(128.f)));
  EXPECT_EQ(-128.f,
            iree_math_f8e4m3fnuz_to_f32(iree_math_f32_to_f8e4m3fnuz(-128.f)));
  // Overflow
  EXPECT_TRUE(std::isnan(
      iree_math_f8e4m3fnuz_to_f32(iree_math_f32_to_f8e4m3fnuz(FLT_MAX))));
  EXPECT_TRUE(std::isnan(
      iree_math_f8e4m3fnuz_to_f32(iree_math_f32_to_f8e4m3fnuz(-FLT_MAX))));
  EXPECT_GT(kF8E4M3FNUZMax + 1.f,
            iree_math_f8e4m3fnuz_to_f32(
                iree_math_f32_to_f8e4m3fnuz(kF8E4M3FNUZMax + 1.f)));
  // Underflow
  EXPECT_EQ(0.0f,
            iree_math_f8e4m3fnuz_to_f32(iree_math_f32_to_f8e4m3fnuz(FLT_MIN)));
  EXPECT_EQ(0.0f,
            iree_math_f8e4m3fnuz_to_f32(iree_math_f32_to_f8e4m3fnuz(-FLT_MIN)));
  // Inf and Nan
  EXPECT_TRUE(std::isnan(
      iree_math_f8e4m3fnuz_to_f32(iree_math_f32_to_f8e4m3fnuz(INFINITY))));
  EXPECT_TRUE(std::isnan(
      iree_math_f8e4m3fnuz_to_f32(iree_math_f32_to_f8e4m3fnuz(-INFINITY))));
  // Check that the result is a Nan with nan != nan.
  float nan = iree_math_f8e4m3fnuz_to_f32(iree_math_f32_to_f8e4m3fnuz(NAN));
  EXPECT_NE(nan, nan);
}

//==============================================================================
// F6E3M2FN support
//==============================================================================

// See
// https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
// Paragraph 5.3.2.

TEST(F6E3M2FNConversionTest, F32ToF6E3M2FN) {
  constexpr float kF6E3M2FNMax = 28.f;
  constexpr float kF6E3M2FNMin = 0.25f;
  // Within range, normal truncation.
  EXPECT_EQ(0x00, iree_math_f32_to_f6e3m2fn(0.f));
  EXPECT_EQ(0x20, iree_math_f32_to_f6e3m2fn(-0.f));
  EXPECT_EQ(0x04, iree_math_f32_to_f6e3m2fn(0.25f));
  EXPECT_EQ(0x39, iree_math_f32_to_f6e3m2fn(-10.f));
  // Extra large finite values thanks to not having infinities.
  EXPECT_EQ(0x1F, iree_math_f32_to_f6e3m2fn(kF6E3M2FNMax));
  EXPECT_EQ(0x3F, iree_math_f32_to_f6e3m2fn(-kF6E3M2FNMax));
  // Min normal values.
  EXPECT_EQ(0x04, iree_math_f32_to_f6e3m2fn(kF6E3M2FNMin));
  EXPECT_EQ(0x24, iree_math_f32_to_f6e3m2fn(-kF6E3M2FNMin));
  // Infinity clamped to max finite.
  EXPECT_EQ(0x1F, iree_math_f32_to_f6e3m2fn(INFINITY));
  EXPECT_EQ(0x3F, iree_math_f32_to_f6e3m2fn(-INFINITY));
  // Large finite value clamped to max finite.
  EXPECT_EQ(0x1F, iree_math_f32_to_f6e3m2fn(FLT_MAX));
  EXPECT_EQ(0x3F, iree_math_f32_to_f6e3m2fn(-FLT_MAX));
  // Test some round-to-nearest-even behavior.
  EXPECT_EQ(0x18, iree_math_f32_to_f6e3m2fn(8.0f));
  EXPECT_EQ(0x18, iree_math_f32_to_f6e3m2fn(9.0f));
  EXPECT_EQ(0x19, iree_math_f32_to_f6e3m2fn(10.0f));
  EXPECT_EQ(0x1A, iree_math_f32_to_f6e3m2fn(11.0f));
  EXPECT_EQ(0x1A, iree_math_f32_to_f6e3m2fn(12.0f));
  // Underflow
  EXPECT_EQ(0, iree_math_f32_to_f6e3m2fn(FLT_MIN));
  EXPECT_EQ(0x20, iree_math_f32_to_f6e3m2fn(-FLT_MIN));
  // NaN conversion is implementation-defined. We canonicalize to +0.0.
  EXPECT_EQ(0, iree_math_f32_to_f6e3m2fn(NAN));
}

TEST(F6E3M2FNConversionTest, Denormals) {
  CheckDenormals<uint8_t>(3, 2, /*bias_tweak=*/0, /*have_neg_zero=*/true,
                          iree_math_f32_to_f6e3m2fn, iree_math_f6e3m2fn_to_f32);
}

TEST(F6E3M2FNConversionTest, F6E3M2FNToF32) {
  for (int sign_bit = 0; sign_bit <= 0x20; sign_bit += 0x20) {
    float sign = sign_bit ? -1.f : 1.f;
    // Zero
    EXPECT_EQ(sign * 0x0.0p0f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x00));
    // Denormals
    EXPECT_EQ(sign * 0x0.4p-2f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x01));
    EXPECT_EQ(sign * 0x0.8p-2f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x02));
    EXPECT_EQ(sign * 0x0.Cp-2f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x03));
    // Normal finite values
    EXPECT_EQ(sign * 0x1.0p-2f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x04));
    EXPECT_EQ(sign * 0x1.4p-2f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x05));
    EXPECT_EQ(sign * 0x1.8p-2f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x06));
    EXPECT_EQ(sign * 0x1.Cp-2f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x07));
    EXPECT_EQ(sign * 0x1.0p-1f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x08));
    EXPECT_EQ(sign * 0x1.4p-1f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x09));
    EXPECT_EQ(sign * 0x1.8p-1f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x0A));
    EXPECT_EQ(sign * 0x1.Cp-1f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x0B));
    EXPECT_EQ(sign * 0x1.0p+0f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x0C));
    EXPECT_EQ(sign * 0x1.4p+0f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x0D));
    EXPECT_EQ(sign * 0x1.8p+0f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x0E));
    EXPECT_EQ(sign * 0x1.Cp+0f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x0F));
    EXPECT_EQ(sign * 0x1.0p+1f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x10));
    EXPECT_EQ(sign * 0x1.4p+1f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x11));
    EXPECT_EQ(sign * 0x1.8p+1f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x12));
    EXPECT_EQ(sign * 0x1.Cp+1f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x13));
    EXPECT_EQ(sign * 0x1.0p+2f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x14));
    EXPECT_EQ(sign * 0x1.4p+2f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x15));
    EXPECT_EQ(sign * 0x1.8p+2f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x16));
    EXPECT_EQ(sign * 0x1.Cp+2f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x17));
    EXPECT_EQ(sign * 0x1.0p+3f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x18));
    EXPECT_EQ(sign * 0x1.4p+3f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x19));
    EXPECT_EQ(sign * 0x1.8p+3f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x1A));
    EXPECT_EQ(sign * 0x1.Cp+3f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x1B));
    // Extra finite values in the top exponent thanks to no Inf and no NaN.
    EXPECT_EQ(sign * 0x1.0p+4f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x1C));
    EXPECT_EQ(sign * 0x1.4p+4f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x1D));
    EXPECT_EQ(sign * 0x1.8p+4f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x1E));
    EXPECT_EQ(sign * 0x1.Cp+4f, iree_math_f6e3m2fn_to_f32(sign_bit | 0x1F));
  }
}

//==============================================================================
// F6E2M3FN support
//==============================================================================

// See
// https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
// Paragraph 5.3.2.

TEST(F6E2M3FNConversionTest, F32ToF6E2M3FN) {
  constexpr float kF6E2M3FNMax = 7.5f;
  constexpr float kF6E2M3FNMin = 1.0f;
  // Within range, normal truncation.
  EXPECT_EQ(0x00, iree_math_f32_to_f6e2m3fn(0.f));
  EXPECT_EQ(0x20, iree_math_f32_to_f6e2m3fn(-0.f));
  EXPECT_EQ(0x02, iree_math_f32_to_f6e2m3fn(0.25f));
  EXPECT_EQ(0x34, iree_math_f32_to_f6e2m3fn(-3.0f));
  // Extra large finite values thanks to not having infinities.
  EXPECT_EQ(0x1F, iree_math_f32_to_f6e2m3fn(kF6E2M3FNMax));
  EXPECT_EQ(0x3F, iree_math_f32_to_f6e2m3fn(-kF6E2M3FNMax));
  // Min normal values.
  EXPECT_EQ(0x08, iree_math_f32_to_f6e2m3fn(kF6E2M3FNMin));
  EXPECT_EQ(0x28, iree_math_f32_to_f6e2m3fn(-kF6E2M3FNMin));
  // Infinity clamped to max finite.
  EXPECT_EQ(0x1F, iree_math_f32_to_f6e2m3fn(INFINITY));
  EXPECT_EQ(0x3F, iree_math_f32_to_f6e2m3fn(-INFINITY));
  // Large finite value clamped to max finite.
  EXPECT_EQ(0x1F, iree_math_f32_to_f6e2m3fn(FLT_MAX));
  EXPECT_EQ(0x3F, iree_math_f32_to_f6e2m3fn(-FLT_MAX));
  // Test some round-to-nearest-even behavior.
  EXPECT_EQ(0x18, iree_math_f32_to_f6e2m3fn(4.0f));
  EXPECT_EQ(0x18, iree_math_f32_to_f6e2m3fn(4.25f));
  EXPECT_EQ(0x19, iree_math_f32_to_f6e2m3fn(4.5f));
  EXPECT_EQ(0x1A, iree_math_f32_to_f6e2m3fn(4.75f));
  EXPECT_EQ(0x1A, iree_math_f32_to_f6e2m3fn(5.0f));
  // Underflow
  EXPECT_EQ(0, iree_math_f32_to_f6e2m3fn(FLT_MIN));
  EXPECT_EQ(0x20, iree_math_f32_to_f6e2m3fn(-FLT_MIN));
  // NaN conversion is implementation-defined. We canonicalize to +0.0.
  EXPECT_EQ(0, iree_math_f32_to_f6e2m3fn(NAN));
}

TEST(F6E2M3FNConversionTest, Denormals) {
  CheckDenormals<uint8_t>(2, 3, /*bias_tweak=*/0, /*have_neg_zero=*/true,
                          iree_math_f32_to_f6e2m3fn, iree_math_f6e2m3fn_to_f32);
}

TEST(F6E2M3FNConversionTest, F6E2M3FNToF32) {
  for (int sign_bit = 0; sign_bit <= 0x20; sign_bit += 0x20) {
    float sign = sign_bit ? -1.f : 1.f;
    // Zero
    EXPECT_EQ(sign * 0x0.0p0f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x00));
    // Denormals
    EXPECT_EQ(sign * 0x0.2p+0f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x01));
    EXPECT_EQ(sign * 0x0.4p+0f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x02));
    EXPECT_EQ(sign * 0x0.6p+0f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x03));
    EXPECT_EQ(sign * 0x0.8p+0f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x04));
    EXPECT_EQ(sign * 0x0.Ap+0f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x05));
    EXPECT_EQ(sign * 0x0.Cp+0f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x06));
    EXPECT_EQ(sign * 0x0.Ep+0f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x07));
    // Normal finite values
    EXPECT_EQ(sign * 0x1.0p+0f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x08));
    EXPECT_EQ(sign * 0x1.2p+0f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x09));
    EXPECT_EQ(sign * 0x1.4p+0f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x0A));
    EXPECT_EQ(sign * 0x1.6p+0f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x0B));
    EXPECT_EQ(sign * 0x1.8p+0f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x0C));
    EXPECT_EQ(sign * 0x1.Ap+0f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x0D));
    EXPECT_EQ(sign * 0x1.Cp+0f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x0E));
    EXPECT_EQ(sign * 0x1.Ep+0f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x0F));
    EXPECT_EQ(sign * 0x1.0p+1f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x10));
    EXPECT_EQ(sign * 0x1.2p+1f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x11));
    EXPECT_EQ(sign * 0x1.4p+1f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x12));
    EXPECT_EQ(sign * 0x1.6p+1f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x13));
    EXPECT_EQ(sign * 0x1.8p+1f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x14));
    EXPECT_EQ(sign * 0x1.Ap+1f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x15));
    EXPECT_EQ(sign * 0x1.Cp+1f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x16));
    EXPECT_EQ(sign * 0x1.Ep+1f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x17));
    // Extra finite values in the top exponent thanks to no Inf and no NaN.
    EXPECT_EQ(sign * 0x1.0p+2f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x18));
    EXPECT_EQ(sign * 0x1.2p+2f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x19));
    EXPECT_EQ(sign * 0x1.4p+2f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x1A));
    EXPECT_EQ(sign * 0x1.6p+2f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x1B));
    EXPECT_EQ(sign * 0x1.8p+2f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x1C));
    EXPECT_EQ(sign * 0x1.Ap+2f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x1D));
    EXPECT_EQ(sign * 0x1.Cp+2f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x1E));
    EXPECT_EQ(sign * 0x1.Ep+2f, iree_math_f6e2m3fn_to_f32(sign_bit | 0x1F));
  }
}

//==============================================================================
// F4E2M1FN support
//==============================================================================

// See
// https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
// Paragraph 5.3.3.

TEST(F4E2M1FNConversionTest, F32ToF4E2M1FN) {
  constexpr float kF4E2M1FNMax = 6.0f;
  constexpr float kF4E2M1FNMin = 1.0f;
  // Within range, normal truncation.
  EXPECT_EQ(0x00, iree_math_f32_to_f4e2m1fn(0.f));
  EXPECT_EQ(0x08, iree_math_f32_to_f4e2m1fn(-0.f));
  EXPECT_EQ(0x02, iree_math_f32_to_f4e2m1fn(1.0f));
  EXPECT_EQ(0x0D, iree_math_f32_to_f4e2m1fn(-3.0f));
  // Extra large finite values thanks to not having infinities.
  EXPECT_EQ(0x07, iree_math_f32_to_f4e2m1fn(kF4E2M1FNMax));
  EXPECT_EQ(0x0F, iree_math_f32_to_f4e2m1fn(-kF4E2M1FNMax));
  // Min normal values.
  EXPECT_EQ(0x02, iree_math_f32_to_f4e2m1fn(kF4E2M1FNMin));
  EXPECT_EQ(0x0A, iree_math_f32_to_f4e2m1fn(-kF4E2M1FNMin));
  // Infinity clamped to max finite.
  EXPECT_EQ(0x07, iree_math_f32_to_f4e2m1fn(INFINITY));
  EXPECT_EQ(0x0F, iree_math_f32_to_f4e2m1fn(-INFINITY));
  // Large finite value clamped to max finite.
  EXPECT_EQ(0x07, iree_math_f32_to_f4e2m1fn(FLT_MAX));
  EXPECT_EQ(0x0F, iree_math_f32_to_f4e2m1fn(-FLT_MAX));
  // Test some round-to-nearest-even behavior.
  EXPECT_EQ(0x04, iree_math_f32_to_f4e2m1fn(2.0f));
  EXPECT_EQ(0x04, iree_math_f32_to_f4e2m1fn(2.5f));
  EXPECT_EQ(0x05, iree_math_f32_to_f4e2m1fn(3.0f));
  EXPECT_EQ(0x06, iree_math_f32_to_f4e2m1fn(3.5f));
  EXPECT_EQ(0x06, iree_math_f32_to_f4e2m1fn(4.0f));
  // Underflow
  EXPECT_EQ(0x00, iree_math_f32_to_f4e2m1fn(FLT_MIN));
  EXPECT_EQ(0x08, iree_math_f32_to_f4e2m1fn(-FLT_MIN));
  // NaN conversion is implementation-defined. We canonicalize to +0.0.
  EXPECT_EQ(0, iree_math_f32_to_f4e2m1fn(NAN));
}

TEST(F4E2M1FNConversionTest, Denormals) {
  CheckDenormals<uint8_t>(2, 1, /*bias_tweak=*/0, /*have_neg_zero=*/true,
                          iree_math_f32_to_f4e2m1fn, iree_math_f4e2m1fn_to_f32);
}

TEST(F4E2M1FNConversionTest, F4E2M1FNToF32) {
  for (int sign_bit = 0; sign_bit <= 0x08; sign_bit += 0x08) {
    float sign = sign_bit ? -1.f : 1.f;
    // Zero
    EXPECT_EQ(sign * 0x0.0p0f, iree_math_f4e2m1fn_to_f32(sign_bit | 0x00));
    // Denormals
    EXPECT_EQ(sign * 0x0.8p+0f, iree_math_f4e2m1fn_to_f32(sign_bit | 0x01));
    // Normal finite values
    EXPECT_EQ(sign * 0x1.0p+0f, iree_math_f4e2m1fn_to_f32(sign_bit | 0x02));
    EXPECT_EQ(sign * 0x1.8p+0f, iree_math_f4e2m1fn_to_f32(sign_bit | 0x03));
    EXPECT_EQ(sign * 0x1.0p+1f, iree_math_f4e2m1fn_to_f32(sign_bit | 0x04));
    EXPECT_EQ(sign * 0x1.8p+1f, iree_math_f4e2m1fn_to_f32(sign_bit | 0x05));
    // Extra finite values in the top exponent thanks to no Inf and no NaN.
    EXPECT_EQ(sign * 0x1.0p+2f, iree_math_f4e2m1fn_to_f32(sign_bit | 0x06));
    EXPECT_EQ(sign * 0x1.8p+2f, iree_math_f4e2m1fn_to_f32(sign_bit | 0x07));
  }
}

//==============================================================================
// F8E8M0FNU support
//==============================================================================

// See
// https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
// Paragraph 5.4.1.

TEST(F8E8M0FNUConversionTest, F32ToF8E8M0FNU) {
  EXPECT_EQ(0xFF, iree_math_f32_to_f8e8m0fnu(NAN));
  for (int exp = -127; exp <= 127; ++exp) {
    EXPECT_EQ(127 + exp, iree_math_f32_to_f8e8m0fnu(ldexpf(1.0f, exp)));
    // 1.5 Should get rounded to the next exponent value or to NaN if that
    // overflows.
    EXPECT_EQ(128 + exp, iree_math_f32_to_f8e8m0fnu(ldexpf(1.5f, exp)));
  }
  EXPECT_EQ(0xFF, iree_math_f32_to_f8e8m0fnu(NAN));
  EXPECT_EQ(0xFF, iree_math_f32_to_f8e8m0fnu(+INFINITY));
  EXPECT_EQ(0xFF, iree_math_f32_to_f8e8m0fnu(-INFINITY));
  EXPECT_EQ(0x00, iree_math_f32_to_f8e8m0fnu(-1.0f));
  EXPECT_EQ(0x00, iree_math_f32_to_f8e8m0fnu(0.0f));
  EXPECT_EQ(0x01, iree_math_f32_to_f8e8m0fnu(FLT_MIN));
  // Overflow to NaN
  EXPECT_EQ(0xFF, iree_math_f32_to_f8e8m0fnu(FLT_MAX));
}

TEST(F8E8M0FNUConversionTest, F8E8M0FNUToF32) {
  for (int value = 0; value <= 0xFE; ++value) {
    EXPECT_EQ(ldexpf(1.0f, value - 127), iree_math_f8e8m0fnu_to_f32(value));
  }
  EXPECT_TRUE(isnan(iree_math_f8e8m0fnu_to_f32(0xFF)));
}

}  // namespace
