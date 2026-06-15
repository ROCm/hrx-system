// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/math.h"

#include <climits>
#include <cstdint>

#include "iree/testing/gtest.h"

namespace loom {
namespace {

//===----------------------------------------------------------------------===//
// loom_checked_add_i64
//===----------------------------------------------------------------------===//

TEST(CheckedAddI64, Normal) {
  int64_t out = 0;
  EXPECT_TRUE(loom_checked_add_i64(0, 0, &out));
  EXPECT_EQ(out, 0);
  EXPECT_TRUE(loom_checked_add_i64(100, 200, &out));
  EXPECT_EQ(out, 300);
  EXPECT_TRUE(loom_checked_add_i64(-100, -200, &out));
  EXPECT_EQ(out, -300);
  EXPECT_TRUE(loom_checked_add_i64(-100, 200, &out));
  EXPECT_EQ(out, 100);
  EXPECT_TRUE(loom_checked_add_i64(100, -200, &out));
  EXPECT_EQ(out, -100);
}

TEST(CheckedAddI64, Boundary) {
  int64_t out = 0;
  EXPECT_TRUE(loom_checked_add_i64(INT64_MAX, 0, &out));
  EXPECT_EQ(out, INT64_MAX);
  EXPECT_TRUE(loom_checked_add_i64(INT64_MIN, 0, &out));
  EXPECT_EQ(out, INT64_MIN);
  EXPECT_TRUE(loom_checked_add_i64(INT64_MAX - 1, 1, &out));
  EXPECT_EQ(out, INT64_MAX);
  EXPECT_TRUE(loom_checked_add_i64(INT64_MIN + 1, -1, &out));
  EXPECT_EQ(out, INT64_MIN);
}

TEST(CheckedAddI64, Overflow) {
  int64_t out = 0;
  EXPECT_FALSE(loom_checked_add_i64(INT64_MAX, 1, &out));
  EXPECT_FALSE(loom_checked_add_i64(INT64_MIN, -1, &out));
  EXPECT_FALSE(
      loom_checked_add_i64(INT64_MAX / 2 + 1, INT64_MAX / 2 + 1, &out));
  EXPECT_FALSE(
      loom_checked_add_i64(INT64_MIN / 2 - 1, INT64_MIN / 2 - 1, &out));
}

TEST(CheckedAddI64, OppositeSignsNeverOverflow) {
  // Adding values of opposite signs can never overflow.
  int64_t out = 0;
  EXPECT_TRUE(loom_checked_add_i64(INT64_MAX, INT64_MIN, &out));
  EXPECT_EQ(out, -1);
  EXPECT_TRUE(loom_checked_add_i64(INT64_MIN, INT64_MAX, &out));
  EXPECT_EQ(out, -1);
}

//===----------------------------------------------------------------------===//
// loom_checked_sub_i64
//===----------------------------------------------------------------------===//

TEST(CheckedSubI64, Normal) {
  int64_t out = 0;
  EXPECT_TRUE(loom_checked_sub_i64(300, 200, &out));
  EXPECT_EQ(out, 100);
  EXPECT_TRUE(loom_checked_sub_i64(0, 0, &out));
  EXPECT_EQ(out, 0);
  EXPECT_TRUE(loom_checked_sub_i64(-100, 200, &out));
  EXPECT_EQ(out, -300);
}

TEST(CheckedSubI64, Boundary) {
  int64_t out = 0;
  EXPECT_TRUE(loom_checked_sub_i64(INT64_MIN + 1, 1, &out));
  EXPECT_EQ(out, INT64_MIN);
  EXPECT_TRUE(loom_checked_sub_i64(INT64_MAX - 1, -1, &out));
  EXPECT_EQ(out, INT64_MAX);
}

TEST(CheckedSubI64, Overflow) {
  int64_t out = 0;
  EXPECT_FALSE(loom_checked_sub_i64(INT64_MIN, 1, &out));
  EXPECT_FALSE(loom_checked_sub_i64(INT64_MAX, -1, &out));
  EXPECT_FALSE(loom_checked_sub_i64(INT64_MIN, INT64_MAX, &out));
}

TEST(CheckedSubI64, SameSignNeverOverflow) {
  // Subtracting values of the same sign can never overflow.
  int64_t out = 0;
  EXPECT_TRUE(loom_checked_sub_i64(INT64_MAX, INT64_MAX, &out));
  EXPECT_EQ(out, 0);
  EXPECT_TRUE(loom_checked_sub_i64(INT64_MIN, INT64_MIN, &out));
  EXPECT_EQ(out, 0);
}

//===----------------------------------------------------------------------===//
// loom_checked_mul_i64
//===----------------------------------------------------------------------===//

TEST(CheckedMulI64, Normal) {
  int64_t out = 0;
  EXPECT_TRUE(loom_checked_mul_i64(6, 7, &out));
  EXPECT_EQ(out, 42);
  EXPECT_TRUE(loom_checked_mul_i64(-6, 7, &out));
  EXPECT_EQ(out, -42);
  EXPECT_TRUE(loom_checked_mul_i64(-6, -7, &out));
  EXPECT_EQ(out, 42);
  EXPECT_TRUE(loom_checked_mul_i64(6, -7, &out));
  EXPECT_EQ(out, -42);
}

TEST(CheckedMulI64, Zero) {
  int64_t out = 99;
  EXPECT_TRUE(loom_checked_mul_i64(0, INT64_MAX, &out));
  EXPECT_EQ(out, 0);
  EXPECT_TRUE(loom_checked_mul_i64(INT64_MIN, 0, &out));
  EXPECT_EQ(out, 0);
  EXPECT_TRUE(loom_checked_mul_i64(0, 0, &out));
  EXPECT_EQ(out, 0);
}

TEST(CheckedMulI64, Identity) {
  int64_t out = 0;
  EXPECT_TRUE(loom_checked_mul_i64(INT64_MAX, 1, &out));
  EXPECT_EQ(out, INT64_MAX);
  EXPECT_TRUE(loom_checked_mul_i64(INT64_MIN, 1, &out));
  EXPECT_EQ(out, INT64_MIN);
}

TEST(CheckedMulI64, Overflow) {
  int64_t out = 0;
  EXPECT_FALSE(loom_checked_mul_i64(INT64_MAX, 2, &out));
  EXPECT_FALSE(loom_checked_mul_i64(INT64_MIN, -1, &out));
  EXPECT_FALSE(loom_checked_mul_i64(INT64_MIN, 2, &out));
  EXPECT_FALSE(loom_checked_mul_i64(INT64_MAX / 2 + 1, 2, &out));
}

TEST(CheckedMulI64, NearMiss) {
  int64_t out = 0;
  EXPECT_TRUE(loom_checked_mul_i64(INT64_MAX / 2, 2, &out));
  EXPECT_EQ(out, (INT64_MAX / 2) * 2);
  EXPECT_TRUE(loom_checked_mul_i64(INT64_MIN / 2, 2, &out));
  EXPECT_EQ(out, INT64_MIN);
}

//===----------------------------------------------------------------------===//
// loom_gcd_i64
//===----------------------------------------------------------------------===//

TEST(GcdI64, BothZero) {
  // Sentinel for the facts system: gcd(0, 0) = 1.
  EXPECT_EQ(loom_gcd_i64(0, 0), 1);
}

TEST(GcdI64, OneZero) {
  EXPECT_EQ(loom_gcd_i64(0, 5), 5);
  EXPECT_EQ(loom_gcd_i64(5, 0), 5);
  EXPECT_EQ(loom_gcd_i64(0, 1), 1);
}

TEST(GcdI64, Normal) {
  EXPECT_EQ(loom_gcd_i64(12, 8), 4);
  EXPECT_EQ(loom_gcd_i64(8, 12), 4);
  EXPECT_EQ(loom_gcd_i64(17, 13), 1);
  EXPECT_EQ(loom_gcd_i64(100, 100), 100);
  EXPECT_EQ(loom_gcd_i64(16, 24), 8);
  EXPECT_EQ(loom_gcd_i64(48, 18), 6);
}

TEST(GcdI64, Negative) {
  EXPECT_EQ(loom_gcd_i64(-12, 8), 4);
  EXPECT_EQ(loom_gcd_i64(12, -8), 4);
  EXPECT_EQ(loom_gcd_i64(-12, -8), 4);
}

TEST(GcdI64, Int64Min) {
  // INT64_MIN = -2^63. gcd(INT64_MIN, 2) should be 2.
  EXPECT_EQ(loom_gcd_i64(INT64_MIN, 2), 2);
  EXPECT_EQ(loom_gcd_i64(2, INT64_MIN), 2);
  EXPECT_EQ(loom_gcd_i64(INT64_MIN, 4), 4);
}

TEST(GcdI64, PowerOfTwo) {
  EXPECT_EQ(loom_gcd_i64(64, 16), 16);
  EXPECT_EQ(loom_gcd_i64(256, 1024), 256);
}

//===----------------------------------------------------------------------===//
// loom_lcm_i64
//===----------------------------------------------------------------------===//

TEST(LcmI64, Zero) {
  int64_t out = 99;
  EXPECT_TRUE(loom_lcm_i64(0, 5, &out));
  EXPECT_EQ(out, 0);
  EXPECT_TRUE(loom_lcm_i64(5, 0, &out));
  EXPECT_EQ(out, 0);
}

TEST(LcmI64, Normal) {
  int64_t out = 0;
  EXPECT_TRUE(loom_lcm_i64(4, 6, &out));
  EXPECT_EQ(out, 12);
  EXPECT_TRUE(loom_lcm_i64(3, 5, &out));
  EXPECT_EQ(out, 15);
  EXPECT_TRUE(loom_lcm_i64(7, 7, &out));
  EXPECT_EQ(out, 7);
  EXPECT_TRUE(loom_lcm_i64(16, 24, &out));
  EXPECT_EQ(out, 48);
}

TEST(LcmI64, Negative) {
  int64_t out = 0;
  EXPECT_TRUE(loom_lcm_i64(-4, 6, &out));
  EXPECT_EQ(out, 12);
}

TEST(LcmI64, Overflow) {
  int64_t out = 0;
  EXPECT_FALSE(loom_lcm_i64(INT64_MAX, 2, &out));
}

//===----------------------------------------------------------------------===//
// loom_ilog2_i64
//===----------------------------------------------------------------------===//

TEST(Ilog2I64, PowersOfTwo) {
  EXPECT_EQ(loom_ilog2_i64(1), 0);
  EXPECT_EQ(loom_ilog2_i64(2), 1);
  EXPECT_EQ(loom_ilog2_i64(4), 2);
  EXPECT_EQ(loom_ilog2_i64(8), 3);
  EXPECT_EQ(loom_ilog2_i64(1024), 10);
  EXPECT_EQ(loom_ilog2_i64(1LL << 62), 62);
}

TEST(Ilog2I64, NonPowers) {
  EXPECT_EQ(loom_ilog2_i64(3), 1);
  EXPECT_EQ(loom_ilog2_i64(5), 2);
  EXPECT_EQ(loom_ilog2_i64(7), 2);
  EXPECT_EQ(loom_ilog2_i64(9), 3);
  EXPECT_EQ(loom_ilog2_i64(INT64_MAX), 62);
}

//===----------------------------------------------------------------------===//
// Bit-width-limited counts
//===----------------------------------------------------------------------===//

TEST(CountLeadingZerosU64Width, UsesDeclaredWidth) {
  EXPECT_EQ(loom_count_leading_zeros_u64_width(1, 32), 31);
  EXPECT_EQ(loom_count_leading_zeros_u64_width(1, 8), 7);
  EXPECT_EQ(loom_count_leading_zeros_u64_width(0, 32), 32);
}

TEST(CountTrailingZerosU64Width, UsesDeclaredWidth) {
  EXPECT_EQ(loom_count_trailing_zeros_u64_width(8, 32), 3);
  EXPECT_EQ(loom_count_trailing_zeros_u64_width(0, 8), 8);
}

TEST(CountOnesU64Width, MasksToDeclaredWidth) {
  EXPECT_EQ(loom_count_ones_u64_width(UINT64_MAX, 32), 32);
  EXPECT_EQ(loom_count_ones_u64_width(UINT64_MAX, 8), 8);
}

//===----------------------------------------------------------------------===//
// loom_is_power_of_two_i64
//===----------------------------------------------------------------------===//

TEST(IsPowerOfTwoI64, True) {
  EXPECT_TRUE(loom_is_power_of_two_i64(1));
  EXPECT_TRUE(loom_is_power_of_two_i64(2));
  EXPECT_TRUE(loom_is_power_of_two_i64(4));
  EXPECT_TRUE(loom_is_power_of_two_i64(16));
  EXPECT_TRUE(loom_is_power_of_two_i64(1024));
  EXPECT_TRUE(loom_is_power_of_two_i64(1LL << 62));
}

TEST(IsPowerOfTwoI64, False) {
  EXPECT_FALSE(loom_is_power_of_two_i64(0));
  EXPECT_FALSE(loom_is_power_of_two_i64(-1));
  EXPECT_FALSE(loom_is_power_of_two_i64(-2));
  EXPECT_FALSE(loom_is_power_of_two_i64(3));
  EXPECT_FALSE(loom_is_power_of_two_i64(5));
  EXPECT_FALSE(loom_is_power_of_two_i64(6));
  EXPECT_FALSE(loom_is_power_of_two_i64(7));
  EXPECT_FALSE(loom_is_power_of_two_i64(INT64_MIN));
}

}  // namespace
}  // namespace loom
