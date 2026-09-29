// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <climits>
#include <cstdint>

#include "iree/testing/gtest.h"
#include "loom/ir/facts.h"

namespace loom {
namespace {

static uint64_t RawBits(int64_t value, int32_t bit_count) {
  return (uint64_t)value & iree_math_mask_low_bits_u64(UINT64_MAX, bit_count);
}

static int64_t SignedRepresentation(uint64_t raw_bits, int32_t bit_count) {
  raw_bits &= iree_math_mask_low_bits_u64(UINT64_MAX, bit_count);
  if (bit_count == 1) {
    return (int64_t)raw_bits;
  }
  const uint64_t sign_bit = UINT64_C(1) << (bit_count - 1);
  if ((raw_bits & sign_bit) == 0) {
    return (int64_t)raw_bits;
  }
  const int64_t signed_sign_bit =
      bit_count == 64 ? INT64_MIN : -(int64_t)sign_bit;
  return signed_sign_bit + (int64_t)(raw_bits & (sign_bit - 1));
}

static loom_value_facts_t ExactBits(uint64_t raw_bits, int32_t bit_count) {
  return loom_value_facts_exact_i64(SignedRepresentation(raw_bits, bit_count));
}

static void ExpectContains(loom_value_facts_t facts, int64_t value) {
  EXPECT_LE(facts.range_lo, value);
  EXPECT_GE(facts.range_hi, value);
  EXPECT_EQ(value % facts.known_divisor, 0);
}

TEST(UnsignedDivisionTransfer, ExactSmallWidths) {
  for (int32_t bit_count = 1; bit_count <= 8; ++bit_count) {
    const uint64_t mask = iree_math_mask_low_bits_u64(UINT64_MAX, bit_count);
    for (uint64_t lhs_bits = 0; lhs_bits <= mask; ++lhs_bits) {
      const loom_value_facts_t lhs = ExactBits(lhs_bits, bit_count);
      for (uint64_t rhs_bits = 1; rhs_bits <= mask; ++rhs_bits) {
        SCOPED_TRACE(::testing::Message() << "width=" << bit_count << " lhs="
                                          << lhs_bits << " rhs=" << rhs_bits);
        const loom_value_facts_t rhs = ExactBits(rhs_bits, bit_count);
        loom_value_facts_t quotient;
        loom_value_facts_divui(&lhs, &rhs, bit_count, &quotient);
        EXPECT_TRUE(loom_value_facts_is_exact(quotient));
        EXPECT_EQ(quotient.range_lo,
                  SignedRepresentation(lhs_bits / rhs_bits, bit_count));

        loom_value_facts_t remainder;
        loom_value_facts_remui(&lhs, &rhs, bit_count, &remainder);
        EXPECT_TRUE(loom_value_facts_is_exact(remainder));
        EXPECT_EQ(remainder.range_lo,
                  SignedRepresentation(lhs_bits % rhs_bits, bit_count));
      }
    }
  }
}

TEST(UnsignedDivisionTransfer, RepresentativeIntervalsContainEveryResult) {
  constexpr int64_t kEndpoints[] = {-8, -5, -1, 0, 1, 3, 7};
  for (int64_t lhs_lo : kEndpoints) {
    for (int64_t lhs_hi : kEndpoints) {
      if (lhs_lo > lhs_hi) {
        continue;
      }
      for (int64_t lhs_divisor :
           {INT64_C(1), INT64_C(2), INT64_C(3), INT64_C(4)}) {
        const loom_value_facts_t lhs =
            loom_value_facts_make(lhs_lo, lhs_hi, lhs_divisor);
        for (int64_t rhs_lo : kEndpoints) {
          for (int64_t rhs_hi : kEndpoints) {
            if (rhs_lo > rhs_hi) {
              continue;
            }
            SCOPED_TRACE(::testing::Message()
                         << "lhs=[" << lhs_lo << "," << lhs_hi
                         << "] divisor=" << lhs_divisor << " rhs=[" << rhs_lo
                         << "," << rhs_hi << "]");
            const loom_value_facts_t rhs =
                loom_value_facts_make(rhs_lo, rhs_hi, 1);
            loom_value_facts_t quotient;
            loom_value_facts_divui(&lhs, &rhs, 4, &quotient);
            loom_value_facts_t remainder;
            loom_value_facts_remui(&lhs, &rhs, 4, &remainder);
            for (int64_t lhs_value = lhs_lo; lhs_value <= lhs_hi; ++lhs_value) {
              if (lhs_value % lhs.known_divisor != 0) {
                continue;
              }
              const uint64_t lhs_bits = RawBits(lhs_value, 4);
              for (int64_t rhs_value = rhs_lo; rhs_value <= rhs_hi;
                   ++rhs_value) {
                if (rhs_value == 0) {
                  continue;
                }
                const uint64_t rhs_bits = RawBits(rhs_value, 4);
                ExpectContains(quotient,
                               SignedRepresentation(lhs_bits / rhs_bits, 4));
                ExpectContains(remainder,
                               SignedRepresentation(lhs_bits % rhs_bits, 4));
              }
            }
          }
        }
      }
    }
  }
}

TEST(UnsignedDivisionTransfer, ExactNegativeRepresentationsUseDeclaredWidth) {
  const loom_value_facts_t dividend = loom_value_facts_exact_i64(-30);
  const loom_value_facts_t divisor = loom_value_facts_exact_i64(5);
  for (int32_t bit_count : {32, 64}) {
    const uint64_t dividend_bits = RawBits(-30, bit_count);
    loom_value_facts_t quotient;
    loom_value_facts_divui(&dividend, &divisor, bit_count, &quotient);
    EXPECT_TRUE(loom_value_facts_is_exact(quotient));
    EXPECT_EQ(quotient.range_lo,
              SignedRepresentation(dividend_bits / 5, bit_count));
    EXPECT_NE(quotient.known_divisor, 6);

    loom_value_facts_t remainder;
    loom_value_facts_remui(&dividend, &divisor, bit_count, &remainder);
    EXPECT_TRUE(loom_value_facts_is_exact(remainder));
    EXPECT_EQ(remainder.range_lo,
              SignedRepresentation(dividend_bits % 5, bit_count));
  }
}

TEST(UnsignedDivisionTransfer, UnknownDividendRetainsSmallDivisorBounds) {
  loom_value_facts_t dividend = loom_value_facts_unknown();
  loom_value_facts_mark_lane_varying(&dividend);
  const loom_value_facts_t divisor = loom_value_facts_exact_i64(9);
  for (int32_t bit_count : {8, 16, 32, 64}) {
    loom_value_facts_t quotient;
    loom_value_facts_divui(&dividend, &divisor, bit_count, &quotient);
    EXPECT_EQ(quotient.range_lo, 0);
    EXPECT_EQ(
        quotient.range_hi,
        (int64_t)(iree_math_mask_low_bits_u64(UINT64_MAX, bit_count) / 9));
    EXPECT_TRUE(loom_value_facts_is_lane_varying(quotient));

    loom_value_facts_t remainder;
    loom_value_facts_remui(&dividend, &divisor, bit_count, &remainder);
    EXPECT_EQ(remainder.range_lo, 0);
    EXPECT_EQ(remainder.range_hi, 8);
    EXPECT_TRUE(loom_value_facts_is_non_negative(remainder));
    EXPECT_TRUE(loom_value_facts_is_lane_varying(remainder));
  }
}

TEST(UnsignedDivisionTransfer, NegativeRangeRetainsOnlyRawBitDivisibility) {
  const loom_value_facts_t dividend = loom_value_facts_make(-32, -16, 16);
  const loom_value_facts_t divisor = loom_value_facts_exact_i64(4);
  loom_value_facts_t quotient;
  loom_value_facts_divui(&dividend, &divisor, 8, &quotient);
  EXPECT_EQ(quotient.range_lo, 56);
  EXPECT_EQ(quotient.range_hi, 60);
  EXPECT_EQ(quotient.known_divisor, 4);

  loom_value_facts_t remainder;
  loom_value_facts_remui(&dividend, &divisor, 8, &remainder);
  EXPECT_TRUE(loom_value_facts_is_exact(remainder));
  EXPECT_EQ(remainder.range_lo, 0);

  const loom_value_facts_t odd_dividend = loom_value_facts_make(-30, -15, 15);
  const loom_value_facts_t five = loom_value_facts_exact_i64(5);
  loom_value_facts_divui(&odd_dividend, &five, 8, &quotient);
  EXPECT_EQ(quotient.known_divisor, 1);
}

TEST(UnsignedDivisionTransfer, HighBitDivisorsUseRawWidth) {
  const loom_value_facts_t dividend = loom_value_facts_unknown();
  loom_value_facts_t divisor = ExactBits(UINT64_C(0x80), 8);
  loom_value_facts_t quotient;
  loom_value_facts_divui(&dividend, &divisor, 8, &quotient);
  EXPECT_EQ(quotient.range_lo, 0);
  EXPECT_EQ(quotient.range_hi, 1);
  loom_value_facts_t remainder;
  loom_value_facts_remui(&dividend, &divisor, 8, &remainder);
  EXPECT_EQ(remainder.range_lo, 0);
  EXPECT_EQ(remainder.range_hi, 127);

  divisor = ExactBits(UINT64_C(0xFF), 8);
  loom_value_facts_remui(&dividend, &divisor, 8, &remainder);
  EXPECT_EQ(remainder.range_lo, -128);
  EXPECT_EQ(remainder.range_hi, 127);
}

TEST(UnsignedDivisionTransfer, NonzeroFactExcludesDynamicZeroDivisor) {
  const loom_value_facts_t dividend = loom_value_facts_make(0, 63, 1);
  loom_value_facts_t divisor = loom_value_facts_make(0, 9, 1);
  loom_value_facts_t quotient;
  loom_value_facts_divui(&dividend, &divisor, 8, &quotient);
  EXPECT_EQ(quotient.range_lo, -128);
  EXPECT_EQ(quotient.range_hi, 127);
  loom_value_facts_t remainder;
  loom_value_facts_remui(&dividend, &divisor, 8, &remainder);
  EXPECT_EQ(remainder.range_lo, -128);
  EXPECT_EQ(remainder.range_hi, 127);

  divisor.flags |= LOOM_VALUE_FACT_NON_ZERO;
  loom_value_facts_divui(&dividend, &divisor, 8, &quotient);
  EXPECT_EQ(quotient.range_lo, 0);
  EXPECT_EQ(quotient.range_hi, 63);
  loom_value_facts_remui(&dividend, &divisor, 8, &remainder);
  EXPECT_EQ(remainder.range_lo, 0);
  EXPECT_EQ(remainder.range_hi, 8);
}

TEST(UnsignedDivisionTransfer, SupportsEitherOutputAlias) {
  loom_value_facts_t dividend = loom_value_facts_exact_i64(-30);
  loom_value_facts_t divisor = loom_value_facts_exact_i64(5);
  loom_value_facts_divui(&dividend, &divisor, 32, &dividend);
  EXPECT_EQ(dividend.range_lo, 858993453);

  dividend = loom_value_facts_exact_i64(-30);
  loom_value_facts_remui(&dividend, &divisor, 32, &divisor);
  EXPECT_EQ(divisor.range_lo, 1);
}

TEST(RemsiTransfer, ExactMinimumOverflowPair) {
  loom_value_facts_t dividend = loom_value_facts_exact_i64(INT64_MIN);
  loom_value_facts_t divisor = loom_value_facts_exact_i64(-1);
  loom_value_facts_t out;
  loom_value_facts_remsi(&dividend, &divisor, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, 0);
}

TEST(RemsiTransfer, ExactRemainderPreservesDividendSign) {
  loom_value_facts_t out;
  loom_value_facts_t dividend = loom_value_facts_exact_i64(-17);
  loom_value_facts_t divisor = loom_value_facts_exact_i64(5);
  loom_value_facts_remsi(&dividend, &divisor, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, -2);

  dividend = loom_value_facts_exact_i64(17);
  divisor = loom_value_facts_exact_i64(-5);
  loom_value_facts_remsi(&dividend, &divisor, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, 2);
}

TEST(RemsiTransfer, MinimumDivisorHasRepresentableRemainderBounds) {
  loom_value_facts_t dividend = loom_value_facts_unknown();
  loom_value_facts_t divisor = loom_value_facts_exact_i64(INT64_MIN);
  loom_value_facts_t out;
  loom_value_facts_remsi(&dividend, &divisor, &out);
  EXPECT_EQ(out.range_lo, -INT64_MAX);
  EXPECT_EQ(out.range_hi, INT64_MAX);
}

}  // namespace
}  // namespace loom
