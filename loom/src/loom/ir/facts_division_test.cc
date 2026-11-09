// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <climits>
#include <cstdint>

#include "iree/testing/gtest.h"
#include "loom/ir/facts.h"
#include "loom/ir/facts_division_test_util.h"

namespace loom {
namespace {

static void ExpectContains(loom_value_facts_t facts, int64_t value) {
  EXPECT_LE(facts.range_lo, value);
  EXPECT_GE(facts.range_hi, value);
  EXPECT_EQ(value % facts.known_divisor, 0);
  if (loom_value_facts_is_non_zero(facts)) {
    EXPECT_NE(value, 0);
  }
  if (loom_value_facts_is_power_of_two(facts)) {
    EXPECT_GT(value, 0);
    EXPECT_EQ(static_cast<uint64_t>(value) & (static_cast<uint64_t>(value) - 1),
              0);
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
            const bool identity =
                loom_value_facts_remui_is_identity(lhs, rhs, 4);
            for (int64_t lhs_value = lhs_lo; lhs_value <= lhs_hi; ++lhs_value) {
              if (lhs_value % lhs.known_divisor != 0) {
                continue;
              }
              const uint64_t lhs_bits = testing::RawIntegerBits(lhs_value, 4);
              for (int64_t rhs_value = rhs_lo; rhs_value <= rhs_hi;
                   ++rhs_value) {
                if (rhs_value == 0) {
                  continue;
                }
                const uint64_t rhs_bits = testing::RawIntegerBits(rhs_value, 4);
                if (identity) {
                  EXPECT_LT(lhs_bits, rhs_bits);
                }
                ExpectContains(quotient, testing::SignedIntegerRepresentation(
                                             lhs_bits / rhs_bits, 4));
                ExpectContains(remainder, testing::SignedIntegerRepresentation(
                                              lhs_bits % rhs_bits, 4));
              }
            }
          }
        }
      }
    }
  }
}

TEST(UnsignedDivisionTransfer, WideOperandEndpointsRetainTheirSignedDomain) {
  const auto nonnegative = loom_value_facts_make(0, INT64_MAX, 1);
  const auto positive = loom_value_facts_make(1, INT64_MAX, 1);
  const auto high_bits = loom_value_facts_make(INT64_MIN, -1, 1);
  loom_value_facts_t quotient;
  loom_value_facts_t remainder;
  loom_value_facts_divui(&nonnegative, &positive, 64, &quotient);
  EXPECT_EQ(quotient.range_lo, 0);
  EXPECT_EQ(quotient.range_hi, INT64_MAX);
  loom_value_facts_remui(&nonnegative, &positive, 64, &remainder);
  EXPECT_EQ(remainder.range_lo, 0);
  EXPECT_EQ(remainder.range_hi, INT64_MAX - 1);

  EXPECT_TRUE(loom_value_facts_remui_is_identity(nonnegative, high_bits, 64));
  loom_value_facts_remui(&nonnegative, &high_bits, 64, &remainder);
  EXPECT_TRUE(loom_value_facts_equal(remainder, nonnegative));
  const auto two = loom_value_facts_exact_i64(2);
  loom_value_facts_divui(&high_bits, &two, 64, &quotient);
  EXPECT_EQ(quotient.range_lo, INT64_C(1) << 62);
  EXPECT_EQ(quotient.range_hi, INT64_MAX);

  const auto one = loom_value_facts_exact_i64(1);
  for (const auto interval : {nonnegative, high_bits}) {
    const auto preimage = loom_value_facts_quotient_preimage(
        LOOM_VALUE_FACTS_QUOTIENT_DIVUI, 64, interval, one);
    EXPECT_EQ(preimage.range_lo, interval.range_lo);
    EXPECT_EQ(preimage.range_hi, interval.range_hi);
  }
}

TEST(UnsignedDivisionTransfer, ExactNegativeRepresentationsUseDeclaredWidth) {
  const loom_value_facts_t dividend = loom_value_facts_exact_i64(-30);
  const loom_value_facts_t divisor = loom_value_facts_exact_i64(5);
  for (int32_t bit_count : {32, 64}) {
    const uint64_t dividend_bits = testing::RawIntegerBits(-30, bit_count);
    loom_value_facts_t quotient;
    loom_value_facts_divui(&dividend, &divisor, bit_count, &quotient);
    EXPECT_TRUE(loom_value_facts_is_exact(quotient));
    EXPECT_EQ(quotient.range_lo, testing::SignedIntegerRepresentation(
                                     dividend_bits / 5, bit_count));
    EXPECT_NE(quotient.known_divisor, 6);

    loom_value_facts_t remainder;
    loom_value_facts_remui(&dividend, &divisor, bit_count, &remainder);
    EXPECT_TRUE(loom_value_facts_is_exact(remainder));
    EXPECT_EQ(remainder.range_lo, testing::SignedIntegerRepresentation(
                                      dividend_bits % 5, bit_count));
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
  loom_value_facts_t divisor = testing::ExactIntegerBits(UINT64_C(0x80), 8);
  loom_value_facts_t quotient;
  loom_value_facts_divui(&dividend, &divisor, 8, &quotient);
  EXPECT_EQ(quotient.range_lo, 0);
  EXPECT_EQ(quotient.range_hi, 1);
  loom_value_facts_t remainder;
  loom_value_facts_remui(&dividend, &divisor, 8, &remainder);
  EXPECT_EQ(remainder.range_lo, 0);
  EXPECT_EQ(remainder.range_hi, 127);

  divisor = testing::ExactIntegerBits(UINT64_C(0xFF), 8);
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

TEST(UnsignedDivisionTransfer, IdentityRetainsDividendFacts) {
  for (int32_t bit_count : {8, 16, 32, 64}) {
    loom_value_facts_t dividend = loom_value_facts_make(12, 60, 6);
    loom_value_facts_mark_uniform_at_scope(
        &dividend, LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP);
    loom_value_facts_t divisor = loom_value_facts_make(61, 100, 1);
    loom_value_facts_mark_lane_varying(&divisor);
    EXPECT_TRUE(
        loom_value_facts_remui_is_identity(dividend, divisor, bit_count));
    loom_value_facts_t remainder;
    loom_value_facts_remui(&dividend, &divisor, bit_count, &remainder);
    EXPECT_TRUE(loom_value_facts_equal(remainder, dividend));

    // The negative representation is contiguous above the unsigned sign bit.
    dividend = loom_value_facts_make(-32, -16, 4);
    divisor = loom_value_facts_make(-15, -1, 1);
    EXPECT_TRUE(
        loom_value_facts_remui_is_identity(dividend, divisor, bit_count));
    loom_value_facts_remui(&dividend, &divisor, bit_count, &remainder);
    EXPECT_TRUE(loom_value_facts_equal(remainder, dividend));
  }
}

TEST(UnsignedDivisionTransfer, IdentityExcludesZeroOverlapAndWrapping) {
  for (int32_t bit_count : {1, 8, 16, 32, 64}) {
    const loom_value_facts_t zero = loom_value_facts_exact_i64(0);
    const loom_value_facts_t one = loom_value_facts_exact_i64(1);
    EXPECT_FALSE(loom_value_facts_remui_is_identity(zero, zero, bit_count));
    EXPECT_TRUE(loom_value_facts_remui_is_identity(zero, one, bit_count));
    EXPECT_FALSE(loom_value_facts_remui_is_identity(one, one, bit_count));
    EXPECT_FALSE(loom_value_facts_remui_is_identity(
        one, loom_value_facts_make(0, 1, 1), bit_count));
    EXPECT_FALSE(loom_value_facts_remui_is_identity(loom_value_facts_unknown(),
                                                    one, bit_count));
  }
  EXPECT_FALSE(loom_value_facts_remui_is_identity(
      loom_value_facts_make(0, 255, 1), loom_value_facts_exact_i64(257), 8));
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

TEST(RemsiTransfer, DynamicDivisorsPreserveDividendSign) {
  for (const auto divisor :
       {loom_value_facts_make(1, 64, 1), loom_value_facts_make(-64, -1, 1),
        loom_value_facts_make(INT64_MIN, -1, 1)}) {
    const int64_t bound = divisor.range_lo == INT64_MIN ? INT64_MAX : 63;
    for (const auto dividend :
         {loom_value_facts_make(0, INT64_MAX, 1),
          loom_value_facts_make(INT64_MIN, 0, 1), loom_value_facts_unknown()}) {
      loom_value_facts_t remainder;
      loom_value_facts_remsi(&dividend, &divisor, &remainder);
      EXPECT_EQ(remainder.range_lo, dividend.range_lo >= 0 ? 0 : -bound);
      EXPECT_EQ(remainder.range_hi, dividend.range_hi <= 0 ? 0 : bound);
    }
  }
}

TEST(RemsiTransfer, RepresentativeIntervalsContainEveryResult) {
  constexpr int64_t kEndpoints[] = {-8, -5, -1, 0, 1, 3, 7};
  for (int64_t dividend_lower : kEndpoints) {
    for (int64_t dividend_upper : kEndpoints) {
      if (dividend_lower > dividend_upper) {
        continue;
      }
      const auto dividend =
          loom_value_facts_make(dividend_lower, dividend_upper, 1);
      for (int64_t divisor_lower : kEndpoints) {
        for (int64_t divisor_upper : kEndpoints) {
          if (divisor_lower > divisor_upper) {
            continue;
          }
          const auto divisor =
              loom_value_facts_make(divisor_lower, divisor_upper, 1);
          SCOPED_TRACE(::testing::Message()
                       << "dividend=[" << dividend_lower << ","
                       << dividend_upper << "] divisor=[" << divisor_lower
                       << "," << divisor_upper << "]");
          loom_value_facts_t remainder;
          loom_value_facts_remsi(&dividend, &divisor, &remainder);
          if (divisor_lower > 0 || divisor_upper < 0) {
            if (dividend_lower >= 0) {
              EXPECT_GE(remainder.range_lo, 0);
            }
            if (dividend_upper <= 0) {
              EXPECT_LE(remainder.range_hi, 0);
            }
          }
          for (int64_t lhs = dividend_lower; lhs <= dividend_upper; ++lhs) {
            for (int64_t rhs = divisor_lower; rhs <= divisor_upper; ++rhs) {
              if (rhs != 0) {
                ExpectContains(remainder, lhs % rhs);
              }
            }
          }
        }
      }
    }
  }
}

TEST(QuotientPreimage, WideCarrierBoundaryPoints) {
  for (int32_t bit_count : {16, 32, 64}) {
    const auto domain = loom_value_facts_make_signed_bit_count_range(bit_count);
    const int64_t values[] = {domain.range_lo,
                              domain.range_lo + 1,
                              domain.range_lo / 2,
                              -63,
                              -3,
                              -1,
                              0,
                              1,
                              2,
                              3,
                              31,
                              63,
                              domain.range_hi / 2,
                              domain.range_hi - 1,
                              domain.range_hi};
    for (auto kind : testing::kQuotientKinds) {
      for (int64_t scale : values) {
        for (int64_t dividend : values) {
          int64_t quotient = 0;
          const bool constrained = testing::EvaluateQuotient(
              kind, bit_count, dividend, scale, &quotient);
          for (int64_t lower : {domain.range_lo, quotient}) {
            for (int64_t upper : {quotient, domain.range_hi}) {
              const auto preimage = loom_value_facts_quotient_preimage(
                  kind, bit_count, loom_value_facts_make(lower, upper, 1),
                  loom_value_facts_exact_i64(scale));
              SCOPED_TRACE(::testing::Message()
                           << "kind=" << kind << " width=" << bit_count
                           << " scale=" << scale << " dividend=" << dividend
                           << " constrained=" << constrained);
              ExpectContains(preimage, dividend);
            }
          }
        }
      }
    }
  }
}

TEST(QuotientPreimage, RetainsRoundingResiduesAndWideEndpoints) {
  struct Case {
    // Quotient operation semantics.
    loom_value_facts_quotient_kind_t kind;
    // Inclusive quotient lower bound.
    int64_t lower_quotient;
    // Inclusive quotient upper bound.
    int64_t upper_quotient;
    // Exact divisor or shift amount.
    int64_t scale;
    // Expected inclusive dividend lower bound.
    int64_t lower;
    // Expected inclusive dividend upper bound.
    int64_t upper;
  };
  const Case cases[] = {
      {LOOM_VALUE_FACTS_QUOTIENT_SHRUI, 0, 72, 5, 0, 2335},
      {LOOM_VALUE_FACTS_QUOTIENT_DIVSI, 0, 0, 32, -31, 31},
      {LOOM_VALUE_FACTS_QUOTIENT_FLOORDIVSI, 0, 0, 32, 0, 31},
      {LOOM_VALUE_FACTS_QUOTIENT_CEILDIVSI, 0, 0, 32, -31, 0},
      {LOOM_VALUE_FACTS_QUOTIENT_FLOORDIVSI, -2, -1, -32, 1, 64},
      {LOOM_VALUE_FACTS_QUOTIENT_CEILDIVSI, -2, -1, -32, 32, 95},
      {LOOM_VALUE_FACTS_QUOTIENT_DIVUI, -1, -1, 1, -1, -1},
      {LOOM_VALUE_FACTS_QUOTIENT_CEILDIVUI, -1, -1, 1, -1, -1},
      {LOOM_VALUE_FACTS_QUOTIENT_SHRUI, 1, 1, 63, INT64_MIN, -1},
      {LOOM_VALUE_FACTS_QUOTIENT_SHRSI, -1, -1, 63, INT64_MIN, -1},
      {LOOM_VALUE_FACTS_QUOTIENT_DIVSI, INT64_MAX, INT64_MAX, 1, INT64_MAX,
       INT64_MAX},
      {LOOM_VALUE_FACTS_QUOTIENT_DIVSI, INT64_MIN, INT64_MIN, 1, INT64_MIN,
       INT64_MIN},
      {LOOM_VALUE_FACTS_QUOTIENT_DIVSI, 1, 1, INT64_MIN, INT64_MIN, INT64_MIN},
  };
  for (auto test : cases) {
    const auto actual = loom_value_facts_quotient_preimage(
        test.kind, 64,
        loom_value_facts_make(test.lower_quotient, test.upper_quotient, 1),
        loom_value_facts_exact_i64(test.scale));
    EXPECT_EQ(actual.range_lo, test.lower);
    EXPECT_EQ(actual.range_hi, test.upper);
  }
}

TEST(DivisionTransfer, PositivePowersAcrossCarriersAndAliases) {
  for (int32_t bit_count : {8, 16, 32, 64}) {
    const int32_t exponents[] = {0, 1, 2, bit_count - 3, bit_count - 2};
    for (int32_t a : exponents) {
      for (int32_t b : exponents) {
        auto numerator = loom_value_facts_make(
            INT64_C(1) << a, INT64_C(1) << (bit_count - 2), 1);
        auto denominator = loom_value_facts_make(1, INT64_C(1) << b, 1);
        numerator.flags |= LOOM_VALUE_FACT_POWER_OF_TWO;
        denominator.flags |= LOOM_VALUE_FACT_POWER_OF_TWO;
        for (bool unsigned_quotient : {false, true}) {
          for (int output_index = 0; output_index < 3; ++output_index) {
            loom_value_facts_t operands[] = {numerator, denominator,
                                             loom_value_facts_unknown()};
            if (unsigned_quotient) {
              loom_value_facts_divui(&operands[0], &operands[1], bit_count,
                                     &operands[output_index]);
            } else {
              loom_value_facts_divsi(&operands[0], &operands[1],
                                     &operands[output_index]);
            }
            EXPECT_EQ(loom_value_facts_is_power_of_two(operands[output_index]),
                      a >= b);
            for (int32_t lhs_exponent : exponents) {
              if (lhs_exponent < a) {
                continue;
              }
              for (int32_t rhs_exponent : exponents) {
                if (rhs_exponent > b) {
                  continue;
                }
                ExpectContains(operands[output_index],
                               (INT64_C(1) << lhs_exponent) /
                                   (INT64_C(1) << rhs_exponent));
              }
            }
          }
        }
      }
    }
  }
  const auto unknown_power = loom_value_facts_make(1, 4, 1);
  auto powers = loom_value_facts_make(4, 16, 1);
  powers.flags |= LOOM_VALUE_FACT_POWER_OF_TWO;
  loom_value_facts_t result;
  loom_value_facts_divsi(&powers, &unknown_power, &result);
  EXPECT_FALSE(loom_value_facts_is_power_of_two(result));
  loom_value_facts_divui(&powers, &unknown_power, 8, &result);
  EXPECT_FALSE(loom_value_facts_is_power_of_two(result));
}

}  // namespace
}  // namespace loom
