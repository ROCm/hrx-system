// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"
#include "loom/ir/facts.h"

namespace loom {
namespace {

using ExtremaTransfer = void (*)(const loom_value_facts_t*,
                                 const loom_value_facts_t*, int32_t,
                                 loom_value_facts_t*);
static const ExtremaTransfer kExtremaTransfers[] = {
    loom_value_facts_minsi,
    loom_value_facts_maxsi,
    [](const loom_value_facts_t* lhs, const loom_value_facts_t* rhs, int32_t,
       loom_value_facts_t* out) { loom_value_facts_minui(lhs, rhs, out); },
    [](const loom_value_facts_t* lhs, const loom_value_facts_t* rhs, int32_t,
       loom_value_facts_t* out) { loom_value_facts_maxui(lhs, rhs, out); },
};

//===----------------------------------------------------------------------===//
// Transfer functions: minsi / maxsi
//===----------------------------------------------------------------------===//

TEST(MinsiTransfer, ExactValues) {
  loom_value_facts_t a = loom_value_facts_exact_i64(5);
  loom_value_facts_t b = loom_value_facts_exact_i64(3);
  loom_value_facts_t out;
  loom_value_facts_minsi(&a, &b, 64, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, 3);
}

TEST(MaxsiTransfer, Ranges) {
  loom_value_facts_t a = loom_value_facts_make(1, 10, 1);
  loom_value_facts_t b = loom_value_facts_make(5, 20, 1);
  loom_value_facts_t out;
  loom_value_facts_maxsi(&a, &b, 64, &out);
  EXPECT_EQ(out.range_lo, 5);
  EXPECT_EQ(out.range_hi, 20);
}

TEST(IntegerExtremaTransfer, DeclaredWidthAndAliasing) {
  struct Case {
    // Declared integer storage width.
    int32_t bit_count;
    // Left input in its declared signed or Boolean fact representation.
    int64_t lhs;
    // Right input in the same declared width.
    int64_t rhs;
    // Signed min/max followed by unsigned min/max, in fact representation.
    int64_t expected[4];
  };
  const Case cases[] = {
      {1, 0, 1, {1, 0, 0, 1}},
      {1, 1, 0, {1, 0, 0, 1}},
      {1, 1, 1, {1, 1, 1, 1}},
      {8, -128, 127, {-128, 127, 127, -128}},
      {8, -128, -1, {-128, -1, -128, -1}},
      {16, -1, 0, {-1, 0, 0, -1}},
      {32, INT32_MIN, INT32_MAX, {INT32_MIN, INT32_MAX, INT32_MAX, INT32_MIN}},
      {64, INT64_MIN, INT64_MAX, {INT64_MIN, INT64_MAX, INT64_MAX, INT64_MIN}},
      {64, 0, -1, {-1, 0, 0, -1}},
      {64, INT64_MIN, -1, {INT64_MIN, -1, INT64_MIN, -1}},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.bit_count);
    SCOPED_TRACE(test_case.lhs);
    SCOPED_TRACE(test_case.rhs);
    for (size_t operation = 0; operation < IREE_ARRAYSIZE(kExtremaTransfers);
         ++operation) {
      SCOPED_TRACE(operation);
      for (int output_index = 0; output_index < 3; ++output_index) {
        loom_value_facts_t facts[] = {
            loom_value_facts_exact_i64(test_case.lhs),
            loom_value_facts_exact_i64(test_case.rhs),
            loom_value_facts_unknown(),
        };
        kExtremaTransfers[operation](&facts[0], &facts[1], test_case.bit_count,
                                     &facts[output_index]);
        EXPECT_TRUE(loom_value_facts_is_exact(facts[output_index]));
        EXPECT_EQ(facts[output_index].range_lo, test_case.expected[operation]);
      }
    }
  }
}

TEST(IntegerExtremaTransfer, BooleanAndNarrowRanges) {
  loom_value_facts_t boolean = loom_value_facts_make(0, 1, 1);
  loom_value_facts_mark_lane_varying(&boolean);
  loom_value_facts_t zero = loom_value_facts_exact_i64(0);
  loom_value_facts_t one = loom_value_facts_exact_i64(1);
  loom_value_facts_t result;
  loom_value_facts_minsi(&boolean, &zero, 1, &result);
  EXPECT_EQ(result.range_lo, 0);
  EXPECT_EQ(result.range_hi, 1);
  EXPECT_TRUE(loom_value_facts_is_lane_varying(result));
  loom_value_facts_maxsi(&boolean, &one, 1, &result);
  EXPECT_EQ(result.range_lo, 0);
  EXPECT_EQ(result.range_hi, 1);
  EXPECT_TRUE(loom_value_facts_is_lane_varying(result));

  loom_value_facts_t negative = loom_value_facts_make(-128, -124, 2);
  loom_value_facts_t positive = loom_value_facts_make(2, 10, 2);
  loom_value_facts_minsi(&negative, &positive, 8, &result);
  EXPECT_EQ(result.range_lo, -128);
  EXPECT_EQ(result.range_hi, -124);
  EXPECT_EQ(result.known_divisor, 2);
  loom_value_facts_maxsi(&negative, &positive, 8, &result);
  EXPECT_EQ(result.range_lo, 2);
  EXPECT_EQ(result.range_hi, 10);
  EXPECT_EQ(result.known_divisor, 2);
}

TEST(IntegerExtremaTransfer, CommonPredicatesAndZeroDivisibility) {
  auto left = loom_value_facts_make(2, 8, 2);
  left.flags |= LOOM_VALUE_FACT_POWER_OF_TWO;
  auto right = loom_value_facts_make(4, 16, 4);
  right.flags |= LOOM_VALUE_FACT_POWER_OF_TWO;
  for (size_t operation = 0; operation < IREE_ARRAYSIZE(kExtremaTransfers);
       ++operation) {
    auto transfer = kExtremaTransfers[operation];
    loom_value_facts_t result;
    transfer(&left, &right, 8, &result);
    EXPECT_TRUE(loom_value_facts_is_power_of_two(result));
    EXPECT_EQ(result.known_divisor, 2);
    auto unproven = right;
    unproven.flags &= ~LOOM_VALUE_FACT_POWER_OF_TWO;
    transfer(&left, &unproven, 8, &result);
    EXPECT_FALSE(loom_value_facts_is_power_of_two(result));

    auto nonzero = loom_value_facts_make(-8, 8, 4);
    nonzero.flags |= LOOM_VALUE_FACT_NON_ZERO;
    transfer(&nonzero, &left, 8, &result);
    EXPECT_TRUE(loom_value_facts_is_non_zero(result));
    const auto zero = loom_value_facts_exact_i64(0);
    transfer(&nonzero, &zero, 8, &result);
    EXPECT_TRUE(result.known_divisor == 4 || loom_value_facts_is_zero(result));
    if (operation != 3) {
      EXPECT_FALSE(loom_value_facts_is_non_zero(result));
    }
  }
}

TEST(IntegerExtremaTransfer, UnsignedSignPartitionsRetainSelectedFacts) {
  auto low = loom_value_facts_make(4, 16, 4);
  low.flags |= LOOM_VALUE_FACT_POWER_OF_TWO;
  loom_value_facts_mark_workgroup_uniform(&low);
  auto high = loom_value_facts_make(-16, -8, 8);
  loom_value_facts_mark_lane_varying(&high);
  loom_value_facts_t result;
  loom_value_facts_minui(&low, &high, &result);
  EXPECT_TRUE(loom_value_facts_equal(result, low));
  loom_value_facts_maxui(&low, &high, &result);
  EXPECT_TRUE(loom_value_facts_equal(result, high));
  loom_value_facts_minui(&high, &low, &high);
  EXPECT_TRUE(loom_value_facts_equal(high, low));
}

TEST(IntegerExtremaTransfer, MaskAndAliasingContracts) {
  auto mask = loom_value_facts_make(0, 63, 1);
  loom_value_facts_mark_subgroup_lane_mask(&mask);
  const auto zero = loom_value_facts_exact_i64(0);
  const auto ordinary = loom_value_facts_make(0, 63, 1);
  for (auto transfer : kExtremaTransfers) {
    for (const auto& right : {mask, zero, ordinary}) {
      loom_value_facts_t expected;
      transfer(&mask, &right, 8, &expected);
      EXPECT_EQ(loom_value_facts_is_subgroup_lane_mask(expected),
                loom_value_facts_is_subgroup_lane_mask(right) ||
                    loom_value_facts_is_zero(right));
      auto left_alias = mask;
      transfer(&left_alias, &right, 8, &left_alias);
      EXPECT_TRUE(loom_value_facts_equal(left_alias, expected));
      auto right_alias = right;
      transfer(&mask, &right_alias, 8, &right_alias);
      EXPECT_TRUE(loom_value_facts_equal(right_alias, expected));
    }
  }
}

// A four-bit carrier enumerates every concrete pair admitted by each abstract
// input. The oracle compares values directly, independently of interval joins.
TEST(IntegerExtremaTransfer, EveryConcreteValueInSmallSignedCarrier) {
  std::vector<loom_value_facts_t> inputs = {
      loom_value_facts_make(-8, 7, 1), loom_value_facts_make(-8, -1, 1),
      loom_value_facts_make(0, 7, 1),  loom_value_facts_make(1, 7, 1),
      loom_value_facts_make(-4, 4, 2), loom_value_facts_make(-4, -2, 2),
      loom_value_facts_make(2, 6, 2),  loom_value_facts_exact_i64(-8),
      loom_value_facts_exact_i64(-1),  loom_value_facts_exact_i64(0),
      loom_value_facts_exact_i64(1),   loom_value_facts_exact_i64(3),
      loom_value_facts_exact_i64(7)};
  auto nonzero = inputs[0];
  nonzero.flags |= LOOM_VALUE_FACT_NON_ZERO;
  inputs.push_back(nonzero);
  auto powers = loom_value_facts_make(1, 4, 1);
  powers.flags |= LOOM_VALUE_FACT_POWER_OF_TWO;
  inputs.push_back(powers);
  auto contains = [](const loom_value_facts_t& facts, int64_t value) {
    return value >= facts.range_lo && value <= facts.range_hi &&
           value % facts.known_divisor == 0 &&
           (!loom_value_facts_is_non_zero(facts) || value != 0) &&
           (!loom_value_facts_is_power_of_two(facts) ||
            (value > 0 && (value & (value - 1)) == 0));
  };
  for (const auto& left : inputs) {
    for (const auto& right : inputs) {
      for (size_t operation = 0; operation < IREE_ARRAYSIZE(kExtremaTransfers);
           ++operation) {
        SCOPED_TRACE(operation);
        SCOPED_TRACE(left.range_lo);
        SCOPED_TRACE(left.range_hi);
        SCOPED_TRACE(right.range_lo);
        SCOPED_TRACE(right.range_hi);
        loom_value_facts_t result;
        kExtremaTransfers[operation](&left, &right, 4, &result);
        for (int64_t a = -8; a <= 7; ++a) {
          if (!contains(left, a)) {
            continue;
          }
          for (int64_t b = -8; b <= 7; ++b) {
            if (!contains(right, b)) {
              continue;
            }
            const bool less = operation < 2
                                  ? a < b
                                  : (static_cast<uint64_t>(a) & 15) <
                                        (static_cast<uint64_t>(b) & 15);
            const int64_t expected = less == (operation % 2 == 0) ? a : b;
            EXPECT_TRUE(contains(result, expected)) << a << ", " << b;
          }
        }
      }
    }
  }
}

}  // namespace
}  // namespace loom
