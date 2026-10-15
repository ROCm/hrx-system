// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <climits>
#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"
#include "loom/ir/facts.h"

namespace loom {
namespace {

//===----------------------------------------------------------------------===//
// Transfer functions: shifts
//===----------------------------------------------------------------------===//

TEST(ShliTransfer, ExactShift) {
  loom_value_facts_t a = loom_value_facts_exact_i64(5);
  loom_value_facts_t b = loom_value_facts_exact_i64(3);
  loom_value_facts_t out;
  loom_value_facts_shli(&a, &b, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, 40);
}

TEST(ShliTransfer, DivisorMultiplied) {
  // Shift left by 4: divisor *= 16.
  loom_value_facts_t a = loom_value_facts_make(1, 10, 2);
  loom_value_facts_t b = loom_value_facts_exact_i64(4);
  loom_value_facts_t out;
  loom_value_facts_shli(&a, &b, &out);
  EXPECT_EQ(out.range_lo, 16);
  EXPECT_EQ(out.range_hi, 160);
  EXPECT_EQ(out.known_divisor, 32);  // 2 * 16 = 32.
}

TEST(ShliTransfer, NonExactShiftAmount) {
  // Non-exact shift amount → unknown.
  loom_value_facts_t a = loom_value_facts_exact_i64(5);
  loom_value_facts_t b = loom_value_facts_make(1, 3, 1);
  loom_value_facts_t out;
  loom_value_facts_shli(&a, &b, &out);
  EXPECT_TRUE(loom_value_facts_is_unknown(out));
}

TEST(ShliTransfer, SignBitShift) {
  loom_value_facts_t shift = loom_value_facts_exact_i64(63);
  loom_value_facts_t out;

  loom_value_facts_t odd = loom_value_facts_exact_i64(1);
  loom_value_facts_shli(&odd, &shift, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, INT64_MIN);

  loom_value_facts_t even = loom_value_facts_exact_i64(2);
  loom_value_facts_shli(&even, &shift, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, 0);
}

TEST(ShruiTransfer, SignBitShift) {
  loom_value_facts_t shift = loom_value_facts_exact_i64(63);
  loom_value_facts_t out;

  loom_value_facts_t non_negative = loom_value_facts_make(0, INT64_MAX, 1);
  loom_value_facts_shrui(&non_negative, &shift, 64, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, 0);

  loom_value_facts_t negative = loom_value_facts_make(INT64_MIN, -1, 1);
  loom_value_facts_shrui(&negative, &shift, 64, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, 1);

  loom_value_facts_t unknown = loom_value_facts_unknown();
  loom_value_facts_shrui(&unknown, &shift, 64, &out);
  EXPECT_EQ(out.range_lo, 0);
  EXPECT_EQ(out.range_hi, 1);
}

TEST(ShruiTransfer, ByteRangesContainEveryShiftedValue) {
  for (int64_t divisor : {1, 2, 3, 6, 16}) {
    for (int64_t lo : {-128, -97, -32, -1, 0, 17, 96, 127}) {
      for (int64_t hi : {-128, -32, -1, 0, 31, 127}) {
        if (lo > hi) {
          continue;
        }
        loom_value_facts_t source = loom_value_facts_make(lo, hi, divisor);
        for (int64_t count = 0; count < 8; ++count) {
          const loom_value_facts_t shift = loom_value_facts_exact_i64(count);
          loom_value_facts_t result;
          loom_value_facts_shrui(&source, &shift, 8, &result);
          for (int64_t value = lo; value <= hi; ++value) {
            if (value % source.known_divisor != 0) {
              continue;
            }
            const int64_t expected =
                count == 0 ? value
                           : ((value + 256) % 256) / (INT64_C(1) << count);
            EXPECT_LE(result.range_lo, expected);
            EXPECT_GE(result.range_hi, expected);
            EXPECT_EQ(expected % result.known_divisor, 0);
          }
        }
      }
    }
  }
}

TEST(ShruiTransfer, ExactBytesAndWidthBounds) {
  for (int64_t value = -128; value <= 127; ++value) {
    const loom_value_facts_t source = loom_value_facts_exact_i64(value);
    for (int64_t count = 0; count < 8; ++count) {
      const loom_value_facts_t shift = loom_value_facts_exact_i64(count);
      loom_value_facts_t result;
      loom_value_facts_shrui(&source, &shift, 8, &result);
      const int64_t expected =
          count == 0 ? value : ((value + 256) % 256) / (INT64_C(1) << count);
      EXPECT_TRUE(loom_value_facts_is_exact(result));
      EXPECT_EQ(result.range_lo, expected);
    }
  }
  for (int32_t width : {1, 2, 4, 8, 16, 32, 64}) {
    const loom_value_facts_t source = loom_value_facts_unknown();
    for (int64_t count = 0; count < width; ++count) {
      const loom_value_facts_t shift = loom_value_facts_exact_i64(count);
      loom_value_facts_t result;
      loom_value_facts_shrui(&source, &shift, width, &result);
      const bool signed_range = count == 0 && width > 1;
      const int64_t maximum =
          (int64_t)(UINT64_MAX >> (64 - width + (signed_range ? 1 : count)));
      EXPECT_EQ(result.range_lo, signed_range ? -maximum - 1 : 0);
      EXPECT_EQ(result.range_hi, maximum);
    }
  }
}

TEST(ShruiTransfer, DistributionAndUnknownAmounts) {
  loom_value_facts_t source = loom_value_facts_make(-128, 127, 16);
  loom_value_facts_mark_lane_varying(&source);
  const loom_value_facts_t shift = loom_value_facts_exact_i64(3);
  loom_value_facts_t result;
  loom_value_facts_shrui(&source, &shift, 8, &result);
  EXPECT_EQ(result.range_lo, 0);
  EXPECT_EQ(result.range_hi, 31);
  EXPECT_EQ(result.known_divisor, 2);
  EXPECT_TRUE(loom_value_facts_is_lane_varying(result));
  EXPECT_FALSE(loom_value_facts_is_subgroup_uniform(result));

  for (loom_value_facts_t amount :
       {loom_value_facts_unknown(), loom_value_facts_make(1, 7, 1),
        loom_value_facts_exact_i64(8), loom_value_facts_exact_i64(-1)}) {
    loom_value_facts_shrui(&source, &amount, 8, &result);
    EXPECT_EQ(result.range_lo, INT64_MIN);
    EXPECT_EQ(result.range_hi, INT64_MAX);
    EXPECT_TRUE(loom_value_facts_is_lane_varying(result));
  }
}

TEST(ShrsiTransfer, SignBitShift) {
  loom_value_facts_t shift = loom_value_facts_exact_i64(63);
  loom_value_facts_t out;

  loom_value_facts_t non_negative = loom_value_facts_make(0, INT64_MAX, 1);
  loom_value_facts_shrsi(&non_negative, &shift, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, 0);

  loom_value_facts_t negative = loom_value_facts_make(INT64_MIN, -1, 1);
  loom_value_facts_shrsi(&negative, &shift, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, -1);

  loom_value_facts_t unknown = loom_value_facts_unknown();
  loom_value_facts_shrsi(&unknown, &shift, &out);
  EXPECT_EQ(out.range_lo, -1);
  EXPECT_EQ(out.range_hi, 0);
}

TEST(ShrsiTransfer, NegativeValuesUseArithmeticSemantics) {
  loom_value_facts_t shift = loom_value_facts_exact_i64(1);
  loom_value_facts_t out;

  loom_value_facts_t exact = loom_value_facts_exact_i64(-5);
  loom_value_facts_shrsi(&exact, &shift, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, -3);

  loom_value_facts_t minimum = loom_value_facts_exact_i64(INT64_MIN);
  loom_value_facts_shrsi(&minimum, &shift, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, INT64_MIN / 2);

  loom_value_facts_t range = loom_value_facts_make(-9, -5, 1);
  loom_value_facts_shrsi(&range, &shift, &out);
  EXPECT_EQ(out.range_lo, -5);
  EXPECT_EQ(out.range_hi, -3);
}

//===----------------------------------------------------------------------===//
// Transfer functions: bitwise
//===----------------------------------------------------------------------===//

TEST(AndiTransfer, ExactValues) {
  loom_value_facts_t a = loom_value_facts_exact_i64(0xFF);
  loom_value_facts_t b = loom_value_facts_exact_i64(0x0F);
  loom_value_facts_t out;
  loom_value_facts_andi(&a, &b, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, 0x0F);
}

TEST(AndiTransfer, MaskBounds) {
  // AND with a non-negative mask bounds the result.
  loom_value_facts_t a = loom_value_facts_make(0, 1000, 1);
  loom_value_facts_t b = loom_value_facts_exact_i64(0xFF);
  loom_value_facts_t out;
  loom_value_facts_andi(&a, &b, &out);
  EXPECT_EQ(out.range_lo, 0);
  EXPECT_EQ(out.range_hi, 0xFF);
}

TEST(AndiTransfer, MaskPreservesOnlyPowerOfTwoDivisibility) {
  loom_value_facts_t input = loom_value_facts_make(3, 9, 3);
  loom_value_facts_t mask = loom_value_facts_exact_i64(6);
  loom_value_facts_t output;
  loom_value_facts_andi(&input, &mask, &output);
  // Three is a possible input and 3 & 6 is two, not a multiple of three.
  EXPECT_EQ(output.known_divisor, 2);

  input = loom_value_facts_make(0, 96, 24);
  mask = loom_value_facts_make(0, 96, 12);
  loom_value_facts_andi(&input, &mask, &output);
  EXPECT_EQ(output.known_divisor, 8);
}

TEST(AndiTransfer, NegativeMaskPreservesNonnegativeBound) {
  loom_value_facts_t input = loom_value_facts_make(0, 31, 1);
  loom_value_facts_t mask = loom_value_facts_exact_i64(-8);
  loom_value_facts_t output;
  loom_value_facts_andi(&input, &mask, &output);
  EXPECT_EQ(output.range_lo, 0);
  EXPECT_EQ(output.range_hi, 24);
  EXPECT_EQ(output.known_divisor, 8);
}

TEST(AndiTransfer, LowMasksEliminateAlignedValues) {
  for (int shift = 1; shift < 63; ++shift) {
    const int64_t alignment = INT64_C(1) << shift;
    loom_value_facts_t aligned =
        loom_value_facts_make(INT64_MIN, INT64_MAX, alignment);
    loom_value_facts_mark_lane_varying(&aligned);
    const loom_value_facts_t mask = loom_value_facts_exact_i64(alignment - 1);
    loom_value_facts_t result;
    loom_value_facts_andi(&aligned, &mask, &result);
    EXPECT_TRUE(loom_value_facts_is_zero(result));
    EXPECT_TRUE(loom_value_facts_is_cluster_uniform(result));
    loom_value_facts_andi(&mask, &aligned, &result);
    EXPECT_TRUE(loom_value_facts_is_zero(result));

    // Including the first possible bit permits a nonzero result.
    const loom_value_facts_t inclusive_mask =
        loom_value_facts_exact_i64(alignment);
    loom_value_facts_andi(&aligned, &inclusive_mask, &result);
    EXPECT_EQ(result.range_lo, 0);
    EXPECT_EQ(result.range_hi, alignment);
    EXPECT_FALSE(loom_value_facts_is_exact(result));
  }
}

TEST(AndiTransfer, SignedBoundaryMasksSupportInPlaceTransfer) {
  for (int64_t mask : {INT64_MIN, INT64_MIN + 1, INT64_MAX, INT64_C(-1),
                       INT64_C(0), INT64_C(1) << 62}) {
    const loom_value_facts_t mask_facts = loom_value_facts_exact_i64(mask);
    const loom_value_facts_t input_facts = loom_value_facts_unknown();
    loom_value_facts_t output;
    loom_value_facts_andi(&input_facts, &mask_facts, &output);
    ASSERT_GE(output.known_divisor, 1);
    for (int64_t value : {INT64_MIN, INT64_MIN + 1, INT64_C(-9), INT64_C(-1),
                          INT64_C(0), INT64_C(8), INT64_MAX}) {
      const int64_t result = value & mask;
      ASSERT_LE(output.range_lo, result);
      ASSERT_GE(output.range_hi, result);
      ASSERT_EQ(result % output.known_divisor, 0);
    }
    loom_value_facts_t in_place_input = input_facts;
    loom_value_facts_andi(&in_place_input, &mask_facts, &in_place_input);
    EXPECT_EQ(in_place_input.range_lo, output.range_lo);
    EXPECT_EQ(in_place_input.range_hi, output.range_hi);
    EXPECT_EQ(in_place_input.known_divisor, output.known_divisor);
    EXPECT_EQ(in_place_input.flags, output.flags);
    loom_value_facts_t in_place_mask = mask_facts;
    loom_value_facts_andi(&input_facts, &in_place_mask, &in_place_mask);
    EXPECT_EQ(in_place_mask.range_lo, output.range_lo);
    EXPECT_EQ(in_place_mask.range_hi, output.range_hi);
    EXPECT_EQ(in_place_mask.known_divisor, output.known_divisor);
    EXPECT_EQ(in_place_mask.flags, output.flags);
  }
}

TEST(XoriTransfer, SelfCancel) {
  loom_value_facts_t a = loom_value_facts_exact_i64(42);
  loom_value_facts_t out;
  loom_value_facts_xori(&a, &a, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, 0);
}

TEST(XoriTransfer, NonNegativeRangeStaysWithinOperandBitWidth) {
  loom_value_facts_t a = loom_value_facts_make(0, 63, 1);
  loom_value_facts_t b = loom_value_facts_make(0, 3, 1);
  loom_value_facts_t out;
  loom_value_facts_xori(&a, &b, &out);
  EXPECT_EQ(out.range_lo, 0);
  EXPECT_EQ(out.range_hi, 63);
  EXPECT_TRUE(loom_value_facts_fit_unsigned_bit_count(out, 6));
}

TEST(OriTransfer, NonNegativeCodebookIndex) {
  const loom_value_facts_t low = loom_value_facts_make(0, 255, 1);
  const loom_value_facts_t high = loom_value_facts_make(0, 256, 256);
  loom_value_facts_t result;
  loom_value_facts_ori(&low, &high, &result);
  EXPECT_EQ(result.range_lo, 0);
  EXPECT_EQ(result.range_hi, 511);
  EXPECT_TRUE(loom_value_facts_fit_unsigned_bit_count(result, 9));

  const loom_value_facts_t selected_high = loom_value_facts_exact_i64(256);
  loom_value_facts_ori(&low, &selected_high, &result);
  EXPECT_EQ(result.range_lo, 256);
  EXPECT_EQ(result.range_hi, 511);
  EXPECT_TRUE(loom_value_facts_is_non_zero(result));
}

TEST(OriTransfer, RangeEndpointsAreNotBitwiseBounds) {
  const loom_value_facts_t input = loom_value_facts_make(0, 8, 1);
  loom_value_facts_t result;
  loom_value_facts_ori(&input, &input, &result);
  // Each operand can be seven or eight, so OR can set all four low bits.
  EXPECT_EQ(result.range_lo, 0);
  EXPECT_EQ(result.range_hi, 15);
}

TEST(OriTransfer, RetainsRangeAndPredicateNonzeroProofs) {
  for (loom_value_facts_t nonzero :
       {loom_value_facts_exact_i64(1), loom_value_facts_exact_i64(-1),
        loom_value_facts_make(1, 255, 1),
        loom_value_facts_make(INT64_MIN, -1, 1),
        loom_value_facts_make(0, 255, 1), loom_value_facts_unknown()}) {
    nonzero.flags |= LOOM_VALUE_FACT_NON_ZERO;
    for (const loom_value_facts_t other :
         {loom_value_facts_unknown(), loom_value_facts_make(0, 256, 1)}) {
      loom_value_facts_t result;
      loom_value_facts_ori(&nonzero, &other, &result);
      EXPECT_TRUE(loom_value_facts_is_non_zero(result));
      loom_value_facts_ori(&other, &nonzero, &result);
      EXPECT_TRUE(loom_value_facts_is_non_zero(result));
    }
  }
  const loom_value_facts_t unknown = loom_value_facts_unknown();
  loom_value_facts_t result;
  loom_value_facts_ori(&unknown, &unknown, &result);
  EXPECT_FALSE(loom_value_facts_is_non_zero(result));
}

TEST(BitwiseTransfer, NonNegativeEnvelopeAtEveryBitWidth) {
  for (auto transfer : {loom_value_facts_ori, loom_value_facts_xori}) {
    for (uint32_t bit = 0; bit < 63; ++bit) {
      SCOPED_TRACE(bit);
      const int64_t upper = INT64_C(1) << bit;
      const int64_t envelope = (int64_t)((UINT64_C(1) << (bit + 1)) - 1);
      const loom_value_facts_t bounded = loom_value_facts_make(0, upper, 1);
      for (const loom_value_facts_t other :
           {loom_value_facts_exact_i64(0), bounded,
            loom_value_facts_make(0, envelope, 1)}) {
        loom_value_facts_t result;
        transfer(&bounded, &other, &result);
        EXPECT_EQ(result.range_lo, 0);
        EXPECT_EQ(result.range_hi, envelope);
        transfer(&other, &bounded, &result);
        EXPECT_EQ(result.range_lo, 0);
        EXPECT_EQ(result.range_hi, envelope);
      }
    }
  }
}

TEST(BitwiseTransfer, RetainsOnlyPowerOfTwoFactors) {
  for (auto transfer :
       {loom_value_facts_andi, loom_value_facts_ori, loom_value_facts_xori}) {
    const bool is_and = transfer == loom_value_facts_andi;
    for (int64_t lower : {INT64_C(-96), INT64_C(0)}) {
      const loom_value_facts_t left = loom_value_facts_make(lower, 96, 24);
      const loom_value_facts_t right = loom_value_facts_make(lower, 96, 12);
      loom_value_facts_t result;
      transfer(&left, &right, &result);
      EXPECT_EQ(result.known_divisor, is_and ? 8 : 4);
      EXPECT_EQ(result.range_hi % result.known_divisor, 0);
      if (lower == 0) {
        EXPECT_EQ(result.range_lo, 0);
        EXPECT_EQ(result.range_hi, is_and ? 96 : 124);
      }
    }
    const loom_value_facts_t odd = loom_value_facts_make(-9, 9, 3);
    loom_value_facts_t result;
    transfer(&odd, &odd, &result);
    EXPECT_EQ(result.known_divisor, 1);
  }
}

TEST(BitwiseTransfer, ExactZeroAndSignBitRetainRepresentableFactors) {
  const loom_value_facts_t aligned = loom_value_facts_make(-96, 96, 24);
  for (auto transfer :
       {loom_value_facts_andi, loom_value_facts_ori, loom_value_facts_xori}) {
    for (int64_t value : {INT64_C(0), INT64_MIN}) {
      const loom_value_facts_t exact = loom_value_facts_exact_i64(value);
      loom_value_facts_t result;
      transfer(&aligned, &exact, &result);
      if (transfer == loom_value_facts_andi && value == 0) {
        EXPECT_TRUE(loom_value_facts_is_zero(result));
      } else {
        EXPECT_EQ(result.known_divisor,
                  transfer == loom_value_facts_andi ? INT64_C(1) << 62 : 8);
      }
      for (int64_t input : {INT64_C(-96), INT64_C(0), INT64_C(96)}) {
        const int64_t expected =
            transfer == loom_value_facts_andi  ? input & value
            : transfer == loom_value_facts_ori ? input | value
                                               : input ^ value;
        EXPECT_LE(result.range_lo, expected);
        EXPECT_GE(result.range_hi, expected);
        EXPECT_EQ(expected % result.known_divisor, 0);
      }
    }
  }
}

TEST(BitwiseTransfer, AlignedEnvelopeAtEveryBitWidth) {
  for (auto transfer : {loom_value_facts_ori, loom_value_facts_xori}) {
    for (uint32_t bit = 1; bit < 63; ++bit) {
      for (uint32_t shift = 1; shift <= bit; ++shift) {
        SCOPED_TRACE(bit);
        SCOPED_TRACE(shift);
        const int64_t alignment = INT64_C(1) << shift;
        const int64_t upper = INT64_C(1) << bit;
        const int64_t envelope =
            (int64_t)((UINT64_C(1) << (bit + 1)) - alignment);
        const loom_value_facts_t bounded =
            loom_value_facts_make(0, upper, alignment);
        loom_value_facts_t result;
        transfer(&bounded, &bounded, &result);
        EXPECT_EQ(result.range_lo, 0);
        EXPECT_EQ(result.range_hi, envelope);
        EXPECT_EQ(result.known_divisor, alignment);
      }
    }
  }
}

TEST(BitwiseTransfer, SignedBoundaryConstantsFoldExactly) {
  for (int64_t lhs : {INT64_MIN, INT64_MIN + 1, INT64_C(-1), INT64_C(0),
                      INT64_C(1), INT64_C(1) << 62, INT64_MAX}) {
    const loom_value_facts_t left = loom_value_facts_exact_i64(lhs);
    for (int64_t rhs : {INT64_MIN, INT64_MIN + 1, INT64_C(-1), INT64_C(0),
                        INT64_C(1), INT64_C(1) << 62, INT64_MAX}) {
      const loom_value_facts_t right = loom_value_facts_exact_i64(rhs);
      loom_value_facts_t result;
      loom_value_facts_andi(&left, &right, &result);
      EXPECT_TRUE(loom_value_facts_equal(
          result, loom_value_facts_exact_i64(lhs & rhs)));
      loom_value_facts_ori(&left, &right, &result);
      EXPECT_TRUE(loom_value_facts_equal(
          result, loom_value_facts_exact_i64(lhs | rhs)));
      loom_value_facts_xori(&left, &right, &result);
      EXPECT_TRUE(loom_value_facts_equal(
          result, loom_value_facts_exact_i64(lhs ^ rhs)));
    }
  }
}

TEST(BitwiseTransfer, BoundedIntervalsAndDivisorsContainEveryConcreteResult) {
  std::vector<loom_value_facts_t> inputs;
  for (int64_t lower = -8; lower <= 8; ++lower) {
    for (int64_t upper = lower; upper <= 8; ++upper) {
      for (int64_t divisor = 1; divisor <= 8; ++divisor) {
        // Exact constructors derive their divisor from the value.
        if (lower == upper && divisor != 1) {
          continue;
        }
        bool has_value = false;
        for (int64_t value = lower; value <= upper; ++value) {
          has_value |= value % divisor == 0;
        }
        if (has_value) {
          inputs.push_back(loom_value_facts_make(lower, upper, divisor));
        }
      }
    }
  }
  for (const auto& left : inputs) {
    for (const auto& right : inputs) {
      loom_value_facts_t result_and;
      loom_value_facts_t result_or;
      loom_value_facts_t result_xor;
      loom_value_facts_andi(&left, &right, &result_and);
      loom_value_facts_ori(&left, &right, &result_or);
      loom_value_facts_xori(&left, &right, &result_xor);
      ASSERT_GE(result_and.known_divisor, 1);
      ASSERT_GE(result_or.known_divisor, 1);
      ASSERT_GE(result_xor.known_divisor, 1);
      for (int64_t lhs = left.range_lo; lhs <= left.range_hi; ++lhs) {
        if (lhs % left.known_divisor != 0) {
          continue;
        }
        for (int64_t rhs = right.range_lo; rhs <= right.range_hi; ++rhs) {
          if (rhs % right.known_divisor != 0) {
            continue;
          }
          const int64_t value_and = lhs & rhs;
          const int64_t value_or = lhs | rhs;
          const int64_t value_xor = lhs ^ rhs;
          ASSERT_LE(result_and.range_lo, value_and) << lhs << " & " << rhs;
          ASSERT_GE(result_and.range_hi, value_and) << lhs << " & " << rhs;
          ASSERT_LE(result_or.range_lo, value_or) << lhs << " | " << rhs;
          ASSERT_GE(result_or.range_hi, value_or) << lhs << " | " << rhs;
          ASSERT_LE(result_xor.range_lo, value_xor) << lhs << " ^ " << rhs;
          ASSERT_GE(result_xor.range_hi, value_xor) << lhs << " ^ " << rhs;
          ASSERT_EQ(value_and % result_and.known_divisor, 0);
          ASSERT_EQ(value_or % result_or.known_divisor, 0);
          ASSERT_EQ(value_xor % result_xor.known_divisor, 0);
          ASSERT_TRUE(!loom_value_facts_is_non_zero(result_and) ||
                      value_and != 0);
          ASSERT_TRUE(!loom_value_facts_is_non_zero(result_or) ||
                      value_or != 0);
          ASSERT_TRUE(!loom_value_facts_is_non_zero(result_xor) ||
                      value_xor != 0);
        }
      }
    }
  }
}

TEST(BitwiseTransfer, InPlaceTransferPreservesEveryFact) {
  loom_value_facts_t varying_mask = loom_value_facts_make(0, 255, 1);
  loom_value_facts_mark_lane_varying(&varying_mask);
  loom_value_facts_mark_subgroup_lane_mask(&varying_mask);
  loom_value_facts_t nonzero = loom_value_facts_unknown();
  nonzero.flags |= LOOM_VALUE_FACT_NON_ZERO;
  loom_value_facts_mark_workgroup_uniform(&nonzero);
  const loom_value_facts_t inputs[] = {loom_value_facts_unknown(),
                                       loom_value_facts_make(INT64_MIN, -1, 1),
                                       loom_value_facts_exact_i64(INT64_MIN),
                                       loom_value_facts_exact_i64(INT64_MAX),
                                       loom_value_facts_exact_i64(0),
                                       loom_value_facts_make(0, 256, 256),
                                       loom_value_facts_make(-96, 96, 24),
                                       varying_mask,
                                       nonzero};
  for (auto transfer :
       {loom_value_facts_andi, loom_value_facts_ori, loom_value_facts_xori}) {
    for (const auto& left : inputs) {
      for (const auto& right : inputs) {
        loom_value_facts_t result;
        transfer(&left, &right, &result);
        loom_value_facts_t in_place_left = left;
        transfer(&in_place_left, &right, &in_place_left);
        EXPECT_TRUE(loom_value_facts_equal(in_place_left, result));
        loom_value_facts_t in_place_right = right;
        transfer(&left, &in_place_right, &in_place_right);
        EXPECT_TRUE(loom_value_facts_equal(in_place_right, result));
      }
    }
  }
}

TEST(BitwiseTransfer, RetainsDistributionAndLaneMaskIdentity) {
  for (auto transfer :
       {loom_value_facts_andi, loom_value_facts_ori, loom_value_facts_xori}) {
    loom_value_facts_t left = loom_value_facts_make(0, 255, 4);
    loom_value_facts_mark_workgroup_uniform(&left);
    loom_value_facts_mark_subgroup_lane_mask(&left);
    // Overlapping possible bits keep each operation dependent on both inputs.
    loom_value_facts_t right = loom_value_facts_make(0, 256, 128);
    loom_value_facts_mark_subgroup_uniform(&right);
    loom_value_facts_mark_subgroup_lane_mask(&right);
    loom_value_facts_t result;
    transfer(&left, &right, &result);
    EXPECT_TRUE(loom_value_facts_is_subgroup_uniform(result));
    EXPECT_FALSE(loom_value_facts_is_workgroup_uniform(result));
    EXPECT_TRUE(loom_value_facts_is_subgroup_lane_mask(result));

    loom_value_facts_mark_lane_varying(&right);
    transfer(&left, &right, &result);
    EXPECT_TRUE(loom_value_facts_is_lane_varying(result));
    EXPECT_FALSE(loom_value_facts_is_subgroup_uniform(result));
    EXPECT_TRUE(loom_value_facts_is_subgroup_lane_mask(result));

    const loom_value_facts_t zero = loom_value_facts_exact_i64(0);
    transfer(&left, &zero, &result);
    EXPECT_TRUE(loom_value_facts_is_subgroup_lane_mask(result));
    EXPECT_TRUE(loom_value_facts_is_workgroup_uniform(result));
    transfer(&zero, &right, &result);
    EXPECT_TRUE(loom_value_facts_is_subgroup_lane_mask(result));

    const loom_value_facts_t unknown = loom_value_facts_unknown();
    transfer(&left, &unknown, &result);
    EXPECT_FALSE(loom_value_facts_is_subgroup_lane_mask(result));
    EXPECT_FALSE(loom_value_facts_is_subgroup_uniform(result));
  }
}

}  // namespace
}  // namespace loom
