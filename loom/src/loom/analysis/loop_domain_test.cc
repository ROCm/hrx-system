// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/loop_domain.h"

#include <algorithm>
#include <cstdint>
#include <numeric>

#include "iree/testing/gtest.h"
#include "loom/analysis/loop_domain_test_util.h"
#include "loom/ir/facts.h"
#include "loom/ir/float_facts.h"

namespace loom {
namespace {

TEST(LoopDomainTest, RangeFactsProveNonemptyDomain) {
  const auto lower = loom_value_facts_make(0, 4, 1);
  const auto upper = loom_value_facts_make(8, 16, 1);
  const auto step = loom_value_facts_make(1, 4, 1);
  EXPECT_TRUE(loom_loop_domain_proven_nonempty(lower, upper, step));
  EXPECT_FALSE(loom_loop_domain_proven_empty(lower, upper, step));
}

TEST(LoopDomainTest, RangeFactsProveEmptyDomain) {
  const auto lower = loom_value_facts_make(16, 24, 1);
  const auto upper = loom_value_facts_make(0, 16, 1);
  const auto step = loom_value_facts_exact_i64(1);
  EXPECT_TRUE(loom_loop_domain_proven_empty(lower, upper, step));
  EXPECT_FALSE(loom_loop_domain_proven_nonempty(lower, upper, step));
}

TEST(LoopDomainTest, OverlappingBoundsProveNeitherDomainState) {
  const auto lower = loom_value_facts_make(0, 12, 1);
  const auto upper = loom_value_facts_make(8, 16, 1);
  const auto step = loom_value_facts_exact_i64(1);
  EXPECT_FALSE(loom_loop_domain_proven_empty(lower, upper, step));
  EXPECT_FALSE(loom_loop_domain_proven_nonempty(lower, upper, step));
}

TEST(LoopDomainTest, NonpositiveOrFloatingStepProvesNeitherDomainState) {
  const auto lower = loom_value_facts_exact_i64(0);
  const auto upper = loom_value_facts_exact_i64(16);
  for (const auto step :
       {loom_value_facts_exact_i64(0), loom_value_facts_exact_i64(-1),
        loom_value_facts_unknown(),
        loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 1.0)}) {
    EXPECT_FALSE(loom_loop_domain_proven_empty(lower, upper, step));
    EXPECT_FALSE(loom_loop_domain_proven_nonempty(lower, upper, step));
  }
}

TEST(LoopDomainTest, FloatingBoundsProveNeitherDomainState) {
  const auto floating = loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 0.0);
  const auto lower = loom_value_facts_exact_i64(0);
  const auto upper = loom_value_facts_exact_i64(16);
  const auto step = loom_value_facts_exact_i64(1);
  EXPECT_FALSE(loom_loop_domain_proven_empty(floating, upper, step));
  EXPECT_FALSE(loom_loop_domain_proven_nonempty(floating, upper, step));
  EXPECT_FALSE(loom_loop_domain_proven_empty(lower, floating, step));
  EXPECT_FALSE(loom_loop_domain_proven_nonempty(lower, floating, step));
}

constexpr loom_loop_bound_flags_t kUnsignedExclusive = LOOM_LOOP_BOUND_NONE;
constexpr loom_loop_bound_flags_t kUnsignedInclusive =
    LOOM_LOOP_BOUND_INCLUSIVE;
constexpr loom_loop_bound_flags_t kSignedExclusive = LOOM_LOOP_BOUND_SIGNED;
constexpr loom_loop_bound_flags_t kSignedInclusive =
    LOOM_LOOP_BOUND_SIGNED | LOOM_LOOP_BOUND_INCLUSIVE;
constexpr loom_loop_bound_flags_t kComparisonFlags[] = {
    kUnsignedExclusive,
    kUnsignedInclusive,
    kSignedExclusive,
    kSignedInclusive,
    kUnsignedExclusive | LOOM_LOOP_BOUND_DESCENDING,
    kUnsignedInclusive | LOOM_LOOP_BOUND_DESCENDING,
    kSignedExclusive | LOOM_LOOP_BOUND_DESCENDING,
    kSignedInclusive | LOOM_LOOP_BOUND_DESCENDING,
};

// Three-bit carriers retain 4,672 complete tuples across comparison modes and
// expose signed, unsigned, wrap, and interior values. The manual stress target
// extends the same independent execution oracle through six-bit carriers.
constexpr uint8_t kPresubmitCarrierBitwidth = 3;

void ExpectCount(loom_loop_bound_flags_t bound_flags, uint8_t bitwidth,
                 uint64_t initial, uint64_t bound, uint64_t step,
                 uint64_t expected) {
  uint64_t actual = UINT64_MAX;
  ASSERT_TRUE(loom_loop_domain_trip_count(bound_flags, bitwidth, initial, bound,
                                          step, &actual));
  EXPECT_EQ(actual, expected);
}

void ExpectUnknown(loom_loop_bound_flags_t bound_flags, uint8_t bitwidth,
                   uint64_t initial, uint64_t bound, uint64_t step) {
  uint64_t actual = UINT64_MAX;
  EXPECT_FALSE(loom_loop_domain_trip_count(bound_flags, bitwidth, initial,
                                           bound, step, &actual));
  EXPECT_EQ(actual, 0u);
}

TEST(LoopDomainTripCountTest, ComparisonSignedness) {
  ExpectCount(kSignedExclusive, 32, -1, 4, 1, 5);
  ExpectCount(kUnsignedExclusive, 32, -1, 4, 1, 0);
  ExpectCount(kSignedInclusive, 32, -1, 4, 1, 6);
  ExpectCount(kUnsignedInclusive, 32, -1, 4, 1, 0);
  ExpectCount(kSignedExclusive, 64, -1, 4, 1, 5);
  ExpectCount(kUnsignedExclusive, 64, -1, 4, 1, 0);
  ExpectCount(kUnsignedExclusive, 32, INT32_MAX, INT32_MIN, 1, 1);
  ExpectCount(kSignedExclusive, 32, INT32_MAX, INT32_MIN, 1, 0);
}

TEST(LoopDomainTripCountTest, NonzeroStartsAndPartialFinalSteps) {
  ExpectCount(kSignedExclusive, 32, 7, 19, 4, 3);
  ExpectCount(kSignedInclusive, 32, 7, 19, 4, 4);
  ExpectCount(kSignedExclusive, 32, 7, 20, 4, 4);
  ExpectCount(kSignedInclusive, 32, 7, 20, 4, 4);
  ExpectCount(kSignedExclusive, 64, -9, 4, 5, 3);
  ExpectCount(kSignedInclusive, 64, -9, -4, 5, 2);
}

TEST(LoopDomainTripCountTest, TerminalIncrementMustFitCarrier) {
  ExpectCount(kSignedExclusive, 32, INT32_MAX - 1, INT32_MAX, 1, 1);
  ExpectUnknown(kSignedExclusive, 32, INT32_MAX - 1, INT32_MAX, 2);
  ExpectUnknown(kSignedInclusive, 32, INT32_MAX, INT32_MAX, 1);
  ExpectCount(kSignedInclusive, 32, INT32_MAX - 1, INT32_MAX - 1, 1, 1);
  ExpectCount(kUnsignedExclusive, 32, UINT32_MAX - 1, UINT32_MAX, 1, 1);
  ExpectUnknown(kUnsignedExclusive, 32, UINT32_MAX - 1, UINT32_MAX, 2);
  ExpectUnknown(kUnsignedInclusive, 32, UINT32_MAX, UINT32_MAX, 1);
  ExpectCount(kSignedExclusive, 64, INT64_MAX - 1, INT64_MAX, 1, 1);
  ExpectUnknown(kSignedExclusive, 64, INT64_MAX - 1, INT64_MAX, 2);
  ExpectUnknown(kSignedInclusive, 64, INT64_MAX, INT64_MAX, 1);
  ExpectCount(kUnsignedExclusive, 64, UINT64_MAX - 1, UINT64_MAX, 1, 1);
  ExpectUnknown(kUnsignedExclusive, 64, UINT64_MAX - 1, UINT64_MAX, 2);
  ExpectUnknown(kUnsignedInclusive, 64, UINT64_MAX, UINT64_MAX, 1);
}

TEST(LoopDomainTripCountTest, FullCarrierSpans) {
  ExpectCount(kSignedExclusive, 64, INT64_MIN, INT64_MAX, 1, UINT64_MAX);
  ExpectCount(kSignedExclusive, 64, INT64_MIN, 0, UINT64_C(1) << 63, 1);
  ExpectCount(kUnsignedExclusive, 64, 0, UINT64_MAX, 1, UINT64_MAX);
  ExpectUnknown(kSignedInclusive, 64, INT64_MIN, INT64_MAX, 1);
  ExpectUnknown(kUnsignedInclusive, 64, 0, UINT64_MAX, 1);
  ExpectUnknown(kUnsignedExclusive, 64, 0, UINT64_MAX, 2);
}

TEST(LoopDomainTripCountTest, EmptyLoopsAndZeroIncrements) {
  ExpectCount(kSignedExclusive, 32, 4, 4, 0, 0);
  ExpectCount(kUnsignedInclusive, 32, 5, 4, -1, 0);
  ExpectUnknown(kSignedExclusive, 32, 3, 4, 0);
  ExpectUnknown(kSignedInclusive, 32, 4, 4, 0);
  ExpectUnknown(kUnsignedExclusive, 32, 0, 4, UINT64_C(1) << 32);
}

TEST(LoopDomainTripCountTest, TruncatesAllInputsToSelectedCarrier) {
  ExpectCount(kSignedExclusive, 32, (UINT64_C(1) << 32) + 3,
              (UINT64_C(1) << 33) + 9, (UINT64_C(1) << 34) + 2, 3);
  ExpectCount(kUnsignedExclusive, 32, UINT64_MAX, 4, 1, 0);
  ExpectCount(kSignedExclusive, 32, UINT32_MAX, 4, 1, 5);
}

TEST(LoopDomainTripCountTest, DescendingCarrierBoundaries) {
  for (uint8_t bitwidth : {8, 16, 32, 64}) {
    const int64_t maximum = INT64_MAX >> (64 - bitwidth);
    const int64_t minimum = -maximum - 1;
    const uint64_t mask = UINT64_MAX >> (64 - bitwidth);
    for (auto flags : kComparisonFlags) {
      if (!(flags & LOOM_LOOP_BOUND_DESCENDING)) {
        continue;
      }
      const bool is_signed = flags & LOOM_LOOP_BOUND_SIGNED;
      const bool inclusive = flags & LOOM_LOOP_BOUND_INCLUSIVE;
      const int64_t terminal = is_signed ? minimum : 0;
      ExpectCount(flags, bitwidth, terminal + 2, terminal + inclusive, -1, 2);
      ExpectUnknown(flags, bitwidth, terminal + 1, terminal + inclusive, -2);
      if (inclusive) {
        ExpectUnknown(flags, bitwidth, terminal, terminal, -1);
      } else {
        ExpectCount(flags, bitwidth, is_signed ? uint64_t(maximum) : mask,
                    terminal, -1, mask);
      }
      ExpectCount(flags, bitwidth, terminal, terminal + inclusive, 0, 0);
    }
  }
  const auto descending = kSignedExclusive | LOOM_LOOP_BOUND_DESCENDING;
  ExpectCount(descending, 64, 0, INT64_MIN, INT64_MIN, 1);
  ExpectUnknown(descending, 64, -1, INT64_MIN, INT64_MIN);
  ExpectCount(descending, 32, (UINT64_C(1) << 32) + 9, (UINT64_C(1) << 33) + 3,
              (UINT64_C(1) << 34) - 2, 3);
}

void ExpectNumericRange(loom_value_facts_t facts, int64_t lower, int64_t upper,
                        int64_t divisor) {
  EXPECT_EQ(facts.range_lo, lower);
  EXPECT_EQ(facts.range_hi, upper);
  EXPECT_EQ(facts.known_divisor, divisor);
}

TEST(LoopDomainLogicalRightShiftRecurrenceTest,
     ExactAndRangedSeedsRetainFiniteCounts) {
  const auto shift2 = loom_value_facts_exact_i64(2);
  auto facts = loom_loop_domain_shrui_to_zero_recurrence_facts(
      64, loom_value_facts_exact_i64(0), shift2);
  EXPECT_EQ(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_EXACT);
  EXPECT_EQ(facts.trip_count, 0u);
  ExpectNumericRange(facts.values, 0, 0, 1);
  EXPECT_TRUE(loom_value_facts_is_unknown(facts.body_values));
  ExpectNumericRange(facts.exit_value, 0, 0, 1);

  facts = loom_loop_domain_shrui_to_zero_recurrence_facts(
      64, loom_value_facts_exact_i64(-1), shift2);
  EXPECT_EQ(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_EXACT);
  EXPECT_EQ(facts.trip_count, 32u);
  ExpectNumericRange(facts.values, -1, INT64_MAX / 2, 1);
  ExpectNumericRange(facts.body_values, -1, INT64_MAX / 2, 1);
  ExpectNumericRange(facts.exit_value, 0, 0, 1);

  facts = loom_loop_domain_shrui_to_zero_recurrence_facts(
      64, loom_value_facts_unknown(), shift2);
  EXPECT_EQ(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_BOUNDED);
  EXPECT_EQ(facts.trip_count, 32u);
  ExpectNumericRange(facts.values, INT64_MIN, INT64_MAX, 1);
  ExpectNumericRange(facts.body_values, INT64_MIN, INT64_MAX, 1);
  ExpectNumericRange(facts.exit_value, 0, 0, 1);

  facts = loom_loop_domain_shrui_to_zero_recurrence_facts(
      64, loom_value_facts_make(0, 255, 1), loom_value_facts_exact_i64(4));
  EXPECT_EQ(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_BOUNDED);
  EXPECT_EQ(facts.trip_count, 2u);
  ExpectNumericRange(facts.values, 0, 255, 1);
  ExpectNumericRange(facts.body_values, 1, 255, 1);
  ExpectNumericRange(facts.exit_value, 0, 0, 1);

  facts = loom_loop_domain_shrui_to_zero_recurrence_facts(
      8, loom_value_facts_exact_i64(-128), loom_value_facts_exact_i64(3));
  EXPECT_EQ(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_EXACT);
  EXPECT_EQ(facts.trip_count, 3u);
  ExpectNumericRange(facts.values, -128, 16, 1);
  ExpectNumericRange(facts.body_values, -128, 16, 1);
  ExpectNumericRange(facts.exit_value, 0, 0, 1);
}

TEST(LoopDomainLogicalRightShiftRecurrenceTest,
     InvalidDomainsPublishNoPartialProof) {
  const auto initial = loom_value_facts_make(-128, 127, 1);
  for (const auto shift :
       {loom_value_facts_exact_i64(-1), loom_value_facts_exact_i64(0),
        loom_value_facts_exact_i64(8), loom_value_facts_make(1, 2, 1),
        loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 1.0)}) {
    const auto facts =
        loom_loop_domain_shrui_to_zero_recurrence_facts(8, initial, shift);
    EXPECT_EQ(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_UNKNOWN);
    EXPECT_EQ(facts.trip_count, 0u);
    EXPECT_TRUE(loom_value_facts_is_unknown(facts.values));
    EXPECT_TRUE(loom_value_facts_is_unknown(facts.body_values));
    EXPECT_TRUE(loom_value_facts_is_unknown(facts.exit_value));
  }
  const auto floating = loom_loop_domain_shrui_to_zero_recurrence_facts(
      8, loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 1.0),
      loom_value_facts_exact_i64(1));
  EXPECT_EQ(floating.trip_count_kind, LOOM_LOOP_TRIP_COUNT_UNKNOWN);
  for (uint8_t bitwidth : {uint8_t{0}, uint8_t{1}, uint8_t{65}}) {
    const auto facts = loom_loop_domain_shrui_to_zero_recurrence_facts(
        bitwidth, initial, loom_value_facts_exact_i64(1));
    EXPECT_EQ(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_UNKNOWN);
  }
}

TEST(LoopDomainLogicalRightShiftRecurrenceTest,
     SmallRangesContainEveryExecution) {
  auto contains = [](loom_value_facts_t facts, int64_t value) {
    return facts.range_lo <= value && value <= facts.range_hi &&
           value % facts.known_divisor == 0;
  };
  for (uint8_t bitwidth = 2; bitwidth <= 6; ++bitwidth) {
    const uint64_t modulus = UINT64_C(1) << bitwidth;
    const uint64_t mask = modulus - 1;
    const int64_t minimum = -int64_t(modulus / 2);
    const int64_t maximum = int64_t(modulus / 2) - 1;
    for (int64_t lower = minimum; lower <= maximum; ++lower) {
      for (int64_t upper = lower; upper <= maximum; ++upper) {
        for (uint8_t shift = 1; shift < bitwidth; ++shift) {
          const auto facts = loom_loop_domain_shrui_to_zero_recurrence_facts(
              bitwidth, loom_value_facts_make(lower, upper, 1),
              loom_value_facts_exact_i64(shift));
          ASSERT_NE(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_UNKNOWN);
          EXPECT_EQ(facts.trip_count_kind, lower == upper
                                               ? LOOM_LOOP_TRIP_COUNT_EXACT
                                               : LOOM_LOOP_TRIP_COUNT_BOUNDED);
          for (int64_t initial = lower; initial <= upper; ++initial) {
            uint64_t state = (uint64_t)initial & mask;
            uint64_t trip_count = 0;
            int64_t header_minimum = initial;
            int64_t header_maximum = initial;
            int64_t body_minimum = INT64_MAX;
            int64_t body_maximum = INT64_MIN;
            while (state != 0) {
              const int64_t body_value =
                  trip_count == 0 ? initial : (int64_t)state;
              body_minimum = std::min(body_minimum, body_value);
              body_maximum = std::max(body_maximum, body_value);
              state >>= shift;
              ++trip_count;
              header_minimum = std::min(header_minimum, (int64_t)state);
              header_maximum = std::max(header_maximum, (int64_t)state);
            }
            const bool exact =
                facts.trip_count_kind == LOOM_LOOP_TRIP_COUNT_EXACT;
            if (trip_count > facts.trip_count ||
                (exact && trip_count != facts.trip_count) ||
                !contains(facts.values, header_minimum) ||
                !contains(facts.values, header_maximum) ||
                !contains(facts.exit_value, 0) ||
                (trip_count != 0 &&
                 (!contains(facts.body_values, body_minimum) ||
                  !contains(facts.body_values, body_maximum)))) {
              FAIL() << "width=" << int(bitwidth) << " initial=[" << lower
                     << "," << upper << "] shift=" << int(shift)
                     << " witness=" << initial;
            }
          }
        }
      }
    }
  }
}

TEST(LoopDomainAdditiveRecurrenceTest, PositiveAndNegativeRangeSeeds) {
  loom_loop_recurrence_facts_t facts;
  ASSERT_TRUE(loom_loop_domain_additive_recurrence_facts(
      loom_value_facts_make(0, 256, 256), 64, 4, &facts));
  EXPECT_EQ(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_EXACT);
  EXPECT_EQ(facts.trip_count, 4u);
  ExpectNumericRange(facts.values, 0, 512, 64);
  ExpectNumericRange(facts.body_values, 0, 448, 64);
  ExpectNumericRange(facts.exit_value, 256, 512, 256);

  ASSERT_TRUE(loom_loop_domain_additive_recurrence_facts(
      loom_value_facts_make(256, 512, 256), -64, 4, &facts));
  ExpectNumericRange(facts.values, 0, 512, 64);
  ExpectNumericRange(facts.body_values, 64, 512, 64);
  ExpectNumericRange(facts.exit_value, 0, 256, 256);
}

TEST(LoopDomainAdditiveRecurrenceTest, ZeroOriginRetainsStrideDivisibility) {
  loom_loop_recurrence_facts_t facts;
  ASSERT_TRUE(loom_loop_domain_additive_recurrence_facts(
      loom_value_facts_exact_i64(0), 128, 8, &facts));
  ExpectNumericRange(facts.values, 0, 1024, 128);
  ExpectNumericRange(facts.body_values, 0, 896, 128);
  ExpectNumericRange(facts.exit_value, 1024, 1024, 1024);
}

TEST(LoopDomainAdditiveRecurrenceTest, OneTripRetainsInitialBodyDivisor) {
  loom_loop_recurrence_facts_t facts;
  ASSERT_TRUE(loom_loop_domain_additive_recurrence_facts(
      loom_value_facts_make(0, 512, 256), 64, 1, &facts));
  ExpectNumericRange(facts.values, 0, 576, 64);
  ExpectNumericRange(facts.body_values, 0, 512, 256);
  ExpectNumericRange(facts.exit_value, 64, 576, 64);

  ASSERT_TRUE(loom_loop_domain_additive_recurrence_facts(
      loom_value_facts_exact_i64(0), -128, 1, &facts));
  ExpectNumericRange(facts.values, -128, 0, 128);
  ExpectNumericRange(facts.body_values, 0, 0, 1);
  ExpectNumericRange(facts.exit_value, -128, -128, 128);
}

TEST(LoopDomainAdditiveRecurrenceTest, ZeroTripsAndStationaryRanges) {
  const auto initial = loom_value_facts_make(128, 256, 128);
  loom_loop_recurrence_facts_t facts;
  for (int64_t step :
       {INT64_MIN, INT64_C(-1), INT64_C(0), INT64_C(1), INT64_MAX}) {
    ASSERT_TRUE(
        loom_loop_domain_additive_recurrence_facts(initial, step, 0, &facts));
    EXPECT_EQ(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_EXACT);
    EXPECT_EQ(facts.trip_count, 0u);
    ExpectNumericRange(facts.values, 128, 256, 128);
    ExpectNumericRange(facts.exit_value, 128, 256, 128);
    EXPECT_TRUE(loom_value_facts_is_unknown(facts.body_values));
  }
  ASSERT_TRUE(loom_loop_domain_additive_recurrence_facts(initial, 0, UINT64_MAX,
                                                         &facts));
  EXPECT_EQ(facts.trip_count, UINT64_MAX);
  ExpectNumericRange(facts.values, 128, 256, 128);
  ExpectNumericRange(facts.body_values, 128, 256, 128);
  ExpectNumericRange(facts.exit_value, 128, 256, 128);
  ASSERT_TRUE(loom_loop_domain_additive_recurrence_facts(
      loom_value_facts_unknown(), 0, UINT64_MAX, &facts));
  ExpectNumericRange(facts.values, INT64_MIN, INT64_MAX, 1);
}

TEST(LoopDomainAdditiveRecurrenceTest, FullSignedSpan) {
  for (int64_t step : {INT64_C(-1), INT64_C(1)}) {
    const int64_t initial = step > 0 ? INT64_MIN : INT64_MAX;
    const int64_t terminal = step > 0 ? INT64_MAX : INT64_MIN;
    loom_loop_recurrence_facts_t facts;
    ASSERT_TRUE(loom_loop_domain_additive_recurrence_facts(
        loom_value_facts_exact_i64(initial), step, UINT64_MAX, &facts));
    EXPECT_EQ(facts.trip_count, UINT64_MAX);
    ExpectNumericRange(facts.values, INT64_MIN, INT64_MAX, 1);
    ExpectNumericRange(facts.body_values, step > 0 ? INT64_MIN : INT64_MIN + 1,
                       step > 0 ? INT64_MAX - 1 : INT64_MAX, 1);
    ExpectNumericRange(facts.exit_value, terminal, terminal,
                       step > 0 ? INT64_MAX : 1);
  }
}

TEST(LoopDomainAdditiveRecurrenceTest, UnsignedDistanceAndMinimumSignedStep) {
  loom_loop_recurrence_facts_t facts;
  ASSERT_TRUE(loom_loop_domain_additive_recurrence_facts(
      loom_value_facts_exact_i64(INT64_MIN), INT64_MAX, 2, &facts));
  ExpectNumericRange(facts.values, INT64_MIN, INT64_MAX - 1, 1);
  ExpectNumericRange(facts.body_values, INT64_MIN, -1, 1);
  ExpectNumericRange(facts.exit_value, INT64_MAX - 1, INT64_MAX - 1,
                     INT64_MAX - 1);

  ASSERT_TRUE(loom_loop_domain_additive_recurrence_facts(
      loom_value_facts_make(0, INT64_MAX, 1), INT64_MIN, 1, &facts));
  ExpectNumericRange(facts.values, INT64_MIN, INT64_MAX, 1);
  ExpectNumericRange(facts.body_values, 0, INT64_MAX, 1);
  ExpectNumericRange(facts.exit_value, INT64_MIN, -1, 1);

  ASSERT_TRUE(loom_loop_domain_additive_recurrence_facts(
      loom_value_facts_exact_i64(0), INT64_MIN, 1, &facts));
  ExpectNumericRange(facts.values, INT64_MIN, 0, 1);
  ExpectNumericRange(facts.body_values, 0, 0, 1);
  ExpectNumericRange(facts.exit_value, INT64_MIN, INT64_MIN, 1);

  ASSERT_TRUE(loom_loop_domain_additive_recurrence_facts(
      loom_value_facts_exact_i64(INT64_MIN), 128, 2, &facts));
  ExpectNumericRange(facts.values, INT64_MIN, INT64_MIN + 256, 128);
  ExpectNumericRange(facts.body_values, INT64_MIN, INT64_MIN + 128, 128);
  ExpectNumericRange(facts.exit_value, INT64_MIN + 256, INT64_MIN + 256,
                     INT64_MAX - 255);
}

TEST(LoopDomainAdditiveRecurrenceTest, OverflowPublishesNoPartialProof) {
  auto expect_unproven = [](loom_value_facts_t initial, int64_t step,
                            uint64_t count) {
    loom_loop_recurrence_facts_t facts;
    EXPECT_FALSE(loom_loop_domain_additive_recurrence_facts(initial, step,
                                                            count, &facts));
    EXPECT_EQ(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_EXACT);
    EXPECT_EQ(facts.trip_count, count);
    EXPECT_TRUE(loom_value_facts_is_unknown(facts.values));
    EXPECT_TRUE(loom_value_facts_is_unknown(facts.body_values));
    EXPECT_TRUE(loom_value_facts_is_unknown(facts.exit_value));
  };
  expect_unproven(loom_value_facts_exact_i64(INT64_MAX), 1, 1);
  expect_unproven(loom_value_facts_exact_i64(INT64_MIN), -1, 1);
  expect_unproven(loom_value_facts_exact_i64(INT64_MAX), INT64_MIN, 2);
  expect_unproven(loom_value_facts_make(INT64_MIN, INT64_MIN + 1, 1), 1,
                  UINT64_MAX);
  expect_unproven(loom_value_facts_make(INT64_MAX - 1, INT64_MAX, 1), -1,
                  UINT64_MAX);
}

TEST(LoopDomainAdditiveRecurrenceTest, SmallRangesMatchExplicitIteration) {
  for (int64_t lower = -4; lower <= 4; ++lower) {
    for (int64_t upper = lower; upper <= 4; ++upper) {
      for (int64_t step = -4; step <= 4; ++step) {
        for (uint64_t count = 0; count <= 8; ++count) {
          SCOPED_TRACE(::testing::Message()
                       << "initial=[" << lower << "," << upper
                       << "] step=" << step << " count=" << count);
          int64_t header_lower = INT64_MAX, header_upper = INT64_MIN;
          int64_t body_lower = INT64_MAX, body_upper = INT64_MIN;
          int64_t exit_lower = INT64_MAX, exit_upper = INT64_MIN;
          int64_t header_divisor = 0, body_divisor = 0, exit_divisor = 0;
          for (int64_t initial = lower; initial <= upper; ++initial) {
            int64_t value = initial;
            for (uint64_t iteration = 0; iteration <= count; ++iteration) {
              header_lower = std::min(header_lower, value);
              header_upper = std::max(header_upper, value);
              header_divisor = std::gcd(header_divisor, value);
              if (iteration != count) {
                body_lower = std::min(body_lower, value);
                body_upper = std::max(body_upper, value);
                body_divisor = std::gcd(body_divisor, value);
                value += step;
              }
            }
            exit_lower = std::min(exit_lower, value);
            exit_upper = std::max(exit_upper, value);
            exit_divisor = std::gcd(exit_divisor, value);
          }
          loom_loop_recurrence_facts_t facts;
          ASSERT_TRUE(loom_loop_domain_additive_recurrence_facts(
              loom_value_facts_make(lower, upper, 1), step, count, &facts));
          EXPECT_EQ(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_EXACT);
          EXPECT_EQ(facts.trip_count, count);
          ExpectNumericRange(facts.values, header_lower, header_upper,
                             std::max(INT64_C(1), header_divisor));
          ExpectNumericRange(facts.exit_value, exit_lower, exit_upper,
                             std::max(INT64_C(1), exit_divisor));
          if (count != 0) {
            ExpectNumericRange(facts.body_values, body_lower, body_upper,
                               std::max(INT64_C(1), body_divisor));
          } else {
            EXPECT_TRUE(loom_value_facts_is_unknown(facts.body_values));
          }
        }
      }
    }
  }
}

TEST(LoopDomainRecurrenceTest, FullSignedCarrierAndTerminalBoundaries) {
  for (const uint8_t bitwidth : {32, 64}) {
    const int64_t maximum = INT64_MAX >> (64 - bitwidth);
    const int64_t minimum = -maximum - 1;
    const auto full = loom_loop_domain_recurrence_facts(
        kSignedExclusive, bitwidth, loom_value_facts_exact_i64(minimum),
        loom_value_facts_exact_i64(maximum), loom_value_facts_exact_i64(1));
    EXPECT_EQ(full.trip_count_kind, LOOM_LOOP_TRIP_COUNT_EXACT);
    EXPECT_EQ(full.values.range_lo, minimum);
    EXPECT_EQ(full.values.range_hi, maximum);
    EXPECT_EQ(full.body_values.range_lo, minimum);
    EXPECT_EQ(full.body_values.range_hi, maximum - 1);
    EXPECT_EQ(full.exit_value.range_lo, maximum);
    EXPECT_EQ(full.exit_value.range_hi, maximum);
    const auto tail = loom_loop_domain_recurrence_facts(
        kSignedExclusive, bitwidth, loom_value_facts_exact_i64(maximum - 1),
        loom_value_facts_exact_i64(maximum), loom_value_facts_exact_i64(2));
    EXPECT_EQ(tail.trip_count_kind, LOOM_LOOP_TRIP_COUNT_UNKNOWN);
    EXPECT_TRUE(loom_value_facts_is_unknown(tail.values));
    const auto crossing = loom_loop_domain_recurrence_facts(
        kUnsignedExclusive, bitwidth, loom_value_facts_exact_i64(maximum),
        loom_value_facts_exact_i64(minimum), loom_value_facts_exact_i64(1));
    EXPECT_EQ(crossing.trip_count_kind, LOOM_LOOP_TRIP_COUNT_EXACT);
    EXPECT_EQ(crossing.trip_count, 1u);
    EXPECT_TRUE(loom_value_facts_is_unknown(crossing.values));
  }
}

TEST(LoopDomainRecurrenceTest, SourceValueMustFitCarrier) {
  const auto facts = loom_loop_domain_recurrence_facts(
      kSignedExclusive, 32, loom_value_facts_exact_i64(INT64_C(1) << 32),
      loom_value_facts_exact_i64(4), loom_value_facts_exact_i64(1));
  EXPECT_EQ(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_EXACT);
  EXPECT_EQ(facts.trip_count, 4u);
  EXPECT_TRUE(loom_value_facts_is_unknown(facts.values));
}

TEST(LoopDomainRecurrenceTest, ExactExitRetainsUniformity) {
  const auto empty = loom_loop_domain_recurrence_facts(
      kSignedExclusive, 32, loom_value_facts_exact_i64(5),
      loom_value_facts_exact_i64(4), loom_value_facts_exact_i64(1));
  EXPECT_TRUE(loom_value_facts_is_cluster_uniform(empty.values));
  EXPECT_TRUE(loom_value_facts_is_cluster_uniform(empty.exit_value));
  const auto nonempty = loom_loop_domain_recurrence_facts(
      kSignedExclusive, 32, loom_value_facts_exact_i64(0),
      loom_value_facts_exact_i64(4), loom_value_facts_exact_i64(1));
  EXPECT_TRUE(loom_value_facts_is_cluster_uniform(nonempty.exit_value));
}

TEST(LoopDomainRecurrenceTest, NextGuardRetainsCarriedObservations) {
  for (uint8_t bitwidth : {8, 16, 32, 64}) {
    SCOPED_TRACE(::testing::Message() << "width=" << int(bitwidth));
    const auto ascending = loom_loop_domain_next_guard_recurrence_facts(
        kSignedExclusive, bitwidth, loom_value_facts_exact_i64(0),
        loom_value_facts_exact_i64(6), loom_value_facts_exact_i64(2));
    EXPECT_EQ(ascending.trip_count_kind, LOOM_LOOP_TRIP_COUNT_EXACT);
    EXPECT_EQ(ascending.trip_count, 2u);
    ExpectNumericRange(ascending.values, 0, 4, 2);
    ExpectNumericRange(ascending.body_values, 0, 2, 2);
    ExpectNumericRange(ascending.exit_value, 4, 4, 4);

    const auto descending = loom_loop_domain_next_guard_recurrence_facts(
        kSignedInclusive | LOOM_LOOP_BOUND_DESCENDING, bitwidth,
        loom_value_facts_exact_i64(10), loom_value_facts_exact_i64(2),
        loom_value_facts_exact_i64(-2));
    EXPECT_EQ(descending.trip_count_kind, LOOM_LOOP_TRIP_COUNT_EXACT);
    EXPECT_EQ(descending.trip_count, 4u);
    ExpectNumericRange(descending.values, 2, 10, 2);
    ExpectNumericRange(descending.body_values, 4, 10, 2);
    ExpectNumericRange(descending.exit_value, 2, 2, 2);
  }
}

TEST(LoopDomainRecurrenceTest, NextGuardBoundsRangedStarts) {
  const auto facts = loom_loop_domain_next_guard_recurrence_facts(
      kUnsignedInclusive, 32, loom_value_facts_make(0, 128, 1),
      loom_value_facts_make(0, 128, 1), loom_value_facts_exact_i64(16));
  EXPECT_EQ(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_BOUNDED);
  EXPECT_EQ(facts.trip_count, 8u);
  ExpectNumericRange(facts.values, 0, 256, 1);
  ExpectNumericRange(facts.body_values, 0, 240, 1);
  ExpectNumericRange(facts.exit_value, 0, 256, 1);

  const auto empty = loom_loop_domain_next_guard_recurrence_facts(
      kSignedExclusive, 32, loom_value_facts_exact_i64(4),
      loom_value_facts_exact_i64(4), loom_value_facts_exact_i64(1));
  EXPECT_EQ(empty.trip_count_kind, LOOM_LOOP_TRIP_COUNT_EXACT);
  EXPECT_EQ(empty.trip_count, 0u);
  ExpectNumericRange(empty.values, 4, 4, 4);
  ExpectNumericRange(empty.exit_value, 4, 4, 4);
  EXPECT_TRUE(loom_value_facts_is_unknown(empty.body_values));
}

TEST(LoopDomainRecurrenceTest, NextGuardPreservesTerminalWrap) {
  const auto facts = loom_loop_domain_next_guard_recurrence_facts(
      kSignedExclusive, 8, loom_value_facts_exact_i64(126),
      loom_value_facts_exact_i64(127), loom_value_facts_exact_i64(2));
  EXPECT_EQ(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_UNKNOWN);
  EXPECT_TRUE(loom_value_facts_is_unknown(facts.values));
}

TEST(LoopDomainRecurrenceTest, BoundedCountRetainsAllPossibleExits) {
  for (uint8_t bitwidth : {8, 16, 32, 64}) {
    for (auto flags : {kSignedExclusive, kSignedInclusive, kUnsignedExclusive,
                       kUnsignedInclusive}) {
      SCOPED_TRACE(::testing::Message()
                   << "width=" << int(bitwidth) << " flags=" << int(flags));
      const int64_t maximum = flags & LOOM_LOOP_BOUND_INCLUSIVE ? 4 : 3;
      const auto facts = loom_loop_domain_recurrence_facts(
          flags, bitwidth, loom_value_facts_exact_i64(0),
          loom_value_facts_make(0, 3, 1), loom_value_facts_exact_i64(1));
      EXPECT_EQ(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_BOUNDED);
      EXPECT_EQ(facts.trip_count, uint64_t(maximum));
      ExpectNumericRange(facts.values, 0, maximum, 1);
      ExpectNumericRange(facts.body_values, 0, maximum - 1, 1);
      ExpectNumericRange(facts.exit_value, 0, maximum, 1);
      EXPECT_FALSE(loom_value_facts_is_exact(facts.exit_value));
    }
  }
}

TEST(LoopDomainRecurrenceTest, RangedStartsAndUnsignedNegativePartition) {
  const auto signed_facts = loom_loop_domain_recurrence_facts(
      kSignedExclusive, 8, loom_value_facts_make(-6, -2, 1),
      loom_value_facts_make(-1, 3, 1), loom_value_facts_exact_i64(2));
  EXPECT_EQ(signed_facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_BOUNDED);
  EXPECT_EQ(signed_facts.trip_count, 5u);
  ExpectNumericRange(signed_facts.values, -6, 4, 1);
  ExpectNumericRange(signed_facts.body_values, -6, 2, 1);
  ExpectNumericRange(signed_facts.exit_value, -6, 4, 1);

  const auto unsigned_facts = loom_loop_domain_recurrence_facts(
      kUnsignedExclusive, 8, loom_value_facts_make(-12, -8, 1),
      loom_value_facts_make(-4, -2, 1), loom_value_facts_exact_i64(2));
  EXPECT_EQ(unsigned_facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_BOUNDED);
  EXPECT_EQ(unsigned_facts.trip_count, 5u);
  ExpectNumericRange(unsigned_facts.values, -12, -1, 1);
  ExpectNumericRange(unsigned_facts.body_values, -12, -3, 1);
  ExpectNumericRange(unsigned_facts.exit_value, -12, -1, 1);

  const auto aligned = loom_loop_domain_recurrence_facts(
      kSignedExclusive, 32, loom_value_facts_make(0, 64, 64),
      loom_value_facts_make(128, 192, 64), loom_value_facts_exact_i64(64));
  EXPECT_EQ(aligned.trip_count_kind, LOOM_LOOP_TRIP_COUNT_BOUNDED);
  EXPECT_EQ(aligned.trip_count, 3u);
  EXPECT_EQ(aligned.values.known_divisor, 64);
  EXPECT_EQ(aligned.body_values.known_divisor, 64);
  EXPECT_EQ(aligned.exit_value.known_divisor, 64);
}

TEST(LoopDomainRecurrenceTest, EveryTerminalResidueMustFit) {
  for (uint8_t bitwidth : {8, 16, 32, 64}) {
    const int64_t maximum = INT64_MAX >> (64 - bitwidth);
    for (auto flags : {kSignedExclusive, kSignedInclusive, kUnsignedExclusive,
                       kUnsignedInclusive}) {
      SCOPED_TRACE(::testing::Message()
                   << "width=" << int(bitwidth) << " flags=" << int(flags));
      const bool is_signed = flags & LOOM_LOOP_BOUND_SIGNED;
      const bool inclusive = flags & LOOM_LOOP_BOUND_INCLUSIVE;
      const int64_t terminal = is_signed ? maximum : -1;
      const auto initial = loom_value_facts_make(terminal - 2, terminal - 1, 1);
      const auto upper = loom_value_facts_exact_i64(terminal - inclusive);
      // The first seed reaches the terminal safely with step two; the next
      // seed has another residue and overflows on its final increment.
      const auto unsafe = loom_loop_domain_recurrence_facts(
          flags, bitwidth, initial, upper, loom_value_facts_exact_i64(2));
      EXPECT_EQ(unsafe.trip_count_kind, LOOM_LOOP_TRIP_COUNT_UNKNOWN);
      EXPECT_EQ(unsafe.trip_count, 0u);
      EXPECT_TRUE(loom_value_facts_is_unknown(unsafe.values));
      const auto safe = loom_loop_domain_recurrence_facts(
          flags, bitwidth, initial, upper, loom_value_facts_exact_i64(1));
      EXPECT_EQ(safe.trip_count_kind, LOOM_LOOP_TRIP_COUNT_BOUNDED);
      EXPECT_EQ(safe.trip_count, 2u);
      ExpectNumericRange(safe.values, terminal - 2, terminal, 1);
      ExpectNumericRange(safe.body_values, terminal - 2, terminal - 1, 1);
    }
  }
}

TEST(LoopDomainRecurrenceTest, BoundedCountAndSourceRangesAreIndependent) {
  for (uint8_t bitwidth : {8, 16, 32, 64}) {
    const int64_t minimum = -(INT64_MAX >> (64 - bitwidth)) - 1;
    const auto crossing = loom_loop_domain_recurrence_facts(
        kUnsignedExclusive, bitwidth, loom_value_facts_make(0, 1, 1),
        loom_value_facts_make(minimum, minimum + 1, 1),
        loom_value_facts_exact_i64(1));
    EXPECT_EQ(crossing.trip_count_kind, LOOM_LOOP_TRIP_COUNT_BOUNDED);
    EXPECT_EQ(crossing.trip_count, (UINT64_C(1) << (bitwidth - 1)) + 1);
    EXPECT_TRUE(loom_value_facts_is_unknown(crossing.values));
    EXPECT_TRUE(loom_value_facts_is_unknown(crossing.body_values));
    EXPECT_TRUE(loom_value_facts_is_unknown(crossing.exit_value));
  }
  const auto discontinuous = loom_loop_domain_recurrence_facts(
      kUnsignedExclusive, 8, loom_value_facts_make(-1, 1, 1),
      loom_value_facts_make(4, 8, 1), loom_value_facts_exact_i64(1));
  EXPECT_EQ(discontinuous.trip_count_kind, LOOM_LOOP_TRIP_COUNT_UNKNOWN);
  const auto unsigned_carrier = loom_loop_domain_recurrence_facts(
      kUnsignedExclusive, 8, loom_value_facts_make(127, 128, 1),
      loom_value_facts_make(130, 131, 1), loom_value_facts_exact_i64(1));
  EXPECT_EQ(unsigned_carrier.trip_count_kind, LOOM_LOOP_TRIP_COUNT_BOUNDED);
  EXPECT_EQ(unsigned_carrier.trip_count, 4u);
  EXPECT_TRUE(loom_value_facts_is_unknown(unsigned_carrier.values));
}

TEST(LoopDomainRecurrenceTest, BoundedFullSignedSpan) {
  for (uint8_t bitwidth : {8, 16, 32, 64}) {
    const int64_t maximum = INT64_MAX >> (64 - bitwidth);
    const int64_t minimum = -maximum - 1;
    const auto facts = loom_loop_domain_recurrence_facts(
        kSignedExclusive, bitwidth, loom_value_facts_exact_i64(minimum),
        loom_value_facts_make(maximum - 1, maximum, 1),
        loom_value_facts_exact_i64(1));
    EXPECT_EQ(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_BOUNDED);
    EXPECT_EQ(facts.trip_count, UINT64_MAX >> (64 - bitwidth));
    ExpectNumericRange(facts.values, minimum, maximum, 1);
    ExpectNumericRange(facts.body_values, minimum, maximum - 1, 1);
    ExpectNumericRange(facts.exit_value, minimum, maximum, 1);
  }
}

TEST(LoopDomainRecurrenceTest, ZeroMaximumIsExactWithoutKnownStep) {
  for (auto flags : {kSignedExclusive, kSignedInclusive, kUnsignedExclusive,
                     kUnsignedInclusive}) {
    const auto facts = loom_loop_domain_recurrence_facts(
        flags, 8, loom_value_facts_make(5, 8, 1),
        loom_value_facts_make(0, 4, 1), loom_value_facts_unknown());
    EXPECT_EQ(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_EXACT);
    EXPECT_EQ(facts.trip_count, 0u);
    ExpectNumericRange(facts.values, 5, 8, 1);
    ExpectNumericRange(facts.exit_value, 5, 8, 1);
    EXPECT_TRUE(loom_value_facts_is_unknown(facts.body_values));
  }
}

TEST(LoopDomainRecurrenceTest, RangedProofDoesNotTruncateOrGuessStep) {
  const auto initial = loom_value_facts_make(0, 1, 1);
  const auto upper = loom_value_facts_make(4, 8, 1);
  const auto unit_step = loom_value_facts_exact_i64(1);
  const auto outside_carrier =
      loom_value_facts_make(INT64_C(1) << 32, (INT64_C(1) << 32) + 1, 1);
  for (auto flags : {kSignedExclusive, kSignedInclusive, kUnsignedExclusive,
                     kUnsignedInclusive}) {
    for (const auto step :
         {loom_value_facts_unknown(), loom_value_facts_make(1, 2, 1),
          loom_value_facts_exact_i64(0), loom_value_facts_exact_i64(-1),
          loom_value_facts_exact_i64((INT64_C(1) << 32) + 1)}) {
      const auto facts =
          loom_loop_domain_recurrence_facts(flags, 32, initial, upper, step);
      EXPECT_EQ(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_UNKNOWN);
    }
    for (bool wide_initial : {false, true}) {
      const auto facts = loom_loop_domain_recurrence_facts(
          flags, 32, wide_initial ? outside_carrier : initial,
          wide_initial ? upper : outside_carrier, unit_step);
      EXPECT_EQ(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_UNKNOWN);
    }
    const auto unknown_index = loom_loop_domain_recurrence_facts(
        flags, 32, loom_value_facts_unknown(), upper, unit_step);
    EXPECT_EQ(unknown_index.trip_count_kind, LOOM_LOOP_TRIP_COUNT_UNKNOWN);
    const auto floating =
        loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 1.0);
    for (int operand = 0; operand < 3; ++operand) {
      const auto facts = loom_loop_domain_recurrence_facts(
          flags, 32, operand == 0 ? floating : initial,
          operand == 1 ? floating : upper, operand == 2 ? floating : unit_step);
      EXPECT_EQ(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_UNKNOWN);
    }
  }
}

TEST(LoopDomainRecurrenceTest, DescendingRangesRetainEveryExit) {
  for (uint8_t bitwidth : {8, 16, 32, 64}) {
    const int64_t minimum = -(INT64_MAX >> (64 - bitwidth)) - 1;
    for (auto flags : kComparisonFlags) {
      if (!(flags & LOOM_LOOP_BOUND_DESCENDING)) {
        continue;
      }
      const bool is_signed = flags & LOOM_LOOP_BOUND_SIGNED;
      const bool inclusive = flags & LOOM_LOOP_BOUND_INCLUSIVE;
      SCOPED_TRACE(::testing::Message()
                   << "width=" << int(bitwidth) << " flags=" << int(flags));
      const auto facts = loom_loop_domain_recurrence_facts(
          flags, bitwidth, loom_value_facts_make(0, 3, 1),
          loom_value_facts_exact_i64(inclusive ? 1 : 0),
          loom_value_facts_exact_i64(-1));
      EXPECT_EQ(facts.trip_count_kind, LOOM_LOOP_TRIP_COUNT_BOUNDED);
      EXPECT_EQ(facts.trip_count, 3u);
      ExpectNumericRange(facts.values, 0, 3, 1);
      ExpectNumericRange(facts.body_values, 1, 3, 1);
      ExpectNumericRange(facts.exit_value, 0, 3, 1);

      const int64_t terminal = is_signed ? minimum : 0;
      const auto initial = loom_value_facts_make(terminal + 1, terminal + 2, 1);
      const auto bound = loom_value_facts_exact_i64(terminal + inclusive);
      const auto unsafe = loom_loop_domain_recurrence_facts(
          flags, bitwidth, initial, bound, loom_value_facts_exact_i64(-2));
      EXPECT_EQ(unsafe.trip_count_kind, LOOM_LOOP_TRIP_COUNT_UNKNOWN);
      const auto safe = loom_loop_domain_recurrence_facts(
          flags, bitwidth, initial, bound, loom_value_facts_exact_i64(-1));
      EXPECT_EQ(safe.trip_count_kind, LOOM_LOOP_TRIP_COUNT_BOUNDED);
      EXPECT_EQ(safe.trip_count, 2u);
      ExpectNumericRange(safe.values, terminal, terminal + 2, 1);
      ExpectNumericRange(safe.body_values, terminal + 1, terminal + 2, 1);

      const auto empty = loom_loop_domain_recurrence_facts(
          flags, bitwidth, loom_value_facts_make(0, 3, 1),
          loom_value_facts_exact_i64(4), loom_value_facts_unknown());
      EXPECT_EQ(empty.trip_count_kind, LOOM_LOOP_TRIP_COUNT_EXACT);
      EXPECT_EQ(empty.trip_count, 0u);
      ExpectNumericRange(empty.exit_value, 0, 3, 1);
      for (const auto step :
           {loom_value_facts_exact_i64(1), loom_value_facts_make(-2, -1, 1),
            loom_value_facts_unknown()}) {
        const auto unknown = loom_loop_domain_recurrence_facts(
            flags, bitwidth, initial, bound, step);
        EXPECT_EQ(unknown.trip_count_kind, LOOM_LOOP_TRIP_COUNT_UNKNOWN);
      }
    }
  }
  const auto descending = kSignedExclusive | LOOM_LOOP_BOUND_DESCENDING;
  const auto varying_bounds = loom_loop_domain_recurrence_facts(
      descending, 32, loom_value_facts_exact_i64(3),
      loom_value_facts_make(0, 3, 1), loom_value_facts_exact_i64(-1));
  EXPECT_EQ(varying_bounds.trip_count_kind, LOOM_LOOP_TRIP_COUNT_BOUNDED);
  EXPECT_EQ(varying_bounds.trip_count, 3u);
  ExpectNumericRange(varying_bounds.exit_value, 0, 3, 1);
  const auto aligned = loom_loop_domain_recurrence_facts(
      descending, 32, loom_value_facts_make(64, 128, 64),
      loom_value_facts_exact_i64(0), loom_value_facts_exact_i64(-64));
  EXPECT_EQ(aligned.trip_count_kind, LOOM_LOOP_TRIP_COUNT_BOUNDED);
  EXPECT_EQ(aligned.trip_count, 2u);
  EXPECT_EQ(aligned.values.known_divisor, 64);
  EXPECT_EQ(aligned.body_values.known_divisor, 64);
  EXPECT_EQ(aligned.exit_value.known_divisor, 64);
}

TEST(LoopDomainRecurrenceTest, DescendingSourceAndCarrierDomainsStayDistinct) {
  for (uint8_t bitwidth : {8, 16, 32, 64}) {
    const int64_t maximum = INT64_MAX >> (64 - bitwidth);
    const int64_t minimum = -maximum - 1;
    const auto signed_flags = kSignedExclusive | LOOM_LOOP_BOUND_DESCENDING;
    const auto unsigned_flags = kUnsignedExclusive | LOOM_LOOP_BOUND_DESCENDING;
    const auto full = loom_loop_domain_recurrence_facts(
        signed_flags, bitwidth, loom_value_facts_exact_i64(maximum),
        loom_value_facts_make(minimum, minimum + 1, 1),
        loom_value_facts_exact_i64(-1));
    EXPECT_EQ(full.trip_count_kind, LOOM_LOOP_TRIP_COUNT_BOUNDED);
    EXPECT_EQ(full.trip_count, UINT64_MAX >> (64 - bitwidth));
    ExpectNumericRange(full.values, minimum, maximum, 1);
    ExpectNumericRange(full.body_values, minimum + 1, maximum, 1);
    const auto crossing = loom_loop_domain_recurrence_facts(
        unsigned_flags, bitwidth,
        loom_value_facts_make(minimum, minimum + 1, 1),
        loom_value_facts_exact_i64(maximum), loom_value_facts_exact_i64(-1));
    EXPECT_EQ(crossing.trip_count_kind, LOOM_LOOP_TRIP_COUNT_BOUNDED);
    EXPECT_EQ(crossing.trip_count, 2u);
    EXPECT_TRUE(loom_value_facts_is_unknown(crossing.values));
    EXPECT_TRUE(loom_value_facts_is_unknown(crossing.exit_value));
    const auto negative_partition = loom_loop_domain_recurrence_facts(
        unsigned_flags, bitwidth, loom_value_facts_make(-2, -1, 1),
        loom_value_facts_exact_i64(-4), loom_value_facts_exact_i64(-1));
    EXPECT_EQ(negative_partition.trip_count_kind, LOOM_LOOP_TRIP_COUNT_BOUNDED);
    EXPECT_EQ(negative_partition.trip_count, 3u);
    ExpectNumericRange(negative_partition.values, -4, -1, 1);
    ExpectNumericRange(negative_partition.body_values, -3, -1, 1);
  }
  const auto flags = kSignedExclusive | LOOM_LOOP_BOUND_DESCENDING;
  const auto exact = loom_loop_domain_recurrence_facts(
      flags, 64, loom_value_facts_exact_i64(0),
      loom_value_facts_exact_i64(INT64_MIN),
      loom_value_facts_exact_i64(INT64_MIN));
  EXPECT_EQ(exact.trip_count_kind, LOOM_LOOP_TRIP_COUNT_EXACT);
  EXPECT_EQ(exact.trip_count, 1u);
  ExpectNumericRange(exact.values, INT64_MIN, 0, 1);
  ExpectNumericRange(exact.exit_value, INT64_MIN, INT64_MIN, 1);
  const auto bounded = loom_loop_domain_recurrence_facts(
      flags, 64, loom_value_facts_make(0, INT64_MAX, 1),
      loom_value_facts_exact_i64(-1), loom_value_facts_exact_i64(INT64_MIN));
  EXPECT_EQ(bounded.trip_count_kind, LOOM_LOOP_TRIP_COUNT_BOUNDED);
  EXPECT_EQ(bounded.trip_count, 1u);
  ExpectNumericRange(bounded.values, INT64_MIN, INT64_MAX, 1);
  ExpectNumericRange(bounded.body_values, 0, INT64_MAX, 1);
  const auto unsafe = loom_loop_domain_recurrence_facts(
      flags, 64, loom_value_facts_make(-1, 0, 1),
      loom_value_facts_exact_i64(INT64_MIN),
      loom_value_facts_exact_i64(INT64_MIN));
  EXPECT_EQ(unsafe.trip_count_kind, LOOM_LOOP_TRIP_COUNT_UNKNOWN);
  const auto unsigned_flags = kUnsignedExclusive | LOOM_LOOP_BOUND_DESCENDING;
  const auto unsigned_carrier = loom_loop_domain_recurrence_facts(
      unsigned_flags, 8, loom_value_facts_make(130, 131, 1),
      loom_value_facts_exact_i64(128), loom_value_facts_exact_i64(-1));
  EXPECT_EQ(unsigned_carrier.trip_count_kind, LOOM_LOOP_TRIP_COUNT_BOUNDED);
  EXPECT_EQ(unsigned_carrier.trip_count, 3u);
  EXPECT_TRUE(loom_value_facts_is_unknown(unsigned_carrier.values));
  const auto discontinuous = loom_loop_domain_recurrence_facts(
      unsigned_flags, 8, loom_value_facts_make(-1, 1, 1),
      loom_value_facts_exact_i64(0), loom_value_facts_exact_i64(-1));
  EXPECT_EQ(discontinuous.trip_count_kind, LOOM_LOOP_TRIP_COUNT_UNKNOWN);
  const auto outside = loom_loop_domain_recurrence_facts(
      flags, 32,
      loom_value_facts_make(INT64_C(1) << 32, (INT64_C(1) << 32) + 1, 1),
      loom_value_facts_exact_i64(0), loom_value_facts_exact_i64(-1));
  EXPECT_EQ(outside.trip_count_kind, LOOM_LOOP_TRIP_COUNT_UNKNOWN);
}

TEST(LoopDomainRecurrenceTest, SmallCarrierNextGuardExecution) {
  for (uint8_t bitwidth = 1; bitwidth <= kPresubmitCarrierBitwidth;
       ++bitwidth) {
    const uint64_t limit = UINT64_C(1) << bitwidth;
    for (auto flags : kComparisonFlags) {
      for (uint64_t initial = 0; initial < limit; ++initial) {
        for (uint64_t bound = 0; bound < limit; ++bound) {
          for (uint64_t step = 0; step < limit; ++step) {
            ASSERT_TRUE(loom::testing::CheckNextGuardLoopDomainExecution(
                flags, bitwidth, initial, bound, step));
          }
        }
      }
    }
  }
}

TEST(LoopDomainTripCountTest, SmallCarrierModularExecution) {
  for (uint8_t bitwidth = 1; bitwidth <= kPresubmitCarrierBitwidth;
       ++bitwidth) {
    const uint64_t limit = UINT64_C(1) << bitwidth;
    for (auto flags : kComparisonFlags) {
      for (uint64_t initial = 0; initial < limit; ++initial) {
        for (uint64_t bound = 0; bound < limit; ++bound) {
          for (uint64_t step = 0; step < limit; ++step) {
            ASSERT_TRUE(loom::testing::CheckLoopDomainExecution(
                flags, bitwidth, initial, bound, step));
          }
        }
      }
    }
  }
}

TEST(LoopDomainRecurrenceTest, SmallRangesContainEveryModularExecution) {
  auto contains = [](loom_value_facts_t facts, int64_t value) {
    return facts.range_lo <= value && value <= facts.range_hi &&
           value % facts.known_divisor == 0;
  };
  for (uint8_t bitwidth = 1; bitwidth <= 3; ++bitwidth) {
    const uint64_t modulus = UINT64_C(1) << bitwidth;
    const int64_t minimum = -int64_t(modulus / 2);
    const int64_t maximum = int64_t(modulus / 2) - 1;
    for (auto flags : kComparisonFlags) {
      for (int64_t initial_lower = minimum; initial_lower <= maximum;
           ++initial_lower) {
        for (int64_t initial_upper = initial_lower; initial_upper <= maximum;
             ++initial_upper) {
          for (int64_t bound_lower = minimum; bound_lower <= maximum;
               ++bound_lower) {
            for (int64_t bound_upper = bound_lower; bound_upper <= maximum;
                 ++bound_upper) {
              for (uint64_t step = 1; step < modulus; ++step) {
                const int64_t delta = flags & LOOM_LOOP_BOUND_DESCENDING
                                          ? -int64_t(step)
                                          : int64_t(step);
                const auto facts = loom_loop_domain_recurrence_facts(
                    flags, bitwidth,
                    loom_value_facts_make(initial_lower, initial_upper, 1),
                    loom_value_facts_make(bound_lower, bound_upper, 1),
                    loom_value_facts_exact_i64(delta));
                if (facts.trip_count_kind == LOOM_LOOP_TRIP_COUNT_UNKNOWN) {
                  EXPECT_EQ(facts.trip_count, 0u);
                  continue;
                }
                for (int64_t initial = initial_lower; initial <= initial_upper;
                     ++initial) {
                  for (int64_t bound = bound_lower; bound <= bound_upper;
                       ++bound) {
                    const auto observed =
                        loom::testing::InterpretLoopDomainExecution(
                            flags, bitwidth, uint64_t(initial) & (modulus - 1),
                            uint64_t(bound) & (modulus - 1),
                            uint64_t(delta) & (modulus - 1));
                    const bool exact =
                        facts.trip_count_kind == LOOM_LOOP_TRIP_COUNT_EXACT;
                    if (!observed.terminates || !observed.advances ||
                        observed.trip_count > facts.trip_count ||
                        (exact && observed.trip_count != facts.trip_count) ||
                        !contains(facts.values, observed.minimum) ||
                        !contains(facts.values, observed.maximum) ||
                        !contains(facts.exit_value, observed.exit_value) ||
                        (observed.trip_count != 0 &&
                         (!contains(facts.body_values, observed.body_minimum) ||
                          !contains(facts.body_values,
                                    observed.body_maximum)))) {
                      FAIL() << "width=" << int(bitwidth)
                             << " flags=" << int(flags) << " initial=["
                             << initial_lower << "," << initial_upper
                             << "] bound=[" << bound_lower << "," << bound_upper
                             << "] step=" << delta << " witness=" << initial
                             << "," << bound;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace
}  // namespace loom
