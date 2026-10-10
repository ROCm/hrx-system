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

struct ObservedLoop {
  // Whether modular execution reaches a false guard before revisiting a value.
  bool terminates;
  // Whether every executed addition advances in the selected direction.
  bool advances;
  // Number of body executions before exit or recurrence.
  uint64_t trip_count;
  // Whether each increment advances in the signed source representation too.
  bool source_advances;
  // Smallest signed source value observed at the header, including its exit.
  int64_t minimum;
  // Largest signed source value observed at the header, including its exit.
  int64_t maximum;
  // Smallest signed source value observed during an executed body.
  int64_t body_minimum;
  // Largest signed source value observed during an executed body.
  int64_t body_maximum;
  // Signed source value at the first false guard, if execution terminates.
  int64_t exit_value;
};

// Small-width interpretation is independent of the closed-form count proof.
// Execute carrier additions and comparisons, stopping when a value repeats.
ObservedLoop InterpretLoop(loom_loop_bound_flags_t bound_flags,
                           uint8_t bitwidth, uint64_t initial, uint64_t bound,
                           uint64_t step) {
  const int64_t modulus = INT64_C(1) << bitwidth;
  const bool is_signed = bound_flags & LOOM_LOOP_BOUND_SIGNED;
  const bool inclusive = bound_flags & LOOM_LOOP_BOUND_INCLUSIVE;
  const bool descending = bound_flags & LOOM_LOOP_BOUND_DESCENDING;
  auto ordered_value = [&](uint64_t bits) -> int64_t {
    return is_signed && bits >= uint64_t(modulus / 2) ? int64_t(bits) - modulus
                                                      : int64_t(bits);
  };
  auto source_value = [&](uint64_t bits) -> int64_t {
    return bits >= uint64_t(modulus / 2) ? int64_t(bits) - modulus
                                         : int64_t(bits);
  };
  ObservedLoop observed = {true,
                           true,
                           0,
                           true,
                           source_value(initial),
                           source_value(initial),
                           INT64_MAX,
                           INT64_MIN,
                           0};
  uint64_t value = initial;
  do {
    const int64_t current = ordered_value(value);
    const int64_t limit = ordered_value(bound);
    const bool more = (descending ? current > limit : current < limit) ||
                      (inclusive && current == limit);
    if (!more) {
      observed.exit_value = source_value(value);
      return observed;
    }
    observed.body_minimum =
        std::min(observed.body_minimum, source_value(value));
    observed.body_maximum =
        std::max(observed.body_maximum, source_value(value));
    uint64_t next = (value + step) % modulus;
    observed.advances &= descending ? ordered_value(next) < current
                                    : ordered_value(next) > current;
    observed.source_advances &= descending
                                    ? source_value(next) < source_value(value)
                                    : source_value(next) > source_value(value);
    observed.minimum = std::min(observed.minimum, source_value(next));
    observed.maximum = std::max(observed.maximum, source_value(next));
    ++observed.trip_count;
    value = next;
  } while (value != initial);
  observed.terminates = false;
  return observed;
}

TEST(LoopDomainTripCountTest, ExhaustiveModularExecution) {
  for (uint8_t bitwidth = 1; bitwidth <= 6; ++bitwidth) {
    const uint64_t limit = UINT64_C(1) << bitwidth;
    for (auto bound_flags : kComparisonFlags) {
      for (uint64_t initial = 0; initial < limit; ++initial) {
        for (uint64_t bound = 0; bound < limit; ++bound) {
          for (uint64_t step = 0; step < limit; ++step) {
            const ObservedLoop expected =
                InterpretLoop(bound_flags, bitwidth, initial, bound, step);
            uint64_t actual = UINT64_MAX;
            const bool exact = loom_loop_domain_trip_count(
                bound_flags, bitwidth, initial, bound, step, &actual);
            const bool expected_exact =
                expected.terminates && expected.advances;
            auto source_value = [&](uint64_t bits) -> int64_t {
              return bits >= limit / 2 ? int64_t(bits) - int64_t(limit)
                                       : int64_t(bits);
            };
            const auto facts = loom_loop_domain_recurrence_facts(
                bound_flags, bitwidth,
                loom_value_facts_exact_i64(source_value(initial)),
                loom_value_facts_exact_i64(source_value(bound)),
                loom_value_facts_exact_i64(source_value(step)));
            const bool expected_range =
                expected_exact && (expected.trip_count == 0 ||
                                   ((bound_flags & LOOM_LOOP_BOUND_DESCENDING
                                         ? source_value(step) < 0
                                         : source_value(step) > 0) &&
                                    expected.source_advances));
            if ((facts.trip_count_kind == LOOM_LOOP_TRIP_COUNT_EXACT) !=
                    exact ||
                facts.trip_count != actual ||
                (expected_range &&
                 (facts.values.range_lo != expected.minimum ||
                  facts.values.range_hi != expected.maximum)) ||
                (!expected_range &&
                 !loom_value_facts_is_unknown(facts.values))) {
              FAIL() << "range width=" << int(bitwidth)
                     << " flags=" << int(bound_flags) << " initial=" << initial
                     << " bound=" << bound << " step=" << step << " actual=["
                     << facts.values.range_lo << "," << facts.values.range_hi
                     << "] expected_range=" << expected_range << " expected=["
                     << expected.minimum << "," << expected.maximum << "]";
            }
            if (exact != expected_exact ||
                actual != (expected_exact ? expected.trip_count : 0)) {
              FAIL() << "width=" << int(bitwidth)
                     << " bound_flags=" << int(bound_flags)
                     << " initial=" << initial << " bound=" << bound
                     << " step=" << step << " exact=" << exact
                     << " count=" << actual
                     << " expected_exact=" << expected_exact
                     << " expected_count=" << expected.trip_count;
            }
            if (expected_range) {
              ASSERT_EQ(facts.exit_value.range_lo, expected.exit_value);
              ASSERT_EQ(facts.exit_value.range_hi, expected.exit_value);
              if (expected.trip_count != 0) {
                ASSERT_EQ(facts.body_values.range_lo, expected.body_minimum);
                ASSERT_EQ(facts.body_values.range_hi, expected.body_maximum);
              } else {
                ASSERT_TRUE(loom_value_facts_is_unknown(facts.body_values));
              }
            } else {
              ASSERT_TRUE(loom_value_facts_is_unknown(facts.body_values));
              ASSERT_TRUE(loom_value_facts_is_unknown(facts.exit_value));
            }
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
                    const auto observed = InterpretLoop(
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
