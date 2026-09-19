// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/loop_domain.h"

#include <algorithm>
#include <cstdint>

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

TEST(LoopDomainRecurrenceTest, FullSignedCarrierAndTerminalBoundaries) {
  for (const uint8_t bitwidth : {32, 64}) {
    const int64_t maximum = INT64_MAX >> (64 - bitwidth);
    const int64_t minimum = -maximum - 1;
    const auto full = loom_loop_domain_recurrence_facts(
        kSignedExclusive, bitwidth, minimum, maximum, 1);
    EXPECT_TRUE(full.trip_count_known);
    EXPECT_EQ(full.values.range_lo, minimum);
    EXPECT_EQ(full.values.range_hi, maximum);
    const auto tail = loom_loop_domain_recurrence_facts(
        kSignedExclusive, bitwidth, maximum - 1, maximum, 2);
    EXPECT_FALSE(tail.trip_count_known);
    EXPECT_TRUE(loom_value_facts_is_unknown(tail.values));
    const auto crossing = loom_loop_domain_recurrence_facts(
        kUnsignedExclusive, bitwidth, maximum, minimum, 1);
    EXPECT_TRUE(crossing.trip_count_known);
    EXPECT_EQ(crossing.trip_count, 1u);
    EXPECT_TRUE(loom_value_facts_is_unknown(crossing.values));
  }
}

TEST(LoopDomainRecurrenceTest, SourceValueMustFitCarrier) {
  const auto facts = loom_loop_domain_recurrence_facts(kSignedExclusive, 32,
                                                       INT64_C(1) << 32, 4, 1);
  EXPECT_TRUE(facts.trip_count_known);
  EXPECT_EQ(facts.trip_count, 4u);
  EXPECT_TRUE(loom_value_facts_is_unknown(facts.values));
}

struct ObservedLoop {
  // Whether modular execution reaches a false guard before revisiting a value.
  bool terminates;
  // Whether every executed addition advances in the bound's comparison order.
  bool increases;
  // Number of body executions before exit or recurrence.
  uint64_t trip_count;
  // Whether each increment also advances in the signed source representation.
  bool source_increases;
  // Smallest signed source value observed at the header, including its exit.
  int64_t minimum;
  // Largest signed source value observed at the header, including its exit.
  int64_t maximum;
};

// Small-width interpretation is independent of the closed-form count proof.
// Execute carrier additions and comparisons, stopping when a value repeats.
ObservedLoop InterpretLoop(loom_loop_bound_flags_t bound_flags,
                           uint8_t bitwidth, uint64_t initial, uint64_t bound,
                           uint64_t step) {
  const int64_t modulus = INT64_C(1) << bitwidth;
  const bool is_signed =
      bound_flags == kSignedExclusive || bound_flags == kSignedInclusive;
  auto ordered_value = [&](uint64_t bits) -> int64_t {
    return is_signed && bits >= uint64_t(modulus / 2) ? int64_t(bits) - modulus
                                                      : int64_t(bits);
  };
  auto source_value = [&](uint64_t bits) -> int64_t {
    return bits >= uint64_t(modulus / 2) ? int64_t(bits) - modulus
                                         : int64_t(bits);
  };
  ObservedLoop observed = {
      true, true, 0, true, source_value(initial), source_value(initial)};
  uint64_t value = initial;
  do {
    bool more =
        bound_flags == kSignedInclusive || bound_flags == kUnsignedInclusive
            ? ordered_value(value) <= ordered_value(bound)
            : ordered_value(value) < ordered_value(bound);
    if (!more) {
      return observed;
    }
    uint64_t next = (value + step) % modulus;
    observed.increases &= ordered_value(next) > ordered_value(value);
    observed.source_increases &= source_value(next) > source_value(value);
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
    for (loom_loop_bound_flags_t bound_flags :
         {kSignedExclusive, kSignedInclusive, kUnsignedExclusive,
          kUnsignedInclusive}) {
      for (uint64_t initial = 0; initial < limit; ++initial) {
        for (uint64_t bound = 0; bound < limit; ++bound) {
          for (uint64_t step = 0; step < limit; ++step) {
            const ObservedLoop expected =
                InterpretLoop(bound_flags, bitwidth, initial, bound, step);
            uint64_t actual = UINT64_MAX;
            const bool exact = loom_loop_domain_trip_count(
                bound_flags, bitwidth, initial, bound, step, &actual);
            const bool expected_exact =
                expected.terminates && expected.increases;
            auto source_value = [&](uint64_t bits) -> int64_t {
              return bits >= limit / 2 ? int64_t(bits) - int64_t(limit)
                                       : int64_t(bits);
            };
            const auto facts = loom_loop_domain_recurrence_facts(
                bound_flags, bitwidth, source_value(initial),
                source_value(bound), source_value(step));
            const bool expected_range =
                expected_exact &&
                (expected.trip_count == 0 ||
                 (source_value(step) > 0 && expected.source_increases));
            if (facts.trip_count_known != exact || facts.trip_count != actual ||
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
          }
        }
      }
    }
  }
}

}  // namespace
}  // namespace loom
