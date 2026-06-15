// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/facts.h"

#include <climits>
#include <cmath>
#include <cstdint>

#include "iree/testing/gtest.h"

namespace loom {
namespace {

//===----------------------------------------------------------------------===//
// Constructors
//===----------------------------------------------------------------------===//

TEST(FactsUnknown, IsUnknown) {
  loom_value_facts_t f = loom_value_facts_unknown();
  EXPECT_TRUE(loom_value_facts_is_unknown(f));
  EXPECT_EQ(f.range_lo, INT64_MIN);
  EXPECT_EQ(f.range_hi, INT64_MAX);
  EXPECT_EQ(f.known_divisor, 1);
  EXPECT_EQ(f.flags, 0u);
}

TEST(FactsUnknown, NoFlagsSet) {
  loom_value_facts_t f = loom_value_facts_unknown();
  EXPECT_FALSE(loom_value_facts_is_exact(f));
  EXPECT_FALSE(loom_value_facts_is_non_negative(f));
  EXPECT_FALSE(loom_value_facts_is_non_zero(f));
  EXPECT_FALSE(loom_value_facts_is_positive(f));
  EXPECT_FALSE(loom_value_facts_is_power_of_two(f));
  EXPECT_FALSE(loom_value_facts_is_boolean(f));
}

TEST(FactsExactI64, Zero) {
  loom_value_facts_t f = loom_value_facts_exact_i64(0);
  EXPECT_TRUE(loom_value_facts_is_exact(f));
  EXPECT_TRUE(loom_value_facts_is_non_negative(f));
  EXPECT_TRUE(loom_value_facts_is_boolean(f));
  EXPECT_FALSE(loom_value_facts_is_non_zero(f));
  EXPECT_FALSE(loom_value_facts_is_positive(f));
  EXPECT_FALSE(loom_value_facts_is_power_of_two(f));
  EXPECT_EQ(f.known_divisor, 1);
}

TEST(FactsExactI64, One) {
  loom_value_facts_t f = loom_value_facts_exact_i64(1);
  EXPECT_TRUE(loom_value_facts_is_exact(f));
  EXPECT_TRUE(loom_value_facts_is_non_negative(f));
  EXPECT_TRUE(loom_value_facts_is_non_zero(f));
  EXPECT_TRUE(loom_value_facts_is_positive(f));
  EXPECT_TRUE(loom_value_facts_is_boolean(f));
  EXPECT_TRUE(loom_value_facts_is_power_of_two(f));
  EXPECT_EQ(f.known_divisor, 1);
}

TEST(FactsExactI64, NegativeOne) {
  loom_value_facts_t f = loom_value_facts_exact_i64(-1);
  EXPECT_TRUE(loom_value_facts_is_exact(f));
  EXPECT_TRUE(loom_value_facts_is_non_zero(f));
  EXPECT_FALSE(loom_value_facts_is_non_negative(f));
  EXPECT_FALSE(loom_value_facts_is_positive(f));
  EXPECT_FALSE(loom_value_facts_is_boolean(f));
  EXPECT_EQ(f.known_divisor, 1);
}

TEST(FactsExactI64, FortyTwo) {
  loom_value_facts_t f = loom_value_facts_exact_i64(42);
  EXPECT_TRUE(loom_value_facts_is_exact(f));
  EXPECT_TRUE(loom_value_facts_is_positive(f));
  EXPECT_FALSE(loom_value_facts_is_power_of_two(f));
  EXPECT_EQ(f.known_divisor, 42);
  EXPECT_TRUE(loom_value_facts_divisible_by(f, 6));
  EXPECT_TRUE(loom_value_facts_divisible_by(f, 7));
  EXPECT_TRUE(loom_value_facts_divisible_by(f, 42));
  EXPECT_FALSE(loom_value_facts_divisible_by(f, 5));
}

TEST(FactsExactI64, PowerOfTwo) {
  loom_value_facts_t f = loom_value_facts_exact_i64(64);
  EXPECT_TRUE(loom_value_facts_is_power_of_two(f));
  EXPECT_EQ(f.known_divisor, 64);
}

TEST(FactsExactI64, NegativePowerOfTwo) {
  loom_value_facts_t f = loom_value_facts_exact_i64(-64);
  EXPECT_TRUE(loom_value_facts_is_non_zero(f));
  EXPECT_FALSE(loom_value_facts_is_power_of_two(f));
  EXPECT_EQ(f.known_divisor, 64);
}

TEST(FactsExactI64, Int64Min) {
  // INT64_MIN: llabs is UB, so divisor falls back to 1.
  loom_value_facts_t f = loom_value_facts_exact_i64(INT64_MIN);
  EXPECT_TRUE(loom_value_facts_is_exact(f));
  EXPECT_TRUE(loom_value_facts_is_non_zero(f));
  EXPECT_FALSE(loom_value_facts_is_non_negative(f));
  EXPECT_EQ(f.known_divisor, 1);
}

TEST(FactsExactI64, Int64Max) {
  loom_value_facts_t f = loom_value_facts_exact_i64(INT64_MAX);
  EXPECT_TRUE(loom_value_facts_is_exact(f));
  EXPECT_TRUE(loom_value_facts_is_positive(f));
  EXPECT_EQ(f.known_divisor, INT64_MAX);
}

TEST(FactsExactF64, Pi) {
  loom_value_facts_t f = loom_value_facts_exact_f64(3.14159265358979);
  EXPECT_TRUE(loom_value_facts_is_exact(f));
  EXPECT_TRUE(loom_value_facts_is_float(f));
  EXPECT_DOUBLE_EQ(loom_value_facts_as_f64(f), 3.14159265358979);
}

TEST(FactsExactF64, Zero) {
  loom_value_facts_t f = loom_value_facts_exact_f64(0.0);
  EXPECT_TRUE(loom_value_facts_is_exact(f));
  EXPECT_TRUE(loom_value_facts_is_float(f));
  EXPECT_DOUBLE_EQ(loom_value_facts_as_f64(f), 0.0);
}

TEST(FactsExactF64, NegativeZeroDiffersFromPositive) {
  loom_value_facts_t pos = loom_value_facts_exact_f64(0.0);
  loom_value_facts_t neg = loom_value_facts_exact_f64(-0.0);
  // IEEE 754: +0.0 and -0.0 have different bit patterns.
  EXPECT_FALSE(loom_value_facts_equal(pos, neg));
}

TEST(FactsPredicates, IntegerTruthiness) {
  bool value = true;
  loom_value_facts_t unknown = loom_value_facts_unknown();
  EXPECT_FALSE(loom_value_facts_is_zero(unknown));
  EXPECT_FALSE(loom_value_facts_is_false(unknown));
  EXPECT_FALSE(loom_value_facts_is_true(unknown));
  EXPECT_FALSE(loom_value_facts_as_exact_bool(unknown, &value));

  loom_value_facts_t zero = loom_value_facts_exact_i64(0);
  EXPECT_TRUE(loom_value_facts_is_zero(zero));
  EXPECT_TRUE(loom_value_facts_is_false(zero));
  EXPECT_FALSE(loom_value_facts_is_true(zero));
  EXPECT_TRUE(loom_value_facts_as_exact_bool(zero, &value));
  EXPECT_FALSE(value);

  loom_value_facts_t one = loom_value_facts_exact_i64(1);
  EXPECT_FALSE(loom_value_facts_is_zero(one));
  EXPECT_FALSE(loom_value_facts_is_false(one));
  EXPECT_TRUE(loom_value_facts_is_true(one));
  EXPECT_TRUE(loom_value_facts_as_exact_bool(one, &value));
  EXPECT_TRUE(value);

  loom_value_facts_t negative = loom_value_facts_exact_i64(-7);
  EXPECT_FALSE(loom_value_facts_is_zero(negative));
  EXPECT_TRUE(loom_value_facts_is_true(negative));
  EXPECT_TRUE(loom_value_facts_as_exact_bool(negative, &value));
  EXPECT_TRUE(value);
}

TEST(FactsPredicates, NonExactTruthiness) {
  bool value = false;
  loom_value_facts_t positive_range = loom_value_facts_make(1, 10, 1);
  EXPECT_TRUE(loom_value_facts_is_true(positive_range));
  EXPECT_FALSE(loom_value_facts_as_exact_bool(positive_range, &value));

  loom_value_facts_t negative_range = loom_value_facts_make(-10, -1, 1);
  EXPECT_TRUE(loom_value_facts_is_true(negative_range));
  EXPECT_FALSE(loom_value_facts_as_exact_bool(negative_range, &value));

  loom_value_facts_t maybe_zero = loom_value_facts_make(0, 10, 1);
  EXPECT_FALSE(loom_value_facts_is_zero(maybe_zero));
  EXPECT_FALSE(loom_value_facts_is_false(maybe_zero));
  EXPECT_FALSE(loom_value_facts_is_true(maybe_zero));
}

TEST(FactsPredicates, FloatFactsAreNotIntegerTruthiness) {
  bool value = true;
  loom_value_facts_t zero = loom_value_facts_exact_f64(0.0);
  EXPECT_FALSE(loom_value_facts_is_zero(zero));
  EXPECT_FALSE(loom_value_facts_is_false(zero));
  EXPECT_FALSE(loom_value_facts_is_true(zero));
  EXPECT_FALSE(loom_value_facts_as_exact_bool(zero, &value));
}

//===----------------------------------------------------------------------===//
// General constructor: make
//===----------------------------------------------------------------------===//

TEST(FactsMake, NonNegativeRange) {
  loom_value_facts_t f = loom_value_facts_make(0, 100, 1);
  EXPECT_TRUE(loom_value_facts_is_non_negative(f));
  EXPECT_FALSE(loom_value_facts_is_non_zero(f));
  EXPECT_FALSE(loom_value_facts_is_exact(f));
  EXPECT_EQ(f.range_lo, 0);
  EXPECT_EQ(f.range_hi, 100);
}

TEST(FactsMake, PositiveRange) {
  loom_value_facts_t f = loom_value_facts_make(1, 100, 1);
  EXPECT_TRUE(loom_value_facts_is_positive(f));
  EXPECT_TRUE(loom_value_facts_is_non_zero(f));
}

TEST(FactsMake, NegativeRange) {
  loom_value_facts_t f = loom_value_facts_make(-100, -1, 1);
  EXPECT_TRUE(loom_value_facts_is_non_zero(f));
  EXPECT_FALSE(loom_value_facts_is_non_negative(f));
}

TEST(FactsMake, SpanningZero) {
  loom_value_facts_t f = loom_value_facts_make(-50, 50, 1);
  EXPECT_FALSE(loom_value_facts_is_non_negative(f));
  EXPECT_FALSE(loom_value_facts_is_non_zero(f));
}

TEST(FactsMake, BooleanRange) {
  loom_value_facts_t f = loom_value_facts_make(0, 1, 1);
  EXPECT_TRUE(loom_value_facts_is_boolean(f));
  EXPECT_TRUE(loom_value_facts_is_non_negative(f));
}

TEST(FactsMake, SinglePoint) {
  loom_value_facts_t f = loom_value_facts_make(42, 42, 42);
  EXPECT_TRUE(loom_value_facts_is_exact(f));
  EXPECT_EQ(f.known_divisor, 42);
}

TEST(FactsMake, SinglePointStrengthensDivisor) {
  loom_value_facts_t f = loom_value_facts_make(42, 42, 1);
  EXPECT_TRUE(loom_value_facts_is_exact(f));
  EXPECT_EQ(f.known_divisor, 42);
}

TEST(FactsMake, DivisorClamped) {
  // Divisor < 1 gets clamped to 1.
  loom_value_facts_t f = loom_value_facts_make(0, 100, 0);
  EXPECT_EQ(f.known_divisor, 1);
  f = loom_value_facts_make(0, 100, -5);
  EXPECT_EQ(f.known_divisor, 1);
}

TEST(FactsMake, InvalidRange) {
  // lo > hi: returns unknown.
  loom_value_facts_t f = loom_value_facts_make(100, 50, 1);
  EXPECT_TRUE(loom_value_facts_is_unknown(f));
}

//===----------------------------------------------------------------------===//
// Bit-width fit predicates
//===----------------------------------------------------------------------===//

TEST(FactsFitBitCount, SignedRange) {
  EXPECT_TRUE(loom_value_facts_fit_signed_bit_count(
      loom_value_facts_make(INT32_MIN, INT32_MAX, 1), 32));
  EXPECT_FALSE(loom_value_facts_fit_signed_bit_count(
      loom_value_facts_make(0, (int64_t)UINT32_MAX, 1), 32));
  EXPECT_FALSE(
      loom_value_facts_fit_signed_bit_count(loom_value_facts_unknown(), 32));
}

TEST(FactsFitBitCount, UnsignedRange) {
  EXPECT_TRUE(loom_value_facts_fit_unsigned_bit_count(
      loom_value_facts_make(0, (int64_t)UINT32_MAX, 1), 32));
  EXPECT_FALSE(loom_value_facts_fit_unsigned_bit_count(
      loom_value_facts_make(-1, INT32_MAX, 1), 32));
  EXPECT_FALSE(
      loom_value_facts_fit_unsigned_bit_count(loom_value_facts_unknown(), 32));
}

//===----------------------------------------------------------------------===//
// Equality
//===----------------------------------------------------------------------===//

TEST(FactsEqual, SameAreEqual) {
  loom_value_facts_t a = loom_value_facts_exact_i64(42);
  loom_value_facts_t b = loom_value_facts_exact_i64(42);
  EXPECT_TRUE(loom_value_facts_equal(a, b));
}

TEST(FactsEqual, DifferentAreNotEqual) {
  loom_value_facts_t a = loom_value_facts_exact_i64(42);
  loom_value_facts_t b = loom_value_facts_exact_i64(43);
  EXPECT_FALSE(loom_value_facts_equal(a, b));
}

TEST(FactsEqual, UnknownsAreEqual) {
  loom_value_facts_t a = loom_value_facts_unknown();
  loom_value_facts_t b = loom_value_facts_unknown();
  EXPECT_TRUE(loom_value_facts_equal(a, b));
}

//===----------------------------------------------------------------------===//
// Meet
//===----------------------------------------------------------------------===//

TEST(FactsMeet, WidensRange) {
  loom_value_facts_t a = loom_value_facts_make(10, 20, 4);
  loom_value_facts_t b = loom_value_facts_make(5, 25, 6);
  loom_value_facts_t out;
  loom_value_facts_meet(&a, &b, &out);
  EXPECT_EQ(out.range_lo, 5);
  EXPECT_EQ(out.range_hi, 25);
  EXPECT_EQ(out.known_divisor, 2);  // gcd(4, 6) = 2.
}

TEST(FactsMeet, ExactValues) {
  loom_value_facts_t a = loom_value_facts_exact_i64(5);
  loom_value_facts_t b = loom_value_facts_exact_i64(10);
  loom_value_facts_t out;
  loom_value_facts_meet(&a, &b, &out);
  EXPECT_EQ(out.range_lo, 5);
  EXPECT_EQ(out.range_hi, 10);
  EXPECT_EQ(out.known_divisor, 5);  // gcd(5, 10) = 5.
  EXPECT_FALSE(loom_value_facts_is_exact(out));
}

TEST(FactsMeet, WithUnknown) {
  loom_value_facts_t a = loom_value_facts_unknown();
  loom_value_facts_t b = loom_value_facts_exact_i64(5);
  loom_value_facts_t out;
  loom_value_facts_meet(&a, &b, &out);
  EXPECT_TRUE(loom_value_facts_is_unknown(out));
}

TEST(FactsMeet, OverlappingRanges) {
  loom_value_facts_t a = loom_value_facts_make(0, 100, 8);
  loom_value_facts_t b = loom_value_facts_make(50, 200, 12);
  loom_value_facts_t out;
  loom_value_facts_meet(&a, &b, &out);
  EXPECT_EQ(out.range_lo, 0);
  EXPECT_EQ(out.range_hi, 200);
  EXPECT_EQ(out.known_divisor, 4);  // gcd(8, 12) = 4.
}

//===----------------------------------------------------------------------===//
// Predicate application
//===----------------------------------------------------------------------===//

static loom_predicate_t make_predicate_1(loom_predicate_kind_t kind,
                                         int64_t arg) {
  loom_predicate_t pred = {0};
  pred.kind = (uint8_t)kind;
  pred.arg_count = 2;
  pred.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  pred.arg_tags[1] = LOOM_PRED_ARG_CONST;
  pred.args[1] = arg;
  return pred;
}

static loom_predicate_t make_predicate_range(int64_t lo, int64_t hi) {
  loom_predicate_t pred = {0};
  pred.kind = (uint8_t)LOOM_PREDICATE_RANGE;
  pred.arg_count = 3;
  pred.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  pred.arg_tags[1] = LOOM_PRED_ARG_CONST;
  pred.arg_tags[2] = LOOM_PRED_ARG_CONST;
  pred.args[1] = lo;
  pred.args[2] = hi;
  return pred;
}

static loom_predicate_t make_predicate_pow2(void) {
  loom_predicate_t pred = {0};
  pred.kind = (uint8_t)LOOM_PREDICATE_POW2;
  pred.arg_count = 1;
  pred.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  return pred;
}

TEST(FactsApplyPredicate, Eq) {
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t pred = make_predicate_1(LOOM_PREDICATE_EQ, 42);
  loom_value_facts_apply_predicate(&f, &pred);
  EXPECT_TRUE(loom_value_facts_is_exact(f));
  EXPECT_EQ(f.range_lo, 42);
  EXPECT_EQ(f.range_hi, 42);
}

TEST(FactsApplyPredicate, ValueBoundDoesNotCorruptRange) {
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t pred = make_predicate_1(LOOM_PREDICATE_LT, 42);
  pred.arg_tags[1] = LOOM_PRED_ARG_VALUE;
  pred.args[1] = 7;
  loom_value_facts_apply_predicate(&f, &pred);
  EXPECT_TRUE(loom_value_facts_is_unknown(f));
}

TEST(FactsApplyPredicate, NeIsRepresentedButDoesNotTightenInterval) {
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t pred = make_predicate_1(LOOM_PREDICATE_NE, 42);
  loom_value_facts_apply_predicate(&f, &pred);
  EXPECT_TRUE(loom_value_facts_is_unknown(f));
}

TEST(FactsApplyPredicate, Ge) {
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t pred = make_predicate_1(LOOM_PREDICATE_GE, 10);
  loom_value_facts_apply_predicate(&f, &pred);
  EXPECT_EQ(f.range_lo, 10);
  EXPECT_EQ(f.range_hi, INT64_MAX);
  EXPECT_TRUE(loom_value_facts_is_non_negative(f));
  EXPECT_TRUE(loom_value_facts_is_positive(f));
}

TEST(FactsApplyPredicate, Le) {
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t pred = make_predicate_1(LOOM_PREDICATE_LE, 100);
  loom_value_facts_apply_predicate(&f, &pred);
  EXPECT_EQ(f.range_lo, INT64_MIN);
  EXPECT_EQ(f.range_hi, 100);
}

TEST(FactsApplyPredicate, Mul) {
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t pred = make_predicate_1(LOOM_PREDICATE_MUL, 16);
  loom_value_facts_apply_predicate(&f, &pred);
  EXPECT_EQ(f.known_divisor, 16);
  EXPECT_TRUE(loom_value_facts_divisible_by(f, 16));
  EXPECT_TRUE(loom_value_facts_divisible_by(f, 8));
  EXPECT_TRUE(loom_value_facts_divisible_by(f, 4));
}

TEST(FactsApplyPredicate, MulComposition) {
  // mul(16) then mul(24) → lcm(16, 24) = 48.
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t pred1 = make_predicate_1(LOOM_PREDICATE_MUL, 16);
  loom_predicate_t pred2 = make_predicate_1(LOOM_PREDICATE_MUL, 24);
  loom_value_facts_apply_predicate(&f, &pred1);
  loom_value_facts_apply_predicate(&f, &pred2);
  EXPECT_EQ(f.known_divisor, 48);
}

TEST(FactsApplyPredicate, Min) {
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t pred = make_predicate_1(LOOM_PREDICATE_MIN, 64);
  loom_value_facts_apply_predicate(&f, &pred);
  EXPECT_EQ(f.range_lo, 64);
  EXPECT_TRUE(loom_value_facts_is_positive(f));
}

TEST(FactsApplyPredicate, Max) {
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t pred = make_predicate_1(LOOM_PREDICATE_MAX, 1024);
  loom_value_facts_apply_predicate(&f, &pred);
  EXPECT_EQ(f.range_hi, 1024);
}

TEST(FactsApplyPredicate, Pow2) {
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t pred = make_predicate_pow2();
  loom_value_facts_apply_predicate(&f, &pred);
  EXPECT_TRUE(loom_value_facts_is_power_of_two(f));
}

TEST(FactsApplyPredicate, Range) {
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t pred = make_predicate_range(10, 100);
  loom_value_facts_apply_predicate(&f, &pred);
  EXPECT_EQ(f.range_lo, 10);
  EXPECT_EQ(f.range_hi, 100);
  EXPECT_TRUE(loom_value_facts_is_positive(f));
}

TEST(FactsApplyPredicate, ComposedTilePredicate) {
  // Typical tile dimension: mul(64), min(64).
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t mul_pred = make_predicate_1(LOOM_PREDICATE_MUL, 64);
  loom_predicate_t min_pred = make_predicate_1(LOOM_PREDICATE_MIN, 64);
  loom_value_facts_apply_predicate(&f, &mul_pred);
  loom_value_facts_apply_predicate(&f, &min_pred);
  EXPECT_EQ(f.range_lo, 64);
  EXPECT_EQ(f.range_hi, INT64_MAX);
  EXPECT_EQ(f.known_divisor, 64);
  EXPECT_TRUE(loom_value_facts_is_positive(f));
  EXPECT_TRUE(loom_value_facts_divisible_by(f, 16));
}

//===----------------------------------------------------------------------===//
// Transfer functions: addi
//===----------------------------------------------------------------------===//

TEST(AddiTransfer, ExactPlusExact) {
  loom_value_facts_t a = loom_value_facts_exact_i64(3);
  loom_value_facts_t b = loom_value_facts_exact_i64(4);
  loom_value_facts_t out;
  loom_value_facts_addi(&a, &b, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, 7);
  EXPECT_EQ(out.known_divisor, 7);
}

TEST(AddiTransfer, RangePlusExact) {
  loom_value_facts_t a = loom_value_facts_make(10, 20, 4);
  loom_value_facts_t b = loom_value_facts_exact_i64(8);
  loom_value_facts_t out;
  loom_value_facts_addi(&a, &b, &out);
  EXPECT_EQ(out.range_lo, 18);
  EXPECT_EQ(out.range_hi, 28);
  EXPECT_EQ(out.known_divisor, 4);  // gcd(4, 8) = 4.
}

TEST(AddiTransfer, RangePlusRange) {
  loom_value_facts_t a = loom_value_facts_make(0, 100, 16);
  loom_value_facts_t b = loom_value_facts_make(0, 100, 8);
  loom_value_facts_t out;
  loom_value_facts_addi(&a, &b, &out);
  EXPECT_EQ(out.range_lo, 0);
  EXPECT_EQ(out.range_hi, 200);
  EXPECT_EQ(out.known_divisor, 8);  // gcd(16, 8) = 8.
}

TEST(AddiTransfer, Overflow) {
  loom_value_facts_t a = loom_value_facts_exact_i64(INT64_MAX);
  loom_value_facts_t b = loom_value_facts_exact_i64(1);
  loom_value_facts_t out;
  loom_value_facts_addi(&a, &b, &out);
  EXPECT_EQ(out.range_lo, INT64_MIN);
  EXPECT_EQ(out.range_hi, INT64_MAX);
  EXPECT_TRUE(loom_value_facts_is_uniform(out));
}

TEST(AddiTransfer, InPlaceAccumulation) {
  // Output aliases lhs.
  loom_value_facts_t accumulator = loom_value_facts_exact_i64(10);
  loom_value_facts_t increment = loom_value_facts_exact_i64(5);
  loom_value_facts_addi(&accumulator, &increment, &accumulator);
  EXPECT_TRUE(loom_value_facts_is_exact(accumulator));
  EXPECT_EQ(accumulator.range_lo, 15);
  EXPECT_EQ(accumulator.known_divisor, 15);
}

//===----------------------------------------------------------------------===//
// Transfer functions: subi
//===----------------------------------------------------------------------===//

TEST(SubiTransfer, ExactMinusExact) {
  loom_value_facts_t a = loom_value_facts_exact_i64(10);
  loom_value_facts_t b = loom_value_facts_exact_i64(3);
  loom_value_facts_t out;
  loom_value_facts_subi(&a, &b, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, 7);
  EXPECT_EQ(out.known_divisor, 7);
}

TEST(SubiTransfer, RangeBounds) {
  // [10,20] - [3,5] → [10-5, 20-3] = [5, 17].
  loom_value_facts_t a = loom_value_facts_make(10, 20, 1);
  loom_value_facts_t b = loom_value_facts_make(3, 5, 1);
  loom_value_facts_t out;
  loom_value_facts_subi(&a, &b, &out);
  EXPECT_EQ(out.range_lo, 5);
  EXPECT_EQ(out.range_hi, 17);
}

//===----------------------------------------------------------------------===//
// Transfer functions: muli
//===----------------------------------------------------------------------===//

TEST(MuliTransfer, ExactTimesExact) {
  loom_value_facts_t a = loom_value_facts_exact_i64(6);
  loom_value_facts_t b = loom_value_facts_exact_i64(7);
  loom_value_facts_t out;
  loom_value_facts_muli(&a, &b, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, 42);
  EXPECT_EQ(out.known_divisor, 42);
}

TEST(MuliTransfer, RangeTimesExact) {
  loom_value_facts_t a = loom_value_facts_make(1, 10, 2);
  loom_value_facts_t b = loom_value_facts_exact_i64(4);
  loom_value_facts_t out;
  loom_value_facts_muli(&a, &b, &out);
  EXPECT_EQ(out.range_lo, 4);
  EXPECT_EQ(out.range_hi, 40);
  EXPECT_EQ(out.known_divisor, 8);  // 2 * 4 = 8.
}

TEST(MuliTransfer, NegativeTimesPositive) {
  loom_value_facts_t a = loom_value_facts_make(-5, -1, 1);
  loom_value_facts_t b = loom_value_facts_make(2, 4, 1);
  loom_value_facts_t out;
  loom_value_facts_muli(&a, &b, &out);
  EXPECT_EQ(out.range_lo, -20);  // -5 * 4.
  EXPECT_EQ(out.range_hi, -2);   // -1 * 2.
}

TEST(MuliTransfer, InPlaceAccumulation) {
  // Element count accumulation pattern: acc *= dim.
  loom_value_facts_t accumulator = loom_value_facts_exact_i64(1);
  loom_value_facts_t dim = loom_value_facts_make(64, 1024, 64);
  loom_value_facts_muli(&accumulator, &dim, &accumulator);
  EXPECT_EQ(accumulator.range_lo, 64);
  EXPECT_EQ(accumulator.range_hi, 1024);
  EXPECT_EQ(accumulator.known_divisor, 64);
}

//===----------------------------------------------------------------------===//
// Transfer functions: divui
//===----------------------------------------------------------------------===//

TEST(DivuiTransfer, ExactDivExact) {
  loom_value_facts_t a = loom_value_facts_exact_i64(42);
  loom_value_facts_t b = loom_value_facts_exact_i64(7);
  loom_value_facts_t out;
  loom_value_facts_divui(&a, &b, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, 6);
}

TEST(DivuiTransfer, DivisorPreserved) {
  // [0, 1000] with divisor 64 / exact 16 → divisor 64/16 = 4.
  loom_value_facts_t a = loom_value_facts_make(0, 1000, 64);
  loom_value_facts_t b = loom_value_facts_exact_i64(16);
  loom_value_facts_t out;
  loom_value_facts_divui(&a, &b, &out);
  EXPECT_EQ(out.known_divisor, 4);
}

TEST(DivuiTransfer, NegativeDividend) {
  // Negative dividend is not supported for unsigned division.
  loom_value_facts_t a = loom_value_facts_make(-10, 10, 1);
  loom_value_facts_t b = loom_value_facts_exact_i64(2);
  loom_value_facts_t out;
  loom_value_facts_divui(&a, &b, &out);
  EXPECT_TRUE(loom_value_facts_is_unknown(out));
}

//===----------------------------------------------------------------------===//
// Transfer functions: remui
//===----------------------------------------------------------------------===//

TEST(RemuiTransfer, ExactDivisible) {
  // 64 % 16 = 0 (exact and divisible).
  loom_value_facts_t a = loom_value_facts_exact_i64(64);
  loom_value_facts_t b = loom_value_facts_exact_i64(16);
  loom_value_facts_t out;
  loom_value_facts_remui(&a, &b, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, 0);
}

TEST(RemuiTransfer, ExactNonDivisible) {
  loom_value_facts_t a = loom_value_facts_exact_i64(18);
  loom_value_facts_t b = loom_value_facts_exact_i64(5);
  loom_value_facts_t out;
  loom_value_facts_remui(&a, &b, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, 3);
}

TEST(RemuiTransfer, DivisibleByDivisor) {
  // Range with divisor 64, mod by 16 → always 0.
  loom_value_facts_t a = loom_value_facts_make(64, 1024, 64);
  loom_value_facts_t b = loom_value_facts_exact_i64(16);
  loom_value_facts_t out;
  loom_value_facts_remui(&a, &b, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, 0);
}

TEST(RemuiTransfer, NotDivisible) {
  // Range not divisible by divisor: result in [0, divisor - 1].
  loom_value_facts_t a = loom_value_facts_make(0, 100, 1);
  loom_value_facts_t b = loom_value_facts_exact_i64(16);
  loom_value_facts_t out;
  loom_value_facts_remui(&a, &b, &out);
  EXPECT_EQ(out.range_lo, 0);
  EXPECT_EQ(out.range_hi, 15);
}

TEST(RemuiTransfer, DynamicDivisorClampedByDividend) {
  loom_value_facts_t a = loom_value_facts_make(0, 63, 1);
  loom_value_facts_t b = loom_value_facts_make(1, 1024, 1);
  loom_value_facts_t out;
  loom_value_facts_remui(&a, &b, &out);
  EXPECT_EQ(out.range_lo, 0);
  EXPECT_EQ(out.range_hi, 63);
}

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

//===----------------------------------------------------------------------===//
// Transfer functions: negi / absi
//===----------------------------------------------------------------------===//

TEST(NegiTransfer, Exact) {
  loom_value_facts_t a = loom_value_facts_exact_i64(5);
  loom_value_facts_t out;
  loom_value_facts_negi(&a, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, -5);
}

TEST(NegiTransfer, RangeSwaps) {
  loom_value_facts_t a = loom_value_facts_make(3, 10, 1);
  loom_value_facts_t out;
  loom_value_facts_negi(&a, &out);
  EXPECT_EQ(out.range_lo, -10);
  EXPECT_EQ(out.range_hi, -3);
}

TEST(NegiTransfer, PreservesDivisor) {
  loom_value_facts_t a = loom_value_facts_make(16, 64, 16);
  loom_value_facts_t out;
  loom_value_facts_negi(&a, &out);
  EXPECT_EQ(out.known_divisor, 16);
}

TEST(AbsiTransfer, NonNegativePassthrough) {
  loom_value_facts_t a = loom_value_facts_make(5, 20, 5);
  loom_value_facts_t out;
  loom_value_facts_absi(&a, &out);
  EXPECT_EQ(out.range_lo, 5);
  EXPECT_EQ(out.range_hi, 20);
  EXPECT_EQ(out.known_divisor, 5);
}

TEST(AbsiTransfer, NegativeFlips) {
  loom_value_facts_t a = loom_value_facts_make(-20, -5, 5);
  loom_value_facts_t out;
  loom_value_facts_absi(&a, &out);
  EXPECT_EQ(out.range_lo, 5);
  EXPECT_EQ(out.range_hi, 20);
}

TEST(AbsiTransfer, SpanningZero) {
  loom_value_facts_t a = loom_value_facts_make(-10, 5, 1);
  loom_value_facts_t out;
  loom_value_facts_absi(&a, &out);
  EXPECT_EQ(out.range_lo, 0);
  EXPECT_EQ(out.range_hi, 10);
}

//===----------------------------------------------------------------------===//
// Transfer functions: minsi / maxsi
//===----------------------------------------------------------------------===//

TEST(MinsiTransfer, ExactValues) {
  loom_value_facts_t a = loom_value_facts_exact_i64(5);
  loom_value_facts_t b = loom_value_facts_exact_i64(3);
  loom_value_facts_t out;
  loom_value_facts_minsi(&a, &b, &out);
  EXPECT_TRUE(loom_value_facts_is_exact(out));
  EXPECT_EQ(out.range_lo, 3);
}

TEST(MaxsiTransfer, Ranges) {
  loom_value_facts_t a = loom_value_facts_make(1, 10, 1);
  loom_value_facts_t b = loom_value_facts_make(5, 20, 1);
  loom_value_facts_t out;
  loom_value_facts_maxsi(&a, &b, &out);
  EXPECT_EQ(out.range_lo, 5);
  EXPECT_EQ(out.range_hi, 20);
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

//===----------------------------------------------------------------------===//
// Shaped type helpers
//===----------------------------------------------------------------------===//

TEST(ElementCountFacts, StaticShapeIsExact) {
  loom_type_t type =
      loom_type_shaped_2d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), loom_dim_pack_static(8), 0);
  loom_value_facts_t count = {};
  loom_value_facts_element_count(type, nullptr, 0, &count);
  EXPECT_TRUE(loom_value_facts_is_exact(count));
  EXPECT_EQ(count.range_lo, 32);
  EXPECT_EQ(count.known_divisor, 32);
}

TEST(ElementCountFacts, RankZeroShapedTypeIsOneElement) {
  loom_type_t type =
      loom_type_shaped_0d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, 0);
  loom_value_facts_t count = {};
  loom_value_facts_element_count(type, nullptr, 0, &count);
  EXPECT_TRUE(loom_value_facts_is_exact(count));
  EXPECT_EQ(count.range_lo, 1);
  EXPECT_EQ(count.known_divisor, 1);
}

TEST(ElementCountFacts, StaticZeroExtentIsExactZero) {
  loom_type_t type = loom_type_shaped_2d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                                         loom_dim_pack_static(0),
                                         loom_dim_pack_static(1024), 0);
  loom_value_facts_t count = {};
  loom_value_facts_element_count(type, nullptr, 0, &count);
  EXPECT_TRUE(loom_value_facts_is_exact(count));
  EXPECT_EQ(count.range_lo, 0);
  EXPECT_TRUE(loom_value_facts_is_non_negative(count));
}

TEST(ElementCountFacts, DynamicShapeUsesDimensionFacts) {
  loom_value_facts_t facts[2] = {loom_value_facts_unknown(),
                                 loom_value_facts_make(8, 64, 8)};
  loom_type_t type =
      loom_type_shaped_2d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(1), loom_dim_pack_static(4), 0);
  loom_value_facts_t count = {};
  loom_value_facts_element_count(type, facts, IREE_ARRAYSIZE(facts), &count);
  EXPECT_EQ(count.range_lo, 32);
  EXPECT_EQ(count.range_hi, 256);
  EXPECT_EQ(count.known_divisor, 32);
}

TEST(ElementCountFacts, OverflowDimensionsUseDimensionFacts) {
  loom_overflow_dim_t dimensions[3] = {
      loom_dim_pack_static(2),
      loom_dim_pack_dynamic(1),
      loom_dim_pack_static(8),
  };
  loom_type_t type = {};
  type.header =
      loom_type_make_header(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I8, 3, 0);
  type.dims[0] = (uint64_t)(uintptr_t)dimensions;
  loom_value_facts_t facts[2] = {loom_value_facts_unknown(),
                                 loom_value_facts_make(4, 12, 4)};

  loom_value_facts_t count = {};
  loom_value_facts_element_count(type, facts, IREE_ARRAYSIZE(facts), &count);
  EXPECT_EQ(count.range_lo, 64);
  EXPECT_EQ(count.range_hi, 192);
  EXPECT_EQ(count.known_divisor, 64);
}

TEST(ElementCountFacts, MalformedInlineRankDegradesSafely) {
  loom_type_t type = {};
  type.header = loom_type_make_header(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, 3,
                                      LOOM_TYPE_FLAG_INLINE_DIMS);
  type.dims[0] = loom_dim_pack_static(2);
  type.dims[1] = loom_dim_pack_static(4);

  loom_value_facts_t count = {};
  loom_value_facts_element_count(type, nullptr, 0, &count);
  EXPECT_EQ(count.range_lo, 0);
  EXPECT_EQ(count.range_hi, INT64_MAX);
  EXPECT_TRUE(loom_value_facts_is_non_negative(count));
}

TEST(ElementCountFacts, DynamicShapeUsesPredicateRefinedFacts) {
  loom_value_facts_t facts[3] = {loom_value_facts_unknown(),
                                 loom_value_facts_unknown(),
                                 loom_value_facts_unknown()};
  loom_predicate_t range_pred = make_predicate_range(16, 128);
  loom_predicate_t multiple_pred = make_predicate_1(LOOM_PREDICATE_MUL, 16);
  loom_value_facts_apply_predicate(&facts[2], &range_pred);
  loom_value_facts_apply_predicate(&facts[2], &multiple_pred);
  loom_type_t type =
      loom_type_shaped_2d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I8,
                          loom_dim_pack_static(3), loom_dim_pack_dynamic(2), 0);

  loom_value_facts_t count = {};
  loom_value_facts_element_count(type, facts, IREE_ARRAYSIZE(facts), &count);
  EXPECT_EQ(count.range_lo, 48);
  EXPECT_EQ(count.range_hi, 384);
  EXPECT_EQ(count.known_divisor, 48);
}

TEST(ElementCountFacts, MissingDynamicFactsDegradeToNonNegativeExtent) {
  loom_type_t type = loom_type_shaped_2d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                                         loom_dim_pack_dynamic(99),
                                         loom_dim_pack_static(4), 0);
  loom_value_facts_t count = {};
  loom_value_facts_element_count(type, nullptr, 0, &count);
  EXPECT_EQ(count.range_lo, 0);
  EXPECT_EQ(count.range_hi, INT64_MAX);
  EXPECT_EQ(count.known_divisor, 4);
  EXPECT_TRUE(loom_value_facts_is_non_negative(count));
}

TEST(ElementCountFacts, NegativeDynamicFactsDoNotEscapeExtentDomain) {
  loom_value_facts_t facts[1] = {loom_value_facts_exact_i64(-4)};
  loom_type_t type = loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                                         loom_dim_pack_dynamic(0), 0);
  loom_value_facts_t count = {};
  loom_value_facts_element_count(type, facts, IREE_ARRAYSIZE(facts), &count);
  EXPECT_EQ(count.range_lo, 0);
  EXPECT_EQ(count.range_hi, INT64_MAX);
  EXPECT_TRUE(loom_value_facts_is_non_negative(count));
}

TEST(ElementCountFacts, NonShapedTypeIsUnknown) {
  loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_facts_t count = {};
  loom_value_facts_element_count(type, nullptr, 0, &count);
  EXPECT_TRUE(loom_value_facts_is_unknown(count));
}

TEST(ElementCountFacts, DivisorHelper) {
  loom_value_facts_t facts[1] = {loom_value_facts_make(64, 1024, 64)};
  loom_type_t type =
      loom_type_shaped_2d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(0), loom_dim_pack_static(2), 0);
  EXPECT_EQ(loom_value_facts_element_count_divisor(type, facts,
                                                   IREE_ARRAYSIZE(facts)),
            128);
}

TEST(ElementCountFacts, EqualStaticCounts) {
  loom_type_t lhs =
      loom_type_shaped_2d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), loom_dim_pack_static(8), 0);
  loom_type_t rhs = loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                                        loom_dim_pack_static(32), 0);
  EXPECT_TRUE(
      loom_value_facts_element_counts_equal(lhs, nullptr, 0, rhs, nullptr, 0));
}

TEST(ElementCountFacts, EqualStructuralDynamicCounts) {
  loom_type_t lhs =
      loom_type_shaped_2d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(7), loom_dim_pack_static(4), 0);
  loom_type_t rhs =
      loom_type_shaped_2d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I8,
                          loom_dim_pack_dynamic(7), loom_dim_pack_static(4), 0);
  EXPECT_TRUE(
      loom_value_facts_element_counts_equal(lhs, nullptr, 0, rhs, nullptr, 0));
}

TEST(ElementCountFacts, EqualDynamicCountsFromExactFacts) {
  loom_value_facts_t lhs_facts[1] = {loom_value_facts_exact_i64(32)};
  loom_value_facts_t rhs_facts[1] = {loom_value_facts_exact_i64(8)};
  loom_type_t lhs = loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                                        loom_dim_pack_dynamic(0), 0);
  loom_type_t rhs =
      loom_type_shaped_2d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I8,
                          loom_dim_pack_dynamic(0), loom_dim_pack_static(4), 0);
  EXPECT_TRUE(loom_value_facts_element_counts_equal(
      lhs, lhs_facts, IREE_ARRAYSIZE(lhs_facts), rhs, rhs_facts,
      IREE_ARRAYSIZE(rhs_facts)));
}

TEST(ElementCountFacts, EqualRangesAreNotProof) {
  loom_value_facts_t facts[2] = {loom_value_facts_make(1, 64, 1),
                                 loom_value_facts_make(1, 64, 1)};
  loom_type_t lhs = loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                                        loom_dim_pack_dynamic(0), 0);
  loom_type_t rhs = loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I8,
                                        loom_dim_pack_dynamic(1), 0);
  EXPECT_FALSE(loom_value_facts_element_counts_equal(
      lhs, facts, IREE_ARRAYSIZE(facts), rhs, facts, IREE_ARRAYSIZE(facts)));
}

}  // namespace
}  // namespace loom
