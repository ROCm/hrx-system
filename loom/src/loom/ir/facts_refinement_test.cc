// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <climits>
#include <cstdint>
#include <limits>

#include "iree/testing/gtest.h"
#include "loom/ir/facts.h"
#include "loom/ir/float_facts.h"

namespace loom {
namespace {

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

static loom_predicate_t make_predicate_power_of_two(void) {
  loom_predicate_t pred = {0};
  pred.kind = (uint8_t)LOOM_PREDICATE_POWER_OF_TWO;
  pred.arg_count = 1;
  pred.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  return pred;
}

static loom_predicate_t make_predicate_not_nan(void) {
  loom_predicate_t pred = {0};
  pred.kind = (uint8_t)LOOM_PREDICATE_NOT_NAN;
  pred.arg_count = 1;
  pred.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  return pred;
}

static loom_predicate_t make_predicate_not_inf(void) {
  loom_predicate_t pred = {0};
  pred.kind = (uint8_t)LOOM_PREDICATE_NOT_INF;
  pred.arg_count = 1;
  pred.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  return pred;
}

static loom_predicate_t make_predicate_finite(void) {
  loom_predicate_t pred = {0};
  pred.kind = (uint8_t)LOOM_PREDICATE_FINITE;
  pred.arg_count = 1;
  pred.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  return pred;
}

TEST(FactsPredicateConflict, RangeDisjointFromKnownRange) {
  loom_value_facts_t f = loom_value_facts_make(0, 63, 1);
  loom_predicate_t pred = make_predicate_range(128, 255);
  loom_value_fact_predicate_conflict_t conflict = {};
  ASSERT_TRUE(loom_value_facts_predicate_conflict(f, &pred, &conflict));
  EXPECT_EQ(conflict.kind, LOOM_VALUE_FACT_PREDICATE_CONFLICT_RANGE);
  EXPECT_EQ(conflict.known_minimum, 0);
  EXPECT_EQ(conflict.known_maximum, 63);
  EXPECT_EQ(conflict.required_minimum, 128);
  EXPECT_EQ(conflict.required_maximum, 255);
}

TEST(FactsPredicateConflict, RangeOverlappingKnownRangeIsSatisfiable) {
  loom_value_facts_t f = loom_value_facts_make(0, 255, 1);
  loom_predicate_t pred = make_predicate_range(128, 511);
  EXPECT_FALSE(loom_value_facts_predicate_conflict(f, &pred, NULL));
}

TEST(FactsPredicateConflict, EmptyPredicateRangeAlwaysConflicts) {
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t pred = make_predicate_1(LOOM_PREDICATE_LT, INT64_MIN);
  loom_value_fact_predicate_conflict_t conflict = {};
  ASSERT_TRUE(loom_value_facts_predicate_conflict(f, &pred, &conflict));
  EXPECT_EQ(conflict.kind, LOOM_VALUE_FACT_PREDICATE_CONFLICT_RANGE);
  EXPECT_GT(conflict.required_minimum, conflict.required_maximum);
}

TEST(FactsPredicateConflict, NotEqualExcludesExactKnownValue) {
  loom_value_facts_t f = loom_value_facts_exact_i64(64);
  loom_predicate_t pred = make_predicate_1(LOOM_PREDICATE_NE, 64);
  loom_value_fact_predicate_conflict_t conflict = {};
  ASSERT_TRUE(loom_value_facts_predicate_conflict(f, &pred, &conflict));
  EXPECT_EQ(conflict.kind, LOOM_VALUE_FACT_PREDICATE_CONFLICT_EXACT_I64);
  EXPECT_EQ(conflict.known_value, 64);
  EXPECT_EQ(conflict.predicate_value, 64);
}

TEST(FactsPredicateConflict, UnsignedRelationUsesCarrierBitOrder) {
  loom_predicate_t pred = make_predicate_1(LOOM_PREDICATE_ULE, INT64_MIN);
  EXPECT_TRUE(loom_value_facts_predicate_conflict(
      loom_value_facts_exact_i64(-1), &pred, NULL));
  EXPECT_FALSE(loom_value_facts_predicate_conflict(
      loom_value_facts_exact_i64(INT64_MIN), &pred, NULL));
}

TEST(FactsPredicateConflict, DivisibilityRejectsExactNonMultiple) {
  loom_value_facts_t f = loom_value_facts_exact_i64(66);
  loom_predicate_t pred = make_predicate_1(LOOM_PREDICATE_MULTIPLE_OF, 64);
  loom_value_fact_predicate_conflict_t conflict = {};
  ASSERT_TRUE(loom_value_facts_predicate_conflict(f, &pred, &conflict));
  EXPECT_EQ(conflict.kind, LOOM_VALUE_FACT_PREDICATE_CONFLICT_EXACT_I64);
  EXPECT_EQ(conflict.known_value, 66);
  EXPECT_EQ(conflict.predicate_value, 64);
}

TEST(FactsPredicateConflict, PowerOfTwoRejectsExactNonPower) {
  loom_value_facts_t f = loom_value_facts_exact_i64(66);
  loom_predicate_t pred = make_predicate_power_of_two();
  loom_value_fact_predicate_conflict_t conflict = {};
  ASSERT_TRUE(loom_value_facts_predicate_conflict(f, &pred, &conflict));
  EXPECT_EQ(conflict.kind, LOOM_VALUE_FACT_PREDICATE_CONFLICT_EXACT_I64);
  EXPECT_EQ(conflict.known_value, 66);
}

TEST(FactsPredicateConflict, NotNanRejectsExactNan) {
  loom_value_facts_t f = loom_value_facts_exact_float(
      LOOM_SCALAR_TYPE_F64, std::numeric_limits<double>::quiet_NaN());
  loom_predicate_t pred = make_predicate_not_nan();
  loom_value_fact_predicate_conflict_t conflict = {};
  ASSERT_TRUE(loom_value_facts_predicate_conflict(f, &pred, &conflict));
  EXPECT_EQ(conflict.kind, LOOM_VALUE_FACT_PREDICATE_CONFLICT_FLOAT_CLASS);
  EXPECT_EQ(conflict.known_float_class,
            LOOM_VALUE_FACT_PREDICATE_CONFLICT_FLOAT_CLASS_NAN);
}

TEST(FactsPredicateConflict, NotInfRejectsExactInfinity) {
  loom_value_facts_t f = loom_value_facts_exact_float(
      LOOM_SCALAR_TYPE_F64, std::numeric_limits<double>::infinity());
  loom_predicate_t pred = make_predicate_not_inf();
  loom_value_fact_predicate_conflict_t conflict = {};
  ASSERT_TRUE(loom_value_facts_predicate_conflict(f, &pred, &conflict));
  EXPECT_EQ(conflict.kind, LOOM_VALUE_FACT_PREDICATE_CONFLICT_FLOAT_CLASS);
  EXPECT_EQ(conflict.known_float_class,
            LOOM_VALUE_FACT_PREDICATE_CONFLICT_FLOAT_CLASS_INFINITY);
}

TEST(FactsPredicateConflict, FiniteRejectsExactNanAndInfinity) {
  loom_predicate_t pred = make_predicate_finite();

  loom_value_fact_predicate_conflict_t nan_conflict = {};
  ASSERT_TRUE(loom_value_facts_predicate_conflict(
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F64,
                                   std::numeric_limits<double>::quiet_NaN()),
      &pred, &nan_conflict));
  EXPECT_EQ(nan_conflict.kind, LOOM_VALUE_FACT_PREDICATE_CONFLICT_FLOAT_CLASS);
  EXPECT_EQ(nan_conflict.known_float_class,
            LOOM_VALUE_FACT_PREDICATE_CONFLICT_FLOAT_CLASS_NAN);

  loom_value_fact_predicate_conflict_t infinity_conflict = {};
  ASSERT_TRUE(loom_value_facts_predicate_conflict(
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F64,
                                   -std::numeric_limits<double>::infinity()),
      &pred, &infinity_conflict));
  EXPECT_EQ(infinity_conflict.kind,
            LOOM_VALUE_FACT_PREDICATE_CONFLICT_FLOAT_CLASS);
  EXPECT_EQ(infinity_conflict.known_float_class,
            LOOM_VALUE_FACT_PREDICATE_CONFLICT_FLOAT_CLASS_INFINITY);
}

TEST(FactsPredicateConflict, FiniteAcceptsExactFiniteFloat) {
  loom_value_facts_t f =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F64, 1.5);
  loom_predicate_t pred = make_predicate_finite();
  EXPECT_FALSE(loom_value_facts_predicate_conflict(f, &pred, NULL));
}

TEST(FactsApplyPredicate, Eq) {
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t pred = make_predicate_1(LOOM_PREDICATE_EQ, 42);
  loom_value_facts_apply_predicate(&f, &pred);
  EXPECT_TRUE(loom_value_facts_is_exact(f));
  EXPECT_EQ(f.range_lo, 42);
  EXPECT_EQ(f.range_hi, 42);
}

TEST(FactsApplyPredicate, FloatEqPreservesAConservativeFloatDomain) {
  loom_value_facts_t facts = loom_value_facts_unknown();
  facts.flags |= LOOM_VALUE_FACT_FLOAT;
  loom_predicate_t predicate = make_predicate_1(LOOM_PREDICATE_EQ, 0);
  loom_value_facts_apply_predicate(&facts, &predicate);

  EXPECT_TRUE(loom_value_facts_is_float(facts));
  EXPECT_TRUE(loom_value_facts_is_finite(facts));
  EXPECT_FALSE(loom_value_facts_is_exact(facts));
  EXPECT_EQ(facts.range_lo, INT64_MIN);
  EXPECT_EQ(facts.range_hi, INT64_MAX);
}

TEST(FactsApplyPredicate, FloatNeZeroRetainsFloatAndNonzeroFacts) {
  loom_value_facts_t facts = loom_value_facts_unknown();
  facts.flags |= LOOM_VALUE_FACT_FLOAT;
  loom_predicate_t predicate = make_predicate_1(LOOM_PREDICATE_NE, 0);
  loom_value_facts_apply_predicate(&facts, &predicate);

  EXPECT_TRUE(loom_value_facts_is_float(facts));
  EXPECT_TRUE(loom_value_facts_is_non_zero(facts));
  EXPECT_FALSE(loom_value_facts_is_exact(facts));
  EXPECT_EQ(facts.range_lo, INT64_MIN);
  EXPECT_EQ(facts.range_hi, INT64_MAX);
}

TEST(FactsApplyPredicate, ValueBoundDoesNotCorruptRange) {
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t pred = make_predicate_1(LOOM_PREDICATE_LT, 42);
  pred.arg_tags[1] = LOOM_PRED_ARG_VALUE;
  pred.args[1] = 7;
  loom_value_facts_apply_predicate(&f, &pred);
  EXPECT_TRUE(loom_value_facts_is_unknown(f));
}

TEST(FactsApplyPredicate, RangeRetainsEachLiteralEndpoint) {
  loom_predicate_t predicate = make_predicate_range(10, 100);

  predicate.arg_tags[2] = LOOM_PRED_ARG_VALUE;
  predicate.args[2] = 7;
  loom_value_facts_t literal_lower = loom_value_facts_unknown();
  loom_value_facts_apply_predicate(&literal_lower, &predicate);
  EXPECT_EQ(literal_lower.range_lo, 10);
  EXPECT_EQ(literal_lower.range_hi, INT64_MAX);

  predicate.arg_tags[1] = LOOM_PRED_ARG_VALUE;
  predicate.args[1] = 8;
  predicate.arg_tags[2] = LOOM_PRED_ARG_CONST;
  predicate.args[2] = 100;
  loom_value_facts_t literal_upper = loom_value_facts_unknown();
  loom_value_facts_apply_predicate(&literal_upper, &predicate);
  EXPECT_EQ(literal_upper.range_lo, INT64_MIN);
  EXPECT_EQ(literal_upper.range_hi, 100);

  predicate.arg_tags[2] = LOOM_PRED_ARG_VALUE;
  predicate.args[2] = 9;
  loom_value_facts_t dynamic_endpoints = loom_value_facts_unknown();
  loom_value_facts_apply_predicate(&dynamic_endpoints, &predicate);
  EXPECT_TRUE(loom_value_facts_is_unknown(dynamic_endpoints));
}

TEST(FactsApplyPredicate, NeIsRepresentedButDoesNotTightenInterval) {
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t pred = make_predicate_1(LOOM_PREDICATE_NE, 42);
  loom_value_facts_apply_predicate(&f, &pred);
  EXPECT_TRUE(loom_value_facts_is_unknown(f));
}

TEST(FactsApplyPredicate, NeTightensExcludedIntervalEdge) {
  loom_predicate_t ne_pred = make_predicate_1(LOOM_PREDICATE_NE, 0);

  loom_value_facts_t lower_edge = loom_value_facts_make(0, 1024, 1);
  loom_value_facts_apply_predicate(&lower_edge, &ne_pred);
  EXPECT_EQ(lower_edge.range_lo, 1);
  EXPECT_EQ(lower_edge.range_hi, 1024);
  EXPECT_TRUE(loom_value_facts_is_positive(lower_edge));

  ne_pred.args[1] = 1024;
  loom_value_facts_t upper_edge = loom_value_facts_make(0, 1024, 1);
  loom_value_facts_apply_predicate(&upper_edge, &ne_pred);
  EXPECT_EQ(upper_edge.range_lo, 0);
  EXPECT_EQ(upper_edge.range_hi, 1023);

  ne_pred.args[1] = 512;
  loom_value_facts_t interior = loom_value_facts_make(0, 1024, 1);
  loom_value_facts_apply_predicate(&interior, &ne_pred);
  EXPECT_EQ(interior.range_lo, 0);
  EXPECT_EQ(interior.range_hi, 1024);
}

TEST(FactsApplyPredicate, NeZeroSetsNonzeroWithoutTighteningInterval) {
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t ne_pred = make_predicate_1(LOOM_PREDICATE_NE, 0);
  loom_value_facts_apply_predicate(&f, &ne_pred);
  EXPECT_EQ(f.range_lo, INT64_MIN);
  EXPECT_EQ(f.range_hi, INT64_MAX);
  EXPECT_TRUE(loom_value_facts_is_non_zero(f));
  EXPECT_FALSE(loom_value_facts_is_positive(f));

  loom_predicate_t upper_bound_predicate =
      make_predicate_1(LOOM_PREDICATE_LE, 1024);
  loom_value_facts_apply_predicate(&f, &upper_bound_predicate);
  EXPECT_EQ(f.range_lo, INT64_MIN);
  EXPECT_EQ(f.range_hi, 1024);
  EXPECT_TRUE(loom_value_facts_is_non_zero(f));
  EXPECT_FALSE(loom_value_facts_is_positive(f));
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

TEST(FactsApplyPredicate, UnsignedRelationRefinesOneSignPartition) {
  loom_value_facts_t negative = loom_value_facts_make(-10, -1, 1);
  loom_predicate_t pred = make_predicate_1(LOOM_PREDICATE_ULT, -5);
  loom_value_facts_apply_predicate(&negative, &pred);
  EXPECT_EQ(negative.range_lo, -10);
  EXPECT_EQ(negative.range_hi, -6);

  loom_value_facts_t split = loom_value_facts_make(-10, 10, 1);
  loom_value_facts_apply_predicate(&split, &pred);
  EXPECT_EQ(split.range_lo, -10);
  EXPECT_EQ(split.range_hi, 10);
}

TEST(FactsApplyPredicate, MultipleOf) {
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t pred = make_predicate_1(LOOM_PREDICATE_MULTIPLE_OF, 16);
  loom_value_facts_apply_predicate(&f, &pred);
  EXPECT_EQ(f.known_divisor, 16);
  EXPECT_TRUE(loom_value_facts_divisible_by(f, 16));
  EXPECT_TRUE(loom_value_facts_divisible_by(f, 8));
  EXPECT_TRUE(loom_value_facts_divisible_by(f, 4));
}

TEST(FactsApplyPredicate, MultipleOfComposition) {
  // multiple_of(16) then multiple_of(24) implies a divisor of 48.
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t pred1 = make_predicate_1(LOOM_PREDICATE_MULTIPLE_OF, 16);
  loom_predicate_t pred2 = make_predicate_1(LOOM_PREDICATE_MULTIPLE_OF, 24);
  loom_value_facts_apply_predicate(&f, &pred1);
  loom_value_facts_apply_predicate(&f, &pred2);
  EXPECT_EQ(f.known_divisor, 48);
}

TEST(FactsApplyPredicate, PowerOfTwo) {
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t pred = make_predicate_power_of_two();
  loom_value_facts_apply_predicate(&f, &pred);
  EXPECT_TRUE(loom_value_facts_is_power_of_two(f));
}

TEST(FactsApplyPredicate, NotNan) {
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t pred = make_predicate_not_nan();
  loom_value_facts_apply_predicate(&f, &pred);
  EXPECT_TRUE(loom_value_facts_is_float(f));
  EXPECT_TRUE(loom_value_facts_is_not_nan(f));
  EXPECT_FALSE(loom_value_facts_is_finite(f));
}

TEST(FactsApplyPredicate, NotInf) {
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t pred = make_predicate_not_inf();
  loom_value_facts_apply_predicate(&f, &pred);
  EXPECT_TRUE(loom_value_facts_is_float(f));
  EXPECT_TRUE(loom_value_facts_is_not_inf(f));
  EXPECT_FALSE(loom_value_facts_is_finite(f));
}

TEST(FactsApplyPredicate, Finite) {
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t pred = make_predicate_finite();
  loom_value_facts_apply_predicate(&f, &pred);
  EXPECT_TRUE(loom_value_facts_is_float(f));
  EXPECT_TRUE(loom_value_facts_is_not_nan(f));
  EXPECT_TRUE(loom_value_facts_is_not_inf(f));
  EXPECT_TRUE(loom_value_facts_is_finite(f));
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
  // Typical tile dimension: multiple_of(64), ge(64).
  loom_value_facts_t f = loom_value_facts_unknown();
  loom_predicate_t multiple_predicate =
      make_predicate_1(LOOM_PREDICATE_MULTIPLE_OF, 64);
  loom_predicate_t lower_bound_predicate =
      make_predicate_1(LOOM_PREDICATE_GE, 64);
  loom_value_facts_apply_predicate(&f, &multiple_predicate);
  loom_value_facts_apply_predicate(&f, &lower_bound_predicate);
  EXPECT_EQ(f.range_lo, 64);
  EXPECT_EQ(f.range_hi, INT64_MAX);
  EXPECT_EQ(f.known_divisor, 64);
  EXPECT_TRUE(loom_value_facts_is_positive(f));
  EXPECT_TRUE(loom_value_facts_divisible_by(f, 16));
}

TEST(FactsApplyPredicate, BoundsRetainDivisibility) {
  struct TestCase {
    // Predicate applied to multiples of four in [-64, 64].
    loom_predicate_kind_t kind;
    // Predicate's literal operand.
    int64_t constant;
    // Inclusive refined lower bound.
    int64_t lower;
    // Inclusive refined upper bound.
    int64_t upper;
  };
  const TestCase cases[] = {
      {LOOM_PREDICATE_LT, 64, -64, 60},   {LOOM_PREDICATE_LE, 63, -64, 60},
      {LOOM_PREDICATE_LT, -60, -64, -64}, {LOOM_PREDICATE_LE, -61, -64, -64},
      {LOOM_PREDICATE_GT, -64, -60, 64},  {LOOM_PREDICATE_GE, -63, -60, 64},
      {LOOM_PREDICATE_GT, 60, 64, 64},    {LOOM_PREDICATE_GE, 61, 64, 64},
      {LOOM_PREDICATE_NE, -64, -60, 64},  {LOOM_PREDICATE_NE, 64, -64, 60},
      {LOOM_PREDICATE_ULT, 64, 0, 60},    {LOOM_PREDICATE_ULE, 63, 0, 60},
      {LOOM_PREDICATE_UGT, -64, -60, -4}, {LOOM_PREDICATE_UGE, -61, -60, -4},
  };
  for (const TestCase& test_case : cases) {
    SCOPED_TRACE(test_case.kind);
    SCOPED_TRACE(test_case.constant);
    loom_value_facts_t facts = loom_value_facts_make(-64, 64, 4);
    loom_value_facts_mark_workgroup_uniform(&facts);
    const loom_predicate_t predicate =
        make_predicate_1(test_case.kind, test_case.constant);
    loom_value_facts_apply_predicate(&facts, &predicate);
    EXPECT_EQ(facts.range_lo, test_case.lower);
    EXPECT_EQ(facts.range_hi, test_case.upper);
    EXPECT_EQ(facts.known_divisor, 4);
    EXPECT_TRUE(loom_value_facts_is_workgroup_uniform(facts));
    EXPECT_EQ(loom_value_facts_is_exact(facts),
              test_case.lower == test_case.upper);
  }
}

TEST(FactsApplyPredicate, DivisorAndRangeOrderAgree) {
  const loom_predicate_t range = make_predicate_range(-11, 23);
  const loom_predicate_t multiple =
      make_predicate_1(LOOM_PREDICATE_MULTIPLE_OF, 6);
  loom_value_facts_t range_first = loom_value_facts_unknown();
  loom_value_facts_apply_predicate(&range_first, &range);
  loom_value_facts_apply_predicate(&range_first, &multiple);
  loom_value_facts_t divisor_first = loom_value_facts_unknown();
  loom_value_facts_apply_predicate(&divisor_first, &multiple);
  loom_value_facts_apply_predicate(&divisor_first, &range);
  EXPECT_TRUE(loom_value_facts_equal(range_first, divisor_first));
  EXPECT_EQ(range_first.range_lo, -6);
  EXPECT_EQ(range_first.range_hi, 18);
  EXPECT_EQ(range_first.known_divisor, 6);
}

TEST(FactsApplyPredicate, NotEqualPreservesEarlierNonzeroConstraint) {
  loom_value_facts_t facts = loom_value_facts_make(-8, 8, 4);
  const loom_predicate_t nonzero = make_predicate_1(LOOM_PREDICATE_NE, 0);
  const loom_predicate_t exclude_upper = make_predicate_1(LOOM_PREDICATE_NE, 8);
  loom_value_facts_apply_predicate(&facts, &nonzero);
  loom_value_facts_apply_predicate(&facts, &exclude_upper);
  EXPECT_EQ(facts.range_lo, -8);
  EXPECT_EQ(facts.range_hi, 4);
  EXPECT_TRUE(loom_value_facts_is_non_zero(facts));
}

TEST(FactsApplyPredicate, DivisibleRangesMatchEnumeratedIntegers) {
  for (int64_t divisor = 1; divisor <= 8; ++divisor) {
    for (int64_t lower = -16; lower <= 16; ++lower) {
      for (int64_t upper = lower; upper <= 16; ++upper) {
        SCOPED_TRACE(divisor);
        SCOPED_TRACE(lower);
        SCOPED_TRACE(upper);
        int64_t expected_lower = INT64_MAX;
        int64_t expected_upper = INT64_MIN;
        for (int64_t value = lower; value <= upper; ++value) {
          if (value % divisor != 0) {
            continue;
          }
          if (expected_lower == INT64_MAX) {
            expected_lower = value;
          }
          expected_upper = value;
        }
        // Empty branch domains have no representable bottom in this lattice.
        if (expected_lower > expected_upper) {
          continue;
        }
        loom_value_facts_t facts = loom_value_facts_make(-32, 32, divisor);
        const loom_predicate_t predicate = make_predicate_range(lower, upper);
        loom_value_facts_apply_predicate(&facts, &predicate);
        EXPECT_EQ(facts.range_lo, expected_lower);
        EXPECT_EQ(facts.range_hi, expected_upper);
        loom_value_facts_t direct =
            loom_value_facts_make(lower, upper, divisor);
        loom_value_facts_refine_divisible_range(&direct);
        EXPECT_EQ(direct.range_lo, expected_lower);
        EXPECT_EQ(direct.range_hi, expected_upper);
        EXPECT_EQ(loom_value_facts_is_exact(direct),
                  expected_lower == expected_upper);
      }
    }
  }
}

TEST(FactsApplyPredicate, DivisibleBoundsPreserveUnboundedAndOverflowDomains) {
  struct TestCase {
    // Inclusive incoming lower bound.
    int64_t lower;
    // Inclusive incoming upper bound.
    int64_t upper;
    // Known divisor applied to the interval.
    int64_t divisor;
    // Inclusive refined lower bound.
    int64_t expected_lower;
    // Inclusive refined upper bound.
    int64_t expected_upper;
  };
  const TestCase cases[] = {
      {INT64_MIN, INT64_MAX, 4, INT64_MIN, INT64_MAX},
      {INT64_MIN, -1, 4, INT64_MIN, -4},
      {1, INT64_MAX, 4, 4, INT64_MAX},
      {INT64_MAX - 1, INT64_MAX, 4, INT64_MAX - 1, INT64_MAX},
      {INT64_MIN, INT64_MIN + 1, 3, INT64_MIN, INT64_MIN + 1},
      {INT64_MIN + 1, INT64_MAX - 1, INT64_MAX, -INT64_MAX, 0},
      {-3, -1, 4, -3, -1},
      {1, 3, 4, 1, 3},
      {-1, 1, 4, 0, 0},
  };
  for (const TestCase& test_case : cases) {
    SCOPED_TRACE(test_case.lower);
    SCOPED_TRACE(test_case.upper);
    SCOPED_TRACE(test_case.divisor);
    loom_value_facts_t facts = loom_value_facts_make(
        test_case.lower, test_case.upper, test_case.divisor);
    loom_value_facts_t direct = facts;
    loom_value_facts_refine_divisible_range(&direct);
    EXPECT_EQ(direct.range_lo, test_case.expected_lower);
    EXPECT_EQ(direct.range_hi, test_case.expected_upper);
    const loom_predicate_t predicate =
        make_predicate_range(test_case.lower, test_case.upper);
    loom_value_facts_apply_predicate(&facts, &predicate);
    EXPECT_EQ(facts.range_lo, test_case.expected_lower);
    EXPECT_EQ(facts.range_hi, test_case.expected_upper);
    EXPECT_EQ(facts.known_divisor, test_case.divisor);
  }
}

TEST(FactsRefineDivisibleRange, PreservesIndependentFacts) {
  loom_value_facts_t facts = loom_value_facts_make(-13, 23, 8);
  facts.flags |= LOOM_VALUE_FACT_NON_ZERO | LOOM_VALUE_FACT_POWER_OF_TWO;
  loom_value_facts_mark_workgroup_uniform(&facts);
  loom_value_facts_refine_divisible_range(&facts);
  EXPECT_EQ(facts.range_lo, -8);
  EXPECT_EQ(facts.range_hi, 16);
  EXPECT_TRUE(loom_value_facts_is_non_zero(facts));
  EXPECT_TRUE(loom_value_facts_is_power_of_two(facts));
  EXPECT_TRUE(loom_value_facts_is_workgroup_uniform(facts));

  const loom_value_facts_t floating =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F64, 3.5);
  facts = floating;
  loom_value_facts_refine_divisible_range(&facts);
  EXPECT_TRUE(loom_value_facts_equal(facts, floating));
}

TEST(FactsRefineRelation, NarrowingRetainsExplicitNonzeroFacts) {
  loom_value_facts_t nonzero = loom_value_facts_make(0, 255, 1);
  nonzero.flags |= LOOM_VALUE_FACT_NON_ZERO;
  const loom_value_facts_t upper = loom_value_facts_make(32, 4096, 1);
  EXPECT_TRUE(loom_value_facts_refine_relation(LOOM_PREDICATE_LE, nonzero,
                                               upper, &nonzero, nullptr));
  EXPECT_TRUE(loom_value_facts_is_non_zero(nonzero));

  const loom_value_facts_t lower = loom_value_facts_make(-128, -32, 1);
  EXPECT_TRUE(loom_value_facts_refine_relation(LOOM_PREDICATE_LE, lower,
                                               nonzero, nullptr, &nonzero));
  EXPECT_TRUE(loom_value_facts_is_non_zero(nonzero));
}

static bool unsigned_relation_holds(loom_predicate_kind_t kind, int64_t lhs,
                                    int64_t rhs) {
  switch (kind) {
    case LOOM_PREDICATE_ULT:
      return static_cast<uint64_t>(lhs) < static_cast<uint64_t>(rhs);
    case LOOM_PREDICATE_ULE:
      return static_cast<uint64_t>(lhs) <= static_cast<uint64_t>(rhs);
    case LOOM_PREDICATE_UGT:
      return static_cast<uint64_t>(lhs) > static_cast<uint64_t>(rhs);
    case LOOM_PREDICATE_UGE:
      return static_cast<uint64_t>(lhs) >= static_cast<uint64_t>(rhs);
    default:
      ADD_FAILURE() << "expected an unsigned comparison";
      return false;
  }
}

static void expect_unsigned_refinement(loom_predicate_kind_t kind,
                                       loom_value_facts_t lhs,
                                       loom_value_facts_t rhs,
                                       loom_value_facts_t expected_lhs,
                                       loom_value_facts_t expected_rhs) {
  loom_value_facts_t refined_lhs = lhs;
  loom_value_facts_t refined_rhs = rhs;
  ASSERT_TRUE(loom_value_facts_refine_relation(kind, lhs, rhs, &refined_lhs,
                                               &refined_rhs));
  EXPECT_EQ(refined_lhs.range_lo, expected_lhs.range_lo);
  EXPECT_EQ(refined_lhs.range_hi, expected_lhs.range_hi);
  EXPECT_EQ(refined_rhs.range_lo, expected_rhs.range_lo);
  EXPECT_EQ(refined_rhs.range_hi, expected_rhs.range_hi);
  if (loom_value_facts_is_exact(rhs)) {
    // Literal predicates reach the same transfer through apply_predicate.
    const auto predicate = make_predicate_1(kind, rhs.range_lo);
    auto literal_lhs = lhs;
    loom_value_facts_apply_predicate(&literal_lhs, &predicate);
    EXPECT_EQ(literal_lhs.range_lo, expected_lhs.range_lo);
    EXPECT_EQ(literal_lhs.range_hi, expected_lhs.range_hi);
  }

  // Snapshot inputs and one-sided alias updates have the same meaning as a
  // two-sided refinement. Neither consumer needs a writable counterpart.
  auto lhs_only = lhs;
  auto rhs_only = rhs;
  ASSERT_TRUE(loom_value_facts_refine_relation(kind, lhs_only, rhs, &lhs_only,
                                               nullptr));
  ASSERT_TRUE(loom_value_facts_refine_relation(kind, lhs, rhs_only, nullptr,
                                               &rhs_only));
  EXPECT_EQ(lhs_only.range_lo, expected_lhs.range_lo);
  EXPECT_EQ(lhs_only.range_hi, expected_lhs.range_hi);
  EXPECT_EQ(rhs_only.range_lo, expected_rhs.range_lo);
  EXPECT_EQ(rhs_only.range_hi, expected_rhs.range_hi);
}

TEST(FactsRefineRelation, UnsignedOrdersMatchEverySmallIntervalPair) {
  for (loom_predicate_kind_t kind : {LOOM_PREDICATE_ULT, LOOM_PREDICATE_ULE,
                                     LOOM_PREDICATE_UGT, LOOM_PREDICATE_UGE}) {
    SCOPED_TRACE(kind);
    for (int64_t lhs_lo = -8; lhs_lo <= 7; ++lhs_lo) {
      for (int64_t lhs_hi = lhs_lo; lhs_hi <= 7; ++lhs_hi) {
        for (int64_t rhs_lo = -8; rhs_lo <= 7; ++rhs_lo) {
          for (int64_t rhs_hi = rhs_lo; rhs_hi <= 7; ++rhs_hi) {
            SCOPED_TRACE(::testing::Message() << lhs_lo << ":" << lhs_hi << " "
                                              << rhs_lo << ":" << rhs_hi);
            int64_t surviving_lhs_lo = INT64_MAX;
            int64_t surviving_lhs_hi = INT64_MIN;
            int64_t surviving_rhs_lo = INT64_MAX;
            int64_t surviving_rhs_hi = INT64_MIN;
            for (int64_t lhs = lhs_lo; lhs <= lhs_hi; ++lhs) {
              for (int64_t rhs = rhs_lo; rhs <= rhs_hi; ++rhs) {
                if (!unsigned_relation_holds(kind, lhs, rhs)) {
                  continue;
                }
                surviving_lhs_lo = iree_min(surviving_lhs_lo, lhs);
                surviving_lhs_hi = iree_max(surviving_lhs_hi, lhs);
                surviving_rhs_lo = iree_min(surviving_rhs_lo, rhs);
                surviving_rhs_hi = iree_max(surviving_rhs_hi, rhs);
              }
            }
            const auto lhs = loom_value_facts_make(lhs_lo, lhs_hi, 1);
            const auto rhs = loom_value_facts_make(rhs_lo, rhs_hi, 1);
            const bool satisfiable = surviving_lhs_lo <= surviving_lhs_hi;
            expect_unsigned_refinement(
                kind, lhs, rhs,
                satisfiable ? loom_value_facts_make(surviving_lhs_lo,
                                                    surviving_lhs_hi, 1)
                            : lhs,
                satisfiable ? loom_value_facts_make(surviving_rhs_lo,
                                                    surviving_rhs_hi, 1)
                            : rhs);
          }
        }
      }
    }
  }
}

TEST(FactsRefineRelation, UnsignedOrdersCoverCarrierExtrema) {
  for (int bit_count : {8, 16, 32, 64}) {
    SCOPED_TRACE(bit_count);
    const int64_t minimum =
        bit_count == 64 ? INT64_MIN : -(INT64_C(1) << (bit_count - 1));
    const int64_t maximum = -(minimum + 1);
    const auto domain = loom_value_facts_make(minimum, maximum, 1);
    const auto zero = loom_value_facts_exact_i64(0);
    const auto all_ones = loom_value_facts_exact_i64(-1);
    expect_unsigned_refinement(LOOM_PREDICATE_ULT, domain, zero, domain, zero);
    expect_unsigned_refinement(LOOM_PREDICATE_ULE, domain, zero, zero, zero);
    expect_unsigned_refinement(LOOM_PREDICATE_UGT, domain, all_ones, domain,
                               all_ones);
    expect_unsigned_refinement(LOOM_PREDICATE_UGE, domain, all_ones, all_ones,
                               all_ones);

    const auto positive_bound = loom_value_facts_make(1, 6, 1);
    expect_unsigned_refinement(LOOM_PREDICATE_ULT, domain, positive_bound,
                               loom_value_facts_make(0, 5, 1), positive_bound);
    const auto negative_bound = loom_value_facts_make(-6, -2, 1);
    expect_unsigned_refinement(LOOM_PREDICATE_UGT, domain, negative_bound,
                               loom_value_facts_make(-5, -1, 1),
                               negative_bound);

    const auto lower_partition = loom_value_facts_make(0, maximum, 1);
    const auto sign_bit = loom_value_facts_exact_i64(minimum);
    expect_unsigned_refinement(LOOM_PREDICATE_ULT, domain, sign_bit,
                               lower_partition, sign_bit);
    const auto upper_partition = loom_value_facts_make(minimum, -1, 1);
    const auto below_sign_bit = loom_value_facts_exact_i64(maximum);
    expect_unsigned_refinement(LOOM_PREDICATE_UGT, domain, below_sign_bit,
                               upper_partition, below_sign_bit);
  }
}

TEST(FactsApplyPredicate, UnsignedBoundsRetainIndependentFactsAndDivisibility) {
  loom_value_facts_t facts = loom_value_facts_make(-64, 64, 4);
  facts.flags |= LOOM_VALUE_FACT_NON_ZERO | LOOM_VALUE_FACT_POWER_OF_TWO;
  loom_value_facts_mark_workgroup_uniform(&facts);
  facts.extension_id = 7;
  const auto predicate = make_predicate_1(LOOM_PREDICATE_ULT, 6);
  loom_value_facts_apply_predicate(&facts, &predicate);
  EXPECT_EQ(facts.range_lo, 0);
  EXPECT_EQ(facts.range_hi, 4);
  EXPECT_EQ(facts.known_divisor, 4);
  EXPECT_EQ(facts.extension_id, 7);
  EXPECT_TRUE(loom_value_facts_is_non_zero(facts));
  EXPECT_TRUE(loom_value_facts_is_power_of_two(facts));
  EXPECT_TRUE(loom_value_facts_is_workgroup_uniform(facts));
}

TEST(ElementCountFacts, DynamicShapeUsesPredicateRefinedFacts) {
  loom_value_facts_t facts[3] = {loom_value_facts_unknown(),
                                 loom_value_facts_unknown(),
                                 loom_value_facts_unknown()};
  loom_predicate_t range_pred = make_predicate_range(16, 128);
  loom_predicate_t multiple_pred =
      make_predicate_1(LOOM_PREDICATE_MULTIPLE_OF, 16);
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

}  // namespace
}  // namespace loom
