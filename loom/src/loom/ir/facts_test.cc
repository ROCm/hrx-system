// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/facts.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "iree/testing/gtest.h"
#include "loom/ir/float_facts.h"

namespace loom {
namespace {

//===----------------------------------------------------------------------===//
// Flags
//===----------------------------------------------------------------------===//

TEST(FactsFlags, EnumeratorsAreUniqueSingleBits) {
  constexpr loom_value_fact_flags_t kFlags[] = {
      LOOM_VALUE_FACT_NON_NEGATIVE,
      LOOM_VALUE_FACT_NON_ZERO,
      LOOM_VALUE_FACT_POSITIVE,
      LOOM_VALUE_FACT_POWER_OF_TWO,
      LOOM_VALUE_FACT_EXACT,
      LOOM_VALUE_FACT_BOOLEAN,
      LOOM_VALUE_FACT_FLOAT,
      LOOM_VALUE_FACT_SUBGROUP_UNIFORM,
      LOOM_VALUE_FACT_LANE_VARYING,
      LOOM_VALUE_FACT_LANE_PREDICATE,
      LOOM_VALUE_FACT_SUBGROUP_LANE_MASK,
      LOOM_VALUE_FACT_NOT_NAN,
      LOOM_VALUE_FACT_NOT_INF,
      LOOM_VALUE_FACT_FINITE,
      LOOM_VALUE_FACT_TOPOLOGY_WORKITEM_X,
      LOOM_VALUE_FACT_TOPOLOGY_WORKITEM_Y,
      LOOM_VALUE_FACT_TOPOLOGY_WORKITEM_Z,
      LOOM_VALUE_FACT_TOPOLOGY_SUBGROUP_LANE,
      LOOM_VALUE_FACT_TOPOLOGY_WORKGROUP_X,
      LOOM_VALUE_FACT_TOPOLOGY_WORKGROUP_Y,
      LOOM_VALUE_FACT_TOPOLOGY_WORKGROUP_Z,
      LOOM_VALUE_FACT_NAN,
      LOOM_VALUE_FACT_INF,
      LOOM_VALUE_FACT_NOT_SUBNORMAL,
      LOOM_VALUE_FACT_WORKGROUP_UNIFORM,
      LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_X,
      LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_Y,
      LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_Z,
      LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_WORKGROUP_X,
      LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_WORKGROUP_Y,
      LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_WORKGROUP_Z,
      LOOM_VALUE_FACT_CLUSTER_UNIFORM,
  };

  loom_value_fact_flags_t accumulated_flags = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kFlags); ++i) {
    SCOPED_TRACE(i);
    const loom_value_fact_flags_t flag = kFlags[i];
    EXPECT_NE(flag, 0u);
    EXPECT_EQ(flag & (flag - 1), 0u);
    EXPECT_EQ(accumulated_flags & flag, 0u);
    accumulated_flags |= flag;
  }
}

TEST(FactsFlags, FloatClassesAndUniformityScopesAreIndependent) {
  struct TestCase {
    loom_value_fact_flags_t flag;
    bool is_nan;
    bool is_inf;
    bool is_not_subnormal;
    bool is_workgroup_uniform;
    bool is_cluster_uniform;
  };
  const TestCase test_cases[] = {
      {LOOM_VALUE_FACT_NAN, true, false, false, false, false},
      {LOOM_VALUE_FACT_INF, false, true, false, false, false},
      {LOOM_VALUE_FACT_NOT_SUBNORMAL, false, false, true, false, false},
      {LOOM_VALUE_FACT_WORKGROUP_UNIFORM, false, false, false, true, false},
      {LOOM_VALUE_FACT_CLUSTER_UNIFORM, false, false, false, true, true},
  };

  for (const TestCase& test_case : test_cases) {
    loom_value_facts_t facts = loom_value_facts_unknown();
    facts.flags = test_case.flag;
    EXPECT_EQ(loom_value_facts_is_nan(facts), test_case.is_nan);
    EXPECT_EQ(loom_value_facts_is_inf(facts), test_case.is_inf);
    EXPECT_EQ(loom_value_facts_is_not_subnormal(facts),
              test_case.is_not_subnormal);
    EXPECT_EQ(loom_value_facts_is_workgroup_uniform(facts),
              test_case.is_workgroup_uniform);
    EXPECT_EQ(loom_value_facts_is_cluster_uniform(facts),
              test_case.is_cluster_uniform);
  }
}

//===----------------------------------------------------------------------===//
// Constructors
//===----------------------------------------------------------------------===//

TEST(FactsNarrowing, IntegerFeedbackHasFiniteDescendingHeight) {
  const loom_scalar_type_t types[] = {
      LOOM_SCALAR_TYPE_I8,  LOOM_SCALAR_TYPE_I16,   LOOM_SCALAR_TYPE_I32,
      LOOM_SCALAR_TYPE_I64, LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_OFFSET};
  for (auto type : types) {
    SCOPED_TRACE(type);
    int64_t lower;
    int64_t upper;
    ASSERT_TRUE(loom_value_facts_scalar_type_domain(type, &lower, &upper));
    auto facts = loom_value_facts_make(lower, upper, 1);
    loom_value_facts_mark_lane_varying(&facts);
    auto incoming = loom_value_facts_make(4, 100, 2);
    facts = loom_value_facts_narrow(type, facts, incoming);
    EXPECT_EQ(facts.range_lo, 4);
    EXPECT_EQ(facts.range_hi, 100);
    EXPECT_EQ(facts.known_divisor, 2);
    incoming = loom_value_facts_make(8, 96, 4);
    loom_value_facts_mark_workgroup_uniform(&incoming);
    facts = loom_value_facts_narrow(type, facts, incoming);
    EXPECT_EQ(facts.range_lo, 4);
    EXPECT_EQ(facts.range_hi, 100);
    EXPECT_EQ(facts.known_divisor, 2);
    EXPECT_TRUE(loom_value_facts_is_workgroup_uniform(facts));
    EXPECT_FALSE(loom_value_facts_is_lane_varying(facts));
    EXPECT_TRUE(loom_value_facts_equal(
        facts, loom_value_facts_narrow(type, facts, incoming)));
  }
}

TEST(FactsNarrowing, UnknownEndpointCanRefineAfterTheOtherEndpoint) {
  const auto left_bounded = loom_value_facts_make(0, INT64_MAX, 1);
  auto facts = loom_value_facts_narrow(
      LOOM_SCALAR_TYPE_I64, loom_value_facts_unknown(), left_bounded);
  facts = loom_value_facts_narrow(LOOM_SCALAR_TYPE_I64, facts,
                                  loom_value_facts_make(8, 64, 8));
  EXPECT_EQ(facts.range_lo, 0);
  EXPECT_EQ(facts.range_hi, 64);
  EXPECT_EQ(facts.known_divisor, 8);
}

TEST(FactsNarrowing, UntypedUnknownDoesNotEraseTheIntegerDomain) {
  const auto domain = loom_value_facts_make(INT32_MIN, INT32_MAX, 1);
  const auto facts = loom_value_facts_narrow(LOOM_SCALAR_TYPE_I32, domain,
                                             loom_value_facts_unknown());
  EXPECT_EQ(facts.range_lo, INT32_MIN);
  EXPECT_EQ(facts.range_hi, INT32_MAX);
}

TEST(FactsNarrowing, ExactValuesRemainDiscoverableAfterBoundedIntervals) {
  const auto integer = loom_value_facts_narrow(LOOM_SCALAR_TYPE_I32,
                                               loom_value_facts_make(-8, 8, 2),
                                               loom_value_facts_exact_i64(4));
  EXPECT_TRUE(loom_value_facts_is_exact(integer));
  EXPECT_EQ(integer.range_lo, 4);
  const auto floating = loom_value_facts_narrow(
      LOOM_SCALAR_TYPE_F32,
      loom_value_facts_make_float_range(LOOM_SCALAR_TYPE_F32, -8.0, 8.0),
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 4.0));
  EXPECT_TRUE(loom_value_facts_is_exact(floating));
}

TEST(FactsNarrowing, FloatClassesCanRefineBeforeAndAfterTheInterval) {
  const loom_scalar_type_t types[] = {
      LOOM_SCALAR_TYPE_F8E4M3, LOOM_SCALAR_TYPE_F8E5M2, LOOM_SCALAR_TYPE_F16,
      LOOM_SCALAR_TYPE_BF16,   LOOM_SCALAR_TYPE_F32,    LOOM_SCALAR_TYPE_F64};
  for (auto type : types) {
    SCOPED_TRACE(type);
    auto classes = loom_value_facts_unknown();
    classes.flags = LOOM_VALUE_FACT_FLOAT | LOOM_VALUE_FACT_NOT_NAN;
    auto facts =
        loom_value_facts_narrow(type, loom_value_facts_unknown(), classes);
    facts = loom_value_facts_narrow(
        type, facts, loom_value_facts_make_float_range(type, -8.0, 8.0));
    auto incoming = loom_value_facts_make_float_range(type, -4.0, 4.0);
    incoming.flags |= LOOM_VALUE_FACT_NOT_SUBNORMAL;
    facts = loom_value_facts_narrow(type, facts, incoming);
    double lower;
    double upper;
    ASSERT_TRUE(loom_value_facts_as_float_range(type, facts, &lower, &upper));
    EXPECT_EQ(lower, -8.0);
    EXPECT_EQ(upper, 8.0);
    EXPECT_TRUE(facts.flags & LOOM_VALUE_FACT_NOT_NAN);
    EXPECT_TRUE(facts.flags & LOOM_VALUE_FACT_NOT_SUBNORMAL);
    EXPECT_TRUE(loom_value_facts_equal(
        facts, loom_value_facts_narrow(type, facts, incoming)));
  }
}

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

TEST(FactsFloatPredicates, FiniteImpliesNotNanAndNotInf) {
  loom_value_facts_t f = loom_value_facts_unknown();
  f.flags = LOOM_VALUE_FACT_FLOAT | LOOM_VALUE_FACT_FINITE;
  EXPECT_TRUE(loom_value_facts_is_finite(f));
  EXPECT_TRUE(loom_value_facts_is_not_nan(f));
  EXPECT_TRUE(loom_value_facts_is_not_inf(f));
}

TEST(FactsTopologyDomain, LookupAndMarkRoundTrip) {
  struct TestCase {
    loom_value_fact_flags_t fact_flag;
    loom_value_fact_topology_value_kind_t value_kind;
    loom_value_fact_topology_axis_t axis;
  };
  const TestCase cases[] = {
      {LOOM_VALUE_FACT_TOPOLOGY_WORKITEM_X,
       LOOM_VALUE_FACT_TOPOLOGY_VALUE_WORKITEM_ID,
       LOOM_VALUE_FACT_TOPOLOGY_AXIS_X},
      {LOOM_VALUE_FACT_TOPOLOGY_WORKITEM_Y,
       LOOM_VALUE_FACT_TOPOLOGY_VALUE_WORKITEM_ID,
       LOOM_VALUE_FACT_TOPOLOGY_AXIS_Y},
      {LOOM_VALUE_FACT_TOPOLOGY_WORKITEM_Z,
       LOOM_VALUE_FACT_TOPOLOGY_VALUE_WORKITEM_ID,
       LOOM_VALUE_FACT_TOPOLOGY_AXIS_Z},
      {LOOM_VALUE_FACT_TOPOLOGY_WORKGROUP_X,
       LOOM_VALUE_FACT_TOPOLOGY_VALUE_WORKGROUP_ID,
       LOOM_VALUE_FACT_TOPOLOGY_AXIS_X},
      {LOOM_VALUE_FACT_TOPOLOGY_WORKGROUP_Y,
       LOOM_VALUE_FACT_TOPOLOGY_VALUE_WORKGROUP_ID,
       LOOM_VALUE_FACT_TOPOLOGY_AXIS_Y},
      {LOOM_VALUE_FACT_TOPOLOGY_WORKGROUP_Z,
       LOOM_VALUE_FACT_TOPOLOGY_VALUE_WORKGROUP_ID,
       LOOM_VALUE_FACT_TOPOLOGY_AXIS_Z},
      {LOOM_VALUE_FACT_TOPOLOGY_SUBGROUP_LANE,
       LOOM_VALUE_FACT_TOPOLOGY_VALUE_SUBGROUP_LANE_ID,
       LOOM_VALUE_FACT_TOPOLOGY_AXIS_LANE},
      {LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_X,
       LOOM_VALUE_FACT_TOPOLOGY_VALUE_CLUSTER_ID,
       LOOM_VALUE_FACT_TOPOLOGY_AXIS_X},
      {LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_Y,
       LOOM_VALUE_FACT_TOPOLOGY_VALUE_CLUSTER_ID,
       LOOM_VALUE_FACT_TOPOLOGY_AXIS_Y},
      {LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_Z,
       LOOM_VALUE_FACT_TOPOLOGY_VALUE_CLUSTER_ID,
       LOOM_VALUE_FACT_TOPOLOGY_AXIS_Z},
      {LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_WORKGROUP_X,
       LOOM_VALUE_FACT_TOPOLOGY_VALUE_CLUSTER_WORKGROUP_ID,
       LOOM_VALUE_FACT_TOPOLOGY_AXIS_X},
      {LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_WORKGROUP_Y,
       LOOM_VALUE_FACT_TOPOLOGY_VALUE_CLUSTER_WORKGROUP_ID,
       LOOM_VALUE_FACT_TOPOLOGY_AXIS_Y},
      {LOOM_VALUE_FACT_TOPOLOGY_CLUSTER_WORKGROUP_Z,
       LOOM_VALUE_FACT_TOPOLOGY_VALUE_CLUSTER_WORKGROUP_ID,
       LOOM_VALUE_FACT_TOPOLOGY_AXIS_Z},
  };
  for (const TestCase& test_case : cases) {
    const loom_value_fact_topology_domain_t* domain =
        loom_value_fact_topology_domain_from_flags(test_case.fact_flag);
    ASSERT_NE(domain, nullptr);
    EXPECT_EQ(domain->value_kind, test_case.value_kind);
    EXPECT_EQ(domain->axis, test_case.axis);

    loom_value_facts_t facts = loom_value_facts_make(0, 63, 1);
    facts.flags |= LOOM_VALUE_FACT_TOPOLOGY_WORKITEM_X;
    ASSERT_TRUE(loom_value_facts_mark_topology_domain(
        &facts, test_case.value_kind, test_case.axis));
    EXPECT_EQ(facts.flags & LOOM_VALUE_FACT_TOPOLOGY_DOMAIN_MASK,
              test_case.fact_flag);
    EXPECT_EQ(loom_value_facts_topology_domain(facts), domain);
  }

  EXPECT_EQ(loom_value_fact_topology_domain_from_flags(0), nullptr);
  EXPECT_EQ(loom_value_fact_topology_domain_from_flags(
                LOOM_VALUE_FACT_TOPOLOGY_WORKITEM_X |
                LOOM_VALUE_FACT_TOPOLOGY_WORKITEM_Y),
            nullptr);
  loom_value_facts_t facts = loom_value_facts_unknown();
  EXPECT_FALSE(loom_value_facts_mark_topology_domain(
      &facts, LOOM_VALUE_FACT_TOPOLOGY_VALUE_NONE,
      LOOM_VALUE_FACT_TOPOLOGY_AXIS_X));
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
  EXPECT_TRUE(loom_value_facts_divisible_by(f, 1));
  EXPECT_TRUE(loom_value_facts_divisible_by(f, 3));
  EXPECT_TRUE(loom_value_facts_divisible_by(f, 16));
  EXPECT_TRUE(loom_value_facts_divisible_by(f, INT64_MAX));
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
  loom_value_facts_t f =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F64, 3.14159265358979);
  EXPECT_TRUE(loom_value_facts_is_exact(f));
  EXPECT_TRUE(loom_value_facts_is_float(f));
  EXPECT_TRUE(loom_value_facts_is_not_nan(f));
  EXPECT_TRUE(loom_value_facts_is_finite(f));
  EXPECT_TRUE(loom_value_facts_is_not_subnormal(f));
  double value = 0.0;
  ASSERT_TRUE(loom_value_facts_as_exact_float(LOOM_SCALAR_TYPE_F64, f, &value));
  EXPECT_DOUBLE_EQ(value, 3.14159265358979);
}

TEST(FactsExactF64, Zero) {
  loom_value_facts_t f =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F64, 0.0);
  EXPECT_TRUE(loom_value_facts_is_exact(f));
  EXPECT_TRUE(loom_value_facts_is_float(f));
  EXPECT_TRUE(loom_value_facts_is_not_nan(f));
  EXPECT_TRUE(loom_value_facts_is_finite(f));
  EXPECT_TRUE(loom_value_facts_is_not_subnormal(f));
  double value = 1.0;
  ASSERT_TRUE(loom_value_facts_as_exact_float(LOOM_SCALAR_TYPE_F64, f, &value));
  EXPECT_DOUBLE_EQ(value, 0.0);
}

TEST(FactsExactF64, Subnormal) {
  loom_value_facts_t f = loom_value_facts_exact_float(
      LOOM_SCALAR_TYPE_F64, std::numeric_limits<double>::denorm_min());
  EXPECT_TRUE(loom_value_facts_is_exact(f));
  EXPECT_TRUE(loom_value_facts_is_float(f));
  EXPECT_TRUE(loom_value_facts_is_not_nan(f));
  EXPECT_TRUE(loom_value_facts_is_finite(f));
  EXPECT_FALSE(loom_value_facts_is_not_subnormal(f));
  double value = 0.0;
  ASSERT_TRUE(loom_value_facts_as_exact_float(LOOM_SCALAR_TYPE_F64, f, &value));
  EXPECT_DOUBLE_EQ(value, std::numeric_limits<double>::denorm_min());
}

TEST(FactsExactF64, NaN) {
  loom_value_facts_t f = loom_value_facts_exact_float(
      LOOM_SCALAR_TYPE_F64, std::numeric_limits<double>::quiet_NaN());
  EXPECT_TRUE(loom_value_facts_is_exact(f));
  EXPECT_TRUE(loom_value_facts_is_float(f));
  EXPECT_TRUE(loom_value_facts_is_nan(f));
  EXPECT_FALSE(loom_value_facts_is_not_nan(f));
  EXPECT_FALSE(loom_value_facts_is_finite(f));
}

TEST(FactsExactF64, Infinity) {
  loom_value_facts_t f = loom_value_facts_exact_float(
      LOOM_SCALAR_TYPE_F64, std::numeric_limits<double>::infinity());
  EXPECT_TRUE(loom_value_facts_is_exact(f));
  EXPECT_TRUE(loom_value_facts_is_float(f));
  EXPECT_TRUE(loom_value_facts_is_inf(f));
  EXPECT_TRUE(loom_value_facts_is_not_nan(f));
  EXPECT_FALSE(loom_value_facts_is_finite(f));
}

TEST(FactsExactF64, NegativeZeroDiffersFromPositive) {
  loom_value_facts_t pos =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F64, 0.0);
  loom_value_facts_t neg =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F64, -0.0);
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
  loom_value_facts_t zero =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F64, 0.0);
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

TEST(FactsClampDomain, PreservesIndependentNonzeroPredicate) {
  for (int64_t bound : {127, 1024}) {
    loom_value_facts_t facts = loom_value_facts_make(-bound, bound, 4);
    facts.flags |= LOOM_VALUE_FACT_NON_ZERO;
    loom_value_facts_mark_lane_varying(&facts);
    facts.extension_id = 7;

    loom_value_facts_t clamped =
        loom_value_facts_clamp_domain(facts, -128, 127);
    EXPECT_EQ(clamped.range_lo, bound == 127 ? -127 : -128);
    EXPECT_EQ(clamped.range_hi, 127);
    EXPECT_EQ(clamped.known_divisor, 4);
    EXPECT_TRUE(loom_value_facts_is_non_zero(clamped));
    EXPECT_TRUE(loom_value_facts_is_lane_varying(clamped));
    EXPECT_EQ(clamped.extension_id, 7u);
  }
}

TEST(FactsClampDomain, RetainsTopologyOnlyForUnchangedDomain) {
  loom_value_facts_t facts = loom_value_facts_make(0, 63, 1);
  facts.flags |= LOOM_VALUE_FACT_TOPOLOGY_SUBGROUP_LANE;
  EXPECT_NE(loom_value_facts_topology_domain(
                loom_value_facts_clamp_domain(facts, -128, 127)),
            nullptr);
  EXPECT_EQ(loom_value_facts_topology_domain(
                loom_value_facts_clamp_domain(facts, 0, 31)),
            nullptr);
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
// Integer wrapping and signed extension
//===----------------------------------------------------------------------===//

TEST(FactsWrapInteger, ExactAndRangedLowBits) {
  for (int64_t divisor : {1, 2, 3, 6, 16, 256, 512}) {
    for (int64_t lo : {-300, -128, -32, 0, 30, 120, 255}) {
      for (int64_t length : {0, 4, 20, 300}) {
        loom_value_facts_t source =
            loom_value_facts_make(lo, lo + length, divisor);
        if (length != 0) {
          loom_value_facts_mark_lane_varying(&source);
        }
        loom_value_facts_t result = loom_value_facts_wrap_integer(source, 8);
        for (int64_t value = lo; value <= lo + length; ++value) {
          if (value % source.known_divisor != 0) {
            continue;
          }
          const int64_t bits = (uint64_t)value & 255;
          const int64_t wrapped = bits >= 128 ? bits - 256 : bits;
          EXPECT_LE(result.range_lo, wrapped);
          EXPECT_GE(result.range_hi, wrapped);
          EXPECT_EQ(wrapped % result.known_divisor, 0);
        }
        EXPECT_EQ(loom_value_facts_is_lane_varying(result), length != 0);
      }
    }
  }
}

TEST(FactsWrapInteger, LogicalAndWideDomains) {
  EXPECT_EQ(
      loom_value_facts_wrap_integer(loom_value_facts_exact_i64(2), 1).range_lo,
      0);
  EXPECT_EQ(
      loom_value_facts_wrap_integer(loom_value_facts_exact_i64(-1), 1).range_lo,
      1);
  loom_value_facts_t source = loom_value_facts_make(INT64_MIN, INT64_MAX, 6);
  loom_value_facts_t result = loom_value_facts_wrap_integer(source, 64);
  EXPECT_EQ(result.range_lo, INT64_MIN);
  EXPECT_EQ(result.range_hi, INT64_MAX);
  EXPECT_EQ(result.known_divisor, 2);
  source = loom_value_facts_make(0, 60, 6);
  EXPECT_TRUE(loom_value_facts_equal(loom_value_facts_wrap_integer(source, 64),
                                     source));
}

TEST(FactsSignExtend, LogicalExactValues) {
  loom_value_facts_t zero =
      loom_value_facts_sign_extend(loom_value_facts_exact_i64(0), 1);
  EXPECT_TRUE(loom_value_facts_is_exact(zero));
  EXPECT_EQ(zero.range_lo, 0);

  loom_value_facts_t one =
      loom_value_facts_sign_extend(loom_value_facts_exact_i64(1), 1);
  EXPECT_TRUE(loom_value_facts_is_exact(one));
  EXPECT_EQ(one.range_lo, -1);
}

TEST(FactsSignExtend, LogicalDynamicDomain) {
  loom_value_facts_t source = loom_value_facts_make(0, 1, 1);
  loom_value_facts_mark_lane_predicate(&source);
  loom_value_facts_t result = loom_value_facts_sign_extend(source, 1);
  EXPECT_EQ(result.range_lo, -1);
  EXPECT_EQ(result.range_hi, 0);
  EXPECT_TRUE(loom_value_facts_is_lane_varying(result));
  EXPECT_FALSE(loom_value_facts_is_lane_predicate(result));
  EXPECT_FALSE(loom_value_facts_is_boolean(result));
}

TEST(FactsSignExtend, WiderSignedDomainIsIdentity) {
  loom_value_facts_t source = loom_value_facts_make(-8, 7, 1);
  loom_value_facts_mark_workgroup_uniform(&source);
  EXPECT_TRUE(
      loom_value_facts_equal(loom_value_facts_sign_extend(source, 8), source));
}

//===----------------------------------------------------------------------===//
// Fixed-width integer wrapping
//===----------------------------------------------------------------------===//

TEST(FactsWrapInteger, ExactBitPatterns) {
  for (int32_t bit_count : {1, 4, 8, 16, 32, 64}) {
    for (int64_t value :
         {INT64_MIN, int64_t{-257}, int64_t{-129}, int64_t{-1}, int64_t{0},
          int64_t{1}, int64_t{128}, int64_t{256}, INT64_MAX}) {
      SCOPED_TRACE(bit_count);
      SCOPED_TRACE(value);
      loom_value_facts_t result = loom_value_facts_wrap_integer(
          loom_value_facts_exact_i64(value), bit_count);
      const uint64_t mask =
          bit_count == 64 ? UINT64_MAX : (UINT64_C(1) << bit_count) - 1;
      const uint64_t bits = (uint64_t)value & mask;
      EXPECT_TRUE(loom_value_facts_is_exact(result));
      EXPECT_EQ((uint64_t)result.range_lo & mask, bits);
      if (bit_count == 1) {
        EXPECT_GE(result.range_lo, 0);
        EXPECT_LE(result.range_hi, 1);
      } else if (bit_count < 64) {
        EXPECT_GE(result.range_lo, -(INT64_C(1) << (bit_count - 1)));
        EXPECT_LT(result.range_hi, INT64_C(1) << (bit_count - 1));
      }
    }
  }
}

TEST(FactsWrapInteger, RangesAndDivisibility) {
  struct TestCase {
    // Mathematical source range and divisor before fixed-width wrapping.
    loom_value_facts_t source;
    // Fixed-width destination bit count.
    int32_t bit_count;
    // Expected signed range minimum, or logical minimum for i1.
    int64_t minimum;
    // Expected signed range maximum, or logical maximum for i1.
    int64_t maximum;
    // Divisor retained after possible modular reduction.
    int64_t divisor;
  };
  const TestCase cases[] = {
      {loom_value_facts_make(0, 120, 3), 8, 0, 120, 3},
      {loom_value_facts_make(120, 143, 1), 8, -128, 127, 1},
      {loom_value_facts_make(130, 140, 1), 8, -126, -116, 1},
      {loom_value_facts_make(-140, -130, 1), 8, 116, 126, 1},
      {loom_value_facts_make(250, 260, 1), 8, -6, 4, 1},
      {loom_value_facts_make(120, 400, 1), 8, -128, 127, 1},
      {loom_value_facts_make(126, 132, 3), 8, -128, 127, 1},
      {loom_value_facts_make(120, 144, 12), 8, -128, 127, 4},
      {loom_value_facts_make(0, 1024, 512), 8, -128, 127, 256},
      {loom_value_facts_make(2, 3, 1), 1, 0, 1, 1},
      {loom_value_facts_make(0, 2, 2), 1, 0, 1, 2},
      {loom_value_facts_make(16, int64_t{INT32_MAX} + 16, 1), 32, INT32_MIN,
       INT32_MAX, 1},
      {loom_value_facts_make(INT64_MIN, INT64_MAX, 3), 64, INT64_MIN, INT64_MAX,
       1},
      {loom_value_facts_make(INT64_MIN, INT64_MAX, 12), 64, INT64_MIN,
       INT64_MAX, 4},
      {loom_value_facts_make(0, INT64_MAX, 3), 64, INT64_MIN, INT64_MAX, 1},
  };
  for (const TestCase& test_case : cases) {
    SCOPED_TRACE(test_case.bit_count);
    SCOPED_TRACE(test_case.source.range_lo);
    SCOPED_TRACE(test_case.source.range_hi);
    loom_value_facts_t result =
        loom_value_facts_wrap_integer(test_case.source, test_case.bit_count);
    EXPECT_EQ(result.range_lo, test_case.minimum);
    EXPECT_EQ(result.range_hi, test_case.maximum);
    EXPECT_EQ(result.known_divisor, test_case.divisor);
  }
}

TEST(FactsWrapInteger, PreservesDistributionButNotNonzero) {
  loom_value_facts_t source = loom_value_facts_make(250, 260, 1);
  loom_value_facts_mark_workgroup_uniform(&source);
  loom_value_facts_t result = loom_value_facts_wrap_integer(source, 8);
  EXPECT_TRUE(loom_value_facts_is_workgroup_uniform(result));
  EXPECT_FALSE(loom_value_facts_is_non_zero(result));
  EXPECT_FALSE(loom_value_facts_is_non_negative(result));
}

TEST(FactsWrapUnsignedInteger, ExactAndCrossingRanges) {
  loom_value_facts_t exact = loom_value_facts_wrap_unsigned_integer(
      loom_value_facts_exact_i64(-1), 32);
  EXPECT_TRUE(loom_value_facts_is_exact(exact));
  EXPECT_EQ(exact.range_lo, (int64_t)UINT32_MAX);

  loom_value_facts_t crossing = loom_value_facts_wrap_unsigned_integer(
      loom_value_facts_make(-4, 0, 4), 32);
  EXPECT_EQ(crossing.range_lo, 0);
  EXPECT_EQ(crossing.range_hi, (int64_t)UINT32_MAX);
  EXPECT_EQ(crossing.known_divisor, 4);

  loom_value_facts_t wrapped = loom_value_facts_wrap_unsigned_integer(
      loom_value_facts_make(INT64_C(4294967296), INT64_C(4294967299), 1), 32);
  EXPECT_EQ(wrapped.range_lo, 0);
  EXPECT_EQ(wrapped.range_hi, 3);

  loom_value_facts_t signed_midpoint = loom_value_facts_wrap_unsigned_integer(
      loom_value_facts_make(INT32_MAX, (int64_t)INT32_MAX + 1, 1), 32);
  EXPECT_EQ(signed_midpoint.range_lo, INT32_MAX);
  EXPECT_EQ(signed_midpoint.range_hi, (int64_t)INT32_MAX + 1);
}

TEST(FactsWrapUnsignedInteger, ExactBitPatterns) {
  for (int32_t bit_count : {1, 4, 8, 16, 32, 63}) {
    for (int64_t value :
         {INT64_MIN, int64_t{-257}, int64_t{-129}, int64_t{-1}, int64_t{0},
          int64_t{1}, int64_t{128}, int64_t{256}, INT64_MAX}) {
      SCOPED_TRACE(bit_count);
      SCOPED_TRACE(value);
      const loom_value_facts_t result = loom_value_facts_wrap_unsigned_integer(
          loom_value_facts_exact_i64(value), bit_count);
      const uint64_t mask = (UINT64_C(1) << bit_count) - 1;
      EXPECT_TRUE(loom_value_facts_is_exact(result));
      EXPECT_EQ((uint64_t)result.range_lo, (uint64_t)value & mask);
      EXPECT_GE(result.range_lo, 0);
      EXPECT_LE((uint64_t)result.range_hi, mask);
    }
  }

  const loom_value_facts_t positive64 = loom_value_facts_wrap_unsigned_integer(
      loom_value_facts_exact_i64(INT64_MAX), 64);
  EXPECT_TRUE(loom_value_facts_is_exact(positive64));
  EXPECT_EQ(positive64.range_lo, INT64_MAX);
}

TEST(FactsWrapUnsignedInteger, RangesContainEveryPossibleValue) {
  for (int64_t divisor : {1, 2, 3, 6, 16, 256, 512}) {
    for (int64_t lo : {-300, -128, -32, 0, 30, 120, 255}) {
      for (int64_t length : {0, 4, 20, 300}) {
        SCOPED_TRACE(divisor);
        SCOPED_TRACE(lo);
        SCOPED_TRACE(length);
        loom_value_facts_t source =
            loom_value_facts_make(lo, lo + length, divisor);
        if (length != 0) {
          loom_value_facts_mark_lane_varying(&source);
        }
        const loom_value_facts_t result =
            loom_value_facts_wrap_unsigned_integer(source, 8);
        for (int64_t value = lo; value <= lo + length; ++value) {
          if (value % source.known_divisor != 0) {
            continue;
          }
          const int64_t wrapped = (uint64_t)value & UINT64_C(255);
          EXPECT_LE(result.range_lo, wrapped);
          EXPECT_GE(result.range_hi, wrapped);
          EXPECT_EQ(wrapped % result.known_divisor, 0);
        }
        EXPECT_EQ(loom_value_facts_is_lane_varying(result), length != 0);
      }
    }
  }
}

TEST(FactsWrapUnsignedInteger, ReducesDivisibilityAcrossWrap) {
  loom_value_facts_t odd = loom_value_facts_wrap_unsigned_integer(
      loom_value_facts_make(-15, -3, 3), 32);
  EXPECT_EQ(odd.range_lo, INT64_C(4294967281));
  EXPECT_EQ(odd.range_hi, INT64_C(4294967293));
  EXPECT_EQ(odd.known_divisor, 1);

  loom_value_facts_t power_of_two = loom_value_facts_wrap_unsigned_integer(
      loom_value_facts_make(-16, -4, 4), 32);
  EXPECT_EQ(power_of_two.range_lo, INT64_C(4294967280));
  EXPECT_EQ(power_of_two.range_hi, INT64_C(4294967292));
  EXPECT_EQ(power_of_two.known_divisor, 4);
}

TEST(FactsWrapUnsignedInteger, RetainsRepresentableWideValues) {
  loom_value_facts_t source =
      loom_value_facts_make(INT64_C(4611686018427387904), INT64_MAX - 7, 8);
  loom_value_facts_mark_workgroup_uniform(&source);

  loom_value_facts_t unsigned63 =
      loom_value_facts_wrap_unsigned_integer(source, 63);
  EXPECT_TRUE(loom_value_facts_unsigned_wrap_is_identity(source, 63));
  EXPECT_TRUE(loom_value_facts_equal(unsigned63, source));

  loom_value_facts_t unsigned64 =
      loom_value_facts_wrap_unsigned_integer(source, 64);
  EXPECT_TRUE(loom_value_facts_unsigned_wrap_is_identity(source, 64));
  EXPECT_TRUE(loom_value_facts_equal(unsigned64, source));

  // A nonexact range ending at INT64_MAX may extend past the fact lattice's
  // saturation sentinel. Preserve soundness instead of treating the endpoint
  // as a proven mathematical bound.
  loom_value_facts_t saturated_source =
      loom_value_facts_make(INT64_C(4611686018427387904), INT64_MAX, 8);
  EXPECT_FALSE(
      loom_value_facts_unsigned_wrap_is_identity(saturated_source, 64));
  loom_value_facts_t saturated =
      loom_value_facts_wrap_unsigned_integer(saturated_source, 64);
  EXPECT_EQ(saturated.range_lo, 0);
  EXPECT_EQ(saturated.range_hi, INT64_MAX);
  EXPECT_EQ(saturated.known_divisor, 8);

  loom_value_facts_t saturated63 =
      loom_value_facts_wrap_unsigned_integer(saturated_source, 63);
  EXPECT_EQ(saturated63.range_lo, 0);
  EXPECT_EQ(saturated63.range_hi, INT64_MAX);
  EXPECT_EQ(saturated63.known_divisor, 8);
}

TEST(FactsWrapUnsignedInteger, WidensUnrepresentableU64Results) {
  loom_value_facts_t exact = loom_value_facts_wrap_unsigned_integer(
      loom_value_facts_exact_i64(-1), 64);
  EXPECT_EQ(exact.range_lo, 0);
  EXPECT_EQ(exact.range_hi, INT64_MAX);
  EXPECT_FALSE(loom_value_facts_is_exact(exact));

  loom_value_facts_t source = loom_value_facts_make(-15, -3, 3);
  loom_value_facts_mark_lane_varying(&source);
  loom_value_facts_t result =
      loom_value_facts_wrap_unsigned_integer(source, 64);
  EXPECT_EQ(result.range_lo, 0);
  EXPECT_EQ(result.range_hi, INT64_MAX);
  EXPECT_EQ(result.known_divisor, 1);
  EXPECT_TRUE(loom_value_facts_is_lane_varying(result));
}

TEST(FactsMaximum, NonNegativeFiniteRange) {
  int64_t maximum = -1;
  EXPECT_TRUE(loom_value_facts_as_non_negative_i64_maximum(
      loom_value_facts_make(0, 8192, 1), &maximum));
  EXPECT_EQ(maximum, 8192);
  EXPECT_TRUE(loom_value_facts_as_non_negative_i64_maximum(
      loom_value_facts_exact_i64(0), &maximum));
  EXPECT_EQ(maximum, 0);
}

TEST(FactsMaximum, RejectsNegativeOrUnboundedRange) {
  int64_t maximum = -1;
  EXPECT_FALSE(loom_value_facts_as_non_negative_i64_maximum(
      loom_value_facts_make(-1, 8192, 1), &maximum));
  EXPECT_FALSE(loom_value_facts_as_non_negative_i64_maximum(
      loom_value_facts_make(0, INT64_MAX, 1), &maximum));
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

TEST(FactsMeet, ExactZeroPreservesCommonDivisibility) {
  loom_value_facts_t zero = loom_value_facts_exact_i64(0);
  loom_value_facts_t slots = loom_value_facts_make(128, 384, 128);
  loom_value_facts_t forward;
  loom_value_facts_meet(&zero, &slots, &forward);
  EXPECT_EQ(forward.range_lo, 0);
  EXPECT_EQ(forward.range_hi, 384);
  EXPECT_EQ(forward.known_divisor, 128);
  loom_value_facts_t reverse;
  loom_value_facts_meet(&slots, &zero, &reverse);
  EXPECT_TRUE(loom_value_facts_equal(forward, reverse));
  loom_value_facts_t both_zero;
  loom_value_facts_meet(&zero, &zero, &both_zero);
  EXPECT_TRUE(loom_value_facts_is_zero(both_zero));
  EXPECT_EQ(both_zero.known_divisor, 1);
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
// Execution distribution
//===----------------------------------------------------------------------===//

TEST(ExecutionDistribution, ExactValuesAreClusterUniform) {
  const loom_value_facts_t facts = loom_value_facts_exact_i64(7);
  EXPECT_TRUE(loom_value_facts_is_subgroup_uniform(facts));
  EXPECT_TRUE(loom_value_facts_is_workgroup_uniform(facts));
  EXPECT_TRUE(loom_value_facts_is_cluster_uniform(facts));
}

TEST(ExecutionDistribution, ClusterScopeImpliesWorkgroupAndSubgroupScopes) {
  loom_value_facts_t facts = loom_value_facts_unknown();
  loom_value_facts_mark_cluster_uniform(&facts);
  EXPECT_TRUE(loom_value_facts_is_cluster_uniform(facts));
  EXPECT_TRUE(loom_value_facts_is_workgroup_uniform(facts));
  EXPECT_TRUE(loom_value_facts_is_subgroup_uniform(facts));
}

TEST(ExecutionDistribution, BinaryUsesClusterScopeWhenBothInputsHaveIt) {
  loom_value_facts_t lhs = loom_value_facts_unknown();
  loom_value_facts_mark_cluster_uniform(&lhs);
  loom_value_facts_t rhs = loom_value_facts_unknown();
  loom_value_facts_mark_cluster_uniform(&rhs);
  loom_value_facts_t output = loom_value_facts_unknown();
  loom_value_facts_propagate_binary_distribution(lhs, rhs, &output);
  EXPECT_TRUE(loom_value_facts_is_cluster_uniform(output));
}

TEST(ExecutionDistribution, UnaryPreservesSubgroupScope) {
  loom_value_facts_t input = loom_value_facts_unknown();
  loom_value_facts_mark_subgroup_uniform(&input);
  loom_value_facts_t output = loom_value_facts_unknown();
  loom_value_facts_propagate_unary_distribution(input, &output);
  EXPECT_TRUE(loom_value_facts_is_subgroup_uniform(output));
  EXPECT_FALSE(loom_value_facts_is_workgroup_uniform(output));
}

TEST(ExecutionDistribution, BinaryUsesWeakestUniformScope) {
  loom_value_facts_t lhs = loom_value_facts_unknown();
  loom_value_facts_mark_workgroup_uniform(&lhs);
  loom_value_facts_t rhs = loom_value_facts_unknown();
  loom_value_facts_mark_subgroup_uniform(&rhs);
  loom_value_facts_t output = loom_value_facts_unknown();
  loom_value_facts_propagate_binary_distribution(lhs, rhs, &output);
  EXPECT_TRUE(loom_value_facts_is_subgroup_uniform(output));
  EXPECT_FALSE(loom_value_facts_is_workgroup_uniform(output));
}

TEST(ExecutionDistribution, UnknownInputPreventsUniformProof) {
  loom_value_facts_t lhs = loom_value_facts_unknown();
  loom_value_facts_mark_workgroup_uniform(&lhs);
  const loom_value_facts_t rhs = loom_value_facts_unknown();
  loom_value_facts_t output = loom_value_facts_unknown();
  loom_value_facts_propagate_binary_distribution(lhs, rhs, &output);
  EXPECT_FALSE(loom_value_facts_is_subgroup_uniform(output));
  EXPECT_FALSE(loom_value_facts_is_lane_varying(output));
}

TEST(ExecutionDistribution, LaneVaryingInputDominatesUniformScope) {
  loom_value_facts_t lhs = loom_value_facts_unknown();
  loom_value_facts_mark_workgroup_uniform(&lhs);
  loom_value_facts_t rhs = loom_value_facts_unknown();
  loom_value_facts_mark_lane_varying(&rhs);
  loom_value_facts_t output = loom_value_facts_unknown();
  loom_value_facts_propagate_binary_distribution(lhs, rhs, &output);
  EXPECT_TRUE(loom_value_facts_is_lane_varying(output));
  EXPECT_FALSE(loom_value_facts_is_subgroup_uniform(output));
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
  EXPECT_TRUE(loom_value_facts_is_workgroup_uniform(out));
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

TEST(AddiTransfer, ZeroPreservesFactsAndSupportsEitherOutputAlias) {
  loom_value_facts_t powers = loom_value_facts_make(16, 64, 16);
  powers.flags |= LOOM_VALUE_FACT_POWER_OF_TWO;
  loom_value_facts_mark_workgroup_uniform(&powers);
  loom_value_facts_t nonzero = loom_value_facts_unknown();
  nonzero.flags |= LOOM_VALUE_FACT_NON_ZERO;
  loom_value_facts_mark_lane_varying(&nonzero);
  const loom_value_facts_t zero = loom_value_facts_exact_i64(0);
  for (loom_value_facts_t input :
       {powers, nonzero, loom_value_facts_make(-64, -16, 16), zero,
        loom_value_facts_exact_i64(INT64_MIN)}) {
    for (bool zero_first : {false, true}) {
      const loom_value_facts_t lhs = zero_first ? zero : input;
      const loom_value_facts_t rhs = zero_first ? input : zero;
      loom_value_facts_t output;
      loom_value_facts_addi(&lhs, &rhs, &output);
      EXPECT_TRUE(loom_value_facts_equal(output, input));
      output = lhs;
      loom_value_facts_addi(&output, &rhs, &output);
      EXPECT_TRUE(loom_value_facts_equal(output, input));
      output = rhs;
      loom_value_facts_addi(&lhs, &output, &output);
      EXPECT_TRUE(loom_value_facts_equal(output, input));
    }
  }
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

TEST(SubiTransfer, ZeroPreservesFactsAndSupportsEitherOutputAlias) {
  loom_value_facts_t input = loom_value_facts_make(16, 64, 16);
  input.flags |= LOOM_VALUE_FACT_POWER_OF_TWO;
  loom_value_facts_mark_workgroup_uniform(&input);
  const loom_value_facts_t zero = loom_value_facts_exact_i64(0);
  loom_value_facts_t output;
  loom_value_facts_subi(&input, &zero, &output);
  EXPECT_TRUE(loom_value_facts_equal(output, input));
  output = input;
  loom_value_facts_subi(&output, &zero, &output);
  EXPECT_TRUE(loom_value_facts_equal(output, input));
  output = zero;
  loom_value_facts_subi(&input, &output, &output);
  EXPECT_TRUE(loom_value_facts_equal(output, input));
}

TEST(SubiTransfer, ZeroMinusRangePreservesDivisibility) {
  const loom_value_facts_t zero = loom_value_facts_exact_i64(0);
  loom_value_facts_t input = loom_value_facts_make(16, 64, 16);
  loom_value_facts_mark_lane_varying(&input);
  loom_value_facts_t output;
  loom_value_facts_subi(&zero, &input, &output);
  EXPECT_EQ(output.range_lo, -64);
  EXPECT_EQ(output.range_hi, -16);
  EXPECT_EQ(output.known_divisor, 16);
  EXPECT_FALSE(loom_value_facts_is_non_negative(output));
  EXPECT_TRUE(loom_value_facts_is_lane_varying(output));
}

TEST(SubiTransfer, ZeroMinusMinimumOverflows) {
  const loom_value_facts_t zero = loom_value_facts_exact_i64(0);
  const loom_value_facts_t minimum = loom_value_facts_exact_i64(INT64_MIN);
  loom_value_facts_t output;
  loom_value_facts_subi(&zero, &minimum, &output);
  EXPECT_EQ(output.range_lo, INT64_MIN);
  EXPECT_EQ(output.range_hi, INT64_MAX);
  EXPECT_EQ(output.known_divisor, 1);
  EXPECT_FALSE(loom_value_facts_is_exact(output));
  EXPECT_TRUE(loom_value_facts_is_cluster_uniform(output));
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
// Transfer functions: fmai
//===----------------------------------------------------------------------===//

TEST(FmaiTransfer, ZeroAddendPreservesProductDivisibility) {
  const loom_value_facts_t multiplier = loom_value_facts_exact_i64(768);
  const loom_value_facts_t zero = loom_value_facts_exact_i64(0);
  for (const auto input :
       {loom_value_facts_make(0, 7, 1), loom_value_facts_make(-7, 0, 1),
        loom_value_facts_unknown()}) {
    loom_value_facts_t output;
    loom_value_facts_fmai(&input, &multiplier, &zero, &output);
    EXPECT_EQ(output.known_divisor, 768);
    if (input.range_lo != INT64_MIN) {
      EXPECT_EQ(output.range_lo, input.range_lo * 768);
      EXPECT_EQ(output.range_hi, input.range_hi * 768);
    } else {
      EXPECT_EQ(output.range_lo, INT64_MIN);
      EXPECT_EQ(output.range_hi, INT64_MAX);
    }
  }
}

TEST(FmaiTransfer, ZeroProductPreservesAddendFacts) {
  loom_value_facts_t addend = loom_value_facts_make(16, 64, 16);
  loom_value_facts_mark_workgroup_uniform(&addend);
  loom_value_facts_t varying = loom_value_facts_unknown();
  loom_value_facts_mark_lane_varying(&varying);
  const loom_value_facts_t zero = loom_value_facts_exact_i64(0);
  for (int zero_operand = 0; zero_operand < 2; ++zero_operand) {
    for (int output_operand = 0; output_operand < 3; ++output_operand) {
      loom_value_facts_t operands[] = {varying, varying, addend};
      operands[zero_operand] = zero;
      loom_value_facts_fmai(&operands[0], &operands[1], &operands[2],
                            &operands[output_operand]);
      EXPECT_TRUE(loom_value_facts_equal(operands[output_operand], addend));
    }
  }
}

TEST(FmaiTransfer, ZeroAddendSupportsEveryOutputAlias) {
  for (int output_operand = 0; output_operand < 3; ++output_operand) {
    loom_value_facts_t operands[] = {loom_value_facts_make(0, 7, 1),
                                     loom_value_facts_exact_i64(768),
                                     loom_value_facts_exact_i64(0)};
    loom_value_facts_fmai(&operands[0], &operands[1], &operands[2],
                          &operands[output_operand]);
    EXPECT_EQ(operands[output_operand].range_lo, 0);
    EXPECT_EQ(operands[output_operand].range_hi, 5376);
    EXPECT_EQ(operands[output_operand].known_divisor, 768);
  }
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

TEST(AbsiTransfer, MinimumSignedValueMayRemainNegative) {
  for (int64_t hi : {INT64_MIN, INT64_C(-1), INT64_C(1)}) {
    loom_value_facts_t input = loom_value_facts_make(INT64_MIN, hi, 1);
    loom_value_facts_t output;
    loom_value_facts_absi(&input, &output);
    EXPECT_EQ(output.range_lo, INT64_MIN);
    EXPECT_FALSE(loom_value_facts_is_non_negative(output));
  }
}

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
  using Transfer =
      void (*)(const loom_value_facts_t*, const loom_value_facts_t*, int32_t,
               loom_value_facts_t*);
  const Transfer transfers[] = {
      loom_value_facts_minsi,
      loom_value_facts_maxsi,
      [](const loom_value_facts_t* lhs, const loom_value_facts_t* rhs, int32_t,
         loom_value_facts_t* out) { loom_value_facts_minui(lhs, rhs, out); },
      [](const loom_value_facts_t* lhs, const loom_value_facts_t* rhs, int32_t,
         loom_value_facts_t* out) { loom_value_facts_maxui(lhs, rhs, out); },
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.bit_count);
    SCOPED_TRACE(test_case.lhs);
    SCOPED_TRACE(test_case.rhs);
    for (size_t operation = 0; operation < IREE_ARRAYSIZE(transfers);
         ++operation) {
      SCOPED_TRACE(operation);
      for (int output_index = 0; output_index < 3; ++output_index) {
        loom_value_facts_t facts[] = {
            loom_value_facts_exact_i64(test_case.lhs),
            loom_value_facts_exact_i64(test_case.rhs),
            loom_value_facts_unknown(),
        };
        transfers[operation](&facts[0], &facts[1], test_case.bit_count,
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

//===----------------------------------------------------------------------===//
// Transfer functions: rotates
//===----------------------------------------------------------------------===//

TEST(IntegerRotateTransfer, SignedAmountsWrapModuloEveryIntegerWidth) {
  struct Case {
    int32_t bit_count;
    int64_t value;
    int64_t left_one;
    int64_t right_one;
  };
  static constexpr Case kCases[] = {
      {1, 1, 1, 1},
      {8, -127, 3, -64},
      {16, -32767, 3, -16384},
      {32, INT32_MIN + INT64_C(1), 3, INT64_C(-1073741824)},
      {64, INT64_MIN + INT64_C(1), 3, INT64_C(-4611686018427387904)},
  };
  using Transfer =
      void (*)(const loom_value_facts_t*, const loom_value_facts_t*, int32_t,
               loom_value_facts_t*);
  auto evaluate = [](Transfer transfer, int32_t bit_count, int64_t value,
                     int64_t amount) {
    loom_value_facts_t lhs = loom_value_facts_exact_i64(value);
    loom_value_facts_t rhs = loom_value_facts_exact_i64(amount);
    loom_value_facts_t result = loom_value_facts_unknown();
    transfer(&lhs, &rhs, bit_count, &result);
    EXPECT_TRUE(loom_value_facts_is_exact(result));
    return result.range_lo;
  };

  for (const auto& test_case : kCases) {
    SCOPED_TRACE(test_case.bit_count);
    EXPECT_EQ(evaluate(loom_value_facts_rotli, test_case.bit_count,
                       test_case.value, 1),
              test_case.left_one);
    EXPECT_EQ(evaluate(loom_value_facts_rotli, test_case.bit_count,
                       test_case.value, test_case.bit_count + 1),
              test_case.left_one);
    EXPECT_EQ(evaluate(loom_value_facts_rotli, test_case.bit_count,
                       test_case.value, -1),
              test_case.right_one);
    EXPECT_EQ(evaluate(loom_value_facts_rotri, test_case.bit_count,
                       test_case.value, 1),
              test_case.right_one);
    EXPECT_EQ(evaluate(loom_value_facts_rotri, test_case.bit_count,
                       test_case.value, test_case.bit_count + 1),
              test_case.right_one);
    EXPECT_EQ(evaluate(loom_value_facts_rotri, test_case.bit_count,
                       test_case.value, -1),
              test_case.left_one);
    EXPECT_EQ(evaluate(loom_value_facts_rotli, test_case.bit_count,
                       test_case.value, test_case.bit_count),
              test_case.value);
  }
}

TEST(IntegerRotateTransfer, PreservesDistributionWithoutExactValues) {
  loom_value_facts_t lhs = loom_value_facts_unknown();
  loom_value_facts_mark_workgroup_uniform(&lhs);
  loom_value_facts_t rhs = loom_value_facts_unknown();
  loom_value_facts_mark_subgroup_uniform(&rhs);
  loom_value_facts_t result = loom_value_facts_unknown();

  loom_value_facts_rotli(&lhs, &rhs, 32, &result);

  EXPECT_TRUE(loom_value_facts_is_subgroup_uniform(result));
  EXPECT_FALSE(loom_value_facts_is_workgroup_uniform(result));
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
  loom_type_t type = {
      .header =
          loom_type_make_header(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I8, 3, 0),
  };
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
  loom_type_t type = {
      .header = loom_type_make_header(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, 3,
                                      LOOM_TYPE_FLAG_INLINE_DIMS),
  };
  type.dims[0] = loom_dim_pack_static(2);
  type.dims[1] = loom_dim_pack_static(4);

  loom_value_facts_t count = {};
  loom_value_facts_element_count(type, nullptr, 0, &count);
  EXPECT_EQ(count.range_lo, 0);
  EXPECT_EQ(count.range_hi, INT64_MAX);
  EXPECT_TRUE(loom_value_facts_is_non_negative(count));
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
