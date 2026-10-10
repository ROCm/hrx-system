// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/float_facts.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "iree/base/internal/fpu_state.h"
#include "iree/testing/gtest.h"

namespace loom {
namespace {

static float AddF32(float lhs, float rhs) { return lhs + rhs; }
static double AddF64(double lhs, double rhs) { return lhs + rhs; }
static float SubF32(float lhs, float rhs) { return lhs - rhs; }
static double SubF64(double lhs, double rhs) { return lhs - rhs; }
static float MulF32(float lhs, float rhs) { return lhs * rhs; }
static double MulF64(double lhs, double rhs) { return lhs * rhs; }

static double ExactFloatValue(loom_scalar_type_t scalar_type,
                              loom_value_facts_t facts) {
  double value = 0.0;
  EXPECT_TRUE(loom_value_facts_as_exact_float(scalar_type, facts, &value));
  return value;
}

static uint64_t DoubleBits(double value) {
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static void ExpectFloatRange(loom_scalar_type_t scalar_type,
                             loom_value_facts_t facts, double expected_lo,
                             double expected_hi) {
  double actual_lo = 0.0;
  double actual_hi = 0.0;
  ASSERT_TRUE(loom_value_facts_as_float_range(scalar_type, facts, &actual_lo,
                                              &actual_hi));
  EXPECT_EQ(DoubleBits(actual_lo), DoubleBits(expected_lo));
  EXPECT_EQ(DoubleBits(actual_hi), DoubleBits(expected_hi));
}

static loom_value_facts_t EvaluateIntegerToFloat(
    loom_scalar_type_t source_type, loom_scalar_type_t result_type,
    loom_float_integer_conversion_kind_t kind,
    loom_value_facts_t source_facts) {
  loom_value_facts_t result = loom_value_facts_unknown();
  loom_value_facts_eval_integer_to_float(source_type, result_type, kind,
                                         &source_facts, &result);
  return result;
}

static loom_value_facts_t EvaluateTurnsFacts(loom_scalar_type_t scalar_type,
                                             loom_float_turns_kind_t kind,
                                             loom_value_facts_t input) {
  loom_value_facts_t result = loom_value_facts_unknown();
  loom_value_facts_eval_float_turns(scalar_type, kind, &input, &result);
  return result;
}

static double EvaluateTurns(loom_scalar_type_t scalar_type,
                            loom_float_turns_kind_t kind, double input) {
  return ExactFloatValue(
      scalar_type,
      EvaluateTurnsFacts(scalar_type, kind,
                         loom_value_facts_exact_float(scalar_type, input)));
}

TEST(FloatFacts, RetainsRoundedFiniteIntervals) {
  const loom_value_facts_t range =
      loom_value_facts_make_float_range(LOOM_SCALAR_TYPE_F32, -128.1, 127.1);
  const double expected_lo = static_cast<double>(static_cast<float>(-128.1));
  const double expected_hi = static_cast<double>(static_cast<float>(127.1));
  ExpectFloatRange(LOOM_SCALAR_TYPE_F32, range, expected_lo, expected_hi);
  EXPECT_TRUE(loom_value_facts_is_finite(range));
  EXPECT_TRUE(loom_value_facts_is_not_nan(range));
  EXPECT_TRUE(loom_value_facts_is_not_inf(range));
  EXPECT_FALSE(loom_value_facts_is_exact(range));
  EXPECT_TRUE(loom_value_facts_float_range_within(LOOM_SCALAR_TYPE_F32, range,
                                                  -448.0, 448.0));
  EXPECT_FALSE(loom_value_facts_float_range_within(LOOM_SCALAR_TYPE_F32, range,
                                                   -128.0, 128.0));

  const loom_value_facts_t collapsed = loom_value_facts_make_float_range(
      LOOM_SCALAR_TYPE_F32, 1.0, 1.0 + 0x1p-30);
  EXPECT_TRUE(loom_value_facts_is_exact(collapsed));
  EXPECT_DOUBLE_EQ(ExactFloatValue(LOOM_SCALAR_TYPE_F32, collapsed), 1.0);

  loom_value_facts_t class_only = loom_value_facts_unknown();
  class_only.flags = LOOM_VALUE_FACT_FLOAT | LOOM_VALUE_FACT_NOT_NAN |
                     LOOM_VALUE_FACT_NOT_INF | LOOM_VALUE_FACT_FINITE;
  double lo = 0.0;
  double hi = 0.0;
  EXPECT_FALSE(loom_value_facts_as_float_range(LOOM_SCALAR_TYPE_F32, class_only,
                                               &lo, &hi));

  const loom_value_facts_t overflow = loom_value_facts_make_float_range(
      LOOM_SCALAR_TYPE_F16, -70000.0, 70000.0);
  EXPECT_TRUE(loom_value_facts_is_float(overflow));
  EXPECT_TRUE(loom_value_facts_is_not_nan(overflow));
  EXPECT_FALSE(loom_value_facts_is_finite(overflow));
  EXPECT_FALSE(loom_value_facts_as_float_range(LOOM_SCALAR_TYPE_F16, overflow,
                                               &lo, &hi));
  EXPECT_TRUE(loom_value_facts_is_unknown(
      loom_value_facts_make_float_range(LOOM_SCALAR_TYPE_F32, 2.0, 1.0)));
}

TEST(FloatFacts, ClassifiesExactPowersOfTwoAtEveryDeclaredWidth) {
  struct Case {
    loom_scalar_type_t type;
    int32_t minimum_subnormal_exponent;
  };
  const Case cases[] = {
      {LOOM_SCALAR_TYPE_F8E4M3, -9}, {LOOM_SCALAR_TYPE_F8E5M2, -16},
      {LOOM_SCALAR_TYPE_F16, -24},   {LOOM_SCALAR_TYPE_BF16, -133},
      {LOOM_SCALAR_TYPE_F32, -149},  {LOOM_SCALAR_TYPE_F64, -1074},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(loom_scalar_type_name(test_case.type));
    int32_t exponent = 0;
    EXPECT_TRUE(loom_value_facts_as_exact_power_of_two_float(
        test_case.type, loom_value_facts_exact_float(test_case.type, -8.0),
        &exponent));
    EXPECT_EQ(exponent, 3);

    loom_value_facts_t minimum_subnormal = loom_value_facts_unknown();
    ASSERT_TRUE(loom_value_facts_from_float_bits(test_case.type, 1,
                                                 &minimum_subnormal));
    EXPECT_TRUE(loom_value_facts_as_exact_power_of_two_float(
        test_case.type, minimum_subnormal, &exponent));
    EXPECT_EQ(exponent, test_case.minimum_subnormal_exponent);

    EXPECT_FALSE(loom_value_facts_as_exact_power_of_two_float(
        test_case.type, loom_value_facts_exact_float(test_case.type, 1.5),
        &exponent));
    EXPECT_FALSE(loom_value_facts_as_exact_power_of_two_float(
        test_case.type, loom_value_facts_exact_float(test_case.type, -0.0),
        &exponent));
  }

  int32_t exponent = 0;
  EXPECT_FALSE(loom_value_facts_as_exact_power_of_two_float(
      LOOM_SCALAR_TYPE_F32,
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, INFINITY), &exponent));
  EXPECT_FALSE(loom_value_facts_as_exact_power_of_two_float(
      LOOM_SCALAR_TYPE_F32, loom_value_facts_known_nan(), &exponent));
  EXPECT_FALSE(loom_value_facts_as_exact_power_of_two_float(
      LOOM_SCALAR_TYPE_F32, loom_value_facts_unknown(), &exponent));
}

TEST(FloatFacts, ClassifiesSubnormalsAtEveryDeclaredWidth) {
  struct Case {
    loom_scalar_type_t type;
    int32_t minimum_normal_exponent;
    int32_t minimum_subnormal_exponent;
  };
  const Case cases[] = {
      {LOOM_SCALAR_TYPE_F8E4M3, -6, -9},  {LOOM_SCALAR_TYPE_F8E5M2, -14, -16},
      {LOOM_SCALAR_TYPE_F16, -14, -24},   {LOOM_SCALAR_TYPE_BF16, -126, -133},
      {LOOM_SCALAR_TYPE_F32, -126, -149}, {LOOM_SCALAR_TYPE_F64, -1022, -1074},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(loom_scalar_type_name(test_case.type));
    EXPECT_FALSE(loom_value_facts_is_not_subnormal(loom_value_facts_exact_float(
        test_case.type,
        std::ldexp(1.0, test_case.minimum_subnormal_exponent))));
    EXPECT_TRUE(loom_value_facts_is_not_subnormal(
        loom_value_facts_exact_float(test_case.type, 0.0)));
    EXPECT_TRUE(loom_value_facts_is_not_subnormal(loom_value_facts_exact_float(
        test_case.type, std::ldexp(1.0, test_case.minimum_normal_exponent))));
    EXPECT_TRUE(loom_value_facts_is_not_subnormal(
        loom_value_facts_exact_float(test_case.type, INFINITY)));
    EXPECT_TRUE(loom_value_facts_is_not_subnormal(
        loom_value_facts_exact_float(test_case.type, NAN)));
  }

  EXPECT_TRUE(loom_value_facts_is_not_subnormal(
      loom_value_facts_make_float_range(LOOM_SCALAR_TYPE_F32, 1.0, 2.0)));
  EXPECT_TRUE(loom_value_facts_is_not_subnormal(
      loom_value_facts_make_float_range(LOOM_SCALAR_TYPE_F32, -0.0, 0.0)));
  EXPECT_FALSE(loom_value_facts_is_not_subnormal(
      loom_value_facts_make_float_range(LOOM_SCALAR_TYPE_F32, 0.0, 2.0)));
}

TEST(FloatFacts, AppliesIndependentConversionSubnormalPermissions) {
  const loom_float_conversion_policy_t strict = {0};
  const loom_float_conversion_policy_t flush_input = {
      .may_flush_input_subnormal = true,
      .may_flush_result_subnormal = false,
  };
  const loom_float_conversion_policy_t flush_result = {
      .may_flush_input_subnormal = false,
      .may_flush_result_subnormal = true,
  };

  const loom_value_facts_t f16_subnormal =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F16, 0x1p-24);
  loom_value_facts_t result = loom_value_facts_unknown();
  loom_value_facts_eval_float_conversion(LOOM_SCALAR_TYPE_F16,
                                         LOOM_SCALAR_TYPE_F32, strict,
                                         &f16_subnormal, &result);
  EXPECT_TRUE(loom_value_facts_is_exact(result));
  EXPECT_DOUBLE_EQ(ExactFloatValue(LOOM_SCALAR_TYPE_F32, result), 0x1p-24);

  loom_value_facts_eval_float_conversion(LOOM_SCALAR_TYPE_F16,
                                         LOOM_SCALAR_TYPE_F32, flush_input,
                                         &f16_subnormal, &result);
  EXPECT_FALSE(loom_value_facts_is_exact(result));
  ExpectFloatRange(LOOM_SCALAR_TYPE_F32, result, 0.0, 0x1p-24);

  const loom_value_facts_t f64_normal_f32_subnormal =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F64, 0x1p-149);
  loom_value_facts_eval_float_conversion(LOOM_SCALAR_TYPE_F64,
                                         LOOM_SCALAR_TYPE_F32, flush_input,
                                         &f64_normal_f32_subnormal, &result);
  EXPECT_TRUE(loom_value_facts_is_exact(result));
  EXPECT_DOUBLE_EQ(ExactFloatValue(LOOM_SCALAR_TYPE_F32, result), 0x1p-149);

  loom_value_facts_eval_float_conversion(LOOM_SCALAR_TYPE_F64,
                                         LOOM_SCALAR_TYPE_F32, flush_result,
                                         &f64_normal_f32_subnormal, &result);
  EXPECT_FALSE(loom_value_facts_is_exact(result));
  ExpectFloatRange(LOOM_SCALAR_TYPE_F32, result, 0.0, 0x1p-149);

  const loom_value_facts_t normal =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 1.0);
  const loom_float_conversion_policy_t flush_both = {
      .may_flush_input_subnormal = true,
      .may_flush_result_subnormal = true,
  };
  loom_value_facts_eval_float_conversion(LOOM_SCALAR_TYPE_F32,
                                         LOOM_SCALAR_TYPE_BF16, flush_both,
                                         &normal, &result);
  EXPECT_TRUE(loom_value_facts_is_exact(result));
  EXPECT_DOUBLE_EQ(ExactFloatValue(LOOM_SCALAR_TYPE_BF16, result), 1.0);

  const loom_value_facts_t positive_subnormal_range =
      loom_value_facts_make_float_range(LOOM_SCALAR_TYPE_F16, 0x1p-24, 0x1p-23);
  loom_value_facts_eval_float_conversion(LOOM_SCALAR_TYPE_F16,
                                         LOOM_SCALAR_TYPE_F32, flush_input,
                                         &positive_subnormal_range, &result);
  ExpectFloatRange(LOOM_SCALAR_TYPE_F32, result, 0.0, 0x1p-23);

  const loom_value_facts_t negative_subnormal_range =
      loom_value_facts_make_float_range(LOOM_SCALAR_TYPE_F16, -0x1p-23,
                                        -0x1p-24);
  loom_value_facts_eval_float_conversion(LOOM_SCALAR_TYPE_F16,
                                         LOOM_SCALAR_TYPE_F32, flush_input,
                                         &negative_subnormal_range, &result);
  ExpectFloatRange(LOOM_SCALAR_TYPE_F32, result, -0x1p-23, -0.0);
}

TEST(FloatFacts, JoinsFiniteIntervalsAndSupportsAliasedOutput) {
  loom_value_facts_t lhs =
      loom_value_facts_make_float_range(LOOM_SCALAR_TYPE_F32, -2.0, 1.0);
  const loom_value_facts_t rhs =
      loom_value_facts_make_float_range(LOOM_SCALAR_TYPE_F32, -1.0, 3.0);
  loom_value_facts_meet(&lhs, &rhs, &lhs);
  ExpectFloatRange(LOOM_SCALAR_TYPE_F32, lhs, -2.0, 3.0);
  EXPECT_TRUE(loom_value_facts_is_finite(lhs));
  EXPECT_FALSE(loom_value_facts_is_exact(lhs));

  loom_value_facts_t class_only = loom_value_facts_unknown();
  class_only.flags = LOOM_VALUE_FACT_FLOAT | LOOM_VALUE_FACT_NOT_NAN |
                     LOOM_VALUE_FACT_NOT_INF | LOOM_VALUE_FACT_FINITE;
  loom_value_facts_t joined = loom_value_facts_unknown();
  loom_value_facts_meet(&lhs, &class_only, &joined);
  EXPECT_TRUE(loom_value_facts_is_finite(joined));
  double lo = 0.0;
  double hi = 0.0;
  EXPECT_FALSE(
      loom_value_facts_as_float_range(LOOM_SCALAR_TYPE_F32, joined, &lo, &hi));

  const loom_value_facts_t negative_zero =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, -0.0);
  const loom_value_facts_t positive_zero =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 0.0);
  loom_value_facts_meet(&negative_zero, &positive_zero, &joined);
  EXPECT_TRUE(loom_value_facts_is_finite(joined));
  EXPECT_FALSE(loom_value_facts_is_exact(joined));
}

TEST(FloatFacts, IntegerToFloatRetainsDeclaredTypeDomains) {
  struct Case {
    loom_scalar_type_t source_type;
    loom_float_integer_conversion_kind_t kind;
    double expected_lo;
    double expected_hi;
  };
  const Case cases[] = {
      {LOOM_SCALAR_TYPE_I1, LOOM_FLOAT_INTEGER_CONVERSION_SIGNED, -1.0, 0.0},
      {LOOM_SCALAR_TYPE_I8, LOOM_FLOAT_INTEGER_CONVERSION_SIGNED, -128.0,
       127.0},
      {LOOM_SCALAR_TYPE_I16, LOOM_FLOAT_INTEGER_CONVERSION_SIGNED, -32768.0,
       32767.0},
      {LOOM_SCALAR_TYPE_I32, LOOM_FLOAT_INTEGER_CONVERSION_SIGNED,
       -2147483648.0, 2147483648.0},
      {LOOM_SCALAR_TYPE_I1, LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED, 0.0, 1.0},
      {LOOM_SCALAR_TYPE_I8, LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED, 0.0, 255.0},
      {LOOM_SCALAR_TYPE_I16, LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED, 0.0,
       65535.0},
      {LOOM_SCALAR_TYPE_I32, LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED, 0.0,
       4294967296.0},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(loom_scalar_type_name(test_case.source_type));
    const loom_value_facts_t result =
        EvaluateIntegerToFloat(test_case.source_type, LOOM_SCALAR_TYPE_F32,
                               test_case.kind, loom_value_facts_unknown());
    ExpectFloatRange(LOOM_SCALAR_TYPE_F32, result, test_case.expected_lo,
                     test_case.expected_hi);
    EXPECT_TRUE(loom_value_facts_is_finite(result));
    EXPECT_TRUE(loom_value_facts_is_not_subnormal(result));
  }

  const loom_value_facts_t signed_i64 = EvaluateIntegerToFloat(
      LOOM_SCALAR_TYPE_I64, LOOM_SCALAR_TYPE_F64,
      LOOM_FLOAT_INTEGER_CONVERSION_SIGNED, loom_value_facts_unknown());
  ExpectFloatRange(LOOM_SCALAR_TYPE_F64, signed_i64,
                   static_cast<double>(INT64_MIN),
                   static_cast<double>(INT64_MAX));
  const loom_value_facts_t unsigned_i64 = EvaluateIntegerToFloat(
      LOOM_SCALAR_TYPE_I64, LOOM_SCALAR_TYPE_F64,
      LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED, loom_value_facts_unknown());
  ExpectFloatRange(LOOM_SCALAR_TYPE_F64, unsigned_i64, 0.0,
                   static_cast<double>(UINT64_MAX));
}

TEST(FloatFacts, IntegerToFloatAvoidsWideIntegerDoubleRounding) {
  constexpr int64_t kAboveF32Midpoint =
      (INT64_C(1) << 62) + (INT64_C(1) << 38) + 1;
  const loom_value_facts_t result =
      EvaluateIntegerToFloat(LOOM_SCALAR_TYPE_I64, LOOM_SCALAR_TYPE_F32,
                             LOOM_FLOAT_INTEGER_CONVERSION_SIGNED,
                             loom_value_facts_exact_i64(kAboveF32Midpoint));
  ASSERT_TRUE(loom_value_facts_is_exact(result));
  EXPECT_DOUBLE_EQ(ExactFloatValue(LOOM_SCALAR_TYPE_F32, result),
                   static_cast<double>(static_cast<float>(kAboveF32Midpoint)));
  EXPECT_NE(ExactFloatValue(LOOM_SCALAR_TYPE_F32, result),
            static_cast<double>(
                static_cast<float>(static_cast<double>(kAboveF32Midpoint))));
}

TEST(FloatFacts, IntegerToFloatRetainsUnsignedSubdomains) {
  const loom_value_facts_t negative = loom_value_facts_make(-4, -1, 1);
  ExpectFloatRange(
      LOOM_SCALAR_TYPE_F32,
      EvaluateIntegerToFloat(LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_F32,
                             LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED, negative),
      252.0, 255.0);

  const loom_value_facts_t crossing = loom_value_facts_make(-4, 3, 1);
  ExpectFloatRange(
      LOOM_SCALAR_TYPE_F32,
      EvaluateIntegerToFloat(LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_F32,
                             LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED, crossing),
      0.0, 255.0);

  const loom_value_facts_t positive = loom_value_facts_make(3, 9, 1);
  ExpectFloatRange(
      LOOM_SCALAR_TYPE_F32,
      EvaluateIntegerToFloat(LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_F32,
                             LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED, positive),
      3.0, 9.0);

  const loom_value_facts_t exact = EvaluateIntegerToFloat(
      LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_F32,
      LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED, loom_value_facts_exact_i64(-1));
  EXPECT_TRUE(loom_value_facts_is_exact(exact));
  EXPECT_DOUBLE_EQ(ExactFloatValue(LOOM_SCALAR_TYPE_F32, exact), 255.0);

  const loom_value_facts_t signed_boolean = EvaluateIntegerToFloat(
      LOOM_SCALAR_TYPE_I1, LOOM_SCALAR_TYPE_F32,
      LOOM_FLOAT_INTEGER_CONVERSION_SIGNED, loom_value_facts_exact_i64(1));
  EXPECT_DOUBLE_EQ(ExactFloatValue(LOOM_SCALAR_TYPE_F32, signed_boolean), -1.0);
}

TEST(FloatFacts, IntegerToFloatDropsOverflowingIntervals) {
  loom_value_facts_t source = loom_value_facts_unknown();
  loom_value_facts_mark_lane_varying(&source);
  const loom_value_facts_t result =
      EvaluateIntegerToFloat(LOOM_SCALAR_TYPE_I64, LOOM_SCALAR_TYPE_F16,
                             LOOM_FLOAT_INTEGER_CONVERSION_SIGNED, source);
  EXPECT_TRUE(loom_value_facts_is_float(result));
  EXPECT_TRUE(loom_value_facts_is_not_nan(result));
  EXPECT_TRUE(loom_value_facts_is_not_subnormal(result));
  EXPECT_FALSE(loom_value_facts_is_finite(result));
  EXPECT_TRUE(loom_value_facts_is_lane_varying(result));
  double lo = 0.0;
  double hi = 0.0;
  EXPECT_FALSE(
      loom_value_facts_as_float_range(LOOM_SCALAR_TYPE_F16, result, &lo, &hi));

  const loom_value_facts_t saturated = EvaluateIntegerToFloat(
      LOOM_SCALAR_TYPE_I16, LOOM_SCALAR_TYPE_F8E4M3,
      LOOM_FLOAT_INTEGER_CONVERSION_SIGNED, loom_value_facts_unknown());
  ExpectFloatRange(LOOM_SCALAR_TYPE_F8E4M3, saturated, -448.0, 448.0);
}

static void ExpectTurnsPeriodic(loom_scalar_type_t scalar_type, double lhs,
                                double rhs) {
  EXPECT_EQ(DoubleBits(EvaluateTurns(scalar_type, LOOM_FLOAT_TURNS_SIN, lhs)),
            DoubleBits(EvaluateTurns(scalar_type, LOOM_FLOAT_TURNS_SIN, rhs)));
  EXPECT_EQ(DoubleBits(EvaluateTurns(scalar_type, LOOM_FLOAT_TURNS_COS, lhs)),
            DoubleBits(EvaluateTurns(scalar_type, LOOM_FLOAT_TURNS_COS, rhs)));
}

static uint64_t BitMask(int32_t bit_count) {
  return bit_count == 64 ? UINT64_MAX
                         : (UINT64_C(1) << bit_count) - UINT64_C(1);
}

static double NextSourceValue(loom_scalar_type_t source_type, double value,
                              double direction) {
  if (source_type == LOOM_SCALAR_TYPE_F32) {
    return static_cast<double>(std::nextafter(static_cast<float>(value),
                                              static_cast<float>(direction)));
  }
  return std::nextafter(value, direction);
}

static void ExpectFloatToInteger(loom_scalar_type_t source_type,
                                 loom_scalar_type_t result_type,
                                 loom_float_integer_conversion_kind_t kind,
                                 double source_value, uint64_t expected_bits) {
  const loom_value_facts_t source =
      loom_value_facts_exact_float(source_type, source_value);
  loom_value_facts_t result = loom_value_facts_unknown();
  loom_value_facts_eval_float_to_integer(source_type, result_type, kind,
                                         &source, &result);

  const int32_t result_bit_count = loom_scalar_type_bitwidth(result_type);
  uint64_t actual_bits = 0;
  ASSERT_TRUE(loom_value_facts_as_exact_raw_bits(result, result_bit_count,
                                                 &actual_bits));
  EXPECT_EQ(expected_bits & BitMask(result_bit_count), actual_bits);
}

static void ExpectInvalidFloatToInteger(
    loom_scalar_type_t source_type, loom_scalar_type_t result_type,
    loom_float_integer_conversion_kind_t kind, double source_value) {
  const loom_value_facts_t source =
      loom_value_facts_exact_float(source_type, source_value);
  loom_value_facts_t result = loom_value_facts_exact_i64(0);
  loom_value_facts_eval_float_to_integer(source_type, result_type, kind,
                                         &source, &result);
  EXPECT_TRUE(loom_value_facts_is_unknown(result));
}

TEST(FloatFacts, RoundsEveryF32Instruction) {
  loom_value_facts_t large =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 16777216.0);
  loom_value_facts_t one =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 1.0);
  loom_value_facts_t sum = loom_value_facts_unknown();
  loom_value_facts_eval_float_binary(LOOM_SCALAR_TYPE_F32, &large, &one, AddF32,
                                     AddF64, &sum);
  loom_value_facts_t result = loom_value_facts_unknown();
  loom_value_facts_eval_float_binary(LOOM_SCALAR_TYPE_F32, &sum, &large, SubF32,
                                     SubF64, &result);

  EXPECT_FLOAT_EQ(
      static_cast<float>(ExactFloatValue(LOOM_SCALAR_TYPE_F32, result)), 0.0f);
}

TEST(FloatFacts, PreservesF64InstructionPrecision) {
  loom_value_facts_t large =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F64, 16777216.0);
  loom_value_facts_t one =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F64, 1.0);
  loom_value_facts_t sum = loom_value_facts_unknown();
  loom_value_facts_eval_float_binary(LOOM_SCALAR_TYPE_F64, &large, &one, AddF32,
                                     AddF64, &sum);
  loom_value_facts_t result = loom_value_facts_unknown();
  loom_value_facts_eval_float_binary(LOOM_SCALAR_TYPE_F64, &sum, &large, SubF32,
                                     SubF64, &result);

  EXPECT_DOUBLE_EQ(ExactFloatValue(LOOM_SCALAR_TYPE_F64, result), 1.0);
}

TEST(FloatFacts, LogisticPreservesSelectedWidthNegativeTails) {
  const float f32_result = loom_float_logistic_f32(-100.0f);
  EXPECT_GT(f32_result, 0.0f);
  EXPECT_EQ(std::fpclassify(f32_result), FP_SUBNORMAL);

  const double f64_result = loom_float_logistic_f64(-710.0);
  EXPECT_GT(f64_result, 0.0);
  EXPECT_EQ(std::fpclassify(f64_result), FP_SUBNORMAL);
}

TEST(FloatFacts, LogisticHandlesFiniteAndExceptionalValues) {
  const float f32_negative_infinity =
      loom_float_logistic_f32(-std::numeric_limits<float>::infinity());
  EXPECT_EQ(f32_negative_infinity, 0.0f);
  EXPECT_FALSE(std::signbit(f32_negative_infinity));
  EXPECT_EQ(loom_float_logistic_f32(-0.0f), 0.5f);
  EXPECT_EQ(loom_float_logistic_f32(0.0f), 0.5f);
  EXPECT_EQ(loom_float_logistic_f32(std::numeric_limits<float>::infinity()),
            1.0f);
  EXPECT_TRUE(std::isnan(
      loom_float_logistic_f32(std::numeric_limits<float>::quiet_NaN())));
  const float f32_negative = loom_float_logistic_f32(-1.0f);
  EXPECT_GT(f32_negative, 0.0f);
  EXPECT_LT(f32_negative, 1.0f);
  const float f32_positive = loom_float_logistic_f32(1.0f);
  EXPECT_GT(f32_positive, 0.0f);
  EXPECT_LT(f32_positive, 1.0f);

  const double f64_negative_infinity =
      loom_float_logistic_f64(-std::numeric_limits<double>::infinity());
  EXPECT_EQ(f64_negative_infinity, 0.0);
  EXPECT_FALSE(std::signbit(f64_negative_infinity));
  EXPECT_EQ(loom_float_logistic_f64(-0.0), 0.5);
  EXPECT_EQ(loom_float_logistic_f64(0.0), 0.5);
  EXPECT_EQ(loom_float_logistic_f64(std::numeric_limits<double>::infinity()),
            1.0);
  EXPECT_TRUE(std::isnan(
      loom_float_logistic_f64(std::numeric_limits<double>::quiet_NaN())));
  const double f64_negative = loom_float_logistic_f64(-1.0);
  EXPECT_GT(f64_negative, 0.0);
  EXPECT_LT(f64_negative, 1.0);
  const double f64_positive = loom_float_logistic_f64(1.0);
  EXPECT_GT(f64_positive, 0.0);
  EXPECT_LT(f64_positive, 1.0);
}

TEST(FloatFacts, TurnsPreserveCardinalsAtEveryDeclaredWidth) {
  constexpr loom_scalar_type_t kScalarTypes[] = {
      LOOM_SCALAR_TYPE_F8E4M3, LOOM_SCALAR_TYPE_F8E5M2, LOOM_SCALAR_TYPE_F16,
      LOOM_SCALAR_TYPE_BF16,   LOOM_SCALAR_TYPE_F32,    LOOM_SCALAR_TYPE_F64,
  };
  for (const loom_scalar_type_t scalar_type : kScalarTypes) {
    SCOPED_TRACE(loom_scalar_type_name(scalar_type));

    const double sin_positive_zero =
        EvaluateTurns(scalar_type, LOOM_FLOAT_TURNS_SIN, 0.0);
    EXPECT_EQ(sin_positive_zero, 0.0);
    EXPECT_FALSE(std::signbit(sin_positive_zero));
    const double sin_negative_zero =
        EvaluateTurns(scalar_type, LOOM_FLOAT_TURNS_SIN, -0.0);
    EXPECT_EQ(sin_negative_zero, 0.0);
    EXPECT_TRUE(std::signbit(sin_negative_zero));
    EXPECT_EQ(EvaluateTurns(scalar_type, LOOM_FLOAT_TURNS_COS, 0.0), 1.0);
    EXPECT_EQ(EvaluateTurns(scalar_type, LOOM_FLOAT_TURNS_COS, -0.0), 1.0);

    EXPECT_EQ(EvaluateTurns(scalar_type, LOOM_FLOAT_TURNS_SIN, 0.25), 1.0);
    EXPECT_EQ(EvaluateTurns(scalar_type, LOOM_FLOAT_TURNS_SIN, -0.25), -1.0);
    const double cos_positive_quarter =
        EvaluateTurns(scalar_type, LOOM_FLOAT_TURNS_COS, 0.25);
    EXPECT_EQ(cos_positive_quarter, 0.0);
    EXPECT_FALSE(std::signbit(cos_positive_quarter));
    const double cos_negative_quarter =
        EvaluateTurns(scalar_type, LOOM_FLOAT_TURNS_COS, -0.25);
    EXPECT_EQ(cos_negative_quarter, 0.0);
    EXPECT_FALSE(std::signbit(cos_negative_quarter));

    const double sin_positive_half =
        EvaluateTurns(scalar_type, LOOM_FLOAT_TURNS_SIN, 0.5);
    EXPECT_EQ(sin_positive_half, 0.0);
    EXPECT_FALSE(std::signbit(sin_positive_half));
    const double sin_negative_half =
        EvaluateTurns(scalar_type, LOOM_FLOAT_TURNS_SIN, -0.5);
    EXPECT_EQ(sin_negative_half, 0.0);
    EXPECT_TRUE(std::signbit(sin_negative_half));
    EXPECT_EQ(EvaluateTurns(scalar_type, LOOM_FLOAT_TURNS_COS, 0.5), -1.0);
    EXPECT_EQ(EvaluateTurns(scalar_type, LOOM_FLOAT_TURNS_COS, -0.5), -1.0);
  }
}

TEST(FloatFacts, TurnsPreserveLargeInputPeriodicity) {
  ExpectTurnsPeriodic(LOOM_SCALAR_TYPE_F32, 0.125, 1048576.125);
  ExpectTurnsPeriodic(LOOM_SCALAR_TYPE_F32, -0.125, -1048576.125);
  ExpectTurnsPeriodic(LOOM_SCALAR_TYPE_F64, 0.125, 562949953421312.125);
  ExpectTurnsPeriodic(LOOM_SCALAR_TYPE_F64, -0.125, -562949953421312.125);

  EXPECT_EQ(
      EvaluateTurns(LOOM_SCALAR_TYPE_F32, LOOM_FLOAT_TURNS_SIN, 2097151.75),
      -1.0);
  EXPECT_EQ(
      EvaluateTurns(LOOM_SCALAR_TYPE_F32, LOOM_FLOAT_TURNS_COS, 2097151.75),
      0.0);
  EXPECT_EQ(EvaluateTurns(LOOM_SCALAR_TYPE_F64, LOOM_FLOAT_TURNS_SIN,
                          1125899906842623.75),
            -1.0);
  EXPECT_EQ(EvaluateTurns(LOOM_SCALAR_TYPE_F64, LOOM_FLOAT_TURNS_COS,
                          1125899906842623.75),
            0.0);

  constexpr loom_scalar_type_t kScalarTypes[] = {
      LOOM_SCALAR_TYPE_F32,
      LOOM_SCALAR_TYPE_F64,
  };
  constexpr double kLargeIntegers[] = {
      16777216.0,
      9007199254740992.0,
  };
  const double kMaxFinite[] = {
      std::numeric_limits<float>::max(),
      std::numeric_limits<double>::max(),
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kScalarTypes); ++i) {
    const loom_scalar_type_t scalar_type = kScalarTypes[i];
    SCOPED_TRACE(loom_scalar_type_name(scalar_type));
    const double magnitudes[] = {kLargeIntegers[i], kMaxFinite[i]};
    for (const double magnitude : magnitudes) {
      const double positive_sin =
          EvaluateTurns(scalar_type, LOOM_FLOAT_TURNS_SIN, magnitude);
      EXPECT_EQ(positive_sin, 0.0);
      EXPECT_FALSE(std::signbit(positive_sin));
      const double negative_sin =
          EvaluateTurns(scalar_type, LOOM_FLOAT_TURNS_SIN, -magnitude);
      EXPECT_EQ(negative_sin, 0.0);
      EXPECT_TRUE(std::signbit(negative_sin));
      EXPECT_EQ(EvaluateTurns(scalar_type, LOOM_FLOAT_TURNS_COS, magnitude),
                1.0);
      EXPECT_EQ(EvaluateTurns(scalar_type, LOOM_FLOAT_TURNS_COS, -magnitude),
                1.0);
    }
  }
}

TEST(FloatFacts, TurnsProduceArithmeticNanForNonFiniteInputs) {
  constexpr loom_scalar_type_t kScalarTypes[] = {
      LOOM_SCALAR_TYPE_F32,
      LOOM_SCALAR_TYPE_F64,
  };
  constexpr uint64_t kSignalingNanBits[] = {
      UINT64_C(0x7F800001),
      UINT64_C(0x7FF0000000000001),
  };
  constexpr loom_scalar_type_t kIntegerTypes[] = {
      LOOM_SCALAR_TYPE_I32,
      LOOM_SCALAR_TYPE_I64,
  };
  constexpr loom_float_turns_kind_t kKinds[] = {
      LOOM_FLOAT_TURNS_SIN,
      LOOM_FLOAT_TURNS_COS,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kScalarTypes); ++i) {
    const loom_scalar_type_t scalar_type = kScalarTypes[i];
    loom_value_facts_t signaling_nan_bits =
        loom_value_facts_exact_i64(static_cast<int64_t>(kSignalingNanBits[i]));
    loom_value_facts_t signaling_nan = loom_value_facts_unknown();
    loom_value_facts_eval_scalar_bitcast(kIntegerTypes[i], scalar_type,
                                         &signaling_nan_bits, &signaling_nan);
    ASSERT_TRUE(loom_value_facts_is_nan(signaling_nan));
    ASSERT_FALSE(loom_value_facts_is_exact(signaling_nan));

    const loom_value_facts_t inputs[] = {
        loom_value_facts_exact_float(scalar_type,
                                     std::numeric_limits<double>::quiet_NaN()),
        signaling_nan,
        loom_value_facts_exact_float(scalar_type, INFINITY),
        loom_value_facts_exact_float(scalar_type, -INFINITY),
    };
    for (const loom_float_turns_kind_t kind : kKinds) {
      for (const loom_value_facts_t input : inputs) {
        const loom_value_facts_t result =
            EvaluateTurnsFacts(scalar_type, kind, input);
        EXPECT_TRUE(loom_value_facts_is_nan(result));
        EXPECT_TRUE(loom_value_facts_is_not_inf(result));
      }
    }
  }
}

TEST(FloatFacts, FloatToIntegerUsesEveryDeclaredWidth) {
  constexpr loom_scalar_type_t kSourceTypes[] = {
      LOOM_SCALAR_TYPE_F32,
      LOOM_SCALAR_TYPE_F64,
  };
  constexpr loom_scalar_type_t kResultTypes[] = {
      LOOM_SCALAR_TYPE_I8,
      LOOM_SCALAR_TYPE_I16,
      LOOM_SCALAR_TYPE_I32,
      LOOM_SCALAR_TYPE_I64,
  };
  for (const loom_scalar_type_t source_type : kSourceTypes) {
    for (const loom_scalar_type_t result_type : kResultTypes) {
      SCOPED_TRACE(loom_scalar_type_name(source_type));
      SCOPED_TRACE(loom_scalar_type_name(result_type));
      ExpectFloatToInteger(source_type, result_type,
                           LOOM_FLOAT_INTEGER_CONVERSION_SIGNED, -42.75,
                           static_cast<uint64_t>(-INT64_C(42)));
      ExpectFloatToInteger(source_type, result_type,
                           LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED, 200.75,
                           UINT64_C(200));
    }
  }
}

TEST(FloatFacts, FloatToIntegerChecksTheTruncatedValue) {
  ExpectFloatToInteger(LOOM_SCALAR_TYPE_F32, LOOM_SCALAR_TYPE_I8,
                       LOOM_FLOAT_INTEGER_CONVERSION_SIGNED, -128.75,
                       UINT64_C(0x80));
  ExpectFloatToInteger(LOOM_SCALAR_TYPE_F64, LOOM_SCALAR_TYPE_I16,
                       LOOM_FLOAT_INTEGER_CONVERSION_SIGNED, -32768.75,
                       UINT64_C(0x8000));
  ExpectFloatToInteger(LOOM_SCALAR_TYPE_F64, LOOM_SCALAR_TYPE_I32,
                       LOOM_FLOAT_INTEGER_CONVERSION_SIGNED, -2147483648.75,
                       UINT64_C(0x80000000));
  ExpectFloatToInteger(LOOM_SCALAR_TYPE_F32, LOOM_SCALAR_TYPE_I8,
                       LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED, -0.75,
                       UINT64_C(0));
  ExpectFloatToInteger(LOOM_SCALAR_TYPE_F64, LOOM_SCALAR_TYPE_I64,
                       LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED, -0.75,
                       UINT64_C(0));
}

TEST(FloatFacts, FloatToUnsignedIntegerCanonicalizesBooleanBits) {
  ExpectFloatToInteger(LOOM_SCALAR_TYPE_F32, LOOM_SCALAR_TYPE_I1,
                       LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED, 1.75,
                       UINT64_C(1));
}

TEST(FloatFacts, FloatToIntegerCoversDestinationBoundaries) {
  constexpr loom_scalar_type_t kSourceTypes[] = {
      LOOM_SCALAR_TYPE_F32,
      LOOM_SCALAR_TYPE_F64,
  };
  constexpr loom_scalar_type_t kResultTypes[] = {
      LOOM_SCALAR_TYPE_I8,
      LOOM_SCALAR_TYPE_I16,
      LOOM_SCALAR_TYPE_I32,
      LOOM_SCALAR_TYPE_I64,
  };
  for (const loom_scalar_type_t source_type : kSourceTypes) {
    for (const loom_scalar_type_t result_type : kResultTypes) {
      SCOPED_TRACE(loom_scalar_type_name(source_type));
      SCOPED_TRACE(loom_scalar_type_name(result_type));
      const int32_t result_bit_count = loom_scalar_type_bitwidth(result_type);
      const double signed_extent = std::ldexp(1.0, result_bit_count - 1);
      const double unsigned_extent = std::ldexp(1.0, result_bit_count);
      const double highest_signed_source =
          NextSourceValue(source_type, signed_extent, 0.0);
      const double highest_unsigned_source =
          NextSourceValue(source_type, unsigned_extent, 0.0);

      ExpectFloatToInteger(source_type, result_type,
                           LOOM_FLOAT_INTEGER_CONVERSION_SIGNED, -signed_extent,
                           UINT64_C(1) << (result_bit_count - 1));
      ExpectFloatToInteger(
          source_type, result_type, LOOM_FLOAT_INTEGER_CONVERSION_SIGNED,
          highest_signed_source, static_cast<uint64_t>(highest_signed_source));
      ExpectInvalidFloatToInteger(
          source_type, result_type, LOOM_FLOAT_INTEGER_CONVERSION_SIGNED,
          NextSourceValue(source_type, -signed_extent - 1.0, -INFINITY));
      ExpectInvalidFloatToInteger(source_type, result_type,
                                  LOOM_FLOAT_INTEGER_CONVERSION_SIGNED,
                                  signed_extent);

      ExpectFloatToInteger(source_type, result_type,
                           LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED,
                           highest_unsigned_source,
                           static_cast<uint64_t>(highest_unsigned_source));
      ExpectInvalidFloatToInteger(source_type, result_type,
                                  LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED, -1.0);
      ExpectInvalidFloatToInteger(source_type, result_type,
                                  LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED,
                                  unsigned_extent);
    }
  }
}

TEST(FloatFacts, FloatToUnsignedIntegerPreservesTheUpperI64Bit) {
  constexpr loom_scalar_type_t kSourceTypes[] = {
      LOOM_SCALAR_TYPE_F32,
      LOOM_SCALAR_TYPE_F64,
  };
  for (const loom_scalar_type_t source_type : kSourceTypes) {
    SCOPED_TRACE(loom_scalar_type_name(source_type));
    ExpectFloatToInteger(source_type, LOOM_SCALAR_TYPE_I64,
                         LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED, 0x1p63,
                         UINT64_C(0x8000000000000000));
  }
}

TEST(FloatFacts, FloatToIntegerRejectsNonFiniteValues) {
  constexpr double kInvalidValues[] = {
      NAN,
      INFINITY,
      -INFINITY,
  };
  for (const double value : kInvalidValues) {
    ExpectInvalidFloatToInteger(LOOM_SCALAR_TYPE_F32, LOOM_SCALAR_TYPE_I32,
                                LOOM_FLOAT_INTEGER_CONVERSION_SIGNED, value);
    ExpectInvalidFloatToInteger(LOOM_SCALAR_TYPE_F64, LOOM_SCALAR_TYPE_I64,
                                LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED, value);
  }
}
TEST(FloatFacts, F8E4M3SaturatesOverflow) {
  auto expect_saturated = [](double source_value, double expected_value,
                             uint64_t expected_bits) {
    loom_value_facts_t value =
        loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F8E4M3, source_value);
    EXPECT_DOUBLE_EQ(ExactFloatValue(LOOM_SCALAR_TYPE_F8E4M3, value),
                     expected_value);

    loom_value_facts_t bits = loom_value_facts_unknown();
    loom_value_facts_eval_scalar_bitcast(LOOM_SCALAR_TYPE_F8E4M3,
                                         LOOM_SCALAR_TYPE_I8, &value, &bits);
    uint64_t actual_bits = 0;
    EXPECT_TRUE(loom_value_facts_as_exact_raw_bits(bits, 8, &actual_bits));
    EXPECT_EQ(expected_bits, actual_bits);
  };

  expect_saturated(449.0, 448.0, UINT64_C(0x7E));
  expect_saturated(-449.0, -448.0, UINT64_C(0xFE));
  expect_saturated(INFINITY, 448.0, UINT64_C(0x7E));
  expect_saturated(-INFINITY, -448.0, UINT64_C(0xFE));
}

static void ExpectFloatFma(loom_scalar_type_t scalar_type, double a, double b,
                           double c, double expected) {
  const loom_value_facts_t a_facts =
      loom_value_facts_exact_float(scalar_type, a);
  const loom_value_facts_t b_facts =
      loom_value_facts_exact_float(scalar_type, b);
  const loom_value_facts_t c_facts =
      loom_value_facts_exact_float(scalar_type, c);
  loom_value_facts_t result = loom_value_facts_unknown();
  loom_value_facts_eval_float_fma(scalar_type, &a_facts, &b_facts, &c_facts,
                                  &result);
  const double actual = ExactFloatValue(scalar_type, result);
  if (std::isnan(expected)) {
    EXPECT_TRUE(std::isnan(actual));
  } else {
    EXPECT_EQ(DoubleBits(actual), DoubleBits(expected));
  }
}

TEST(FloatFacts, NarrowFmaRoundsOnceAtMidpoints) {
  // 129/128 * 3/2 is halfway between two BF16 values. A tiny signed addend
  // selects the side even when it disappears in both F32 and F64 arithmetic.
  for (const double sign : {1.0, -1.0}) {
    ExpectFloatFma(LOOM_SCALAR_TYPE_BF16, sign * 0x1.02p0, 1.5, -sign * 0x1p-30,
                   sign * 0x1.82p0);
    ExpectFloatFma(LOOM_SCALAR_TYPE_BF16, sign * 0x1.02p0, 1.5,
                   -sign * 0x1p-133, sign * 0x1.82p0);
    ExpectFloatFma(LOOM_SCALAR_TYPE_BF16, sign * 0x1.06p0, 1.5, sign * 0x1p-133,
                   sign * 0x1.8ap0);
    ExpectFloatFma(LOOM_SCALAR_TYPE_F16, sign * 0x1.004p0, 1.5, -sign * 0x1p-24,
                   sign * 0x1.804p0);
    ExpectFloatFma(LOOM_SCALAR_TYPE_F8E5M2, -sign * 0x1.4p7, 0x1.8p6,
                   sign * 0x1.4p-14, -sign * 0x1.cp13);
  }
  // Exact midpoints retain ties-to-even rather than always rounding away.
  ExpectFloatFma(LOOM_SCALAR_TYPE_BF16, 0x1.02p0, 1.5, 0.0, 0x1.84p0);
  ExpectFloatFma(LOOM_SCALAR_TYPE_BF16, 0x1.06p0, 1.5, 0.0, 0x1.88p0);
  ExpectFloatFma(LOOM_SCALAR_TYPE_F8E4M3, 1.125, 1.5, 0.0, 1.75);
}

TEST(FloatFacts, NarrowFmaPreservesSubnormalAndOverflowBoundaries) {
  // Products may exceed the narrow range before cancellation by the addend.
  ExpectFloatFma(LOOM_SCALAR_TYPE_BF16, 0x1.fep127, 2.0, -0x1.fep127,
                 0x1.fep127);
  ExpectFloatFma(LOOM_SCALAR_TYPE_F16, 65504.0, 2.0, -65504.0, 65504.0);
  ExpectFloatFma(LOOM_SCALAR_TYPE_F8E5M2, 57344.0, 2.0, -57344.0, 57344.0);
  ExpectFloatFma(LOOM_SCALAR_TYPE_F8E4M3, 448.0, 2.0, -448.0, 448.0);
  ExpectFloatFma(LOOM_SCALAR_TYPE_F8E4M3, 448.0, 2.0, 0.0, 448.0);
  ExpectFloatFma(LOOM_SCALAR_TYPE_F16, 65504.0, 2.0, 0.0, INFINITY);

  // Subnormal result rounding, including a tie to zero and a tie to two units.
  ExpectFloatFma(LOOM_SCALAR_TYPE_BF16, 0x1p-133, 0.5, 0.0, 0.0);
  ExpectFloatFma(LOOM_SCALAR_TYPE_BF16, -0x1p-133, 0.5, -0.0, -0.0);
  ExpectFloatFma(LOOM_SCALAR_TYPE_BF16, 0x1p-133, 0.5, 0x1p-133, 0x1p-132);
  ExpectFloatFma(LOOM_SCALAR_TYPE_F16, 0x1p-24, 0.5, 0x1p-24, 0x1p-23);
  ExpectFloatFma(LOOM_SCALAR_TYPE_F8E5M2, 0x1p-16, 0.5, 0x1p-16, 0x1p-15);
  ExpectFloatFma(LOOM_SCALAR_TYPE_F8E4M3, 0x1p-9, 0.5, 0x1p-9, 0x1p-8);
  ExpectFloatFma(LOOM_SCALAR_TYPE_BF16, 0x1p-133, 0x1p-133, 1.0, 1.0);
}

TEST(FloatFacts, FmaPreservesSpecialValuesAndDynamicOperands) {
  for (const auto scalar_type :
       {LOOM_SCALAR_TYPE_F8E4M3, LOOM_SCALAR_TYPE_F8E5M2, LOOM_SCALAR_TYPE_F16,
        LOOM_SCALAR_TYPE_BF16, LOOM_SCALAR_TYPE_F32, LOOM_SCALAR_TYPE_F64}) {
    SCOPED_TRACE(static_cast<int>(scalar_type));
    ExpectFloatFma(scalar_type, -0.0, 1.0, -0.0, -0.0);
    ExpectFloatFma(scalar_type, -0.0, 1.0, 0.0, 0.0);
    ExpectFloatFma(scalar_type, 2.0, 3.0, -6.0, 0.0);
    ExpectFloatFma(scalar_type, NAN, 1.0, 1.0, NAN);
    if (scalar_type != LOOM_SCALAR_TYPE_F8E4M3) {
      ExpectFloatFma(scalar_type, INFINITY, 1.0, -INFINITY, NAN);
      ExpectFloatFma(scalar_type, INFINITY, 0.0, 1.0, NAN);
      ExpectFloatFma(scalar_type, 1.0, 1.0, -INFINITY, -INFINITY);
    }
    for (int i = 0; i < 3; ++i) {
      const auto one = loom_value_facts_exact_float(scalar_type, 1.0);
      loom_value_facts_t operands[] = {one, one, one};
      operands[i] = loom_value_facts_unknown();
      loom_value_facts_t result = loom_value_facts_unknown();
      loom_value_facts_eval_float_fma(scalar_type, &operands[0], &operands[1],
                                      &operands[2], &result);
      EXPECT_TRUE(loom_value_facts_is_unknown(result));
    }
  }
  // The binary64 path keeps true fused arithmetic instead of staging a product.
  ExpectFloatFma(LOOM_SCALAR_TYPE_F64, 0x1.0000002p0, 0x1.ffffffcp-1, -1.0,
                 -0x1p-54);
}

TEST(FloatFacts, NarrowConstantsRetainTheirSideOfRoundingMidpoints) {
  const struct {
    // Destination of the source literal conversion.
    loom_scalar_type_t type;
    // F64 source immediately beside a destination rounding midpoint.
    double source;
    // Independently calculated destination value.
    double expected;
  } cases[] = {
      {LOOM_SCALAR_TYPE_F8E4M3, 0x1.1000000000001p0, 1.125},
      {LOOM_SCALAR_TYPE_F8E4M3, 0x1.2ffffffffffffp0, 1.125},
      {LOOM_SCALAR_TYPE_F8E5M2, 0x1.2000000000001p0, 1.25},
      {LOOM_SCALAR_TYPE_F8E5M2, 0x1.5ffffffffffffp0, 1.25},
      {LOOM_SCALAR_TYPE_F16, 0x1.0020000000001p0, 0x1.004p0},
      {LOOM_SCALAR_TYPE_BF16, 0x1.0100000000001p0, 0x1.02p0},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(loom_scalar_type_name(test.type));
    for (double sign : {1.0, -1.0}) {
      EXPECT_EQ(ExactFloatValue(test.type, loom_value_facts_exact_float(
                                               test.type, sign * test.source)),
                sign * test.expected);
    }
  }
}

TEST(FloatFacts, DistinguishesFusedAndStagedF32Arithmetic) {
  loom_value_facts_t a =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 4097.0);
  loom_value_facts_t b =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 4097.0);
  loom_value_facts_t c =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, -16785408.0);

  loom_value_facts_t fused = loom_value_facts_unknown();
  loom_value_facts_eval_float_fma(LOOM_SCALAR_TYPE_F32, &a, &b, &c, &fused);

  loom_value_facts_t product = loom_value_facts_unknown();
  loom_value_facts_eval_float_binary(LOOM_SCALAR_TYPE_F32, &a, &b, MulF32,
                                     MulF64, &product);
  loom_value_facts_t staged = loom_value_facts_unknown();
  loom_value_facts_eval_float_binary(LOOM_SCALAR_TYPE_F32, &product, &c, AddF32,
                                     AddF64, &staged);

  EXPECT_FLOAT_EQ(
      static_cast<float>(ExactFloatValue(LOOM_SCALAR_TYPE_F32, fused)), 1.0f);
  EXPECT_FLOAT_EQ(
      static_cast<float>(ExactFloatValue(LOOM_SCALAR_TYPE_F32, staged)), 0.0f);
}

TEST(FloatFacts, MinimumAndMaximumSelectSignedZero) {
  loom_value_facts_t positive_zero =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 0.0);
  loom_value_facts_t negative_zero =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, -0.0);
  loom_value_facts_t minimum = loom_value_facts_unknown();
  loom_value_facts_t maximum = loom_value_facts_unknown();
  loom_value_facts_eval_float_minmax(LOOM_SCALAR_TYPE_F32,
                                     LOOM_FLOAT_MINMAX_MINIMUM, &positive_zero,
                                     &negative_zero, &minimum);
  loom_value_facts_eval_float_minmax(LOOM_SCALAR_TYPE_F32,
                                     LOOM_FLOAT_MINMAX_MAXIMUM, &positive_zero,
                                     &negative_zero, &maximum);

  EXPECT_TRUE(std::signbit(ExactFloatValue(LOOM_SCALAR_TYPE_F32, minimum)));
  EXPECT_FALSE(std::signbit(ExactFloatValue(LOOM_SCALAR_TYPE_F32, maximum)));
}

TEST(FloatFacts, RawNanBitcastRemainsNonMaterializable) {
  loom_value_facts_t signaling_nan_bits =
      loom_value_facts_exact_i64(UINT32_C(0x7F800001));
  loom_value_facts_t signaling_nan = loom_value_facts_unknown();
  loom_value_facts_eval_scalar_bitcast(LOOM_SCALAR_TYPE_I32,
                                       LOOM_SCALAR_TYPE_F32,
                                       &signaling_nan_bits, &signaling_nan);

  EXPECT_TRUE(loom_value_facts_is_nan(signaling_nan));
  EXPECT_FALSE(loom_value_facts_is_exact(signaling_nan));

  loom_value_facts_t negated = loom_value_facts_unknown();
  loom_value_facts_eval_float_negate(LOOM_SCALAR_TYPE_F32, &signaling_nan,
                                     &negated);
  EXPECT_TRUE(loom_value_facts_is_nan(negated));
  EXPECT_FALSE(loom_value_facts_is_exact(negated));

  loom_value_facts_t roundtrip_bits = loom_value_facts_unknown();
  loom_value_facts_eval_scalar_bitcast(
      LOOM_SCALAR_TYPE_F32, LOOM_SCALAR_TYPE_I32, &negated, &roundtrip_bits);
  EXPECT_TRUE(loom_value_facts_is_unknown(roundtrip_bits));
}

TEST(FloatFacts, ConstructsExactFactsFromDeclaredWidthBits) {
  loom_value_facts_t bf16_value = loom_value_facts_unknown();
  ASSERT_TRUE(loom_value_facts_from_float_bits(
      LOOM_SCALAR_TYPE_BF16, UINT64_C(0xFFFF000000003F80), &bf16_value));
  EXPECT_DOUBLE_EQ(ExactFloatValue(LOOM_SCALAR_TYPE_BF16, bf16_value), 1.0);

  loom_value_facts_t negative_zero = loom_value_facts_unknown();
  ASSERT_TRUE(loom_value_facts_from_float_bits(
      LOOM_SCALAR_TYPE_F32, UINT64_C(0xFFFFFFFF80000000), &negative_zero));
  const double negative_zero_value =
      ExactFloatValue(LOOM_SCALAR_TYPE_F32, negative_zero);
  EXPECT_EQ(negative_zero_value, 0.0);
  EXPECT_TRUE(std::signbit(negative_zero_value));
}

TEST(FloatFacts, ConstructsKnownNanFactsFromRawPayloads) {
  loom_value_facts_t signaling_nan = loom_value_facts_unknown();
  ASSERT_TRUE(loom_value_facts_from_float_bits(
      LOOM_SCALAR_TYPE_F32, UINT32_C(0x7F800001), &signaling_nan));
  EXPECT_TRUE(loom_value_facts_is_nan(signaling_nan));
  EXPECT_FALSE(loom_value_facts_is_exact(signaling_nan));

  loom_value_facts_t unsupported = loom_value_facts_exact_i64(7);
  EXPECT_FALSE(loom_value_facts_from_float_bits(
      LOOM_SCALAR_TYPE_I32, UINT32_C(0x3F800000), &unsupported));
  EXPECT_TRUE(loom_value_facts_is_exact(unsupported));
}

TEST(FloatFacts, BitcastPreservesBF16SubnormalUnderFlushToZero) {
  loom_value_facts_t source_bits = loom_value_facts_exact_i64(1);
  loom_value_facts_t value = loom_value_facts_unknown();

  const iree_fpu_state_t fpu_state =
      iree_fpu_state_push(IREE_FPU_STATE_FLAG_FLUSH_DENORMALS_TO_ZERO);
  loom_value_facts_eval_scalar_bitcast(
      LOOM_SCALAR_TYPE_I16, LOOM_SCALAR_TYPE_BF16, &source_bits, &value);
  double exact_value = 0.0;
  const bool is_exact = loom_value_facts_as_exact_float(LOOM_SCALAR_TYPE_BF16,
                                                        value, &exact_value);
  uint64_t exact_bits = 0;
  std::memcpy(&exact_bits, &exact_value, sizeof(exact_bits));
  iree_fpu_state_pop(fpu_state);

  EXPECT_TRUE(is_exact);
  EXPECT_EQ(UINT64_C(0x37A0000000000000), exact_bits);
}

TEST(FloatFacts, MinnumSkipsOneNanAndCanonicalizesTwo) {
  loom_value_facts_t nan = loom_value_facts_known_nan();
  loom_value_facts_t number =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 3.0);
  loom_value_facts_t one_nan = loom_value_facts_unknown();
  loom_value_facts_eval_float_minmax(
      LOOM_SCALAR_TYPE_F32, LOOM_FLOAT_MINMAX_MINNUM, &nan, &number, &one_nan);
  EXPECT_FLOAT_EQ(
      static_cast<float>(ExactFloatValue(LOOM_SCALAR_TYPE_F32, one_nan)), 3.0f);

  loom_value_facts_t two_nans = loom_value_facts_unknown();
  loom_value_facts_eval_float_minmax(
      LOOM_SCALAR_TYPE_F32, LOOM_FLOAT_MINMAX_MINNUM, &nan, &nan, &two_nans);
  EXPECT_TRUE(loom_value_facts_is_nan(two_nans));
  EXPECT_TRUE(loom_value_facts_is_exact(two_nans));
}

TEST(FloatFacts, ClampModesHaveDistinctNanSemantics) {
  loom_value_facts_t value =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 2.0);
  loom_value_facts_t lower = loom_value_facts_known_nan();
  loom_value_facts_t upper =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 1.0);
  loom_value_facts_t ordered = loom_value_facts_unknown();
  loom_value_facts_t number = loom_value_facts_unknown();
  loom_value_facts_t ieee = loom_value_facts_unknown();
  loom_value_facts_eval_float_clamp(LOOM_SCALAR_TYPE_F32,
                                    LOOM_FLOAT_CLAMP_ORDERED, &value, &lower,
                                    &upper, &ordered);
  loom_value_facts_eval_float_clamp(LOOM_SCALAR_TYPE_F32,
                                    LOOM_FLOAT_CLAMP_NUMBER, &value, &lower,
                                    &upper, &number);
  loom_value_facts_eval_float_clamp(LOOM_SCALAR_TYPE_F32, LOOM_FLOAT_CLAMP_IEEE,
                                    &value, &lower, &upper, &ieee);

  EXPECT_FLOAT_EQ(
      static_cast<float>(ExactFloatValue(LOOM_SCALAR_TYPE_F32, ordered)), 1.0f);
  EXPECT_FLOAT_EQ(
      static_cast<float>(ExactFloatValue(LOOM_SCALAR_TYPE_F32, number)), 1.0f);
  EXPECT_TRUE(loom_value_facts_is_nan(ieee));
}

TEST(FloatFacts, MeetPreservesSharedFloatClassFacts) {
  loom_value_facts_t one =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 1.0);
  loom_value_facts_t two =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 2.0);
  loom_value_facts_t finite = loom_value_facts_unknown();
  loom_value_facts_meet(&one, &two, &finite);

  EXPECT_TRUE(loom_value_facts_is_float(finite));
  EXPECT_FALSE(loom_value_facts_is_exact(finite));
  EXPECT_TRUE(loom_value_facts_is_finite(finite));
  EXPECT_TRUE(loom_value_facts_is_workgroup_uniform(finite));

  loom_value_facts_t raw_nan = loom_value_facts_known_nan();
  loom_value_facts_t exact_nan = loom_value_facts_exact_float(
      LOOM_SCALAR_TYPE_F32, std::numeric_limits<float>::quiet_NaN());
  loom_value_facts_t nan = loom_value_facts_unknown();
  loom_value_facts_meet(&raw_nan, &exact_nan, &nan);

  EXPECT_TRUE(loom_value_facts_is_float(nan));
  EXPECT_TRUE(loom_value_facts_is_nan(nan));
  EXPECT_FALSE(loom_value_facts_is_exact(nan));
}

}  // namespace
}  // namespace loom
