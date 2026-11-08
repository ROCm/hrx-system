// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <climits>
#include <cstdint>

#include "iree/testing/gtest.h"
#include "loom/ir/facts.h"

namespace loom {
namespace {

static void ExpectExact(loom_value_facts_t facts, int64_t expected) {
  EXPECT_TRUE(loom_value_facts_is_exact(facts));
  EXPECT_EQ(facts.range_lo, expected);
}

TEST(ModularIntegerFacts, ExactArithmeticAcrossWidths) {
  struct TestCase {
    int32_t bit_count;
    int64_t maximum;
    int64_t minimum;
  };
  const TestCase cases[] = {
      {1, 1, 0},
      {8, INT8_MAX, INT8_MIN},
      {16, INT16_MAX, INT16_MIN},
      {32, INT32_MAX, INT32_MIN},
      {64, INT64_MAX, INT64_MIN},
  };
  for (const TestCase& test_case : cases) {
    SCOPED_TRACE(test_case.bit_count);
    const loom_value_facts_t maximum =
        loom_value_facts_exact_i64(test_case.maximum);
    const loom_value_facts_t minimum =
        loom_value_facts_exact_i64(test_case.minimum);
    const loom_value_facts_t one = loom_value_facts_exact_i64(1);
    const loom_value_facts_t multiplier =
        loom_value_facts_exact_i64(test_case.bit_count == 1 ? 1 : 2);
    loom_value_facts_t result;

    loom_value_facts_addi_modular(&maximum, &one, test_case.bit_count, &result);
    ExpectExact(result, test_case.minimum);
    loom_value_facts_subi_modular(&minimum, &one, test_case.bit_count, &result);
    ExpectExact(result, test_case.maximum);
    loom_value_facts_muli_modular(&maximum, &multiplier, test_case.bit_count,
                                  &result);
    ExpectExact(result, test_case.bit_count == 1 ? 1 : -2);
    loom_value_facts_fmai_modular(&maximum, &multiplier, &one,
                                  test_case.bit_count, &result);
    ExpectExact(result, test_case.bit_count == 1 ? 0 : -1);
  }
}

TEST(ModularIntegerFacts, ExactUnaryAndShiftAcrossWidths) {
  struct TestCase {
    int32_t bit_count;
    int64_t minimum;
  };
  const TestCase cases[] = {
      {1, 1}, {8, INT8_MIN}, {16, INT16_MIN}, {32, INT32_MIN}, {64, INT64_MIN},
  };
  for (const TestCase& test_case : cases) {
    SCOPED_TRACE(test_case.bit_count);
    const loom_value_facts_t minimum =
        loom_value_facts_exact_i64(test_case.minimum);
    const loom_value_facts_t one = loom_value_facts_exact_i64(1);
    const loom_value_facts_t shift =
        loom_value_facts_exact_i64(test_case.bit_count - 1);
    loom_value_facts_t result;

    loom_value_facts_negi_modular(&minimum, test_case.bit_count, &result);
    ExpectExact(result, test_case.minimum);
    loom_value_facts_absi_modular(&minimum, test_case.bit_count, &result);
    ExpectExact(result, test_case.minimum);
    loom_value_facts_shli_modular(&one, &shift, test_case.bit_count, &result);
    ExpectExact(result, test_case.minimum);
  }
}

TEST(ModularIntegerFacts, ExactCarryAndBorrowAcrossWidths) {
  for (int32_t bit_count : {1, 8, 16, 32, 64}) {
    SCOPED_TRACE(bit_count);
    const loom_value_facts_t all_bits =
        loom_value_facts_exact_i64(bit_count == 1 ? 1 : -1);
    const loom_value_facts_t zero = loom_value_facts_exact_i64(0);
    const loom_value_facts_t one = loom_value_facts_exact_i64(1);
    loom_value_facts_t value;
    loom_value_facts_t carry_or_borrow;

    loom_value_facts_addcarryui(&all_bits, &one, bit_count, &value,
                                &carry_or_borrow);
    ExpectExact(value, 0);
    ExpectExact(carry_or_borrow, 1);

    loom_value_facts_subborrowui(&zero, &all_bits, bit_count, &value,
                                 &carry_or_borrow);
    ExpectExact(value, 1);
    ExpectExact(carry_or_borrow, 1);

    loom_value_facts_addcarryui(&zero, &zero, bit_count, &value,
                                &carry_or_borrow);
    ExpectExact(value, 0);
    ExpectExact(carry_or_borrow, 0);
    loom_value_facts_subborrowui(&all_bits, &zero, bit_count, &value,
                                 &carry_or_borrow);
    ExpectExact(value, bit_count == 1 ? 1 : -1);
    ExpectExact(carry_or_borrow, 0);
  }
}

TEST(ModularIntegerFacts, CarryAndBorrowPreserveLanePredicates) {
  loom_value_facts_t lhs = loom_value_facts_make(INT8_MIN, INT8_MAX, 1);
  loom_value_facts_mark_lane_varying(&lhs);
  const loom_value_facts_t rhs = loom_value_facts_make(INT8_MIN, INT8_MAX, 1);
  loom_value_facts_t value;
  loom_value_facts_t carry_or_borrow;

  loom_value_facts_addcarryui(&lhs, &rhs, 8, &value, &carry_or_borrow);
  EXPECT_TRUE(loom_value_facts_is_lane_varying(value));
  EXPECT_FALSE(loom_value_facts_is_lane_predicate(value));
  EXPECT_TRUE(loom_value_facts_is_lane_predicate(carry_or_borrow));

  loom_value_facts_subborrowui(&lhs, &rhs, 8, &value, &carry_or_borrow);
  EXPECT_TRUE(loom_value_facts_is_lane_varying(value));
  EXPECT_FALSE(loom_value_facts_is_lane_predicate(value));
  EXPECT_TRUE(loom_value_facts_is_lane_predicate(carry_or_borrow));
}

TEST(ModularIntegerFacts, RangesRemainSoundAcrossWrap) {
  loom_value_facts_t lhs = loom_value_facts_make(120, 127, 1);
  loom_value_facts_mark_lane_varying(&lhs);
  const loom_value_facts_t rhs = loom_value_facts_make(1, 16, 1);
  loom_value_facts_t result;
  loom_value_facts_addi_modular(&lhs, &rhs, 8, &result);
  EXPECT_EQ(result.range_lo, INT8_MIN);
  EXPECT_EQ(result.range_hi, INT8_MAX);
  EXPECT_TRUE(loom_value_facts_is_lane_varying(result));

  lhs = loom_value_facts_make(60, 64, 4);
  const loom_value_facts_t two = loom_value_facts_exact_i64(2);
  loom_value_facts_muli_modular(&lhs, &two, 8, &result);
  EXPECT_EQ(result.range_lo, INT8_MIN);
  EXPECT_EQ(result.range_hi, INT8_MAX);
  EXPECT_EQ(result.known_divisor, 8);
}

TEST(ModularIntegerFacts, OutOfDomainShiftDoesNotFold) {
  const loom_value_facts_t one = loom_value_facts_exact_i64(1);
  const loom_value_facts_t width = loom_value_facts_exact_i64(8);
  loom_value_facts_t result;
  loom_value_facts_shli_modular(&one, &width, 8, &result);
  EXPECT_EQ(result.range_lo, INT8_MIN);
  EXPECT_EQ(result.range_hi, INT8_MAX);
}

TEST(ModularIntegerFacts, ZeroMultiplyAddTermsRetainDivisibilityAcrossWidths) {
  const loom_value_facts_t zero = loom_value_facts_exact_i64(0);
  for (int32_t bit_count : {1, 8, 16, 32, 64}) {
    SCOPED_TRACE(bit_count);
    const int64_t divisor = bit_count == 1 ? 1 : 16;
    const loom_value_facts_t factor = loom_value_facts_exact_i64(divisor);
    const loom_value_facts_t input =
        loom_value_facts_make(0, bit_count == 1 ? 1 : 31, 1);
    loom_value_facts_t output;
    loom_value_facts_fmai_modular(&input, &factor, &zero, bit_count, &output);
    EXPECT_EQ(output.known_divisor, divisor);
    EXPECT_EQ(output.range_lo, bit_count == 8 ? INT8_MIN : 0);
    EXPECT_EQ(output.range_hi, bit_count == 1   ? 1
                               : bit_count == 8 ? INT8_MAX
                                                : 496);
    const loom_value_facts_t addend = output;
    loom_value_facts_fmai_modular(&input, &zero, &addend, bit_count, &output);
    EXPECT_TRUE(loom_value_facts_equal(output, addend));
    loom_value_facts_fmai_modular(&zero, &input, &addend, bit_count, &output);
    EXPECT_TRUE(loom_value_facts_equal(output, addend));
  }
}

TEST(ModularIntegerFacts, OutputMayAliasInputs) {
  loom_value_facts_t lhs = loom_value_facts_exact_i64(INT64_MAX);
  const loom_value_facts_t one = loom_value_facts_exact_i64(1);
  loom_value_facts_addi_modular(&lhs, &one, 64, &lhs);
  ExpectExact(lhs, INT64_MIN);

  loom_value_facts_t rhs = loom_value_facts_exact_i64(1);
  loom_value_facts_subi_modular(&lhs, &rhs, 64, &rhs);
  ExpectExact(rhs, INT64_MAX);
}

}  // namespace
}  // namespace loom
