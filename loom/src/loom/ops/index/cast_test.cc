// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/index/cast.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(IndexCastTest, DescribesEveryFixedWidthAddressInterpretation) {
  for (auto type :
       {LOOM_SCALAR_TYPE_I1, LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_I16,
        LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I64}) {
    for (auto address : {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_OFFSET}) {
      loom_integer_cast_kind_t kind;
      uint8_t bit_count;
      ASSERT_TRUE(loom_index_cast_payload_interpretation(address, type, &kind,
                                                         &bit_count));
      EXPECT_EQ(bit_count, loom_scalar_type_bitwidth(type));
      EXPECT_EQ(kind, type == LOOM_SCALAR_TYPE_I1 ? LOOM_INTEGER_CAST_UNSIGNED
                                                  : LOOM_INTEGER_CAST_SIGNED);
      const bool fixed_interpretation = loom_index_cast_payload_interpretation(
          type, address, &kind, &bit_count);
      EXPECT_EQ(fixed_interpretation, type != LOOM_SCALAR_TYPE_I64 ||
                                          address == LOOM_SCALAR_TYPE_INDEX);
      if (fixed_interpretation) {
        EXPECT_EQ(bit_count, loom_scalar_type_bitwidth(type));
        EXPECT_EQ(kind, type == LOOM_SCALAR_TYPE_I1 ||
                                address == LOOM_SCALAR_TYPE_OFFSET
                            ? LOOM_INTEGER_CAST_UNSIGNED
                            : LOOM_INTEGER_CAST_SIGNED);
      }
    }
  }
  for (auto input : {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_OFFSET}) {
    for (auto result : {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_OFFSET}) {
      loom_integer_cast_kind_t kind;
      uint8_t bit_count;
      EXPECT_FALSE(loom_index_cast_payload_interpretation(input, result, &kind,
                                                          &bit_count));
    }
  }
}

TEST(IndexCastTest, IdentifiesValuePreservingIntegerAddressCasts) {
  struct TestCase {
    loom_scalar_type_t input_type;
    loom_scalar_type_t result_type;
    loom_value_facts_t input_facts;
    bool preserves_value;
  };
  const loom_value_facts_t unknown = loom_value_facts_unknown();
  const TestCase test_cases[] = {
      {LOOM_SCALAR_TYPE_I1, LOOM_SCALAR_TYPE_INDEX, unknown, true},
      {LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_INDEX, unknown, true},
      {LOOM_SCALAR_TYPE_I16, LOOM_SCALAR_TYPE_INDEX, unknown, true},
      {LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_INDEX, unknown, true},
      {LOOM_SCALAR_TYPE_I64, LOOM_SCALAR_TYPE_INDEX, unknown, true},
      {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_I64, unknown, true},
      {LOOM_SCALAR_TYPE_OFFSET, LOOM_SCALAR_TYPE_INDEX, unknown, true},
      {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_OFFSET, unknown, false},
      {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_OFFSET,
       loom_value_facts_make(0, INT32_MAX, 1), true},
      {LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_OFFSET, unknown, false},
      {LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_OFFSET,
       loom_value_facts_make(0, INT32_MAX, 1), true},
      {LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_OFFSET,
       loom_value_facts_exact_i64(-1), false},
      {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_I32, unknown, false},
      {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_I32,
       loom_value_facts_make(INT32_MIN, INT32_MAX, 1), true},
      {LOOM_SCALAR_TYPE_OFFSET, LOOM_SCALAR_TYPE_I32, unknown, false},
      {LOOM_SCALAR_TYPE_OFFSET, LOOM_SCALAR_TYPE_I32,
       loom_value_facts_make(0, INT32_MAX, 1), true},
      {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_I8,
       loom_value_facts_make(INT8_MIN, INT8_MAX, 1), true},
      {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_I8,
       loom_value_facts_make(INT8_MIN, INT8_MAX + 1, 1), false},
      {LOOM_SCALAR_TYPE_F32, LOOM_SCALAR_TYPE_INDEX, unknown, false},
      {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_F32, unknown, false},
  };

  for (const auto& test_case : test_cases) {
    EXPECT_EQ(
        loom_index_cast_transfer_preserves_value(
            test_case.input_type, test_case.result_type, test_case.input_facts),
        test_case.preserves_value)
        << loom_scalar_type_name(test_case.input_type) << " -> "
        << loom_scalar_type_name(test_case.result_type);
  }
}

TEST(IndexCastTest, UnsignedWideningRetainsNonzeroAndDistribution) {
  for (auto type :
       {LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_I16, LOOM_SCALAR_TYPE_I32}) {
    for (auto input :
         {loom_value_facts_make(-8, 8, 2), loom_value_facts_make(-8, -2, 2),
          loom_value_facts_make(2, 8, 2)}) {
      input.flags |= LOOM_VALUE_FACT_NON_ZERO;
      loom_value_facts_mark_workgroup_uniform(&input);
      auto result =
          loom_index_cast_transfer_facts(type, LOOM_SCALAR_TYPE_OFFSET, input);
      EXPECT_TRUE(loom_value_facts_is_positive(result));
      EXPECT_TRUE(loom_value_facts_is_workgroup_uniform(result));
      EXPECT_EQ(result.known_divisor, 2);
      result = loom_index_cast_transfer_facts(LOOM_SCALAR_TYPE_OFFSET,
                                              LOOM_SCALAR_TYPE_INDEX, result);
      EXPECT_TRUE(loom_value_facts_is_positive(result));
    }
  }
  const auto wide = loom_value_facts_make(1, 256, 1);
  const auto narrowed = loom_index_cast_transfer_facts(
      LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_I8, wide);
  EXPECT_FALSE(loom_value_facts_is_non_zero(narrowed));
}

}  // namespace
}  // namespace loom
