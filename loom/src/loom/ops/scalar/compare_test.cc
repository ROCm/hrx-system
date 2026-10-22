// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/scalar/compare.h"

#include "iree/testing/gtest.h"
#include "loom/ops/scalar/ops.h"

namespace loom {
namespace {

TEST(ScalarCompareTest, IdentityPredicatesPreserveNanSemantics) {
  const struct {
    // Predicate interpreted with two identical floating-point values.
    uint8_t predicate;
    // Result when the value is ordered, including infinities and signed zero.
    bool ordered_result;
    // Result when the value is NaN.
    bool nan_result;
  } cases[] = {
      {LOOM_SCALAR_CMPF_PREDICATE_OEQ, true, false},
      {LOOM_SCALAR_CMPF_PREDICATE_OGT, false, false},
      {LOOM_SCALAR_CMPF_PREDICATE_OGE, true, false},
      {LOOM_SCALAR_CMPF_PREDICATE_OLT, false, false},
      {LOOM_SCALAR_CMPF_PREDICATE_OLE, true, false},
      {LOOM_SCALAR_CMPF_PREDICATE_ONE, false, false},
      {LOOM_SCALAR_CMPF_PREDICATE_ORD, true, false},
      {LOOM_SCALAR_CMPF_PREDICATE_UEQ, true, true},
      {LOOM_SCALAR_CMPF_PREDICATE_UGT, false, true},
      {LOOM_SCALAR_CMPF_PREDICATE_UGE, true, true},
      {LOOM_SCALAR_CMPF_PREDICATE_ULT, false, true},
      {LOOM_SCALAR_CMPF_PREDICATE_ULE, true, true},
      {LOOM_SCALAR_CMPF_PREDICATE_UNE, false, true},
      {LOOM_SCALAR_CMPF_PREDICATE_UNO, false, true},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(unsigned(test.predicate));
    bool result = false;
    const bool proven = loom_scalar_cmpf_constant_result(
        test.predicate, 1, 1, /*fastmath=*/0, &result);
    EXPECT_EQ(proven, test.ordered_result == test.nan_result);
    if (proven) {
      EXPECT_EQ(result, test.ordered_result);
    }
    ASSERT_TRUE(loom_scalar_cmpf_constant_result(
        test.predicate, 1, 1, LOOM_SCALAR_FASTMATHFLAGS_NNAN, &result));
    EXPECT_EQ(result, test.ordered_result);
  }
}

TEST(ScalarCompareTest, UnsignedRangesMatchEnumeratedIntegerValues) {
  constexpr int bounds[] = {-128, -2, -1, 0, 1, 126, 127};
  constexpr uint8_t predicates[] = {
      LOOM_SCALAR_CMPI_PREDICATE_ULT, LOOM_SCALAR_CMPI_PREDICATE_ULE,
      LOOM_SCALAR_CMPI_PREDICATE_UGT, LOOM_SCALAR_CMPI_PREDICATE_UGE};
  for (int lhs_lower : bounds) {
    for (int lhs_upper : bounds) {
      if (lhs_lower > lhs_upper) {
        continue;
      }
      const auto lhs_facts = loom_value_facts_make(lhs_lower, lhs_upper, 1);
      for (int rhs_lower : bounds) {
        for (int rhs_upper : bounds) {
          if (rhs_lower > rhs_upper) {
            continue;
          }
          const auto rhs_facts = loom_value_facts_make(rhs_lower, rhs_upper, 1);
          for (uint8_t predicate : predicates) {
            unsigned outcomes = 0;
            for (int lhs = lhs_lower; lhs <= lhs_upper && outcomes != 3;
                 ++lhs) {
              for (int rhs = rhs_lower; rhs <= rhs_upper && outcomes != 3;
                   ++rhs) {
                const auto left = static_cast<uint8_t>(lhs);
                const auto right = static_cast<uint8_t>(rhs);
                bool result = false;
                switch (predicate) {
                  case LOOM_SCALAR_CMPI_PREDICATE_ULT:
                    result = left < right;
                    break;
                  case LOOM_SCALAR_CMPI_PREDICATE_ULE:
                    result = left <= right;
                    break;
                  case LOOM_SCALAR_CMPI_PREDICATE_UGT:
                    result = left > right;
                    break;
                  case LOOM_SCALAR_CMPI_PREDICATE_UGE:
                    result = left >= right;
                    break;
                }
                outcomes |= result ? 2u : 1u;
              }
            }
            SCOPED_TRACE(::testing::Message()
                         << lhs_lower << ".." << lhs_upper << ", " << rhs_lower
                         << ".." << rhs_upper << ", predicate "
                         << unsigned(predicate));
            bool result = false;
            const bool proven = loom_scalar_cmpi_result_from_facts(
                LOOM_SCALAR_TYPE_I8, predicate, &lhs_facts, &rhs_facts,
                &result);
            ASSERT_EQ(proven, outcomes != 3);
            if (proven) {
              EXPECT_EQ(result, outcomes == 2);
            }
          }
        }
      }
    }
  }
}

TEST(ScalarCompareTest, UnsignedEndpointsAcrossScalarWidths) {
  for (auto type :
       {LOOM_SCALAR_TYPE_I1, LOOM_SCALAR_TYPE_I8, LOOM_SCALAR_TYPE_I16,
        LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I64}) {
    int64_t minimum = 0;
    int64_t maximum = 0;
    ASSERT_TRUE(loom_value_facts_scalar_type_domain(type, &minimum, &maximum));
    const auto range = loom_value_facts_make(minimum, maximum, 1);
    const auto zero = loom_value_facts_exact_i64(0);
    const auto all_ones =
        loom_value_facts_exact_i64(type == LOOM_SCALAR_TYPE_I1 ? 1 : -1);
    const struct {
      // Operand interval on the left side of the comparison.
      loom_value_facts_t lhs;
      // Operand interval on the right side of the comparison.
      loom_value_facts_t rhs;
      // Unsigned comparison predicate.
      uint8_t predicate;
      // Proven Boolean result.
      bool expected;
    } cases[] = {{range, zero, LOOM_SCALAR_CMPI_PREDICATE_ULT, false},
                 {range, zero, LOOM_SCALAR_CMPI_PREDICATE_UGE, true},
                 {range, all_ones, LOOM_SCALAR_CMPI_PREDICATE_UGT, false},
                 {range, all_ones, LOOM_SCALAR_CMPI_PREDICATE_ULE, true},
                 {zero, range, LOOM_SCALAR_CMPI_PREDICATE_ULE, true},
                 {all_ones, range, LOOM_SCALAR_CMPI_PREDICATE_ULT, false}};
    for (const auto& test : cases) {
      bool result = false;
      ASSERT_TRUE(loom_scalar_cmpi_result_from_facts(
          type, test.predicate, &test.lhs, &test.rhs, &result));
      EXPECT_EQ(result, test.expected);
    }
  }
}

}  // namespace
}  // namespace loom
