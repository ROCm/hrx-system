// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/decision/predicate.h"

#include <cmath>
#include <cstdint>

#include "iree/testing/gtest.h"
#include "loom/ir/float_facts.h"

namespace {

static loom_decision_predicate_operand_t Operand(
    loom_value_facts_t facts,
    uint32_t identity = LOOM_DECISION_OPERAND_IDENTITY_NONE) {
  return {
      /*.facts=*/facts,
      /*.identity=*/identity,
  };
}

static loom_decision_truth_t TruthFromPossibilities(bool can_be_false,
                                                    bool can_be_true) {
  if (can_be_false && can_be_true) return LOOM_DECISION_TRUTH_UNKNOWN;
  return can_be_true ? LOOM_DECISION_TRUTH_TRUE : LOOM_DECISION_TRUTH_FALSE;
}

static bool EvaluateExactRelation(loom_predicate_kind_t predicate_kind,
                                  int64_t lhs, int64_t rhs) {
  switch (predicate_kind) {
    case LOOM_PREDICATE_EQ:
      return lhs == rhs;
    case LOOM_PREDICATE_NE:
      return lhs != rhs;
    case LOOM_PREDICATE_LT:
      return lhs < rhs;
    case LOOM_PREDICATE_LE:
      return lhs <= rhs;
    case LOOM_PREDICATE_GT:
      return lhs > rhs;
    case LOOM_PREDICATE_GE:
      return lhs >= rhs;
    case LOOM_PREDICATE_MIN:
      return lhs >= rhs;
    case LOOM_PREDICATE_MAX:
      return lhs <= rhs;
    default:
      return false;
  }
}

static loom_decision_truth_t ReferenceRelation(
    loom_predicate_kind_t predicate_kind, int64_t lhs_lo, int64_t lhs_hi,
    int64_t rhs_lo, int64_t rhs_hi) {
  bool can_be_false = false;
  bool can_be_true = false;
  for (int64_t lhs = lhs_lo; lhs <= lhs_hi; ++lhs) {
    for (int64_t rhs = rhs_lo; rhs <= rhs_hi; ++rhs) {
      if (EvaluateExactRelation(predicate_kind, lhs, rhs)) {
        can_be_true = true;
      } else {
        can_be_false = true;
      }
    }
  }
  return TruthFromPossibilities(can_be_false, can_be_true);
}

TEST(PredicateTest, ExhaustiveSmallIntegerRelations) {
  constexpr loom_predicate_kind_t kPredicateKinds[] = {
      LOOM_PREDICATE_EQ,  LOOM_PREDICATE_NE,  LOOM_PREDICATE_LT,
      LOOM_PREDICATE_LE,  LOOM_PREDICATE_GT,  LOOM_PREDICATE_GE,
      LOOM_PREDICATE_MIN, LOOM_PREDICATE_MAX,
  };
  for (loom_predicate_kind_t predicate_kind : kPredicateKinds) {
    for (int64_t lhs_lo = -3; lhs_lo <= 3; ++lhs_lo) {
      for (int64_t lhs_hi = lhs_lo; lhs_hi <= 3; ++lhs_hi) {
        for (int64_t rhs_lo = -3; rhs_lo <= 3; ++rhs_lo) {
          for (int64_t rhs_hi = rhs_lo; rhs_hi <= 3; ++rhs_hi) {
            const loom_decision_predicate_operand_t operands[3] = {
                Operand(loom_value_facts_make(lhs_lo, lhs_hi, 1)),
                Operand(loom_value_facts_make(rhs_lo, rhs_hi, 1)),
            };
            EXPECT_EQ(
                loom_decision_predicate_evaluate(predicate_kind, operands),
                ReferenceRelation(predicate_kind, lhs_lo, lhs_hi, rhs_lo,
                                  rhs_hi))
                << "kind=" << static_cast<int>(predicate_kind) << " lhs=["
                << lhs_lo << ", " << lhs_hi << "] rhs=[" << rhs_lo << ", "
                << rhs_hi << "]";
          }
        }
      }
    }
  }
}

TEST(PredicateTest, SameRuntimeIdentityProvesReflexiveRelations) {
  const loom_decision_predicate_operand_t operands[3] = {
      Operand(loom_value_facts_unknown(), 7),
      Operand(loom_value_facts_unknown(), 7),
  };
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_EQ, operands),
            LOOM_DECISION_TRUTH_TRUE);
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_NE, operands),
            LOOM_DECISION_TRUTH_FALSE);
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_LT, operands),
            LOOM_DECISION_TRUTH_FALSE);
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_LE, operands),
            LOOM_DECISION_TRUTH_TRUE);
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_GT, operands),
            LOOM_DECISION_TRUTH_FALSE);
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_GE, operands),
            LOOM_DECISION_TRUTH_TRUE);
}

TEST(PredicateTest, IntegerExtremesDoNotOverflow) {
  loom_decision_predicate_operand_t operands[3] = {
      Operand(loom_value_facts_exact_i64(INT64_MIN)),
      Operand(loom_value_facts_exact_i64(INT64_MAX)),
  };
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_LT, operands),
            LOOM_DECISION_TRUTH_TRUE);
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_GE, operands),
            LOOM_DECISION_TRUTH_FALSE);

  operands[1] = Operand(loom_value_facts_exact_i64(1));
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_MUL, operands),
            LOOM_DECISION_TRUTH_TRUE);
  operands[1] = Operand(loom_value_facts_exact_i64(INT64_MAX));
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_MUL, operands),
            LOOM_DECISION_TRUTH_FALSE);

  operands[1] = Operand(loom_value_facts_exact_i64(INT64_MIN));
  operands[2] = Operand(loom_value_facts_exact_i64(INT64_MIN));
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_RANGE, operands),
            LOOM_DECISION_TRUTH_TRUE);
}

TEST(PredicateTest, FloatRelationsRemainUnknown) {
  const loom_decision_predicate_operand_t operands[3] = {
      Operand(loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 1.0), 7),
      Operand(loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 1.0), 7),
  };
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_EQ, operands),
            LOOM_DECISION_TRUTH_UNKNOWN);
}

TEST(PredicateTest, ExhaustiveExactMultiples) {
  for (int64_t value = -100; value <= 100; ++value) {
    for (int64_t divisor = 1; divisor <= 16; ++divisor) {
      const loom_decision_predicate_operand_t operands[3] = {
          Operand(loom_value_facts_exact_i64(value)),
          Operand(loom_value_facts_exact_i64(divisor)),
      };
      EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_MUL, operands),
                value % divisor == 0 ? LOOM_DECISION_TRUTH_TRUE
                                     : LOOM_DECISION_TRUTH_FALSE)
          << "value=" << value << " divisor=" << divisor;
    }
  }
}

TEST(PredicateTest, MultipleUsesDivisorAndRangeProofs) {
  loom_decision_predicate_operand_t operands[3] = {
      Operand(loom_value_facts_make(16, 80, 16)),
      Operand(loom_value_facts_exact_i64(16)),
  };
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_MUL, operands),
            LOOM_DECISION_TRUTH_TRUE);

  operands[0] = Operand(loom_value_facts_make(1, 15, 1));
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_MUL, operands),
            LOOM_DECISION_TRUTH_FALSE);

  operands[0] = Operand(loom_value_facts_make(-15, -1, 1));
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_MUL, operands),
            LOOM_DECISION_TRUTH_FALSE);

  operands[0] = Operand(loom_value_facts_make(1, 16, 1));
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_MUL, operands),
            LOOM_DECISION_TRUTH_UNKNOWN);

  operands[1] = Operand(loom_value_facts_exact_i64(0));
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_MUL, operands),
            LOOM_DECISION_TRUTH_UNKNOWN);

  operands[1] = Operand(loom_value_facts_exact_i64(-16));
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_MUL, operands),
            LOOM_DECISION_TRUTH_UNKNOWN);
}

TEST(PredicateTest, ExhaustiveExactPowerOfTwo) {
  for (int64_t value = -4; value <= 70; ++value) {
    const loom_decision_predicate_operand_t operands[3] = {
        Operand(loom_value_facts_exact_i64(value)),
    };
    const bool is_power_of_two = value > 0 && (value & (value - 1)) == 0;
    EXPECT_EQ(
        loom_decision_predicate_evaluate(LOOM_PREDICATE_POW2, operands),
        is_power_of_two ? LOOM_DECISION_TRUTH_TRUE : LOOM_DECISION_TRUTH_FALSE)
        << "value=" << value;
  }
}

TEST(PredicateTest, PowerOfTwoUsesKnownPredicateFacts) {
  loom_value_facts_t known_power_of_two = loom_value_facts_make(2, 1024, 1);
  known_power_of_two.flags |= LOOM_VALUE_FACT_POWER_OF_TWO;
  loom_decision_predicate_operand_t operands[3] = {
      Operand(known_power_of_two),
  };
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_POW2, operands),
            LOOM_DECISION_TRUTH_TRUE);

  operands[0] = Operand(loom_value_facts_make(-10, 0, 1));
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_POW2, operands),
            LOOM_DECISION_TRUTH_FALSE);

  operands[0] = Operand(loom_value_facts_make(1, 1024, 1));
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_POW2, operands),
            LOOM_DECISION_TRUTH_UNKNOWN);
}

TEST(PredicateTest, ExhaustiveSmallRanges) {
  for (int64_t value_lo = -3; value_lo <= 3; ++value_lo) {
    for (int64_t value_hi = value_lo; value_hi <= 3; ++value_hi) {
      for (int64_t lower = -4; lower <= 4; ++lower) {
        for (int64_t upper = -4; upper <= 4; ++upper) {
          bool can_be_false = false;
          bool can_be_true = false;
          for (int64_t value = value_lo; value <= value_hi; ++value) {
            if (value >= lower && value <= upper) {
              can_be_true = true;
            } else {
              can_be_false = true;
            }
          }
          const loom_decision_predicate_operand_t operands[3] = {
              Operand(loom_value_facts_make(value_lo, value_hi, 1)),
              Operand(loom_value_facts_exact_i64(lower)),
              Operand(loom_value_facts_exact_i64(upper)),
          };
          EXPECT_EQ(
              loom_decision_predicate_evaluate(LOOM_PREDICATE_RANGE, operands),
              TruthFromPossibilities(can_be_false, can_be_true))
              << "value=[" << value_lo << ", " << value_hi << "] bounds=["
              << lower << ", " << upper << "]";
        }
      }
    }
  }
}

TEST(PredicateTest, FloatClassPredicatesAreTernary) {
  const loom_value_facts_t finite =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 1.0);
  const loom_value_facts_t nan = loom_value_facts_known_nan();
  const loom_value_facts_t inf =
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, INFINITY);
  loom_value_facts_t unknown = loom_value_facts_unknown();
  unknown.flags |= LOOM_VALUE_FACT_FLOAT;

  struct TestCase {
    loom_value_facts_t facts;
    loom_decision_truth_t not_nan;
    loom_decision_truth_t not_inf;
    loom_decision_truth_t finite;
  };
  const TestCase test_cases[] = {
      {finite, LOOM_DECISION_TRUTH_TRUE, LOOM_DECISION_TRUTH_TRUE,
       LOOM_DECISION_TRUTH_TRUE},
      {nan, LOOM_DECISION_TRUTH_FALSE, LOOM_DECISION_TRUTH_TRUE,
       LOOM_DECISION_TRUTH_FALSE},
      {inf, LOOM_DECISION_TRUTH_TRUE, LOOM_DECISION_TRUTH_FALSE,
       LOOM_DECISION_TRUTH_FALSE},
      {unknown, LOOM_DECISION_TRUTH_UNKNOWN, LOOM_DECISION_TRUTH_UNKNOWN,
       LOOM_DECISION_TRUTH_UNKNOWN},
  };
  for (const TestCase& test_case : test_cases) {
    const loom_decision_predicate_operand_t operands[3] = {
        Operand(test_case.facts),
    };
    EXPECT_EQ(
        loom_decision_predicate_evaluate(LOOM_PREDICATE_NOT_NAN, operands),
        test_case.not_nan);
    EXPECT_EQ(
        loom_decision_predicate_evaluate(LOOM_PREDICATE_NOT_INF, operands),
        test_case.not_inf);
    EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_FINITE, operands),
              test_case.finite);
  }
}

TEST(PredicateTest, SeparateFloatExclusionsProveFiniteWithoutTypeBit) {
  loom_value_facts_t facts = loom_value_facts_unknown();
  // Type compatibility is verified outside the scalar fact payload. Unknown
  // float block arguments can carry class exclusions without a FLOAT bit.
  facts.flags |= LOOM_VALUE_FACT_NOT_NAN | LOOM_VALUE_FACT_NOT_INF;
  const loom_decision_predicate_operand_t operands[3] = {Operand(facts)};
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_FINITE, operands),
            LOOM_DECISION_TRUTH_TRUE);
}

TEST(PredicateTest, FloatClassPredicatesRejectIntegerInterpretation) {
  const loom_decision_predicate_operand_t operands[3] = {
      Operand(loom_value_facts_exact_i64(1)),
  };
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_NOT_NAN, operands),
            LOOM_DECISION_TRUTH_UNKNOWN);
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_NOT_INF, operands),
            LOOM_DECISION_TRUTH_UNKNOWN);
  EXPECT_EQ(loom_decision_predicate_evaluate(LOOM_PREDICATE_FINITE, operands),
            LOOM_DECISION_TRUTH_UNKNOWN);
}

}  // namespace
