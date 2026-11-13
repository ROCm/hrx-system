// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/decision/predicate.h"

static loom_decision_truth_t loom_decision_truth_from_bool(bool value) {
  return value ? LOOM_DECISION_TRUTH_TRUE : LOOM_DECISION_TRUTH_FALSE;
}

static bool loom_decision_operands_have_same_identity(
    const loom_decision_predicate_operand_t* lhs,
    const loom_decision_predicate_operand_t* rhs) {
  return lhs->identity != LOOM_DECISION_OPERAND_IDENTITY_NONE &&
         lhs->identity == rhs->identity;
}

static bool loom_decision_facts_are_exact_zero(loom_value_facts_t facts) {
  int64_t value = 0;
  return loom_value_facts_as_exact_i64(facts, &value) && value == 0;
}

// Converts one sign-stable fact interval into monotonically ordered unsigned
// carrier payloads. A signed interval crossing zero represents two disjoint
// unsigned intervals and cannot be bounded without losing information.
static bool loom_decision_predicate_unsigned_bounds(loom_value_facts_t facts,
                                                    uint64_t* out_lower,
                                                    uint64_t* out_upper) {
  if (loom_value_facts_is_float(facts) ||
      (facts.range_lo < 0 && facts.range_hi >= 0)) {
    return false;
  }
  *out_lower = (uint64_t)facts.range_lo;
  *out_upper = (uint64_t)facts.range_hi;
  return true;
}

static loom_decision_truth_t loom_decision_predicate_evaluate_unsigned_relation(
    loom_predicate_kind_t predicate_kind,
    const loom_decision_predicate_operand_t* lhs,
    const loom_decision_predicate_operand_t* rhs) {
  if (loom_decision_operands_have_same_identity(lhs, rhs)) {
    switch (predicate_kind) {
      case LOOM_PREDICATE_ULE:
      case LOOM_PREDICATE_UGE:
        return LOOM_DECISION_TRUTH_TRUE;
      case LOOM_PREDICATE_ULT:
      case LOOM_PREDICATE_UGT:
        return LOOM_DECISION_TRUTH_FALSE;
      default:
        IREE_ASSERT_UNREACHABLE("expected an unsigned relation predicate");
        IREE_BUILTIN_UNREACHABLE();
    }
  }

  uint64_t lhs_lower = 0;
  uint64_t lhs_upper = 0;
  uint64_t rhs_lower = 0;
  uint64_t rhs_upper = 0;
  if (!loom_decision_predicate_unsigned_bounds(lhs->facts, &lhs_lower,
                                               &lhs_upper) ||
      !loom_decision_predicate_unsigned_bounds(rhs->facts, &rhs_lower,
                                               &rhs_upper)) {
    return LOOM_DECISION_TRUTH_UNKNOWN;
  }
  switch (predicate_kind) {
    case LOOM_PREDICATE_ULT:
      if (lhs_upper < rhs_lower) {
        return LOOM_DECISION_TRUTH_TRUE;
      }
      if (lhs_lower >= rhs_upper) {
        return LOOM_DECISION_TRUTH_FALSE;
      }
      return LOOM_DECISION_TRUTH_UNKNOWN;
    case LOOM_PREDICATE_ULE:
      if (lhs_upper <= rhs_lower) {
        return LOOM_DECISION_TRUTH_TRUE;
      }
      if (lhs_lower > rhs_upper) {
        return LOOM_DECISION_TRUTH_FALSE;
      }
      return LOOM_DECISION_TRUTH_UNKNOWN;
    case LOOM_PREDICATE_UGT:
      if (lhs_lower > rhs_upper) {
        return LOOM_DECISION_TRUTH_TRUE;
      }
      if (lhs_upper <= rhs_lower) {
        return LOOM_DECISION_TRUTH_FALSE;
      }
      return LOOM_DECISION_TRUTH_UNKNOWN;
    case LOOM_PREDICATE_UGE:
      if (lhs_lower >= rhs_upper) {
        return LOOM_DECISION_TRUTH_TRUE;
      }
      if (lhs_upper < rhs_lower) {
        return LOOM_DECISION_TRUTH_FALSE;
      }
      return LOOM_DECISION_TRUTH_UNKNOWN;
    default:
      IREE_ASSERT_UNREACHABLE("expected an unsigned relation predicate");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static IREE_ATTRIBUTE_ALWAYS_INLINE loom_decision_truth_t
loom_decision_predicate_evaluate_relation(
    loom_predicate_kind_t predicate_kind,
    const loom_decision_predicate_operand_t* lhs,
    const loom_decision_predicate_operand_t* rhs) {
  if (loom_value_facts_is_float(lhs->facts) ||
      loom_value_facts_is_float(rhs->facts)) {
    return LOOM_DECISION_TRUTH_UNKNOWN;
  }

  if (loom_decision_operands_have_same_identity(lhs, rhs)) {
    switch (predicate_kind) {
      case LOOM_PREDICATE_EQ:
      case LOOM_PREDICATE_LE:
      case LOOM_PREDICATE_GE:
        return LOOM_DECISION_TRUTH_TRUE;
      case LOOM_PREDICATE_NE:
      case LOOM_PREDICATE_LT:
      case LOOM_PREDICATE_GT:
        return LOOM_DECISION_TRUTH_FALSE;
      default:
        IREE_ASSERT_UNREACHABLE("expected an integer relation predicate");
        IREE_BUILTIN_UNREACHABLE();
    }
  }

  switch (predicate_kind) {
    case LOOM_PREDICATE_EQ:
      if ((loom_value_facts_is_non_zero(lhs->facts) &&
           loom_decision_facts_are_exact_zero(rhs->facts)) ||
          (loom_value_facts_is_non_zero(rhs->facts) &&
           loom_decision_facts_are_exact_zero(lhs->facts))) {
        return LOOM_DECISION_TRUTH_FALSE;
      }
      if (lhs->facts.range_lo == lhs->facts.range_hi &&
          rhs->facts.range_lo == rhs->facts.range_hi &&
          lhs->facts.range_lo == rhs->facts.range_lo) {
        return LOOM_DECISION_TRUTH_TRUE;
      }
      if (lhs->facts.range_hi < rhs->facts.range_lo ||
          rhs->facts.range_hi < lhs->facts.range_lo) {
        return LOOM_DECISION_TRUTH_FALSE;
      }
      return LOOM_DECISION_TRUTH_UNKNOWN;
    case LOOM_PREDICATE_NE:
      if ((loom_value_facts_is_non_zero(lhs->facts) &&
           loom_decision_facts_are_exact_zero(rhs->facts)) ||
          (loom_value_facts_is_non_zero(rhs->facts) &&
           loom_decision_facts_are_exact_zero(lhs->facts))) {
        return LOOM_DECISION_TRUTH_TRUE;
      }
      if (lhs->facts.range_hi < rhs->facts.range_lo ||
          rhs->facts.range_hi < lhs->facts.range_lo) {
        return LOOM_DECISION_TRUTH_TRUE;
      }
      if (lhs->facts.range_lo == lhs->facts.range_hi &&
          rhs->facts.range_lo == rhs->facts.range_hi &&
          lhs->facts.range_lo == rhs->facts.range_lo) {
        return LOOM_DECISION_TRUTH_FALSE;
      }
      return LOOM_DECISION_TRUTH_UNKNOWN;
    case LOOM_PREDICATE_LT:
      if (lhs->facts.range_hi < rhs->facts.range_lo) {
        return LOOM_DECISION_TRUTH_TRUE;
      }
      if (lhs->facts.range_lo >= rhs->facts.range_hi) {
        return LOOM_DECISION_TRUTH_FALSE;
      }
      return LOOM_DECISION_TRUTH_UNKNOWN;
    case LOOM_PREDICATE_LE:
      if (lhs->facts.range_hi <= rhs->facts.range_lo) {
        return LOOM_DECISION_TRUTH_TRUE;
      }
      if (lhs->facts.range_lo > rhs->facts.range_hi) {
        return LOOM_DECISION_TRUTH_FALSE;
      }
      return LOOM_DECISION_TRUTH_UNKNOWN;
    case LOOM_PREDICATE_GT:
      if (lhs->facts.range_lo > rhs->facts.range_hi) {
        return LOOM_DECISION_TRUTH_TRUE;
      }
      if (lhs->facts.range_hi <= rhs->facts.range_lo) {
        return LOOM_DECISION_TRUTH_FALSE;
      }
      return LOOM_DECISION_TRUTH_UNKNOWN;
    case LOOM_PREDICATE_GE:
      if (lhs->facts.range_lo >= rhs->facts.range_hi) {
        return LOOM_DECISION_TRUTH_TRUE;
      }
      if (lhs->facts.range_hi < rhs->facts.range_lo) {
        return LOOM_DECISION_TRUTH_FALSE;
      }
      return LOOM_DECISION_TRUTH_UNKNOWN;
    default:
      IREE_ASSERT_UNREACHABLE("expected an integer relation predicate");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static loom_decision_truth_t loom_decision_predicate_evaluate_multiple(
    const loom_decision_predicate_operand_t* value,
    const loom_decision_predicate_operand_t* divisor) {
  int64_t exact_divisor = 0;
  if (loom_value_facts_is_float(value->facts) ||
      !loom_value_facts_as_exact_i64(divisor->facts, &exact_divisor) ||
      exact_divisor <= 0) {
    return LOOM_DECISION_TRUTH_UNKNOWN;
  }

  int64_t exact_value = 0;
  if (loom_value_facts_as_exact_i64(value->facts, &exact_value)) {
    return loom_decision_truth_from_bool(exact_value % exact_divisor == 0);
  }
  if (loom_value_facts_divisible_by(value->facts, exact_divisor)) {
    return LOOM_DECISION_TRUTH_TRUE;
  }
  if ((value->facts.range_lo > 0 && value->facts.range_hi < exact_divisor) ||
      (value->facts.range_hi < 0 && value->facts.range_lo > -exact_divisor)) {
    return LOOM_DECISION_TRUTH_FALSE;
  }
  return LOOM_DECISION_TRUTH_UNKNOWN;
}

static loom_decision_truth_t loom_decision_predicate_evaluate_power_of_two(
    const loom_decision_predicate_operand_t* value) {
  if (loom_value_facts_is_float(value->facts)) {
    return LOOM_DECISION_TRUTH_UNKNOWN;
  }
  if (loom_value_facts_is_power_of_two(value->facts)) {
    return LOOM_DECISION_TRUTH_TRUE;
  }
  if (loom_value_facts_is_exact(value->facts) || value->facts.range_hi < 1) {
    return LOOM_DECISION_TRUTH_FALSE;
  }
  return LOOM_DECISION_TRUTH_UNKNOWN;
}

static loom_decision_truth_t loom_decision_predicate_evaluate_range(
    const loom_decision_predicate_operand_t* value,
    const loom_decision_predicate_operand_t* lower,
    const loom_decision_predicate_operand_t* upper) {
  int64_t lower_bound = 0;
  int64_t upper_bound = 0;
  if (loom_value_facts_is_float(value->facts) ||
      !loom_value_facts_as_exact_i64(lower->facts, &lower_bound) ||
      !loom_value_facts_as_exact_i64(upper->facts, &upper_bound)) {
    return LOOM_DECISION_TRUTH_UNKNOWN;
  }
  if (lower_bound > upper_bound) {
    return LOOM_DECISION_TRUTH_FALSE;
  }
  if (value->facts.range_lo >= lower_bound &&
      value->facts.range_hi <= upper_bound) {
    return LOOM_DECISION_TRUTH_TRUE;
  }
  if (value->facts.range_hi < lower_bound ||
      value->facts.range_lo > upper_bound) {
    return LOOM_DECISION_TRUTH_FALSE;
  }
  return LOOM_DECISION_TRUTH_UNKNOWN;
}

static loom_decision_truth_t loom_decision_predicate_evaluate_not_nan(
    const loom_decision_predicate_operand_t* value) {
  if (loom_value_facts_is_not_nan(value->facts)) {
    return LOOM_DECISION_TRUTH_TRUE;
  }
  if (loom_value_facts_is_nan(value->facts)) {
    return LOOM_DECISION_TRUTH_FALSE;
  }
  return LOOM_DECISION_TRUTH_UNKNOWN;
}

static loom_decision_truth_t loom_decision_predicate_evaluate_not_inf(
    const loom_decision_predicate_operand_t* value) {
  if (loom_value_facts_is_not_inf(value->facts)) {
    return LOOM_DECISION_TRUTH_TRUE;
  }
  if (loom_value_facts_is_inf(value->facts)) {
    return LOOM_DECISION_TRUTH_FALSE;
  }
  return LOOM_DECISION_TRUTH_UNKNOWN;
}

static loom_decision_truth_t loom_decision_predicate_evaluate_finite(
    const loom_decision_predicate_operand_t* value) {
  if (loom_value_facts_is_finite(value->facts) ||
      (loom_value_facts_is_not_nan(value->facts) &&
       loom_value_facts_is_not_inf(value->facts))) {
    return LOOM_DECISION_TRUTH_TRUE;
  }
  if (loom_value_facts_is_nan(value->facts) ||
      loom_value_facts_is_inf(value->facts)) {
    return LOOM_DECISION_TRUTH_FALSE;
  }
  return LOOM_DECISION_TRUTH_UNKNOWN;
}

extern IREE_ATTRIBUTE_ALWAYS_INLINE loom_decision_truth_t
loom_decision_predicate_evaluate(
    loom_predicate_kind_t predicate_kind,
    const loom_decision_predicate_operand_t operands[3]) {
  switch (predicate_kind) {
    case LOOM_PREDICATE_EQ:
    case LOOM_PREDICATE_NE:
    case LOOM_PREDICATE_LT:
    case LOOM_PREDICATE_LE:
    case LOOM_PREDICATE_GT:
    case LOOM_PREDICATE_GE:
      return loom_decision_predicate_evaluate_relation(
          predicate_kind, &operands[0], &operands[1]);
    case LOOM_PREDICATE_ULT:
    case LOOM_PREDICATE_ULE:
    case LOOM_PREDICATE_UGT:
    case LOOM_PREDICATE_UGE:
      return loom_decision_predicate_evaluate_unsigned_relation(
          predicate_kind, &operands[0], &operands[1]);
    case LOOM_PREDICATE_MULTIPLE_OF:
      return loom_decision_predicate_evaluate_multiple(&operands[0],
                                                       &operands[1]);
    case LOOM_PREDICATE_POWER_OF_TWO:
      return loom_decision_predicate_evaluate_power_of_two(&operands[0]);
    case LOOM_PREDICATE_RANGE:
      return loom_decision_predicate_evaluate_range(&operands[0], &operands[1],
                                                    &operands[2]);
    case LOOM_PREDICATE_NOT_NAN:
      return loom_decision_predicate_evaluate_not_nan(&operands[0]);
    case LOOM_PREDICATE_NOT_INF:
      return loom_decision_predicate_evaluate_not_inf(&operands[0]);
    case LOOM_PREDICATE_FINITE:
      return loom_decision_predicate_evaluate_finite(&operands[0]);
    case LOOM_PREDICATE_COUNT_:
      IREE_ASSERT_UNREACHABLE("invalid decision predicate kind");
      IREE_BUILTIN_UNREACHABLE();
  }
  IREE_ASSERT_UNREACHABLE("invalid decision predicate kind");
  IREE_BUILTIN_UNREACHABLE();
}
