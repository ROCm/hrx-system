// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/facts.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/rewrite/rewriter.h"

static bool loom_sanitizer_find_value(loom_value_slice_t values,
                                      loom_value_id_t value_id,
                                      uint16_t* out_ordinal) {
  for (uint16_t i = 0; i < values.count; ++i) {
    if (values.values[i] != value_id) continue;
    if (out_ordinal) *out_ordinal = i;
    return true;
  }
  return false;
}

static bool loom_sanitizer_predicate_arg_facts(
    const loom_predicate_t* predicate, uint8_t argument_index,
    loom_rewriter_t* rewriter, loom_value_slice_t values,
    loom_value_facts_t* out_facts) {
  if (argument_index >= predicate->arg_count) return false;
  switch ((loom_predicate_arg_tag_t)predicate->arg_tags[argument_index]) {
    case LOOM_PRED_ARG_CONST:
      *out_facts = loom_value_facts_exact_i64(predicate->args[argument_index]);
      return true;
    case LOOM_PRED_ARG_VALUE: {
      if (predicate->args[argument_index] < 0) return false;
      const loom_value_id_t value_id =
          (loom_value_id_t)predicate->args[argument_index];
      if (!loom_sanitizer_find_value(values, value_id, NULL)) {
        return false;
      }
      *out_facts = loom_rewriter_value_facts(rewriter, value_id);
      return true;
    }
    case LOOM_PRED_ARG_NONE:
    case LOOM_PRED_ARG_COUNT_:
      return false;
  }
  return false;
}

static bool loom_sanitizer_ranges_are_disjoint(loom_value_facts_t lhs,
                                               loom_value_facts_t rhs) {
  return lhs.range_hi < rhs.range_lo || rhs.range_hi < lhs.range_lo;
}

static bool loom_sanitizer_predicate_is_proven(
    const loom_predicate_t* predicate, loom_rewriter_t* rewriter,
    loom_value_slice_t values) {
  if (predicate->arg_count == 0 ||
      predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE || predicate->args[0] < 0) {
    return false;
  }
  const loom_value_id_t target_value = (loom_value_id_t)predicate->args[0];
  if (!loom_sanitizer_find_value(values, target_value, NULL)) {
    return false;
  }
  loom_value_facts_t target_facts =
      loom_rewriter_value_facts(rewriter, target_value);

  loom_value_facts_t rhs_facts = {0};
  int64_t rhs_exact = 0;
  int64_t lower = 0;
  int64_t upper = 0;
  switch ((loom_predicate_kind_t)predicate->kind) {
    case LOOM_PREDICATE_EQ:
      if (loom_value_facts_is_float(target_facts)) return false;
      if (!loom_sanitizer_predicate_arg_facts(predicate, 1, rewriter, values,
                                              &rhs_facts) ||
          loom_value_facts_is_float(rhs_facts)) {
        return false;
      }
      return loom_value_facts_is_exact(target_facts) &&
             loom_value_facts_is_exact(rhs_facts) &&
             target_facts.range_lo == rhs_facts.range_lo;
    case LOOM_PREDICATE_NE:
      if (loom_value_facts_is_float(target_facts)) return false;
      if (!loom_sanitizer_predicate_arg_facts(predicate, 1, rewriter, values,
                                              &rhs_facts) ||
          loom_value_facts_is_float(rhs_facts)) {
        return false;
      }
      if (loom_value_facts_as_exact_i64(rhs_facts, &rhs_exact) &&
          rhs_exact == 0 && loom_value_facts_is_non_zero(target_facts)) {
        return true;
      }
      return loom_sanitizer_ranges_are_disjoint(target_facts, rhs_facts);
    case LOOM_PREDICATE_LT:
      if (loom_value_facts_is_float(target_facts)) return false;
      if (!loom_sanitizer_predicate_arg_facts(predicate, 1, rewriter, values,
                                              &rhs_facts) ||
          loom_value_facts_is_float(rhs_facts)) {
        return false;
      }
      return target_facts.range_hi < rhs_facts.range_lo;
    case LOOM_PREDICATE_LE:
      if (loom_value_facts_is_float(target_facts)) return false;
      if (!loom_sanitizer_predicate_arg_facts(predicate, 1, rewriter, values,
                                              &rhs_facts) ||
          loom_value_facts_is_float(rhs_facts)) {
        return false;
      }
      return target_facts.range_hi <= rhs_facts.range_lo;
    case LOOM_PREDICATE_GT:
      if (loom_value_facts_is_float(target_facts)) return false;
      if (!loom_sanitizer_predicate_arg_facts(predicate, 1, rewriter, values,
                                              &rhs_facts) ||
          loom_value_facts_is_float(rhs_facts)) {
        return false;
      }
      return target_facts.range_lo > rhs_facts.range_hi;
    case LOOM_PREDICATE_GE:
      if (loom_value_facts_is_float(target_facts)) return false;
      if (!loom_sanitizer_predicate_arg_facts(predicate, 1, rewriter, values,
                                              &rhs_facts) ||
          loom_value_facts_is_float(rhs_facts)) {
        return false;
      }
      return target_facts.range_lo >= rhs_facts.range_hi;
    case LOOM_PREDICATE_MUL:
      if (loom_value_facts_is_float(target_facts)) return false;
      if (!loom_sanitizer_predicate_arg_facts(predicate, 1, rewriter, values,
                                              &rhs_facts) ||
          !loom_value_facts_as_exact_i64(rhs_facts, &rhs_exact) ||
          rhs_exact == 0) {
        return false;
      }
      return loom_value_facts_divisible_by(target_facts, rhs_exact);
    case LOOM_PREDICATE_MIN:
      if (loom_value_facts_is_float(target_facts)) return false;
      if (!loom_sanitizer_predicate_arg_facts(predicate, 1, rewriter, values,
                                              &rhs_facts) ||
          loom_value_facts_is_float(rhs_facts)) {
        return false;
      }
      return target_facts.range_lo >= rhs_facts.range_hi;
    case LOOM_PREDICATE_MAX:
      if (loom_value_facts_is_float(target_facts)) return false;
      if (!loom_sanitizer_predicate_arg_facts(predicate, 1, rewriter, values,
                                              &rhs_facts) ||
          loom_value_facts_is_float(rhs_facts)) {
        return false;
      }
      return target_facts.range_hi <= rhs_facts.range_lo;
    case LOOM_PREDICATE_POW2:
      if (loom_value_facts_is_float(target_facts)) return false;
      return loom_value_facts_is_power_of_two(target_facts);
    case LOOM_PREDICATE_RANGE:
      if (loom_value_facts_is_float(target_facts)) return false;
      if (!loom_sanitizer_predicate_arg_facts(predicate, 1, rewriter, values,
                                              &rhs_facts) ||
          !loom_value_facts_as_exact_i64(rhs_facts, &lower) ||
          !loom_sanitizer_predicate_arg_facts(predicate, 2, rewriter, values,
                                              &rhs_facts) ||
          !loom_value_facts_as_exact_i64(rhs_facts, &upper)) {
        return false;
      }
      return target_facts.range_lo >= lower && target_facts.range_hi <= upper;
    case LOOM_PREDICATE_NOT_NAN:
      return loom_value_facts_is_not_nan(target_facts);
    case LOOM_PREDICATE_NOT_INF:
      return loom_value_facts_is_not_inf(target_facts);
    case LOOM_PREDICATE_FINITE:
      return loom_value_facts_is_finite(target_facts) ||
             (loom_value_facts_is_not_nan(target_facts) &&
              loom_value_facts_is_not_inf(target_facts));
    case LOOM_PREDICATE_COUNT_:
      return false;
  }
  return false;
}

static bool loom_sanitizer_predicate_list_is_proven(loom_attribute_t predicates,
                                                    loom_rewriter_t* rewriter,
                                                    loom_value_slice_t values) {
  if (predicates.kind != LOOM_ATTR_PREDICATE_LIST) return false;
  for (uint16_t i = 0; i < predicates.count; ++i) {
    if (!loom_sanitizer_predicate_is_proven(&predicates.predicate_list[i],
                                            rewriter, values)) {
      return false;
    }
  }
  return true;
}

iree_status_t loom_sanitizer_assert_value_canonicalize(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  loom_value_slice_t values = loom_sanitizer_assert_value_values(op);
  loom_value_slice_t results = loom_sanitizer_assert_value_results(op);
  if (values.count != results.count) return iree_ok_status();
  if (!loom_sanitizer_predicate_list_is_proven(
          loom_sanitizer_assert_value_predicates(op), rewriter, values)) {
    return iree_ok_status();
  }
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, values.values,
                                                  values.count);
}

iree_status_t loom_sanitizer_assert_op_canonicalize(loom_op_t* op,
                                                    loom_rewriter_t* rewriter) {
  if (!loom_sanitizer_predicate_list_is_proven(
          loom_sanitizer_assert_op_predicates(op), rewriter,
          loom_sanitizer_assert_op_values(op))) {
    return iree_ok_status();
  }
  return loom_rewriter_erase(rewriter, op);
}
