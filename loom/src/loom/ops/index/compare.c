// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/index/compare.h"

#include "loom/ops/index/ops.h"

bool loom_index_cmp_same_value_result(uint8_t predicate, bool* out_result) {
  switch ((loom_index_cmp_predicate_t)predicate) {
    case LOOM_INDEX_CMP_PREDICATE_EQ:
    case LOOM_INDEX_CMP_PREDICATE_SLE:
    case LOOM_INDEX_CMP_PREDICATE_SGE:
    case LOOM_INDEX_CMP_PREDICATE_ULE:
    case LOOM_INDEX_CMP_PREDICATE_UGE:
      *out_result = true;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_NE:
    case LOOM_INDEX_CMP_PREDICATE_SLT:
    case LOOM_INDEX_CMP_PREDICATE_SGT:
    case LOOM_INDEX_CMP_PREDICATE_ULT:
    case LOOM_INDEX_CMP_PREDICATE_UGT:
      *out_result = false;
      return true;
    default:
      return false;
  }
}

static bool loom_index_cmp_facts_are_non_overlapping(
    const loom_value_facts_t* lhs_facts, const loom_value_facts_t* rhs_facts) {
  return lhs_facts->range_hi < rhs_facts->range_lo ||
         rhs_facts->range_hi < lhs_facts->range_lo;
}

static bool loom_index_signed_cmp_facts_result(
    uint8_t predicate, const loom_value_facts_t* lhs_facts,
    const loom_value_facts_t* rhs_facts, bool* out_result) {
  if (loom_value_facts_is_float(*lhs_facts) ||
      loom_value_facts_is_float(*rhs_facts)) {
    return false;
  }
  switch ((loom_index_cmp_predicate_t)predicate) {
    case LOOM_INDEX_CMP_PREDICATE_EQ:
      if (loom_value_facts_is_exact(*lhs_facts) &&
          loom_value_facts_is_exact(*rhs_facts) &&
          lhs_facts->range_lo == rhs_facts->range_lo) {
        *out_result = true;
        return true;
      }
      if (loom_index_cmp_facts_are_non_overlapping(lhs_facts, rhs_facts)) {
        *out_result = false;
        return true;
      }
      return false;
    case LOOM_INDEX_CMP_PREDICATE_NE:
      if (loom_value_facts_is_exact(*lhs_facts) &&
          loom_value_facts_is_exact(*rhs_facts) &&
          lhs_facts->range_lo == rhs_facts->range_lo) {
        *out_result = false;
        return true;
      }
      if (loom_index_cmp_facts_are_non_overlapping(lhs_facts, rhs_facts)) {
        *out_result = true;
        return true;
      }
      return false;
    case LOOM_INDEX_CMP_PREDICATE_SLT:
      if (lhs_facts->range_hi < rhs_facts->range_lo) {
        *out_result = true;
        return true;
      }
      if (lhs_facts->range_lo >= rhs_facts->range_hi) {
        *out_result = false;
        return true;
      }
      return false;
    case LOOM_INDEX_CMP_PREDICATE_SLE:
      if (lhs_facts->range_hi <= rhs_facts->range_lo) {
        *out_result = true;
        return true;
      }
      if (lhs_facts->range_lo > rhs_facts->range_hi) {
        *out_result = false;
        return true;
      }
      return false;
    case LOOM_INDEX_CMP_PREDICATE_SGT:
      if (lhs_facts->range_lo > rhs_facts->range_hi) {
        *out_result = true;
        return true;
      }
      if (lhs_facts->range_hi <= rhs_facts->range_lo) {
        *out_result = false;
        return true;
      }
      return false;
    case LOOM_INDEX_CMP_PREDICATE_SGE:
      if (lhs_facts->range_lo >= rhs_facts->range_hi) {
        *out_result = true;
        return true;
      }
      if (lhs_facts->range_hi < rhs_facts->range_lo) {
        *out_result = false;
        return true;
      }
      return false;
    default:
      return false;
  }
}

static bool loom_index_unsigned_cmp_facts_result(
    uint8_t predicate, const loom_value_facts_t* lhs_facts,
    const loom_value_facts_t* rhs_facts, bool* out_result) {
  if (!loom_value_facts_is_non_negative(*lhs_facts) ||
      !loom_value_facts_is_non_negative(*rhs_facts)) {
    return false;
  }
  switch ((loom_index_cmp_predicate_t)predicate) {
    case LOOM_INDEX_CMP_PREDICATE_ULT:
      return loom_index_signed_cmp_facts_result(
          LOOM_INDEX_CMP_PREDICATE_SLT, lhs_facts, rhs_facts, out_result);
    case LOOM_INDEX_CMP_PREDICATE_ULE:
      return loom_index_signed_cmp_facts_result(
          LOOM_INDEX_CMP_PREDICATE_SLE, lhs_facts, rhs_facts, out_result);
    case LOOM_INDEX_CMP_PREDICATE_UGT:
      return loom_index_signed_cmp_facts_result(
          LOOM_INDEX_CMP_PREDICATE_SGT, lhs_facts, rhs_facts, out_result);
    case LOOM_INDEX_CMP_PREDICATE_UGE:
      return loom_index_signed_cmp_facts_result(
          LOOM_INDEX_CMP_PREDICATE_SGE, lhs_facts, rhs_facts, out_result);
    default:
      return false;
  }
}

bool loom_index_cmp_result_from_facts(uint8_t predicate,
                                      const loom_value_facts_t* lhs_facts,
                                      const loom_value_facts_t* rhs_facts,
                                      bool* out_result) {
  return loom_index_signed_cmp_facts_result(predicate, lhs_facts, rhs_facts,
                                            out_result) ||
         loom_index_unsigned_cmp_facts_result(predicate, lhs_facts, rhs_facts,
                                              out_result);
}
