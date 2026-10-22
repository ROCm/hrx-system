// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/index/compare.h"

#include "loom/ops/index/carrier.h"
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

bool loom_index_cmp_facts_fit_target_carrier(
    const loom_fact_context_t* context, loom_scalar_type_t operand_scalar_type,
    uint8_t predicate, const loom_value_facts_t* lhs_facts,
    const loom_value_facts_t* rhs_facts) {
  switch ((loom_index_cmp_predicate_t)predicate) {
    case LOOM_INDEX_CMP_PREDICATE_EQ:
    case LOOM_INDEX_CMP_PREDICATE_NE:
      return (loom_index_value_facts_fit_signed_target_carrier(
                  context, operand_scalar_type, *lhs_facts) &&
              loom_index_value_facts_fit_signed_target_carrier(
                  context, operand_scalar_type, *rhs_facts)) ||
             (loom_index_value_facts_fit_unsigned_target_carrier(
                  context, operand_scalar_type, *lhs_facts) &&
              loom_index_value_facts_fit_unsigned_target_carrier(
                  context, operand_scalar_type, *rhs_facts));
    case LOOM_INDEX_CMP_PREDICATE_SLT:
    case LOOM_INDEX_CMP_PREDICATE_SLE:
    case LOOM_INDEX_CMP_PREDICATE_SGT:
    case LOOM_INDEX_CMP_PREDICATE_SGE:
      return loom_index_value_facts_fit_signed_target_carrier(
                 context, operand_scalar_type, *lhs_facts) &&
             loom_index_value_facts_fit_signed_target_carrier(
                 context, operand_scalar_type, *rhs_facts);
    case LOOM_INDEX_CMP_PREDICATE_ULT:
    case LOOM_INDEX_CMP_PREDICATE_ULE:
    case LOOM_INDEX_CMP_PREDICATE_UGT:
    case LOOM_INDEX_CMP_PREDICATE_UGE:
      return loom_index_value_facts_fit_unsigned_target_carrier(
                 context, operand_scalar_type, *lhs_facts) &&
             loom_index_value_facts_fit_unsigned_target_carrier(
                 context, operand_scalar_type, *rhs_facts);
    default:
      return false;
  }
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

// Mathematical ranges outside the carrier cannot establish a particular bit
// pattern. Keep the full carrier domain so width-independent endpoint proofs
// remain available without inferring how an out-of-range source was narrowed.
static loom_value_facts_t loom_index_unsigned_comparison_facts(
    loom_value_facts_t facts, uint8_t bit_count) {
  if (loom_value_facts_fit_signed_bit_count(facts, bit_count) ||
      loom_value_facts_fit_unsigned_bit_count(facts, bit_count)) {
    return facts;
  }
  return loom_value_facts_make_signed_bit_count_range(bit_count);
}

bool loom_index_cmp_result_from_facts(const loom_fact_context_t* context,
                                      loom_scalar_type_t operand_scalar_type,
                                      uint8_t predicate,
                                      const loom_value_facts_t* lhs_facts,
                                      const loom_value_facts_t* rhs_facts,
                                      bool* out_result) {
  uint8_t predicate_kind;
  switch ((loom_index_cmp_predicate_t)predicate) {
    case LOOM_INDEX_CMP_PREDICATE_ULT:
      predicate_kind = LOOM_PREDICATE_ULT;
      break;
    case LOOM_INDEX_CMP_PREDICATE_ULE:
      predicate_kind = LOOM_PREDICATE_ULE;
      break;
    case LOOM_INDEX_CMP_PREDICATE_UGT:
      predicate_kind = LOOM_PREDICATE_UGT;
      break;
    case LOOM_INDEX_CMP_PREDICATE_UGE:
      predicate_kind = LOOM_PREDICATE_UGE;
      break;
    default:
      return loom_index_cmp_facts_fit_target_carrier(
                 context, operand_scalar_type, predicate, lhs_facts,
                 rhs_facts) &&
             loom_index_signed_cmp_facts_result(predicate, lhs_facts, rhs_facts,
                                                out_result);
  }
  int32_t bit_count =
      loom_index_target_carrier_bitwidth(context, operand_scalar_type);
  if (bit_count < 0 || loom_value_facts_is_float(*lhs_facts) ||
      loom_value_facts_is_float(*rhs_facts)) {
    return false;
  }
  loom_value_facts_t lhs = *lhs_facts;
  loom_value_facts_t rhs = *rhs_facts;
  if (bit_count == 0) {
    // Nonnegative source intervals retain mathematical ordering. Once signed
    // bit patterns participate, only zero and all-ones are invariant under
    // every carrier width; preserve those endpoints and widen other operands.
    // In particular, -1 and UINT32_MAX alias in 32 bits but differ in 64 bits.
    bit_count = 64;
    if (lhs.range_lo < 0 || rhs.range_lo < 0) {
      if (!loom_value_facts_is_exact(lhs) ||
          (lhs.range_lo != 0 && lhs.range_lo != -1)) {
        lhs = loom_value_facts_unknown();
      }
      if (!loom_value_facts_is_exact(rhs) ||
          (rhs.range_lo != 0 && rhs.range_lo != -1)) {
        rhs = loom_value_facts_unknown();
      }
    }
  } else {
    lhs = loom_index_unsigned_comparison_facts(lhs, (uint8_t)bit_count);
    rhs = loom_index_unsigned_comparison_facts(rhs, (uint8_t)bit_count);
  }
  return loom_value_facts_unsigned_comparison_result(
      predicate_kind, (uint8_t)bit_count, &lhs, &rhs, out_result);
}
