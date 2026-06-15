// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/scalar/compare.h"

#include <math.h>

#include "loom/ops/scalar/ops.h"

bool loom_scalar_cmpi_same_value_result(uint8_t predicate, bool* out_result) {
  switch ((loom_scalar_cmpi_predicate_t)predicate) {
    case LOOM_SCALAR_CMPI_PREDICATE_EQ:
    case LOOM_SCALAR_CMPI_PREDICATE_SLE:
    case LOOM_SCALAR_CMPI_PREDICATE_SGE:
    case LOOM_SCALAR_CMPI_PREDICATE_ULE:
    case LOOM_SCALAR_CMPI_PREDICATE_UGE:
      *out_result = true;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_NE:
    case LOOM_SCALAR_CMPI_PREDICATE_SLT:
    case LOOM_SCALAR_CMPI_PREDICATE_SGT:
    case LOOM_SCALAR_CMPI_PREDICATE_ULT:
    case LOOM_SCALAR_CMPI_PREDICATE_UGT:
      *out_result = false;
      return true;
    default:
      return false;
  }
}

uint8_t loom_scalar_cmpi_swapped_predicate(uint8_t predicate) {
  switch ((loom_scalar_cmpi_predicate_t)predicate) {
    case LOOM_SCALAR_CMPI_PREDICATE_SLT:
      return LOOM_SCALAR_CMPI_PREDICATE_SGT;
    case LOOM_SCALAR_CMPI_PREDICATE_SLE:
      return LOOM_SCALAR_CMPI_PREDICATE_SGE;
    case LOOM_SCALAR_CMPI_PREDICATE_SGT:
      return LOOM_SCALAR_CMPI_PREDICATE_SLT;
    case LOOM_SCALAR_CMPI_PREDICATE_SGE:
      return LOOM_SCALAR_CMPI_PREDICATE_SLE;
    case LOOM_SCALAR_CMPI_PREDICATE_ULT:
      return LOOM_SCALAR_CMPI_PREDICATE_UGT;
    case LOOM_SCALAR_CMPI_PREDICATE_ULE:
      return LOOM_SCALAR_CMPI_PREDICATE_UGE;
    case LOOM_SCALAR_CMPI_PREDICATE_UGT:
      return LOOM_SCALAR_CMPI_PREDICATE_ULT;
    case LOOM_SCALAR_CMPI_PREDICATE_UGE:
      return LOOM_SCALAR_CMPI_PREDICATE_ULE;
    case LOOM_SCALAR_CMPI_PREDICATE_EQ:
    case LOOM_SCALAR_CMPI_PREDICATE_NE:
    default:
      return predicate;
  }
}

static bool loom_scalar_cmpi_facts_are_non_overlapping(
    const loom_value_facts_t* lhs_facts, const loom_value_facts_t* rhs_facts) {
  return lhs_facts->range_hi < rhs_facts->range_lo ||
         rhs_facts->range_hi < lhs_facts->range_lo;
}

static bool loom_scalar_signed_cmpi_facts_result(
    uint8_t predicate, const loom_value_facts_t* lhs_facts,
    const loom_value_facts_t* rhs_facts, bool* out_result) {
  if (loom_value_facts_is_float(*lhs_facts) ||
      loom_value_facts_is_float(*rhs_facts)) {
    return false;
  }
  switch ((loom_scalar_cmpi_predicate_t)predicate) {
    case LOOM_SCALAR_CMPI_PREDICATE_EQ:
      if (loom_value_facts_is_exact(*lhs_facts) &&
          loom_value_facts_is_exact(*rhs_facts) &&
          lhs_facts->range_lo == rhs_facts->range_lo) {
        *out_result = true;
        return true;
      }
      if (loom_scalar_cmpi_facts_are_non_overlapping(lhs_facts, rhs_facts)) {
        *out_result = false;
        return true;
      }
      return false;
    case LOOM_SCALAR_CMPI_PREDICATE_NE:
      if (loom_value_facts_is_exact(*lhs_facts) &&
          loom_value_facts_is_exact(*rhs_facts) &&
          lhs_facts->range_lo == rhs_facts->range_lo) {
        *out_result = false;
        return true;
      }
      if (loom_scalar_cmpi_facts_are_non_overlapping(lhs_facts, rhs_facts)) {
        *out_result = true;
        return true;
      }
      return false;
    case LOOM_SCALAR_CMPI_PREDICATE_SLT:
      if (lhs_facts->range_hi < rhs_facts->range_lo) {
        *out_result = true;
        return true;
      }
      if (lhs_facts->range_lo >= rhs_facts->range_hi) {
        *out_result = false;
        return true;
      }
      return false;
    case LOOM_SCALAR_CMPI_PREDICATE_SLE:
      if (lhs_facts->range_hi <= rhs_facts->range_lo) {
        *out_result = true;
        return true;
      }
      if (lhs_facts->range_lo > rhs_facts->range_hi) {
        *out_result = false;
        return true;
      }
      return false;
    case LOOM_SCALAR_CMPI_PREDICATE_SGT:
      if (lhs_facts->range_lo > rhs_facts->range_hi) {
        *out_result = true;
        return true;
      }
      if (lhs_facts->range_hi <= rhs_facts->range_lo) {
        *out_result = false;
        return true;
      }
      return false;
    case LOOM_SCALAR_CMPI_PREDICATE_SGE:
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

static bool loom_scalar_unsigned_cmpi_facts_result(
    uint8_t predicate, const loom_value_facts_t* lhs_facts,
    const loom_value_facts_t* rhs_facts, bool* out_result) {
  if (!loom_value_facts_is_non_negative(*lhs_facts) ||
      !loom_value_facts_is_non_negative(*rhs_facts)) {
    return false;
  }
  switch ((loom_scalar_cmpi_predicate_t)predicate) {
    case LOOM_SCALAR_CMPI_PREDICATE_ULT:
      return loom_scalar_signed_cmpi_facts_result(
          LOOM_SCALAR_CMPI_PREDICATE_SLT, lhs_facts, rhs_facts, out_result);
    case LOOM_SCALAR_CMPI_PREDICATE_ULE:
      return loom_scalar_signed_cmpi_facts_result(
          LOOM_SCALAR_CMPI_PREDICATE_SLE, lhs_facts, rhs_facts, out_result);
    case LOOM_SCALAR_CMPI_PREDICATE_UGT:
      return loom_scalar_signed_cmpi_facts_result(
          LOOM_SCALAR_CMPI_PREDICATE_SGT, lhs_facts, rhs_facts, out_result);
    case LOOM_SCALAR_CMPI_PREDICATE_UGE:
      return loom_scalar_signed_cmpi_facts_result(
          LOOM_SCALAR_CMPI_PREDICATE_SGE, lhs_facts, rhs_facts, out_result);
    default:
      return false;
  }
}

bool loom_scalar_cmpi_result_from_facts(uint8_t predicate,
                                        const loom_value_facts_t* lhs_facts,
                                        const loom_value_facts_t* rhs_facts,
                                        bool* out_result) {
  return loom_scalar_signed_cmpi_facts_result(predicate, lhs_facts, rhs_facts,
                                              out_result) ||
         loom_scalar_unsigned_cmpi_facts_result(predicate, lhs_facts, rhs_facts,
                                                out_result);
}

bool loom_scalar_cmpf_same_value_result(uint8_t predicate, bool* out_result) {
  switch ((loom_scalar_cmpf_predicate_t)predicate) {
    case LOOM_SCALAR_CMPF_PREDICATE_OEQ:
    case LOOM_SCALAR_CMPF_PREDICATE_OGE:
    case LOOM_SCALAR_CMPF_PREDICATE_OLE:
    case LOOM_SCALAR_CMPF_PREDICATE_ORD:
    case LOOM_SCALAR_CMPF_PREDICATE_UEQ:
    case LOOM_SCALAR_CMPF_PREDICATE_UGE:
    case LOOM_SCALAR_CMPF_PREDICATE_ULE:
      *out_result = true;
      return true;
    case LOOM_SCALAR_CMPF_PREDICATE_OGT:
    case LOOM_SCALAR_CMPF_PREDICATE_OLT:
    case LOOM_SCALAR_CMPF_PREDICATE_ONE:
    case LOOM_SCALAR_CMPF_PREDICATE_UGT:
    case LOOM_SCALAR_CMPF_PREDICATE_ULT:
    case LOOM_SCALAR_CMPF_PREDICATE_UNE:
    case LOOM_SCALAR_CMPF_PREDICATE_UNO:
      *out_result = false;
      return true;
    default:
      return false;
  }
}

bool loom_scalar_cmpf_exact_result(uint8_t predicate, double lhs, double rhs,
                                   bool* out_result) {
  bool ordered = !isnan(lhs) && !isnan(rhs);
  switch ((loom_scalar_cmpf_predicate_t)predicate) {
    case LOOM_SCALAR_CMPF_PREDICATE_OEQ:
      *out_result = ordered && lhs == rhs;
      return true;
    case LOOM_SCALAR_CMPF_PREDICATE_OGT:
      *out_result = ordered && lhs > rhs;
      return true;
    case LOOM_SCALAR_CMPF_PREDICATE_OGE:
      *out_result = ordered && lhs >= rhs;
      return true;
    case LOOM_SCALAR_CMPF_PREDICATE_OLT:
      *out_result = ordered && lhs < rhs;
      return true;
    case LOOM_SCALAR_CMPF_PREDICATE_OLE:
      *out_result = ordered && lhs <= rhs;
      return true;
    case LOOM_SCALAR_CMPF_PREDICATE_ONE:
      *out_result = ordered && lhs != rhs;
      return true;
    case LOOM_SCALAR_CMPF_PREDICATE_ORD:
      *out_result = ordered;
      return true;
    case LOOM_SCALAR_CMPF_PREDICATE_UEQ:
      *out_result = !ordered || lhs == rhs;
      return true;
    case LOOM_SCALAR_CMPF_PREDICATE_UGT:
      *out_result = !ordered || lhs > rhs;
      return true;
    case LOOM_SCALAR_CMPF_PREDICATE_UGE:
      *out_result = !ordered || lhs >= rhs;
      return true;
    case LOOM_SCALAR_CMPF_PREDICATE_ULT:
      *out_result = !ordered || lhs < rhs;
      return true;
    case LOOM_SCALAR_CMPF_PREDICATE_ULE:
      *out_result = !ordered || lhs <= rhs;
      return true;
    case LOOM_SCALAR_CMPF_PREDICATE_UNE:
      *out_result = !ordered || lhs != rhs;
      return true;
    case LOOM_SCALAR_CMPF_PREDICATE_UNO:
      *out_result = !ordered;
      return true;
    default:
      return false;
  }
}
