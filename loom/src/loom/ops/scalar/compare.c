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

uint8_t loom_scalar_cmpi_range_predicate(loom_scalar_type_t type,
                                         uint8_t predicate) {
  if (type == LOOM_SCALAR_TYPE_I1 &&
      predicate >= LOOM_SCALAR_CMPI_PREDICATE_SLT &&
      predicate <= LOOM_SCALAR_CMPI_PREDICATE_SGE) {
    return loom_scalar_cmpi_swapped_predicate(predicate);
  }
  return predicate;
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
    loom_scalar_type_t type, uint8_t predicate,
    const loom_value_facts_t* lhs_facts, const loom_value_facts_t* rhs_facts,
    bool* out_result) {
  uint8_t predicate_kind;
  switch ((loom_scalar_cmpi_predicate_t)predicate) {
    case LOOM_SCALAR_CMPI_PREDICATE_ULT:
      predicate_kind = LOOM_PREDICATE_ULT;
      break;
    case LOOM_SCALAR_CMPI_PREDICATE_ULE:
      predicate_kind = LOOM_PREDICATE_ULE;
      break;
    case LOOM_SCALAR_CMPI_PREDICATE_UGT:
      predicate_kind = LOOM_PREDICATE_UGT;
      break;
    case LOOM_SCALAR_CMPI_PREDICATE_UGE:
      predicate_kind = LOOM_PREDICATE_UGE;
      break;
    default:
      return false;
  }
  return loom_value_facts_unsigned_comparison_result(
      predicate_kind, (uint8_t)loom_scalar_type_bitwidth(type), lhs_facts,
      rhs_facts, out_result);
}

bool loom_scalar_cmpi_result_from_facts(loom_scalar_type_t type,
                                        uint8_t predicate,
                                        const loom_value_facts_t* lhs_facts,
                                        const loom_value_facts_t* rhs_facts,
                                        bool* out_result) {
  predicate = loom_scalar_cmpi_range_predicate(type, predicate);
  return loom_scalar_signed_cmpi_facts_result(predicate, lhs_facts, rhs_facts,
                                              out_result) ||
         loom_scalar_unsigned_cmpi_facts_result(type, predicate, lhs_facts,
                                                rhs_facts, out_result);
}

bool loom_scalar_cmpf_constant_result(uint8_t predicate, loom_value_id_t lhs,
                                      loom_value_id_t rhs, uint8_t fastmath,
                                      bool* out_result) {
  bool no_nan = (fastmath & LOOM_SCALAR_FASTMATHFLAGS_NNAN) != 0;
  if (no_nan && (predicate == LOOM_SCALAR_CMPF_PREDICATE_ORD ||
                 predicate == LOOM_SCALAR_CMPF_PREDICATE_UNO)) {
    *out_result = predicate == LOOM_SCALAR_CMPF_PREDICATE_ORD;
    return true;
  }
  if (lhs != rhs) {
    return false;
  }
  switch ((loom_scalar_cmpf_predicate_t)predicate) {
    case LOOM_SCALAR_CMPF_PREDICATE_OEQ:
    case LOOM_SCALAR_CMPF_PREDICATE_OGE:
    case LOOM_SCALAR_CMPF_PREDICATE_OLE:
    case LOOM_SCALAR_CMPF_PREDICATE_ORD:
      if (!no_nan) {
        return false;
      }
      *out_result = true;
      return true;
    case LOOM_SCALAR_CMPF_PREDICATE_UEQ:
    case LOOM_SCALAR_CMPF_PREDICATE_UGE:
    case LOOM_SCALAR_CMPF_PREDICATE_ULE:
      *out_result = true;
      return true;
    case LOOM_SCALAR_CMPF_PREDICATE_OGT:
    case LOOM_SCALAR_CMPF_PREDICATE_OLT:
    case LOOM_SCALAR_CMPF_PREDICATE_ONE:
      *out_result = false;
      return true;
    case LOOM_SCALAR_CMPF_PREDICATE_UGT:
    case LOOM_SCALAR_CMPF_PREDICATE_ULT:
    case LOOM_SCALAR_CMPF_PREDICATE_UNE:
    case LOOM_SCALAR_CMPF_PREDICATE_UNO:
      if (!no_nan) {
        return false;
      }
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
