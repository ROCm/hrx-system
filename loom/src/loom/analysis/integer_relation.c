// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/integer_relation.h"

loom_symbolic_integer_relation_t loom_symbolic_integer_relation_invert(
    loom_symbolic_integer_relation_t relation) {
  switch (relation) {
    case LOOM_SYMBOLIC_INTEGER_RELATION_EQ:
      return LOOM_SYMBOLIC_INTEGER_RELATION_NE;
    case LOOM_SYMBOLIC_INTEGER_RELATION_NE:
      return LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
    case LOOM_SYMBOLIC_INTEGER_RELATION_LT:
      return LOOM_SYMBOLIC_INTEGER_RELATION_GE;
    case LOOM_SYMBOLIC_INTEGER_RELATION_LE:
      return LOOM_SYMBOLIC_INTEGER_RELATION_GT;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GT:
      return LOOM_SYMBOLIC_INTEGER_RELATION_LE;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GE:
      return LOOM_SYMBOLIC_INTEGER_RELATION_LT;
    default:
      return relation;
  }
}

loom_symbolic_integer_relation_t loom_symbolic_integer_relation_swap(
    loom_symbolic_integer_relation_t relation) {
  switch (relation) {
    case LOOM_SYMBOLIC_INTEGER_RELATION_LT:
      return LOOM_SYMBOLIC_INTEGER_RELATION_GT;
    case LOOM_SYMBOLIC_INTEGER_RELATION_LE:
      return LOOM_SYMBOLIC_INTEGER_RELATION_GE;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GT:
      return LOOM_SYMBOLIC_INTEGER_RELATION_LT;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GE:
      return LOOM_SYMBOLIC_INTEGER_RELATION_LE;
    case LOOM_SYMBOLIC_INTEGER_RELATION_EQ:
    case LOOM_SYMBOLIC_INTEGER_RELATION_NE:
    default:
      return relation;
  }
}

bool loom_symbolic_integer_relation_implies(
    loom_symbolic_integer_relation_t implied_relation,
    loom_symbolic_integer_relation_t queried_relation, bool* out_result) {
  if (implied_relation == queried_relation) {
    *out_result = true;
    return true;
  }
  if (loom_symbolic_integer_relation_invert(implied_relation) ==
      queried_relation) {
    *out_result = false;
    return true;
  }

  switch (implied_relation) {
    case LOOM_SYMBOLIC_INTEGER_RELATION_EQ:
      if (queried_relation == LOOM_SYMBOLIC_INTEGER_RELATION_LE ||
          queried_relation == LOOM_SYMBOLIC_INTEGER_RELATION_GE) {
        *out_result = true;
        return true;
      }
      return false;
    case LOOM_SYMBOLIC_INTEGER_RELATION_LT:
      if (queried_relation == LOOM_SYMBOLIC_INTEGER_RELATION_LE ||
          queried_relation == LOOM_SYMBOLIC_INTEGER_RELATION_NE) {
        *out_result = true;
        return true;
      }
      if (queried_relation == LOOM_SYMBOLIC_INTEGER_RELATION_EQ ||
          queried_relation == LOOM_SYMBOLIC_INTEGER_RELATION_GT) {
        *out_result = false;
        return true;
      }
      return false;
    case LOOM_SYMBOLIC_INTEGER_RELATION_LE:
      if (queried_relation == LOOM_SYMBOLIC_INTEGER_RELATION_GT) {
        *out_result = false;
        return true;
      }
      return false;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GT:
      if (queried_relation == LOOM_SYMBOLIC_INTEGER_RELATION_GE ||
          queried_relation == LOOM_SYMBOLIC_INTEGER_RELATION_NE) {
        *out_result = true;
        return true;
      }
      if (queried_relation == LOOM_SYMBOLIC_INTEGER_RELATION_EQ ||
          queried_relation == LOOM_SYMBOLIC_INTEGER_RELATION_LT) {
        *out_result = false;
        return true;
      }
      return false;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GE:
      if (queried_relation == LOOM_SYMBOLIC_INTEGER_RELATION_LT) {
        *out_result = false;
        return true;
      }
      return false;
    case LOOM_SYMBOLIC_INTEGER_RELATION_NE:
    default:
      return false;
  }
}
