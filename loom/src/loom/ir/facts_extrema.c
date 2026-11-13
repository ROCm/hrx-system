// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/math.h"
#include "loom/ir/facts.h"

//===----------------------------------------------------------------------===//
// Transfer functions: min / max
//===----------------------------------------------------------------------===//

static void loom_value_facts_minimum_range(const loom_value_facts_t* lhs,
                                           const loom_value_facts_t* rhs,
                                           loom_value_facts_t* out) {
  *out = loom_value_facts_make(iree_min(lhs->range_lo, rhs->range_lo),
                               iree_min(lhs->range_hi, rhs->range_hi),
                               loom_value_facts_common_divisor(lhs, rhs));
  out->flags |= loom_value_facts_common_integer_predicates(lhs, rhs);
  loom_value_facts_propagate_binary_distribution(*lhs, *rhs, out);
}

static void loom_value_facts_maximum_range(const loom_value_facts_t* lhs,
                                           const loom_value_facts_t* rhs,
                                           loom_value_facts_t* out) {
  *out = loom_value_facts_make(iree_max(lhs->range_lo, rhs->range_lo),
                               iree_max(lhs->range_hi, rhs->range_hi),
                               loom_value_facts_common_divisor(lhs, rhs));
  out->flags |= loom_value_facts_common_integer_predicates(lhs, rhs);
  loom_value_facts_propagate_binary_distribution(*lhs, *rhs, out);
}

void loom_value_facts_minsi(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs, int32_t bit_count,
                            loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  // Boolean facts store true as 1, while signed i1 orders it below false.
  if (bit_count == 1) {
    loom_value_facts_maximum_range(&lhs_facts, &rhs_facts, out);
  } else {
    loom_value_facts_minimum_range(&lhs_facts, &rhs_facts, out);
  }
}

void loom_value_facts_maxsi(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs, int32_t bit_count,
                            loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  if (bit_count == 1) {
    loom_value_facts_minimum_range(&lhs_facts, &rhs_facts, out);
  } else {
    loom_value_facts_maximum_range(&lhs_facts, &rhs_facts, out);
  }
}

void loom_value_facts_minui(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  if ((lhs_facts.range_lo >= 0 && rhs_facts.range_lo >= 0) ||
      (lhs_facts.range_hi < 0 && rhs_facts.range_hi < 0)) {
    loom_value_facts_minimum_range(&lhs_facts, &rhs_facts, out);
    return;
  }
  // Unsigned order agrees within each signed half; every nonnegative value
  // precedes every negative value in the source carrier's unsigned order.
  if (lhs_facts.range_lo >= 0 && rhs_facts.range_hi < 0) {
    *out = lhs_facts;
    return;
  }
  if (rhs_facts.range_lo >= 0 && lhs_facts.range_hi < 0) {
    *out = rhs_facts;
    return;
  }
  loom_value_facts_meet(&lhs_facts, &rhs_facts, out);
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
}

void loom_value_facts_maxui(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  if ((lhs_facts.range_lo >= 0 && rhs_facts.range_lo >= 0) ||
      (lhs_facts.range_hi < 0 && rhs_facts.range_hi < 0)) {
    loom_value_facts_maximum_range(&lhs_facts, &rhs_facts, out);
    return;
  }
  if (lhs_facts.range_lo >= 0 && rhs_facts.range_hi < 0) {
    *out = rhs_facts;
    return;
  }
  if (rhs_facts.range_lo >= 0 && lhs_facts.range_hi < 0) {
    *out = lhs_facts;
    return;
  }
  loom_value_facts_meet(&lhs_facts, &rhs_facts, out);
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
}
