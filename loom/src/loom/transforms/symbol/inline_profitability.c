// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/inline_profitability.h"

static uint32_t loom_inline_profitability_saturating_add(uint32_t lhs,
                                                         uint32_t rhs) {
  return rhs > UINT32_MAX - lhs ? UINT32_MAX : lhs + rhs;
}

static uint32_t loom_inline_profitability_saturating_multiply(uint32_t lhs,
                                                              uint32_t rhs) {
  return lhs != 0 && rhs > UINT32_MAX / lhs ? UINT32_MAX : lhs * rhs;
}

uint32_t loom_inline_profitability_add_expansion(uint32_t enclosing_cost,
                                                 uint32_t callee_cost) {
  const uint32_t replacement_cost = callee_cost > 0 ? callee_cost - 1 : 0;
  return loom_inline_profitability_saturating_add(enclosing_cost,
                                                  replacement_cost);
}

loom_inline_profitability_decision_t loom_inline_profitability_select(
    loom_inline_profitability_options_t options, uint32_t expanded_body_cost,
    uint32_t ordinary_call_count, uint32_t hot_call_count) {
  loom_inline_profitability_decision_t decision = {0};
  const uint32_t total_call_count = loom_inline_profitability_saturating_add(
      ordinary_call_count, hot_call_count);

  const uint32_t total_score = loom_inline_profitability_saturating_multiply(
      expanded_body_cost, total_call_count);
  if (options.budget != 0 && total_score <= options.budget) {
    decision.ordinary = ordinary_call_count != 0;
    decision.hot = hot_call_count != 0;
    return decision;
  }

  const uint32_t hot_score = loom_inline_profitability_saturating_multiply(
      expanded_body_cost, hot_call_count);
  if (options.hot_budget != 0 && hot_call_count != 0 &&
      hot_score <= options.hot_budget) {
    decision.hot = true;
  }
  return decision;
}
