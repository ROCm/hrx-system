// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Deterministic materialization-budget policy for optional callable inlining.

#ifndef LOOM_TRANSFORMS_SYMBOL_INLINE_PROFITABILITY_H_
#define LOOM_TRANSFORMS_SYMBOL_INLINE_PROFITABILITY_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_INLINE_PROFITABILITY_DEFAULT_BUDGET 128u
#define LOOM_INLINE_PROFITABILITY_DEFAULT_HOT_BUDGET 512u

// Materialization budgets measured in expanded Loom body units. Each operation
// contributes at least one unit; scalar and shaped-value operations may
// contribute more according to their static 32-bit payload word count. A zero
// budget disables its corresponding selection path.
typedef struct loom_inline_profitability_options_t {
  // Combined ordinary and hot materialization budget.
  uint32_t budget;
  // Hot-only fallback materialization budget.
  uint32_t hot_budget;
} loom_inline_profitability_options_t;

// Whole-cohort selection for one callee.
typedef struct loom_inline_profitability_decision_t {
  // Select every eligible ordinary-temperature call to the callee.
  bool ordinary;
  // Select every eligible hot call to the callee.
  bool hot;
} loom_inline_profitability_decision_t;

// Adds one callee expansion to an enclosing body cost. The replaced call op is
// already present in the enclosing cost, so only callee_cost - 1 is added.
// Arithmetic saturates at UINT32_MAX.
uint32_t loom_inline_profitability_add_expansion(uint32_t enclosing_cost,
                                                 uint32_t callee_cost);

// Selects complete temperature cohorts for one callee.
loom_inline_profitability_decision_t loom_inline_profitability_select(
    loom_inline_profitability_options_t options, uint32_t expanded_body_cost,
    uint32_t ordinary_call_count, uint32_t hot_call_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SYMBOL_INLINE_PROFITABILITY_H_
