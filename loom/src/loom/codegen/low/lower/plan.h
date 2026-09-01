// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Retained source-to-Low lowering plans.

#ifndef LOOM_CODEGEN_LOW_LOWER_PLAN_H_
#define LOOM_CODEGEN_LOW_LOWER_PLAN_H_

#include "loom/codegen/low/lower/lower_rules.h"

#ifdef __cplusplus
extern "C" {
#endif

enum loom_low_lower_selected_plan_flag_bits_e {
  // The selected source op is intentionally skipped because none of its results
  // require target-low storage.
  LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED = (uint8_t)1u << 0,
};
typedef uint8_t loom_low_lower_selected_plan_flags_t;

typedef enum loom_low_lower_selected_plan_kind_e {
  // Selection came from a table-driven source-to-low rule.
  LOOM_LOW_LOWER_SELECTED_PLAN_RULE = 0,
  // Selection came from a shared descriptor-matrix contract row.
  LOOM_LOW_LOWER_SELECTED_PLAN_DESCRIPTOR_MATRIX = 1,
  // Selection came from a target-owned callback plan.
  LOOM_LOW_LOWER_SELECTED_PLAN_CALLBACK = 2,
} loom_low_lower_selected_plan_kind_t;

// One source operation's lowering decision retained between planning and
// emission.
typedef struct loom_low_lower_selected_plan_t {
  // Source op this selected plan lowers.
  const loom_op_t* source_op;
  // Selected plan representation.
  loom_low_lower_selected_plan_kind_t kind;
  // Selection lifecycle flags.
  loom_low_lower_selected_plan_flags_t flags;
  // Policy rule-set ordinal for table-driven selections.
  uint16_t rule_set_index;
  // Rule-table ordinal for table-driven selections.
  uint16_t rule_index;
  // Rule set owning |rule|, or NULL for target-owned callbacks.
  const loom_low_lower_rule_set_t* rule_set;
  // Table rule selected during planning, or NULL for target-owned callbacks.
  const loom_low_lower_rule_t* rule;
  // Resolved emit rows for |rule|, or NULL for target-owned callbacks.
  const loom_low_lower_resolved_emit_t* resolved_emits;
  // Canonical source-memory plan retained from rule selection, or NULL when
  // the selected rule does not consume source memory.
  const loom_low_source_memory_access_plan_t* source_memory_access;
  // Target-owned plan selected during planning, or empty for table rules.
  loom_low_lower_plan_t plan;
} loom_low_lower_selected_plan_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_PLAN_H_
