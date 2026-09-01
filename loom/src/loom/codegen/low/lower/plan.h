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

// Shared descriptor-matrix plan retained between contract selection and
// emission.
typedef struct loom_low_lower_descriptor_matrix_plan_t {
  // Shared source adapter used by this matrix descriptor plan.
  loom_target_contract_descriptor_matrix_source_t source;
  // Descriptor row selected by the target matrix projection.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Target-independent request facts used to materialize descriptor operands.
  loom_contract_request_t contract_request;
  // Target-owned immediate attributes materialized from request facts.
  loom_named_attr_slice_t attrs;
  // Native contraction placement selected by the target query.
  const loom_native_contraction_facts_t* native_contraction_facts;
} loom_low_lower_descriptor_matrix_plan_t;

// Returns true when structured Low control flow is selected.
bool loom_low_lower_structured_low_enabled(
    const loom_low_lower_context_t* context);

// Returns true when |source_op| is supported by structured Low lowering.
bool loom_low_lower_supported_structured_source_op(
    const loom_low_lower_context_t* context, const loom_op_t* source_op);

// Returns true when |op| is lowered by target-independent structural logic.
bool loom_low_lower_op_is_structural(const loom_module_t* module,
                                     const loom_op_t* op);

// Returns true when |kind| carries source-only metadata and emits no Low op.
bool loom_low_lower_op_is_source_metadata(loom_op_kind_t kind);

// Returns true when |op| requires target-policy selection.
bool loom_low_lower_op_uses_policy(const loom_module_t* module,
                                   const loom_op_t* op);

// Returns true when |op| is a result-free hint that may be discarded.
bool loom_low_lower_op_is_discardable_hint(const loom_module_t* module,
                                           const loom_op_t* op);

// Selects and retains one lowering plan for each policy-owned source op.
iree_status_t loom_low_lower_select_plans(loom_low_lower_context_t* context,
                                          loom_region_t* source_body);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_PLAN_H_
