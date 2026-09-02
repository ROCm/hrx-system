// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Retained source-to-Low lowering plans.
//
// Source planning runs after target legality and before target-Low IR
// construction. It selects one rule, descriptor-matrix row, or target callback
// plan for each non-structural source operation in traversal order. A backward
// demand walk then marks every source value needed by emission and elides pure
// plans whose results have no materialized use. The retained plan is immutable
// during emission except for its monotonic consumption cursor.

#ifndef LOOM_CODEGEN_LOW_LOWER_SOURCE_PLAN_H_
#define LOOM_CODEGEN_LOW_LOWER_SOURCE_PLAN_H_

#include "loom/codegen/low/lower/lower_rules.h"

#ifdef __cplusplus
extern "C" {
#endif

enum loom_low_lower_value_storage_flag_bits_e {
  // The source value must be materialized as a target-Low SSA value.
  LOOM_LOW_LOWER_VALUE_STORAGE_REQUIRED = (uint8_t)1u << 0,
};
typedef uint8_t loom_low_lower_value_storage_flags_t;

enum loom_low_lower_selected_plan_flag_bits_e {
  // The selected source op is intentionally skipped because none of its
  // results require a target-Low SSA value.
  LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED = (uint8_t)1u << 0,
};
typedef uint8_t loom_low_lower_selected_plan_flags_t;

typedef enum loom_low_lower_selected_plan_kind_e {
  // Selection came from a table-driven source-to-Low rule.
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

// Function-local retained plan and source-value materialization state.
typedef struct loom_low_lower_source_plan_t {
  // Per-source-value storage demand flags indexed by source value ordinal.
  loom_low_lower_value_storage_flags_t* value_storage_flags;
  // Selected plans in source traversal order.
  loom_low_lower_selected_plan_t* selected_plans;
  // Number of populated selected plans.
  iree_host_size_t selected_plan_count;
  // Number of allocated selected plan slots.
  iree_host_size_t selected_plan_capacity;
  // Next selected plan consumed by the emission walk.
  iree_host_size_t selected_plan_emit_index;
} loom_low_lower_source_plan_t;

// Builds and finalizes the retained source plan for |source_body|.
//
// The caller must have initialized the function value domain, selected a
// descriptor set, composed the target contract index, and validated the source
// function boundary. The function owns its planning scratch arena lifetime and
// retains plan data in the lowering context's function arena.
iree_status_t loom_low_lower_source_plan_build(
    loom_low_lower_context_t* context, loom_region_t* source_body);

// Returns true when structured Low control flow is selected.
bool loom_low_lower_source_plan_uses_structured_control_flow(
    const loom_low_lower_context_t* context);

// Returns true when |kind| carries source-only metadata and emits no Low op.
bool loom_low_lower_source_plan_op_is_metadata(loom_op_kind_t kind);

// Checks that |source_value_id| has a target-Low type and returns that type.
// User-facing mapping failures are emitted through the lowering context.
iree_status_t loom_low_lower_source_plan_check_mapped_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, loom_type_t* out_low_type);

// Returns true when |source_value_id| needs a materialized target-Low result.
bool loom_low_lower_source_plan_result_storage_required(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id);

// Returns the exact condition value when facts prove a cfg.cond_br direction.
bool loom_low_lower_source_plan_cfg_cond_br_exact_bool(
    const loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_condition);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_SOURCE_PLAN_H_
