// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Retained source-to-Low lowering plans.
//
// Source planning runs after target legality and before target-Low IR
// construction. It selects a rule, shared structural plan, descriptor-matrix
// row, or target callback for each non-structural source op in traversal order.
// A backward demand walk marks every source value needed by emission and elides
// pure plans whose results have no materialized use. The retained plan is
// immutable during emission except for its monotonic consumption cursor.

#ifndef LOOM_CODEGEN_LOW_LOWER_SOURCE_PLAN_H_
#define LOOM_CODEGEN_LOW_LOWER_SOURCE_PLAN_H_

#include "loom/codegen/low/lower/rules.h"
#include "loom/codegen/low/lower/source_memory.h"
#include "loom/codegen/low/lower/structural_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_resolved_emit_t loom_low_lower_resolved_emit_t;
typedef struct loom_low_representation_plan_t loom_low_representation_plan_t;
typedef struct loom_low_lower_realizations_t loom_low_lower_realizations_t;
typedef struct loom_low_lower_source_invoke_plan_t
    loom_low_lower_source_invoke_plan_t;

enum loom_low_lower_value_flag_bits_e {
  // The source value must be materialized as a target-Low SSA value.
  LOOM_LOW_LOWER_VALUE_STORAGE_REQUIRED = (uint8_t)1u << 0,
  // One selected memory plan can reuse this source realization. A second plan
  // requires storage so shared authored arithmetic is not rebuilt per access.
  LOOM_LOW_LOWER_VALUE_STORAGE_MEMORY_REALIZATION_SEEN = (uint8_t)1u << 1,
  // A selected rule addresses this source value through a retained fact.
  // Source-DAG rules must preserve a Low mapping for the value even though the
  // fact reference is not an ordinary SSA use visible while matching the DAG.
  LOOM_LOW_LOWER_VALUE_STORAGE_FACT_REFERENCE = (uint8_t)1u << 2,
  // Storage was required before backward selected-plan demand analysis.
  // Refinement retains these structural and function-boundary requirements.
  LOOM_LOW_LOWER_VALUE_STORAGE_BASELINE_REQUIRED = (uint8_t)1u << 3,
  // The binding slot contains an emitted value instead of a selected type.
  LOOM_LOW_LOWER_VALUE_MATERIALIZED = (uint8_t)1u << 4,
  // The planned binding inherits its carrier from one flattened source ordinal.
  LOOM_LOW_LOWER_VALUE_INHERITED_TYPE = (uint8_t)1u << 5,
  // A later source op is owned by an already selected nonlocal source graph.
  LOOM_LOW_LOWER_VALUE_SOURCE_GRAPH_RESERVED = (uint8_t)1u << 6,
};
typedef uint8_t loom_low_lower_value_flags_t;

enum loom_low_lower_selected_plan_flag_bits_e {
  // The selected source op is intentionally skipped because none of its
  // results require a target-Low SSA value.
  LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED = (uint8_t)1u << 0,
  // The source op is emitted by another selected rule spanning a source DAG.
  LOOM_LOW_LOWER_SELECTED_PLAN_CLAIMED = (uint8_t)1u << 1,
};
typedef uint8_t loom_low_lower_selected_plan_flags_t;

typedef enum loom_low_lower_selected_plan_kind_e {
  // Selection came from a table-driven source-to-Low rule.
  LOOM_LOW_LOWER_SELECTED_PLAN_RULE = 0,
  // Selection came from a shared descriptor-matrix contract row.
  LOOM_LOW_LOWER_SELECTED_PLAN_DESCRIPTOR_MATRIX = 1,
  // Selection came from a target-owned callback plan.
  LOOM_LOW_LOWER_SELECTED_PLAN_CALLBACK = 2,
  // Selection reserves bounded function storage using a target space mapping.
  LOOM_LOW_LOWER_SELECTED_PLAN_FUNCTION_STORAGE = 3,
  // Shared normalization of a semantic invocation into a native helper call.
  LOOM_LOW_LOWER_SELECTED_PLAN_INVOKE = 4,
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
  // Number of source nodes owned by a table rule, including the root. Zero for
  // non-rule plans and claimed placeholders.
  uint8_t source_node_count;
  // Policy rule-set ordinal for table-driven selections.
  uint16_t rule_set_index;
  // Rule-table ordinal for table-driven selections.
  uint16_t rule_index;
  // Rule set owning |rule|, or NULL for non-rule plans and claimed
  // placeholders.
  const loom_low_lower_rule_set_t* rule_set;
  // Table rule selected during planning, or NULL for non-rule plans and claimed
  // placeholders. An earlier selection claimed by a later rule retains its
  // original recipe for reporting.
  const loom_low_lower_rule_t* rule;
  // Resolved emit rows for |rule|, or NULL for non-rule plans.
  const loom_low_lower_resolved_emit_t* resolved_emits;
  // Canonical source-memory plan retained from rule selection, or NULL when
  // the selected rule does not consume source memory.
  const loom_low_source_memory_access_plan_t* source_memory_access;
  // Selected-plan-specific retained payload.
  union {
    // Helper signature and proved preconditions owned by the function plan.
    const loom_low_lower_source_invoke_plan_t* invoke;
    // Shared bounded-allocation plan owned by the function lowering arena.
    const loom_low_lower_function_storage_plan_t* function_storage;
    // Target-owned plan selected during planning.
    loom_low_lower_plan_t target_plan;
    // Function-arena-owned source graph for a multi-node table rule. Entry
    // zero is |source_op|. NULL for root-only rules.
    const loom_op_t* const* source_nodes;
  } data;
} loom_low_lower_selected_plan_t;

// Shared descriptor-matrix plan retained between contract selection and
// emission.
typedef struct loom_low_lower_descriptor_matrix_plan_t {
  // Shared source adapter used by this matrix descriptor plan.
  loom_target_contract_descriptor_matrix_source_t source;
  // Exact source-to-native transform selected with the descriptor.
  loom_target_contract_descriptor_matrix_transform_flags_t transform_flags;
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
  // Shared structural type and branch decisions consumed without value facts.
  loom_low_lower_structural_plan_t structural;
  // Required visibility on mutable global reads; thread scope keeps the
  // ordinary eager acquisition recipe. Fixed before per-operation selection.
  uint8_t read_visibility_scope;
  // Owned source-body permutation in definition-before-use order, copied from
  // dominance. Unreachable blocks follow in storage order. NULL preserves the
  // single-block structured path.
  const uint16_t* block_order;
  // Function-local physical-representation plan, or NULL when the target has
  // no representation observer or before that observer begins.
  loom_low_representation_plan_t* representation_plan;
  // Shared pure-value placement, initialization and supplemental CFG payloads.
  loom_low_lower_realizations_t* realizations;
  // Storage demands and binding states indexed by source value ordinal.
  loom_low_lower_value_flags_t* value_flags;
  // Direct selected-plan indices keyed by each source op's first result value.
  // NULL when all source graphs require lexical adjacency.
  uint32_t* selected_plan_indices_by_value_ordinal;
  // Number of values addressed through selected fact-derived references.
  // Zero keeps ordinary rule selection and demand analysis on the direct path.
  loom_value_ordinal_t fact_storage_demand_count;
  // True only while backward selected-plan storage demands are being marked.
  // Requirements established before this phase survive plan refinement.
  bool is_analyzing_storage_demands;
  // Canonical accesses joined across observation and selection without an op
  // lookup table or a second address-analysis walk.
  struct {
    // First retained access in shared source traversal order.
    loom_low_lower_source_memory_record_t* first;
    // Next access to consume during selection or emission.
    const loom_low_lower_source_memory_record_t* cursor;
    // Access visible to the current observer, selector, or emitter, or NULL.
    const loom_low_lower_source_memory_record_t* current;
  } memory;
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
// function arguments. Discovery establishes target-neutral result mappings;
// targets with a physical representation plan refine those mappings after the
// plan is solved. The function owns its planning scratch arena lifetime and
// retains plan data in the lowering context's function arena.
iree_status_t loom_low_lower_source_plan_build(
    loom_low_lower_context_t* context, loom_region_t* source_body);

// Returns true when structured Low control flow is selected.
bool loom_low_lower_source_plan_uses_structured_control_flow(
    const loom_low_lower_context_t* context);

// Returns true when |kind| carries source-only metadata and emits no Low op.
bool loom_low_lower_source_plan_op_is_metadata(loom_op_kind_t kind);

// Returns true when |source_value_id| needs a materialized target-Low result.
bool loom_low_lower_source_plan_result_storage_required(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_SOURCE_PLAN_H_
