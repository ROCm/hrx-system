// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generated source-to-Low rule matching and rejection diagnostics.
//
// The matcher interprets immutable guard tables against source IR, value facts,
// target features, and descriptor availability. It returns a stable selected
// rule row or the best structured rejection; it never emits target-Low IR.

#ifndef LOOM_CODEGEN_LOW_LOWER_RULE_MATCH_H_
#define LOOM_CODEGEN_LOW_LOWER_RULE_MATCH_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower/rules.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_rule_match_context_t
    loom_low_lower_rule_match_context_t;
typedef struct loom_low_lower_rule_source_memory_state_t
    loom_low_lower_rule_source_memory_state_t;
typedef struct loom_symbolic_expr_context_t loom_symbolic_expr_context_t;

typedef iree_status_t (*loom_low_lower_rule_match_map_value_fn_t)(
    void* user_data, const loom_low_lower_rule_match_context_t* context,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value);

typedef struct loom_low_lower_rule_match_map_value_callback_t {
  // Callback invoked to map one source value into target-low register metadata.
  loom_low_lower_rule_match_map_value_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_rule_match_map_value_callback_t;

typedef iree_status_t (*loom_low_lower_rule_match_can_materialize_value_fn_t)(
    void* user_data, const loom_low_lower_rule_match_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, loom_value_id_t source_value_id,
    bool* out_can_materialize);

typedef struct loom_low_lower_rule_match_can_materialize_value_callback_t {
  // Callback invoked for VALUE_MATERIALIZABLE guards.
  loom_low_lower_rule_match_can_materialize_value_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_rule_match_can_materialize_value_callback_t;

typedef iree_status_t (*loom_low_lower_rule_match_descriptor_ref_fn_t)(
    void* user_data, const loom_low_lower_rule_match_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor);

typedef struct loom_low_lower_rule_match_descriptor_ref_callback_t {
  // Callback invoked to resolve a rule-set-local descriptor ref.
  loom_low_lower_rule_match_descriptor_ref_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_rule_match_descriptor_ref_callback_t;

typedef uint16_t loom_low_lower_rule_match_flags_t;

// Match contract-only rule rows that are visible to read-only legality queries
// but never execute as source-to-low emission programs.
#define LOOM_LOW_LOWER_RULE_MATCH_FLAG_CONTRACT_ONLY \
  ((loom_low_lower_rule_match_flags_t)1u << 0)

struct loom_low_lower_rule_match_context_t {
  // Source module being matched.
  const loom_module_t* module;
  // Source function whose body is being matched.
  loom_func_like_t function;
  // Target bundle selected for this match.
  const loom_target_bundle_t* bundle;
  // Typed target facts selected for this match.
  const loom_target_facts_t* target_facts;
  // Descriptor set selected for the target-low contract.
  const loom_low_descriptor_set_t* descriptor_set;
  // Feature bits selected by the target-low contract.
  uint64_t feature_bits;
  // Source-value to target-low register metadata mapper.
  loom_low_lower_rule_match_map_value_callback_t map_value;
  // Optional source value materializer predicate bridge.
  loom_low_lower_rule_match_can_materialize_value_callback_t can_materialize;
  // Optional rule-local descriptor-ref resolver. Missing uses descriptor keys
  // directly and is intended for tests and cold standalone queries.
  loom_low_lower_rule_match_descriptor_ref_callback_t descriptor_ref;
  // Optional dense source value facts used by fact-backed guard rows.
  const loom_value_fact_table_t* fact_table;
  // Optional active function-local value domain for ordinal-keyed analyses.
  const loom_local_value_domain_t* value_domain;
  // Optional immutable source-program structure for function-wide analyses.
  const loom_source_program_t* source_program;
  // Optional retained target-declared physical source-value facts.
  const loom_source_dataflow_result_t* source_dataflow;
  // Optional precomputed view summaries used by source-memory guard rows.
  const loom_view_region_table_t* view_regions;
  // Caller-owned state retaining the canonical source-memory plan for the
  // source op being matched. Required for rule sets with source-memory rows.
  loom_low_lower_rule_source_memory_state_t* source_memory_state;
  // Optional symbolic proof context used as a cold fallback for fact-backed
  // guard rows whose scalar intervals are inconclusive.
  loom_symbolic_expr_context_t* symbolic_expr_context;
  // Match behavior flags.
  loom_low_lower_rule_match_flags_t flags;
  // One-based policy rule-set ordinal supplied by composed contract selection;
  // zero when no policy owner is known.
  uint16_t policy_rule_set_ordinal;
};

typedef struct loom_low_lower_rule_selection_t {
  // Selected rule row, or NULL when no rule accepted the source op.
  const loom_low_lower_rule_t* rule;
  // Selected rule row ordinal, or UINT16_MAX when no rule accepted the source
  // op.
  uint16_t rule_index;
  // True when the rule set had at least one rule span for the source op kind.
  bool has_source_op_span;
  // Diagnostic row describing the best failed guard when |rule| is NULL.
  uint16_t diagnostic_index;
  // Number of guards matched by the best failed rule candidate.
  uint16_t matched_guard_count;
  // True when a descriptor guard rejected a rule whose source-memory
  // constraints matched the source access.
  bool source_memory_compatible;
  // True when the selected rule consumes the canonical source-memory plan.
  bool uses_source_memory_access;
} loom_low_lower_rule_selection_t;

// Initializes a rule match context backed by a mutable lowering context.
// |source_memory_state| retains the canonical source-memory plan across every
// candidate rule inspected for one source op.
void loom_low_lower_rule_match_context_initialize_from_lowering(
    loom_low_lower_context_t* context,
    const loom_view_region_table_t* view_regions,
    loom_low_lower_rule_source_memory_state_t* source_memory_state,
    loom_low_lower_rule_match_context_t* out_match_context);

// Resolves descriptor refs through a lowering context's function-local cache.
// This adapts the mutable lowering lifecycle to the read-only match callback.
iree_status_t loom_low_lower_rule_match_descriptor_ref_from_lowering(
    void* user_data, const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor);

// Selects the exact lowering rule for |source_op| without emitting user
// diagnostics. Callers that compose rule tables with custom target callbacks
// can use the recorded failure detail if every lowering path rejects the op.
iree_status_t loom_low_lower_rule_set_select(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_selection_t* out_selection);

// Selects the exact target contract rule for |source_op| using the mutable
// lowering context, including contract-only rows.
iree_status_t loom_low_lower_rule_set_select_contract(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_selection_t* out_selection);

// Selects a lowering rule from a caller-selected rule range using the mutable
// lowering context.
iree_status_t loom_low_lower_rule_set_select_rule_range(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t rule_start, uint16_t rule_count,
    loom_low_lower_rule_selection_t* out_selection);

// Selects the exact lowering rule using a read-only match context.
iree_status_t loom_low_lower_rule_set_select_with_match_context(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_selection_t* out_selection);

// Selects a rule from a caller-selected range using a read-only match context.
iree_status_t loom_low_lower_rule_set_select_rule_range_with_match_context(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t rule_start, uint16_t rule_count,
    loom_low_lower_rule_selection_t* out_selection);

// Returns the diagnostic row for a failed selection, or NULL when unavailable.
const loom_low_lower_diagnostic_t* loom_low_lower_rule_set_selection_diagnostic(
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_rule_selection_t selection);

// Materializes generated diagnostic parameter projections for a rejected rule.
void loom_low_lower_rule_materialize_diagnostic_params(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_diagnostic_t* diagnostic,
    loom_diagnostic_param_t* out_params);

// Resolves a rule-set-local descriptor ref against the selected descriptor set.
// Missing optional descriptors return NULL.
iree_status_t loom_low_lower_rule_resolve_descriptor_ref(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor);

// Returns the first descriptor ref emitted by |rule|, or
// LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE when none is emitted.
loom_low_lower_descriptor_ref_t loom_low_lower_rule_first_descriptor_ref(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_t* rule);

// Emits the structured diagnostic for a failed selection.
iree_status_t loom_low_lower_rule_set_emit_selection_failure(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_selection_t selection,
    loom_low_lower_rule_source_memory_state_t* source_memory_state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_RULE_MATCH_H_
