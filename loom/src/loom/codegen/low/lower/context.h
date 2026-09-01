// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Mutable state for one source-to-Low function lowering.

#ifndef LOOM_CODEGEN_LOW_LOWER_CONTEXT_H_
#define LOOM_CODEGEN_LOW_LOWER_CONTEXT_H_

#include "iree/base/internal/arena.h"
#include "loom/analysis/symbolic_expr.h"
#include "loom/analysis/view_regions.h"
#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/lower/plan.h"
#include "loom/codegen/low/memory_access.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_LOW_LOWER_VALUE_ID_ELIDED ((loom_value_id_t)(UINT32_MAX - 1))

enum loom_low_lower_value_storage_flag_bits_e {
  // The source value must be materialized as a low SSA value.
  LOOM_LOW_LOWER_VALUE_STORAGE_REQUIRED = (uint8_t)1u << 0,
};
typedef uint8_t loom_low_lower_value_storage_flags_t;

typedef struct loom_low_lower_memory_expr_entry_t
    loom_low_lower_memory_expr_entry_t;
typedef struct loom_low_lower_rule_descriptor_map_t
    loom_low_lower_rule_descriptor_map_t;
typedef struct loom_low_lower_target_state_record_t
    loom_low_lower_target_state_record_t;

typedef struct loom_low_lower_successor_interpositions_t {
  // Effective low destinations indexed by source terminator successor ordinal.
  // NULL entries use the destination implied by the source successor block.
  loom_block_t** low_dests;
  // Number of entries in low_dests.
  uint16_t low_dest_count;
} loom_low_lower_successor_interpositions_t;

typedef enum loom_low_lower_function_analysis_phase_e {
  LOOM_LOW_LOWER_FUNCTION_ANALYSIS_EMPTY = 0,
  LOOM_LOW_LOWER_FUNCTION_ANALYSIS_EXPRESSIONS = 1,
  LOOM_LOW_LOWER_FUNCTION_ANALYSIS_VIEW_REGIONS = 2,
} loom_low_lower_function_analysis_phase_t;

typedef struct loom_low_lower_function_analysis_t {
  // Furthest analysis phase completed for the active fact table.
  loom_low_lower_function_analysis_phase_t phase;
  // Function-local stable symbolic expressions shared by rules and views.
  loom_symbolic_expr_context_t expression_context;
  // View-region table borrowing expression_context.
  loom_view_region_table_t view_regions;
} loom_low_lower_function_analysis_t;

typedef struct loom_low_lowering_frame_t {
  // Active source-function value domain for dense per-value lowering state.
  loom_local_value_domain_t value_domain;
  // Borrowed source value facts computed before planning.
  loom_value_fact_table_t* fact_table;
  // Reusable traversal state for condition-fact queries.
  loom_condition_query_t condition_query;
  // Stable function analyses advanced monotonically on demand.
  loom_low_lower_function_analysis_t function_analysis;
  // Per-source-value storage demand flags indexed by source value ordinal.
  loom_low_lower_value_storage_flags_t* value_storage_flags;
  // Source local value ordinal to emitted low value ID map.
  loom_value_id_t* value_map;
  // Source block ordinal to emitted low block pointer map.
  loom_block_t** block_map;
  // Source block ordinal to per-successor low destination interpositions.
  loom_low_lower_successor_interpositions_t* successor_interpositions;
  // Source block ordinal to target branch plan selected after low blocks exist.
  loom_low_lower_plan_t* branch_plans;
  // Source function argument ABI mappings.
  loom_low_lower_abi_argument_t* argument_map;
  // Number of entries in argument_map.
  uint16_t argument_map_count;
  // Selected lowering plans for non-structural source ops.
  loom_low_lower_selected_plan_t* selected_plans;
  // Number of selected plan slots used during planning.
  iree_host_size_t selected_plan_count;
  // Number of selected plan slots allocated for planning.
  iree_host_size_t selected_plan_capacity;
  // Next selected plan consumed by the emission walk.
  iree_host_size_t selected_plan_emit_index;
  // Cached source-function CFG block execution counts for memory reports.
  uint64_t* source_block_execution_counts;
  // True when source_block_execution_counts has been initialized.
  bool source_block_execution_counts_initialized;
  // True when every reachable source CFG backedge was counted exactly.
  bool source_block_execution_counts_exact;
  // Function-local symbolic byte expressions interned for report accounting.
  loom_low_lower_memory_expr_entry_t* memory_expr_entries;
  // Number of interned symbolic byte expressions.
  iree_host_size_t memory_expr_entry_count;
  // Capacity of |memory_expr_entries|.
  iree_host_size_t memory_expr_entry_capacity;
  // Source-derived memory access rows copied into options.table_arena.
  loom_low_memory_access_record_t* memory_access_records;
  // Number of memory access rows recorded during emission.
  iree_host_size_t memory_access_record_count;
  // Capacity of memory_access_records.
  iree_host_size_t memory_access_record_capacity;
  // Descriptor set used to build rule_descriptor_maps.
  const loom_low_descriptor_set_t* rule_descriptor_map_set;
  // Per-policy-rule-set descriptor-ref to descriptor-row maps.
  loom_low_lower_rule_descriptor_map_t* rule_descriptor_maps;
  // Number of entries in rule_descriptor_maps.
  uint16_t rule_descriptor_map_count;
  // Function-local target state records keyed by target-owned static storage.
  loom_low_lower_target_state_record_t* target_state_records;
  // Number of populated target_state_records entries.
  iree_host_size_t target_state_record_count;
  // Number of allocated target_state_records entries.
  iree_host_size_t target_state_record_capacity;
} loom_low_lowering_frame_t;

static inline loom_value_ordinal_t loom_low_lowering_frame_value_ordinal(
    const loom_low_lowering_frame_t* frame, loom_value_id_t value_id) {
  return loom_local_value_domain_ordinal(&frame->value_domain, value_id);
}

struct loom_low_lower_context_t {
  // Module being mutated by this lowering run.
  loom_module_t* module;
  // Source function being lowered.
  loom_func_like_t source_function;
  // Caller-owned lowering options.
  const loom_low_lower_options_t* options;
  // Target-low lowering policy selected by the caller.
  const loom_low_lower_policy_t* policy;
  // Immutable policy contract set cached beside hot lowering state.
  const loom_low_lower_contract_set_t* contract_set;
  // Descriptor set selected by source legality.
  const loom_low_descriptor_set_t* descriptor_set;
  // Result object receiving counters and emitted low function metadata.
  loom_low_lower_result_t* result;
  // Arena retaining function plans, maps, analyses, and target state.
  iree_arena_allocator_t function_arena;
  // Arena reset after each source-op planning callback.
  iree_arena_allocator_t planning_arena;
  // True only while a source-op planning callback may request scratch storage.
  bool planning_arena_active;
  // Arena reset after each bounded low-IR emission scope.
  iree_arena_allocator_t emission_arena;
  // True only while a low-IR builder callback may request emission storage.
  bool emission_arena_active;
  // Module-scope state shared by source-to-low calls in the current module
  // pass, or NULL when the caller is lowering a standalone function.
  loom_low_lower_module_state_t* module_state;
  // Function-local state for this source-to-low lowering run.
  loom_low_lowering_frame_t lowering;
  // Builder used while emitting the low function.
  loom_builder_t builder;
  // Emitted target-low function operation, or NULL before emission starts.
  loom_op_t* low_func_op;
};

// Acquires dense source-value ordinals for |source_body| into |context|.
iree_status_t loom_low_lower_context_acquire_value_domain(
    loom_low_lower_context_t* context, loom_region_t* source_body);

// Releases the source-value domain acquired into |context|.
void loom_low_lower_context_release_value_domain(
    loom_low_lower_context_t* context);

// Returns the source function name used in source-to-low diagnostics/reports.
iree_string_view_t loom_low_lower_context_function_name(
    const loom_low_lower_context_t* context);

// Resolves descriptor refs through the lowering context's cached rule maps.
iree_status_t loom_low_lower_rule_match_descriptor_ref_from_lowering(
    void* user_data, const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor);

// Returns true when the lowering context has reached its diagnostic limit.
bool loom_low_lower_context_should_stop(
    const loom_low_lower_context_t* context);

// Emits a TARGET-domain diagnostic with the standard target-low source
// context followed by |extra_params|.
iree_status_t loom_low_lower_emit_target_context_error(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* extra_params,
    iree_host_size_t extra_param_count);

// Emits ERR_TARGET_001 for a source op with no selected target-low contract.
iree_status_t loom_low_lower_emit_no_target_contract(
    loom_low_lower_context_t* context, const loom_op_t* source_op);

// Copies a source SSA value display name onto a low SSA value.
iree_status_t loom_low_lower_copy_value_name(loom_low_lower_context_t* context,
                                             loom_value_id_t source_value_id,
                                             loom_value_id_t low_value_id);

// Returns the body of the emitted Low function.
loom_region_t* loom_low_lower_context_low_body(
    const loom_low_lower_context_t* context);

// Begins a bounded Low IR construction scope.
void loom_low_lower_context_emission_scope_begin(
    loom_low_lower_context_t* context);

// Ends a Low IR construction scope and releases its temporary storage.
void loom_low_lower_context_emission_scope_end(
    loom_low_lower_context_t* context);

// Adapts function-local target state allocation to contract-query callbacks.
iree_status_t loom_low_lower_contract_query_get_or_allocate_target_state(
    void* user_data, const void* key, iree_host_size_t data_length,
    void** out_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_CONTEXT_H_
