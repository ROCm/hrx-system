// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-to-Low lowering context implementation contract.
//
// Public target callbacks use the opaque context accessors declared in
// lower.h. This header owns the stack-local context representation and the
// phase-local analysis and the retained frame shared by source planning,
// callable boundaries, reporting, and rule interpretation.

#ifndef LOOM_CODEGEN_LOW_LOWER_CONTEXT_H_
#define LOOM_CODEGEN_LOW_LOWER_CONTEXT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/cfg_value_identity.h"
#include "loom/analysis/storage_interference.h"
#include "loom/analysis/symbolic_expr.h"
#include "loom/analysis/view_regions.h"
#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/lower/control_plan.h"
#include "loom/codegen/low/lower/execution.h"
#include "loom/codegen/low/lower/function_boundary.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/codegen/low/lower/report.h"
#include "loom/codegen/low/lower/rules.h"
#include "loom/codegen/low/lower/source_plan.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel indicating that a source value was intentionally erased.
#define LOOM_LOW_LOWER_VALUE_ID_ELIDED ((loom_value_id_t)(UINT32_MAX - 1))

typedef struct loom_low_lower_rule_descriptor_cache_t
    loom_low_lower_rule_descriptor_cache_t;
typedef struct loom_low_lower_source_query_coverage_t
    loom_low_lower_source_query_coverage_t;
typedef struct loom_low_lower_source_graph_placement_t
    loom_low_lower_source_graph_placement_t;
typedef struct loom_low_lower_participation_state_t
    loom_low_lower_participation_state_t;
typedef struct loom_control_participant_table_t
    loom_control_participant_table_t;

typedef struct loom_low_lower_target_state_record_t {
  // Target-owned static key identifying this function-local state object.
  const void* key;
  // Byte length of state storage.
  iree_host_size_t data_length;
  // Zero-initialized state storage allocated from the lowering arena.
  void* data;
} loom_low_lower_target_state_record_t;

typedef enum loom_low_lower_function_analysis_phase_e {
  LOOM_LOW_LOWER_FUNCTION_ANALYSIS_EMPTY = 0,
  LOOM_LOW_LOWER_FUNCTION_ANALYSIS_EXPRESSIONS = 1,
  LOOM_LOW_LOWER_FUNCTION_ANALYSIS_VIEW_REGIONS = 2,
} loom_low_lower_function_analysis_phase_t;

typedef struct loom_low_lower_function_analysis_t {
  // Furthest analysis phase completed for the active fact table.
  loom_low_lower_function_analysis_phase_t phase;
  // Exact CFG representatives retained from source footprint verification.
  loom_cfg_value_identity_table_t value_identities;
  // Function-local stable symbolic expressions shared by rules and views.
  loom_symbolic_expr_context_t expression_context;
  // View-region table borrowing expression_context.
  loom_view_region_table_t view_regions;
  // Source body multiplicities retained independently of optional reports.
  loom_low_lower_execution_counts_t execution_counts;
  // Source control participation shared by target planning and reports.
  loom_low_lower_participation_state_t* participation;
  // Compact block guarantees projected during common condition verification.
  loom_control_participant_table_t* participant_zeros;
  // Shared source allocation provenance, accesses, and instance lifetimes.
  loom_storage_interference_t* storage_interference;
} loom_low_lower_function_analysis_t;

typedef struct loom_low_lowering_frame_t {
  // Stable source IDs and ordinals; borrows module scratch only while this
  // plan is being built or executed.
  loom_local_value_domain_t value_domain;
  // Declared terminator kind for direct exits from the source callable body,
  // or unknown while querying a bodyless callable.
  loom_op_kind_t source_callable_exit_kind;
  // Retained source lowering decisions and value materialization demands.
  loom_low_lower_source_plan_t source_plan;
  // Source ordinal to selected type, replaced in place by the emitted value.
  loom_low_lower_value_binding_t* value_bindings;
  // Planned block topology and branch expansions, before Low construction.
  loom_low_lower_control_plan_t* control_plan;
  // Callable signature, argument imports, and ABI layout retained for emission.
  loom_low_lower_function_boundary_t boundary;
  // Shared descriptor bindings for the active immutable target tables. Without
  // module state, this is the head of the function-owned cache list.
  loom_low_lower_rule_descriptor_cache_t* rule_descriptor_cache;
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
  // Descriptor set selected by source legality.
  const loom_low_descriptor_set_t* descriptor_set;
  // Result object receiving counters and emitted low function metadata.
  loom_low_lower_result_t* result;
  // Stable owner of retained function plans and target payloads.
  iree_arena_allocator_t* function_arena;
  // Function-local analysis storage retired before any plan executes.
  iree_arena_allocator_t analysis_arena;
  // Borrowed source value facts computed before planning.
  loom_value_fact_table_t* fact_table;
  // Demand scope over immutable source, retired before plan execution.
  loom_storage_access_scope_t* storage_access;
  // Reusable traversal state for condition-fact queries.
  loom_condition_query_t condition_query;
  // Stable function analyses advanced monotonically on demand.
  loom_low_lower_function_analysis_t function_analysis;
  // Source-graph coverage for the primary fact table, owned by analysis_arena.
  loom_low_lower_source_query_coverage_t* source_query_coverage;
  // Lazy source-graph emission placement state owned by analysis_arena.
  loom_low_lower_source_graph_placement_t* source_graph_placement;
  // Arena reset after each source-op planning callback.
  iree_arena_allocator_t planning_arena;
  // True only while a source-op planning callback may request scratch storage.
  bool planning_arena_active;
  // Active function emission storage, with transient callback scratch after
  // its block bindings. Released when this function finishes emitting.
  iree_arena_allocator_t emission_arena;
  // Function-owned emission storage preceding the active callback scratch.
  iree_arena_checkpoint_t emission_checkpoint;
  // True only while a low-IR builder callback may request emission storage.
  bool emission_arena_active;
  // Module-scope state shared by source-to-low calls in the current module
  // pass, or NULL when the caller is lowering a standalone function.
  loom_low_lower_module_state_t* module_state;
  // Plan-owned frame; its address remains stable between planning and emission.
  loom_low_lowering_frame_t* lowering;
  // Active phase reporting scratch; completed report rows belong to result.
  loom_low_lower_report_state_t report;
  // Emission-owned block bindings in planned order; absent during planning.
  loom_block_t** block_map;
  // Builder used while emitting the low function.
  loom_builder_t builder;
  // Emitted target-low function operation, or NULL before emission starts.
  loom_op_t* low_func_op;
};

// Returns the source function name used in source-to-low diagnostics/reports.
iree_string_view_t loom_low_lower_context_function_name(
    const loom_low_lower_context_t* context);

// Queries the active policy's native representation for a source value.
// An unsupported value produces none without a diagnostic. Mapping may allocate
// register types and function analysis state, but does not rewrite source IR.
iree_status_t loom_low_lower_query_value(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_value_id_t source_value_id,
                                         loom_type_t* out_low_type);

// Returns true when the lowering context has reached its diagnostic limit.
bool loom_low_lower_context_diagnostic_limit_reached(
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

// Begins a bounded target-Low IR construction scope.
void loom_low_lower_emission_scope_begin(loom_low_lower_context_t* context);

// Ends the active target-Low IR construction scope and releases its scratch
// storage.
void loom_low_lower_emission_scope_end(loom_low_lower_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_CONTEXT_H_
