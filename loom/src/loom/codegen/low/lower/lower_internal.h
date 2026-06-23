// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Private state shared by source-to-target-low lowering implementation files.

#ifndef LOOM_CODEGEN_LOW_LOWER_LOWER_INTERNAL_H_
#define LOOM_CODEGEN_LOW_LOWER_LOWER_INTERNAL_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbolic_expr.h"
#include "loom/analysis/view_regions.h"
#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/codegen/low/lower/lower_rules.h"
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
  // Target-owned plan selected during planning, or empty for table rules.
  loom_low_lower_plan_t plan;
} loom_low_lower_selected_plan_t;

typedef struct loom_low_lower_rule_descriptor_map_t {
  // Rule set whose local descriptor refs are resolved by descriptors.
  const loom_low_lower_rule_set_t* rule_set;
  // Descriptor rows indexed by rule-set-local descriptor ref.
  const loom_low_descriptor_t* const* descriptors;
  // Number of entries in descriptors.
  uint16_t descriptor_count;
} loom_low_lower_rule_descriptor_map_t;

typedef struct loom_low_lower_successor_interpositions_t {
  // Effective low destinations indexed by source terminator successor ordinal.
  // NULL entries use the destination implied by the source successor block.
  loom_block_t** low_dests;
  // Number of entries in low_dests.
  uint8_t low_dest_count;
} loom_low_lower_successor_interpositions_t;

typedef struct loom_low_lower_target_state_record_t {
  // Target-owned static key identifying this function-local state object.
  const void* key;
  // Byte length of state storage.
  iree_host_size_t data_length;
  // Zero-initialized state storage allocated from the lowering arena.
  void* data;
} loom_low_lower_target_state_record_t;

typedef struct loom_low_lower_module_target_state_record_t {
  // Target-owned static key identifying this module-scope state object.
  const void* key;
  // Byte length of state storage.
  iree_host_size_t data_length;
  // Zero-initialized state storage allocated from the module-state arena.
  void* data;
} loom_low_lower_module_target_state_record_t;

struct loom_low_lower_module_state_t {
  // Arena used for module-scope target state records and payloads.
  iree_arena_allocator_t* arena;
  // Module-scope target state records keyed by target-owned static storage.
  loom_low_lower_module_target_state_record_t* target_state_records;
  // Number of populated target_state_records entries.
  iree_host_size_t target_state_record_count;
  // Number of allocated target_state_records entries.
  iree_host_size_t target_state_record_capacity;
};

typedef struct loom_low_lowering_frame_t {
  // Active source-function value domain for dense per-value lowering state.
  loom_local_value_domain_t value_domain;
  // Borrowed source value facts computed before planning.
  loom_value_fact_table_t* fact_table;
  // Function-local symbolic proof context initialized on first rule query.
  loom_symbolic_expr_context_t expression_context;
  // Fact table used to initialize expression_context.
  const loom_value_fact_table_t* expression_context_fact_table;
  // True once expression_context owns arena-backed memo/scratch storage.
  bool expression_context_initialized;
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
  // Source-derived memory access rows copied into options.table_arena.
  loom_low_memory_access_record_t* memory_access_records;
  // Number of memory access rows recorded during emission.
  iree_host_size_t memory_access_record_count;
  // Capacity of memory_access_records.
  iree_host_size_t memory_access_record_capacity;
  // View-region table for the source function, initialized on first use.
  loom_view_region_table_t view_regions;
  // True after view_regions has been initialized against value_domain.
  bool view_regions_initialized;
  // True after view_regions has recorded per-view read/write flags.
  bool view_regions_analyzed;
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
  // Descriptor set selected by source legality.
  const loom_low_descriptor_set_t* descriptor_set;
  // Result object receiving counters and emitted low function metadata.
  loom_low_lower_result_t* result;
  // Scratch arena for transient maps and remapped operand lists.
  iree_arena_allocator_t arena;
  // Module-scope state shared by source-to-low calls in the current module
  // pass, or NULL when the caller is lowering a standalone function.
  loom_low_lower_module_state_t* module_state;
  // Function-local state for this source-to-low lowering run.
  loom_low_lowering_frame_t lowering;
  // Dense root contract index composed from the active policy shards.
  loom_target_contract_index_t contract_index;
  // Builder used while emitting the low function.
  loom_builder_t builder;
  // Emitted target-low function operation, or NULL before emission starts.
  loom_op_t* low_func_op;
};

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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_LOWER_INTERNAL_H_
