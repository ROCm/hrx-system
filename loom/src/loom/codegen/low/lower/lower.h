// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-to-target-low lowering foundation.
//
// This layer owns the target-independent mechanics of lowering a verified
// source function into a target-low function entry: target legality gating,
// function/block cloning, source-to-low SSA value mapping, source location
// preservation, and structural CFG terminators. Target packages provide
// descriptor choices and type mappings through callbacks so this library never
// links concrete backend descriptor tables.

#ifndef LOOM_CODEGEN_LOW_LOWER_LOWER_H_
#define LOOM_CODEGEN_LOW_LOWER_LOWER_H_

#include "iree/base/api.h"
#include "loom/analysis/contract_vector.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/memory_access.h"
#include "loom/error/emitter.h"
#include "loom/error/error_defs.h"
#include "loom/ir/ir.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/low_legality.h"
#include "loom/target/types.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_context_t loom_low_lower_context_t;
typedef struct loom_low_lower_source_query_scope_t
    loom_low_lower_source_query_scope_t;
typedef struct loom_low_lower_rule_set_t loom_low_lower_rule_set_t;
typedef struct loom_low_source_memory_access_plan_t
    loom_low_source_memory_access_plan_t;
typedef struct loom_view_region_table_t loom_view_region_table_t;

typedef struct loom_low_lower_rule_mapped_value_t {
  // True when the source value maps to a target-low register.
  bool is_register;
  // Descriptor-set register-class ID, or LOOM_LOW_REG_CLASS_NONE for
  // non-register values.
  uint16_t descriptor_register_class_id;
  // Number of target-low allocation units occupied by the mapped register.
  uint32_t register_unit_count;
} loom_low_lower_rule_mapped_value_t;

// Creates a non-register mapped value.
static inline loom_low_lower_rule_mapped_value_t
loom_low_lower_rule_mapped_value_none(void) {
  return (loom_low_lower_rule_mapped_value_t){
      /*.is_register=*/false,
      /*.descriptor_register_class_id=*/LOOM_LOW_REG_CLASS_NONE,
      /*.register_unit_count=*/0,
  };
}

// Creates a mapped register value addressed by descriptor-set register class.
static inline loom_low_lower_rule_mapped_value_t
loom_low_lower_rule_mapped_value_register(uint16_t descriptor_register_class_id,
                                          uint32_t register_unit_count) {
  return (loom_low_lower_rule_mapped_value_t){
      /*.is_register=*/true,
      /*.descriptor_register_class_id=*/descriptor_register_class_id,
      /*.register_unit_count=*/register_unit_count,
  };
}

typedef struct loom_low_lower_rule_set_list_t {
  // Total number of rule sets in the list.
  uint16_t count;
  // Rule set list or NULL if no rule sets.
  const loom_low_lower_rule_set_t* const* values;
} loom_low_lower_rule_set_list_t;

// Returns an empty rule set list.
static inline loom_low_lower_rule_set_list_t loom_low_lower_rule_set_list_empty(
    void) {
  return (loom_low_lower_rule_set_list_t){0, NULL};
}

// Returns true when |list| contains no rule sets.
static inline bool loom_low_lower_rule_set_list_is_empty(
    loom_low_lower_rule_set_list_t list) {
  return list.count == 0;
}

typedef iree_status_t (*loom_low_lower_map_type_fn_t)(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_type_t source_type,
    loom_type_t* out_low_type);

typedef struct loom_low_lower_map_type_callback_t {
  // Callback invoked to map one source value type to a low register type.
  loom_low_lower_map_type_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_map_type_callback_t;

typedef iree_status_t (*loom_low_lower_map_value_fn_t)(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_type_t source_type, loom_type_t* out_low_type);

typedef struct loom_low_lower_map_value_callback_t {
  // Optional callback invoked to map one concrete source SSA value to a low
  // register type. Missing uses |map_type|.
  loom_low_lower_map_value_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_map_value_callback_t;

typedef iree_status_t (*loom_low_lower_map_contract_value_fn_t)(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value);

typedef struct loom_low_lower_map_contract_value_callback_t {
  // Optional callback invoked during read-only contract queries to map one
  // source value into target-low register metadata without creating register
  // types or formatting diagnostics.
  loom_low_lower_map_contract_value_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_map_contract_value_callback_t;

typedef enum loom_low_lower_abi_argument_kind_e {
  // Source argument is passed as a low function block argument.
  LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT = 0,
  // Source argument is imported through a function-local low.resource op.
  LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE = 1,
} loom_low_lower_abi_argument_kind_t;

typedef struct loom_low_lower_abi_argument_t {
  // ABI path used for the source argument.
  loom_low_lower_abi_argument_kind_t kind;
  // Register type used by the low argument or imported low.resource result.
  loom_type_t abi_type;
  // Resource import kind used when |kind| is RESOURCE.
  loom_low_resource_import_kind_t resource_import_kind;
  // Dense target ABI resource index used when |kind| is RESOURCE.
  int64_t resource_index;
  // Semantic source type recorded on low.resource. None defaults to the source
  // argument type.
  loom_type_t resource_source_type;
  // Optional low.resource builder flags for resource-specific metadata.
  loom_low_resource_build_flags_t resource_build_flags;
  // Static byte extent for byte-addressable resources when the corresponding
  // builder flag is set.
  int64_t resource_extent;
  // Resource-level cache swizzle byte stride when the corresponding builder
  // flag is set.
  int64_t resource_cache_swizzle_stride;
} loom_low_lower_abi_argument_t;

typedef iree_status_t (*loom_low_lower_map_argument_fn_t)(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument);

typedef struct loom_low_lower_map_argument_callback_t {
  // Optional callback invoked to map a source function argument to a direct low
  // argument or target ABI resource. Missing uses direct |map_type| behavior.
  loom_low_lower_map_argument_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_map_argument_callback_t;

typedef iree_status_t (*loom_low_lower_emit_preamble_fn_t)(
    void* user_data, loom_low_lower_context_t* context);

typedef struct loom_low_lower_emit_preamble_callback_t {
  // Optional callback invoked after the low function and entry block are
  // created, before ABI resources and source body packets are emitted.
  loom_low_lower_emit_preamble_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_emit_preamble_callback_t;

typedef iree_status_t (*loom_low_lower_emit_entry_setup_fn_t)(
    void* user_data, loom_low_lower_context_t* context);

typedef struct loom_low_lower_emit_entry_setup_callback_t {
  // Optional callback invoked after ABI live-ins and resources are emitted,
  // before source body packets are emitted.
  loom_low_lower_emit_entry_setup_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_emit_entry_setup_callback_t;

typedef iree_status_t (*loom_low_lower_prepare_branch_fn_t)(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_terminator);

typedef struct loom_low_lower_prepare_branch_callback_t {
  // Optional callback invoked after source blocks have low blocks, before any
  // source body operations are emitted. Targets use this to plan structural
  // branch expansion and interpose low-only destination blocks.
  loom_low_lower_prepare_branch_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_prepare_branch_callback_t;

typedef iree_status_t (*loom_low_lower_materialize_branch_arg_fn_t)(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_terminator, uint8_t successor_index,
    uint16_t arg_index, loom_value_id_t source_value_id,
    loom_value_id_t low_value_id, loom_type_t required_low_type,
    loom_value_id_t* out_low_value_id);

typedef struct loom_low_lower_materialize_branch_arg_callback_t {
  // Optional callback invoked when a structural branch payload's already
  // lowered value does not match the destination block argument type. Targets
  // use this to materialize edge-local register-class copies without changing
  // the source value's canonical low mapping.
  loom_low_lower_materialize_branch_arg_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_materialize_branch_arg_callback_t;

typedef iree_status_t (*loom_low_lower_emit_cond_branch_fn_t)(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_value_id_t low_condition,
    loom_block_t* low_true_dest, loom_block_t* low_false_dest);

typedef struct loom_low_lower_emit_cond_branch_callback_t {
  // Optional callback invoked to emit a target-specific conditional branch.
  // Targets that provide this callback own all conditional branch emission for
  // the policy, including falling back to low.cond_br for native condition-code
  // values when appropriate.
  loom_low_lower_emit_cond_branch_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_emit_cond_branch_callback_t;

typedef enum loom_low_lower_abi_layout_kind_e {
  // Low function boundary layout on low.func.def/decl.
  LOOM_LOW_LOWER_ABI_LAYOUT_KIND_FUNC = 0,
  // Low kernel entry layout on low.kernel.def.
  LOOM_LOW_LOWER_ABI_LAYOUT_KIND_KERNEL = 1,
} loom_low_lower_abi_layout_kind_t;

typedef iree_status_t (*loom_low_lower_map_abi_layout_fn_t)(
    void* user_data, loom_low_lower_context_t* context,
    loom_low_lower_abi_layout_kind_t layout_kind, const loom_type_t* arg_types,
    iree_host_size_t arg_count, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_named_attr_slice_t* out_abi_layout);

typedef struct loom_low_lower_map_abi_layout_callback_t {
  // Optional callback invoked once while building a low boundary op. The
  // callback returns target-owned structured ABI layout facts; the low op
  // builder canonicalizes and copies the returned slice into the module arena.
  loom_low_lower_map_abi_layout_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_map_abi_layout_callback_t;

typedef uint64_t loom_low_lower_plan_id_t;

#define LOOM_LOW_LOWER_PLAN_ID_NONE UINT64_MAX

typedef struct loom_low_lower_plan_t {
  // Target-owned dense plan id selected during planning.
  loom_low_lower_plan_id_t id;
  // Arena-owned target plan data selected during planning and consumed by
  // emission during the same lowering run. Core lowering never interprets it.
  const void* target_data;
} loom_low_lower_plan_t;

// Returns a selected lowering plan. |target_data| points at immutable
// target-owned storage allocated from the current lowering arena, or NULL when
// |id| fully describes the plan.
static inline loom_low_lower_plan_t loom_low_lower_plan_make(
    loom_low_lower_plan_id_t id, const void* target_data) {
  return (loom_low_lower_plan_t){
      /*.id=*/id,
      /*.target_data=*/target_data,
  };
}

static inline loom_low_lower_plan_t loom_low_lower_plan_empty(void) {
  return loom_low_lower_plan_make(LOOM_LOW_LOWER_PLAN_ID_NONE, NULL);
}

static inline bool loom_low_lower_plan_is_empty(loom_low_lower_plan_t plan) {
  return plan.id == LOOM_LOW_LOWER_PLAN_ID_NONE;
}

typedef iree_status_t (*loom_low_lower_descriptor_matrix_options_fn_t)(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    loom_contract_vector_mma_options_t* out_options);

typedef iree_status_t (*loom_low_lower_descriptor_matrix_query_fn_t)(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_op_t* source_op, const loom_contract_request_t* request,
    loom_target_contract_query_result_t* out_result);

typedef iree_status_t (*loom_low_lower_descriptor_matrix_attrs_fn_t)(
    void* user_data, loom_low_lower_context_t* context,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_contract_request_t* request,
    const loom_low_descriptor_t* descriptor,
    loom_named_attr_slice_t* out_attrs);

typedef struct loom_low_lower_descriptor_matrix_t {
  // Supplies target-specific options for shared source-to-matrix adapters.
  loom_low_lower_descriptor_matrix_options_fn_t options;
  // Projects a generic matrix contract request to a target descriptor.
  loom_low_lower_descriptor_matrix_query_fn_t query;
  // Materializes target-owned immediate attributes for the selected descriptor.
  loom_low_lower_descriptor_matrix_attrs_fn_t attrs;
  // Caller-owned payload passed to descriptor-matrix callbacks.
  void* user_data;
} loom_low_lower_descriptor_matrix_t;

typedef struct loom_low_lower_selected_plan_view_t {
  // Source op this selected plan lowers.
  const loom_op_t* source_op;
  // Target-owned plan selected during planning. Table-driven rule rows return
  // an empty plan because their rule data is owned by core lowering.
  loom_low_lower_plan_t plan;
  // True when demand analysis proved the source op has no required low storage.
  bool elided;
} loom_low_lower_selected_plan_view_t;

typedef enum loom_low_lower_report_selection_kind_e {
  // No source-low selection was recorded.
  LOOM_LOW_LOWER_REPORT_SELECTION_NONE = 0,
  // Selection came from a table-driven lowering rule.
  LOOM_LOW_LOWER_REPORT_SELECTION_RULE = 1,
  // Selection came from a target-owned callback plan.
  LOOM_LOW_LOWER_REPORT_SELECTION_PLAN = 2,
} loom_low_lower_report_selection_kind_t;

// One source-to-target-low selection row captured for production diagnostics.
typedef struct loom_low_lower_report_row_t {
  // Source function symbol containing the lowered source operation.
  iree_string_view_t function_name;
  // Source operation mnemonic lowered by this row.
  iree_string_view_t source_op_name;
  // Numeric source operation kind lowered by this row.
  loom_op_kind_t source_op_kind;
  // Selection mechanism used for this source operation.
  loom_low_lower_report_selection_kind_t selection_kind;
  // Policy rule-set ordinal for table-driven rules, or UINT16_MAX otherwise.
  uint16_t rule_set_index;
  // Rule-table ordinal inside |rule_set_index|, or UINT16_MAX otherwise.
  uint16_t rule_index;
  // Target-owned plan id for callback selections, or PLAN_ID_NONE otherwise.
  loom_low_lower_plan_id_t plan_id;
  // Stable target-owned description of the selected plan variant, if any.
  iree_string_view_t plan_detail;
  // First stable low descriptor id emitted by a table rule, or none for plans.
  uint64_t descriptor_id;
  // Number of low operations emitted for this source operation.
  uint32_t emitted_low_op_count;
} loom_low_lower_report_row_t;

// Linked storage block for homogeneous source-to-low report rows.
typedef struct loom_low_lower_report_row_vec_t {
  // Next row block in allocation order, or NULL for the final block.
  struct loom_low_lower_report_row_vec_t* next;
  // Number of rows populated in this block.
  iree_host_size_t count;
  // Maximum number of rows that fit in this block.
  iree_host_size_t capacity;
} loom_low_lower_report_row_vec_t;

// Owned linked list of source-to-low report rows.
typedef struct loom_low_lower_report_row_list_t {
  // First row storage block, or NULL when empty.
  loom_low_lower_report_row_vec_t* head;
  // Last row storage block, or NULL when empty.
  loom_low_lower_report_row_vec_t* tail;
  // Total number of rows stored across all blocks.
  iree_host_size_t count;
} loom_low_lower_report_row_list_t;

// Returns mutable row storage for |vec|.
static inline loom_low_lower_report_row_t* loom_low_lower_report_row_vec_rows(
    loom_low_lower_report_row_vec_t* vec) {
  return (loom_low_lower_report_row_t*)(vec + 1);
}

// Returns immutable row storage for |vec|.
static inline const loom_low_lower_report_row_t*
loom_low_lower_report_row_vec_const_rows(
    const loom_low_lower_report_row_vec_t* vec) {
  return (const loom_low_lower_report_row_t*)(vec + 1);
}

typedef iree_status_t (*loom_low_lower_select_op_fn_t)(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t* out_plan);

typedef struct loom_low_lower_select_op_callback_t {
  // Callback invoked during planning to select the exact lowering plan for
  // one non-structural source op. Missing or unsupported plans must return
  // loom_low_lower_plan_empty().
  loom_low_lower_select_op_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_select_op_callback_t;

typedef iree_status_t (*loom_low_lower_emit_op_fn_t)(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t plan);

typedef struct loom_low_lower_emit_op_callback_t {
  // Callback invoked during emission to execute a plan selected during
  // planning for one non-structural source op.
  loom_low_lower_emit_op_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_emit_op_callback_t;

typedef void (*loom_low_lower_mark_plan_storage_demands_fn_t)(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t plan);

typedef struct loom_low_lower_mark_plan_storage_demands_callback_t {
  // Optional callback invoked during demand analysis for one target-owned
  // callback plan. Targets must mark every source SSA value that emission may
  // look up through the low value map.
  loom_low_lower_mark_plan_storage_demands_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_mark_plan_storage_demands_callback_t;

typedef iree_string_view_t (*loom_low_lower_describe_plan_fn_t)(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t plan);

typedef struct loom_low_lower_describe_plan_callback_t {
  // Optional callback returning a stable target-owned detail string for one
  // selected callback plan. The string is used only for production compile
  // reports and must remain borrowed/static.
  loom_low_lower_describe_plan_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_describe_plan_callback_t;

typedef struct loom_low_lower_policy_t {
  // Stable policy name used in diagnostics and status messages.
  iree_string_view_t name;
  // Catalog resolving compact diagnostic refs carried by this policy's
  // generated rules and contract fragments.
  const loom_error_catalog_t* error_catalog;
  // Maps source semantic types to target-low register types.
  loom_low_lower_map_type_callback_t map_type;
  // Optionally reports source types accepted by target-low legality because
  // |map_type| can map them to target-low values.
  loom_target_low_legality_type_supported_callback_t source_type_supported;
  // Optionally maps concrete source SSA values to target-low register types
  // when type alone does not determine the target register class.
  loom_low_lower_map_value_callback_t map_value;
  // Optionally maps concrete source SSA values to descriptor register metadata
  // for read-only target contract queries. Missing means table guards that need
  // register mapping cannot match.
  loom_low_lower_map_contract_value_callback_t map_contract_value;
  // Optionally maps source function arguments to non-direct ABI imports.
  loom_low_lower_map_argument_callback_t map_argument;
  // Optionally emits target live-ins or other structural preamble packets.
  loom_low_lower_emit_preamble_callback_t emit_preamble;
  // Optionally emits target entry-block setup packets after ABI imports.
  loom_low_lower_emit_entry_setup_callback_t emit_entry_setup;
  // Optionally materializes target-owned low boundary attrs/layout from the
  // source function signature and mapped low signature.
  loom_low_lower_map_abi_layout_callback_t map_abi_layout;
  // Optionally plans target-specific branch expansion after low blocks exist.
  loom_low_lower_prepare_branch_callback_t prepare_branch;
  // Optionally materializes branch payloads to the exact destination block
  // argument type after the canonical low value has been looked up.
  loom_low_lower_materialize_branch_arg_callback_t materialize_branch_arg;
  // Optionally emits conditional branches that need target-specific structural
  // control packets instead of plain low.cond_br.
  loom_low_lower_emit_cond_branch_callback_t emit_cond_branch;
  // Low declaration import kind for target-bound source imports, or zero when
  // this policy does not lower import declarations.
  loom_low_func_decl_import_kind_t import_decl_kind;
  // Optional table-driven source-op lowering rule sets in selection order. Rule
  // sets may overlap; the first matching rule wins and failed diagnostics use
  // the most-specific rejected candidate.
  loom_low_lower_rule_set_list_t rule_sets;
  // Active contract fragments composed into a dense root index for direct
  // source-op lookup and read-only legality queries.
  const loom_target_contract_binding_t* contract_bindings;
  // Number of active contract fragments.
  uint16_t contract_binding_count;
  // Optional target-owned descriptor-matrix projection used by generated
  // descriptor-matrix contract cases.
  loom_low_lower_descriptor_matrix_t descriptor_matrix;
  // Optional target-owned selector that runs before generated table rules for
  // source ops whose legality is described by contract fragments but whose
  // emission needs target-owned planning.
  loom_low_lower_select_op_callback_t preselect_op;
  // Optional target-owned selector used before a target has table rules.
  loom_low_lower_select_op_callback_t select_op;
  // Optional target-owned source storage demand marker for callback-selected
  // plans. Missing preserves the conservative all-operands behavior.
  loom_low_lower_mark_plan_storage_demands_callback_t mark_plan_storage_demands;
  // Optional target-owned callback plan detail formatter for compile reports.
  loom_low_lower_describe_plan_callback_t describe_plan;
  // Optional target-owned emitter for plans selected by |select_op|.
  loom_low_lower_emit_op_callback_t emit_op;
} loom_low_lower_policy_t;

typedef struct loom_low_lower_policy_registry_entry_t {
  // Target contract-set key that selects |policy|.
  iree_string_view_t contract_set_key;
  // Borrowed lowering policy used for bundles naming |contract_set_key|.
  const loom_low_lower_policy_t* policy;
} loom_low_lower_policy_registry_entry_t;

typedef struct loom_low_lower_policy_registry_t {
  // Borrowed contract-set key to policy table linked into the target package.
  const loom_low_lower_policy_registry_entry_t* entries;
  // Number of rows in |entries|.
  iree_host_size_t entry_count;
} loom_low_lower_policy_registry_t;

// Initializes |out_registry| from a target-owned entry table. The table is
// borrowed and must outlive |out_registry|.
void loom_low_lower_policy_registry_initialize_from_entries(
    loom_low_lower_policy_registry_t* out_registry,
    const loom_low_lower_policy_registry_entry_t* entries,
    iree_host_size_t entry_count);

// Looks up the lowering policy for |contract_set_key|, or returns NULL when no
// row matches. The registry table is target-owned static data; production
// lookup trusts its row ordering and returns the first matching entry.
const loom_low_lower_policy_t* loom_low_lower_policy_registry_lookup(
    const loom_low_lower_policy_registry_t* registry,
    iree_string_view_t contract_set_key);

// Looks up the lowering policy for |bundle|'s target-contract key, or returns
// NULL when no row matches.
const loom_low_lower_policy_t* loom_low_lower_policy_registry_lookup_for_bundle(
    const loom_low_lower_policy_registry_t* registry,
    const loom_target_bundle_t* bundle);

typedef enum loom_low_control_flow_lowering_e {
  // Source control flow has already been lowered to CFG before source-to-low.
  LOOM_LOW_CONTROL_FLOW_LOWERING_CFG = 0,
  // Source-to-low preserves supported structured control as low structured ops.
  LOOM_LOW_CONTROL_FLOW_LOWERING_STRUCTURED_LOW = 1,
} loom_low_control_flow_lowering_t;

typedef struct loom_low_lower_options_t {
  // Module-local target record symbol used by the emitted low function.
  loom_symbol_ref_t target_ref;
  // Target bundle selected for this lowering attempt.
  const loom_target_bundle_t* bundle;
  // Target-owned payload associated with |bundle|, or NULL.
  const void* target_data;
  // Low descriptor registry linked into the current compiler binary.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Optional target-specific legality providers forwarded to source legality.
  loom_target_low_legality_provider_list_t legality_provider_list;
  // Optional source legality feedback diagnostics forwarded to providers.
  loom_target_low_legality_diagnostic_flags_t legality_diagnostic_flags;
  // Target lowering policy for descriptor and type choices.
  const loom_low_lower_policy_t* policy;
  // Borrowed source value facts for |source_function| and |bundle|. Lowering
  // is a pure consumer of facts; callers own acquisition and invalidation.
  loom_value_fact_table_t* fact_table;
  // Structured diagnostic emitter for user legality and lowering failures.
  iree_diagnostic_emitter_t emitter;
  // Maximum number of errors to emit before aborting. Zero means no limit.
  uint32_t max_errors;
  // Control-flow shape expected at the source-to-low boundary.
  loom_low_control_flow_lowering_t control_flow_lowering;
  // Optional arena receiving production tables that must outlive lowering,
  // such as source-derived memory access summaries consumed by packetization.
  iree_arena_allocator_t* table_arena;
  // Optional allocator enabling production source-low report rows.
  iree_allocator_t report_allocator;
} loom_low_lower_options_t;

typedef struct loom_low_lower_result_t {
  // Number of error diagnostics emitted.
  uint32_t error_count;
  // Number of remark diagnostics emitted by legality providers and lowering
  // callbacks.
  uint32_t remark_count;
  // Descriptor set selected by |options.bundle|.
  const loom_low_descriptor_set_t* descriptor_set;
  // Emitted low function op, or NULL when user diagnostics prevented emission.
  loom_op_t* low_func_op;
  // Module-local symbol reference for |low_func_op|.
  loom_symbol_ref_t low_func_ref;
  // Reported number of non-structural source operations selected for lowering.
  uint64_t selected_source_op_count;
  // Reported number of low operations emitted from source operation selections.
  uint64_t emitted_low_op_count;
  // Allocator used for owned source-low report rows.
  iree_allocator_t report_allocator;
  // Owned source-low report rows.
  loom_low_lower_report_row_list_t report_rows;
  // Source-derived memory access summaries for emitted low memory packets.
  loom_low_memory_access_table_t memory_access_table;
} loom_low_lower_result_t;

typedef struct loom_low_lower_resolved_descriptor_t {
  // Descriptor row selected from the active descriptor set.
  const loom_low_descriptor_t* descriptor;
  // Module string ID for the selected descriptor key.
  loom_string_id_t opcode_id;
} loom_low_lower_resolved_descriptor_t;

// Lowers one func.def-like source function into a target-low function in place.
//
// User IR failures are emitted through |options->emitter| and counted in
// |out_result|. The function returns OK in that case and does not emit a low
// function. On success the emitted target-low function preserves the source
// function symbol and replaces the source op at the same module position.
// Infrastructure failures such as malformed options, invalid target symbols,
// or a policy that violates the lowering contract are returned as status
// failures.
iree_status_t loom_low_lower_function(loom_module_t* module,
                                      loom_func_like_t source_function,
                                      const loom_low_lower_options_t* options,
                                      loom_low_lower_result_t* out_result);

// Releases report row storage owned by |result|.
void loom_low_lower_result_deinitialize(loom_low_lower_result_t* result);

// Lowers one target-bound external function declaration into a low.func.decl.
//
// The emitted low declaration preserves the source symbol identity, maps the
// source signature through |options->policy|, and records the policy import
// kind plus the resolved import symbol as the low code symbol.
iree_status_t loom_low_lower_import_declaration(
    loom_module_t* module, loom_func_like_t source_declaration,
    const loom_low_lower_options_t* options,
    loom_low_lower_result_t* out_result);

// Creates a read-only source-to-low target contract query scope for one source
// function.
//
// The scope shares source-to-low's contract adapter: generated rule refs,
// policy map-contract-value callbacks, descriptor-matrix queries, and value
// materializer predicates all observe the same lowering context shape used by
// loom_low_lower_function. Callers must destroy the scope before releasing
// |module| or |options| even though the scope storage itself is arena-owned.
iree_status_t loom_low_lower_source_query_scope_create(
    loom_module_t* module, loom_func_like_t source_function,
    const loom_low_lower_options_t* options, iree_arena_allocator_t* arena,
    loom_low_lower_source_query_scope_t** out_scope);

// Releases analyses owned by |scope|. The scope allocation itself remains owned
// by the arena passed to loom_low_lower_source_query_scope_create.
void loom_low_lower_source_query_scope_destroy(
    loom_low_lower_source_query_scope_t* scope);

// Returns a contract-query callback backed by |scope|.
loom_target_contract_query_callback_t
loom_low_lower_source_query_scope_callback(
    loom_low_lower_source_query_scope_t* scope);

// Returns the active function-local value domain owned by |scope|, or NULL when
// the scoped source function has no body.
const loom_local_value_domain_t* loom_low_lower_source_query_scope_value_domain(
    const loom_low_lower_source_query_scope_t* scope);

// Returns the module being mutated by the current lowering.
loom_module_t* loom_low_lower_context_module(loom_low_lower_context_t* context);

// Returns the builder positioned in the current low block. Only valid while
// emit_preamble, emit_entry_setup, or emit_op callback code is emitting;
// select_op callbacks must not mutate IR.
loom_builder_t* loom_low_lower_context_builder(
    loom_low_lower_context_t* context);

// Returns the source function being lowered.
loom_func_like_t loom_low_lower_context_source_function(
    const loom_low_lower_context_t* context);

// Returns the source function name used in source-to-low diagnostics.
iree_string_view_t loom_low_lower_context_function_name(
    const loom_low_lower_context_t* context);

// Returns the emitted target-low function op, or NULL during planning.
loom_op_t* loom_low_lower_context_low_function(
    const loom_low_lower_context_t* context);

// Returns the number of error diagnostics emitted by the current lowering run.
uint32_t loom_low_lower_context_error_count(
    const loom_low_lower_context_t* context);

// Returns enabled source-to-low diagnostic categories.
loom_target_low_legality_diagnostic_flags_t
loom_low_lower_context_diagnostic_flags(
    const loom_low_lower_context_t* context);

// Returns the selected target bundle.
const loom_target_bundle_t* loom_low_lower_context_bundle(
    const loom_low_lower_context_t* context);

// Returns the module-local target record symbol used by the emitted low
// function.
loom_symbol_ref_t loom_low_lower_context_target_ref(
    const loom_low_lower_context_t* context);

// Returns the selected target bundle key used in generated diagnostics.
iree_string_view_t loom_low_lower_context_target_key(
    const loom_low_lower_context_t* context);

// Returns the selected target export name used in generated diagnostics.
iree_string_view_t loom_low_lower_context_export_name(
    const loom_low_lower_context_t* context);

// Returns the selected target config key used in generated diagnostics.
iree_string_view_t loom_low_lower_context_config_key(
    const loom_low_lower_context_t* context);

// Returns the selected descriptor set.
const loom_low_descriptor_set_t* loom_low_lower_context_descriptor_set(
    const loom_low_lower_context_t* context);

// Returns source value facts computed before planning. The table describes
// the source function being lowered and remains valid only during callbacks.
const loom_value_fact_table_t* loom_low_lower_context_fact_table(
    const loom_low_lower_context_t* context);

// Returns a lazily analyzed view-region table for the source function being
// lowered. The table remains valid only during the current lowering callback.
iree_status_t loom_low_lower_context_view_regions(
    loom_low_lower_context_t* context,
    const loom_view_region_table_t** out_view_regions);

// Returns the number of non-structural source-op lowering plans selected during
// planning. Preamble callbacks may inspect these plans before body emission.
iree_host_size_t loom_low_lower_context_selected_plan_count(
    const loom_low_lower_context_t* context);

// Returns one selected lowering plan view.
loom_low_lower_selected_plan_view_t loom_low_lower_context_selected_plan_view(
    const loom_low_lower_context_t* context, iree_host_size_t index);

// Requires low SSA storage for one source value during plan demand analysis.
// Target-owned mark_plan_storage_demands callbacks use this for each source
// value they may look up while emitting the selected callback plan.
void loom_low_lower_require_source_value_storage(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id);

// Requires low SSA storage for every operand of |source_op|. This is the
// conservative callback-plan fallback used when a target does not provide exact
// storage demands for a selected plan.
void loom_low_lower_require_source_operands_storage(
    loom_low_lower_context_t* context, const loom_op_t* source_op);

// Allocates transient storage from the current lowering arena. The allocation
// remains valid until the current loom_low_lower_function call returns.
iree_status_t loom_low_lower_allocate_scratch_array(
    loom_low_lower_context_t* context, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr);

// Allocates one target-owned selected-plan payload from the current lowering
// arena. The returned storage remains valid until the current
// loom_low_lower_function call returns. Targets should populate the payload in
// place during planning and treat it as immutable during emission.
iree_status_t loom_low_lower_allocate_plan_data(
    loom_low_lower_context_t* context, iree_host_size_t data_length,
    void** out_data);

// Appends a low-only block to the low function being emitted.
//
// This is for target control packets that need a dispatch/restore block with no
// corresponding source block. Source block remapping remains fixed.
iree_status_t loom_low_lower_append_low_block(loom_low_lower_context_t* context,
                                              loom_block_t** out_block);

// Looks up the effective low destination for one source terminator successor.
//
// This accounts for target interpositions registered by
// loom_low_lower_interpose_successor_dest. Callers that are lowering structural
// terminators should prefer this over raw block lookup.
iree_status_t loom_low_lower_lookup_successor_dest(
    loom_low_lower_context_t* context, const loom_op_t* source_terminator,
    uint8_t successor_index, loom_block_t** out_low_dest);

// Maps one source successor payload to low values accepted by |low_dest|.
//
// This accounts for target interpositions and edge-local register-class
// materialization. Structural branch lowering should use this instead of raw
// loom_low_lower_lookup_value loops when forwarding block arguments.
iree_status_t loom_low_lower_remap_successor_args(
    loom_low_lower_context_t* context, const loom_op_t* source_terminator,
    uint8_t successor_index, loom_block_t* low_dest,
    const loom_value_id_t* source_args, uint16_t source_arg_count,
    loom_value_id_t** out_low_args);

// Interposes a low-only destination block on one source successor edge.
//
// The new block receives the same edge payload as the source terminator edge:
// no arguments for cfg.cond_br and the cfg.br payload for cfg.br.
// |out_previous_low_dest| receives the effective destination that the
// interposed block should eventually branch to when it wants to preserve the
// original edge behavior.
iree_status_t loom_low_lower_interpose_successor_dest(
    loom_low_lower_context_t* context, const loom_op_t* source_terminator,
    uint8_t successor_index, loom_block_t* interposed_low_block,
    loom_block_t** out_previous_low_dest);

// Records one target-owned structural branch plan for |source_terminator|.
iree_status_t loom_low_lower_set_branch_plan(loom_low_lower_context_t* context,
                                             const loom_op_t* source_terminator,
                                             loom_low_lower_plan_t plan);

// Looks up the target-owned structural branch plan for |source_terminator|.
bool loom_low_lower_lookup_branch_plan(loom_low_lower_context_t* context,
                                       const loom_op_t* source_terminator,
                                       loom_low_lower_plan_t* out_plan);

// Creates a module-local symbol derived from the emitted low function symbol.
// The result is suitable for target-owned function artifacts. |suffix| is
// appended to the low function name; |index| is then appended in decimal form
// when |append_index| is true.
iree_status_t loom_low_lower_create_function_symbol(
    loom_low_lower_context_t* context, iree_string_view_t suffix,
    bool append_index, uint32_t index, loom_symbol_ref_t* out_symbol_ref);

// Maps |source_type| through the active policy. A policy that rejects a user
// type emits a diagnostic and returns loom_type_none() in |out_low_type|.
iree_status_t loom_low_lower_map_type(loom_low_lower_context_t* context,
                                      const loom_op_t* source_op,
                                      loom_type_t source_type,
                                      loom_type_t* out_low_type);

// Maps |source_value_id|'s type through the active policy. A policy that
// rejects a user value emits a diagnostic and returns loom_type_none() in
// |out_low_type|.
iree_status_t loom_low_lower_map_value(loom_low_lower_context_t* context,
                                       const loom_op_t* source_op,
                                       loom_value_id_t source_value_id,
                                       loom_type_t* out_low_type);

// Looks up the low SSA value already bound to |source_value_id|.
iree_status_t loom_low_lower_lookup_value(loom_low_lower_context_t* context,
                                          loom_value_id_t source_value_id,
                                          loom_value_id_t* out_low_value_id);

// Looks up the emitted low block corresponding to |source_block| in the source
// function currently being lowered.
iree_status_t loom_low_lower_lookup_block(loom_low_lower_context_t* context,
                                          const loom_block_t* source_block,
                                          loom_block_t** out_low_block);

// Binds one source SSA value to the corresponding low SSA value. The source
// value's display name is copied when available.
iree_status_t loom_low_lower_bind_value(loom_low_lower_context_t* context,
                                        loom_value_id_t source_value_id,
                                        loom_value_id_t low_value_id);

// Binds |result_value_id| to the low SSA value already selected for
// |source_value_id|. This is for target callbacks that preserve a source-level
// alias while relying on facts to carry view or offset semantics separately.
iree_status_t loom_low_lower_bind_value_alias(loom_low_lower_context_t* context,
                                              loom_value_id_t source_value_id,
                                              loom_value_id_t result_value_id);

// Marks one source SSA value as intentionally erased by the selected lowering
// rule. This is only for source-level sequencing/control values whose users are
// also lowered away, such as async tokens and groups. Elided values must never
// be consumed as target-low operands.
iree_status_t loom_low_lower_elide_value(loom_low_lower_context_t* context,
                                         loom_value_id_t source_value_id);

// Creates a target-low register type from a descriptor-set register-class ID.
iree_status_t loom_low_lower_make_register_type(
    loom_low_lower_context_t* context, uint16_t reg_class_id,
    uint32_t unit_count, loom_type_t* out_type);

// Resolves a descriptor row from the selected descriptor set.
//
// This is for selectors that already walked a descriptor-set table and selected
// a concrete row. It interns the row's opcode string into the low module
// without repeating the stable-ID lookup.
iree_status_t loom_low_lower_resolve_descriptor_row(
    loom_low_lower_context_t* context, const loom_low_descriptor_t* descriptor,
    loom_low_lower_resolved_descriptor_t* out_descriptor);

// Emits a descriptor-backed low.op from a descriptor row resolved during
// selection.
iree_status_t loom_low_lower_emit_resolved_descriptor_op(
    loom_low_lower_context_t* context,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attrs, const loom_type_t* result_types,
    iree_host_size_t result_count, const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count, loom_location_id_t location,
    loom_op_t** out_op);

// Emits a descriptor-backed low.const from a descriptor row resolved during
// selection.
iree_status_t loom_low_lower_emit_resolved_descriptor_const(
    loom_low_lower_context_t* context,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_named_attr_slice_t attrs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

// Records a source-derived memory summary for an emitted low memory packet.
//
// The row is copied into options.table_arena when provided. Calls are ignored
// when the current lowering run has no table arena, preserving conservative
// descriptor-only scheduling for callers that do not need source precision.
iree_status_t loom_low_lower_record_memory_access_summary(
    loom_low_lower_context_t* context, const loom_op_t* low_op,
    const loom_low_memory_access_summary_t* summary);

// Records a source memory access plan for an emitted low memory packet.
iree_status_t loom_low_lower_record_source_memory_access(
    loom_low_lower_context_t* context, const loom_op_t* low_op,
    const loom_low_source_memory_access_plan_t* source_plan);

// Emits ERR_TARGET_033 for a source value type rejected by the active
// target-low policy.
iree_status_t loom_low_lower_emit_source_type_unsupported(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t field_name, loom_type_t actual_type);

// Emits ERR_TARGET_034 for a CFG branch shape rejected by the active
// target-low policy.
iree_status_t loom_low_lower_emit_branch_constraint(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t branch_constraint);

// Emits ERR_TARGET_035 for a branch condition type rejected by the active
// target-low policy.
iree_status_t loom_low_lower_emit_branch_condition_type_unsupported(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t actual_type, iree_string_view_t type_constraint);

// Emits a generated structured lowering diagnostic.
iree_status_t loom_low_lower_emit_error_ref(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_error_ref_t error_ref, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_LOWER_H_
