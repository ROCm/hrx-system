// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured module compilation reports.

#ifndef LOOM_TARGET_COMPILE_REPORT_H_
#define LOOM_TARGET_COMPILE_REPORT_H_

#include "iree/base/api.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_target_compile_artifact_kind_e {
  // No artifact was produced.
  LOOM_TARGET_COMPILE_ARTIFACT_KIND_NONE = 0,
  // IREE VM bytecode archive artifact.
  LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE = 1,
  // IREE HAL loader executable artifact.
  LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE = 2,
  // Target-native HAL kernel library artifact.
  LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_KERNEL_LIBRARY = 3,
  // Target-native artifact such as ELF, SPIR-V, WASM, or object bytes.
  LOOM_TARGET_COMPILE_ARTIFACT_KIND_TARGET_ARTIFACT = 4,
} loom_target_compile_artifact_kind_t;

typedef uint32_t loom_target_compile_report_detail_flags_t;
enum {
  // No optional report details are populated.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_NONE = 0u,
  // |artifact_size| is populated.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_ARTIFACT_SIZE = 1u << 0,
  // Schedule node, dependency, pressure, and resource summaries are populated.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE = 1u << 1,
  // Allocation assignment, copy, and spill counts are populated.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION = 1u << 2,
  // Target private/local memory estimates are populated.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_MEMORY = 1u << 3,
  // Target emission instruction and code-size summaries are populated.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION = 1u << 4,
  // Per-pressure-class rows were recorded or counted.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS = 1u << 5,
  // Per-spill-plan rows were recorded or counted.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS = 1u << 6,
  // Source-to-target-low selection rows were recorded or counted.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS = 1u << 7,
  // Residual target move causes were recorded or counted.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_MOVE_CAUSES = 1u << 8,
  // Static instruction-mix feature counters were recorded.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_STATIC_INSTRUCTION_MIX = 1u << 9,
  // Target-legalization decision rows were recorded or counted.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_LEGALIZATION_ROWS = 1u << 10,
  // Per-entry native artifact summaries were recorded.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_ENTRIES = 1u << 11,
};

typedef enum loom_target_compile_report_move_cause_e {
  // No residual target move cause was recorded.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_NONE = 0,
  // A low constant packet materialized an immediate value into a register.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_CONSTANT_MATERIALIZATION,
  // A low.copy packet survived allocation and must be emitted as a move.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_COPY,
  // A low.slice packet survived allocation and must be emitted as moves.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_SLICE,
  // A low.concat packet survived allocation and must be emitted as moves.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_CONCAT,
  // A control-flow edge payload must be emitted as moves.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_BRANCH_EDGE,
  // A descriptor packet materialized an operand into a required register bank.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_OPERAND_BANK_MATERIALIZATION,
  // A tied, destructive, or fixed operand constraint required repair moves.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_OPERAND_CONSTRAINT_REPAIR,
  // ABI lowering inserted entry, exit, or call-boundary copies.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_ABI_COPY,
  // Spill or reload materialization inserted target moves.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_SPILL_RELOAD,
  // Partial-register lowering inserted target moves.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_PARTIAL_REGISTER_REPAIR,
  // A residual move could not be assigned a more precise structural cause.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_UNKNOWN,
  // Number of residual target move cause values, including NONE.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_COUNT,
} loom_target_compile_report_move_cause_t;

typedef enum loom_target_compile_report_source_low_selection_kind_e {
  // No source-low selection was recorded.
  LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_NONE = 0,
  // Selection came from a table-driven lowering rule.
  LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_RULE = 1,
  // Selection came from a target-owned callback plan.
  LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_PLAN = 2,
} loom_target_compile_report_source_low_selection_kind_t;

typedef enum loom_target_compile_report_legalization_mode_e {
  // No target-legalization mode was recorded.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_MODE_NONE = 0,
  // Eager legalization may leave unsupported ops for later specialization.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_MODE_EAGER = 1,
  // Final legalization must leave the function accepted by target lowering.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_MODE_FINAL = 2,
} loom_target_compile_report_legalization_mode_t;

typedef enum loom_target_compile_report_legalization_policy_e {
  // No target-legalization policy was recorded.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_POLICY_NONE = 0,
  // Target-native contracts and rewrites are preferred over reference fallback.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_POLICY_PREFER_NATIVE = 1,
  // Target-native rewrites are skipped in favor of reference rewrites.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_POLICY_REFERENCE_ONLY = 2,
  // Reference rewrites are rejected so native coverage gaps stay visible.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_POLICY_REQUIRE_NATIVE = 3,
} loom_target_compile_report_legalization_policy_t;

typedef enum loom_target_compile_report_legalization_action_e {
  // No target-legalization action was recorded.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_NONE = 0,
  // The target contract already accepts the source op.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_LEGAL = 1,
  // A legalizer rewrote or erased the source op.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REWRITTEN = 2,
  // A legalizer recognized the source op but deferred it to a later phase.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_DEFERRED = 3,
  // The source op violates the source contract required before legalization.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REJECT_INVALID_IR = 4,
  // The source op is recognized but unsupported by the final target contract.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REJECT_UNSUPPORTED_FINAL = 5,
  // No composed legalizer had an opinion about the unsupported source op.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_UNHANDLED = 6,
} loom_target_compile_report_legalization_action_t;

typedef enum loom_target_compile_report_legalization_outcome_e {
  // No terminal target-legalization outcome was recorded.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_NONE = 0,
  // The selected target contract already accepts the source op.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_ALREADY_LEGAL = 1,
  // A target-specific legalizer rewrote or erased the source op.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_TARGET_REWRITE = 2,
  // A target-independent reference legalizer rewrote or erased the source op.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_REFERENCE_FALLBACK = 3,
  // A legalizer intentionally left the source op for a later phase.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_DEFERRED = 4,
  // The source op violates the source contract required before legalization.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_REJECT_INVALID_IR = 5,
  // The source op is recognized but unsupported by the final target contract.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_REJECT_UNSUPPORTED = 6,
  // No composed legalizer had an opinion about the unsupported source op.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_UNHANDLED = 7,
} loom_target_compile_report_legalization_outcome_t;

typedef enum loom_target_compile_report_contract_outcome_e {
  // No target-contract query outcome was recorded.
  LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_NONE = 0,
  // No linked target-contract fragment had an opinion about the op.
  LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_UNHANDLED = 1,
  // The op is already legal for the selected target contract.
  LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_LEGAL = 2,
  // The op family is recognized but unsupported by the selected target.
  LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_UNSUPPORTED = 3,
  // The op violates the source contract required before target selection.
  LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_INVALID_IR = 4,
} loom_target_compile_report_contract_outcome_t;

typedef enum loom_target_compile_report_legalizer_strategy_e {
  // No target-legalizer strategy was recorded.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_NONE = 0,
  // Target-specific rewrite intended to reach a native target contract.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_TARGET = 1,
  // Target-independent reference rewrite used as a portable fallback.
  LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_REFERENCE = 2,
} loom_target_compile_report_legalizer_strategy_t;

// Residual move-cause counters for one category.
typedef struct loom_target_compile_report_move_cause_counts_t {
  // Number of target packets attributed to this cause.
  uint64_t packet_count;
  // Number of register-unit moves attributed to this cause.
  uint64_t unit_count;
} loom_target_compile_report_move_cause_counts_t;

// Static feature counters for descriptor-backed low schedule nodes.
//
// These counters are compile-time proxies derived from descriptor semantic
// tags, schedule classes, and schedule resources. They are intentionally
// separate from measured HAL profiling counters and may overlap: for example an
// AMDGPU global atomic packet is both global memory and atomic.
typedef struct loom_target_compile_report_static_instruction_mix_t {
  // Descriptor-backed schedule nodes inspected for feature classification.
  uint64_t descriptor_count;
  // Descriptor-backed schedule nodes with no recognized static feature.
  uint64_t unknown_count;
  // Descriptor-backed nodes that use scalar ALU resources.
  uint64_t scalar_alu_count;
  // Descriptor-backed nodes that use vector ALU resources.
  uint64_t vector_alu_count;
  // Descriptor-backed nodes that use matrix/tensor-core-like resources.
  uint64_t matrix_count;
  // Descriptor-backed nodes identified as MFMA-like matrix instructions.
  uint64_t mfma_count;
  // Descriptor-backed nodes identified as WMMA-like matrix instructions.
  uint64_t wmma_count;
  // Descriptor-backed nodes identified as dot-product instructions.
  uint64_t dot_count;
  // Descriptor-backed nodes identified as global or vector-memory operations.
  uint64_t global_memory_count;
  // Descriptor-backed nodes identified as local/shared/workgroup memory ops.
  uint64_t local_memory_count;
  // Descriptor-backed nodes identified as scalar-memory operations.
  uint64_t scalar_memory_count;
  // Descriptor-backed nodes identified as generic memory operations.
  uint64_t generic_memory_count;
  // Descriptor-backed nodes identified as atomic memory operations.
  uint64_t atomic_count;
  // Descriptor-backed nodes identified as branch, return, or call control flow.
  uint64_t branch_count;
  // Descriptor-backed nodes identified as barrier or synchronization packets.
  uint64_t barrier_count;
  // Descriptor-backed nodes identified as other control packets.
  uint64_t control_count;
  // Descriptor-backed nodes identified as numeric conversion packets.
  uint64_t conversion_count;
  // Descriptor-backed nodes identified as cache maintenance or prefetch
  // packets.
  uint64_t cache_count;
  // Descriptor-backed nodes identified as register moves or copies.
  uint64_t register_move_count;
} loom_target_compile_report_static_instruction_mix_t;

// One emitted artifact entry summary in a compile report.
typedef struct loom_target_compile_report_entry_t {
  // Target artifact function symbol emitted for this entry.
  iree_string_view_t function_name;
  // Source or target-low function symbol that produced this entry.
  iree_string_view_t source_function_name;
  // Resolved target record name selected for this entry.
  iree_string_view_t target_bundle_name;
  // Resolved target snapshot name selected for this entry.
  iree_string_view_t target_snapshot_name;
  // Resolved target export-plan name selected for this entry.
  iree_string_view_t target_export_name;
  // Target artifact export symbol requested by the export plan, if any.
  iree_string_view_t target_export_symbol;
  // Resolved target config name selected for this entry.
  iree_string_view_t target_config_name;
  // Optional detail flags indicating which summary groups are populated.
  loom_target_compile_report_detail_flags_t detail_flags;
  // Number of low schedule nodes before target emission.
  uint64_t schedule_node_count;
  // Number of low schedule nodes in scheduled order.
  uint64_t scheduled_node_count;
  // Number of low schedule dependency edges.
  uint64_t schedule_dependency_count;
  // Number of descriptor resource-use records.
  uint64_t schedule_resource_use_count;
  // Number of required schedule hazard gaps.
  uint64_t schedule_hazard_gap_count;
  // Number of schedule model-quality summary records.
  uint64_t schedule_model_summary_count;
  // Number of register-pressure summary records.
  uint64_t register_pressure_summary_count;
  // Maximum boundary-live register units observed for this entry.
  uint64_t register_pressure_peak_live_units;
  // Number of allocation assignments.
  uint64_t allocation_assignment_count;
  // Number of values assigned to spill slots.
  uint64_t allocation_spill_count;
  // Number of synthetic spill plans.
  uint64_t allocation_spill_plan_count;
  // Number of low.copy ops coalesced away by allocation.
  uint64_t allocation_coalesced_copy_count;
  // Number of low.copy ops that must remain materialized.
  uint64_t allocation_materialized_copy_count;
  // Number of target instructions or bytecode opcodes emitted.
  uint64_t emitted_instruction_count;
  // Number of semantic target code bytes before target-local padding.
  uint64_t emitted_code_byte_count;
  // Number of target code storage bytes including target-local padding.
  uint64_t emitted_code_storage_byte_count;
  // Estimated target private memory bytes.
  uint64_t private_memory_bytes;
  // Estimated target local/shared memory bytes.
  uint64_t local_memory_bytes;
  // Residual target move counts indexed by
  // loom_target_compile_report_move_cause_t.
  loom_target_compile_report_move_cause_counts_t
      move_causes[LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_COUNT];
  // Static descriptor-backed instruction-mix feature counters.
  loom_target_compile_report_static_instruction_mix_t static_instruction_mix;
  // Number of detailed register-pressure rows copied for this entry.
  iree_host_size_t pressure_row_count;
  // Number of detailed spill rows copied for this entry.
  iree_host_size_t spill_row_count;
} loom_target_compile_report_entry_t;

// One register-pressure peak row in a compile report.
typedef struct loom_target_compile_report_pressure_row_t {
  // Target artifact function symbol containing this pressure peak.
  iree_string_view_t function_name;
  // Register class name for register values, or an empty string otherwise.
  iree_string_view_t register_class;
  // Numeric Loom type kind for the pressure class.
  uint32_t type_kind;
  // Numeric Loom scalar element type for the pressure class.
  uint32_t element_type;
  // Maximum boundary-live units observed for the class.
  uint64_t peak_live_units;
  // Maximum simultaneously live values observed at the same point.
  uint64_t peak_live_values;
  // Program point associated with the peak.
  uint32_t peak_point;
  // Block label containing the peak, or a fallback diagnostic name.
  iree_string_view_t peak_block_name;
  // Operation name after which the peak was observed, or a boundary marker.
  iree_string_view_t peak_operation_name;
} loom_target_compile_report_pressure_row_t;

// One predicted spill row in a compile report.
typedef struct loom_target_compile_report_spill_row_t {
  // Target artifact function symbol containing this spill plan.
  iree_string_view_t function_name;
  // SSA value name represented by the spilled assignment.
  iree_string_view_t value_name;
  // Register class name for the spilled value.
  iree_string_view_t register_class;
  // Numeric Loom type kind for the spilled value class.
  uint32_t type_kind;
  // Numeric Loom scalar element type for the spilled value class.
  uint32_t element_type;
  // Allocation assignment index associated with this spill.
  uint32_t assignment_index;
  // Spill slot ordinal assigned to the interval.
  uint32_t slot_index;
  // Spill storage-space name.
  iree_string_view_t slot_space;
  // Slot size in bytes.
  uint64_t byte_size;
  // Required slot alignment in bytes.
  uint64_t byte_alignment;
  // Predicted stores needed by the current synthetic spill plan.
  uint64_t store_count;
  // Predicted operand-use reloads in the current synthetic spill plan.
  uint64_t reload_count;
} loom_target_compile_report_spill_row_t;

// One source-to-target-low selection row copied into a compile report.
typedef struct loom_target_compile_report_source_low_row_t {
  // Source function symbol containing the lowered source operation.
  iree_string_view_t function_name;
  // Source operation mnemonic lowered by this row.
  iree_string_view_t source_op_name;
  // Numeric source operation kind lowered by this row.
  uint32_t source_op_kind;
  // Selection mechanism used for this source operation.
  loom_target_compile_report_source_low_selection_kind_t selection_kind;
  // Policy rule-set ordinal for table-driven rules, or UINT16_MAX otherwise.
  uint16_t rule_set_index;
  // Rule-table ordinal inside |rule_set_index|, or UINT16_MAX otherwise.
  uint16_t rule_index;
  // Target-owned plan id for callback selections, or UINT64_MAX otherwise.
  uint64_t plan_id;
  // Stable target-owned description of the selected plan variant, if any.
  iree_string_view_t plan_detail;
  // First stable low descriptor id emitted by a table rule, or none for plans.
  uint64_t descriptor_id;
  // Number of low operations emitted for this source operation.
  uint32_t emitted_low_op_count;
} loom_target_compile_report_source_low_row_t;

// One target-legalization decision row copied into a compile report.
typedef struct loom_target_compile_report_legalization_row_t {
  // Source function symbol containing the legalized source operation.
  iree_string_view_t function_name;
  // Source operation mnemonic considered by target legalization.
  iree_string_view_t source_op_name;
  // Numeric source operation kind considered by target legalization.
  uint32_t source_op_kind;
  // Target bundle selected for the containing function.
  iree_string_view_t target_bundle_name;
  // Target config selected for the containing function, if any.
  iree_string_view_t target_config_name;
  // Stable provider name for the legalizer that decided this row, if any.
  iree_string_view_t legalizer_name;
  // Strategy for the deciding legalizer, if any.
  loom_target_compile_report_legalizer_strategy_t legalizer_strategy;
  // Target-legalization phase in which the decision was made.
  loom_target_compile_report_legalization_mode_t mode;
  // Target-legalization strategy policy in effect for this decision.
  loom_target_compile_report_legalization_policy_t policy;
  // Legalization decision recorded for this source operation.
  loom_target_compile_report_legalization_action_t action;
  // Terminal outcome after applying the selected legalization action.
  loom_target_compile_report_legalization_outcome_t legalization_outcome;
  // Read-only target-contract query outcome observed before rewriting.
  loom_target_compile_report_contract_outcome_t contract_outcome;
  // Active target-contract fragment binding ordinal, or UINT16_MAX.
  uint16_t binding_index;
  // Composed target-contract case ordinal, or UINT16_MAX.
  uint16_t case_index;
  // Policy rule-set ordinal selected or rejected, or UINT16_MAX.
  uint16_t rule_set_index;
  // Policy rule ordinal selected or rejected, or UINT16_MAX.
  uint16_t rule_index;
  // Diagnostic row ordinal retained by the rejected rule, or UINT16_MAX.
  uint16_t diagnostic_index;
  // Low descriptor stable id selected by the accepted rule, or UINT64_MAX.
  uint64_t descriptor_id;
  // Compact target-independent rejection flags.
  uint32_t source_rejection_bits;
  // Optional target-independent rejection detail enum.
  uint32_t source_rejection_detail;
  // Compact target-owned rejection flags.
  uint32_t target_rejection_bits;
  // Target feature bits missing from the selected bundle.
  uint32_t missing_feature_bits;
  // Value fact categories missing for the selected rule.
  uint32_t missing_fact_bits;
  // Operations created by the deciding legalizer.
  uint64_t created_op_count;
  // Operations erased by the deciding legalizer.
  uint64_t erased_op_count;
} loom_target_compile_report_legalization_row_t;

// Linked storage block for homogeneous compile report detail rows.
//
// Row payloads are stored immediately after this header. Blocks are allocator
// owned by the report that contains the list, and their payload row type is
// determined by the report field that references the list.
typedef struct loom_target_compile_report_vec_t {
  // Next row block in allocation order, or NULL for the final block.
  struct loom_target_compile_report_vec_t* next;
  // Number of rows populated in this block.
  iree_host_size_t count;
  // Maximum number of rows that fit in this block.
  iree_host_size_t capacity;
} loom_target_compile_report_vec_t;

// Owned linked list of homogeneous compile report detail rows.
typedef struct loom_target_compile_report_row_list_t {
  // First row storage block, or NULL when empty.
  loom_target_compile_report_vec_t* head;
  // Last row storage block, or NULL when empty.
  loom_target_compile_report_vec_t* tail;
  // Total number of rows stored across all blocks.
  iree_host_size_t count;
} loom_target_compile_report_row_list_t;

// Returns mutable row storage for |vec|.
static inline void* loom_target_compile_report_vec_rows(
    loom_target_compile_report_vec_t* vec) {
  return (void*)(vec + 1);
}

// Returns immutable row storage for |vec|.
static inline const void* loom_target_compile_report_vec_const_rows(
    const loom_target_compile_report_vec_t* vec) {
  return (const void*)(vec + 1);
}

// Structured feedback from one module-to-artifact compilation.
//
// Reports borrow every string view from the compiled module, target records,
// compile options, backend tables, or artifact storage. Detail row lists are
// owned by the report and allocated from |allocator| as rows are recorded.
// Consumers that need a report to outlive those string owners must copy the
// strings before releasing the module or candidate.
typedef struct loom_target_compile_report_t {
  // Host allocator used for owned row storage.
  iree_allocator_t allocator;
  // Artifact kind requested or produced by compilation.
  loom_target_compile_artifact_kind_t artifact_kind;
  // Terminal status code observed by compilation.
  iree_status_code_t status_code;
  // Detail categories requested by the caller. Producers use this to avoid
  // detail-only work in summary mode.
  loom_target_compile_report_detail_flags_t requested_detail_flags;
  // Optional detail flags indicating which numeric summaries are populated.
  loom_target_compile_report_detail_flags_t detail_flags;
  // Target artifact function symbol when exactly one entry is described.
  iree_string_view_t function_name;
  // VM module name requested for archive emission.
  iree_string_view_t module_name;
  // Execution or codegen backend name that produced the candidate, if any.
  iree_string_view_t backend_name;
  // Target family name selected by the backend, if any.
  iree_string_view_t target_family_name;
  // Backend-facing target key selected for compilation, if any.
  iree_string_view_t target_key;
  // Resolved target record name selected for compilation, if any.
  iree_string_view_t target_bundle_name;
  // Resolved target snapshot name selected for compilation, if any.
  iree_string_view_t target_snapshot_name;
  // Resolved target export-plan name selected for compilation, if any.
  iree_string_view_t target_export_name;
  // Target artifact export symbol requested by the selected export plan, if
  // any.
  iree_string_view_t target_export_symbol;
  // Resolved target config name selected for compilation, if any.
  iree_string_view_t target_config_name;
  // Low function symbol produced or selected after lowering, if any.
  iree_string_view_t lowered_symbol;
  // HAL executable format string, if a HAL artifact was produced.
  iree_string_view_t executable_format;
  // Number of bytes in the produced artifact.
  uint64_t artifact_size;
  // Number of low schedule nodes before target emission.
  uint64_t schedule_node_count;
  // Number of low schedule nodes in scheduled order.
  uint64_t scheduled_node_count;
  // Number of low schedule dependency edges.
  uint64_t schedule_dependency_count;
  // Number of descriptor resource-use records.
  uint64_t schedule_resource_use_count;
  // Number of required schedule hazard gaps.
  uint64_t schedule_hazard_gap_count;
  // Number of schedule model-quality summary records.
  uint64_t schedule_model_summary_count;
  // Number of register-pressure summary records.
  uint64_t register_pressure_summary_count;
  // Maximum boundary-live register units observed across pressure summaries.
  uint64_t register_pressure_peak_live_units;
  // Number of allocation assignments.
  uint64_t allocation_assignment_count;
  // Number of values assigned to spill slots.
  uint64_t allocation_spill_count;
  // Number of synthetic spill plans.
  uint64_t allocation_spill_plan_count;
  // Number of low.copy ops coalesced away by allocation.
  uint64_t allocation_coalesced_copy_count;
  // Number of low.copy ops that must remain materialized.
  uint64_t allocation_materialized_copy_count;
  // Number of target instructions or bytecode opcodes emitted.
  uint64_t emitted_instruction_count;
  // Number of semantic target code bytes before target-local padding.
  uint64_t emitted_code_byte_count;
  // Number of target code storage bytes including target-local padding.
  uint64_t emitted_code_storage_byte_count;
  // Number of source operations selected during source-to-low lowering.
  uint64_t source_low_selected_op_count;
  // Number of low operations emitted during source-to-low lowering.
  uint64_t source_low_emitted_op_count;
  // Number of source ops already accepted by target legalization.
  uint64_t target_legalization_legal_op_count;
  // Number of source ops rewritten by target legalization.
  uint64_t target_legalization_rewritten_op_count;
  // Number of target-specific native-path rewrites.
  uint64_t target_legalization_target_rewritten_op_count;
  // Number of portable reference fallback rewrites.
  uint64_t target_legalization_reference_rewritten_op_count;
  // Number of source ops deferred by target legalization.
  uint64_t target_legalization_deferred_op_count;
  // Number of invalid source ops observed by target legalization.
  uint64_t target_legalization_invalid_ir_op_count;
  // Number of final unsupported source ops observed by target legalization.
  uint64_t target_legalization_unsupported_op_count;
  // Number of unsupported source ops with no legalizer opinion.
  uint64_t target_legalization_unhandled_op_count;
  // Residual target move counts indexed by
  // loom_target_compile_report_move_cause_t.
  loom_target_compile_report_move_cause_counts_t
      move_causes[LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_COUNT];
  // Static descriptor-backed instruction-mix feature counters.
  loom_target_compile_report_static_instruction_mix_t static_instruction_mix;
  // Owned emitted artifact entry summary rows.
  loom_target_compile_report_row_list_t entry_rows;
  // Owned register-pressure peak rows.
  loom_target_compile_report_row_list_t pressure_rows;
  // Owned predicted spill rows.
  loom_target_compile_report_row_list_t spill_rows;
  // Owned source-to-low selection rows.
  loom_target_compile_report_row_list_t source_low_rows;
  // Owned target-legalization decision rows.
  loom_target_compile_report_row_list_t target_legalization_rows;
  // Estimated target private memory bytes.
  uint64_t private_memory_bytes;
  // Estimated target local/shared memory bytes.
  uint64_t local_memory_bytes;
} loom_target_compile_report_t;

// Initializes an empty compile report using |allocator| for row storage.
void loom_target_compile_report_initialize(
    loom_target_compile_report_t* out_report, iree_allocator_t allocator);

// Returns true when |report| requests all |detail_flags|.
static inline bool loom_target_compile_report_wants_details(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_detail_flags_t detail_flags) {
  return report != NULL &&
         iree_all_bits_set(report->requested_detail_flags, detail_flags);
}

// Releases row storage owned by |report| and resets it to zero.
void loom_target_compile_report_deinitialize(
    loom_target_compile_report_t* report);

// Initializes |out_target| as a deep copy of |source| using |allocator| for
// owned row storage. String views remain borrowed from the same owners
// referenced by |source|.
iree_status_t loom_target_compile_report_clone(
    const loom_target_compile_report_t* source, iree_allocator_t allocator,
    loom_target_compile_report_t* out_target);

// Initializes a zeroed report only when no details have been requested or
// populated yet. Artifact emitters use this to support direct zeroed-report
// callers without overwriting caller-selected row storage or pass-phase rows
// already appended by a compile pipeline.
void loom_target_compile_report_initialize_if_empty(
    loom_target_compile_report_t* report, iree_allocator_t allocator);

// Records a terminal status code in |report|.
void loom_target_compile_report_record_status(
    loom_target_compile_report_t* report, iree_status_code_t status_code);

// Records the target bundle selected for compilation.
void loom_target_compile_report_record_target_bundle(
    loom_target_compile_report_t* report, const loom_target_bundle_t* bundle);

// Records the produced artifact byte size in |report|.
void loom_target_compile_report_record_artifact_size(
    loom_target_compile_report_t* report, uint64_t artifact_size);

// Records target-low schedule summary counts in |report|.
void loom_target_compile_report_record_schedule(
    loom_target_compile_report_t* report, uint64_t node_count,
    uint64_t scheduled_node_count, uint64_t dependency_count,
    uint64_t resource_use_count, uint64_t hazard_gap_count,
    uint64_t model_summary_count, uint64_t pressure_summary_count,
    uint64_t peak_live_units);

// Records target-low allocation summary counts in |report|.
void loom_target_compile_report_record_allocation(
    loom_target_compile_report_t* report, uint64_t assignment_count,
    uint64_t spill_count, uint64_t spill_plan_count,
    uint64_t coalesced_copy_count, uint64_t materialized_copy_count);

// Records target move materialization attributed to one residual move cause.
void loom_target_compile_report_record_move_cause(
    loom_target_compile_report_t* report,
    loom_target_compile_report_move_cause_t cause, uint64_t packet_count,
    uint64_t unit_count);

// Records static descriptor-backed instruction-mix feature counters.
void loom_target_compile_report_record_static_instruction_mix(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_static_instruction_mix_t* mix);

// Records target emission instruction and code-size summary counts in |report|.
void loom_target_compile_report_record_emission(
    loom_target_compile_report_t* report, uint64_t instruction_count,
    uint64_t code_byte_count, uint64_t code_storage_byte_count);

// Records target memory estimates in |report|.
void loom_target_compile_report_record_memory(
    loom_target_compile_report_t* report, uint64_t private_memory_bytes,
    uint64_t local_memory_bytes);

// Records one emitted artifact entry and copies its detailed pressure and spill
// rows into |report|. String views remain borrowed from |entry_report|'s
// original owners.
iree_status_t loom_target_compile_report_record_entry_report(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_t* entry_report);

// Records one pressure row.
iree_status_t loom_target_compile_report_record_pressure_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_pressure_row_t* row);

// Records one spill row.
iree_status_t loom_target_compile_report_record_spill_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_spill_row_t* row);

// Records one source-low row.
iree_status_t loom_target_compile_report_record_source_low_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_row_t* row);

// Records one target-legalization decision for summary counters without
// materializing a detailed row.
void loom_target_compile_report_record_legalization_summary(
    loom_target_compile_report_t* report,
    loom_target_compile_report_legalization_action_t action,
    loom_target_compile_report_legalizer_strategy_t legalizer_strategy);

// Records one target-legalization row.
iree_status_t loom_target_compile_report_record_legalization_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_legalization_row_t* row);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_COMPILE_REPORT_H_
