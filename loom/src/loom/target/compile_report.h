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
  // Per-allocation-failure rows were recorded or counted.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_FAILURE_ROWS = 1u << 12,
  // Final target resource and occupancy summaries were recorded.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_RESOURCES = 1u << 13,
  // Target math-legalization recipe rows were recorded or counted.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_MATH_LEGALIZATION_ROWS = 1u << 14,
  // Per-pressure-peak origin contribution rows were recorded or counted.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS = 1u << 15,
  // Consecutive low-schedule band rows were recorded or counted.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS = 1u << 16,
  // Per-register-class allocation high-water rows were recorded or counted.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_HIGH_WATER_ROWS = 1u << 17,
  // Target wait-planning summaries or rows were recorded.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN = 1u << 18,
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

typedef enum loom_target_compile_report_allocation_failure_blocking_kind_e {
  // No specific blocking constraint was recorded.
  LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_UNKNOWN = 0,
  // The failing interval itself is wider than the register-class budget.
  LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_INTERVAL_EXCEEDS_BUDGET =
      1,
  // A live assignment occupies a candidate location and cannot be evicted.
  LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_ACTIVE_ASSIGNMENT = 2,
  // A fixed value, reserved range, or storage lease blocks a candidate
  // location.
  LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_LOCATION_CONSTRAINT =
      3,
  // The allocator scanned candidate locations without finding a legal
  // placement or a more specific blocking constraint.
  LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_NO_ASSIGNABLE_LOCATION =
      4,
} loom_target_compile_report_allocation_failure_blocking_kind_t;

typedef enum loom_target_compile_report_pressure_origin_kind_e {
  // No specific pressure origin was classified.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_UNKNOWN = 0,
  // Function, block, or region argument live across the peak.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_BLOCK_ARGUMENT = 1,
  // Target-low constant materialization value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONSTANT = 2,
  // Target-low copy value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_COPY = 3,
  // Target-low slice value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_SLICE = 4,
  // Target-low concat value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONCAT = 5,
  // Target-low storage address or storage view value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_STORAGE = 6,
  // Target-low spill or reload value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_SPILL_RELOAD = 7,
  // Descriptor-backed scalar ALU value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_SCALAR_ALU = 8,
  // Descriptor-backed vector ALU value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_VECTOR_ALU = 9,
  // Descriptor-backed matrix or tensor-core-like value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_MATRIX = 10,
  // Descriptor-backed dot-product value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_DOT = 11,
  // Descriptor-backed global or vector-memory value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_GLOBAL_MEMORY = 12,
  // Descriptor-backed local, LDS, or workgroup-memory value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_LOCAL_MEMORY = 13,
  // Descriptor-backed scalar-memory value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_SCALAR_MEMORY = 14,
  // Descriptor-backed generic memory value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_GENERIC_MEMORY = 15,
  // Descriptor-backed control-flow value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONTROL = 16,
  // Descriptor-backed barrier or synchronization value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_BARRIER = 17,
  // Descriptor-backed numeric conversion value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONVERSION = 18,
  // Descriptor-backed register move or repair value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_REGISTER_MOVE = 19,
  // Descriptor-backed cache or prefetch value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CACHE = 20,
  // Operation-backed value not covered by a more specific origin.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_OPERATION = 21,
} loom_target_compile_report_pressure_origin_kind_t;

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

typedef enum loom_target_compile_report_math_action_e {
  // No target math-legalization action was recorded.
  LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_NONE = 0,
  // A target math policy rewrote the source op through a recipe.
  LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_REWRITTEN = 1,
  // A target math policy rejected the source op.
  LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_REJECTED = 2,
  // No target math policy was available for the source op.
  LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_MISSING_POLICY = 3,
  // The selected target math recipe was not implemented by the compiler.
  LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_MISSING_RECIPE = 4,
} loom_target_compile_report_math_action_t;

// Residual move-cause counters for one category.
typedef struct loom_target_compile_report_move_cause_counts_t {
  // Number of target packets attributed to this cause.
  uint64_t packet_count;
  // Number of register-unit moves attributed to this cause.
  uint64_t unit_count;
} loom_target_compile_report_move_cause_counts_t;

// Static feature counters for low packets that survive target emission.
//
// These counters are compile-time proxies derived from descriptor semantic
// tags, schedule classes, schedule resources, and structural low terminators.
// They are intentionally separate from measured HAL profiling counters and may
// overlap: for example an AMDGPU global atomic packet is both global memory and
// atomic. Byte counts are static descriptor-effect widths for one issue of the
// emitted low stream, not evaluated dynamic workload traffic.
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
  // Descriptor-backed AMDGPU global_load-family memory instructions.
  uint64_t global_load_count;
  // Descriptor-backed AMDGPU global_store-family memory instructions.
  uint64_t global_store_count;
  // Descriptor-backed AMDGPU buffer_load-family memory instructions.
  uint64_t buffer_load_count;
  // Descriptor-backed AMDGPU buffer_store-family memory instructions.
  uint64_t buffer_store_count;
  // Descriptor-backed AMDGPU flat-memory instructions.
  uint64_t flat_memory_count;
  // Descriptor-backed nodes identified as local/shared/workgroup memory ops.
  uint64_t local_memory_count;
  // Descriptor-backed nodes identified as scalar-memory operations.
  uint64_t scalar_memory_count;
  // Descriptor-backed nodes identified as generic memory operations.
  uint64_t generic_memory_count;
  // Descriptor memory-effect reads with zero or non-byte-aligned widths.
  uint64_t memory_read_unknown_width_count;
  // Descriptor memory-effect writes with zero or non-byte-aligned widths.
  uint64_t memory_write_unknown_width_count;
  // Static bytes read by descriptor memory effects across all memory spaces.
  uint64_t memory_read_byte_count;
  // Static bytes written by descriptor memory effects across all memory spaces.
  uint64_t memory_write_byte_count;
  // Static bytes read by global_load-family descriptor effects.
  uint64_t global_load_byte_count;
  // Static bytes written by global_store-family descriptor effects.
  uint64_t global_store_byte_count;
  // Static bytes read by buffer_load-family descriptor effects.
  uint64_t buffer_load_byte_count;
  // Static bytes written by buffer_store-family descriptor effects.
  uint64_t buffer_store_byte_count;
  // Static bytes read by flat-memory descriptor effects.
  uint64_t flat_read_byte_count;
  // Static bytes written by flat-memory descriptor effects.
  uint64_t flat_write_byte_count;
  // Static bytes read by local/shared/workgroup descriptor effects.
  uint64_t local_read_byte_count;
  // Static bytes written by local/shared/workgroup descriptor effects.
  uint64_t local_write_byte_count;
  // Static bytes read by scalar-memory descriptor effects.
  uint64_t scalar_read_byte_count;
  // Static bytes written by scalar-memory descriptor effects.
  uint64_t scalar_write_byte_count;
  // Static bytes read by memory effects without a specific packet family.
  uint64_t unclassified_read_byte_count;
  // Static bytes written by memory effects without a specific packet family.
  uint64_t unclassified_write_byte_count;
  // Descriptor-backed nodes identified as atomic memory operations.
  uint64_t atomic_count;
  // Low packets identified as branch, return, or call control flow.
  uint64_t branch_count;
  // Descriptor-backed nodes identified as barrier or synchronization packets.
  uint64_t barrier_count;
  // Low packets identified as control flow or other control packets.
  uint64_t control_count;
  // Descriptor-backed nodes identified as numeric conversion packets.
  uint64_t conversion_count;
  // Descriptor-backed nodes identified as cache maintenance or prefetch
  // packets.
  uint64_t cache_count;
  // Descriptor-backed nodes identified as register moves or copies.
  uint64_t register_move_count;
} loom_target_compile_report_static_instruction_mix_t;

// Register-class pressure peak retained for summary target-resource reporting.
typedef struct loom_target_compile_report_pressure_summary_t {
  // Stable target register class counted by |peak_live_units|.
  iree_string_view_t register_class;
  // Peak live units observed for |register_class|.
  uint64_t peak_live_units;
} loom_target_compile_report_pressure_summary_t;

// Final target resource and occupancy summary for one emitted entry.
typedef struct loom_target_compile_report_target_resources_t {
  // Stable target register class counted by |scalar_register_count|.
  iree_string_view_t scalar_register_class;
  // Final scalar register units declared by target metadata.
  uint64_t scalar_register_count;
  // Peak live units observed for |scalar_register_class| before final target
  // metadata rounding and hidden target resources.
  uint64_t scalar_pressure_peak_live_units;
  // Extra final scalar register units beyond |scalar_pressure_peak_live_units|.
  uint64_t scalar_register_overhead_units;
  // Stable target register class counted by |vector_register_count|.
  iree_string_view_t vector_register_class;
  // Final vector register units declared by target metadata.
  uint64_t vector_register_count;
  // Peak live units observed for |vector_register_class| before final target
  // metadata rounding and hidden target resources.
  uint64_t vector_pressure_peak_live_units;
  // Extra final vector register units beyond |vector_pressure_peak_live_units|.
  uint64_t vector_register_overhead_units;
  // Target subgroup width in lanes.
  uint32_t subgroup_size;
  // Maximum resident subgroups per SIMD modeled for the target.
  uint32_t max_subgroups_per_simd;
  // Estimated resident subgroups per SIMD after final target resources.
  uint32_t resident_subgroups_per_simd;
  // Estimated final occupancy as a percentage of |max_subgroups_per_simd|.
  uint32_t occupancy_percent;
  // Stable resource name limiting final occupancy, or "max_waves".
  iree_string_view_t limiting_resource;
} loom_target_compile_report_target_resources_t;

// Wait-counter planning summary for target packets that survive emission.
typedef struct loom_target_compile_report_wait_plan_t {
  // Number of wait-counter actions recorded by target planning.
  uint64_t action_count;
  // Number of wait-counter actions already present in the low stream.
  uint64_t explicit_action_count;
  // Number of wait-counter actions inserted by target planning.
  uint64_t planned_action_count;
  // Number of wait actions that drain all outstanding packets.
  uint64_t full_drain_count;
  // Number of wait actions that leave younger packets outstanding.
  uint64_t partial_wait_count;
  // Maximum outstanding packet count observed before any wait action.
  uint64_t max_outstanding_before;
  // Maximum outstanding packet count observed before a full-drain action.
  uint64_t max_full_drain_outstanding_before;
} loom_target_compile_report_wait_plan_t;

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
  // Number of target storage-lease records.
  uint64_t allocation_storage_lease_count;
  // Number of assignment-backed target storage-lease instances.
  uint64_t allocation_storage_lease_instance_count;
  // Number of allocator-requested storage release actions.
  uint64_t allocation_storage_release_action_count;
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
  // Static descriptor-backed instruction and effect feature counters.
  loom_target_compile_report_static_instruction_mix_t static_instruction_mix;
  // Final target resource and occupancy summary.
  loom_target_compile_report_target_resources_t target_resources;
  // Target wait-counter planning summary.
  loom_target_compile_report_wait_plan_t wait_plan;
  // Number of detailed register-pressure rows copied for this entry.
  iree_host_size_t pressure_row_count;
  // Number of detailed pressure-origin rows copied for this entry.
  iree_host_size_t pressure_origin_row_count;
  // Number of detailed low-schedule band rows copied for this entry.
  iree_host_size_t schedule_band_row_count;
  // Number of low-schedule band summary rows copied for this entry.
  iree_host_size_t schedule_band_summary_row_count;
  // Number of detailed spill rows copied for this entry.
  iree_host_size_t spill_row_count;
  // Number of detailed allocation high-water rows copied for this entry.
  iree_host_size_t allocation_high_water_row_count;
  // Number of target wait-counter rows copied for this entry.
  iree_host_size_t wait_counter_row_count;
  // Number of target wait-action rows copied for this entry.
  iree_host_size_t wait_action_row_count;
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

// One origin contribution to a register-pressure peak.
typedef struct loom_target_compile_report_pressure_origin_row_t {
  // Target artifact function symbol containing this pressure peak.
  iree_string_view_t function_name;
  // Register class name for register values, or an empty string otherwise.
  iree_string_view_t register_class;
  // Numeric Loom type kind for the pressure class.
  uint32_t type_kind;
  // Numeric Loom scalar element type for the pressure class.
  uint32_t element_type;
  // Program point associated with the pressure peak.
  uint32_t peak_point;
  // Block label containing the pressure peak.
  iree_string_view_t peak_block_name;
  // Operation name after which the peak was observed, or a boundary marker.
  iree_string_view_t peak_operation_name;
  // Structured family that produced this group of live values.
  loom_target_compile_report_pressure_origin_kind_t origin_kind;
  // Defining operation mnemonic for this group when available.
  iree_string_view_t origin_operation_name;
  // Descriptor semantic tag for descriptor-backed origins, if any.
  iree_string_view_t semantic_tag;
  // Representative SSA value from this contribution group.
  iree_string_view_t sample_value_name;
  // Number of live units contributed by this origin group at the peak.
  uint64_t live_units;
  // Number of live values contributed by this origin group at the peak.
  uint64_t live_values;
} loom_target_compile_report_pressure_origin_row_t;

// One consecutive low-schedule band in a compile report.
typedef struct loom_target_compile_report_schedule_band_row_t {
  // Target artifact function symbol containing this schedule band.
  iree_string_view_t function_name;
  // Region block label containing this schedule band.
  iree_string_view_t block_name;
  // Region block index containing this schedule band.
  uint32_t block_index;
  // First global scheduled packet index in this band.
  uint64_t first_packet_index;
  // First scheduled ordinal within |block_name| in this band.
  uint32_t first_scheduled_ordinal;
  // Number of consecutive scheduled nodes in this band.
  uint32_t node_count;
  // Structured origin family shared by this band.
  loom_target_compile_report_pressure_origin_kind_t origin_kind;
  // Representative operation mnemonic for this band.
  iree_string_view_t origin_operation_name;
  // Shared descriptor semantic tag for descriptor-backed bands, if any.
  iree_string_view_t semantic_tag;
  // Representative SSA result value produced in this band, if any.
  iree_string_view_t sample_value_name;
  // Static descriptor-backed instruction and effect feature counters for this
  // band.
  loom_target_compile_report_static_instruction_mix_t static_instruction_mix;
  // Number of result values produced by nodes in this band.
  uint64_t result_value_count;
  // Number of allocation units produced by result values in this band.
  uint64_t result_unit_count;
} loom_target_compile_report_schedule_band_row_t;

// Aggregate over low-schedule bands with the same block and origin family.
typedef struct loom_target_compile_report_schedule_band_summary_row_t {
  // Target artifact function symbol containing these schedule bands.
  iree_string_view_t function_name;
  // Region block label containing these schedule bands.
  iree_string_view_t block_name;
  // Region block index containing these schedule bands.
  uint32_t block_index;
  // First global scheduled packet index in the first matching band.
  uint64_t first_packet_index;
  // Number of matching bands in this block.
  uint64_t band_count;
  // Number of scheduled nodes across matching bands.
  uint64_t node_count;
  // Maximum consecutive node count in one matching band.
  uint32_t max_band_node_count;
  // Structured origin family shared by these bands.
  loom_target_compile_report_pressure_origin_kind_t origin_kind;
  // Representative operation mnemonic for these bands.
  iree_string_view_t origin_operation_name;
  // Shared descriptor semantic tag for descriptor-backed bands, if any.
  iree_string_view_t semantic_tag;
  // Representative SSA result value produced by these bands, if any.
  iree_string_view_t sample_value_name;
  // Static descriptor-backed instruction and effect feature counters for these
  // bands.
  loom_target_compile_report_static_instruction_mix_t static_instruction_mix;
  // Number of result values produced by these bands.
  uint64_t result_value_count;
  // Number of allocation units produced by result values in these bands.
  uint64_t result_unit_count;
} loom_target_compile_report_schedule_band_summary_row_t;

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

// One hard allocation-failure row in a compile report.
typedef struct loom_target_compile_report_allocation_failure_row_t {
  // Target artifact function symbol containing this allocation failure.
  iree_string_view_t function_name;
  // SSA value name whose interval could not be assigned.
  iree_string_view_t value_name;
  // Register class name for the failed value.
  iree_string_view_t register_class;
  // Numeric Loom type kind for the failed value class.
  uint32_t type_kind;
  // Numeric Loom scalar element type for the failed value class.
  uint32_t element_type;
  // Stable structured diagnostic failure code.
  iree_string_view_t failure_code;
  // Structured category describing the blocking constraint.
  loom_target_compile_report_allocation_failure_blocking_kind_t blocking_kind;
  // Operation mnemonic that produced the failed value, or a fallback context.
  iree_string_view_t origin_operation_name;
  // Block label containing |origin_operation_name|, or empty when unavailable.
  iree_string_view_t origin_block_name;
  // Program point where the failed interval starts.
  uint32_t start_point;
  // One-past-last storage program point required by the failed interval.
  uint32_t end_point;
  // Allocation units required by the failed interval.
  uint32_t required_unit_count;
  // Maximum allocation units available, or UINT32_MAX when unbounded.
  uint32_t budget_units;
  // Maximum boundary-live units observed for this pressure class.
  uint32_t peak_live_units;
  // Candidate location kind used while diagnosing the failure.
  iree_string_view_t location_kind;
  // Candidate base physical register or target ID, or UINT32_MAX.
  uint32_t location_base;
  // Candidate location width, or zero when unavailable.
  uint32_t location_count;
  // Assignment index for an active-assignment conflict, or UINT32_MAX.
  uint32_t conflict_assignment_index;
  // SSA value name occupying the conflicting assignment, or empty.
  iree_string_view_t conflict_value_name;
  // Program point where the conflicting assignment starts, or UINT32_MAX.
  uint32_t conflict_start_point;
  // One-past-last storage program point for the conflicting assignment, or
  // UINT32_MAX.
  uint32_t conflict_end_point;
  // Conflicting assignment location kind, or empty.
  iree_string_view_t conflict_location_kind;
  // Conflicting assignment base physical register or target ID, or UINT32_MAX.
  uint32_t conflict_location_base;
  // Conflicting assignment location width, or zero when unavailable.
  uint32_t conflict_location_count;
} loom_target_compile_report_allocation_failure_row_t;

// One allocation assignment that sets the high-water mark for a register class.
typedef struct loom_target_compile_report_allocation_high_water_row_t {
  // Target artifact function symbol containing this allocation.
  iree_string_view_t function_name;
  // SSA value represented by the high-water assignment.
  iree_string_view_t value_name;
  // Register class name for the assignment.
  iree_string_view_t register_class;
  // Numeric Loom type kind for the assignment value class.
  uint32_t type_kind;
  // Numeric Loom scalar element type for the assignment value class.
  uint32_t element_type;
  // Allocation assignment index that sets this high-water mark.
  uint32_t assignment_index;
  // Operation mnemonic that produced the value, or a fallback context.
  iree_string_view_t origin_operation_name;
  // Structured family that produced this value.
  loom_target_compile_report_pressure_origin_kind_t origin_kind;
  // Descriptor semantic tag for descriptor-backed origins, if any.
  iree_string_view_t semantic_tag;
  // Program point where the assignment storage begins.
  uint32_t start_point;
  // One-past-last storage program point for the assignment.
  uint32_t end_point;
  // Allocation units required by the assignment interval.
  uint32_t required_unit_count;
  // Concrete allocation location kind.
  iree_string_view_t location_kind;
  // Base physical register, target ID, or spill slot ordinal.
  uint32_t location_base;
  // Number of contiguous units assigned at |location_base|.
  uint32_t location_count;
  // One-past-last concrete location unit reached by this assignment.
  uint64_t high_water_units;
  // Number of unoccupied concrete units below |location_base| at |start_point|.
  uint64_t lower_free_unit_count;
  // Number of contiguous unoccupied-unit runs below |location_base|.
  uint32_t lower_free_run_count;
  // Largest contiguous unoccupied-unit run below |location_base|.
  uint32_t lower_largest_free_run_unit_count;
  // Upper-bound free units below |location_base| when pressure-releasable
  // storage leases are discounted.
  uint64_t lower_pressure_releasable_free_unit_count;
  // Upper-bound free runs below |location_base| when pressure-releasable
  // storage leases are discounted.
  uint32_t lower_pressure_releasable_free_run_count;
  // Largest upper-bound free run below |location_base| when pressure-releasable
  // storage leases are discounted.
  uint32_t lower_pressure_releasable_largest_free_run_unit_count;
  // Number of other active assignments below |high_water_units|.
  uint32_t active_assignment_blocker_count;
  // Total units occupied by other active assignments below |high_water_units|.
  uint64_t active_assignment_blocker_units;
  // Number of active target storage leases below |high_water_units|.
  uint32_t active_storage_lease_blocker_count;
  // Total target storage-lease units below |high_water_units|.
  uint64_t active_storage_lease_blocker_units;
  // Number of pressure-releasable storage leases below |high_water_units|.
  uint32_t active_pressure_storage_lease_blocker_count;
  // Total pressure-releasable storage-lease units below |high_water_units|.
  uint64_t active_pressure_storage_lease_blocker_units;
  // Number of fallback-release storage leases below |high_water_units|.
  uint32_t active_fallback_storage_lease_blocker_count;
  // Total fallback-release storage-lease units below |high_water_units|.
  uint64_t active_fallback_storage_lease_blocker_units;
} loom_target_compile_report_allocation_high_water_row_t;

// One target wait-counter summary row copied into a compile report.
typedef struct loom_target_compile_report_wait_counter_row_t {
  // Target artifact function symbol containing this wait plan.
  iree_string_view_t function_name;
  // Stable target-owned wait-counter name.
  iree_string_view_t counter_name;
  // Target-owned wait-counter id.
  uint32_t counter_id;
  // Aggregate wait-plan counts for this counter.
  loom_target_compile_report_wait_plan_t summary;
} loom_target_compile_report_wait_counter_row_t;

// One target wait-action row copied into a compile report.
typedef struct loom_target_compile_report_wait_action_row_t {
  // Target artifact function symbol containing this wait action.
  iree_string_view_t function_name;
  // Stable target-owned wait-counter name.
  iree_string_view_t counter_name;
  // Stable target-owned action name.
  iree_string_view_t action_name;
  // Stable target-owned wait reason name.
  iree_string_view_t reason_name;
  // Target-owned wait-counter id.
  uint32_t counter_id;
  // Target-owned wait-action id.
  uint32_t action_id;
  // Target-owned wait reason id.
  uint32_t reason_id;
  // Region block containing the insertion point or explicit wait.
  uint32_t block_index;
  // Node before which a planned wait is inserted, or explicit wait node.
  uint32_t node_index;
  // Scheduled ordinal before which a planned wait is inserted, or explicit
  // wait node's scheduled ordinal.
  uint32_t scheduled_ordinal;
  // Producer node that forced the wait, or UINT32_MAX.
  uint32_t producer_node;
  // Scheduled ordinal of |producer_node|, or UINT32_MAX.
  uint32_t producer_scheduled_ordinal;
  // Operation mnemonic for |producer_node|, or empty.
  iree_string_view_t producer_operation_name;
  // Descriptor key for |producer_node|, or empty.
  iree_string_view_t producer_descriptor_key;
  // Descriptor semantic tag for |producer_node|, or empty.
  iree_string_view_t producer_semantic_tag;
  // Consumer node that needs the wait, or UINT32_MAX.
  uint32_t consumer_node;
  // Scheduled ordinal of |consumer_node|, or UINT32_MAX.
  uint32_t consumer_scheduled_ordinal;
  // Operation mnemonic for |consumer_node|, or empty.
  iree_string_view_t consumer_operation_name;
  // Descriptor key for |consumer_node|, or empty.
  iree_string_view_t consumer_descriptor_key;
  // Descriptor semantic tag for |consumer_node|, or empty.
  iree_string_view_t consumer_semantic_tag;
  // Wait target value. Zero drains all outstanding packets for the counter.
  uint32_t target_count;
  // Outstanding packet count for this counter before the wait action.
  uint32_t outstanding_before;
} loom_target_compile_report_wait_action_row_t;

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
  // Stable target-owned key identifying the selected plan variant, if any.
  iree_string_view_t plan_key;
  // First stable low descriptor id emitted by a table rule, or none for plans.
  uint64_t descriptor_id;
  // Number of low operations emitted for this source operation.
  uint32_t emitted_low_op_count;
} loom_target_compile_report_source_low_row_t;

// One emitted source-memory packet row copied into a compile report.
typedef struct loom_target_compile_report_source_low_memory_row_t {
  // Source function symbol containing the lowered source operation.
  iree_string_view_t function_name;
  // Source operation mnemonic that emitted this memory packet.
  iree_string_view_t source_op_name;
  // Numeric source operation kind that emitted this memory packet.
  uint32_t source_op_kind;
  // Target-independent memory-space key selected by the target.
  iree_string_view_t memory_space;
  // Source memory operation kind selected by the target.
  iree_string_view_t operation_kind;
  // Stable target packet key selected for this emitted low operation.
  iree_string_view_t packet_key;
  // Stable target address-form key selected for this emitted low operation.
  iree_string_view_t address_form;
  // Stable target dynamic-term operand key for the source address.
  iree_string_view_t dynamic_term_kind;
  // Stable target-owned reason key for address-form selection or fallback.
  iree_string_view_t fallback_reason;
  // Static source byte offset before target packet splitting.
  int64_t static_offset_bytes;
  // Byte count of one addressed source element.
  uint32_t element_byte_count;
  // Number of source vector lanes moved by this packet.
  uint32_t vector_lane_count;
  // Byte stride between adjacent dynamic workitem terms, or zero when unknown.
  uint32_t dynamic_stride_bytes;
  // Byte stride between adjacent source vector lanes.
  uint32_t vector_lane_stride_bytes;
  // Distance between adjacent workitems in target bank words.
  uint32_t bank_stride_words;
  // Estimated bank conflict degree across one bank cycle, or zero if unknown.
  uint32_t bank_conflict_degree;
  // Stable target-owned bank-conflict classification key.
  iree_string_view_t bank_conflict_kind;
} loom_target_compile_report_source_low_memory_row_t;

// One target math-legalization decision row copied into a compile report.
typedef struct loom_target_compile_report_math_row_t {
  // Source function symbol containing the legalized math operation.
  iree_string_view_t function_name;
  // Source operation mnemonic considered by math legalization.
  iree_string_view_t source_op_name;
  // Numeric source operation kind considered by math legalization.
  uint32_t source_op_kind;
  // Target bundle selected for the containing function.
  iree_string_view_t target_bundle_name;
  // Target config selected for the containing function, if any.
  iree_string_view_t target_config_name;
  // Stable target math policy name that decided this row, if any.
  iree_string_view_t policy_name;
  // Stable structured constraint or recipe key selected by the policy.
  iree_string_view_t constraint_key;
  // Semantic math operation requested by the source op.
  uint32_t math_op;
  // Whether the source op computes one scalar lane or a vector of lanes.
  uint32_t lane_domain;
  // Scalar element type rewritten or rejected by math legalization.
  uint32_t element_type;
  // Target math-legalization decision recorded for this source operation.
  loom_target_compile_report_math_action_t action;
  // Recipe selected when |action| is REWRITTEN or MISSING_RECIPE.
  uint32_t recipe;
  // Source fast-math flags observed on the original operation.
  uint8_t source_fastmath_flags;
  // Extra fast-math flags applied by the selected recipe.
  uint8_t recipe_fastmath_flags;
  // Operations created by the math legalization recipe.
  uint64_t created_op_count;
  // Operations erased by the math legalization recipe.
  uint64_t erased_op_count;
} loom_target_compile_report_math_row_t;

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
  // Number of target storage-lease records.
  uint64_t allocation_storage_lease_count;
  // Number of assignment-backed target storage-lease instances.
  uint64_t allocation_storage_lease_instance_count;
  // Number of allocator-requested storage release actions.
  uint64_t allocation_storage_release_action_count;
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
  // Number of source math ops rewritten by target math legalization.
  uint64_t math_legalization_rewritten_op_count;
  // Number of source math ops rejected by target math legalization.
  uint64_t math_legalization_rejected_op_count;
  // Number of source math ops without a target math policy.
  uint64_t math_legalization_missing_policy_op_count;
  // Number of source math ops selecting an unimplemented recipe.
  uint64_t math_legalization_missing_recipe_op_count;
  // Residual target move counts indexed by
  // loom_target_compile_report_move_cause_t.
  loom_target_compile_report_move_cause_counts_t
      move_causes[LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_COUNT];
  // Static descriptor-backed instruction and effect feature counters.
  loom_target_compile_report_static_instruction_mix_t static_instruction_mix;
  // Final target resource and occupancy summary.
  loom_target_compile_report_target_resources_t target_resources;
  // Target wait-counter planning summary.
  loom_target_compile_report_wait_plan_t wait_plan;
  // Owned emitted artifact entry summary rows.
  loom_target_compile_report_row_list_t entry_rows;
  // Owned register-class pressure summaries used by target resources.
  loom_target_compile_report_row_list_t pressure_summaries;
  // Owned register-pressure peak rows.
  loom_target_compile_report_row_list_t pressure_rows;
  // Owned register-pressure origin contribution rows.
  loom_target_compile_report_row_list_t pressure_origin_rows;
  // Owned consecutive low-schedule band rows.
  loom_target_compile_report_row_list_t schedule_band_rows;
  // Owned low-schedule band summary rows.
  loom_target_compile_report_row_list_t schedule_band_summary_rows;
  // Owned predicted spill rows.
  loom_target_compile_report_row_list_t spill_rows;
  // Owned hard allocation-failure rows.
  loom_target_compile_report_row_list_t allocation_failure_rows;
  // Owned allocation high-water rows.
  loom_target_compile_report_row_list_t allocation_high_water_rows;
  // Owned target wait-counter summary rows.
  loom_target_compile_report_row_list_t wait_counter_rows;
  // Owned target wait-action rows.
  loom_target_compile_report_row_list_t wait_action_rows;
  // Owned source-to-low selection rows.
  loom_target_compile_report_row_list_t source_low_rows;
  // Owned emitted source-memory packet rows.
  loom_target_compile_report_row_list_t source_low_memory_rows;
  // Owned target math-legalization decision rows.
  loom_target_compile_report_row_list_t math_legalization_rows;
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
    uint64_t coalesced_copy_count, uint64_t materialized_copy_count,
    uint64_t storage_lease_count, uint64_t storage_lease_instance_count,
    uint64_t storage_release_action_count);

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

// Records final target resource and occupancy summary facts.
void loom_target_compile_report_record_target_resources(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_target_resources_t* target_resources);

// Records one register-class pressure summary for aggregate reporting.
iree_status_t loom_target_compile_report_record_pressure_summary(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_pressure_summary_t* summary);

// Records target wait-counter planning summary facts.
void loom_target_compile_report_record_wait_plan(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_wait_plan_t* wait_plan);

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

// Records one pressure-origin contribution row.
iree_status_t loom_target_compile_report_record_pressure_origin_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_pressure_origin_row_t* row);

// Records one low-schedule band row.
iree_status_t loom_target_compile_report_record_schedule_band_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_schedule_band_row_t* row);

// Records one low-schedule band summary row.
iree_status_t loom_target_compile_report_record_schedule_band_summary_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_schedule_band_summary_row_t* row);

// Records one spill row.
iree_status_t loom_target_compile_report_record_spill_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_spill_row_t* row);

// Records one hard allocation-failure row.
iree_status_t loom_target_compile_report_record_allocation_failure_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_allocation_failure_row_t* row);

// Records one allocation high-water row.
iree_status_t loom_target_compile_report_record_allocation_high_water_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_allocation_high_water_row_t* row);

// Records one target wait-counter summary row. This does not update the
// aggregate wait-plan summary; callers that produce detail rows must also call
// loom_target_compile_report_record_wait_plan with their aggregate summary.
iree_status_t loom_target_compile_report_record_wait_counter_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_wait_counter_row_t* row);

// Records one target wait-action row.
iree_status_t loom_target_compile_report_record_wait_action_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_wait_action_row_t* row);

// Records one source-low row.
iree_status_t loom_target_compile_report_record_source_low_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_row_t* row);

// Records one emitted source-memory packet row.
iree_status_t loom_target_compile_report_record_source_low_memory_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_row_t* row);

// Records one target math-legalization row.
iree_status_t loom_target_compile_report_record_math_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_math_row_t* row);

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
