// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured module compilation reports.

#ifndef LOOM_TARGET_REPORTING_REPORT_H_
#define LOOM_TARGET_REPORTING_REPORT_H_

#include "iree/base/api.h"
#include "loom/codegen/low/planning_statistics.h"
#include "loom/target/reporting/target_insertion.h"
#include "loom/target/residency.h"
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
  // Portable command-program artifact.
  LOOM_TARGET_COMPILE_ARTIFACT_KIND_COMMAND_PROGRAM = 5,
  // Host launch-config program artifact.
  LOOM_TARGET_COMPILE_ARTIFACT_KIND_LAUNCH_CONFIG = 6,
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
  // Target capability rows were recorded or counted.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_CAPABILITY_ROWS = 1u << 19,
  // Static launch workload facts were recorded.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_WORKLOAD = 1u << 20,
  // Per-workitem dynamic instruction-mix estimates were recorded.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_DYNAMIC_INSTRUCTION_MIX = 1u << 21,
  // Aggregated low-schedule band summary rows were recorded or counted.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS = 1u << 22,
  // Invocation config bindings materialized before compilation.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_CONFIG_BINDING_ROWS = 1u << 23,
  // Coarse target-low planning work and memory statistics were recorded.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_LOW_PLANNING = 1u << 24,
  // Target-inserted native packet rows were recorded or counted.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_INSERTION_ROWS = 1u << 25,
  // Final target body, entry, and coissued instruction counts were recorded.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION_BREAKDOWN = 1u << 26,
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
  // Descriptor-backed private or stack memory value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_PRIVATE_MEMORY = 15,
  // Descriptor-backed generic memory value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_GENERIC_MEMORY = 16,
  // Descriptor-backed control-flow value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONTROL = 17,
  // Descriptor-backed barrier or synchronization value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_BARRIER = 18,
  // Descriptor-backed numeric conversion value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONVERSION = 19,
  // Descriptor-backed register move or repair value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_REGISTER_MOVE = 20,
  // Descriptor-backed cache or prefetch value.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CACHE = 21,
  // Operation-backed value not covered by a more specific origin.
  LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_OPERATION = 22,
} loom_target_compile_report_pressure_origin_kind_t;

typedef enum loom_target_compile_report_spill_row_kind_e {
  // Unknown spill row source.
  LOOM_TARGET_COMPILE_REPORT_SPILL_ROW_UNKNOWN = 0,
  // Spill row came from a current allocation spill plan.
  LOOM_TARGET_COMPILE_REPORT_SPILL_ROW_PLANNED = 1,
  // Spill row came from storage traffic materialized before the final frame.
  LOOM_TARGET_COMPILE_REPORT_SPILL_ROW_MATERIALIZED = 2,
} loom_target_compile_report_spill_row_kind_t;

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

// Final target instruction-count decomposition.
//
// The total emitted instruction count includes the target entry envelope and
// the final target body. Coissued instructions are a subset of the body and
// retain all of their semantic components in the instruction mix.
typedef struct loom_target_compile_report_emission_breakdown_t {
  // Final native instructions in the scheduled target body.
  uint64_t body_instruction_count;
  // Target-owned instructions outside the scheduled body at hardware entry.
  uint64_t entry_instruction_count;
  // Final native instructions that coissue multiple semantic components.
  uint64_t coissued_instruction_count;
  // Semantic instruction components carried by coissued instructions.
  uint64_t coissued_component_count;
} loom_target_compile_report_emission_breakdown_t;

// Static feature counters for low packets and finalized structural moves that
// survive target emission.
//
// These counters are compile-time proxies derived from generated descriptor
// instruction classes, structural low terminators, and allocation-owned final
// move groups. They are intentionally separate from measured HAL profiling
// counters and may overlap: for example a global atomic packet is both global
// memory and atomic. Byte counts are static descriptor-effect widths for one
// issue of the emitted low stream, not evaluated dynamic workload traffic.
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
  // Descriptor-backed nodes identified as scaled MFMA-like instructions.
  uint64_t smfmac_count;
  // Descriptor-backed nodes identified as WMMA-like matrix instructions.
  uint64_t wmma_count;
  // Descriptor-backed nodes identified as scaled WMMA-like instructions.
  uint64_t swmmac_count;
  // Descriptor-backed nodes identified as dot-product instructions.
  uint64_t dot_count;
  // Descriptor-backed nodes identified as global or vector-memory operations.
  uint64_t global_memory_count;
  // Descriptor-backed raw global-load-family memory instructions.
  uint64_t global_load_count;
  // Descriptor-backed raw global-store-family memory instructions.
  uint64_t global_store_count;
  // Descriptor-backed resource-buffer-load-family memory instructions.
  uint64_t buffer_load_count;
  // Descriptor-backed resource-buffer-store-family memory instructions.
  uint64_t buffer_store_count;
  // Descriptor-backed flat-memory-family instructions.
  uint64_t flat_memory_count;
  // Descriptor-backed nodes identified as local/shared/workgroup memory ops.
  uint64_t local_memory_count;
  // Descriptor-backed nodes identified as scalar-memory operations.
  uint64_t scalar_memory_count;
  // Descriptor-backed nodes identified as private or stack memory operations.
  uint64_t private_memory_count;
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
  // Static bytes read by private or stack memory descriptor effects.
  uint64_t private_read_byte_count;
  // Static bytes written by private or stack memory descriptor effects.
  uint64_t private_write_byte_count;
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
  // Descriptor-backed register moves and finalized structural physical moves.
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
  // Exact target residency transition summary, or zero when unavailable.
  loom_target_residency_summary_t residency_summary;
} loom_target_compile_report_target_resources_t;

typedef uint8_t loom_target_compile_report_capability_value_kind_t;
typedef enum loom_target_compile_report_capability_value_kind_e {
  // No capability value kind is available.
  LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_NONE = 0,
  // Capability value is a boolean stored in value_u64 as 0 or 1.
  LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_BOOL = 1,
  // Capability value is an unsigned integer stored in value_u64.
  LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_U64 = 2,
  // Capability value is a borrowed string stored in value_string.
  LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_STRING = 3,
} loom_target_compile_report_capability_value_kind_e;

// Selected target capability fact recorded for an emitted entry.
typedef struct loom_target_compile_report_target_capability_row_t {
  // Target artifact function symbol associated with this capability.
  iree_string_view_t function_name;
  // Target family that owns the capability namespace.
  iree_string_view_t target_family_name;
  // Stable capability namespace such as "target" or "amdgpu".
  iree_string_view_t namespace_name;
  // Stable capability key within namespace_name.
  iree_string_view_t key;
  // Value representation used by this capability row.
  loom_target_compile_report_capability_value_kind_t value_kind;
  // Boolean or unsigned integer payload for this capability row.
  uint64_t value_u64;
  // Borrowed string payload for this capability row.
  iree_string_view_t value_string;
} loom_target_compile_report_target_capability_row_t;

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
  // Total packets drained by all wait actions.
  uint64_t drained_count;
  // Maximum packets drained by any wait action.
  uint64_t max_drained_count;
  // Maximum outstanding packet count observed before any wait action.
  uint64_t max_outstanding_before;
  // Maximum outstanding packet count observed before a full-drain action.
  uint64_t max_full_drain_outstanding_before;
} loom_target_compile_report_wait_plan_t;

typedef uint32_t loom_target_compile_report_workload_flags_t;
enum {
  // No static workload facts were recorded.
  LOOM_TARGET_COMPILE_REPORT_WORKLOAD_NONE = 0u,
  // |workgroup_size| is populated.
  LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_SIZE = 1u << 0,
  // |workgroup_count| is populated.
  LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_COUNT = 1u << 1,
  // |flat_workgroup_size| is populated.
  LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_SIZE = 1u << 2,
  // |dispatch_workgroup_count| is populated.
  LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKGROUP_COUNT = 1u << 3,
  // |dispatch_workitem_count| is populated.
  LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT = 1u << 4,
  // |workgroup_cluster_size| is populated.
  LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_CLUSTER_SIZE = 1u << 5,
  // |flat_workgroup_cluster_size| is populated.
  LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_CLUSTER_SIZE = 1u << 6,
};

// Static launch workload facts proven for a compiled entry or shared by every
// entry in a report.
typedef struct loom_target_compile_report_workload_t {
  // Workload fields populated in this record.
  loom_target_compile_report_workload_flags_t flags;
  // Static local workgroup size.
  loom_target_workgroup_size_t workgroup_size;
  // Static dispatch workgroup count.
  loom_target_dispatch_workgroup_count_t workgroup_count;
  // Static nontrivial workgroup-cluster size.
  loom_target_workgroup_cluster_size_t workgroup_cluster_size;
  // Product of workgroup_size x/y/z.
  uint64_t flat_workgroup_size;
  // Product of workgroup_count x/y/z.
  uint64_t dispatch_workgroup_count;
  // Product of workgroup_cluster_size x/y/z.
  uint64_t flat_workgroup_cluster_size;
  // Product of flat workgroup size and dispatch workgroup count.
  uint64_t dispatch_workitem_count;
} loom_target_compile_report_workload_t;

// Structural target bank-service evidence accumulated across source packets.
typedef struct loom_target_compile_report_bank_service_summary_t {
  // Number of source packets for which a target service model was selected.
  uint64_t modeled_packet_count;
  // Number of modeled source packets with exact service evidence.
  uint64_t exact_packet_count;
  // Number of modeled source packets without exact service evidence.
  uint64_t unknown_packet_count;
  // Number of exact source packets requiring no extra service rounds.
  uint64_t conflict_free_packet_count;
  // Number of exact source packets requiring extra service rounds.
  uint64_t conflicted_packet_count;
  // Required service rounds summed once per exact source packet.
  uint64_t required_round_count;
  // Uncontended service rounds summed once per exact source packet.
  uint64_t uncontended_round_count;
  // Extra service rounds summed once per exact source packet.
  uint64_t extra_round_count;
  // Number of source packets with exact dynamic service contributions.
  uint64_t exact_dynamic_packet_count;
  // Number of source packets without exact dynamic service contributions.
  uint64_t unknown_dynamic_packet_count;
  // Dynamic executions represented by exact service contributions.
  uint64_t dynamic_packet_count;
  // Required service rounds across exact dynamic packet executions.
  uint64_t dynamic_required_round_count;
  // Uncontended service rounds across exact dynamic packet executions.
  uint64_t dynamic_uncontended_round_count;
  // Extra service rounds across exact dynamic packet executions.
  uint64_t dynamic_extra_round_count;
  // Maximum requests assigned to one bank in one exact packet phase.
  uint16_t maximum_request_multiplicity;
} loom_target_compile_report_bank_service_summary_t;

// Structural and exact-dynamic subgroup access coverage counters.
typedef struct loom_target_compile_report_subgroup_access_summary_t {
  // Number of packets carrying subgroup access evidence.
  uint64_t modeled_packet_count;
  // Number of packets with exact active-subgroup address geometry.
  uint64_t exact_packet_count;
  // Number of packets without exact active-subgroup address geometry.
  uint64_t unknown_packet_count;
  // Number of exact packets whose request intervals cover a dense span.
  uint64_t dense_packet_count;
  // Number of exact packets whose request intervals contain uncovered gaps.
  uint64_t gapped_packet_count;
  // Number of exact packets containing overlapping lane requests.
  uint64_t overlapping_packet_count;
  // Number of source packets with exact dynamic geometry contributions.
  uint64_t exact_dynamic_packet_count;
  // Number of source packets without exact dynamic geometry contributions.
  uint64_t unknown_dynamic_packet_count;
  // Dynamic executions represented by exact subgroup geometry contributions.
  uint64_t dynamic_packet_count;
  // Exact dynamic packet executions with dense request intervals.
  uint64_t dynamic_dense_packet_count;
  // Exact dynamic packet executions with gapped request intervals.
  uint64_t dynamic_gapped_packet_count;
  // Exact dynamic packet executions containing overlapping lane requests.
  uint64_t dynamic_overlapping_packet_count;
} loom_target_compile_report_subgroup_access_summary_t;

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
  // Number of spill storage slots materialized before the final frame.
  uint64_t allocation_materialized_spill_storage_count;
  // Byte size of spill storage materialized before the final frame.
  uint64_t allocation_materialized_spill_storage_bytes;
  // Number of low.spill stores materialized before the final frame.
  uint64_t allocation_materialized_spill_store_count;
  // Byte traffic from low.spill stores materialized before the final frame.
  uint64_t allocation_materialized_spill_store_bytes;
  // Number of low.reload ops materialized before the final frame.
  uint64_t allocation_materialized_reload_count;
  // Byte traffic from low.reload ops materialized before the final frame.
  uint64_t allocation_materialized_reload_bytes;
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
  // Final target instruction-count decomposition.
  loom_target_compile_report_emission_breakdown_t emission_breakdown;
  // Estimated target private memory bytes.
  uint64_t private_memory_bytes;
  // Estimated target local/shared memory bytes.
  uint64_t local_memory_bytes;
  // Structural target bank-service evidence for this entry.
  loom_target_compile_report_bank_service_summary_t bank_service_summary;
  // Structural subgroup address geometry for this entry.
  loom_target_compile_report_subgroup_access_summary_t subgroup_access_summary;
  // Residual target move counts indexed by
  // loom_target_compile_report_move_cause_t.
  loom_target_compile_report_move_cause_counts_t
      move_causes[LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_COUNT];
  // Static descriptor-backed instruction and effect feature counters.
  loom_target_compile_report_static_instruction_mix_t static_instruction_mix;
  // Per-workitem instruction and effect counters after multiplying by exact
  // statically-provable loop trip counts.
  loom_target_compile_report_static_instruction_mix_t dynamic_instruction_mix;
  // Final target resource and occupancy summary.
  loom_target_compile_report_target_resources_t target_resources;
  // Target wait-counter planning summary.
  loom_target_compile_report_wait_plan_t wait_plan;
  // Static launch workload facts for this entry.
  loom_target_compile_report_workload_t workload;
  // Coarse target-low planning work and memory statistics for this entry.
  loom_low_planning_statistics_t low_planning;
  // Target-inserted native packet counts for this entry.
  loom_target_compile_report_target_insertion_summary_t
      target_insertion_summary;
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
  // Number of target wait-reason summary rows copied for this entry.
  iree_host_size_t wait_reason_summary_row_count;
  // Number of target wait-action rows copied for this entry.
  iree_host_size_t wait_action_row_count;
  // Number of selected-target capability rows copied for this entry.
  iree_host_size_t target_capability_row_count;
  // Number of target-inserted native packet rows copied for this entry.
  iree_host_size_t target_insertion_row_count;
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

typedef uint32_t loom_target_compile_report_schedule_band_flags_t;
enum {
  // Dynamic instruction mix was proven exactly for this schedule band.
  LOOM_TARGET_COMPILE_REPORT_SCHEDULE_BAND_DYNAMIC_INSTRUCTION_MIX = 1u << 0,
};

// One consecutive low-schedule band in a compile report.
typedef struct loom_target_compile_report_schedule_band_row_t {
  // Optional fields populated for this row.
  loom_target_compile_report_schedule_band_flags_t flags;
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
  // Per-workitem instruction and effect counters after multiplying this band by
  // exact statically-provable loop trip counts.
  loom_target_compile_report_static_instruction_mix_t dynamic_instruction_mix;
  // Number of result values produced by nodes in this band.
  uint64_t result_value_count;
  // Number of allocation units produced by result values in this band.
  uint64_t result_unit_count;
} loom_target_compile_report_schedule_band_row_t;

// Aggregate over low-schedule bands with the same block and origin family.
typedef struct loom_target_compile_report_schedule_band_summary_row_t {
  // Optional fields populated for this summary row.
  loom_target_compile_report_schedule_band_flags_t flags;
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
  // Per-workitem instruction and effect counters after multiplying these bands
  // by exact statically-provable loop trip counts.
  loom_target_compile_report_static_instruction_mix_t dynamic_instruction_mix;
  // Number of result values produced by these bands.
  uint64_t result_value_count;
  // Number of allocation units produced by result values in these bands.
  uint64_t result_unit_count;
} loom_target_compile_report_schedule_band_summary_row_t;

// One planned or materialized spill row in a compile report.
typedef struct loom_target_compile_report_spill_row_t {
  // Source of this spill row.
  loom_target_compile_report_spill_row_kind_t kind;
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
  // Structured family that produced the spilled value.
  loom_target_compile_report_pressure_origin_kind_t origin_kind;
  // Operation mnemonic that produced the spilled value.
  iree_string_view_t origin_operation_name;
  // Descriptor semantic tag for descriptor-backed origins, if any.
  iree_string_view_t semantic_tag;
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
  // Byte traffic from predicted or materialized spill stores.
  uint64_t store_bytes;
  // Predicted operand-use reloads in the current synthetic spill plan.
  uint64_t reload_count;
  // Byte traffic from predicted or materialized reloads.
  uint64_t reload_bytes;
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

// One target wait-reason summary row copied into a compile report.
typedef struct loom_target_compile_report_wait_reason_summary_row_t {
  // Target artifact function symbol containing this wait plan.
  iree_string_view_t function_name;
  // Stable target-owned wait-counter name.
  iree_string_view_t counter_name;
  // Stable target-owned wait reason name.
  iree_string_view_t reason_name;
  // Target-owned wait-counter id.
  uint32_t counter_id;
  // Target-owned wait reason id.
  uint32_t reason_id;
  // Aggregate wait-plan counts for this counter/reason pair.
  loom_target_compile_report_wait_plan_t summary;
} loom_target_compile_report_wait_reason_summary_row_t;

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
  // Outstanding packet count for this counter after the wait action.
  uint32_t outstanding_after;
  // Number of outstanding packets drained by this wait action.
  uint32_t drained_count;
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
  // Stable target-owned key identifying the selected plan variant, if any.
  iree_string_view_t plan_key;
  // First low descriptor key emitted by this source op, if any.
  iree_string_view_t descriptor_key;
  // First low descriptor semantic tag emitted by this source op, if any.
  iree_string_view_t descriptor_semantic_tag;
  // Number of low operations emitted for this source operation.
  uint32_t emitted_low_op_count;
  // Exact source execution count plus one, or zero when unknown.
  uint64_t execution_count_plus_one;
} loom_target_compile_report_source_low_row_t;

// Source execution count evidence is not statically known.
#define LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_EXECUTION_COUNT_PLUS_ONE_UNKNOWN \
  0u

// One source function target selection copied into a compile report.
typedef struct loom_target_compile_report_source_low_target_row_t {
  // Source function symbol selected for source-to-low lowering.
  iree_string_view_t function_name;
  // Whether the effective target was authored or bound by specialization.
  loom_target_binding_source_t target_source;
  // Module target symbol used as the effective target record.
  iree_string_view_t target_symbol_name;
  // Effective target bundle name selected for the source function.
  iree_string_view_t target_bundle_name;
  // Effective target snapshot name selected for the source function.
  iree_string_view_t target_snapshot_name;
  // Effective target config name selected for the source function.
  iree_string_view_t target_config_name;
  // Effective target fixed subgroup size, or zero when unspecified.
  uint32_t target_subgroup_size;
  // Compatible module target records with different topology.
  uint32_t candidate_target_count;
  // First compatible different-topology target symbol name, if any.
  iree_string_view_t candidate_target_symbol_name;
  // First compatible different-topology target bundle name, if any.
  iree_string_view_t candidate_target_bundle_name;
  // First compatible different-topology target snapshot name, if any.
  iree_string_view_t candidate_target_snapshot_name;
  // First compatible different-topology target config name, if any.
  iree_string_view_t candidate_target_config_name;
  // First compatible different-topology target subgroup size, if any.
  uint32_t candidate_target_subgroup_size;
} loom_target_compile_report_source_low_target_row_t;

// One source transform decision copied into a compile report.
typedef struct loom_target_compile_report_source_low_transform_row_t {
  // Source function symbol containing the transformed source operation.
  iree_string_view_t function_name;
  // Source operation mnemonic anchoring this transform decision.
  iree_string_view_t source_op_name;
  // Numeric source operation kind anchoring this transform decision.
  uint32_t source_op_kind;
  // Stable pass or transform key.
  iree_string_view_t transform_key;
  // Stable transform outcome key.
  iree_string_view_t outcome;
  // Stable transform reason key.
  iree_string_view_t reason;
  // Source values structurally eligible for this transform.
  uint32_t candidate_value_count;
  // Source values selected for this transform.
  uint32_t selected_value_count;
  // Loop-carried values removed from the transformed source operation.
  uint32_t removed_loop_carried_value_count;
  // Estimated 32-bit payload registers removed from loop-carried state.
  uint64_t removed_loop_carried_payload_register_count;
  // Source fragment logical block count, or zero when not applicable.
  uint64_t block_count;
  // Source fragment logical row count, or zero when not applicable.
  uint64_t row_count;
  // Source fragment logical column count, or zero when not applicable.
  uint64_t column_count;
  // Workgroup memory bytes introduced by this transform, or zero.
  uint64_t workgroup_memory_byte_count;
  // Source-level load operations introduced by this transform.
  uint32_t inserted_load_op_count;
  // Source-level store operations introduced by this transform.
  uint32_t inserted_store_op_count;
  // Source-level barrier operations introduced by this transform.
  uint32_t inserted_barrier_op_count;
} loom_target_compile_report_source_low_transform_row_t;

// One invocation config binding materialized into the compiled module.
typedef struct loom_target_compile_report_config_binding_row_t {
  // Config symbol name without the textual '@' sigil.
  iree_string_view_t key;
  // Caller-provided textual value bound to |key|.
  iree_string_view_t value;
} loom_target_compile_report_config_binding_row_t;

// Summary of source-to-target-low selections grouped by stable lowering shape.
typedef struct loom_target_compile_report_source_low_selection_summary_t {
  // Source function symbol containing this lowering shape.
  iree_string_view_t function_name;
  // Source operation mnemonic lowered by this shape.
  iree_string_view_t source_op_name;
  // Numeric source operation kind lowered by this shape.
  uint32_t source_op_kind;
  // Selection mechanism used for this source operation.
  loom_target_compile_report_source_low_selection_kind_t selection_kind;
  // Stable target-owned key identifying the selected plan variant, if any.
  iree_string_view_t plan_key;
  // First low descriptor key emitted by this source op, if any.
  iree_string_view_t descriptor_key;
  // First low descriptor semantic tag emitted by this source op, if any.
  iree_string_view_t descriptor_semantic_tag;
  // Number of selected source operations represented by this summary.
  uint64_t selected_op_count;
  // Number of low operations emitted by this lowering shape.
  uint64_t emitted_low_op_count;
  // Number of source operations with exact dynamic execution evidence.
  uint64_t exact_dynamic_op_count;
  // Number of source operations without exact dynamic execution evidence.
  uint64_t unknown_dynamic_op_count;
  // Exact dynamic source operation executions across exact rows.
  uint64_t dynamic_selected_op_count;
  // Exact dynamic low operations emitted across exact rows.
  uint64_t dynamic_emitted_low_op_count;
} loom_target_compile_report_source_low_selection_summary_t;

// Source memory packet execution count evidence is not statically known.
#define LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_EXECUTION_COUNT_PLUS_ONE_UNKNOWN \
  LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_EXECUTION_COUNT_PLUS_ONE_UNKNOWN

typedef uint32_t loom_target_compile_report_memory_interval_flags_t;
enum {
  // The interval carries a bounded range for the byte interval begin.
  LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_RANGE = 1u << 0,
  // The interval carries a bounded range for the exclusive byte interval end.
  LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_RANGE = 1u << 1,
  // The interval length is exact even when the begin offset is dynamic.
  LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_EXACT_LENGTH = 1u << 2,
  // The interval begin has a report-private exact expression identity.
  LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_EXPR = 1u << 3,
  // The exclusive interval end has a report-private exact expression identity.
  LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_EXPR = 1u << 4,
};

// Conservative source byte interval evidence for one memory packet.
typedef struct loom_target_compile_report_memory_interval_t {
  // Bitset of loom_target_compile_report_memory_interval_flags_t values.
  loom_target_compile_report_memory_interval_flags_t flags;
  // Minimum possible byte offset for the interval begin.
  int64_t begin_min_bytes;
  // Maximum possible byte offset for the interval begin.
  int64_t begin_max_bytes;
  // Minimum possible byte offset for the exclusive interval end.
  int64_t end_min_bytes;
  // Maximum possible byte offset for the exclusive interval end.
  int64_t end_max_bytes;
  // Exact touched byte length for each dynamic instance, or zero when unknown.
  uint64_t exact_length_bytes;
  // Report-private exact expression ID for begin when BEGIN_EXPR is set.
  uint32_t begin_expr_id;
  // Report-private exact expression ID for end when END_EXPR is set.
  uint32_t end_expr_id;
} loom_target_compile_report_memory_interval_t;

// Conservative source byte interval envelope for a packet group.
typedef struct loom_target_compile_report_memory_interval_summary_t {
  // Number of packets carrying bounded source interval evidence.
  uint64_t packet_count;
  // Number of packets with exact static intervals used for unique accounting.
  uint64_t exact_static_packet_count;
  // Number of packets with exact symbolic intervals used for unique accounting.
  uint64_t exact_symbolic_packet_count;
  // Minimum possible byte offset across all interval begins.
  int64_t envelope_begin_min_bytes;
  // Maximum possible exclusive byte offset across all interval ends.
  int64_t envelope_end_max_bytes;
  // Conservative byte span from |envelope_begin_min_bytes| to
  // |envelope_end_max_bytes|.
  uint64_t envelope_byte_count;
  // Proven unique bytes across exact packet intervals.
  uint64_t unique_byte_count;
} loom_target_compile_report_memory_interval_summary_t;

// Maximum target bank-service phases retained in a compile report row.
#define LOOM_TARGET_COMPILE_REPORT_BANK_SERVICE_PHASE_CAPACITY 8

// Target-owned bank-service evidence for one emitted source memory packet.
typedef struct loom_target_compile_report_bank_service_t {
  // Exactness of the result: "exact", "unknown", or empty when not analyzed.
  iree_string_view_t proof;
  // Exact result class: "conflict-free", "conflicted", or empty when unknown.
  iree_string_view_t classification;
  // Stable target packet-service model key.
  iree_string_view_t model_key;
  // Immutable source revision defining the selected model.
  iree_string_view_t model_revision;
  // Provenance strength of the selected model.
  iree_string_view_t model_evidence;
  // Treatment of requests to identical bank-word addresses.
  iree_string_view_t request_policy;
  // Proof used to derive per-lane packet base addresses.
  iree_string_view_t lane_address_proof;
  // Proof used to derive the active lane set.
  iree_string_view_t active_lane_proof;
  // Proof covering unknown common LDS base translations.
  iree_string_view_t base_residue_proof;
  // Stable reason key when |proof| is "unknown".
  iree_string_view_t unknown_reason;
  // Number of lanes represented by the model phases.
  uint8_t wave_size;
  // Number of independently serviced LDS banks.
  uint8_t bank_count;
  // Byte width of one LDS bank word.
  uint8_t bank_word_byte_count;
  // Number of consecutive bank words requested by each active lane.
  uint8_t packet_word_count;
  // Number of populated phase entries.
  uint8_t phase_count;
  // Number of active model lanes in each service phase.
  uint8_t
      phase_lane_counts[LOOM_TARGET_COMPILE_REPORT_BANK_SERVICE_PHASE_CAPACITY];
  // Number of common bank-word base residues covered by the result.
  uint8_t base_residue_count;
  // Required bank service rounds for each model phase.
  uint16_t phase_required_rounds
      [LOOM_TARGET_COMPILE_REPORT_BANK_SERVICE_PHASE_CAPACITY];
  // Sum of |phase_required_rounds|.
  uint16_t required_rounds;
  // One round for each phase containing at least one active lane.
  uint16_t uncontended_rounds;
  // Difference between |required_rounds| and |uncontended_rounds|.
  uint16_t extra_rounds;
  // Maximum requests assigned to one bank in one phase.
  uint16_t maximum_request_multiplicity;
} loom_target_compile_report_bank_service_t;

// Maximum lossless subgroup lane-address terms retained in a compile report.
#define LOOM_TARGET_COMPILE_REPORT_SUBGROUP_ACCESS_TERM_CAPACITY 4

// One term in an exact subgroup lane-address function.
typedef struct loom_target_compile_report_subgroup_access_term_t {
  // Power-of-two divisor applied to the subgroup lane ID.
  uint16_t divisor;
  // Optional power-of-two modulus applied after division; zero omits it.
  uint16_t modulus;
  // Byte stride multiplied by the resulting lane digit.
  uint32_t byte_stride;
} loom_target_compile_report_subgroup_access_term_t;

// Target-owned subgroup address geometry for one emitted memory packet.
typedef struct loom_target_compile_report_subgroup_access_t {
  // Exactness of the complete active-subgroup geometry: "exact" or "unknown".
  iree_string_view_t proof;
  // Proof used to derive the per-lane relative byte addresses.
  iree_string_view_t lane_address_proof;
  // Proof used to derive the active lane set.
  iree_string_view_t active_lane_proof;
  // Address-function form: "uniform", "linear", or "digit-terms".
  iree_string_view_t lane_mapping;
  // Exact byte-interval coverage: "dense", "gapped", or empty when unknown.
  iree_string_view_t interval_coverage;
  // Stable reason key when |proof| is "unknown".
  iree_string_view_t unknown_reason;
  // Number of lanes in the modeled subgroup.
  uint8_t subgroup_size;
  // Number of populated lossless lane-address terms.
  uint8_t lane_term_count;
  // Lossless relative address terms evaluated in array order.
  loom_target_compile_report_subgroup_access_term_t
      lane_terms[LOOM_TARGET_COMPILE_REPORT_SUBGROUP_ACCESS_TERM_CAPACITY];
  // Selected target packet byte width issued by each active lane.
  uint32_t per_lane_packet_byte_count;
  // Constant byte stride between adjacent lanes, or zero when non-linear.
  uint32_t linear_lane_byte_stride;
  // Sum of per-lane packet byte widths, including overlapping requests.
  uint64_t subgroup_requested_byte_count;
  // Unique bytes covered by the union of all active-lane packet intervals.
  uint64_t subgroup_unique_byte_count;
  // Byte span from the minimum request begin to the maximum request end.
  uint64_t subgroup_span_byte_count;
  // Maximum absolute byte delta between adjacent subgroup lane addresses.
  uint64_t maximum_adjacent_lane_delta_bytes;
  // Maximum uncovered byte gap between sorted request intervals.
  uint64_t maximum_uncovered_byte_gap_bytes;
  // Number of distinct active-lane packet start addresses.
  uint16_t distinct_lane_address_count;
  // Number of disjoint byte regions covered by active-lane packets.
  uint16_t contiguous_region_count;
} loom_target_compile_report_subgroup_access_t;

// One emitted source-memory packet row copied into a compile report.
typedef struct loom_target_compile_report_source_low_memory_row_t {
  // Source function symbol containing the lowered source operation.
  iree_string_view_t function_name;
  // Source operation mnemonic that emitted this memory packet.
  iree_string_view_t source_op_name;
  // Numeric source operation kind that emitted this memory packet.
  uint32_t source_op_kind;
  // Named source memory root selected by value facts, if available.
  iree_string_view_t source_root_name;
  // Source function entry argument index for the memory root, or UINT16_MAX.
  uint16_t source_root_argument_index;
  // Target-independent memory-space key selected by the target.
  iree_string_view_t memory_space;
  // Source memory operation kind selected by the target.
  iree_string_view_t operation_kind;
  // Stable target packet key selected for this emitted low operation.
  iree_string_view_t packet_key;
  // Stable target-owned strategy key selected for this memory packet, if any.
  iree_string_view_t strategy_key;
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
  // Byte count read by the emitted target packet effect.
  uint32_t issued_read_byte_count;
  // Byte count written by the emitted target packet effect.
  uint32_t issued_write_byte_count;
  // Number of read effects without a known byte-aligned target width.
  uint16_t issued_read_unknown_width_count;
  // Number of write effects without a known byte-aligned target width.
  uint16_t issued_write_unknown_width_count;
  // Byte stride between adjacent dynamic workitem terms, or zero when unknown.
  uint32_t dynamic_stride_bytes;
  // Byte stride between adjacent source vector lanes.
  uint32_t vector_lane_stride_bytes;
  // Logical storage element format recovered from source encoding facts.
  iree_string_view_t storage_element_format;
  // Primary scale format recovered from source encoding facts.
  iree_string_view_t storage_scale_format;
  // Secondary scale format recovered from source encoding facts.
  iree_string_view_t storage_secondary_scale_format;
  // Physical payload packing recovered from source encoding facts.
  iree_string_view_t storage_payload_packing;
  // Scale topology recovered from source encoding facts.
  iree_string_view_t storage_scale_topology;
  // Affine payload interpretation recovered from source encoding facts.
  iree_string_view_t storage_affine_policy;
  // Rounding or finite-policy contract recovered from source encoding facts.
  iree_string_view_t storage_rounding_policy;
  // Codebook ownership contract recovered from source encoding facts.
  iree_string_view_t storage_codebook_policy;
  // Sparse metadata contract recovered from source encoding facts.
  iree_string_view_t storage_sparsity_policy;
  // Target-owned bank-service evidence for this packet.
  loom_target_compile_report_bank_service_t bank_service;
  // Target-owned subgroup address geometry for this packet.
  loom_target_compile_report_subgroup_access_t subgroup_access;
  // Conservative source byte interval evidence for this memory packet.
  loom_target_compile_report_memory_interval_t source_interval;
  // Exact source execution count plus one, or zero when unknown.
  uint64_t execution_count_plus_one;
} loom_target_compile_report_source_low_memory_row_t;

// Bank-service evidence grouped by one stable source and target packet shape.
typedef struct loom_target_compile_report_source_low_bank_service_summary_t {
  // Source function symbol containing the modeled memory packets.
  iree_string_view_t function_name;
  // Source operation mnemonic that emitted the modeled memory packets.
  iree_string_view_t source_op_name;
  // Numeric source operation kind that emitted the modeled memory packets.
  uint32_t source_op_kind;
  // Named source memory root selected by value facts, if available.
  iree_string_view_t source_root_name;
  // Source function entry argument index for the memory root, or UINT16_MAX.
  uint16_t source_root_argument_index;
  // Target-independent memory-space key selected by the target.
  iree_string_view_t memory_space;
  // Source memory operation kind selected by the target.
  iree_string_view_t operation_kind;
  // Stable target packet key selected for the modeled packets.
  iree_string_view_t packet_key;
  // Stable target-owned strategy key selected for the modeled packets.
  iree_string_view_t strategy_key;
  // Stable target packet-service model key.
  iree_string_view_t model_key;
  // Immutable source revision defining the selected model.
  iree_string_view_t model_revision;
  // Provenance strength of the selected model.
  iree_string_view_t model_evidence;
  // Treatment of requests to identical bank-word addresses.
  iree_string_view_t request_policy;
  // Stable reason shared by unknown rows, or empty when none or mixed.
  iree_string_view_t unknown_reason;
  // Whether unknown rows carried more than one stable reason.
  bool has_mixed_unknown_reasons;
  // Number of lanes represented by the model phases.
  uint8_t wave_size;
  // Number of independently serviced banks.
  uint8_t bank_count;
  // Byte width of one bank word.
  uint8_t bank_word_byte_count;
  // Number of consecutive bank words requested by each active lane.
  uint8_t packet_word_count;
  // Accumulated structural service evidence for the packet group.
  loom_target_compile_report_bank_service_summary_t summary;
} loom_target_compile_report_source_low_bank_service_summary_t;

// Subgroup access evidence grouped by one stable source and packet shape.
typedef struct loom_target_compile_report_source_low_subgroup_access_summary_t {
  // Source function symbol containing the modeled memory packets.
  iree_string_view_t function_name;
  // Source operation mnemonic that emitted the modeled memory packets.
  iree_string_view_t source_op_name;
  // Numeric source operation kind that emitted the modeled memory packets.
  uint32_t source_op_kind;
  // Named source memory root selected by value facts, if available.
  iree_string_view_t source_root_name;
  // Source function entry argument index for the root, or UINT16_MAX.
  uint16_t source_root_argument_index;
  // Target-independent memory-space key selected by the target.
  iree_string_view_t memory_space;
  // Source memory operation kind selected by the target.
  iree_string_view_t operation_kind;
  // Stable target packet key selected for the modeled packets.
  iree_string_view_t packet_key;
  // Stable target-owned strategy key selected for the modeled packets.
  iree_string_view_t strategy_key;
  // Exact or explicitly unknown subgroup address geometry shared by the group.
  loom_target_compile_report_subgroup_access_t access;
  // Accumulated structural and exact-dynamic coverage for the group.
  loom_target_compile_report_subgroup_access_summary_t summary;
} loom_target_compile_report_source_low_subgroup_access_summary_t;

// Summary of emitted source-memory packet shape.
typedef struct loom_target_compile_report_source_low_memory_summary_t {
  // Number of emitted source-memory packets.
  uint64_t packet_count;
  // Number of packets without exact dynamic contribution evidence.
  uint64_t unknown_dynamic_packet_count;
  // Number of packets with exact dynamic contribution evidence.
  uint64_t exact_dynamic_packet_count;
  // Number of load packets.
  uint64_t load_packet_count;
  // Number of store packets.
  uint64_t store_packet_count;
  // Number of packets moving exactly one source lane.
  uint64_t scalar_packet_count;
  // Number of packets moving more than one source lane.
  uint64_t vector_packet_count;
  // Number of source lanes represented by packets with known lane counts.
  uint64_t source_lane_count;
  // Logical source bytes represented by packets with known element sizes.
  uint64_t source_byte_count;
  // Logical source bytes read by load packets.
  uint64_t read_byte_count;
  // Logical source bytes written by store packets.
  uint64_t write_byte_count;
  // Target packet bytes read by emitted load effects with known widths.
  uint64_t issued_read_byte_count;
  // Target packet bytes written by emitted store effects with known widths.
  uint64_t issued_write_byte_count;
  // Number of emitted read effects without known byte-aligned widths.
  uint64_t issued_read_unknown_width_count;
  // Number of emitted write effects without known byte-aligned widths.
  uint64_t issued_write_unknown_width_count;
  // Dynamic source-memory packet executions with exact source trip counts.
  uint64_t dynamic_packet_count;
  // Dynamic source bytes represented by packets with known element sizes.
  uint64_t dynamic_source_byte_count;
  // Dynamic logical source bytes read by load packets.
  uint64_t dynamic_read_byte_count;
  // Dynamic logical source bytes written by store packets.
  uint64_t dynamic_write_byte_count;
  // Dynamic target packet bytes read by emitted load effects.
  uint64_t dynamic_issued_read_byte_count;
  // Dynamic target packet bytes written by emitted store effects.
  uint64_t dynamic_issued_write_byte_count;
  // Dynamic read effects without known byte-aligned widths.
  uint64_t dynamic_issued_read_unknown_width_count;
  // Dynamic write effects without known byte-aligned widths.
  uint64_t dynamic_issued_write_unknown_width_count;
  // Number of vector packets whose source lanes are element-contiguous.
  uint64_t contiguous_vector_packet_count;
  // Number of vector packets with a known non-contiguous source lane stride.
  uint64_t strided_vector_packet_count;
  // Number of vector packets without a usable source lane-stride fact.
  uint64_t unknown_stride_vector_packet_count;
  // Source byte interval envelope across packets with bounded intervals.
  loom_target_compile_report_memory_interval_summary_t interval_envelope;
  // Source byte interval envelope across load packets with bounded intervals.
  loom_target_compile_report_memory_interval_summary_t read_interval_envelope;
  // Source byte interval envelope across store packets with bounded intervals.
  loom_target_compile_report_memory_interval_summary_t write_interval_envelope;
} loom_target_compile_report_source_low_memory_summary_t;

// Summary of emitted source-memory packet shape grouped by source memory root.
typedef struct loom_target_compile_report_source_low_memory_root_summary_t {
  // Source function symbol containing the source memory root.
  iree_string_view_t function_name;
  // Named source memory root selected by value facts.
  iree_string_view_t source_root_name;
  // Source function entry argument index for the memory root, or UINT16_MAX.
  uint16_t source_root_argument_index;
  // Target-independent memory-space key selected by the target.
  iree_string_view_t memory_space;
  // Emitted source-memory packet shape for this root.
  loom_target_compile_report_source_low_memory_summary_t summary;
} loom_target_compile_report_source_low_memory_root_summary_t;

// Summary of emitted source-memory packet shape grouped by source function
// entry argument.
typedef struct loom_target_compile_report_source_low_memory_argument_summary_t {
  // Source function symbol containing the source memory argument.
  iree_string_view_t function_name;
  // Common named source memory root for this argument, if known and unique.
  iree_string_view_t source_root_name;
  // Source function entry argument index for the memory root.
  uint16_t source_root_argument_index;
  // Target-independent memory-space key selected by the target.
  iree_string_view_t memory_space;
  // Emitted source-memory packet shape for this argument.
  loom_target_compile_report_source_low_memory_summary_t summary;
} loom_target_compile_report_source_low_memory_argument_summary_t;

// Summary of emitted source-memory packet shape grouped by source function
// argument and selected target packet.
typedef struct
    loom_target_compile_report_source_low_memory_argument_packet_summary_t {
  // Source function symbol containing the source memory argument.
  iree_string_view_t function_name;
  // Common named source memory root for this argument, if known and unique.
  iree_string_view_t source_root_name;
  // Source function entry argument index for the memory root.
  uint16_t source_root_argument_index;
  // Target-independent memory-space key selected by the target.
  iree_string_view_t memory_space;
  // Source memory operation kind selected by the target.
  iree_string_view_t operation_kind;
  // Stable target packet key selected for this packet group.
  iree_string_view_t packet_key;
  // Stable target-owned strategy key selected for this packet group.
  iree_string_view_t strategy_key;
  // Stable target-owned reason key for strategy selection or fallback.
  iree_string_view_t fallback_reason;
  // Logical storage element format recovered from source encoding facts.
  iree_string_view_t storage_element_format;
  // Primary scale format recovered from source encoding facts.
  iree_string_view_t storage_scale_format;
  // Secondary scale format recovered from source encoding facts.
  iree_string_view_t storage_secondary_scale_format;
  // Physical payload packing recovered from source encoding facts.
  iree_string_view_t storage_payload_packing;
  // Scale topology recovered from source encoding facts.
  iree_string_view_t storage_scale_topology;
  // Affine payload interpretation recovered from source encoding facts.
  iree_string_view_t storage_affine_policy;
  // Rounding or finite-policy contract recovered from source encoding facts.
  iree_string_view_t storage_rounding_policy;
  // Codebook ownership contract recovered from source encoding facts.
  iree_string_view_t storage_codebook_policy;
  // Sparse metadata contract recovered from source encoding facts.
  iree_string_view_t storage_sparsity_policy;
  // Emitted source-memory packet shape for this argument and packet.
  loom_target_compile_report_source_low_memory_summary_t summary;
} loom_target_compile_report_source_low_memory_argument_packet_summary_t;

// Summary of emitted source-memory packet shape grouped by target strategy.
typedef struct loom_target_compile_report_source_low_memory_strategy_summary_t {
  // Source function symbol containing the strategy use.
  iree_string_view_t function_name;
  // Target-independent memory-space key selected by the target.
  iree_string_view_t memory_space;
  // Source memory operation kind selected by the target.
  iree_string_view_t operation_kind;
  // Stable target packet key selected for this packet group.
  iree_string_view_t packet_key;
  // Stable target-owned strategy key selected for this packet group.
  iree_string_view_t strategy_key;
  // Stable target-owned reason key for strategy selection or fallback.
  iree_string_view_t fallback_reason;
  // Logical storage element format recovered from source encoding facts.
  iree_string_view_t storage_element_format;
  // Primary scale format recovered from source encoding facts.
  iree_string_view_t storage_scale_format;
  // Secondary scale format recovered from source encoding facts.
  iree_string_view_t storage_secondary_scale_format;
  // Physical payload packing recovered from source encoding facts.
  iree_string_view_t storage_payload_packing;
  // Scale topology recovered from source encoding facts.
  iree_string_view_t storage_scale_topology;
  // Affine payload interpretation recovered from source encoding facts.
  iree_string_view_t storage_affine_policy;
  // Rounding or finite-policy contract recovered from source encoding facts.
  iree_string_view_t storage_rounding_policy;
  // Codebook ownership contract recovered from source encoding facts.
  iree_string_view_t storage_codebook_policy;
  // Sparse metadata contract recovered from source encoding facts.
  iree_string_view_t storage_sparsity_policy;
  // Emitted source-memory packet shape for this strategy.
  loom_target_compile_report_source_low_memory_summary_t summary;
} loom_target_compile_report_source_low_memory_strategy_summary_t;

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
  // Low descriptor key selected by the accepted rule, or empty.
  iree_string_view_t descriptor_key;
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
// compile options, config bindings, backend tables, or artifact storage. Detail
// row lists are owned by the report and allocated from |allocator| as rows are
// recorded.
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
  // Backend-facing target key carried by the emitted artifact, if any.
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
  // Canonical target artifact format name, if an artifact was produced.
  iree_string_view_t artifact_format;
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
  // Number of spill storage slots materialized before the final frame.
  uint64_t allocation_materialized_spill_storage_count;
  // Byte size of spill storage materialized before the final frame.
  uint64_t allocation_materialized_spill_storage_bytes;
  // Number of low.spill stores materialized before the final frame.
  uint64_t allocation_materialized_spill_store_count;
  // Byte traffic from low.spill stores materialized before the final frame.
  uint64_t allocation_materialized_spill_store_bytes;
  // Number of low.reload ops materialized before the final frame.
  uint64_t allocation_materialized_reload_count;
  // Byte traffic from low.reload ops materialized before the final frame.
  uint64_t allocation_materialized_reload_bytes;
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
  // Final target instruction-count decomposition.
  loom_target_compile_report_emission_breakdown_t emission_breakdown;
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
  // Per-workitem instruction and effect counters after multiplying by exact
  // statically-provable loop trip counts shared by all entries.
  loom_target_compile_report_static_instruction_mix_t dynamic_instruction_mix;
  // Final target resource and occupancy summary.
  loom_target_compile_report_target_resources_t target_resources;
  // Target wait-counter planning summary.
  loom_target_compile_report_wait_plan_t wait_plan;
  // Static launch workload facts shared by all entries in this report.
  loom_target_compile_report_workload_t workload;
  // Coarse target-low planning work and memory statistics across entries.
  loom_low_planning_statistics_t low_planning;
  // Target-inserted native packet counts across entries.
  loom_target_compile_report_target_insertion_summary_t
      target_insertion_summary;
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
  // Owned target wait-reason summary rows.
  loom_target_compile_report_row_list_t wait_reason_summary_rows;
  // Owned target wait-action rows.
  loom_target_compile_report_row_list_t wait_action_rows;
  // Owned target-inserted native packet rows.
  loom_target_compile_report_row_list_t target_insertion_rows;
  // Owned invocation config bindings materialized before compilation.
  loom_target_compile_report_row_list_t config_binding_rows;
  // Owned source-to-low selection rows.
  loom_target_compile_report_row_list_t source_low_rows;
  // Owned source function target-selection rows.
  loom_target_compile_report_row_list_t source_low_target_rows;
  // Owned source transform decision rows.
  loom_target_compile_report_row_list_t source_low_transform_rows;
  // Owned source-to-low selection summaries.
  loom_target_compile_report_row_list_t source_low_selection_summaries;
  // Owned emitted source-memory packet rows.
  loom_target_compile_report_row_list_t source_low_memory_rows;
  // Owned source-memory summaries grouped by named source memory root.
  loom_target_compile_report_row_list_t source_low_memory_root_summaries;
  // Owned source-memory summaries grouped by source function argument.
  loom_target_compile_report_row_list_t source_low_memory_argument_summaries;
  // Owned source-memory summaries grouped by argument and selected packet.
  loom_target_compile_report_row_list_t
      source_low_memory_argument_packet_summaries;
  // Owned source-memory summaries grouped by selected target strategy.
  loom_target_compile_report_row_list_t source_low_memory_strategy_summaries;
  // Owned bank-service summaries grouped by source root and target packet.
  loom_target_compile_report_row_list_t source_low_bank_service_summaries;
  // Owned subgroup-access summaries grouped by source root and target packet.
  loom_target_compile_report_row_list_t source_low_subgroup_access_summaries;
  // Derived summary of emitted source-memory packet shape.
  loom_target_compile_report_source_low_memory_summary_t
      source_low_memory_summary;
  // Derived structural bank-service evidence across emitted source packets.
  loom_target_compile_report_bank_service_summary_t bank_service_summary;
  // Derived subgroup access coverage across emitted source packets.
  loom_target_compile_report_subgroup_access_summary_t subgroup_access_summary;
  // Owned target math-legalization decision rows.
  loom_target_compile_report_row_list_t math_legalization_rows;
  // Owned target-legalization decision rows.
  loom_target_compile_report_row_list_t target_legalization_rows;
  // Owned selected-target capability rows.
  loom_target_compile_report_row_list_t target_capability_rows;
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

// Records spill traffic materialized while reaching the final frame.
void loom_target_compile_report_record_allocation_materialization(
    loom_target_compile_report_t* report, uint64_t spill_storage_count,
    uint64_t spill_storage_bytes, uint64_t spill_store_count,
    uint64_t spill_store_bytes, uint64_t reload_count, uint64_t reload_bytes);

// Records target move materialization attributed to one residual move cause.
void loom_target_compile_report_record_move_cause(
    loom_target_compile_report_t* report,
    loom_target_compile_report_move_cause_t cause, uint64_t packet_count,
    uint64_t unit_count);

// Records static descriptor-backed instruction-mix feature counters.
void loom_target_compile_report_record_static_instruction_mix(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_static_instruction_mix_t* mix);

// Records per-workitem instruction-mix counters weighted by exact static loop
// trip counts.
void loom_target_compile_report_record_dynamic_instruction_mix(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_static_instruction_mix_t* mix);

// Records target emission instruction and code-size summary counts in |report|.
void loom_target_compile_report_record_emission(
    loom_target_compile_report_t* report, uint64_t instruction_count,
    uint64_t code_byte_count, uint64_t code_storage_byte_count);

// Records an authoritative decomposition of final target instructions.
void loom_target_compile_report_record_emission_breakdown(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_emission_breakdown_t* breakdown);

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

// Records static launch workload facts in |report|.
void loom_target_compile_report_record_workload(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_workload_t* workload);

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

// Records one target wait-reason summary row. This does not update the
// aggregate wait-plan summary; callers that produce detail rows must also call
// loom_target_compile_report_record_wait_plan with their aggregate summary.
iree_status_t loom_target_compile_report_record_wait_reason_summary_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_wait_reason_summary_row_t* row);

// Records one target wait-action row.
iree_status_t loom_target_compile_report_record_wait_action_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_wait_action_row_t* row);

// Records one selected-target capability row.
iree_status_t loom_target_compile_report_record_target_capability_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_target_capability_row_t* row);

// Records one source-low row.
iree_status_t loom_target_compile_report_record_source_low_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_row_t* row);

// Records one materialized invocation config binding row.
iree_status_t loom_target_compile_report_record_config_binding_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_config_binding_row_t* row);

// Records one source function target-selection row.
iree_status_t loom_target_compile_report_record_source_low_target_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_target_row_t* row);

// Records one source transform decision row.
iree_status_t loom_target_compile_report_record_source_low_transform_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_transform_row_t* row);

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

#endif  // LOOM_TARGET_REPORTING_REPORT_H_
