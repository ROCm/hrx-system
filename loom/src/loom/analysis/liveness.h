// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Liveness, live intervals, and register-pressure summaries for Loom regions.
//
// The analysis is intentionally target-independent. It reads ordinary Loom SSA
// values, CFG successor edges, block arguments, branch operands, and SSA value
// references embedded in types. Target-specific consumers map the resulting
// pressure classes to allocation policies, diagnostics, and schedule scoring.
//
// Program points are instruction boundaries. A block's start point is before
// its first operation; operation operands are live into the current boundary,
// and operation results are defined at the following boundary. This lets a
// dead operand and a result share a register across the same instruction while
// still keeping values live from their defining boundary to their final use.

#ifndef LOOM_ANALYSIS_LIVENESS_H_
#define LOOM_ANALYSIS_LIVENESS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/ir/local_value_domain.h"

#ifdef __cplusplus
extern "C" {
#endif

// Class used for pressure accounting.
//
// Register values use LOOM_TYPE_REGISTER plus compact target-low descriptor
// identity so pressure groups naturally by descriptor-set register class.
// Non-register values use their type kind and element type with an invalid
// register class.
typedef struct loom_liveness_value_class_t {
  // Loom type kind carrying the value.
  loom_type_kind_t type_kind;
  // Element/scalar type for scalar and shaped semantic values.
  loom_scalar_type_t element_type;
  // Low descriptor-set stable ID for LOOM_TYPE_REGISTER, otherwise zero.
  uint64_t register_descriptor_set_stable_id;
  // Descriptor-set-local register class for LOOM_TYPE_REGISTER.
  uint16_t register_class_id;
} loom_liveness_value_class_t;

// Returns true when two values contribute to the same pressure class.
bool loom_liveness_value_class_equal(loom_liveness_value_class_t lhs,
                                     loom_liveness_value_class_t rhs);

// Half-open live interval for one value over a region-local program-point
// number line. An interval with start_point == end_point represents a value
// defined but not live across any point, such as a dead result.
typedef struct loom_liveness_interval_t {
  // SSA value represented by this interval.
  loom_value_id_t value_id;
  // First program point where the value is live or defined.
  uint32_t start_point;
  // One-past-last program point where the value is live.
  uint32_t end_point;
  // Pressure class used for grouped summaries.
  loom_liveness_value_class_t value_class;
  // Number of units contributed to |value_class| when live.
  uint32_t unit_count;
} loom_liveness_interval_t;

// Liveness for one block in the analyzed region.
typedef struct loom_liveness_block_info_t {
  // Region block represented by this record.
  const loom_block_t* block;
  // Program point before the block's first operation.
  uint32_t start_point;
  // Program point after the block's last operation.
  uint32_t end_point;
  // Values live at block entry.
  const loom_value_id_t* live_in_values;
  // Number of values in |live_in_values|.
  iree_host_size_t live_in_count;
  // Values live at block exit.
  const loom_value_id_t* live_out_values;
  // Number of values in |live_out_values|.
  iree_host_size_t live_out_count;
} loom_liveness_block_info_t;

// Peak boundary pressure for one value class.
//
// This target-independent summary reports values simultaneously live at
// block/op boundaries. Target-specific consumers combine it with descriptor
// operand/result constraints when estimating per-instruction transient
// pressure.
typedef struct loom_liveness_pressure_summary_t {
  // Pressure class being summarized.
  loom_liveness_value_class_t value_class;
  // Maximum boundary-live units observed for this class.
  uint32_t peak_live_units;
  // Maximum simultaneously live values observed at the same point.
  uint32_t peak_live_values;
  // Block containing the peak program point.
  const loom_block_t* peak_block;
  // Operation after which the peak was observed. NULL means block entry/exit.
  const loom_op_t* peak_op;
  // Program point associated with the peak.
  uint32_t peak_point;
} loom_liveness_pressure_summary_t;

// Register-pressure budget for one value class. UINT32_MAX means the
// corresponding live-units or live-values limit is disabled.
typedef struct loom_liveness_pressure_budget_t {
  // Pressure class constrained by this budget.
  loom_liveness_value_class_t value_class;
  // Maximum allowed boundary-live units for |value_class|.
  uint32_t max_live_units;
  // Maximum allowed boundary-live values for |value_class|.
  uint32_t max_live_values;
} loom_liveness_pressure_budget_t;

// Explicit operation order for one block.
//
// The order must be a permutation of operations in |block|. Analyses use this
// order only for intra-block program points; CFG live-in/live-out dataflow
// still follows the region's block successor structure.
typedef struct loom_liveness_block_order_t {
  // Block whose operation order is overridden.
  const loom_block_t* block;
  // Ordered operation pointers for |block|.
  const loom_op_t* const* ops;
  // Number of entries in |ops|.
  iree_host_size_t op_count;
} loom_liveness_block_order_t;

// Optional operation order for all blocks in a region.
typedef struct loom_liveness_order_t {
  // Per-block operation orders in region block order.
  const loom_liveness_block_order_t* blocks;
  // Number of entries in |blocks|.
  iree_host_size_t block_count;
} loom_liveness_order_t;

static inline loom_liveness_order_t loom_liveness_order_empty(void) {
  return (loom_liveness_order_t){0};
}

static inline bool loom_liveness_order_is_empty(loom_liveness_order_t order) {
  return order.block_count == 0;
}

enum loom_liveness_pressure_budget_violation_bits_e {
  // Peak live units exceeded the budget.
  LOOM_LIVENESS_PRESSURE_BUDGET_VIOLATION_LIVE_UNITS = 1u << 0,
  // Peak live values exceeded the budget.
  LOOM_LIVENESS_PRESSURE_BUDGET_VIOLATION_LIVE_VALUES = 1u << 1,
};
typedef uint32_t loom_liveness_pressure_budget_violation_flags_t;

// One pressure budget violation. The pointed-to summary is owned by the
// analysis; the record array itself is owned by the caller-provided arena.
typedef struct loom_liveness_pressure_budget_violation_t {
  // Index of the budget entry that was exceeded.
  iree_host_size_t budget_index;
  // Budget that was exceeded.
  loom_liveness_pressure_budget_t budget;
  // Analysis pressure summary that exceeded |budget|.
  const loom_liveness_pressure_summary_t* summary;
  // Bitfield describing which budget dimensions were exceeded.
  loom_liveness_pressure_budget_violation_flags_t violation_bits;
} loom_liveness_pressure_budget_violation_t;

enum loom_liveness_analysis_flag_bits_e {
  // Intervals include values defined in recursively nested structured regions.
  LOOM_LIVENESS_ANALYSIS_FLAG_REGION_TREE = 1u << 0,
};
typedef uint16_t loom_liveness_analysis_flags_t;

// Liveness analysis result for one region. All arrays are arena-owned by the
// caller-provided arena passed to loom_liveness_analyze_region.
typedef struct loom_liveness_analysis_t {
  // Module containing the analyzed region.
  const loom_module_t* module;
  // Region analyzed.
  const loom_region_t* region;
  // Analysis mode flags.
  loom_liveness_analysis_flags_t flags;
  // True when the region had CFG successor structure.
  bool is_cfg;
  // Per-block liveness summaries in region block order.
  const loom_liveness_block_info_t* blocks;
  // Number of records in |blocks|.
  iree_host_size_t block_count;
  // Live intervals for values touched by the region.
  const loom_liveness_interval_t* intervals;
  // Number of records in |intervals|.
  iree_host_size_t interval_count;
  // Value IDs indexed by region-local value ordinal.
  const loom_value_id_t* value_ids;
  // Number of records in |value_ids|.
  iree_host_size_t value_count;
  // Dense local-value-ordinal to interval-index table. Entries without an
  // interval contain UINT32_MAX. The table has |value_count| entries.
  const uint32_t* value_interval_indices;
  // Peak pressure summaries grouped by value class.
  const loom_liveness_pressure_summary_t* pressure_summaries;
  // Number of records in |pressure_summaries|.
  iree_host_size_t pressure_summary_count;
} loom_liveness_analysis_t;

// Returns true when |analysis| models recursively nested structured regions.
static inline bool loom_liveness_analysis_includes_region_tree(
    const loom_liveness_analysis_t* analysis) {
  return analysis && iree_any_bit_set(analysis->flags,
                                      LOOM_LIVENESS_ANALYSIS_FLAG_REGION_TREE);
}

// Computes liveness for |region|. The caller must keep |module| and |region|
// semantically immutable for as long as |out_analysis| is used and must keep
// |arena| alive. The analysis acquires a local value domain while it runs.
//
// CFG regions use explicit successor edges. Structured regions use local block
// use/def sets only; each block is analyzed independently because structured
// control-flow semantics are represented by the containing op, not by sibling
// block successor edges.
iree_status_t loom_liveness_analyze_region(
    loom_module_t* module, const loom_region_t* region,
    iree_arena_allocator_t* arena, loom_liveness_analysis_t* out_analysis);

// Computes liveness over an already-acquired local value domain.
//
// The domain must cover the analyzed region and remain acquired until the
// analysis completes. This lets adjacent compiler phases share one dense local
// value map instead of rebuilding module scratch ownership per analysis.
iree_status_t loom_liveness_analyze_local_value_domain(
    const loom_local_value_domain_t* value_domain, loom_liveness_order_t order,
    iree_arena_allocator_t* arena, loom_liveness_analysis_t* out_analysis);

// Computes liveness using an explicit per-block operation order.
//
// This is used by target-low packetization after scheduling: allocation must
// see intervals over the scheduled packet stream, not the source operation
// order. Pass an empty order to use ordinary source order.
iree_status_t loom_liveness_analyze_region_with_order(
    loom_module_t* module, const loom_region_t* region,
    loom_liveness_order_t order, iree_arena_allocator_t* arena,
    loom_liveness_analysis_t* out_analysis);

// Resolves |op|'s program point in |analysis|.
//
// For region-tree analyses, nested operations are located relative to the
// structured parent op that owns them. |order| applies only to direct
// operations in |analysis->region|; nested regions keep source order.
iree_status_t loom_liveness_op_program_point(
    const loom_liveness_analysis_t* analysis, loom_liveness_order_t order,
    const loom_op_t* op, uint32_t* out_program_point);

// Returns the interval for |value_id|, or NULL when the value is not touched by
// the analyzed region. This convenience helper scans the compact local value
// list; production hot paths should keep their own frame-local direct lookup.
const loom_liveness_interval_t* loom_liveness_interval_for_value(
    const loom_liveness_analysis_t* analysis, loom_value_id_t value_id);

// Returns the interval for |value_ordinal|, or NULL when that local value has
// no interval.
const loom_liveness_interval_t* loom_liveness_interval_for_value_ordinal(
    const loom_liveness_analysis_t* analysis,
    loom_value_ordinal_t value_ordinal);

// Returns the block record for |block|, or NULL when |block| is not owned by
// the analyzed region.
const loom_liveness_block_info_t* loom_liveness_block_info_for_block(
    const loom_liveness_analysis_t* analysis, const loom_block_t* block);

// Collects all pressure summaries that exceed |budgets|. Missing value classes
// do not violate a budget because no value in that class was live in the
// analyzed region.
iree_status_t loom_liveness_collect_pressure_budget_violations(
    const loom_liveness_analysis_t* analysis,
    const loom_liveness_pressure_budget_t* budgets,
    iree_host_size_t budget_count, iree_arena_allocator_t* arena,
    const loom_liveness_pressure_budget_violation_t** out_violations,
    iree_host_size_t* out_violation_count);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_LIVENESS_H_
