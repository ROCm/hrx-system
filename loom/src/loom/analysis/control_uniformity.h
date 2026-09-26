// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scope-aware execution-uniformity analysis for nested control flow.

#ifndef LOOM_ANALYSIS_CONTROL_UNIFORMITY_H_
#define LOOM_ANALYSIS_CONTROL_UNIFORMITY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/condition_facts.h"
#include "loom/ir/facts.h"
#include "loom/ir/ir.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Control source that prevented an execution-uniformity proof.
typedef enum loom_control_uniformity_source_e {
  LOOM_CONTROL_UNIFORMITY_SOURCE_REGION_SELECTOR = 0,
  LOOM_CONTROL_UNIFORMITY_SOURCE_LOOP_LOWER_BOUND = 1,
  LOOM_CONTROL_UNIFORMITY_SOURCE_LOOP_UPPER_BOUND = 2,
  LOOM_CONTROL_UNIFORMITY_SOURCE_LOOP_STEP = 3,
  LOOM_CONTROL_UNIFORMITY_SOURCE_LOOP_CONDITION = 4,
  LOOM_CONTROL_UNIFORMITY_SOURCE_CFG_SELECTOR = 5,
  LOOM_CONTROL_UNIFORMITY_SOURCE_CFG_EXECUTION = 6,
} loom_control_uniformity_source_t;

// One control value whose facts are weaker than the required scope.
typedef struct loom_control_uniformity_failure_t {
  // Op that owns the controlling value or condition.
  const loom_op_t* control_op;
  // Controlling SSA value, or LOOM_VALUE_ID_INVALID when an interface does not
  // expose the condition value.
  loom_value_id_t control_value;
  // Facts available for control_value, or unknown when it is not exposed.
  loom_value_facts_t control_facts;
  // Semantic role of control_value on control_op.
  loom_control_uniformity_source_t source;
} loom_control_uniformity_failure_t;

typedef struct loom_control_uniformity_cfg_region_t
    loom_control_uniformity_cfg_region_t;

// Reusable execution-uniformity analysis over one populated value-fact scope.
typedef struct loom_control_uniformity_info_t {
  // Module containing analyzed operations.
  const loom_module_t* module;
  // Populated value facts and cached CFG graphs borrowed for the query
  // lifetime.
  const loom_value_fact_table_t* fact_table;
  // Arena receiving dominance and scratch for mutual-exclusion queries.
  iree_arena_allocator_t* arena;
  // Dominance and query scratch for CFG snapshots reached by exclusion queries.
  struct {
    // Open-addressed slots keyed by region address.
    loom_control_uniformity_cfg_region_t** slots;
    // Power-of-two slot capacity.
    iree_host_size_t capacity;
    // Number of populated slots.
    iree_host_size_t count;
  } cfg_regions;
} loom_control_uniformity_info_t;

// Initializes an empty reusable analysis. This performs no allocation or IR
// walk. The analysis borrows the fact scope and its current CFG snapshots;
// replacing a snapshot ends the lifetime of query scratch referring to it.
void loom_control_uniformity_info_initialize(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    iree_arena_allocator_t* arena, loom_control_uniformity_info_t* out_info);

// Proves that every structured and CFG control value governing |op| is uniform
// at |required_scope|. CFG execution distributions are direct lookups in the
// fact scope's maintained control result. Returns false when participation
// cannot be proven. If |out_failure| is supplied, the fact owner materializes
// diagnostic witnesses once per changed scope before describing an insufficient
// controller. This query does not allocate.
bool loom_control_uniformity_prove_execution(
    const loom_control_uniformity_info_t* info, const loom_op_t* op,
    loom_value_fact_uniform_scope_t required_scope,
    loom_control_uniformity_failure_t* out_failure);

// Proves that every entry into |block| selects one Boolean branch outcome from
// execution uniform at |required_scope|. The returned condition describes the
// entire active subset of that entry, including false-edge polarity. It need
// not itself be uniform or numerically decidable.
//
// Requires a reachable non-entry block with exactly one incoming CFG edge;
// backedges and duplicate successors count as entries. The retained graph and
// execution facts establish the proof without rebuilding or scanning the CFG.
bool loom_control_uniformity_prove_single_entry(
    const loom_control_uniformity_info_t* info, const loom_block_t* block,
    loom_value_fact_uniform_scope_t required_scope,
    loom_condition_assumption_t* out_condition);

// Proves that every operation in |lhs_ops| and |rhs_ops| executes on disjoint
// alternatives of a common RegionBranch or CFG controller whose selector is
// uniform at |required_scope|. Structured operations are matched by ancestor
// region. A CFG alternative's target must dominate its footprint, and its
// retained entry predecessor proves that the choice cannot be bypassed.
// Controllers inside loops or CFG cycles are rejected because distinct
// alternatives may execute on different iterations. Incomplete ancestry or
// CFG facts conservatively produce a failed proof.
//
// Structured queries follow operation ancestry. The first CFG query for a
// region builds and retains dominance and mandatory entry choices from the fact
// scope's graph, alongside reusable query scratch; later queries follow that
// tree and the graph's cycle membership. No query scans sibling operations or
// rebuilds the CFG. Allocation failures are returned as status; an ordinary
// failed proof writes false to |out_proven|.
iree_status_t loom_control_uniformity_prove_mutually_exclusive_execution(
    loom_control_uniformity_info_t* info, iree_host_size_t lhs_op_count,
    const loom_op_t* const* lhs_ops, iree_host_size_t rhs_op_count,
    const loom_op_t* const* rhs_ops,
    loom_value_fact_uniform_scope_t required_scope, bool* out_proven);

// Returns the stable diagnostic name for a control source.
iree_string_view_t loom_control_uniformity_source_name(
    loom_control_uniformity_source_t source);

// Returns the stable diagnostic name for a required uniform scope.
iree_string_view_t loom_control_uniformity_scope_name(
    loom_value_fact_uniform_scope_t scope);

// Returns the stable diagnostic name for the distribution encoded by |facts|.
iree_string_view_t loom_control_uniformity_fact_distribution_name(
    loom_value_facts_t facts);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_CONTROL_UNIFORMITY_H_
