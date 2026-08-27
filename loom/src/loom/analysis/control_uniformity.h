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
  // Arena receiving lazily constructed CFG control summaries.
  iree_arena_allocator_t* arena;
  // CFG region summaries constructed by prior queries.
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
// walk; CFG summaries are derived only for regions reached by later queries.
void loom_control_uniformity_info_initialize(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    iree_arena_allocator_t* arena, loom_control_uniformity_info_t* out_info);

// Proves that every structured and CFG control value governing |op| is uniform
// at |required_scope|. CFG construction is reused from |fact_table|. The first
// query in a CFG region constructs a near-linear postdominator summary; later
// queries in that region are allocation-free block lookups. Infrastructure
// failures are returned as status. An ordinary failed proof writes false to
// |out_proven| and describes one insufficient controller in |out_failure|.
iree_status_t loom_control_uniformity_prove_execution(
    loom_control_uniformity_info_t* info, const loom_op_t* op,
    loom_value_fact_uniform_scope_t required_scope,
    loom_control_uniformity_failure_t* out_failure, bool* out_proven);

// Proves that every operation in |lhs_ops| and |rhs_ops| executes on distinct
// alternatives of a common CFG controller whose selector is uniform at
// |required_scope|. Controllers inside a CFG cycle are rejected because
// distinct alternatives may execute on different loop iterations. Different
// regions, structured-only control, and incomplete CFG facts conservatively
// produce a failed proof.
//
// The first query in a CFG region lazily retains its control-dependence edges
// and cycle membership. Later queries compare those retained summaries without
// walking IR or recomputing graph structure. Infrastructure failures are
// returned as status; an ordinary failed proof writes false to |out_proven|.
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
