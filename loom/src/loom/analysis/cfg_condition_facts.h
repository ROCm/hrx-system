// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Path-sensitive condition facts for explicit CFG regions.
//
// This analysis derives the facts that are true at each CFG block entry from
// predecessor branch conditions. It is intentionally separate from the generic
// condition extractor so non-CFG users do not pull in graph or dominance
// dependencies.

#ifndef LOOM_ANALYSIS_CFG_CONDITION_FACTS_H_
#define LOOM_ANALYSIS_CFG_CONDITION_FACTS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/cfg_condition_derivation.h"
#include "loom/analysis/cfg_condition_relation_anchors.h"
#include "loom/analysis/cfg_value_identity.h"
#include "loom/analysis/condition_facts.h"
#include "loom/analysis/condition_relation_matrix.h"
#include "loom/ir/ir.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/dominance.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_cfg_condition_operand_domain_t
    loom_cfg_condition_operand_domain_t;

// Immutable condition facts at one block entry or along one CFG edge.
typedef struct loom_cfg_condition_relation_view_t {
  // Factorized integer relation rows over the table's operand domain.
  loom_condition_relation_matrix_view_t integer_relations;

  // Exact-false and exact-true Boolean value sets.
  loom_condition_relation_set_id_t boolean_values[2];
} loom_cfg_condition_relation_view_t;

// Complete finite condition facts for one immutable CFG snapshot.
typedef struct loom_cfg_condition_relation_table_t {
  // Compact operand domain shared by every relation view.
  const loom_cfg_condition_operand_domain_t* operand_domain;

  // Immutable set nodes shared by every relation and Boolean root.
  loom_condition_relation_set_index_t set_index;

  // Block views followed by separately retained predecessor-edge views.
  const loom_cfg_condition_relation_view_t* views;

  // View ordinal for each stable CFG edge, or UINT32_MAX when unavailable.
  const uint32_t* edge_view_indices;

  // Number of entries in views.
  uint32_t view_count;

  // Number of leading block views in views.
  uint32_t block_count;

  // Number of entries in edge_view_indices.
  uint32_t edge_count;

  // Number of sparse derived-anchor entries stored after operand_domain.
  uint32_t derived_anchor_count;
} loom_cfg_condition_relation_table_t;

// Visits one strongest retained relation incident to an anchored operand.
// Returning false stops iteration.
typedef bool (*loom_cfg_condition_relation_visit_fn_t)(
    void* user_data, const loom_condition_integer_relation_t* relation);

// Computes complete block-entry and predecessor-edge condition facts for
// |graph| with finite monotone propagation. The local value domain and exact
// identity table must cover the graph and remain valid for the returned table
// lifetime. When provided, |anchor_provider| is queried once per SSA relation
// operand and the table retains direct incidence from each emitted anchor to
// the original operand. Construction scratch is released before returning;
// only compact immutable views remain in |arena|.
iree_status_t loom_cfg_condition_relation_table_compute(
    const loom_module_t* module, const loom_cfg_graph_t* graph,
    const loom_value_fact_table_t* fact_table,
    const loom_dominance_info_t* dominance,
    loom_local_value_domain_t* value_domain,
    const loom_cfg_value_identity_table_t* identities,
    const loom_cfg_condition_relation_anchor_provider_t* anchor_provider,
    iree_arena_allocator_t* arena,
    loom_cfg_condition_relation_table_t* out_table);

// Computes the same retained relation table from shared edge derivations.
// |edge_facts| contains graph->edge_count entries and is borrowed only during
// the call. Consumers projecting other compact facts can share those inputs
// without repeating condition-expression traversal or retaining proof scratch.
iree_status_t loom_cfg_condition_relation_table_compute_from_edges(
    const loom_module_t* module, const loom_cfg_graph_t* graph,
    const loom_value_fact_table_t* fact_table,
    const loom_dominance_info_t* dominance,
    loom_local_value_domain_t* value_domain,
    const loom_cfg_value_identity_table_t* identities,
    const loom_cfg_condition_relation_anchor_provider_t* anchor_provider,
    const loom_cfg_condition_edge_facts_t* edge_facts,
    iree_arena_allocator_t* scratch_arena, iree_arena_allocator_t* arena,
    loom_cfg_condition_relation_table_t* out_table);

// Returns the immutable facts at |block_index|, or NULL when unavailable.
const loom_cfg_condition_relation_view_t*
loom_cfg_condition_relation_table_block(
    const loom_cfg_condition_relation_table_t* table, uint16_t block_index);

// Returns the immutable facts along |edge_index| after payload projection, or
// NULL when unavailable.
const loom_cfg_condition_relation_view_t*
loom_cfg_condition_relation_table_edge(
    const loom_cfg_condition_relation_table_t* table,
    loom_cfg_edge_index_t edge_index);

// Returns true when |value_id| has one exact Boolean result in |view|.
bool loom_cfg_condition_relation_view_query_boolean(
    const loom_cfg_condition_relation_table_t* table,
    const loom_cfg_condition_relation_view_t* view, loom_value_id_t value_id,
    bool* out_value);

// Returns the comparison outcomes excluded by |view| for an ordered operand
// pair. Exact ambient integer facts participate in operand identity so an SSA
// constant and its literal value query the same retained relation.
loom_integer_order_outcome_bits_t
loom_cfg_condition_relation_view_query_excluded_outcomes(
    const loom_cfg_condition_relation_table_t* table,
    const loom_cfg_condition_relation_view_t* view,
    const loom_value_fact_table_t* fact_table,
    loom_condition_integer_operand_t left,
    loom_condition_integer_operand_t right);

// Visits retained relations incident to |anchor| without scanning unrelated
// rows. Exact ambient integer facts participate in operand identity. Returns
// false when |visit| stops iteration.
bool loom_cfg_condition_relation_view_for_each_while(
    const loom_cfg_condition_relation_table_t* table,
    const loom_cfg_condition_relation_view_t* view,
    const loom_value_fact_table_t* fact_table,
    loom_condition_integer_operand_t anchor,
    loom_cfg_condition_relation_visit_fn_t visit, void* user_data);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_CFG_CONDITION_FACTS_H_
