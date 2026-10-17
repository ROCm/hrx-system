// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/cfg_condition_derivation.h"

#include <string.h>

#include "loom/ops/cfg/ops.h"
#include "loom/ops/low/ops.h"

iree_status_t loom_cfg_condition_derive_edges(
    const loom_module_t* module, const loom_cfg_graph_t* graph,
    const loom_value_fact_table_t* fact_table,
    loom_local_value_domain_t* value_domain, iree_arena_allocator_t* arena,
    loom_cfg_condition_edge_facts_t** out_edges) {
  *out_edges = NULL;
  if (graph->edge_count == 0) {
    return iree_ok_status();
  }
  loom_cfg_condition_edge_facts_t* edges = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, graph->edge_count, sizeof(*edges), (void**)&edges));
  memset(edges, 0, graph->edge_count * sizeof(*edges));
  loom_condition_query_t query;
  loom_condition_query_initialize(module, value_domain, arena, &query);
  loom_condition_derivation_t derivation;
  loom_condition_derivation_initialize(arena, &derivation);
  for (loom_cfg_edge_index_t index = 0; index < graph->edge_count; ++index) {
    const loom_cfg_edge_info_t* edge = &graph->edges[index];
    if (!loom_cfg_graph_block_is_reachable(graph, edge->source_block_index) ||
        !loom_cfg_graph_block_is_reachable(graph, edge->target_block_index) ||
        (!loom_cfg_cond_br_isa(edge->terminator) &&
         !loom_low_cond_br_isa(edge->terminator))) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_condition_facts_query_complete(
        &query, fact_table, edge->selector_value_id, edge->successor_index == 0,
        &derivation));
    if (derivation.integer_facts.integer_relation_count > UINT32_MAX ||
        derivation.boolean_fact_count > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "condition derivation exceeds uint32_t");
    }
    loom_cfg_condition_edge_facts_t* facts = &edges[index];
    const iree_host_size_t relation_count =
        derivation.integer_facts.integer_relation_count;
    if (relation_count != 0) {
      loom_condition_integer_relation_t* relations = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, relation_count, sizeof(*relations), (void**)&relations));
      memcpy(relations, derivation.integer_facts.integer_relations,
             relation_count * sizeof(*relations));
      facts->integer_relations = relations;
      facts->integer_relation_count = (uint32_t)relation_count;
    }
    if (derivation.boolean_fact_count != 0) {
      loom_condition_boolean_fact_t* booleans = NULL;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(arena, derivation.boolean_fact_count,
                                    sizeof(*booleans), (void**)&booleans));
      memcpy(booleans, derivation.boolean_facts,
             derivation.boolean_fact_count * sizeof(*booleans));
      facts->boolean_facts = booleans;
      facts->boolean_fact_count = (uint32_t)derivation.boolean_fact_count;
    }
  }
  *out_edges = edges;
  return iree_ok_status();
}
