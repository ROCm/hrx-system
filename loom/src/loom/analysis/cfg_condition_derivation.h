// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared edge-local inputs to condition analyses over an immutable CFG.

#ifndef LOOM_ANALYSIS_CFG_CONDITION_DERIVATION_H_
#define LOOM_ANALYSIS_CFG_CONDITION_DERIVATION_H_

#include "loom/analysis/condition_facts.h"
#include "loom/util/cfg_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_cfg_condition_edge_facts_t {
  // Complete integer relations implied by this edge's selector outcome.
  const loom_condition_integer_relation_t* integer_relations;
  // Exact Boolean truths implied by the same outcome.
  const loom_condition_boolean_fact_t* boolean_facts;
  // Number of entries in integer_relations.
  uint32_t integer_relation_count;
  // Number of entries in boolean_facts.
  uint32_t boolean_fact_count;
} loom_cfg_condition_edge_facts_t;

// Derives each reachable conditional edge once. Entries are indexed by the
// retained graph's edge ordinal; other edges have empty facts. Storage belongs
// to |arena| and is construction input, not retained analysis query state.
iree_status_t loom_cfg_condition_derive_edges(
    const loom_module_t* module, const loom_cfg_graph_t* graph,
    const loom_value_fact_table_t* fact_table,
    loom_local_value_domain_t* value_domain, iree_arena_allocator_t* arena,
    loom_cfg_condition_edge_facts_t** out_edges);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_CFG_CONDITION_DERIVATION_H_
