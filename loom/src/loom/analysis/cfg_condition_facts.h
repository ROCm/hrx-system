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
#include "loom/analysis/condition_facts.h"
#include "loom/ir/ir.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/dominance.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_CFG_CONDITION_FACT_RELATION_CAPACITY 32

typedef struct loom_cfg_block_entry_condition_facts_t {
  // SSA condition proven by every reachable predecessor edge into the block.
  loom_value_id_t condition;
  // Boolean value proven for condition at block entry.
  bool condition_value;
  // True when condition and condition_value contain a usable entry fact.
  bool condition_known;
  // Integer relations proven by every reachable predecessor edge.
  const loom_condition_integer_relation_t* integer_relations;
  // Number of valid entries in integer_relations.
  iree_host_size_t integer_relation_count;
} loom_cfg_block_entry_condition_facts_t;

typedef struct loom_cfg_condition_fact_table_t {
  // Per-block facts in the same dense order as graph->blocks.
  loom_cfg_block_entry_condition_facts_t* block_facts;
  // Number of entries in block_facts.
  iree_host_size_t block_count;
} loom_cfg_condition_fact_table_t;

// Computes the facts implied at the entry of |block| along one predecessor
// edge.
//
// |current_facts| may be NULL, or it may point at a block-indexed fact table
// from a previous fixed-point iteration. |relation_storage| is caller-owned
// scratch storage that remains owned by the caller; out_fact references it
// until the caller copies the relations elsewhere.
void loom_cfg_condition_facts_compute_predecessor_edge(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_dominance_info_t* dominance, const loom_block_t* block,
    const loom_op_t* predecessor_terminator, uint16_t predecessor_index,
    const loom_cfg_block_entry_condition_facts_t* current_facts,
    loom_condition_integer_relation_t* relation_storage,
    iree_host_size_t relation_capacity,
    loom_cfg_block_entry_condition_facts_t* out_fact);

// Computes a fixed-point table of block-entry condition facts for |graph|.
//
// Malformed graphs and empty regions produce an empty table so verifiers can
// own user-facing CFG diagnostics.
iree_status_t loom_cfg_condition_fact_table_compute(
    const loom_module_t* module, const loom_cfg_graph_t* graph,
    const loom_value_fact_table_t* fact_table,
    const loom_dominance_info_t* dominance, iree_arena_allocator_t* arena,
    loom_cfg_condition_fact_table_t* out_table);

// Returns the condition fact set for |block_index|, or NULL if unavailable.
const loom_cfg_block_entry_condition_facts_t*
loom_cfg_condition_fact_table_block(
    const loom_cfg_condition_fact_table_t* table, uint16_t block_index);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_CFG_CONDITION_FACTS_H_
