// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/control_uniformity.h"

#include <string.h>

#include "loom/analysis/scc.h"
#include "loom/ops/op_defs.h"
#include "loom/util/cfg_graph.h"

#define LOOM_CONTROL_UNIFORMITY_NODE_INVALID UINT32_MAX
#define LOOM_CONTROL_UNIFORMITY_RECORD_INVALID UINT32_MAX

typedef enum loom_control_uniformity_cfg_node_flag_bits_e {
  LOOM_CONTROL_UNIFORMITY_CFG_NODE_CAN_REACH_EXIT = 1u << 0,
  LOOM_CONTROL_UNIFORMITY_CFG_NODE_IS_CYCLIC = 1u << 1,
} loom_control_uniformity_cfg_node_flag_bits_t;
typedef uint8_t loom_control_uniformity_cfg_node_flags_t;

typedef struct loom_control_uniformity_cfg_record_t {
  // CFG edge selecting the control alternative.
  loom_cfg_edge_index_t edge_index;
  // Next control record for the same block.
  uint32_t next_record_index;
} loom_control_uniformity_cfg_record_t;

typedef struct loom_control_uniformity_cfg_node_t {
  // One-based reverse-CFG depth-first-search number, or zero when unvisited.
  uint32_t dfs_number;
  // Parent node in the reverse-CFG depth-first-search tree.
  uint32_t dfs_parent;
  // Lengauer-Tarjan semidominator DFS number.
  uint32_t semidominator;
  // Best semidominator representative maintained by path compression.
  uint32_t label;
  // Lengauer-Tarjan union-forest ancestor.
  uint32_t ancestor;
  // Immediate postdominator, represented as a dense analysis node index.
  uint32_t immediate_postdominator;
  // Head node in the semidominator bucket collision chain.
  uint32_t bucket_head;
  // Next node in the semidominator bucket collision chain.
  uint32_t bucket_next;
  // Reused first as a DFS successor cursor and then as a path-skip parent.
  uint32_t scratch;
  // CFG edge identifying the weakest controller assigned to this block.
  loom_cfg_edge_index_t controller_edge_index;
  // Head of the lazily retained control-alternative record list.
  uint32_t control_record_head;
  // Depth in the postdominator tree.
  uint32_t postdominator_depth;
  // Strongest execution scope proven for the block.
  loom_value_fact_uniform_scope_t execution_scope;
  // Analysis flags from loom_control_uniformity_cfg_node_flag_bits_t.
  loom_control_uniformity_cfg_node_flags_t flags;
} loom_control_uniformity_cfg_node_t;

struct loom_control_uniformity_cfg_region_t {
  // CFG region summarized by this entry.
  const loom_region_t* region;
  // CFG graph borrowed from the value fact table.
  const loom_cfg_graph_t* graph;
  // Per-block analysis nodes followed by one synthetic exit node.
  loom_control_uniformity_cfg_node_t* nodes;
  // Synthetic exit node index, equal to graph->block_count.
  uint32_t exit_node;
  // Control-alternative records shared by per-block intrusive lists.
  loom_control_uniformity_cfg_record_t* control_records;
  // Number of initialized control-alternative records.
  uint32_t control_record_count;
  // Allocated control-alternative record capacity.
  uint32_t control_record_capacity;
  // Per-block generation marks used by control-context queries.
  uint32_t* query_marks;
  // Current nonzero generation in |query_marks|.
  uint32_t query_generation;
  // Block traversal stack used by control-context queries.
  uint32_t* query_stack;
  // Collected left-hand control-context edges.
  loom_cfg_edge_index_t* query_lhs_edges;
  // Collected right-hand control-context edges.
  loom_cfg_edge_index_t* query_rhs_edges;
  // True after control alternatives and cyclic controllers are retained.
  bool exclusivity_initialized;
};

static iree_host_size_t loom_control_uniformity_region_hash(
    const loom_region_t* region) {
  uintptr_t bits = (uintptr_t)region;
  bits ^= bits >> 17;
  bits *= (uintptr_t)0xed5ad4bbU;
  bits ^= bits >> 11;
  return (iree_host_size_t)bits;
}

static void loom_control_uniformity_insert_cfg_region(
    loom_control_uniformity_cfg_region_t** slots, iree_host_size_t capacity,
    loom_control_uniformity_cfg_region_t* summary) {
  iree_host_size_t slot_index =
      loom_control_uniformity_region_hash(summary->region) & (capacity - 1);
  while (slots[slot_index]) {
    slot_index = (slot_index + 1) & (capacity - 1);
  }
  slots[slot_index] = summary;
}

static iree_status_t loom_control_uniformity_reserve_cfg_regions(
    loom_control_uniformity_info_t* info, iree_host_size_t minimum_count) {
  iree_host_size_t capacity = info->cfg_regions.capacity;
  if (capacity == 0) capacity = 8;
  while (minimum_count > capacity - capacity / 4) {
    if (capacity > SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "control summary capacity overflow");
    }
    capacity *= 2;
  }
  if (capacity == info->cfg_regions.capacity) return iree_ok_status();

  loom_control_uniformity_cfg_region_t** slots = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      info->arena, capacity, sizeof(*slots), (void**)&slots));
  memset(slots, 0, capacity * sizeof(*slots));
  for (iree_host_size_t i = 0; i < info->cfg_regions.capacity; ++i) {
    loom_control_uniformity_cfg_region_t* summary = info->cfg_regions.slots[i];
    if (summary) {
      loom_control_uniformity_insert_cfg_region(slots, capacity, summary);
    }
  }
  info->cfg_regions.slots = slots;
  info->cfg_regions.capacity = capacity;
  return iree_ok_status();
}

static loom_control_uniformity_cfg_region_t*
loom_control_uniformity_lookup_cfg_region(
    const loom_control_uniformity_info_t* info, const loom_region_t* region) {
  if (info->cfg_regions.capacity == 0) return NULL;
  iree_host_size_t slot_index = loom_control_uniformity_region_hash(region) &
                                (info->cfg_regions.capacity - 1);
  while (info->cfg_regions.slots[slot_index]) {
    loom_control_uniformity_cfg_region_t* summary =
        info->cfg_regions.slots[slot_index];
    if (summary->region == region) return summary;
    slot_index = (slot_index + 1) & (info->cfg_regions.capacity - 1);
  }
  return NULL;
}

static bool loom_control_uniformity_cfg_block_is_synthetic_exit(
    const loom_control_uniformity_cfg_region_t* summary, uint32_t block_index) {
  const loom_control_uniformity_cfg_node_t* node = &summary->nodes[block_index];
  const loom_cfg_block_index_span_t successors =
      loom_cfg_graph_successors(summary->graph, (uint16_t)block_index);
  return successors.count == 0 ||
         !iree_any_bit_set(node->flags,
                           LOOM_CONTROL_UNIFORMITY_CFG_NODE_CAN_REACH_EXIT);
}

static void loom_control_uniformity_cfg_mark_exit_reachability(
    loom_control_uniformity_cfg_region_t* summary, uint32_t* stack) {
  iree_host_size_t stack_count = 0;
  for (uint32_t block_index = 0; block_index < summary->exit_node;
       ++block_index) {
    if (!loom_cfg_graph_block_is_reachable(summary->graph,
                                           (uint16_t)block_index) ||
        loom_cfg_graph_successors(summary->graph, (uint16_t)block_index)
                .count != 0) {
      continue;
    }
    summary->nodes[block_index].flags |=
        LOOM_CONTROL_UNIFORMITY_CFG_NODE_CAN_REACH_EXIT;
    stack[stack_count++] = block_index;
  }
  while (stack_count > 0) {
    const uint32_t block_index = stack[--stack_count];
    const loom_cfg_block_index_span_t predecessors =
        loom_cfg_graph_predecessors(summary->graph, (uint16_t)block_index);
    for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
      const uint32_t predecessor_index = predecessors.values[i];
      loom_control_uniformity_cfg_node_t* predecessor =
          &summary->nodes[predecessor_index];
      if (!loom_cfg_graph_block_is_reachable(summary->graph,
                                             (uint16_t)predecessor_index) ||
          iree_any_bit_set(predecessor->flags,
                           LOOM_CONTROL_UNIFORMITY_CFG_NODE_CAN_REACH_EXIT)) {
        continue;
      }
      predecessor->flags |= LOOM_CONTROL_UNIFORMITY_CFG_NODE_CAN_REACH_EXIT;
      stack[stack_count++] = predecessor_index;
    }
  }
}

static bool loom_control_uniformity_cfg_next_reverse_successor(
    loom_control_uniformity_cfg_region_t* summary, uint32_t node_index,
    uint32_t* out_successor_index) {
  loom_control_uniformity_cfg_node_t* node = &summary->nodes[node_index];
  if (node_index == summary->exit_node) {
    while (node->scratch < summary->exit_node) {
      const uint32_t candidate_index = node->scratch++;
      if (loom_cfg_graph_block_is_reachable(summary->graph,
                                            (uint16_t)candidate_index) &&
          loom_control_uniformity_cfg_block_is_synthetic_exit(
              summary, candidate_index)) {
        *out_successor_index = candidate_index;
        return true;
      }
    }
    return false;
  }

  const loom_cfg_block_index_span_t predecessors =
      loom_cfg_graph_predecessors(summary->graph, (uint16_t)node_index);
  while (node->scratch < predecessors.count) {
    const uint32_t candidate_index = predecessors.values[node->scratch++];
    if (loom_cfg_graph_block_is_reachable(summary->graph,
                                          (uint16_t)candidate_index)) {
      *out_successor_index = candidate_index;
      return true;
    }
  }
  return false;
}

static uint32_t loom_control_uniformity_cfg_build_reverse_dfs(
    loom_control_uniformity_cfg_region_t* summary, uint32_t* vertex_by_dfs,
    uint32_t* stack) {
  uint32_t dfs_count = 1;
  loom_control_uniformity_cfg_node_t* exit =
      &summary->nodes[summary->exit_node];
  exit->dfs_number = dfs_count;
  exit->dfs_parent = LOOM_CONTROL_UNIFORMITY_NODE_INVALID;
  vertex_by_dfs[dfs_count] = summary->exit_node;

  iree_host_size_t stack_count = 0;
  stack[stack_count++] = summary->exit_node;
  while (stack_count > 0) {
    const uint32_t node_index = stack[stack_count - 1];
    uint32_t successor_index = LOOM_CONTROL_UNIFORMITY_NODE_INVALID;
    if (!loom_control_uniformity_cfg_next_reverse_successor(summary, node_index,
                                                            &successor_index)) {
      --stack_count;
      continue;
    }
    loom_control_uniformity_cfg_node_t* successor =
        &summary->nodes[successor_index];
    if (successor->dfs_number != 0) continue;
    successor->dfs_number = ++dfs_count;
    successor->dfs_parent = node_index;
    vertex_by_dfs[dfs_count] = successor_index;
    stack[stack_count++] = successor_index;
  }
  return dfs_count;
}

static bool loom_control_uniformity_cfg_next_reverse_predecessor(
    const loom_control_uniformity_cfg_region_t* summary, uint32_t node_index,
    uint32_t* cursor, uint32_t* out_predecessor_index) {
  if (node_index == summary->exit_node) return false;
  const loom_cfg_block_index_span_t successors =
      loom_cfg_graph_successors(summary->graph, (uint16_t)node_index);
  while (*cursor < successors.count) {
    const uint32_t candidate_index = successors.values[(*cursor)++];
    if (summary->nodes[candidate_index].dfs_number != 0) {
      *out_predecessor_index = candidate_index;
      return true;
    }
  }
  if (*cursor == successors.count) {
    ++*cursor;
    if (loom_control_uniformity_cfg_block_is_synthetic_exit(summary,
                                                            node_index)) {
      *out_predecessor_index = summary->exit_node;
      return true;
    }
  }
  return false;
}

static uint32_t loom_control_uniformity_cfg_eval(
    loom_control_uniformity_cfg_node_t* nodes, uint32_t node_index,
    uint32_t* stack) {
  if (nodes[node_index].ancestor == LOOM_CONTROL_UNIFORMITY_NODE_INVALID) {
    return nodes[node_index].label;
  }

  iree_host_size_t stack_count = 0;
  uint32_t current_index = node_index;
  while (nodes[current_index].ancestor !=
             LOOM_CONTROL_UNIFORMITY_NODE_INVALID &&
         nodes[nodes[current_index].ancestor].ancestor !=
             LOOM_CONTROL_UNIFORMITY_NODE_INVALID) {
    stack[stack_count++] = current_index;
    current_index = nodes[current_index].ancestor;
  }
  while (stack_count > 0) {
    current_index = stack[--stack_count];
    const uint32_t ancestor_index = nodes[current_index].ancestor;
    if (nodes[nodes[ancestor_index].label].semidominator <
        nodes[nodes[current_index].label].semidominator) {
      nodes[current_index].label = nodes[ancestor_index].label;
    }
    nodes[current_index].ancestor = nodes[ancestor_index].ancestor;
  }
  return nodes[node_index].label;
}

static void loom_control_uniformity_cfg_compute_postdominators(
    loom_control_uniformity_cfg_region_t* summary, uint32_t dfs_count,
    const uint32_t* vertex_by_dfs, uint32_t* stack) {
  loom_control_uniformity_cfg_node_t* nodes = summary->nodes;
  for (uint32_t dfs_number = 1; dfs_number <= dfs_count; ++dfs_number) {
    const uint32_t node_index = vertex_by_dfs[dfs_number];
    loom_control_uniformity_cfg_node_t* node = &nodes[node_index];
    node->semidominator = dfs_number;
    node->label = node_index;
    node->ancestor = LOOM_CONTROL_UNIFORMITY_NODE_INVALID;
    node->immediate_postdominator = LOOM_CONTROL_UNIFORMITY_NODE_INVALID;
    node->bucket_head = LOOM_CONTROL_UNIFORMITY_NODE_INVALID;
    node->bucket_next = LOOM_CONTROL_UNIFORMITY_NODE_INVALID;
  }

  for (uint32_t dfs_number = dfs_count; dfs_number > 1; --dfs_number) {
    const uint32_t node_index = vertex_by_dfs[dfs_number];
    loom_control_uniformity_cfg_node_t* node = &nodes[node_index];
    uint32_t predecessor_cursor = 0;
    uint32_t predecessor_index = LOOM_CONTROL_UNIFORMITY_NODE_INVALID;
    while (loom_control_uniformity_cfg_next_reverse_predecessor(
        summary, node_index, &predecessor_cursor, &predecessor_index)) {
      const uint32_t representative =
          loom_control_uniformity_cfg_eval(nodes, predecessor_index, stack);
      node->semidominator =
          iree_min(node->semidominator, nodes[representative].semidominator);
    }

    const uint32_t semidominator_index = vertex_by_dfs[node->semidominator];
    node->bucket_next = nodes[semidominator_index].bucket_head;
    nodes[semidominator_index].bucket_head = node_index;
    node->ancestor = node->dfs_parent;

    loom_control_uniformity_cfg_node_t* parent = &nodes[node->dfs_parent];
    uint32_t bucket_index = parent->bucket_head;
    parent->bucket_head = LOOM_CONTROL_UNIFORMITY_NODE_INVALID;
    while (bucket_index != LOOM_CONTROL_UNIFORMITY_NODE_INVALID) {
      loom_control_uniformity_cfg_node_t* bucket_node = &nodes[bucket_index];
      const uint32_t next_bucket_index = bucket_node->bucket_next;
      const uint32_t representative =
          loom_control_uniformity_cfg_eval(nodes, bucket_index, stack);
      bucket_node->immediate_postdominator =
          nodes[representative].semidominator < bucket_node->semidominator
              ? representative
              : node->dfs_parent;
      bucket_index = next_bucket_index;
    }
  }

  nodes[summary->exit_node].immediate_postdominator = summary->exit_node;
  for (uint32_t dfs_number = 2; dfs_number <= dfs_count; ++dfs_number) {
    const uint32_t node_index = vertex_by_dfs[dfs_number];
    loom_control_uniformity_cfg_node_t* node = &nodes[node_index];
    const uint32_t semidominator_index = vertex_by_dfs[node->semidominator];
    if (node->immediate_postdominator != semidominator_index) {
      node->immediate_postdominator =
          nodes[node->immediate_postdominator].immediate_postdominator;
    }
    node->postdominator_depth =
        nodes[node->immediate_postdominator].postdominator_depth + 1;
  }
}

static uint32_t loom_control_uniformity_cfg_find_path_parent(
    loom_control_uniformity_cfg_node_t* nodes, uint32_t node_index) {
  uint32_t root_index = node_index;
  while (nodes[root_index].scratch != root_index) {
    root_index = nodes[root_index].scratch;
  }
  while (nodes[node_index].scratch != node_index) {
    const uint32_t next_index = nodes[node_index].scratch;
    nodes[node_index].scratch = root_index;
    node_index = next_index;
  }
  return root_index;
}

static bool loom_control_uniformity_cfg_block_has_distinct_successors(
    const loom_cfg_graph_t* graph, uint32_t block_index) {
  const loom_cfg_block_index_span_t successors =
      loom_cfg_graph_successors(graph, (uint16_t)block_index);
  if (successors.count < 2) return false;
  const uint16_t first_successor = successors.values[0];
  for (iree_host_size_t i = 1; i < successors.count; ++i) {
    if (successors.values[i] != first_successor) return true;
  }
  return false;
}

static loom_value_fact_uniform_scope_t
loom_control_uniformity_cfg_selector_scope(
    const loom_control_uniformity_info_t* info,
    const loom_cfg_edge_info_t* edge) {
  if (!edge || edge->selector_value_id == LOOM_VALUE_ID_INVALID) {
    return LOOM_VALUE_FACT_UNIFORM_SCOPE_NONE;
  }
  return loom_value_facts_uniform_scope(
      loom_value_fact_table_lookup(info->fact_table, edge->selector_value_id));
}

static void loom_control_uniformity_cfg_initialize_path_parents(
    loom_control_uniformity_cfg_region_t* summary) {
  for (uint32_t node_index = 0; node_index <= summary->exit_node;
       ++node_index) {
    if (summary->nodes[node_index].dfs_number != 0) {
      summary->nodes[node_index].scratch = node_index;
    }
  }
}

static void loom_control_uniformity_cfg_assign_control_path(
    loom_control_uniformity_cfg_region_t* summary,
    const loom_cfg_edge_info_t* edge,
    loom_value_fact_uniform_scope_t execution_scope) {
  loom_control_uniformity_cfg_node_t* nodes = summary->nodes;
  const uint32_t stop_index =
      nodes[edge->source_block_index].immediate_postdominator;
  uint32_t node_index = loom_control_uniformity_cfg_find_path_parent(
      nodes, edge->target_block_index);
  while (nodes[node_index].postdominator_depth >
         nodes[stop_index].postdominator_depth) {
    loom_control_uniformity_cfg_node_t* node = &nodes[node_index];
    if (node->execution_scope > execution_scope) {
      node->execution_scope = execution_scope;
      node->controller_edge_index =
          (loom_cfg_edge_index_t)(edge - summary->graph->edges);
    }
    const uint32_t parent_index = loom_control_uniformity_cfg_find_path_parent(
        nodes, node->immediate_postdominator);
    node->scratch = parent_index;
    node_index = parent_index;
  }
}

static void loom_control_uniformity_cfg_assign_control_scope(
    const loom_control_uniformity_info_t* info,
    loom_control_uniformity_cfg_region_t* summary,
    loom_value_fact_uniform_scope_t selector_scope) {
  loom_control_uniformity_cfg_initialize_path_parents(summary);
  for (uint32_t block_index = 0; block_index < summary->exit_node;
       ++block_index) {
    if (summary->nodes[block_index].dfs_number == 0 ||
        !loom_control_uniformity_cfg_block_has_distinct_successors(
            summary->graph, block_index)) {
      continue;
    }
    const loom_cfg_edge_index_span_t edges =
        loom_cfg_graph_successor_edges(summary->graph, (uint16_t)block_index);
    if (edges.count == 0) continue;
    const loom_cfg_edge_info_t* first_edge =
        loom_cfg_graph_edge(summary->graph, edges.values[0]);
    if (loom_control_uniformity_cfg_selector_scope(info, first_edge) !=
        selector_scope) {
      continue;
    }
    for (iree_host_size_t i = 0; i < edges.count; ++i) {
      const loom_cfg_edge_info_t* edge =
          loom_cfg_graph_edge(summary->graph, edges.values[i]);
      if (edge && summary->nodes[edge->target_block_index].dfs_number != 0) {
        loom_control_uniformity_cfg_assign_control_path(summary, edge,
                                                        selector_scope);
      }
    }
  }
}

static iree_status_t loom_control_uniformity_cfg_visit_scc_successors(
    void* user_data, iree_host_size_t node,
    loom_scc_successor_callback_t successor) {
  const loom_cfg_graph_t* graph = (const loom_cfg_graph_t*)user_data;
  const loom_cfg_block_index_span_t successors =
      loom_cfg_graph_successors(graph, (uint16_t)node);
  for (iree_host_size_t i = 0; i < successors.count; ++i) {
    IREE_RETURN_IF_ERROR(
        successor.fn(successor.user_data, successors.values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_control_uniformity_cfg_mark_cycles(
    loom_control_uniformity_info_t* info,
    loom_control_uniformity_cfg_region_t* summary) {
  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(info->arena);
  const loom_scc_graph_t scc_graph = {
      .node_count = summary->graph->block_count,
      .visit_successors = loom_scc_visit_successors_callback_make(
          loom_control_uniformity_cfg_visit_scc_successors,
          (void*)summary->graph),
  };
  loom_scc_list_t sccs = {0};
  iree_status_t status =
      loom_scc_compute(&scc_graph, /*options=*/NULL, info->arena, &sccs);
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < sccs.count; ++i) {
      const loom_scc_t* scc = &sccs.values[i];
      if (!scc->is_cycle) continue;
      for (iree_host_size_t j = 0; j < scc->node_count; ++j) {
        summary->nodes[scc->nodes[j]].flags |=
            LOOM_CONTROL_UNIFORMITY_CFG_NODE_IS_CYCLIC;
      }
    }
  }
  iree_arena_checkpoint_restore(&checkpoint);
  return status;
}

static iree_status_t loom_control_uniformity_cfg_append_control_record(
    loom_control_uniformity_info_t* info,
    loom_control_uniformity_cfg_region_t* summary, uint32_t node_index,
    loom_cfg_edge_index_t edge_index) {
  const uint32_t minimum_capacity = summary->control_record_count + 1;
  if (minimum_capacity > summary->control_record_capacity) {
    iree_host_size_t capacity = summary->control_record_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        info->arena, summary->control_record_count,
        iree_max((iree_host_size_t)minimum_capacity, (iree_host_size_t)16),
        sizeof(*summary->control_records), &capacity,
        (void**)&summary->control_records));
    IREE_ASSERT_LE(capacity, UINT32_MAX);
    summary->control_record_capacity = (uint32_t)capacity;
  }
  loom_control_uniformity_cfg_node_t* node = &summary->nodes[node_index];
  const uint32_t record_index = summary->control_record_count++;
  summary->control_records[record_index] =
      (loom_control_uniformity_cfg_record_t){
          .edge_index = edge_index,
          .next_record_index = node->control_record_head,
      };
  node->control_record_head = record_index;
  return iree_ok_status();
}

static iree_status_t loom_control_uniformity_cfg_retain_control_path(
    loom_control_uniformity_info_t* info,
    loom_control_uniformity_cfg_region_t* summary,
    loom_cfg_edge_index_t edge_index) {
  const loom_cfg_edge_info_t* edge =
      loom_cfg_graph_edge(summary->graph, edge_index);
  const uint32_t stop_index =
      summary->nodes[edge->source_block_index].immediate_postdominator;
  uint32_t node_index = edge->target_block_index;
  while (summary->nodes[node_index].postdominator_depth >
         summary->nodes[stop_index].postdominator_depth) {
    IREE_RETURN_IF_ERROR(loom_control_uniformity_cfg_append_control_record(
        info, summary, node_index, edge_index));
    node_index = summary->nodes[node_index].immediate_postdominator;
  }
  return iree_ok_status();
}

static iree_status_t loom_control_uniformity_cfg_initialize_exclusivity(
    loom_control_uniformity_info_t* info,
    loom_control_uniformity_cfg_region_t* summary) {
  if (summary->exclusivity_initialized || !summary->nodes) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_control_uniformity_cfg_mark_cycles(info, summary));
  for (uint32_t block_index = 0; block_index < summary->exit_node;
       ++block_index) {
    if (summary->nodes[block_index].dfs_number == 0 ||
        !loom_control_uniformity_cfg_block_has_distinct_successors(
            summary->graph, block_index)) {
      continue;
    }
    const loom_cfg_edge_index_span_t edges =
        loom_cfg_graph_successor_edges(summary->graph, (uint16_t)block_index);
    for (iree_host_size_t i = 0; i < edges.count; ++i) {
      const loom_cfg_edge_info_t* edge =
          loom_cfg_graph_edge(summary->graph, edges.values[i]);
      if (edge && summary->nodes[edge->target_block_index].dfs_number != 0) {
        IREE_RETURN_IF_ERROR(loom_control_uniformity_cfg_retain_control_path(
            info, summary, edges.values[i]));
      }
    }
  }
  if (summary->exit_node != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        info->arena, summary->exit_node, sizeof(*summary->query_marks),
        (void**)&summary->query_marks));
    memset(summary->query_marks, 0,
           summary->exit_node * sizeof(*summary->query_marks));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        info->arena, summary->exit_node, sizeof(*summary->query_stack),
        (void**)&summary->query_stack));
  }
  if (summary->control_record_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        info->arena, summary->control_record_count,
        sizeof(*summary->query_lhs_edges), (void**)&summary->query_lhs_edges));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        info->arena, summary->control_record_count,
        sizeof(*summary->query_rhs_edges), (void**)&summary->query_rhs_edges));
  }
  summary->exclusivity_initialized = true;
  return iree_ok_status();
}

static uint32_t loom_control_uniformity_cfg_next_query_generation(
    loom_control_uniformity_cfg_region_t* summary) {
  ++summary->query_generation;
  if (summary->query_generation == 0) {
    memset(summary->query_marks, 0,
           summary->exit_node * sizeof(*summary->query_marks));
    summary->query_generation = 1;
  }
  return summary->query_generation;
}

static iree_host_size_t loom_control_uniformity_cfg_collect_control_context(
    loom_control_uniformity_info_t* info,
    loom_control_uniformity_cfg_region_t* summary, uint32_t node_index,
    loom_value_fact_uniform_scope_t required_scope,
    loom_cfg_edge_index_t* out_edges) {
  const uint32_t generation =
      loom_control_uniformity_cfg_next_query_generation(summary);
  iree_host_size_t stack_count = 1;
  summary->query_stack[0] = node_index;
  summary->query_marks[node_index] = generation;
  iree_host_size_t edge_count = 0;
  while (stack_count != 0) {
    const uint32_t current_node_index = summary->query_stack[--stack_count];
    uint32_t record_index =
        summary->nodes[current_node_index].control_record_head;
    while (record_index != LOOM_CONTROL_UNIFORMITY_RECORD_INVALID) {
      const loom_control_uniformity_cfg_record_t* record =
          &summary->control_records[record_index];
      const loom_cfg_edge_info_t* edge =
          loom_cfg_graph_edge(summary->graph, record->edge_index);
      const uint32_t controller_index = edge->source_block_index;
      const loom_control_uniformity_cfg_node_t* controller =
          &summary->nodes[controller_index];
      if (!iree_any_bit_set(controller->flags,
                            LOOM_CONTROL_UNIFORMITY_CFG_NODE_IS_CYCLIC) &&
          loom_control_uniformity_cfg_selector_scope(info, edge) >=
              required_scope) {
        out_edges[edge_count++] = record->edge_index;
      }
      if (summary->query_marks[controller_index] != generation) {
        summary->query_marks[controller_index] = generation;
        summary->query_stack[stack_count++] = controller_index;
      }
      record_index = record->next_record_index;
    }
  }
  IREE_ASSERT_LE(edge_count, summary->control_record_count);
  return edge_count;
}

static iree_status_t loom_control_uniformity_cfg_region_initialize(
    const loom_control_uniformity_info_t* info, const loom_region_t* region,
    loom_control_uniformity_cfg_region_t* summary) {
  memset(summary, 0, sizeof(*summary));
  summary->region = region;
  summary->graph =
      loom_value_fact_table_lookup_cfg_graph(info->fact_table, region);
  if (!summary->graph || summary->graph->malformed) {
    return iree_ok_status();
  }

  summary->exit_node = (uint32_t)summary->graph->block_count;
  const iree_host_size_t node_count = summary->graph->block_count + 1;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(info->arena, node_count,
                                                 sizeof(*summary->nodes),
                                                 (void**)&summary->nodes));
  memset(summary->nodes, 0, node_count * sizeof(*summary->nodes));
  for (iree_host_size_t i = 0; i < node_count; ++i) {
    summary->nodes[i].dfs_parent = LOOM_CONTROL_UNIFORMITY_NODE_INVALID;
    summary->nodes[i].ancestor = LOOM_CONTROL_UNIFORMITY_NODE_INVALID;
    summary->nodes[i].immediate_postdominator =
        LOOM_CONTROL_UNIFORMITY_NODE_INVALID;
    summary->nodes[i].bucket_head = LOOM_CONTROL_UNIFORMITY_NODE_INVALID;
    summary->nodes[i].bucket_next = LOOM_CONTROL_UNIFORMITY_NODE_INVALID;
    summary->nodes[i].controller_edge_index = LOOM_CFG_EDGE_INDEX_INVALID;
    summary->nodes[i].control_record_head =
        LOOM_CONTROL_UNIFORMITY_RECORD_INVALID;
    summary->nodes[i].execution_scope = LOOM_VALUE_FACT_UNIFORM_SCOPE_NONE;
  }

  const iree_arena_checkpoint_t scratch_checkpoint =
      iree_arena_checkpoint_save(info->arena);
  uint32_t* vertex_by_dfs = NULL;
  uint32_t* stack = NULL;
  iree_status_t status =
      iree_arena_allocate_array(info->arena, node_count + 1,
                                sizeof(*vertex_by_dfs), (void**)&vertex_by_dfs);
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(info->arena, node_count, sizeof(*stack),
                                       (void**)&stack);
  }
  if (iree_status_is_ok(status)) {
    loom_control_uniformity_cfg_mark_exit_reachability(summary, stack);
    const uint32_t dfs_count = loom_control_uniformity_cfg_build_reverse_dfs(
        summary, vertex_by_dfs, stack);
    loom_control_uniformity_cfg_compute_postdominators(summary, dfs_count,
                                                       vertex_by_dfs, stack);
  }
  iree_arena_checkpoint_restore(&scratch_checkpoint);
  IREE_RETURN_IF_ERROR(status);

  // Reachable CFG blocks begin cluster-uniform and are weakened by each
  // selector scope below. This ceiling lets the same summary answer subgroup,
  // workgroup, and cluster collective queries without special-case walks.
  for (uint32_t block_index = 0; block_index < summary->exit_node;
       ++block_index) {
    if (loom_cfg_graph_block_is_reachable(summary->graph,
                                          (uint16_t)block_index) &&
        summary->nodes[block_index].dfs_number != 0) {
      summary->nodes[block_index].execution_scope =
          LOOM_VALUE_FACT_UNIFORM_SCOPE_CLUSTER;
    }
  }
  loom_control_uniformity_cfg_assign_control_scope(
      info, summary, LOOM_VALUE_FACT_UNIFORM_SCOPE_NONE);
  loom_control_uniformity_cfg_assign_control_scope(
      info, summary, LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP);
  loom_control_uniformity_cfg_assign_control_scope(
      info, summary, LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP);
  return iree_ok_status();
}

static iree_status_t loom_control_uniformity_cfg_region(
    loom_control_uniformity_info_t* info, const loom_region_t* region,
    loom_control_uniformity_cfg_region_t** out_summary) {
  *out_summary = loom_control_uniformity_lookup_cfg_region(info, region);
  if (*out_summary) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_control_uniformity_reserve_cfg_regions(
      info, info->cfg_regions.count + 1));
  loom_control_uniformity_cfg_region_t* summary = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(info->arena, sizeof(*summary), (void**)&summary));
  IREE_RETURN_IF_ERROR(
      loom_control_uniformity_cfg_region_initialize(info, region, summary));
  loom_control_uniformity_insert_cfg_region(
      info->cfg_regions.slots, info->cfg_regions.capacity, summary);
  ++info->cfg_regions.count;
  *out_summary = summary;
  return iree_ok_status();
}

static bool loom_control_uniformity_prove_value(
    const loom_control_uniformity_info_t* info, const loom_op_t* control_op,
    loom_value_id_t control_value, loom_control_uniformity_source_t source,
    loom_value_fact_uniform_scope_t required_scope,
    loom_control_uniformity_failure_t* out_failure) {
  const loom_value_facts_t facts =
      control_value == LOOM_VALUE_ID_INVALID
          ? loom_value_facts_unknown()
          : loom_value_fact_table_lookup(info->fact_table, control_value);
  if (loom_value_facts_is_uniform_at_scope(facts, required_scope)) {
    return true;
  }
  if (out_failure) {
    *out_failure = (loom_control_uniformity_failure_t){
        .control_op = control_op,
        .control_value = control_value,
        .control_facts = facts,
        .source = source,
    };
  }
  return false;
}

static iree_status_t loom_control_uniformity_prove_cfg_block(
    loom_control_uniformity_info_t* info, const loom_block_t* block,
    loom_value_fact_uniform_scope_t required_scope,
    loom_control_uniformity_failure_t* out_failure, bool* out_proven) {
  *out_proven = false;
  if (!block || !block->parent_region ||
      (!iree_any_bit_set(block->parent_region->flags,
                         LOOM_REGION_INSTANCE_FLAG_CFG) &&
       block->parent_region->block_count <= 1)) {
    *out_proven = true;
    return iree_ok_status();
  }

  loom_control_uniformity_cfg_region_t* summary = NULL;
  IREE_RETURN_IF_ERROR(
      loom_control_uniformity_cfg_region(info, block->parent_region, &summary));
  if (!summary->nodes || block->region_index >= summary->exit_node) {
    if (out_failure) {
      *out_failure = (loom_control_uniformity_failure_t){
          .control_value = LOOM_VALUE_ID_INVALID,
          .control_facts = loom_value_facts_unknown(),
          .source = LOOM_CONTROL_UNIFORMITY_SOURCE_CFG_EXECUTION,
      };
    }
    return iree_ok_status();
  }

  const loom_control_uniformity_cfg_node_t* node =
      &summary->nodes[block->region_index];
  if (node->execution_scope >= required_scope) {
    *out_proven = true;
    return iree_ok_status();
  }

  const loom_cfg_edge_info_t* edge =
      loom_cfg_graph_edge(summary->graph, node->controller_edge_index);
  if (out_failure) {
    const loom_value_id_t selector =
        edge ? edge->selector_value_id : LOOM_VALUE_ID_INVALID;
    *out_failure = (loom_control_uniformity_failure_t){
        .control_op = edge ? edge->terminator : NULL,
        .control_value = selector,
        .control_facts =
            selector == LOOM_VALUE_ID_INVALID
                ? loom_value_facts_unknown()
                : loom_value_fact_table_lookup(info->fact_table, selector),
        .source = edge ? LOOM_CONTROL_UNIFORMITY_SOURCE_CFG_SELECTOR
                       : LOOM_CONTROL_UNIFORMITY_SOURCE_CFG_EXECUTION,
    };
  }
  return iree_ok_status();
}

void loom_control_uniformity_info_initialize(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    iree_arena_allocator_t* arena, loom_control_uniformity_info_t* out_info) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(fact_table);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_info);
  *out_info = (loom_control_uniformity_info_t){
      .module = module,
      .fact_table = fact_table,
      .arena = arena,
  };
}

iree_status_t loom_control_uniformity_prove_execution(
    loom_control_uniformity_info_t* info, const loom_op_t* op,
    loom_value_fact_uniform_scope_t required_scope,
    loom_control_uniformity_failure_t* out_failure, bool* out_proven) {
  IREE_ASSERT_ARGUMENT(info);
  IREE_ASSERT_ARGUMENT(op);
  IREE_ASSERT_ARGUMENT(out_proven);
  IREE_ASSERT(required_scope == LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP ||
              required_scope == LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP ||
              required_scope == LOOM_VALUE_FACT_UNIFORM_SCOPE_CLUSTER);
  *out_proven = false;
  if (out_failure) {
    *out_failure = (loom_control_uniformity_failure_t){
        .control_value = LOOM_VALUE_ID_INVALID,
        .control_facts = loom_value_facts_unknown(),
    };
  }
  for (const loom_op_t* current_op = op; current_op;
       current_op = current_op->parent_op) {
    bool cfg_proven = false;
    IREE_RETURN_IF_ERROR(loom_control_uniformity_prove_cfg_block(
        info, current_op->parent_block, required_scope, out_failure,
        &cfg_proven));
    if (!cfg_proven) return iree_ok_status();

    const loom_op_t* ancestor_op = current_op->parent_op;
    if (!ancestor_op) continue;
    const loom_region_branch_t branch =
        loom_region_branch_cast(info->module, (loom_op_t*)ancestor_op);
    if (loom_region_branch_isa(branch) &&
        !loom_control_uniformity_prove_value(
            info, ancestor_op, loom_region_branch_selector(branch),
            LOOM_CONTROL_UNIFORMITY_SOURCE_REGION_SELECTOR, required_scope,
            out_failure)) {
      return iree_ok_status();
    }

    const loom_loop_like_t loop =
        loom_loop_like_cast(info->module, (loom_op_t*)ancestor_op);
    if (!loom_loop_like_isa(loop)) continue;
    if (!loom_loop_like_has_counted_range(loop)) {
      loom_control_uniformity_prove_value(
          info, ancestor_op, LOOM_VALUE_ID_INVALID,
          LOOM_CONTROL_UNIFORMITY_SOURCE_LOOP_CONDITION, required_scope,
          out_failure);
      return iree_ok_status();
    }
    if (!loom_control_uniformity_prove_value(
            info, ancestor_op, loom_loop_like_lower_bound(loop),
            LOOM_CONTROL_UNIFORMITY_SOURCE_LOOP_LOWER_BOUND, required_scope,
            out_failure) ||
        !loom_control_uniformity_prove_value(
            info, ancestor_op, loom_loop_like_upper_bound(loop),
            LOOM_CONTROL_UNIFORMITY_SOURCE_LOOP_UPPER_BOUND, required_scope,
            out_failure) ||
        !loom_control_uniformity_prove_value(
            info, ancestor_op, loom_loop_like_step(loop),
            LOOM_CONTROL_UNIFORMITY_SOURCE_LOOP_STEP, required_scope,
            out_failure)) {
      return iree_ok_status();
    }
  }
  *out_proven = true;
  return iree_ok_status();
}

iree_status_t loom_control_uniformity_prove_mutually_exclusive_execution(
    loom_control_uniformity_info_t* info, iree_host_size_t lhs_op_count,
    const loom_op_t* const* lhs_ops, iree_host_size_t rhs_op_count,
    const loom_op_t* const* rhs_ops,
    loom_value_fact_uniform_scope_t required_scope, bool* out_proven) {
  IREE_ASSERT_ARGUMENT(info);
  IREE_ASSERT_GT(lhs_op_count, 0u);
  IREE_ASSERT_ARGUMENT(lhs_ops);
  IREE_ASSERT_GT(rhs_op_count, 0u);
  IREE_ASSERT_ARGUMENT(rhs_ops);
  IREE_ASSERT_ARGUMENT(out_proven);
  IREE_ASSERT(required_scope == LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP ||
              required_scope == LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP ||
              required_scope == LOOM_VALUE_FACT_UNIFORM_SCOPE_CLUSTER);
  *out_proven = false;
  const loom_block_t* first_block = lhs_ops[0]->parent_block;
  if (!first_block || !first_block->parent_region) {
    return iree_ok_status();
  }
  const loom_region_t* region = first_block->parent_region;
  for (iree_host_size_t i = 0; i < lhs_op_count; ++i) {
    IREE_ASSERT_ARGUMENT(lhs_ops[i]);
    const loom_block_t* block = lhs_ops[i]->parent_block;
    if (!block || block->parent_region != region) return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < rhs_op_count; ++i) {
    IREE_ASSERT_ARGUMENT(rhs_ops[i]);
    const loom_block_t* block = rhs_ops[i]->parent_block;
    if (!block || block->parent_region != region) return iree_ok_status();
  }

  loom_control_uniformity_cfg_region_t* summary = NULL;
  IREE_RETURN_IF_ERROR(
      loom_control_uniformity_cfg_region(info, region, &summary));
  if (!summary->nodes) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      loom_control_uniformity_cfg_initialize_exclusivity(info, summary));

  if (first_block->region_index >= summary->exit_node) return iree_ok_status();
  iree_host_size_t candidate_edge_count =
      loom_control_uniformity_cfg_collect_control_context(
          info, summary, first_block->region_index, required_scope,
          summary->query_lhs_edges);
  for (iree_host_size_t i = 1; i < lhs_op_count; ++i) {
    const loom_block_t* block = lhs_ops[i]->parent_block;
    if (block->region_index >= summary->exit_node) return iree_ok_status();
    const iree_host_size_t edge_count =
        loom_control_uniformity_cfg_collect_control_context(
            info, summary, block->region_index, required_scope,
            summary->query_rhs_edges);
    iree_host_size_t retained_edge_count = 0;
    for (iree_host_size_t j = 0; j < candidate_edge_count; ++j) {
      const loom_cfg_edge_index_t candidate_edge = summary->query_lhs_edges[j];
      bool retained = false;
      for (iree_host_size_t k = 0; k < edge_count; ++k) {
        if (candidate_edge == summary->query_rhs_edges[k]) {
          retained = true;
          break;
        }
      }
      if (retained) {
        summary->query_lhs_edges[retained_edge_count++] = candidate_edge;
      }
    }
    candidate_edge_count = retained_edge_count;
    if (candidate_edge_count == 0) return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < rhs_op_count; ++i) {
    const loom_block_t* block = rhs_ops[i]->parent_block;
    if (block->region_index >= summary->exit_node) return iree_ok_status();
    const iree_host_size_t edge_count =
        loom_control_uniformity_cfg_collect_control_context(
            info, summary, block->region_index, required_scope,
            summary->query_rhs_edges);
    iree_host_size_t retained_edge_count = 0;
    for (iree_host_size_t j = 0; j < candidate_edge_count; ++j) {
      const loom_cfg_edge_index_t candidate_edge = summary->query_lhs_edges[j];
      const loom_cfg_edge_info_t* lhs_edge =
          loom_cfg_graph_edge(summary->graph, candidate_edge);
      bool retained = false;
      for (iree_host_size_t k = 0; k < edge_count; ++k) {
        const loom_cfg_edge_info_t* rhs_edge =
            loom_cfg_graph_edge(summary->graph, summary->query_rhs_edges[k]);
        if (lhs_edge->terminator == rhs_edge->terminator &&
            lhs_edge->successor_index != rhs_edge->successor_index &&
            lhs_edge->target_block_index != rhs_edge->target_block_index) {
          retained = true;
          break;
        }
      }
      if (retained) {
        summary->query_lhs_edges[retained_edge_count++] = candidate_edge;
      }
    }
    candidate_edge_count = retained_edge_count;
    if (candidate_edge_count == 0) return iree_ok_status();
  }
  *out_proven = true;
  return iree_ok_status();
}

iree_string_view_t loom_control_uniformity_source_name(
    loom_control_uniformity_source_t source) {
  static const iree_string_view_t names[] = {
      [LOOM_CONTROL_UNIFORMITY_SOURCE_REGION_SELECTOR] =
          IREE_SVL("region_selector"),
      [LOOM_CONTROL_UNIFORMITY_SOURCE_LOOP_LOWER_BOUND] =
          IREE_SVL("loop_lower_bound"),
      [LOOM_CONTROL_UNIFORMITY_SOURCE_LOOP_UPPER_BOUND] =
          IREE_SVL("loop_upper_bound"),
      [LOOM_CONTROL_UNIFORMITY_SOURCE_LOOP_STEP] = IREE_SVL("loop_step"),
      [LOOM_CONTROL_UNIFORMITY_SOURCE_LOOP_CONDITION] =
          IREE_SVL("loop_condition"),
      [LOOM_CONTROL_UNIFORMITY_SOURCE_CFG_SELECTOR] = IREE_SVL("cfg_selector"),
      [LOOM_CONTROL_UNIFORMITY_SOURCE_CFG_EXECUTION] =
          IREE_SVL("cfg_execution"),
  };
  IREE_ASSERT_LT((uint32_t)source, IREE_ARRAYSIZE(names));
  return names[source];
}

iree_string_view_t loom_control_uniformity_scope_name(
    loom_value_fact_uniform_scope_t scope) {
  static const iree_string_view_t names[] = {
      [LOOM_VALUE_FACT_UNIFORM_SCOPE_NONE] = IREE_SVL("unknown"),
      [LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP] = IREE_SVL("subgroup"),
      [LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP] = IREE_SVL("workgroup"),
      [LOOM_VALUE_FACT_UNIFORM_SCOPE_CLUSTER] = IREE_SVL("cluster"),
  };
  IREE_ASSERT_LT((uint32_t)scope, IREE_ARRAYSIZE(names));
  return names[scope];
}

iree_string_view_t loom_control_uniformity_fact_distribution_name(
    loom_value_facts_t facts) {
  if (loom_value_facts_is_lane_predicate(facts)) {
    return IREE_SV("lane_predicate");
  }
  if (loom_value_facts_is_lane_varying(facts)) {
    return IREE_SV("lane_varying");
  }
  return loom_control_uniformity_scope_name(
      loom_value_facts_uniform_scope(facts));
}
