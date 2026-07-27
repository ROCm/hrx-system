// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/call_graph.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/util/walk.h"

//===----------------------------------------------------------------------===//
// Edge collection
//===----------------------------------------------------------------------===//

// Temporary edge list during construction. Arena-allocated with
// growth support.
typedef struct loom_edge_list_t {
  loom_symbol_id_t* callees;
  uint16_t count;
  uint16_t capacity;
} loom_edge_list_t;

static iree_status_t loom_call_graph_grow_u16_bounded_array(
    iree_arena_allocator_t* arena, iree_host_size_t existing_count,
    iree_host_size_t minimum_capacity, iree_host_size_t element_size,
    const char* storage_name, uint16_t* inout_capacity, void** inout_ptr) {
  if (minimum_capacity > UINT16_MAX || existing_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "%s exceeds storage limit (%u)", storage_name,
                            (unsigned)UINT16_MAX);
  }
  iree_host_size_t doubled_capacity = (iree_host_size_t)*inout_capacity * 2;
  iree_host_size_t new_capacity =
      doubled_capacity > minimum_capacity ? doubled_capacity : minimum_capacity;
  if (new_capacity > UINT16_MAX) {
    new_capacity = UINT16_MAX;
  }

  void* new_ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, new_capacity, element_size, &new_ptr));
  if (*inout_ptr && existing_count > 0) {
    memcpy(new_ptr, *inout_ptr, existing_count * element_size);
  }
  *inout_ptr = new_ptr;
  *inout_capacity = (uint16_t)new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_edge_list_add(loom_edge_list_t* list,
                                        iree_arena_allocator_t* arena,
                                        loom_symbol_id_t callee) {
  // Dedup: check if already present.
  for (uint16_t i = 0; i < list->count; ++i) {
    if (list->callees[i] == callee) {
      return iree_ok_status();
    }
  }
  if (list->count >= list->capacity) {
    if (list->count == UINT16_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "call graph callee list exceeds storage limit");
    }
    IREE_RETURN_IF_ERROR(loom_call_graph_grow_u16_bounded_array(
        arena, list->count,
        iree_max((iree_host_size_t)list->count + 1, (iree_host_size_t)8),
        sizeof(*list->callees), "call graph callee list", &list->capacity,
        (void**)&list->callees));
  }
  list->callees[list->count++] = callee;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Walk callback for finding call ops
//===----------------------------------------------------------------------===//

typedef struct loom_call_collector_t {
  loom_edge_list_t* edges;
  iree_arena_allocator_t* arena;
  const loom_module_t* module;
} loom_call_collector_t;

static iree_status_t loom_call_collector_visit(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  *out_result = LOOM_WALK_CONTINUE;
  loom_call_collector_t* collector = (loom_call_collector_t*)user_data;
  loom_call_like_t call = loom_call_like_cast(collector->module, op);
  if (!loom_call_like_isa(call)) {
    return iree_ok_status();
  }
  loom_call_like_kind_t kind = loom_call_like_kind(call);
  if (kind != LOOM_CALL_LIKE_KIND_SEMANTIC &&
      kind != LOOM_CALL_LIKE_KIND_LOW_INTERNAL &&
      kind != LOOM_CALL_LIKE_KIND_LOW_INVOKE) {
    return iree_ok_status();
  }
  loom_symbol_ref_t callee_ref = loom_call_like_callee(call);
  if (!loom_symbol_ref_is_valid(callee_ref) || callee_ref.module_id != 0 ||
      callee_ref.symbol_id >= collector->module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "call-like op target symbol ref {module=%u, symbol=%u} is invalid",
        (unsigned)callee_ref.module_id, (unsigned)callee_ref.symbol_id);
  }
  return loom_edge_list_add(collector->edges, collector->arena,
                            callee_ref.symbol_id);
}

//===----------------------------------------------------------------------===//
// Tarjan's SCC (iterative)
//===----------------------------------------------------------------------===//

typedef struct loom_tarjan_state_t {
  uint16_t index;    // Discovery index.
  uint16_t lowlink;  // Lowest reachable index.
  bool on_stack;     // Currently on the Tarjan stack.
} loom_tarjan_state_t;

// Iterative Tarjan frame for the DFS stack.
typedef struct loom_tarjan_frame_t {
  uint16_t node_index;
  uint16_t callee_cursor;  // Next callee to visit.
} loom_tarjan_frame_t;

static iree_status_t loom_call_graph_compute_sccs(
    loom_call_graph_t* graph, iree_arena_allocator_t* arena) {
  uint16_t node_count = graph->node_count;
  if (node_count == 0) {
    graph->scc_count = 0;
    graph->scc_recursive = NULL;
    return iree_ok_status();
  }

  // Per-node Tarjan state.
  loom_tarjan_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, node_count, sizeof(loom_tarjan_state_t), (void**)&state));
  for (uint16_t i = 0; i < node_count; ++i) {
    state[i].index = UINT16_MAX;  // Unvisited.
    state[i].lowlink = UINT16_MAX;
    state[i].on_stack = false;
  }

  // Tarjan stack (node indices).
  uint16_t* tarjan_stack = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, node_count, sizeof(uint16_t), (void**)&tarjan_stack));
  uint16_t tarjan_top = 0;

  // DFS stack.
  loom_tarjan_frame_t* dfs_stack = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, node_count, sizeof(loom_tarjan_frame_t), (void**)&dfs_stack));
  uint16_t dfs_top = 0;

  // SCC assignment. Grow as SCCs are found.
  uint16_t scc_capacity = 16;
  bool* scc_recursive = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, scc_capacity, sizeof(bool), (void**)&scc_recursive));
  uint16_t scc_count = 0;

  uint16_t discovery_counter = 0;

  for (uint16_t root = 0; root < node_count; ++root) {
    if (state[root].index != UINT16_MAX) continue;  // Already visited.

    // Push root.
    dfs_stack[dfs_top++] =
        (loom_tarjan_frame_t){.node_index = root, .callee_cursor = 0};
    state[root].index = discovery_counter;
    state[root].lowlink = discovery_counter;
    ++discovery_counter;
    state[root].on_stack = true;
    tarjan_stack[tarjan_top++] = root;

    while (dfs_top > 0) {
      loom_tarjan_frame_t* frame = &dfs_stack[dfs_top - 1];
      uint16_t v = frame->node_index;
      loom_call_graph_node_t* node = &graph->nodes[v];

      if (frame->callee_cursor < node->callee_count) {
        // Process next callee.
        loom_symbol_id_t callee_sym = node->callees[frame->callee_cursor++];
        uint16_t w = graph->symbol_to_node[callee_sym];
        if (w == UINT16_MAX) continue;  // External/declaration only.

        if (state[w].index == UINT16_MAX) {
          // Not yet visited — push onto DFS stack.
          state[w].index = discovery_counter;
          state[w].lowlink = discovery_counter;
          ++discovery_counter;
          state[w].on_stack = true;
          tarjan_stack[tarjan_top++] = w;
          dfs_stack[dfs_top++] =
              (loom_tarjan_frame_t){.node_index = w, .callee_cursor = 0};
        } else if (state[w].on_stack) {
          // Back edge: update lowlink.
          if (state[w].index < state[v].lowlink) {
            state[v].lowlink = state[w].index;
          }
        }
      } else {
        // All callees processed. Pop this node.
        --dfs_top;

        // Propagate lowlink to parent.
        if (dfs_top > 0) {
          uint16_t parent = dfs_stack[dfs_top - 1].node_index;
          if (state[v].lowlink < state[parent].lowlink) {
            state[parent].lowlink = state[v].lowlink;
          }
        }

        // SCC root?
        if (state[v].lowlink == state[v].index) {
          // Grow SCC arrays if needed.
          if (scc_count >= scc_capacity) {
            if (scc_count == UINT16_MAX) {
              return iree_make_status(
                  IREE_STATUS_RESOURCE_EXHAUSTED,
                  "call graph SCC list exceeds storage limit");
            }
            IREE_RETURN_IF_ERROR(loom_call_graph_grow_u16_bounded_array(
                arena, scc_count, (iree_host_size_t)scc_count + 1,
                sizeof(*scc_recursive), "call graph SCC list", &scc_capacity,
                (void**)&scc_recursive));
          }

          uint16_t scc_id = scc_count++;
          uint16_t scc_member_count = 0;
          bool has_self_edge = false;

          // Pop nodes from Tarjan stack until we hit v.
          uint16_t w;
          do {
            w = tarjan_stack[--tarjan_top];
            state[w].on_stack = false;
            graph->nodes[w].scc_id = scc_id;
            ++scc_member_count;

            // Check for self-edge.
            loom_call_graph_node_t* wnode = &graph->nodes[w];
            for (uint16_t c = 0; c < wnode->callee_count; ++c) {
              if (wnode->callees[c] == wnode->symbol_id) {
                has_self_edge = true;
                break;
              }
            }
          } while (w != v);

          scc_recursive[scc_id] = (scc_member_count > 1) || has_self_edge;
        }
      }
    }
  }

  graph->scc_count = scc_count;
  graph->scc_recursive = scc_recursive;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Build
//===----------------------------------------------------------------------===//

iree_status_t loom_call_graph_build(const loom_module_t* module,
                                    iree_arena_allocator_t* arena,
                                    loom_call_graph_t* out_graph) {
  memset(out_graph, 0, sizeof(*out_graph));
  out_graph->module = module;

  // Count function-like symbols with bodies.
  uint16_t func_count = 0;
  loom_symbol_t* sym = NULL;
  loom_module_for_each_symbol(module, sym) {
    if (loom_symbol_implements(sym, LOOM_SYMBOL_INTERFACE_FUNC_LIKE) &&
        sym->defining_op) {
      loom_func_like_t func = loom_func_like_cast(module, sym->defining_op);
      if (loom_func_like_body(func)) ++func_count;
    }
  }

  if (func_count == 0) {
    out_graph->node_count = 0;
    out_graph->scc_count = 0;
    return iree_ok_status();
  }

  // Allocate nodes and symbol→node map.
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, func_count,
                                                 sizeof(loom_call_graph_node_t),
                                                 (void**)&out_graph->nodes));
  out_graph->node_count = func_count;

  out_graph->symbol_to_node_count = (uint16_t)module->symbols.count;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, module->symbols.count, sizeof(uint16_t),
                                (void**)&out_graph->symbol_to_node));
  memset(out_graph->symbol_to_node, 0xFF,
         module->symbols.count * sizeof(uint16_t));

  // Allocate per-node edge lists.
  loom_edge_list_t* edge_lists = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, func_count, sizeof(loom_edge_list_t), (void**)&edge_lists));
  memset(edge_lists, 0, func_count * sizeof(loom_edge_list_t));

  // Walk each function body to find call ops.
  uint16_t node_index = 0;
  loom_module_for_each_symbol(module, sym) {
    if (!loom_symbol_implements(sym, LOOM_SYMBOL_INTERFACE_FUNC_LIKE) ||
        !sym->defining_op) {
      continue;
    }
    loom_func_like_t func = loom_func_like_cast(module, sym->defining_op);
    loom_region_t* body = loom_func_like_body(func);
    if (!body) continue;

    loom_symbol_id_t sym_id = (loom_symbol_id_t)(sym - module->symbols.entries);
    out_graph->nodes[node_index].symbol_id = sym_id;
    out_graph->nodes[node_index].scc_id = UINT16_MAX;
    out_graph->symbol_to_node[sym_id] = node_index;

    loom_call_collector_t collector = {
        .edges = &edge_lists[node_index],
        .arena = arena,
        .module = module,
    };
    loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
    IREE_RETURN_IF_ERROR(loom_walk_region(
        module, body, LOOM_WALK_PRE_ORDER,
        (loom_walk_callback_t){loom_call_collector_visit, &collector}, arena,
        &walk_result));

    out_graph->nodes[node_index].callee_count = edge_lists[node_index].count;
    out_graph->nodes[node_index].callees = edge_lists[node_index].callees;
    ++node_index;
  }

  // Compute SCCs.
  return loom_call_graph_compute_sccs(out_graph, arena);
}

//===----------------------------------------------------------------------===//
// Queries
//===----------------------------------------------------------------------===//

bool loom_call_graph_is_recursive(const loom_call_graph_t* graph,
                                  loom_symbol_id_t symbol_id) {
  if (symbol_id >= graph->symbol_to_node_count) return false;
  uint16_t node_index = graph->symbol_to_node[symbol_id];
  if (node_index == UINT16_MAX) return false;
  uint16_t scc_id = graph->nodes[node_index].scc_id;
  if (scc_id == UINT16_MAX || scc_id >= graph->scc_count) return false;
  return graph->scc_recursive[scc_id];
}

const loom_call_graph_node_t* loom_call_graph_node(
    const loom_call_graph_t* graph, loom_symbol_id_t symbol_id) {
  if (symbol_id >= graph->symbol_to_node_count) return NULL;
  uint16_t node_index = graph->symbol_to_node[symbol_id];
  if (node_index == UINT16_MAX) return NULL;
  return &graph->nodes[node_index];
}
