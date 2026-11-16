// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/call_graph.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

typedef struct loom_call_graph_builder_t {
  // Immutable source supplying resolved local call references.
  const loom_module_t* module;
  // Owns the graph and local collection storage.
  iree_arena_allocator_t* arena;
  // Graph whose stable node ordinals form the discovery queue.
  loom_call_graph_t* graph;
  // Allocated entries in graph.nodes.
  iree_host_size_t node_capacity;
  // Optional local consumers of the shared traversal.
  loom_call_graph_visitor_t visitor;
  // Node whose root regions are currently being walked.
  uint16_t caller;
  // Current caller's first-seen callee symbols, including declarations.
  loom_symbol_id_t* callees;
  // Number of distinct symbols in callees.
  uint16_t callee_count;
  // Allocated entries in callees.
  iree_host_size_t callee_capacity;
  // Current caller's membership, rebuilt without a module-sized symbol array.
  loom_ordinal_map_t callee_membership;
  // Whole-module membership words, lazily allocated and reused by clearing
  // only the current caller's reached symbols. NULL for rooted discovery.
  uint64_t* callee_words;
} loom_call_graph_builder_t;

bool loom_call_graph_tracks_call_kind(loom_call_like_kind_t kind) {
  return kind == LOOM_CALL_LIKE_KIND_SEMANTIC ||
         kind == LOOM_CALL_LIKE_KIND_LOW_INTERNAL ||
         kind == LOOM_CALL_LIKE_KIND_LOW_INVOKE ||
         kind == LOOM_CALL_LIKE_KIND_COMMAND_PROGRAM;
}

static iree_status_t loom_call_graph_admit(loom_call_graph_builder_t* builder,
                                           loom_symbol_id_t symbol_id) {
  if (loom_call_graph_node(builder->graph, symbol_id)) {
    return iree_ok_status();
  }
  const loom_symbol_t* symbol = &builder->module->symbols.entries[symbol_id];
  if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE) ||
      !symbol->defining_op) {
    return iree_ok_status();
  }
  const loom_func_like_t function =
      loom_func_like_cast(builder->module, symbol->defining_op);
  if (!loom_func_like_body(function)) {
    return iree_ok_status();
  }
  loom_call_graph_t* graph = builder->graph;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      builder->arena, graph->node_count,
      (iree_host_size_t)graph->node_count + 1, sizeof(*graph->nodes),
      &builder->node_capacity, (void**)&graph->nodes));
  if (graph->symbol_to_node) {
    graph->symbol_to_node[symbol_id] = graph->node_count;
  } else {
    IREE_RETURN_IF_ERROR(loom_ordinal_map_insert(&graph->reached_symbols, 0,
                                                 symbol_id, graph->node_count));
  }
  graph->nodes[graph->node_count++] = (loom_call_graph_node_t){
      .symbol_id = symbol_id,
  };
  return iree_ok_status();
}

static iree_status_t loom_call_graph_collect_edge(
    loom_call_graph_builder_t* builder, loom_symbol_id_t callee) {
  if (builder->graph->symbol_to_node) {
    if (!builder->callee_words) {
      const iree_host_size_t word_count =
          (builder->module->symbols.count + 63) / 64;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          builder->arena, word_count, sizeof(*builder->callee_words),
          (void**)&builder->callee_words));
      memset(builder->callee_words, 0,
             word_count * sizeof(*builder->callee_words));
    }
    uint64_t* word = &builder->callee_words[callee / 64];
    const uint64_t bit = UINT64_C(1) << (callee % 64);
    if (*word & bit) {
      return iree_ok_status();
    }
    *word |= bit;
  } else {
    uint32_t unused = 0;
    if (loom_ordinal_map_lookup(&builder->callee_membership, 0, callee,
                                &unused)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        loom_ordinal_map_insert(&builder->callee_membership, 0, callee, 0));
  }
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      builder->arena, builder->callee_count,
      (iree_host_size_t)builder->callee_count + 1, sizeof(*builder->callees),
      &builder->callee_capacity, (void**)&builder->callees));
  builder->callees[builder->callee_count++] = callee;
  return loom_call_graph_admit(builder, callee);
}

static iree_status_t loom_call_graph_visit(void* user_data, loom_op_t* op,
                                           const loom_walk_context_t* context,
                                           loom_walk_result_t* out_result) {
  *out_result = LOOM_WALK_CONTINUE;
  loom_call_graph_builder_t* builder = user_data;
  loom_call_like_t call = loom_call_like_cast(builder->module, op);
  if (loom_call_like_isa(call) &&
      loom_call_graph_tracks_call_kind(loom_call_like_kind(call))) {
    builder->graph->call_kinds |= (uint8_t)(1u << loom_call_like_kind(call));
    IREE_RETURN_IF_ERROR(loom_call_graph_collect_edge(
        builder, loom_call_like_callee(call).symbol_id));
    if (builder->visitor.call) {
      IREE_RETURN_IF_ERROR(builder->visitor.call(
          builder->visitor.user_data,
          builder->graph->nodes[builder->caller].symbol_id, call));
    }
  } else {
    call = (loom_call_like_t){0};
  }
  return builder->visitor.operation
             ? builder->visitor.operation(builder->visitor.user_data,
                                          builder->caller, op, call, context)
             : iree_ok_status();
}

static iree_status_t loom_call_graph_visit_successors(
    void* user_data, iree_host_size_t node_index,
    loom_scc_successor_callback_t successor) {
  loom_call_graph_t* graph = user_data;
  const loom_call_graph_node_t* node = &graph->nodes[node_index];
  for (uint16_t i = 0; i < node->callee_count; ++i) {
    const loom_call_graph_node_t* callee =
        loom_call_graph_node(graph, node->callees[i]);
    if (callee) {
      const iree_host_size_t callee_index = callee - graph->nodes;
      ++graph->nodes[callee_index].caller_count;
      IREE_RETURN_IF_ERROR(successor.fn(successor.user_data, callee_index));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_call_graph_build(const loom_module_t* module,
                                    const loom_call_graph_options_t* options,
                                    iree_arena_allocator_t* arena,
                                    loom_call_graph_t* out_graph) {
  *out_graph = (loom_call_graph_t){.module = module};
  loom_ordinal_map_initialize(arena, &out_graph->reached_symbols);
  loom_call_graph_builder_t builder = {
      .module = module,
      .arena = arena,
      .graph = out_graph,
      .visitor = options ? options->visitor : (loom_call_graph_visitor_t){0},
  };
  if (options && options->roots) {
    for (iree_host_size_t i = 0; i < options->root_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_call_graph_admit(&builder, options->roots[i]));
    }
  } else {
    loom_symbol_t* symbol = NULL;
    loom_module_for_each_symbol(module, symbol) {
      const loom_func_like_t function =
          loom_func_like_cast(module, symbol->defining_op);
      if (loom_func_like_isa(function) && loom_func_like_body(function)) {
        ++builder.node_capacity;
      }
    }
    if (builder.node_capacity == 0) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, builder.node_capacity,
                                                   sizeof(*out_graph->nodes),
                                                   (void**)&out_graph->nodes));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, module->symbols.count, sizeof(*out_graph->symbol_to_node),
        (void**)&out_graph->symbol_to_node));
    out_graph->symbol_to_node_count = module->symbols.count;
    memset(out_graph->symbol_to_node, 0xFF,
           module->symbols.count * sizeof(*out_graph->symbol_to_node));
    loom_module_for_each_symbol(module, symbol) {
      IREE_RETURN_IF_ERROR(loom_call_graph_admit(
          &builder, (loom_symbol_id_t)(symbol - module->symbols.entries)));
    }
  }

  // Admitting a callee may grow nodes. Keep ordinals across callbacks and
  // continue until every transitively reached definition has been visited.
  for (uint16_t i = 0; i < out_graph->node_count; ++i) {
    builder.caller = i;
    builder.callees = NULL;
    builder.callee_count = 0;
    builder.callee_capacity = 0;
    loom_ordinal_map_initialize(arena, &builder.callee_membership);
    const loom_func_like_t function = loom_func_like_cast(
        module,
        module->symbols.entries[out_graph->nodes[i].symbol_id].defining_op);
    if (builder.visitor.function) {
      IREE_RETURN_IF_ERROR(
          builder.visitor.function(builder.visitor.user_data, i, function));
    }
    loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
    IREE_RETURN_IF_ERROR(loom_walk_function(
        module, function, LOOM_WALK_PRE_ORDER,
        (loom_walk_callback_t){loom_call_graph_visit, &builder}, &walk_result));
    out_graph->nodes[i].callees = builder.callees;
    out_graph->nodes[i].callee_count = builder.callee_count;
    if (builder.callee_words) {
      for (uint16_t j = 0; j < builder.callee_count; ++j) {
        const loom_symbol_id_t callee = builder.callees[j];
        builder.callee_words[callee / 64] &= ~(UINT64_C(1) << (callee % 64));
      }
    }
  }
  if (out_graph->node_count == 0) {
    return iree_ok_status();
  }
  const loom_scc_graph_t graph = {
      .node_count = out_graph->node_count,
      .visit_successors = loom_scc_visit_successors_callback_make(
          loom_call_graph_visit_successors, out_graph),
  };
  IREE_RETURN_IF_ERROR(
      loom_scc_compute(&graph, NULL, arena, &out_graph->components));
  for (iree_host_size_t i = 0; i < out_graph->components.count; ++i) {
    const loom_scc_t* component = &out_graph->components.values[i];
    for (iree_host_size_t j = 0; j < component->node_count; ++j) {
      out_graph->nodes[component->nodes[j]].scc_id = (uint16_t)i;
    }
  }
  return iree_ok_status();
}

bool loom_call_graph_is_recursive(const loom_call_graph_t* graph,
                                  loom_symbol_id_t symbol_id) {
  const loom_call_graph_node_t* node = loom_call_graph_node(graph, symbol_id);
  return node && graph->components.values[node->scc_id].is_cycle;
}

const loom_call_graph_node_t* loom_call_graph_node(
    const loom_call_graph_t* graph, loom_symbol_id_t symbol_id) {
  if (graph->node_count == 0) {
    return NULL;
  }
  if (graph->symbol_to_node) {
    if (symbol_id >= graph->symbol_to_node_count) {
      return NULL;
    }
    const uint16_t node_index = graph->symbol_to_node[symbol_id];
    return node_index != UINT16_MAX ? &graph->nodes[node_index] : NULL;
  }
  uint32_t node_index = 0;
  return loom_ordinal_map_lookup(&graph->reached_symbols, 0, symbol_id,
                                 &node_index)
             ? &graph->nodes[node_index]
             : NULL;
}
