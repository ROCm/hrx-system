// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/cfg_graph.h"

#include <string.h>

#include "loom/ir/context.h"

static iree_status_t loom_cfg_graph_count_edges(const loom_region_t* region,
                                                loom_cfg_graph_t* graph) {
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);
    if (!block) {
      graph->malformed = true;
      continue;
    }
    const loom_op_t* op = block->first_op;
    while (op) {
      if (op->successor_count > 0) {
        if (op != block->last_op) {
          graph->malformed = true;
        }
        loom_block_t* const* successors = loom_op_const_successors(op);
        for (uint8_t i = 0; i < op->successor_count; ++i) {
          uint16_t target_index = 0;
          if (!loom_region_try_block_index(region, successors[i],
                                           &target_index)) {
            graph->malformed = true;
            continue;
          }
          if (graph->edge_count == IREE_HOST_SIZE_MAX) {
            return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                    "CFG edge count overflow");
          }
          ++graph->edge_count;
          ++graph->blocks[block_index].successor_count;
          ++graph->blocks[target_index].predecessor_count;
        }
      }
      op = op->next_op;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_graph_assign_edge_storage(
    iree_arena_allocator_t* arena, loom_cfg_graph_t* graph,
    iree_host_size_t** out_successor_write_positions,
    iree_host_size_t** out_predecessor_write_positions) {
  *out_successor_write_positions = NULL;
  *out_predecessor_write_positions = NULL;
  if (graph->edge_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, graph->edge_count,
                                                   sizeof(*graph->edges),
                                                   (void**)&graph->edges));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, graph->edge_count, sizeof(*graph->successor_indices),
        (void**)&graph->successor_indices));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, graph->edge_count, sizeof(*graph->successor_edge_indices),
        (void**)&graph->successor_edge_indices));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, graph->edge_count, sizeof(*graph->predecessor_indices),
        (void**)&graph->predecessor_indices));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, graph->edge_count, sizeof(*graph->predecessor_edge_indices),
        (void**)&graph->predecessor_edge_indices));
  }

  if (graph->block_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, graph->block_count, sizeof(**out_successor_write_positions),
        (void**)out_successor_write_positions));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, graph->block_count, sizeof(**out_predecessor_write_positions),
        (void**)out_predecessor_write_positions));
  }

  iree_host_size_t successor_start = 0;
  iree_host_size_t predecessor_start = 0;
  for (iree_host_size_t block_index = 0; block_index < graph->block_count;
       ++block_index) {
    loom_cfg_block_info_t* block_info = &graph->blocks[block_index];
    block_info->successor_start = successor_start;
    block_info->successor_edge_start = successor_start;
    block_info->predecessor_start = predecessor_start;
    block_info->predecessor_edge_start = predecessor_start;
    (*out_successor_write_positions)[block_index] = successor_start;
    (*out_predecessor_write_positions)[block_index] = predecessor_start;
    successor_start += block_info->successor_count;
    predecessor_start += block_info->predecessor_count;
  }
  return iree_ok_status();
}

static loom_value_id_t loom_cfg_graph_selector_value(
    const loom_module_t* module, const loom_op_t* terminator,
    bool* inout_malformed) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, terminator);
  if (!vtable ||
      !iree_any_bit_set(vtable->control_flow_flags,
                        LOOM_OP_CONTROL_FLOW_HAS_SUCCESSOR_SELECTOR)) {
    return LOOM_VALUE_ID_INVALID;
  }
  uint16_t selector_index = vtable->successor_selector_operand_index;
  if (selector_index >= terminator->operand_count) {
    *inout_malformed = true;
    return LOOM_VALUE_ID_INVALID;
  }
  return loom_op_const_operands(terminator)[selector_index];
}

static void loom_cfg_graph_write_edges(
    const loom_module_t* module, const loom_region_t* region,
    loom_cfg_graph_t* graph, iree_host_size_t* successor_write_positions,
    iree_host_size_t* predecessor_write_positions) {
  loom_cfg_edge_index_t edge_index = 0;
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);
    if (!block) {
      continue;
    }
    const loom_op_t* op = block->first_op;
    while (op) {
      if (op->successor_count > 0) {
        loom_block_t* const* successors = loom_op_const_successors(op);
        for (uint8_t i = 0; i < op->successor_count; ++i) {
          uint16_t target_index = 0;
          if (!loom_region_try_block_index(region, successors[i],
                                           &target_index)) {
            continue;
          }
          bool edge_malformed = false;
          graph->edges[edge_index] = (loom_cfg_edge_info_t){
              .terminator = op,
              .source_block_index = block_index,
              .target_block_index = target_index,
              .successor_index = i,
              .selector_value_id =
                  loom_cfg_graph_selector_value(module, op, &edge_malformed),
          };
          graph->malformed = graph->malformed || edge_malformed;
          iree_host_size_t successor_position =
              successor_write_positions[block_index]++;
          graph->successor_indices[successor_position] = target_index;
          graph->successor_edge_indices[successor_position] = edge_index;
          iree_host_size_t predecessor_position =
              predecessor_write_positions[target_index]++;
          graph->predecessor_indices[predecessor_position] = block_index;
          graph->predecessor_edge_indices[predecessor_position] = edge_index;
          ++edge_index;
        }
      }
      op = op->next_op;
    }
  }
}

static iree_status_t loom_cfg_graph_mark_reachable(
    iree_arena_allocator_t* arena, loom_cfg_graph_t* graph) {
  if (graph->block_count == 0) return iree_ok_status();
  uint16_t* stack = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, graph->block_count, sizeof(*stack), (void**)&stack));

  iree_host_size_t stack_count = 0;
  graph->blocks[0].reachable = true;
  stack[stack_count++] = 0;
  while (stack_count > 0) {
    uint16_t block_index = stack[--stack_count];
    loom_cfg_block_index_span_t successors =
        loom_cfg_graph_successors(graph, block_index);
    for (iree_host_size_t i = 0; i < successors.count; ++i) {
      uint16_t successor_index = successors.values[i];
      if (graph->blocks[successor_index].reachable) continue;
      graph->blocks[successor_index].reachable = true;
      stack[stack_count++] = successor_index;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_cfg_graph_build(const loom_module_t* module,
                                   const loom_region_t* region,
                                   iree_arena_allocator_t* arena,
                                   loom_cfg_graph_t* out_graph) {
  if (!module || !region || !arena || !out_graph) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "CFG graph build requires a module, region, arena, and output graph");
  }
  memset(out_graph, 0, sizeof(*out_graph));
  out_graph->module = module;
  out_graph->region = region;
  out_graph->block_count = region->block_count;

  if (region->block_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, region->block_count,
                                                   sizeof(*out_graph->blocks),
                                                   (void**)&out_graph->blocks));
    memset(out_graph->blocks, 0,
           (iree_host_size_t)region->block_count * sizeof(*out_graph->blocks));
    for (uint16_t block_index = 0; block_index < region->block_count;
         ++block_index) {
      out_graph->blocks[block_index].block =
          loom_region_const_block(region, block_index);
      const loom_block_t* block = out_graph->blocks[block_index].block;
      if (!loom_region_try_block_index(region, block, NULL) ||
          block->region_index != block_index) {
        out_graph->malformed = true;
      }
    }
  }

  IREE_RETURN_IF_ERROR(loom_cfg_graph_count_edges(region, out_graph));

  iree_host_size_t* successor_write_positions = NULL;
  iree_host_size_t* predecessor_write_positions = NULL;
  IREE_RETURN_IF_ERROR(loom_cfg_graph_assign_edge_storage(
      arena, out_graph, &successor_write_positions,
      &predecessor_write_positions));
  loom_cfg_graph_write_edges(module, region, out_graph,
                             successor_write_positions,
                             predecessor_write_positions);
  return loom_cfg_graph_mark_reachable(arena, out_graph);
}

iree_host_size_t loom_cfg_graph_block_index(const loom_cfg_graph_t* graph,
                                            const loom_block_t* block) {
  if (!graph || !block) {
    return IREE_HOST_SIZE_MAX;
  }
  uint16_t block_index = 0;
  if (!loom_region_try_block_index(graph->region, block, &block_index) ||
      block_index >= graph->block_count ||
      graph->blocks[block_index].block != block) {
    return IREE_HOST_SIZE_MAX;
  }
  return block_index;
}

loom_cfg_block_index_span_t loom_cfg_graph_successors(
    const loom_cfg_graph_t* graph, uint16_t block_index) {
  if (!graph || block_index >= graph->block_count) {
    return (loom_cfg_block_index_span_t){0};
  }
  const loom_cfg_block_info_t* block_info = &graph->blocks[block_index];
  if (block_info->successor_count == 0) {
    return (loom_cfg_block_index_span_t){0};
  }
  return (loom_cfg_block_index_span_t){
      .values = graph->successor_indices + block_info->successor_start,
      .count = block_info->successor_count,
  };
}

loom_cfg_edge_index_span_t loom_cfg_graph_successor_edges(
    const loom_cfg_graph_t* graph, uint16_t block_index) {
  if (!graph || block_index >= graph->block_count) {
    return (loom_cfg_edge_index_span_t){0};
  }
  const loom_cfg_block_info_t* block_info = &graph->blocks[block_index];
  if (block_info->successor_count == 0) {
    return (loom_cfg_edge_index_span_t){0};
  }
  return (loom_cfg_edge_index_span_t){
      .values =
          graph->successor_edge_indices + block_info->successor_edge_start,
      .count = block_info->successor_count,
  };
}

loom_cfg_block_index_span_t loom_cfg_graph_predecessors(
    const loom_cfg_graph_t* graph, uint16_t block_index) {
  if (!graph || block_index >= graph->block_count) {
    return (loom_cfg_block_index_span_t){0};
  }
  const loom_cfg_block_info_t* block_info = &graph->blocks[block_index];
  if (block_info->predecessor_count == 0) {
    return (loom_cfg_block_index_span_t){0};
  }
  return (loom_cfg_block_index_span_t){
      .values = graph->predecessor_indices + block_info->predecessor_start,
      .count = block_info->predecessor_count,
  };
}

loom_cfg_edge_index_span_t loom_cfg_graph_predecessor_edges(
    const loom_cfg_graph_t* graph, uint16_t block_index) {
  if (!graph || block_index >= graph->block_count) {
    return (loom_cfg_edge_index_span_t){0};
  }
  const loom_cfg_block_info_t* block_info = &graph->blocks[block_index];
  if (block_info->predecessor_count == 0) {
    return (loom_cfg_edge_index_span_t){0};
  }
  return (loom_cfg_edge_index_span_t){
      .values =
          graph->predecessor_edge_indices + block_info->predecessor_edge_start,
      .count = block_info->predecessor_count,
  };
}

const loom_cfg_edge_info_t* loom_cfg_graph_edge(
    const loom_cfg_graph_t* graph, loom_cfg_edge_index_t edge_index) {
  return graph && edge_index < graph->edge_count ? &graph->edges[edge_index]
                                                 : NULL;
}

bool loom_cfg_graph_block_is_reachable(const loom_cfg_graph_t* graph,
                                       uint16_t block_index) {
  return graph && block_index < graph->block_count &&
         graph->blocks[block_index].reachable;
}
