// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/graph.h"

// Public dependencies connect whole operations. The original edges still
// constrain device work; this edge also orders its host preparation/completion.
static iree_status_t iree_hal_streaming_graph_extend_copy_dependency(
    iree_hal_streaming_graph_node_t* from, iree_hal_streaming_graph_node_t* to,
    iree_arena_allocator_t* arena, iree_hal_streaming_graph_edge_t** edges) {
  if (iree_any_bit_set(from->flags | to->flags,
                       IREE_HAL_STREAMING_GRAPH_NODE_FLAG_HIDDEN)) {
    return iree_ok_status();
  }
  iree_hal_streaming_graph_node_t* completion =
      from->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY &&
              iree_hal_streaming_graph_node_memcpy(from)->post_callback
          ? iree_hal_streaming_graph_node_memcpy(from)->post_callback
          : from;
  iree_hal_streaming_graph_node_t* preparation =
      to->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY &&
              iree_hal_streaming_graph_node_memcpy(to)->pre_callback
          ? iree_hal_streaming_graph_node_memcpy(to)->pre_callback
          : to;
  if (completion == from && preparation == to) {
    return iree_ok_status();
  }
  iree_hal_streaming_graph_edge_t* edge = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*edge), (void**)&edge));
  *edge = (iree_hal_streaming_graph_edge_t){
      .next = *edges,
      .from = completion,
      .to = preparation,
  };
  *edges = edge;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_graph_schedule_copy_dependencies(
    iree_hal_streaming_graph_t* graph, iree_arena_allocator_t* arena,
    iree_hal_streaming_graph_edge_t** out_edges) {
  *out_edges = NULL;
  if (graph->copy_helper_count == 0) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  for (iree_hal_streaming_node_block_t* block = graph->node_blocks;
       block && iree_status_is_ok(status); block = block->next) {
    for (iree_host_size_t i = 0; i < block->count && iree_status_is_ok(status);
         ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      for (uint32_t j = 0;
           j < node->dependency_count && iree_status_is_ok(status); ++j) {
        status = iree_hal_streaming_graph_extend_copy_dependency(
            iree_hal_streaming_graph_node_dependencies(node)[j], node, arena,
            out_edges);
      }
      for (const iree_hal_streaming_graph_dependency_t* dependency =
               iree_hal_streaming_graph_additional_dependencies(node);
           dependency && iree_status_is_ok(status);
           dependency = dependency->next) {
        status = iree_hal_streaming_graph_extend_copy_dependency(
            dependency->from, node, arena, out_edges);
      }
    }
  }
  return status;
}

iree_status_t iree_hal_streaming_graph_schedule_nodes(
    iree_hal_streaming_node_block_t* node_blocks, iree_host_size_t node_count,
    const uint8_t* disabled_nodes, iree_host_size_t disabled_node_count,
    iree_hal_streaming_graph_edge_t* extra_edges, iree_arena_allocator_t* arena,
    hrx_graph_schedule_t* out_schedule) {
  hrx_graph_analysis_t analysis;
  IREE_RETURN_IF_ERROR(
      hrx_graph_analysis_initialize(node_count, arena, &analysis));
  uint64_t edge_count = 0;
  for (iree_hal_streaming_node_block_t* block = node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      const uint32_t index = node->node_index;
      analysis.sources[index] = node;
      analysis.offsets[index + 1] = node->dependency_count;
      edge_count += node->dependency_count;
      for (const iree_hal_streaming_graph_dependency_t* dependency =
               iree_hal_streaming_graph_additional_dependencies(node);
           dependency; dependency = dependency->next) {
        ++analysis.offsets[index + 1];
        ++edge_count;
      }
      const iree_hal_streaming_graph_node_t* operation =
          node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_HOST_CALL &&
                  iree_hal_streaming_graph_node_host(node)->copy_node
              ? iree_hal_streaming_graph_node_host(node)->copy_node
              : node;
      const bool disabled =
          ((node->flags | operation->flags) &
           IREE_HAL_STREAMING_GRAPH_NODE_FLAG_DISABLED) != 0 ||
          (disabled_nodes && operation->node_index < disabled_node_count &&
           disabled_nodes[operation->node_index]);
      if (disabled || node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EMPTY) {
        analysis.flags[index] = HRX_GRAPH_ANALYSIS_NODE_EMPTY;
      } else if (iree_hal_streaming_graph_node_is_recordable(node->type) &&
                 !(node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL &&
                   iree_hal_streaming_graph_node_kernel(node)->cooperative)) {
        analysis.flags[index] = HRX_GRAPH_ANALYSIS_NODE_RECORDABLE;
      } else {
        analysis.flags[index] = HRX_GRAPH_ANALYSIS_NODE_ISOLATED;
      }
    }
  }
  for (const iree_hal_streaming_graph_edge_t* edge = extra_edges; edge;
       edge = edge->next) {
    ++analysis.offsets[edge->to->node_index + 1];
    ++edge_count;
  }
  if (edge_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph has too many dependencies");
  }
  IREE_RETURN_IF_ERROR(
      hrx_graph_analysis_allocate_dependencies(&analysis, arena));
  bool is_topologically_ordered = true;
  for (uint32_t i = 0; i < analysis.node_count; ++i) {
    const iree_hal_streaming_graph_node_t* node = analysis.sources[i];
    for (uint32_t j = 0; j < node->dependency_count; ++j) {
      analysis.dependencies[analysis.order[i]++] =
          iree_hal_streaming_graph_node_dependencies(node)[j]->node_index;
      is_topologically_ordered &=
          iree_hal_streaming_graph_node_dependencies(node)[j]->node_index < i;
    }
    for (const iree_hal_streaming_graph_dependency_t* dependency =
             iree_hal_streaming_graph_additional_dependencies(node);
         dependency; dependency = dependency->next) {
      analysis.dependencies[analysis.order[i]++] = dependency->from->node_index;
      is_topologically_ordered &= dependency->from->node_index < i;
    }
    if (analysis.flags[i] == HRX_GRAPH_ANALYSIS_NODE_RECORDABLE) {
      analysis.sources[i] = (void*)node->data;
    }
  }
  for (const iree_hal_streaming_graph_edge_t* edge = extra_edges; edge;
       edge = edge->next) {
    analysis.dependencies[analysis.order[edge->to->node_index]++] =
        edge->from->node_index;
    is_topologically_ordered &= edge->from->node_index < edge->to->node_index;
  }
  analysis.is_topologically_ordered = is_topologically_ordered;
  return hrx_graph_analyze(&analysis, arena, out_schedule);
}
