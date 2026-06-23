// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/graph.h"

#include "common/internal.h"

//===----------------------------------------------------------------------===//
// iree_hal_streaming_graph_t (template)
//===----------------------------------------------------------------------===//

static void iree_hal_streaming_graph_destroy(iree_hal_streaming_graph_t* graph);

static void iree_hal_streaming_graph_node_deinitialize_attrs(
    iree_hal_streaming_graph_node_t* node) {
  switch (node->type) {
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH:
      iree_hal_streaming_graph_release(node->attrs.child_graph.graph);
      node->attrs.child_graph.graph = NULL;
      break;
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD:
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT:
      iree_hal_streaming_event_release(node->attrs.event.event);
      node->attrs.event.event = NULL;
      break;
    default:
      break;
  }
}

iree_status_t iree_hal_streaming_graph_create(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_graph_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_streaming_graph_t** out_graph) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_graph);
  *out_graph = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_graph_t* graph = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*graph), (void**)&graph));

  iree_atomic_ref_count_init(&graph->ref_count);

  // Initialize the arena using the device's block pool.
  iree_hal_streaming_device_t* device = context->device_entry;
  iree_arena_initialize(&device->block_pool, &graph->arena);
  graph->arena_allocator = iree_arena_allocator(&graph->arena);

  graph->node_blocks = NULL;
  graph->current_node_block = NULL;
  graph->node_count = 0;
  graph->child_graph_node_count = 0;
  graph->next_clone_source_node_index = 0;
  graph->root_blocks = NULL;
  graph->current_root_block = NULL;
  graph->root_count = 0;
  graph->additional_edges = NULL;
  graph->additional_edge_count = 0;
  graph->owned_host_allocations = NULL;
  graph->flags = flags;
  graph->context = context;
  iree_hal_streaming_context_retain(context);
  graph->host_allocator = host_allocator;

  *out_graph = graph;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_streaming_graph_destroy(
    iree_hal_streaming_graph_t* graph) {
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_deinitialize_attrs(block->nodes[i]);
    }
  }

  iree_hal_streaming_graph_owned_host_allocation_t* owned_allocation =
      graph->owned_host_allocations;
  while (owned_allocation) {
    if (owned_allocation->host_ptr) {
      iree_status_ignore(iree_hal_streaming_memory_free_host(
          graph->context, owned_allocation->host_ptr));
    }
    owned_allocation = owned_allocation->next;
  }

  // Reset the arena - this frees all nodes and arrays at once.
  // The arena returns all blocks to the device's block pool for reuse.
  iree_arena_deinitialize(&graph->arena);

  // Release context.
  iree_hal_streaming_context_release(graph->context);

  // Free graph memory itself (not allocated from arena).
  const iree_allocator_t host_allocator = graph->host_allocator;
  iree_allocator_free(host_allocator, graph);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_streaming_graph_allocate_host_staging(
    iree_hal_streaming_graph_t* graph, iree_device_size_t size,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_allocate_host(
              graph->context, size,
              IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT, &buffer));

  iree_hal_streaming_graph_owned_host_allocation_t* owned_allocation = NULL;
  iree_status_t status = iree_arena_allocate(
      &graph->arena, sizeof(*owned_allocation), (void**)&owned_allocation);
  if (iree_status_is_ok(status)) {
    owned_allocation->host_ptr = buffer->host_ptr;
    owned_allocation->next = graph->owned_host_allocations;
    graph->owned_host_allocations = owned_allocation;
    *out_buffer = buffer;
  } else {
    iree_status_ignore(
        iree_hal_streaming_memory_free_host(graph->context, buffer->host_ptr));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_streaming_graph_retain(iree_hal_streaming_graph_t* graph) {
  if (graph) {
    iree_atomic_ref_count_inc(&graph->ref_count);
  }
}

void iree_hal_streaming_graph_release(iree_hal_streaming_graph_t* graph) {
  if (graph && iree_atomic_ref_count_dec(&graph->ref_count) == 1) {
    iree_hal_streaming_graph_destroy(graph);
  }
}

iree_host_size_t iree_hal_streaming_graph_size(
    iree_hal_streaming_graph_t* graph) {
  IREE_ASSERT_ARGUMENT(graph);
  return graph->node_count;
}

void iree_hal_streaming_graph_get_nodes(
    iree_hal_streaming_graph_t* graph, iree_host_size_t count,
    iree_hal_streaming_graph_node_t** nodes) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(nodes || count == 0);

  // Iterate through the node blocks to collect nodes.
  iree_host_size_t copied_count = 0;
  iree_hal_streaming_node_block_t* block = graph->node_blocks;
  while (block && copied_count < count) {
    iree_host_size_t nodes_to_copy = block->count;
    if (copied_count + nodes_to_copy > count) {
      nodes_to_copy = count - copied_count;
    }

    // Copy nodes from this block.
    for (iree_host_size_t i = 0; i < nodes_to_copy; i++) {
      nodes[copied_count++] = block->nodes[i];
    }

    block = block->next;
  }
}

static void iree_hal_streaming_graph_renumber_nodes(
    iree_hal_streaming_graph_t* graph) {
  uint32_t node_index = 0;
  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      block->nodes[i]->node_index = node_index++;
    }
  }
}

static iree_hal_streaming_graph_node_t* iree_hal_streaming_graph_node_at_index(
    const iree_hal_streaming_graph_t* graph, uint32_t node_index) {
  iree_host_size_t skipped_count = 0;
  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    if (node_index < skipped_count + block->count) {
      return block->nodes[node_index - skipped_count];
    }
    skipped_count += block->count;
  }
  return NULL;
}

static bool iree_hal_streaming_graph_node_is_active_in_graph(
    const iree_hal_streaming_graph_t* graph,
    const iree_hal_streaming_graph_node_t* node) {
  if (!node || node->graph != graph) return false;
  if (node->node_index >= graph->node_count) return false;
  return iree_hal_streaming_graph_node_at_index(graph, node->node_index) ==
         node;
}

static bool iree_hal_streaming_graph_dependency_exists(
    const iree_hal_streaming_graph_t* graph,
    const iree_hal_streaming_graph_node_t* from_node,
    const iree_hal_streaming_graph_node_t* to_node) {
  for (uint32_t i = 0; i < to_node->dependency_count; ++i) {
    if (to_node->dependencies[i] == from_node) return true;
  }
  for (iree_hal_streaming_graph_edge_t* edge = graph->additional_edges; edge;
       edge = edge->next) {
    if (edge->from == from_node && edge->to == to_node) return true;
  }
  return false;
}

static iree_status_t iree_hal_streaming_graph_validate_dependencies(
    const iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count) {
  if (dependency_count == 0) return iree_ok_status();
  if (!dependencies) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dependency array must be provided");
  }
  for (iree_host_size_t i = 0; i < dependency_count; ++i) {
    if (!iree_hal_streaming_graph_node_is_active_in_graph(graph,
                                                          dependencies[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dependency at index %" PRIhsz
                              " does not belong to the target graph",
                              i);
    }
  }
  return iree_ok_status();
}

static bool iree_hal_streaming_graph_list_contains(
    iree_hal_streaming_graph_t** graphs, iree_host_size_t graph_count,
    iree_hal_streaming_graph_t* graph) {
  for (iree_host_size_t i = 0; i < graph_count; ++i) {
    if (graphs[i] == graph) return true;
  }
  return false;
}

static iree_status_t iree_hal_streaming_graph_list_append(
    iree_allocator_t host_allocator, iree_hal_streaming_graph_t*** graphs,
    iree_host_size_t* graph_count, iree_host_size_t* graph_capacity,
    iree_hal_streaming_graph_t* graph) {
  if (*graph_count >= *graph_capacity) {
    iree_host_size_t new_capacity = 8;
    if (*graph_capacity && IREE_UNLIKELY(!iree_host_size_checked_mul(
                               *graph_capacity, 2, &new_capacity))) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "child graph search capacity overflow");
    }
    iree_host_size_t allocation_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            new_capacity, sizeof(**graphs), &allocation_size))) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "child graph search size overflow");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(host_allocator, allocation_size,
                                                (void**)graphs));
    *graph_capacity = new_capacity;
  }
  (*graphs)[(*graph_count)++] = graph;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_validate_child_graph(
    iree_hal_streaming_graph_t* parent_graph,
    iree_hal_streaming_graph_t* child_graph) {
  if (parent_graph == child_graph) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "child graph cannot be the parent graph");
  }
  if (parent_graph->context != child_graph->context) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "child graph must belong to the parent context");
  }
  if (child_graph->child_graph_node_count == 0) {
    return iree_ok_status();
  }

  iree_hal_streaming_graph_t** graphs = NULL;
  iree_host_size_t graph_count = 0;
  iree_host_size_t graph_capacity = 0;
  iree_allocator_t host_allocator = parent_graph->host_allocator;
  iree_status_t status = iree_hal_streaming_graph_list_append(
      host_allocator, &graphs, &graph_count, &graph_capacity, child_graph);

  for (iree_host_size_t search_index = 0;
       iree_status_is_ok(status) && search_index < graph_count;
       ++search_index) {
    iree_hal_streaming_graph_t* graph = graphs[search_index];
    if (graph == parent_graph) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "child graph would create recursive graph containment");
      break;
    }
    if (graph->child_graph_node_count == 0) continue;

    for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
         block = block->next) {
      for (iree_host_size_t i = 0; i < block->count; ++i) {
        iree_hal_streaming_graph_node_t* node = block->nodes[i];
        if (node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH ||
            !node->attrs.child_graph.graph ||
            iree_hal_streaming_graph_list_contains(
                graphs, graph_count, node->attrs.child_graph.graph)) {
          continue;
        }
        status = iree_hal_streaming_graph_list_append(
            host_allocator, &graphs, &graph_count, &graph_capacity,
            node->attrs.child_graph.graph);
        if (!iree_status_is_ok(status)) break;
      }
      if (!iree_status_is_ok(status)) break;
    }
  }

  iree_allocator_free(host_allocator, graphs);
  return status;
}

static bool iree_hal_streaming_graph_remove_from_blocks(
    iree_hal_streaming_node_block_t* blocks,
    iree_hal_streaming_graph_node_t* node) {
  for (iree_hal_streaming_node_block_t* block = blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      if (block->nodes[i] != node) continue;
      for (iree_host_size_t j = i + 1; j < block->count; ++j) {
        block->nodes[j - 1] = block->nodes[j];
      }
      --block->count;
      return true;
    }
  }
  return false;
}

static void iree_hal_streaming_graph_remove_dependency_refs(
    iree_hal_streaming_graph_t* graph, iree_hal_streaming_graph_node_t* node) {
  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* existing_node = block->nodes[i];
      uint32_t dependency_index = 0;
      while (dependency_index < existing_node->dependency_count) {
        if (existing_node->dependencies[dependency_index] != node) {
          ++dependency_index;
          continue;
        }
        for (uint32_t j = dependency_index + 1;
             j < existing_node->dependency_count; ++j) {
          existing_node->dependencies[j - 1] = existing_node->dependencies[j];
        }
        --existing_node->dependency_count;
      }
    }
  }
}

static void iree_hal_streaming_graph_remove_additional_edges(
    iree_hal_streaming_graph_t* graph, iree_hal_streaming_graph_node_t* node) {
  iree_hal_streaming_graph_edge_t** next_edge = &graph->additional_edges;
  while (*next_edge) {
    iree_hal_streaming_graph_edge_t* edge = *next_edge;
    if (edge->from == node || edge->to == node) {
      *next_edge = edge->next;
      --graph->additional_edge_count;
      continue;
    }
    next_edge = &edge->next;
  }
}

// Helper to allocate a graph node with trailing dependencies and extra data.
static iree_status_t iree_hal_streaming_graph_allocate_node(
    iree_allocator_t allocator, iree_host_size_t dependency_count,
    iree_host_size_t extra_data_size,
    iree_hal_streaming_graph_node_t** out_node, uint8_t** out_extra_data) {
  IREE_ASSERT_ARGUMENT(out_node);
  *out_node = NULL;
  if (out_extra_data) *out_extra_data = NULL;

  // Calculate total size needed.
  const iree_host_size_t node_size = sizeof(iree_hal_streaming_graph_node_t);
  iree_host_size_t deps_size = 0;
  iree_host_size_t total_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(dependency_count,
                                      sizeof(iree_hal_streaming_graph_node_t*),
                                      &deps_size) ||
          !iree_host_size_checked_add(node_size, deps_size, &total_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph node allocation size overflow");
  }

  // Align for extra data if needed.
  if (extra_data_size > 0) {
    if (IREE_UNLIKELY(!iree_host_size_checked_align(
            total_size, iree_max_align_t, &total_size))) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "graph node allocation size overflow");
    }
    const iree_host_size_t extra_data_offset = total_size;
    if (IREE_UNLIKELY(!iree_host_size_checked_add(total_size, extra_data_size,
                                                  &total_size))) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "graph node allocation size overflow");
    }

    // Allocate the entire block.
    iree_hal_streaming_graph_node_t* node = NULL;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator, total_size, (void**)&node));
    memset(node, 0, total_size);

    *out_node = node;
    if (out_extra_data) {
      *out_extra_data = (uint8_t*)node + extra_data_offset;
    }
  } else {
    // Allocate just the node and dependencies.
    iree_hal_streaming_graph_node_t* node = NULL;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator, total_size, (void**)&node));
    memset(node, 0, total_size);
    *out_node = node;
  }

  return iree_ok_status();
}

// Helper to allocate a new block for node storage.
static iree_status_t iree_hal_streaming_allocate_node_block(
    iree_allocator_t allocator, iree_host_size_t capacity,
    iree_hal_streaming_node_block_t** out_block) {
  iree_host_size_t block_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul_add(
          capacity, sizeof(iree_hal_streaming_graph_node_t*),
          sizeof(iree_hal_streaming_node_block_t), &block_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph node block allocation size overflow");
  }

  iree_hal_streaming_node_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, block_size, (void**)&block));

  block->next = NULL;
  block->capacity = capacity;
  block->count = 0;
  *out_block = block;
  return iree_ok_status();
}

// Helper to add a node to the graph.
static iree_status_t iree_hal_streaming_graph_add_node(
    iree_hal_streaming_graph_t* graph, iree_hal_streaming_graph_node_t* node) {
  if (IREE_UNLIKELY(graph->node_count >= UINT32_MAX ||
                    graph->next_clone_source_node_index == UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph node count exceeds supported range");
  }

  // Assign unique index to the node that can be used to get the logical index
  // in the graph for use as dependency references.
  node->graph = graph;
  node->node_index = (uint32_t)graph->node_count;
  node->clone_source_node_index = graph->next_clone_source_node_index++;

  // Add to node blocks.
  if (!graph->current_node_block ||
      graph->current_node_block->count >= graph->current_node_block->capacity) {
    // Need a new block.
    const iree_host_size_t block_capacity =
        graph->node_count < 64 ? 16 : 64;  // Grow block size for larger graphs.
    iree_hal_streaming_node_block_t* new_block = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_allocate_node_block(
        graph->arena_allocator, block_capacity, &new_block));

    // Chain the new block.
    if (graph->current_node_block) {
      graph->current_node_block->next = new_block;
    } else {
      graph->node_blocks = new_block;
    }
    graph->current_node_block = new_block;
  }

  graph->current_node_block->nodes[graph->current_node_block->count++] = node;
  ++graph->node_count;

  // Add to root nodes if no dependencies.
  if (node->dependency_count == 0) {
    if (!graph->current_root_block || graph->current_root_block->count >=
                                          graph->current_root_block->capacity) {
      // Need a new root block.
      const iree_host_size_t block_capacity = 8;
      iree_hal_streaming_node_block_t* new_block = NULL;
      IREE_RETURN_IF_ERROR(iree_hal_streaming_allocate_node_block(
          graph->arena_allocator, block_capacity, &new_block));

      // Chain the new block.
      if (graph->current_root_block) {
        graph->current_root_block->next = new_block;
      } else {
        graph->root_blocks = new_block;
      }
      graph->current_root_block = new_block;
    }

    graph->current_root_block->nodes[graph->current_root_block->count++] = node;
    ++graph->root_count;
  }

  return iree_ok_status();
}

iree_status_t iree_hal_streaming_graph_clone(
    iree_hal_streaming_graph_t* source_graph,
    iree_hal_streaming_graph_t** out_graph) {
  IREE_ASSERT_ARGUMENT(source_graph);
  IREE_ASSERT_ARGUMENT(out_graph);
  *out_graph = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_graph_t* clone_graph = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_create(
              source_graph->context, source_graph->flags,
              source_graph->host_allocator, &clone_graph));

  iree_hal_streaming_graph_node_t** node_map = NULL;
  iree_status_t status = iree_ok_status();
  if (source_graph->node_count > 0) {
    iree_host_size_t node_map_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            source_graph->node_count, sizeof(*node_map), &node_map_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "graph clone node map size overflow");
    } else {
      status = iree_allocator_malloc(source_graph->host_allocator,
                                     node_map_size, (void**)&node_map);
    }
    if (!iree_status_is_ok(status)) {
      iree_hal_streaming_graph_release(clone_graph);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    memset(node_map, 0, node_map_size);
  }

  for (iree_hal_streaming_node_block_t* block = source_graph->node_blocks;
       iree_status_is_ok(status) && block; block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* source_node = block->nodes[i];
      iree_host_size_t constants_size = 0;
      iree_host_size_t bindings_size = 0;
      iree_host_size_t extra_data_size = 0;
      if (source_node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL) {
        iree_host_size_t bindings_byte_size = 0;
        if (IREE_UNLIKELY(
                !iree_host_size_checked_align(
                    source_node->attrs.kernel.constants.data_length,
                    iree_max_align_t, &constants_size) ||
                !iree_host_size_checked_mul(
                    source_node->attrs.kernel.bindings.count,
                    sizeof(*source_node->attrs.kernel.bindings.values),
                    &bindings_byte_size) ||
                !iree_host_size_checked_align(
                    bindings_byte_size, iree_max_align_t, &bindings_size) ||
                !iree_host_size_checked_add(constants_size, bindings_size,
                                            &extra_data_size))) {
          status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                    "graph clone node data size overflow");
          break;
        }
      }

      iree_hal_streaming_graph_node_t* clone_node = NULL;
      uint8_t* extra_data = NULL;
      status = iree_hal_streaming_graph_allocate_node(
          clone_graph->arena_allocator, source_node->dependency_count,
          extra_data_size, &clone_node, &extra_data);
      if (!iree_status_is_ok(status)) break;

      clone_node->type = source_node->type;
      clone_node->flags = source_node->flags;
      clone_node->dependency_count = source_node->dependency_count;
      clone_node->attrs = source_node->attrs;

      if (source_node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL) {
        void* constants = extra_data_size ? extra_data : NULL;
        if (source_node->attrs.kernel.constants.data_length > 0) {
          memcpy(constants, source_node->attrs.kernel.constants.data,
                 source_node->attrs.kernel.constants.data_length);
        }
        clone_node->attrs.kernel.constants = iree_make_const_byte_span(
            constants, source_node->attrs.kernel.constants.data_length);
        iree_hal_buffer_ref_t* bindings =
            extra_data_size
                ? (iree_hal_buffer_ref_t*)(extra_data + constants_size)
                : NULL;
        clone_node->attrs.kernel.bindings.values = bindings;
        if (source_node->attrs.kernel.bindings.count > 0) {
          memcpy(bindings, source_node->attrs.kernel.bindings.values,
                 source_node->attrs.kernel.bindings.count *
                     sizeof(*source_node->attrs.kernel.bindings.values));
        }
      } else if (source_node->type ==
                     IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH &&
                 source_node->attrs.child_graph.graph) {
        iree_hal_streaming_graph_retain(source_node->attrs.child_graph.graph);
      } else if ((source_node->type ==
                      IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD ||
                  source_node->type ==
                      IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT) &&
                 source_node->attrs.event.event) {
        iree_hal_streaming_event_retain(source_node->attrs.event.event);
      }

      status = iree_hal_streaming_graph_add_node(clone_graph, clone_node);
      if (!iree_status_is_ok(status)) {
        iree_hal_streaming_graph_node_deinitialize_attrs(clone_node);
        break;
      }
      if (clone_node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH) {
        ++clone_graph->child_graph_node_count;
      }
      clone_node->clone_source_node_index =
          source_node->clone_source_node_index;
      node_map[source_node->node_index] = clone_node;
    }
  }

  for (iree_hal_streaming_node_block_t* block = source_graph->node_blocks;
       iree_status_is_ok(status) && block; block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* source_node = block->nodes[i];
      iree_hal_streaming_graph_node_t* clone_node =
          node_map[source_node->node_index];
      for (uint32_t j = 0; j < source_node->dependency_count; ++j) {
        clone_node->dependencies[j] =
            node_map[source_node->dependencies[j]->node_index];
      }
    }
  }

  for (iree_hal_streaming_graph_edge_t* edge = source_graph->additional_edges;
       iree_status_is_ok(status) && edge; edge = edge->next) {
    iree_hal_streaming_graph_node_t* from_node =
        node_map[edge->from->node_index];
    iree_hal_streaming_graph_node_t* to_node = node_map[edge->to->node_index];
    status = iree_hal_streaming_graph_add_dependencies(clone_graph, &from_node,
                                                       &to_node, 1);
  }

  if (node_map) {
    iree_allocator_free(source_graph->host_allocator, node_map);
  }
  if (iree_status_is_ok(status)) {
    *out_graph = clone_graph;
    clone_graph->next_clone_source_node_index =
        source_graph->next_clone_source_node_index;
  } else {
    iree_hal_streaming_graph_release(clone_graph);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_graph_add_empty_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count,
    iree_hal_streaming_graph_node_t** out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_validate_dependencies(graph, dependencies,
                                                         dependency_count));

  // Allocate node with dependencies in a single allocation.
  iree_hal_streaming_graph_node_t* node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_allocate_node(
              graph->arena_allocator, dependency_count, 0, &node, NULL));

  node->type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EMPTY;
  node->dependency_count = dependency_count;

  // Copy dependencies to the trailing array.
  if (dependency_count > 0) {
    memcpy(node->dependencies, dependencies,
           dependency_count * sizeof(*dependencies));
  }

  // No attributes today.
  // iree_hal_streaming_graph_empty_attrs_t* attrs = &node->attrs.empty;

  iree_status_t status = iree_hal_streaming_graph_add_node(graph, node);
  if (iree_status_is_ok(status) && out_node) {
    *out_node = node;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_streaming_pack_raw_argument_list(
    const iree_hal_streaming_parameter_info_t* parameters,
    void** parameter_list, void* out_constants) {
  IREE_ASSERT_ARGUMENT(parameters);
  if (parameters->buffer_size == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_ARGUMENT(parameter_list);
  IREE_ASSERT_ARGUMENT(out_constants);

  uint8_t* constants = (uint8_t*)out_constants;
  const iree_hal_streaming_parameter_op_t* op = &parameters->ops[0];
  for (uint32_t i = 0; i < parameters->copy_count; ++i, ++op) {
    const iree_hal_streaming_parameter_copy_op_t copy_op = op->copy;
    void* param_ptr = parameter_list[copy_op.src_ordinal];
    memcpy(constants + copy_op.direct_dst_offset, param_ptr, copy_op.size);
  }

  for (uint32_t i = 0; i < parameters->binding_count; ++i, ++op) {
    const iree_hal_streaming_parameter_resolve_op_t resolve_op = op->resolve;
    void* param_ptr = parameter_list[resolve_op.src_ordinal];
    void* device_ptr = *(void**)param_ptr;
    memcpy(constants + resolve_op.dst_offset, &device_ptr, sizeof(void*));
  }

  return iree_ok_status();
}

iree_status_t iree_hal_streaming_graph_add_kernel_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, iree_hal_streaming_symbol_t* symbol,
    const iree_hal_streaming_dispatch_params_t* params,
    iree_hal_streaming_graph_node_t** out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(symbol);
  IREE_ASSERT_ARGUMENT(params);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_validate_dependencies(graph, dependencies,
                                                         dependency_count));

  // Verify the symbol is a function.
  if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol is not a function (type=%d)", symbol->type);
  }

  const bool is_pre_packed =
      (params->flags & IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED) != 0;
  const bool is_args_array =
      (params->flags & IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY) != 0;
  const bool is_native_kernel = symbol->parameters.binding_count == 0 &&
                                symbol->parameters.copy_count == 0;

  iree_host_size_t constants_capacity = symbol->parameters.constant_bytes;
  if (params->buffer_size > constants_capacity) {
    constants_capacity = params->buffer_size;
  }
  if ((is_args_array || is_native_kernel) &&
      symbol->parameters.direct_arg_bytes > constants_capacity) {
    constants_capacity = symbol->parameters.direct_arg_bytes;
  }
  if (is_native_kernel && params->buffer_size > constants_capacity) {
    constants_capacity = params->buffer_size;
  }

  // Allocate node with dependencies and params storage in a single allocation.
  iree_hal_streaming_graph_node_t* node = NULL;
  iree_host_size_t constants_size = 0;
  iree_host_size_t bindings_byte_size = 0;
  iree_host_size_t bindings_size = 0;
  iree_host_size_t extra_data_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_align(constants_capacity, iree_max_align_t,
                                        &constants_size) ||
          !iree_host_size_checked_mul(symbol->parameters.binding_count,
                                      sizeof(iree_hal_buffer_ref_t),
                                      &bindings_byte_size) ||
          !iree_host_size_checked_align(bindings_byte_size, iree_max_align_t,
                                        &bindings_size) ||
          !iree_host_size_checked_add(constants_size, bindings_size,
                                      &extra_data_size))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph kernel node data size overflow");
  }
  uint8_t* extra_data = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_allocate_node(
              graph->arena_allocator, dependency_count, extra_data_size, &node,
              &extra_data));

  node->type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL;
  node->dependency_count = dependency_count;

  // Copy dependencies to the trailing array.
  if (dependency_count > 0) {
    memcpy(node->dependencies, dependencies,
           dependency_count * sizeof(*dependencies));
  }

  // Copy kernel dispatch parameters.
  iree_hal_streaming_graph_kernel_node_attrs_t* attrs = &node->attrs.kernel;
  attrs->symbol = symbol;
  memcpy(attrs->grid_dim, params->grid_dim, sizeof(params->grid_dim));
  memcpy(attrs->block_dim, params->block_dim, sizeof(params->block_dim));
  attrs->shared_memory_bytes = params->shared_memory_bytes;

  // Unpack parameters.
  void* constants = extra_data;
  if (constants_capacity > 0) {
    memset(constants, 0, constants_capacity);
  }
  attrs->constants =
      iree_make_const_byte_span(constants, symbol->parameters.constant_bytes);
  attrs->bindings.count = symbol->parameters.binding_count;
  attrs->bindings.values =
      (iree_hal_buffer_ref_t*)(extra_data + constants_size);
  iree_status_t unpack_status = iree_ok_status();
  if (is_pre_packed && params->buffer) {
    iree_host_size_t captured_size = symbol->parameters.constant_bytes;
    if (params->buffer_size > captured_size) {
      captured_size = params->buffer_size;
    }
    if (captured_size > 0) {
      const iree_host_size_t copy_size =
          params->buffer_size ? params->buffer_size : captured_size;
      memcpy(constants, params->buffer, copy_size);
    }
    attrs->constants = iree_make_const_byte_span(constants, captured_size);
    attrs->bindings = iree_hal_buffer_ref_list_empty();
  } else if (is_native_kernel && params->buffer) {
    iree_host_size_t captured_size = symbol->parameters.direct_arg_bytes
                                         ? symbol->parameters.direct_arg_bytes
                                         : symbol->parameters.constant_bytes;
    if (params->buffer_size > captured_size) {
      captured_size = params->buffer_size;
    }
    if (captured_size > 0) {
      const iree_host_size_t copy_size =
          params->buffer_size ? params->buffer_size : captured_size;
      memcpy(constants, params->buffer, copy_size);
    }
    attrs->constants = iree_make_const_byte_span(constants, captured_size);
    attrs->bindings = iree_hal_buffer_ref_list_empty();
  } else if (is_args_array) {
    unpack_status = iree_hal_streaming_unpack_parameter_list(
        graph->context, &symbol->parameters, (void**)params->buffer, constants,
        &attrs->bindings);
    if (iree_status_code(unpack_status) == IREE_STATUS_NOT_FOUND) {
      iree_status_ignore(unpack_status);
      memset(constants, 0, constants_capacity);
      unpack_status = iree_hal_streaming_pack_raw_argument_list(
          &symbol->parameters, (void**)params->buffer, constants);
      attrs->constants = iree_make_const_byte_span(
          constants, symbol->parameters.direct_arg_bytes
                         ? symbol->parameters.direct_arg_bytes
                         : symbol->parameters.constant_bytes);
      attrs->bindings = iree_hal_buffer_ref_list_empty();
    }
  } else {
    unpack_status = iree_hal_streaming_unpack_parameters(
        graph->context, &symbol->parameters, params->buffer, constants,
        &attrs->bindings);
    if (iree_status_code(unpack_status) == IREE_STATUS_NOT_FOUND) {
      iree_status_ignore(unpack_status);
      const iree_host_size_t captured_size =
          params->buffer_size ? params->buffer_size : constants_capacity;
      if (captured_size > 0) {
        memcpy(constants, params->buffer, captured_size);
      }
      attrs->constants = iree_make_const_byte_span(constants, captured_size);
      attrs->bindings = iree_hal_buffer_ref_list_empty();
      unpack_status = iree_ok_status();
    }
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, unpack_status);

  iree_status_t status = iree_hal_streaming_graph_add_node(graph, node);
  if (iree_status_is_ok(status) && out_node) {
    *out_node = node;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_graph_add_memcpy_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_deviceptr_t src, iree_host_size_t size,
    iree_hal_streaming_graph_node_t** out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_validate_dependencies(graph, dependencies,
                                                         dependency_count));

  iree_hal_streaming_buffer_ref_t dst_ref;
  if (size > 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_streaming_memory_lookup_range(graph->context, dst, size,
                                               &dst_ref),
        "resolving `dst` buffer ref %p with size %" PRIhsz, (void*)dst, size);
  } else {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_memory_lookup(graph->context, dst, &dst_ref),
        "resolving `dst` buffer ref %p", (void*)dst);
  }
  iree_hal_streaming_buffer_ref_t src_ref;
  if (size > 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_streaming_memory_lookup_range(graph->context, src, size,
                                               &src_ref),
        "resolving `src` buffer ref %p with size %" PRIhsz, (void*)src, size);
  } else {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_memory_lookup(graph->context, src, &src_ref),
        "resolving `src` buffer ref %p", (void*)src);
  }

  // Allocate node with dependencies in a single allocation.
  iree_hal_streaming_graph_node_t* node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_allocate_node(
              graph->arena_allocator, dependency_count, 0, &node, NULL));

  node->type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY;
  node->dependency_count = dependency_count;

  // Copy dependencies to the trailing array.
  if (dependency_count > 0) {
    memcpy(node->dependencies, dependencies,
           dependency_count * sizeof(*dependencies));
  }

  // Copy memcpy data.
  iree_hal_streaming_graph_memcpy_node_attrs_t* attrs = &node->attrs.memcpy;
  attrs->dst_ref = dst_ref;
  attrs->src_ref = src_ref;
  attrs->size = size;

  iree_status_t status = iree_hal_streaming_graph_add_node(graph, node);
  if (iree_status_is_ok(status) && out_node) {
    *out_node = node;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_graph_add_memset_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, iree_hal_streaming_deviceptr_t dst,
    uint32_t pattern, iree_device_size_t pattern_size, iree_device_size_t count,
    iree_hal_streaming_graph_node_t** out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_validate_dependencies(graph, dependencies,
                                                         dependency_count));

  iree_device_size_t total_size = 0;
  if (IREE_UNLIKELY(
          !iree_device_size_checked_mul(pattern_size, count, &total_size))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "memset size overflows device size");
  }

  iree_hal_streaming_buffer_ref_t dst_ref;
  if (total_size > 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_streaming_memory_lookup_range(graph->context, dst, total_size,
                                               &dst_ref),
        "resolving `dst` buffer ref %p with size %" PRIu64, (void*)dst,
        total_size);
  } else {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_memory_lookup(graph->context, dst, &dst_ref),
        "resolving `dst` buffer ref %p", (void*)dst);
  }

  // Allocate node with dependencies in a single allocation.
  iree_hal_streaming_graph_node_t* node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_allocate_node(
              graph->arena_allocator, dependency_count, 0, &node, NULL));

  node->type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET;
  node->dependency_count = dependency_count;

  // Copy dependencies to the trailing array.
  if (dependency_count > 0) {
    memcpy(node->dependencies, dependencies,
           dependency_count * sizeof(*dependencies));
  }

  // Copy memset data.
  iree_hal_streaming_graph_memset_node_attrs_t* attrs = &node->attrs.memset;
  attrs->dst_ref = dst_ref;
  attrs->pattern = pattern;
  attrs->pattern_size = pattern_size;
  attrs->count = count;

  iree_status_t status = iree_hal_streaming_graph_add_node(graph, node);
  if (iree_status_is_ok(status) && out_node) {
    *out_node = node;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_graph_add_host_call_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, void (*fn)(void*), void* user_data,
    iree_hal_streaming_graph_node_t** out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(fn);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_validate_dependencies(graph, dependencies,
                                                         dependency_count));

  // Allocate node with dependencies in a single allocation.
  iree_hal_streaming_graph_node_t* node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_allocate_node(
              graph->arena_allocator, dependency_count, 0, &node, NULL));

  node->type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_HOST_CALL;
  node->dependency_count = dependency_count;

  // Copy dependencies to the trailing array.
  if (dependency_count > 0) {
    memcpy(node->dependencies, dependencies,
           dependency_count * sizeof(*dependencies));
  }

  // Copy host function data.
  iree_hal_streaming_graph_host_call_node_attrs_t* attrs = &node->attrs.host;
  attrs->fn = fn;
  attrs->user_data = user_data;

  iree_status_t status = iree_hal_streaming_graph_add_node(graph, node);
  if (iree_status_is_ok(status) && out_node) {
    *out_node = node;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_graph_add_event_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count,
    iree_hal_streaming_graph_node_type_t type,
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_graph_node_t** out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(event);
  IREE_TRACE_ZONE_BEGIN(z0);
  if (type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD &&
      type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid graph event node type");
  }
  if (event->context != graph->context) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "event must belong to the graph context");
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_validate_dependencies(graph, dependencies,
                                                         dependency_count));

  iree_hal_streaming_graph_node_t* node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_allocate_node(
              graph->arena_allocator, dependency_count, 0, &node, NULL));

  node->type = type;
  node->dependency_count = dependency_count;
  if (dependency_count > 0) {
    memcpy(node->dependencies, dependencies,
           dependency_count * sizeof(*dependencies));
  }
  node->attrs.event.event = event;
  iree_hal_streaming_event_retain(event);

  iree_status_t status = iree_hal_streaming_graph_add_node(graph, node);
  if (iree_status_is_ok(status)) {
    if (out_node) *out_node = node;
  } else {
    iree_hal_streaming_graph_node_deinitialize_attrs(node);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_graph_add_child_graph_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, iree_hal_streaming_graph_t* child_graph,
    iree_hal_streaming_graph_node_t** out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(child_graph);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_validate_dependencies(graph, dependencies,
                                                         dependency_count));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_validate_child_graph(graph, child_graph));

  iree_hal_streaming_graph_node_t* node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_allocate_node(
              graph->arena_allocator, dependency_count, 0, &node, NULL));

  node->type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH;
  node->dependency_count = dependency_count;
  if (dependency_count > 0) {
    memcpy(node->dependencies, dependencies,
           dependency_count * sizeof(*dependencies));
  }
  node->attrs.child_graph.graph = child_graph;
  iree_hal_streaming_graph_retain(child_graph);

  iree_status_t status = iree_hal_streaming_graph_add_node(graph, node);
  if (iree_status_is_ok(status)) {
    ++graph->child_graph_node_count;
    if (out_node) *out_node = node;
  } else {
    iree_hal_streaming_graph_node_deinitialize_attrs(node);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_graph_add_dependencies(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** from_nodes,
    iree_hal_streaming_graph_node_t** to_nodes, iree_host_size_t count) {
  IREE_ASSERT_ARGUMENT(graph);
  if (count == 0) return iree_ok_status();
  if (!from_nodes || !to_nodes) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dependency arrays must be provided");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_streaming_graph_node_t* from_node = from_nodes[i];
    iree_hal_streaming_graph_node_t* to_node = to_nodes[i];
    if (!from_node || !to_node) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "null node in dependency list at index %" PRIhsz,
                              i);
    }
    if (from_node == to_node) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "node cannot depend on itself at index %" PRIhsz,
                              i);
    }
    if (!iree_hal_streaming_graph_node_is_active_in_graph(graph, from_node) ||
        !iree_hal_streaming_graph_node_is_active_in_graph(graph, to_node)) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dependency node at index %" PRIhsz
                              " does not belong to the target graph",
                              i);
    }
    if (iree_hal_streaming_graph_dependency_exists(graph, from_node, to_node)) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate dependency at index %" PRIhsz, i);
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (from_nodes[j] == from_node && to_nodes[j] == to_node) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "duplicate dependency within request at index %" PRIhsz, i);
      }
    }
  }

  for (iree_host_size_t i = 0; i < count; ++i) {
    // Allocate edge from arena.
    iree_hal_streaming_graph_edge_t* edge = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_malloc(graph->arena_allocator,
                                  sizeof(iree_hal_streaming_graph_edge_t),
                                  (void**)&edge));

    edge->from = from_nodes[i];
    edge->to = to_nodes[i];
    edge->next = graph->additional_edges;
    graph->additional_edges = edge;
    ++graph->additional_edge_count;

    // If 'to' node was a root (no dependencies), it's no longer a root.
    // We need to remove it from the root blocks.
    // For simplicity, we'll handle this during graph analysis instead.
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_graph_destroy_node(
    iree_hal_streaming_graph_node_t* node) {
  if (!node || !node->graph) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "node must belong to an active graph");
  }
  iree_hal_streaming_graph_t* graph = node->graph;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_graph_remove_dependency_refs(graph, node);
  iree_hal_streaming_graph_remove_additional_edges(graph, node);

  const bool removed_from_nodes =
      iree_hal_streaming_graph_remove_from_blocks(graph->node_blocks, node);
  if (removed_from_nodes) {
    --graph->node_count;
    if (node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH) {
      --graph->child_graph_node_count;
    }
  }
  if (iree_hal_streaming_graph_remove_from_blocks(graph->root_blocks, node)) {
    --graph->root_count;
  }
  if (removed_from_nodes) {
    iree_hal_streaming_graph_node_deinitialize_attrs(node);
    iree_hal_streaming_graph_renumber_nodes(graph);
    node->graph = NULL;
    node->dependency_count = 0;
  }

  IREE_TRACE_ZONE_END(z0);

  if (!removed_from_nodes) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "node not found in owning graph");
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_streaming_graph_exec_t (instantiation)
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_graph_instantiate(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_instantiate_flags_t flags,
    iree_hal_streaming_graph_exec_t** out_exec) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(out_exec);
  *out_exec = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Create an uninitialized exec object.
  iree_hal_streaming_graph_exec_t* exec = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_exec_create(graph->context, graph, flags,
                                               graph->host_allocator, &exec));

  // Instantiate from the graph template. HIP/CUDA graph objects are not
  // internally synchronized; callers must externally serialize access to a
  // graph while it is being modified, queried, or instantiated.
  iree_status_t status =
      iree_hal_streaming_graph_exec_instantiate_from_template(
          exec, graph->node_blocks, graph->node_count);

  if (iree_status_is_ok(status)) {
    *out_exec = exec;
  } else {
    iree_hal_streaming_graph_exec_release(exec);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Stream capture internal functions
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_streaming_grow_capture_dependencies(
    iree_hal_streaming_stream_t* stream, iree_host_size_t required_capacity);

iree_status_t iree_hal_streaming_begin_capture(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_capture_mode_t mode) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Flush any pending operations before starting capture.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_streaming_stream_flush(stream));

  iree_slim_mutex_lock(&stream->mutex);

  // Check if already capturing.
  if (stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
    iree_slim_mutex_unlock(&stream->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream is already capturing");
  }

  // Create a new graph for capture.
  iree_status_t status = iree_hal_streaming_graph_create(
      stream->context, /*flags=*/0, stream->host_allocator,
      &stream->capture_graph);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&stream->mutex);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Set capture state.
  stream->capture_mode = mode;
  stream->capture_graph_owned = true;
  stream->capture_id++;  // Increment capture ID.
  stream->capture_owner_thread_id = iree_hal_streaming_current_thread_token();
  iree_hal_streaming_stream_set_capture_status(
      stream, IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE);

  iree_slim_mutex_unlock(&stream->mutex);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_begin_capture_to_graph(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, iree_hal_streaming_capture_mode_t mode) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(graph);
  IREE_TRACE_ZONE_BEGIN(z0);
  if (dependency_count > 0 && !dependencies) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dependency array must be provided");
  }

  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_streaming_stream_flush(stream));

  iree_slim_mutex_lock(&stream->mutex);

  if (stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
    iree_slim_mutex_unlock(&stream->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream is already capturing");
  }

  if (dependency_count > stream->capture_dependency_capacity) {
    iree_status_t status =
        iree_hal_streaming_grow_capture_dependencies(stream, dependency_count);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&stream->mutex);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }
  if (dependency_count > 0) {
    memcpy(stream->capture_dependencies, dependencies,
           dependency_count * sizeof(*dependencies));
  }

  stream->capture_mode = mode;
  stream->capture_graph = graph;
  stream->capture_graph_owned = false;
  stream->capture_id++;
  stream->capture_owner_thread_id = iree_hal_streaming_current_thread_token();
  stream->capture_dependency_count = dependency_count;
  iree_hal_streaming_stream_set_capture_status(
      stream, IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE);

  iree_slim_mutex_unlock(&stream->mutex);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_streaming_clear_capture_participants(
    iree_hal_streaming_stream_t* origin_stream,
    iree_hal_streaming_graph_t* graph) {
  iree_hal_streaming_context_t* context = origin_stream->context;
  iree_host_size_t owned_graph_release_count = 0;
  iree_slim_mutex_lock(&context->stream_list_mutex);
  for (iree_host_size_t i = 0; i < context->stream_count; ++i) {
    iree_hal_streaming_stream_t* stream = context->streams[i];
    if (stream == origin_stream) continue;

    iree_slim_mutex_lock(&stream->mutex);
    if ((stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE ||
         stream->capture_status ==
             IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED) &&
        stream->capture_graph == graph) {
      iree_hal_streaming_stream_set_capture_status(
          stream, IREE_HAL_STREAMING_CAPTURE_STATUS_NONE);
      stream->capture_id = 0;
      stream->capture_owner_thread_id = 0;
      const bool capture_graph_owned = stream->capture_graph_owned;
      stream->capture_graph = NULL;
      stream->capture_graph_owned = false;
      stream->capture_dependency_count = 0;
      iree_slim_mutex_unlock(&stream->mutex);
      if (capture_graph_owned) {
        ++owned_graph_release_count;
      }
      continue;
    }
    iree_slim_mutex_unlock(&stream->mutex);
  }
  iree_slim_mutex_unlock(&context->stream_list_mutex);
  while (owned_graph_release_count-- > 0) {
    iree_hal_streaming_graph_release(graph);
  }
}

typedef struct iree_hal_streaming_graph_additional_edge_index_t {
  uint32_t* head_indices;
  uint32_t* next_indices;
  iree_hal_streaming_graph_node_t** from_nodes;
} iree_hal_streaming_graph_additional_edge_index_t;

static void iree_hal_streaming_graph_deinitialize_additional_edge_index(
    iree_allocator_t host_allocator,
    iree_hal_streaming_graph_additional_edge_index_t* index) {
  iree_allocator_free(host_allocator, index->from_nodes);
  iree_allocator_free(host_allocator, index->next_indices);
  iree_allocator_free(host_allocator, index->head_indices);
}

static iree_status_t iree_hal_streaming_graph_initialize_additional_edge_index(
    iree_hal_streaming_graph_t* graph, iree_allocator_t host_allocator,
    iree_hal_streaming_graph_additional_edge_index_t* out_index) {
  memset(out_index, 0, sizeof(*out_index));
  if (graph->node_count == 0 || graph->additional_edge_count == 0) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(graph->additional_edge_count >= UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph edge count exceeds supported range");
  }

  iree_host_size_t head_index_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          graph->node_count, sizeof(*out_index->head_indices),
          &head_index_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph edge index allocation size overflow");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, head_index_size,
                                             (void**)&out_index->head_indices));
  for (iree_host_size_t i = 0; i < graph->node_count; ++i) {
    out_index->head_indices[i] = UINT32_MAX;
  }

  iree_host_size_t edge_index_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          graph->additional_edge_count, sizeof(*out_index->next_indices),
          &edge_index_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph edge index allocation size overflow");
  }
  iree_status_t status = iree_allocator_malloc(
      host_allocator, edge_index_size, (void**)&out_index->next_indices);
  if (iree_status_is_ok(status)) {
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            graph->additional_edge_count, sizeof(*out_index->from_nodes),
            &edge_index_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "graph edge index allocation size overflow");
    } else {
      status = iree_allocator_malloc(host_allocator, edge_index_size,
                                     (void**)&out_index->from_nodes);
    }
  }
  if (!iree_status_is_ok(status)) {
    return status;
  }

  uint32_t edge_index = 0;
  for (iree_hal_streaming_graph_edge_t* edge = graph->additional_edges; edge;
       edge = edge->next, ++edge_index) {
    if (!edge->from || !edge->to || edge->from->graph != graph ||
        edge->to->graph != graph || edge->to->node_index >= graph->node_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid graph additional edge");
    }
    const uint32_t to_index = edge->to->node_index;
    out_index->from_nodes[edge_index] = edge->from;
    out_index->next_indices[edge_index] = out_index->head_indices[to_index];
    out_index->head_indices[to_index] = edge_index;
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_capture_push_reachable_node(
    iree_hal_streaming_graph_t* graph, iree_hal_streaming_graph_node_t* node,
    uint8_t* reachable_nodes, iree_hal_streaming_graph_node_t** stack,
    iree_host_size_t* stack_count) {
  if (!node || node->graph != graph || node->node_index >= graph->node_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "capture dependency is not in the capture graph");
  }
  if (reachable_nodes[node->node_index]) {
    return iree_ok_status();
  }
  reachable_nodes[node->node_index] = 1;
  stack[(*stack_count)++] = node;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_capture_mark_frontier_reachable(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** frontier_nodes,
    iree_host_size_t frontier_node_count,
    const iree_hal_streaming_graph_additional_edge_index_t*
        additional_edge_index,
    uint8_t* reachable_nodes, iree_hal_streaming_graph_node_t** stack) {
  iree_host_size_t stack_count = 0;
  for (iree_host_size_t i = 0; i < frontier_node_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_streaming_capture_push_reachable_node(
        graph, frontier_nodes[i], reachable_nodes, stack, &stack_count));
  }

  while (stack_count > 0) {
    iree_hal_streaming_graph_node_t* node = stack[--stack_count];
    for (uint32_t i = 0; i < node->dependency_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_streaming_capture_push_reachable_node(
          graph, node->dependencies[i], reachable_nodes, stack, &stack_count));
    }
    if (!additional_edge_index->head_indices) continue;
    for (uint32_t edge_index =
             additional_edge_index->head_indices[node->node_index];
         edge_index != UINT32_MAX;
         edge_index = additional_edge_index->next_indices[edge_index]) {
      IREE_RETURN_IF_ERROR(iree_hal_streaming_capture_push_reachable_node(
          graph, additional_edge_index->from_nodes[edge_index], reachable_nodes,
          stack, &stack_count));
    }
  }

  return iree_ok_status();
}

static bool iree_hal_streaming_capture_frontier_is_joined(
    iree_hal_streaming_graph_t* graph, const uint8_t* reachable_nodes,
    const iree_hal_streaming_stream_t* participant_stream) {
  for (iree_host_size_t i = 0; i < participant_stream->capture_dependency_count;
       ++i) {
    iree_hal_streaming_graph_node_t* node =
        participant_stream->capture_dependencies[i];
    if (!node || node->graph != graph ||
        node->node_index >= graph->node_count ||
        !reachable_nodes[node->node_index]) {
      return false;
    }
  }
  return true;
}

static iree_status_t iree_hal_streaming_has_unjoined_capture_participants(
    iree_hal_streaming_stream_t* origin_stream,
    iree_hal_streaming_graph_t* graph, bool* out_has_unjoined_participant) {
  *out_has_unjoined_participant = false;

  iree_allocator_t host_allocator = origin_stream->host_allocator;
  uint8_t* reachable_nodes = NULL;
  iree_hal_streaming_graph_node_t** stack = NULL;
  iree_hal_streaming_graph_additional_edge_index_t additional_edge_index;
  memset(&additional_edge_index, 0, sizeof(additional_edge_index));

  iree_status_t status = iree_ok_status();
  if (graph->node_count > 0) {
    status = iree_allocator_malloc(host_allocator, graph->node_count,
                                   (void**)&reachable_nodes);
    if (iree_status_is_ok(status)) {
      memset(reachable_nodes, 0, graph->node_count);
      iree_host_size_t stack_size = 0;
      if (IREE_UNLIKELY(!iree_host_size_checked_mul(
              graph->node_count, sizeof(*stack), &stack_size))) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "capture reachability stack size overflow");
      } else {
        status =
            iree_allocator_malloc(host_allocator, stack_size, (void**)&stack);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_initialize_additional_edge_index(
        graph, host_allocator, &additional_edge_index);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_capture_mark_frontier_reachable(
        graph, origin_stream->capture_dependencies,
        origin_stream->capture_dependency_count, &additional_edge_index,
        reachable_nodes, stack);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_graph_deinitialize_additional_edge_index(
        host_allocator, &additional_edge_index);
    iree_allocator_free(host_allocator, stack);
    iree_allocator_free(host_allocator, reachable_nodes);
    return status;
  }

  iree_hal_streaming_context_t* context = origin_stream->context;
  iree_slim_mutex_lock(&context->stream_list_mutex);
  for (iree_host_size_t i = 0; i < context->stream_count; ++i) {
    iree_hal_streaming_stream_t* stream = context->streams[i];
    if (stream == origin_stream) {
      continue;
    }

    iree_slim_mutex_lock(&stream->mutex);
    if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE &&
        stream->capture_graph == graph &&
        !iree_hal_streaming_capture_frontier_is_joined(graph, reachable_nodes,
                                                       stream)) {
      *out_has_unjoined_participant = true;
    }
    iree_slim_mutex_unlock(&stream->mutex);
    if (*out_has_unjoined_participant) break;
  }
  iree_slim_mutex_unlock(&context->stream_list_mutex);

  iree_hal_streaming_graph_deinitialize_additional_edge_index(
      host_allocator, &additional_edge_index);
  iree_allocator_free(host_allocator, stack);
  iree_allocator_free(host_allocator, reachable_nodes);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_end_capture(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_graph_t** out_graph) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&stream->mutex);

  // Check capture status.
  if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
    iree_slim_mutex_unlock(&stream->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream is not capturing");
  }
  if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED) {
    iree_hal_streaming_graph_t* graph = stream->capture_graph;
    const bool capture_graph_owned = stream->capture_graph_owned;
    iree_hal_streaming_stream_set_capture_status(
        stream, IREE_HAL_STREAMING_CAPTURE_STATUS_NONE);
    stream->capture_graph = NULL;
    stream->capture_graph_owned = false;
    stream->capture_id = 0;
    stream->capture_owner_thread_id = 0;
    stream->capture_dependency_count = 0;
    iree_slim_mutex_unlock(&stream->mutex);
    iree_hal_streaming_clear_capture_participants(stream, graph);
    if (capture_graph_owned) {
      iree_hal_streaming_graph_release(graph);
    }
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "stream capture has been invalidated");
  }

  iree_hal_streaming_graph_t* graph = stream->capture_graph;
  bool has_unjoined_participant = false;
  iree_status_t joined_status =
      iree_hal_streaming_has_unjoined_capture_participants(
          stream, graph, &has_unjoined_participant);
  if (!iree_status_is_ok(joined_status)) {
    iree_slim_mutex_unlock(&stream->mutex);
    IREE_TRACE_ZONE_END(z0);
    return joined_status;
  }
  if (has_unjoined_participant) {
    iree_slim_mutex_unlock(&stream->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_ABORTED,
        "stream capture has participant work not joined to the origin stream");
  }

  stream->capture_graph = NULL;
  stream->capture_graph_owned = false;

  // Clear capture state.
  iree_hal_streaming_stream_set_capture_status(
      stream, IREE_HAL_STREAMING_CAPTURE_STATUS_NONE);
  stream->capture_id = 0;
  stream->capture_owner_thread_id = 0;

  // Reset dependency count but keep the buffer for reuse.
  stream->capture_dependency_count = 0;
  // Note: keeping capture_dependencies and capture_dependency_capacity
  // unchanged for reuse in next capture session.

  iree_slim_mutex_unlock(&stream->mutex);

  iree_hal_streaming_clear_capture_participants(stream, graph);

  if (out_graph) {
    *out_graph = graph;
  } else {
    iree_hal_streaming_graph_release(graph);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_capture_status(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_capture_status_t* out_status,
    unsigned long long* out_id) {
  IREE_ASSERT_ARGUMENT(stream);

  iree_slim_mutex_lock(&stream->mutex);

  if (out_status) {
    *out_status = stream->capture_status;
  }
  if (out_id) {
    *out_id = stream->capture_id;
  }

  iree_slim_mutex_unlock(&stream->mutex);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_is_capturing(
    iree_hal_streaming_stream_t* stream, bool* out_is_capturing) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(out_is_capturing);

  iree_slim_mutex_lock(&stream->mutex);
  *out_is_capturing =
      (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE);
  iree_slim_mutex_unlock(&stream->mutex);

  return iree_ok_status();
}

// Helper to grow the capture dependencies array.
static iree_status_t iree_hal_streaming_grow_capture_dependencies(
    iree_hal_streaming_stream_t* stream, iree_host_size_t required_capacity) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, required_capacity);

  // Calculate new capacity (at least 2x required).
  iree_host_size_t new_capacity = 0;
  iree_host_size_t allocation_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(required_capacity, 2, &new_capacity) ||
          !iree_host_size_checked_mul(new_capacity,
                                      sizeof(iree_hal_streaming_graph_node_t*),
                                      &allocation_size))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "capture dependency array allocation size overflow");
  }

  // Use realloc to potentially extend in-place.
  iree_status_t status =
      iree_allocator_realloc(stream->host_allocator, allocation_size,
                             (void**)&stream->capture_dependencies);
  if (iree_status_is_ok(status)) {
    stream->capture_dependency_capacity = new_capacity;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_update_capture_dependencies(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count,
    iree_hal_streaming_capture_dependencies_mode_t mode) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, dependency_count);
  if (dependency_count > 0 && !dependencies) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dependency array must be provided");
  }

  iree_slim_mutex_lock(&stream->mutex);

  // Check if capturing.
  if (stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    iree_slim_mutex_unlock(&stream->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream is not actively capturing");
  }

  // Calculate total count based on mode.
  iree_host_size_t total_count = dependency_count;
  if (mode == IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_ADD &&
      IREE_UNLIKELY(!iree_host_size_checked_add(
          stream->capture_dependency_count, dependency_count, &total_count))) {
    iree_slim_mutex_unlock(&stream->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "capture dependency count overflow");
  }

  // Grow dependency array if needed.
  if (total_count > stream->capture_dependency_capacity) {
    iree_status_t status =
        iree_hal_streaming_grow_capture_dependencies(stream, total_count);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&stream->mutex);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }

  // Copy dependencies based on mode.
  if (dependency_count > 0) {
    void* dest =
        (mode == IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_ADD)
            ? stream->capture_dependencies + stream->capture_dependency_count
            : stream->capture_dependencies;
    memcpy(dest, dependencies, dependency_count * sizeof(*dependencies));
  }

  stream->capture_dependency_count = total_count;

  iree_slim_mutex_unlock(&stream->mutex);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_capture_set_last_node(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_graph_node_t* node) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(node);
  return iree_hal_streaming_update_capture_dependencies(
      stream, &node, 1, IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_SET);
}
