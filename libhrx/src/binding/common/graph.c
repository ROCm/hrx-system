// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/graph.h"

#include "common/graph_memory.h"
#include "common/internal.h"
#include "common/kernel_arguments.h"
#include "common/memory.h"
#include "common/peer.h"
#include "iree/base/threading/call_once.h"

//===----------------------------------------------------------------------===//
// iree_hal_streaming_graph_t (template)
//===----------------------------------------------------------------------===//

static void iree_hal_streaming_graph_destroy(iree_hal_streaming_graph_t* graph);

static void iree_hal_streaming_graph_node_deinitialize_attrs(
    iree_hal_streaming_graph_node_t* node) {
  switch (node->type) {
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL:
      iree_hal_streaming_module_release(node->attrs.kernel.module);
      node->attrs.kernel.module = NULL;
      node->attrs.kernel.symbol = NULL;
      break;
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH:
      iree_hal_streaming_graph_release(node->attrs.child_graph.graph);
      node->attrs.child_graph.graph = NULL;
      break;
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD:
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT:
      iree_hal_streaming_event_release(node->attrs.event.event);
      node->attrs.event.event = NULL;
      break;
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC: {
      if (node->graph->owns_graph_memory_node_claims) {
        iree_status_t unmap_status =
            iree_hal_streaming_graph_memory_allocation_unmap_if_mapped(
                node->attrs.mem_alloc.allocation);
        if (!iree_status_is_ok(unmap_status)) {
          iree_status_abort(unmap_status);
        }
        if (iree_hal_streaming_graph_memory_allocation_claim_unexecuted_pointer_reference(
                node->attrs.mem_alloc.allocation)) {
          iree_hal_streaming_graph_memory_allocation_release(
              node->attrs.mem_alloc.allocation);
        }
      }
      iree_hal_streaming_graph_memory_allocation_release(
          node->attrs.mem_alloc.allocation);
      node->attrs.mem_alloc.params = NULL;
      node->attrs.mem_alloc.params_size = 0;
      node->attrs.mem_alloc.dptr = NULL;
      node->attrs.mem_alloc.bytesize = 0;
      node->attrs.mem_alloc.allocation = NULL;
      node->attrs.mem_alloc.has_in_graph_free_node = false;
      break;
    }
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_FREE:
      if (node->graph->owns_graph_memory_node_claims) {
        iree_hal_streaming_graph_memory_allocation_release_free_node_reference(
            node->attrs.mem_free.allocation);
      }
      iree_hal_streaming_graph_memory_allocation_release(
          node->attrs.mem_free.allocation);
      node->attrs.mem_free.dptr = NULL;
      node->attrs.mem_free.allocation = NULL;
      break;
    case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_BATCH_MEM_OP:
      for (iree_host_size_t i = 0; i < node->attrs.batch_mem_op.operation_count;
           ++i) {
        iree_hal_buffer_release(
            node->attrs.batch_mem_op.operations[i].target_buffer);
        hrx_buffer_release(node->attrs.batch_mem_op.owners[i]);
      }
      node->attrs.batch_mem_op.operation_count = 0;
      break;
    default:
      break;
  }
}

typedef struct iree_hal_streaming_graph_batch_mem_op_layout_t {
  // Aligned byte capacity reserved for the opaque node parameters.
  iree_host_size_t params_capacity;
  // Byte offset of the opaque operation array.
  iree_host_size_t param_array_offset;
  // Aligned byte capacity reserved for the opaque operation array.
  iree_host_size_t param_array_capacity;
  // Byte offset of the resolved operation array.
  iree_host_size_t operations_offset;
  // Byte offset of the allocation-owner array.
  iree_host_size_t owners_offset;
  // Total storage required by all arrays.
  iree_host_size_t total_size;
} iree_hal_streaming_graph_batch_mem_op_layout_t;

static iree_status_t iree_hal_streaming_graph_batch_mem_op_layout_calculate(
    iree_host_size_t params_size, iree_host_size_t param_array_size,
    iree_host_size_t operation_count,
    iree_hal_streaming_graph_batch_mem_op_layout_t* out_layout) {
  *out_layout = (iree_hal_streaming_graph_batch_mem_op_layout_t){0};
  iree_host_size_t operations_size = 0;
  iree_host_size_t owners_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_align(params_size, iree_max_align_t,
                                        &out_layout->params_capacity) ||
          !iree_host_size_checked_align(param_array_size, iree_max_align_t,
                                        &out_layout->param_array_capacity) ||
          !iree_host_size_checked_mul(
              operation_count, sizeof(iree_hal_streaming_value_operation_t),
              &operations_size) ||
          !iree_host_size_checked_align(operations_size, iree_max_align_t,
                                        &operations_size) ||
          !iree_host_size_checked_mul(operation_count, sizeof(hrx_buffer_t),
                                      &owners_size) ||
          !iree_host_size_checked_align(owners_size, iree_max_align_t,
                                        &owners_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "batch mem op node payload size overflow");
  }
  out_layout->param_array_offset = out_layout->params_capacity;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_add(out_layout->param_array_offset,
                                      out_layout->param_array_capacity,
                                      &out_layout->operations_offset) ||
          !iree_host_size_checked_add(out_layout->operations_offset,
                                      operations_size,
                                      &out_layout->owners_offset) ||
          !iree_host_size_checked_add(out_layout->owners_offset, owners_size,
                                      &out_layout->total_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "batch mem op node payload offset overflow");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_validate_batch_mem_operations(
    const iree_hal_streaming_value_operation_t* operations,
    const hrx_buffer_t* owners, iree_host_size_t operation_count) {
  if (IREE_UNLIKELY(operation_count == 0 || !operations || !owners)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "batch mem op node requires operations");
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < operation_count && iree_status_is_ok(status);
       ++i) {
    if (IREE_UNLIKELY(!owners[i])) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "batch mem op target has no allocation owner");
    } else {
      status = iree_hal_streaming_value_operation_validate(&operations[i]);
    }
  }
  return status;
}

void iree_hal_streaming_graph_refresh_memory_allocation_free_node_state(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_memory_allocation_t* allocation) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(allocation);

  iree_hal_streaming_graph_node_t* allocation_node = NULL;
  bool has_free_node = false;
  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC &&
          node->attrs.mem_alloc.allocation == allocation) {
        allocation_node = node;
      } else if (node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_FREE &&
                 node->attrs.mem_free.allocation == allocation) {
        has_free_node = true;
      }
      if (allocation_node && has_free_node) {
        break;
      }
    }
    if (allocation_node && has_free_node) {
      break;
    }
  }
  if (allocation_node) {
    allocation_node->attrs.mem_alloc.has_in_graph_free_node = has_free_node;
  }
}

static iree_atomic_uint64_t iree_hal_streaming_next_graph_debug_id =
    IREE_ATOMIC_VAR_INIT(1);
static iree_atomic_uint64_t iree_hal_streaming_next_node_debug_id =
    IREE_ATOMIC_VAR_INIT(1);

// Graph handles cross public bindings as raw pointers. This intrusive registry
// permits identity checks without dereferencing an arbitrary address. Graph
// destruction removes the entry before releasing the object's storage.
static iree_once_flag iree_hal_streaming_live_graph_mutex_once =
    IREE_ONCE_FLAG_INIT;
static iree_slim_mutex_t iree_hal_streaming_live_graph_mutex;
static iree_hal_streaming_graph_t* iree_hal_streaming_live_graph_head = NULL;

static void iree_hal_streaming_live_graph_mutex_initialize(void) {
  iree_slim_mutex_initialize(&iree_hal_streaming_live_graph_mutex);
}

static void iree_hal_streaming_live_graph_lock(void) {
  iree_call_once(&iree_hal_streaming_live_graph_mutex_once,
                 iree_hal_streaming_live_graph_mutex_initialize);
  iree_slim_mutex_lock(&iree_hal_streaming_live_graph_mutex);
}

static void iree_hal_streaming_live_graph_register(
    iree_hal_streaming_graph_t* graph) {
  iree_hal_streaming_live_graph_lock();
  graph->next_live_graph = iree_hal_streaming_live_graph_head;
  graph->is_live_registered = true;
  iree_hal_streaming_live_graph_head = graph;
  iree_slim_mutex_unlock(&iree_hal_streaming_live_graph_mutex);
}

static void iree_hal_streaming_live_graph_unregister(
    iree_hal_streaming_graph_t* graph) {
  iree_hal_streaming_live_graph_lock();
  iree_hal_streaming_graph_t** current = &iree_hal_streaming_live_graph_head;
  while (*current && *current != graph) {
    current = &(*current)->next_live_graph;
  }
  IREE_ASSERT(*current == graph, "graph missing from identity registry");
  if (*current == graph) {
    *current = graph->next_live_graph;
  }
  graph->next_live_graph = NULL;
  graph->is_live_registered = false;
  iree_slim_mutex_unlock(&iree_hal_streaming_live_graph_mutex);
}

bool iree_hal_streaming_graph_is_capture_active(
    const iree_hal_streaming_graph_t* graph) {
  if (!graph) {
    return false;
  }
  iree_hal_streaming_live_graph_lock();
  bool is_live = false;
  for (const iree_hal_streaming_graph_t* current =
           iree_hal_streaming_live_graph_head;
       current; current = current->next_live_graph) {
    if (current == graph) {
      is_live = true;
      break;
    }
  }
  const bool is_active =
      is_live &&
      iree_atomic_load(&graph->capture_state, iree_memory_order_acquire) !=
          IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_INACTIVE;
  iree_slim_mutex_unlock(&iree_hal_streaming_live_graph_mutex);
  return is_active;
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
  graph->next_live_graph = NULL;
  graph->is_live_registered = false;
  iree_slim_mutex_initialize(&graph->capture_mutex);
  iree_atomic_store(&graph->capture_state,
                    IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_INACTIVE,
                    iree_memory_order_relaxed);
  graph->capture_id = 0;
  graph->capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL;
  graph->capture_owner_thread_id = 0;

  // Initialize the arena using the device's block pool.
  iree_hal_streaming_device_t* device = context->device_entry;
  iree_arena_initialize(&device->block_pool, &graph->arena);
  graph->arena_allocator = iree_arena_allocator(&graph->arena);

  graph->node_blocks = NULL;
  graph->current_node_block = NULL;
  graph->node_count = 0;
  graph->child_graph_node_count = 0;
  graph->debug_id = iree_atomic_fetch_add(
      &iree_hal_streaming_next_graph_debug_id, 1, iree_memory_order_relaxed);
  graph->clone_source_graph_debug_id = 0;
  graph->next_clone_source_node_index = 0;
  graph->root_blocks = NULL;
  graph->current_root_block = NULL;
  graph->root_count = 0;
  graph->additional_edges = NULL;
  graph->additional_edge_count = 0;
  graph->owned_host_allocations = NULL;
  graph->user_object_refs = NULL;
  graph->has_graph_memory_nodes = false;
  graph->owns_graph_memory_node_claims = true;
  iree_atomic_store(&graph->active_graph_memory_exec_count, 0,
                    iree_memory_order_relaxed);
  graph->flags = flags;
  graph->context = context;
  iree_hal_streaming_context_retain(context);
  graph->execution_context_hint = context;
  iree_hal_streaming_context_retain(context);
  graph->host_allocator = host_allocator;

  iree_hal_streaming_live_graph_register(graph);

  *out_graph = graph;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_streaming_graph_destroy(
    iree_hal_streaming_graph_t* graph) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (graph->is_live_registered) {
    iree_hal_streaming_live_graph_unregister(graph);
  }

  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_deinitialize_attrs(block->nodes[i]);
    }
  }

  iree_hal_streaming_graph_owned_host_allocation_t* owned_allocation =
      graph->owned_host_allocations;
  while (owned_allocation) {
    // Executables retain their source graph through launch completion, so no
    // queued operation can still reference graph-private staging here.
    iree_hal_streaming_memory_release_wrapped_buffer(owned_allocation->buffer);
    owned_allocation->buffer = NULL;
    owned_allocation = owned_allocation->next;
  }

  for (iree_hal_streaming_graph_user_object_ref_t* user_ref =
           graph->user_object_refs;
       user_ref; user_ref = user_ref->next) {
    if (user_ref->count > 0) {
      user_ref->release(user_ref->object, user_ref->count);
    }
  }

  // Reset the arena - this frees all nodes and arrays at once.
  // The arena returns all blocks to the device's block pool for reuse.
  iree_arena_deinitialize(&graph->arena);

  iree_slim_mutex_deinitialize(&graph->capture_mutex);
  // Release context.
  iree_hal_streaming_context_release(graph->execution_context_hint);
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

  // Reserve graph metadata first. If this fails no external allocation needs
  // to be released while a capture transaction may hold graph/stream locks.
  iree_hal_streaming_graph_owned_host_allocation_t* owned_allocation = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&graph->arena, sizeof(*owned_allocation),
                              (void**)&owned_allocation));

  iree_hal_streaming_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_allocate_host_staging(graph->context, size,
                                                          &buffer));

  owned_allocation->buffer = buffer;
  owned_allocation->host_ptr = buffer->host_ptr;
  owned_allocation->device_ptr = buffer->device_ptr;
  owned_allocation->size = size;
  owned_allocation->next = graph->owned_host_allocations;
  graph->owned_host_allocations = owned_allocation;
  *out_buffer = buffer;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
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
  if (!node || node->graph != graph) {
    return false;
  }
  if (node->node_index >= graph->node_count) {
    return false;
  }
  return iree_hal_streaming_graph_node_at_index(graph, node->node_index) ==
         node;
}

static bool iree_hal_streaming_graph_dependency_exists(
    const iree_hal_streaming_graph_t* graph,
    const iree_hal_streaming_graph_node_t* from_node,
    const iree_hal_streaming_graph_node_t* to_node) {
  for (uint32_t i = 0; i < to_node->dependency_count; ++i) {
    if (to_node->dependencies[i] == from_node) {
      return true;
    }
  }
  for (iree_hal_streaming_graph_edge_t* edge = graph->additional_edges; edge;
       edge = edge->next) {
    if (edge->from == from_node && edge->to == to_node) {
      return true;
    }
  }
  return false;
}

static iree_status_t iree_hal_streaming_graph_validate_dependencies(
    const iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count) {
  if (dependency_count == 0) {
    return iree_ok_status();
  }
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
    if (graphs[i] == graph) {
      return true;
    }
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

iree_status_t iree_hal_streaming_graph_validate_child_graph(
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
    if (graph->child_graph_node_count == 0) {
      continue;
    }

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
        if (!iree_status_is_ok(status)) {
          break;
        }
      }
      if (!iree_status_is_ok(status)) {
        break;
      }
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
      if (block->nodes[i] != node) {
        continue;
      }
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
  if (out_extra_data) {
    *out_extra_data = NULL;
  }

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
          sizeof(iree_hal_streaming_node_block_t), capacity,
          sizeof(iree_hal_streaming_graph_node_t*), &block_size))) {
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

  const bool needs_node_block =
      !graph->current_node_block ||
      graph->current_node_block->count >= graph->current_node_block->capacity;
  const bool needs_root_block =
      node->dependency_count == 0 &&
      (!graph->current_root_block ||
       graph->current_root_block->count >= graph->current_root_block->capacity);

  // Reserve every block before publishing the node. Arena allocations cannot
  // be rolled back individually, but an allocation failure must leave all graph
  // topology and counters unchanged so a later insertion can proceed normally.
  iree_hal_streaming_node_block_t* new_node_block = NULL;
  iree_hal_streaming_node_block_t* new_root_block = NULL;
  if (needs_node_block) {
    const iree_host_size_t block_capacity =
        graph->node_count < 64 ? 16 : 64;  // Grow block size for larger graphs.
    IREE_RETURN_IF_ERROR(iree_hal_streaming_allocate_node_block(
        graph->arena_allocator, block_capacity, &new_node_block));
  }
  if (needs_root_block) {
    IREE_RETURN_IF_ERROR(iree_hal_streaming_allocate_node_block(
        graph->arena_allocator, /*capacity=*/8, &new_root_block));
  }

  // Assign the logical identity only after every fallible preparation step.
  node->graph = graph;
  node->node_index = (uint32_t)graph->node_count;
  node->clone_source_node_index = graph->next_clone_source_node_index++;
  node->debug_id = iree_atomic_fetch_add(&iree_hal_streaming_next_node_debug_id,
                                         1, iree_memory_order_relaxed);

  if (new_node_block) {
    if (graph->current_node_block) {
      graph->current_node_block->next = new_node_block;
    } else {
      graph->node_blocks = new_node_block;
    }
    graph->current_node_block = new_node_block;
  }

  graph->current_node_block->nodes[graph->current_node_block->count++] = node;
  ++graph->node_count;

  if (node->dependency_count == 0) {
    if (new_root_block) {
      if (graph->current_root_block) {
        graph->current_root_block->next = new_root_block;
      } else {
        graph->root_blocks = new_root_block;
      }
      graph->current_root_block = new_root_block;
    }
    graph->current_root_block->nodes[graph->current_root_block->count++] = node;
    ++graph->root_count;
  }

  return iree_ok_status();
}

typedef struct iree_hal_streaming_graph_clone_host_allocation_t {
  const iree_hal_streaming_graph_owned_host_allocation_t* source;
  iree_hal_streaming_graph_owned_host_allocation_t* clone;
} iree_hal_streaming_graph_clone_host_allocation_t;

typedef struct iree_hal_streaming_graph_clone_host_memcpy_prefix_t {
  void* dst;
  const void* src;
  iree_device_size_t count;
} iree_hal_streaming_graph_clone_host_memcpy_prefix_t;

static bool iree_hal_streaming_graph_clone_rewrite_owned_host_pointer(
    const iree_hal_streaming_graph_clone_host_allocation_t* allocations,
    iree_host_size_t allocation_count, const void* source_ptr,
    const void** out_clone_ptr) {
  *out_clone_ptr = source_ptr;
  const uintptr_t source_address = (uintptr_t)source_ptr;
  for (iree_host_size_t i = 0; i < allocation_count; ++i) {
    const iree_hal_streaming_graph_owned_host_allocation_t* source =
        allocations[i].source;
    iree_hal_streaming_graph_owned_host_allocation_t* clone =
        allocations[i].clone;
    const uintptr_t source_start = (uintptr_t)source->host_ptr;
    const uintptr_t source_end = source_start + source->size;
    if (source_address >= source_start && source_address < source_end) {
      *out_clone_ptr =
          (uint8_t*)clone->host_ptr + (source_address - source_start);
      return true;
    }
  }
  return false;
}

static bool iree_hal_streaming_graph_clone_rewrite_owned_buffer_ref(
    const iree_hal_streaming_graph_clone_host_allocation_t* allocations,
    iree_host_size_t allocation_count, iree_hal_streaming_buffer_ref_t* ref) {
  if (!ref || !ref->buffer) {
    return false;
  }
  for (iree_host_size_t i = 0; i < allocation_count; ++i) {
    if (ref->buffer == allocations[i].source->buffer) {
      ref->buffer = allocations[i].clone->buffer;
      return true;
    }
  }
  return false;
}

static iree_status_t iree_hal_streaming_graph_clone_host_allocations(
    iree_hal_streaming_graph_t* source_graph,
    iree_hal_streaming_graph_t* clone_graph,
    iree_hal_streaming_graph_clone_host_allocation_t** out_allocations,
    iree_host_size_t* out_allocation_count) {
  *out_allocations = NULL;
  *out_allocation_count = 0;

  iree_host_size_t allocation_count = 0;
  for (iree_hal_streaming_graph_owned_host_allocation_t* source =
           source_graph->owned_host_allocations;
       source; source = source->next) {
    ++allocation_count;
  }
  if (allocation_count == 0) {
    return iree_ok_status();
  }

  iree_host_size_t allocation_map_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          allocation_count, sizeof(**out_allocations), &allocation_map_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph clone host allocation map size overflow");
  }

  iree_hal_streaming_graph_clone_host_allocation_t* allocations = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      source_graph->host_allocator, allocation_map_size, (void**)&allocations));
  memset(allocations, 0, allocation_map_size);

  iree_status_t status = iree_ok_status();
  iree_host_size_t index = 0;
  for (iree_hal_streaming_graph_owned_host_allocation_t* source =
           source_graph->owned_host_allocations;
       iree_status_is_ok(status) && source; source = source->next, ++index) {
    iree_hal_streaming_buffer_t* clone_buffer = NULL;
    status = iree_hal_streaming_graph_allocate_host_staging(
        clone_graph, source->size, &clone_buffer);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (source->size > 0) {
      memcpy(clone_buffer->host_ptr, source->host_ptr, source->size);
    }
    allocations[index].source = source;
    allocations[index].clone = clone_graph->owned_host_allocations;
  }

  if (iree_status_is_ok(status)) {
    *out_allocations = allocations;
    *out_allocation_count = allocation_count;
  } else {
    iree_allocator_free(source_graph->host_allocator, allocations);
  }
  return status;
}

static iree_status_t iree_hal_streaming_graph_clone_impl(
    iree_hal_streaming_graph_t* source_graph,
    iree_hal_streaming_context_t* target_context, bool is_exec_template,
    iree_hal_streaming_graph_t** out_graph) {
  IREE_ASSERT_ARGUMENT(source_graph);
  IREE_ASSERT_ARGUMENT(target_context);
  IREE_ASSERT_ARGUMENT(out_graph);
  *out_graph = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (source_graph->has_graph_memory_nodes && !is_exec_template) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "cloning graphs with memory allocation nodes is not supported");
  }

  iree_hal_streaming_graph_t* clone_graph = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_create(target_context, source_graph->flags,
                                          source_graph->host_allocator,
                                          &clone_graph));
  if (is_exec_template) {
    iree_hal_streaming_live_graph_unregister(clone_graph);
    clone_graph->owns_graph_memory_node_claims = false;
  }
  clone_graph->has_graph_memory_nodes = source_graph->has_graph_memory_nodes;
  clone_graph->clone_source_graph_debug_id = source_graph->debug_id;
  if (!is_exec_template) {
    iree_hal_streaming_context_release(clone_graph->execution_context_hint);
    clone_graph->execution_context_hint = source_graph->execution_context_hint;
    iree_hal_streaming_context_retain(clone_graph->execution_context_hint);
  }

  iree_hal_streaming_graph_node_t** node_map = NULL;
  iree_hal_streaming_graph_clone_host_allocation_t* host_allocation_map = NULL;
  iree_host_size_t host_allocation_count = 0;
  iree_status_t status = iree_ok_status();
  status = iree_hal_streaming_graph_clone_host_allocations(
      source_graph, clone_graph, &host_allocation_map, &host_allocation_count);
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_graph_release(clone_graph);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
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
      if (host_allocation_map) {
        iree_allocator_free(source_graph->host_allocator, host_allocation_map);
      }
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
      iree_hal_streaming_graph_batch_mem_op_layout_t batch_mem_op_layout = {0};
      if (source_node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL) {
        iree_host_size_t bindings_byte_size = 0;
        if (IREE_UNLIKELY(
                !iree_host_size_checked_align(
                    source_node->attrs.kernel.constants_capacity,
                    iree_max_align_t, &constants_size) ||
                !iree_host_size_checked_mul(
                    source_node->attrs.kernel.binding_capacity,
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
      } else if (source_node->type ==
                 IREE_HAL_STREAMING_GRAPH_NODE_TYPE_BATCH_MEM_OP) {
        status = iree_hal_streaming_graph_batch_mem_op_layout_calculate(
            source_node->attrs.batch_mem_op.params_size,
            source_node->attrs.batch_mem_op.param_array_size,
            source_node->attrs.batch_mem_op.operation_count,
            &batch_mem_op_layout);
        if (!iree_status_is_ok(status)) {
          break;
        }
        extra_data_size = batch_mem_op_layout.total_size;
      }

      iree_hal_streaming_graph_node_t* clone_node = NULL;
      uint8_t* extra_data = NULL;
      status = iree_hal_streaming_graph_allocate_node(
          clone_graph->arena_allocator, source_node->dependency_count,
          extra_data_size, &clone_node, &extra_data);
      if (!iree_status_is_ok(status)) {
        break;
      }

      clone_node->type = source_node->type;
      clone_node->flags = source_node->flags;
      clone_node->dependency_count = source_node->dependency_count;
      clone_node->attrs = source_node->attrs;
      clone_node->graph = clone_graph;

      if (source_node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL) {
        iree_hal_streaming_module_retain(source_node->attrs.kernel.module);
        void* constants = extra_data_size ? extra_data : NULL;
        if (source_node->attrs.kernel.constants.data_length > 0) {
          memcpy(constants, source_node->attrs.kernel.constants.data,
                 source_node->attrs.kernel.constants.data_length);
        }
        clone_node->attrs.kernel.constants = iree_make_const_byte_span(
            constants, source_node->attrs.kernel.constants.data_length);
        clone_node->attrs.kernel.constants_capacity =
            source_node->attrs.kernel.constants_capacity;
        iree_hal_buffer_ref_t* bindings =
            extra_data_size
                ? (iree_hal_buffer_ref_t*)(extra_data + constants_size)
                : NULL;
        clone_node->attrs.kernel.bindings.values = bindings;
        clone_node->attrs.kernel.binding_capacity =
            source_node->attrs.kernel.binding_capacity;
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
      } else if (source_node->type ==
                 IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY) {
        iree_hal_streaming_graph_clone_rewrite_owned_buffer_ref(
            host_allocation_map, host_allocation_count,
            &clone_node->attrs.memcpy.dst_ref);
        iree_hal_streaming_graph_clone_rewrite_owned_buffer_ref(
            host_allocation_map, host_allocation_count,
            &clone_node->attrs.memcpy.src_ref);
        status = iree_hal_streaming_graph_memcpy_node_set_buffer_refs(
            clone_node, clone_node->attrs.memcpy.dst_ref,
            clone_node->attrs.memcpy.src_ref);
        if (!iree_status_is_ok(status)) {
          break;
        }
      } else if (source_node->type ==
                     IREE_HAL_STREAMING_GRAPH_NODE_TYPE_HOST_CALL &&
                 (source_node->flags &
                  IREE_HAL_STREAMING_GRAPH_NODE_FLAG_HIDDEN) != 0 &&
                 source_node->attrs.host.user_data &&
                 source_node->attrs.host.user_data_size > 0) {
        void* clone_data = NULL;
        status = iree_arena_allocate(&clone_graph->arena,
                                     source_node->attrs.host.user_data_size,
                                     (void**)&clone_data);
        if (!iree_status_is_ok(status)) {
          break;
        }
        memcpy(clone_data, source_node->attrs.host.user_data,
               source_node->attrs.host.user_data_size);

        // Graph-owned hidden host callbacks currently use a memcpy-compatible
        // prefix. Preserve the rest of the callback-private payload verbatim
        // while remapping staged host pointers into the clone's allocations.
        if (source_node->attrs.host.user_data_size >=
            sizeof(iree_hal_streaming_graph_clone_host_memcpy_prefix_t)) {
          iree_hal_streaming_graph_clone_host_memcpy_prefix_t* source_data =
              (iree_hal_streaming_graph_clone_host_memcpy_prefix_t*)
                  source_node->attrs.host.user_data;
          iree_hal_streaming_graph_clone_host_memcpy_prefix_t* clone_prefix =
              (iree_hal_streaming_graph_clone_host_memcpy_prefix_t*)clone_data;
          const void* clone_ptr = NULL;
          if (iree_hal_streaming_graph_clone_rewrite_owned_host_pointer(
                  host_allocation_map, host_allocation_count, source_data->dst,
                  &clone_ptr)) {
            clone_prefix->dst = (void*)clone_ptr;
          }
          if (iree_hal_streaming_graph_clone_rewrite_owned_host_pointer(
                  host_allocation_map, host_allocation_count, source_data->src,
                  &clone_ptr)) {
            clone_prefix->src = clone_ptr;
          }
        }
        clone_node->attrs.host.user_data = clone_data;
      } else if (source_node->type ==
                     IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC &&
                 source_node->attrs.mem_alloc.allocation) {
        iree_hal_streaming_graph_memory_allocation_retain(
            source_node->attrs.mem_alloc.allocation);
      } else if (source_node->type ==
                     IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_FREE &&
                 source_node->attrs.mem_free.allocation) {
        iree_hal_streaming_graph_memory_allocation_retain(
            source_node->attrs.mem_free.allocation);
      } else if (source_node->type ==
                 IREE_HAL_STREAMING_GRAPH_NODE_TYPE_BATCH_MEM_OP) {
        const iree_hal_streaming_graph_batch_mem_op_node_attrs_t* source_attrs =
            &source_node->attrs.batch_mem_op;
        iree_hal_streaming_graph_batch_mem_op_node_attrs_t* clone_attrs =
            &clone_node->attrs.batch_mem_op;
        clone_attrs->params = source_attrs->params_size > 0 ? extra_data : NULL;
        clone_attrs->params_size = source_attrs->params_size;
        clone_attrs->params_capacity = batch_mem_op_layout.params_capacity;
        if (source_attrs->params_size > 0) {
          memcpy(clone_attrs->params, source_attrs->params,
                 source_attrs->params_size);
        }
        clone_attrs->param_array =
            source_attrs->param_array_size > 0
                ? extra_data + batch_mem_op_layout.param_array_offset
                : NULL;
        clone_attrs->param_array_size = source_attrs->param_array_size;
        clone_attrs->param_array_capacity =
            batch_mem_op_layout.param_array_capacity;
        if (source_attrs->param_array_size > 0) {
          memcpy(clone_attrs->param_array, source_attrs->param_array,
                 source_attrs->param_array_size);
        }
        clone_attrs->operations =
            (iree_hal_streaming_value_operation_t*)(extra_data +
                                                    batch_mem_op_layout
                                                        .operations_offset);
        clone_attrs->operation_count = source_attrs->operation_count;
        clone_attrs->operation_capacity = source_attrs->operation_count;
        clone_attrs->owners =
            (hrx_buffer_t*)(extra_data + batch_mem_op_layout.owners_offset);
        memcpy(
            clone_attrs->operations, source_attrs->operations,
            source_attrs->operation_count * sizeof(*clone_attrs->operations));
        memcpy(clone_attrs->owners, source_attrs->owners,
               source_attrs->operation_count * sizeof(*clone_attrs->owners));
        for (iree_host_size_t j = 0; j < clone_attrs->operation_count; ++j) {
          iree_hal_buffer_retain(clone_attrs->operations[j].target_buffer);
          hrx_buffer_retain(clone_attrs->owners[j]);
        }
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

  for (iree_hal_streaming_graph_user_object_ref_t* source_ref =
           is_exec_template ? NULL : source_graph->user_object_refs;
       iree_status_is_ok(status) && source_ref; source_ref = source_ref->next) {
    iree_hal_streaming_graph_user_object_ref_t* clone_ref = NULL;
    status = iree_arena_allocate(&clone_graph->arena, sizeof(*clone_ref),
                                 (void**)&clone_ref);
    if (!iree_status_is_ok(status)) {
      break;
    }
    clone_ref->object = source_ref->object;
    clone_ref->count = source_ref->count;
    clone_ref->retain = source_ref->retain;
    clone_ref->release = source_ref->release;
    clone_ref->next = NULL;
    if (clone_ref->count > 0) {
      status = clone_ref->retain(clone_ref->object, clone_ref->count);
    }
    if (iree_status_is_ok(status)) {
      clone_ref->next = clone_graph->user_object_refs;
      clone_graph->user_object_refs = clone_ref;
    }
  }

  if (node_map) {
    iree_allocator_free(source_graph->host_allocator, node_map);
  }
  if (host_allocation_map) {
    iree_allocator_free(source_graph->host_allocator, host_allocation_map);
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

iree_status_t iree_hal_streaming_graph_clone(
    iree_hal_streaming_graph_t* source_graph,
    iree_hal_streaming_context_t* target_context,
    iree_hal_streaming_graph_t** out_graph) {
  return iree_hal_streaming_graph_clone_impl(
      source_graph, target_context, /*is_exec_template=*/false, out_graph);
}

iree_status_t iree_hal_streaming_graph_clone_for_exec(
    iree_hal_streaming_graph_t* source_graph,
    iree_hal_streaming_context_t* target_context,
    iree_hal_streaming_graph_t** out_graph) {
  IREE_ASSERT_ARGUMENT(source_graph);
  IREE_ASSERT_ARGUMENT(target_context);
  return iree_hal_streaming_graph_clone_impl(
      source_graph, target_context, /*is_exec_template=*/true, out_graph);
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

// Copies a caller-owned native kernarg byte image into graph-owned storage.
// The declared byte length is authoritative: zero is a valid empty span and
// must never be replaced with reflected metadata before copying or dispatch.
static iree_status_t iree_hal_streaming_graph_copy_prepacked_arguments(
    const iree_hal_streaming_dispatch_params_t* params,
    iree_host_size_t destination_capacity, void* destination,
    iree_const_byte_span_t* out_arguments) {
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(out_arguments);
  if (params->buffer_size > destination_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "pre-packed kernel arguments exceed graph storage");
  }
  if (params->buffer_size > 0 && !params->buffer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pre-packed kernel arguments require storage when length is non-zero");
  }
  if (params->buffer_size > 0) {
    memcpy(destination, params->buffer, params->buffer_size);
  }
  *out_arguments = iree_make_const_byte_span(destination, params->buffer_size);
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
  const bool is_empty_native_kernel =
      is_native_kernel &&
      iree_hal_streaming_parameter_info_is_empty(&symbol->parameters);
  if (is_args_array && is_native_kernel && !is_empty_native_kernel) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "non-empty args-array graph kernel launch requires parameter metadata");
  }
  if (is_pre_packed) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_streaming_validate_prepacked_kernel_arguments(symbol, params));
  }

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
  attrs->hip_function = params->binding_function;
  attrs->symbol = symbol;
  attrs->module = symbol->module;
  memcpy(attrs->grid_dim, params->grid_dim, sizeof(params->grid_dim));
  memcpy(attrs->block_dim, params->block_dim, sizeof(params->block_dim));
  memcpy(attrs->workitem_count, params->workitem_count,
         sizeof(params->workitem_count));
  attrs->shared_memory_bytes = params->shared_memory_bytes;
  attrs->cooperative = iree_any_bit_set(
      params->flags, IREE_HAL_STREAMING_DISPATCH_FLAG_COOPERATIVE);

  // Capture native kernarg bytes. HIP pointers can be stored anywhere in the
  // argument payload, so graph nodes keep the byte image instead of retaining a
  // HAL binding list for only the reflected pointer slots.
  void* constants = extra_data;
  if (constants_capacity > 0) {
    memset(constants, 0, constants_capacity);
  }
  attrs->constants =
      iree_make_const_byte_span(constants, symbol->parameters.constant_bytes);
  attrs->constants_capacity = constants_capacity;
  attrs->bindings.count = symbol->parameters.binding_count;
  attrs->bindings.values =
      symbol->parameters.binding_count
          ? (iree_hal_buffer_ref_t*)(extra_data + constants_size)
          : NULL;
  attrs->binding_capacity = symbol->parameters.binding_count;
  iree_status_t unpack_status = iree_ok_status();
  if (is_pre_packed) {
    unpack_status = iree_hal_streaming_graph_copy_prepacked_arguments(
        params, constants_capacity, constants, &attrs->constants);
    attrs->bindings.count = 0;
  } else if (is_args_array && is_empty_native_kernel) {
    // HIP host stubs may pass a {NULL} args array for no-argument kernels.
    attrs->constants = iree_make_const_byte_span(constants, 0);
    attrs->bindings.count = 0;
  } else if (is_args_array) {
    iree_host_size_t captured_size = 0;
    unpack_status = iree_hal_streaming_pack_raw_argument_list(
        &symbol->parameters, (void**)params->buffer, constants, &captured_size);
    attrs->constants = iree_make_const_byte_span(constants, captured_size);
    attrs->bindings.count = 0;
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
    attrs->bindings.count = 0;
  } else if (is_empty_native_kernel) {
    const iree_host_size_t captured_size =
        symbol->parameters.direct_arg_bytes
            ? symbol->parameters.direct_arg_bytes
            : symbol->parameters.constant_bytes;
    attrs->constants = iree_make_const_byte_span(constants, captured_size);
    attrs->bindings.count = 0;
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
      attrs->bindings.count = 0;
      unpack_status = iree_ok_status();
    }
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, unpack_status);

  // The node does not own the module until all argument processing succeeds.
  // This keeps arena-allocated but unlinked nodes from leaking module
  // ownership on malformed argument lists.
  iree_hal_streaming_module_retain(attrs->module);
  iree_status_t status = iree_hal_streaming_graph_add_node(graph, node);
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_module_release(attrs->module);
    attrs->module = NULL;
    attrs->symbol = NULL;
  }
  if (iree_status_is_ok(status) && out_node) {
    *out_node = node;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_graph_set_kernel_node_params(
    iree_hal_streaming_graph_node_t* node, iree_hal_streaming_symbol_t* symbol,
    const iree_hal_streaming_dispatch_params_t* params) {
  IREE_ASSERT_ARGUMENT(node);
  IREE_ASSERT_ARGUMENT(symbol);
  IREE_ASSERT_ARGUMENT(params);
  if (!node->graph || node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL ||
      symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }

  const bool is_pre_packed =
      (params->flags & IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED) != 0;
  const bool is_args_array =
      (params->flags & IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY) != 0;
  const bool is_native_kernel = symbol->parameters.binding_count == 0 &&
                                symbol->parameters.copy_count == 0;
  const bool is_empty_native_kernel =
      is_native_kernel &&
      iree_hal_streaming_parameter_info_is_empty(&symbol->parameters);
  if (is_args_array && is_native_kernel && !is_empty_native_kernel) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "non-empty args-array graph kernel launch requires parameter metadata");
  }
  if (is_pre_packed) {
    IREE_RETURN_IF_ERROR(
        iree_hal_streaming_validate_prepacked_kernel_arguments(symbol, params));
  }

  iree_host_size_t constants_capacity = symbol->parameters.constant_bytes;
  if (params->buffer_size > constants_capacity) {
    constants_capacity = params->buffer_size;
  }
  if ((is_args_array || is_native_kernel) &&
      symbol->parameters.direct_arg_bytes > constants_capacity) {
    constants_capacity = symbol->parameters.direct_arg_bytes;
  }
  if (constants_capacity > node->attrs.kernel.constants_capacity ||
      symbol->parameters.binding_count > node->attrs.kernel.binding_capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }

  iree_hal_streaming_graph_kernel_node_attrs_t* attrs = &node->attrs.kernel;
  const iree_host_size_t constants_storage_capacity = attrs->constants_capacity;
  iree_hal_buffer_ref_t* binding_storage =
      (iree_hal_buffer_ref_t*)attrs->bindings.values;
  iree_host_size_t temporary_constants_size = 0;
  iree_host_size_t temporary_bindings_size = 0;
  iree_host_size_t temporary_storage_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_align(constants_storage_capacity,
                                                  iree_max_align_t,
                                                  &temporary_constants_size) ||
                    !iree_host_size_checked_mul(attrs->binding_capacity,
                                                sizeof(iree_hal_buffer_ref_t),
                                                &temporary_bindings_size) ||
                    !iree_host_size_checked_add(temporary_constants_size,
                                                temporary_bindings_size,
                                                &temporary_storage_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph kernel parameter storage overflow");
  }

  // Graph parameter updates are cold operations. Keep the replacement image in
  // temporary host storage so a failed update cannot partially change the node
  // or consume unbounded stack space for a large reflected binding list.
  uint8_t* temporary_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(node->graph->host_allocator,
                                             temporary_storage_size,
                                             (void**)&temporary_storage));
  uint8_t* temporary_constants = temporary_storage;
  iree_hal_buffer_ref_t* temporary_binding_values =
      attrs->binding_capacity
          ? (iree_hal_buffer_ref_t*)(temporary_storage +
                                     temporary_constants_size)
          : NULL;
  if (temporary_storage_size > 0) {
    memset(temporary_storage, 0, temporary_storage_size);
  }
  iree_hal_buffer_ref_list_t bindings = {
      .count = symbol->parameters.binding_count,
      .values = temporary_binding_values,
  };

  iree_const_byte_span_t constants_span = iree_make_const_byte_span(
      temporary_constants, symbol->parameters.constant_bytes);
  iree_status_t unpack_status = iree_ok_status();
  if (is_pre_packed) {
    unpack_status = iree_hal_streaming_graph_copy_prepacked_arguments(
        params, constants_storage_capacity, temporary_constants,
        &constants_span);
    bindings = iree_hal_buffer_ref_list_empty();
  } else if (is_args_array && is_empty_native_kernel) {
    // HIP host stubs may pass a {NULL} args array for no-argument kernels.
    constants_span = iree_make_const_byte_span(temporary_constants, 0);
    bindings = iree_hal_buffer_ref_list_empty();
  } else if (is_args_array) {
    iree_host_size_t captured_size = 0;
    unpack_status = iree_hal_streaming_pack_raw_argument_list(
        &symbol->parameters, (void**)params->buffer, temporary_constants,
        &captured_size);
    constants_span =
        iree_make_const_byte_span(temporary_constants, captured_size);
    bindings = iree_hal_buffer_ref_list_empty();
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
      memcpy(temporary_constants, params->buffer, copy_size);
    }
    constants_span =
        iree_make_const_byte_span(temporary_constants, captured_size);
    bindings = iree_hal_buffer_ref_list_empty();
  } else if (is_empty_native_kernel) {
    const iree_host_size_t captured_size =
        symbol->parameters.direct_arg_bytes
            ? symbol->parameters.direct_arg_bytes
            : symbol->parameters.constant_bytes;
    constants_span =
        iree_make_const_byte_span(temporary_constants, captured_size);
    bindings = iree_hal_buffer_ref_list_empty();
  } else {
    unpack_status = iree_hal_streaming_unpack_parameters(
        node->graph->context, &symbol->parameters, params->buffer,
        temporary_constants, &bindings);
    if (iree_status_code(unpack_status) == IREE_STATUS_NOT_FOUND) {
      iree_status_ignore(unpack_status);
      const iree_host_size_t captured_size =
          params->buffer_size ? params->buffer_size : constants_capacity;
      if (captured_size > 0) {
        memcpy(temporary_constants, params->buffer, captured_size);
      }
      constants_span =
          iree_make_const_byte_span(temporary_constants, captured_size);
      bindings = iree_hal_buffer_ref_list_empty();
      unpack_status = iree_ok_status();
    }
  }
  if (!iree_status_is_ok(unpack_status)) {
    iree_allocator_free(node->graph->host_allocator, temporary_storage);
    return unpack_status;
  }

  // Commit only after every source pointer, argument range, and binding has
  // been validated. Failed graph updates must leave the existing node byte
  // image and binding list unchanged.
  if (constants_storage_capacity > 0) {
    memcpy((void*)attrs->constants.data, temporary_constants,
           constants_storage_capacity);
  }
  if (attrs->binding_capacity > 0) {
    memcpy(binding_storage, temporary_binding_values,
           attrs->binding_capacity * sizeof(*temporary_binding_values));
  }
  iree_hal_streaming_module_t* previous_module = attrs->module;
  attrs->hip_function = params->binding_function;
  attrs->symbol = symbol;
  attrs->module = symbol->module;
  iree_hal_streaming_module_retain(attrs->module);
  memcpy(attrs->grid_dim, params->grid_dim, sizeof(params->grid_dim));
  memcpy(attrs->block_dim, params->block_dim, sizeof(params->block_dim));
  memcpy(attrs->workitem_count, params->workitem_count,
         sizeof(params->workitem_count));
  attrs->shared_memory_bytes = params->shared_memory_bytes;
  attrs->cooperative = iree_any_bit_set(
      params->flags, IREE_HAL_STREAMING_DISPATCH_FLAG_COOPERATIVE);
  attrs->constants = iree_make_const_byte_span(attrs->constants.data,
                                               constants_span.data_length);
  attrs->bindings = (iree_hal_buffer_ref_list_t){
      .count = bindings.count,
      .values = binding_storage,
  };
  iree_hal_streaming_module_release(previous_module);
  iree_allocator_free(node->graph->host_allocator, temporary_storage);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_import_copy_operand(
    iree_hal_streaming_graph_t* graph, iree_hal_streaming_buffer_ref_t ref,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  if (!ref.buffer || !ref.buffer->context) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "graph copy operand has no owning allocation");
  }

  bool allow_peer_device_allocation = false;
  if (ref.buffer->context->device_ordinal != graph->context->device_ordinal) {
    IREE_RETURN_IF_ERROR(iree_hal_streaming_device_can_access_peer(
        graph->context->device_ordinal, ref.buffer->context->device_ordinal,
        &allow_peer_device_allocation));
    if (!allow_peer_device_allocation) {
      return iree_make_status(
          IREE_STATUS_PERMISSION_DENIED,
          "graph device %zu cannot access allocation device %zu",
          graph->context->device_ordinal, ref.buffer->context->device_ordinal);
    }
  }
  return iree_hal_streaming_memory_buffer_for_context(
      graph->context, ref.buffer, allow_peer_device_allocation, out_buffer);
}

iree_status_t iree_hal_streaming_graph_memcpy_node_set_buffer_refs(
    iree_hal_streaming_graph_node_t* node,
    iree_hal_streaming_buffer_ref_t dst_ref,
    iree_hal_streaming_buffer_ref_t src_ref) {
  IREE_ASSERT_ARGUMENT(node);
  if (!node->graph || node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "node is not a graph memcpy node");
  }

  iree_hal_buffer_t* execution_dst_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_import_copy_operand(
      node->graph, dst_ref, &execution_dst_buffer));
  iree_hal_buffer_t* execution_src_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_import_copy_operand(
      node->graph, src_ref, &execution_src_buffer));

  node->attrs.memcpy.dst_ref = dst_ref;
  node->attrs.memcpy.src_ref = src_ref;
  node->attrs.memcpy.execution_dst_buffer = execution_dst_buffer;
  node->attrs.memcpy.execution_src_buffer = execution_src_buffer;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_add_copy_buffer_node_resolved(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, iree_hal_streaming_buffer_ref_t dst_ref,
    iree_hal_streaming_buffer_ref_t src_ref, void* hip_dst, const void* hip_src,
    iree_host_size_t size, iree_hal_streaming_graph_node_t** out_node) {
  iree_hal_buffer_t* execution_dst_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_import_copy_operand(
      graph, dst_ref, &execution_dst_buffer));
  iree_hal_buffer_t* execution_src_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_import_copy_operand(
      graph, src_ref, &execution_src_buffer));

  // Allocate node with dependencies in a single allocation.
  iree_hal_streaming_graph_node_t* node = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_allocate_node(
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
  attrs->execution_dst_buffer = execution_dst_buffer;
  attrs->execution_src_buffer = execution_src_buffer;
  attrs->size = size;
  attrs->execution_dst_pitch = size;
  attrs->execution_src_pitch = size;
  attrs->execution_dst_ysize = 1;
  attrs->execution_src_ysize = 1;
  attrs->execution_extent_width = size;
  attrs->execution_extent_height = 1;
  attrs->execution_extent_depth = 1;
  attrs->hip_dst = hip_dst;
  attrs->hip_src = hip_src;
  attrs->hip_dst_position_x = 0;
  attrs->hip_dst_position_y = 0;
  attrs->hip_dst_position_z = 0;
  attrs->hip_src_position_x = 0;
  attrs->hip_src_position_y = 0;
  attrs->hip_src_position_z = 0;
  attrs->hip_dst_pitch = size;
  attrs->hip_src_pitch = size;
  attrs->hip_dst_xsize = size;
  attrs->hip_src_xsize = size;
  attrs->hip_dst_ysize = 1;
  attrs->hip_src_ysize = 1;
  attrs->hip_extent_width = size;
  attrs->hip_extent_height = 1;
  attrs->hip_extent_depth = 1;
  attrs->hip_kind = 3;

  iree_status_t status = iree_hal_streaming_graph_add_node(graph, node);
  if (iree_status_is_ok(status) && out_node) {
    *out_node = node;
  }
  return status;
}

static iree_status_t iree_hal_streaming_graph_resolve_copy_ptr(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_deviceptr_t device_ptr, iree_device_size_t size,
    iree_hal_streaming_buffer_ref_t* out_ref) {
  iree_status_t status = iree_hal_streaming_memory_lookup_range(
      graph->context, device_ptr, size, out_ref);
  if (!iree_status_is_ok(status) &&
      iree_status_code(status) == IREE_STATUS_NOT_FOUND) {
    iree_status_ignore(status);
    iree_hal_streaming_context_t* owner_context = NULL;
    status = iree_hal_streaming_memory_lookup_range_across_contexts(
        device_ptr, size, &owner_context, out_ref);
    iree_hal_streaming_context_release(owner_context);
  }
  return status;
}

static iree_status_t iree_hal_streaming_graph_resolve_copy_ptrs(
    iree_hal_streaming_graph_t* graph, iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_buffer_ref_t* out_dst_ref,
    iree_hal_streaming_buffer_ref_t* out_src_ref) {
  if (size == 0) {
    IREE_RETURN_IF_ERROR(
        iree_hal_streaming_memory_lookup(graph->context, dst, out_dst_ref),
        "resolving `dst` buffer ref %p", (void*)dst);
    IREE_RETURN_IF_ERROR(
        iree_hal_streaming_memory_lookup(graph->context, src, out_src_ref),
        "resolving `src` buffer ref %p", (void*)src);
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_graph_resolve_copy_ptr(graph, dst, size, out_dst_ref),
      "resolving `dst` buffer ref %p with size %" PRIhsz, (void*)dst, size);
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_graph_resolve_copy_ptr(graph, src, size, out_src_ref),
      "resolving `src` buffer ref %p with size %" PRIhsz, (void*)src, size);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_graph_add_copy_ptr_node(
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
  iree_hal_streaming_buffer_ref_t src_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_resolve_copy_ptrs(graph, dst, src, size,
                                                     &dst_ref, &src_ref));

  iree_status_t status = iree_hal_streaming_graph_add_copy_buffer_node_resolved(
      graph, dependencies, dependency_count, dst_ref, src_ref, (void*)dst,
      (const void*)src, size, out_node);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_graph_add_copy_buffer_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, iree_hal_streaming_buffer_ref_t dst_ref,
    iree_hal_streaming_buffer_ref_t src_ref, iree_device_size_t size,
    iree_hal_streaming_graph_node_t** out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_validate_dependencies(graph, dependencies,
                                                         dependency_count));
  iree_status_t status = iree_hal_streaming_graph_add_copy_buffer_node_resolved(
      graph, dependencies, dependency_count, dst_ref, src_ref, NULL, NULL, size,
      out_node);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t
iree_hal_streaming_graph_add_copy_buffer_node_with_extra_dependency(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count,
    iree_hal_streaming_graph_node_t* extra_dependency,
    iree_hal_streaming_buffer_ref_t dst_ref,
    iree_hal_streaming_buffer_ref_t src_ref, iree_device_size_t size,
    iree_hal_streaming_graph_node_t** out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  if (!extra_dependency) {
    return iree_hal_streaming_graph_add_copy_buffer_node(
        graph, dependencies, dependency_count, dst_ref, src_ref, size,
        out_node);
  }
  if (dependency_count > 0 && !dependencies) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dependency array must be provided");
  }
  for (iree_host_size_t i = 0; i < dependency_count; ++i) {
    if (dependencies[i] == extra_dependency) {
      return iree_hal_streaming_graph_add_copy_buffer_node(
          graph, dependencies, dependency_count, dst_ref, src_ref, size,
          out_node);
    }
  }

  iree_host_size_t total_count = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_add(dependency_count, 1, &total_count))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph dependency count overflow");
  }
  if (dependency_count == 0) {
    return iree_hal_streaming_graph_add_copy_buffer_node(
        graph, &extra_dependency, 1, dst_ref, src_ref, size, out_node);
  }
  iree_host_size_t dependency_list_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          total_count, sizeof(*dependencies), &dependency_list_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph dependency list size overflow");
  }

  iree_hal_streaming_graph_node_t** merged_dependencies = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(graph->host_allocator,
                                             dependency_list_size,
                                             (void**)&merged_dependencies));
  memcpy(merged_dependencies, dependencies,
         dependency_count * sizeof(*dependencies));
  merged_dependencies[dependency_count] = extra_dependency;

  iree_status_t status = iree_hal_streaming_graph_add_copy_buffer_node(
      graph, merged_dependencies, total_count, dst_ref, src_ref, size,
      out_node);
  iree_allocator_free(graph->host_allocator, merged_dependencies);
  return status;
}

iree_status_t iree_hal_streaming_graph_add_copy_ptr_node_with_extra_dependency(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count,
    iree_hal_streaming_graph_node_t* extra_dependency,
    iree_hal_streaming_deviceptr_t dst, iree_hal_streaming_deviceptr_t src,
    iree_device_size_t size, iree_hal_streaming_graph_node_t** out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  iree_hal_streaming_buffer_ref_t dst_ref;
  iree_hal_streaming_buffer_ref_t src_ref;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_resolve_copy_ptrs(
      graph, dst, src, size, &dst_ref, &src_ref));
  return iree_hal_streaming_graph_add_copy_buffer_node_with_extra_dependency(
      graph, dependencies, dependency_count, extra_dependency, dst_ref, src_ref,
      size, out_node);
}

iree_status_t iree_hal_streaming_graph_add_fill_ptr_node(
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
        "resolving `dst` buffer ref %p with size %" PRIdsz, (void*)dst,
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
  attrs->hip_dst = (void*)dst;
  attrs->hip_width = count;
  attrs->hip_height = 1;
  attrs->hip_pitch = 0;

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
  attrs->user_data_size = 0;

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
    if (out_node) {
      *out_node = node;
    }
  } else {
    iree_hal_streaming_graph_node_deinitialize_attrs(node);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_graph_add_batch_mem_op_node(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, const void* params,
    iree_host_size_t params_size, const void* param_array,
    iree_host_size_t param_array_size,
    const iree_hal_streaming_value_operation_t* operations,
    const hrx_buffer_t* owners, iree_host_size_t operation_count,
    iree_hal_streaming_graph_node_t** out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(out_node);
  *out_node = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  if ((params_size > 0 && !params) || (param_array_size > 0 && !param_array)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "batch mem op payload must be provided");
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_validate_dependencies(graph, dependencies,
                                                         dependency_count));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_validate_batch_mem_operations(
              operations, owners, operation_count));

  iree_hal_streaming_graph_batch_mem_op_layout_t layout;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_batch_mem_op_layout_calculate(
              params_size, param_array_size, operation_count, &layout));

  iree_hal_streaming_graph_node_t* node = NULL;
  uint8_t* extra_data = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_allocate_node(
              graph->arena_allocator, dependency_count, layout.total_size,
              &node, &extra_data));

  node->type = IREE_HAL_STREAMING_GRAPH_NODE_TYPE_BATCH_MEM_OP;
  node->dependency_count = dependency_count;
  if (dependency_count > 0) {
    memcpy(node->dependencies, dependencies,
           dependency_count * sizeof(*dependencies));
  }

  iree_hal_streaming_graph_batch_mem_op_node_attrs_t* attrs =
      &node->attrs.batch_mem_op;
  attrs->params = params_size > 0 ? extra_data : NULL;
  attrs->params_size = params_size;
  attrs->params_capacity = layout.params_capacity;
  if (params_size > 0) {
    memcpy(attrs->params, params, params_size);
  }
  attrs->param_array =
      param_array_size > 0 ? extra_data + layout.param_array_offset : NULL;
  attrs->param_array_size = param_array_size;
  attrs->param_array_capacity = layout.param_array_capacity;
  if (param_array_size > 0) {
    memcpy(attrs->param_array, param_array, param_array_size);
  }
  attrs->operations =
      (iree_hal_streaming_value_operation_t*)(extra_data +
                                              layout.operations_offset);
  attrs->operation_count = operation_count;
  attrs->operation_capacity = operation_count;
  attrs->owners = (hrx_buffer_t*)(extra_data + layout.owners_offset);
  memcpy(attrs->operations, operations,
         operation_count * sizeof(*attrs->operations));
  memcpy(attrs->owners, owners, operation_count * sizeof(*attrs->owners));
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    iree_hal_buffer_retain(attrs->operations[i].target_buffer);
    hrx_buffer_retain(attrs->owners[i]);
  }

  iree_status_t status = iree_hal_streaming_graph_add_node(graph, node);
  if (iree_status_is_ok(status)) {
    *out_node = node;
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
    if (out_node) {
      *out_node = node;
    }
  } else {
    iree_hal_streaming_graph_node_deinitialize_attrs(node);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_graph_set_batch_mem_op_node_params(
    iree_hal_streaming_graph_node_t* node, const void* params,
    iree_host_size_t params_size, const void* param_array,
    iree_host_size_t param_array_size,
    const iree_hal_streaming_value_operation_t* operations,
    const hrx_buffer_t* owners, iree_host_size_t operation_count) {
  if (!node || !node->graph ||
      node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_BATCH_MEM_OP) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "node must be a batch mem op node");
  }
  if ((params_size > 0 && !params) || (param_array_size > 0 && !param_array)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "batch mem op payload must be provided");
  }
  IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_validate_batch_mem_operations(
      operations, owners, operation_count));

  iree_hal_streaming_graph_batch_mem_op_node_attrs_t* attrs =
      &node->attrs.batch_mem_op;
  void* params_storage = attrs->params;
  void* param_array_storage = attrs->param_array;
  iree_hal_streaming_value_operation_t* operation_storage = attrs->operations;
  hrx_buffer_t* owner_storage = attrs->owners;
  iree_host_size_t params_capacity = attrs->params_capacity;
  iree_host_size_t param_array_capacity = attrs->param_array_capacity;
  iree_host_size_t operation_capacity = attrs->operation_capacity;
  if (params_size > params_capacity ||
      param_array_size > param_array_capacity ||
      operation_count > operation_capacity) {
    iree_hal_streaming_graph_batch_mem_op_layout_t layout;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_batch_mem_op_layout_calculate(
        params_size, param_array_size, operation_count, &layout));
    uint8_t* storage = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        &node->graph->arena, layout.total_size, (void**)&storage));
    params_storage = params_size > 0 ? storage : NULL;
    param_array_storage =
        param_array_size > 0 ? storage + layout.param_array_offset : NULL;
    operation_storage =
        (iree_hal_streaming_value_operation_t*)(storage +
                                                layout.operations_offset);
    owner_storage = (hrx_buffer_t*)(storage + layout.owners_offset);
    params_capacity = layout.params_capacity;
    param_array_capacity = layout.param_array_capacity;
    operation_capacity = operation_count;
  }
  // Acquire the complete replacement ownership before releasing the current
  // operation array. This also permits callers to set a node from its own
  // graph-owned query payload.
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    iree_hal_buffer_retain(operations[i].target_buffer);
    hrx_buffer_retain(owners[i]);
  }
  for (iree_host_size_t i = 0; i < attrs->operation_count; ++i) {
    iree_hal_buffer_release(attrs->operations[i].target_buffer);
    hrx_buffer_release(attrs->owners[i]);
  }

  if (params_size > 0) {
    memmove(params_storage, params, params_size);
  }
  if (param_array_size > 0) {
    memmove(param_array_storage, param_array, param_array_size);
  }
  memmove(operation_storage, operations,
          operation_count * sizeof(*operation_storage));
  memmove(owner_storage, owners, operation_count * sizeof(*owner_storage));

  attrs->params = params_storage;
  attrs->params_size = params_size;
  attrs->params_capacity = params_capacity;
  attrs->param_array = param_array_storage;
  attrs->param_array_size = param_array_size;
  attrs->param_array_capacity = param_array_capacity;
  attrs->operations = operation_storage;
  attrs->operation_count = operation_count;
  attrs->operation_capacity = operation_capacity;
  attrs->owners = owner_storage;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_graph_add_dependencies(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** from_nodes,
    iree_hal_streaming_graph_node_t** to_nodes, iree_host_size_t count) {
  IREE_ASSERT_ARGUMENT(graph);
  if (count == 0) {
    return iree_ok_status();
  }
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
  if (node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC ||
      node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_FREE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "destroying graph memory allocation and free nodes is not supported");
  }
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

static void iree_hal_streaming_graph_consider_execution_context(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_t** candidate_context, bool* is_ambiguous) {
  if (!context || *is_ambiguous) {
    return;
  }
  if (!*candidate_context) {
    *candidate_context = context;
  } else if (*candidate_context != context) {
    *is_ambiguous = true;
  }
}

static void iree_hal_streaming_graph_consider_buffer_context(
    const iree_hal_streaming_buffer_ref_t* ref,
    iree_hal_streaming_context_t** candidate_context, bool* is_ambiguous) {
  if (!ref || !ref->buffer || !ref->buffer->context ||
      iree_any_bit_set((iree_hal_memory_type_t)ref->buffer->memory_type,
                       IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    return;
  }
  iree_hal_streaming_graph_consider_execution_context(
      ref->buffer->context, candidate_context, is_ambiguous);
}

static iree_hal_streaming_context_t*
iree_hal_streaming_graph_infer_execution_context(
    iree_hal_streaming_graph_t* graph) {
  // Public clones may recreate graph-owned staging in another context without
  // changing where unchanged nodes execute. A unique context referenced by
  // explicit node resources takes precedence. Host-only and mixed-device
  // graphs retain the source graph's execution hint instead of guessing an
  // affinity from opaque argument contents.
  iree_hal_streaming_context_t* candidate_context = NULL;
  bool is_ambiguous = false;
  for (iree_hal_streaming_node_block_t* block = graph->node_blocks;
       block && !is_ambiguous; block = block->next) {
    for (iree_host_size_t i = 0; i < block->count && !is_ambiguous; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      switch (node->type) {
        case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL:
          if (node->attrs.kernel.symbol && node->attrs.kernel.symbol->module) {
            iree_hal_streaming_graph_consider_execution_context(
                node->attrs.kernel.symbol->module->context, &candidate_context,
                &is_ambiguous);
          }
          break;
        case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY:
          iree_hal_streaming_graph_consider_buffer_context(
              &node->attrs.memcpy.dst_ref, &candidate_context, &is_ambiguous);
          iree_hal_streaming_graph_consider_buffer_context(
              &node->attrs.memcpy.src_ref, &candidate_context, &is_ambiguous);
          break;
        case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET:
          iree_hal_streaming_graph_consider_buffer_context(
              &node->attrs.memset.dst_ref, &candidate_context, &is_ambiguous);
          break;
        case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC:
          iree_hal_streaming_graph_consider_execution_context(
              iree_hal_streaming_graph_memory_allocation_context(
                  node->attrs.mem_alloc.allocation),
              &candidate_context, &is_ambiguous);
          break;
        case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_FREE:
          iree_hal_streaming_graph_consider_execution_context(
              iree_hal_streaming_graph_memory_allocation_context(
                  node->attrs.mem_free.allocation),
              &candidate_context, &is_ambiguous);
          break;
        case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH:
          if (node->attrs.child_graph.graph) {
            iree_hal_streaming_graph_consider_execution_context(
                iree_hal_streaming_graph_infer_execution_context(
                    node->attrs.child_graph.graph),
                &candidate_context, &is_ambiguous);
          }
          break;
        case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT:
        case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD:
          if (node->attrs.event.event) {
            iree_hal_streaming_graph_consider_execution_context(
                node->attrs.event.event->context, &candidate_context,
                &is_ambiguous);
          }
          break;
        case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_BATCH_MEM_OP:
          is_ambiguous = true;
          break;
        default:
          break;
      }
    }
  }

  return !is_ambiguous && candidate_context ? candidate_context
                                            : graph->execution_context_hint;
}

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
  iree_hal_streaming_context_t* execution_context =
      iree_hal_streaming_graph_infer_execution_context(graph);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_exec_create(execution_context, graph, flags,
                                               graph->host_allocator, &exec));

  // Instantiate from the graph template. HIP graph objects are not internally
  // synchronized; callers must externally serialize access to a graph while it
  // is being modified, queried, or instantiated.
  iree_status_t status =
      iree_hal_streaming_graph_exec_instantiate_from_template(exec, exec);

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
    iree_hal_streaming_stream_t* stream, iree_host_size_t required_capacity) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, required_capacity);

  iree_host_size_t new_capacity = 0;
  iree_host_size_t allocation_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(required_capacity, 2, &new_capacity) ||
          !iree_host_size_checked_mul(new_capacity,
                                      sizeof(*stream->capture_dependencies),
                                      &allocation_size))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "capture dependency array allocation size overflow");
  }

  iree_status_t status =
      iree_allocator_realloc(stream->host_allocator, allocation_size,
                             (void**)&stream->capture_dependencies);
  if (iree_status_is_ok(status)) {
    stream->capture_dependency_capacity = new_capacity;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

typedef struct iree_hal_streaming_capture_transaction_t {
  iree_hal_streaming_stream_t* stream;
  iree_hal_streaming_graph_t* graph;
} iree_hal_streaming_capture_transaction_t;

// Enters the graph-wide capture transaction while obeying the global capture
// lock order: graph/session, then stream-list (when needed), then stream. A
// short stream-only snapshot is used to retain the graph before taking the
// graph mutex; the stream is then revalidated under both locks.
static iree_status_t iree_hal_streaming_capture_transaction_begin(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_capture_transaction_t* out_transaction,
    bool* out_was_capturing) {
  *out_transaction = (iree_hal_streaming_capture_transaction_t){0};
  *out_was_capturing = false;

  for (;;) {
    iree_slim_mutex_lock(&stream->mutex);
    if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
      iree_slim_mutex_unlock(&stream->mutex);
      return iree_ok_status();
    }
    *out_was_capturing = true;
    if (stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE ||
        !stream->capture_graph) {
      iree_slim_mutex_unlock(&stream->mutex);
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "stream capture has been invalidated");
    }
    iree_hal_streaming_graph_t* graph = stream->capture_graph;
    const unsigned long long capture_id = stream->capture_id;
    iree_hal_streaming_graph_retain(graph);
    iree_slim_mutex_unlock(&stream->mutex);

    iree_slim_mutex_lock(&graph->capture_mutex);
    iree_slim_mutex_lock(&stream->mutex);
    if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE ||
        stream->capture_graph != graph || stream->capture_id != capture_id) {
      iree_slim_mutex_unlock(&stream->mutex);
      iree_slim_mutex_unlock(&graph->capture_mutex);
      iree_hal_streaming_graph_release(graph);
      *out_was_capturing = false;
      continue;
    }

    const iree_hal_streaming_graph_capture_state_t capture_state =
        (iree_hal_streaming_graph_capture_state_t)iree_atomic_load(
            &graph->capture_state, iree_memory_order_acquire);
    if (stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE ||
        graph->capture_id != capture_id ||
        capture_state != IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_ACTIVE) {
      if (capture_state == IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_INVALIDATED) {
        iree_hal_streaming_stream_set_capture_status(
            stream, IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED);
      }
      iree_slim_mutex_unlock(&stream->mutex);
      iree_slim_mutex_unlock(&graph->capture_mutex);
      iree_hal_streaming_graph_release(graph);
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "stream capture has been invalidated");
    }

    out_transaction->stream = stream;
    out_transaction->graph = graph;
    return iree_ok_status();
  }
}

static void iree_hal_streaming_capture_transaction_unlock(
    iree_hal_streaming_capture_transaction_t* transaction) {
  iree_slim_mutex_unlock(&transaction->stream->mutex);
  iree_slim_mutex_unlock(&transaction->graph->capture_mutex);
  iree_hal_streaming_graph_release(transaction->graph);
  *transaction = (iree_hal_streaming_capture_transaction_t){0};
}

static void iree_hal_streaming_capture_transaction_invalidate(
    iree_hal_streaming_capture_transaction_t* transaction) {
  iree_atomic_store(&transaction->graph->capture_state,
                    IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_INVALIDATED,
                    iree_memory_order_release);
  iree_hal_streaming_stream_set_capture_status(
      transaction->stream, IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED);
}

bool iree_hal_streaming_capture_graph_invalidate(
    iree_hal_streaming_graph_t* graph, unsigned long long capture_id) {
  IREE_ASSERT_ARGUMENT(graph);
  bool invalidated = false;
  iree_slim_mutex_lock(&graph->capture_mutex);
  const int32_t capture_state =
      iree_atomic_load(&graph->capture_state, iree_memory_order_acquire);
  if (capture_id != 0 && graph->capture_id == capture_id &&
      (capture_state == IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_ACTIVE ||
       capture_state == IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_INVALIDATED)) {
    iree_atomic_store(&graph->capture_state,
                      IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_INVALIDATED,
                      iree_memory_order_release);
    invalidated = true;
  }
  iree_slim_mutex_unlock(&graph->capture_mutex);
  return invalidated;
}

iree_hal_streaming_capture_status_t iree_hal_streaming_capture_invalidate(
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);

  // Retain from a short stream-only snapshot, then follow graph -> stream.
  iree_slim_mutex_lock(&stream->mutex);
  if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE ||
      !stream->capture_graph) {
    iree_slim_mutex_unlock(&stream->mutex);
    return IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
  }
  iree_hal_streaming_graph_t* graph = stream->capture_graph;
  const unsigned long long capture_id = stream->capture_id;
  iree_hal_streaming_graph_retain(graph);
  iree_slim_mutex_unlock(&stream->mutex);

  iree_hal_streaming_capture_status_t observed_status =
      IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
  iree_slim_mutex_lock(&graph->capture_mutex);
  iree_slim_mutex_lock(&stream->mutex);
  if (stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE &&
      stream->capture_graph == graph && stream->capture_id == capture_id) {
    const int32_t capture_state =
        iree_atomic_load(&graph->capture_state, iree_memory_order_acquire);
    if (capture_state == IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_ACTIVE) {
      observed_status = stream->capture_status;
      iree_atomic_store(&graph->capture_state,
                        IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_INVALIDATED,
                        iree_memory_order_release);
    } else if (capture_state ==
               IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_INVALIDATED) {
      observed_status = IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED;
    }
    if (observed_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
      iree_hal_streaming_stream_set_capture_status(
          stream, IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED);
    }
  }
  iree_slim_mutex_unlock(&stream->mutex);
  iree_slim_mutex_unlock(&graph->capture_mutex);
  iree_hal_streaming_graph_release(graph);
  return observed_status;
}

iree_status_t iree_hal_streaming_capture_try_record_node(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_capture_record_node_fn_t record_fn, void* user_data,
    bool* out_was_capturing) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(record_fn);
  IREE_ASSERT_ARGUMENT(out_was_capturing);

  iree_hal_streaming_capture_transaction_t transaction;
  iree_status_t status = iree_hal_streaming_capture_transaction_begin(
      stream, &transaction, out_was_capturing);
  if (!iree_status_is_ok(status) || !*out_was_capturing) {
    return status;
  }

  // Reserve the one-entry terminal frontier before allowing the callback to
  // mutate the graph. Once recording starts, any failure invalidates the whole
  // shared session: graph arenas can own staging allocations in addition to
  // visible nodes, so removing only the nodes cannot restore prior state.
  if (stream->capture_dependency_capacity == 0) {
    status = iree_hal_streaming_grow_capture_dependencies(stream, 1);
  }
  iree_hal_streaming_graph_node_t* terminal_node = NULL;
  if (iree_status_is_ok(status)) {
    status =
        record_fn(transaction.graph, stream->capture_dependencies,
                  stream->capture_dependency_count, user_data, &terminal_node);
  }
  if (iree_status_is_ok(status) &&
      iree_atomic_load(&transaction.graph->capture_state,
                       iree_memory_order_acquire) !=
          IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_ACTIVE) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "stream capture has been invalidated");
  }
  if (iree_status_is_ok(status) &&
      !iree_hal_streaming_graph_node_is_active_in_graph(transaction.graph,
                                                        terminal_node)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "capture recorder returned a node outside the active graph");
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_capture_transaction_invalidate(&transaction);
  } else {
    stream->capture_dependencies[0] = terminal_node;
    stream->capture_dependency_count = 1;
  }
  iree_hal_streaming_capture_transaction_unlock(&transaction);
  return status;
}

iree_status_t iree_hal_streaming_capture_try_record_noop(
    iree_hal_streaming_stream_t* stream, bool* out_was_capturing) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(out_was_capturing);

  iree_hal_streaming_capture_transaction_t transaction;
  iree_status_t status = iree_hal_streaming_capture_transaction_begin(
      stream, &transaction, out_was_capturing);
  if (!iree_status_is_ok(status) || !*out_was_capturing) {
    return status;
  }
  if (iree_atomic_load(&transaction.graph->capture_state,
                       iree_memory_order_acquire) !=
      IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_ACTIVE) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "stream capture has been invalidated");
  }
  iree_hal_streaming_capture_transaction_unlock(&transaction);
  return status;
}

iree_status_t iree_hal_streaming_capture_try_record_event(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_event_t* event,
    bool* out_was_capturing) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(event);
  IREE_ASSERT_ARGUMENT(out_was_capturing);

  iree_hal_streaming_capture_transaction_t transaction;
  iree_status_t status = iree_hal_streaming_capture_transaction_begin(
      stream, &transaction, out_was_capturing);
  if (!iree_status_is_ok(status) || !*out_was_capturing) {
    return status;
  }

  iree_hal_streaming_graph_t* dropped_graph = NULL;
  iree_slim_mutex_lock(&event->mutex);
  if (event->capture_dependency_capacity < stream->capture_dependency_count) {
    iree_host_size_t allocation_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            stream->capture_dependency_count,
            sizeof(*event->capture_dependencies), &allocation_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "event capture frontier size overflow");
    } else {
      status = iree_allocator_realloc(event->host_allocator, allocation_size,
                                      (void**)&event->capture_dependencies);
    }
    if (iree_status_is_ok(status)) {
      event->capture_dependency_capacity = stream->capture_dependency_count;
    }
  }
  if (iree_status_is_ok(status)) {
    if (stream->capture_dependency_count > 0) {
      memcpy(event->capture_dependencies, stream->capture_dependencies,
             stream->capture_dependency_count *
                 sizeof(*event->capture_dependencies));
    }
    event->capture_dependency_count = stream->capture_dependency_count;
    event->capture_id = stream->capture_id;
    if (event->capture_graph != transaction.graph) {
      iree_hal_streaming_graph_retain(transaction.graph);
      dropped_graph = event->capture_graph;
      event->capture_graph = transaction.graph;
    }
  }
  iree_slim_mutex_unlock(&event->mutex);

  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_capture_transaction_invalidate(&transaction);
  }
  iree_hal_streaming_capture_transaction_unlock(&transaction);
  iree_hal_streaming_graph_release(dropped_graph);
  return status;
}

iree_status_t iree_hal_streaming_capture_join_graph(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_graph_t* graph,
    unsigned long long capture_id,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(graph);
  if (dependency_count > 0 && !dependencies) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dependency array must be provided");
  }

  iree_slim_mutex_lock(&graph->capture_mutex);
  if (graph->capture_id != capture_id ||
      iree_atomic_load(&graph->capture_state, iree_memory_order_acquire) !=
          IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_ACTIVE) {
    iree_slim_mutex_unlock(&graph->capture_mutex);
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "captured event session is no longer active");
  }

  iree_hal_streaming_context_t* context = graph->context;
  iree_slim_mutex_lock(&context->stream_list_mutex);
  iree_slim_mutex_lock(&stream->mutex);
  const bool adopt_graph =
      stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
  iree_status_t status = iree_ok_status();
  if (stream->context != context ||
      stream->registration_state !=
          IREE_HAL_STREAMING_STREAM_REGISTRATION_STATE_REGISTERED) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "stream is no longer registered");
  } else if (!adopt_graph && (stream->capture_status !=
                                  IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE ||
                              stream->capture_graph != graph ||
                              stream->capture_id != capture_id)) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "event wait crosses different active capture graphs");
  }
  for (iree_host_size_t i = 0;
       i < dependency_count && iree_status_is_ok(status); ++i) {
    if (!iree_hal_streaming_graph_node_is_active_in_graph(graph,
                                                          dependencies[i])) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "captured event dependency at index %" PRIhsz
                                " is not in the active graph",
                                i);
    }
  }
  iree_host_size_t total_count = dependency_count;
  if (!adopt_graph && iree_status_is_ok(status) &&
      IREE_UNLIKELY(!iree_host_size_checked_add(
          stream->capture_dependency_count, dependency_count, &total_count))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "capture dependency count overflow");
  }
  if (iree_status_is_ok(status) &&
      total_count > stream->capture_dependency_capacity) {
    status = iree_hal_streaming_grow_capture_dependencies(stream, total_count);
  }
  if (iree_status_is_ok(status) && adopt_graph) {
    status = iree_hal_streaming_stream_flush_locked(stream);
  }

  if (iree_status_is_ok(status)) {
    if (adopt_graph) {
      stream->capture_mode = graph->capture_mode;
      stream->capture_graph = graph;
      stream->capture_graph_owned = true;
      stream->capture_origin = false;
      stream->capture_id = capture_id;
      stream->capture_owner_thread_id = graph->capture_owner_thread_id;
      stream->capture_dependency_count = 0;
      iree_hal_streaming_graph_retain(graph);
      iree_hal_streaming_stream_set_capture_status(
          stream, IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE);
    }
    if (dependency_count > 0) {
      memcpy(stream->capture_dependencies + stream->capture_dependency_count,
             dependencies, dependency_count * sizeof(*dependencies));
    }
    stream->capture_dependency_count = total_count;
  }

  iree_slim_mutex_unlock(&stream->mutex);
  iree_slim_mutex_unlock(&context->stream_list_mutex);
  iree_slim_mutex_unlock(&graph->capture_mutex);
  return status;
}

static iree_status_t iree_hal_streaming_capture_publish_begin(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, iree_hal_streaming_capture_mode_t mode,
    bool graph_owned) {
  iree_hal_streaming_context_t* context = graph->context;
  unsigned long long capture_id = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_context_allocate_capture_id(context, &capture_id));

  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&graph->capture_mutex);
  iree_slim_mutex_lock(&context->stream_list_mutex);
  iree_slim_mutex_lock(&stream->mutex);
  if (stream->context != context ||
      stream->registration_state !=
          IREE_HAL_STREAMING_STREAM_REGISTRATION_STATE_REGISTERED) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "stream is no longer registered");
  } else if (iree_atomic_load(&graph->capture_state,
                              iree_memory_order_acquire) !=
             IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_INACTIVE) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "graph is already participating in capture");
  } else if (stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "stream is already capturing");
  }
  for (iree_host_size_t i = 0;
       i < dependency_count && iree_status_is_ok(status); ++i) {
    if (!iree_hal_streaming_graph_node_is_active_in_graph(graph,
                                                          dependencies[i])) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "capture dependency at index %" PRIhsz
                                " does not belong to the capture graph",
                                i);
    }
  }
  if (iree_status_is_ok(status) &&
      dependency_count > stream->capture_dependency_capacity) {
    status =
        iree_hal_streaming_grow_capture_dependencies(stream, dependency_count);
  }
  if (iree_status_is_ok(status) && dependency_count > 0) {
    memcpy(stream->capture_dependencies, dependencies,
           dependency_count * sizeof(*dependencies));
  }

  // Flushing while the stream lock is held makes prior submission and capture
  // publication one transition. No operation can be accepted between them and
  // escape both the pre-capture stream and the capture graph.
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_stream_flush_locked(stream);
  }

  if (iree_status_is_ok(status)) {
    const uintptr_t owner_thread_id = iree_hal_streaming_current_thread_token();
    graph->capture_id = capture_id;
    graph->capture_mode = mode;
    graph->capture_owner_thread_id = owner_thread_id;
    iree_atomic_store(&graph->capture_state,
                      IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_ACTIVE,
                      iree_memory_order_release);
    stream->capture_mode = mode;
    stream->capture_graph = graph;
    stream->capture_graph_owned = graph_owned;
    stream->capture_origin = true;
    stream->capture_id = capture_id;
    stream->capture_owner_thread_id = owner_thread_id;
    stream->capture_dependency_count = dependency_count;
    iree_hal_streaming_stream_set_capture_status(
        stream, IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE);
  }

  iree_slim_mutex_unlock(&stream->mutex);
  iree_slim_mutex_unlock(&context->stream_list_mutex);
  iree_slim_mutex_unlock(&graph->capture_mutex);
  return status;
}

static iree_status_t iree_hal_streaming_begin_capture_impl(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_capture_mode_t mode) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_context_t* context = NULL;
  if (!iree_hal_streaming_stream_retain_context(stream, &context)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream execution context has been destroyed");
  }

  iree_hal_streaming_graph_t* graph = NULL;
  iree_status_t status = iree_hal_streaming_graph_create(
      context, /*flags=*/0, context->host_allocator, &graph);
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_capture_publish_begin(
        stream, graph, /*dependencies=*/NULL, /*dependency_count=*/0, mode,
        /*graph_owned=*/true);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_graph_release(graph);
  }
  iree_hal_streaming_context_release(context);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_begin_capture(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_capture_mode_t mode) {
  IREE_ASSERT_ARGUMENT(stream);
  iree_hal_streaming_context_t* context = NULL;
  if (!iree_hal_streaming_stream_retain_context(stream, &context)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream execution context has been destroyed");
  }
  iree_hal_streaming_capture_admission_begin_transition(
      &context->capture_admission);
  iree_status_t status = iree_hal_streaming_begin_capture_impl(stream, mode);
  iree_hal_streaming_capture_admission_end_transition(
      &context->capture_admission);
  iree_hal_streaming_context_release(context);
  return status;
}

static iree_status_t iree_hal_streaming_begin_capture_to_graph_impl(
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
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_validate_dependencies(graph, dependencies,
                                                         dependency_count));

  iree_status_t status = iree_hal_streaming_capture_publish_begin(
      stream, graph, dependencies, dependency_count, mode,
      /*graph_owned=*/false);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
// Clears one stream's membership in |graph|. The graph capture mutex and the
// stream mutex must both be held. Returns whether the stream owned a graph
// reference that must be released after all capture locks are dropped.
static bool iree_hal_streaming_clear_capture_stream_locked(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_graph_t* graph,
    unsigned long long capture_id) {
  if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE ||
      stream->capture_graph != graph || stream->capture_id != capture_id) {
    return false;
  }
  const bool capture_graph_owned = stream->capture_graph_owned;
  iree_hal_streaming_stream_set_capture_status(
      stream, IREE_HAL_STREAMING_CAPTURE_STATUS_NONE);
  stream->capture_graph = NULL;
  stream->capture_graph_owned = false;
  stream->capture_origin = false;
  stream->capture_id = 0;
  stream->capture_owner_thread_id = 0;
  stream->capture_dependency_count = 0;
  return capture_graph_owned;
}

iree_status_t iree_hal_streaming_begin_capture_to_graph(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count, iree_hal_streaming_capture_mode_t mode) {
  IREE_ASSERT_ARGUMENT(stream);
  iree_hal_streaming_context_t* context = NULL;
  if (!iree_hal_streaming_stream_retain_context(stream, &context)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream execution context has been destroyed");
  }
  iree_hal_streaming_capture_admission_begin_transition(
      &context->capture_admission);
  iree_status_t status = iree_hal_streaming_begin_capture_to_graph_impl(
      stream, graph, dependencies, dependency_count, mode);
  iree_hal_streaming_capture_admission_end_transition(
      &context->capture_admission);
  iree_hal_streaming_context_release(context);
  return status;
}

// Clears every stream belonging to the exact capture session and resets its
// graph metadata. The caller holds the graph capture mutex and the context
// stream-list mutex; this helper takes at most one stream mutex at a time,
// preserving the graph -> stream-list -> stream lifecycle order.
//
// Returns the number of owned graph references to release after all locks are
// dropped. |preserve_owned_stream|, when non-NULL, transfers that stream's
// owned reference to the caller instead of releasing it.
static iree_host_size_t iree_hal_streaming_clear_capture_session_locked(
    iree_hal_streaming_context_t* context, iree_hal_streaming_graph_t* graph,
    unsigned long long capture_id,
    iree_hal_streaming_stream_t* preserve_owned_stream) {
  iree_host_size_t release_count = 0;
  for (iree_host_size_t i = 0; i < context->stream_count; ++i) {
    iree_hal_streaming_stream_t* participant = context->streams[i];
    iree_slim_mutex_lock(&participant->mutex);
    if (iree_hal_streaming_clear_capture_stream_locked(participant, graph,
                                                       capture_id) &&
        participant != preserve_owned_stream) {
      ++release_count;
    }
    iree_slim_mutex_unlock(&participant->mutex);
  }

  if (graph->capture_id == capture_id) {
    graph->capture_id = 0;
    graph->capture_owner_thread_id = 0;
    iree_atomic_store(&graph->capture_state,
                      IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_INACTIVE,
                      iree_memory_order_release);
  }
  return release_count;
}

bool iree_hal_streaming_capture_unregister_stream(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(stream);

  for (;;) {
    // Take a retained session snapshot without nesting locks, then revalidate
    // it after acquiring the global lifecycle order.
    iree_hal_streaming_graph_t* graph = NULL;
    unsigned long long capture_id = 0;
    iree_slim_mutex_lock(&stream->mutex);
    if (stream->context != context ||
        stream->registration_state !=
            IREE_HAL_STREAMING_STREAM_REGISTRATION_STATE_REGISTERED) {
      iree_slim_mutex_unlock(&stream->mutex);
      return false;
    }
    graph = stream->capture_graph;
    capture_id = stream->capture_id;
    iree_hal_streaming_graph_retain(graph);
    iree_slim_mutex_unlock(&stream->mutex);

    if (graph) {
      iree_slim_mutex_lock(&graph->capture_mutex);
    }
    iree_slim_mutex_lock(&context->stream_list_mutex);
    iree_slim_mutex_lock(&stream->mutex);
    const bool snapshot_matches =
        stream->context == context &&
        stream->registration_state ==
            IREE_HAL_STREAMING_STREAM_REGISTRATION_STATE_REGISTERED &&
        stream->capture_graph == graph && stream->capture_id == capture_id;
    if (!snapshot_matches) {
      iree_slim_mutex_unlock(&stream->mutex);
      iree_slim_mutex_unlock(&context->stream_list_mutex);
      if (graph) {
        iree_slim_mutex_unlock(&graph->capture_mutex);
      }
      iree_hal_streaming_graph_release(graph);
      continue;
    }

    iree_host_size_t stream_index = context->stream_count;
    for (iree_host_size_t i = 0; i < context->stream_count; ++i) {
      if (context->streams[i] == stream) {
        stream_index = i;
        break;
      }
    }
    if (stream_index == context->stream_count) {
      stream->registration_state =
          IREE_HAL_STREAMING_STREAM_REGISTRATION_STATE_UNREGISTERED;
      iree_slim_mutex_unlock(&stream->mutex);
      iree_slim_mutex_unlock(&context->stream_list_mutex);
      if (graph) {
        iree_slim_mutex_unlock(&graph->capture_mutex);
      }
      iree_hal_streaming_graph_release(graph);
      return false;
    }

    const bool was_capture_origin = stream->capture_origin;
    stream->registration_state =
        IREE_HAL_STREAMING_STREAM_REGISTRATION_STATE_DETACHING;
    iree_slim_mutex_unlock(&stream->mutex);

    iree_host_size_t release_owned_graph_count = 0;
    if (graph) {
      const bool exact_session = capture_id != 0 && graph->context == context &&
                                 graph->capture_id == capture_id;
      if (was_capture_origin && exact_session) {
        // With no origin left there is no API that can finish this session.
        // Detach every exact member now so surviving joined streams can be
        // reused, and leave all graph destruction until after unlocking.
        release_owned_graph_count =
            iree_hal_streaming_clear_capture_session_locked(
                context, graph, capture_id,
                /*preserve_owned_stream=*/NULL);
      } else {
        const int32_t capture_state =
            iree_atomic_load(&graph->capture_state, iree_memory_order_acquire);
        if (exact_session &&
            capture_state == IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_ACTIVE) {
          iree_atomic_store(&graph->capture_state,
                            IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_INVALIDATED,
                            iree_memory_order_release);
        }
        iree_slim_mutex_lock(&stream->mutex);
        if (iree_hal_streaming_clear_capture_stream_locked(stream, graph,
                                                           capture_id)) {
          ++release_owned_graph_count;
        }
        iree_slim_mutex_unlock(&stream->mutex);
      }
    }

    iree_slim_mutex_lock(&stream->mutex);
    context->streams[stream_index] =
        context->streams[context->stream_count - 1];
    --context->stream_count;
    stream->registration_state =
        IREE_HAL_STREAMING_STREAM_REGISTRATION_STATE_UNREGISTERED;
    iree_slim_mutex_unlock(&stream->mutex);

    iree_slim_mutex_unlock(&context->stream_list_mutex);
    if (graph) {
      iree_slim_mutex_unlock(&graph->capture_mutex);
    }
    while (release_owned_graph_count-- > 0) {
      iree_hal_streaming_graph_release(graph);
    }
    // Drop the temporary reference used to cross the lock-order handoff.
    iree_hal_streaming_graph_release(graph);
    return true;
  }
}

typedef struct iree_hal_streaming_graph_additional_edge_index_t {
  uint32_t* head_indices;
  uint32_t* next_indices;
  iree_hal_streaming_graph_node_t** from_nodes;
} iree_hal_streaming_graph_additional_edge_index_t;

typedef struct iree_hal_streaming_capture_reachability_scratch_t {
  uint8_t* reachable_nodes;
  iree_hal_streaming_graph_node_t** stack;
  iree_hal_streaming_graph_additional_edge_index_t additional_edge_index;
} iree_hal_streaming_capture_reachability_scratch_t;

static void iree_hal_streaming_graph_deinitialize_additional_edge_index(
    iree_allocator_t host_allocator,
    iree_hal_streaming_graph_additional_edge_index_t* index) {
  iree_allocator_free(host_allocator, index->from_nodes);
  iree_allocator_free(host_allocator, index->next_indices);
  iree_allocator_free(host_allocator, index->head_indices);
}

static void iree_hal_streaming_capture_reachability_scratch_deinitialize(
    iree_allocator_t host_allocator,
    iree_hal_streaming_capture_reachability_scratch_t* scratch) {
  iree_hal_streaming_graph_deinitialize_additional_edge_index(
      host_allocator, &scratch->additional_edge_index);
  iree_allocator_free(host_allocator, scratch->stack);
  iree_allocator_free(host_allocator, scratch->reachable_nodes);
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
    if (!additional_edge_index->head_indices) {
      continue;
    }
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
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_capture_reachability_scratch_t* scratch,
    bool* out_has_unjoined_participant) {
  *out_has_unjoined_participant = false;
  memset(scratch, 0, sizeof(*scratch));

  iree_allocator_t host_allocator = graph->host_allocator;
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

  scratch->reachable_nodes = reachable_nodes;
  scratch->stack = stack;
  scratch->additional_edge_index = additional_edge_index;
  if (!iree_status_is_ok(status)) {
    return status;
  }

  iree_hal_streaming_context_t* context = graph->context;
  iree_slim_mutex_lock(&context->stream_list_mutex);
  for (iree_host_size_t i = 0; i < context->stream_count; ++i) {
    iree_hal_streaming_stream_t* stream = context->streams[i];
    if (stream == origin_stream) {
      continue;
    }

    iree_slim_mutex_lock(&stream->mutex);
    if (stream->registration_state ==
            IREE_HAL_STREAMING_STREAM_REGISTRATION_STATE_REGISTERED &&
        stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE &&
        stream->capture_graph == graph &&
        stream->capture_id == graph->capture_id &&
        !iree_hal_streaming_capture_frontier_is_joined(graph, reachable_nodes,
                                                       stream)) {
      *out_has_unjoined_participant = true;
    }
    iree_slim_mutex_unlock(&stream->mutex);
    if (*out_has_unjoined_participant) {
      break;
    }
  }
  iree_slim_mutex_unlock(&context->stream_list_mutex);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_end_capture_impl(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_graph_t** out_graph) {
  IREE_ASSERT_ARGUMENT(stream);
  if (out_graph) {
    *out_graph = NULL;
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  const uintptr_t current_thread_id = iree_hal_streaming_current_thread_token();

  // Retain the graph from a short stream-only snapshot, then reacquire locks
  // in the global graph-before-stream order and revalidate the snapshot.
  iree_slim_mutex_lock(&stream->mutex);
  if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
    iree_slim_mutex_unlock(&stream->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream is not capturing");
  }
  iree_hal_streaming_graph_t* graph = stream->capture_graph;
  const unsigned long long capture_id = stream->capture_id;
  if (!graph) {
    iree_slim_mutex_unlock(&stream->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "stream capture has no graph");
  }
  iree_hal_streaming_graph_retain(graph);
  iree_slim_mutex_unlock(&stream->mutex);

  iree_hal_streaming_context_t* context = graph->context;
  iree_slim_mutex_lock(&graph->capture_mutex);
  iree_slim_mutex_lock(&context->stream_list_mutex);
  iree_slim_mutex_lock(&stream->mutex);
  iree_status_t status = iree_ok_status();
  if (stream->context != context || graph->capture_id != capture_id ||
      stream->registration_state !=
          IREE_HAL_STREAMING_STREAM_REGISTRATION_STATE_REGISTERED ||
      stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE ||
      stream->capture_graph != graph || stream->capture_id != capture_id) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "stream capture ended concurrently");
  } else if (!stream->capture_origin) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "stream did not originate this capture");
  } else if (stream->capture_mode != IREE_HAL_STREAMING_CAPTURE_MODE_RELAXED &&
             stream->capture_owner_thread_id != current_thread_id) {
    status = iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                              "stream capture ended from wrong thread");
  }
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&stream->mutex);
    iree_slim_mutex_unlock(&context->stream_list_mutex);
    iree_slim_mutex_unlock(&graph->capture_mutex);
    iree_hal_streaming_graph_release(graph);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  const bool origin_graph_owned = stream->capture_graph_owned;
  iree_slim_mutex_unlock(&stream->mutex);
  iree_slim_mutex_unlock(&context->stream_list_mutex);

  // Reachability allocations live until the capture transaction completes,
  // but releasing them may invoke arbitrary host allocator work. Keep cleanup
  // outside all graph, stream-list, and stream locks.
  iree_hal_streaming_capture_reachability_scratch_t scratch = {0};
  bool has_unjoined_participant = false;
  const int32_t capture_state =
      iree_atomic_load(&graph->capture_state, iree_memory_order_acquire);
  if (capture_state == IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_ACTIVE) {
    status = iree_hal_streaming_has_unjoined_capture_participants(
        stream, graph, &scratch, &has_unjoined_participant);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&graph->capture_mutex);
      iree_hal_streaming_capture_reachability_scratch_deinitialize(
          graph->host_allocator, &scratch);
      iree_hal_streaming_graph_release(graph);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    if (has_unjoined_participant) {
      status = iree_make_status(
          IREE_STATUS_ABORTED,
          "stream capture has participant work not joined to the origin "
          "stream");
    }
  } else if (capture_state ==
             IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_INVALIDATED) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "stream capture has been invalidated");
  } else {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "stream capture session is inconsistent");
  }

  // The graph transaction excludes recorders and adoption while the shared
  // lifecycle helper detaches every exact member in graph -> stream-list ->
  // stream order. Preserve an origin-owned reference for successful transfer;
  // all other ownership is released after every capture lock is dropped.
  iree_slim_mutex_lock(&context->stream_list_mutex);
  iree_host_size_t graph_release_count =
      iree_hal_streaming_clear_capture_session_locked(
          context, graph, capture_id, origin_graph_owned ? stream : NULL);
  iree_slim_mutex_unlock(&context->stream_list_mutex);

  iree_slim_mutex_unlock(&graph->capture_mutex);
  iree_hal_streaming_capture_reachability_scratch_deinitialize(
      graph->host_allocator, &scratch);

  while (graph_release_count-- > 0) {
    iree_hal_streaming_graph_release(graph);
  }
  if (iree_status_is_ok(status)) {
    if (out_graph) {
      *out_graph = graph;
    } else if (origin_graph_owned) {
      iree_hal_streaming_graph_release(graph);
    }
  } else if (origin_graph_owned) {
    iree_hal_streaming_graph_release(graph);
  }
  // Drop the temporary reference used to cross the lock-order handoff.
  iree_hal_streaming_graph_release(graph);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_end_capture(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_graph_t** out_graph) {
  IREE_ASSERT_ARGUMENT(stream);
  iree_hal_streaming_context_t* context = NULL;
  if (!iree_hal_streaming_stream_retain_context(stream, &context)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream execution context has been destroyed");
  }
  iree_hal_streaming_capture_admission_begin_transition(
      &context->capture_admission);
  iree_status_t status = iree_hal_streaming_end_capture_impl(stream, out_graph);
  iree_hal_streaming_capture_admission_end_transition(
      &context->capture_admission);
  iree_hal_streaming_context_release(context);
  return status;
}

iree_status_t iree_hal_streaming_capture_status(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_capture_status_t* out_status,
    unsigned long long* out_id) {
  IREE_ASSERT_ARGUMENT(stream);

  iree_slim_mutex_lock(&stream->mutex);
  iree_hal_streaming_capture_status_t capture_status = stream->capture_status;
  if (capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE &&
      stream->capture_graph &&
      iree_atomic_load(&stream->capture_graph->capture_state,
                       iree_memory_order_acquire) ==
          IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_INVALIDATED) {
    capture_status = IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED;
  }
  if (out_status) {
    *out_status = capture_status;
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
      stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE &&
      stream->capture_graph &&
      iree_atomic_load(&stream->capture_graph->capture_state,
                       iree_memory_order_acquire) ==
          IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_ACTIVE;
  iree_slim_mutex_unlock(&stream->mutex);

  return iree_ok_status();
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

  bool was_capturing = false;
  iree_hal_streaming_capture_transaction_t transaction;
  iree_status_t status = iree_hal_streaming_capture_transaction_begin(
      stream, &transaction, &was_capturing);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  if (!was_capturing) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stream is not actively capturing");
  }

  for (iree_host_size_t i = 0;
       i < dependency_count && iree_status_is_ok(status); ++i) {
    if (!iree_hal_streaming_graph_node_is_active_in_graph(transaction.graph,
                                                          dependencies[i])) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "capture dependency at index %" PRIhsz
                                " does not belong to the active capture graph",
                                i);
    }
  }

  iree_host_size_t total_count = dependency_count;
  if (iree_status_is_ok(status) &&
      mode == IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_ADD &&
      IREE_UNLIKELY(!iree_host_size_checked_add(
          stream->capture_dependency_count, dependency_count, &total_count))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "capture dependency count overflow");
  }
  if (iree_status_is_ok(status) &&
      total_count > stream->capture_dependency_capacity) {
    status = iree_hal_streaming_grow_capture_dependencies(stream, total_count);
  }
  if (iree_status_is_ok(status) &&
      iree_atomic_load(&transaction.graph->capture_state,
                       iree_memory_order_acquire) !=
          IREE_HAL_STREAMING_GRAPH_CAPTURE_STATE_ACTIVE) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "stream capture has been invalidated");
    iree_hal_streaming_stream_set_capture_status(
        stream, IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED);
  }
  if (iree_status_is_ok(status)) {
    if (dependency_count > 0) {
      void* dest =
          mode == IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_ADD
              ? stream->capture_dependencies + stream->capture_dependency_count
              : stream->capture_dependencies;
      memcpy(dest, dependencies, dependency_count * sizeof(*dependencies));
    }
    stream->capture_dependency_count = total_count;
  }

  iree_hal_streaming_capture_transaction_unlock(&transaction);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
