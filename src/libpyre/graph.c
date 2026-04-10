// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_internal.h"

#include "buffer_table.h"

//===----------------------------------------------------------------------===//
// Node and block allocation helpers
//===----------------------------------------------------------------------===//

static pyre_status_t pyre_graph_allocate_node(
    iree_allocator_t allocator, iree_host_size_t dependency_count,
    iree_host_size_t extra_data_size, pyre_graph_node_s** out_node,
    uint8_t** out_extra_data) {
  IREE_ASSERT_ARGUMENT(out_node);
  *out_node = NULL;
  if (out_extra_data) *out_extra_data = NULL;

  const iree_host_size_t node_size = sizeof(pyre_graph_node_s);
  const iree_host_size_t deps_size =
      dependency_count * sizeof(pyre_graph_node_s*);
  iree_host_size_t total_size = node_size + deps_size;

  if (extra_data_size > 0) {
    total_size = iree_host_align(total_size, iree_max_align_t);
    const iree_host_size_t extra_data_offset = total_size;
    total_size += extra_data_size;

    pyre_graph_node_s* node = NULL;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator, total_size, (void**)&node));
    *out_node = node;
    if (out_extra_data) {
      *out_extra_data = (uint8_t*)node + extra_data_offset;
    }
  } else {
    pyre_graph_node_s* node = NULL;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator, total_size, (void**)&node));
    *out_node = node;
  }

  return iree_ok_status();
}

static pyre_status_t pyre_graph_allocate_node_block(
    iree_allocator_t allocator, iree_host_size_t capacity,
    pyre_graph_node_block_t** out_block) {
  const iree_host_size_t block_size =
      sizeof(pyre_graph_node_block_t) +
      capacity * sizeof(pyre_graph_node_s*);

  pyre_graph_node_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, block_size, (void**)&block));

  block->next = NULL;
  block->capacity = capacity;
  block->count = 0;
  *out_block = block;
  return iree_ok_status();
}

static pyre_status_t pyre_graph_add_node_internal(
    pyre_graph_s* graph, pyre_graph_node_s* node) {
  node->node_index = (uint32_t)graph->node_count;

  if (!graph->current_node_block ||
      graph->current_node_block->count >= graph->current_node_block->capacity) {
    const iree_host_size_t block_capacity =
        graph->node_count < 64 ? 16 : 64;
    pyre_graph_node_block_t* new_block = NULL;
    IREE_RETURN_IF_ERROR(pyre_graph_allocate_node_block(
        graph->arena_allocator, block_capacity, &new_block));

    if (graph->current_node_block) {
      graph->current_node_block->next = new_block;
    } else {
      graph->node_blocks = new_block;
    }
    graph->current_node_block = new_block;
  }

  graph->current_node_block->nodes[graph->current_node_block->count++] = node;
  ++graph->node_count;

  if (node->dependency_count == 0) {
    if (!graph->current_root_block || graph->current_root_block->count >=
                                          graph->current_root_block->capacity) {
      const iree_host_size_t block_capacity = 8;
      pyre_graph_node_block_t* new_block = NULL;
      IREE_RETURN_IF_ERROR(pyre_graph_allocate_node_block(
          graph->arena_allocator, block_capacity, &new_block));

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

//===----------------------------------------------------------------------===//
// pyre_graph_t (template)
//===----------------------------------------------------------------------===//

static void pyre_graph_destroy(pyre_graph_s* graph);

pyre_status_t pyre_graph_create(pyre_device_t device, uint32_t flags,
                                pyre_graph_t* out_graph) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_graph);
  *out_graph = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  pyre_graph_s* graph = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(iree_allocator_system(), sizeof(*graph),
                            (void**)&graph));

  iree_atomic_ref_count_init(&graph->ref_count);
  graph->device = device;

  iree_arena_initialize(&device->block_pool, &graph->arena);
  graph->arena_allocator = iree_arena_allocator(&graph->arena);

  graph->node_blocks = NULL;
  graph->current_node_block = NULL;
  graph->node_count = 0;
  graph->root_blocks = NULL;
  graph->current_root_block = NULL;
  graph->root_count = 0;
  graph->additional_edges = NULL;
  graph->additional_edge_count = 0;
  graph->flags = flags;
  iree_slim_mutex_initialize(&graph->mutex);

  *out_graph = graph;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void pyre_graph_destroy(pyre_graph_s* graph) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_arena_deinitialize(&graph->arena);
  iree_slim_mutex_deinitialize(&graph->mutex);
  iree_allocator_free(iree_allocator_system(), graph);

  IREE_TRACE_ZONE_END(z0);
}

void pyre_graph_retain(pyre_graph_t graph) {
  if (graph) {
    iree_atomic_ref_count_inc(&graph->ref_count);
  }
}

void pyre_graph_release(pyre_graph_t graph) {
  if (graph && iree_atomic_ref_count_dec(&graph->ref_count) == 1) {
    pyre_graph_destroy(graph);
  }
}

pyre_status_t pyre_graph_size(pyre_graph_t graph, size_t* out_count) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(out_count);
  *out_count = graph->node_count;
  return iree_ok_status();
}

pyre_status_t pyre_graph_get_nodes(pyre_graph_t graph,
                                   pyre_graph_node_t* nodes,
                                   size_t* inout_count) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(inout_count);

  size_t requested = *inout_count;
  if (requested > graph->node_count) requested = graph->node_count;

  iree_host_size_t copied = 0;
  pyre_graph_node_block_t* block = graph->node_blocks;
  while (block && copied < requested) {
    iree_host_size_t to_copy = block->count;
    if (copied + to_copy > requested) to_copy = requested - copied;
    for (iree_host_size_t i = 0; i < to_copy; i++) {
      nodes[copied++] = block->nodes[i];
    }
    block = block->next;
  }

  *inout_count = copied;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Node creation
//===----------------------------------------------------------------------===//

pyre_status_t pyre_graph_add_empty_node(pyre_graph_t graph,
                                        const pyre_graph_node_t* deps,
                                        size_t dep_count,
                                        pyre_graph_node_t* out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_TRACE_ZONE_BEGIN(z0);

  pyre_graph_node_s* node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, pyre_graph_allocate_node(graph->arena_allocator, dep_count, 0, &node,
                                   NULL));

  node->type = PYRE_GRAPH_NODE_TYPE_INTERNAL_EMPTY;
  node->dependency_count = dep_count;
  if (dep_count > 0) {
    memcpy(node->dependencies, deps, dep_count * sizeof(*deps));
  }

  iree_status_t status = pyre_graph_add_node_internal(graph, node);
  if (iree_status_is_ok(status) && out_node) {
    *out_node = node;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

pyre_status_t pyre_graph_add_kernel_node(
    pyre_graph_t graph, const pyre_graph_node_t* deps, size_t dep_count,
    const pyre_graph_kernel_node_attrs_t* attrs,
    pyre_graph_node_t* out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(attrs);
  IREE_ASSERT_ARGUMENT(attrs->executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_host_size_t constants_size =
      iree_host_align(attrs->constants_size, iree_max_align_t);
  const iree_host_size_t bindings_size = iree_host_align(
      attrs->binding_count * sizeof(iree_hal_buffer_ref_t), iree_max_align_t);
  uint8_t* extra_data = NULL;
  pyre_graph_node_s* node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, pyre_graph_allocate_node(graph->arena_allocator, dep_count,
                                   constants_size + bindings_size, &node,
                                   &extra_data));

  node->type = PYRE_GRAPH_NODE_TYPE_INTERNAL_KERNEL;
  node->dependency_count = dep_count;
  if (dep_count > 0) {
    memcpy(node->dependencies, deps, dep_count * sizeof(*deps));
  }

  pyre_graph_kernel_node_attrs_internal_t* k = &node->attrs.kernel;
  k->executable = attrs->executable->hal_executable;
  k->export_ordinal = attrs->export_ordinal;
  memcpy(k->grid_dim, attrs->config.workgroup_count, sizeof(k->grid_dim));
  memcpy(k->block_dim, attrs->config.workgroup_size, sizeof(k->block_dim));
  k->shared_memory_bytes = 0;

  void* constants_copy = extra_data;
  if (attrs->constants_size > 0 && attrs->constants) {
    memcpy(constants_copy, attrs->constants, attrs->constants_size);
  }
  k->constants =
      iree_make_const_byte_span(constants_copy, attrs->constants_size);

  iree_hal_buffer_ref_t* bindings_data =
      (iree_hal_buffer_ref_t*)(extra_data + constants_size);
  for (size_t i = 0; i < attrs->binding_count; i++) {
    pyre_buffer_s* buf = attrs->bindings[i].buffer;
    bindings_data[i] = (iree_hal_buffer_ref_t){
        .buffer = buf ? buf->hal_buffer : NULL,
        .offset = (iree_device_size_t)attrs->bindings[i].offset,
        .length = (iree_device_size_t)attrs->bindings[i].length,
    };
  }
  k->bindings.count = attrs->binding_count;
  k->bindings.values = bindings_data;

  iree_status_t status = pyre_graph_add_node_internal(graph, node);
  if (iree_status_is_ok(status) && out_node) {
    *out_node = node;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

pyre_status_t pyre_graph_add_memcpy_node(
    pyre_graph_t graph, const pyre_graph_node_t* deps, size_t dep_count,
    const pyre_graph_memcpy_node_attrs_t* attrs,
    pyre_graph_node_t* out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(attrs);
  IREE_TRACE_ZONE_BEGIN(z0);

  pyre_buffer_t dst_buf = NULL;
  size_t dst_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, pyre_buffer_table_find(&graph->device->buffer_table,
                                 (uint64_t)(uintptr_t)attrs->dst,
                                 &dst_buf, &dst_offset, NULL),
      "resolving dst buffer %p", attrs->dst);
  pyre_buffer_t src_buf = NULL;
  size_t src_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, pyre_buffer_table_find(&graph->device->buffer_table,
                                 (uint64_t)(uintptr_t)attrs->src,
                                 &src_buf, &src_offset, NULL),
      "resolving src buffer %p", attrs->src);

  pyre_graph_node_s* node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, pyre_graph_allocate_node(graph->arena_allocator, dep_count, 0, &node,
                                   NULL));

  node->type = PYRE_GRAPH_NODE_TYPE_INTERNAL_MEMCPY;
  node->dependency_count = dep_count;
  if (dep_count > 0) {
    memcpy(node->dependencies, deps, dep_count * sizeof(*deps));
  }

  pyre_graph_memcpy_node_attrs_internal_t* m = &node->attrs.memcpy;
  m->dst_ref = (iree_hal_buffer_ref_t){
      .buffer = dst_buf->hal_buffer,
      .offset = (iree_device_size_t)dst_offset,
      .length = (iree_device_size_t)attrs->size,
  };
  m->src_ref = (iree_hal_buffer_ref_t){
      .buffer = src_buf->hal_buffer,
      .offset = (iree_device_size_t)src_offset,
      .length = (iree_device_size_t)attrs->size,
  };
  m->size = attrs->size;
  m->flags = 0;

  iree_status_t status = pyre_graph_add_node_internal(graph, node);
  if (iree_status_is_ok(status) && out_node) {
    *out_node = node;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

pyre_status_t pyre_graph_add_memset_node(
    pyre_graph_t graph, const pyre_graph_node_t* deps, size_t dep_count,
    const pyre_graph_memset_node_attrs_t* attrs,
    pyre_graph_node_t* out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(attrs);
  IREE_TRACE_ZONE_BEGIN(z0);

  pyre_buffer_t dst_buf = NULL;
  size_t dst_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, pyre_buffer_table_find(&graph->device->buffer_table,
                                 (uint64_t)(uintptr_t)attrs->dst,
                                 &dst_buf, &dst_offset, NULL),
      "resolving dst buffer %p", attrs->dst);

  pyre_graph_node_s* node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, pyre_graph_allocate_node(graph->arena_allocator, dep_count, 0, &node,
                                   NULL));

  node->type = PYRE_GRAPH_NODE_TYPE_INTERNAL_MEMSET;
  node->dependency_count = dep_count;
  if (dep_count > 0) {
    memcpy(node->dependencies, deps, dep_count * sizeof(*deps));
  }

  pyre_graph_memset_node_attrs_internal_t* m = &node->attrs.memset;
  m->dst_ref = (iree_hal_buffer_ref_t){
      .buffer = dst_buf->hal_buffer,
      .offset = (iree_device_size_t)dst_offset,
      .length = (iree_device_size_t)attrs->count,
  };
  m->pattern = attrs->value;
  m->pattern_size = 1;
  m->count = attrs->count;
  m->flags = 0;

  iree_status_t status = pyre_graph_add_node_internal(graph, node);
  if (iree_status_is_ok(status) && out_node) {
    *out_node = node;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

pyre_status_t pyre_graph_add_host_call_node(
    pyre_graph_t graph, const pyre_graph_node_t* deps, size_t dep_count,
    const pyre_graph_host_call_node_attrs_t* attrs,
    pyre_graph_node_t* out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(attrs);
  IREE_ASSERT_ARGUMENT(attrs->fn);
  IREE_TRACE_ZONE_BEGIN(z0);

  pyre_graph_node_s* node = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, pyre_graph_allocate_node(graph->arena_allocator, dep_count, 0, &node,
                                   NULL));

  node->type = PYRE_GRAPH_NODE_TYPE_INTERNAL_HOST_CALL;
  node->dependency_count = dep_count;
  if (dep_count > 0) {
    memcpy(node->dependencies, deps, dep_count * sizeof(*deps));
  }

  pyre_graph_host_call_node_attrs_internal_t* h = &node->attrs.host;
  h->fn = (void (*)(void*))attrs->fn;
  h->user_data = attrs->user_data;

  iree_status_t status = pyre_graph_add_node_internal(graph, node);
  if (iree_status_is_ok(status) && out_node) {
    *out_node = node;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

pyre_status_t pyre_graph_add_dependencies(pyre_graph_t graph,
                                          const pyre_graph_node_t* from_nodes,
                                          const pyre_graph_node_t* to_nodes,
                                          size_t count) {
  IREE_ASSERT_ARGUMENT(graph);
  if (count == 0) return iree_ok_status();
  IREE_ASSERT_ARGUMENT(from_nodes);
  IREE_ASSERT_ARGUMENT(to_nodes);
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < count; ++i) {
    if (!from_nodes[i] || !to_nodes[i]) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "null node in dependency list at index %" PRIhsz,
                              i);
    }

    pyre_graph_edge_t* edge = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_malloc(graph->arena_allocator,
                                  sizeof(pyre_graph_edge_t), (void**)&edge));

    edge->from = from_nodes[i];
    edge->to = to_nodes[i];
    edge->next = graph->additional_edges;
    graph->additional_edges = edge;
    ++graph->additional_edge_count;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Instantiation
//===----------------------------------------------------------------------===//

pyre_status_t pyre_graph_instantiate(pyre_graph_t graph, uint32_t flags,
                                     pyre_graph_exec_t* out_exec) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(out_exec);
  *out_exec = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  pyre_graph_exec_s* exec = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(iree_allocator_system(), sizeof(*exec),
                                (void**)&exec));

  iree_atomic_ref_count_init(&exec->ref_count);
  exec->device = graph->device;
  exec->graph = graph;
  pyre_graph_retain(graph);
  iree_arena_initialize(&graph->device->block_pool, &exec->arena_allocator);
  exec->blocks = NULL;
  exec->block_count = 0;
  exec->semaphores = NULL;
  exec->semaphore_count = 0;
  exec->semaphore_base_values = NULL;
  exec->resource_set = NULL;
  exec->flags = flags;
  iree_slim_mutex_initialize(&exec->mutex);

  iree_status_t status = iree_hal_resource_set_allocate(
      &graph->device->block_pool, &exec->resource_set);

  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&graph->mutex);
    status = pyre_graph_exec_instantiate_locked(exec, graph->node_blocks,
                                                graph->node_count);
    iree_slim_mutex_unlock(&graph->mutex);
  }

  if (iree_status_is_ok(status)) {
    *out_exec = exec;
  } else {
    pyre_graph_exec_release(exec);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Graph exec lifecycle
//===----------------------------------------------------------------------===//

static void pyre_graph_exec_destroy(pyre_graph_exec_s* exec) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_resource_set_free(exec->resource_set);
  iree_arena_deinitialize(&exec->arena_allocator);
  pyre_graph_release(exec->graph);
  iree_slim_mutex_deinitialize(&exec->mutex);
  iree_allocator_free(iree_allocator_system(), exec);

  IREE_TRACE_ZONE_END(z0);
}

void pyre_graph_exec_retain(pyre_graph_exec_t exec) {
  if (exec) {
    iree_atomic_ref_count_inc(&exec->ref_count);
  }
}

void pyre_graph_exec_release(pyre_graph_exec_t exec) {
  if (exec && iree_atomic_ref_count_dec(&exec->ref_count) == 1) {
    pyre_graph_exec_destroy(exec);
  }
}

pyre_status_t pyre_graph_exec_update(pyre_graph_exec_t exec,
                                     pyre_graph_t graph) {
  IREE_ASSERT_ARGUMENT(exec);
  IREE_ASSERT_ARGUMENT(graph);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "graph update not yet implemented");
}

//===----------------------------------------------------------------------===//
// Stream capture
//===----------------------------------------------------------------------===//

pyre_status_t pyre_stream_begin_capture(pyre_stream_t stream,
                                        pyre_capture_mode_t mode) {
  IREE_ASSERT_ARGUMENT(stream);
  (void)mode;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "stream capture not yet implemented in libpyre");
}

pyre_status_t pyre_stream_end_capture(pyre_stream_t stream,
                                      pyre_graph_t* out_graph) {
  IREE_ASSERT_ARGUMENT(stream);
  (void)out_graph;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "stream capture not yet implemented in libpyre");
}

pyre_capture_status_t pyre_stream_capture_status(pyre_stream_t stream) {
  (void)stream;
  return PYRE_CAPTURE_STATUS_NONE;
}

bool pyre_stream_is_capturing(pyre_stream_t stream) {
  (void)stream;
  return false;
}
