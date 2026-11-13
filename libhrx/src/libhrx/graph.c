// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "buffer_table.h"
#include "hrx_internal.h"

//===----------------------------------------------------------------------===//
// Node and block allocation helpers
//===----------------------------------------------------------------------===//

static iree_status_t hrx_graph_allocate_node(iree_allocator_t allocator,
                                             iree_host_size_t dependency_count,
                                             iree_host_size_t extra_data_size,
                                             hrx_graph_node_s** out_node,
                                             uint8_t** out_extra_data) {
  IREE_ASSERT_ARGUMENT(out_node);
  *out_node = NULL;
  if (out_extra_data) {
    *out_extra_data = NULL;
  }

  iree_host_size_t dependencies_size = 0;
  iree_host_size_t total_size = 0;
  if (dependency_count > UINT32_MAX ||
      !iree_host_size_checked_mul(dependency_count, sizeof(hrx_graph_node_s*),
                                  &dependencies_size) ||
      !iree_host_size_checked_add(sizeof(hrx_graph_node_s), dependencies_size,
                                  &total_size)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "graph node dependencies exceed addressable storage");
  }
  iree_host_size_t extra_data_offset = 0;
  if (extra_data_size) {
    if (!iree_host_size_checked_add(total_size, iree_max_align_t - 1,
                                    &total_size)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "graph node payload alignment overflow");
    }
    extra_data_offset = total_size & ~(iree_max_align_t - 1);
    if (!iree_host_size_checked_add(extra_data_offset, extra_data_size,
                                    &total_size)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "graph node payload exceeds addressable storage");
    }
  }
  hrx_graph_node_s* node = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, total_size, (void**)&node));
  *out_node = node;
  if (out_extra_data && extra_data_size) {
    *out_extra_data = (uint8_t*)node + extra_data_offset;
  }

  return iree_ok_status();
}

static iree_status_t hrx_graph_allocate_node_block(
    iree_allocator_t allocator, iree_host_size_t capacity,
    hrx_graph_node_block_t** out_block) {
  const iree_host_size_t block_size =
      sizeof(hrx_graph_node_block_t) + capacity * sizeof(hrx_graph_node_s*);

  hrx_graph_node_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, block_size, (void**)&block));

  block->next = NULL;
  block->capacity = capacity;
  block->count = 0;
  *out_block = block;
  return iree_ok_status();
}

static iree_status_t hrx_graph_add_node_internal(hrx_graph_s* graph,
                                                 hrx_graph_node_s* node) {
  if (graph->node_count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph node ordinals exhausted");
  }
  node->node_index = (uint32_t)graph->node_count;

  if (!graph->current_node_block ||
      graph->current_node_block->count >= graph->current_node_block->capacity) {
    const iree_host_size_t block_capacity = graph->node_count < 64 ? 16 : 64;
    hrx_graph_node_block_t* new_block = NULL;
    IREE_RETURN_IF_ERROR(hrx_graph_allocate_node_block(
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

  return iree_ok_status();
}

// Captures the explicit native buffer handle. Its source pool and allocation
// epoch remain the caller's responsibility.
static iree_status_t hrx_graph_capture_buffer(hrx_graph_s* graph,
                                              hrx_buffer_t buffer) {
  if (!buffer) {
    return iree_ok_status();
  }
  void* buffer_resource = buffer->hal_buffer;
  return iree_hal_resource_set_insert(graph->node_resource_set, 1,
                                      &buffer_resource);
}

//===----------------------------------------------------------------------===//
// hrx_graph_t (template)
//===----------------------------------------------------------------------===//

static void hrx_graph_destroy(hrx_graph_s* graph);

hrx_status_t hrx_graph_create(hrx_device_t device, uint32_t flags,
                              hrx_graph_t* out_graph) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_graph);
  *out_graph = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  hrx_graph_s* graph = NULL;
  HRX_RETURN_AND_END_ZONE_IF_IREE_ERROR(
      z0, iree_allocator_malloc(iree_allocator_system(), sizeof(*graph),
                                (void**)&graph));

  iree_atomic_ref_count_init(&graph->ref_count);
  graph->device = device;
  hrx_device_retain(device);

  iree_arena_initialize(&device->block_pool, &graph->arena);
  graph->arena_allocator = iree_arena_allocator(&graph->arena);

  graph->node_resource_set = NULL;

  graph->node_blocks = NULL;
  graph->current_node_block = NULL;
  graph->node_count = 0;
  graph->additional_edges = NULL;
  graph->additional_edge_count = 0;
  graph->flags = flags;

  iree_status_t status = iree_hal_resource_set_allocate(
      &device->block_pool, &graph->node_resource_set);
  if (iree_status_is_ok(status)) {
    *out_graph = graph;
  } else {
    hrx_graph_destroy(graph);
  }
  IREE_TRACE_ZONE_END(z0);
  return hrx_status_from_iree(status);
}

static void hrx_graph_destroy(hrx_graph_s* graph) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_resource_set_free(graph->node_resource_set);
  iree_arena_deinitialize(&graph->arena);
  hrx_device_release(graph->device);
  iree_allocator_free(iree_allocator_system(), graph);

  IREE_TRACE_ZONE_END(z0);
}

void hrx_graph_retain(hrx_graph_t graph) {
  if (graph) {
    iree_atomic_ref_count_inc(&graph->ref_count);
  }
}

void hrx_graph_release(hrx_graph_t graph) {
  if (graph && iree_atomic_ref_count_dec(&graph->ref_count) == 1) {
    hrx_graph_destroy(graph);
  }
}

hrx_status_t hrx_graph_size(hrx_graph_t graph, size_t* out_count) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(out_count);
  *out_count = graph->node_count;
  return hrx_ok_status();
}

hrx_status_t hrx_graph_get_nodes(hrx_graph_t graph, hrx_graph_node_t* nodes,
                                 size_t* inout_count) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(inout_count);

  size_t requested = *inout_count;
  if (requested > graph->node_count) {
    requested = graph->node_count;
  }

  iree_host_size_t copied = 0;
  hrx_graph_node_block_t* block = graph->node_blocks;
  while (block && copied < requested) {
    iree_host_size_t to_copy = block->count;
    if (copied + to_copy > requested) {
      to_copy = requested - copied;
    }
    for (iree_host_size_t i = 0; i < to_copy; i++) {
      nodes[copied++] = block->nodes[i];
    }
    block = block->next;
  }

  *inout_count = copied;
  return hrx_ok_status();
}

//===----------------------------------------------------------------------===//
// Node creation
//===----------------------------------------------------------------------===//

hrx_status_t hrx_graph_add_empty_node(hrx_graph_t graph,
                                      const hrx_graph_node_t* deps,
                                      size_t dep_count,
                                      hrx_graph_node_t* out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_TRACE_ZONE_BEGIN(z0);

  hrx_graph_node_s* node = NULL;
  HRX_RETURN_AND_END_ZONE_IF_IREE_ERROR(
      z0, hrx_graph_allocate_node(graph->arena_allocator, dep_count, 0, &node,
                                  NULL));

  node->type = HRX_GRAPH_NODE_TYPE_INTERNAL_EMPTY;
  node->dependency_count = dep_count;
  if (dep_count > 0) {
    memcpy(node->dependencies, deps, dep_count * sizeof(*deps));
  }

  iree_status_t status = hrx_graph_add_node_internal(graph, node);
  if (iree_status_is_ok(status) && out_node) {
    *out_node = node;
  }
  IREE_TRACE_ZONE_END(z0);
  return hrx_status_from_iree(status);
}

hrx_status_t hrx_graph_add_kernel_node(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    const hrx_graph_kernel_node_attrs_t* attrs, hrx_graph_node_t* out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(attrs);
  IREE_ASSERT_ARGUMENT(attrs->executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_dispatch_flags_t hal_flags = IREE_HAL_DISPATCH_FLAG_NONE;
  hrx_status_t flag_status =
      hrx_iree_dispatch_flags_from_hrx(attrs->flags, &hal_flags);
  if (!hrx_status_is_ok(flag_status)) {
    HRX_RETURN_AND_END_ZONE(z0, flag_status);
  }

  const iree_host_size_t constants_size =
      iree_host_align(attrs->constants_size, iree_max_align_t);
  const iree_host_size_t bindings_size = iree_host_align(
      attrs->binding_count * sizeof(iree_hal_buffer_ref_t), iree_max_align_t);
  uint8_t* extra_data = NULL;
  hrx_graph_node_s* node = NULL;
  HRX_RETURN_AND_END_ZONE_IF_IREE_ERROR(
      z0, hrx_graph_allocate_node(graph->arena_allocator, dep_count,
                                  constants_size + bindings_size, &node,
                                  &extra_data));

  node->type = HRX_GRAPH_NODE_TYPE_INTERNAL_KERNEL;
  node->dependency_count = dep_count;
  if (dep_count > 0) {
    memcpy(node->dependencies, deps, dep_count * sizeof(*deps));
  }

  hrx_graph_kernel_node_attrs_internal_t* k = &node->attrs.kernel;
  k->executable = attrs->executable->hal_executable;
  k->export_ordinal = attrs->export_ordinal;
  memcpy(k->grid_dim, attrs->config.workgroup_count, sizeof(k->grid_dim));
  memcpy(k->block_dim, attrs->config.workgroup_size, sizeof(k->block_dim));
  k->flags = hal_flags;

  void* constants_copy = extra_data;
  if (attrs->constants_size > 0 && attrs->constants) {
    memcpy(constants_copy, attrs->constants, attrs->constants_size);
  }
  k->constants =
      iree_make_const_byte_span(constants_copy, attrs->constants_size);

  iree_hal_buffer_ref_t* bindings_data =
      (iree_hal_buffer_ref_t*)(extra_data + constants_size);
  for (size_t i = 0; i < attrs->binding_count; i++) {
    hrx_buffer_s* buf = attrs->bindings[i].buffer;
    bindings_data[i] = (iree_hal_buffer_ref_t){
        .buffer = buf ? buf->hal_buffer : NULL,
        .offset = (iree_device_size_t)attrs->bindings[i].offset,
        .length = (iree_device_size_t)attrs->bindings[i].length,
    };
  }
  k->bindings.count = attrs->binding_count;
  k->bindings.values = bindings_data;

  void* executable_resource = k->executable;
  iree_status_t status = iree_hal_resource_set_insert(graph->node_resource_set,
                                                      1, &executable_resource);
  for (size_t i = 0; i < attrs->binding_count && iree_status_is_ok(status);
       ++i) {
    status = hrx_graph_capture_buffer(graph, attrs->bindings[i].buffer);
  }
  if (iree_status_is_ok(status)) {
    status = hrx_graph_add_node_internal(graph, node);
  }
  if (iree_status_is_ok(status) && out_node) {
    *out_node = node;
  }
  IREE_TRACE_ZONE_END(z0);
  return hrx_status_from_iree(status);
}

static hrx_status_t hrx_graph_validate_buffer_ref(hrx_graph_t graph,
                                                  hrx_buffer_ref_t ref,
                                                  const char* name) {
  if (!ref.buffer) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, name);
  }
  if (ref.buffer->group != graph->device->group) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "graph buffer belongs to another device group");
  }
  if (ref.length == 0 || ref.offset > ref.buffer->size ||
      ref.length > ref.buffer->size - ref.offset) {
    return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                           "graph buffer reference is out of range");
  }
  return hrx_ok_status();
}

static hrx_status_t hrx_graph_add_copy_buffer_node_internal(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    hrx_buffer_ref_t src, hrx_buffer_ref_t dst, hrx_graph_node_t* out_node) {
  hrx_graph_node_s* node = NULL;
  iree_status_t allocation_status = hrx_graph_allocate_node(
      graph->arena_allocator, dep_count, 0, &node, NULL);
  if (!iree_status_is_ok(allocation_status)) {
    return hrx_status_from_iree(allocation_status);
  }

  node->type = HRX_GRAPH_NODE_TYPE_INTERNAL_MEMCPY;
  node->dependency_count = dep_count;
  if (dep_count > 0) {
    memcpy(node->dependencies, deps, dep_count * sizeof(*deps));
  }

  hrx_graph_memcpy_node_attrs_internal_t* m = &node->attrs.memcpy;
  m->dst_ref = (iree_hal_buffer_ref_t){
      .buffer = dst.buffer->hal_buffer,
      .offset = (iree_device_size_t)dst.offset,
      .length = (iree_device_size_t)dst.length,
  };
  m->src_ref = (iree_hal_buffer_ref_t){
      .buffer = src.buffer->hal_buffer,
      .offset = (iree_device_size_t)src.offset,
      .length = (iree_device_size_t)src.length,
  };
  m->size = src.length;
  m->flags = 0;

  iree_status_t status = hrx_graph_capture_buffer(graph, src.buffer);
  if (iree_status_is_ok(status)) {
    status = hrx_graph_capture_buffer(graph, dst.buffer);
  }
  if (iree_status_is_ok(status)) {
    status = hrx_graph_add_node_internal(graph, node);
  }
  if (iree_status_is_ok(status) && out_node) {
    *out_node = node;
  }
  return hrx_status_from_iree(status);
}

static hrx_status_t hrx_graph_add_fill_buffer_node_internal(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    hrx_buffer_ref_t dst, uint32_t pattern, size_t pattern_size,
    hrx_graph_node_t* out_node) {
  hrx_graph_node_s* node = NULL;
  iree_status_t allocation_status = hrx_graph_allocate_node(
      graph->arena_allocator, dep_count, 0, &node, NULL);
  if (!iree_status_is_ok(allocation_status)) {
    return hrx_status_from_iree(allocation_status);
  }

  node->type = HRX_GRAPH_NODE_TYPE_INTERNAL_MEMSET;
  node->dependency_count = dep_count;
  if (dep_count > 0) {
    memcpy(node->dependencies, deps, dep_count * sizeof(*deps));
  }

  hrx_graph_memset_node_attrs_internal_t* m = &node->attrs.memset;
  m->dst_ref = (iree_hal_buffer_ref_t){
      .buffer = dst.buffer->hal_buffer,
      .offset = (iree_device_size_t)dst.offset,
      .length = (iree_device_size_t)dst.length,
  };
  m->pattern = pattern;
  m->pattern_size = (uint8_t)pattern_size;
  m->count = dst.length;
  m->flags = 0;

  iree_status_t status = hrx_graph_capture_buffer(graph, dst.buffer);
  if (iree_status_is_ok(status)) {
    status = hrx_graph_add_node_internal(graph, node);
  }
  if (iree_status_is_ok(status) && out_node) {
    *out_node = node;
  }
  return hrx_status_from_iree(status);
}

hrx_status_t hrx_graph_add_copy_buffer_node(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    const hrx_graph_copy_buffer_node_attrs_t* attrs,
    hrx_graph_node_t* out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(attrs);
  IREE_TRACE_ZONE_BEGIN(z0);
  hrx_status_t status = hrx_graph_validate_buffer_ref(
      graph, attrs->src, "copy source buffer is NULL");
  if (hrx_status_is_ok(status)) {
    status = hrx_graph_validate_buffer_ref(graph, attrs->dst,
                                           "copy destination buffer is NULL");
  }
  if (hrx_status_is_ok(status) && attrs->src.length != attrs->dst.length) {
    status = hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                             "copy source and destination lengths differ");
  }
  if (hrx_status_is_ok(status)) {
    status = hrx_graph_add_copy_buffer_node_internal(
        graph, deps, dep_count, attrs->src, attrs->dst, out_node);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

hrx_status_t hrx_graph_add_fill_buffer_node(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    const hrx_graph_fill_buffer_node_attrs_t* attrs,
    hrx_graph_node_t* out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(attrs);
  IREE_TRACE_ZONE_BEGIN(z0);
  hrx_status_t status = hrx_graph_validate_buffer_ref(
      graph, attrs->dst, "fill destination buffer is NULL");
  if (hrx_status_is_ok(status) && attrs->pattern_size != 1 &&
      attrs->pattern_size != 2 && attrs->pattern_size != 4) {
    status = hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                             "fill pattern size must be 1, 2, or 4");
  }
  if (hrx_status_is_ok(status) &&
      (attrs->dst.offset % attrs->pattern_size != 0 ||
       attrs->dst.length % attrs->pattern_size != 0)) {
    status = hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                             "fill range is not aligned to its pattern size");
  }
  if (hrx_status_is_ok(status)) {
    status = hrx_graph_add_fill_buffer_node_internal(
        graph, deps, dep_count, attrs->dst, attrs->pattern, attrs->pattern_size,
        out_node);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

hrx_status_t hrx_graph_add_copy_ptr_node(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    const hrx_graph_copy_ptr_node_attrs_t* attrs, hrx_graph_node_t* out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(attrs);
  IREE_TRACE_ZONE_BEGIN(z0);
  hrx_buffer_t dst_buf = NULL;
  size_t dst_offset = 0;
  HRX_RETURN_AND_END_ZONE(
      z0, hrx_buffer_table_find(&graph->device->buffer_table,
                                (uint64_t)(uintptr_t)attrs->dst, &dst_buf,
                                &dst_offset, NULL));
  hrx_buffer_t src_buf = NULL;
  size_t src_offset = 0;
  HRX_RETURN_AND_END_ZONE(
      z0, hrx_buffer_table_find(&graph->device->buffer_table,
                                (uint64_t)(uintptr_t)attrs->src, &src_buf,
                                &src_offset, NULL));
  const hrx_graph_copy_buffer_node_attrs_t native_attrs = {
      .src = {src_buf, src_offset, attrs->size},
      .dst = {dst_buf, dst_offset, attrs->size},
  };
  hrx_status_t status = hrx_graph_add_copy_buffer_node(graph, deps, dep_count,
                                                       &native_attrs, out_node);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

hrx_status_t hrx_graph_add_fill_ptr_node(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    const hrx_graph_fill_ptr_node_attrs_t* attrs, hrx_graph_node_t* out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(attrs);
  IREE_TRACE_ZONE_BEGIN(z0);
  hrx_buffer_t dst_buf = NULL;
  size_t dst_offset = 0;
  HRX_RETURN_AND_END_ZONE(
      z0, hrx_buffer_table_find(&graph->device->buffer_table,
                                (uint64_t)(uintptr_t)attrs->dst, &dst_buf,
                                &dst_offset, NULL));
  const hrx_graph_fill_buffer_node_attrs_t native_attrs = {
      .dst = {dst_buf, dst_offset, attrs->count},
      .pattern = attrs->value,
      .pattern_size = 1,
  };
  hrx_status_t status = hrx_graph_add_fill_buffer_node(graph, deps, dep_count,
                                                       &native_attrs, out_node);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

hrx_status_t hrx_graph_add_host_call_node(
    hrx_graph_t graph, const hrx_graph_node_t* deps, size_t dep_count,
    const hrx_graph_host_call_node_attrs_t* attrs, hrx_graph_node_t* out_node) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(attrs);
  IREE_ASSERT_ARGUMENT(attrs->fn);
  IREE_TRACE_ZONE_BEGIN(z0);

  hrx_graph_node_s* node = NULL;
  HRX_RETURN_AND_END_ZONE_IF_IREE_ERROR(
      z0, hrx_graph_allocate_node(graph->arena_allocator, dep_count, 0, &node,
                                  NULL));

  node->type = HRX_GRAPH_NODE_TYPE_INTERNAL_HOST_CALL;
  node->dependency_count = dep_count;
  if (dep_count > 0) {
    memcpy(node->dependencies, deps, dep_count * sizeof(*deps));
  }

  hrx_graph_host_call_node_attrs_internal_t* h = &node->attrs.host;
  h->fn = attrs->fn;
  h->user_data = attrs->user_data;

  iree_status_t status = hrx_graph_add_node_internal(graph, node);
  if (iree_status_is_ok(status) && out_node) {
    *out_node = node;
  }
  IREE_TRACE_ZONE_END(z0);
  return hrx_status_from_iree(status);
}

hrx_status_t hrx_graph_add_dependencies(hrx_graph_t graph,
                                        const hrx_graph_node_t* from_nodes,
                                        const hrx_graph_node_t* to_nodes,
                                        size_t count) {
  IREE_ASSERT_ARGUMENT(graph);
  if (count == 0) {
    return hrx_ok_status();
  }
  IREE_ASSERT_ARGUMENT(from_nodes);
  IREE_ASSERT_ARGUMENT(to_nodes);
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < count; ++i) {
    if (!from_nodes[i] || !to_nodes[i]) {
      IREE_TRACE_ZONE_END(z0);
      return hrx_status_from_iree(iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "null node in dependency list at index %" PRIhsz, i));
    }

    hrx_graph_edge_t* edge = NULL;
    HRX_RETURN_AND_END_ZONE_IF_IREE_ERROR(
        z0, iree_allocator_malloc(graph->arena_allocator,
                                  sizeof(hrx_graph_edge_t), (void**)&edge));

    edge->from = from_nodes[i];
    edge->to = to_nodes[i];
    edge->next = graph->additional_edges;
    graph->additional_edges = edge;
    ++graph->additional_edge_count;
  }

  IREE_TRACE_ZONE_END(z0);
  return hrx_ok_status();
}

// Normalization is the native graph input boundary. The shared analyzer
// receives dense, in-graph ordinals and does not repeat handle validation at
// every phase.
iree_status_t hrx_graph_schedule_nodes(hrx_graph_node_block_t* node_blocks,
                                       iree_host_size_t node_count,
                                       hrx_graph_edge_t* additional_edges,
                                       iree_arena_allocator_t* arena,
                                       hrx_graph_schedule_t* out_schedule) {
  hrx_graph_analysis_t analysis;
  IREE_RETURN_IF_ERROR(
      hrx_graph_analysis_initialize(node_count, arena, &analysis));
  uint64_t edge_count = 0;
  for (hrx_graph_node_block_t* block = node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      hrx_graph_node_s* node = block->nodes[i];
      const uint32_t index = node->node_index;
      analysis.sources[index] = node;
      analysis.offsets[index + 1] = node->dependency_count;
      edge_count += node->dependency_count;
      analysis.flags[index] =
          hrx_graph_node_is_recordable(node->type)
              ? HRX_GRAPH_ANALYSIS_NODE_RECORDABLE
          : node->type == HRX_GRAPH_NODE_TYPE_INTERNAL_HOST_CALL
              ? HRX_GRAPH_ANALYSIS_NODE_ISOLATED
              : HRX_GRAPH_ANALYSIS_NODE_EMPTY;
    }
  }
  iree_status_t status = iree_ok_status();
  for (const hrx_graph_edge_t* edge = additional_edges;
       iree_status_is_ok(status) && edge; edge = edge->next) {
    if (!edge->from || !edge->to || edge->from->node_index >= node_count ||
        edge->to->node_index >= node_count ||
        analysis.sources[edge->from->node_index] != edge->from ||
        analysis.sources[edge->to->node_index] != edge->to) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "dependency node does not belong to the graph");
    } else {
      ++analysis.offsets[edge->to->node_index + 1];
      ++edge_count;
    }
  }
  IREE_RETURN_IF_ERROR(status);
  if (edge_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph has too many dependencies");
  }
  IREE_RETURN_IF_ERROR(
      hrx_graph_analysis_allocate_dependencies(&analysis, arena));
  bool is_topologically_ordered = true;
  for (uint32_t i = 0; iree_status_is_ok(status) && i < analysis.node_count;
       ++i) {
    const hrx_graph_node_s* node = analysis.sources[i];
    for (uint32_t j = 0;
         iree_status_is_ok(status) && j < node->dependency_count; ++j) {
      const hrx_graph_node_s* predecessor = node->dependencies[j];
      if (!predecessor || predecessor->node_index >= node_count ||
          analysis.sources[predecessor->node_index] != predecessor) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "dependency node does not belong to the graph");
      } else {
        analysis.dependencies[analysis.order[i]++] = predecessor->node_index;
        is_topologically_ordered &= predecessor->node_index < i;
      }
    }
  }
  for (const hrx_graph_edge_t* edge = additional_edges;
       iree_status_is_ok(status) && edge; edge = edge->next) {
    analysis.dependencies[analysis.order[edge->to->node_index]++] =
        edge->from->node_index;
    is_topologically_ordered &= edge->from->node_index < edge->to->node_index;
  }
  analysis.is_topologically_ordered = is_topologically_ordered;
  if (iree_status_is_ok(status)) {
    status = hrx_graph_analyze(&analysis, arena, out_schedule);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Stream capture
//===----------------------------------------------------------------------===//

hrx_status_t hrx_stream_begin_capture(hrx_stream_t stream,
                                      hrx_capture_mode_t mode) {
  IREE_ASSERT_ARGUMENT(stream);
  (void)mode;
  return hrx_make_status(HRX_STATUS_UNIMPLEMENTED,
                         "stream capture not yet implemented in libhrx");
}

hrx_status_t hrx_stream_end_capture(hrx_stream_t stream,
                                    hrx_graph_t* out_graph) {
  IREE_ASSERT_ARGUMENT(stream);
  (void)out_graph;
  return hrx_make_status(HRX_STATUS_UNIMPLEMENTED,
                         "stream capture not yet implemented in libhrx");
}

hrx_capture_status_t hrx_stream_capture_status(hrx_stream_t stream) {
  (void)stream;
  return HRX_CAPTURE_STATUS_NONE;
}

bool hrx_stream_is_capturing(hrx_stream_t stream) {
  (void)stream;
  return false;
}
