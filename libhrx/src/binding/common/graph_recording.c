// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/graph_recording.h"

#include "common/context.h"
#include "common/device.h"
#include "common/event.h"
#include "common/graph.h"
#include "libhrx/src/libhrx/device.h"

typedef struct iree_hal_streaming_graph_compiler_t {
  // Borrowed immutable compilation inputs.
  const iree_hal_streaming_graph_compile_params_t* params;
  // Recording under construction, never visible to a launch.
  iree_hal_streaming_graph_recording_t* recording;
} iree_hal_streaming_graph_compiler_t;

typedef struct iree_hal_streaming_graph_memory_claim_t {
  // Next admission held by this compiled graph version.
  struct iree_hal_streaming_graph_memory_claim_t* next;
  // Retained original allocation owner, including after recording replacement.
  iree_hal_streaming_graph_t* graph;
} iree_hal_streaming_graph_memory_claim_t;

// Compilation acquires each original allocation owner once. Rebuilds and
// nested graphs share the same logical executable identity, while independent
// executables cannot concurrently reuse these fixed graph allocation addresses.
static iree_status_t iree_hal_streaming_graph_memory_acquire_claim(
    iree_hal_streaming_graph_compiler_t* compiler,
    iree_hal_streaming_graph_t* graph) {
  for (iree_hal_streaming_graph_memory_claim_t* claim =
           compiler->recording->memory_claims;
       claim; claim = claim->next) {
    if (claim->graph == graph) {
      return iree_ok_status();
    }
  }
  iree_hal_streaming_graph_memory_claim_t* claim = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      &compiler->recording->arena_allocator, sizeof(*claim), (void**)&claim));
  iree_slim_mutex_lock(&graph->capture_mutex);
  const bool available =
      !graph->memory_admission.executable ||
      graph->memory_admission.executable == compiler->params->execution_owner;
  if (available) {
    graph->memory_admission.executable = compiler->params->execution_owner;
    ++graph->memory_admission.count;
  }
  iree_slim_mutex_unlock(&graph->capture_mutex);
  if (!available) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "graph allocation nodes already belong to another executable");
  }
  iree_hal_streaming_graph_retain(graph);
  claim->graph = graph;
  claim->next = compiler->recording->memory_claims;
  compiler->recording->memory_claims = claim;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_memory_acquire_claims(
    iree_hal_streaming_graph_compiler_t* compiler) {
  for (iree_hal_streaming_node_block_t* block =
           compiler->params->graph->node_blocks;
       block; block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC) {
        IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_memory_acquire_claim(
            compiler,
            iree_hal_streaming_graph_node_mem_alloc(node)->allocation_owner
                ? iree_hal_streaming_graph_node_mem_alloc(node)
                      ->allocation_owner
                : node->graph));
      }
    }
  }
  return iree_ok_status();
}

static void iree_hal_streaming_graph_recording_release_children(
    iree_hal_streaming_graph_recording_t* recording) {
  for (uint32_t i = 0; recording->blocks && i < recording->block_count; ++i) {
    hrx_graph_block_t* block = recording->blocks[i];
    switch (block->type) {
      case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH:
        iree_hal_streaming_graph_exec_release(
            iree_hal_streaming_graph_block_child_graph(block)->exec);
        iree_hal_streaming_graph_block_child_graph(block)->exec = NULL;
        break;
      case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD:
      case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_WAIT:
        iree_hal_streaming_event_release(
            iree_hal_streaming_graph_block_event(block)->event);
        iree_hal_streaming_graph_block_event(block)->event = NULL;
        break;
      default:
        break;
    }
  }
}

static void iree_hal_streaming_graph_memory_release_claims(
    iree_hal_streaming_graph_recording_t* recording) {
  for (iree_hal_streaming_graph_memory_claim_t* claim =
           recording->memory_claims;
       claim; claim = claim->next) {
    iree_hal_streaming_graph_t* graph = claim->graph;
    iree_slim_mutex_lock(&graph->capture_mutex);
    if (--graph->memory_admission.count == 0) {
      graph->memory_admission.executable = NULL;
    }
    iree_slim_mutex_unlock(&graph->capture_mutex);
    iree_hal_streaming_graph_release(graph);
  }
  recording->memory_claims = NULL;
}

static iree_status_t iree_hal_streaming_graph_block_allocate(
    iree_arena_allocator_t* arena_allocator,
    iree_hal_streaming_graph_block_type_t type,
    hrx_graph_dependencies_t dependencies, hrx_graph_block_t** out_block) {
  static const uint16_t payload_sizes[] = {
      sizeof(hrx_graph_barrier_block_t),
      sizeof(hrx_graph_execute_block_t),
      sizeof(iree_hal_streaming_graph_host_call_block_attrs_t),
      sizeof(iree_hal_streaming_graph_event_block_attrs_t),
      sizeof(iree_hal_streaming_graph_event_block_attrs_t),
      sizeof(iree_hal_streaming_graph_dispatch_block_attrs_t),
      sizeof(iree_hal_streaming_graph_memory_block_attrs_t),
      sizeof(iree_hal_streaming_graph_memory_block_attrs_t),
      sizeof(iree_hal_streaming_graph_child_graph_block_attrs_t),
  };
  return hrx_graph_block_allocate(arena_allocator, (uint8_t)type, dependencies,
                                  payload_sizes[type], out_block);
}

static iree_status_t iree_hal_streaming_graph_allocate_barrier_block(
    iree_hal_streaming_graph_compiler_t* compiler,
    hrx_graph_dependencies_t dependencies, iree_hal_queue_barrier_flags_t flags,
    hrx_graph_block_t** out_block) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the operation and its compact predecessor span.
  hrx_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &compiler->recording->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER, dependencies,
              &block));

  // Set barrier attributes.
  hrx_graph_barrier_block_t* attrs = hrx_graph_block_barrier(block);
  attrs->flags = flags;

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_allocate_host_call_block(
    iree_hal_streaming_graph_compiler_t* compiler,
    hrx_graph_dependencies_t dependencies,
    iree_hal_streaming_host_callback_t callback, void* user_data,
    const iree_hal_streaming_graph_node_t* source_node,
    iree_hal_host_call_flags_t flags, hrx_graph_block_t** out_block) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the operation and its compact predecessor span.
  hrx_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &compiler->recording->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL, dependencies,
              &block));

  // Set host call attributes.
  iree_hal_streaming_graph_host_call_block_attrs_t* attrs =
      iree_hal_streaming_graph_block_host_call(block);
  attrs->callback = callback;
  attrs->user_data = user_data;
  attrs->user_data_size =
      source_node
          ? iree_hal_streaming_graph_node_host(source_node)->user_data_size
          : 0;
  attrs->resource =
      source_node ? iree_hal_streaming_graph_node_host(source_node)->resource
                  : NULL;
  if (attrs->resource) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_resource_set_insert(compiler->recording->resource_set, 1,
                                         &attrs->resource));
  }
  if (source_node &&
      iree_hal_streaming_graph_node_host(source_node)->user_data_size > 0 &&
      user_data) {
    void* copied_data = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(
                &compiler->recording->arena_allocator,
                iree_hal_streaming_graph_node_host(source_node)->user_data_size,
                (void**)&copied_data));
    memcpy(copied_data, user_data,
           iree_hal_streaming_graph_node_host(source_node)->user_data_size);
    attrs->user_data = copied_data;
  }
  attrs->flags = flags;

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_allocate_event_block(
    iree_hal_streaming_graph_compiler_t* compiler,
    iree_hal_streaming_graph_block_type_t type,
    hrx_graph_dependencies_t dependencies,
    iree_hal_streaming_graph_node_t* source_node,
    iree_hal_streaming_event_t* event, hrx_graph_block_t** out_block) {
  IREE_TRACE_ZONE_BEGIN(z0);
  hrx_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_graph_block_allocate(
          &compiler->recording->arena_allocator, type, dependencies, &block));

  iree_hal_streaming_graph_block_event(block)->source_node = source_node;
  iree_hal_streaming_graph_block_event(block)->event = event;
  iree_hal_streaming_event_retain(event);
  // The one place a record block is built, and so where |compiler| learns it
  // holds one. A launch reads this to answer for every record at once.
  if (type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD) {
    compiler->recording->records_events = true;
  } else {
    compiler->recording->waits_for_events = true;
  }

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_allocate_dispatch_block(
    iree_hal_streaming_graph_compiler_t* compiler,
    hrx_graph_dependencies_t dependencies, iree_hal_executable_t* executable,
    iree_host_size_t entry_point, iree_hal_dispatch_config_t config,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags, hrx_graph_block_t** out_block) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the operation and its compact predecessor span.
  hrx_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &compiler->recording->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH, dependencies,
              &block));

  // Set dispatch attributes.
  iree_hal_streaming_graph_dispatch_block_attrs_t* attrs =
      iree_hal_streaming_graph_block_dispatch(block);
  attrs->executable = executable;
  attrs->entry_point = entry_point;
  attrs->config = config;
  attrs->flags = flags;

  // Copy constants if provided.
  if (constants.data_length > 0) {
    void* constants_copy = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(&compiler->recording->arena_allocator,
                                constants.data_length, &constants_copy));
    memcpy(constants_copy, constants.data, constants.data_length);
    attrs->constants =
        iree_make_const_byte_span(constants_copy, constants.data_length);
  } else {
    attrs->constants = iree_const_byte_span_empty();
  }

  // Copy bindings if provided.
  if (bindings.count > 0) {
    iree_hal_buffer_ref_t* bindings_copy = NULL;
    const iree_host_size_t bindings_size =
        bindings.count * sizeof(*bindings_copy);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(&compiler->recording->arena_allocator,
                                bindings_size, (void**)&bindings_copy));
    memcpy(bindings_copy, bindings.values, bindings_size);
    attrs->bindings.count = bindings.count;
    attrs->bindings.values = bindings_copy;

    // Add buffers to resource set.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_resource_set_insert_strided(
                compiler->recording->resource_set, bindings.count,
                bindings.values, offsetof(iree_hal_buffer_ref_t, buffer),
                sizeof(iree_hal_buffer_ref_t)));
  } else {
    attrs->bindings = iree_hal_buffer_ref_list_empty();
  }

  // Add executable to resource set.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(compiler->recording->resource_set, 1,
                                       &executable));

  if (iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_COOPERATIVE)) {
    compiler->recording->has_cooperative_dispatches = true;
  }
  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_allocate_execute_block(
    iree_hal_streaming_graph_compiler_t* compiler,
    hrx_graph_dependencies_t dependencies, iree_hal_queue_execute_flags_t flags,
    hrx_graph_block_t** out_block) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the operation and its compact predecessor span.
  hrx_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &compiler->recording->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE, dependencies,
              &block));

  hrx_graph_execute_block_t* attrs = hrx_graph_block_execute(block);
  attrs->flags = flags;

  // Parameter updates can replace this compiled graph while an older launch
  // is pending. Each command buffer retains its recorded buffers/executables
  // until all accepted submissions retire. Raw HIP pointers in kernel
  // arguments remain governed by the application's allocation lifetime.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_create(
              iree_hal_queue_family(compiler->params->context->queue),
              hrx_device_profile_command_buffer_mode(
                  compiler->params->context->device_entry->hrx_device),
              IREE_HAL_COMMAND_CATEGORY_TRANSFER |
                  IREE_HAL_COMMAND_CATEGORY_DISPATCH |
                  IREE_HAL_COMMAND_CATEGORY_ATOMIC,
              /*binding_capacity=*/0, &attrs->command_buffer));

  // Add to resource set for cleanup.
  iree_status_t status = iree_hal_resource_set_insert(
      compiler->recording->resource_set, 1, &attrs->command_buffer);

  // We don't technically retain a reference to it past here on the stack, just
  // in the resource set associated with the compiler.
  iree_hal_command_buffer_release(attrs->command_buffer);

  if (iree_status_is_ok(status)) {
    *out_block = block;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_streaming_graph_allocate_child_graph_block(
    iree_hal_streaming_graph_compiler_t* compiler,
    hrx_graph_dependencies_t dependencies,
    iree_hal_streaming_graph_t* child_graph, hrx_graph_block_t** out_block) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_graph_exec_t* child_exec = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_graph_exec_create(
          compiler->params->context, child_graph,
          (iree_hal_streaming_graph_instantiate_flags_t)compiler->params->flags,
          compiler->params->execution_owner, compiler->params->host_allocator,
          &child_exec));

  hrx_graph_block_t* block = NULL;
  iree_status_t status = iree_hal_streaming_graph_block_allocate(
      &compiler->recording->arena_allocator,
      IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH, dependencies, &block);
  if (iree_status_is_ok(status)) {
    iree_hal_streaming_graph_block_child_graph(block)->exec = child_exec;
    const iree_hal_streaming_graph_recording_t* child_recording =
        iree_hal_streaming_graph_exec_recording(child_exec);
    // A launch of this recording walks the child's blocks too, so the records
    // they hold are records of this launch. The child instantiated above
    // already carries its own children's, which makes the property transitive.
    compiler->recording->records_events |= child_recording->records_events;
    compiler->recording->waits_for_events |= child_recording->waits_for_events;
    compiler->recording->has_cooperative_dispatches |=
        child_recording->has_cooperative_dispatches;
    *out_block = block;
  } else {
    iree_hal_streaming_graph_exec_release(child_exec);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Helper to record nodes from a partition into a command buffer.
static iree_status_t iree_hal_streaming_graph_record_partition(
    iree_hal_streaming_graph_compiler_t* compiler,
    const hrx_graph_schedule_t* schedule, uint32_t node_start_index,
    uint32_t node_count, uint8_t stream_id,
    iree_hal_command_buffer_t* command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Begin recording command buffer.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_begin(command_buffer));

  // Scope the partition into a debug group.
  const iree_string_view_t label_name =
      iree_make_cstring_view("graph partition");
  const iree_hal_label_location_t* location = NULL;
  const iree_hal_label_color_t label_color = iree_hal_label_color_unspecified();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_begin_debug_group(command_buffer, label_name,
                                                    label_color, location));

  // A dependency orders both command execution and memory visibility. Graph
  // nodes can alternate between dispatches, transfers, and atomics, so make
  // writes from any operation class available to reads by any class.
  static const iree_hal_memory_barrier_t memory_barrier = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE |
                      IREE_HAL_ACCESS_SCOPE_ATOMIC_READ |
                      IREE_HAL_ACCESS_SCOPE_ATOMIC_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE |
                      IREE_HAL_ACCESS_SCOPE_ATOMIC_READ |
                      IREE_HAL_ACCESS_SCOPE_ATOMIC_WRITE,
  };
  const iree_hal_barrier_t execution_barrier = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH |
                           IREE_HAL_EXECUTION_STAGE_TRANSFER |
                           IREE_HAL_EXECUTION_STAGE_ATOMIC,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH |
                           IREE_HAL_EXECUTION_STAGE_TRANSFER |
                           IREE_HAL_EXECUTION_STAGE_ATOMIC,
      .memory_barrier_count = 1,
      .memory_barriers = &memory_barrier,
  };
  iree_hal_barrier_flags_t release_flags = IREE_HAL_BARRIER_FLAG_NONE;
  iree_status_t status = hrx_graph_record_commands(
      schedule, node_start_index, node_count, stream_id, &execution_barrier,
      command_buffer, &release_flags);
  if (iree_status_is_ok(status) && release_flags) {
    // Publish registered-host destinations before signaling completion or
    // releasing a dependent CPU graph node.
    iree_hal_barrier_t release_barrier = execution_barrier;
    release_barrier.flags = release_flags;
    status = iree_hal_command_buffer_barrier(command_buffer, &release_barrier);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end_debug_group(command_buffer);
  }

  // End recording.
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_streaming_graph_compile_block(
    iree_hal_streaming_graph_compiler_t* compiler,
    const hrx_graph_schedule_t* schedule,
    const hrx_graph_dependency_block_t* plan, hrx_graph_block_t** out_block) {
  IREE_TRACE_ZONE_BEGIN(z0);
  hrx_graph_block_t* block = NULL;
  const hrx_graph_partition_t* partition = plan->partition;
  if (!partition) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_graph_allocate_barrier_block(
                compiler, plan->dependencies, IREE_HAL_QUEUE_BARRIER_FLAG_NONE,
                &block));
  } else {
    const uint32_t first_index = schedule->order[partition->start_index];
    const uint8_t kind =
        schedule->flags[first_index] & HRX_GRAPH_ANALYSIS_NODE_KIND_MASK;
    if (kind == HRX_GRAPH_ANALYSIS_NODE_RECORDABLE) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_streaming_graph_allocate_execute_block(
                  compiler, plan->dependencies,
                  IREE_HAL_QUEUE_EXECUTE_FLAG_NONE, &block));
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_streaming_graph_record_partition(
                  compiler, schedule, partition->start_index, partition->count,
                  plan->stream_index,
                  hrx_graph_block_execute(block)->command_buffer));
    } else {
      iree_hal_streaming_graph_node_t* first_node =
          schedule->sources[first_index];
      const iree_hal_streaming_graph_node_type_t type = first_node->type;
      if (type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_HOST_CALL) {
        // Host call gets its own block.
        iree_hal_streaming_graph_node_t* node = first_node;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_streaming_graph_allocate_host_call_block(
                    compiler, plan->dependencies,
                    iree_hal_streaming_graph_node_host(node)->callback,
                    iree_hal_streaming_graph_node_host(node)->user_data, node,
                    IREE_HAL_HOST_CALL_FLAG_NONE, &block));
      } else if (type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL) {
        iree_hal_streaming_graph_node_t* node = first_node;
        const iree_hal_streaming_graph_kernel_node_attrs_t* attrs =
            iree_hal_streaming_graph_node_kernel(node);
        const hrx_command_dispatch_t* dispatch = &attrs->command.dispatch;
        const iree_hal_dispatch_flags_t flags =
            dispatch->flags |
            (attrs->cooperative ? IREE_HAL_DISPATCH_FLAG_COOPERATIVE : 0);
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0,
            iree_hal_streaming_graph_allocate_dispatch_block(
                compiler, plan->dependencies, dispatch->executable,
                dispatch->export_ordinal, hrx_command_dispatch_config(dispatch),
                dispatch->constants, dispatch->bindings, flags, &block));
      } else if (type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH) {
        iree_hal_streaming_graph_node_t* node = first_node;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_streaming_graph_allocate_child_graph_block(
                    compiler, plan->dependencies,
                    iree_hal_streaming_graph_node_child_graph(node)->graph,
                    &block));
      } else {
        iree_hal_streaming_graph_node_t* node = first_node;
        if (type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC ||
            type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_FREE) {
          const bool allocate =
              type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC;
          IREE_RETURN_AND_END_ZONE_IF_ERROR(
              z0,
              iree_hal_streaming_graph_block_allocate(
                  &compiler->recording->arena_allocator,
                  allocate ? IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_MEMORY_ALLOCATE
                           : IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_MEMORY_FREE,
                  plan->dependencies, &block));
          iree_hal_streaming_graph_memory_block_attrs_t* memory =
              iree_hal_streaming_graph_block_memory(block);
          memory->allocation =
              allocate
                  ? iree_hal_streaming_graph_node_mem_alloc(node)->allocation
                  : iree_hal_streaming_graph_node_mem_free(node)->allocation;
          memory->flags =
              allocate ? IREE_HAL_STREAMING_GRAPH_ALLOCATION_FLAG_ALLOCATE
                       : IREE_HAL_STREAMING_GRAPH_ALLOCATION_FLAG_NONE;
          if (allocate &&
              iree_any_bit_set(
                  compiler->params->flags,
                  IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)) {
            memory->flags |= IREE_HAL_STREAMING_GRAPH_ALLOCATION_FLAG_AUTO_FREE;
          }
          IREE_RETURN_AND_END_ZONE_IF_ERROR(
              z0,
              iree_hal_resource_set_insert(compiler->recording->resource_set, 1,
                                           &memory->allocation));
        } else {
          // Event record/wait are the remaining isolated operations. Empty
          // and disabled nodes have no partition in the compiled schedule.
          const iree_hal_streaming_graph_block_type_t block_type =
              type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD
                  ? IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD
                  : IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_WAIT;
          IREE_RETURN_AND_END_ZONE_IF_ERROR(
              z0,
              iree_hal_streaming_graph_allocate_event_block(
                  compiler, block_type, plan->dependencies, node,
                  iree_hal_streaming_graph_node_event(node)->event, &block));
        }
      }
    }
  }
  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_recording_compile_schedule(
    iree_hal_streaming_graph_compiler_t* compiler,
    const hrx_graph_schedule_t* schedule) {
  IREE_TRACE_ZONE_BEGIN(z0);
  hrx_graph_dependency_cursor_t cursor;
  hrx_graph_dependency_cursor_initialize(schedule, &cursor);
  if (!cursor.block_count) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(
              &compiler->recording->arena_allocator,
              cursor.block_count * sizeof(*compiler->recording->blocks),
              (void**)&compiler->recording->blocks));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, hrx_graph_dependencies_allocate(compiler->params->context->device,
                                          cursor.semaphore_count,
                                          &compiler->recording->arena_allocator,
                                          compiler->recording->resource_set,
                                          &compiler->recording->semaphores));
  compiler->recording->semaphore_count = cursor.semaphore_count;

  hrx_graph_dependency_block_t plan;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         hrx_graph_dependency_cursor_next(&cursor, &plan)) {
    hrx_graph_block_t* block = NULL;
    status = iree_hal_streaming_graph_compile_block(compiler, schedule, &plan,
                                                    &block);
    if (iree_status_is_ok(status)) {
      // Publish only initialized entries so partial compilation releases child
      // executable and event references alongside the shared resource set.
      compiler->recording->blocks[compiler->recording->block_count++] = block;
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_streaming_graph_recording_compile(
    iree_hal_streaming_graph_compiler_t* compiler) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&compiler->params->context->device_entry->block_pool,
                        &arena);
  iree_hal_streaming_graph_edge_t* copy_dependencies = NULL;
  iree_status_t status = iree_hal_streaming_graph_schedule_copy_dependencies(
      compiler->params->graph, &arena, &copy_dependencies);
  hrx_graph_schedule_t schedule;
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_schedule_nodes(
        compiler->params->graph->node_blocks,
        compiler->params->graph->node_count,
        compiler->params->node_disabled_states,
        compiler->params->node_disabled_state_count, copy_dependencies, &arena,
        &schedule);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_recording_compile_schedule(compiler,
                                                                 &schedule);
  }
  iree_arena_deinitialize(&arena);
  return status;
}

// The caller can always release a partially built recording.
iree_status_t iree_hal_streaming_graph_recording_initialize(
    const iree_hal_streaming_graph_compile_params_t* params,
    iree_hal_streaming_graph_recording_t* out_recording) {
  *out_recording = (iree_hal_streaming_graph_recording_t){0};
  iree_arena_initialize(&params->context->device_entry->block_pool,
                        &out_recording->arena_allocator);
  iree_hal_streaming_graph_compiler_t compiler = {params, out_recording};
  iree_status_t status = iree_hal_resource_set_allocate(
      &params->context->device_entry->block_pool, &out_recording->resource_set);
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_memory_acquire_claims(&compiler);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_recording_compile(&compiler);
  }
  return status;
}

void iree_hal_streaming_graph_recording_deinitialize(
    iree_hal_streaming_graph_recording_t* recording) {
  iree_hal_streaming_graph_recording_release_children(recording);
  iree_hal_resource_set_free(recording->resource_set);
  iree_hal_streaming_graph_memory_release_claims(recording);
  iree_arena_deinitialize(&recording->arena_allocator);
}
