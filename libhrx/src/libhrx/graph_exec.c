// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Graph execution: instantiation of graph templates into executable blocks,
// and launch of those blocks onto a stream.

#include "graph_exec.h"

#include <string.h>

#include "graph_dependencies.h"
#include "hrx_internal.h"
#include "iree/base/api.h"
#include "iree/hal/utils/resource_set.h"

//===----------------------------------------------------------------------===//
// Block types and structures
//===----------------------------------------------------------------------===//

typedef enum hrx_graph_block_type_e {
  HRX_GRAPH_BLOCK_TYPE_QUEUE_BARRIER = 0,
  HRX_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL,
  HRX_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE,
} hrx_graph_block_type_t;

typedef struct hrx_graph_exec_block_t {
  // Recorded command buffer, application callback, or terminal dependency join.
  hrx_graph_block_type_t type;
  // Compact predecessor span relative to this block's submission ordinal.
  hrx_graph_dependencies_t dependencies;
  // Operation state; accepted submissions retain their referenced resources.
  union {
    // Borrowed application callback and parameters for an isolated host node.
    struct {
      // Application callback returning the status published to dependent work.
      hrx_host_call_fn_t fn;
      // Borrowed application state, live through callback completion.
      void* user_data;
    } host_call;
    // Borrowed command buffer retained by the recording's resource set.
    iree_hal_command_buffer_t* command_buffer;
  } attrs;
} hrx_graph_exec_block_t;

//===----------------------------------------------------------------------===//
// Partition recording into command buffers
//===----------------------------------------------------------------------===//

static iree_status_t hrx_graph_record_partition(
    const hrx_graph_schedule_t* schedule, uint32_t node_start_index,
    uint32_t node_count, uint8_t stream_id,
    iree_hal_command_buffer_t* command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_begin(command_buffer));

  // Queue waits order external producers; this partition has no local work
  // to drain at entry. Payload acquisition remains an explicit dependency.
  const iree_hal_barrier_t acquire_barrier = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_barrier(command_buffer, &acquire_barrier));

  const iree_string_view_t label_name =
      iree_make_cstring_view("graph partition");
  const iree_hal_label_location_t* location = NULL;
  const iree_hal_label_color_t label_color = iree_hal_label_color_unspecified();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_begin_debug_group(command_buffer, label_name,
                                                    label_color, location));

  // Every operation in a recorded partition executes on this device. Host
  // calls and other partitions cross the system-scope entry/exit barriers;
  // dependencies within the partition only order device memory accesses.
  const iree_hal_access_scope_t device_access =
      IREE_HAL_ACCESS_SCOPE_CONSTANT_READ |
      IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
      IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE;
  const iree_hal_memory_barrier_t memory_barrier = {
      .source_scope = device_access,
      .target_scope = device_access,
  };
  const iree_hal_barrier_t execution_barrier = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      .memory_barrier_count = 1,
      .memory_barriers = &memory_barrier,
  };
  iree_hal_barrier_flags_t release_flags = IREE_HAL_BARRIER_FLAG_NONE;
  iree_status_t status = hrx_graph_record_commands(
      schedule, node_start_index, node_count, stream_id, &execution_barrier,
      command_buffer, &release_flags);

  if (iree_status_is_ok(status)) {
    const iree_hal_barrier_t release_barrier = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
    };
    status = iree_hal_command_buffer_barrier(command_buffer, &release_barrier);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end_debug_group(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Instantiation
//===----------------------------------------------------------------------===//

static iree_status_t hrx_graph_recording_compile_schedule(
    const iree_hal_queue_family_t* queue_family,
    const hrx_graph_schedule_t* schedule, hrx_graph_recording_t* recording) {
  IREE_TRACE_ZONE_BEGIN(z0);
  hrx_graph_dependency_cursor_t cursor;
  hrx_graph_dependency_cursor_initialize(schedule, &cursor);
  if (!cursor.block_count) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&recording->arena_allocator,
                              cursor.block_count * sizeof(*recording->blocks),
                              (void**)&recording->blocks));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, hrx_graph_dependencies_allocate(
              recording->graph->device->hal_device, cursor.semaphore_count,
              &recording->arena_allocator, recording->resource_set,
              &recording->semaphores));
  recording->semaphore_count = cursor.semaphore_count;

  const iree_hal_command_category_t command_categories =
      hrx_queue_family_command_categories(queue_family);
  uint32_t block_index = 0;
  hrx_graph_dependency_block_t plan;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         hrx_graph_dependency_cursor_next(&cursor, &plan)) {
    hrx_graph_exec_block_t* block = NULL;
    status = iree_arena_allocate(&recording->arena_allocator, sizeof(*block),
                                 (void**)&block);
    if (!iree_status_is_ok(status)) {
      break;
    }
    block->dependencies = plan.dependencies;
    const hrx_graph_partition_t* partition = plan.partition;
    if (!partition) {
      block->type = HRX_GRAPH_BLOCK_TYPE_QUEUE_BARRIER;
    } else {
      const uint32_t first_index = schedule->order[partition->start_index];
      const uint8_t kind =
          schedule->flags[first_index] & HRX_GRAPH_ANALYSIS_NODE_KIND_MASK;
      if (kind == HRX_GRAPH_ANALYSIS_NODE_RECORDABLE) {
        block->type = HRX_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE;
        iree_hal_command_buffer_t* command_buffer = NULL;
        status = iree_hal_command_buffer_create(
            queue_family,
            hrx_device_profile_command_buffer_mode(recording->graph->device),
            command_categories, /*binding_capacity=*/0, &command_buffer);
        if (iree_status_is_ok(status)) {
          status = iree_hal_resource_set_insert(recording->resource_set, 1,
                                                &command_buffer);
        }
        if (iree_status_is_ok(status)) {
          block->attrs.command_buffer = command_buffer;
          status = hrx_graph_record_partition(
              schedule, partition->start_index, partition->count,
              plan.stream_index, command_buffer);
        }
        iree_hal_command_buffer_release(command_buffer);
      } else {
        // Host calls are the only isolated native operations. Empty nodes
        // have no partition in the compiled schedule.
        block->type = HRX_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL;
        const hrx_graph_host_call_node_attrs_internal_t* host_call =
            schedule->sources[first_index];
        block->attrs.host_call.fn = host_call->fn;
        block->attrs.host_call.user_data = host_call->user_data;
      }
    }
    recording->blocks[block_index++] = block;
  }
  if (iree_status_is_ok(status)) {
    recording->block_count = block_index;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t hrx_graph_recording_compile(
    const iree_hal_queue_family_t* queue_family,
    hrx_graph_recording_t* recording) {
  hrx_graph_t graph = recording->graph;
  iree_arena_allocator_t arena;
  iree_arena_initialize(&graph->device->block_pool, &arena);
  hrx_graph_schedule_t schedule;
  iree_status_t status =
      hrx_graph_schedule_nodes(graph->node_blocks, graph->node_count,
                               graph->additional_edges, &arena, &schedule);
  if (iree_status_is_ok(status)) {
    status = hrx_graph_recording_compile_schedule(queue_family, &schedule,
                                                  recording);
  }
  iree_arena_deinitialize(&arena);
  return status;
}

//===----------------------------------------------------------------------===//
// Recording ownership and replacement
//===----------------------------------------------------------------------===//

static void hrx_graph_recording_deinitialize(hrx_graph_recording_t* recording) {
  iree_hal_resource_set_free(recording->resource_set);
  iree_arena_deinitialize(&recording->arena_allocator);
  hrx_graph_release(recording->graph);
}

// Initializes even on failure so the caller can release rejected resources
// outside the executable's locks.
static iree_status_t hrx_graph_recording_initialize(
    hrx_graph_t graph, const iree_hal_queue_family_t* queue_family,
    hrx_graph_recording_t* out_recording) {
  memset(out_recording, 0, sizeof(*out_recording));
  out_recording->graph = graph;
  hrx_graph_retain(graph);
  iree_arena_initialize(&graph->device->block_pool,
                        &out_recording->arena_allocator);
  IREE_RETURN_IF_ERROR(iree_hal_resource_set_allocate(
      &graph->device->block_pool, &out_recording->resource_set));
  return hrx_graph_recording_compile(queue_family, out_recording);
}

hrx_status_t hrx_graph_instantiate(hrx_graph_t graph, hrx_queue_t queue,
                                   uint32_t flags, hrx_graph_exec_t* out_exec) {
  if (!out_exec) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "graph executable output is NULL");
  }
  *out_exec = NULL;
  if (!graph || !queue || graph->device != queue->device) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "graph and queue must belong to the same device");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  hrx_graph_exec_s* exec = NULL;
  HRX_RETURN_AND_END_ZONE_IF_IREE_ERROR(
      z0, iree_allocator_malloc_aligned(iree_allocator_system(), sizeof(*exec),
                                        iree_alignof(hrx_graph_exec_s), 0,
                                        (void**)&exec));
  iree_atomic_ref_count_init(&exec->ref_count);
  exec->queue_family = iree_hal_queue_family(queue->hal_queue);
  exec->flags = flags;
  iree_slim_mutex_initialize(&exec->launch_mutex);
  iree_slim_mutex_initialize(&exec->update_mutex);
  iree_status_t status = hrx_graph_recording_initialize(
      graph, exec->queue_family, &exec->recording);
  if (iree_status_is_ok(status)) {
    *out_exec = exec;
  } else {
    hrx_graph_exec_release(exec);
  }
  IREE_TRACE_ZONE_END(z0);
  return hrx_status_from_iree(status);
}

static void hrx_graph_exec_destroy(hrx_graph_exec_s* exec) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_semaphore_release(exec->last_launch_semaphore);
  hrx_graph_recording_deinitialize(&exec->recording);
  iree_slim_mutex_deinitialize(&exec->update_mutex);
  iree_slim_mutex_deinitialize(&exec->launch_mutex);
  iree_allocator_free_aligned(iree_allocator_system(), exec);
  IREE_TRACE_ZONE_END(z0);
}

void hrx_graph_exec_retain(hrx_graph_exec_t exec) {
  if (exec) {
    iree_atomic_ref_count_inc(&exec->ref_count);
  }
}

void hrx_graph_exec_release(hrx_graph_exec_t exec) {
  if (exec && iree_atomic_ref_count_dec(&exec->ref_count) == 1) {
    hrx_graph_exec_destroy(exec);
  }
}

hrx_status_t hrx_graph_exec_update(hrx_graph_exec_t exec, hrx_graph_t graph) {
  if (!exec || !graph) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "graph executable or replacement graph is NULL");
  }
  if (graph->device->hal_device !=
      iree_hal_queue_family_device(exec->queue_family)) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "replacement graph belongs to a different device");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&exec->update_mutex);
  hrx_graph_recording_t candidate;
  iree_status_t status =
      hrx_graph_recording_initialize(graph, exec->queue_family, &candidate);
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&exec->launch_mutex);
    hrx_graph_recording_t previous = exec->recording;
    exec->recording = candidate;
    candidate = previous;
    iree_slim_mutex_unlock(&exec->launch_mutex);
  }
  iree_slim_mutex_unlock(&exec->update_mutex);
  hrx_graph_recording_deinitialize(&candidate);
  IREE_TRACE_ZONE_END(z0);
  return hrx_status_from_iree(status);
}

//===----------------------------------------------------------------------===//
// Launch
//===----------------------------------------------------------------------===//

static iree_status_t hrx_graph_host_callback(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context) {
  IREE_TRACE_ZONE_BEGIN(z0);
  hrx_host_call_fn_t call_fn = (hrx_host_call_fn_t)args[0];
  void* call_user_data = (void*)args[1];
  iree_status_t status = hrx_status_to_iree(call_fn(call_user_data));
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Records a reserved point even on rejection after an accepted prefix. Failure
// is terminal, not permission to reuse points or leave a missing exit pending.
// Returns any detached reference for release outside submission serialization.
static iree_hal_semaphore_t* hrx_graph_exec_publish_launch(
    hrx_graph_exec_t exec, hrx_stream_t stream, uint64_t signal_value,
    iree_status_t status) {
  iree_hal_semaphore_t* stream_semaphore = stream->semaphore->hal_semaphore;
  stream->timepoint = signal_value;
  iree_hal_semaphore_t* previous_semaphore = NULL;
  if (exec->last_launch_semaphore != stream_semaphore) {
    iree_hal_semaphore_retain(stream_semaphore);
    previous_semaphore = exec->last_launch_semaphore;
    exec->last_launch_semaphore = stream_semaphore;
  }
  exec->last_launch_value = signal_value;
  if (!iree_status_is_ok(status)) {
    for (uint32_t i = 0; i < exec->recording.semaphore_count; ++i) {
      iree_hal_semaphore_fail(exec->recording.semaphores[i],
                              iree_status_clone(status));
    }
    iree_hal_semaphore_fail(stream_semaphore, iree_status_clone(status));
  }
  return previous_semaphore;
}

// An empty replacement has no payload, but a different stream must still join
// the executable's last accepted launch. On the same stream its tail suffices.
static iree_status_t hrx_graph_exec_launch_empty(
    hrx_graph_exec_t exec, hrx_stream_t stream,
    iree_hal_semaphore_t** out_previous_semaphore) {
  *out_previous_semaphore = NULL;
  iree_hal_semaphore_t* stream_semaphore = stream->semaphore->hal_semaphore;
  if (!exec->last_launch_semaphore ||
      exec->last_launch_semaphore == stream_semaphore) {
    return iree_ok_status();
  }
  iree_hal_semaphore_t* wait_semaphores[] = {exec->last_launch_semaphore,
                                             stream_semaphore};
  uint64_t wait_values[] = {exec->last_launch_value, stream->timepoint};
  uint64_t signal_value = stream->timepoint + 1;
  const iree_hal_semaphore_list_t waits = {
      .count = stream->timepoint ? 2 : 1,
      .semaphores = wait_semaphores,
      .payload_values = wait_values,
  };
  const iree_hal_semaphore_list_t signals = {
      .count = 1,
      .semaphores = &stream_semaphore,
      .payload_values = &signal_value,
  };
  const iree_hal_barrier_list_t empty_list = {0};
  const iree_hal_queue_barriers_t barriers = {&empty_list, &empty_list};
  iree_status_t status =
      iree_hal_queue_barrier(stream->hal_queue, waits, signals, &barriers,
                             IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
  *out_previous_semaphore =
      hrx_graph_exec_publish_launch(exec, stream, signal_value, status);
  return status;
}

hrx_status_t hrx_graph_exec_launch(hrx_graph_exec_t exec, hrx_stream_t stream) {
  if (!exec || !stream) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "graph executable or stream is NULL");
  }
  if (iree_hal_queue_family(stream->hal_queue) != exec->queue_family) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "stream queue family differs from graph recording");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  // Flush pending stream work before graph launch.
  hrx_status_t flush_status = hrx_stream_flush(stream);
  if (!hrx_status_is_ok(flush_status)) {
    HRX_RETURN_AND_END_ZONE(z0, flush_status);
  }

  // Reservations describe submitted work, not completed payloads. Each edge
  // has one producer per launch; the previous launch must finish before any
  // producer can publish the next generation and satisfy an earlier reader.
  iree_slim_mutex_lock(&exec->launch_mutex);
  const hrx_graph_recording_t* recording = &exec->recording;
  if (recording->block_count == 0) {
    iree_hal_semaphore_t* previous_semaphore = NULL;
    iree_status_t status =
        hrx_graph_exec_launch_empty(exec, stream, &previous_semaphore);
    iree_slim_mutex_unlock(&exec->launch_mutex);
    iree_hal_semaphore_release(previous_semaphore);
    IREE_TRACE_ZONE_END(z0);
    return hrx_status_from_iree(status);
  }
  const uint64_t generation = ++exec->launch_generation;
  iree_hal_semaphore_t* stream_semaphore = stream->semaphore->hal_semaphore;
  const uint64_t stream_wait_value = stream->timepoint;
  uint64_t stream_signal_value = stream_wait_value + 1;

  iree_hal_semaphore_t* entry_semaphores[2];
  uint64_t entry_values[2];
  iree_hal_semaphore_list_t entry_waits = {
      .semaphores = entry_semaphores,
      .payload_values = entry_values,
  };
  if (stream_wait_value > 0) {
    entry_semaphores[entry_waits.count] = stream_semaphore;
    entry_values[entry_waits.count++] = stream_wait_value;
  }
  // On the same stream its tail already includes the preceding launch.
  if (exec->last_launch_semaphore &&
      exec->last_launch_semaphore != stream_semaphore) {
    entry_semaphores[entry_waits.count] = exec->last_launch_semaphore;
    entry_values[entry_waits.count++] = exec->last_launch_value;
  }
  const iree_hal_semaphore_list_t exit_signals = {
      .count = 1,
      .semaphores = &stream_semaphore,
      .payload_values = &stream_signal_value,
  };
  uint64_t values[HRX_GRAPH_MAX_WORKSTREAMS];
  if (recording->semaphore_count) {
    for (uint32_t i = 0; i < IREE_ARRAYSIZE(values); ++i) {
      values[i] = generation;
    }
  }
  iree_status_t status = iree_ok_status();
  for (uint32_t block_index = 0;
       iree_status_is_ok(status) && block_index < recording->block_count;
       ++block_index) {
    const hrx_graph_exec_block_t* block = recording->blocks[block_index];
    iree_hal_semaphore_list_t wait_semaphores;
    iree_hal_semaphore_list_t signal_semaphores;
    hrx_graph_dependencies_resolve(
        block->dependencies, block_index, recording->semaphore_count,
        recording->semaphores, values, entry_waits, exit_signals,
        &wait_semaphores, &signal_semaphores);

    switch (block->type) {
      case HRX_GRAPH_BLOCK_TYPE_QUEUE_BARRIER: {
        // Empty graph nodes and terminal joins only order semaphore points.
        // Recorded work owns its acquire/release boundaries; repeating system
        // visibility here would add a cache flush to a pure dependency join.
        const iree_hal_barrier_list_t empty_list = {0};
        const iree_hal_queue_barriers_t barriers = {&empty_list, &empty_list};
        status = iree_hal_queue_barrier(stream->hal_queue, wait_semaphores,
                                        signal_semaphores, &barriers,
                                        IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
        break;
      }
      case HRX_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE:
        status = iree_hal_queue_execute(
            stream->hal_queue, wait_semaphores, signal_semaphores,
            block->attrs.command_buffer, iree_hal_buffer_binding_table_empty(),
            IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
        break;
      case HRX_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL: {
        uint64_t call_args[4] = {
            (uint64_t)(uintptr_t)block->attrs.host_call.fn,
            (uint64_t)(uintptr_t)block->attrs.host_call.user_data,
        };
        status = iree_hal_queue_host_call(
            stream->hal_queue, wait_semaphores, signal_semaphores,
            iree_hal_make_host_call(hrx_graph_host_callback, NULL), call_args,
            IREE_HAL_HOST_CALL_FLAG_NONE);
        break;
      }
    }
  }

  iree_hal_semaphore_t* previous_semaphore =
      hrx_graph_exec_publish_launch(exec, stream, stream_signal_value, status);
  iree_slim_mutex_unlock(&exec->launch_mutex);
  iree_hal_semaphore_release(previous_semaphore);
  IREE_TRACE_ZONE_END(z0);
  return hrx_status_from_iree(status);
}
