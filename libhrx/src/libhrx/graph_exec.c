// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Graph execution: instantiation of graph templates into executable blocks,
// and launch of those blocks onto a stream.

#include "hrx_internal.h"
#include "iree/base/api.h"
#include "iree/hal/utils/resource_set.h"

//===----------------------------------------------------------------------===//
// Block types and structures
//===----------------------------------------------------------------------===//

typedef enum hrx_graph_block_type_e {
  HRX_GRAPH_BLOCK_TYPE_QUEUE_BARRIER = 0,
  HRX_GRAPH_BLOCK_TYPE_QUEUE_FILL,
  HRX_GRAPH_BLOCK_TYPE_QUEUE_COPY,
  HRX_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL,
  HRX_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH,
  HRX_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE,
} hrx_graph_block_type_t;

typedef void (*hrx_graph_host_callback_fn_t)(void* user_data);

typedef struct hrx_graph_barrier_block_attrs_t {
  iree_hal_execute_flags_t flags;
} hrx_graph_barrier_block_attrs_t;

typedef struct hrx_graph_fill_block_attrs_t {
  iree_hal_buffer_t* target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
  uint64_t pattern;
  iree_host_size_t pattern_length;
  iree_hal_fill_flags_t flags;
} hrx_graph_fill_block_attrs_t;

typedef struct hrx_graph_copy_block_attrs_t {
  iree_hal_buffer_t* source_buffer;
  iree_device_size_t source_offset;
  iree_hal_buffer_t* target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
  iree_hal_copy_flags_t flags;
} hrx_graph_copy_block_attrs_t;

typedef struct hrx_graph_host_call_block_attrs_t {
  hrx_graph_host_callback_fn_t fn;
  void* user_data;
  iree_hal_host_call_flags_t flags;
} hrx_graph_host_call_block_attrs_t;

typedef struct hrx_graph_dispatch_block_attrs_t {
  iree_hal_executable_t* executable;
  iree_host_size_t entry_point;
  iree_hal_dispatch_config_t config;
  iree_const_byte_span_t constants;
  iree_hal_buffer_ref_list_t bindings;
  iree_hal_dispatch_flags_t flags;
} hrx_graph_dispatch_block_attrs_t;

typedef struct hrx_graph_execute_block_attrs_t {
  iree_hal_command_buffer_t* command_buffer;
  iree_hal_execute_flags_t flags;
} hrx_graph_execute_block_attrs_t;

typedef union hrx_graph_block_attrs_t {
  hrx_graph_barrier_block_attrs_t barrier;
  hrx_graph_fill_block_attrs_t fill;
  hrx_graph_copy_block_attrs_t copy;
  hrx_graph_host_call_block_attrs_t host_call;
  hrx_graph_dispatch_block_attrs_t dispatch;
  hrx_graph_execute_block_attrs_t execute;
} hrx_graph_block_attrs_t;

typedef struct hrx_graph_exec_block_t {
  hrx_graph_block_type_t type;
  uint32_t node_start_index;
  uint32_t node_count;
  uint16_t wait_semaphore_count;
  uint16_t signal_semaphore_count;
  // Variable-length trailing data follows.
} hrx_graph_exec_block_t;

typedef struct hrx_graph_block_ptrs_t {
  uint16_t* wait_semaphore_indices;
  uint32_t* wait_payload_deltas;
  uint16_t* signal_semaphore_indices;
  uint32_t* signal_payload_deltas;
  hrx_graph_block_attrs_t* attrs;
} hrx_graph_block_ptrs_t;

//===----------------------------------------------------------------------===//
// Block helpers
//===----------------------------------------------------------------------===//

static inline void hrx_graph_block_get_ptrs(hrx_graph_exec_block_t* block,
                                            hrx_graph_block_ptrs_t* out_ptrs) {
  uint8_t* ptr = (uint8_t*)block + sizeof(*block);
  out_ptrs->wait_semaphore_indices = (uint16_t*)ptr;
  ptr += block->wait_semaphore_count * sizeof(uint16_t);
  out_ptrs->wait_payload_deltas = (uint32_t*)ptr;
  ptr += block->wait_semaphore_count * sizeof(uint32_t);
  out_ptrs->signal_semaphore_indices = (uint16_t*)ptr;
  ptr += block->signal_semaphore_count * sizeof(uint16_t);
  out_ptrs->signal_payload_deltas = (uint32_t*)ptr;
  ptr += block->signal_semaphore_count * sizeof(uint32_t);
  out_ptrs->attrs = (hrx_graph_block_attrs_t*)ptr;
}

static iree_status_t hrx_graph_block_allocate(
    iree_arena_allocator_t* arena, hrx_graph_block_type_t type,
    uint32_t node_start_index, uint32_t node_count,
    uint16_t wait_semaphore_count, uint16_t signal_semaphore_count,
    hrx_graph_exec_block_t** out_block, hrx_graph_block_ptrs_t* out_ptrs) {
  iree_host_size_t size = sizeof(hrx_graph_exec_block_t);
  size += wait_semaphore_count * sizeof(uint16_t);
  size += wait_semaphore_count * sizeof(uint32_t);
  size += signal_semaphore_count * sizeof(uint16_t);
  size += signal_semaphore_count * sizeof(uint32_t);
  size += sizeof(hrx_graph_block_attrs_t);

  hrx_graph_exec_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, size, (void**)&block));

  block->type = type;
  block->node_start_index = node_start_index;
  block->node_count = node_count;
  block->wait_semaphore_count = wait_semaphore_count;
  block->signal_semaphore_count = signal_semaphore_count;

  hrx_graph_block_get_ptrs(block, out_ptrs);
  *out_block = block;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Hazard tracking for command buffer recording
//===----------------------------------------------------------------------===//

void hrx_graph_barrier_state_reset(hrx_graph_barrier_state_t* state) {
  state->count = 0;
  state->invalid = 0;
}

static bool hrx_graph_barrier_state_test_hazard(
    const hrx_graph_barrier_state_t* state, uint32_t value) {
  if (state->invalid) return true;
  for (uint32_t i = 0; i < state->count; ++i) {
    if (state->values[i] == value) return true;
  }
  return false;
}

static void hrx_graph_barrier_state_insert(hrx_graph_barrier_state_t* state,
                                           uint32_t value) {
  if (state->count >= IREE_ARRAYSIZE(state->values)) {
    state->invalid = 1;
    return;
  }
  state->values[state->count++] = value;
}

iree_status_t hrx_graph_record_node_barrier(
    hrx_graph_barrier_state_t* state, const hrx_graph_node_s* node,
    const hrx_graph_edge_t* additional_edges, const uint32_t* node_index_map,
    uint32_t sorted_index, iree_hal_command_buffer_t* command_buffer,
    bool* out_did_barrier) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(node);
  IREE_ASSERT_ARGUMENT(node_index_map);
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(out_did_barrier);
  *out_did_barrier = false;

  bool has_hazard = false;
  for (uint32_t i = 0; i < node->dependency_count; ++i) {
    const uint32_t dependency_sort_index =
        node_index_map[node->dependencies[i]->node_index];
    if (dependency_sort_index == UINT32_MAX) continue;
    if (hrx_graph_barrier_state_test_hazard(state, dependency_sort_index)) {
      has_hazard = true;
      break;
    }
  }
  for (const hrx_graph_edge_t* edge = additional_edges;
       !has_hazard && edge != NULL; edge = edge->next) {
    if (edge->to != node) continue;
    const uint32_t dependency_sort_index =
        node_index_map[edge->from->node_index];
    if (dependency_sort_index == UINT32_MAX) continue;
    has_hazard =
        hrx_graph_barrier_state_test_hazard(state, dependency_sort_index);
  }

  if (has_hazard) {
    const iree_hal_memory_barrier_t memory_barrier = {
        .source_scope = IREE_HAL_MEMORY_ACCESS_ALL,
        .target_scope = IREE_HAL_MEMORY_ACCESS_ALL,
    };
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_execution_barrier(
        command_buffer, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
        IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 1, &memory_barrier, 0, NULL));
    hrx_graph_barrier_state_reset(state);
    *out_did_barrier = true;
  }

  hrx_graph_barrier_state_insert(state, sorted_index);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Partition recording into command buffers
//===----------------------------------------------------------------------===//

static iree_status_t hrx_graph_record_partition(
    hrx_graph_exec_s* exec, hrx_graph_sort_node_t* sorted_nodes,
    uint32_t node_start_index, uint32_t node_count,
    const uint32_t* node_index_map, uint8_t stream_id,
    const hrx_graph_edge_t* additional_edges,
    iree_hal_command_buffer_t* command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_begin(command_buffer));

  const iree_string_view_t label_name = iree_make_cstring_view("tbd_partition");
  const iree_hal_label_location_t* location = NULL;
  const iree_hal_label_color_t label_color = iree_hal_label_color_unspecified();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_begin_debug_group(command_buffer, label_name,
                                                    label_color, location));

  iree_status_t status = iree_ok_status();
  hrx_graph_barrier_state_t barrier_state;
  hrx_graph_barrier_state_reset(&barrier_state);
  for (uint32_t i = 0; iree_status_is_ok(status) && i < node_count; ++i) {
    hrx_graph_sort_node_t* sort_node = &sorted_nodes[node_start_index + i];
    if (sort_node->stream_id != stream_id) continue;
    hrx_graph_node_s* node = sort_node->node;
    bool did_barrier = false;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, hrx_graph_record_node_barrier(
                &barrier_state, node, additional_edges, node_index_map,
                sort_node->sorted_index, command_buffer, &did_barrier));
    switch (node->type) {
      case HRX_GRAPH_NODE_TYPE_INTERNAL_KERNEL: {
        const hrx_graph_kernel_node_attrs_internal_t* attrs =
            &node->attrs.kernel;
        const iree_hal_dispatch_config_t config = {
            .workgroup_size = {attrs->block_dim[0], attrs->block_dim[1],
                               attrs->block_dim[2]},
            .workgroup_count = {attrs->grid_dim[0], attrs->grid_dim[1],
                                attrs->grid_dim[2]},
            .dynamic_workgroup_local_memory = attrs->shared_memory_bytes,
        };
        const iree_hal_dispatch_flags_t flags =
            attrs->bindings.count
                ? IREE_HAL_DISPATCH_FLAG_NONE
                : IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS;
        status = iree_hal_command_buffer_dispatch(
            command_buffer, attrs->executable,
            iree_hal_executable_function_from_index(attrs->export_ordinal),
            config, attrs->constants, attrs->bindings, flags);
        break;
      }
      case HRX_GRAPH_NODE_TYPE_INTERNAL_MEMCPY: {
        const hrx_graph_memcpy_node_attrs_internal_t* attrs =
            &node->attrs.memcpy;
        status = iree_hal_command_buffer_copy_buffer(
            command_buffer, attrs->src_ref, attrs->dst_ref, attrs->flags);
        if (iree_status_is_ok(status)) {
          void* resources[] = {attrs->src_ref.buffer, attrs->dst_ref.buffer};
          status = iree_hal_resource_set_insert(
              exec->resource_set, IREE_ARRAYSIZE(resources), resources);
        }
        break;
      }
      case HRX_GRAPH_NODE_TYPE_INTERNAL_MEMSET: {
        const hrx_graph_memset_node_attrs_internal_t* attrs =
            &node->attrs.memset;
        status = iree_hal_command_buffer_fill_buffer(
            command_buffer, attrs->dst_ref, &attrs->pattern,
            attrs->pattern_size, attrs->flags);
        if (iree_status_is_ok(status)) {
          status = iree_hal_resource_set_insert(exec->resource_set, 1,
                                                &attrs->dst_ref.buffer);
        }
        break;
      }
      default: {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "non-recordable node type %d in recordable partition",
            (int)node->type);
        break;
      }
    }
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

iree_status_t hrx_graph_exec_instantiate_locked(
    hrx_graph_exec_t exec, hrx_graph_node_block_t* node_blocks,
    iree_host_size_t node_count) {
  IREE_ASSERT_ARGUMENT(exec);
  IREE_TRACE_ZONE_BEGIN(z0);

  hrx_graph_schedule_t schedule;
  hrx_graph_edge_t* additional_edges =
      exec->graph ? exec->graph->additional_edges : NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, hrx_graph_schedule_nodes(node_blocks, node_count, additional_edges,
                                   &exec->arena_allocator, &schedule));

  exec->block_count = schedule.block_count;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&exec->arena_allocator,
                              exec->block_count * sizeof(*exec->blocks),
                              (void**)&exec->blocks));

  if (schedule.partition_count == 0) {
    exec->block_count = 0;
    exec->blocks = NULL;
    exec->semaphore_count = 0;
    exec->semaphores = NULL;
    exec->semaphore_base_values = NULL;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  uint32_t semaphore_count = 0;
  if (schedule.partition_count > 1) {
    for (iree_host_size_t i = 0; i < schedule.partition_count - 1; i++) {
      if (schedule.partitions[i].stream_count > 1) {
        semaphore_count += schedule.partitions[i].stream_count;
      } else {
        semaphore_count += 1;
      }
    }
  }
  exec->semaphore_count = semaphore_count;

  if (exec->semaphore_count > 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_arena_allocate(&exec->arena_allocator,
                            exec->semaphore_count * sizeof(*exec->semaphores),
                            (void**)&exec->semaphores));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(
                &exec->arena_allocator,
                exec->semaphore_count * sizeof(*exec->semaphore_base_values),
                (void**)&exec->semaphore_base_values));

    iree_hal_device_t* hal_device = exec->device->hal_device;
    for (uint32_t i = 0; i < exec->semaphore_count; i++) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_semaphore_create(
                  hal_device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0ull,
                  IREE_HAL_SEMAPHORE_FLAG_NONE, &exec->semaphores[i]));
      exec->semaphore_base_values[i] = 0;
    }

    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_resource_set_insert(
                exec->resource_set, exec->semaphore_count, exec->semaphores));
  }

  uint32_t block_index = 0;
  uint32_t semaphore_index = 0;
  for (iree_host_size_t p = 0; p < schedule.partition_count; p++) {
    const hrx_graph_partition_t* partition = &schedule.partitions[p];

    uint16_t wait_semaphore_count = 0;
    if (p > 0) {
      hrx_graph_partition_t* prev_partition = &schedule.partitions[p - 1];
      wait_semaphore_count =
          prev_partition->stream_count > 1 ? prev_partition->stream_count : 1;
    }

    uint16_t signal_semaphore_count = 0;
    if (p < schedule.partition_count - 1) {
      signal_semaphore_count =
          partition->stream_count > 1 ? partition->stream_count : 1;
    }

    if (partition->type == HRX_GRAPH_PARTITION_TYPE_RECORDABLE) {
      const uint8_t stream_count = partition->stream_count;
      const uint32_t partition_wait_semaphore_start =
          semaphore_index - wait_semaphore_count;
      const uint32_t partition_signal_semaphore_start = semaphore_index;
      for (uint8_t s = 0; s < stream_count; s++) {
        hrx_graph_exec_block_t* block = NULL;

        uint16_t block_signal_count = 0;
        if (signal_semaphore_count > 0) {
          block_signal_count = (stream_count > 1) ? 1 : signal_semaphore_count;
        }
        hrx_graph_block_ptrs_t ptrs;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, hrx_graph_block_allocate(
                    &exec->arena_allocator, HRX_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE,
                    partition->start_index, partition->count,
                    wait_semaphore_count, block_signal_count, &block, &ptrs));

        // Create command buffer for recording.
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0,
            iree_hal_command_buffer_create(
                exec->device->hal_device,
                IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED,
                IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
                /*binding_capacity=*/0, &ptrs.attrs->execute.command_buffer));
        ptrs.attrs->execute.flags = IREE_HAL_EXECUTE_FLAG_NONE;

        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0,
            iree_hal_resource_set_insert(exec->resource_set, 1,
                                         &ptrs.attrs->execute.command_buffer));
        iree_hal_command_buffer_release(ptrs.attrs->execute.command_buffer);

        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, hrx_graph_record_partition(
                    exec, schedule.sorted_nodes, partition->start_index,
                    partition->count, schedule.node_index_map, s,
                    additional_edges, ptrs.attrs->execute.command_buffer));

        if (wait_semaphore_count > 0) {
          for (uint16_t w = 0; w < wait_semaphore_count; w++) {
            ptrs.wait_semaphore_indices[w] = partition_wait_semaphore_start + w;
            ptrs.wait_payload_deltas[w] = 1;
          }
        }
        if (block_signal_count > 0) {
          if (stream_count > 1) {
            ptrs.signal_semaphore_indices[0] =
                partition_signal_semaphore_start + s;
            ptrs.signal_payload_deltas[0] = 1;
          } else {
            for (uint16_t i = 0; i < block_signal_count; i++) {
              ptrs.signal_semaphore_indices[i] =
                  partition_signal_semaphore_start + i;
              ptrs.signal_payload_deltas[i] = 1;
            }
          }
        }

        exec->blocks[block_index++] = block;
      }
      semaphore_index += signal_semaphore_count;
    } else {
      hrx_graph_exec_block_t* block = NULL;
      hrx_graph_block_ptrs_t ptrs;
      if (partition->type == HRX_GRAPH_PARTITION_TYPE_HOST_CALL) {
        hrx_graph_node_s* node =
            schedule.sorted_nodes[partition->start_index].node;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0,
            hrx_graph_block_allocate(
                &exec->arena_allocator, HRX_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL,
                partition->start_index, partition->count, wait_semaphore_count,
                signal_semaphore_count, &block, &ptrs));
        ptrs.attrs->host_call.fn = node->attrs.host.fn;
        ptrs.attrs->host_call.user_data = node->attrs.host.user_data;
        ptrs.attrs->host_call.flags = IREE_HAL_HOST_CALL_FLAG_NONE;
      } else {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0,
            hrx_graph_block_allocate(
                &exec->arena_allocator, HRX_GRAPH_BLOCK_TYPE_QUEUE_BARRIER,
                partition->start_index, partition->count, wait_semaphore_count,
                signal_semaphore_count, &block, &ptrs));
        ptrs.attrs->barrier.flags = IREE_HAL_EXECUTE_FLAG_NONE;
      }
      if (wait_semaphore_count > 0) {
        for (uint16_t w = 0; w < wait_semaphore_count; w++) {
          ptrs.wait_semaphore_indices[w] =
              semaphore_index - wait_semaphore_count + w;
          ptrs.wait_payload_deltas[w] = 1;
        }
      }
      if (signal_semaphore_count > 0) {
        for (uint16_t i = 0; i < signal_semaphore_count; i++) {
          ptrs.signal_semaphore_indices[i] = semaphore_index + i;
          ptrs.signal_payload_deltas[i] = 1;
        }
      }
      semaphore_index += signal_semaphore_count;
      exec->blocks[block_index++] = block;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Launch
//===----------------------------------------------------------------------===//

static iree_status_t hrx_graph_host_callback(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context) {
  IREE_TRACE_ZONE_BEGIN(z0);
  hrx_graph_host_callback_fn_t call_fn = (hrx_graph_host_callback_fn_t)args[0];
  void* call_user_data = (void*)args[1];
  call_fn(call_user_data);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

hrx_status_t hrx_graph_exec_launch(hrx_graph_exec_t exec, hrx_stream_t stream) {
  IREE_ASSERT_ARGUMENT(exec);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (exec->block_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return hrx_ok_status();
  }

  // Flush pending stream work before graph launch.
  hrx_status_t flush_status = hrx_stream_flush(stream);
  if (!hrx_status_is_ok(flush_status)) {
    HRX_RETURN_AND_END_ZONE(z0, flush_status);
  }

  // Multiple threads may launch the same executable. Serialize the semaphore
  // query and advancement as one transaction so each launch observes the
  // timeline values committed by the preceding launch.
  iree_slim_mutex_lock(&exec->mutex);

  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; iree_status_is_ok(status) && i < exec->semaphore_count;
       i++) {
    status = iree_hal_semaphore_query(exec->semaphores[i],
                                      &exec->semaphore_base_values[i]);
  }

  uint64_t stream_wait_value = stream->timepoint;
  uint64_t stream_signal_value = stream->timepoint + 1;

  uint64_t* new_base_values = NULL;
  if (exec->semaphore_count > 0) {
    const iree_host_size_t base_values_size =
        exec->semaphore_count * sizeof(uint64_t);
    new_base_values = (uint64_t*)iree_alloca(base_values_size);
    memcpy(new_base_values, exec->semaphore_base_values, base_values_size);
  }

  iree_hal_device_t* hal_device = exec->device->hal_device;
  for (uint32_t block_index = 0;
       iree_status_is_ok(status) && block_index < exec->block_count;
       block_index++) {
    hrx_graph_exec_block_t* block = exec->blocks[block_index];
    hrx_graph_block_ptrs_t ptrs;
    hrx_graph_block_get_ptrs(block, &ptrs);

    const iree_host_size_t total_semaphores =
        block->wait_semaphore_count + block->signal_semaphore_count + 2;
    iree_hal_semaphore_t** semaphore_array =
        (iree_hal_semaphore_t**)iree_alloca(total_semaphores *
                                            sizeof(iree_hal_semaphore_t*));
    uint64_t* value_array =
        (uint64_t*)iree_alloca(total_semaphores * sizeof(uint64_t));

    iree_hal_semaphore_t** wait_sems = semaphore_array;
    uint64_t* wait_vals = value_array;
    iree_hal_semaphore_t** signal_sems =
        semaphore_array + (block->wait_semaphore_count + 1);
    uint64_t* signal_vals = value_array + (block->wait_semaphore_count + 1);

    iree_host_size_t wait_count = 0;
    if (block_index == 0 && stream_wait_value > 0) {
      wait_sems[wait_count] = stream->semaphore->hal_semaphore;
      wait_vals[wait_count] = stream_wait_value;
      wait_count++;
    }

    for (uint16_t i = 0; i < block->wait_semaphore_count; i++) {
      const uint16_t sem_idx = ptrs.wait_semaphore_indices[i];
      const uint32_t delta = ptrs.wait_payload_deltas[i];
      wait_sems[wait_count] = exec->semaphores[sem_idx];
      wait_vals[wait_count] = exec->semaphore_base_values[sem_idx] + delta;
      wait_count++;
    }

    iree_host_size_t signal_count = 0;
    for (uint16_t i = 0; i < block->signal_semaphore_count; i++) {
      const uint16_t sem_idx = ptrs.signal_semaphore_indices[i];
      const uint32_t delta = ptrs.signal_payload_deltas[i];
      signal_sems[signal_count] = exec->semaphores[sem_idx];
      signal_vals[signal_count] = exec->semaphore_base_values[sem_idx] + delta;
      new_base_values[sem_idx] = signal_vals[signal_count];
      signal_count++;
    }

    if (block_index == exec->block_count - 1) {
      signal_sems[signal_count] = stream->semaphore->hal_semaphore;
      signal_vals[signal_count] = stream_signal_value;
      signal_count++;
    }

    iree_hal_semaphore_list_t wait_semaphores = {
        .count = wait_count,
        .semaphores = wait_sems,
        .payload_values = wait_vals,
    };
    iree_hal_semaphore_list_t signal_semaphores = {
        .count = signal_count,
        .semaphores = signal_sems,
        .payload_values = signal_vals,
    };

    switch (block->type) {
      case HRX_GRAPH_BLOCK_TYPE_QUEUE_BARRIER:
        status = iree_hal_device_queue_barrier(
            hal_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphores,
            signal_semaphores, ptrs.attrs->barrier.flags);
        break;
      case HRX_GRAPH_BLOCK_TYPE_QUEUE_FILL:
        status = iree_hal_device_queue_fill(
            hal_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphores,
            signal_semaphores, ptrs.attrs->fill.target_buffer,
            ptrs.attrs->fill.target_offset, ptrs.attrs->fill.length,
            &ptrs.attrs->fill.pattern, ptrs.attrs->fill.pattern_length,
            ptrs.attrs->fill.flags);
        break;
      case HRX_GRAPH_BLOCK_TYPE_QUEUE_COPY:
        status = iree_hal_device_queue_copy(
            hal_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphores,
            signal_semaphores, ptrs.attrs->copy.source_buffer,
            ptrs.attrs->copy.source_offset, ptrs.attrs->copy.target_buffer,
            ptrs.attrs->copy.target_offset, ptrs.attrs->copy.length,
            ptrs.attrs->copy.flags);
        break;
      case HRX_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH: {
        iree_hal_buffer_ref_list_t bindings_list = {
            .count = ptrs.attrs->dispatch.bindings.count,
            .values = ptrs.attrs->dispatch.bindings.values,
        };
        status = iree_hal_device_queue_dispatch(
            hal_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphores,
            signal_semaphores, ptrs.attrs->dispatch.executable,
            iree_hal_executable_function_from_index(
                (uint32_t)ptrs.attrs->dispatch.entry_point),
            ptrs.attrs->dispatch.config, ptrs.attrs->dispatch.constants,
            bindings_list, ptrs.attrs->dispatch.flags);
        break;
      }
      case HRX_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE:
        status = iree_hal_device_queue_execute(
            hal_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphores,
            signal_semaphores, ptrs.attrs->execute.command_buffer,
            iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE);
        break;
      case HRX_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL: {
        uint64_t call_args[4] = {
            (uint64_t)(uintptr_t)ptrs.attrs->host_call.fn,
            (uint64_t)(uintptr_t)ptrs.attrs->host_call.user_data,
        };
        status = iree_hal_device_queue_host_call(
            hal_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphores,
            signal_semaphores,
            iree_hal_make_host_call(hrx_graph_host_callback, NULL), call_args,
            ptrs.attrs->host_call.flags);
        break;
      }
      default:
        status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                  "unsupported block type %u", block->type);
        break;
    }
  }

  if (iree_status_is_ok(status) && exec->semaphore_count > 0) {
    memcpy(exec->semaphore_base_values, new_base_values,
           exec->semaphore_count * sizeof(uint64_t));
  }

  if (iree_status_is_ok(status)) {
    stream->timepoint = stream_signal_value;
  }

  iree_slim_mutex_unlock(&exec->mutex);
  IREE_TRACE_ZONE_END(z0);
  return hrx_status_from_iree(status);
}
