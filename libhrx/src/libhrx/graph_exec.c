// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Graph execution: instantiation of graph templates into executable blocks,
// and launch of those blocks onto a stream.

#include "graph_exec.h"

#include <string.h>

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

typedef struct hrx_graph_barrier_block_attrs_t {
  iree_hal_queue_barrier_flags_t flags;
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
  // Application callback returning the status published to dependent work.
  hrx_host_call_fn_t fn;
  // Borrowed application state, live through callback completion.
  void* user_data;
  // HAL execution policy for this host node.
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
  iree_hal_queue_execute_flags_t flags;
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
  // Type-specific attributes, stored inline so the compiler supplies the
  // pointer alignment the union requires and the block header ends on that
  // same boundary.
  hrx_graph_block_attrs_t attrs;
  // Variable-length trailing data follows.
} hrx_graph_exec_block_t;

// Blocks are carved out of an arena, which guarantees iree_max_align_t and
// nothing stronger.
static_assert(iree_alignof(hrx_graph_exec_block_t) <= iree_max_align_t,
              "block alignment must be satisfiable by an arena allocation");

// The first trailing region begins at the end of the header and holds
// uint32_t, so the header's size decides whether that region is aligned.
// The region order only keeps the regions aligned relative to each other;
// this assert is what pins the offset they start from.
static_assert(sizeof(hrx_graph_exec_block_t) % iree_alignof(uint32_t) == 0,
              "block header must end on the first trailing region's alignment");

// Resolved pointers to a block's payload: the attributes stored inline in the
// block and the variable-length arrays trailing it.
typedef struct hrx_graph_block_ptrs_t {
  // Indices of the preceding partition's completion semaphores.
  uint32_t* wait_semaphore_indices;
  // Indices of the semaphores produced by this block.
  uint32_t* signal_semaphore_indices;
  // Type-specific attributes embedded in the block header.
  hrx_graph_block_attrs_t* attrs;
} hrx_graph_block_ptrs_t;

//===----------------------------------------------------------------------===//
// Block helpers
//===----------------------------------------------------------------------===//

// Resolves the semaphore indices trailing |block|. Every edge carries the
// launch generation, so blocks only store indices, not per-edge payloads.
// hrx_graph_block_allocate sizes the allocation from this same order.
static inline void hrx_graph_block_get_ptrs(hrx_graph_exec_block_t* block,
                                            hrx_graph_block_ptrs_t* out_ptrs) {
  uint8_t* ptr = (uint8_t*)block + sizeof(*block);
  out_ptrs->wait_semaphore_indices = (uint32_t*)ptr;
  ptr += block->wait_semaphore_count * sizeof(uint32_t);
  out_ptrs->signal_semaphore_indices = (uint32_t*)ptr;
  out_ptrs->attrs = &block->attrs;
}

static iree_status_t hrx_graph_block_allocate(
    iree_arena_allocator_t* arena, hrx_graph_block_type_t type,
    uint32_t node_start_index, uint32_t node_count,
    uint16_t wait_semaphore_count, uint16_t signal_semaphore_count,
    hrx_graph_exec_block_t** out_block, hrx_graph_block_ptrs_t* out_ptrs) {
  // Region order matches hrx_graph_block_get_ptrs.
  iree_host_size_t size = sizeof(hrx_graph_exec_block_t);
  size += wait_semaphore_count * sizeof(uint32_t);
  size += signal_semaphore_count * sizeof(uint32_t);

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
// Dependency barriers selected by graph analysis
//===----------------------------------------------------------------------===//

static iree_status_t hrx_graph_record_dependency_barrier(
    iree_hal_command_buffer_t* command_buffer) {
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
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_barrier(command_buffer, &execution_barrier));
  return iree_ok_status();
}

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

  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; iree_status_is_ok(status) && i < node_count; ++i) {
    const uint32_t index = schedule->order[node_start_index + i];
    const uint8_t flags = schedule->flags[index];
    if ((flags >> HRX_GRAPH_ANALYSIS_NODE_STREAM_SHIFT) != stream_id) {
      continue;
    }
    hrx_graph_node_s* node = schedule->sources[index];
    if (flags & HRX_GRAPH_ANALYSIS_NODE_BARRIER_BEFORE) {
      status = hrx_graph_record_dependency_barrier(command_buffer);
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
    switch (node->type) {
      case HRX_GRAPH_NODE_TYPE_INTERNAL_KERNEL: {
        const hrx_graph_kernel_node_attrs_internal_t* attrs =
            &node->attrs.kernel;
        const iree_hal_dispatch_config_t config = {
            .workgroup_size = {attrs->block_dim[0], attrs->block_dim[1],
                               attrs->block_dim[2]},
            .workgroup_count = {attrs->grid_dim[0], attrs->grid_dim[1],
                                attrs->grid_dim[2]},
        };
        status = iree_hal_command_buffer_dispatch(
            command_buffer, attrs->executable,
            iree_hal_executable_function_from_index(attrs->export_ordinal),
            config, attrs->constants, attrs->bindings, attrs->flags);
        break;
      }
      case HRX_GRAPH_NODE_TYPE_INTERNAL_MEMCPY: {
        const hrx_graph_memcpy_node_attrs_internal_t* attrs =
            &node->attrs.memcpy;
        status = iree_hal_command_buffer_copy_buffer(
            command_buffer, attrs->src_ref, attrs->dst_ref, attrs->flags);
        break;
      }
      case HRX_GRAPH_NODE_TYPE_INTERNAL_MEMSET: {
        const hrx_graph_memset_node_attrs_internal_t* attrs =
            &node->attrs.memset;
        status = iree_hal_command_buffer_fill_buffer(
            command_buffer, attrs->dst_ref, &attrs->pattern,
            attrs->pattern_size, attrs->flags);
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

  if (schedule->partition_count == 0) {
    recording->block_count = 0;
    recording->blocks = NULL;
    recording->semaphore_count = 0;
    recording->semaphores = NULL;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  const hrx_graph_partition_t* final_partition =
      &schedule->partitions[schedule->partition_count - 1];
  const bool needs_final_join = final_partition->stream_count > 1;
  const iree_host_size_t signaling_partition_count =
      schedule->partition_count - (needs_final_join ? 0 : 1);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_arena_allocate(&recording->arena_allocator,
                          (schedule->block_count + (needs_final_join ? 1 : 0)) *
                              sizeof(*recording->blocks),
                          (void**)&recording->blocks));

  uint32_t semaphore_count = 0;
  for (iree_host_size_t i = 0; i < signaling_partition_count; ++i) {
    semaphore_count += schedule->partitions[i].stream_count;
  }
  if (semaphore_count > 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_arena_allocate(&recording->arena_allocator,
                            semaphore_count * sizeof(*recording->semaphores),
                            (void**)&recording->semaphores));
    // Each fresh semaphore transfers its sole retained reference to the set;
    // this array borrows those references, including on partial construction.
    memset(recording->semaphores, 0,
           semaphore_count * sizeof(*recording->semaphores));

    iree_hal_device_t* hal_device = recording->graph->device->hal_device;
    iree_status_t status = iree_ok_status();
    for (uint32_t i = 0; iree_status_is_ok(status) && i < semaphore_count;
         i++) {
      iree_hal_semaphore_t* semaphore = NULL;
      status = iree_hal_semaphore_create(
          hal_device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0ull,
          IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore);
      if (iree_status_is_ok(status)) {
        status = iree_hal_resource_set_insert(recording->resource_set, 1,
                                              &semaphore);
        iree_hal_semaphore_release(semaphore);
      }
      if (iree_status_is_ok(status)) {
        recording->semaphores[i] = semaphore;
      }
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
  }

  // Published only once every semaphore has been created, so a failure above
  // returns with the count still zero rather than describing an array of null
  // entries as populated.
  recording->semaphore_count = semaphore_count;

  uint32_t block_index = 0;
  uint32_t semaphore_index = 0;
  const iree_hal_command_category_t command_categories =
      hrx_queue_family_command_categories(queue_family);
  for (iree_host_size_t p = 0; p < schedule->partition_count; p++) {
    const hrx_graph_partition_t* partition = &schedule->partitions[p];

    const uint32_t first_index = schedule->order[partition->start_index];
    const uint8_t kind =
        schedule->flags[first_index] & HRX_GRAPH_ANALYSIS_NODE_KIND_MASK;
    uint16_t wait_semaphore_count = 0;
    if (p > 0) {
      const hrx_graph_partition_t* prev_partition =
          &schedule->partitions[p - 1];
      wait_semaphore_count =
          prev_partition->stream_count > 1 ? prev_partition->stream_count : 1;
    }

    uint16_t signal_semaphore_count = 0;
    if (p < signaling_partition_count) {
      signal_semaphore_count =
          partition->stream_count > 1 ? partition->stream_count : 1;
    }

    if (kind == HRX_GRAPH_ANALYSIS_NODE_RECORDABLE) {
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
            z0, hrx_graph_block_allocate(&recording->arena_allocator,
                                         HRX_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE,
                                         partition->start_index,
                                         partition->count, wait_semaphore_count,
                                         block_signal_count, &block, &ptrs));

        // The recording owns the command buffer; an accepted queue submission
        // retains it independently, including all of its explicit resources.
        iree_hal_command_buffer_t* command_buffer = NULL;
        iree_hal_command_buffer_mode_t command_buffer_mode =
            hrx_device_profile_command_buffer_mode(recording->graph->device);
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_command_buffer_create(
                    queue_family, command_buffer_mode, command_categories,
                    /*binding_capacity=*/0, &command_buffer));
        iree_status_t status = iree_hal_resource_set_insert(
            recording->resource_set, 1, &command_buffer);
        if (iree_status_is_ok(status)) {
          ptrs.attrs->execute.command_buffer = command_buffer;
          ptrs.attrs->execute.flags = IREE_HAL_QUEUE_EXECUTE_FLAG_NONE;
          status =
              hrx_graph_record_partition(schedule, partition->start_index,
                                         partition->count, s, command_buffer);
        }
        iree_hal_command_buffer_release(command_buffer);
        IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

        if (wait_semaphore_count > 0) {
          for (uint16_t w = 0; w < wait_semaphore_count; w++) {
            ptrs.wait_semaphore_indices[w] = partition_wait_semaphore_start + w;
          }
        }
        if (block_signal_count > 0) {
          if (stream_count > 1) {
            ptrs.signal_semaphore_indices[0] =
                partition_signal_semaphore_start + s;
          } else {
            for (uint16_t i = 0; i < block_signal_count; i++) {
              ptrs.signal_semaphore_indices[i] =
                  partition_signal_semaphore_start + i;
            }
          }
        }

        recording->blocks[block_index++] = block;
      }
      semaphore_index += signal_semaphore_count;
    } else {
      hrx_graph_exec_block_t* block = NULL;
      hrx_graph_block_ptrs_t ptrs;
      // Host calls are the only isolated operations; empty nodes have no
      // partition in the compiled schedule.
      hrx_graph_node_s* node = schedule->sources[first_index];
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, hrx_graph_block_allocate(&recording->arena_allocator,
                                       HRX_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL,
                                       partition->start_index, partition->count,
                                       wait_semaphore_count,
                                       signal_semaphore_count, &block, &ptrs));
      ptrs.attrs->host_call.fn = node->attrs.host.fn;
      ptrs.attrs->host_call.user_data = node->attrs.host.user_data;
      ptrs.attrs->host_call.flags = IREE_HAL_HOST_CALL_FLAG_NONE;
      if (wait_semaphore_count > 0) {
        for (uint16_t w = 0; w < wait_semaphore_count; w++) {
          ptrs.wait_semaphore_indices[w] =
              semaphore_index - wait_semaphore_count + w;
        }
      }
      if (signal_semaphore_count > 0) {
        for (uint16_t i = 0; i < signal_semaphore_count; i++) {
          ptrs.signal_semaphore_indices[i] = semaphore_index + i;
        }
      }
      semaphore_index += signal_semaphore_count;
      recording->blocks[block_index++] = block;
    }
  }

  if (needs_final_join) {
    hrx_graph_exec_block_t* block = NULL;
    hrx_graph_block_ptrs_t ptrs;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, hrx_graph_block_allocate(
                &recording->arena_allocator, HRX_GRAPH_BLOCK_TYPE_QUEUE_BARRIER,
                /*node_start_index=*/0, /*node_count=*/0,
                final_partition->stream_count, /*signal_semaphore_count=*/0,
                &block, &ptrs));
    ptrs.attrs->barrier.flags = IREE_HAL_QUEUE_BARRIER_FLAG_NONE;
    for (uint16_t i = 0; i < final_partition->stream_count; ++i) {
      ptrs.wait_semaphore_indices[i] =
          semaphore_count - final_partition->stream_count + i;
    }
    recording->blocks[block_index++] = block;
  }

  // Published only once every entry has been constructed. A failure above
  // returns with the count still zero rather than describing the uninitialized
  // arena bytes past the failure point as blocks.
  recording->block_count = block_index;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
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
  const uint64_t stream_signal_value = stream_wait_value + 1;

  // The partitioner bounds the widest join. Reuse these lists for every block
  // instead of accumulating graph-sized scratch on the stack.
  iree_hal_semaphore_t* wait_sems[HRX_GRAPH_MAX_WORKSTREAMS + 2];
  uint64_t wait_vals[IREE_ARRAYSIZE(wait_sems)];
  iree_hal_semaphore_t* signal_sems[1];
  uint64_t signal_vals[IREE_ARRAYSIZE(signal_sems)];
  iree_status_t status = iree_ok_status();

  for (uint32_t block_index = 0;
       iree_status_is_ok(status) && block_index < recording->block_count;
       block_index++) {
    hrx_graph_exec_block_t* block = recording->blocks[block_index];
    hrx_graph_block_ptrs_t ptrs;
    hrx_graph_block_get_ptrs(block, &ptrs);

    iree_host_size_t wait_count = 0;
    if (block->wait_semaphore_count == 0) {
      if (stream_wait_value > 0) {
        wait_sems[wait_count] = stream_semaphore;
        wait_vals[wait_count++] = stream_wait_value;
      }
      // On the same stream its tail already includes the preceding launch.
      if (exec->last_launch_semaphore &&
          exec->last_launch_semaphore != stream_semaphore) {
        wait_sems[wait_count] = exec->last_launch_semaphore;
        wait_vals[wait_count++] = exec->last_launch_value;
      }
    }
    for (uint16_t i = 0; i < block->wait_semaphore_count; ++i) {
      wait_sems[wait_count] =
          recording->semaphores[ptrs.wait_semaphore_indices[i]];
      wait_vals[wait_count++] = generation;
    }

    // Every block has exactly one completion producer: either an internal
    // edge or, for the single exit/join, the external stream timeline.
    if (block->signal_semaphore_count) {
      signal_sems[0] = recording->semaphores[ptrs.signal_semaphore_indices[0]];
      signal_vals[0] = generation;
    } else {
      signal_sems[0] = stream_semaphore;
      signal_vals[0] = stream_signal_value;
    }

    iree_hal_semaphore_list_t wait_semaphores = {
        .count = wait_count,
        .semaphores = wait_sems,
        .payload_values = wait_vals,
    };
    iree_hal_semaphore_list_t signal_semaphores = {
        .count = 1,
        .semaphores = signal_sems,
        .payload_values = signal_vals,
    };

    switch (block->type) {
      case HRX_GRAPH_BLOCK_TYPE_QUEUE_BARRIER: {
        // Empty graph nodes and terminal joins only order semaphore points.
        // Recorded work owns its acquire/release boundaries; repeating system
        // visibility here would add a cache flush to a pure dependency join.
        const iree_hal_barrier_list_t empty_list = {0};
        const iree_hal_queue_barriers_t barriers = {&empty_list, &empty_list};
        status = iree_hal_queue_barrier(stream->hal_queue, wait_semaphores,
                                        signal_semaphores, &barriers,
                                        ptrs.attrs->barrier.flags);
        break;
      }
      case HRX_GRAPH_BLOCK_TYPE_QUEUE_FILL:
        status = iree_hal_queue_fill(
            stream->hal_queue, wait_semaphores, signal_semaphores,
            ptrs.attrs->fill.target_buffer, ptrs.attrs->fill.target_offset,
            ptrs.attrs->fill.length, &ptrs.attrs->fill.pattern,
            ptrs.attrs->fill.pattern_length, /*barriers=*/NULL,
            ptrs.attrs->fill.flags);
        break;
      case HRX_GRAPH_BLOCK_TYPE_QUEUE_COPY:
        status = iree_hal_queue_copy(
            stream->hal_queue, wait_semaphores, signal_semaphores,
            ptrs.attrs->copy.source_buffer, ptrs.attrs->copy.source_offset,
            ptrs.attrs->copy.target_buffer, ptrs.attrs->copy.target_offset,
            ptrs.attrs->copy.length, /*barriers=*/NULL, ptrs.attrs->copy.flags);
        break;
      case HRX_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH: {
        iree_hal_buffer_ref_list_t bindings_list = {
            .count = ptrs.attrs->dispatch.bindings.count,
            .values = ptrs.attrs->dispatch.bindings.values,
        };
        status = iree_hal_queue_dispatch(
            stream->hal_queue, wait_semaphores, signal_semaphores,
            ptrs.attrs->dispatch.executable,
            iree_hal_executable_function_from_index(
                (uint32_t)ptrs.attrs->dispatch.entry_point),
            ptrs.attrs->dispatch.config, ptrs.attrs->dispatch.constants,
            bindings_list, /*barriers=*/NULL, ptrs.attrs->dispatch.flags);
        break;
      }
      case HRX_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE:
        status = iree_hal_queue_execute(stream->hal_queue, wait_semaphores,
                                        signal_semaphores,
                                        ptrs.attrs->execute.command_buffer,
                                        iree_hal_buffer_binding_table_empty(),
                                        IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
        break;
      case HRX_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL: {
        uint64_t call_args[4] = {
            (uint64_t)(uintptr_t)ptrs.attrs->host_call.fn,
            (uint64_t)(uintptr_t)ptrs.attrs->host_call.user_data,
        };
        status = iree_hal_queue_host_call(
            stream->hal_queue, wait_semaphores, signal_semaphores,
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

  iree_hal_semaphore_t* previous_semaphore =
      hrx_graph_exec_publish_launch(exec, stream, stream_signal_value, status);
  iree_slim_mutex_unlock(&exec->launch_mutex);
  iree_hal_semaphore_release(previous_semaphore);
  IREE_TRACE_ZONE_END(z0);
  return hrx_status_from_iree(status);
}
