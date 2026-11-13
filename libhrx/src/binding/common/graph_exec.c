// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/context.h"
#include "common/device.h"
#include "common/event.h"
#include "common/graph.h"
#include "common/graph_memory.h"
#include "common/graph_update.h"
#include "common/host_callback.h"
#include "common/module.h"
#include "common/occupancy.h"
#include "common/stream.h"
#include "common/stream_queue.h"
#include "common/stream_value.h"
#include "iree/base/api.h"
#include "iree/hal/utils/resource_set.h"
#include "libhrx/src/libhrx/device.h"
#include "libhrx/src/libhrx/graph_dependencies.h"

//===----------------------------------------------------------------------===//
// iree_hal_streaming_graph_exec_t (instantiation)
//===----------------------------------------------------------------------===//

typedef enum iree_hal_streaming_graph_block_type_e {
  // iree_hal_queue_barrier
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER = 0,
  // iree_hal_queue_host_call
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL,
  // Event record node.
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD,
  // Event wait node.
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_WAIT,
  // iree_hal_queue_dispatch
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH,
  // iree_hal_queue_execute
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE,
  // Native graph allocation on the graph timeline.
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_MEMORY_ALLOCATE,
  // Native graph free on the graph timeline.
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_MEMORY_FREE,
  // Nested graph executable.
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH,
} iree_hal_streaming_graph_block_type_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER
typedef struct iree_hal_streaming_graph_barrier_block_attrs_t {
  // HAL submission flags for this dependency-only block.
  iree_hal_queue_barrier_flags_t flags;
} iree_hal_streaming_graph_barrier_block_attrs_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL
typedef struct iree_hal_streaming_graph_host_call_block_attrs_t {
  // Host callback invoked when the queued host call reaches this block.
  iree_hal_streaming_host_callback_t callback;
  // User data passed to the host callback.
  void* user_data;
  // Owned parameter bytes captured by each invocation, or zero for a borrowed
  // application argument.
  iree_host_size_t user_data_size;
  // Parameter-referenced storage retained by the compiled resource set.
  iree_hal_resource_t* resource;
  // Host-call submission flags.
  iree_hal_host_call_flags_t flags;
} iree_hal_streaming_graph_host_call_block_attrs_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD/WAIT
typedef struct iree_hal_streaming_graph_event_block_attrs_t {
  // Source graph node this block was instantiated from.
  iree_hal_streaming_graph_node_t* source_node;
  // Event handle to record or wait on.
  iree_hal_streaming_event_t* event;
} iree_hal_streaming_graph_event_block_attrs_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH
typedef struct iree_hal_streaming_graph_dispatch_block_attrs_t {
  // Executable borrowed from the compiled resource set.
  iree_hal_executable_t* executable;
  // Export ordinal within the executable.
  iree_host_size_t entry_point;
  // Grid, workgroup, and dynamic local-memory dimensions.
  iree_hal_dispatch_config_t config;
  // Immutable argument bytes copied into the compiled arena.
  iree_const_byte_span_t constants;
  // Arena-owned binding descriptors with buffers held by the resource set.
  iree_hal_buffer_ref_list_t bindings;
  // Dispatch ABI and cooperative-execution requirements.
  iree_hal_dispatch_flags_t flags;
} iree_hal_streaming_graph_dispatch_block_attrs_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE
typedef struct iree_hal_streaming_graph_execute_block_attrs_t {
  // Recorded commands borrowed from the compiled resource set.
  iree_hal_command_buffer_t* command_buffer;
  // HAL execution flags captured at compilation.
  iree_hal_queue_execute_flags_t flags;
} iree_hal_streaming_graph_execute_block_attrs_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH
typedef struct iree_hal_streaming_graph_child_graph_block_attrs_t {
  // Owned child executable, released with this compiled block.
  iree_hal_streaming_graph_exec_t* exec;
} iree_hal_streaming_graph_child_graph_block_attrs_t;

// Block-specific data stored inline in the block header.
typedef union iree_hal_streaming_graph_block_attrs_t {
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER
  iree_hal_streaming_graph_barrier_block_attrs_t barrier;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL
  iree_hal_streaming_graph_host_call_block_attrs_t host_call;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD/WAIT
  iree_hal_streaming_graph_event_block_attrs_t event;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH
  iree_hal_streaming_graph_dispatch_block_attrs_t dispatch;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE
  iree_hal_streaming_graph_execute_block_attrs_t execute;
  // MEMORY_ALLOCATE/MEMORY_FREE; owner retained by the compiled resource set.
  struct {
    // Stable virtual address and current physical lifetime.
    iree_hal_streaming_graph_allocation_t* allocation;
    // Prepared allocation/free action and replay policy.
    iree_hal_streaming_graph_allocation_flags_t flags;
  } memory;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH
  iree_hal_streaming_graph_child_graph_block_attrs_t child_graph;
} iree_hal_streaming_graph_block_attrs_t;

// One recorded range, isolated frontend operation, or terminal dependency join.
typedef struct iree_hal_streaming_graph_block_t {
  // Operation interpreted by the frontend's submission policy.
  iree_hal_streaming_graph_block_type_t type;
  // Compact predecessor span relative to this block's submission ordinal.
  hrx_graph_dependencies_t dependencies;
  // Parameters owned by the executable and retained by accepted submissions.
  iree_hal_streaming_graph_block_attrs_t attrs;
} iree_hal_streaming_graph_block_t;

// Original handles remain stable even when template node indices are compacted.
// Entries follow source creation order, so stable source indices are sorted.
typedef struct iree_hal_streaming_graph_node_mapping_t {
  // Borrowed original handle, protected by source_graph.
  iree_hal_streaming_graph_node_t* source;
  // Borrowed executable-owned parameters, protected by graph.
  iree_hal_streaming_graph_node_t* target;
} iree_hal_streaming_graph_node_mapping_t;

typedef struct iree_hal_streaming_graph_exec_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;

  // Retained context owning compilation and queue resources.
  iree_hal_streaming_context_t* context;
  // Retained original template providing the public node-handle identity.
  iree_hal_streaming_graph_t* source_graph;
  // Owned private descriptions used for every compilation and update.
  iree_hal_streaming_graph_t* graph;
  // Source-to-private mappings sorted by stable source node index.
  iree_hal_streaming_graph_node_mapping_t* node_mappings;
  // Number of original nodes captured at instantiation.
  iree_host_size_t node_mapping_count;
  // True after the public HIP graph-exec handle has been destroyed.
  bool is_destroyed;

  // Arena allocator used for block allocations and inlined data.
  iree_arena_allocator_t arena_allocator;

  // Immutable block list created during instantiate.
  iree_hal_streaming_graph_block_t** blocks;
  // Initialized block entries, including during partial construction.
  uint32_t block_count;
  // True when |blocks|, or the blocks of a child-graph executable they launch,
  // hold an event record. Every such record names an event of |context|, so a
  // launch on a stream of any other context would have every one of its
  // records refused.
  bool records_events;
  // True when this graph or a child waits on an event whose producer queue can
  // change between launches. Such launches need independent queue progression.
  bool waits_for_events;
  // True when |blocks|, or the blocks of a child-graph executable they launch,
  // contain a cooperative dispatch requiring launch-time residency preflight.
  bool has_cooperative_dispatches;
  // Number of graph nodes present when this executable was instantiated.
  iree_host_size_t instantiated_node_count;
  // Per-instantiated-node disabled state keyed by graph node index.
  uint8_t* node_disabled_states;
  // Number of entries in |node_disabled_states|.
  iree_host_size_t node_disabled_state_count;

  // Number of compiled internal completion points (block_count minus one).
  uint32_t semaphore_count;
  // Contiguous internal completion semaphores borrowed from resource_set.
  iree_hal_semaphore_t** semaphores;
  // Reserved once per invocation, including a partially rejected launch.
  uint64_t launch_generation;

  // Resource set for automatic cleanup.
  iree_hal_resource_set_t* resource_set;

  // Stable logical root identity shared by children and rebuild candidates.
  // Borrowed by children, which are owned by the root's compiled blocks.
  iree_hal_streaming_graph_exec_t* execution_owner;
  // Arena-owned admission claims, one per original direct allocation owner.
  struct iree_hal_streaming_graph_memory_claim_t* memory_claims;

  // Completion of the last accepted launch, preserved through parameter edits.
  struct {
    // Retained stream timeline; NULL before the first accepted launch.
    iree_hal_semaphore_t* semaphore;
    // Value proving all workstreams of that launch completed.
    uint64_t value;
    // A dependent stream must not block the shared producer queue.
    bool requires_independent_queue;
  } last_launch;

  unsigned long long flags;

  // Mutex needed for launch/update.
  iree_slim_mutex_t mutex;
} iree_hal_streaming_graph_exec_t;

typedef struct iree_hal_streaming_graph_memory_claim_t {
  // Next admission held by this compiled graph version.
  struct iree_hal_streaming_graph_memory_claim_t* next;
  // Original allocation owner, borrowed through the retained template chain.
  iree_hal_streaming_graph_t* graph;
} iree_hal_streaming_graph_memory_claim_t;

static iree_status_t iree_hal_streaming_graph_record_memcpy_node(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_streaming_graph_memcpy_node_attrs_t* attrs) {
  const iree_device_size_t width =
      attrs->execution_extent_width
          ? attrs->execution_extent_width
          : (attrs->hip_extent_width ? attrs->hip_extent_width : attrs->size);
  const iree_device_size_t height =
      attrs->execution_extent_height
          ? attrs->execution_extent_height
          : (attrs->hip_extent_height ? attrs->hip_extent_height : 1);
  const iree_device_size_t depth =
      attrs->execution_extent_depth
          ? attrs->execution_extent_depth
          : (attrs->hip_extent_depth ? attrs->hip_extent_depth : 1);
  if (width == 0 || height == 0 || depth == 0) {
    return iree_ok_status();
  }

  const iree_device_size_t src_pitch =
      attrs->execution_src_pitch
          ? attrs->execution_src_pitch
          : (attrs->hip_src_pitch ? attrs->hip_src_pitch : width);
  const iree_device_size_t dst_pitch =
      attrs->execution_dst_pitch
          ? attrs->execution_dst_pitch
          : (attrs->hip_dst_pitch ? attrs->hip_dst_pitch : width);
  const iree_device_size_t src_rows_per_slice =
      attrs->execution_src_ysize
          ? attrs->execution_src_ysize
          : (attrs->hip_src_ysize ? attrs->hip_src_ysize : height);
  const iree_device_size_t dst_rows_per_slice =
      attrs->execution_dst_ysize
          ? attrs->execution_dst_ysize
          : (attrs->hip_dst_ysize ? attrs->hip_dst_ysize : height);

  iree_device_size_t total_size = 0;
  iree_device_size_t src_slice_pitch = 0;
  iree_device_size_t dst_slice_pitch = 0;
  if (IREE_UNLIKELY(
          !iree_device_size_checked_mul(width, height, &total_size) ||
          !iree_device_size_checked_mul(total_size, depth, &total_size) ||
          !iree_device_size_checked_mul(src_pitch, src_rows_per_slice,
                                        &src_slice_pitch) ||
          !iree_device_size_checked_mul(dst_pitch, dst_rows_per_slice,
                                        &dst_slice_pitch))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "graph memcpy node geometry overflows");
  }

  const bool compact_layout = src_pitch == width && dst_pitch == width &&
                              src_rows_per_slice == height &&
                              dst_rows_per_slice == height;
  if (compact_layout) {
    if (attrs->src_ref.buffer == attrs->dst_ref.buffer &&
        attrs->src_ref.offset == attrs->dst_ref.offset) {
      return iree_ok_status();
    }
    return iree_hal_command_buffer_copy_buffer(
        command_buffer,
        iree_hal_streaming_convert_range_buffer_ref(attrs->src_ref, total_size),
        iree_hal_streaming_convert_range_buffer_ref(attrs->dst_ref, total_size),
        attrs->flags);
  }

  for (iree_device_size_t z = 0; z < depth; ++z) {
    iree_device_size_t src_slice_offset = 0;
    iree_device_size_t dst_slice_offset = 0;
    if (IREE_UNLIKELY(!iree_device_size_checked_mul(z, src_slice_pitch,
                                                    &src_slice_offset) ||
                      !iree_device_size_checked_mul(z, dst_slice_pitch,
                                                    &dst_slice_offset))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "graph memcpy slice offset overflows");
    }
    for (iree_device_size_t y = 0; y < height; ++y) {
      iree_device_size_t src_row_offset = 0;
      iree_device_size_t dst_row_offset = 0;
      iree_device_size_t src_offset = 0;
      iree_device_size_t dst_offset = 0;
      if (IREE_UNLIKELY(
              !iree_device_size_checked_mul(y, src_pitch, &src_row_offset) ||
              !iree_device_size_checked_mul(y, dst_pitch, &dst_row_offset) ||
              !iree_device_size_checked_add(src_slice_offset, src_row_offset,
                                            &src_offset) ||
              !iree_device_size_checked_add(dst_slice_offset, dst_row_offset,
                                            &dst_offset) ||
              !iree_device_size_checked_add(attrs->src_ref.offset, src_offset,
                                            &src_offset) ||
              !iree_device_size_checked_add(attrs->dst_ref.offset, dst_offset,
                                            &dst_offset))) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "graph memcpy row offset overflows");
      }
      if (attrs->src_ref.buffer == attrs->dst_ref.buffer &&
          src_offset == dst_offset) {
        continue;
      }
      iree_hal_streaming_buffer_ref_t src_ref = attrs->src_ref;
      src_ref.offset = src_offset;
      iree_hal_streaming_buffer_ref_t dst_ref = attrs->dst_ref;
      dst_ref.offset = dst_offset;
      IREE_RETURN_IF_ERROR(iree_hal_command_buffer_copy_buffer(
          command_buffer,
          iree_hal_streaming_convert_range_buffer_ref(src_ref, width),
          iree_hal_streaming_convert_range_buffer_ref(dst_ref, width),
          attrs->flags));
    }
  }
  return iree_ok_status();
}

static void iree_hal_streaming_graph_exec_destroy(
    iree_hal_streaming_graph_exec_t* exec);
static void iree_hal_streaming_graph_exec_release_child_blocks(
    iree_hal_streaming_graph_exec_t* exec);
static void iree_hal_streaming_graph_exec_deinitialize_compiled_state(
    iree_hal_streaming_graph_exec_t* exec);

// Compilation acquires each original allocation owner once. Rebuilds and
// nested graphs share the same logical executable identity, while independent
// executables cannot concurrently reuse these fixed graph allocation addresses.
static iree_status_t iree_hal_streaming_graph_memory_acquire_claim(
    iree_hal_streaming_graph_exec_t* exec, iree_hal_streaming_graph_t* graph) {
  for (iree_hal_streaming_graph_memory_claim_t* claim = exec->memory_claims;
       claim; claim = claim->next) {
    if (claim->graph == graph) {
      return iree_ok_status();
    }
  }
  iree_hal_streaming_graph_memory_claim_t* claim = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&exec->arena_allocator,
                                           sizeof(*claim), (void**)&claim));
  iree_slim_mutex_lock(&graph->capture_mutex);
  const bool available =
      !graph->memory_admission.executable ||
      graph->memory_admission.executable == exec->execution_owner;
  if (available) {
    graph->memory_admission.executable = exec->execution_owner;
    ++graph->memory_admission.count;
  }
  iree_slim_mutex_unlock(&graph->capture_mutex);
  if (!available) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "graph allocation nodes already belong to another executable");
  }
  claim->graph = graph;
  claim->next = exec->memory_claims;
  exec->memory_claims = claim;
  return iree_ok_status();
}

static void iree_hal_streaming_graph_memory_release_claims(
    iree_hal_streaming_graph_exec_t* exec) {
  for (iree_hal_streaming_graph_memory_claim_t* claim = exec->memory_claims;
       claim; claim = claim->next) {
    iree_hal_streaming_graph_t* graph = claim->graph;
    iree_slim_mutex_lock(&graph->capture_mutex);
    if (--graph->memory_admission.count == 0) {
      graph->memory_admission.executable = NULL;
    }
    iree_slim_mutex_unlock(&graph->capture_mutex);
  }
  exec->memory_claims = NULL;
}

static iree_status_t iree_hal_streaming_graph_memory_acquire_claims(
    iree_hal_streaming_graph_exec_t* exec) {
  for (iree_hal_streaming_node_block_t* block = exec->graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC) {
        IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_memory_acquire_claim(
            exec,
            iree_hal_streaming_graph_node_mem_alloc(node)->allocation_owner
                ? iree_hal_streaming_graph_node_mem_alloc(node)
                      ->allocation_owner
                : node->graph));
      }
    }
  }
  return iree_ok_status();
}

// Internal: Create an exec object (called by graph.c).
iree_status_t iree_hal_streaming_graph_exec_create(
    iree_hal_streaming_context_t* context, iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_instantiate_flags_t flags,
    iree_hal_streaming_graph_exec_t* execution_owner,
    iree_allocator_t host_allocator,
    iree_hal_streaming_graph_exec_t** out_exec) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(out_exec);
  *out_exec = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_graph_exec_t* exec = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*exec), (void**)&exec));

  iree_atomic_ref_count_init(&exec->ref_count);
  exec->host_allocator = host_allocator;
  exec->context = context;
  iree_hal_streaming_context_retain(exec->context);
  exec->source_graph = graph;
  iree_hal_streaming_graph_retain(exec->source_graph);
  exec->is_destroyed = false;
  iree_arena_initialize(&context->device_entry->block_pool,
                        &exec->arena_allocator);
  exec->blocks = NULL;
  exec->block_count = 0;
  exec->records_events = false;
  exec->waits_for_events = false;
  exec->has_cooperative_dispatches = false;
  exec->instantiated_node_count = 0;
  exec->node_disabled_states = NULL;
  exec->node_disabled_state_count = graph->node_count;
  exec->semaphores = NULL;
  exec->semaphore_count = 0;
  exec->launch_generation = 0;
  exec->resource_set = NULL;
  exec->execution_owner = execution_owner ? execution_owner : exec;
  exec->memory_claims = NULL;
  exec->last_launch.semaphore = NULL;
  exec->last_launch.value = 0;
  exec->last_launch.requires_independent_queue = false;
  exec->flags = flags;
  iree_slim_mutex_initialize(&exec->mutex);

  iree_status_t status = iree_ok_status();
  if (graph->is_executable_snapshot) {
    // A compiled child's descriptions were already captured recursively by
    // its parent. Share that private snapshot instead of copying each subtree
    // again at every level of nesting.
    exec->graph = graph;
    iree_hal_streaming_graph_retain(graph);
  } else {
    status = iree_hal_streaming_graph_clone(
        graph, IREE_HAL_STREAMING_GRAPH_CLONE_MODE_EXECUTABLE, &exec->graph);
  }
  if (iree_status_is_ok(status) && graph->node_count > 0) {
    status = iree_allocator_malloc(
        host_allocator, graph->node_count * sizeof(*exec->node_mappings),
        (void**)&exec->node_mappings);
  }
  if (iree_status_is_ok(status)) {
    exec->node_mapping_count = graph->node_count;
    for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
         block = block->next) {
      for (iree_host_size_t i = 0; i < block->count; ++i) {
        iree_hal_streaming_graph_node_t* node = block->nodes[i];
        exec->node_mappings[node->node_index].source = node;
      }
    }
    for (iree_hal_streaming_node_block_t* block = exec->graph->node_blocks;
         block; block = block->next) {
      for (iree_host_size_t i = 0; i < block->count; ++i) {
        iree_hal_streaming_graph_node_t* node = block->nodes[i];
        exec->node_mappings[node->node_index].target = node;
      }
    }
    status = iree_hal_resource_set_allocate(&context->device_entry->block_pool,
                                            &exec->resource_set);
  }
  if (iree_status_is_ok(status) && exec->node_disabled_state_count > 0) {
    status = iree_allocator_malloc(
        host_allocator,
        exec->node_disabled_state_count * sizeof(*exec->node_disabled_states),
        (void**)&exec->node_disabled_states);
    if (iree_status_is_ok(status)) {
      memset(exec->node_disabled_states, 0,
             exec->node_disabled_state_count *
                 sizeof(*exec->node_disabled_states));
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_memory_acquire_claims(exec);
  }

  if (iree_status_is_ok(status)) {
    *out_exec = exec;
  } else {
    iree_hal_streaming_graph_exec_destroy(exec);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_streaming_graph_exec_destroy(
    iree_hal_streaming_graph_exec_t* exec) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_graph_exec_deinitialize_compiled_state(exec);

  iree_hal_semaphore_release(exec->last_launch.semaphore);

  iree_allocator_free(exec->host_allocator, exec->node_disabled_states);
  iree_allocator_free(exec->host_allocator, exec->node_mappings);

  iree_hal_streaming_graph_release(exec->graph);
  iree_hal_streaming_graph_release(exec->source_graph);
  iree_hal_streaming_context_release(exec->context);
  iree_slim_mutex_deinitialize(&exec->mutex);

  iree_allocator_t host_allocator = exec->host_allocator;
  iree_allocator_free(host_allocator, exec);

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_streaming_graph_exec_release_child_blocks(
    iree_hal_streaming_graph_exec_t* exec) {
  for (uint32_t i = 0; exec->blocks && i < exec->block_count; ++i) {
    iree_hal_streaming_graph_block_t* block = exec->blocks[i];
    switch (block->type) {
      case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH:
        iree_hal_streaming_graph_exec_release(block->attrs.child_graph.exec);
        block->attrs.child_graph.exec = NULL;
        break;
      case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD:
      case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_WAIT:
        iree_hal_streaming_event_release(block->attrs.event.event);
        block->attrs.event.event = NULL;
        break;
      default:
        break;
    }
  }
}

static void iree_hal_streaming_graph_exec_initialize_compiled_state(
    iree_hal_streaming_graph_exec_t* exec) {
  iree_arena_initialize(&exec->context->device_entry->block_pool,
                        &exec->arena_allocator);
  exec->blocks = NULL;
  exec->block_count = 0;
  exec->records_events = false;
  exec->waits_for_events = false;
  exec->has_cooperative_dispatches = false;
  exec->instantiated_node_count = 0;
  exec->semaphores = NULL;
  exec->semaphore_count = 0;
  exec->resource_set = NULL;
  exec->memory_claims = NULL;
}

static void iree_hal_streaming_graph_exec_deinitialize_compiled_state(
    iree_hal_streaming_graph_exec_t* exec) {
  iree_hal_streaming_graph_exec_release_child_blocks(exec);
  if (exec->resource_set) {
    iree_hal_resource_set_free(exec->resource_set);
    exec->resource_set = NULL;
  }
  iree_hal_streaming_graph_memory_release_claims(exec);
  iree_arena_deinitialize(&exec->arena_allocator);
  exec->blocks = NULL;
  exec->block_count = 0;
  exec->records_events = false;
  exec->waits_for_events = false;
  exec->has_cooperative_dispatches = false;
  exec->instantiated_node_count = 0;
  exec->semaphores = NULL;
  exec->semaphore_count = 0;
}

static void iree_hal_streaming_graph_exec_move_compiled_state(
    iree_hal_streaming_graph_exec_t* target,
    iree_hal_streaming_graph_exec_t* source) {
  target->arena_allocator = source->arena_allocator;
  target->blocks = source->blocks;
  target->block_count = source->block_count;
  target->records_events = source->records_events;
  target->waits_for_events = source->waits_for_events;
  target->has_cooperative_dispatches = source->has_cooperative_dispatches;
  target->instantiated_node_count = source->instantiated_node_count;
  target->semaphores = source->semaphores;
  target->semaphore_count = source->semaphore_count;
  target->resource_set = source->resource_set;
  target->memory_claims = source->memory_claims;
  source->memory_claims = NULL;

  source->blocks = NULL;
  source->block_count = 0;
  source->records_events = false;
  source->waits_for_events = false;
  source->has_cooperative_dispatches = false;
  source->instantiated_node_count = 0;
  source->semaphores = NULL;
  source->semaphore_count = 0;
  source->resource_set = NULL;
}

void iree_hal_streaming_graph_exec_retain(
    iree_hal_streaming_graph_exec_t* exec) {
  if (exec) {
    iree_atomic_ref_count_inc(&exec->ref_count);
  }
}

void iree_hal_streaming_graph_exec_release(
    iree_hal_streaming_graph_exec_t* exec) {
  if (exec && iree_atomic_ref_count_dec(&exec->ref_count) == 1) {
    iree_hal_streaming_graph_exec_destroy(exec);
  }
}

bool iree_hal_streaming_graph_exec_try_retain_live(
    iree_hal_streaming_graph_exec_t* exec) {
  if (!exec) {
    return false;
  }
  iree_slim_mutex_lock(&exec->mutex);
  const bool is_live = !exec->is_destroyed;
  if (is_live) {
    iree_hal_streaming_graph_exec_retain(exec);
  }
  iree_slim_mutex_unlock(&exec->mutex);
  return is_live;
}

bool iree_hal_streaming_graph_exec_is_live(
    iree_hal_streaming_graph_exec_t* exec) {
  return exec && !exec->is_destroyed;
}

iree_status_t iree_hal_streaming_graph_exec_destroy_handle(
    iree_hal_streaming_graph_exec_t* exec) {
  if (!exec) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }

  iree_slim_mutex_lock(&exec->mutex);
  if (exec->is_destroyed) {
    iree_slim_mutex_unlock(&exec->mutex);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  exec->is_destroyed = true;
  iree_hal_semaphore_t* semaphore = exec->last_launch.semaphore;
  const uint64_t value = exec->last_launch.value;
  iree_slim_mutex_unlock(&exec->mutex);

  // Replays carry the previous launch into their dependencies, and the final
  // signal joins every graph partition. This point therefore covers every
  // accepted launch, including work compiled before a parameter update. Later
  // stream operations and independent contexts are outside this lifetime.
  // The executable owns the semaphore and context throughout the unlocked wait.
  iree_status_t status = iree_ok_status();
  if (semaphore) {
    status = iree_hal_semaphore_wait(
        semaphore, value, iree_infinite_timeout(),
        iree_hal_streaming_context_wait_flags(exec->context));
  }
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&exec->mutex);
    exec->is_destroyed = false;
    iree_slim_mutex_unlock(&exec->mutex);
    return status;
  }

  iree_hal_streaming_graph_exec_release(exec);
  return iree_ok_status();
}

iree_hal_streaming_graph_instantiate_flags_t
iree_hal_streaming_graph_exec_flags(iree_hal_streaming_graph_exec_t* exec) {
  IREE_ASSERT_ARGUMENT(exec);
  return (iree_hal_streaming_graph_instantiate_flags_t)exec->flags;
}

static iree_hal_streaming_graph_node_t*
iree_hal_streaming_graph_exec_lookup_node(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node) {
  if (!exec || !node || node->graph != exec->source_graph) {
    return NULL;
  }
  iree_host_size_t first = 0;
  iree_host_size_t last = exec->node_mapping_count;
  while (first < last) {
    const iree_host_size_t middle = first + (last - first) / 2;
    const iree_hal_streaming_graph_node_mapping_t* mapping =
        &exec->node_mappings[middle];
    if (mapping->source->clone_source_node_index <
        node->clone_source_node_index) {
      first = middle + 1;
    } else {
      last = middle;
    }
  }
  return first < exec->node_mapping_count &&
                 exec->node_mappings[first].source == node
             ? exec->node_mappings[first].target
             : NULL;
}

bool iree_hal_streaming_graph_exec_owns_node(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node) {
  if (!exec) {
    return false;
  }
  iree_slim_mutex_lock(&exec->mutex);
  const bool owns_node =
      iree_hal_streaming_graph_exec_lookup_node(exec, node) != NULL;
  iree_slim_mutex_unlock(&exec->mutex);
  return owns_node;
}

iree_hal_streaming_graph_node_t* iree_hal_streaming_graph_exec_lock_node(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node) {
  iree_slim_mutex_lock(&exec->mutex);
  return exec->is_destroyed
             ? NULL
             : iree_hal_streaming_graph_exec_lookup_node(exec, node);
}

void iree_hal_streaming_graph_exec_unlock(
    iree_hal_streaming_graph_exec_t* exec) {
  iree_slim_mutex_unlock(&exec->mutex);
}

bool iree_hal_streaming_graph_exec_node_is_enabled(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node) {
  node = iree_hal_streaming_graph_exec_lock_node(exec, node);
  const bool enabled =
      node && exec->node_disabled_states[node->node_index] == 0;
  iree_hal_streaming_graph_exec_unlock(exec);
  return enabled;
}

iree_status_t iree_hal_streaming_graph_exec_set_node_enabled(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node, bool enabled) {
  if (!exec || !node) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  node = iree_hal_streaming_graph_exec_lock_node(exec, node);
  if (!node || node->node_index >= exec->node_disabled_state_count) {
    iree_slim_mutex_unlock(&exec->mutex);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  const uint8_t old_disabled_state =
      exec->node_disabled_states[node->node_index];
  exec->node_disabled_states[node->node_index] = enabled ? 0 : 1;
  iree_status_t status = iree_hal_streaming_graph_exec_rebuild_locked(exec);
  if (!iree_status_is_ok(status)) {
    exec->node_disabled_states[node->node_index] = old_disabled_state;
  }
  iree_slim_mutex_unlock(&exec->mutex);
  return status;
}

iree_status_t iree_hal_streaming_graph_exec_rebuild_locked(
    iree_hal_streaming_graph_exec_t* exec) {
  if (exec->is_destroyed) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }

  iree_hal_streaming_graph_exec_t candidate;
  memset(&candidate, 0, sizeof(candidate));
  candidate.host_allocator = exec->host_allocator;
  candidate.context = exec->context;
  candidate.graph = exec->graph;
  candidate.node_disabled_states = exec->node_disabled_states;
  candidate.node_disabled_state_count = exec->node_disabled_state_count;
  candidate.flags = exec->flags;
  candidate.execution_owner = exec->execution_owner;
  iree_hal_streaming_graph_exec_initialize_compiled_state(&candidate);

  iree_status_t status = iree_hal_resource_set_allocate(
      &candidate.context->device_entry->block_pool, &candidate.resource_set);
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_memory_acquire_claims(&candidate);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_exec_compile(&candidate);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_streaming_graph_exec_deinitialize_compiled_state(exec);
    iree_hal_streaming_graph_exec_move_compiled_state(exec, &candidate);
  } else {
    iree_hal_streaming_graph_exec_deinitialize_compiled_state(&candidate);
  }
  return status;
}

iree_status_t iree_hal_streaming_graph_exec_set_event_node_event(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node,
    iree_hal_streaming_graph_node_type_t type,
    iree_hal_streaming_event_t* event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!exec || !node || !event ||
      (type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD &&
       type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  // The same rule iree_hal_streaming_graph_add_event_node holds the template
  // to, applied to the executable an event node is retargeted in. On a record
  // node it is what lets a launch answer for every record block by comparing
  // one pair of contexts, and an event of another context would be refused at
  // the record itself anyway. A wait node is held to it because that function
  // holds both node types to it: retargeting is another way of naming a node's
  // event, and it must not seat one the template would have refused.
  if (event->context != exec->context) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "event must belong to the graph context");
  }

  const iree_hal_streaming_graph_block_type_t block_type =
      type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD
          ? IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD
          : IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_WAIT;
  bool found = false;
  iree_hal_streaming_event_retain(event);
  iree_hal_streaming_event_t* old_event = NULL;
  node = iree_hal_streaming_graph_exec_lock_node(exec, node);
  if (node) {
    for (uint32_t i = 0; i < exec->block_count; ++i) {
      iree_hal_streaming_graph_block_t* block = exec->blocks[i];
      if (block->type != block_type) {
        continue;
      }
      if (block->attrs.event.source_node != node) {
        continue;
      }
      old_event = block->attrs.event.event;
      block->attrs.event.event = event;
      iree_hal_streaming_event_retain(event);
      iree_hal_streaming_event_release(
          iree_hal_streaming_graph_node_event(node)->event);
      iree_hal_streaming_graph_node_event(node)->event = event;
      found = true;
      break;
    }
  }
  iree_slim_mutex_unlock(&exec->mutex);
  if (found) {
    iree_hal_streaming_event_release(old_event);
  } else {
    iree_hal_streaming_event_release(event);
  }

  IREE_TRACE_ZONE_END(z0);
  return found ? iree_ok_status()
               : iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
}

static iree_status_t iree_hal_streaming_graph_block_allocate(
    iree_arena_allocator_t* arena_allocator,
    iree_hal_streaming_graph_block_type_t type,
    hrx_graph_dependencies_t dependencies,
    iree_hal_streaming_graph_block_t** out_block) {
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena_allocator, sizeof(*block), (void**)&block));
  block->type = type;
  block->dependencies = dependencies;
  *out_block = block;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_allocate_barrier_block(
    iree_hal_streaming_graph_exec_t* exec,
    hrx_graph_dependencies_t dependencies, iree_hal_queue_barrier_flags_t flags,
    iree_hal_streaming_graph_block_t** out_block) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the operation and its compact predecessor span.
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER, dependencies,
              &block));

  // Set barrier attributes.
  iree_hal_streaming_graph_barrier_block_attrs_t* attrs = &block->attrs.barrier;
  attrs->flags = flags;

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_allocate_host_call_block(
    iree_hal_streaming_graph_exec_t* exec,
    hrx_graph_dependencies_t dependencies,
    iree_hal_streaming_host_callback_t callback, void* user_data,
    const iree_hal_streaming_graph_node_t* source_node,
    iree_hal_host_call_flags_t flags,
    iree_hal_streaming_graph_block_t** out_block) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the operation and its compact predecessor span.
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL, dependencies,
              &block));

  // Set host call attributes.
  iree_hal_streaming_graph_host_call_block_attrs_t* attrs =
      &block->attrs.host_call;
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
        z0,
        iree_hal_resource_set_insert(exec->resource_set, 1, &attrs->resource));
  }
  if (source_node &&
      iree_hal_streaming_graph_node_host(source_node)->user_data_size > 0 &&
      user_data) {
    void* copied_data = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(
                &exec->arena_allocator,
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
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_block_type_t type,
    hrx_graph_dependencies_t dependencies,
    iree_hal_streaming_graph_node_t* source_node,
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_graph_block_t** out_block) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(&exec->arena_allocator, type,
                                                  dependencies, &block));

  block->attrs.event.source_node = source_node;
  block->attrs.event.event = event;
  iree_hal_streaming_event_retain(event);
  // The one place a record block is built, and so where |exec| learns it holds
  // one. A launch reads this to answer for every record at once.
  if (type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD) {
    exec->records_events = true;
  } else {
    exec->waits_for_events = true;
  }

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_allocate_dispatch_block(
    iree_hal_streaming_graph_exec_t* exec,
    hrx_graph_dependencies_t dependencies, iree_hal_executable_t* executable,
    iree_host_size_t entry_point, iree_hal_dispatch_config_t config,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags,
    iree_hal_streaming_graph_block_t** out_block) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the operation and its compact predecessor span.
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH, dependencies,
              &block));

  // Set dispatch attributes.
  iree_hal_streaming_graph_dispatch_block_attrs_t* attrs =
      &block->attrs.dispatch;
  attrs->executable = executable;
  attrs->entry_point = entry_point;
  attrs->config = config;
  attrs->flags = flags;

  // Copy constants if provided.
  if (constants.data_length > 0) {
    void* constants_copy = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(&exec->arena_allocator, constants.data_length,
                                &constants_copy));
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
        z0, iree_arena_allocate(&exec->arena_allocator, bindings_size,
                                (void**)&bindings_copy));
    memcpy(bindings_copy, bindings.values, bindings_size);
    attrs->bindings.count = bindings.count;
    attrs->bindings.values = bindings_copy;

    // Add buffers to resource set.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_resource_set_insert_strided(
                exec->resource_set, bindings.count, bindings.values,
                offsetof(iree_hal_buffer_ref_t, buffer),
                sizeof(iree_hal_buffer_ref_t)));
  } else {
    attrs->bindings = iree_hal_buffer_ref_list_empty();
  }

  // Add executable to resource set.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(exec->resource_set, 1, &executable));

  if (iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_COOPERATIVE)) {
    exec->has_cooperative_dispatches = true;
  }
  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_allocate_execute_block(
    iree_hal_streaming_graph_exec_t* exec,
    hrx_graph_dependencies_t dependencies, iree_hal_queue_execute_flags_t flags,
    iree_hal_streaming_graph_block_t** out_block) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the operation and its compact predecessor span.
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE, dependencies,
              &block));

  iree_hal_streaming_graph_execute_block_attrs_t* attrs = &block->attrs.execute;
  attrs->flags = flags;

  // Parameter updates can replace this compiled graph while an older launch
  // is pending. Each command buffer retains its recorded buffers/executables
  // until all accepted submissions retire. Raw HIP pointers in kernel
  // arguments remain governed by the application's allocation lifetime.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_create(
              iree_hal_queue_family(exec->context->queue),
              hrx_device_profile_command_buffer_mode(
                  exec->context->device_entry->hrx_device),
              IREE_HAL_COMMAND_CATEGORY_TRANSFER |
                  IREE_HAL_COMMAND_CATEGORY_DISPATCH |
                  IREE_HAL_COMMAND_CATEGORY_ATOMIC,
              /*binding_capacity=*/0, &attrs->command_buffer));

  // Add to resource set for cleanup.
  iree_status_t status = iree_hal_resource_set_insert(exec->resource_set, 1,
                                                      &attrs->command_buffer);

  // We don't technically retain a reference to it past here on the stack, just
  // in the resource set associated with the exec.
  iree_hal_command_buffer_release(attrs->command_buffer);

  if (iree_status_is_ok(status)) {
    *out_block = block;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_streaming_graph_allocate_child_graph_block(
    iree_hal_streaming_graph_exec_t* exec,
    hrx_graph_dependencies_t dependencies,
    iree_hal_streaming_graph_t* child_graph,
    iree_hal_streaming_graph_block_t** out_block) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_graph_exec_t* child_exec = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_exec_create(
              exec->context, child_graph,
              (iree_hal_streaming_graph_instantiate_flags_t)exec->flags,
              exec->execution_owner, exec->host_allocator, &child_exec));

  iree_hal_streaming_graph_block_t* block = NULL;
  iree_status_t status = iree_hal_streaming_graph_exec_compile(child_exec);
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_block_allocate(
        &exec->arena_allocator, IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH,
        dependencies, &block);
  }
  if (iree_status_is_ok(status)) {
    block->attrs.child_graph.exec = child_exec;
    // A launch of |exec| walks the child's blocks too, so the records they hold
    // are records of this launch. The child instantiated above already carries
    // its own children's, which makes the property transitive.
    exec->records_events |= child_exec->records_events;
    exec->waits_for_events |= child_exec->waits_for_events;
    exec->has_cooperative_dispatches |= child_exec->has_cooperative_dispatches;
    *out_block = block;
  } else {
    iree_hal_streaming_graph_exec_release(child_exec);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_streaming_graph_record_dependency_barrier(
    iree_hal_command_buffer_t* command_buffer, iree_hal_barrier_flags_t flags) {
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
      .flags = flags,
      .memory_barrier_count = 1,
      .memory_barriers = &memory_barrier,
  };
  return iree_hal_command_buffer_barrier(command_buffer, &execution_barrier);
}

// Helper to record nodes from a partition into a command buffer.
static iree_status_t iree_hal_streaming_graph_record_partition(
    iree_hal_streaming_graph_exec_t* exec, const hrx_graph_schedule_t* schedule,
    uint32_t node_start_index, uint32_t node_count, uint8_t stream_id,
    iree_hal_command_buffer_t* command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Begin recording command buffer.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_begin(command_buffer));

  // Scope the partition into a debug group.
  // TODO: propagate graph information (name, origin, etc).
  const iree_string_view_t label_name = iree_make_cstring_view("tbd_partition");
  const iree_hal_label_location_t* location = NULL;
  const iree_hal_label_color_t label_color = iree_hal_label_color_unspecified();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_begin_debug_group(command_buffer, label_name,
                                                    label_color, location));

  iree_status_t status = iree_ok_status();
  bool release_to_host = false;
  for (uint32_t i = 0; iree_status_is_ok(status) && i < node_count; ++i) {
    const uint32_t index = schedule->order[node_start_index + i];
    const uint8_t flags = schedule->flags[index];
    if ((flags >> HRX_GRAPH_ANALYSIS_NODE_STREAM_SHIFT) != stream_id) {
      continue;
    }
    iree_hal_streaming_graph_node_t* node = schedule->sources[index];
    if (flags & HRX_GRAPH_ANALYSIS_NODE_BARRIER_BEFORE) {
      status = iree_hal_streaming_graph_record_dependency_barrier(
          command_buffer, IREE_HAL_BARRIER_FLAG_NONE);
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
    switch (node->type) {
      case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL: {
        const iree_hal_streaming_graph_kernel_node_attrs_t* attrs =
            iree_hal_streaming_graph_node_kernel(node);
        iree_hal_streaming_symbol_t* symbol = attrs->symbol;
        const iree_hal_dispatch_config_t config = {
            .workgroup_size =
                {
                    attrs->block_dim[0],
                    attrs->block_dim[1],
                    attrs->block_dim[2],
                },
            .workgroup_count =
                {
                    attrs->grid_dim[0],
                    attrs->grid_dim[1],
                    attrs->grid_dim[2],
                },
            .workitem_count =
                {
                    attrs->workitem_count[0],
                    attrs->workitem_count[1],
                    attrs->workitem_count[2],
                },
            .dynamic_workgroup_local_memory = attrs->shared_memory_bytes,
        };
        const iree_hal_dispatch_flags_t flags =
            attrs->bindings.count
                ? IREE_HAL_DISPATCH_FLAG_NONE
                : IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS;
        const iree_hal_dispatch_flags_t exact_flags =
            (attrs->workitem_count[0] || attrs->workitem_count[1] ||
             attrs->workitem_count[2])
                ? IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT
                : IREE_HAL_DISPATCH_FLAG_NONE;
        status = iree_hal_command_buffer_dispatch(
            command_buffer, symbol->executable,
            iree_hal_executable_function_from_index(symbol->export_ordinal),
            config, attrs->constants, attrs->bindings, flags | exact_flags);
        break;
      }
      case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY: {
        const iree_hal_streaming_graph_memcpy_node_attrs_t* attrs =
            iree_hal_streaming_graph_node_memcpy(node);
        status =
            iree_hal_streaming_graph_record_memcpy_node(command_buffer, attrs);
        release_to_host |= attrs->dst_ref.buffer->memory_kind !=
                           IREE_HAL_STREAMING_MEMORY_KIND_DEVICE;
        break;
      }
      case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET: {
        const iree_hal_streaming_graph_memset_node_attrs_t* attrs =
            iree_hal_streaming_graph_node_memset(node);
        iree_device_size_t fill_length = 0;
        if (IREE_UNLIKELY(!iree_device_size_checked_mul(
                attrs->pattern_size, attrs->count, &fill_length))) {
          status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                    "memset node size overflows device size");
          break;
        }
        status = iree_hal_command_buffer_fill_buffer(
            command_buffer,
            iree_hal_streaming_convert_range_buffer_ref(attrs->dst_ref,
                                                        fill_length),
            &attrs->pattern, attrs->pattern_size, attrs->flags);
        break;
      }
      case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_BATCH_MEM_OP: {
        const iree_hal_streaming_graph_batch_mem_op_node_attrs_t* attrs =
            iree_hal_streaming_graph_node_batch_mem_op(node);
        status = hrx_command_record_atomics(
            command_buffer, attrs->operation_count, attrs->operations,
            IREE_HAL_EXECUTION_STAGE_DISPATCH |
                IREE_HAL_EXECUTION_STAGE_TRANSFER |
                IREE_HAL_EXECUTION_STAGE_ATOMIC,
            IREE_HAL_EXECUTION_STAGE_ATOMIC);
        break;
      }
      default: {
        // Non-recordable nodes shouldn't be here.
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "non-recordable node type %d in recordable partition",
            (int)node->type);
        break;
      }
    }
  }

  if (iree_status_is_ok(status) && release_to_host) {
    // Publish all registered-host destinations before this partition signals
    // completion or releases a dependent CPU graph node.
    status = iree_hal_streaming_graph_record_dependency_barrier(
        command_buffer, IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE);
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
    iree_hal_streaming_graph_exec_t* exec, const hrx_graph_schedule_t* schedule,
    const hrx_graph_dependency_block_t* plan,
    iree_hal_streaming_graph_block_t** out_block) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_graph_block_t* block = NULL;
  const hrx_graph_partition_t* partition = plan->partition;
  if (!partition) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_graph_allocate_barrier_block(
                exec, plan->dependencies, IREE_HAL_QUEUE_BARRIER_FLAG_NONE,
                &block));
  } else {
    const uint32_t first_index = schedule->order[partition->start_index];
    const uint8_t kind =
        schedule->flags[first_index] & HRX_GRAPH_ANALYSIS_NODE_KIND_MASK;
    iree_hal_streaming_graph_node_t* first_node =
        schedule->sources[first_index];
    const iree_hal_streaming_graph_node_type_t type = first_node->type;
    if (kind == HRX_GRAPH_ANALYSIS_NODE_RECORDABLE) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_streaming_graph_allocate_execute_block(
                  exec, plan->dependencies, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE,
                  &block));
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_streaming_graph_record_partition(
                  exec, schedule, partition->start_index, partition->count,
                  plan->stream_index, block->attrs.execute.command_buffer));
    } else {
      if (type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_HOST_CALL) {
        // Host call gets its own block.
        iree_hal_streaming_graph_node_t* node = first_node;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_streaming_graph_allocate_host_call_block(
                    exec, plan->dependencies,
                    iree_hal_streaming_graph_node_host(node)->callback,
                    iree_hal_streaming_graph_node_host(node)->user_data, node,
                    IREE_HAL_HOST_CALL_FLAG_NONE, &block));
      } else if (type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL) {
        iree_hal_streaming_graph_node_t* node = first_node;
        const iree_hal_streaming_graph_kernel_node_attrs_t* attrs =
            iree_hal_streaming_graph_node_kernel(node);
        const iree_hal_dispatch_config_t config = {
            .workgroup_size =
                {
                    attrs->block_dim[0],
                    attrs->block_dim[1],
                    attrs->block_dim[2],
                },
            .workgroup_count =
                {
                    attrs->grid_dim[0],
                    attrs->grid_dim[1],
                    attrs->grid_dim[2],
                },
            .workitem_count =
                {
                    attrs->workitem_count[0],
                    attrs->workitem_count[1],
                    attrs->workitem_count[2],
                },
            .dynamic_workgroup_local_memory = attrs->shared_memory_bytes,
        };
        iree_hal_dispatch_flags_t flags =
            attrs->cooperative ? IREE_HAL_DISPATCH_FLAG_COOPERATIVE
                               : IREE_HAL_DISPATCH_FLAG_NONE;
        if (attrs->bindings.count == 0) {
          flags |= IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS;
        }
        if (attrs->workitem_count[0] || attrs->workitem_count[1] ||
            attrs->workitem_count[2]) {
          flags |= IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT;
        }
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_streaming_graph_allocate_dispatch_block(
                    exec, plan->dependencies, attrs->symbol->executable,
                    attrs->symbol->export_ordinal, config, attrs->constants,
                    attrs->bindings, flags, &block));
      } else if (type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH) {
        iree_hal_streaming_graph_node_t* node = first_node;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_streaming_graph_allocate_child_graph_block(
                    exec, plan->dependencies,
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
                  &exec->arena_allocator,
                  allocate ? IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_MEMORY_ALLOCATE
                           : IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_MEMORY_FREE,
                  plan->dependencies, &block));
          block->attrs.memory.allocation =
              allocate
                  ? iree_hal_streaming_graph_node_mem_alloc(node)->allocation
                  : iree_hal_streaming_graph_node_mem_free(node)->allocation;
          block->attrs.memory.flags =
              allocate ? IREE_HAL_STREAMING_GRAPH_ALLOCATION_FLAG_ALLOCATE
                       : IREE_HAL_STREAMING_GRAPH_ALLOCATION_FLAG_NONE;
          if (allocate &&
              iree_any_bit_set(
                  exec->flags,
                  IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)) {
            block->attrs.memory.flags |=
                IREE_HAL_STREAMING_GRAPH_ALLOCATION_FLAG_AUTO_FREE;
          }
          IREE_RETURN_AND_END_ZONE_IF_ERROR(
              z0, iree_hal_resource_set_insert(
                      exec->resource_set, 1, &block->attrs.memory.allocation));
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
                  exec, block_type, plan->dependencies, node,
                  iree_hal_streaming_graph_node_event(node)->event, &block));
        }
      }
    }
  }
  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_exec_compile_schedule(
    iree_hal_streaming_graph_exec_t* exec,
    const hrx_graph_schedule_t* schedule) {
  IREE_TRACE_ZONE_BEGIN(z0);
  hrx_graph_dependency_cursor_t cursor;
  hrx_graph_dependency_cursor_initialize(schedule, &cursor);
  if (!cursor.block_count) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&exec->arena_allocator,
                              cursor.block_count * sizeof(*exec->blocks),
                              (void**)&exec->blocks));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, hrx_graph_dependencies_allocate(
              exec->context->device, cursor.semaphore_count,
              &exec->arena_allocator, exec->resource_set, &exec->semaphores));
  exec->semaphore_count = cursor.semaphore_count;

  hrx_graph_dependency_block_t plan;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         hrx_graph_dependency_cursor_next(&cursor, &plan)) {
    iree_hal_streaming_graph_block_t* block = NULL;
    status =
        iree_hal_streaming_graph_compile_block(exec, schedule, &plan, &block);
    if (iree_status_is_ok(status)) {
      // Publish only initialized entries so partial compilation releases child
      // executable and event references alongside the shared resource set.
      exec->blocks[exec->block_count++] = block;
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_graph_exec_compile(
    iree_hal_streaming_graph_exec_t* exec) {
  exec->instantiated_node_count = exec->graph->node_count;
  iree_arena_allocator_t arena;
  iree_arena_initialize(&exec->context->device_entry->block_pool, &arena);
  iree_hal_streaming_graph_edge_t* copy_dependencies = NULL;
  iree_status_t status = iree_hal_streaming_graph_schedule_copy_dependencies(
      exec->graph, &arena, &copy_dependencies);
  hrx_graph_schedule_t schedule;
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_schedule_nodes(
        exec->graph->node_blocks, exec->graph->node_count,
        exec->node_disabled_states, exec->node_disabled_state_count,
        copy_dependencies, &arena, &schedule);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_exec_compile_schedule(exec, &schedule);
  }
  iree_arena_deinitialize(&arena);
  return status;
}

// Cells one chunk of the dropped-graph list holds. The first chunk lives in
// the launching frame, so this is also how many references a launch drops
// without reaching the heap.
#define IREE_HAL_STREAMING_DROPPED_GRAPH_CHUNK_CAPACITY 16

// Capture graph references dropped by the event records a launch enqueues.
// Releasing the last reference to a captured graph frees the allocations the
// graph owns, which synchronizes every context and relocks the stream the
// launch submits on; the launch collects the references here and releases them
// once it has dropped its locks.
//
// Storage is chunked and never moves, so a cell handed out stays valid for the
// life of the list.
typedef struct iree_hal_streaming_dropped_graph_chunk_t {
  // Next chunk, NULL at the tail.
  struct iree_hal_streaming_dropped_graph_chunk_t* next;
  // Number of cells in |graphs| that are part of the list.
  iree_host_size_t count;
  // Owned references, NULL for a cell whose record never committed.
  iree_hal_streaming_graph_t*
      graphs[IREE_HAL_STREAMING_DROPPED_GRAPH_CHUNK_CAPACITY];
} iree_hal_streaming_dropped_graph_chunk_t;

typedef struct iree_hal_streaming_dropped_graph_list_t {
  // Allocator the heap chunks come from.
  iree_allocator_t host_allocator;
  // Chunk cells are added to, never NULL; |first| until the list grows.
  iree_hal_streaming_dropped_graph_chunk_t* tail;
  // Storage covering launches dropping no more references than it holds,
  // living in the launching frame.
  iree_hal_streaming_dropped_graph_chunk_t first;
} iree_hal_streaming_dropped_graph_list_t;

// Prepares |out_list| to collect references, growing from |host_allocator|.
static void iree_hal_streaming_dropped_graph_list_initialize(
    iree_allocator_t host_allocator,
    iree_hal_streaming_dropped_graph_list_t* out_list) {
  out_list->host_allocator = host_allocator;
  out_list->tail = &out_list->first;
  out_list->first.next = NULL;
  out_list->first.count = 0;
}

// Adds an empty cell to |list| and returns it. The cell counts as part of the
// list from here: deinitializing releases whatever it holds, and a cell left
// NULL releases nothing. Growing is the only step that can fail and it runs
// before the record that displaces a reference is enqueued, so a failure drops
// nothing. The cell stays valid for the life of the list.
IREE_MUST_USE_RESULT static iree_status_t
iree_hal_streaming_dropped_graph_list_push_empty(
    iree_hal_streaming_dropped_graph_list_t* list,
    iree_hal_streaming_graph_t*** out_cell) {
  *out_cell = NULL;
  iree_hal_streaming_dropped_graph_chunk_t* tail = list->tail;
  if (tail->count == IREE_ARRAYSIZE(tail->graphs)) {
    iree_hal_streaming_dropped_graph_chunk_t* chunk = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(list->host_allocator,
                                               sizeof(*chunk), (void**)&chunk));
    tail->next = chunk;
    list->tail = chunk;
    tail = chunk;
  }
  iree_hal_streaming_graph_t** cell = &tail->graphs[tail->count++];
  *cell = NULL;
  *out_cell = cell;
  return iree_ok_status();
}

// Releases every reference |list| collected and frees the chunks it grew into.
// The list is dead afterwards: the launch that initialized it is the only
// owner and it deinitializes once, on its way out.
static void iree_hal_streaming_dropped_graph_list_deinitialize(
    iree_hal_streaming_dropped_graph_list_t* list) {
  iree_hal_streaming_dropped_graph_chunk_t* chunk = &list->first;
  while (chunk) {
    for (iree_host_size_t i = 0; i < chunk->count; ++i) {
      iree_hal_streaming_graph_release(chunk->graphs[i]);
    }
    iree_hal_streaming_dropped_graph_chunk_t* next = chunk->next;
    if (chunk != &list->first) {
      iree_allocator_free(list->host_allocator, chunk);
    }
    chunk = next;
  }
}

// Signals from actually accepted graph blocks. Failure recovery waits on this
// prefix because rejected blocks cannot signal the reserved final stream value.
#define IREE_HAL_STREAMING_ACCEPTED_SIGNAL_INLINE_CAPACITY 16

typedef struct iree_hal_streaming_accepted_signal_t {
  // Borrowed semaphore kept alive by the locked stream or executable tree.
  iree_hal_semaphore_t* semaphore;
  // Payload the accepted block will signal or permanently fail.
  uint64_t value;
} iree_hal_streaming_accepted_signal_t;

typedef struct iree_hal_streaming_accepted_signal_list_t {
  // Allocator used only if |inline_signals| fills.
  iree_allocator_t host_allocator;
  // Number of reserved cells; empty cells have a NULL semaphore.
  iree_host_size_t count;
  // Number of cells available through |signals|.
  iree_host_size_t capacity;
  // Active cell storage.
  iree_hal_streaming_accepted_signal_t* signals;
  // Common-case storage owned by the launching frame.
  iree_hal_streaming_accepted_signal_t
      inline_signals[IREE_HAL_STREAMING_ACCEPTED_SIGNAL_INLINE_CAPACITY];
} iree_hal_streaming_accepted_signal_list_t;

static void iree_hal_streaming_accepted_signal_list_initialize(
    iree_allocator_t host_allocator,
    iree_hal_streaming_accepted_signal_list_t* out_list) {
  out_list->host_allocator = host_allocator;
  out_list->count = 0;
  out_list->capacity = IREE_ARRAYSIZE(out_list->inline_signals);
  out_list->signals = out_list->inline_signals;
}

// Reserves a cell before a fallible queue call. A failed queue call leaves the
// cell empty, so only actual acceptance contributes a drain obligation.
static iree_status_t iree_hal_streaming_accepted_signal_list_push_empty(
    iree_hal_streaming_accepted_signal_list_t* list,
    iree_hal_streaming_accepted_signal_t** out_signal) {
  *out_signal = NULL;
  if (list->count == list->capacity) {
    iree_host_size_t new_capacity = 0;
    iree_host_size_t allocation_size = 0;
    if (IREE_UNLIKELY(
            !iree_host_size_checked_mul(list->capacity, 2, &new_capacity) ||
            !iree_host_size_checked_mul(new_capacity, sizeof(list->signals[0]),
                                        &allocation_size))) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "accepted graph signal list size overflow");
    }
    iree_hal_streaming_accepted_signal_t* new_signals = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        list->host_allocator, allocation_size, (void**)&new_signals));
    memcpy(new_signals, list->signals, list->count * sizeof(list->signals[0]));
    if (list->signals != list->inline_signals) {
      iree_allocator_free(list->host_allocator, list->signals);
    }
    list->signals = new_signals;
    list->capacity = new_capacity;
  }

  iree_hal_streaming_accepted_signal_t* signal = &list->signals[list->count++];
  *signal = (iree_hal_streaming_accepted_signal_t){0};
  *out_signal = signal;
  return iree_ok_status();
}

static void iree_hal_streaming_accepted_signal_list_deinitialize(
    iree_hal_streaming_accepted_signal_list_t* list) {
  if (list->signals != list->inline_signals) {
    iree_allocator_free(list->host_allocator, list->signals);
  }
}

// Resolves accepted-block outcomes after a later block is rejected. A failed
// semaphore does not prove retirement: submitted command buffers and callback
// captures retain their own state, and the caller preserves any failed queue
// lease until queue teardown.
static iree_status_t iree_hal_streaming_accepted_signal_list_drain(
    iree_hal_streaming_accepted_signal_list_t* list) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    if (!list->signals[i].semaphore) {
      continue;
    }
    iree_status_t wait_status = iree_hal_semaphore_wait(
        list->signals[i].semaphore, list->signals[i].value,
        iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
    status = iree_status_join(status, wait_status);
  }
  return status;
}

static iree_status_t iree_hal_streaming_graph_exec_submit_blocks_locked(
    iree_hal_streaming_graph_exec_t* exec, iree_hal_streaming_stream_t* stream,
    iree_hal_queue_t* queue, iree_hal_queue_t* cooperative_queue,
    uint64_t launch_stream_tail_value,
    iree_hal_semaphore_list_t external_wait_semaphores,
    iree_hal_semaphore_list_t external_signal_semaphores,
    iree_hal_streaming_dropped_graph_list_t* dropped_graphs,
    iree_hal_streaming_accepted_signal_list_t* accepted_signals);

// Checks every cooperative dispatch in |exec| and its child executable tree
// against the queue selected for |stream|. The caller holds the root
// executable mutex and the stream mutex, which keep the compiled block tree
// and borrowed queue stable throughout the recursive walk.
static iree_status_t
iree_hal_streaming_graph_exec_preflight_cooperative_dispatches_locked(
    iree_hal_streaming_graph_exec_t* exec, iree_hal_queue_t* cooperative_queue,
    bool* out_exceeds_residency) {
  for (uint32_t block_index = 0; block_index < exec->block_count;
       ++block_index) {
    iree_hal_streaming_graph_block_t* block = exec->blocks[block_index];
    if (block->type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH) {
      if (!block->attrs.child_graph.exec->has_cooperative_dispatches) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          iree_hal_streaming_graph_exec_preflight_cooperative_dispatches_locked(
              block->attrs.child_graph.exec, cooperative_queue,
              out_exceeds_residency));
      if (*out_exceeds_residency) {
        return iree_ok_status();
      }
      continue;
    }
    if (block->type != IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH ||
        !iree_any_bit_set(block->attrs.dispatch.flags,
                          IREE_HAL_DISPATCH_FLAG_COOPERATIVE)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        iree_hal_streaming_check_cooperative_dispatch_residency(
            cooperative_queue, block->attrs.dispatch.executable,
            iree_hal_executable_function_from_index(
                (uint32_t)block->attrs.dispatch.entry_point),
            block->attrs.dispatch.config, out_exceeds_residency));
    if (*out_exceeds_residency) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

// Submits |block| on |stream| behind |wait_semaphores|, signaling
// |signal_semaphores| when it completes. |launch_stream_tail_value| and
// |dropped_graphs| are meaningful only to a child graph block, which carries
// them down into the child's own walk, and |record_point| only to an event
// record block, which receives it describing the point the record signals and
// leaves it owning what it names.
static iree_status_t iree_hal_streaming_graph_submit_block(
    iree_hal_streaming_graph_block_t* block,
    iree_hal_streaming_stream_t* stream, iree_hal_queue_t* queue,
    iree_hal_queue_t* cooperative_queue, uint64_t launch_stream_tail_value,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores,
    iree_hal_streaming_recorded_point_t* record_point,
    iree_hal_streaming_dropped_graph_list_t* dropped_graphs,
    iree_hal_streaming_accepted_signal_list_t* accepted_signals) {
  if (block->type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH) {
    iree_hal_streaming_graph_exec_t* child_exec = block->attrs.child_graph.exec;
    // Every child entry block carries this block's waits, so the launch tail
    // carries down unchanged. The shared accepted list includes every actual
    // child submission if a later child block rejects.
    return iree_hal_streaming_graph_exec_submit_blocks_locked(
        child_exec, stream, queue, cooperative_queue, launch_stream_tail_value,
        wait_semaphores, signal_semaphores, dropped_graphs, accepted_signals);
  }
  // Compilation gives every block a completion, joining parallel exits before
  // publishing the external signal. Failure cleanup relies on that proof.
  IREE_ASSERT(signal_semaphores.count > 0);

  // Claim storage before the queue call so an accepted operation always has a
  // terminal signal the launch can drain if a later block rejects.
  iree_hal_streaming_accepted_signal_t* accepted_signal = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_accepted_signal_list_push_empty(
      accepted_signals, &accepted_signal));

  iree_status_t status = iree_ok_status();
  switch (block->type) {
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_MEMORY_ALLOCATE:
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_MEMORY_FREE:
      status = iree_hal_streaming_graph_allocation_enqueue(
          block->attrs.memory.allocation, queue, wait_semaphores,
          signal_semaphores, block->attrs.memory.flags,
          &accepted_signal->semaphore, &accepted_signal->value);
      break;
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD:
      status = iree_hal_streaming_event_enqueue_record(
          block->attrs.event.event, stream->context, queue, wait_semaphores,
          signal_semaphores, record_point);
      break;
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_WAIT:
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER: {
      const iree_hal_queue_barrier_flags_t flags =
          block->type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER
              ? block->attrs.barrier.flags
              : IREE_HAL_QUEUE_BARRIER_FLAG_NONE;
      status = iree_hal_queue_barrier(queue, wait_semaphores, signal_semaphores,
                                      /*barriers=*/NULL, flags);
      break;
    }
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH: {
      iree_hal_queue_t* dispatch_queue = queue;
      if (iree_any_bit_set(block->attrs.dispatch.flags,
                           IREE_HAL_DISPATCH_FLAG_COOPERATIVE)) {
        IREE_ASSERT(cooperative_queue);
        dispatch_queue = cooperative_queue;
      }
      iree_hal_buffer_ref_list_t bindings_list = {
          .count = block->attrs.dispatch.bindings.count,
          .values = block->attrs.dispatch.bindings.values,
      };
      status = iree_hal_queue_dispatch(
          dispatch_queue, wait_semaphores, signal_semaphores,
          block->attrs.dispatch.executable,
          iree_hal_executable_function_from_index(
              (uint32_t)block->attrs.dispatch.entry_point),
          block->attrs.dispatch.config, block->attrs.dispatch.constants,
          bindings_list, /*barriers=*/NULL, block->attrs.dispatch.flags);
      break;
    }
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE:
      status = iree_hal_queue_execute(queue, wait_semaphores, signal_semaphores,
                                      block->attrs.execute.command_buffer,
                                      iree_hal_buffer_binding_table_empty(),
                                      IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
      break;
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL: {
      iree_hal_host_call_t call;
      status = iree_hal_streaming_host_callback_prepare(
          stream->context->device_entry->host_callbacks,
          block->attrs.host_call.callback, block->attrs.host_call.user_data,
          block->attrs.host_call.user_data_size,
          block->attrs.host_call.resource, &call);
      if (iree_status_is_ok(status)) {
        const uint64_t args[4] = {0};
        status =
            iree_hal_queue_host_call(queue, wait_semaphores, signal_semaphores,
                                     call, args, block->attrs.host_call.flags);
        iree_hal_resource_release(call.resource);
      }
      break;
    }
    default:
      status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unsupported block type %u", block->type);
      break;
  }

  if (iree_status_is_ok(status)) {
    accepted_signal->semaphore = signal_semaphores.semaphores[0];
    accepted_signal->value = signal_semaphores.payload_values[0];
  }
  return status;
}

// |launch_stream_tail_value| is the value on |stream|'s timeline that the whole
// launch waits behind, or 0 when the stream had never submitted. Every block
// in the first partition carries the external waits. Subsequent partitions
// join every workstream in their predecessor, including the final completion.
static iree_status_t iree_hal_streaming_graph_exec_submit_blocks_locked(
    iree_hal_streaming_graph_exec_t* exec, iree_hal_streaming_stream_t* stream,
    iree_hal_queue_t* queue, iree_hal_queue_t* cooperative_queue,
    uint64_t launch_stream_tail_value,
    iree_hal_semaphore_list_t external_wait_semaphores,
    iree_hal_semaphore_list_t external_signal_semaphores,
    iree_hal_streaming_dropped_graph_list_t* dropped_graphs,
    iree_hal_streaming_accepted_signal_list_t* accepted_signals) {
  if (exec->block_count == 0) {
    iree_hal_streaming_accepted_signal_t* accepted_signal = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_accepted_signal_list_push_empty(
        accepted_signals, &accepted_signal));
    iree_status_t status = iree_hal_queue_barrier(
        queue, external_wait_semaphores, external_signal_semaphores,
        /*barriers=*/NULL, IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
    if (iree_status_is_ok(status)) {
      accepted_signal->semaphore = external_signal_semaphores.semaphores[0];
      accepted_signal->value = external_signal_semaphores.payload_values[0];
    }
    return status;
  }

  // Every replay follows the prior launch's completion. Reserve values instead
  // of querying asynchronous progress: a later signal must never satisfy an
  // earlier invocation's wait. Rejected prefixes are drained by the caller;
  // skipping their generation leaves all subsequent edge values unambiguous.
  const uint64_t generation = ++exec->launch_generation;
  uint64_t values[HRX_GRAPH_MAX_WORKSTREAMS];
  if (exec->semaphore_count) {
    for (uint32_t i = 0; i < IREE_ARRAYSIZE(values); ++i) {
      values[i] = generation;
    }
  }
  iree_status_t status = iree_ok_status();
  for (uint32_t block_index = 0;
       iree_status_is_ok(status) && block_index < exec->block_count;
       ++block_index) {
    iree_hal_streaming_graph_block_t* block = exec->blocks[block_index];
    iree_hal_semaphore_list_t wait_semaphores;
    iree_hal_semaphore_list_t signal_semaphores;
    hrx_graph_dependencies_resolve(
        block->dependencies, block_index, exec->semaphore_count,
        exec->semaphores, values, external_wait_semaphores,
        external_signal_semaphores, &wait_semaphores, &signal_semaphores);

    // Event nodes add one dynamic point to a bounded partition join. Child
    // entry waits inherit their parent's span (at most MAX_WORKSTREAMS), or
    // the root's two external stream dependencies. Other blocks borrow their
    // semaphore arrays directly from the compiled recording or caller.
    const bool block_waits_event =
        block->type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_WAIT;
    const bool block_records_event =
        block->type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD;
    iree_hal_semaphore_t* event_wait_semaphores[HRX_GRAPH_MAX_WORKSTREAMS + 1];
    uint64_t event_wait_values[IREE_ARRAYSIZE(event_wait_semaphores)];
    iree_hal_streaming_recorded_point_t event_wait_point = {0};
    if (block_waits_event) {
      // Keep the named timeline alive through the queue call even if another
      // thread records this event concurrently. The point does not enter the
      // stream's memory-reuse ledger; this only withholds otherwise safe reuse.
      iree_hal_streaming_event_acquire_recorded_point(block->attrs.event.event,
                                                      &event_wait_point);
      if (event_wait_point.semaphore) {
        for (iree_host_size_t i = 0; i < wait_semaphores.count; ++i) {
          event_wait_semaphores[i] = wait_semaphores.semaphores[i];
          event_wait_values[i] = wait_semaphores.payload_values[i];
        }
        event_wait_semaphores[wait_semaphores.count] =
            event_wait_point.semaphore;
        event_wait_values[wait_semaphores.count] = event_wait_point.value;
        wait_semaphores = (iree_hal_semaphore_list_t){
            .count = wait_semaphores.count + 1,
            .semaphores = event_wait_semaphores,
            .payload_values = event_wait_values,
        };
      }
    }

    iree_hal_streaming_recorded_point_t record_point = {0};
    iree_hal_streaming_graph_t** dropped_capture_graph = NULL;
    if (block_records_event) {
      // An isolated record node names its own completion. Only a terminal
      // record signals the launch stream; internal points retain the stream
      // tail known at entry without claiming the whole graph has completed.
      const bool signals_launch_stream_timeline =
          signal_semaphores.semaphores[0] == stream->timeline_semaphore;
      record_point = (iree_hal_streaming_recorded_point_t){
          .semaphore = signal_semaphores.semaphores[0],
          .value = signal_semaphores.payload_values[0],
          .ordered_after_stream_id = stream->stream_id,
          .ordered_after_stream_value =
              signals_launch_stream_timeline
                  ? signal_semaphores.payload_values[0]
                  : launch_stream_tail_value,
      };
      // Reserve cleanup storage before acceptance; release capture references
      // outside the launch locks only after the record was accepted.
      status = iree_hal_streaming_dropped_graph_list_push_empty(
          dropped_graphs, &dropped_capture_graph);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_streaming_graph_submit_block(
          block, stream, queue, cooperative_queue, launch_stream_tail_value,
          wait_semaphores, signal_semaphores, &record_point, dropped_graphs,
          accepted_signals);
      if (iree_status_is_ok(status) && block_records_event) {
        // A rejected record keeps its old point. Later blocks in this launch
        // must observe the new point immediately after its acceptance.
        *dropped_capture_graph = iree_hal_streaming_event_commit_recorded_point(
            block->attrs.event.event, record_point);
      }
    }
    iree_hal_streaming_event_release_recorded_point(&event_wait_point);
  }
  return status;
}

iree_status_t iree_hal_streaming_graph_exec_launch(
    iree_hal_streaming_graph_exec_t* exec, iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_graph_exec_launch_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(exec);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(out_result);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_ERROR;

  // Mutex needed for launch as multiple threads can submit at once.
  // It also protects the reserved generation of this executable and its
  // privately owned children; no child can be launched independently.
  iree_slim_mutex_lock(&exec->mutex);
  if (exec->is_destroyed) {
    iree_slim_mutex_unlock(&exec->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  // A graph may begin with a host callback. Publish preceding native writes
  // under the same stream lock that captures the graph's incoming dependency.
  iree_slim_mutex_lock(&stream->mutex);
  iree_status_t status = iree_hal_streaming_stream_flush_locked(stream);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&stream->mutex);
    iree_slim_mutex_unlock(&exec->mutex);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  // An empty recording needs no submission unless it must carry a previous
  // launch across streams after a parameter update removed all executable work.
  if (exec->block_count == 0 &&
      (!exec->last_launch.semaphore ||
       exec->last_launch.semaphore == stream->timeline_semaphore)) {
    *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_SUCCESS;
    iree_slim_mutex_unlock(&stream->mutex);
    iree_slim_mutex_unlock(&exec->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  // Every record this executable holds names an event of |exec->context|, so on
  // a stream of any other context iree_hal_streaming_event_enqueue_record would
  // refuse each of them in turn. Deciding the same question here refuses the
  // launch before it submits any of the graph, where the walk below breaks on
  // the first refusal and leaves the blocks ahead of the record in flight with
  // nothing signaling the launching stream's timeline.
  if (exec->records_events && stream->context != exec->context) {
    iree_slim_mutex_unlock(&stream->mutex);
    iree_slim_mutex_unlock(&exec->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INCOMPATIBLE,
        "an event can only be recorded on a stream of the context that "
        "created it");
  }
  // Event records this launch enqueues end their events' capture associations.
  // The references they drop are collected here and released once the locks
  // below have been dropped.
  iree_hal_streaming_dropped_graph_list_t dropped_graphs;
  iree_hal_streaming_dropped_graph_list_initialize(exec->host_allocator,
                                                   &dropped_graphs);
  iree_hal_streaming_accepted_signal_list_t accepted_signals;
  iree_hal_streaming_accepted_signal_list_initialize(exec->host_allocator,
                                                     &accepted_signals);

  iree_hal_queue_t* cooperative_queue = NULL;
  bool cooperative_launch_too_large = false;
  if (exec->has_cooperative_dispatches) {
    status = iree_hal_streaming_stream_select_cooperative_queue_locked(
        stream, &cooperative_queue);
  }
  if (iree_status_is_ok(status) && exec->has_cooperative_dispatches) {
    status =
        iree_hal_streaming_graph_exec_preflight_cooperative_dispatches_locked(
            exec, cooperative_queue, &cooperative_launch_too_large);
  }
  if (iree_status_is_ok(status) && cooperative_launch_too_large) {
    *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_COOPERATIVE_TOO_LARGE;
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "cooperative graph dispatch exceeds concurrent residency");
  }
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&stream->mutex);
    iree_slim_mutex_unlock(&exec->mutex);
    iree_hal_streaming_dropped_graph_list_deinitialize(&dropped_graphs);
    iree_hal_streaming_accepted_signal_list_deinitialize(&accepted_signals);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Reserve the next stream timeline value while holding the stream lock so
  // concurrent host threads cannot submit same-stream work with the same wait
  // or signal value. The graph waits on the current stream tail, not the last
  // completion observed by the host; HIP stream ordering requires repeated
  // graph launches on the same stream to execute FIFO even when the caller does
  // not synchronize between launches.
  uint64_t stream_wait_value = 0;
  uint64_t stream_signal_value = 0;
  status = iree_hal_streaming_stream_reserve_next_value_locked(
      stream, &stream_wait_value, &stream_signal_value);

  iree_hal_semaphore_t* wait_semaphores_storage[2];
  uint64_t wait_values_storage[2];
  iree_hal_semaphore_list_t wait_semaphores = {
      .count = 0,
      .semaphores = wait_semaphores_storage,
      .payload_values = wait_values_storage,
  };
  if (stream_wait_value > 0) {
    wait_semaphores_storage[wait_semaphores.count] = stream->timeline_semaphore;
    wait_values_storage[wait_semaphores.count++] = stream_wait_value;
  }
  // Replays of one executable are ordered even when issued on different
  // streams. The same-stream tail already contains this edge. Cross-stream
  // launches carry it directly into the graph instead of adding a host wait
  // or a separate queue operation.
  const bool waits_for_previous_launch =
      exec->last_launch.semaphore &&
      exec->last_launch.semaphore != stream->timeline_semaphore;
  if (waits_for_previous_launch) {
    wait_semaphores_storage[wait_semaphores.count] =
        exec->last_launch.semaphore;
    wait_values_storage[wait_semaphores.count++] = exec->last_launch.value;
  }
  const bool requires_independent_queue =
      exec->waits_for_events || (waits_for_previous_launch &&
                                 exec->last_launch.requires_independent_queue);
  iree_hal_streaming_submission_t submission = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_submission_prepare_locked(
        stream, stream->queue, stream_signal_value,
        requires_independent_queue
            ? IREE_HAL_STREAMING_SUBMISSION_FLAG_INDEPENDENT
            : IREE_HAL_STREAMING_SUBMISSION_FLAG_NONE,
        &submission);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_exec_submit_blocks_locked(
        exec, stream, submission.queue, cooperative_queue, stream_wait_value,
        wait_semaphores, submission.signals, &dropped_graphs,
        &accepted_signals);
  }

  if (!iree_status_is_ok(status)) {
    // No rejected block can signal |stream_signal_value|, so do not publish a
    // phantom stream tail. Keep both locks held while resolving every actual
    // accepted signal, preventing teardown or later same-stream work from
    // overtaking the prefix. Join keeps the submission failure primary.
    iree_status_t drain_status =
        iree_hal_streaming_accepted_signal_list_drain(&accepted_signals);
    if (!iree_status_is_ok(drain_status) && submission.completion) {
      // A failed prefix signal cannot prove hardware retirement. The final
      // graph signal was not accepted, so rejecting its prepared observer is
      // insufficient: retain the lane for authoritative queue teardown.
      iree_slim_mutex_lock(&stream->context->value_wait_lane_mutex);
      submission.completion->lane->has_failed_submission = true;
      iree_slim_mutex_unlock(&stream->context->value_wait_lane_mutex);
    }
    status = iree_status_join(status, drain_status);
  }

  iree_hal_streaming_submission_finish_locked(stream, &submission,
                                              iree_status_is_ok(status));
  if (iree_status_is_ok(status)) {
    *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_SUCCESS;
    if (exec->last_launch.semaphore != stream->timeline_semaphore) {
      iree_hal_semaphore_retain(stream->timeline_semaphore);
      iree_hal_semaphore_release(exec->last_launch.semaphore);
      exec->last_launch.semaphore = stream->timeline_semaphore;
    }
    exec->last_launch.value = stream_signal_value;
    exec->last_launch.requires_independent_queue =
        submission.queue != stream->context->queue;
  }

  iree_slim_mutex_unlock(&stream->mutex);
  iree_slim_mutex_unlock(&exec->mutex);
  // Release captured graph references outside admission locks. The records
  // ended these associations even if a later block was rejected.
  iree_hal_streaming_dropped_graph_list_deinitialize(&dropped_graphs);
  iree_hal_streaming_accepted_signal_list_deinitialize(&accepted_signals);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static bool iree_hal_streaming_graph_node_is_visible(
    const iree_hal_streaming_graph_node_t* node) {
  return node && (node->flags & IREE_HAL_STREAMING_GRAPH_NODE_FLAG_HIDDEN) == 0;
}

iree_status_t iree_hal_streaming_graph_exec_set_child_graph(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node,
    iree_hal_streaming_graph_t* child_graph) {
  node = iree_hal_streaming_graph_exec_lock_node(exec, node);
  if (!node || node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH) {
    iree_hal_streaming_graph_exec_unlock(exec);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  iree_hal_streaming_graph_t* old_child_graph =
      iree_hal_streaming_graph_node_child_graph(node)->graph;
  iree_hal_streaming_graph_t* replacement = NULL;
  iree_status_t status = iree_hal_streaming_graph_validate_child_graph(
      exec->source_graph, child_graph);
  iree_hal_streaming_graph_node_t* error_node = NULL;
  iree_hal_streaming_graph_exec_update_result_t update_result;
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_check_update(old_child_graph, child_graph,
                                                   &exec->arena_allocator,
                                                   &error_node, &update_result);
  }
  if (iree_status_is_ok(status) &&
      update_result != IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_SUCCESS) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "child graph update is not compatible");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_clone(
        child_graph, IREE_HAL_STREAMING_GRAPH_CLONE_MODE_EXECUTABLE,
        &replacement);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_streaming_graph_node_child_graph(node)->graph = replacement;
    status = iree_hal_streaming_graph_exec_rebuild_locked(exec);
    if (!iree_status_is_ok(status)) {
      iree_hal_streaming_graph_node_child_graph(node)->graph = old_child_graph;
    }
  }
  iree_hal_streaming_graph_exec_unlock(exec);
  iree_hal_streaming_graph_release(iree_status_is_ok(status) ? old_child_graph
                                                             : replacement);
  return status;
}

iree_status_t iree_hal_streaming_graph_exec_update(
    iree_hal_streaming_graph_exec_t* exec, iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** out_error_node,
    iree_hal_streaming_graph_exec_update_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(exec);
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(out_error_node);
  IREE_ASSERT_ARGUMENT(out_result);
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_error_node = NULL;
  *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_ERROR;

  iree_slim_mutex_lock(&exec->mutex);
  if (exec->is_destroyed) {
    iree_slim_mutex_unlock(&exec->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  iree_status_t status = iree_hal_streaming_graph_check_update(
      exec->graph, graph, &exec->arena_allocator, out_error_node, out_result);
  if (iree_status_is_ok(status) &&
      *out_result != IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_SUCCESS) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "graph update is not compatible");
  }
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&exec->mutex);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_ERROR;
  iree_hal_streaming_graph_t* replacement = NULL;
  uint8_t* replacement_disabled_states = NULL;
  status = iree_hal_streaming_graph_clone(
      graph, IREE_HAL_STREAMING_GRAPH_CLONE_MODE_EXECUTABLE, &replacement);
  if (iree_status_is_ok(status) && replacement->node_count > 0) {
    status =
        iree_allocator_malloc(exec->host_allocator, replacement->node_count,
                              (void**)&replacement_disabled_states);
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t mapping_index = 0;
    for (iree_hal_streaming_node_block_t* block = replacement->node_blocks;
         block; block = block->next) {
      for (iree_host_size_t i = 0; i < block->count; ++i) {
        iree_hal_streaming_graph_node_t* node = block->nodes[i];
        if (!iree_hal_streaming_graph_node_is_visible(node)) {
          continue;
        }
        while (!iree_hal_streaming_graph_node_is_visible(
            exec->node_mappings[mapping_index].target)) {
          ++mapping_index;
        }
        replacement_disabled_states[node->node_index] =
            exec->node_disabled_states[exec->node_mappings[mapping_index++]
                                           .target->node_index];
      }
    }
  }

  iree_hal_streaming_graph_t* old_graph = exec->graph;
  uint8_t* old_disabled_states = exec->node_disabled_states;
  const iree_host_size_t old_disabled_state_count =
      exec->node_disabled_state_count;
  if (iree_status_is_ok(status)) {
    exec->graph = replacement;
    exec->node_disabled_states = replacement_disabled_states;
    exec->node_disabled_state_count = replacement->node_count;
    status = iree_hal_streaming_graph_exec_rebuild_locked(exec);
    if (!iree_status_is_ok(status)) {
      exec->graph = old_graph;
      exec->node_disabled_states = old_disabled_states;
      exec->node_disabled_state_count = old_disabled_state_count;
    }
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t mapping_index = 0;
    for (iree_hal_streaming_node_block_t* block = replacement->node_blocks;
         block; block = block->next) {
      for (iree_host_size_t i = 0; i < block->count; ++i) {
        iree_hal_streaming_graph_node_t* node = block->nodes[i];
        if (!iree_hal_streaming_graph_node_is_visible(node)) {
          continue;
        }
        while (!iree_hal_streaming_graph_node_is_visible(
            exec->node_mappings[mapping_index].target)) {
          exec->node_mappings[mapping_index++].target = NULL;
        }
        exec->node_mappings[mapping_index++].target = node;
      }
    }
    while (mapping_index < exec->node_mapping_count) {
      exec->node_mappings[mapping_index++].target = NULL;
    }
    *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_SUCCESS;
  }
  iree_slim_mutex_unlock(&exec->mutex);
  // Release old or rejected descriptions outside the executable lock.
  if (iree_status_is_ok(status)) {
    iree_allocator_free(exec->host_allocator, old_disabled_states);
    iree_hal_streaming_graph_release(old_graph);
  } else {
    iree_allocator_free(exec->host_allocator, replacement_disabled_states);
    iree_hal_streaming_graph_release(replacement);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
