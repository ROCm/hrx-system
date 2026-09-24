// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/graph.h"
#include "common/graph_memory.h"
#include "common/internal.h"
#include "common/stream.h"
#include "iree/base/api.h"
#include "iree/hal/utils/resource_set.h"

//===----------------------------------------------------------------------===//
// iree_hal_streaming_graph_exec_t (instantiation)
//===----------------------------------------------------------------------===//

typedef enum iree_hal_streaming_graph_block_type_e {
  // iree_hal_queue_barrier
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER = 0,
  // iree_hal_queue_fill
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_FILL,
  // iree_hal_queue_copy
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_COPY,
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
  // Stored batch memory operation awaiting graph queue support.
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_BATCH_MEM_OP,
  // Nested graph executable.
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH,
} iree_hal_streaming_graph_block_type_t;

typedef void (*iree_hal_streaming_host_callback_t)(void* user_data);

static iree_status_t iree_hal_streaming_graph_host_callback(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context);
// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER
typedef struct iree_hal_streaming_graph_barrier_block_attrs_t {
  iree_hal_queue_barrier_flags_t flags;
} iree_hal_streaming_graph_barrier_block_attrs_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_FILL
typedef struct iree_hal_streaming_graph_fill_block_attrs_t {
  iree_hal_buffer_t* target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
  uint64_t pattern;
  iree_host_size_t pattern_length;
  iree_hal_fill_flags_t flags;
} iree_hal_streaming_graph_fill_block_attrs_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_COPY
typedef struct iree_hal_streaming_graph_copy_block_attrs_t {
  iree_hal_buffer_t* source_buffer;
  iree_device_size_t source_offset;
  iree_hal_buffer_t* target_buffer;
  iree_device_size_t target_offset;
  iree_device_size_t length;
  iree_hal_copy_flags_t flags;
} iree_hal_streaming_graph_copy_block_attrs_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL
typedef struct iree_hal_streaming_graph_host_call_block_attrs_t {
  // Host call invoked when the queued operation reaches this block.
  iree_hal_host_call_t call;
  // Persistent queue-host-call argument storage referenced by the device queue.
  uint64_t args[4];
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
  iree_hal_executable_t* executable;
  iree_host_size_t entry_point;
  iree_hal_dispatch_config_t config;
  iree_const_byte_span_t constants;
  iree_hal_buffer_ref_list_t bindings;
  iree_hal_dispatch_flags_t flags;
} iree_hal_streaming_graph_dispatch_block_attrs_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE
typedef struct iree_hal_streaming_graph_execute_block_attrs_t {
  iree_hal_command_buffer_t* command_buffer;
  iree_hal_queue_execute_flags_t flags;
} iree_hal_streaming_graph_execute_block_attrs_t;

// Last launch point recorded for one stream using a graph executable. The
// executable keeps one record per distinct stream so destroying its public
// handle can defer resource release until every accepted launch is terminal.
typedef struct iree_hal_streaming_graph_launch_point_t {
  // Next distinct stream that has accepted a launch from this executable.
  struct iree_hal_streaming_graph_launch_point_t* next;
  // Stream owning |value|, retained until executable-handle destruction.
  iree_hal_streaming_stream_t* stream;
  // Latest timeline value accepted on |stream| for this executable.
  uint64_t value;
} iree_hal_streaming_graph_launch_point_t;

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH
typedef struct iree_hal_streaming_graph_child_graph_block_attrs_t {
  iree_hal_streaming_graph_exec_t* exec;
} iree_hal_streaming_graph_child_graph_block_attrs_t;

// Block-specific data stored inline in the block header.
typedef union iree_hal_streaming_graph_block_attrs_t {
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER
  iree_hal_streaming_graph_barrier_block_attrs_t barrier;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_FILL
  iree_hal_streaming_graph_fill_block_attrs_t fill;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_COPY
  iree_hal_streaming_graph_copy_block_attrs_t copy;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL
  iree_hal_streaming_graph_host_call_block_attrs_t host_call;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD/WAIT
  iree_hal_streaming_graph_event_block_attrs_t event;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH
  iree_hal_streaming_graph_dispatch_block_attrs_t dispatch;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE
  iree_hal_streaming_graph_execute_block_attrs_t execute;
  // IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH
  iree_hal_streaming_graph_child_graph_block_attrs_t child_graph;
} iree_hal_streaming_graph_block_attrs_t;

// Represents an atomically executable block of work in a graph.
typedef struct iree_hal_streaming_graph_block_t {
  iree_hal_streaming_graph_block_type_t type;

  // First node in sorted array.
  uint32_t node_start_index;
  // Number of nodes in this block.
  uint32_t node_count;

  // Semaphore synchronization.
  uint16_t wait_semaphore_count;
  uint16_t signal_semaphore_count;

  // Type-specific attributes, stored inline so the compiler supplies the
  // pointer alignment the union requires and the block header ends on that
  // same boundary.
  iree_hal_streaming_graph_block_attrs_t attrs;

  // Variable-length trailing data follows in descending alignment order:
  // - uint32_t wait_payload_deltas[wait_semaphore_count]
  // - uint32_t signal_payload_deltas[signal_semaphore_count]
  // - uint16_t wait_semaphore_indices[wait_semaphore_count]
  // - uint16_t signal_semaphore_indices[signal_semaphore_count]
} iree_hal_streaming_graph_block_t;

// Blocks are carved out of an arena, which guarantees iree_max_align_t and
// nothing stronger.
static_assert(iree_alignof(iree_hal_streaming_graph_block_t) <=
                  iree_max_align_t,
              "block alignment must be satisfiable by an arena allocation");

// The first trailing region begins at the end of the header and holds
// uint32_t, so the header's size decides whether that region is aligned.
// The region order only keeps the regions aligned relative to each other;
// this assert is what pins the offset they start from.
static_assert(sizeof(iree_hal_streaming_graph_block_t) %
                      iree_alignof(uint32_t) ==
                  0,
              "block header must end on the first trailing region's alignment");

// Pointers to the inline attributes and variable-length arrays in a block.
typedef struct iree_hal_streaming_graph_block_ptrs_t {
  uint32_t* wait_payload_deltas;
  uint32_t* signal_payload_deltas;
  uint16_t* wait_semaphore_indices;
  uint16_t* signal_semaphore_indices;
  iree_hal_streaming_graph_block_attrs_t* attrs;
} iree_hal_streaming_graph_block_ptrs_t;

typedef struct iree_hal_streaming_graph_exec_t {
  // Resource header used to retain executable storage through queued calls.
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  // Context retained for executable compilation and launch.
  iree_hal_streaming_context_t* context;
  // Public source graph retained for node identity and update compatibility.
  iree_hal_streaming_graph_t* graph;
  // Executable-private graph template containing cumulative node overrides.
  iree_hal_streaming_graph_t* template_graph;
  // True after the public HIP graph-exec handle has been destroyed.
  bool is_destroyed;

  // Arena allocator used for block allocations and inlined data.
  iree_arena_allocator_t arena_allocator;

  // Immutable block list created during instantiate.
  iree_hal_streaming_graph_block_t** blocks;
  uint32_t block_count;
  // True when |blocks|, or the blocks of a child-graph executable they launch,
  // hold an event record. Every such record names an event of |context|, so a
  // launch on a stream of any other context would have every one of its
  // records refused.
  bool records_events;
  // Number of graph nodes present when this executable was instantiated.
  iree_host_size_t instantiated_node_count;
  // Number of HIP-visible graph nodes present at instantiation/update time.
  iree_host_size_t instantiated_visible_node_count;
  // Per-instantiated-node disabled state keyed by graph node index.
  uint8_t* node_disabled_states;
  // Number of entries in |node_disabled_states|.
  iree_host_size_t node_disabled_state_count;

  // Semaphore pool for internal synchronization.
  uint32_t semaphore_count;
  iree_hal_semaphore_t** semaphores;
  uint64_t* semaphore_base_values;

  // Resource set for automatic cleanup.
  iree_hal_resource_set_t* resource_set;

  // True when this graph has allocation nodes without same-graph free nodes.
  bool has_graph_alloc_nodes_without_free_nodes;
  // Number of successful launches of this exec.
  uint64_t launch_count;

  // True if this exec contributes to graph memory-node instantiation limits.
  bool uses_graph_memory_nodes;

  // Stream retained while the most recent graph-memory launch is in flight.
  iree_hal_streaming_stream_t* graph_memory_active_launch_stream;
  // Timeline value signaled by |graph_memory_active_launch_stream|.
  uint64_t graph_memory_active_launch_value;

  // Launch completion records used only when the public handle is destroyed.
  // The first record is inline because nearly all executables use one stream.
  iree_hal_streaming_graph_launch_point_t* launch_points;
  // Inline storage for the common single-stream launch case.
  iree_hal_streaming_graph_launch_point_t inline_launch_point;

  unsigned long long flags;

  // Mutex needed for launch/update.
  iree_slim_mutex_t mutex;
} iree_hal_streaming_graph_exec_t;

// Retains an executable until a stream-ordered retirement call either runs or
// is cancelled. A separate resource is required because the normal callback
// releases the executable before its signal semaphore advances.
typedef struct iree_hal_streaming_graph_exec_retirement_t {
  // Resource retained by the queued host call through invocation or
  // cancellation.
  iree_hal_resource_t resource;
  // Allocator that owns this retirement record.
  iree_allocator_t host_allocator;
  // Executable reference transferred to the host callback when it runs.
  iree_hal_streaming_graph_exec_t* exec;
} iree_hal_streaming_graph_exec_retirement_t;

static void iree_hal_streaming_graph_exec_destroy(
    iree_hal_streaming_graph_exec_t* exec);

static void iree_hal_streaming_graph_exec_retirement_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_streaming_graph_exec_retirement_t* retirement =
      (iree_hal_streaming_graph_exec_retirement_t*)resource;
  iree_hal_streaming_graph_exec_release(retirement->exec);
  retirement->exec = NULL;
  iree_allocator_free(retirement->host_allocator, retirement);
}

static const iree_hal_resource_vtable_t
    iree_hal_streaming_graph_exec_retirement_vtable = {
        .destroy = iree_hal_streaming_graph_exec_retirement_destroy,
};

static void iree_hal_streaming_graph_exec_resource_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_streaming_graph_exec_destroy(
      (iree_hal_streaming_graph_exec_t*)resource);
}

static const iree_hal_resource_vtable_t
    iree_hal_streaming_graph_exec_resource_vtable = {
        .destroy = iree_hal_streaming_graph_exec_resource_destroy,
};
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
    if (attrs->execution_src_buffer == attrs->execution_dst_buffer &&
        attrs->src_ref.offset == attrs->dst_ref.offset) {
      return iree_ok_status();
    }
    return iree_hal_command_buffer_copy_buffer(
        command_buffer,
        iree_hal_make_buffer_ref(attrs->execution_src_buffer,
                                 attrs->src_ref.offset, total_size),
        iree_hal_make_buffer_ref(attrs->execution_dst_buffer,
                                 attrs->dst_ref.offset, total_size),
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
      if (attrs->execution_src_buffer == attrs->execution_dst_buffer &&
          src_offset == dst_offset) {
        continue;
      }
      IREE_RETURN_IF_ERROR(iree_hal_command_buffer_copy_buffer(
          command_buffer,
          iree_hal_make_buffer_ref(attrs->execution_src_buffer, src_offset,
                                   width),
          iree_hal_make_buffer_ref(attrs->execution_dst_buffer, dst_offset,
                                   width),
          attrs->flags));
    }
  }
  return iree_ok_status();
}

static inline void iree_hal_streaming_graph_block_get_ptrs(
    iree_hal_streaming_graph_block_t* block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs);
static void iree_hal_streaming_graph_exec_destroy(
    iree_hal_streaming_graph_exec_t* exec);
static void iree_hal_streaming_graph_exec_release_child_blocks(
    iree_hal_streaming_graph_exec_t* exec);
static void iree_hal_streaming_graph_exec_deinitialize_compiled_state(
    iree_hal_streaming_graph_exec_t* exec);
static iree_status_t iree_hal_streaming_graph_exec_rebuild_from_template_locked(
    iree_hal_streaming_graph_exec_t* exec);
static iree_host_size_t iree_hal_streaming_graph_visible_node_count(
    const iree_hal_streaming_graph_t* graph);

static void iree_hal_streaming_graph_exec_scan_nodes(
    iree_hal_streaming_graph_exec_t* exec) {
  for (iree_hal_streaming_node_block_t* block =
           exec->template_graph->node_blocks;
       block; block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC) {
        exec->has_graph_alloc_nodes_without_free_nodes |=
            !node->attrs.mem_alloc.has_in_graph_free_node;
      }
    }
  }
}

static iree_status_t iree_hal_streaming_graph_memory_auto_free(
    iree_hal_streaming_graph_exec_t* exec) {
  for (iree_hal_streaming_node_block_t* block =
           exec->template_graph->node_blocks;
       block; block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC ||
          node->attrs.mem_alloc.has_in_graph_free_node) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          iree_hal_streaming_graph_memory_allocation_unmap_if_mapped(
              node->attrs.mem_alloc.allocation));
    }
  }
  return iree_ok_status();
}

// Checks whether an unmatched allocation from a prior launch still has live
// physical backing. External stream-ordered frees can clear this state, so
// graph-template metadata alone cannot decide whether relaunch is valid.
static bool iree_hal_streaming_graph_memory_has_live_unfreed_allocations(
    iree_hal_streaming_graph_exec_t* exec) {
  for (iree_hal_streaming_node_block_t* block =
           exec->template_graph->node_blocks;
       block; block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC &&
          !node->attrs.mem_alloc.has_in_graph_free_node &&
          iree_hal_streaming_graph_memory_allocation_is_mapped(
              node->attrs.mem_alloc.allocation)) {
        return true;
      }
    }
  }
  return false;
}

static iree_status_t iree_hal_streaming_graph_memory_prepare_launch_pointers(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_memory_allocation_t*** out_publications,
    iree_host_size_t* out_publication_count) {
  IREE_ASSERT_ARGUMENT(exec);
  IREE_ASSERT_ARGUMENT(out_publications);
  IREE_ASSERT_ARGUMENT(out_publication_count);
  *out_publications = NULL;
  *out_publication_count = 0;

  iree_host_size_t allocation_count = 0;
  for (iree_hal_streaming_node_block_t* block =
           exec->template_graph->node_blocks;
       block; block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      allocation_count +=
          block->nodes[i]->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC;
    }
  }
  if (allocation_count == 0) {
    return iree_ok_status();
  }

  iree_host_size_t publication_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          allocation_count, sizeof(**out_publications), &publication_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "graph allocation publication list overflows");
  }
  iree_hal_streaming_graph_memory_allocation_t** publications = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      exec->host_allocator, publication_size, (void**)&publications));

  iree_status_t status = iree_ok_status();
  iree_host_size_t publication_count = 0;
  for (iree_hal_streaming_node_block_t* block =
           exec->template_graph->node_blocks;
       block && iree_status_is_ok(status); block = block->next) {
    for (iree_host_size_t i = 0; i < block->count && iree_status_is_ok(status);
         ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC) {
        continue;
      }
      bool did_publish = false;
      status =
          iree_hal_streaming_graph_memory_allocation_prepare_pointer_for_launch(
              node->attrs.mem_alloc.allocation, &did_publish);
      if (did_publish) {
        publications[publication_count++] = node->attrs.mem_alloc.allocation;
      }
    }
  }

  if (!iree_status_is_ok(status)) {
    while (publication_count > 0) {
      status = iree_status_join(
          status,
          iree_hal_streaming_graph_memory_allocation_rollback_launch_pointer(
              publications[--publication_count]));
    }
    iree_allocator_free(exec->host_allocator, publications);
    return status;
  }

  *out_publications = publications;
  *out_publication_count = publication_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_memory_rollback_launch_pointers(
    iree_host_size_t publication_count,
    iree_hal_streaming_graph_memory_allocation_t** publications) {
  iree_status_t status = iree_ok_status();
  while (publication_count > 0) {
    status = iree_status_join(
        status,
        iree_hal_streaming_graph_memory_allocation_rollback_launch_pointer(
            publications[--publication_count]));
  }
  return status;
}

static bool iree_hal_streaming_graph_memory_exec_try_claim(
    iree_hal_streaming_graph_t* graph) {
  if (!graph || !graph->has_graph_memory_nodes) {
    return true;
  }
  int32_t expected = 0;
  return iree_atomic_compare_exchange_strong(
      &graph->active_graph_memory_exec_count, &expected, 1,
      iree_memory_order_acq_rel, iree_memory_order_acquire);
}

static void iree_hal_streaming_graph_memory_exec_release(
    iree_hal_streaming_graph_t* graph) {
  if (!graph || !graph->has_graph_memory_nodes) {
    return;
  }
  const int32_t previous = iree_atomic_fetch_sub(
      &graph->active_graph_memory_exec_count, 1, iree_memory_order_acq_rel);
  IREE_ASSERT(previous == 1,
              "graph memory executable claim was not exclusively owned");
}

// Internal: Create an exec object (called by graph.c).
iree_status_t iree_hal_streaming_graph_exec_create(
    iree_hal_streaming_context_t* context, iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_instantiate_flags_t flags,
    iree_allocator_t host_allocator,
    iree_hal_streaming_graph_exec_t** out_exec) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(out_exec);
  *out_exec = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!iree_hal_streaming_graph_memory_exec_try_claim(graph)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "graphs with memory allocation nodes support one live executable");
  }

  iree_hal_streaming_graph_exec_t* exec = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*exec), (void**)&exec);
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_graph_memory_exec_release(graph);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_hal_resource_initialize(&iree_hal_streaming_graph_exec_resource_vtable,
                               &exec->resource);
  exec->host_allocator = host_allocator;
  exec->context = context;
  iree_hal_streaming_context_retain(exec->context);
  exec->graph = graph;
  iree_hal_streaming_graph_retain(exec->graph);
  exec->template_graph = NULL;
  exec->is_destroyed = false;
  iree_arena_initialize(&context->device_entry->block_pool,
                        &exec->arena_allocator);
  exec->blocks = NULL;
  exec->block_count = 0;
  exec->records_events = false;
  exec->instantiated_node_count = 0;
  exec->instantiated_visible_node_count = 0;
  exec->node_disabled_states = NULL;
  exec->node_disabled_state_count = graph->node_count;
  exec->semaphores = NULL;
  exec->semaphore_count = 0;
  exec->semaphore_base_values = NULL;
  exec->resource_set = NULL;
  exec->has_graph_alloc_nodes_without_free_nodes = false;
  exec->launch_count = 0;
  exec->uses_graph_memory_nodes = graph->has_graph_memory_nodes;
  exec->graph_memory_active_launch_stream = NULL;
  exec->graph_memory_active_launch_value = 0;
  exec->launch_points = NULL;
  memset(&exec->inline_launch_point, 0, sizeof(exec->inline_launch_point));
  exec->flags = flags;
  iree_slim_mutex_initialize(&exec->mutex);

  // Create resource set for automatic cleanup.
  status = iree_hal_resource_set_allocate(&context->device_entry->block_pool,
                                          &exec->resource_set);
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_clone_for_exec(graph, context,
                                                     &exec->template_graph);
  }
  if (iree_status_is_ok(status) && exec->node_disabled_state_count > 0) {
    iree_host_size_t node_disabled_states_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            exec->node_disabled_state_count,
            sizeof(*exec->node_disabled_states), &node_disabled_states_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "node disabled-state size overflow");
    } else {
      status = iree_allocator_malloc(host_allocator, node_disabled_states_size,
                                     (void**)&exec->node_disabled_states);
    }
    if (iree_status_is_ok(status)) {
      memset(exec->node_disabled_states, 0,
             exec->node_disabled_state_count *
                 sizeof(*exec->node_disabled_states));
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_streaming_graph_exec_scan_nodes(exec);
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

  if (exec->uses_graph_memory_nodes) {
    iree_hal_streaming_graph_memory_exec_release(exec->graph);
  }
  iree_hal_streaming_stream_release(exec->graph_memory_active_launch_stream);
  exec->graph_memory_active_launch_stream = NULL;
  exec->graph_memory_active_launch_value = 0;
  IREE_ASSERT(!exec->launch_points,
              "graph executable destroyed with unretired launch points");

  iree_allocator_free(exec->host_allocator, exec->node_disabled_states);

  iree_hal_streaming_graph_release(exec->template_graph);
  iree_hal_streaming_graph_release(exec->graph);
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
    if (!block) {
      continue;
    }
    iree_hal_streaming_graph_block_ptrs_t ptrs;
    iree_hal_streaming_graph_block_get_ptrs(block, &ptrs);
    switch (block->type) {
      case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH:
        iree_hal_streaming_graph_exec_release(ptrs.attrs->child_graph.exec);
        ptrs.attrs->child_graph.exec = NULL;
        break;
      case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD:
      case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_WAIT:
        iree_hal_streaming_event_release(ptrs.attrs->event.event);
        ptrs.attrs->event.event = NULL;
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
  exec->instantiated_node_count = 0;
  exec->instantiated_visible_node_count = 0;
  exec->semaphores = NULL;
  exec->semaphore_count = 0;
  exec->semaphore_base_values = NULL;
  exec->resource_set = NULL;
  exec->has_graph_alloc_nodes_without_free_nodes = false;
}

static void iree_hal_streaming_graph_exec_deinitialize_compiled_state(
    iree_hal_streaming_graph_exec_t* exec) {
  iree_hal_streaming_graph_exec_release_child_blocks(exec);
  if (exec->resource_set) {
    iree_hal_resource_set_free(exec->resource_set);
    exec->resource_set = NULL;
  }
  iree_arena_deinitialize(&exec->arena_allocator);
  exec->blocks = NULL;
  exec->block_count = 0;
  exec->records_events = false;
  exec->instantiated_node_count = 0;
  exec->instantiated_visible_node_count = 0;
  exec->semaphores = NULL;
  exec->semaphore_count = 0;
  exec->semaphore_base_values = NULL;
  exec->has_graph_alloc_nodes_without_free_nodes = false;
}

static void iree_hal_streaming_graph_exec_move_compiled_state(
    iree_hal_streaming_graph_exec_t* target,
    iree_hal_streaming_graph_exec_t* source) {
  target->arena_allocator = source->arena_allocator;
  target->blocks = source->blocks;
  target->block_count = source->block_count;
  target->records_events = source->records_events;
  target->instantiated_node_count = source->instantiated_node_count;
  target->instantiated_visible_node_count =
      source->instantiated_visible_node_count;
  target->semaphores = source->semaphores;
  target->semaphore_count = source->semaphore_count;
  target->semaphore_base_values = source->semaphore_base_values;
  target->resource_set = source->resource_set;
  target->has_graph_alloc_nodes_without_free_nodes =
      source->has_graph_alloc_nodes_without_free_nodes;

  source->blocks = NULL;
  source->block_count = 0;
  source->records_events = false;
  source->instantiated_node_count = 0;
  source->instantiated_visible_node_count = 0;
  source->semaphores = NULL;
  source->semaphore_count = 0;
  source->semaphore_base_values = NULL;
  source->resource_set = NULL;
  source->has_graph_alloc_nodes_without_free_nodes = false;
}

void iree_hal_streaming_graph_exec_retain(
    iree_hal_streaming_graph_exec_t* exec) {
  iree_hal_resource_retain(exec ? &exec->resource : NULL);
}

void iree_hal_streaming_graph_exec_release(
    iree_hal_streaming_graph_exec_t* exec) {
  iree_hal_resource_release(exec ? &exec->resource : NULL);
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

static iree_status_t iree_hal_streaming_graph_exec_retire_host_call(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context) {
  (void)args;
  (void)context;
  iree_hal_streaming_graph_exec_retirement_t* retirement =
      (iree_hal_streaming_graph_exec_retirement_t*)user_data;

  // Returning OK advances the stream timeline. Release first so observing the
  // retirement point also proves executable-owned arenas and graph references
  // are no longer being torn down by the queue worker.
  iree_hal_streaming_graph_exec_t* exec = retirement->exec;
  retirement->exec = NULL;
  iree_hal_streaming_graph_exec_release(exec);
  return iree_ok_status();
}

static void iree_hal_streaming_graph_exec_retire_launch_point(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_launch_point_t* point) {
  iree_hal_streaming_graph_exec_retirement_t* retirement = NULL;
  iree_status_t status = iree_allocator_malloc(
      exec->host_allocator, sizeof(*retirement), (void**)&retirement);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    // Fall back to synchronous retirement when no callback record can be
    // allocated. The destroy routine still owns the public-handle reference
    // while this waits and drops it after every launch point is terminal.
    iree_status_ignore(iree_hal_semaphore_wait(
        point->stream->timeline_semaphore, point->value,
        iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    return;
  }

  iree_hal_resource_initialize(&iree_hal_streaming_graph_exec_retirement_vtable,
                               &retirement->resource);
  retirement->host_allocator = exec->host_allocator;
  retirement->exec = exec;
  iree_hal_streaming_graph_exec_retain(exec);

  const uint64_t args[4] = {0};
  const iree_hal_host_call_t retire_call =
      iree_hal_make_host_call_with_resource(
          iree_hal_streaming_graph_exec_retire_host_call, retirement,
          &retirement->resource);

  // Queueing through the stream gives the retirement callback a completion
  // point of its own. It may wait on work submitted after this executable's
  // last launch, but destruction is cold and the stronger ordering avoids a
  // callback with no completion signal while keeping the public call
  // nonblocking.
  status = iree_hal_streaming_queue_host_call(point->stream, retire_call, args,
                                              IREE_HAL_HOST_CALL_FLAG_NONE);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    // Queue rejection transferred no resource reference. Waiting for the
    // recorded point is the allocation-free fallback that still proves every
    // operation which can reference executable-owned storage is terminal.
    iree_status_ignore(iree_hal_semaphore_wait(
        point->stream->timeline_semaphore, point->value,
        iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }
  iree_hal_resource_release(&retirement->resource);
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
  iree_hal_streaming_graph_launch_point_t* launch_points = exec->launch_points;
  exec->launch_points = NULL;
  iree_slim_mutex_unlock(&exec->mutex);

  while (launch_points) {
    iree_hal_streaming_graph_launch_point_t* point = launch_points;
    launch_points = point->next;
    iree_hal_streaming_graph_exec_retire_launch_point(exec, point);
    iree_hal_streaming_stream_release(point->stream);
    point->stream = NULL;
    point->value = 0;
    point->next = NULL;
    if (point != &exec->inline_launch_point) {
      iree_allocator_free(exec->host_allocator, point);
    }
  }

  iree_hal_streaming_graph_exec_release(exec);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_exec_prepare_launch_point_locked(
    iree_hal_streaming_graph_exec_t* exec, iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_graph_launch_point_t** out_point, bool* out_inserted) {
  *out_point = NULL;
  *out_inserted = false;
  for (iree_hal_streaming_graph_launch_point_t* point = exec->launch_points;
       point; point = point->next) {
    if (point->stream == stream) {
      *out_point = point;
      return iree_ok_status();
    }
  }

  iree_hal_streaming_graph_launch_point_t* point = NULL;
  if (!exec->inline_launch_point.stream) {
    point = &exec->inline_launch_point;
  } else {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(exec->host_allocator,
                                               sizeof(*point), (void**)&point));
  }
  memset(point, 0, sizeof(*point));
  iree_hal_streaming_stream_retain(stream);
  point->stream = stream;
  point->next = exec->launch_points;
  exec->launch_points = point;
  *out_point = point;
  *out_inserted = true;
  return iree_ok_status();
}

static void iree_hal_streaming_graph_exec_rollback_launch_point_locked(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_launch_point_t* point) {
  iree_hal_streaming_graph_launch_point_t** previous_next =
      &exec->launch_points;
  while (*previous_next && *previous_next != point) {
    previous_next = &(*previous_next)->next;
  }
  IREE_ASSERT(*previous_next == point,
              "launch point is not linked to executable");
  *previous_next = point->next;
  iree_hal_streaming_stream_release(point->stream);
  point->stream = NULL;
  point->value = 0;
  point->next = NULL;
  if (point != &exec->inline_launch_point) {
    iree_allocator_free(exec->host_allocator, point);
  }
}

iree_hal_streaming_graph_instantiate_flags_t
iree_hal_streaming_graph_exec_flags(iree_hal_streaming_graph_exec_t* exec) {
  IREE_ASSERT_ARGUMENT(exec);
  return (iree_hal_streaming_graph_instantiate_flags_t)exec->flags;
}

bool iree_hal_streaming_graph_exec_owns_node(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node) {
  return exec && node && node->graph == exec->graph &&
         node->node_index < exec->instantiated_node_count;
}

static iree_hal_streaming_graph_node_t*
iree_hal_streaming_graph_exec_template_node_at_index(
    iree_hal_streaming_graph_exec_t* exec, uint32_t node_index) {
  iree_host_size_t skipped_count = 0;
  for (iree_hal_streaming_node_block_t* block =
           exec->template_graph->node_blocks;
       block; block = block->next) {
    if (node_index < skipped_count + block->count) {
      return block->nodes[node_index - skipped_count];
    }
    skipped_count += block->count;
  }
  return NULL;
}

iree_status_t iree_hal_streaming_graph_exec_begin_node_update(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* source_node,
    iree_hal_streaming_graph_node_t** out_template_node) {
  if (!exec || !source_node || !out_template_node) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  *out_template_node = NULL;
  iree_slim_mutex_lock(&exec->mutex);
  if (exec->is_destroyed ||
      !iree_hal_streaming_graph_exec_owns_node(exec, source_node)) {
    iree_slim_mutex_unlock(&exec->mutex);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  iree_hal_streaming_graph_node_t* template_node =
      iree_hal_streaming_graph_exec_template_node_at_index(
          exec, source_node->node_index);
  if (!template_node || template_node->type != source_node->type) {
    iree_slim_mutex_unlock(&exec->mutex);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "executable graph template does not match source");
  }
  *out_template_node = template_node;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_graph_exec_rebuild_node_update(
    iree_hal_streaming_graph_exec_t* exec) {
  IREE_ASSERT_ARGUMENT(exec);
  return iree_hal_streaming_graph_exec_rebuild_from_template_locked(exec);
}

void iree_hal_streaming_graph_exec_end_node_update(
    iree_hal_streaming_graph_exec_t* exec) {
  IREE_ASSERT_ARGUMENT(exec);
  iree_slim_mutex_unlock(&exec->mutex);
}

bool iree_hal_streaming_graph_exec_node_is_enabled(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node) {
  if (!exec) {
    return false;
  }
  iree_slim_mutex_lock(&exec->mutex);
  const bool is_enabled = iree_hal_streaming_graph_exec_owns_node(exec, node) &&
                          node->node_index < exec->node_disabled_state_count &&
                          exec->node_disabled_states[node->node_index] == 0;
  iree_slim_mutex_unlock(&exec->mutex);
  return is_enabled;
}

iree_status_t iree_hal_streaming_graph_exec_set_node_enabled(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node, bool enabled) {
  if (!exec || !node) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  iree_slim_mutex_lock(&exec->mutex);
  if (exec->is_destroyed ||
      !iree_hal_streaming_graph_exec_owns_node(exec, node) ||
      node->node_index >= exec->node_disabled_state_count) {
    iree_slim_mutex_unlock(&exec->mutex);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  const uint8_t old_disabled_state =
      exec->node_disabled_states[node->node_index];
  exec->node_disabled_states[node->node_index] = enabled ? 0 : 1;
  iree_status_t status =
      iree_hal_streaming_graph_exec_rebuild_from_template_locked(exec);
  if (!iree_status_is_ok(status)) {
    exec->node_disabled_states[node->node_index] = old_disabled_state;
  }
  iree_slim_mutex_unlock(&exec->mutex);
  return status;
}

iree_status_t iree_hal_streaming_graph_exec_rebuild_from_template(
    iree_hal_streaming_graph_exec_t* exec) {
  IREE_ASSERT_ARGUMENT(exec);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&exec->mutex);
  iree_status_t status =
      iree_hal_streaming_graph_exec_rebuild_from_template_locked(exec);
  iree_slim_mutex_unlock(&exec->mutex);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_streaming_graph_exec_rebuild_from_template_locked(
    iree_hal_streaming_graph_exec_t* exec) {
  if (exec->is_destroyed) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }

  iree_hal_streaming_graph_exec_t candidate;
  memset(&candidate, 0, sizeof(candidate));
  candidate.host_allocator = exec->host_allocator;
  candidate.context = exec->context;
  candidate.graph = exec->graph;
  candidate.template_graph = exec->template_graph;
  candidate.node_disabled_states = exec->node_disabled_states;
  candidate.node_disabled_state_count = exec->node_disabled_state_count;
  candidate.uses_graph_memory_nodes = exec->uses_graph_memory_nodes;
  candidate.flags = exec->flags;
  iree_hal_streaming_graph_exec_initialize_compiled_state(&candidate);

  iree_status_t status = iree_hal_resource_set_allocate(
      &candidate.context->device_entry->block_pool, &candidate.resource_set);
  if (iree_status_is_ok(status)) {
    iree_hal_streaming_graph_exec_scan_nodes(&candidate);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_exec_instantiate_from_template(&candidate,
                                                                     exec);
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

  iree_hal_streaming_graph_node_t* template_node = NULL;
  iree_status_t status = iree_hal_streaming_graph_exec_begin_node_update(
      exec, node, &template_node);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  if (template_node->type != type) {
    iree_hal_streaming_graph_exec_end_node_update(exec);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }

  iree_hal_streaming_event_t* old_event = template_node->attrs.event.event;
  iree_hal_streaming_event_retain(event);
  template_node->attrs.event.event = event;
  status = iree_hal_streaming_graph_exec_rebuild_node_update(exec);
  if (iree_status_is_ok(status)) {
    iree_hal_streaming_event_release(old_event);
  } else {
    template_node->attrs.event.event = old_event;
    iree_hal_streaming_event_release(event);
  }
  iree_hal_streaming_graph_exec_end_node_update(exec);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Calculate the size needed for a block with variable-length arrays.
static iree_host_size_t iree_hal_streaming_graph_block_calculate_size(
    uint16_t wait_semaphore_count, uint16_t signal_semaphore_count) {
  iree_host_size_t size = sizeof(iree_hal_streaming_graph_block_t);
  size += wait_semaphore_count * sizeof(uint32_t);    // wait_payload_deltas
  size += signal_semaphore_count * sizeof(uint32_t);  // signal_payload_deltas
  size += wait_semaphore_count * sizeof(uint16_t);    // wait_semaphore_indices
  size +=
      signal_semaphore_count * sizeof(uint16_t);  // signal_semaphore_indices
  return size;
}

// Resolves the payload pointers for |block|. The trailing arrays run in
// descending alignment order - both payload delta arrays before both semaphore
// index arrays - starting at the end of the block header, whose size is a
// multiple of the block's own alignment. Every region therefore lands on its
// natural alignment and the block holds no interior padding.
// iree_hal_streaming_graph_block_calculate_size uses this same order.
static inline void iree_hal_streaming_graph_block_get_ptrs(
    iree_hal_streaming_graph_block_t* block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  uint8_t* ptr = (uint8_t*)block + sizeof(*block);

  out_ptrs->wait_payload_deltas = (uint32_t*)ptr;
  ptr += block->wait_semaphore_count * sizeof(*out_ptrs->wait_payload_deltas);

  out_ptrs->signal_payload_deltas = (uint32_t*)ptr;
  ptr +=
      block->signal_semaphore_count * sizeof(*out_ptrs->signal_payload_deltas);

  out_ptrs->wait_semaphore_indices = (uint16_t*)ptr;
  ptr +=
      block->wait_semaphore_count * sizeof(*out_ptrs->wait_semaphore_indices);

  out_ptrs->signal_semaphore_indices = (uint16_t*)ptr;
  out_ptrs->attrs = &block->attrs;
}

// Allocates a block with variable-length arrays.
static iree_status_t iree_hal_streaming_graph_block_allocate(
    iree_arena_allocator_t* arena_allocator,
    iree_hal_streaming_graph_block_type_t type, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count,
    iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  const iree_host_size_t total_size =
      iree_hal_streaming_graph_block_calculate_size(wait_semaphore_count,
                                                    signal_semaphore_count);
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena_allocator, total_size, (void**)&block));

  block->type = type;
  block->node_start_index = node_start_index;
  block->node_count = node_count;
  block->wait_semaphore_count = wait_semaphore_count;
  block->signal_semaphore_count = signal_semaphore_count;

  iree_hal_streaming_graph_block_get_ptrs(block, out_ptrs);
  *out_block = block;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_create_barrier_block(
    iree_hal_streaming_graph_exec_t* exec, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count, iree_hal_queue_barrier_flags_t flags,
    iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate block with variable-length arrays.
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER,
              node_start_index, node_count, wait_semaphore_count,
              signal_semaphore_count, &block, out_ptrs));

  // Set barrier attributes.
  iree_hal_streaming_graph_barrier_block_attrs_t* attrs =
      &out_ptrs->attrs->barrier;
  attrs->flags = flags;

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_create_fill_block(
    iree_hal_streaming_graph_exec_t* exec, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    const void* pattern, iree_host_size_t pattern_length,
    iree_hal_fill_flags_t flags, iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate block with variable-length arrays.
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_FILL, node_start_index,
              node_count, wait_semaphore_count, signal_semaphore_count, &block,
              out_ptrs));

  // Set fill attributes.
  iree_hal_streaming_graph_fill_block_attrs_t* attrs = &out_ptrs->attrs->fill;
  attrs->target_buffer = target_buffer;
  attrs->target_offset = target_offset;
  attrs->length = length;
  attrs->flags = flags;

  // Copy pattern data if provided.
  if (pattern_length > 0) {
    IREE_ASSERT(pattern_length < sizeof(attrs->pattern));
    memcpy(&attrs->pattern, pattern, pattern_length);
    attrs->pattern_length = pattern_length;
  }

  // Add buffer to resource set.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(exec->resource_set, 1, &target_buffer));

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_create_copy_block(
    iree_hal_streaming_graph_exec_t* exec, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_hal_copy_flags_t flags, iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate block with variable-length arrays.
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_COPY, node_start_index,
              node_count, wait_semaphore_count, signal_semaphore_count, &block,
              out_ptrs));

  // Set copy attributes.
  iree_hal_streaming_graph_copy_block_attrs_t* attrs = &out_ptrs->attrs->copy;
  attrs->source_buffer = source_buffer;
  attrs->source_offset = source_offset;
  attrs->target_buffer = target_buffer;
  attrs->target_offset = target_offset;
  attrs->length = length;
  attrs->flags = flags;

  // Add buffers to resource set.
  void* resources[2] = {source_buffer, target_buffer};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(exec->resource_set,
                                       IREE_ARRAYSIZE(resources), resources));

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_create_host_call_block(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_exec_t* resource_owner, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count, iree_hal_host_call_t call,
    const uint64_t args[4], const iree_hal_streaming_graph_node_t* source_node,
    iree_hal_host_call_flags_t flags,
    iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate block with variable-length arrays.
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL,
              node_start_index, node_count, wait_semaphore_count,
              signal_semaphore_count, &block, out_ptrs));

  // Set host call attributes.
  iree_hal_streaming_graph_host_call_block_attrs_t* attrs =
      &out_ptrs->attrs->host_call;
  attrs->call = call;
  memcpy(attrs->args, args, sizeof(attrs->args));
  if (source_node && source_node->attrs.host.user_data_size > 0 && args[1]) {
    void* copied_data = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(&exec->arena_allocator,
                                source_node->attrs.host.user_data_size,
                                (void**)&copied_data));
    memcpy(copied_data, (const void*)args[1],
           source_node->attrs.host.user_data_size);
    attrs->args[1] = (uint64_t)copied_data;
  }
  attrs->flags = flags;
  // The queue retains executable-backed callback storage while it is in flight.
  // The compiled resource set must not retain its persistent |resource_owner|:
  // rebuilds compile into |exec| as a temporary container and then move that
  // resource set into |resource_owner|, where the reference would form a cycle.
  if (call.resource && call.resource != &resource_owner->resource) {
    void* resources[1] = {call.resource};
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_resource_set_insert(exec->resource_set, 1, resources));
  }

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_create_event_block(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_block_type_t type, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count,
    iree_hal_streaming_graph_node_t* source_node,
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (type != IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD &&
      type != IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_WAIT) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid graph event block type");
  }

  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator, type, node_start_index, node_count,
              wait_semaphore_count, signal_semaphore_count, &block, out_ptrs));

  out_ptrs->attrs->event.source_node = source_node;
  out_ptrs->attrs->event.event = event;
  iree_hal_streaming_event_retain(event);
  // The one place a record block is built, and so where |exec| learns it holds
  // one. A launch reads this to answer for every record at once.
  if (type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD) {
    exec->records_events = true;
  }

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_create_dispatch_block(
    iree_hal_streaming_graph_exec_t* exec, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count, iree_hal_executable_t* executable,
    iree_host_size_t entry_point, iree_hal_dispatch_config_t config,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags,
    iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate block with variable-length arrays.
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH,
              node_start_index, node_count, wait_semaphore_count,
              signal_semaphore_count, &block, out_ptrs));

  // Set dispatch attributes.
  iree_hal_streaming_graph_dispatch_block_attrs_t* attrs =
      &out_ptrs->attrs->dispatch;
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

  *out_block = block;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_create_execute_block(
    iree_hal_streaming_graph_exec_t* exec, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count, iree_hal_queue_execute_flags_t flags,
    iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate block with variable-length arrays.
  iree_hal_streaming_graph_block_t* block = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_block_allocate(
              &exec->arena_allocator,
              IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE,
              node_start_index, node_count, wait_semaphore_count,
              signal_semaphore_count, &block, out_ptrs));

  iree_hal_streaming_graph_execute_block_attrs_t* attrs =
      &out_ptrs->attrs->execute;
  attrs->flags = flags;

  // Create command buffer. HIP pointer lifetime is enforced by explicit API
  // ordering, not by scanning graph or kernarg contents: synchronous
  // destruction waits for active streams, and stream-ordered destruction queues
  // the release after prior work.
  //
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_create(
              iree_hal_queue_family(exec->context->queue),
              IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED,
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

static iree_status_t iree_hal_streaming_graph_create_child_graph_block(
    iree_hal_streaming_graph_exec_t* exec, uint32_t node_start_index,
    uint32_t node_count, uint16_t wait_semaphore_count,
    uint16_t signal_semaphore_count, iree_hal_streaming_graph_t* child_graph,
    iree_hal_streaming_graph_block_t** out_block,
    iree_hal_streaming_graph_block_ptrs_t* out_ptrs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_graph_exec_t* child_exec = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_instantiate(
              child_graph,
              (iree_hal_streaming_graph_instantiate_flags_t)exec->flags,
              &child_exec));

  iree_hal_streaming_graph_block_t* block = NULL;
  iree_status_t status = iree_hal_streaming_graph_block_allocate(
      &exec->arena_allocator, IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH,
      node_start_index, node_count, wait_semaphore_count,
      signal_semaphore_count, &block, out_ptrs);
  if (iree_status_is_ok(status)) {
    out_ptrs->attrs->child_graph.exec = child_exec;
    // A launch of |exec| walks the child's blocks too, so the records they hold
    // are records of this launch. The child instantiated above already carries
    // its own children's, which makes the property transitive.
    exec->records_events |= child_exec->records_events;
    *out_block = block;
  } else {
    iree_hal_streaming_graph_exec_release(child_exec);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

typedef struct iree_hal_streaming_node_index_set_t {
  uint32_t values[8];
  uint32_t count : 31;
  uint32_t invalid : 1;
} iree_hal_streaming_node_index_set_t;

// Resets the set to empty.
static inline void iree_hal_streaming_node_index_set_reset(
    iree_hal_streaming_node_index_set_t* set) {
  set->count = 0;
  set->invalid = 0;
}

// Returns true if the |set| is invalid or |value| is present.
static bool iree_hal_streaming_node_index_set_test_hazard(
    const iree_hal_streaming_node_index_set_t* set, uint32_t value) {
  if (set->invalid) {
    return true;
  }
  for (uint32_t i = 0; i < set->count; ++i) {
    if (set->values[i] == value) {
      return true;
    }
  }
  return false;
}

// Inserts |value| into the |set|.
// If the set has reached capacity it is set to invalid and all future tests
// will return a hazard.
static void iree_hal_streaming_node_index_set_insert(
    iree_hal_streaming_node_index_set_t* set, uint32_t value) {
  if (set->count >= IREE_ARRAYSIZE(set->values)) {
    set->invalid = 1;
    return;
  }
  set->values[set->count++] = value;
}

static bool iree_hal_streaming_graph_node_has_recorded_dependency_hazard(
    const iree_hal_streaming_graph_node_t* node,
    const iree_hal_streaming_graph_edge_t* additional_edges,
    const uint32_t* node_index_map,
    const iree_hal_streaming_node_index_set_t* barrier_index_set) {
  for (uint32_t j = 0; j < node->dependency_count; ++j) {
    const uint32_t dependency_sort_index =
        node_index_map[node->dependencies[j]->node_index];
    if (dependency_sort_index == UINT32_MAX) {
      continue;
    }
    if (iree_hal_streaming_node_index_set_test_hazard(barrier_index_set,
                                                      dependency_sort_index)) {
      return true;
    }
  }

  for (const iree_hal_streaming_graph_edge_t* edge = additional_edges; edge;
       edge = edge->next) {
    if (edge->to != node) {
      continue;
    }
    const uint32_t dependency_sort_index =
        node_index_map[edge->from->node_index];
    if (dependency_sort_index == UINT32_MAX) {
      continue;
    }
    if (iree_hal_streaming_node_index_set_test_hazard(barrier_index_set,
                                                      dependency_sort_index)) {
      return true;
    }
  }

  return false;
}

static iree_status_t iree_hal_streaming_graph_record_dependency_barrier(
    iree_hal_command_buffer_t* command_buffer) {
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
  return iree_hal_command_buffer_execution_barrier(
      command_buffer,
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER |
          IREE_HAL_EXECUTION_STAGE_ATOMIC,
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER |
          IREE_HAL_EXECUTION_STAGE_ATOMIC,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 1, &memory_barrier, 0, NULL);
}

// Helper to record nodes from a partition into a command buffer.
static iree_status_t iree_hal_streaming_graph_record_partition(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_sort_node_t* sorted_nodes,
    uint32_t node_start_index, uint32_t node_count,
    const uint32_t* node_index_map, uint8_t stream_id,
    iree_hal_streaming_graph_edge_t* additional_edges,
    bool preserve_sorted_order, iree_hal_command_buffer_t* command_buffer) {
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

  // Record nodes assigned to this stream.
  //
  // HIP graph dependency edges define the minimum ordering. Graph memory nodes
  // are currently represented by graph-template allocations rather than
  // per-launch stream-ordered allocations, so executables containing them keep
  // sorted command order as a conservative lifetime boundary.
  //
  // We use a small linear scan set to make the test for hazards faster: we have
  // the original unsorted node indices of dependencies but not the sorted ones
  // we'd need to index into the sorted_nodes list and this avoids needing to
  // do that mapping.
  iree_status_t status = iree_ok_status();
  uint32_t in_stream_count = 0;
  iree_hal_streaming_node_index_set_t barrier_index_set;
  iree_hal_streaming_node_index_set_reset(&barrier_index_set);
  for (uint32_t i = 0; iree_status_is_ok(status) && i < node_count; ++i) {
    iree_hal_streaming_graph_sort_node_t* sort_node =
        &sorted_nodes[node_start_index + i];
    // Ignore nodes from other streams.
    if (sort_node->stream_id != stream_id) {
      continue;
    }
    iree_hal_streaming_graph_node_t* node = sort_node->node;
    if (in_stream_count > 0) {
      const bool has_dependency_hazard =
          iree_hal_streaming_graph_node_has_recorded_dependency_hazard(
              node, additional_edges, node_index_map, &barrier_index_set);
      if (preserve_sorted_order || has_dependency_hazard) {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0,
            iree_hal_streaming_graph_record_dependency_barrier(command_buffer));
        iree_hal_streaming_node_index_set_reset(&barrier_index_set);
      }
    }
    ++in_stream_count;
    switch (node->type) {
      case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL: {
        const iree_hal_streaming_graph_kernel_node_attrs_t* attrs =
            &node->attrs.kernel;
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
            &node->attrs.memcpy;
        status =
            iree_hal_streaming_graph_record_memcpy_node(command_buffer, attrs);
        break;
      }
      case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET: {
        const iree_hal_streaming_graph_memset_node_attrs_t* attrs =
            &node->attrs.memset;
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
            &node->attrs.batch_mem_op;
        status = iree_hal_streaming_command_buffer_append_value_operations(
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
    iree_hal_streaming_node_index_set_insert(&barrier_index_set,
                                             node_index_map[node->node_index]);
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

iree_status_t iree_hal_streaming_graph_exec_instantiate_from_template(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_exec_t* resource_owner) {
  IREE_ASSERT_ARGUMENT(exec);
  IREE_ASSERT_ARGUMENT(resource_owner);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_node_block_t* node_blocks =
      exec->template_graph->node_blocks;
  const iree_host_size_t node_count = exec->template_graph->node_count;
  exec->instantiated_node_count = node_count;
  exec->instantiated_visible_node_count =
      iree_hal_streaming_graph_visible_node_count(exec->graph);

  // Use the new scheduler to analyze and partition the graph.
  iree_hal_streaming_graph_schedule_t schedule;
  iree_hal_streaming_graph_edge_t* additional_edges =
      exec->template_graph->additional_edges;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_schedule_nodes(
              node_blocks, node_count, exec->node_disabled_states,
              exec->node_disabled_state_count, additional_edges,
              &exec->arena_allocator, &schedule));

  // Allocate block array.
  exec->block_count = schedule.block_count;
  if (schedule.partition_count == 0) {
    exec->block_count = 0;
    exec->blocks = NULL;
    exec->semaphore_count = 0;
    exec->semaphores = NULL;
    exec->semaphore_base_values = NULL;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&exec->arena_allocator,
                              exec->block_count * sizeof(*exec->blocks),
                              (void**)&exec->blocks));
  memset(exec->blocks, 0, exec->block_count * sizeof(*exec->blocks));

  // Calculate semaphore count needed.
  // We need semaphores at partition boundaries for synchronization.
  // Multi-stream partitions need join semaphores.
  //
  // This currently allocates semaphores per block boundary. A future
  // optimization can reduce this to the maximum layer size by advancing
  // timeline values between partitions.
  uint32_t semaphore_count = 0;
  if (schedule.partition_count > 1) {
    for (iree_host_size_t i = 0; i < schedule.partition_count - 1; i++) {
      if (schedule.partitions[i].stream_count > 1) {
        // Multi-stream partition needs one semaphore per stream for join.
        semaphore_count += schedule.partitions[i].stream_count;
      } else {
        // Single stream needs one semaphore.
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
    memset(exec->semaphores, 0,
           exec->semaphore_count * sizeof(*exec->semaphores));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(
                &exec->arena_allocator,
                exec->semaphore_count * sizeof(*exec->semaphore_base_values),
                (void**)&exec->semaphore_base_values));
    memset(exec->semaphore_base_values, 0,
           exec->semaphore_count * sizeof(*exec->semaphore_base_values));

    // Create the internal semaphores that carry values between partitions. The
    // resource set is their sole owner; exec->semaphores are borrowed pointers
    // and anything outliving the executable takes its own reference.
    iree_status_t status = iree_ok_status();
    for (uint32_t i = 0; i < exec->semaphore_count && iree_status_is_ok(status);
         i++) {
      iree_hal_semaphore_t* semaphore = NULL;
      status = iree_hal_semaphore_create(
          exec->context->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0ull,
          IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore);
      if (iree_status_is_ok(status)) {
        status =
            iree_hal_resource_set_insert(exec->resource_set, 1, &semaphore);
        iree_hal_semaphore_release(semaphore);
      }
      if (iree_status_is_ok(status)) {
        exec->semaphores[i] = semaphore;
      }
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
  }

  // Create blocks from partitions.
  uint32_t block_index = 0;
  uint32_t semaphore_index = 0;
  for (iree_host_size_t p = 0; p < schedule.partition_count; p++) {
    const iree_hal_streaming_graph_partition_t* partition =
        &schedule.partitions[p];

    // Determine wait semaphores for chaining FROM the previous partition.
    // The first partition waits on the original submission stream timeline
    // semaphores. Subsequent partitions wait on the previous partitions
    // semaphores, of which there may be several for a join operation.
    // Note that we don't allocate space for the initial semaphores as those are
    // part of the submission, not the exec object.
    uint16_t wait_semaphore_count = 0;
    if (p > 0) {
      iree_hal_streaming_graph_partition_t* prev_partition =
          &schedule.partitions[p - 1];
      wait_semaphore_count =
          prev_partition->stream_count > 1 ? prev_partition->stream_count : 1;
    }

    // Determine signal semaphores for chaining TO the next partition.
    // Each partition gets at least one signal semaphore while multi-stream
    // partitions get one per stream to allow the subsequent partition to join
    // them. The final partition signals the original submission stream timeline
    // semaphores. Note that we don't allocate space for the final semaphores as
    // those are part of the submission, not the exec object.
    uint16_t signal_semaphore_count = 0;
    if (p < schedule.partition_count - 1) {
      signal_semaphore_count =
          partition->stream_count > 1 ? partition->stream_count : 1;
    }

    if (partition->type == IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_RECORDABLE) {
      // If only one node is in the partition and it's recordable, we
      // may be able to route it to a dedicated partition type after this
      // function is split into smaller helpers.
      const uint8_t stream_count = partition->stream_count;
      const uint32_t partition_wait_semaphore_start =
          semaphore_index - wait_semaphore_count;
      const uint32_t partition_signal_semaphore_start = semaphore_index;
      for (uint8_t s = 0; s < stream_count; s++) {
        iree_hal_streaming_graph_block_t* block = NULL;

        // All streams in partition wait on same semaphores from previous.
        // Each stream in multi-stream partition signals its own semaphore.
        // Single stream signals all semaphores for the partition.
        // But if this is the last partition (signal_semaphore_count=0), don't
        // signal.
        uint16_t block_signal_count = 0;
        if (signal_semaphore_count > 0) {
          block_signal_count = (stream_count > 1) ? 1 : signal_semaphore_count;
        }
        iree_hal_streaming_graph_block_ptrs_t ptrs;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_streaming_graph_create_execute_block(
                    exec, partition->start_index, partition->count,
                    wait_semaphore_count, block_signal_count,
                    IREE_HAL_QUEUE_EXECUTE_FLAG_NONE, &block, &ptrs));

        // Record nodes for this stream into the command buffer.
        // For single stream (s=0), records all nodes.
        // For multi-stream, records nodes filtered by stream_id.
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_streaming_graph_record_partition(
                    exec, schedule.sorted_nodes, partition->start_index,
                    partition->count, schedule.node_index_map, s,
                    additional_edges, exec->uses_graph_memory_nodes,
                    ptrs.attrs->execute.command_buffer));

        // Set up semaphore indices.
        if (wait_semaphore_count > 0) {
          for (uint16_t w = 0; w < wait_semaphore_count; w++) {
            ptrs.wait_semaphore_indices[w] = partition_wait_semaphore_start + w;
            ptrs.wait_payload_deltas[w] = 1;
          }
        }
        if (block_signal_count > 0) {
          if (stream_count > 1) {
            // Multi-stream: each stream signals its own semaphore.
            ptrs.signal_semaphore_indices[0] =
                partition_signal_semaphore_start + s;
            ptrs.signal_payload_deltas[0] = 1;
          } else {
            // Single stream: signal all semaphores.
            for (uint16_t i = 0; i < block_signal_count; i++) {
              ptrs.signal_semaphore_indices[i] =
                  partition_signal_semaphore_start + i;
              ptrs.signal_payload_deltas[i] = 1;
            }
          }
        }

        exec->blocks[block_index++] = block;
      }

      // Advance semaphore index by the number of signal semaphores.
      semaphore_index += signal_semaphore_count;
    } else {
      // Set up semaphore indices.
      iree_hal_streaming_graph_block_t* block = NULL;
      iree_hal_streaming_graph_block_ptrs_t ptrs;
      if (partition->type ==
          IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_HOST_CALL) {
        // Host call gets its own block.
        iree_hal_streaming_graph_node_t* node =
            schedule.sorted_nodes[partition->start_index].node;
        const uint64_t args[4] = {(uint64_t)node->attrs.host.fn,
                                  (uint64_t)node->attrs.host.user_data, 0, 0};
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0,
            iree_hal_streaming_graph_create_host_call_block(
                exec, resource_owner, partition->start_index, partition->count,
                wait_semaphore_count, signal_semaphore_count,
                iree_hal_make_host_call_with_resource(
                    iree_hal_streaming_graph_host_callback, NULL,
                    &resource_owner->resource),
                args, node, IREE_HAL_HOST_CALL_FLAG_NONE, &block, &ptrs));
      } else if (partition->type ==
                 IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_DISPATCH) {
        iree_hal_streaming_graph_node_t* node =
            schedule.sorted_nodes[partition->start_index].node;
        const iree_hal_streaming_graph_kernel_node_attrs_t* attrs =
            &node->attrs.kernel;
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
            z0, iree_hal_streaming_graph_create_dispatch_block(
                    exec, partition->start_index, partition->count,
                    wait_semaphore_count, signal_semaphore_count,
                    attrs->symbol->executable, attrs->symbol->export_ordinal,
                    config, attrs->constants, attrs->bindings, flags, &block,
                    &ptrs));
      } else if (partition->type ==
                 IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_GRAPH) {
        iree_hal_streaming_graph_node_t* node =
            schedule.sorted_nodes[partition->start_index].node;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_streaming_graph_create_child_graph_block(
                    exec, partition->start_index, partition->count,
                    wait_semaphore_count, signal_semaphore_count,
                    node->attrs.child_graph.graph, &block, &ptrs));
      } else {
        iree_hal_streaming_graph_node_t* node =
            schedule.sorted_nodes[partition->start_index].node;
        if (node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC ||
            node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_FREE) {
          const bool is_allocation =
              node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEM_ALLOC;
          iree_hal_streaming_graph_memory_allocation_t* allocation =
              is_allocation ? node->attrs.mem_alloc.allocation
                            : node->attrs.mem_free.allocation;
          if (!allocation) {
            IREE_RETURN_AND_END_ZONE_IF_ERROR(
                z0,
                iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                 "graph memory node has no allocation record"));
          }
          const uint64_t args[4] = {0, 0, 0, 0};
          IREE_RETURN_AND_END_ZONE_IF_ERROR(
              z0,
              iree_hal_streaming_graph_create_host_call_block(
                  exec, resource_owner, partition->start_index,
                  partition->count, wait_semaphore_count,
                  signal_semaphore_count,
                  is_allocation
                      ? iree_hal_streaming_graph_memory_allocation_map_call(
                            allocation)
                      : iree_hal_streaming_graph_memory_allocation_unmap_call(
                            allocation),
                  args, /*source_node=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE,
                  &block, &ptrs));
        } else if (node->type ==
                       IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD ||
                   node->type ==
                       IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT) {
          const iree_hal_streaming_graph_block_type_t block_type =
              node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD
                  ? IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD
                  : IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_WAIT;
          IREE_RETURN_AND_END_ZONE_IF_ERROR(
              z0,
              iree_hal_streaming_graph_create_event_block(
                  exec, block_type, partition->start_index, partition->count,
                  wait_semaphore_count, signal_semaphore_count, node,
                  node->attrs.event.event, &block, &ptrs));
        } else if (node->type ==
                   IREE_HAL_STREAMING_GRAPH_NODE_TYPE_BATCH_MEM_OP) {
          IREE_RETURN_AND_END_ZONE_IF_ERROR(
              z0,
              iree_hal_streaming_graph_block_allocate(
                  &exec->arena_allocator,
                  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_BATCH_MEM_OP,
                  partition->start_index, partition->count,
                  wait_semaphore_count, signal_semaphore_count, &block, &ptrs));
        } else {
          // Empty/barrier partition.
          IREE_RETURN_AND_END_ZONE_IF_ERROR(
              z0, iree_hal_streaming_graph_create_barrier_block(
                      exec, partition->start_index, partition->count,
                      wait_semaphore_count, signal_semaphore_count,
                      IREE_HAL_QUEUE_BARRIER_FLAG_NONE, &block, &ptrs));
        }
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
      // Advance semaphore index by all signal semaphores.
      semaphore_index += signal_semaphore_count;
      exec->blocks[block_index++] = block;
    }
  }
  exec->block_count = block_index;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
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

// Signals from actually accepted graph blocks. A failed launch must resolve
// every one before releasing executable-owned command/callback state because
// no rejected block will signal the reserved final stream timeline value.
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

// Waits every accepted block to a successful signal or permanent semaphore
// failure. Infinite waits cannot time out; semaphore failures are terminal and
// still permit executable-owned resources to be released.
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
    uint64_t launch_stream_tail_value,
    iree_hal_semaphore_list_t external_wait_semaphores,
    iree_hal_semaphore_list_t external_signal_semaphores,
    iree_hal_streaming_dropped_graph_list_t* dropped_graphs,
    iree_hal_streaming_accepted_signal_list_t* accepted_signals);

static iree_status_t iree_hal_streaming_graph_host_callback(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_host_callback_t call_fn =
      (iree_hal_streaming_host_callback_t)args[0];
  void* call_user_data = (void*)args[1];
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, args[0]);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, args[1]);
  call_fn(call_user_data);
  IREE_TRACE_ZONE_END(z0);
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
    const iree_hal_streaming_graph_block_ptrs_t* ptrs,
    iree_hal_streaming_stream_t* stream, uint64_t launch_stream_tail_value,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores,
    iree_hal_streaming_recorded_point_t* record_point,
    iree_hal_streaming_dropped_graph_list_t* dropped_graphs,
    iree_hal_streaming_accepted_signal_list_t* accepted_signals) {
  if (block->type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH) {
    iree_hal_streaming_graph_exec_t* child_exec = ptrs->attrs->child_graph.exec;
    // Only the child's block 0 carries this block's waits, which are
    // themselves behind the launch tail; a record inside the child sits in a
    // single-block partition that either is block 0 or chains back to it, so
    // the tail carries down unchanged. The shared accepted list includes every
    // actual child submission if a later child block rejects.
    return iree_hal_streaming_graph_exec_submit_blocks_locked(
        child_exec, stream, launch_stream_tail_value, wait_semaphores,
        signal_semaphores, dropped_graphs, accepted_signals);
  }
  if (IREE_UNLIKELY(signal_semaphores.count == 0)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "graph block signals no timeline value");
  }

  // Claim storage before the queue call so an accepted operation always has a
  // terminal signal the launch can drain if a later block rejects.
  iree_hal_streaming_accepted_signal_t* accepted_signal = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_accepted_signal_list_push_empty(
      accepted_signals, &accepted_signal));

  iree_status_t status = iree_ok_status();
  switch (block->type) {
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD:
      status = iree_hal_streaming_event_enqueue_record(
          ptrs->attrs->event.event, stream->context, stream->queue,
          wait_semaphores, signal_semaphores, record_point);
      break;
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_WAIT:
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER: {
      const iree_hal_queue_barrier_flags_t flags =
          block->type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER
              ? ptrs->attrs->barrier.flags
              : IREE_HAL_QUEUE_BARRIER_FLAG_NONE;
      status = iree_hal_queue_barrier(stream->queue, wait_semaphores,
                                      signal_semaphores, flags);
      break;
    }
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_FILL:
      status = iree_hal_queue_fill(
          stream->queue, wait_semaphores, signal_semaphores,
          ptrs->attrs->fill.target_buffer, ptrs->attrs->fill.target_offset,
          ptrs->attrs->fill.length, &ptrs->attrs->fill.pattern,
          ptrs->attrs->fill.pattern_length, ptrs->attrs->fill.flags);
      break;
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_COPY:
      status = iree_hal_queue_copy(
          stream->queue, wait_semaphores, signal_semaphores,
          ptrs->attrs->copy.source_buffer, ptrs->attrs->copy.source_offset,
          ptrs->attrs->copy.target_buffer, ptrs->attrs->copy.target_offset,
          ptrs->attrs->copy.length, ptrs->attrs->copy.flags);
      break;
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH: {
      iree_hal_queue_t* dispatch_queue = stream->queue;
      if (iree_any_bit_set(ptrs->attrs->dispatch.flags,
                           IREE_HAL_DISPATCH_FLAG_COOPERATIVE)) {
        status = iree_hal_streaming_stream_select_cooperative_queue_locked(
            stream, &dispatch_queue);
      }
      if (iree_status_is_ok(status)) {
        iree_hal_buffer_ref_list_t bindings_list = {
            .count = ptrs->attrs->dispatch.bindings.count,
            .values = ptrs->attrs->dispatch.bindings.values,
        };
        status = iree_hal_queue_dispatch(
            dispatch_queue, wait_semaphores, signal_semaphores,
            ptrs->attrs->dispatch.executable,
            iree_hal_executable_function_from_index(
                (uint32_t)ptrs->attrs->dispatch.entry_point),
            ptrs->attrs->dispatch.config, ptrs->attrs->dispatch.constants,
            bindings_list, ptrs->attrs->dispatch.flags);
      }
      break;
    }
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE:
      status = iree_hal_queue_execute(stream->queue, wait_semaphores,
                                      signal_semaphores,
                                      ptrs->attrs->execute.command_buffer,
                                      iree_hal_buffer_binding_table_empty(),
                                      IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
      break;
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL:
      status = iree_hal_queue_host_call(
          stream->queue, wait_semaphores, signal_semaphores,
          ptrs->attrs->host_call.call, ptrs->attrs->host_call.args,
          ptrs->attrs->host_call.flags);
      break;
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_BATCH_MEM_OP:
      status = iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "batch memory operations are not executable in graphs");
      break;
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
// launch waits behind, or 0 when the stream had never submitted. Only block 0
// carries the launch's external waits, and the extra workstream blocks of the
// first partition wait on nothing, so a block is behind the tail only when it
// chains back to block 0. An event record node is never recordable and so gets
// a partition of its own holding a single block, which either is block 0 or
// waits on every signal of the partition ahead of it, back to block 0.
static iree_status_t iree_hal_streaming_graph_exec_submit_blocks_locked(
    iree_hal_streaming_graph_exec_t* exec, iree_hal_streaming_stream_t* stream,
    uint64_t launch_stream_tail_value,
    iree_hal_semaphore_list_t external_wait_semaphores,
    iree_hal_semaphore_list_t external_signal_semaphores,
    iree_hal_streaming_dropped_graph_list_t* dropped_graphs,
    iree_hal_streaming_accepted_signal_list_t* accepted_signals) {
  enum {
    IREE_HAL_STREAMING_GRAPH_STACK_BASE_VALUE_COUNT = 64,
    IREE_HAL_STREAMING_GRAPH_STACK_SEMAPHORE_COUNT = 16,
  };

  for (uint32_t i = 0; i < exec->semaphore_count; i++) {
    uint64_t current_value = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_semaphore_query(exec->semaphores[i], &current_value));
    exec->semaphore_base_values[i] =
        iree_max(exec->semaphore_base_values[i], current_value);
  }

  if (exec->block_count == 0) {
    if (external_wait_semaphores.count == 0 &&
        external_signal_semaphores.count == 0) {
      return iree_ok_status();
    }
    if (IREE_UNLIKELY(external_signal_semaphores.count == 0)) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "empty child graph signals no timeline value");
    }
    iree_hal_streaming_accepted_signal_t* accepted_signal = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_accepted_signal_list_push_empty(
        accepted_signals, &accepted_signal));
    iree_status_t status = iree_hal_queue_barrier(
        stream->queue, external_wait_semaphores, external_signal_semaphores,
        IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
    if (iree_status_is_ok(status)) {
      accepted_signal->semaphore = external_signal_semaphores.semaphores[0];
      accepted_signal->value = external_signal_semaphores.payload_values[0];
    }
    return status;
  }

  iree_status_t status = iree_ok_status();
  uint64_t stack_base_values[IREE_HAL_STREAMING_GRAPH_STACK_BASE_VALUE_COUNT];
  uint64_t* new_base_values = NULL;
  bool free_base_values = false;
  if (exec->semaphore_count > 0) {
    iree_host_size_t base_values_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            exec->semaphore_count, sizeof(uint64_t), &base_values_size))) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "graph semaphore base value size overflow");
    }
    if (exec->semaphore_count <=
        IREE_HAL_STREAMING_GRAPH_STACK_BASE_VALUE_COUNT) {
      new_base_values = stack_base_values;
    } else {
      IREE_RETURN_IF_ERROR(iree_allocator_malloc(
          exec->host_allocator, base_values_size, (void**)&new_base_values));
      free_base_values = true;
    }
    memcpy(new_base_values, exec->semaphore_base_values, base_values_size);
  }

  for (uint32_t block_index = 0;
       iree_status_is_ok(status) && block_index < exec->block_count;
       block_index++) {
    iree_hal_streaming_graph_block_t* block = exec->blocks[block_index];

    iree_hal_streaming_graph_block_ptrs_t ptrs;
    iree_hal_streaming_graph_block_get_ptrs(block, &ptrs);

    const bool block_waits_event =
        block->type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_WAIT;
    const bool block_records_event =
        block->type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD;
    const iree_host_size_t total_wait_count =
        block->wait_semaphore_count +
        (block_index == 0 ? external_wait_semaphores.count : 0) +
        (block_waits_event ? 1 : 0);
    const iree_host_size_t total_signal_count =
        block->signal_semaphore_count + (block_index == exec->block_count - 1
                                             ? external_signal_semaphores.count
                                             : 0);
    iree_host_size_t total_semaphores = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            total_wait_count, total_signal_count, &total_semaphores))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "graph launch semaphore count overflow");
      break;
    }

    iree_hal_semaphore_t*
        stack_semaphore_array[IREE_HAL_STREAMING_GRAPH_STACK_SEMAPHORE_COUNT];
    uint64_t stack_value_array[IREE_HAL_STREAMING_GRAPH_STACK_SEMAPHORE_COUNT];
    iree_hal_semaphore_t** semaphore_array = NULL;
    uint64_t* value_array = NULL;
    bool free_semaphore_array = false;
    bool free_value_array = false;
    if (total_semaphores > 0) {
      if (total_semaphores <= IREE_HAL_STREAMING_GRAPH_STACK_SEMAPHORE_COUNT) {
        semaphore_array = stack_semaphore_array;
        value_array = stack_value_array;
      } else {
        iree_host_size_t semaphore_array_size = 0;
        iree_host_size_t value_array_size = 0;
        if (IREE_UNLIKELY(
                !iree_host_size_checked_mul(total_semaphores,
                                            sizeof(iree_hal_semaphore_t*),
                                            &semaphore_array_size) ||
                !iree_host_size_checked_mul(total_semaphores, sizeof(uint64_t),
                                            &value_array_size))) {
          status = iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "graph launch semaphore list allocation size overflow");
          break;
        }
        status =
            iree_allocator_malloc(exec->host_allocator, semaphore_array_size,
                                  (void**)&semaphore_array);
        if (iree_status_is_ok(status)) {
          free_semaphore_array = true;
          status = iree_allocator_malloc(exec->host_allocator, value_array_size,
                                         (void**)&value_array);
          free_value_array = iree_status_is_ok(status);
        }
        if (!iree_status_is_ok(status)) {
          iree_allocator_free(exec->host_allocator, semaphore_array);
          break;
        }
      }
    }

    iree_hal_semaphore_t** wait_sems = semaphore_array;
    uint64_t* wait_vals = value_array;
    iree_hal_semaphore_t** signal_sems =
        semaphore_array ? semaphore_array + total_wait_count : NULL;
    uint64_t* signal_vals = value_array ? value_array + total_wait_count : NULL;

    iree_host_size_t wait_count = 0;
    if (block_index == 0) {
      for (iree_host_size_t i = 0; i < external_wait_semaphores.count; ++i) {
        wait_sems[wait_count] = external_wait_semaphores.semaphores[i];
        wait_vals[wait_count] = external_wait_semaphores.payload_values[i];
        ++wait_count;
      }
    }
    for (uint16_t i = 0; i < block->wait_semaphore_count; i++) {
      const uint16_t semaphore_index = ptrs.wait_semaphore_indices[i];
      const uint32_t delta = ptrs.wait_payload_deltas[i];
      wait_sems[wait_count] = exec->semaphores[semaphore_index];
      wait_vals[wait_count] =
          exec->semaphore_base_values[semaphore_index] + delta;
      ++wait_count;
    }
    // Retained across the submission so a concurrent record of the same event
    // cannot drop the last reference to the timeline this block names. Unlike
    // iree_hal_streaming_stream_wait_event, the point's stream ordering is not
    // filed in |stream|'s memory reuse ledger; a missing entry only withholds
    // allocations from reuse, so the omission costs reuse and can never grant
    // it.
    iree_hal_streaming_recorded_point_t event_wait_point = {0};
    if (block_waits_event) {
      iree_hal_streaming_event_acquire_recorded_point(ptrs.attrs->event.event,
                                                      &event_wait_point);
      // A wait on an event with no submitted record has nothing to wait for.
      if (event_wait_point.semaphore) {
        wait_sems[wait_count] = event_wait_point.semaphore;
        wait_vals[wait_count] = event_wait_point.value;
        ++wait_count;
      }
    }

    iree_host_size_t signal_count = 0;
    for (uint16_t i = 0; i < block->signal_semaphore_count; i++) {
      const uint16_t semaphore_index = ptrs.signal_semaphore_indices[i];
      const uint32_t delta = ptrs.signal_payload_deltas[i];
      signal_sems[signal_count] = exec->semaphores[semaphore_index];
      signal_vals[signal_count] =
          exec->semaphore_base_values[semaphore_index] + delta;
      ++signal_count;
    }
    if (block_index == exec->block_count - 1) {
      for (iree_host_size_t i = 0; i < external_signal_semaphores.count; ++i) {
        signal_sems[signal_count] = external_signal_semaphores.semaphores[i];
        signal_vals[signal_count] =
            external_signal_semaphores.payload_values[i];
        ++signal_count;
      }
    }

    // A record node marks the point its own block reaches, which is the block's
    // first signal value: the partitioner gives a record node a partition of
    // its own, so that value is signaled exactly when the node's dependencies
    // complete. Every block signals either an internal semaphore or the
    // launch's external signals, so a record block always has a value to name.
    if (block_records_event && IREE_UNLIKELY(signal_count == 0)) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "event record block signals no timeline value");
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
    if (iree_status_is_ok(status)) {
      iree_hal_streaming_recorded_point_t record_point = {0};
      iree_hal_streaming_graph_t** dropped_capture_graph = NULL;
      if (block_records_event) {
        // The block's first signal is the point the record names. When that
        // signal is the launching stream's own timeline the point is exactly a
        // stream point; otherwise it is internal to the launch and all that is
        // known about the stream is that the launch waited behind its tail.
        const bool signals_launch_stream_timeline =
            signal_sems[0] == stream->timeline_semaphore;
        record_point = (iree_hal_streaming_recorded_point_t){
            .semaphore = signal_sems[0],
            .value = signal_vals[0],
            .ordered_after_stream_id = stream->stream_id,
            .ordered_after_stream_value = signals_launch_stream_timeline
                                              ? signal_vals[0]
                                              : launch_stream_tail_value,
        };
        // Room for the capture graph reference the record ends is claimed
        // before the record is enqueued, so the commit below always has
        // somewhere to put it. Nothing has been dropped yet, so a failure here
        // loses nothing.
        status = iree_hal_streaming_dropped_graph_list_push_empty(
            dropped_graphs, &dropped_capture_graph);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_streaming_graph_submit_block(
            block, &ptrs, stream, launch_stream_tail_value, wait_semaphores,
            signal_semaphores, &record_point, dropped_graphs, accepted_signals);
        if (iree_status_is_ok(status)) {
          for (uint16_t i = 0; i < block->signal_semaphore_count; ++i) {
            const uint16_t semaphore_index = ptrs.signal_semaphore_indices[i];
            new_base_values[semaphore_index] = signal_vals[i];
          }
          // A rejected record keeps its old point. This commit stays inside
          // the iteration because later blocks may wait on the new point.
          if (block_records_event) {
            *dropped_capture_graph =
                iree_hal_streaming_event_commit_recorded_point(
                    ptrs.attrs->event.event, record_point);
          }
        }
      }
    }
    iree_hal_streaming_event_release_recorded_point(&event_wait_point);
    if (free_value_array) {
      iree_allocator_free(exec->host_allocator, value_array);
    }
    if (free_semaphore_array) {
      iree_allocator_free(exec->host_allocator, semaphore_array);
    }
  }

  // Commit only values belonging to accepted blocks. Accepted prefix signals
  // must never be reused after a later rejection, while a rejected block's
  // unsignaled values remain available to the next launch.
  if (exec->semaphore_count > 0) {
    memcpy(exec->semaphore_base_values, new_base_values,
           exec->semaphore_count * sizeof(uint64_t));
  }
  if (free_base_values) {
    iree_allocator_free(exec->host_allocator, new_base_values);
  }
  return status;
}

iree_status_t iree_hal_streaming_graph_exec_launch(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(exec);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Flush stream to ensure all prior operations are submitted.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_streaming_stream_flush(stream));

  // Mutex needed for launch as multiple threads can submit at once.
  // It also protects the executable semaphore base values below, which
  // are reused and advanced across launches of the same executable graph.
  iree_slim_mutex_lock(&exec->mutex);
  if (exec->is_destroyed) {
    iree_slim_mutex_unlock(&exec->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  // Handle empty graph - nothing to do.
  if (exec->block_count == 0) {
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
    iree_slim_mutex_unlock(&exec->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INCOMPATIBLE,
        "an event can only be recorded on a stream of the context that "
        "created it");
  }

  // Graph allocations reuse stable virtual addresses. A later launch of the
  // same executable cannot remap those addresses until prior work completes.
  while (exec->uses_graph_memory_nodes &&
         exec->graph_memory_active_launch_stream) {
    iree_hal_streaming_stream_t* active_stream =
        exec->graph_memory_active_launch_stream;
    const uint64_t active_value = exec->graph_memory_active_launch_value;
    iree_hal_streaming_stream_retain(active_stream);
    iree_slim_mutex_unlock(&exec->mutex);

    iree_status_t wait_status = iree_hal_semaphore_wait(
        active_stream->timeline_semaphore, active_value,
        iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);

    iree_slim_mutex_lock(&exec->mutex);
    if (exec->graph_memory_active_launch_stream == active_stream &&
        exec->graph_memory_active_launch_value == active_value) {
      iree_hal_streaming_stream_release(
          exec->graph_memory_active_launch_stream);
      exec->graph_memory_active_launch_stream = NULL;
      exec->graph_memory_active_launch_value = 0;
    }
    iree_hal_streaming_stream_release(active_stream);
    if (!iree_status_is_ok(wait_status)) {
      iree_slim_mutex_unlock(&exec->mutex);
      IREE_TRACE_ZONE_END(z0);
      return wait_status;
    }
    if (exec->is_destroyed) {
      iree_slim_mutex_unlock(&exec->mutex);
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
    }
  }

  if (exec->has_graph_alloc_nodes_without_free_nodes &&
      exec->launch_count > 0 &&
      iree_hal_streaming_graph_memory_has_live_unfreed_allocations(exec)) {
    if (!iree_all_bits_set(
            exec->flags,
            IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)) {
      iree_slim_mutex_unlock(&exec->mutex);
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "graph contains live allocation nodes from a previous launch");
    }
    iree_status_t auto_free_status =
        iree_hal_streaming_graph_memory_auto_free(exec);
    if (!iree_status_is_ok(auto_free_status)) {
      iree_slim_mutex_unlock(&exec->mutex);
      IREE_TRACE_ZONE_END(z0);
      return auto_free_status;
    }
  }

  iree_hal_streaming_graph_memory_allocation_t** graph_memory_publications =
      NULL;
  iree_host_size_t graph_memory_publication_count = 0;
  iree_status_t status =
      iree_hal_streaming_graph_memory_prepare_launch_pointers(
          exec, &graph_memory_publications, &graph_memory_publication_count);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&exec->mutex);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_hal_streaming_graph_launch_point_t* launch_point = NULL;
  bool launch_point_inserted = false;
  status = iree_hal_streaming_graph_exec_prepare_launch_point_locked(
      exec, stream, &launch_point, &launch_point_inserted);
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(
        status, iree_hal_streaming_graph_memory_rollback_launch_pointers(
                    graph_memory_publication_count, graph_memory_publications));
    iree_allocator_free(exec->host_allocator, graph_memory_publications);
    iree_slim_mutex_unlock(&exec->mutex);
    IREE_TRACE_ZONE_END(z0);
    return status;
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

  iree_slim_mutex_lock(&stream->mutex);

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

  iree_hal_semaphore_t* wait_semaphore = stream->timeline_semaphore;
  uint64_t wait_payload_value = stream_wait_value;
  iree_hal_semaphore_list_t wait_semaphores = {
      .count = stream_wait_value > 0 ? 1 : 0,
      .semaphores = &wait_semaphore,
      .payload_values = &wait_payload_value,
  };
  iree_hal_semaphore_t* signal_semaphore = stream->timeline_semaphore;
  uint64_t signal_payload_value = stream_signal_value;
  iree_hal_semaphore_list_t signal_semaphores = {
      .count = 1,
      .semaphores = &signal_semaphore,
      .payload_values = &signal_payload_value,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_exec_submit_blocks_locked(
        exec, stream, stream_wait_value, wait_semaphores, signal_semaphores,
        &dropped_graphs, &accepted_signals);
  }

  if (!iree_status_is_ok(status)) {
    // No rejected block can signal |stream_signal_value|, so do not publish a
    // phantom stream tail. Keep both locks held while resolving every actual
    // accepted signal, preventing teardown or later same-stream work from
    // overtaking the prefix. Join keeps the submission failure primary.
    iree_status_t drain_status =
        iree_hal_streaming_accepted_signal_list_drain(&accepted_signals);
    status = iree_status_join(status, drain_status);
  }

  if (iree_status_is_ok(status)) {
    stream->pending_value = stream_signal_value;
    launch_point->value = stream_signal_value;
    if (exec->uses_graph_memory_nodes) {
      iree_hal_streaming_stream_release(
          exec->graph_memory_active_launch_stream);
      iree_hal_streaming_stream_retain(stream);
      exec->graph_memory_active_launch_stream = stream;
      exec->graph_memory_active_launch_value = stream_signal_value;
    }
    ++exec->launch_count;
  } else if (launch_point_inserted) {
    iree_hal_streaming_graph_exec_rollback_launch_point_locked(exec,
                                                               launch_point);
  }

  iree_slim_mutex_unlock(&stream->mutex);
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(
        status, iree_hal_streaming_graph_memory_rollback_launch_pointers(
                    graph_memory_publication_count, graph_memory_publications));
  }
  iree_allocator_free(exec->host_allocator, graph_memory_publications);
  iree_slim_mutex_unlock(&exec->mutex);
  // The last reference to a captured graph frees the allocations it owns, which
  // synchronizes every context and relocks this stream, so the references the
  // launch's records ended are dropped here. The records that ended them were
  // enqueued whatever a later block did, so this runs on both paths.
  iree_hal_streaming_dropped_graph_list_deinitialize(&dropped_graphs);
  iree_hal_streaming_accepted_signal_list_deinitialize(&accepted_signals);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static bool iree_hal_streaming_graph_node_is_visible(
    const iree_hal_streaming_graph_node_t* node) {
  return node && (node->flags & IREE_HAL_STREAMING_GRAPH_NODE_FLAG_HIDDEN) == 0;
}

static iree_host_size_t iree_hal_streaming_graph_visible_node_count(
    const iree_hal_streaming_graph_t* graph) {
  iree_host_size_t visible_count = 0;
  if (!graph) {
    return 0;
  }
  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      if (iree_hal_streaming_graph_node_is_visible(block->nodes[i])) {
        ++visible_count;
      }
    }
  }
  return visible_count;
}

static iree_hal_streaming_graph_node_t*
iree_hal_streaming_graph_visible_node_at_index(
    const iree_hal_streaming_graph_t* graph, iree_host_size_t visible_index) {
  iree_host_size_t current_visible_index = 0;
  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (!iree_hal_streaming_graph_node_is_visible(node)) {
        continue;
      }
      if (current_visible_index == visible_index) {
        return node;
      }
      ++current_visible_index;
    }
  }
  return NULL;
}

static bool iree_hal_streaming_graph_visible_index_of_node(
    const iree_hal_streaming_graph_t* graph,
    const iree_hal_streaming_graph_node_t* node,
    iree_host_size_t* out_visible_index) {
  iree_host_size_t current_visible_index = 0;
  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* candidate = block->nodes[i];
      if (!iree_hal_streaming_graph_node_is_visible(candidate)) {
        continue;
      }
      if (candidate == node) {
        *out_visible_index = current_visible_index;
        return true;
      }
      ++current_visible_index;
    }
  }
  return false;
}

static iree_host_size_t iree_hal_streaming_graph_visible_dependency_count(
    const iree_hal_streaming_graph_t* graph,
    const iree_hal_streaming_graph_node_t* node) {
  iree_host_size_t dependency_count = 0;
  for (uint32_t i = 0; i < node->dependency_count; ++i) {
    if (iree_hal_streaming_graph_node_is_visible(node->dependencies[i])) {
      ++dependency_count;
    }
  }
  for (iree_hal_streaming_graph_edge_t* edge = graph->additional_edges; edge;
       edge = edge->next) {
    if (edge->to == node &&
        iree_hal_streaming_graph_node_is_visible(edge->from)) {
      ++dependency_count;
    }
  }
  return dependency_count;
}

static bool iree_hal_streaming_graph_has_visible_dependency(
    const iree_hal_streaming_graph_t* graph,
    const iree_hal_streaming_graph_node_t* node,
    iree_host_size_t dependency_visible_index) {
  for (uint32_t i = 0; i < node->dependency_count; ++i) {
    iree_host_size_t visible_index = 0;
    if (iree_hal_streaming_graph_visible_index_of_node(
            graph, node->dependencies[i], &visible_index) &&
        visible_index == dependency_visible_index) {
      return true;
    }
  }
  for (iree_hal_streaming_graph_edge_t* edge = graph->additional_edges; edge;
       edge = edge->next) {
    iree_host_size_t visible_index = 0;
    if (edge->to == node &&
        iree_hal_streaming_graph_visible_index_of_node(graph, edge->from,
                                                       &visible_index) &&
        visible_index == dependency_visible_index) {
      return true;
    }
  }
  return false;
}

static bool iree_hal_streaming_graph_visible_dependencies_match(
    const iree_hal_streaming_graph_t* old_graph,
    const iree_hal_streaming_graph_node_t* old_node,
    const iree_hal_streaming_graph_t* new_graph,
    const iree_hal_streaming_graph_node_t* new_node) {
  if (iree_hal_streaming_graph_visible_dependency_count(old_graph, old_node) !=
      iree_hal_streaming_graph_visible_dependency_count(new_graph, new_node)) {
    return false;
  }
  for (uint32_t i = 0; i < old_node->dependency_count; ++i) {
    iree_host_size_t old_dependency_visible_index = 0;
    if (!iree_hal_streaming_graph_visible_index_of_node(
            old_graph, old_node->dependencies[i],
            &old_dependency_visible_index)) {
      continue;
    }
    if (!iree_hal_streaming_graph_has_visible_dependency(
            new_graph, new_node, old_dependency_visible_index)) {
      return false;
    }
  }
  for (iree_hal_streaming_graph_edge_t* edge = old_graph->additional_edges;
       edge; edge = edge->next) {
    if (edge->to != old_node) {
      continue;
    }
    iree_host_size_t old_dependency_visible_index = 0;
    if (!iree_hal_streaming_graph_visible_index_of_node(
            old_graph, edge->from, &old_dependency_visible_index)) {
      continue;
    }
    if (!iree_hal_streaming_graph_has_visible_dependency(
            new_graph, new_node, old_dependency_visible_index)) {
      return false;
    }
  }
  return true;
}

static iree_hal_streaming_graph_node_t*
iree_hal_streaming_graph_first_visible_kernel_node(
    const iree_hal_streaming_graph_t* graph) {
  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      if (iree_hal_streaming_graph_node_is_visible(node) &&
          node->type == IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL) {
        return node;
      }
    }
  }
  return iree_hal_streaming_graph_visible_node_at_index(graph, 0);
}

static bool iree_hal_streaming_graph_memcpy_update_is_compatible(
    const iree_hal_streaming_graph_memcpy_node_attrs_t* old_attrs,
    const iree_hal_streaming_graph_memcpy_node_attrs_t* new_attrs) {
  return old_attrs->hip_kind == new_attrs->hip_kind &&
         old_attrs->size == new_attrs->size &&
         old_attrs->execution_extent_width ==
             new_attrs->execution_extent_width &&
         old_attrs->execution_extent_height ==
             new_attrs->execution_extent_height &&
         old_attrs->execution_extent_depth ==
             new_attrs->execution_extent_depth &&
         old_attrs->execution_dst_pitch == new_attrs->execution_dst_pitch &&
         old_attrs->execution_src_pitch == new_attrs->execution_src_pitch &&
         old_attrs->execution_dst_ysize == new_attrs->execution_dst_ysize &&
         old_attrs->execution_src_ysize == new_attrs->execution_src_ysize &&
         old_attrs->hip_extent_width == new_attrs->hip_extent_width &&
         old_attrs->hip_extent_height == new_attrs->hip_extent_height &&
         old_attrs->hip_extent_depth == new_attrs->hip_extent_depth &&
         old_attrs->hip_dst_pitch == new_attrs->hip_dst_pitch &&
         old_attrs->hip_src_pitch == new_attrs->hip_src_pitch &&
         old_attrs->hip_dst_xsize == new_attrs->hip_dst_xsize &&
         old_attrs->hip_src_xsize == new_attrs->hip_src_xsize &&
         old_attrs->hip_dst_ysize == new_attrs->hip_dst_ysize &&
         old_attrs->hip_src_ysize == new_attrs->hip_src_ysize;
}

static bool iree_hal_streaming_graph_memset_update_is_compatible(
    const iree_hal_streaming_graph_memset_node_attrs_t* old_attrs,
    const iree_hal_streaming_graph_memset_node_attrs_t* new_attrs) {
  return old_attrs->pattern_size == new_attrs->pattern_size &&
         old_attrs->count == new_attrs->count &&
         old_attrs->hip_width == new_attrs->hip_width &&
         old_attrs->hip_height == new_attrs->hip_height &&
         old_attrs->hip_pitch == new_attrs->hip_pitch;
}

static bool iree_hal_streaming_graph_update_is_compatible(
    const iree_hal_streaming_graph_t* old_graph,
    iree_host_size_t old_visible_node_count,
    const iree_hal_streaming_graph_t* new_graph,
    iree_hal_streaming_graph_node_t** out_error_node,
    iree_hal_streaming_graph_exec_update_result_t* out_result) {
  if (old_graph->context != new_graph->context) {
    *out_error_node =
        iree_hal_streaming_graph_first_visible_kernel_node(new_graph);
    *out_result =
        IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_UNSUPPORTED_FUNCTION_CHANGE;
    return false;
  }
  const iree_host_size_t new_visible_node_count =
      iree_hal_streaming_graph_visible_node_count(new_graph);
  if (old_visible_node_count != new_visible_node_count) {
    *out_error_node = NULL;
    *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_TOPOLOGY_CHANGED;
    return false;
  }

  for (iree_host_size_t i = 0; i < old_visible_node_count; ++i) {
    iree_hal_streaming_graph_node_t* old_node =
        iree_hal_streaming_graph_visible_node_at_index(old_graph, i);
    iree_hal_streaming_graph_node_t* new_node =
        iree_hal_streaming_graph_visible_node_at_index(new_graph, i);
    if (!old_node || !new_node) {
      *out_error_node = new_node;
      *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_TOPOLOGY_CHANGED;
      return false;
    }
    if (old_node->type != new_node->type) {
      *out_error_node = new_node;
      *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_NODE_TYPE_CHANGED;
      return false;
    }
    if (!iree_hal_streaming_graph_visible_dependencies_match(
            old_graph, old_node, new_graph, new_node)) {
      *out_error_node = new_node;
      *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_TOPOLOGY_CHANGED;
      return false;
    }

    switch (old_node->type) {
      case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY:
        if (!iree_hal_streaming_graph_memcpy_update_is_compatible(
                &old_node->attrs.memcpy, &new_node->attrs.memcpy)) {
          *out_error_node = new_node;
          *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_PARAMETERS_CHANGED;
          return false;
        }
        break;
      case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET:
        if (!iree_hal_streaming_graph_memset_update_is_compatible(
                &old_node->attrs.memset, &new_node->attrs.memset)) {
          *out_error_node = new_node;
          *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_PARAMETERS_CHANGED;
          return false;
        }
        break;
      case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH:
        if (!old_node->attrs.child_graph.graph ||
            !new_node->attrs.child_graph.graph ||
            !iree_hal_streaming_graph_update_is_compatible(
                old_node->attrs.child_graph.graph,
                iree_hal_streaming_graph_visible_node_count(
                    old_node->attrs.child_graph.graph),
                new_node->attrs.child_graph.graph, out_error_node,
                out_result)) {
          *out_error_node = new_node;
          return false;
        }
        break;
      default:
        break;
    }
  }
  return true;
}

static void iree_hal_streaming_graph_exec_set_graph_locked(
    iree_hal_streaming_graph_exec_t* exec, iree_hal_streaming_graph_t* graph) {
  exec->graph = graph;
  exec->uses_graph_memory_nodes = graph && graph->has_graph_memory_nodes;
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
  if (!iree_hal_streaming_graph_update_is_compatible(
          exec->graph, exec->instantiated_visible_node_count, graph,
          out_error_node, out_result)) {
    iree_slim_mutex_unlock(&exec->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "graph update is not compatible");
  }

  iree_hal_streaming_graph_t* new_template_graph = NULL;
  iree_status_t status = iree_hal_streaming_graph_clone_for_exec(
      graph, exec->context, &new_template_graph);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&exec->mutex);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_hal_streaming_graph_t* old_graph = exec->graph;
  iree_hal_streaming_graph_t* old_template_graph = exec->template_graph;
  const bool graph_changed = graph != old_graph;
  const bool candidate_claimed = graph_changed && graph->has_graph_memory_nodes;
  if (candidate_claimed &&
      !iree_hal_streaming_graph_memory_exec_try_claim(graph)) {
    iree_hal_streaming_graph_release(new_template_graph);
    iree_slim_mutex_unlock(&exec->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "graphs with memory allocation nodes support one live executable");
  }
  if (graph_changed) {
    iree_hal_streaming_graph_retain(graph);
    iree_hal_streaming_graph_exec_set_graph_locked(exec, graph);
  }
  exec->template_graph = new_template_graph;

  status = iree_hal_streaming_graph_exec_rebuild_from_template_locked(exec);
  if (iree_status_is_ok(status)) {
    *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_SUCCESS;
    iree_hal_streaming_graph_release(old_template_graph);
    if (graph_changed) {
      if (old_graph->has_graph_memory_nodes) {
        iree_hal_streaming_graph_memory_exec_release(old_graph);
      }
      iree_hal_streaming_graph_release(old_graph);
    }
  } else {
    exec->template_graph = old_template_graph;
    iree_hal_streaming_graph_release(new_template_graph);
    if (graph_changed) {
      iree_hal_streaming_graph_exec_set_graph_locked(exec, old_graph);
      if (candidate_claimed) {
        iree_hal_streaming_graph_memory_exec_release(graph);
      }
      iree_hal_streaming_graph_release(graph);
    }
  }

  iree_slim_mutex_unlock(&exec->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
