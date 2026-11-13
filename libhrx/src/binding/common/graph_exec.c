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

// Original handles remain stable even when template node indices are compacted.
// Entries follow source creation order, so stable source indices are sorted.
typedef struct iree_hal_streaming_graph_node_mapping_t {
  // Borrowed original handle, protected by source_graph.
  iree_hal_streaming_graph_node_t* source;
  // Borrowed executable-owned parameters, protected by graph.
  iree_hal_streaming_graph_node_t* target;
} iree_hal_streaming_graph_node_mapping_t;

typedef struct iree_hal_streaming_graph_exec_t {
  // Retained executable handles.
  iree_atomic_ref_count_t ref_count;
  // Allocator for executable and metadata storage.
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
  // Handle invalidation writes hold edit_mutex then launch_mutex; either
  // actor can read under its own lock.
  bool is_destroyed;

  // Installed compiled state, protected by launch_mutex.
  iree_hal_streaming_graph_recording_t recording;

  // Per-instantiated-node disabled state keyed by graph node index.
  uint8_t* node_disabled_states;
  // Number of entries in |node_disabled_states|.
  iree_host_size_t node_disabled_state_count;

  // Reserved once per invocation, including a partially rejected launch.
  uint64_t launch_generation;

  // Stable logical root identity shared by children and rebuild candidates.
  // Borrowed by children, which are owned by the root's compiled blocks.
  iree_hal_streaming_graph_exec_t* execution_owner;

  // Completion of the last accepted launch, preserved through parameter edits.
  struct {
    // Retained stream timeline; NULL before the first accepted launch.
    iree_hal_semaphore_t* semaphore;
    // Value proving all workstreams of that launch completed.
    uint64_t value;
    // A dependent stream must not block the shared producer queue.
    bool requires_independent_queue;
  } last_launch;

  // Immutable instantiation policy.
  unsigned long long flags;

  // Serializes mutable private descriptions and replacement preparation.
  iree_slim_mutex_t edit_mutex;
  // Serializes launch and installation of completed recordings.
  iree_slim_mutex_t launch_mutex;
} iree_hal_streaming_graph_exec_t;

static void iree_hal_streaming_graph_exec_destroy(
    iree_hal_streaming_graph_exec_t* exec);

static iree_hal_streaming_graph_compile_params_t
iree_hal_streaming_graph_exec_compile_params(
    iree_hal_streaming_graph_exec_t* exec) {
  return (iree_hal_streaming_graph_compile_params_t){
      .context = exec->context,
      .graph = exec->graph,
      .node_disabled_states = exec->node_disabled_states,
      .node_disabled_state_count = exec->node_disabled_state_count,
      .flags = exec->flags,
      .execution_owner = exec->execution_owner,
      .host_allocator = exec->host_allocator,
  };
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
  exec->node_disabled_states = NULL;
  exec->node_disabled_state_count = graph->node_count;
  exec->launch_generation = 0;
  exec->execution_owner = execution_owner ? execution_owner : exec;
  exec->last_launch.semaphore = NULL;
  exec->last_launch.value = 0;
  exec->last_launch.requires_independent_queue = false;
  exec->flags = flags;
  iree_slim_mutex_initialize(&exec->edit_mutex);
  iree_slim_mutex_initialize(&exec->launch_mutex);

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
    const iree_hal_streaming_graph_compile_params_t params =
        iree_hal_streaming_graph_exec_compile_params(exec);
    status = iree_hal_streaming_graph_recording_initialize(&params,
                                                           &exec->recording);
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

  iree_hal_streaming_graph_recording_deinitialize(&exec->recording);

  iree_hal_semaphore_release(exec->last_launch.semaphore);

  iree_allocator_free(exec->host_allocator, exec->node_disabled_states);
  iree_allocator_free(exec->host_allocator, exec->node_mappings);

  iree_hal_streaming_graph_release(exec->graph);
  iree_hal_streaming_graph_release(exec->source_graph);
  iree_hal_streaming_context_release(exec->context);
  iree_slim_mutex_deinitialize(&exec->launch_mutex);
  iree_slim_mutex_deinitialize(&exec->edit_mutex);

  iree_allocator_t host_allocator = exec->host_allocator;
  iree_allocator_free(host_allocator, exec);

  IREE_TRACE_ZONE_END(z0);
}

const iree_hal_streaming_graph_recording_t*
iree_hal_streaming_graph_exec_recording(
    const iree_hal_streaming_graph_exec_t* exec) {
  return &exec->recording;
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
  iree_slim_mutex_lock(&exec->launch_mutex);
  const bool is_live = !exec->is_destroyed;
  if (is_live) {
    iree_hal_streaming_graph_exec_retain(exec);
  }
  iree_slim_mutex_unlock(&exec->launch_mutex);
  return is_live;
}

iree_status_t iree_hal_streaming_graph_exec_destroy_handle(
    iree_hal_streaming_graph_exec_t* exec) {
  if (!exec) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }

  iree_slim_mutex_lock(&exec->edit_mutex);
  iree_slim_mutex_lock(&exec->launch_mutex);
  if (exec->is_destroyed) {
    iree_slim_mutex_unlock(&exec->launch_mutex);
    iree_slim_mutex_unlock(&exec->edit_mutex);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  exec->is_destroyed = true;
  iree_hal_semaphore_t* semaphore = exec->last_launch.semaphore;
  const uint64_t value = exec->last_launch.value;
  iree_slim_mutex_unlock(&exec->launch_mutex);
  iree_slim_mutex_unlock(&exec->edit_mutex);

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
    iree_slim_mutex_lock(&exec->edit_mutex);
    iree_slim_mutex_lock(&exec->launch_mutex);
    exec->is_destroyed = false;
    iree_slim_mutex_unlock(&exec->launch_mutex);
    iree_slim_mutex_unlock(&exec->edit_mutex);
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
  iree_slim_mutex_lock(&exec->edit_mutex);
  const bool owns_node =
      !exec->is_destroyed &&
      iree_hal_streaming_graph_exec_lookup_node(exec, node) != NULL;
  iree_slim_mutex_unlock(&exec->edit_mutex);
  return owns_node;
}

iree_hal_streaming_graph_node_t* iree_hal_streaming_graph_exec_lock_node(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node,
    iree_hal_streaming_graph_exec_edit_t* out_edit) {
  *out_edit = (iree_hal_streaming_graph_exec_edit_t){0};
  iree_slim_mutex_lock(&exec->edit_mutex);
  return exec->is_destroyed
             ? NULL
             : iree_hal_streaming_graph_exec_lookup_node(exec, node);
}

void iree_hal_streaming_graph_exec_end_edit(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_exec_edit_t* edit) {
  iree_slim_mutex_unlock(&exec->edit_mutex);
  iree_hal_streaming_graph_recording_deinitialize(&edit->recording);
}

bool iree_hal_streaming_graph_exec_node_is_enabled(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node) {
  iree_slim_mutex_lock(&exec->edit_mutex);
  node = exec->is_destroyed
             ? NULL
             : iree_hal_streaming_graph_exec_lookup_node(exec, node);
  const bool enabled =
      node && exec->node_disabled_states[node->node_index] == 0;
  iree_slim_mutex_unlock(&exec->edit_mutex);
  return enabled;
}

iree_status_t iree_hal_streaming_graph_exec_set_node_enabled(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_node_t* node, bool enabled) {
  if (!exec || !node) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  iree_hal_streaming_graph_exec_edit_t edit;
  node = iree_hal_streaming_graph_exec_lock_node(exec, node, &edit);
  if (!node) {
    iree_hal_streaming_graph_exec_end_edit(exec, &edit);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  const uint8_t old_disabled_state =
      exec->node_disabled_states[node->node_index];
  exec->node_disabled_states[node->node_index] = enabled ? 0 : 1;
  iree_status_t status =
      iree_hal_streaming_graph_exec_rebuild_locked(exec, &edit);
  if (!iree_status_is_ok(status)) {
    exec->node_disabled_states[node->node_index] = old_disabled_state;
  }
  iree_hal_streaming_graph_exec_end_edit(exec, &edit);
  return status;
}

iree_status_t iree_hal_streaming_graph_exec_rebuild_locked(
    iree_hal_streaming_graph_exec_t* exec,
    iree_hal_streaming_graph_exec_edit_t* edit) {
  const iree_hal_streaming_graph_compile_params_t params =
      iree_hal_streaming_graph_exec_compile_params(exec);
  iree_status_t status =
      iree_hal_streaming_graph_recording_initialize(&params, &edit->recording);
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&exec->launch_mutex);
    iree_hal_streaming_graph_recording_t previous = exec->recording;
    exec->recording = edit->recording;
    edit->recording = previous;
    iree_slim_mutex_unlock(&exec->launch_mutex);
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
  iree_hal_streaming_graph_exec_edit_t edit;
  node = iree_hal_streaming_graph_exec_lock_node(exec, node, &edit);
  hrx_graph_block_t* selected_block = NULL;
  if (node) {
    for (uint32_t i = 0; i < exec->recording.block_count; ++i) {
      hrx_graph_block_t* block = exec->recording.blocks[i];
      if (block->type == block_type &&
          iree_hal_streaming_graph_block_event(block)->source_node == node) {
        selected_block = block;
        break;
      }
    }
  }
  iree_hal_streaming_event_t* old_block_event = NULL;
  iree_hal_streaming_event_t* old_node_event = NULL;
  if (selected_block) {
    iree_hal_streaming_event_retain(event);
    iree_hal_streaming_event_retain(event);
    old_node_event = iree_hal_streaming_graph_node_event(node)->event;
    iree_hal_streaming_graph_node_event(node)->event = event;
    iree_slim_mutex_lock(&exec->launch_mutex);
    old_block_event =
        iree_hal_streaming_graph_block_event(selected_block)->event;
    iree_hal_streaming_graph_block_event(selected_block)->event = event;
    iree_slim_mutex_unlock(&exec->launch_mutex);
  }
  iree_hal_streaming_graph_exec_end_edit(exec, &edit);
  iree_hal_streaming_event_release(old_block_event);
  iree_hal_streaming_event_release(old_node_event);

  IREE_TRACE_ZONE_END(z0);
  return selected_block ? iree_ok_status()
                        : iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
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

static iree_status_t iree_hal_streaming_graph_exec_submit_blocks_locked(
    iree_hal_streaming_graph_exec_t* exec, iree_hal_streaming_stream_t* stream,
    iree_hal_queue_t* queue, iree_hal_queue_t* cooperative_queue,
    uint64_t launch_stream_tail_value,
    iree_hal_semaphore_list_t external_wait_semaphores,
    iree_hal_semaphore_list_t external_signal_semaphores,
    iree_hal_streaming_dropped_graph_list_t* dropped_graphs,
    bool* inout_retirement_failed);

// Checks every cooperative dispatch in |exec| and its child executable tree
// against the queue selected for |stream|. The caller holds the root
// executable mutex and the stream mutex, which keep the compiled block tree
// and borrowed queue stable throughout the recursive walk.
static iree_status_t
iree_hal_streaming_graph_exec_preflight_cooperative_dispatches_locked(
    iree_hal_streaming_graph_exec_t* exec, iree_hal_queue_t* cooperative_queue,
    bool* out_exceeds_residency) {
  for (uint32_t block_index = 0; block_index < exec->recording.block_count;
       ++block_index) {
    hrx_graph_block_t* block = exec->recording.blocks[block_index];
    if (block->type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH) {
      iree_hal_streaming_graph_exec_t* child =
          iree_hal_streaming_graph_block_child_graph(block)->exec;
      if (!child->recording.has_cooperative_dispatches) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          iree_hal_streaming_graph_exec_preflight_cooperative_dispatches_locked(
              child, cooperative_queue, out_exceeds_residency));
      if (*out_exceeds_residency) {
        return iree_ok_status();
      }
      continue;
    }
    if (block->type != IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH) {
      continue;
    }
    const iree_hal_streaming_graph_dispatch_block_attrs_t* dispatch =
        iree_hal_streaming_graph_block_dispatch(block);
    if (!iree_any_bit_set(dispatch->flags,
                          IREE_HAL_DISPATCH_FLAG_COOPERATIVE)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        iree_hal_streaming_check_cooperative_dispatch_residency(
            cooperative_queue, dispatch->executable,
            iree_hal_executable_function_from_index(
                (uint32_t)dispatch->entry_point),
            dispatch->config, out_exceeds_residency));
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
    hrx_graph_block_t* block, iree_hal_streaming_stream_t* stream,
    iree_hal_queue_t* queue, iree_hal_queue_t* cooperative_queue,
    uint64_t launch_stream_tail_value,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores,
    iree_hal_streaming_recorded_point_t* record_point,
    iree_hal_streaming_dropped_graph_list_t* dropped_graphs,
    bool* inout_retirement_failed) {
  if (block->type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH) {
    iree_hal_streaming_graph_exec_t* child_exec =
        iree_hal_streaming_graph_block_child_graph(block)->exec;
    // Every child entry carries this block's waits. A rejected child drains
    // its partial prefix before the parent drains its own preceding blocks.
    return iree_hal_streaming_graph_exec_submit_blocks_locked(
        child_exec, stream, queue, cooperative_queue, launch_stream_tail_value,
        wait_semaphores, signal_semaphores, dropped_graphs,
        inout_retirement_failed);
  }
  // Compilation gives every block a completion, joining parallel exits before
  // publishing the external signal. Failure cleanup relies on that proof.
  IREE_ASSERT(signal_semaphores.count > 0);

  iree_status_t status = iree_ok_status();
  switch (block->type) {
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_MEMORY_ALLOCATE:
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_MEMORY_FREE: {
      const iree_hal_streaming_graph_memory_block_attrs_t* memory =
          iree_hal_streaming_graph_block_memory(block);
      iree_hal_semaphore_t* accepted_semaphore = NULL;
      uint64_t accepted_value = 0;
      status = iree_hal_streaming_graph_allocation_enqueue(
          memory->allocation, queue, wait_semaphores, signal_semaphores,
          memory->flags, &accepted_semaphore, &accepted_value);
      if (!iree_status_is_ok(status) && accepted_semaphore) {
        // An allocation block can accept an allocation/unmap before a later
        // map rejects. Its private point covers that prefix; the compiled
        // block's output was never accepted and must not be waited upon.
        iree_status_t drain_status = iree_hal_semaphore_wait(
            accepted_semaphore, accepted_value, iree_infinite_timeout(),
            IREE_ASYNC_WAIT_FLAG_NONE);
        if (!iree_status_is_ok(drain_status)) {
          *inout_retirement_failed = true;
        }
        status = iree_status_join(status, drain_status);
      }
      break;
    }
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD:
      status = iree_hal_streaming_event_enqueue_record(
          iree_hal_streaming_graph_block_event(block)->event, stream->context,
          queue, wait_semaphores, signal_semaphores, record_point);
      break;
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_WAIT:
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER: {
      const iree_hal_queue_barrier_flags_t flags =
          block->type == IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER
              ? hrx_graph_block_barrier(block)->flags
              : IREE_HAL_QUEUE_BARRIER_FLAG_NONE;
      status = iree_hal_queue_barrier(queue, wait_semaphores, signal_semaphores,
                                      /*barriers=*/NULL, flags);
      break;
    }
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH: {
      const iree_hal_streaming_graph_dispatch_block_attrs_t* dispatch =
          iree_hal_streaming_graph_block_dispatch(block);
      iree_hal_queue_t* dispatch_queue = queue;
      if (iree_any_bit_set(dispatch->flags,
                           IREE_HAL_DISPATCH_FLAG_COOPERATIVE)) {
        IREE_ASSERT(cooperative_queue);
        dispatch_queue = cooperative_queue;
      }
      status = iree_hal_queue_dispatch(dispatch_queue, wait_semaphores,
                                       signal_semaphores, dispatch->executable,
                                       iree_hal_executable_function_from_index(
                                           (uint32_t)dispatch->entry_point),
                                       dispatch->config, dispatch->constants,
                                       dispatch->bindings,
                                       /*barriers=*/NULL, dispatch->flags);
      break;
    }
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE: {
      const hrx_graph_execute_block_t* execute = hrx_graph_block_execute(block);
      status = iree_hal_queue_execute(
          queue, wait_semaphores, signal_semaphores, execute->command_buffer,
          iree_hal_buffer_binding_table_empty(), execute->flags);
      break;
    }
    case IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL: {
      const iree_hal_streaming_graph_host_call_block_attrs_t* host_call =
          iree_hal_streaming_graph_block_host_call(block);
      iree_hal_host_call_t call;
      status = iree_hal_streaming_host_callback_prepare(
          stream->context->device_entry->host_callbacks, host_call->callback,
          host_call->user_data, host_call->user_data_size, host_call->resource,
          &call);
      if (iree_status_is_ok(status)) {
        const uint64_t args[4] = {0};
        status =
            iree_hal_queue_host_call(queue, wait_semaphores, signal_semaphores,
                                     call, args, host_call->flags);
        iree_hal_resource_release(call.resource);
      }
      break;
    }
    default:
      status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unsupported block type %u", block->type);
      break;
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
    bool* inout_retirement_failed) {
  if (exec->recording.block_count == 0) {
    return iree_hal_queue_barrier(
        queue, external_wait_semaphores, external_signal_semaphores,
        /*barriers=*/NULL, IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
  }

  // Every replay follows the prior launch's completion. Reserve values instead
  // of querying asynchronous progress: a later signal must never satisfy an
  // earlier invocation's wait. Rejected prefixes are drained before return;
  // skipping their generation leaves all subsequent edge values unambiguous.
  const uint64_t generation = ++exec->launch_generation;
  uint64_t values[HRX_GRAPH_MAX_WORKSTREAMS];
  if (exec->recording.semaphore_count) {
    for (uint32_t i = 0; i < IREE_ARRAYSIZE(values); ++i) {
      values[i] = generation;
    }
  }
  iree_status_t status = iree_ok_status();
  uint32_t block_index = 0;
  for (; iree_status_is_ok(status) && block_index < exec->recording.block_count;
       ++block_index) {
    hrx_graph_block_t* block = exec->recording.blocks[block_index];
    iree_hal_semaphore_list_t wait_semaphores;
    iree_hal_semaphore_list_t signal_semaphores;
    hrx_graph_dependencies_resolve(
        block->dependencies, block_index, exec->recording.semaphore_count,
        exec->recording.semaphores, values, external_wait_semaphores,
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
      iree_hal_streaming_event_acquire_recorded_point(
          iree_hal_streaming_graph_block_event(block)->event,
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
          inout_retirement_failed);
      if (iree_status_is_ok(status) && block_records_event) {
        // A rejected record keeps its old point. Later blocks in this launch
        // must observe the new point immediately after its acceptance.
        *dropped_capture_graph = iree_hal_streaming_event_commit_recorded_point(
            iree_hal_streaming_graph_block_event(block)->event, record_point);
      }
    }
    iree_hal_streaming_event_release_recorded_point(&event_wait_point);
  }
  if (!iree_status_is_ok(status)) {
    // The failed iteration has advanced block_index once. Every earlier block
    // was accepted and signals its indexed internal semaphore at generation;
    // only the final block signals externally, so no rejected tail enters this
    // span. A completed child contributes its parent point, while a rejected
    // child already drained its own partial prefix during recursive unwind.
    const uint32_t accepted_count = block_index - 1;
    for (uint32_t i = 0; i < accepted_count; ++i) {
      iree_status_t drain_status = iree_hal_semaphore_wait(
          exec->recording.semaphores[i], generation, iree_infinite_timeout(),
          IREE_ASYNC_WAIT_FLAG_NONE);
      if (!iree_status_is_ok(drain_status)) {
        *inout_retirement_failed = true;
      }
      status = iree_status_join(status, drain_status);
    }
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
  iree_slim_mutex_lock(&exec->launch_mutex);
  if (exec->is_destroyed) {
    iree_slim_mutex_unlock(&exec->launch_mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  // A graph may begin with a host callback. Publish preceding native writes
  // under the same stream lock that captures the graph's incoming dependency.
  iree_slim_mutex_lock(&stream->mutex);
  iree_status_t status = iree_hal_streaming_stream_flush_locked(stream);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&stream->mutex);
    iree_slim_mutex_unlock(&exec->launch_mutex);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  // An empty recording needs no submission unless it must carry a previous
  // launch across streams after a parameter update removed all executable work.
  if (exec->recording.block_count == 0 &&
      (!exec->last_launch.semaphore ||
       exec->last_launch.semaphore == stream->timeline_semaphore)) {
    *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_SUCCESS;
    iree_slim_mutex_unlock(&stream->mutex);
    iree_slim_mutex_unlock(&exec->launch_mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  // Every record this executable holds names an event of |exec->context|, so on
  // a stream of any other context iree_hal_streaming_event_enqueue_record would
  // refuse each of them in turn. Deciding the same question here refuses the
  // launch before it submits any of the graph, where the walk below breaks on
  // the first refusal and leaves the blocks ahead of the record in flight with
  // nothing signaling the launching stream's timeline.
  if (exec->recording.records_events && stream->context != exec->context) {
    iree_slim_mutex_unlock(&stream->mutex);
    iree_slim_mutex_unlock(&exec->launch_mutex);
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

  iree_hal_queue_t* cooperative_queue = NULL;
  bool cooperative_launch_too_large = false;
  if (exec->recording.has_cooperative_dispatches) {
    status = iree_hal_streaming_stream_select_cooperative_queue_locked(
        stream, &cooperative_queue);
  }
  if (iree_status_is_ok(status) && exec->recording.has_cooperative_dispatches) {
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
    iree_slim_mutex_unlock(&exec->launch_mutex);
    iree_hal_streaming_dropped_graph_list_deinitialize(&dropped_graphs);
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
      exec->recording.waits_for_events ||
      (waits_for_previous_launch &&
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
  bool retirement_failed = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_exec_submit_blocks_locked(
        exec, stream, submission.queue, cooperative_queue, stream_wait_value,
        wait_semaphores, submission.signals, &dropped_graphs,
        &retirement_failed);
  }

  if (retirement_failed && submission.completion) {
    // A failed prefix signal cannot prove hardware retirement. The final graph
    // signal was not accepted, so preserve the lane for queue teardown.
    iree_slim_mutex_lock(&stream->context->value_wait_lane_mutex);
    submission.completion->lane->has_failed_submission = true;
    iree_slim_mutex_unlock(&stream->context->value_wait_lane_mutex);
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
  iree_slim_mutex_unlock(&exec->launch_mutex);
  // Release captured graph references outside admission locks. The records
  // ended these associations even if a later block was rejected.
  iree_hal_streaming_dropped_graph_list_deinitialize(&dropped_graphs);
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
  iree_hal_streaming_graph_exec_edit_t edit;
  node = iree_hal_streaming_graph_exec_lock_node(exec, node, &edit);
  if (!node || node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH) {
    iree_hal_streaming_graph_exec_end_edit(exec, &edit);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  iree_arena_allocator_t scratch;
  iree_arena_initialize(&exec->context->device_entry->block_pool, &scratch);
  iree_hal_streaming_graph_t* old_child_graph =
      iree_hal_streaming_graph_node_child_graph(node)->graph;
  iree_hal_streaming_graph_t* replacement = NULL;
  iree_status_t status = iree_hal_streaming_graph_validate_child_graph(
      exec->source_graph, child_graph);
  iree_hal_streaming_graph_node_t* error_node = NULL;
  iree_hal_streaming_graph_exec_update_result_t update_result;
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_graph_check_update(
        old_child_graph, child_graph, &scratch, &error_node, &update_result);
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
    status = iree_hal_streaming_graph_exec_rebuild_locked(exec, &edit);
    if (!iree_status_is_ok(status)) {
      iree_hal_streaming_graph_node_child_graph(node)->graph = old_child_graph;
    }
  }
  iree_hal_streaming_graph_exec_end_edit(exec, &edit);
  iree_arena_deinitialize(&scratch);
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

  iree_slim_mutex_lock(&exec->edit_mutex);
  if (exec->is_destroyed) {
    iree_slim_mutex_unlock(&exec->edit_mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }
  iree_arena_allocator_t scratch;
  iree_arena_initialize(&exec->context->device_entry->block_pool, &scratch);
  iree_status_t status = iree_hal_streaming_graph_check_update(
      exec->graph, graph, &scratch, out_error_node, out_result);
  if (iree_status_is_ok(status) &&
      *out_result != IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_SUCCESS) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "graph update is not compatible");
  }
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&exec->edit_mutex);
    iree_arena_deinitialize(&scratch);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_ERROR;
  iree_hal_streaming_graph_exec_edit_t edit = {0};
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
    status = iree_hal_streaming_graph_exec_rebuild_locked(exec, &edit);
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
  iree_hal_streaming_graph_exec_end_edit(exec, &edit);
  iree_arena_deinitialize(&scratch);
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
