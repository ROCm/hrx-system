// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_COMMON_GRAPH_RECORDING_H_
#define LIBHRX_SRC_BINDING_COMMON_GRAPH_RECORDING_H_

#include "common/graph_memory.h"
#include "common/host_callback.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/resource_set.h"
#include "libhrx/src/libhrx/graph_replay.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_graph_t iree_hal_streaming_graph_t;
typedef struct iree_hal_streaming_graph_exec_t iree_hal_streaming_graph_exec_t;
typedef struct iree_hal_streaming_graph_node_t iree_hal_streaming_graph_node_t;
typedef struct iree_hal_streaming_event_t iree_hal_streaming_event_t;

typedef enum iree_hal_streaming_graph_block_type_e {
  // iree_hal_queue_barrier
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_BARRIER =
      HRX_GRAPH_BLOCK_TYPE_QUEUE_BARRIER,
  // iree_hal_queue_execute
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE =
      HRX_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE,
  // iree_hal_queue_host_call
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_HOST_CALL =
      HRX_GRAPH_BLOCK_TYPE_FRONTEND,
  // Event record node.
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_RECORD,
  // Event wait node.
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_EVENT_WAIT,
  // iree_hal_queue_dispatch
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_QUEUE_DISPATCH,
  // Native graph allocation on the graph timeline.
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_MEMORY_ALLOCATE,
  // Native graph free on the graph timeline.
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_MEMORY_FREE,
  // Nested graph executable.
  IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH,
} iree_hal_streaming_graph_block_type_t;

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

// IREE_HAL_STREAMING_GRAPH_BLOCK_TYPE_CHILD_GRAPH
typedef struct iree_hal_streaming_graph_child_graph_block_attrs_t {
  // Owned child executable, released with this compiled block.
  iree_hal_streaming_graph_exec_t* exec;
} iree_hal_streaming_graph_child_graph_block_attrs_t;

// MEMORY_ALLOCATE/MEMORY_FREE share the same prepared allocation action.
typedef struct iree_hal_streaming_graph_memory_block_attrs_t {
  // Stable address and physical lifetime retained by the recording's resources.
  iree_hal_streaming_graph_allocation_t* allocation;
  // Prepared allocation/free action and replay policy.
  iree_hal_streaming_graph_allocation_flags_t flags;
} iree_hal_streaming_graph_memory_block_attrs_t;

static inline iree_hal_streaming_graph_host_call_block_attrs_t*
iree_hal_streaming_graph_block_host_call(const hrx_graph_block_t* block) {
  return (iree_hal_streaming_graph_host_call_block_attrs_t*)block->data;
}

static inline iree_hal_streaming_graph_event_block_attrs_t*
iree_hal_streaming_graph_block_event(const hrx_graph_block_t* block) {
  return (iree_hal_streaming_graph_event_block_attrs_t*)block->data;
}

static inline iree_hal_streaming_graph_dispatch_block_attrs_t*
iree_hal_streaming_graph_block_dispatch(const hrx_graph_block_t* block) {
  return (iree_hal_streaming_graph_dispatch_block_attrs_t*)block->data;
}

static inline iree_hal_streaming_graph_child_graph_block_attrs_t*
iree_hal_streaming_graph_block_child_graph(const hrx_graph_block_t* block) {
  return (iree_hal_streaming_graph_child_graph_block_attrs_t*)block->data;
}

static inline iree_hal_streaming_graph_memory_block_attrs_t*
iree_hal_streaming_graph_block_memory(const hrx_graph_block_t* block) {
  return (iree_hal_streaming_graph_memory_block_attrs_t*)block->data;
}

// Replaceable compiled state. It is embedded in an executable or temporarily
// held by an editor; accepted queue operations retain their own resources.
typedef struct iree_hal_streaming_graph_recording_t {
  // Owns block descriptors and copied submission parameters.
  iree_arena_allocator_t arena_allocator;
  // Arena-owned immutable submission order.
  hrx_graph_block_t** blocks;
  // Initialized entries, including during partial compilation.
  uint32_t block_count;
  // This recording or one of its children records an event.
  bool records_events;
  // An event producer can change between launches and need an independent
  // queue.
  bool waits_for_events;
  // A cooperative dispatch needs residency admission before submission.
  bool has_cooperative_dispatches;
  // Internal completion points used by the compiled dependency spans.
  uint32_t semaphore_count;
  // Contiguous timeline references borrowed from resource_set.
  iree_hal_semaphore_t** semaphores;
  // Owns explicitly captured buffers, executables, and command buffers.
  iree_hal_resource_set_t* resource_set;
  // Arena-owned graph-allocation claims held by this recording.
  struct iree_hal_streaming_graph_memory_claim_t* memory_claims;
} iree_hal_streaming_graph_recording_t;

// Borrowed inputs valid for the duration of compilation. The caller serializes
// edits to the descriptions; compilation does not inspect installed recordings.
typedef struct iree_hal_streaming_graph_compile_params_t {
  // Supplies the queue family, native device, and arena block pool.
  iree_hal_streaming_context_t* context;
  // Private executable descriptions to compile.
  iree_hal_streaming_graph_t* graph;
  // Per-node disabled state, indexed by the private graph's node index.
  const uint8_t* node_disabled_states;
  // Number of disabled-state entries.
  iree_host_size_t node_disabled_state_count;
  // Instantiation policy, including automatic graph allocation reuse.
  uint64_t flags;
  // Stable root executable identity for fixed-address allocation admission.
  iree_hal_streaming_graph_exec_t* execution_owner;
  // Host allocator for owned child executables.
  iree_allocator_t host_allocator;
} iree_hal_streaming_graph_compile_params_t;

// Initializes even on failure. The caller deinitializes rejected or replaced
// recordings after releasing all executable locks.
iree_status_t iree_hal_streaming_graph_recording_initialize(
    const iree_hal_streaming_graph_compile_params_t* params,
    iree_hal_streaming_graph_recording_t* out_recording);
void iree_hal_streaming_graph_recording_deinitialize(
    iree_hal_streaming_graph_recording_t* recording);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_COMMON_GRAPH_RECORDING_H_
