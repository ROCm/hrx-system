// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBHRX_SRC_LIBHRX_GRAPH_REPLAY_H_
#define LIBHRX_SRC_LIBHRX_GRAPH_REPLAY_H_

#include "graph_dependencies.h"

#ifdef __cplusplus
extern "C" {
#endif

enum hrx_graph_block_type_e {
  HRX_GRAPH_BLOCK_TYPE_QUEUE_BARRIER = 0,
  HRX_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE,
  // Higher values belong to the frontend that compiled the recording.
  HRX_GRAPH_BLOCK_TYPE_FRONTEND,
};

// A compiled operation with only its own payload, stored in the recording's
// arena. The recording and accepted submissions retain their usual resources;
// a block introduces no independent resource ownership.
typedef struct hrx_graph_block_t {
  // Shared operation or frontend-specific operation tag.
  uint8_t type;
  // Predecessor span relative to this block's submission ordinal.
  hrx_graph_dependencies_t dependencies;
  // Inline kind-specific parameters, aligned for pointers and 64-bit flags.
  iree_alignas(8) uint8_t data[];
} hrx_graph_block_t;

typedef struct hrx_graph_barrier_block_t {
  // HAL submission flags for this dependency join.
  iree_hal_queue_barrier_flags_t flags;
} hrx_graph_barrier_block_t;

typedef struct hrx_graph_execute_block_t {
  // Recorded commands borrowed from the recording's resource set.
  iree_hal_command_buffer_t* command_buffer;
  // HAL execution flags captured at compilation.
  iree_hal_queue_execute_flags_t flags;
} hrx_graph_execute_block_t;

// Compiled submission order and its indexed internal completion points.
// Storage and resource retention belong to the enclosing recording.
typedef struct hrx_graph_block_list_t {
  // Number of blocks published by the compiler.
  uint32_t count;
  // Every nonfinal block signals one internal timeline at its block ordinal.
  uint32_t semaphore_count;
  // Arena-owned block pointers in submission order.
  hrx_graph_block_t** values;
  // Borrowed timelines retained by the recording's resource set.
  iree_hal_semaphore_t** semaphores;
} hrx_graph_block_list_t;

// Allocates a compiler-selected payload. Its type and size are established by
// the producer; replay consumes them without dynamic checks or size metadata.
iree_status_t hrx_graph_block_allocate(iree_arena_allocator_t* arena,
                                       uint8_t type,
                                       hrx_graph_dependencies_t dependencies,
                                       iree_host_size_t payload_size,
                                       hrx_graph_block_t** out_block);

// Header and inline payload share storage and lifetime. Callers serialize
// mutations with recording publication and consume the compiled type directly.
static inline hrx_graph_barrier_block_t* hrx_graph_block_barrier(
    const hrx_graph_block_t* block) {
  return (hrx_graph_barrier_block_t*)block->data;
}

static inline hrx_graph_execute_block_t* hrx_graph_block_execute(
    const hrx_graph_block_t* block) {
  return (hrx_graph_execute_block_t*)block->data;
}

// Submits ordinary blocks starting at |inout_block_index|, stopping at a
// frontend operation, a rejected queue call, or the end of the recording.
// The index advances only after acceptance and therefore also identifies the
// accepted prefix on failure. Frontends perform their own operations and resume
// with the same generation and external points; this loop owns no stream,
// submission state, resource references, or failure/retry policy.
//
// |barriers| preserves the frontend's cache contract for dependency joins.
// Command buffers carry their own memory transitions. |values| is the existing
// bounded generation array used by hrx_graph_dependencies_resolve.
static inline iree_status_t hrx_graph_blocks_submit(
    const hrx_graph_block_list_t* blocks, iree_hal_queue_t* queue,
    uint64_t* values, iree_hal_semaphore_list_t external_waits,
    iree_hal_semaphore_list_t external_signals,
    const iree_hal_queue_barriers_t* barriers, uint32_t* inout_block_index) {
  iree_status_t status = iree_ok_status();
  uint32_t block_index = *inout_block_index;
  while (iree_status_is_ok(status) && block_index < blocks->count) {
    const hrx_graph_block_t* block = blocks->values[block_index];
    if (block->type >= HRX_GRAPH_BLOCK_TYPE_FRONTEND) {
      break;
    }
    iree_hal_semaphore_list_t waits;
    iree_hal_semaphore_list_t signals;
    hrx_graph_dependencies_resolve(block->dependencies, block_index,
                                   blocks->semaphore_count, blocks->semaphores,
                                   values, external_waits, external_signals,
                                   &waits, &signals);
    if (block->type == HRX_GRAPH_BLOCK_TYPE_QUEUE_EXECUTE) {
      const hrx_graph_execute_block_t* execute = hrx_graph_block_execute(block);
      status = iree_hal_queue_execute(
          queue, waits, signals, execute->command_buffer,
          iree_hal_buffer_binding_table_empty(), execute->flags);
    } else {
      status = iree_hal_queue_barrier(queue, waits, signals, barriers,
                                      hrx_graph_block_barrier(block)->flags);
    }
    if (iree_status_is_ok(status)) {
      ++block_index;
    }
  }
  *inout_block_index = block_index;
  return status;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_LIBHRX_GRAPH_REPLAY_H_
