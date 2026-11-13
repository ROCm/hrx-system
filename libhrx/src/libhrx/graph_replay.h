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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_LIBHRX_GRAPH_REPLAY_H_
