// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBHRX_GRAPH_DEPENDENCIES_H_
#define LIBHRX_GRAPH_DEPENDENCIES_H_

#include "graph_analysis.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/resource_set.h"

#ifdef __cplusplus
extern "C" {
#endif

// Every block except the final completion producer signals the internal
// semaphore at its own block ordinal. A partition waits on all workstreams in
// its predecessor; its bounded width makes those indices relative distances.
// The first partition borrows the launch's external waits, and the final block
// borrows its external signals. No frontend payload or stream state lives here.
typedef struct hrx_graph_dependencies_t {
  // Number of predecessor workstreams, or zero for external entry waits.
  uint8_t wait_count;
  // Distance from this block ordinal to the first predecessor semaphore.
  uint8_t wait_distance;
} hrx_graph_dependencies_t;

// Compilation-only cursor over a verified schedule, including the final join
// when its last partition contains parallel workstreams.
typedef struct hrx_graph_dependency_cursor_t {
  // Borrowed analysis result, live throughout compilation.
  const hrx_graph_schedule_t* schedule;
  // Total compiled blocks, including a synthetic terminal join if needed.
  uint32_t block_count;
  // Internal semaphore count; every nonempty graph has one external producer.
  uint32_t semaphore_count;
  // Current partition; partition_count denotes the optional terminal join.
  uint32_t partition_index;
  // Next workstream within the current partition.
  uint8_t stream_index;
} hrx_graph_dependency_cursor_t;

typedef struct hrx_graph_dependency_block_t {
  // Borrowed recording range, or NULL for the terminal dependency-only join.
  const hrx_graph_partition_t* partition;
  // Dependency encoding copied into the frontend's compiled block.
  hrx_graph_dependencies_t dependencies;
  // Workstream to record within |partition|; zero for isolated operations.
  uint8_t stream_index;
} hrx_graph_dependency_block_t;

// Consumes producer-established schedule invariants without revalidating nodes.
void hrx_graph_dependency_cursor_initialize(
    const hrx_graph_schedule_t* schedule,
    hrx_graph_dependency_cursor_t* out_cursor);
bool hrx_graph_dependency_cursor_next(hrx_graph_dependency_cursor_t* cursor,
                                      hrx_graph_dependency_block_t* out_block);

// Allocates borrowed semaphore pointers in |arena| and transfers each created
// semaphore reference to |resource_set|. The caller owns both containers and
// releases them on partial construction failure as well as normal retirement.
iree_status_t hrx_graph_dependencies_allocate(
    iree_hal_device_t* device, uint32_t semaphore_count,
    iree_arena_allocator_t* arena, iree_hal_resource_set_t* resource_set,
    iree_hal_semaphore_t*** out_semaphores);

// Resolves immutable dependency spans without copying semaphore pointers.
// |values| repeats the reserved launch generation HRX_GRAPH_MAX_WORKSTREAMS
// times. The caller orders successive launches by the previous completion and
// never reuses a generation, including after a partially rejected launch.
// Returned lists borrow their inputs through the immediate queue call only.
static inline void hrx_graph_dependencies_resolve(
    hrx_graph_dependencies_t dependencies, uint32_t block_index,
    uint32_t semaphore_count, iree_hal_semaphore_t** semaphores,
    uint64_t* values, iree_hal_semaphore_list_t external_waits,
    iree_hal_semaphore_list_t external_signals,
    iree_hal_semaphore_list_t* out_waits,
    iree_hal_semaphore_list_t* out_signals) {
  *out_waits = external_waits;
  if (dependencies.wait_count) {
    *out_waits = (iree_hal_semaphore_list_t){
        .count = dependencies.wait_count,
        .semaphores = semaphores + block_index - dependencies.wait_distance,
        .payload_values = values,
    };
  }
  *out_signals = external_signals;
  if (block_index < semaphore_count) {
    *out_signals = (iree_hal_semaphore_list_t){
        .count = 1,
        .semaphores = semaphores + block_index,
        .payload_values = values,
    };
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_GRAPH_DEPENDENCIES_H_
