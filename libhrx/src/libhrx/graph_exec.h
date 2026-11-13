// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBHRX_SRC_LIBHRX_GRAPH_EXEC_H_
#define LIBHRX_SRC_LIBHRX_GRAPH_EXEC_H_

#include "hrx_runtime.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/resource_set.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compiled state installed as one unit. Accepted queue operations retain their
// own command buffers and semaphore lists, so retiring this CPU recording never
// waits for device completion. Pools, allocation epochs, and application
// callback data remain owned by the caller.
typedef struct hrx_graph_recording_t {
  // Retained source graph and device, including captured native resources.
  hrx_graph_t graph;
  // Storage for compiled blocks and borrowed semaphore pointers.
  iree_arena_allocator_t arena_allocator;
  // Blocks in submission order, ending in a single completion producer.
  struct hrx_graph_exec_block_t** blocks;
  // Number of initialized entries in |blocks|.
  uint32_t block_count;
  // Number of internal edge semaphores.
  uint32_t semaphore_count;
  // Internal edge semaphores borrowed from |resource_set|.
  iree_hal_semaphore_t** semaphores;
  // Owns retaining command buffers and internal edge semaphores.
  iree_hal_resource_set_t* resource_set;
} hrx_graph_recording_t;

typedef struct hrx_graph_exec_s {
  // References to this executable and its installed recording.
  iree_atomic_ref_count_t ref_count;
  // Recording family borrowed from the device retained by |recording.graph|.
  const iree_hal_queue_family_t* queue_family;
  // Instantiation flags supplied by the application.
  uint32_t flags;
  // Serializes submission ordering and recording publication, never
  // preparation.
  iree_slim_mutex_t launch_mutex;
  // Reserved generation signaled once per internal edge of each launch.
  uint64_t launch_generation;
  // Retained completion semaphore ordering replays across streams and updates.
  iree_hal_semaphore_t* last_launch_semaphore;
  // Accepted value on |last_launch_semaphore|, never an observed payload.
  uint64_t last_launch_value;
  // Installed compiled state, read and exchanged under |launch_mutex|.
  hrx_graph_recording_t recording;
  // Serializes edit preparation. Launch never takes this lock. Its own cache
  // line keeps waiting editors from invalidating submission state.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_slim_mutex_t update_mutex;
} hrx_graph_exec_s;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_LIBHRX_GRAPH_EXEC_H_
