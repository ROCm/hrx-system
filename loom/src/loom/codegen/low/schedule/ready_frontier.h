// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Indexed dependency-ready scheduler frontiers.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_READY_FRONTIER_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_READY_FRONTIER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/util/segmented_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_LOW_SCHEDULE_READY_NODE_NONE UINT32_MAX

// Maximum number of ordered nodes copied from one ready view at a time.
#define LOOM_LOW_SCHEDULE_READY_COPY_CAPACITY 16u

typedef enum loom_low_schedule_ready_view_e {
  // Stable source-order fallback.
  LOOM_LOW_SCHEDULE_READY_VIEW_SOURCE = 0,
  // Candidate-local register-pressure relief.
  LOOM_LOW_SCHEDULE_READY_VIEW_PRESSURE = 1,
  // Strategy-specific latency or issue-readiness priority.
  LOOM_LOW_SCHEDULE_READY_VIEW_SCHEDULE = 2,
  // Storage compaction and alias-transfer opportunity.
  LOOM_LOW_SCHEDULE_READY_VIEW_STORAGE = 3,
  LOOM_LOW_SCHEDULE_READY_VIEW_COUNT = 4,
} loom_low_schedule_ready_view_t;

// Nomination keys for one ready node. Smaller keys have higher priority.
typedef struct loom_low_schedule_ready_keys_t {
  // Keys indexed by loom_low_schedule_ready_view_t.
  uint64_t values[LOOM_LOW_SCHEDULE_READY_VIEW_COUNT];
} loom_low_schedule_ready_keys_t;

// One indexed heap view over the shared ready membership.
typedef struct loom_low_schedule_ready_heap_t {
  // Segmented heap node indices.
  loom_segmented_storage_t nodes;
  // Number of populated heap entries.
  uint32_t count;
} loom_low_schedule_ready_heap_t;

// Shared dependency-ready membership and priority views for one function.
typedef struct loom_low_schedule_ready_frontier_t {
  // Per-node keys, heap positions, and descriptor links.
  loom_segmented_storage_t node_states;
  // Active heap views, beginning with the source-order view.
  loom_low_schedule_ready_heap_t views[LOOM_LOW_SCHEDULE_READY_VIEW_COUNT];
  // Head ready node for each descriptor ordinal.
  uint32_t* descriptor_heads;
  // Number of ready nodes for each descriptor ordinal.
  uint32_t* descriptor_counts;
  // Maximum node index plus one represented by this frontier.
  uint32_t node_capacity;
  // Number of descriptor ordinals represented by descriptor_heads.
  uint32_t descriptor_count;
  // Number of active leading entries in views.
  uint8_t view_count;
} loom_low_schedule_ready_frontier_t;

// Initializes an empty frontier. |view_count| must be in [1, VIEW_COUNT]. All
// retained storage is owned by |arena|.
iree_status_t loom_low_schedule_ready_frontier_initialize(
    uint32_t node_capacity, uint32_t descriptor_count, uint8_t view_count,
    iree_arena_allocator_t* arena,
    loom_low_schedule_ready_frontier_t* out_frontier);

// Returns true when |node_index| is in the frontier.
bool loom_low_schedule_ready_frontier_contains(
    const loom_low_schedule_ready_frontier_t* frontier, uint32_t node_index);

// Returns the number of ready nodes.
uint32_t loom_low_schedule_ready_frontier_count(
    const loom_low_schedule_ready_frontier_t* frontier);

// Inserts one node into every active view and its optional descriptor list.
void loom_low_schedule_ready_frontier_insert(
    loom_low_schedule_ready_frontier_t* frontier, uint32_t node_index,
    const loom_low_schedule_ready_keys_t* keys, uint32_t descriptor_ordinal);

// Removes one node from every active view and its descriptor list.
void loom_low_schedule_ready_frontier_remove(
    loom_low_schedule_ready_frontier_t* frontier, uint32_t node_index);

// Copies up to |capacity| nodes from |view| in priority order. The traversal is
// bounded by |capacity| and does not modify the indexed heap.
uint8_t loom_low_schedule_ready_frontier_copy_best(
    const loom_low_schedule_ready_frontier_t* frontier,
    loom_low_schedule_ready_view_t view, uint8_t capacity,
    uint32_t* out_node_indices);

// Changes one nomination key and repairs its indexed heap position.
void loom_low_schedule_ready_frontier_update_key(
    loom_low_schedule_ready_frontier_t* frontier,
    loom_low_schedule_ready_view_t view, uint32_t node_index, uint64_t key);

// Returns an arbitrary ready node with |descriptor_ordinal|, or
// READY_NODE_NONE when no matching node is ready.
uint32_t loom_low_schedule_ready_frontier_descriptor_head(
    const loom_low_schedule_ready_frontier_t* frontier,
    uint32_t descriptor_ordinal);

// Returns the number of ready nodes with |descriptor_ordinal|.
uint32_t loom_low_schedule_ready_frontier_descriptor_count(
    const loom_low_schedule_ready_frontier_t* frontier,
    uint32_t descriptor_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_READY_FRONTIER_H_
