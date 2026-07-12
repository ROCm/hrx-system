// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bounded AMDGPU cross-block memory wait-counter state.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_MEMORY_FRONTIER_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_MEMORY_FRONTIER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/target/arch/amdgpu/planning/wait_counters.h"

#ifdef __cplusplus
extern "C" {
#endif

// One bit for each normalized low memory space.
typedef uint8_t loom_amdgpu_wait_memory_space_flags_t;

typedef enum loom_amdgpu_wait_memory_access_flag_bits_e {
  // Outstanding asynchronous reads.
  LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_READ = 1u << 0,
  // Outstanding asynchronous writes.
  LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE = 1u << 1,
} loom_amdgpu_wait_memory_access_flag_bits_t;
typedef uint8_t loom_amdgpu_wait_memory_access_flags_t;

// Target memory-counter classification for one schedule node.
typedef struct loom_amdgpu_wait_memory_node_t {
  // Counter classes advanced by dependency-participating reads.
  uint32_t read_counter_mask;
  // Counter classes advanced by dependency-participating writes.
  uint32_t write_counter_mask;
  // Counter classes fully drained after this node issued.
  uint32_t drained_after_production_counter_mask;
  // Normalized memory spaces read by this node.
  loom_amdgpu_wait_memory_space_flags_t read_space_flags;
  // Normalized memory spaces written by this node.
  loom_amdgpu_wait_memory_space_flags_t write_space_flags;
} loom_amdgpu_wait_memory_node_t;

// Outstanding memory-space flags grouped by target counter and access kind.
typedef struct loom_amdgpu_wait_memory_state_t {
  // Packed read and write space flags indexed by target counter slot. Read
  // flags occupy the low byte and write flags occupy the high byte.
  uint16_t counter_access_space_flags[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
} loom_amdgpu_wait_memory_state_t;

// Cross-block frontier for one scheduled function.
typedef struct loom_amdgpu_wait_memory_frontier_t {
  // Schedule whose CFG and block order define the frontier.
  const loom_low_schedule_table_t* schedule;
  // Per-node target memory classifications.
  const loom_amdgpu_wait_memory_node_t* nodes;
  // Conservative transitive outgoing state for every block.
  loom_amdgpu_wait_memory_state_t* static_outgoing_states;
  // Refined outgoing state recorded after each processed block.
  loom_amdgpu_wait_memory_state_t* resolved_outgoing_states;
  // Per-block worklist and resolved-state bits.
  uint8_t* block_flags;
  // Incoming state active while the current block is processed.
  loom_amdgpu_wait_memory_state_t active_state;
  // Current block index, or UINT16_MAX outside block processing.
  uint16_t active_block_index;
} loom_amdgpu_wait_memory_frontier_t;

// Returns the normalized memory-space bit for |memory_space|.
loom_amdgpu_wait_memory_space_flags_t loom_amdgpu_wait_memory_space_flag(
    loom_low_memory_space_t memory_space);

// Initializes bounded cross-block state from the schedule CFG and node
// classifications. All retained storage is owned by |arena|.
iree_status_t loom_amdgpu_wait_memory_frontier_initialize(
    const loom_low_schedule_table_t* schedule,
    const loom_amdgpu_wait_memory_node_t* nodes, iree_arena_allocator_t* arena,
    loom_amdgpu_wait_memory_frontier_t* out_frontier);

// Begins processing |block_index| and selects predecessor state. Processed
// predecessors contribute refined state; backedges and forward-unknown
// predecessors contribute conservative static state.
void loom_amdgpu_wait_memory_frontier_begin_block(
    loom_amdgpu_wait_memory_frontier_t* frontier, uint16_t block_index);

// Returns counter masks from incoming producers in aliasing memory spaces and
// selected access classes.
uint32_t loom_amdgpu_wait_memory_frontier_query(
    const loom_amdgpu_wait_memory_frontier_t* frontier,
    loom_amdgpu_wait_memory_space_flags_t space_flags,
    loom_amdgpu_wait_memory_access_flags_t access_flags);

// Returns full-drain counters needed by the current node's incoming memory
// antidependencies.
uint32_t loom_amdgpu_wait_memory_frontier_dependency_mask(
    const loom_amdgpu_wait_memory_frontier_t* frontier,
    const loom_amdgpu_wait_memory_node_t* node);

// Removes fully drained counter classes from every active memory-space state.
void loom_amdgpu_wait_memory_frontier_drain(
    loom_amdgpu_wait_memory_frontier_t* frontier, uint32_t counter_mask);

// Publishes remaining incoming state and locally produced undrained counters
// as the refined outgoing state for the active block.
void loom_amdgpu_wait_memory_frontier_end_block(
    loom_amdgpu_wait_memory_frontier_t* frontier);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_MEMORY_FRONTIER_H_
