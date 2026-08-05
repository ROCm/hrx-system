// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bounded AMDGPU wait state propagated across scheduled control flow.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_FRONTIER_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_FRONTIER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/table.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/target/arch/amdgpu/planning/wait_counters.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#ifdef __cplusplus
extern "C" {
#endif

// One bit for each normalized low memory space.
typedef uint8_t loom_amdgpu_wait_memory_space_flags_t;

enum {
  // Number of normalized memory-space alias classes from GENERIC through
  // WASM_MEMORY.
  LOOM_AMDGPU_WAIT_MEMORY_SPACE_COUNT =
      LOOM_LOW_MEMORY_SPACE_WASM_MEMORY - LOOM_LOW_MEMORY_SPACE_GENERIC + 1u,
};

typedef enum loom_amdgpu_wait_memory_access_flag_bits_e {
  // Outstanding asynchronous reads.
  LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_READ = 1u << 0,
  // Outstanding asynchronous writes.
  LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE = 1u << 1,
} loom_amdgpu_wait_memory_access_flag_bits_t;
typedef uint8_t loom_amdgpu_wait_memory_access_flags_t;

typedef enum loom_amdgpu_wait_xcnt_group_flag_bits_e {
  // Outstanding vector-memory translation source reads.
  LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_VMEM = 1u << 0,
  // Outstanding scalar-memory translation source reads.
  LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_SMEM = 1u << 1,
} loom_amdgpu_wait_xcnt_group_flag_bits_t;
typedef uint8_t loom_amdgpu_wait_xcnt_group_flags_t;

// Target wait classification for one schedule node.
typedef struct loom_amdgpu_wait_frontier_node_t {
  // Counter classes advanced by dependency-participating reads.
  uint32_t read_counter_mask;
  // Counter classes advanced by dependency-participating writes.
  uint32_t write_counter_mask;
  // Counter classes fully drained after this node issued.
  uint32_t drained_after_production_counter_mask;
  // Counter classes drained when this node executes.
  uint32_t drain_counter_mask;
  // Gfx125x XCNT translation group produced by this node, or zero.
  loom_amdgpu_wait_xcnt_group_flags_t xcnt_group_flags;
  // Normalized memory spaces read by this node.
  loom_amdgpu_wait_memory_space_flags_t read_space_flags;
  // Normalized memory spaces written by this node.
  loom_amdgpu_wait_memory_space_flags_t write_space_flags;
  // Completion-order class for asynchronous VMEM results.
  loom_amdgpu_vmem_result_order_class_t vmem_result_order_class;
} loom_amdgpu_wait_frontier_node_t;

// Outstanding counter masks indexed by aliasing consumer memory space.
typedef struct loom_amdgpu_wait_memory_state_t {
  // Packed read counters in the low byte and write counters in the high byte
  // for each normalized consumer space.
  uint16_t access_counter_masks[LOOM_AMDGPU_WAIT_MEMORY_SPACE_COUNT];
} loom_amdgpu_wait_memory_state_t;

// Cross-block wait frontier for one scheduled function.
typedef struct loom_amdgpu_wait_frontier_t {
  // Schedule whose CFG and block order define the frontier.
  const loom_low_schedule_table_t* schedule;
  // Final physical allocation used to map scheduled result ordinals.
  const loom_low_allocation_table_t* allocation;
  // Per-node target wait classifications.
  const loom_amdgpu_wait_frontier_node_t* nodes;
  // Memory dependency state.
  struct {
    // Conservative transitive outgoing state for every block.
    loom_amdgpu_wait_memory_state_t* static_outgoing_states;
    // Refined outgoing state recorded after each processed block.
    loom_amdgpu_wait_memory_state_t* resolved_outgoing_states;
    // Incoming state active while the current block is processed.
    loom_amdgpu_wait_memory_state_t active_state;
  } memory;
  // Outstanding VMEM result writes by physical vector-register unit.
  struct {
    // Number of physical VGPR units in the packed state domain.
    iree_host_size_t vgpr_unit_count;
    // Number of physical AGPR units after the VGPR state domain.
    iree_host_size_t agpr_unit_count;
    // Number of packed state words per block.
    iree_host_size_t word_count;
    // Conservative transitive outgoing words for every block.
    uint64_t* static_outgoing_words;
    // Refined outgoing words recorded after each processed block.
    uint64_t* resolved_outgoing_words;
    // Incoming words active while the current block is processed.
    uint64_t* active_words;
    // Flags summarizing active VMEM result state.
    uint8_t active_flags;
  } vmem_results;
  // Assignment-backed storage leases that remain active across block edges.
  struct {
    // Number of allocation storage-lease instances in the packed domain.
    iree_host_size_t lease_count;
    // Number of packed state words per block.
    iree_host_size_t word_count;
    // Lease membership grouped by release counter.
    struct {
      // Inline words for a one-word storage frontier.
      uint64_t inline_words[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
      // Words indexed by AMDGPU wait-counter slot. Points into |inline_words|
      // for one-word frontiers and otherwise into the arena; present whenever
      // |active_words| is present.
      uint64_t* words;
    } release_membership;
    // Conservative transitive outgoing words for every block.
    uint64_t* static_outgoing_words;
    // Refined outgoing words recorded after each processed block.
    uint64_t* resolved_outgoing_words;
    // Incoming words active while the current block is processed.
    uint64_t* active_words;
  } storage_leases;
  // Gfx125x XCNT translation groups that may be active across block edges.
  struct {
    // Conservative transitive outgoing flags for every block.
    loom_amdgpu_wait_xcnt_group_flags_t* static_outgoing_flags;
    // Refined outgoing flags recorded after each processed block.
    loom_amdgpu_wait_xcnt_group_flags_t* resolved_outgoing_flags;
    // Incoming and locally produced groups active in the current block.
    loom_amdgpu_wait_xcnt_group_flags_t active_flags;
  } xcnt;
  // Counter classes fully drained on every path through each block.
  uint32_t* block_drain_counter_masks;
  // Per-block worklist and resolved-state bits.
  uint8_t* block_flags;
  // Current block index, or UINT16_MAX outside block processing.
  uint16_t active_block_index;
} loom_amdgpu_wait_frontier_t;

// Returns the normalized memory-space bit for |memory_space|.
loom_amdgpu_wait_memory_space_flags_t loom_amdgpu_wait_memory_space_flag(
    loom_low_memory_space_t memory_space);

// Initializes bounded cross-block state from the schedule CFG, allocation,
// and node classifications. Dynamically retained storage is owned by |arena|;
// inline state lives in |out_frontier|.
iree_status_t loom_amdgpu_wait_frontier_initialize(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_wait_frontier_node_t* nodes,
    iree_host_size_t vgpr_unit_count, iree_host_size_t agpr_unit_count,
    const uint32_t* planned_block_drain_counter_masks,
    iree_arena_allocator_t* arena, loom_amdgpu_wait_frontier_t* out_frontier);

// Begins processing |block_index| and selects predecessor state. Processed
// predecessors contribute refined state; backedges and forward-unknown
// predecessors contribute conservative static state.
void loom_amdgpu_wait_frontier_begin_block(
    loom_amdgpu_wait_frontier_t* frontier, uint16_t block_index);

// Returns counter masks from incoming producers in aliasing memory spaces and
// selected access classes.
uint32_t loom_amdgpu_wait_frontier_memory_query(
    const loom_amdgpu_wait_frontier_t* frontier,
    loom_amdgpu_wait_memory_space_flags_t space_flags,
    loom_amdgpu_wait_memory_access_flags_t access_flags);

// Returns full-drain counters needed by the current node's incoming memory
// antidependencies.
uint32_t loom_amdgpu_wait_frontier_memory_dependency_mask(
    const loom_amdgpu_wait_frontier_t* frontier,
    const loom_amdgpu_wait_frontier_node_t* node);

// Returns the single known completion-order class for outstanding VMEM writes
// overlapping |assignment|. UNKNOWN represents either an unclassified write or
// writes from multiple classes; NONE means no outstanding write overlaps.
loom_amdgpu_vmem_result_order_class_t
loom_amdgpu_wait_frontier_query_vmem_result(
    const loom_amdgpu_wait_frontier_t* frontier,
    const loom_low_allocation_assignment_t* assignment);

// Returns true when assignment-backed storage lease |lease_index| may still be
// active on entry to the current program point.
bool loom_amdgpu_wait_frontier_storage_lease_is_active(
    const loom_amdgpu_wait_frontier_t* frontier, iree_host_size_t lease_index);

// Retires one active assignment-backed storage lease after its exact producer
// has been proven complete without a full counter drain.
void loom_amdgpu_wait_frontier_retire_storage_lease(
    loom_amdgpu_wait_frontier_t* frontier, iree_host_size_t lease_index);

// Returns gfx125x translation groups that may still retain source state at the
// current program point.
loom_amdgpu_wait_xcnt_group_flags_t
loom_amdgpu_wait_frontier_active_xcnt_groups(
    const loom_amdgpu_wait_frontier_t* frontier);

// Applies the implicit gfx125x group transition before a translation packet.
// Source leases from other groups are released; same-group leases remain.
void loom_amdgpu_wait_frontier_prepare_xcnt_producer(
    loom_amdgpu_wait_frontier_t* frontier,
    loom_amdgpu_wait_xcnt_group_flags_t group_flags);

// Records a gfx125x translation packet in the active group.
void loom_amdgpu_wait_frontier_note_xcnt_producer(
    loom_amdgpu_wait_frontier_t* frontier,
    loom_amdgpu_wait_xcnt_group_flags_t group_flags);

// Removes fully drained counter classes from active frontier state.
void loom_amdgpu_wait_frontier_drain(loom_amdgpu_wait_frontier_t* frontier,
                                     uint32_t counter_mask);

// Publishes the refined outgoing state for the active block.
void loom_amdgpu_wait_frontier_end_block(loom_amdgpu_wait_frontier_t* frontier);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_FRONTIER_H_
