// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical loop-interval analysis for AMDGPU wait placement.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_LOOP_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_LOOP_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/target/arch/amdgpu/planning/wait_counters.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_wait_loop_t loom_amdgpu_wait_loop_t;

typedef enum loom_amdgpu_wait_loop_dependency_flag_bits_e {
  // The dependency is an SSA use eligible for canonical-loop relocation.
  LOOM_AMDGPU_WAIT_LOOP_DEPENDENCY_FLAG_SSA_USE = 1u << 0,
} loom_amdgpu_wait_loop_dependency_flag_bits_t;
typedef uint8_t loom_amdgpu_wait_loop_dependency_flags_t;

// One target-counter dependency retained by AMDGPU wait planning.
typedef struct loom_amdgpu_wait_loop_dependency_t {
  // Producer node for the dependency.
  uint32_t producer_node;
  // Consumer node for the dependency.
  uint32_t consumer_node;
  // Next dependency for the same consumer or relocated loop-entry slot.
  uint32_t next_dependency;
  // Counters produced by |producer_node| and needed by this use.
  uint32_t counter_mask;
  // Target wait-plan reason identifier preserved for action provenance.
  uint16_t reason_id;
  // Dependency classification flags.
  loom_amdgpu_wait_loop_dependency_flags_t flags;
} loom_amdgpu_wait_loop_dependency_t;

// Immutable per-node facts consumed by loop-counter frontier analysis.
typedef struct loom_amdgpu_wait_loop_node_t {
  // Counters advanced when this node executes.
  uint32_t producer_counter_mask;
  // Counters advanced by writes when this node executes.
  uint32_t write_counter_mask;
  // Counters unconditionally reset when this node executes.
  uint32_t reset_counter_mask;
  // Counter domains in which the node can create a target hazard.
  uint32_t hazard_counter_mask;
  // Counter classes advanced by workgroup-memory writes.
  uint32_t workgroup_write_counter_mask;
  // Workgroup-memory write counters observed by this node's barrier.
  uint32_t workgroup_barrier_counter_mask;
} loom_amdgpu_wait_loop_node_t;

typedef enum loom_amdgpu_wait_loop_cyclic_frontier_flag_bits_e {
  // The block has a proven fixed-point incoming counter state.
  LOOM_AMDGPU_WAIT_LOOP_CYCLIC_FRONTIER_FLAG_VALID = 1u << 0,
} loom_amdgpu_wait_loop_cyclic_frontier_flag_bits_t;
typedef uint8_t loom_amdgpu_wait_loop_cyclic_frontier_flags_t;

// One stable incoming counter epoch reconstructed for a canonical loop block.
typedef struct loom_amdgpu_wait_loop_cyclic_frontier_t {
  // Fixed-point validity flags. A valid frontier may have no outstanding work.
  loom_amdgpu_wait_loop_cyclic_frontier_flags_t flags;
  // First scheduled ordinal in the trailing producer epoch.
  uint32_t producer_start_ordinal;
  // Outstanding packets entering the next execution of the block.
  uint32_t outstanding_count;
  // Outstanding writes entering the next execution of the block.
  uint32_t outstanding_write_count;
  // Outstanding workgroup writes entering the next execution of the block.
  uint32_t outstanding_workgroup_write_count;
} loom_amdgpu_wait_loop_cyclic_frontier_t;

// Transient AMDGPU eligibility and ancestor index over the schedule's preserved
// canonical loop forest. Unsupported target-low edge shapes remain valid but
// produce no relocation target.
typedef struct loom_amdgpu_wait_loop_analysis_t {
  // Schedule whose CFG and node table are analyzed.
  const loom_low_schedule_table_t* schedule;
  // Canonical loop records in header order.
  const loom_amdgpu_wait_loop_t* loops;
  // Binary-lifted valid ancestor loop indices.
  const uint32_t* valid_ancestor_loops;
  // Number of canonical loop records.
  iree_host_size_t loop_count;
} loom_amdgpu_wait_loop_analysis_t;

// Builds target eligibility and a bounded ancestor index in |arena| from the
// function model's preserved loop forest. Construction takes O(16L) time and
// storage for L candidate loops. Acyclic schedules perform no allocations and
// return an empty analysis.
iree_status_t loom_amdgpu_wait_loop_analysis_initialize(
    const loom_low_schedule_table_t* schedule, iree_arena_allocator_t* arena,
    loom_amdgpu_wait_loop_analysis_t* out_analysis);

// Returns the dedicated preheader that safely dominates |consumer_node| and
// follows |producer_node|, choosing the outermost eligible loop. Returns
// UINT16_MAX when the dependency must remain at its exact consumer. Queries
// perform at most 16 ancestor probes independent of loop nesting depth.
uint16_t loom_amdgpu_wait_loop_analysis_preheader(
    const loom_amdgpu_wait_loop_analysis_t* analysis, uint32_t producer_node,
    uint32_t consumer_node);

// Returns the innermost supported canonical loop whose cyclic execution places
// |producer_node| after |consumer_node| in schedule order. The returned
// interval remains owned by the schedule. Returns NULL when the dependency is
// forward within one iteration or the enclosing loop shape is not eligible for
// target wait relocation.
const loom_cfg_loop_interval_t* loom_amdgpu_wait_loop_analysis_cyclic_interval(
    const loom_amdgpu_wait_loop_analysis_t* analysis, uint32_t producer_node,
    uint32_t consumer_node);

// Derives stable incoming counter epochs for supported canonical loop blocks.
// The returned dense table is indexed by block then AMDGPU counter slot and is
// owned by |arena|. Unsupported counters and loop shapes have zero records and
// retain conservative full-drain behavior in the caller.
iree_status_t loom_amdgpu_wait_loop_analysis_build_cyclic_frontiers(
    const loom_amdgpu_wait_loop_analysis_t* analysis,
    const loom_amdgpu_wait_loop_node_t* nodes,
    const uint32_t* first_dependency_by_consumer,
    const loom_amdgpu_wait_loop_dependency_t* dependencies,
    iree_host_size_t dependency_count, iree_arena_allocator_t* arena,
    const loom_amdgpu_wait_loop_cyclic_frontier_t** out_frontiers);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_LOOP_H_
