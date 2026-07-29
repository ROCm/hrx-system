// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU matrix/vector coexecution release tracking.
//
// Qualified processors retain matrix source and result VGPRs for a bounded
// number of vector issue slots. Final native packetization feeds this component
// once while it is already deciding VOPD pairs. The component then propagates
// sparse physical release frontiers across the CFG. Fixed wait-state planning
// consumes those retained facts while traversing the final packet stream and
// turns residual slots into its unified V_NOP action.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_MATRIX_COEXECUTION_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_MATRIX_COEXECUTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/packet.h"
#include "loom/target/arch/amdgpu/planning/structural_packet.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_matrix_coexecution_t
    loom_amdgpu_matrix_coexecution_t;

// Strongest matrix coexecution dependency observed for one consumer.
typedef struct loom_amdgpu_matrix_coexecution_match_t {
  // Schedule node that opened the release window, or
  // LOOM_LOW_SCHEDULE_NODE_NONE when the dependency crossed unresolved CFG
  // state.
  uint32_t producer_node;
  // Original vector issue distance for the selected dependency.
  uint16_t required_issue_count;
  // Vector issue progress observed after the selected producer.
  uint16_t observed_issue_count;
  // Additional V_NOP issue slots required before the consumer.
  uint16_t residual_issue_count;
  // True when the consumer is another matrix packet.
  bool matrix_consumer;
} loom_amdgpu_matrix_coexecution_match_t;

// Allocates coexecution state for a selected processor profile. Profiles
// without qualified rules and schedules without a coexecution-source resource
// return a NULL component in O(1). The returned component is transient and must
// span final native packetization and fixed wait-state planning.
iree_status_t loom_amdgpu_matrix_coexecution_allocate(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    loom_amdgpu_matrix_coexecution_profile_t profile,
    iree_arena_allocator_t* arena,
    loom_amdgpu_matrix_coexecution_t** out_coexecution);

// Begins the existing final-packet traversal for one block.
void loom_amdgpu_matrix_coexecution_begin_static_block(
    loom_amdgpu_matrix_coexecution_t* coexecution, uint16_t block_index);

// Commits one final unpaired packet and its already-computed structural facts
// to the current block transfer summary.
void loom_amdgpu_matrix_coexecution_commit_static_packet(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_structural_packet_info_t* structural_info);

// Commits two final VOPD components as one vector issue slot. Matrix-source
// packets are never legal VOPD components.
void loom_amdgpu_matrix_coexecution_commit_static_vopd_pair(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    const loom_low_packet_view_t* first_packet,
    const loom_low_packet_view_t* second_packet);

// Publishes the sparse local outgoing frontier for the current block.
iree_status_t loom_amdgpu_matrix_coexecution_end_static_block(
    loom_amdgpu_matrix_coexecution_t* coexecution);

// Propagates the retained sparse block summaries through the scheduled CFG.
iree_status_t loom_amdgpu_matrix_coexecution_finalize_static(
    loom_amdgpu_matrix_coexecution_t* coexecution);

// Begins fixed wait-state planning for one block using refined outgoing state
// from processed predecessors and conservative state from unresolved
// predecessors.
void loom_amdgpu_matrix_coexecution_begin_block(
    loom_amdgpu_matrix_coexecution_t* coexecution, uint16_t block_index);

// Queries coexecution hazards for one packet and returns the packet's vector
// issue contribution, saturated at the profile's release-window bound.
// |structural_info| is required exactly when |packet| has no descriptor.
void loom_amdgpu_matrix_coexecution_inspect_packet(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_structural_packet_info_t* structural_info,
    loom_amdgpu_matrix_coexecution_match_t* inout_match,
    uint16_t* out_vector_issue_count);

// Advances the active release frontier by inserted vector issue slots.
void loom_amdgpu_matrix_coexecution_advance(
    loom_amdgpu_matrix_coexecution_t* coexecution, uint16_t issue_count);

// Commits one non-VOPD packet after all insertions before it have been applied.
void loom_amdgpu_matrix_coexecution_commit_packet(
    loom_amdgpu_matrix_coexecution_t* coexecution,
    const loom_low_packet_view_t* packet, uint16_t vector_issue_count);

// Commits a VOPD pair as one vector issue after both components were queried
// against the same pre-packet frontier.
void loom_amdgpu_matrix_coexecution_commit_vopd_pair(
    loom_amdgpu_matrix_coexecution_t* coexecution);

// Publishes refined sparse outgoing state for the active block.
iree_status_t loom_amdgpu_matrix_coexecution_end_block(
    loom_amdgpu_matrix_coexecution_t* coexecution);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_MATRIX_COEXECUTION_H_
