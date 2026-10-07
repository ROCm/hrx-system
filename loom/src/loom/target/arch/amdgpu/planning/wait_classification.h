// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU wait-node state classified from a completed low schedule.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_CLASSIFICATION_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_CLASSIFICATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/table.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/planning/wait_completion.h"
#include "loom/target/arch/amdgpu/planning/wait_frontier.h"
#include "loom/target/arch/amdgpu/planning/wait_packet_tables.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_wait_node_state_flag_bits_e {
  // Structural node forwards wait dependencies to its users.
  LOOM_AMDGPU_WAIT_NODE_STATE_FORWARDS_DEPENDENCIES = 1u << 0,
  // Node has a counter effect without a concrete counter id.
  LOOM_AMDGPU_WAIT_NODE_STATE_GENERIC_COUNTER_EFFECT = 1u << 1,
  // Node has a dependency-participating memory read effect.
  LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_READ = 1u << 2,
  // Node has a dependency-participating memory write effect.
  LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_WRITE = 1u << 3,
  // Node has a memory read effect using the descriptor's completion hazards.
  LOOM_AMDGPU_WAIT_NODE_STATE_DEFAULT_DEPENDENCY_READ = 1u << 4,
  // Node has a memory write effect using the descriptor's completion hazards.
  LOOM_AMDGPU_WAIT_NODE_STATE_DEFAULT_DEPENDENCY_WRITE = 1u << 5,
  // Node issues on the vector ALU.
  LOOM_AMDGPU_WAIT_NODE_STATE_USES_VECTOR_ALU = 1u << 6,
  // Node issues on the scalar ALU.
  LOOM_AMDGPU_WAIT_NODE_STATE_USES_SCALAR_ALU = 1u << 7,
  // Node is a transcendental VALU packet.
  LOOM_AMDGPU_WAIT_NODE_STATE_TRANSCENDENTAL = 1u << 8,
  // Node materializes its scheduled SSA results into physical locations.
  LOOM_AMDGPU_WAIT_NODE_STATE_MATERIALIZES_RESULTS = 1u << 9,
  // Node produces one gfx125x VMEM XCNT event.
  LOOM_AMDGPU_WAIT_NODE_STATE_XCNT_VMEM_PRODUCER = 1u << 10,
  // Node produces one gfx125x SMEM XCNT event.
  LOOM_AMDGPU_WAIT_NODE_STATE_XCNT_SMEM_PRODUCER = 1u << 11,
  // Node writes architectural EXEC state.
  LOOM_AMDGPU_WAIT_NODE_STATE_WRITES_EXEC = 1u << 12,
  // The emitted packet implicitly drains gfx125x XCNT before it executes.
  LOOM_AMDGPU_WAIT_NODE_STATE_XCNT_IMPLICIT_DRAIN = 1u << 13,
  // Counter-only packet whose payload indexes immutable decoded bounds.
  LOOM_AMDGPU_WAIT_NODE_STATE_EXPLICIT_WAIT = 1u << 14,
  // Structural packet emits no native instructions or physical moves.
  LOOM_AMDGPU_WAIT_NODE_STATE_ZERO_NATIVE_WORK = 1u << 15,
  // Generic-address completion can retire early in either memory domain.
  LOOM_AMDGPU_WAIT_NODE_STATE_UNORDERED_FLAT_COMPLETION = 1u << 16,
  // An asynchronous result preserves the tied source's disjoint register part.
  LOOM_AMDGPU_WAIT_NODE_STATE_PRESERVES_RESULT_PART = 1u << 17,
  // GFX12.0 ordinary SYSTEM store drains prior asynchronous memory work.
  LOOM_AMDGPU_WAIT_NODE_STATE_SYSTEM_SCOPE_STORE_DRAIN = 1u << 18,
} loom_amdgpu_wait_node_state_flag_bits_t;
typedef uint32_t loom_amdgpu_wait_node_state_flags_t;

// Compact target wait classification for one schedule node.
typedef struct loom_amdgpu_wait_node_state_t {
  // Classification flags for this schedule node.
  loom_amdgpu_wait_node_state_flags_t flags;
  // Disjoint payload roles: explicit waits never produce counter work.
  union {
    // One-based mutable producer row, or zero for other non-wait nodes.
    uint32_t producer_state_ordinal;
    // Immutable bounds row for an EXPLICIT_WAIT node.
    uint32_t wait_bounds_index;
  } state;
  // Counters drained by explicit counter effects on this node.
  loom_amdgpu_wait_counter_mask_t explicit_wait_counter_mask;
  // Counters implicitly drained by the emitted packet.
  loom_amdgpu_wait_counter_mask_t implicit_wait_counter_mask;
  // Counters that must be drained before this barrier node executes.
  loom_amdgpu_wait_counter_mask_t barrier_counter_mask;
} loom_amdgpu_wait_node_state_t;

static_assert(sizeof(loom_amdgpu_wait_node_state_t) == 12,
              "classified wait-node state must remain compact");

// Target wait state indexed by the completed low schedule.
typedef struct loom_amdgpu_wait_classification_t {
  // Compact target state indexed by schedule node. Classification flags and
  // decoded effects are stable; action planning may add path-resolved implicit
  // counter drains.
  loom_amdgpu_wait_node_state_t* node_states;
  // Memory facts indexed by schedule node. Action planning records locally
  // completed producer prefixes in the same compact rows.
  loom_amdgpu_wait_frontier_node_t* frontier_nodes;
  // Counter facts indexed by schedule node. Classification owns the producer,
  // write, hazard, and workgroup fields; dependency completion analysis
  // finalizes reset and block-exit fields before action planning begins.
  loom_amdgpu_wait_completion_node_t* completion_nodes;
  // Decoded explicit-wait bounds indexed by node-state payload.
  loom_amdgpu_wait_packet_bounds_t* wait_bounds;
  // Number of populated rows in |wait_bounds|.
  iree_host_size_t wait_bounds_count;
  // Number of schedule nodes that require mutable producer state.
  iree_host_size_t producer_state_count;
  // Number of structural nodes that forward readiness dependencies.
  iree_host_size_t forwarding_node_count;
  // Number of nodes producing tracked transcendental results.
  iree_host_size_t trans_result_node_count;
  // Number of nodes producing asynchronous VMEM register results.
  iree_host_size_t vmem_result_node_count;
} loom_amdgpu_wait_classification_t;

// Initializes target wait state once from |schedule| and |allocation|. All
// returned tables are owned by |arena|. Classification flags, decoded bounds,
// and row counts remain stable; the documented completion fields are refined
// by their owning downstream analyses. Mutable per-producer epochs use the
// returned producer ordinals but are allocated separately by the action owner.
iree_status_t loom_amdgpu_wait_classification_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_processor_properties_t* processor_properties,
    const loom_amdgpu_wait_packet_target_t* wait_packet_target,
    iree_arena_allocator_t* arena,
    loom_amdgpu_wait_classification_t* IREE_RESTRICT out_classification);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_CLASSIFICATION_H_
