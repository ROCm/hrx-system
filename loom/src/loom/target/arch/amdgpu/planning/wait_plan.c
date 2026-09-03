// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_plan.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/bitfield.h"
#include "iree/base/internal/math.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/allocation/storage_lease.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/packet_hazard_plan_json.h"
#include "loom/ir/ir.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/planning/structural_packet.h"
#include "loom/target/arch/amdgpu/planning/wait_frontier.h"
#include "loom/target/arch/amdgpu/planning/wait_loop.h"
#include "loom/target/arch/amdgpu/planning/wait_packet_tables.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/segmented_storage.h"

// Target payload size for lazily appended wait-action segments. Segments stay
// small enough to share normal compiler workspace blocks with other planning
// state while avoiding repeated copies as the final action count becomes known.
#define LOOM_AMDGPU_WAIT_PLAN_ACTION_SEGMENT_BYTE_LENGTH (4u * 1024u)

// Number of wait actions stored in each append segment.
#define LOOM_AMDGPU_WAIT_PLAN_ACTIONS_PER_SEGMENT     \
  (LOOM_AMDGPU_WAIT_PLAN_ACTION_SEGMENT_BYTE_LENGTH / \
   sizeof(loom_amdgpu_wait_plan_action_t))

static_assert(LOOM_AMDGPU_WAIT_PLAN_ACTIONS_PER_SEGMENT > 0,
              "wait action must fit in one append segment");

// One stable segment of wait actions populated before exact finalization.
typedef struct loom_amdgpu_wait_plan_action_segment_t {
  // Wait actions in append order.
  loom_amdgpu_wait_plan_action_t
      actions[LOOM_AMDGPU_WAIT_PLAN_ACTIONS_PER_SEGMENT];
} loom_amdgpu_wait_plan_action_segment_t;

static_assert(sizeof(loom_amdgpu_wait_plan_action_segment_t) <=
                  LOOM_AMDGPU_WAIT_PLAN_ACTION_SEGMENT_BYTE_LENGTH,
              "wait action segment exceeds its byte budget");

typedef enum loom_amdgpu_wait_node_state_flag_bits_e {
  // Structural node forwards wait dependencies to its users.
  LOOM_AMDGPU_WAIT_NODE_STATE_FORWARDS_DEPENDENCIES = 1u << 0,
  // Node has a counter effect without a concrete counter id.
  LOOM_AMDGPU_WAIT_NODE_STATE_GENERIC_COUNTER_EFFECT = 1u << 1,
  // Node has a dependency-participating memory read effect.
  LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_READ = 1u << 2,
  // Node has a dependency-participating memory write effect.
  LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_WRITE = 1u << 3,
  // Node has a memory read effect using the target's default read counter.
  LOOM_AMDGPU_WAIT_NODE_STATE_DEFAULT_DEPENDENCY_READ = 1u << 4,
  // Node has a memory write effect using the target's default write counter.
  LOOM_AMDGPU_WAIT_NODE_STATE_DEFAULT_DEPENDENCY_WRITE = 1u << 5,
  // Node has a workgroup write effect using the target's default write counter.
  LOOM_AMDGPU_WAIT_NODE_STATE_DEFAULT_WORKGROUP_WRITE = 1u << 6,
  // Node issues on the vector ALU.
  LOOM_AMDGPU_WAIT_NODE_STATE_USES_VECTOR_ALU = 1u << 7,
  // Node issues on the scalar ALU.
  LOOM_AMDGPU_WAIT_NODE_STATE_USES_SCALAR_ALU = 1u << 8,
  // Node is a transcendental VALU packet.
  LOOM_AMDGPU_WAIT_NODE_STATE_TRANSCENDENTAL = 1u << 9,
  // Node materializes its scheduled SSA results into physical locations.
  LOOM_AMDGPU_WAIT_NODE_STATE_MATERIALIZES_RESULTS = 1u << 10,
  // Node produces one gfx125x VMEM XCNT event.
  LOOM_AMDGPU_WAIT_NODE_STATE_XCNT_VMEM_PRODUCER = 1u << 11,
  // Node produces one gfx125x SMEM XCNT event.
  LOOM_AMDGPU_WAIT_NODE_STATE_XCNT_SMEM_PRODUCER = 1u << 12,
  // Node writes architectural EXEC state.
  LOOM_AMDGPU_WAIT_NODE_STATE_WRITES_EXEC = 1u << 13,
  // The emitted packet implicitly drains gfx125x XCNT before it executes.
  LOOM_AMDGPU_WAIT_NODE_STATE_XCNT_IMPLICIT_DRAIN = 1u << 14,
} loom_amdgpu_wait_node_state_flag_bits_t;
typedef uint16_t loom_amdgpu_wait_node_state_flags_t;

typedef struct loom_amdgpu_wait_node_state_t {
  // Classification flags for this schedule node.
  loom_amdgpu_wait_node_state_flags_t flags;
  // Counters observed on WAIT_COUNTER hazard rows for this node.
  uint32_t hazard_counter_mask;
  // Counters drained by explicit counter effects on this node.
  uint32_t explicit_wait_counter_mask;
  // Counters produced by RDNA TRANS result hazards on this node.
  uint32_t trans_result_counter_mask;
  // Counters produced only to retain issued source storage on this node.
  uint32_t source_counter_mask;
  // Counters implicitly drained by the emitted packet.
  uint32_t implicit_wait_counter_mask;
  // Write counters whose effects are visible to workgroup-memory barriers.
  uint32_t workgroup_write_counter_mask;
  // Epoch for each counter produced by this node.
  uint32_t produced_counter_epoch[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
  // Monotonic producer position in the counter epoch. This is only meaningful
  // within |produced_counter_epoch|.
  uint32_t produced_counter_position[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
  // Counters that must be drained before this barrier node executes.
  uint32_t barrier_counter_mask;
  // Workgroup-memory write counters drained before this barrier executes.
  uint32_t workgroup_barrier_counter_mask;
} loom_amdgpu_wait_node_state_t;

typedef struct loom_amdgpu_wait_block_arg_source_t {
  // Source value ordinal passed along the incoming low.br edge.
  loom_value_ordinal_t source_ordinal;
  // Next source record for the same destination block argument.
  uint32_t next_source;
} loom_amdgpu_wait_block_arg_source_t;

typedef enum loom_amdgpu_wait_trans_result_vgpr_flag_bits_e {
  LOOM_AMDGPU_WAIT_TRANS_RESULT_VGPR_FLAG_VALID = 1u << 0,
} loom_amdgpu_wait_trans_result_vgpr_flag_bits_t;
typedef uint8_t loom_amdgpu_wait_trans_result_vgpr_flags_t;

typedef struct loom_amdgpu_wait_trans_result_vgpr_t {
  // Active-state flags for this physical VGPR.
  loom_amdgpu_wait_trans_result_vgpr_flags_t flags;
  // TRANS node whose result is still within the RDNA va_vdst hazard window.
  uint32_t producer_node;
  // ALU counter epoch when the TRANS packet was issued.
  uint32_t counter_epoch;
  // Block epoch when this VGPR state was recorded.
  uint64_t block_epoch;
  // Number of VALU packets since the TRANS producer, saturated past the limit.
  uint8_t valu_interval;
  // Number of TRANS packets since the TRANS producer, saturated past the limit.
  uint8_t trans_interval;
} loom_amdgpu_wait_trans_result_vgpr_t;

typedef enum loom_amdgpu_wait_sgpr_read_flag_bits_e {
  LOOM_AMDGPU_WAIT_SGPR_READ_FLAG_TRACKED = 1u << 0,
  LOOM_AMDGPU_WAIT_SGPR_READ_FLAG_SALU_HAZARD = 1u << 1,
  LOOM_AMDGPU_WAIT_SGPR_READ_FLAG_VALU_HAZARD = 1u << 2,
} loom_amdgpu_wait_sgpr_read_flag_bits_t;
typedef uint8_t loom_amdgpu_wait_sgpr_read_flags_t;

typedef struct loom_amdgpu_wait_sgpr_read_register_t {
  // Tracking and active hazard bits for this physical SGPR unit.
  loom_amdgpu_wait_sgpr_read_flags_t flags;
  // ALU node whose SGPR write forced the active hazard.
  uint32_t producer_node;
} loom_amdgpu_wait_sgpr_read_register_t;

typedef enum loom_amdgpu_wait_xcnt_group_e {
  LOOM_AMDGPU_WAIT_XCNT_GROUP_NONE = 0,
  LOOM_AMDGPU_WAIT_XCNT_GROUP_VMEM = 1,
  LOOM_AMDGPU_WAIT_XCNT_GROUP_SMEM = 2,
} loom_amdgpu_wait_xcnt_group_t;

static_assert((uint32_t)LOOM_AMDGPU_WAIT_XCNT_GROUP_VMEM ==
                  (uint32_t)LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_VMEM,
              "XCNT VMEM group encodings must agree with the CFG frontier");
static_assert((uint32_t)LOOM_AMDGPU_WAIT_XCNT_GROUP_SMEM ==
                  (uint32_t)LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_SMEM,
              "XCNT SMEM group encodings must agree with the CFG frontier");

typedef struct loom_amdgpu_wait_plan_builder_t {
  // Schedule table being analyzed.
  const loom_low_schedule_table_t* schedule;
  // Optional physical assignment table for post-allocation hazards.
  const loom_low_allocation_table_t* allocation;
  // Arena that owns tables retained by the completed plan.
  iree_arena_allocator_t* arena;
  // Arena that owns builder state discarded after plan construction.
  iree_arena_allocator_t* transient_arena;
  // Processor properties selected by the low target, or NULL if unavailable.
  const loom_amdgpu_processor_properties_t* processor_properties;
  // Generated wait-packet descriptors selected by the low target.
  loom_amdgpu_wait_packet_target_t wait_packet_target;
  // Per-node counter classification.
  loom_amdgpu_wait_node_state_t* node_states;
  // Producer node for each scheduled SSA value ordinal.
  uint32_t* producer_nodes;
  // Per-node memory counter and address-space classification.
  loom_amdgpu_wait_frontier_node_t* frontier_nodes;
  // Per-node immutable counter facts consumed by canonical-loop analysis.
  loom_amdgpu_wait_loop_node_t* loop_nodes;
  // Bounded cross-block wait state.
  loom_amdgpu_wait_frontier_t frontier;
  // Target eligibility and ancestor index over canonical schedule loops.
  loom_amdgpu_wait_loop_analysis_t loop_analysis;
  // Counter masks drained for a producer in the current block epoch.
  uint32_t* current_block_drained_counter_masks;
  // Block epoch for each producer-local drained counter mask.
  uint64_t* current_block_drained_epochs;
  // First relevant counter dependency link per consumer node.
  uint32_t* first_dependency_link_by_consumer;
  // Relevant counter dependency links.
  loom_amdgpu_wait_loop_dependency_t* dependency_links;
  // First incoming low.br source per destination block-argument value ordinal.
  uint32_t* first_block_arg_source_by_value;
  // Incoming low.br source records for non-entry block arguments.
  loom_amdgpu_wait_block_arg_source_t* block_arg_sources;
  // First SSA dependency indexed by loop-entry block and counter.
  uint32_t* loop_entry_dependency_links;
  // Counter classes fully drained at each planned loop-entry block.
  uint32_t* loop_entry_drain_counter_masks;
  // Derived incoming counter epochs indexed by loop block and counter.
  const loom_amdgpu_wait_loop_cyclic_frontier_t* cyclic_frontiers;
  // Storage-release actions grouped by insertion node.
  loom_low_storage_release_action_index_t storage_release_action_index;
  // DFS visit epoch per value while forwarding SSA wait dependencies.
  uint32_t* dependency_visit_epochs;
  // Explicit worklist used with |dependency_visit_epochs|.
  loom_value_ordinal_t* dependency_visit_worklist;
  // Number of populated dependency links.
  iree_host_size_t dependency_link_count;
  // Allocated dependency link capacity.
  iree_host_size_t dependency_link_capacity;
  // Number of populated block-argument source records.
  iree_host_size_t block_arg_source_count;
  // Number of nodes that forward dependency-producing operands to users.
  iree_host_size_t forwarding_node_count;
  // Number of nodes producing transcendental results tracked by depctr.
  iree_host_size_t trans_result_node_count;
  // Number of nodes producing asynchronous VMEM register results.
  iree_host_size_t vmem_result_node_count;
  // Monotonic DFS epoch for dependency forwarding.
  uint32_t dependency_visit_epoch;
  // Sparse append state used while the final action count is unknown.
  struct {
    // Stable segments populated in action order.
    loom_segmented_storage_t segments;
    // Current append segment, or NULL before the first action.
    loom_amdgpu_wait_plan_action_segment_t* tail;
    // Number of populated actions in |tail|.
    iree_host_size_t tail_count;
  } action_stream;
  // Final contiguous output action rows.
  loom_amdgpu_wait_plan_action_t* actions;
  // Number of populated action rows.
  iree_host_size_t action_count;
  // Cursor into packet-ordered actions while projecting residual hazards.
  iree_host_size_t hazard_action_cursor;
  // Packet index of the next residual hazard, or IREE_HOST_SIZE_MAX.
  iree_host_size_t next_hazard_packet_index;
  // Exact number of canonical packet-progress rows.
  iree_host_size_t progress_event_count;
  // Exact number of target-owned canonical packet-hazard rows.
  iree_host_size_t hazard_event_count;
  // Canonical packet-progress table populated after wait actions are known.
  loom_low_packet_progress_table_t progress;
  // Canonical packet hazard table populated after wait actions are known.
  loom_low_packet_hazard_plan_t hazard_plan;
  // Current epoch per wait counter.
  uint32_t counter_epochs[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
  // Oldest producer positions already known complete in the current epoch.
  uint32_t completed_position_counts[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
  // Current block epoch for lazy invalidation of physical-register state.
  uint64_t block_epoch;
  // Counters fully drained earlier in the current straight-line block.
  uint32_t current_block_full_drain_counter_mask;
  // Translation group currently represented by outstanding gfx125x XCNT
  // events. Hardware implicitly drains XCNT when this group changes.
  loom_amdgpu_wait_xcnt_group_t xcnt_group;
  // Physical register-file extents derived from the allocation table.
  struct {
    // Number of assigned VGPR units.
    iree_host_size_t vgpr_count;
    // Number of assigned AGPR units.
    iree_host_size_t agpr_count;
    // Number of assigned SGPR units.
    iree_host_size_t sgpr_count;
  } physical_registers;
  // Outstanding packet count per wait counter.
  uint32_t outstanding_counts[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
  // Outstanding packet count per wait counter for memory writes.
  uint32_t outstanding_write_counts[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
  // Outstanding packet count per wait counter for workgroup memory writes.
  uint32_t
      outstanding_workgroup_write_counts[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
  // Per-physical-VGPR state for outstanding RDNA TRANS result hazards.
  loom_amdgpu_wait_trans_result_vgpr_t* trans_result_vgprs;
  // Number of entries in |trans_result_vgprs|.
  iree_host_size_t trans_result_vgpr_count;
  // Number of currently live TRANS result VGPR records.
  iree_host_size_t active_trans_result_vgpr_count;
  // Per-physical-SGPR state for GFX12 VALU/SALU SGPR-read hazards.
  loom_amdgpu_wait_sgpr_read_register_t* sgpr_read_registers;
  // Number of entries in |sgpr_read_registers|.
  iree_host_size_t sgpr_read_register_count;
} loom_amdgpu_wait_plan_builder_t;

iree_string_view_t loom_amdgpu_wait_counter_name(uint16_t counter_id) {
  switch (counter_id) {
    case LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD:
      return IREE_SV("vmem_load");
    case LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE:
      return IREE_SV("vmem_store");
    case LOOM_AMDGPU_WAIT_COUNTER_LDS:
      return IREE_SV("lds");
    case LOOM_AMDGPU_WAIT_COUNTER_SMEM:
      return IREE_SV("smem");
    case LOOM_AMDGPU_WAIT_COUNTER_ALU:
      return IREE_SV("alu");
    case LOOM_AMDGPU_WAIT_COUNTER_TENSOR:
      return IREE_SV("tensor");
    case LOOM_AMDGPU_WAIT_COUNTER_ASYNC:
      return IREE_SV("async");
    case LOOM_AMDGPU_WAIT_COUNTER_X:
      return IREE_SV("x");
    case LOOM_AMDGPU_WAIT_COUNTER_NONE:
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_amdgpu_wait_counter_progress_class_name(
    uint16_t counter_id) {
  switch (counter_id) {
    case LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD:
      return IREE_SV("amdgpu.vmem_load");
    case LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE:
      return IREE_SV("amdgpu.vmem_store");
    case LOOM_AMDGPU_WAIT_COUNTER_LDS:
      return IREE_SV("amdgpu.lds");
    case LOOM_AMDGPU_WAIT_COUNTER_SMEM:
      return IREE_SV("amdgpu.smem");
    case LOOM_AMDGPU_WAIT_COUNTER_ALU:
      return IREE_SV("amdgpu.alu");
    case LOOM_AMDGPU_WAIT_COUNTER_TENSOR:
      return IREE_SV("amdgpu.tensor");
    case LOOM_AMDGPU_WAIT_COUNTER_ASYNC:
      return IREE_SV("amdgpu.async");
    case LOOM_AMDGPU_WAIT_COUNTER_X:
      return IREE_SV("amdgpu.x");
    case LOOM_AMDGPU_WAIT_COUNTER_NONE:
    default:
      return IREE_SV("amdgpu.unknown");
  }
}

iree_string_view_t loom_amdgpu_wait_plan_reason_name(
    loom_amdgpu_wait_plan_reason_t reason) {
  switch (reason) {
    case LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET:
      return IREE_SV("amdgpu.explicit_packet");
    case LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE:
      return IREE_SV("amdgpu.ssa_use");
    case LOOM_AMDGPU_WAIT_PLAN_REASON_BARRIER:
      return IREE_SV("amdgpu.barrier");
    case LOOM_AMDGPU_WAIT_PLAN_REASON_READ_RESULT_REUSE:
      return IREE_SV("amdgpu.read_result_reuse");
    case LOOM_AMDGPU_WAIT_PLAN_REASON_TRANS_RESULT_USE:
      return IREE_SV("amdgpu.trans_result_use");
    case LOOM_AMDGPU_WAIT_PLAN_REASON_VALU_SGPR_READ:
      return IREE_SV("amdgpu.valu_sgpr_read");
    case LOOM_AMDGPU_WAIT_PLAN_REASON_MEMORY_EFFECT:
      return IREE_SV("amdgpu.memory_effect");
    case LOOM_AMDGPU_WAIT_PLAN_REASON_PROGRAM_EXIT:
      return IREE_SV("amdgpu.program_exit");
    case LOOM_AMDGPU_WAIT_PLAN_REASON_MEMORY_SOURCE_REUSE:
      return IREE_SV("amdgpu.memory_source_reuse");
    case LOOM_AMDGPU_WAIT_PLAN_REASON_XCNT_EXEC_REUSE:
      return IREE_SV("amdgpu.xcnt_exec_reuse");
    case LOOM_AMDGPU_WAIT_PLAN_REASON_LOOP_ENTRY_DERIVED_SSA_USE:
      return IREE_SV("amdgpu.loop_entry_derived_ssa_use");
    case LOOM_AMDGPU_WAIT_PLAN_REASON_LOOP_ENTRY_CONSERVATIVE_SSA_USE:
      return IREE_SV("amdgpu.loop_entry_conservative_ssa_use");
    case LOOM_AMDGPU_WAIT_PLAN_REASON_LOOP_CARRIED_DERIVED_SSA_USE:
      return IREE_SV("amdgpu.loop_carried_derived_ssa_use");
    case LOOM_AMDGPU_WAIT_PLAN_REASON_LOOP_CARRIED_CONSERVATIVE_SSA_USE:
      return IREE_SV("amdgpu.loop_carried_conservative_ssa_use");
    case LOOM_AMDGPU_WAIT_PLAN_REASON_UNKNOWN:
    default:
      return IREE_SV("amdgpu.unknown");
  }
}

iree_string_view_t loom_amdgpu_wait_plan_residual_action_name(
    uint16_t action_id) {
  switch (action_id) {
    case LOOM_AMDGPU_WAIT_PLAN_RESIDUAL_ACTION_WAIT_PACKET:
      return IREE_SV("amdgpu.wait_packet");
    default:
      return IREE_SV("unknown");
  }
}

static bool loom_amdgpu_wait_effect_is_dependency_memory(
    const loom_low_schedule_effect_use_t* effect_use) {
  if ((effect_use->effect_flags & LOOM_LOW_EFFECT_FLAG_DEPENDENCY) == 0) {
    return false;
  }
  switch (effect_use->memory_space) {
    case LOOM_LOW_MEMORY_SPACE_GENERIC:
    case LOOM_LOW_MEMORY_SPACE_GLOBAL:
    case LOOM_LOW_MEMORY_SPACE_STACK:
    case LOOM_LOW_MEMORY_SPACE_WORKGROUP:
      return true;
    default:
      return false;
  }
}

static uint32_t loom_amdgpu_wait_effect_counter_mask(
    const loom_low_schedule_effect_use_t* effect) {
  if (effect->counter_id == LOOM_AMDGPU_WAIT_COUNTER_NONE) {
    return 0;
  }
  return loom_amdgpu_wait_counter_mask(effect->counter_id);
}

static bool loom_amdgpu_wait_plan_reason_has_consumer(
    loom_amdgpu_wait_plan_reason_t reason) {
  switch (reason) {
    case LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE:
    case LOOM_AMDGPU_WAIT_PLAN_REASON_READ_RESULT_REUSE:
    case LOOM_AMDGPU_WAIT_PLAN_REASON_TRANS_RESULT_USE:
    case LOOM_AMDGPU_WAIT_PLAN_REASON_VALU_SGPR_READ:
    case LOOM_AMDGPU_WAIT_PLAN_REASON_MEMORY_EFFECT:
    case LOOM_AMDGPU_WAIT_PLAN_REASON_PROGRAM_EXIT:
    case LOOM_AMDGPU_WAIT_PLAN_REASON_MEMORY_SOURCE_REUSE:
    case LOOM_AMDGPU_WAIT_PLAN_REASON_XCNT_EXEC_REUSE:
    case LOOM_AMDGPU_WAIT_PLAN_REASON_LOOP_ENTRY_DERIVED_SSA_USE:
    case LOOM_AMDGPU_WAIT_PLAN_REASON_LOOP_ENTRY_CONSERVATIVE_SSA_USE:
    case LOOM_AMDGPU_WAIT_PLAN_REASON_LOOP_CARRIED_DERIVED_SSA_USE:
    case LOOM_AMDGPU_WAIT_PLAN_REASON_LOOP_CARRIED_CONSERVATIVE_SSA_USE:
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_wait_plan_reason_is_storage_release(
    loom_amdgpu_wait_plan_reason_t reason) {
  switch (reason) {
    case LOOM_AMDGPU_WAIT_PLAN_REASON_READ_RESULT_REUSE:
    case LOOM_AMDGPU_WAIT_PLAN_REASON_MEMORY_SOURCE_REUSE:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_amdgpu_wait_plan_build_storage_release_action_index(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_allocation_table_t* allocation = builder->allocation;
  if (allocation == NULL || allocation->storage_release_action_count == 0) {
    return iree_ok_status();
  }
  const loom_low_schedule_table_t* schedule = builder->schedule;
  return loom_low_storage_release_action_index_build(
      allocation->storage_release_actions,
      allocation->storage_release_action_count,
      LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_BY_INSERTION_NODE,
      schedule->node_count, builder->transient_arena,
      &builder->storage_release_action_index);
}

static iree_status_t loom_amdgpu_wait_plan_allocate(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  if (schedule->node_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->transient_arena, schedule->node_count,
        sizeof(*builder->node_states), (void**)&builder->node_states));
    memset(builder->node_states, 0,
           schedule->node_count * sizeof(*builder->node_states));

    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->transient_arena, schedule->node_count,
        sizeof(*builder->frontier_nodes), (void**)&builder->frontier_nodes));
    memset(builder->frontier_nodes, 0,
           schedule->node_count * sizeof(*builder->frontier_nodes));

    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->transient_arena, schedule->node_count,
        sizeof(*builder->loop_nodes), (void**)&builder->loop_nodes));
    memset(builder->loop_nodes, 0,
           schedule->node_count * sizeof(*builder->loop_nodes));

    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->transient_arena, schedule->node_count,
        sizeof(*builder->current_block_drained_counter_masks),
        (void**)&builder->current_block_drained_counter_masks));
    memset(builder->current_block_drained_counter_masks, 0,
           schedule->node_count *
               sizeof(*builder->current_block_drained_counter_masks));

    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->transient_arena, schedule->node_count,
        sizeof(*builder->current_block_drained_epochs),
        (void**)&builder->current_block_drained_epochs));
    memset(
        builder->current_block_drained_epochs, 0,
        schedule->node_count * sizeof(*builder->current_block_drained_epochs));

    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->transient_arena, schedule->node_count,
        sizeof(*builder->first_dependency_link_by_consumer),
        (void**)&builder->first_dependency_link_by_consumer));
    for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
      builder->first_dependency_link_by_consumer[i] =
          LOOM_LOW_SCHEDULE_NODE_NONE;
    }
  }
  return loom_amdgpu_wait_plan_build_storage_release_action_index(builder);
}

static const loom_low_allocation_assignment_t* loom_amdgpu_wait_plan_assignment(
    const loom_low_allocation_table_t* allocation, loom_value_id_t value_id) {
  if (allocation == NULL || value_id == LOOM_VALUE_ID_INVALID) {
    return NULL;
  }
  return loom_low_allocation_try_map_active_value_assignment(allocation,
                                                             value_id, NULL);
}

static loom_amdgpu_structural_packet_flags_t
loom_amdgpu_wait_plan_classify_structural_node(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  if (node->kind != LOOM_LOW_SCHEDULE_NODE_STRUCTURAL || node->op == NULL) {
    return 0;
  }
  return loom_amdgpu_structural_packet_analyze(
             builder->schedule, builder->allocation, node,
             LOOM_AMDGPU_STRUCTURAL_PACKET_ANALYSIS_FLAG_REQUIRE_ALLOCATION)
      .flags;
}

static bool loom_amdgpu_wait_plan_node_forwards_dependencies(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  return node_index < builder->schedule->node_count &&
         iree_any_bit_set(builder->node_states[node_index].flags,
                          LOOM_AMDGPU_WAIT_NODE_STATE_FORWARDS_DEPENDENCIES);
}

static bool loom_amdgpu_wait_plan_needs_trans_result_state(
    const loom_amdgpu_wait_plan_builder_t* builder) {
  return builder->trans_result_node_count != 0;
}

static bool loom_amdgpu_wait_plan_needs_sgpr_read_state(
    const loom_amdgpu_wait_plan_builder_t* builder) {
  return loom_amdgpu_processor_properties_have_scheduling(
      builder->processor_properties,
      LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_DEPCTR);
}

static bool loom_amdgpu_wait_plan_needs_vmem_result_state(
    const loom_amdgpu_wait_plan_builder_t* builder) {
  if (builder->schedule->block_count <= 1 ||
      builder->schedule->cfg_graph.blocks == NULL) {
    return false;
  }
  return builder->vmem_result_node_count != 0;
}

static iree_status_t loom_amdgpu_wait_plan_allocate_physical_state(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_allocation_table_t* allocation = builder->allocation;
  if (allocation == NULL) {
    return iree_ok_status();
  }
  const bool needs_trans_result_state =
      loom_amdgpu_wait_plan_needs_trans_result_state(builder);
  const bool needs_sgpr_read_state =
      loom_amdgpu_wait_plan_needs_sgpr_read_state(builder);
  const bool needs_vmem_result_state =
      loom_amdgpu_wait_plan_needs_vmem_result_state(builder);
  if (!needs_trans_result_state && !needs_sgpr_read_state &&
      !needs_vmem_result_state) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->location_kind !=
        LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
      continue;
    }
    const uint64_t end =
        (uint64_t)assignment->location_base + assignment->location_count;
    IREE_ASSERT_LE(end, IREE_HOST_SIZE_MAX);
    if (assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
      if ((needs_trans_result_state || needs_vmem_result_state) &&
          (iree_host_size_t)end > builder->physical_registers.vgpr_count) {
        builder->physical_registers.vgpr_count = (iree_host_size_t)end;
      }
      continue;
    }
    if (assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
      if (needs_sgpr_read_state &&
          (iree_host_size_t)end > builder->physical_registers.sgpr_count) {
        builder->physical_registers.sgpr_count = (iree_host_size_t)end;
      }
      continue;
    }
    if (needs_vmem_result_state &&
        iree_any_bit_set(loom_amdgpu_reg_class_traits(
                             builder->schedule->target.descriptor_set,
                             assignment->descriptor_reg_class_id),
                         LOOM_AMDGPU_REG_CLASS_TRAIT_AGPR) &&
        (iree_host_size_t)end > builder->physical_registers.agpr_count) {
      builder->physical_registers.agpr_count = (iree_host_size_t)end;
    }
  }
  if (builder->physical_registers.vgpr_count != 0 && needs_trans_result_state) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->transient_arena, builder->physical_registers.vgpr_count,
        sizeof(*builder->trans_result_vgprs),
        (void**)&builder->trans_result_vgprs));
    memset(builder->trans_result_vgprs, 0,
           builder->physical_registers.vgpr_count *
               sizeof(*builder->trans_result_vgprs));
    builder->trans_result_vgpr_count = builder->physical_registers.vgpr_count;
  }
  if (builder->physical_registers.sgpr_count != 0 && needs_sgpr_read_state) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->transient_arena, builder->physical_registers.sgpr_count,
        sizeof(*builder->sgpr_read_registers),
        (void**)&builder->sgpr_read_registers));
    memset(builder->sgpr_read_registers, 0,
           builder->physical_registers.sgpr_count *
               sizeof(*builder->sgpr_read_registers));
    builder->sgpr_read_register_count = builder->physical_registers.sgpr_count;
  }
  return iree_ok_status();
}

static bool loom_amdgpu_wait_plan_action_is_residual_hazard(
    const loom_amdgpu_wait_plan_action_t* action) {
  return action->kind == LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED &&
         !iree_any_bit_set(action->flags,
                           LOOM_AMDGPU_WAIT_PLAN_ACTION_FLAG_STORAGE_RELEASE);
}

static iree_status_t loom_amdgpu_wait_plan_append_action(
    loom_amdgpu_wait_plan_builder_t* builder,
    loom_amdgpu_wait_plan_action_t action) {
  if (builder->action_stream.tail_count ==
      LOOM_AMDGPU_WAIT_PLAN_ACTIONS_PER_SEGMENT) {
    builder->action_stream.tail = NULL;
    builder->action_stream.tail_count = 0;
  }
  if (builder->action_stream.tail == NULL) {
    void* segment = NULL;
    IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
        &builder->action_stream.segments, builder->transient_arena, &segment));
    builder->action_stream.tail =
        (loom_amdgpu_wait_plan_action_segment_t*)segment;
  }
  builder->action_stream.tail->actions[builder->action_stream.tail_count++] =
      action;
  ++builder->action_count;
  if (loom_amdgpu_wait_plan_action_is_residual_hazard(&action)) {
    ++builder->hazard_event_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_finalize_actions(
    loom_amdgpu_wait_plan_builder_t* builder) {
  if (builder->action_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, builder->action_count, sizeof(*builder->actions),
      (void**)&builder->actions));

  iree_host_size_t output_count = 0;
  for (uint32_t segment_index = 0;
       segment_index < builder->action_stream.segments.segment_count;
       ++segment_index) {
    const loom_amdgpu_wait_plan_action_segment_t* segment =
        (const loom_amdgpu_wait_plan_action_segment_t*)
            loom_segmented_storage_const_segment(
                &builder->action_stream.segments, segment_index);
    const iree_host_size_t segment_action_count =
        iree_min(builder->action_count - output_count,
                 (iree_host_size_t)LOOM_AMDGPU_WAIT_PLAN_ACTIONS_PER_SEGMENT);
    memcpy(&builder->actions[output_count], segment->actions,
           segment_action_count * sizeof(*builder->actions));
    output_count += segment_action_count;
  }
  IREE_ASSERT_EQ(output_count, builder->action_count);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_ensure_dependency_link_capacity(
    loom_amdgpu_wait_plan_builder_t* builder,
    iree_host_size_t additional_count) {
  iree_host_size_t required_capacity = 0;
  const bool capacity_is_representable = iree_host_size_checked_add(
      builder->dependency_link_count, additional_count, &required_capacity);
  IREE_ASSERT(capacity_is_representable);
  if (required_capacity <= builder->dependency_link_capacity) {
    return iree_ok_status();
  }
  return iree_arena_grow_array(
      builder->transient_arena, builder->dependency_link_count,
      iree_max(required_capacity, (iree_host_size_t)16),
      sizeof(*builder->dependency_links), &builder->dependency_link_capacity,
      (void**)&builder->dependency_links);
}

static iree_status_t loom_amdgpu_wait_plan_append_dependency_link(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t producer_node,
    uint32_t consumer_node, uint32_t counter_mask,
    loom_amdgpu_wait_plan_reason_t reason) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_ensure_dependency_link_capacity(
      builder, /*additional_count=*/1));
  IREE_ASSERT_LT(builder->dependency_link_count,
                 builder->dependency_link_capacity);
  loom_amdgpu_wait_loop_dependency_t* link =
      &builder->dependency_links[builder->dependency_link_count];
  *link = (loom_amdgpu_wait_loop_dependency_t){
      .producer_node = producer_node,
      .consumer_node = consumer_node,
      .next_dependency =
          builder->first_dependency_link_by_consumer[consumer_node],
      .counter_mask = counter_mask,
      .reason_id = (uint16_t)reason,
      .flags = reason == LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE
                   ? LOOM_AMDGPU_WAIT_LOOP_DEPENDENCY_FLAG_SSA_USE
                   : 0,
  };
  IREE_ASSERT_LT(builder->dependency_link_count, UINT32_MAX);
  builder->first_dependency_link_by_consumer[consumer_node] =
      (uint32_t)builder->dependency_link_count;
  ++builder->dependency_link_count;
  return iree_ok_status();
}

static bool loom_amdgpu_wait_plan_node_has_wait_consuming_operands(
    const loom_low_schedule_node_t* node) {
  return node->op == NULL || !loom_low_br_isa(node->op);
}

static const loom_low_allocation_edge_copy_group_t*
loom_amdgpu_wait_plan_edge_copy_group(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (builder->allocation == NULL ||
      node_index >= builder->schedule->node_count) {
    return NULL;
  }
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  if (node->op == NULL || !loom_low_br_isa(node->op)) {
    return NULL;
  }
  return loom_low_allocation_find_edge_copy_group_by_source_ordinal(
      builder->allocation, node->source_ordinal);
}

static bool loom_amdgpu_wait_plan_edge_copy_materializes(
    const loom_amdgpu_wait_plan_builder_t* builder,
    const loom_low_allocation_edge_copy_t* edge_copy) {
  const loom_low_allocation_table_t* allocation = builder->allocation;
  const loom_low_allocation_assignment_t* source_assignment =
      &allocation->assignments[edge_copy->source_assignment_index];
  const loom_low_allocation_assignment_t* destination_assignment =
      &allocation->assignments[edge_copy->destination_assignment_index];
  return !loom_low_allocation_storage_assignment_subranges_equal(
      builder->schedule->target.descriptor_set, source_assignment,
      edge_copy->source_unit_offset, destination_assignment,
      edge_copy->destination_unit_offset, edge_copy->unit_count);
}

static void loom_amdgpu_wait_plan_count_block_arg_sources(
    const loom_low_schedule_table_t* schedule,
    iree_host_size_t* out_block_arg_count, iree_host_size_t* out_source_count) {
  iree_host_size_t block_arg_count = 0;
  for (iree_host_size_t i = 0; i < schedule->block_count; ++i) {
    const loom_block_t* block = schedule->blocks[i].block;
    if (block == NULL) {
      continue;
    }
    IREE_ASSERT_LE(block->arg_count, IREE_HOST_SIZE_MAX - block_arg_count);
    block_arg_count += block->arg_count;
  }

  iree_host_size_t source_count = 0;
  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    const loom_low_schedule_node_t* node = &schedule->nodes[i];
    if (node->op == NULL || !loom_low_br_isa(node->op)) {
      continue;
    }
    const loom_block_t* dest = loom_low_br_dest(node->op);
    IREE_ASSERT_NE(dest, NULL);
    IREE_ASSERT_EQ(node->operand_count, dest->arg_count);
    IREE_ASSERT_LE(node->operand_count, IREE_HOST_SIZE_MAX - source_count);
    source_count += node->operand_count;
  }

  *out_block_arg_count = block_arg_count;
  *out_source_count = source_count;
}

static void loom_amdgpu_wait_plan_compute_block_arg_offsets(
    const loom_low_schedule_table_t* schedule, uint32_t* block_arg_offsets) {
  iree_host_size_t next_offset = 0;
  for (iree_host_size_t i = 0; i < schedule->block_count; ++i) {
    IREE_ASSERT_LE(next_offset, UINT32_MAX);
    block_arg_offsets[i] = (uint32_t)next_offset;
    const loom_block_t* block = schedule->blocks[i].block;
    if (block != NULL) {
      IREE_ASSERT_LE(block->arg_count, IREE_HOST_SIZE_MAX - next_offset);
      next_offset += block->arg_count;
    }
  }
}

static void loom_amdgpu_wait_plan_initialize_block_arg_ordinals(
    const loom_low_schedule_table_t* schedule,
    const uint32_t* block_arg_offsets, iree_host_size_t block_arg_count,
    loom_value_ordinal_t* block_arg_ordinals) {
  for (iree_host_size_t i = 0; i < block_arg_count; ++i) {
    block_arg_ordinals[i] = LOOM_VALUE_ORDINAL_INVALID;
  }
  for (loom_value_ordinal_t i = 0; i < schedule->value_count; ++i) {
    const loom_value_id_t value_id = schedule->value_ids[i];
    const loom_value_t* value = loom_module_value(schedule->module, value_id);
    if (!loom_value_is_block_arg(value)) {
      continue;
    }
    const loom_block_t* block = loom_value_def_block(value);
    IREE_ASSERT_NE(block, NULL);
    IREE_ASSERT_LT(block->region_index, schedule->block_count);
    IREE_ASSERT_EQ(schedule->blocks[block->region_index].block, block);
    const uint16_t arg_index = loom_value_def_index(value);
    IREE_ASSERT_LT(arg_index, block->arg_count);
    const uint32_t offset = block_arg_offsets[block->region_index] + arg_index;
    IREE_ASSERT_LT(offset, block_arg_count);
    block_arg_ordinals[offset] = i;
  }
}

static iree_status_t loom_amdgpu_wait_plan_build_block_arg_sources(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  iree_host_size_t block_arg_count = 0;
  iree_host_size_t source_count = 0;
  loom_amdgpu_wait_plan_count_block_arg_sources(schedule, &block_arg_count,
                                                &source_count);
  if (block_arg_count == 0 || source_count == 0) {
    return iree_ok_status();
  }

  uint32_t* block_arg_offsets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->transient_arena, schedule->block_count,
      sizeof(*block_arg_offsets), (void**)&block_arg_offsets));
  loom_amdgpu_wait_plan_compute_block_arg_offsets(schedule, block_arg_offsets);

  loom_value_ordinal_t* block_arg_ordinals = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->transient_arena, block_arg_count, sizeof(*block_arg_ordinals),
      (void**)&block_arg_ordinals));
  loom_amdgpu_wait_plan_initialize_block_arg_ordinals(
      schedule, block_arg_offsets, block_arg_count, block_arg_ordinals);

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->transient_arena, schedule->value_count,
      sizeof(*builder->first_block_arg_source_by_value),
      (void**)&builder->first_block_arg_source_by_value));
  for (iree_host_size_t i = 0; i < schedule->value_count; ++i) {
    builder->first_block_arg_source_by_value[i] = LOOM_LOW_SCHEDULE_NODE_NONE;
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(builder->transient_arena, source_count,
                                sizeof(*builder->block_arg_sources),
                                (void**)&builder->block_arg_sources));

  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    const loom_low_schedule_node_t* node = &schedule->nodes[i];
    if (node->op == NULL || !loom_low_br_isa(node->op)) {
      continue;
    }
    const loom_block_t* dest = loom_low_br_dest(node->op);
    const uint32_t dest_offset = block_arg_offsets[dest->region_index];
    const loom_value_ordinal_t* operand_ordinals =
        loom_low_schedule_node_const_operand_ordinals(node);
    for (uint16_t arg_index = 0; arg_index < dest->arg_count; ++arg_index) {
      const loom_value_ordinal_t dest_ordinal =
          block_arg_ordinals[dest_offset + arg_index];
      IREE_ASSERT_NE(dest_ordinal, LOOM_VALUE_ORDINAL_INVALID);
      IREE_ASSERT_LT(dest_ordinal, schedule->value_count);
      IREE_ASSERT_LT(builder->block_arg_source_count, source_count);
      builder->block_arg_sources[builder->block_arg_source_count] =
          (loom_amdgpu_wait_block_arg_source_t){
              .source_ordinal = operand_ordinals[arg_index],
              .next_source =
                  builder->first_block_arg_source_by_value[dest_ordinal],
          };
      builder->first_block_arg_source_by_value[dest_ordinal] =
          (uint32_t)builder->block_arg_source_count++;
    }
  }
  IREE_ASSERT_EQ(builder->block_arg_source_count, source_count);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_ensure_dependency_visit_state(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  if (schedule->value_count == 0 ||
      (builder->first_block_arg_source_by_value == NULL &&
       builder->forwarding_node_count == 0)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(builder->transient_arena, schedule->value_count,
                                sizeof(*builder->dependency_visit_epochs),
                                (void**)&builder->dependency_visit_epochs));
  memset(builder->dependency_visit_epochs, 0,
         schedule->value_count * sizeof(*builder->dependency_visit_epochs));
  return iree_arena_allocate_array(builder->transient_arena,
                                   schedule->value_count,
                                   sizeof(*builder->dependency_visit_worklist),
                                   (void**)&builder->dependency_visit_worklist);
}

static uint32_t loom_amdgpu_wait_plan_begin_dependency_visit(
    loom_amdgpu_wait_plan_builder_t* builder) {
  if (builder->dependency_visit_epochs == NULL) {
    return 0;
  }
  if (builder->dependency_visit_epoch == UINT32_MAX) {
    memset(builder->dependency_visit_epochs, 0,
           builder->schedule->value_count *
               sizeof(*builder->dependency_visit_epochs));
    builder->dependency_visit_epoch = 0;
  }
  return ++builder->dependency_visit_epoch;
}

static void loom_amdgpu_wait_plan_push_dependency_visit_ordinal(
    loom_amdgpu_wait_plan_builder_t* builder, iree_host_size_t value_count,
    loom_value_ordinal_t value_ordinal, uint32_t visit_epoch,
    iree_host_size_t* worklist_count) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(worklist_count);
  IREE_ASSERT_NE(visit_epoch, 0u);
  IREE_ASSERT_LT(value_ordinal, value_count);
  if (builder->dependency_visit_epochs[value_ordinal] == visit_epoch) {
    return;
  }
  builder->dependency_visit_epochs[value_ordinal] = visit_epoch;
  IREE_ASSERT_LT(*worklist_count, value_count);
  builder->dependency_visit_worklist[(*worklist_count)++] = value_ordinal;
}

static void loom_amdgpu_wait_plan_reverse_dependency_visit_range(
    loom_value_ordinal_t* worklist, iree_host_size_t begin,
    iree_host_size_t end) {
  while (end > begin + 1) {
    loom_value_ordinal_t tmp = worklist[begin];
    worklist[begin++] = worklist[--end];
    worklist[end] = tmp;
  }
}

static iree_status_t loom_amdgpu_wait_plan_append_direct_dependency_link(
    loom_amdgpu_wait_plan_builder_t* builder, const uint32_t* producer_nodes,
    iree_host_size_t value_count, loom_value_ordinal_t operand_ordinal,
    uint32_t consumer_node) {
  IREE_ASSERT_LT(operand_ordinal, value_count);
  const uint32_t producer_node = producer_nodes[operand_ordinal];
  if (producer_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
      producer_node == consumer_node) {
    return iree_ok_status();
  }
  const uint32_t counter_mask =
      builder->frontier_nodes[producer_node].read_counter_mask;
  if (counter_mask == 0) {
    return iree_ok_status();
  }
  return loom_amdgpu_wait_plan_append_dependency_link(
      builder, producer_node, consumer_node, counter_mask,
      LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE);
}

static iree_status_t loom_amdgpu_wait_plan_visit_dependency_links(
    loom_amdgpu_wait_plan_builder_t* builder, const uint32_t* producer_nodes,
    iree_host_size_t value_count, loom_value_ordinal_t operand_ordinal,
    uint32_t consumer_node, uint32_t visit_epoch) {
  if (builder->dependency_visit_worklist == NULL) {
    return loom_amdgpu_wait_plan_append_direct_dependency_link(
        builder, producer_nodes, value_count, operand_ordinal, consumer_node);
  }
  iree_host_size_t worklist_count = 0;
  loom_amdgpu_wait_plan_push_dependency_visit_ordinal(
      builder, value_count, operand_ordinal, visit_epoch, &worklist_count);
  while (worklist_count != 0) {
    const loom_value_ordinal_t current_ordinal =
        builder->dependency_visit_worklist[--worklist_count];
    if (builder->first_block_arg_source_by_value != NULL) {
      uint32_t source_index =
          builder->first_block_arg_source_by_value[current_ordinal];
      if (source_index != LOOM_LOW_SCHEDULE_NODE_NONE) {
        const iree_host_size_t range_begin = worklist_count;
        while (source_index != LOOM_LOW_SCHEDULE_NODE_NONE) {
          const loom_amdgpu_wait_block_arg_source_t* source =
              &builder->block_arg_sources[source_index];
          loom_amdgpu_wait_plan_push_dependency_visit_ordinal(
              builder, value_count, source->source_ordinal, visit_epoch,
              &worklist_count);
          source_index = source->next_source;
        }
        loom_amdgpu_wait_plan_reverse_dependency_visit_range(
            builder->dependency_visit_worklist, range_begin, worklist_count);
        continue;
      }
    }

    const uint32_t producer_node = producer_nodes[current_ordinal];
    if (producer_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
        producer_node == consumer_node) {
      continue;
    }
    if (!loom_amdgpu_wait_plan_node_forwards_dependencies(builder,
                                                          producer_node)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_append_direct_dependency_link(
          builder, producer_nodes, value_count, current_ordinal,
          consumer_node));
      continue;
    }

    const loom_low_schedule_node_t* producer =
        &builder->schedule->nodes[producer_node];
    const loom_value_ordinal_t* producer_operands =
        loom_low_schedule_node_const_operand_ordinals(producer);
    const iree_host_size_t range_begin = worklist_count;
    for (uint16_t i = 0; i < producer->operand_count; ++i) {
      loom_amdgpu_wait_plan_push_dependency_visit_ordinal(
          builder, value_count, producer_operands[i], visit_epoch,
          &worklist_count);
    }
    loom_amdgpu_wait_plan_reverse_dependency_visit_range(
        builder->dependency_visit_worklist, range_begin, worklist_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_build_edge_copy_dependency_links(
    loom_amdgpu_wait_plan_builder_t* builder, const uint32_t* producer_nodes,
    iree_host_size_t value_count, uint32_t consumer_node) {
  const loom_low_allocation_edge_copy_group_t* group =
      loom_amdgpu_wait_plan_edge_copy_group(builder, consumer_node);
  if (group == NULL) {
    const loom_low_schedule_node_t* node =
        &builder->schedule->nodes[consumer_node];
    if (node->op != NULL && loom_low_br_isa(node->op) &&
        loom_low_br_args(node->op).count != 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU wait planning requires allocation edge copies for low.br "
          "payloads");
    }
    return iree_ok_status();
  }
  const loom_low_allocation_table_t* allocation = builder->allocation;
  IREE_ASSERT_LE(group->copy_start, allocation->edge_copy_count);
  IREE_ASSERT_LE(group->copy_count,
                 allocation->edge_copy_count - group->copy_start);
  for (iree_host_size_t i = 0; i < group->copy_count; ++i) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &allocation->edge_copies[group->copy_start + i];
    if (!loom_amdgpu_wait_plan_edge_copy_materializes(builder, edge_copy)) {
      continue;
    }
    const loom_value_ordinal_t source_ordinal =
        loom_module_value_ordinal_scratch_lookup(builder->schedule->module,
                                                 edge_copy->source_value_id);
    if (source_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
        source_ordinal >= value_count ||
        builder->schedule->value_ids[source_ordinal] !=
            edge_copy->source_value_id) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU edge-copy source is outside the scheduled value domain");
    }
    const uint32_t visit_epoch =
        loom_amdgpu_wait_plan_begin_dependency_visit(builder);
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_visit_dependency_links(
        builder, producer_nodes, value_count, source_ordinal, consumer_node,
        visit_epoch));
  }
  return iree_ok_status();
}

static uint32_t loom_amdgpu_wait_plan_memory_effect_counter_mask(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t producer_node,
    uint32_t consumer_node) {
  const loom_amdgpu_wait_frontier_node_t* producer_memory =
      &builder->frontier_nodes[producer_node];
  const loom_amdgpu_wait_node_state_t* consumer_state =
      &builder->node_states[consumer_node];
  const loom_low_schedule_node_t* consumer =
      &builder->schedule->nodes[consumer_node];
  if (iree_any_bit_set(consumer->flags,
                       LOOM_LOW_SCHEDULE_NODE_FLAG_PROGRAM_EXIT_MEMORY)) {
    return producer_memory->write_counter_mask &
           LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE;
  }
  uint32_t counter_mask = 0;
  if (iree_any_bit_set(consumer_state->flags,
                       LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_READ)) {
    counter_mask |= producer_memory->write_counter_mask;
  }
  if (iree_any_bit_set(consumer_state->flags,
                       LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_WRITE)) {
    counter_mask |= producer_memory->read_counter_mask;
  }
  return counter_mask;
}

static loom_amdgpu_wait_plan_reason_t
loom_amdgpu_wait_plan_memory_effect_reason(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t consumer_node) {
  const loom_low_schedule_node_t* consumer =
      &builder->schedule->nodes[consumer_node];
  return iree_any_bit_set(consumer->flags,
                          LOOM_LOW_SCHEDULE_NODE_FLAG_PROGRAM_EXIT_MEMORY)
             ? LOOM_AMDGPU_WAIT_PLAN_REASON_PROGRAM_EXIT
             : LOOM_AMDGPU_WAIT_PLAN_REASON_MEMORY_EFFECT;
}

static iree_status_t loom_amdgpu_wait_plan_visit_effect_dependency_link(
    loom_amdgpu_wait_plan_builder_t* builder,
    const loom_low_schedule_dependency_t* dependency) {
  if (dependency->kind != LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT) {
    return iree_ok_status();
  }
  IREE_ASSERT_LT(dependency->producer_node, builder->schedule->node_count);
  IREE_ASSERT_LT(dependency->consumer_node, builder->schedule->node_count);
  if (loom_amdgpu_wait_plan_node_forwards_dependencies(
          builder, dependency->consumer_node)) {
    return iree_ok_status();
  }
  const uint32_t counter_mask =
      loom_amdgpu_wait_plan_memory_effect_counter_mask(
          builder, dependency->producer_node, dependency->consumer_node);
  if (counter_mask == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_append_dependency_link(
      builder, dependency->producer_node, dependency->consumer_node,
      counter_mask,
      loom_amdgpu_wait_plan_memory_effect_reason(builder,
                                                 dependency->consumer_node)));
  return iree_ok_status();
}

static bool loom_amdgpu_wait_plan_assignment_is_physical_vgpr(
    const loom_low_allocation_assignment_t* assignment) {
  return assignment != NULL &&
         assignment->location_kind ==
             LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER &&
         assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR;
}

static bool loom_amdgpu_wait_plan_assignment_is_physical_sgpr(
    const loom_low_allocation_assignment_t* assignment) {
  return assignment != NULL &&
         assignment->location_kind ==
             LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER &&
         assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR;
}

static bool loom_amdgpu_wait_plan_has_trans_result_state(
    const loom_amdgpu_wait_plan_builder_t* builder) {
  return builder->trans_result_vgprs != NULL &&
         builder->trans_result_vgpr_count != 0;
}

static bool loom_amdgpu_wait_plan_has_sgpr_read_state(
    const loom_amdgpu_wait_plan_builder_t* builder) {
  return builder->sgpr_read_registers != NULL &&
         builder->sgpr_read_register_count != 0;
}

static void loom_amdgpu_wait_plan_clear_trans_result_assignment(
    loom_amdgpu_wait_plan_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment) {
  if (!loom_amdgpu_wait_plan_has_trans_result_state(builder) ||
      !loom_amdgpu_wait_plan_assignment_is_physical_vgpr(assignment)) {
    return;
  }
  const uint64_t end =
      (uint64_t)assignment->location_base + assignment->location_count;
  if (end > builder->trans_result_vgpr_count) {
    return;
  }
  for (uint32_t i = 0; i < assignment->location_count; ++i) {
    loom_amdgpu_wait_trans_result_vgpr_t* vgpr =
        &builder->trans_result_vgprs[assignment->location_base + i];
    if (iree_any_bit_set(vgpr->flags,
                         LOOM_AMDGPU_WAIT_TRANS_RESULT_VGPR_FLAG_VALID) &&
        vgpr->block_epoch == builder->block_epoch &&
        vgpr->counter_epoch ==
            builder->counter_epochs[loom_amdgpu_wait_counter_slot_from_id(
                LOOM_AMDGPU_WAIT_COUNTER_ALU)] &&
        builder->active_trans_result_vgpr_count != 0) {
      --builder->active_trans_result_vgpr_count;
    }
    *vgpr = (loom_amdgpu_wait_trans_result_vgpr_t){0};
  }
}

static void loom_amdgpu_wait_plan_record_trans_result_assignment(
    loom_amdgpu_wait_plan_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment,
    uint32_t producer_node) {
  if (!loom_amdgpu_wait_plan_has_trans_result_state(builder) ||
      !loom_amdgpu_wait_plan_assignment_is_physical_vgpr(assignment)) {
    return;
  }
  const uint64_t end =
      (uint64_t)assignment->location_base + assignment->location_count;
  if (end > builder->trans_result_vgpr_count) {
    return;
  }
  const uint32_t alu_slot =
      loom_amdgpu_wait_counter_slot_from_id(LOOM_AMDGPU_WAIT_COUNTER_ALU);
  for (uint32_t i = 0; i < assignment->location_count; ++i) {
    loom_amdgpu_wait_trans_result_vgpr_t* vgpr =
        &builder->trans_result_vgprs[assignment->location_base + i];
    if (!iree_any_bit_set(vgpr->flags,
                          LOOM_AMDGPU_WAIT_TRANS_RESULT_VGPR_FLAG_VALID) ||
        vgpr->block_epoch != builder->block_epoch ||
        vgpr->counter_epoch != builder->counter_epochs[alu_slot]) {
      ++builder->active_trans_result_vgpr_count;
    }
    *vgpr = (loom_amdgpu_wait_trans_result_vgpr_t){
        .flags = LOOM_AMDGPU_WAIT_TRANS_RESULT_VGPR_FLAG_VALID,
        .producer_node = producer_node,
        .counter_epoch = builder->counter_epochs[alu_slot],
        .block_epoch = builder->block_epoch,
    };
  }
}

static bool loom_amdgpu_wait_plan_trans_result_vgpr_is_active(
    const loom_amdgpu_wait_plan_builder_t* builder,
    const loom_amdgpu_wait_trans_result_vgpr_t* vgpr) {
  if (!iree_any_bit_set(vgpr->flags,
                        LOOM_AMDGPU_WAIT_TRANS_RESULT_VGPR_FLAG_VALID)) {
    return false;
  }
  if (vgpr->block_epoch != builder->block_epoch) {
    return false;
  }
  if (vgpr->counter_epoch !=
      builder->counter_epochs[loom_amdgpu_wait_counter_slot_from_id(
          LOOM_AMDGPU_WAIT_COUNTER_ALU)]) {
    return false;
  }
  return vgpr->valu_interval <=
             LOOM_AMDGPU_VALU_TRANS_USE_DEPCTR_MAX_VALU_INTERVAL &&
         vgpr->trans_interval <=
             LOOM_AMDGPU_VALU_TRANS_USE_DEPCTR_MAX_TRANS_INTERVAL;
}

static bool loom_amdgpu_wait_plan_assignment_reuses_trans_result(
    const loom_amdgpu_wait_plan_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment,
    uint32_t* out_producer_node) {
  *out_producer_node = LOOM_LOW_SCHEDULE_NODE_NONE;
  if (!loom_amdgpu_wait_plan_has_trans_result_state(builder) ||
      builder->active_trans_result_vgpr_count == 0 ||
      !loom_amdgpu_wait_plan_assignment_is_physical_vgpr(assignment)) {
    return false;
  }
  const uint64_t end =
      (uint64_t)assignment->location_base + assignment->location_count;
  if (end > builder->trans_result_vgpr_count) {
    return false;
  }
  for (uint32_t i = 0; i < assignment->location_count; ++i) {
    const loom_amdgpu_wait_trans_result_vgpr_t* vgpr =
        &builder->trans_result_vgprs[assignment->location_base + i];
    if (!loom_amdgpu_wait_plan_trans_result_vgpr_is_active(builder, vgpr)) {
      continue;
    }
    *out_producer_node = vgpr->producer_node;
    return true;
  }
  return false;
}

static uint32_t loom_amdgpu_wait_plan_sgpr_pair_base(uint32_t register_index) {
  return register_index & ~1u;
}

static bool loom_amdgpu_wait_plan_sgpr_pair_is_tracked(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t register_index) {
  const uint32_t pair_base =
      loom_amdgpu_wait_plan_sgpr_pair_base(register_index);
  for (uint32_t i = 0; i < 2; ++i) {
    const uint32_t pair_index = pair_base + i;
    if (pair_index >= builder->sgpr_read_register_count) {
      continue;
    }
    const loom_amdgpu_wait_sgpr_read_register_t* register_state =
        &builder->sgpr_read_registers[pair_index];
    if (iree_any_bit_set(register_state->flags,
                         LOOM_AMDGPU_WAIT_SGPR_READ_FLAG_TRACKED)) {
      return true;
    }
  }
  return false;
}

static void loom_amdgpu_wait_plan_track_sgpr_pair(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t register_index) {
  const uint32_t pair_base =
      loom_amdgpu_wait_plan_sgpr_pair_base(register_index);
  for (uint32_t i = 0; i < 2; ++i) {
    const uint32_t pair_index = pair_base + i;
    if (pair_index >= builder->sgpr_read_register_count) {
      continue;
    }
    builder->sgpr_read_registers[pair_index].flags |=
        LOOM_AMDGPU_WAIT_SGPR_READ_FLAG_TRACKED;
  }
}

static void loom_amdgpu_wait_plan_track_sgpr_read_assignment(
    loom_amdgpu_wait_plan_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment) {
  if (!loom_amdgpu_wait_plan_has_sgpr_read_state(builder) ||
      !loom_amdgpu_wait_plan_assignment_is_physical_sgpr(assignment)) {
    return;
  }
  const uint64_t end =
      (uint64_t)assignment->location_base + assignment->location_count;
  if (end > builder->sgpr_read_register_count) {
    return;
  }
  for (uint32_t i = 0; i < assignment->location_count; ++i) {
    loom_amdgpu_wait_plan_track_sgpr_pair(builder,
                                          assignment->location_base + i);
  }
}

static void loom_amdgpu_wait_plan_record_sgpr_read_write_assignment(
    loom_amdgpu_wait_plan_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment, bool is_vector_alu,
    bool is_scalar_alu, uint32_t producer_node) {
  if (!loom_amdgpu_wait_plan_has_sgpr_read_state(builder) ||
      !loom_amdgpu_wait_plan_assignment_is_physical_sgpr(assignment) ||
      (!is_vector_alu && !is_scalar_alu)) {
    return;
  }
  const uint64_t end =
      (uint64_t)assignment->location_base + assignment->location_count;
  if (end > builder->sgpr_read_register_count) {
    return;
  }
  const loom_amdgpu_wait_sgpr_read_flags_t hazard_flag =
      is_vector_alu ? LOOM_AMDGPU_WAIT_SGPR_READ_FLAG_VALU_HAZARD
                    : LOOM_AMDGPU_WAIT_SGPR_READ_FLAG_SALU_HAZARD;
  for (uint32_t i = 0; i < assignment->location_count; ++i) {
    const uint32_t register_index = assignment->location_base + i;
    if (!loom_amdgpu_wait_plan_sgpr_pair_is_tracked(builder, register_index)) {
      continue;
    }
    loom_amdgpu_wait_sgpr_read_register_t* register_state =
        &builder->sgpr_read_registers[register_index];
    register_state->flags &= ~(LOOM_AMDGPU_WAIT_SGPR_READ_FLAG_SALU_HAZARD |
                               LOOM_AMDGPU_WAIT_SGPR_READ_FLAG_VALU_HAZARD);
    register_state->flags |= hazard_flag;
    register_state->producer_node = producer_node;
  }
}

static bool loom_amdgpu_wait_plan_sgpr_read_assignment_has_hazard(
    const loom_amdgpu_wait_plan_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment, bool is_vector_alu,
    bool is_scalar_alu, uint32_t* out_producer_node) {
  *out_producer_node = LOOM_LOW_SCHEDULE_NODE_NONE;
  if (!loom_amdgpu_wait_plan_has_sgpr_read_state(builder) ||
      !loom_amdgpu_wait_plan_assignment_is_physical_sgpr(assignment) ||
      (!is_vector_alu && !is_scalar_alu)) {
    return false;
  }
  const uint64_t end =
      (uint64_t)assignment->location_base + assignment->location_count;
  if (end > builder->sgpr_read_register_count) {
    return false;
  }
  for (uint32_t i = 0; i < assignment->location_count; ++i) {
    const loom_amdgpu_wait_sgpr_read_register_t* register_state =
        &builder->sgpr_read_registers[assignment->location_base + i];
    const bool waits_for_salu = iree_any_bit_set(
        register_state->flags, LOOM_AMDGPU_WAIT_SGPR_READ_FLAG_SALU_HAZARD);
    const bool waits_for_valu =
        is_vector_alu &&
        iree_any_bit_set(register_state->flags,
                         LOOM_AMDGPU_WAIT_SGPR_READ_FLAG_VALU_HAZARD);
    if (waits_for_salu || waits_for_valu) {
      *out_producer_node = register_state->producer_node;
      return true;
    }
  }
  return false;
}

static void loom_amdgpu_wait_plan_clear_sgpr_read_hazards(
    loom_amdgpu_wait_plan_builder_t* builder) {
  if (!loom_amdgpu_wait_plan_has_sgpr_read_state(builder)) {
    return;
  }
  for (iree_host_size_t i = 0; i < builder->sgpr_read_register_count; ++i) {
    builder->sgpr_read_registers[i].flags &=
        ~(LOOM_AMDGPU_WAIT_SGPR_READ_FLAG_SALU_HAZARD |
          LOOM_AMDGPU_WAIT_SGPR_READ_FLAG_VALU_HAZARD);
    builder->sgpr_read_registers[i].producer_node = LOOM_LOW_SCHEDULE_NODE_NONE;
  }
}

static void loom_amdgpu_wait_plan_expire_trans_results(
    loom_amdgpu_wait_plan_builder_t* builder) {
  if (!loom_amdgpu_wait_plan_has_trans_result_state(builder) ||
      builder->active_trans_result_vgpr_count == 0) {
    return;
  }
  const uint32_t alu_slot =
      loom_amdgpu_wait_counter_slot_from_id(LOOM_AMDGPU_WAIT_COUNTER_ALU);
  ++builder->counter_epochs[alu_slot];
  builder->completed_position_counts[alu_slot] = 0;
  builder->outstanding_counts[alu_slot] = 0;
  builder->active_trans_result_vgpr_count = 0;
}

static uint8_t loom_amdgpu_wait_plan_saturated_increment(uint8_t value,
                                                         uint8_t limit) {
  return value <= limit ? (uint8_t)(value + 1u) : value;
}

static void loom_amdgpu_wait_plan_increment_trans_result_intervals(
    loom_amdgpu_wait_plan_builder_t* builder, bool is_vector_alu,
    bool is_transcendental) {
  if (!loom_amdgpu_wait_plan_has_trans_result_state(builder) ||
      builder->active_trans_result_vgpr_count == 0 ||
      (!is_vector_alu && !is_transcendental)) {
    return;
  }
  for (iree_host_size_t i = 0; i < builder->trans_result_vgpr_count; ++i) {
    loom_amdgpu_wait_trans_result_vgpr_t* vgpr =
        &builder->trans_result_vgprs[i];
    if (!loom_amdgpu_wait_plan_trans_result_vgpr_is_active(builder, vgpr)) {
      continue;
    }
    if (is_vector_alu) {
      vgpr->valu_interval = loom_amdgpu_wait_plan_saturated_increment(
          vgpr->valu_interval,
          LOOM_AMDGPU_VALU_TRANS_USE_DEPCTR_MAX_VALU_INTERVAL);
    }
    if (is_transcendental) {
      vgpr->trans_interval = loom_amdgpu_wait_plan_saturated_increment(
          vgpr->trans_interval,
          LOOM_AMDGPU_VALU_TRANS_USE_DEPCTR_MAX_TRANS_INTERVAL);
    }
    if (!loom_amdgpu_wait_plan_trans_result_vgpr_is_active(builder, vgpr)) {
      *vgpr = (loom_amdgpu_wait_trans_result_vgpr_t){0};
      --builder->active_trans_result_vgpr_count;
    }
  }
}

static void loom_amdgpu_wait_plan_classify_hazards(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  for (iree_host_size_t i = 0; i < schedule->hazard_use_count; ++i) {
    const loom_low_schedule_hazard_use_t* hazard = &schedule->hazard_uses[i];
    if (hazard->kind != LOOM_LOW_HAZARD_KIND_WAIT_COUNTER) {
      continue;
    }
    IREE_ASSERT_EQ(hazard->reference_kind,
                   LOOM_LOW_HAZARD_REFERENCE_KIND_COUNTER);
    IREE_ASSERT_LT(hazard->node_index, schedule->node_count);
    const uint32_t counter_mask =
        loom_amdgpu_wait_counter_mask(hazard->reference_id);
    builder->node_states[hazard->node_index].hazard_counter_mask |=
        counter_mask;
  }
}

static void loom_amdgpu_wait_plan_classify_effects(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  for (iree_host_size_t i = 0; i < schedule->effect_use_count; ++i) {
    const loom_low_schedule_effect_use_t* effect = &schedule->effect_uses[i];
    IREE_ASSERT_LT(effect->node_index, schedule->node_count);
    loom_amdgpu_wait_node_state_t* node_state =
        &builder->node_states[effect->node_index];
    loom_amdgpu_wait_frontier_node_t* frontier_node =
        &builder->frontier_nodes[effect->node_index];
    switch (effect->kind) {
      case LOOM_LOW_EFFECT_KIND_READ: {
        if (!loom_amdgpu_wait_effect_is_dependency_memory(effect)) {
          break;
        }
        node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_READ;
        frontier_node->read_space_flags |=
            loom_amdgpu_wait_memory_space_flag(effect->memory_space);
        const uint32_t counter_mask =
            loom_amdgpu_wait_effect_counter_mask(effect);
        if (counter_mask == 0) {
          node_state->flags |=
              LOOM_AMDGPU_WAIT_NODE_STATE_DEFAULT_DEPENDENCY_READ;
        } else {
          frontier_node->read_counter_mask |= counter_mask;
        }
        break;
      }
      case LOOM_LOW_EFFECT_KIND_WRITE: {
        if (!loom_amdgpu_wait_effect_is_dependency_memory(effect)) {
          break;
        }
        node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_WRITE;
        frontier_node->write_space_flags |=
            loom_amdgpu_wait_memory_space_flag(effect->memory_space);
        const uint32_t counter_mask =
            loom_amdgpu_wait_effect_counter_mask(effect);
        if (counter_mask == 0) {
          node_state->flags |=
              LOOM_AMDGPU_WAIT_NODE_STATE_DEFAULT_DEPENDENCY_WRITE;
          if (effect->memory_space == LOOM_LOW_MEMORY_SPACE_WORKGROUP) {
            node_state->flags |=
                LOOM_AMDGPU_WAIT_NODE_STATE_DEFAULT_WORKGROUP_WRITE;
          }
        } else {
          frontier_node->write_counter_mask |= counter_mask;
          if (effect->memory_space == LOOM_LOW_MEMORY_SPACE_WORKGROUP) {
            node_state->workgroup_write_counter_mask |= counter_mask;
          }
        }
        break;
      }
      case LOOM_LOW_EFFECT_KIND_BARRIER:
        if (loom_amdgpu_wait_effect_is_dependency_memory(effect)) {
          const uint32_t counter_mask =
              loom_amdgpu_wait_effect_counter_mask(effect);
          if (effect->memory_space == LOOM_LOW_MEMORY_SPACE_WORKGROUP) {
            node_state->workgroup_barrier_counter_mask |=
                counter_mask == 0 ? LOOM_AMDGPU_WAIT_COUNTER_MASK_MEMORY
                                  : counter_mask;
          } else {
            node_state->barrier_counter_mask |=
                counter_mask == 0 ? LOOM_AMDGPU_WAIT_COUNTER_MASK_MEMORY
                                  : counter_mask;
          }
        }
        break;
      case LOOM_LOW_EFFECT_KIND_COUNTER: {
        if (effect->counter_id == LOOM_AMDGPU_WAIT_COUNTER_NONE) {
          node_state->flags |=
              LOOM_AMDGPU_WAIT_NODE_STATE_GENERIC_COUNTER_EFFECT;
          break;
        }
        const uint32_t counter_mask =
            loom_amdgpu_wait_counter_mask(effect->counter_id);
        node_state->explicit_wait_counter_mask |= counter_mask;
        break;
      }
      default:
        break;
    }
  }
}

static bool loom_amdgpu_wait_plan_descriptor_has_xcnt_source_lease(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  if (descriptor == NULL || descriptor->storage_lease_count == 0) {
    return false;
  }
  IREE_ASSERT_LE(descriptor->storage_lease_start,
                 descriptor_set->storage_lease_count);
  IREE_ASSERT_LE(
      descriptor->storage_lease_count,
      descriptor_set->storage_lease_count - descriptor->storage_lease_start);
  for (uint16_t i = 0; i < descriptor->storage_lease_count; ++i) {
    const loom_low_descriptor_storage_lease_t* lease =
        &descriptor_set->storage_leases[descriptor->storage_lease_start + i];
    if (lease->kind == LOOM_LOW_STORAGE_LEASE_SOURCE_READ &&
        lease->release_class_id == LOOM_AMDGPU_WAIT_COUNTER_X) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_wait_plan_node_is_smem_schedule_class(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_schedule_node_t* node) {
  if (node->schedule_class == NULL) return false;
  const iree_string_view_t schedule_class_name = loom_low_descriptor_set_string(
      descriptor_set, node->schedule_class->name_string_offset);
  return iree_string_view_equal(schedule_class_name,
                                IREE_SV("amdgpu.smem.load")) ||
         iree_string_view_equal(schedule_class_name,
                                IREE_SV("amdgpu.smem.store"));
}

static bool loom_amdgpu_wait_plan_descriptor_writes_exec(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  if (descriptor == NULL) return false;
  IREE_ASSERT_LE(descriptor->operand_start, descriptor_set->operand_count);
  IREE_ASSERT_LE(descriptor->operand_count,
                 descriptor_set->operand_count - descriptor->operand_start);
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + i];
    if (!iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_STATE_WRITE)) {
      continue;
    }
    IREE_ASSERT_LE(operand->reg_class_alt_start,
                   descriptor_set->reg_class_alt_count);
    IREE_ASSERT_LE(
        operand->reg_class_alt_count,
        descriptor_set->reg_class_alt_count - operand->reg_class_alt_start);
    for (uint16_t j = 0; j < operand->reg_class_alt_count; ++j) {
      const uint16_t reg_class_id =
          descriptor_set->reg_class_alts[operand->reg_class_alt_start + j]
              .reg_class_id;
      if (reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
          reg_class_id >= descriptor_set->reg_class_count) {
        continue;
      }
      const iree_string_view_t reg_class_name = loom_low_descriptor_set_string(
          descriptor_set,
          descriptor_set->reg_classes[reg_class_id].name_string_offset);
      if (iree_string_view_equal(reg_class_name, IREE_SV("amdgpu.exec"))) {
        return true;
      }
    }
  }
  return false;
}

static bool loom_amdgpu_wait_plan_structural_node_implicitly_drains_xcnt(
    const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_node_t* node) {
  const loom_op_t* op = node->op;
  if (op == NULL) return false;
  if (loom_low_return_isa(op)) return true;
  if (loom_low_br_isa(op)) {
    const uint32_t destination_block_index =
        loom_low_packet_block_index(schedule, loom_low_br_dest(op));
    return destination_block_index != node->block_index + 1;
  }
  if (loom_low_cond_br_isa(op)) {
    const loom_block_t* true_dest = loom_low_cond_br_true_dest(op);
    const loom_block_t* false_dest = loom_low_cond_br_false_dest(op);
    if (true_dest != false_dest) return true;
    const uint32_t destination_block_index =
        loom_low_packet_block_index(schedule, true_dest);
    return destination_block_index != node->block_index + 1;
  }
  return false;
}

static iree_status_t loom_amdgpu_wait_plan_finish_node_classification(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  const bool has_valu_trans_use_depctr =
      loom_amdgpu_processor_properties_have_scheduling(
          builder->processor_properties,
          LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR);
  IREE_ASSERT_LT(LOOM_AMDGPU_WAIT_COUNTER_MASK_X,
                 builder->wait_packet_target.selection_count);
  const bool supports_xcnt =
      builder->wait_packet_target.selections[LOOM_AMDGPU_WAIT_COUNTER_MASK_X]
          .covered_counter_mask == LOOM_AMDGPU_WAIT_COUNTER_MASK_X;
  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    loom_amdgpu_wait_node_state_t* node_state = &builder->node_states[i];
    loom_amdgpu_wait_frontier_node_t* frontier_node =
        &builder->frontier_nodes[i];
    const loom_low_schedule_node_t* node = &schedule->nodes[i];
    const loom_amdgpu_wait_memory_space_flags_t generic_space =
        loom_amdgpu_wait_memory_space_flag(LOOM_LOW_MEMORY_SPACE_GENERIC);
    if (node->descriptor == NULL &&
        !iree_any_bit_set(node->flags,
                          LOOM_LOW_SCHEDULE_NODE_FLAG_PROGRAM_EXIT_MEMORY)) {
      if (iree_any_bit_set(node->traits, LOOM_TRAIT_READS_MEMORY)) {
        node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_READ;
        frontier_node->read_space_flags |= generic_space;
      }
      if (iree_any_bit_set(node->traits, LOOM_TRAIT_WRITES_MEMORY)) {
        node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_WRITE;
        frontier_node->write_space_flags |= generic_space;
      }
      if (iree_any_bit_set(node->traits, LOOM_TRAIT_NON_DETERMINISTIC |
                                             LOOM_TRAIT_UNKNOWN_EFFECTS |
                                             LOOM_TRAIT_CONVERGENT)) {
        node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_READ |
                             LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_WRITE;
        frontier_node->read_space_flags |= generic_space;
        frontier_node->write_space_flags |= generic_space;
      }
    }
    const loom_amdgpu_descriptor_traits_t descriptor_traits =
        loom_amdgpu_descriptor_traits(descriptor_set, node->descriptor);
    if (loom_amdgpu_wait_plan_descriptor_has_xcnt_source_lease(
            descriptor_set, node->descriptor)) {
      node_state->source_counter_mask |= LOOM_AMDGPU_WAIT_COUNTER_MASK_X;
      node_state->flags |= loom_amdgpu_wait_plan_node_is_smem_schedule_class(
                               descriptor_set, node)
                               ? LOOM_AMDGPU_WAIT_NODE_STATE_XCNT_SMEM_PRODUCER
                               : LOOM_AMDGPU_WAIT_NODE_STATE_XCNT_VMEM_PRODUCER;
      frontier_node->xcnt_group_flags =
          iree_any_bit_set(node_state->flags,
                           LOOM_AMDGPU_WAIT_NODE_STATE_XCNT_SMEM_PRODUCER)
              ? LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_SMEM
              : LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_VMEM;
    }
    if (supports_xcnt && loom_amdgpu_wait_plan_descriptor_writes_exec(
                             descriptor_set, node->descriptor)) {
      node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_WRITES_EXEC;
    }
    if (supports_xcnt &&
        (iree_any_bit_set(descriptor_traits,
                          LOOM_AMDGPU_DESCRIPTOR_TRAIT_XCNT_IMPLICIT_DRAIN) ||
         loom_amdgpu_wait_plan_structural_node_implicitly_drains_xcnt(schedule,
                                                                      node))) {
      node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_XCNT_IMPLICIT_DRAIN;
      node_state->implicit_wait_counter_mask |= LOOM_AMDGPU_WAIT_COUNTER_MASK_X;
    }
    frontier_node->vmem_result_order_class =
        loom_amdgpu_descriptor_vmem_result_order_class(descriptor_set,
                                                       node->descriptor);
    if (frontier_node->vmem_result_order_class !=
        LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE) {
      ++builder->vmem_result_node_count;
    }
    if (node->descriptor != NULL && node->result_count != 0) {
      node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_MATERIALIZES_RESULTS;
    }
    if (iree_any_bit_set(descriptor_traits,
                         LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU)) {
      node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_USES_VECTOR_ALU;
    }
    if (iree_any_bit_set(descriptor_traits,
                         LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_ALU)) {
      node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_USES_SCALAR_ALU;
    }
    if (iree_any_bit_set(descriptor_traits,
                         LOOM_AMDGPU_DESCRIPTOR_TRAIT_TRANSCENDENTAL)) {
      node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_TRANSCENDENTAL;
    }
    const loom_amdgpu_wait_node_state_flags_t flags = node_state->flags;
    const bool has_generic_counter_effect = iree_any_bit_set(
        flags, LOOM_AMDGPU_WAIT_NODE_STATE_GENERIC_COUNTER_EFFECT);
    if (has_generic_counter_effect) {
      node_state->explicit_wait_counter_mask |= node_state->hazard_counter_mask;
    }
    if (node_state->explicit_wait_counter_mask != 0 &&
        !has_generic_counter_effect) {
      node_state->explicit_wait_counter_mask =
          loom_amdgpu_wait_packet_explicit_counter_mask(
              descriptor_set, node->descriptor, &builder->wait_packet_target,
              schedule->module, node->op);
    }
    IREE_ASSERT(node_state->explicit_wait_counter_mask == 0 ||
                node_state->hazard_counter_mask != 0);
    frontier_node->drain_counter_mask = node_state->explicit_wait_counter_mask |
                                        node_state->implicit_wait_counter_mask;
    if (iree_any_bit_set(flags,
                         LOOM_AMDGPU_WAIT_NODE_STATE_DEFAULT_DEPENDENCY_READ)) {
      const uint32_t default_read_counter_mask =
          node_state->hazard_counter_mask & LOOM_AMDGPU_WAIT_COUNTER_MASK_READ;
      IREE_ASSERT_NE(default_read_counter_mask, 0u);
      frontier_node->read_counter_mask |= default_read_counter_mask;
    }
    IREE_ASSERT_EQ(
        frontier_node->read_counter_mask & ~node_state->hazard_counter_mask,
        0u);
    if (iree_any_bit_set(
            flags, LOOM_AMDGPU_WAIT_NODE_STATE_DEFAULT_DEPENDENCY_WRITE)) {
      const uint32_t default_write_counter_mask =
          node_state->hazard_counter_mask & LOOM_AMDGPU_WAIT_COUNTER_MASK_WRITE;
      IREE_ASSERT_NE(default_write_counter_mask, 0u);
      frontier_node->write_counter_mask |= default_write_counter_mask;
      if (iree_any_bit_set(
              flags, LOOM_AMDGPU_WAIT_NODE_STATE_DEFAULT_WORKGROUP_WRITE)) {
        node_state->workgroup_write_counter_mask |= default_write_counter_mask;
      }
    }
    IREE_ASSERT_EQ(
        frontier_node->write_counter_mask & ~node_state->hazard_counter_mask,
        0u);
    const loom_amdgpu_structural_packet_flags_t structural_flags =
        loom_amdgpu_wait_plan_classify_structural_node(builder, (uint32_t)i);
    if (iree_any_bit_set(
            structural_flags,
            LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_FORWARDS_DEPENDENCIES)) {
      node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_FORWARDS_DEPENDENCIES;
      ++builder->forwarding_node_count;
    }
    if (node->result_count != 0 &&
        iree_any_bit_set(structural_flags,
                         LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES)) {
      node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_MATERIALIZES_RESULTS;
    }
    if (has_valu_trans_use_depctr &&
        iree_any_bit_set(flags, LOOM_AMDGPU_WAIT_NODE_STATE_TRANSCENDENTAL)) {
      node_state->trans_result_counter_mask |=
          LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU;
      ++builder->trans_result_node_count;
    }
    builder->loop_nodes[i] = (loom_amdgpu_wait_loop_node_t){
        .producer_counter_mask = frontier_node->read_counter_mask |
                                 frontier_node->write_counter_mask |
                                 node_state->trans_result_counter_mask |
                                 node_state->source_counter_mask,
        .write_counter_mask = frontier_node->write_counter_mask,
        .reset_counter_mask = node_state->explicit_wait_counter_mask |
                              node_state->implicit_wait_counter_mask |
                              node_state->barrier_counter_mask,
        .hazard_counter_mask = node_state->hazard_counter_mask,
        .workgroup_write_counter_mask =
            node_state->workgroup_write_counter_mask,
        .workgroup_barrier_counter_mask =
            node_state->workgroup_barrier_counter_mask,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_build_dependency_links(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  const iree_host_size_t value_count = schedule->value_count;
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_build_block_arg_sources(builder));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_ensure_dependency_visit_state(builder));

  if (value_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->transient_arena, value_count, sizeof(*builder->producer_nodes),
        (void**)&builder->producer_nodes));
    for (iree_host_size_t i = 0; i < value_count; ++i) {
      builder->producer_nodes[i] = LOOM_LOW_SCHEDULE_NODE_NONE;
    }
  }

  for (uint32_t node_index = 0; node_index < schedule->node_count;
       ++node_index) {
    const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
    const loom_value_ordinal_t* result_ordinals =
        loom_low_schedule_node_const_result_ordinals(node);
    for (uint16_t i = 0; i < node->result_count; ++i) {
      const loom_value_ordinal_t result_ordinal = result_ordinals[i];
      IREE_ASSERT_LT(result_ordinal, value_count);
      builder->producer_nodes[result_ordinal] = node_index;
    }
  }

  for (uint32_t consumer_node = 0; consumer_node < schedule->node_count;
       ++consumer_node) {
    if (loom_amdgpu_wait_plan_node_forwards_dependencies(builder,
                                                         consumer_node)) {
      continue;
    }
    const loom_low_schedule_node_t* node = &schedule->nodes[consumer_node];
    if (node->op != NULL && loom_low_br_isa(node->op)) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_wait_plan_build_edge_copy_dependency_links(
              builder, builder->producer_nodes, value_count, consumer_node));
      continue;
    }
    if (!loom_amdgpu_wait_plan_node_has_wait_consuming_operands(node)) {
      continue;
    }
    const loom_value_ordinal_t* operand_ordinals =
        loom_low_schedule_node_const_operand_ordinals(node);
    if (node->descriptor != NULL) {
      const loom_low_descriptor_set_t* descriptor_set =
          schedule->target.descriptor_set;
      const loom_low_operand_t* descriptor_operands =
          &descriptor_set->operands[node->descriptor->operand_start];
      for (uint16_t i = node->descriptor->result_count;
           i < node->descriptor->operand_count; ++i) {
        const loom_low_operand_t* descriptor_operand = &descriptor_operands[i];
        if (!loom_low_operand_role_is_packet_operand(
                descriptor_operand->role) ||
            iree_any_bit_set(descriptor_operand->flags,
                             LOOM_LOW_OPERAND_FLAG_STORAGE_CONTINUATION)) {
          continue;
        }
        IREE_ASSERT_LT(descriptor_operand->source_value_index,
                       node->operand_count);
        const uint32_t visit_epoch =
            loom_amdgpu_wait_plan_begin_dependency_visit(builder);
        IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_visit_dependency_links(
            builder, builder->producer_nodes, value_count,
            operand_ordinals[descriptor_operand->source_value_index],
            consumer_node, visit_epoch));
      }
    } else {
      for (uint16_t i = 0; i < node->operand_count; ++i) {
        const uint32_t visit_epoch =
            loom_amdgpu_wait_plan_begin_dependency_visit(builder);
        IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_visit_dependency_links(
            builder, builder->producer_nodes, value_count, operand_ordinals[i],
            consumer_node, visit_epoch));
      }
    }
  }
  for (uint32_t i = 0; i < schedule->dependencies.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_visit_effect_dependency_link(
        builder,
        loom_low_schedule_dependency_graph_at(&schedule->dependencies, i)));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_classify_nodes(
    loom_amdgpu_wait_plan_builder_t* builder) {
  loom_amdgpu_wait_plan_classify_hazards(builder);
  loom_amdgpu_wait_plan_classify_effects(builder);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_finish_node_classification(builder));
  return loom_amdgpu_wait_plan_build_dependency_links(builder);
}

static iree_host_size_t loom_amdgpu_wait_plan_loop_entry_slot_index(
    iree_host_size_t block_index, uint32_t counter_slot) {
  IREE_ASSERT_LT(counter_slot, LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT);
  return (iree_host_size_t)block_index * LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT +
         counter_slot;
}

static uint32_t loom_amdgpu_wait_plan_node_producer_counter_mask(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  const loom_amdgpu_wait_node_state_t* node_state =
      &builder->node_states[node_index];
  const loom_amdgpu_wait_frontier_node_t* frontier_node =
      &builder->frontier_nodes[node_index];
  return frontier_node->read_counter_mask | frontier_node->write_counter_mask |
         node_state->trans_result_counter_mask |
         node_state->source_counter_mask;
}

static iree_status_t loom_amdgpu_wait_plan_allocate_loop_entry_tables(
    loom_amdgpu_wait_plan_builder_t* builder) {
  if (builder->schedule->block_count >
      IREE_HOST_SIZE_MAX / LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AMDGPU loop-entry dependency table exceeds host size");
  }
  const iree_host_size_t entry_slot_count =
      builder->schedule->block_count * LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(builder->transient_arena, entry_slot_count,
                                sizeof(*builder->loop_entry_dependency_links),
                                (void**)&builder->loop_entry_dependency_links));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->transient_arena, builder->schedule->block_count,
      sizeof(*builder->loop_entry_drain_counter_masks),
      (void**)&builder->loop_entry_drain_counter_masks));
  for (iree_host_size_t i = 0; i < entry_slot_count; ++i) {
    builder->loop_entry_dependency_links[i] = LOOM_LOW_SCHEDULE_NODE_NONE;
  }
  memset(builder->loop_entry_drain_counter_masks, 0,
         builder->schedule->block_count *
             sizeof(*builder->loop_entry_drain_counter_masks));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_relocate_loop_entry_dependencies(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_loop_analysis_initialize(
      schedule, builder->transient_arena, &builder->loop_analysis));
  if (builder->dependency_link_count == 0 || schedule->block_count <= 1 ||
      schedule->cfg_graph.blocks == NULL) {
    return iree_ok_status();
  }
  if (builder->loop_analysis.loop_count == 0) return iree_ok_status();

  // Consumer adjacency lists partition the dependency-link table. The nested
  // loop therefore takes O(N+16D) for N nodes and D dependency links: it visits
  // each node and link once, then performs at most 16 ancestor probes per link.
  // It never rescans links or loop bodies for each natural loop.
  for (uint32_t consumer_node = 0; consumer_node < schedule->node_count;
       ++consumer_node) {
    uint32_t* link_index_ptr =
        &builder->first_dependency_link_by_consumer[consumer_node];
    while (*link_index_ptr != LOOM_LOW_SCHEDULE_NODE_NONE) {
      const uint32_t link_index = *link_index_ptr;
      loom_amdgpu_wait_loop_dependency_t* link =
          &builder->dependency_links[link_index];
      const uint32_t next_dependency = link->next_dependency;
      if (!iree_any_bit_set(link->flags,
                            LOOM_AMDGPU_WAIT_LOOP_DEPENDENCY_FLAG_SSA_USE) ||
          iree_math_count_ones_u32(link->counter_mask) != 1) {
        link_index_ptr = &link->next_dependency;
        continue;
      }
      const uint16_t preheader_index = loom_amdgpu_wait_loop_analysis_preheader(
          &builder->loop_analysis, link->producer_node, consumer_node);
      if (preheader_index == UINT16_MAX) {
        link_index_ptr = &link->next_dependency;
        continue;
      }

      if (builder->loop_entry_dependency_links == NULL) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_wait_plan_allocate_loop_entry_tables(builder));
      }

      *link_index_ptr = next_dependency;
      const uint32_t counter_slot =
          (uint32_t)iree_math_count_trailing_zeros_u32(link->counter_mask);
      const iree_host_size_t entry_slot_index =
          loom_amdgpu_wait_plan_loop_entry_slot_index(preheader_index,
                                                      counter_slot);
      link->next_dependency =
          builder->loop_entry_dependency_links[entry_slot_index];
      builder->loop_entry_dependency_links[entry_slot_index] = link_index;
    }
  }

  if (builder->loop_entry_dependency_links == NULL) return iree_ok_status();

  // A relocated action is visible to the static memory frontier only when its
  // target is necessarily zero. Local ordered producers retain a partial
  // frontier whenever younger packets follow the strictest required producer;
  // cross-block and SMEM dependencies remain conservative full drains.
  for (iree_host_size_t preheader_index = 0;
       preheader_index < schedule->block_count; ++preheader_index) {
    const loom_low_schedule_block_t* block = &schedule->blocks[preheader_index];
    for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT;
         ++slot) {
      const iree_host_size_t entry_slot_index =
          loom_amdgpu_wait_plan_loop_entry_slot_index(preheader_index, slot);
      uint32_t link_index =
          builder->loop_entry_dependency_links[entry_slot_index];
      if (link_index == LOOM_LOW_SCHEDULE_NODE_NONE) continue;
      const uint16_t counter_id = loom_amdgpu_wait_counter_id_from_slot(slot);
      bool is_full_drain = counter_id == LOOM_AMDGPU_WAIT_COUNTER_SMEM;
      uint32_t strictest_producer_ordinal = 0;
      while (link_index != LOOM_LOW_SCHEDULE_NODE_NONE) {
        const loom_amdgpu_wait_loop_dependency_t* link =
            &builder->dependency_links[link_index];
        const loom_low_schedule_node_t* producer =
            &schedule->nodes[link->producer_node];
        if (producer->block_index != preheader_index) {
          is_full_drain = true;
        } else {
          strictest_producer_ordinal =
              iree_max(strictest_producer_ordinal, producer->scheduled_ordinal);
        }
        link_index = link->next_dependency;
      }
      if (!is_full_drain) {
        is_full_drain = true;
        for (uint32_t i = strictest_producer_ordinal + 1;
             i < block->scheduled_node_count; ++i) {
          const uint32_t packet_index = block->scheduled_node_start + i;
          const uint32_t node_index =
              schedule->scheduled_node_indices[packet_index];
          if ((loom_amdgpu_wait_plan_node_producer_counter_mask(builder,
                                                                node_index) &
               loom_amdgpu_wait_counter_mask_from_slot(slot)) != 0) {
            is_full_drain = false;
            break;
          }
        }
      }
      if (is_full_drain) {
        builder->loop_entry_drain_counter_masks[preheader_index] |=
            loom_amdgpu_wait_counter_mask_from_slot(slot);
      }
    }
  }
  return iree_ok_status();
}

static void loom_amdgpu_wait_plan_mark_drained_producers(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index,
    uint32_t slot, uint32_t completed_position_count) {
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  const uint32_t counter_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
  const uint32_t block_index = node->block_index;
  const loom_low_schedule_block_t* block =
      &builder->schedule->blocks[block_index];
  for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
    const uint32_t packet_index = block->scheduled_node_start + i;
    if (i == node->scheduled_ordinal) {
      break;
    }
    const uint32_t prior_node_index =
        builder->schedule->scheduled_node_indices[packet_index];
    loom_amdgpu_wait_node_state_t* prior_state =
        &builder->node_states[prior_node_index];
    loom_amdgpu_wait_frontier_node_t* prior_memory =
        &builder->frontier_nodes[prior_node_index];
    const uint32_t producer_counter_mask =
        prior_memory->read_counter_mask | prior_memory->write_counter_mask |
        prior_state->trans_result_counter_mask |
        prior_state->source_counter_mask;
    if ((producer_counter_mask & counter_mask) == 0 ||
        prior_state->produced_counter_epoch[slot] !=
            builder->counter_epochs[slot]) {
      continue;
    }
    const uint32_t produced_position =
        prior_state->produced_counter_position[slot];
    if (produced_position != 0 &&
        produced_position <= completed_position_count) {
      prior_memory->drained_after_production_counter_mask |= counter_mask;
    }
  }
}

static bool loom_amdgpu_wait_plan_node_follows_producer_in_same_block(
    const loom_low_schedule_table_t* schedule, uint32_t node_index,
    uint32_t producer_node) {
  if (node_index >= schedule->node_count ||
      producer_node >= schedule->node_count) {
    return false;
  }
  const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
  const loom_low_schedule_node_t* producer = &schedule->nodes[producer_node];
  return node->block_index == producer->block_index &&
         producer->scheduled_ordinal < node->scheduled_ordinal;
}

static bool loom_amdgpu_wait_plan_current_block_drained_producer(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t producer_node,
    uint32_t counter_mask) {
  return producer_node < builder->schedule->node_count &&
         builder->current_block_drained_epochs[producer_node] ==
             builder->block_epoch &&
         iree_any_bit_set(
             builder->current_block_drained_counter_masks[producer_node],
             counter_mask);
}

static bool loom_amdgpu_wait_plan_current_block_drained_counter(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t counter_mask) {
  return iree_any_bit_set(builder->current_block_full_drain_counter_mask,
                          counter_mask);
}

static bool loom_amdgpu_wait_plan_current_block_satisfies_producer(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t producer_node,
    uint32_t counter_mask) {
  return loom_amdgpu_wait_plan_current_block_drained_counter(builder,
                                                             counter_mask) ||
         loom_amdgpu_wait_plan_current_block_drained_producer(
             builder, producer_node, counter_mask);
}

static void loom_amdgpu_wait_plan_record_current_block_drained_producer(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t producer_node,
    uint32_t counter_mask) {
  if (producer_node >= builder->schedule->node_count) {
    return;
  }
  if (builder->current_block_drained_epochs[producer_node] !=
      builder->block_epoch) {
    builder->current_block_drained_epochs[producer_node] = builder->block_epoch;
    builder->current_block_drained_counter_masks[producer_node] = 0;
  }
  builder->current_block_drained_counter_masks[producer_node] |= counter_mask;
}

static uint16_t loom_amdgpu_wait_plan_normalize_target_count(
    const loom_amdgpu_wait_plan_builder_t* builder, uint16_t counter_id,
    uint16_t target_count) {
  const uint32_t slot = loom_amdgpu_wait_counter_slot_from_id(counter_id);
  const uint32_t outstanding_count = builder->outstanding_counts[slot];
  target_count = (uint16_t)iree_min((uint32_t)target_count, outstanding_count);
  if (counter_id == LOOM_AMDGPU_WAIT_COUNTER_X && target_count != 0 &&
      builder->xcnt_group == LOOM_AMDGPU_WAIT_XCNT_GROUP_SMEM) {
    // Scalar-memory translations may complete out of order. A nonzero XCNT
    // threshold cannot prove that any particular SMEM source was released.
    target_count = 0;
  }
  return target_count;
}

static void loom_amdgpu_wait_plan_apply_counter_progress(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index,
    uint32_t producer_node, uint16_t counter_id, uint16_t target_count) {
  const uint32_t slot = loom_amdgpu_wait_counter_slot_from_id(counter_id);
  const uint32_t counter_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
  const uint32_t outstanding_before = builder->outstanding_counts[slot];
  IREE_ASSERT_LE(target_count, outstanding_before);
  const uint32_t drained_position_count = outstanding_before - target_count;
  const uint32_t completed_position_count = iree_math_saturating_add_u32(
      builder->completed_position_counts[slot], drained_position_count);
  loom_amdgpu_wait_plan_mark_drained_producers(builder, node_index, slot,
                                               completed_position_count);
  if (target_count == 0 &&
      loom_amdgpu_wait_plan_node_follows_producer_in_same_block(
          builder->schedule, node_index, producer_node)) {
    builder->frontier_nodes[producer_node]
        .drained_after_production_counter_mask |= counter_mask;
  } else if (target_count == 0) {
    loom_amdgpu_wait_plan_record_current_block_drained_producer(
        builder, producer_node, counter_mask);
  }
  if (target_count == 0) {
    builder->current_block_full_drain_counter_mask |= counter_mask;
    loom_amdgpu_wait_frontier_drain(&builder->frontier, counter_mask);
    ++builder->counter_epochs[slot];
    builder->completed_position_counts[slot] = 0;
  } else {
    builder->completed_position_counts[slot] = completed_position_count;
  }
  builder->outstanding_counts[slot] = target_count;
  builder->outstanding_write_counts[slot] =
      iree_min(builder->outstanding_write_counts[slot], (uint32_t)target_count);
  builder->outstanding_workgroup_write_counts[slot] =
      iree_min(builder->outstanding_workgroup_write_counts[slot],
               (uint32_t)target_count);
  if (counter_id == LOOM_AMDGPU_WAIT_COUNTER_ALU && target_count == 0) {
    builder->active_trans_result_vgpr_count = 0;
    loom_amdgpu_wait_plan_clear_sgpr_read_hazards(builder);
  }
  if (counter_id == LOOM_AMDGPU_WAIT_COUNTER_X && target_count == 0) {
    builder->xcnt_group = LOOM_AMDGPU_WAIT_XCNT_GROUP_NONE;
  }
}

static iree_status_t loom_amdgpu_wait_plan_wait_counter_at(
    loom_amdgpu_wait_plan_builder_t* builder,
    loom_amdgpu_wait_plan_action_kind_t kind,
    loom_amdgpu_wait_plan_action_flags_t flags,
    loom_amdgpu_wait_plan_reason_t reason, uint32_t insertion_node,
    uint32_t producer_node, uint32_t consumer_node, uint16_t counter_id,
    uint16_t target_count) {
  const loom_low_schedule_node_t* node =
      &builder->schedule->nodes[insertion_node];
  IREE_ASSERT(loom_amdgpu_wait_counter_id_is_valid(counter_id));
  const uint32_t slot = loom_amdgpu_wait_counter_slot_from_id(counter_id);
  const uint32_t outstanding_before = builder->outstanding_counts[slot];
  target_count = loom_amdgpu_wait_plan_normalize_target_count(
      builder, counter_id, target_count);
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_append_action(
      builder, (loom_amdgpu_wait_plan_action_t){
                   .kind = kind,
                   .flags = flags,
                   .reason = reason,
                   .counter_id = counter_id,
                   .target_count = target_count,
                   .block_index = node->block_index,
                   .node_index = insertion_node,
                   .scheduled_ordinal = node->scheduled_ordinal,
                   .producer_node = producer_node,
                   .consumer_node = consumer_node,
                   .outstanding_before = outstanding_before,
               }));
  loom_amdgpu_wait_plan_apply_counter_progress(
      builder, insertion_node, producer_node, counter_id, target_count);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_wait_counter(
    loom_amdgpu_wait_plan_builder_t* builder,
    loom_amdgpu_wait_plan_action_kind_t kind,
    loom_amdgpu_wait_plan_action_flags_t flags,
    loom_amdgpu_wait_plan_reason_t reason, uint32_t node_index,
    uint32_t producer_node, uint16_t counter_id, uint16_t target_count) {
  const uint32_t consumer_node =
      loom_amdgpu_wait_plan_reason_has_consumer(reason)
          ? node_index
          : LOOM_LOW_SCHEDULE_NODE_NONE;
  return loom_amdgpu_wait_plan_wait_counter_at(
      builder, kind, flags, reason, node_index, producer_node, consumer_node,
      counter_id, target_count);
}

static iree_status_t loom_amdgpu_wait_plan_drain_counter(
    loom_amdgpu_wait_plan_builder_t* builder,
    loom_amdgpu_wait_plan_action_kind_t kind,
    loom_amdgpu_wait_plan_reason_t reason, uint32_t node_index,
    uint32_t producer_node, uint16_t counter_id) {
  return loom_amdgpu_wait_plan_wait_counter(
      builder, kind, /*flags=*/0, reason, node_index, producer_node, counter_id,
      /*target_count=*/0);
}

static iree_status_t loom_amdgpu_wait_plan_drain_mask(
    loom_amdgpu_wait_plan_builder_t* builder,
    loom_amdgpu_wait_plan_action_kind_t kind,
    loom_amdgpu_wait_plan_reason_t reason, uint32_t node_index,
    uint32_t producer_node, uint32_t counter_mask) {
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    if ((counter_mask & loom_amdgpu_wait_counter_mask_from_slot(slot)) == 0) {
      continue;
    }
    const uint16_t counter_id = loom_amdgpu_wait_counter_id_from_slot(slot);
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_drain_counter(
        builder, kind, reason, node_index, producer_node, counter_id));
  }
  return iree_ok_status();
}

static bool loom_amdgpu_wait_plan_producer_is_drained(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t producer_node,
    uint32_t counter_mask) {
  return iree_any_bit_set(builder->frontier_nodes[producer_node]
                              .drained_after_production_counter_mask,
                          counter_mask);
}

static bool loom_amdgpu_wait_plan_producer_target_count(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t producer_node,
    uint32_t slot, uint16_t* out_target_count) {
  *out_target_count = 0;
  const uint32_t counter_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
  const loom_amdgpu_wait_node_state_t* producer_state =
      &builder->node_states[producer_node];
  if (loom_amdgpu_wait_plan_producer_is_drained(builder, producer_node,
                                                counter_mask)) {
    return false;
  }
  if (producer_state->produced_counter_epoch[slot] !=
      builder->counter_epochs[slot]) {
    return false;
  }
  const uint32_t produced_position =
      producer_state->produced_counter_position[slot];
  if (produced_position == 0 ||
      produced_position <= builder->completed_position_counts[slot]) {
    return false;
  }
  const uint32_t outstanding_producer_position =
      produced_position - builder->completed_position_counts[slot];
  if (builder->outstanding_counts[slot] < outstanding_producer_position) {
    return false;
  }
  const uint32_t target_count =
      builder->outstanding_counts[slot] - outstanding_producer_position;
  *out_target_count =
      target_count > UINT16_MAX ? UINT16_MAX : (uint16_t)target_count;
  return true;
}

static bool loom_amdgpu_wait_plan_producer_is_complete_in_current_epoch(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t producer_node,
    uint32_t slot) {
  const loom_amdgpu_wait_node_state_t* producer_state =
      &builder->node_states[producer_node];
  const uint32_t produced_position =
      producer_state->produced_counter_position[slot];
  return producer_state->produced_counter_epoch[slot] ==
             builder->counter_epochs[slot] &&
         produced_position != 0 &&
         produced_position <= builder->completed_position_counts[slot];
}

static iree_status_t loom_amdgpu_wait_plan_handle_loop_entry_dependencies(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (builder->loop_entry_dependency_links == NULL) return iree_ok_status();
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  const loom_low_schedule_block_t* block =
      &builder->schedule->blocks[node->block_index];
  if (node->op != block->block->last_op) return iree_ok_status();

  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    const iree_host_size_t entry_slot_index =
        loom_amdgpu_wait_plan_loop_entry_slot_index(node->block_index, slot);
    uint32_t link_index =
        builder->loop_entry_dependency_links[entry_slot_index];
    if (link_index == LOOM_LOW_SCHEDULE_NODE_NONE) continue;

    const uint32_t counter_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
    const uint16_t counter_id = loom_amdgpu_wait_counter_id_from_slot(slot);
    uint16_t strictest_target_count = UINT16_MAX;
    uint32_t strictest_producer = LOOM_LOW_SCHEDULE_NODE_NONE;
    uint32_t strictest_consumer = LOOM_LOW_SCHEDULE_NODE_NONE;
    bool has_active_dependency = false;
    bool is_derived = true;
    while (link_index != LOOM_LOW_SCHEDULE_NODE_NONE) {
      const loom_amdgpu_wait_loop_dependency_t* link =
          &builder->dependency_links[link_index];
      uint16_t target_count = 0;
      const loom_low_schedule_node_t* producer =
          &builder->schedule->nodes[link->producer_node];
      const bool producer_is_local = producer->block_index == node->block_index;
      if ((producer_is_local &&
           loom_amdgpu_wait_plan_producer_is_drained(
               builder, link->producer_node, counter_mask)) ||
          (!producer_is_local &&
           loom_amdgpu_wait_plan_current_block_satisfies_producer(
               builder, link->producer_node, counter_mask))) {
        link_index = link->next_dependency;
        continue;
      }
      const bool target_is_derived =
          counter_id != LOOM_AMDGPU_WAIT_COUNTER_SMEM && producer_is_local &&
          loom_amdgpu_wait_plan_producer_target_count(
              builder, link->producer_node, slot, &target_count);
      if (!target_is_derived) {
        target_count = 0;
        is_derived = false;
      }
      has_active_dependency = true;
      const bool prefers_conservative_dependency =
          target_count == strictest_target_count && !target_is_derived;
      if (strictest_producer == LOOM_LOW_SCHEDULE_NODE_NONE ||
          target_count < strictest_target_count ||
          prefers_conservative_dependency) {
        strictest_target_count = target_count;
        strictest_producer = link->producer_node;
        strictest_consumer = link->consumer_node;
      }
      link_index = link->next_dependency;
    }
    if (!has_active_dependency) continue;

    const loom_amdgpu_wait_plan_reason_t reason =
        is_derived
            ? LOOM_AMDGPU_WAIT_PLAN_REASON_LOOP_ENTRY_DERIVED_SSA_USE
            : LOOM_AMDGPU_WAIT_PLAN_REASON_LOOP_ENTRY_CONSERVATIVE_SSA_USE;
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_wait_counter_at(
        builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED, /*flags=*/0, reason,
        node_index, strictest_producer, strictest_consumer, counter_id,
        strictest_target_count));
  }
  return iree_ok_status();
}

static void loom_amdgpu_wait_plan_seed_cyclic_frontiers(
    loom_amdgpu_wait_plan_builder_t* builder, uint16_t block_index) {
  if (builder->cyclic_frontiers == NULL) return;
  const loom_low_schedule_table_t* schedule = builder->schedule;
  const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    const iree_host_size_t frontier_index =
        loom_amdgpu_wait_plan_loop_entry_slot_index(block_index, slot);
    const loom_amdgpu_wait_loop_cyclic_frontier_t* frontier =
        &builder->cyclic_frontiers[frontier_index];
    if (!iree_any_bit_set(frontier->flags,
                          LOOM_AMDGPU_WAIT_LOOP_CYCLIC_FRONTIER_FLAG_VALID)) {
      continue;
    }
    builder->outstanding_counts[slot] = frontier->outstanding_count;
    builder->outstanding_write_counts[slot] = frontier->outstanding_write_count;
    builder->outstanding_workgroup_write_counts[slot] =
        frontier->outstanding_workgroup_write_count;

    uint32_t producer_position = 0;
    const uint32_t counter_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
    for (uint32_t i = frontier->producer_start_ordinal;
         i < block->scheduled_node_count; ++i) {
      const uint32_t packet_index = block->scheduled_node_start + i;
      const uint32_t node_index =
          schedule->scheduled_node_indices[packet_index];
      if ((loom_amdgpu_wait_plan_node_producer_counter_mask(builder,
                                                            node_index) &
           counter_mask) == 0) {
        continue;
      }
      loom_amdgpu_wait_node_state_t* node_state =
          &builder->node_states[node_index];
      node_state->produced_counter_epoch[slot] = builder->counter_epochs[slot];
      node_state->produced_counter_position[slot] = ++producer_position;
    }
    IREE_ASSERT_EQ(producer_position, frontier->outstanding_count);
  }
}

static iree_status_t loom_amdgpu_wait_plan_verify_cyclic_frontiers(
    const loom_amdgpu_wait_plan_builder_t* builder, uint16_t block_index) {
  if (builder->cyclic_frontiers == NULL) return iree_ok_status();
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    const iree_host_size_t frontier_index =
        loom_amdgpu_wait_plan_loop_entry_slot_index(block_index, slot);
    const loom_amdgpu_wait_loop_cyclic_frontier_t* frontier =
        &builder->cyclic_frontiers[frontier_index];
    if (!iree_any_bit_set(frontier->flags,
                          LOOM_AMDGPU_WAIT_LOOP_CYCLIC_FRONTIER_FLAG_VALID)) {
      continue;
    }
    if (builder->outstanding_counts[slot] != frontier->outstanding_count ||
        builder->outstanding_write_counts[slot] !=
            frontier->outstanding_write_count ||
        builder->outstanding_workgroup_write_counts[slot] !=
            frontier->outstanding_workgroup_write_count) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "AMDGPU cyclic wait frontier for block %u counter %u is not a "
          "fixed point: expected %" PRIu32 " outstanding but observed %" PRIu32,
          block_index, loom_amdgpu_wait_counter_id_from_slot(slot),
          frontier->outstanding_count, builder->outstanding_counts[slot]);
    }
  }
  return iree_ok_status();
}

static loom_amdgpu_wait_plan_reason_t
loom_amdgpu_wait_plan_storage_release_reason(
    const loom_low_storage_release_action_t* action) {
  const loom_amdgpu_wait_plan_reason_t reason =
      (loom_amdgpu_wait_plan_reason_t)action->release_reason_id;
  IREE_ASSERT(loom_amdgpu_wait_plan_reason_is_storage_release(reason));
  return reason;
}

static bool loom_amdgpu_wait_plan_storage_release_is_satisfied(
    const loom_amdgpu_wait_plan_builder_t* builder,
    const loom_low_storage_release_action_t* action,
    const loom_low_storage_lease_record_t* lease_record) {
  const uint32_t counter_mask =
      loom_amdgpu_wait_counter_mask(action->release_class_id);
  const uint32_t slot =
      loom_amdgpu_wait_counter_slot_from_id(action->release_class_id);
  IREE_ASSERT_LT(lease_record->node_index, builder->schedule->node_count);
  const loom_amdgpu_wait_node_state_t* producer_state =
      &builder->node_states[lease_record->node_index];
  const uint32_t producer_block =
      builder->schedule->nodes[lease_record->node_index].block_index;
  const uint32_t insertion_block = action->block_index;
  if (loom_amdgpu_wait_plan_producer_is_drained(
          builder, lease_record->node_index, counter_mask)) {
    return true;
  }
  if (producer_block == insertion_block) {
    return producer_state->produced_counter_epoch[slot] !=
           builder->counter_epochs[slot];
  }
  return loom_amdgpu_wait_plan_producer_is_drained(
             builder, lease_record->node_index, counter_mask) ||
         loom_amdgpu_wait_plan_current_block_satisfies_producer(
             builder, lease_record->node_index, counter_mask);
}

static loom_amdgpu_wait_plan_action_flags_t
loom_amdgpu_wait_plan_storage_release_action_flags(
    const loom_amdgpu_wait_plan_builder_t* builder,
    const loom_low_storage_release_action_t* action) {
  const loom_amdgpu_wait_plan_action_flags_t allocation_action_flags =
      LOOM_AMDGPU_WAIT_PLAN_ACTION_FLAG_STORAGE_RELEASE;
  const loom_low_storage_release_action_index_t* index =
      &builder->storage_release_action_index;
  if (index->first_action_indices == NULL) return 0;
  for (uint32_t action_index =
           index->first_action_indices[action->insertion_node_index];
       action_index != LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE;
       action_index = index->next_action_indices[action_index]) {
    const loom_low_storage_release_action_t* indexed_action =
        &builder->allocation->storage_release_actions[action_index];
    if (indexed_action->lease_record_index == action->lease_record_index &&
        indexed_action->release_class_id == action->release_class_id &&
        indexed_action->release_action_id == action->release_action_id &&
        indexed_action->release_reason_id == action->release_reason_id) {
      // The generic packet-hazard builder reports indexed allocation actions.
      // Mark this wait so the target provider does not report it a second time.
      return allocation_action_flags;
    }
  }
  return 0;
}

static bool loom_amdgpu_wait_plan_storage_release_is_ordered_vmem_reuse(
    const loom_amdgpu_wait_plan_builder_t* builder,
    const loom_low_storage_release_action_t* action,
    const loom_low_storage_lease_record_t* lease_record,
    loom_amdgpu_wait_plan_reason_t reason) {
  if (reason != LOOM_AMDGPU_WAIT_PLAN_REASON_READ_RESULT_REUSE ||
      action->release_class_id != LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD ||
      !loom_amdgpu_processor_properties_have_scheduling(
          builder->processor_properties,
          LOOM_AMDGPU_PROCESSOR_SCHEDULING_VMEM_RESULT_WRITES_IN_ORDER)) {
    return false;
  }
  const loom_amdgpu_vmem_result_order_class_t producer_order_class =
      builder->frontier_nodes[lease_record->node_index].vmem_result_order_class;
  const loom_amdgpu_vmem_result_order_class_t consumer_order_class =
      builder->frontier_nodes[action->insertion_node_index]
          .vmem_result_order_class;
  return producer_order_class != LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE &&
         producer_order_class != LOOM_AMDGPU_VMEM_RESULT_ORDER_UNKNOWN &&
         producer_order_class == consumer_order_class;
}

static loom_amdgpu_wait_xcnt_group_t loom_amdgpu_wait_plan_node_xcnt_group(
    const loom_amdgpu_wait_node_state_t* node_state) {
  if (iree_any_bit_set(node_state->flags,
                       LOOM_AMDGPU_WAIT_NODE_STATE_XCNT_VMEM_PRODUCER)) {
    return LOOM_AMDGPU_WAIT_XCNT_GROUP_VMEM;
  }
  if (iree_any_bit_set(node_state->flags,
                       LOOM_AMDGPU_WAIT_NODE_STATE_XCNT_SMEM_PRODUCER)) {
    return LOOM_AMDGPU_WAIT_XCNT_GROUP_SMEM;
  }
  return LOOM_AMDGPU_WAIT_XCNT_GROUP_NONE;
}

static iree_status_t loom_amdgpu_wait_plan_handle_xcnt_pre_dependencies(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  loom_amdgpu_wait_node_state_t* node_state = &builder->node_states[node_index];
  const uint32_t x_slot =
      loom_amdgpu_wait_counter_slot_from_id(LOOM_AMDGPU_WAIT_COUNTER_X);
  const loom_amdgpu_wait_xcnt_group_flags_t incoming_group_flags =
      loom_amdgpu_wait_frontier_active_xcnt_groups(&builder->frontier);

  if (iree_any_bit_set(node_state->flags,
                       LOOM_AMDGPU_WAIT_NODE_STATE_XCNT_IMPLICIT_DRAIN)) {
    if (builder->outstanding_counts[x_slot] != 0 || incoming_group_flags != 0) {
      loom_amdgpu_wait_plan_apply_counter_progress(
          builder, node_index, LOOM_LOW_SCHEDULE_NODE_NONE,
          LOOM_AMDGPU_WAIT_COUNTER_X, /*target_count=*/0);
    }
    return iree_ok_status();
  }

  const loom_amdgpu_wait_xcnt_group_t current_group =
      loom_amdgpu_wait_plan_node_xcnt_group(node_state);
  if (current_group != LOOM_AMDGPU_WAIT_XCNT_GROUP_NONE &&
      builder->xcnt_group != LOOM_AMDGPU_WAIT_XCNT_GROUP_NONE &&
      current_group != builder->xcnt_group) {
    // Gfx125x hardware drains XCNT between interleaved scalar-memory and
    // vector-memory translation groups. Reflect that progress in the planner
    // without emitting a redundant wait packet.
    node_state->implicit_wait_counter_mask |= LOOM_AMDGPU_WAIT_COUNTER_MASK_X;
    loom_amdgpu_wait_plan_apply_counter_progress(
        builder, node_index, LOOM_LOW_SCHEDULE_NODE_NONE,
        LOOM_AMDGPU_WAIT_COUNTER_X, /*target_count=*/0);
  }
  if (current_group != LOOM_AMDGPU_WAIT_XCNT_GROUP_NONE) {
    loom_amdgpu_wait_frontier_prepare_xcnt_producer(
        &builder->frontier, (loom_amdgpu_wait_xcnt_group_flags_t)current_group);
  }

  if (iree_any_bit_set(node_state->flags,
                       LOOM_AMDGPU_WAIT_NODE_STATE_WRITES_EXEC) &&
      ((builder->xcnt_group == LOOM_AMDGPU_WAIT_XCNT_GROUP_VMEM &&
        builder->outstanding_counts[x_slot] != 0) ||
       iree_any_bit_set(
           loom_amdgpu_wait_frontier_active_xcnt_groups(&builder->frontier),
           LOOM_AMDGPU_WAIT_XCNT_GROUP_FLAG_VMEM))) {
    // Every outstanding VMEM translation retains the EXEC value observed at
    // issue. An EXEC definition therefore requires the entire VMEM XCNT group
    // to retire.
    return loom_amdgpu_wait_plan_drain_counter(
        builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED,
        LOOM_AMDGPU_WAIT_PLAN_REASON_XCNT_EXEC_REUSE, node_index,
        LOOM_LOW_SCHEDULE_NODE_NONE, LOOM_AMDGPU_WAIT_COUNTER_X);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_handle_storage_release_action(
    loom_amdgpu_wait_plan_builder_t* builder,
    const loom_low_storage_release_action_t* action) {
  IREE_ASSERT_EQ(action->release_action_id,
                 LOOM_AMDGPU_WAIT_PLAN_RESIDUAL_ACTION_WAIT_PACKET);
  IREE_ASSERT_EQ(action->required_progress, 1u);
  IREE_ASSERT_LT(action->lease_record_index,
                 builder->allocation->storage_leases.record_count);

  const loom_low_storage_lease_record_t* lease_record =
      &builder->allocation->storage_leases.records[action->lease_record_index];
  const loom_amdgpu_wait_plan_reason_t reason =
      loom_amdgpu_wait_plan_storage_release_reason(action);
  if (loom_amdgpu_wait_plan_storage_release_is_satisfied(builder, action,
                                                         lease_record)) {
    return iree_ok_status();
  }
  if (loom_amdgpu_wait_plan_storage_release_is_ordered_vmem_reuse(
          builder, action, lease_record, reason)) {
    return iree_ok_status();
  }
  uint16_t target_count = 0;
  const uint32_t producer_block =
      builder->schedule->nodes[lease_record->node_index].block_index;
  if (producer_block == action->block_index) {
    const uint32_t slot =
        loom_amdgpu_wait_counter_slot_from_id(action->release_class_id);
    if (!loom_amdgpu_wait_plan_producer_target_count(
            builder, lease_record->node_index, slot, &target_count)) {
      target_count = 0;
    }
  }
  if (action->release_class_id == LOOM_AMDGPU_WAIT_COUNTER_X &&
      loom_amdgpu_wait_plan_node_xcnt_group(
          &builder->node_states[action->insertion_node_index]) ==
          LOOM_AMDGPU_WAIT_XCNT_GROUP_VMEM) {
    // VMEM translations are ordered. A VMEM packet that overwrites an older
    // VMEM source makes the hardware internally progress XCNT just far enough
    // to release that source. No wait instruction is required. Cross-block
    // counts are deliberately not reconstructed; the per-block marker records
    // the specific producer proven retired on this path.
    if (producer_block == action->block_index) {
      target_count = loom_amdgpu_wait_plan_normalize_target_count(
          builder, LOOM_AMDGPU_WAIT_COUNTER_X, target_count);
      loom_amdgpu_wait_plan_apply_counter_progress(
          builder, action->insertion_node_index, lease_record->node_index,
          LOOM_AMDGPU_WAIT_COUNTER_X, target_count);
    } else {
      loom_amdgpu_wait_plan_record_current_block_drained_producer(
          builder, lease_record->node_index, LOOM_AMDGPU_WAIT_COUNTER_MASK_X);
    }
    return iree_ok_status();
  }
  return loom_amdgpu_wait_plan_wait_counter(
      builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED,
      loom_amdgpu_wait_plan_storage_release_action_flags(builder, action),
      reason, action->insertion_node_index, lease_record->node_index,
      action->release_class_id, target_count);
}

static iree_status_t loom_amdgpu_wait_plan_handle_storage_release_actions(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  const loom_low_allocation_table_t* allocation = builder->allocation;
  if (allocation == NULL ||
      builder->storage_release_action_index.first_action_indices == NULL) {
    return iree_ok_status();
  }
  const loom_low_storage_release_action_index_t* index =
      &builder->storage_release_action_index;
  for (uint32_t action_index = index->first_action_indices[node_index];
       action_index != LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE;
       action_index = index->next_action_indices[action_index]) {
    const loom_low_storage_release_action_t* action =
        &allocation->storage_release_actions[action_index];
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_wait_plan_handle_storage_release_action(builder, action));
  }
  return iree_ok_status();
}

static bool loom_amdgpu_wait_plan_storage_lease_is_live_at_node(
    const loom_amdgpu_wait_plan_builder_t* builder,
    const loom_low_allocation_storage_lease_t* lease,
    const loom_low_storage_lease_record_t* record, uint32_t node_index) {
  if (record->node_index == node_index) {
    return false;
  }
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  IREE_ASSERT_LT(node->block_index, builder->allocation->liveness.block_count);
  const loom_liveness_block_info_t* block_info =
      &builder->allocation->liveness.blocks[node->block_index];
  IREE_ASSERT_LE(node->scheduled_ordinal, UINT32_MAX - block_info->start_point);
  const uint32_t program_point =
      block_info->start_point + node->scheduled_ordinal;
  // A release action executes before the instruction at its end point, so the
  // lease remains active at that exact point. This distinction matters for
  // structural packets whose logical definition is coalesced into earlier
  // physical result writes.
  return lease->start_point <= program_point &&
         lease->end_point >= program_point;
}

static iree_status_t loom_amdgpu_wait_plan_handle_physical_write_range(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index,
    uint32_t continuation_producer_node,
    loom_low_allocation_location_kind_t location_kind,
    uint16_t descriptor_reg_class_id, uint32_t location_base,
    uint32_t location_count) {
  const loom_low_allocation_table_t* allocation = builder->allocation;
  if (location_count == 0 ||
      !loom_low_allocation_location_kind_is_register_like(location_kind)) {
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(allocation->storage_lease_instance_count,
                 allocation->storage_leases.record_count);
  IREE_ASSERT(loom_low_allocation_storage_lease_unit_index_is_enabled(
      allocation->storage_lease_unit_index));
  loom_low_allocation_storage_lease_unit_query_t query;
  loom_low_allocation_storage_lease_unit_query_initialize(
      allocation->storage_lease_unit_index,
      builder->schedule->target.descriptor_set, descriptor_reg_class_id,
      location_kind, location_base, location_count, &query);
  uint32_t storage_lease_index = 0;
  while (loom_low_allocation_storage_lease_unit_query_next(
      &query, &storage_lease_index)) {
    IREE_ASSERT_LT(storage_lease_index,
                   allocation->storage_lease_instance_count);
    const loom_low_allocation_storage_lease_t* lease =
        &allocation->storage_lease_instances[storage_lease_index];
    if (query.active_location !=
        iree_max(location_base, lease->location_base)) {
      continue;
    }
    IREE_ASSERT_LT(lease->lease_record_index,
                   allocation->storage_leases.record_count);
    const loom_low_storage_lease_record_t* record =
        &allocation->storage_leases.records[lease->lease_record_index];
    if (record->kind == LOOM_LOW_STORAGE_LEASE_RESULT_WRITE &&
        record->node_index == continuation_producer_node) {
      // A storage continuation writes a disjoint register part into the tied
      // source's allocation. It neither clobbers nor consumes the pending
      // result write that populates the preserved part.
      continue;
    }
    const bool incoming_lease_active =
        loom_amdgpu_wait_frontier_storage_lease_is_active(&builder->frontier,
                                                          storage_lease_index);
    const bool local_lease_active =
        loom_amdgpu_wait_plan_storage_lease_is_live_at_node(builder, lease,
                                                            record, node_index);
    if (!incoming_lease_active && !local_lease_active) {
      continue;
    }
    if (record->release_scope !=
            LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS ||
        record->release_action_id !=
            LOOM_AMDGPU_WAIT_PLAN_RESIDUAL_ACTION_WAIT_PACKET) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU physical result write overlaps a storage lease without a "
          "wait-counter release contract");
    }
    const loom_low_schedule_node_t* node =
        &builder->schedule->nodes[node_index];
    const loom_low_storage_release_action_t action = {
        .insertion_node_index = node_index,
        .block_index = node->block_index,
        .scheduled_ordinal = node->scheduled_ordinal,
        .release_class_id = record->release_class_id,
        .release_class_name = record->release_class_name,
        .release_action_id = record->release_action_id,
        .release_action_name = record->release_action_name,
        .release_reason_id = record->release_reason_id,
        .release_reason_name = record->release_reason_name,
        .required_progress = 1,
        .lease_record_index = lease->lease_record_index,
    };
    const loom_amdgpu_wait_plan_reason_t reason =
        loom_amdgpu_wait_plan_storage_release_reason(&action);
    if (incoming_lease_active &&
        loom_amdgpu_wait_plan_storage_release_is_ordered_vmem_reuse(
            builder, &action, record, reason)) {
      // The current same-class VMEM result write proves the older incoming
      // dynamic instance complete. Retire only that incoming lease; block-end
      // publication will represent the current instruction's new instance.
      loom_amdgpu_wait_frontier_retire_storage_lease(&builder->frontier,
                                                     storage_lease_index);
      continue;
    }
    if (incoming_lease_active) {
      // Incoming counts are path-dependent and intentionally absent from the
      // block-local counter positions. A full wait is the only threshold that
      // proves this concrete lease released on every predecessor path.
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_wait_counter(
          builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED,
          loom_amdgpu_wait_plan_storage_release_action_flags(builder, &action),
          reason, node_index, record->node_index, record->release_class_id,
          /*target_count=*/0));
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_wait_plan_handle_storage_release_action(builder, &action));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_handle_cycle_scratch_writes(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index,
    const loom_low_move_group_t* move_group) {
  const loom_low_allocation_table_t* allocation = builder->allocation;
  if (move_group->scratch_move_index_count == 0) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < move_group->scratch_move_index_count; ++i) {
    const iree_host_size_t move_index =
        allocation
            ->scratch_move_indices[move_group->scratch_move_index_start + i];
    const loom_low_move_location_t* destination =
        &allocation->moves[move_index].destination;
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_handle_physical_write_range(
        builder, node_index, LOOM_LOW_SCHEDULE_NODE_NONE,
        destination->location_kind, destination->descriptor_reg_class_id,
        destination->location,
        /*location_count=*/1));
  }
  return iree_ok_status();
}

static uint32_t loom_amdgpu_wait_plan_result_storage_continuation_producer_node(
    const loom_amdgpu_wait_plan_builder_t* builder,
    const loom_low_schedule_node_t* node, uint16_t result_index) {
  if (node->descriptor == NULL) {
    return LOOM_LOW_SCHEDULE_NODE_NONE;
  }
  IREE_ASSERT_LT(result_index, node->descriptor->result_count);
  const loom_low_operand_t* result_operand =
      &builder->schedule->target.descriptor_set
           ->operands[node->descriptor->operand_start + result_index];
  if (!iree_any_bit_set(result_operand->flags,
                        LOOM_LOW_OPERAND_FLAG_STORAGE_CONTINUATION)) {
    return LOOM_LOW_SCHEDULE_NODE_NONE;
  }

  const loom_tied_result_t* tied_results = loom_op_tied_results(node->op);
  for (uint16_t i = 0; i < node->op->tied_result_count; ++i) {
    if (tied_results[i].result_index != result_index) {
      continue;
    }
    IREE_ASSERT_LT(tied_results[i].operand_index, node->operand_count);
    const loom_value_ordinal_t* operand_ordinals =
        loom_low_schedule_node_const_operand_ordinals(node);
    const loom_value_ordinal_t source_ordinal =
        operand_ordinals[tied_results[i].operand_index];
    IREE_ASSERT_LT(source_ordinal, builder->schedule->value_count);
    return builder->producer_nodes[source_ordinal];
  }
  IREE_ASSERT_UNREACHABLE(
      "storage-continuation result must have a tied source operand");
  return LOOM_LOW_SCHEDULE_NODE_NONE;
}

static iree_status_t loom_amdgpu_wait_plan_handle_materialized_result_writes(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (builder->allocation == NULL ||
      builder->allocation->storage_lease_instance_count == 0 ||
      !iree_any_bit_set(builder->node_states[node_index].flags,
                        LOOM_AMDGPU_WAIT_NODE_STATE_MATERIALIZES_RESULTS)) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (uint16_t i = 0; i < node->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_low_allocation_assignment_for_value_ordinal(
            builder->allocation, result_ordinals[i], NULL);
    if (assignment == NULL) {
      continue;
    }
    const uint32_t continuation_producer_node =
        loom_amdgpu_wait_plan_result_storage_continuation_producer_node(
            builder, node, i);
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_handle_physical_write_range(
        builder, node_index, continuation_producer_node,
        assignment->location_kind, assignment->descriptor_reg_class_id,
        assignment->location_base, assignment->location_count));
  }
  if (node->op != NULL &&
      (loom_low_copy_isa(node->op) || loom_low_move_isa(node->op) ||
       loom_low_slice_isa(node->op) || loom_low_concat_isa(node->op))) {
    const loom_low_allocation_packet_move_group_t* group =
        loom_low_allocation_find_packet_move_group_by_source_ordinal(
            builder->allocation, node->source_ordinal);
    if (group != NULL) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_handle_cycle_scratch_writes(
          builder, node_index, &group->move_group));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_handle_edge_copy_writes(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  const loom_low_allocation_edge_copy_group_t* group =
      loom_amdgpu_wait_plan_edge_copy_group(builder, node_index);
  if (group == NULL) {
    return iree_ok_status();
  }
  const loom_low_allocation_table_t* allocation = builder->allocation;
  if (allocation->storage_lease_instance_count == 0) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < group->copy_count; ++i) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &allocation->edge_copies[group->copy_start + i];
    if (!loom_amdgpu_wait_plan_edge_copy_materializes(builder, edge_copy)) {
      continue;
    }
    const loom_low_allocation_assignment_t* destination =
        &allocation->assignments[edge_copy->destination_assignment_index];
    const uint32_t location_base =
        destination->location_base + edge_copy->destination_unit_offset;
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_handle_physical_write_range(
        builder, node_index, LOOM_LOW_SCHEDULE_NODE_NONE,
        destination->location_kind, destination->descriptor_reg_class_id,
        location_base, edge_copy->unit_count));
  }
  return loom_amdgpu_wait_plan_handle_cycle_scratch_writes(builder, node_index,
                                                           &group->move_group);
}

static uint32_t loom_amdgpu_wait_plan_outstanding_counter_mask(
    const uint32_t outstanding_counts[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT],
    uint32_t counter_mask) {
  uint32_t outstanding_counter_mask = 0;
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    const uint32_t slot_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
    if ((counter_mask & slot_mask) != 0 && outstanding_counts[slot] != 0) {
      outstanding_counter_mask |= slot_mask;
    }
  }
  return outstanding_counter_mask;
}

static iree_status_t
loom_amdgpu_wait_plan_handle_cross_block_memory_dependencies(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  const loom_amdgpu_wait_frontier_node_t* frontier_node =
      &builder->frontier_nodes[node_index];
  if (builder->frontier.memory.static_outgoing_states == NULL ||
      (frontier_node->read_space_flags == 0 &&
       frontier_node->write_space_flags == 0)) {
    return iree_ok_status();
  }
  const uint32_t counter_mask =
      loom_amdgpu_wait_frontier_memory_dependency_mask(&builder->frontier,
                                                       frontier_node);
  if (counter_mask == 0) {
    return iree_ok_status();
  }
  return loom_amdgpu_wait_plan_drain_mask(
      builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED,
      LOOM_AMDGPU_WAIT_PLAN_REASON_MEMORY_EFFECT, node_index,
      LOOM_LOW_SCHEDULE_NODE_NONE, counter_mask);
}

static iree_status_t loom_amdgpu_wait_plan_handle_consumer(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  uint32_t active_counter_mask = 0;
  uint32_t active_producers[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
  loom_amdgpu_wait_plan_reason_t
      active_reasons[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
  uint16_t target_counts[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    active_producers[slot] = LOOM_LOW_SCHEDULE_NODE_NONE;
    active_reasons[slot] = LOOM_AMDGPU_WAIT_PLAN_REASON_UNKNOWN;
    target_counts[slot] = UINT16_MAX;
  }
  for (uint32_t link_index =
           builder->first_dependency_link_by_consumer[node_index];
       link_index != LOOM_LOW_SCHEDULE_NODE_NONE;) {
    const loom_amdgpu_wait_loop_dependency_t* link =
        &builder->dependency_links[link_index];
    for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT;
         ++slot) {
      const uint32_t counter_mask =
          loom_amdgpu_wait_counter_mask_from_slot(slot);
      if ((link->counter_mask & counter_mask) == 0) {
        continue;
      }
      const uint32_t producer_block =
          builder->schedule->nodes[link->producer_node].block_index;
      const uint32_t consumer_block =
          builder->schedule->nodes[node_index].block_index;
      const bool producer_precedes_consumer =
          builder->schedule->nodes[link->producer_node].scheduled_ordinal <
          builder->schedule->nodes[node_index].scheduled_ordinal;
      uint16_t target_count = 0;
      loom_amdgpu_wait_plan_reason_t reason =
          (loom_amdgpu_wait_plan_reason_t)link->reason_id;
      if (producer_block == consumer_block && producer_precedes_consumer) {
        // Epochs are block-local because outstanding counts reset at block
        // entry. Within one block, a newer epoch or drained producer marker
        // means an earlier wait already drained the producer.
        if (!loom_amdgpu_wait_plan_producer_target_count(
                builder, link->producer_node, slot, &target_count)) {
          continue;
        }
        if (counter_mask == LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM) {
          // Scalar-memory result dependencies require the producing SMEM
          // packet to be fully drained before a later packet consumes the
          // SGPR. Partial lgkmcnt waits are insufficient for SMEM data or
          // address dependencies even when the producer is oldest among
          // several outstanding scalar-memory packets.
          target_count = 0;
        }
      } else if (producer_block == consumer_block) {
        const loom_cfg_loop_interval_t* cyclic_interval =
            loom_amdgpu_wait_loop_analysis_cyclic_interval(
                &builder->loop_analysis, link->producer_node, node_index);
        const iree_host_size_t frontier_index =
            loom_amdgpu_wait_plan_loop_entry_slot_index(
                (uint16_t)consumer_block, slot);
        const loom_amdgpu_wait_loop_cyclic_frontier_t* derived_frontier =
            cyclic_interval != NULL && builder->cyclic_frontiers != NULL
                ? &builder->cyclic_frontiers[frontier_index]
                : NULL;
        const bool has_derived_frontier =
            derived_frontier != NULL &&
            iree_any_bit_set(derived_frontier->flags,
                             LOOM_AMDGPU_WAIT_LOOP_CYCLIC_FRONTIER_FLAG_VALID);
        if (has_derived_frontier) {
          if (derived_frontier->outstanding_count == 0) continue;
          if (!loom_amdgpu_wait_plan_producer_target_count(
                  builder, link->producer_node, slot, &target_count)) {
            if (loom_amdgpu_wait_plan_producer_is_complete_in_current_epoch(
                    builder, link->producer_node, slot) ||
                loom_amdgpu_wait_plan_producer_is_drained(
                    builder, link->producer_node, counter_mask) ||
                loom_amdgpu_wait_plan_current_block_satisfies_producer(
                    builder, link->producer_node, counter_mask)) {
              continue;
            }
            target_count = 0;
            reason =
                LOOM_AMDGPU_WAIT_PLAN_REASON_LOOP_CARRIED_CONSERVATIVE_SSA_USE;
          } else {
            reason = LOOM_AMDGPU_WAIT_PLAN_REASON_LOOP_CARRIED_DERIVED_SSA_USE;
          }
        } else {
          if (loom_amdgpu_wait_plan_producer_is_drained(
                  builder, link->producer_node, counter_mask) ||
              loom_amdgpu_wait_plan_current_block_satisfies_producer(
                  builder, link->producer_node, counter_mask)) {
            continue;
          }
          target_count = 0;
          if (cyclic_interval != NULL) {
            reason =
                LOOM_AMDGPU_WAIT_PLAN_REASON_LOOP_CARRIED_CONSERVATIVE_SSA_USE;
          }
        }
      } else {
        // Across block boundaries, the producer is safe only if a wait in its
        // own block drained it before control could reach the consumer block.
        if (loom_amdgpu_wait_plan_producer_is_drained(
                builder, link->producer_node, counter_mask) ||
            loom_amdgpu_wait_plan_current_block_satisfies_producer(
                builder, link->producer_node, counter_mask)) {
          continue;
        }
        target_count = 0;
        if (loom_amdgpu_wait_loop_analysis_cyclic_interval(
                &builder->loop_analysis, link->producer_node, node_index) !=
            NULL) {
          reason =
              LOOM_AMDGPU_WAIT_PLAN_REASON_LOOP_CARRIED_CONSERVATIVE_SSA_USE;
        }
      }
      active_counter_mask |= counter_mask;
      const bool prefers_conservative_reason =
          target_count == target_counts[slot] &&
          reason ==
              LOOM_AMDGPU_WAIT_PLAN_REASON_LOOP_CARRIED_CONSERVATIVE_SSA_USE;
      if (active_producers[slot] == LOOM_LOW_SCHEDULE_NODE_NONE ||
          target_count < target_counts[slot] || prefers_conservative_reason) {
        active_producers[slot] = link->producer_node;
        active_reasons[slot] = reason;
        target_counts[slot] = target_count;
      }
    }
    link_index = link->next_dependency;
  }

  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    const uint32_t counter_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
    if ((active_counter_mask & counter_mask) == 0) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_wait_counter(
        builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED, /*flags=*/0,
        active_reasons[slot], node_index, active_producers[slot],
        loom_amdgpu_wait_counter_id_from_slot(slot), target_counts[slot]));
  }
  return iree_ok_status();
}

static bool loom_amdgpu_wait_plan_node_is_trans_result_consumer(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (!loom_amdgpu_wait_plan_has_trans_result_state(builder)) {
    return false;
  }
  return iree_any_bit_set(builder->node_states[node_index].flags,
                          LOOM_AMDGPU_WAIT_NODE_STATE_USES_VECTOR_ALU);
}

static iree_status_t loom_amdgpu_wait_plan_handle_trans_result_use(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (!loom_amdgpu_wait_plan_node_is_trans_result_consumer(builder,
                                                           node_index)) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  const loom_op_t* op = node->op;
  if (op == NULL) {
    return iree_ok_status();
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_amdgpu_wait_plan_assignment(builder->allocation, operands[i]);
    uint32_t producer_node = LOOM_LOW_SCHEDULE_NODE_NONE;
    if (!loom_amdgpu_wait_plan_assignment_reuses_trans_result(
            builder, assignment, &producer_node)) {
      continue;
    }
    return loom_amdgpu_wait_plan_drain_counter(
        builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED,
        LOOM_AMDGPU_WAIT_PLAN_REASON_TRANS_RESULT_USE, node_index,
        producer_node, LOOM_AMDGPU_WAIT_COUNTER_ALU);
  }
  return iree_ok_status();
}

static bool loom_amdgpu_wait_plan_node_uses_scalar_alu(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  return iree_any_bit_set(builder->node_states[node_index].flags,
                          LOOM_AMDGPU_WAIT_NODE_STATE_USES_SCALAR_ALU);
}

static bool loom_amdgpu_wait_plan_node_uses_vector_alu(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  return iree_any_bit_set(builder->node_states[node_index].flags,
                          LOOM_AMDGPU_WAIT_NODE_STATE_USES_VECTOR_ALU);
}

static iree_status_t loom_amdgpu_wait_plan_handle_sgpr_read_hazard(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (!loom_amdgpu_wait_plan_has_sgpr_read_state(builder)) {
    return iree_ok_status();
  }
  const bool is_vector_alu =
      loom_amdgpu_wait_plan_node_uses_vector_alu(builder, node_index);
  const bool is_scalar_alu =
      loom_amdgpu_wait_plan_node_uses_scalar_alu(builder, node_index);
  if (!is_vector_alu && !is_scalar_alu) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  const loom_op_t* op = node->op;
  if (op == NULL) {
    return iree_ok_status();
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_amdgpu_wait_plan_assignment(builder->allocation, operands[i]);
    uint32_t producer_node = LOOM_LOW_SCHEDULE_NODE_NONE;
    if (!loom_amdgpu_wait_plan_sgpr_read_assignment_has_hazard(
            builder, assignment, is_vector_alu, is_scalar_alu,
            &producer_node)) {
      continue;
    }
    return loom_amdgpu_wait_plan_drain_counter(
        builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED,
        LOOM_AMDGPU_WAIT_PLAN_REASON_VALU_SGPR_READ, node_index, producer_node,
        LOOM_AMDGPU_WAIT_COUNTER_ALU);
  }
  return iree_ok_status();
}

static void loom_amdgpu_wait_plan_track_sgpr_reads(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (!loom_amdgpu_wait_plan_has_sgpr_read_state(builder) ||
      !loom_amdgpu_wait_plan_node_uses_vector_alu(builder, node_index)) {
    return;
  }
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  const loom_op_t* op = node->op;
  if (op == NULL) {
    return;
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_amdgpu_wait_plan_assignment(builder->allocation, operands[i]);
    loom_amdgpu_wait_plan_track_sgpr_read_assignment(builder, assignment);
  }
}

static void loom_amdgpu_wait_plan_record_sgpr_read_writes(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (!loom_amdgpu_wait_plan_has_sgpr_read_state(builder)) {
    return;
  }
  const bool is_vector_alu =
      loom_amdgpu_wait_plan_node_uses_vector_alu(builder, node_index);
  const bool is_scalar_alu =
      loom_amdgpu_wait_plan_node_uses_scalar_alu(builder, node_index);
  if (!is_vector_alu && !is_scalar_alu) {
    return;
  }
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  const loom_op_t* op = node->op;
  if (op == NULL) {
    return;
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_amdgpu_wait_plan_assignment(builder->allocation, results[i]);
    loom_amdgpu_wait_plan_record_sgpr_read_write_assignment(
        builder, assignment, is_vector_alu, is_scalar_alu, node_index);
  }
}

static iree_status_t loom_amdgpu_wait_plan_handle_barrier(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  const loom_amdgpu_wait_node_state_t* node_state =
      &builder->node_states[node_index];
  if ((node_state->barrier_counter_mask |
       node_state->workgroup_barrier_counter_mask) == 0) {
    return iree_ok_status();
  }
  const loom_amdgpu_wait_memory_space_flags_t generic_space =
      loom_amdgpu_wait_memory_space_flag(LOOM_LOW_MEMORY_SPACE_GENERIC);
  const loom_amdgpu_wait_memory_space_flags_t workgroup_space =
      loom_amdgpu_wait_memory_space_flag(LOOM_LOW_MEMORY_SPACE_WORKGROUP);
  uint32_t outstanding_counter_mask =
      loom_amdgpu_wait_plan_outstanding_counter_mask(
          builder->outstanding_counts, node_state->barrier_counter_mask);
  outstanding_counter_mask |=
      loom_amdgpu_wait_frontier_memory_query(
          &builder->frontier, generic_space,
          LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_READ |
              LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE) &
      node_state->barrier_counter_mask;
  outstanding_counter_mask |= loom_amdgpu_wait_plan_outstanding_counter_mask(
      builder->outstanding_workgroup_write_counts,
      node_state->workgroup_barrier_counter_mask);
  outstanding_counter_mask |= loom_amdgpu_wait_frontier_memory_query(
                                  &builder->frontier, workgroup_space,
                                  LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE) &
                              node_state->workgroup_barrier_counter_mask;
  if (outstanding_counter_mask == 0) {
    return iree_ok_status();
  }
  return loom_amdgpu_wait_plan_drain_mask(
      builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED,
      LOOM_AMDGPU_WAIT_PLAN_REASON_BARRIER, node_index,
      LOOM_LOW_SCHEDULE_NODE_NONE, outstanding_counter_mask);
}

static iree_status_t loom_amdgpu_wait_plan_handle_program_exit(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  if (!iree_any_bit_set(node->flags,
                        LOOM_LOW_SCHEDULE_NODE_FLAG_PROGRAM_EXIT_MEMORY)) {
    return iree_ok_status();
  }
  const loom_amdgpu_wait_memory_space_flags_t generic_space =
      loom_amdgpu_wait_memory_space_flag(LOOM_LOW_MEMORY_SPACE_GENERIC);
  uint32_t outstanding_counter_mask =
      loom_amdgpu_wait_plan_outstanding_counter_mask(
          builder->outstanding_counts,
          LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE);
  outstanding_counter_mask |= loom_amdgpu_wait_frontier_memory_query(
                                  &builder->frontier, generic_space,
                                  LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE) &
                              LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE;
  if (outstanding_counter_mask == 0) {
    return iree_ok_status();
  }
  return loom_amdgpu_wait_plan_drain_mask(
      builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED,
      LOOM_AMDGPU_WAIT_PLAN_REASON_PROGRAM_EXIT, node_index,
      LOOM_LOW_SCHEDULE_NODE_NONE, outstanding_counter_mask);
}

static void loom_amdgpu_wait_plan_increment_outstanding_counts(
    uint32_t outstanding_counts[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT],
    uint32_t counter_mask) {
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    const uint32_t slot_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
    if ((counter_mask & slot_mask) == 0) {
      continue;
    }
    IREE_ASSERT_NE(outstanding_counts[slot], UINT32_MAX);
    ++outstanding_counts[slot];
  }
}

static void loom_amdgpu_wait_plan_clear_trans_results(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (!loom_amdgpu_wait_plan_has_trans_result_state(builder)) {
    return;
  }
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  const loom_op_t* op = node->op;
  if (op == NULL) {
    return;
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_amdgpu_wait_plan_assignment(builder->allocation, results[i]);
    loom_amdgpu_wait_plan_clear_trans_result_assignment(builder, assignment);
  }
}

static void loom_amdgpu_wait_plan_record_trans_results(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (!loom_amdgpu_wait_plan_has_trans_result_state(builder)) {
    return;
  }
  const loom_amdgpu_wait_node_state_t* node_state =
      &builder->node_states[node_index];
  if ((node_state->trans_result_counter_mask &
       LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU) == 0) {
    return;
  }
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  const loom_op_t* op = node->op;
  if (op == NULL) {
    return;
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_amdgpu_wait_plan_assignment(builder->allocation, results[i]);
    loom_amdgpu_wait_plan_record_trans_result_assignment(builder, assignment,
                                                         node_index);
  }
}

static void loom_amdgpu_wait_plan_apply_trans_result_interval(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (!loom_amdgpu_wait_plan_has_trans_result_state(builder)) {
    return;
  }
  const loom_amdgpu_wait_node_state_t* node_state =
      &builder->node_states[node_index];
  const bool is_vector_alu = iree_any_bit_set(
      node_state->flags, LOOM_AMDGPU_WAIT_NODE_STATE_USES_VECTOR_ALU);
  const bool is_transcendental = iree_any_bit_set(
      node_state->flags, LOOM_AMDGPU_WAIT_NODE_STATE_TRANSCENDENTAL);
  loom_amdgpu_wait_plan_increment_trans_result_intervals(builder, is_vector_alu,
                                                         is_transcendental);
}

static bool loom_amdgpu_wait_plan_node_expires_trans_results(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (!loom_amdgpu_wait_plan_has_trans_result_state(builder)) {
    return false;
  }
  const loom_amdgpu_wait_frontier_node_t* frontier_node =
      &builder->frontier_nodes[node_index];
  const uint32_t expiring_counter_mask =
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM | LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS;
  return iree_any_bit_set(
      frontier_node->read_counter_mask | frontier_node->write_counter_mask,
      expiring_counter_mask);
}

static bool loom_amdgpu_wait_plan_node_materializes_results(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  return iree_any_bit_set(builder->node_states[node_index].flags,
                          LOOM_AMDGPU_WAIT_NODE_STATE_MATERIALIZES_RESULTS);
}

static iree_status_t loom_amdgpu_wait_plan_handle_vmem_result_reuse(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (builder->allocation == NULL ||
      builder->frontier.vmem_results.active_words == NULL ||
      !loom_amdgpu_wait_plan_node_materializes_results(builder, node_index)) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  const loom_amdgpu_vmem_result_order_class_t current_order_class =
      builder->frontier_nodes[node_index].vmem_result_order_class;
  const bool same_class_writes_are_ordered =
      current_order_class != LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE &&
      current_order_class != LOOM_AMDGPU_VMEM_RESULT_ORDER_UNKNOWN &&
      loom_amdgpu_processor_properties_have_scheduling(
          builder->processor_properties,
          LOOM_AMDGPU_PROCESSOR_SCHEDULING_VMEM_RESULT_WRITES_IN_ORDER);
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (uint16_t i = 0; i < node->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_low_allocation_assignment_for_value_ordinal(
            builder->allocation, result_ordinals[i], NULL);
    const loom_amdgpu_vmem_result_order_class_t pending_order_class =
        loom_amdgpu_wait_frontier_query_vmem_result(&builder->frontier,
                                                    assignment);
    if (pending_order_class == LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE ||
        (same_class_writes_are_ordered &&
         pending_order_class == current_order_class)) {
      continue;
    }
    return loom_amdgpu_wait_plan_drain_counter(
        builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED,
        LOOM_AMDGPU_WAIT_PLAN_REASON_READ_RESULT_REUSE, node_index,
        LOOM_LOW_SCHEDULE_NODE_NONE, LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_note_producer(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  loom_amdgpu_wait_node_state_t* node_state = &builder->node_states[node_index];
  const loom_amdgpu_wait_frontier_node_t* frontier_node =
      &builder->frontier_nodes[node_index];
  const uint32_t counter_mask =
      frontier_node->read_counter_mask | frontier_node->write_counter_mask |
      node_state->trans_result_counter_mask | node_state->source_counter_mask;
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    if ((counter_mask & loom_amdgpu_wait_counter_mask_from_slot(slot)) == 0) {
      continue;
    }
    node_state->produced_counter_epoch[slot] = builder->counter_epochs[slot];
    const uint32_t active_position =
        iree_math_saturating_add_u32(builder->completed_position_counts[slot],
                                     builder->outstanding_counts[slot]);
    node_state->produced_counter_position[slot] =
        iree_math_saturating_add_u32(active_position, 1u);
  }
  loom_amdgpu_wait_plan_increment_outstanding_counts(
      builder->outstanding_counts, counter_mask);
  loom_amdgpu_wait_plan_increment_outstanding_counts(
      builder->outstanding_write_counts, frontier_node->write_counter_mask);
  loom_amdgpu_wait_plan_increment_outstanding_counts(
      builder->outstanding_workgroup_write_counts,
      node_state->workgroup_write_counter_mask);
  const loom_amdgpu_wait_xcnt_group_t xcnt_group =
      loom_amdgpu_wait_plan_node_xcnt_group(node_state);
  if (xcnt_group != LOOM_AMDGPU_WAIT_XCNT_GROUP_NONE) {
    IREE_ASSERT(iree_any_bit_set(node_state->source_counter_mask,
                                 LOOM_AMDGPU_WAIT_COUNTER_MASK_X));
    IREE_ASSERT(builder->xcnt_group == LOOM_AMDGPU_WAIT_XCNT_GROUP_NONE ||
                builder->xcnt_group == xcnt_group);
    builder->xcnt_group = xcnt_group;
    loom_amdgpu_wait_frontier_note_xcnt_producer(
        &builder->frontier, (loom_amdgpu_wait_xcnt_group_flags_t)xcnt_group);
  }
  loom_amdgpu_wait_plan_clear_trans_results(builder, node_index);
  loom_amdgpu_wait_plan_record_trans_results(builder, node_index);
  loom_amdgpu_wait_plan_record_sgpr_read_writes(builder, node_index);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_process_node(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  loom_amdgpu_wait_node_state_t* node_state = &builder->node_states[node_index];

  // Loop-entry waits execute before branch-edge copies and the branch packet.
  // Their producer/consumer provenance still names the original loop use.
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_handle_loop_entry_dependencies(
      builder, node_index));
  // Branch-edge copies execute before the packet represented by the source
  // node. Protect them before crediting any progress supplied by that packet,
  // including an implicit XCNT drain from a hardware branch.
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_handle_edge_copy_writes(builder, node_index));
  // Architectural XCNT drains and group transitions then precede packet-local
  // result writes and allocation reuse. Apply that progress before checking
  // whether those packet-local writes overlap retained source storage.
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_handle_xcnt_pre_dependencies(builder, node_index));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_handle_materialized_result_writes(
      builder, node_index));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_handle_storage_release_actions(
      builder, node_index));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_handle_cross_block_memory_dependencies(builder,
                                                                   node_index));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_handle_consumer(builder, node_index));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_handle_trans_result_use(builder, node_index));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_handle_sgpr_read_hazard(builder, node_index));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_handle_barrier(builder, node_index));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_handle_program_exit(builder, node_index));
  if (node_state->explicit_wait_counter_mask != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_drain_mask(
        builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_EXPLICIT,
        LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET, node_index,
        LOOM_LOW_SCHEDULE_NODE_NONE, node_state->explicit_wait_counter_mask));
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_handle_vmem_result_reuse(builder, node_index));
  if (loom_amdgpu_wait_plan_node_expires_trans_results(builder, node_index)) {
    loom_amdgpu_wait_plan_expire_trans_results(builder);
  }
  loom_amdgpu_wait_plan_track_sgpr_reads(builder, node_index);
  loom_amdgpu_wait_plan_apply_trans_result_interval(builder, node_index);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_note_producer(builder, node_index));
  const loom_amdgpu_wait_frontier_node_t* frontier_node =
      &builder->frontier_nodes[node_index];
  const uint32_t reset_counter_mask = node_state->explicit_wait_counter_mask |
                                      node_state->implicit_wait_counter_mask;
  const uint32_t producer_counter_mask =
      frontier_node->read_counter_mask | frontier_node->write_counter_mask |
      node_state->trans_result_counter_mask | node_state->source_counter_mask;
  const iree_host_size_t node_event_count =
      (iree_host_size_t)iree_math_count_ones_u32(reset_counter_mask) +
      (iree_host_size_t)iree_math_count_ones_u32(producer_counter_mask);
  builder->progress_event_count += node_event_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_build_actions(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  const iree_host_size_t maximum_events_per_packet =
      2 * (iree_host_size_t)LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT;
  if (schedule->scheduled_node_count >
      IREE_HOST_SIZE_MAX / maximum_events_per_packet) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AMDGPU wait-plan progress event count exceeds host size");
  }
  for (iree_host_size_t block_index = 0; block_index < schedule->block_count;
       ++block_index) {
    const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
    ++builder->block_epoch;
    builder->current_block_full_drain_counter_mask = 0;
    builder->xcnt_group = LOOM_AMDGPU_WAIT_XCNT_GROUP_NONE;
    memset(builder->counter_epochs, 0, sizeof(builder->counter_epochs));
    memset(builder->completed_position_counts, 0,
           sizeof(builder->completed_position_counts));
    memset(builder->outstanding_counts, 0, sizeof(builder->outstanding_counts));
    memset(builder->outstanding_write_counts, 0,
           sizeof(builder->outstanding_write_counts));
    memset(builder->outstanding_workgroup_write_counts, 0,
           sizeof(builder->outstanding_workgroup_write_counts));
    builder->active_trans_result_vgpr_count = 0;
    if (builder->sgpr_read_register_count != 0) {
      memset(builder->sgpr_read_registers, 0,
             builder->sgpr_read_register_count *
                 sizeof(*builder->sgpr_read_registers));
    }
    loom_amdgpu_wait_plan_seed_cyclic_frontiers(builder, (uint16_t)block_index);
    loom_amdgpu_wait_frontier_begin_block(&builder->frontier,
                                          (uint16_t)block_index);
    for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
      const uint32_t packet_index = block->scheduled_node_start + i;
      IREE_ASSERT_LT(packet_index, schedule->scheduled_node_count);
      const uint32_t node_index =
          schedule->scheduled_node_indices[packet_index];
      IREE_ASSERT_LT(node_index, schedule->node_count);
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_wait_plan_process_node(builder, node_index));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_verify_cyclic_frontiers(
        builder, (uint16_t)block_index));
    loom_amdgpu_wait_frontier_end_block(&builder->frontier);
  }
  return iree_ok_status();
}

static void loom_amdgpu_wait_plan_emit_counter_progress(
    loom_low_packet_progress_emit_fn_t emit, void* emit_user_data,
    uint16_t counter_id, loom_low_packet_progress_action_t action,
    uint32_t units) {
  const loom_low_packet_progress_event_t event = {
      .progress_class_id = counter_id,
      .progress_class_name =
          loom_amdgpu_wait_counter_progress_class_name(counter_id),
      .action = action,
      .units = units,
  };
  emit(emit_user_data, &event);
}

static void loom_amdgpu_wait_plan_emit_counter_progress_mask(
    loom_low_packet_progress_emit_fn_t emit, void* emit_user_data,
    uint32_t counter_mask, loom_low_packet_progress_action_t action,
    uint32_t units) {
  while (counter_mask != 0) {
    const uint32_t slot =
        (uint32_t)iree_math_count_trailing_zeros_u32(counter_mask);
    const uint16_t counter_id = loom_amdgpu_wait_counter_id_from_slot(slot);
    loom_amdgpu_wait_plan_emit_counter_progress(emit, emit_user_data,
                                                counter_id, action, units);
    counter_mask &= counter_mask - 1;
  }
}

static void loom_amdgpu_wait_plan_progress_query(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_view_t* packet,
    loom_low_packet_progress_emit_fn_t emit, void* emit_user_data) {
  (void)schedule;
  (void)allocation;
  const loom_amdgpu_wait_plan_builder_t* builder =
      (const loom_amdgpu_wait_plan_builder_t*)user_data;
  const loom_amdgpu_wait_node_state_t* node_state =
      &builder->node_states[packet->node_index];
  const loom_amdgpu_wait_frontier_node_t* frontier_node =
      &builder->frontier_nodes[packet->node_index];
  loom_amdgpu_wait_plan_emit_counter_progress_mask(
      emit, emit_user_data,
      node_state->explicit_wait_counter_mask |
          node_state->implicit_wait_counter_mask,
      LOOM_LOW_PACKET_PROGRESS_ACTION_RESET, 0);
  const uint32_t producer_counter_mask =
      frontier_node->read_counter_mask | frontier_node->write_counter_mask |
      node_state->trans_result_counter_mask | node_state->source_counter_mask;
  loom_amdgpu_wait_plan_emit_counter_progress_mask(
      emit, emit_user_data, producer_counter_mask,
      LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE, 1);
}

static void loom_amdgpu_wait_plan_emit_hazard_action(
    const loom_amdgpu_wait_plan_action_t* action,
    loom_low_packet_hazard_plan_emit_fn_t emit, void* emit_user_data) {
  const uint32_t observed_progress = action->target_count;
  uint32_t required_progress = action->outstanding_before;
  if (required_progress <= observed_progress) {
    // Some wait-counter predicates are epoch or control-flow hazards whose
    // counted outstanding packets are block-local. Record the action as one
    // unsatisfied target progress unit instead of losing the residual hazard.
    required_progress = observed_progress + 1;
  }
  const uint32_t residual_progress = required_progress - observed_progress;
  const loom_low_packet_hazard_plan_event_t event = {
      .kind = LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION,
      .action_id = LOOM_AMDGPU_WAIT_PLAN_RESIDUAL_ACTION_WAIT_PACKET,
      .action_name = loom_amdgpu_wait_plan_residual_action_name(
          LOOM_AMDGPU_WAIT_PLAN_RESIDUAL_ACTION_WAIT_PACKET),
      .reason_id = (uint16_t)action->reason,
      .reason_name = loom_amdgpu_wait_plan_reason_name(action->reason),
      .producer_node_index = action->producer_node,
      .progress_class_id = action->counter_id,
      .progress_class_name =
          loom_amdgpu_wait_counter_progress_class_name(action->counter_id),
      .required_progress = required_progress,
      .observed_progress = observed_progress,
      .residual_progress = residual_progress,
  };
  emit(emit_user_data, &event);
}

static void loom_amdgpu_wait_plan_advance_hazard_action(
    loom_amdgpu_wait_plan_builder_t* builder) {
  while (builder->hazard_action_cursor < builder->action_count) {
    const loom_amdgpu_wait_plan_action_t* action =
        &builder->actions[builder->hazard_action_cursor];
    if (loom_amdgpu_wait_plan_action_is_residual_hazard(action)) {
      const loom_low_schedule_block_t* block =
          &builder->schedule->blocks[action->block_index];
      builder->next_hazard_packet_index =
          (iree_host_size_t)block->scheduled_node_start +
          action->scheduled_ordinal;
      return;
    }
    ++builder->hazard_action_cursor;
  }
  builder->next_hazard_packet_index = IREE_HOST_SIZE_MAX;
}

static void loom_amdgpu_wait_plan_hazard_query(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_table_t* progress,
    const loom_low_packet_view_t* packet,
    loom_low_packet_hazard_plan_emit_fn_t emit, void* emit_user_data) {
  (void)schedule;
  (void)allocation;
  (void)progress;
  loom_amdgpu_wait_plan_builder_t* builder =
      (loom_amdgpu_wait_plan_builder_t*)user_data;
  if (builder->next_hazard_packet_index != packet->packet_index) {
    return;
  }
  do {
    const loom_amdgpu_wait_plan_action_t* action =
        &builder->actions[builder->hazard_action_cursor];
    loom_amdgpu_wait_plan_emit_hazard_action(action, emit, emit_user_data);
    ++builder->hazard_action_cursor;
    loom_amdgpu_wait_plan_advance_hazard_action(builder);
  } while (builder->next_hazard_packet_index == packet->packet_index);
}

static iree_status_t loom_amdgpu_wait_plan_build_common_tables(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_packet_progress_table_t* progress = NULL;
  if (builder->allocation != NULL) {
    const loom_low_packet_progress_provider_t progress_provider = {
        .user_data = builder,
        .event_count = builder->progress_event_count,
        .query = loom_amdgpu_wait_plan_progress_query,
    };
    IREE_RETURN_IF_ERROR(loom_low_packet_progress_build(
        builder->schedule, builder->allocation, &progress_provider,
        builder->arena, &builder->progress));
    progress = &builder->progress;
  }

  builder->hazard_action_cursor = 0;
  loom_amdgpu_wait_plan_advance_hazard_action(builder);
  const loom_low_packet_hazard_plan_provider_t hazard_provider = {
      .user_data = builder,
      .event_count = builder->hazard_event_count,
      .query = loom_amdgpu_wait_plan_hazard_query,
  };
  return loom_low_packet_hazard_plan_build(
      builder->schedule, builder->allocation, progress, &hazard_provider,
      builder->arena, &builder->hazard_plan);
}

iree_status_t loom_amdgpu_wait_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* transient_arena,
    loom_amdgpu_wait_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_wait_plan_t){0};
  loom_amdgpu_wait_plan_builder_t builder = {
      .schedule = schedule,
      .allocation = allocation,
      .arena = arena,
      .transient_arena = transient_arena,
      .processor_properties =
          loom_amdgpu_target_processor_properties_from_resolved_target(
              &schedule->target),
  };
  loom_amdgpu_wait_packet_analyze_target(schedule->target.descriptor_set,
                                         &builder.wait_packet_target);
  loom_segmented_storage_initialize(
      sizeof(loom_amdgpu_wait_plan_action_segment_t),
      iree_alignof(loom_amdgpu_wait_plan_action_segment_t),
      &builder.action_stream.segments);
  loom_low_allocation_value_scratch_t scratch = {0};
  iree_status_t status =
      allocation != NULL
          ? loom_low_allocation_acquire_value_scratch(allocation, &scratch)
          : iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_plan_allocate(&builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_plan_classify_nodes(&builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_plan_relocate_loop_entry_dependencies(&builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_plan_allocate_physical_state(&builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_frontier_initialize(
        schedule, allocation, builder.frontier_nodes,
        builder.physical_registers.vgpr_count,
        builder.physical_registers.agpr_count,
        builder.loop_entry_drain_counter_masks, transient_arena,
        &builder.frontier);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_loop_analysis_build_cyclic_frontiers(
        &builder.loop_analysis, builder.loop_nodes,
        builder.first_dependency_link_by_consumer, builder.dependency_links,
        builder.dependency_link_count, transient_arena,
        &builder.cyclic_frontiers);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_plan_build_actions(&builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_plan_finalize_actions(&builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_plan_build_common_tables(&builder);
  }
  if (iree_status_is_ok(status)) {
    *out_plan = (loom_amdgpu_wait_plan_t){
        .schedule = schedule,
        .allocation = allocation,
        .progress = builder.progress,
        .hazard_plan = builder.hazard_plan,
        .actions = builder.actions,
        .action_count = builder.action_count,
    };
    if (builder.hazard_plan.progress == &builder.progress) {
      out_plan->hazard_plan.progress = &out_plan->progress;
    }
  }
  loom_low_allocation_release_value_scratch(&scratch);
  return status;
}

iree_status_t loom_amdgpu_wait_plan_format_json(
    const loom_amdgpu_wait_plan_t* plan, iree_string_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(builder);
  return loom_low_packet_hazard_plan_format_json(&plan->hazard_plan, builder);
}
