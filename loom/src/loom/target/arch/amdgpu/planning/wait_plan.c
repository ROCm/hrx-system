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
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/packet_hazard_plan_json.h"
#include "loom/ir/ir.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/planning/descriptor_semantics.h"
#include "loom/target/arch/amdgpu/planning/structural_packet.h"
#include "loom/target/arch/amdgpu/planning/wait_packet_tables.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_id/target_id.h"

#define LOOM_AMDGPU_WAIT_PLAN_ACTION_INDEX_NONE UINT32_MAX

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
} loom_amdgpu_wait_node_state_flag_bits_t;
typedef uint8_t loom_amdgpu_wait_node_state_flags_t;

typedef struct loom_amdgpu_wait_node_state_t {
  // Classification flags for this schedule node.
  loom_amdgpu_wait_node_state_flags_t flags;
  // Counters observed on WAIT_COUNTER hazard rows for this node.
  uint32_t hazard_counter_mask;
  // Counters drained by explicit counter effects on this node.
  uint32_t explicit_wait_counter_mask;
  // Counters produced by dependency-participating memory reads on this node.
  uint32_t read_counter_mask;
  // Counters produced by dependency-participating memory writes on this node.
  uint32_t write_counter_mask;
  // Counters produced by RDNA TRANS result hazards on this node.
  uint32_t trans_result_counter_mask;
  // Write counters whose effects are visible to workgroup-memory barriers.
  uint32_t workgroup_write_counter_mask;
  // Epoch for each counter produced by this node.
  uint32_t produced_counter_epoch[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
  // Monotonic producer position in the counter epoch. This is only meaningful
  // within |produced_counter_epoch|.
  uint32_t produced_counter_position[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
  // Counters produced by this node that were drained before control left the
  // producing block.
  uint32_t drained_after_production_counter_mask;
  // Counters that must be drained before this barrier node executes.
  uint32_t barrier_counter_mask;
  // Workgroup-memory write counters drained before this barrier executes.
  uint32_t workgroup_barrier_counter_mask;
} loom_amdgpu_wait_node_state_t;

typedef struct loom_amdgpu_wait_dependency_link_t {
  // Producer node for the dependency.
  uint32_t producer_node;
  // Next dependency link for the same consumer node.
  uint32_t next_link;
  // Counters produced by |producer_node| and needed by this use.
  uint32_t counter_mask;
  // Reason attached to waits emitted for this dependency link.
  loom_amdgpu_wait_plan_reason_t reason;
} loom_amdgpu_wait_dependency_link_t;

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

typedef struct loom_amdgpu_wait_plan_builder_t {
  // Schedule table being analyzed.
  const loom_low_schedule_table_t* schedule;
  // Optional physical assignment table for post-allocation hazards.
  const loom_low_allocation_table_t* allocation;
  // Arena that owns all output and scratch arrays.
  iree_arena_allocator_t* arena;
  // Processor facts selected by the low target, or NULL if unavailable.
  const loom_amdgpu_processor_info_t* processor;
  // Per-node counter classification.
  loom_amdgpu_wait_node_state_t* node_states;
  // Counter masks drained for a producer in the current block epoch.
  uint32_t* current_block_drained_counter_masks;
  // Block epoch for each producer-local drained counter mask.
  uint64_t* current_block_drained_epochs;
  // First relevant SSA dependency link per consumer node.
  uint32_t* first_dependency_link_by_consumer;
  // Relevant SSA dependency links.
  loom_amdgpu_wait_dependency_link_t* dependency_links;
  // First incoming low.br source per destination block-argument value ordinal.
  uint32_t* first_block_arg_source_by_value;
  // Incoming low.br source records for non-entry block arguments.
  loom_amdgpu_wait_block_arg_source_t* block_arg_sources;
  // Storage-release actions grouped by insertion node.
  loom_low_storage_release_action_index_t storage_release_action_index;
  // First residual hazard action per insertion node.
  uint32_t* first_hazard_action_by_node;
  // Next residual hazard action for the same insertion node.
  uint32_t* next_hazard_action;
  // DFS visit epoch per value while forwarding SSA wait dependencies.
  uint32_t* dependency_visit_epochs;
  // Number of populated dependency links.
  iree_host_size_t dependency_link_count;
  // Allocated dependency link capacity.
  iree_host_size_t dependency_link_capacity;
  // Number of populated block-argument source records.
  iree_host_size_t block_arg_source_count;
  // Monotonic DFS epoch for dependency forwarding.
  uint32_t dependency_visit_epoch;
  // Output action rows.
  loom_amdgpu_wait_plan_action_t* actions;
  // Number of populated action rows.
  iree_host_size_t action_count;
  // Allocated action row capacity.
  iree_host_size_t action_capacity;
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
    case LOOM_AMDGPU_WAIT_PLAN_REASON_STORE_SOURCE_REUSE:
      return IREE_SV("amdgpu.store_source_reuse");
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

iree_status_t loom_amdgpu_wait_counter_mask(uint16_t counter_id,
                                            uint32_t* out_mask) {
  switch (counter_id) {
    case LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD:
      *out_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD;
      return iree_ok_status();
    case LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE:
      *out_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE;
      return iree_ok_status();
    case LOOM_AMDGPU_WAIT_COUNTER_LDS:
      *out_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS;
      return iree_ok_status();
    case LOOM_AMDGPU_WAIT_COUNTER_SMEM:
      *out_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM;
      return iree_ok_status();
    case LOOM_AMDGPU_WAIT_COUNTER_ALU:
      *out_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU;
      return iree_ok_status();
    default:
      *out_mask = 0;
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown AMDGPU wait counter id %" PRIu16,
                              counter_id);
  }
}

static uint16_t loom_amdgpu_wait_counter_id_from_slot(uint32_t slot) {
  return (uint16_t)(slot + 1);
}

static uint32_t loom_amdgpu_wait_counter_mask_from_slot(uint32_t slot) {
  return 1u << slot;
}

static uint32_t loom_amdgpu_wait_counter_slot(uint16_t counter_id) {
  return counter_id - 1;
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

static iree_status_t loom_amdgpu_wait_effect_counter_mask(
    const loom_low_schedule_effect_use_t* effect, uint32_t* out_counter_mask) {
  if (effect->counter_id == LOOM_AMDGPU_WAIT_COUNTER_NONE) {
    *out_counter_mask = 0;
    return iree_ok_status();
  }
  return loom_amdgpu_wait_counter_mask(effect->counter_id, out_counter_mask);
}

static bool loom_amdgpu_wait_plan_reason_has_consumer(
    loom_amdgpu_wait_plan_reason_t reason) {
  switch (reason) {
    case LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE:
    case LOOM_AMDGPU_WAIT_PLAN_REASON_STORE_SOURCE_REUSE:
    case LOOM_AMDGPU_WAIT_PLAN_REASON_READ_RESULT_REUSE:
    case LOOM_AMDGPU_WAIT_PLAN_REASON_TRANS_RESULT_USE:
    case LOOM_AMDGPU_WAIT_PLAN_REASON_VALU_SGPR_READ:
    case LOOM_AMDGPU_WAIT_PLAN_REASON_MEMORY_EFFECT:
    case LOOM_AMDGPU_WAIT_PLAN_REASON_PROGRAM_EXIT:
    case LOOM_AMDGPU_WAIT_PLAN_REASON_MEMORY_SOURCE_REUSE:
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_wait_plan_reason_is_storage_release(
    loom_amdgpu_wait_plan_reason_t reason) {
  switch (reason) {
    case LOOM_AMDGPU_WAIT_PLAN_REASON_STORE_SOURCE_REUSE:
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
  IREE_RETURN_IF_ERROR(loom_low_storage_release_action_index_build(
      allocation->storage_release_actions,
      allocation->storage_release_action_count,
      LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_BY_INSERTION_NODE,
      schedule->node_count, builder->arena,
      &builder->storage_release_action_index));
  for (iree_host_size_t i = 0; i < allocation->storage_release_action_count;
       ++i) {
    const loom_low_storage_release_action_t* action =
        &allocation->storage_release_actions[i];
    const loom_low_schedule_node_t* node =
        &schedule->nodes[action->insertion_node_index];
    if (action->block_index != node->block_index ||
        action->scheduled_ordinal != node->scheduled_ordinal) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU storage release action does not match its insertion node");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_allocate(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  if (schedule->node_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, schedule->node_count, sizeof(*builder->node_states),
        (void**)&builder->node_states));
    memset(builder->node_states, 0,
           schedule->node_count * sizeof(*builder->node_states));

    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, schedule->node_count,
        sizeof(*builder->current_block_drained_counter_masks),
        (void**)&builder->current_block_drained_counter_masks));
    memset(builder->current_block_drained_counter_masks, 0,
           schedule->node_count *
               sizeof(*builder->current_block_drained_counter_masks));

    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, schedule->node_count,
        sizeof(*builder->current_block_drained_epochs),
        (void**)&builder->current_block_drained_epochs));
    memset(
        builder->current_block_drained_epochs, 0,
        schedule->node_count * sizeof(*builder->current_block_drained_epochs));

    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, schedule->node_count,
        sizeof(*builder->first_dependency_link_by_consumer),
        (void**)&builder->first_dependency_link_by_consumer));
    for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
      builder->first_dependency_link_by_consumer[i] =
          LOOM_LOW_SCHEDULE_NODE_NONE;
    }
  }
  return loom_amdgpu_wait_plan_build_storage_release_action_index(builder);
}

static iree_status_t loom_amdgpu_wait_plan_allocate_actions(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  const loom_low_allocation_table_t* allocation = builder->allocation;
  iree_host_size_t action_input_capacity = 0;
  if (!iree_host_size_checked_add(builder->dependency_link_capacity,
                                  schedule->effect_use_count,
                                  &action_input_capacity) ||
      !iree_host_size_checked_add(action_input_capacity,
                                  schedule->scheduled_node_count,
                                  &action_input_capacity) ||
      (allocation != NULL &&
       !iree_host_size_checked_add(action_input_capacity,
                                   schedule->scheduled_node_count,
                                   &action_input_capacity)) ||
      (allocation != NULL &&
       !iree_host_size_checked_add(action_input_capacity,
                                   allocation->storage_release_action_count,
                                   &action_input_capacity)) ||
      !iree_host_size_checked_mul(action_input_capacity,
                                  LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT,
                                  &builder->action_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU wait-plan action capacity overflows");
  }
  if (builder->action_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, builder->action_capacity, sizeof(*builder->actions),
        (void**)&builder->actions));
  }
  return iree_ok_status();
}

static const loom_low_allocation_assignment_t* loom_amdgpu_wait_plan_assignment(
    const loom_low_allocation_table_t* allocation, loom_value_id_t value_id) {
  if (allocation == NULL || value_id == LOOM_VALUE_ID_INVALID) {
    return NULL;
  }
  return loom_low_allocation_try_map_active_value_assignment(allocation,
                                                             value_id, NULL);
}

static iree_status_t loom_amdgpu_wait_plan_compute_node_forwards_dependencies(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index,
    bool* out_forwards_dependencies) {
  *out_forwards_dependencies = false;
  if (node_index >= builder->schedule->node_count) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  if (node->kind != LOOM_LOW_SCHEDULE_NODE_STRUCTURAL || node->op == NULL) {
    return iree_ok_status();
  }
  loom_amdgpu_structural_packet_info_t info = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_structural_packet_analyze(
      builder->allocation, node->op,
      LOOM_AMDGPU_STRUCTURAL_PACKET_ANALYSIS_FLAG_REQUIRE_ALLOCATION, &info));
  *out_forwards_dependencies = iree_any_bit_set(
      info.flags, LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_FORWARDS_DEPENDENCIES);
  return iree_ok_status();
}

static bool loom_amdgpu_wait_plan_node_forwards_dependencies(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  return node_index < builder->schedule->node_count &&
         iree_any_bit_set(builder->node_states[node_index].flags,
                          LOOM_AMDGPU_WAIT_NODE_STATE_FORWARDS_DEPENDENCIES);
}

static bool loom_amdgpu_wait_plan_needs_trans_result_state(
    const loom_amdgpu_wait_plan_builder_t* builder) {
  if (!loom_amdgpu_processor_has_scheduling(
          builder->processor,
          LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < builder->schedule->node_count; ++i) {
    if (builder->node_states[i].trans_result_counter_mask != 0) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_wait_plan_needs_sgpr_read_state(
    const loom_amdgpu_wait_plan_builder_t* builder) {
  return loom_amdgpu_processor_has_scheduling(
      builder->processor,
      LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_DEPCTR);
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
  if (!needs_trans_result_state && !needs_sgpr_read_state) {
    return iree_ok_status();
  }
  iree_host_size_t vgpr_count = 0;
  iree_host_size_t sgpr_count = 0;
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->location_kind !=
        LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
      continue;
    }
    const uint64_t end =
        (uint64_t)assignment->location_base + assignment->location_count;
    if (end > IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU physical register range exceeds host "
                              "size");
    }
    if (assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR &&
        needs_trans_result_state && (iree_host_size_t)end > vgpr_count) {
      vgpr_count = (iree_host_size_t)end;
    }
    if (assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR &&
        needs_sgpr_read_state && (iree_host_size_t)end > sgpr_count) {
      sgpr_count = (iree_host_size_t)end;
    }
  }
  if (vgpr_count != 0 && needs_trans_result_state) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, vgpr_count, sizeof(*builder->trans_result_vgprs),
        (void**)&builder->trans_result_vgprs));
    memset(builder->trans_result_vgprs, 0,
           vgpr_count * sizeof(*builder->trans_result_vgprs));
    builder->trans_result_vgpr_count = vgpr_count;
  }
  if (sgpr_count != 0 && needs_sgpr_read_state) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, sgpr_count, sizeof(*builder->sgpr_read_registers),
        (void**)&builder->sgpr_read_registers));
    memset(builder->sgpr_read_registers, 0,
           sgpr_count * sizeof(*builder->sgpr_read_registers));
    builder->sgpr_read_register_count = sgpr_count;
  }
  return iree_ok_status();
}

static void loom_amdgpu_wait_plan_append_action(
    loom_amdgpu_wait_plan_builder_t* builder,
    loom_amdgpu_wait_plan_action_t action) {
  IREE_ASSERT_LT(builder->action_count, builder->action_capacity);
  builder->actions[builder->action_count++] = action;
}

static void loom_amdgpu_wait_plan_append_dependency_link(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t producer_node,
    uint32_t consumer_node, uint32_t counter_mask,
    loom_amdgpu_wait_plan_reason_t reason) {
  IREE_ASSERT_LT(builder->dependency_link_count,
                 builder->dependency_link_capacity);
  loom_amdgpu_wait_dependency_link_t* link =
      &builder->dependency_links[builder->dependency_link_count];
  *link = (loom_amdgpu_wait_dependency_link_t){
      .producer_node = producer_node,
      .next_link = builder->first_dependency_link_by_consumer[consumer_node],
      .counter_mask = counter_mask,
      .reason = reason,
  };
  builder->first_dependency_link_by_consumer[consumer_node] =
      (uint32_t)builder->dependency_link_count++;
}

static bool loom_amdgpu_wait_plan_node_has_wait_consuming_operands(
    const loom_low_schedule_node_t* node) {
  return node->op == NULL || !loom_low_br_isa(node->op);
}

static iree_status_t loom_amdgpu_wait_plan_count_block_arg_sources(
    const loom_low_schedule_table_t* schedule,
    iree_host_size_t* out_block_arg_count, iree_host_size_t* out_source_count) {
  iree_host_size_t block_arg_count = 0;
  for (iree_host_size_t i = 0; i < schedule->block_count; ++i) {
    const loom_block_t* block = schedule->blocks[i].block;
    if (block == NULL) {
      continue;
    }
    if (!iree_host_size_checked_add(block_arg_count, block->arg_count,
                                    &block_arg_count)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU wait block argument count overflows host size");
    }
  }

  iree_host_size_t source_count = 0;
  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    const loom_low_schedule_node_t* node = &schedule->nodes[i];
    if (node->op == NULL || !loom_low_br_isa(node->op)) {
      continue;
    }
    const loom_block_t* dest = loom_low_br_dest(node->op);
    if (dest == NULL) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "low.br has no destination block");
    }
    if (node->operand_count != dest->arg_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low.br payload count does not match destination block arguments");
    }
    if (!iree_host_size_checked_add(source_count, node->operand_count,
                                    &source_count)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU wait block argument source count overflows host size");
    }
  }

  *out_block_arg_count = block_arg_count;
  *out_source_count = source_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_compute_block_arg_offsets(
    const loom_low_schedule_table_t* schedule, uint32_t* block_arg_offsets) {
  iree_host_size_t next_offset = 0;
  for (iree_host_size_t i = 0; i < schedule->block_count; ++i) {
    if (next_offset > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU wait block argument offset exceeds uint32_t");
    }
    block_arg_offsets[i] = (uint32_t)next_offset;
    const loom_block_t* block = schedule->blocks[i].block;
    if (block != NULL && !iree_host_size_checked_add(
                             next_offset, block->arg_count, &next_offset)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU wait block argument offset overflows host size");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_initialize_block_arg_ordinals(
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
    if (block == NULL || block->region_index >= schedule->block_count ||
        schedule->blocks[block->region_index].block != block) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU wait block argument is outside scheduled region");
    }
    const uint16_t arg_index = loom_value_def_index(value);
    if (arg_index >= block->arg_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU wait block argument index exceeds owning block arity");
    }
    const uint32_t offset = block_arg_offsets[block->region_index] + arg_index;
    if (offset >= block_arg_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU wait block argument ordinal offset is out of range");
    }
    block_arg_ordinals[offset] = i;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_build_block_arg_sources(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  iree_host_size_t block_arg_count = 0;
  iree_host_size_t source_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_count_block_arg_sources(
      schedule, &block_arg_count, &source_count));
  if (block_arg_count == 0 || source_count == 0) {
    return iree_ok_status();
  }

  uint32_t* block_arg_offsets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, schedule->block_count, sizeof(*block_arg_offsets),
      (void**)&block_arg_offsets));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_compute_block_arg_offsets(
      schedule, block_arg_offsets));

  loom_value_ordinal_t* block_arg_ordinals = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, block_arg_count, sizeof(*block_arg_ordinals),
      (void**)&block_arg_ordinals));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_initialize_block_arg_ordinals(
      schedule, block_arg_offsets, block_arg_count, block_arg_ordinals));

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, schedule->value_count,
      sizeof(*builder->first_block_arg_source_by_value),
      (void**)&builder->first_block_arg_source_by_value));
  for (iree_host_size_t i = 0; i < schedule->value_count; ++i) {
    builder->first_block_arg_source_by_value[i] = LOOM_LOW_SCHEDULE_NODE_NONE;
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, source_count, sizeof(*builder->block_arg_sources),
      (void**)&builder->block_arg_sources));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(builder->arena, schedule->value_count,
                                sizeof(*builder->dependency_visit_epochs),
                                (void**)&builder->dependency_visit_epochs));
  memset(builder->dependency_visit_epochs, 0,
         schedule->value_count * sizeof(*builder->dependency_visit_epochs));

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
      if (dest_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
          dest_ordinal >= schedule->value_count) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU wait branch destination argument is not scheduled");
      }
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

typedef enum loom_amdgpu_wait_dependency_link_mode_e {
  LOOM_AMDGPU_WAIT_DEPENDENCY_LINK_COUNT = 0,
  LOOM_AMDGPU_WAIT_DEPENDENCY_LINK_APPEND = 1,
} loom_amdgpu_wait_dependency_link_mode_t;

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

static iree_status_t loom_amdgpu_wait_plan_visit_dependency_links(
    loom_amdgpu_wait_plan_builder_t* builder, const uint32_t* producer_nodes,
    iree_host_size_t value_count, loom_value_ordinal_t operand_ordinal,
    uint32_t consumer_node, loom_amdgpu_wait_dependency_link_mode_t mode,
    iree_host_size_t* inout_count, uint32_t visit_epoch) {
  if (operand_ordinal >= value_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU wait dependency operand ordinal exceeds liveness domain");
  }
  if (visit_epoch != 0) {
    if (builder->dependency_visit_epochs[operand_ordinal] == visit_epoch) {
      return iree_ok_status();
    }
    builder->dependency_visit_epochs[operand_ordinal] = visit_epoch;
  }
  if (builder->first_block_arg_source_by_value != NULL) {
    uint32_t source_index =
        builder->first_block_arg_source_by_value[operand_ordinal];
    if (source_index != LOOM_LOW_SCHEDULE_NODE_NONE) {
      while (source_index != LOOM_LOW_SCHEDULE_NODE_NONE) {
        const loom_amdgpu_wait_block_arg_source_t* source =
            &builder->block_arg_sources[source_index];
        IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_visit_dependency_links(
            builder, producer_nodes, value_count, source->source_ordinal,
            consumer_node, mode, inout_count, visit_epoch));
        source_index = source->next_source;
      }
      return iree_ok_status();
    }
  }
  const uint32_t producer_node = producer_nodes[operand_ordinal];
  if (producer_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
      producer_node == consumer_node) {
    return iree_ok_status();
  }
  if (loom_amdgpu_wait_plan_node_forwards_dependencies(builder,
                                                       producer_node)) {
    const loom_low_schedule_node_t* producer =
        &builder->schedule->nodes[producer_node];
    const loom_value_ordinal_t* producer_operands =
        loom_low_schedule_node_const_operand_ordinals(producer);
    for (uint16_t i = 0; i < producer->operand_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_visit_dependency_links(
          builder, producer_nodes, value_count, producer_operands[i],
          consumer_node, mode, inout_count, visit_epoch));
    }
    return iree_ok_status();
  }
  const uint32_t counter_mask =
      builder->node_states[producer_node].read_counter_mask;
  if (counter_mask == 0) {
    return iree_ok_status();
  }
  if (mode == LOOM_AMDGPU_WAIT_DEPENDENCY_LINK_COUNT) {
    if (*inout_count == IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU wait dependency count overflows");
    }
    ++*inout_count;
    return iree_ok_status();
  }
  loom_amdgpu_wait_plan_append_dependency_link(
      builder, producer_node, consumer_node, counter_mask,
      LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE);
  return iree_ok_status();
}

static uint32_t loom_amdgpu_wait_plan_memory_effect_counter_mask(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t producer_node,
    uint32_t consumer_node) {
  const loom_amdgpu_wait_node_state_t* producer_state =
      &builder->node_states[producer_node];
  const loom_amdgpu_wait_node_state_t* consumer_state =
      &builder->node_states[consumer_node];
  const loom_low_schedule_node_t* consumer =
      &builder->schedule->nodes[consumer_node];
  if (consumer->op != NULL && loom_low_return_isa(consumer->op)) {
    return producer_state->write_counter_mask &
           LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE;
  }
  uint32_t counter_mask = 0;
  if (iree_any_bit_set(consumer_state->flags,
                       LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_READ)) {
    counter_mask |= producer_state->write_counter_mask;
  }
  if (iree_any_bit_set(consumer_state->flags,
                       LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_WRITE)) {
    counter_mask |= producer_state->read_counter_mask;
  }
  return counter_mask;
}

static loom_amdgpu_wait_plan_reason_t
loom_amdgpu_wait_plan_memory_effect_reason(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t consumer_node) {
  const loom_low_schedule_node_t* consumer =
      &builder->schedule->nodes[consumer_node];
  return consumer->op != NULL && loom_low_return_isa(consumer->op)
             ? LOOM_AMDGPU_WAIT_PLAN_REASON_PROGRAM_EXIT
             : LOOM_AMDGPU_WAIT_PLAN_REASON_MEMORY_EFFECT;
}

static iree_status_t loom_amdgpu_wait_plan_visit_effect_dependency_link(
    loom_amdgpu_wait_plan_builder_t* builder,
    const loom_low_schedule_dependency_t* dependency,
    loom_amdgpu_wait_dependency_link_mode_t mode,
    iree_host_size_t* inout_count) {
  if (dependency->kind != LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT) {
    return iree_ok_status();
  }
  if (dependency->producer_node >= builder->schedule->node_count ||
      dependency->consumer_node >= builder->schedule->node_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU wait effect dependency references node outside schedule");
  }
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
  if (mode == LOOM_AMDGPU_WAIT_DEPENDENCY_LINK_COUNT) {
    if (*inout_count == IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU wait effect dependency count overflows");
    }
    ++*inout_count;
    return iree_ok_status();
  }
  loom_amdgpu_wait_plan_append_dependency_link(
      builder, dependency->producer_node, dependency->consumer_node,
      counter_mask,
      loom_amdgpu_wait_plan_memory_effect_reason(builder,
                                                 dependency->consumer_node));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_visit_visibility_dependency_link(
    loom_amdgpu_wait_plan_builder_t* builder,
    const loom_low_schedule_visibility_dependency_t* dependency,
    loom_amdgpu_wait_dependency_link_mode_t mode,
    iree_host_size_t* inout_count) {
  if (dependency->kind != LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT) {
    return iree_ok_status();
  }
  if (dependency->producer_node >= builder->schedule->node_count ||
      dependency->consumer_node >= builder->schedule->node_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU wait visibility dependency references node outside schedule");
  }
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
  if (mode == LOOM_AMDGPU_WAIT_DEPENDENCY_LINK_COUNT) {
    if (*inout_count == IREE_HOST_SIZE_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU wait visibility dependency count overflows");
    }
    ++*inout_count;
    return iree_ok_status();
  }
  loom_amdgpu_wait_plan_append_dependency_link(
      builder, dependency->producer_node, dependency->consumer_node,
      counter_mask,
      loom_amdgpu_wait_plan_memory_effect_reason(builder,
                                                 dependency->consumer_node));
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
            builder->counter_epochs[loom_amdgpu_wait_counter_slot(
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
      loom_amdgpu_wait_counter_slot(LOOM_AMDGPU_WAIT_COUNTER_ALU);
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
      builder->counter_epochs[loom_amdgpu_wait_counter_slot(
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
      loom_amdgpu_wait_counter_slot(LOOM_AMDGPU_WAIT_COUNTER_ALU);
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

static iree_status_t loom_amdgpu_wait_plan_classify_hazards(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  for (iree_host_size_t i = 0; i < schedule->hazard_use_count; ++i) {
    const loom_low_schedule_hazard_use_t* hazard = &schedule->hazard_uses[i];
    if (hazard->kind != LOOM_LOW_HAZARD_KIND_WAIT_COUNTER) {
      continue;
    }
    if (hazard->reference_kind != LOOM_LOW_HAZARD_REFERENCE_KIND_COUNTER) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU wait-counter hazard on node %" PRIu32
                              " does not reference a counter",
                              hazard->node_index);
    }
    if (hazard->node_index >= schedule->node_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU wait-counter hazard references node "
                              "%" PRIu32 " but schedule has %zu node(s)",
                              hazard->node_index, schedule->node_count);
    }
    uint32_t counter_mask = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_wait_counter_mask(hazard->reference_id, &counter_mask));
    builder->node_states[hazard->node_index].hazard_counter_mask |=
        counter_mask;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_classify_effects(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  for (iree_host_size_t i = 0; i < schedule->effect_use_count; ++i) {
    const loom_low_schedule_effect_use_t* effect = &schedule->effect_uses[i];
    if (effect->node_index >= schedule->node_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU wait effect references node %" PRIu32
                              " but schedule has %zu node(s)",
                              effect->node_index, schedule->node_count);
    }
    loom_amdgpu_wait_node_state_t* node_state =
        &builder->node_states[effect->node_index];
    switch (effect->kind) {
      case LOOM_LOW_EFFECT_KIND_READ: {
        if (!loom_amdgpu_wait_effect_is_dependency_memory(effect)) {
          break;
        }
        node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_READ;
        uint32_t counter_mask = 0;
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_wait_effect_counter_mask(effect, &counter_mask));
        if (counter_mask == 0) {
          node_state->flags |=
              LOOM_AMDGPU_WAIT_NODE_STATE_DEFAULT_DEPENDENCY_READ;
        } else {
          node_state->read_counter_mask |= counter_mask;
        }
        break;
      }
      case LOOM_LOW_EFFECT_KIND_WRITE: {
        if (!loom_amdgpu_wait_effect_is_dependency_memory(effect)) {
          break;
        }
        node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_DEPENDENCY_WRITE;
        uint32_t counter_mask = 0;
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_wait_effect_counter_mask(effect, &counter_mask));
        if (counter_mask == 0) {
          node_state->flags |=
              LOOM_AMDGPU_WAIT_NODE_STATE_DEFAULT_DEPENDENCY_WRITE;
          if (effect->memory_space == LOOM_LOW_MEMORY_SPACE_WORKGROUP) {
            node_state->flags |=
                LOOM_AMDGPU_WAIT_NODE_STATE_DEFAULT_WORKGROUP_WRITE;
          }
        } else {
          node_state->write_counter_mask |= counter_mask;
          if (effect->memory_space == LOOM_LOW_MEMORY_SPACE_WORKGROUP) {
            node_state->workgroup_write_counter_mask |= counter_mask;
          }
        }
        break;
      }
      case LOOM_LOW_EFFECT_KIND_BARRIER:
        if (loom_amdgpu_wait_effect_is_dependency_memory(effect)) {
          uint32_t counter_mask = 0;
          IREE_RETURN_IF_ERROR(
              loom_amdgpu_wait_effect_counter_mask(effect, &counter_mask));
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
        uint32_t counter_mask = 0;
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_wait_counter_mask(effect->counter_id, &counter_mask));
        node_state->explicit_wait_counter_mask |= counter_mask;
        break;
      }
      default:
        break;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_finish_node_classification(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  const bool has_valu_trans_use_depctr = loom_amdgpu_processor_has_scheduling(
      builder->processor,
      LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR);
  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    loom_amdgpu_wait_node_state_t* node_state = &builder->node_states[i];
    const loom_amdgpu_wait_node_state_flags_t flags = node_state->flags;
    const bool has_generic_counter_effect = iree_any_bit_set(
        flags, LOOM_AMDGPU_WAIT_NODE_STATE_GENERIC_COUNTER_EFFECT);
    if (has_generic_counter_effect) {
      node_state->explicit_wait_counter_mask |= node_state->hazard_counter_mask;
    }
    if (node_state->explicit_wait_counter_mask != 0 ||
        has_generic_counter_effect) {
      const loom_low_schedule_node_t* node = &schedule->nodes[i];
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_explicit_counter_mask(
          descriptor_set, node->descriptor, schedule->module, node->op,
          &node_state->explicit_wait_counter_mask));
    }
    if (node_state->explicit_wait_counter_mask != 0 &&
        node_state->hazard_counter_mask == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU counter-effect node %zu has no "
                              "wait-counter hazard rows",
                              i);
    }
    if (iree_any_bit_set(flags,
                         LOOM_AMDGPU_WAIT_NODE_STATE_DEFAULT_DEPENDENCY_READ)) {
      const uint32_t default_read_counter_mask =
          node_state->hazard_counter_mask & LOOM_AMDGPU_WAIT_COUNTER_MASK_READ;
      if (default_read_counter_mask == 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU dependency memory read node %zu has no read counter hazard",
            i);
      }
      node_state->read_counter_mask |= default_read_counter_mask;
    }
    if (iree_any_bit_set(node_state->read_counter_mask,
                         ~node_state->hazard_counter_mask)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU dependency memory read node %zu references a counter without "
          "a wait-counter hazard row",
          i);
    }
    if (iree_any_bit_set(
            flags, LOOM_AMDGPU_WAIT_NODE_STATE_DEFAULT_DEPENDENCY_WRITE)) {
      const uint32_t default_write_counter_mask =
          node_state->hazard_counter_mask & LOOM_AMDGPU_WAIT_COUNTER_MASK_WRITE;
      if (default_write_counter_mask == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AMDGPU dependency memory write node %zu has "
                                "no write counter hazard",
                                i);
      }
      node_state->write_counter_mask |= default_write_counter_mask;
      if (iree_any_bit_set(
              flags, LOOM_AMDGPU_WAIT_NODE_STATE_DEFAULT_WORKGROUP_WRITE)) {
        node_state->workgroup_write_counter_mask |= default_write_counter_mask;
      }
    }
    if (iree_any_bit_set(node_state->write_counter_mask,
                         ~node_state->hazard_counter_mask)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU dependency memory write node %zu "
                              "references a counter without "
                              "a wait-counter hazard row",
                              i);
    }
    const loom_low_schedule_node_t* node = &schedule->nodes[i];
    bool forwards_dependencies = false;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_wait_plan_compute_node_forwards_dependencies(
            builder, (uint32_t)i, &forwards_dependencies));
    if (forwards_dependencies) {
      node_state->flags |= LOOM_AMDGPU_WAIT_NODE_STATE_FORWARDS_DEPENDENCIES;
    }
    if (has_valu_trans_use_depctr && loom_amdgpu_descriptor_is_transcendental(
                                         descriptor_set, node->descriptor)) {
      node_state->trans_result_counter_mask |=
          LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_build_dependency_links(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  const iree_host_size_t value_count = schedule->value_count;
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_build_block_arg_sources(builder));

  uint32_t* producer_nodes = NULL;
  if (value_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(builder->arena, value_count,
                                                   sizeof(*producer_nodes),
                                                   (void**)&producer_nodes));
    for (iree_host_size_t i = 0; i < value_count; ++i) {
      producer_nodes[i] = LOOM_LOW_SCHEDULE_NODE_NONE;
    }
  }

  for (uint32_t node_index = 0; node_index < schedule->node_count;
       ++node_index) {
    const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
    const loom_value_ordinal_t* result_ordinals =
        loom_low_schedule_node_const_result_ordinals(node);
    for (uint16_t i = 0; i < node->result_count; ++i) {
      const loom_value_ordinal_t result_ordinal = result_ordinals[i];
      if (result_ordinal >= value_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "AMDGPU wait dependency result ordinal exceeds schedule value "
            "domain");
      }
      producer_nodes[result_ordinal] = node_index;
    }
  }

  for (uint32_t consumer_node = 0; consumer_node < schedule->node_count;
       ++consumer_node) {
    if (loom_amdgpu_wait_plan_node_forwards_dependencies(builder,
                                                         consumer_node)) {
      continue;
    }
    const loom_low_schedule_node_t* node = &schedule->nodes[consumer_node];
    if (!loom_amdgpu_wait_plan_node_has_wait_consuming_operands(node)) {
      continue;
    }
    const loom_value_ordinal_t* operand_ordinals =
        loom_low_schedule_node_const_operand_ordinals(node);
    for (uint16_t i = 0; i < node->operand_count; ++i) {
      const uint32_t visit_epoch =
          loom_amdgpu_wait_plan_begin_dependency_visit(builder);
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_visit_dependency_links(
          builder, producer_nodes, value_count, operand_ordinals[i],
          consumer_node, LOOM_AMDGPU_WAIT_DEPENDENCY_LINK_COUNT,
          &builder->dependency_link_capacity, visit_epoch));
    }
  }
  for (iree_host_size_t i = 0; i < schedule->dependency_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_visit_effect_dependency_link(
        builder, &schedule->dependencies[i],
        LOOM_AMDGPU_WAIT_DEPENDENCY_LINK_COUNT,
        &builder->dependency_link_capacity));
  }
  for (iree_host_size_t i = 0; i < schedule->visibility_dependency_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_visit_visibility_dependency_link(
        builder, &schedule->visibility_dependencies[i],
        LOOM_AMDGPU_WAIT_DEPENDENCY_LINK_COUNT,
        &builder->dependency_link_capacity));
  }
  if (builder->dependency_link_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, builder->dependency_link_capacity,
        sizeof(*builder->dependency_links),
        (void**)&builder->dependency_links));
  }

  for (uint32_t consumer_node = 0; consumer_node < schedule->node_count;
       ++consumer_node) {
    if (loom_amdgpu_wait_plan_node_forwards_dependencies(builder,
                                                         consumer_node)) {
      continue;
    }
    const loom_low_schedule_node_t* node = &schedule->nodes[consumer_node];
    if (!loom_amdgpu_wait_plan_node_has_wait_consuming_operands(node)) {
      continue;
    }
    const loom_value_ordinal_t* operand_ordinals =
        loom_low_schedule_node_const_operand_ordinals(node);
    for (uint16_t i = 0; i < node->operand_count; ++i) {
      const uint32_t visit_epoch =
          loom_amdgpu_wait_plan_begin_dependency_visit(builder);
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_visit_dependency_links(
          builder, producer_nodes, value_count, operand_ordinals[i],
          consumer_node, LOOM_AMDGPU_WAIT_DEPENDENCY_LINK_APPEND, NULL,
          visit_epoch));
    }
  }
  for (iree_host_size_t i = 0; i < schedule->dependency_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_visit_effect_dependency_link(
        builder, &schedule->dependencies[i],
        LOOM_AMDGPU_WAIT_DEPENDENCY_LINK_APPEND, NULL));
  }
  for (iree_host_size_t i = 0; i < schedule->visibility_dependency_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_visit_visibility_dependency_link(
        builder, &schedule->visibility_dependencies[i],
        LOOM_AMDGPU_WAIT_DEPENDENCY_LINK_APPEND, NULL));
  }
  IREE_ASSERT_EQ(builder->dependency_link_count,
                 builder->dependency_link_capacity);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_classify_nodes(
    loom_amdgpu_wait_plan_builder_t* builder) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_classify_hazards(builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_classify_effects(builder));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_finish_node_classification(builder));
  return loom_amdgpu_wait_plan_build_dependency_links(builder);
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
    const uint32_t producer_counter_mask =
        prior_state->read_counter_mask | prior_state->write_counter_mask |
        prior_state->trans_result_counter_mask;
    if ((producer_counter_mask & counter_mask) == 0 ||
        prior_state->produced_counter_epoch[slot] !=
            builder->counter_epochs[slot]) {
      continue;
    }
    const uint32_t produced_position =
        prior_state->produced_counter_position[slot];
    if (produced_position != 0 &&
        produced_position <= completed_position_count) {
      prior_state->drained_after_production_counter_mask |= counter_mask;
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

static iree_status_t loom_amdgpu_wait_plan_wait_counter(
    loom_amdgpu_wait_plan_builder_t* builder,
    loom_amdgpu_wait_plan_action_kind_t kind,
    loom_amdgpu_wait_plan_reason_t reason, uint32_t node_index,
    uint32_t producer_node, uint16_t counter_id, uint16_t target_count) {
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  if (counter_id == LOOM_AMDGPU_WAIT_COUNTER_NONE ||
      counter_id > LOOM_AMDGPU_WAIT_COUNTER_ALU) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown AMDGPU wait counter id %" PRIu16,
                            counter_id);
  }
  const uint32_t slot = loom_amdgpu_wait_counter_slot(counter_id);
  const uint32_t counter_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
  const uint32_t outstanding_before = builder->outstanding_counts[slot];
  if (target_count > outstanding_before) {
    target_count = (uint16_t)outstanding_before;
  }
  const uint32_t drained_position_count =
      outstanding_before > target_count ? outstanding_before - target_count : 0;
  const uint32_t completed_position_count = iree_math_saturating_add_u32(
      builder->completed_position_counts[slot], drained_position_count);
  loom_amdgpu_wait_plan_mark_drained_producers(builder, node_index, slot,
                                               completed_position_count);
  loom_amdgpu_wait_plan_append_action(
      builder,
      (loom_amdgpu_wait_plan_action_t){
          .kind = kind,
          .reason = reason,
          .counter_id = counter_id,
          .target_count = target_count,
          .block_index = node->block_index,
          .node_index = node_index,
          .scheduled_ordinal = node->scheduled_ordinal,
          .producer_node = producer_node,
          .consumer_node = loom_amdgpu_wait_plan_reason_has_consumer(reason)
                               ? node_index
                               : LOOM_LOW_SCHEDULE_NODE_NONE,
          .outstanding_before = outstanding_before,
      });
  if (target_count == 0 &&
      loom_amdgpu_wait_plan_node_follows_producer_in_same_block(
          builder->schedule, node_index, producer_node)) {
    builder->node_states[producer_node].drained_after_production_counter_mask |=
        counter_mask;
  } else if (target_count == 0) {
    loom_amdgpu_wait_plan_record_current_block_drained_producer(
        builder, producer_node, counter_mask);
  }
  if (target_count == 0) {
    builder->current_block_full_drain_counter_mask |= counter_mask;
    ++builder->counter_epochs[slot];
    builder->completed_position_counts[slot] = 0;
  } else {
    builder->completed_position_counts[slot] = completed_position_count;
  }
  builder->outstanding_counts[slot] =
      iree_min(builder->outstanding_counts[slot], (uint32_t)target_count);
  builder->outstanding_write_counts[slot] =
      iree_min(builder->outstanding_write_counts[slot], (uint32_t)target_count);
  builder->outstanding_workgroup_write_counts[slot] =
      iree_min(builder->outstanding_workgroup_write_counts[slot],
               (uint32_t)target_count);
  if (counter_id == LOOM_AMDGPU_WAIT_COUNTER_ALU && target_count == 0) {
    builder->active_trans_result_vgpr_count = 0;
    loom_amdgpu_wait_plan_clear_sgpr_read_hazards(builder);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_drain_counter(
    loom_amdgpu_wait_plan_builder_t* builder,
    loom_amdgpu_wait_plan_action_kind_t kind,
    loom_amdgpu_wait_plan_reason_t reason, uint32_t node_index,
    uint32_t producer_node, uint16_t counter_id) {
  return loom_amdgpu_wait_plan_wait_counter(builder, kind, reason, node_index,
                                            producer_node, counter_id,
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
    const loom_amdgpu_wait_node_state_t* producer_state,
    uint32_t counter_mask) {
  return iree_any_bit_set(producer_state->drained_after_production_counter_mask,
                          counter_mask);
}

static bool loom_amdgpu_wait_plan_producer_target_count(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t producer_node,
    uint32_t slot, uint16_t* out_target_count) {
  *out_target_count = 0;
  const uint32_t counter_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
  const loom_amdgpu_wait_node_state_t* producer_state =
      &builder->node_states[producer_node];
  if (loom_amdgpu_wait_plan_producer_is_drained(producer_state, counter_mask)) {
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

static iree_status_t loom_amdgpu_wait_plan_storage_release_reason(
    const loom_low_storage_release_action_t* action,
    loom_amdgpu_wait_plan_reason_t* out_reason) {
  loom_amdgpu_wait_plan_reason_t reason =
      (loom_amdgpu_wait_plan_reason_t)action->release_reason_id;
  if (!loom_amdgpu_wait_plan_reason_is_storage_release(reason)) {
    *out_reason = LOOM_AMDGPU_WAIT_PLAN_REASON_UNKNOWN;
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU storage release action has unsupported reason id %" PRIu16,
        action->release_reason_id);
  }
  *out_reason = reason;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_storage_release_is_satisfied(
    const loom_amdgpu_wait_plan_builder_t* builder,
    const loom_low_storage_release_action_t* action,
    const loom_low_storage_lease_record_t* lease_record, bool* out_satisfied) {
  *out_satisfied = false;
  uint32_t counter_mask = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_counter_mask(action->release_class_id, &counter_mask));
  const uint32_t slot = loom_amdgpu_wait_counter_slot(action->release_class_id);
  if (lease_record->node_index >= builder->schedule->node_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU storage lease producer node %" PRIu32
                            " is out of range",
                            lease_record->node_index);
  }
  const loom_amdgpu_wait_node_state_t* producer_state =
      &builder->node_states[lease_record->node_index];
  const uint32_t producer_block =
      builder->schedule->nodes[lease_record->node_index].block_index;
  const uint32_t insertion_block = action->block_index;
  if (loom_amdgpu_wait_plan_producer_is_drained(producer_state, counter_mask)) {
    *out_satisfied = true;
    return iree_ok_status();
  }
  if (producer_block == insertion_block) {
    *out_satisfied = producer_state->produced_counter_epoch[slot] !=
                     builder->counter_epochs[slot];
    return iree_ok_status();
  }
  *out_satisfied =
      iree_any_bit_set(producer_state->drained_after_production_counter_mask,
                       counter_mask) ||
      loom_amdgpu_wait_plan_current_block_satisfies_producer(
          builder, lease_record->node_index, counter_mask);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_handle_storage_release_action(
    loom_amdgpu_wait_plan_builder_t* builder,
    const loom_low_storage_release_action_t* action) {
  if (action->release_action_id !=
      LOOM_AMDGPU_WAIT_PLAN_RESIDUAL_ACTION_WAIT_PACKET) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU storage release action has unsupported action id %" PRIu16,
        action->release_action_id);
  }
  if (action->required_progress != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU storage release action requires unsupported progress %" PRIu32,
        action->required_progress);
  }
  if (action->lease_record_index >=
      builder->allocation->storage_leases.record_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU storage release action references lease record %" PRIu32
        " but allocation has %" PRIhsz " storage lease(s)",
        action->lease_record_index,
        builder->allocation->storage_leases.record_count);
  }

  const loom_low_storage_lease_record_t* lease_record =
      &builder->allocation->storage_leases.records[action->lease_record_index];
  loom_amdgpu_wait_plan_reason_t reason = LOOM_AMDGPU_WAIT_PLAN_REASON_UNKNOWN;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_storage_release_reason(action, &reason));
  bool satisfied = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_storage_release_is_satisfied(
      builder, action, lease_record, &satisfied));
  if (satisfied) {
    return iree_ok_status();
  }
  uint16_t target_count = 0;
  const uint32_t producer_block =
      builder->schedule->nodes[lease_record->node_index].block_index;
  if (producer_block == action->block_index) {
    const uint32_t slot =
        loom_amdgpu_wait_counter_slot(action->release_class_id);
    if (!loom_amdgpu_wait_plan_producer_target_count(
            builder, lease_record->node_index, slot, &target_count)) {
      target_count = 0;
    }
  }
  return loom_amdgpu_wait_plan_wait_counter(
      builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED, reason,
      action->insertion_node_index, lease_record->node_index,
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
    const loom_amdgpu_wait_dependency_link_t* link =
        &builder->dependency_links[link_index];
    const loom_amdgpu_wait_node_state_t* producer_state =
        &builder->node_states[link->producer_node];
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
      } else {
        // Across block boundaries, the producer is safe only if a wait in its
        // own block drained it before control could reach the consumer block.
        if (loom_amdgpu_wait_plan_producer_is_drained(producer_state,
                                                      counter_mask) ||
            loom_amdgpu_wait_plan_current_block_satisfies_producer(
                builder, link->producer_node, counter_mask)) {
          continue;
        }
        target_count = 0;
      }
      active_counter_mask |= counter_mask;
      if (active_producers[slot] == LOOM_LOW_SCHEDULE_NODE_NONE ||
          target_count < target_counts[slot]) {
        active_producers[slot] = link->producer_node;
        active_reasons[slot] = link->reason;
        target_counts[slot] = target_count;
      }
    }
    link_index = link->next_link;
  }

  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    const uint32_t counter_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
    if ((active_counter_mask & counter_mask) == 0) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_wait_counter(
        builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED, active_reasons[slot],
        node_index, active_producers[slot],
        loom_amdgpu_wait_counter_id_from_slot(slot), target_counts[slot]));
  }
  return iree_ok_status();
}

static bool loom_amdgpu_wait_plan_node_is_trans_result_consumer(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (!loom_amdgpu_wait_plan_has_trans_result_state(builder)) {
    return false;
  }
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  return loom_amdgpu_descriptor_uses_vector_alu(
      builder->schedule->target.descriptor_set, node->descriptor);
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
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  return loom_amdgpu_descriptor_uses_scalar_alu(
      builder->schedule->target.descriptor_set, node->descriptor);
}

static bool loom_amdgpu_wait_plan_node_uses_vector_alu(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  return loom_amdgpu_descriptor_uses_vector_alu(
      builder->schedule->target.descriptor_set, node->descriptor);
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
  uint32_t outstanding_counter_mask =
      loom_amdgpu_wait_plan_outstanding_counter_mask(
          builder->outstanding_counts, node_state->barrier_counter_mask);
  outstanding_counter_mask |= loom_amdgpu_wait_plan_outstanding_counter_mask(
      builder->outstanding_workgroup_write_counts,
      node_state->workgroup_barrier_counter_mask);
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
  if (node->op == NULL || !loom_low_return_isa(node->op)) {
    return iree_ok_status();
  }
  const uint32_t outstanding_counter_mask =
      loom_amdgpu_wait_plan_outstanding_counter_mask(
          builder->outstanding_counts,
          LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE);
  if (outstanding_counter_mask == 0) {
    return iree_ok_status();
  }
  return loom_amdgpu_wait_plan_drain_mask(
      builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED,
      LOOM_AMDGPU_WAIT_PLAN_REASON_PROGRAM_EXIT, node_index,
      LOOM_LOW_SCHEDULE_NODE_NONE, outstanding_counter_mask);
}

static iree_status_t loom_amdgpu_wait_plan_increment_outstanding_counts(
    uint32_t outstanding_counts[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT],
    uint32_t counter_mask) {
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    const uint32_t slot_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
    if ((counter_mask & slot_mask) == 0) {
      continue;
    }
    if (outstanding_counts[slot] == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "AMDGPU wait outstanding count overflows");
    }
    ++outstanding_counts[slot];
  }
  return iree_ok_status();
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
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  const loom_low_descriptor_set_t* descriptor_set =
      builder->schedule->target.descriptor_set;
  const bool is_vector_alu =
      loom_amdgpu_descriptor_uses_vector_alu(descriptor_set, node->descriptor);
  const bool is_transcendental = loom_amdgpu_descriptor_is_transcendental(
      descriptor_set, node->descriptor);
  loom_amdgpu_wait_plan_increment_trans_result_intervals(builder, is_vector_alu,
                                                         is_transcendental);
}

static bool loom_amdgpu_wait_plan_node_expires_trans_results(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (!loom_amdgpu_wait_plan_has_trans_result_state(builder)) {
    return false;
  }
  const loom_amdgpu_wait_node_state_t* node_state =
      &builder->node_states[node_index];
  const uint32_t expiring_counter_mask =
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM | LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS;
  return iree_any_bit_set(
      node_state->read_counter_mask | node_state->write_counter_mask,
      expiring_counter_mask);
}

static iree_status_t loom_amdgpu_wait_plan_note_producer(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  loom_amdgpu_wait_node_state_t* node_state = &builder->node_states[node_index];
  const uint32_t counter_mask = node_state->read_counter_mask |
                                node_state->write_counter_mask |
                                node_state->trans_result_counter_mask;
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_increment_outstanding_counts(
      builder->outstanding_counts, counter_mask));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_increment_outstanding_counts(
      builder->outstanding_write_counts, node_state->write_counter_mask));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_increment_outstanding_counts(
      builder->outstanding_workgroup_write_counts,
      node_state->workgroup_write_counter_mask));
  loom_amdgpu_wait_plan_clear_trans_results(builder, node_index);
  loom_amdgpu_wait_plan_record_trans_results(builder, node_index);
  loom_amdgpu_wait_plan_record_sgpr_read_writes(builder, node_index);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_process_node(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  loom_amdgpu_wait_node_state_t* node_state = &builder->node_states[node_index];

  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_handle_storage_release_actions(
      builder, node_index));
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
  if (loom_amdgpu_wait_plan_node_expires_trans_results(builder, node_index)) {
    loom_amdgpu_wait_plan_expire_trans_results(builder);
  }
  loom_amdgpu_wait_plan_track_sgpr_reads(builder, node_index);
  loom_amdgpu_wait_plan_apply_trans_result_interval(builder, node_index);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_note_producer(builder, node_index));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_build_actions(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  for (iree_host_size_t block_index = 0; block_index < schedule->block_count;
       ++block_index) {
    const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
    ++builder->block_epoch;
    builder->current_block_full_drain_counter_mask = 0;
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
    for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
      const uint32_t packet_index = block->scheduled_node_start + i;
      if (packet_index >= schedule->scheduled_node_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "AMDGPU wait plan block references scheduled node outside stream");
      }
      const uint32_t node_index =
          schedule->scheduled_node_indices[packet_index];
      if (node_index >= schedule->node_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "AMDGPU wait plan stream references node outside schedule");
      }
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_wait_plan_process_node(builder, node_index));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_build_hazard_action_index(
    loom_amdgpu_wait_plan_builder_t* builder) {
  if (builder->action_count == 0) {
    return iree_ok_status();
  }
  if (builder->action_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU wait-plan action count exceeds uint32_t");
  }
  const loom_low_schedule_table_t* schedule = builder->schedule;
  if (schedule->node_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(builder->arena, schedule->node_count,
                                sizeof(*builder->first_hazard_action_by_node),
                                (void**)&builder->first_hazard_action_by_node));
  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    builder->first_hazard_action_by_node[i] =
        LOOM_AMDGPU_WAIT_PLAN_ACTION_INDEX_NONE;
  }

  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(builder->arena, builder->action_count,
                                sizeof(*builder->next_hazard_action),
                                (void**)&builder->next_hazard_action));
  for (iree_host_size_t i = 0; i < builder->action_count; ++i) {
    builder->next_hazard_action[i] = LOOM_AMDGPU_WAIT_PLAN_ACTION_INDEX_NONE;
  }

  for (iree_host_size_t i = builder->action_count; i > 0; --i) {
    const uint32_t action_index = (uint32_t)(i - 1);
    const loom_amdgpu_wait_plan_action_t* action =
        &builder->actions[action_index];
    if (action->kind != LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED ||
        loom_amdgpu_wait_plan_reason_is_storage_release(action->reason)) {
      continue;
    }
    if (action->node_index >= schedule->node_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU wait-plan action references node %" PRIu32
                              " but schedule has %" PRIhsz " node(s)",
                              action->node_index, schedule->node_count);
    }
    const loom_low_schedule_node_t* node = &schedule->nodes[action->node_index];
    if (action->block_index != node->block_index ||
        action->scheduled_ordinal != node->scheduled_ordinal) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU wait-plan action does not match its insertion node");
    }
    builder->next_hazard_action[action_index] =
        builder->first_hazard_action_by_node[action->node_index];
    builder->first_hazard_action_by_node[action->node_index] = action_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_emit_counter_progress(
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
  return emit(emit_user_data, &event);
}

static iree_status_t loom_amdgpu_wait_plan_emit_counter_progress_mask(
    loom_low_packet_progress_emit_fn_t emit, void* emit_user_data,
    uint32_t counter_mask, loom_low_packet_progress_action_t action,
    uint32_t units) {
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    if ((counter_mask & loom_amdgpu_wait_counter_mask_from_slot(slot)) == 0) {
      continue;
    }
    const uint16_t counter_id = loom_amdgpu_wait_counter_id_from_slot(slot);
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_emit_counter_progress(
        emit, emit_user_data, counter_id, action, units));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_progress_query(
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_emit_counter_progress_mask(
      emit, emit_user_data, node_state->explicit_wait_counter_mask,
      LOOM_LOW_PACKET_PROGRESS_ACTION_RESET, 0));
  const uint32_t producer_counter_mask = node_state->read_counter_mask |
                                         node_state->write_counter_mask |
                                         node_state->trans_result_counter_mask;
  return loom_amdgpu_wait_plan_emit_counter_progress_mask(
      emit, emit_user_data, producer_counter_mask,
      LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE, 1);
}

static iree_status_t loom_amdgpu_wait_plan_emit_hazard_action(
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
  return emit(emit_user_data, &event);
}

static iree_status_t loom_amdgpu_wait_plan_hazard_query(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_table_t* progress,
    const loom_low_packet_view_t* packet,
    loom_low_packet_hazard_plan_emit_fn_t emit, void* emit_user_data) {
  (void)schedule;
  (void)allocation;
  (void)progress;
  const loom_amdgpu_wait_plan_builder_t* builder =
      (const loom_amdgpu_wait_plan_builder_t*)user_data;
  if (builder->first_hazard_action_by_node == NULL) {
    return iree_ok_status();
  }
  for (uint32_t action_index =
           builder->first_hazard_action_by_node[packet->node_index];
       action_index != LOOM_AMDGPU_WAIT_PLAN_ACTION_INDEX_NONE;
       action_index = builder->next_hazard_action[action_index]) {
    const loom_amdgpu_wait_plan_action_t* action =
        &builder->actions[action_index];
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_wait_plan_emit_hazard_action(action, emit, emit_user_data));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_build_common_tables(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_packet_progress_table_t* progress = NULL;
  if (builder->allocation != NULL) {
    const loom_low_packet_progress_provider_t progress_provider = {
        .user_data = builder,
        .query = loom_amdgpu_wait_plan_progress_query,
    };
    IREE_RETURN_IF_ERROR(loom_low_packet_progress_build(
        builder->schedule, builder->allocation, &progress_provider,
        builder->arena, &builder->progress));
    progress = &builder->progress;
  }

  const loom_low_packet_hazard_plan_provider_t hazard_provider = {
      .user_data = builder,
      .query = loom_amdgpu_wait_plan_hazard_query,
  };
  return loom_low_packet_hazard_plan_build(
      builder->schedule, builder->allocation, progress, &hazard_provider,
      builder->arena, &builder->hazard_plan);
}

iree_status_t loom_amdgpu_wait_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_arena_allocator_t* arena, loom_amdgpu_wait_plan_t* out_plan) {
  if (out_plan == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU wait-plan output is required");
  }
  *out_plan = (loom_amdgpu_wait_plan_t){0};
  if (schedule == NULL || arena == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "schedule and arena are required for AMDGPU "
                            "wait-counter planning");
  }
  if (allocation != NULL) {
    IREE_RETURN_IF_ERROR(loom_low_packet_validate_tables(schedule, allocation));
  }
  loom_amdgpu_wait_plan_builder_t builder = {
      .schedule = schedule,
      .allocation = allocation,
      .arena = arena,
      .processor = loom_amdgpu_target_processor_from_resolved_target(
          schedule->module, &schedule->target),
  };
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
    status = loom_amdgpu_wait_plan_allocate_actions(&builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_plan_allocate_physical_state(&builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_plan_build_actions(&builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_plan_build_hazard_action_index(&builder);
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
  if (plan == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU wait-plan JSON requires a plan");
  }
  return loom_low_packet_hazard_plan_format_json(&plan->hazard_plan, builder);
}
