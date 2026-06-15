// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_plan.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/packet_hazard_plan_json.h"
#include "loom/ir/ir.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/planning/descriptor_semantics.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_id/target_id.h"

#define LOOM_AMDGPU_WAIT_COUNTER_COUNT 5
#define LOOM_AMDGPU_WAIT_TRANS_RESULT_MAX_VALU_INTERVAL 5u
#define LOOM_AMDGPU_WAIT_TRANS_RESULT_MAX_TRANS_INTERVAL 1u

typedef struct loom_amdgpu_wait_node_state_t {
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
  uint32_t produced_counter_epoch[LOOM_AMDGPU_WAIT_COUNTER_COUNT];
  // Read counters whose result was drained before control left the producing
  // block.
  uint32_t drained_after_production_counter_mask;
  // Counters that must be drained before this barrier node executes.
  uint32_t barrier_counter_mask;
  // Workgroup-memory write counters drained before this barrier executes.
  uint32_t workgroup_barrier_counter_mask;
  // Whether the node has a counter effect without a concrete counter id.
  bool has_generic_counter_effect;
  // Whether a memory read effect uses the target's default read counter.
  bool has_default_dependency_read;
  // Whether a memory write effect uses the target's default write counter.
  bool has_default_dependency_write;
  // Whether a workgroup write effect uses the target's default write counter.
  bool has_default_workgroup_write;
} loom_amdgpu_wait_node_state_t;

typedef struct loom_amdgpu_wait_dependency_link_t {
  // Producer node for the SSA dependency.
  uint32_t producer_node;
  // Next dependency link for the same consumer node.
  uint32_t next_link;
  // Counters produced by |producer_node| and needed by this use.
  uint32_t counter_mask;
} loom_amdgpu_wait_dependency_link_t;

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
  // First relevant SSA dependency link per consumer node.
  uint32_t* first_dependency_link_by_consumer;
  // Relevant SSA dependency links.
  loom_amdgpu_wait_dependency_link_t* dependency_links;
  // Number of populated dependency links.
  iree_host_size_t dependency_link_count;
  // Allocated dependency link capacity.
  iree_host_size_t dependency_link_capacity;
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
  uint32_t counter_epochs[LOOM_AMDGPU_WAIT_COUNTER_COUNT];
  // Current block epoch for lazy invalidation of physical-register state.
  uint64_t block_epoch;
  // Outstanding packet count per wait counter.
  uint32_t outstanding_counts[LOOM_AMDGPU_WAIT_COUNTER_COUNT];
  // Outstanding packet count per wait counter for memory writes.
  uint32_t outstanding_write_counts[LOOM_AMDGPU_WAIT_COUNTER_COUNT];
  // Outstanding packet count per wait counter for workgroup memory writes.
  uint32_t outstanding_workgroup_write_counts[LOOM_AMDGPU_WAIT_COUNTER_COUNT];
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
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_amdgpu_wait_plan_allocate(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  const loom_low_allocation_table_t* allocation = builder->allocation;
  if (schedule->node_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, schedule->node_count, sizeof(*builder->node_states),
        (void**)&builder->node_states));
    memset(builder->node_states, 0,
           schedule->node_count * sizeof(*builder->node_states));

    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, schedule->node_count,
        sizeof(*builder->first_dependency_link_by_consumer),
        (void**)&builder->first_dependency_link_by_consumer));
    for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
      builder->first_dependency_link_by_consumer[i] =
          LOOM_LOW_SCHEDULE_NODE_NONE;
    }
  }

  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    builder->dependency_link_capacity += schedule->nodes[i].operand_count;
  }
  if (builder->dependency_link_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, builder->dependency_link_capacity,
        sizeof(*builder->dependency_links),
        (void**)&builder->dependency_links));
  }

  iree_host_size_t action_input_capacity = 0;
  if (!iree_host_size_checked_add(builder->dependency_link_capacity,
                                  schedule->effect_use_count,
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
                                  LOOM_AMDGPU_WAIT_COUNTER_COUNT,
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

static bool loom_amdgpu_wait_plan_processor_has_valu_trans_use_depctr(
    const loom_amdgpu_processor_info_t* processor) {
  return processor != NULL &&
         iree_any_bit_set(
             processor->features.scheduling,
             LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR);
}

static bool loom_amdgpu_wait_plan_processor_has_valu_sgpr_read_depctr(
    const loom_amdgpu_processor_info_t* processor) {
  return processor != NULL &&
         iree_any_bit_set(
             processor->features.scheduling,
             LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_DEPCTR);
}

static bool loom_amdgpu_wait_plan_needs_trans_result_state(
    const loom_amdgpu_wait_plan_builder_t* builder) {
  if (!loom_amdgpu_wait_plan_processor_has_valu_trans_use_depctr(
          builder->processor)) {
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
  return loom_amdgpu_wait_plan_processor_has_valu_sgpr_read_depctr(
      builder->processor);
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

static iree_status_t loom_amdgpu_wait_plan_append_action(
    loom_amdgpu_wait_plan_builder_t* builder,
    loom_amdgpu_wait_plan_action_t action) {
  if (builder->action_count >= builder->action_capacity) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU wait plan exceeded precomputed action capacity");
  }
  builder->actions[builder->action_count++] = action;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_append_dependency_link(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t producer_node,
    uint32_t consumer_node, uint32_t counter_mask) {
  if (builder->dependency_link_count >= builder->dependency_link_capacity) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU wait plan exceeded precomputed dependency capacity");
  }
  loom_amdgpu_wait_dependency_link_t* link =
      &builder->dependency_links[builder->dependency_link_count];
  *link = (loom_amdgpu_wait_dependency_link_t){
      .producer_node = producer_node,
      .next_link = builder->first_dependency_link_by_consumer[consumer_node],
      .counter_mask = counter_mask,
  };
  builder->first_dependency_link_by_consumer[consumer_node] =
      (uint32_t)builder->dependency_link_count++;
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

static const loom_low_allocation_assignment_t* loom_amdgpu_wait_plan_assignment(
    const loom_low_allocation_table_t* allocation, loom_value_id_t value_id) {
  if (allocation == NULL) {
    return NULL;
  }
  return loom_low_allocation_try_map_active_value_assignment(allocation,
                                                             value_id, NULL);
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
             LOOM_AMDGPU_WAIT_TRANS_RESULT_MAX_VALU_INTERVAL &&
         vgpr->trans_interval <=
             LOOM_AMDGPU_WAIT_TRANS_RESULT_MAX_TRANS_INTERVAL;
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
          vgpr->valu_interval, LOOM_AMDGPU_WAIT_TRANS_RESULT_MAX_VALU_INTERVAL);
    }
    if (is_transcendental) {
      vgpr->trans_interval = loom_amdgpu_wait_plan_saturated_increment(
          vgpr->trans_interval,
          LOOM_AMDGPU_WAIT_TRANS_RESULT_MAX_TRANS_INTERVAL);
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
        uint32_t counter_mask = 0;
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_wait_effect_counter_mask(effect, &counter_mask));
        if (counter_mask == 0) {
          node_state->has_default_dependency_read = true;
        } else {
          node_state->read_counter_mask |= counter_mask;
        }
        break;
      }
      case LOOM_LOW_EFFECT_KIND_WRITE: {
        if (!loom_amdgpu_wait_effect_is_dependency_memory(effect)) {
          break;
        }
        uint32_t counter_mask = 0;
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_wait_effect_counter_mask(effect, &counter_mask));
        if (counter_mask == 0) {
          node_state->has_default_dependency_write = true;
          if (effect->memory_space == LOOM_LOW_MEMORY_SPACE_WORKGROUP) {
            node_state->has_default_workgroup_write = true;
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
          node_state->has_generic_counter_effect = true;
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
  const bool has_valu_trans_use_depctr =
      loom_amdgpu_wait_plan_processor_has_valu_trans_use_depctr(
          builder->processor);
  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    loom_amdgpu_wait_node_state_t* node_state = &builder->node_states[i];
    if (node_state->has_generic_counter_effect) {
      node_state->explicit_wait_counter_mask |= node_state->hazard_counter_mask;
    }
    if (node_state->has_generic_counter_effect &&
        node_state->explicit_wait_counter_mask == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU generic counter-effect node %zu has no "
                              "wait-counter hazard rows",
                              i);
    }
    if (node_state->explicit_wait_counter_mask != 0 &&
        node_state->hazard_counter_mask == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU counter-effect node %zu has no "
                              "wait-counter hazard rows",
                              i);
    }
    if (node_state->has_default_dependency_read) {
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
    if (node_state->has_default_dependency_write) {
      const uint32_t default_write_counter_mask =
          node_state->hazard_counter_mask & LOOM_AMDGPU_WAIT_COUNTER_MASK_WRITE;
      if (default_write_counter_mask == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AMDGPU dependency memory write node %zu has "
                                "no write counter hazard",
                                i);
      }
      node_state->write_counter_mask |= default_write_counter_mask;
      if (node_state->has_default_workgroup_write) {
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
    if (has_valu_trans_use_depctr &&
        loom_amdgpu_descriptor_is_transcendental(descriptor_set,
                                                 node->descriptor) &&
        iree_any_bit_set(node_state->hazard_counter_mask,
                         LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU)) {
      node_state->trans_result_counter_mask |=
          LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_build_dependency_links(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  const iree_host_size_t value_count = schedule->liveness.value_count;
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
            "AMDGPU wait dependency result ordinal exceeds liveness domain");
      }
      producer_nodes[result_ordinal] = node_index;
    }
  }

  for (uint32_t consumer_node = 0; consumer_node < schedule->node_count;
       ++consumer_node) {
    const loom_low_schedule_node_t* node = &schedule->nodes[consumer_node];
    const loom_value_ordinal_t* operand_ordinals =
        loom_low_schedule_node_const_operand_ordinals(node);
    for (uint16_t i = 0; i < node->operand_count; ++i) {
      const loom_value_ordinal_t operand_ordinal = operand_ordinals[i];
      if (operand_ordinal >= value_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "AMDGPU wait dependency operand ordinal exceeds liveness domain");
      }
      const uint32_t producer_node = producer_nodes[operand_ordinal];
      if (producer_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
          producer_node == consumer_node) {
        continue;
      }
      const uint32_t counter_mask =
          builder->node_states[producer_node].read_counter_mask;
      if (counter_mask == 0) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_append_dependency_link(
          builder, producer_node, consumer_node, counter_mask));
    }
  }
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

static iree_status_t loom_amdgpu_wait_plan_drain_counter(
    loom_amdgpu_wait_plan_builder_t* builder,
    loom_amdgpu_wait_plan_action_kind_t kind,
    loom_amdgpu_wait_plan_reason_t reason, uint32_t node_index,
    uint32_t producer_node, uint16_t counter_id) {
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  if (counter_id == LOOM_AMDGPU_WAIT_COUNTER_NONE ||
      counter_id > LOOM_AMDGPU_WAIT_COUNTER_ALU) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown AMDGPU wait counter id %" PRIu16,
                            counter_id);
  }
  const uint32_t slot = loom_amdgpu_wait_counter_slot(counter_id);
  const uint32_t outstanding_before = builder->outstanding_counts[slot];
  const uint32_t counter_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
  const uint32_t block_index = node->block_index;
  const loom_low_schedule_block_t* block =
      &builder->schedule->blocks[block_index];
  for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
    const uint32_t packet_index = block->scheduled_node_start + i;
    if (packet_index == node->scheduled_ordinal) {
      break;
    }
    const uint32_t prior_node_index =
        builder->schedule->scheduled_node_indices[packet_index];
    loom_amdgpu_wait_node_state_t* prior_state =
        &builder->node_states[prior_node_index];
    if ((prior_state->read_counter_mask & counter_mask) != 0 &&
        prior_state->produced_counter_epoch[slot] ==
            builder->counter_epochs[slot]) {
      prior_state->drained_after_production_counter_mask |= counter_mask;
    }
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_append_action(
      builder,
      (loom_amdgpu_wait_plan_action_t){
          .kind = kind,
          .reason = reason,
          .counter_id = counter_id,
          .target_count = 0,
          .block_index = node->block_index,
          .node_index = node_index,
          .scheduled_ordinal = node->scheduled_ordinal,
          .producer_node = producer_node,
          .consumer_node = loom_amdgpu_wait_plan_reason_has_consumer(reason)
                               ? node_index
                               : LOOM_LOW_SCHEDULE_NODE_NONE,
          .outstanding_before = outstanding_before,
      }));
  ++builder->counter_epochs[slot];
  builder->outstanding_counts[slot] = 0;
  builder->outstanding_write_counts[slot] = 0;
  builder->outstanding_workgroup_write_counts[slot] = 0;
  if (counter_id == LOOM_AMDGPU_WAIT_COUNTER_ALU) {
    builder->active_trans_result_vgpr_count = 0;
    loom_amdgpu_wait_plan_clear_sgpr_read_hazards(builder);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_drain_mask(
    loom_amdgpu_wait_plan_builder_t* builder,
    loom_amdgpu_wait_plan_action_kind_t kind,
    loom_amdgpu_wait_plan_reason_t reason, uint32_t node_index,
    uint32_t producer_node, uint32_t counter_mask) {
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_COUNT; ++slot) {
    if ((counter_mask & loom_amdgpu_wait_counter_mask_from_slot(slot)) == 0) {
      continue;
    }
    const uint16_t counter_id = loom_amdgpu_wait_counter_id_from_slot(slot);
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_drain_counter(
        builder, kind, reason, node_index, producer_node, counter_id));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_storage_release_reason(
    const loom_low_storage_release_action_t* action,
    loom_amdgpu_wait_plan_reason_t* out_reason) {
  switch (action->release_reason_id) {
    case LOOM_AMDGPU_WAIT_PLAN_REASON_STORE_SOURCE_REUSE:
    case LOOM_AMDGPU_WAIT_PLAN_REASON_READ_RESULT_REUSE:
      *out_reason = (loom_amdgpu_wait_plan_reason_t)action->release_reason_id;
      return iree_ok_status();
    default:
      *out_reason = LOOM_AMDGPU_WAIT_PLAN_REASON_UNKNOWN;
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU storage release action has unsupported reason id %" PRIu16,
          action->release_reason_id);
  }
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
  if (producer_block == insertion_block) {
    *out_satisfied = producer_state->produced_counter_epoch[slot] !=
                     builder->counter_epochs[slot];
    return iree_ok_status();
  }
  *out_satisfied = (producer_state->drained_after_production_counter_mask &
                    counter_mask) != 0;
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
  return loom_amdgpu_wait_plan_drain_counter(
      builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED, reason,
      action->insertion_node_index, lease_record->node_index,
      action->release_class_id);
}

static iree_status_t loom_amdgpu_wait_plan_handle_storage_release_actions(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  const loom_low_allocation_table_t* allocation = builder->allocation;
  if (allocation == NULL || allocation->storage_release_action_count == 0) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  for (iree_host_size_t i = 0; i < allocation->storage_release_action_count;
       ++i) {
    const loom_low_storage_release_action_t* action =
        &allocation->storage_release_actions[i];
    if (action->insertion_node_index != node_index) {
      continue;
    }
    if (action->block_index != node->block_index ||
        action->scheduled_ordinal != node->scheduled_ordinal) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU storage release action does not match its insertion node");
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_wait_plan_handle_storage_release_action(builder, action));
  }
  return iree_ok_status();
}

static uint32_t loom_amdgpu_wait_plan_outstanding_counter_mask(
    const uint32_t outstanding_counts[LOOM_AMDGPU_WAIT_COUNTER_COUNT],
    uint32_t counter_mask) {
  uint32_t outstanding_counter_mask = 0;
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_COUNT; ++slot) {
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
  uint32_t active_producers[LOOM_AMDGPU_WAIT_COUNTER_COUNT] = {
      LOOM_LOW_SCHEDULE_NODE_NONE, LOOM_LOW_SCHEDULE_NODE_NONE,
      LOOM_LOW_SCHEDULE_NODE_NONE, LOOM_LOW_SCHEDULE_NODE_NONE,
      LOOM_LOW_SCHEDULE_NODE_NONE,
  };
  for (uint32_t link_index =
           builder->first_dependency_link_by_consumer[node_index];
       link_index != LOOM_LOW_SCHEDULE_NODE_NONE;) {
    const loom_amdgpu_wait_dependency_link_t* link =
        &builder->dependency_links[link_index];
    const loom_amdgpu_wait_node_state_t* producer_state =
        &builder->node_states[link->producer_node];
    for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_COUNT; ++slot) {
      const uint32_t counter_mask =
          loom_amdgpu_wait_counter_mask_from_slot(slot);
      if ((link->counter_mask & counter_mask) == 0) {
        continue;
      }
      const uint32_t producer_block =
          builder->schedule->nodes[link->producer_node].block_index;
      const uint32_t consumer_block =
          builder->schedule->nodes[node_index].block_index;
      if (producer_block == consumer_block) {
        // Epochs are block-local because outstanding counts reset at block
        // entry. Within one block, a newer epoch means an earlier wait already
        // drained the producer.
        if (producer_state->produced_counter_epoch[slot] !=
            builder->counter_epochs[slot]) {
          continue;
        }
      } else {
        // Across block boundaries, the producer is safe only if a wait in its
        // own block drained it before control could reach the consumer block.
        if ((producer_state->drained_after_production_counter_mask &
             counter_mask) != 0) {
          continue;
        }
      }
      active_counter_mask |= counter_mask;
      if (active_producers[slot] == LOOM_LOW_SCHEDULE_NODE_NONE) {
        active_producers[slot] = link->producer_node;
      }
    }
    link_index = link->next_link;
  }

  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_COUNT; ++slot) {
    const uint32_t counter_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
    if ((active_counter_mask & counter_mask) == 0) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_drain_counter(
        builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED,
        LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE, node_index,
        active_producers[slot], loom_amdgpu_wait_counter_id_from_slot(slot)));
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

static iree_status_t loom_amdgpu_wait_plan_increment_outstanding_counts(
    uint32_t outstanding_counts[LOOM_AMDGPU_WAIT_COUNTER_COUNT],
    uint32_t counter_mask) {
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_COUNT; ++slot) {
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
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_COUNT; ++slot) {
    if ((counter_mask & loom_amdgpu_wait_counter_mask_from_slot(slot)) == 0) {
      continue;
    }
    node_state->produced_counter_epoch[slot] = builder->counter_epochs[slot];
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
    memset(builder->counter_epochs, 0, sizeof(builder->counter_epochs));
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
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_COUNT; ++slot) {
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

static bool loom_amdgpu_wait_plan_action_matches_packet(
    const loom_amdgpu_wait_plan_action_t* action,
    const loom_low_packet_view_t* packet) {
  return action->node_index == packet->node_index &&
         action->block_index == packet->node->block_index &&
         action->scheduled_ordinal == packet->node->scheduled_ordinal;
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
      .target_detail = iree_string_view_empty(),
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
  for (iree_host_size_t i = 0; i < builder->action_count; ++i) {
    const loom_amdgpu_wait_plan_action_t* action = &builder->actions[i];
    if (action->kind != LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED ||
        !loom_amdgpu_wait_plan_action_matches_packet(action, packet)) {
      continue;
    }
    if (action->reason == LOOM_AMDGPU_WAIT_PLAN_REASON_STORE_SOURCE_REUSE ||
        action->reason == LOOM_AMDGPU_WAIT_PLAN_REASON_READ_RESULT_REUSE) {
      continue;
    }
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
    status = loom_amdgpu_wait_plan_allocate_physical_state(&builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_plan_build_actions(&builder);
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
