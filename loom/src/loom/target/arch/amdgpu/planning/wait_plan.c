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
#include "loom/codegen/low/memory_access.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/packet_hazard_plan_json.h"
#include "loom/ir/ir.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/planning/sgpr_read_hazard.h"
#include "loom/target/arch/amdgpu/planning/trans_result_window.h"
#include "loom/target/arch/amdgpu/planning/wait_actions.h"
#include "loom/target/arch/amdgpu/planning/wait_classification.h"
#include "loom/target/arch/amdgpu/planning/wait_completion.h"
#include "loom/target/arch/amdgpu/planning/wait_dependency_visit.h"
#include "loom/target/arch/amdgpu/planning/wait_frontier.h"
#include "loom/target/arch/amdgpu/planning/wait_loop.h"
#include "loom/target/arch/amdgpu/planning/wait_packet_tables.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

// Gfx125x TDM issues require an intervening tensorcnt bound at most this value.
// This issue hazard is independent of particular memory dependency edges.
#define LOOM_AMDGPU_TENSOR_ISSUE_MAXIMUM_PENDING 10u

// Mutable execution state retained only for nodes that produce counter work.
typedef struct loom_amdgpu_wait_producer_state_t {
  // Epoch for each counter produced by this node.
  uint32_t epochs[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
  // Monotonic producer positions within the corresponding epochs.
  uint32_t positions[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
  // Block epoch owning |drained_counter_mask|.
  uint64_t drained_epoch;
  // Counters already drained for this producer in |drained_epoch|.
  loom_amdgpu_wait_counter_mask_t drained_counter_mask;
  // Register parts preserved by this producer, or zero for a complete result.
  uint8_t preserved_part_mask;
  // The source ordinal bypasses every same-part continuation in its chain.
  bool preserved_source_resolved;
  // Value supplying the preserved part, canonicalized before dependency use.
  loom_value_ordinal_t preserved_source_ordinal;
} loom_amdgpu_wait_producer_state_t;

static_assert(sizeof(loom_amdgpu_wait_producer_state_t) == 80,
              "preserved readiness must fit existing producer-state padding");

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
  // Completed physical assignment table for post-allocation hazards.
  const loom_low_allocation_table_t* allocation;
  // Arena that owns tables retained by the completed plan.
  iree_arena_allocator_t* arena;
  // Arena that owns builder state discarded after plan construction.
  iree_arena_allocator_t* transient_arena;
  // Processor properties selected by the low target, or NULL if unavailable.
  const loom_amdgpu_processor_properties_t* processor_properties;
  // Common VMEM completion class in the current epoch, or UNKNOWN for mixed
  // classes and accesses without a single known completion domain.
  loom_amdgpu_vmem_result_order_class_t vmem_epoch_order_class;
  // Local counter epochs containing flat requests that can retire early.
  // Each bit remains set until that counter is fully drained.
  uint32_t unordered_flat_counter_mask;
  // Generated wait-packet descriptors selected by the low target.
  loom_amdgpu_wait_packet_target_t wait_packet_target;
  // Target wait state classified from the completed schedule.
  loom_amdgpu_wait_classification_t classification;
  // Mutable counter state packed over classified producer nodes only.
  loom_amdgpu_wait_producer_state_t* producer_states;
  // Borrowed schedule-owned producer node indexed by SSA value ordinal.
  const uint32_t* producer_nodes;
  // Bounded cross-block wait state.
  loom_amdgpu_wait_frontier_t frontier;
  // Target eligibility and ancestor index over canonical schedule loops.
  loom_amdgpu_wait_loop_analysis_t loop_analysis;
  // First relevant counter dependency link per consumer node.
  uint32_t* first_dependency_link_by_consumer;
  // Relevant counter dependency links.
  loom_amdgpu_wait_dependency_t* dependency_links;
  // Borrowed first coalesced incoming copy by destination value ordinal.
  const uint32_t* first_coalesced_incoming_copy_by_value_ordinal;
  // Borrowed edge-copy rows linked by the incoming-copy index.
  const loom_low_allocation_edge_copy_t* edge_copies;
  // First SSA dependency indexed by loop-entry block and counter.
  uint32_t* loop_entry_dependency_links;
  // Counter classes fully drained at each planned loop-entry block.
  uint32_t* loop_entry_drain_counter_masks;
  // Derived incoming counter epochs indexed by loop block and counter.
  const loom_amdgpu_wait_loop_cyclic_frontier_t* cyclic_frontiers;
  // Canonical insertion points within native-instruction boundaries.
  struct {
    // Borrowed address-state overlay built before wait planning.
    const loom_amdgpu_address_state_plan_t* address_state;
    // Next address-state transition in scheduled order.
    iree_host_size_t address_state_cursor;
    // First planned wait since the preceding native work or block boundary.
    uint32_t anchor_node;
  } insertion;
  // Reusable exact-range traversal state for forwarded SSA dependencies.
  loom_amdgpu_wait_dependency_visit_t dependency_visit;
  // Number of populated dependency links.
  iree_host_size_t dependency_link_count;
  // Allocated dependency link capacity.
  iree_host_size_t dependency_link_capacity;
  // Sparse action accumulation before exact retained publication.
  loom_amdgpu_wait_actions_t actions;
  // Lazily allocated output bitset of redundant authored full memory waits.
  uint64_t* elided_wait_nodes;
  // Exact number of canonical packet-progress rows.
  iree_host_size_t progress_event_count;
  // Canonical packet-progress table populated after wait actions are known.
  loom_low_packet_progress_table_t progress;
  // Canonical packet hazard table populated after wait actions are known.
  loom_low_packet_hazard_plan_t hazard_plan;
  // Current epoch per wait counter.
  uint32_t counter_epochs[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
  // Oldest producer positions already known complete in the current epoch.
  uint32_t completed_position_counts[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
  // First scheduled ordinal not retired for each counter in the current block.
  // Each cursor advances across epochs and stops at the first pending producer.
  uint32_t retirement_ordinals[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
  // Current block epoch for lazy invalidation of physical-register state.
  uint64_t block_epoch;
  // Counters fully drained earlier in the current straight-line block.
  uint32_t current_block_full_drain_counter_mask;
  // Counters proven empty since their last issue or opaque control boundary.
  uint32_t known_empty_counter_mask;
  // A preceding tensor issue has no intervening qualifying tensorcnt bound.
  bool tensor_issue_requires_drain;
  // Translation group currently represented by outstanding gfx125x XCNT
  // events. Hardware implicitly drains XCNT when this group changes.
  loom_amdgpu_wait_xcnt_group_t xcnt_group;
  // Outstanding packet count per wait counter.
  uint32_t outstanding_counts[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
  // Outstanding packet count per wait counter for memory writes.
  uint32_t outstanding_write_counts[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
  // Outstanding packet count per wait counter for workgroup memory accesses.
  uint32_t
      outstanding_workgroup_access_counts[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT];
  // Active GFX11 transcendental-result windows by physical VGPR.
  loom_amdgpu_trans_result_window_t trans_result_window;
  // Active GFX12 VALU/SALU dependencies by physical SGPR.
  loom_amdgpu_sgpr_read_hazard_t sgpr_read_hazard;
} loom_amdgpu_wait_plan_builder_t;

static loom_amdgpu_wait_producer_state_t* loom_amdgpu_wait_plan_producer_state(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  IREE_ASSERT_LT(node_index, builder->schedule->node_count);
  const uint32_t producer_state_ordinal =
      builder->classification.node_states[node_index]
          .state.producer_state_ordinal;
  IREE_ASSERT_NE(producer_state_ordinal, 0u);
  IREE_ASSERT_LE(producer_state_ordinal,
                 builder->classification.producer_state_count);
  return &builder->producer_states[producer_state_ordinal - 1];
}

static const loom_amdgpu_wait_producer_state_t*
loom_amdgpu_wait_plan_const_producer_state(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  IREE_ASSERT_LT(node_index, builder->schedule->node_count);
  const uint32_t producer_state_ordinal =
      builder->classification.node_states[node_index]
          .state.producer_state_ordinal;
  IREE_ASSERT_NE(producer_state_ordinal, 0u);
  IREE_ASSERT_LE(producer_state_ordinal,
                 builder->classification.producer_state_count);
  return &builder->producer_states[producer_state_ordinal - 1];
}

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
    case LOOM_AMDGPU_WAIT_PLAN_REASON_TENSOR_ISSUE_DRAIN:
      return IREE_SV("amdgpu.tensor_issue_drain");
    case LOOM_AMDGPU_WAIT_PLAN_REASON_SYSTEM_SCOPE_STORE:
      return IREE_SV("amdgpu.system_scope_store");
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
    case LOOM_AMDGPU_WAIT_PLAN_REASON_TENSOR_ISSUE_DRAIN:
    case LOOM_AMDGPU_WAIT_PLAN_REASON_SYSTEM_SCOPE_STORE:
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

static bool loom_amdgpu_wait_plan_node_forwards_dependencies(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  return node_index < builder->schedule->node_count &&
         iree_any_bit_set(builder->classification.node_states[node_index].flags,
                          LOOM_AMDGPU_WAIT_NODE_STATE_FORWARDS_DEPENDENCIES);
}

static bool loom_amdgpu_wait_plan_needs_trans_result_state(
    const loom_amdgpu_wait_plan_builder_t* builder) {
  return builder->classification.trans_result_node_count != 0;
}

static bool loom_amdgpu_wait_plan_needs_sgpr_read_state(
    const loom_amdgpu_wait_plan_builder_t* builder) {
  return loom_amdgpu_processor_properties_have_scheduling(
      builder->processor_properties,
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
  if (needs_trans_result_state) {
    const iree_host_size_t vgpr_count =
        allocation->physical_extents
            .ends_by_reg_class[LOOM_AMDGPU_REG_CLASS_ID_VGPR];
    IREE_RETURN_IF_ERROR(loom_amdgpu_trans_result_window_initialize(
        vgpr_count, LOOM_AMDGPU_TRANS_RESULT_WINDOW_FLAG_TRACK_ORIGINS,
        builder->transient_arena, &builder->trans_result_window));
  }
  if (needs_sgpr_read_state) {
    const uint32_t sgpr_count =
        allocation->physical_extents
            .ends_by_reg_class[LOOM_AMDGPU_REG_CLASS_ID_SGPR];
    const loom_low_descriptor_set_t* descriptor_set =
        allocation->target.descriptor_set;
    const loom_low_reg_class_t* sgpr_class =
        &descriptor_set->reg_classes[LOOM_AMDGPU_REG_CLASS_ID_SGPR];
    const uint32_t fixed_sgpr_end =
        allocation->physical_extents
            .fixed_ends_by_reg_class[LOOM_AMDGPU_REG_CLASS_ID_SGPR];
    const uint32_t fixed_sgpr_count =
        fixed_sgpr_end > sgpr_class->fixed_location_base
            ? fixed_sgpr_end - sgpr_class->fixed_location_base
            : 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sgpr_read_hazard_initialize(
        sgpr_count, sgpr_class->fixed_location_base, fixed_sgpr_count,
        builder->transient_arena, &builder->sgpr_read_hazard));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_append_action(
    loom_amdgpu_wait_plan_builder_t* builder,
    loom_amdgpu_wait_plan_action_t action) {
  if (action.counter_id == LOOM_AMDGPU_WAIT_COUNTER_TENSOR &&
      action.target_count <= LOOM_AMDGPU_TENSOR_ISSUE_MAXIMUM_PENDING) {
    builder->tensor_issue_requires_drain = false;
  }
  // Logical progress and producer/consumer provenance retain their original
  // nodes. Only concrete insertion shares the boundary before a zero-work run.
  if (action.kind == LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED) {
    if (builder->insertion.anchor_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
      builder->insertion.anchor_node = action.node_index;
    }
    action.node_index = builder->insertion.anchor_node;
    action.scheduled_ordinal =
        builder->schedule->nodes[action.node_index].scheduled_ordinal;
  }
  return loom_amdgpu_wait_actions_append(&builder->actions, &action,
                                         builder->transient_arena);
}

static iree_status_t loom_amdgpu_wait_plan_allocate_dependency_heads(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const iree_host_size_t node_count = builder->schedule->node_count;
  if (node_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->transient_arena, node_count,
      sizeof(*builder->first_dependency_link_by_consumer),
      (void**)&builder->first_dependency_link_by_consumer));
  for (iree_host_size_t i = 0; i < node_count; ++i) {
    builder->first_dependency_link_by_consumer[i] = LOOM_LOW_SCHEDULE_NODE_NONE;
  }
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
  loom_amdgpu_wait_dependency_t* link =
      &builder->dependency_links[builder->dependency_link_count];
  *link = (loom_amdgpu_wait_dependency_t){
      .producer_node = producer_node,
      .consumer_node = consumer_node,
      .next_dependency =
          builder->first_dependency_link_by_consumer[consumer_node],
      .counter_mask = counter_mask,
      .reason_id = (uint16_t)reason,
      .flags = reason == LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE
                   ? LOOM_AMDGPU_WAIT_DEPENDENCY_FLAG_SSA_USE
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

static bool loom_amdgpu_wait_plan_op_has_packet_transfers(const loom_op_t* op) {
  return op != NULL && (loom_low_copy_isa(op) || loom_low_move_isa(op) ||
                        loom_low_slice_isa(op) || loom_low_concat_isa(op));
}

static const loom_low_allocation_packet_move_group_t*
loom_amdgpu_wait_plan_packet_transfer_group(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (builder->allocation == NULL ||
      node_index >= builder->schedule->node_count) {
    return NULL;
  }
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  if (!loom_amdgpu_wait_plan_op_has_packet_transfers(node->op)) {
    return NULL;
  }
  return loom_low_allocation_find_packet_move_group_by_source_ordinal(
      builder->allocation, node->source_ordinal);
}

static iree_status_t loom_amdgpu_wait_plan_ensure_dependency_visit_state(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  const bool has_packet_transfers =
      builder->allocation != NULL &&
      builder->allocation->packet_move_group_count != 0;
  if (schedule->value_count == 0 ||
      (builder->first_coalesced_incoming_copy_by_value_ordinal == NULL &&
       builder->classification.forwarding_node_count == 0 &&
       !has_packet_transfers)) {
    return iree_ok_status();
  }
  return loom_amdgpu_wait_dependency_visit_initialize(
      schedule->value_count, builder->transient_arena,
      &builder->dependency_visit);
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
      builder->classification.frontier_nodes[producer_node].read_counter_mask;
  if (counter_mask == 0) {
    return iree_ok_status();
  }
  return loom_amdgpu_wait_plan_append_dependency_link(
      builder, producer_node, consumer_node, counter_mask,
      LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE);
}

static loom_value_ordinal_t loom_amdgpu_wait_plan_readiness_source(
    const loom_amdgpu_wait_plan_builder_t* builder,
    loom_value_ordinal_t value_ordinal,
    loom_low_register_part_mask_t read_mask) {
  if (read_mask == UINT32_MAX) {
    return value_ordinal;
  }
  const uint32_t producer_node = builder->producer_nodes[value_ordinal];
  if (producer_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
      !iree_any_bit_set(
          builder->classification.node_states[producer_node].flags,
          LOOM_AMDGPU_WAIT_NODE_STATE_PRESERVES_RESULT_PART)) {
    return value_ordinal;
  }
  const loom_amdgpu_wait_producer_state_t* producer =
      loom_amdgpu_wait_plan_const_producer_state(builder, producer_node);
  return (read_mask & ~producer->preserved_part_mask) == 0
             ? producer->preserved_source_ordinal
             : value_ordinal;
}

static uint32_t loom_amdgpu_wait_plan_value_unit_count(
    const loom_amdgpu_wait_plan_builder_t* builder,
    loom_value_ordinal_t value_ordinal) {
  IREE_ASSERT_LT(value_ordinal, builder->schedule->value_count);
  if (builder->allocation == NULL) {
    return 1;
  }
  const loom_liveness_interval_t* interval =
      loom_liveness_interval_for_value_ordinal(&builder->allocation->liveness,
                                               value_ordinal);
  return interval != NULL && interval->unit_count != 0 ? interval->unit_count
                                                       : 1;
}

static iree_status_t loom_amdgpu_wait_plan_push_dependency_range(
    loom_amdgpu_wait_plan_builder_t* builder,
    loom_value_ordinal_t value_ordinal, uint32_t unit_offset,
    uint32_t unit_count) {
  return loom_amdgpu_wait_dependency_visit_push(
      &builder->dependency_visit, value_ordinal,
      loom_amdgpu_wait_plan_value_unit_count(builder, value_ordinal),
      unit_offset, unit_count);
}

static iree_status_t loom_amdgpu_wait_plan_push_coalesced_incoming_ranges(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t first_copy_index,
    const loom_amdgpu_wait_dependency_visit_range_t* range) {
  const uint32_t requested_end = range->unit_offset + range->unit_count;
  const iree_host_size_t worklist_begin =
      builder->dependency_visit.worklist_count;
  uint32_t copy_index = first_copy_index;
  while (copy_index != LOOM_LOW_ALLOCATION_EDGE_COPY_INDEX_NONE) {
    IREE_ASSERT_LT(copy_index, builder->allocation->edge_copy_count);
    const loom_low_allocation_edge_copy_t* copy =
        &builder->edge_copies[copy_index];
    const uint32_t destination_end =
        copy->destination_unit_offset + copy->unit_count;
    const uint32_t intersection_begin =
        iree_max(range->unit_offset, copy->destination_unit_offset);
    const uint32_t intersection_end = iree_min(requested_end, destination_end);
    if (intersection_begin < intersection_end) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_push_dependency_range(
          builder, copy->source_ordinal,
          copy->source_unit_offset +
              (intersection_begin - copy->destination_unit_offset),
          intersection_end - intersection_begin));
    }
    copy_index = copy->next_coalesced_incoming_copy_index;
  }
  loom_amdgpu_wait_dependency_visit_reverse(&builder->dependency_visit,
                                            worklist_begin);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_push_forwarded_packet_ranges(
    loom_amdgpu_wait_plan_builder_t* builder,
    const loom_low_allocation_packet_move_group_t* group,
    uint32_t producer_node,
    const loom_amdgpu_wait_dependency_visit_range_t* range) {
  const loom_low_allocation_table_t* allocation = builder->allocation;
  const uint32_t requested_end = range->unit_offset + range->unit_count;
  const iree_host_size_t worklist_begin =
      builder->dependency_visit.worklist_count;
  const bool has_exact_ranges =
      iree_any_bit_set(group->transfer_flags,
                       LOOM_LOW_ALLOCATION_PACKET_TRANSFER_GROUP_FLAG_EXACT);
  const loom_low_schedule_node_t* node =
      &builder->schedule->nodes[producer_node];
  IREE_ASSERT_EQ(node->result_count, 1u);
  const loom_value_ordinal_t result_ordinal =
      loom_low_schedule_node_const_result_ordinals(node)[0];
  const loom_low_placement_relation_range_t relation_range =
      loom_low_placement_relation_range_for_value_ordinal(
          &allocation->placement, result_ordinal);
  const uint32_t candidate_count =
      has_exact_ranges ? group->transfer_count : relation_range.count;
  for (uint32_t i = 0; i < candidate_count; ++i) {
    const loom_low_allocation_packet_transfer_t* transfer =
        has_exact_ranges
            ? &allocation->packet_transfers[group->transfer_start + i]
            : NULL;
    if (transfer != NULL &&
        loom_low_allocation_packet_transfer_is_materialized(transfer)) {
      continue;
    }
    const uint32_t relation_index =
        transfer != NULL
            ? loom_low_allocation_packet_transfer_relation_index(transfer)
            : relation_range.start + i;
    IREE_ASSERT_LT(relation_index, allocation->placement.relation_count);
    const loom_low_placement_relation_t* relation =
        &allocation->placement.relations[relation_index];
    if (relation->op != node->op || relation->cause != group->cause) {
      continue;
    }
    const uint32_t relation_unit_offset =
        transfer != NULL ? transfer->relation_unit_offset : 0;
    const uint32_t transfer_unit_count =
        transfer != NULL ? transfer->unit_count : relation->unit_count;
    const uint32_t result_offset =
        relation->result_unit_offset + relation_unit_offset;
    const uint32_t result_end = result_offset + transfer_unit_count;
    const uint32_t intersection_begin =
        iree_max(range->unit_offset, result_offset);
    const uint32_t intersection_end = iree_min(requested_end, result_end);
    if (intersection_begin >= intersection_end) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_push_dependency_range(
        builder, relation->source_ordinal,
        relation->source_unit_offset + relation_unit_offset +
            (intersection_begin - result_offset),
        intersection_end - intersection_begin));
  }
  loom_amdgpu_wait_dependency_visit_reverse(&builder->dependency_visit,
                                            worklist_begin);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_traverse_dependency_links(
    loom_amdgpu_wait_plan_builder_t* builder, const uint32_t* producer_nodes,
    iree_host_size_t value_count, uint32_t consumer_node,
    loom_low_register_part_mask_t read_mask) {
  loom_amdgpu_wait_dependency_visit_range_t range;
  while (loom_amdgpu_wait_dependency_visit_pop(&builder->dependency_visit,
                                               &range)) {
    const loom_value_ordinal_t readiness_ordinal =
        loom_amdgpu_wait_plan_readiness_source(builder, range.value_ordinal,
                                               read_mask);
    if (readiness_ordinal != range.value_ordinal) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_push_dependency_range(
          builder, readiness_ordinal, range.unit_offset, range.unit_count));
      continue;
    }
    if (builder->first_coalesced_incoming_copy_by_value_ordinal != NULL) {
      const uint32_t copy_index =
          builder->first_coalesced_incoming_copy_by_value_ordinal
              [range.value_ordinal];
      if (copy_index != LOOM_LOW_ALLOCATION_EDGE_COPY_INDEX_NONE) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_wait_plan_push_coalesced_incoming_ranges(
                builder, copy_index, &range));
        continue;
      }
    }

    const uint32_t producer_node = producer_nodes[range.value_ordinal];
    if (producer_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
        producer_node == consumer_node) {
      continue;
    }
    if (!loom_amdgpu_wait_plan_node_forwards_dependencies(builder,
                                                          producer_node)) {
      if (loom_amdgpu_wait_dependency_visit_complete_value(
              &builder->dependency_visit, range.value_ordinal)) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_wait_plan_append_direct_dependency_link(
                builder, producer_nodes, value_count, range.value_ordinal,
                consumer_node));
      }
      continue;
    }

    const loom_low_allocation_packet_move_group_t* packet_group =
        loom_amdgpu_wait_plan_packet_transfer_group(builder, producer_node);
    if (packet_group != NULL) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_push_forwarded_packet_ranges(
          builder, packet_group, producer_node, &range));
      continue;
    }

    const loom_low_schedule_node_t* producer =
        &builder->schedule->nodes[producer_node];
    const loom_value_ordinal_t* producer_operands =
        loom_low_schedule_node_const_operand_ordinals(producer);
    const iree_host_size_t worklist_begin =
        builder->dependency_visit.worklist_count;
    for (uint16_t i = 0; i < producer->operand_count; ++i) {
      const uint32_t unit_count =
          loom_amdgpu_wait_plan_value_unit_count(builder, producer_operands[i]);
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_push_dependency_range(
          builder, producer_operands[i], /*unit_offset=*/0, unit_count));
    }
    loom_amdgpu_wait_dependency_visit_reverse(&builder->dependency_visit,
                                              worklist_begin);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_visit_dependency_links(
    loom_amdgpu_wait_plan_builder_t* builder, const uint32_t* producer_nodes,
    iree_host_size_t value_count, loom_value_ordinal_t operand_ordinal,
    uint32_t operand_unit_offset, uint32_t operand_unit_count,
    uint32_t consumer_node, loom_low_register_part_mask_t read_mask) {
  if (builder->dependency_visit.value_epochs == NULL) {
    return loom_amdgpu_wait_plan_append_direct_dependency_link(
        builder, producer_nodes, value_count,
        loom_amdgpu_wait_plan_readiness_source(builder, operand_ordinal,
                                               read_mask),
        consumer_node);
  }
  loom_amdgpu_wait_dependency_visit_begin(&builder->dependency_visit);
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_push_dependency_range(
      builder, operand_ordinal, operand_unit_offset, operand_unit_count));
  return loom_amdgpu_wait_plan_traverse_dependency_links(
      builder, producer_nodes, value_count, consumer_node, read_mask);
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
    if (edge_copy->kind == LOOM_LOW_ALLOCATION_COPY_COALESCED) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_visit_dependency_links(
        builder, producer_nodes, value_count, edge_copy->source_ordinal,
        edge_copy->source_unit_offset, edge_copy->unit_count, consumer_node,
        UINT32_MAX));
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_wait_plan_build_packet_transfer_dependency_links(
    loom_amdgpu_wait_plan_builder_t* builder, const uint32_t* producer_nodes,
    iree_host_size_t value_count, uint32_t consumer_node,
    const loom_low_allocation_packet_move_group_t* group) {
  if (!iree_any_bit_set(
          group->transfer_flags,
          LOOM_LOW_ALLOCATION_PACKET_TRANSFER_GROUP_FLAG_MATERIALIZED)) {
    return iree_ok_status();
  }
  IREE_ASSERT(builder->dependency_visit.value_epochs != NULL);
  loom_amdgpu_wait_dependency_visit_begin(&builder->dependency_visit);
  const loom_low_allocation_table_t* allocation = builder->allocation;
  const bool has_exact_ranges =
      iree_any_bit_set(group->transfer_flags,
                       LOOM_LOW_ALLOCATION_PACKET_TRANSFER_GROUP_FLAG_EXACT);
  const loom_low_schedule_node_t* node =
      &builder->schedule->nodes[consumer_node];
  IREE_ASSERT_EQ(node->result_count, 1u);
  const loom_value_ordinal_t result_ordinal =
      loom_low_schedule_node_const_result_ordinals(node)[0];
  const loom_low_placement_relation_range_t relation_range =
      loom_low_placement_relation_range_for_value_ordinal(
          &allocation->placement, result_ordinal);
  const uint32_t candidate_count =
      has_exact_ranges ? group->transfer_count : relation_range.count;
  for (uint32_t i = 0; i < candidate_count; ++i) {
    const loom_low_allocation_packet_transfer_t* transfer =
        has_exact_ranges
            ? &allocation->packet_transfers[group->transfer_start + i]
            : NULL;
    if (transfer != NULL &&
        !loom_low_allocation_packet_transfer_is_materialized(transfer)) {
      continue;
    }
    const uint32_t relation_index =
        transfer != NULL
            ? loom_low_allocation_packet_transfer_relation_index(transfer)
            : relation_range.start + i;
    IREE_ASSERT_LT(relation_index, allocation->placement.relation_count);
    const loom_low_placement_relation_t* relation =
        &allocation->placement.relations[relation_index];
    if (relation->op != node->op || relation->cause != group->cause) {
      continue;
    }
    const uint32_t relation_unit_offset =
        transfer != NULL ? transfer->relation_unit_offset : 0;
    const uint32_t unit_count =
        transfer != NULL ? transfer->unit_count : relation->unit_count;
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_push_dependency_range(
        builder, relation->source_ordinal,
        relation->source_unit_offset + relation_unit_offset, unit_count));
  }
  loom_amdgpu_wait_dependency_visit_reverse(&builder->dependency_visit,
                                            /*begin=*/0);
  return loom_amdgpu_wait_plan_traverse_dependency_links(
      builder, producer_nodes, value_count, consumer_node, UINT32_MAX);
}

static bool loom_amdgpu_wait_plan_node_is_volatile_memory(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  return node->descriptor != NULL &&
         iree_any_bit_set(node->op->instance_flags,
                          LOOM_MEMORY_ACCESS_FLAG_VOLATILE);
}

// Vector memory instructions issued by one wave are processed in issue order,
// so a later VMEM access observes every earlier same-wave VMEM access to the
// same address without draining vmcnt first. Memory-effect dependencies whose
// consumer is itself a VMEM-only access therefore need no VMEM counter wait;
// register reuse and SSA uses of loaded values retain their own waits. Other
// counters (LDS, SMEM, tensor, async) and program-exit ordering are unchanged.
// Volatile accesses keep their waits: they must also stay ordered against
// accesses to different addresses, which hardware does not guarantee.
static uint32_t loom_amdgpu_wait_plan_drop_same_wave_vmem_ordering(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t producer_node,
    uint32_t consumer_node, uint32_t counter_mask) {
  if ((counter_mask & LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM) == 0) {
    return counter_mask;
  }
  if (loom_amdgpu_wait_plan_node_is_volatile_memory(builder, producer_node) ||
      loom_amdgpu_wait_plan_node_is_volatile_memory(builder, consumer_node)) {
    return counter_mask;
  }
  const loom_amdgpu_wait_frontier_node_t* consumer_memory =
      &builder->classification.frontier_nodes[consumer_node];
  const uint32_t consumer_counters = consumer_memory->read_counter_mask |
                                     consumer_memory->write_counter_mask;
  if (consumer_counters == 0 ||
      (consumer_counters & ~LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM) != 0) {
    return counter_mask;
  }
  return counter_mask & ~LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM;
}

static uint32_t loom_amdgpu_wait_plan_memory_effect_counter_mask(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t producer_node,
    uint32_t consumer_node) {
  const loom_amdgpu_wait_frontier_node_t* producer_memory =
      &builder->classification.frontier_nodes[producer_node];
  const loom_amdgpu_wait_node_state_t* consumer_state =
      &builder->classification.node_states[consumer_node];
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
  return loom_amdgpu_wait_plan_drop_same_wave_vmem_ordering(
      builder, producer_node, consumer_node, counter_mask);
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

static iree_status_t loom_amdgpu_wait_plan_visit_memory_completion_edge(
    loom_amdgpu_wait_plan_builder_t* builder,
    const loom_low_schedule_memory_completion_edge_t* edge) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  IREE_ASSERT_LT(edge->producer_effect_use, schedule->effect_use_count);
  IREE_ASSERT_LT(edge->consumer_effect_use, schedule->effect_use_count);
  const loom_low_schedule_effect_use_t* producer =
      &schedule->effect_uses[edge->producer_effect_use];
  const loom_low_schedule_effect_use_t* consumer =
      &schedule->effect_uses[edge->consumer_effect_use];
  IREE_ASSERT((producer->kind == LOOM_LOW_EFFECT_KIND_READ &&
               consumer->kind == LOOM_LOW_EFFECT_KIND_WRITE) ||
              (producer->kind == LOOM_LOW_EFFECT_KIND_WRITE &&
               consumer->kind == LOOM_LOW_EFFECT_KIND_READ));
  IREE_ASSERT_NE(producer->block_index, consumer->block_index);
  if (loom_amdgpu_wait_plan_node_forwards_dependencies(builder,
                                                       consumer->node_index)) {
    return iree_ok_status();
  }

  uint32_t counter_mask = 0;
  if (producer->counter_id == LOOM_AMDGPU_WAIT_COUNTER_NONE) {
    const loom_amdgpu_wait_frontier_node_t* producer_node =
        &builder->classification.frontier_nodes[producer->node_index];
    counter_mask = producer->kind == LOOM_LOW_EFFECT_KIND_READ
                       ? producer_node->read_counter_mask
                       : producer_node->write_counter_mask;
  } else {
    counter_mask = loom_amdgpu_wait_counter_mask(producer->counter_id);
  }
  counter_mask = loom_amdgpu_wait_plan_drop_same_wave_vmem_ordering(
      builder, producer->node_index, consumer->node_index, counter_mask);
  if (counter_mask == 0) {
    return iree_ok_status();
  }
  return loom_amdgpu_wait_plan_append_dependency_link(
      builder, producer->node_index, consumer->node_index, counter_mask,
      LOOM_AMDGPU_WAIT_PLAN_REASON_MEMORY_EFFECT);
}

static bool loom_amdgpu_wait_plan_has_trans_result_state(
    const loom_amdgpu_wait_plan_builder_t* builder) {
  return loom_amdgpu_trans_result_window_is_initialized(
      &builder->trans_result_window);
}

static bool loom_amdgpu_wait_plan_has_sgpr_read_state(
    const loom_amdgpu_wait_plan_builder_t* builder) {
  return loom_amdgpu_sgpr_read_hazard_is_initialized(
      &builder->sgpr_read_hazard);
}

static void loom_amdgpu_wait_plan_expire_trans_results(
    loom_amdgpu_wait_plan_builder_t* builder) {
  if (!loom_amdgpu_trans_result_window_has_active(
          &builder->trans_result_window)) {
    return;
  }
  const uint32_t alu_slot =
      loom_amdgpu_wait_counter_slot_from_id(LOOM_AMDGPU_WAIT_COUNTER_ALU);
  ++builder->counter_epochs[alu_slot];
  builder->completed_position_counts[alu_slot] = 0;
  builder->outstanding_counts[alu_slot] = 0;
  loom_amdgpu_trans_result_window_clear(&builder->trans_result_window);
}

static iree_status_t loom_amdgpu_wait_plan_allocate_producer_states(
    loom_amdgpu_wait_plan_builder_t* builder) {
  if (builder->classification.producer_state_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->transient_arena, builder->classification.producer_state_count,
      sizeof(*builder->producer_states), (void**)&builder->producer_states));
  memset(builder->producer_states, 0,
         builder->classification.producer_state_count *
             sizeof(*builder->producer_states));
  return iree_ok_status();
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
      builder->classification.frontier_nodes[lease_record->node_index]
          .vmem_result_order_class;
  const loom_amdgpu_vmem_result_order_class_t consumer_order_class =
      builder->classification.frontier_nodes[action->insertion_node_index]
          .vmem_result_order_class;
  // A flat result may return through LDS while another request writes through
  // VMEM. Matching VMEM classes alone cannot order those physical writes.
  return builder->classification.frontier_nodes[lease_record->node_index]
                 .read_counter_mask ==
             LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD &&
         builder->classification.frontier_nodes[action->insertion_node_index]
                 .read_counter_mask ==
             LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD &&
         producer_order_class != LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE &&
         producer_order_class != LOOM_AMDGPU_VMEM_RESULT_ORDER_UNKNOWN &&
         producer_order_class == consumer_order_class;
}

// D16 memory results preserve a tied register part without reading it. Retain
// that source at the producer; the newest SSA owner alone does not identify
// the asynchronous event that makes a preserved-half read ready.
static void loom_amdgpu_wait_plan_classify_preserved_source(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  if (node->descriptor == NULL || node->result_count == 0 ||
      builder->classification.frontier_nodes[node_index].read_counter_mask ==
          0) {
    return;
  }
  const loom_low_descriptor_set_t* descriptors =
      builder->schedule->target.descriptor_set;
  const loom_low_operand_t* result =
      &descriptors->operands[node->descriptor->operand_start];
  if (!iree_any_bit_set(result->flags,
                        LOOM_LOW_OPERAND_FLAG_STORAGE_CONTINUATION)) {
    return;
  }
  // Generated asynchronous continuation contracts have one one-unit VGPR
  // result and exactly one tied source. Synchronous partial VALU writes do not
  // publish preserved readiness: their physical writes first retire old loads.
  const loom_value_ordinal_t result_ordinal =
      loom_low_schedule_node_const_result_ordinals(node)[0];
  const loom_value_ordinal_t source_ordinal =
      loom_low_placement_tied_source_for_value_ordinal(
          &builder->allocation->placement, result_ordinal);
  const loom_low_reg_class_alt_t* result_alternative =
      &descriptors->reg_class_alts[result->reg_class_alt_start];
  const loom_low_register_part_mask_t full_mask =
      descriptors->reg_classes[result_alternative->reg_class_id]
          .full_register_part_mask;
  loom_amdgpu_wait_producer_state_t* producer =
      loom_amdgpu_wait_plan_producer_state(builder, node_index);
  producer->preserved_part_mask =
      (uint8_t)(full_mask &
                ~descriptors
                     ->register_parts[result_alternative->register_part_id]
                     .mask);
  producer->preserved_source_ordinal = source_ordinal;
  builder->classification.node_states[node_index].flags |=
      LOOM_AMDGPU_WAIT_NODE_STATE_PRESERVES_RESULT_PART;
}

static loom_amdgpu_wait_producer_state_t*
loom_amdgpu_wait_plan_preserving_producer(
    loom_amdgpu_wait_plan_builder_t* builder, loom_value_ordinal_t ordinal) {
  const uint32_t node_index = builder->producer_nodes[ordinal];
  return node_index != LOOM_LOW_SCHEDULE_NODE_NONE &&
                 iree_any_bit_set(
                     builder->classification.node_states[node_index].flags,
                     LOOM_AMDGPU_WAIT_NODE_STATE_PRESERVES_RESULT_PART)
             ? loom_amdgpu_wait_plan_producer_state(builder, node_index)
             : NULL;
}

// Resolve each same-part chain once, regardless of source block layout. Each
// unresolved producer is visited at most twice; uses perform one indexed
// lookup. Structural aliases and block arguments remain boundaries owned by the
// existing coalesced-storage dependency projection, not new IR walks here.
static void loom_amdgpu_wait_plan_resolve_preserved_sources(
    loom_amdgpu_wait_plan_builder_t* builder) {
  for (iree_host_size_t i = 0; i < builder->classification.producer_state_count;
       ++i) {
    loom_amdgpu_wait_producer_state_t* producer = &builder->producer_states[i];
    if (producer->preserved_part_mask == 0 ||
        producer->preserved_source_resolved) {
      continue;
    }
    const uint8_t part_mask = producer->preserved_part_mask;
    loom_value_ordinal_t root = producer->preserved_source_ordinal;
    loom_amdgpu_wait_producer_state_t* parent =
        loom_amdgpu_wait_plan_preserving_producer(builder, root);
    while (parent != NULL && parent->preserved_part_mask == part_mask) {
      root = parent->preserved_source_ordinal;
      if (parent->preserved_source_resolved) {
        break;
      }
      parent = loom_amdgpu_wait_plan_preserving_producer(builder, root);
    }
    while (producer != NULL && producer->preserved_part_mask == part_mask &&
           !producer->preserved_source_resolved) {
      parent = loom_amdgpu_wait_plan_preserving_producer(
          builder, producer->preserved_source_ordinal);
      producer->preserved_source_ordinal = root;
      producer->preserved_source_resolved = true;
      producer = parent;
    }
  }
}

static loom_low_register_part_mask_t
loom_amdgpu_wait_plan_operand_readiness_mask(
    const loom_low_descriptor_set_t* descriptors,
    const loom_low_operand_t* operand,
    const loom_amdgpu_wait_frontier_node_t* consumer) {
  // D16 asynchronous writes may interfere with the other half of a VALU access.
  // Only memory operands have independently readable register parts.
  if ((consumer->read_counter_mask | consumer->write_counter_mask) == 0) {
    return UINT32_MAX;
  }
  const loom_low_reg_class_alt_t* alternative = loom_low_operand_reg_class_alt(
      descriptors, operand, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  return alternative != NULL &&
                 alternative->register_part_id != LOOM_LOW_REGISTER_PART_NONE
             ? descriptors->register_parts[alternative->register_part_id].mask
             : UINT32_MAX;
}

static iree_status_t loom_amdgpu_wait_plan_build_dependency_links(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  const iree_host_size_t value_count = schedule->value_count;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_ensure_dependency_visit_state(builder));

  const uint32_t* producer_nodes = builder->producer_nodes;

  for (uint32_t node_index = 0; node_index < schedule->node_count;
       ++node_index) {
    loom_amdgpu_wait_plan_classify_preserved_source(builder, node_index);
  }
  loom_amdgpu_wait_plan_resolve_preserved_sources(builder);

  for (uint32_t consumer_node = 0; consumer_node < schedule->node_count;
       ++consumer_node) {
    const loom_low_schedule_node_t* node = &schedule->nodes[consumer_node];
    const loom_low_allocation_packet_move_group_t* packet_group =
        loom_amdgpu_wait_plan_packet_transfer_group(builder, consumer_node);
    if (packet_group != NULL) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_wait_plan_build_packet_transfer_dependency_links(
              builder, producer_nodes, value_count, consumer_node,
              packet_group));
      continue;
    }
    if (builder->allocation != NULL &&
        loom_amdgpu_wait_plan_op_has_packet_transfers(node->op)) {
      continue;
    }
    if (loom_amdgpu_wait_plan_node_forwards_dependencies(builder,
                                                         consumer_node)) {
      continue;
    }
    if (node->op != NULL && loom_low_br_isa(node->op)) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_wait_plan_build_edge_copy_dependency_links(
              builder, producer_nodes, value_count, consumer_node));
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
        const loom_value_ordinal_t operand_ordinal =
            operand_ordinals[descriptor_operand->source_value_index];
        IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_visit_dependency_links(
            builder, producer_nodes, value_count, operand_ordinal,
            /*operand_unit_offset=*/0,
            loom_amdgpu_wait_plan_value_unit_count(builder, operand_ordinal),
            consumer_node,
            loom_amdgpu_wait_plan_operand_readiness_mask(
                descriptor_set, descriptor_operand,
                &builder->classification.frontier_nodes[consumer_node])));
      }
    } else {
      for (uint16_t i = 0; i < node->operand_count; ++i) {
        const loom_value_ordinal_t operand_ordinal = operand_ordinals[i];
        IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_visit_dependency_links(
            builder, producer_nodes, value_count, operand_ordinal,
            /*operand_unit_offset=*/0,
            loom_amdgpu_wait_plan_value_unit_count(builder, operand_ordinal),
            consumer_node, UINT32_MAX));
      }
    }
  }
  for (uint32_t i = 0; i < schedule->effect_dependencies.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_visit_effect_dependency_link(
        builder,
        loom_low_schedule_dependency_range_at(
            &schedule->dependencies, schedule->effect_dependencies, i)));
  }
  for (iree_host_size_t i = 0; i < schedule->memory_completion_edge_count;
       ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_visit_memory_completion_edge(
        builder, &schedule->memory_completion_edges[i]));
  }
  // Physical reuse closes the same completion epoch as a payload consumer.
  // Carry the allocator's exact release points into loop analysis before it
  // derives the outstanding requests that can survive a backedge.
  const loom_low_allocation_table_t* allocation = builder->allocation;
  if (allocation != NULL) {
    for (iree_host_size_t i = 0; i < allocation->storage_release_action_count;
         ++i) {
      const loom_low_storage_release_action_t* action =
          &allocation->storage_release_actions[i];
      const loom_low_storage_lease_record_t* record =
          &allocation->storage_leases.records[action->lease_record_index];
      const loom_amdgpu_wait_plan_reason_t reason =
          (loom_amdgpu_wait_plan_reason_t)action->release_reason_id;
      if (loom_amdgpu_wait_plan_storage_release_is_ordered_vmem_reuse(
              builder, action, record, reason)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_append_dependency_link(
          builder, record->node_index, action->insertion_node_index,
          loom_amdgpu_wait_counter_mask(action->release_class_id), reason));
    }
  }
  return iree_ok_status();
}

static iree_host_size_t loom_amdgpu_wait_plan_loop_entry_slot_index(
    iree_host_size_t block_index, uint32_t counter_slot) {
  IREE_ASSERT_LT(counter_slot, LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT);
  return (iree_host_size_t)block_index * LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT +
         counter_slot;
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
  if (builder->loop_analysis.loop_count == 0) {
    return iree_ok_status();
  }

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
      loom_amdgpu_wait_dependency_t* link =
          &builder->dependency_links[link_index];
      const uint32_t next_dependency = link->next_dependency;
      if (!iree_any_bit_set(link->flags,
                            LOOM_AMDGPU_WAIT_DEPENDENCY_FLAG_SSA_USE) ||
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

  if (builder->loop_entry_dependency_links == NULL) {
    return iree_ok_status();
  }

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
      if (link_index == LOOM_LOW_SCHEDULE_NODE_NONE) {
        continue;
      }
      const uint16_t counter_id = loom_amdgpu_wait_counter_id_from_slot(slot);
      bool is_full_drain = counter_id == LOOM_AMDGPU_WAIT_COUNTER_SMEM;
      uint32_t strictest_producer_ordinal = 0;
      while (link_index != LOOM_LOW_SCHEDULE_NODE_NONE) {
        const loom_amdgpu_wait_dependency_t* link =
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
          if ((builder->classification.completion_nodes[node_index]
                   .producer_counter_mask &
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
  uint32_t* retirement_ordinal = &builder->retirement_ordinals[slot];
  for (; *retirement_ordinal < node->scheduled_ordinal; ++*retirement_ordinal) {
    const uint32_t i = *retirement_ordinal;
    const uint32_t packet_index = block->scheduled_node_start + i;
    const uint32_t prior_node_index =
        builder->schedule->scheduled_node_indices[packet_index];
    loom_amdgpu_wait_frontier_node_t* prior_memory =
        &builder->classification.frontier_nodes[prior_node_index];
    if ((builder->classification.completion_nodes[prior_node_index]
             .producer_counter_mask &
         counter_mask) == 0) {
      continue;
    }
    const loom_amdgpu_wait_producer_state_t* prior_state =
        loom_amdgpu_wait_plan_const_producer_state(builder, prior_node_index);
    if (prior_state->epochs[slot] != builder->counter_epochs[slot]) {
      continue;
    }
    const uint32_t produced_position = prior_state->positions[slot];
    if (produced_position > completed_position_count) {
      // Producer positions increase in schedule order within a counter epoch.
      // A later wait resumes here instead of revisiting the completed prefix.
      break;
    }
    if (produced_position != 0) {
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
  if (producer_node >= builder->schedule->node_count) {
    return false;
  }
  const loom_amdgpu_wait_producer_state_t* producer_state =
      loom_amdgpu_wait_plan_const_producer_state(builder, producer_node);
  return producer_state->drained_epoch == builder->block_epoch &&
         iree_any_bit_set(producer_state->drained_counter_mask, counter_mask);
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
  loom_amdgpu_wait_producer_state_t* producer_state =
      loom_amdgpu_wait_plan_producer_state(builder, producer_node);
  if (producer_state->drained_epoch != builder->block_epoch) {
    producer_state->drained_epoch = builder->block_epoch;
    producer_state->drained_counter_mask = 0;
  }
  producer_state->drained_counter_mask |= counter_mask;
}

static uint16_t loom_amdgpu_wait_plan_normalize_target_count(
    const loom_amdgpu_wait_plan_builder_t* builder, uint16_t counter_id,
    uint16_t target_count) {
  const uint32_t slot = loom_amdgpu_wait_counter_slot_from_id(counter_id);
  const uint32_t outstanding_count = builder->outstanding_counts[slot];
  target_count = (uint16_t)iree_min((uint32_t)target_count, outstanding_count);
  if (iree_any_bit_set(builder->unordered_flat_counter_mask,
                       loom_amdgpu_wait_counter_mask_from_slot(slot))) {
    // Early completion in the unused flat domain breaks issue-position bounds.
    // Only a full drain identifies completion of a particular request.
    target_count = 0;
  }
  const bool is_smem_counter =
      counter_id == LOOM_AMDGPU_WAIT_COUNTER_SMEM ||
      (counter_id == LOOM_AMDGPU_WAIT_COUNTER_X &&
       builder->xcnt_group == LOOM_AMDGPU_WAIT_XCNT_GROUP_SMEM);
  if (is_smem_counter && target_count != 0) {
    // Scalar-memory requests may complete out of order. A nonzero logical SMEM
    // or XCNT threshold cannot prove completion of any particular request.
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
    builder->classification.frontier_nodes[producer_node]
        .drained_after_production_counter_mask |= counter_mask;
  } else if (target_count == 0 ||
             (producer_node < builder->schedule->node_count &&
              builder->schedule->nodes[producer_node].block_index !=
                  builder->schedule->nodes[node_index].block_index)) {
    loom_amdgpu_wait_plan_record_current_block_drained_producer(
        builder, producer_node, counter_mask);
  }
  if (target_count == 0) {
    builder->unordered_flat_counter_mask &= ~counter_mask;
    builder->current_block_full_drain_counter_mask |= counter_mask;
    builder->known_empty_counter_mask |= counter_mask;
    loom_amdgpu_wait_frontier_drain(&builder->frontier, counter_mask);
    ++builder->counter_epochs[slot];
    builder->completed_position_counts[slot] = 0;
  } else {
    builder->completed_position_counts[slot] = completed_position_count;
  }
  builder->outstanding_counts[slot] = target_count;
  builder->outstanding_write_counts[slot] =
      iree_min(builder->outstanding_write_counts[slot], (uint32_t)target_count);
  builder->outstanding_workgroup_access_counts[slot] =
      iree_min(builder->outstanding_workgroup_access_counts[slot],
               (uint32_t)target_count);
  if (counter_id == LOOM_AMDGPU_WAIT_COUNTER_ALU && target_count == 0) {
    loom_amdgpu_trans_result_window_clear(&builder->trans_result_window);
    loom_amdgpu_sgpr_read_hazard_clear_dependencies(&builder->sgpr_read_hazard);
  }
  if (counter_id == LOOM_AMDGPU_WAIT_COUNTER_X && target_count == 0) {
    builder->xcnt_group = LOOM_AMDGPU_WAIT_XCNT_GROUP_NONE;
  }
  if (counter_id == LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD && target_count == 0) {
    builder->vmem_epoch_order_class = LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE;
  }
}

static loom_amdgpu_vmem_result_order_class_t
loom_amdgpu_wait_plan_vmem_completion_order(
    const loom_amdgpu_wait_frontier_node_t* node) {
  // Generic-address requests can complete through LDS on targets whose flat
  // memory instructions do not preserve VMEM completion order.
  if (node->vmem_result_order_class == LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE ||
      node->read_counter_mask != LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD ||
      iree_any_bit_set(
          node->read_space_flags,
          loom_amdgpu_wait_memory_space_flag(LOOM_LOW_MEMORY_SPACE_GENERIC))) {
    return LOOM_AMDGPU_VMEM_RESULT_ORDER_UNKNOWN;
  }
  return node->vmem_result_order_class;
}

static void loom_amdgpu_wait_plan_note_vmem_order(
    loom_amdgpu_wait_plan_builder_t* builder,
    const loom_amdgpu_wait_frontier_node_t* node) {
  const loom_amdgpu_vmem_result_order_class_t order_class =
      loom_amdgpu_wait_plan_vmem_completion_order(node);
  if (builder->vmem_epoch_order_class == LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE) {
    builder->vmem_epoch_order_class = order_class;
  } else if (builder->vmem_epoch_order_class != order_class) {
    builder->vmem_epoch_order_class = LOOM_AMDGPU_VMEM_RESULT_ORDER_UNKNOWN;
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
  // A planned completion bound must never encode the reserved no-wait value.
  // Record the effective bound before publishing completion or report counts.
  target_count = iree_min(
      target_count, builder->wait_packet_target.maximum_target_counts[slot]);
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
  if (kind == LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED && target_count == 0) {
    const uint32_t counter_mask = loom_amdgpu_wait_counter_mask(counter_id);
    uint32_t coupled_counter_mask =
        builder->wait_packet_target.selections[counter_mask]
            .full_drain_counter_mask &
        ~counter_mask;
    while (coupled_counter_mask != 0) {
      const uint32_t coupled_slot =
          (uint32_t)iree_math_count_trailing_zeros_u32(coupled_counter_mask);
      loom_amdgpu_wait_plan_apply_counter_progress(
          builder, insertion_node, LOOM_LOW_SCHEDULE_NODE_NONE,
          loom_amdgpu_wait_counter_id_from_slot(coupled_slot),
          /*target_count=*/0);
      coupled_counter_mask &= coupled_counter_mask - 1;
    }
  }
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
  return iree_any_bit_set(builder->classification.frontier_nodes[producer_node]
                              .drained_after_production_counter_mask,
                          counter_mask);
}

static bool loom_amdgpu_wait_plan_producer_target_count(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t producer_node,
    uint32_t slot, uint16_t* out_target_count) {
  *out_target_count = 0;
  const uint32_t counter_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
  const loom_amdgpu_wait_producer_state_t* producer_state =
      loom_amdgpu_wait_plan_const_producer_state(builder, producer_node);
  if (loom_amdgpu_wait_plan_producer_is_drained(builder, producer_node,
                                                counter_mask)) {
    return false;
  }
  if (producer_state->epochs[slot] != builder->counter_epochs[slot]) {
    return false;
  }
  const uint32_t produced_position = producer_state->positions[slot];
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
  const loom_amdgpu_wait_producer_state_t* producer_state =
      loom_amdgpu_wait_plan_const_producer_state(builder, producer_node);
  const uint32_t produced_position = producer_state->positions[slot];
  return producer_state->epochs[slot] == builder->counter_epochs[slot] &&
         produced_position != 0 &&
         produced_position <= builder->completed_position_counts[slot];
}

static uint16_t loom_amdgpu_wait_plan_incoming_producer_target_count(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t producer_node,
    uint16_t consumer_block, uint32_t slot) {
  if (loom_amdgpu_wait_counter_id_from_slot(slot) !=
      LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD) {
    return 0;
  }
  const loom_amdgpu_vmem_result_order_class_t order_class =
      loom_amdgpu_wait_plan_vmem_completion_order(
          &builder->classification.frontier_nodes[producer_node]);
  if (order_class == LOOM_AMDGPU_VMEM_RESULT_ORDER_UNKNOWN ||
      order_class != builder->vmem_epoch_order_class) {
    return 0;
  }
  const iree_host_size_t frontier_index =
      loom_amdgpu_wait_plan_loop_entry_slot_index(consumer_block, slot);
  if (builder->cyclic_frontiers != NULL &&
      builder->cyclic_frontiers[frontier_index].outstanding_count != 0) {
    return 0;
  }
  // Every tracked request was issued after block entry, hence after the
  // incoming producer. Matching completion order lets the older result retire
  // while those younger requests remain pending, regardless of incoming count.
  return (uint16_t)iree_min(builder->outstanding_counts[slot], UINT16_MAX);
}

static iree_status_t loom_amdgpu_wait_plan_handle_loop_entry_dependencies(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (builder->loop_entry_dependency_links == NULL) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  const loom_low_schedule_block_t* block =
      &builder->schedule->blocks[node->block_index];
  if (node->op != block->block->last_op) {
    return iree_ok_status();
  }

  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    const iree_host_size_t entry_slot_index =
        loom_amdgpu_wait_plan_loop_entry_slot_index(node->block_index, slot);
    uint32_t link_index =
        builder->loop_entry_dependency_links[entry_slot_index];
    if (link_index == LOOM_LOW_SCHEDULE_NODE_NONE) {
      continue;
    }

    const uint32_t counter_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
    const uint16_t counter_id = loom_amdgpu_wait_counter_id_from_slot(slot);
    uint16_t strictest_target_count = UINT16_MAX;
    uint32_t strictest_producer = LOOM_LOW_SCHEDULE_NODE_NONE;
    uint32_t strictest_consumer = LOOM_LOW_SCHEDULE_NODE_NONE;
    bool has_active_dependency = false;
    bool is_derived = true;
    while (link_index != LOOM_LOW_SCHEDULE_NODE_NONE) {
      const loom_amdgpu_wait_dependency_t* link =
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
    if (!has_active_dependency) {
      continue;
    }

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
  if (builder->cyclic_frontiers == NULL) {
    return;
  }
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
    builder->outstanding_workgroup_access_counts[slot] =
        frontier->outstanding_workgroup_access_count;

    uint32_t producer_position = 0;
    const uint32_t counter_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
    for (uint32_t i = frontier->producer_start_ordinal;
         i < block->scheduled_node_count; ++i) {
      const uint32_t packet_index = block->scheduled_node_start + i;
      const uint32_t node_index =
          schedule->scheduled_node_indices[packet_index];
      if ((builder->classification.completion_nodes[node_index]
               .producer_counter_mask &
           counter_mask) == 0) {
        continue;
      }
      loom_amdgpu_wait_node_state_t* node_state =
          &builder->classification.node_states[node_index];
      loom_amdgpu_wait_producer_state_t* producer_state =
          loom_amdgpu_wait_plan_producer_state(builder, node_index);
      producer_state->epochs[slot] = builder->counter_epochs[slot];
      producer_state->positions[slot] = ++producer_position;
      if (iree_any_bit_set(
              node_state->flags,
              LOOM_AMDGPU_WAIT_NODE_STATE_UNORDERED_FLAT_COMPLETION)) {
        builder->unordered_flat_counter_mask |= counter_mask;
      }
      if (iree_any_bit_set(counter_mask,
                           LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD)) {
        loom_amdgpu_wait_plan_note_vmem_order(
            builder, &builder->classification.frontier_nodes[node_index]);
      }
    }
    IREE_ASSERT_EQ(producer_position, frontier->outstanding_count);
  }
}

static void loom_amdgpu_wait_plan_verify_cyclic_frontiers(
    const loom_amdgpu_wait_plan_builder_t* builder, uint16_t block_index) {
  if (builder->cyclic_frontiers == NULL) {
    return;
  }
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    const iree_host_size_t frontier_index =
        loom_amdgpu_wait_plan_loop_entry_slot_index(block_index, slot);
    const loom_amdgpu_wait_loop_cyclic_frontier_t* frontier =
        &builder->cyclic_frontiers[frontier_index];
    if (!iree_any_bit_set(frontier->flags,
                          LOOM_AMDGPU_WAIT_LOOP_CYCLIC_FRONTIER_FLAG_VALID)) {
      continue;
    }
    // The outgoing suffix must fit the seeded producer list. Authored partial
    // waits can retire its prefix, but cannot introduce unmodeled producers.
    IREE_ASSERT_LE(builder->outstanding_counts[slot],
                   frontier->outstanding_count);
    IREE_ASSERT_LE(builder->outstanding_write_counts[slot],
                   frontier->outstanding_write_count);
    IREE_ASSERT_LE(builder->outstanding_workgroup_access_counts[slot],
                   frontier->outstanding_workgroup_access_count);
  }
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
  const loom_amdgpu_wait_producer_state_t* producer_state =
      loom_amdgpu_wait_plan_const_producer_state(builder,
                                                 lease_record->node_index);
  const uint32_t producer_block =
      builder->schedule->nodes[lease_record->node_index].block_index;
  const uint32_t insertion_block = action->block_index;
  if (loom_amdgpu_wait_plan_producer_is_drained(
          builder, lease_record->node_index, counter_mask)) {
    return true;
  }
  if (producer_block == insertion_block) {
    return producer_state->epochs[slot] != builder->counter_epochs[slot] ||
           loom_amdgpu_wait_plan_producer_is_complete_in_current_epoch(
               builder, lease_record->node_index, slot);
  }
  // Incoming lease membership already joins all predecessor paths and retains
  // implicit source release, including XCNT group and branch transitions.
  return loom_amdgpu_wait_plan_current_block_satisfies_producer(
             builder, lease_record->node_index, counter_mask) ||
         !loom_amdgpu_wait_frontier_storage_lease_is_active(
             &builder->frontier, action->lease_record_index);
}

static loom_amdgpu_wait_plan_action_flags_t
loom_amdgpu_wait_plan_storage_release_action_flags(
    const loom_amdgpu_wait_plan_builder_t* builder,
    const loom_low_storage_release_action_t* action) {
  const loom_low_allocation_storage_lease_t* lease =
      &builder->allocation->storage_lease_instances[action->lease_record_index];
  if (lease->release_action_index ==
      LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE) {
    return 0;
  }
  const loom_low_storage_release_action_t* indexed_action =
      &builder->allocation
           ->storage_release_actions[lease->release_action_index];
  // The allocator retains exactly one action per lease and rewrites it when an
  // earlier physical conflict wins. A matching insertion point is therefore
  // the same action that the generic packet-hazard builder will report.
  return indexed_action->insertion_node_index == action->insertion_node_index
             ? LOOM_AMDGPU_WAIT_PLAN_ACTION_FLAG_STORAGE_RELEASE
             : 0;
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
  loom_amdgpu_wait_node_state_t* node_state =
      &builder->classification.node_states[node_index];
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
                       LOOM_AMDGPU_WAIT_NODE_STATE_XCNT_VMEM_DRAIN_REQUIRED) &&
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
  } else {
    target_count = loom_amdgpu_wait_plan_incoming_producer_target_count(
        builder, lease_record->node_index, (uint16_t)action->block_index,
        loom_amdgpu_wait_counter_slot_from_id(action->release_class_id));
  }
  if (action->release_class_id == LOOM_AMDGPU_WAIT_COUNTER_X &&
      loom_amdgpu_wait_plan_node_xcnt_group(
          &builder->classification.node_states[action->insertion_node_index]) ==
          LOOM_AMDGPU_WAIT_XCNT_GROUP_VMEM &&
      !iree_any_bit_set(
          builder->classification.node_states[action->insertion_node_index]
              .barrier_counter_mask,
          LOOM_AMDGPU_WAIT_COUNTER_MASK_X)) {
    // VMEM translations are ordered. A VMEM packet that overwrites an older
    // VMEM source makes the hardware internally progress XCNT just far enough
    // to release that source. No wait instruction is required. Cross-block
    // counts are deliberately not reconstructed; the per-block marker records
    // the specific producer proven retired on this path.
    // A replay barrier needs completion before the memory effect itself, so
    // that packet's later result write cannot satisfy the barrier.
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
  if (allocation == NULL || allocation->storage_release_action_count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT(allocation->first_storage_release_action_by_node != NULL);
  for (uint32_t action_index =
           allocation->first_storage_release_action_by_node[node_index];
       action_index != LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE;
       action_index = allocation->storage_release_actions[action_index]
                          .next_same_insertion_node_action_index) {
    const loom_low_storage_release_action_t* action =
        &allocation->storage_release_actions[action_index];
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_wait_plan_handle_storage_release_action(builder, action));
  }
  return iree_ok_status();
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
  const loom_low_schedule_node_t* write_node =
      &builder->schedule->nodes[node_index];
  const uint32_t program_point =
      allocation->liveness.blocks[write_node->block_index].start_point +
      write_node->scheduled_ordinal;
  const loom_low_allocation_storage_lease_selection_t* incoming_selection =
      builder->frontier.storage_leases.active_words != NULL
          ? &builder->frontier.storage_leases.active_selection
          : NULL;
  // A release executes before the instruction at its end point, so that point
  // remains included. This matters when a structural definition coalesces into
  // earlier physical result writes. Incoming dynamic instances are included
  // independently of their static linear interval.
  loom_low_allocation_storage_lease_unit_query_t query;
  loom_low_allocation_storage_lease_unit_query_initialize(
      allocation->storage_lease_unit_index,
      builder->schedule->target.descriptor_set, descriptor_reg_class_id,
      location_kind, location_base, location_count, program_point,
      (uint64_t)program_point + 1u, incoming_selection, &query);
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
    const bool incoming_lease_active =
        loom_amdgpu_wait_frontier_storage_lease_is_active(&builder->frontier,
                                                          storage_lease_index);
    // The query already established temporal or incoming membership. Only a
    // current instruction's own new instance is excluded; a pending incoming
    // instance of that same instruction still needs completion on a backedge.
    if (!incoming_lease_active && record->node_index == node_index) {
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
    const loom_low_storage_release_action_t action = {
        .insertion_node_index = node_index,
        .block_index = write_node->block_index,
        .scheduled_ordinal = write_node->scheduled_ordinal,
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
    if (record->kind == LOOM_LOW_STORAGE_LEASE_RESULT_WRITE &&
        record->node_index == continuation_producer_node &&
        ((record->release_class_id == LOOM_AMDGPU_WAIT_COUNTER_LDS &&
          builder->classification.frontier_nodes[node_index]
                  .read_counter_mask == LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS) ||
         loom_amdgpu_wait_plan_storage_release_is_ordered_vmem_reuse(
             builder, &action, record, reason))) {
      // Disjoint partial writes may overlap within one ordered result
      // pipeline: completing the continuation also completes the preserved
      // part. Other pipelines must release the old result before this write;
      // storage identity alone proves neither completion nor safe overlap.
      continue;
    }
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
      const uint32_t slot =
          loom_amdgpu_wait_counter_slot_from_id(record->release_class_id);
      const loom_low_schedule_node_t* producer =
          &builder->schedule->nodes[record->node_index];
      const iree_host_size_t frontier_index =
          loom_amdgpu_wait_plan_loop_entry_slot_index(
              (uint16_t)write_node->block_index, slot);
      const bool has_cyclic_position =
          producer->block_index == write_node->block_index &&
          producer->scheduled_ordinal >= write_node->scheduled_ordinal &&
          builder->cyclic_frontiers != NULL &&
          iree_any_bit_set(builder->cyclic_frontiers[frontier_index].flags,
                           LOOM_AMDGPU_WAIT_LOOP_CYCLIC_FRONTIER_FLAG_VALID);
      if (has_cyclic_position ||
          producer->block_index != write_node->block_index) {
        // Use the same completion and partial-count facts as payload
        // consumers, then retire the old storage instance before publishing
        // the new one at block exit. Canonical cycles retain exact positions;
        // cross-block producers may retain a known younger local suffix.
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_wait_plan_handle_storage_release_action(builder,
                                                                &action));
        loom_amdgpu_wait_frontier_retire_storage_lease(&builder->frontier,
                                                       storage_lease_index);
        continue;
      }
      // Other incoming counts remain path-dependent. Full completion protects
      // the concrete lease on every predecessor path.
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

  const loom_value_ordinal_t result_ordinal =
      loom_low_schedule_node_const_result_ordinals(node)[result_index];
  const loom_value_ordinal_t source_ordinal =
      loom_low_placement_tied_source_for_value_ordinal(
          &builder->allocation->placement, result_ordinal);
  return builder->producer_nodes[source_ordinal];
}

static iree_status_t loom_amdgpu_wait_plan_handle_packet_move_writes(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index,
    const loom_low_move_group_t* move_group) {
  const loom_low_move_t* moves =
      &builder->allocation->moves[move_group->moves.start];
  iree_host_size_t move_index = 0;
  while (move_index < move_group->moves.count) {
    const loom_low_move_location_t* destination =
        &moves[move_index].destination;
    uint32_t location_count = 1;
    while (move_index + location_count < move_group->moves.count) {
      const loom_low_move_location_t* next_destination =
          &moves[move_index + location_count].destination;
      if (next_destination->location_kind != destination->location_kind ||
          next_destination->descriptor_reg_class_id !=
              destination->descriptor_reg_class_id ||
          (uint64_t)next_destination->location !=
              (uint64_t)destination->location + location_count) {
        break;
      }
      ++location_count;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_handle_physical_write_range(
        builder, node_index, LOOM_LOW_SCHEDULE_NODE_NONE,
        destination->location_kind, destination->descriptor_reg_class_id,
        destination->location, location_count));
    move_index += location_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_handle_materialized_result_writes(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (builder->allocation == NULL ||
      builder->allocation->storage_lease_instance_count == 0 ||
      !iree_any_bit_set(builder->classification.node_states[node_index].flags,
                        LOOM_AMDGPU_WAIT_NODE_STATE_MATERIALIZES_RESULTS)) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  if (loom_amdgpu_wait_plan_op_has_packet_transfers(node->op)) {
    const loom_low_allocation_packet_move_group_t* group =
        loom_amdgpu_wait_plan_packet_transfer_group(builder, node_index);
    return group != NULL ? loom_amdgpu_wait_plan_handle_packet_move_writes(
                               builder, node_index, &group->move_group)
                         : iree_ok_status();
  }
  const loom_low_packet_view_t packet =
      loom_low_packet_at_node(builder->schedule, node_index);
  for (uint16_t i = 0; i < node->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_low_packet_result_assignment(builder->allocation, &packet, i);
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
    if (edge_copy->kind == LOOM_LOW_ALLOCATION_COPY_COALESCED) {
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
      &builder->classification.frontier_nodes[node_index];
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
    const loom_amdgpu_wait_dependency_t* link =
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
      } else if (producer_block == consumer_block) {
        // The producer has not reissued in this block yet. Incoming completion
        // therefore describes the older instance carried by this use.
        if (loom_amdgpu_wait_frontier_producer_is_complete(
                &builder->frontier, link->producer_node, counter_mask)) {
          continue;
        }
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
          if (derived_frontier->outstanding_count == 0) {
            continue;
          }
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
        // A producer-block or current-block drain proves completion directly.
        // The incoming frontier also retains full drains through intermediate
        // blocks, with conservative state for every unresolved incoming path.
        if (loom_amdgpu_wait_plan_producer_is_drained(
                builder, link->producer_node, counter_mask) ||
            loom_amdgpu_wait_plan_current_block_satisfies_producer(
                builder, link->producer_node, counter_mask) ||
            loom_amdgpu_wait_frontier_producer_is_complete(
                &builder->frontier, link->producer_node, counter_mask)) {
          continue;
        }
        target_count = loom_amdgpu_wait_plan_incoming_producer_target_count(
            builder, link->producer_node, (uint16_t)consumer_block, slot);
        if (loom_amdgpu_wait_loop_analysis_cyclic_interval(
                &builder->loop_analysis, link->producer_node, node_index) !=
            NULL) {
          reason =
              target_count != 0
                  ? LOOM_AMDGPU_WAIT_PLAN_REASON_LOOP_CARRIED_DERIVED_SSA_USE
                  : LOOM_AMDGPU_WAIT_PLAN_REASON_LOOP_CARRIED_CONSERVATIVE_SSA_USE;
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
  return iree_any_bit_set(builder->classification.node_states[node_index].flags,
                          LOOM_AMDGPU_WAIT_NODE_STATE_USES_VECTOR_ALU);
}

static iree_status_t loom_amdgpu_wait_plan_handle_trans_result_use(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (!loom_amdgpu_wait_plan_node_is_trans_result_consumer(builder,
                                                           node_index)) {
    return iree_ok_status();
  }
  const loom_low_packet_view_t packet =
      loom_low_packet_at_node(builder->schedule, node_index);
  for (uint16_t i = 0; i < packet.node->operand_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_low_packet_operand_assignment(builder->allocation, &packet, i);
    uint32_t producer_node = LOOM_LOW_SCHEDULE_NODE_NONE;
    if (!loom_amdgpu_trans_result_window_query_assignment_origin(
            &builder->trans_result_window, assignment, &producer_node)) {
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
  return iree_any_bit_set(builder->classification.node_states[node_index].flags,
                          LOOM_AMDGPU_WAIT_NODE_STATE_USES_SCALAR_ALU);
}

static bool loom_amdgpu_wait_plan_node_uses_vector_alu(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  return iree_any_bit_set(builder->classification.node_states[node_index].flags,
                          LOOM_AMDGPU_WAIT_NODE_STATE_USES_VECTOR_ALU);
}

static loom_amdgpu_sgpr_read_hazard_alu_flags_t
loom_amdgpu_wait_plan_node_sgpr_read_alu_flags(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  loom_amdgpu_sgpr_read_hazard_alu_flags_t flags = 0;
  if (loom_amdgpu_wait_plan_node_uses_scalar_alu(builder, node_index)) {
    flags |= LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR;
  }
  if (loom_amdgpu_wait_plan_node_uses_vector_alu(builder, node_index)) {
    flags |= LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_VECTOR;
  }
  return flags;
}

static iree_status_t loom_amdgpu_wait_plan_handle_sgpr_read_hazard(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (!loom_amdgpu_wait_plan_has_sgpr_read_state(builder)) {
    return iree_ok_status();
  }
  const loom_amdgpu_sgpr_read_hazard_alu_flags_t alu_flags =
      loom_amdgpu_wait_plan_node_sgpr_read_alu_flags(builder, node_index);
  if (alu_flags == 0) {
    return iree_ok_status();
  }
  const loom_low_packet_view_t packet =
      loom_low_packet_at_node(builder->schedule, node_index);
  for (uint16_t i = 0; i < packet.node->operand_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_low_packet_operand_assignment(builder->allocation, &packet, i);
    uint32_t producer_node = LOOM_LOW_SCHEDULE_NODE_NONE;
    if (!loom_amdgpu_sgpr_read_hazard_query_read(&builder->sgpr_read_hazard,
                                                 assignment, alu_flags,
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
  const loom_low_packet_view_t packet =
      loom_low_packet_at_node(builder->schedule, node_index);
  for (uint16_t i = 0; i < packet.node->operand_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_low_packet_operand_assignment(builder->allocation, &packet, i);
    loom_amdgpu_sgpr_read_hazard_track_read(&builder->sgpr_read_hazard,
                                            assignment);
  }
}

static void loom_amdgpu_wait_plan_record_sgpr_read_writes(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (!loom_amdgpu_wait_plan_has_sgpr_read_state(builder)) {
    return;
  }
  const loom_amdgpu_sgpr_read_hazard_alu_flags_t alu_flags =
      loom_amdgpu_wait_plan_node_sgpr_read_alu_flags(builder, node_index);
  if (alu_flags == 0) {
    return;
  }
  const loom_low_packet_view_t packet =
      loom_low_packet_at_node(builder->schedule, node_index);
  for (uint16_t i = 0; i < packet.node->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_low_packet_result_assignment(builder->allocation, &packet, i);
    loom_amdgpu_sgpr_read_hazard_record_write(
        &builder->sgpr_read_hazard, assignment, alu_flags, node_index);
  }
}

static iree_status_t loom_amdgpu_wait_plan_handle_barrier(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  const loom_amdgpu_wait_node_state_t* node_state =
      &builder->classification.node_states[node_index];
  const loom_amdgpu_wait_completion_node_t* completion_node =
      &builder->classification.completion_nodes[node_index];
  if ((node_state->barrier_counter_mask |
       completion_node->workgroup_barrier_counter_mask) == 0) {
    return iree_ok_status();
  }
  const loom_amdgpu_wait_memory_space_flags_t generic_space =
      loom_amdgpu_wait_memory_space_flag(LOOM_LOW_MEMORY_SPACE_GENERIC);
  const loom_amdgpu_wait_memory_space_flags_t workgroup_space =
      loom_amdgpu_wait_memory_space_flag(LOOM_LOW_MEMORY_SPACE_WORKGROUP);
  uint32_t outstanding_counter_mask =
      loom_amdgpu_wait_plan_outstanding_counter_mask(
          builder->outstanding_counts, node_state->barrier_counter_mask);
  if (iree_any_bit_set(node_state->barrier_counter_mask,
                       LOOM_AMDGPU_WAIT_COUNTER_MASK_X) &&
      loom_amdgpu_wait_frontier_active_xcnt_groups(&builder->frontier) != 0) {
    // Translation leases may enter through a fallthrough block without a
    // hardware branch to drain them. Memory completion frontiers do not carry
    // this independently tracked source lifetime.
    outstanding_counter_mask |= LOOM_AMDGPU_WAIT_COUNTER_MASK_X;
  }
  outstanding_counter_mask |=
      loom_amdgpu_wait_frontier_memory_query(
          &builder->frontier, generic_space,
          LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_READ |
              LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE) &
      node_state->barrier_counter_mask;
  outstanding_counter_mask |= loom_amdgpu_wait_plan_outstanding_counter_mask(
      builder->outstanding_workgroup_access_counts,
      completion_node->workgroup_barrier_counter_mask);
  outstanding_counter_mask |=
      loom_amdgpu_wait_frontier_memory_query(
          &builder->frontier, workgroup_space,
          LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_READ |
              LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE) &
      completion_node->workgroup_barrier_counter_mask;
  if (outstanding_counter_mask == 0) {
    return iree_ok_status();
  }
  const loom_amdgpu_wait_plan_reason_t reason =
      iree_any_bit_set(node_state->flags,
                       LOOM_AMDGPU_WAIT_NODE_STATE_SYSTEM_SCOPE_STORE_DRAIN)
          ? LOOM_AMDGPU_WAIT_PLAN_REASON_SYSTEM_SCOPE_STORE
          : LOOM_AMDGPU_WAIT_PLAN_REASON_BARRIER;
  return loom_amdgpu_wait_plan_drain_mask(
      builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED, reason, node_index,
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
  const loom_low_packet_view_t packet =
      loom_low_packet_at_node(builder->schedule, node_index);
  for (uint16_t i = 0; i < packet.node->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_low_packet_result_assignment(builder->allocation, &packet, i);
    loom_amdgpu_trans_result_window_clear_assignment(
        &builder->trans_result_window, assignment);
  }
}

static void loom_amdgpu_wait_plan_record_trans_results(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (!loom_amdgpu_wait_plan_has_trans_result_state(builder)) {
    return;
  }
  const loom_amdgpu_wait_node_state_t* node_state =
      &builder->classification.node_states[node_index];
  const loom_amdgpu_wait_completion_node_t* completion_node =
      &builder->classification.completion_nodes[node_index];
  if (!iree_any_bit_set(node_state->flags,
                        LOOM_AMDGPU_WAIT_NODE_STATE_TRANSCENDENTAL) ||
      !iree_any_bit_set(completion_node->producer_counter_mask,
                        LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU)) {
    return;
  }
  const loom_low_packet_view_t packet =
      loom_low_packet_at_node(builder->schedule, node_index);
  for (uint16_t i = 0; i < packet.node->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_low_packet_result_assignment(builder->allocation, &packet, i);
    loom_amdgpu_trans_result_window_record_assignment(
        &builder->trans_result_window, assignment, node_index);
  }
}

static void loom_amdgpu_wait_plan_apply_trans_result_interval(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (!loom_amdgpu_wait_plan_has_trans_result_state(builder)) {
    return;
  }
  const loom_amdgpu_wait_node_state_t* node_state =
      &builder->classification.node_states[node_index];
  loom_amdgpu_trans_result_packet_flags_t packet_flags = 0;
  if (iree_any_bit_set(node_state->flags,
                       LOOM_AMDGPU_WAIT_NODE_STATE_USES_VECTOR_ALU)) {
    packet_flags |= LOOM_AMDGPU_TRANS_RESULT_PACKET_FLAG_VECTOR_ALU;
  }
  if (iree_any_bit_set(node_state->flags,
                       LOOM_AMDGPU_WAIT_NODE_STATE_TRANSCENDENTAL)) {
    packet_flags |= LOOM_AMDGPU_TRANS_RESULT_PACKET_FLAG_TRANSCENDENTAL;
  }
  loom_amdgpu_trans_result_window_advance(&builder->trans_result_window,
                                          packet_flags);
}

static bool loom_amdgpu_wait_plan_node_expires_trans_results(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  if (!loom_amdgpu_wait_plan_has_trans_result_state(builder)) {
    return false;
  }
  const loom_amdgpu_wait_frontier_node_t* frontier_node =
      &builder->classification.frontier_nodes[node_index];
  const uint32_t expiring_counter_mask =
      LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM | LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS;
  return iree_any_bit_set(
      frontier_node->read_counter_mask | frontier_node->write_counter_mask,
      expiring_counter_mask);
}

static iree_status_t loom_amdgpu_wait_plan_note_producer(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  const loom_amdgpu_wait_node_state_t* node_state =
      &builder->classification.node_states[node_index];
  const loom_amdgpu_wait_frontier_node_t* frontier_node =
      &builder->classification.frontier_nodes[node_index];
  const loom_amdgpu_wait_completion_node_t* completion_node =
      &builder->classification.completion_nodes[node_index];
  const uint32_t counter_mask = completion_node->producer_counter_mask;
  if (iree_any_bit_set(counter_mask, LOOM_AMDGPU_WAIT_COUNTER_MASK_TENSOR)) {
    builder->tensor_issue_requires_drain = true;
  }
  // A pre-control drain completes older work, but an opaque control operation
  // may start new work of its own. Preserve authored waits after that boundary.
  // Keep historical completion separate: older producers remain complete.
  builder->known_empty_counter_mask &= ~(node_state->barrier_counter_mask != 0
                                             ? LOOM_AMDGPU_WAIT_COUNTER_MASK_ALL
                                             : counter_mask);
  loom_amdgpu_wait_producer_state_t* producer_state =
      counter_mask == 0
          ? NULL
          : loom_amdgpu_wait_plan_producer_state(builder, node_index);
  if (iree_any_bit_set(node_state->flags,
                       LOOM_AMDGPU_WAIT_NODE_STATE_UNORDERED_FLAT_COMPLETION)) {
    builder->unordered_flat_counter_mask |=
        frontier_node->read_counter_mask | frontier_node->write_counter_mask;
  }
  if (iree_any_bit_set(counter_mask, LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD)) {
    loom_amdgpu_wait_plan_note_vmem_order(builder, frontier_node);
  }
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    if ((counter_mask & loom_amdgpu_wait_counter_mask_from_slot(slot)) == 0) {
      continue;
    }
    producer_state->epochs[slot] = builder->counter_epochs[slot];
    const uint32_t active_position =
        iree_math_saturating_add_u32(builder->completed_position_counts[slot],
                                     builder->outstanding_counts[slot]);
    producer_state->positions[slot] =
        iree_math_saturating_add_u32(active_position, 1u);
  }
  loom_amdgpu_wait_plan_increment_outstanding_counts(
      builder->outstanding_counts, counter_mask);
  loom_amdgpu_wait_plan_increment_outstanding_counts(
      builder->outstanding_write_counts, completion_node->write_counter_mask);
  loom_amdgpu_wait_plan_increment_outstanding_counts(
      builder->outstanding_workgroup_access_counts,
      completion_node->workgroup_access_counter_mask);
  const loom_amdgpu_wait_xcnt_group_t xcnt_group =
      loom_amdgpu_wait_plan_node_xcnt_group(node_state);
  if (xcnt_group != LOOM_AMDGPU_WAIT_XCNT_GROUP_NONE) {
    IREE_ASSERT(iree_any_bit_set(completion_node->producer_counter_mask,
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

static bool loom_amdgpu_wait_plan_partial_bound_orders_producers(
    const loom_amdgpu_wait_plan_builder_t* builder, uint16_t counter_id) {
  if (iree_any_bit_set(builder->unordered_flat_counter_mask,
                       loom_amdgpu_wait_counter_mask(counter_id))) {
    return false;
  }
  if (counter_id == LOOM_AMDGPU_WAIT_COUNTER_LDS ||
      counter_id == LOOM_AMDGPU_WAIT_COUNTER_TENSOR) {
    return true;
  }
  return counter_id == LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD &&
         builder->vmem_epoch_order_class >
             LOOM_AMDGPU_VMEM_RESULT_ORDER_UNKNOWN;
}

static iree_status_t loom_amdgpu_wait_plan_handle_partial_wait(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  const loom_amdgpu_wait_node_state_t* node_state =
      &builder->classification.node_states[node_index];
  if (!iree_any_bit_set(node_state->flags,
                        LOOM_AMDGPU_WAIT_NODE_STATE_EXPLICIT_WAIT)) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  const loom_amdgpu_wait_packet_bounds_t* wait_bounds =
      &builder->classification.wait_bounds[node_state->state.wait_bounds_index];
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    const uint16_t target_count = wait_bounds->target_counts[slot];
    if (target_count == 0 || target_count == UINT16_MAX) {
      continue;
    }
    const uint16_t counter_id = loom_amdgpu_wait_counter_id_from_slot(slot);
    const uint32_t outstanding_before = builder->outstanding_counts[slot];
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_append_action(
        builder, (loom_amdgpu_wait_plan_action_t){
                     .kind = LOOM_AMDGPU_WAIT_PLAN_ACTION_EXPLICIT,
                     .reason = LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET,
                     .counter_id = counter_id,
                     .target_count = target_count,
                     .block_index = node->block_index,
                     .node_index = node_index,
                     .scheduled_ordinal = node->scheduled_ordinal,
                     .producer_node = LOOM_LOW_SCHEDULE_NODE_NONE,
                     .consumer_node = LOOM_LOW_SCHEDULE_NODE_NONE,
                     .outstanding_before = outstanding_before,
                 }));
    // A bound on the hardware total also bounds the local ordered suffix.
    // It cannot identify an out-of-order producer, nor become a full drain
    // when the local count is smaller than the authored nonzero threshold.
    if (target_count < outstanding_before &&
        loom_amdgpu_wait_plan_partial_bound_orders_producers(builder,
                                                             counter_id)) {
      loom_amdgpu_wait_plan_apply_counter_progress(builder, node_index,
                                                   LOOM_LOW_SCHEDULE_NODE_NONE,
                                                   counter_id, target_count);
    }
    ++builder->progress_event_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_handle_tensor_issue(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  const loom_amdgpu_wait_frontier_node_t* frontier_node =
      &builder->classification.frontier_nodes[node_index];
  if (!builder->tensor_issue_requires_drain ||
      !iree_any_bit_set(
          frontier_node->read_counter_mask | frontier_node->write_counter_mask,
          LOOM_AMDGPU_WAIT_COUNTER_MASK_TENSOR) ||
      !loom_amdgpu_processor_properties_have_scheduling(
          builder->processor_properties,
          LOOM_AMDGPU_PROCESSOR_SCHEDULING_TENSOR_ISSUE_DRAIN)) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  const uint32_t slot =
      loom_amdgpu_wait_counter_slot_from_id(LOOM_AMDGPU_WAIT_COUNTER_TENSOR);
  const uint32_t outstanding_before = builder->outstanding_counts[slot];
  // Preserve the hardware bound even when the local packet count is smaller.
  // Clamping it would turn an issue-limit wait into a completion wait.
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_append_action(
      builder, (loom_amdgpu_wait_plan_action_t){
                   .kind = LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED,
                   .reason = LOOM_AMDGPU_WAIT_PLAN_REASON_TENSOR_ISSUE_DRAIN,
                   .counter_id = LOOM_AMDGPU_WAIT_COUNTER_TENSOR,
                   .target_count = LOOM_AMDGPU_TENSOR_ISSUE_MAXIMUM_PENDING,
                   .block_index = node->block_index,
                   .node_index = node_index,
                   .scheduled_ordinal = node->scheduled_ordinal,
                   .producer_node = LOOM_LOW_SCHEDULE_NODE_NONE,
                   .consumer_node = node_index,
                   .outstanding_before = outstanding_before,
               }));
  if (outstanding_before > LOOM_AMDGPU_TENSOR_ISSUE_MAXIMUM_PENDING) {
    loom_amdgpu_wait_plan_apply_counter_progress(
        builder, node_index, LOOM_LOW_SCHEDULE_NODE_NONE,
        LOOM_AMDGPU_WAIT_COUNTER_TENSOR,
        LOOM_AMDGPU_TENSOR_ISSUE_MAXIMUM_PENDING);
  }
  return iree_ok_status();
}

static bool loom_amdgpu_wait_plan_full_wait_is_redundant(
    const loom_amdgpu_wait_plan_builder_t* builder,
    const loom_amdgpu_wait_node_state_t* node_state) {
  const uint32_t counter_mask = node_state->explicit_wait_counter_mask;
  if (!iree_any_bit_set(node_state->flags,
                        LOOM_AMDGPU_WAIT_NODE_STATE_EXPLICIT_WAIT) ||
      counter_mask == 0 ||
      iree_any_bit_set(counter_mask, LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU) ||
      (counter_mask & ~builder->known_empty_counter_mask) != 0) {
    return false;
  }
  // Full drains establish emptiness even for unknown incoming work. Retain
  // every authored partial bound; packed ALU controls are not ordinary
  // remaining-memory-operation counts.
  const loom_amdgpu_wait_packet_bounds_t* wait_bounds =
      &builder->classification.wait_bounds[node_state->state.wait_bounds_index];
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    const uint16_t bound = wait_bounds->target_counts[slot];
    if (bound != UINT16_MAX && bound != 0) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_wait_plan_node_is_elided_wait(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  return builder->elided_wait_nodes != NULL &&
         (builder->elided_wait_nodes[node_index / 64] &
          (UINT64_C(1) << (node_index % 64))) != 0;
}

static bool loom_amdgpu_wait_plan_node_has_zero_native_work(
    const loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  return iree_any_bit_set(builder->classification.node_states[node_index].flags,
                          LOOM_AMDGPU_WAIT_NODE_STATE_ZERO_NATIVE_WORK) ||
         loom_amdgpu_wait_plan_node_is_elided_wait(builder, node_index);
}

static iree_status_t loom_amdgpu_wait_plan_process_node(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  loom_amdgpu_wait_node_state_t* node_state =
      &builder->classification.node_states[node_index];

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
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_handle_tensor_issue(builder, node_index));
  if (loom_amdgpu_wait_plan_full_wait_is_redundant(builder, node_state)) {
    if (builder->elided_wait_nodes == NULL) {
      const iree_host_size_t word_count =
          builder->schedule->node_count / 64 +
          (builder->schedule->node_count % 64 != 0);
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          builder->arena, word_count, sizeof(*builder->elided_wait_nodes),
          (void**)&builder->elided_wait_nodes));
      memset(builder->elided_wait_nodes, 0,
             word_count * sizeof(*builder->elided_wait_nodes));
    }
    builder->elided_wait_nodes[node_index / 64] |= UINT64_C(1)
                                                   << (node_index % 64);
    return iree_ok_status();
  }
  if (node_state->explicit_wait_counter_mask != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_drain_mask(
        builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_EXPLICIT,
        LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET, node_index,
        LOOM_LOW_SCHEDULE_NODE_NONE, node_state->explicit_wait_counter_mask));
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_handle_partial_wait(builder, node_index));
  if (loom_amdgpu_wait_plan_node_expires_trans_results(builder, node_index)) {
    loom_amdgpu_wait_plan_expire_trans_results(builder);
  }
  loom_amdgpu_wait_plan_track_sgpr_reads(builder, node_index);
  loom_amdgpu_wait_plan_apply_trans_result_interval(builder, node_index);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_note_producer(builder, node_index));
  const uint32_t reset_counter_mask = node_state->explicit_wait_counter_mask |
                                      node_state->implicit_wait_counter_mask;
  const uint32_t producer_counter_mask =
      builder->classification.completion_nodes[node_index]
          .producer_counter_mask;
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
    builder->insertion.anchor_node = LOOM_LOW_SCHEDULE_NODE_NONE;
    ++builder->block_epoch;
    builder->current_block_full_drain_counter_mask = 0;
    builder->known_empty_counter_mask = 0;
    builder->xcnt_group = LOOM_AMDGPU_WAIT_XCNT_GROUP_NONE;
    builder->vmem_epoch_order_class = LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE;
    builder->unordered_flat_counter_mask = 0;
    memset(builder->counter_epochs, 0, sizeof(builder->counter_epochs));
    memset(builder->completed_position_counts, 0,
           sizeof(builder->completed_position_counts));
    memset(builder->retirement_ordinals, 0,
           sizeof(builder->retirement_ordinals));
    memset(builder->outstanding_counts, 0, sizeof(builder->outstanding_counts));
    memset(builder->outstanding_write_counts, 0,
           sizeof(builder->outstanding_write_counts));
    memset(builder->outstanding_workgroup_access_counts, 0,
           sizeof(builder->outstanding_workgroup_access_counts));
    loom_amdgpu_trans_result_window_clear(&builder->trans_result_window);
    loom_amdgpu_sgpr_read_hazard_begin_block(&builder->sgpr_read_hazard);
    loom_amdgpu_wait_plan_seed_cyclic_frontiers(builder, (uint16_t)block_index);
    loom_amdgpu_wait_frontier_begin_block(&builder->frontier,
                                          (uint16_t)block_index);
    builder->tensor_issue_requires_drain = iree_any_bit_set(
        loom_amdgpu_wait_frontier_memory_query(
            &builder->frontier,
            loom_amdgpu_wait_memory_space_flag(LOOM_LOW_MEMORY_SPACE_GENERIC),
            LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_READ |
                LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE),
        LOOM_AMDGPU_WAIT_COUNTER_MASK_TENSOR);
    for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
      const uint32_t packet_index = block->scheduled_node_start + i;
      IREE_ASSERT_LT(packet_index, schedule->scheduled_node_count);
      const uint32_t node_index =
          schedule->scheduled_node_indices[packet_index];
      IREE_ASSERT_LT(node_index, schedule->node_count);
      const loom_amdgpu_address_state_plan_t* address_state =
          builder->insertion.address_state;
      if (builder->insertion.address_state_cursor <
              address_state->transition_count &&
          address_state->transitions[builder->insertion.address_state_cursor]
                  .node_index == node_index) {
        // The emitter places this transition before this node's wait packets.
        builder->insertion.anchor_node = LOOM_LOW_SCHEDULE_NODE_NONE;
        ++builder->insertion.address_state_cursor;
      }
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_wait_plan_process_node(builder, node_index));
      if (!loom_amdgpu_wait_plan_node_has_zero_native_work(builder,
                                                           node_index)) {
        builder->insertion.anchor_node = LOOM_LOW_SCHEDULE_NODE_NONE;
      }
    }
    loom_amdgpu_wait_plan_verify_cyclic_frontiers(builder,
                                                  (uint16_t)block_index);
    loom_amdgpu_wait_frontier_end_block(&builder->frontier);
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_wait_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_address_state_plan_t* address_state,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* transient_arena,
    loom_amdgpu_wait_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_wait_plan_t){0};
  IREE_ASSERT(schedule->value_count == 0 ||
              schedule->value_producer_nodes != NULL);
  IREE_ASSERT(allocation->edge_copy_count == 0 ||
              allocation->first_coalesced_incoming_copy_by_value_ordinal !=
                  NULL);
  loom_amdgpu_wait_plan_builder_t builder = {
      .schedule = schedule,
      .allocation = allocation,
      .arena = arena,
      .transient_arena = transient_arena,
      .producer_nodes = schedule->value_producer_nodes,
      .insertion = {.address_state = address_state},
      .processor_properties =
          loom_amdgpu_target_processor_properties_from_resolved_target(
              &schedule->target),
  };
  if (allocation->edge_copy_count != 0) {
    builder.first_coalesced_incoming_copy_by_value_ordinal =
        allocation->first_coalesced_incoming_copy_by_value_ordinal;
    builder.edge_copies = allocation->edge_copies;
  }
  loom_amdgpu_wait_packet_analyze_target(schedule->target.descriptor_set,
                                         &builder.wait_packet_target);
  loom_amdgpu_wait_actions_initialize(&builder.actions);
  iree_status_t status = loom_amdgpu_wait_classification_build(
      schedule, allocation, builder.processor_properties,
      &builder.wait_packet_target, transient_arena, &builder.classification);
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_plan_allocate_dependency_heads(&builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_plan_allocate_producer_states(&builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_plan_build_dependency_links(&builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_plan_relocate_loop_entry_dependencies(&builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_plan_allocate_physical_state(&builder);
  }
  if (iree_status_is_ok(status)) {
    loom_amdgpu_wait_completion_analyze(
        schedule, builder.first_dependency_link_by_consumer,
        builder.dependency_links, builder.classification.completion_nodes);
    status = loom_amdgpu_wait_frontier_initialize(
        schedule, allocation, builder.classification.frontier_nodes,
        builder.classification.completion_nodes, builder.dependency_links,
        builder.dependency_link_count, builder.loop_entry_drain_counter_masks,
        transient_arena, &builder.frontier);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_loop_analysis_build_cyclic_frontiers(
        &builder.loop_analysis, builder.classification.completion_nodes,
        builder.first_dependency_link_by_consumer, builder.dependency_links,
        builder.dependency_link_count, transient_arena,
        &builder.cyclic_frontiers);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_plan_build_actions(&builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_actions_finalize(&builder.actions, arena);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_actions_build_common_tables(
        schedule, allocation, &builder.classification,
        builder.elided_wait_nodes, builder.progress_event_count,
        &builder.actions, arena, &builder.progress, &builder.hazard_plan);
  }
  if (iree_status_is_ok(status)) {
    *out_plan = (loom_amdgpu_wait_plan_t){
        .schedule = schedule,
        .allocation = allocation,
        .progress = builder.progress,
        .hazard_plan = builder.hazard_plan,
        .actions = builder.actions.actions,
        .action_count = builder.actions.action_count,
        .elided_wait_nodes = builder.elided_wait_nodes,
    };
    if (builder.hazard_plan.progress == &builder.progress) {
      out_plan->hazard_plan.progress = &out_plan->progress;
    }
  }
  return status;
}

iree_status_t loom_amdgpu_wait_plan_format_json(
    const loom_amdgpu_wait_plan_t* plan, iree_string_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(builder);
  return loom_low_packet_hazard_plan_format_json(&plan->hazard_plan, builder);
}
