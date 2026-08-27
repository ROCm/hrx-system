// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/vopd_plan.h"

#include <inttypes.h>
#include <stdint.h>
#include <string.h>

#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/packet.h"
#include "loom/ir/ir.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/encoding/encoding.h"
#include "loom/target/arch/amdgpu/planning/descriptor_semantics.h"
#include "loom/target/arch/amdgpu/planning/matrix_coexecution.h"
#include "loom/target/arch/amdgpu/planning/structural_packet.h"
#include "loom/target/arch/amdgpu/planning/vopd_data.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

typedef struct loom_amdgpu_vopd_candidate_component_t {
  // Descriptor-local rule that decoded this component.
  const loom_amdgpu_vopd_component_rule_t* rule;
  // Destination VGPR.
  uint16_t vdst;
  // First explicit source VGPR before unified-source encoding bias.
  uint16_t src0;
  // Second explicit source VGPR.
  uint16_t vsrc1;
  // Unified source selector for forms without a VGPR SRC0.
  uint16_t src0_selector;
  // Component-local payload flags.
  loom_amdgpu_vopd_pair_flags_t flags;
  // Component literal payload when LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL is set.
  uint32_t literal_u32;
} loom_amdgpu_vopd_candidate_component_t;

typedef struct loom_amdgpu_vopd_candidate_pair_t {
  // Why this VOPD pair can be formed.
  loom_amdgpu_vopd_pair_reason_t reason;
  // X-slot component.
  loom_amdgpu_vopd_candidate_component_t x;
  // Y-slot component.
  loom_amdgpu_vopd_candidate_component_t y;
  // Pair-local payload and encoding flags.
  loom_amdgpu_vopd_pair_flags_t flags;
  // Shared literal payload when LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL is set.
  uint32_t literal_u32;
} loom_amdgpu_vopd_candidate_pair_t;

typedef struct loom_amdgpu_vopd_pair_analysis_t {
  // Decoded first component facts.
  loom_amdgpu_vopd_candidate_component_t first_component;
  // True if the first packet decoded as a supported VOPD component form.
  bool first_eligible;
  // Decoded second component facts.
  loom_amdgpu_vopd_candidate_component_t second_component;
  // True if the second packet decoded as a supported VOPD component form.
  bool second_eligible;
  // Candidate pair facts populated when both components can be paired.
  loom_amdgpu_vopd_candidate_pair_t candidate;
  // True if the adjacent packets satisfy VOPD pair legality.
  bool matched;
  // Stable rejection reason when |matched| is false.
  loom_amdgpu_vopd_rejection_reason_t rejection_reason;
  // Failed physical register constraints when rejected for that reason.
  loom_amdgpu_vopd_register_constraint_flags_t register_constraint_flags;
} loom_amdgpu_vopd_pair_analysis_t;

typedef struct loom_amdgpu_vopd_visible_packet_t {
  // Final scheduled packet.
  loom_low_packet_view_t packet;
  // Native facts for a structural packet. Descriptor packets leave this zero.
  loom_amdgpu_structural_packet_info_t structural;
} loom_amdgpu_vopd_visible_packet_t;

typedef enum loom_amdgpu_vopd_packet_flag_bits_e {
  // Packet has a native insertion point that prevents second-component fusion.
  LOOM_AMDGPU_VOPD_PACKET_FLAG_INSERTION_BLOCKED = 1u << 0,
  // Packet has a planned ALU depctr wait before it.
  LOOM_AMDGPU_VOPD_PACKET_FLAG_TRANS_RESULT_CLEAR = 1u << 1,
  // Packet forwards dependencies without materializing a target packet.
  LOOM_AMDGPU_VOPD_PACKET_FLAG_TRANSPARENT = 1u << 2,
} loom_amdgpu_vopd_packet_flag_bits_t;
typedef uint8_t loom_amdgpu_vopd_packet_flags_t;

typedef enum loom_amdgpu_vopd_trans_result_vgpr_flag_bits_e {
  // Physical VGPR currently holds a result in the TRANS hazard window.
  LOOM_AMDGPU_VOPD_TRANS_RESULT_VGPR_FLAG_VALID = 1u << 0,
} loom_amdgpu_vopd_trans_result_vgpr_flag_bits_t;
typedef uint8_t loom_amdgpu_vopd_trans_result_vgpr_flags_t;

typedef struct loom_amdgpu_vopd_trans_result_vgpr_t {
  // State bits for this physical VGPR.
  loom_amdgpu_vopd_trans_result_vgpr_flags_t flags;
  // Number of vector ALU packets since the TRANS result was produced.
  uint8_t valu_interval;
  // Number of TRANS packets since the TRANS result was produced.
  uint8_t trans_interval;
  // Dense active-list position for this physical VGPR while valid.
  uint32_t active_list_index;
} loom_amdgpu_vopd_trans_result_vgpr_t;

typedef struct loom_amdgpu_vopd_plan_builder_t {
  // Schedule table being analyzed.
  const loom_low_schedule_table_t* schedule;
  // Allocation table supplying physical register assignments.
  const loom_low_allocation_table_t* allocation;
  // Processor properties for architecture-specific packetization hazards.
  const loom_amdgpu_processor_properties_t* processor_properties;
  // Optional address-state transitions that block second-component fusion.
  const loom_amdgpu_address_state_plan_t* address_state;
  // Optional planned wait packets that block second-component fusion.
  const loom_amdgpu_wait_packet_plan_t* wait_packets;
  // Arena owning all output and scratch arrays.
  iree_arena_allocator_t* arena;
  // Arena owning analysis scratch discarded after packet-plan construction.
  iree_arena_allocator_t* transient_arena;
  // Optional matrix coexecution consumer of final native packetization.
  loom_amdgpu_matrix_coexecution_t* matrix_coexecution;
  // Descriptor-ordinal-indexed VOPD component rule index + 1 rows.
  const uint8_t* component_rule_lookup;
  // Number of rows in |component_rule_lookup|.
  iree_host_size_t component_rule_lookup_count;
  // Per-scheduled-packet private planner flags.
  loom_amdgpu_vopd_packet_flags_t* packet_flags;
  // True when the target requires VALU/TRANS result-window tracking.
  bool tracks_trans_result_windows;
  // Per-physical-VGPR state for GFX11 TRANS-result packetization windows.
  loom_amdgpu_vopd_trans_result_vgpr_t* trans_result_vgprs;
  // Active physical VGPR indices with valid TRANS-result state.
  uint32_t* active_trans_result_vgpr_indices;
  // Number of entries in |trans_result_vgprs|.
  iree_host_size_t trans_result_vgpr_count;
  // Number of currently active TRANS-result VGPR records.
  iree_host_size_t active_trans_result_vgpr_count;
  // Output VOPD pair records.
  loom_amdgpu_vopd_pair_t* pairs;
  // Number of populated VOPD pair records.
  iree_host_size_t pair_count;
  // Allocated VOPD pair capacity.
  iree_host_size_t pair_capacity;
  // Output rejected adjacent VOPD candidates.
  loom_amdgpu_vopd_rejection_t* rejections;
  // Number of populated VOPD rejection records.
  iree_host_size_t rejection_count;
  // Allocated VOPD rejection capacity.
  iree_host_size_t rejection_capacity;
  // Output per-packet membership records.
  loom_amdgpu_vopd_packet_t* packets;
} loom_amdgpu_vopd_plan_builder_t;

static const loom_amdgpu_vopd_component_info_t*
loom_amdgpu_vopd_component_info_for_index_plus_one(
    uint8_t info_index_plus_one) {
  if (info_index_plus_one == 0) {
    return NULL;
  }
  return &loom_amdgpu_vopd_component_infos[info_index_plus_one - 1];
}

const loom_amdgpu_vopd_component_info_t* loom_amdgpu_vopd_component_info_for_op(
    uint16_t op) {
  if (op >= IREE_ARRAYSIZE(loom_amdgpu_vopd_component_info_indices_by_op)) {
    return NULL;
  }
  return loom_amdgpu_vopd_component_info_for_index_plus_one(
      loom_amdgpu_vopd_component_info_indices_by_op[op]);
}

static const loom_amdgpu_vopd_component_info_t*
loom_amdgpu_vopd_component_info_for_reason(
    loom_amdgpu_vopd_pair_reason_t reason) {
  if (reason >=
      IREE_ARRAYSIZE(
          loom_amdgpu_vopd_component_info_indices_by_same_op_reason)) {
    return NULL;
  }
  return loom_amdgpu_vopd_component_info_for_index_plus_one(
      loom_amdgpu_vopd_component_info_indices_by_same_op_reason[reason]);
}

iree_string_view_t loom_amdgpu_vopd_packet_role_name(
    loom_amdgpu_vopd_packet_role_t role) {
  switch (role) {
    case LOOM_AMDGPU_VOPD_PACKET_ROLE_NONE:
      return IREE_SV("none");
    case LOOM_AMDGPU_VOPD_PACKET_ROLE_FIRST:
      return IREE_SV("first");
    case LOOM_AMDGPU_VOPD_PACKET_ROLE_SECOND:
      return IREE_SV("second");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_amdgpu_vopd_pair_reason_name(
    loom_amdgpu_vopd_pair_reason_t reason) {
  if (reason == LOOM_AMDGPU_VOPD_PAIR_REASON_MIXED_COMPONENTS) {
    return IREE_SV("mixed_components");
  }
  const loom_amdgpu_vopd_component_info_t* info =
      loom_amdgpu_vopd_component_info_for_reason(reason);
  return info != NULL ? info->same_op_reason_name : IREE_SV("unknown");
}

iree_string_view_t loom_amdgpu_vopd_rejection_reason_name(
    loom_amdgpu_vopd_rejection_reason_t reason) {
  switch (reason) {
    case LOOM_AMDGPU_VOPD_REJECTION_REASON_COMPONENT_OPCODE_MISMATCH:
      return IREE_SV("component_opcode_mismatch");
    case LOOM_AMDGPU_VOPD_REJECTION_REASON_FIRST_RESULT_USED_BY_SECOND:
      return IREE_SV("first_result_used_by_second");
    case LOOM_AMDGPU_VOPD_REJECTION_REASON_LITERAL_MISMATCH:
      return IREE_SV("literal_mismatch");
    case LOOM_AMDGPU_VOPD_REJECTION_REASON_REGISTER_CONSTRAINTS:
      return IREE_SV("register_constraints");
    case LOOM_AMDGPU_VOPD_REJECTION_REASON_SECOND_PACKET_HAS_INSERTION:
      return IREE_SV("second_packet_has_insertion");
    case LOOM_AMDGPU_VOPD_REJECTION_REASON_TRANS_RESULT_WINDOW:
      return IREE_SV("trans_result_window");
    case LOOM_AMDGPU_VOPD_REJECTION_REASON_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

typedef struct loom_amdgpu_vopd_register_constraint_flag_name_t {
  // Register constraint represented by this row.
  loom_amdgpu_vopd_register_constraint_flags_t flag;
  // Stable JSON spelling for flag.
  iree_string_view_t name;
} loom_amdgpu_vopd_register_constraint_flag_name_t;

static const loom_amdgpu_vopd_register_constraint_flag_name_t
    kVopdRegisterConstraintFlagNames[] = {
        {LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_DESTINATION_PARITY,
         IREE_SVL("destination_parity")},
        {LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_SRC0_BANK,
         IREE_SVL("src0_bank")},
        {LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_VSRC1_BANK,
         IREE_SVL("vsrc1_bank")},
        {LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_X_DESTINATION_Y_SRC0,
         IREE_SVL("x_destination_y_src0_alias")},
        {LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_X_DESTINATION_Y_VSRC1,
         IREE_SVL("x_destination_y_vsrc1_alias")},
        {LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_Y_DESTINATION_X_SRC0,
         IREE_SVL("y_destination_x_src0_alias")},
        {LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_Y_DESTINATION_X_VSRC1,
         IREE_SVL("y_destination_x_vsrc1_alias")},
};

static iree_string_view_t loom_amdgpu_vopd_op_name(uint16_t op) {
  const loom_amdgpu_vopd_component_info_t* info =
      loom_amdgpu_vopd_component_info_for_op(op);
  return info != NULL ? info->op_name : IREE_SV("unknown");
}

const loom_amdgpu_vopd_packet_t* loom_amdgpu_vopd_plan_packet_at(
    const loom_amdgpu_vopd_plan_t* plan, iree_host_size_t packet_index) {
  if (plan == NULL || packet_index >= plan->packet_count) {
    return NULL;
  }
  const loom_amdgpu_vopd_packet_t* packet = &plan->packets[packet_index];
  return packet->role == LOOM_AMDGPU_VOPD_PACKET_ROLE_NONE ? NULL : packet;
}

static const uint8_t* loom_amdgpu_vopd_component_lookup_for_descriptor_set(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_host_size_t* out_lookup_count) {
  *out_lookup_count = 0;
  if (descriptor_set == NULL) {
    return NULL;
  }
  const uint16_t descriptor_set_ordinal =
      descriptor_set->descriptor_set_ordinal;
  const loom_amdgpu_vopd_component_descriptor_lookup_range_t* range =
      &loom_amdgpu_vopd_component_descriptor_lookup_ranges
          [descriptor_set_ordinal];
  if (range->descriptor_lookup_count == 0) {
    return NULL;
  }
  *out_lookup_count = range->descriptor_lookup_count;
  return &loom_amdgpu_vopd_component_descriptor_lookups
      [range->first_descriptor_lookup];
}

static const loom_amdgpu_vopd_component_rule_t*
loom_amdgpu_vopd_component_rule_for_descriptor_ordinal(
    const uint8_t* descriptor_lookup, iree_host_size_t descriptor_lookup_count,
    uint32_t descriptor_ordinal) {
  if (descriptor_lookup == NULL ||
      descriptor_ordinal >= descriptor_lookup_count) {
    return NULL;
  }
  const uint8_t rule_index_plus_one = descriptor_lookup[descriptor_ordinal];
  if (rule_index_plus_one == 0) {
    return NULL;
  }
  const uint8_t rule_index = (uint8_t)(rule_index_plus_one - 1);
  return &loom_amdgpu_vopd_component_rules[rule_index];
}

static const loom_amdgpu_vopd_pair_affinity_row_t*
loom_amdgpu_vopd_pair_affinities_for_descriptor_set(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_host_size_t* out_pair_affinity_count) {
  *out_pair_affinity_count = 0;
  if (descriptor_set == NULL) {
    return NULL;
  }
  const uint16_t descriptor_set_ordinal =
      descriptor_set->descriptor_set_ordinal;
  const loom_amdgpu_vopd_pair_affinity_range_t* range =
      &loom_amdgpu_vopd_pair_affinity_ranges[descriptor_set_ordinal];
  if (range->pair_affinity_count == 0) {
    return NULL;
  }
  *out_pair_affinity_count = range->pair_affinity_count;
  return &loom_amdgpu_vopd_pair_affinities[range->first_pair_affinity];
}

static bool loom_amdgpu_vopd_descriptor_set_supports_packetization(
    const loom_low_descriptor_set_t* descriptor_set) {
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      descriptor_set != NULL ? loom_amdgpu_target_info_descriptor_set_at(
                                   descriptor_set->descriptor_set_ordinal)
                             : NULL;
  return loom_amdgpu_descriptor_set_info_supports_vopd(descriptor_set_info);
}

static bool loom_amdgpu_vopd_target_supports_base_vopd(
    const loom_low_resolved_target_t* target) {
  const loom_low_descriptor_set_t* descriptor_set =
      target != NULL ? target->descriptor_set : NULL;
  if (descriptor_set == NULL ||
      descriptor_set->target_stable_id != LOOM_AMDGPU_TARGET_STABLE_ID) {
    return false;
  }
  if (loom_low_resolved_target_bundle(target)->snapshot->subgroup_size != 32) {
    return false;
  }
  return loom_amdgpu_vopd_descriptor_set_supports_packetization(descriptor_set);
}

static bool loom_amdgpu_vopd_component_can_use_lane(
    const loom_amdgpu_vopd_component_info_t* info,
    loom_amdgpu_vopd_component_lane_mask_t lane) {
  return info != NULL && iree_all_bits_set(info->lane_mask, lane);
}

static bool loom_amdgpu_vopd_component_can_pair(
    const loom_amdgpu_vopd_component_info_t* info,
    loom_amdgpu_vopd_component_pair_mask_t pairing_mode) {
  return info != NULL && iree_all_bits_set(info->pairing_mask, pairing_mode);
}

static bool loom_amdgpu_vopd_component_infos_pair_reason(
    const loom_amdgpu_vopd_component_info_t* first_info,
    const loom_amdgpu_vopd_component_info_t* second_info,
    loom_amdgpu_vopd_pair_reason_t* out_reason) {
  *out_reason = LOOM_AMDGPU_VOPD_PAIR_REASON_UNKNOWN;
  if (!loom_amdgpu_vopd_component_can_use_lane(
          first_info, LOOM_AMDGPU_VOPD_COMPONENT_LANE_X) ||
      !loom_amdgpu_vopd_component_can_use_lane(
          second_info, LOOM_AMDGPU_VOPD_COMPONENT_LANE_Y)) {
    return false;
  }
  if (first_info->op == second_info->op) {
    if (!loom_amdgpu_vopd_component_can_pair(
            first_info, LOOM_AMDGPU_VOPD_COMPONENT_PAIR_SAME_OPCODE)) {
      return false;
    }
    *out_reason = first_info->same_op_reason;
    return *out_reason != LOOM_AMDGPU_VOPD_PAIR_REASON_UNKNOWN;
  }
  if (!loom_amdgpu_vopd_component_can_pair(
          first_info, LOOM_AMDGPU_VOPD_COMPONENT_PAIR_MIXED_OPCODE) ||
      !loom_amdgpu_vopd_component_can_pair(
          second_info, LOOM_AMDGPU_VOPD_COMPONENT_PAIR_MIXED_OPCODE)) {
    return false;
  }
  *out_reason = LOOM_AMDGPU_VOPD_PAIR_REASON_MIXED_COMPONENTS;
  return true;
}

iree_status_t loom_amdgpu_vopd_build_schedule_pair_affinities(
    const loom_low_resolved_target_t* target, iree_arena_allocator_t* arena,
    loom_low_schedule_pair_affinity_list_t* out_affinities) {
  *out_affinities = loom_low_schedule_pair_affinity_list_empty();
  if (target == NULL || target->descriptor_set == NULL || arena == NULL ||
      !loom_amdgpu_vopd_target_supports_base_vopd(target)) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set = target->descriptor_set;

  iree_host_size_t affinity_count = 0;
  const loom_amdgpu_vopd_pair_affinity_row_t* rows =
      loom_amdgpu_vopd_pair_affinities_for_descriptor_set(descriptor_set,
                                                          &affinity_count);
  if (affinity_count == 0) {
    return iree_ok_status();
  }

  loom_low_schedule_pair_affinity_t* affinities = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, affinity_count, sizeof(*affinities), (void**)&affinities));
  for (iree_host_size_t i = 0; i < affinity_count; ++i) {
    const loom_amdgpu_vopd_pair_affinity_row_t* row = &rows[i];
    const loom_low_descriptor_t* first_descriptor =
        loom_low_descriptor_set_descriptor_at(descriptor_set,
                                              row->first_descriptor_ordinal);
    const loom_low_descriptor_t* second_descriptor =
        loom_low_descriptor_set_descriptor_at(descriptor_set,
                                              row->second_descriptor_ordinal);
    IREE_ASSERT(first_descriptor != NULL);
    IREE_ASSERT(second_descriptor != NULL);
    affinities[i] = (loom_low_schedule_pair_affinity_t){
        .first_descriptor = first_descriptor,
        .second_descriptor = second_descriptor,
        .priority = row->priority,
        .placement_recipe_index = row->placement_recipe_index_plus_one,
    };
  }
  *out_affinities = (loom_low_schedule_pair_affinity_list_t){
      .values = affinities,
      .count = affinity_count,
      .placement_recipes = loom_amdgpu_vopd_pair_placement_recipes,
      .placement_recipe_count = loom_amdgpu_vopd_pair_placement_recipe_count,
  };
  return iree_ok_status();
}

static uint32_t loom_amdgpu_vopd_packet_index_for_insertion(
    const loom_low_schedule_table_t* schedule, uint32_t block_index,
    uint32_t scheduled_ordinal) {
  IREE_ASSERT_LT(block_index, schedule->block_count);
  const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
  const uint32_t packet_index = block->scheduled_node_start + scheduled_ordinal;
  IREE_ASSERT_LT(packet_index, schedule->scheduled_node_count);
  return packet_index;
}

static void loom_amdgpu_vopd_mark_address_state_insertions(
    loom_amdgpu_vopd_plan_builder_t* builder) {
  if (builder->address_state == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < builder->address_state->transition_count;
       ++i) {
    const loom_amdgpu_address_state_transition_t* transition =
        &builder->address_state->transitions[i];
    const uint32_t packet_index = loom_amdgpu_vopd_packet_index_for_insertion(
        builder->schedule, transition->block_index,
        transition->scheduled_ordinal);
    builder->packet_flags[packet_index] |=
        LOOM_AMDGPU_VOPD_PACKET_FLAG_INSERTION_BLOCKED;
  }
}

static void loom_amdgpu_vopd_mark_wait_packet_insertions(
    loom_amdgpu_vopd_plan_builder_t* builder) {
  if (builder->wait_packets == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < builder->wait_packets->packet_count; ++i) {
    const loom_amdgpu_wait_packet_t* wait_packet =
        &builder->wait_packets->packets[i];
    const uint32_t packet_index = loom_amdgpu_vopd_packet_index_for_insertion(
        builder->schedule, wait_packet->block_index,
        wait_packet->scheduled_ordinal);
    builder->packet_flags[packet_index] |=
        LOOM_AMDGPU_VOPD_PACKET_FLAG_INSERTION_BLOCKED;
    if (iree_any_bit_set(wait_packet->counter_mask,
                         LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU)) {
      builder->packet_flags[packet_index] |=
          LOOM_AMDGPU_VOPD_PACKET_FLAG_TRANS_RESULT_CLEAR;
    }
  }
}

static loom_amdgpu_vopd_visible_packet_t loom_amdgpu_vopd_classify_packet(
    loom_amdgpu_vopd_plan_builder_t* builder, iree_host_size_t packet_index) {
  loom_amdgpu_vopd_visible_packet_t visible = {
      .packet = loom_low_packet_at(builder->schedule, packet_index),
  };
  loom_amdgpu_vopd_packet_flags_t* packet_flags =
      &builder->packet_flags[packet_index];
  if (visible.packet.descriptor == NULL) {
    visible.structural = loom_amdgpu_structural_packet_analyze(
        builder->schedule, builder->allocation, visible.packet.node, 0);
    if (!iree_any_bit_set(*packet_flags,
                          LOOM_AMDGPU_VOPD_PACKET_FLAG_INSERTION_BLOCKED) &&
        iree_any_bit_set(
            visible.structural.flags,
            LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_FORWARDS_DEPENDENCIES)) {
      *packet_flags |= LOOM_AMDGPU_VOPD_PACKET_FLAG_TRANSPARENT;
    }
  }
  return visible;
}

static iree_status_t loom_amdgpu_vopd_plan_initialize_trans_result_guard(
    loom_amdgpu_vopd_plan_builder_t* builder) {
  if (builder->trans_result_vgprs != NULL) {
    return iree_ok_status();
  }
  const iree_host_size_t vgpr_count =
      builder->allocation->physical_extents
          .ends_by_reg_class[LOOM_AMDGPU_REG_CLASS_ID_VGPR];
  if (vgpr_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(builder->transient_arena, vgpr_count,
                                sizeof(*builder->trans_result_vgprs),
                                (void**)&builder->trans_result_vgprs));
  memset(builder->trans_result_vgprs, 0,
         vgpr_count * sizeof(*builder->trans_result_vgprs));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->transient_arena, vgpr_count,
      sizeof(*builder->active_trans_result_vgpr_indices),
      (void**)&builder->active_trans_result_vgpr_indices));
  builder->trans_result_vgpr_count = vgpr_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_plan_allocate(
    loom_amdgpu_vopd_plan_builder_t* builder) {
  const iree_host_size_t packet_count = builder->schedule->scheduled_node_count;
  builder->pair_capacity = packet_count / 2;
  builder->rejection_capacity = packet_count == 0 ? 0 : packet_count - 1;
  if (builder->pair_capacity > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU VOPD pair capacity exceeds 32-bit index "
                            "range");
  }
  if (packet_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(builder->arena, packet_count,
                                                   sizeof(*builder->packets),
                                                   (void**)&builder->packets));
    for (iree_host_size_t i = 0; i < packet_count; ++i) {
      builder->packets[i] = (loom_amdgpu_vopd_packet_t){
          .role = LOOM_AMDGPU_VOPD_PACKET_ROLE_NONE,
          .pair_index = LOOM_AMDGPU_VOPD_PAIR_NONE,
      };
    }
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->transient_arena, packet_count, sizeof(*builder->packet_flags),
        (void**)&builder->packet_flags));
    memset(builder->packet_flags, 0,
           packet_count * sizeof(*builder->packet_flags));
  }
  if (builder->pair_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, builder->pair_capacity, sizeof(*builder->pairs),
        (void**)&builder->pairs));
  }
  if (builder->rejection_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, builder->rejection_capacity,
        sizeof(*builder->rejections), (void**)&builder->rejections));
  }
  loom_amdgpu_vopd_mark_address_state_insertions(builder);
  loom_amdgpu_vopd_mark_wait_packet_insertions(builder);
  return iree_ok_status();
}

static const loom_low_allocation_assignment_t* loom_amdgpu_vopd_map_assignment(
    const loom_low_allocation_table_t* allocation, loom_value_id_t value_id) {
  return loom_low_allocation_try_map_active_value_assignment(allocation,
                                                             value_id, NULL);
}

static bool loom_amdgpu_vopd_assignment_single_physical_vgpr(
    const loom_low_allocation_assignment_t* assignment,
    uint16_t* out_register) {
  *out_register = 0;
  if (assignment == NULL ||
      assignment->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
      assignment->descriptor_reg_class_id != LOOM_AMDGPU_REG_CLASS_ID_VGPR ||
      assignment->location_count != 1 || assignment->location_base > 255) {
    return false;
  }
  *out_register = (uint16_t)assignment->location_base;
  return true;
}

static bool loom_amdgpu_vopd_assignments_match(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  return lhs != NULL && rhs != NULL &&
         lhs->location_kind == rhs->location_kind &&
         lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id &&
         lhs->location_base == rhs->location_base &&
         lhs->location_count == rhs->location_count;
}

static bool loom_amdgpu_vopd_assignment_is_physical_vgpr(
    const loom_low_allocation_assignment_t* assignment) {
  return assignment != NULL &&
         assignment->location_kind ==
             LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER &&
         assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR;
}

static bool loom_amdgpu_vopd_has_trans_result_guard(
    const loom_amdgpu_vopd_plan_builder_t* builder) {
  return builder->trans_result_vgprs != NULL &&
         builder->active_trans_result_vgpr_indices != NULL &&
         builder->trans_result_vgpr_count != 0;
}

static bool loom_amdgpu_vopd_trans_result_vgpr_is_tracked(
    const loom_amdgpu_vopd_trans_result_vgpr_t* vgpr) {
  return iree_any_bit_set(vgpr->flags,
                          LOOM_AMDGPU_VOPD_TRANS_RESULT_VGPR_FLAG_VALID);
}

static bool loom_amdgpu_vopd_trans_result_vgpr_is_active(
    const loom_amdgpu_vopd_trans_result_vgpr_t* vgpr) {
  return loom_amdgpu_vopd_trans_result_vgpr_is_tracked(vgpr) &&
         vgpr->valu_interval <=
             LOOM_AMDGPU_VALU_TRANS_USE_DEPCTR_MAX_VALU_INTERVAL &&
         vgpr->trans_interval <=
             LOOM_AMDGPU_VALU_TRANS_USE_DEPCTR_MAX_TRANS_INTERVAL;
}

static bool loom_amdgpu_vopd_has_active_trans_result_window(
    const loom_amdgpu_vopd_plan_builder_t* builder) {
  return loom_amdgpu_vopd_has_trans_result_guard(builder) &&
         builder->active_trans_result_vgpr_count != 0;
}

static void loom_amdgpu_vopd_clear_trans_result_windows(
    loom_amdgpu_vopd_plan_builder_t* builder) {
  if (!loom_amdgpu_vopd_has_active_trans_result_window(builder)) {
    return;
  }
  while (builder->active_trans_result_vgpr_count != 0) {
    const uint32_t active_list_index =
        (uint32_t)(builder->active_trans_result_vgpr_count - 1);
    const uint32_t vgpr_index =
        builder->active_trans_result_vgpr_indices[active_list_index];
    builder->trans_result_vgprs[vgpr_index] =
        (loom_amdgpu_vopd_trans_result_vgpr_t){0};
    --builder->active_trans_result_vgpr_count;
  }
}

static uint8_t loom_amdgpu_vopd_saturated_increment(uint8_t value,
                                                    uint8_t limit) {
  return value <= limit ? (uint8_t)(value + 1u) : value;
}

static void loom_amdgpu_vopd_remove_active_trans_result_vgpr(
    loom_amdgpu_vopd_plan_builder_t* builder, uint32_t vgpr_index) {
  IREE_ASSERT_LT(vgpr_index, builder->trans_result_vgpr_count);
  loom_amdgpu_vopd_trans_result_vgpr_t* vgpr =
      &builder->trans_result_vgprs[vgpr_index];
  if (!loom_amdgpu_vopd_trans_result_vgpr_is_tracked(vgpr)) {
    *vgpr = (loom_amdgpu_vopd_trans_result_vgpr_t){0};
    return;
  }
  IREE_ASSERT_NE(builder->active_trans_result_vgpr_count, 0);
  const uint32_t removed_list_index = vgpr->active_list_index;
  const uint32_t last_list_index =
      (uint32_t)(builder->active_trans_result_vgpr_count - 1);
  IREE_ASSERT_LT(removed_list_index, builder->active_trans_result_vgpr_count);
  if (removed_list_index != last_list_index) {
    const uint32_t moved_vgpr_index =
        builder->active_trans_result_vgpr_indices[last_list_index];
    builder->active_trans_result_vgpr_indices[removed_list_index] =
        moved_vgpr_index;
    builder->trans_result_vgprs[moved_vgpr_index].active_list_index =
        removed_list_index;
  }
  --builder->active_trans_result_vgpr_count;
  *vgpr = (loom_amdgpu_vopd_trans_result_vgpr_t){0};
}

static void loom_amdgpu_vopd_activate_trans_result_vgpr(
    loom_amdgpu_vopd_plan_builder_t* builder, uint32_t vgpr_index) {
  IREE_ASSERT_LT(vgpr_index, builder->trans_result_vgpr_count);
  loom_amdgpu_vopd_trans_result_vgpr_t* vgpr =
      &builder->trans_result_vgprs[vgpr_index];
  if (loom_amdgpu_vopd_trans_result_vgpr_is_tracked(vgpr)) {
    return;
  }
  IREE_ASSERT_LT(builder->active_trans_result_vgpr_count,
                 builder->trans_result_vgpr_count);
  const uint32_t active_list_index =
      (uint32_t)builder->active_trans_result_vgpr_count++;
  builder->active_trans_result_vgpr_indices[active_list_index] = vgpr_index;
  vgpr->active_list_index = active_list_index;
}

static void loom_amdgpu_vopd_increment_trans_result_windows(
    loom_amdgpu_vopd_plan_builder_t* builder, bool is_vector_alu,
    bool is_transcendental) {
  if (!loom_amdgpu_vopd_has_active_trans_result_window(builder) ||
      (!is_vector_alu && !is_transcendental)) {
    return;
  }
  for (iree_host_size_t i = 0; i < builder->active_trans_result_vgpr_count;) {
    const uint32_t vgpr_index = builder->active_trans_result_vgpr_indices[i];
    loom_amdgpu_vopd_trans_result_vgpr_t* vgpr =
        &builder->trans_result_vgprs[vgpr_index];
    IREE_ASSERT(loom_amdgpu_vopd_trans_result_vgpr_is_tracked(vgpr));
    if (is_vector_alu) {
      vgpr->valu_interval = loom_amdgpu_vopd_saturated_increment(
          vgpr->valu_interval,
          LOOM_AMDGPU_VALU_TRANS_USE_DEPCTR_MAX_VALU_INTERVAL);
    }
    if (is_transcendental) {
      vgpr->trans_interval = loom_amdgpu_vopd_saturated_increment(
          vgpr->trans_interval,
          LOOM_AMDGPU_VALU_TRANS_USE_DEPCTR_MAX_TRANS_INTERVAL);
    }
    if (!loom_amdgpu_vopd_trans_result_vgpr_is_active(vgpr)) {
      loom_amdgpu_vopd_remove_active_trans_result_vgpr(builder, vgpr_index);
      continue;
    }
    ++i;
  }
}

static void loom_amdgpu_vopd_clear_trans_result_assignment(
    loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment) {
  if (!loom_amdgpu_vopd_has_trans_result_guard(builder) ||
      !loom_amdgpu_vopd_assignment_is_physical_vgpr(assignment)) {
    return;
  }
  const uint64_t end =
      (uint64_t)assignment->location_base + assignment->location_count;
  if (end > builder->trans_result_vgpr_count) {
    return;
  }
  for (uint32_t i = 0; i < assignment->location_count; ++i) {
    loom_amdgpu_vopd_remove_active_trans_result_vgpr(
        builder, assignment->location_base + i);
  }
}

static void loom_amdgpu_vopd_record_trans_result_assignment(
    loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment) {
  if (!loom_amdgpu_vopd_has_trans_result_guard(builder) ||
      !loom_amdgpu_vopd_assignment_is_physical_vgpr(assignment)) {
    return;
  }
  const uint64_t end =
      (uint64_t)assignment->location_base + assignment->location_count;
  if (end > builder->trans_result_vgpr_count) {
    return;
  }
  for (uint32_t i = 0; i < assignment->location_count; ++i) {
    const uint32_t vgpr_index = assignment->location_base + i;
    loom_amdgpu_vopd_activate_trans_result_vgpr(builder, vgpr_index);
    loom_amdgpu_vopd_trans_result_vgpr_t* vgpr =
        &builder->trans_result_vgprs[vgpr_index];
    const uint32_t active_list_index = vgpr->active_list_index;
    *vgpr = (loom_amdgpu_vopd_trans_result_vgpr_t){
        .flags = LOOM_AMDGPU_VOPD_TRANS_RESULT_VGPR_FLAG_VALID,
        .active_list_index = active_list_index,
    };
  }
}

static void loom_amdgpu_vopd_clear_trans_result_packet_results(
    loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* packet) {
  if (!loom_amdgpu_vopd_has_trans_result_guard(builder) ||
      packet->node->op == NULL) {
    return;
  }
  const loom_op_t* op = packet->node->op;
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_amdgpu_vopd_map_assignment(builder->allocation, results[i]);
    loom_amdgpu_vopd_clear_trans_result_assignment(builder, assignment);
  }
}

static void loom_amdgpu_vopd_record_trans_result_packet_results(
    loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* packet) {
  if (!loom_amdgpu_vopd_has_trans_result_guard(builder) ||
      packet->node->op == NULL) {
    return;
  }
  const loom_op_t* op = packet->node->op;
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_amdgpu_vopd_map_assignment(builder->allocation, results[i]);
    loom_amdgpu_vopd_record_trans_result_assignment(builder, assignment);
  }
}

static void loom_amdgpu_vopd_apply_trans_result_insertion(
    loom_amdgpu_vopd_plan_builder_t* builder, iree_host_size_t packet_index) {
  if (!loom_amdgpu_vopd_has_trans_result_guard(builder) ||
      builder->packet_flags == NULL ||
      !iree_any_bit_set(builder->packet_flags[packet_index],
                        LOOM_AMDGPU_VOPD_PACKET_FLAG_TRANS_RESULT_CLEAR)) {
    return;
  }
  loom_amdgpu_vopd_clear_trans_result_windows(builder);
}

static iree_status_t loom_amdgpu_vopd_advance_trans_result_packet(
    loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* packet) {
  if (!builder->tracks_trans_result_windows) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      builder->schedule->target.descriptor_set;
  const loom_amdgpu_descriptor_traits_t descriptor_traits =
      loom_amdgpu_descriptor_traits(descriptor_set, packet->descriptor);
  const bool is_transcendental = iree_any_bit_set(
      descriptor_traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_TRANSCENDENTAL);
  if (is_transcendental && !loom_amdgpu_vopd_has_trans_result_guard(builder)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_vopd_plan_initialize_trans_result_guard(builder));
  }
  if (!loom_amdgpu_vopd_has_trans_result_guard(builder)) {
    return iree_ok_status();
  }
  if (iree_any_bit_set(descriptor_traits,
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY)) {
    loom_amdgpu_vopd_clear_trans_result_windows(builder);
  }
  loom_amdgpu_vopd_increment_trans_result_windows(
      builder,
      iree_any_bit_set(descriptor_traits,
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU),
      is_transcendental);
  loom_amdgpu_vopd_clear_trans_result_packet_results(builder, packet);
  if (is_transcendental) {
    loom_amdgpu_vopd_record_trans_result_packet_results(builder, packet);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_advance_trans_result_pair(
    loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* first, const loom_low_packet_view_t* second) {
  if (!builder->tracks_trans_result_windows) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      builder->schedule->target.descriptor_set;
  const loom_amdgpu_descriptor_traits_t first_traits =
      loom_amdgpu_descriptor_traits(descriptor_set, first->descriptor);
  const loom_amdgpu_descriptor_traits_t second_traits =
      loom_amdgpu_descriptor_traits(descriptor_set, second->descriptor);
  const bool first_trans = iree_any_bit_set(
      first_traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_TRANSCENDENTAL);
  const bool second_trans = iree_any_bit_set(
      second_traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_TRANSCENDENTAL);
  if ((first_trans || second_trans) &&
      !loom_amdgpu_vopd_has_trans_result_guard(builder)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_vopd_plan_initialize_trans_result_guard(builder));
  }
  if (!loom_amdgpu_vopd_has_trans_result_guard(builder)) {
    return iree_ok_status();
  }
  loom_amdgpu_vopd_increment_trans_result_windows(
      builder,
      iree_any_bit_set(first_traits | second_traits,
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU),
      first_trans || second_trans);
  loom_amdgpu_vopd_clear_trans_result_packet_results(builder, first);
  loom_amdgpu_vopd_clear_trans_result_packet_results(builder, second);
  if (first_trans) {
    loom_amdgpu_vopd_record_trans_result_packet_results(builder, first);
  }
  if (second_trans) {
    loom_amdgpu_vopd_record_trans_result_packet_results(builder, second);
  }
  return iree_ok_status();
}

static bool loom_amdgpu_vopd_component_result_is_used_by(
    const loom_low_packet_view_t* producer,
    const loom_low_packet_view_t* consumer) {
  const loom_op_t* producer_op = producer->node->op;
  const loom_op_t* consumer_op = consumer->node->op;
  const loom_value_id_t* results = loom_op_const_results(producer_op);
  const loom_value_id_t* operands = loom_op_const_operands(consumer_op);
  for (iree_host_size_t result_index = 0;
       result_index < producer_op->result_count; ++result_index) {
    for (iree_host_size_t operand_index = 0;
         operand_index < consumer_op->operand_count; ++operand_index) {
      if (results[result_index] == operands[operand_index]) {
        return true;
      }
    }
  }
  return false;
}

static const loom_named_attr_t* loom_amdgpu_vopd_find_packet_attr(
    const loom_low_packet_view_t* packet, loom_string_id_t name_id) {
  if (name_id == LOOM_STRING_ID_INVALID) {
    return NULL;
  }
  loom_named_attr_slice_t attrs = loom_low_packet_attrs(packet);
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (attr->name_id == name_id) {
      return attr;
    }
  }
  return NULL;
}

static uint32_t loom_amdgpu_vopd_read_immediate_u32(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* packet, uint16_t descriptor_immediate_index) {
  IREE_ASSERT_LT(descriptor_immediate_index,
                 packet->descriptor->immediate_count);
  const loom_low_descriptor_set_t* descriptor_set =
      builder->schedule->target.descriptor_set;
  const uint32_t immediate_row =
      packet->descriptor->immediate_start + descriptor_immediate_index;
  IREE_ASSERT_LT(immediate_row, descriptor_set->immediate_count);
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[immediate_row];
  iree_string_view_t immediate_name = loom_low_descriptor_set_string(
      descriptor_set, immediate->field_name_string_offset);
  const loom_string_id_t immediate_name_id =
      loom_module_lookup_string(builder->schedule->module, immediate_name);
  const loom_named_attr_t* attr =
      loom_amdgpu_vopd_find_packet_attr(packet, immediate_name_id);
  IREE_ASSERT(attr != NULL);
  IREE_ASSERT_EQ(attr->value.kind, LOOM_ATTR_I64);
  const int64_t value = attr->value.i64;
  IREE_ASSERT(value >= 0 && value <= UINT32_MAX);
  return (uint32_t)value;
}

static bool loom_amdgpu_vopd_bank_compatible(uint16_t x_register,
                                             uint16_t y_register,
                                             uint16_t bank_mask) {
  return (x_register & bank_mask) != (y_register & bank_mask);
}

static bool loom_amdgpu_vopd_components_share_source_cache(
    const loom_amdgpu_vopd_candidate_component_t* x,
    const loom_amdgpu_vopd_candidate_component_t* y,
    loom_amdgpu_vopd_component_source_mask_t source_mask) {
  if (source_mask != LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_SRC0 ||
      x->rule->info->op != y->rule->info->op) {
    return true;
  }
  const loom_amdgpu_vopd_component_rule_flags_t shared_flags =
      x->rule->flags & y->rule->flags;
  return !iree_any_bit_set(shared_flags,
                           LOOM_AMDGPU_VOPD_COMPONENT_FLAG_DUAL_MOV_SRC2_CACHE);
}

static loom_amdgpu_vopd_register_constraint_flags_t
loom_amdgpu_vopd_register_constraint_flags(
    const loom_amdgpu_vopd_candidate_pair_t* candidate) {
  loom_amdgpu_vopd_register_constraint_flags_t flags = 0;
  if (((candidate->x.vdst ^ candidate->y.vdst) & 1u) == 0) {
    flags |= LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_DESTINATION_PARITY;
  }
  const loom_amdgpu_vopd_component_source_mask_t paired_source_registers =
      candidate->x.rule->source_register_mask &
      candidate->y.rule->source_register_mask;
  if (iree_any_bit_set(paired_source_registers,
                       LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_SRC0) &&
      loom_amdgpu_vopd_components_share_source_cache(
          &candidate->x, &candidate->y,
          LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_SRC0) &&
      !loom_amdgpu_vopd_bank_compatible(candidate->x.src0, candidate->y.src0,
                                        3)) {
    flags |= LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_SRC0_BANK;
  }
  if (iree_any_bit_set(paired_source_registers,
                       LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_VSRC1) &&
      !loom_amdgpu_vopd_bank_compatible(candidate->x.vsrc1, candidate->y.vsrc1,
                                        3)) {
    flags |= LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_VSRC1_BANK;
  }
  if (iree_any_bit_set(candidate->y.rule->source_register_mask,
                       LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_SRC0) &&
      candidate->x.vdst == candidate->y.src0) {
    flags |= LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_X_DESTINATION_Y_SRC0;
  }
  if (iree_any_bit_set(candidate->y.rule->source_register_mask,
                       LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_VSRC1) &&
      candidate->x.vdst == candidate->y.vsrc1) {
    flags |= LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_X_DESTINATION_Y_VSRC1;
  }
  if (iree_any_bit_set(candidate->x.rule->source_register_mask,
                       LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_SRC0) &&
      candidate->y.vdst == candidate->x.src0) {
    flags |= LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_Y_DESTINATION_X_SRC0;
  }
  if (iree_any_bit_set(candidate->x.rule->source_register_mask,
                       LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_VSRC1) &&
      candidate->y.vdst == candidate->x.vsrc1) {
    flags |= LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_Y_DESTINATION_X_VSRC1;
  }
  return flags;
}

static void loom_amdgpu_vopd_swap_component_sources(
    loom_amdgpu_vopd_candidate_component_t* component) {
  IREE_ASSERT(iree_all_bits_set(component->rule->source_register_mask,
                                LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY));
  const uint16_t src0 = component->src0;
  component->src0 = component->vsrc1;
  component->vsrc1 = src0;
}

static bool loom_amdgpu_vopd_try_register_orientations(
    loom_amdgpu_vopd_candidate_pair_t* candidate,
    loom_amdgpu_vopd_register_constraint_flags_t* out_constraint_flags) {
  *out_constraint_flags = loom_amdgpu_vopd_register_constraint_flags(candidate);
  if (*out_constraint_flags == 0) {
    return true;
  }

  const bool can_swap_x =
      iree_any_bit_set(candidate->x.rule->flags,
                       LOOM_AMDGPU_VOPD_COMPONENT_FLAG_COMMUTABLE_SOURCES);
  const bool can_swap_y =
      iree_any_bit_set(candidate->y.rule->flags,
                       LOOM_AMDGPU_VOPD_COMPONENT_FLAG_COMMUTABLE_SOURCES);
  if (can_swap_x) {
    loom_amdgpu_vopd_swap_component_sources(&candidate->x);
    if (loom_amdgpu_vopd_register_constraint_flags(candidate) == 0) {
      candidate->flags |= LOOM_AMDGPU_VOPD_PAIR_FLAG_X_SOURCES_SWAPPED;
      return true;
    }
    loom_amdgpu_vopd_swap_component_sources(&candidate->x);
  }
  if (can_swap_y) {
    loom_amdgpu_vopd_swap_component_sources(&candidate->y);
    if (loom_amdgpu_vopd_register_constraint_flags(candidate) == 0) {
      candidate->flags |= LOOM_AMDGPU_VOPD_PAIR_FLAG_Y_SOURCES_SWAPPED;
      return true;
    }
    loom_amdgpu_vopd_swap_component_sources(&candidate->y);
  }
  if (can_swap_x && can_swap_y) {
    loom_amdgpu_vopd_swap_component_sources(&candidate->x);
    loom_amdgpu_vopd_swap_component_sources(&candidate->y);
    if (loom_amdgpu_vopd_register_constraint_flags(candidate) == 0) {
      candidate->flags |= LOOM_AMDGPU_VOPD_PAIR_FLAG_X_SOURCES_SWAPPED |
                          LOOM_AMDGPU_VOPD_PAIR_FLAG_Y_SOURCES_SWAPPED;
      return true;
    }
    loom_amdgpu_vopd_swap_component_sources(&candidate->x);
    loom_amdgpu_vopd_swap_component_sources(&candidate->y);
  }
  return false;
}

static bool loom_amdgpu_vopd_inline_u32_source(
    const loom_amdgpu_vopd_plan_builder_t* builder, uint32_t value,
    uint16_t* out_source_selector) {
  const loom_low_descriptor_set_t* descriptor_set =
      builder->schedule->target.descriptor_set;
  const loom_amdgpu_encoding_table_t* encoding_table =
      loom_amdgpu_encoding_table_for_descriptor_set_ordinal(
          descriptor_set->descriptor_set_ordinal);
  return loom_amdgpu_encoding_inline_u32_source(encoding_table, value,
                                                out_source_selector);
}

static bool loom_amdgpu_vopd_literal_immediate_index(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* packet, uint16_t* out_immediate_index) {
  const loom_amdgpu_descriptor_immediate_slots_t immediate_slots =
      loom_amdgpu_descriptor_immediate_slots(
          builder->schedule->target.descriptor_set, packet->descriptor);
  if (immediate_slots.literal == LOOM_LOW_ID_NONE) {
    return false;
  }
  *out_immediate_index = immediate_slots.literal;
  return true;
}

static bool loom_amdgpu_vopd_read_tied_accumulate_component(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_vopd_component_rule_t* rule,
    loom_amdgpu_vopd_candidate_component_t* out_component) {
  *out_component = (loom_amdgpu_vopd_candidate_component_t){0};

  const loom_op_t* op = packet->node->op;
  if (op->result_count != 1 || packet->descriptor->immediate_count != 0 ||
      rule->operands.accumulator_index >= op->operand_count ||
      rule->operands.src0_index >= op->operand_count ||
      rule->operands.vsrc1_index >= op->operand_count) {
    return false;
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_low_allocation_assignment_t* result_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, results[0]);
  const loom_low_allocation_assignment_t* accumulator_assignment =
      loom_amdgpu_vopd_map_assignment(
          builder->allocation, operands[rule->operands.accumulator_index]);
  const loom_low_allocation_assignment_t* src0_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation,
                                      operands[rule->operands.src0_index]);
  const loom_low_allocation_assignment_t* vsrc1_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation,
                                      operands[rule->operands.vsrc1_index]);
  if (!loom_amdgpu_vopd_assignments_match(result_assignment,
                                          accumulator_assignment)) {
    return false;
  }
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(result_assignment,
                                                        &out_component->vdst)) {
    return false;
  }
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(src0_assignment,
                                                        &out_component->src0)) {
    return false;
  }
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(
          vsrc1_assignment, &out_component->vsrc1)) {
    return false;
  }
  return true;
}

static bool loom_amdgpu_vopd_read_literal_fma_component(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_vopd_candidate_component_t* out_component) {
  *out_component = (loom_amdgpu_vopd_candidate_component_t){
      .flags = LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL,
  };

  const loom_op_t* op = packet->node->op;
  uint16_t literal_immediate_index = LOOM_LOW_ID_NONE;
  if (op->result_count != 1 || op->operand_count != 2 ||
      !loom_amdgpu_vopd_literal_immediate_index(builder, packet,
                                                &literal_immediate_index)) {
    return false;
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_low_allocation_assignment_t* result_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, results[0]);
  const loom_low_allocation_assignment_t* src0_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, operands[0]);
  const loom_low_allocation_assignment_t* vsrc1_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, operands[1]);
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(result_assignment,
                                                        &out_component->vdst)) {
    return false;
  }
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(src0_assignment,
                                                        &out_component->src0)) {
    return false;
  }
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(
          vsrc1_assignment, &out_component->vsrc1)) {
    return false;
  }
  out_component->literal_u32 = loom_amdgpu_vopd_read_immediate_u32(
      builder, packet, literal_immediate_index);
  return true;
}

static bool loom_amdgpu_vopd_read_binary_vgpr_component_with_operand_count(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* packet,
    iree_host_size_t expected_operand_count,
    loom_amdgpu_vopd_candidate_component_t* out_component) {
  *out_component = (loom_amdgpu_vopd_candidate_component_t){0};

  const loom_op_t* op = packet->node->op;
  if (op->result_count != 1 || op->operand_count != expected_operand_count ||
      packet->descriptor->immediate_count != 0) {
    return false;
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_low_allocation_assignment_t* result_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, results[0]);
  const loom_low_allocation_assignment_t* src0_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, operands[0]);
  const loom_low_allocation_assignment_t* vsrc1_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, operands[1]);
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(result_assignment,
                                                        &out_component->vdst)) {
    return false;
  }
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(src0_assignment,
                                                        &out_component->src0)) {
    return false;
  }
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(
          vsrc1_assignment, &out_component->vsrc1)) {
    return false;
  }
  return true;
}

static bool loom_amdgpu_vopd_read_binary_vgpr_component(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_vopd_candidate_component_t* out_component) {
  return loom_amdgpu_vopd_read_binary_vgpr_component_with_operand_count(
      builder, packet, 2, out_component);
}

static bool loom_amdgpu_vopd_read_cndmask_vcc_component(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_vopd_candidate_component_t* out_component) {
  return loom_amdgpu_vopd_read_binary_vgpr_component_with_operand_count(
      builder, packet, 3, out_component);
}

static bool loom_amdgpu_vopd_read_inline_mov_component(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_vopd_candidate_component_t* out_component) {
  *out_component = (loom_amdgpu_vopd_candidate_component_t){0};

  const loom_op_t* op = packet->node->op;
  uint16_t literal_immediate_index = LOOM_LOW_ID_NONE;
  if (op->result_count != 1 || op->operand_count != 0 ||
      !loom_amdgpu_vopd_literal_immediate_index(builder, packet,
                                                &literal_immediate_index)) {
    return false;
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  const loom_low_allocation_assignment_t* result_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, results[0]);
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(result_assignment,
                                                        &out_component->vdst)) {
    return false;
  }
  out_component->literal_u32 = loom_amdgpu_vopd_read_immediate_u32(
      builder, packet, literal_immediate_index);
  if (!loom_amdgpu_vopd_inline_u32_source(builder, out_component->literal_u32,
                                          &out_component->src0_selector)) {
    return false;
  }
  return true;
}

static bool loom_amdgpu_vopd_read_register_mov_component(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_vopd_candidate_component_t* out_component) {
  *out_component = (loom_amdgpu_vopd_candidate_component_t){0};

  const loom_op_t* op = packet->node->op;
  if (op->result_count != 1 || op->operand_count != 1 ||
      packet->descriptor->immediate_count != 0) {
    return false;
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_low_allocation_assignment_t* result_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, results[0]);
  const loom_low_allocation_assignment_t* source_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, operands[0]);
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(result_assignment,
                                                        &out_component->vdst)) {
    return false;
  }
  return loom_amdgpu_vopd_assignment_single_physical_vgpr(source_assignment,
                                                          &out_component->src0);
}

static bool loom_amdgpu_vopd_read_component(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_vopd_candidate_component_t* out_component) {
  *out_component = (loom_amdgpu_vopd_candidate_component_t){0};

  const loom_low_descriptor_set_t* descriptor_set =
      builder->schedule->target.descriptor_set;
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(descriptor_set,
                                                 packet->descriptor);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return false;
  }
  const loom_amdgpu_vopd_component_rule_t* rule =
      loom_amdgpu_vopd_component_rule_for_descriptor_ordinal(
          builder->component_rule_lookup, builder->component_rule_lookup_count,
          descriptor_ordinal);
  if (rule == NULL) {
    return false;
  }
  bool eligible = false;
  switch (rule->form) {
    case LOOM_AMDGPU_VOPD_COMPONENT_FORM_TIED_ACCUMULATE:
      eligible = loom_amdgpu_vopd_read_tied_accumulate_component(
          builder, packet, rule, out_component);
      break;
    case LOOM_AMDGPU_VOPD_COMPONENT_FORM_FMAAK_LITERAL:
    case LOOM_AMDGPU_VOPD_COMPONENT_FORM_FMAMK_LITERAL:
      eligible = loom_amdgpu_vopd_read_literal_fma_component(builder, packet,
                                                             out_component);
      break;
    case LOOM_AMDGPU_VOPD_COMPONENT_FORM_BINARY_VGPR:
      eligible = loom_amdgpu_vopd_read_binary_vgpr_component(builder, packet,
                                                             out_component);
      break;
    case LOOM_AMDGPU_VOPD_COMPONENT_FORM_INLINE_MOV:
      eligible = loom_amdgpu_vopd_read_inline_mov_component(builder, packet,
                                                            out_component);
      break;
    case LOOM_AMDGPU_VOPD_COMPONENT_FORM_REGISTER_MOV:
      eligible = loom_amdgpu_vopd_read_register_mov_component(builder, packet,
                                                              out_component);
      break;
    case LOOM_AMDGPU_VOPD_COMPONENT_FORM_CNDMASK_VCC:
      eligible = loom_amdgpu_vopd_read_cndmask_vcc_component(builder, packet,
                                                             out_component);
      break;
    default:
      IREE_ASSERT_UNREACHABLE("AMDGPU VOPD component rule has unknown form");
      return false;
  }
  if (!eligible) {
    return false;
  }
  out_component->rule = rule;
  return true;
}

static bool loom_amdgpu_vopd_pair_reason_for_components(
    const loom_amdgpu_vopd_candidate_component_t* first,
    const loom_amdgpu_vopd_candidate_component_t* second,
    loom_amdgpu_vopd_pair_reason_t* out_reason) {
  return loom_amdgpu_vopd_component_infos_pair_reason(
      first->rule->info, second->rule->info, out_reason);
}

static bool loom_amdgpu_vopd_resolve_pair_literal(
    const loom_amdgpu_vopd_candidate_component_t* first,
    const loom_amdgpu_vopd_candidate_component_t* second,
    loom_amdgpu_vopd_candidate_pair_t* candidate) {
  const bool first_has_literal =
      iree_any_bit_set(first->flags, LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL);
  const bool second_has_literal =
      iree_any_bit_set(second->flags, LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL);
  if (first_has_literal && second_has_literal &&
      first->literal_u32 != second->literal_u32) {
    return false;
  }
  candidate->flags = first->flags | second->flags;
  if (first_has_literal) {
    candidate->literal_u32 = first->literal_u32;
  } else if (second_has_literal) {
    candidate->literal_u32 = second->literal_u32;
  }
  return true;
}

static void loom_amdgpu_vopd_analyze_pair(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* first, const loom_low_packet_view_t* second,
    loom_amdgpu_vopd_pair_analysis_t* out_analysis) {
  *out_analysis = (loom_amdgpu_vopd_pair_analysis_t){
      .rejection_reason = LOOM_AMDGPU_VOPD_REJECTION_REASON_UNKNOWN,
  };

  out_analysis->first_eligible = loom_amdgpu_vopd_read_component(
      builder, first, &out_analysis->first_component);
  if (!out_analysis->first_eligible) {
    return;
  }

  out_analysis->second_eligible = loom_amdgpu_vopd_read_component(
      builder, second, &out_analysis->second_component);
  if (!out_analysis->second_eligible) {
    return;
  }

  loom_amdgpu_vopd_candidate_pair_t candidate = {
      .x = out_analysis->first_component,
      .y = out_analysis->second_component,
  };
  out_analysis->rejection_reason =
      LOOM_AMDGPU_VOPD_REJECTION_REASON_COMPONENT_OPCODE_MISMATCH;
  if (!loom_amdgpu_vopd_pair_reason_for_components(
          &out_analysis->first_component, &out_analysis->second_component,
          &candidate.reason)) {
    return;
  }
  out_analysis->rejection_reason =
      LOOM_AMDGPU_VOPD_REJECTION_REASON_FIRST_RESULT_USED_BY_SECOND;
  if (loom_amdgpu_vopd_component_result_is_used_by(first, second)) {
    return;
  }
  out_analysis->rejection_reason =
      LOOM_AMDGPU_VOPD_REJECTION_REASON_LITERAL_MISMATCH;
  if (!loom_amdgpu_vopd_resolve_pair_literal(&out_analysis->first_component,
                                             &out_analysis->second_component,
                                             &candidate)) {
    return;
  }
  out_analysis->rejection_reason =
      LOOM_AMDGPU_VOPD_REJECTION_REASON_REGISTER_CONSTRAINTS;
  if (!loom_amdgpu_vopd_try_register_orientations(
          &candidate, &out_analysis->register_constraint_flags)) {
    return;
  }
  out_analysis->candidate = candidate;
  out_analysis->matched = true;
  out_analysis->rejection_reason = LOOM_AMDGPU_VOPD_REJECTION_REASON_UNKNOWN;
}

static loom_amdgpu_vopd_component_t loom_amdgpu_vopd_final_component(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_amdgpu_vopd_candidate_component_t* candidate) {
  uint16_t src0_selector = 0;
  if (candidate->rule->form == LOOM_AMDGPU_VOPD_COMPONENT_FORM_INLINE_MOV) {
    src0_selector = candidate->src0_selector;
  } else {
    IREE_ASSERT(iree_any_bit_set(candidate->rule->source_register_mask,
                                 LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_SRC0));
    const loom_low_descriptor_set_t* descriptor_set =
        builder->schedule->target.descriptor_set;
    const loom_amdgpu_encoding_table_t* encoding_table =
        loom_amdgpu_encoding_table_for_descriptor_set_ordinal(
            descriptor_set->descriptor_set_ordinal);
    IREE_ASSERT(encoding_table != NULL);
    IREE_ASSERT_LE(encoding_table->vector_source_vgpr0,
                   UINT16_MAX - candidate->src0);
    src0_selector =
        (uint16_t)(encoding_table->vector_source_vgpr0 + candidate->src0);
  }
  return (loom_amdgpu_vopd_component_t){
      .op = candidate->rule->info->op,
      .form = candidate->rule->form,
      .vdst = candidate->vdst,
      .src0 = candidate->src0,
      .vsrc1 = candidate->vsrc1,
      .src0_selector = src0_selector,
      .immediate_u32 = candidate->literal_u32,
  };
}

static void loom_amdgpu_vopd_append_pair(
    loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* first, const loom_low_packet_view_t* second,
    const loom_amdgpu_vopd_candidate_pair_t* candidate) {
  IREE_ASSERT_LT(builder->pair_count, builder->pair_capacity);
  IREE_ASSERT_LT(builder->pair_count, UINT32_MAX);
  const uint32_t pair_index = (uint32_t)builder->pair_count;
  builder->pairs[builder->pair_count++] = (loom_amdgpu_vopd_pair_t){
      .reason = candidate->reason,
      .block_index = first->node->block_index,
      .first_packet_index = (uint32_t)first->packet_index,
      .second_packet_index = (uint32_t)second->packet_index,
      .first_node_index = first->node_index,
      .second_node_index = second->node_index,
      .x = loom_amdgpu_vopd_final_component(builder, &candidate->x),
      .y = loom_amdgpu_vopd_final_component(builder, &candidate->y),
      .flags = candidate->flags,
      .literal_u32 = candidate->literal_u32,
  };
  builder->packets[first->packet_index] = (loom_amdgpu_vopd_packet_t){
      .role = LOOM_AMDGPU_VOPD_PACKET_ROLE_FIRST,
      .pair_index = pair_index,
  };
  builder->packets[second->packet_index] = (loom_amdgpu_vopd_packet_t){
      .role = LOOM_AMDGPU_VOPD_PACKET_ROLE_SECOND,
      .pair_index = pair_index,
  };
}

static loom_amdgpu_vopd_rejection_component_t
loom_amdgpu_vopd_rejection_component_from_candidate(
    const loom_amdgpu_vopd_candidate_component_t* component) {
  return (loom_amdgpu_vopd_rejection_component_t){
      .op = component->rule->info->op,
      .vdst = component->vdst,
      .src0 = component->src0,
      .vsrc1 = component->vsrc1,
      .flags = component->flags,
      .literal_u32 = component->literal_u32,
  };
}

static void loom_amdgpu_vopd_append_rejection(
    loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* first, const loom_low_packet_view_t* second,
    const loom_amdgpu_vopd_pair_analysis_t* analysis,
    loom_amdgpu_vopd_rejection_reason_t reason) {
  if (!analysis->first_eligible || !analysis->second_eligible) {
    return;
  }
  IREE_ASSERT_LT(builder->rejection_count, builder->rejection_capacity);
  builder->rejections[builder->rejection_count++] =
      (loom_amdgpu_vopd_rejection_t){
          .reason = reason,
          .register_constraint_flags = analysis->register_constraint_flags,
          .block_index = first->node->block_index,
          .first_packet_index = (uint32_t)first->packet_index,
          .second_packet_index = (uint32_t)second->packet_index,
          .first_node_index = first->node_index,
          .second_node_index = second->node_index,
          .first = loom_amdgpu_vopd_rejection_component_from_candidate(
              &analysis->first_component),
          .second = loom_amdgpu_vopd_rejection_component_from_candidate(
              &analysis->second_component),
      };
}

static bool loom_amdgpu_vopd_find_visible_packet(
    loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_schedule_block_t* block,
    iree_host_size_t* inout_search_packet_index,
    loom_amdgpu_vopd_visible_packet_t* out_visible) {
  const iree_host_size_t block_end =
      block->scheduled_node_start + block->scheduled_node_count;
  for (iree_host_size_t packet_index = *inout_search_packet_index;
       packet_index < block_end; ++packet_index) {
    const loom_amdgpu_vopd_visible_packet_t visible =
        loom_amdgpu_vopd_classify_packet(builder, packet_index);
    *inout_search_packet_index = packet_index + 1;
    if (iree_any_bit_set(builder->packet_flags[packet_index],
                         LOOM_AMDGPU_VOPD_PACKET_FLAG_TRANSPARENT)) {
      continue;
    }
    *out_visible = visible;
    return true;
  }
  return false;
}

static void loom_amdgpu_vopd_commit_static_packet(
    loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_amdgpu_vopd_visible_packet_t* visible) {
  if (builder->matrix_coexecution == NULL) return;
  const loom_amdgpu_structural_packet_info_t* structural_info =
      visible->packet.descriptor == NULL ? &visible->structural : NULL;
  loom_amdgpu_matrix_coexecution_commit_static_packet(
      builder->matrix_coexecution, &visible->packet, structural_info);
}

static iree_status_t loom_amdgpu_vopd_plan_block(
    loom_amdgpu_vopd_plan_builder_t* builder, uint16_t block_index,
    const loom_low_schedule_block_t* block) {
  loom_amdgpu_vopd_clear_trans_result_windows(builder);
  if (builder->matrix_coexecution != NULL) {
    loom_amdgpu_matrix_coexecution_begin_static_block(
        builder->matrix_coexecution, block_index);
  }
  iree_host_size_t search_packet_index = block->scheduled_node_start;
  loom_amdgpu_vopd_visible_packet_t first = {0};
  bool has_first = loom_amdgpu_vopd_find_visible_packet(
      builder, block, &search_packet_index, &first);
  while (has_first) {
    loom_amdgpu_vopd_apply_trans_result_insertion(builder,
                                                  first.packet.packet_index);
    loom_amdgpu_vopd_visible_packet_t second = {0};
    if (!loom_amdgpu_vopd_find_visible_packet(builder, block,
                                              &search_packet_index, &second)) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_vopd_advance_trans_result_packet(builder, &first.packet));
      loom_amdgpu_vopd_commit_static_packet(builder, &first);
      break;
    }
    loom_amdgpu_vopd_pair_analysis_t analysis = {0};
    loom_amdgpu_vopd_analyze_pair(builder, &first.packet, &second.packet,
                                  &analysis);
    if (!analysis.matched) {
      loom_amdgpu_vopd_append_rejection(builder, &first.packet, &second.packet,
                                        &analysis, analysis.rejection_reason);
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_vopd_advance_trans_result_packet(builder, &first.packet));
      loom_amdgpu_vopd_commit_static_packet(builder, &first);
      first = second;
      continue;
    }
    if (loom_amdgpu_vopd_has_active_trans_result_window(builder)) {
      loom_amdgpu_vopd_append_rejection(
          builder, &first.packet, &second.packet, &analysis,
          LOOM_AMDGPU_VOPD_REJECTION_REASON_TRANS_RESULT_WINDOW);
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_vopd_advance_trans_result_packet(builder, &first.packet));
      loom_amdgpu_vopd_commit_static_packet(builder, &first);
      first = second;
      continue;
    }
    if (iree_any_bit_set(builder->packet_flags[second.packet.packet_index],
                         LOOM_AMDGPU_VOPD_PACKET_FLAG_INSERTION_BLOCKED)) {
      loom_amdgpu_vopd_append_rejection(
          builder, &first.packet, &second.packet, &analysis,
          LOOM_AMDGPU_VOPD_REJECTION_REASON_SECOND_PACKET_HAS_INSERTION);
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_vopd_advance_trans_result_packet(builder, &first.packet));
      loom_amdgpu_vopd_commit_static_packet(builder, &first);
      first = second;
      continue;
    }
    loom_amdgpu_vopd_append_pair(builder, &first.packet, &second.packet,
                                 &analysis.candidate);
    IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_advance_trans_result_pair(
        builder, &first.packet, &second.packet));
    if (builder->matrix_coexecution != NULL) {
      loom_amdgpu_matrix_coexecution_commit_static_vopd_pair(
          builder->matrix_coexecution, &first.packet, &second.packet);
    }
    has_first = loom_amdgpu_vopd_find_visible_packet(
        builder, block, &search_packet_index, &first);
  }
  if (builder->matrix_coexecution != NULL) {
    return loom_amdgpu_matrix_coexecution_end_static_block(
        builder->matrix_coexecution);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_plan_build_pairs(
    loom_amdgpu_vopd_plan_builder_t* builder) {
  loom_low_allocation_value_scratch_t scratch = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_acquire_value_scratch(builder->allocation, &scratch));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < builder->schedule->block_count && iree_status_is_ok(status); ++i) {
    status = loom_amdgpu_vopd_plan_block(builder, (uint16_t)i,
                                         &builder->schedule->blocks[i]);
  }
  loom_low_allocation_release_value_scratch(&scratch);
  return status;
}

static iree_status_t loom_amdgpu_vopd_plan_unpaired_matrix_stream(
    loom_amdgpu_vopd_plan_builder_t* builder) {
  IREE_ASSERT(builder->matrix_coexecution != NULL);
  for (iree_host_size_t block_index = 0;
       block_index < builder->schedule->block_count; ++block_index) {
    loom_amdgpu_matrix_coexecution_begin_static_block(
        builder->matrix_coexecution, (uint16_t)block_index);
    const loom_low_schedule_block_t* block =
        &builder->schedule->blocks[block_index];
    const iree_host_size_t packet_end =
        block->scheduled_node_start + block->scheduled_node_count;
    for (iree_host_size_t packet_index = block->scheduled_node_start;
         packet_index < packet_end; ++packet_index) {
      const loom_low_packet_view_t packet =
          loom_low_packet_at(builder->schedule, packet_index);
      loom_amdgpu_structural_packet_info_t structural = {0};
      if (packet.descriptor == NULL) {
        structural = loom_amdgpu_structural_packet_analyze(
            builder->schedule, builder->allocation, packet.node, 0);
      }
      loom_amdgpu_matrix_coexecution_commit_static_packet(
          builder->matrix_coexecution, &packet,
          packet.descriptor == NULL ? &structural : NULL);
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_coexecution_end_static_block(
        builder->matrix_coexecution));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_plan_write_packet_descriptor_json(
    const loom_amdgpu_vopd_plan_t* plan, uint32_t packet_index,
    loom_output_stream_t* stream) {
  if (packet_index >= plan->packet_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU VOPD packet index %" PRIu32 " is out of range", packet_index);
  }
  const loom_low_packet_view_t packet =
      loom_low_packet_at(plan->schedule, packet_index);
  if (packet.descriptor == NULL) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(
          plan->schedule->target.descriptor_set, packet.descriptor);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU VOPD packet descriptor row does not belong to the selected "
        "descriptor set");
  }
  iree_string_view_t descriptor_key =
      loom_low_descriptor_set_string(plan->schedule->target.descriptor_set,
                                     packet.descriptor->key_string_offset);
  return loom_json_write_escaped_string(stream, descriptor_key);
}

static iree_status_t loom_amdgpu_vopd_plan_write_component_json(
    const loom_amdgpu_vopd_plan_t* plan, uint32_t packet_index,
    uint32_t node_index, uint16_t op, uint16_t vdst, uint16_t src0,
    uint16_t vsrc1, bool sources_swapped, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("packet"), packet_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("node"), node_index));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("descriptor")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_plan_write_packet_descriptor_json(
      plan, packet_index, stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint32_field(&object, IREE_SV("op_id"), op));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("op"), loom_amdgpu_vopd_op_name(op)));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint32_field(&object, IREE_SV("vdst"), vdst));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint32_field(&object, IREE_SV("src0"), src0));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint32_field(&object, IREE_SV("vsrc1"), vsrc1));
  if (sources_swapped) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &object, IREE_SV("sources_swapped"), true));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_amdgpu_vopd_plan_write_pair_json(
    const loom_amdgpu_vopd_plan_t* plan, iree_host_size_t pair_index,
    loom_output_stream_t* stream) {
  const loom_amdgpu_vopd_pair_t* pair = &plan->pairs[pair_index];
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), pair_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("reason"),
      loom_amdgpu_vopd_pair_reason_name(pair->reason)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("block"), pair->block_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("flags"), pair->flags));
  if (iree_any_bit_set(pair->flags, LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("literal_u32"), pair->literal_u32));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_null_field(&object, IREE_SV("literal_u32")));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("x")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_plan_write_component_json(
      plan, pair->first_packet_index, pair->first_node_index, pair->x.op,
      pair->x.vdst, pair->x.src0, pair->x.vsrc1,
      iree_any_bit_set(pair->flags,
                       LOOM_AMDGPU_VOPD_PAIR_FLAG_X_SOURCES_SWAPPED),
      stream));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("y")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_plan_write_component_json(
      plan, pair->second_packet_index, pair->second_node_index, pair->y.op,
      pair->y.vdst, pair->y.src0, pair->y.vsrc1,
      iree_any_bit_set(pair->flags,
                       LOOM_AMDGPU_VOPD_PAIR_FLAG_Y_SOURCES_SWAPPED),
      stream));
  return loom_json_object_end(&object);
}

static iree_status_t loom_amdgpu_vopd_plan_write_rejection_component_json(
    const loom_amdgpu_vopd_plan_t* plan, uint32_t packet_index,
    uint32_t node_index,
    const loom_amdgpu_vopd_rejection_component_t* component,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("packet"), packet_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("node"), node_index));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("descriptor")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_plan_write_packet_descriptor_json(
      plan, packet_index, stream));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("op_id"), component->op));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("op"), loom_amdgpu_vopd_op_name(component->op)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("vdst"), component->vdst));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("src0"), component->src0));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("vsrc1"), component->vsrc1));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("flags"), component->flags));
  if (iree_any_bit_set(component->flags, LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("literal_u32"), component->literal_u32));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_null_field(&object, IREE_SV("literal_u32")));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_amdgpu_vopd_plan_write_register_constraints_json(
    loom_amdgpu_vopd_register_constraint_flags_t flags,
    loom_output_stream_t* stream) {
  loom_json_array_writer_t constraints;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &constraints));
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kVopdRegisterConstraintFlagNames); ++i) {
    const loom_amdgpu_vopd_register_constraint_flag_name_t* row =
        &kVopdRegisterConstraintFlagNames[i];
    if (!iree_any_bit_set(flags, row->flag)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_json_array_write_string_element(&constraints, row->name));
  }
  return loom_json_array_end(&constraints);
}

static iree_status_t loom_amdgpu_vopd_plan_write_rejection_json(
    const loom_amdgpu_vopd_plan_t* plan, iree_host_size_t rejection_index,
    loom_output_stream_t* stream) {
  const loom_amdgpu_vopd_rejection_t* rejection =
      &plan->rejections[rejection_index];
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), rejection_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("reason"),
      loom_amdgpu_vopd_rejection_reason_name(rejection->reason)));
  if (rejection->reason ==
      LOOM_AMDGPU_VOPD_REJECTION_REASON_REGISTER_CONSTRAINTS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("register_constraints")));
    IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_plan_write_register_constraints_json(
        rejection->register_constraint_flags, stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("block"), rejection->block_index));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("first")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_plan_write_rejection_component_json(
      plan, rejection->first_packet_index, rejection->first_node_index,
      &rejection->first, stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("second")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_plan_write_rejection_component_json(
      plan, rejection->second_packet_index, rejection->second_node_index,
      &rejection->second, stream));
  return loom_json_object_end(&object);
}

static iree_status_t loom_amdgpu_vopd_plan_write_packet_json(
    const loom_amdgpu_vopd_plan_t* plan, iree_host_size_t packet_index,
    loom_output_stream_t* stream) {
  const loom_amdgpu_vopd_packet_t* packet = &plan->packets[packet_index];
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), packet_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("role"),
      loom_amdgpu_vopd_packet_role_name(packet->role)));
  if (packet->pair_index == LOOM_AMDGPU_VOPD_PAIR_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_null_field(&object, IREE_SV("pair")));
  } else {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("pair"), packet->pair_index));
  }
  return loom_json_object_end(&object);
}

iree_status_t loom_amdgpu_vopd_plan_format_json(
    const loom_amdgpu_vopd_plan_t* plan, iree_string_builder_t* builder) {
  if (plan == NULL || builder == NULL || plan->schedule == NULL ||
      plan->allocation == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VOPD plan and builder are required");
  }
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("format"), IREE_SV("loom.amdgpu.vopd_plan.v0")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("function"),
      loom_low_diagnostic_function_name(plan->schedule->module,
                                        plan->schedule->function_op)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("target"), plan->schedule->target.target_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("descriptor_set"),
      plan->schedule->target.descriptor_set_key));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("pair_count"), plan->pair_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("rejection_count"), plan->rejection_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("packet_count"), plan->packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("pairs")));
  loom_json_array_writer_t pairs;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &pairs));
  for (iree_host_size_t i = 0; i < plan->pair_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&pairs));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_vopd_plan_write_pair_json(plan, i, &stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&pairs));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("rejections")));
  loom_json_array_writer_t rejections;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &rejections));
  for (iree_host_size_t i = 0; i < plan->rejection_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&rejections));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_vopd_plan_write_rejection_json(plan, i, &stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&rejections));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("packets")));
  loom_json_array_writer_t packets;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &packets));
  for (iree_host_size_t i = 0; i < plan->packet_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&packets));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_vopd_plan_write_packet_json(plan, i, &stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&packets));
  return loom_json_object_end(&object);
}

iree_status_t loom_amdgpu_vopd_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_processor_properties_t* processor_properties,
    const loom_amdgpu_address_state_plan_t* address_state,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    loom_amdgpu_matrix_coexecution_t* matrix_coexecution,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* transient_arena,
    loom_amdgpu_vopd_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vopd_plan_t){0};
  if (!loom_amdgpu_vopd_target_supports_base_vopd(&schedule->target)) {
    if (matrix_coexecution != NULL) {
      loom_amdgpu_vopd_plan_builder_t builder = {
          .schedule = schedule,
          .allocation = allocation,
          .processor_properties = processor_properties,
          .address_state = address_state,
          .wait_packets = wait_packets,
          .arena = arena,
          .transient_arena = transient_arena,
          .matrix_coexecution = matrix_coexecution,
      };
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_vopd_plan_unpaired_matrix_stream(&builder));
    }
    *out_plan = (loom_amdgpu_vopd_plan_t){
        .schedule = schedule,
        .allocation = allocation,
    };
    return iree_ok_status();
  }

  iree_host_size_t component_rule_lookup_count = 0;
  const uint8_t* component_rule_lookup =
      loom_amdgpu_vopd_component_lookup_for_descriptor_set(
          schedule->target.descriptor_set, &component_rule_lookup_count);
  IREE_ASSERT(component_rule_lookup != NULL);

  loom_amdgpu_vopd_plan_builder_t builder = {
      .schedule = schedule,
      .allocation = allocation,
      .processor_properties = processor_properties,
      .address_state = address_state,
      .wait_packets = wait_packets,
      .arena = arena,
      .transient_arena = transient_arena,
      .matrix_coexecution = matrix_coexecution,
      .component_rule_lookup = component_rule_lookup,
      .component_rule_lookup_count = component_rule_lookup_count,
      .tracks_trans_result_windows =
          loom_amdgpu_processor_properties_have_scheduling(
              processor_properties,
              LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR),
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_plan_allocate(&builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_plan_build_pairs(&builder));
  *out_plan = (loom_amdgpu_vopd_plan_t){
      .schedule = schedule,
      .allocation = allocation,
      .pairs = builder.pairs,
      .pair_count = builder.pair_count,
      .rejections = builder.rejections,
      .rejection_count = builder.rejection_count,
      .packets = builder.packets,
      .packet_count = schedule->scheduled_node_count,
  };
  return iree_ok_status();
}
