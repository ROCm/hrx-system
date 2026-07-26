// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/address_state.h"

#include <inttypes.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/packet.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/encoding/encoding.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static iree_status_t loom_amdgpu_address_state_insert_slot_bank(
    loom_amdgpu_vgpr_msb_slot_t slot, uint32_t bank,
    loom_amdgpu_address_state_requirement_t* requirement) {
  if (slot < LOOM_AMDGPU_VGPR_MSB_SLOT_SRC0 ||
      slot > LOOM_AMDGPU_VGPR_MSB_SLOT_DST) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU address-state operand has invalid VGPR-MSB slot %u",
        (unsigned)slot);
  }
  if (bank > 3) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU address-state operand requires VGPR-MSB bank %" PRIu32, bank);
  }
  const uint8_t shift = loom_amdgpu_vgpr_msb_slot_shift(slot);
  const uint8_t slot_mask = (uint8_t)(0x3u << shift);
  const uint8_t slot_value = (uint8_t)(bank << shift);
  if ((requirement->mask & slot_mask) != 0 &&
      (requirement->value & slot_mask) != slot_value) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU packet requires different VGPR-MSB banks for one operand "
        "slot");
  }
  requirement->mask |= slot_mask;
  requirement->value =
      (uint8_t)((requirement->value & ~slot_mask) | slot_value);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_address_state_operand_assignment(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_schedule_node_t* node, uint16_t descriptor_operand_index,
    const loom_low_allocation_assignment_t** out_assignment) {
  *out_assignment = NULL;
  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  const loom_low_operand_t* operand =
      &descriptor_set->operands[node->descriptor->operand_start +
                                descriptor_operand_index];
  const uint16_t result_count = node->descriptor->result_count;
  if (descriptor_operand_index < result_count) {
    const uint16_t result_index = operand->source_value_index;
    if (result_index >= node->result_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU address-state descriptor result %" PRIu16
                              " has no matching schedule result",
                              descriptor_operand_index);
    }
    const loom_value_ordinal_t* result_ordinals =
        loom_low_schedule_node_const_result_ordinals(node);
    *out_assignment = loom_low_allocation_assignment_for_value_ordinal(
        allocation, result_ordinals[result_index], NULL);
    return iree_ok_status();
  }

  if (loom_low_operand_role_is_packet_operand(operand->role)) {
    const uint16_t packet_operand_index = operand->source_value_index;
    if (packet_operand_index >= node->operand_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU address-state descriptor operand %" PRIu16
                              " has no matching schedule operand",
                              descriptor_operand_index);
    }
    const loom_value_ordinal_t* operand_ordinals =
        loom_low_schedule_node_const_operand_ordinals(node);
    *out_assignment = loom_low_allocation_assignment_for_value_ordinal(
        allocation, operand_ordinals[packet_operand_index], NULL);
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "AMDGPU address-state descriptor operand %" PRIu16
                          " has no matching schedule operand",
                          descriptor_operand_index);
}

iree_status_t loom_amdgpu_address_state_query_requirement(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_schedule_node_t* node,
    loom_amdgpu_address_state_requirement_t* out_requirement) {
  if (schedule == NULL || allocation == NULL || node == NULL ||
      out_requirement == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "schedule, allocation, node, and output requirement are required");
  }
  *out_requirement = (loom_amdgpu_address_state_requirement_t){0};
  if (node->kind != LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR) {
    return iree_ok_status();
  }
  const loom_low_descriptor_t* descriptor = node->descriptor;
  if (descriptor == NULL) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + (uint32_t)i];
    if (operand->address_state_slot == 0) {
      if (operand->address_map_kind ==
          LOOM_LOW_OPERAND_ADDRESS_MAP_TARGET_STATE) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU target-state operand has no VGPR-MSB slot");
      }
      continue;
    }
    if (operand->address_map_kind !=
            LOOM_LOW_OPERAND_ADDRESS_MAP_TARGET_STATE &&
        operand->address_map_kind != LOOM_LOW_OPERAND_ADDRESS_MAP_LOW_SUBSET) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU address-state operand uses unsupported address map %u",
          (unsigned)operand->address_map_kind);
    }
    const loom_low_allocation_assignment_t* assignment = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_address_state_operand_assignment(
        schedule, allocation, node, i, &assignment));
    if (assignment == NULL ||
        assignment->location_kind !=
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
        assignment->descriptor_reg_class_id != LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
      continue;
    }
    if (assignment->location_count != operand->unit_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU address-state operand has %" PRIu32
          " assigned VGPRs but descriptor requires %" PRIu16,
          assignment->location_count, operand->unit_count);
    }
    if (assignment->location_count == 0) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU address-state operand is empty");
    }
    const uint64_t assigned_last =
        (uint64_t)assignment->location_base + assignment->location_count - 1u;
    uint32_t bank = 0;
    if (operand->address_map_kind ==
        LOOM_LOW_OPERAND_ADDRESS_MAP_TARGET_STATE) {
      if (operand->addressable_unit_count != LOOM_AMDGPU_VGPR_MSB_WINDOW_SIZE) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU target-state operand must use a %u-VGPR window",
            LOOM_AMDGPU_VGPR_MSB_WINDOW_SIZE);
      }
      if (assignment->location_base / LOOM_AMDGPU_VGPR_MSB_WINDOW_SIZE !=
          assigned_last / LOOM_AMDGPU_VGPR_MSB_WINDOW_SIZE) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "AMDGPU target-state VGPR range v[%" PRIu32
                                ":%" PRIu64 "] crosses a %u-register window",
                                assignment->location_base, assigned_last,
                                LOOM_AMDGPU_VGPR_MSB_WINDOW_SIZE);
      }
      bank = assignment->location_base / LOOM_AMDGPU_VGPR_MSB_WINDOW_SIZE;
    } else if (operand->addressable_unit_count == 0 ||
               operand->addressable_unit_count >
                   LOOM_AMDGPU_VGPR_MSB_WINDOW_SIZE ||
               assigned_last >= operand->addressable_unit_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU low-subset address-state VGPR range v[%" PRIu32 ":%" PRIu64
          "] exceeds its low %" PRIu16 "-register subset",
          assignment->location_base, assigned_last,
          operand->addressable_unit_count);
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_address_state_insert_slot_bank(
        (loom_amdgpu_vgpr_msb_slot_t)operand->address_state_slot, bank,
        out_requirement));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_address_state_read_mode_immediate(
    const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_node_t* node, uint16_t* out_mode_immediate) {
  *out_mode_immediate = 0;
  const loom_op_t* op = node->op;
  if (!loom_low_op_isa(op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU s_set_vgpr_msb packet is not a low.op");
  }
  const loom_string_id_t mode_name_id =
      loom_module_lookup_string(schedule->module, IREE_SV("mode"));
  const loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
  const loom_named_attr_t* attr = NULL;
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    if (attrs.entries[i].name_id == mode_name_id) {
      attr = &attrs.entries[i];
      break;
    }
  }
  if (attr == NULL || attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU s_set_vgpr_msb packet is missing an i64 "
                            "mode immediate");
  }
  if (attr->value.i64 < 0 || attr->value.i64 > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU s_set_vgpr_msb mode immediate %" PRId64
                            " is not a u16",
                            attr->value.i64);
  }
  *out_mode_immediate = (uint16_t)attr->value.i64;
  return iree_ok_status();
}

static void loom_amdgpu_address_state_append_transition(
    loom_amdgpu_address_state_transition_t* transitions,
    iree_host_size_t* transition_count, const loom_low_schedule_node_t* node,
    uint32_t node_index, uint8_t previous_mode, uint8_t new_mode) {
  transitions[(*transition_count)++] = (loom_amdgpu_address_state_transition_t){
      .block_index = node->block_index,
      .node_index = node_index,
      .scheduled_ordinal = node->scheduled_ordinal,
      .mode_immediate = (uint16_t)(((uint16_t)previous_mode << 8) | new_mode),
  };
}

iree_status_t loom_amdgpu_address_state_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_arena_allocator_t* arena, loom_amdgpu_address_state_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_address_state_plan_t){
      .schedule = schedule,
      .allocation = allocation,
  };
  const loom_low_descriptor_t* set_vgpr_msb_descriptor =
      loom_amdgpu_descriptor_ref_descriptor(
          schedule->target.descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_S_SET_VGPR_MSB);
  if (set_vgpr_msb_descriptor == NULL) {
    return iree_ok_status();
  }

  iree_host_size_t transition_capacity = 0;
  if (!iree_host_size_checked_add(schedule->scheduled_node_count,
                                  schedule->block_count,
                                  &transition_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU address-state transition capacity "
                            "overflows host size");
  }
  loom_amdgpu_address_state_transition_t* transitions = NULL;
  if (transition_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, transition_capacity,
                                                   sizeof(*transitions),
                                                   (void**)&transitions));
  }
  iree_host_size_t transition_count = 0;

  for (iree_host_size_t block_index = 0; block_index < schedule->block_count;
       ++block_index) {
    const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
    uint8_t current_mode = 0;
    const loom_low_schedule_node_t* terminator = NULL;
    uint32_t terminator_node_index = LOOM_LOW_SCHEDULE_NODE_NONE;
    for (uint32_t scheduled_ordinal = 0;
         scheduled_ordinal < block->scheduled_node_count; ++scheduled_ordinal) {
      const loom_low_packet_view_t packet = loom_low_packet_at_block_ordinal(
          schedule, (uint32_t)block_index, scheduled_ordinal);
      const uint32_t node_index = packet.node_index;
      const loom_low_schedule_node_t* node = packet.node;
      if (iree_any_bit_set(node->traits, LOOM_TRAIT_TERMINATOR)) {
        terminator = node;
        terminator_node_index = node_index;
      }
      if (node->descriptor == set_vgpr_msb_descriptor) {
        uint16_t mode_immediate = 0;
        IREE_RETURN_IF_ERROR(loom_amdgpu_address_state_read_mode_immediate(
            schedule, node, &mode_immediate));
        if ((uint8_t)(mode_immediate >> 8) != current_mode) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "AMDGPU authored s_set_vgpr_msb previous mode does not match "
              "the scheduled address state");
        }
        current_mode = (uint8_t)(mode_immediate & 0xFFu);
        continue;
      }

      loom_amdgpu_address_state_requirement_t requirement = {0};
      IREE_RETURN_IF_ERROR(loom_amdgpu_address_state_query_requirement(
          schedule, allocation, node, &requirement));
      if (requirement.mask == 0 || (current_mode & requirement.mask) ==
                                       (requirement.value & requirement.mask)) {
        continue;
      }
      if (iree_any_bit_set(node->traits, LOOM_TRAIT_TERMINATOR)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU target-state terminator cannot restore MODE after its "
            "own operands");
      }
      const uint8_t new_mode =
          (uint8_t)((current_mode & ~requirement.mask) |
                    (requirement.value & requirement.mask));
      loom_amdgpu_address_state_append_transition(
          transitions, &transition_count, node, node_index, current_mode,
          new_mode);
      current_mode = new_mode;
    }
    if (current_mode == 0) {
      continue;
    }
    if (terminator == NULL ||
        terminator->scheduled_ordinal + 1 != block->scheduled_node_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU address-state block must end in a scheduled terminator to "
          "restore MODE");
    }
    loom_amdgpu_address_state_append_transition(
        transitions, &transition_count, terminator, terminator_node_index,
        current_mode, /*new_mode=*/0);
  }

  out_plan->transitions = transitions;
  out_plan->transition_count = transition_count;
  return loom_amdgpu_address_state_plan_verify(schedule, allocation, out_plan);
}

iree_status_t loom_amdgpu_address_state_plan_verify(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_address_state_plan_t* plan) {
  if (plan == NULL) {
    return iree_ok_status();
  }
  if (schedule == NULL || allocation == NULL || plan->schedule != schedule ||
      plan->allocation != allocation) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU address-state plan must be derived from the emitted schedule "
        "and allocation");
  }
  if (plan->transition_count != 0 && plan->transitions == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU address-state plan has transitions but no "
                            "transition table");
  }
  uint64_t previous_position = 0;
  bool has_previous_position = false;
  for (iree_host_size_t i = 0; i < plan->transition_count; ++i) {
    const loom_amdgpu_address_state_transition_t* transition =
        &plan->transitions[i];
    if (transition->reserved != 0 ||
        transition->block_index >= schedule->block_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU address-state transition %" PRIhsz
                              " has invalid metadata",
                              i);
    }
    const loom_low_schedule_block_t* block =
        &schedule->blocks[transition->block_index];
    if (transition->scheduled_ordinal >= block->scheduled_node_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU address-state transition %" PRIhsz
                              " references a missing scheduled packet",
                              i);
    }
    const iree_host_size_t packet_index =
        (iree_host_size_t)block->scheduled_node_start +
        transition->scheduled_ordinal;
    if (packet_index >= schedule->scheduled_node_count ||
        schedule->scheduled_node_indices[packet_index] !=
            transition->node_index) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU address-state transition %" PRIhsz
                              " does not match its scheduled node",
                              i);
    }
    const uint64_t position = ((uint64_t)transition->block_index << 32) |
                              transition->scheduled_ordinal;
    if (has_previous_position && position <= previous_position) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU address-state transitions are not in strict scheduled "
          "order");
    }
    if ((uint8_t)(transition->mode_immediate >> 8) ==
        (uint8_t)(transition->mode_immediate & 0xFFu)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU address-state transition does not "
                              "change MODE");
    }
    previous_position = position;
    has_previous_position = true;
  }

  const loom_low_descriptor_t* set_vgpr_msb_descriptor =
      loom_amdgpu_descriptor_ref_descriptor(
          schedule->target.descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_S_SET_VGPR_MSB);
  iree_host_size_t transition_index = 0;
  for (iree_host_size_t block_index = 0; block_index < schedule->block_count;
       ++block_index) {
    const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
    uint8_t current_mode = 0;
    for (uint32_t scheduled_ordinal = 0;
         scheduled_ordinal < block->scheduled_node_count; ++scheduled_ordinal) {
      const iree_host_size_t packet_index =
          (iree_host_size_t)block->scheduled_node_start + scheduled_ordinal;
      const uint32_t node_index =
          schedule->scheduled_node_indices[packet_index];
      const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
      if (transition_index < plan->transition_count) {
        const loom_amdgpu_address_state_transition_t* transition =
            &plan->transitions[transition_index];
        if (transition->block_index == block_index &&
            transition->scheduled_ordinal == scheduled_ordinal) {
          const uint8_t previous_mode =
              (uint8_t)(transition->mode_immediate >> 8);
          if (previous_mode != current_mode) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "AMDGPU address-state transition %" PRIhsz
                " previous mode does not match the scheduled stream",
                transition_index);
          }
          current_mode = (uint8_t)(transition->mode_immediate & 0xFFu);
          ++transition_index;
        }
      }

      if (set_vgpr_msb_descriptor != NULL &&
          node->descriptor == set_vgpr_msb_descriptor) {
        uint16_t mode_immediate = 0;
        IREE_RETURN_IF_ERROR(loom_amdgpu_address_state_read_mode_immediate(
            schedule, node, &mode_immediate));
        if ((uint8_t)(mode_immediate >> 8) != current_mode) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "AMDGPU authored s_set_vgpr_msb previous mode does not match "
              "the verified address-state stream");
        }
        current_mode = (uint8_t)(mode_immediate & 0xFFu);
        continue;
      }

      loom_amdgpu_address_state_requirement_t requirement = {0};
      IREE_RETURN_IF_ERROR(loom_amdgpu_address_state_query_requirement(
          schedule, allocation, node, &requirement));
      if ((current_mode & requirement.mask) !=
          (requirement.value & requirement.mask)) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "AMDGPU address-state plan does not satisfy "
                                "scheduled node %" PRIu32,
                                node_index);
      }
    }
    if (current_mode != 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU address-state plan leaves MODE active at the end of block "
          "%" PRIhsz,
          block_index);
    }
  }
  if (transition_index != plan->transition_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU address-state plan contains an unmatched transition");
  }
  return iree_ok_status();
}
