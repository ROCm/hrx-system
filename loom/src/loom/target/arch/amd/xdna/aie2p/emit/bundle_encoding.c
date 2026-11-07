// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_encoding.h"

#include "loom/codegen/low/read_only_data.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"

loom_aie2p_instruction_info_t loom_aie2p_bundle_encoding_instruction_info(
    loom_aie2p_instruction_id_t id) {
  loom_aie2p_instruction_info_t info;
  const bool found = loom_aie2p_encoding_query_instruction_info(id, &info);
  IREE_ASSERT(found && "generated descriptor encoding ID must be valid");
  return info;
}

uint8_t loom_aie2p_bundle_encoding_single_slot_byte_length(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal) {
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_ordinal];
  const loom_aie2p_instruction_info_t instruction_info =
      loom_aie2p_bundle_encoding_instruction_info(descriptor->encoding_id);
  const loom_aie2p_bundle_format_id_t format =
      loom_aie2p_encoding_find_bundle_format_for_slots(&instruction_info.slot,
                                                       1);
  IREE_ASSERT(format != LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID &&
              "structural AIE2P instruction must have a standalone bundle");
  loom_aie2p_bundle_format_info_t format_info;
  const bool found =
      loom_aie2p_encoding_query_bundle_format_info(format, &format_info);
  IREE_ASSERT(found && format_info.slot_count == 1 &&
              format_info.bit_count % 8 == 0);
  return format_info.bit_count / 8;
}

uint8_t loom_aie2p_bundle_encoding_format_byte_length(
    loom_aie2p_bundle_format_id_t format, iree_host_size_t slot_count) {
  loom_aie2p_bundle_format_info_t format_info;
  const bool found =
      loom_aie2p_encoding_query_bundle_format_info(format, &format_info);
  IREE_ASSERT(found && format_info.slot_count == slot_count &&
              format_info.bit_count % 8 == 0);
  return format_info.bit_count / 8;
}

loom_aie2p_encoded_slot_t loom_aie2p_bundle_encoding_encode_slot_nop(
    loom_aie2p_slot_t slot) {
  iree_string_view_t instruction_name = iree_string_view_empty();
  switch (slot) {
    case LOOM_AIE2P_SLOT_ALU:
      instruction_name = IREE_SV("NOPX");
      break;
    case LOOM_AIE2P_SLOT_LDA:
      instruction_name = IREE_SV("NOPA");
      break;
    case LOOM_AIE2P_SLOT_LDB:
      instruction_name = IREE_SV("NOPB");
      break;
    case LOOM_AIE2P_SLOT_LNG:
      instruction_name = IREE_SV("NOPXM");
      break;
    case LOOM_AIE2P_SLOT_MV:
      instruction_name = IREE_SV("NOPM");
      break;
    case LOOM_AIE2P_SLOT_NOP:
      instruction_name = IREE_SV("NOP");
      break;
    case LOOM_AIE2P_SLOT_ST:
      instruction_name = IREE_SV("NOPS");
      break;
    case LOOM_AIE2P_SLOT_VEC:
      instruction_name = IREE_SV("NOPV");
      break;
    default:
      IREE_ASSERT(false && "AIE2P bundle format must contain physical slots");
      return (loom_aie2p_encoded_slot_t){0};
  }
  const loom_aie2p_instruction_id_t instruction =
      loom_aie2p_encoding_find_instruction(instruction_name);
  loom_aie2p_instruction_info_t instruction_info;
  const bool found = loom_aie2p_encoding_query_instruction_info(
      instruction, &instruction_info);
  IREE_ASSERT(found && instruction_info.slot == slot &&
              instruction_info.field_count == 0 &&
              "AIE2P physical slot must have a native NOP encoding");
  return loom_aie2p_encoding_pack_verified_instruction(instruction, NULL, 0);
}

const loom_low_allocation_packet_move_group_t*
loom_aie2p_bundle_encoding_packet_move_group(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet) {
  return loom_low_allocation_find_packet_move_group_by_source_ordinal(
      &frame->allocation, packet->node_index);
}

void loom_aie2p_bundle_encoding_encode_packet(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet,
    loom_aie2p_encoded_slot_t* out_encoded_slot,
    uint32_t* out_read_only_data_ordinal) {
  *out_read_only_data_ordinal = UINT32_MAX;
  const loom_low_descriptor_set_t* descriptor_set =
      frame->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = packet->descriptor;
  const loom_low_allocation_assignment_t** operand_assignments =
      descriptor->operand_count
          ? (const loom_low_allocation_assignment_t**)iree_alloca(
                descriptor->operand_count * sizeof(*operand_assignments))
          : NULL;
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        loom_low_descriptor_operand(descriptor_set, descriptor, i);
    if (operand->encoding_field_id == 0) {
      operand_assignments[i] = NULL;
      continue;
    }
    const loom_low_allocation_assignment_t* assignment =
        loom_low_packet_descriptor_operand_assignment(&frame->allocation,
                                                      packet, i);
    IREE_ASSERT(assignment->location_kind ==
                    LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER &&
                "spill-free physical descriptor operand must be allocated");
    operand_assignments[i] = assignment;
  }

  const loom_low_descriptor_layout_t* descriptor_layout =
      loom_low_descriptor_layout(descriptor_set, descriptor);
  int64_t* immediate_values =
      descriptor_layout->immediate_count
          ? (int64_t*)iree_alloca(descriptor_layout->immediate_count *
                                  sizeof(*immediate_values))
          : NULL;
  // Required core immediates and canonical IR dictionaries share field-name
  // order. Construction and Low verification establish their positional
  // binding.
  const loom_named_attr_slice_t attrs = loom_low_packet_attrs(packet);
  for (uint16_t i = 0; i < descriptor_layout->immediate_count; ++i) {
    const loom_attribute_t value = attrs.entries[i].value;
    if (value.kind == LOOM_ATTR_SYMBOL) {
      IREE_ASSERT_EQ(packet->descriptor_ordinal,
                     AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32);
      const uint32_t ordinal = loom_low_read_only_data_ordinal(
          &frame->schedule.requirements.read_only_data,
          loom_attr_as_symbol(value));
      IREE_ASSERT_NE(ordinal, UINT32_MAX);
      IREE_ASSERT_EQ(*out_read_only_data_ordinal, UINT32_MAX);
      *out_read_only_data_ordinal = ordinal;
      immediate_values[i] = 0;
      continue;
    }
    immediate_values[i] = value.i64;
  }

  *out_encoded_slot =
      loom_aie2p_descriptor_encode(descriptor_set, packet->descriptor_ordinal,
                                   operand_assignments, immediate_values);
}

void loom_aie2p_bundle_encoding_encode_storage_address(
    const loom_low_emission_frame_t* frame,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_packet_view_t* packet,
    loom_aie2p_encoded_slot_t* out_encoded_slot) {
  const loom_low_schedule_node_t* node = packet->node;
  IREE_ASSERT(loom_low_storage_address_isa(node->op));
  IREE_ASSERT_EQ(node->result_count, 1u);
  const loom_low_allocation_assignment_t* result_assignment =
      loom_low_packet_result_assignment(&frame->allocation, packet, 0);
  IREE_ASSERT(result_assignment != NULL &&
              result_assignment->location_kind ==
                  LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER &&
              result_assignment->descriptor_reg_class_id ==
                  AIE2P_CORE_REG_CLASS_ID_AIE2P_EP &&
              result_assignment->unit_count == 1 &&
              result_assignment->location_count == 1 &&
              "verified AIE2P storage address must have one EP assignment");

  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors
           [AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32];
  IREE_ASSERT_EQ(descriptor->operand_count, 1u);
  IREE_ASSERT_EQ(descriptor->result_count, 1u);
  IREE_ASSERT_EQ(
      loom_low_descriptor_layout(descriptor_set, descriptor)->immediate_count,
      1u);
  const loom_low_allocation_assignment_t* operand_assignments[] = {
      result_assignment,
  };
  const int64_t immediate_values[] = {0};
  *out_encoded_slot = loom_aie2p_descriptor_encode(
      descriptor_set, AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_LOCAL_ADDRESS_I32,
      operand_assignments, immediate_values);
}

loom_aie2p_encoded_slot_t
loom_aie2p_bundle_encoding_encode_structural_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal) {
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_ordinal];
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        loom_low_descriptor_operand(descriptor_set, descriptor, i);
    IREE_ASSERT(operand->encoding_field_id == 0 &&
                "structural descriptor operands must not encode bits");
  }
  IREE_ASSERT(
      loom_low_descriptor_layout(descriptor_set, descriptor)->immediate_count ==
          0 &&
      "structural descriptors must not carry immediates");
  return loom_aie2p_descriptor_encode(descriptor_set, descriptor_ordinal, NULL,
                                      NULL);
}

loom_aie2p_encoded_slot_t
loom_aie2p_bundle_encoding_encode_loop_setup_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal, int64_t immediate) {
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_ordinal];
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        loom_low_descriptor_operand(descriptor_set, descriptor, i);
    IREE_ASSERT(operand->source_value_index == LOOM_LOW_ID_NONE &&
                "hardware-loop state must use an implicit physical register");
  }
  IREE_ASSERT_EQ(
      loom_low_descriptor_layout(descriptor_set, descriptor)->immediate_count,
      1u);
  const int64_t immediate_values[] = {immediate};
  return loom_aie2p_descriptor_encode(descriptor_set, descriptor_ordinal,
                                      /*operand_assignments=*/NULL,
                                      immediate_values);
}
