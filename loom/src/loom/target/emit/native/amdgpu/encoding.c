// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/encoding.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/packet.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/encoding/encoding.h"
#include "loom/target/arch/amdgpu/planning/packet_plan.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/branch_layout.h"
#include "loom/target/emit/native/amdgpu/register_class.h"
#include "loom/target/emit/native/amdgpu/storage_layout.h"

#define LOOM_AMDGPU_SGPR_COUNT 128u

// Maximum byte distance between an unrelaxed block boundary and branch packet
// for which every possible signed SOPP displacement is directly encodable.
#define LOOM_AMDGPU_SOPP_BRANCH_SPAN_BYTE_COUNT \
  ((uint64_t)(-(int64_t)INT16_MIN) * 4u)

typedef uint8_t loom_amdgpu_pc_component_t;
enum loom_amdgpu_pc_component_e {
  // SGPR does not currently hold a component of an s_getpc_b64 result.
  LOOM_AMDGPU_PC_COMPONENT_NONE = 0,
  // SGPR holds the low 32 bits of the captured PC.
  LOOM_AMDGPU_PC_COMPONENT_LO = 1,
  // SGPR holds the high 32 bits of the captured PC.
  LOOM_AMDGPU_PC_COMPONENT_HI = 2,
};

typedef struct loom_amdgpu_pc_register_state_t {
  // PC byte offset returned by the originating s_getpc_b64.
  uint64_t base_pc_byte_offset;
  // Component of the PC value currently held in this register.
  loom_amdgpu_pc_component_t component;
} loom_amdgpu_pc_register_state_t;

typedef struct loom_amdgpu_native_descriptor_refs_t {
  // s_mov_b32 descriptor row, or NULL on targets without that packet.
  const loom_low_descriptor_t* s_mov_b32;
  // s_mov_b32_m0_imm descriptor row, or NULL on targets without that packet.
  const loom_low_descriptor_t* s_mov_b32_m0_imm;
  // v_mov_b32 descriptor row, or NULL on targets without that packet.
  const loom_low_descriptor_t* v_mov_b32;
  // s_set_vgpr_msb descriptor row, or NULL on targets without that packet.
  const loom_low_descriptor_t* set_vgpr_msb;
  // Low-half PC-relative data-symbol add descriptor row.
  const loom_low_descriptor_t* rel32_lo;
  // High-half PC-relative data-symbol add-with-carry descriptor row.
  const loom_low_descriptor_t* rel32_hi;
} loom_amdgpu_native_descriptor_refs_t;

typedef struct loom_amdgpu_encode_inputs_t {
  // Schedule table being encoded.
  const loom_low_schedule_table_t* schedule;
  // Allocation table supplying physical locations.
  const loom_low_allocation_table_t* allocation;
  // Resolved native target encoding profile.
  const loom_amdgpu_descriptor_set_info_t* target;
  // Resolved bit-encoding table for target descriptor packets, or NULL when
  // this descriptor set has no native encoding table.
  const loom_amdgpu_encoding_table_t* encoding_table;
  // Fixed-segment layout shared by low.storage.address packets.
  const loom_amdgpu_storage_layout_t* storage_layout;
  // Module string IDs for descriptor immediate rows, indexed by descriptor-set
  // immediate ordinal.
  const loom_string_id_t* immediate_name_ids;
  // Number of entries in immediate_name_ids.
  iree_host_size_t immediate_name_id_count;
  // Optional encoding products requested by the caller.
  loom_amdgpu_encode_instruction_stream_flags_t flags;
  // Cached descriptor rows needed by native encoding helper packets.
  loom_amdgpu_native_descriptor_refs_t descriptors;
} loom_amdgpu_encode_inputs_t;

typedef struct loom_amdgpu_encode_packet_plan_state_t {
  // Optional planned address-state transitions consumed in scheduled order.
  const loom_amdgpu_address_state_plan_t* address_state;
  // Optional planned wait packets consumed in scheduled order.
  const loom_amdgpu_wait_packet_plan_t* wait_packets;
  // Optional planned wait states consumed in scheduled order.
  const loom_amdgpu_wait_state_plan_t* wait_states;
  // Optional planned VOPD pairs consumed by packet index.
  const loom_amdgpu_vopd_plan_t* vopd_plan;
  // Next address-state transition to compare with the scheduled packet.
  iree_host_size_t next_address_state_index;
  // Next wait-packet row to compare with the current scheduled packet.
  iree_host_size_t next_wait_packet_index;
  // Next wait-state row to compare with the current scheduled packet.
  iree_host_size_t next_wait_state_index;
} loom_amdgpu_encode_packet_plan_state_t;

typedef struct loom_amdgpu_encode_stream_output_t {
  // Destination byte storage, or NULL during the sizing pass.
  uint8_t* data;
  // Capacity of |data| in bytes, or zero during the sizing pass.
  iree_host_size_t capacity;
  // Number of bytes planned or written so far.
  iree_host_size_t length;
  // Number of native instructions planned or written so far.
  uint64_t instruction_count;
} loom_amdgpu_encode_stream_output_t;

typedef struct loom_amdgpu_encode_text_fixup_output_t {
  // Text literal fixups, or NULL during the sizing pass.
  loom_amdgpu_hsaco_text_fixup_t* values;
  // Capacity of |values| in entries, or zero during the sizing pass.
  iree_host_size_t capacity;
  // Number of text literal fixups planned or written so far.
  iree_host_size_t count;
} loom_amdgpu_encode_text_fixup_output_t;

typedef struct loom_amdgpu_encode_native_insertion_output_t {
  // Native insertion rows, or NULL during the sizing pass.
  loom_amdgpu_native_insertion_t* values;
  // Capacity of |values| in entries, or zero during the sizing pass.
  iree_host_size_t capacity;
  // Number of native insertion rows planned or written so far.
  iree_host_size_t count;
} loom_amdgpu_encode_native_insertion_output_t;

typedef struct loom_amdgpu_encode_traversal_state_t {
  // Scheduled packet currently being expanded into native instructions.
  loom_low_packet_view_t current_packet;
  // Low byte of MODE's current VGPR-MSB selector state.
  uint8_t current_vgpr_msb_mode;
  // Block-local per-SGPR PC component facts used by symbolic rel32 fixups.
  loom_amdgpu_pc_register_state_t pc_registers[LOOM_AMDGPU_SGPR_COUNT];
} loom_amdgpu_encode_traversal_state_t;

typedef struct loom_amdgpu_encode_branch_measurement_t {
  // Planned byte offset for each scheduled block.
  loom_amdgpu_branch_layout_block_t* blocks;
  // Emitted SOPP edges captured during exact branch measurement.
  loom_amdgpu_branch_layout_input_edge_t* edges;
  // Capacity of |edges|.
  iree_host_size_t edge_capacity;
  // Number of entries written to |edges|.
  iree_host_size_t edge_count;
  // Positive-size packet boundaries captured during exact measurement.
  loom_amdgpu_branch_layout_anchor_t* anchors;
  // Capacity of |anchors|.
  iree_host_size_t anchor_capacity;
  // Number of entries written to |anchors|.
  iree_host_size_t anchor_count;
} loom_amdgpu_encode_branch_measurement_t;

typedef struct loom_amdgpu_encode_branch_emission_t {
  // Optional converged island layout consumed during the writing pass.
  const loom_amdgpu_branch_layout_t* layout;
  // Next original edge consumed from |layout|.
  iree_host_size_t next_edge_index;
  // Next island group consumed from |layout|.
  iree_host_size_t next_group_index;
} loom_amdgpu_encode_branch_emission_t;

typedef struct loom_amdgpu_encode_branch_state_t {
  // Unrelaxed offsets and optional exceptional-path measurements.
  loom_amdgpu_encode_branch_measurement_t measurement;
  // Final relaxed plan and writing-pass cursors.
  loom_amdgpu_encode_branch_emission_t emission;
} loom_amdgpu_encode_branch_state_t;

typedef struct loom_amdgpu_encode_state_t {
  // Immutable inputs shared by the sizing and writing passes.
  loom_amdgpu_encode_inputs_t inputs;
  // Target packet plan and its pass-local consumption cursors.
  loom_amdgpu_encode_packet_plan_state_t packet_plan;
  // Encoded instruction bytes and aggregate instruction count.
  loom_amdgpu_encode_stream_output_t stream;
  // Relocation records emitted alongside the instruction stream.
  loom_amdgpu_encode_text_fixup_output_t text_fixups;
  // Optional target-owned insertion records emitted for reports.
  loom_amdgpu_encode_native_insertion_output_t native_insertions;
  // Function-local branch placement state.
  loom_amdgpu_encode_branch_state_t branches;
  // Packet traversal and simulated architectural state.
  loom_amdgpu_encode_traversal_state_t traversal;
} loom_amdgpu_encode_state_t;

static iree_string_view_t loom_amdgpu_descriptor_string(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset) {
  return loom_low_descriptor_set_string(descriptor_set, string_offset);
}

static void loom_amdgpu_append_u32(loom_amdgpu_encode_state_t* state,
                                   uint32_t value) {
  if (state->stream.data != NULL) {
    IREE_ASSERT(state->stream.length <= state->stream.capacity &&
                sizeof(value) <= state->stream.capacity - state->stream.length);
    state->stream.data[state->stream.length + 0] = (uint8_t)(value & 0xFFu);
    state->stream.data[state->stream.length + 1] =
        (uint8_t)((value >> 8) & 0xFFu);
    state->stream.data[state->stream.length + 2] =
        (uint8_t)((value >> 16) & 0xFFu);
    state->stream.data[state->stream.length + 3] =
        (uint8_t)((value >> 24) & 0xFFu);
  }
  state->stream.length += sizeof(value);
}

static uint64_t loom_amdgpu_low_bit_mask(uint16_t bit_count) {
  if (bit_count >= 64) {
    return UINT64_MAX;
  }
  return (UINT64_C(1) << bit_count) - 1;
}

static void loom_amdgpu_vgpr_msb_insert_requirement(
    loom_amdgpu_vgpr_msb_slot_t slot, uint32_t bank, uint8_t* mask,
    uint8_t* value) {
  IREE_ASSERT_LT(bank, LOOM_AMDGPU_VGPR_MSB_BANK_COUNT);
  const uint8_t shift = loom_amdgpu_vgpr_msb_slot_shift(slot);
  const uint8_t slot_mask = (uint8_t)(0x3u << shift);
  const uint8_t slot_value = (uint8_t)(bank << shift);
  *mask |= slot_mask;
  *value = (uint8_t)((*value & ~slot_mask) | slot_value);
}

static loom_named_attr_slice_t loom_amdgpu_packet_attrs(
    const loom_low_packet_view_t* packet) {
  const loom_op_t* op = packet->node->op;
  if (loom_low_op_isa(op)) {
    return loom_low_op_attrs(op);
  }
  if (loom_low_const_isa(op)) {
    return loom_low_const_attrs(op);
  }
  return loom_make_named_attr_slice(NULL, 0);
}

static const loom_named_attr_t* loom_amdgpu_find_packet_attr_by_name_id(
    loom_named_attr_slice_t attrs, loom_string_id_t name_id) {
  if (name_id == LOOM_STRING_ID_INVALID) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (attr->name_id == name_id) {
      return attr;
    }
  }
  return NULL;
}

static const loom_low_allocation_assignment_t* loom_amdgpu_map_assignment(
    const loom_low_allocation_table_t* allocation, loom_value_id_t value_id) {
  return loom_low_allocation_map_active_value_assignment(allocation, value_id,
                                                         NULL);
}

static iree_status_t loom_amdgpu_verify_scc_assignment(
    const loom_low_allocation_table_t* allocation, loom_value_id_t value_id) {
  const loom_low_allocation_assignment_t* assignment =
      loom_amdgpu_map_assignment(allocation, value_id);
  if (assignment->descriptor_reg_class_id != LOOM_AMDGPU_REG_CLASS_ID_SCC) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU native encoding conditional branch condition must be SCC");
  }
  if (assignment->location_base != 0 || assignment->location_count != 1) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU native encoding SCC condition must use "
                            "the single architectural SCC register");
  }
  return iree_ok_status();
}

static uint16_t loom_amdgpu_move_location_sgpr(
    const loom_low_move_location_t* location) {
  IREE_ASSERT_LT(location->location, LOOM_AMDGPU_SGPR_COUNT);
  return (uint16_t)location->location;
}

static uint16_t loom_amdgpu_packet_descriptor_operand_sgpr(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_packet_view_t* packet, uint16_t descriptor_operand_index) {
  const loom_low_allocation_assignment_t* assignment =
      loom_low_packet_descriptor_operand_assignment(
          state->inputs.allocation, packet, descriptor_operand_index);
  IREE_ASSERT_LT(assignment->location_base, LOOM_AMDGPU_SGPR_COUNT);
  return (uint16_t)assignment->location_base;
}

static void loom_amdgpu_sgpr_register_range(
    const loom_low_allocation_assignment_t* assignment, uint16_t* out_base,
    uint16_t* out_count) {
  const uint64_t end =
      (uint64_t)assignment->location_base + assignment->location_count;
  IREE_ASSERT_LE(end, LOOM_AMDGPU_SGPR_COUNT);
  *out_base = (uint16_t)assignment->location_base;
  *out_count = (uint16_t)assignment->location_count;
}

static void loom_amdgpu_invalidate_pc_register_range(
    loom_amdgpu_encode_state_t* state, uint32_t base, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i) {
    state->traversal.pc_registers[base + i] =
        (loom_amdgpu_pc_register_state_t){0};
  }
}

static void loom_amdgpu_set_pc_register(loom_amdgpu_encode_state_t* state,
                                        uint16_t sgpr,
                                        uint64_t base_pc_byte_offset,
                                        loom_amdgpu_pc_component_t component) {
  state->traversal.pc_registers[sgpr] = (loom_amdgpu_pc_register_state_t){
      .base_pc_byte_offset = base_pc_byte_offset,
      .component = component,
  };
}

static void loom_amdgpu_propagate_pc_register(loom_amdgpu_encode_state_t* state,
                                              uint16_t destination_sgpr,
                                              uint16_t source_sgpr) {
  state->traversal.pc_registers[destination_sgpr] =
      state->traversal.pc_registers[source_sgpr];
}

static void loom_amdgpu_append_encoding_packet(
    loom_amdgpu_encode_state_t* state,
    const loom_amdgpu_encoding_packet_t* packet) {
  for (uint16_t i = 0; i < packet->word_count; ++i) {
    loom_amdgpu_append_u32(state, packet->words[i]);
  }
  ++state->stream.instruction_count;
}

static void loom_amdgpu_record_native_insertion(
    loom_amdgpu_encode_state_t* state, loom_amdgpu_native_insertion_kind_t kind,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint16_t immediate) {
  if (!iree_all_bits_set(
          state->inputs.flags,
          LOOM_AMDGPU_ENCODE_INSTRUCTION_STREAM_FLAG_CAPTURE_NATIVE_INSERTIONS)) {
    return;
  }
  if (state->native_insertions.values != NULL) {
    IREE_ASSERT_LT(state->native_insertions.count,
                   state->native_insertions.capacity);
    const loom_low_schedule_node_t* node = state->traversal.current_packet.node;
    state->native_insertions.values[state->native_insertions.count] =
        (loom_amdgpu_native_insertion_t){
            .kind = kind,
            .block_index = node->block_index,
            .node_index = state->traversal.current_packet.node_index,
            .scheduled_ordinal = node->scheduled_ordinal,
            .immediate = immediate,
            .descriptor_ref = descriptor_ref,
        };
  }
  ++state->native_insertions.count;
}

static void loom_amdgpu_record_native_descriptor_insertion(
    loom_amdgpu_encode_state_t* state, loom_amdgpu_native_insertion_kind_t kind,
    const loom_low_descriptor_t* descriptor, uint16_t immediate) {
  if (!iree_all_bits_set(
          state->inputs.flags,
          LOOM_AMDGPU_ENCODE_INSTRUCTION_STREAM_FLAG_CAPTURE_NATIVE_INSERTIONS)) {
    return;
  }
  const loom_amdgpu_descriptor_ref_t descriptor_ref =
      loom_amdgpu_descriptor_ref_for_descriptor(
          state->inputs.schedule->target.descriptor_set, descriptor);
  IREE_ASSERT_NE(descriptor_ref, LOOM_AMDGPU_DESCRIPTOR_REF_NONE);
  loom_amdgpu_record_native_insertion(state, kind, descriptor_ref, immediate);
}

static iree_status_t loom_amdgpu_encode_vgpr_msb_mode(
    loom_amdgpu_encode_state_t* state, uint8_t new_mode) {
  if (state->traversal.current_vgpr_msb_mode == new_mode) {
    return iree_ok_status();
  }
  IREE_ASSERT(state->inputs.descriptors.set_vgpr_msb != NULL);
  const uint16_t immediate =
      (uint16_t)(((uint16_t)state->traversal.current_vgpr_msb_mode << 8) |
                 new_mode);
  loom_amdgpu_encoding_packet_t encoded_packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack_sopp_simm16(
      state->inputs.encoding_table,
      state->inputs.descriptors.set_vgpr_msb->encoding_id, immediate,
      &encoded_packet));
  loom_amdgpu_append_encoding_packet(state, &encoded_packet);
  loom_amdgpu_record_native_insertion(
      state, LOOM_AMDGPU_NATIVE_INSERTION_ADDRESS_STATE,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_SET_VGPR_MSB, immediate);
  state->traversal.current_vgpr_msb_mode = new_mode;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_vgpr_msb_requirement(
    loom_amdgpu_encode_state_t* state, uint8_t mask, uint8_t value) {
  const uint8_t new_mode =
      (uint8_t)((state->traversal.current_vgpr_msb_mode & ~mask) |
                (value & mask));
  return loom_amdgpu_encode_vgpr_msb_mode(state, new_mode);
}

static uint16_t loom_amdgpu_assignment_vgpr_low_register(
    const loom_low_allocation_assignment_t* assignment,
    const loom_low_operand_t* operand) {
  uint32_t low_register = assignment->location_base;
  if (operand->address_map_kind == LOOM_LOW_OPERAND_ADDRESS_MAP_TARGET_STATE) {
    low_register %= operand->addressable_unit_count;
  }
  return (uint16_t)low_register;
}

static uint16_t loom_amdgpu_packet_descriptor_operand_vgpr_low_register(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_packet_view_t* packet, uint16_t descriptor_operand_index) {
  const loom_low_descriptor_set_t* descriptor_set =
      state->inputs.schedule->target.descriptor_set;
  const loom_low_operand_t* operand =
      &descriptor_set->operands[packet->descriptor->operand_start +
                                descriptor_operand_index];
  const loom_low_allocation_assignment_t* assignment =
      loom_low_packet_descriptor_operand_assignment(
          state->inputs.allocation, packet, descriptor_operand_index);
  return loom_amdgpu_assignment_vgpr_low_register(assignment, operand);
}

static iree_status_t loom_amdgpu_assignment_field_value(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_allocation_assignment_t* assignment,
    const loom_low_operand_t* operand, uint64_t* out_value) {
  *out_value = 0;
  if (assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    *out_value = assignment->location_base;
    return iree_ok_status();
  }
  if (assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    *out_value = loom_amdgpu_assignment_vgpr_low_register(assignment, operand);
    if (loom_amdgpu_encoding_field_uses_unified_source(
            operand->encoding_field_id)) {
      *out_value += state->inputs.encoding_table->vector_source_vgpr0;
    }
    return iree_ok_status();
  }
  iree_string_view_t register_class = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_register_class_name(
      state->inputs.allocation, assignment, &register_class));
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "AMDGPU native encoding register class '%.*s' is not "
                          "supported by the generic packet encoder",
                          (int)register_class.size, register_class.data);
}

static void loom_amdgpu_push_encoding_field_value(
    loom_amdgpu_encoding_field_value_t* field_values,
    iree_host_size_t* field_value_count, uint16_t field_id, uint64_t value) {
  if (field_id == 0) {
    return;
  }
  IREE_ASSERT_LT(*field_value_count,
                 LOOM_AMDGPU_ENCODING_PACKET_FIELD_VALUE_CAPACITY);
  field_values[*field_value_count] = (loom_amdgpu_encoding_field_value_t){
      .field_id = field_id,
      .reserved = 0,
      .value = value,
  };
  ++*field_value_count;
}

static uint16_t loom_amdgpu_descriptor_single_fixed_encoding_field_u16(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_descriptor_t* descriptor) {
  const loom_low_descriptor_set_t* descriptor_set =
      state->inputs.schedule->target.descriptor_set;
  const loom_low_encoding_field_value_t* field_value =
      &descriptor_set
           ->encoding_field_values[descriptor->encoding_field_value_start];
  return (uint16_t)field_value->value;
}

static bool loom_amdgpu_descriptor_operand_field_already_encoded(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_packet_view_t* packet, uint16_t operand_index,
    const loom_low_operand_t* operand) {
  const loom_low_descriptor_set_t* descriptor_set =
      state->inputs.schedule->target.descriptor_set;
  const loom_low_operand_t* operands =
      &descriptor_set->operands[packet->descriptor->operand_start];
  for (uint16_t previous_index = 0; previous_index < operand_index;
       ++previous_index) {
    const loom_low_operand_t* previous = &operands[previous_index];
    if (previous->encoding_field_id != operand->encoding_field_id) {
      continue;
    }
    return true;
  }
  return false;
}

static uint32_t loom_amdgpu_descriptor_immediate_row(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    uint16_t descriptor_immediate_index) {
  IREE_ASSERT(descriptor_set->immediates != NULL);
  IREE_ASSERT_LT(descriptor_immediate_index, descriptor->immediate_count);
  const uint64_t immediate_row =
      (uint64_t)descriptor->immediate_start + descriptor_immediate_index;
  IREE_ASSERT_LE(immediate_row, UINT32_MAX);
  IREE_ASSERT_LT(immediate_row, descriptor_set->immediate_count);
  return (uint32_t)immediate_row;
}

static const loom_low_immediate_t* loom_amdgpu_descriptor_immediate(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    uint16_t descriptor_immediate_index) {
  return &descriptor_set->immediates[loom_amdgpu_descriptor_immediate_row(
      descriptor_set, descriptor, descriptor_immediate_index)];
}

static loom_string_id_t loom_amdgpu_descriptor_immediate_name_id(
    const loom_amdgpu_encode_state_t* state, uint32_t immediate_row) {
  IREE_ASSERT(state->inputs.immediate_name_ids != NULL);
  IREE_ASSERT_LT(immediate_row, state->inputs.immediate_name_id_count);
  return state->inputs.immediate_name_ids[immediate_row];
}

static iree_status_t loom_amdgpu_read_immediate_field_value(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_packet_view_t* packet, uint16_t descriptor_immediate_index,
    int64_t* out_value) {
  *out_value = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      state->inputs.schedule->target.descriptor_set;
  const uint32_t immediate_row = loom_amdgpu_descriptor_immediate_row(
      descriptor_set, packet->descriptor, descriptor_immediate_index);
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[immediate_row];
  const loom_string_id_t field_name_id =
      loom_amdgpu_descriptor_immediate_name_id(state, immediate_row);
  const loom_named_attr_t* attr = loom_amdgpu_find_packet_attr_by_name_id(
      loom_amdgpu_packet_attrs(packet), field_name_id);
  if (attr != NULL && attr->value.kind == LOOM_ATTR_I64) {
    *out_value = attr->value.i64;
    return iree_ok_status();
  }

  const iree_string_view_t field_name = loom_amdgpu_descriptor_string(
      descriptor_set, immediate->field_name_string_offset);
  if (attr == NULL) {
    if (iree_any_bit_set(immediate->flags,
                         LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE)) {
      *out_value = immediate->default_value;
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU native encoding immediate '%.*s' is "
                            "required",
                            (int)field_name.size, field_name.data);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "AMDGPU native encoding immediate '%.*s' must be an "
                          "i64",
                          (int)field_name.size, field_name.data);
}

static iree_string_view_t loom_amdgpu_read_immediate_field_name(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_packet_view_t* packet, uint16_t descriptor_immediate_index) {
  const loom_low_descriptor_set_t* descriptor_set =
      state->inputs.schedule->target.descriptor_set;
  const loom_low_immediate_t* immediate = loom_amdgpu_descriptor_immediate(
      descriptor_set, packet->descriptor, descriptor_immediate_index);
  return loom_amdgpu_descriptor_string(descriptor_set,
                                       immediate->field_name_string_offset);
}

static iree_status_t loom_amdgpu_read_immediate_u32(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_packet_view_t* packet, uint16_t descriptor_immediate_index,
    uint32_t* out_value) {
  *out_value = 0;
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_immediate_field_value(
      state, packet, descriptor_immediate_index, &value));
  if (value < 0 || value > UINT32_MAX) {
    const iree_string_view_t field_name = loom_amdgpu_read_immediate_field_name(
        state, packet, descriptor_immediate_index);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU native encoding immediate '%.*s' value "
                            "%" PRId64 " is not a u32",
                            (int)field_name.size, field_name.data, value);
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_read_immediate_u16(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_packet_view_t* packet, uint16_t descriptor_immediate_index,
    uint16_t* out_value) {
  *out_value = 0;
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_immediate_field_value(
      state, packet, descriptor_immediate_index, &value));
  if (value < 0 || value > UINT16_MAX) {
    const iree_string_view_t field_name = loom_amdgpu_read_immediate_field_name(
        state, packet, descriptor_immediate_index);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU native encoding immediate '%.*s' value "
                            "%" PRId64 " is not a u16",
                            (int)field_name.size, field_name.data, value);
  }
  *out_value = (uint16_t)value;
  return iree_ok_status();
}

static bool loom_amdgpu_descriptor_semantic_tag_is(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_descriptor_t* descriptor, iree_string_view_t expected_tag) {
  if (descriptor->semantic_tag_string_offset == LOOM_LOW_STRING_OFFSET_NONE) {
    return false;
  }
  const iree_string_view_t semantic_tag = loom_low_descriptor_set_string(
      state->inputs.schedule->target.descriptor_set,
      descriptor->semantic_tag_string_offset);
  return iree_string_view_equal(semantic_tag, expected_tag);
}

static iree_status_t loom_amdgpu_symbol_name_from_attr(
    const loom_amdgpu_encode_state_t* state, const loom_named_attr_t* attr,
    iree_string_view_t field_name, iree_string_view_t* out_symbol_name) {
  *out_symbol_name = iree_string_view_empty();
  if (attr == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU native encoding immediate '%.*s' is "
                            "required",
                            (int)field_name.size, field_name.data);
  }
  if (attr->value.kind != LOOM_ATTR_SYMBOL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU native encoding immediate '%.*s' must be "
                            "a symbol reference",
                            (int)field_name.size, field_name.data);
  }
  const loom_symbol_ref_t symbol_ref = attr->value.symbol;
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= state->inputs.schedule->module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU native encoding immediate '%.*s' references an invalid symbol",
        (int)field_name.size, field_name.data);
  }
  const loom_symbol_t* symbol =
      &state->inputs.schedule->module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= state->inputs.schedule->module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU native encoding immediate '%.*s' "
                            "references an unnamed symbol",
                            (int)field_name.size, field_name.data);
  }
  *out_symbol_name =
      state->inputs.schedule->module->strings.entries[symbol->name_id];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_read_immediate_symbol(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_packet_view_t* packet, uint16_t descriptor_immediate_index,
    iree_string_view_t* out_symbol_name) {
  *out_symbol_name = iree_string_view_empty();
  const loom_low_descriptor_set_t* descriptor_set =
      state->inputs.schedule->target.descriptor_set;
  const uint32_t immediate_row = loom_amdgpu_descriptor_immediate_row(
      descriptor_set, packet->descriptor, descriptor_immediate_index);
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[immediate_row];
  const iree_string_view_t field_name = loom_amdgpu_descriptor_string(
      descriptor_set, immediate->field_name_string_offset);
  const loom_string_id_t field_name_id =
      loom_amdgpu_descriptor_immediate_name_id(state, immediate_row);
  const loom_named_attr_t* attr = loom_amdgpu_find_packet_attr_by_name_id(
      loom_amdgpu_packet_attrs(packet), field_name_id);
  return loom_amdgpu_symbol_name_from_attr(state, attr, field_name,
                                           out_symbol_name);
}

static iree_status_t loom_amdgpu_read_immediate_u64(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_packet_view_t* packet, uint16_t descriptor_immediate_index,
    uint64_t* out_value) {
  *out_value = 0;
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_immediate_field_value(
      state, packet, descriptor_immediate_index, &value));
  if (value < 0) {
    const iree_string_view_t field_name = loom_amdgpu_read_immediate_field_name(
        state, packet, descriptor_immediate_index);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU native encoding immediate '%.*s' value "
                            "%" PRId64 " is negative",
                            (int)field_name.size, field_name.data, value);
  }
  *out_value = (uint64_t)value;
  return iree_ok_status();
}

typedef struct loom_amdgpu_rel32_text_fixup_info_t {
  // HSA code-object relocation applied to the packet literal.
  loom_amdgpu_hsaco_text_fixup_kind_t kind;
  // s_getpc_b64 component required in the packet lhs.
  loom_amdgpu_pc_component_t pc_component;
} loom_amdgpu_rel32_text_fixup_info_t;

static loom_amdgpu_rel32_text_fixup_info_t
loom_amdgpu_descriptor_rel32_text_fixup_info(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_descriptor_t* descriptor) {
  if (descriptor == state->inputs.descriptors.rel32_lo) {
    return (loom_amdgpu_rel32_text_fixup_info_t){
        .kind = LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_LO,
        .pc_component = LOOM_AMDGPU_PC_COMPONENT_LO,
    };
  }
  if (descriptor == state->inputs.descriptors.rel32_hi) {
    return (loom_amdgpu_rel32_text_fixup_info_t){
        .kind = LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_HI,
        .pc_component = LOOM_AMDGPU_PC_COMPONENT_HI,
    };
  }
  return (loom_amdgpu_rel32_text_fixup_info_t){0};
}

static void loom_amdgpu_append_text_fixup(
    loom_amdgpu_encode_state_t* state,
    const loom_amdgpu_hsaco_text_fixup_t* fixup) {
  if (state->text_fixups.values != NULL) {
    IREE_ASSERT_LT(state->text_fixups.count, state->text_fixups.capacity);
    state->text_fixups.values[state->text_fixups.count] = *fixup;
  }
  ++state->text_fixups.count;
}

static iree_status_t loom_amdgpu_packet_lhs_pc_base(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_pc_component_t expected_component,
    uint64_t* out_base_pc_byte_offset) {
  *out_base_pc_byte_offset = 0;
  const uint16_t lhs_operand_index = packet->descriptor->result_count;
  const loom_low_allocation_assignment_t* assignment =
      loom_low_packet_descriptor_operand_assignment(state->inputs.allocation,
                                                    packet, lhs_operand_index);
  IREE_ASSERT_LT(assignment->location_base, LOOM_AMDGPU_SGPR_COUNT);
  const uint16_t sgpr_base = (uint16_t)assignment->location_base;
  const loom_amdgpu_pc_register_state_t pc_register =
      state->traversal.pc_registers[sgpr_base];
  if (pc_register.component == LOOM_AMDGPU_PC_COMPONENT_NONE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU native encoding rel32 add lhs s%" PRIu32
                            " does not hold an s_getpc_b64 component",
                            sgpr_base);
  }
  if (pc_register.component != expected_component) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU native encoding rel32 add lhs s%" PRIu32
                            " holds the wrong s_getpc_b64 component",
                            sgpr_base);
  }
  *out_base_pc_byte_offset = pc_register.base_pc_byte_offset;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_read_immediate_encoding_field_value(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_packet_view_t* packet, uint16_t descriptor_immediate_index,
    uint64_t* out_value) {
  *out_value = 0;
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_immediate_field_value(
      state, packet, descriptor_immediate_index, &value));
  const loom_low_descriptor_set_t* descriptor_set =
      state->inputs.schedule->target.descriptor_set;
  const loom_low_immediate_t* immediate = loom_amdgpu_descriptor_immediate(
      descriptor_set, packet->descriptor, descriptor_immediate_index);
  if (immediate->kind == LOOM_LOW_IMMEDIATE_KIND_SIGNED) {
    const int64_t maximum = immediate->unsigned_max > INT64_MAX
                                ? INT64_MAX
                                : (int64_t)immediate->unsigned_max;
    if (value < immediate->signed_min || value > maximum) {
      const iree_string_view_t field_name = loom_amdgpu_descriptor_string(
          descriptor_set, immediate->field_name_string_offset);
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU native encoding signed immediate '%.*s' "
                              "value %" PRId64 " is out of range",
                              (int)field_name.size, field_name.data, value);
    }
    if (immediate->bit_width == 0 || immediate->bit_width > 64) {
      const iree_string_view_t field_name = loom_amdgpu_descriptor_string(
          descriptor_set, immediate->field_name_string_offset);
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU native encoding signed immediate '%.*s' has invalid bit "
          "width %" PRIu16,
          (int)field_name.size, field_name.data, immediate->bit_width);
    }
    *out_value =
        (uint64_t)value & loom_amdgpu_low_bit_mask(immediate->bit_width);
    return iree_ok_status();
  }
  if (value < 0) {
    const iree_string_view_t field_name = loom_amdgpu_descriptor_string(
        descriptor_set, immediate->field_name_string_offset);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU native encoding immediate '%.*s' value "
                            "%" PRId64 " is negative",
                            (int)field_name.size, field_name.data, value);
  }
  const uint64_t unsigned_value = (uint64_t)value;
  switch (immediate->encoding_id) {
    case LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_SOURCE_INLINE_U32: {
      uint16_t source = 0;
      const bool encoded = loom_amdgpu_encoding_inline_u32_source(
          state->inputs.encoding_table, (uint32_t)unsigned_value, &source);
      IREE_ASSERT(encoded);
      *out_value = source;
      return iree_ok_status();
    }
    case LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_SOURCE_INLINE_F32: {
      uint16_t source = 0;
      const bool encoded = loom_amdgpu_encoding_inline_f32_source(
          state->inputs.encoding_table, (uint32_t)unsigned_value, &source);
      IREE_ASSERT(encoded);
      *out_value = source;
      return iree_ok_status();
    }
    default:
      *out_value = unsigned_value;
      return iree_ok_status();
  }
}

static iree_status_t loom_amdgpu_encode_sop1_s_mov_b32(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  uint16_t sdst = 0;
  if (packet->descriptor == state->inputs.descriptors.s_mov_b32) {
    sdst = loom_amdgpu_packet_descriptor_operand_sgpr(state, packet, 0);
  } else {
    sdst = loom_amdgpu_descriptor_single_fixed_encoding_field_u16(
        state, packet->descriptor);
  }
  uint32_t imm32 = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_read_immediate_u32(state, packet, 0, &imm32));
  loom_amdgpu_encoding_packet_t encoded_packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack_s_mov_b32_u32(
      state->inputs.encoding_table, sdst, imm32, &encoded_packet));
  loom_amdgpu_append_encoding_packet(state, &encoded_packet);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_vop1_v_mov_b32(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  IREE_ASSERT(packet->descriptor == state->inputs.descriptors.v_mov_b32);
  const uint16_t vdst =
      loom_amdgpu_packet_descriptor_operand_vgpr_low_register(state, packet, 0);
  uint32_t imm32 = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_read_immediate_u32(state, packet, 0, &imm32));
  loom_amdgpu_encoding_packet_t encoded_packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack_v_mov_b32_u32(
      state->inputs.encoding_table, vdst, imm32, &encoded_packet));
  loom_amdgpu_append_encoding_packet(state, &encoded_packet);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_s_mov_b32_register(
    loom_amdgpu_encode_state_t* state, uint16_t sdst, uint16_t ssrc0) {
  if (sdst == ssrc0) {
    return iree_ok_status();
  }
  loom_amdgpu_encoding_packet_t encoded_packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack_s_mov_b32_sgpr(
      state->inputs.encoding_table, sdst, ssrc0, &encoded_packet));
  loom_amdgpu_append_encoding_packet(state, &encoded_packet);
  loom_amdgpu_propagate_pc_register(state, sdst, ssrc0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_v_mov_b32_register(
    loom_amdgpu_encode_state_t* state, uint16_t vdst, uint16_t src0) {
  if (state->inputs.encoding_table == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU native encoding descriptor set '%.*s' has no encoding table "
        "for v_mov_b32 register moves",
        (int)state->inputs.target->key.size, state->inputs.target->key.data);
  }

  loom_amdgpu_encoding_packet_t encoded_packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack_v_mov_b32_vgpr(
      state->inputs.encoding_table, vdst, src0, &encoded_packet));
  loom_amdgpu_append_encoding_packet(state, &encoded_packet);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_v_mov_b32_u32(
    loom_amdgpu_encode_state_t* state, uint16_t vdst, uint32_t imm32) {
  if (state->inputs.encoding_table == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU native encoding descriptor set '%.*s' has no encoding table "
        "for v_mov_b32 immediate moves",
        (int)state->inputs.target->key.size, state->inputs.target->key.data);
  }

  loom_amdgpu_encoding_packet_t encoded_packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack_v_mov_b32_u32(
      state->inputs.encoding_table, vdst, imm32, &encoded_packet));
  loom_amdgpu_append_encoding_packet(state, &encoded_packet);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_vgpr_move_location(
    loom_amdgpu_encode_state_t* state,
    const loom_low_move_location_t* destination,
    const loom_low_move_location_t* source) {
  if (destination->location == source->location) {
    return iree_ok_status();
  }
  if (state->inputs.encoding_table == NULL ||
      state->inputs.encoding_table->vector_source_vgpr_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU native encoding descriptor set '%.*s' has no VGPR source "
        "window for move encoding",
        (int)state->inputs.target->key.size, state->inputs.target->key.data);
  }
  const uint32_t window =
      state->inputs.encoding_table->vector_source_vgpr_count;
  const uint32_t destination_bank = destination->location / window;
  const uint32_t source_bank = source->location / window;
  const uint16_t destination_low_register =
      (uint16_t)(destination->location % window);
  const uint16_t source_low_register = (uint16_t)(source->location % window);
  uint8_t mask = 0;
  uint8_t value = 0;
  loom_amdgpu_vgpr_msb_insert_requirement(LOOM_AMDGPU_VGPR_MSB_SLOT_DST,
                                          destination_bank, &mask, &value);
  loom_amdgpu_vgpr_msb_insert_requirement(LOOM_AMDGPU_VGPR_MSB_SLOT_SRC0,
                                          source_bank, &mask, &value);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_encode_vgpr_msb_requirement(state, mask, value));
  return loom_amdgpu_encode_v_mov_b32_register(state, destination_low_register,
                                               source_low_register);
}

static iree_status_t loom_amdgpu_encode_vgpr_move_immediate(
    loom_amdgpu_encode_state_t* state, uint32_t destination_register,
    uint32_t imm32) {
  if (state->inputs.encoding_table == NULL ||
      state->inputs.encoding_table->vector_source_vgpr_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU native encoding descriptor set '%.*s' has no VGPR source "
        "window for immediate move encoding",
        (int)state->inputs.target->key.size, state->inputs.target->key.data);
  }
  const uint32_t window =
      state->inputs.encoding_table->vector_source_vgpr_count;
  const uint32_t destination_bank = destination_register / window;
  const uint16_t destination_low_register =
      (uint16_t)(destination_register % window);
  uint8_t mask = 0;
  uint8_t value = 0;
  loom_amdgpu_vgpr_msb_insert_requirement(LOOM_AMDGPU_VGPR_MSB_SLOT_DST,
                                          destination_bank, &mask, &value);
  const uint8_t saved_mode = state->traversal.current_vgpr_msb_mode;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_encode_vgpr_msb_requirement(state, mask, value));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_encode_v_mov_b32_u32(state, destination_low_register, imm32));
  return loom_amdgpu_encode_vgpr_msb_mode(state, saved_mode);
}

static iree_status_t loom_amdgpu_encode_move(
    void* user_data, const loom_low_move_location_t* destination,
    const loom_low_move_location_t* source) {
  loom_amdgpu_encode_state_t* state = (loom_amdgpu_encode_state_t*)user_data;
  if (destination->descriptor_reg_class_id != source->descriptor_reg_class_id) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU native encoding move between descriptor register class IDs "
        "%" PRIu16 " and %" PRIu16 " is unsupported",
        destination->descriptor_reg_class_id, source->descriptor_reg_class_id);
  }
  if (destination->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    return loom_amdgpu_encode_vgpr_move_location(state, destination, source);
  }
  if (destination->descriptor_reg_class_id != LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU native encoding move for descriptor register class ID %" PRIu16
        " is unsupported",
        destination->descriptor_reg_class_id);
  }
  const uint16_t sdst = loom_amdgpu_move_location_sgpr(destination);
  const uint16_t ssrc0 = loom_amdgpu_move_location_sgpr(source);
  return loom_amdgpu_encode_s_mov_b32_register(state, sdst, ssrc0);
}

static iree_status_t loom_amdgpu_encode_move_range(
    loom_amdgpu_encode_state_t* state, loom_low_move_range_t move_range) {
  const uint8_t saved_mode = state->traversal.current_vgpr_msb_mode;
  for (iree_host_size_t i = 0; i < move_range.count; ++i) {
    const loom_low_move_t* move =
        &state->inputs.allocation->moves[move_range.start + i];
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_encode_move(state, &move->destination, &move->source));
  }
  return loom_amdgpu_encode_vgpr_msb_mode(state, saved_mode);
}

static iree_status_t loom_amdgpu_encode_edge_copy_group(
    loom_amdgpu_encode_state_t* state,
    const loom_low_allocation_edge_copy_group_t* group) {
  return loom_amdgpu_encode_move_range(state, group->move_group.moves);
}

static iree_status_t loom_amdgpu_encode_packet_moves(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  const loom_low_allocation_packet_move_group_t* group =
      loom_low_allocation_find_packet_move_group_by_source_ordinal(
          state->inputs.allocation, packet->node->source_ordinal);
  return loom_amdgpu_encode_move_range(state, group == NULL
                                                  ? (loom_low_move_range_t){0}
                                                  : group->move_group.moves);
}

static iree_status_t loom_amdgpu_encode_copy_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  return loom_amdgpu_encode_packet_moves(state, packet);
}

static iree_status_t loom_amdgpu_encode_slice_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  return loom_amdgpu_encode_packet_moves(state, packet);
}

static iree_status_t loom_amdgpu_encode_concat_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  return loom_amdgpu_encode_packet_moves(state, packet);
}

static iree_status_t loom_amdgpu_encode_sopp_simm16(
    loom_amdgpu_encode_state_t* state, uint16_t opcode, uint16_t immediate) {
  loom_amdgpu_encoding_packet_t encoded_packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack_sopp_simm16(
      state->inputs.encoding_table, opcode, immediate, &encoded_packet));
  loom_amdgpu_append_encoding_packet(state, &encoded_packet);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_s_nop_cycles(
    loom_amdgpu_encode_state_t* state, uint16_t cycle_count) {
  while (cycle_count != 0) {
    const uint16_t chunk = cycle_count > LOOM_AMDGPU_WAIT_STATE_MAX_S_NOP_CYCLES
                               ? LOOM_AMDGPU_WAIT_STATE_MAX_S_NOP_CYCLES
                               : cycle_count;
    IREE_RETURN_IF_ERROR(loom_amdgpu_encode_sopp_simm16(
        state, state->inputs.target->sopp.nop, (uint16_t)(chunk - 1)));
    loom_amdgpu_record_native_insertion(
        state, LOOM_AMDGPU_NATIVE_INSERTION_S_NOP,
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE, (uint16_t)(chunk - 1));
    cycle_count -= chunk;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_s_delay_alu(
    loom_amdgpu_encode_state_t* state, uint16_t delay_alu_immediate) {
  if (state->inputs.target->sopp.delay_alu == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU target does not support S_DELAY_ALU");
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_encode_sopp_simm16(
      state, state->inputs.target->sopp.delay_alu, delay_alu_immediate));
  loom_amdgpu_record_native_insertion(
      state, LOOM_AMDGPU_NATIVE_INSERTION_S_DELAY_ALU,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_DELAY_ALU, delay_alu_immediate);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_v_nop_slots(
    loom_amdgpu_encode_state_t* state, uint16_t issue_count) {
  for (uint16_t i = 0; i < issue_count; ++i) {
    loom_amdgpu_encoding_packet_t encoded_packet;
    IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack_v_nop(
        state->inputs.encoding_table, &encoded_packet));
    loom_amdgpu_append_encoding_packet(state, &encoded_packet);
    loom_amdgpu_record_native_insertion(state,
                                        LOOM_AMDGPU_NATIVE_INSERTION_V_NOP,
                                        LOOM_AMDGPU_DESCRIPTOR_REF_NONE, 0);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_wait_state_action(
    loom_amdgpu_encode_state_t* state,
    const loom_amdgpu_wait_state_t* wait_state) {
  switch (wait_state->action) {
    case LOOM_AMDGPU_WAIT_STATE_ACTION_S_NOP:
      return loom_amdgpu_encode_s_nop_cycles(state, wait_state->cycle_count);
    case LOOM_AMDGPU_WAIT_STATE_ACTION_S_DELAY_ALU:
      return loom_amdgpu_encode_s_delay_alu(state,
                                            wait_state->delay_alu_immediate);
    case LOOM_AMDGPU_WAIT_STATE_ACTION_V_NOP:
      return loom_amdgpu_encode_v_nop_slots(state, wait_state->cycle_count);
    case LOOM_AMDGPU_WAIT_STATE_ACTION_UNKNOWN:
    default: {
      IREE_ASSERT_UNREACHABLE(
          "verified AMDGPU wait-state plans must use known actions");
      IREE_BUILTIN_UNREACHABLE();
    }
  }
}

static bool loom_amdgpu_wait_state_matches_packet(
    const loom_amdgpu_wait_state_t* wait_state,
    const loom_low_packet_view_t* packet) {
  const loom_low_schedule_node_t* node = packet->node;
  return wait_state->block_index == node->block_index &&
         wait_state->scheduled_ordinal == node->scheduled_ordinal &&
         wait_state->node_index == packet->node_index;
}

static void loom_amdgpu_push_immediate_encoding_field_values(
    loom_amdgpu_encode_state_t* state, const loom_low_immediate_t* immediate,
    uint64_t value, loom_amdgpu_encoding_field_value_t* field_values,
    iree_host_size_t* field_value_count) {
  if (immediate->encoding_slice_count == 0) {
    loom_amdgpu_push_encoding_field_value(field_values, field_value_count,
                                          immediate->encoding_field_id, value);
    return;
  }

  const loom_low_descriptor_set_t* descriptor_set =
      state->inputs.schedule->target.descriptor_set;
  for (uint16_t i = 0; i < immediate->encoding_slice_count; ++i) {
    const loom_low_immediate_encoding_slice_t* slice =
        &descriptor_set
             ->immediate_encoding_slices[immediate->encoding_slice_start + i];
    const uint64_t field_value = (value >> slice->source_bit_offset) &
                                 loom_amdgpu_low_bit_mask(slice->bit_count);
    loom_amdgpu_push_encoding_field_value(
        field_values, field_value_count, slice->encoding_field_id, field_value);
  }
}

static iree_status_t loom_amdgpu_descriptor_literal_immediate_u32(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet,
    uint32_t* out_value, bool* out_found) {
  *out_value = 0;
  *out_found = false;
  const loom_low_descriptor_set_t* descriptor_set =
      state->inputs.schedule->target.descriptor_set;
  for (uint16_t i = 0; i < packet->descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        loom_amdgpu_descriptor_immediate(descriptor_set, packet->descriptor, i);
    if (!loom_amdgpu_encoding_field_is_literal(immediate->encoding_field_id) ||
        immediate->encoding_slice_count != 0) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_read_immediate_u32(state, packet, i, out_value));
    *out_found = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_try_encode_vop2_u32_vgpr_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet,
    bool* out_encoded) {
  *out_encoded = false;
  IREE_ASSERT(state->inputs.encoding_table != NULL);
  if (packet->descriptor->result_count != 1 ||
      packet->descriptor->operand_count != 2) {
    return iree_ok_status();
  }

  uint32_t literal = 0;
  bool has_literal = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_literal_immediate_u32(
      state, packet, &literal, &has_literal));
  if (!has_literal) {
    return iree_ok_status();
  }
  bool replaced_src0_literal = false;
  const loom_low_descriptor_set_t* descriptor_set =
      state->inputs.schedule->target.descriptor_set;
  for (uint16_t i = 0; i < packet->descriptor->encoding_field_value_count;
       ++i) {
    const loom_low_encoding_field_value_t* field_value =
        &descriptor_set->encoding_field_values
             [packet->descriptor->encoding_field_value_start + i];
    if (loom_amdgpu_encoding_field_is_src0(field_value->encoding_field_id) &&
        field_value->value == state->inputs.encoding_table->source_literal) {
      replaced_src0_literal = true;
    }
  }
  if (!replaced_src0_literal) {
    return iree_ok_status();
  }

  const uint16_t vdst =
      loom_amdgpu_packet_descriptor_operand_vgpr_low_register(state, packet, 0);
  const uint16_t vsrc1 =
      loom_amdgpu_packet_descriptor_operand_vgpr_low_register(
          state, packet, packet->descriptor->result_count);
  loom_amdgpu_encoding_packet_t encoded_packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack_vop2_u32_vgpr(
      state->inputs.encoding_table, packet->descriptor->encoding_id, vdst,
      literal, vsrc1, &encoded_packet));
  loom_amdgpu_append_encoding_packet(state, &encoded_packet);
  *out_encoded = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_try_encode_special_descriptor_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet,
    bool* out_encoded) {
  *out_encoded = false;
  switch (packet->descriptor->encoding_format_id) {
    case LOOM_AMDGPU_ENCODING_FORMAT_SOP1:
      if (packet->descriptor == state->inputs.descriptors.s_mov_b32 ||
          packet->descriptor == state->inputs.descriptors.s_mov_b32_m0_imm) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_encode_sop1_s_mov_b32(state, packet));
        *out_encoded = true;
      }
      return iree_ok_status();
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP1_LITERAL:
      if (packet->descriptor == state->inputs.descriptors.v_mov_b32) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_encode_vop1_v_mov_b32(state, packet));
        *out_encoded = true;
      }
      return iree_ok_status();
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP2_LITERAL: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_try_encode_vop2_u32_vgpr_packet(
          state, packet, out_encoded));
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_amdgpu_verify_unplanned_descriptor_address_state(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  const loom_amdgpu_address_state_requirement_t requirement =
      loom_amdgpu_address_state_requirement_for_packet(state->inputs.allocation,
                                                       packet);
  if ((state->traversal.current_vgpr_msb_mode & requirement.mask) !=
      (requirement.value & requirement.mask)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU descriptor packet reached native encoding without its "
        "required address-state transition");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_generic_descriptor_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  IREE_ASSERT(state->inputs.encoding_table != NULL);
  loom_amdgpu_encoding_field_value_t
      field_values[LOOM_AMDGPU_ENCODING_PACKET_FIELD_VALUE_CAPACITY];
  iree_host_size_t field_value_count = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      state->inputs.schedule->target.descriptor_set;
  const loom_amdgpu_rel32_text_fixup_info_t rel32_fixup =
      loom_amdgpu_descriptor_rel32_text_fixup_info(state, packet->descriptor);
  const bool has_rel32_text_fixup =
      rel32_fixup.kind != LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_NONE;
  iree_string_view_t rel32_target_symbol = iree_string_view_empty();
  uint64_t rel32_target_symbol_byte_offset = 0;
  uint64_t rel32_base_pc_byte_offset = 0;
  if (has_rel32_text_fixup) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_read_immediate_symbol(
        state, packet, LOOM_AMDGPU_REL32_SYMBOL_IMMEDIATE_SLOT,
        &rel32_target_symbol));
    IREE_RETURN_IF_ERROR(loom_amdgpu_read_immediate_u64(
        state, packet, LOOM_AMDGPU_REL32_BYTE_OFFSET_IMMEDIATE_SLOT,
        &rel32_target_symbol_byte_offset));
    IREE_RETURN_IF_ERROR(loom_amdgpu_packet_lhs_pc_base(
        state, packet, rel32_fixup.pc_component, &rel32_base_pc_byte_offset));
  }

  for (uint16_t i = 0; i < packet->descriptor->encoding_field_value_count;
       ++i) {
    const loom_low_encoding_field_value_t* field_value =
        &descriptor_set->encoding_field_values
             [packet->descriptor->encoding_field_value_start + i];
    loom_amdgpu_push_encoding_field_value(field_values, &field_value_count,
                                          field_value->encoding_field_id,
                                          field_value->value);
  }

  for (uint16_t i = 0; i < packet->descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[packet->descriptor->operand_start + i];
    if (operand->encoding_field_id == 0) {
      continue;
    }
    if (loom_amdgpu_descriptor_operand_field_already_encoded(state, packet, i,
                                                             operand)) {
      continue;
    }
    const loom_low_allocation_assignment_t* assignment =
        loom_low_packet_descriptor_operand_assignment(state->inputs.allocation,
                                                      packet, i);
    uint64_t value = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_assignment_field_value(state, assignment, operand, &value));
    loom_amdgpu_push_encoding_field_value(field_values, &field_value_count,
                                          operand->encoding_field_id, value);
  }

  for (uint16_t i = 0; i < packet->descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        loom_amdgpu_descriptor_immediate(descriptor_set, packet->descriptor, i);
    if (immediate->encoding_field_id == 0 &&
        immediate->encoding_slice_count == 0) {
      continue;
    }
    uint64_t value = 0;
    if (has_rel32_text_fixup && i == LOOM_AMDGPU_REL32_SYMBOL_IMMEDIATE_SLOT) {
      value = 0;
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_read_immediate_encoding_field_value(
          state, packet, i, &value));
    }
    loom_amdgpu_push_immediate_encoding_field_values(
        state, immediate, value, field_values, &field_value_count);
  }

  loom_amdgpu_encoding_packet_t encoded_packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack(
      state->inputs.encoding_table, packet->descriptor->encoding_format_id,
      packet->descriptor->encoding_id, field_values, field_value_count,
      &encoded_packet));
  if (has_rel32_text_fixup) {
    iree_host_size_t literal_byte_offset = 0;
    if (!iree_host_size_checked_add(state->stream.length, sizeof(uint32_t),
                                    &literal_byte_offset)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "AMDGPU native encoding rel32 literal offset "
                              "overflowed");
    }
    const loom_amdgpu_hsaco_text_fixup_t fixup = {
        .kind = rel32_fixup.kind,
        .literal_byte_offset = literal_byte_offset,
        .base_pc_byte_offset = rel32_base_pc_byte_offset,
        .target_symbol = rel32_target_symbol,
        .target_symbol_byte_offset = rel32_target_symbol_byte_offset,
    };
    loom_amdgpu_append_text_fixup(state, &fixup);
  }
  loom_amdgpu_append_encoding_packet(state, &encoded_packet);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_regular_descriptor_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  if (state->packet_plan.address_state == NULL) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_verify_unplanned_descriptor_address_state(state, packet));
  }

  bool encoded = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_encode_special_descriptor_packet(
      state, packet, &encoded));
  if (encoded) {
    return iree_ok_status();
  }

  return loom_amdgpu_encode_generic_descriptor_packet(state, packet);
}

static iree_status_t loom_amdgpu_encode_vopd_pair(
    loom_amdgpu_encode_state_t* state, const loom_amdgpu_vopd_pair_t* pair) {
  loom_amdgpu_encoding_vopdxy_fields_t fields = {
      .op_x = pair->x.op,
      .op_y = pair->y.op,
      .src0_x = pair->x.src0_selector,
      .vsrc1_x = pair->x.vsrc1,
      .vdst_x = pair->x.vdst,
      .src0_y = pair->y.src0_selector,
      .vsrc1_y = pair->y.vsrc1,
      .vdst_y = pair->y.vdst,
  };

  loom_amdgpu_encoding_packet_t encoded_packet;
  if (iree_any_bit_set(pair->flags, LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack_vopdxy_literal(
        state->inputs.encoding_table, &fields, pair->literal_u32,
        &encoded_packet));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack_vopdxy(
        state->inputs.encoding_table, &fields, &encoded_packet));
  }
  loom_amdgpu_append_encoding_packet(state, &encoded_packet);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_try_encode_vopd_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet,
    bool* out_handled) {
  *out_handled = false;
  if (state->packet_plan.vopd_plan == NULL) {
    return iree_ok_status();
  }
  const loom_amdgpu_vopd_packet_t* vopd_packet =
      loom_amdgpu_vopd_plan_packet_at(state->packet_plan.vopd_plan,
                                      packet->packet_index);
  if (vopd_packet == NULL) {
    return iree_ok_status();
  }
  IREE_ASSERT(vopd_packet->pair_index <
              state->packet_plan.vopd_plan->pair_count);
  *out_handled = true;
  if (vopd_packet->role == LOOM_AMDGPU_VOPD_PACKET_ROLE_SECOND) {
    return iree_ok_status();
  }
  IREE_ASSERT(vopd_packet->role == LOOM_AMDGPU_VOPD_PACKET_ROLE_FIRST);
  const loom_amdgpu_vopd_pair_t* pair =
      &state->packet_plan.vopd_plan->pairs[vopd_packet->pair_index];
  return loom_amdgpu_encode_vopd_pair(state, pair);
}

static iree_status_t loom_amdgpu_encode_descriptor_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  if (!loom_amdgpu_descriptor_set_info_has_flags(
          state->inputs.target,
          LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor packet encoding for descriptor "
                            "set '%.*s' is not supported yet",
                            (int)state->inputs.target->key.size,
                            state->inputs.target->key.data);
  }
  if (state->inputs.encoding_table == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU native encoding has no bit table for "
                            "descriptor set '%.*s'",
                            (int)state->inputs.target->key.size,
                            state->inputs.target->key.data);
  }

  const uint16_t encoding_format = packet->descriptor->encoding_format_id;
  if (!loom_amdgpu_encoding_table_has_format(state->inputs.encoding_table,
                                             encoding_format)) {
    const iree_string_view_t key = loom_amdgpu_descriptor_string(
        state->inputs.schedule->target.descriptor_set,
        packet->descriptor->key_string_offset);
    iree_string_view_t format_name =
        loom_amdgpu_encoding_format_name(encoding_format);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU descriptor '%.*s' uses unsupported native encoding format "
        "'%.*s'",
        (int)key.size, key.data, (int)format_name.size, format_name.data);
  }

  return loom_amdgpu_encode_regular_descriptor_packet(state, packet);
}

static iree_status_t loom_amdgpu_update_vgpr_msb_mode_after_descriptor(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  if (state->inputs.descriptors.set_vgpr_msb == NULL ||
      packet->descriptor != state->inputs.descriptors.set_vgpr_msb) {
    return iree_ok_status();
  }
  uint16_t mode = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_immediate_u16(state, packet, 0, &mode));
  state->traversal.current_vgpr_msb_mode = (uint8_t)(mode & 0xFFu);
  return iree_ok_status();
}

static void loom_amdgpu_invalidate_pc_registers_after_descriptor(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  for (uint16_t i = 0; i < packet->descriptor->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_low_packet_descriptor_operand_assignment(state->inputs.allocation,
                                                      packet, i);
    if (assignment->descriptor_reg_class_id != LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
      continue;
    }
    uint16_t sgpr_base = 0;
    uint16_t sgpr_count = 0;
    loom_amdgpu_sgpr_register_range(assignment, &sgpr_base, &sgpr_count);
    loom_amdgpu_invalidate_pc_register_range(state, sgpr_base, sgpr_count);
  }
}

static void loom_amdgpu_record_pc_registers_after_descriptor(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet,
    uint64_t base_pc_byte_offset) {
  if (!loom_amdgpu_descriptor_semantic_tag_is(state, packet->descriptor,
                                              IREE_SV("address.pc.get.u64"))) {
    return;
  }
  const loom_low_allocation_assignment_t* assignment =
      loom_low_packet_descriptor_operand_assignment(state->inputs.allocation,
                                                    packet, 0);
  uint16_t sgpr_base = 0;
  uint16_t sgpr_count = 0;
  loom_amdgpu_sgpr_register_range(assignment, &sgpr_base, &sgpr_count);
  IREE_ASSERT_EQ(sgpr_count, 2);
  loom_amdgpu_set_pc_register(state, sgpr_base, base_pc_byte_offset,
                              LOOM_AMDGPU_PC_COMPONENT_LO);
  loom_amdgpu_set_pc_register(state, (uint16_t)(sgpr_base + 1),
                              base_pc_byte_offset, LOOM_AMDGPU_PC_COMPONENT_HI);
}

static void loom_amdgpu_update_pc_registers_after_descriptor(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet,
    iree_host_size_t packet_end) {
  loom_amdgpu_invalidate_pc_registers_after_descriptor(state, packet);
  loom_amdgpu_record_pc_registers_after_descriptor(state, packet,
                                                   (uint64_t)packet_end);
}

static iree_status_t loom_amdgpu_encode_return_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  (void)packet;
  return loom_amdgpu_encode_sopp_simm16(state,
                                        state->inputs.target->sopp.endpgm, 0);
}

static iree_status_t loom_amdgpu_encode_branch_offset(
    loom_amdgpu_encode_state_t* state, const loom_block_t* target_block,
    uint16_t* out_immediate) {
  *out_immediate = 0;
  const uint32_t target_block_index =
      loom_low_packet_block_index(state->inputs.schedule, target_block);
  if (target_block_index == LOOM_LOW_PACKET_INDEX_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU native encoding branch target block is not "
                            "in the scheduled low function");
  }
  if (state->stream.data == NULL) {
    loom_amdgpu_encode_branch_measurement_t* measurement =
        &state->branches.measurement;
    if (measurement->edges != NULL) {
      IREE_ASSERT_LT(measurement->edge_count, measurement->edge_capacity);
      measurement->edges[measurement->edge_count++] =
          (loom_amdgpu_branch_layout_input_edge_t){
              .source_byte_offset = state->stream.length,
              .target_block_index = target_block_index,
          };
    }
    return iree_ok_status();
  }
  loom_amdgpu_encode_branch_emission_t* emission = &state->branches.emission;
  if (emission->layout != NULL) {
    IREE_ASSERT_LT(emission->next_edge_index, emission->layout->edge_count);
    const loom_amdgpu_branch_layout_edge_t* edge =
        &emission->layout->edges[emission->next_edge_index++];
    *out_immediate = (uint16_t)edge->relative_dword_offset;
    return iree_ok_status();
  }
  const loom_amdgpu_branch_layout_block_t* blocks =
      state->branches.measurement.blocks;
  IREE_ASSERT(blocks != NULL);
  if (state->stream.length > (iree_host_size_t)INT64_MAX - 4 ||
      blocks[target_block_index].byte_offset > (uint64_t)INT64_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU native encoding branch target byte offset exceeds int64_t");
  }

  const int64_t branch_base_offset = (int64_t)state->stream.length + 4;
  const int64_t target_offset = (int64_t)blocks[target_block_index].byte_offset;
  const int64_t relative_byte_offset = target_offset - branch_base_offset;
  if ((relative_byte_offset % 4) != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU native encoding branch target offset is not dword aligned");
  }
  const int64_t relative_dword_offset = relative_byte_offset / 4;
  if (relative_dword_offset < INT16_MIN || relative_dword_offset > INT16_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU native encoding branch target offset %" PRId64
        " dword(s) does not fit a signed 16-bit SOPP label",
        relative_dword_offset);
  }
  *out_immediate =
      (uint16_t)((uint32_t)relative_dword_offset & UINT32_C(0xFFFF));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_branch_groups_before_packet(
    loom_amdgpu_encode_state_t* state, iree_host_size_t packet_index) {
  loom_amdgpu_encode_branch_emission_t* emission = &state->branches.emission;
  if (emission->layout == NULL) return iree_ok_status();
  while (emission->next_group_index < emission->layout->group_count) {
    const loom_amdgpu_branch_layout_group_t* group =
        &emission->layout->groups[emission->next_group_index];
    if (group->packet_index != packet_index) break;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_encode_sopp_simm16(state, state->inputs.target->sopp.branch,
                                       (uint16_t)group->island_count));
    loom_amdgpu_record_native_insertion(
        state, LOOM_AMDGPU_NATIVE_INSERTION_BRANCH_ISLAND_SKIP,
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE, (uint16_t)group->island_count);
    for (uint32_t i = 0; i < group->island_count; ++i) {
      const loom_amdgpu_branch_layout_island_t* island =
          &emission->layout->islands[group->island_start + i];
      IREE_RETURN_IF_ERROR(loom_amdgpu_encode_sopp_simm16(
          state, state->inputs.target->sopp.branch,
          (uint16_t)island->relative_dword_offset));
      loom_amdgpu_record_native_insertion(
          state, LOOM_AMDGPU_NATIVE_INSERTION_BRANCH_ISLAND_HOP,
          LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
          (uint16_t)island->relative_dword_offset);
    }
    ++emission->next_group_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_branch_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  const loom_op_t* op = packet->node->op;
  const loom_block_t* dest = loom_low_br_dest(op);
  loom_value_slice_t args = loom_low_br_args(op);
  if (args.count != 0) {
    const loom_low_allocation_edge_copy_group_t* group =
        loom_low_allocation_find_edge_copy_group_by_source_ordinal(
            state->inputs.allocation, packet->node->source_ordinal);
    if (!group) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU native encoding branch edge copies are missing from "
          "allocation");
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_encode_edge_copy_group(state, group));
  }
  const uint32_t current_block_index = packet->node->block_index;
  const uint32_t dest_block_index =
      loom_low_packet_block_index(state->inputs.schedule, dest);
  if (dest_block_index == current_block_index + 1) {
    return iree_ok_status();
  }
  uint16_t immediate = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_encode_branch_offset(state, dest, &immediate));
  return loom_amdgpu_encode_sopp_simm16(
      state, state->inputs.target->sopp.branch, immediate);
}

static iree_status_t loom_amdgpu_encode_cond_branch_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  const loom_op_t* op = packet->node->op;
  IREE_RETURN_IF_ERROR(loom_amdgpu_verify_scc_assignment(
      state->inputs.allocation, loom_low_cond_br_condition(op)));
  const loom_block_t* true_dest = loom_low_cond_br_true_dest(op);
  const loom_block_t* false_dest = loom_low_cond_br_false_dest(op);
  const uint32_t current_block_index = packet->node->block_index;
  const uint32_t true_block_index =
      loom_low_packet_block_index(state->inputs.schedule, true_dest);
  const uint32_t false_block_index =
      loom_low_packet_block_index(state->inputs.schedule, false_dest);
  if (true_dest == false_dest) {
    if (true_block_index == current_block_index + 1) {
      return iree_ok_status();
    }
    uint16_t immediate = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_encode_branch_offset(state, true_dest, &immediate));
    return loom_amdgpu_encode_sopp_simm16(
        state, state->inputs.target->sopp.branch, immediate);
  }
  if (true_block_index == current_block_index + 1) {
    uint16_t false_immediate = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_encode_branch_offset(state, false_dest, &false_immediate));
    return loom_amdgpu_encode_sopp_simm16(
        state, state->inputs.target->sopp.conditional_branch_scc0,
        false_immediate);
  }
  uint16_t true_immediate = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_encode_branch_offset(state, true_dest, &true_immediate));
  if (false_block_index == current_block_index + 1) {
    return loom_amdgpu_encode_sopp_simm16(
        state, state->inputs.target->sopp.conditional_branch_scc1,
        true_immediate);
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_encode_sopp_simm16(
      state, state->inputs.target->sopp.conditional_branch_scc1,
      true_immediate));
  uint16_t false_immediate = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_encode_branch_offset(state, false_dest, &false_immediate));
  return loom_amdgpu_encode_sopp_simm16(
      state, state->inputs.target->sopp.branch, false_immediate);
}

static iree_status_t loom_amdgpu_encode_live_in_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  (void)state;
  (void)packet;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_storage_reserve_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  (void)state;
  (void)packet;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_storage_view_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  (void)state;
  (void)packet;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_storage_address_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  const loom_op_t* op = packet->node->op;
  loom_amdgpu_storage_layout_reference_t reference;
  loom_amdgpu_storage_layout_lookup_reference(
      state->inputs.storage_layout, state->inputs.schedule->module,
      loom_low_storage_address_storage(op), &reference);
  const uint64_t offset = (uint64_t)loom_low_storage_address_offset(op);
  uint64_t byte_offset = reference.reservation.byte_offset;
  if (byte_offset > UINT32_MAX ||
      reference.byte_offset > UINT32_MAX - byte_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU native encoding low.storage.address byte offset exceeds u32");
  }
  byte_offset += reference.byte_offset;
  if (offset > UINT32_MAX - byte_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU native encoding low.storage.address byte offset exceeds u32");
  }
  const loom_low_allocation_assignment_t* assignment =
      loom_amdgpu_map_assignment(state->inputs.allocation,
                                 loom_low_storage_address_result(op));
  return loom_amdgpu_encode_vgpr_move_immediate(
      state, assignment->location_base, (uint32_t)(byte_offset + offset));
}

static bool loom_amdgpu_wait_packet_matches_packet(
    const loom_amdgpu_wait_packet_t* wait_packet,
    const loom_low_packet_view_t* packet) {
  const loom_low_schedule_node_t* node = packet->node;
  return wait_packet->block_index == node->block_index &&
         wait_packet->scheduled_ordinal == node->scheduled_ordinal &&
         wait_packet->node_index == packet->node_index;
}

static uint16_t loom_amdgpu_wait_packet_immediate_value(
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    const loom_amdgpu_wait_packet_t* wait_packet,
    uint16_t descriptor_immediate_index, uint16_t default_value) {
  for (iree_host_size_t i = 0; i < wait_packet->immediate_count; ++i) {
    const iree_host_size_t immediate_index = wait_packet->immediate_start + i;
    const loom_amdgpu_wait_packet_immediate_t* immediate =
        &wait_packets->immediates[immediate_index];
    if (immediate->descriptor_immediate_index == descriptor_immediate_index) {
      return immediate->value;
    }
  }
  return default_value;
}

static iree_status_t loom_amdgpu_encode_generic_wait_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_descriptor_t* descriptor,
    const loom_amdgpu_wait_packet_t* wait_packet) {
  if (state->inputs.encoding_table == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU native encoding has no bit table for "
                            "descriptor set '%.*s'",
                            (int)state->inputs.target->key.size,
                            state->inputs.target->key.data);
  }

  loom_amdgpu_encoding_field_value_t
      field_values[LOOM_AMDGPU_ENCODING_PACKET_FIELD_VALUE_CAPACITY];
  iree_host_size_t field_value_count = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      state->inputs.schedule->target.descriptor_set;
  for (uint16_t i = 0; i < descriptor->encoding_field_value_count; ++i) {
    const loom_low_encoding_field_value_t* field_value =
        &descriptor_set
             ->encoding_field_values[descriptor->encoding_field_value_start +
                                     i];
    loom_amdgpu_push_encoding_field_value(field_values, &field_value_count,
                                          field_value->encoding_field_id,
                                          field_value->value);
  }

  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* descriptor_immediate =
        loom_amdgpu_descriptor_immediate(descriptor_set, descriptor, i);
    if (descriptor_immediate->encoding_field_id == 0) {
      continue;
    }
    uint16_t default_value = 0;
    if (iree_any_bit_set(descriptor_immediate->flags,
                         LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE)) {
      default_value = (uint16_t)descriptor_immediate->default_value;
    }
    const uint16_t value = loom_amdgpu_wait_packet_immediate_value(
        state->packet_plan.wait_packets, wait_packet, i, default_value);
    loom_amdgpu_push_encoding_field_value(
        field_values, &field_value_count,
        descriptor_immediate->encoding_field_id, value);
  }

  loom_amdgpu_encoding_packet_t encoded_packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack(
      state->inputs.encoding_table, descriptor->encoding_format_id,
      descriptor->encoding_id, field_values, field_value_count,
      &encoded_packet));
  loom_amdgpu_append_encoding_packet(state, &encoded_packet);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_wait_packet(
    loom_amdgpu_encode_state_t* state,
    const loom_amdgpu_wait_packet_t* wait_packet) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_encode_generic_wait_packet(
      state, wait_packet->descriptor, wait_packet));
  loom_amdgpu_record_native_descriptor_insertion(
      state, LOOM_AMDGPU_NATIVE_INSERTION_WAIT, wait_packet->descriptor, 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_wait_packets_before_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  if (state->packet_plan.wait_packets == NULL) {
    return iree_ok_status();
  }
  while (state->packet_plan.next_wait_packet_index <
         state->packet_plan.wait_packets->packet_count) {
    const loom_amdgpu_wait_packet_t* wait_packet =
        &state->packet_plan.wait_packets
             ->packets[state->packet_plan.next_wait_packet_index];
    if (!loom_amdgpu_wait_packet_matches_packet(wait_packet, packet)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_encode_wait_packet(state, wait_packet));
    ++state->packet_plan.next_wait_packet_index;
  }
  return iree_ok_status();
}

static bool loom_amdgpu_address_state_matches_packet(
    const loom_amdgpu_address_state_transition_t* transition,
    const loom_low_packet_view_t* packet) {
  const loom_low_schedule_node_t* node = packet->node;
  return transition->block_index == node->block_index &&
         transition->scheduled_ordinal == node->scheduled_ordinal &&
         transition->node_index == packet->node_index;
}

static iree_status_t loom_amdgpu_encode_address_state_before_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  if (state->packet_plan.address_state == NULL) {
    return iree_ok_status();
  }
  while (state->packet_plan.next_address_state_index <
         state->packet_plan.address_state->transition_count) {
    const loom_amdgpu_address_state_transition_t* transition =
        &state->packet_plan.address_state
             ->transitions[state->packet_plan.next_address_state_index];
    if (!loom_amdgpu_address_state_matches_packet(transition, packet)) {
      return iree_ok_status();
    }
    const uint8_t new_mode = (uint8_t)(transition->mode_immediate & 0xFFu);
    IREE_RETURN_IF_ERROR(loom_amdgpu_encode_vgpr_msb_mode(state, new_mode));
    ++state->packet_plan.next_address_state_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_wait_states_before_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  if (state->packet_plan.wait_states == NULL) {
    return iree_ok_status();
  }
  while (state->packet_plan.next_wait_state_index <
         state->packet_plan.wait_states->state_count) {
    const loom_amdgpu_wait_state_t* wait_state =
        &state->packet_plan.wait_states
             ->states[state->packet_plan.next_wait_state_index];
    if (!loom_amdgpu_wait_state_matches_packet(wait_state, packet)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_encode_wait_state_action(state, wait_state));
    ++state->packet_plan.next_wait_state_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  if (loom_low_packet_is_compile_time_only(packet)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_encode_address_state_before_packet(state, packet));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_encode_wait_packets_before_packet(state, packet));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_encode_wait_states_before_packet(state, packet));
  if (packet->descriptor != NULL) {
    bool handled_vopd = false;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_try_encode_vopd_packet(state, packet, &handled_vopd));
    if (handled_vopd) {
      return iree_ok_status();
    }
    const iree_host_size_t packet_start = state->stream.length;
    IREE_RETURN_IF_ERROR(loom_amdgpu_encode_descriptor_packet(state, packet));
    if (state->stream.length != packet_start) {
      loom_amdgpu_update_pc_registers_after_descriptor(state, packet,
                                                       state->stream.length);
    }
    return loom_amdgpu_update_vgpr_msb_mode_after_descriptor(state, packet);
  }
  const loom_op_t* op = packet->node->op;
  switch (op->kind) {
    case LOOM_OP_LOW_LIVE_IN:
      return loom_amdgpu_encode_live_in_packet(state, packet);
    case LOOM_OP_LOW_STORAGE_RESERVE:
      return loom_amdgpu_encode_storage_reserve_packet(state, packet);
    case LOOM_OP_LOW_STORAGE_VIEW:
      return loom_amdgpu_encode_storage_view_packet(state, packet);
    case LOOM_OP_LOW_COPY:
    case LOOM_OP_LOW_MOVE:
      return loom_amdgpu_encode_copy_packet(state, packet);
    case LOOM_OP_LOW_SLICE:
      return loom_amdgpu_encode_slice_packet(state, packet);
    case LOOM_OP_LOW_CONCAT:
      return loom_amdgpu_encode_concat_packet(state, packet);
    case LOOM_OP_LOW_STORAGE_ADDRESS:
      return loom_amdgpu_encode_storage_address_packet(state, packet);
    case LOOM_OP_LOW_BR:
      return loom_amdgpu_encode_branch_packet(state, packet);
    case LOOM_OP_LOW_COND_BR:
      return loom_amdgpu_encode_cond_branch_packet(state, packet);
    case LOOM_OP_LOW_RETURN:
      return loom_amdgpu_encode_return_packet(state, packet);
    default:
      break;
  }
  const loom_op_vtable_t* vtable =
      loom_op_vtable(state->inputs.schedule->module, op);
  iree_string_view_t op_name =
      vtable ? loom_op_vtable_name(vtable) : IREE_SV("<unknown>");
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "AMDGPU native encoding does not support "
                          "structural op %.*s",
                          (int)op_name.size, op_name.data);
}

static iree_status_t loom_amdgpu_resolve_encoding_target(
    const loom_low_schedule_table_t* schedule,
    const loom_amdgpu_descriptor_set_info_t** out_target) {
  *out_target = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_info_lookup_descriptor_set_by_ordinal(
      schedule->target.descriptor_set->descriptor_set_ordinal, out_target));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_resolve_immediate_name_ids(
    const loom_low_schedule_table_t* schedule,
    const loom_string_id_t** out_immediate_name_ids,
    iree_host_size_t* out_immediate_name_id_count,
    iree_arena_allocator_t* arena) {
  *out_immediate_name_ids = NULL;
  *out_immediate_name_id_count = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  if (descriptor_set->immediate_count == 0) {
    return iree_ok_status();
  }

  loom_string_id_t* immediate_name_ids = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, descriptor_set->immediate_count, sizeof(*immediate_name_ids),
      (void**)&immediate_name_ids));
  for (iree_host_size_t i = 0; i < descriptor_set->immediate_count; ++i) {
    const loom_low_immediate_t* immediate = &descriptor_set->immediates[i];
    const iree_string_view_t field_name = loom_amdgpu_descriptor_string(
        descriptor_set, immediate->field_name_string_offset);
    immediate_name_ids[i] =
        loom_module_lookup_string(schedule->module, field_name);
  }
  *out_immediate_name_ids = immediate_name_ids;
  *out_immediate_name_id_count = descriptor_set->immediate_count;
  return iree_ok_status();
}

typedef struct loom_amdgpu_branch_measurement_requirements_t {
  // Exact number of emitted non-fallthrough SOPP edges.
  iree_host_size_t edge_count;
  // Whether at least one edge may exceed the signed SOPP range.
  bool may_require_relaxation;
} loom_amdgpu_branch_measurement_requirements_t;

static void loom_amdgpu_accumulate_branch_measurement_requirement(
    const loom_low_schedule_table_t* schedule,
    const loom_amdgpu_branch_layout_block_t* blocks, uint64_t byte_length,
    uint32_t source_block_index, const loom_block_t* target_block,
    loom_amdgpu_branch_measurement_requirements_t* requirements) {
  ++requirements->edge_count;
  const uint32_t target_block_index =
      loom_low_packet_block_index(schedule, target_block);
  IREE_ASSERT_NE(target_block_index, LOOM_LOW_PACKET_INDEX_NONE);
  const uint64_t source_start = blocks[source_block_index].byte_offset;
  const uint64_t source_end = source_block_index + 1u < schedule->block_count
                                  ? blocks[source_block_index + 1u].byte_offset
                                  : byte_length;
  const uint64_t target = blocks[target_block_index].byte_offset;
  if ((target > source_start &&
       target - source_start > LOOM_AMDGPU_SOPP_BRANCH_SPAN_BYTE_COUNT) ||
      (source_end > target &&
       source_end - target > LOOM_AMDGPU_SOPP_BRANCH_SPAN_BYTE_COUNT)) {
    requirements->may_require_relaxation = true;
  }
}

static loom_amdgpu_branch_measurement_requirements_t
loom_amdgpu_query_branch_measurement_requirements(
    const loom_low_schedule_table_t* schedule,
    const loom_amdgpu_branch_layout_block_t* blocks, uint64_t byte_length) {
  loom_amdgpu_branch_measurement_requirements_t requirements = {0};
  if (byte_length <= LOOM_AMDGPU_SOPP_BRANCH_SPAN_BYTE_COUNT) {
    return requirements;
  }
  for (uint32_t block_index = 0; block_index < schedule->block_count;
       ++block_index) {
    const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
    if (block->scheduled_node_count == 0) continue;
    const loom_low_packet_view_t packet = loom_low_packet_at_block_ordinal(
        schedule, block_index, block->scheduled_node_count - 1u);
    const loom_op_t* op = packet.node->op;
    if (loom_low_br_isa(op)) {
      const loom_block_t* target_block = loom_low_br_dest(op);
      const uint32_t target_block_index =
          loom_low_packet_block_index(schedule, target_block);
      IREE_ASSERT_NE(target_block_index, LOOM_LOW_PACKET_INDEX_NONE);
      if (target_block_index != block_index + 1u) {
        loom_amdgpu_accumulate_branch_measurement_requirement(
            schedule, blocks, byte_length, block_index, target_block,
            &requirements);
      }
      continue;
    }
    if (!loom_low_cond_br_isa(op)) continue;
    const loom_block_t* true_target = loom_low_cond_br_true_dest(op);
    const loom_block_t* false_target = loom_low_cond_br_false_dest(op);
    const uint32_t true_target_index =
        loom_low_packet_block_index(schedule, true_target);
    const uint32_t false_target_index =
        loom_low_packet_block_index(schedule, false_target);
    IREE_ASSERT_NE(true_target_index, LOOM_LOW_PACKET_INDEX_NONE);
    IREE_ASSERT_NE(false_target_index, LOOM_LOW_PACKET_INDEX_NONE);
    if (true_target == false_target) {
      if (true_target_index != block_index + 1u) {
        loom_amdgpu_accumulate_branch_measurement_requirement(
            schedule, blocks, byte_length, block_index, true_target,
            &requirements);
      }
    } else if (true_target_index == block_index + 1u) {
      loom_amdgpu_accumulate_branch_measurement_requirement(
          schedule, blocks, byte_length, block_index, false_target,
          &requirements);
    } else {
      loom_amdgpu_accumulate_branch_measurement_requirement(
          schedule, blocks, byte_length, block_index, true_target,
          &requirements);
      if (false_target_index != block_index + 1u) {
        loom_amdgpu_accumulate_branch_measurement_requirement(
            schedule, blocks, byte_length, block_index, false_target,
            &requirements);
      }
    }
  }
  return requirements;
}

static iree_status_t loom_amdgpu_encode_instruction_stream_into_state(
    loom_amdgpu_encode_state_t* state) {
  state->stream.length = 0;
  state->stream.instruction_count = 0;
  state->text_fixups.count = 0;
  state->native_insertions.count = 0;
  state->traversal.current_vgpr_msb_mode = 0;
  state->packet_plan.next_address_state_index = 0;
  state->packet_plan.next_wait_packet_index = 0;
  state->packet_plan.next_wait_state_index = 0;
  state->branches.measurement.edge_count = 0;
  state->branches.measurement.anchor_count = 0;
  state->branches.emission.next_edge_index = 0;
  state->branches.emission.next_group_index = 0;
  memset(state->traversal.pc_registers, 0,
         sizeof(state->traversal.pc_registers));
  for (iree_host_size_t block_index = 0;
       block_index < state->inputs.schedule->block_count; ++block_index) {
    const loom_low_schedule_block_t* block =
        &state->inputs.schedule->blocks[block_index];
    memset(state->traversal.pc_registers, 0,
           sizeof(state->traversal.pc_registers));
    if (state->branches.measurement.blocks != NULL) {
      state->branches.measurement.blocks[block_index].byte_offset =
          state->stream.length;
    }
    for (uint32_t scheduled_ordinal = 0;
         scheduled_ordinal < block->scheduled_node_count; ++scheduled_ordinal) {
      const iree_host_size_t packet_index =
          block->scheduled_node_start + scheduled_ordinal;
      state->traversal.current_packet =
          loom_low_packet_at(state->inputs.schedule, packet_index);
      const loom_low_packet_view_t* packet = &state->traversal.current_packet;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_encode_branch_groups_before_packet(state, packet_index));
      const iree_host_size_t packet_start = state->stream.length;
      IREE_RETURN_IF_ERROR(loom_amdgpu_encode_packet(state, packet));
      loom_amdgpu_encode_branch_measurement_t* measurement =
          &state->branches.measurement;
      if (measurement->anchors != NULL &&
          state->stream.length != packet_start) {
        IREE_ASSERT_LT(measurement->anchor_count, measurement->anchor_capacity);
        IREE_ASSERT_LE(packet_index, UINT32_MAX);
        measurement->anchors[measurement->anchor_count++] =
            (loom_amdgpu_branch_layout_anchor_t){
                .byte_offset = packet_start,
                .packet_index = (uint32_t)packet_index,
            };
      }
    }
    if (state->packet_plan.address_state == NULL) {
      if (state->traversal.current_vgpr_msb_mode != 0) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU native encoding block left VGPR-MSB address state active");
      }
    } else {
      IREE_ASSERT_EQ(state->traversal.current_vgpr_msb_mode, 0);
    }
  }
  if (state->packet_plan.address_state != NULL) {
    IREE_ASSERT_EQ(state->packet_plan.next_address_state_index,
                   state->packet_plan.address_state->transition_count);
  }
  if (state->packet_plan.wait_packets != NULL) {
    IREE_ASSERT_EQ(state->packet_plan.next_wait_packet_index,
                   state->packet_plan.wait_packets->packet_count);
  }
  if (state->packet_plan.wait_states != NULL) {
    IREE_ASSERT_EQ(state->packet_plan.next_wait_state_index,
                   state->packet_plan.wait_states->state_count);
  }
  if (state->branches.emission.layout != NULL) {
    IREE_ASSERT_EQ(state->branches.emission.next_edge_index,
                   state->branches.emission.layout->edge_count);
    IREE_ASSERT_EQ(state->branches.emission.next_group_index,
                   state->branches.emission.layout->group_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_instruction_stream_internal(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_encode_instruction_stream_options_t* options,
    loom_amdgpu_encoded_instruction_stream_t* out_stream,
    iree_arena_allocator_t* arena) {
  *out_stream = (loom_amdgpu_encoded_instruction_stream_t){0};
  const loom_amdgpu_packet_plan_t* packet_plan =
      options ? options->packet_plan : NULL;
  const loom_amdgpu_address_state_plan_t* address_state =
      packet_plan ? &packet_plan->address_state : NULL;
  const loom_amdgpu_wait_packet_plan_t* wait_packets =
      packet_plan ? &packet_plan->wait_packets : NULL;
  const loom_amdgpu_wait_state_plan_t* wait_states =
      packet_plan ? &packet_plan->wait_states : NULL;
  const loom_amdgpu_vopd_plan_t* vopd_plan =
      packet_plan ? &packet_plan->vopd_plan : NULL;
  const loom_amdgpu_encode_instruction_stream_flags_t flags =
      options ? options->flags
              : LOOM_AMDGPU_ENCODE_INSTRUCTION_STREAM_FLAG_NONE;
  const loom_amdgpu_descriptor_set_info_t* target = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_encoding_target(schedule, &target));
  const loom_amdgpu_encoding_table_t* encoding_table =
      loom_amdgpu_encoding_table_for_descriptor_set_ordinal(target->ordinal);
  const loom_string_id_t* immediate_name_ids = NULL;
  iree_host_size_t immediate_name_id_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_immediate_name_ids(
      schedule, &immediate_name_ids, &immediate_name_id_count, arena));
  loom_amdgpu_storage_layout_t derived_storage_layout;
  const loom_amdgpu_storage_layout_t* storage_layout =
      options ? options->storage_layout : NULL;
  if (storage_layout == NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_build(
        &schedule->storage_layout, arena, &derived_storage_layout));
    storage_layout = &derived_storage_layout;
  }
  const loom_amdgpu_native_descriptor_refs_t descriptors = {
      .s_mov_b32 = loom_amdgpu_descriptor_ref_descriptor(
          schedule->target.descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32),
      .s_mov_b32_m0_imm = loom_amdgpu_descriptor_ref_descriptor(
          schedule->target.descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM),
      .v_mov_b32 = loom_amdgpu_descriptor_ref_descriptor(
          schedule->target.descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32),
      .set_vgpr_msb = loom_amdgpu_descriptor_ref_descriptor(
          schedule->target.descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_S_SET_VGPR_MSB),
      .rel32_lo = loom_amdgpu_descriptor_ref_descriptor(
          schedule->target.descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32_RHS_SYMBOL_REL32_LO),
      .rel32_hi = loom_amdgpu_descriptor_ref_descriptor(
          schedule->target.descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32_RHS_SYMBOL_REL32_HI),
  };
  loom_amdgpu_branch_layout_block_t* branch_blocks = NULL;
  if (schedule->block_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, schedule->block_count,
                                                   sizeof(*branch_blocks),
                                                   (void**)&branch_blocks));
  }

  loom_low_allocation_value_scratch_t scratch = {0};
  iree_status_t status =
      loom_low_allocation_acquire_value_scratch(allocation, &scratch);
  const loom_amdgpu_encode_inputs_t inputs = {
      .schedule = schedule,
      .allocation = allocation,
      .target = target,
      .encoding_table = encoding_table,
      .storage_layout = storage_layout,
      .immediate_name_ids = immediate_name_ids,
      .immediate_name_id_count = immediate_name_id_count,
      .flags = flags,
      .descriptors = descriptors,
  };
  const loom_amdgpu_encode_packet_plan_state_t packet_plan_state = {
      .address_state = address_state,
      .wait_packets = wait_packets,
      .wait_states = wait_states,
      .vopd_plan = vopd_plan,
  };
  loom_amdgpu_encode_state_t sizing_state = {
      .inputs = inputs,
      .packet_plan = packet_plan_state,
      .branches =
          {
              .measurement =
                  {
                      .blocks = branch_blocks,
                  },
          },
  };
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_encode_instruction_stream_into_state(&sizing_state);
  }

  loom_amdgpu_branch_layout_t branch_layout = {0};
  if (iree_status_is_ok(status)) {
    const loom_amdgpu_branch_measurement_requirements_t requirements =
        loom_amdgpu_query_branch_measurement_requirements(
            schedule, branch_blocks, sizing_state.stream.length);
    if (requirements.may_require_relaxation) {
      loom_amdgpu_branch_layout_input_edge_t* measured_edges = NULL;
      status = iree_arena_allocate_array(arena, requirements.edge_count,
                                         sizeof(*measured_edges),
                                         (void**)&measured_edges);
      loom_amdgpu_branch_layout_anchor_t* measured_anchors = NULL;
      if (iree_status_is_ok(status)) {
        status = iree_arena_allocate_array(
            arena, schedule->scheduled_node_count, sizeof(*measured_anchors),
            (void**)&measured_anchors);
      }
      if (iree_status_is_ok(status)) {
        sizing_state.branches.measurement.edges = measured_edges;
        sizing_state.branches.measurement.edge_capacity =
            requirements.edge_count;
        sizing_state.branches.measurement.anchors = measured_anchors;
        sizing_state.branches.measurement.anchor_capacity =
            schedule->scheduled_node_count;
        status =
            loom_amdgpu_encode_instruction_stream_into_state(&sizing_state);
      }
      if (iree_status_is_ok(status)) {
        IREE_ASSERT_EQ(sizing_state.branches.measurement.edge_count,
                       requirements.edge_count);
        const loom_amdgpu_branch_layout_input_t branch_layout_input = {
            .byte_length = sizing_state.stream.length,
            .blocks = branch_blocks,
            .block_count = schedule->block_count,
            .edges = measured_edges,
            .edge_count = sizing_state.branches.measurement.edge_count,
            .anchors = measured_anchors,
            .anchor_count = sizing_state.branches.measurement.anchor_count,
        };
        status = loom_amdgpu_branch_layout_build(&branch_layout_input, arena,
                                                 &branch_layout);
      }
    }
  }

  iree_host_size_t final_byte_length = sizing_state.stream.length;
  const loom_amdgpu_branch_layout_t* active_branch_layout = NULL;
  if (iree_status_is_ok(status) && branch_layout.island_count != 0) {
    if (branch_layout.byte_length > IREE_HOST_SIZE_MAX) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU relaxed branch layout exceeds host address range");
    } else {
      final_byte_length = (iree_host_size_t)branch_layout.byte_length;
      active_branch_layout = &branch_layout;
    }
  }

  uint8_t* data = NULL;
  if (iree_status_is_ok(status) && final_byte_length != 0) {
    status = iree_arena_allocate(arena, final_byte_length, (void**)&data);
  }
  loom_amdgpu_hsaco_text_fixup_t* text_fixups = NULL;
  if (iree_status_is_ok(status) && sizing_state.text_fixups.count != 0) {
    status =
        iree_arena_allocate_array(arena, sizing_state.text_fixups.count,
                                  sizeof(text_fixups[0]), (void**)&text_fixups);
  }
  iree_host_size_t native_insertion_capacity =
      sizing_state.native_insertions.count;
  if (active_branch_layout != NULL &&
      iree_all_bits_set(
          flags,
          LOOM_AMDGPU_ENCODE_INSTRUCTION_STREAM_FLAG_CAPTURE_NATIVE_INSERTIONS)) {
    IREE_ASSERT_LE(active_branch_layout->group_count,
                   IREE_HOST_SIZE_MAX - native_insertion_capacity);
    native_insertion_capacity += active_branch_layout->group_count;
    IREE_ASSERT_LE(active_branch_layout->island_count,
                   IREE_HOST_SIZE_MAX - native_insertion_capacity);
    native_insertion_capacity += active_branch_layout->island_count;
  }
  loom_amdgpu_native_insertion_t* native_insertions = NULL;
  if (iree_status_is_ok(status) && native_insertion_capacity != 0) {
    status = iree_arena_allocate_array(arena, native_insertion_capacity,
                                       sizeof(native_insertions[0]),
                                       (void**)&native_insertions);
  }
  loom_amdgpu_encode_state_t writing_state = {
      .inputs = inputs,
      .packet_plan = packet_plan_state,
      .stream =
          {
              .data = data,
              .capacity = final_byte_length,
          },
      .text_fixups =
          {
              .values = text_fixups,
              .capacity = sizing_state.text_fixups.count,
          },
      .native_insertions =
          {
              .values = native_insertions,
              .capacity = native_insertion_capacity,
          },
      .branches =
          {
              .measurement =
                  {
                      .blocks = branch_blocks,
                  },
              .emission =
                  {
                      .layout = active_branch_layout,
                  },
          },
  };
  if (iree_status_is_ok(status) && final_byte_length != 0) {
    status = loom_amdgpu_encode_instruction_stream_into_state(&writing_state);
  }
  if (iree_status_is_ok(status)) {
    IREE_ASSERT_EQ(writing_state.stream.length, final_byte_length);
    IREE_ASSERT_EQ(writing_state.stream.instruction_count,
                   sizing_state.stream.instruction_count +
                       branch_layout.group_count + branch_layout.island_count);
    IREE_ASSERT_EQ(writing_state.text_fixups.count,
                   sizing_state.text_fixups.count);
    IREE_ASSERT_EQ(writing_state.native_insertions.count,
                   native_insertion_capacity);
    *out_stream = (loom_amdgpu_encoded_instruction_stream_t){
        .text = iree_make_const_byte_span(data, writing_state.stream.length),
        .instruction_count = writing_state.stream.instruction_count,
        .branch_layout = branch_layout,
        .text_fixups = text_fixups,
        .text_fixup_count = writing_state.text_fixups.count,
        .native_insertions = native_insertions,
        .native_insertion_count = writing_state.native_insertions.count,
    };
  }
  loom_low_allocation_release_value_scratch(&scratch);
  return status;
}

iree_status_t loom_amdgpu_encode_instruction_stream(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_const_byte_span_t* out_text, iree_arena_allocator_t* arena) {
  *out_text = iree_const_byte_span_empty();
  loom_amdgpu_encoded_instruction_stream_t stream = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_encode_instruction_stream_internal(
      schedule, allocation, NULL, &stream, arena));
  *out_text = stream.text;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_encode_instruction_stream_with_options(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_encode_instruction_stream_options_t* options,
    iree_const_byte_span_t* out_text, iree_arena_allocator_t* arena) {
  *out_text = iree_const_byte_span_empty();
  loom_amdgpu_encoded_instruction_stream_t stream = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_encode_instruction_stream_internal(
      schedule, allocation, options, &stream, arena));
  *out_text = stream.text;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_encode_instruction_stream_result(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    loom_amdgpu_encoded_instruction_stream_t* out_stream,
    iree_arena_allocator_t* arena) {
  return loom_amdgpu_encode_instruction_stream_internal(
      schedule, allocation, NULL, out_stream, arena);
}

iree_status_t loom_amdgpu_encode_instruction_stream_result_with_options(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_encode_instruction_stream_options_t* options,
    loom_amdgpu_encoded_instruction_stream_t* out_stream,
    iree_arena_allocator_t* arena) {
  return loom_amdgpu_encode_instruction_stream_internal(
      schedule, allocation, options, out_stream, arena);
}
