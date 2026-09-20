// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/function.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/ir/context.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/x86/descriptors/encoding_defs.h"
#include "loom/target/arch/x86/register_classes.h"
#include "loom/target/arch/x86/sysv_frame.h"

#define LOOM_X86_FUNCTION_PLAN_NONE UINT32_MAX
#define LOOM_X86_FUNCTION_REGISTER_CLASS_UNKNOWN UINT8_MAX

typedef struct loom_x86_function_stream_t {
  uint8_t* data;
  iree_host_size_t capacity;
  iree_host_size_t length;
} loom_x86_function_stream_t;

typedef struct loom_x86_memory_operand_t {
  int32_t displacement;
  uint32_t base_register;
  uint32_t index_register;
  uint8_t scale_log2;
  bool has_base;
  bool has_index;
} loom_x86_memory_operand_t;

typedef struct loom_x86_function_encode_state_t {
  const loom_low_emission_frame_t* frame;
  const loom_x86_sysv_frame_plan_t* plan;
  loom_x86_function_stream_t stream;
  uint64_t* block_offsets;
  uint32_t* call_plan_indices;
  uint32_t* return_plan_indices;
  loom_string_id_t* immediate_name_ids;
  uint8_t* register_class_kinds;
  loom_x86_function_call_fixup_t* call_fixups;
  iree_host_size_t next_call_fixup;
} loom_x86_function_encode_state_t;

static void loom_x86_function_append_u8(loom_x86_function_encode_state_t* state,
                                        uint8_t value) {
  if (state->stream.data != NULL) {
    IREE_ASSERT_LT(state->stream.length, state->stream.capacity);
    state->stream.data[state->stream.length] = value;
  }
  ++state->stream.length;
}

static void loom_x86_function_append_u32(
    loom_x86_function_encode_state_t* state, uint32_t value) {
  if (state->stream.data != NULL) {
    IREE_ASSERT(state->stream.length <= state->stream.capacity &&
                sizeof(value) <= state->stream.capacity - state->stream.length);
    uint8_t* target = state->stream.data + state->stream.length;
    target[0] = (uint8_t)value;
    target[1] = (uint8_t)(value >> 8);
    target[2] = (uint8_t)(value >> 16);
    target[3] = (uint8_t)(value >> 24);
  }
  state->stream.length += sizeof(value);
}

static void loom_x86_function_append_u64(
    loom_x86_function_encode_state_t* state, uint64_t value) {
  loom_x86_function_append_u32(state, (uint32_t)value);
  loom_x86_function_append_u32(state, (uint32_t)(value >> 32));
}

static void loom_x86_function_append_rex(
    loom_x86_function_encode_state_t* state, bool wide, bool reg_extension,
    bool index_extension, bool base_extension, bool force) {
  const uint8_t rex =
      (uint8_t)(0x40u | ((uint8_t)wide << 3) | ((uint8_t)reg_extension << 2) |
                ((uint8_t)index_extension << 1) | (uint8_t)base_extension);
  if (rex != 0x40u || force) {
    loom_x86_function_append_u8(state, rex);
  }
}

static void loom_x86_function_append_opcode(
    loom_x86_function_encode_state_t* state, uint16_t opcode) {
  if (opcode > UINT8_MAX) {
    loom_x86_function_append_u8(state, (uint8_t)(opcode >> 8));
  }
  loom_x86_function_append_u8(state, (uint8_t)opcode);
}

static iree_status_t loom_x86_function_require_register(
    uint32_t physical_register) {
  if (physical_register >= 16) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 GPR index %" PRIu32 " is out of range",
                            physical_register);
  }
  return iree_ok_status();
}

static void loom_x86_function_encode_register_modrm(
    loom_x86_function_encode_state_t* state, uint16_t opcode, bool wide,
    uint32_t reg_field, uint32_t rm_register, bool force_rex) {
  loom_x86_function_append_rex(state, wide, reg_field >= 8, false,
                               rm_register >= 8, force_rex);
  loom_x86_function_append_opcode(state, opcode);
  loom_x86_function_append_u8(
      state, (uint8_t)(0xC0u | ((reg_field & 7u) << 3) | (rm_register & 7u)));
}

static bool loom_x86_function_displacement_fits_i8(int32_t displacement) {
  return displacement >= INT8_MIN && displacement <= INT8_MAX;
}

static void loom_x86_function_encode_memory_modrm(
    loom_x86_function_encode_state_t* state, uint16_t opcode, bool wide,
    uint32_t reg_field, const loom_x86_memory_operand_t* memory,
    bool force_rex) {
  const uint32_t base_low = memory->base_register & 7u;
  uint8_t mod = 0;
  uint8_t displacement_size = 0;
  if (!memory->has_base) {
    displacement_size = 4;
  } else if (memory->displacement == 0 && base_low != 5u) {
    displacement_size = 0;
  } else if (loom_x86_function_displacement_fits_i8(memory->displacement)) {
    mod = 1;
    displacement_size = 1;
  } else {
    mod = 2;
    displacement_size = 4;
  }

  const bool has_sib =
      memory->has_index || (memory->has_base && base_low == 4u);
  const uint8_t rm = has_sib ? 4u : (memory->has_base ? (uint8_t)base_low : 5u);
  loom_x86_function_append_rex(state, wide, reg_field >= 8,
                               memory->has_index && memory->index_register >= 8,
                               memory->has_base && memory->base_register >= 8,
                               force_rex);
  loom_x86_function_append_opcode(state, opcode);
  loom_x86_function_append_u8(
      state, (uint8_t)((mod << 6) | ((reg_field & 7u) << 3) | rm));
  if (has_sib) {
    const uint8_t index =
        memory->has_index ? (uint8_t)(memory->index_register & 7u) : 4u;
    const uint8_t base = memory->has_base ? (uint8_t)base_low : 5u;
    loom_x86_function_append_u8(
        state, (uint8_t)((memory->scale_log2 << 6) | (index << 3) | base));
  }
  if (displacement_size == 1) {
    loom_x86_function_append_u8(state, (uint8_t)memory->displacement);
  } else if (displacement_size == 4) {
    loom_x86_function_append_u32(state, (uint32_t)memory->displacement);
  }
}

static iree_status_t loom_x86_function_register_class(
    loom_x86_function_encode_state_t* state, uint16_t descriptor_reg_class_id,
    loom_x86_register_class_t* out_register_class) {
  const loom_low_descriptor_set_t* descriptor_set =
      state->frame->target.descriptor_set;
  if (descriptor_reg_class_id >= descriptor_set->reg_class_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 register class ID %" PRIu16 " is out of range",
                            descriptor_reg_class_id);
  }
  uint8_t* cached_kind = &state->register_class_kinds[descriptor_reg_class_id];
  if (*cached_kind == LOOM_X86_FUNCTION_REGISTER_CLASS_UNKNOWN) {
    loom_x86_register_class_t register_class = 0;
    IREE_RETURN_IF_ERROR(loom_x86_descriptor_set_logical_register_class(
        descriptor_set, descriptor_reg_class_id, &register_class));
    *cached_kind = (uint8_t)register_class;
  }
  *out_register_class = (loom_x86_register_class_t)*cached_kind;
  return iree_ok_status();
}

static iree_status_t loom_x86_function_assignment_register(
    loom_x86_function_encode_state_t* state,
    const loom_low_allocation_assignment_t* assignment,
    uint32_t* out_physical_register,
    loom_x86_register_class_t* out_register_class) {
  if (assignment == NULL ||
      assignment->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
      assignment->location_count != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "x86 scalar encoding requires one physical-register assignment");
  }
  IREE_RETURN_IF_ERROR(
      loom_x86_function_require_register(assignment->location_base));
  IREE_RETURN_IF_ERROR(loom_x86_function_register_class(
      state, assignment->descriptor_reg_class_id, out_register_class));
  if (*out_register_class != LOOM_X86_REGISTER_CLASS_GPR32 &&
      *out_register_class != LOOM_X86_REGISTER_CLASS_GPR64) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 scalar encoding does not support register "
                            "class %u",
                            (unsigned)*out_register_class);
  }
  *out_physical_register = assignment->location_base;
  return iree_ok_status();
}

static iree_status_t loom_x86_function_packet_operand_register(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet, uint16_t operand_index,
    uint32_t* out_physical_register,
    loom_x86_register_class_t* out_register_class) {
  return loom_x86_function_assignment_register(
      state,
      loom_low_packet_operand_assignment(&state->frame->allocation, packet,
                                         operand_index),
      out_physical_register, out_register_class);
}

static iree_status_t loom_x86_function_packet_result_register(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet, uint16_t result_index,
    uint32_t* out_physical_register,
    loom_x86_register_class_t* out_register_class) {
  return loom_x86_function_assignment_register(
      state,
      loom_low_packet_result_assignment(&state->frame->allocation, packet,
                                        result_index),
      out_physical_register, out_register_class);
}

static iree_status_t loom_x86_function_read_immediate(
    const loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet, uint16_t immediate_index,
    int64_t* out_value) {
  const loom_low_descriptor_set_t* descriptor_set =
      state->frame->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = packet->descriptor;
  IREE_ASSERT_LT(immediate_index, descriptor->immediate_count);
  const uint32_t immediate_row = descriptor->immediate_start + immediate_index;
  IREE_ASSERT_LT(immediate_row, descriptor_set->immediate_count);
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[immediate_row];
  const loom_string_id_t name_id = state->immediate_name_ids[immediate_row];
  const loom_named_attr_slice_t attrs = loom_low_packet_attrs(packet);
  if (name_id != LOOM_STRING_ID_INVALID) {
    for (iree_host_size_t i = 0; i < attrs.count; ++i) {
      const loom_named_attr_t* attr = &attrs.entries[i];
      if (attr->name_id != name_id) {
        continue;
      }
      if (attr->value.kind != LOOM_ATTR_I64) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "x86 immediate attribute must be i64");
      }
      *out_value = loom_attr_as_i64(attr->value);
      return iree_ok_status();
    }
  }
  if (iree_any_bit_set(immediate->flags,
                       LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE)) {
    *out_value = immediate->default_value;
    return iree_ok_status();
  }
  const iree_string_view_t name = loom_low_descriptor_set_string(
      descriptor_set, immediate->field_name_string_offset);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "x86 encoding requires immediate '%.*s'",
                          (int)name.size, name.data);
}

static iree_status_t loom_x86_function_scale_log2(int64_t scale,
                                                  uint8_t* out_scale_log2) {
  switch (scale) {
    case 1:
      *out_scale_log2 = 0;
      return iree_ok_status();
    case 2:
      *out_scale_log2 = 1;
      return iree_ok_status();
    case 4:
      *out_scale_log2 = 2;
      return iree_ok_status();
    case 8:
      *out_scale_log2 = 3;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "x86 address scale %" PRId64 " is unsupported",
                              scale);
  }
}

static iree_status_t loom_x86_function_packet_memory_operand(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet, uint16_t base_operand_index,
    uint16_t index_operand_index, int64_t displacement, int64_t scale,
    loom_x86_memory_operand_t* out_memory) {
  memset(out_memory, 0, sizeof(*out_memory));
  if (displacement < INT32_MIN || displacement > INT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 address displacement is outside i32");
  }
  out_memory->displacement = (int32_t)displacement;
  loom_x86_register_class_t register_class = 0;
  if (base_operand_index != UINT16_MAX) {
    IREE_RETURN_IF_ERROR(loom_x86_function_packet_operand_register(
        state, packet, base_operand_index, &out_memory->base_register,
        &register_class));
    if (register_class != LOOM_X86_REGISTER_CLASS_GPR64) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "x86 address base must use a GPR64");
    }
    out_memory->has_base = true;
  }
  if (index_operand_index != UINT16_MAX) {
    IREE_RETURN_IF_ERROR(loom_x86_function_packet_operand_register(
        state, packet, index_operand_index, &out_memory->index_register,
        &register_class));
    if (register_class != LOOM_X86_REGISTER_CLASS_GPR64) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "x86 address index must use a GPR64");
    }
    if (out_memory->index_register == LOOM_X86_SYSV_GPR_RSP) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "rsp cannot be an x86 address index");
    }
    out_memory->has_index = true;
  }
  return loom_x86_function_scale_log2(scale, &out_memory->scale_log2);
}

static iree_status_t loom_x86_function_encode_binary_packet(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet, bool reg_rm) {
  uint32_t result_register = 0;
  loom_x86_register_class_t result_class = 0;
  IREE_RETURN_IF_ERROR(loom_x86_function_packet_result_register(
      state, packet, 0, &result_register, &result_class));
  const uint16_t source_operand_index = packet->node->operand_count - 1u;
  uint32_t source_register = 0;
  loom_x86_register_class_t source_class = 0;
  IREE_RETURN_IF_ERROR(loom_x86_function_packet_operand_register(
      state, packet, source_operand_index, &source_register, &source_class));
  const bool force_32_bit = iree_any_bit_set(packet->descriptor->encoding_id,
                                             LOOM_X86_ENCODING_ID_FORCE_32_BIT);
  const bool wide =
      result_class == LOOM_X86_REGISTER_CLASS_GPR64 && !force_32_bit;
  const uint16_t opcode = (uint16_t)(packet->descriptor->encoding_id &
                                     LOOM_X86_ENCODING_ID_OPCODE_MASK);
  if (reg_rm) {
    loom_x86_function_encode_register_modrm(
        state, opcode, wide, result_register, source_register, false);
  } else {
    loom_x86_function_encode_register_modrm(
        state, opcode, wide, source_register, result_register, false);
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_function_encode_imul_immediate_packet(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet) {
  uint32_t result_register = 0;
  loom_x86_register_class_t result_class = 0;
  IREE_RETURN_IF_ERROR(loom_x86_function_packet_result_register(
      state, packet, 0, &result_register, &result_class));
  uint32_t source_register = 0;
  loom_x86_register_class_t source_class = 0;
  IREE_RETURN_IF_ERROR(loom_x86_function_packet_operand_register(
      state, packet, 0, &source_register, &source_class));
  int64_t immediate = 0;
  IREE_RETURN_IF_ERROR(
      loom_x86_function_read_immediate(state, packet, 0, &immediate));
  if (immediate < INT32_MIN || immediate > INT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 imul immediate is outside i32");
  }
  loom_x86_function_encode_register_modrm(
      state, packet->descriptor->encoding_id,
      result_class == LOOM_X86_REGISTER_CLASS_GPR64, result_register,
      source_register, false);
  loom_x86_function_append_u32(state, (uint32_t)immediate);
  return iree_ok_status();
}

static iree_status_t loom_x86_function_encode_group_immediate_packet(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet) {
  uint32_t result_register = 0;
  loom_x86_register_class_t result_class = 0;
  IREE_RETURN_IF_ERROR(loom_x86_function_packet_result_register(
      state, packet, 0, &result_register, &result_class));
  int64_t immediate = 0;
  IREE_RETURN_IF_ERROR(
      loom_x86_function_read_immediate(state, packet, 0, &immediate));
  const bool use_imm8 = packet->descriptor->encoding_format_id ==
                        LOOM_X86_ENCODING_FORMAT_RM_IMM8;
  if (use_imm8) {
    if (immediate < 0 || immediate > UINT8_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "x86 group immediate is outside u8");
    }
  }
  const uint16_t encoding_id = packet->descriptor->encoding_id;
  const uint32_t modrm_extension = encoding_id >> 8;
  loom_x86_function_encode_register_modrm(
      state, (uint8_t)encoding_id,
      result_class == LOOM_X86_REGISTER_CLASS_GPR64, modrm_extension,
      result_register, false);
  if (use_imm8) {
    loom_x86_function_append_u8(state, (uint8_t)immediate);
  } else {
    loom_x86_function_append_u32(state, (uint32_t)immediate);
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_function_encode_group_register_packet(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet) {
  const uint16_t source_operand_index = packet->node->operand_count - 1u;
  uint32_t source_register = 0;
  loom_x86_register_class_t source_class = 0;
  IREE_RETURN_IF_ERROR(loom_x86_function_packet_operand_register(
      state, packet, source_operand_index, &source_register, &source_class));
  const uint16_t encoding_id = packet->descriptor->encoding_id;
  loom_x86_function_encode_register_modrm(
      state, (uint8_t)encoding_id,
      source_class == LOOM_X86_REGISTER_CLASS_GPR64, encoding_id >> 8,
      source_register, false);
  return iree_ok_status();
}

static iree_status_t loom_x86_function_encode_conditional_subtract_packet(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet) {
  uint32_t result_register = 0;
  loom_x86_register_class_t result_class = 0;
  IREE_RETURN_IF_ERROR(loom_x86_function_packet_result_register(
      state, packet, 0, &result_register, &result_class));
  uint32_t source_register = 0;
  loom_x86_register_class_t source_class = 0;
  IREE_RETURN_IF_ERROR(loom_x86_function_packet_operand_register(
      state, packet, 0, &source_register, &source_class));
  int64_t immediate = 0;
  IREE_RETURN_IF_ERROR(
      loom_x86_function_read_immediate(state, packet, 0, &immediate));
  loom_x86_function_encode_register_modrm(state, 0x89, false, source_register,
                                          result_register, false);
  const uint16_t encoding_id = packet->descriptor->encoding_id;
  loom_x86_function_encode_register_modrm(state, (uint8_t)encoding_id, false,
                                          encoding_id >> 8, result_register,
                                          false);
  loom_x86_function_append_u32(state, (uint32_t)immediate);
  loom_x86_function_encode_register_modrm(state, 0x0F42, false, result_register,
                                          source_register, false);
  return iree_ok_status();
}

static iree_status_t loom_x86_function_encode_select_packet(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet) {
  uint32_t result_register = 0;
  loom_x86_register_class_t result_class = 0;
  IREE_RETURN_IF_ERROR(loom_x86_function_packet_result_register(
      state, packet, 0, &result_register, &result_class));
  uint32_t condition_register = 0;
  loom_x86_register_class_t condition_class = 0;
  IREE_RETURN_IF_ERROR(loom_x86_function_packet_operand_register(
      state, packet, 0, &condition_register, &condition_class));
  uint32_t true_register = 0;
  loom_x86_register_class_t true_class = 0;
  IREE_RETURN_IF_ERROR(loom_x86_function_packet_operand_register(
      state, packet, 1, &true_register, &true_class));
  loom_x86_function_encode_register_modrm(
      state, 0x85, condition_class == LOOM_X86_REGISTER_CLASS_GPR64,
      condition_register, condition_register, false);
  loom_x86_function_encode_register_modrm(
      state, (uint16_t)(0x0F40u | packet->descriptor->encoding_id),
      result_class == LOOM_X86_REGISTER_CLASS_GPR64, result_register,
      true_register, false);
  return iree_ok_status();
}

static iree_status_t loom_x86_function_encode_compare_packet(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet) {
  uint32_t result_register = 0;
  loom_x86_register_class_t result_class = 0;
  IREE_RETURN_IF_ERROR(loom_x86_function_packet_result_register(
      state, packet, 0, &result_register, &result_class));
  uint32_t lhs_register = 0;
  loom_x86_register_class_t lhs_class = 0;
  IREE_RETURN_IF_ERROR(loom_x86_function_packet_operand_register(
      state, packet, 0, &lhs_register, &lhs_class));
  if (packet->descriptor->immediate_count != 0) {
    int64_t immediate = 0;
    IREE_RETURN_IF_ERROR(
        loom_x86_function_read_immediate(state, packet, 0, &immediate));
    loom_x86_function_encode_register_modrm(
        state, 0x81, lhs_class == LOOM_X86_REGISTER_CLASS_GPR64, 7,
        lhs_register, false);
    loom_x86_function_append_u32(state, (uint32_t)immediate);
  } else {
    uint32_t rhs_register = 0;
    loom_x86_register_class_t rhs_class = 0;
    IREE_RETURN_IF_ERROR(loom_x86_function_packet_operand_register(
        state, packet, 1, &rhs_register, &rhs_class));
    loom_x86_function_encode_register_modrm(
        state, 0x39, lhs_class == LOOM_X86_REGISTER_CLASS_GPR64, rhs_register,
        lhs_register, false);
  }
  loom_x86_function_encode_register_modrm(
      state, (uint16_t)(0x0F90u | packet->descriptor->encoding_id), false, 0,
      result_register, result_register >= 4 && result_register < 8);
  loom_x86_function_encode_register_modrm(
      state, 0x0FB6, false, result_register, result_register,
      result_register >= 4 && result_register < 8);
  return iree_ok_status();
}

static iree_status_t loom_x86_function_encode_immediate_packet(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet) {
  uint32_t result_register = 0;
  loom_x86_register_class_t result_class = 0;
  IREE_RETURN_IF_ERROR(loom_x86_function_packet_result_register(
      state, packet, 0, &result_register, &result_class));
  int64_t immediate = 0;
  IREE_RETURN_IF_ERROR(
      loom_x86_function_read_immediate(state, packet, 0, &immediate));
  const bool use_imm32 = result_class == LOOM_X86_REGISTER_CLASS_GPR32 ||
                         (uint64_t)immediate <= UINT32_MAX;
  loom_x86_function_append_rex(state, !use_imm32, false, false,
                               result_register >= 8, false);
  loom_x86_function_append_u8(state, (uint8_t)(packet->descriptor->encoding_id +
                                               (result_register & 7u)));
  if (use_imm32) {
    loom_x86_function_append_u32(state, (uint32_t)immediate);
  } else {
    loom_x86_function_append_u64(state, (uint64_t)immediate);
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_function_encode_memory_packet(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet, bool load) {
  const uint16_t value_operand_index = load ? UINT16_MAX : 0;
  const uint16_t base_operand_index = load ? 0 : 1;
  const uint16_t index_operand_index =
      packet->node->operand_count == (load ? 2 : 3)
          ? (uint16_t)(base_operand_index + 1u)
          : UINT16_MAX;
  int64_t displacement = 0;
  IREE_RETURN_IF_ERROR(
      loom_x86_function_read_immediate(state, packet, 0, &displacement));
  int64_t scale = 1;
  if (index_operand_index != UINT16_MAX) {
    IREE_RETURN_IF_ERROR(
        loom_x86_function_read_immediate(state, packet, 1, &scale));
  }
  loom_x86_memory_operand_t memory = {0};
  IREE_RETURN_IF_ERROR(loom_x86_function_packet_memory_operand(
      state, packet, base_operand_index, index_operand_index, displacement,
      scale, &memory));

  uint32_t value_register = 0;
  loom_x86_register_class_t value_class = 0;
  if (load) {
    IREE_RETURN_IF_ERROR(loom_x86_function_packet_result_register(
        state, packet, 0, &value_register, &value_class));
  } else {
    IREE_RETURN_IF_ERROR(loom_x86_function_packet_operand_register(
        state, packet, value_operand_index, &value_register, &value_class));
  }
  const bool byte_store = !load && packet->descriptor->encoding_id == 0x88u;
  loom_x86_function_encode_memory_modrm(
      state, packet->descriptor->encoding_id,
      value_class == LOOM_X86_REGISTER_CLASS_GPR64, value_register, &memory,
      byte_store && value_register >= 4 && value_register < 8);
  return iree_ok_status();
}

static iree_status_t loom_x86_function_encode_lea_packet(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet) {
  uint32_t result_register = 0;
  loom_x86_register_class_t result_class = 0;
  IREE_RETURN_IF_ERROR(loom_x86_function_packet_result_register(
      state, packet, 0, &result_register, &result_class));
  uint16_t base_operand_index = UINT16_MAX;
  uint16_t index_operand_index = UINT16_MAX;
  if (packet->node->operand_count == 2) {
    base_operand_index = 0;
    index_operand_index = 1;
  } else if (packet->descriptor->immediate_count == 2) {
    index_operand_index = 0;
  } else {
    base_operand_index = 0;
  }
  int64_t displacement = 0;
  if (packet->descriptor->immediate_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_x86_function_read_immediate(state, packet, 0, &displacement));
  }
  int64_t scale = 1;
  if (packet->descriptor->immediate_count == 2) {
    IREE_RETURN_IF_ERROR(
        loom_x86_function_read_immediate(state, packet, 1, &scale));
  }
  loom_x86_memory_operand_t memory = {0};
  IREE_RETURN_IF_ERROR(loom_x86_function_packet_memory_operand(
      state, packet, base_operand_index, index_operand_index, displacement,
      scale, &memory));
  loom_x86_function_encode_memory_modrm(
      state, packet->descriptor->encoding_id,
      result_class == LOOM_X86_REGISTER_CLASS_GPR64, result_register, &memory,
      false);
  return iree_ok_status();
}

static iree_status_t loom_x86_function_append_relative_branch(
    loom_x86_function_encode_state_t* state, uint16_t opcode,
    uint32_t target_block_index) {
  if (target_block_index >= state->frame->schedule.block_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 branch target block is out of range");
  }
  loom_x86_function_append_opcode(state, opcode);
  const uint64_t displacement_offset = state->stream.length;
  int64_t displacement = 0;
  if (state->stream.data != NULL) {
    displacement = (int64_t)state->block_offsets[target_block_index] -
                   (int64_t)(displacement_offset + 4u);
    if (displacement < INT32_MIN || displacement > INT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "x86 branch displacement is outside rel32");
    }
  }
  loom_x86_function_append_u32(state, (uint32_t)displacement);
  return iree_ok_status();
}

static iree_status_t loom_x86_function_encode_descriptor_packet(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet) {
  switch (packet->descriptor->encoding_format_id) {
    case LOOM_X86_ENCODING_FORMAT_RM_REG:
      return loom_x86_function_encode_binary_packet(state, packet, false);
    case LOOM_X86_ENCODING_FORMAT_REG_RM:
      return loom_x86_function_encode_binary_packet(state, packet, true);
    case LOOM_X86_ENCODING_FORMAT_REG_RM_IMM32:
      return loom_x86_function_encode_imul_immediate_packet(state, packet);
    case LOOM_X86_ENCODING_FORMAT_RM_IMM8:
    case LOOM_X86_ENCODING_FORMAT_RM_IMM32:
      return loom_x86_function_encode_group_immediate_packet(state, packet);
    case LOOM_X86_ENCODING_FORMAT_RM_GROUP:
      return loom_x86_function_encode_group_register_packet(state, packet);
    case LOOM_X86_ENCODING_FORMAT_CONDITIONAL_SUBTRACT:
      return loom_x86_function_encode_conditional_subtract_packet(state,
                                                                  packet);
    case LOOM_X86_ENCODING_FORMAT_SELECT:
      return loom_x86_function_encode_select_packet(state, packet);
    case LOOM_X86_ENCODING_FORMAT_COMPARE:
      return loom_x86_function_encode_compare_packet(state, packet);
    case LOOM_X86_ENCODING_FORMAT_MOV_IMMEDIATE:
      return loom_x86_function_encode_immediate_packet(state, packet);
    case LOOM_X86_ENCODING_FORMAT_MEMORY_REG_RM:
      return loom_x86_function_encode_memory_packet(state, packet, true);
    case LOOM_X86_ENCODING_FORMAT_MEMORY_RM_REG:
      return loom_x86_function_encode_memory_packet(state, packet, false);
    case LOOM_X86_ENCODING_FORMAT_LEA:
      return loom_x86_function_encode_lea_packet(state, packet);
    case LOOM_X86_ENCODING_FORMAT_DIRECT_BRANCH: {
      int64_t target_block = 0;
      IREE_RETURN_IF_ERROR(
          loom_x86_function_read_immediate(state, packet, 0, &target_block));
      if (target_block < 0 || target_block > UINT32_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "x86 direct branch target is out of range");
      }
      return loom_x86_function_append_relative_branch(
          state, packet->descriptor->encoding_id, (uint32_t)target_block);
    }
    default:
      break;
  }
  const iree_string_view_t key =
      loom_low_descriptor_set_string(state->frame->target.descriptor_set,
                                     packet->descriptor->key_string_offset);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "x86 native encoding for descriptor '%.*s' is "
                          "unavailable",
                          (int)key.size, key.data);
}

static iree_status_t loom_x86_function_frame_slot_offset(
    const loom_x86_function_encode_state_t* state, uint32_t slot_index,
    int32_t* out_offset) {
  IREE_ASSERT_LT(slot_index, state->plan->slot_count);
  const loom_x86_sysv_frame_slot_t* slot = &state->plan->slots[slot_index];
  uint64_t byte_offset = slot->byte_offset;
  if (slot->kind == LOOM_X86_SYSV_FRAME_SLOT_INCOMING_ARGUMENT) {
    if (state->plan->frame_size > UINT64_MAX - 8u ||
        state->plan->frame_size + 8u > UINT64_MAX - slot->byte_offset) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "x86 incoming argument offset overflows");
    }
    byte_offset = state->plan->frame_size + 8u + slot->byte_offset;
  }
  if (byte_offset > INT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 frame slot offset is outside disp32");
  }
  *out_offset = (int32_t)byte_offset;
  return iree_ok_status();
}

static iree_status_t loom_x86_function_encode_move(
    loom_x86_function_encode_state_t* state, const loom_low_move_t* move) {
  loom_x86_register_class_t destination_class = 0;
  IREE_RETURN_IF_ERROR(loom_x86_function_register_class(
      state, move->destination.descriptor_reg_class_id, &destination_class));
  loom_x86_register_class_t source_class = 0;
  IREE_RETURN_IF_ERROR(loom_x86_function_register_class(
      state, move->source.descriptor_reg_class_id, &source_class));
  if (destination_class != source_class) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "x86 native move between distinct register classes is unsupported");
  }
  if (destination_class != LOOM_X86_REGISTER_CLASS_GPR32 &&
      destination_class != LOOM_X86_REGISTER_CLASS_GPR64) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 native move for register class %u is "
                            "unsupported",
                            (unsigned)destination_class);
  }
  if (move->destination.location_kind == move->source.location_kind &&
      move->destination.location == move->source.location) {
    return iree_ok_status();
  }
  const bool wide = destination_class == LOOM_X86_REGISTER_CLASS_GPR64;
  const bool destination_is_register =
      move->destination.location_kind ==
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  const bool source_is_register =
      move->source.location_kind ==
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  if (destination_is_register && source_is_register) {
    IREE_RETURN_IF_ERROR(
        loom_x86_function_require_register(move->destination.location));
    IREE_RETURN_IF_ERROR(
        loom_x86_function_require_register(move->source.location));
    loom_x86_function_encode_register_modrm(state, 0x89, wide,
                                            move->source.location,
                                            move->destination.location, false);
    return iree_ok_status();
  }

  loom_x86_memory_operand_t memory = {
      .base_register = LOOM_X86_SYSV_GPR_RSP,
      .has_base = true,
  };
  if (destination_is_register &&
      move->source.location_kind == LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT) {
    IREE_RETURN_IF_ERROR(
        loom_x86_function_require_register(move->destination.location));
    IREE_RETURN_IF_ERROR(loom_x86_function_frame_slot_offset(
        state, move->source.location, &memory.displacement));
    loom_x86_function_encode_memory_modrm(
        state, 0x8B, wide, move->destination.location, &memory, false);
    return iree_ok_status();
  }
  if (source_is_register && move->destination.location_kind ==
                                LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT) {
    IREE_RETURN_IF_ERROR(
        loom_x86_function_require_register(move->source.location));
    IREE_RETURN_IF_ERROR(loom_x86_function_frame_slot_offset(
        state, move->destination.location, &memory.displacement));
    loom_x86_function_encode_memory_modrm(
        state, 0x89, wide, move->source.location, &memory, false);
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "x86 native stack-to-stack move is unsupported");
}

static iree_status_t loom_x86_function_encode_move_range(
    loom_x86_function_encode_state_t* state, const loom_low_move_t* moves,
    iree_host_size_t move_start, iree_host_size_t move_count) {
  for (iree_host_size_t i = 0; i < move_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_x86_function_encode_move(state, &moves[move_start + i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_function_adjust_stack(
    loom_x86_function_encode_state_t* state, bool subtract) {
  const uint64_t frame_size = state->plan->frame_size;
  if (frame_size == 0) {
    return iree_ok_status();
  }
  if (frame_size > INT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 frame size is outside signed imm32");
  }
  if (frame_size <= INT8_MAX) {
    loom_x86_function_append_u8(state, 0x48);
    loom_x86_function_append_u8(state, 0x83);
    loom_x86_function_append_u8(state, subtract ? 0xEC : 0xC4);
    loom_x86_function_append_u8(state, (uint8_t)frame_size);
  } else {
    loom_x86_function_append_u8(state, 0x48);
    loom_x86_function_append_u8(state, 0x81);
    loom_x86_function_append_u8(state, subtract ? 0xEC : 0xC4);
    loom_x86_function_append_u32(state, (uint32_t)frame_size);
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_function_encode_call(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet) {
  const uint32_t plan_index = state->call_plan_indices[packet->node_index];
  IREE_ASSERT_NE(plan_index, LOOM_X86_FUNCTION_PLAN_NONE);
  const loom_x86_sysv_call_plan_t* call = &state->plan->calls[plan_index];
  uint32_t move_start = call->move_start;
  IREE_RETURN_IF_ERROR(loom_x86_function_encode_move_range(
      state, state->plan->moves, move_start, call->save_move_count));
  move_start += call->save_move_count;
  IREE_RETURN_IF_ERROR(loom_x86_function_encode_move_range(
      state, state->plan->moves, move_start, call->argument_move_count));
  move_start += call->argument_move_count;

  loom_x86_function_append_u8(state, 0xE8);
  if (state->stream.data != NULL) {
    IREE_ASSERT_LT(state->next_call_fixup, state->plan->call_count);
    IREE_ASSERT_LE(state->stream.length, UINT32_MAX);
    state->call_fixups[state->next_call_fixup++] =
        (loom_x86_function_call_fixup_t){
            .text_offset = (uint32_t)state->stream.length,
            .target = loom_low_func_call_callee(packet->node->op),
        };
  }
  loom_x86_function_append_u32(state, 0);
  return loom_x86_function_encode_move_range(
      state, state->plan->moves, move_start,
      call->result_and_restore_move_count);
}

static iree_status_t loom_x86_function_encode_return(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet) {
  const uint32_t plan_index = state->return_plan_indices[packet->node_index];
  IREE_ASSERT_NE(plan_index, LOOM_X86_FUNCTION_PLAN_NONE);
  const loom_x86_sysv_return_plan_t* return_plan =
      &state->plan->returns[plan_index];
  IREE_RETURN_IF_ERROR(loom_x86_function_encode_move_range(
      state, state->plan->moves, return_plan->result_moves.start,
      return_plan->result_moves.count));
  IREE_RETURN_IF_ERROR(loom_x86_function_encode_move_range(
      state, state->plan->moves, state->plan->callee_restore_moves.start,
      state->plan->callee_restore_moves.count));
  IREE_RETURN_IF_ERROR(loom_x86_function_adjust_stack(state, false));
  loom_x86_function_append_u8(state, 0xC3);
  return iree_ok_status();
}

static iree_status_t loom_x86_function_encode_branch(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet) {
  const loom_op_t* op = packet->node->op;
  const loom_value_slice_t args = loom_low_br_args(op);
  if (args.count != 0) {
    const loom_low_allocation_edge_copy_group_t* group =
        loom_low_allocation_find_edge_copy_group_by_source_ordinal(
            &state->frame->allocation, packet->node->source_ordinal);
    IREE_ASSERT(group != NULL);
    IREE_RETURN_IF_ERROR(loom_x86_function_encode_move_range(
        state, state->frame->allocation.moves, group->move_group.moves.start,
        group->move_group.moves.count));
  }
  const uint32_t target_block_index = loom_low_packet_block_index(
      &state->frame->schedule, loom_low_br_dest(op));
  IREE_ASSERT_NE(target_block_index, LOOM_LOW_PACKET_INDEX_NONE);
  if (target_block_index == packet->node->block_index + 1u) {
    return iree_ok_status();
  }
  return loom_x86_function_append_relative_branch(state, 0xE9,
                                                  target_block_index);
}

static iree_status_t loom_x86_function_encode_condition_test(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet) {
  uint32_t condition_register = 0;
  loom_x86_register_class_t condition_class = 0;
  IREE_RETURN_IF_ERROR(loom_x86_function_packet_operand_register(
      state, packet, 0, &condition_register, &condition_class));
  loom_x86_function_encode_register_modrm(
      state, 0x85, condition_class == LOOM_X86_REGISTER_CLASS_GPR64,
      condition_register, condition_register, false);
  return iree_ok_status();
}

static iree_status_t loom_x86_function_encode_cond_branch(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet) {
  IREE_RETURN_IF_ERROR(loom_x86_function_encode_condition_test(state, packet));
  const loom_op_t* op = packet->node->op;
  const uint32_t current_block_index = packet->node->block_index;
  const uint32_t true_block_index = loom_low_packet_block_index(
      &state->frame->schedule, loom_low_cond_br_true_dest(op));
  const uint32_t false_block_index = loom_low_packet_block_index(
      &state->frame->schedule, loom_low_cond_br_false_dest(op));
  IREE_ASSERT_NE(true_block_index, LOOM_LOW_PACKET_INDEX_NONE);
  IREE_ASSERT_NE(false_block_index, LOOM_LOW_PACKET_INDEX_NONE);
  if (true_block_index == false_block_index) {
    if (true_block_index == current_block_index + 1u) {
      return iree_ok_status();
    }
    return loom_x86_function_append_relative_branch(state, 0xE9,
                                                    true_block_index);
  }
  if (true_block_index == current_block_index + 1u) {
    return loom_x86_function_append_relative_branch(state, 0x0F84,
                                                    false_block_index);
  }
  IREE_RETURN_IF_ERROR(loom_x86_function_append_relative_branch(
      state, 0x0F85, true_block_index));
  if (false_block_index != current_block_index + 1u) {
    IREE_RETURN_IF_ERROR(loom_x86_function_append_relative_branch(
        state, 0xE9, false_block_index));
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_function_encode_packet_moves(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet) {
  const loom_low_allocation_packet_move_group_t* group =
      loom_low_allocation_find_packet_move_group_by_source_ordinal(
          &state->frame->allocation, packet->node->source_ordinal);
  if (group == NULL) {
    return iree_ok_status();
  }
  return loom_x86_function_encode_move_range(
      state, state->frame->allocation.moves, group->move_group.moves.start,
      group->move_group.moves.count);
}

static iree_status_t loom_x86_function_encode_storage_transfer(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet) {
  const loom_op_t* op = packet->node->op;
  const bool is_store = loom_low_spill_isa(op);
  const loom_value_id_t storage_value =
      is_store ? loom_low_spill_storage(op) : loom_low_reload_storage(op);
  const uint64_t relative_offset =
      (uint64_t)(is_store ? loom_low_spill_offset(op)
                          : loom_low_reload_offset(op));
  loom_low_storage_layout_reference_t reference = {0};
  loom_low_storage_layout_lookup_reference(
      &state->frame->schedule.requirements.storage_layout, state->frame->module,
      storage_value, &reference);

  uint64_t byte_offset = state->plan->stack_storage_offset;
  if (!iree_checked_add_u64(byte_offset, reference.reservation.byte_offset,
                            &byte_offset) ||
      !iree_checked_add_u64(byte_offset, reference.byte_offset, &byte_offset) ||
      !iree_checked_add_u64(byte_offset, relative_offset, &byte_offset) ||
      byte_offset > INT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 stack storage offset is outside disp32");
  }

  const loom_low_allocation_assignment_t* assignment =
      is_store ? loom_low_packet_operand_assignment(&state->frame->allocation,
                                                    packet, 0)
               : loom_low_packet_result_assignment(&state->frame->allocation,
                                                   packet, 0);
  uint32_t physical_register = 0;
  loom_x86_register_class_t register_class = 0;
  IREE_RETURN_IF_ERROR(loom_x86_function_assignment_register(
      state, assignment, &physical_register, &register_class));
  const loom_x86_memory_operand_t memory = {
      .displacement = (int32_t)byte_offset,
      .base_register = LOOM_X86_SYSV_GPR_RSP,
      .has_base = true,
  };
  loom_x86_function_encode_memory_modrm(
      state, is_store ? 0x89 : 0x8B,
      register_class == LOOM_X86_REGISTER_CLASS_GPR64, physical_register,
      &memory, false);
  return iree_ok_status();
}

static iree_status_t loom_x86_function_encode_structural_packet(
    loom_x86_function_encode_state_t* state,
    const loom_low_packet_view_t* packet) {
  const loom_op_t* op = packet->node->op;
  if (loom_low_live_in_isa(op) || loom_low_storage_reserve_isa(op) ||
      loom_low_storage_view_isa(op)) {
    return iree_ok_status();
  }
  if (loom_low_copy_isa(op) || loom_low_move_isa(op) ||
      loom_low_slice_isa(op) || loom_low_concat_isa(op)) {
    return loom_x86_function_encode_packet_moves(state, packet);
  }
  if (loom_low_spill_isa(op) || loom_low_reload_isa(op)) {
    return loom_x86_function_encode_storage_transfer(state, packet);
  }
  if (loom_low_func_call_isa(op)) {
    return loom_x86_function_encode_call(state, packet);
  }
  if (loom_low_return_isa(op)) {
    return loom_x86_function_encode_return(state, packet);
  }
  if (loom_low_br_isa(op)) {
    return loom_x86_function_encode_branch(state, packet);
  }
  if (loom_low_cond_br_isa(op)) {
    return loom_x86_function_encode_cond_branch(state, packet);
  }
  const iree_string_view_t name = loom_op_name(state->frame->module, op);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "x86 native encoding for structural op '%.*s' is "
                          "unavailable",
                          (int)name.size, name.data);
}

static iree_status_t loom_x86_function_encode_body(
    loom_x86_function_encode_state_t* state) {
  state->stream.length = 0;
  state->next_call_fixup = 0;
  IREE_RETURN_IF_ERROR(loom_x86_function_adjust_stack(state, true));
  IREE_RETURN_IF_ERROR(loom_x86_function_encode_move_range(
      state, state->plan->moves, state->plan->callee_save_moves.start,
      state->plan->callee_save_moves.count));
  IREE_RETURN_IF_ERROR(loom_x86_function_encode_move_range(
      state, state->plan->moves, state->plan->entry_moves.start,
      state->plan->entry_moves.count));

  const loom_low_schedule_table_t* schedule = &state->frame->schedule;
  for (uint32_t block_index = 0; block_index < schedule->block_count;
       ++block_index) {
    if (state->stream.data == NULL) {
      state->block_offsets[block_index] = state->stream.length;
    } else {
      IREE_ASSERT_EQ(state->block_offsets[block_index], state->stream.length);
    }
    const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
    for (uint32_t ordinal = 0; ordinal < block->scheduled_node_count;
         ++ordinal) {
      const loom_low_packet_view_t packet =
          loom_low_packet_at_block_ordinal(schedule, block_index, ordinal);
      if (loom_low_packet_is_compile_time_only(&packet)) {
        continue;
      }
      if (packet.descriptor != NULL) {
        IREE_RETURN_IF_ERROR(
            loom_x86_function_encode_descriptor_packet(state, &packet));
      } else {
        IREE_RETURN_IF_ERROR(
            loom_x86_function_encode_structural_packet(state, &packet));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_function_prepare_state(
    const loom_low_emission_frame_t* frame,
    const loom_x86_sysv_frame_plan_t* plan,
    iree_arena_allocator_t* scratch_arena,
    loom_x86_function_encode_state_t* out_state) {
  memset(out_state, 0, sizeof(*out_state));
  out_state->frame = frame;
  out_state->plan = plan;
  const loom_low_schedule_table_t* schedule = &frame->schedule;
  const loom_low_descriptor_set_t* descriptor_set =
      frame->target.descriptor_set;
  if (schedule->block_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, schedule->block_count, sizeof(*out_state->block_offsets),
        (void**)&out_state->block_offsets));
  }
  if (schedule->node_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, schedule->node_count,
                                  sizeof(*out_state->call_plan_indices),
                                  (void**)&out_state->call_plan_indices));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, schedule->node_count,
                                  sizeof(*out_state->return_plan_indices),
                                  (void**)&out_state->return_plan_indices));
    memset(out_state->call_plan_indices, 0xFF,
           schedule->node_count * sizeof(*out_state->call_plan_indices));
    memset(out_state->return_plan_indices, 0xFF,
           schedule->node_count * sizeof(*out_state->return_plan_indices));
  }
  for (uint32_t i = 0; i < plan->call_count; ++i) {
    IREE_ASSERT_LT(plan->calls[i].node_index, schedule->node_count);
    out_state->call_plan_indices[plan->calls[i].node_index] = i;
  }
  for (uint32_t i = 0; i < plan->return_count; ++i) {
    IREE_ASSERT_LT(plan->returns[i].node_index, schedule->node_count);
    out_state->return_plan_indices[plan->returns[i].node_index] = i;
  }
  if (descriptor_set->immediate_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, descriptor_set->immediate_count,
        sizeof(*out_state->immediate_name_ids),
        (void**)&out_state->immediate_name_ids));
    for (uint32_t i = 0; i < descriptor_set->immediate_count; ++i) {
      const loom_low_immediate_t* immediate = &descriptor_set->immediates[i];
      const iree_string_view_t name = loom_low_descriptor_set_string(
          descriptor_set, immediate->field_name_string_offset);
      out_state->immediate_name_ids[i] =
          loom_module_lookup_string(frame->module, name);
    }
  }
  if (descriptor_set->reg_class_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, descriptor_set->reg_class_count,
        sizeof(*out_state->register_class_kinds),
        (void**)&out_state->register_class_kinds));
    memset(out_state->register_class_kinds,
           LOOM_X86_FUNCTION_REGISTER_CLASS_UNKNOWN,
           descriptor_set->reg_class_count *
               sizeof(*out_state->register_class_kinds));
  }
  return iree_ok_status();
}

iree_status_t loom_x86_encode_sysv_function(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_x86_function_encoding_t* out_encoding) {
  memset(out_encoding, 0, sizeof(*out_encoding));
  if (frame->allocation.spill_count != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "x86 native encoding requires a spill-free frame");
  }
  const iree_string_view_t target_key = loom_low_descriptor_set_string(
      frame->target.descriptor_set,
      frame->target.descriptor_set->target_key_string_offset);
  if (!iree_string_view_equal(target_key, IREE_SV("x86"))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "x86 encoder received target '%.*s'",
                            (int)target_key.size, target_key.data);
  }

  const iree_arena_checkpoint_t output_checkpoint =
      iree_arena_checkpoint_save(arena);
  iree_arena_allocator_t plan_arena;
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &plan_arena);
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  loom_x86_sysv_frame_plan_t plan = {0};
  loom_x86_function_encode_state_t state = {0};
  uint8_t* text = NULL;
  loom_x86_function_call_fixup_t* call_fixups = NULL;

  iree_status_t status =
      loom_x86_sysv_frame_plan_build(frame, &plan_arena, &scratch_arena, &plan);
  if (iree_status_is_ok(status)) {
    status =
        loom_x86_function_prepare_state(frame, &plan, &scratch_arena, &state);
  }
  if (iree_status_is_ok(status)) {
    status = loom_x86_function_encode_body(&state);
  }
  const iree_host_size_t text_length = state.stream.length;
  if (iree_status_is_ok(status) && text_length > INT32_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "x86 function exceeds signed 32-bit extent");
  }
  if (iree_status_is_ok(status) && text_length != 0) {
    status = iree_arena_allocate(arena, text_length, (void**)&text);
  }
  if (iree_status_is_ok(status) && plan.call_count != 0) {
    status = iree_arena_allocate_array(
        arena, plan.call_count, sizeof(*call_fixups), (void**)&call_fixups);
  }
  if (iree_status_is_ok(status)) {
    state.stream = (loom_x86_function_stream_t){
        .data = text,
        .capacity = text_length,
    };
    state.call_fixups = call_fixups;
    status = loom_x86_function_encode_body(&state);
  }
  if (iree_status_is_ok(status)) {
    IREE_ASSERT_EQ(state.stream.length, text_length);
    IREE_ASSERT_EQ(state.next_call_fixup, plan.call_count);
    out_encoding->text = iree_make_const_byte_span(text, text_length);
    out_encoding->call_fixups = call_fixups;
    out_encoding->call_fixup_count = plan.call_count;
  } else {
    iree_arena_checkpoint_restore(&output_checkpoint);
  }
  iree_arena_deinitialize(&scratch_arena);
  iree_arena_deinitialize(&plan_arena);
  return status;
}
