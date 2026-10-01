// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/encoding.h"

static void loom_x86_encode_byte(loom_x86_encoded_instruction_t* instruction,
                                 uint8_t value) {
  instruction->bytes[instruction->length++] = value;
}

static void loom_x86_encode_integer(loom_x86_encoded_instruction_t* instruction,
                                    uint64_t value, uint8_t byte_count) {
  for (uint8_t i = 0; i < byte_count; ++i) {
    loom_x86_encode_byte(instruction, (uint8_t)(value >> (i * 8)));
  }
}

static void loom_x86_encode_prefix(loom_x86_encoded_instruction_t* instruction,
                                   uint16_t encoding_id, uint8_t reg,
                                   uint8_t base, uint8_t index,
                                   uint8_t byte_register) {
  if (encoding_id & LOOM_X86_ENCODING_OPERAND_16) {
    loom_x86_encode_byte(instruction, 0x66);
  }
  uint8_t rex = 0x40 | ((encoding_id & LOOM_X86_ENCODING_REX_W) ? 8 : 0) |
                ((reg >> 3) << 2) | ((index >> 3) << 1) | (base >> 3);
  // Without REX, byte register numbers 4..7 name AH..BH, not SPL..DIL.
  bool byte_rex = (encoding_id & LOOM_X86_ENCODING_BYTE) && byte_register >= 4;
  if (rex != 0x40 || byte_rex) {
    loom_x86_encode_byte(instruction, rex);
  }
}

static void loom_x86_encode_opcode(loom_x86_encoded_instruction_t* instruction,
                                   uint16_t encoding_id) {
  if (encoding_id & LOOM_X86_ENCODING_OPCODE_0F) {
    loom_x86_encode_byte(instruction, 0x0f);
  }
  loom_x86_encode_byte(instruction, (uint8_t)encoding_id);
}

static void loom_x86_encode_registers(
    loom_x86_encoded_instruction_t* instruction, uint16_t encoding_id,
    uint8_t reg, uint8_t base) {
  loom_x86_encode_prefix(instruction, encoding_id, reg, base, 0, base);
  loom_x86_encode_opcode(instruction, encoding_id);
  loom_x86_encode_byte(instruction, 0xc0 | ((reg & 7) << 3) | (base & 7));
}

// Values outside the architectural register range represent missing address
// components. They are private encoding state, never allocation locations.
enum { LOOM_X86_ADDRESS_REGISTER_NONE = 16 };

static void loom_x86_encode_memory(loom_x86_encoded_instruction_t* instruction,
                                   uint16_t encoding_id, uint8_t reg,
                                   uint8_t base, uint8_t index, uint8_t scale,
                                   int32_t displacement) {
  bool has_base = base != LOOM_X86_ADDRESS_REGISTER_NONE;
  bool has_index = index != LOOM_X86_ADDRESS_REGISTER_NONE;
  loom_x86_encode_prefix(instruction, encoding_id, reg, has_base ? base : 0,
                         has_index ? index : 0, reg);
  loom_x86_encode_opcode(instruction, encoding_id);
  uint8_t displacement_length =
      !has_base                                              ? 4
      : displacement == 0 && (base & 7) != 5                 ? 0
      : displacement >= INT8_MIN && displacement <= INT8_MAX ? 1
                                                             : 4;
  uint8_t mode = !has_base || displacement_length == 0 ? 0
                 : displacement_length == 1            ? 0x40
                                                       : 0x80;
  bool has_sib = has_index || !has_base || (base & 7) == 4;
  loom_x86_encode_byte(instruction,
                       mode | ((reg & 7) << 3) | (has_sib ? 4 : (base & 7)));
  if (has_sib) {
    loom_x86_encode_byte(instruction, (scale << 6) |
                                          ((has_index ? (index & 7) : 4) << 3) |
                                          (has_base ? (base & 7) : 5));
  }
  loom_x86_encode_integer(instruction, (uint32_t)displacement,
                          displacement_length);
}

static void loom_x86_encode_predicate(
    loom_x86_encoded_instruction_t* instruction, uint8_t condition_opcode,
    uint8_t result) {
  loom_x86_encode_registers(
      instruction,
      LOOM_X86_ENCODING_OPCODE_0F | condition_opcode | LOOM_X86_ENCODING_BYTE,
      0, result);
  // SETcc defines only a byte; predicate values own a complete GPR32.
  loom_x86_encode_registers(
      instruction,
      (LOOM_X86_ENCODING_OPCODE_0F | 0xb6) | LOOM_X86_ENCODING_BYTE, result,
      result);
}

uint16_t loom_x86_encoding_gpr_writes(
    loom_x86_encoding_form_t form,
    const loom_x86_encoding_operands_t* operands) {
  switch (form) {
    case LOOM_X86_ENCODING_FORM_MULTIPLY_HIGH:
      return (1u << 0) | (1u << 2);
    case LOOM_X86_ENCODING_FORM_STORE:
    case LOOM_X86_ENCODING_FORM_JUMP:
    case LOOM_X86_ENCODING_FORM_BRANCH_ZERO:
    case LOOM_X86_ENCODING_FORM_BRANCH_NONZERO:
      return 0;
    case LOOM_X86_ENCODING_FORM_PUSH:
    case LOOM_X86_ENCODING_FORM_RETURN:
      return 1u << 4;
    case LOOM_X86_ENCODING_FORM_POP:
      return (1u << 4) | (1u << operands->inputs[0]);
    default:
      return 1u << operands->result;
  }
}

void loom_x86_encode_instruction(
    loom_x86_encoding_form_t form, uint16_t encoding_id,
    const loom_x86_encoding_operands_t* operands,
    loom_x86_encoded_instruction_t* out_instruction) {
  loom_x86_encoded_instruction_t* instruction = out_instruction;
  instruction->length = 0;
  uint8_t result = operands->result;
  uint8_t lhs = operands->inputs[0];
  uint8_t rhs = operands->inputs[1];
  uint8_t extension = (encoding_id >> 9) & 7;
  uint16_t width_flags = encoding_id & LOOM_X86_ENCODING_REX_W;
  switch (form) {
    case LOOM_X86_ENCODING_FORM_BINARY_RM_R:
      loom_x86_encode_registers(instruction, encoding_id, rhs, result);
      break;
    case LOOM_X86_ENCODING_FORM_BINARY_R_RM:
      loom_x86_encode_registers(instruction, encoding_id, result, rhs);
      break;
    case LOOM_X86_ENCODING_FORM_MULTIPLY_HIGH:
      loom_x86_encode_registers(instruction, encoding_id, extension, rhs);
      return;
    case LOOM_X86_ENCODING_FORM_MULTIPLY_IMMEDIATE:
      loom_x86_encode_registers(instruction, encoding_id, result, lhs);
      loom_x86_encode_integer(instruction, operands->immediate, 4);
      break;
    case LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE:
      loom_x86_encode_registers(instruction, encoding_id, extension, result);
      loom_x86_encode_integer(instruction, operands->immediate, 4);
      break;
    case LOOM_X86_ENCODING_FORM_SHIFT_IMMEDIATE:
      loom_x86_encode_registers(instruction, encoding_id, extension, result);
      loom_x86_encode_byte(instruction, (uint8_t)operands->immediate);
      break;
    case LOOM_X86_ENCODING_FORM_SHIFT_COUNT:
      loom_x86_encode_registers(instruction, encoding_id, extension, result);
      break;
    case LOOM_X86_ENCODING_FORM_MOVE:
      loom_x86_encode_registers(instruction, encoding_id, result, lhs);
      break;
    case LOOM_X86_ENCODING_FORM_SELECT:
      loom_x86_encode_registers(instruction, 0x85, lhs, lhs);
      loom_x86_encode_registers(instruction, encoding_id, result, rhs);
      break;
    case LOOM_X86_ENCODING_FORM_SUBTRACT_IF_UGE:
      loom_x86_encode_registers(instruction, 0x8b, result, lhs);
      loom_x86_encode_registers(instruction, 0x81, 5, result);
      loom_x86_encode_integer(instruction, operands->immediate, 4);
      loom_x86_encode_registers(instruction, encoding_id, result, lhs);
      break;
    case LOOM_X86_ENCODING_FORM_COMPARE:
      loom_x86_encode_registers(instruction, 0x39 | width_flags, rhs, lhs);
      loom_x86_encode_predicate(instruction, (uint8_t)encoding_id, result);
      break;
    case LOOM_X86_ENCODING_FORM_COMPARE_IMMEDIATE:
      loom_x86_encode_registers(instruction, 0x81 | width_flags, 7, lhs);
      loom_x86_encode_integer(instruction, operands->immediate, 4);
      loom_x86_encode_predicate(instruction, (uint8_t)encoding_id, result);
      break;
    case LOOM_X86_ENCODING_FORM_CONSTANT:
      if ((uint64_t)operands->immediate <= UINT32_MAX) {
        encoding_id &= ~LOOM_X86_ENCODING_REX_W;
        width_flags = 0;
      }
      loom_x86_encode_prefix(instruction, encoding_id, 0, result, 0, 0);
      loom_x86_encode_byte(instruction, (uint8_t)encoding_id + (result & 7));
      loom_x86_encode_integer(instruction, operands->immediate,
                              width_flags ? 8 : 4);
      break;
    case LOOM_X86_ENCODING_FORM_LOAD:
      loom_x86_encode_memory(instruction, encoding_id, result, lhs,
                             (encoding_id & LOOM_X86_ENCODING_INDEXED)
                                 ? rhs
                                 : LOOM_X86_ADDRESS_REGISTER_NONE,
                             operands->scale, (int32_t)operands->immediate);
      break;
    case LOOM_X86_ENCODING_FORM_STORE:
      loom_x86_encode_memory(instruction, encoding_id, lhs, rhs,
                             (encoding_id & LOOM_X86_ENCODING_INDEXED)
                                 ? operands->inputs[2]
                                 : LOOM_X86_ADDRESS_REGISTER_NONE,
                             operands->scale, (int32_t)operands->immediate);
      return;
    case LOOM_X86_ENCODING_FORM_ADDRESS_ADD:
      loom_x86_encode_memory(instruction, encoding_id, result, lhs, rhs, 0, 0);
      break;
    case LOOM_X86_ENCODING_FORM_ADDRESS_DISPLACEMENT:
      loom_x86_encode_memory(instruction, encoding_id, result, lhs,
                             LOOM_X86_ADDRESS_REGISTER_NONE, 0,
                             (int32_t)operands->immediate);
      break;
    case LOOM_X86_ENCODING_FORM_ADDRESS_SCALE:
      loom_x86_encode_memory(instruction, encoding_id, result,
                             LOOM_X86_ADDRESS_REGISTER_NONE, lhs,
                             operands->scale, (int32_t)operands->immediate);
      break;
    case LOOM_X86_ENCODING_FORM_ADDRESS_ADD_SCALE:
      loom_x86_encode_memory(instruction, encoding_id, result, lhs, rhs,
                             operands->scale, (int32_t)operands->immediate);
      break;
    case LOOM_X86_ENCODING_FORM_BRANCH_NONZERO:
    case LOOM_X86_ENCODING_FORM_BRANCH_ZERO:
      loom_x86_encode_registers(instruction, 0x85 | width_flags, lhs, lhs);
      loom_x86_encode_opcode(
          instruction,
          LOOM_X86_ENCODING_OPCODE_0F |
              (form == LOOM_X86_ENCODING_FORM_BRANCH_ZERO ? 0x84 : 0x85));
      loom_x86_encode_integer(instruction, operands->immediate, 4);
      return;
    case LOOM_X86_ENCODING_FORM_JUMP:
      loom_x86_encode_byte(instruction, 0xe9);
      loom_x86_encode_integer(instruction, operands->immediate, 4);
      return;
    case LOOM_X86_ENCODING_FORM_PUSH:
    case LOOM_X86_ENCODING_FORM_POP:
      loom_x86_encode_prefix(instruction, 0, 0, lhs, 0, 0);
      loom_x86_encode_byte(
          instruction,
          (form == LOOM_X86_ENCODING_FORM_PUSH ? 0x50 : 0x58) | (lhs & 7));
      return;
    case LOOM_X86_ENCODING_FORM_RETURN:
      loom_x86_encode_byte(instruction, 0xc3);
      return;
    case LOOM_X86_ENCODING_FORM_NONE:
    default:
      IREE_ASSERT_UNREACHABLE("unsupported x86 native encoding form");
  }
}
