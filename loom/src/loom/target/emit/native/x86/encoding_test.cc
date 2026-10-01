// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/encoding.h"

#include <initializer_list>

#include "iree/testing/gtest.h"

namespace {

void ExpectEncoding(loom_x86_encoding_form_t form, uint16_t encoding_id,
                    const loom_x86_encoding_operands_t& operands,
                    std::initializer_list<uint8_t> bytes,
                    uint16_t written_registers) {
  loom_x86_encoded_instruction_t instruction;
  loom_x86_encode_instruction(form, encoding_id, &operands, &instruction);
  ASSERT_EQ(instruction.length, bytes.size());
  EXPECT_EQ(loom_x86_encoding_gpr_writes(form, &operands), written_registers);
  size_t index = 0;
  for (uint8_t byte : bytes) {
    EXPECT_EQ(instruction.bytes[index], byte) << "byte " << index;
    ++index;
  }
}

TEST(EncodingTest, RegisterDirectionAndWidth) {
  // ADD r9,r10 uses r/m as the result; IMUL r9,r10 uses reg as the result.
  loom_x86_encoding_operands_t operands = {};
  operands.result = 9;
  operands.inputs[0] = 9;
  operands.inputs[1] = 10;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_BINARY_RM_R,
                 0x01 | LOOM_X86_ENCODING_REX_W, operands, {0x4d, 0x01, 0xd1},
                 1u << 9);
  ExpectEncoding(LOOM_X86_ENCODING_FORM_BINARY_R_RM,
                 LOOM_X86_ENCODING_OPCODE_0F | 0xaf | LOOM_X86_ENCODING_REX_W,
                 operands, {0x4d, 0x0f, 0xaf, 0xca}, 1u << 9);
  ExpectEncoding(LOOM_X86_ENCODING_FORM_BINARY_RM_R, 0x01, operands,
                 {0x45, 0x01, 0xd1}, 1u << 9);
}

TEST(EncodingTest, ByteRegisterPrefixAndFullWidthDefinition) {
  // MOVZX esi,sil requires a REX prefix even without extended registers.
  loom_x86_encoding_operands_t operands = {};
  operands.result = 6;
  operands.inputs[0] = 6;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_MOVE,
                 LOOM_X86_ENCODING_OPCODE_0F | 0xb6 | LOOM_X86_ENCODING_BYTE,
                 operands, {0x40, 0x0f, 0xb6, 0xf6}, 1u << 6);
  // MOV esi,esi must still be emitted: it clears the high half of RSI.
  ExpectEncoding(LOOM_X86_ENCODING_FORM_MOVE, 0x8b, operands, {0x8b, 0xf6},
                 1u << 6);
}

TEST(EncodingTest, AddressDisplacementAndSib) {
  loom_x86_encoding_operands_t operands = {};
  operands.result = 9;
  operands.inputs[0] = 13;
  // [r13] needs an explicit zero displacement, unlike [r12].
  ExpectEncoding(LOOM_X86_ENCODING_FORM_LOAD, 0x8b | LOOM_X86_ENCODING_REX_W,
                 operands, {0x4d, 0x8b, 0x4d, 0x00}, 1u << 9);
  operands.inputs[0] = 12;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_LOAD, 0x8b | LOOM_X86_ENCODING_REX_W,
                 operands, {0x4d, 0x8b, 0x0c, 0x24}, 1u << 9);
  operands.immediate = -128;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_LOAD, 0x8b | LOOM_X86_ENCODING_REX_W,
                 operands, {0x4d, 0x8b, 0x4c, 0x24, 0x80}, 1u << 9);
  operands.immediate = 128;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_LOAD, 0x8b | LOOM_X86_ENCODING_REX_W,
                 operands, {0x4d, 0x8b, 0x8c, 0x24, 0x80, 0x00, 0x00, 0x00},
                 1u << 9);
}

TEST(EncodingTest, IndexedByteStoreAndNoBaseAddress) {
  loom_x86_encoding_operands_t operands = {};
  operands.inputs[0] = 7;
  operands.inputs[1] = 13;
  operands.inputs[2] = 12;
  operands.scale = 2;
  // MOV byte ptr [r13+r12*4],dil exercises all three REX roles.
  ExpectEncoding(LOOM_X86_ENCODING_FORM_STORE,
                 0x88 | LOOM_X86_ENCODING_BYTE | LOOM_X86_ENCODING_INDEXED,
                 operands, {0x43, 0x88, 0x7c, 0xa5, 0x00}, 0);
  operands.result = 9;
  operands.inputs[0] = 12;
  operands.immediate = 16;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_ADDRESS_SCALE,
                 0x8d | LOOM_X86_ENCODING_REX_W, operands,
                 {0x4e, 0x8d, 0x0c, 0xa5, 0x10, 0, 0, 0}, 1u << 9);
}

TEST(EncodingTest, PredicateSequenceAndImplicitWrites) {
  loom_x86_encoding_operands_t operands = {};
  operands.result = 7;
  operands.inputs[0] = 8;
  operands.inputs[1] = 9;
  // CMP r8,r9; SETB dil; MOVZX edi,dil defines a canonical full predicate.
  ExpectEncoding(
      LOOM_X86_ENCODING_FORM_COMPARE, 0x92 | LOOM_X86_ENCODING_REX_W, operands,
      {0x4d, 0x39, 0xc8, 0x40, 0x0f, 0x92, 0xc7, 0x40, 0x0f, 0xb6, 0xff},
      1u << 7);
  operands.result = 2;
  operands.inputs[0] = 0;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_MULTIPLY_HIGH,
                 0xf7 | (4u << 9) | LOOM_X86_ENCODING_REX_W, operands,
                 {0x49, 0xf7, 0xe1}, (1u << 0) | (1u << 2));
}

TEST(EncodingTest, ImmediateAndStackEncoding) {
  loom_x86_encoding_operands_t operands = {};
  operands.result = 8;
  operands.immediate = INT64_C(0x123456789abcdef0);
  ExpectEncoding(
      LOOM_X86_ENCODING_FORM_CONSTANT, 0xb8 | LOOM_X86_ENCODING_REX_W, operands,
      {0x49, 0xb8, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12}, 1u << 8);
  operands.inputs[0] = 12;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_PUSH, 0, operands, {0x41, 0x54},
                 1u << 4);
  ExpectEncoding(LOOM_X86_ENCODING_FORM_POP, 0, operands, {0x41, 0x5c},
                 (1u << 4) | (1u << 12));
}

TEST(EncodingTest, ImmediateWidthsAndCountRegister) {
  loom_x86_encoding_operands_t operands = {};
  operands.result = 9;
  operands.inputs[0] = 9;
  operands.inputs[1] = 1;
  operands.immediate = UINT32_MAX;
  // A 32-bit write represents this complete 64-bit constant without REX.W.
  ExpectEncoding(LOOM_X86_ENCODING_FORM_CONSTANT,
                 0xb8 | LOOM_X86_ENCODING_REX_W, operands,
                 {0x41, 0xb9, 0xff, 0xff, 0xff, 0xff}, 1u << 9);
  operands.immediate = INT32_MIN;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_MULTIPLY_IMMEDIATE,
                 0x69 | LOOM_X86_ENCODING_REX_W, operands,
                 {0x4d, 0x69, 0xc9, 0x00, 0x00, 0x00, 0x80}, 1u << 9);
  ExpectEncoding(LOOM_X86_ENCODING_FORM_SHIFT_COUNT,
                 0xd3 | (7u << 9) | LOOM_X86_ENCODING_REX_W, operands,
                 {0x49, 0xd3, 0xf9}, 1u << 9);
  operands.immediate = 63;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_SHIFT_IMMEDIATE,
                 0xc1 | (5u << 9) | LOOM_X86_ENCODING_REX_W, operands,
                 {0x49, 0xc1, 0xe9, 0x3f}, 1u << 9);
}

TEST(EncodingTest, ConditionalSequences) {
  loom_x86_encoding_operands_t operands = {};
  operands.result = 8;
  operands.inputs[0] = 1;
  operands.inputs[1] = 9;
  operands.inputs[2] = 8;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_SELECT,
                 LOOM_X86_ENCODING_OPCODE_0F | 0x45 | LOOM_X86_ENCODING_REX_W,
                 operands, {0x85, 0xc9, 0x4d, 0x0f, 0x45, 0xc1}, 1u << 8);
  // SUB changes flags before CMOVB restores the original value on borrow.
  operands.inputs[0] = 7;
  operands.immediate = 3;
  ExpectEncoding(
      LOOM_X86_ENCODING_FORM_SUBTRACT_IF_UGE,
      LOOM_X86_ENCODING_OPCODE_0F | 0x42, operands,
      {0x44, 0x8b, 0xc7, 0x41, 0x81, 0xe8, 3, 0, 0, 0, 0x44, 0x0f, 0x42, 0xc7},
      1u << 8);
}

TEST(EncodingTest, WordStoreAndSignedExtension) {
  loom_x86_encoding_operands_t operands = {};
  operands.inputs[0] = 15;
  operands.inputs[1] = 13;
  operands.immediate = 8;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_STORE,
                 0x89 | LOOM_X86_ENCODING_OPERAND_16, operands,
                 {0x66, 0x45, 0x89, 0x7d, 0x08}, 0);
  operands.result = 9;
  operands.inputs[0] = 10;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_MOVE, 0x63 | LOOM_X86_ENCODING_REX_W,
                 operands, {0x4d, 0x63, 0xca}, 1u << 9);
}

TEST(EncodingTest, FullWidthBranchAndRelativeDisplacements) {
  loom_x86_encoding_operands_t operands = {};
  operands.inputs[0] = 8;
  operands.immediate = 0x12345678;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_BRANCH_ZERO, LOOM_X86_ENCODING_REX_W,
                 operands,
                 {0x4d, 0x85, 0xc0, 0x0f, 0x84, 0x78, 0x56, 0x34, 0x12}, 0);
  operands.inputs[0] = 1;
  operands.immediate = -6;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_BRANCH_NONZERO, 0, operands,
                 {0x85, 0xc9, 0x0f, 0x85, 0xfa, 0xff, 0xff, 0xff}, 0);
  ExpectEncoding(LOOM_X86_ENCODING_FORM_JUMP, 0, operands,
                 {0xe9, 0xfa, 0xff, 0xff, 0xff}, 0);
  ExpectEncoding(LOOM_X86_ENCODING_FORM_RETURN, 0, operands, {0xc3}, 1u << 4);
}

}  // namespace
