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

void ExpectVectorEncoding(uint16_t encoding_format_id, uint16_t encoding_id,
                          const loom_x86_encoding_operands_t& operands,
                          std::initializer_list<uint8_t> bytes) {
  loom_x86_encoded_instruction_t instruction;
  loom_x86_encode_instruction(encoding_format_id, encoding_id, &operands,
                              &instruction);
  ASSERT_EQ(instruction.length, bytes.size());
  size_t index = 0;
  for (uint8_t byte : bytes) {
    EXPECT_EQ(instruction.bytes[index], byte) << "byte " << index;
    ++index;
  }
}

TEST(EncodingTest, VectorRegisterPrefixesAndBehaviors) {
  loom_x86_encoding_operands_t operands = {
      .result = 1,
  };
  operands.inputs[0] = 2;
  operands.inputs[1] = 3;
  // VEX.128 vpaddd xmm1,xmm2,xmm3.
  ExpectVectorEncoding(0x8210, 0x05fe, operands, {0xc5, 0xe9, 0xfe, 0xcb});
  // EVEX.512 vpaddd zmm17,zmm18,zmm19 exercises both high extension bits.
  operands.result = 17;
  operands.inputs[0] = 18;
  operands.inputs[1] = 19;
  ExpectVectorEncoding(0x8210, 0xa5fe, operands,
                       {0x62, 0xa1, 0x6d, 0x40, 0xfe, 0xcb});

  // EVEX vpcmpd k1,xmm2,xmm3,1 owns a trailing immediate.
  operands = {};
  operands.result = 1;
  operands.inputs[0] = 2;
  operands.inputs[1] = 3;
  operands.immediate = 1;
  ExpectVectorEncoding(0x9210, 0x271f, operands,
                       {0x62, 0xf3, 0x6d, 0x08, 0x1f, 0xcb, 0x01});

  // EVEX vpblendmd xmm1{k2},xmm4,xmm3 sources its explicit writemask from
  // input 0 without making mask policy part of byte encoding.
  operands = {};
  operands.result = 1;
  operands.inputs[0] = 2;
  operands.inputs[1] = 3;
  operands.inputs[2] = 4;
  ExpectVectorEncoding(0xb230, 0x2664, operands,
                       {0x62, 0xf2, 0x5d, 0x0a, 0x64, 0xcb});

  // AVX512-FP16 uses the extended EVEX opcode maps without widening the
  // 16-bit instruction record. VADDPH uses map 5 and VFMADD231PH uses map 6.
  operands = {};
  operands.result = 1;
  operands.inputs[0] = 2;
  operands.inputs[1] = 3;
  ExpectVectorEncoding(0x8210, 0x8058, operands,
                       {0x62, 0xf5, 0x6c, 0x48, 0x58, 0xcb});
  operands.inputs[1] = 2;
  operands.inputs[2] = 3;
  ExpectVectorEncoding(0x8320, 0xa4b8, operands,
                       {0x62, 0xf6, 0x6d, 0x48, 0xb8, 0xcb});
}

TEST(EncodingTest, AvxVnniInt8FamilyHasExactReferenceBytes) {
  loom_x86_encoding_operands_t operands = {.result = 1};
  operands.inputs[0] = 1;
  operands.inputs[1] = 2;
  operands.inputs[2] = 3;

  ExpectVectorEncoding(0x8320, 0x0e50, operands,
                       {0xc4, 0xe2, 0x6b, 0x50, 0xcb});
  ExpectVectorEncoding(0x8320, 0x4e50, operands,
                       {0xc4, 0xe2, 0x6f, 0x50, 0xcb});
  ExpectVectorEncoding(0x8320, 0x0e51, operands,
                       {0xc4, 0xe2, 0x6b, 0x51, 0xcb});
  ExpectVectorEncoding(0x8320, 0x4e51, operands,
                       {0xc4, 0xe2, 0x6f, 0x51, 0xcb});
  ExpectVectorEncoding(0x8320, 0x0a50, operands,
                       {0xc4, 0xe2, 0x6a, 0x50, 0xcb});
  ExpectVectorEncoding(0x8320, 0x4a50, operands,
                       {0xc4, 0xe2, 0x6e, 0x50, 0xcb});
  ExpectVectorEncoding(0x8320, 0x0a51, operands,
                       {0xc4, 0xe2, 0x6a, 0x51, 0xcb});
  ExpectVectorEncoding(0x8320, 0x4a51, operands,
                       {0xc4, 0xe2, 0x6e, 0x51, 0xcb});
  ExpectVectorEncoding(0x8320, 0x0250, operands,
                       {0xc4, 0xe2, 0x68, 0x50, 0xcb});
  ExpectVectorEncoding(0x8320, 0x4250, operands,
                       {0xc4, 0xe2, 0x6c, 0x50, 0xcb});
  ExpectVectorEncoding(0x8320, 0x0251, operands,
                       {0xc4, 0xe2, 0x68, 0x51, 0xcb});
  ExpectVectorEncoding(0x8320, 0x4251, operands,
                       {0xc4, 0xe2, 0x6c, 0x51, 0xcb});
}

TEST(EncodingTest, AvxNeConvertFamilyHasExactReferenceBytes) {
  loom_x86_encoding_operands_t operands = {
      .result = 4,
  };
  operands.inputs[0] = 8;
  operands.inputs[1] = 9;
  operands.immediate = 0x1234;
  operands.scale = 2;

  ExpectVectorEncoding(0xc1a0, 0x0ab1, operands,
                       {0xc4, 0x82, 0x7a, 0xb1, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1a0, 0x4ab1, operands,
                       {0xc4, 0x82, 0x7e, 0xb1, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1a0, 0x06b1, operands,
                       {0xc4, 0x82, 0x79, 0xb1, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1a0, 0x46b1, operands,
                       {0xc4, 0x82, 0x7d, 0xb1, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1a0, 0x0ab0, operands,
                       {0xc4, 0x82, 0x7a, 0xb0, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1a0, 0x4ab0, operands,
                       {0xc4, 0x82, 0x7e, 0xb0, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1a0, 0x06b0, operands,
                       {0xc4, 0x82, 0x79, 0xb0, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1a0, 0x46b0, operands,
                       {0xc4, 0x82, 0x7d, 0xb0, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1a0, 0x0eb0, operands,
                       {0xc4, 0x82, 0x7b, 0xb0, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1a0, 0x4eb0, operands,
                       {0xc4, 0x82, 0x7f, 0xb0, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1a0, 0x02b0, operands,
                       {0xc4, 0x82, 0x78, 0xb0, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1a0, 0x42b0, operands,
                       {0xc4, 0x82, 0x7c, 0xb0, 0xa4, 0x88, 0x34, 0x12, 0, 0});

  operands = {};
  operands.result = 4;
  operands.inputs[0] = 5;
  ExpectVectorEncoding(0x8140, 0x0a72, operands,
                       {0xc4, 0xe2, 0x7a, 0x72, 0xe5});
  ExpectVectorEncoding(0x8140, 0x4a72, operands,
                       {0xc4, 0xe2, 0x7e, 0x72, 0xe5});
}

TEST(EncodingTest, VectorMemoryDisplacementsAndCanonicalSib) {
  loom_x86_encoding_operands_t operands = {
      .result = 1,
  };
  operands.inputs[0] = 0;
  operands.immediate = 16;
  // VEX vmovdqu xmm1,[rax+16].
  ExpectVectorEncoding(0xc1c0, 0x096f, operands,
                       {0xc5, 0xfa, 0x6f, 0x48, 0x10});

  // A high base forces VEX3 without changing the direct operand recipe.
  operands.result = 9;
  operands.inputs[0] = 8;
  ExpectVectorEncoding(0xc1c0, 0x096f, operands,
                       {0xc4, 0x41, 0x7a, 0x6f, 0x48, 0x10});

  // EVEX full-vector tuples compress a 64-byte displacement for ZMM.
  operands.result = 1;
  operands.inputs[0] = 0;
  operands.immediate = 64;
  ExpectVectorEncoding(0xc1c0, 0xa96f, operands,
                       {0x62, 0xf1, 0x7e, 0x48, 0x6f, 0x48, 0x01});

  // Map-5 VCVTPS2PHX is also EVEX despite its clear prefix/map extension bit.
  // Its ZMM source therefore compresses the same 64-byte displacement.
  ExpectVectorEncoding(0xc1c0, 0x841d, operands,
                       {0x62, 0xf5, 0x7d, 0x48, 0x1d, 0x48, 0x01});

  // R12 requires a SIB byte whose absent index has canonical scale zero.
  operands.inputs[0] = 12;
  operands.immediate = 0;
  ExpectVectorEncoding(0xc1c0, 0x096f, operands,
                       {0xc4, 0xc1, 0x7a, 0x6f, 0x0c, 0x24});
}

TEST(EncodingTest, EveryVectorRecipeHasExactReferenceBytes) {
  struct Reference {
    uint16_t encoding_format_id;
    uint16_t encoding_id;
    loom_x86_encoding_operands_t operands;
    uint8_t byte_count;
    uint8_t bytes[8];
  };
  static const Reference references[] = {
      {0x8000, 0x0157, {0, 1, {0, 0, 0}, 0}, 4, {0xc5, 0xf0, 0x57, 0xc9}},
      {0x8041, 0x057e, {0, 0, {2, 0, 0}, 0}, 4, {0xc5, 0xf9, 0x7e, 0xd0}},
      {0x8120,
       0xb68d,
       {0, 1, {2, 3, 0}, 0},
       6,
       {0x62, 0xf2, 0xe5, 0x48, 0x8d, 0xca}},
      {0x8140,
       0x2413,
       {0, 1, {2, 0, 0}, 0},
       6,
       {0x62, 0xf6, 0x7d, 0x08, 0x13, 0xca}},
      {0x8140,
       0x041d,
       {0, 1, {2, 0, 0}, 0},
       6,
       {0x62, 0xf5, 0x7d, 0x08, 0x1d, 0xca}},
      {0x8140, 0x056e, {0, 1, {1, 0, 0}, 0}, 4, {0xc5, 0xf9, 0x6e, 0xc9}},
      {0x8210, 0x4616, {0, 1, {2, 3, 0}, 0}, 5, {0xc4, 0xe2, 0x6d, 0x16, 0xcb}},
      {0x8320, 0x06b8, {0, 1, {1, 3, 4}, 0}, 5, {0xc4, 0xe2, 0x61, 0xb8, 0xcc}},
      {0x9041,
       0x0714,
       {1, 0, {2, 0, 0}, 0},
       6,
       {0xc4, 0xe3, 0x79, 0x14, 0xd0, 0x01}},
      {0x9105, 0x0573, {1, 1, {2, 0, 0}, 0}, 5, {0xc5, 0xf1, 0x73, 0xd2, 0x01}},
      {0x9106, 0x0573, {1, 1, {2, 0, 0}, 0}, 5, {0xc5, 0xf1, 0x73, 0xda, 0x01}},
      {0x9108, 0x0573, {1, 1, {2, 0, 0}, 0}, 5, {0xc5, 0xf1, 0x73, 0xf2, 0x01}},
      {0x9140, 0x05c5, {1, 0, {2, 0, 0}, 0}, 5, {0xc5, 0xf9, 0xc5, 0xc2, 0x01}},
      {0x9140,
       0x0d70,
       {0x1b, 1, {2, 0, 0}, 0},
       5,
       {0xc5, 0xfb, 0x70, 0xca, 0x1b}},
      {0x9210,
       0x0720,
       {1, 1, {2, 2, 0}, 0},
       6,
       {0xc4, 0xe3, 0x69, 0x20, 0xca, 0x01}},
      {0xa210,
       0x074b,
       {0, 1, {2, 3, 4}, 0},
       6,
       {0xc4, 0xe3, 0x69, 0x4b, 0xcb, 0x40}},
      {0xb230,
       0x2666,
       {0, 1, {2, 3, 4}, 0},
       6,
       {0x62, 0xf2, 0x5d, 0x0a, 0x66, 0xcb}},
      {0xc1a0,
       0x096f,
       {16, 1, {1, 2, 0}, 1},
       6,
       {0xc5, 0xfa, 0x6f, 0x4c, 0x51, 0x10}},
      {0xc1c0,
       0x0d10,
       {16, 1, {0, 0, 0}, 0},
       5,
       {0xc5, 0xfb, 0x10, 0x48, 0x10}},
      {0xc1c0,
       0x096f,
       {16, 1, {1, 0, 0}, 0},
       5,
       {0xc5, 0xfa, 0x6f, 0x49, 0x10}},
      {0xd2b1,
       0x097f,
       {16, 0, {1, 1, 2}, 1},
       6,
       {0xc5, 0xfa, 0x7f, 0x4c, 0x51, 0x10}},
      {0xd2c1,
       0x0d11,
       {16, 0, {1, 0, 0}, 0},
       5,
       {0xc5, 0xfb, 0x11, 0x48, 0x10}},
      {0xd2c1,
       0x097f,
       {16, 0, {1, 1, 0}, 0},
       5,
       {0xc5, 0xfa, 0x7f, 0x49, 0x10}},
      {0xe440,
       0x096f,
       {16, 1, {0, 0, 0}, 0},
       8,
       {0xc5, 0xfa, 0x6f, 0x0d, 0, 0, 0, 0}},
  };

  for (const Reference& reference : references) {
    loom_x86_encoded_instruction_t instruction;
    loom_x86_encode_instruction(reference.encoding_format_id,
                                reference.encoding_id, &reference.operands,
                                &instruction);
    ASSERT_EQ(instruction.length, reference.byte_count)
        << "recipe " << reference.encoding_format_id;
    for (uint8_t i = 0; i < reference.byte_count; ++i) {
      EXPECT_EQ(instruction.bytes[i], reference.bytes[i])
          << "recipe " << reference.encoding_format_id << " byte " << i;
    }
  }
}

TEST(EncodingTest, RegisterDirectionAndWidth) {
  // ADD r9,r10 uses r/m as the result; IMUL r9,r10 uses reg as the result.
  loom_x86_encoding_operands_t operands = {.result = 9};
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
  loom_x86_encoding_operands_t operands = {.result = 6};
  operands.inputs[0] = 6;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_MOVE,
                 LOOM_X86_ENCODING_OPCODE_0F | 0xb6 | LOOM_X86_ENCODING_BYTE,
                 operands, {0x40, 0x0f, 0xb6, 0xf6}, 1u << 6);
  // MOVSX uses the same byte-register prefix and defines the full GPR32.
  ExpectEncoding(LOOM_X86_ENCODING_FORM_MOVE,
                 LOOM_X86_ENCODING_OPCODE_0F | 0xbe | LOOM_X86_ENCODING_BYTE,
                 operands, {0x40, 0x0f, 0xbe, 0xf6}, 1u << 6);
  // The word source needs neither a byte-register REX prefix nor 16-bit result
  // selection; MOVSX still defines ESI.
  ExpectEncoding(LOOM_X86_ENCODING_FORM_MOVE,
                 LOOM_X86_ENCODING_OPCODE_0F | 0xbf, operands,
                 {0x0f, 0xbf, 0xf6}, 1u << 6);
  // MOV esi,esi must still be emitted: it clears the high half of RSI.
  ExpectEncoding(LOOM_X86_ENCODING_FORM_MOVE, 0x8b, operands, {0x8b, 0xf6},
                 1u << 6);
}

TEST(EncodingTest, AddressDisplacementAndSib) {
  loom_x86_encoding_operands_t operands = {
      .result = 9,
  };
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

TEST(EncodingTest, PcRelativeAddressUsesNoBaseRegisterOrSib) {
  loom_x86_encoding_operands_t operands = {
      .immediate = -4,
      .result = 13,
  };
  ExpectEncoding(LOOM_X86_ENCODING_FORM_ADDRESS_PC_RELATIVE,
                 0x8d | LOOM_X86_ENCODING_REX_W, operands,
                 {0x4c, 0x8d, 0x2d, 0xfc, 0xff, 0xff, 0xff}, 1u << 13);
  operands.result = 0;
  operands.immediate = 0x12345678;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_ADDRESS_PC_RELATIVE,
                 0x8d | LOOM_X86_ENCODING_REX_W, operands,
                 {0x48, 0x8d, 0x05, 0x78, 0x56, 0x34, 0x12}, 1u << 0);
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
  loom_x86_encoding_operands_t operands = {
      .result = 7,
  };
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
  loom_x86_encoding_operands_t operands = {
      .immediate = INT64_C(0x123456789abcdef0),
      .result = 8,
  };
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
  loom_x86_encoding_operands_t operands = {
      .result = 9,
  };
  operands.inputs[0] = 9;
  operands.inputs[1] = 1;
  // Exact float narrowing uses signed add/sub immediates at GPR32 width.
  operands.immediate = -1006108673;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE, 0x81, operands,
                 {0x41, 0x81, 0xc1, 0xff, 0xff, 0x07, 0xc4}, 1u << 9);
  operands.immediate = 1;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE, 0x81 | (5u << 9),
                 operands, {0x41, 0x81, 0xe9, 0x01, 0x00, 0x00, 0x00}, 1u << 9);
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
  loom_x86_encoding_operands_t operands = {
      .result = 8,
  };
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
  ExpectEncoding(LOOM_X86_ENCODING_FORM_CALL, 0, operands,
                 {0xe8, 0xfa, 0xff, 0xff, 0xff}, 1u << 4);
  operands.immediate = 0x12345678;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_CALL, 0, operands,
                 {0xe8, 0x78, 0x56, 0x34, 0x12}, 1u << 4);
  ExpectEncoding(LOOM_X86_ENCODING_FORM_RETURN, 0, operands, {0xc3}, 1u << 4);
}

}  // namespace
