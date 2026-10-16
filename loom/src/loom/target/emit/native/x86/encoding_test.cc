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
                    uint16_t written_registers, uint8_t instruction_count = 1) {
  loom_x86_encoded_instruction_t instruction;
  loom_x86_encode_instruction(form, encoding_id, &operands, &instruction);
  ASSERT_EQ(instruction.length, bytes.size());
  EXPECT_EQ(instruction.instruction_count, instruction_count);
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
  EXPECT_EQ(instruction.instruction_count, 1);
  size_t index = 0;
  for (uint8_t byte : bytes) {
    EXPECT_EQ(instruction.bytes[index], byte) << "byte " << index;
    ++index;
  }
}

constexpr uint16_t VectorRecipe(uint8_t reg, uint8_t middle, uint8_t rm,
                                loom_x86_vector_encoding_behavior_t behavior) {
  return LOOM_X86_ENCODING_FORMAT_VECTOR | reg | (middle << 4) | (rm << 8) |
         (behavior << 12);
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

TEST(EncodingTest, IntegerToScalarFloat) {
  loom_x86_encoding_operands_t operands = {
      .result = 1,
      .inputs = {2, 11},
  };
  // VEX conversions read r11d/r11 with independent XMM upper-bit passthrough.
  ExpectVectorEncoding(0x8210, 0x092a, operands,
                       {0xc4, 0xc1, 0x6a, 0x2a, 0xcb});
  ExpectVectorEncoding(0x8210, 0x192a, operands,
                       {0xc4, 0xc1, 0xea, 0x2a, 0xcb});
  ExpectVectorEncoding(0x8210, 0x0d2a, operands,
                       {0xc4, 0xc1, 0x6b, 0x2a, 0xcb});
  ExpectVectorEncoding(0x8210, 0x1d2a, operands,
                       {0xc4, 0xc1, 0xeb, 0x2a, 0xcb});
  // EVEX unsigned conversions extend both destination and passthrough to
  // xmm17/xmm18 without treating the GPR source as a high SIMD register.
  operands.result = 17;
  operands.inputs[0] = 18;
  ExpectVectorEncoding(0x8210, 0x297b, operands,
                       {0x62, 0xc1, 0x6e, 0x00, 0x7b, 0xcb});
  ExpectVectorEncoding(0x8210, 0x397b, operands,
                       {0x62, 0xc1, 0xee, 0x00, 0x7b, 0xcb});
  ExpectVectorEncoding(0x8210, 0x2d7b, operands,
                       {0x62, 0xc1, 0x6f, 0x00, 0x7b, 0xcb});
  ExpectVectorEncoding(0x8210, 0x3d7b, operands,
                       {0x62, 0xc1, 0xef, 0x00, 0x7b, 0xcb});
}

TEST(EncodingTest, ScalarAndPackedFloatWidthConversions) {
  loom_x86_encoding_operands_t operands = {
      .result = 1,
      .inputs = {2, 3},
  };
  // Independent assembler bytes cover the scalar passthrough operand and the
  // packed conversion's asymmetric source and destination widths.
  ExpectVectorEncoding(0x8210, 0x095a, operands, {0xc5, 0xea, 0x5a, 0xcb});
  ExpectVectorEncoding(0x8140, 0x015a, operands, {0xc5, 0xf8, 0x5a, 0xca});
  ExpectVectorEncoding(0x8140, 0x415a, operands, {0xc5, 0xfc, 0x5a, 0xca});
  ExpectVectorEncoding(0x8210, 0x0d5a, operands, {0xc5, 0xeb, 0x5a, 0xcb});
  ExpectVectorEncoding(0x8140, 0x055a, operands, {0xc5, 0xf9, 0x5a, 0xca});
  ExpectVectorEncoding(0x8140, 0x455a, operands, {0xc5, 0xfd, 0x5a, 0xca});
  operands.result = 17;
  operands.inputs[0] = 18;
  operands.inputs[1] = 19;
  ExpectVectorEncoding(0x8210, 0x295a, operands,
                       {0x62, 0xa1, 0x6e, 0x00, 0x5a, 0xcb});
  ExpectVectorEncoding(0x8140, 0x215a, operands,
                       {0x62, 0xa1, 0x7c, 0x08, 0x5a, 0xca});
  ExpectVectorEncoding(0x8140, 0x615a, operands,
                       {0x62, 0xa1, 0x7c, 0x28, 0x5a, 0xca});
  ExpectVectorEncoding(0x8140, 0xa15a, operands,
                       {0x62, 0xa1, 0x7c, 0x48, 0x5a, 0xca});
  ExpectVectorEncoding(0x8210, 0x3d5a, operands,
                       {0x62, 0xa1, 0xef, 0x00, 0x5a, 0xcb});
  ExpectVectorEncoding(0x8140, 0x355a, operands,
                       {0x62, 0xa1, 0xfd, 0x08, 0x5a, 0xca});
  ExpectVectorEncoding(0x8140, 0x755a, operands,
                       {0x62, 0xa1, 0xfd, 0x28, 0x5a, 0xca});
  ExpectVectorEncoding(0x8140, 0xb55a, operands,
                       {0x62, 0xa1, 0xfd, 0x48, 0x5a, 0xca});
}

TEST(EncodingTest, LegacyHalfConversions) {
  // Independent LLVM assembler bytes cover both directions, every width, and
  // EVEX high registers. Narrowing reverses the ModRM source/result fields.
  loom_x86_encoding_operands_t operands = {.result = 1};
  operands.inputs[0] = 2;
  operands.immediate = 0;
  ExpectVectorEncoding(0x8140, 0x0613, operands,
                       {0xc4, 0xe2, 0x79, 0x13, 0xca});
  ExpectVectorEncoding(0x8140, 0x4613, operands,
                       {0xc4, 0xe2, 0x7d, 0x13, 0xca});
  ExpectVectorEncoding(0x9041, 0x071d, operands,
                       {0xc4, 0xe3, 0x79, 0x1d, 0xd1, 0x00});
  ExpectVectorEncoding(0x9041, 0x471d, operands,
                       {0xc4, 0xe3, 0x7d, 0x1d, 0xd1, 0x00});
  operands.result = 17;
  operands.inputs[0] = 18;
  ExpectVectorEncoding(0x8140, 0x2613, operands,
                       {0x62, 0xa2, 0x7d, 0x08, 0x13, 0xca});
  ExpectVectorEncoding(0x8140, 0x6613, operands,
                       {0x62, 0xa2, 0x7d, 0x28, 0x13, 0xca});
  ExpectVectorEncoding(0x8140, 0xa613, operands,
                       {0x62, 0xa2, 0x7d, 0x48, 0x13, 0xca});
  ExpectVectorEncoding(0x9041, 0x271d, operands,
                       {0x62, 0xa3, 0x7d, 0x08, 0x1d, 0xd1, 0x00});
  ExpectVectorEncoding(0x9041, 0x671d, operands,
                       {0x62, 0xa3, 0x7d, 0x28, 0x1d, 0xd1, 0x00});
  ExpectVectorEncoding(0x9041, 0xa71d, operands,
                       {0x62, 0xa3, 0x7d, 0x48, 0x1d, 0xd1, 0x00});
}

TEST(EncodingTest, PackedLowInterleaveWidths) {
  // LLVM's assembler distinguishes byte unpack (60) from word unpack (61).
  loom_x86_encoding_operands_t operands = {.result = 1, .inputs = {2, 3}};
  ExpectVectorEncoding(0x8210, 0x0560, operands, {0xc5, 0xe9, 0x60, 0xcb});
  ExpectVectorEncoding(0x8210, 0x0561, operands, {0xc5, 0xe9, 0x61, 0xcb});
  ExpectVectorEncoding(0x8210, 0x0562, operands, {0xc5, 0xe9, 0x62, 0xcb});
  ExpectVectorEncoding(0x8210, 0x056c, operands, {0xc5, 0xe9, 0x6c, 0xcb});
}

TEST(EncodingTest, EvexNdsMemoryBroadcastAndMaskModifiers) {
  // These bytes agree with independent LLVM and GNU assembler outputs.
  constexpr uint16_t kUnaryMerge = VectorRecipe(
      LOOM_X86_VECTOR_REGISTER_RESULT, LOOM_X86_VECTOR_REGISTER_NONE,
      LOOM_X86_VECTOR_REGISTER_INPUT_1, LOOM_X86_VECTOR_ENCODING_EVEX_MASK);
  constexpr uint16_t kUnaryZero = kUnaryMerge | 8;
  constexpr uint16_t kBinaryZero = VectorRecipe(
      LOOM_X86_VECTOR_REGISTER_RESULT | 8, LOOM_X86_VECTOR_REGISTER_INPUT_1,
      LOOM_X86_VECTOR_REGISTER_INPUT_2, LOOM_X86_VECTOR_ENCODING_EVEX_MASK);
  constexpr uint16_t kBinaryLoad = VectorRecipe(
      LOOM_X86_VECTOR_REGISTER_RESULT, LOOM_X86_VECTOR_REGISTER_INPUT_0 | 8,
      LOOM_X86_VECTOR_REGISTER_INPUT_1, LOOM_X86_VECTOR_ENCODING_LOAD);
  constexpr uint16_t kBinaryBroadcast = VectorRecipe(
      LOOM_X86_VECTOR_REGISTER_RESULT, LOOM_X86_VECTOR_REGISTER_INPUT_0,
      LOOM_X86_VECTOR_REGISTER_INPUT_1 | 8, LOOM_X86_VECTOR_ENCODING_LOAD);
  constexpr uint16_t kMaskedUnaryLoad = VectorRecipe(
      LOOM_X86_VECTOR_REGISTER_RESULT | 8, LOOM_X86_VECTOR_REGISTER_NONE | 8,
      LOOM_X86_VECTOR_REGISTER_INPUT_1,
      LOOM_X86_VECTOR_ENCODING_EVEX_MASK_LOAD);
  constexpr uint16_t kMaskedUnaryBroadcast = VectorRecipe(
      LOOM_X86_VECTOR_REGISTER_RESULT | 8, LOOM_X86_VECTOR_REGISTER_NONE,
      LOOM_X86_VECTOR_REGISTER_INPUT_1 | 8,
      LOOM_X86_VECTOR_ENCODING_EVEX_MASK_LOAD);
  constexpr uint16_t kMaskedBinaryLoad = VectorRecipe(
      LOOM_X86_VECTOR_REGISTER_RESULT | 8, LOOM_X86_VECTOR_REGISTER_INPUT_1 | 8,
      LOOM_X86_VECTOR_REGISTER_INPUT_2,
      LOOM_X86_VECTOR_ENCODING_EVEX_MASK_LOAD);
  constexpr uint16_t kMaskedBinaryBroadcast = VectorRecipe(
      LOOM_X86_VECTOR_REGISTER_RESULT | 8, LOOM_X86_VECTOR_REGISTER_INPUT_1,
      LOOM_X86_VECTOR_REGISTER_INPUT_2 | 8,
      LOOM_X86_VECTOR_ENCODING_EVEX_MASK_LOAD);
  constexpr uint16_t kUnaryEncoding = 0x2a72;
  constexpr uint16_t kBinaryEncoding = 0x2e72;

  loom_x86_encoding_operands_t operands = {
      .result = 17,
  };
  operands.inputs[0] = 2;
  operands.inputs[1] = 18;
  operands.inputs[2] = 19;
  operands.inputs[3] = 20;
  ExpectVectorEncoding(kUnaryMerge, kUnaryEncoding, operands,
                       {0x62, 0xa2, 0x7e, 0x0a, 0x72, 0xca});
  ExpectVectorEncoding(kUnaryZero, kUnaryEncoding, operands,
                       {0x62, 0xa2, 0x7e, 0x8a, 0x72, 0xca});
  ExpectVectorEncoding(kBinaryZero, kBinaryEncoding, operands,
                       {0x62, 0xa2, 0x6f, 0x82, 0x72, 0xcb});

  operands = {};
  operands.result = 17;
  operands.inputs[0] = 18;
  operands.inputs[1] = 8;
  operands.inputs[2] = 9;
  operands.immediate = 64;
  operands.secondary_immediate = 2;
  operands.has_index = true;
  ExpectVectorEncoding(kBinaryLoad, kBinaryEncoding, operands,
                       {0x62, 0x82, 0x6f, 0x00, 0x72, 0x4c, 0x88, 0x04});
  ExpectVectorEncoding(kBinaryBroadcast, kBinaryEncoding, operands,
                       {0x62, 0x82, 0x6f, 0x10, 0x72, 0x4c, 0x88, 0x10});

  operands = {};
  operands.result = 17;
  operands.inputs[0] = 2;
  operands.inputs[1] = 8;
  operands.inputs[2] = 9;
  operands.inputs[3] = 19;
  operands.immediate = 64;
  operands.secondary_immediate = 2;
  operands.has_index = true;
  ExpectVectorEncoding(kMaskedUnaryLoad, kUnaryEncoding, operands,
                       {0x62, 0x82, 0x7e, 0x8a, 0x72, 0x4c, 0x88, 0x04});
  ExpectVectorEncoding(kMaskedUnaryBroadcast, kUnaryEncoding, operands,
                       {0x62, 0x82, 0x7e, 0x9a, 0x72, 0x4c, 0x88, 0x10});

  operands.inputs[1] = 18;
  operands.inputs[2] = 8;
  operands.inputs[3] = 9;
  operands.inputs[4] = 19;
  ExpectVectorEncoding(kMaskedBinaryLoad, kBinaryEncoding, operands,
                       {0x62, 0x82, 0x6f, 0x82, 0x72, 0x4c, 0x88, 0x04});
  ExpectVectorEncoding(kMaskedBinaryBroadcast, kBinaryEncoding, operands,
                       {0x62, 0x82, 0x6f, 0x92, 0x72, 0x4c, 0x88, 0x10});
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
  operands.secondary_immediate = 2;
  operands.has_index = true;

  ExpectVectorEncoding(0xc1c0, 0x0ab1, operands,
                       {0xc4, 0x82, 0x7a, 0xb1, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1c0, 0x4ab1, operands,
                       {0xc4, 0x82, 0x7e, 0xb1, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1c0, 0x06b1, operands,
                       {0xc4, 0x82, 0x79, 0xb1, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1c0, 0x46b1, operands,
                       {0xc4, 0x82, 0x7d, 0xb1, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1c0, 0x0ab0, operands,
                       {0xc4, 0x82, 0x7a, 0xb0, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1c0, 0x4ab0, operands,
                       {0xc4, 0x82, 0x7e, 0xb0, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1c0, 0x06b0, operands,
                       {0xc4, 0x82, 0x79, 0xb0, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1c0, 0x46b0, operands,
                       {0xc4, 0x82, 0x7d, 0xb0, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1c0, 0x0eb0, operands,
                       {0xc4, 0x82, 0x7b, 0xb0, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1c0, 0x4eb0, operands,
                       {0xc4, 0x82, 0x7f, 0xb0, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1c0, 0x02b0, operands,
                       {0xc4, 0x82, 0x78, 0xb0, 0xa4, 0x88, 0x34, 0x12, 0, 0});
  ExpectVectorEncoding(0xc1c0, 0x42b0, operands,
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

TEST(EncodingTest, ScalarFloatMemoryWidthsAndAddressForms) {
  loom_x86_encoding_operands_t operands = {
      .immediate = -4,
      .result = 10,
      .inputs = {3, 1},
      .secondary_immediate = 3,
      .has_index = true,
  };
  // LLVM MC reference: vmovss/vmovsd -4(%rbx,%rcx,8),%xmm10.
  ExpectVectorEncoding(0xc1c0, 0x0910, operands,
                       {0xc5, 0x7a, 0x10, 0x54, 0xcb, 0xfc});
  ExpectVectorEncoding(0xc1c0, 0x0d10, operands,
                       {0xc5, 0x7b, 0x10, 0x54, 0xcb, 0xfc});
  // The same scalar widths with displacement-only addressing.
  operands.has_index = false;
  ExpectVectorEncoding(0xc1c0, 0x0910, operands,
                       {0xc5, 0x7a, 0x10, 0x53, 0xfc});
  ExpectVectorEncoding(0xc1c0, 0x0d10, operands,
                       {0xc5, 0x7b, 0x10, 0x53, 0xfc});
  operands.inputs[0] = 10;
  operands.inputs[1] = 3;
  operands.inputs[2] = 1;
  operands.has_index = true;
  ExpectVectorEncoding(0xd2c1, 0x0911, operands,
                       {0xc5, 0x7a, 0x11, 0x54, 0xcb, 0xfc});
  ExpectVectorEncoding(0xd2c1, 0x0d11, operands,
                       {0xc5, 0x7b, 0x11, 0x54, 0xcb, 0xfc});
  operands.has_index = false;
  ExpectVectorEncoding(0xd2c1, 0x0911, operands,
                       {0xc5, 0x7a, 0x11, 0x53, 0xfc});
  ExpectVectorEncoding(0xd2c1, 0x0d11, operands,
                       {0xc5, 0x7b, 0x11, 0x53, 0xfc});
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
      {0xc1c0,
       0x096f,
       {16, 1, {1, 2}, 1, true},
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
      {0xd2c1,
       0x097f,
       {16, 0, {1, 1, 2}, 1, true},
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

TEST(EncodingTest, DestructiveUnaryRegisterWidth) {
  loom_x86_encoding_operands_t operands = {.result = 0};
  operands.inputs[0] = 0;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_UNARY_RM, 0xf7 | (3u << 9), operands,
                 {0xf7, 0xd8}, 1u << 0);
  ExpectEncoding(LOOM_X86_ENCODING_FORM_UNARY_RM,
                 0xf7 | (3u << 9) | LOOM_X86_ENCODING_REX_W, operands,
                 {0x48, 0xf7, 0xd8}, 1u << 0);

  operands.result = 9;
  operands.inputs[0] = 9;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_UNARY_RM, 0xf7 | (3u << 9), operands,
                 {0x41, 0xf7, 0xd9}, 1u << 9);
  ExpectEncoding(LOOM_X86_ENCODING_FORM_UNARY_RM,
                 0xf7 | (3u << 9) | LOOM_X86_ENCODING_REX_W, operands,
                 {0x49, 0xf7, 0xd9}, 1u << 9);
}

TEST(EncodingTest, FixedDividendDivisionFamily) {
  loom_x86_encoding_operands_t operands = {.result = 2};
  ExpectEncoding(LOOM_X86_ENCODING_FORM_ZERO_REGISTER, 0x31, operands,
                 {0x31, 0xd2}, 1u << 2);
  ExpectEncoding(LOOM_X86_ENCODING_FORM_ZERO_REGISTER,
                 0x31 | LOOM_X86_ENCODING_REX_W, operands, {0x48, 0x31, 0xd2},
                 1u << 2);

  operands.inputs[0] = 0;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_SIGN_EXTEND_DIVIDEND, 0x99, operands,
                 {0x99}, 1u << 2);
  ExpectEncoding(LOOM_X86_ENCODING_FORM_SIGN_EXTEND_DIVIDEND,
                 0x99 | LOOM_X86_ENCODING_REX_W, operands, {0x48, 0x99},
                 1u << 2);

  operands.inputs[1] = 2;
  operands.inputs[2] = 9;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_DIVIDE, 0xf7 | (6u << 9), operands,
                 {0x41, 0xf7, 0xf1}, (1u << 0) | (1u << 2));
  ExpectEncoding(LOOM_X86_ENCODING_FORM_DIVIDE, 0xf7 | (7u << 9), operands,
                 {0x41, 0xf7, 0xf9}, (1u << 0) | (1u << 2));
  ExpectEncoding(LOOM_X86_ENCODING_FORM_DIVIDE,
                 0xf7 | (6u << 9) | LOOM_X86_ENCODING_REX_W, operands,
                 {0x49, 0xf7, 0xf1}, (1u << 0) | (1u << 2));
  ExpectEncoding(LOOM_X86_ENCODING_FORM_DIVIDE,
                 0xf7 | (7u << 9) | LOOM_X86_ENCODING_REX_W, operands,
                 {0x49, 0xf7, 0xf9}, (1u << 0) | (1u << 2));
}

TEST(EncodingTest, ByteRegisterPrefixAndFullWidthDefinition) {
  // MOVZX esi,sil requires a REX prefix even without extended registers.
  loom_x86_encoding_operands_t operands = {.result = 6};
  operands.inputs[0] = 6;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_UNARY_R_RM,
                 LOOM_X86_ENCODING_OPCODE_0F | 0xb6 | LOOM_X86_ENCODING_BYTE,
                 operands, {0x40, 0x0f, 0xb6, 0xf6}, 1u << 6);
  // MOVSX uses the same byte-register prefix and defines the full GPR32.
  ExpectEncoding(LOOM_X86_ENCODING_FORM_UNARY_R_RM,
                 LOOM_X86_ENCODING_OPCODE_0F | 0xbe | LOOM_X86_ENCODING_BYTE,
                 operands, {0x40, 0x0f, 0xbe, 0xf6}, 1u << 6);
  // The word source needs neither a byte-register REX prefix nor 16-bit result
  // selection; MOVSX still defines ESI.
  ExpectEncoding(LOOM_X86_ENCODING_FORM_UNARY_R_RM,
                 LOOM_X86_ENCODING_OPCODE_0F | 0xbf, operands,
                 {0x0f, 0xbf, 0xf6}, 1u << 6);
  // MOV esi,esi must still be emitted: it clears the high half of RSI.
  ExpectEncoding(LOOM_X86_ENCODING_FORM_UNARY_R_RM, 0x8b, operands,
                 {0x8b, 0xf6}, 1u << 6);

  // Narrow sources can directly define wide carriers. Unsigned forms retain
  // a 32-bit destination because x86 zeroes its complete 64-bit register.
  operands.result = 9;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_UNARY_R_RM,
                 LOOM_X86_ENCODING_OPCODE_0F | 0xb6 | LOOM_X86_ENCODING_BYTE,
                 operands, {0x44, 0x0f, 0xb6, 0xce}, 1u << 9);
  ExpectEncoding(LOOM_X86_ENCODING_FORM_UNARY_R_RM,
                 LOOM_X86_ENCODING_OPCODE_0F | 0xb7, operands,
                 {0x44, 0x0f, 0xb7, 0xce}, 1u << 9);
  ExpectEncoding(LOOM_X86_ENCODING_FORM_UNARY_R_RM,
                 LOOM_X86_ENCODING_OPCODE_0F | 0xbe | LOOM_X86_ENCODING_BYTE |
                     LOOM_X86_ENCODING_REX_W,
                 operands, {0x4c, 0x0f, 0xbe, 0xce}, 1u << 9);
  ExpectEncoding(LOOM_X86_ENCODING_FORM_UNARY_R_RM,
                 LOOM_X86_ENCODING_OPCODE_0F | 0xbf | LOOM_X86_ENCODING_REX_W,
                 operands, {0x4c, 0x0f, 0xbf, 0xce}, 1u << 9);
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

TEST(EncodingTest, DestructiveBinaryMemorySource) {
  loom_x86_encoding_operands_t operands = {
      .result = 9,
  };
  operands.inputs[0] = 9;
  operands.inputs[1] = 13;
  // ADD r9d,[r13] needs the explicit zero displacement required by r13.
  ExpectEncoding(LOOM_X86_ENCODING_FORM_BINARY_R_MEMORY, 0x03, operands,
                 {0x45, 0x03, 0x4d, 0x00}, 1u << 9);

  operands.inputs[1] = 12;
  operands.inputs[2] = 10;
  operands.immediate = -128;
  operands.secondary_immediate = 2;
  operands.has_index = true;
  // ADD r9,[r12+r10*4-128] exercises all three REX roles and a disp8.
  ExpectEncoding(LOOM_X86_ENCODING_FORM_BINARY_R_MEMORY,
                 0x03 | LOOM_X86_ENCODING_REX_W, operands,
                 {0x4f, 0x03, 0x4c, 0x94, 0x80}, 1u << 9);

  operands.has_index = false;
  operands.immediate = 128;
  operands.secondary_immediate = 0;
  // IMUL r9d,[r12+128] covers a two-byte opcode and disp32.
  ExpectEncoding(LOOM_X86_ENCODING_FORM_BINARY_R_MEMORY,
                 LOOM_X86_ENCODING_OPCODE_0F | 0xaf, operands,
                 {0x45, 0x0f, 0xaf, 0x8c, 0x24, 0x80, 0x00, 0x00, 0x00},
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
  operands.secondary_immediate = 2;
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
      1u << 7, 3);
  operands.result = 2;
  operands.inputs[0] = 0;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_MULTIPLY_HIGH,
                 0xf7 | (4u << 9) | LOOM_X86_ENCODING_REX_W, operands,
                 {0x49, 0xf7, 0xe1}, (1u << 0) | (1u << 2));
}

TEST(EncodingTest, CompareSelectPredicateFamily) {
  constexpr uint8_t kConditionCodes[] = {
      0x4,  // eq
      0x5,  // ne
      0xc,  // slt
      0xe,  // sle
      0xf,  // sgt
      0xd,  // sge
      0x2,  // ult
      0x6,  // ule
      0x7,  // ugt
      0x3,  // uge
  };
  loom_x86_encoding_operands_t operands = {
      .result = 10,
  };
  operands.inputs[0] = 8;
  operands.inputs[1] = 9;
  operands.inputs[2] = 11;
  operands.inputs[3] = 10;
  for (uint8_t condition_code : kConditionCodes) {
    operands.immediate = condition_code;
    loom_x86_encoded_instruction_t instruction;
    loom_x86_encode_instruction(LOOM_X86_ENCODING_FORM_COMPARE_SELECT, 0x39,
                                &operands, &instruction);
    ASSERT_EQ(instruction.length, 7);
    EXPECT_EQ(instruction.instruction_count, 2);
    EXPECT_EQ(instruction.bytes[0], 0x45);
    EXPECT_EQ(instruction.bytes[1], 0x39);
    EXPECT_EQ(instruction.bytes[2], 0xc8);
    EXPECT_EQ(instruction.bytes[3], 0x45);
    EXPECT_EQ(instruction.bytes[4], 0x0f);
    EXPECT_EQ(instruction.bytes[5], 0x40 | condition_code);
    EXPECT_EQ(instruction.bytes[6], 0xd3);
    EXPECT_EQ(loom_x86_encoding_gpr_writes(
                  LOOM_X86_ENCODING_FORM_COMPARE_SELECT, &operands),
              1u << 10);
  }
}

TEST(EncodingTest, CompareBranchPredicateFamily) {
  constexpr uint8_t kConditionCodes[] = {
      0x4,  // eq
      0x5,  // ne
      0xc,  // slt
      0xe,  // sle
      0xf,  // sgt
      0xd,  // sge
      0x2,  // ult
      0x6,  // ule
      0x7,  // ugt
      0x3,  // uge
  };
  loom_x86_encoding_operands_t operands = {};
  operands.inputs[0] = 8;
  operands.inputs[1] = 9;
  for (uint8_t condition_code : kConditionCodes) {
    operands.secondary_immediate = condition_code;
    loom_x86_encoded_instruction_t instruction;
    loom_x86_encode_instruction(LOOM_X86_ENCODING_FORM_COMPARE_BRANCH, 0x39,
                                &operands, &instruction);
    ASSERT_EQ(instruction.length, 9);
    EXPECT_EQ(instruction.instruction_count, 2);
    EXPECT_EQ(instruction.bytes[0], 0x45);
    EXPECT_EQ(instruction.bytes[1], 0x39);
    EXPECT_EQ(instruction.bytes[2], 0xc8);
    EXPECT_EQ(instruction.bytes[3], 0x0f);
    EXPECT_EQ(instruction.bytes[4], 0x80 | condition_code);
    EXPECT_EQ(loom_x86_encoding_gpr_writes(
                  LOOM_X86_ENCODING_FORM_COMPARE_BRANCH, &operands),
              0);
  }
}

TEST(EncodingTest, CompareBranchWidthsAndImmediate) {
  loom_x86_encoding_operands_t operands = {
      .secondary_immediate = 0x2,  // ult
  };
  operands.inputs[0] = 8;
  operands.inputs[1] = 9;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_COMPARE_BRANCH,
                 0x38 | LOOM_X86_ENCODING_BYTE, operands,
                 {0x45, 0x38, 0xc8, 0x0f, 0x82, 0, 0, 0, 0}, 0, 2);
  ExpectEncoding(LOOM_X86_ENCODING_FORM_COMPARE_BRANCH,
                 0x39 | LOOM_X86_ENCODING_OPERAND_16, operands,
                 {0x66, 0x45, 0x39, 0xc8, 0x0f, 0x82, 0, 0, 0, 0}, 0, 2);
  ExpectEncoding(LOOM_X86_ENCODING_FORM_COMPARE_BRANCH, 0x39, operands,
                 {0x45, 0x39, 0xc8, 0x0f, 0x82, 0, 0, 0, 0}, 0, 2);
  ExpectEncoding(LOOM_X86_ENCODING_FORM_COMPARE_BRANCH,
                 0x39 | LOOM_X86_ENCODING_REX_W, operands,
                 {0x4d, 0x39, 0xc8, 0x0f, 0x82, 0, 0, 0, 0}, 0, 2);

  operands.secondary_immediate = 0x5;  // ne
  operands.immediate = -17;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_COMPARE_IMMEDIATE_BRANCH,
                 0x80 | (7u << 9) | LOOM_X86_ENCODING_BYTE, operands,
                 {0x41, 0x80, 0xf8, 0xef, 0x0f, 0x85, 0, 0, 0, 0}, 0, 2);
  operands.immediate = -129;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_COMPARE_IMMEDIATE_BRANCH,
                 0x81 | (7u << 9) | LOOM_X86_ENCODING_OPERAND_16, operands,
                 {0x66, 0x41, 0x81, 0xf8, 0x7f, 0xff, 0x0f, 0x85, 0, 0, 0, 0},
                 0, 2);
  operands.immediate = 128;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_COMPARE_IMMEDIATE_BRANCH,
                 0x81 | (7u << 9), operands,
                 {0x41, 0x81, 0xf8, 0x80, 0, 0, 0, 0x0f, 0x85, 0, 0, 0, 0}, 0,
                 2);
  operands.immediate = -129;
  ExpectEncoding(
      LOOM_X86_ENCODING_FORM_COMPARE_IMMEDIATE_BRANCH,
      0x81 | (7u << 9) | LOOM_X86_ENCODING_REX_W, operands,
      {0x49, 0x81, 0xf8, 0x7f, 0xff, 0xff, 0xff, 0x0f, 0x85, 0, 0, 0, 0}, 0, 2);
}

TEST(EncodingTest, CompareSelectWidthsAndImmediate) {
  loom_x86_encoding_operands_t operands = {
      .immediate = 0xc,  // slt
      .result = 10,
  };
  operands.inputs[0] = 8;
  operands.inputs[1] = 9;
  operands.inputs[2] = 11;
  operands.inputs[3] = 10;
  // CMP r8,r9; CMOVL r10,r11 uses independent compare and result widths.
  ExpectEncoding(
      LOOM_X86_ENCODING_FORM_COMPARE_SELECT,
      0x39 | LOOM_X86_ENCODING_REX_W | LOOM_X86_ENCODING_RESULT_REX_W, operands,
      {0x4d, 0x39, 0xc8, 0x4d, 0x0f, 0x4c, 0xd3}, 1u << 10, 2);
  ExpectEncoding(LOOM_X86_ENCODING_FORM_COMPARE_SELECT,
                 0x39 | LOOM_X86_ENCODING_RESULT_REX_W, operands,
                 {0x45, 0x39, 0xc8, 0x4d, 0x0f, 0x4c, 0xd3}, 1u << 10, 2);

  operands.immediate = 17;
  operands.inputs[1] = 11;
  operands.inputs[2] = 10;
  operands.secondary_immediate = 0x6;  // ule
  ExpectEncoding(LOOM_X86_ENCODING_FORM_COMPARE_IMMEDIATE_SELECT,
                 0x81 | (7u << 9) | LOOM_X86_ENCODING_REX_W, operands,
                 {0x49, 0x83, 0xf8, 0x11, 0x45, 0x0f, 0x46, 0xd3}, 1u << 10, 2);
  ExpectEncoding(LOOM_X86_ENCODING_FORM_COMPARE_IMMEDIATE_SELECT,
                 0x81 | (7u << 9) | LOOM_X86_ENCODING_RESULT_REX_W, operands,
                 {0x41, 0x83, 0xf8, 0x11, 0x4d, 0x0f, 0x46, 0xd3}, 1u << 10, 2);
}

TEST(EncodingTest, NarrowComparePredicates) {
  loom_x86_encoding_operands_t operands = {
      .immediate = 0x2,  // ult
      .result = 7,
  };
  operands.inputs[0] = 4;
  operands.inputs[1] = 5;
  // CMP spl,bpl; SETB dil; MOVZX edi,dil. Both byte comparison operands need
  // the REX namespace even though neither register is extended.
  ExpectEncoding(
      LOOM_X86_ENCODING_FORM_COMPARE_PREDICATE, 0x38 | LOOM_X86_ENCODING_BYTE,
      operands,
      {0x40, 0x38, 0xec, 0x40, 0x0f, 0x92, 0xc7, 0x40, 0x0f, 0xb6, 0xff},
      1u << 7, 3);

  operands.inputs[0] = 8;
  operands.inputs[1] = 9;
  operands.immediate = 0xc;  // slt
  ExpectEncoding(
      LOOM_X86_ENCODING_FORM_COMPARE_PREDICATE,
      0x39 | LOOM_X86_ENCODING_OPERAND_16, operands,
      {0x66, 0x45, 0x39, 0xc8, 0x40, 0x0f, 0x9c, 0xc7, 0x40, 0x0f, 0xb6, 0xff},
      1u << 7, 3);

  operands.inputs[0] = 4;
  operands.immediate = -128;
  operands.secondary_immediate = 0xc;  // slt
  ExpectEncoding(
      LOOM_X86_ENCODING_FORM_COMPARE_IMMEDIATE_PREDICATE,
      0x80 | (7u << 9) | LOOM_X86_ENCODING_BYTE, operands,
      {0x40, 0x80, 0xfc, 0x80, 0x40, 0x0f, 0x9c, 0xc7, 0x40, 0x0f, 0xb6, 0xff},
      1u << 7, 3);

  operands.inputs[0] = 8;
  operands.immediate = -129;
  operands.secondary_immediate = 0x6;  // ule
  ExpectEncoding(LOOM_X86_ENCODING_FORM_COMPARE_IMMEDIATE_PREDICATE,
                 0x81 | (7u << 9) | LOOM_X86_ENCODING_OPERAND_16, operands,
                 {0x66, 0x41, 0x81, 0xf8, 0x7f, 0xff, 0x40, 0x0f, 0x96, 0xc7,
                  0x40, 0x0f, 0xb6, 0xff},
                 1u << 7, 3);
}

TEST(EncodingTest, NarrowCompareSelectWidths) {
  loom_x86_encoding_operands_t operands = {
      .immediate = 0x2,  // ult
      .result = 10,
  };
  operands.inputs[0] = 4;
  operands.inputs[1] = 5;
  operands.inputs[2] = 11;
  operands.inputs[3] = 10;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_COMPARE_SELECT,
                 0x38 | LOOM_X86_ENCODING_BYTE | LOOM_X86_ENCODING_RESULT_REX_W,
                 operands, {0x40, 0x38, 0xec, 0x4d, 0x0f, 0x42, 0xd3}, 1u << 10,
                 2);

  operands.inputs[0] = 8;
  operands.inputs[1] = 11;
  operands.inputs[2] = 10;
  operands.immediate = -129;
  operands.secondary_immediate = 0xc;  // slt
  ExpectEncoding(LOOM_X86_ENCODING_FORM_COMPARE_IMMEDIATE_SELECT,
                 0x81 | (7u << 9) | LOOM_X86_ENCODING_OPERAND_16 |
                     LOOM_X86_ENCODING_RESULT_REX_W,
                 operands,
                 {0x66, 0x41, 0x81, 0xf8, 0x7f, 0xff, 0x4d, 0x0f, 0x4c, 0xd3},
                 1u << 10, 2);
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
                 operands, {0x41, 0x83, 0xe9, 0x01}, 1u << 9);
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

TEST(EncodingTest, SignedImmediateBoundariesUseShortestExactEncoding) {
  loom_x86_encoding_operands_t operands = {
      .result = 9,
  };
  operands.inputs[0] = 10;

  // Group-one arithmetic uses opcode 83 only when the value survives signed
  // imm8 extension. Both carrier widths otherwise retain opcode 81 and imm32.
  operands.immediate = -129;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE, 0x81, operands,
                 {0x41, 0x81, 0xc1, 0x7f, 0xff, 0xff, 0xff}, 1u << 9);
  operands.immediate = -128;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE, 0x81, operands,
                 {0x41, 0x83, 0xc1, 0x80}, 1u << 9);
  operands.immediate = 127;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE, 0x81, operands,
                 {0x41, 0x83, 0xc1, 0x7f}, 1u << 9);
  operands.immediate = 128;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE, 0x81, operands,
                 {0x41, 0x81, 0xc1, 0x80, 0x00, 0x00, 0x00}, 1u << 9);
  operands.immediate = -129;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
                 0x81 | LOOM_X86_ENCODING_REX_W, operands,
                 {0x49, 0x81, 0xc1, 0x7f, 0xff, 0xff, 0xff}, 1u << 9);
  operands.immediate = -128;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
                 0x81 | LOOM_X86_ENCODING_REX_W, operands,
                 {0x49, 0x83, 0xc1, 0x80}, 1u << 9);
  operands.immediate = 127;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
                 0x81 | LOOM_X86_ENCODING_REX_W, operands,
                 {0x49, 0x83, 0xc1, 0x7f}, 1u << 9);
  operands.immediate = 128;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
                 0x81 | LOOM_X86_ENCODING_REX_W, operands,
                 {0x49, 0x81, 0xc1, 0x80, 0x00, 0x00, 0x00}, 1u << 9);

  // Two-operand IMUL has the same boundary with opcodes 6B and 69.
  operands.immediate = -129;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_MULTIPLY_IMMEDIATE, 0x69, operands,
                 {0x45, 0x69, 0xca, 0x7f, 0xff, 0xff, 0xff}, 1u << 9);
  operands.immediate = -128;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_MULTIPLY_IMMEDIATE, 0x69, operands,
                 {0x45, 0x6b, 0xca, 0x80}, 1u << 9);
  operands.immediate = 127;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_MULTIPLY_IMMEDIATE, 0x69, operands,
                 {0x45, 0x6b, 0xca, 0x7f}, 1u << 9);
  operands.immediate = 128;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_MULTIPLY_IMMEDIATE, 0x69, operands,
                 {0x45, 0x69, 0xca, 0x80, 0x00, 0x00, 0x00}, 1u << 9);
  operands.immediate = -129;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_MULTIPLY_IMMEDIATE,
                 0x69 | LOOM_X86_ENCODING_REX_W, operands,
                 {0x4d, 0x69, 0xca, 0x7f, 0xff, 0xff, 0xff}, 1u << 9);
  operands.immediate = -128;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_MULTIPLY_IMMEDIATE,
                 0x69 | LOOM_X86_ENCODING_REX_W, operands,
                 {0x4d, 0x6b, 0xca, 0x80}, 1u << 9);
  operands.immediate = 127;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_MULTIPLY_IMMEDIATE,
                 0x69 | LOOM_X86_ENCODING_REX_W, operands,
                 {0x4d, 0x6b, 0xca, 0x7f}, 1u << 9);
  operands.immediate = 128;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_MULTIPLY_IMMEDIATE,
                 0x69 | LOOM_X86_ENCODING_REX_W, operands,
                 {0x4d, 0x69, 0xca, 0x80, 0x00, 0x00, 0x00}, 1u << 9);

  // Compare uses the same group-one encoding before SETB and MOVZX produce a
  // canonical predicate. Result register seven exercises the byte-register
  // REX prefix independently of the compared carrier width.
  operands.result = 7;
  operands.immediate = -129;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_COMPARE_IMMEDIATE, 0x92, operands,
                 {0x41, 0x81, 0xfa, 0x7f, 0xff, 0xff, 0xff, 0x40, 0x0f, 0x92,
                  0xc7, 0x40, 0x0f, 0xb6, 0xff},
                 1u << 7, 3);
  operands.immediate = -128;
  ExpectEncoding(
      LOOM_X86_ENCODING_FORM_COMPARE_IMMEDIATE, 0x92, operands,
      {0x41, 0x83, 0xfa, 0x80, 0x40, 0x0f, 0x92, 0xc7, 0x40, 0x0f, 0xb6, 0xff},
      1u << 7, 3);
  operands.immediate = 127;
  ExpectEncoding(
      LOOM_X86_ENCODING_FORM_COMPARE_IMMEDIATE, 0x92, operands,
      {0x41, 0x83, 0xfa, 0x7f, 0x40, 0x0f, 0x92, 0xc7, 0x40, 0x0f, 0xb6, 0xff},
      1u << 7, 3);
  operands.immediate = 128;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_COMPARE_IMMEDIATE, 0x92, operands,
                 {0x41, 0x81, 0xfa, 0x80, 0x00, 0x00, 0x00, 0x40, 0x0f, 0x92,
                  0xc7, 0x40, 0x0f, 0xb6, 0xff},
                 1u << 7, 3);
  operands.immediate = -129;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_COMPARE_IMMEDIATE,
                 0x92 | LOOM_X86_ENCODING_REX_W, operands,
                 {0x49, 0x81, 0xfa, 0x7f, 0xff, 0xff, 0xff, 0x40, 0x0f, 0x92,
                  0xc7, 0x40, 0x0f, 0xb6, 0xff},
                 1u << 7, 3);
  operands.immediate = -128;
  ExpectEncoding(
      LOOM_X86_ENCODING_FORM_COMPARE_IMMEDIATE, 0x92 | LOOM_X86_ENCODING_REX_W,
      operands,
      {0x49, 0x83, 0xfa, 0x80, 0x40, 0x0f, 0x92, 0xc7, 0x40, 0x0f, 0xb6, 0xff},
      1u << 7, 3);
  operands.immediate = 127;
  ExpectEncoding(
      LOOM_X86_ENCODING_FORM_COMPARE_IMMEDIATE, 0x92 | LOOM_X86_ENCODING_REX_W,
      operands,
      {0x49, 0x83, 0xfa, 0x7f, 0x40, 0x0f, 0x92, 0xc7, 0x40, 0x0f, 0xb6, 0xff},
      1u << 7, 3);
  operands.immediate = 128;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_COMPARE_IMMEDIATE,
                 0x92 | LOOM_X86_ENCODING_REX_W, operands,
                 {0x49, 0x81, 0xfa, 0x80, 0x00, 0x00, 0x00, 0x40, 0x0f, 0x92,
                  0xc7, 0x40, 0x0f, 0xb6, 0xff},
                 1u << 7, 3);
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
                 operands, {0x85, 0xc9, 0x4d, 0x0f, 0x45, 0xc1}, 1u << 8, 2);
  // SUB changes flags before CMOVB restores the original value on borrow.
  operands.inputs[0] = 7;
  operands.immediate = 3;
  ExpectEncoding(
      LOOM_X86_ENCODING_FORM_SUBTRACT_IF_UGE,
      LOOM_X86_ENCODING_OPCODE_0F | 0x42, operands,
      {0x44, 0x8b, 0xc7, 0x41, 0x83, 0xe8, 3, 0x44, 0x0f, 0x42, 0xc7}, 1u << 8,
      3);
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
  ExpectEncoding(LOOM_X86_ENCODING_FORM_UNARY_R_RM,
                 0x63 | LOOM_X86_ENCODING_REX_W, operands, {0x4d, 0x63, 0xca},
                 1u << 9);
}

TEST(EncodingTest, FullWidthBranchAndRelativeDisplacements) {
  loom_x86_encoding_operands_t operands = {};
  operands.inputs[0] = 8;
  operands.immediate = 0x12345678;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_BRANCH_ZERO, LOOM_X86_ENCODING_REX_W,
                 operands,
                 {0x4d, 0x85, 0xc0, 0x0f, 0x84, 0x78, 0x56, 0x34, 0x12}, 0, 2);
  operands.inputs[0] = 1;
  operands.immediate = -6;
  ExpectEncoding(LOOM_X86_ENCODING_FORM_BRANCH_NONZERO, 0, operands,
                 {0x85, 0xc9, 0x0f, 0x85, 0xfa, 0xff, 0xff, 0xff}, 0, 2);
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
