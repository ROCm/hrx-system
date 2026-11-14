// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <string>

#include "iree/testing/gtest.h"
#include "loom/target/arch/x86/descriptors/avx512_descriptors.h"
#include "loom/target/emit/native/x86/encoding.h"

namespace {

struct PermutationEncoding {
  // Architectural mnemonic shared by all register widths.
  const char* mnemonic;
  // EVEX selects 32 registers and includes the ZMM form.
  bool evex;
  // Two-source permutes overwrite their first semantic input.
  bool destructive;
  // Independent LLVM assembly bytes for XMM registers 0, 1, and 2.
  std::array<uint8_t, 6> low;
  // Independent LLVM assembly bytes for the highest three XMM registers.
  std::array<uint8_t, 6> high;
};

// Intel SDM register forms assembled independently with LLVM MC. Every width
// was checked; L/LL alone distinguishes the native vector widths in each row.
constexpr PermutationEncoding kPermutations[] = {
    {"vpunpcklbw",
     false,
     false,
     {0xc5, 0xf1, 0x60, 0xc2},
     {0xc4, 0x41, 0x09, 0x60, 0xfd}},
    {"vpunpckhbw",
     false,
     false,
     {0xc5, 0xf1, 0x68, 0xc2},
     {0xc4, 0x41, 0x09, 0x68, 0xfd}},
    {"vpunpcklwd",
     false,
     false,
     {0xc5, 0xf1, 0x61, 0xc2},
     {0xc4, 0x41, 0x09, 0x61, 0xfd}},
    {"vpunpckhwd",
     false,
     false,
     {0xc5, 0xf1, 0x69, 0xc2},
     {0xc4, 0x41, 0x09, 0x69, 0xfd}},
    {"vpunpckldq",
     false,
     false,
     {0xc5, 0xf1, 0x62, 0xc2},
     {0xc4, 0x41, 0x09, 0x62, 0xfd}},
    {"vpunpckhdq",
     false,
     false,
     {0xc5, 0xf1, 0x6a, 0xc2},
     {0xc4, 0x41, 0x09, 0x6a, 0xfd}},
    {"vpunpcklqdq",
     false,
     false,
     {0xc5, 0xf1, 0x6c, 0xc2},
     {0xc4, 0x41, 0x09, 0x6c, 0xfd}},
    {"vpunpckhqdq",
     false,
     false,
     {0xc5, 0xf1, 0x6d, 0xc2},
     {0xc4, 0x41, 0x09, 0x6d, 0xfd}},
    {"vpunpcklbw",
     true,
     false,
     {0x62, 0xf1, 0x75, 0x08, 0x60, 0xc2},
     {0x62, 0x01, 0x0d, 0x00, 0x60, 0xfd}},
    {"vpunpckhbw",
     true,
     false,
     {0x62, 0xf1, 0x75, 0x08, 0x68, 0xc2},
     {0x62, 0x01, 0x0d, 0x00, 0x68, 0xfd}},
    {"vpunpcklwd",
     true,
     false,
     {0x62, 0xf1, 0x75, 0x08, 0x61, 0xc2},
     {0x62, 0x01, 0x0d, 0x00, 0x61, 0xfd}},
    {"vpunpckhwd",
     true,
     false,
     {0x62, 0xf1, 0x75, 0x08, 0x69, 0xc2},
     {0x62, 0x01, 0x0d, 0x00, 0x69, 0xfd}},
    {"vpunpckldq",
     true,
     false,
     {0x62, 0xf1, 0x75, 0x08, 0x62, 0xc2},
     {0x62, 0x01, 0x0d, 0x00, 0x62, 0xfd}},
    {"vpunpckhdq",
     true,
     false,
     {0x62, 0xf1, 0x75, 0x08, 0x6a, 0xc2},
     {0x62, 0x01, 0x0d, 0x00, 0x6a, 0xfd}},
    {"vpunpcklqdq",
     true,
     false,
     {0x62, 0xf1, 0xf5, 0x08, 0x6c, 0xc2},
     {0x62, 0x01, 0x8d, 0x00, 0x6c, 0xfd}},
    {"vpunpckhqdq",
     true,
     false,
     {0x62, 0xf1, 0xf5, 0x08, 0x6d, 0xc2},
     {0x62, 0x01, 0x8d, 0x00, 0x6d, 0xfd}},
    {"vpermi2b",
     true,
     true,
     {0x62, 0xf2, 0x75, 0x08, 0x75, 0xc2},
     {0x62, 0x02, 0x0d, 0x00, 0x75, 0xfd}},
    {"vpermt2b",
     true,
     true,
     {0x62, 0xf2, 0x75, 0x08, 0x7d, 0xc2},
     {0x62, 0x02, 0x0d, 0x00, 0x7d, 0xfd}},
    {"vpermi2w",
     true,
     true,
     {0x62, 0xf2, 0xf5, 0x08, 0x75, 0xc2},
     {0x62, 0x02, 0x8d, 0x00, 0x75, 0xfd}},
    {"vpermt2w",
     true,
     true,
     {0x62, 0xf2, 0xf5, 0x08, 0x7d, 0xc2},
     {0x62, 0x02, 0x8d, 0x00, 0x7d, 0xfd}},
    {"vpermi2d",
     true,
     true,
     {0x62, 0xf2, 0x75, 0x08, 0x76, 0xc2},
     {0x62, 0x02, 0x0d, 0x00, 0x76, 0xfd}},
    {"vpermt2d",
     true,
     true,
     {0x62, 0xf2, 0x75, 0x08, 0x7e, 0xc2},
     {0x62, 0x02, 0x0d, 0x00, 0x7e, 0xfd}},
    {"vpermi2q",
     true,
     true,
     {0x62, 0xf2, 0xf5, 0x08, 0x76, 0xc2},
     {0x62, 0x02, 0x8d, 0x00, 0x76, 0xfd}},
    {"vpermt2q",
     true,
     true,
     {0x62, 0xf2, 0xf5, 0x08, 0x7e, 0xc2},
     {0x62, 0x02, 0x8d, 0x00, 0x7e, 0xfd}},
};

TEST(PermutationEncodingTest, AllWidthsAndRegisterDomains) {
  const auto* descriptor_set = loom_x86_avx512_core_descriptor_set();
  const char* suffixes[] = {"xmm", "ymm", "zmm"};
  for (const auto& row : kPermutations) {
    for (uint32_t width = 0; width < (row.evex ? 3u : 2u); ++width) {
      const std::string key =
          std::string(row.evex ? "x86.avx512." : "x86.avx2.") + row.mnemonic +
          "." + suffixes[width];
      SCOPED_TRACE(key);
      const uint32_t ordinal = loom_low_descriptor_set_lookup_descriptor(
          descriptor_set, iree_make_string_view(key.data(), key.size()));
      ASSERT_NE(ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
      const auto& descriptor = descriptor_set->descriptors[ordinal];
      for (bool high : {false, true}) {
        SCOPED_TRACE(high);
        const uint8_t result = high ? (row.evex ? 31 : 15) : 0;
        const uint8_t lhs = high ? result - 1 : 1;
        const uint8_t rhs = high ? result - 2 : 2;
        loom_x86_encoding_operands_t operands = {.result = result};
        if (row.destructive) {
          operands.inputs[0] = result;
          operands.inputs[1] = lhs;
          operands.inputs[2] = rhs;
        } else {
          operands.inputs[0] = lhs;
          operands.inputs[1] = rhs;
        }
        loom_x86_encoded_instruction_t instruction;
        loom_x86_encode_instruction(
            loom_low_descriptor_layout(descriptor_set, &descriptor)
                ->encoding_format_id,
            descriptor.encoding_id, &operands, &instruction);
        auto expected = high ? row.high : row.low;
        const bool short_vex = expected[0] == 0xc5;
        expected[row.evex ? 3 : (short_vex ? 1 : 2)] |= width
                                                        << (row.evex ? 5 : 2);
        const size_t length = row.evex ? 6 : (short_vex ? 4 : 5);
        ASSERT_EQ(instruction.length, length);
        EXPECT_EQ(instruction.instruction_count, 1);
        for (size_t byte = 0; byte < length; ++byte) {
          EXPECT_EQ(instruction.bytes[byte], expected[byte]) << "byte " << byte;
        }
      }
    }
  }
}

struct ImmediateShiftEncoding {
  // Architectural packed immediate shift.
  const char* mnemonic;
  // Independent LLVM bytes for XMM destination 0, source 1, and count 3.
  std::array<uint8_t, 7> low;
  // Independent LLVM bytes for XMM destination 31 and source 30.
  std::array<uint8_t, 7> high;
};

// Pair extraction uses right shifts before narrowing. The complete shift
// descriptor family preserves the same EVEX register domain at every width.
constexpr ImmediateShiftEncoding kImmediateShifts[] = {
    {"vpsllw",
     {0x62, 0xf1, 0x7d, 0x08, 0x71, 0xf1, 0x03},
     {0x62, 0x91, 0x05, 0x00, 0x71, 0xf6, 0x03}},
    {"vpslld",
     {0x62, 0xf1, 0x7d, 0x08, 0x72, 0xf1, 0x03},
     {0x62, 0x91, 0x05, 0x00, 0x72, 0xf6, 0x03}},
    {"vpsllq",
     {0x62, 0xf1, 0xfd, 0x08, 0x73, 0xf1, 0x03},
     {0x62, 0x91, 0x85, 0x00, 0x73, 0xf6, 0x03}},
    {"vpsraw",
     {0x62, 0xf1, 0x7d, 0x08, 0x71, 0xe1, 0x03},
     {0x62, 0x91, 0x05, 0x00, 0x71, 0xe6, 0x03}},
    {"vpsrad",
     {0x62, 0xf1, 0x7d, 0x08, 0x72, 0xe1, 0x03},
     {0x62, 0x91, 0x05, 0x00, 0x72, 0xe6, 0x03}},
    {"vpsraq",
     {0x62, 0xf1, 0xfd, 0x08, 0x72, 0xe1, 0x03},
     {0x62, 0x91, 0x85, 0x00, 0x72, 0xe6, 0x03}},
    {"vpsrlw",
     {0x62, 0xf1, 0x7d, 0x08, 0x71, 0xd1, 0x03},
     {0x62, 0x91, 0x05, 0x00, 0x71, 0xd6, 0x03}},
    {"vpsrld",
     {0x62, 0xf1, 0x7d, 0x08, 0x72, 0xd1, 0x03},
     {0x62, 0x91, 0x05, 0x00, 0x72, 0xd6, 0x03}},
    {"vpsrlq",
     {0x62, 0xf1, 0xfd, 0x08, 0x73, 0xd1, 0x03},
     {0x62, 0x91, 0x85, 0x00, 0x73, 0xd6, 0x03}},
};

TEST(PermutationEncodingTest, ImmediateShiftsPreserveEveryRegisterWidth) {
  const auto* descriptor_set = loom_x86_avx512_core_descriptor_set();
  const char* suffixes[] = {"xmm", "ymm", "zmm"};
  for (const auto& row : kImmediateShifts) {
    for (uint32_t width = 0; width < 3; ++width) {
      const std::string key =
          std::string("x86.avx512.") + row.mnemonic + "." + suffixes[width];
      SCOPED_TRACE(key);
      const uint32_t ordinal = loom_low_descriptor_set_lookup_descriptor(
          descriptor_set, iree_make_string_view(key.data(), key.size()));
      ASSERT_NE(ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
      const auto& descriptor = descriptor_set->descriptors[ordinal];
      for (bool high : {false, true}) {
        SCOPED_TRACE(high);
        const loom_x86_encoding_operands_t operands = {
            .immediate = 3,
            .result = static_cast<uint8_t>(high ? 31 : 0),
            .inputs = {static_cast<uint8_t>(high ? 30 : 1)},
        };
        loom_x86_encoded_instruction_t instruction;
        loom_x86_encode_instruction(
            loom_low_descriptor_layout(descriptor_set, &descriptor)
                ->encoding_format_id,
            descriptor.encoding_id, &operands, &instruction);
        auto expected = high ? row.high : row.low;
        expected[3] |= width << 5;
        ASSERT_EQ(instruction.length, expected.size());
        EXPECT_EQ(instruction.instruction_count, 1);
        for (size_t byte = 0; byte < expected.size(); ++byte) {
          EXPECT_EQ(instruction.bytes[byte], expected[byte]) << "byte " << byte;
        }
      }
    }
  }
}

}  // namespace
