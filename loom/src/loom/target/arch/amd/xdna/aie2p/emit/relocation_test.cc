// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/relocation.h"

#include <array>
#include <cstring>
#include <tuple>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/aie2p/encoding/encoding.h"

namespace loom {
namespace {

class Aie2pCodeRelocationTest
    : public ::testing::TestWithParam<std::tuple<const char*, bool>> {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Owns the assembler's native section storage through each test.
  iree_arena_block_pool_t block_pool_;
  // Independent assembled outputs; input contributions remain immutable.
  iree_arena_allocator_t arena_;
};

TEST_P(Aie2pCodeRelocationTest, KeepsTargetFixedWhenCallerSectionMoves) {
  const auto [instruction_name, bundled] = GetParam();
  const bool conditional = std::strcmp(instruction_name, "JZ") == 0 ||
                           std::strcmp(instruction_name, "JNZ") == 0;
  const bool address = std::strcmp(instruction_name, "MOVXM") == 0;
  const bool call = std::strcmp(instruction_name, "JL_lng") == 0;
  const loom_aie2p_instruction_id_t instruction =
      loom_aie2p_encoding_find_instruction(
          iree_make_cstring_view(instruction_name));
  ASSERT_NE(instruction, LOOM_AIE2P_INSTRUCTION_ID_INVALID);
  loom_aie2p_encoding_field_value_t fields[] = {
      {loom_aie2p_encoding_find_field(IREE_SV("i")), 0},
      {loom_aie2p_encoding_find_field(address ? IREE_SV("dst") : IREE_SV("s0")),
       3},
  };
  const iree_host_size_t field_count = conditional || address ? 2 : 1;
  uint64_t instruction_value = 0;
  IREE_ASSERT_OK(loom_aie2p_encoding_pack_instruction(
      instruction, fields, field_count, &instruction_value));
  const loom_aie2p_instruction_id_t load =
      loom_aie2p_encoding_find_instruction(IREE_SV("VLDA_dmx_lda_x_idx_imm"));
  const loom_aie2p_encoding_field_value_t load_fields[] = {
      {loom_aie2p_encoding_find_field(IREE_SV("dst")), 1},
      {loom_aie2p_encoding_find_field(IREE_SV("imm")), 0},
      {loom_aie2p_encoding_find_field(IREE_SV("ptr")), 2},
  };
  uint64_t load_value = 0;
  IREE_ASSERT_OK(loom_aie2p_encoding_pack_instruction(
      load, load_fields, IREE_ARRAYSIZE(load_fields), &load_value));
  loom_aie2p_encoded_slot_t slots[] = {
      {LOOM_AIE2P_SLOT_LNG, instruction_value},
      {LOOM_AIE2P_SLOT_LDA, load_value},
  };
  const loom_aie2p_bundle_format_id_t format =
      loom_aie2p_encoding_find_bundle_format(bundled ? IREE_SV("I80_LDA_LNG")
                                                     : IREE_SV("I48_LNG"));
  loom_aie2p_encoding_packet_t packet;
  IREE_ASSERT_OK(
      loom_aie2p_encoding_pack_bundle(format, slots, bundled ? 2 : 1, &packet));
  std::array<uint8_t, 32> code = {};
  std::memcpy(code.data() + 16, packet.data, packet.data_length);
  const auto original_code = code;
  const loom_native_section_contribution_t sections[] = {
      {
          .section_name = IREE_SV(".text.caller"),
          .storage = LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
          .access = LOOM_NATIVE_SECTION_ACCESS_READ |
                    LOOM_NATIVE_SECTION_ACCESS_EXECUTE,
          .contribution_alignment = 16,
          .contents = iree_make_const_byte_span(code.data(), code.size()),
      },
      {
          .section_name = IREE_SV(".text.resident"),
          .storage = LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
          .access = LOOM_NATIVE_SECTION_ACCESS_READ |
                    LOOM_NATIVE_SECTION_ACCESS_EXECUTE,
          .contribution_alignment = 16,
          .contents = iree_make_const_byte_span(code.data(), code.size()),
      },
  };
  const loom_native_object_symbol_t symbol = {
      .name = IREE_SV("resident"),
      .section_contribution_index = 1,
      .section_offset = 0,
      .size = code.size(),
      .binding = LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL,
      .visibility = LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_HIDDEN,
      .kind = LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
  };
  const loom_native_object_fixup_t fixup = {
      .section_contribution_index = 0,
      .section_offset = 16,
      .relocation_kind = static_cast<uint32_t>(
          address ? LOOM_AIE2P_NATIVE_RELOCATION_KIND_CORE_CODE_ADDRESS_ABSOLUTE
          : call  ? LOOM_AIE2P_NATIVE_RELOCATION_KIND_CORE_CALL_ABSOLUTE
                  : LOOM_AIE2P_NATIVE_RELOCATION_KIND_CORE_BRANCH_ABSOLUTE),
      .target_symbol_index = 0,
      .addend = 16,
  };
  const loom_native_object_contribution_t object = {
      .sections = sections,
      .section_count = IREE_ARRAYSIZE(sections),
      .symbols = &symbol,
      .symbol_count = 1,
      .fixups = &fixup,
      .fixup_count = 1,
  };

  // Both directions use the same absolute target. Moving a caller does not
  // require rewriting the resident section or changing another packet slot.
  for (uint64_t caller_address : {0x1000u, 0x3000u}) {
    loom_native_section_contribution_assembly_t assembly = {};
    IREE_ASSERT_OK(loom_native_assemble_section_contributions(
        object.sections, object.section_count, &arena_, &assembly));
    ASSERT_EQ(assembly.section_count, 2u);
    assembly.sections[0].address = caller_address;
    assembly.sections[1].address = 0x2000;
    IREE_ASSERT_OK(
        loom_aie2p_native_object_apply_fixups(&object, &assembly, &arena_));

    fields[0].value = 0x2010;
    IREE_ASSERT_OK(loom_aie2p_encoding_pack_instruction(
        instruction, fields, field_count, &slots[0].value));
    loom_aie2p_encoding_packet_t expected_packet;
    IREE_ASSERT_OK(loom_aie2p_encoding_pack_bundle(
        format, slots, bundled ? 2 : 1, &expected_packet));
    auto expected_code = original_code;
    std::memcpy(expected_code.data() + 16, expected_packet.data,
                expected_packet.data_length);
    EXPECT_EQ(0, std::memcmp(assembly.sections[0].contents.data,
                             expected_code.data(), expected_code.size()));
    EXPECT_EQ(0, std::memcmp(assembly.sections[1].contents.data,
                             original_code.data(), original_code.size()));
    EXPECT_EQ(code, original_code);
  }
}

INSTANTIATE_TEST_SUITE_P(AbsoluteCodeAddresses, Aie2pCodeRelocationTest,
                         ::testing::Combine(::testing::Values("J_lng", "JZ",
                                                              "JNZ", "JL_lng",
                                                              "MOVXM"),
                                            ::testing::Bool()));

}  // namespace
}  // namespace loom
