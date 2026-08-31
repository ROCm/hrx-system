// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/tile_image.h"

#include <array>
#include <cstring>
#include <memory>
#include <string>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/relocation.h"
#include "loom/target/arch/amd/xdna/aie2p/encoding/encoding.h"

namespace loom {
namespace {

using StreamPtr =
    std::unique_ptr<iree_io_stream_t, void (*)(iree_io_stream_t*)>;

uint16_t LoadLeU16(const std::string& bytes, size_t offset) {
  return (uint16_t)(uint8_t)bytes[offset] |
         ((uint16_t)(uint8_t)bytes[offset + 1] << 8);
}

uint32_t LoadLeU32(const std::string& bytes, size_t offset) {
  return (uint32_t)(uint8_t)bytes[offset] |
         ((uint32_t)(uint8_t)bytes[offset + 1] << 8) |
         ((uint32_t)(uint8_t)bytes[offset + 2] << 16) |
         ((uint32_t)(uint8_t)bytes[offset + 3] << 24);
}

TEST(Aie2pTileImageTest, WritesExecutableElfFromLeafContribution) {
  const uint8_t code[16] = {
      0x7f, 0x20, 0x4a, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x19, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };
  const loom_native_section_contribution_t section = {
      /*.section_name=*/IREE_SV(".text.kernel"),
      /*.section_type=*/LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
      /*.section_flags=*/LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
          LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR,
      /*.contribution_alignment=*/16,
      /*.entry_size=*/0,
      /*.link=*/0,
      /*.info=*/0,
      /*.contents=*/iree_make_const_byte_span(code, sizeof(code)),
  };
  const loom_native_object_symbol_t symbol = {
      /*.name=*/IREE_SV("kernel"),
      /*.section_contribution_index=*/0,
      /*.section_offset=*/4,
      /*.size=*/sizeof(code) - 4,
      /*.binding=*/LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
      /*.visibility=*/LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT,
      /*.kind=*/LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
  };
  const loom_aie2p_leaf_contribution_t contribution = {
      /*.object=*/
      {
          /*.sections=*/&section,
          /*.section_count=*/1,
          /*.symbols=*/&symbol,
          /*.symbol_count=*/1,
      },
      /*.realization=*/
      {
          /*.target_identity=*/LOOM_AIE2P_LEAF_TARGET_IDENTITY,
          /*.abi_identity=*/LOOM_AIE2P_LEAF_ABI_IDENTITY,
          /*.entry_symbol_index=*/0,
          /*.elf_machine=*/LOOM_XDNA_ELF_MACHINE_AIE,
          /*.target_generation=*/LOOM_XDNA_TARGET_GENERATION_AIE2P,
          /*.elf_flags=*/LOOM_XDNA_ELF_AIE2P_FLAGS,
          /*.capability_flags=*/0,
          /*.code=*/{sizeof(code), 16},
      },
  };

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  iree_io_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      1024, iree_allocator_system(), &stream));
  StreamPtr stream_owner(stream, iree_io_stream_release);

  const loom_aie2p_tile_image_layout_t undersized_layout = {
      /*.program_address=*/0,
      /*.program_byte_capacity=*/sizeof(code) - 1u,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_aie2p_tile_image_write(
                            &contribution, &undersized_layout, stream, &arena));
  const loom_aie2p_tile_image_layout_t layout = {
      /*.program_address=*/0,
      /*.program_byte_capacity=*/16 * 1024,
  };
  IREE_ASSERT_OK(
      loom_aie2p_tile_image_write(&contribution, &layout, stream, &arena));
  const iree_io_stream_pos_t length = iree_io_stream_length(stream);
  ASSERT_GT(length, 0);
  std::string bytes((size_t)length, '\0');
  IREE_ASSERT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(
      iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));

  EXPECT_EQ(bytes.substr(0, 4), std::string("\x7f"
                                            "ELF",
                                            4));
  EXPECT_EQ((uint8_t)bytes[4], 1u);
  EXPECT_EQ(LoadLeU16(bytes, 16), LOOM_NATIVE_ELF_FILE_TYPE_EXEC);
  EXPECT_EQ(LoadLeU16(bytes, 18), LOOM_NATIVE_ELF_MACHINE_AIE);
  EXPECT_EQ(LoadLeU32(bytes, 24), 4u);
  EXPECT_EQ(LoadLeU32(bytes, 28), 52u);
  EXPECT_EQ(LoadLeU32(bytes, 36), 3u);
  EXPECT_EQ(LoadLeU16(bytes, 42), 32u);
  EXPECT_EQ(LoadLeU16(bytes, 44), 1u);

  constexpr size_t kProgramHeaderOffset = 52;
  EXPECT_EQ(LoadLeU32(bytes, kProgramHeaderOffset + 0),
            LOOM_NATIVE_ELF_PROGRAM_TYPE_LOAD);
  const uint32_t code_offset = LoadLeU32(bytes, kProgramHeaderOffset + 4);
  EXPECT_EQ(code_offset % 16, 0u);
  EXPECT_EQ(LoadLeU32(bytes, kProgramHeaderOffset + 8), 0u);
  EXPECT_EQ(LoadLeU32(bytes, kProgramHeaderOffset + 12), 0u);
  EXPECT_EQ(LoadLeU32(bytes, kProgramHeaderOffset + 16), sizeof(code));
  EXPECT_EQ(LoadLeU32(bytes, kProgramHeaderOffset + 20), sizeof(code));
  EXPECT_EQ(
      LoadLeU32(bytes, kProgramHeaderOffset + 24),
      LOOM_NATIVE_ELF_PROGRAM_FLAG_READ | LOOM_NATIVE_ELF_PROGRAM_FLAG_EXECUTE);
  EXPECT_EQ(LoadLeU32(bytes, kProgramHeaderOffset + 28), 16u);
  ASSERT_LE((size_t)code_offset + sizeof(code), bytes.size());
  EXPECT_EQ(0, std::memcmp(bytes.data() + code_offset, code, sizeof(code)));

  const uint32_t section_header_offset = LoadLeU32(bytes, 32);
  const uint16_t section_header_count = LoadLeU16(bytes, 48);
  ASSERT_EQ(section_header_count, 5u);
  constexpr size_t kSectionHeaderSize = 40;
  const size_t string_table_section = section_header_offset + 2 * 40;
  const size_t symbol_table_section = section_header_offset + 3 * 40;
  EXPECT_EQ(LoadLeU32(bytes, string_table_section + 4),
            LOOM_NATIVE_ELF_SECTION_TYPE_STRTAB);
  EXPECT_EQ(LoadLeU32(bytes, symbol_table_section + 4),
            LOOM_NATIVE_ELF_SECTION_TYPE_SYMTAB);
  EXPECT_EQ(LoadLeU32(bytes, symbol_table_section + 24), 2u);
  EXPECT_EQ(LoadLeU32(bytes, symbol_table_section + 28), 1u);
  EXPECT_EQ(LoadLeU32(bytes, symbol_table_section + 36), 16u);

  const uint32_t symbol_table_offset =
      LoadLeU32(bytes, symbol_table_section + 16);
  const uint32_t symbol_table_size =
      LoadLeU32(bytes, symbol_table_section + 20);
  ASSERT_EQ(symbol_table_size, 32u);
  ASSERT_LE((size_t)symbol_table_offset + symbol_table_size, bytes.size());
  const size_t kernel_symbol = symbol_table_offset + 16;
  EXPECT_EQ(LoadLeU32(bytes, kernel_symbol + 0), 1u);
  EXPECT_EQ(LoadLeU32(bytes, kernel_symbol + 4), 4u);
  EXPECT_EQ(LoadLeU32(bytes, kernel_symbol + 8), sizeof(code) - 4);
  EXPECT_EQ((uint8_t)bytes[kernel_symbol + 12], 0x12u);
  EXPECT_EQ((uint8_t)bytes[kernel_symbol + 13], 0u);
  EXPECT_EQ(LoadLeU16(bytes, kernel_symbol + 14), 1u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(Aie2pTileImageTest, AppliesBranchFixupAfterContributionPlacement) {
  const std::array<uint8_t, 16> prefix_code = {
      0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a,
      0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a,
  };
  std::array<uint8_t, 32> function_code = {};
  function_code[0] = 0x84;
  const loom_native_section_contribution_t sections[] = {
      {
          /*.section_name=*/IREE_SV(".text.kernel"),
          /*.section_type=*/LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
          /*.section_flags=*/LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
              LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR,
          /*.contribution_alignment=*/16,
          /*.entry_size=*/0,
          /*.link=*/0,
          /*.info=*/0,
          /*.contents=*/
          iree_make_const_byte_span(prefix_code.data(), prefix_code.size()),
      },
      {
          /*.section_name=*/IREE_SV(".text.kernel"),
          /*.section_type=*/LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
          /*.section_flags=*/LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
              LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR,
          /*.contribution_alignment=*/16,
          /*.entry_size=*/0,
          /*.link=*/0,
          /*.info=*/0,
          /*.contents=*/
          iree_make_const_byte_span(function_code.data(), function_code.size()),
      },
  };
  const loom_native_object_symbol_t symbol = {
      /*.name=*/IREE_SV("kernel"),
      /*.section_contribution_index=*/1,
      /*.section_offset=*/0,
      /*.size=*/function_code.size(),
      /*.binding=*/LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
      /*.visibility=*/LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT,
      /*.kind=*/LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
  };
  const loom_native_object_fixup_t fixup = {
      /*.section_contribution_index=*/1,
      /*.section_offset=*/0,
      /*.relocation_kind=*/
      LOOM_AIE2P_NATIVE_RELOCATION_KIND_CORE_BRANCH_ABSOLUTE,
      /*.target_symbol_index=*/0,
      /*.addend=*/16,
  };
  const loom_aie2p_leaf_contribution_t contribution = {
      /*.object=*/
      {
          /*.sections=*/sections,
          /*.section_count=*/IREE_ARRAYSIZE(sections),
          /*.symbols=*/&symbol,
          /*.symbol_count=*/1,
          /*.fixups=*/&fixup,
          /*.fixup_count=*/1,
      },
      /*.realization=*/
      {
          /*.target_identity=*/LOOM_AIE2P_LEAF_TARGET_IDENTITY,
          /*.abi_identity=*/LOOM_AIE2P_LEAF_ABI_IDENTITY,
          /*.entry_symbol_index=*/0,
          /*.elf_machine=*/LOOM_XDNA_ELF_MACHINE_AIE,
          /*.target_generation=*/LOOM_XDNA_TARGET_GENERATION_AIE2P,
          /*.elf_flags=*/LOOM_XDNA_ELF_AIE2P_FLAGS,
          /*.capability_flags=*/LOOM_AIE2P_LEAF_CAPABILITY_FLAG_NATIVE_FIXUPS,
          /*.code=*/{48, 16},
      },
  };

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  iree_io_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      1024, iree_allocator_system(), &stream));
  StreamPtr stream_owner(stream, iree_io_stream_release);

  const loom_aie2p_tile_image_layout_t layout = {
      /*.program_address=*/0,
      /*.program_byte_capacity=*/16 * 1024,
  };
  IREE_ASSERT_OK(
      loom_aie2p_tile_image_write(&contribution, &layout, stream, &arena));
  const iree_io_stream_pos_t length = iree_io_stream_length(stream);
  std::string bytes((size_t)length, '\0');
  IREE_ASSERT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(
      iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));

  EXPECT_EQ(LoadLeU32(bytes, 24), 16u);
  constexpr size_t kProgramHeaderOffset = 52;
  const uint32_t code_offset = LoadLeU32(bytes, kProgramHeaderOffset + 4);
  EXPECT_EQ(LoadLeU32(bytes, kProgramHeaderOffset + 16), 48u);
  ASSERT_LE((size_t)code_offset + 48u, bytes.size());
  EXPECT_EQ(0, std::memcmp(bytes.data() + code_offset, prefix_code.data(),
                           prefix_code.size()));
  constexpr std::array<uint8_t, 6> kBranchToAddress32 = {
      0x84, 0x00, 0x00, 0x10, 0x00, 0x00,
  };
  EXPECT_EQ(
      0, std::memcmp(bytes.data() + code_offset + 16, kBranchToAddress32.data(),
                     kBranchToAddress32.size()));
  EXPECT_EQ(function_code[0], 0x84u);
  for (size_t i = 1; i < 6; ++i) EXPECT_EQ(function_code[i], 0u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(Aie2pTileImageTest, RelocatesMovxmInsideAMultiSlotBundle) {
  const loom_aie2p_instruction_id_t movxm =
      loom_aie2p_encoding_find_instruction(IREE_SV("MOVXM"));
  const loom_aie2p_instruction_id_t vector_load =
      loom_aie2p_encoding_find_instruction(IREE_SV("VLDA_dmx_lda_x_idx_imm"));
  const loom_aie2p_encoding_field_id_t dst =
      loom_aie2p_encoding_find_field(IREE_SV("dst"));
  const loom_aie2p_encoding_field_id_t immediate =
      loom_aie2p_encoding_find_field(IREE_SV("i"));
  const loom_aie2p_encoding_field_id_t load_immediate =
      loom_aie2p_encoding_find_field(IREE_SV("imm"));
  const loom_aie2p_encoding_field_id_t pointer =
      loom_aie2p_encoding_find_field(IREE_SV("ptr"));
  ASSERT_NE(movxm, LOOM_AIE2P_INSTRUCTION_ID_INVALID);
  ASSERT_NE(vector_load, LOOM_AIE2P_INSTRUCTION_ID_INVALID);

  const loom_aie2p_encoding_field_value_t movxm_fields[] = {
      {dst, 0},
      {immediate, 0},
  };
  uint64_t movxm_value = 0;
  IREE_ASSERT_OK(loom_aie2p_encoding_pack_instruction(
      movxm, movxm_fields, IREE_ARRAYSIZE(movxm_fields), &movxm_value));
  const loom_aie2p_encoding_field_value_t load_fields[] = {
      {dst, 0},
      {load_immediate, 0},
      {pointer, 1},
  };
  uint64_t load_value = 0;
  IREE_ASSERT_OK(loom_aie2p_encoding_pack_instruction(
      vector_load, load_fields, IREE_ARRAYSIZE(load_fields), &load_value));
  const loom_aie2p_encoded_slot_t encoded_slots[] = {
      {LOOM_AIE2P_SLOT_LDA, load_value},
      {LOOM_AIE2P_SLOT_LNG, movxm_value},
  };
  const loom_aie2p_bundle_format_id_t bundle_format =
      loom_aie2p_encoding_find_bundle_format(IREE_SV("I80_LDA_LNG"));
  loom_aie2p_encoding_packet_t packet;
  IREE_ASSERT_OK(loom_aie2p_encoding_pack_bundle(
      bundle_format, encoded_slots, IREE_ARRAYSIZE(encoded_slots), &packet));
  ASSERT_EQ(packet.data_length, 10u);

  const loom_native_section_contribution_t sections[] = {
      {
          /*.section_name=*/IREE_SV(".text.multi_slot"),
          /*.section_type=*/LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
          /*.section_flags=*/LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
              LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR,
          /*.contribution_alignment=*/16,
          /*.entry_size=*/0,
          /*.link=*/0,
          /*.info=*/0,
          /*.contents=*/
          iree_make_const_byte_span(packet.data, packet.data_length),
      },
      {
          /*.section_name=*/IREE_SV(".storage.multi_slot.workgroup"),
          /*.section_type=*/LOOM_NATIVE_ELF_SECTION_TYPE_NOBITS,
          /*.section_flags=*/LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
              LOOM_NATIVE_ELF_SECTION_FLAG_WRITE,
          /*.contribution_alignment=*/64,
          /*.entry_size=*/0,
          /*.link=*/0,
          /*.info=*/0,
          /*.contents=*/iree_const_byte_span_empty(),
          /*.zero_fill_length=*/320,
      },
  };
  const loom_native_object_symbol_t symbol = {
      /*.name=*/IREE_SV("multi_slot.workgroup"),
      /*.section_contribution_index=*/1,
      /*.section_offset=*/0,
      /*.size=*/320,
      /*.binding=*/LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL,
      /*.visibility=*/LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_HIDDEN,
      /*.kind=*/LOOM_NATIVE_OBJECT_SYMBOL_KIND_DATA,
  };
  const loom_native_object_fixup_t fixup = {
      /*.section_contribution_index=*/0,
      /*.section_offset=*/0,
      /*.relocation_kind=*/
      LOOM_AIE2P_NATIVE_RELOCATION_KIND_LOCAL_ADDRESS_ABSOLUTE,
      /*.target_symbol_index=*/0,
      /*.addend=*/144,
  };
  const loom_native_object_contribution_t object = {
      /*.sections=*/sections,
      /*.section_count=*/IREE_ARRAYSIZE(sections),
      /*.symbols=*/&symbol,
      /*.symbol_count=*/1,
      /*.fixups=*/&fixup,
      /*.fixup_count=*/1,
  };

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_native_section_contribution_assembly_t assembly = {};
  IREE_ASSERT_OK(loom_native_assemble_section_contributions(
      object.sections, object.section_count, &assembly, &arena));
  ASSERT_EQ(assembly.section_count, 2u);
  assembly.sections[0].address = 0;
  assembly.sections[1].address = 0x70000;
  IREE_ASSERT_OK(
      loom_aie2p_native_object_apply_fixups(&object, &assembly, &arena));

  loom_aie2p_decoded_bundle_t decoded_bundle;
  IREE_ASSERT_OK(loom_aie2p_encoding_decode_bundle(
      assembly.sections[0].contents, &decoded_bundle));
  ASSERT_EQ(decoded_bundle.format, bundle_format);
  ASSERT_EQ(decoded_bundle.slot_count, 2u);
  const loom_aie2p_encoded_slot_t* relocated_movxm = nullptr;
  for (uint8_t i = 0; i < decoded_bundle.slot_count; ++i) {
    const loom_aie2p_encoded_slot_t& slot = decoded_bundle.slots[i];
    if (slot.slot == LOOM_AIE2P_SLOT_LDA) {
      EXPECT_EQ(slot.value, load_value);
    } else if (slot.slot == LOOM_AIE2P_SLOT_LNG) {
      relocated_movxm = &slot;
    }
  }
  ASSERT_NE(relocated_movxm, nullptr);
  loom_aie2p_encoding_field_value_t relocated_fields[2];
  iree_host_size_t relocated_field_count = 0;
  IREE_ASSERT_OK(loom_aie2p_encoding_unpack_instruction(
      movxm, relocated_movxm->value, IREE_ARRAYSIZE(relocated_fields),
      relocated_fields, &relocated_field_count));
  ASSERT_EQ(relocated_field_count, 2u);
  bool found_relocated_immediate = false;
  for (iree_host_size_t i = 0; i < relocated_field_count; ++i) {
    if (relocated_fields[i].field_id == immediate) {
      EXPECT_EQ(relocated_fields[i].value, 0x70090u);
      found_relocated_immediate = true;
    }
  }
  EXPECT_TRUE(found_relocated_immediate);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(Aie2pTileImageTest, PlacesAndRelocatesFunctionLocalStorage) {
  const std::array<uint8_t, 6> code = {
      0x44, 0x00, 0xc0, 0x00, 0x00, 0x00,
  };
  const loom_native_section_contribution_t sections[] = {
      {
          /*.section_name=*/IREE_SV(".text.local_address"),
          /*.section_type=*/LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
          /*.section_flags=*/LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
              LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR,
          /*.contribution_alignment=*/16,
          /*.entry_size=*/0,
          /*.link=*/0,
          /*.info=*/0,
          /*.contents=*/iree_make_const_byte_span(code.data(), code.size()),
      },
      {
          /*.section_name=*/IREE_SV(".storage.local_address.workgroup"),
          /*.section_type=*/LOOM_NATIVE_ELF_SECTION_TYPE_NOBITS,
          /*.section_flags=*/LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
              LOOM_NATIVE_ELF_SECTION_FLAG_WRITE,
          /*.contribution_alignment=*/64,
          /*.entry_size=*/0,
          /*.link=*/0,
          /*.info=*/0,
          /*.contents=*/iree_const_byte_span_empty(),
          /*.zero_fill_length=*/320,
      },
  };
  const loom_native_object_symbol_t symbols[] = {
      {
          /*.name=*/IREE_SV("local_address"),
          /*.section_contribution_index=*/0,
          /*.section_offset=*/0,
          /*.size=*/code.size(),
          /*.binding=*/LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
          /*.visibility=*/LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT,
          /*.kind=*/LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
      },
      {
          /*.name=*/IREE_SV("local_address.workgroup"),
          /*.section_contribution_index=*/1,
          /*.section_offset=*/0,
          /*.size=*/320,
          /*.binding=*/LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL,
          /*.visibility=*/LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_HIDDEN,
          /*.kind=*/LOOM_NATIVE_OBJECT_SYMBOL_KIND_DATA,
      },
  };
  const loom_native_object_fixup_t fixup = {
      /*.section_contribution_index=*/0,
      /*.section_offset=*/0,
      /*.relocation_kind=*/
      LOOM_AIE2P_NATIVE_RELOCATION_KIND_LOCAL_ADDRESS_ABSOLUTE,
      /*.target_symbol_index=*/1,
      /*.addend=*/144,
  };
  const loom_aie2p_leaf_storage_domain_t storage_domain = {
      /*.storage_space=*/LOOM_STORAGE_SPACE_WORKGROUP,
      /*.section_contribution_index=*/1,
      /*.symbol_index=*/1,
  };
  const loom_aie2p_leaf_contribution_t contribution = {
      /*.object=*/
      {
          /*.sections=*/sections,
          /*.section_count=*/IREE_ARRAYSIZE(sections),
          /*.symbols=*/symbols,
          /*.symbol_count=*/IREE_ARRAYSIZE(symbols),
          /*.fixups=*/&fixup,
          /*.fixup_count=*/1,
      },
      /*.realization=*/
      {
          /*.target_identity=*/LOOM_AIE2P_LEAF_TARGET_IDENTITY,
          /*.abi_identity=*/LOOM_AIE2P_LEAF_ABI_IDENTITY,
          /*.entry_symbol_index=*/0,
          /*.elf_machine=*/LOOM_XDNA_ELF_MACHINE_AIE,
          /*.target_generation=*/LOOM_XDNA_TARGET_GENERATION_AIE2P,
          /*.elf_flags=*/LOOM_XDNA_ELF_AIE2P_FLAGS,
          /*.capability_flags=*/
          LOOM_AIE2P_LEAF_CAPABILITY_FLAG_NATIVE_FIXUPS |
              LOOM_AIE2P_LEAF_CAPABILITY_FLAG_FUNCTION_STORAGE,
          /*.code=*/{code.size(), 16},
          /*.read_only_data=*/{},
          /*.initialized_data=*/{},
          /*.zero_fill=*/{},
          /*.stack=*/{},
          /*.scratch=*/{},
          /*.private_storage=*/{},
          /*.workgroup_storage=*/{320, 64},
          /*.spill=*/{},
          /*.storage_domains=*/&storage_domain,
          /*.storage_domain_count=*/1,
      },
  };
  const loom_aie2p_tile_storage_placement_t storage_placement = {
      /*.storage_space=*/LOOM_STORAGE_SPACE_WORKGROUP,
      /*.load_address=*/0x70000,
  };
  const loom_aie2p_tile_image_layout_t layout = {
      /*.program_address=*/0,
      /*.program_byte_capacity=*/16 * 1024,
      /*.storage_placements=*/&storage_placement,
      /*.storage_placement_count=*/1,
  };

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  iree_io_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      1024, iree_allocator_system(), &stream));
  StreamPtr stream_owner(stream, iree_io_stream_release);

  IREE_ASSERT_OK(
      loom_aie2p_tile_image_write(&contribution, &layout, stream, &arena));
  const iree_io_stream_pos_t length = iree_io_stream_length(stream);
  std::string bytes((size_t)length, '\0');
  IREE_ASSERT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(
      iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));

  EXPECT_EQ(LoadLeU16(bytes, 44), 2u);
  constexpr size_t kProgramHeaderOffset = 52;
  const uint32_t code_offset = LoadLeU32(bytes, kProgramHeaderOffset + 4);
  EXPECT_EQ(LoadLeU32(bytes, kProgramHeaderOffset + 8), 0u);
  EXPECT_EQ(LoadLeU32(bytes, kProgramHeaderOffset + 16), code.size());
  EXPECT_EQ(LoadLeU32(bytes, kProgramHeaderOffset + 20), code.size());
  EXPECT_EQ(
      LoadLeU32(bytes, kProgramHeaderOffset + 24),
      LOOM_NATIVE_ELF_PROGRAM_FLAG_READ | LOOM_NATIVE_ELF_PROGRAM_FLAG_EXECUTE);

  constexpr size_t kStorageProgramHeaderOffset = kProgramHeaderOffset + 32;
  EXPECT_EQ(LoadLeU32(bytes, kStorageProgramHeaderOffset + 8), 0x70000u);
  EXPECT_EQ(LoadLeU32(bytes, kStorageProgramHeaderOffset + 12), 0x70000u);
  EXPECT_EQ(LoadLeU32(bytes, kStorageProgramHeaderOffset + 16), 0u);
  EXPECT_EQ(LoadLeU32(bytes, kStorageProgramHeaderOffset + 20), 320u);
  EXPECT_EQ(
      LoadLeU32(bytes, kStorageProgramHeaderOffset + 24),
      LOOM_NATIVE_ELF_PROGRAM_FLAG_READ | LOOM_NATIVE_ELF_PROGRAM_FLAG_WRITE);
  EXPECT_EQ(LoadLeU32(bytes, kStorageProgramHeaderOffset + 28), 64u);

  constexpr std::array<uint8_t, 6> kMovxmLocalAddress = {
      0x44, 0x20, 0xc1, 0x00, 0x07, 0x00,
  };
  ASSERT_LE((size_t)code_offset + kMovxmLocalAddress.size(), bytes.size());
  EXPECT_EQ(0,
            std::memcmp(bytes.data() + code_offset, kMovxmLocalAddress.data(),
                        kMovxmLocalAddress.size()));

  const uint32_t section_header_offset = LoadLeU32(bytes, 32);
  ASSERT_EQ(LoadLeU16(bytes, 48), 6u);
  constexpr size_t kSectionHeaderSize = 40;
  const size_t storage_section = section_header_offset + 2 * kSectionHeaderSize;
  EXPECT_EQ(LoadLeU32(bytes, storage_section + 4),
            LOOM_NATIVE_ELF_SECTION_TYPE_NOBITS);
  EXPECT_EQ(LoadLeU32(bytes, storage_section + 12), 0x70000u);
  EXPECT_EQ(LoadLeU32(bytes, storage_section + 20), 320u);
  EXPECT_EQ(LoadLeU32(bytes, storage_section + 32), 64u);
  const size_t symbol_table_section =
      section_header_offset + 4 * kSectionHeaderSize;
  EXPECT_EQ(LoadLeU32(bytes, symbol_table_section + 24), 3u);
  EXPECT_EQ(LoadLeU32(bytes, symbol_table_section + 28), 2u);
  const uint32_t symbol_table_offset =
      LoadLeU32(bytes, symbol_table_section + 16);
  ASSERT_EQ(LoadLeU32(bytes, symbol_table_section + 20), 48u);
  const size_t storage_symbol = symbol_table_offset + 16;
  EXPECT_EQ(LoadLeU32(bytes, storage_symbol + 4), 0x70000u);
  EXPECT_EQ(LoadLeU32(bytes, storage_symbol + 8), 320u);
  EXPECT_EQ((uint8_t)bytes[storage_symbol + 12], 0x01u);
  EXPECT_EQ((uint8_t)bytes[storage_symbol + 13], 0x02u);
  EXPECT_EQ(LoadLeU16(bytes, storage_symbol + 14), 2u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
