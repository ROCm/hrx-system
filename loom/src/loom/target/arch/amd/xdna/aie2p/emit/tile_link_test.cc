// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/tile_link.h"

#include <array>
#include <cstring>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/relocation.h"
#include "loom/target/arch/amd/xdna/aie2p/encoding/encoding.h"

namespace loom {
namespace {

TEST(Aie2pTileLinkTest, PlacesExecutableContribution) {
  const uint8_t code[16] = {
      0x7f, 0x20, 0x4a, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x19, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };
  const loom_native_section_contribution_t section = {
      /*.section_name=*/IREE_SV(".text.kernel"),
      /*.storage=*/LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
      /*.access=*/LOOM_NATIVE_SECTION_ACCESS_READ |
          LOOM_NATIVE_SECTION_ACCESS_EXECUTE,
      /*.contribution_alignment=*/16,
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
          /*.capability_flags=*/0,
          /*.code=*/{sizeof(code), 16},
      },
  };

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  const loom_aie2p_tile_link_layout_t undersized_layout = {
      /*.program_address=*/0,
      /*.program_owner_offset=*/0,
      /*.program_byte_capacity=*/sizeof(code) - 1u,
  };
  loom_aie2p_linked_tile_t linked_tile = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_aie2p_tile_link(&contribution, &undersized_layout,
                                             &arena, &linked_tile));
  const loom_aie2p_tile_link_layout_t layout = {
      /*.program_address=*/0,
      /*.program_owner_offset=*/0,
      /*.program_byte_capacity=*/16 * 1024,
  };
  IREE_ASSERT_OK(
      loom_aie2p_tile_link(&contribution, &layout, &arena, &linked_tile));
  ASSERT_EQ(linked_tile.assembly.section_count, 1u);
  EXPECT_EQ(linked_tile.entry_section_index, 0u);
  EXPECT_EQ(linked_tile.entry_address, 4u);
  EXPECT_EQ(linked_tile.assembly.sections[0].address, 0u);
  ASSERT_EQ(linked_tile.section_placement_count, 1u);
  EXPECT_EQ(linked_tile.section_placements[0].memory_space,
            LOOM_XDNA_MEMORY_SPACE_PROGRAM);
  EXPECT_EQ(linked_tile.section_placements[0].owner_offset, 0u);
  ASSERT_EQ(linked_tile.assembly.sections[0].contents.data_length,
            sizeof(code));
  EXPECT_EQ(0, std::memcmp(linked_tile.assembly.sections[0].contents.data, code,
                           sizeof(code)));

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(Aie2pTileLinkTest, AppliesBranchFixupAfterContributionPlacement) {
  const std::array<uint8_t, 16> prefix_code = {
      0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a,
      0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a,
  };
  std::array<uint8_t, 32> function_code = {};
  function_code[0] = 0x84;
  const loom_native_section_contribution_t sections[] = {
      {
          /*.section_name=*/IREE_SV(".text.kernel"),
          /*.storage=*/LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
          /*.access=*/LOOM_NATIVE_SECTION_ACCESS_READ |
              LOOM_NATIVE_SECTION_ACCESS_EXECUTE,
          /*.contribution_alignment=*/16,
          /*.contents=*/
          iree_make_const_byte_span(prefix_code.data(), prefix_code.size()),
      },
      {
          /*.section_name=*/IREE_SV(".text.kernel"),
          /*.storage=*/LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
          /*.access=*/LOOM_NATIVE_SECTION_ACCESS_READ |
              LOOM_NATIVE_SECTION_ACCESS_EXECUTE,
          /*.contribution_alignment=*/16,
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
          /*.capability_flags=*/LOOM_AIE2P_LEAF_CAPABILITY_FLAG_NATIVE_FIXUPS,
          /*.code=*/{48, 16},
      },
  };

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  const loom_aie2p_tile_link_layout_t layout = {
      /*.program_address=*/0,
      /*.program_owner_offset=*/0,
      /*.program_byte_capacity=*/16 * 1024,
  };
  loom_aie2p_linked_tile_t linked_tile = {};
  IREE_ASSERT_OK(
      loom_aie2p_tile_link(&contribution, &layout, &arena, &linked_tile));
  ASSERT_EQ(linked_tile.assembly.section_count, 1u);
  ASSERT_EQ(linked_tile.assembly.sections[0].contents.data_length, 48u);
  const uint8_t* linked_code = linked_tile.assembly.sections[0].contents.data;
  EXPECT_EQ(0,
            std::memcmp(linked_code, prefix_code.data(), prefix_code.size()));
  constexpr std::array<uint8_t, 6> kBranchToAddress32 = {
      0x84, 0x00, 0x00, 0x10, 0x00, 0x00,
  };
  EXPECT_EQ(0, std::memcmp(linked_code + 16, kBranchToAddress32.data(),
                           kBranchToAddress32.size()));
  EXPECT_EQ(function_code[0], 0x84u);
  for (size_t i = 1; i < 6; ++i) {
    EXPECT_EQ(function_code[i], 0u);
  }

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(Aie2pTileLinkTest, RelocatesMovxmInsideAMultiSlotBundle) {
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
          /*.storage=*/LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
          /*.access=*/LOOM_NATIVE_SECTION_ACCESS_READ |
              LOOM_NATIVE_SECTION_ACCESS_EXECUTE,
          /*.contribution_alignment=*/16,
          /*.contents=*/
          iree_make_const_byte_span(packet.data, packet.data_length),
      },
      {
          /*.section_name=*/IREE_SV(".storage.multi_slot.workgroup"),
          /*.storage=*/LOOM_NATIVE_SECTION_STORAGE_RESERVATION,
          /*.access=*/LOOM_NATIVE_SECTION_ACCESS_READ |
              LOOM_NATIVE_SECTION_ACCESS_WRITE,
          /*.contribution_alignment=*/64,
          /*.contents=*/iree_const_byte_span_empty(),
          /*.reservation_length=*/320,
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

TEST(Aie2pTileLinkTest, PlacesAndRelocatesReadOnlyData) {
  const std::array<uint8_t, 6> code = {
      0x44, 0x00, 0xc0, 0x00, 0x00, 0x00,
  };
  const std::array<uint8_t, 5> table = {
      0x11, 0x22, 0x33, 0x44, 0x55,
  };
  const loom_native_section_contribution_t sections[] = {
      {
          /*.section_name=*/IREE_SV(".text.read_table"),
          /*.storage=*/LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
          /*.access=*/LOOM_NATIVE_SECTION_ACCESS_READ |
              LOOM_NATIVE_SECTION_ACCESS_EXECUTE,
          /*.contribution_alignment=*/16,
          /*.contents=*/iree_make_const_byte_span(code.data(), code.size()),
      },
      {
          /*.section_name=*/IREE_SV(".rodata.table"),
          /*.storage=*/LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
          /*.access=*/LOOM_NATIVE_SECTION_ACCESS_READ,
          /*.contribution_alignment=*/32,
          /*.contents=*/iree_make_const_byte_span(table.data(), table.size()),
      },
  };
  const loom_native_object_symbol_t symbols[] = {
      {
          /*.name=*/IREE_SV("read_table"),
          /*.section_contribution_index=*/0,
          /*.section_offset=*/0,
          /*.size=*/code.size(),
          /*.binding=*/LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
          /*.visibility=*/LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT,
          /*.kind=*/LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
      },
      {
          /*.name=*/IREE_SV("table"),
          /*.section_contribution_index=*/1,
          /*.section_offset=*/0,
          /*.size=*/table.size(),
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
      /*.addend=*/0,
  };
  const loom_aie2p_leaf_read_only_data_domain_t read_only_data = {
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
          /*.capability_flags=*/
          LOOM_AIE2P_LEAF_CAPABILITY_FLAG_NATIVE_FIXUPS |
              LOOM_AIE2P_LEAF_CAPABILITY_FLAG_READ_ONLY_DATA,
          /*.code=*/{code.size(), 16},
          /*.read_only_data=*/&read_only_data,
          /*.read_only_data_count=*/1,
      },
  };
  const loom_aie2p_tile_read_only_data_placement_t data_placement = {
      /*.owner_offset=*/0x4020,
      /*.load_address=*/0x74020,
      /*.byte_length=*/table.size(),
  };
  const loom_aie2p_tile_link_layout_t layout = {
      /*.program_address=*/0,
      /*.program_owner_offset=*/0,
      /*.program_byte_capacity=*/16 * 1024,
      /*.storage_placements=*/nullptr,
      /*.storage_placement_count=*/0,
      /*.read_only_data_placements=*/&data_placement,
      /*.read_only_data_placement_count=*/1,
  };

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_aie2p_linked_tile_t linked_tile = {};
  IREE_ASSERT_OK(
      loom_aie2p_tile_link(&contribution, &layout, &arena, &linked_tile));

  ASSERT_EQ(linked_tile.assembly.section_count, 2u);
  const loom_native_section_t& linked_code = linked_tile.assembly.sections[0];
  const loom_native_section_t& linked_table = linked_tile.assembly.sections[1];
  EXPECT_EQ(linked_table.address, data_placement.load_address);
  EXPECT_EQ(linked_table.storage, LOOM_NATIVE_SECTION_STORAGE_CONTENTS);
  EXPECT_EQ(linked_table.access, LOOM_NATIVE_SECTION_ACCESS_READ);
  ASSERT_EQ(linked_tile.section_placement_count, 2u);
  EXPECT_EQ(linked_tile.section_placements[0].memory_space,
            LOOM_XDNA_MEMORY_SPACE_PROGRAM);
  EXPECT_EQ(linked_tile.section_placements[0].owner_offset, 0u);
  EXPECT_EQ(linked_tile.section_placements[1].memory_space,
            LOOM_XDNA_MEMORY_SPACE_DATA);
  EXPECT_EQ(linked_tile.section_placements[1].owner_offset,
            data_placement.owner_offset);
  ASSERT_EQ(linked_table.contents.data_length, table.size());
  EXPECT_EQ(
      0, std::memcmp(linked_table.contents.data, table.data(), table.size()));

  loom_aie2p_decoded_bundle_t decoded_bundle;
  IREE_ASSERT_OK(
      loom_aie2p_encoding_decode_bundle(linked_code.contents, &decoded_bundle));
  ASSERT_EQ(decoded_bundle.slot_count, 1u);
  const loom_aie2p_instruction_id_t movxm =
      loom_aie2p_encoding_find_instruction(IREE_SV("MOVXM"));
  const loom_aie2p_encoding_field_id_t immediate =
      loom_aie2p_encoding_find_field(IREE_SV("i"));
  loom_aie2p_encoding_field_value_t fields[2];
  iree_host_size_t field_count = 0;
  IREE_ASSERT_OK(loom_aie2p_encoding_unpack_instruction(
      movxm, decoded_bundle.slots[0].value, IREE_ARRAYSIZE(fields), fields,
      &field_count));
  bool found_immediate = false;
  for (iree_host_size_t i = 0; i < field_count; ++i) {
    if (fields[i].field_id == immediate) {
      EXPECT_EQ(fields[i].value, data_placement.load_address);
      found_immediate = true;
    }
  }
  EXPECT_TRUE(found_immediate);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(Aie2pTileLinkTest, PlacesAndRelocatesFunctionLocalStorage) {
  const std::array<uint8_t, 6> code = {
      0x44, 0x00, 0xc0, 0x00, 0x00, 0x00,
  };
  const loom_native_section_contribution_t sections[] = {
      {
          /*.section_name=*/IREE_SV(".text.local_address"),
          /*.storage=*/LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
          /*.access=*/LOOM_NATIVE_SECTION_ACCESS_READ |
              LOOM_NATIVE_SECTION_ACCESS_EXECUTE,
          /*.contribution_alignment=*/16,
          /*.contents=*/iree_make_const_byte_span(code.data(), code.size()),
      },
      {
          /*.section_name=*/IREE_SV(".storage.local_address.workgroup"),
          /*.storage=*/LOOM_NATIVE_SECTION_STORAGE_RESERVATION,
          /*.access=*/LOOM_NATIVE_SECTION_ACCESS_READ |
              LOOM_NATIVE_SECTION_ACCESS_WRITE,
          /*.contribution_alignment=*/64,
          /*.contents=*/iree_const_byte_span_empty(),
          /*.reservation_length=*/320,
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
          /*.capability_flags=*/
          LOOM_AIE2P_LEAF_CAPABILITY_FLAG_NATIVE_FIXUPS |
              LOOM_AIE2P_LEAF_CAPABILITY_FLAG_FUNCTION_STORAGE,
          /*.code=*/{code.size(), 16},
          /*.read_only_data=*/{},
          /*.read_only_data_count=*/0,
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
      /*.owner_offset=*/0,
      /*.load_address=*/0x70000,
  };
  const loom_aie2p_tile_link_layout_t layout = {
      /*.program_address=*/0,
      /*.program_owner_offset=*/0,
      /*.program_byte_capacity=*/16 * 1024,
      /*.storage_placements=*/&storage_placement,
      /*.storage_placement_count=*/1,
  };

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_aie2p_linked_tile_t linked_tile = {};
  IREE_ASSERT_OK(
      loom_aie2p_tile_link(&contribution, &layout, &arena, &linked_tile));
  ASSERT_EQ(linked_tile.assembly.section_count, 2u);
  EXPECT_EQ(linked_tile.entry_section_index, 0u);
  const loom_native_section_t& linked_code = linked_tile.assembly.sections[0];
  const loom_native_section_t& linked_storage =
      linked_tile.assembly.sections[1];
  EXPECT_EQ(linked_code.address, 0u);
  EXPECT_EQ(linked_storage.address, 0x70000u);
  EXPECT_EQ(linked_storage.storage, LOOM_NATIVE_SECTION_STORAGE_RESERVATION);
  EXPECT_EQ(linked_storage.reservation_length, 320u);
  EXPECT_EQ(linked_storage.alignment, 64u);
  ASSERT_EQ(linked_tile.section_placement_count, 2u);
  EXPECT_EQ(linked_tile.section_placements[1].memory_space,
            LOOM_XDNA_MEMORY_SPACE_DATA);
  EXPECT_EQ(linked_tile.section_placements[1].owner_offset, 0u);

  constexpr std::array<uint8_t, 6> kMovxmLocalAddress = {
      0x44, 0x20, 0xc1, 0x00, 0x07, 0x00,
  };
  ASSERT_EQ(linked_code.contents.data_length, kMovxmLocalAddress.size());
  EXPECT_EQ(0, std::memcmp(linked_code.contents.data, kMovxmLocalAddress.data(),
                           kMovxmLocalAddress.size()));
  ASSERT_EQ(linked_tile.symbol_layout_count, 2u);
  EXPECT_EQ(linked_tile.symbol_layouts[1].section_index, 1u);
  EXPECT_EQ(linked_tile.symbol_layouts[1].section_offset, 0u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
