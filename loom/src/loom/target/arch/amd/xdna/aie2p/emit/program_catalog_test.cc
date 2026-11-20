// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program_catalog.h"

#include <initializer_list>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class Aie2pProgramCatalogTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool);
    iree_arena_initialize(&pool, &arena);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena);
    iree_arena_block_pool_deinitialize(&pool);
  }

  iree_arena_block_pool_t pool;
  iree_arena_allocator_t arena;
};

TEST_F(Aie2pProgramCatalogTest, SharesRelocatedStreamsAndPreservesInvocation) {
  const auto* family = loom_xdna_npu2_array_family();
  std::vector<loom_aie2p_execution_tile_t> occupancy(family->column_count *
                                                     family->row_count);
  auto& shim = occupancy[loom_xdna_array_physical_tile_index(family, {0, 0})];
  auto& memory = occupancy[loom_xdna_array_physical_tile_index(family, {0, 1})];
  shim.mm2s_count = shim.descriptor_counts[0] = 1;
  memory.s2mm_count = memory.descriptor_counts[0] = memory.lock_count = 1;
  memory.memory_length = 130;
  loom_aie2p_code_transport_t transport = {};
  transport.plan.tiles = occupancy.data();
  transport.plan.routing.family = family;
  const loom_aie2p_code_transport_binding_t binding = {
      .compute = {0, 2},
      .memory = {0, 1},
      .engine = 3,
      .first_descriptor = 24,
      .descriptor_count = 3,
      .queue_packet = 7,
      .code_packet = 3,
      .reply_packet = 23,
  };
  uint8_t bytes[18];
  for (size_t i = 0; i < sizeof(bytes); ++i) {
    bytes[i] = static_cast<uint8_t>(i + 1);
  }
  loom_native_section_t sections[4] = {};
  loom_native_section_contribution_layout_t contributions[4] = {};
  loom_aie2p_program_layout_group_t groups[4] = {};
  for (size_t i = 0; i < 4; ++i) {
    sections[i] = {
        .storage = LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
        .access = LOOM_NATIVE_SECTION_ACCESS_READ |
                  LOOM_NATIVE_SECTION_ACCESS_EXECUTE,
        .address = i == 3 ? 128u : 64u,
        .alignment = 16,
        .contents = iree_make_const_byte_span(bytes, sizeof(bytes)),
    };
    contributions[i].section_index = i;
    groups[i].native.section_contribution_index = i;
  }
  const loom_aie2p_program_layout_t layout = {
      .groups = groups,
      .group_count = 4,
  };
  const loom_aie2p_linked_tile_t linked = {
      .assembly =
          {
              .sections = sections,
              .section_count = 4,
              .contribution_layouts = contributions,
              .contribution_layout_count = 4,
          },
  };
  loom_aie2p_program_catalog_t catalog;
  iree_string_view_t requirement;
  IREE_ASSERT_OK(loom_aie2p_program_catalog_build(
      &transport, &binding, &layout, &linked, &arena, &catalog, &requirement));
  ASSERT_TRUE(iree_string_view_is_empty(requirement));
  EXPECT_EQ(catalog.offset, 132u);
  EXPECT_EQ(catalog.memory.column, binding.memory.column);
  EXPECT_EQ(catalog.memory.row, binding.memory.row);
  EXPECT_EQ(memory.memory_length, 220u);
  ASSERT_EQ(catalog.descriptor_count, 3u);
  const auto& first = catalog.descriptors[0].value.register_block_write32;
  const auto& repeated = catalog.descriptors[1].value.register_block_write32;
  const auto& relocated = catalog.descriptors[2].value.register_block_write32;
  EXPECT_EQ(first.address, 0x1a0000u + 24 * 32);
  EXPECT_EQ(repeated.address, first.address + 32);
  EXPECT_EQ(relocated.address, first.address + 64);
  EXPECT_EQ(first.words[0], 11u);
  EXPECT_EQ(repeated.words[1], first.words[1]);
  EXPECT_EQ(relocated.words[1], first.words[1] + 11);
  const auto& payload = catalog.contents;
  EXPECT_EQ(payload.address, 0x100000u + catalog.offset);
  ASSERT_EQ(payload.word_count, 22u);
  for (size_t i = 0; i < 2; ++i) {
    const auto* words = payload.words + i * 11;
    const uint32_t address = 0x20000u + (i == 0 ? 64 : 128);
    EXPECT_EQ(words[0] & 31u, binding.code_packet);
    EXPECT_EQ(words[6] & 31u, binding.code_packet);
    EXPECT_EQ(words[9] & 31u, binding.code_packet);
    EXPECT_EQ(words[1] & 0xfffffu, address);
    EXPECT_EQ((words[1] >> 20) & 3u, 3u);
    EXPECT_EQ((words[1] >> 22) & 3u, 0u);
    for (size_t j = 0; j < 16; ++j) {
      EXPECT_EQ((words[2 + j / 4] >> ((j % 4) * 8)) & 255u, bytes[j]);
    }
    EXPECT_EQ(words[7] & 0xfffffu, address + 16);
    EXPECT_EQ((words[7] >> 20) & 3u, 0u);
    EXPECT_EQ(words[8], static_cast<uint32_t>(bytes[16] | bytes[17] << 8));
    EXPECT_EQ(words[10] & 0xfffffu, 0x3ff30u);
    EXPECT_EQ((words[10] >> 22) & 3u, 1u);
    EXPECT_EQ((words[10] >> 24) & 31u, binding.reply_packet);
    for (size_t j : {0u, 1u, 6u, 7u, 9u, 10u}) {
      uint32_t parity = words[j];
      parity ^= parity >> 16;
      parity ^= parity >> 8;
      parity ^= parity >> 4;
      parity ^= parity >> 2;
      parity ^= parity >> 1;
      EXPECT_EQ(parity & 1u, 1u);
    }
  }
  const loom_aie2p_program_record_t initial = {
      .type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32,
      .value = {.register_write32 = {0x232000, 2}},
  };
  const loom_aie2p_program_record_t invoke = {
      .type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32,
      .value = {.register_write32 = {0x232000, 1}},
  };
  loom_aie2p_array_program_t program = {
      .array_records = &initial,
      .array_record_count = 1,
      .control_records = &invoke,
      .control_record_count = 1,
  };
  IREE_ASSERT_OK(loom_aie2p_program_catalog_install(&transport, &catalog, 1, 1,
                                                    &arena, &program));
  EXPECT_EQ(program.array_record_count, 7u);
  EXPECT_EQ(program.array_records[0].value.register_write32.value, 2u);
  EXPECT_EQ(program.array_records[1].value.register_write32.address,
            catalog.reset.address);
  EXPECT_EQ(program.array_records[1].value.register_write32.value,
            catalog.reset.value);
  EXPECT_EQ(program.array_records[5].value.register_block_write32.words,
            payload.words);
  EXPECT_EQ(program.array_records[6].value.register_write32.address,
            catalog.reset.address);
  EXPECT_EQ(program.array_records[6].value.register_write32.value, 0u);
  EXPECT_EQ(program.control_records, &invoke);
  EXPECT_EQ(program.control_record_count, 1u);
  EXPECT_EQ(program.upload, nullptr);

  const uint64_t full =
      loom_xdna_array_tile_facts(family, binding.memory)->memory.local_capacity;
  memory.memory_length = full;
  IREE_ASSERT_OK(loom_aie2p_program_catalog_build(
      &transport, &binding, &layout, &linked, &arena, &catalog, &requirement));
  EXPECT_FALSE(iree_string_view_is_empty(requirement));
  EXPECT_EQ(catalog.descriptor_count, 0u);
  EXPECT_EQ(memory.memory_length, full);
}

TEST_F(Aie2pProgramCatalogTest, UploadsPlacedCatalogsOnOwnedColumns) {
  const auto* family = loom_xdna_npu2_array_family();
  std::vector<loom_aie2p_execution_tile_t> occupancy(family->column_count *
                                                     family->row_count);
  loom_aie2p_code_transport_t transport = {};
  transport.plan.tiles = occupancy.data();
  transport.plan.routing.family = family;
  auto& shim = occupancy[loom_xdna_array_physical_tile_index(family, {2, 0})];
  auto& memory = occupancy[loom_xdna_array_physical_tile_index(family, {2, 1})];
  shim.mm2s_count = shim.descriptor_counts[0] = 1;
  memory.s2mm_count = memory.descriptor_counts[0] = memory.lock_count = 1;
  memory.memory_length = 1024;

  std::vector<uint8_t> code(4 * 1024);
  loom_native_section_t section = {
      .storage = LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
      .access =
          LOOM_NATIVE_SECTION_ACCESS_READ | LOOM_NATIVE_SECTION_ACCESS_EXECUTE,
      .alignment = 16,
  };
  loom_native_section_contribution_layout_t contribution = {};
  loom_aie2p_program_layout_group_t groups[2] = {};
  const loom_aie2p_program_layout_t layout = {
      .groups = groups,
      .group_count = 2,
  };
  const loom_aie2p_linked_tile_t linked = {
      .assembly =
          {
              .sections = &section,
              .section_count = 1,
              .contribution_layouts = &contribution,
              .contribution_layout_count = 1,
          },
  };
  // Only the combined catalogs in column two amortize a transfer. Column
  // five has enough data in a single catalog but no application-owned shim
  // ingress. Present the workers in reverse placement order to the installer.
  loom_aie2p_program_catalog_t catalogs[3] = {};
  const loom_xdna_tile_coordinate_t workers[] = {{2, 2}, {5, 2}, {2, 3}};
  for (uint16_t i = 0; i < 3; ++i) {
    code[0] = static_cast<uint8_t>(19 + i);
    section.contents = iree_make_const_byte_span(
        code.data(), workers[i].column == 2 ? 2 * 1024 : code.size());
    const loom_aie2p_code_transport_binding_t binding = {
        .compute = workers[i],
        .memory = {workers[i].column, 1},
        .engine = static_cast<uint16_t>(workers[i].row - 2),
        .first_descriptor = static_cast<uint16_t>(24 + i),
        .descriptor_count = 1,
        .code_packet = static_cast<uint8_t>(3 + i),
        .reply_packet = static_cast<uint8_t>(23 + i),
    };
    iree_string_view_t requirement;
    IREE_ASSERT_OK(loom_aie2p_program_catalog_build(
        &transport, &binding, &layout, &linked, &arena, &catalogs[2 - i],
        &requirement));
    ASSERT_TRUE(iree_string_view_is_empty(requirement));
  }
  const loom_aie2p_program_record_t initial[] = {
      {
          .type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32,
          .value = {.register_write32 = {0x232000, 2}},
      },
      {
          .type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32,
          .value = {.register_write32 = {0x1c000, 0}},
      },
  };
  loom_aie2p_array_program_t program = {
      .array_records = initial,
      .array_record_count = 2,
  };
  IREE_ASSERT_OK(loom_aie2p_program_catalog_install(&transport, catalogs, 3, 1,
                                                    &arena, &program));
  ASSERT_NE(program.upload, nullptr);
  const auto& upload = *program.upload;
  ASSERT_EQ(upload.patch_count, 1u);
  const auto& patch = upload.patches[0];
  ASSERT_LT(patch.record_index, upload.record_count);
  const auto& descriptor = upload.records[patch.record_index];
  ASSERT_EQ(descriptor.type, LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32);
  EXPECT_EQ(
      descriptor.value.register_block_write32.address >> family->column_shift,
      2u);
  EXPECT_EQ(descriptor.value.register_block_write32.words[0],
            patch.byte_length / 4);
  ASSERT_EQ(patch.source_offset, 0u);
  ASSERT_EQ(
      patch.byte_length,
      (catalogs[2].contents.word_count + catalogs[0].contents.word_count) * 4);
  ASSERT_EQ(upload.contents.data_length, patch.byte_length);
  for (size_t i : {2u, 0u}) {
    const auto& catalog = catalogs[i];
    const uint8_t* data =
        upload.contents.data + catalog.offset - catalogs[2].offset;
    for (size_t word = 0; word < catalog.contents.word_count; ++word) {
      ASSERT_EQ(iree_unaligned_load_le_u32(data + 4 * word),
                catalog.contents.words[word]);
    }
  }
  EXPECT_EQ(upload.records[0].value.register_write32.address,
            initial[0].value.register_write32.address);
  size_t inline_count = 0;
  for (size_t i = 0; i < upload.record_count; ++i) {
    const auto& record = upload.records[i];
    if (record.type != LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32) {
      continue;
    }
    for (const auto& catalog : catalogs) {
      if (record.value.register_block_write32.words == catalog.contents.words) {
        EXPECT_EQ(catalog.memory.column, 5u);
        ++inline_count;
      }
    }
  }
  EXPECT_EQ(inline_count, 1u);
}

}  // namespace
