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

TEST(Aie2pProgramCatalogTest, SharesRelocatedStreamsAndPreservesInvocation) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);
  const auto* family = loom_xdna_npu2_array_family();
  std::vector<loom_aie2p_execution_tile_t> occupancy(family->column_count *
                                                     family->row_count);
  occupancy[1].memory_length = 130;
  loom_aie2p_code_transport_t transport = {};
  transport.plan.tiles = occupancy.data();
  transport.plan.routing.family = family;
  const loom_aie2p_code_transport_binding_t binding = {.compute = {0, 2},
                                                       .memory = {0, 1},
                                                       .engine = 3,
                                                       .first_descriptor = 24,
                                                       .descriptor_count = 3,
                                                       .queue_packet = 7,
                                                       .reply_packet = 23};
  uint8_t bytes[18];
  for (size_t i = 0; i < sizeof(bytes); ++i) {
    bytes[i] = static_cast<uint8_t>(i + 1);
  }
  loom_native_section_t sections[4] = {};
  loom_native_section_contribution_layout_t contributions[4] = {};
  loom_aie2p_program_layout_group_t groups[4] = {};
  for (size_t i = 0; i < 4; ++i) {
    sections[i] = {.storage = LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
                   .access = LOOM_NATIVE_SECTION_ACCESS_READ |
                             LOOM_NATIVE_SECTION_ACCESS_EXECUTE,
                   .address = i == 3 ? 128u : 64u,
                   .alignment = 16,
                   .contents = iree_make_const_byte_span(bytes, sizeof(bytes))};
    contributions[i].section_index = i;
    groups[i].native.section_contribution_index = i;
  }
  const loom_aie2p_program_layout_t layout = {.groups = groups,
                                              .group_count = 4};
  const loom_aie2p_linked_tile_t linked = {
      .assembly = {.sections = sections,
                   .section_count = 4,
                   .contribution_layouts = contributions,
                   .contribution_layout_count = 4}};
  loom_aie2p_program_catalog_t catalog;
  iree_string_view_t requirement;
  IREE_ASSERT_OK(loom_aie2p_program_catalog_build(
      &transport, &binding, &layout, &linked, &arena, &catalog, &requirement));
  ASSERT_TRUE(iree_string_view_is_empty(requirement));
  EXPECT_EQ(catalog.offset, 132u);
  EXPECT_EQ(catalog.byte_length, 88u);
  EXPECT_EQ(occupancy[1].memory_length, 220u);
  ASSERT_EQ(catalog.record_count, 6u);
  const auto& first = catalog.records[1].value.register_block_write32;
  const auto& repeated = catalog.records[2].value.register_block_write32;
  const auto& relocated = catalog.records[3].value.register_block_write32;
  EXPECT_EQ(first.address, 0x1a0000u + 24 * 32);
  EXPECT_EQ(repeated.address, first.address + 32);
  EXPECT_EQ(relocated.address, first.address + 64);
  EXPECT_EQ(first.words[0], 11u);
  EXPECT_EQ(repeated.words[1], first.words[1]);
  EXPECT_EQ(relocated.words[1], first.words[1] + 11);
  const auto& payload = catalog.records[4].value.register_block_write32;
  EXPECT_EQ(payload.address, 0x100000u + catalog.offset);
  ASSERT_EQ(payload.word_count, 22u);
  for (size_t i = 0; i < 2; ++i) {
    const auto* words = payload.words + i * 11;
    const uint32_t address = 0x20000u + (i == 0 ? 64 : 128);
    EXPECT_EQ(words[0] & 31u, 0u);
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
      .value = {.register_write32 = {0x232000, 2}}};
  const loom_aie2p_program_record_t invoke = {
      .type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32,
      .value = {.register_write32 = {0x232000, 1}}};
  loom_aie2p_array_program_t program = {.array_records = &initial,
                                        .array_record_count = 1,
                                        .control_records = &invoke,
                                        .control_record_count = 1};
  IREE_ASSERT_OK(loom_aie2p_program_catalog_install(&transport, &catalog, 1,
                                                    &arena, &program));
  EXPECT_EQ(program.array_record_count, 7u);
  EXPECT_EQ(program.array_records[0].value.register_write32.value, 2u);
  EXPECT_EQ(program.control_records, &invoke);
  EXPECT_EQ(program.control_record_count, 1u);

  const uint64_t full =
      loom_xdna_array_tile_facts(family, binding.memory)->memory.local_capacity;
  occupancy[1].memory_length = full;
  IREE_ASSERT_OK(loom_aie2p_program_catalog_build(
      &transport, &binding, &layout, &linked, &arena, &catalog, &requirement));
  EXPECT_FALSE(iree_string_view_is_empty(requirement));
  EXPECT_EQ(catalog.record_count, 0u);
  EXPECT_EQ(occupancy[1].memory_length, full);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

}  // namespace
