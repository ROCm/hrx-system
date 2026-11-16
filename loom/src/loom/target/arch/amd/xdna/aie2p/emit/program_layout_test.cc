// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program_layout.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/tile_link.h"

namespace {

loom_pipeline_lifetimes_t MakeLifetimes() {
  constexpr auto none = IREE_HOST_SIZE_MAX;
  static const loom_pipeline_lifetime_t nodes[] = {
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 0, none, 1, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 0, 0, none, 2},
      {LOOM_PIPELINE_LIFETIME_SERIAL, 0, 0, 3, none},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 1, 2, 5, 4},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 2, 2, 6, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 1, 3, none, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 2, 4, none, none},
  };
  static const iree_host_size_t owners[] = {1, 5, 6};
  return {
      .nodes = nodes,
      .count = IREE_ARRAYSIZE(nodes),
      .owners = owners,
      .has_serial = true,
  };
}

TEST(Aie2pProgramLayoutTest, SharesInstructionMemoryWithoutLoadingEmptyWindow) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);
  loom_pipeline_lifetimes_t lifetimes = MakeLifetimes();
  IREE_ASSERT_OK(loom_pipeline_lifetimes_index(&arena, &lifetimes));
  const iree_host_size_t lengths[] = {26, 18, 12002, 14002};
  const iree_string_view_t names[] = {IREE_SV("entry"), IREE_SV("shared"),
                                      IREE_SV("first"), IREE_SV("second")};
  const iree_host_size_t function_owners[] = {0, 0, 1, 2};
  std::vector<uint8_t> code(14002);
  loom_native_section_contribution_t sections[4];
  loom_native_object_symbol_t symbols[4];
  loom_aie2p_leaf_contribution_t functions[4];
  const loom_aie2p_leaf_contribution_t* contributions[4];
  for (iree_host_size_t i = 0; i < 4; ++i) {
    sections[i] = {
        .section_name = names[i],
        .storage = LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
        .access = LOOM_NATIVE_SECTION_ACCESS_READ |
                  LOOM_NATIVE_SECTION_ACCESS_EXECUTE,
        .contribution_alignment = 16,
        .contents = iree_make_const_byte_span(code.data(), lengths[i]),
    };
    symbols[i] = {
        .name = names[i],
        .section_contribution_index = 0,
        .size = lengths[i],
        .binding = LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
        .kind = LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
    };
    functions[i] = {
        .object =
            {
                .sections = &sections[i],
                .section_count = 1,
                .symbols = &symbols[i],
                .symbol_count = 1,
            },
        .realization = {.code = {lengths[i], 16}},
    };
    contributions[i] = &functions[i];
  }
  loom_aie2p_program_layout_t program;
  IREE_ASSERT_OK(loom_aie2p_program_layout_initialize(
      &lifetimes, function_owners, 4, &arena, &arena, &program));
  loom_aie2p_program_code_group_t native_groups[3];
  loom_aie2p_program_native_t native;
  IREE_ASSERT_OK(
      loom_aie2p_program_native_initialize(contributions, 4, &arena, &native));
  IREE_ASSERT_OK(loom_aie2p_program_object_partition(
      &native, nullptr, 0, program.function_groups, program.group_count, &arena,
      &arena, &program.program, native_groups));
  IREE_ASSERT_OK(loom_aie2p_program_layout_finish(native_groups, &program));
  ASSERT_EQ(program.group_count, 3u);
  EXPECT_EQ(program.program.contribution.realization.code.byte_length, 26054u);
  EXPECT_EQ(program.byte_length, 14068u);
  EXPECT_EQ(program.groups[0].offset, 0u);
  EXPECT_EQ(program.groups[1].offset, 64u);
  EXPECT_EQ(program.groups[2].offset, 64u);
  const loom_aie2p_tile_link_layout_t layout = {
      .program_address = 0,
      .program_byte_capacity = 16384,
      .program_layout = &program,
  };
  loom_aie2p_linked_tile_t linked;
  IREE_ASSERT_OK(loom_aie2p_tile_link(&program.program.contribution, &layout,
                                      &arena, &linked));
  ASSERT_EQ(linked.assembly.section_count, 3u);
  uint64_t initial_bytes = 0;
  for (iree_host_size_t i = 0; i < program.group_count; ++i) {
    const auto section_index =
        linked.assembly
            .contribution_layouts[program.groups[i]
                                      .native.section_contribution_index]
            .section_index;
    const auto& section = linked.assembly.sections[section_index];
    const auto& placement = linked.section_placements[section_index];
    EXPECT_EQ(section.address, program.groups[i].offset);
    EXPECT_EQ(placement.owner_offset, program.groups[i].offset);
    EXPECT_EQ(placement.memory_space, LOOM_XDNA_MEMORY_SPACE_PROGRAM);
    EXPECT_EQ(placement.load, i == 0 ? LOOM_AIE2P_SECTION_LOAD_INITIAL
                                     : LOOM_AIE2P_SECTION_LOAD_ON_DEMAND);
    if (placement.load == LOOM_AIE2P_SECTION_LOAD_INITIAL) {
      initial_bytes += section.contents.data_length;
    }
  }
  EXPECT_EQ(initial_bytes, 50u);
  EXPECT_EQ(linked.entry_address, 0u);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Aie2pProgramLayoutTest, RetainsEquivalentBodiesBeforePhysicalComposition) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);
  loom_pipeline_lifetimes_t lifetimes = MakeLifetimes();
  IREE_ASSERT_OK(loom_pipeline_lifetimes_index(&arena, &lifetimes));
  const iree_host_size_t lengths[] = {512, 4096, 4096, 4096, 6144};
  const iree_host_size_t owners[] = {0, 1, 1, 2, 2};
  const iree_string_view_t names[] = {IREE_SV("entry"), IREE_SV("shared"),
                                      IREE_SV("first"), IREE_SV("alias"),
                                      IREE_SV("second")};
  std::vector<uint8_t> code[5];
  loom_native_section_contribution_t sections[5];
  loom_native_object_symbol_t symbols[5];
  loom_aie2p_leaf_contribution_t functions[5];
  const loom_aie2p_leaf_contribution_t* contributions[5];
  for (size_t i = 0; i < 5; ++i) {
    code[i].resize(lengths[i]);
    code[i][0] = i == 3 ? 1 : i;
    sections[i] = {
        .section_name = names[i],
        .storage = LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
        .access = LOOM_NATIVE_SECTION_ACCESS_READ |
                  LOOM_NATIVE_SECTION_ACCESS_EXECUTE,
        .contribution_alignment = 16,
        .contents = iree_make_const_byte_span(code[i].data(), code[i].size()),
    };
    symbols[i] = {
        .name = names[i],
        .section_contribution_index = 0,
        .size = lengths[i],
        .binding = LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
        .kind = LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
    };
    functions[i] = {
        .object =
            {
                .sections = &sections[i],
                .section_count = 1,
                .symbols = &symbols[i],
                .symbol_count = 1,
            },
        .realization = {.code = {lengths[i], 16}},
    };
    contributions[i] = &functions[i];
  }
  loom_aie2p_program_native_t native;
  IREE_ASSERT_OK(
      loom_aie2p_program_native_initialize(contributions, 5, &arena, &native));
  loom_aie2p_program_layout_t layout;
  IREE_ASSERT_OK(loom_aie2p_program_layout_initialize(&lifetimes, owners, 5,
                                                      &arena, &arena, &layout));
  IREE_ASSERT_OK(loom_aie2p_program_layout_retain(&native, &arena, &layout));
  const iree_host_size_t expected_groups[] = {0, 0, 1, 0, 2};
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(layout.function_groups[i], expected_groups[i]);
  }
  loom_aie2p_program_code_group_t groups[3];
  IREE_ASSERT_OK(loom_aie2p_program_object_partition(
      &native, nullptr, 0, layout.function_groups, 3, &arena, &arena,
      &layout.program, groups));
  IREE_ASSERT_OK(loom_aie2p_program_layout_finish(groups, &layout));
  EXPECT_EQ(layout.program.contribution.object.section_count, 4u);
  EXPECT_EQ(layout.program.functions[3].canonical_index, 1u);
  EXPECT_EQ(layout.program.contribution.realization.code.byte_length, 14848u);
  EXPECT_EQ(layout.byte_length, 10752u);
  EXPECT_EQ(groups[0].code.byte_length, 4608u);
  EXPECT_EQ(groups[1].code.byte_length, 4096u);
  EXPECT_EQ(groups[2].code.byte_length, 6144u);
  EXPECT_EQ(layout.groups[1].offset, 4608u);
  EXPECT_EQ(layout.groups[2].offset, 4608u);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

}  // namespace
