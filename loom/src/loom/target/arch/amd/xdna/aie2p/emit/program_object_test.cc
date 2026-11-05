// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program_object.h"

#include <array>
#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/relocation.h"

namespace loom {
namespace {

loom_aie2p_leaf_contribution_t MakeHelper(
    iree_string_view_t name, const uint8_t* code, iree_host_size_t code_length,
    loom_native_section_contribution_t* section,
    loom_native_object_symbol_t* symbol) {
  *section = {
      .section_name = name,
      .storage = LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
      .access =
          LOOM_NATIVE_SECTION_ACCESS_READ | LOOM_NATIVE_SECTION_ACCESS_EXECUTE,
      .contribution_alignment = 16,
      .contents = iree_make_const_byte_span(code, code_length),
  };
  *symbol = {
      .name = name,
      .section_contribution_index = 0,
      .section_offset = 0,
      .size = code_length,
      .binding = LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
      .visibility = LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT,
      .kind = LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
  };
  return {
      .object =
          {
              .sections = section,
              .section_count = 1,
              .symbols = symbol,
              .symbol_count = 1,
          },
      .realization =
          {
              .target_identity = LOOM_AIE2P_LEAF_TARGET_IDENTITY,
              .abi_identity = LOOM_AIE2P_LEAF_ABI_IDENTITY,
              .entry_symbol_index = 0,
              .code = {code_length, 16},
          },
  };
}

TEST(Aie2pProgramObjectTest, LeavesLargestAlignmentGapAfterFinalHelper) {
  const std::array<uint8_t, 6> entry_code = {};
  const std::array<uint8_t, 2> helper_a_code = {};
  const std::array<uint8_t, 14> helper_b_code = {};
  const std::array<uint8_t, 8> helper_c_code = {};
  loom_native_section_contribution_t sections[4];
  loom_native_object_symbol_t symbols[5];

  loom_aie2p_leaf_contribution_t contributions[4];
  contributions[0] = MakeHelper(IREE_SV("entry"), entry_code.data(),
                                entry_code.size(), &sections[0], &symbols[0]);
  symbols[1] = {
      .name = IREE_SV("helper_a"),
      .section_contribution_index = IREE_HOST_SIZE_MAX,
      .binding = LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
      .visibility = LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT,
      .kind = LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
  };
  const loom_native_object_fixup_t entry_fixup = {
      .section_contribution_index = 0,
      .section_offset = 0,
      .relocation_kind = LOOM_AIE2P_NATIVE_RELOCATION_KIND_CORE_CALL_ABSOLUTE,
      .target_symbol_index = 1,
  };
  contributions[0].object.symbols = symbols;
  contributions[0].object.symbol_count = 2;
  contributions[0].object.fixups = &entry_fixup;
  contributions[0].object.fixup_count = 1;
  contributions[1] =
      MakeHelper(IREE_SV("helper_a"), helper_a_code.data(),
                 helper_a_code.size(), &sections[1], &symbols[2]);
  contributions[2] =
      MakeHelper(IREE_SV("helper_b"), helper_b_code.data(),
                 helper_b_code.size(), &sections[2], &symbols[3]);
  contributions[3] =
      MakeHelper(IREE_SV("helper_c"), helper_c_code.data(),
                 helper_c_code.size(), &sections[3], &symbols[4]);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_aie2p_program_object_t program = {};
  IREE_ASSERT_OK(loom_aie2p_program_object_merge(
      contributions, IREE_ARRAYSIZE(contributions), &arena, &arena, &program));

  ASSERT_EQ(program.contribution.object.section_count, 4u);
  EXPECT_EQ(program.contribution.object.sections[0].contents.data,
            entry_code.data());
  EXPECT_EQ(program.contribution.object.sections[1].contents.data,
            helper_b_code.data());
  EXPECT_EQ(program.contribution.object.sections[2].contents.data,
            helper_c_code.data());
  EXPECT_EQ(program.contribution.object.sections[3].contents.data,
            helper_a_code.data());
  EXPECT_EQ(program.contribution.realization.code.byte_length, 50u);
  EXPECT_EQ(program.contribution.realization.code.minimum_alignment, 16u);

  ASSERT_EQ(program.contribution.object.fixup_count, 1u);
  const loom_native_object_fixup_t& resolved_call =
      program.contribution.object.fixups[0];
  ASSERT_LT(resolved_call.target_symbol_index,
            program.contribution.object.symbol_count);
  const loom_native_object_symbol_t& helper_a =
      program.contribution.object.symbols[resolved_call.target_symbol_index];
  EXPECT_TRUE(iree_string_view_equal(helper_a.name, IREE_SV("helper_a")));
  EXPECT_EQ(helper_a.section_contribution_index, 3u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(Aie2pProgramObjectTest, SharesBodiesAndRebindsCallsWithSelfRelocations) {
  const std::array<uint8_t, 6> entry_code = {};
  const std::array<uint8_t, 2> helper_code = {};
  const std::array<uint8_t, 14> other_code = {};
  loom_native_section_contribution_t sections[4];
  loom_native_object_symbol_t symbols[6];
  loom_aie2p_leaf_contribution_t contributions[4];
  contributions[0] = MakeHelper(IREE_SV("entry"), entry_code.data(),
                                entry_code.size(), &sections[0], &symbols[0]);
  for (int i = 0; i < 2; ++i) {
    symbols[i + 1] = {
        .name = i == 0 ? IREE_SV("first") : IREE_SV("second"),
        .section_contribution_index = IREE_HOST_SIZE_MAX,
        .binding = LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
        .kind = LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
    };
  }
  const loom_native_object_fixup_t entry_fixups[] = {
      {
          .section_contribution_index = 0,
          .section_offset = 0,
          .relocation_kind =
              LOOM_AIE2P_NATIVE_RELOCATION_KIND_CORE_CALL_ABSOLUTE,
          .target_symbol_index = 1,
      },
      {
          .section_contribution_index = 0,
          .section_offset = 2,
          .relocation_kind =
              LOOM_AIE2P_NATIVE_RELOCATION_KIND_CORE_CALL_ABSOLUTE,
          .target_symbol_index = 2,
      },
  };
  contributions[0].object.symbol_count = 3;
  contributions[0].object.fixups = entry_fixups;
  contributions[0].object.fixup_count = IREE_ARRAYSIZE(entry_fixups);
  contributions[1] = MakeHelper(IREE_SV("first"), helper_code.data(),
                                helper_code.size(), &sections[1], &symbols[3]);
  contributions[2] = MakeHelper(IREE_SV("other"), other_code.data(),
                                other_code.size(), &sections[2], &symbols[4]);
  contributions[3] = MakeHelper(IREE_SV("second"), helper_code.data(),
                                helper_code.size(), &sections[3], &symbols[5]);
  const loom_native_object_fixup_t helper_fixup = {
      .section_contribution_index = 0,
      .section_offset = 0,
      .relocation_kind = LOOM_AIE2P_NATIVE_RELOCATION_KIND_CORE_BRANCH_ABSOLUTE,
      .target_symbol_index = 0,
      .addend = 0,
  };
  for (int i : {1, 3}) {
    contributions[i].object.fixups = &helper_fixup;
    contributions[i].object.fixup_count = 1;
  }

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_aie2p_program_object_t program = {};
  IREE_ASSERT_OK(loom_aie2p_program_object_merge(
      contributions, IREE_ARRAYSIZE(contributions), &arena, &arena, &program));

  EXPECT_EQ(program.canonical_indices[0], 0u);
  EXPECT_EQ(program.canonical_indices[1], 1u);
  EXPECT_EQ(program.canonical_indices[2], 2u);
  EXPECT_EQ(program.canonical_indices[3], 1u);
  ASSERT_EQ(program.contribution.object.section_count, 3u);
  EXPECT_EQ(program.contribution.object.sections[1].contents.data,
            other_code.data());
  EXPECT_EQ(program.contribution.object.sections[2].contents.data,
            helper_code.data());
  EXPECT_EQ(program.contribution.realization.code.byte_length, 34u);
  ASSERT_EQ(program.contribution.object.symbol_count, 6u);
  // The canonical body is placed last, after the second symbol's input slot.
  // Both names resolve to that body; only its own relocation is emitted once.
  EXPECT_EQ(program.contribution.object.symbols[3].section_contribution_index,
            2u);
  EXPECT_EQ(program.contribution.object.symbols[5].section_contribution_index,
            2u);
  ASSERT_EQ(program.contribution.object.fixup_count, 3u);
  EXPECT_EQ(program.contribution.object.fixups[0].target_symbol_index, 3u);
  EXPECT_EQ(program.contribution.object.fixups[1].target_symbol_index, 5u);
  EXPECT_EQ(program.contribution.object.fixups[2].section_contribution_index,
            2u);
  EXPECT_EQ(program.contribution.object.fixups[2].target_symbol_index, 3u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(Aie2pProgramObjectTest, PreservesFirstIdentityAcrossInterleavedBodies) {
  constexpr size_t kBodyCount = 31;
  constexpr size_t kHelperCount = kBodyCount * 4;
  std::array<std::array<uint8_t, 16>, kBodyCount> code = {};
  for (size_t i = 0; i < code.size(); ++i) {
    code[i][0] = (uint8_t)i;
  }
  std::array<loom_native_section_contribution_t, kHelperCount + 1> sections;
  std::array<loom_native_object_symbol_t, kHelperCount + 1> symbols;
  std::array<loom_aie2p_leaf_contribution_t, kHelperCount + 1> contributions;
  std::array<std::string, kHelperCount + 1> names;
  for (size_t i = 0; i < contributions.size(); ++i) {
    const size_t body_index = i == 0 ? 0 : ((i - 1) * 17) % kBodyCount;
    names[i] = "body_" + std::to_string(i);
    contributions[i] =
        MakeHelper(iree_make_string_view(names[i].data(), names[i].size()),
                   code[body_index].data(), code[body_index].size(),
                   &sections[i], &symbols[i]);
  }
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_aie2p_program_object_t program = {};
  IREE_ASSERT_OK(loom_aie2p_program_object_merge(
      contributions.data(), contributions.size(), &arena, &arena, &program));
  ASSERT_EQ(program.contribution.object.section_count, kBodyCount + 1);
  EXPECT_EQ(program.contribution.realization.code.byte_length,
            (kBodyCount + 1) * 16);
  EXPECT_EQ(program.canonical_indices[0], 0u);
  for (size_t i = 1; i < contributions.size(); ++i) {
    const size_t first = (i - 1) % kBodyCount + 1;
    EXPECT_EQ(program.canonical_indices[i], first);
    EXPECT_EQ(program.contribution.object.symbols[i].section_contribution_index,
              first);
  }
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(Aie2pProgramObjectTest, PreservesDifferentInstructionsAndRelocations) {
  const std::array<uint8_t, 8> code = {};
  loom_native_section_contribution_t sections[3];
  loom_native_object_symbol_t symbols[3];
  loom_aie2p_leaf_contribution_t contributions[3];
  contributions[0] = MakeHelper(IREE_SV("entry"), code.data(), code.size(),
                                &sections[0], &symbols[0]);
  contributions[1] = MakeHelper(IREE_SV("first"), code.data(), code.size(),
                                &sections[1], &symbols[1]);
  contributions[2] = MakeHelper(IREE_SV("second"), code.data(), code.size(),
                                &sections[2], &symbols[2]);
  loom_native_object_fixup_t fixups[] = {
      {
          .section_contribution_index = 0,
          .section_offset = 0,
          .relocation_kind =
              LOOM_AIE2P_NATIVE_RELOCATION_KIND_CORE_BRANCH_ABSOLUTE,
          .target_symbol_index = 0,
          .addend = 2,
      },
      {
          .section_contribution_index = 0,
          .section_offset = 0,
          .relocation_kind =
              LOOM_AIE2P_NATIVE_RELOCATION_KIND_CORE_BRANCH_ABSOLUTE,
          .target_symbol_index = 0,
          .addend = 4,
      },
  };
  contributions[1].object.fixups = &fixups[0];
  contributions[1].object.fixup_count = 1;
  contributions[2].object.fixups = &fixups[1];
  contributions[2].object.fixup_count = 1;

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_aie2p_program_object_t program = {};
  IREE_ASSERT_OK(loom_aie2p_program_object_merge(
      contributions, IREE_ARRAYSIZE(contributions), &arena, &arena, &program));
  EXPECT_EQ(program.contribution.object.section_count, 3u);
  EXPECT_EQ(program.contribution.realization.code.byte_length, 40u);
  EXPECT_EQ(program.contribution.object.symbols[1].section_contribution_index,
            1u);
  EXPECT_EQ(program.contribution.object.symbols[2].section_contribution_index,
            2u);
  ASSERT_EQ(program.contribution.object.fixup_count, 2u);
  EXPECT_EQ(program.contribution.object.fixups[0].addend, 2);
  EXPECT_EQ(program.contribution.object.fixups[1].addend, 4);
  EXPECT_EQ(program.contribution.object.fixups[0].target_symbol_index, 1u);
  EXPECT_EQ(program.contribution.object.fixups[1].target_symbol_index, 2u);

  // Equal entry/helper code never folds the externally entered program body.
  contributions[1].object.fixup_count = 0;
  contributions[2].object.fixup_count = 0;
  IREE_ASSERT_OK(loom_aie2p_program_object_merge(
      contributions, IREE_ARRAYSIZE(contributions), &arena, &arena, &program));
  EXPECT_EQ(program.contribution.object.section_count, 2u);
  EXPECT_EQ(program.contribution.realization.code.byte_length, 24u);
  EXPECT_EQ(program.contribution.object.symbols[0].section_contribution_index,
            0u);
  EXPECT_EQ(program.contribution.object.symbols[1].section_contribution_index,
            1u);
  EXPECT_EQ(program.contribution.object.symbols[2].section_contribution_index,
            1u);

  const std::array<uint8_t, 8> different_code = {1};
  sections[2].contents =
      iree_make_const_byte_span(different_code.data(), different_code.size());
  IREE_ASSERT_OK(loom_aie2p_program_object_merge(
      contributions, IREE_ARRAYSIZE(contributions), &arena, &arena, &program));
  EXPECT_EQ(program.contribution.object.section_count, 3u);
  EXPECT_EQ(program.contribution.realization.code.byte_length, 40u);
  EXPECT_NE(program.contribution.object.symbols[1].section_contribution_index,
            program.contribution.object.symbols[2].section_contribution_index);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
