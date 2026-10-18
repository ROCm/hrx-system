// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program_object.h"

#include <array>

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
  loom_aie2p_leaf_contribution_t program = {};
  IREE_ASSERT_OK(loom_aie2p_program_object_merge(
      contributions, IREE_ARRAYSIZE(contributions), &arena, &program));

  ASSERT_EQ(program.object.section_count, 4u);
  EXPECT_EQ(program.object.sections[0].contents.data, entry_code.data());
  EXPECT_EQ(program.object.sections[1].contents.data, helper_b_code.data());
  EXPECT_EQ(program.object.sections[2].contents.data, helper_c_code.data());
  EXPECT_EQ(program.object.sections[3].contents.data, helper_a_code.data());
  EXPECT_EQ(program.realization.code.byte_length, 50u);
  EXPECT_EQ(program.realization.code.minimum_alignment, 16u);

  ASSERT_EQ(program.object.fixup_count, 1u);
  const loom_native_object_fixup_t& resolved_call = program.object.fixups[0];
  ASSERT_LT(resolved_call.target_symbol_index, program.object.symbol_count);
  const loom_native_object_symbol_t& helper_a =
      program.object.symbols[resolved_call.target_symbol_index];
  EXPECT_TRUE(iree_string_view_equal(helper_a.name, IREE_SV("helper_a")));
  EXPECT_EQ(helper_a.section_contribution_index, 3u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
