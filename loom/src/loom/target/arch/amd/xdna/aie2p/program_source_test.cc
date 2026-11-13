// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "loom/target/arch/amd/xdna/aie2p/program_source_test_util.h"

namespace loom::aie2p::testing {
namespace {

class ProgramSourceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  static std::vector<loom_symbol_id_t> Helpers(const SourceInventory& source,
                                               size_t root) {
    std::vector<loom_symbol_id_t> result;
    const auto helpers = source.function(root).helpers;
    for (size_t i = 0; i < helpers.count; ++i) {
      result.push_back(helpers.values[i].symbol_id);
    }
    return result;
  }

  // Reusable blocks outlive the source products allocated during each test.
  iree_arena_block_pool_t pool_;
  // Owns the requested complete program closures.
  iree_arena_allocator_t arena_;
};

TEST_F(ProgramSourceTest, InteriorFunctionsRetainOnlyDirectDependencies) {
  SourceInventory source({{1}, {2}, {3}, {}});
  source.Reset(&arena_);
  source.Admit(0);
  EXPECT_EQ(Helpers(source, 0), (std::vector<loom_symbol_id_t>{3, 2, 1}));
  EXPECT_EQ(source.RetainedHelperCount(), 3u);
  for (size_t i = 1; i < 4; ++i) {
    EXPECT_FALSE(source.function(i).helpers_admitted);
    EXPECT_EQ(source.function(i).helpers.count, 0u);
  }

  // An interior function can later become an independently emitted root.
  source.Admit(2);
  EXPECT_EQ(Helpers(source, 2), (std::vector<loom_symbol_id_t>{3}));
  EXPECT_EQ(source.RetainedHelperCount(), 4u);
  const auto* original = source.function(0).helpers.values;
  const size_t used = arena_.used_allocation_size;
  source.Admit(0);
  EXPECT_EQ(source.function(0).helpers.values, original);
  EXPECT_EQ(arena_.used_allocation_size, used);
}

TEST_F(ProgramSourceTest, CompletedRootCanBecomeHelperInSharedDiamond) {
  SourceInventory source({{1, 2}, {3}, {3}, {}});
  source.Reset(&arena_);
  source.Admit(1);
  EXPECT_EQ(Helpers(source, 1), (std::vector<loom_symbol_id_t>{3}));
  source.Admit(0);
  EXPECT_EQ(Helpers(source, 0), (std::vector<loom_symbol_id_t>{3, 1, 2}));
  EXPECT_EQ(source.RetainedHelperCount(), 4u);
  EXPECT_FALSE(source.function(2).helpers_admitted);
  EXPECT_FALSE(source.function(3).helpers_admitted);
}

TEST_F(ProgramSourceTest, IndependentRootsOwnCompleteClosures) {
  SourceInventory source({{2}, {2}, {3}, {}});
  source.Reset(&arena_);
  source.Admit(0);
  source.Admit(1);
  EXPECT_EQ(Helpers(source, 0), (std::vector<loom_symbol_id_t>{3, 2}));
  EXPECT_EQ(Helpers(source, 1), (std::vector<loom_symbol_id_t>{3, 2}));
  EXPECT_EQ(source.RetainedHelperCount(), 4u);
}

}  // namespace
}  // namespace loom::aie2p::testing
