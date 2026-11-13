// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/pipeline/entries.h"
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

TEST_F(ProgramSourceTest, RetainedPhaseOwnersFollowTheAdmittedHelperOrder) {
  constexpr auto none = IREE_HOST_SIZE_MAX;
  const loom_pipeline_lifetime_t nodes[] = {
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 0, none, 1, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 0, 0, none, 2},
      {LOOM_PIPELINE_LIFETIME_SERIAL, 0, 0, 3, none},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 1, 2, 5, 4},
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 2, 2, 6, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 1, 3, none, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 2, 4, none, none},
  };
  const iree_host_size_t owners[] = {1, 5, 6};
  loom_pipeline_entry_scope_t scope = {
      .lifetimes =
          {
              .nodes = nodes,
              .count = IREE_ARRAYSIZE(nodes),
              .owners = owners,
              .has_serial = true,
          },
      .owner_count = IREE_ARRAYSIZE(owners),
  };
  IREE_ASSERT_OK(loom_pipeline_lifetimes_index(&arena_, &scope.lifetimes));
  const loom_pipeline_entry_binding_t bindings[] = {
      {
          .scope = &scope,
          .owner = 0,
          .engine = 7,
          .role = LOOM_PIPELINE_ENTRY_ROOT,
      },
      {
          .scope = &scope,
          .owner = 1,
          .engine = 7,
          .role = LOOM_PIPELINE_ENTRY_COMPLETE,
      },
      {
          .scope = &scope,
          .owner = 2,
          .engine = 7,
          .role = LOOM_PIPELINE_ENTRY_COMPLETE,
      },
  };
  // A/B share H -> G, while a third helper belongs only to A.
  SourceInventory source({{1, 2}, {3, 5}, {3}, {4}, {}, {}});
  source.Reset(&arena_);
  for (size_t i = 0; i < IREE_ARRAYSIZE(bindings); ++i) {
    source.BindPipeline(i, &bindings[i]);
  }
  source.Admit(0);
  const auto& root = source.function(0);
  ASSERT_NE(root.entry_owners, nullptr);
  const iree_host_size_t expected[] = {0, 1, 2, 0, 0, 1};
  for (size_t i = 0; i < root.helpers.count; ++i) {
    EXPECT_EQ(root.entry_owners[i], expected[root.helpers.values[i].symbol_id]);
  }
  EXPECT_EQ(root.entry_owners[root.helpers.count], 0u);

  // The same helper graph under A alone has a different ownership projection.
  // Admitting that root must not overwrite the complete pipeline's result.
  const size_t used = arena_.used_allocation_size;
  source.Admit(0);
  EXPECT_EQ(arena_.used_allocation_size, used);
  source.Admit(1);
  const auto& phase = source.function(1);
  ASSERT_NE(phase.entry_owners, nullptr);
  for (size_t i = 0; i <= phase.helpers.count; ++i) {
    EXPECT_EQ(phase.entry_owners[i], 1u);
  }
  for (size_t i = 0; i < root.helpers.count; ++i) {
    EXPECT_EQ(root.entry_owners[i], expected[root.helpers.values[i].symbol_id]);
  }
}

TEST_F(ProgramSourceTest, CallFreePhaseBorrowsItsImmutableOwner) {
  constexpr auto none = IREE_HOST_SIZE_MAX;
  const loom_pipeline_lifetime_t nodes[] = {
      {LOOM_PIPELINE_LIFETIME_CONCURRENT, 0, none, 1, none},
      {LOOM_PIPELINE_LIFETIME_OWNER, 0, 0, none, none},
  };
  const iree_host_size_t owner = 1;
  loom_pipeline_entry_scope_t scope = {
      .lifetimes =
          {
              .nodes = nodes,
              .count = IREE_ARRAYSIZE(nodes),
              .owners = &owner,
          },
      .owner_count = 1,
  };
  IREE_ASSERT_OK(loom_pipeline_lifetimes_index(&arena_, &scope.lifetimes));
  const loom_pipeline_entry_binding_t binding = {
      .scope = &scope,
      .owner = 0,
      .role = LOOM_PIPELINE_ENTRY_COMPLETE,
  };
  SourceInventory source({{}});
  source.Reset(&arena_);
  source.BindPipeline(0, &binding);
  const size_t used = arena_.used_allocation_size;
  source.Admit(0);
  EXPECT_EQ(source.function(0).entry_owners, &binding.owner);
  EXPECT_EQ(arena_.used_allocation_size, used);
}

}  // namespace
}  // namespace loom::aie2p::testing
