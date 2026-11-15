// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/entry_interface.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class EntryInterfaceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  iree_arena_block_pool_t pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(EntryInterfaceTest, DiamondAndRepeatedCallsRetainEachVariableOnce) {
  loom_spirv_entry_interface_t leaf = {};
  IREE_ASSERT_OK(loom_spirv_entry_interface_declare(&leaf, 1, 101, &arena_));

  loom_spirv_entry_interface_t left = {};
  IREE_ASSERT_OK(loom_spirv_entry_interface_include(&left, 2, &leaf, &arena_));
  IREE_ASSERT_OK(loom_spirv_entry_interface_declare(&left, 2, 102, &arena_));

  loom_spirv_entry_interface_t right = {};
  IREE_ASSERT_OK(loom_spirv_entry_interface_declare(&right, 3, 103, &arena_));
  IREE_ASSERT_OK(loom_spirv_entry_interface_include(&right, 3, &leaf, &arena_));

  loom_spirv_entry_interface_t entry = {};
  IREE_ASSERT_OK(loom_spirv_entry_interface_include(&entry, 4, &left, &arena_));
  IREE_ASSERT_OK(
      loom_spirv_entry_interface_include(&entry, 4, &right, &arena_));
  IREE_ASSERT_OK(loom_spirv_entry_interface_include(&entry, 4, &left, &arena_));
  ASSERT_EQ(entry.count, 3u);
  EXPECT_EQ(entry.variables[0]->id, 101u);
  EXPECT_EQ(entry.variables[1]->id, 102u);
  EXPECT_EQ(entry.variables[2]->id, 103u);

  // A separate entry shares only the globals in its own reachable closure.
  loom_spirv_entry_interface_t other_entry = {};
  IREE_ASSERT_OK(
      loom_spirv_entry_interface_include(&other_entry, 5, &right, &arena_));
  ASSERT_EQ(other_entry.count, 2u);
  EXPECT_EQ(other_entry.variables[0]->id, 103u);
  EXPECT_EQ(other_entry.variables[1]->id, 101u);
  EXPECT_EQ(entry.count, 3u);
  EXPECT_EQ(leaf.count, 1u);
}

TEST_F(EntryInterfaceTest, GrowthPreservesVariableIdentityAndUseOrder) {
  loom_spirv_entry_interface_t leaf = {};
  for (uint32_t i = 0; i < 100; ++i) {
    IREE_ASSERT_OK(
        loom_spirv_entry_interface_declare(&leaf, 1, 1000 + i, &arena_));
  }
  loom_spirv_entry_interface_t entry = {};
  IREE_ASSERT_OK(loom_spirv_entry_interface_include(&entry, 2, &leaf, &arena_));
  IREE_ASSERT_OK(loom_spirv_entry_interface_include(&entry, 2, &leaf, &arena_));
  ASSERT_EQ(entry.count, 100u);
  for (uint32_t i = 0; i < 100; ++i) {
    EXPECT_EQ(entry.variables[i], leaf.variables[i]);
    EXPECT_EQ(entry.variables[i]->id, 1000 + i);
  }
}

}  // namespace
