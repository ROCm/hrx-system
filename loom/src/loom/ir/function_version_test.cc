// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/function_version.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

TEST(FunctionVersionOwnerTest, GrowsStableListViewInInsertionOrder) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_function_version_owner_t owner;
  loom_function_version_owner_initialize(&arena, &owner);
  const loom_function_version_list_t* list =
      loom_function_version_owner_list(&owner);
  ASSERT_NE(list, nullptr);
  EXPECT_EQ(list->values, nullptr);
  EXPECT_EQ(list->count, 0u);

  const loom_function_version_type_t version_type = {
      /*.name=*/IREE_SVL("test"),
  };
  loom_function_version_t versions[19] = {};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(versions); ++i) {
    versions[i].type = &version_type;
    IREE_ASSERT_OK(loom_function_version_owner_append(&owner, &versions[i]));
  }

  EXPECT_EQ(loom_function_version_owner_list(&owner), list);
  ASSERT_EQ(list->count, IREE_ARRAYSIZE(versions));
  EXPECT_GE(owner.capacity, list->count);
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    EXPECT_EQ(list->values[i], &versions[i]);
  }

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(FunctionVersionOwnerTest, ReservePreservesExistingVersions) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_function_version_owner_t owner;
  loom_function_version_owner_initialize(&arena, &owner);
  loom_function_version_t version = {};
  IREE_ASSERT_OK(loom_function_version_owner_append(&owner, &version));
  IREE_ASSERT_OK(loom_function_version_owner_reserve(&owner, 64));

  const loom_function_version_list_t* list =
      loom_function_version_owner_list(&owner);
  ASSERT_EQ(list->count, 1u);
  EXPECT_EQ(list->values[0], &version);
  EXPECT_GE(owner.capacity, 64u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(FunctionVersionOwnerTest, RemovePreservesRemainingOrder) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_function_version_owner_t owner;
  loom_function_version_owner_initialize(&arena, &owner);
  loom_function_version_t versions[3] = {};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(versions); ++i) {
    IREE_ASSERT_OK(loom_function_version_owner_append(&owner, &versions[i]));
  }

  EXPECT_TRUE(loom_function_version_owner_remove(&owner, &versions[1]));
  const loom_function_version_list_t* list =
      loom_function_version_owner_list(&owner);
  ASSERT_EQ(list->count, 2u);
  EXPECT_EQ(list->values[0], &versions[0]);
  EXPECT_EQ(list->values[1], &versions[2]);
  EXPECT_FALSE(loom_function_version_owner_remove(&owner, &versions[1]));

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
