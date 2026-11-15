// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/cfg_condition_projection_cache.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class CfgConditionProjectionCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32768, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    cache_.arena = &arena_;
    IREE_ASSERT_OK(loom_condition_relation_set_builder_allocate(256, &arena_,
                                                                &set_builder_));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_condition_relation_set_id_t Singleton(uint32_t value) const {
    return loom_condition_relation_set_builder_singleton(set_builder_, value);
  }

  // Shared backing storage for the set domain and projection cache.
  iree_arena_block_pool_t block_pool_;
  // Owns all cache slots until the fixture ends.
  iree_arena_allocator_t arena_;
  // Finite source and target domain used by the mapping fixtures.
  loom_condition_relation_set_builder_t* set_builder_ = nullptr;
  // Initially empty cache with lazy slot allocation.
  loom_cfg_condition_projection_cache_t cache_ = {};
};

TEST_F(CfgConditionProjectionCacheTest, EmptyImageIsDistinctFromMiss) {
  const auto source = Singleton(7);
  const auto initial_bytes = arena_.used_allocation_size;
  loom_condition_relation_set_id_t target = 0;
  EXPECT_FALSE(
      loom_cfg_condition_projection_cache_lookup(&cache_, 0, source, &target));
  EXPECT_EQ(arena_.used_allocation_size, initial_bytes);

  IREE_ASSERT_OK(loom_cfg_condition_projection_cache_insert(
      &cache_, 0, source, LOOM_CONDITION_RELATION_SET_EMPTY));
  target = Singleton(3);
  EXPECT_TRUE(
      loom_cfg_condition_projection_cache_lookup(&cache_, 0, source, &target));
  EXPECT_EQ(target, LOOM_CONDITION_RELATION_SET_EMPTY);
  EXPECT_FALSE(
      loom_cfg_condition_projection_cache_lookup(&cache_, 1, source, &target));
  EXPECT_FALSE(loom_cfg_condition_projection_cache_lookup(
      &cache_, 0, Singleton(8), &target));
}

TEST_F(CfgConditionProjectionCacheTest, DistinctMappingsSurviveGrowth) {
  for (uint32_t edge = 0; edge < 8; ++edge) {
    for (uint32_t value = 0; value < 128; ++value) {
      loom_condition_relation_set_id_t target = 0;
      const auto source = Singleton(value);
      EXPECT_FALSE(loom_cfg_condition_projection_cache_lookup(&cache_, edge,
                                                              source, &target));
      IREE_ASSERT_OK(loom_cfg_condition_projection_cache_insert(
          &cache_, edge, source, Singleton((value + edge) % 256)));
    }
  }
  for (uint32_t value = 0; value < 128; ++value) {
    for (uint32_t edge = 0; edge < 8; ++edge) {
      loom_condition_relation_set_id_t target = 0;
      EXPECT_TRUE(loom_cfg_condition_projection_cache_lookup(
          &cache_, edge, Singleton(value), &target));
      EXPECT_EQ(target, Singleton((value + edge) % 256));
    }
  }
  loom_condition_relation_set_id_t target = 0;
  EXPECT_FALSE(loom_cfg_condition_projection_cache_lookup(
      &cache_, 8, Singleton(0), &target));
  EXPECT_FALSE(loom_cfg_condition_projection_cache_lookup(
      &cache_, 0, Singleton(255), &target));
}

}  // namespace
}  // namespace loom
