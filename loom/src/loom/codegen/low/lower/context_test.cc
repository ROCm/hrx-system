// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/context.h"

#include <cstdint>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class LowLowerContextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &context_.function_arena);
    iree_arena_initialize(&block_pool_, &context_.emission_arena);
  }

  void TearDown() override {
    iree_arena_deinitialize(&context_.emission_arena);
    iree_arena_deinitialize(&context_.function_arena);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  loom_low_lower_context_t context_ = {};
};

TEST_F(LowLowerContextTest, FunctionTargetStateIsStableAndZeroInitialized) {
  struct TargetState {
    uint32_t sequence;
    uint32_t payload;
  };
  static const uint8_t kKeys[9] = {};
  TargetState* states[IREE_ARRAYSIZE(kKeys)] = {};

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kKeys); ++i) {
    void* storage = nullptr;
    IREE_ASSERT_OK(loom_low_lower_get_or_allocate_target_state(
        &context_, &kKeys[i], sizeof(TargetState), &storage));
    ASSERT_NE(storage, nullptr);
    states[i] = static_cast<TargetState*>(storage);
    EXPECT_EQ(states[i]->sequence, 0u);
    EXPECT_EQ(states[i]->payload, 0u);
    states[i]->sequence = static_cast<uint32_t>(i + 1);
    states[i]->payload = static_cast<uint32_t>((i + 1) * 17);
  }
  EXPECT_EQ(context_.lowering.target_state_record_count, IREE_ARRAYSIZE(kKeys));

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kKeys); ++i) {
    void* storage = nullptr;
    IREE_ASSERT_OK(loom_low_lower_get_or_allocate_target_state(
        &context_, &kKeys[i], sizeof(TargetState), &storage));
    EXPECT_EQ(storage, states[i]);
    EXPECT_EQ(states[i]->sequence, i + 1);
    EXPECT_EQ(states[i]->payload, (i + 1) * 17);
  }
  EXPECT_EQ(context_.lowering.target_state_record_count, IREE_ARRAYSIZE(kKeys));
}

TEST_F(LowLowerContextTest, EmissionScopeReleasesScratchStorage) {
  EXPECT_FALSE(context_.emission_arena_active);
  EXPECT_EQ(context_.emission_arena.used_allocation_size, 0u);

  loom_low_lower_emission_scope_begin(&context_);
  EXPECT_TRUE(context_.emission_arena_active);
  void* storage = nullptr;
  IREE_ASSERT_OK(loom_low_lower_allocate_emission_array(
      &context_, /*count=*/8, /*element_size=*/sizeof(uint32_t), &storage));
  EXPECT_NE(storage, nullptr);
  EXPECT_GT(context_.emission_arena.used_allocation_size, 0u);

  loom_low_lower_emission_scope_end(&context_);
  EXPECT_FALSE(context_.emission_arena_active);
  EXPECT_EQ(context_.emission_arena.used_allocation_size, 0u);
  EXPECT_EQ(context_.emission_arena.total_allocation_size, 0u);
  EXPECT_EQ(context_.emission_arena.block_head, nullptr);
}

}  // namespace
}  // namespace loom
