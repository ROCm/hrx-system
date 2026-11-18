// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_actions.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/planning/wait_plan.h"

namespace loom {
namespace {

class AmdgpuWaitActionsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &pool_);
    iree_arena_initialize(&pool_, &arena_);
    iree_arena_initialize(&pool_, &transient_arena_);
    loom_amdgpu_wait_actions_initialize(&actions_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&transient_arena_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Shared backing pool for transient and retained action storage.
  iree_arena_block_pool_t pool_;
  // Arena owning the finalized action sequence.
  iree_arena_allocator_t arena_;
  // Arena owning sparse append segments.
  iree_arena_allocator_t transient_arena_;
  // Production action storage exercised by each test.
  loom_amdgpu_wait_actions_t actions_ = {};
};

TEST_F(AmdgpuWaitActionsTest, EmptyStreamPublishesEmptySequence) {
  IREE_ASSERT_OK(loom_amdgpu_wait_actions_finalize(&actions_, &arena_));
  EXPECT_EQ(actions_.actions, nullptr);
  EXPECT_EQ(actions_.action_count, 0u);
  EXPECT_EQ(actions_.hazard_event_count, 0u);
}

TEST_F(AmdgpuWaitActionsTest, FinalSequenceSurvivesTransientSegments) {
  constexpr uint32_t kActionCount = 400;
  iree_host_size_t expected_hazard_count = 0;
  for (uint32_t i = 0; i < kActionCount; ++i) {
    const bool is_planned = i % 3 == 0;
    const loom_amdgpu_wait_plan_action_t action = {
        /*kind=*/is_planned ? LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED
                            : LOOM_AMDGPU_WAIT_PLAN_ACTION_EXPLICIT,
        /*reason=*/LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE,
        /*counter_id=*/LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD,
        /*target_count=*/static_cast<uint16_t>(i),
        /*block_index=*/i / 100,
        /*node_index=*/i,
        /*scheduled_ordinal=*/i % 100,
        /*producer_node=*/i + 1,
        /*consumer_node=*/i + 2,
        /*outstanding_before=*/i + 3,
    };
    if (is_planned) {
      ++expected_hazard_count;
    }
    IREE_ASSERT_OK(
        loom_amdgpu_wait_actions_append(&actions_, &action, &transient_arena_));
  }

  IREE_ASSERT_OK(loom_amdgpu_wait_actions_finalize(&actions_, &arena_));
  iree_arena_reset(&transient_arena_);

  ASSERT_EQ(actions_.action_count, kActionCount);
  EXPECT_EQ(actions_.hazard_event_count, expected_hazard_count);
  for (uint32_t i = 0; i < kActionCount; ++i) {
    SCOPED_TRACE(i);
    const loom_amdgpu_wait_plan_action_t& action = actions_.actions[i];
    EXPECT_EQ(action.target_count, i);
    EXPECT_EQ(action.block_index, i / 100);
    EXPECT_EQ(action.node_index, i);
    EXPECT_EQ(action.scheduled_ordinal, i % 100);
    EXPECT_EQ(action.producer_node, i + 1);
    EXPECT_EQ(action.consumer_node, i + 2);
    EXPECT_EQ(action.outstanding_before, i + 3);
  }
}

}  // namespace
}  // namespace loom
