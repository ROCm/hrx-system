// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/verify/verify_state.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

// Tests the scoped visibility API with value IDs, not constructed IR. Authored
// signature, body and CFG behavior is exercised by isolation.loom-test.
class VerifyStateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &state_.arena);
    module_.values.count = IREE_ARRAYSIZE(definition_depths_);
    state_.module = &module_;
    state_.visibility.definition_depths = definition_depths_;
    state_.visibility.minimum_depth = 1;
    state_.visibility.scope_minimum_depth_storage =
        state_.visibility.scope_minimum_depths;
    state_.scope_watermark_storage = state_.scope_watermarks;
  }
  void TearDown() override {
    iree_arena_deinitialize(&state_.arena);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Scratch backing the production definition stack.
  iree_arena_block_pool_t pool_ = {};
  // Zero-initialized definition tags for every supported scope depth.
  uint8_t definition_depths_[LOOM_VERIFY_MAX_SCOPE_DEPTH + 1] = {};
  // Minimal module header supplying the valid value-ID range.
  loom_module_t module_ = {};
  // Verifier state exercised directly without a constructed operation tree.
  loom_verify_state_t state_ = {};
};

TEST_F(VerifyStateTest, ReusableSortedValuesSupportMembershipAndDuplicates) {
  const loom_value_id_t first[] = {5, 2, 5, LOOM_VALUE_ID_INVALID,
                                   LOOM_VERIFY_MAX_SCOPE_DEPTH + 1};
  IREE_ASSERT_OK(
      loom_verify_sorted_values_assign(&state_, first, IREE_ARRAYSIZE(first)));
  EXPECT_EQ(state_.sorted_values.count, 3u);
  EXPECT_TRUE(loom_verify_sorted_values_contains(&state_, 2));
  EXPECT_TRUE(loom_verify_sorted_values_contains(&state_, 5));
  EXPECT_FALSE(loom_verify_sorted_values_contains(&state_, 3));
  EXPECT_FALSE(loom_verify_sorted_values_contains(
      &state_, LOOM_VERIFY_MAX_SCOPE_DEPTH + 1));
  EXPECT_TRUE(loom_verify_sorted_values_contain_duplicate(&state_, 5));
  EXPECT_FALSE(loom_verify_sorted_values_contain_duplicate(&state_, 2));

  const loom_value_id_t second[] = {7};
  IREE_ASSERT_OK(loom_verify_sorted_values_assign(&state_, second,
                                                  IREE_ARRAYSIZE(second)));
  EXPECT_EQ(state_.sorted_values.count, 1u);
  EXPECT_TRUE(loom_verify_sorted_values_contains(&state_, 7));
  EXPECT_FALSE(loom_verify_sorted_values_contains(&state_, 5));
}

TEST_F(VerifyStateTest, RootDefinitionsAndUndefinedValues) {
  EXPECT_FALSE(loom_verify_value_is_visible(&state_, 0));
  IREE_ASSERT_OK(loom_verify_define_value(&state_, 0));
  EXPECT_TRUE(loom_verify_value_is_visible(&state_, 0));
  EXPECT_FALSE(loom_verify_value_is_visible(&state_, 1));
  loom_verify_restore_definitions(&state_, 0);
  EXPECT_FALSE(loom_verify_value_is_visible(&state_, 0));
}

TEST_F(VerifyStateTest, IsolatedBodyHidesParentResultsAndRestoresOuterScope) {
  IREE_ASSERT_OK(loom_verify_define_value(&state_, 0));
  IREE_ASSERT_OK(loom_verify_push_scope(&state_, /*isolated=*/true));
  EXPECT_FALSE(loom_verify_value_is_visible(&state_, 0));
  IREE_ASSERT_OK(loom_verify_define_value(&state_, 1));
  IREE_ASSERT_OK(loom_verify_push_scope(&state_, /*isolated=*/false));
  EXPECT_FALSE(loom_verify_value_is_visible(&state_, 0));
  EXPECT_TRUE(loom_verify_value_is_visible(&state_, 1));
  loom_verify_pop_scope(&state_);
  loom_verify_pop_scope(&state_);
  EXPECT_TRUE(loom_verify_value_is_visible(&state_, 0));
  EXPECT_FALSE(loom_verify_value_is_visible(&state_, 1));
  IREE_ASSERT_OK(loom_verify_push_scope(&state_, /*isolated=*/true));
  EXPECT_FALSE(loom_verify_value_is_visible(&state_, 0));
  EXPECT_FALSE(loom_verify_value_is_visible(&state_, 1));
  loom_verify_pop_scope(&state_);
}

TEST_F(VerifyStateTest, DefinitionWatermarkPreservesIsolationFloor) {
  IREE_ASSERT_OK(loom_verify_define_value(&state_, 0));
  IREE_ASSERT_OK(loom_verify_push_scope(&state_, /*isolated=*/true));
  IREE_ASSERT_OK(loom_verify_define_value(&state_, 1));
  const iree_host_size_t watermark = state_.defined_stack_count;
  IREE_ASSERT_OK(loom_verify_define_value(&state_, 2));
  loom_verify_restore_definitions(&state_, watermark);
  EXPECT_FALSE(loom_verify_value_is_visible(&state_, 0));
  EXPECT_TRUE(loom_verify_value_is_visible(&state_, 1));
  EXPECT_FALSE(loom_verify_value_is_visible(&state_, 2));
  loom_verify_pop_scope(&state_);
  EXPECT_TRUE(loom_verify_value_is_visible(&state_, 0));
  EXPECT_FALSE(loom_verify_value_is_visible(&state_, 1));
}

TEST_F(VerifyStateTest, EveryIsolationDepthAndLimit) {
  IREE_ASSERT_OK(loom_verify_define_value(&state_, 0));
  for (uint32_t depth = 1; depth <= LOOM_VERIFY_MAX_SCOPE_DEPTH; ++depth) {
    IREE_ASSERT_OK(loom_verify_push_scope(&state_, /*isolated=*/true));
    if (depth <= LOOM_VERIFY_INLINE_SCOPE_DEPTH) {
      EXPECT_EQ(state_.scope_watermark_storage, state_.scope_watermarks);
      EXPECT_EQ(state_.visibility.scope_minimum_depth_storage,
                state_.visibility.scope_minimum_depths);
    } else {
      EXPECT_NE(state_.scope_watermark_storage, state_.scope_watermarks);
      EXPECT_NE(state_.visibility.scope_minimum_depth_storage,
                state_.visibility.scope_minimum_depths);
    }
    EXPECT_FALSE(loom_verify_value_is_visible(&state_, depth - 1));
    IREE_ASSERT_OK(loom_verify_define_value(&state_, depth));
    EXPECT_TRUE(loom_verify_value_is_visible(&state_, depth));
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_verify_push_scope(&state_, /*isolated=*/true));
  for (uint32_t depth = LOOM_VERIFY_MAX_SCOPE_DEPTH; depth > 0; --depth) {
    loom_verify_pop_scope(&state_);
    EXPECT_FALSE(loom_verify_value_is_visible(&state_, depth));
    EXPECT_TRUE(loom_verify_value_is_visible(&state_, depth - 1));
  }
}

}  // namespace
}  // namespace loom
