// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/context.h"

#include <cstdint>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_catalog.h"

namespace loom {
namespace {

class LowLowerContextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    context_.function_arena = &function_arena_;
    context_.lowering = &frame_;
    iree_arena_initialize(&block_pool_, &function_arena_);
    iree_arena_initialize(&block_pool_, &context_.analysis_arena);
    iree_arena_initialize(&block_pool_, &context_.emission_arena);
  }

  void TearDown() override {
    iree_arena_deinitialize(&context_.emission_arena);
    iree_arena_deinitialize(&context_.analysis_arena);
    iree_arena_deinitialize(&function_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  // Stable storage for retained function decisions.
  iree_arena_allocator_t function_arena_;
  // Decisions and value bindings shared by the active context.
  loom_low_lowering_frame_t frame_ = {};
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
  EXPECT_EQ(context_.lowering->target_state_record_count,
            IREE_ARRAYSIZE(kKeys));

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kKeys); ++i) {
    void* storage = nullptr;
    IREE_ASSERT_OK(loom_low_lower_get_or_allocate_target_state(
        &context_, &kKeys[i], sizeof(TargetState), &storage));
    EXPECT_EQ(storage, states[i]);
    EXPECT_EQ(states[i]->sequence, i + 1);
    EXPECT_EQ(states[i]->payload, (i + 1) * 17);
  }
  EXPECT_EQ(context_.lowering->target_state_record_count,
            IREE_ARRAYSIZE(kKeys));
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

  // Function-wide block bindings precede scratch and survive every callback.
  uint32_t* bindings = nullptr;
  IREE_ASSERT_OK(
      iree_arena_allocate_array(&context_.emission_arena, 2, sizeof(*bindings),
                                reinterpret_cast<void**>(&bindings)));
  bindings[0] = 17;
  bindings[1] = 29;
  const auto retained_used = context_.emission_arena.used_allocation_size;
  const auto retained_reserved = context_.emission_arena.total_allocation_size;
  for (int i = 0; i < 2; ++i) {
    loom_low_lower_emission_scope_begin(&context_);
    IREE_ASSERT_OK(loom_low_lower_allocate_emission_array(
        &context_, /*count=*/4096, /*element_size=*/sizeof(uint32_t),
        &storage));
    loom_low_lower_emission_scope_end(&context_);
    EXPECT_EQ(bindings[0], 17u);
    EXPECT_EQ(bindings[1], 29u);
    EXPECT_EQ(context_.emission_arena.used_allocation_size, retained_used);
    EXPECT_EQ(context_.emission_arena.total_allocation_size, retained_reserved);
  }
}

TEST_F(LowLowerContextTest, DiagnosticsRetainContextAndRejectWithoutCallback) {
  loom_low_lower_options_t options = {.max_errors = 1};
  loom_low_lower_result_t result = {};
  context_.options = &options;
  context_.result = &result;
  loom_diagnostic_emission_t emission = {.error = LOOM_ERR_TARGET_051};
  const loom_diagnostic_emission_t* received = nullptr;
  context_.diagnostic_emitter = {
      [](void* user_data, const loom_diagnostic_emission_t* value) {
        *static_cast<const loom_diagnostic_emission_t**>(user_data) = value;
        return iree_ok_status();
      },
      &received,
  };
  IREE_ASSERT_OK(loom_low_lower_emit_diagnostic(&context_, &emission));
  // Forward the original request, including indexed origins and related ranges.
  EXPECT_EQ(received, &emission);
  EXPECT_EQ(result.error_count, 1u);
  received = nullptr;
  IREE_ASSERT_OK(loom_low_lower_emit_diagnostic(&context_, &emission));
  EXPECT_EQ(received, nullptr);
  EXPECT_EQ(result.error_count, 1u);

  context_.diagnostic_emitter = {};
  result.error_count = 0;
  IREE_ASSERT_OK(loom_low_lower_emit_diagnostic(&context_, &emission));
  EXPECT_EQ(result.error_count, 1u);
}

}  // namespace
}  // namespace loom
