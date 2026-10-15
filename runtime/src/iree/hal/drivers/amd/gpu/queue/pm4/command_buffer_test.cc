// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/command_buffer.h"

#include <cstring>

#include "iree/hal/device_spec.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class CommandBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_pm4_barrier_profile_initialize({11, 0, 0}, &profile_));
    family_.spec = &spec_;
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_command_buffer_create(
        &family_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH, 0, 65536, &profile_,
        iree_allocator_system(), &recording_));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording_));
  }
  void TearDown() override { iree_hal_command_buffer_release(recording_); }

  // Pure packet recording needs family identity, not native device allocation.
  iree_hal_queue_family_spec_t spec_ = {};
  // Borrowed family remains alive through recording and invocation destruction.
  iree_hal_queue_family_t family_ = {};
  // Immutable target-selected cache encoding.
  iree_hal_amd_gpu_pm4_barrier_profile_t profile_ = {};
  // Actual command buffer under test, including public validation state.
  iree_hal_command_buffer_t* recording_ = nullptr;
};

TEST_F(CommandBufferTest, EmptyRecordingOwnsNoNativeCommandsOrArguments) {
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording_));
  iree_hal_amd_gpu_pm4_command_invocation_t invocation;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_command_invocation_initialize(
      recording_, {}, nullptr, 0, &invocation));
  const auto arguments =
      iree_hal_amd_gpu_pm4_command_next_arguments(invocation.cursor);
  EXPECT_EQ(arguments.length, 0u);
  EXPECT_EQ(arguments.alignment, 1u);
  iree_hal_amd_gpu_pm4_command_chunk_t chunk;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_command_prepare_chunk(
      &invocation, IREE_HAL_AMD_GPU_PM4_COMMAND_MAX_WORDS, {}, 0, 0, &chunk));
  EXPECT_EQ(chunk.word_count, 0u);
  EXPECT_EQ(chunk.argument_length, 0u);
  EXPECT_EQ(chunk.dispatch_count, 0u);
  EXPECT_EQ(chunk.next.block, nullptr);
  iree_hal_amd_gpu_pm4_command_invocation_deinitialize(&invocation);
}

TEST_F(CommandBufferTest, AdjacentBarriersCombineWithoutInvocationScan) {
  const iree_hal_barrier_t barrier = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  for (uint32_t i = 0; i < 10000; ++i) {
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording_, &barrier));
  }
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording_));
  iree_hal_amd_gpu_pm4_command_invocation_t invocation;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_command_invocation_initialize(
      recording_, {}, nullptr, 0, &invocation));
  iree_hal_command_buffer_release(recording_);
  recording_ = nullptr;
  iree_hal_amd_gpu_pm4_command_chunk_t chunk;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_command_prepare_chunk(&invocation, 9, {},
                                                            0, 0, &chunk));
  EXPECT_EQ(chunk.word_count, 0u);
  EXPECT_EQ(chunk.next.block, invocation.cursor.block);
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_command_prepare_chunk(&invocation, 10, {},
                                                            0, 0, &chunk));
  EXPECT_EQ(chunk.word_count, 10u);
  EXPECT_EQ(chunk.dispatch_count, 0u);
  EXPECT_EQ(chunk.argument_length, 0u);
  EXPECT_EQ(chunk.next.block, nullptr);
  EXPECT_NE(invocation.cursor.block, nullptr);
  // Repeating preparation does not consume the candidate or mutate its owner.
  iree_hal_amd_gpu_pm4_command_chunk_t repeated;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_command_prepare_chunk(&invocation, 10, {},
                                                            0, 0, &repeated));
  EXPECT_EQ(std::memcmp(chunk.words, repeated.words, 10 * sizeof(uint32_t)), 0);
  iree_hal_amd_gpu_pm4_command_invocation_deinitialize(&invocation);
}

}  // namespace
