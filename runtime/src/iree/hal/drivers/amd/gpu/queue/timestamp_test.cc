// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/timestamp.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TimestampWritesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_hal_amd_gpu_recording_initialize(iree_allocator_system(), &recording_);
  }
  void TearDown() override {
    iree_hal_amd_gpu_timestamp_writes_deinitialize(iree_allocator_system(),
                                                   &writes_);
    iree_hal_amd_gpu_recording_deinitialize(&recording_);
  }
  bool Overlaps(iree_hal_buffer_ref_t range) {
    const iree_hal_buffer_barrier_t payload = {
        .source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
        .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ,
        .buffer_ref = range,
    };
    const iree_hal_barrier_t barrier = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
        .buffer_barrier_count = 1,
        .buffer_barriers = &payload,
    };
    return iree_hal_amd_gpu_timestamp_writes_overlap(&writes_, &barrier);
  }
  // Production append storage owns the invocation-private scratch layout.
  iree_hal_amd_gpu_recording_t recording_ = {};
  // Recording-only pending writes under test.
  iree_hal_amd_gpu_timestamp_writes_t writes_ = {};
};

TEST_F(TimestampWritesTest, ExactSymbolicRangesAndUnresolvedAliases) {
  IREE_ASSERT_OK(iree_hal_amd_gpu_timestamp_writes_append(
      &recording_, iree_hal_make_indirect_buffer_ref(2, 16, 128), &writes_));
  EXPECT_EQ(writes_.pending[0].length, 8u);
  EXPECT_FALSE(Overlaps(iree_hal_make_indirect_buffer_ref(2, 0, 16)));
  EXPECT_TRUE(Overlaps(iree_hal_make_indirect_buffer_ref(2, 15, 2)));
  EXPECT_TRUE(Overlaps(iree_hal_make_indirect_buffer_ref(2, 23, 1)));
  EXPECT_FALSE(Overlaps(iree_hal_make_indirect_buffer_ref(2, 24, 8)));
  EXPECT_FALSE(Overlaps(iree_hal_make_indirect_buffer_ref(2, 20, 0)));
  EXPECT_TRUE(Overlaps(
      iree_hal_make_indirect_buffer_ref(2, 23, IREE_HAL_WHOLE_BUFFER)));
  EXPECT_FALSE(Overlaps(
      iree_hal_make_indirect_buffer_ref(2, 24, IREE_HAL_WHOLE_BUFFER)));
  EXPECT_TRUE(Overlaps(iree_hal_make_indirect_buffer_ref(1, 0, 8)));
  EXPECT_FALSE(Overlaps(iree_hal_make_indirect_buffer_ref(1, 20, 0)));
}

TEST_F(TimestampWritesTest, DirectSubspansAndMixedBindingIdentities) {
  static const iree_hal_buffer_vtable_t vtable = {
      .destroy = [](iree_hal_buffer_t*) {},
  };
  iree_hal_buffer_t backing = {}, subspan = {}, separate = {};
  for (auto* buffer : {&backing, &separate}) {
    iree_hal_buffer_initialize({}, buffer, 64, 0, 64,
                               IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                               IREE_HAL_MEMORY_ACCESS_ALL,
                               IREE_HAL_BUFFER_USAGE_TRANSFER, &vtable, buffer);
  }
  iree_hal_buffer_initialize({}, &backing, 64, 16, 32,
                             IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                             IREE_HAL_MEMORY_ACCESS_ALL,
                             IREE_HAL_BUFFER_USAGE_TRANSFER, &vtable, &subspan);
  IREE_ASSERT_OK(iree_hal_amd_gpu_timestamp_writes_append(
      &recording_, iree_hal_make_buffer_ref(&subspan, 8, 8), &writes_));
  EXPECT_TRUE(Overlaps(iree_hal_make_buffer_ref(&backing, 24, 8)));
  EXPECT_FALSE(Overlaps(iree_hal_make_buffer_ref(&backing, 32, 8)));
  EXPECT_TRUE(Overlaps(iree_hal_make_buffer_ref(&separate, 24, 8)));
  EXPECT_TRUE(Overlaps(iree_hal_make_indirect_buffer_ref(0, 0, 8)));
  iree_hal_buffer_release(&subspan);
  iree_hal_buffer_release(&separate);
  iree_hal_buffer_release(&backing);
}

TEST_F(TimestampWritesTest, ExecutionDoesNotDeliverPayload) {
  IREE_ASSERT_OK(iree_hal_amd_gpu_timestamp_writes_append(
      &recording_, iree_hal_make_indirect_buffer_ref(0, 0, 8), &writes_));
  const iree_hal_memory_barrier_t payload = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ,
  };
  iree_hal_barrier_t barrier = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .memory_barriers = &payload,
  };
  EXPECT_FALSE(iree_hal_amd_gpu_timestamp_writes_overlap(&writes_, &barrier));
  barrier.target_stage_mask = IREE_HAL_EXECUTION_STAGE_HOST;
  EXPECT_FALSE(iree_hal_amd_gpu_timestamp_writes_overlap(&writes_, &barrier));
  barrier.memory_barrier_count = 1;
  EXPECT_TRUE(iree_hal_amd_gpu_timestamp_writes_overlap(&writes_, &barrier));
  barrier.source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH;
  EXPECT_FALSE(iree_hal_amd_gpu_timestamp_writes_overlap(&writes_, &barrier));
  barrier.source_stage_mask = 0;
  EXPECT_TRUE(iree_hal_amd_gpu_timestamp_writes_overlap(&writes_, &barrier));
  barrier.memory_barrier_count = 0;
  barrier.flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE;
  EXPECT_TRUE(iree_hal_amd_gpu_timestamp_writes_overlap(&writes_, &barrier));
  writes_.count = 0;
  EXPECT_FALSE(iree_hal_amd_gpu_timestamp_writes_overlap(&writes_, &barrier));
}

TEST_F(TimestampWritesTest, RecordingReusesOneInvocationScratchWord) {
  EXPECT_EQ(recording_.data.length, 0u);
  for (uint32_t i = 0; i < 9; ++i) {
    IREE_ASSERT_OK(iree_hal_amd_gpu_timestamp_writes_append(
        &recording_, iree_hal_make_indirect_buffer_ref(0, i * 8, 8), &writes_));
  }
  EXPECT_EQ(recording_.data.length, 4u);
  EXPECT_EQ(writes_.count, 9u);
  EXPECT_EQ(writes_.scratch_offset, 0u);
  EXPECT_TRUE(Overlaps(iree_hal_make_indirect_buffer_ref(0, 64, 8)));
  writes_.count = 0;
  IREE_ASSERT_OK(iree_hal_amd_gpu_timestamp_writes_append(
      &recording_, iree_hal_make_indirect_buffer_ref(1, 8, 8), &writes_));
  EXPECT_EQ(recording_.data.length, 4u);
  EXPECT_EQ(writes_.count, 1u);
}

}  // namespace
