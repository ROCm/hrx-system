// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/timestamp.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TimestampWritesTest : public ::testing::Test {
 protected:
  void TearDown() override {
    iree_hal_amd_gpu_timestamp_writes_deinitialize(iree_allocator_system(),
                                                   &writes_);
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
  // Recording-only pending writes under test.
  iree_hal_amd_gpu_timestamp_writes_t writes_ = {};
};

TEST_F(TimestampWritesTest, ExactSymbolicRangesAndUnresolvedAliases) {
  IREE_ASSERT_OK(iree_hal_amd_gpu_timestamp_writes_append(
      iree_allocator_system(), iree_hal_make_indirect_buffer_ref(2, 16, 128),
      &writes_));
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
      iree_allocator_system(), iree_hal_make_buffer_ref(&subspan, 8, 8),
      &writes_));
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
      iree_allocator_system(), iree_hal_make_indirect_buffer_ref(0, 0, 8),
      &writes_));
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
  barrier.flags = iree_hal_barrier_resolve_flags(&barrier);
  EXPECT_TRUE(iree_hal_amd_gpu_timestamp_writes_overlap(&writes_, &barrier));
  barrier.target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH;
  barrier.flags = 0;
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

TEST_F(TimestampWritesTest, HostCacheVisibilityPreservesPayloadScope) {
  IREE_ASSERT_OK(iree_hal_amd_gpu_timestamp_writes_append(
      iree_allocator_system(), iree_hal_make_indirect_buffer_ref(0, 16, 8),
      &writes_));
  iree_hal_buffer_barrier_t range = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_HOST_READ,
      .buffer_ref = iree_hal_make_indirect_buffer_ref(0, 32, 8),
  };
  iree_hal_barrier_t barrier = {
      .source_stage_mask =
          IREE_HAL_EXECUTION_STAGE_TRANSFER | IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_HOST,
      .buffer_barrier_count = 1,
      .buffer_barriers = &range,
  };
  barrier.flags = iree_hal_barrier_resolve_flags(&barrier);
  EXPECT_EQ(barrier.flags, 0u);
  EXPECT_EQ(iree_hal_barrier_resolve_cache_flags(&barrier),
            IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE);
  EXPECT_FALSE(iree_hal_amd_gpu_timestamp_writes_overlap(&writes_, &barrier));
  range.buffer_ref.offset = 16;
  EXPECT_TRUE(iree_hal_amd_gpu_timestamp_writes_overlap(&writes_, &barrier));
  range.source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE;
  EXPECT_FALSE(iree_hal_amd_gpu_timestamp_writes_overlap(&writes_, &barrier));

  // A global access dependency for shader writes does not consume clock writes.
  const iree_hal_memory_barrier_t memory = {
      .source_scope = range.source_scope,
      .target_scope = range.target_scope,
  };
  barrier.buffer_barrier_count = 0;
  barrier.memory_barrier_count = 1;
  barrier.memory_barriers = &memory;
  barrier.flags = iree_hal_barrier_resolve_flags(&barrier);
  EXPECT_FALSE(iree_hal_amd_gpu_timestamp_writes_overlap(&writes_, &barrier));

  // Explicit global publication consumes all covered writes despite the range.
  barrier.memory_barrier_count = 0;
  barrier.buffer_barrier_count = 1;
  barrier.flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE;
  EXPECT_TRUE(iree_hal_amd_gpu_timestamp_writes_overlap(&writes_, &barrier));
  barrier.flags = 0;
  barrier.effects.bits = IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM;
  barrier.flags = iree_hal_barrier_resolve_flags(&barrier);
  EXPECT_TRUE(iree_hal_amd_gpu_timestamp_writes_overlap(&writes_, &barrier));
}

TEST_F(TimestampWritesTest, RetiredCohortReusesDestinationStorage) {
  for (uint32_t i = 0; i < 9; ++i) {
    IREE_ASSERT_OK(iree_hal_amd_gpu_timestamp_writes_append(
        iree_allocator_system(), iree_hal_make_indirect_buffer_ref(0, i * 8, 8),
        &writes_));
  }
  EXPECT_EQ(writes_.count, 9u);
  EXPECT_TRUE(Overlaps(iree_hal_make_indirect_buffer_ref(0, 64, 8)));
  const auto* pending = writes_.pending;
  const auto capacity = writes_.capacity;
  writes_.count = 0;
  IREE_ASSERT_OK(iree_hal_amd_gpu_timestamp_writes_append(
      iree_allocator_system(), iree_hal_make_indirect_buffer_ref(1, 8, 8),
      &writes_));
  EXPECT_EQ(writes_.count, 1u);
  EXPECT_EQ(writes_.pending, pending);
  EXPECT_EQ(writes_.capacity, capacity);
}

}  // namespace
