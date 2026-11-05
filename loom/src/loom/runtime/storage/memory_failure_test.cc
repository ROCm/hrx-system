// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/memory_test_fixture.h"

namespace {

using loom_runtime::testing::memory::MemoryTest;

TEST_F(MemoryTest, RejectedAllocationPublishesFailureWithoutBacking) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking());
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, loom_runtime_pool_weight_geometry(storage_owner), kSlabSize, 256,
      &statistics[0], &first));
  deferred_backing->reject_next_acquisition = true;
  const loom_runtime_memory_range_t range = {first, 0, kSlabSize};
  loom_runtime_memory_admission_t admission;
  loom_runtime_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_runtime_memory_pool_commit_async(
      pool, 1, &range, nullptr, 0, &admission, &operation));
  ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
  EXPECT_FALSE(deferred_backing->reject_next_acquisition);
  loom_runtime_memory_operation_await(operation);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_runtime_memory_operation_query_status(operation));
  loom_runtime_memory_operation_free(operation);
  IREE_EXPECT_OK(loom_runtime_memory_pool_query_status(pool));
  EXPECT_EQ(statistics[0].pending_bytes, 0u);
  EXPECT_EQ(statistics[0].committed_bytes, 0u);

  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, kSlabSize));
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);
}

TEST_F(MemoryTest, FailedAcquisitionPropagatesBeforeBackingPublication) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking());
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, loom_runtime_pool_weight_geometry(storage_owner), kSlabSize, 256,
      &statistics[0], &first));
  deferred_backing->defer = true;
  const loom_runtime_memory_range_t range = {first, 0, kSlabSize};
  loom_runtime_memory_admission_t admission;
  loom_runtime_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_runtime_memory_pool_commit_async(
      pool, 1, &range, nullptr, 0, &admission, &operation));
  ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
  deferred_backing->WaitForAcquisition();
  EXPECT_FALSE(loom_runtime_memory_operation_is_complete(operation));

  deferred_backing->FailOne();
  loom_runtime_memory_operation_await(operation);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_runtime_memory_operation_query_status(operation));
  // Failed readiness forbids resolving the borrowed parent. Publication returns
  // its lease, and no physical handle was exposed to file-loading producers.
  loom_runtime_memory_operation_free(operation);
  IREE_EXPECT_OK(loom_runtime_memory_pool_query_status(pool));
  EXPECT_EQ(statistics[0].pending_bytes, 0u);
  EXPECT_EQ(statistics[0].committed_bytes, 0u);

  deferred_backing->ResumeAcquisitions();
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, kSlabSize));
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 0, 0x87654321);
}

TEST_F(MemoryTest, NativeAllocationFailurePreservesWarmWorkAndAllowsRetry) {
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, loom_runtime_pool_weight_geometry(storage_owner), kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, loom_runtime_pool_weight_geometry(storage_owner), kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, kSlabSize));
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);

  native.FailNextAllocation(IREE_ALLOCATOR_COMMAND_CALLOC,
                            /*native_only=*/true);
  const loom_runtime_memory_range_t range = {second, 0, kSlabSize};
  loom_runtime_memory_admission_t admission;
  loom_runtime_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_runtime_memory_pool_commit_async(
      pool, 1, &range, nullptr, 0, &admission, &operation));
  ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
  loom_runtime_memory_operation_await(operation);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_runtime_memory_operation_query_status(operation));
  loom_runtime_memory_operation_free(operation);
  IREE_EXPECT_OK(loom_runtime_memory_pool_query_status(pool));
  EXPECT_EQ(statistics[1].pending_bytes, 0u);
  EXPECT_EQ(statistics[1].committed_bytes, 0u);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);

  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(second, 0, kSlabSize));
  RoundTrip(loom_runtime_virtual_buffer_handle(second), 0, 0x87654321);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);
}

TEST_F(MemoryTest, FailedMappingKeepsSourcesUntilCheckedDetach) {
  constexpr uint64_t kMappingBytes = 2 * kSlabSize;
  mapping_queue.Initialize(loom_runtime_device_transfer_queue(device));
  ASSERT_NO_FATAL_FAILURE(
      UseDeferredBacking(kMappingBytes + kSlabSize, &mapping_queue));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, loom_runtime_pool_weight_geometry(storage_owner), kMappingBytes,
      256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, loom_runtime_pool_weight_geometry(storage_owner), kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(second, 0, kSlabSize));
  RoundTrip(loom_runtime_virtual_buffer_handle(second), 0, 0x12345678);

  // Mapping cannot start until its actual sources exist. The later rejection
  // must keep those sources until checked detach establishes native safety.
  deferred_backing->defer = true;
  mapping_queue.fail_next_map = true;
  const loom_runtime_memory_range_t range = {first, 0, kMappingBytes};
  loom_runtime_memory_admission_t commit_admission;
  loom_runtime_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_runtime_memory_pool_commit_async(
      pool, 1, &range, nullptr, 0, &commit_admission, &operation));
  ASSERT_EQ(commit_admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
  deferred_backing->WaitForAcquisition();
  EXPECT_TRUE(mapping_queue.fail_next_map);
  EXPECT_FALSE(loom_runtime_memory_operation_is_complete(operation));
  deferred_backing->ResumeAcquisitions();
  loom_runtime_memory_operation_await(operation);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_runtime_memory_operation_query_status(operation));
  loom_runtime_memory_operation_free(operation);
  IREE_EXPECT_OK(loom_runtime_memory_pool_query_status(pool));
  EXPECT_EQ(statistics[0].pending_bytes, 0u);
  EXPECT_EQ(statistics[0].committed_bytes, kMappingBytes);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_runtime_virtual_buffer_commit(first, 0, kSlabSize));
  CheckRetained(loom_runtime_virtual_buffer_handle(second), 0, 0x12345678);

  IREE_ASSERT_OK(loom_runtime_virtual_buffer_free(first));
  first = nullptr;
  loom_runtime_memory_pool_trim(pool, 0);
  IREE_EXPECT_OK(loom_runtime_memory_pool_await(pool));
  EXPECT_EQ(loom_runtime_memory_pool_statistics(pool).committed_bytes,
            kSlabSize);
  CheckRetained(loom_runtime_virtual_buffer_handle(second), 0, 0x12345678);
}

}  // namespace
