// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/storage/memory_test_fixture.h"

namespace {

using loom_serve::testing::memory::MemoryTest;

TEST_F(MemoryTest, RejectedAllocationPublishesFailureWithoutBacking) {
  mapping_queue.Initialize(loom_serve_device_transfer_queue(device));
  ASSERT_NO_FATAL_FAILURE(UseBacking(false, 3 * kSlabSize, &mapping_queue));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_pool_weight_geometry(storage_owner), kSlabSize, 256,
      &statistics[0], &first));
  mapping_queue.fail_next_allocation = true;
  const loom_serve_memory_range_t range = {first, 0, kSlabSize};
  loom_serve_memory_admission_t admission;
  loom_serve_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
      pool, 1, &range, nullptr, 0, &admission, &operation));
  ASSERT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
  EXPECT_FALSE(mapping_queue.fail_next_allocation);
  EXPECT_EQ(loom_serve_memory_operation_backing(operation, 0), nullptr);
  loom_serve_memory_operation_await(operation);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_memory_operation_query_status(operation));
  loom_serve_memory_operation_free(operation);
  IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool));
  EXPECT_EQ(statistics[0].pending_bytes, 0u);
  EXPECT_EQ(statistics[0].committed_bytes, 0u);

  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
}

TEST_F(MemoryTest, FailedAcquisitionKeepsPublishedBackingCapturable) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking());
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_pool_weight_geometry(storage_owner), kSlabSize, 256,
      &statistics[0], &first));
  deferred_backing->defer = true;
  const loom_serve_memory_range_t range = {first, 0, kSlabSize};
  loom_serve_memory_admission_t admission;
  loom_serve_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
      pool, 1, &range, nullptr, 0, &admission, &operation));
  ASSERT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
  deferred_backing->WaitForAcquisition();
  EXPECT_FALSE(loom_serve_memory_operation_is_complete(operation));
  iree_hal_buffer_t* backing =
      loom_serve_memory_operation_backing(operation, 0);
  ASSERT_NE(backing, nullptr);

  deferred_backing->FailOne();
  loom_serve_memory_operation_await(operation);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_memory_operation_query_status(operation));
  // A consumer can arrive after native failure, before it observes readiness.
  // Completion must preserve the published handle until that consumer captures
  // it or releases the operation, even when no backing allocation succeeded.
  EXPECT_EQ(loom_serve_memory_operation_backing(operation, 0), backing);
  iree_hal_buffer_retain(backing);
  loom_serve_memory_operation_free(operation);
  EXPECT_EQ(iree_hal_buffer_byte_length(backing), kSlabSize);
  iree_hal_buffer_release(backing);
  IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool));
  EXPECT_EQ(statistics[0].pending_bytes, 0u);
  EXPECT_EQ(statistics[0].committed_bytes, 0u);

  deferred_backing->ResumeAcquisitions();
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x87654321);
}

TEST_F(MemoryTest, NativeAllocationFailurePreservesWarmWorkAndAllowsRetry) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_pool_weight_geometry(storage_owner), kSlabSize, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_pool_weight_geometry(storage_owner), kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);

  native.FailNextAllocation(IREE_ALLOCATOR_COMMAND_CALLOC,
                            /*native_only=*/true);
  const loom_serve_memory_range_t range = {second, 0, kSlabSize};
  loom_serve_memory_admission_t admission;
  loom_serve_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
      pool, 1, &range, nullptr, 0, &admission, &operation));
  ASSERT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
  loom_serve_memory_operation_await(operation);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_memory_operation_query_status(operation));
  loom_serve_memory_operation_free(operation);
  IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool));
  EXPECT_EQ(statistics[1].pending_bytes, 0u);
  EXPECT_EQ(statistics[1].committed_bytes, 0u);
  CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);

  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x87654321);
  CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
}

TEST_F(MemoryTest, FailedMappingKeepsSourcesUntilCheckedDetach) {
  constexpr uint64_t kMappingBytes = 2 * kSlabSize;
  mapping_queue.Initialize(loom_serve_device_transfer_queue(device));
  ASSERT_NO_FATAL_FAILURE(
      UseDeferredBacking(kMappingBytes + kSlabSize, &mapping_queue));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_pool_weight_geometry(storage_owner), kMappingBytes, 256,
      &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_pool_weight_geometry(storage_owner), kSlabSize, 256,
      &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);

  // Mapping rejection cannot retire the allocation already accepted by the
  // native queue. Keep it pending until the failed mapping has been observed.
  deferred_backing->defer = true;
  mapping_queue.fail_next_map = true;
  const loom_serve_memory_range_t range = {first, 0, kMappingBytes};
  loom_serve_memory_admission_t commit_admission;
  loom_serve_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
      pool, 1, &range, nullptr, 0, &commit_admission, &operation));
  ASSERT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
  deferred_backing->WaitForAcquisition();
  EXPECT_FALSE(mapping_queue.fail_next_map);
  EXPECT_FALSE(loom_serve_memory_operation_is_complete(operation));
  deferred_backing->ResumeAcquisitions();
  loom_serve_memory_operation_await(operation);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_memory_operation_query_status(operation));
  loom_serve_memory_operation_free(operation);
  IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool));
  EXPECT_EQ(statistics[0].pending_bytes, 0u);
  EXPECT_EQ(statistics[0].committed_bytes, kMappingBytes);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);

  IREE_ASSERT_OK(loom_serve_virtual_buffer_free(first));
  first = nullptr;
  loom_serve_memory_pool_trim(pool, 0);
  IREE_EXPECT_OK(loom_serve_memory_pool_await(pool));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, kSlabSize);
  CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
}

}  // namespace
