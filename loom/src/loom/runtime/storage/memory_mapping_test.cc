// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/memory_test_fixture.h"

namespace {

using CompletedOperation =
    std::unique_ptr<loom_runtime_memory_operation_t,
                    decltype(&loom_runtime_memory_operation_free)>;

CompletedOperation AwaitOperation(loom_runtime_memory_operation_t* operation) {
  loom_runtime_memory_operation_await(operation);
  return CompletedOperation(operation, loom_runtime_memory_operation_free);
}

class MemoryMappingTest : public loom_runtime::testing::memory::MemoryTest {
 protected:
  void SetUpPool() override {
    mapping_queue.Initialize(loom_runtime_device_transfer_queue(device));
    UseBacking(false, 3 * kSlabSize, &mapping_queue);
    mapping_queue.observe_mappings = true;
  }
};

TEST_F(MemoryMappingTest, ContiguousRunKeepsIndependentPageOwnership) {
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.mapping_size;
  ASSERT_GE(geometry.backing_size, 4 * page);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, 4 * page, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, 2 * page, 256, &statistics[1], &second));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, 4 * page));
  auto observations = mapping_queue.WaitForMappings(1);
  ASSERT_EQ(observations[0].updates.size(), 1u);
  const auto original = observations[0].updates[0];
  EXPECT_EQ(original.operation, IREE_HAL_MEMORY_MAPPING_OPERATION_BIND);
  EXPECT_EQ(original.target_range.byte_offset, 0u);
  EXPECT_EQ(original.target_range.byte_length, 4 * page);
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 0, 0x11223344);
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 2 * page, 0x55667788);

  // Keeping alternate pages must split the unmap without widening ownership.
  loom_runtime_virtual_buffer_begin_trim(first);
  Keep(first, 0, page);
  Keep(first, 2 * page, page);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_trim(first));
  observations = mapping_queue.WaitForMappings(2);
  ASSERT_EQ(observations[1].updates.size(), 2u);
  for (size_t i = 0; i < 2; ++i) {
    const auto& update = observations[1].updates[i];
    EXPECT_EQ(update.operation, IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND);
    EXPECT_EQ(update.target_range.byte_offset, (2 * i + 1) * page);
    EXPECT_EQ(update.target_range.byte_length, page);
    EXPECT_EQ(update.previous_source, original.source);
    EXPECT_EQ(update.previous_source_offset,
              original.source_offset + (2 * i + 1) * page);
  }
  EXPECT_EQ(statistics[0].committed_bytes, 2 * page);

  // A contiguous destination reusing physical holes still needs two runs.
  const auto retained = loom_runtime_memory_pool_retained_bytes(pool);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(second, 0, 2 * page));
  observations = mapping_queue.WaitForMappings(3);
  ASSERT_EQ(observations[2].updates.size(), 2u);
  for (size_t i = 0; i < 2; ++i) {
    const auto& update = observations[2].updates[i];
    EXPECT_EQ(update.target_range.byte_offset, i * page);
    EXPECT_EQ(update.target_range.byte_length, page);
    EXPECT_EQ(update.source, original.source);
    EXPECT_EQ(update.source_offset,
              original.source_offset + (2 * i + 1) * page);
    RoundTrip(loom_runtime_virtual_buffer_handle(second), i * page,
              0xaabbccdd + i);
  }
  EXPECT_EQ(loom_runtime_memory_pool_retained_bytes(pool), retained);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), 0, 0x11223344);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), 2 * page,
                0x55667788);
}

TEST_F(MemoryMappingTest, RunsStopAtSourceParentsAndIndependentReservations) {
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.mapping_size;
  const uint64_t parent = geometry.backing_size;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, parent + page, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(pool, geometry, page, 256,
                                                      &statistics[1], &second));
  const loom_runtime_memory_range_t ranges[] = {{first, 0, parent + page},
                                                {second, 0, page}};
  bool admitted = false;
  IREE_ASSERT_OK(loom_runtime_memory_pool_try_commit(
      pool, IREE_ARRAYSIZE(ranges), ranges, &admitted));
  ASSERT_TRUE(admitted);
  const auto observations = mapping_queue.WaitForMappings(1);
  ASSERT_EQ(observations[0].updates.size(), 3u);
  const auto& updates = observations[0].updates;
  EXPECT_EQ(updates[0].target, loom_runtime_virtual_buffer_handle(first));
  EXPECT_EQ(updates[0].target_range.byte_length, parent);
  EXPECT_EQ(updates[1].target, updates[0].target);
  EXPECT_EQ(updates[1].target_range.byte_offset, parent);
  EXPECT_EQ(updates[1].target_range.byte_length, page);
  EXPECT_NE(updates[0].source, updates[1].source);
  EXPECT_EQ(updates[2].source, updates[1].source);
  EXPECT_EQ(updates[2].source_offset, updates[1].source_offset + page);
  EXPECT_EQ(updates[2].target, loom_runtime_virtual_buffer_handle(second));
  EXPECT_EQ(updates[2].target_range.byte_length, page);
  RoundTrip(loom_runtime_virtual_buffer_handle(first), parent, 0x12345678);
  RoundTrip(loom_runtime_virtual_buffer_handle(second), 0, 0x87654321);
  loom_runtime_virtual_buffer_begin_trim(first);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_trim(first));
  const auto trimmed = mapping_queue.WaitForMappings(2);
  ASSERT_EQ(trimmed[1].updates.size(), 2u);
  EXPECT_EQ(trimmed[1].updates[0].target_range.byte_length, parent);
  EXPECT_EQ(trimmed[1].updates[1].target_range.byte_length, page);
  EXPECT_EQ(statistics[0].committed_bytes, 0u);
  EXPECT_EQ(statistics[1].committed_bytes, page);
  CheckRetained(loom_runtime_virtual_buffer_handle(second), 0, 0x87654321);
}

TEST_F(MemoryMappingTest, SourceIoKeepsCallerOrderAndOverlappingRanges) {
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.mapping_size;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, 4 * page, 256, &statistics[0], &first));
  const loom_runtime_memory_range_t ranges[] = {{first, 2 * page, 2 * page},
                                                {first, 0, 2 * page},
                                                {first, 2 * page, page}};
  loom_runtime_memory_operation_t* operation = nullptr;
  loom_runtime_memory_admission_t admission;
  IREE_ASSERT_OK(loom_runtime_memory_pool_commit_async(
      pool, IREE_ARRAYSIZE(ranges), ranges, nullptr, 0, &admission,
      &operation));
  ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
  const auto completed = AwaitOperation(operation);
  IREE_ASSERT_OK(loom_runtime_memory_operation_query_status(operation));
  for (size_t i = 0; i < 4; ++i) {
    const auto input = loom_runtime_memory_operation_backing(operation, i);
    const auto published =
        loom_runtime_virtual_buffer_backing(first, (i + 2) % 4);
    EXPECT_EQ(input.buffer, published.buffer);
    EXPECT_EQ(input.offset, published.offset);
    EXPECT_EQ(input.length, published.length);
  }
  const auto observations = mapping_queue.WaitForMappings(1);
  ASSERT_EQ(observations[0].updates.size(), 2u);
  EXPECT_EQ(observations[0].updates[0].target_range.byte_offset, 2 * page);
  EXPECT_EQ(observations[0].updates[1].target_range.byte_offset, 0u);
  for (const auto& update : observations[0].updates) {
    EXPECT_EQ(update.target_range.byte_length, 2 * page);
    RoundTrip(loom_runtime_virtual_buffer_handle(first),
              update.target_range.byte_offset, 0x12345678);
  }
  EXPECT_EQ(statistics[0].committed_bytes, 4 * page);
}

TEST_F(MemoryMappingTest, GrowthRebindsRunsWithoutCopyingOrFillingHoles) {
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.mapping_size;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, 5 * page, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, 2 * page));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 3 * page, 2 * page));
  RoundTrip(loom_runtime_virtual_buffer_handle(first), page, 0x12345678);
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 4 * page, 0x87654321);
  const auto retained = loom_runtime_memory_pool_retained_bytes(pool);
  auto* previous = loom_runtime_virtual_buffer_handle(first);
  iree_hal_buffer_retain(previous);
  const std::unique_ptr<iree_hal_buffer_t, decltype(&iree_hal_buffer_release)>
      previous_owner(previous, iree_hal_buffer_release);
  loom_runtime_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_grow_async(
      first, 9 * page, nullptr, 0, &operation));
  const auto completed = AwaitOperation(operation);
  IREE_ASSERT_OK(loom_runtime_memory_operation_query_status(operation));
  const auto observations = mapping_queue.WaitForMappings(4);
  const auto& attached = observations[2].updates;
  const auto& detached = observations[3].updates;
  ASSERT_EQ(attached.size(), 2u);
  ASSERT_EQ(detached.size(), 2u);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(attached[i].target, loom_runtime_virtual_buffer_handle(first));
    EXPECT_EQ(attached[i].target_range.byte_offset, i * 3 * page);
    EXPECT_EQ(attached[i].target_range.byte_length, 2 * page);
    EXPECT_EQ(detached[i].target, previous);
    EXPECT_EQ(detached[i].target_range.byte_offset,
              attached[i].target_range.byte_offset);
    EXPECT_EQ(detached[i].target_range.byte_length, 2 * page);
    EXPECT_EQ(detached[i].previous_source, attached[i].source);
    EXPECT_EQ(detached[i].previous_source_offset, attached[i].source_offset);
  }
  EXPECT_EQ(statistics[0].committed_bytes, 4 * page);
  EXPECT_EQ(loom_runtime_memory_pool_retained_bytes(pool), retained);
  EXPECT_FALSE(loom_runtime_virtual_buffer_is_backed(first, 2 * page, page));
  CheckRetained(loom_runtime_virtual_buffer_handle(first), page, 0x12345678);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), 4 * page,
                0x87654321);
}

TEST_F(MemoryMappingTest, DeferredSourcesPublishExactTailBeforeAdjacentState) {
  ASSERT_NO_FATAL_FAILURE(
      UseDeferredBacking(3 * kSlabSize, &mapping_queue, true));
  const auto bulk = loom_runtime_pool_weight_geometry(storage_owner);
  const auto state = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t tail = state.mapping_size;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, bulk, bulk.mapping_size + tail, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, state, 2 * tail, 256, &statistics[1], &second));
  const loom_runtime_memory_range_t ranges[] = {
      {first, 0, bulk.mapping_size + tail}, {second, 0, 2 * tail}};
  deferred_backing->defer = true;
  loom_runtime_memory_operation_t* operation = nullptr;
  loom_runtime_memory_admission_t admission;
  IREE_ASSERT_OK(loom_runtime_memory_pool_commit_async(
      pool, IREE_ARRAYSIZE(ranges), ranges, nullptr, 0, &admission,
      &operation));
  ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
  deferred_backing->WaitForAcquisition();
  EXPECT_FALSE(loom_runtime_memory_operation_is_complete(operation));
  deferred_backing->ResumeAcquisitions();
  const auto completed = AwaitOperation(operation);
  IREE_ASSERT_OK(loom_runtime_memory_operation_query_status(operation));
  const auto observations = mapping_queue.WaitForMappings(1);
  ASSERT_EQ(observations[0].updates.size(), 3u);
  const auto& updates = observations[0].updates;
  EXPECT_EQ(updates[0].target_range.byte_length, bulk.mapping_size);
  EXPECT_EQ(updates[1].target_range.byte_offset, bulk.mapping_size);
  EXPECT_EQ(updates[1].target_range.byte_length, tail);
  EXPECT_EQ(updates[2].target_range.byte_length, 2 * tail);
  EXPECT_EQ(updates[1].source, updates[2].source);
  EXPECT_EQ(updates[1].source_offset + tail, updates[2].source_offset);
  RoundTrip(loom_runtime_virtual_buffer_handle(first), bulk.mapping_size,
            0x12345678);
  RoundTrip(loom_runtime_virtual_buffer_handle(second), tail, 0x87654321);
}

TEST_F(MemoryMappingTest, FailedRunDetachRetainsEveryPageForCheckedCleanup) {
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.mapping_size;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, 3 * page, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(pool, geometry, page, 256,
                                                      &statistics[1], &second));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, 3 * page));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(second, 0, page));
  RoundTrip(loom_runtime_virtual_buffer_handle(second), 0, 0x12345678);
  loom_runtime_virtual_buffer_begin_trim(first);
  mapping_queue.fail_next_map = true;
  loom_runtime_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_runtime_memory_pool_trim_async(pool, 1, &first, nullptr,
                                                     0, &operation));
  const auto completed = AwaitOperation(operation);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_runtime_memory_operation_query_status(operation));
  const auto observations = mapping_queue.WaitForMappings(3);
  ASSERT_EQ(observations[2].updates.size(), 1u);
  EXPECT_EQ(observations[2].updates[0].target_range.byte_length, 3 * page);
  EXPECT_EQ(statistics[0].committed_bytes, 3 * page);
  EXPECT_EQ(statistics[0].releasing_bytes, 0u);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_free(first));
  first = nullptr;
  EXPECT_EQ(statistics[0].committed_bytes, 0u);
  EXPECT_EQ(statistics[1].committed_bytes, page);
  CheckRetained(loom_runtime_virtual_buffer_handle(second), 0, 0x12345678);
}

}  // namespace
