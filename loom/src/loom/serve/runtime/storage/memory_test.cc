// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/storage/memory.h"

#include <array>
#include <atomic>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/testing/device.h"
#include "loom/serve/testing/native_allocator.h"

namespace {

class MemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_serve_host_t* host = nullptr;
    IREE_ASSERT_OK(
        loom_serve_host_create(1, nullptr, iree_allocator_system(), &host));
    iree_hal_driver_t* driver = nullptr;
    IREE_ASSERT_OK(
        loom_serve_test_driver_create(host, iree_allocator_system(), &driver));
    const loom_serve_device_options_t options = {
        .driver = driver,
        .uri = loom_serve_test_device()->device_uri,
        .backing = LOOM_SERVE_DEVICE_BACKING_ELASTIC,
        .slab_size = kSlabSize,
        .memory_limit = 3 * kSlabSize,
    };
    auto status = loom_serve_device_group_allocate(host, 1, &options,
                                                   native.allocator(), &group);
    iree_hal_driver_release(driver);
    loom_serve_host_release(host);
    IREE_ASSERT_OK(status);
    device = loom_serve_device_group_at(group, 0);
    pool = loom_serve_device_memory_pool(device);
    IREE_ASSERT_OK(loom_serve_device_create_execution(device, &execution));
    IREE_ASSERT_OK(native.Observe(device));
  }

  void TearDown() override {
    iree_hal_buffer_release(workspace);
    IREE_ASSERT_OK(loom_serve_execution_drain(execution));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_free(first));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_free(second));
    // Intentional failure cases retain their original status while native
    // disposal still reaches its physical capacity boundary.
    iree_status_ignore(loom_serve_memory_pool_await(pool));
    const auto statistics = loom_serve_memory_pool_statistics(pool);
    EXPECT_EQ(statistics.committed_bytes, 0u);
    EXPECT_EQ(statistics.reserved_bytes, 0u);
    loom_serve_execution_release(execution);
    IREE_EXPECT_OK(loom_serve_device_group_free(group));
  }

  void RoundTrip(iree_hal_buffer_t* buffer, uint64_t offset, uint32_t value) {
    std::array<uint32_t, 64> observed = {};
    const iree_hal_transfer_operation_t operations[] = {
        {
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
            .fill =
                {
                    .target_buffer = buffer,
                    .target_offset = offset,
                    .length = sizeof(observed),
                    .pattern = &value,
                    .pattern_length = sizeof(value),
                },
        },
        {
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
            .download =
                {
                    .source_buffer = buffer,
                    .source_offset = offset,
                    .target = observed.data(),
                    .length = sizeof(observed),
                },
        },
    };
    uint64_t completion = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_transfer(execution, 1, operations, &completion));
    IREE_ASSERT_OK(loom_serve_execution_transfer(execution, 1, operations + 1,
                                                 &completion));
    IREE_ASSERT_OK(loom_serve_execution_wait(execution, completion));
    for (const auto word : observed) {
      EXPECT_EQ(word, value);
    }
  }

  void CheckRetained(iree_hal_buffer_t* buffer, uint64_t offset,
                     uint32_t value) {
    std::array<uint32_t, 64> observed = {};
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = buffer,
                .source_offset = offset,
                .target = observed.data(),
                .length = sizeof(observed),
            },
    };
    uint64_t completion = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_transfer(execution, 1, &download, &completion));
    IREE_ASSERT_OK(loom_serve_execution_wait(execution, completion));
    for (const auto word : observed) {
      EXPECT_EQ(word, value);
    }
  }

  // Device allocator dependency outliving native cleanup and the group.
  loom_serve::testing::NativeAllocator native;
  // Coarse physical slabs, independent of logical KV block size.
  static constexpr uint64_t kSlabSize = 2 * 1024 * 1024;
  // Offset deliberately crossing the 32-bit byte-address boundary.
  static constexpr uint64_t kHighOffset = (uint64_t{1} << 32) + 512;
  // Application group enclosing all device resources.
  loom_serve_device_group_t* group = nullptr;
  // Borrowed device record within the application group.
  loom_serve_device_t* device = nullptr;
  // Owned model timeline for transfers and workspace allocations.
  loom_serve_execution_t* execution = nullptr;
  // One physical budget shared by the independent reservations.
  loom_serve_memory_pool_t* pool = nullptr;
  // Independent accounting groups competing for the same physical budget.
  std::array<loom_serve_memory_statistics_t, 2> statistics = {};
  // First model-like reservation, sparsely backed beyond four GiB.
  loom_serve_virtual_buffer_t* first = nullptr;
  // Independent reservation competing for the same physical budget.
  loom_serve_virtual_buffer_t* second = nullptr;
  // Private command allocation retained until its queue release is submitted.
  iree_hal_buffer_t* workspace = nullptr;
};

TEST_F(MemoryTest, PendingReleaseRetainsBudgetAndLeavesKeptSlabsUsable) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, 3 * kSlabSize, 256,
                                                    &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, 2 * kSlabSize, 256,
                                                    &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 3 * kSlabSize));
  ASSERT_NO_FATAL_FAILURE(
      RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0xA1B2C3D4));
  std::atomic<uint32_t> wakes = 0;
  loom_serve_memory_pool_set_observer(
      pool,
      [](void* user_data) {
        static_cast<std::atomic<uint32_t>*>(user_data)->fetch_add(1);
      },
      &wakes);
  loom_serve_memory_operation_t* operation = nullptr;
  iree_hal_semaphore_t* ready = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready));
  {
    loom_serve::testing::ScopedNativeHold gate(native,
                                               IREE_ALLOCATOR_COMMAND_FREE);
    loom_serve_virtual_buffer_begin_trim(first);
    loom_serve_virtual_buffer_keep(first, 0, kSlabSize);
    IREE_ASSERT_OK(loom_serve_memory_pool_trim_async(pool, 1, &first, ready, 1,
                                                     &operation));
    ASSERT_NE(operation, nullptr);
    gate.WaitEntered();
    loom_serve_memory_operation_await(operation);
    EXPECT_TRUE(loom_serve_memory_operation_is_complete(operation));
    const auto memory = loom_serve_memory_pool_statistics(pool);
    EXPECT_EQ(memory.committed_bytes, 3 * kSlabSize);
    EXPECT_EQ(memory.releasing_bytes, 2 * kSlabSize);
    EXPECT_EQ(statistics[0].releasing_bytes, 0u);
    loom_serve_memory_admission_t admission;
    IREE_ASSERT_OK(
        loom_serve_memory_pool_prepare(pool, 2 * kSlabSize, &admission));
    EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_DEFERRED);
    ASSERT_NO_FATAL_FAILURE(
        CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0xA1B2C3D4));
    EXPECT_EQ(wakes.load(), 1u);
    uint64_t value = 0;
    IREE_EXPECT_OK(iree_hal_semaphore_query(ready, &value));
    EXPECT_EQ(value, 1u);
  }
  IREE_ASSERT_OK(iree_hal_semaphore_wait(ready, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_TRUE(loom_serve_memory_operation_is_complete(operation));
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  EXPECT_GE(wakes.load(), 1u);
  EXPECT_EQ(statistics[0].releasing_bytes, 0u);
  EXPECT_EQ(statistics[0].committed_bytes, kSlabSize);
  EXPECT_EQ(statistics[0].released_bytes, 2 * kSlabSize);
  loom_serve_memory_operation_free(operation);
  loom_serve_memory_pool_set_observer(pool, nullptr, nullptr);
  operation = nullptr;
  {
    loom_serve::testing::ScopedNativeHold gate(native,
                                               IREE_ALLOCATOR_COMMAND_CALLOC);
    IREE_ASSERT_OK(loom_serve_memory_pool_trim_async(pool, 1, &first, ready, 2,
                                                     &operation));
    EXPECT_EQ(operation, nullptr);
    uint64_t value = 0;
    IREE_EXPECT_OK(iree_hal_semaphore_query(ready, &value));
    EXPECT_EQ(value, 2u);
  }
  iree_hal_semaphore_release(ready);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, 2 * kSlabSize));
  ASSERT_NO_FATAL_FAILURE(
      RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x55667788));
}

TEST_F(MemoryTest, PoolObserverDetachesFromAlreadyAcceptedNativeWork) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, kSlabSize, 256,
                                                    &statistics[0], &first));
  std::atomic<uint32_t> wakes = 0;
  loom_serve_memory_pool_set_observer(
      pool,
      [](void* user_data) {
        static_cast<std::atomic<uint32_t>*>(user_data)->fetch_add(1);
      },
      &wakes);
  loom_serve_memory_operation_t* operation = nullptr;
  {
    loom_serve::testing::ScopedNativeHold gate(native,
                                               IREE_ALLOCATOR_COMMAND_CALLOC);
    const loom_serve_memory_range_t range = {first, 0, kSlabSize};
    bool admitted = false;
    IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &range, nullptr, 0, &admitted, &operation));
    ASSERT_TRUE(admitted);
    gate.WaitEntered();
    loom_serve_memory_pool_set_observer(pool, nullptr, nullptr);
  }
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  EXPECT_EQ(wakes.load(), 0u);
  loom_serve_memory_operation_free(operation);
}

TEST_F(MemoryTest, SparseGrowthTrimRegrowthAndSharedAdmission) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, kHighOffset + kSlabSize, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, 2 * kSlabSize, 256,
                                                    &statistics[1], &second));
  auto* const identity = loom_serve_virtual_buffer_handle(first);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_device_group_free(group));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, 0u);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, kHighOffset, 256));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 512));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 256, 256));
  RoundTrip(identity, kHighOffset, 0x12345678);
  RoundTrip(identity, 0, 0xABCDEF01);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            2 * kSlabSize);

  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, 256));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_virtual_buffer_commit(second, kSlabSize, 256));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            3 * kSlabSize);
  loom_serve_virtual_buffer_begin_trim(first);
  loom_serve_virtual_buffer_keep(first, kHighOffset, 256);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(first));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).released_bytes, kSlabSize);
  EXPECT_EQ(loom_serve_virtual_buffer_handle(first), identity);
  CheckRetained(identity, kHighOffset, 0x12345678);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, kSlabSize, 256));
  RoundTrip(loom_serve_virtual_buffer_handle(second), kSlabSize, 0xDEADBEEF);
  loom_serve_virtual_buffer_begin_trim(second);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 512));
  EXPECT_EQ(loom_serve_virtual_buffer_handle(first), identity);
  RoundTrip(identity, 0, 0xCAFEBABE);
  CheckRetained(identity, kHighOffset, 0x12345678);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).peak_bytes, 3 * kSlabSize);
  EXPECT_EQ(statistics[0].committed_bytes, 2 * kSlabSize);
  EXPECT_EQ(statistics[1].committed_bytes, 0u);
  EXPECT_EQ(statistics[0].released_bytes, kSlabSize);
  EXPECT_EQ(statistics[1].released_bytes, 2 * kSlabSize);
  EXPECT_EQ(statistics[0].peak_bytes, 2 * kSlabSize);
  EXPECT_EQ(statistics[1].peak_bytes, 2 * kSlabSize);
}

TEST_F(MemoryTest, CohortAdmissionCountsUniqueSlabsAndLeavesDenialUntouched) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, 3 * kSlabSize, 256,
                                                    &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, 2 * kSlabSize, 256,
                                                    &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 256));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
  const loom_serve_memory_range_t ranges[] = {
      {first, 0, 256},
      {first, kSlabSize, 256},
      {first, kSlabSize + 128, 256},
      {second, kSlabSize - 128, 256},
  };
  bool admitted = true;
  IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(pool, IREE_ARRAYSIZE(ranges),
                                                   ranges, &admitted));
  EXPECT_FALSE(admitted);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, kSlabSize);
  EXPECT_EQ(statistics[1].committed_bytes, 0u);
  CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
  IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(pool, 3, ranges, &admitted));
  EXPECT_TRUE(admitted);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            2 * kSlabSize);
  RoundTrip(loom_serve_virtual_buffer_handle(first), kSlabSize, 0x87654321);
  IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(pool, 3, ranges, &admitted));
  EXPECT_TRUE(admitted);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            2 * kSlabSize);
  CheckRetained(loom_serve_virtual_buffer_handle(first), kSlabSize, 0x87654321);
  const loom_serve_memory_range_t last = {second, 0, 256};
  IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(pool, 1, &last, &admitted));
  EXPECT_TRUE(admitted);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            3 * kSlabSize);
}

TEST_F(MemoryTest, RecommittedSlabsPublishEveryWordOfTheNewGeneration) {
  constexpr size_t kGenerationCount = 4;
  constexpr uint64_t kOrigin = uint64_t{1} << 32;
  constexpr size_t kWordsPerSlab = kSlabSize / sizeof(uint32_t);
  const std::array<loom_serve_virtual_buffer_t**, 2> destinations = {&first,
                                                                     &second};
  for (size_t i = 0; i < destinations.size(); ++i) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
        pool, kOrigin + 4 * kSlabSize, 256, &statistics[i], destinations[i]));
  }
  const std::array<iree_hal_buffer_t*, 2> handles = {
      loom_serve_virtual_buffer_handle(first),
      loom_serve_virtual_buffer_handle(second)};
  // Prepare every generation before the first mapping turnover. Comparing or
  // filling a large host image between remap and device access would perturb
  // the publication boundary this case exercises.
  std::array<std::vector<uint32_t>, kGenerationCount> expected;
  for (size_t generation = 0; generation < kGenerationCount; ++generation) {
    expected[generation].resize(2 * kWordsPerSlab);
    for (size_t word = 0; word < expected[generation].size(); ++word) {
      expected[generation][word] =
          static_cast<uint32_t>((generation + 1) << 28) |
          static_cast<uint32_t>(word);
    }
  }
  std::vector<uint32_t> observed(2 * kWordsPerSlab);
  std::vector<uint32_t> captured(2 * kWordsPerSlab);
  uint64_t completion = 0;
  for (size_t generation = 0; generation < kGenerationCount; ++generation) {
    SCOPED_TRACE(generation);
    const uint64_t offset = kOrigin + (generation % 2) * 2 * kSlabSize;
    const std::array<loom_serve_memory_range_t, 2> ranges = {{
        {first, offset, kSlabSize},
        {second, offset, kSlabSize},
    }};
    std::array<iree_hal_transfer_operation_t, 2> uploads = {};
    std::array<iree_hal_transfer_operation_t, 2> downloads = {};
    for (size_t i = 0; i < handles.size(); ++i) {
      uploads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
      uploads[i].upload = {expected[generation].data() + i * kWordsPerSlab,
                           handles[i], offset, kSlabSize};
      downloads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
      downloads[i].download = {handles[i], offset,
                               observed.data() + i * kWordsPerSlab, kSlabSize};
    }
    bool admitted = false;
    IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(pool, ranges.size(),
                                                     ranges.data(), &admitted));
    ASSERT_TRUE(admitted);
    IREE_ASSERT_OK(loom_serve_execution_transfer(execution, uploads.size(),
                                                 uploads.data(), &completion));
    IREE_ASSERT_OK(loom_serve_execution_transfer(
        execution, downloads.size(), downloads.data(), &completion));
    IREE_ASSERT_OK(loom_serve_execution_wait(execution, completion));
    ASSERT_EQ(observed, expected[generation]);
    if (generation) {
      ASSERT_EQ(captured, expected[generation - 1]);
    }
    // Capture and immediately relinquish both physical slabs, as in a cold
    // session suspension. The next iteration reuses the retained VA roots.
    for (size_t i = 0; i < handles.size(); ++i) {
      downloads[i].download.target = captured.data() + i * kWordsPerSlab;
    }
    IREE_ASSERT_OK(loom_serve_execution_transfer(
        execution, downloads.size(), downloads.data(), &completion));
    IREE_ASSERT_OK(loom_serve_execution_wait(execution, completion));
    for (auto* buffer : {first, second}) {
      loom_serve_virtual_buffer_begin_trim(buffer);
      IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(buffer));
    }
  }
  ASSERT_EQ(captured, expected.back());
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, 0u);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).peak_bytes, 2 * kSlabSize);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).released_bytes,
            kGenerationCount * 2 * kSlabSize);
}

TEST_F(MemoryTest, FailedReclamationRetainsSharedFailureAndRemainingOwnership) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, kSlabSize, 256,
                                                    &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, kSlabSize, 256,
                                                    &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 256));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, 256));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x87654321);
  loom_serve_memory_admission_t admission;
  IREE_ASSERT_OK(
      loom_serve_memory_pool_prepare(pool, 4 * kSlabSize, &admission));
  EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_CAPACITY);
  IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool));

  // The dependency releases real native backing before reporting a subsequent
  // failure. The requested capacity would now fit, but cannot be admitted.
  loom_serve_memory_pool_set_reclaimer(
      pool, {first, [](void* self, uint64_t target_bytes) -> iree_status_t {
               auto* buffer = static_cast<loom_serve_virtual_buffer_t*>(self);
               loom_serve_virtual_buffer_begin_trim(buffer);
               IREE_RETURN_IF_ERROR(loom_serve_virtual_buffer_trim(buffer));
               return iree_make_status(IREE_STATUS_UNAVAILABLE,
                                       "reclamation failed after release");
             }});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      loom_serve_memory_pool_prepare(pool, 2 * kSlabSize, &admission));
  EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_CAPACITY);
  EXPECT_EQ(statistics[0].committed_bytes, 0u);
  EXPECT_EQ(statistics[1].committed_bytes, kSlabSize);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).reserved_bytes,
            2 * kSlabSize);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, kSlabSize);
  loom_serve_memory_pool_set_reclaimer(pool, {});
  for (int i = 0; i < 2; ++i) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                          loom_serve_memory_pool_query_status(pool));
  }
  // No new device work follows shared failure. TearDown drains accepted work
  // and releases the remaining ownership through the ordinary native path.
}

TEST_F(MemoryTest, DeviceTrimPreservesLiveWorkspaceAndMutableState) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, kSlabSize, 256,
                                                    &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 256));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0xABCDEF01);
  IREE_ASSERT_OK(
      loom_serve_execution_reserve_workspace(execution, kSlabSize, 256));
  const iree_hal_pool_reservation_request_t request = {
      .allocation_size = kSlabSize,
      .min_alignment = 256,
  };
  for (int iteration = 0; iteration < 2; ++iteration) {
    IREE_ASSERT_OK(
        loom_serve_execution_alloca(execution, &request, &workspace));
    RoundTrip(workspace, 0, 0x12345678);
    const auto before = loom_serve_device_memory_statistics(device);
    EXPECT_EQ(before.retained.committed_bytes, kSlabSize);
    EXPECT_EQ(before.workspace.bytes_committed, kSlabSize);
    EXPECT_EQ(before.workspace.reservation_count, 1u);
    IREE_ASSERT_OK(loom_serve_test_device_trim_and_drain(device, 0));
    const auto live = loom_serve_device_memory_statistics(device);
    EXPECT_EQ(live.retained.committed_bytes, kSlabSize);
    EXPECT_EQ(live.workspace.bytes_committed, kSlabSize);
    EXPECT_EQ(live.workspace.reservation_count, 1u);
    // A real consumer still reaches the allocation after an unreachable trim.
    CheckRetained(workspace, 0, 0x12345678);
    RoundTrip(workspace, 0, 0x87654321);
    CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0xABCDEF01);
    uint64_t completion = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_dealloca(execution, workspace, &completion));
    iree_hal_buffer_release(workspace);
    workspace = nullptr;
    IREE_ASSERT_OK(loom_serve_execution_drain(execution));
    IREE_ASSERT_OK(
        loom_serve_test_device_trim_and_drain(device, 2 * kSlabSize));
    EXPECT_EQ(
        loom_serve_device_memory_statistics(device).workspace.bytes_committed,
        kSlabSize);
    IREE_ASSERT_OK(loom_serve_test_device_trim_and_drain(device, kSlabSize));
    const auto idle = loom_serve_device_memory_statistics(device);
    EXPECT_EQ(idle.retained.committed_bytes, kSlabSize);
    EXPECT_EQ(idle.workspace.bytes_committed, 0u);
    EXPECT_EQ(idle.workspace.reservation_count, 0u);
    CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0xABCDEF01);
  }
}

TEST_F(MemoryTest, NativeAllocationPublicationDoesNotDoubleChargeAdmission) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, kSlabSize, 256,
                                                    &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, 2 * kSlabSize, 256,
                                                    &statistics[1], &second));
  std::array<loom_serve_memory_operation_t*, 2> operations = {};
  {
    // The source pool frees its acquisition capture after accounting native
    // backing and before publishing completion to the queue. Hold that real
    // allocator boundary while the owner admits an independent commitment.
    loom_serve::testing::ScopedNativeHold gate(native,
                                               IREE_ALLOCATOR_COMMAND_FREE);
    const loom_serve_memory_range_t first_range = {first, 0, kSlabSize};
    bool admitted = false;
    IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &first_range, nullptr, 0, &admitted, &operations[0]));
    ASSERT_TRUE(admitted);
    gate.WaitEntered();
    EXPECT_FALSE(loom_serve_memory_operation_is_complete(operations[0]));
    const auto pending = loom_serve_memory_pool_statistics(pool);
    EXPECT_EQ(pending.peak_bytes, kSlabSize);
    EXPECT_EQ(pending.pending_bytes, kSlabSize);
    EXPECT_EQ(pending.committed_bytes, 0u);
    EXPECT_EQ(pending.releasing_bytes, 0u);
    loom_serve_memory_admission_t admission;
    IREE_ASSERT_OK(
        loom_serve_memory_pool_prepare(pool, 2 * kSlabSize, &admission));
    EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    const loom_serve_memory_range_t second_range = {second, 0, 2 * kSlabSize};
    IREE_EXPECT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &second_range, nullptr, 0, &admitted, &operations[1]));
    EXPECT_TRUE(admitted);
    EXPECT_NE(operations[1], nullptr);
    EXPECT_EQ(loom_serve_memory_pool_statistics(pool).pending_bytes,
              3 * kSlabSize);
  }
  for (auto* operation : operations) {
    loom_serve_memory_operation_await(operation);
    IREE_EXPECT_OK(loom_serve_memory_operation_query_status(operation));
    loom_serve_memory_operation_free(operation);
  }
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  const auto completed = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(completed.pending_bytes, 0u);
  EXPECT_EQ(completed.committed_bytes, 3 * kSlabSize);
  EXPECT_EQ(completed.releasing_bytes, 0u);
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
  RoundTrip(loom_serve_virtual_buffer_handle(second), kSlabSize, 0x87654321);
}

TEST_F(MemoryTest, PendingCommitChargesBudgetAndLeavesPublishedStorageUsable) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, 3 * kSlabSize, 256,
                                                    &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, kSlabSize, 256,
                                                    &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
  iree_hal_semaphore_t* ready = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready));
  loom_serve_memory_operation_t* operation = nullptr;
  {
    loom_serve::testing::ScopedNativeHold gate(native,
                                               IREE_ALLOCATOR_COMMAND_CALLOC);
    const loom_serve_memory_range_t ranges[] = {
        {first, kSlabSize, 2 * kSlabSize},
        {first, kSlabSize + 256, kSlabSize},
    };
    bool admitted = false;
    IREE_EXPECT_OK(loom_serve_memory_pool_commit_async(
        pool, IREE_ARRAYSIZE(ranges), ranges, ready, 1, &admitted, &operation));
    EXPECT_TRUE(admitted);
    gate.WaitEntered();
    EXPECT_FALSE(loom_serve_memory_operation_is_complete(operation));
    const auto pending = loom_serve_memory_pool_statistics(pool);
    EXPECT_EQ(pending.committed_bytes, kSlabSize);
    EXPECT_EQ(pending.pending_bytes, 2 * kSlabSize);
    EXPECT_EQ(statistics[0].pending_bytes, 2 * kSlabSize);
    const loom_serve_memory_range_t other = {second, 0, kSlabSize};
    IREE_EXPECT_OK(
        loom_serve_memory_pool_try_commit(pool, 1, &other, &admitted));
    EXPECT_FALSE(admitted);
    // Native work already backed by a different range remains executable while
    // commitment is held. No source-file readiness or queue ordering is mocked.
    CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
  }
  loom_serve_memory_operation_await(operation);
  IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool));
  const auto completed = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(completed.committed_bytes, 3 * kSlabSize);
  EXPECT_EQ(completed.pending_bytes, 0u);
  RoundTrip(loom_serve_virtual_buffer_handle(first), 2 * kSlabSize, 0x87654321);
  loom_serve_memory_operation_free(operation);
  operation = nullptr;
  // Already-backed requests signal readiness without another cold job, even
  // when the shared executor is occupied. They consume no additional budget.
  {
    loom_serve::testing::ScopedNativeHold gate(native,
                                               IREE_ALLOCATOR_COMMAND_CALLOC);
    const loom_serve_memory_range_t range = {first, kSlabSize, kSlabSize};
    bool admitted = false;
    IREE_EXPECT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &range, ready, 2, &admitted, &operation));
    EXPECT_TRUE(admitted);
    EXPECT_EQ(operation, nullptr);
    uint64_t value = 0;
    IREE_EXPECT_OK(iree_hal_semaphore_query(ready, &value));
    EXPECT_EQ(value, 2u);
  }
  iree_hal_semaphore_release(ready);
}

TEST_F(MemoryTest, FailedReadinessDoesNotReleaseNativeOperationOwnership) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, 2 * kSlabSize, 256,
                                                    &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, kSlabSize, 256,
                                                    &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
  iree_hal_semaphore_t* ready = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready));
  loom_serve_memory_operation_t* operation = nullptr;
  {
    loom_serve::testing::ScopedNativeHold gate(native,
                                               IREE_ALLOCATOR_COMMAND_CALLOC);
    const loom_serve_memory_range_t range = {first, 0, kSlabSize};
    bool admitted = false;
    IREE_EXPECT_OK(loom_serve_memory_pool_commit_async(
        pool, 1, &range, ready, 1, &admitted, &operation));
    EXPECT_TRUE(admitted);
    gate.WaitEntered();
    // A failed timeline can be observed before an accepted native producer has
    // even started. It is not permission to destroy its reservation or job.
    iree_hal_semaphore_fail(
        ready, iree_make_status(IREE_STATUS_CANCELLED, "cancelled timeline"));
    uint64_t value = 0;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                          iree_hal_semaphore_query(ready, &value));
    EXPECT_FALSE(loom_serve_memory_operation_is_complete(operation));
    EXPECT_EQ(loom_serve_memory_pool_statistics(pool).pending_bytes, kSlabSize);
    CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
  }
  loom_serve_memory_operation_await(operation);
  EXPECT_TRUE(loom_serve_memory_operation_is_complete(operation));
  IREE_EXPECT_OK(loom_serve_memory_operation_query_status(operation));
  IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool));
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        iree_hal_semaphore_query(ready, &value));
  const auto memory = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(memory.committed_bytes, 2 * kSlabSize);
  EXPECT_EQ(memory.pending_bytes, 0u);
  loom_serve_memory_operation_free(operation);
  iree_hal_semaphore_release(ready);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, kSlabSize, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(first), kSlabSize, 0x87654321);
  CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
}

TEST_F(MemoryTest, NativeAllocationFailurePreservesWarmWorkAndAllowsRetry) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, kSlabSize, 256,
                                                    &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, kSlabSize, 256,
                                                    &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);

  native.FailNextAllocation(IREE_ALLOCATOR_COMMAND_CALLOC,
                            /*native_only=*/true);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_virtual_buffer_commit(second, 0, kSlabSize));
  IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool));
  EXPECT_EQ(statistics[1].pending_bytes, 0u);
  EXPECT_EQ(statistics[1].committed_bytes, 0u);
  CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);

  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x87654321);
  CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0x12345678);
}

TEST_F(MemoryTest, FailedMappingKeepsSourcesUntilCheckedDetach) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, 2 * kSlabSize, 256,
                                                    &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(pool, kSlabSize, 256,
                                                    &statistics[1], &second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, 0, kSlabSize));
  RoundTrip(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);

  // The native virtual target grows its mapping inventory during BIND, after
  // physical acquisition. Failing that dependency leaves owned sources but
  // cannot claim that the complete batch is installed.
  native.FailNextAllocation(IREE_ALLOCATOR_COMMAND_REALLOC);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_virtual_buffer_commit(first, 0, 2 * kSlabSize));
  IREE_EXPECT_OK(loom_serve_memory_pool_query_status(pool));
  EXPECT_EQ(statistics[0].pending_bytes, 0u);
  EXPECT_EQ(statistics[0].committed_bytes, 2 * kSlabSize);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_virtual_buffer_commit(first, 0, kSlabSize));
  CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);

  IREE_ASSERT_OK(loom_serve_virtual_buffer_free(first));
  first = nullptr;
  IREE_EXPECT_OK(loom_serve_memory_pool_await(pool));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, kSlabSize);
  CheckRetained(loom_serve_virtual_buffer_handle(second), 0, 0x12345678);
}

}  // namespace
