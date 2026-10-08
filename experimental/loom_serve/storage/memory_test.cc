// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/storage/memory.h"

#include <array>
#include <vector>

#include "experimental/loom_serve/runtime/device.h"
#include "experimental/loom_serve/runtime/execution.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class MemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const loom_serve_device_options_t options = {
        .uri = IREE_SV("amdgpu"),
        .backing = LOOM_SERVE_DEVICE_BACKING_ELASTIC,
        .slab_size = kSlabSize,
        .memory_limit = 3 * kSlabSize};
    loom_serve_host_t* host = nullptr;
    IREE_ASSERT_OK(loom_serve_host_create(1, &host, iree_allocator_system()));
    auto status = loom_serve_device_create(host, &options, &device,
                                           iree_allocator_system());
    loom_serve_host_release(host);
    IREE_ASSERT_OK(status);
    pool = loom_serve_device_memory_pool(device);
    IREE_ASSERT_OK(loom_serve_device_create_execution(device, &execution));
  }

  void TearDown() override {
    iree_hal_buffer_release(workspace);
    IREE_ASSERT_OK(loom_serve_execution_drain(execution));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_destroy(first));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_destroy(second));
    const auto statistics = loom_serve_memory_pool_statistics(pool);
    EXPECT_EQ(statistics.committed_bytes, 0u);
    EXPECT_EQ(statistics.reserved_bytes, 0u);
    loom_serve_execution_release(execution);
    IREE_EXPECT_OK(loom_serve_device_destroy(device));
  }

  void RoundTrip(iree_hal_buffer_t* buffer, uint64_t offset, uint32_t value) {
    std::array<uint32_t, 64> observed = {};
    const iree_hal_transfer_operation_t operations[] = {
        {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
         .fill = {.target_buffer = buffer,
                  .target_offset = offset,
                  .length = sizeof(observed),
                  .pattern = &value,
                  .pattern_length = sizeof(value)}},
        {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
         .download = {.source_buffer = buffer,
                      .source_offset = offset,
                      .target = observed.data(),
                      .length = sizeof(observed)}},
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
        .download = {.source_buffer = buffer,
                     .source_offset = offset,
                     .target = observed.data(),
                     .length = sizeof(observed)}};
    uint64_t completion = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_transfer(execution, 1, &download, &completion));
    IREE_ASSERT_OK(loom_serve_execution_wait(execution, completion));
    for (const auto word : observed) {
      EXPECT_EQ(word, value);
    }
  }

  // Coarse physical slabs, independent of logical KV block size.
  static constexpr uint64_t kSlabSize = 2 * 1024 * 1024;
  // Offset deliberately crossing the 32-bit byte-address boundary.
  static constexpr uint64_t kHighOffset = (uint64_t{1} << 32) + 512;
  // Device owner enclosing all physical and virtual storage.
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

TEST_F(MemoryTest, SparseGrowthTrimRegrowthAndSharedAdmission) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_create(pool, kHighOffset + kSlabSize,
                                                  256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_create(pool, 2 * kSlabSize, 256,
                                                  &statistics[1], &second));
  auto* const identity = loom_serve_virtual_buffer_handle(first);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_device_destroy(device));
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
  IREE_ASSERT_OK(loom_serve_virtual_buffer_create(pool, 3 * kSlabSize, 256,
                                                  &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_create(pool, 2 * kSlabSize, 256,
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
    IREE_ASSERT_OK(loom_serve_virtual_buffer_create(
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

TEST_F(MemoryTest, DeviceTrimPreservesLiveWorkspaceAndMutableState) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_create(pool, kSlabSize, 256,
                                                  &statistics[0], &first));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 256));
  RoundTrip(loom_serve_virtual_buffer_handle(first), 0, 0xABCDEF01);
  IREE_ASSERT_OK(
      loom_serve_execution_reserve_workspace(execution, kSlabSize, 256));
  const iree_hal_pool_reservation_request_t request = {
      .params = {.usage = IREE_HAL_BUFFER_USAGE_STORAGE |
                          IREE_HAL_BUFFER_USAGE_TRANSFER,
                 .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                 .min_alignment = 256},
      .allocation_size = kSlabSize};
  for (int iteration = 0; iteration < 2; ++iteration) {
    IREE_ASSERT_OK(
        loom_serve_execution_alloca(execution, &request, &workspace));
    RoundTrip(workspace, 0, 0x12345678);
    const auto before = loom_serve_device_memory_statistics(device);
    EXPECT_EQ(before.retained.committed_bytes, kSlabSize);
    EXPECT_EQ(before.workspace.bytes_committed, kSlabSize);
    EXPECT_EQ(before.workspace.reservation_count, 1u);
    IREE_ASSERT_OK(loom_serve_device_trim(device, 0));
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
    IREE_ASSERT_OK(loom_serve_device_trim(device, 2 * kSlabSize));
    EXPECT_EQ(
        loom_serve_device_memory_statistics(device).workspace.bytes_committed,
        kSlabSize);
    IREE_ASSERT_OK(loom_serve_device_trim(device, kSlabSize));
    const auto idle = loom_serve_device_memory_statistics(device);
    EXPECT_EQ(idle.retained.committed_bytes, kSlabSize);
    EXPECT_EQ(idle.workspace.bytes_committed, 0u);
    EXPECT_EQ(idle.workspace.reservation_count, 0u);
    CheckRetained(loom_serve_virtual_buffer_handle(first), 0, 0xABCDEF01);
  }
}

}  // namespace
