// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/storage/relocation.h"

#include <array>
#include <memory>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/execution/retirement.h"
#include "loom/serve/runtime/storage/memory.h"
#include "loom/serve/testing/device.h"
#include "loom/serve/testing/native_allocator.h"

namespace {

class RelocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_serve_retirement_initialize(&retirement);
    loom_serve_host_t* host = nullptr;
    IREE_ASSERT_OK(
        loom_serve_host_create(1, nullptr, iree_allocator_system(), &host));
    iree_hal_driver_t* driver = nullptr;
    IREE_ASSERT_OK(loom_serve_test_driver_acquire(host, &driver));
    const loom_serve_device_options_t options = {
        .driver = driver,
        .uri = loom_serve_test_device()->device_uri,
    };
    const loom_serve_pool_options_t pool_options = {
        .elastic = true,
        .weight_slab_size = 2 * 1024 * 1024,
        .state_slab_size = 2 * 1024 * 1024,
    };
    auto status = loom_serve_device_group_allocate(host, 1, &options,
                                                   native.allocator(), &group);
    iree_hal_driver_release(driver);
    loom_serve_host_release(host);
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(loom_serve_device_group_create_pool(
        group, {IREE_SV("pool"), 1}, &pool_options, &storage_owner));
    device = loom_serve_device_group_at(group, 0);
    pool = loom_serve_pool_memory(storage_owner);
    IREE_ASSERT_OK(native.Observe(storage_owner));
    IREE_ASSERT_OK(
        loom_serve_device_create_execution(device, storage_owner, &execution));
  }

  void TearDown() override {
    if (execution) {
      IREE_ASSERT_OK(loom_serve_execution_drain(execution));
    }
    iree_hal_buffer_release(view);
    loom_serve_retirement_await(&retirement);
    loom_serve_retirement_deinitialize(&retirement);
    IREE_ASSERT_OK(loom_serve_virtual_buffer_free(buffer));
    loom_serve_execution_release(execution);
    IREE_EXPECT_OK(loom_serve_device_group_free(group));
  }

  // Allocator dependency outliving the real native allocation worker.
  loom_serve::testing::NativeAllocator native;
  // Application group enclosing all device resources.
  loom_serve_device_group_t* group = nullptr;
  // Borrowed device record within the application group.
  loom_serve_device_t* device = nullptr;
  // Independent storage owner selected for every reservation.
  loom_serve_pool_t* storage_owner = nullptr;
  // Owned model work/feedback retirement domain.
  loom_serve_execution_t* execution = nullptr;
  // Physical owner of the sparse plane reservation.
  loom_serve_memory_pool_t* pool = nullptr;
  // Mutable state accounting, independent of other pool consumers.
  loom_serve_memory_statistics_t statistics = {};
  // Source and destination allocation with stable identity.
  loom_serve_virtual_buffer_t* buffer = nullptr;
  // Whole exported view capturing every relocation and transfer reference.
  iree_hal_buffer_t* view = nullptr;
  // Actual view retirement, independent of success or failure readiness.
  loom_serve_retirement_t retirement = {};
};

TEST_F(RelocationTest, RepeatedPlanesCrossBatchAndAddressBoundaries) {
  // The first destination straddles a physical slab boundary, and later
  // planes cross 4 GiB. More than 64 copies exercise queue-batch retirement.
  constexpr uint32_t kPlaneCount = 3;
  const loom_serve_block_region_t region = {2 * 1024 * 1024 - 256, kPlaneCount,
                                            (uint64_t{1} << 32) + 512, 4096};
  constexpr uint32_t kLiveBlocks = 35;
  constexpr uint32_t kBlockCount = kLiveBlocks * 2;
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_pool_state_geometry(storage_owner),
      region.origin + 2 * region.stride + kBlockCount * region.block_bytes, 256,
      &statistics, &buffer));
  auto* const handle = loom_serve_virtual_buffer_handle(buffer);
  view = handle;
  iree_hal_buffer_retain(view);
  IREE_ASSERT_OK(
      loom_serve_retirement_track(&retirement, &view, iree_allocator_system()));
  std::array<loom_serve_memory_range_t, kPlaneCount> commitments;
  std::array<iree_hal_transfer_operation_t, kPlaneCount * kLiveBlocks> fills;
  std::array<uint32_t, kPlaneCount * kLiveBlocks> fill_values;
  size_t initialization_count = 0;
  uint64_t completion = 0;
  for (uint32_t plane = 0; plane < region.count; ++plane) {
    commitments[plane] = {
        buffer,
        region.origin + plane * region.stride +
            kLiveBlocks * region.block_bytes,
        kLiveBlocks * region.block_bytes,
    };
    for (uint32_t block = kLiveBlocks; block < kBlockCount; ++block) {
      const uint64_t offset =
          region.origin + plane * region.stride + block * region.block_bytes;
      const size_t initialization_index = initialization_count++;
      fill_values[initialization_index] = plane * 1000 + block;
      fills[initialization_index] = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
          .fill =
              {
                  .target_buffer = view,
                  .target_offset = offset,
                  .length = region.block_bytes,
                  .pattern = &fill_values[initialization_index],
                  .pattern_length = sizeof(fill_values[initialization_index]),
              },
      };
    }
  }
  bool admitted = false;
  IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(
      pool, commitments.size(), commitments.data(), &admitted));
  ASSERT_TRUE(admitted);
  IREE_ASSERT_OK(loom_serve_execution_transfer(execution, initialization_count,
                                               fills.data(), &completion));
  IREE_ASSERT_OK(loom_serve_execution_drain(execution));
  std::array<uint32_t, kBlockCount> destinations;
  for (uint32_t block = 0; block < kBlockCount; ++block) {
    destinations[block] =
        block < kLiveBlocks ? UINT32_MAX : block - kLiveBlocks;
  }
  uint64_t copied_bytes = 0;
  std::array<loom_serve_memory_range_t, kPlaneCount> ranges;
  for (uint32_t plane = 0; plane < region.count; ++plane) {
    ranges[plane] = {buffer, region.origin + plane * region.stride,
                     kLiveBlocks * region.block_bytes};
  }
  // The initial slices share a larger physical parent. Occupy its remaining
  // capacity so the destination commitment below must cross the native
  // allocation boundary exercised by the hold.
  const auto before_filler = loom_serve_memory_pool_statistics(pool);
  ASSERT_GT(before_filler.cached_bytes, 0u);
  loom_serve_memory_statistics_t filler_statistics = {0};
  loom_serve_virtual_buffer_t* filler = nullptr;
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool, loom_serve_pool_state_geometry(storage_owner),
      before_filler.cached_bytes, 256, &filler_statistics, &filler));
  const auto free_filler = [](loom_serve_virtual_buffer_t* buffer) {
    IREE_EXPECT_OK(loom_serve_virtual_buffer_free(buffer));
  };
  std::unique_ptr<loom_serve_virtual_buffer_t, decltype(free_filler)>
      filler_owner(filler, free_filler);
  IREE_ASSERT_OK(
      loom_serve_virtual_buffer_commit(filler, 0, before_filler.cached_bytes));
  const auto after_filler = loom_serve_memory_pool_statistics(pool);
  EXPECT_EQ(after_filler.committed_bytes, before_filler.committed_bytes);
  EXPECT_EQ(after_filler.cached_bytes, 0u);
  iree_hal_semaphore_t* ready = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready));
  loom_serve_memory_operation_t* commitment = nullptr;
  {
    loom_serve::testing::ScopedNativeHold gate(device);
    loom_serve_memory_admission_t commit_admission;
    IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
        pool, ranges.size(), ranges.data(), ready, 1, &commit_admission,
        &commitment));
    ASSERT_EQ(commit_admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    ASSERT_NE(commitment, nullptr);
    gate.WaitEntered();
    uint64_t value = 1;
    IREE_ASSERT_OK(
        loom_serve_execution_barrier(execution, {1, &ready, &value}));
    IREE_ASSERT_OK(
        loom_serve_block_region_relocate(execution, view, &region, kBlockCount,
                                         destinations.data(), &copied_bytes));
    EXPECT_FALSE(loom_serve_memory_operation_is_complete(commitment));
  }
  IREE_ASSERT_OK(loom_serve_execution_drain(execution));
  EXPECT_TRUE(loom_serve_memory_operation_is_complete(commitment));
  loom_serve_memory_operation_free(commitment);
  iree_hal_semaphore_release(ready);
  EXPECT_EQ(copied_bytes, region.count * kLiveBlocks * region.block_bytes);
  EXPECT_EQ(loom_serve_virtual_buffer_handle(buffer), handle);
  loom_serve_virtual_buffer_begin_trim(buffer);
  for (uint32_t plane = 0; plane < region.count; ++plane) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_keep(
        buffer, region.origin + plane * region.stride,
        kLiveBlocks * region.block_bytes, nullptr));
  }
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(buffer));
  std::vector<uint32_t> observed(kLiveBlocks * region.block_bytes /
                                 sizeof(uint32_t));
  for (uint32_t plane = 0; plane < region.count; ++plane) {
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = view,
                .source_offset = region.origin + plane * region.stride,
                .target = observed.data(),
                .length = kLiveBlocks * region.block_bytes,
            },
    };
    IREE_ASSERT_OK(
        loom_serve_execution_transfer(execution, 1, &download, &completion));
    IREE_ASSERT_OK(loom_serve_execution_wait(execution, completion));
    for (size_t word = 0; word < observed.size(); ++word) {
      EXPECT_EQ(observed[word],
                plane * 1000 + kLiveBlocks +
                    word * sizeof(uint32_t) / region.block_bytes);
    }
  }
}

}  // namespace
