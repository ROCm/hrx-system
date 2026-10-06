// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/storage/relocation.h"

#include <array>
#include <vector>

#include "experimental/loom_serve/runtime/device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class RelocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(loom_serve_device_create(IREE_SV("amdgpu"),
                                            iree_allocator_system(), &device));
    IREE_ASSERT_OK(loom_serve_memory_pool_create(
        iree_hal_device_allocator(loom_serve_device_handle(device)),
        IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 2 * 1024 * 1024, 0, &pool,
        iree_allocator_system()));
    execution = loom_serve_device_execution(device);
  }

  void TearDown() override {
    IREE_ASSERT_OK(loom_serve_execution_drain(execution));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_destroy(buffer));
    loom_serve_memory_pool_destroy(pool);
    loom_serve_device_destroy(device);
  }

  // Device owner enclosing all queue work and memory.
  loom_serve_device_t* device = nullptr;
  // Borrowed work/feedback retirement domain.
  loom_serve_execution_t* execution = nullptr;
  // Physical owner of the sparse plane reservation.
  loom_serve_memory_pool_t* pool = nullptr;
  // Source and destination allocation with stable identity.
  loom_serve_virtual_buffer_t* buffer = nullptr;
};

TEST_F(RelocationTest, RepeatedPlanesCrossBatchAndAddressBoundaries) {
  // The first destination straddles a physical slab boundary, and later
  // planes cross 4 GiB. More than 64 copies exercise queue-batch retirement.
  const loom_serve_block_region_t region = {2 * 1024 * 1024 - 256, 3,
                                            (uint64_t{1} << 32) + 512, 4096};
  constexpr uint32_t kLiveBlocks = 35;
  constexpr uint32_t kBlockCount = kLiveBlocks * 2;
  IREE_ASSERT_OK(loom_serve_virtual_buffer_create(
      pool,
      region.origin + 2 * region.stride + kBlockCount * region.block_bytes, 256,
      &buffer));
  auto* const handle = loom_serve_virtual_buffer_handle(buffer);
  uint64_t completion = 0;
  for (uint32_t plane = 0; plane < region.count; ++plane) {
    for (uint32_t block = kLiveBlocks; block < kBlockCount; ++block) {
      const uint64_t offset =
          region.origin + plane * region.stride + block * region.block_bytes;
      IREE_ASSERT_OK(
          loom_serve_virtual_buffer_commit(buffer, offset, region.block_bytes));
      const uint32_t value = plane * 1000 + block;
      const iree_hal_transfer_operation_t fill = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
          .fill = {.target_buffer = handle,
                   .target_offset = offset,
                   .length = region.block_bytes,
                   .pattern = &value,
                   .pattern_length = sizeof(value)}};
      IREE_ASSERT_OK(
          loom_serve_execution_transfer(execution, 1, &fill, &completion));
    }
  }
  IREE_ASSERT_OK(loom_serve_execution_drain(execution));
  std::array<uint32_t, kBlockCount> destinations;
  for (uint32_t block = 0; block < kBlockCount; ++block) {
    destinations[block] =
        block < kLiveBlocks ? UINT32_MAX : block - kLiveBlocks;
  }
  uint64_t copied_bytes = 0;
  IREE_ASSERT_OK(
      loom_serve_block_region_relocate(execution, buffer, &region, kBlockCount,
                                       destinations.data(), &copied_bytes));
  IREE_ASSERT_OK(loom_serve_execution_drain(execution));
  EXPECT_EQ(copied_bytes, region.count * kLiveBlocks * region.block_bytes);
  EXPECT_EQ(loom_serve_virtual_buffer_handle(buffer), handle);
  loom_serve_virtual_buffer_begin_trim(buffer);
  for (uint32_t plane = 0; plane < region.count; ++plane) {
    loom_serve_virtual_buffer_keep(buffer,
                                   region.origin + plane * region.stride,
                                   kLiveBlocks * region.block_bytes);
  }
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(buffer));
  std::vector<uint32_t> observed(kLiveBlocks * region.block_bytes /
                                 sizeof(uint32_t));
  for (uint32_t plane = 0; plane < region.count; ++plane) {
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download = {.source_buffer = handle,
                     .source_offset = region.origin + plane * region.stride,
                     .target = observed.data(),
                     .length = kLiveBlocks * region.block_bytes}};
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
