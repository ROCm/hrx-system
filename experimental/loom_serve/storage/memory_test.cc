// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/storage/memory.h"

#include <array>

#include "experimental/loom_serve/runtime/device.h"
#include "experimental/loom_serve/runtime/execution.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class MemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(loom_serve_device_create(IREE_SV("amdgpu"),
                                            iree_allocator_system(), &device));
    IREE_ASSERT_OK(loom_serve_memory_pool_create(
        iree_hal_device_allocator(loom_serve_device_handle(device)),
        IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, kSlabSize, 3 * kSlabSize, &pool,
        iree_allocator_system()));
  }

  void TearDown() override {
    IREE_ASSERT_OK(
        loom_serve_execution_drain(loom_serve_device_execution(device)));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_destroy(first));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_destroy(second));
    const auto statistics = loom_serve_memory_pool_statistics(pool);
    EXPECT_EQ(statistics.committed_bytes, 0u);
    EXPECT_EQ(statistics.reserved_bytes, 0u);
    loom_serve_memory_pool_destroy(pool);
    loom_serve_device_destroy(device);
  }

  void RoundTrip(loom_serve_virtual_buffer_t* buffer, uint64_t offset,
                 uint32_t value) {
    std::array<uint32_t, 64> observed = {};
    const iree_hal_transfer_operation_t operations[] = {
        {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
         .fill = {.target_buffer = loom_serve_virtual_buffer_handle(buffer),
                  .target_offset = offset,
                  .length = sizeof(observed),
                  .pattern = &value,
                  .pattern_length = sizeof(value)}},
        {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
         .download = {.source_buffer = loom_serve_virtual_buffer_handle(buffer),
                      .source_offset = offset,
                      .target = observed.data(),
                      .length = sizeof(observed)}},
    };
    uint64_t completion = 0;
    IREE_ASSERT_OK(loom_serve_execution_transfer(
        loom_serve_device_execution(device), IREE_ARRAYSIZE(operations),
        operations, &completion));
    IREE_ASSERT_OK(loom_serve_execution_wait(
        loom_serve_device_execution(device), completion));
    for (const auto word : observed) {
      EXPECT_EQ(word, value);
    }
  }

  void CheckRetained(loom_serve_virtual_buffer_t* buffer, uint64_t offset,
                     uint32_t value) {
    std::array<uint32_t, 64> observed = {};
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download = {.source_buffer = loom_serve_virtual_buffer_handle(buffer),
                     .source_offset = offset,
                     .target = observed.data(),
                     .length = sizeof(observed)}};
    uint64_t completion = 0;
    IREE_ASSERT_OK(loom_serve_execution_transfer(
        loom_serve_device_execution(device), 1, &download, &completion));
    IREE_ASSERT_OK(loom_serve_execution_wait(
        loom_serve_device_execution(device), completion));
    for (const auto word : observed) {
      EXPECT_EQ(word, value);
    }
  }

  // Coarse physical slabs, independent of logical KV block size.
  static constexpr uint64_t kSlabSize = 2 * 1024 * 1024;
  // Offset deliberately crossing the 32-bit byte-address boundary.
  static constexpr uint64_t kHighOffset = (uint64_t{1} << 32) + 512;
  // Device and timeline owner enclosing all physical and virtual storage.
  loom_serve_device_t* device = nullptr;
  // One physical budget shared by the independent reservations.
  loom_serve_memory_pool_t* pool = nullptr;
  // First model-like reservation, sparsely backed beyond four GiB.
  loom_serve_virtual_buffer_t* first = nullptr;
  // Independent reservation competing for the same physical budget.
  loom_serve_virtual_buffer_t* second = nullptr;
};

TEST_F(MemoryTest, SparseGrowthTrimRegrowthAndSharedAdmission) {
  IREE_ASSERT_OK(loom_serve_virtual_buffer_create(pool, kHighOffset + kSlabSize,
                                                  256, &first));
  IREE_ASSERT_OK(
      loom_serve_virtual_buffer_create(pool, 2 * kSlabSize, 256, &second));
  auto* const identity = loom_serve_virtual_buffer_handle(first);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes, 0u);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, kHighOffset, 256));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 512));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 256, 256));
  RoundTrip(first, kHighOffset, 0x12345678);
  RoundTrip(first, 0, 0xABCDEF01);
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
  CheckRetained(first, kHighOffset, 0x12345678);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(second, kSlabSize, 256));
  RoundTrip(second, kSlabSize, 0xDEADBEEF);
  loom_serve_virtual_buffer_begin_trim(second);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(first, 0, 512));
  EXPECT_EQ(loom_serve_virtual_buffer_handle(first), identity);
  RoundTrip(first, 0, 0xCAFEBABE);
  CheckRetained(first, kHighOffset, 0x12345678);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).peak_bytes, 3 * kSlabSize);
}

}  // namespace
