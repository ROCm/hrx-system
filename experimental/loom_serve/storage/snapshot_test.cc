// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/storage/snapshot.h"

#include <array>

#include "experimental/loom_serve/runtime/device.h"
#include "experimental/loom_serve/storage/memory.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class SnapshotTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const loom_serve_device_options_t options = {
        .uri = IREE_SV("amdgpu"),
        .backing = LOOM_SERVE_DEVICE_BACKING_ELASTIC,
        .slab_size = kSlabSize,
        .memory_limit = 3 * kSlabSize};
    IREE_ASSERT_OK(
        loom_serve_device_create(&options, &device, iree_allocator_system()));
    execution = loom_serve_device_execution(device);
    pool = loom_serve_device_memory_pool(device);
  }

  void TearDown() override {
    IREE_EXPECT_OK(loom_serve_execution_drain(execution));
    loom_serve_snapshot_destroy(snapshot);
    for (auto* reservation : reservations) {
      IREE_EXPECT_OK(loom_serve_virtual_buffer_destroy(reservation));
    }
    IREE_EXPECT_OK(loom_serve_device_destroy(device));
  }

  // Capacity granularity, independent of the packed transfer lengths.
  static constexpr uint64_t kSlabSize = 2 * 1024 * 1024;
  // Source contents deliberately use wide addresses.
  static constexpr uint64_t kHighOffset = (uint64_t{1} << 32) + 512;
  // More ranges than one native transfer batch can hold.
  static constexpr size_t kRangeCount = 70;
  // Shared device owner enclosing all reservation/transfer lifetimes.
  loom_serve_device_t* device = nullptr;
  // Borrowed ordered submission owner.
  loom_serve_execution_t* execution = nullptr;
  // Physical budget enclosing both independent roots.
  loom_serve_memory_pool_t* pool = nullptr;
  // Accounting survives destruction and replacement of the roots.
  loom_serve_memory_statistics_t statistics = {};
  // Source roots are destroyed before replacement roots are created here.
  std::array<loom_serve_virtual_buffer_t*, 2> reservations = {};
  // Host image remains valid without any device root from capture.
  loom_serve_snapshot_t* snapshot = nullptr;
};

TEST_F(SnapshotTest, LogicalRangesSurviveSourceDestructionAndFreshPlacement) {
  std::array<iree_hal_buffer_t*, 2> roots = {};
  for (size_t i = 0; i < roots.size(); ++i) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_create(
        pool, kHighOffset + kSlabSize, 256, &statistics, &reservations[i]));
    roots[i] = loom_serve_virtual_buffer_handle(reservations[i]);
    IREE_ASSERT_OK(
        loom_serve_virtual_buffer_commit(reservations[i], kHighOffset, 32768));
  }
  std::array<loom_serve_snapshot_range_t, kRangeCount> ranges;
  std::array<uint32_t, kRangeCount> patterns;
  std::array<iree_hal_transfer_operation_t, kRangeCount> fills = {};
  for (size_t i = 0; i < ranges.size(); ++i) {
    // Logical order intentionally opposes physical offset order.
    ranges[i] = {i % 2, kHighOffset + (kRangeCount - i - 1) * 256, 16};
    patterns[i] = 0xABC00000u + static_cast<uint32_t>(i * 17);
    fills[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
    fills[i].fill = {roots[ranges[i].buffer_index], ranges[i].offset,
                     ranges[i].length, &patterns[i], sizeof(patterns[i])};
  }
  uint64_t completion = 0;
  IREE_ASSERT_OK(loom_serve_execution_transfer(execution, fills.size(),
                                               fills.data(), &completion));
  IREE_ASSERT_OK(loom_serve_snapshot_capture(
      execution, roots.size(), roots.data(), ranges.size(), ranges.data(),
      &snapshot, iree_allocator_system()));
  EXPECT_EQ(loom_serve_snapshot_size(snapshot), ranges.size() * 16);
  for (auto*& reservation : reservations) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_destroy(reservation));
    reservation = nullptr;
  }
  EXPECT_EQ(statistics.committed_bytes, 0u);
  EXPECT_EQ(statistics.reserved_bytes, 0u);
  for (size_t i = 0; i < roots.size(); ++i) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_create(
        pool, kSlabSize, 256, &statistics, &reservations[i]));
    roots[i] = loom_serve_virtual_buffer_handle(reservations[i]);
    IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(reservations[i], 0, 32768));
  }
  for (size_t i = 0; i < ranges.size(); ++i) {
    ranges[i].buffer_index = 1 - ranges[i].buffer_index;
    ranges[i].offset = 1024 + i * 128;
  }
  IREE_ASSERT_OK(loom_serve_snapshot_restore(snapshot, execution, roots.size(),
                                             roots.data(), ranges.size(),
                                             ranges.data()));
  std::array<std::array<uint32_t, 4>, kRangeCount> observed = {};
  std::array<iree_hal_transfer_operation_t, kRangeCount> downloads = {};
  for (size_t i = 0; i < ranges.size(); ++i) {
    downloads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    downloads[i].download = {roots[ranges[i].buffer_index], ranges[i].offset,
                             observed[i].data(), ranges[i].length};
  }
  IREE_ASSERT_OK(loom_serve_execution_transfer(execution, downloads.size(),
                                               downloads.data(), &completion));
  IREE_ASSERT_OK(loom_serve_execution_wait(execution, completion));
  for (size_t i = 0; i < observed.size(); ++i) {
    for (const auto value : observed[i]) {
      EXPECT_EQ(value, patterns[i]);
    }
  }
}

}  // namespace
