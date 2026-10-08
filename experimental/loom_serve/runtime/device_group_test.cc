// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "experimental/loom_serve/runtime/device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class DeviceGroupTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(loom_serve_host_create(1, &host_, iree_allocator_system()));
  }

  void TearDown() override {
    for (auto* execution : executions_) {
      if (execution) {
        IREE_EXPECT_OK(loom_serve_execution_drain(execution));
      }
      loom_serve_execution_release(execution);
    }
    for (auto* reservation : reservations_) {
      IREE_EXPECT_OK(loom_serve_virtual_buffer_destroy(reservation));
    }
    IREE_EXPECT_OK(loom_serve_device_group_destroy(group_));
    loom_serve_host_release(host_);
  }

  // Physical granularity, independent of each device's admission budget.
  static constexpr uint64_t kSlabBytes = 2 * 1024 * 1024;
  // Shared workers and asynchronous I/O outliving all device resources.
  loom_serve_host_t* host_ = nullptr;
  // One application group, including nullable partial construction.
  loom_serve_device_group_t* group_ = nullptr;
  // Independent model-like timelines, one per device record.
  std::array<loom_serve_execution_t*, 2> executions_ = {};
  // Independent sparse roots that prevent premature group destruction.
  std::array<loom_serve_virtual_buffer_t*, 2> reservations_ = {};
  // Accounting borrowed through each reservation's final destruction.
  std::array<loom_serve_memory_statistics_t, 2> statistics_ = {};
  // Download destinations outliving accepted transfers on either device.
  std::array<uint32_t, 2> output_ = {};
};

TEST_F(DeviceGroupTest, LaterDeviceFailureReleasesEarlierHandle) {
  const loom_serve_device_options_t options[] = {
      {.uri = IREE_SV("amdgpu")},
      {.uri = IREE_SV("unregistered-serving-device")}};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_serve_device_group_create(host_, IREE_ARRAYSIZE(options), options,
                                     &group_, iree_allocator_system()));
  EXPECT_EQ(group_, nullptr);
}

TEST_F(DeviceGroupTest, LaterPoolFailureReleasesSealedGroupAndEarlierPool) {
  const loom_serve_device_options_t options[] = {
      {.uri = IREE_SV("amdgpu"),
       .backing = LOOM_SERVE_DEVICE_BACKING_ELASTIC,
       .slab_size = kSlabBytes},
      {.uri = IREE_SV("amdgpu"),
       .backing = LOOM_SERVE_DEVICE_BACKING_ELASTIC,
       .slab_size = UINT64_MAX}};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_device_group_create(host_, IREE_ARRAYSIZE(options), options,
                                     &group_, iree_allocator_system()));
  EXPECT_EQ(group_, nullptr);
}

TEST_F(DeviceGroupTest, SeparateBudgetsAndWholeGroupLifetime) {
  const loom_serve_device_options_t options[] = {
      {.uri = IREE_SV("amdgpu"),
       .backing = LOOM_SERVE_DEVICE_BACKING_ELASTIC,
       .slab_size = kSlabBytes,
       .memory_limit = kSlabBytes},
      {.uri = IREE_SV("amdgpu"),
       .backing = LOOM_SERVE_DEVICE_BACKING_ELASTIC,
       .slab_size = kSlabBytes,
       .memory_limit = 2 * kSlabBytes}};
  IREE_ASSERT_OK(loom_serve_device_group_create(host_, IREE_ARRAYSIZE(options),
                                                options, &group_,
                                                iree_allocator_system()));
  auto* first = loom_serve_device_group_at(group_, 0);
  auto* second = loom_serve_device_group_at(group_, 1);
  EXPECT_EQ(loom_serve_device_group(first), loom_serve_device_group(second));
  EXPECT_NE(loom_serve_device_memory_pool(first),
            loom_serve_device_memory_pool(second));
  EXPECT_NE(loom_serve_device_residency_cache(first),
            loom_serve_device_residency_cache(second));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_create(
      loom_serve_device_memory_pool(second), 2 * kSlabBytes, 256,
      &statistics_[1], &reservations_[1]));
  // Only the last record has a reservation. Rejection must preserve the first
  // record's resources too; it can still admit work after the failed teardown.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_device_group_destroy(group_));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_create(
      loom_serve_device_memory_pool(first), 2 * kSlabBytes, 256,
      &statistics_[0], &reservations_[0]));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(reservations_[0], 0, 4));
  IREE_ASSERT_OK(
      loom_serve_virtual_buffer_commit(reservations_[1], 0, 2 * kSlabBytes));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_virtual_buffer_commit(reservations_[0], kSlabBytes, 4));
  for (iree_host_size_t i = 0; i < 2; ++i) {
    auto* device = loom_serve_device_group_at(group_, i);
    IREE_ASSERT_OK(loom_serve_device_create_execution(device, &executions_[i]));
    auto* buffer = loom_serve_virtual_buffer_handle(reservations_[i]);
    const uint32_t pattern = 0x12345678u + i;
    const iree_hal_transfer_operation_t fill = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
        .fill = {.target_buffer = buffer,
                 .length = sizeof(pattern),
                 .pattern = &pattern,
                 .pattern_length = sizeof(pattern)}};
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download = {.source_buffer = buffer,
                     .target = &output_[i],
                     .length = sizeof(output_[i])}};
    uint64_t value = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_transfer(executions_[i], 1, &fill, &value));
    IREE_ASSERT_OK(
        loom_serve_execution_transfer(executions_[i], 1, &download, &value));
    IREE_ASSERT_OK(loom_serve_execution_drain(executions_[i]));
    EXPECT_EQ(output_[i], pattern);
  }
  loom_serve_virtual_buffer_begin_trim(reservations_[1]);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(reservations_[1]));
  EXPECT_EQ(loom_serve_device_memory_statistics(first).retained.committed_bytes,
            kSlabBytes);
  EXPECT_EQ(
      loom_serve_device_memory_statistics(second).retained.committed_bytes, 0u);
}

}  // namespace
