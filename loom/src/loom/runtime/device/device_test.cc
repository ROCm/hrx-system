// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/device/device.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(DeviceGroupTest, ExternallyOwnedResourcesHaveNoProgressObligations) {
  loom_runtime_device_group_set_observer(nullptr, {});
  IREE_ASSERT_OK(loom_runtime_device_group_advance(nullptr));
  EXPECT_FALSE(loom_runtime_device_group_is_pending(nullptr));
}

class DeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        loom_runtime_host_create(1, nullptr, iree_allocator_system(), &host));
  }
  void TearDown() override { loom_runtime_host_release(host); }

  // Application host shared by device creation and failure cleanup.
  loom_runtime_host_t* host = nullptr;
};

TEST_F(DeviceTest, AllocationFailureReturnsNoOwner) {
  loom_runtime_device_group_t* group = nullptr;
  const loom_runtime_device_options_t options = {
      .driver = loom_runtime_host_task_driver(host),
      .uri = IREE_SV("task"),
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_runtime_device_group_allocate(
                            host, 1, &options, iree_allocator_null(), &group));
  EXPECT_EQ(group, nullptr);
  IREE_EXPECT_OK(loom_runtime_device_group_free(group));
}

TEST_F(DeviceTest, InvalidCountRejectsBeforeReadingOptions) {
  loom_runtime_device_group_t* group = nullptr;
  const iree_host_size_t counts[] = {0, IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT + 1};
  for (iree_host_size_t count : counts) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_runtime_device_group_allocate(host, count, nullptr,
                                           iree_allocator_system(), &group));
    EXPECT_EQ(group, nullptr);
  }
}

TEST_F(DeviceTest, InvalidNamesRejectBeforeNativeCreation) {
  const iree_string_view_t invalid[] = {
      IREE_SV(""), IREE_SV("first"), IREE_SV("bad name"), IREE_SV("bad\nname")};
  for (const auto name : invalid) {
    // No driver is present: invalid names must reject before native creation.
    const loom_runtime_device_options_t options[] = {
        {.uri = IREE_SV("task"), .name = IREE_SV("first")},
        {.uri = IREE_SV("task"), .name = name}};
    loom_runtime_device_group_t* group = nullptr;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_runtime_device_group_allocate(
                              host, IREE_ARRAYSIZE(options), options,
                              iree_allocator_system(), &group));
    EXPECT_EQ(group, nullptr);
  }
}

}  // namespace
