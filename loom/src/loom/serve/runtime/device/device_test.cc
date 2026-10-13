// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/device/device.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class DeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(loom_serve_host_create(1, &host, iree_allocator_system()));
  }
  void TearDown() override { loom_serve_host_release(host); }

  // Application host shared by device creation and failure cleanup.
  loom_serve_host_t* host = nullptr;
};

TEST_F(DeviceTest, MissingDriverReleasesPartialOwnership) {
  loom_serve_device_group_t* group = nullptr;
  const loom_serve_device_options_t options = {
      .uri = IREE_SV("unregistered-serving-device"),
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, loom_serve_device_group_create(
                                                   host, 1, &options, &group,
                                                   iree_allocator_system()));
  EXPECT_EQ(group, nullptr);
  IREE_EXPECT_OK(loom_serve_device_group_destroy(group));
}

TEST_F(DeviceTest, AllocationFailureReturnsNoOwner) {
  loom_serve_device_group_t* group = nullptr;
  const loom_serve_device_options_t options = {.uri = IREE_SV("amdgpu")};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_device_group_create(
                            host, 1, &options, &group, iree_allocator_null()));
  EXPECT_EQ(group, nullptr);
  IREE_EXPECT_OK(loom_serve_device_group_destroy(group));
}

TEST_F(DeviceTest, InvalidBackingRejectsBeforeDeviceCreation) {
  loom_serve_device_group_t* group = nullptr;
  loom_serve_device_options_t options = {
      .uri = IREE_SV("unregistered-serving-device"),
      .memory_limit = 1,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_device_group_create(host, 1, &options, &group,
                                     iree_allocator_system()));
  EXPECT_EQ(group, nullptr);
  options.memory_limit = 0;
  options.backing = static_cast<loom_serve_device_backing_t>(2);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_device_group_create(host, 1, &options, &group,
                                     iree_allocator_system()));
  EXPECT_EQ(group, nullptr);
}

TEST_F(DeviceTest, InvalidCountRejectsBeforeReadingOptions) {
  loom_serve_device_group_t* group = nullptr;
  const iree_host_size_t counts[] = {0, IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT + 1};
  for (iree_host_size_t count : counts) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_device_group_create(host, count, nullptr, &group,
                                       iree_allocator_system()));
    EXPECT_EQ(group, nullptr);
  }
}

TEST_F(DeviceTest, EveryOptionIsValidatedBeforeCreatingDevices) {
  loom_serve_device_group_t* group = nullptr;
  const loom_serve_device_options_t options[] = {
      {.uri = IREE_SV("unregistered-serving-device")},
      {.uri = IREE_SV("unregistered-serving-device"), .memory_limit = 1}};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_device_group_create(host, IREE_ARRAYSIZE(options), options,
                                     &group, iree_allocator_system()));
  EXPECT_EQ(group, nullptr);
}

}  // namespace
