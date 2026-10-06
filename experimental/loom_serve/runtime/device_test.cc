// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/device.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(DeviceTest, MissingDriverReleasesPartialOwnership) {
  loom_serve_device_t* device = nullptr;
  const loom_serve_device_options_t options = {
      .uri = IREE_SV("unregistered-serving-device")};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_serve_device_create(&options, &device, iree_allocator_system()));
  EXPECT_EQ(device, nullptr);
  IREE_EXPECT_OK(loom_serve_device_destroy(device));
}

TEST(DeviceTest, AllocationFailureReturnsNoOwner) {
  loom_serve_device_t* device = nullptr;
  const loom_serve_device_options_t options = {.uri = IREE_SV("amdgpu")};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_device_create(&options, &device, iree_allocator_null()));
  EXPECT_EQ(device, nullptr);
  IREE_EXPECT_OK(loom_serve_device_destroy(device));
}

TEST(DeviceTest, InvalidBackingRejectsBeforeDeviceCreation) {
  loom_serve_device_t* device = nullptr;
  loom_serve_device_options_t options = {
      .uri = IREE_SV("unregistered-serving-device"), .memory_limit = 1};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_device_create(&options, &device, iree_allocator_system()));
  EXPECT_EQ(device, nullptr);
  options.memory_limit = 0;
  options.backing = static_cast<loom_serve_device_backing_t>(2);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_device_create(&options, &device, iree_allocator_system()));
  EXPECT_EQ(device, nullptr);
}

}  // namespace
