// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/device.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(DeviceTest, MissingDriverReleasesPartialOwnership) {
  loom_serve_device_t* device = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_serve_device_create(IREE_SV("unregistered-serving-device"),
                               iree_allocator_system(), &device));
  EXPECT_EQ(device, nullptr);
  loom_serve_device_destroy(device);
}

TEST(DeviceTest, AllocationFailureReturnsNoOwner) {
  loom_serve_device_t* device = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_device_create(
                            IREE_SV("amdgpu"), iree_allocator_null(), &device));
  EXPECT_EQ(device, nullptr);
  loom_serve_device_destroy(device);
}

}  // namespace
