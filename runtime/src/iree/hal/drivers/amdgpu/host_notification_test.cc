// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_notification.h"

#include <thread>

#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

class HostNotificationTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        iree_allocator_system(), &libhsa_);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
  }

  static iree_hal_amdgpu_libhsa_t libhsa_;
};

iree_hal_amdgpu_libhsa_t HostNotificationTest::libhsa_;

TEST_F(HostNotificationTest, WakeReleasesBlockedWait) {
  iree_hal_amdgpu_system_t system = {};
  system.libhsa = libhsa_;
  iree_hal_amdgpu_logical_device_t device = {};
  device.host_allocator = iree_allocator_system();
  device.system = &system;

  iree_hal_host_notification_t* notification = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_host_notification_create(&device, &notification));
  ASSERT_NE(nullptr, notification);
  EXPECT_NE(0u, iree_hal_host_notification_device_token(notification));

  uint64_t observed_value = IREE_HAL_HOST_NOTIFICATION_INITIAL_VALUE;
  std::thread waiter([&]() {
    observed_value = iree_hal_host_notification_wait(
        notification, IREE_HAL_HOST_NOTIFICATION_INITIAL_VALUE);
  });
  iree_hal_host_notification_wake(notification);
  waiter.join();

  EXPECT_EQ(IREE_HAL_HOST_NOTIFICATION_INITIAL_VALUE + 1, observed_value);
  iree_hal_host_notification_release(notification);
}

TEST_F(HostNotificationTest, WakeDiffersFromPostNotificationValue) {
  iree_hal_amdgpu_system_t system = {};
  system.libhsa = libhsa_;
  iree_hal_amdgpu_logical_device_t device = {};
  device.host_allocator = iree_allocator_system();
  device.system = &system;

  iree_hal_host_notification_t* notification = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_host_notification_create(&device, &notification));
  const hsa_signal_t signal = {
      /*.handle=*/iree_hal_host_notification_device_token(notification),
  };
  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), signal, 0);
  iree_hal_host_notification_wake(notification);
  EXPECT_EQ(1u, iree_hal_host_notification_wait(notification,
                                                /*observed_value=*/0));
  iree_hal_host_notification_release(notification);
}

}  // namespace
}  // namespace iree::hal::amdgpu
