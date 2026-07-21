// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/host_notification.h"

#include "iree/hal/testing/mock_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(HostNotificationTest, RejectsUnsupportedDevice) {
  iree_hal_mock_device_options_t options;
  iree_hal_mock_device_options_initialize(&options);

  iree_hal_device_t* device = NULL;
  IREE_ASSERT_OK(
      iree_hal_mock_device_create(&options, iree_allocator_system(), &device));

  iree_hal_host_notification_t* notification = NULL;
  iree_status_t status =
      iree_hal_host_notification_create(device, &notification);
  EXPECT_EQ(IREE_STATUS_UNIMPLEMENTED, iree_status_code(status));
  iree_status_ignore(status);
  EXPECT_EQ(NULL, notification);

  iree_hal_device_release(device);
}

}  // namespace
