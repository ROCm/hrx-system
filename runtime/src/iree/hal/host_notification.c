// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/host_notification.h"

#include "iree/hal/detail.h"
#include "iree/hal/device.h"

#define _VTABLE_DISPATCH(notification, method_name)                  \
  IREE_HAL_VTABLE_DISPATCH(notification, iree_hal_host_notification, \
                           method_name)

IREE_HAL_API_RETAIN_RELEASE(host_notification);

IREE_API_EXPORT iree_status_t iree_hal_host_notification_create(
    iree_hal_device_t* device,
    iree_hal_host_notification_t** out_notification) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_notification);
  *out_notification = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = IREE_HAL_VTABLE_DISPATCH(
      device, iree_hal_device, create_host_notification)(device,
                                                         out_notification);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_host_notification_create_unimplemented(
    iree_hal_device_t* device,
    iree_hal_host_notification_t** out_notification) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_notification);
  *out_notification = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "device does not support host notifications");
}

IREE_API_EXPORT uint64_t iree_hal_host_notification_device_token(
    iree_hal_host_notification_t* notification) {
  IREE_ASSERT_ARGUMENT(notification);
  return _VTABLE_DISPATCH(notification, device_token)(notification);
}

IREE_API_EXPORT uint64_t iree_hal_host_notification_wait(
    iree_hal_host_notification_t* notification, uint64_t observed_value) {
  IREE_ASSERT_ARGUMENT(notification);
  IREE_TRACE_ZONE_BEGIN(z0);
  uint64_t value =
      _VTABLE_DISPATCH(notification, wait)(notification, observed_value);
  IREE_TRACE_ZONE_END(z0);
  return value;
}

IREE_API_EXPORT void iree_hal_host_notification_wake(
    iree_hal_host_notification_t* notification) {
  IREE_ASSERT_ARGUMENT(notification);
  _VTABLE_DISPATCH(notification, wake)(notification);
}
