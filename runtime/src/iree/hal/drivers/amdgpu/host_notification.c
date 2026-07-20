// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_notification.h"

#include <string.h>

#include "iree/hal/detail.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

typedef struct iree_hal_amdgpu_host_notification_t {
  // Base resource interface.
  iree_hal_resource_t resource;
  // Allocator used to free the notification.
  iree_allocator_t host_allocator;
  // Retained HSA API table used for native signal operations.
  iree_hal_amdgpu_libhsa_t libhsa;
  // Signal shared by device requesters and the host listener.
  hsa_signal_t signal;
} iree_hal_amdgpu_host_notification_t;

static const iree_hal_host_notification_vtable_t
    iree_hal_amdgpu_host_notification_vtable;

static iree_hal_amdgpu_host_notification_t*
iree_hal_amdgpu_host_notification_cast(
    iree_hal_host_notification_t* base_notification) {
  IREE_HAL_ASSERT_TYPE(base_notification,
                       &iree_hal_amdgpu_host_notification_vtable);
  return (iree_hal_amdgpu_host_notification_t*)base_notification;
}

static void iree_hal_amdgpu_host_notification_destroy(
    iree_hal_host_notification_t* base_notification) {
  iree_hal_amdgpu_host_notification_t* notification =
      iree_hal_amdgpu_host_notification_cast(base_notification);
  iree_allocator_t host_allocator = notification->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (notification->signal.handle) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_signal_destroy_raw(
        &notification->libhsa, notification->signal));
  }
  iree_hal_amdgpu_libhsa_deinitialize(&notification->libhsa);
  iree_allocator_free(host_allocator, notification);

  IREE_TRACE_ZONE_END(z0);
}

static uint64_t iree_hal_amdgpu_host_notification_device_token(
    iree_hal_host_notification_t* base_notification) {
  iree_hal_amdgpu_host_notification_t* notification =
      iree_hal_amdgpu_host_notification_cast(base_notification);
  return notification->signal.handle;
}

static uint64_t iree_hal_amdgpu_host_notification_wait(
    iree_hal_host_notification_t* base_notification, uint64_t observed_value) {
  iree_hal_amdgpu_host_notification_t* notification =
      iree_hal_amdgpu_host_notification_cast(base_notification);
  return (uint64_t)iree_hsa_signal_wait_scacquire(
      IREE_LIBHSA(&notification->libhsa), notification->signal,
      HSA_SIGNAL_CONDITION_NE, (hsa_signal_value_t)observed_value, UINT64_MAX,
      HSA_WAIT_STATE_BLOCKED);
}

static void iree_hal_amdgpu_host_notification_wake(
    iree_hal_host_notification_t* base_notification) {
  iree_hal_amdgpu_host_notification_t* notification =
      iree_hal_amdgpu_host_notification_cast(base_notification);
  iree_hsa_signal_store_screlease(IREE_LIBHSA(&notification->libhsa),
                                  notification->signal, 0);
}

static const iree_hal_host_notification_vtable_t
    iree_hal_amdgpu_host_notification_vtable = {
        .destroy = iree_hal_amdgpu_host_notification_destroy,
        .device_token = iree_hal_amdgpu_host_notification_device_token,
        .wait = iree_hal_amdgpu_host_notification_wait,
        .wake = iree_hal_amdgpu_host_notification_wake,
};

iree_status_t iree_hal_amdgpu_host_notification_create(
    iree_hal_amdgpu_logical_device_t* device,
    iree_hal_host_notification_t** out_notification) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_notification);
  *out_notification = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_host_notification_t* notification = NULL;
  iree_status_t status = iree_allocator_malloc(
      device->host_allocator, sizeof(*notification), (void**)&notification);
  if (iree_status_is_ok(status)) {
    memset(notification, 0, sizeof(*notification));
    notification->host_allocator = device->host_allocator;
    status = iree_hal_amdgpu_libhsa_copy(&device->system->libhsa,
                                         &notification->libhsa);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hsa_amd_signal_create(IREE_LIBHSA(&notification->libhsa),
                                   IREE_HAL_HOST_NOTIFICATION_INITIAL_VALUE,
                                   /*num_consumers=*/0, /*consumers=*/NULL,
                                   /*attributes=*/0, &notification->signal);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_amdgpu_host_notification_vtable,
                                 &notification->resource);
    *out_notification = (iree_hal_host_notification_t*)notification;
  } else if (notification) {
    if (notification->signal.handle) {
      iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_signal_destroy_raw(
          &notification->libhsa, notification->signal));
    }
    iree_hal_amdgpu_libhsa_deinitialize(&notification->libhsa);
    iree_allocator_free(notification->host_allocator, notification);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
