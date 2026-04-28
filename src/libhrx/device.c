// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Device operations. Generic across accelerator types once you have a handle.

#include "hrx_internal.h"

#include <string.h>

hrx_status_t hrx_device_get_property(hrx_device_t device,
                                     hrx_device_property_t prop, void *value,
                                     size_t value_size) {
  if (!device || !value) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "device or value is NULL");
  }
  switch (prop) {
  case HRX_DEVICE_PROPERTY_NAME: {
    size_t len = strlen(device->name);
    if (value_size < len + 1) {
      return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                             "buffer too small for device name");
    }
    memcpy(value, device->name, len + 1);
    return hrx_ok_status();
  }
  case HRX_DEVICE_PROPERTY_ARCHITECTURE: {
    size_t len = strlen(device->architecture);
    if (value_size < len + 1) {
      return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                             "buffer too small for architecture string");
    }
    memcpy(value, device->architecture, len + 1);
    return hrx_ok_status();
  }
  case HRX_DEVICE_PROPERTY_TOTAL_MEMORY: {
    if (value_size < sizeof(uint64_t)) {
      return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                             "buffer too small for uint64_t");
    }
    // Query from HAL device.
    int64_t mem_size = 0;
    iree_status_t s = iree_hal_device_query_i64(
        device->hal_device, iree_make_cstring_view("hal.device"),
        iree_make_cstring_view("memory.total"), &mem_size);
    if (!iree_status_is_ok(s)) {
      iree_status_ignore(s);
      mem_size = 0; // Unknown.
    }
    *(uint64_t *)value = (uint64_t)mem_size;
    return hrx_ok_status();
  }
  case HRX_DEVICE_PROPERTY_COMPUTE_UNITS:
  case HRX_DEVICE_PROPERTY_MAX_WORKGROUP_SIZE: {
    if (value_size < sizeof(uint32_t)) {
      return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                             "buffer too small for uint32_t");
    }
    *(uint32_t *)value = 0; // Not available from local-task driver.
    return hrx_ok_status();
  }
  default:
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "unknown device property");
  }
}

hrx_status_t hrx_device_synchronize(hrx_device_t device) {
  if (!device) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "device is NULL");
  }
  // Deprecated no-op compatibility shim. IREE requires callers to wait on
  // explicit semaphore payloads; an empty wait list returns immediately.
  iree_hal_semaphore_list_t empty = {0};
  iree_status_t status = iree_hal_device_wait_semaphores(
      device->hal_device, IREE_ASYNC_WAIT_MODE_ALL, empty,
      iree_infinite_timeout(), /*flags=*/0);
  return hrx_status_from_iree(status);
}

hrx_status_t hrx_device_get_type(hrx_device_t device,
                                 hrx_accelerator_type_t *type) {
  if (!device || !type) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "device or type is NULL");
  }
  *type = device->type;
  return hrx_ok_status();
}

void hrx_device_retain(hrx_device_t device) {
  iree_hal_device_retain(device->hal_device);
  iree_atomic_ref_count_inc(&device->ref_count);
}

void hrx_device_release(hrx_device_t device) {
  iree_hal_device_t *hal_device = device->hal_device;
  iree_hal_device_group_t *hal_device_group = device->hal_device_group;
  if (iree_atomic_ref_count_dec(&device->ref_count) == 1) {
    iree_hal_allocator_release(device->allocator.hal_allocator);
    device->allocator.hal_allocator = NULL;
    device->hal_device = NULL;
    device->hal_device_group = NULL;
  }
  iree_hal_device_release(hal_device);
  iree_hal_device_group_release(hal_device_group);
}
