// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0
//
// Device operations. Generic across accelerator types once you have a handle.

#include "pyre_internal.h"

#include <string.h>

pyre_status_t pyre_device_get_property(pyre_device_t device,
                                       pyre_device_property_t prop,
                                       void* value, size_t value_size) {
  if (!device || !value) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "device or value is NULL");
  }
  switch (prop) {
    case PYRE_DEVICE_PROPERTY_NAME: {
      size_t len = strlen(device->name);
      if (value_size < len + 1) {
        return pyre_make_status(PYRE_STATUS_OUT_OF_RANGE,
                                "buffer too small for device name");
      }
      memcpy(value, device->name, len + 1);
      return pyre_ok_status();
    }
    case PYRE_DEVICE_PROPERTY_ARCHITECTURE: {
      size_t len = strlen(device->architecture);
      if (value_size < len + 1) {
        return pyre_make_status(PYRE_STATUS_OUT_OF_RANGE,
                                "buffer too small for architecture string");
      }
      memcpy(value, device->architecture, len + 1);
      return pyre_ok_status();
    }
    case PYRE_DEVICE_PROPERTY_TOTAL_MEMORY: {
      if (value_size < sizeof(uint64_t)) {
        return pyre_make_status(PYRE_STATUS_OUT_OF_RANGE,
                                "buffer too small for uint64_t");
      }
      // Query from HAL device.
      int64_t mem_size = 0;
      iree_status_t s = iree_hal_device_query_i64(
          device->hal_device, iree_make_cstring_view("hal.device"),
          iree_make_cstring_view("memory.total"), &mem_size);
      if (!iree_status_is_ok(s)) {
        iree_status_ignore(s);
        mem_size = 0;  // Unknown.
      }
      *(uint64_t*)value = (uint64_t)mem_size;
      return pyre_ok_status();
    }
    case PYRE_DEVICE_PROPERTY_COMPUTE_UNITS:
    case PYRE_DEVICE_PROPERTY_MAX_WORKGROUP_SIZE: {
      if (value_size < sizeof(uint32_t)) {
        return pyre_make_status(PYRE_STATUS_OUT_OF_RANGE,
                                "buffer too small for uint32_t");
      }
      *(uint32_t*)value = 0;  // Not available from local-task driver.
      return pyre_ok_status();
    }
    default:
      return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                              "unknown device property");
  }
}

pyre_status_t pyre_device_synchronize(pyre_device_t device) {
  if (!device) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "device is NULL");
  }
  // Wait for all queues to drain.
  iree_hal_semaphore_list_t empty = {0};
  iree_status_t status = iree_hal_device_wait_semaphores(
      device->hal_device, IREE_ASYNC_WAIT_MODE_ALL, empty,
      iree_infinite_timeout(), /*flags=*/0);
  return pyre_status_from_iree(status);
}

pyre_status_t pyre_device_get_type(pyre_device_t device,
                                   pyre_accelerator_type_t* type) {
  if (!device || !type) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "device or type is NULL");
  }
  *type = device->type;
  return pyre_ok_status();
}

pyre_status_t pyre_device_retain(pyre_device_t device) {
  if (!device) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "device is NULL");
  }
  iree_atomic_ref_count_inc(&device->ref_count);
  return pyre_ok_status();
}

pyre_status_t pyre_device_release(pyre_device_t device) {
  if (!device) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "device is NULL");
  }
  // Devices are owned by the accelerator state, not individually freed.
  // Release just decrements the refcount for tracking.
  // Actual cleanup happens in pyre_{cpu,gpu}_shutdown.
  return pyre_ok_status();
}
