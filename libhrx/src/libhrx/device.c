// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Device operations. Generic across accelerator types once you have a handle.

#include <stdint.h>
#include <string.h>

#include "hrx_internal.h"

hrx_status_t hrx_device_query_total_memory_from_spec(
    hrx_device_t device, bool* out_known, iree_device_size_t* out_total) {
  if (!device || !out_known || !out_total) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  *out_known = false;
  *out_total = 0;

  iree_hal_device_observation_t observation;
  iree_hal_device_observation_initialize(
      IREE_HAL_DEVICE_OBSERVATION_FLAG_MEMORY, &observation);
  iree_status_t status =
      iree_hal_device_observation_populate_memory_total_from_spec(
          iree_hal_device_spec(device->hal_device), &observation);
  if (!iree_status_is_ok(status)) return hrx_status_from_iree(status);
  if (iree_all_bits_set(observation.memory.flags,
                        IREE_HAL_DEVICE_MEMORY_OBSERVATION_FLAG_TOTAL_BYTES)) {
    *out_known = true;
    *out_total = observation.memory.total_bytes;
  }
  return hrx_ok_status();
}

static hrx_status_t hrx_device_sample_memory(
    hrx_device_t device, iree_device_size_t* out_total,
    iree_device_size_t* out_available) {
  iree_hal_device_observation_t observation;
  iree_status_t status = iree_hal_device_sample_observation(
      device->hal_device, IREE_HAL_DEVICE_OBSERVATION_FLAG_MEMORY,
      &observation);
  if (!iree_status_is_ok(status)) return hrx_status_from_iree(status);
  if (!iree_all_bits_set(observation.provided_flags,
                         IREE_HAL_DEVICE_OBSERVATION_FLAG_MEMORY)) {
    return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                           "HAL device did not provide a memory observation");
  }
  if (!iree_all_bits_set(observation.memory.flags,
                         IREE_HAL_DEVICE_MEMORY_OBSERVATION_FLAG_TOTAL_BYTES)) {
    return hrx_make_status(
        HRX_STATUS_UNAVAILABLE,
        "HAL device did not provide total memory in its observation");
  }
  if (out_available &&
      !iree_all_bits_set(
          observation.memory.flags,
          IREE_HAL_DEVICE_MEMORY_OBSERVATION_FLAG_AVAILABLE_BYTES)) {
    return hrx_make_status(
        HRX_STATUS_UNAVAILABLE,
        "HAL device did not provide available memory in its observation");
  }
  *out_total = observation.memory.total_bytes;
  if (out_available) *out_available = observation.memory.available_bytes;
  return hrx_ok_status();
}

hrx_status_t hrx_device_get_property(hrx_device_t device,
                                     hrx_device_property_t prop, void* value,
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
      iree_device_size_t total_bytes = 0;
      bool total_memory_known = false;
      hrx_status_t status = hrx_device_query_total_memory_from_spec(
          device, &total_memory_known, &total_bytes);
      if (!hrx_status_is_ok(status)) return status;
      if (!total_memory_known) {
        return hrx_make_status(
            HRX_STATUS_UNAVAILABLE,
            "HAL device spec did not provide a known total memory capacity");
      }
      *(uint64_t*)value = (uint64_t)total_bytes;
      return status;
    }
    case HRX_DEVICE_PROPERTY_COMPUTE_UNITS:
    case HRX_DEVICE_PROPERTY_MAX_WORKGROUP_SIZE: {
      if (value_size < sizeof(uint32_t)) {
        return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                               "buffer too small for uint32_t");
      }
      *(uint32_t*)value = 0;  // Not available from local-task driver.
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
  iree_hal_semaphore_list_t empty = iree_hal_semaphore_list_empty();
  iree_status_t status = iree_hal_device_wait_semaphores(
      device->hal_device, IREE_ASYNC_WAIT_MODE_ALL, empty,
      iree_infinite_timeout(), /*flags=*/0);
  return hrx_status_from_iree(status);
}

hrx_status_t hrx_device_get_type(hrx_device_t device,
                                 hrx_accelerator_type_t* type) {
  if (!device || !type) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "device or type is NULL");
  }
  *type = device->type;
  return hrx_ok_status();
}

void hrx_device_retain(hrx_device_t device) {
  if (!device) return;
  iree_hal_device_retain(device->hal_device);
  iree_hal_device_group_retain(device->hal_device_group);
  iree_atomic_ref_count_inc(&device->ref_count);
}

void hrx_device_release(hrx_device_t device) {
  if (!device) return;
  iree_hal_device_t* hal_device = device->hal_device;
  iree_hal_device_group_t* hal_device_group = device->hal_device_group;
  if (iree_atomic_ref_count_dec(&device->ref_count) == 1) {
    iree_hal_allocator_release(device->allocator.hal_allocator);
    device->allocator.hal_allocator = NULL;
    device->hal_device = NULL;
    device->hal_device_group = NULL;
  }
  iree_hal_device_group_release(hal_device_group);
  iree_hal_device_release(hal_device);
}

hrx_status_t hrx_device_memory_info(hrx_device_t device, size_t* free_bytes,
                                    size_t* total_bytes) {
  if (!device || !free_bytes || !total_bytes) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }

  iree_device_size_t total = 0;
  iree_device_size_t available = 0;
  hrx_status_t status = hrx_device_sample_memory(device, &total, &available);
  if (!hrx_status_is_ok(status)) return status;
  if (total > SIZE_MAX || available > SIZE_MAX) {
    return hrx_make_status(
        HRX_STATUS_OUT_OF_RANGE,
        "HAL memory observation exceeds the representable size_t range");
  }
  *total_bytes = (size_t)total;
  *free_bytes = (size_t)available;
  return status;
}

hrx_status_t hrx_device_can_access_peer(hrx_device_t device_a,
                                        hrx_device_t device_b,
                                        bool* can_access) {
  if (!device_a || !device_b || !can_access) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  if (device_a == device_b) {
    *can_access = true;
    return hrx_ok_status();
  }
  *can_access = (device_a->type == HRX_ACCELERATOR_GPU &&
                 device_b->type == HRX_ACCELERATOR_GPU);
  return hrx_ok_status();
}
