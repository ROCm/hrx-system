// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <string.h>

#include "hrx_internal.h"

hrx_allocator_t hrx_device_allocator(hrx_device_t device) {
  return &device->allocator;
}

void hrx_allocator_retain(hrx_allocator_t allocator) {
  if (!allocator) {
    return;
  }
  hrx_device_retain(allocator->device);
}

void hrx_allocator_release(hrx_allocator_t allocator) {
  if (!allocator) {
    return;
  }
  hrx_device_release(allocator->device);
}

hrx_status_t hrx_allocator_allocate_buffer(hrx_allocator_t allocator,
                                           hrx_buffer_params_t params,
                                           size_t size, hrx_buffer_t* buffer) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_allocator_allocate_buffer");
  HRX_TRACE_ZONE_APPEND_BYTES(z0, size);
  if (!allocator || !buffer) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                                                "allocator or buffer is NULL"));
  }

  iree_hal_pool_t* pool = NULL;
  iree_hal_buffer_t* hal_buffer = NULL;
  iree_status_t status = hrx_mem_pool_backing_retain_allocation_source(
      allocator->device, params.type, params.usage, &pool);
  if (iree_status_is_ok(status)) {
    const iree_hal_pool_reservation_request_t request =
        iree_hal_make_pool_reservation_request((iree_device_size_t)size,
                                               /*min_alignment=*/0);
    status = iree_hal_pool_allocate_buffer(
        pool, request, iree_infinite_timeout(), &hal_buffer);
  }
  iree_hal_pool_release(pool);
  if (!iree_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  status = hrx_buffer_create_from_hal(hal_buffer, allocator->device->group,
                                      params.type, size,
                                      /*device_ptr=*/NULL, buffer);
  iree_hal_buffer_release(hal_buffer);
  HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
}

hrx_status_t hrx_allocator_import_buffer(hrx_allocator_t allocator,
                                         hrx_buffer_params_t params,
                                         void* host_ptr, size_t size,
                                         hrx_buffer_t* buffer) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_allocator_import_buffer");
  HRX_TRACE_ZONE_APPEND_BYTES(z0, size);
  if (!allocator || !host_ptr || !buffer) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "allocator, host_ptr, or buffer is NULL"));
  }
  if (iree_any_bit_set(params.usage, HRX_BUFFER_USAGE_MAPPING_SOURCE)) {
    HRX_RETURN_AND_END_ZONE(
        z0,
        hrx_make_status(HRX_STATUS_UNAVAILABLE,
                        "host registration cannot provide remappable backing"));
  }

  iree_hal_external_storage_t ext = {
      .storage_class.type = IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER,
      .byte_length = (iree_device_size_t)size,
      .payload.host_pointer = host_ptr,
      .host_properties = (IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED |
                          IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK),
      .maintenance_range =
          iree_hal_make_buffer_range(0, (iree_device_size_t)size),
  };

  iree_hal_pool_t* pool = NULL;
  iree_hal_buffer_t* hal_buffer = NULL;
  iree_status_t status = hrx_mem_pool_backing_retain_import_source(
      allocator->device, ext.storage_class, &pool);
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_import_buffer(
        pool, &ext, iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
        &hal_buffer);
  }
  iree_hal_pool_release(pool);
  if (!iree_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  status = hrx_buffer_create_from_hal(hal_buffer, allocator->device->group,
                                      params.type, size, /*device_ptr=*/NULL,
                                      buffer);
  iree_hal_buffer_release(hal_buffer);
  HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
}
