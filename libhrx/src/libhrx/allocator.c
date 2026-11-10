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
  iree_hal_allocator_retain(allocator->hal_allocator);
  hrx_device_retain(allocator->device);
}

void hrx_allocator_release(hrx_allocator_t allocator) {
  if (!allocator) {
    return;
  }
  iree_hal_allocator_release(allocator->hal_allocator);
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

//===----------------------------------------------------------------------===//
// Virtual memory
//===----------------------------------------------------------------------===//

static iree_status_t hrx_allocator_require_legacy_virtual_memory(
    hrx_allocator_t allocator) {
  if (!allocator->hal_allocator ||
      !iree_hal_allocator_supports_virtual_memory(allocator->hal_allocator)) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "legacy virtual memory requires a legacy device allocator; use "
        "virtual buffers and queue_map for ordinary memory backends");
  }
  return iree_ok_status();
}

hrx_status_t hrx_allocator_query_virtual_memory(hrx_allocator_t allocator,
                                                hrx_memory_type_t mem_type,
                                                bool* supported,
                                                size_t* min_page_size,
                                                size_t* recommended_page_size) {
  if (!allocator || !supported) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "allocator or supported is NULL");
  }

  if (!allocator->hal_allocator ||
      !iree_hal_allocator_supports_virtual_memory(allocator->hal_allocator)) {
    *supported = false;
    if (min_page_size) {
      *min_page_size = 0;
    }
    if (recommended_page_size) {
      *recommended_page_size = 0;
    }
    return hrx_ok_status();
  }

  iree_hal_physical_memory_params_t hal_params = {
      .type = (iree_hal_memory_type_t)mem_type,
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
  };
  iree_device_size_t hal_minimum_page_size = 0;
  iree_device_size_t hal_recommended_page_size = 0;
  iree_status_t status = iree_hal_allocator_virtual_memory_query_granularity(
      allocator->hal_allocator, hal_params, &hal_minimum_page_size,
      &hal_recommended_page_size);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }

  *supported = true;
  if (min_page_size) {
    *min_page_size = (size_t)hal_minimum_page_size;
  }
  if (recommended_page_size) {
    *recommended_page_size = (size_t)hal_recommended_page_size;
  }
  return hrx_ok_status();
}

hrx_status_t hrx_allocator_virtual_memory_reserve(
    hrx_allocator_t allocator, size_t size, hrx_buffer_t* virtual_buffer) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_allocator_virtual_memory_reserve");
  HRX_TRACE_ZONE_APPEND_BYTES(z0, size);
  if (!allocator || !virtual_buffer) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument"));
  }
  HRX_RETURN_AND_END_ZONE_IF_IREE_ERROR(
      z0, hrx_allocator_require_legacy_virtual_memory(allocator));

  iree_hal_buffer_t* hal_buffer = NULL;
  iree_status_t status = iree_hal_allocator_virtual_memory_reserve(
      allocator->hal_allocator, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      (iree_device_size_t)size, &hal_buffer);
  if (!iree_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  hrx_buffer_t buf = NULL;
  iree_status_t alloc_status = iree_allocator_malloc(
      iree_allocator_system(), sizeof(hrx_buffer_s), (void**)&buf);
  if (!iree_status_is_ok(alloc_status)) {
    alloc_status = iree_status_join(alloc_status,
                                    iree_hal_allocator_virtual_memory_release(
                                        allocator->hal_allocator, hal_buffer));
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(alloc_status));
  }

  memset(buf, 0, sizeof(*buf));
  iree_atomic_ref_count_init(&buf->ref_count);
  buf->hal_buffer = hal_buffer;
  buf->group = allocator->device->group;
  hrx_device_group_retain(buf->group);
  buf->mem_type = HRX_MEMORY_TYPE_DEVICE_LOCAL;
  buf->size = size;
  *virtual_buffer = buf;
  HRX_RETURN_AND_END_ZONE(z0, hrx_ok_status());
}

hrx_status_t hrx_allocator_virtual_memory_release(hrx_allocator_t allocator,
                                                  hrx_buffer_t virtual_buffer) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_allocator_virtual_memory_release");
  if (virtual_buffer) {
    HRX_TRACE_ZONE_APPEND_BYTES(z0, virtual_buffer->size);
  }
  if (!allocator || !virtual_buffer) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument"));
  }
  HRX_RETURN_AND_END_ZONE_IF_IREE_ERROR(
      z0, hrx_allocator_require_legacy_virtual_memory(allocator));
  iree_status_t status = iree_hal_allocator_virtual_memory_release(
      allocator->hal_allocator, virtual_buffer->hal_buffer);
  if (iree_status_is_ok(status)) {
    // Release consumes the HAL buffer only on success. Keep the wrapper usable
    // when active mappings prevent release so the caller can unmap and retry.
    virtual_buffer->hal_buffer = NULL;
    hrx_device_group_release(virtual_buffer->group);
    iree_allocator_free(iree_allocator_system(), virtual_buffer);
  }
  HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
}

hrx_status_t hrx_allocator_physical_memory_allocate(
    hrx_allocator_t allocator, hrx_memory_type_t mem_type, size_t size,
    hrx_physical_memory_t* physical) {
  if (!allocator || !physical) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  HRX_RETURN_IF_IREE_ERROR(
      hrx_allocator_require_legacy_virtual_memory(allocator));
  iree_hal_physical_memory_params_t hal_params = {
      .type = (iree_hal_memory_type_t)mem_type,
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
  };
  return hrx_status_from_iree(iree_hal_allocator_physical_memory_allocate(
      allocator->hal_allocator, hal_params, (iree_device_size_t)size,
      iree_allocator_system(), (iree_hal_physical_memory_t**)physical));
}

hrx_status_t hrx_allocator_physical_memory_free(
    hrx_allocator_t allocator, hrx_physical_memory_t physical) {
  if (!allocator || !physical) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  HRX_RETURN_IF_IREE_ERROR(
      hrx_allocator_require_legacy_virtual_memory(allocator));
  return hrx_status_from_iree(iree_hal_allocator_physical_memory_free(
      allocator->hal_allocator, (iree_hal_physical_memory_t*)physical));
}

hrx_status_t hrx_allocator_virtual_memory_map(hrx_allocator_t allocator,
                                              hrx_buffer_t virtual_buffer,
                                              size_t virtual_offset,
                                              hrx_physical_memory_t physical,
                                              size_t physical_offset,
                                              size_t size) {
  if (!allocator || !virtual_buffer || !physical) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  HRX_RETURN_IF_IREE_ERROR(
      hrx_allocator_require_legacy_virtual_memory(allocator));
  return hrx_status_from_iree(iree_hal_allocator_virtual_memory_map(
      allocator->hal_allocator, virtual_buffer->hal_buffer,
      (iree_device_size_t)virtual_offset, (iree_hal_physical_memory_t*)physical,
      (iree_device_size_t)physical_offset, (iree_device_size_t)size));
}

hrx_status_t hrx_allocator_virtual_memory_unmap(hrx_allocator_t allocator,
                                                hrx_buffer_t virtual_buffer,
                                                size_t virtual_offset,
                                                size_t size) {
  if (!allocator || !virtual_buffer) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  HRX_RETURN_IF_IREE_ERROR(
      hrx_allocator_require_legacy_virtual_memory(allocator));
  return hrx_status_from_iree(iree_hal_allocator_virtual_memory_unmap(
      allocator->hal_allocator, virtual_buffer->hal_buffer,
      (iree_device_size_t)virtual_offset, (iree_device_size_t)size));
}

static iree_status_t hrx_virtual_memory_access_scope_to_hal(
    hrx_virtual_memory_access_scope_t access_scope,
    iree_hal_virtual_memory_access_scope_t* out_access_scope) {
  switch (access_scope) {
    case HRX_VIRTUAL_MEMORY_ACCESS_SCOPE_ALL:
      *out_access_scope = IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_ALL;
      return iree_ok_status();
    case HRX_VIRTUAL_MEMORY_ACCESS_SCOPE_HOST:
      *out_access_scope = IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_HOST;
      return iree_ok_status();
    case HRX_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE:
      *out_access_scope = IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid virtual-memory access scope");
  }
}

hrx_status_t hrx_allocator_virtual_memory_protect(
    hrx_allocator_t allocator, hrx_buffer_t virtual_buffer,
    size_t virtual_offset, size_t size, hrx_memory_protection_t protection) {
  return hrx_allocator_virtual_memory_protect_scoped(
      allocator, virtual_buffer, virtual_offset, size,
      HRX_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE, protection);
}

hrx_status_t hrx_allocator_virtual_memory_protect_scoped(
    hrx_allocator_t allocator, hrx_buffer_t virtual_buffer,
    size_t virtual_offset, size_t size,
    hrx_virtual_memory_access_scope_t access_scope,
    hrx_memory_protection_t protection) {
  if (!allocator || !virtual_buffer) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  HRX_RETURN_IF_IREE_ERROR(
      hrx_allocator_require_legacy_virtual_memory(allocator));
  iree_hal_virtual_memory_access_scope_t hal_access_scope;
  iree_status_t status =
      hrx_virtual_memory_access_scope_to_hal(access_scope, &hal_access_scope);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }
  return hrx_status_from_iree(iree_hal_allocator_virtual_memory_protect(
      allocator->hal_allocator, virtual_buffer->hal_buffer,
      (iree_device_size_t)virtual_offset, (iree_device_size_t)size,
      /*queue_family_affinity=*/IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      hal_access_scope, (iree_hal_memory_protection_t)protection));
}
