// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_internal.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Allocator
//===----------------------------------------------------------------------===//

pyre_allocator_t pyre_device_allocator(pyre_device_t device) {
  return &device->allocator;
}

pyre_status_t pyre_allocator_retain(pyre_allocator_t allocator) {
  if (!allocator) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "allocator is NULL");
  }
  iree_atomic_ref_count_inc(&allocator->ref_count);
  return pyre_ok_status();
}

pyre_status_t pyre_allocator_release(pyre_allocator_t allocator) {
  if (!allocator) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "allocator is NULL");
  }
  // The allocator is inline in the device — release just decrements the
  // refcount. The actual hal_allocator is released when the device is
  // destroyed. This release is only for explicit retain/release pairs.
  if (iree_atomic_ref_count_dec(&allocator->ref_count) == 1) {
    // Refcount hit zero — but allocator is owned by device, not freed here.
  }
  return pyre_ok_status();
}

pyre_status_t pyre_allocator_allocate_buffer(pyre_allocator_t allocator,
                                             pyre_buffer_params_t params,
                                             size_t size,
                                             pyre_buffer_t* buffer) {
  if (!allocator || !buffer) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "allocator or buffer is NULL");
  }

  iree_hal_buffer_params_t hal_params = {
      .usage = (iree_hal_buffer_usage_t)params.usage,
      .access = (iree_hal_memory_access_t)params.access,
      .type = (iree_hal_memory_type_t)params.type,
      .queue_affinity = (iree_hal_queue_affinity_t)params.queue_affinity,
  };

  iree_hal_buffer_t* hal_buffer = NULL;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      allocator->hal_allocator, hal_params, (iree_device_size_t)size,
      &hal_buffer);
  if (!iree_status_is_ok(status)) {
    return pyre_status_from_iree(status);
  }

  pyre_buffer_t buf = NULL;
  iree_status_t alloc_status = iree_allocator_malloc(
      iree_allocator_system(), sizeof(pyre_buffer_s), (void**)&buf);
  if (!iree_status_is_ok(alloc_status)) {
    iree_hal_buffer_release(hal_buffer);
    return pyre_status_from_iree(alloc_status);
  }

  memset(buf, 0, sizeof(*buf));
  iree_atomic_ref_count_init(&buf->ref_count);
  buf->hal_buffer = hal_buffer;
  buf->device = allocator->device;
  buf->mem_type = params.type;
  buf->size = size;
  *buffer = buf;
  return pyre_ok_status();
}

pyre_status_t pyre_allocator_import_buffer(pyre_allocator_t allocator,
                                           pyre_buffer_params_t params,
                                           void* host_ptr, size_t size,
                                           pyre_buffer_t* buffer) {
  if (!allocator || !host_ptr || !buffer) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "allocator, host_ptr, or buffer is NULL");
  }

  iree_hal_buffer_params_t hal_params = {
      .usage = (iree_hal_buffer_usage_t)params.usage,
      .access = (iree_hal_memory_access_t)params.access,
      .type = (iree_hal_memory_type_t)params.type,
      .queue_affinity = (iree_hal_queue_affinity_t)params.queue_affinity,
  };

  iree_hal_external_buffer_t ext = {
      .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
      .flags = 0,
      .size = (iree_device_size_t)size,
      .handle.host_allocation.ptr = host_ptr,
  };

  iree_hal_buffer_t* hal_buffer = NULL;
  iree_status_t status = iree_hal_allocator_import_buffer(
      allocator->hal_allocator, hal_params, &ext,
      iree_hal_buffer_release_callback_null(), &hal_buffer);
  if (!iree_status_is_ok(status)) {
    return pyre_status_from_iree(status);
  }

  pyre_buffer_t buf = NULL;
  iree_status_t alloc_status = iree_allocator_malloc(
      iree_allocator_system(), sizeof(pyre_buffer_s), (void**)&buf);
  if (!iree_status_is_ok(alloc_status)) {
    iree_hal_buffer_release(hal_buffer);
    return pyre_status_from_iree(alloc_status);
  }

  memset(buf, 0, sizeof(*buf));
  iree_atomic_ref_count_init(&buf->ref_count);
  buf->hal_buffer = hal_buffer;
  buf->device = allocator->device;
  buf->mem_type = params.type;
  buf->size = size;
  *buffer = buf;
  return pyre_ok_status();
}

//===----------------------------------------------------------------------===//
// Virtual memory
//===----------------------------------------------------------------------===//

pyre_status_t pyre_allocator_query_virtual_memory(
    pyre_allocator_t allocator, pyre_memory_type_t mem_type, bool* supported,
    size_t* min_page_size, size_t* recommended_page_size) {
  if (!allocator || !supported) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "allocator or supported is NULL");
  }

  *supported = iree_hal_allocator_supports_virtual_memory(
      allocator->hal_allocator);
  if (!*supported) {
    if (min_page_size) *min_page_size = 0;
    if (recommended_page_size) *recommended_page_size = 0;
    return pyre_ok_status();
  }

  iree_hal_buffer_params_t hal_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = (iree_hal_memory_type_t)mem_type,
  };
  iree_device_size_t iree_min = 0, iree_rec = 0;
  iree_status_t status =
      iree_hal_allocator_virtual_memory_query_granularity(
          allocator->hal_allocator, hal_params, &iree_min, &iree_rec);
  if (!iree_status_is_ok(status)) {
    *supported = false;
    if (min_page_size) *min_page_size = 0;
    if (recommended_page_size) *recommended_page_size = 0;
    iree_status_ignore(status);
    return pyre_ok_status();
  }

  if (min_page_size) *min_page_size = (size_t)iree_min;
  if (recommended_page_size) *recommended_page_size = (size_t)iree_rec;
  return pyre_ok_status();
}

pyre_status_t pyre_allocator_virtual_memory_reserve(
    pyre_allocator_t allocator, pyre_queue_affinity_t affinity, size_t size,
    pyre_buffer_t* virtual_buffer) {
  if (!allocator || !virtual_buffer) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "NULL argument");
  }

  iree_hal_buffer_t* hal_buffer = NULL;
  iree_status_t status = iree_hal_allocator_virtual_memory_reserve(
      allocator->hal_allocator, (iree_hal_queue_affinity_t)affinity,
      (iree_device_size_t)size, &hal_buffer);
  if (!iree_status_is_ok(status)) {
    return pyre_status_from_iree(status);
  }

  pyre_buffer_t buf = NULL;
  iree_status_t alloc_status = iree_allocator_malloc(
      iree_allocator_system(), sizeof(pyre_buffer_s), (void**)&buf);
  if (!iree_status_is_ok(alloc_status)) {
    iree_hal_allocator_virtual_memory_release(allocator->hal_allocator,
                                              hal_buffer);
    return pyre_status_from_iree(alloc_status);
  }

  memset(buf, 0, sizeof(*buf));
  iree_atomic_ref_count_init(&buf->ref_count);
  buf->hal_buffer = hal_buffer;
  buf->device = allocator->device;
  buf->mem_type = PYRE_MEMORY_TYPE_DEVICE_LOCAL;
  buf->size = size;
  *virtual_buffer = buf;
  return pyre_ok_status();
}

pyre_status_t pyre_allocator_virtual_memory_release(
    pyre_allocator_t allocator, pyre_buffer_t virtual_buffer) {
  if (!allocator || !virtual_buffer) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  iree_status_t status = iree_hal_allocator_virtual_memory_release(
      allocator->hal_allocator, virtual_buffer->hal_buffer);
  // Free the pyre wrapper (hal_buffer ownership transferred).
  virtual_buffer->hal_buffer = NULL;
  iree_allocator_free(iree_allocator_system(), virtual_buffer);
  return pyre_status_from_iree(status);
}

pyre_status_t pyre_allocator_physical_memory_allocate(
    pyre_allocator_t allocator, pyre_memory_type_t mem_type, size_t size,
    pyre_physical_memory_t* physical) {
  if (!allocator || !physical) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  iree_hal_buffer_params_t hal_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = (iree_hal_memory_type_t)mem_type,
  };
  return pyre_status_from_iree(iree_hal_allocator_physical_memory_allocate(
      allocator->hal_allocator, hal_params, (iree_device_size_t)size,
      iree_allocator_system(), (iree_hal_physical_memory_t**)physical));
}

pyre_status_t pyre_allocator_physical_memory_free(
    pyre_allocator_t allocator, pyre_physical_memory_t physical) {
  if (!allocator || !physical) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  return pyre_status_from_iree(iree_hal_allocator_physical_memory_free(
      allocator->hal_allocator, (iree_hal_physical_memory_t*)physical));
}

pyre_status_t pyre_allocator_virtual_memory_map(
    pyre_allocator_t allocator, pyre_buffer_t virtual_buffer,
    size_t virtual_offset, pyre_physical_memory_t physical,
    size_t physical_offset, size_t size) {
  if (!allocator || !virtual_buffer || !physical) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  return pyre_status_from_iree(iree_hal_allocator_virtual_memory_map(
      allocator->hal_allocator, virtual_buffer->hal_buffer,
      (iree_device_size_t)virtual_offset,
      (iree_hal_physical_memory_t*)physical,
      (iree_device_size_t)physical_offset, (iree_device_size_t)size));
}

pyre_status_t pyre_allocator_virtual_memory_unmap(
    pyre_allocator_t allocator, pyre_buffer_t virtual_buffer,
    size_t virtual_offset, size_t size) {
  if (!allocator || !virtual_buffer) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  return pyre_status_from_iree(iree_hal_allocator_virtual_memory_unmap(
      allocator->hal_allocator, virtual_buffer->hal_buffer,
      (iree_device_size_t)virtual_offset, (iree_device_size_t)size));
}

pyre_status_t pyre_allocator_virtual_memory_protect(
    pyre_allocator_t allocator, pyre_buffer_t virtual_buffer,
    size_t virtual_offset, size_t size, pyre_memory_protection_t protection) {
  if (!allocator || !virtual_buffer) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  return pyre_status_from_iree(iree_hal_allocator_virtual_memory_protect(
      allocator->hal_allocator, virtual_buffer->hal_buffer,
      (iree_device_size_t)virtual_offset, (iree_device_size_t)size,
      /*queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
      (iree_hal_memory_protection_t)protection));
}
