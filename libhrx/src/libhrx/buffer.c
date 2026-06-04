// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Buffer allocation, mapping, and lifecycle.
// Adapted from iree-hal-streaming's memory.c allocation patterns.

#include <string.h>

#include "hrx_internal.h"

static iree_status_t hrx_buffer_unmap_internal(hrx_buffer_t buffer) {
  iree_status_t status = iree_hal_buffer_unmap_range(&buffer->mapping);
  buffer->is_mapped = false;
  buffer->mapped_ptr = NULL;
  return status;
}

hrx_status_t hrx_buffer_allocate(hrx_stream_t stream, size_t size,
                                 hrx_memory_type_t mem_type,
                                 hrx_buffer_usage_t usage,
                                 hrx_buffer_t* buffer) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_buffer_allocate");
  HRX_TRACE_ZONE_APPEND_BYTES(z0, size);
  if (!stream || !buffer) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                                                "stream or buffer is NULL"));
  }
  if (size == 0) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                                                "allocation size must be > 0"));
  }

  hrx_buffer_s* buf = NULL;
  iree_status_t alloc_s = iree_allocator_malloc(
      iree_allocator_system(), sizeof(hrx_buffer_s), (void**)&buf);
  if (!iree_status_is_ok(alloc_s)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(alloc_s));
  }
  memset(buf, 0, sizeof(*buf));

  iree_hal_buffer_params_t params = {
      .type = (iree_hal_memory_type_t)mem_type,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .usage = (iree_hal_buffer_usage_t)usage,
  };
  const iree_hal_buffer_compatibility_t compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          stream->device->allocator.hal_allocator, params,
          (iree_device_size_t)size, &params,
          /*out_allocation_size=*/NULL);
  if (!iree_all_bits_set(compatibility,
                         IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
    iree_allocator_free(iree_allocator_system(), buf);
    HRX_RETURN_AND_END_ZONE(
        z0,
        hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                        "buffer params are not allocatable on this device"));
  }

  hrx_status_t flush_status = hrx_stream_flush(stream);
  if (!hrx_status_is_ok(flush_status)) {
    iree_allocator_free(iree_allocator_system(), buf);
    HRX_RETURN_AND_END_ZONE(z0, flush_status);
  }

  uint64_t wait_value = stream->timepoint;
  uint64_t signal_value = stream->timepoint + 1;
  iree_hal_semaphore_t* sem = stream->semaphore->hal_semaphore;
  iree_hal_semaphore_list_t wait_list = {
      .count = (stream->timepoint > 0) ? 1 : 0,
      .semaphores = &sem,
      .payload_values = &wait_value,
  };
  iree_hal_semaphore_list_t signal_list = {
      .count = 1,
      .semaphores = &sem,
      .payload_values = &signal_value,
  };

  iree_status_t status = iree_ok_status();
  status = hrx_iree_exact_pool_create(stream->device->allocator.hal_allocator,
                                      params, &buf->hal_pool);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_alloca(
        stream->device->hal_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list,
        signal_list, buf->hal_pool, params, (iree_device_size_t)size,
        IREE_HAL_ALLOCA_FLAG_NONE, &buf->hal_buffer);
  }
  if (iree_status_is_ok(status)) {
    // The AMDGPU transient allocator resolves committed backing while recording
    // later command buffer operations, so make the queued alloca visible now.
    status = iree_hal_semaphore_wait(sem, signal_value, iree_infinite_timeout(),
                                     /*flags=*/0);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(buf->hal_buffer);
    iree_hal_pool_release(buf->hal_pool);
    iree_allocator_free(iree_allocator_system(), buf);
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  iree_atomic_ref_count_init(&buf->ref_count);
  buf->device = stream->device;
  hrx_device_retain(buf->device);
  buf->mem_type = mem_type;
  buf->size = size;
  buf->mapped_ptr = NULL;
  stream->timepoint = signal_value;

  *buffer = buf;
  HRX_RETURN_AND_END_ZONE(z0, hrx_ok_status());
}

void hrx_buffer_retain(hrx_buffer_t buffer) {
  iree_hal_buffer_retain(buffer->hal_buffer);
  iree_hal_pool_retain(buffer->hal_pool);
  hrx_device_retain(buffer->device);
  iree_atomic_ref_count_inc(&buffer->ref_count);
}

void hrx_buffer_release(hrx_buffer_t buffer) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_buffer_release");
  if (buffer) {
    HRX_TRACE_ZONE_APPEND_BYTES(z0, buffer->size);
  }
  iree_hal_buffer_t* hal_buffer = buffer->hal_buffer;
  iree_hal_pool_t* hal_pool = buffer->hal_pool;
  hrx_device_t device = buffer->device;
  if (iree_atomic_ref_count_dec(&buffer->ref_count) == 1) {
    if (buffer->is_mapped) {
      iree_status_ignore(hrx_buffer_unmap_internal(buffer));
    }
    iree_allocator_free(iree_allocator_system(), buffer);
  }
  iree_hal_buffer_release(hal_buffer);
  iree_hal_pool_release(hal_pool);
  hrx_device_release(device);
  HRX_TRACE_ZONE_END(z0);
}

hrx_status_t hrx_buffer_map(hrx_buffer_t buffer, hrx_map_flags_t flags,
                            size_t offset, size_t size, void** mapped_ptr) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_buffer_map");
  HRX_TRACE_ZONE_APPEND_BYTES(z0, size);
  if (!buffer || !mapped_ptr) {
    HRX_RETURN_AND_END_ZONE(z0,
                            hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                                            "buffer or mapped_ptr is NULL"));
  }
  if (buffer->is_mapped) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_make_status(HRX_STATUS_FAILED_PRECONDITION,
                                                "buffer is already mapped"));
  }

  iree_hal_memory_access_t access = 0;
  if (flags & HRX_MAP_READ) access |= IREE_HAL_MEMORY_ACCESS_READ;
  if (flags & HRX_MAP_WRITE) access |= IREE_HAL_MEMORY_ACCESS_WRITE;
  if (flags & HRX_MAP_DISCARD) access |= IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE;

  iree_status_t status = iree_hal_buffer_map_range(
      buffer->hal_buffer, IREE_HAL_MAPPING_MODE_SCOPED, access,
      (iree_device_size_t)offset, (iree_device_size_t)size, &buffer->mapping);
  if (!iree_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  buffer->is_mapped = true;
  buffer->mapped_ptr = buffer->mapping.contents.data;
  *mapped_ptr = buffer->mapping.contents.data;
  HRX_RETURN_AND_END_ZONE(z0, hrx_ok_status());
}

hrx_status_t hrx_buffer_unmap(hrx_buffer_t buffer) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_buffer_unmap");
  if (buffer) {
    HRX_TRACE_ZONE_APPEND_BYTES(z0, buffer->size);
  }
  if (!buffer) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "buffer is NULL"));
  }
  if (!buffer->is_mapped) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_ok_status());  // Not mapped, no-op.
  }

  iree_status_t status = hrx_buffer_unmap_internal(buffer);
  HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
}

hrx_status_t hrx_buffer_get_device_ptr(hrx_buffer_t buffer, void** device_ptr) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_buffer_get_device_ptr");
  if (buffer) {
    HRX_TRACE_ZONE_APPEND_BYTES(z0, buffer->size);
  }
  if (!buffer || !device_ptr) {
    HRX_RETURN_AND_END_ZONE(z0,
                            hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                                            "buffer or device_ptr is NULL"));
  }
  // For local-task (CPU) devices, the device pointer is available via mapping.
  // For real GPU devices, this would use iree_hal_buffer_export.
  // For now, map the buffer to get a usable pointer.
  if (buffer->mapped_ptr) {
    *device_ptr = buffer->mapped_ptr;
    HRX_RETURN_AND_END_ZONE(z0, hrx_ok_status());
  }

  // Try to get a native allocation pointer.
  iree_status_t status = iree_hal_buffer_map_range(
      buffer->hal_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_ALL, 0, buffer->size, &buffer->mapping);
  if (iree_status_is_ok(status)) {
    buffer->is_mapped = true;
    buffer->mapped_ptr = buffer->mapping.contents.data;
    *device_ptr = buffer->mapping.contents.data;
    HRX_RETURN_AND_END_ZONE(z0, hrx_ok_status());
  }

  iree_status_ignore(status);
  *device_ptr = NULL;
  HRX_RETURN_AND_END_ZONE(
      z0, hrx_make_status(HRX_STATUS_UNAVAILABLE,
                          "cannot get device pointer for this buffer type"));
}

hrx_status_t hrx_buffer_get_size(hrx_buffer_t buffer, size_t* size) {
  if (!buffer || !size) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "buffer or size is NULL");
  }
  *size = buffer->size;
  return hrx_ok_status();
}

hrx_status_t hrx_host_memory_register(hrx_device_t device, void* host_ptr,
                                      size_t size, uint32_t flags) {
  (void)flags;
  if (!device || !host_ptr || size == 0) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }

  hrx_buffer_s* buf = NULL;
  iree_status_t alloc_s = iree_allocator_malloc(
      iree_allocator_system(), sizeof(hrx_buffer_s), (void**)&buf);
  if (!iree_status_is_ok(alloc_s)) {
    return hrx_status_from_iree(alloc_s);
  }
  memset(buf, 0, sizeof(*buf));
  iree_atomic_ref_count_init(&buf->ref_count);
  buf->device = device;
  buf->mem_type = HRX_MEMORY_TYPE_HOST_LOCAL;
  buf->size = size;
  buf->mapped_ptr = host_ptr;
  buf->hal_buffer = NULL;

  uint64_t key = (uint64_t)(uintptr_t)host_ptr;
  hrx_status_t status = hrx_buffer_table_insert(&device->buffer_table, key,
                                                host_ptr, size, buf, NULL);
  if (!hrx_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), buf);
    return status;
  }
  return hrx_ok_status();
}

hrx_status_t hrx_host_memory_unregister(hrx_device_t device, void* host_ptr) {
  if (!device || !host_ptr) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }

  uint64_t key = (uint64_t)(uintptr_t)host_ptr;
  hrx_buffer_t buf = NULL;
  size_t offset = 0;
  hrx_status_t status =
      hrx_buffer_table_find(&device->buffer_table, key, &buf, &offset, NULL);
  if (!hrx_status_is_ok(status)) return status;

  hrx_buffer_table_remove(&device->buffer_table, key);
  if (buf) {
    iree_allocator_free(iree_allocator_system(), buf);
  }
  return hrx_ok_status();
}

hrx_status_t hrx_buffer_lookup(hrx_device_t device, const void* device_ptr,
                               hrx_buffer_t* buffer, size_t* offset) {
  if (!device || !device_ptr) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  return hrx_buffer_table_find(&device->buffer_table,
                               (uint64_t)(uintptr_t)device_ptr, buffer, offset,
                               NULL);
}
