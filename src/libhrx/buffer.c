// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Buffer allocation, mapping, and lifecycle.
// Adapted from iree-hal-streaming's memory.c allocation patterns.

#include "hrx_internal.h"

#include <string.h>

hrx_status_t hrx_buffer_allocate(hrx_stream_t stream, size_t size,
                                   hrx_memory_type_t mem_type,
                                   hrx_buffer_usage_t usage,
                                   hrx_buffer_t* buffer) {
  if (!stream || !buffer) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "stream or buffer is NULL");
  }
  if (size == 0) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "allocation size must be > 0");
  }

  hrx_buffer_s* buf = NULL;
  iree_status_t alloc_s = iree_allocator_malloc(
      iree_allocator_system(), sizeof(hrx_buffer_s), (void**)&buf);
  if (!iree_status_is_ok(alloc_s)) {
    return hrx_status_from_iree(alloc_s);
  }
  memset(buf, 0, sizeof(*buf));

  iree_hal_buffer_params_t params = {
      .type = (iree_hal_memory_type_t)mem_type,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .usage = (iree_hal_buffer_usage_t)usage,
  };

  hrx_status_t flush_status = hrx_stream_flush(stream);
  if (!hrx_status_is_ok(flush_status)) {
    iree_allocator_free(iree_allocator_system(), buf);
    return flush_status;
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

  iree_status_t status = iree_hal_device_queue_alloca(
      stream->device->hal_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list,
      signal_list, IREE_HAL_ALLOCATOR_POOL_DEFAULT, params,
      (iree_device_size_t)size, IREE_HAL_ALLOCA_FLAG_NONE,
      &buf->hal_buffer);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), buf);
    return hrx_status_from_iree(status);
  }

  iree_atomic_ref_count_init(&buf->ref_count);
  buf->device = stream->device;
  hrx_device_retain(buf->device);
  buf->mem_type = mem_type;
  buf->size = size;
  buf->mapped_ptr = NULL;
  stream->timepoint = signal_value;

  *buffer = buf;
  return hrx_ok_status();
}

void hrx_buffer_retain(hrx_buffer_t buffer) {
  iree_hal_buffer_retain(buffer->hal_buffer);
  hrx_device_retain(buffer->device);
  iree_atomic_ref_count_inc(&buffer->ref_count);
}

void hrx_buffer_release(hrx_buffer_t buffer) {
  iree_hal_buffer_t* hal_buffer = buffer->hal_buffer;
  hrx_device_t device = buffer->device;
  if (iree_atomic_ref_count_dec(&buffer->ref_count) == 1) {
    if (buffer->mapped_ptr) {
      iree_hal_buffer_unmap_range(
          &(iree_hal_buffer_mapping_t){.contents = {
              .data = buffer->mapped_ptr,
              .data_length = buffer->size,
          }});
    }
    iree_allocator_free(iree_allocator_system(), buffer);
  }
  iree_hal_buffer_release(hal_buffer);
  hrx_device_release(device);
}

hrx_status_t hrx_buffer_map(hrx_buffer_t buffer, hrx_map_flags_t flags,
                              size_t offset, size_t size, void** mapped_ptr) {
  if (!buffer || !mapped_ptr) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "buffer or mapped_ptr is NULL");
  }

  iree_hal_memory_access_t access = 0;
  if (flags & HRX_MAP_READ) access |= IREE_HAL_MEMORY_ACCESS_READ;
  if (flags & HRX_MAP_WRITE) access |= IREE_HAL_MEMORY_ACCESS_WRITE;
  if (flags & HRX_MAP_DISCARD) access |= IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE;

  iree_hal_buffer_mapping_t mapping;
  iree_status_t status = iree_hal_buffer_map_range(
      buffer->hal_buffer, IREE_HAL_MAPPING_MODE_SCOPED, access,
      (iree_device_size_t)offset, (iree_device_size_t)size, &mapping);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }

  buffer->mapped_ptr = mapping.contents.data;
  *mapped_ptr = mapping.contents.data;
  return hrx_ok_status();
}

hrx_status_t hrx_buffer_unmap(hrx_buffer_t buffer) {
  if (!buffer) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "buffer is NULL");
  }
  if (!buffer->mapped_ptr) {
    return hrx_ok_status();  // Not mapped, no-op.
  }

  iree_hal_buffer_mapping_t mapping = {
      .contents = {
          .data = buffer->mapped_ptr,
          .data_length = buffer->size,
      },
  };
  iree_hal_buffer_unmap_range(&mapping);
  buffer->mapped_ptr = NULL;
  return hrx_ok_status();
}

hrx_status_t hrx_buffer_get_device_ptr(hrx_buffer_t buffer,
                                         void** device_ptr) {
  if (!buffer || !device_ptr) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "buffer or device_ptr is NULL");
  }
  // For local-task (CPU) devices, the device pointer is available via mapping.
  // For real GPU devices, this would use iree_hal_buffer_export.
  // For now, map the buffer to get a usable pointer.
  if (buffer->mapped_ptr) {
    *device_ptr = buffer->mapped_ptr;
    return hrx_ok_status();
  }

  // Try to get a native allocation pointer.
  iree_hal_buffer_mapping_t mapping;
  iree_status_t status = iree_hal_buffer_map_range(
      buffer->hal_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_ALL,
      0, buffer->size, &mapping);
  if (iree_status_is_ok(status)) {
    *device_ptr = mapping.contents.data;
    buffer->mapped_ptr = mapping.contents.data;
    return hrx_ok_status();
  }

  iree_status_ignore(status);
  *device_ptr = NULL;
  return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                          "cannot get device pointer for this buffer type");
}

hrx_status_t hrx_buffer_get_size(hrx_buffer_t buffer, size_t* size) {
  if (!buffer || !size) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "buffer or size is NULL");
  }
  *size = buffer->size;
  return hrx_ok_status();
}
