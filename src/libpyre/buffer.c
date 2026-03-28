// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0
//
// Buffer allocation, mapping, and lifecycle.
// Adapted from iree-hal-streaming's memory.c allocation patterns.

#include "pyre_internal.h"

#include <stdlib.h>

static iree_hal_buffer_params_t pyre_mem_type_to_hal_params(
    pyre_memory_type_t mem_type) {
  iree_hal_buffer_params_t params = {0};
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;

  // All memory types include DEVICE_VISIBLE so the heap allocator grants
  // QUEUE_TRANSFER compatibility (required for command buffer fill/copy).
  switch (mem_type) {
    case PYRE_MEMORY_DEVICE_LOCAL:
      params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                    IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
      break;
    case PYRE_MEMORY_HOST_VISIBLE:
      params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                    IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
                    IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
      params.usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
      break;
    case PYRE_MEMORY_HOST_LOCAL:
      params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                    IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                    IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
      params.usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
      break;
    default:
      params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                    IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
      break;
  }
  return params;
}

pyre_status_t pyre_buffer_allocate(pyre_stream_t stream, size_t size,
                                   pyre_memory_type_t mem_type,
                                   pyre_buffer_t* buffer) {
  if (!stream || !buffer) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "stream or buffer is NULL");
  }
  if (size == 0) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "allocation size must be > 0");
  }

  pyre_buffer_s* buf = (pyre_buffer_s*)calloc(1, sizeof(pyre_buffer_s));
  if (!buf) {
    return pyre_make_status(PYRE_STATUS_OUT_OF_MEMORY,
                            "failed to allocate buffer wrapper");
  }

  iree_hal_buffer_params_t params = pyre_mem_type_to_hal_params(mem_type);

  iree_status_t status = iree_hal_allocator_allocate_buffer(
      stream->device->allocator, params, (iree_device_size_t)size,
      &buf->hal_buffer);
  if (!iree_status_is_ok(status)) {
    free(buf);
    return pyre_status_from_iree(status);
  }

  iree_atomic_ref_count_init(&buf->ref_count);
  buf->device = stream->device;
  buf->mem_type = mem_type;
  buf->size = size;
  buf->mapped_ptr = NULL;

  *buffer = buf;
  return pyre_ok_status();
}

pyre_status_t pyre_buffer_retain(pyre_buffer_t buffer) {
  if (!buffer) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "buffer is NULL");
  }
  iree_atomic_ref_count_inc(&buffer->ref_count);
  return pyre_ok_status();
}

pyre_status_t pyre_buffer_release(pyre_buffer_t buffer) {
  if (!buffer) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "buffer is NULL");
  }
  if (iree_atomic_ref_count_dec(&buffer->ref_count) == 1) {
    if (buffer->mapped_ptr) {
      iree_hal_buffer_unmap_range(
          &(iree_hal_buffer_mapping_t){.contents = {
              .data = buffer->mapped_ptr,
              .data_length = buffer->size,
          }});
    }
    if (buffer->hal_buffer) {
      iree_hal_buffer_release(buffer->hal_buffer);
    }
    free(buffer);
  }
  return pyre_ok_status();
}

pyre_status_t pyre_buffer_map(pyre_buffer_t buffer, pyre_map_flags_t flags,
                              size_t offset, size_t size, void** mapped_ptr) {
  if (!buffer || !mapped_ptr) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "buffer or mapped_ptr is NULL");
  }

  iree_hal_memory_access_t access = 0;
  if (flags & PYRE_MAP_READ) access |= IREE_HAL_MEMORY_ACCESS_READ;
  if (flags & PYRE_MAP_WRITE) access |= IREE_HAL_MEMORY_ACCESS_WRITE;
  if (flags & PYRE_MAP_DISCARD) access |= IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE;

  iree_hal_buffer_mapping_t mapping;
  iree_status_t status = iree_hal_buffer_map_range(
      buffer->hal_buffer, IREE_HAL_MAPPING_MODE_SCOPED, access,
      (iree_device_size_t)offset, (iree_device_size_t)size, &mapping);
  if (!iree_status_is_ok(status)) {
    return pyre_status_from_iree(status);
  }

  buffer->mapped_ptr = mapping.contents.data;
  *mapped_ptr = mapping.contents.data;
  return pyre_ok_status();
}

pyre_status_t pyre_buffer_unmap(pyre_buffer_t buffer) {
  if (!buffer) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "buffer is NULL");
  }
  if (!buffer->mapped_ptr) {
    return pyre_ok_status();  // Not mapped, no-op.
  }

  iree_hal_buffer_mapping_t mapping = {
      .contents = {
          .data = buffer->mapped_ptr,
          .data_length = buffer->size,
      },
  };
  iree_hal_buffer_unmap_range(&mapping);
  buffer->mapped_ptr = NULL;
  return pyre_ok_status();
}

pyre_status_t pyre_buffer_get_device_ptr(pyre_buffer_t buffer,
                                         void** device_ptr) {
  if (!buffer || !device_ptr) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "buffer or device_ptr is NULL");
  }
  // For local-task (CPU) devices, the device pointer is available via mapping.
  // For real GPU devices, this would use iree_hal_buffer_export.
  // For now, map the buffer to get a usable pointer.
  if (buffer->mapped_ptr) {
    *device_ptr = buffer->mapped_ptr;
    return pyre_ok_status();
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
    return pyre_ok_status();
  }

  iree_status_ignore(status);
  *device_ptr = NULL;
  return pyre_make_status(PYRE_STATUS_UNAVAILABLE,
                          "cannot get device pointer for this buffer type");
}
