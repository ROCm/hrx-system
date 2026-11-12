// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Buffer allocation, mapping, and lifecycle.
// Adapted from iree-hal-streaming's memory.c allocation patterns.

#include "buffer.h"

#include <string.h>

#include "device_group.h"
#include "hrx_internal.h"

IREE_STATIC_ASSERT_ENUM_EQ(HRX_BUFFER_INTERFACE_HOST,
                           IREE_HAL_BUFFER_INTERFACE_HOST,
                           "buffer interfaces must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_BUFFER_INTERFACE_DEVICE_ADDRESS,
                           IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                           "buffer interfaces must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_BUFFER_INTERFACE_VULKAN_BUFFER,
                           IREE_HAL_BUFFER_INTERFACE_VULKAN_BUFFER,
                           "buffer interfaces must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_BUFFER_INTERFACE_XDNA_SHIM_DMA,
                           IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
                           "buffer interfaces must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_BUFFER_INTERFACE_XDNA_FIRMWARE,
                           IREE_HAL_BUFFER_INTERFACE_XDNA_FIRMWARE,
                           "buffer interfaces must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_BUFFER_INTERFACE_RDMA,
                           IREE_HAL_BUFFER_INTERFACE_RDMA,
                           "buffer interfaces must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_BUFFER_INTERFACE_REGISTERED_IO,
                           IREE_HAL_BUFFER_INTERFACE_REGISTERED_IO,
                           "buffer interfaces must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_BUFFER_INTERFACE_REMOTE,
                           IREE_HAL_BUFFER_INTERFACE_REMOTE,
                           "buffer interfaces must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_BUFFER_INTERFACE_WEBGPU_BUFFER,
                           IREE_HAL_BUFFER_INTERFACE_WEBGPU_BUFFER,
                           "buffer interfaces must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_BUFFER_INTERFACE_MAPPING_SOURCE,
                           IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
                           "buffer interfaces must match HAL");

iree_status_t hrx_buffer_create_from_hal(iree_hal_buffer_t* hal_buffer,
                                         hrx_device_group_t group,
                                         hrx_memory_type_t mem_type,
                                         size_t size, void* device_ptr,
                                         hrx_buffer_t* out_buffer) {
  hrx_buffer_s* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(iree_allocator_system(),
                                             sizeof(*buffer), (void**)&buffer));
  memset(buffer, 0, sizeof(*buffer));
  iree_atomic_ref_count_init(&buffer->ref_count);
  buffer->hal_buffer = hal_buffer;
  if (hal_buffer) {
    iree_hal_buffer_retain(hal_buffer);
  }
  buffer->group = group;
  hrx_device_group_retain(group);
  buffer->mem_type = mem_type;
  buffer->size = size;
  buffer->device_ptr = device_ptr;
  *out_buffer = buffer;
  return iree_ok_status();
}

static iree_status_t hrx_buffer_unmap_internal(hrx_buffer_t buffer) {
  iree_status_t status = iree_hal_buffer_unmap_range(&buffer->mapping);
  buffer->is_mapped = false;
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

  iree_hal_pool_t* pool = NULL;
  iree_status_t status = hrx_mem_pool_backing_retain_allocation_source(
      stream->queue->device, mem_type, usage, &pool);
  if (iree_status_is_ok(status)) {
    const iree_hal_pool_reservation_request_t request =
        iree_hal_make_pool_reservation_request((iree_device_size_t)size,
                                               /*min_alignment=*/0);
    status =
        iree_hal_queue_alloca(stream->hal_queue, wait_list, signal_list, pool,
                              /*request_count=*/1, &request, &buf->hal_buffer);
  }
  if (iree_status_is_ok(status)) {
    // The returned buffer is immediately available for mapping and native
    // address queries, which require the queued backing to be committed.
    status = iree_hal_semaphore_wait(sem, signal_value, iree_infinite_timeout(),
                                     /*flags=*/0);
  }
  // The device owns convenience sources until group teardown. Buffers retain
  // the group, so no additional per-buffer pool reference is needed.
  iree_hal_pool_release(pool);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(buf->hal_buffer);
    iree_allocator_free(iree_allocator_system(), buf);
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  iree_atomic_ref_count_init(&buf->ref_count);
  buf->group = stream->queue->device->group;
  hrx_device_group_retain(buf->group);
  buf->mem_type = mem_type;
  buf->size = size;
  stream->timepoint = signal_value;

  *buffer = buf;
  HRX_RETURN_AND_END_ZONE(z0, hrx_ok_status());
}

void hrx_buffer_retain(hrx_buffer_t buffer) {
  if (!buffer) {
    return;
  }
  iree_atomic_ref_count_inc(&buffer->ref_count);
}

void hrx_buffer_release(hrx_buffer_t buffer) {
  if (!buffer) {
    return;
  }
  if (iree_atomic_ref_count_dec(&buffer->ref_count) != 1) {
    return;
  }
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_buffer_release");
  HRX_TRACE_ZONE_APPEND_BYTES(z0, buffer->size);
  if (buffer->is_mapped) {
    hrx_runtime_report_failure(buffer->group ? buffer->group->runtime : NULL,
                               NULL, hrx_buffer_unmap_internal(buffer));
  }
  iree_hal_buffer_release(buffer->hal_buffer);
  if (buffer->release_callback) {
    buffer->release_callback(buffer->release_user_data, buffer->size);
  }
  hrx_device_group_release(buffer->group);
  iree_allocator_free(iree_allocator_system(), buffer);
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
  *mapped_ptr = NULL;
  if (!buffer->hal_buffer) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_UNAVAILABLE,
                            "host-only registration has no HAL mapping"));
  }

  if (flags &
      ~(HRX_MAP_READ | HRX_MAP_WRITE | HRX_MAP_DISCARD | HRX_MAP_MAY_ALIAS)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                                                "unsupported map flags"));
  }
  iree_hal_memory_access_t access = IREE_HAL_MEMORY_ACCESS_NONE;
  iree_hal_buffer_map_flags_t map_flags = IREE_HAL_BUFFER_MAP_FLAG_NONE;
  if (flags & HRX_MAP_READ) {
    access |= IREE_HAL_MEMORY_ACCESS_READ;
  }
  if (flags & HRX_MAP_WRITE) {
    access |= IREE_HAL_MEMORY_ACCESS_WRITE;
  }
  if (flags & HRX_MAP_DISCARD) {
    access |= IREE_HAL_MEMORY_ACCESS_WRITE;
    map_flags |= IREE_HAL_BUFFER_MAP_FLAG_DISCARD;
  }

  if (flags & HRX_MAP_MAY_ALIAS) {
    map_flags |= IREE_HAL_BUFFER_MAP_FLAG_MAY_ALIAS;
  }

  iree_status_t status = iree_hal_buffer_map_range(
      buffer->hal_buffer, IREE_HAL_MAPPING_MODE_SCOPED, access, map_flags,
      (iree_device_size_t)offset, (iree_device_size_t)size, &buffer->mapping);
  if (!iree_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  buffer->is_mapped = true;
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

hrx_status_t hrx_buffer_query_address(hrx_buffer_t buffer,
                                      hrx_memory_scope_t scope,
                                      hrx_buffer_interface_t type,
                                      uint64_t* out_address) {
  if (!out_address) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "address output is NULL");
  }
  *out_address = 0;
  if (!buffer || !buffer->hal_buffer) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "buffer has no prepared memory contract");
  }
  if (type != HRX_BUFFER_INTERFACE_HOST &&
      type != HRX_BUFFER_INTERFACE_DEVICE_ADDRESS &&
      type != HRX_BUFFER_INTERFACE_XDNA_SHIM_DMA &&
      type != HRX_BUFFER_INTERFACE_XDNA_FIRMWARE) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "native type is not a scalar address");
  }
  const iree_hal_buffer_memory_view_t memory =
      iree_hal_buffer_memory_view(buffer->hal_buffer);
  if (!memory.contract || scope.domain != memory.contract->domain ||
      scope.id == IREE_HAL_MEMORY_SCOPE_ANY ||
      scope.id >= memory.contract->scope_count) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "address scope is outside the buffer memory domain");
  }
  const uint16_t index = memory.contract->scopes[scope.id].bindings[type];
  if (index == IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE) {
    return hrx_make_status(HRX_STATUS_PERMISSION_DENIED,
                           "address type is not prepared for this scope");
  }
  const iree_hal_buffer_native_binding_slot_t slot = {
      .index = index,
      .type = (uint16_t)type,
  };
  const iree_hal_buffer_native_binding_t binding =
      iree_hal_buffer_memory_native_binding(&memory, slot);
  *out_address = type == HRX_BUFFER_INTERFACE_HOST
                     ? (uint64_t)(uintptr_t)binding.host_pointer
                     : binding.device_address;
  return hrx_ok_status();
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

  hrx_buffer_t buffer = NULL;
  iree_status_t create_status = hrx_buffer_create_from_hal(
      NULL, device->group, HRX_MEMORY_TYPE_HOST_LOCAL, size, host_ptr, &buffer);
  if (!iree_status_is_ok(create_status)) {
    return hrx_status_from_iree(create_status);
  }

  uint64_t key = (uint64_t)(uintptr_t)host_ptr;
  hrx_status_t status = hrx_buffer_table_insert(&device->buffer_table, key,
                                                host_ptr, size, buffer, NULL);
  if (!hrx_status_is_ok(status)) {
    hrx_buffer_release(buffer);
  }
  return status;
}

hrx_status_t hrx_host_memory_unregister(hrx_device_t device, void* host_ptr) {
  if (!device || !host_ptr) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }

  uint64_t key = (uint64_t)(uintptr_t)host_ptr;
  hrx_buffer_table_entry_t entry = {0};
  hrx_status_t status = hrx_buffer_table_remove_reserved_if(
      &device->buffer_table, key, NULL, NULL, &entry, NULL);
  if (!hrx_status_is_ok(status)) {
    return status;
  }

  hrx_buffer_table_cancel_reserved_insert(&device->buffer_table);
  hrx_buffer_release(entry.buffer);
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
