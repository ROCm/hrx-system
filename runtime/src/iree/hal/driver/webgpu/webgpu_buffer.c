// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/webgpu/webgpu_buffer.h"

#include "iree/hal/driver/webgpu/webgpu_imports.h"

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_buffer_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_webgpu_buffer_t {
  iree_hal_buffer_t base;
  iree_allocator_t host_allocator;

  // Prepared native binding table.
  iree_hal_buffer_native_binding_t
      bindings[IREE_HAL_WEBGPU_BUFFER_BINDING_COUNT];

  // Shadow buffer in wasm linear memory for host mapping. WebGPU mapped ranges
  // live in JS memory (not wasm linear memory), so we maintain a shadow that
  // the C code reads/writes. Data is synchronized between the shadow and the
  // GPU mapped range on map/unmap transitions.
  void* shadow_buffer;
  iree_device_size_t shadow_buffer_size;

  // Whether the GPU buffer is currently in the mapped state.
  bool is_mapped;

  // Whether the buffer was created with mappedAtCreation:true. These buffers
  // start in the mapped state and the shadow buffer is allocated at creation.
  bool mapped_at_creation;
} iree_hal_webgpu_buffer_t;

static const iree_hal_buffer_vtable_t iree_hal_webgpu_buffer_vtable;

const iree_hal_buffer_binding_layout_t* iree_hal_webgpu_buffer_binding_layout(
    void) {
  static const uint16_t types[] = {
      IREE_HAL_BUFFER_INTERFACE_WEBGPU_BUFFER,
  };
  static const iree_hal_buffer_binding_layout_t layout = {
      .byte_length = sizeof(iree_hal_buffer_native_binding_t) *
                     IREE_HAL_WEBGPU_BUFFER_BINDING_COUNT,
      .binding_count = IREE_ARRAYSIZE(types),
      .host_binding_index = IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE,
      .types = types,
  };
  return &layout;
}

static iree_hal_webgpu_buffer_t* iree_hal_webgpu_buffer_cast(
    iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_buffer_vtable);
  return (iree_hal_webgpu_buffer_t*)base_value;
}

static const iree_hal_webgpu_buffer_t* iree_hal_webgpu_buffer_const_cast(
    const iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_buffer_vtable);
  return (const iree_hal_webgpu_buffer_t*)base_value;
}

iree_status_t iree_hal_webgpu_buffer_create(
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_hal_webgpu_handle_t buffer_handle, bool mapped_at_creation,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_buffer = NULL;

  iree_hal_webgpu_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer));

  iree_hal_buffer_initialize(placement, &buffer->base, allocation_size,
                             /*byte_offset=*/0, /*byte_length=*/allocation_size,
                             memory_type, allowed_access, allowed_usage,
                             &iree_hal_webgpu_buffer_vtable, &buffer->base);
  buffer->host_allocator = host_allocator;
  buffer->bindings[IREE_HAL_WEBGPU_BUFFER_BINDING_RESOURCE].webgpu.buffer =
      buffer_handle;
  buffer->bindings[IREE_HAL_WEBGPU_BUFFER_BINDING_RESOURCE].webgpu.offset = 0;
  buffer->base.memory.bindings = buffer->bindings;
  buffer->base.memory.binding_offset = 0;
  buffer->shadow_buffer = NULL;
  buffer->shadow_buffer_size = 0;
  buffer->is_mapped = mapped_at_creation;
  buffer->mapped_at_creation = mapped_at_creation;

  // For mappedAtCreation buffers, allocate the shadow buffer immediately.
  // The host can write into the shadow, and on unmap we flush to the GPU.
  iree_status_t status = iree_ok_status();
  if (mapped_at_creation) {
    status =
        iree_allocator_malloc(host_allocator, (iree_host_size_t)allocation_size,
                              &buffer->shadow_buffer);
    if (iree_status_is_ok(status)) {
      buffer->shadow_buffer_size = allocation_size;
      // Pull the initial mapped contents from the GPU buffer into the shadow.
      // For mappedAtCreation the buffer is zero-initialized by WebGPU, so
      // this copies zeroes — but it ensures the shadow matches the GPU state.
      iree_hal_webgpu_import_buffer_get_mapped_range(
          buffer_handle, /*offset=*/0, /*size=*/(uint64_t)allocation_size,
          (uint32_t)(uintptr_t)buffer->shadow_buffer);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_buffer = &buffer->base;
  } else {
    // The caller retains native ownership unless wrapper creation succeeds.
    buffer->bindings[IREE_HAL_WEBGPU_BUFFER_BINDING_RESOURCE].webgpu.buffer =
        IREE_HAL_WEBGPU_HANDLE_NULL;
    iree_hal_buffer_release(&buffer->base);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_hal_webgpu_handle_t iree_hal_webgpu_buffer_handle(
    const iree_hal_buffer_t* base_buffer) {
  base_buffer = iree_hal_buffer_allocated_buffer(base_buffer);
  const iree_hal_webgpu_buffer_t* buffer =
      iree_hal_webgpu_buffer_const_cast(base_buffer);
  return (iree_hal_webgpu_handle_t)buffer
      ->bindings[IREE_HAL_WEBGPU_BUFFER_BINDING_RESOURCE]
      .webgpu.buffer;
}

static void iree_hal_webgpu_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_webgpu_buffer_t* buffer = iree_hal_webgpu_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_webgpu_handle_t buffer_handle =
      iree_hal_webgpu_buffer_handle(base_buffer);
  if (buffer_handle != IREE_HAL_WEBGPU_HANDLE_NULL) {
    // If the buffer is still mapped, unmap it.
    if (buffer->is_mapped) {
      iree_hal_webgpu_import_buffer_unmap(buffer_handle);
      buffer->is_mapped = false;
    }

    // Free the shadow buffer.
    if (buffer->shadow_buffer) {
      iree_allocator_free(host_allocator, buffer->shadow_buffer);
      buffer->shadow_buffer = NULL;
    }

    // Destroy the GPU buffer via the bridge.
    iree_hal_webgpu_import_buffer_destroy(buffer_handle);
  }

  iree_allocator_free(host_allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_webgpu_buffer_export_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length,
    iree_hal_external_storage_class_t requested_class,
    iree_hal_external_storage_t* out_storage,
    iree_hal_buffer_range_t* out_logical_range) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "WebGPU does not support external buffer export");
}

static iree_status_t iree_hal_webgpu_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access, iree_hal_buffer_map_flags_t flags,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_webgpu_buffer_t* buffer = iree_hal_webgpu_buffer_cast(base_buffer);

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(base_buffer),
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(base_buffer),
      mapping_mode == IREE_HAL_MAPPING_MODE_PERSISTENT
          ? IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT
          : IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED));

  if (!buffer->is_mapped) {
    // The GPU buffer is not in mapped state. For mappedAtCreation buffers this
    // means the buffer was already unmapped (the buffer was flushed to the GPU
    // and cannot be re-mapped for writing). For read-back buffers, mapAsync
    // must have been called and completed before map_range.
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "buffer is not in mapped state; for staging-write buffers "
        "mappedAtCreation mapping is single-use, for staging-read buffers "
        "mapAsync must complete before mapping");
  }

  // Ensure the shadow buffer covers the requested range.
  if (!buffer->shadow_buffer ||
      local_byte_offset + local_byte_length > buffer->shadow_buffer_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "requested mapping range [%" PRIu64 ", %" PRIu64
                            ") exceeds shadow buffer size %" PRIu64,
                            (uint64_t)local_byte_offset,
                            (uint64_t)(local_byte_offset + local_byte_length),
                            (uint64_t)buffer->shadow_buffer_size);
  }

  // For read access, pull data from the GPU mapped range into the shadow.
  if (iree_any_bit_set(memory_access, IREE_HAL_MEMORY_ACCESS_READ)) {
    iree_hal_webgpu_import_buffer_get_mapped_range(
        iree_hal_webgpu_buffer_handle(base_buffer), (uint64_t)local_byte_offset,
        (uint64_t)local_byte_length,
        (uint32_t)(uintptr_t)((uint8_t*)buffer->shadow_buffer +
                              local_byte_offset));
  }

  // Return a pointer into the shadow buffer.
  mapping->contents = iree_make_byte_span(
      (uint8_t*)buffer->shadow_buffer + local_byte_offset, local_byte_length);

  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  iree_hal_webgpu_buffer_t* buffer = iree_hal_webgpu_buffer_cast(base_buffer);

  if (!buffer->is_mapped) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "buffer is not mapped");
  }

  // For write access, flush the shadow buffer contents to the GPU mapped range.
  // We always flush the entire mapped region to keep the GPU state consistent.
  // The cost is a single memcpy through the bridge — the mapped range is JS
  // memory and will be uploaded to the GPU when the buffer is unmapped.
  iree_hal_webgpu_import_buffer_set_mapped_range(
      iree_hal_webgpu_buffer_handle(base_buffer), (uint64_t)local_byte_offset,
      (uint64_t)local_byte_length,
      (uint32_t)(uintptr_t)((uint8_t*)buffer->shadow_buffer +
                            local_byte_offset));

  // For mappedAtCreation buffers, unmap transitions the buffer to the
  // unmapped state permanently (it cannot be re-mapped for writing).
  // Unmap triggers the GPU upload of the staged data.
  if (buffer->mapped_at_creation) {
    iree_hal_webgpu_import_buffer_unmap(
        iree_hal_webgpu_buffer_handle(base_buffer));
    buffer->is_mapped = false;

    // Free the shadow buffer — the data has been flushed to the GPU.
    iree_allocator_free(buffer->host_allocator, buffer->shadow_buffer);
    buffer->shadow_buffer = NULL;
    buffer->shadow_buffer_size = 0;
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // WebGPU buffers are coherent — no explicit cache invalidation needed.
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // WebGPU buffers are coherent — no explicit cache flush needed.
  return iree_ok_status();
}

static const iree_hal_buffer_vtable_t iree_hal_webgpu_buffer_vtable = {
    .destroy = iree_hal_webgpu_buffer_destroy,
    .export_range = iree_hal_webgpu_buffer_export_range,
    .map_range = iree_hal_webgpu_buffer_map_range,
    .unmap_range = iree_hal_webgpu_buffer_unmap_range,
    .invalidate_range = iree_hal_webgpu_buffer_invalidate_range,
    .flush_range = iree_hal_webgpu_buffer_flush_range,
};
