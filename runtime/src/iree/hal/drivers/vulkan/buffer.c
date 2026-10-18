// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/buffer.h"

#include <string.h>

#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/vulkan/memory_backend.h"
#include "iree/hal/drivers/vulkan/sparse_buffer.h"
#include "iree/hal/drivers/vulkan/transient_buffer.h"

VkBufferUsageFlags iree_hal_vulkan_buffer_usage_from_hal(
    iree_hal_vulkan_features_t enabled_features,
    iree_hal_buffer_usage_t hal_usage) {
  VkBufferUsageFlags usage = 0;
  if (iree_all_bits_set(hal_usage, IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE)) {
    usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  }
  if (iree_all_bits_set(hal_usage, IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET)) {
    usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  }
  if (iree_any_bit_set(hal_usage,
                       IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ)) {
    usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  }
  if (iree_any_bit_set(hal_usage, IREE_HAL_BUFFER_USAGE_STORAGE)) {
    usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  }
  if (iree_any_bit_set(hal_usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
    // Device-side transfer polyfills bind buffers as storage buffers even when
    // the public HAL usage only exposes transfer operations.
    usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  }
  if (iree_any_bit_set(hal_usage,
                       IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS)) {
    usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  }
  if (iree_all_bits_set(
          enabled_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES) &&
      iree_any_bit_set(hal_usage, IREE_HAL_BUFFER_USAGE_DISPATCH)) {
    usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  }
  return usage;
}

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_buffer_t
//===----------------------------------------------------------------------===//

typedef uint32_t iree_hal_vulkan_buffer_ownership_t;
enum iree_hal_vulkan_buffer_ownership_bits_e {
  IREE_HAL_VULKAN_BUFFER_OWNS_NONE = 0u,
  IREE_HAL_VULKAN_BUFFER_OWNS_HANDLE = 1u << 0,
  IREE_HAL_VULKAN_BUFFER_OWNS_DEVICE_MEMORY = 1u << 1,
  IREE_HAL_VULKAN_BUFFER_OWNS_MAPPING_STATE = 1u << 2,
};

struct iree_hal_vulkan_buffer_mapping_state_t {
  // Mutex protecting |mapped_data| and |active_mapping_count|.
  iree_slim_mutex_t mutex;

  // Host pointer returned by vkMapMemory while the allocation is mapped.
  void* mapped_data;

  // Live HAL mappings plus the import owner's maintenance lease, when present.
  iree_host_size_t active_mapping_count;

  // Byte length of the dense VkDeviceMemory allocation.
  VkDeviceSize device_memory_size;
};

typedef struct iree_hal_vulkan_buffer_t {
  // Base HAL buffer resource returned to callers.
  iree_hal_buffer_t base;

  // Physical allocation ledger, absent for imported storage and views.
  iree_hal_memory_trace_t allocation_trace;

  // Host allocator used to free wrapper storage.
  iree_allocator_t host_allocator;

  // Device-level Vulkan dispatch table copied from the creating device.
  iree_hal_vulkan_device_syms_t syms;

  // Vulkan logical device that owns the native buffer and memory.
  VkDevice logical_device;

  // Byte offset within native device memory where this allocation begins.
  VkDeviceSize device_memory_offset;

  // Original host pointer supplied to native import, never a vkMapMemory alias.
  void* imported_host_pointer;

  // Shared mapping state for the dense device memory.
  iree_hal_vulkan_buffer_mapping_state_t* mapping_state;

  // Mapping state storage for buffers that own dense device memory.
  iree_hal_vulkan_buffer_mapping_state_t inline_mapping_state;

  // Inline native facts borrowed by ordinary buffer views.
  iree_hal_vulkan_buffer_native_t native;

  // Vulkan memory properties for map/flush policy.
  VkMemoryPropertyFlags memory_property_flags;

  // Exact native type of owned dense memory; absent for borrowed views.
  uint32_t memory_type_index;

  // Physical device nonCoherentAtomSize used for mapped-memory ranges.
  VkDeviceSize non_coherent_atom_size;

  // Vulkan resource ownership bits controlling destroy-time cleanup.
  iree_hal_vulkan_buffer_ownership_t ownership;

  // Optional callback issued after Vulkan resources are released.
  iree_hal_buffer_release_callback_t release_callback;
} iree_hal_vulkan_buffer_t;

static const iree_hal_buffer_vtable_t iree_hal_vulkan_buffer_vtable;

static iree_hal_vulkan_buffer_t* iree_hal_vulkan_buffer_cast(
    iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_buffer_vtable);
  return (iree_hal_vulkan_buffer_t*)base_value;
}

iree_hal_buffer_allocation_info_t iree_hal_vulkan_buffer_allocation_info(
    iree_hal_buffer_t* base_buffer) {
  const iree_hal_vulkan_buffer_t* buffer =
      iree_hal_vulkan_buffer_cast(base_buffer);
  return (iree_hal_buffer_allocation_info_t){
      .byte_length = buffer->mapping_state->device_memory_size,
      .byte_offset = buffer->device_memory_offset,
  };
}

void iree_hal_vulkan_buffer_prepare_mapping_source(
    iree_hal_buffer_t* base_buffer, iree_hal_buffer_t* base_source,
    iree_device_size_t granularity) {
  iree_hal_vulkan_buffer_t* buffer = iree_hal_vulkan_buffer_cast(base_buffer);
  iree_hal_vulkan_buffer_t* source = iree_hal_vulkan_buffer_cast(base_source);
  buffer->native.bindings[IREE_HAL_VULKAN_BUFFER_BINDING_MAPPING_SOURCE] =
      (iree_hal_buffer_native_binding_t){
          .mapping_source =
              {
                  .descriptor = source,
                  .offset = source->device_memory_offset -
                            base_source->memory.binding_offset,
                  .granularity = (uint32_t)granularity,
              },
      };
}

iree_hal_vulkan_mapping_source_info_t
iree_hal_vulkan_buffer_mapping_source_info(const void* descriptor) {
  const iree_hal_vulkan_buffer_t* source = descriptor;
  return (iree_hal_vulkan_mapping_source_info_t){
      .logical_device = source->logical_device,
      .memory = source->native.device_memory,
      .memory_type_index = source->memory_type_index,
      .byte_length = source->mapping_state->device_memory_size,
  };
}

const iree_hal_buffer_binding_layout_t* iree_hal_vulkan_buffer_binding_layout(
    void) {
  static const uint16_t types[] = {
      IREE_HAL_BUFFER_INTERFACE_VULKAN_BUFFER,
      IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
      IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
  };
  static const iree_hal_buffer_binding_layout_t layout = {
      .byte_length = sizeof(iree_hal_vulkan_buffer_native_t),
      .binding_count = IREE_ARRAYSIZE(types),
      .host_binding_index = IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE,
      .types = types,
  };
  return &layout;
}

void iree_hal_vulkan_buffer_initialize_bindings(
    iree_hal_buffer_t* buffer, VkDeviceMemory device_memory, VkBuffer handle,
    VkDeviceSize handle_length, VkDeviceAddress device_address,
    VkDeviceSize handle_offset, iree_hal_vulkan_buffer_native_t* native) {
  *native = (iree_hal_vulkan_buffer_native_t){
      .bindings =
          {
              [IREE_HAL_VULKAN_BUFFER_BINDING_RESOURCE] =
                  {
                      .vulkan = {.buffer = (uint64_t)handle},
                  },
              [IREE_HAL_VULKAN_BUFFER_BINDING_DEVICE_ADDRESS] =
                  {
                      .device_address = device_address,
                  },
          },
      .device_memory = device_memory,
      .handle_length = handle_length,
  };
  buffer->memory.bindings = native->bindings;
  buffer->memory.binding_offset = handle_offset + buffer->byte_offset;
}

static void iree_hal_vulkan_buffer_mapping_state_initialize(
    VkDeviceSize device_memory_size,
    iree_hal_vulkan_buffer_mapping_state_t* out_mapping_state) {
  memset(out_mapping_state, 0, sizeof(*out_mapping_state));
  iree_slim_mutex_initialize(&out_mapping_state->mutex);
  out_mapping_state->device_memory_size = device_memory_size;
}

static void iree_hal_vulkan_buffer_mapping_state_deinitialize(
    iree_hal_vulkan_buffer_t* buffer,
    iree_hal_vulkan_buffer_mapping_state_t* mapping_state) {
  iree_slim_mutex_lock(&mapping_state->mutex);
  if (mapping_state->mapped_data) {
    iree_vkUnmapMemory(IREE_VULKAN_DEVICE(&buffer->syms),
                       buffer->logical_device, buffer->native.device_memory);
    mapping_state->mapped_data = NULL;
    mapping_state->active_mapping_count = 0;
  }
  iree_slim_mutex_unlock(&mapping_state->mutex);
  iree_slim_mutex_deinitialize(&mapping_state->mutex);
}

static iree_status_t iree_hal_vulkan_buffer_create_internal(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    VkMemoryPropertyFlags memory_property_flags, uint32_t memory_type_index,
    VkDeviceSize non_coherent_atom_size, VkDeviceMemory device_memory,
    VkDeviceSize handle_offset, VkDeviceSize device_memory_offset,
    VkDeviceSize device_memory_size,
    iree_hal_vulkan_buffer_mapping_state_t* borrowed_mapping_state,
    VkBuffer handle, VkDeviceSize handle_length, VkDeviceAddress device_address,
    void* imported_host_pointer, iree_hal_buffer_t* host_allocation,
    iree_hal_vulkan_buffer_ownership_t ownership,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)byte_length);

  if (IREE_UNLIKELY(byte_offset > allocation_size ||
                    byte_length > allocation_size - byte_offset)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan buffer byte range offset %" PRIdsz
                            " with length %" PRIdsz
                            " exceeds allocation size %" PRIdsz,
                            byte_offset, byte_length, allocation_size);
  }

  iree_hal_vulkan_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer));
  memset(buffer, 0, sizeof(*buffer));
  iree_hal_buffer_initialize(placement, &buffer->base, allocation_size,
                             byte_offset, byte_length, memory_type,
                             allowed_access, allowed_usage,
                             &iree_hal_vulkan_buffer_vtable, &buffer->base);
  buffer->host_allocator = host_allocator;
  buffer->syms = *syms;
  buffer->logical_device = logical_device;
  buffer->device_memory_offset = device_memory_offset;
  buffer->imported_host_pointer = imported_host_pointer;
  if (iree_any_bit_set(ownership, IREE_HAL_VULKAN_BUFFER_OWNS_MAPPING_STATE)) {
    iree_hal_vulkan_buffer_mapping_state_initialize(
        device_memory_size, &buffer->inline_mapping_state);
    buffer->mapping_state = &buffer->inline_mapping_state;
  } else {
    buffer->mapping_state = borrowed_mapping_state;
  }
  iree_hal_vulkan_buffer_initialize_bindings(
      &buffer->base, device_memory, handle, handle_length, device_address,
      handle_offset, &buffer->native);
  buffer->native.host_allocation =
      imported_host_pointer ? &buffer->base : host_allocation;
  buffer->memory_property_flags = memory_property_flags;
  buffer->memory_type_index = memory_type_index;
  buffer->non_coherent_atom_size =
      non_coherent_atom_size ? non_coherent_atom_size : 1;
  buffer->ownership = ownership;
  buffer->release_callback = release_callback;

  iree_status_t status = iree_ok_status();
  if (imported_host_pointer &&
      !iree_any_bit_set(memory_property_flags,
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
    status = iree_vkMapMemory(IREE_VULKAN_DEVICE(&buffer->syms), logical_device,
                              device_memory,
                              /*offset=*/0, device_memory_size, /*flags=*/0,
                              &buffer->mapping_state->mapped_data);
    if (iree_status_is_ok(status)) {
      // A construction-time lease keeps maintenance valid independently of
      // public mappings. Teardown releases it with the native allocation.
      buffer->mapping_state->active_mapping_count = 1;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_buffer = &buffer->base;
  } else {
    iree_hal_vulkan_buffer_mapping_state_deinitialize(buffer,
                                                      buffer->mapping_state);
    iree_allocator_free(host_allocator, buffer);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_buffer_create(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_length, VkMemoryPropertyFlags memory_property_flags,
    uint32_t memory_type_index, VkDeviceSize non_coherent_atom_size,
    VkDeviceMemory device_memory, VkDeviceSize device_memory_offset,
    VkDeviceSize device_memory_size, VkBuffer handle,
    VkDeviceSize handle_length, VkDeviceSize handle_offset,
    VkDeviceAddress device_address, void* imported_host_pointer,
    const iree_hal_memory_trace_t* allocation_trace,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  iree_status_t status = iree_hal_vulkan_buffer_create_internal(
      syms, logical_device, placement, memory_type, allowed_access,
      allowed_usage, allocation_size, /*byte_offset=*/0, byte_length,
      memory_property_flags, memory_type_index, non_coherent_atom_size,
      device_memory, handle_offset, device_memory_offset, device_memory_size,
      /*borrowed_mapping_state=*/NULL, handle, handle_length, device_address,
      imported_host_pointer, /*host_allocation=*/NULL,
      IREE_HAL_VULKAN_BUFFER_OWNS_HANDLE |
          IREE_HAL_VULKAN_BUFFER_OWNS_DEVICE_MEMORY |
          IREE_HAL_VULKAN_BUFFER_OWNS_MAPPING_STATE,
      release_callback, host_allocator, out_buffer);
  if (iree_status_is_ok(status) && allocation_trace) {
    iree_hal_vulkan_buffer_t* buffer = iree_hal_vulkan_buffer_cast(*out_buffer);
    buffer->allocation_trace = *allocation_trace;
    iree_hal_memory_trace_alloc(&buffer->allocation_trace, buffer,
                                device_memory_size);
  }
  return status;
}

iree_status_t iree_hal_vulkan_buffer_create_borrowed(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t handle_offset, iree_device_size_t byte_length,
    VkMemoryPropertyFlags memory_property_flags,
    VkDeviceSize non_coherent_atom_size, VkDeviceMemory device_memory,
    iree_hal_vulkan_buffer_mapping_state_t* mapping_state, VkBuffer handle,
    VkDeviceAddress device_address, iree_hal_buffer_t* host_allocation,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  if (IREE_UNLIKELY(handle_offset > allocation_size ||
                    byte_length > allocation_size - handle_offset)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan borrowed buffer handle range offset %" PRIdsz
        " with length %" PRIdsz " exceeds allocation size %" PRIdsz,
        handle_offset, byte_length, allocation_size);
  }
  return iree_hal_vulkan_buffer_create_internal(
      syms, logical_device, placement, memory_type, allowed_access,
      allowed_usage, byte_length, /*byte_offset=*/0, byte_length,
      memory_property_flags, /*memory_type_index=*/UINT32_MAX,
      non_coherent_atom_size, device_memory, (VkDeviceSize)handle_offset,
      (VkDeviceSize)handle_offset,
      mapping_state ? mapping_state->device_memory_size : 0, mapping_state,
      handle, (VkDeviceSize)allocation_size, device_address,
      /*imported_host_pointer=*/NULL, host_allocation,
      IREE_HAL_VULKAN_BUFFER_OWNS_NONE, release_callback, host_allocator,
      out_buffer);
}

static void iree_hal_vulkan_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_vulkan_buffer_t* buffer = iree_hal_vulkan_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0, (int64_t)iree_hal_buffer_allocation_size(base_buffer));

  if (iree_any_bit_set(buffer->ownership,
                       IREE_HAL_VULKAN_BUFFER_OWNS_MAPPING_STATE)) {
    iree_hal_vulkan_buffer_mapping_state_deinitialize(buffer,
                                                      buffer->mapping_state);
  }
  const VkBuffer handle =
      (VkBuffer)buffer->native.bindings[IREE_HAL_VULKAN_BUFFER_BINDING_RESOURCE]
          .vulkan.buffer;
  if (handle &&
      iree_any_bit_set(buffer->ownership, IREE_HAL_VULKAN_BUFFER_OWNS_HANDLE)) {
    iree_vkDestroyBuffer(IREE_VULKAN_DEVICE(&buffer->syms),
                         buffer->logical_device, handle,
                         /*pAllocator=*/NULL);
  }
  if (buffer->native.device_memory &&
      iree_any_bit_set(buffer->ownership,
                       IREE_HAL_VULKAN_BUFFER_OWNS_DEVICE_MEMORY)) {
    iree_vkFreeMemory(IREE_VULKAN_DEVICE(&buffer->syms), buffer->logical_device,
                      buffer->native.device_memory,
                      /*pAllocator=*/NULL);
  }
  iree_hal_memory_trace_free(&buffer->allocation_trace, buffer);
  if (buffer->release_callback.fn) {
    buffer->release_callback.fn(buffer->release_callback.user_data,
                                base_buffer);
  }

  iree_allocator_free(host_allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_vulkan_buffer_isa(iree_hal_buffer_t* buffer) {
  return iree_hal_resource_is((const iree_hal_resource_t*)buffer,
                              &iree_hal_vulkan_buffer_vtable);
}

bool iree_hal_vulkan_buffer_is_native(iree_hal_buffer_t* allocated_buffer) {
  return iree_hal_vulkan_buffer_isa(allocated_buffer) ||
         iree_hal_vulkan_sparse_buffer_isa(allocated_buffer) ||
         iree_hal_vulkan_transient_buffer_isa(allocated_buffer) ||
         iree_hal_vulkan_virtual_buffer_isa(allocated_buffer);
}

iree_hal_vulkan_buffer_mapping_state_t* iree_hal_vulkan_buffer_mapping_state(
    iree_hal_buffer_t* buffer) {
  if (!iree_hal_vulkan_buffer_isa(buffer)) {
    return NULL;
  }
  iree_hal_vulkan_buffer_t* vulkan_buffer = iree_hal_vulkan_buffer_cast(buffer);
  return vulkan_buffer->mapping_state;
}

VkMemoryPropertyFlags iree_hal_vulkan_buffer_memory_property_flags(
    iree_hal_buffer_t* buffer) {
  return iree_hal_vulkan_buffer_cast(buffer)->memory_property_flags;
}

iree_status_t iree_hal_vulkan_buffer_resolve_backing(
    iree_hal_buffer_t* buffer, iree_hal_buffer_t** out_backing_buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_backing_buffer);
  *out_backing_buffer = NULL;
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer);
  if (iree_hal_vulkan_transient_buffer_isa(allocated_buffer)) {
    iree_hal_buffer_t* backing_buffer =
        iree_hal_vulkan_transient_buffer_backing_buffer(allocated_buffer);
    if (!backing_buffer) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "transient buffer has no staged Vulkan backing; ensure the buffer "
          "was returned from queue_alloca before submitting dependent work");
    }
    buffer = backing_buffer;
    allocated_buffer = iree_hal_buffer_allocated_buffer(buffer);
  }
  if (!iree_hal_vulkan_buffer_is_native(allocated_buffer)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "buffer is not backed by the Vulkan HAL");
  }
  *out_backing_buffer = buffer;
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_buffer_resolve_backing_offset(
    iree_hal_buffer_t* buffer, iree_hal_buffer_t* backing_buffer,
    iree_device_size_t local_byte_offset,
    iree_device_size_t* out_backing_byte_offset) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(backing_buffer);
  IREE_ASSERT_ARGUMENT(out_backing_byte_offset);
  const iree_hal_vulkan_buffer_native_t* native =
      iree_hal_vulkan_buffer_native(backing_buffer);
  iree_device_size_t backing_byte_offset =
      native->bindings[IREE_HAL_VULKAN_BUFFER_BINDING_RESOURCE].vulkan.offset +
      backing_buffer->memory.binding_offset;
  if (backing_buffer != buffer &&
      !iree_device_size_checked_add(backing_byte_offset,
                                    iree_hal_buffer_byte_offset(buffer),
                                    &backing_byte_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan buffer backing offset overflows");
  }
  if (!iree_device_size_checked_add(backing_byte_offset, local_byte_offset,
                                    &backing_byte_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan buffer local offset overflows");
  }
  *out_backing_byte_offset = backing_byte_offset;
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_buffer_resolve_host_span(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length, iree_hal_vulkan_buffer_host_span_t* out_span) {
  *out_span = (iree_hal_vulkan_buffer_host_span_t){0};
  iree_hal_buffer_t* backing = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_resolve_backing(buffer, &backing));
  iree_hal_buffer_t* allocation =
      iree_hal_vulkan_buffer_native(backing)->host_allocation;
  if (!allocation) {
    return iree_ok_status();
  }
  iree_device_size_t allocation_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing_offset(
      buffer, backing, offset, &allocation_offset));
  const iree_hal_vulkan_buffer_t* owner =
      iree_hal_vulkan_buffer_cast(allocation);
  *out_span = (iree_hal_vulkan_buffer_host_span_t){
      .contents = iree_make_byte_span(
          (uint8_t*)owner->imported_host_pointer + allocation_offset,
          (iree_host_size_t)length),
      .allocation = allocation,
      .allocation_offset = allocation_offset,
      .coherent = iree_any_bit_set(owner->memory_property_flags,
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
  };
  return iree_ok_status();
}

iree_hal_vulkan_buffer_range_alignment_t
iree_hal_vulkan_buffer_range_dword_alignment(
    iree_hal_buffer_t* buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  const iree_device_size_t dword_size = sizeof(uint32_t);
  const iree_device_size_t buffer_byte_length =
      iree_hal_buffer_byte_length(buffer);
  if (local_byte_offset > buffer_byte_length) {
    return IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_UNKNOWN;
  }
  if (local_byte_length == IREE_HAL_WHOLE_BUFFER) {
    local_byte_length = buffer_byte_length - local_byte_offset;
  } else if (local_byte_length > buffer_byte_length - local_byte_offset) {
    return IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_UNKNOWN;
  }
  if (local_byte_length == 0) {
    return IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_ALIGNED;
  }

  iree_hal_buffer_t* backing_buffer = buffer;
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer);
  if (iree_hal_vulkan_transient_buffer_isa(allocated_buffer)) {
    backing_buffer =
        iree_hal_vulkan_transient_buffer_backing_buffer(allocated_buffer);
    if (!backing_buffer) {
      return IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_UNKNOWN;
    }
  }

  iree_device_size_t offset_remainder = local_byte_offset % dword_size;
  offset_remainder += backing_buffer->memory.binding_offset % dword_size;
  allocated_buffer = iree_hal_buffer_allocated_buffer(backing_buffer);
  if (!iree_hal_vulkan_buffer_is_native(allocated_buffer)) {
    return IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_UNKNOWN;
  }
  offset_remainder += iree_hal_vulkan_buffer_native(backing_buffer)
                          ->bindings[IREE_HAL_VULKAN_BUFFER_BINDING_RESOURCE]
                          .vulkan.offset %
                      dword_size;
  if (backing_buffer != buffer) {
    offset_remainder += iree_hal_buffer_byte_offset(buffer) % dword_size;
  }
  return offset_remainder % dword_size == 0 &&
                 local_byte_length % dword_size == 0
             ? IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_ALIGNED
             : IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_UNALIGNED;
}

iree_status_t iree_hal_vulkan_buffer_handle(iree_hal_buffer_t* buffer,
                                            VkDeviceMemory* out_memory,
                                            VkBuffer* out_handle) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_memory);
  IREE_ASSERT_ARGUMENT(out_handle);
  iree_hal_buffer_t* backing_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_resolve_backing(buffer, &backing_buffer));
  const iree_hal_vulkan_buffer_native_t* native =
      iree_hal_vulkan_buffer_native(backing_buffer);
  *out_memory = native->device_memory;
  *out_handle =
      (VkBuffer)native->bindings[IREE_HAL_VULKAN_BUFFER_BINDING_RESOURCE]
          .vulkan.buffer;
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_buffer_handle_length(
    iree_hal_buffer_t* buffer, VkDeviceSize* out_handle_length) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_handle_length);
  iree_hal_buffer_t* backing_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_resolve_backing(buffer, &backing_buffer));
  *out_handle_length =
      iree_hal_vulkan_buffer_native(backing_buffer)->handle_length;
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_buffer_device_address(
    iree_hal_buffer_t* buffer, VkDeviceAddress* out_device_address) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_device_address);
  iree_hal_buffer_t* backing_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_resolve_backing(buffer, &backing_buffer));
  const VkDeviceAddress base_address =
      iree_hal_vulkan_buffer_native(backing_buffer)
          ->bindings[IREE_HAL_VULKAN_BUFFER_BINDING_DEVICE_ADDRESS]
          .device_address;
  if (base_address == 0) {
    *out_device_address = 0;
    return iree_ok_status();
  }
  iree_device_size_t byte_offset = backing_buffer->memory.binding_offset;
  if (backing_buffer != buffer &&
      !iree_device_size_checked_add(byte_offset, buffer->byte_offset,
                                    &byte_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan buffer view offset overflows");
  }
  uint64_t device_address = 0;
  if (!iree_checked_add_u64(base_address, byte_offset, &device_address)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan buffer device address overflows");
  }
  *out_device_address = device_address;
  return iree_ok_status();
}

static bool iree_hal_vulkan_buffer_is_host_coherent(
    const iree_hal_vulkan_buffer_t* buffer) {
  return iree_all_bits_set(buffer->memory_property_flags,
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

static iree_status_t iree_hal_vulkan_buffer_make_mapped_memory_range(
    const iree_hal_vulkan_buffer_t* buffer,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    VkMappedMemoryRange* out_range) {
  if (!buffer->mapping_state) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan buffer has no dense device memory mapping state");
  }
  const VkDeviceSize atom_size = buffer->non_coherent_atom_size;
  const VkDeviceSize atom_mask = atom_size - 1;
  const VkDeviceSize device_memory_size =
      buffer->mapping_state->device_memory_size;
  if (IREE_UNLIKELY((VkDeviceSize)local_byte_offset > device_memory_size ||
                    buffer->device_memory_offset >
                        device_memory_size - (VkDeviceSize)local_byte_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan mapped-memory range offset overflows");
  }
  VkDeviceSize range_offset =
      buffer->device_memory_offset + (VkDeviceSize)local_byte_offset;
  if (IREE_UNLIKELY((VkDeviceSize)local_byte_length >
                    device_memory_size - range_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan mapped-memory range exceeds allocation");
  }
  VkDeviceSize range_end = range_offset + (VkDeviceSize)local_byte_length;
  range_offset &= ~atom_mask;
  if (range_end < device_memory_size) {
    range_end = atom_mask > device_memory_size - range_end
                    ? device_memory_size
                    : (range_end + atom_mask) & ~atom_mask;
  }
  if (range_end > device_memory_size) {
    range_end = device_memory_size;
  }
  *out_range = (VkMappedMemoryRange){
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = buffer->native.device_memory,
      .offset = range_offset,
      .size = range_end == device_memory_size ? VK_WHOLE_SIZE
                                              : range_end - range_offset,
  };
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_buffer_export_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length,
    iree_hal_external_storage_class_t requested_class,
    iree_hal_external_storage_t* out_storage,
    iree_hal_buffer_range_t* out_logical_range) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "Vulkan buffer does not support external export");
}

static iree_status_t iree_hal_vulkan_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access, iree_hal_buffer_map_flags_t flags,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_vulkan_buffer_t* buffer = iree_hal_vulkan_buffer_cast(base_buffer);
  (void)memory_access;

  if (!buffer->native.device_memory) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan buffer has no dense device memory to map");
  }
  if (!buffer->mapping_state) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan buffer has no dense device memory mapping state");
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(base_buffer),
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(base_buffer),
      mapping_mode == IREE_HAL_MAPPING_MODE_PERSISTENT
          ? IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT
          : IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED));
  if (local_byte_length == 0) {
    mapping->contents = iree_make_byte_span(NULL, 0);
    return iree_ok_status();
  }

  iree_hal_vulkan_buffer_mapping_state_t* mapping_state = buffer->mapping_state;
  if (IREE_UNLIKELY(
          (VkDeviceSize)local_byte_offset > mapping_state->device_memory_size ||
          buffer->device_memory_offset > mapping_state->device_memory_size -
                                             (VkDeviceSize)local_byte_offset ||
          (VkDeviceSize)local_byte_length >
              mapping_state->device_memory_size - buffer->device_memory_offset -
                  (VkDeviceSize)local_byte_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan mapped byte range exceeds VkDeviceMemory");
  }

  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&mapping_state->mutex);
  if (!mapping_state->mapped_data) {
    status =
        iree_vkMapMemory(IREE_VULKAN_DEVICE(&buffer->syms),
                         buffer->logical_device, buffer->native.device_memory,
                         /*offset=*/0, mapping_state->device_memory_size,
                         /*flags=*/0, &mapping_state->mapped_data);
  }
  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(mapping_state->active_mapping_count ==
                    IREE_HOST_SIZE_MAX)) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "Vulkan buffer mapping count overflow");
  }
  if (iree_status_is_ok(status)) {
    mapping_state->active_mapping_count += 1;
    mapping->impl.reserved[0] = (uint64_t)(uintptr_t)mapping_state;
    uint8_t* data = (uint8_t*)mapping_state->mapped_data +
                    buffer->device_memory_offset + local_byte_offset;
    mapping->contents = iree_make_byte_span(data, local_byte_length);
  }
  iree_slim_mutex_unlock(&mapping_state->mutex);
  IREE_RETURN_IF_ERROR(status);

#if !defined(NDEBUG)
  if (iree_any_bit_set(flags, IREE_HAL_BUFFER_MAP_FLAG_DISCARD)) {
    memset(mapping->contents.data, 0xCD, local_byte_length);
  }
#endif  // !defined(NDEBUG)

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  iree_hal_vulkan_buffer_t* buffer = iree_hal_vulkan_buffer_cast(base_buffer);
  (void)local_byte_offset;
  (void)local_byte_length;
  iree_hal_vulkan_buffer_mapping_state_t* mapping_state =
      (iree_hal_vulkan_buffer_mapping_state_t*)(uintptr_t)
          mapping->impl.reserved[0];
  if (!mapping_state) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan buffer mapping has no mapping state");
  }
  iree_slim_mutex_lock(&mapping_state->mutex);
  if (IREE_UNLIKELY(mapping_state->active_mapping_count == 0)) {
    iree_slim_mutex_unlock(&mapping_state->mutex);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan buffer mapping count underflow");
  }
  mapping_state->active_mapping_count -= 1;
  if (mapping_state->active_mapping_count == 0) {
    iree_vkUnmapMemory(IREE_VULKAN_DEVICE(&buffer->syms),
                       buffer->logical_device, buffer->native.device_memory);
    mapping_state->mapped_data = NULL;
  }
  iree_slim_mutex_unlock(&mapping_state->mutex);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_vulkan_buffer_t* buffer = iree_hal_vulkan_buffer_cast(base_buffer);
  if (local_byte_length == 0 ||
      iree_hal_vulkan_buffer_is_host_coherent(buffer)) {
    return iree_ok_status();
  }
  VkMappedMemoryRange range;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_make_mapped_memory_range(
      buffer, local_byte_offset, local_byte_length, &range));
  return iree_vkInvalidateMappedMemoryRanges(IREE_VULKAN_DEVICE(&buffer->syms),
                                             buffer->logical_device,
                                             /*memoryRangeCount=*/1, &range);
}

static iree_status_t iree_hal_vulkan_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_vulkan_buffer_t* buffer = iree_hal_vulkan_buffer_cast(base_buffer);
  if (local_byte_length == 0 ||
      iree_hal_vulkan_buffer_is_host_coherent(buffer)) {
    return iree_ok_status();
  }
  VkMappedMemoryRange range;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_make_mapped_memory_range(
      buffer, local_byte_offset, local_byte_length, &range));
  return iree_vkFlushMappedMemoryRanges(IREE_VULKAN_DEVICE(&buffer->syms),
                                        buffer->logical_device,
                                        /*memoryRangeCount=*/1, &range);
}

static const iree_hal_buffer_vtable_t iree_hal_vulkan_buffer_vtable = {
    .destroy = iree_hal_vulkan_buffer_destroy,
    .export_range = iree_hal_vulkan_buffer_export_range,
    .map_range = iree_hal_vulkan_buffer_map_range,
    .unmap_range = iree_hal_vulkan_buffer_unmap_range,
    .invalidate_range = iree_hal_vulkan_buffer_invalidate_range,
    .flush_range = iree_hal_vulkan_buffer_flush_range,
};

iree_status_t iree_hal_vulkan_buffer_flush_host_span(
    const iree_hal_vulkan_buffer_host_span_t* span) {
  return iree_hal_vulkan_buffer_flush_range(
      span->allocation, span->allocation_offset, span->contents.data_length);
}

iree_status_t iree_hal_vulkan_buffer_invalidate_host_span(
    const iree_hal_vulkan_buffer_host_span_t* span) {
  return iree_hal_vulkan_buffer_invalidate_range(
      span->allocation, span->allocation_offset, span->contents.data_length);
}
