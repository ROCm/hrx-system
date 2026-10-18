// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/sparse_buffer.h"

#include <string.h>

#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/hal/drivers/vulkan/queue.h"

VkMemoryRequirements iree_hal_vulkan_sparse_buffer_query_requirements(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_vulkan_features_t enabled_features,
    iree_hal_buffer_usage_t usage) {
  const VkBufferCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT |
               VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT |
               VK_BUFFER_CREATE_SPARSE_ALIASED_BIT,
      .size = 4,
      .usage = iree_hal_vulkan_buffer_usage_from_hal(enabled_features, usage),
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  const VkDeviceBufferMemoryRequirements query = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS,
      .pCreateInfo = &create_info,
  };
  VkMemoryRequirements2 requirements = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
  };
  iree_vkGetDeviceBufferMemoryRequirements(
      IREE_VULKAN_DEVICE(syms), logical_device, &query, &requirements);
  return requirements.memoryRequirements;
}

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_sparse_buffer_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vulkan_sparse_buffer_t {
  // Base HAL buffer resource returned to callers.
  iree_hal_buffer_t base;

  // Physical bytes in the owned blocks, independent of their virtual mapping.
  iree_hal_memory_trace_t allocation_trace;

  // Host allocator used to free wrapper storage and temporary bind arrays.
  iree_allocator_t host_allocator;

  // Device-level Vulkan dispatch table copied from the creating device.
  iree_hal_vulkan_device_syms_t syms;

  // Vulkan logical device that owns the sparse buffer and |physical_blocks|.
  VkDevice logical_device;

  // Inline native facts shared with dense and queue-ordered Vulkan storage.
  iree_hal_vulkan_buffer_native_t native;

  // Count of allocated physical VkDeviceMemory blocks.
  iree_host_size_t physical_block_count;

  // Physical VkDeviceMemory blocks bound into the sparse buffer in order.
  VkDeviceMemory physical_blocks[];
} iree_hal_vulkan_sparse_buffer_t;

static const iree_hal_buffer_vtable_t iree_hal_vulkan_sparse_buffer_vtable;

static iree_hal_vulkan_sparse_buffer_t* iree_hal_vulkan_sparse_buffer_cast(
    iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_sparse_buffer_vtable);
  return (iree_hal_vulkan_sparse_buffer_t*)base_value;
}

static iree_status_t iree_hal_vulkan_sparse_buffer_calculate_block_size(
    VkMemoryRequirements memory_requirements, VkDeviceSize max_allocation_size,
    VkDeviceSize* out_physical_block_size,
    iree_host_size_t* out_physical_block_count) {
  *out_physical_block_size = 0;
  *out_physical_block_count = 0;

  const VkDeviceSize alignment = memory_requirements.alignment;
  if (!alignment || !iree_device_size_is_valid_alignment(alignment)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan sparse buffer memory requirements reported invalid alignment "
        "%" PRIu64,
        (uint64_t)alignment);
  }

  const VkDeviceSize physical_block_size =
      iree_device_size_floor_div(max_allocation_size, alignment) * alignment;
  if (!physical_block_size) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "Vulkan maxMemoryAllocationSize %" PRIu64
                            " is smaller than sparse buffer alignment %" PRIu64,
                            (uint64_t)max_allocation_size, (uint64_t)alignment);
  }

  const uint64_t physical_block_count =
      iree_device_size_ceil_div(memory_requirements.size, physical_block_size);
  if (physical_block_count > IREE_HOST_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan sparse buffer requires %" PRIu64
        " physical blocks, which exceeds host-size indexing",
        physical_block_count);
  }
  if (physical_block_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan sparse buffer requires %" PRIu64
        " physical blocks, which exceeds VkSparseBufferMemoryBindInfo::"
        "bindCount",
        physical_block_count);
  }

  *out_physical_block_size = physical_block_size;
  *out_physical_block_count = (iree_host_size_t)physical_block_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_sparse_buffer_allocate_bind_array(
    iree_host_size_t physical_block_count, iree_allocator_t host_allocator,
    VkSparseMemoryBind** out_binds) {
  *out_binds = NULL;

  iree_host_size_t bind_array_size = 0;
  if (!iree_host_size_checked_mul(
          physical_block_count, sizeof(VkSparseMemoryBind), &bind_array_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan sparse buffer bind array overflows host-size storage");
  }
  return iree_allocator_malloc(host_allocator, bind_array_size,
                               (void**)out_binds);
}

static iree_status_t iree_hal_vulkan_sparse_buffer_allocate_physical_blocks(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    VkMemoryRequirements memory_requirements, uint32_t memory_type_index,
    VkDeviceSize physical_block_size, iree_host_size_t physical_block_count,
    VkMemoryAllocateFlags memory_allocate_flags,
    VkDeviceMemory out_physical_blocks[], VkSparseMemoryBind out_binds[]) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)memory_requirements.size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)physical_block_size);

  VkMemoryAllocateFlagsInfo allocate_flags_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags = memory_allocate_flags,
  };
  VkMemoryAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &allocate_flags_info,
      .memoryTypeIndex = memory_type_index,
  };

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < physical_block_count; ++i) {
    allocate_info.allocationSize =
        i + 1 < physical_block_count
            ? physical_block_size
            : memory_requirements.size - physical_block_size * i;
    IREE_TRACE_ZONE_BEGIN_NAMED(z1, "vkAllocateMemory");
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z1, (int64_t)allocate_info.allocationSize);
    status = iree_vkAllocateMemory(IREE_VULKAN_DEVICE(syms), logical_device,
                                   &allocate_info, /*pAllocator=*/NULL,
                                   &out_physical_blocks[i]);
    IREE_TRACE_ZONE_END(z1);
    if (iree_status_is_ok(status)) {
      out_binds[i] = (VkSparseMemoryBind){
          .resourceOffset = physical_block_size * i,
          .size = allocate_info.allocationSize,
          .memory = out_physical_blocks[i],
          .memoryOffset = 0,
          .flags = 0,
      };
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_sparse_buffer_free_physical_blocks(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_host_size_t physical_block_count, VkDeviceMemory physical_blocks[]) {
  for (iree_host_size_t i = 0; i < physical_block_count; ++i) {
    if (physical_blocks[i]) {
      iree_vkFreeMemory(IREE_VULKAN_DEVICE(syms), logical_device,
                        physical_blocks[i], /*pAllocator=*/NULL);
    }
  }
}

static void iree_hal_vulkan_sparse_buffer_free_unowned_handle(
    iree_hal_vulkan_sparse_buffer_t* buffer) {
  iree_hal_vulkan_sparse_buffer_free_physical_blocks(
      &buffer->syms, buffer->logical_device, buffer->physical_block_count,
      buffer->physical_blocks);
  iree_allocator_free(buffer->host_allocator, buffer);
}

static iree_status_t iree_hal_vulkan_sparse_buffer_commit_sync(
    iree_hal_vulkan_sparse_buffer_t* buffer,
    iree_hal_vulkan_queue_t* sparse_binding_queue,
    iree_hal_buffer_placement_t placement,
    VkMemoryRequirements memory_requirements, uint32_t memory_type_index,
    VkDeviceSize max_allocation_size,
    VkMemoryAllocateFlags memory_allocate_flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)memory_requirements.size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)memory_requirements.alignment);

  VkDeviceSize physical_block_size = 0;
  iree_host_size_t physical_block_count = 0;
  iree_status_t status = iree_hal_vulkan_sparse_buffer_calculate_block_size(
      memory_requirements, max_allocation_size, &physical_block_size,
      &physical_block_count);

  VkSparseMemoryBind* binds = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_sparse_buffer_allocate_bind_array(
        physical_block_count, buffer->host_allocator, &binds);
  }
  if (iree_status_is_ok(status)) {
    buffer->physical_block_count = physical_block_count;
    status = iree_hal_vulkan_sparse_buffer_allocate_physical_blocks(
        &buffer->syms, buffer->logical_device, memory_requirements,
        memory_type_index, physical_block_size, physical_block_count,
        memory_allocate_flags, buffer->physical_blocks, binds);
  }
  if (iree_status_is_ok(status)) {
    const VkSparseBufferMemoryBindInfo buffer_bind = {
        .buffer = (VkBuffer)buffer->native
                      .bindings[IREE_HAL_VULKAN_BUFFER_BINDING_RESOURCE]
                      .vulkan.buffer,
        .bindCount = (uint32_t)physical_block_count,
        .pBinds = binds,
    };
    status = iree_hal_vulkan_queue_submit_sparse_bind_sync(sparse_binding_queue,
                                                           1, &buffer_bind);
  }

  iree_allocator_free(buffer->host_allocator, binds);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_sparse_buffer_create_bound_sync(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_vulkan_queue_t* sparse_binding_queue,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_length, VkBuffer handle, VkDeviceSize handle_length,
    VkDeviceSize handle_offset, VkDeviceAddress device_address,
    VkMemoryRequirements memory_requirements, uint32_t memory_type_index,
    VkDeviceSize max_allocation_size,
    VkMemoryAllocateFlags memory_allocate_flags,
    const iree_hal_memory_trace_t* allocation_trace,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(sparse_binding_queue);
  IREE_ASSERT_ARGUMENT(placement.device);
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)allocation_size);

  VkDeviceSize physical_block_size = 0;
  iree_host_size_t physical_block_count = 0;
  iree_status_t status = iree_hal_vulkan_sparse_buffer_calculate_block_size(
      memory_requirements, max_allocation_size, &physical_block_size,
      &physical_block_count);

  iree_host_size_t total_size = 0;
  if (iree_status_is_ok(status) &&
      !iree_host_size_checked_mul_add(sizeof(iree_hal_vulkan_sparse_buffer_t),
                                      physical_block_count,
                                      sizeof(VkDeviceMemory), &total_size)) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan sparse buffer physical block table overflows host storage");
  }

  iree_hal_vulkan_sparse_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, total_size, (void**)&buffer);
  }
  if (iree_status_is_ok(status)) {
    memset(buffer, 0, total_size);
    iree_hal_buffer_initialize(
        placement, &buffer->base, allocation_size,
        /*byte_offset=*/0, byte_length, memory_type, allowed_access,
        allowed_usage, &iree_hal_vulkan_sparse_buffer_vtable, &buffer->base);
    buffer->host_allocator = host_allocator;
    buffer->syms = *syms;
    buffer->logical_device = logical_device;
    iree_hal_vulkan_buffer_initialize_bindings(
        &buffer->base, VK_NULL_HANDLE, handle, handle_length, device_address,
        handle_offset, &buffer->native);

    status = iree_hal_vulkan_sparse_buffer_commit_sync(
        buffer, sparse_binding_queue, placement, memory_requirements,
        memory_type_index, max_allocation_size, memory_allocate_flags);
  }

  if (iree_status_is_ok(status)) {
    buffer->allocation_trace = *allocation_trace;
    iree_hal_memory_trace_alloc(&buffer->allocation_trace, buffer,
                                memory_requirements.size);
    *out_buffer = &buffer->base;
  } else if (buffer) {
    iree_hal_vulkan_sparse_buffer_free_physical_blocks(
        &buffer->syms, buffer->logical_device, buffer->physical_block_count,
        buffer->physical_blocks);
    iree_allocator_free(host_allocator, buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_sparse_buffer_destroy(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_vulkan_sparse_buffer_t* buffer =
      iree_hal_vulkan_sparse_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0, (int64_t)iree_hal_buffer_allocation_size(base_buffer));

  const VkBuffer handle =
      (VkBuffer)buffer->native.bindings[IREE_HAL_VULKAN_BUFFER_BINDING_RESOURCE]
          .vulkan.buffer;
  if (handle) {
    iree_vkDestroyBuffer(IREE_VULKAN_DEVICE(&buffer->syms),
                         buffer->logical_device, handle,
                         /*pAllocator=*/NULL);
  }
  iree_hal_vulkan_sparse_buffer_free_physical_blocks(
      &buffer->syms, buffer->logical_device, buffer->physical_block_count,
      buffer->physical_blocks);
  iree_hal_memory_trace_free(&buffer->allocation_trace, buffer);
  iree_allocator_free(host_allocator, buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_vulkan_sparse_buffer_isa(iree_hal_buffer_t* buffer) {
  return iree_hal_resource_is((const iree_hal_resource_t*)buffer,
                              &iree_hal_vulkan_sparse_buffer_vtable);
}

static iree_status_t iree_hal_vulkan_sparse_buffer_export_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length,
    iree_hal_external_storage_class_t requested_class,
    iree_hal_external_storage_t* out_storage,
    iree_hal_buffer_range_t* out_logical_range) {
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "Vulkan sparse buffer does not support external export");
}

static iree_status_t iree_hal_vulkan_sparse_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access, iree_hal_buffer_map_flags_t flags,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  (void)base_buffer;
  (void)mapping_mode;
  (void)memory_access;
  (void)local_byte_offset;
  (void)local_byte_length;
  (void)mapping;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "Vulkan sparse buffers do not support mapping");
}

static iree_status_t iree_hal_vulkan_sparse_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  (void)base_buffer;
  (void)local_byte_offset;
  (void)local_byte_length;
  (void)mapping;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "Vulkan sparse buffers do not support mapping");
}

static iree_status_t iree_hal_vulkan_sparse_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  (void)base_buffer;
  (void)local_byte_offset;
  (void)local_byte_length;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "Vulkan sparse buffers do not support mapping");
}

static iree_status_t iree_hal_vulkan_sparse_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  (void)base_buffer;
  (void)local_byte_offset;
  (void)local_byte_length;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "Vulkan sparse buffers do not support mapping");
}

static const iree_hal_buffer_vtable_t iree_hal_vulkan_sparse_buffer_vtable = {
    .destroy = iree_hal_vulkan_sparse_buffer_destroy,
    .export_range = iree_hal_vulkan_sparse_buffer_export_range,
    .map_range = iree_hal_vulkan_sparse_buffer_map_range,
    .unmap_range = iree_hal_vulkan_sparse_buffer_unmap_range,
    .invalidate_range = iree_hal_vulkan_sparse_buffer_invalidate_range,
    .flush_range = iree_hal_vulkan_sparse_buffer_flush_range,
};
