// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/allocator.h"

#include <string.h>

#include "iree/async/notification.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/vulkan/atomic.h"
#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/hal/drivers/vulkan/device_plan.h"
#include "iree/hal/drivers/vulkan/queue.h"
#include "iree/hal/drivers/vulkan/sparse_buffer.h"
#include "iree/hal/memory/maintenance_thread.h"

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_allocator_t
//===----------------------------------------------------------------------===//

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char* IREE_HAL_VULKAN_ALLOCATOR_ID = "iree-hal-vulkan-unpooled";
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

typedef struct iree_hal_vulkan_allocator_memory_placement_t {
  // Vulkan memory type index selected for the allocation.
  uint32_t memory_type_index;

  // Vulkan memory property flags for |memory_type_index|.
  VkMemoryPropertyFlags memory_property_flags;

  // HAL memory type exposed by allocated buffers.
  iree_hal_memory_type_t memory_type;
} iree_hal_vulkan_allocator_memory_placement_t;

typedef struct iree_hal_vulkan_allocator_buffer_sharing_t {
  // Vulkan sharing mode derived from the buffer queue family affinity.
  VkSharingMode mode;

  // Number of queue_family_indices entries passed to Vulkan. This is zero for
  // exclusive sharing and greater than one for concurrent sharing.
  uint32_t queue_family_index_count;

  // Deduplicated Vulkan queue families used for concurrent sharing.
  uint32_t queue_family_indices[IREE_HAL_MAX_QUEUE_FAMILIES];
} iree_hal_vulkan_allocator_buffer_sharing_t;

typedef struct iree_hal_vulkan_allocator_virtual_memory_mapping_t {
  // Next mapping in the allocator-owned registry.
  struct iree_hal_vulkan_allocator_virtual_memory_mapping_t* next;

  // Virtual sparse buffer containing this mapped range.
  iree_hal_buffer_t* virtual_buffer;

  // Physical allocation bound into |virtual_buffer|.
  iree_hal_physical_memory_t* physical_memory;

  // Byte offset in |virtual_buffer| where the mapping begins.
  iree_device_size_t virtual_offset;

  // Byte offset in |physical_memory| where the mapping begins.
  iree_device_size_t physical_offset;

  // Byte length of the mapped range.
  iree_device_size_t size;
} iree_hal_vulkan_allocator_virtual_memory_mapping_t;

struct iree_hal_physical_memory_t {
  // Host allocator used to free this wrapper.
  iree_allocator_t host_allocator;

  // Device-level Vulkan dispatch table copied from the allocator.
  iree_hal_vulkan_device_syms_t syms;

  // Vulkan logical device that owns |device_memory|.
  VkDevice logical_device;

  // Allocator that owns registry synchronization for this handle.
  iree_hal_vulkan_allocator_t* owner_allocator;

  // Standalone device memory allocation.
  VkDeviceMemory device_memory;

  // Allocated physical memory byte length.
  iree_device_size_t allocation_size;

  // Total mapped byte count protected by the parent allocator registry mutex.
  iree_device_size_t mapped_size;

  // Vulkan memory type index used for |device_memory|.
  uint32_t memory_type_index;

  // HAL memory type exposed by this physical allocation.
  iree_hal_memory_type_t memory_type;
};

struct iree_hal_vulkan_allocator_t {
  // HAL resource header.
  iree_hal_resource_t resource;

  // Host allocator used for allocator-owned host allocations.
  iree_allocator_t host_allocator;

  // Parent logical device. Unowned; device owns this allocator.
  iree_hal_device_t* parent_device;

  // Device-level Vulkan dispatch table copied from the logical device.
  iree_hal_vulkan_device_syms_t syms;

  // Vulkan logical device that owns allocations.
  VkDevice logical_device;

  // Physical-device properties captured during logical-device creation.
  VkPhysicalDeviceProperties2 properties2;

  // Vulkan 1.1 property set including maxMemoryAllocationSize.
  VkPhysicalDeviceVulkan11Properties properties11;

  // VK_EXT_external_memory_host properties, if the extension is enabled.
  VkPhysicalDeviceExternalMemoryHostPropertiesEXT
      external_memory_host_properties;

  // Physical-device memory properties captured during logical-device creation.
  VkPhysicalDeviceMemoryProperties2 memory_properties2;

  // HAL features enabled on the logical device.
  iree_hal_vulkan_features_t enabled_features;

  // Device extension bits enabled on the logical device.
  iree_hal_vulkan_device_extensions_t enabled_extensions;

  // Queue family affinity bits supported by this logical device.
  iree_hal_queue_family_affinity_t queue_family_affinity_mask;

  // Number of initialized queue_families entries.
  iree_host_size_t queue_family_count;

  // Vulkan queue families addressable by canonical HAL family ordinal.
  iree_hal_vulkan_allocator_queue_family_t
      queue_families[IREE_HAL_MAX_QUEUE_FAMILIES];

  // Internal queue lane used to perform sparse memory binding. Borrowed.
  iree_hal_vulkan_queue_t* sparse_binding_queue;

  // Protects |virtual_memory_mappings| and physical-memory mapped sizes.
  iree_slim_mutex_t virtual_memory_mutex;

  // Registry of currently mapped sparse virtual memory ranges.
  iree_hal_vulkan_allocator_virtual_memory_mapping_t* virtual_memory_mappings;

  // Shared notification published when Vulkan pool capacity changes.
  iree_async_notification_t* memory_notification;

  // Shared cold memory worker independent of execution queue failure/lifetime.
  iree_hal_memory_maintenance_t* memory_maintenance;

  // Aggregate allocation statistics.
  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
};

static const iree_hal_allocator_vtable_t iree_hal_vulkan_allocator_vtable;

static iree_hal_vulkan_allocator_t* iree_hal_vulkan_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_allocator_vtable);
  return (iree_hal_vulkan_allocator_t*)base_value;
}

static bool iree_hal_vulkan_allocator_ranges_overlap(
    iree_device_size_t lhs_offset, iree_device_size_t lhs_size,
    iree_device_size_t rhs_offset, iree_device_size_t rhs_size) {
  const iree_device_size_t lhs_end = lhs_offset + lhs_size;
  const iree_device_size_t rhs_end = rhs_offset + rhs_size;
  return lhs_offset < rhs_end && rhs_offset < lhs_end;
}

static void iree_hal_vulkan_allocator_deinitialize_virtual_memory_registry(
    iree_hal_vulkan_allocator_t* allocator) {
  IREE_ASSERT(allocator->virtual_memory_mappings == NULL);
  iree_hal_vulkan_allocator_virtual_memory_mapping_t* mapping =
      allocator->virtual_memory_mappings;
  while (mapping) {
    iree_hal_vulkan_allocator_virtual_memory_mapping_t* next_mapping =
        mapping->next;
    iree_allocator_free(allocator->host_allocator, mapping);
    mapping = next_mapping;
  }
  allocator->virtual_memory_mappings = NULL;
  iree_slim_mutex_deinitialize(&allocator->virtual_memory_mutex);
}

static iree_status_t iree_hal_vulkan_allocator_validate_queue_family_mappings(
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_host_size_t queue_family_count,
    const iree_hal_vulkan_allocator_queue_family_t* queue_families) {
  if (queue_family_count == 0 ||
      queue_family_count > IREE_HAL_MAX_QUEUE_FAMILIES) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan allocator queue family mapping count %" PRIhsz
        " is outside the supported range [1, %" PRIhsz "]",
        queue_family_count, IREE_HAL_MAX_QUEUE_FAMILIES);
  }
  for (iree_host_size_t i = 0; i < queue_family_count; ++i) {
    const iree_hal_vulkan_allocator_queue_family_t* queue_family =
        &queue_families[i];
    if (queue_family->native_family_index >=
        physical_device->queue_family_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan allocator queue family mapping %" PRIhsz
          " selects family %u but the physical device exposes %u families",
          i, queue_family->native_family_index,
          physical_device->queue_family_count);
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (queue_families[j].native_family_index ==
          queue_family->native_family_index) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Vulkan allocator HAL queue family ordinals %" PRIhsz
            " and %" PRIhsz " both map to native family %u",
            j, i, queue_family->native_family_index);
      }
    }
  }
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_allocator_create(
    iree_hal_device_t* parent_device, const iree_hal_vulkan_device_syms_t* syms,
    VkDevice logical_device,
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_hal_vulkan_features_t enabled_features,
    iree_hal_vulkan_device_extensions_t enabled_extensions,
    iree_host_size_t queue_family_count,
    const iree_hal_vulkan_allocator_queue_family_t* queue_families,
    iree_hal_vulkan_queue_t* sparse_binding_queue,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(parent_device);
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_ASSERT_ARGUMENT(queue_families);
  IREE_ASSERT_ARGUMENT(out_allocator);
  *out_allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_allocator_validate_queue_family_mappings(
      physical_device, queue_family_count, queue_families));
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_allocator_t* allocator = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                (void**)&allocator));
  memset(allocator, 0, sizeof(*allocator));
  iree_hal_resource_initialize(&iree_hal_vulkan_allocator_vtable,
                               &allocator->resource);
  allocator->host_allocator = host_allocator;
  allocator->parent_device = parent_device;
  allocator->syms = *syms;
  allocator->logical_device = logical_device;
  allocator->properties2 = physical_device->properties2;
  allocator->properties2.pNext = NULL;
  allocator->properties11 = physical_device->properties11;
  allocator->properties11.pNext = NULL;
  allocator->external_memory_host_properties =
      physical_device->external_memory_host_properties;
  allocator->external_memory_host_properties.pNext = NULL;
  allocator->memory_properties2 = physical_device->memory_properties2;
  allocator->enabled_features = enabled_features;
  allocator->enabled_extensions = enabled_extensions;
  allocator->queue_family_affinity_mask =
      queue_family_count == IREE_HAL_MAX_QUEUE_FAMILIES
          ? IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY
          : (iree_hal_queue_family_affinity_t)((1ull << queue_family_count) -
                                               1ull);
  allocator->queue_family_count = queue_family_count;
  memcpy(allocator->queue_families, queue_families,
         queue_family_count * sizeof(queue_families[0]));
  allocator->sparse_binding_queue = sparse_binding_queue;
  iree_slim_mutex_initialize(&allocator->virtual_memory_mutex);

  *out_allocator = (iree_hal_allocator_t*)allocator;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_vulkan_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_allocator_deinitialize_memory_services(base_allocator);
  iree_hal_vulkan_allocator_deinitialize_virtual_memory_registry(allocator);
  iree_allocator_free(host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_vulkan_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  const iree_hal_vulkan_allocator_t* allocator =
      (const iree_hal_vulkan_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static void iree_hal_vulkan_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  memset(out_statistics, 0, sizeof(*out_statistics));
  IREE_STATISTICS({
    iree_hal_vulkan_allocator_t* allocator =
        iree_hal_vulkan_allocator_cast(base_allocator);
    memcpy(out_statistics, &allocator->statistics, sizeof(*out_statistics));
  });
}

static iree_device_size_t iree_hal_vulkan_allocator_max_allocation_size(
    const iree_hal_vulkan_allocator_t* allocator,
    const VkMemoryHeap* memory_heap) {
  iree_device_size_t limit = memory_heap->size;
  if (allocator->properties11.maxMemoryAllocationSize != 0 &&
      allocator->properties11.maxMemoryAllocationSize < limit) {
    limit = allocator->properties11.maxMemoryAllocationSize;
  }
  return limit;
}

static const VkMemoryHeap* iree_hal_vulkan_allocator_memory_heap_for_type(
    const iree_hal_vulkan_allocator_t* allocator, uint32_t memory_type_index) {
  const VkPhysicalDeviceMemoryProperties* memory_properties =
      &allocator->memory_properties2.memoryProperties;
  const VkMemoryType* memory_type =
      &memory_properties->memoryTypes[memory_type_index];
  return &memory_properties->memoryHeaps[memory_type->heapIndex];
}

static iree_device_size_t
iree_hal_vulkan_allocator_max_allocation_size_for_type(
    const iree_hal_vulkan_allocator_t* allocator, uint32_t memory_type_index) {
  return iree_hal_vulkan_allocator_max_allocation_size(
      allocator, iree_hal_vulkan_allocator_memory_heap_for_type(
                     allocator, memory_type_index));
}

static iree_hal_memory_type_t
iree_hal_vulkan_allocator_memory_type_from_properties(
    const iree_hal_vulkan_allocator_t* allocator,
    VkMemoryPropertyFlags memory_property_flags) {
  return iree_hal_vulkan_memory_type_from_properties(
      allocator->properties2.properties.deviceType, memory_property_flags);
}

static iree_hal_buffer_usage_t iree_hal_vulkan_allowed_usage_from_memory_type(
    iree_hal_memory_type_t memory_type) {
  iree_hal_buffer_usage_t allowed_usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH |
      IREE_HAL_BUFFER_USAGE_SHARING_REPLICATE |
      IREE_HAL_BUFFER_USAGE_SHARING_CONCURRENT |
      IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE;
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    allowed_usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                     IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
                     IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL |
                     IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM |
                     IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE;
  }
  return allowed_usage;
}

static iree_hal_atomic_operation_capabilities_t
iree_hal_vulkan_allocator_atomic_operations(
    const iree_hal_vulkan_allocator_t* allocator,
    iree_hal_memory_type_t memory_type, iree_hal_buffer_usage_t allowed_usage) {
  if (!iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE) ||
      !iree_all_bits_set(allowed_usage, IREE_HAL_BUFFER_USAGE_STORAGE)) {
    return (iree_hal_atomic_operation_capabilities_t){0};
  }
  return iree_hal_vulkan_atomic_capabilities(allocator->enabled_features)
      .operations;
}

// Heap queries feed first-match consumers such as the caching allocator. Prefer
// private device-local memory for common dispatches, then UMA/BAR memory, then
// host-local staging/readback classes.
static int32_t iree_hal_vulkan_allocator_query_memory_heap_priority(
    iree_hal_memory_type_t memory_type) {
  int32_t priority = 0;
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    priority += 1000;
    if (!iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
      priority += 300;
    }
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    priority += 200;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    priority += 100;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_CACHED)) {
    priority += 40;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    priority += 20;
  }
  return priority;
}

static bool iree_hal_vulkan_allocator_memory_type_precedes(
    const iree_hal_vulkan_allocator_t* allocator,
    uint32_t left_memory_type_index, uint32_t right_memory_type_index) {
  const VkPhysicalDeviceMemoryProperties* memory_properties =
      &allocator->memory_properties2.memoryProperties;
  const VkMemoryType* left_memory_type =
      &memory_properties->memoryTypes[left_memory_type_index];
  const VkMemoryType* right_memory_type =
      &memory_properties->memoryTypes[right_memory_type_index];
  const iree_hal_memory_type_t left_hal_memory_type =
      iree_hal_vulkan_allocator_memory_type_from_properties(
          allocator, left_memory_type->propertyFlags);
  const iree_hal_memory_type_t right_hal_memory_type =
      iree_hal_vulkan_allocator_memory_type_from_properties(
          allocator, right_memory_type->propertyFlags);
  const int32_t left_priority =
      iree_hal_vulkan_allocator_query_memory_heap_priority(
          left_hal_memory_type);
  const int32_t right_priority =
      iree_hal_vulkan_allocator_query_memory_heap_priority(
          right_hal_memory_type);
  if (left_priority != right_priority) {
    return left_priority > right_priority;
  }
  const iree_device_size_t left_max_allocation_size =
      iree_hal_vulkan_allocator_max_allocation_size_for_type(
          allocator, left_memory_type_index);
  const iree_device_size_t right_max_allocation_size =
      iree_hal_vulkan_allocator_max_allocation_size_for_type(
          allocator, right_memory_type_index);
  if (left_max_allocation_size != right_max_allocation_size) {
    return left_max_allocation_size > right_max_allocation_size;
  }
  return left_memory_type_index < right_memory_type_index;
}

void iree_hal_vulkan_allocator_deinitialize_memory_services(
    iree_hal_allocator_t* base_allocator) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  iree_async_notification_release(allocator->memory_notification);
  allocator->memory_notification = NULL;
  iree_hal_memory_maintenance_release(allocator->memory_maintenance);
  allocator->memory_maintenance = NULL;
}

iree_status_t iree_hal_vulkan_allocator_initialize_memory_services(
    iree_hal_allocator_t* base_allocator, iree_async_proactor_t* proactor,
    iree_thread_affinity_t memory_affinity) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_async_notification_create(
      proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE,
      &allocator->memory_notification);
  if (iree_status_is_ok(status)) {
    status = iree_hal_memory_maintenance_thread_create(
        memory_affinity, allocator->host_allocator,
        &allocator->memory_maintenance);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_allocator_deinitialize_memory_services(base_allocator);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_async_notification_t* iree_hal_vulkan_allocator_memory_notification(
    iree_hal_allocator_t* base_allocator) {
  return iree_hal_vulkan_allocator_cast(base_allocator)->memory_notification;
}

iree_hal_memory_maintenance_t* iree_hal_vulkan_allocator_memory_maintenance(
    iree_hal_allocator_t* base_allocator) {
  return iree_hal_vulkan_allocator_cast(base_allocator)->memory_maintenance;
}

static iree_status_t iree_hal_vulkan_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  IREE_ASSERT_ARGUMENT(out_count);
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  const VkPhysicalDeviceMemoryProperties* memory_properties =
      &allocator->memory_properties2.memoryProperties;
  const iree_host_size_t heap_count = memory_properties->memoryTypeCount;
  *out_count = heap_count;
  if (heaps != NULL && capacity < heap_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan memory heap query capacity %" PRIhsz
                            " is smaller than the memory type count %" PRIhsz,
                            capacity, heap_count);
  }
  if (heaps == NULL) {
    return iree_ok_status();
  }

  uint32_t memory_type_indices[VK_MAX_MEMORY_TYPES];
  for (iree_host_size_t i = 0; i < heap_count; ++i) {
    memory_type_indices[i] = (uint32_t)i;
  }
  for (iree_host_size_t i = 1; i < heap_count; ++i) {
    const uint32_t memory_type_index = memory_type_indices[i];
    iree_host_size_t insertion_index = i;
    while (insertion_index > 0 &&
           iree_hal_vulkan_allocator_memory_type_precedes(
               allocator, memory_type_index,
               memory_type_indices[insertion_index - 1])) {
      memory_type_indices[insertion_index] =
          memory_type_indices[insertion_index - 1];
      --insertion_index;
    }
    memory_type_indices[insertion_index] = memory_type_index;
  }

  for (iree_host_size_t i = 0; i < heap_count; ++i) {
    const uint32_t memory_type_index = memory_type_indices[i];
    const VkMemoryType* memory_type =
        &memory_properties->memoryTypes[memory_type_index];
    const iree_hal_memory_type_t hal_memory_type =
        iree_hal_vulkan_allocator_memory_type_from_properties(
            allocator, memory_type->propertyFlags);
    const iree_hal_buffer_usage_t allowed_usage =
        iree_hal_vulkan_allowed_usage_from_memory_type(hal_memory_type);
    heaps[i] = (iree_hal_allocator_memory_heap_t){
        .type = hal_memory_type,
        .allowed_usage = allowed_usage,
        .atomic_operations = iree_hal_vulkan_allocator_atomic_operations(
            allocator, hal_memory_type, allowed_usage),
        .max_allocation_size =
            iree_hal_vulkan_allocator_max_allocation_size_for_type(
                allocator, memory_type_index),
        .min_alignment = 1,
    };
  }
  return iree_ok_status();
}

static bool iree_hal_vulkan_allocator_normalize_queue_family_affinity(
    const iree_hal_vulkan_allocator_t* allocator,
    iree_hal_vulkan_buffer_spec_t* params) {
  iree_hal_queue_family_affinity_t queue_family_affinity =
      iree_hal_queue_family_affinity_is_any(params->queue_family_affinity)
          ? allocator->queue_family_affinity_mask
          : params->queue_family_affinity;
  if (iree_hal_queue_family_affinity_is_empty(queue_family_affinity) ||
      (queue_family_affinity & ~allocator->queue_family_affinity_mask) != 0) {
    return false;
  }
  params->queue_family_affinity = queue_family_affinity;
  return true;
}

static const iree_hal_buffer_usage_t iree_hal_vulkan_mapping_usage_bits =
    IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
    IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
    IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL |
    IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM |
    IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE;

static bool iree_hal_vulkan_allocator_strip_optional_mapping(
    iree_hal_vulkan_buffer_spec_t* params) {
  if (!iree_any_bit_set(params->usage, iree_hal_vulkan_mapping_usage_bits)) {
    return true;
  }
  if (!iree_all_bits_set(params->usage,
                         IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL)) {
    return false;
  }
  params->usage &= ~iree_hal_vulkan_mapping_usage_bits;
  return true;
}

static int iree_hal_vulkan_allocator_score_memory_type(
    const iree_hal_vulkan_buffer_spec_t* params,
    iree_hal_memory_type_t memory_type) {
  const bool mapping_requested =
      iree_any_bit_set(params->usage, iree_hal_vulkan_mapping_usage_bits);
  const bool dispatch_requested =
      iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_DISPATCH);
  int score = 0;
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    score += dispatch_requested && !mapping_requested ? 100 : 20;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    score += mapping_requested ? 80 : 5;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_CACHED)) {
    score += mapping_requested ? 20 : 0;
  }
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    score += mapping_requested ? 10 : 0;
  }
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_HOST) &&
      iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    score += 50;
  }
  if (iree_all_bits_set(params->type,
                        IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE) &&
      iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    score += 50;
  }
  return score;
}

static bool iree_hal_vulkan_allocator_resolve_memory_placement(
    const iree_hal_vulkan_allocator_t* allocator,
    uint32_t allowed_memory_type_bits, iree_hal_vulkan_buffer_spec_t* params,
    iree_hal_vulkan_allocator_memory_placement_t* out_placement) {
  memset(out_placement, 0, sizeof(*out_placement));
  if (!iree_hal_vulkan_allocator_normalize_queue_family_affinity(allocator,
                                                                 params)) {
    return false;
  }
  if (!iree_device_size_is_valid_alignment(params->min_alignment)) {
    return false;
  }

  const iree_hal_memory_type_t required_type =
      params->type & ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
  const VkPhysicalDeviceMemoryProperties* memory_properties =
      &allocator->memory_properties2.memoryProperties;

  bool found = false;
  int best_score = 0;
  iree_hal_vulkan_buffer_spec_t best_params = *params;
  iree_hal_vulkan_allocator_memory_placement_t best_placement;
  memset(&best_placement, 0, sizeof(best_placement));

  for (uint32_t i = 0; i < memory_properties->memoryTypeCount; ++i) {
    if (!iree_all_bits_set(allowed_memory_type_bits, 1u << i)) {
      continue;
    }
    const VkMemoryType* vk_memory_type = &memory_properties->memoryTypes[i];
    const iree_hal_memory_type_t memory_type =
        iree_hal_vulkan_allocator_memory_type_from_properties(
            allocator, vk_memory_type->propertyFlags);
    if (!iree_all_bits_set(memory_type, required_type)) {
      continue;
    }

    iree_hal_vulkan_buffer_spec_t candidate_params = *params;
    const iree_hal_buffer_usage_t allowed_usage =
        iree_hal_vulkan_allowed_usage_from_memory_type(memory_type);
    if (!iree_all_bits_set(allowed_usage, candidate_params.usage)) {
      if (!iree_hal_vulkan_allocator_strip_optional_mapping(
              &candidate_params)) {
        continue;
      }
      if (!iree_all_bits_set(allowed_usage, candidate_params.usage)) {
        continue;
      }
    }

    const int score = iree_hal_vulkan_allocator_score_memory_type(
        &candidate_params, memory_type);
    // Equal-scoring types may refer to a small host-visible device heap and a
    // much larger cached host heap. Prefer the larger allocation range instead
    // of depending on the memory type enumeration order.
    if (!found || score > best_score ||
        (score == best_score &&
         iree_hal_vulkan_allocator_max_allocation_size_for_type(allocator, i) >
             iree_hal_vulkan_allocator_max_allocation_size_for_type(
                 allocator, best_placement.memory_type_index))) {
      found = true;
      best_score = score;
      best_params = candidate_params;
      best_params.type = memory_type;
      best_placement = (iree_hal_vulkan_allocator_memory_placement_t){
          .memory_type_index = i,
          .memory_property_flags = vk_memory_type->propertyFlags,
          .memory_type = memory_type,
      };
    }
  }

  if (!found) {
    return false;
  }
  *params = best_params;
  *out_placement = best_placement;
  return true;
}

static iree_status_t iree_hal_vulkan_allocator_align_allocation_size(
    iree_device_size_t* allocation_size) {
  if (*allocation_size == 0) {
    *allocation_size = 4;
  }
  if (!iree_device_size_checked_align(*allocation_size, 4, allocation_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan allocation size overflows 4-byte "
                            "buffer alignment");
  }
  return iree_ok_status();
}

static bool iree_hal_vulkan_allocator_allocation_size_in_range(
    const iree_hal_vulkan_allocator_t* allocator,
    iree_device_size_t allocation_size) {
  return allocator->properties11.maxMemoryAllocationSize == 0 ||
         allocation_size <= allocator->properties11.maxMemoryAllocationSize;
}

static bool iree_hal_vulkan_allocator_supports_sparse_binding(
    const iree_hal_vulkan_allocator_t* allocator) {
  return iree_all_bits_set(allocator->enabled_features.general,
                           IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING) &&
         allocator->sparse_binding_queue != NULL;
}

static bool iree_hal_vulkan_allocator_supports_sparse_virtual_memory(
    const iree_hal_vulkan_allocator_t* allocator) {
  return iree_all_bits_set(
             allocator->enabled_features.general,
             IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_RESIDENCY_ALIASED) &&
         allocator->sparse_binding_queue != NULL;
}

bool iree_hal_vulkan_allocator_supports_host_allocation_import(
    const iree_hal_vulkan_allocator_t* allocator) {
  if (!iree_all_bits_set(
          allocator->enabled_extensions,
          IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_EXTERNAL_MEMORY_HOST)) {
    return false;
  }
  // Arbitrary host pointers only become usable Vulkan buffers when the device
  // and host share memory. Discrete devices may expose the extension while
  // still rejecting normal heap pointers for device-visible storage.
  return iree_hal_vulkan_physical_device_type_uses_unified_memory(
      allocator->properties2.properties.deviceType);
}

static iree_status_t iree_hal_vulkan_allocator_prepare_sparse_buffer_params(
    const iree_hal_vulkan_allocator_t* allocator,
    iree_device_size_t allocation_size, iree_device_size_t max_allocation_size,
    iree_hal_vulkan_buffer_spec_t* params) {
  if (!iree_hal_vulkan_allocator_supports_sparse_binding(allocator)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "Vulkan sparse binding support is required for allocation size %" PRIu64
        " above per-memory-type max allocation size %" PRIu64
        " but sparseBinding is not enabled with a sparse-capable queue",
        (uint64_t)allocation_size, (uint64_t)max_allocation_size);
  }
  if (!iree_hal_vulkan_allocator_strip_optional_mapping(params)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan sparse buffers cannot satisfy required mapping usage");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_allocator_require_virtual_memory(
    const iree_hal_vulkan_allocator_t* allocator) {
  if (iree_hal_vulkan_allocator_supports_sparse_virtual_memory(allocator)) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "Vulkan virtual memory requires sparseBinding, sparseResidencyBuffer, "
      "sparseResidencyAliased, and a sparse-capable queue");
}

static iree_hal_buffer_usage_t iree_hal_vulkan_allocator_virtual_buffer_usage(
    void) {
  return IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH;
}

static iree_hal_physical_memory_params_t
iree_hal_vulkan_allocator_virtual_memory_params(
    iree_hal_queue_family_affinity_t queue_family_affinity) {
  return (iree_hal_physical_memory_params_t){
      .type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE |
              IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .usage = iree_hal_vulkan_allocator_virtual_buffer_usage(),
      .queue_family_affinity = queue_family_affinity,
  };
}

static iree_status_t iree_hal_vulkan_allocator_prepare_virtual_memory_params(
    const iree_hal_vulkan_allocator_t* allocator,
    iree_hal_physical_memory_params_t* params,
    iree_hal_vulkan_buffer_spec_t* out_buffer_params) {
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_memory(allocator));
  iree_hal_physical_memory_params_canonicalize(params);
  *out_buffer_params = (iree_hal_vulkan_buffer_spec_t){
      .usage = params->usage,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = params->type,
      .queue_family_affinity = params->queue_family_affinity,
  };
  if (!iree_hal_vulkan_allocator_strip_optional_mapping(out_buffer_params)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan sparse virtual memory cannot satisfy required mapping usage");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_allocator_require_virtual_buffer(
    iree_hal_buffer_t* virtual_buffer) {
  if (iree_hal_vulkan_sparse_buffer_is_virtual_reservation(virtual_buffer)) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "Vulkan virtual memory operation requires a sparse virtual memory "
      "reservation");
}

static iree_status_t iree_hal_vulkan_allocator_resolve_buffer_sharing(
    const iree_hal_vulkan_allocator_t* allocator,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_vulkan_allocator_buffer_sharing_t* out_sharing) {
  memset(out_sharing, 0, sizeof(*out_sharing));
  out_sharing->mode = VK_SHARING_MODE_EXCLUSIVE;

  if (iree_hal_queue_family_affinity_is_any(queue_family_affinity)) {
    queue_family_affinity = allocator->queue_family_affinity_mask;
  } else if (iree_hal_queue_family_affinity_is_empty(queue_family_affinity) ||
             (queue_family_affinity & ~allocator->queue_family_affinity_mask) !=
                 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan buffer queue family affinity 0x%016" PRIx64
                            " is outside the supported mask 0x%016" PRIx64,
                            queue_family_affinity,
                            allocator->queue_family_affinity_mask);
  }
  uint32_t selected_family_count = 0;
  for (iree_host_size_t i = 0; i < allocator->queue_family_count; ++i) {
    const iree_hal_vulkan_allocator_queue_family_t* queue_family =
        &allocator->queue_families[i];
    if (!iree_any_bit_set(queue_family_affinity,
                          iree_hal_make_queue_family_affinity(
                              (iree_hal_queue_family_ordinal_t)i))) {
      continue;
    }
    out_sharing->queue_family_indices[selected_family_count++] =
        queue_family->native_family_index;
  }
  if (selected_family_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan buffer has no accessible queue family");
  }
  if (selected_family_count > 1) {
    out_sharing->mode = VK_SHARING_MODE_CONCURRENT;
    out_sharing->queue_family_index_count = selected_family_count;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_allocator_create_buffer_handle(
    iree_hal_vulkan_allocator_t* allocator,
    const iree_hal_vulkan_buffer_spec_t* params,
    iree_device_size_t allocation_size, VkBufferCreateFlags create_flags,
    const void* create_info_pnext, VkBuffer* out_buffer) {
  *out_buffer = VK_NULL_HANDLE;

  iree_hal_vulkan_allocator_buffer_sharing_t sharing;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_allocator_resolve_buffer_sharing(
      allocator, params->queue_family_affinity, &sharing));

  VkBufferCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = create_info_pnext,
      .flags = create_flags,
      .size = (VkDeviceSize)allocation_size,
      .usage = iree_hal_vulkan_buffer_usage_from_hal(
          allocator->enabled_features, params->usage),
      .sharingMode = sharing.mode,
      .queueFamilyIndexCount = sharing.queue_family_index_count,
      .pQueueFamilyIndices = sharing.queue_family_index_count
                                 ? sharing.queue_family_indices
                                 : NULL,
  };
  return iree_vkCreateBuffer(IREE_VULKAN_DEVICE(&allocator->syms),
                             allocator->logical_device, &create_info,
                             /*pAllocator=*/NULL, out_buffer);
}

static iree_status_t iree_hal_vulkan_allocator_create_sparse_virtual_handle(
    iree_hal_vulkan_allocator_t* allocator,
    const iree_hal_vulkan_buffer_spec_t* params,
    iree_device_size_t allocation_size, VkBuffer* out_buffer,
    VkMemoryRequirements* out_memory_requirements) {
  *out_buffer = VK_NULL_HANDLE;
  memset(out_memory_requirements, 0, sizeof(*out_memory_requirements));
  const VkBufferCreateFlags create_flags =
      VK_BUFFER_CREATE_SPARSE_BINDING_BIT |
      VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT |
      VK_BUFFER_CREATE_SPARSE_ALIASED_BIT;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_allocator_create_buffer_handle(
      allocator, params, allocation_size, create_flags,
      /*create_info_pnext=*/NULL, out_buffer));
  iree_vkGetBufferMemoryRequirements(IREE_VULKAN_DEVICE(&allocator->syms),
                                     allocator->logical_device, *out_buffer,
                                     out_memory_requirements);
  return iree_ok_status();
}

static bool iree_hal_vulkan_allocator_uses_buffer_device_address(
    const iree_hal_vulkan_allocator_t* allocator,
    iree_hal_buffer_usage_t hal_usage) {
  return iree_all_bits_set(
             allocator->enabled_features.general,
             IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES) &&
         iree_any_bit_set(hal_usage, IREE_HAL_BUFFER_USAGE_DISPATCH);
}

static VkMemoryAllocateFlags iree_hal_vulkan_allocator_memory_allocate_flags(
    const iree_hal_vulkan_allocator_t* allocator,
    iree_hal_buffer_usage_t hal_usage) {
  return iree_hal_vulkan_allocator_uses_buffer_device_address(allocator,
                                                              hal_usage)
             ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
             : 0;
}

static iree_status_t iree_hal_vulkan_allocator_allocate_memory(
    iree_hal_vulkan_allocator_t* allocator,
    const iree_hal_vulkan_allocator_memory_placement_t* memory_placement,
    iree_hal_buffer_usage_t hal_usage,
    const VkMemoryRequirements* memory_requirements,
    VkDeviceMemory* out_device_memory) {
  *out_device_memory = VK_NULL_HANDLE;

  VkMemoryAllocateFlagsInfo allocate_flags_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags =
          iree_hal_vulkan_allocator_memory_allocate_flags(allocator, hal_usage),
  };
  VkMemoryAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &allocate_flags_info,
      .allocationSize = memory_requirements->size,
      .memoryTypeIndex = memory_placement->memory_type_index,
  };
  return iree_vkAllocateMemory(IREE_VULKAN_DEVICE(&allocator->syms),
                               allocator->logical_device, &allocate_info,
                               /*pAllocator=*/NULL, out_device_memory);
}

static VkDeviceAddress iree_hal_vulkan_allocator_query_device_address(
    iree_hal_vulkan_allocator_t* allocator, iree_hal_buffer_usage_t hal_usage,
    VkBuffer handle) {
  if (!iree_hal_vulkan_allocator_uses_buffer_device_address(allocator,
                                                            hal_usage)) {
    return 0;
  }
  VkBufferDeviceAddressInfo address_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = handle,
  };
  return iree_vkGetBufferDeviceAddress(IREE_VULKAN_DEVICE(&allocator->syms),
                                       allocator->logical_device,
                                       &address_info);
}

// Returns the offset of the first |alignment|-aligned address in |handle|.
// A zero address denotes a resource without the device-address interface; its
// native buffer coordinate starts at zero and already satisfies any alignment.
static VkDeviceSize iree_hal_vulkan_allocator_aligned_handle_offset(
    VkDeviceAddress device_address, iree_device_size_t alignment) {
  return device_address
             ? (VkDeviceSize)(-device_address & ((uint64_t)alignment - 1))
             : 0;
}

static iree_status_t iree_hal_vulkan_allocator_make_buffer_params_status(
    const iree_hal_vulkan_buffer_spec_t* params) {
#if IREE_STATUS_MODE
  iree_bitfield_string_temp_t temp0, temp1;
  iree_string_view_t memory_type_str =
      iree_hal_memory_type_format(params->type, &temp0);
  iree_string_view_t usage_str =
      iree_hal_buffer_usage_format(params->usage, &temp1);
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "allocator cannot allocate a Vulkan buffer with the given parameters; "
      "memory_type=%.*s, usage=%.*s",
      (int)memory_type_str.size, memory_type_str.data, (int)usage_str.size,
      usage_str.data);
#else
  (void)params;
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "allocator cannot allocate a Vulkan buffer with the given parameters");
#endif  // IREE_STATUS_MODE
}

static iree_status_t
iree_hal_vulkan_allocator_allocate_direct_buffer_with_memory_type_bits(
    iree_hal_vulkan_allocator_t* allocator, uint32_t allowed_memory_type_bits,
    const iree_hal_vulkan_buffer_spec_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  const iree_device_size_t byte_length = allocation_size;
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)byte_length);

  const iree_device_size_t min_alignment =
      params->min_alignment ? params->min_alignment : 1;
  iree_status_t status = iree_ok_status();
  if (!iree_device_size_is_power_of_two(min_alignment)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "requested Vulkan buffer alignment %" PRIu64
                              " is not a power-of-two",
                              (uint64_t)params->min_alignment);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_align_allocation_size(&allocation_size);
  }
  if (iree_status_is_ok(status) &&
      iree_hal_vulkan_allocator_uses_buffer_device_address(allocator,
                                                           params->usage) &&
      !iree_device_size_checked_add(allocation_size, min_alignment - 1,
                                    &allocation_size)) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan buffer allocation size overflows while reserving %" PRIu64
        "-byte device address alignment",
        (uint64_t)min_alignment);
  }

  bool use_sparse_allocation = false;
  if (iree_status_is_ok(status)) {
    use_sparse_allocation = !iree_hal_vulkan_allocator_allocation_size_in_range(
        allocator, allocation_size);
  }

  iree_hal_vulkan_buffer_spec_t compat_params = *params;
  if (iree_status_is_ok(status) && use_sparse_allocation) {
    status = iree_hal_vulkan_allocator_prepare_sparse_buffer_params(
        allocator, allocation_size,
        allocator->properties11.maxMemoryAllocationSize, &compat_params);
  }

  iree_hal_vulkan_allocator_memory_placement_t memory_placement;
  if (iree_status_is_ok(status) &&
      !iree_hal_vulkan_allocator_resolve_memory_placement(
          allocator, allowed_memory_type_bits, &compat_params,
          &memory_placement)) {
    status = iree_hal_vulkan_allocator_make_buffer_params_status(params);
  }

  VkBuffer handle = VK_NULL_HANDLE;
  if (iree_status_is_ok(status)) {
    const VkBufferCreateFlags create_flags =
        use_sparse_allocation ? VK_BUFFER_CREATE_SPARSE_BINDING_BIT : 0;
    status = iree_hal_vulkan_allocator_create_buffer_handle(
        allocator, &compat_params, allocation_size, create_flags,
        /*create_info_pnext=*/NULL, &handle);
  }

  VkMemoryRequirements memory_requirements = {0};
  if (iree_status_is_ok(status)) {
    iree_vkGetBufferMemoryRequirements(IREE_VULKAN_DEVICE(&allocator->syms),
                                       allocator->logical_device, handle,
                                       &memory_requirements);
  }
  iree_device_size_t max_allocation_size = 0;
  if (iree_status_is_ok(status)) {
    max_allocation_size =
        iree_hal_vulkan_allocator_max_allocation_size_for_type(
            allocator, memory_placement.memory_type_index);
  }
  if (iree_status_is_ok(status) && !use_sparse_allocation &&
      memory_requirements.size > max_allocation_size) {
    iree_vkDestroyBuffer(IREE_VULKAN_DEVICE(&allocator->syms),
                         allocator->logical_device, handle,
                         /*pAllocator=*/NULL);
    handle = VK_NULL_HANDLE;
    use_sparse_allocation = true;
    compat_params = *params;
    status = iree_hal_vulkan_allocator_prepare_sparse_buffer_params(
        allocator, memory_requirements.size, max_allocation_size,
        &compat_params);
    if (iree_status_is_ok(status) &&
        !iree_hal_vulkan_allocator_resolve_memory_placement(
            allocator, allowed_memory_type_bits, &compat_params,
            &memory_placement)) {
      status = iree_hal_vulkan_allocator_make_buffer_params_status(params);
    }
    if (iree_status_is_ok(status)) {
      max_allocation_size =
          iree_hal_vulkan_allocator_max_allocation_size_for_type(
              allocator, memory_placement.memory_type_index);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_vulkan_allocator_create_buffer_handle(
          allocator, &compat_params, allocation_size,
          VK_BUFFER_CREATE_SPARSE_BINDING_BIT, /*create_info_pnext=*/NULL,
          &handle);
    }
    if (iree_status_is_ok(status)) {
      iree_vkGetBufferMemoryRequirements(IREE_VULKAN_DEVICE(&allocator->syms),
                                         allocator->logical_device, handle,
                                         &memory_requirements);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_resolve_memory_placement(
                 allocator,
                 memory_requirements.memoryTypeBits & allowed_memory_type_bits,
                 &compat_params, &memory_placement)
                 ? iree_ok_status()
                 : iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "allocator cannot find a Vulkan memory "
                                    "type compatible with buffer usage");
  }

  VkDeviceMemory device_memory = VK_NULL_HANDLE;
  if (iree_status_is_ok(status) && !use_sparse_allocation) {
    status = iree_hal_vulkan_allocator_allocate_memory(
        allocator, &memory_placement, compat_params.usage, &memory_requirements,
        &device_memory);
  }
  if (iree_status_is_ok(status) && !use_sparse_allocation) {
    status = iree_vkBindBufferMemory(IREE_VULKAN_DEVICE(&allocator->syms),
                                     allocator->logical_device, handle,
                                     device_memory, /*memoryOffset=*/0);
  }

  VkDeviceAddress device_address = 0;
  VkDeviceSize handle_offset = 0;
  iree_device_size_t logical_allocation_size = 0;
  if (iree_status_is_ok(status)) {
    device_address = iree_hal_vulkan_allocator_query_device_address(
        allocator, compat_params.usage, handle);
    handle_offset = iree_hal_vulkan_allocator_aligned_handle_offset(
        device_address, min_alignment);
    logical_allocation_size = allocation_size - handle_offset;
  }

  iree_hal_buffer_t* buffer = NULL;
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
  VkBuffer trace_handle = VK_NULL_HANDLE;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
  if (iree_status_is_ok(status)) {
    const iree_hal_buffer_placement_t buffer_placement = {
        .device = allocator->parent_device,
        .queue_family_affinity = compat_params.queue_family_affinity,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
    };
    if (use_sparse_allocation) {
      status = iree_hal_vulkan_sparse_buffer_create_bound_sync(
          &allocator->syms, allocator->logical_device,
          allocator->sparse_binding_queue, buffer_placement,
          memory_placement.memory_type, compat_params.access,
          compat_params.usage, logical_allocation_size, byte_length, handle,
          (VkDeviceSize)allocation_size, handle_offset, device_address,
          memory_requirements, memory_placement.memory_type_index,
          max_allocation_size,
          iree_hal_vulkan_allocator_memory_allocate_flags(allocator,
                                                          compat_params.usage),
          allocator->host_allocator, &buffer);
      if (iree_status_is_ok(status)) {
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
        trace_handle = handle;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
        handle = VK_NULL_HANDLE;
      }
    } else {
      status = iree_hal_vulkan_buffer_create(
          &allocator->syms, allocator->logical_device, buffer_placement,
          memory_placement.memory_type, compat_params.access,
          compat_params.usage, logical_allocation_size, byte_length,
          memory_placement.memory_property_flags,
          allocator->properties2.properties.limits.nonCoherentAtomSize,
          device_memory, /*device_memory_offset=*/handle_offset,
          memory_requirements.size, handle, (VkDeviceSize)allocation_size,
          handle_offset, device_address,
          iree_hal_buffer_release_callback_null(), allocator->host_allocator,
          &buffer);
      if (iree_status_is_ok(status)) {
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
        trace_handle = handle;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
        device_memory = VK_NULL_HANDLE;
        handle = VK_NULL_HANDLE;
      }
    }
  }

  if (iree_status_is_ok(status)) {
    IREE_TRACE_ALLOC_NAMED(IREE_HAL_VULKAN_ALLOCATOR_ID, (void*)trace_handle,
                           (iree_host_size_t)allocation_size);
    *out_buffer = buffer;
  } else {
    iree_hal_buffer_release(buffer);
    if (device_memory) {
      iree_vkFreeMemory(IREE_VULKAN_DEVICE(&allocator->syms),
                        allocator->logical_device, device_memory,
                        /*pAllocator=*/NULL);
    }
    if (handle) {
      iree_vkDestroyBuffer(IREE_VULKAN_DEVICE(&allocator->syms),
                           allocator->logical_device, handle,
                           /*pAllocator=*/NULL);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_allocator_allocate_direct_buffer_from_type(
    iree_hal_vulkan_allocator_t* allocator, uint32_t memory_type_index,
    const iree_hal_vulkan_buffer_spec_t* params,
    iree_device_size_t allocation_size, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  if (memory_type_index >=
      allocator->memory_properties2.memoryProperties.memoryTypeCount) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan memory type index %u exceeds physical-device memory type "
        "count %u",
        memory_type_index,
        allocator->memory_properties2.memoryProperties.memoryTypeCount);
  }
  return iree_hal_vulkan_allocator_allocate_direct_buffer_with_memory_type_bits(
      allocator, 1u << memory_type_index, params, allocation_size, out_buffer);
}

iree_status_t iree_hal_vulkan_allocator_allocate_direct_buffer(
    iree_hal_allocator_t* base_allocator,
    const iree_hal_vulkan_buffer_spec_t* params,
    iree_device_size_t allocation_size, iree_hal_buffer_t** out_buffer) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  return iree_hal_vulkan_allocator_allocate_direct_buffer_with_memory_type_bits(
      allocator, UINT32_MAX, params, allocation_size, out_buffer);
}

static void iree_hal_vulkan_allocator_normalize_host_allocation_import_params(
    iree_hal_vulkan_buffer_spec_t* params) {
  const bool requires_host_local =
      iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL);
  if (iree_any_bit_set(params->type, IREE_HAL_MEMORY_TYPE_OPTIMAL)) {
    params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
  }
  if (requires_host_local) {
    params->type &= ~IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
    // Imported host allocations are host-local by construction. Coherence is
    // an independent property of the compatible Vulkan memory type and may
    // require explicit flush and invalidate operations.
    params->type |= IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  }
  params->type |= IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
}

static iree_status_t iree_hal_vulkan_allocator_validate_host_allocation_import(
    const iree_hal_vulkan_allocator_t* allocator,
    const iree_hal_vulkan_buffer_spec_t* params,
    const iree_hal_external_storage_t* external_buffer) {
  if (external_buffer->storage_class.provenance[0] ||
      external_buffer->storage_class.provenance[1]) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "Vulkan host pointer import has no matching opaque provenance");
  }
  if (external_buffer->storage_class.type !=
      IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Vulkan external storage type is unsupported");
  }
  if (!iree_hal_vulkan_allocator_supports_host_allocation_import(allocator)) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "Vulkan host allocation import requires VK_EXT_external_memory_host");
  }
  if (!external_buffer->payload.host_pointer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan host allocation import requires a "
                            "non-null pointer");
  }
  if (external_buffer->byte_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan host allocation import requires a "
                            "non-zero size");
  }
  if (!iree_device_size_is_valid_alignment(params->min_alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "requested Vulkan import alignment %" PRIu64
                            " is not a power-of-two",
                            (uint64_t)params->min_alignment);
  }
  if (params->min_alignment != 0 &&
      (params->min_alignment > IREE_HOST_SIZE_MAX ||
       !iree_host_ptr_has_alignment(external_buffer->payload.host_pointer,
                                    (iree_host_size_t)params->min_alignment))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "host allocation import pointer does not satisfy requested Vulkan "
        "alignment %" PRIu64,
        (uint64_t)params->min_alignment);
  }
  const VkDeviceSize import_alignment =
      allocator->external_memory_host_properties
          .minImportedHostPointerAlignment;
  if (import_alignment > IREE_DEVICE_SIZE_MAX ||
      (import_alignment != 0 && !iree_device_size_is_valid_alignment(
                                    (iree_device_size_t)import_alignment))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan reported invalid minImportedHostPointerAlignment %" PRIu64,
        (uint64_t)import_alignment);
  }
  if (import_alignment != 0 &&
      (import_alignment > IREE_HOST_SIZE_MAX ||
       !iree_host_ptr_has_alignment(external_buffer->payload.host_pointer,
                                    (iree_host_size_t)import_alignment))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "host allocation import pointer does not satisfy Vulkan "
        "minImportedHostPointerAlignment %" PRIu64,
        (uint64_t)import_alignment);
  }
  if (import_alignment != 0 &&
      !iree_device_size_has_alignment(external_buffer->byte_length,
                                      (iree_device_size_t)import_alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "host allocation import size %" PRIu64
        " does not satisfy Vulkan minImportedHostPointerAlignment %" PRIu64,
        (uint64_t)external_buffer->byte_length, (uint64_t)import_alignment);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_allocator_allocate_imported_host_memory(
    iree_hal_vulkan_allocator_t* allocator,
    const iree_hal_vulkan_allocator_memory_placement_t* memory_placement,
    iree_hal_buffer_usage_t hal_usage, VkDeviceSize allocation_size,
    void* host_pointer, VkDeviceMemory* out_device_memory) {
  *out_device_memory = VK_NULL_HANDLE;

  VkImportMemoryHostPointerInfoEXT import_info = {
      .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
      .pHostPointer = host_pointer,
  };
  VkMemoryAllocateFlagsInfo allocate_flags_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .pNext = &import_info,
      .flags =
          iree_hal_vulkan_allocator_memory_allocate_flags(allocator, hal_usage),
  };
  VkMemoryAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &allocate_flags_info,
      .allocationSize = allocation_size,
      .memoryTypeIndex = memory_placement->memory_type_index,
  };
  return iree_vkAllocateMemory(IREE_VULKAN_DEVICE(&allocator->syms),
                               allocator->logical_device, &allocate_info,
                               /*pAllocator=*/NULL, out_device_memory);
}

static iree_status_t
iree_hal_vulkan_allocator_import_buffer_with_memory_type_bits(
    iree_hal_vulkan_allocator_t* IREE_RESTRICT allocator,
    uint32_t allowed_memory_type_bits,
    const iree_hal_vulkan_buffer_spec_t* IREE_RESTRICT params,
    const iree_hal_external_storage_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(external_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)external_buffer->byte_length);

  iree_status_t status =
      iree_hal_vulkan_allocator_validate_host_allocation_import(
          allocator, params, external_buffer);

  iree_device_size_t allocation_size = external_buffer->byte_length;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_align_allocation_size(&allocation_size);
  }

  iree_hal_vulkan_buffer_spec_t compat_params = *params;
  iree_hal_vulkan_allocator_normalize_host_allocation_import_params(
      &compat_params);

  VkMemoryHostPointerPropertiesEXT host_pointer_properties = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
  };
  if (iree_status_is_ok(status)) {
    status = iree_vkGetMemoryHostPointerPropertiesEXT(
        IREE_VULKAN_DEVICE(&allocator->syms), allocator->logical_device,
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
        external_buffer->payload.host_pointer, &host_pointer_properties);
  }

  VkExternalMemoryBufferCreateInfo external_create_info = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
  };
  VkBuffer handle = VK_NULL_HANDLE;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_create_buffer_handle(
        allocator, &compat_params, allocation_size,
        /*create_flags=*/0, &external_create_info, &handle);
  }

  VkMemoryRequirements memory_requirements = {0};
  if (iree_status_is_ok(status)) {
    iree_vkGetBufferMemoryRequirements(IREE_VULKAN_DEVICE(&allocator->syms),
                                       allocator->logical_device, handle,
                                       &memory_requirements);
  }
  if (iree_status_is_ok(status) &&
      memory_requirements.size > external_buffer->byte_length) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan host allocation import requires %" PRIu64
        " bytes but the external allocation only provides %" PRIu64,
        (uint64_t)memory_requirements.size,
        (uint64_t)external_buffer->byte_length);
  }

  iree_hal_vulkan_allocator_memory_placement_t memory_placement;
  if (iree_status_is_ok(status) &&
      !iree_hal_vulkan_allocator_resolve_memory_placement(
          allocator,
          memory_requirements.memoryTypeBits &
              host_pointer_properties.memoryTypeBits & allowed_memory_type_bits,
          &compat_params, &memory_placement)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot find a Vulkan memory type compatible with the "
        "imported host pointer; buffer memory type bits=0x%08" PRIx32
        ", host pointer memory type bits=0x%08" PRIx32
        ", allowed memory type bits=0x%08" PRIx32,
        memory_requirements.memoryTypeBits,
        host_pointer_properties.memoryTypeBits, allowed_memory_type_bits);
  }
  if (iree_status_is_ok(status) &&
      memory_requirements.size >
          iree_hal_vulkan_allocator_max_allocation_size_for_type(
              allocator, memory_placement.memory_type_index)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Vulkan host allocation import requires %" PRIu64
                              " bytes, exceeding the selected memory type "
                              "limit",
                              (uint64_t)memory_requirements.size);
  }

  VkDeviceMemory device_memory = VK_NULL_HANDLE;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_allocate_imported_host_memory(
        allocator, &memory_placement, compat_params.usage,
        memory_requirements.size, external_buffer->payload.host_pointer,
        &device_memory);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vkBindBufferMemory(IREE_VULKAN_DEVICE(&allocator->syms),
                                     allocator->logical_device, handle,
                                     device_memory, /*memoryOffset=*/0);
  }

  iree_hal_buffer_t* buffer = NULL;
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
  VkBuffer trace_handle = VK_NULL_HANDLE;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
  if (iree_status_is_ok(status)) {
    const iree_hal_buffer_placement_t placement = {
        .device = allocator->parent_device,
        .queue_family_affinity = compat_params.queue_family_affinity,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
    };
    iree_hal_memory_type_t imported_memory_type = memory_placement.memory_type;
    if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
      imported_memory_type |= IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
    }
    status = iree_hal_vulkan_buffer_create(
        &allocator->syms, allocator->logical_device, placement,
        imported_memory_type, compat_params.access, compat_params.usage,
        allocation_size, external_buffer->byte_length,
        memory_placement.memory_property_flags,
        allocator->properties2.properties.limits.nonCoherentAtomSize,
        device_memory, /*device_memory_offset=*/0, memory_requirements.size,
        handle, (VkDeviceSize)allocation_size, /*handle_offset=*/0,
        iree_hal_vulkan_allocator_query_device_address(
            allocator, compat_params.usage, handle),
        release_callback, allocator->host_allocator, &buffer);
    if (iree_status_is_ok(status)) {
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
      trace_handle = handle;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
      device_memory = VK_NULL_HANDLE;
      handle = VK_NULL_HANDLE;
    }
  }

  if (iree_status_is_ok(status)) {
    IREE_TRACE_ALLOC_NAMED(IREE_HAL_VULKAN_ALLOCATOR_ID, (void*)trace_handle,
                           (iree_host_size_t)allocation_size);
    *out_buffer = buffer;
  } else {
    iree_hal_buffer_release(buffer);
    if (device_memory) {
      iree_vkFreeMemory(IREE_VULKAN_DEVICE(&allocator->syms),
                        allocator->logical_device, device_memory,
                        /*pAllocator=*/NULL);
    }
    if (handle) {
      iree_vkDestroyBuffer(IREE_VULKAN_DEVICE(&allocator->syms),
                           allocator->logical_device, handle,
                           /*pAllocator=*/NULL);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_allocator_import_direct_buffer(
    iree_hal_vulkan_allocator_t* allocator, uint32_t allowed_memory_type_bits,
    const iree_hal_vulkan_buffer_spec_t* params,
    const iree_hal_external_storage_t* storage,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(storage);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  return iree_hal_vulkan_allocator_import_buffer_with_memory_type_bits(
      allocator, allowed_memory_type_bits, params, storage,
      iree_hal_buffer_release_callback_null(), out_buffer);
}

static iree_status_t iree_hal_vulkan_allocator_validate_sparse_range(
    iree_device_size_t offset, iree_device_size_t size,
    iree_device_size_t container_size, iree_device_size_t page_size,
    const char* range_name) {
  if (size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan sparse %s range must be non-zero",
                            range_name);
  }
  if (!iree_device_size_has_alignment(offset, page_size) ||
      !iree_device_size_has_alignment(size, page_size)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan sparse %s range offset %" PRIu64 " and size %" PRIu64
        " must be aligned to page size %" PRIu64,
        range_name, (uint64_t)offset, (uint64_t)size, (uint64_t)page_size);
  }
  iree_device_size_t range_end = 0;
  if (!iree_device_size_checked_add(offset, size, &range_end) ||
      range_end > container_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan sparse %s range [%" PRIu64 ", %" PRIu64
                            ") exceeds container size %" PRIu64,
                            range_name, (uint64_t)offset, (uint64_t)range_end,
                            (uint64_t)container_size);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_allocator_validate_sparse_page_size(
    VkMemoryRequirements memory_requirements) {
  if (memory_requirements.alignment == 0 ||
      memory_requirements.alignment > IREE_DEVICE_SIZE_MAX ||
      !iree_device_size_is_valid_alignment(
          (iree_device_size_t)memory_requirements.alignment)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan sparse buffer reported invalid page size %" PRIu64,
        (uint64_t)memory_requirements.alignment);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_allocator_allocate_virtual_memory_mapping(
    iree_hal_vulkan_allocator_t* allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t* IREE_RESTRICT physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size,
    iree_hal_vulkan_allocator_virtual_memory_mapping_t** out_mapping) {
  *out_mapping = NULL;
  iree_hal_vulkan_allocator_virtual_memory_mapping_t* mapping = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator->host_allocator, sizeof(*mapping), (void**)&mapping));
  mapping->next = NULL;
  mapping->virtual_buffer = virtual_buffer;
  mapping->physical_memory = physical_memory;
  mapping->virtual_offset = virtual_offset;
  mapping->physical_offset = physical_offset;
  mapping->size = size;
  *out_mapping = mapping;
  return iree_ok_status();
}

static bool iree_hal_vulkan_allocator_has_virtual_memory_mappings_locked(
    iree_hal_vulkan_allocator_t* allocator, iree_hal_buffer_t* virtual_buffer) {
  for (iree_hal_vulkan_allocator_virtual_memory_mapping_t* mapping =
           allocator->virtual_memory_mappings;
       mapping; mapping = mapping->next) {
    if (mapping->virtual_buffer == virtual_buffer) {
      return true;
    }
  }
  return false;
}

static iree_status_t
iree_hal_vulkan_allocator_validate_virtual_memory_range_unmapped_locked(
    iree_hal_vulkan_allocator_t* allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size) {
  for (iree_hal_vulkan_allocator_virtual_memory_mapping_t* mapping =
           allocator->virtual_memory_mappings;
       mapping; mapping = mapping->next) {
    if (mapping->virtual_buffer != virtual_buffer) {
      continue;
    }
    if (iree_hal_vulkan_allocator_ranges_overlap(
            virtual_offset, size, mapping->virtual_offset, mapping->size)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Vulkan sparse virtual memory range [%" PRIu64 ", %" PRIu64
          ") overlaps existing mapping [%" PRIu64 ", %" PRIu64 ")",
          (uint64_t)virtual_offset, (uint64_t)(virtual_offset + size),
          (uint64_t)mapping->virtual_offset,
          (uint64_t)(mapping->virtual_offset + mapping->size));
    }
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_vulkan_allocator_validate_virtual_memory_range_mapped_locked(
    iree_hal_vulkan_allocator_t* allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size) {
  const iree_device_size_t range_end = virtual_offset + size;
  iree_device_size_t cursor = virtual_offset;
  while (cursor < range_end) {
    const iree_hal_vulkan_allocator_virtual_memory_mapping_t* covering_mapping =
        NULL;
    for (iree_hal_vulkan_allocator_virtual_memory_mapping_t* mapping =
             allocator->virtual_memory_mappings;
         mapping; mapping = mapping->next) {
      const iree_device_size_t mapping_end =
          mapping->virtual_offset + mapping->size;
      if (mapping->virtual_buffer == virtual_buffer &&
          mapping->virtual_offset <= cursor && mapping_end > cursor) {
        covering_mapping = mapping;
        break;
      }
    }
    if (!covering_mapping) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "Vulkan sparse virtual memory range [%" PRIu64
                              ", %" PRIu64 ") is not fully mapped",
                              (uint64_t)virtual_offset, (uint64_t)range_end);
    }
    const iree_device_size_t mapping_end =
        covering_mapping->virtual_offset + covering_mapping->size;
    cursor = mapping_end < range_end ? mapping_end : range_end;
  }
  return iree_ok_status();
}

static void iree_hal_vulkan_allocator_insert_virtual_memory_mapping_locked(
    iree_hal_vulkan_allocator_t* allocator,
    iree_hal_vulkan_allocator_virtual_memory_mapping_t* mapping) {
  IREE_ASSERT(mapping->physical_memory->mapped_size <=
              IREE_DEVICE_SIZE_MAX - mapping->size);
  mapping->next = allocator->virtual_memory_mappings;
  allocator->virtual_memory_mappings = mapping;
  mapping->physical_memory->mapped_size += mapping->size;
}

static void iree_hal_vulkan_allocator_remove_virtual_memory_mapping_locked(
    iree_hal_vulkan_allocator_t* allocator,
    iree_hal_vulkan_allocator_virtual_memory_mapping_t** mapping_ptr,
    iree_device_size_t remove_begin, iree_device_size_t remove_end,
    iree_hal_vulkan_allocator_virtual_memory_mapping_t* split_tail_mapping,
    bool* split_tail_mapping_used) {
  iree_hal_vulkan_allocator_virtual_memory_mapping_t* mapping = *mapping_ptr;
  const iree_device_size_t mapping_begin = mapping->virtual_offset;
  const iree_device_size_t mapping_end =
      mapping->virtual_offset + mapping->size;
  const iree_device_size_t overlap_begin =
      remove_begin > mapping_begin ? remove_begin : mapping_begin;
  const iree_device_size_t overlap_end =
      remove_end < mapping_end ? remove_end : mapping_end;
  const iree_device_size_t overlap_size = overlap_end - overlap_begin;

  IREE_ASSERT(mapping->physical_memory->mapped_size >= overlap_size);
  mapping->physical_memory->mapped_size -= overlap_size;
  if (overlap_begin == mapping_begin && overlap_end == mapping_end) {
    *mapping_ptr = mapping->next;
    iree_allocator_free(allocator->host_allocator, mapping);
  } else if (overlap_begin == mapping_begin) {
    mapping->virtual_offset = overlap_end;
    mapping->physical_offset += overlap_size;
    mapping->size = mapping_end - overlap_end;
    *mapping_ptr = mapping;
  } else if (overlap_end == mapping_end) {
    mapping->size = overlap_begin - mapping_begin;
    *mapping_ptr = mapping;
  } else {
    IREE_ASSERT(split_tail_mapping != NULL);
    IREE_ASSERT(!*split_tail_mapping_used);
    split_tail_mapping->next = mapping->next;
    split_tail_mapping->virtual_buffer = mapping->virtual_buffer;
    split_tail_mapping->physical_memory = mapping->physical_memory;
    split_tail_mapping->virtual_offset = overlap_end;
    split_tail_mapping->physical_offset =
        mapping->physical_offset + (overlap_end - mapping_begin);
    split_tail_mapping->size = mapping_end - overlap_end;
    mapping->size = overlap_begin - mapping_begin;
    mapping->next = split_tail_mapping;
    *split_tail_mapping_used = true;
    *mapping_ptr = mapping;
  }
}

static void iree_hal_vulkan_allocator_unmap_virtual_memory_range_locked(
    iree_hal_vulkan_allocator_t* allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_vulkan_allocator_virtual_memory_mapping_t* split_tail_mapping,
    bool* split_tail_mapping_used) {
  const iree_device_size_t range_end = virtual_offset + size;
  iree_hal_vulkan_allocator_virtual_memory_mapping_t** mapping_ptr =
      &allocator->virtual_memory_mappings;
  while (*mapping_ptr) {
    iree_hal_vulkan_allocator_virtual_memory_mapping_t* mapping = *mapping_ptr;
    if (mapping->virtual_buffer != virtual_buffer ||
        !iree_hal_vulkan_allocator_ranges_overlap(
            virtual_offset, size, mapping->virtual_offset, mapping->size)) {
      mapping_ptr = &mapping->next;
      continue;
    }

    iree_hal_vulkan_allocator_remove_virtual_memory_mapping_locked(
        allocator, mapping_ptr, virtual_offset, range_end, split_tail_mapping,
        split_tail_mapping_used);
    if (*mapping_ptr == mapping) {
      mapping_ptr = &mapping->next;
    }
  }
}

static bool iree_hal_vulkan_allocator_supports_virtual_memory(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  return iree_hal_vulkan_allocator_supports_sparse_virtual_memory(allocator);
}

static iree_status_t iree_hal_vulkan_allocator_virtual_memory_query_granularity(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_physical_memory_params_t params,
    iree_device_size_t* IREE_RESTRICT out_minimum_page_size,
    iree_device_size_t* IREE_RESTRICT out_recommended_page_size) {
  *out_minimum_page_size = 0;
  *out_recommended_page_size = 0;
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);

  iree_hal_vulkan_buffer_spec_t buffer_params;
  iree_status_t status =
      iree_hal_vulkan_allocator_prepare_virtual_memory_params(
          allocator, &params, &buffer_params);
  VkBuffer handle = VK_NULL_HANDLE;
  VkMemoryRequirements memory_requirements = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_create_sparse_virtual_handle(
        allocator, &buffer_params, /*allocation_size=*/4, &handle,
        &memory_requirements);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_validate_sparse_page_size(
        memory_requirements);
  }
  if (handle) {
    iree_vkDestroyBuffer(IREE_VULKAN_DEVICE(&allocator->syms),
                         allocator->logical_device, handle,
                         /*pAllocator=*/NULL);
  }
  if (iree_status_is_ok(status)) {
    *out_minimum_page_size = (iree_device_size_t)memory_requirements.alignment;
    *out_recommended_page_size =
        (iree_device_size_t)memory_requirements.alignment;
  }
  return status;
}

static iree_status_t iree_hal_vulkan_allocator_virtual_memory_reserve(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_device_size_t size,
    iree_hal_buffer_t** IREE_RESTRICT out_virtual_buffer) {
  *out_virtual_buffer = NULL;
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  if (size == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan virtual memory reservations must be non-zero");
  }

  iree_hal_physical_memory_params_t params =
      iree_hal_vulkan_allocator_virtual_memory_params(queue_family_affinity);
  iree_hal_vulkan_buffer_spec_t buffer_params;
  iree_status_t status =
      iree_hal_vulkan_allocator_prepare_virtual_memory_params(
          allocator, &params, &buffer_params);

  iree_hal_vulkan_allocator_memory_placement_t memory_placement;
  memset(&memory_placement, 0, sizeof(memory_placement));
  VkBuffer handle = VK_NULL_HANDLE;
  VkMemoryRequirements memory_requirements = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_create_sparse_virtual_handle(
        allocator, &buffer_params, size, &handle, &memory_requirements);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_validate_sparse_page_size(
        memory_requirements);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_validate_sparse_range(
        /*offset=*/0, size, size,
        (iree_device_size_t)memory_requirements.alignment, "reservation");
  }
  if (iree_status_is_ok(status) &&
      !iree_hal_vulkan_allocator_resolve_memory_placement(
          allocator, memory_requirements.memoryTypeBits, &buffer_params,
          &memory_placement)) {
    status =
        iree_hal_vulkan_allocator_make_buffer_params_status(&buffer_params);
  }

  iree_hal_buffer_t* virtual_buffer = NULL;
  if (iree_status_is_ok(status)) {
    const iree_hal_buffer_placement_t placement = {
        .device = allocator->parent_device,
        .queue_family_affinity = buffer_params.queue_family_affinity,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
    };
    status = iree_hal_vulkan_sparse_buffer_create_unbound(
        &allocator->syms, allocator->logical_device, placement,
        memory_placement.memory_type, buffer_params.access, buffer_params.usage,
        size, size, handle, memory_requirements, allocator->host_allocator,
        &virtual_buffer);
    if (iree_status_is_ok(status)) {
      handle = VK_NULL_HANDLE;
    }
  }

  if (iree_status_is_ok(status)) {
    *out_virtual_buffer = virtual_buffer;
  } else {
    iree_hal_buffer_release(virtual_buffer);
    if (handle) {
      iree_vkDestroyBuffer(IREE_VULKAN_DEVICE(&allocator->syms),
                           allocator->logical_device, handle,
                           /*pAllocator=*/NULL);
    }
  }
  return status;
}

static iree_status_t iree_hal_vulkan_allocator_virtual_memory_release(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_memory(allocator));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_buffer(virtual_buffer));

  iree_slim_mutex_lock(&allocator->virtual_memory_mutex);
  iree_status_t status = iree_ok_status();
  if (iree_hal_vulkan_allocator_has_virtual_memory_mappings_locked(
          allocator, virtual_buffer)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan sparse virtual memory reservations must be fully unmapped "
        "before release");
  } else {
    iree_hal_buffer_destroy(virtual_buffer);
  }
  iree_slim_mutex_unlock(&allocator->virtual_memory_mutex);
  return status;
}

static iree_status_t iree_hal_vulkan_allocator_physical_memory_allocate(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_physical_memory_params_t params, iree_device_size_t size,
    iree_allocator_t host_allocator,
    iree_hal_physical_memory_t** IREE_RESTRICT out_physical_memory) {
  *out_physical_memory = NULL;
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  if (size == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan physical memory allocations must be non-zero");
  }

  iree_hal_vulkan_buffer_spec_t buffer_params;
  iree_status_t status =
      iree_hal_vulkan_allocator_prepare_virtual_memory_params(
          allocator, &params, &buffer_params);
  VkBuffer scratch_handle = VK_NULL_HANDLE;
  VkMemoryRequirements memory_requirements = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_create_sparse_virtual_handle(
        allocator, &buffer_params, size, &scratch_handle, &memory_requirements);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_validate_sparse_page_size(
        memory_requirements);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocator_validate_sparse_range(
        /*offset=*/0, size, size,
        (iree_device_size_t)memory_requirements.alignment,
        "physical allocation");
  }

  iree_hal_vulkan_allocator_memory_placement_t memory_placement;
  memset(&memory_placement, 0, sizeof(memory_placement));
  if (iree_status_is_ok(status) &&
      !iree_hal_vulkan_allocator_resolve_memory_placement(
          allocator, memory_requirements.memoryTypeBits, &buffer_params,
          &memory_placement)) {
    status =
        iree_hal_vulkan_allocator_make_buffer_params_status(&buffer_params);
  }
  if (scratch_handle) {
    iree_vkDestroyBuffer(IREE_VULKAN_DEVICE(&allocator->syms),
                         allocator->logical_device, scratch_handle,
                         /*pAllocator=*/NULL);
  }

  if (iree_status_is_ok(status)) {
    const iree_device_size_t max_allocation_size =
        iree_hal_vulkan_allocator_max_allocation_size_for_type(
            allocator, memory_placement.memory_type_index);
    if (size > max_allocation_size) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan physical memory allocation size %" PRIu64
          " exceeds per-memory-type max allocation size %" PRIu64,
          (uint64_t)size, (uint64_t)max_allocation_size);
    }
  }

  iree_hal_physical_memory_t* physical_memory = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, sizeof(*physical_memory),
                                   (void**)&physical_memory);
  }
  if (iree_status_is_ok(status)) {
    memset(physical_memory, 0, sizeof(*physical_memory));
    physical_memory->host_allocator = host_allocator;
    physical_memory->syms = allocator->syms;
    physical_memory->logical_device = allocator->logical_device;
    physical_memory->owner_allocator = allocator;
    physical_memory->allocation_size = size;
    physical_memory->memory_type_index = memory_placement.memory_type_index;
    physical_memory->memory_type = memory_placement.memory_type;

    VkMemoryAllocateFlagsInfo allocate_flags_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = iree_hal_vulkan_allocator_memory_allocate_flags(
            allocator, iree_hal_vulkan_allocator_virtual_buffer_usage()),
    };
    VkMemoryAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &allocate_flags_info,
        .allocationSize = size,
        .memoryTypeIndex = memory_placement.memory_type_index,
    };
    status = iree_vkAllocateMemory(IREE_VULKAN_DEVICE(&allocator->syms),
                                   allocator->logical_device, &allocate_info,
                                   /*pAllocator=*/NULL,
                                   &physical_memory->device_memory);
  }

  if (iree_status_is_ok(status)) {
    IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, physical_memory->memory_type,
        physical_memory->allocation_size));
    *out_physical_memory = physical_memory;
  } else {
    if (physical_memory && physical_memory->device_memory) {
      iree_vkFreeMemory(IREE_VULKAN_DEVICE(&allocator->syms),
                        allocator->logical_device,
                        physical_memory->device_memory,
                        /*pAllocator=*/NULL);
    }
    iree_allocator_free(host_allocator, physical_memory);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_allocator_physical_memory_free(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_physical_memory_t* IREE_RESTRICT physical_memory) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_memory(allocator));
  if (physical_memory->owner_allocator != allocator) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan physical memory allocation belongs to a "
                            "different allocator");
  }

  iree_slim_mutex_lock(&allocator->virtual_memory_mutex);
  iree_status_t status = iree_ok_status();
  if (physical_memory->mapped_size != 0) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan physical memory allocation still has %" PRIu64 " mapped bytes",
        (uint64_t)physical_memory->mapped_size);
  } else {
    IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
        &allocator->statistics, physical_memory->memory_type,
        physical_memory->allocation_size));
    iree_vkFreeMemory(IREE_VULKAN_DEVICE(&physical_memory->syms),
                      physical_memory->logical_device,
                      physical_memory->device_memory,
                      /*pAllocator=*/NULL);
    iree_allocator_free(physical_memory->host_allocator, physical_memory);
  }
  iree_slim_mutex_unlock(&allocator->virtual_memory_mutex);
  return status;
}

static iree_status_t iree_hal_vulkan_allocator_virtual_memory_map(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t* IREE_RESTRICT physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_memory(allocator));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_buffer(virtual_buffer));
  if (physical_memory->owner_allocator != allocator) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan physical memory allocation belongs to a "
                            "different allocator");
  }

  VkDeviceMemory ignored_memory = VK_NULL_HANDLE;
  VkBuffer handle = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_handle(virtual_buffer, &ignored_memory, &handle));
  VkMemoryRequirements memory_requirements = {0};
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_sparse_buffer_memory_requirements(
      virtual_buffer, &memory_requirements));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_validate_sparse_page_size(memory_requirements));
  const iree_device_size_t page_size =
      (iree_device_size_t)memory_requirements.alignment;

  IREE_RETURN_IF_ERROR(iree_hal_vulkan_allocator_validate_sparse_range(
      virtual_offset, size, iree_hal_buffer_allocation_size(virtual_buffer),
      page_size, "virtual map"));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_allocator_validate_sparse_range(
      physical_offset, size, physical_memory->allocation_size, page_size,
      "physical map"));
  if (!iree_all_bits_set(memory_requirements.memoryTypeBits,
                         1u << physical_memory->memory_type_index)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan physical memory type %u is not compatible with sparse virtual "
        "buffer memory type bits 0x%x",
        physical_memory->memory_type_index, memory_requirements.memoryTypeBits);
  }

  iree_hal_vulkan_allocator_virtual_memory_mapping_t* mapping = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_allocate_virtual_memory_mapping(
          allocator, virtual_buffer, virtual_offset, physical_memory,
          physical_offset, size, &mapping));

  const VkSparseMemoryBind bind = {
      .resourceOffset = (VkDeviceSize)virtual_offset,
      .size = (VkDeviceSize)size,
      .memory = physical_memory->device_memory,
      .memoryOffset = (VkDeviceSize)physical_offset,
      .flags = 0,
  };
  iree_slim_mutex_lock(&allocator->virtual_memory_mutex);
  iree_status_t status =
      iree_hal_vulkan_allocator_validate_virtual_memory_range_unmapped_locked(
          allocator, virtual_buffer, virtual_offset, size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_sparse_buffer_bind_sync(
        allocator->sparse_binding_queue,
        iree_hal_buffer_allocation_placement(virtual_buffer), handle,
        /*bind_count=*/1, &bind);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_allocator_insert_virtual_memory_mapping_locked(allocator,
                                                                   mapping);
    mapping = NULL;
  }
  iree_slim_mutex_unlock(&allocator->virtual_memory_mutex);
  iree_allocator_free(allocator->host_allocator, mapping);
  return status;
}

static iree_status_t iree_hal_vulkan_allocator_virtual_memory_unmap(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_memory(allocator));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_buffer(virtual_buffer));

  VkDeviceMemory ignored_memory = VK_NULL_HANDLE;
  VkBuffer handle = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_handle(virtual_buffer, &ignored_memory, &handle));
  VkMemoryRequirements memory_requirements = {0};
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_sparse_buffer_memory_requirements(
      virtual_buffer, &memory_requirements));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_validate_sparse_page_size(memory_requirements));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_allocator_validate_sparse_range(
      virtual_offset, size, iree_hal_buffer_allocation_size(virtual_buffer),
      (iree_device_size_t)memory_requirements.alignment, "virtual unmap"));

  iree_hal_vulkan_allocator_virtual_memory_mapping_t* split_tail_mapping = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_allocate_virtual_memory_mapping(
          allocator, virtual_buffer, virtual_offset, /*physical_memory=*/NULL,
          /*physical_offset=*/0, size, &split_tail_mapping));

  const VkSparseMemoryBind bind = {
      .resourceOffset = (VkDeviceSize)virtual_offset,
      .size = (VkDeviceSize)size,
      .memory = VK_NULL_HANDLE,
      .memoryOffset = 0,
      .flags = 0,
  };
  iree_slim_mutex_lock(&allocator->virtual_memory_mutex);
  iree_status_t status =
      iree_hal_vulkan_allocator_validate_virtual_memory_range_mapped_locked(
          allocator, virtual_buffer, virtual_offset, size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_sparse_buffer_bind_sync(
        allocator->sparse_binding_queue,
        iree_hal_buffer_allocation_placement(virtual_buffer), handle,
        /*bind_count=*/1, &bind);
  }
  bool split_tail_mapping_used = false;
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_allocator_unmap_virtual_memory_range_locked(
        allocator, virtual_buffer, virtual_offset, size, split_tail_mapping,
        &split_tail_mapping_used);
  }
  iree_slim_mutex_unlock(&allocator->virtual_memory_mutex);
  if (!split_tail_mapping_used) {
    iree_allocator_free(allocator->host_allocator, split_tail_mapping);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_allocator_virtual_memory_protect(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_virtual_memory_access_scope_t access_scope,
    iree_hal_memory_protection_t protection) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_memory(allocator));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_buffer(virtual_buffer));
  const iree_hal_queue_family_affinity_t reservation_affinity =
      iree_hal_buffer_allocation_placement(virtual_buffer)
          .queue_family_affinity;
  if (!iree_hal_queue_family_affinity_is_any(queue_family_affinity) &&
      (iree_hal_queue_family_affinity_is_empty(queue_family_affinity) ||
       (queue_family_affinity & ~reservation_affinity) != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan virtual protection queue family affinity 0x%016" PRIx64
        " is outside the reservation affinity 0x%016" PRIx64,
        queue_family_affinity, reservation_affinity);
  }
  if (IREE_UNLIKELY(access_scope !=
                    IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Vulkan sparse virtual memory supports only device access scope");
  }
  VkMemoryRequirements memory_requirements = {0};
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_sparse_buffer_memory_requirements(
      virtual_buffer, &memory_requirements));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_validate_sparse_page_size(memory_requirements));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_allocator_validate_sparse_range(
      virtual_offset, size, iree_hal_buffer_allocation_size(virtual_buffer),
      (iree_device_size_t)memory_requirements.alignment, "virtual protect"));
  if (protection == IREE_HAL_MEMORY_PROTECTION_READ_WRITE) {
    iree_slim_mutex_lock(&allocator->virtual_memory_mutex);
    iree_status_t status =
        iree_hal_vulkan_allocator_validate_virtual_memory_range_mapped_locked(
            allocator, virtual_buffer, virtual_offset, size);
    iree_slim_mutex_unlock(&allocator->virtual_memory_mutex);
    return status;
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "Vulkan sparse virtual memory cannot enforce protection flags 0x%" PRIx64
      "; use unmap to revoke access",
      protection);
}

static iree_status_t iree_hal_vulkan_allocator_virtual_memory_advise(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_memory_advice_t advice) {
  iree_hal_vulkan_allocator_t* allocator =
      iree_hal_vulkan_allocator_cast(base_allocator);
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_memory(allocator));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_allocator_require_virtual_buffer(virtual_buffer));
  const iree_hal_queue_family_affinity_t reservation_affinity =
      iree_hal_buffer_allocation_placement(virtual_buffer)
          .queue_family_affinity;
  if (!iree_hal_queue_family_affinity_is_any(queue_family_affinity) &&
      (iree_hal_queue_family_affinity_is_empty(queue_family_affinity) ||
       (queue_family_affinity & ~reservation_affinity) != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan virtual memory advice queue family affinity 0x%016" PRIx64
        " is outside the reservation affinity 0x%016" PRIx64,
        queue_family_affinity, reservation_affinity);
  }
  (void)advice;
  iree_device_size_t range_end = 0;
  if (!iree_device_size_checked_add(virtual_offset, size, &range_end) ||
      range_end > iree_hal_buffer_allocation_size(virtual_buffer)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan sparse virtual memory advice range [%" PRIu64 ", %" PRIu64
        ") exceeds reservation size %" PRIu64,
        (uint64_t)virtual_offset, (uint64_t)range_end,
        (uint64_t)iree_hal_buffer_allocation_size(virtual_buffer));
  }
  return iree_ok_status();
}

static const iree_hal_allocator_vtable_t iree_hal_vulkan_allocator_vtable = {
    .destroy = iree_hal_vulkan_allocator_destroy,
    .host_allocator = iree_hal_vulkan_allocator_host_allocator,
    .query_statistics = iree_hal_vulkan_allocator_query_statistics,
    .query_memory_heaps = iree_hal_vulkan_allocator_query_memory_heaps,
    .supports_virtual_memory =
        iree_hal_vulkan_allocator_supports_virtual_memory,
    .virtual_memory_query_granularity =
        iree_hal_vulkan_allocator_virtual_memory_query_granularity,
    .virtual_memory_reserve = iree_hal_vulkan_allocator_virtual_memory_reserve,
    .virtual_memory_release = iree_hal_vulkan_allocator_virtual_memory_release,
    .physical_memory_allocate =
        iree_hal_vulkan_allocator_physical_memory_allocate,
    .physical_memory_free = iree_hal_vulkan_allocator_physical_memory_free,
    .virtual_memory_map = iree_hal_vulkan_allocator_virtual_memory_map,
    .virtual_memory_unmap = iree_hal_vulkan_allocator_virtual_memory_unmap,
    .virtual_memory_protect = iree_hal_vulkan_allocator_virtual_memory_protect,
    .virtual_memory_advise = iree_hal_vulkan_allocator_virtual_memory_advise,
};
