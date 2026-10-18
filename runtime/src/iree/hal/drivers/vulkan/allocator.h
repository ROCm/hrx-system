// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_VULKAN_ALLOCATOR_H_

#include "iree/async/api.h"
#include "iree/base/api.h"
#include "iree/base/threading/affinity.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/physical_device.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_allocator_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vulkan_allocator_t iree_hal_vulkan_allocator_t;
typedef struct iree_hal_vulkan_allocation_queue_t
    iree_hal_vulkan_allocation_queue_t;
typedef struct iree_hal_vulkan_queue_t iree_hal_vulkan_queue_t;

typedef uint32_t iree_hal_vulkan_buffer_allocation_flags_t;
enum iree_hal_vulkan_buffer_allocation_flag_bits_e {
  IREE_HAL_VULKAN_BUFFER_ALLOCATION_FLAG_NONE = 0,
  // Requires one dense allocation that can back additional sparse resources.
  // Allocation fails when native size limits require several physical blocks.
  IREE_HAL_VULKAN_BUFFER_ALLOCATION_FLAG_MAPPING_SOURCE = 1u << 0,
};

// Exact native properties used to create one Vulkan buffer.
//
// This is private allocation policy consumed by the Vulkan allocator and slab
// provider. Public pools fix these properties at construction time; individual
// reservations carry only their size and alignment.
typedef struct iree_hal_vulkan_buffer_spec_t {
  // Memory properties required or preferred during memory-type selection.
  iree_hal_memory_type_t type;
  // Access permissions published by the resulting HAL buffer.
  iree_hal_memory_access_t access;
  // Native usage fixed when the VkBuffer is created.
  iree_hal_buffer_usage_t usage;
  // HAL queue families permitted to access the VkBuffer.
  iree_hal_queue_family_affinity_t queue_family_affinity;
  // Minimum required native address alignment, or zero for the native default.
  iree_device_size_t min_alignment;
  // Native backing constraints established by the constructing pool.
  iree_hal_vulkan_buffer_allocation_flags_t flags;
} iree_hal_vulkan_buffer_spec_t;

// Maps one canonical HAL queue family ordinal to its Vulkan queue family.
// Array position is the HAL queue family ordinal.
typedef struct iree_hal_vulkan_allocator_queue_family_t {
  // Native Vulkan queue family index backing the HAL queue family.
  uint32_t native_family_index;
} iree_hal_vulkan_allocator_queue_family_t;

// Creates the Vulkan allocator object for a logical device.
//
// Slab providers delegate whole-slab materialization through the direct
// allocation helpers below so Vulkan object creation and sparse-binding rules
// remain centralized.
iree_status_t iree_hal_vulkan_allocator_create(
    iree_hal_device_t* parent_device, const iree_hal_vulkan_device_syms_t* syms,
    VkDevice logical_device,
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_hal_vulkan_features_t enabled_features,
    iree_hal_vulkan_device_extensions_t enabled_extensions,
    iree_host_size_t queue_family_count,
    const iree_hal_vulkan_allocator_queue_family_t* queue_families,
    iree_hal_vulkan_queue_t* sparse_binding_queue,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator);

// Creates the native allocation queue and capacity notification for this
// device. The caller keeps |proactor| alive through service teardown.
iree_status_t iree_hal_vulkan_allocator_initialize_memory_services(
    iree_hal_allocator_t* base_allocator, iree_async_proactor_t* proactor,
    iree_thread_affinity_t memory_affinity);

// Joins native allocation/free before releasing the queue and notification.
void iree_hal_vulkan_allocator_deinitialize_memory_services(
    iree_hal_allocator_t* base_allocator);

// Borrows the notification shared by explicit Vulkan memory pools. Available
// after memory-service initialization and until deinitialization.
iree_async_notification_t* iree_hal_vulkan_allocator_memory_notification(
    iree_hal_allocator_t* base_allocator);

// Borrows the device-owned native allocation queue. Available after
// memory-service initialization and until deinitialization.
iree_hal_vulkan_allocation_queue_t* iree_hal_vulkan_allocator_allocation_queue(
    iree_hal_allocator_t* base_allocator);

// Allocates one whole Vulkan buffer from a required memory type index.
//
// This is the primitive used by Vulkan slab providers and private staging. It
// creates a standalone dense or fully-bound sparse buffer without consulting
// any application pool inventory.
iree_status_t iree_hal_vulkan_allocator_allocate_direct_buffer_from_type(
    iree_hal_vulkan_allocator_t* allocator, uint32_t memory_type_index,
    const iree_hal_vulkan_buffer_spec_t* spec,
    iree_device_size_t allocation_size, iree_hal_buffer_t** out_buffer);

// Returns whether this allocator can attach ordinary host allocations as
// Vulkan memory. The extension enables this route on both integrated and
// discrete devices; attachment queries the actual pointer's compatible types.
bool iree_hal_vulkan_allocator_supports_host_allocation_import(
    const iree_hal_vulkan_allocator_t* allocator);

// Attaches one external host allocation through a compatible Vulkan memory
// type selected from |allowed_memory_type_bits|.
//
// The returned root buffer independently owns the Vulkan attachment and never
// consumes or releases |storage|'s transport owner. |storage| must remain alive
// until the returned buffer is released. Pool providers prequalify the allowed
// types against their immutable memory contract; the actual pointer and
// external buffer resource may further narrow the set at import time.
iree_status_t iree_hal_vulkan_allocator_import_direct_buffer(
    iree_hal_vulkan_allocator_t* allocator, uint32_t allowed_memory_type_bits,
    const iree_hal_vulkan_buffer_spec_t* spec,
    const iree_hal_external_storage_t* storage, iree_hal_buffer_t** out_buffer);

// Allocates one whole Vulkan buffer directly from a compatible memory type.
//
// The caller has exclusive ownership of the underlying VkDeviceMemory mapping
// lifetime.
iree_status_t iree_hal_vulkan_allocator_allocate_direct_buffer(
    iree_hal_allocator_t* base_allocator,
    const iree_hal_vulkan_buffer_spec_t* spec,
    iree_device_size_t allocation_size, iree_hal_buffer_t** out_buffer);

// Allocates private staging storage backed by ordinary host pages when host
// import is available. |out_host_allocation| then owns the original allocation
// and must be freed with iree_allocator_free_aligned after releasing
// |out_buffer|. Native I/O uses this original pointer: vkMapMemory may return a
// different alias whose native page-pinning properties are not guaranteed.
//
// Without host import, returns a native Vulkan allocation and an empty host
// allocation. That route supports host copies and buffered files, not direct
// native files. Both outputs are empty on failure. This is cold ring setup;
// transfer submission neither allocates backing nor discovers capabilities.
iree_status_t iree_hal_vulkan_allocator_allocate_staging_buffer(
    iree_hal_allocator_t* base_allocator,
    const iree_hal_vulkan_buffer_spec_t* spec,
    iree_device_size_t allocation_size, iree_allocator_t host_allocator,
    iree_byte_span_t* out_host_allocation, iree_hal_buffer_t** out_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_ALLOCATOR_H_
