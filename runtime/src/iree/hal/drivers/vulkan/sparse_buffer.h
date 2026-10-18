// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_SPARSE_BUFFER_H_
#define IREE_HAL_DRIVERS_VULKAN_SPARSE_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/api.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"
#include "iree/hal/memory/tracing.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_sparse_buffer_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vulkan_queue_t iree_hal_vulkan_queue_t;

// Queries sparse buffer geometry without acquiring addresses or physical
// storage. The caller has qualified sparse residency/aliasing feature support.
// Alignment and memory types are fixed by flags and usage, independently of
// buffer size and queue sharing mode.
VkMemoryRequirements iree_hal_vulkan_sparse_buffer_query_requirements(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_vulkan_features_t enabled_features, iree_hal_buffer_usage_t usage);

// Creates a fully-bound sparse Vulkan buffer.
//
// Sparse binding lets one VkBuffer span multiple VkDeviceMemory allocations.
// This is the primitive the allocator uses when a requested logical allocation
// is larger than maxMemoryAllocationSize. The returned HAL buffer owns |handle|
// and all allocated physical blocks.
//
// Sparse buffers are not host-mappable in this primitive. Independently mapped
// addresses use virtual_buffer.h; this helper creates fully resident
// ordinary storage.
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
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Returns true if |buffer| is a Vulkan sparse HAL buffer.
bool iree_hal_vulkan_sparse_buffer_isa(iree_hal_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_SPARSE_BUFFER_H_
