// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_MEMORY_BACKEND_H_
#define IREE_HAL_DRIVERS_VULKAN_MEMORY_BACKEND_H_

#include "iree/hal/drivers/vulkan/allocator.h"
#include "iree/hal/memory_backend.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Borrowed native owners established before the device group is published.
// The containing device and sealed group outlive all constructed pools.
typedef struct iree_hal_vulkan_memory_backend_t {
  // Generic native schema and shared Vulkan construction factory.
  iree_hal_memory_backend_t base;
  // Native allocation owner used for buffer placement metadata.
  iree_hal_device_t* device;
  // Existing whole-buffer allocation and sparse-binding services.
  iree_hal_vulkan_allocator_t* allocator;
  // Existing sparse publication queue, or NULL when sparse binding is disabled.
  iree_hal_vulkan_queue_t* sparse_binding_queue;
  // Device-owned native dispatch table.
  const iree_hal_vulkan_device_syms_t* syms;
  // Native resource and memory namespace.
  VkDevice logical_device;
  // Device-owned immutable native property snapshot used during construction.
  const iree_hal_vulkan_physical_device_snapshot_t* physical_device;
  // Features actually enabled on this native owner.
  iree_hal_vulkan_features_t enabled_features;
  // Existing placement-local capacity notification.
  iree_async_notification_t* notification;
  // Device-owned native slab allocation/free queue.
  iree_hal_vulkan_allocation_queue_t* allocation_queue;
  // Existing completion probe with a group-lifetime borrowed context.
  iree_hal_pool_epoch_query_t epoch_query;
} iree_hal_vulkan_memory_backend_t;

// Publishes the factory header after the caller fills the borrowed native and
// progress owners. Creates no native objects, payload, or allocation policy.
void iree_hal_vulkan_memory_backend_initialize(
    iree_hal_vulkan_memory_backend_t* backend);

// Qualifies a complete Vulkan address scope without acquiring native resources.
iree_status_t iree_hal_vulkan_virtual_buffer_query_capabilities(
    const iree_hal_memory_backend_t* backend, iree_hal_device_group_t* group,
    iree_hal_pool_scope_t scope, iree_allocator_t host_allocator,
    iree_hal_virtual_buffer_capabilities_t* out_capabilities);

// Creates an ordinary virtual HAL buffer owning one sparse VkBuffer.
iree_status_t iree_hal_vulkan_virtual_buffer_create(
    const iree_hal_memory_backend_t* backend, iree_hal_device_group_t* group,
    iree_hal_pool_scope_t scope,
    const iree_hal_virtual_buffer_options_t* options,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Identifies a Vulkan reservation's allocated buffer before native binding use.
bool iree_hal_vulkan_virtual_buffer_isa(iree_hal_buffer_t* buffer);

// Checks native owners and permission support before queue capture is accepted.
iree_status_t iree_hal_vulkan_memory_mapping_batch_validate(
    const iree_hal_memory_mapping_batch_t* batch, VkDevice logical_device);

// Resolves committed sources and fills preallocated native vectors. Called only
// after allocation dependencies publish. No mappings change until submission.
iree_status_t iree_hal_vulkan_memory_mapping_batch_resolve(
    iree_hal_memory_mapping_batch_t* batch, VkDevice logical_device,
    VkSparseBufferMemoryBindInfo* buffers, VkSparseMemoryBind* binds);

// Poisons the batch's address owners after a failed native publication. Their
// mappings are no longer certified; checked detach remains available.
void iree_hal_vulkan_memory_mapping_batch_fail(
    iree_hal_memory_mapping_batch_t* batch, iree_status_code_t failure_code);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_MEMORY_BACKEND_H_
