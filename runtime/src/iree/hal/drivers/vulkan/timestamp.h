// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_TIMESTAMP_H_
#define IREE_HAL_DRIVERS_VULKAN_TIMESTAMP_H_

#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Queue-owned lazy pipeline for delivering resolved ticks through BDA bindings.
// Initialization is serialized by the queue's native command-pool lock.
typedef struct iree_hal_vulkan_timestamp_pipeline_t {
  // Pipeline layout accepting a source address and a destination-pointer slot.
  VkPipelineLayout layout;
  // Compute pipeline copying one complete eight-byte tick.
  VkPipeline handle;
} iree_hal_vulkan_timestamp_pipeline_t;

// Mutable timestamp resources owned by one native command-buffer cache slot.
// The slot's lease prevents reset, writes, or destruction while it is in
// flight. Cached recordings retain the lease, so their embedded query/buffer
// handles remain stable. Zero initialization creates an empty state.
typedef struct iree_hal_vulkan_timestamp_state_t {
  // Query pool exclusive to the leased native command buffer.
  VkQueryPool queries;
  // Number of slots allocated in queries.
  uint32_t query_capacity;
  // Number of eight-byte values allocated in resolve_buffer.
  uint32_t resolve_capacity;
  // Private device storage for query results delivered by the resolve shader.
  iree_hal_buffer_t* resolve_buffer;
  // Native buffer backing resolve_buffer.
  VkBuffer resolve_handle;
  // Native byte offset corresponding to resolve_buffer byte zero.
  VkDeviceSize resolve_offset;
  // Device address corresponding to resolve_buffer byte zero.
  VkDeviceAddress resolve_address;
  // Queue-owned pipeline, borrowed while resolve_buffer is present.
  const iree_hal_vulkan_timestamp_pipeline_t* pipeline;
} iree_hal_vulkan_timestamp_state_t;

// Ensures capacity while the native command-buffer slot is exclusively leased
// for recording. Existing resources are reused when large enough. Native
// allocation and shader compilation occur only on first use or capacity growth.
iree_status_t iree_hal_vulkan_timestamp_state_prepare(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_allocator_t* device_allocator,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    uint32_t query_count, uint32_t resolve_count,
    iree_hal_vulkan_timestamp_pipeline_t* pipeline,
    iree_hal_vulkan_timestamp_state_t* state);

// Releases retired native command-buffer resources.
void iree_hal_vulkan_timestamp_state_deinitialize(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_vulkan_timestamp_state_t* state);

// Releases the queue-owned pipeline after all native recordings retire.
void iree_hal_vulkan_timestamp_pipeline_deinitialize(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_vulkan_timestamp_pipeline_t* pipeline);

// Captures a query and writes it directly to a native transfer destination.
// The caller resets the query before capture. Execution dependencies do not
// imply payload memory visibility; ordinary HAL barriers own that contract.
void iree_hal_vulkan_timestamp_record_direct(
    const iree_hal_vulkan_device_syms_t* syms, VkQueueFlags queue_flags,
    VkCommandBuffer command_buffer, VkQueryPool queries, uint32_t query_index,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask, VkBuffer target_buffer,
    VkDeviceSize target_offset);

// Captures through private resolve storage and copies the tick to the address
// in |target_slot_address|. Host publication of that slot must be visible to
// compute reads before this call. Each live capture uses its own resolve index.
void iree_hal_vulkan_timestamp_record_indirect(
    const iree_hal_vulkan_device_syms_t* syms, VkQueueFlags queue_flags,
    VkCommandBuffer command_buffer,
    const iree_hal_vulkan_timestamp_state_t* state, uint32_t query_index,
    uint32_t resolve_index, iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    VkDeviceAddress target_slot_address);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_TIMESTAMP_H_
