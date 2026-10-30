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
  // Pipeline layout accepting source/destination tables and a value count.
  VkPipelineLayout layout;
  // Compute pipeline scattering complete eight-byte ticks.
  VkPipeline handle;
} iree_hal_vulkan_timestamp_pipeline_t;

// One captured query awaiting delivery. Populated during native recording;
// cached replay executes the resulting commands without revisiting this table.
typedef struct iree_hal_vulkan_timestamp_pending_t {
  // Borrowed resolved target buffer; NULL once delivered.
  iree_hal_buffer_t* buffer;
  // Byte offset in buffer, including the submission's binding displacement.
  iree_device_size_t offset;
  // Native direct destination, or NULL for binding-relative shader delivery.
  VkBuffer target_handle;
  // Byte offset in target_handle, or in the private indirect result/address
  // tables when target_handle is NULL.
  VkDeviceSize target_offset;
} iree_hal_vulkan_timestamp_pending_t;

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
  // Reusable native-recording workspace, indexed by query ordinal.
  iree_hal_vulkan_timestamp_pending_t* pending;
  // Number of entries allocated in pending.
  uint32_t pending_capacity;
} iree_hal_vulkan_timestamp_state_t;

// Invocation-local cursor over the exclusively leased native-recording
// workspace.
typedef struct iree_hal_vulkan_timestamp_recording_t {
  // Resources leased for this native recording; NULL when there are no queries.
  const iree_hal_vulkan_timestamp_state_t* state;
  // Device address of the contiguous published indirect destination table.
  VkDeviceAddress target_table;
  // First captured query not yet delivered, or count when none remain.
  uint32_t first_pending;
  // Number of queries captured so far.
  uint32_t count;
} iree_hal_vulkan_timestamp_recording_t;

// Ensures capacity while the native command-buffer slot is exclusively leased
// for recording. Existing resources are reused when large enough. Native
// allocation and shader compilation occur only on first use or capacity growth.
iree_status_t iree_hal_vulkan_timestamp_state_prepare(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_allocator_t* device_allocator,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    uint32_t query_count, uint32_t resolve_count,
    iree_hal_vulkan_timestamp_pipeline_t* pipeline,
    iree_allocator_t host_allocator, iree_hal_vulkan_timestamp_state_t* state);

// Releases retired native command-buffer resources.
void iree_hal_vulkan_timestamp_state_deinitialize(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_allocator_t host_allocator, iree_hal_vulkan_timestamp_state_t* state);

// Releases the queue-owned pipeline after all native recordings retire.
void iree_hal_vulkan_timestamp_pipeline_deinitialize(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_vulkan_timestamp_pipeline_t* pipeline);

// Captures after the requested source stages and before the target stages.
// The caller resets the query before capture. No result delivery is recorded.
void iree_hal_vulkan_timestamp_record_capture(
    const iree_hal_vulkan_device_syms_t* syms, VkQueueFlags queue_flags,
    VkCommandBuffer command_buffer, VkQueryPool queries, uint32_t query_index,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask);

// Captures and immediately delivers a standalone queue timestamp.
void iree_hal_vulkan_timestamp_record_direct(
    const iree_hal_vulkan_device_syms_t* syms, VkQueueFlags queue_flags,
    VkCommandBuffer command_buffer, VkQueryPool queries, uint32_t query_index,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask, VkBuffer target_buffer,
    VkDeviceSize target_offset);

// Delivers pending queries overlapping a resolved range, or all queries if
// |range.buffer| is NULL. Adjacent queries share native copies and scatter
// dispatches. The caller records the consuming visibility dependency afterward.
void iree_hal_vulkan_timestamp_record_delivery(
    const iree_hal_vulkan_device_syms_t* syms, VkCommandBuffer command_buffer,
    iree_hal_vulkan_timestamp_recording_t* recording,
    iree_hal_buffer_ref_t range);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_TIMESTAMP_H_
