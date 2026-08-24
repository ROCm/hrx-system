// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_BYTE_TRANSFER_H_
#define IREE_HAL_DRIVERS_VULKAN_BYTE_TRANSFER_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_vulkan_physical_device_snapshot_t
    iree_hal_vulkan_physical_device_snapshot_t;

// Native command implementation selected while capturing a transfer.
typedef enum iree_hal_vulkan_transfer_strategy_e {
  // The complete range is proven dword aligned and uses a Vulkan transfer
  // command.
  IREE_HAL_VULKAN_TRANSFER_STRATEGY_NATIVE = 0,

  // Known unaligned edges use compute patches around an aligned native
  // transfer interior.
  IREE_HAL_VULKAN_TRANSFER_STRATEGY_EDGE_PATCHED_NATIVE = 1,

  // The entire transfer uses a compute implementation with device addresses
  // that are stable when the transfer is captured.
  IREE_HAL_VULKAN_TRANSFER_STRATEGY_COMPUTE = 2,

  // The entire transfer uses a compute implementation whose resolved buffer
  // parameters are loaded from host-published device storage. The publication
  // address is stable across native command buffer replays while its contents
  // are refreshed for each issue.
  IREE_HAL_VULKAN_TRANSFER_STRATEGY_INDIRECT_COMPUTE = 3,

  // The transfer range shape is not known until issue time and must be
  // recorded directly against that issue's resolved binding table.
  IREE_HAL_VULKAN_TRANSFER_STRATEGY_PER_ISSUE_COMPUTE = 4,
} iree_hal_vulkan_transfer_strategy_t;

// Returns true when |strategy| executes a compute transfer pipeline.
static inline bool iree_hal_vulkan_transfer_strategy_uses_compute(
    iree_hal_vulkan_transfer_strategy_t strategy) {
  return strategy == IREE_HAL_VULKAN_TRANSFER_STRATEGY_COMPUTE ||
         strategy == IREE_HAL_VULKAN_TRANSFER_STRATEGY_INDIRECT_COMPUTE ||
         strategy == IREE_HAL_VULKAN_TRANSFER_STRATEGY_PER_ISSUE_COMPUTE;
}

// Device-owned compute pipelines implementing byte-granular transfers.
typedef struct iree_hal_vulkan_byte_transfer_pipelines_t {
  // Device-level Vulkan dispatch table copied at initialization.
  iree_hal_vulkan_device_syms_t syms;

  // Vulkan logical device owning all pipeline handles.
  VkDevice logical_device;

  // Maximum X-dimension workgroup count supported by the physical device.
  uint32_t max_workgroup_count_x;

  // Push-constant-only layout shared by the byte transfer pipelines.
  VkPipelineLayout pipeline_layout;

  // Compute pipeline implementing byte-granular buffer copies.
  VkPipeline copy_pipeline;

  // Compute pipeline implementing byte-granular buffer fills.
  VkPipeline fill_pipeline;
} iree_hal_vulkan_byte_transfer_pipelines_t;

// Initializes byte-granular transfer pipelines for |logical_device|.
iree_status_t iree_hal_vulkan_byte_transfer_pipelines_initialize(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_hal_vulkan_byte_transfer_pipelines_t* out_pipelines);

// Deinitializes byte-granular transfer pipelines and releases device handles.
void iree_hal_vulkan_byte_transfer_pipelines_deinitialize(
    iree_hal_vulkan_byte_transfer_pipelines_t* pipelines);

// Records a compute copy of |length| bytes between buffer device addresses.
iree_status_t iree_hal_vulkan_byte_transfer_record_copy(
    const iree_hal_vulkan_byte_transfer_pipelines_t* pipelines,
    VkCommandBuffer command_buffer, VkDeviceAddress source_address,
    VkDeviceAddress target_address, VkDeviceSize length);

// Returns the host publication storage required by an indirect copy.
iree_status_t iree_hal_vulkan_byte_transfer_copy_publication_length(
    VkDeviceSize length, iree_device_size_t* out_publication_length);

// Publishes resolved indirect copy parameters into |publication_span|.
iree_status_t iree_hal_vulkan_byte_transfer_publish_copy(
    VkDeviceAddress source_address, VkDeviceAddress target_address,
    VkDeviceSize length, iree_byte_span_t publication_span);

// Records an indirect compute copy whose parameters are read from the stable
// device range beginning at |publication_address|.
iree_status_t iree_hal_vulkan_byte_transfer_record_copy_indirect(
    const iree_hal_vulkan_byte_transfer_pipelines_t* pipelines,
    VkCommandBuffer command_buffer, VkDeviceAddress publication_address,
    VkDeviceSize length);

// Records a compute fill of |length| bytes at |target_address|.
iree_status_t iree_hal_vulkan_byte_transfer_record_fill(
    const iree_hal_vulkan_byte_transfer_pipelines_t* pipelines,
    VkCommandBuffer command_buffer, VkDeviceAddress target_address,
    VkDeviceSize length, const uint8_t* pattern,
    iree_host_size_t pattern_length);

// Returns the host publication storage required by an indirect fill.
iree_status_t iree_hal_vulkan_byte_transfer_fill_publication_length(
    VkDeviceSize length, iree_device_size_t* out_publication_length);

// Publishes resolved indirect fill parameters into |publication_span|.
iree_status_t iree_hal_vulkan_byte_transfer_publish_fill(
    VkDeviceAddress target_address, VkDeviceSize length,
    iree_byte_span_t publication_span);

// Records an indirect compute fill whose parameters are read from the stable
// device range beginning at |publication_address|.
iree_status_t iree_hal_vulkan_byte_transfer_record_fill_indirect(
    const iree_hal_vulkan_byte_transfer_pipelines_t* pipelines,
    VkCommandBuffer command_buffer, VkDeviceAddress publication_address,
    VkDeviceSize length, const uint8_t* pattern,
    iree_host_size_t pattern_length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_BYTE_TRANSFER_H_
