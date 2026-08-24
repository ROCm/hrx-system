// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_BUILTINS_H_
#define IREE_HAL_DRIVERS_VULKAN_BUILTINS_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/atomic.h"
#include "iree/hal/drivers/vulkan/byte_transfer.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_vulkan_physical_device_snapshot_t
    iree_hal_vulkan_physical_device_snapshot_t;

// Device-owned Vulkan built-in pipelines used for command polyfills.
typedef struct iree_hal_vulkan_builtins_t {
  // Device-level Vulkan dispatch table copied at initialization.
  iree_hal_vulkan_device_syms_t syms;

  // Vulkan logical device owning all built-in handles.
  VkDevice logical_device;

  // Push-constant-only layout shared by the edge-patch pipelines.
  VkPipelineLayout edge_patch_pipeline_layout;

  // Compute pipeline patching partial dwords for unaligned fills.
  VkPipeline fill_edge_pipeline;

  // Compute pipeline patching partial dwords for unaligned updates.
  VkPipeline update_edge_pipeline;

  // BDA compute pipelines implementing byte-granular fills and copies.
  iree_hal_vulkan_byte_transfer_pipelines_t byte_transfer_pipelines;

  // BDA pipelines implementing atomic wait, store, and RMW operations.
  iree_hal_vulkan_atomic_pipelines_t atomic_pipelines;
} iree_hal_vulkan_builtins_t;

// Initializes built-in Vulkan pipelines for |logical_device|.
iree_status_t iree_hal_vulkan_builtins_initialize(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_hal_vulkan_features_t enabled_features,
    iree_hal_vulkan_builtins_t* out_builtins);

// Deinitializes built-in Vulkan pipelines and releases device handles.
void iree_hal_vulkan_builtins_deinitialize(
    iree_hal_vulkan_builtins_t* builtins);

// Records shader patches for the unaligned edges of a buffer fill.
//
// The aligned interior, if any, remains the caller's responsibility and should
// use vkCmdFillBuffer. Edge dwords are patched atomically so concurrent fills
// of disjoint byte ranges cannot conflict through implementation widening.
// |target_address| identifies the first byte of the complete fill range.
iree_status_t iree_hal_vulkan_builtins_record_fill_edges(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    VkDeviceAddress target_address, VkDeviceSize length, const uint8_t* pattern,
    iree_host_size_t pattern_length);

// Records shader patches for the unaligned edges of a buffer update.
//
// The aligned interior, if any, remains the caller's responsibility and should
// use vkCmdUpdateBuffer in aligned chunks. Edge dwords are patched atomically
// so concurrent updates of disjoint byte ranges cannot conflict through
// implementation widening. |target_address| identifies the first byte of the
// complete update range.
iree_status_t iree_hal_vulkan_builtins_record_update_edges(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    VkDeviceAddress target_address, VkDeviceSize length,
    const uint8_t* source_data, iree_host_size_t source_data_length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_BUILTINS_H_
