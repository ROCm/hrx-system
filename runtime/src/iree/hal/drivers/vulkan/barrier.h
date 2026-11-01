// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_BARRIER_H_
#define IREE_HAL_DRIVERS_VULKAN_BARRIER_H_

#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Describes a Vulkan memory barrier using HAL execution stages.
typedef struct iree_hal_vulkan_barrier_t {
  // HAL execution stages producing the dependency.
  iree_hal_execution_stage_t source_stage_mask;

  // HAL execution stages consuming the dependency.
  iree_hal_execution_stage_t target_stage_mask;

  // Memory accesses made available, before transfer polyfill expansion.
  VkAccessFlags2 source_access_mask;

  // Memory accesses made visible, before transfer polyfill expansion.
  VkAccessFlags2 target_access_mask;
} iree_hal_vulkan_barrier_t;

// Captured native boundaries; no descriptor or buffer pointers are retained.
typedef struct iree_hal_vulkan_queue_barriers_t {
  // Global dependency before the logical operation.
  iree_hal_vulkan_barrier_t before;
  // Global dependency after all operation children and staging have completed.
  iree_hal_vulkan_barrier_t after;
} iree_hal_vulkan_queue_barriers_t;

// Resolves stages and global visibility flags into one native dependency.
iree_hal_vulkan_barrier_t iree_hal_vulkan_barrier_resolve(
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_barrier_flags_t flags, bool has_memory_visibility);

// Captures validated direct-operation boundaries, conservatively promoting
// buffer ranges to global visibility. NULL boundaries select broad system
// visibility; explicit empty lists produce empty native dependencies.
iree_hal_vulkan_queue_barriers_t iree_hal_vulkan_queue_barriers_resolve(
    const iree_hal_queue_barriers_t* barriers);

// Returns whether recording this dependency would emit no commands.
static inline bool iree_hal_vulkan_barrier_is_empty(
    const iree_hal_vulkan_barrier_t* barrier) {
  return !(barrier->source_stage_mask | barrier->target_stage_mask |
           barrier->source_access_mask | barrier->target_access_mask);
}

// Returns Vulkan write accesses produced by |stage_mask|.
VkAccessFlags2 iree_hal_vulkan_barrier_source_access_mask(
    iree_hal_execution_stage_t stage_mask);

// Returns Vulkan read/write accesses consumed by |stage_mask|.
VkAccessFlags2 iree_hal_vulkan_barrier_target_access_mask(
    iree_hal_execution_stage_t stage_mask);

// Maps explicit HAL access scopes to Vulkan synchronization2 access flags.
VkAccessFlags2 iree_hal_vulkan_access_scope_mask(
    iree_hal_access_scope_t access_scope);

// Maps logical stages to the native engines that can implement them. An empty
// scope remains empty unless memory accesses require an unscoped dependency.
VkPipelineStageFlags2 iree_hal_vulkan_barrier_stage_mask(
    VkQueueFlags queue_flags, iree_hal_execution_stage_t stage_mask,
    VkAccessFlags2 access_mask);

// Records |barrier| into |command_buffer|. Transfer stages and accesses include
// compute polyfills only when |queue_flags| permits compute commands.
void iree_hal_vulkan_barrier_record(const iree_hal_vulkan_device_syms_t* syms,
                                    VkQueueFlags queue_flags,
                                    VkCommandBuffer command_buffer,
                                    const iree_hal_vulkan_barrier_t* barrier);

// Records |barrier| over one exact native buffer range.
void iree_hal_vulkan_buffer_barrier_record(
    const iree_hal_vulkan_device_syms_t* syms, VkQueueFlags queue_flags,
    VkCommandBuffer command_buffer, const iree_hal_vulkan_barrier_t* barrier,
    VkBuffer buffer, VkDeviceSize offset, VkDeviceSize length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_BARRIER_H_
