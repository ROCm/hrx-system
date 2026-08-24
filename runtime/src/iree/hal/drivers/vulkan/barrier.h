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

// Flags controlling HAL-to-Vulkan barrier expansion.
typedef uint32_t iree_hal_vulkan_barrier_flags_t;
typedef enum iree_hal_vulkan_barrier_flag_bits_e {
  IREE_HAL_VULKAN_BARRIER_FLAG_NONE = 0u,

  // The source scope contains transfers implemented by compute pipelines.
  IREE_HAL_VULKAN_BARRIER_FLAG_COMPUTE_TRANSFER_SOURCE = 1u << 0,

  // The target scope contains transfers implemented by compute pipelines.
  IREE_HAL_VULKAN_BARRIER_FLAG_COMPUTE_TRANSFER_TARGET = 1u << 1,
} iree_hal_vulkan_barrier_flag_bits_e;

// Describes a Vulkan memory barrier using HAL execution stages.
typedef struct iree_hal_vulkan_barrier_t {
  // Controls how HAL execution stages expand to Vulkan stages and accesses.
  iree_hal_vulkan_barrier_flags_t flags;

  // HAL execution stages producing the dependency.
  iree_hal_execution_stage_t source_stage_mask;

  // Vulkan memory accesses made available by the dependency.
  VkAccessFlags2 source_access_mask;

  // HAL execution stages consuming the dependency.
  iree_hal_execution_stage_t target_stage_mask;

  // Vulkan memory accesses made visible by the dependency.
  VkAccessFlags2 target_access_mask;
} iree_hal_vulkan_barrier_t;

// Returns Vulkan write accesses produced by |stage_mask|.
VkAccessFlags2 iree_hal_vulkan_barrier_source_access_mask(
    iree_hal_execution_stage_t stage_mask,
    iree_hal_vulkan_barrier_flags_t flags);

// Returns Vulkan read/write accesses consumed by |stage_mask|.
VkAccessFlags2 iree_hal_vulkan_barrier_target_access_mask(
    iree_hal_execution_stage_t stage_mask,
    iree_hal_vulkan_barrier_flags_t flags);

// Records |barrier| into |command_buffer|.
void iree_hal_vulkan_barrier_record(const iree_hal_vulkan_device_syms_t* syms,
                                    VkCommandBuffer command_buffer,
                                    const iree_hal_vulkan_barrier_t* barrier);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_BARRIER_H_
