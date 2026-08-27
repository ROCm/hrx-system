// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_ATOMIC_H_
#define IREE_HAL_DRIVERS_VULKAN_ATOMIC_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/api.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Device-owned pipelines implementing Vulkan HAL atomic operations.
typedef struct iree_hal_vulkan_atomic_pipelines_t {
  // Device-level Vulkan dispatch table copied at initialization.
  iree_hal_vulkan_device_syms_t syms;

  // Vulkan logical device owning all handles.
  VkDevice logical_device;

  // Pipeline layout shared by both atomic widths.
  VkPipelineLayout pipeline_layout;

  // Pipeline implementing 32-bit atomic operations, or VK_NULL_HANDLE.
  VkPipeline pipeline_32;

  // Pipeline implementing 64-bit atomic operations, or VK_NULL_HANDLE.
  VkPipeline pipeline_64;
} iree_hal_vulkan_atomic_pipelines_t;

// Internal selector consumed by the built-in atomic shaders.
typedef uint32_t iree_hal_vulkan_atomic_operation_t;
typedef enum iree_hal_vulkan_atomic_operation_e {
  IREE_HAL_VULKAN_ATOMIC_OPERATION_WAIT_EQUAL = 0,
  IREE_HAL_VULKAN_ATOMIC_OPERATION_WAIT_NOT_EQUAL = 1,
  IREE_HAL_VULKAN_ATOMIC_OPERATION_WAIT_UNSIGNED_GREATER_EQUAL = 2,
  IREE_HAL_VULKAN_ATOMIC_OPERATION_STORE = 3,
  IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_ADD = 4,
  IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_SUBTRACT = 5,
  IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_AND = 6,
  IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_OR = 7,
  IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_XOR = 8,
} iree_hal_vulkan_atomic_operation_e;

// Validated atomic operation encoded for the built-in shader.
typedef struct iree_hal_vulkan_atomic_params_t {
  // Value compared, stored, or used as the RMW operand.
  uint64_t value;

  // Mask applied by wait operations, or zero for modifications.
  uint64_t mask;

  // HAL ordering and visibility semantics.
  iree_hal_atomic_flags_t flags;

  // Width of the target value.
  iree_hal_atomic_width_t width;

  // Built-in shader operation selector.
  iree_hal_vulkan_atomic_operation_t operation;
} iree_hal_vulkan_atomic_params_t;

// Flags controlling how an atomic operation is recorded.
typedef uint32_t iree_hal_vulkan_atomic_record_flags_t;
typedef enum iree_hal_vulkan_atomic_record_flag_bits_e {
  IREE_HAL_VULKAN_ATOMIC_RECORD_FLAG_NONE = 0u,

  // target_address points to a uint64_t containing the final target address.
  IREE_HAL_VULKAN_ATOMIC_RECORD_FLAG_INDIRECT_TARGET = 1u << 0,
} iree_hal_vulkan_atomic_record_flag_bits_e;

// Initializes optional atomic pipelines supported by |enabled_features|.
iree_status_t iree_hal_vulkan_atomic_pipelines_initialize(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_vulkan_features_t enabled_features,
    iree_hal_vulkan_atomic_pipelines_t* out_pipelines);

// Deinitializes atomic pipelines and releases device handles.
void iree_hal_vulkan_atomic_pipelines_deinitialize(
    iree_hal_vulkan_atomic_pipelines_t* pipelines);

// Returns the atomic capabilities implemented by |enabled_features|.
static inline iree_hal_atomic_capabilities_t
iree_hal_vulkan_atomic_capabilities(
    iree_hal_vulkan_features_t enabled_features) {
  iree_hal_atomic_capabilities_t capabilities;
  memset(&capabilities, 0, sizeof(capabilities));
  const iree_hal_vulkan_features_t required_features =
      IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES |
      IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL |
      IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE;
  if (!iree_all_bits_set(enabled_features, required_features)) {
    return capabilities;
  }
  capabilities.operations.device_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL;
  capabilities.wait_conditions.device_scope_32 =
      IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL;
  if (iree_all_bits_set(
          enabled_features,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BUFFER_INT64_ATOMICS)) {
    capabilities.operations.device_scope_64 =
        IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL;
    capabilities.wait_conditions.device_scope_64 =
        IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL;
  }
  return capabilities;
}

// Returns shader parameters for one validated HAL operation.
iree_hal_vulkan_atomic_params_t iree_hal_vulkan_atomic_params_from_wait(
    iree_hal_atomic_wait_params_t params);
iree_hal_vulkan_atomic_params_t iree_hal_vulkan_atomic_params_from_store(
    iree_hal_atomic_store_params_t params);
iree_hal_vulkan_atomic_params_t iree_hal_vulkan_atomic_params_from_rmw(
    iree_hal_atomic_rmw_params_t params);

// Validates that |pipelines| can execute |params|.
iree_status_t iree_hal_vulkan_atomic_validate(
    const iree_hal_vulkan_atomic_pipelines_t* pipelines,
    iree_hal_vulkan_atomic_params_t params);

// Returns Vulkan shader accesses performed by |params|.
VkAccessFlags2 iree_hal_vulkan_atomic_access_mask(
    iree_hal_vulkan_atomic_params_t params);

// Validates that |target_address| is naturally aligned for |width|.
iree_status_t iree_hal_vulkan_atomic_validate_target_address(
    VkDeviceAddress target_address, iree_hal_atomic_width_t width);

// Resolves one retained HAL buffer target to a final naturally aligned BDA.
iree_status_t iree_hal_vulkan_atomic_resolve_target_address(
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_width_t width, VkDeviceAddress* out_target_address);

// Records one validated built-in atomic operation.
//
// |target_address| is the final target BDA unless INDIRECT_TARGET is set, in
// which case it addresses an aligned uint64_t publication slot containing the
// final target BDA.
void iree_hal_vulkan_atomic_record(
    const iree_hal_vulkan_atomic_pipelines_t* pipelines,
    VkCommandBuffer command_buffer, VkDeviceAddress target_address,
    iree_hal_vulkan_atomic_record_flags_t record_flags,
    iree_hal_vulkan_atomic_params_t params);

// Records one atomic operation and its HAL execution-stage dependencies.
void iree_hal_vulkan_atomic_record_command(
    const iree_hal_vulkan_device_syms_t* syms,
    const iree_hal_vulkan_atomic_pipelines_t* pipelines,
    VkCommandBuffer command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    VkDeviceAddress target_address,
    iree_hal_vulkan_atomic_record_flags_t record_flags,
    iree_hal_vulkan_atomic_params_t params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_ATOMIC_H_
