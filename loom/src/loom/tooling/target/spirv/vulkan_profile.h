// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vulkan HAL device facts used to select SPIR-V target bundles.
//
// This package is the target-specific runtime edge for Vulkan/SPIR-V: it
// converts the active HAL device and executable-cache capabilities into compact
// target facts. It does not run kernels, own invocation, perform correctness
// checks, or emit target binaries.

#ifndef LOOM_TOOLING_EXECUTION_HAL_SPIRV_VULKAN_PROFILE_H_
#define LOOM_TOOLING_EXECUTION_HAL_SPIRV_VULKAN_PROFILE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/api.h"
#include "loom/target/arch/spirv/profile.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_SPIRV_VULKAN_API_VERSION(major, minor, patch)     \
  ((((uint32_t)(major)) << 22) | (((uint32_t)(minor)) << 12) | \
   ((uint32_t)(patch)))

enum {
  // Minimum Vulkan environment for the executable ABI selected here.
  LOOM_SPIRV_VULKAN_API_VERSION_1_3 = LOOM_SPIRV_VULKAN_API_VERSION(1, 3, 0),
};

typedef enum loom_spirv_vulkan_hal_profile_flag_bits_e {
  // The HAL executable cache accepts raw Vulkan BDA SPIR-V modules.
  LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_RAW_BDA_EXECUTABLE = 1u << 0,
  // The logical device exposes buffer device addresses.
  LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_BUFFER_DEVICE_ADDRESS = 1u << 1,
  // The logical device exposes subgroup size control.
  LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SUBGROUP_SIZE_CONTROL = 1u << 2,
  // The logical device exposes VK_KHR_cooperative_matrix.
  LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_COOPERATIVE_MATRIX_KHR = 1u << 3,
  // The logical device exposes storageBuffer8BitAccess.
  LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_8BIT_ACCESS = 1u << 4,
  // The logical device exposes shaderFloat16.
  LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT16 = 1u << 5,
  // The logical device exposes shaderFloat64.
  LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT64 = 1u << 6,
  // The logical device exposes shaderInt8.
  LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INT8 = 1u << 7,
  // The logical device exposes shaderInt16.
  LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INT16 = 1u << 8,
  // The logical device exposes shaderInt64.
  LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INT64 = 1u << 9,
  // The logical device exposes shaderIntegerDotProduct.
  LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INTEGER_DOT_PRODUCT = 1u << 10,
  // The logical device exposes vulkanMemoryModel.
  LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_VULKAN_MEMORY_MODEL = 1u << 11,
  // The logical device exposes vulkanMemoryModelDeviceScope.
  LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_VULKAN_MEMORY_MODEL_DEVICE_SCOPE = 1u
                                                                        << 12,
  // The logical device exposes storageBuffer16BitAccess.
  LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_16BIT_ACCESS = 1u << 13,
  // The logical device exposes shaderBFloat16Type.
  LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_BFLOAT16_TYPE = 1u << 14,
  // The logical device exposes shaderBFloat16DotProduct.
  LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_BFLOAT16_DOT_PRODUCT = 1u << 15,
  // The logical device exposes shaderBFloat16CooperativeMatrix.
  LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_BFLOAT16_COOPERATIVE_MATRIX = 1u
                                                                          << 16,
} loom_spirv_vulkan_hal_profile_flag_bits_t;

typedef uint32_t loom_spirv_vulkan_hal_profile_flags_t;

// Device facts queried from a Vulkan HAL device and executable cache.
typedef struct loom_spirv_vulkan_hal_profile_facts_t {
  // Vulkan API version reported by the selected logical device.
  uint32_t api_version;
  // Vulkan/SPIR-V feature and executable-format facts.
  loom_spirv_vulkan_hal_profile_flags_t flags;
  // Fixed subgroup size in invocations, or zero when the device does not report
  // a fixed target-wide value.
  uint32_t subgroup_size;
  // Maximum flat local workgroup size in invocations.
  uint32_t max_compute_workgroup_invocations;
  // Maximum local workgroup size per dimension.
  loom_target_workgroup_size_t max_compute_workgroup_size;
  // Maximum dispatch workgroup count per dimension.
  loom_target_workgroup_count_limit_t max_compute_workgroup_count;
  // Number of active VK_KHR_cooperative_matrix property rows.
  uint32_t cooperative_matrix_property_count;
  // Bitmask of shader stages supported by VK_KHR_cooperative_matrix.
  uint32_t cooperative_matrix_supported_stages;
} loom_spirv_vulkan_hal_profile_facts_t;

typedef struct loom_spirv_vulkan_hal_target_profile_storage_t {
  // SPIR-V profile payload passed opaquely through core compiler target_data.
  // This must remain the first field so provider deinitialization can recover
  // the owning storage from loom_run_hal_device_target_t.data.
  loom_spirv_target_profile_t profile;
  // Cooperative property storage owned by this target profile.
  loom_spirv_cooperative_property_storage_t cooperative_properties;
} loom_spirv_vulkan_hal_target_profile_storage_t;

// Queries Vulkan/SPIR-V profile facts from |device| and |executable_cache|.
iree_status_t loom_spirv_vulkan_hal_profile_query(
    iree_hal_device_t* device, iree_hal_executable_cache_t* executable_cache,
    loom_spirv_vulkan_hal_profile_facts_t* out_facts);

// Queries active Vulkan cooperative matrix rows into caller-owned storage.
iree_status_t loom_spirv_vulkan_hal_query_cooperative_matrix_properties(
    iree_hal_device_t* device, iree_allocator_t allocator,
    iree_hal_vulkan_cooperative_matrix_property_t** out_properties,
    iree_host_size_t* out_property_count);

// Initializes target-local SPIR-V profile storage from exact Vulkan
// cooperative matrix property rows.
iree_status_t loom_spirv_vulkan_hal_target_profile_storage_initialize(
    const loom_spirv_vulkan_hal_profile_facts_t* facts,
    const iree_hal_vulkan_cooperative_matrix_property_t*
        cooperative_matrix_properties,
    iree_host_size_t cooperative_matrix_property_count,
    iree_allocator_t allocator,
    loom_spirv_vulkan_hal_target_profile_storage_t* out_storage);

// Releases storage allocated by
// loom_spirv_vulkan_hal_target_profile_storage_initialize.
void loom_spirv_vulkan_hal_target_profile_storage_deinitialize(
    loom_spirv_vulkan_hal_target_profile_storage_t* storage,
    iree_allocator_t allocator);

// Materializes a HAL-kernel Vulkan 1.3 raw-BDA SPIR-V target bundle.
//
// The returned storage borrows no fact memory and remains valid until
// overwritten by the caller. Device-reported feature bits are projected into
// the target config so source lowering and low verification see the same
// concrete target that the Vulkan HAL will execute.
iree_status_t loom_spirv_vulkan_hal_profile_initialize_target_bundle(
    const loom_spirv_vulkan_hal_profile_facts_t* facts,
    loom_target_bundle_storage_t* out_storage);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_SPIRV_VULKAN_PROFILE_H_
