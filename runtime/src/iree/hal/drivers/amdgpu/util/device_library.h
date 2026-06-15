// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_DEVICE_LIBRARY_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_DEVICE_LIBRARY_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/loaded_code_object.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_device_kernels_t
    iree_hal_amdgpu_device_kernels_t;

typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_library_t
//===----------------------------------------------------------------------===//

// Loaded device library executable.
typedef struct iree_hal_amdgpu_device_library_t {
  // Unowned libhsa handle. Must be retained by the owner.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // Loaded and frozen executable for all GPU devices.
  hsa_executable_t executable;
} iree_hal_amdgpu_device_library_t;

// Initializes |out_library| by loading the builtin device library for all of
// the GPU devices in the provided topology.
//
// |libhsa| is captured by-reference and must remain valid for the lifetime of
// the cache.
iree_status_t iree_hal_amdgpu_device_library_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_allocator_t host_allocator,
    iree_hal_amdgpu_device_library_t* out_library);

// Deinitializes |library| and releases underlying executable resources.
void iree_hal_amdgpu_device_library_deinitialize(
    iree_hal_amdgpu_device_library_t* library);

// Populates the host/device loaded code-object range for |device_agent|.
iree_status_t
iree_hal_amdgpu_device_library_populate_agent_loaded_code_object_range(
    const iree_hal_amdgpu_device_library_t* library, hsa_agent_t device_agent,
    iree_hal_amdgpu_loaded_code_object_range_t* out_range);

// Populates |out_kernels| with the information based on |device_agent|.
// Each device agent will have its own unique kernel object pointers. Most other
// information should be consistent across all other devices as they must be
// homogeneous within the same HAL device context.
iree_status_t iree_hal_amdgpu_device_library_populate_agent_kernels(
    const iree_hal_amdgpu_device_library_t* library, hsa_agent_t device_agent,
    iree_hal_amdgpu_device_kernels_t* out_kernels);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_DEVICE_LIBRARY_H_
