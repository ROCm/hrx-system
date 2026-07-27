// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_LOADED_CODE_OBJECT_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_LOADED_CODE_OBJECT_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Address range for one HSA loaded code object on one device agent.
typedef struct iree_hal_amdgpu_loaded_code_object_range_t {
  // Host mapping for byte zero of the loaded code object.
  const uint8_t* host_pointer;
  // Device virtual address for byte zero of the loaded code object.
  uint64_t device_pointer;
  // Byte length of the loaded code object.
  uint64_t byte_length;
} iree_hal_amdgpu_loaded_code_object_range_t;

// Finds the loaded code object in |executable| for |device_agent|.
iree_status_t iree_hal_amdgpu_loaded_code_object_find(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_t executable,
    hsa_agent_t device_agent, hsa_loaded_code_object_t* out_loaded_code_object);

// Queries the host/device byte-zero range for |loaded_code_object|.
//
// The host mapping is owned by the HSA loader and remains valid while the HSA
// executable retaining |loaded_code_object| remains alive.
iree_status_t iree_hal_amdgpu_loaded_code_object_query_range(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    hsa_loaded_code_object_t loaded_code_object,
    iree_hal_amdgpu_loaded_code_object_range_t* out_range);

// Finds and queries the loaded code-object range for |device_agent|.
iree_status_t iree_hal_amdgpu_loaded_code_object_query_agent_range(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_executable_t executable,
    hsa_agent_t device_agent,
    iree_hal_amdgpu_loaded_code_object_range_t* out_range);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_LOADED_CODE_OBJECT_H_
