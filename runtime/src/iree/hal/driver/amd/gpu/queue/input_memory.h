// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_INPUT_MEMORY_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_INPUT_MEMORY_H_

#include "iree/hal/driver/amd/gpu/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Family-owned policy for CPU-produced, shader-read replay value tables. These
// are pointees of the fixup ABI, with independent storage and native
// retirement.
typedef struct iree_hal_amd_gpu_input_memory_t {
  // Persistent read-only GPU attachment requirements borrowed by options.
  amdf_memory_device_access_t access;
  // Qualified cached SYSTEM backing and write-only CPU mapping access.
  iree_hal_amd_fabric_memory_options_t options;
} iree_hal_amd_gpu_input_memory_t;

// Queries passive capabilities without allocating backing. Host writes need no
// cache maintenance; the consuming family's existing system acquire makes them
// visible to shaders. Successful output is self-referencing and stays at this
// address through every queue and native acquisition callback using the policy.
iree_status_t iree_hal_amd_gpu_input_memory_initialize(
    const iree_hal_amd_gpu_context_t* context, uint32_t family_ordinal,
    iree_hal_amd_gpu_input_memory_t* out_memory);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_INPUT_MEMORY_H_
