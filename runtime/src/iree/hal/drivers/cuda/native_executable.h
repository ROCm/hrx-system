// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_NATIVE_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_CUDA_NATIVE_EXECUTABLE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/cuda/cuda_dynamic_symbols.h"
#include "iree/hal/drivers/cuda/cuda_headers.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// The max number of per-dispatch bindings allowed in the CUDA HAL
// implementation.
#define IREE_HAL_CUDA_MAX_DISPATCH_BINDING_COUNT 16

// The max number of per-dispatch constants supported by the CUDA HAL
// implementation.
#define IREE_HAL_CUDA_MAX_DISPATCH_CONSTANT_COUNT 64

typedef struct iree_hal_cuda_kernel_params_t {
  // Executable-local function name used for lookup and reflection.
  iree_string_view_t name;

  // CUDA function handle.
  CUfunction function;

  // Total number of 32-bit constants passed at dispatch time.
  uint32_t constant_count;
  // Total number of buffers bound at dispatch time.
  uint32_t binding_count;

  // Required CUDA block dimensions.
  uint32_t block_dims[3];
  // Dynamic shared memory size in bytes.
  uint32_t block_shared_memory_size;

} iree_hal_cuda_kernel_params_t;

// Creates an IREE executable from a CUDA PTX module. The module may contain
// several kernels that can be extracted along with the associated block size.
iree_status_t iree_hal_cuda_native_executable_create(
    iree_hal_device_t* device, const iree_hal_cuda_dynamic_symbols_t* symbols,
    CUdevice cu_device, CUcontext cu_context,
    const iree_hal_executable_load_params_t* load_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

// Returns the kernel launch information for the given |entry_point| in the
// |executable|.
iree_status_t iree_hal_cuda_native_executable_lookup_kernel_params(
    iree_hal_executable_t* executable,
    iree_hal_executable_function_t export_ordinal,
    const iree_hal_cuda_kernel_params_t** out_params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_CUDA_NATIVE_EXECUTABLE_H_
