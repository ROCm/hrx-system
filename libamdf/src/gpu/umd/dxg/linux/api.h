// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_LINUX_API_H_
#define AMDF_SRC_GPU_UMD_DXG_LINUX_API_H_

#include "libamdf/src/gpu/umd/dxg/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Linux links both native GPU implementations. Declare the disjoint DXG/KMT
// symbols from the same signatures that define the operation table.
#define AMDF_GPU_UMD_PLATFORM_OPERATION(result, name, parameters) \
  result amdf_gpu_linux_dxg_##name parameters;
#define AMDF_GPU_UMD_DXG_SHARED_OPERATION(result, name, parameters)
#include "libamdf/src/gpu/umd/api_operations.inc"
#undef AMDF_GPU_UMD_DXG_SHARED_OPERATION
#undef AMDF_GPU_UMD_PLATFORM_OPERATION

// Complete DXG/KMT implementation selected by a WSL endpoint.
extern const amdf_gpu_umd_api_t amdf_gpu_linux_dxg_api;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_DXG_LINUX_API_H_
