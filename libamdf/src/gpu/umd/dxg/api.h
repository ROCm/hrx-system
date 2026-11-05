// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_API_H_
#define AMDF_SRC_GPU_UMD_DXG_API_H_

#include "libamdf/src/gpu/umd/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// DXG shares provider-neutral unsupported operations across Linux DXG and
// WDDM. Declare those implementations from the canonical operation list.
#define AMDF_GPU_UMD_PLATFORM_OPERATION(result, name, parameters)
#define AMDF_GPU_UMD_DXG_SHARED_OPERATION(result, name, parameters) \
  result amdf_gpu_dxg_##name parameters;
#include "libamdf/src/gpu/umd/api_operations.inc"
#undef AMDF_GPU_UMD_DXG_SHARED_OPERATION
#undef AMDF_GPU_UMD_PLATFORM_OPERATION

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_DXG_API_H_
