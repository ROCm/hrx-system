// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/api.h"

const amdf_gpu_umd_api_t amdf_gpu_wddm_api = {
#define AMDF_GPU_UMD_PLATFORM_OPERATION(result, name, parameters) \
  .name = amdf_gpu_umd_##name,
#define AMDF_GPU_UMD_DXG_SHARED_OPERATION(result, name, parameters) \
  .name = amdf_gpu_dxg_##name,
#include "libamdf/src/gpu/umd/api_operations.inc"
#undef AMDF_GPU_UMD_DXG_SHARED_OPERATION
#undef AMDF_GPU_UMD_PLATFORM_OPERATION
};
