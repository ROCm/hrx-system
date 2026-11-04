// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_WDDM_MEMORY_PROFILE_H_
#define AMDF_SRC_GPU_UMD_WDDM_MEMORY_PROFILE_H_

#include "libamdf/src/gpu/umd/dxg/memory_profile.h"
#include "libamdf/src/platform/endpoint.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Queries GPU MMU and CPU-visible update widths through the Windows endpoint
// adapter. No logical device, paging queue, or allocation is created.
amdf_status_t amdf_gpu_wddm_query_memory_capabilities(
    const amdf_platform_endpoint_t* endpoint,
    amdf_gpu_dxg_memory_capabilities_t* out_capabilities);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_WDDM_MEMORY_PROFILE_H_
