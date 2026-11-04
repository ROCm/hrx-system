// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_LINUX_MEMORY_PROFILE_H_
#define AMDF_SRC_GPU_UMD_DXG_LINUX_MEMORY_PROFILE_H_

#include "libamdf/src/gpu/umd/dxg/memory_profile.h"
#include "libamdf/src/platform/linux/dxg/endpoint.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Captures immutable GPU MMU and CPU-visible update capabilities through the
// endpoint's direct Linux DXG transport. No logical device is created.
amdf_status_t amdf_gpu_linux_dxg_query_memory_capabilities(
    const amdf_platform_endpoint_t* endpoint,
    amdf_gpu_dxg_memory_capabilities_t* out_capabilities);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_DXG_LINUX_MEMORY_PROFILE_H_
