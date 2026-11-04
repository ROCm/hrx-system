// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_WDDM_MEMORY_H_
#define AMDF_SRC_GPU_UMD_WDDM_MEMORY_H_

#include "libamdf/src/gpu/umd/dxg/memory.h"

#ifdef __cplusplus
extern "C" {
#endif

// Captures the provider-neutral DXG memory attachment for one live device.
amdf_gpu_dxg_memory_device_t amdf_gpu_wddm_memory_device(
    amdf_gpu_umd_device_t* device);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_GPU_UMD_WDDM_MEMORY_H_
