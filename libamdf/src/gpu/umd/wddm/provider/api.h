// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_WDDM_PROVIDER_API_H_
#define AMDF_SRC_GPU_UMD_WDDM_PROVIDER_API_H_

#include "libamdf/src/gpu/umd/dxg/provider_api.h"
#include "libamdf/src/platform/windows/d3d12_api.h"
#include "libamdf/src/platform/windows/kmt_api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Platform-owned procedure tables borrowed by one native Windows adapter.
typedef struct amdf_gpu_wddm_provider_native_api_t {
  // KMT procedures used for adapter, allocation, queue, and resource work.
  const amdf_kmt_api_t* kmt;
  // Direct3D procedures used to validate typed shared resources.
  const amdf_d3d12_api_t* d3d12;
} amdf_gpu_wddm_provider_native_api_t;

// Returns the statically linked native Windows GPU provider API.
//
// The table and every procedure it references have process lifetime. Querying
// the table performs no allocation, device discovery, or native operation.
const amdf_dxg_provider_api_t* amdf_gpu_wddm_default_api(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_WDDM_PROVIDER_API_H_
