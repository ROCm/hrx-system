// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_ROCDXG_API_H_
#define AMDF_SRC_GPU_UMD_DXG_ROCDXG_API_H_

#include "libamdf/src/gpu/umd/dxg/bridge_api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns the statically linked ROCDXG provider API.
//
// The table and every procedure it references have process lifetime. Querying
// the table performs no allocation, device discovery, or native operation.
const amdf_dxg_bridge_api_t* amdf_gpu_dxg_rocdxg_default_api(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_DXG_ROCDXG_API_H_
