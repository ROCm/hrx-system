// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_WDDM_API_H_
#define AMDF_SRC_GPU_UMD_WDDM_API_H_

#include "libamdf/src/gpu/umd/dxg/api.h"

// Complete native WDDM implementation selected on Windows.
extern const amdf_gpu_umd_api_t amdf_gpu_wddm_api;

#endif  // AMDF_SRC_GPU_UMD_WDDM_API_H_
