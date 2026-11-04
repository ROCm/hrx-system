// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_KFD_API_H_
#define AMDF_SRC_GPU_UMD_KFD_API_H_

#include "libamdf/src/gpu/umd/api.h"

// Complete KFD/DRM implementation selected by a native Linux endpoint.
extern const amdf_gpu_umd_api_t amdf_gpu_kfd_api;

#endif  // AMDF_SRC_GPU_UMD_KFD_API_H_
