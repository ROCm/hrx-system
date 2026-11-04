// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_NATIVE_H_
#define AMDF_SRC_GPU_UMD_NATIVE_H_

#include "libamdf/src/gpu/umd/api.h"

// Returns the immutable GPU implementation selected by one platform endpoint.
// Every descendant object retains or derives this exact table.
const amdf_gpu_umd_api_t* amdf_gpu_umd_api_for_endpoint(
    const amdf_platform_endpoint_t* endpoint);

#endif  // AMDF_SRC_GPU_UMD_NATIVE_H_
