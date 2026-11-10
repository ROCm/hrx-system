// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_NATIVE_H_
#define AMDF_SRC_GPU_UMD_NATIVE_H_

#include "libamdf/src/gpu/umd/api.h"
#include "libamdf/src/platform/endpoint.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns the immutable GPU implementation selected by one platform endpoint.
// Provider composition binds this table and the platform context before profile
// qualification. Every descendant object retains or derives this exact table.
const amdf_gpu_umd_api_t* amdf_gpu_umd_api_for_endpoint(
    const amdf_platform_endpoint_t* endpoint);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_NATIVE_H_
