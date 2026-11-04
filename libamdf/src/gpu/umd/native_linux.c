// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/linux/api.h"
#include "libamdf/src/gpu/umd/kfd/api.h"
#include "libamdf/src/gpu/umd/native.h"
#include "libamdf/src/platform/linux/endpoint_state.h"

const amdf_gpu_umd_api_t* amdf_gpu_umd_api_for_endpoint(
    const amdf_platform_endpoint_t* endpoint) {
  return amdf_linux_endpoint_platform_kind(endpoint) ==
                 AMDF_LINUX_PLATFORM_KIND_DXG
             ? &amdf_gpu_linux_dxg_api
             : &amdf_gpu_kfd_api;
}
