// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/native.h"
#include "libamdf/src/gpu/umd/wddm/api.h"

const amdf_gpu_umd_api_t* amdf_gpu_umd_api_for_endpoint(
    const amdf_platform_endpoint_t* endpoint) {
  (void)endpoint;
  return &amdf_gpu_wddm_api;
}
