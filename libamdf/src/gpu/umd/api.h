// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_API_H_
#define AMDF_SRC_GPU_UMD_API_H_

#include "libamdf/src/gpu/umd/clock.h"
#include "libamdf/src/gpu/umd/device.h"
#include "libamdf/src/gpu/umd/endpoint_profile.h"
#include "libamdf/src/gpu/umd/event.h"
#include "libamdf/src/gpu/umd/kernel_queue.h"
#include "libamdf/src/gpu/umd/memory.h"
#include "libamdf/src/gpu/umd/memory_reservation.h"
#include "libamdf/src/gpu/umd/user_queue.h"

// Complete internal implementation selected for one coherent native GPU
// service. Every object created through an API is consumed through the same
// API.
typedef struct amdf_gpu_umd_api_t {
#define AMDF_GPU_UMD_PLATFORM_OPERATION(result, name, parameters) \
  result(*name) parameters;
#define AMDF_GPU_UMD_DXG_SHARED_OPERATION AMDF_GPU_UMD_PLATFORM_OPERATION
#include "libamdf/src/gpu/umd/api_operations.inc"
#undef AMDF_GPU_UMD_DXG_SHARED_OPERATION
#undef AMDF_GPU_UMD_PLATFORM_OPERATION
} amdf_gpu_umd_api_t;

#endif  // AMDF_SRC_GPU_UMD_API_H_
