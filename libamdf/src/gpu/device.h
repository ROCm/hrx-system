// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_DEVICE_H_
#define AMDF_SRC_GPU_DEVICE_H_

#include "amdf/gpu.h"
#include "libamdf/src/gpu/umd/device.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Materializes one program-independent GPU execution and address domain.
amdf_status_t AMDF_CALL amdf_gpu_device_create(
    amdf_endpoint_t* endpoint, const amdf_gpu_device_create_info_t* create_info,
    amdf_device_t** out_device);

// Copies immutable identity and reset information for one GPU device.
amdf_status_t AMDF_CALL amdf_gpu_device_query_info(
    amdf_device_t* device, amdf_gpu_device_info_t* out_info);

// Returns the UMD device borrowed from one GPU device.
amdf_gpu_umd_device_t* amdf_gpu_device_get_umd(amdf_device_t* device);

// Returns immutable identity and reset state borrowed from one GPU device.
const amdf_gpu_device_info_t* amdf_gpu_device_get_info(
    const amdf_device_t* device);

// Queries the reset epoch cached by one GPU device.
uint64_t amdf_gpu_device_query_reset_epoch(const amdf_device_t* device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_DEVICE_H_
