// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_LINUX_LOGICAL_DEVICE_H_
#define AMDF_SRC_GPU_UMD_DXG_LINUX_LOGICAL_DEVICE_H_

#include "libamdf/src/gpu/umd/dxg/adapter.h"
#include "libamdf/src/platform/linux/dxg/endpoint.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// One Linux DXG logical device and its private AMD adapter state.
typedef struct amdf_gpu_linux_dxg_logical_device_t {
  // KMT transport borrowed from the platform instance through teardown.
  const amdf_linux_dxg_kmt_t* kmt;
  // Native logical device handle owned by this object.
  D3DKMT_HANDLE native;
  // Private provider state borrowing |native| and the KMT transport.
  amdf_gpu_dxg_adapter_t bridge_adapter;
} amdf_gpu_linux_dxg_logical_device_t;

// Creates one logical device and parses its private AMD adapter state. Failure
// consumes all partial native state and preserves both outputs.
amdf_status_t amdf_gpu_linux_dxg_logical_device_initialize(
    const amdf_dxg_bridge_api_t* bridge_api, amdf_platform_endpoint_t* endpoint,
    amdf_allocator_t host_allocator,
    amdf_gpu_linux_dxg_logical_device_t* out_device,
    amdf_dxg_bridge_gpu_properties_t* out_properties);

// Consumes a valid logical device after every dependent native object is gone.
void amdf_gpu_linux_dxg_logical_device_deinitialize(
    amdf_gpu_linux_dxg_logical_device_t* device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_DXG_LINUX_LOGICAL_DEVICE_H_
