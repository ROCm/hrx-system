// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DEVICE_H_
#define AMDF_SRC_GPU_UMD_DEVICE_H_

#include "amdf/gpu.h"
#include "libamdf/src/gpu/umd/instance.h"
#include "libamdf/src/platform/endpoint.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct amdf_gpu_umd_device_t amdf_gpu_umd_device_t;

// Native result used to publish one successfully materialized GPU device.
typedef struct amdf_gpu_umd_device_result_t {
  // Opaque identity of the live native execution domain.
  amdf_device_id_t id;
  // Provider epoch invalidating native state after reset.
  uint64_t reset_epoch;
  // Features supported by this materialized native device and lifetime policy.
  amdf_gpu_device_features_t features;
} amdf_gpu_umd_device_result_t;

// Creates program-independent execution state borrowing the prepared instance
// connections. The caller holds the instance's native-state lock. Failure
// preserves both outputs; any failed rollback remains owned by the endpoint.
amdf_status_t amdf_gpu_umd_device_create(
    amdf_gpu_umd_instance_t* instance, amdf_platform_endpoint_t* endpoint,
    amdf_allocator_t host_allocator, amdf_native_lifetime_t native_lifetime,
    amdf_gpu_umd_device_t** out_device,
    amdf_gpu_umd_device_result_t* out_result);

// Releases native GPU device state in reverse ownership order.
amdf_status_t amdf_gpu_umd_device_destroy(amdf_gpu_umd_device_t* device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_DEVICE_H_
