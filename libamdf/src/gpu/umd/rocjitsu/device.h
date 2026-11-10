// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_ROCJITSU_DEVICE_H_
#define AMDF_SRC_GPU_UMD_ROCJITSU_DEVICE_H_

#include "libamdf/src/gpu/umd/api.h"
#include "libamdf/src/gpu/umd/rocjitsu/source.h"

struct amdf_gpu_umd_device_t {
  // Allocator owned by the common device's instance.
  amdf_allocator_t host_allocator;
  // Immutable configured GPU and its retained execution source.
  amdf_rocjitsu_gpu_t* gpu;
};

// Complete immutable implementation table, including unsupported services.
extern const amdf_gpu_umd_api_t amdf_rocjitsu_api;

amdf_status_t amdf_rocjitsu_device_create(
    amdf_gpu_umd_instance_t* instance, void* endpoint_context,
    amdf_allocator_t host_allocator, amdf_native_lifetime_t native_lifetime,
    amdf_gpu_umd_device_t** out_device,
    amdf_gpu_umd_device_result_t* out_result);
amdf_status_t amdf_rocjitsu_device_destroy(amdf_gpu_umd_device_t* device);
amdf_status_t amdf_rocjitsu_create_endpoint_profile(
    void* endpoint_context, amdf_allocator_t host_allocator,
    amdf_gpu_endpoint_profile_t** out_profile);
amdf_status_t amdf_rocjitsu_sample_memory_scope(
    const void* endpoint_context, amdf_memory_scope_flags_t flags,
    amdf_memory_scope_observation_t* out_observation);
bool amdf_rocjitsu_devices_share_mapping_domain(
    const amdf_gpu_umd_device_t* first, const amdf_gpu_umd_device_t* second);

#endif  // AMDF_SRC_GPU_UMD_ROCJITSU_DEVICE_H_
