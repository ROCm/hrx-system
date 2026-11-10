// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/rocjitsu/device.h"

#include "libamdf/src/allocator.h"
#include "libamdf/src/host_memory.h"

amdf_status_t amdf_rocjitsu_device_create(
    amdf_gpu_umd_instance_t* instance, void* endpoint_context,
    amdf_allocator_t host_allocator, amdf_native_lifetime_t native_lifetime,
    amdf_gpu_umd_device_t** out_device,
    amdf_gpu_umd_device_result_t* out_result) {
  (void)instance;
  (void)native_lifetime;
  amdf_gpu_umd_device_t* device = NULL;
  amdf_status_t status =
      amdf_calloc(host_allocator, sizeof(*device),
                  amdf_alignof(amdf_gpu_umd_device_t), (void**)&device);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  device->host_allocator = host_allocator;
  device->gpu = endpoint_context;
  status = amdf_rocjitsu_source_prepare(device->gpu->source);
  if (amdf_status_is_ok(status)) {
    amdf_gpu_umd_device_result_t result = {
        .id = {{device->gpu->source->generation, (uintptr_t)device}},
        .reset_epoch = 1,
        .features = AMDF_GPU_DEVICE_FEATURE_NATIVE_EVENTS |
                    AMDF_GPU_DEVICE_FEATURE_DEVICE_RECREATION |
                    AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY |
                    AMDF_GPU_DEVICE_FEATURE_HOST_VISIBLE_LOCAL_MEMORY,
        .memory = {.flags = AMDF_MEMORY_SCOPE_FLAG_SYSTEM_BACKED |
                            AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES},
    };
    const amdf_queue_family_placement_info_t placement = {
        .placement_capabilities = AMDF_QUEUE_PLACEMENT_CAPABILITY_HOST_LOCAL,
        .host_publication =
            {
                [AMDF_QUEUE_PLACEMENT_DEFAULT] =
                    AMDF_QUEUE_HOST_PUBLICATION_FLAG_KNOWN,
                [AMDF_QUEUE_PLACEMENT_HOST_LOCAL] =
                    AMDF_QUEUE_HOST_PUBLICATION_FLAG_KNOWN,
            },
    };
    for (uint32_t i = 0; i < device->gpu->profile.queue_family_count; ++i) {
      result.queue_placements[i] = placement;
    }
    *out_result = result;
    *out_device = device;
  } else {
    amdf_free(host_allocator, device);
  }
  return status;
}

amdf_status_t amdf_rocjitsu_device_destroy(amdf_gpu_umd_device_t* device) {
  amdf_free(device->host_allocator, device);
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_rocjitsu_create_endpoint_profile(
    void* endpoint_context, amdf_allocator_t host_allocator,
    amdf_gpu_endpoint_profile_t** out_profile) {
  const amdf_rocjitsu_gpu_t* gpu = endpoint_context;
  amdf_gpu_endpoint_profile_t* profile = NULL;
  const amdf_status_t status =
      amdf_malloc(host_allocator, sizeof(*profile),
                  amdf_alignof(amdf_gpu_endpoint_profile_t), (void**)&profile);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  *profile = gpu->profile;
  *out_profile = profile;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_rocjitsu_sample_memory_scope(
    const void* endpoint_context, amdf_memory_scope_flags_t flags,
    amdf_memory_scope_observation_t* out_observation) {
  (void)endpoint_context;
  (void)flags;
  return amdf_host_memory_sample_observation(out_observation);
}

bool amdf_rocjitsu_devices_share_mapping_domain(
    const amdf_gpu_umd_device_t* first, const amdf_gpu_umd_device_t* second) {
  return first->gpu->source == second->gpu->source;
}
