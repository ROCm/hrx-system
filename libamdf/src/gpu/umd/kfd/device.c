// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/gpu/umd/kfd/device.h"

#include <unistd.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/kfd/hdp.h"
#include "libamdf/src/gpu/umd/kfd/instance.h"
#include "libamdf/src/gpu/umd/kfd/user_queue_native.h"
#include "libamdf/src/platform/linux/endpoint.h"
#include "libamdf/src/platform/linux/host_cache.h"

amdf_status_t amdf_gpu_umd_device_destroy(amdf_gpu_umd_device_t* device) {
  if (device->hdp != NULL) {
    const amdf_status_t status = amdf_gpu_kfd_hdp_destroy(device->hdp);
    device->hdp = NULL;
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  const amdf_allocator_t host_allocator = device->host_allocator;
  amdf_gpu_kfd_topology_deinitialize(&device->topology, host_allocator);
  amdf_free(host_allocator, device);
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_device_create(
    amdf_gpu_umd_instance_t* instance, amdf_platform_endpoint_t* endpoint,
    amdf_allocator_t host_allocator, amdf_native_lifetime_t native_lifetime,
    amdf_gpu_umd_device_t** out_device,
    amdf_gpu_umd_device_result_t* out_result) {
  amdf_gpu_umd_device_t* device = NULL;
  amdf_status_t status =
      amdf_calloc(host_allocator, sizeof(*device),
                  amdf_alignof(amdf_gpu_umd_device_t), (void**)&device);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  device->host_allocator = host_allocator;
  device->instance = instance;
  device->descriptor = -1;
  device->render_descriptor = -1;
  device->memory_lock = amdf_gpu_kfd_instance_memory_lock(instance);
  device->native_lifetime = native_lifetime;
  device->user_queue_native_api = amdf_gpu_kfd_user_queue_default_native_api();

  status = amdf_gpu_kfd_topology_initialize(endpoint, host_allocator,
                                            &device->topology);
  const long page_size = sysconf(_SC_PAGESIZE);
  if (amdf_status_is_ok(status)) {
    if (page_size <= 0 || (page_size & (page_size - 1)) != 0) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    } else {
      device->page_size = (size_t)page_size;
      status = amdf_linux_host_cache_query_line_size(&device->cache_line_size);
    }
  }
  if (amdf_status_is_ok(status)) {
    device->descriptor = amdf_gpu_kfd_instance_descriptor(instance);
    amdf_gpu_kfd_device_lock_memory(device,
                                    AMDF_GPU_KFD_MEMORY_PRIORITY_ACQUIRE);
    status = amdf_gpu_kfd_instance_prepare_vm(
        instance, endpoint, &device->topology, device->page_size,
        &device->render_descriptor);
    amdf_gpu_kfd_device_unlock_memory(device);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_kfd_hdp_create(device, &device->hdp);
  }
  if (amdf_status_is_ok(status)) {
    *out_result = (amdf_gpu_umd_device_result_t){
        .id = {.words = {endpoint->info.id.words[0], (uintptr_t)device}},
        .reset_epoch = 1,
        .features = device->topology.memory_features |
                    AMDF_GPU_DEVICE_FEATURE_NATIVE_EVENTS |
                    AMDF_GPU_DEVICE_FEATURE_DEVICE_RECREATION |
                    (native_lifetime == AMDF_NATIVE_LIFETIME_PROCESS
                         ? AMDF_GPU_DEVICE_FEATURE_HOST_REGISTRATION
                         : 0),
    };
    *out_device = device;
  } else {
    // Connections belong to the instance; this device owns only metadata.
    amdf_gpu_kfd_topology_deinitialize(&device->topology, host_allocator);
    amdf_free(host_allocator, device);
  }
  return status;
}
