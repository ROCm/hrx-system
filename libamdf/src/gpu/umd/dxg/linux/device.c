// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/linux/device.h"

#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/dxg/linux/api.h"
#include "libamdf/src/gpu/umd/dxg/linux/memory_profile.h"
#include "libamdf/src/gpu/umd/dxg/wsl/api.h"

static amdf_status_t amdf_gpu_linux_dxg_device_release_native(
    amdf_gpu_umd_device_t* device) {
  amdf_status_t status = AMDF_STATUS_OK;
  if (device->paging_queue.native != 0) {
    status = amdf_gpu_linux_dxg_paging_queue_consume(&device->paging_queue);
  }
  if (device->logical_device.native != 0) {
    amdf_gpu_linux_dxg_logical_device_deinitialize(&device->logical_device);
  }
  return status;
}

amdf_status_t amdf_gpu_linux_dxg_device_create(
    amdf_gpu_umd_instance_t* instance, amdf_platform_endpoint_t* endpoint,
    amdf_allocator_t host_allocator, amdf_native_lifetime_t native_lifetime,
    amdf_gpu_umd_device_t** out_device,
    amdf_gpu_umd_device_result_t* out_result) {
  // The platform instance owns the DXG connection. Every execution and paging
  // object created here has an explicit inverse under either lifetime policy.
  (void)instance;
  (void)native_lifetime;

  amdf_gpu_dxg_memory_capabilities_t memory_capabilities = {0};
  amdf_status_t status = amdf_gpu_linux_dxg_query_memory_capabilities(
      endpoint, &memory_capabilities);

  amdf_gpu_umd_device_t* device = NULL;
  if (amdf_status_is_ok(status)) {
    status = amdf_calloc(host_allocator, sizeof(*device),
                         amdf_alignof(amdf_gpu_umd_device_t), (void**)&device);
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  device->host_allocator = host_allocator;
  amdf_gpu_linux_dxg_device_status_initialize(&device->status);
  device->adapter = endpoint->adapter;
  device->adapter_luid =
      endpoint->info.native_identity.value.windows_adapter.luid;
  device->physical_adapter_index = endpoint->physical_adapter_index;
  device->memory_capabilities = memory_capabilities;

  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_linux_dxg_logical_device_initialize(
        amdf_gpu_dxg_wsl_default_api(), endpoint, host_allocator,
        &device->logical_device, &device->properties);
  }
  if (amdf_status_is_ok(status)) {
    device->memory_capabilities.program_gfx_major =
        (uint32_t)device->properties.gfx_ip_major;
    device->memory_capabilities.program_host_is_fabric =
        !device->properties.is_discrete;
    device->memory_capabilities.program_host_rmw_supported =
        !device->properties.is_discrete ||
        device->properties.platform_atomic_support;
    status = amdf_gpu_linux_dxg_paging_queue_initialize(
        &device->logical_device, endpoint->physical_adapter_index,
        &device->paging_queue);
  }

  amdf_gpu_umd_device_result_t result = {0};
  if (amdf_status_is_ok(status)) {
    result.memory.flags = AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES;
    if (device->properties.is_discrete) {
      status = amdf_gpu_linux_dxg_query_local_capacity(
          endpoint, &result.memory.capacity_byte_length);
    } else {
      result.memory.flags |= AMDF_MEMORY_SCOPE_FLAG_SYSTEM_BACKED;
    }
  }
  if (amdf_status_is_ok(status)) {
    result.id = (amdf_device_id_t){
        .words =
            {
                endpoint->info.id.words[0],
                ((uint64_t)device->paging_queue.native << 32) |
                    device->logical_device.native,
            },
    };
    result.reset_epoch = 1;
    result.features = AMDF_GPU_DEVICE_FEATURE_HOST_REGISTRATION |
                      AMDF_GPU_DEVICE_FEATURE_DEVICE_RECREATION |
                      AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY;
    *out_result = result;
    *out_device = device;
  } else {
    const amdf_status_t release_status =
        amdf_gpu_linux_dxg_device_release_native(device);
    if (!amdf_status_is_ok(release_status)) {
      status = release_status;
    }
    amdf_free(host_allocator, device);
  }
  return status;
}

amdf_status_t amdf_gpu_linux_dxg_device_destroy(amdf_gpu_umd_device_t* device) {
  const amdf_status_t status = amdf_gpu_linux_dxg_device_release_native(device);
  if (amdf_status_is_ok(status)) {
    const amdf_allocator_t host_allocator = device->host_allocator;
    amdf_free(host_allocator, device);
  }
  return status;
}
