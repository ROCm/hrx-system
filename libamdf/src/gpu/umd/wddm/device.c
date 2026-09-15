// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/device.h"

#include <stddef.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/wddm/device.h"
#include "libamdf/src/gpu/umd/wddm/memory_profile.h"
#include "libamdf/src/platform/windows/endpoint.h"

static amdf_status_t amdf_gpu_wddm_device_release_native(
    amdf_gpu_umd_device_t* device) {
  if (device->wkmi_adapter.native != NULL) {
    const amdf_status_t status =
        amdf_gpu_wddm_wkmi_adapter_deinitialize(&device->wkmi_adapter);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  if (device->wkmi_loader.module != NULL) {
    const amdf_status_t status =
        amdf_gpu_wddm_wkmi_loader_deinitialize(&device->wkmi_loader);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  if (device->paging_queue != 0) {
    D3DDDI_DESTROYPAGINGQUEUE destroy_paging_queue = {0};
    destroy_paging_queue.hPagingQueue = device->paging_queue;
    const amdf_status_t status = amdf_kmt_make_status(
        device->kmt->destroy_paging_queue(&destroy_paging_queue));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    device->paging_queue = 0;
    device->paging_sync_object = 0;
    device->paging_fence = NULL;
  }
  if (device->device != 0) {
    D3DKMT_DESTROYDEVICE destroy_device = {0};
    destroy_device.hDevice = device->device;
    const amdf_status_t status =
        amdf_kmt_make_status(device->kmt->destroy_device(&destroy_device));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    device->device = 0;
  }
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_device_create(
    amdf_gpu_umd_instance_t* instance, amdf_platform_endpoint_t* endpoint,
    amdf_allocator_t host_allocator, amdf_native_lifetime_t native_lifetime,
    amdf_gpu_umd_device_t** out_device,
    amdf_gpu_umd_device_result_t* out_result) {
  // Shared KMT state belongs to the platform instance. Execution objects are
  // explicitly reclaimable under either lifetime policy.
  (void)instance;
  (void)native_lifetime;
  if (!amdf_kmt_api_supports_paging_devices(&endpoint->instance->kmt)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_windows_gpu_memory_capabilities_t memory_capabilities = {0};
  const amdf_status_t capabilities_status =
      amdf_windows_gpu_query_memory_capabilities(endpoint,
                                                 &memory_capabilities);
  if (!amdf_status_is_ok(capabilities_status)) return capabilities_status;

  amdf_gpu_umd_device_t* device = NULL;
  amdf_status_t status =
      amdf_calloc(host_allocator, sizeof(*device),
                  amdf_alignof(amdf_gpu_umd_device_t), (void**)&device);
  if (!amdf_status_is_ok(status)) return status;
  device->host_allocator = host_allocator;
  amdf_kmt_device_status_initialize(&device->status);
  device->kmt = &endpoint->instance->kmt;
  device->adapter = endpoint->adapter;
  device->physical_adapter_index = endpoint->physical_adapter_index;
  device->memory_capabilities = memory_capabilities;

  amdf_wkmi_bridge_gpu_properties_t properties = {0};
  status = amdf_gpu_wddm_wkmi_loader_initialize(host_allocator,
                                                &device->wkmi_loader);
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_wddm_wkmi_adapter_initialize(
        &device->wkmi_loader, endpoint->adapter,
        endpoint->physical_adapter_index, host_allocator, &device->wkmi_adapter,
        &properties);
  }
  (void)properties;

  D3DKMT_CREATEDEVICE create_device = {0};
  create_device.hAdapter = endpoint->adapter;
  if (amdf_status_is_ok(status)) {
    status = amdf_kmt_make_status(device->kmt->create_device(&create_device));
  }
  if (amdf_status_is_ok(status)) {
    device->device = create_device.hDevice;
    if (device->device == 0) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
  }

  D3DKMT_CREATEPAGINGQUEUE create_paging_queue = {0};
  if (amdf_status_is_ok(status)) {
    create_paging_queue.hDevice = device->device;
    create_paging_queue.Priority = D3DDDI_PAGINGQUEUE_PRIORITY_NORMAL;
    create_paging_queue.PhysicalAdapterIndex = endpoint->physical_adapter_index;
    status = amdf_kmt_make_status(
        device->kmt->create_paging_queue(&create_paging_queue));
  }
  if (amdf_status_is_ok(status)) {
    device->paging_queue = create_paging_queue.hPagingQueue;
    device->paging_sync_object = create_paging_queue.hSyncObject;
    device->paging_fence = (const volatile uint64_t*)
                               create_paging_queue.FenceValueCPUVirtualAddress;
    if (device->paging_queue == 0 || device->paging_sync_object == 0 ||
        device->paging_fence == NULL) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
  }

  if (amdf_status_is_ok(status)) {
    amdf_gpu_umd_device_result_t result = {0};
    result.id.words[0] = endpoint->id.words[0];
    result.id.words[1] =
        ((uint64_t)device->paging_queue << 32) | device->device;
    result.reset_epoch = 1;
    result.features = AMDF_GPU_DEVICE_FEATURE_DEVICE_RECREATION;
    if (amdf_kmt_api_supports_gpu_memory(device->kmt)) {
      result.features |= AMDF_GPU_DEVICE_FEATURE_HOST_REGISTRATION |
                         AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY;
    }
    *out_result = result;
    *out_device = device;
  } else {
    const amdf_status_t release_status =
        amdf_gpu_wddm_device_release_native(device);
    if (!amdf_status_is_ok(release_status)) {
      status = release_status;
    }
    // Construction has submitted no work borrowing this host bookkeeping.
    amdf_free(host_allocator, device);
  }
  return status;
}

amdf_status_t amdf_gpu_umd_device_destroy(amdf_gpu_umd_device_t* device) {
  const amdf_status_t status = amdf_gpu_wddm_device_release_native(device);
  if (amdf_status_is_ok(status)) {
    const amdf_allocator_t host_allocator = device->host_allocator;
    amdf_free(host_allocator, device);
  }
  return status;
}
