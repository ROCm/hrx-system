// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/device.h"

#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/windows/endpoint.h"
#include "libamdf/src/xdna/umd/mcdm/device.h"

amdf_xdna_umd_context_capabilities_t amdf_xdna_umd_query_context_capabilities(
    const amdf_xdna_endpoint_profile_t* profile) {
  amdf_xdna_umd_context_capabilities_t capabilities = {0};
  if ((profile->execution_capabilities &
       AMDF_XDNA_EXECUTION_CAPABILITY_TRANSACTION_INTERPRETER_V1) != 0) {
    capabilities.scheduling_modes = AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED;
  }
  // The native admission record does not establish binding physical placement.
  return capabilities;
}

static amdf_status_t amdf_windows_xdna_device_release_native(
    amdf_xdna_umd_device_t* device) {
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

amdf_status_t amdf_xdna_umd_device_create(
    amdf_platform_endpoint_t* endpoint,
    const amdf_xdna_endpoint_profile_t* profile,
    amdf_allocator_t host_allocator, amdf_xdna_umd_device_t** out_device,
    amdf_xdna_umd_device_result_t* out_result) {
  if (!amdf_kmt_api_supports_paging_devices(&endpoint->instance->kmt)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }

  amdf_xdna_umd_device_t* device = NULL;
  amdf_status_t status =
      amdf_calloc(host_allocator, sizeof(*device),
                  amdf_alignof(amdf_xdna_umd_device_t), (void**)&device);
  if (!amdf_status_is_ok(status)) return status;
  device->host_allocator = host_allocator;
  device->profile = profile;
  amdf_kmt_device_status_initialize(&device->status);
  device->kmt = &endpoint->instance->kmt;
  device->adapter = endpoint->adapter;

  D3DKMT_CREATEDEVICE create_device = {0};
  create_device.hAdapter = endpoint->adapter;
  status = amdf_kmt_make_status(device->kmt->create_device(&create_device));
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
    amdf_xdna_umd_device_result_t result = {0};
    result.id.words[0] = endpoint->id.words[0];
    result.id.words[1] = device->device;
    result.reset_epoch = 1;
    *out_result = result;
    *out_device = device;
  } else {
    const amdf_status_t release_status =
        amdf_windows_xdna_device_release_native(device);
    // No paging or execution work has been submitted by construction. Failed
    // native cleanup cannot borrow this unpublished host bookkeeping.
    amdf_free(host_allocator, device);
    if (!amdf_status_is_ok(release_status)) {
      status = release_status;
    }
  }
  return status;
}

amdf_status_t amdf_xdna_umd_device_destroy(amdf_xdna_umd_device_t* device) {
  const amdf_status_t status = amdf_windows_xdna_device_release_native(device);
  if (amdf_status_is_ok(status)) {
    const amdf_allocator_t host_allocator = device->host_allocator;
    amdf_free(host_allocator, device);
  }
  return status;
}
