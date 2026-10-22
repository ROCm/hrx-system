// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/windows/monitored_fence.h"

#include <ntstatus.h>

#include "libamdf/src/allocator.h"

bool amdf_windows_monitored_fence_is_supported(const amdf_kmt_api_t* api) {
  return api->create_synchronization_object != NULL &&
         api->destroy_synchronization_object != NULL &&
         api->share_objects != NULL && api->open_synchronization_object != NULL;
}

amdf_status_t amdf_windows_monitored_fence_initialize(
    const amdf_kmt_api_t* api, D3DKMT_HANDLE device, uint64_t adapter_luid,
    amdf_windows_monitored_fence_t* out_fence) {
  D3DKMT_CREATESYNCHRONIZATIONOBJECT2 create = {0};
  create.hDevice = device;
  create.Info.Type = D3DDDI_MONITORED_FENCE;
  create.Info.Flags.Shared = 1;
  create.Info.Flags.NtSecuritySharing = 1;
  create.Info.MonitoredFence.EngineAffinity = 1;
  amdf_status_t status =
      amdf_kmt_make_status(api->create_synchronization_object(&create));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (create.hSyncObject == 0) {
    return amdf_kmt_make_status(STATUS_INVALID_HANDLE);
  }
  OBJECT_ATTRIBUTES attributes = {0};
  attributes.Length = sizeof(attributes);
  HANDLE shared_handle = NULL;
  if (create.Info.MonitoredFence.FenceValueCPUVirtualAddress == NULL) {
    status = amdf_kmt_make_status(STATUS_INVALID_HANDLE);
  } else {
    status = amdf_kmt_make_status(api->share_objects(
        1, &create.hSyncObject, &attributes, GENERIC_ALL, &shared_handle));
    if (amdf_status_is_ok(status) && shared_handle == NULL) {
      status = amdf_kmt_make_status(STATUS_INVALID_HANDLE);
    }
  }
  if (!amdf_status_is_ok(status)) {
    D3DKMT_DESTROYSYNCHRONIZATIONOBJECT destroy = {0};
    destroy.hSyncObject = create.hSyncObject;
    const amdf_status_t cleanup_status =
        amdf_kmt_make_status(api->destroy_synchronization_object(&destroy));
    return amdf_status_is_ok(cleanup_status) ? status : cleanup_status;
  }
  *out_fence = (amdf_windows_monitored_fence_t){
      .progress =
          {
              .base =
                  {
                      .protocol =
                          AMDF_KERNEL_QUEUE_DEPENDENCY_PROTOCOL_WINDOWS_MONITORED_FENCE,
                      .signal_operation_count = 1,
                  },
              .adapter_luid = adapter_luid,
              .shared_handle = shared_handle,
          },
      .object = create.hSyncObject,
      .pointer = create.Info.MonitoredFence.FenceValueCPUVirtualAddress,
  };
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_windows_monitored_fence_deinitialize(
    const amdf_kmt_api_t* api, amdf_windows_monitored_fence_t* fence) {
  amdf_status_t status = AMDF_STATUS_OK;
  if (!CloseHandle(fence->progress.shared_handle)) {
    status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  D3DKMT_DESTROYSYNCHRONIZATIONOBJECT destroy = {0};
  destroy.hSyncObject = fence->object;
  const amdf_status_t native_status =
      amdf_kmt_make_status(api->destroy_synchronization_object(&destroy));
  return amdf_status_is_ok(status) ? native_status : status;
}

static amdf_status_t amdf_windows_queue_dependency_destroy(
    amdf_kernel_queue_native_dependency_t* base) {
  amdf_windows_queue_dependency_t* dependency =
      (amdf_windows_queue_dependency_t*)base;
  D3DKMT_DESTROYSYNCHRONIZATIONOBJECT destroy = {0};
  destroy.hSyncObject = dependency->object;
  const amdf_status_t status = amdf_kmt_make_status(
      dependency->api->destroy_synchronization_object(&destroy));
  amdf_free(dependency->host_allocator, dependency);
  return status;
}

amdf_status_t amdf_windows_queue_dependency_create(
    amdf_allocator_t host_allocator, const amdf_kmt_api_t* api,
    D3DKMT_HANDLE device, uint64_t adapter_luid,
    const amdf_kernel_queue_native_progress_t* progress,
    amdf_kernel_queue_native_dependency_t** out_native) {
  if (progress->protocol !=
      AMDF_KERNEL_QUEUE_DEPENDENCY_PROTOCOL_WINDOWS_MONITORED_FENCE) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const amdf_windows_queue_progress_t* source =
      (const amdf_windows_queue_progress_t*)progress;
  if (source->adapter_luid != adapter_luid) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_windows_queue_dependency_t* dependency = NULL;
  amdf_status_t status = amdf_calloc(
      host_allocator, sizeof(*dependency),
      amdf_alignof(amdf_windows_queue_dependency_t), (void**)&dependency);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  dependency->base.destroy = amdf_windows_queue_dependency_destroy;
  dependency->base.wait_operation_count = 1;
  dependency->host_allocator = host_allocator;
  dependency->api = api;
  D3DKMT_OPENSYNCOBJECTFROMNTHANDLE2 open = {0};
  open.hNtHandle = source->shared_handle;
  open.hDevice = device;
  open.MonitoredFence.EngineAffinity = 1;
  status = amdf_kmt_make_status(api->open_synchronization_object(&open));
  if (amdf_status_is_ok(status) && open.hSyncObject == 0) {
    status = amdf_kmt_make_status(STATUS_INVALID_HANDLE);
  }
  if (!amdf_status_is_ok(status)) {
    amdf_free(host_allocator, dependency);
    return status;
  }
  dependency->object = open.hSyncObject;
  *out_native = &dependency->base;
  return AMDF_STATUS_OK;
}
