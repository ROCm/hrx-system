// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/linux/device_status.h"

enum {
  AMDF_GPU_LINUX_DXG_STATUS_DEVICE_REMOVED = 0xC00002B6u,
};

void amdf_gpu_linux_dxg_device_status_initialize(
    amdf_gpu_linux_dxg_device_status_t* device_status) {
  amdf_atomic_uint64_initialize(&device_status->terminal_status,
                                AMDF_STATUS_OK);
}

amdf_status_t amdf_gpu_linux_dxg_device_status_query(
    const amdf_gpu_linux_dxg_device_status_t* device_status) {
  return amdf_atomic_uint64_load_acquire(&device_status->terminal_status);
}

static amdf_status_t amdf_gpu_linux_dxg_device_status_publish(
    amdf_gpu_linux_dxg_device_status_t* device_status,
    amdf_status_t terminal_status) {
  uint64_t expected = AMDF_STATUS_OK;
  return amdf_atomic_uint64_compare_exchange_acq_rel(
             &device_status->terminal_status, &expected, terminal_status)
             ? terminal_status
             : expected;
}

void amdf_gpu_linux_dxg_device_status_record_reset(
    amdf_gpu_linux_dxg_device_status_t* device_status) {
  amdf_gpu_linux_dxg_device_status_publish(
      device_status, amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
}

static amdf_status_t amdf_gpu_linux_dxg_device_status_refresh(
    amdf_gpu_linux_dxg_device_status_t* device_status,
    const amdf_linux_dxg_kmt_t* kmt, D3DKMT_HANDLE device) {
  D3DKMT_GETDEVICESTATE query = {
      .hDevice = device,
      .StateType = D3DKMT_DEVICESTATE_EXECUTION,
  };
  const amdf_status_t query_status = amdf_linux_dxg_kmt_make_status(
      amdf_linux_dxg_kmt_get_device_state(kmt, &query));
  if (!amdf_status_is_ok(query_status)) {
    if (amdf_status_domain(query_status) == AMDF_STATUS_DOMAIN_NTSTATUS &&
        amdf_status_code(query_status) ==
            AMDF_GPU_LINUX_DXG_STATUS_DEVICE_REMOVED) {
      return amdf_gpu_linux_dxg_device_status_publish(device_status,
                                                      query_status);
    }
    return query_status;
  }
  switch (query.ExecutionState) {
    case D3DKMT_DEVICEEXECUTION_ACTIVE:
      return amdf_gpu_linux_dxg_device_status_query(device_status);
    case D3DKMT_DEVICEEXECUTION_RESET:
    case D3DKMT_DEVICEEXECUTION_HUNG:
    case D3DKMT_DEVICEEXECUTION_STOPPED:
    case D3DKMT_DEVICEEXECUTION_ERROR_OUTOFMEMORY:
    case D3DKMT_DEVICEEXECUTION_ERROR_DMAFAULT:
    case D3DKMT_DEVICEEXECUTION_ERROR_DMAPAGEFAULT:
      return amdf_gpu_linux_dxg_device_status_publish(
          device_status, amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
    default:
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
}

amdf_status_t amdf_gpu_linux_dxg_device_status_observe_error(
    amdf_gpu_linux_dxg_device_status_t* device_status,
    const amdf_linux_dxg_kmt_t* kmt, D3DKMT_HANDLE device,
    amdf_status_t operation_status) {
  const amdf_status_t terminal_status =
      amdf_gpu_linux_dxg_device_status_query(device_status);
  if (!amdf_status_is_ok(terminal_status)) {
    return terminal_status;
  }
  if (amdf_status_is_ok(operation_status) ||
      amdf_status_domain(operation_status) != AMDF_STATUS_DOMAIN_NTSTATUS) {
    return operation_status;
  }
  if (amdf_status_code(operation_status) ==
      AMDF_GPU_LINUX_DXG_STATUS_DEVICE_REMOVED) {
    return amdf_gpu_linux_dxg_device_status_publish(device_status,
                                                    operation_status);
  }
  const amdf_status_t status =
      amdf_gpu_linux_dxg_device_status_refresh(device_status, kmt, device);
  return amdf_status_is_ok(status) ? operation_status : status;
}

amdf_status_t amdf_gpu_linux_dxg_device_status_wait_for_paging(
    amdf_gpu_linux_dxg_device_status_t* device_status,
    const amdf_linux_dxg_kmt_t* kmt, D3DKMT_HANDLE device,
    D3DKMT_HANDLE paging_synchronization_object,
    const volatile uint64_t* current_paging_fence, uint64_t target_value) {
  if (target_value == 0) {
    return AMDF_STATUS_OK;
  }
  amdf_status_t status = amdf_gpu_linux_dxg_device_status_query(device_status);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  uint64_t progress = 0;
  if (current_paging_fence != NULL) {
    progress = amdf_gpu_linux_dxg_device_status_query_fence_progress(
        device_status, current_paging_fence);
    status = amdf_gpu_linux_dxg_device_status_query(device_status);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  if (progress < target_value) {
    D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU wait = {
        .hDevice = device,
        .ObjectCount = 1,
        .ObjectHandleArray = &paging_synchronization_object,
        .FenceValueArray = &target_value,
    };
    status = amdf_linux_dxg_kmt_make_status(
        amdf_linux_dxg_kmt_wait_from_cpu(kmt, &wait));
    if (!amdf_status_is_ok(status)) {
      return amdf_gpu_linux_dxg_device_status_observe_error(device_status, kmt,
                                                            device, status);
    }
    if (current_paging_fence != NULL) {
      amdf_gpu_linux_dxg_device_status_query_fence_progress(
          device_status, current_paging_fence);
      status = amdf_gpu_linux_dxg_device_status_query(device_status);
      if (!amdf_status_is_ok(status)) {
        return status;
      }
    }
  }
  return amdf_gpu_linux_dxg_device_status_refresh(device_status, kmt, device);
}
