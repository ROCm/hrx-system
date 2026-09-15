// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/windows/device_status.h"

#include <ntstatus.h>

void amdf_kmt_device_status_initialize(
    amdf_kmt_device_status_t* device_status) {
  amdf_atomic_uint64_initialize(&device_status->terminal_status,
                                AMDF_STATUS_OK);
}

amdf_status_t amdf_kmt_device_status_query(
    const amdf_kmt_device_status_t* device_status) {
  return amdf_atomic_uint64_load_acquire(&device_status->terminal_status);
}

static bool amdf_kmt_status_is_device_lost(NTSTATUS status) {
  return status == STATUS_DEVICE_REMOVED || status == STATUS_DEVICE_HUNG ||
         status == STATUS_GRAPHICS_GPU_EXCEPTION_ON_DEVICE;
}

static amdf_status_t amdf_kmt_device_status_publish(
    amdf_kmt_device_status_t* device_status, amdf_status_t terminal_status) {
  uint64_t expected = AMDF_STATUS_OK;
  return amdf_atomic_uint64_compare_exchange_acq_rel(
             &device_status->terminal_status, &expected, terminal_status)
             ? terminal_status
             : expected;
}

amdf_status_t amdf_kmt_device_status_observe_error(
    amdf_kmt_device_status_t* device_status, const amdf_kmt_api_t* api,
    D3DKMT_HANDLE device, amdf_status_t operation_status) {
  const amdf_status_t terminal_status =
      amdf_kmt_device_status_query(device_status);
  if (!amdf_status_is_ok(terminal_status)) {
    return terminal_status;
  }
  if (amdf_status_is_ok(operation_status) ||
      amdf_status_domain(operation_status) != AMDF_STATUS_DOMAIN_NTSTATUS) {
    return operation_status;
  }
  if (amdf_kmt_status_is_device_lost(
          (NTSTATUS)amdf_status_code(operation_status))) {
    return amdf_kmt_device_status_publish(device_status, operation_status);
  }

  D3DKMT_GETDEVICESTATE query = {0};
  query.hDevice = device;
  query.StateType = D3DKMT_DEVICESTATE_EXECUTION;
  const NTSTATUS query_status = api->get_device_state(&query);
  if (amdf_kmt_status_is_device_lost(query_status)) {
    return amdf_kmt_device_status_publish(device_status,
                                          amdf_kmt_make_status(query_status));
  }
  if (query_status != STATUS_SUCCESS) {
    return amdf_kmt_make_status(query_status);
  }
  switch (query.ExecutionState) {
    case D3DKMT_DEVICEEXECUTION_ACTIVE:
      return operation_status;
    case D3DKMT_DEVICEEXECUTION_RESET:
    case D3DKMT_DEVICEEXECUTION_HUNG:
    case D3DKMT_DEVICEEXECUTION_STOPPED:
    case D3DKMT_DEVICEEXECUTION_ERROR_OUTOFMEMORY:
    case D3DKMT_DEVICEEXECUTION_ERROR_DMAFAULT:
    case D3DKMT_DEVICEEXECUTION_ERROR_DMAPAGEFAULT:
      // These execution states explicitly mean the device cannot continue.
      // ERROR_OUTOFMEMORY is terminal here, unlike STATUS_NO_MEMORY returned
      // by an individual operation rejected before acceptance.
      return amdf_kmt_device_status_publish(
          device_status, amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
    default:
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
}
