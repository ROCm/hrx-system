// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/status.h"

static iree_status_code_t iree_hal_amd_status_code_from_amdf_api(
    uint32_t code) {
  switch ((amdf_status_code_t)code) {
    case AMDF_STATUS_CODE_OK:
      return IREE_STATUS_OK;
    case AMDF_STATUS_CODE_INVALID_ARGUMENT:
      return IREE_STATUS_INVALID_ARGUMENT;
    case AMDF_STATUS_CODE_OUT_OF_RANGE:
    case AMDF_STATUS_CODE_BUFFER_TOO_SMALL:
      return IREE_STATUS_OUT_OF_RANGE;
    case AMDF_STATUS_CODE_UNSUPPORTED:
      return IREE_STATUS_UNIMPLEMENTED;
    case AMDF_STATUS_CODE_NOT_FOUND:
      return IREE_STATUS_NOT_FOUND;
    case AMDF_STATUS_CODE_RESOURCE_EXHAUSTED:
      return IREE_STATUS_RESOURCE_EXHAUSTED;
    case AMDF_STATUS_CODE_BUSY:
      return IREE_STATUS_ABORTED;
    case AMDF_STATUS_CODE_DEADLINE_EXCEEDED:
      return IREE_STATUS_DEADLINE_EXCEEDED;
    case AMDF_STATUS_CODE_PERMISSION_DENIED:
      return IREE_STATUS_PERMISSION_DENIED;
    case AMDF_STATUS_CODE_DEVICE_LOST:
      return IREE_STATUS_DATA_LOSS;
    case AMDF_STATUS_CODE_VERSION_MISMATCH:
      return IREE_STATUS_INCOMPATIBLE;
    case AMDF_STATUS_CODE_FAILED_PRECONDITION:
      return IREE_STATUS_FAILED_PRECONDITION;
    case AMDF_STATUS_CODE_INTERNAL:
      return IREE_STATUS_INTERNAL;
    default:
      return IREE_STATUS_UNKNOWN;
  }
}

static iree_status_code_t iree_hal_amd_status_code_from_amdf(
    amdf_status_t status) {
  switch (amdf_status_domain(status)) {
    case AMDF_STATUS_DOMAIN_API:
      return iree_hal_amd_status_code_from_amdf_api(amdf_status_code(status));
    case AMDF_STATUS_DOMAIN_ERRNO:
      return iree_status_code_from_errno((int)amdf_status_code(status));
    default:
      return IREE_STATUS_UNKNOWN;
  }
}

static const char* iree_hal_amd_status_domain_string(
    amdf_status_domain_t domain) {
  switch (domain) {
    case AMDF_STATUS_DOMAIN_API:
      return "API";
    case AMDF_STATUS_DOMAIN_NTSTATUS:
      return "NTSTATUS";
    case AMDF_STATUS_DOMAIN_FIRMWARE:
      return "FIRMWARE";
    case AMDF_STATUS_DOMAIN_ERRNO:
      return "ERRNO";
    case AMDF_STATUS_DOMAIN_WIN32:
      return "WIN32";
    case AMDF_STATUS_DOMAIN_HRESULT:
      return "HRESULT";
    default:
      return "UNKNOWN";
  }
}

iree_status_t iree_hal_amd_status_from_amdf_status(const char* file,
                                                   uint32_t line,
                                                   amdf_status_t amdf_status,
                                                   const char* operation) {
  if (amdf_status_is_ok(amdf_status)) {
    return iree_ok_status();
  }
  const amdf_status_domain_t domain = amdf_status_domain(amdf_status);
  const uint32_t code = amdf_status_code(amdf_status);
  return iree_status_allocate_f(
      iree_hal_amd_status_code_from_amdf(amdf_status), file, line,
      "[%s] libamdf %s status 0x%08X", operation,
      iree_hal_amd_status_domain_string(domain), code);
}

void iree_hal_amd_report_failure(iree_hal_device_event_sink_t event_sink,
                                 iree_string_view_t driver_id,
                                 amdf_status_t status, const char* operation) {
  const iree_hal_device_driver_failure_event_t payload = {
      .record_length = sizeof(payload),
      .abi_version = IREE_HAL_DEVICE_DRIVER_FAILURE_EVENT_ABI_VERSION_0,
      .status_code = iree_hal_amd_status_code_from_amdf(status),
      .backend_result_code = status,
      .message = iree_make_cstring_view(operation),
  };
  iree_hal_device_event_t event = iree_hal_device_event_default();
  event.type = IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE;
  event.severity = IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR;
  event.source.driver_id = driver_id;
  event.payload = iree_make_const_byte_span(&payload, sizeof(payload));
  iree_hal_device_event_sink_publish(event_sink, &event);
}
