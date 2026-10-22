// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/firmware_trace.h"

#include "libamdf/src/device.h"
#include "libamdf/src/structure.h"
#include "libamdf/src/xdna/context.h"
#include "libamdf/src/xdna/device.h"
#include "libamdf/src/xdna/firmware_trace_format.h"
#include "libamdf/src/xdna/umd/firmware_trace.h"

amdf_status_t AMDF_CALL amdf_xdna_context_query_firmware_trace_info(
    amdf_xdna_context_t* context,
    amdf_xdna_context_firmware_trace_info_t* out_info) {
  if (context == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_status_t status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_FIRMWARE_TRACE_INFO,
      sizeof(amdf_xdna_context_firmware_trace_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  amdf_xdna_context_firmware_trace_info_t info = {0};
  status = amdf_xdna_umd_context_query_firmware_trace_info(
      amdf_xdna_context_get_umd(context), &info);
  if (amdf_status_is_ok(status)) {
    info.type = out_info->type;
    info.structure_size = out_info->structure_size;
    info.next = out_info->next;
    *out_info = info;
  }
  return status;
}

amdf_status_t AMDF_CALL amdf_xdna_device_query_firmware_trace_info(
    amdf_device_t* device, amdf_xdna_firmware_trace_info_t* out_info) {
  if (!amdf_device_is_extension(device, AMDF_EXTENSION_XDNA)) {
    return amdf_make_api_status(device == NULL
                                    ? AMDF_STATUS_CODE_INVALID_ARGUMENT
                                    : AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_status_t status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_XDNA_FIRMWARE_TRACE_INFO,
      sizeof(amdf_xdna_firmware_trace_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  amdf_xdna_firmware_trace_info_t info = {0};
  status = amdf_xdna_umd_query_firmware_trace_info(
      amdf_xdna_device_get_umd(device), &info);
  if (amdf_status_is_ok(status)) {
    info.type = out_info->type;
    info.structure_size = out_info->structure_size;
    info.next = out_info->next;
    if (info.enabled &&
        info.format_version == AMDF_XDNA_FIRMWARE_TRACE_FORMAT_2_0) {
      info.clock.frequency_hz = UINT64_C(1000000000);
      info.clock.counter_bit_count = 64;
    }
    *out_info = info;
  }
  return status;
}

amdf_status_t AMDF_CALL amdf_xdna_device_configure_firmware_trace(
    amdf_device_t* device, uint32_t enabled,
    amdf_xdna_firmware_trace_categories_t categories) {
  if (!amdf_device_is_extension(device, AMDF_EXTENSION_XDNA)) {
    return amdf_make_api_status(device == NULL
                                    ? AMDF_STATUS_CODE_INVALID_ARGUMENT
                                    : AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (enabled > 1 || (enabled && !categories)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return amdf_xdna_umd_configure_firmware_trace(
      amdf_xdna_device_get_umd(device), enabled, categories);
}

amdf_status_t AMDF_CALL amdf_xdna_device_read_firmware_trace(
    amdf_device_t* device, uint32_t format_version, uint64_t position,
    void* buffer, uint32_t buffer_byte_length,
    amdf_observation_read_result_t* out_result) {
  if (!amdf_device_is_extension(device, AMDF_EXTENSION_XDNA)) {
    return amdf_make_api_status(device == NULL
                                    ? AMDF_STATUS_CODE_INVALID_ARGUMENT
                                    : AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (buffer == NULL || out_result == NULL || (uintptr_t)buffer % 8 != 0 ||
      position % 32 != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (format_version != AMDF_XDNA_FIRMWARE_TRACE_FORMAT_2_0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_observation_read_result_t result = {0};
  const amdf_status_t status = amdf_xdna_umd_read_firmware_trace(
      amdf_xdna_device_get_umd(device), position, buffer, buffer_byte_length,
      &result);
  if (amdf_status_is_ok(status)) {
    // A native cursor must identify a complete frame boundary even when the
    // payload itself was overwritten. An unusable cursor cannot be published
    // as successful progress for the caller's next read.
    if (result.next_position % 32 != 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
    amdf_xdna_firmware_trace_decode(position, buffer, &result);
    *out_result = result;
  }
  return status;
}
