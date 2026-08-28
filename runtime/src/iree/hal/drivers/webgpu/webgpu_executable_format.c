// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/webgpu_executable_format.h"

#include "iree/base/alignment.h"

enum {
  IREE_HAL_WEBGPU_EXECUTABLE_HEADER_MAGIC_OFFSET = 0,
  IREE_HAL_WEBGPU_EXECUTABLE_HEADER_VERSION_OFFSET = 4,
  IREE_HAL_WEBGPU_EXECUTABLE_HEADER_EXPORT_COUNT_OFFSET = 8,
};

enum {
  IREE_HAL_WEBGPU_EXECUTABLE_EXPORT_WGSL_SOURCE_OFFSET_OFFSET = 0,
  IREE_HAL_WEBGPU_EXECUTABLE_EXPORT_WGSL_SOURCE_LENGTH_OFFSET = 4,
  IREE_HAL_WEBGPU_EXECUTABLE_EXPORT_ENTRY_POINT_OFFSET_OFFSET = 8,
  IREE_HAL_WEBGPU_EXECUTABLE_EXPORT_ENTRY_POINT_LENGTH_OFFSET = 12,
  IREE_HAL_WEBGPU_EXECUTABLE_EXPORT_WORKGROUP_SIZE_X_OFFSET = 16,
  IREE_HAL_WEBGPU_EXECUTABLE_EXPORT_WORKGROUP_SIZE_Y_OFFSET = 20,
  IREE_HAL_WEBGPU_EXECUTABLE_EXPORT_WORKGROUP_SIZE_Z_OFFSET = 24,
  IREE_HAL_WEBGPU_EXECUTABLE_EXPORT_BINDING_COUNT_OFFSET = 28,
};

static iree_status_t iree_hal_webgpu_executable_format_read_payload(
    const iree_hal_webgpu_executable_format_t* format,
    iree_host_size_t export_ordinal, const char* field_name, uint32_t offset,
    uint32_t length, iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  if (IREE_UNLIKELY(length == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "WebGPU executable export[%" PRIhsz
                            "] has an empty %s",
                            export_ordinal, field_name);
  }
  const iree_host_size_t payload_offset = (iree_host_size_t)offset;
  const iree_host_size_t payload_length = (iree_host_size_t)length;
  if (IREE_UNLIKELY(payload_offset < format->payload_offset ||
                    payload_offset > format->data.data_length ||
                    payload_length >
                        format->data.data_length - payload_offset)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "WebGPU executable export[%" PRIhsz
        "] %s range is outside of the payload (offset=%u, length=%u)",
        export_ordinal, field_name, offset, length);
  }
  *out_value = iree_make_string_view(
      (const char*)format->data.data + payload_offset, payload_length);
  return iree_ok_status();
}

iree_status_t iree_hal_webgpu_executable_format_read_export(
    const iree_hal_webgpu_executable_format_t* format, iree_host_size_t ordinal,
    iree_hal_webgpu_executable_export_t* out_export) {
  IREE_ASSERT_ARGUMENT(format);
  IREE_ASSERT_ARGUMENT(out_export);
  memset(out_export, 0, sizeof(*out_export));
  if (IREE_UNLIKELY(ordinal >= format->export_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "WebGPU executable export ordinal %" PRIhsz
                            " is out of range for %u exports",
                            ordinal, format->export_count);
  }

  const iree_host_size_t record_offset =
      IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_HEADER_SIZE +
      ordinal * IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_EXPORT_SIZE;
  const uint8_t* record = format->data.data + record_offset;

  const uint32_t wgsl_source_offset = iree_unaligned_load_le_u32(
      record + IREE_HAL_WEBGPU_EXECUTABLE_EXPORT_WGSL_SOURCE_OFFSET_OFFSET);
  const uint32_t wgsl_source_length = iree_unaligned_load_le_u32(
      record + IREE_HAL_WEBGPU_EXECUTABLE_EXPORT_WGSL_SOURCE_LENGTH_OFFSET);
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_executable_format_read_payload(
      format, ordinal, "WGSL source", wgsl_source_offset, wgsl_source_length,
      &out_export->wgsl_source));

  const uint32_t entry_point_offset = iree_unaligned_load_le_u32(
      record + IREE_HAL_WEBGPU_EXECUTABLE_EXPORT_ENTRY_POINT_OFFSET_OFFSET);
  const uint32_t entry_point_length = iree_unaligned_load_le_u32(
      record + IREE_HAL_WEBGPU_EXECUTABLE_EXPORT_ENTRY_POINT_LENGTH_OFFSET);
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_executable_format_read_payload(
      format, ordinal, "entry-point name", entry_point_offset,
      entry_point_length, &out_export->entry_point));

  out_export->workgroup_size[0] = iree_unaligned_load_le_u32(
      record + IREE_HAL_WEBGPU_EXECUTABLE_EXPORT_WORKGROUP_SIZE_X_OFFSET);
  out_export->workgroup_size[1] = iree_unaligned_load_le_u32(
      record + IREE_HAL_WEBGPU_EXECUTABLE_EXPORT_WORKGROUP_SIZE_Y_OFFSET);
  out_export->workgroup_size[2] = iree_unaligned_load_le_u32(
      record + IREE_HAL_WEBGPU_EXECUTABLE_EXPORT_WORKGROUP_SIZE_Z_OFFSET);
  if (IREE_UNLIKELY(out_export->workgroup_size[0] == 0 ||
                    out_export->workgroup_size[1] == 0 ||
                    out_export->workgroup_size[2] == 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "WebGPU executable export[%" PRIhsz
        "] has an invalid zero workgroup dimension (%ux%ux%u)",
        ordinal, out_export->workgroup_size[0], out_export->workgroup_size[1],
        out_export->workgroup_size[2]);
  }

  const uint32_t binding_count = iree_unaligned_load_le_u32(
      record + IREE_HAL_WEBGPU_EXECUTABLE_EXPORT_BINDING_COUNT_OFFSET);
  if (IREE_UNLIKELY(binding_count > UINT16_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "WebGPU executable export[%" PRIhsz
        "] declares %u bindings, exceeding the HAL reflection limit of %u",
        ordinal, binding_count, (uint32_t)UINT16_MAX);
  }
  out_export->binding_count = (uint16_t)binding_count;
  return iree_ok_status();
}

iree_status_t iree_hal_webgpu_executable_format_parse(
    iree_const_byte_span_t data,
    iree_hal_webgpu_executable_format_t* out_format) {
  IREE_ASSERT_ARGUMENT(out_format);
  memset(out_format, 0, sizeof(*out_format));
  if (IREE_UNLIKELY(data.data_length > 0 && data.data == NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "WebGPU executable data is NULL");
  }
  if (IREE_UNLIKELY(data.data_length <
                    IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_HEADER_SIZE)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "WebGPU executable header is truncated (length=%" PRIhsz
        ", minimum=%u)",
        data.data_length,
        (uint32_t)IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_HEADER_SIZE);
  }

  const uint32_t magic = iree_unaligned_load_le_u32(
      data.data + IREE_HAL_WEBGPU_EXECUTABLE_HEADER_MAGIC_OFFSET);
  if (IREE_UNLIKELY(magic != IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_MAGIC)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "WebGPU executable has invalid magic");
  }
  const uint32_t version = iree_unaligned_load_le_u32(
      data.data + IREE_HAL_WEBGPU_EXECUTABLE_HEADER_VERSION_OFFSET);
  if (IREE_UNLIKELY(version != IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_VERSION)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "WebGPU executable format version %u is "
        "unsupported; expected version %u",
        version, (uint32_t)IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_VERSION);
  }
  const uint32_t export_count = iree_unaligned_load_le_u32(
      data.data + IREE_HAL_WEBGPU_EXECUTABLE_HEADER_EXPORT_COUNT_OFFSET);
  if (IREE_UNLIKELY(export_count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "WebGPU executable has no exports");
  }

  const iree_host_size_t maximum_export_count =
      (data.data_length - IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_HEADER_SIZE) /
      IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_EXPORT_SIZE;
  if (IREE_UNLIKELY((iree_host_size_t)export_count > maximum_export_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "WebGPU executable export table is truncated "
                            "(count=%u, available=%" PRIhsz ")",
                            export_count, maximum_export_count);
  }

  iree_hal_webgpu_executable_format_t format = {
      .data = data,
      .export_count = export_count,
      .payload_offset = IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_HEADER_SIZE +
                        (iree_host_size_t)export_count *
                            IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_EXPORT_SIZE,
  };
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < format.export_count && iree_status_is_ok(status); ++i) {
    iree_hal_webgpu_executable_export_t export;
    status = iree_hal_webgpu_executable_format_read_export(&format, i, &export);
  }
  if (iree_status_is_ok(status)) *out_format = format;
  return status;
}
