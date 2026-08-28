// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/cuda_executable_format.h"

#include "iree/base/alignment.h"

enum {
  IREE_HAL_CUDA_EXECUTABLE_HEADER_MAGIC_OFFSET = 0,
  IREE_HAL_CUDA_EXECUTABLE_HEADER_VERSION_OFFSET = 4,
  IREE_HAL_CUDA_EXECUTABLE_HEADER_MODULE_COUNT_OFFSET = 8,
  IREE_HAL_CUDA_EXECUTABLE_HEADER_EXPORT_COUNT_OFFSET = 12,
};

enum {
  IREE_HAL_CUDA_EXECUTABLE_MODULE_PTX_IMAGE_OFFSET_OFFSET = 0,
  IREE_HAL_CUDA_EXECUTABLE_MODULE_PTX_IMAGE_LENGTH_OFFSET = 4,
};

enum {
  IREE_HAL_CUDA_EXECUTABLE_EXPORT_MODULE_ORDINAL_OFFSET = 0,
  IREE_HAL_CUDA_EXECUTABLE_EXPORT_KERNEL_NAME_OFFSET_OFFSET = 4,
  IREE_HAL_CUDA_EXECUTABLE_EXPORT_KERNEL_NAME_LENGTH_OFFSET = 8,
  IREE_HAL_CUDA_EXECUTABLE_EXPORT_BLOCK_SIZE_X_OFFSET = 12,
  IREE_HAL_CUDA_EXECUTABLE_EXPORT_BLOCK_SIZE_Y_OFFSET = 16,
  IREE_HAL_CUDA_EXECUTABLE_EXPORT_BLOCK_SIZE_Z_OFFSET = 20,
  IREE_HAL_CUDA_EXECUTABLE_EXPORT_BLOCK_SHARED_MEMORY_SIZE_OFFSET = 24,
  IREE_HAL_CUDA_EXECUTABLE_EXPORT_CONSTANT_COUNT_OFFSET = 28,
  IREE_HAL_CUDA_EXECUTABLE_EXPORT_BINDING_COUNT_OFFSET = 32,
};

static iree_status_t iree_hal_cuda_executable_format_read_string(
    const iree_hal_cuda_executable_format_t* format, const char* record_name,
    iree_host_size_t ordinal, const char* field_name, uint32_t offset,
    uint32_t length, iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  if (IREE_UNLIKELY(length == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CUDA executable %s[%" PRIhsz "] has an empty %s",
                            record_name, ordinal, field_name);
  }
  const iree_host_size_t payload_offset = (iree_host_size_t)offset;
  const iree_host_size_t payload_length = (iree_host_size_t)length;
  if (IREE_UNLIKELY(payload_offset < format->payload_offset ||
                    payload_offset >= format->data.data_length ||
                    payload_length >=
                        format->data.data_length - payload_offset)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "CUDA executable %s[%" PRIhsz
        "] %s range is outside of the payload (offset=%u, length=%u)",
        record_name, ordinal, field_name, offset, length);
  }
  const char* value = (const char*)format->data.data + payload_offset;
  if (IREE_UNLIKELY(memchr(value, 0, payload_length) != NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CUDA executable %s[%" PRIhsz
                            "] %s contains an embedded NUL byte",
                            record_name, ordinal, field_name);
  }
  if (IREE_UNLIKELY(value[payload_length] != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CUDA executable %s[%" PRIhsz
                            "] %s is not NUL-terminated",
                            record_name, ordinal, field_name);
  }
  *out_value = iree_make_string_view(value, payload_length);
  return iree_ok_status();
}

iree_status_t iree_hal_cuda_executable_format_read_module(
    const iree_hal_cuda_executable_format_t* format, iree_host_size_t ordinal,
    iree_hal_cuda_executable_module_t* out_module) {
  IREE_ASSERT_ARGUMENT(format);
  IREE_ASSERT_ARGUMENT(out_module);
  memset(out_module, 0, sizeof(*out_module));
  if (IREE_UNLIKELY(ordinal >= format->module_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "CUDA executable module ordinal %" PRIhsz
                            " is out of range for %u modules",
                            ordinal, format->module_count);
  }

  const iree_host_size_t record_offset =
      IREE_HAL_CUDA_EXECUTABLE_FORMAT_HEADER_SIZE +
      ordinal * IREE_HAL_CUDA_EXECUTABLE_FORMAT_MODULE_SIZE;
  const uint8_t* record = format->data.data + record_offset;
  const uint32_t ptx_image_offset = iree_unaligned_load_le_u32(
      record + IREE_HAL_CUDA_EXECUTABLE_MODULE_PTX_IMAGE_OFFSET_OFFSET);
  const uint32_t ptx_image_length = iree_unaligned_load_le_u32(
      record + IREE_HAL_CUDA_EXECUTABLE_MODULE_PTX_IMAGE_LENGTH_OFFSET);
  return iree_hal_cuda_executable_format_read_string(
      format, "module", ordinal, "PTX image", ptx_image_offset,
      ptx_image_length, &out_module->ptx_image);
}

iree_status_t iree_hal_cuda_executable_format_read_export(
    const iree_hal_cuda_executable_format_t* format, iree_host_size_t ordinal,
    iree_hal_cuda_executable_export_t* out_export) {
  IREE_ASSERT_ARGUMENT(format);
  IREE_ASSERT_ARGUMENT(out_export);
  memset(out_export, 0, sizeof(*out_export));
  if (IREE_UNLIKELY(ordinal >= format->export_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "CUDA executable export ordinal %" PRIhsz
                            " is out of range for %u exports",
                            ordinal, format->export_count);
  }

  const iree_host_size_t record_offset =
      format->export_table_offset +
      ordinal * IREE_HAL_CUDA_EXECUTABLE_FORMAT_EXPORT_SIZE;
  const uint8_t* record = format->data.data + record_offset;

  out_export->module_ordinal = iree_unaligned_load_le_u32(
      record + IREE_HAL_CUDA_EXECUTABLE_EXPORT_MODULE_ORDINAL_OFFSET);
  if (IREE_UNLIKELY(out_export->module_ordinal >= format->module_count)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "CUDA executable export[%" PRIhsz
        "] module ordinal %u is out of range for %u modules",
        ordinal, out_export->module_ordinal, format->module_count);
  }

  const uint32_t kernel_name_offset = iree_unaligned_load_le_u32(
      record + IREE_HAL_CUDA_EXECUTABLE_EXPORT_KERNEL_NAME_OFFSET_OFFSET);
  const uint32_t kernel_name_length = iree_unaligned_load_le_u32(
      record + IREE_HAL_CUDA_EXECUTABLE_EXPORT_KERNEL_NAME_LENGTH_OFFSET);
  IREE_RETURN_IF_ERROR(iree_hal_cuda_executable_format_read_string(
      format, "export", ordinal, "kernel name", kernel_name_offset,
      kernel_name_length, &out_export->kernel_name));

  out_export->block_size[0] = iree_unaligned_load_le_u32(
      record + IREE_HAL_CUDA_EXECUTABLE_EXPORT_BLOCK_SIZE_X_OFFSET);
  out_export->block_size[1] = iree_unaligned_load_le_u32(
      record + IREE_HAL_CUDA_EXECUTABLE_EXPORT_BLOCK_SIZE_Y_OFFSET);
  out_export->block_size[2] = iree_unaligned_load_le_u32(
      record + IREE_HAL_CUDA_EXECUTABLE_EXPORT_BLOCK_SIZE_Z_OFFSET);
  if (IREE_UNLIKELY(out_export->block_size[0] == 0 ||
                    out_export->block_size[1] == 0 ||
                    out_export->block_size[2] == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CUDA executable export[%" PRIhsz
                            "] has an invalid zero block dimension (%ux%ux%u)",
                            ordinal, out_export->block_size[0],
                            out_export->block_size[1],
                            out_export->block_size[2]);
  }

  out_export->block_shared_memory_size = iree_unaligned_load_le_u32(
      record + IREE_HAL_CUDA_EXECUTABLE_EXPORT_BLOCK_SHARED_MEMORY_SIZE_OFFSET);
  out_export->constant_count = iree_unaligned_load_le_u32(
      record + IREE_HAL_CUDA_EXECUTABLE_EXPORT_CONSTANT_COUNT_OFFSET);
  out_export->binding_count = iree_unaligned_load_le_u32(
      record + IREE_HAL_CUDA_EXECUTABLE_EXPORT_BINDING_COUNT_OFFSET);
  return iree_ok_status();
}

iree_status_t iree_hal_cuda_executable_format_parse(
    iree_const_byte_span_t data,
    iree_hal_cuda_executable_format_t* out_format) {
  IREE_ASSERT_ARGUMENT(out_format);
  memset(out_format, 0, sizeof(*out_format));
  if (IREE_UNLIKELY(data.data_length > 0 && data.data == NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CUDA executable data is NULL");
  }
  if (IREE_UNLIKELY(data.data_length <
                    IREE_HAL_CUDA_EXECUTABLE_FORMAT_HEADER_SIZE)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "CUDA executable header is truncated (length=%" PRIhsz ", minimum=%u)",
        data.data_length,
        (uint32_t)IREE_HAL_CUDA_EXECUTABLE_FORMAT_HEADER_SIZE);
  }

  const uint32_t magic = iree_unaligned_load_le_u32(
      data.data + IREE_HAL_CUDA_EXECUTABLE_HEADER_MAGIC_OFFSET);
  if (IREE_UNLIKELY(magic != IREE_HAL_CUDA_EXECUTABLE_FORMAT_MAGIC)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CUDA executable has invalid magic");
  }
  const uint32_t version = iree_unaligned_load_le_u32(
      data.data + IREE_HAL_CUDA_EXECUTABLE_HEADER_VERSION_OFFSET);
  if (IREE_UNLIKELY(version != IREE_HAL_CUDA_EXECUTABLE_FORMAT_VERSION)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "CUDA executable format version %u is unsupported; expected version "
        "%u",
        version, (uint32_t)IREE_HAL_CUDA_EXECUTABLE_FORMAT_VERSION);
  }

  const uint32_t module_count = iree_unaligned_load_le_u32(
      data.data + IREE_HAL_CUDA_EXECUTABLE_HEADER_MODULE_COUNT_OFFSET);
  if (IREE_UNLIKELY(module_count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CUDA executable has no modules");
  }
  const iree_host_size_t maximum_module_count =
      (data.data_length - IREE_HAL_CUDA_EXECUTABLE_FORMAT_HEADER_SIZE) /
      IREE_HAL_CUDA_EXECUTABLE_FORMAT_MODULE_SIZE;
  if (IREE_UNLIKELY((iree_host_size_t)module_count > maximum_module_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CUDA executable module table is truncated "
                            "(count=%u, available=%" PRIhsz ")",
                            module_count, maximum_module_count);
  }
  const iree_host_size_t export_table_offset =
      IREE_HAL_CUDA_EXECUTABLE_FORMAT_HEADER_SIZE +
      (iree_host_size_t)module_count *
          IREE_HAL_CUDA_EXECUTABLE_FORMAT_MODULE_SIZE;

  const uint32_t export_count = iree_unaligned_load_le_u32(
      data.data + IREE_HAL_CUDA_EXECUTABLE_HEADER_EXPORT_COUNT_OFFSET);
  if (IREE_UNLIKELY(export_count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CUDA executable has no exports");
  }
  const iree_host_size_t maximum_export_count =
      (data.data_length - export_table_offset) /
      IREE_HAL_CUDA_EXECUTABLE_FORMAT_EXPORT_SIZE;
  if (IREE_UNLIKELY((iree_host_size_t)export_count > maximum_export_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CUDA executable export table is truncated "
                            "(count=%u, available=%" PRIhsz ")",
                            export_count, maximum_export_count);
  }

  iree_hal_cuda_executable_format_t format = {
      .data = data,
      .module_count = module_count,
      .export_count = export_count,
      .export_table_offset = export_table_offset,
      .payload_offset =
          export_table_offset + (iree_host_size_t)export_count *
                                    IREE_HAL_CUDA_EXECUTABLE_FORMAT_EXPORT_SIZE,
  };
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < format.module_count && iree_status_is_ok(status); ++i) {
    iree_hal_cuda_executable_module_t module;
    status = iree_hal_cuda_executable_format_read_module(&format, i, &module);
  }
  for (iree_host_size_t i = 0;
       i < format.export_count && iree_status_is_ok(status); ++i) {
    iree_hal_cuda_executable_export_t export_def;
    status =
        iree_hal_cuda_executable_format_read_export(&format, i, &export_def);
  }
  if (iree_status_is_ok(status)) *out_format = format;
  return status;
}
