// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/executable/format.h"

#include "iree/base/alignment.h"

enum {
  IREE_HAL_HIP_EXECUTABLE_HEADER_MAGIC_OFFSET = 0,
  IREE_HAL_HIP_EXECUTABLE_HEADER_VERSION_OFFSET = 4,
  IREE_HAL_HIP_EXECUTABLE_HEADER_MODULE_COUNT_OFFSET = 8,
  IREE_HAL_HIP_EXECUTABLE_HEADER_EXPORT_COUNT_OFFSET = 12,
};

enum {
  IREE_HAL_HIP_EXECUTABLE_MODULE_HSACO_IMAGE_OFFSET_OFFSET = 0,
  IREE_HAL_HIP_EXECUTABLE_MODULE_HSACO_IMAGE_LENGTH_OFFSET = 4,
};

enum {
  IREE_HAL_HIP_EXECUTABLE_EXPORT_MODULE_ORDINAL_OFFSET = 0,
  IREE_HAL_HIP_EXECUTABLE_EXPORT_KERNEL_NAME_OFFSET_OFFSET = 4,
  IREE_HAL_HIP_EXECUTABLE_EXPORT_KERNEL_NAME_LENGTH_OFFSET = 8,
  IREE_HAL_HIP_EXECUTABLE_EXPORT_BLOCK_SIZE_X_OFFSET = 12,
  IREE_HAL_HIP_EXECUTABLE_EXPORT_BLOCK_SIZE_Y_OFFSET = 16,
  IREE_HAL_HIP_EXECUTABLE_EXPORT_BLOCK_SIZE_Z_OFFSET = 20,
  IREE_HAL_HIP_EXECUTABLE_EXPORT_CONSTANT_COUNT_OFFSET = 24,
  IREE_HAL_HIP_EXECUTABLE_EXPORT_BINDING_COUNT_OFFSET = 28,
};

enum {
  IREE_HAL_HIP_ELF64_HEADER_SIZE = 64,
  IREE_HAL_HIP_ELF64_PROGRAM_HEADER_SIZE = 56,
  IREE_HAL_HIP_ELF64_SECTION_HEADER_SIZE = 64,
  IREE_HAL_HIP_ELF64_MACHINE_AMDGPU = 224,
  IREE_HAL_HIP_ELF64_SECTION_TYPE_NOBITS = 8,
};

enum {
  IREE_HAL_HIP_ELF_IDENT_CLASS_OFFSET = 4,
  IREE_HAL_HIP_ELF_IDENT_DATA_OFFSET = 5,
  IREE_HAL_HIP_ELF_IDENT_VERSION_OFFSET = 6,
  IREE_HAL_HIP_ELF_MACHINE_OFFSET = 18,
  IREE_HAL_HIP_ELF_VERSION_OFFSET = 20,
  IREE_HAL_HIP_ELF64_PROGRAM_HEADER_OFFSET_OFFSET = 32,
  IREE_HAL_HIP_ELF64_SECTION_HEADER_OFFSET_OFFSET = 40,
  IREE_HAL_HIP_ELF64_HEADER_SIZE_OFFSET = 52,
  IREE_HAL_HIP_ELF64_PROGRAM_HEADER_SIZE_OFFSET = 54,
  IREE_HAL_HIP_ELF64_PROGRAM_HEADER_COUNT_OFFSET = 56,
  IREE_HAL_HIP_ELF64_SECTION_HEADER_SIZE_OFFSET = 58,
  IREE_HAL_HIP_ELF64_SECTION_HEADER_COUNT_OFFSET = 60,
  IREE_HAL_HIP_ELF64_SECTION_NAME_INDEX_OFFSET = 62,
};

enum {
  IREE_HAL_HIP_ELF64_PROGRAM_FILE_OFFSET_OFFSET = 8,
  IREE_HAL_HIP_ELF64_PROGRAM_FILE_SIZE_OFFSET = 32,
  IREE_HAL_HIP_ELF64_SECTION_TYPE_OFFSET = 4,
  IREE_HAL_HIP_ELF64_SECTION_FILE_OFFSET_OFFSET = 24,
  IREE_HAL_HIP_ELF64_SECTION_FILE_SIZE_OFFSET = 32,
};

static bool iree_hal_hip_executable_format_u64_range_is_valid(
    uint64_t data_length, uint64_t offset, uint64_t length) {
  return offset <= data_length && length <= data_length - offset;
}

// Verifies every ELF structure that the lengthless HIP module-loading API may
// inspect before it derives an internal code object length.
static iree_status_t iree_hal_hip_executable_format_verify_hsaco(
    iree_host_size_t ordinal, iree_const_byte_span_t hsaco_image) {
  if (IREE_UNLIKELY(hsaco_image.data_length < IREE_HAL_HIP_ELF64_HEADER_SIZE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP executable module[%" PRIhsz
                            "] HSACO header is truncated (length=%" PRIhsz
                            ", minimum=%u)",
                            ordinal, hsaco_image.data_length,
                            (uint32_t)IREE_HAL_HIP_ELF64_HEADER_SIZE);
  }

  const uint8_t* data = hsaco_image.data;
  if (IREE_UNLIKELY(iree_unaligned_load_le_u32(data) != UINT32_C(0x464C457F))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP executable module[%" PRIhsz
                            "] HSACO has invalid ELF magic",
                            ordinal);
  }
  if (IREE_UNLIKELY(data[IREE_HAL_HIP_ELF_IDENT_CLASS_OFFSET] != 2 ||
                    data[IREE_HAL_HIP_ELF_IDENT_DATA_OFFSET] != 1 ||
                    data[IREE_HAL_HIP_ELF_IDENT_VERSION_OFFSET] != 1)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HIP executable module[%" PRIhsz
        "] HSACO must be a current little-endian ELF64 object",
        ordinal);
  }
  if (IREE_UNLIKELY(
          iree_unaligned_load_le_u16(data + IREE_HAL_HIP_ELF_MACHINE_OFFSET) !=
          IREE_HAL_HIP_ELF64_MACHINE_AMDGPU)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP executable module[%" PRIhsz
                            "] HSACO is not an AMDGPU object",
                            ordinal);
  }
  if (IREE_UNLIKELY(iree_unaligned_load_le_u32(
                        data + IREE_HAL_HIP_ELF_VERSION_OFFSET) != 1 ||
                    iree_unaligned_load_le_u16(
                        data + IREE_HAL_HIP_ELF64_HEADER_SIZE_OFFSET) !=
                        IREE_HAL_HIP_ELF64_HEADER_SIZE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP executable module[%" PRIhsz
                            "] HSACO has an invalid ELF64 header",
                            ordinal);
  }

  const uint64_t data_length = (uint64_t)hsaco_image.data_length;
  const uint64_t program_header_offset = iree_unaligned_load_le_u64(
      data + IREE_HAL_HIP_ELF64_PROGRAM_HEADER_OFFSET_OFFSET);
  const uint16_t program_header_size = iree_unaligned_load_le_u16(
      data + IREE_HAL_HIP_ELF64_PROGRAM_HEADER_SIZE_OFFSET);
  const uint16_t program_header_count = iree_unaligned_load_le_u16(
      data + IREE_HAL_HIP_ELF64_PROGRAM_HEADER_COUNT_OFFSET);
  if (IREE_UNLIKELY(program_header_count == UINT16_MAX)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "HIP executable module[%" PRIhsz
                            "] HSACO uses extended program header numbering",
                            ordinal);
  }
  if (program_header_count > 0) {
    if (IREE_UNLIKELY(
            program_header_size != IREE_HAL_HIP_ELF64_PROGRAM_HEADER_SIZE ||
            program_header_offset == 0 ||
            !iree_hal_hip_executable_format_u64_range_is_valid(
                data_length, program_header_offset,
                (uint64_t)program_header_count * program_header_size))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HIP executable module[%" PRIhsz
                              "] HSACO program header table is invalid",
                              ordinal);
    }
    for (uint16_t i = 0; i < program_header_count; ++i) {
      const uint8_t* program_header = data +
                                      (iree_host_size_t)program_header_offset +
                                      (iree_host_size_t)i * program_header_size;
      const uint64_t file_offset = iree_unaligned_load_le_u64(
          program_header + IREE_HAL_HIP_ELF64_PROGRAM_FILE_OFFSET_OFFSET);
      const uint64_t file_size = iree_unaligned_load_le_u64(
          program_header + IREE_HAL_HIP_ELF64_PROGRAM_FILE_SIZE_OFFSET);
      if (IREE_UNLIKELY(!iree_hal_hip_executable_format_u64_range_is_valid(
              data_length, file_offset, file_size))) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "HIP executable module[%" PRIhsz
            "] HSACO program header[%u] file range is out of bounds",
            ordinal, (uint32_t)i);
      }
    }
  }

  const uint64_t section_header_offset = iree_unaligned_load_le_u64(
      data + IREE_HAL_HIP_ELF64_SECTION_HEADER_OFFSET_OFFSET);
  const uint16_t section_header_size = iree_unaligned_load_le_u16(
      data + IREE_HAL_HIP_ELF64_SECTION_HEADER_SIZE_OFFSET);
  const uint16_t section_header_count = iree_unaligned_load_le_u16(
      data + IREE_HAL_HIP_ELF64_SECTION_HEADER_COUNT_OFFSET);
  const uint16_t section_name_index = iree_unaligned_load_le_u16(
      data + IREE_HAL_HIP_ELF64_SECTION_NAME_INDEX_OFFSET);
  if (IREE_UNLIKELY(
          section_header_count == 0 || section_header_offset == 0 ||
          section_header_size != IREE_HAL_HIP_ELF64_SECTION_HEADER_SIZE ||
          !iree_hal_hip_executable_format_u64_range_is_valid(
              data_length, section_header_offset,
              (uint64_t)section_header_count * section_header_size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP executable module[%" PRIhsz
                            "] HSACO section header table is invalid",
                            ordinal);
  }
  if (IREE_UNLIKELY(section_name_index != 0 &&
                    section_name_index >= section_header_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP executable module[%" PRIhsz
                            "] HSACO section name index is out of bounds",
                            ordinal);
  }
  for (uint16_t i = 0; i < section_header_count; ++i) {
    const uint8_t* section_header = data +
                                    (iree_host_size_t)section_header_offset +
                                    (iree_host_size_t)i * section_header_size;
    const uint32_t section_type = iree_unaligned_load_le_u32(
        section_header + IREE_HAL_HIP_ELF64_SECTION_TYPE_OFFSET);
    const uint64_t file_offset = iree_unaligned_load_le_u64(
        section_header + IREE_HAL_HIP_ELF64_SECTION_FILE_OFFSET_OFFSET);
    const uint64_t file_size = iree_unaligned_load_le_u64(
        section_header + IREE_HAL_HIP_ELF64_SECTION_FILE_SIZE_OFFSET);
    const uint64_t bounded_file_size =
        section_type == IREE_HAL_HIP_ELF64_SECTION_TYPE_NOBITS ? 0 : file_size;
    if (IREE_UNLIKELY(!iree_hal_hip_executable_format_u64_range_is_valid(
            data_length, file_offset, bounded_file_size))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HIP executable module[%" PRIhsz
          "] HSACO section header[%u] file range is out of bounds",
          ordinal, (uint32_t)i);
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_hip_executable_format_read_payload(
    const iree_hal_hip_executable_format_t* format, const char* record_name,
    iree_host_size_t ordinal, const char* field_name, uint32_t offset,
    uint32_t length, iree_const_byte_span_t* out_value) {
  *out_value = iree_const_byte_span_empty();
  if (IREE_UNLIKELY(length == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP executable %s[%" PRIhsz "] has an empty %s",
                            record_name, ordinal, field_name);
  }
  const iree_host_size_t payload_offset = (iree_host_size_t)offset;
  const iree_host_size_t payload_length = (iree_host_size_t)length;
  if (IREE_UNLIKELY(payload_offset < format->payload_offset ||
                    payload_offset > format->data.data_length ||
                    payload_length >
                        format->data.data_length - payload_offset)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HIP executable %s[%" PRIhsz
        "] %s range is outside of the payload (offset=%u, length=%u)",
        record_name, ordinal, field_name, offset, length);
  }
  *out_value = iree_make_const_byte_span(format->data.data + payload_offset,
                                         payload_length);
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_executable_format_read_string(
    const iree_hal_hip_executable_format_t* format, const char* record_name,
    iree_host_size_t ordinal, const char* field_name, uint32_t offset,
    uint32_t length, iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  if (IREE_UNLIKELY(length == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP executable %s[%" PRIhsz "] has an empty %s",
                            record_name, ordinal, field_name);
  }
  iree_const_byte_span_t value = iree_const_byte_span_empty();
  if (IREE_UNLIKELY(length == UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP executable %s[%" PRIhsz
                            "] %s is not NUL-terminated",
                            record_name, ordinal, field_name);
  }
  IREE_RETURN_IF_ERROR(iree_hal_hip_executable_format_read_payload(
      format, record_name, ordinal, field_name, offset, length + 1, &value));
  if (IREE_UNLIKELY(memchr(value.data, 0, length) != NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP executable %s[%" PRIhsz
                            "] %s contains an embedded NUL byte",
                            record_name, ordinal, field_name);
  }
  if (IREE_UNLIKELY(value.data[length] != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP executable %s[%" PRIhsz
                            "] %s is not NUL-terminated",
                            record_name, ordinal, field_name);
  }
  *out_value = iree_make_string_view((const char*)value.data, length);
  return iree_ok_status();
}

iree_status_t iree_hal_hip_executable_format_read_module(
    const iree_hal_hip_executable_format_t* format, iree_host_size_t ordinal,
    iree_hal_hip_executable_module_t* out_module) {
  IREE_ASSERT_ARGUMENT(format);
  IREE_ASSERT_ARGUMENT(out_module);
  memset(out_module, 0, sizeof(*out_module));
  if (IREE_UNLIKELY(ordinal >= format->module_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HIP executable module ordinal %" PRIhsz
                            " is out of range for %u modules",
                            ordinal, format->module_count);
  }

  const iree_host_size_t record_offset =
      IREE_HAL_HIP_EXECUTABLE_FORMAT_HEADER_SIZE +
      ordinal * IREE_HAL_HIP_EXECUTABLE_FORMAT_MODULE_SIZE;
  const uint8_t* record = format->data.data + record_offset;
  const uint32_t hsaco_image_offset = iree_unaligned_load_le_u32(
      record + IREE_HAL_HIP_EXECUTABLE_MODULE_HSACO_IMAGE_OFFSET_OFFSET);
  const uint32_t hsaco_image_length = iree_unaligned_load_le_u32(
      record + IREE_HAL_HIP_EXECUTABLE_MODULE_HSACO_IMAGE_LENGTH_OFFSET);
  IREE_RETURN_IF_ERROR(iree_hal_hip_executable_format_read_payload(
      format, "module", ordinal, "HSACO image", hsaco_image_offset,
      hsaco_image_length, &out_module->hsaco_image));
  return iree_hal_hip_executable_format_verify_hsaco(ordinal,
                                                     out_module->hsaco_image);
}

iree_status_t iree_hal_hip_executable_format_read_export(
    const iree_hal_hip_executable_format_t* format, iree_host_size_t ordinal,
    iree_hal_hip_executable_export_t* out_export) {
  IREE_ASSERT_ARGUMENT(format);
  IREE_ASSERT_ARGUMENT(out_export);
  memset(out_export, 0, sizeof(*out_export));
  if (IREE_UNLIKELY(ordinal >= format->export_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HIP executable export ordinal %" PRIhsz
                            " is out of range for %u exports",
                            ordinal, format->export_count);
  }

  const iree_host_size_t record_offset =
      format->export_table_offset +
      ordinal * IREE_HAL_HIP_EXECUTABLE_FORMAT_EXPORT_SIZE;
  const uint8_t* record = format->data.data + record_offset;

  out_export->module_ordinal = iree_unaligned_load_le_u32(
      record + IREE_HAL_HIP_EXECUTABLE_EXPORT_MODULE_ORDINAL_OFFSET);
  if (IREE_UNLIKELY(out_export->module_ordinal >= format->module_count)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HIP executable export[%" PRIhsz
        "] module ordinal %u is out of range for %u modules",
        ordinal, out_export->module_ordinal, format->module_count);
  }

  const uint32_t kernel_name_offset = iree_unaligned_load_le_u32(
      record + IREE_HAL_HIP_EXECUTABLE_EXPORT_KERNEL_NAME_OFFSET_OFFSET);
  const uint32_t kernel_name_length = iree_unaligned_load_le_u32(
      record + IREE_HAL_HIP_EXECUTABLE_EXPORT_KERNEL_NAME_LENGTH_OFFSET);
  IREE_RETURN_IF_ERROR(iree_hal_hip_executable_format_read_string(
      format, "export", ordinal, "kernel name", kernel_name_offset,
      kernel_name_length, &out_export->kernel_name));

  out_export->block_size[0] = iree_unaligned_load_le_u32(
      record + IREE_HAL_HIP_EXECUTABLE_EXPORT_BLOCK_SIZE_X_OFFSET);
  out_export->block_size[1] = iree_unaligned_load_le_u32(
      record + IREE_HAL_HIP_EXECUTABLE_EXPORT_BLOCK_SIZE_Y_OFFSET);
  out_export->block_size[2] = iree_unaligned_load_le_u32(
      record + IREE_HAL_HIP_EXECUTABLE_EXPORT_BLOCK_SIZE_Z_OFFSET);
  if (IREE_UNLIKELY(out_export->block_size[0] == 0 ||
                    out_export->block_size[1] == 0 ||
                    out_export->block_size[2] == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP executable export[%" PRIhsz
                            "] has an invalid zero block dimension (%ux%ux%u)",
                            ordinal, out_export->block_size[0],
                            out_export->block_size[1],
                            out_export->block_size[2]);
  }

  out_export->constant_count = iree_unaligned_load_le_u32(
      record + IREE_HAL_HIP_EXECUTABLE_EXPORT_CONSTANT_COUNT_OFFSET);
  out_export->binding_count = iree_unaligned_load_le_u32(
      record + IREE_HAL_HIP_EXECUTABLE_EXPORT_BINDING_COUNT_OFFSET);
  return iree_ok_status();
}

iree_status_t iree_hal_hip_executable_format_parse(
    iree_const_byte_span_t data, iree_hal_hip_executable_format_t* out_format) {
  IREE_ASSERT_ARGUMENT(out_format);
  memset(out_format, 0, sizeof(*out_format));
  if (IREE_UNLIKELY(data.data_length > 0 && data.data == NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP executable data is NULL");
  }
  if (IREE_UNLIKELY(data.data_length <
                    IREE_HAL_HIP_EXECUTABLE_FORMAT_HEADER_SIZE)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HIP executable header is truncated (length=%" PRIhsz ", minimum=%u)",
        data.data_length, (uint32_t)IREE_HAL_HIP_EXECUTABLE_FORMAT_HEADER_SIZE);
  }

  const uint32_t magic = iree_unaligned_load_le_u32(
      data.data + IREE_HAL_HIP_EXECUTABLE_HEADER_MAGIC_OFFSET);
  if (IREE_UNLIKELY(magic != IREE_HAL_HIP_EXECUTABLE_FORMAT_MAGIC)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP executable has invalid magic");
  }
  const uint32_t version = iree_unaligned_load_le_u32(
      data.data + IREE_HAL_HIP_EXECUTABLE_HEADER_VERSION_OFFSET);
  if (IREE_UNLIKELY(version != IREE_HAL_HIP_EXECUTABLE_FORMAT_VERSION)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "HIP executable format version %u is unsupported; expected version %u",
        version, (uint32_t)IREE_HAL_HIP_EXECUTABLE_FORMAT_VERSION);
  }

  const uint32_t module_count = iree_unaligned_load_le_u32(
      data.data + IREE_HAL_HIP_EXECUTABLE_HEADER_MODULE_COUNT_OFFSET);
  if (IREE_UNLIKELY(module_count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP executable has no modules");
  }
  const iree_host_size_t maximum_module_count =
      (data.data_length - IREE_HAL_HIP_EXECUTABLE_FORMAT_HEADER_SIZE) /
      IREE_HAL_HIP_EXECUTABLE_FORMAT_MODULE_SIZE;
  if (IREE_UNLIKELY((iree_host_size_t)module_count > maximum_module_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP executable module table is truncated "
                            "(count=%u, available=%" PRIhsz ")",
                            module_count, maximum_module_count);
  }
  const iree_host_size_t export_table_offset =
      IREE_HAL_HIP_EXECUTABLE_FORMAT_HEADER_SIZE +
      (iree_host_size_t)module_count *
          IREE_HAL_HIP_EXECUTABLE_FORMAT_MODULE_SIZE;

  const uint32_t export_count = iree_unaligned_load_le_u32(
      data.data + IREE_HAL_HIP_EXECUTABLE_HEADER_EXPORT_COUNT_OFFSET);
  if (IREE_UNLIKELY(export_count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP executable has no exports");
  }
  const iree_host_size_t maximum_export_count =
      (data.data_length - export_table_offset) /
      IREE_HAL_HIP_EXECUTABLE_FORMAT_EXPORT_SIZE;
  if (IREE_UNLIKELY((iree_host_size_t)export_count > maximum_export_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP executable export table is truncated "
                            "(count=%u, available=%" PRIhsz ")",
                            export_count, maximum_export_count);
  }

  iree_hal_hip_executable_format_t format = {
      .data = data,
      .module_count = module_count,
      .export_count = export_count,
      .export_table_offset = export_table_offset,
      .payload_offset =
          export_table_offset + (iree_host_size_t)export_count *
                                    IREE_HAL_HIP_EXECUTABLE_FORMAT_EXPORT_SIZE,
  };
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < format.module_count && iree_status_is_ok(status); ++i) {
    iree_hal_hip_executable_module_t module;
    status = iree_hal_hip_executable_format_read_module(&format, i, &module);
  }
  for (iree_host_size_t i = 0;
       i < format.export_count && iree_status_is_ok(status); ++i) {
    iree_hal_hip_executable_export_t export_def;
    status =
        iree_hal_hip_executable_format_read_export(&format, i, &export_def);
  }
  if (iree_status_is_ok(status)) *out_format = format;
  return status;
}
