// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/metal/executable_format.h"

#include "iree/base/alignment.h"

enum {
  IREE_HAL_METAL_EXECUTABLE_HEADER_MAGIC_OFFSET = 0,
  IREE_HAL_METAL_EXECUTABLE_HEADER_VERSION_OFFSET = 4,
  IREE_HAL_METAL_EXECUTABLE_HEADER_LIBRARY_COUNT_OFFSET = 8,
  IREE_HAL_METAL_EXECUTABLE_HEADER_PIPELINE_COUNT_OFFSET = 12,
};

enum {
  IREE_HAL_METAL_EXECUTABLE_LIBRARY_SOURCE_OFFSET_OFFSET = 0,
  IREE_HAL_METAL_EXECUTABLE_LIBRARY_SOURCE_LENGTH_OFFSET = 4,
  IREE_HAL_METAL_EXECUTABLE_LIBRARY_SOURCE_VERSION_OFFSET = 8,
  IREE_HAL_METAL_EXECUTABLE_LIBRARY_METALLIB_OFFSET_OFFSET = 12,
  IREE_HAL_METAL_EXECUTABLE_LIBRARY_METALLIB_LENGTH_OFFSET = 16,
};

enum {
  IREE_HAL_METAL_EXECUTABLE_PIPELINE_LIBRARY_ORDINAL_OFFSET = 0,
  IREE_HAL_METAL_EXECUTABLE_PIPELINE_ENTRY_POINT_OFFSET_OFFSET = 4,
  IREE_HAL_METAL_EXECUTABLE_PIPELINE_ENTRY_POINT_LENGTH_OFFSET = 8,
  IREE_HAL_METAL_EXECUTABLE_PIPELINE_MAX_THREADS_OFFSET = 12,
  IREE_HAL_METAL_EXECUTABLE_PIPELINE_THREADGROUP_SIZE_X_OFFSET = 16,
  IREE_HAL_METAL_EXECUTABLE_PIPELINE_THREADGROUP_SIZE_Y_OFFSET = 20,
  IREE_HAL_METAL_EXECUTABLE_PIPELINE_THREADGROUP_SIZE_Z_OFFSET = 24,
  IREE_HAL_METAL_EXECUTABLE_PIPELINE_FLAGS_OFFSET = 28,
  IREE_HAL_METAL_EXECUTABLE_PIPELINE_CONSTANT_COUNT_OFFSET = 32,
  IREE_HAL_METAL_EXECUTABLE_PIPELINE_BINDING_COUNT_OFFSET = 36,
  IREE_HAL_METAL_EXECUTABLE_PIPELINE_BINDING_READ_ONLY_BITS_OFFSET = 40,
};

static iree_status_t iree_hal_metal_executable_format_read_payload(
    const iree_hal_metal_executable_format_t* format, const char* record_name,
    iree_host_size_t ordinal, const char* field_name, uint32_t offset,
    uint32_t length, iree_const_byte_span_t* out_value) {
  *out_value = iree_const_byte_span_empty();
  if (IREE_UNLIKELY(length == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Metal executable %s[%" PRIhsz "] has an empty %s",
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
        "Metal executable %s[%" PRIhsz
        "] %s range is outside of the payload (offset=%u, length=%u)",
        record_name, ordinal, field_name, offset, length);
  }
  *out_value = iree_make_const_byte_span(format->data.data + payload_offset,
                                         payload_length);
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_executable_format_read_text(
    const iree_hal_metal_executable_format_t* format, const char* record_name,
    iree_host_size_t ordinal, const char* field_name, uint32_t offset,
    uint32_t length, iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  iree_const_byte_span_t value = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_metal_executable_format_read_payload(
      format, record_name, ordinal, field_name, offset, length, &value));
  if (IREE_UNLIKELY(memchr(value.data, 0, value.data_length) != NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Metal executable %s[%" PRIhsz
                            "] %s contains an embedded NUL byte",
                            record_name, ordinal, field_name);
  }
  *out_value =
      iree_make_string_view((const char*)value.data, value.data_length);
  return iree_ok_status();
}

iree_status_t iree_hal_metal_executable_format_read_library(
    const iree_hal_metal_executable_format_t* format, iree_host_size_t ordinal,
    iree_hal_metal_executable_library_t* out_library) {
  IREE_ASSERT_ARGUMENT(format);
  IREE_ASSERT_ARGUMENT(out_library);
  memset(out_library, 0, sizeof(*out_library));
  if (IREE_UNLIKELY(ordinal >= format->library_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Metal executable library ordinal %" PRIhsz
                            " is out of range for %u libraries",
                            ordinal, format->library_count);
  }

  const iree_host_size_t record_offset =
      IREE_HAL_METAL_EXECUTABLE_FORMAT_HEADER_SIZE +
      ordinal * IREE_HAL_METAL_EXECUTABLE_FORMAT_LIBRARY_SIZE;
  const uint8_t* record = format->data.data + record_offset;

  const uint32_t source_offset = iree_unaligned_load_le_u32(
      record + IREE_HAL_METAL_EXECUTABLE_LIBRARY_SOURCE_OFFSET_OFFSET);
  const uint32_t source_length = iree_unaligned_load_le_u32(
      record + IREE_HAL_METAL_EXECUTABLE_LIBRARY_SOURCE_LENGTH_OFFSET);
  out_library->source_version = iree_unaligned_load_le_u32(
      record + IREE_HAL_METAL_EXECUTABLE_LIBRARY_SOURCE_VERSION_OFFSET);
  const uint32_t metallib_offset = iree_unaligned_load_le_u32(
      record + IREE_HAL_METAL_EXECUTABLE_LIBRARY_METALLIB_OFFSET_OFFSET);
  const uint32_t metallib_length = iree_unaligned_load_le_u32(
      record + IREE_HAL_METAL_EXECUTABLE_LIBRARY_METALLIB_LENGTH_OFFSET);

  if (IREE_UNLIKELY(source_length == 0 && metallib_length == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Metal executable library[%" PRIhsz
                            "] has neither MSL source nor a metallib",
                            ordinal);
  }
  if (source_length == 0) {
    if (IREE_UNLIKELY(source_offset != 0 ||
                      out_library->source_version !=
                          IREE_HAL_METAL_EXECUTABLE_SOURCE_VERSION_DEFAULT)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Metal executable library[%" PRIhsz
                              "] has source metadata without MSL source",
                              ordinal);
    }
  } else {
    IREE_RETURN_IF_ERROR(iree_hal_metal_executable_format_read_text(
        format, "library", ordinal, "MSL source", source_offset, source_length,
        &out_library->source));
  }
  if (metallib_length == 0) {
    if (IREE_UNLIKELY(metallib_offset != 0)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Metal executable library[%" PRIhsz
                              "] has a metallib offset without a metallib",
                              ordinal);
    }
  } else {
    IREE_RETURN_IF_ERROR(iree_hal_metal_executable_format_read_payload(
        format, "library", ordinal, "metallib", metallib_offset,
        metallib_length, &out_library->metallib));
  }
  return iree_ok_status();
}

iree_status_t iree_hal_metal_executable_format_read_pipeline(
    const iree_hal_metal_executable_format_t* format, iree_host_size_t ordinal,
    iree_hal_metal_executable_pipeline_t* out_pipeline) {
  IREE_ASSERT_ARGUMENT(format);
  IREE_ASSERT_ARGUMENT(out_pipeline);
  memset(out_pipeline, 0, sizeof(*out_pipeline));
  if (IREE_UNLIKELY(ordinal >= format->pipeline_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Metal executable pipeline ordinal %" PRIhsz
                            " is out of range for %u pipelines",
                            ordinal, format->pipeline_count);
  }

  const iree_host_size_t record_offset =
      format->pipeline_table_offset +
      ordinal * IREE_HAL_METAL_EXECUTABLE_FORMAT_PIPELINE_SIZE;
  const uint8_t* record = format->data.data + record_offset;

  out_pipeline->library_ordinal = iree_unaligned_load_le_u32(
      record + IREE_HAL_METAL_EXECUTABLE_PIPELINE_LIBRARY_ORDINAL_OFFSET);
  if (IREE_UNLIKELY(out_pipeline->library_ordinal >= format->library_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Metal executable pipeline[%" PRIhsz
                            "] library ordinal %u is out of range for %u "
                            "libraries",
                            ordinal, out_pipeline->library_ordinal,
                            format->library_count);
  }

  const uint32_t entry_point_offset = iree_unaligned_load_le_u32(
      record + IREE_HAL_METAL_EXECUTABLE_PIPELINE_ENTRY_POINT_OFFSET_OFFSET);
  const uint32_t entry_point_length = iree_unaligned_load_le_u32(
      record + IREE_HAL_METAL_EXECUTABLE_PIPELINE_ENTRY_POINT_LENGTH_OFFSET);
  IREE_RETURN_IF_ERROR(iree_hal_metal_executable_format_read_text(
      format, "pipeline", ordinal, "entry-point name", entry_point_offset,
      entry_point_length, &out_pipeline->entry_point));

  out_pipeline->max_threads_per_threadgroup = iree_unaligned_load_le_u32(
      record + IREE_HAL_METAL_EXECUTABLE_PIPELINE_MAX_THREADS_OFFSET);
  out_pipeline->threadgroup_size[0] = iree_unaligned_load_le_u32(
      record + IREE_HAL_METAL_EXECUTABLE_PIPELINE_THREADGROUP_SIZE_X_OFFSET);
  out_pipeline->threadgroup_size[1] = iree_unaligned_load_le_u32(
      record + IREE_HAL_METAL_EXECUTABLE_PIPELINE_THREADGROUP_SIZE_Y_OFFSET);
  out_pipeline->threadgroup_size[2] = iree_unaligned_load_le_u32(
      record + IREE_HAL_METAL_EXECUTABLE_PIPELINE_THREADGROUP_SIZE_Z_OFFSET);
  if (IREE_UNLIKELY(out_pipeline->threadgroup_size[0] == 0 ||
                    out_pipeline->threadgroup_size[1] == 0 ||
                    out_pipeline->threadgroup_size[2] == 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Metal executable pipeline[%" PRIhsz
        "] has an invalid zero threadgroup dimension (%ux%ux%u)",
        ordinal, out_pipeline->threadgroup_size[0],
        out_pipeline->threadgroup_size[1], out_pipeline->threadgroup_size[2]);
  }
  const uint64_t threadgroup_size_xy =
      (uint64_t)out_pipeline->threadgroup_size[0] *
      out_pipeline->threadgroup_size[1];
  if (IREE_UNLIKELY(threadgroup_size_xy >
                    UINT32_MAX / out_pipeline->threadgroup_size[2])) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Metal executable pipeline[%" PRIhsz
                            "] threadgroup size product exceeds UINT32_MAX",
                            ordinal);
  }
  const uint32_t threadgroup_size =
      (uint32_t)(threadgroup_size_xy * out_pipeline->threadgroup_size[2]);
  if (IREE_UNLIKELY(out_pipeline->max_threads_per_threadgroup != 0 &&
                    threadgroup_size >
                        out_pipeline->max_threads_per_threadgroup)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Metal executable pipeline[%" PRIhsz
        "] threadgroup size %u exceeds its maximum of %u threads",
        ordinal, threadgroup_size, out_pipeline->max_threads_per_threadgroup);
  }

  out_pipeline->flags = iree_unaligned_load_le_u32(
      record + IREE_HAL_METAL_EXECUTABLE_PIPELINE_FLAGS_OFFSET);
  const iree_hal_metal_executable_pipeline_flags_t known_flags =
      IREE_HAL_METAL_EXECUTABLE_PIPELINE_FLAG_THREADGROUP_SIZE_ALIGNED;
  if (IREE_UNLIKELY(out_pipeline->flags & ~known_flags)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Metal executable pipeline[%" PRIhsz
                            "] has unknown flags 0x%08X",
                            ordinal, out_pipeline->flags & ~known_flags);
  }

  const uint32_t constant_count = iree_unaligned_load_le_u32(
      record + IREE_HAL_METAL_EXECUTABLE_PIPELINE_CONSTANT_COUNT_OFFSET);
  if (IREE_UNLIKELY(constant_count > IREE_HAL_METAL_MAX_PUSH_CONSTANT_COUNT)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Metal executable pipeline[%" PRIhsz
        "] declares %u constants, exceeding the Metal HAL limit of %u",
        ordinal, constant_count,
        (uint32_t)IREE_HAL_METAL_MAX_PUSH_CONSTANT_COUNT);
  }
  out_pipeline->constant_count = (uint16_t)constant_count;

  const uint32_t binding_count = iree_unaligned_load_le_u32(
      record + IREE_HAL_METAL_EXECUTABLE_PIPELINE_BINDING_COUNT_OFFSET);
  if (IREE_UNLIKELY(binding_count >
                    IREE_HAL_METAL_MAX_DESCRIPTOR_SET_BINDING_COUNT)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Metal executable pipeline[%" PRIhsz
        "] declares %u bindings, exceeding the Metal HAL limit of %u",
        ordinal, binding_count,
        (uint32_t)IREE_HAL_METAL_MAX_DESCRIPTOR_SET_BINDING_COUNT);
  }
  out_pipeline->binding_count = (uint16_t)binding_count;
  out_pipeline->binding_read_only_bits = iree_unaligned_load_le_u64(
      record +
      IREE_HAL_METAL_EXECUTABLE_PIPELINE_BINDING_READ_ONLY_BITS_OFFSET);
  const uint64_t valid_binding_bits =
      binding_count == 0 ? 0 : (UINT64_C(1) << binding_count) - 1;
  if (IREE_UNLIKELY(out_pipeline->binding_read_only_bits &
                    ~valid_binding_bits)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Metal executable pipeline[%" PRIhsz
        "] has read-only bits outside its %u declared bindings",
        ordinal, binding_count);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_metal_executable_format_parse(
    iree_const_byte_span_t data,
    iree_hal_metal_executable_format_t* out_format) {
  IREE_ASSERT_ARGUMENT(out_format);
  memset(out_format, 0, sizeof(*out_format));
  if (IREE_UNLIKELY(data.data_length > 0 && data.data == NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Metal executable data is NULL");
  }
  if (IREE_UNLIKELY(data.data_length <
                    IREE_HAL_METAL_EXECUTABLE_FORMAT_HEADER_SIZE)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Metal executable header is truncated (length=%" PRIhsz ", minimum=%u)",
        data.data_length,
        (uint32_t)IREE_HAL_METAL_EXECUTABLE_FORMAT_HEADER_SIZE);
  }

  const uint32_t magic = iree_unaligned_load_le_u32(
      data.data + IREE_HAL_METAL_EXECUTABLE_HEADER_MAGIC_OFFSET);
  if (IREE_UNLIKELY(magic != IREE_HAL_METAL_EXECUTABLE_FORMAT_MAGIC)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Metal executable has invalid magic");
  }
  const uint32_t version = iree_unaligned_load_le_u32(
      data.data + IREE_HAL_METAL_EXECUTABLE_HEADER_VERSION_OFFSET);
  if (IREE_UNLIKELY(version != IREE_HAL_METAL_EXECUTABLE_FORMAT_VERSION)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Metal executable format version %u is unsupported; expected version "
        "%u",
        version, (uint32_t)IREE_HAL_METAL_EXECUTABLE_FORMAT_VERSION);
  }

  const uint32_t library_count = iree_unaligned_load_le_u32(
      data.data + IREE_HAL_METAL_EXECUTABLE_HEADER_LIBRARY_COUNT_OFFSET);
  if (IREE_UNLIKELY(library_count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Metal executable has no libraries");
  }
  const iree_host_size_t maximum_library_count =
      (data.data_length - IREE_HAL_METAL_EXECUTABLE_FORMAT_HEADER_SIZE) /
      IREE_HAL_METAL_EXECUTABLE_FORMAT_LIBRARY_SIZE;
  if (IREE_UNLIKELY((iree_host_size_t)library_count > maximum_library_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Metal executable library table is truncated "
                            "(count=%u, available=%" PRIhsz ")",
                            library_count, maximum_library_count);
  }
  const iree_host_size_t pipeline_table_offset =
      IREE_HAL_METAL_EXECUTABLE_FORMAT_HEADER_SIZE +
      (iree_host_size_t)library_count *
          IREE_HAL_METAL_EXECUTABLE_FORMAT_LIBRARY_SIZE;

  const uint32_t pipeline_count = iree_unaligned_load_le_u32(
      data.data + IREE_HAL_METAL_EXECUTABLE_HEADER_PIPELINE_COUNT_OFFSET);
  if (IREE_UNLIKELY(pipeline_count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Metal executable has no pipelines");
  }
  const iree_host_size_t maximum_pipeline_count =
      (data.data_length - pipeline_table_offset) /
      IREE_HAL_METAL_EXECUTABLE_FORMAT_PIPELINE_SIZE;
  if (IREE_UNLIKELY((iree_host_size_t)pipeline_count >
                    maximum_pipeline_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Metal executable pipeline table is truncated "
                            "(count=%u, available=%" PRIhsz ")",
                            pipeline_count, maximum_pipeline_count);
  }

  iree_hal_metal_executable_format_t format = {
      .data = data,
      .library_count = library_count,
      .pipeline_count = pipeline_count,
      .pipeline_table_offset = pipeline_table_offset,
      .payload_offset = pipeline_table_offset +
                        (iree_host_size_t)pipeline_count *
                            IREE_HAL_METAL_EXECUTABLE_FORMAT_PIPELINE_SIZE,
  };
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < format.library_count && iree_status_is_ok(status); ++i) {
    iree_hal_metal_executable_library_t library;
    status =
        iree_hal_metal_executable_format_read_library(&format, i, &library);
  }
  for (iree_host_size_t i = 0;
       i < format.pipeline_count && iree_status_is_ok(status); ++i) {
    iree_hal_metal_executable_pipeline_t pipeline;
    status =
        iree_hal_metal_executable_format_read_pipeline(&format, i, &pipeline);
  }
  if (iree_status_is_ok(status)) *out_format = format;
  return status;
}
